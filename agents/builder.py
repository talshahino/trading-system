"""The builder: the manager's approved cards go into the bots by themselves (user, 12.11.2026: "why isn't it decided
end to end?").

Until now an approved card waited for the bot's nightly Claude session to build it (up to a day). Every change that is
exactly a setting is built here at once, on the bots' computer, right after the manager decides:
  bots 1-4, 7   exits / filters cards -> live/bots/<bot>.json "plans" (target, stop, limit entry, market filter - the
                variant the agent tested, run by live/runner.py / live/news_bot.py as is)
  bots 1-4      a scout strategy approved for the bot -> added to the bot's combos at the bot's risk
An undo in the app takes it out again at the next run. Bots 5 and 6's gap-method tracks ("tal_tracks", 13.11.2026) are settings too.
What is code (bot 7 new methods, new rules in the method itself) stays for the bot's nightly agent; information cards have nothing to build.
The bots' computer installs the changed live/bots/*.json about 10 minutes later (the bots restart, open trades stay).
Every build / revert is written in data/agents/replies.json under the card ("built": "auto"), so the nightly agents
don't build it twice.

  python -m agents.builder
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from agents import repo_io as RIO

ROOT = Path(__file__).resolve().parent.parent
BOTS = ROOT / "live" / "bots"
PLAN_BOTS = ("bot1", "bot2", "bot3", "bot4", "bot7")
ADD_BOTS = ("bot1", "bot2", "bot3", "bot4")
SET_BOTS = ("bot5", "bot6")          # virtual bots run as real accounts (12.11.2026): settings files, built at once
ALL_BOTS = PLAN_BOTS + SET_BOTS
TRACK_KNOBS = {"STOP", "DOMINANT", "V5_MAX_COST", "PART_FRAC", "PART_WAY", "TFM", "EDGE", "NEST", "MIN_RR", "GBODY",
               "MIN_GAP", "EXPIRY", "FIRST_ONLY"}


def _read(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _write(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def all_cards():
    out = {}
    for f in (ROOT / "reports" / "agents").glob("bot*/cards*.json"):
        for c in _read(f, {}).get("cards", []):
            out[c.get("id")] = c
    return out


def buildable(c):
    """What a card changes, when it is exactly a setting (else None)."""
    ch, bot = c.get("change") or {}, c.get("bot")
    if c.get("kind") == "improve" and bot in PLAN_BOTS and ch.get("combo") and ch.get("family") in ("exits", "filters", "manage") and ch.get("to"):
        return {"bot": bot, "combo": ch["combo"], "family": ch["family"], "to": ch["to"]}
    if c.get("kind") == "new" and bot in ADD_BOTS and ch.get("combo") and ch.get("kind") == "add":
        return {"bot": bot, "combo": ch["combo"], "family": "add"}
    # bots 5-6: a gap track's rule, a track's risk, the user's gap method's timeframes / rules (bot 6 trades it)
    if bot == "bot5" and c.get("kind") == "improve" and ch.get("track") in ("v5", "v1m") \
            and str(ch.get("knob", "")).split("@")[0] in TRACK_KNOBS and "to" in ch:
        return {"bot": bot, "combo": f"OWN|{ch['track']}", "family": "track", "track": ch["track"],
                "knob": str(ch["knob"]).split("@")[0], "to": ch["to"]}
    if bot == "bot5" and c.get("kind") == "risk" and ch.get("track") == "tal_all" and ch.get("knob") == "ONE_PER_SIDE":
        return {"bot": bot, "combo": "TAL|all", "family": "flag", "key": "tal_one_per_side", "to": bool(ch.get("to"))}
    if bot in SET_BOTS and c.get("kind") == "risk" and ch.get("track") and str(ch.get("knob", "")).startswith("RISK") \
            and ch.get("to") is not None:
        return {"bot": bot, "combo": f"{'OWN' if bot == 'bot5' else 'FUT'}|{ch['track']}", "family": "risk",
                "track": ch["track"], "to": float(ch["to"])}
    # bot 5 (13.11.2026): the gap method's tracks are settings too ("tal_tracks" in live/bots/bot5.json) - a card on a
    # pair that already runs replaces that track's rules, a card on a new pair adds a track at the gap tracks' risk
    if bot == "bot5" and c.get("kind") in ("new", "improve") and ch.get("method") == "tal_gap" and ch.get("pair") \
            and isinstance(ch.get("cfg"), dict):
        hi, lo = ch["pair"].split("/")
        from agents import tal_gap as TG
        if any(p[0] == hi and p[1] == lo for p in TG.PAIRS):
            return {"bot": bot, "combo": f"TAL|{ch['pair']}", "family": "tal5", "pair": [hi, lo], "cfg": ch["cfg"]}
    if bot == "bot6" and c.get("kind") in ("new", "improve") and ch.get("method") == "tal_gap" and ch.get("pair") and ch.get("cfg"):
        hi, lo = ch["pair"].split("/")
        from agents import tal_gap as TG
        pr = next((list(p) for p in TG.PAIRS if p[0] == hi and p[1] == lo), None)
        if pr:   # the main track's pair -> "tal"; another pair -> "tal_tracks" (13.11.2026), decided in apply()
            return {"bot": bot, "combo": f"TAL|{ch['pair']}", "family": "tal6", "pair": pr, "cfg": ch["cfg"]}
    return None


def exit_plan(v):
    tg, _, st, en = v.split("|")
    p = {}
    if tg == "none":
        p["target"] = "none"
    elif tg != "own":
        p["target"] = float(tg.rstrip("R"))
    if float(st[4:]) != 1:
        p["stop"] = float(st[4:])
    if float(en[5:]) != 0:
        p["limit"] = float(en[5:])
    return p


def manage_plan(v):
    """be0.6|part0.3@0.5 / ber2.5|part0 -> the plan keys live/runner.py and live/news_bot.py read."""
    b, pa = v.split("|")
    out = {"be_r": float(b[3:])} if b.startswith("ber") else {"be_way": float(b[2:])}
    if pa != "part0":
        fr, w = pa[4:].split("@")
        out["partial"] = [float(fr), float(w)]
    return out


def describe(b):
    if b["family"] == "track":
        return f"מסלול {b['track']}: {b['knob']} = {b['to']}"
    if b["family"] == "risk":
        return f"מסלול {b['track']}: סיכון {b['to'] * 100:g}% לעסקה"
    if b["family"] == "flag":
        return "שיטת הגאפ: מסלול לא נכנס למכשיר ולכיוון שמסלול אחר כבר מחזיק" if b["to"] else "שיטת הגאפ: בלי הכלל נגד סיכון כפול"
    if b["family"] == "tal5":
        return f"שיטת הגאפ, מסלול {b['pair'][0]} / {b['pair'][1]}: " + ", ".join(f"{k} {v}" for k, v in b["cfg"].items())
    if b["family"] in ("tal", "tal6"):
        return f"שיטת הגאפ: {b['pair'][0]} / {b['pair'][1]}, " + ", ".join(f"{k} {v}" for k, v in b["cfg"].items())
    if b["family"] == "add":
        return f"נוספה לבוט: {b['combo']}, בסיכון של הבוט"
    if b["family"] == "filters":
        return f"{b['combo']}: סינון מצב שוק {b['to']}"
    if b["family"] == "manage":
        from agent_portfolio import explain_manage
        return f"{b['combo']}: {explain_manage(b['to'])}"
    p = exit_plan(b["to"])
    parts = [("בלי יעד" if p.get("target") == "none" else f"יעד {p['target']:g}R") if "target" in p else None,
             f"סטופ x{p['stop']:g}" if "stop" in p else None, f"כניסה בלימיט {p['limit']:g}R טוב יותר" if "limit" in p else None]
    return f"{b['combo']}: " + (", ".join(x for x in parts if x) or "כמו היום")


def apply(cfg, b, card_id):
    if b["family"] == "tal6":
        # bot 6 (13.11.2026): a card on the main track's pair changes "tal"; on a pair another track runs, that track's
        # rules in "tal_tracks"; on a new pair, a new track at the tracks' risk (tal_gap.cards lets a new pair through
        # only when it earned on 2024-2025 and on 2026 - it must add to the bot as a whole)
        main = list((cfg.get("tal") or {}).get("pair") or [])[:2]
        if main == list(b["pair"])[:2] or not main:
            return apply(cfg, dict(b, family="tal"), card_id)
        rk = cfg.setdefault("risk", {})
        tt = cfg.setdefault("tal_tracks", {})
        if not any(list(v.get("pair") or [])[:2] == list(b["pair"])[:2] for v in tt.values()):
            trk = "t" + b["pair"][0].rstrip("m") + "_" + b["pair"][1].rstrip("m")
            others = [float(v) for k, v in rk.items() if k == "tal" or k in tt]
            cfg.setdefault("changes", {})[card_id] = {"family": "tal5", "track": trk, "before": "__none__",
                                                      "risk_before": rk.get(trk, "__none__")}
            tt[trk] = {"pair": list(b["pair"]), "cfg": dict(b["cfg"])}
            rk[trk] = min(others) if others else 0.004     # the tracks' risk now (a new pair is not stronger)
            return
        return apply(cfg, dict(b, family="tal5", pair=list(b["pair"])[:2]), card_id)
    if b["family"] == "flag":
        cfg.setdefault("changes", {})[card_id] = {"family": "flag", "key": b["key"], "before": cfg.get(b["key"], "__none__")}
        cfg[b["key"]] = b["to"]
        return
    if b["family"] == "tal5":
        tt = cfg.setdefault("tal_tracks", {})
        trk = next((k for k, v in tt.items() if list(v.get("pair") or [])[:2] == b["pair"]), None)
        new_trk = trk is None
        if new_trk:
            trk = "t" + b["pair"][0].rstrip("m") + "_" + b["pair"][1].rstrip("m")
        rk = cfg.setdefault("risk", {})
        cfg.setdefault("changes", {})[card_id] = {"family": "tal5", "track": trk,
                                                  "before": "__none__" if new_trk else json.loads(json.dumps(tt[trk])),
                                                  "risk_before": rk.get(trk, "__none__")}
        tt[trk] = {"pair": list(tt[trk]["pair"]) if not new_trk and tt[trk].get("pair") else list(b["pair"]), "cfg": dict(b["cfg"])}
        if new_trk:
            others = [float(v) for k, v in rk.items() if k in tt and k != trk]
            rk[trk] = min(others) if others else 0.004      # the gap tracks' risk now (a new pair is not stronger)
        return
    if b["family"] in ("track", "risk", "tal"):
        # the value before is kept with the card, so an undo / a revert puts it back exactly
        ch = cfg.setdefault("changes", {})
        if b["family"] == "track":
            tr = cfg.setdefault("tracks", {}).setdefault(b["track"], {})
            ch[card_id] = {"family": "track", "track": b["track"], "knob": b["knob"], "before": tr.get(b["knob"], "__none__")}
            tr[b["knob"]] = b["to"]
        elif b["family"] == "risk":
            rk = cfg.setdefault("risk", {})
            ch[card_id] = {"family": "risk", "track": b["track"], "before": rk.get(b["track"], "__none__")}
            rk[b["track"]] = b["to"]
        else:
            tal = cfg.setdefault("tal", {})
            ch[card_id] = {"family": "tal", "before": json.loads(json.dumps(tal))}
            tal.update(pair=b["pair"], cfg=b["cfg"])
        return
    plans = cfg.setdefault("plans", {})
    if b["family"] == "add":
        if b["combo"] not in cfg["combos"]:
            cfg["combos"].append(b["combo"])
        cfg.setdefault("added", {})[b["combo"]] = card_id
        return
    p = plans.setdefault(b["combo"], {})
    if b["family"] == "manage":
        for k in ("be_way", "be_r", "partial"):
            p.pop(k, None)
        p.update(manage_plan(b["to"]))
        p["manage_card"] = card_id
        return
    if b["family"] == "filters":
        if b["to"] == "none":
            p.pop("filter", None)
        else:
            p["filter"] = b["to"]
        p["filter_card"] = card_id
    else:
        for k in ("target", "stop", "limit"):
            p.pop(k, None)
        p.update(exit_plan(b["to"]))
        p["card"] = card_id


def revert(cfg, card_id):
    """Take a card's change out of a bot (undone in the app). True when something changed."""
    done = False
    c = (cfg.get("changes") or {}).pop(card_id, None)
    if c:
        if c["family"] == "track":
            tr = cfg.setdefault("tracks", {}).setdefault(c["track"], {})
            tr.pop(c["knob"], None) if c["before"] == "__none__" else tr.__setitem__(c["knob"], c["before"])
        elif c["family"] == "risk":
            rk = cfg.setdefault("risk", {})
            rk.pop(c["track"], None) if c["before"] == "__none__" else rk.__setitem__(c["track"], c["before"])
        elif c["family"] == "flag":
            cfg.pop(c["key"], None) if c["before"] == "__none__" else cfg.__setitem__(c["key"], c["before"])
        elif c["family"] == "tal5":
            tt, rk = cfg.setdefault("tal_tracks", {}), cfg.setdefault("risk", {})
            tt.pop(c["track"], None) if c["before"] == "__none__" else tt.__setitem__(c["track"], c["before"])
            rk.pop(c["track"], None) if c.get("risk_before") == "__none__" else rk.__setitem__(c["track"], c["risk_before"])
        else:
            cfg["tal"] = c["before"]
        return True
    for combo, cid in list((cfg.get("added") or {}).items()):
        if cid == card_id:
            cfg["combos"] = [c for c in cfg["combos"] if c != combo]
            cfg["added"].pop(combo)
            done = True
    for combo, p in list((cfg.get("plans") or {}).items()):
        if p.get("card") == card_id:
            for k in ("target", "stop", "limit", "card"):
                p.pop(k, None)
            done = True
        if p.get("manage_card") == card_id:
            for k in ("be_way", "be_r", "partial", "manage_card"):
                p.pop(k, None)
            done = True
        if p.get("filter_card") == card_id:
            p.pop("filter", None)
            p.pop("filter_card", None)
            done = True
        if not {k for k in p if k not in ("card", "filter_card", "manage_card")}:
            cfg["plans"].pop(combo)
    return done


def run(now=None):
    at = (now or pd.Timestamp.now(tz="UTC")).isoformat(timespec="seconds")
    dec = _read(ROOT / "reports" / "agents" / "decisions.json", {}).get("decisions", [])
    user = RIO.live_json("live/agent_decisions_state.json")
    undo, ucards = set(user.get("undo") or []), user.get("cards") or {}
    replies = _read(ROOT / "data" / "agents" / "replies.json", {})
    cards = all_cards()
    cfgs = {b: _read(BOTS / f"{b}.json", None) for b in ALL_BOTS}
    touched, did = set(), []
    # the user's own approvals in the app count too (he decided a card himself)
    approved = {d["card"]: d for d in dec if d.get("type") == "card" and d.get("decision") == "approved" and d["id"] not in undo}
    for cid, u in ucards.items():
        if (u or {}).get("decision") == "approved" and cid not in approved:
            approved[cid] = {"card": cid, "id": cid}
    reverted = {d["of"] for d in dec if d.get("decision") == "reverted" and d["id"] not in undo}
    approved = {k: v for k, v in approved.items() if v.get("id") not in reverted}
    undone_cards = {d["card"] for d in dec if d.get("type") == "card" and (d["id"] in undo or d["id"] in reverted)}
    undone_cards |= {cid for cid, u in ucards.items() if (u or {}).get("decision") == "rejected"}
    # 1) undone -> out of the bot
    for cid in undone_cards:
        for b, cfg in cfgs.items():
            if cfg and revert(cfg, cid):
                touched.add(b)
                r = replies.setdefault(cid, {})
                r.update(text="הוחזר לאחור - הוצא מהבוט אוטומטית", at=at, by="builder", reverted=at)
                r.pop("built", None)
                did.append(f"{b}: reverted {cid}")
    # bot 6: one configuration per pair of timeframes of the user's gap method - of the approved cards on the same pair
    # only the strongest goes in (13.11.2026: every pair is its own track, like bot 5; before, one pair for the bot)
    def _score(cid):
        u = (cards.get(cid) or {}).get("unseen") or {}
        return (u.get("mean") or -9) * math.sqrt(max(1.0, u.get("eff_n") or 1))
    tal = [cid for cid in approved if cid in cards and cid not in undone_cards and not (replies.get(cid) or {}).get("built")
           and ((buildable(cards[cid]) or {}).get("family") == "tal6")]
    for cid in tal:
        pr = buildable(cards[cid])["pair"][:2]
        same = [x for x in tal if buildable(cards[x])["pair"][:2] == pr]
        if cid != max(same, key=_score):
            replies[cid] = {"text": "לא נבנה: אושר כרטיס חזק יותר על אותו זוג טיים פריים של שיטת הגאפ, והוא נבנה",
                            "at": at, "by": "builder", "built": "skipped"}
            did.append(f"bot6: {cid} skipped (a stronger card on the same pair)")
    # 2) approved, not built yet -> into the bot
    for cid in approved:
        c = cards.get(cid)
        if not c or cid in undone_cards or (replies.get(cid) or {}).get("built"):
            continue
        b = buildable(c)
        if not b or not cfgs.get(b["bot"]):
            continue
        cfg = cfgs[b["bot"]]
        if b["family"] == "add" and b["combo"].split("|")[0] not in _strategy_ids():
            continue                                     # the strategy's code is not installed yet
        apply(cfg, b, cid)
        touched.add(b["bot"])
        replies[cid] = {"text": f"נבנה אוטומטית ב{_he(b['bot'])}: {describe(b)}. יותקן במחשב הבוטים תוך כ־10 דקות.",
                        "at": at, "by": "builder", "built": "auto"}
        did.append(f"{b['bot']}: built {cid} {describe(b)}")
    for b in touched:
        _write(BOTS / f"{b}.json", cfgs[b])
    if did:
        _write(ROOT / "data" / "agents" / "replies.json", replies)
    for x in did:
        print(x)
    print(f"builder: {len(did)} changes")
    return did


def _he(bot):
    return {"bot7": "בוט 7"}.get(bot, f"בוט {bot[3:]}")


_SIDS = {}


def _strategy_ids():
    if "ids" not in _SIDS:
        from agents import rules_agent as R
        _SIDS["ids"] = set(R.STRATEGIES)
    return _SIDS["ids"]


if __name__ == "__main__":
    run()
