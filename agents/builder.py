"""The builder: the manager's approved cards go into the bots by themselves (user, 12.11.2026: "why isn't it decided
end to end?").

Until now an approved card waited for the bot's nightly Claude session to build it (up to a day). Every change that is
exactly a setting is built here at once, on the bots' computer, right after the manager decides:
  bots 1-4, 7   exits / filters cards -> live/bots/<bot>.json "plans" (target, stop, limit entry, market filter - the
                variant the agent tested, run by live/runner.py / live/news_bot.py as is)
  bots 1-4      a scout strategy approved for the bot -> added to the bot's combos at the bot's risk
An undo in the app takes it out again at the next run. Bots 5 and 6's gap-method tracks ("tal_tracks", 13.11.2026) are settings too.
Bot 6's scout strategies that passed bot 6's own test (agent_bot6.scout_world) go into "scout" in live/bots/bot6.json (13.11.2026).
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
    # the pyramid (14.11.2026, task tal1411anti): live/runner.py pyramid_add (bots 1-4), live/news_bot.py (bot 7)
    if c.get("kind") == "improve" and bot in PLAN_BOTS and ch.get("combo") and ch.get("family") == "pyr" \
            and pyr_plan(ch.get("to")) is not None:
        return {"bot": bot, "combo": ch["combo"], "family": "pyr", "to": ch["to"]}
    if c.get("kind") == "new" and bot in ADD_BOTS and ch.get("combo") and ch.get("kind") == "add":
        return {"bot": bot, "combo": ch["combo"], "family": "add"}
    # bots 5-6: a gap track's rule, a track's risk, the user's gap method's timeframes / rules (bot 6 trades it)
    if bot == "bot5" and c.get("kind") == "improve" and ch.get("track") in ("v5", "v1m") \
            and str(ch.get("knob", "")).split("@")[0] in TRACK_KNOBS and "to" in ch:
        return {"bot": bot, "combo": f"OWN|{ch['track']}", "family": "track", "track": ch["track"],
                "knob": str(ch["knob"]).split("@")[0], "to": ch["to"]}
    if bot in SET_BOTS and c.get("kind") == "risk" and ch.get("track") == "tal_all" and ch.get("knob") == "ONE_PER_SIDE":
        return {"bot": bot, "combo": "TAL|all", "family": "flag", "key": "tal_one_per_side", "to": bool(ch.get("to"))}
    if bot in SET_BOTS and c.get("kind") == "risk" and ch.get("track") == "tal_all" and ch.get("knob") == "MAX_DIR" \
            and isinstance(ch.get("to"), int) and 0 <= ch["to"] <= 10:
        return {"bot": bot, "combo": "TAL|all", "family": "flag", "key": "tal_max_dir", "to": int(ch["to"])}
    # the hour filter (13.11.2026): the UTC hours where the gap tracks enter no new trade (bots 5 and 6 - the shared
    # methods: bot 6's smart agent, 13.11.2026 night; bot 6 reads the same keys in live/bots/bot6.json)
    if bot in SET_BOTS and c.get("kind") == "risk" and ch.get("track") == "tal_all" and ch.get("knob") == "HOURS_OFF" \
            and isinstance(ch.get("to"), list) and all(isinstance(x, int) and 0 <= x <= 23 for x in ch["to"]) and len(ch["to"]) <= 12:
        return {"bot": bot, "combo": "TAL|all", "family": "flag", "key": "tal_hours_off", "to": sorted(set(ch["to"]))}
    # bot 6 (14.11.2026 night): one micro contract for a gap trade whose single contract is more than the track's risk,
    # up to a share of the balance (at most the bot's own 0.4%) - "tal_one_micro_up_to" in live/bots/bot6.json
    if bot == "bot6" and c.get("kind") == "risk" and ch.get("track") == "tal_all" and ch.get("knob") == "ONE_MICRO" \
            and isinstance(ch.get("to"), (int, float)) and 0 <= float(ch["to"]) <= 0.004:
        return {"bot": bot, "combo": "TAL|all", "family": "flag", "key": "tal_one_micro_up_to", "to": round(float(ch["to"]), 5)}
    # one factor on every gap track's risk (13.11.2026 night, tal_gap.scale_card: chosen on 2024-2025 so the account
    # stays two thirds inside the bot's limits) - "tal_risk_scale" in live/bots/bot5.json, read by live/watch_own.py
    if bot == "bot5" and c.get("kind") == "risk" and ch.get("track") == "tal_all" and ch.get("knob") == "RISK_SCALE" \
            and isinstance(ch.get("to"), (int, float)) and 0.1 <= float(ch["to"]) <= 1.0:
        return {"bot": bot, "combo": "TAL|all", "family": "flag", "key": "tal_risk_scale", "to": round(float(ch["to"]), 3)}
    # bots 1-4 (13.11.2026 night, bots 1-4 smart agent): the whole bot's risk sized into the room left to its 9% limit
    # by the worst fall of its own 2026 replay - "dd_fit" in live/bots/<bot>.json, read by live/runner.py allowed_risk
    if bot in ADD_BOTS and c.get("kind") == "risk" and ch.get("kind") == "dd_fit" \
            and isinstance(ch.get("to"), (int, float)) and 0.5 <= float(ch["to"]) <= 9:
        return {"bot": bot, "combo": "BOT|all", "family": "flag", "key": "dd_fit", "to": round(float(ch["to"]), 2)}
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
    # bot 5 (13.11.2026, agents/tal_wide.py): the gap method on one more instrument (a forex pair, a metal, another
    # index) - its own track, only for that instrument
    if bot == "bot5" and c.get("kind") in ("new", "improve") and ch.get("method") == "tal_gap_sym" and ch.get("pair") \
            and ch.get("sym") and isinstance(ch.get("cfg"), dict):
        hi, lo = ch["pair"].split("/")
        from agents import tal_gap as TG
        from agents import ftmo_universe as U
        if any(p[0] == hi and p[1] == lo for p in TG.PAIRS) and ch["sym"] in U.ALL:
            return {"bot": bot, "combo": f"TAL|{ch['pair']}|{ch['sym']}", "family": "tal5sym", "pair": [hi, lo],
                    "sym": ch["sym"], "cfg": ch["cfg"]}
    if bot == "bot6" and c.get("kind") in ("new", "improve") and ch.get("method") == "tal_gap" and ch.get("pair") and ch.get("cfg"):
        hi, lo = ch["pair"].split("/")
        from agents import tal_gap as TG
        pr = next((list(p) for p in TG.PAIRS if p[0] == hi and p[1] == lo), None)
        if pr:   # the main track's pair -> "tal"; another pair -> "tal_tracks" (13.11.2026), decided in apply()
            return {"bot": bot, "combo": f"TAL|{ch['pair']}", "family": "tal6", "pair": pr, "cfg": ch["cfg"]}
    # bot 6 (13.11.2026): a scout strategy that passed bot 6's own test (agent_bot6.scout_world: micro-futures costs,
    # FTMO Futures sessions, chosen on 2024-2025) -> "scout" in live/bots/bot6.json, traded by live/watch_own.py.
    # Never from the CFD test's card (change {"kind": "add"}): that one is tested on bot 6's world first.
    # bot 5 (14.11.2026, bot 5's smart agent): a scout strategy approved on agent_portfolio.py's test (the CFD bots'
    # machine with the live rules, bot 5's instruments, 1-15 minute charts) -> "scout" in live/bots/bot5.json, traded by
    # live/watch_own.py scout5_trades with the same functions. Only a strategy whose code is installed.
    if bot == "bot5" and c.get("kind") in ("new", "improve") and ch.get("kind") == "add" \
            and str(ch.get("combo", "")).count("|") == 2:
        sid, tf, sym = ch["combo"].split("|")
        from agents import ftmo_universe as U
        if tf in ("1Min", "5Min", "15Min", "1Hour") and sym in U.ALL and sid in _strategy_ids():
            return {"bot": bot, "combo": ch["combo"], "family": "scout5"}
    if bot == "bot6" and c.get("kind") in ("new", "improve") and ch.get("family") == "scout_fut" \
            and str(ch.get("combo", "")).count("|") == 2 and isinstance(ch.get("v"), dict):
        return {"bot": bot, "combo": ch["combo"], "family": "scout6", "v": ch["v"]}
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


def pyr_plan(v):
    """pyr1.5@0.5 -> [1.5, 0.5] (at R in profit, fraction of a position added); pyr0 -> []; anything else None."""
    if v == "pyr0":
        return []
    try:
        at, fr = str(v)[3:].split("@")
        at, fr = float(at), float(fr)
    except (TypeError, ValueError):
        return None
    return [at, fr] if str(v).startswith("pyr") and 0 < at <= 5 and 0 < fr <= 1 else None


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
        if b["key"] == "tal_max_dir":
            return f"שיטת הגאפ: לכל היותר {b['to']} עסקאות פתוחות באותו כיוון בכל המדדים יחד" if b["to"] else "שיטת הגאפ: בלי תקרה לכיוון"
        if b["key"] == "tal_risk_scale":
            return f"שיטת הגאפ: הסיכון של כל המסלולים × {b['to']:g}"
        if b["key"] == "dd_fit":
            return (f"הסיכון של כל הבוט קטן כשנשאר עד המגבלה (9%) פחות מ־{b['to']:g}% (הירידה הגרועה בבדיקת הבוט על 2026), "
                    "וחוזר לרגיל לבד")
        if b["key"] == "tal_one_micro_up_to":
            return (f"שיטת הגאפ: עסקה שחוזה מיקרו אחד שלה גדול מהסיכון של המסלול נלקחת בחוזה אחד, אם הוא מסכן עד {b['to'] * 100:g}% "
                    "מהחשבון ויש מקום") if b["to"] else "שיטת הגאפ: בלי חוזה אחד מעל הסיכון של המסלול"
        if b["key"] == "tal_hours_off":
            return ("שיטת הגאפ: בלי כניסות בשעות " + ", ".join(f"{x:02d}:00" for x in b["to"]) + " (UTC)") if b["to"] else "שיטת הגאפ: בלי מסנן שעות"
        return "שיטת הגאפ: מסלול לא נכנס למכשיר ולכיוון שמסלול אחר כבר מחזיק" if b["to"] else "שיטת הגאפ: בלי הכלל נגד סיכון כפול"
    if b["family"] == "tal5sym":
        return f"שיטת הגאפ על {b['sym']}, מסלול {b['pair'][0]} / {b['pair'][1]}: " + ", ".join(f"{k} {v}" for k, v in b["cfg"].items())
    if b["family"] == "tal5":
        return f"שיטת הגאפ, מסלול {b['pair'][0]} / {b['pair'][1]}: " + ", ".join(f"{k} {v}" for k, v in b["cfg"].items())
    if b["family"] in ("tal", "tal6"):
        return f"שיטת הגאפ: {b['pair'][0]} / {b['pair'][1]}, " + ", ".join(f"{k} {v}" for k, v in b["cfg"].items())
    if b["family"] == "add":
        return f"נוספה לבוט: {b['combo']}, בסיכון של הבוט"
    if b["family"] == "scout5":
        return f"נוספה לבוט 5: {b['combo']}, בסיכון של הבוט" + (" ×0.7 (⭐ אולי מזל)" if b.get("star") else "")
    if b["family"] == "scout6":
        from agents import scout_fut as SF
        return f"נוספה לבוט 6: {b['combo']} ({SF.he(b['v'])}), בסיכון של הבוט"
    if b["family"] == "filters":
        return f"{b['combo']}: סינון מצב שוק {b['to']}"
    if b["family"] == "manage":
        from agent_portfolio import explain_manage
        return f"{b['combo']}: {explain_manage(b['to'])}"
    if b["family"] == "pyr":
        v = pyr_plan(b["to"])
        return f"{b['combo']}: " + (f"ב־{v[0]:g}R רווח מוסיפים {v[1]:g} פוזיציה, והסטופ של הכול זז כך שסטופ מפסיד 1R" if v else "בלי הוספה")
    p = exit_plan(b["to"])
    parts = [("בלי יעד" if p.get("target") == "none" else f"יעד {p['target']:g}R") if "target" in p else None,
             f"סטופ x{p['stop']:g}" if "stop" in p else None, f"כניסה בלימיט {p['limit']:g}R טוב יותר" if "limit" in p else None]
    return f"{b['combo']}: " + (", ".join(x for x in parts if x) or "כמו היום")


def apply(cfg, b, card_id):
    if b["family"] == "scout5":
        sc = cfg.setdefault("scout", {})
        cfg.setdefault("changes", {})[card_id] = {"family": "scout5", "combo": b["combo"],
                                                  "before": json.loads(json.dumps(sc.get(b["combo"], "__none__")))}
        sc[b["combo"]] = {"since": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "card": card_id}
        cfg.setdefault("risk", {}).setdefault("scout", 0.004)     # the bot's risk (principle 8)
        if b.get("star"):
            star6(cfg, b["combo"], card_id)                        # "scout:<combo>" = x0.7, as in bot 6
        return
    if b["family"] == "scout6":
        sc = cfg.setdefault("scout", {})
        cfg.setdefault("changes", {})[card_id] = {"family": "scout6", "combo": b["combo"],
                                                  "before": json.loads(json.dumps(sc.get(b["combo"], "__none__")))}
        sc[b["combo"]] = {"v": dict(b["v"]), "since": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "card": card_id}
        cfg.setdefault("risk", {}).setdefault("scout", 0.004)     # the bot's risk (principle 8)
        if b.get("star"):
            star6(cfg, b["combo"], card_id)
        return
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
            if b.get("star"):
                # the manager's star ("maybe luck", principle 9) is a risk x0.7 (bot 6's smart agent, 13.11.2026 late
                # night: the starred 5m/1m track, card 4e4b0542d487, was built at the full 0.4% - its -1R on 13.11
                # 20:40 cost 0.4% of the account instead of 0.28%)
                rk[trk] = round(float(rk[trk]) * STAR6, 6)
            return
        return apply(cfg, dict(b, family="tal5", pair=list(b["pair"])[:2]), card_id)
    if b["family"] == "flag":
        cfg.setdefault("changes", {})[card_id] = {"family": "flag", "key": b["key"], "before": cfg.get(b["key"], "__none__")}
        cfg[b["key"]] = b["to"]
        return
    if b["family"] == "tal5sym":
        tt, rk = cfg.setdefault("tal_tracks", {}), cfg.setdefault("risk", {})
        trk = "t" + b["pair"][0].rstrip("m") + "_" + b["pair"][1].rstrip("m") + "@" + b["sym"]
        new_trk = trk not in tt
        cfg.setdefault("changes", {})[card_id] = {"family": "tal5", "track": trk,
                                                  "before": "__none__" if new_trk else json.loads(json.dumps(tt[trk])),
                                                  "risk_before": rk.get(trk, "__none__")}
        tt[trk] = {"pair": list(b["pair"]), "cfg": dict(b["cfg"]), "symbols": [b["sym"]]}
        if new_trk:
            # the gap tracks' usual risk now (the most common one: a track cut to half is not the bot's risk -
            # principle 8, a new strategy enters at the bot's risk)
            others = [round(float(v), 6) for k, v in rk.items() if k in tt and k != trk and float(v or 0) > 0]
            rk[trk] = max(set(others), key=lambda x: (others.count(x), x)) if others else 0.0028
            if b.get("star"):
                # the manager's star ("maybe luck", principle 9) is a risk x0.7, as in bots 1-4 and 6 (13.11.2026
                # evening, bot 5's smart agent: five starred t5_1@... tracks were built at the full 0.28%)
                rk[trk] = round(float(rk[trk]) * 0.7, 6)
        return
    if b["family"] == "tal5":
        tt = cfg.setdefault("tal_tracks", {})
        trk = next((k for k, v in tt.items() if list(v.get("pair") or [])[:2] == b["pair"] and not v.get("symbols")), None)
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
    if b["family"] == "pyr":
        v = pyr_plan(b["to"])
        p.pop("pyramid", None)
        if v:
            p["pyramid"] = v
        p["pyr_card"] = card_id
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


STAR6 = 0.7          # agents/autodecide.STAR_MULT: the manager's "maybe luck" discount


def star6(cfg, combo, card_id):
    """Bot 6 (13.11.2026 night): the manager's star on a scout strategy is a lowered risk of its own - "scout:<combo>"
    in "risk" (live/watch_own.py) - as the star is a x0.7 risk in bots 1-4. Before this, bot 6 took starred strategies
    at the full 0.4% (SC005 on US100 / US500 and SC020, 13.11.2026). Taken out with the card."""
    rk = cfg.setdefault("risk", {})
    key = f"scout:{combo}"
    ch = cfg.setdefault("changes", {}).setdefault(card_id, {"family": "scout6", "combo": combo, "before": "__none__"})
    if ch.get("star_risk_before") is not None or key in rk:
        return False
    ch["star_risk_before"] = rk.get(key, "__none__")
    rk[key] = round(float(rk.get("scout", 0.004)) * STAR6, 6)
    return True


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
        elif c["family"] in ("scout6", "scout5"):
            sc = cfg.setdefault("scout", {})
            sc.pop(c["combo"], None) if c["before"] == "__none__" else sc.__setitem__(c["combo"], c["before"])
            if c.get("star_risk_before") is not None:
                rk = cfg.setdefault("risk", {})
                k_ = f"scout:{c['combo']}"
                rk.pop(k_, None) if c["star_risk_before"] == "__none__" else rk.__setitem__(k_, c["star_risk_before"])
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
        if p.get("pyr_card") == card_id:
            p.pop("pyramid", None)
            p.pop("pyr_card", None)
            done = True
        if not {k for k in p if k not in ("card", "filter_card", "manage_card", "pyr_card")}:
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
           and ((buildable(cards[cid]) or {}).get("family") in ("tal6", "tal5"))]
    # bot 5 too (13.11.2026): on 12.11 two approved cards on 15m/1m were built one over the other, the later one won
    for cid in tal:
        bb = buildable(cards[cid])
        pr = (bb["bot"], bb["pair"][:2])
        same = [x for x in tal if (buildable(cards[x])["bot"], buildable(cards[x])["pair"][:2]) == pr]
        if cid != max(same, key=_score):
            replies[cid] = {"text": "לא נבנה: אושר כרטיס חזק יותר על אותו זוג טיים פריים של שיטת הגאפ, והוא נבנה",
                            "at": at, "by": "builder", "built": "skipped"}
            did.append(f"{bb['bot']}: {cid} skipped (a stronger card on the same pair)")
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
        if b["family"] in ("scout6", "scout5", "tal5sym", "tal6") and (approved.get(cid) or {}).get("star"):
            b = dict(b, star=True)
        apply(cfg, b, cid)
        touched.add(b["bot"])
        replies[cid] = {"text": f"נבנה אוטומטית ב{_he(b['bot'])}: {describe(b)}. יותקן במחשב הבוטים תוך כ־10 דקות.",
                        "at": at, "by": "builder", "built": "auto"}
        did.append(f"{b['bot']}: built {cid} {describe(b)}")
    # bot 6: a starred scout strategy built before the star had a risk of its own there (13.11.2026) gets it now, once
    cfg6 = cfgs.get("bot6")
    for cid, d in approved.items():
        if not cfg6 or not d.get("star") or cid in undone_cards or d.get("bot") != "bot6":
            continue
        b = buildable(cards.get(cid) or {}) if cid in cards else None
        if b and b["family"] == "scout6" and ((cfg6.get("scout") or {}).get(b["combo"]) or {}).get("card") == cid \
                and star6(cfg6, b["combo"], cid):
            touched.add("bot6")
            r = replies.setdefault(cid, {})
            r["star"] = "⭐ סיכון " + f"{cfg6['risk']['scout:' + b['combo']] * 100:g}" + "% (אולי מזל) - " + at
            did.append(f"bot6: star {cid} {b['combo']} -> {cfg6['risk'][f'scout:' + b['combo']]}")
    # bot 6: a starred new track of the gap method built before the star counted (13.11.2026: t5_1 at 0.4%) gets x0.7
    # now, once - only a track this card added (before "__none__") and still at the risk it was built with
    for cid, d in approved.items():
        if not cfg6 or not d.get("star") or cid in undone_cards or d.get("bot") != "bot6":
            continue
        # the card itself may be gone from cards.json (each run rewrites it) - the bot's own record of the change counts
        r = replies.get(cid) or {}
        if r.get("built") != "auto" or r.get("star"):
            continue
        ch6 = (cfg6.get("changes") or {}).get(cid) or {}
        rk6 = cfg6.get("risk") or {}
        trk = ch6.get("track")
        if ch6.get("family") != "tal5" or ch6.get("before") != "__none__" or not trk or trk not in rk6 \
                or trk not in (cfg6.get("tal_tracks") or {}):
            continue
        rk6[trk] = round(float(rk6[trk]) * STAR6, 6)
        touched.add("bot6")
        replies[cid] = dict(r, star="⭐ סיכון " + f"{rk6[trk] * 100:g}" + "% (אולי מזל) - " + at)
        did.append(f"bot6: star {cid} {trk} -> {rk6[trk]}")
    # bot 5: a starred instrument track of the gap method built before the star counted (13.11.2026) gets x0.7 now, once
    cfg5 = cfgs.get("bot5")
    for cid, d in approved.items():
        if not cfg5 or not d.get("star") or cid in undone_cards or d.get("bot") != "bot5" or cid not in cards:
            continue
        b = buildable(cards[cid])
        r = replies.get(cid) or {}
        if not b or b["family"] != "tal5sym" or r.get("built") != "auto" or r.get("star"):
            continue
        trk = "t" + b["pair"][0].rstrip("m") + "_" + b["pair"][1].rstrip("m") + "@" + b["sym"]
        ch5 = (cfg5.get("changes") or {}).get(cid) or {}
        rk5 = cfg5.get("risk") or {}
        if ch5.get("track") != trk or ch5.get("before") != "__none__" or trk not in rk5:
            continue
        rk5[trk] = round(float(rk5[trk]) * 0.7, 6)
        touched.add("bot5")
        replies[cid] = dict(r, star="⭐ סיכון " + f"{rk5[trk] * 100:g}" + "% (אולי מזל) - " + at)
        did.append(f"bot5: star {cid} {trk} -> {rk5[trk]}")
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
