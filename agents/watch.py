"""The watch: every change the manager made is followed trade by trade right after it takes effect, and taken back at
once when it is clearly bad - without waiting for the morning review or the next experiment (user, 12.11.2026).

Runs on the bots' computer in the continuous layer (every 10 minutes), after the manager and the builder.
  watched     every approved decision that changed a bot (not undone, not withdrawn): a risk change, a card built into
              a bot (exits / filter / an added strategy). Removals stop trading, so there is nothing to watch.
  expected    a built card: what its own test showed on 2026 (mean R per trade, spread, success rate); a risk change:
              the combo's backtest with the live rules (reports/trade_review/risk.json, as the health check)
  live        the combo's closed trades in R after costs that OPENED after the change (live states, branch live)
  clearly bad (any of)
              far below    z = (live mean - expected) / (sd / sqrt(n)) <= -2.5 with at least 4 trades
              streak       losses in a row from the start that the expected success rate gives < 1% (at least 4)
              deep         -8R or worse in total since the change (a safety cap)
  -> reverted at once: a "reverted" decision in reports/agents/decisions.json. The builder takes a built card out of
     the bot, the bots' computer puts a risk value back (live/manager.py auto_apply), and the manager does not make the
     same decision again for 30 days. It is in the 05:00 report like every decision, and can be undone there too.
  done        after 20 trades or 30 days without being clearly bad the change leaves the watch.
State and the list the app shows: reports/agents/watch.json.

  python -m agents.watch
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agents import builder as BLD
from agents import health as H
from agents import repo_io as RIO

ROOT = Path(__file__).resolve().parent.parent
DEC = ROOT / "reports" / "agents" / "decisions.json"
OUT = ROOT / "reports" / "agents" / "watch.json"
Z_BAD, MIN_N, STREAK_P, DEEP_R = -2.5, 4, 0.01, -8.0
DONE_N, DONE_DAYS = 20, 30
SD_DEFAULT = 1.2                       # R per trade, when the test gives no spread


def _read(p, d):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return d


def _did(kind, key):
    import hashlib
    return hashlib.sha1(f"{kind}|{key}".encode()).hexdigest()[:12]


def expected_card(c):
    u = c.get("unseen") or {}
    if u.get("mean") is None:
        return None
    sd = abs(u["mean"]) * math.sqrt(u["eff_n"]) / abs(u["t"]) if u.get("t") and u.get("eff_n") else SD_DEFAULT
    return {"mean": u["mean"], "sd": max(0.3, sd), "win": (u.get("win") or 40) / 100, "from": "בדיקת השינוי על 2026"}


def expected_combo(bot, combo, cache={}):  # noqa: B006
    if bot not in cache:
        cache[bot] = H.expectations(bot)
    e = H.expect_for(bot, combo, cache[bot])
    if not e or e.get("mean") is None:
        return None
    return {"mean": e["mean"], "sd": max(0.3, e.get("sd") or SD_DEFAULT), "win": e.get("win") or 0.4, "from": "הבדיקה של הבוט"}


def watched(decisions, undo, replies, cards):
    """[(decision, bot, combo, since, expected)] of the changes that are in force now."""
    reverted = {d.get("of") for d in decisions if d.get("decision") == "reverted" and d["id"] not in undo}
    out = []
    for d in decisions:
        if d.get("decision") != "approved" or d["id"] in undo or d["id"] in reverted or d.get("undone"):
            continue
        if d.get("type") == "risk":
            if d.get("card_decision") or d.get("star_step"):
                continue                                  # the risk of a card's change: followed with the card
            a = d.get("action") or {}
            if not a.get("combo") or a.get("kind") != "mult" or not a.get("value"):
                continue                                  # a removal stops the combo: nothing to watch
            out.append((d, a["bot"], a["combo"], d["at"], expected_combo(a["bot"], a["combo"])))
        elif d.get("type") == "card":
            c = cards.get(d.get("card"))
            r = replies.get(d.get("card")) or {}
            if not r.get("built") or d.get("kind") == "info":
                continue                                  # not in the bot (yet), or not a change at all
            # the card itself may have been replaced since (a new test) - then its combo from the title
            b = BLD.buildable({**c, "bot": d["bot"]}) if c else None
            if b and b["family"] in TRACKS:
                combo = TRACKS[b["family"]](b)               # bots 5-6: all the trades of the changed track
            else:
                combo = ((c or {}).get("change") or {}).get("combo") or (d["title"].split(":")[0].strip() if d["title"].count("|") >= 2 else None)
            if not combo or (combo.count("|") != 2 and not combo.startswith("re:")):
                continue
            e = expected_card(c) if c else None
            out.append((d, d["bot"], combo, r.get("at") or d["at"], e or expected_combo(d["bot"], combo)))
    return out


# bots 5-6: which live trades a settings change touches (agents/health.py names them OWN|15Min|.., OWN|1Min|.., TAL|1Hour|..)
TRACKS = {"track": lambda b: "re:^OWN\\|" + ("1Min" if b["track"] == "v1m" else "15Min") + "\\|",
          "risk": lambda b: ("re:^TAL\\|" if b["track"] == "tal" else "re:^(OWN|FUT)\\|" + ("1Min" if b["track"] == "v1m" else "15Min") + "\\|"),
          "tal": lambda b: "re:^TAL\\|"}


RISK_BOTS = ("bot1", "bot2", "bot3", "bot4")
STAR_N, STAR_LAST = 20, 40       # trades: the first look, and the last


def _mult_now(decisions, did, undo):
    """The multiplier the card's combo trades with now, from the star / whole-bot risk decisions of that card."""
    steps = [x for x in decisions if (x.get("card_decision") == did or x.get("star_of") == did)
             and x.get("decision") == "approved" and x["id"] not in undo and x.get("action")]
    steps.sort(key=lambda x: x["at"])
    return (steps[-1]["action"]["value"], steps[0]["action"].get("prev")) if steps else (None, None)


def star_steps(d, c, combo, r, e, w, decisions, have, at, adds):
    """The "maybe luck" star (user, 12.11.2026): a small edge on few trades goes in at a lowered risk; once enough live
    trades are in, back to the full risk when it keeps winning, lower when it doesn't, out when it still loses."""
    from agents import autodecide as AD
    star = d.get("star") or (c is not None and AD.maybe_luck(c))
    if not star:
        return None
    w["star"] = True
    m, prev = _mult_now(decisions, d["id"], set())
    base = prev if prev is not None else 1.0

    def step(kind, value, title, why, before=None):
        sid = _did(kind, d["id"])
        if sid in have:
            return
        value = max(0.5, value)          # the bots read 0.5-1.5 only (live/runner.py risk_mult)
        have.add(sid)
        adds.append({"id": sid, "bot": d["bot"], "type": "risk", "kind": "mult", "combo": combo, "title": title,
                     "decision": "approved", "why": why, "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "star_of": d["id"],
                     "star_step": kind, "action": {"bot": d["bot"], "combo": combo, "kind": "mult", "value": round(value, 2),
                                                   "prev": m if before is None else before}, "at": at, "by": "manager", "read": False, "undone": False})
    if m is None:          # approved before the star existed: it gets the star's lowered risk now
        from agents import portfolio_check as PC
        cur = float(PC.mults(d["bot"]).get(combo, 1.0) or 1.0)
        step("staron", cur * AD.STAR_MULT, f"⭐ {combo}: סיכון ×{cur * AD.STAR_MULT:.2f} (אולי מזל)",
             "שיפור קטן על מעט עסקאות - נכנס בסיכון מופחת עד שיוכח שאינו מזל", before=None if cur == 1.0 else cur)
        w.pop("done", None)
        return None
    n = len(r)
    mean = float(np.mean(r)) if n else None
    exp_m = (e or {}).get("mean") or 0.0
    w.pop("done", None)                               # a star stays in the watch until its own end
    w["why"] = f"⭐ במעקב מוקפד: {n} עסקאות (בדיקה ב־{STAR_N}, אחרונה ב־{STAR_LAST})"
    if n >= STAR_N and mean is not None and mean >= max(0.0, 0.5 * exp_m):
        if m < base:
            step("staroff", base, f"{combo}: הכוכבית ירדה - חוזר לסיכון ×{base:g}",
                 f"{n} עסקאות מאז השינוי, {mean:+.2f}R לעסקה (צפוי {exp_m:+.2f}R) - זה לא מזל")
        w.update(done=at, why="הכוכבית ירדה: חזר לסיכון הרגיל")
        return None
    if n >= STAR_LAST and mean is not None and mean < 0:
        return f"⭐ אחרי {n} עסקאות עדיין מפסיד ({mean:+.2f}R לעסקה מול {exp_m:+.2f}R צפוי) - זה היה מזל"
    if n >= STAR_N and mean is not None and mean < 0.5 * exp_m and m > 0.5:
        step("stardown", 0.5, f"⭐ {combo}: סיכון יורד ל־×0.5",
             f"{n} עסקאות מאז השינוי, {mean:+.2f}R לעסקה מול {exp_m:+.2f}R צפוי - נשאר בכוכבית ובסיכון נמוך יותר")
    return None


def judge(r, e):
    n = len(r)
    if n == 0 or not e:
        return None, {}
    m = float(np.mean(r))
    z = (m - e["mean"]) / (e["sd"] / math.sqrt(n))
    lead = 0
    for x in r:
        if x > 0:
            break
        lead += 1
    p_streak = (1 - e["win"]) ** lead if lead else 1.0
    info = {"n": n, "mean": round(m, 3), "sum": round(float(np.sum(r)), 2), "z": round(z, 2), "losses_from_start": lead,
            "streak_p": round(p_streak, 4)}
    if n >= MIN_N and z <= Z_BAD:
        return f"מתחת למצופה בבירור: {n} עסקאות, {m:+.2f}R לעסקה מול {e['mean']:+.2f}R ({e['from']})", info
    if lead >= MIN_N and p_streak < STREAK_P:
        return f"{lead} הפסדים ברצף מאז השינוי - לפי הבדיקה זה קורה פחות מ־1% מהפעמים", info
    if float(np.sum(r)) <= DEEP_R:
        return f"{float(np.sum(r)):+.1f}R בסך הכל מאז השינוי", info
    return None, info


def run(now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    at = now.isoformat(timespec="seconds")
    doc = _read(DEC, {"decisions": []})
    decisions = doc.get("decisions", [])
    user = RIO.live_json("live/agent_decisions_state.json")
    undo = set(user.get("undo") or [])
    replies = _read(ROOT / "data" / "agents" / "replies.json", {})
    cards = BLD.all_cards()
    prev = {w["decision"]: w for w in _read(OUT, {}).get("watch", [])}
    states = H.live_states()
    trades = {}
    rows, reverts, adds = [], [], []
    have = {x["id"] for x in decisions}
    for d, bot, combo, since, e in watched(decisions, undo, replies, cards):
        old = prev.get(d["id"]) or {}
        if old.get("done"):
            rows.append(old)
            continue
        if _did("revert", d["id"]) in have:              # you undid the watch's revert: your call, not watched again
            rows.append({**old, "decision": d["id"], "done": at, "why": "החזרה לאחור בוטלה על ידך - נשאר בבוט"})
            continue
        if bot not in trades:
            st = H.state_of(bot, states)
            trades[bot] = H.trades(st, bot) if st else pd.DataFrame(columns=["combo", "time", "r"])
        t = trades[bot]
        if not t.empty:
            hit = t["combo"].str.match(combo[3:]) if combo.startswith("re:") else (t["combo"] == combo)
            t = t[hit & (t["time"] >= pd.Timestamp(since))]
        r = t["r"].to_numpy(float) if not t.empty else np.array([])
        bad, info = judge(r, e)
        age = (now - pd.Timestamp(since)).days
        w = {"decision": d["id"], "bot": bot, "combo": combo, "title": d["title"], "since": since,
             "expected": e, "live": info, "checked": at}
        if bad:
            reverts.append((d, bad, info))
            w.update(reverted=at, why=bad)
        elif len(r) >= DONE_N or age >= DONE_DAYS:
            w.update(done=at, why=f"עבר את המעקב: {len(r)} עסקאות ב־{age} ימים בלי סימן רע")
        if not bad and d.get("type") == "card" and bot in RISK_BOTS and "|" in combo and not combo.startswith("re:"):
            luck = star_steps(d, cards.get(d.get("card")), combo, r, e, w, decisions, have, at, adds)
            if luck:
                reverts.append((d, luck, info))
                w.update(reverted=at, why=luck)
        rows.append(w)
    for x in adds:
        decisions.insert(0, x)
        doc["new"] = list(doc.get("new") or []) + [x["id"]]
        print(f"star: {x['title']}")
    for d, bad, info in reverts:
        decisions.insert(0, {"id": _did("revert", d["id"]), "bot": d["bot"], "type": "revert", "of": d["id"],
                             "kind": d.get("kind"), "title": "המנהל החזיר לאחור: " + d["title"], "decision": "reverted",
                             "why": bad, "effect": ("הערך הקודם חוזר במחשב הבוטים תוך כ־10 דקות" if d.get("type") == "risk"
                                                    else "השינוי יוצא מהבוט אוטומטית ויותקן תוך כ־20 דקות"),
                             "action": d.get("action"), "card": d.get("card"), "live": info,
                             "at": at, "by": "manager", "read": False, "undone": False})
        doc["new"] = list(doc.get("new") or []) + [_did("revert", d["id"])]
        print(f"reverted {d['id']} {d['bot']}: {d['title']} - {bad}")
    if reverts or adds:
        doc["decisions"] = decisions
        DEC.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    OUT.write_text(json.dumps({"updated": at, "rules": {"z": Z_BAD, "min_n": MIN_N, "streak_p": STREAK_P, "deep_r": DEEP_R,
                                                         "done_n": DONE_N, "done_days": DONE_DAYS},
                               "watch": rows}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    live = [w for w in rows if not w.get("done") and not w.get("reverted")]
    print(f"watch: {len(live)} changes followed, {len(reverts)} reverted now")
    return reverts


if __name__ == "__main__":
    run()
