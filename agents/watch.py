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
            combo = ((c or {}).get("change") or {}).get("combo") or (d["title"].split(":")[0].strip() if d["title"].count("|") >= 2 else None)
            if not combo or combo.count("|") != 2:
                continue
            e = expected_card(c) if c else None
            out.append((d, d["bot"], combo, r.get("at") or d["at"], e or expected_combo(d["bot"], combo)))
    return out


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
    rows, reverts = [], []
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
        t = t[(t["combo"] == combo) & (t["time"] >= pd.Timestamp(since))] if not t.empty else t
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
        rows.append(w)
    for d, bad, info in reverts:
        decisions.insert(0, {"id": _did("revert", d["id"]), "bot": d["bot"], "type": "revert", "of": d["id"],
                             "kind": d.get("kind"), "title": "המנהל החזיר לאחור: " + d["title"], "decision": "reverted",
                             "why": bad, "effect": ("הערך הקודם חוזר במחשב הבוטים תוך כ־10 דקות" if d.get("type") == "risk"
                                                    else "השינוי יוצא מהבוט אוטומטית ויותקן תוך כ־20 דקות"),
                             "action": d.get("action"), "card": d.get("card"), "live": info,
                             "at": at, "by": "manager", "read": False, "undone": False})
        doc["new"] = list(doc.get("new") or []) + [_did("revert", d["id"])]
        print(f"reverted {d['id']} {d['bot']}: {d['title']} - {bad}")
    if reverts:
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
