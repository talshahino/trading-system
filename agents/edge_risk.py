"""Every strategy's risk follows its own edge - up and down, once a day (user, 13.11.2026).

    risk = approved risk x (edge now / approved edge)          (agents/loss_review.edge_now: the test of 2026 and the
                                                               live trades together, each by how much it can be trusted)
- approved risk: the strategy's multiplier as the user, the manager or an agent's card set it (not by this formula).
  When someone else changes it, that becomes the new approved risk.
- down: as the edge shrinks; no edge left -> the instrument's minimum volume (until a fix is found - the bot's agent
  gets the strategy first in its test queue).
- up: as the live trades prove more than the test - slowly, because a few good trades weigh little against a stable
  test (no "maybe luck" raise). The ceiling is not a number: the whole bot is replayed on 2026 with the new risk, and
  the raise goes only as far as the bot stays inside its own limits (4.5% a day, 9% in all - live/runner.py).
- a bot with too many strategies not yet tested as they trade now is not raised (the replay would not see them).
Decisions go to reports/agents/decisions.json (the bots' computer applies them within minutes); the state in
data/agents/edge_risk.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "agents" / "edge_risk.json"
DEC = ROOT / "reports" / "agents" / "decisions.json"
BOTS = ("bot1", "bot2", "bot3", "bot4")
TZ = "Asia/Jerusalem"
DAY_LIMIT, TOTAL_LIMIT = 4.5, 9.0          # the bots' own loss limits, % (live/runner.py GUARD_DAILY / GUARD_TOTAL)


def _read(p, d):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return d


def _write(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def inside(r):
    return r["worst_day"] > -DAY_LIMIT and r["max_dd"] < TOTAL_LIMIT


def ceiling(bot, book, combo, lo, hi, risk):
    """The highest multiplier in [lo, hi] at which the whole bot replayed on 2026 stays inside its limits."""
    from agents import portfolio_check as PC
    tr = book[combo][0]
    if inside(PC.replay({**book, combo: (tr, hi)}, risk)):
        return hi
    if not inside(PC.replay({**book, combo: (tr, lo)}, risk)):
        return lo
    for _ in range(12):
        mid = (lo + hi) / 2
        if inside(PC.replay({**book, combo: (tr, mid)}, risk)):
            lo = mid
        else:
            hi = mid
    return lo


def fixes_failed(bot, combo):
    """"First fix, and only if there is none, lower the risk" (the user): a cut needs the fix attempts to have failed -
    recorded by the bot's agent, or every test family run on it without an approvable fix."""
    from agents import autodecide as AD
    fx = _read(ROOT / "data" / "agents" / "fix_attempts.json", {})
    x = fx.get(f"{bot}|{combo}") if isinstance(fx, dict) else None
    return (bool(x) and x.get("status") == "failed") or AD.tried_and_failed(bot, combo)


def run(states=None, force=False):
    from agents import health as H
    from agents import loss_review as LR
    from agents import portfolio_check as PC
    st = _read(STATE, {})
    today = str(pd.Timestamp.now(tz=TZ).date())
    if st.get("day") == today and not force:
        return []
    states = states if states is not None else H.live_states()
    ddoc = _read(DEC, {"decisions": []})
    have = {x["id"] for x in ddoc.get("decisions", [])}
    at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    adds, rows = [], {}
    for bot in BOTS:
        cfg = PC.bot_cfg(bot)
        if not cfg:
            continue
        mm = PC.mults(bot)
        book, missing = PC.book_now(bot, cfg)
        raise_ok = len(missing) <= max(2, 0.2 * len(cfg.get("combos") or []))
        risk = PC.bot_risk(bot, cfg)
        lt = LR.trade_rows(H.state_of(bot, states) or {}) if states else pd.DataFrame()
        mine = st.setdefault("bots", {}).setdefault(bot, {})
        out = rows.setdefault(bot, {"raise_ok": raise_ok, "missing": len(missing), "combos": {}})
        for c in cfg.get("combos") or []:
            cur = float(mm.get(c, 1.0) or 1.0)
            me = mine.get(c) or {}
            # someone else set the risk since our last decision -> that is the approved risk now
            base = me.get("base", cur) if me.get("last") is not None and abs(float(me["last"]) - cur) < 0.005 else cur
            if cur == 0:                       # removed: not ours
                continue
            lr = [float(x) for x in lt[lt["combo"] == c]["r"]] if len(lt) else []
            mult, parts, m0, m = LR.edge_now(bot, c, lr, up=True)
            if mult is None:
                continue
            target = LR.MIN_LOT if mult == LR.MIN_LOT else round(base * mult, 2)
            # the approved risk the ratio multiplies, in the card's words (manager, 14.11.2026: TV58 USDCAD read
            # "×0.69" in the why and "×0.48" in the title - the ×0.7 approved before was not shown)
            why_cap = (f"; כפול הסיכון המאושר ×{base:g} = ×{target:g}"
                       if mult != LR.MIN_LOT and abs(base - 1.0) >= 0.005 else "")
            if target > cur:
                if not raise_ok or c not in book:
                    out["combos"][c] = {"base": base, "now": cur, "target": target, "held": "עוד אין בדיקה לכל הבוט - לא מעלים"}
                    continue
                cap = round(ceiling(bot, {**book, c: (book[c][0], cur)}, c, cur, target, risk), 2)
                if cap < target:
                    why_cap += f"; הבוט השלם על 2026 נשאר במגבלות שלו רק עד ×{cap}"
                    target = cap
            if target < cur and not fixes_failed(bot, c):
                # not yet: first a fix - the strategy goes first in its agent's test queue
                out["combos"][c] = {"base": base, "now": cur, "target": target, "edge0": m0, "edge": m, "parts": parts,
                                    "held": "קודם תיקון - ראשונה בתור לבדיקות התיקון"}
                pr = _read(ROOT / "data" / "agents" / "priority.json", {})
                e = pr.setdefault(bot, {}).setdefault(c, {"families": ["exits", "filters", "manage"], "why": []})
                if not any("יתרון" in w for w in e.get("why", [])):
                    e.setdefault("why", []).append(f"{today}: היתרון ירד ({m0:+.2f}R ← {m:+.2f}R)")
                    e["at"] = at
                    _write(ROOT / "data" / "agents" / "priority.json", pr)
                continue
            if target < cur and (hurt := PC.cut_hurts(bot, c, cur, target)):
                # the whole bot first (user, 12.11.2026): a cut that leaves the bot riskier on 2026 is held
                out["combos"][c] = {"base": base, "now": cur, "target": target, "edge0": m0, "edge": m, "parts": parts,
                                    "held": hurt}
                continue
            out["combos"][c] = {"base": base, "now": cur, "target": target, "edge0": m0, "edge": m, "parts": parts}
            if abs(target - cur) < 0.01:
                mine[c] = {"base": base, "last": cur}
                continue
            did = hashlib.sha1(f"edge|{bot}|{c}|{target}|{today}".encode()).hexdigest()[:12]
            if did in have:
                continue
            up = target > cur
            adds.append({"id": did, "bot": bot, "type": "risk", "kind": "mult", "combo": c,
                         "title": f"{c}: " + ("כמות מינימלית - אין יתרון" if target == LR.MIN_LOT else
                                              f"סיכון ×{cur:g} ← ×{target:g} ({'מצליחה יותר מהבדיקה' if up else 'היתרון קטן'})"),
                         "decision": "approved", "by": "manager", "read": False, "undone": False, "at": at,
                         "why": "המנהל: הסיכון של כל אסטרטגיה הולך אחרי היתרון שלה, למעלה ולמטה: " + "; ".join(parts) + why_cap + ".",
                         "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "edge_risk": True,
                         "action": {"bot": bot, "combo": c, "kind": "mult", "value": target, "prev": cur}})
            have.add(did)
            mine[c] = {"base": base, "last": target}
    if adds:
        ddoc["decisions"] = adds + ddoc.get("decisions", [])
        ddoc["new"] = list(ddoc.get("new") or []) + [x["id"] for x in adds]
        _write(DEC, ddoc)
        for x in adds:
            print("edge risk:", x["title"])
    st["day"], st["at"], st["now"] = today, at, rows
    _write(STATE, st)
    return adds


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
