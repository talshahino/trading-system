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
# manager, 15.11.2026: a raise needs at least this many live trades (as a bot agent's risk card, autodecide
# MIN_TRADES_MULT). On 14.11 22:27 one live trade raised FR03 GBPUSD x1.47, TV37 GBPUSD x1.86, TV02 WMT x1.28, TV28 AMZN
# x1.39 (bot 2 / 3): where the test of 2026 and the selection period disagree, the formula trusts the live trades
# almost alone, and a single winner is luck, not proof ("up: slowly ... no maybe-luck raise").
MIN_LIVE_UP = 5
KEEP_UP = {"bot3"}         # the manager's standing instruction: bot 3's risk is never lowered - its raises are not taken back


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


def _own_last(ddoc):
    """The latest approved, not undone decision of this module per strategy: {(bot, combo): {value, prev, raise}}
    (prev = the approved risk its chain of decisions started from); dropped when someone else set that strategy's risk since."""
    mine_last = {}
    for d in sorted(ddoc.get("decisions", []), key=lambda x: x.get("at", "")):
        a = d.get("action") or {}
        if not (d.get("type") == "risk" and a.get("kind") == "mult" and a.get("combo")):
            continue
        key = (a.get("bot") or d.get("bot"), a["combo"])
        if d.get("decision") != "approved" or d.get("undone"):
            continue
        if d.get("edge_risk"):
            prev = float(a.get("prev") if a.get("prev") is not None else 1.0)
            old = mine_last.get(key)
            base0 = old["prev"] if old and abs(old["value"] - prev) < 0.005 else prev     # a chain of own decisions
            mine_last[key] = {"value": float(a["value"]), "prev": base0, "raise": float(a["value"]) > base0}
        else:
            mine_last.pop(key, None)
    return mine_last


def unwind_thin(states=None):
    """Between the daily runs (cheap, every call): our own raise made from fewer than MIN_LIVE_UP live trades goes
    back to the risk it started from (manager, 15.11.2026 - the raises of 14.11 22:27 were made from 1-3 trades)."""
    from agents import health as H
    from agents import loss_review as LR
    from agents import portfolio_check as PC
    ddoc = _read(DEC, {"decisions": []})
    own = {k: v for k, v in _own_last(ddoc).items() if v["raise"]}
    if not own:
        return []
    states = states if states is not None else H.live_states()
    have = {x["id"] for x in ddoc.get("decisions", [])}
    at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    st = _read(STATE, {})
    adds = []
    for bot in sorted({b for b, _ in own}):
        mm = PC.mults(bot)
        lt = LR.trade_rows(H.state_of(bot, states) or {}) if states else pd.DataFrame()
        for (b, c), v in own.items():
            if b != bot or bot in KEEP_UP:
                continue
            n = int((lt["combo"] == c).sum()) if len(lt) else 0
            cur = float(mm.get(c, 1.0) or 1.0)
            if n >= MIN_LIVE_UP or cur <= v["prev"] + 0.005:
                continue
            did = hashlib.sha1(f"edgeback|{bot}|{c}|{v['value']}|{v['prev']}".encode()).hexdigest()[:12]
            if did in have:
                continue
            adds.append({"id": did, "bot": bot, "type": "risk", "kind": "mult", "combo": c,
                         "title": f"{c}: סיכון ×{cur:g} ← ×{v['prev']:g} (ההעלאה חוזרת - רק {n} עסקאות בחי)",
                         "decision": "approved", "by": "manager", "read": False, "undone": False, "at": at,
                         "why": f"המנהל: העלאת סיכון לפי החי רק מ־{MIN_LIVE_UP} עסקאות בחי; ההעלאה ל־×{cur:g} נעשתה מ־{n} - "
                                "עסקה אחת טובה היא מזל, לא הוכחה. חוזר לסיכון המאושר.",
                         "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "edge_risk": True,
                         "action": {"bot": bot, "combo": c, "kind": "mult", "value": v["prev"], "prev": cur}})
            have.add(did)
            st.setdefault("bots", {}).setdefault(bot, {})[c] = {"base": v["prev"], "last": v["prev"]}
    if adds:
        ddoc["decisions"] = adds + ddoc.get("decisions", [])
        ddoc["new"] = list(ddoc.get("new") or []) + [x["id"] for x in adds]
        _write(DEC, ddoc)
        _write(STATE, st)
        for x in adds:
            print("edge risk back:", x["title"])
    return adds


def run(states=None, force=False):
    from agents import health as H
    from agents import loss_review as LR
    from agents import portfolio_check as PC
    st = _read(STATE, {})
    today = str(pd.Timestamp.now(tz=TZ).date())
    if st.get("day") == today and not force:
        return unwind_thin(states)
    states = states if states is not None else H.live_states()
    ddoc = _read(DEC, {"decisions": []})
    have = {x["id"] for x in ddoc.get("decisions", [])}
    at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    adds, rows = [], {}
    mine_last = _own_last(ddoc)
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
            if me.get("last") is None and (own := mine_last.get((bot, c))) and abs(float(own["value"]) - cur) < 0.005:
                # the state file was lost (e.g. reset by an install): our own latest decision on it says the same
                me = {"base": float(own.get("prev") or 1.0), "last": cur}
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
            if target > cur and len(lr) < MIN_LIVE_UP:
                out["combos"][c] = {"base": base, "now": cur, "target": target,
                                    "held": f"רק {len(lr)} עסקאות בחי - מעלים מ־{MIN_LIVE_UP}"}
                # an own earlier raise from fewer trades goes back (not in bot 3)
                target = cur if (cur <= base or bot in KEEP_UP) else base
                if abs(target - cur) < 0.01:
                    continue
            if target > cur:
                if not raise_ok or c not in book:
                    out["combos"][c] = {"base": base, "now": cur, "target": target, "held": "עוד אין בדיקה לכל הבוט - לא מעלים"}
                    continue
                cap = round(ceiling(bot, {**book, c: (book[c][0], cur)}, c, cur, target, risk), 2)
                if cap < target:
                    why_cap += f"; הבוט השלם על 2026 נשאר במגבלות שלו רק עד ×{cap}"
                    target = cap
            own_back = target < cur and me.get("last") is not None and abs(float(me["last"]) - cur) < 0.005 \
                and target >= base - 0.005                       # only taking back its own raise: no fix needed first
            if target < cur and not own_back and not fixes_failed(bot, c):
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
            if target < cur and not own_back and (hurt := PC.cut_hurts(bot, c, cur, target)):
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
