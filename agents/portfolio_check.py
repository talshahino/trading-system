"""The whole-bot check (user, 12.11.2026): a change can be good on its own and still make the bot worse - FTMO's limits
(5% a day, 10% in all) are on the account, not on each strategy. So before the manager approves a change to a bot, the
whole bot is replayed on 2026 twice: as it trades now, and with the change. Nothing is optimised here: every combo's own
2026 trades from the agents' experiments (reports/agents/<bot>/exp, "tr") are merged in time order into one account.

Per replay, on the account (each trade = R x the bot's risk % x the combo's risk multiplier, booked on its exit day):
  total      profit in % of the opening balance
  worst_day  the worst day in %
  max_dd     the deepest fall from a peak in %
  near_days  days that lost 3% or more (the bot's own daily limit is 4.5%, FTMO's 5%)
  max_open   the most risk open at one time in % (all trades), and max_side: the most open in one direction
Rule (agents/autodecide.py uses it):
  more profit, no more risk          -> approved
  more profit, more risk             -> approved at a lower risk for that combo: the highest multiplier (down to 0.5)
                                        at which the bot's worst day, drawdown and near-limit days are back to today's
  no more profit                     -> not approved, even when the change looks good on its own
"Risk not up" allows a hair (0.1% of the balance) so rounding does not block a change.
Every day the bot as it is now is replayed too (reports/agents/<bot>/portfolio.json) - changes that went in one by one
must not add up to too much risk; an info card when the 2026 replay comes near the limits.

  python -m agents.portfolio_check            (the daily replay of every bot, writes portfolio.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import repo_io as RIO

ROOT = Path(__file__).resolve().parent.parent
BOTS = ("bot1", "bot2", "bot3", "bot4", "bot7")
EPS = 0.1                # % of the balance: "not more risk" tolerance
NEAR = 3.0               # % a day: a day near the limit
RISK7 = 0.005            # bot 7 (virtual): risk per trade


def _read(p, d=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return d


def bot_cfg(bot):
    return _read(ROOT / "live" / "bots" / f"{bot}.json", {}) or {}


def bot_risk(bot, cfg=None):
    cfg = cfg or bot_cfg(bot)
    return float(cfg.get("risk") or RISK7)


def mults(bot):
    """The risk multipliers in force on the bots' computer (branch live, live/manager_state.json)."""
    return ((RIO.live_json("live/manager_state.json").get("risk") or {}).get(bot) or {}).get("mult") or {}


def exp(bot, fam, combo):
    return _read(ROOT / "reports" / "agents" / bot / "exp" / f"{fam}_{combo.replace('|', '_').replace('/', '-')}.json")


def current_trades(bot, combo, cfg):
    """The combo's 2026 trades as it trades now (its built exit plan, or its built filter), None when not stored."""
    from agent_portfolio import built_variant
    p = (cfg.get("plans") or {}).get(combo) or {}
    if p.get("filter"):
        d = exp(bot, "filters", combo) or {}
        return (d.get("tr") or {}).get(p["filter"])
    d = exp(bot, "exits", combo) or {}
    return (d.get("tr") or {}).get(built_variant(bot, combo, "exits"))


def trades_of_card(bot, card):
    ch = card.get("change") or {}
    combo = ch.get("combo")
    if card.get("kind") == "new":
        d = exp(bot, "new", combo) or {}
        return combo, (d.get("tr") or {}).get("new")
    fam = ch.get("family")
    d = exp(bot, fam, combo) or {}
    return combo, (d.get("tr") or {}).get(ch.get("to"))


def replay(book, risk):
    """book: {combo: (trades, multiplier)} -> account numbers in % of the opening balance."""
    rows = [(a, b, d, r * risk * 100 * m, risk * 100 * m) for combo, (tr, m) in book.items() if tr and m > 0
            for a, b, d, r in tr]
    if not rows:
        return {"trades": 0, "total": 0.0, "worst_day": 0.0, "max_dd": 0.0, "near_days": 0, "max_open": 0.0, "max_side": 0.0}
    t = pd.DataFrame(rows, columns=["a", "b", "d", "pnl", "risk"])
    day = pd.to_datetime(t["b"], unit="m", utc=True).dt.tz_convert("Europe/Prague").dt.date      # FTMO's day (CE(S)T)
    daily = t.groupby(day)["pnl"].sum().sort_index()
    eq = np.r_[100.0, 100 + daily.cumsum().to_numpy()]
    dd = float((np.maximum.accumulate(eq) - eq).max())
    ev = sorted([(a, r, d) for a, b, d, _, r in t.itertuples(index=False)] + [(b, -r, d) for a, b, d, _, r in t.itertuples(index=False)],
                key=lambda x: (x[0], x[1]))
    op = side_l = side_s = mo = ms = 0.0
    for _, r, d in ev:
        op += r
        if d > 0:
            side_l += r
        else:
            side_s += r
        mo, ms = max(mo, op), max(ms, side_l, side_s)
    return {"trades": int(len(t)), "total": round(float(daily.sum()), 2), "worst_day": round(float(min(0.0, daily.min())), 2),
            "max_dd": round(dd, 2), "near_days": int((daily <= -NEAR).sum()), "max_open": round(mo, 2), "max_side": round(ms, 2)}


def book_now(bot, cfg=None):
    cfg = cfg or bot_cfg(bot)
    mm = mults(bot)
    combos = cfg.get("combos") or []
    book, missing = {}, []
    for c in combos:
        tr = current_trades(bot, c, cfg)
        if tr is None:
            missing.append(c)
            continue
        book[c] = (tr, float(mm.get(c, 1.0)))
    return book, missing


def riskier(new, old):
    return (new["worst_day"] < old["worst_day"] - EPS or new["max_dd"] > old["max_dd"] + EPS
            or new["near_days"] > old["near_days"])


def check(bot, card):
    """The whole bot with and without the card's change. {"ok": bool, "mult": m, "why": Hebrew, "before", "after"} or
    None when the trades are not stored yet (the agents re-run those combos first)."""
    if bot not in BOTS or card.get("kind") not in ("improve", "new"):
        return None
    cfg = bot_cfg(bot)
    combo, tr = trades_of_card(bot, card)
    if not combo or tr is None:
        return None
    book, missing = book_now(bot, cfg)
    if len(missing) > max(2, 0.2 * len(cfg.get("combos") or [])):
        return None                                   # too much of the bot unknown yet to judge it as a whole
    risk = bot_risk(bot, cfg)
    before = replay(book, risk)
    m0 = book.get(combo, (None, float(mults(bot).get(combo, 1.0))))[1] or 1.0
    after = replay({**book, combo: (tr, m0)}, risk)
    out = {"before": before, "after": after, "missing": missing, "combo": combo}
    if after["total"] <= before["total"]:
        return {**out, "ok": False, "mult": None,
                "why": f"בבוט השלם על 2026 לא מוסיף רווח ({after['total']:+.2f}% מול {before['total']:+.2f}% היום)"}
    if not riskier(after, before):
        return {**out, "ok": True, "mult": None,
                "why": f"בבוט השלם על 2026: רווח {before['total']:+.2f}% → {after['total']:+.2f}%, בלי יותר סיכון "
                       f"(יום גרוע {after['worst_day']:.2f}%, ירידה {after['max_dd']:.2f}%)"}
    for m in (0.9, 0.8, 0.7, 0.6, 0.5):
        a2 = replay({**book, combo: (tr, m0 * m)}, risk)
        if a2["total"] > before["total"] and not riskier(a2, before):
            return {**out, "after": a2, "ok": True, "mult": round(m0 * m, 2),
                    "why": f"בבוט השלם השינוי מוסיף רווח אבל גם סיכון; בסיכון ×{m0 * m:.2f} לאסטרטגיה: רווח "
                           f"{before['total']:+.2f}% → {a2['total']:+.2f}% והסיכון לא עולה"}
    return {**out, "ok": False, "mult": None,
            "why": f"בבוט השלם מוסיף סיכון (יום גרוע {before['worst_day']:.2f}% → {after['worst_day']:.2f}%, "
                   f"ירידה {before['max_dd']:.2f}% → {after['max_dd']:.2f}%) גם בסיכון מוקטן"}


def daily(now=None):
    """The bot as it is now, every day: reports/agents/<bot>/portfolio.json (+ a warning when near the limits)."""
    at = (now or pd.Timestamp.now(tz="UTC")).isoformat(timespec="seconds")
    for bot in BOTS:
        cfg = bot_cfg(bot)
        if not cfg:
            continue
        book, missing = book_now(bot, cfg)
        r = replay(book, bot_risk(bot, cfg))
        warn = []
        if r["worst_day"] <= -3.5:
            warn.append(f"היום הגרוע ב־2026 {r['worst_day']:.2f}% - קרוב למגבלה היומית של הבוט (4.5%)")
        if r["max_dd"] >= 7:
            warn.append(f"ירידה מהשיא {r['max_dd']:.2f}% - קרוב למגבלה הכוללת של הבוט (9%)")
        if r["max_open"] >= 4.5:
            warn.append(f"עד {r['max_open']:.2f}% סיכון פתוח בבת אחת")
        out = ROOT / "reports" / "agents" / bot / "portfolio.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"updated": at, "bot": bot, "replay_2026": r, "combos": len(book), "missing": missing,
                                   "warnings": warn}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(bot, r, f"missing {len(missing)}", warn)


if __name__ == "__main__":
    daily()
