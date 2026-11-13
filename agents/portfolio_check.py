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
    from agent_portfolio import stale_source
    d = _read(ROOT / "reports" / "agents" / bot / "exp" / f"{fam}_{combo.replace('|', '_').replace('/', '-')}.json")
    if stale_source(d):
        return None
    if d and d.get("tr"):
        from agent_portfolio import tr_ok
        if not tr_ok(d):          # trades stored in the old, 1000x too coarse minutes: unknown until run again
            d = {**d, "tr": {}}
    return d


LATE_MIN = 15             # live/runner.py: no new stock trade in the last 15 minutes of the session


def live_entries(combo, doc, tr):
    """A stock combo's trades as the live bot would take them: the experiments stored before 13.11.2026 (no
    "stock_entry" stamp) still hold entries in the last 15 minutes of the session, which the live bots skip."""
    if not tr or (doc or {}).get("stock_entry"):
        return tr
    from agents import ftmo_universe as U
    sym = combo.split("|")[-1]
    if U.CLASS.get(sym) != "stock":
        return tr
    tz, close_min = ("America/New_York", 16 * 60) if "USD" in U.CURRENCIES[sym] else ("Europe/Berlin", 17 * 60 + 30)
    tr = _minutes(tr)
    loc = pd.to_datetime([x[0] for x in tr], unit="m", utc=True).tz_convert(tz)
    keep = (loc.hour * 60 + loc.minute) < close_min - LATE_MIN
    return [x for x, k in zip(tr, keep) if k]


def current_trades(bot, combo, cfg, key="tr"):
    """The combo's 2026 trades as it trades now (its built exit plan, or its built filter), None when not stored.
    key "tr25": the choosing period's trades (30.04.2025-31.12.2025) instead, where the agents stored them."""
    from agent_portfolio import built_variant
    p = (cfg.get("plans") or {}).get(combo) or {}
    if p.get("ladder"):                                          # a built laddered entry (13.11.2026): its own trades
        d = exp(bot, "ladder", combo) or {}
        return live_entries(combo, d, (d.get(key) or {}).get(built_variant(bot, combo, "ladder")))
    if p.get("filter"):
        d = exp(bot, "filters", combo) or {}
        return live_entries(combo, d, (d.get(key) or {}).get(p["filter"]))
    if any(k in p for k in ("be_way", "be_r", "partial")):       # a built management card: its own trades
        v = built_variant(bot, combo, "manage")
        for fam in ("manage", "fib"):                               # a Fibonacci card is built as "manage"
            d = exp(bot, fam, combo) or {}
            tr = (d.get(key) or {}).get(v)
            if tr:
                return live_entries(combo, d, tr)
        return None
    d = exp(bot, "exits", combo) or {}
    return live_entries(combo, d, (d.get(key) or {}).get(built_variant(bot, combo, "exits")))


def trades_of_card(bot, card):
    ch = card.get("change") or {}
    combo = ch.get("combo")
    if card.get("kind") == "new":
        d = exp(bot, "new", combo) or {}
        return combo, live_entries(combo, d, (d.get("tr") or {}).get("new"))
    fam = ch.get("src") or ch.get("family")                         # src "fib": built as "manage", stored as fib
    d = exp(bot, fam, combo) or {}
    return combo, live_entries(combo, d, (d.get("tr") or {}).get(ch.get("to")))


def _minutes(tr):
    """Trades stored before the minutes fix (12.11.2026) are in 1000-minute steps (2026 ~ 29,500 instead of ~29.5
    million): put them at the middle of their step until the agents re-run them (agent_portfolio.legacy_tr)."""
    if tr and tr[0][0] < 10_000_000:
        return [[a * 1000 + 500, b * 1000 + 500, d, r] for a, b, d, r in tr]
    return tr


def replay(book, risk):
    """book: {combo: (trades, multiplier)} -> account numbers in % of the opening balance."""
    rows = [(a, b, d, r * risk * 100 * m, risk * 100 * m) for combo, (tr, m) in book.items() if tr and m > 0
            for a, b, d, r in _minutes(tr)]
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


def book_now(bot, cfg=None, key="tr"):
    cfg = cfg or bot_cfg(bot)
    mm = mults(bot)
    combos = cfg.get("combos") or []
    book, missing = {}, []
    for c in combos:
        tr = current_trades(bot, c, cfg, key)
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


def cut_hurts(bot, combo, cur, val):
    """The bot as a whole comes first (user, 12.11.2026): a risk cut that leaves the whole bot replayed on 2026 riskier
    (deeper fall, worse day or more days near the limit) is no risk cut - the strategy's trades offset the others' bad
    days (13.11.2026: FR01 EURUSD bot 4 at the minimum lot -> the bot's fall on 2026 9.34% instead of 7.96%, beyond the
    bot's 9%). Hebrew reason when it hurts, else None (also None when the strategy has no stored 2026 trades)."""
    try:
        cfg = bot_cfg(bot)
        book, _ = book_now(bot, cfg)
        if combo not in book or val >= cur:
            return None
        risk = bot_risk(bot, cfg)
        tr = book[combo][0]
        before = replay({**book, combo: (tr, cur)}, risk)
        after = replay({**book, combo: (tr, val)}, risk)
    except Exception:  # noqa: BLE001
        return None
    if riskier(after, before):
        return (f"בבוט השלם על 2026 ההורדה מוסיפה סיכון (ירידה מהשיא {before['max_dd']:.2f}% → {after['max_dd']:.2f}%, "
                f"יום גרוע {before['worst_day']:.2f}% → {after['worst_day']:.2f}%) - העסקאות שלה מקזזות ימים רעים של אחרות")
    return None


def side_conflicts(bot, cfg=None):
    """A pattern in the bots' own trades (13.11.2026): a trade opened while ANOTHER strategy of the same bot holds the
    opposite side of the same instrument. On the demo (bot 1, USDJPY 12.11: SE02 buy and TV11 sell both stopped within
    the hour) and in the 2026 replay of bots 1-4 these lost (-0.09R a trade, 172 trades) while trades in the same
    direction (+0.29R) and alone (+0.09R) won. The rule under test: skip a new trade against an open opposite position
    on the same instrument in the same bot. Per period - 'train' (30.04.2025-31.12.2025, chooses) and 'unseen' (2026,
    only shows) - the trades the rule would skip (n, R sum, R a trade) and the whole bot with and without them.
    Nothing changes in the bot here: a card goes up only when the skipped trades lose in both periods."""
    cfg = cfg or bot_cfg(bot)
    risk = bot_risk(bot, cfg)
    out = {}
    for per, key in (("train", "tr25"), ("unseen", "tr")):
        book, missing = book_now(bot, cfg, key)
        rows = sorted((a, b, d, r, c) for c, (tr, m) in book.items() if tr and m > 0 for a, b, d, r in _minutes(tr))
        held, skip = [], set()                     # held: (exit, instrument, direction, combo) of trades taken
        for i, (a, b, d, r, c) in enumerate(rows):
            sym = c.split("|")[2]
            held = [h for h in held if h[0] > a]
            if any(h[1] == sym and h[2] != d and h[3] != c for h in held):
                skip.add(i)
                continue
            held.append((b, sym, d, c))
        sk = [rows[i][3] for i in skip]
        keep = {}
        for i, (a, b, d, r, c) in enumerate(rows):
            if i not in skip:
                keep.setdefault(c, ([], float(mults(bot).get(c, 1.0))))[0].append([a, b, d, r])
        before = replay(book, risk)
        out[per] = {"combos": len(book), "missing": len(missing), "skipped": len(sk), "skipped_r": round(sum(sk), 2),
                    "skipped_mean": round(sum(sk) / len(sk), 3) if sk else None, "before": before,
                    "after": replay(keep, risk) if rows else before}
    return out


def legs(sym, d):
    """The currencies a trade is long (+1) / short (-1): EURUSD buy = +EUR -USD, XAUUSD / XAGUSD buy = -USD, DXY buy =
    +USD. Indices, stocks, energy and crypto carry no currency leg here (manager task 13.11.2026: the same currency in
    the same direction - USDJPY in several strategies of bots 1-2, EURUSD and GBPUSD together at 06:00 in bot 4)."""
    from agents import ftmo_universe as U
    cl, cc = U.CLASS.get(sym), U.CURRENCIES.get(sym) or []
    if cl == "fx" and len(cc) == 2:
        return [(cc[0], d), (cc[1], -d)]
    if cl == "metal" and cc == ["USD"]:
        return [("USD", -d)]
    if sym == "DXY":
        return [("USD", d)]
    return []


CCY_RULES = {"cap1": ("cap", 1.0), "cap1.5": ("cap", 1.5), "cap2": ("cap", 2.0), "half2nd": ("half", None)}


def currency_rule(rows, mm, rule):
    """rows: (entry, exit, dir, R, combo) in time order -> {combo: ([trades], mult)} with the rule applied.
    cap k: a new trade's risk is cut so the open risk on any currency in that direction stays <= k x one trade's risk
           (nothing left -> no trade); half2nd: a trade that adds to an open currency exposure in the same direction
           goes in at half its risk. The open exposure counts every open trade's own multiplier."""
    kind, k = CCY_RULES[rule]
    held, out = [], {}                          # held: (exit, [(ccy, sign)], mult)
    for a, b, d, r, c in rows:
        held = [h for h in held if h[0] > a]
        m = float(mm.get(c, 1.0))
        lg = legs(c.split("|")[2], d)
        if lg:
            exp_ = max((sum(h[2] for h in held if (ccy, s) in h[1]) for ccy, s in lg), default=0.0)
            if kind == "cap":
                m = min(m, max(0.0, k - exp_))
            elif exp_ > 0:
                m *= 0.5
        if m <= 1e-9:
            continue
        held.append((b, lg, m))
        out.setdefault(c, ([], m))
        out[c][0].append([a, b, d, r * m / float(mm.get(c, 1.0) or 1.0)])   # R scaled so replay's own mult stays
    return {c: (tr, float(mm.get(c, 1.0))) for c, (tr, _) in out.items()}


def currency_exposure(bot, cfg=None):
    """The manager's task of 13.11.2026 (bots 1, 2, 4): a cap on the open risk in one currency and one direction, or a
    second trade in the same currency and direction at half risk - the whole bot replayed per period, 'train'
    (30.04.2025-31.12.2025, chooses) and 'unseen' (2026, only shows), by the TOTAL and the fall from the peak."""
    cfg = cfg or bot_cfg(bot)
    risk, mm = bot_risk(bot, cfg), mults(bot)
    out = {}
    for per, key in (("train", "tr25"), ("unseen", "tr")):
        book, missing = book_now(bot, cfg, key)
        rows = sorted((a, b, d, r, c) for c, (tr, m) in book.items() if tr and m > 0 for a, b, d, r in _minutes(tr))
        fx = sum(1 for x in rows if legs(x[4].split("|")[2], x[2]))
        res = {"combos": len(book), "missing": len(missing), "trades": len(rows), "fx_trades": fx,
               "now": replay(book, risk)}
        for rule in CCY_RULES:
            res[rule] = replay(currency_rule(rows, mm, rule), risk) if rows else res["now"]
        out[per] = res
    return out


def daily_pnl(book, risk):
    """The account's profit per FTMO day in % of the opening balance (the same booking as replay())."""
    rows = [(b, r * risk * 100 * m) for combo, (tr, m) in book.items() if tr and m > 0 for a, b, d, r in _minutes(tr)]
    if not rows:
        return []
    t = pd.DataFrame(rows, columns=["b", "pnl"])
    day = pd.to_datetime(t["b"], unit="m", utc=True).dt.tz_convert("Europe/Prague").dt.date
    return [float(x) for x in t.groupby(day)["pnl"].sum().sort_index().to_numpy()]


def fit_stress(days, start, fit, limit=100 * (1 - 0.09)):
    """The 2026 days replayed in order from an account at <start> % (bots 1-4 smart agent, 13.11.2026 night): as the bot
    trades now, and with "dd_fit" (live/runner.py: risk x min(1, room left to the 9% limit / fit), set at the start of
    every day). The account stops at the limit (the bot's own guard). End, lowest point and whether the limit was hit."""
    out = {}
    for name, use in (("now", False), ("dd_fit", True)):
        eq, low, hit = start, start, False
        for p in days:
            f = min(1.0, max(0.0, (eq - limit) / fit)) if use and fit > 0 else 1.0
            eq += p * f
            low = min(low, eq)
            if eq <= limit:
                hit, eq = True, limit
                break
        out[name] = {"end": round(float(eq), 2), "low": round(float(low), 2), "hit_limit": bool(hit)}
    return out


def _balance_pct(bot, cfg):
    h = _read(ROOT / "reports" / "agents" / bot / "health.json", {}) or {}
    try:
        return float(h["balance"]) / float(cfg.get("initial") or 100000) * 100
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


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
        try:
            conf = side_conflicts(bot, cfg)
        except Exception as e:  # noqa: BLE001
            conf = {"error": str(e)[-200:]}
        # the whole-bot risk rule "dd_fit" stress-tested: 2026 replayed from today's balance and from the start
        try:
            days, fit = daily_pnl(book, bot_risk(bot, cfg)), float(cfg.get("dd_fit") or r["max_dd"] or 0)
            bal = _balance_pct(bot, cfg)
            stress = {"fit": fit, "from_100": fit_stress(days, 100.0, fit),
                      **({"balance": round(bal, 2), "from_balance": fit_stress(days, bal, fit)} if bal else {})}
        except Exception as e:  # noqa: BLE001
            stress = {"error": str(e)[-200:]}
        try:
            ccy = currency_exposure(bot, cfg)
        except Exception as e:  # noqa: BLE001
            ccy = {"error": str(e)[-200:]}
        out.write_text(json.dumps({"updated": at, "bot": bot, "replay_2026": r, "combos": len(book), "missing": missing,
                                   "warnings": warn, "side_conflicts": conf, "dd_fit_stress": stress,
                                   "currency_exposure": ccy},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(bot, r, f"missing {len(missing)}", warn)


if __name__ == "__main__":
    daily()
