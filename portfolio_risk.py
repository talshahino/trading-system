"""A risk manager across the 4 bots (research only, nothing in the bots changes).

Every trade is turned into exposures to a few shared "factors": a currency (long EURUSD = +EUR -USD), gold /
silver, US shares (US indices and US stocks), European shares, Japanese shares, oil, crypto. Then caps:
    acct:N   in ONE account, no new trade if it would make more than N open trades leaning the same way on
             the same factor (for example a 4th short-USD trade)
    all:N    the same across the 4 accounts together (the same move hitting every account at once)
Each bot is replayed as its own $100K FTMO account with its own risk (the bots' graded risk and 4.5% / 9%
guards), on the trades of the live combos with the live rules.
    choose  2025-04-30 .. 2025-12-31     check  2026-01-01 .. end
Reported per variant: profit of every account, its deepest drop (closed trades, % of $100K), the worst day of
the 4 accounts together and the number of days on which 2 or more accounts lost over 2%.

  python portfolio_risk.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import ftmo_agent as F

ROOT = Path(__file__).resolve().parent
BOTS_TRADES = ROOT / "reports" / "2026-11-04_bots_compare_run14" / "news_trades.csv.gz"
BOT4_TRADES = ROOT / "reports" / "2026-11-05_tv_run674" / "combo_trades_all.csv.gz"
SPLIT, START = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2025-04-30", tz="UTC")
CCY = {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"}
US_EQ = {"US100", "US30", "US500", "US2000"}
EU_EQ = {"GER40", "EU50", "FRA40", "UK100", "SPN35", "N25", "AIRF", "LVMH", "SIEGn", "RACE"}


def factors(sym, d):
    """{factor: +1/-1} of a trade in direction d."""
    s = sym.replace("/", "").replace(".cash", "")
    if len(s) == 6 and s[:3] in CCY and s[3:] in CCY:
        return {s[:3]: d, s[3:]: -d}
    if s.startswith(("XAU", "XAG", "XPT")):
        return {"METAL": d, "USD": -d}
    if s in ("WTI", "BRENT", "NATGAS"):
        return {"OIL": d}
    if s.endswith("USD") and len(s) <= 7:          # ETHUSD, XRPUSD, BTCUSD
        return {"CRYPTO": d}
    if s == "DXY":
        return {"USD": d}
    if s == "JP225":
        return {"JP_EQ": d, "JPY": -d}
    if s in EU_EQ:
        return {"EU_EQ": d}
    return {"US_EQ": d}                               # US indices and US stocks


def load():
    a = pd.read_csv(BOTS_TRADES)
    b = pd.read_csv(BOT4_TRADES)
    for d in (a, b):
        if "combo" not in d:
            d["combo"] = d["sid"] + "|" + d["tf"] + "|" + d["symbol"]
        for c in ("time", "exit_time"):
            d[c] = pd.to_datetime(d[c], utc=True)
        d["free_time"] = pd.to_datetime(d["free_time"], utc=True, errors="coerce")
    out = {}
    for n in (1, 2, 3, 4):
        cfg = json.load(open(ROOT / "live" / "bots" / f"bot{n}.json", encoding="utf-8"))
        src = a if n < 4 else b
        t = src[src["combo"].isin(cfg["combos"]) & (src["time"] >= START)].drop_duplicates(["combo", "time", "dir"])
        out[n] = (t.dropna(subset=["r"]).sort_values("time").reset_index(drop=True), cfg)
    return out


def capped(books, acct_n=None, all_n=None):
    """Drop trades that would break the caps (walked in time order over all accounts together)."""
    ev = []
    for n, (t, _) in books.items():
        for i, x in t.iterrows():
            ev.append((x["time"], n, i, x["exit_time"], factors(x["symbol"], int(x["dir"]))))
    ev.sort(key=lambda e: e[0])
    open_, keep = [], {n: [] for n in books}
    for tm, n, i, ex, fx in ev:
        open_ = [o for o in open_ if o[1] > tm]
        ok = True
        for f, sgn in fx.items():
            if acct_n is not None and sum(1 for o in open_ if o[0] == n and o[2].get(f) == sgn) >= acct_n:
                ok = False
            if all_n is not None and sum(1 for o in open_ if o[2].get(f) == sgn) >= all_n:
                ok = False
        if ok:
            open_.append((n, ex, fx))
            keep[n].append(i)
    return {n: (t.loc[keep[n]].reset_index(drop=True), cfg) for n, (t, cfg) in books.items()}


def replay(t, cfg, start):
    """The bots' own replay; returns daily P/L in % of $100K and whether the account was blown."""
    F.FREE_RISK_AT_BREAKEVEN = bool(cfg.get("free_risk_at_breakeven", True))
    t = t[t["time"] >= start]
    if t.empty:
        return pd.Series(dtype=float), None
    _, blown, _ = F.account_path(t, cfg["risk"], start=start)
    # closed-trade P/L per day at the bot's risk (the replay's own sizes are close to this: graded risk only
    # shrinks near the limits)
    pnl = (t["r"] * cfg["risk"] * 100).groupby(t["exit_time"].dt.floor("D")).sum()
    if blown is not None:
        pnl = pnl[pnl.index <= blown]
    return pnl, blown


def score(books, start, end):
    res, days = {}, {}
    for n, (t, cfg) in books.items():
        tt = t[(t["time"] >= start) & (t["time"] < end)]
        pnl, blown = replay(tt, cfg, start)
        eq = pnl.cumsum()
        dd = float((eq.cummax().clip(lower=0) - eq).max()) if len(eq) else 0.0
        res[f"bot{n}"] = {"trades": int(len(tt)), "profit_pct": round(float(pnl.sum()), 2), "max_dd_pct": round(dd, 2),
                          "blown": str(blown) if blown is not None else None}
        days[n] = pnl
    d = pd.DataFrame(days).fillna(0.0)
    res["all"] = {"profit_pct_sum": round(float(d.values.sum()), 2), "worst_day_sum_pct": round(float(d.sum(axis=1).min()), 2),
                  "days_2plus_accounts_below_-2pct": int(((d < -2).sum(axis=1) >= 2).sum()),
                  "profit_to_dd": round(float(d.values.sum()) / max(0.01, max(r["max_dd_pct"] for k, r in res.items()
                                                                                if k.startswith("bot"))), 2)}
    return res


def main():
    books = load()
    end = max(t["time"].max() for t, _ in books.values()) + pd.Timedelta(days=1)
    variants = [("today", None, None)] + [(f"acct:{n}", n, None) for n in (2, 3, 4)] + \
               [(f"all:{n}", None, n) for n in (4, 6, 8)] + [("acct:3+all:6", 3, 6)]
    out = {}
    for name, an, al in variants:
        b = capped(books, an, al) if (an or al) else books
        out[name] = {"choose": score(b, START, SPLIT), "check": score(b, SPLIT, end)}
        c, k = out[name]["choose"]["all"], out[name]["check"]["all"]
        print(f"{name:14s} choose: profit {c['profit_pct_sum']:7.2f}% worst day {c['worst_day_sum_pct']:6.2f}% "
              f"2+acct bad days {c['days_2plus_accounts_below_-2pct']:3d} | check: profit {k['profit_pct_sum']:7.2f}% "
              f"worst day {k['worst_day_sum_pct']:6.2f}% 2+acct bad days {k['days_2plus_accounts_below_-2pct']:3d}", flush=True)
    od = ROOT / "reports" / "portfolio_risk"
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
