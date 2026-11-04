"""Monte Carlo check of the live bots: how likely is each bot to pass / breach the FTMO rules if its
trading days had come in a different order?

The backtest shows ONE history. Here each bot's real backtest days (its combos' trades, with the FTMO
news rule applied) are drawn again and again in random order, in blocks of 5 trading days (so a bad week
stays a bad week), for N simulated paths. Each path replays the bot's own rules:

  * risk per trade = the bot's risk x the initial balance (a trade's $ = R x that)
  * the day's equity path: trades by close time; the bot stops for the day at -4.5% (emergency stop)
  * no new trades once the account is 9% down (the bot's total guard): the path is "stuck"
  * FTMO: a day at -5% of the initial balance below the day's start = daily breach; equity at 90% of the
    initial balance = total breach; +10% = target

Two questions per bot:
  challenge  - starting at $100K: target before a breach? (and in how many trading days)
  funded     - one year (250 trading days) of trading: breached at some point? the year's result?

Approximation: equity is counted at each trade's close (the backtest has no floating P/L inside a
trade), so intraday lows are a little better than real; the 4.5% stop makes a 5% breach rare anyway.

  python montecarlo.py --trades reports/X/news_trades.csv.gz --out reports/montecarlo/result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INIT = 100_000.0
DAY_STOP, DAY_LIMIT, TOTAL_GUARD, TOTAL_LIMIT, TARGET = 0.045, 0.05, 0.09, 0.10, 0.10


def bot_days(t, combos, risk, start, end, edge=1.0):
    """One list per trading day (Prague), each a list of $ results in close order. edge < 1 shrinks every
    trade by the same amount so the bot's average R becomes edge x the backtest's (a live bot that earns
    less than its backtest: costs, slippage, a weaker market)."""
    d = t[t["combo"].isin(combos)].copy()
    d["r"] = d["r"] - (1 - edge) * d["r"].mean()
    d["exit_time"] = pd.to_datetime(d["exit_time"], utc=True)
    d["day"] = d["exit_time"].dt.tz_convert("Europe/Prague").dt.normalize().dt.tz_localize(None)
    d = d.sort_values("exit_time")
    usd = INIT * risk
    by = {k: (g["r"].to_numpy() * usd).tolist() for k, g in d.groupby("day")}
    days = pd.bdate_range(start, end)
    return [by.get(x, []) for x in days]


def run_path(days_seq, mode, horizon):
    eq = INIT
    low_total = eq
    stuck = False
    for k, day in enumerate(days_seq[:horizon]):
        start = eq
        for pnl in day:
            if stuck or eq <= INIT * (1 - TOTAL_GUARD):
                stuck = True
                break
            eq += pnl
            low_total = min(low_total, eq)
            if eq <= start - DAY_LIMIT * INIT:
                return ("daily", k + 1, eq, low_total)
            if eq <= INIT * (1 - TOTAL_LIMIT):
                return ("total", k + 1, eq, low_total)
            if eq <= start - DAY_STOP * INIT:          # the bot closes everything and waits for tomorrow
                break
            if mode == "challenge" and eq >= INIT * (1 + TARGET):
                return ("pass", k + 1, eq, low_total)
    if stuck:
        return ("stuck", horizon, eq, low_total)
    return ("open", horizon, eq, low_total)


def simulate(days, n, horizon, mode, block=5, seed=7):
    rng = np.random.default_rng(seed)
    nb = len(days) - block + 1
    out = []
    for _ in range(n):
        seq = []
        while len(seq) < horizon:
            i = int(rng.integers(0, nb))
            seq.extend(days[i:i + block])
        out.append(run_path(seq, mode, horizon))
    return out


def summary(res, mode):
    kinds = pd.Series([r[0] for r in res])
    pct = lambda k: round(100 * float((kinds == k).mean()), 1)  # noqa: E731
    eq = np.array([r[2] for r in res])
    lows = np.array([r[3] for r in res])
    s = {"breach_daily_pct": pct("daily"), "breach_total_pct": pct("total"), "stuck_pct": pct("stuck")}
    if mode == "challenge":
        days = np.array([r[1] for r in res if r[0] == "pass"])
        s.update(pass_pct=pct("pass"), not_reached_pct=pct("open"),
                 days_to_pass_median=int(np.median(days)) if len(days) else None,
                 days_to_pass_p90=int(np.percentile(days, 90)) if len(days) else None)
    else:
        s.update(survive_year_pct=pct("open"),
                 year_return_median_pct=round(100 * (np.median(eq) / INIT - 1), 1),
                 year_return_p10_pct=round(100 * (np.percentile(eq, 10) / INIT - 1), 1),
                 year_return_p90_pct=round(100 * (np.percentile(eq, 90) / INIT - 1), 1),
                 deepest_low_median_pct=round(100 * (np.median(lows) / INIT - 1), 1),
                 deepest_low_p5_pct=round(100 * (np.percentile(lows, 5) / INIT - 1), 1))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--start", default="2025-04-30")
    ap.add_argument("--end", default="2026-10-30")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--combos", default="", help="bot=file of combos it really trades (one per line), e.g. 2=x.txt")
    ap.add_argument("--edge", default="1,0.5,0.25", help="share of the backtest's average R kept, one run each")
    ap.add_argument("--out", default="reports/montecarlo/result.json")
    args = ap.parse_args()
    t = pd.read_csv(args.trades)
    over = dict(x.split("=") for x in args.combos.split(",") if x)
    rows = []
    for edge in (float(x) for x in args.edge.split(",")):
        for b in (1, 2, 3, 4):
            cfg = json.load(open(ROOT / "live" / "bots" / f"bot{b}.json", encoding="utf-8"))
            combos = cfg["combos"]
            if str(b) in over:
                combos = [l.strip() for l in open(over[str(b)], encoding="utf-8") if l.strip()]
            days = bot_days(t, combos, cfg["risk"], args.start, args.end, edge)
            hist = run_path(days, "funded", len(days))
            row = {"bot": b, "edge": edge, "risk": cfg["risk"], "combos": len(combos), "days": len(days),
                   "trades": int(sum(len(x) for x in days)),
                   "history": {"result": hist[0], "end_pct": round(100 * (hist[2] / INIT - 1), 1),
                               "low_pct": round(100 * (hist[3] / INIT - 1), 1)},
                   "challenge": summary(simulate(days, args.n, args.horizon, "challenge"), "challenge"),
                   "funded": summary(simulate(days, args.n, args.horizon, "funded", seed=11), "funded")}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
