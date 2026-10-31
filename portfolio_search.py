"""Build FTMO portfolios out of the passing (strategy, timeframe, instrument) combos.

Goal (user, 31.10.2026): the combos whose combined profit is highest, while together they never
come near the FTMO daily loss limit; then a 2nd and 3rd portfolio from the remaining combos
(for separate accounts). Ranked best, second, third.

Search (fast, closed trades, day granularity):
  - one combo per instrument per portfolio (no two strategies fighting on one chart)
  - worst day (Prague day, by trade exit) no worse than -DAILY_CAP of the $100K
  - lowest balance never under $100K * (1 - MAX_DD)
  - greedy: keep adding the combo that raises the final profit most, until nothing helps
  - every risk level in RISKS is tried, the most profitable wins
The finalists are then checked minute by minute by Trading Prophet (verify_portfolio.py).

  python portfolio_search.py --trades 'reports/*/combo_trades_*.csv.gz' --out data/portfolio/portfolios.json
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

INITIAL = 100_000.0
RISKS = (0.0025, 0.005, 0.0075, 0.01)
DAILY_CAP = 0.03      # 2% buffer under FTMO's 5% for floating loss inside the day
MAX_DD = 0.07
EXCLUDE = {"SPCX"}    # SpaceX history from Alpaca is not reliable (14 trades averaging +55R)         # 3% buffer under FTMO's 10%


def load(pattern):
    fr = [pd.read_csv(f, parse_dates=["time", "exit_time"]) for f in sorted(glob.glob(pattern))]
    t = pd.concat(fr, ignore_index=True)
    t = t[~t["symbol"].isin(EXCLUDE)]
    t["combo"] = t["sid"] + "|" + t["tf"] + "|" + t["symbol"]
    t["day"] = t["exit_time"].dt.tz_convert("Europe/Prague").dt.normalize()
    return t.drop_duplicates(["combo", "time"])


def matrix(t):
    """Daily R per combo: rows = days, columns = combos."""
    m = t.pivot_table(index="day", columns="combo", values="r", aggfunc="sum").fillna(0.0)
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="D", tz=m.index.tz), fill_value=0.0)
    return m


def metrics(daily_r, risk):
    pnl = daily_r * risk * INITIAL
    bal = INITIAL + np.cumsum(pnl)
    return {"profit": float(bal[-1] - INITIAL), "worst_day": float(pnl.min()) if len(pnl) else 0.0,
            "min_bal": float(min(INITIAL, bal.min()))}


def ok(m):
    return m["worst_day"] >= -DAILY_CAP * INITIAL and m["min_bal"] >= INITIAL * (1 - MAX_DD)


def greedy(M, cols, sym_of, risk):
    chosen, cur = [], np.zeros(len(M))
    best = metrics(cur, risk)
    used = set()
    X = M[cols].to_numpy()
    while True:
        pick, pick_m = None, None
        for j, c in enumerate(cols):
            if c in chosen or sym_of[c] in used:
                continue
            m = metrics(cur + X[:, j], risk)
            if ok(m) and m["profit"] > (pick_m or best)["profit"]:
                pick, pick_m = j, m
        if pick is None:
            return chosen, best
        c = cols[pick]
        chosen.append(c)
        used.add(sym_of[c])
        cur = cur + X[:, pick]
        best = pick_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    t = load(args.trades)
    M = matrix(t)
    sym_of = t.drop_duplicates("combo").set_index("combo")["symbol"].to_dict()
    stats = t.groupby("combo")["r"].agg(["count", "mean", "sum"])
    pool = list(M.columns)
    out = []
    for k in range(args.n):
        best = None
        for rk in RISKS:
            chosen, m = greedy(M, pool, sym_of, rk)
            if chosen and (best is None or m["profit"] > best[2]["profit"]):
                best = (rk, chosen, m)
        if best is None:
            break
        rk, chosen, m = best
        out.append({"rank": k + 1, "risk": rk, "metrics": m, "combos": [
            {"combo": c, "trades": int(stats.loc[c, "count"]), "avg_r": round(float(stats.loc[c, "mean"]), 3),
             "sum_r": round(float(stats.loc[c, "sum"]), 1)} for c in chosen]})
        print(f"#{k + 1}: risk {rk:.2%} combos {len(chosen)} profit ${m['profit']:,.0f} "
              f"worst day ${m['worst_day']:,.0f} low ${m['min_bal']:,.0f}")
        pool = [c for c in pool if c not in chosen]
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
