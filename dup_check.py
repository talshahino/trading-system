"""Duplicate check between strategies, by their actual trades (not by their names or ideas).

Two combos (strategy|timeframe|instrument) "overlap" on a trade when they trade the same instrument in the
same direction with entries within WINDOW of each other. For every pair of combos on one instrument:
    share_a = A's trades that B also took (same direction, entries within one bar of the slower timeframe)
    overlap = min(share_a, share_b): both have to mostly trade together
A pair is a duplicate when overlap >= 60% with at least 10 trades each. Strategy level: how many of a
strategy's combos have a duplicate in ANOTHER strategy, and which strategy that mostly is.

Input: every reports/*/combo_trades_*.csv.gz (the newest file wins for a combo that appears twice).
Output: reports/duplicates/pairs.csv (all pairs >= 30%), reports/duplicates/by_strategy.csv, printed summary.
    python dup_check.py [--only SE,IM,MO]   (only report strategies whose id starts with these)
"""
from __future__ import annotations

import argparse
import collections
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WINDOW = pd.Timedelta(hours=2)
DUP, MIN_TRADES = 0.60, 10


def load():
    parts = []
    for f in sorted(glob.glob(str(ROOT / "reports" / "*" / "combo_trades_*.csv.gz"))):
        try:
            d = pd.read_csv(f, usecols=["sid", "tf", "symbol", "time", "dir"])
        except Exception:  # noqa: BLE001
            continue
        d["src"] = f
        parts.append(d)
    t = pd.concat(parts, ignore_index=True)
    t["combo"] = t["sid"] + "|" + t["tf"] + "|" + t["symbol"]
    newest = t.groupby("combo")["src"].max()                    # report folders are dated: last one wins
    t = t[t["src"] == t["combo"].map(newest)].drop_duplicates(["combo", "time", "dir"])
    t["time"] = pd.to_datetime(t["time"], utc=True)
    return t


BAR = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "4Hour": 240}


def pairs(t):
    """overlap = min(share of A's trades that B also took, share of B's trades that A also took): both
    must mostly trade together (a busy 5-minute combo near every trade of a slow one is not a duplicate).
    Two trades match: same instrument and direction, entries within one bar of the slower timeframe."""
    n = t.groupby("combo").size()
    shared = collections.Counter()
    for sym, g in t.groupby("symbol"):
        g = g.sort_values("time")
        mins = ((g["time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy()
        combos, dirs = g["combo"].to_numpy(), g["dir"].to_numpy()
        bar = np.array([BAR.get(c.split("|")[1], 60) for c in combos])
        cells = collections.defaultdict(list)
        for k, (m, d) in enumerate(zip(mins, dirs)):
            cells[(m // 60, d)].append(k)
        for k, (m, d) in enumerate(zip(mins, dirs)):
            seen = set()
            for hc in range(m // 60 - 4, m // 60 + 5):
                for j in cells.get((hc, d), ()):
                    cj = combos[j]
                    if cj == combos[k] or cj in seen:
                        continue
                    if abs(mins[j] - m) <= max(bar[k], bar[j]):
                        seen.add(cj)
            for other in seen:
                shared[(combos[k], other)] += 1        # a trade of this combo that `other` also took
    rows = []
    for (a, b), s_ab in shared.items():
        if a < b:
            s_ba = shared.get((b, a), 0)
            fa, fb = min(1.0, s_ab / n[a]), min(1.0, s_ba / n[b])
            ov = min(fa, fb)
            if ov >= 0.3:
                rows.append({"a": a, "b": b, "trades_a": int(n[a]), "trades_b": int(n[b]),
                             "share_a": round(fa, 3), "share_b": round(fb, 3), "overlap": round(ov, 3)})
    return pd.DataFrame(rows, columns=["a", "b", "trades_a", "trades_b", "share_a", "share_b", "overlap"]), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="strategy id prefixes to report on, e.g. SE,IM,MO,EV,GP,HB")
    args = ap.parse_args()
    t = load()
    pr, n = pairs(t)
    out = ROOT / "reports" / "duplicates"
    out.mkdir(parents=True, exist_ok=True)
    pr.sort_values("overlap", ascending=False).to_csv(out / "pairs.csv", index=False)
    dup = pr[(pr["overlap"] >= DUP) & (pr["trades_a"] >= MIN_TRADES) & (pr["trades_b"] >= MIN_TRADES)].copy()
    dup["sa"], dup["sb"] = dup["a"].str.split("|").str[0], dup["b"].str.split("|").str[0]
    dup = dup[dup["sa"] != dup["sb"]]                  # the same strategy on two timeframes is not a duplicate
    rows = []
    sids = sorted({c.split("|")[0] for c in n.index})
    prefixes = tuple(x for x in args.only.split(",") if x)
    for sid in sids:
        if prefixes and not sid.startswith(prefixes):
            continue
        mine = [c for c in n.index if c.startswith(sid + "|") and n[c] >= MIN_TRADES]
        hits = pd.concat([dup[dup["sa"] == sid].rename(columns={"sb": "other"}), dup[dup["sb"] == sid].rename(columns={"sa": "other"})])
        combos_dup = set(hits["a"]).union(hits["b"]) & set(mine)
        top = hits["other"].value_counts()
        rows.append({"strategy": sid, "combos": len(mine), "combos_with_duplicate": len(combos_dup),
                     "share_pct": round(100 * len(combos_dup) / len(mine), 1) if mine else 0.0,
                     "mostly_like": ", ".join(f"{k} ({v})" for k, v in top.head(3).items()),
                     "max_overlap": round(float(hits["overlap"].max()), 2) if len(hits) else 0.0})
    bs = pd.DataFrame(rows).sort_values("share_pct", ascending=False)
    bs.to_csv(out / "by_strategy.csv", index=False)
    print(f"combos: {len(n)} | trades: {len(t)} | pairs >= 30%: {len(pr)} | duplicate pairs (>= {DUP:.0%}, different strategies): {len(dup)}")
    print(bs.to_string(index=False))


if __name__ == "__main__":
    main()
