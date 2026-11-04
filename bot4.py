"""Should there be a 4th bot? Built only from combos that passed the first test and are in no bot.

Pool trades: the pool's combos re-run with the live rules (FTMO Standard: closed before breaks, news
rule, stocks closed at the session end). The account replay is the bots' own (loo.replay: graded risk,
4.5% / 9% guards).

1. choose   2025-04-30 .. 2025-12-31: greedy - add, one at a time, the combo that improves the
            account most (profit minus twice the deepest drop), while anything still improves it
2. check    2026-01-01 .. end, a fresh account: the chosen set, against the whole pool and against
            300 random sets of the same size (did the choosing help, or would any set do as well?)
3. overlap  correlation of the chosen set's daily results with each live bot's (a 4th bot is only
            worth it if it does not lose on the same days as the others)
Two versions: every instrument (an FTMO account) and only what the MetaQuotes demo trades (no indices,
energy, crypto or European shares).

  python bot4.py --pool 'reports/2026-11-04_tv_run6[12]?/combo_trades_*.csv.gz' --bots reports/X/news_trades.csv.gz
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import loo as LO
from agents import ftmo_universe as U

ROOT = Path(__file__).resolve().parent
H1, H2 = ("2025-04-30", "2026-01-01"), ("2026-01-01", "2100-01-01")
NOT_ON_METAQUOTES = {"index", "energy", "crypto"}


def load(pattern):
    t = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)], ignore_index=True)
    t["combo"] = t["sid"] + "|" + t["tf"] + "|" + t["symbol"]
    for c in ("time", "exit_time", "free_time"):
        t[c] = pd.to_datetime(t[c], utc=True, errors="coerce")
    return t


def half(t, h):
    return t[(t["time"] >= pd.Timestamp(h[0], tz="UTC")) & (t["time"] < pd.Timestamp(h[1], tz="UTC"))]


def score(t, combos, risk):
    x = t[t["combo"].isin(combos)].sort_values("time").reset_index(drop=True)
    r = LO.replay(x, risk)
    return r["end"] - 2 * (LO.F.INITIAL - r["low"]), r


def greedy(t, pool, risk, max_n=25):
    chosen, best = [], score(t, [], risk)[0]
    while len(chosen) < max_n:
        cand = [(score(t, chosen + [c], risk)[0], c) for c in pool if c not in chosen]
        if not cand:
            break
        s, c = max(cand)
        if s <= best:
            break
        chosen.append(c)
        best = s
    return chosen


def daily(t, combos, risk):
    x = t[t["combo"].isin(combos)]
    d = x.groupby(x["exit_time"].dt.tz_convert("Europe/Prague").dt.date)["r"].sum() * risk * LO.F.INITIAL
    return d


def run(pool_t, bots_t, pool, risk, label, rng):
    a = half(pool_t, H1)
    b = half(pool_t, H2)
    chosen = greedy(a, pool, risk)
    _, r_a = score(a, chosen, risk)
    _, r_b = score(b, chosen, risk)
    _, r_all = score(b, pool, risk)
    rand = []
    for _ in range(300):
        s = list(rng.choice(pool, size=min(len(chosen), len(pool)), replace=False))
        rand.append(score(b, s, risk)[1]["end"])
    rand = np.array(rand)
    corr = {}
    mine = daily(b, chosen, risk)
    for bn in (1, 2, 3):
        cfg = json.load(open(ROOT / "live" / "bots" / f"bot{bn}.json", encoding="utf-8"))
        other = daily(half(bots_t, H2), cfg["combos"], cfg["risk"])
        j = pd.concat([mine, other], axis=1).fillna(0)
        corr[f"bot{bn}"] = round(float(j.corr().iloc[0, 1]), 2) if len(j) > 10 else None
    out = {"version": label, "risk": risk, "pool": len(pool), "chosen": chosen,
           "choose_half": r_a, "check_half": r_b, "check_half_whole_pool": r_all,
           "random_sets_check_end": {"p10": int(np.percentile(rand, 10)), "median": int(np.median(rand)),
                                     "p90": int(np.percentile(rand, 90))},
           "chosen_beats_random_pct": round(100 * float((rand < r_b["end"]).mean()), 1),
           "daily_corr_with_live_bots": corr}
    print(json.dumps(out, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--bots", required=True, help="the live bots' trades (bots_compare news_trades.csv.gz)")
    ap.add_argument("--risk", default="0.0025,0.00375,0.005")
    args = ap.parse_args()
    LO.F.FREE_RISK_AT_BREAKEVEN = True
    pool_t, bots_t = load(args.pool), pd.read_csv(args.bots)
    for c in ("time", "exit_time", "free_time"):
        bots_t[c] = pd.to_datetime(bots_t[c], utc=True, errors="coerce")
    pool = sorted(pool_t["combo"].unique())
    mq = [c for c in pool if U.CLASS.get(c.split("|")[2]) not in NOT_ON_METAQUOTES
          and not (U.CLASS.get(c.split("|")[2]) == "stock" and "USD" not in U.CURRENCIES.get(c.split("|")[2], []))]
    rng = np.random.default_rng(5)
    res = []
    for risk in (float(x) for x in args.risk.split(",")):
        res.append(run(pool_t, bots_t, pool, risk, "FTMO (all instruments)", rng))
        res.append(run(pool_t, bots_t, mq, risk, "MetaQuotes demo (FX, metals, US shares)", rng))
    p = ROOT / "reports" / "bot4" / "result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
