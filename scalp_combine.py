"""Reads scalp_opt.py results: each strategy's best setting (chosen on TRAIN only), how it did on VALID, and
the combination of the strategies.

  python scalp_combine.py --opt 'reports/*_scalp_run*/scalp_opt_*.json' --daily '<dir>/**/scalp_daily_*.csv.gz'
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

VALID_FROM = "2025-04-30"


def load_rows(pat):
    rows = []
    for f in glob.glob(pat, recursive=True):
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            for p in ("train", "valid"):
                s = r[p]
                for k in ("n", "sum", "avg", "win", "hold_min", "sd"):
                    r[f"{p}_{k}"] = s.get(k, 0 if k in ("n", "sum") else np.nan)
            rows.append({k: v for k, v in r.items() if k not in ("train", "valid")})
    return pd.DataFrame(rows).drop_duplicates(["sid", "symbol", "variant"])


def tstat(avg, sd, n):
    return avg / sd * np.sqrt(n) if sd and n > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", required=True)
    ap.add_argument("--daily", default="")
    ap.add_argument("--out", default="reports/scalp_summary.json")
    a = ap.parse_args()
    d = load_rows(a.opt)
    print("rows", len(d), "symbols", d.symbol.nunique())
    out = {"symbols": int(d.symbol.nunique()), "strategies": {}}

    # 1. one setting per strategy for all instruments (pooled): chosen on train by average R per trade
    for sid, g in d.groupby("sid"):
        p = g.groupby("variant").agg(tn=("train_n", "sum"), tsum=("train_sum", "sum"), vn=("valid_n", "sum"),
                                     vsum=("valid_sum", "sum"), hold=("valid_hold_min", "median"))
        p["tavg"], p["vavg"] = p.tsum / p.tn, p.vsum / p.vn
        base = p.loc["t1.5|s1|b30"] if "t1.5|s1|b30" in p.index else None
        best = p.sort_values("tavg", ascending=False)
        top = best.iloc[0]
        # 2. per instrument: best setting chosen on train (>= 30 train trades), judged on valid
        q = g[g.train_n >= 30].copy()
        q["t_train"] = [tstat(x, s, n) for x, s, n in zip(q.train_avg, q.train_sd, q.train_n)]
        pick = q.sort_values("train_avg", ascending=False).groupby("symbol").head(1)
        pick["t_valid"] = [tstat(x, s, n) for x, s, n in zip(pick.valid_avg, pick.valid_sd, pick.valid_n)]
        strong = pick[(pick.train_avg > 0) & (pick.t_train >= 2)]
        out["strategies"][sid] = {
            "pooled_best": {"variant": best.index[0], "train_avg": round(top.tavg, 3), "valid_avg": round(top.vavg, 3),
                            "valid_trades": int(top.vn), "hold_min": top.hold},
            "pooled_top5": [{"variant": k, "train_avg": round(r.tavg, 3), "valid_avg": round(r.vavg, 3)} for k, r in best.head(5).iterrows()],
            "pooled_original": None if base is None else {"train_avg": round(base.tavg, 3), "valid_avg": round(base.vavg, 3)},
            "pooled_best_valid_only": {"variant": p.vavg.idxmax(), "valid_avg": round(p.vavg.max(), 3)},
            "grid_share_positive_train": round(float((p.tavg > 0).mean()), 3),
            "per_symbol": {"chosen": int(len(pick)), "train_positive": int((pick.train_avg > 0).sum()),
                           "train_t2": int(len(strong)), "valid_positive_of_train_t2": int((strong.valid_avg > 0).sum()),
                           "valid_positive_all": int((pick.valid_avg > 0).sum())},
            "best_symbols": pick.sort_values("valid_avg", ascending=False).head(8)[
                ["symbol", "variant", "train_n", "train_avg", "t_train", "valid_n", "valid_avg", "t_valid", "valid_hold_min"]
            ].round(3).to_dict("records"),
            "strong_symbols": strong[["symbol", "variant", "train_n", "train_avg", "t_train", "valid_n", "valid_avg", "t_valid"]
                                     ].round(3).to_dict("records"),
        }
        # effect of each knob, averaged over the others (pooled)
        for knob in ("target", "stop", "bars"):
            e = g.groupby(knob).agg(tn=("train_n", "sum"), ts=("train_sum", "sum"), vn=("valid_n", "sum"), vs=("valid_sum", "sum"))
            out["strategies"][sid][f"by_{knob}"] = {str(k): {"train_avg": round(r.ts / r.tn, 3), "valid_avg": round(r.vs / r.vn, 3)}
                                                   for k, r in e.iterrows()}

    # 3. combination: each strategy's train-chosen instruments (train t >= 2), daily R on valid
    if a.daily:
        parts = [pd.read_csv(f) for f in glob.glob(a.daily, recursive=True)]
        if parts:
            dd = pd.concat(parts, ignore_index=True)
            dd = dd[dd.day >= VALID_FROM]
            series = {}
            for sid, s in out["strategies"].items():
                keys = {(x["symbol"], x["variant"]) for x in s["strong_symbols"]}
                if not keys:
                    continue
                m = dd[(dd.sid == sid) & [(a_, b_) in keys for a_, b_ in zip(dd.symbol, dd.variant)]]
                series[sid] = m.groupby("day")["r"].sum() / max(1, len(keys))
            if series:
                df = pd.DataFrame(series).fillna(0.0).sort_index()
                comb = df.mean(axis=1)
                def summ(x):
                    cum = x.cumsum()
                    return {"days": int(len(x)), "sum_r": round(float(x.sum()), 2), "sharpe_daily": round(float(x.mean() / x.std() * np.sqrt(252)), 2) if x.std() > 0 else None,
                            "max_dd_r": round(float((cum.cummax() - cum).max()), 2)}
                out["combination"] = {"each": {k: summ(df[k]) for k in df}, "combined": summ(comb),
                                      "correlation": df.corr().round(2).to_dict()}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
