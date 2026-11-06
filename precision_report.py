"""Summary of precision_entry.py: does a 1-minute limit entry beat the bots' market entry?

Total R per period (a trade that is not filled counts 0), all live combos together and per combo, in the three
periods of improve.py: choose 2025-04-30..2025-12-31, confirm 2026-01-01..06-30, test 2026-07-01 on. A
per-combo choice is made in "choose" only and kept only if it also wins in "confirm"; "test" is the honest gain.

  python precision_report.py --res 'reports/*_precision_run*/precision_*.csv.gz'
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

P1, P2 = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--trades", default="reports/2026-11-05_tv_run674/combo_trades_all.csv.gz")
    ap.add_argument("--out", default="reports/precision_summary.json")
    a = ap.parse_args()
    T = pd.read_csv(a.trades)
    R = pd.concat([pd.read_csv(f) for f in glob.glob(a.res)], ignore_index=True).drop_duplicates("idx").set_index("idx")
    t = T.join(R, how="inner")
    t = t[t.covered.astype(bool)]
    tm = pd.to_datetime(t.time, utc=True)
    t["per"] = np.where(tm < P1, "choose", np.where(tm < P2, "confirm", "test"))
    t["combo"] = t.sid + "|" + t.tf + "|" + t.symbol
    vs = [c for c in R.columns if c.startswith("k")]
    out = {"trades": int(len(t)), "combos": int(t.combo.nunique()), "pooled": {}}
    base = t.groupby("per")["r"].sum()
    out["pooled"]["market (now)"] = {p: {"sum_r": round(float(base[p]), 1), "trades": int((t.per == p).sum())} for p in base.index}
    for v in vs:
        g = t.groupby("per")
        out["pooled"][v] = {p: {"sum_r": round(float(x[v].fillna(0).sum()), 1), "filled": round(float(x[v].notna().mean()), 3),
                                "avg_filled": round(float(x[v].mean()), 3), "base_avg_same": round(float(x.loc[x[v].notna(), "r"].mean()), 3)}
                            for p, x in g}
    # per combo: choose the best variant in "choose" (must beat market by >= 2 R and 15%), keep if it wins in "confirm"
    rows, gain = [], {"choose": 0.0, "confirm": 0.0, "test": 0.0}
    for c, g in t.groupby("combo"):
        s = {p: {"market": float(x["r"].sum()), **{v: float(x[v].fillna(0).sum()) for v in vs}} for p, x in g.groupby("per")}
        if "choose" not in s:
            continue
        ch = s["choose"]
        best = max(vs, key=lambda v: ch[v])
        ok = ch[best] - ch["market"] >= max(2.0, 0.15 * abs(ch["market"]))
        cf = s.get("confirm")
        kept = bool(ok and cf is not None and cf[best] > cf["market"])
        row = {"combo": c, "best": best, "kept": kept}
        for p in ("choose", "confirm", "test"):
            if p in s:
                row[p] = {"market": round(s[p]["market"], 2), "best": round(s[p][best], 2)}
                if kept:
                    gain[p] += s[p][best] - s[p]["market"]
        rows.append(row)
    out["per_combo"] = {"kept": sum(r["kept"] for r in rows), "of": len(rows), "gain_r_if_kept": {k: round(v, 1) for k, v in gain.items()},
                        "kept_list": [r for r in rows if r["kept"]]}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("trades", "combos")}, indent=0))
    for v, x in out["pooled"].items():
        print(v, {p: (y["sum_r"], y.get("filled")) for p, y in x.items()})
    print("per combo:", {k: out["per_combo"][k] for k in ("kept", "of", "gain_r_if_kept")})


if __name__ == "__main__":
    main()
