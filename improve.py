"""Which trade-management changes really improve the bots' strategies? (target / stop distance / entry)

Input: the variant-grid runs (run_rules.py --combos-file ... --optimize, every variant's trades kept):
for every combo of the live bots, 6 targets x 4 stop distances x 3 entries = 72 variants.

Three periods, so the result can be trusted:
    choose   2025-04-30 .. 2025-12-31   each combo's best variant by total R (the "recommendation")
    confirm  2026-01-01 .. 2026-06-30   kept only if it also beats the current way here
    test     2026-07-01 .. end          never looked at while choosing: the honest gain
A recommendation also needs enough trades (>= 15 and >= half of the current way's) and a real margin in
the choosing period (>= 2 R and >= 15% of the current total).

Output (reports/improve/):
    recommendations.json   every combo: current vs recommended, in all three periods, kept or not and why
    trades_current.csv.gz  the bots' trades the current way (one variant per combo)
    trades_improved.csv.gz the same with the kept recommendations
The two trade files go to bots_compare.py for the whole-bot check.

  python improve.py --trades 'reports/2026-11-04_tv_run5[23]?/combo_trades_*.csv.gz'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "improve"
P0, P1, P2 = pd.Timestamp("2025-04-30", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
BASE = "own|be0.6|stop1|entry0"
MIN_N, MIN_GAIN_R, MIN_GAIN_SHARE = 15, 2.0, 0.15


def explain(v):
    tg, _, st, en = v.split("|")
    parts = []
    if tg != "own":
        parts.append("בלי יעד" if tg == "none" else f"יעד {tg}")
    if st != "stop1":
        parts.append(f"סטופ ×{st[4:]}")
    if en != "entry0":
        parts.append(f"כניסה בפקודה ממתינה {en[5:]}R טוב יותר")
    return ", ".join(parts) or "כמו היום"


def period_stats(t):
    out = {}
    for name, a, b in (("choose", P0, P1), ("confirm", P1, P2), ("test", P2, None)):
        x = t[(t["time"] >= a) & ((t["time"] < b) if b is not None else True)]["r"]
        out[name] = {"n": int(len(x)), "sum": round(float(x.sum()), 2), "avg": round(float(x.mean()), 3) if len(x) else 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="glob of the variant-grid combo_trades files")
    args = ap.parse_args()
    files = sorted(set(sum((glob.glob(g) for g in args.trades.split(",")), [])))
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    t = t[t["variant"].notna()].copy()
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t["combo"] = t["sid"] + "|" + t["tf"] + "|" + t["symbol"]
    t = t.drop_duplicates(["combo", "variant", "time", "dir"])
    recs, cur_parts, imp_parts = [], [], []
    tot = {"naive_test_gain": 0.0, "kept_test_gain": 0.0, "kept_confirm_gain": 0.0}
    for combo, g in t.groupby("combo"):
        by = {v: x for v, x in g.groupby("variant")}
        if BASE not in by:
            continue
        st = {v: period_stats(x) for v, x in by.items()}
        b = st[BASE]
        ok = [v for v in st if st[v]["choose"]["n"] >= max(MIN_N, 0.5 * b["choose"]["n"])]
        best = max(ok or [BASE], key=lambda v: st[v]["choose"]["sum"])
        s = st[best]
        gain = {p: round(s[p]["sum"] - b[p]["sum"], 2) for p in ("choose", "confirm", "test")}
        margin = gain["choose"] >= max(MIN_GAIN_R, MIN_GAIN_SHARE * abs(b["choose"]["sum"]))
        kept = best != BASE and margin and gain["confirm"] > 0
        why = ("כמו היום" if best == BASE else "שיפור קטן מדי בתקופת הבחירה" if not margin
               else "לא שיפר בתקופת האימות" if not kept else "עבר בחירה ואימות")
        if best != BASE:
            tot["naive_test_gain"] += gain["test"]
        if kept:
            tot["kept_test_gain"] += gain["test"]
            tot["kept_confirm_gain"] += gain["confirm"]
        recs.append({"combo": combo, "current": BASE, "recommended": best, "change_he": explain(best), "kept": kept,
                     "why_he": why, "gain_r": gain, "current_stats": b, "recommended_stats": s})
        cur_parts.append(by[BASE])
        imp_parts.append(by[best] if kept else by[BASE])
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["sid", "tf", "symbol", "time", "exit_time", "dir", "r", "entry_px", "stop_px", "exit_px", "target_px", "reason", "free_time"]
    pd.concat(cur_parts)[cols].to_csv(OUT / "trades_current.csv.gz", index=False)
    pd.concat(imp_parts)[cols].to_csv(OUT / "trades_improved.csv.gz", index=False)
    kept = [r for r in recs if r["kept"]]
    summary = {"combos": len(recs), "changed_by_choice": sum(r["recommended"] != BASE for r in recs), "kept": len(kept),
               **{k: round(v, 1) for k, v in tot.items()},
               "kept_test_better": sum(r["gain_r"]["test"] > 0 for r in kept),
               "kept_test_worse": sum(r["gain_r"]["test"] < 0 for r in kept),
               "by_change": pd.Series([r["change_he"] for r in kept]).value_counts().to_dict() if kept else {}}
    (OUT / "recommendations.json").write_text(json.dumps({"summary": summary, "periods": {
        "choose": [str(P0.date()), str(P1.date())], "confirm": [str(P1.date()), str(P2.date())], "test": [str(P2.date()), "end"]},
        "combos": recs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in sorted(kept, key=lambda r: -r["gain_r"]["test"]):
        print(f"{r['combo']:28} {r['change_he']:45} choose {r['gain_r']['choose']:+6.1f} confirm {r['gain_r']['confirm']:+6.1f} test {r['gain_r']['test']:+6.1f}")


if __name__ == "__main__":
    main()
