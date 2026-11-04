"""Does every strategy earn its place in its bot? (leave-one-out)

For each bot and each combo it trades: the bot's $100K account replay (agents/ftmo_agent.account_path,
the bot's risk, graded risk, 4.5% / 9% guards) with and without that combo, in two separate halves
(2025-04-30..2025-12-31 and 2026-01-01..end, a fresh account each). A combo is a removal candidate
only if the bot does better WITHOUT it in BOTH halves (more profit and no deeper low).

  python loo.py --trades reports/X/news_trades.csv.gz --combos 1=a.txt,2=b.txt,3=c.txt
Writes reports/loo/result.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agents import ftmo_agent as F

ROOT = Path(__file__).resolve().parent
HALVES = (("2025-04-30", "2026-01-01"), ("2026-01-01", "2100-01-01"))
F.MAX_OPEN, F.GRADED = 99, True
F.GUARD_DAILY, F.GUARD_TOTAL = 0.045, 0.09


def replay(tr, risk):
    if tr.empty:
        return {"end": F.INITIAL, "low": F.INITIAL, "blown": False}
    w = F.account_path(tr, risk, start=tr["time"].iloc[0])
    led = w[0].attrs.get("ledger")
    bal = led["balance_after"].dropna() if led is not None and "balance_after" in led else pd.Series(dtype=float)
    end = float(bal.iloc[-1]) if len(bal) else F.INITIAL
    return {"end": round(end), "low": round(float(min(F.INITIAL, bal.min()))) if len(bal) else F.INITIAL,
            "blown": w[1] is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--combos", default="")
    args = ap.parse_args()
    t = pd.read_csv(args.trades)
    for c in ("time", "exit_time", "free_time"):
        t[c] = pd.to_datetime(t[c], utc=True, errors="coerce")
    over = dict(x.split("=") for x in args.combos.split(",") if x)
    out = []
    for b in (1, 2, 3, 4):
        cfg = json.load(open(ROOT / "live" / "bots" / f"bot{b}.json", encoding="utf-8"))
        F.FREE_RISK_AT_BREAKEVEN = bool(cfg["free_risk_at_breakeven"])
        combos = cfg["combos"] if str(b) not in over else [l.strip() for l in open(over[str(b)], encoding="utf-8") if l.strip()]
        halves = [t[(t["time"] >= pd.Timestamp(a, tz="UTC")) & (t["time"] < pd.Timestamp(z, tz="UTC"))] for a, z in HALVES]
        full = [replay(h[h["combo"].isin(combos)].sort_values("time").reset_index(drop=True), cfg["risk"]) for h in halves]
        rows = []
        for c in combos:
            rest = [x for x in combos if x != c]
            wo = [replay(h[h["combo"].isin(rest)].sort_values("time").reset_index(drop=True), cfg["risk"]) for h in halves]
            own = [h.loc[h["combo"] == c, "r"] for h in halves]
            d_end = [wo[i]["end"] - full[i]["end"] for i in range(2)]
            d_low = [wo[i]["low"] - full[i]["low"] for i in range(2)]
            rows.append({"combo": c, "trades": [int(len(x)) for x in own], "sum_r": [round(float(x.sum()), 1) for x in own],
                         "without_end_change": d_end, "without_low_change": d_low,
                         "remove": all(x > 0 for x in d_end) and all(x >= 0 for x in d_low)})
            print(b, c, rows[-1]["sum_r"], d_end, d_low, "REMOVE" if rows[-1]["remove"] else "")
        out.append({"bot": b, "risk": cfg["risk"], "with_all": full, "combos": rows})
    p = ROOT / "reports" / "loo" / "result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
