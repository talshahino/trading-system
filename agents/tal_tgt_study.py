"""Bot 5's gap method - which swing is the best target (user, 20.11.2026 06:44: "where is it most right to exit at the
target - the first swing? the second? third? fourth? fifth? sixth?"). Now: the 2nd swing beyond the entry, known at the
fill. TSWN 1..6 = the 1st..6th (a setup whose last 400 candles hold fewer swings is not taken - counted in n). The stop
as it runs (beyond the move + room), MIN_RR as live, FTMO's costs, every FTMO forex pair, metal and cash index that fits
(agents/tal_dirtrend_study's universe and runner; "live" = what bot 5 runs now). Chosen on 30.04-31.12.2025, shown on
2026. Results only - nothing in the bot changes.
  python -m agents.tal_tgt_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from agents import tal_dirtrend_study as S

S.M.OUT = S.M.ROOT / "reports" / "agents" / "bot5" / "tal_tgt"
WAYS = {"now|base": {}, **{f"sw|{n}": {"TSWN": n} for n in (1, 3, 4, 5, 6)}}
S.ways = lambda: dict(WAYS)


def report():
    tr, rows = S.trends()
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "trend": tr, "rows": rows}
    (S.M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for sc in ("live", "all"):
        print("==", sc, tr[sc]["n"], "pick", tr[sc]["pick"])
        for w, x in sorted(tr[sc]["ways"].items(), key=lambda kv: -kv[1]["train"]):
            print(f"  {w:12} 2025 {x['train']:9} 2026 {x['unseen']:9} ({x['n_unseen']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(S.universe()))
    elif a.mode == "run":
        S.run(a.sym)
    else:
        report()
