"""Bot 5's gap method - where the stop is best (user, 20.11.2026 06:43: "where it is today, a little toward the target,
a little further away... against all the assets, about 7 different stop options"). Every FTMO forex pair, metal and
cash index that fits the method (agents/tal_dirtrend_study's universe and runner; "live" = what bot 5 runs now):
  now        beyond the whole move + 4x spread+commission (as it runs)
  k0.75/0.9  the stop's distance x 0.75 / 0.9 - closer to the entry (toward the target)
  nopad      beyond the move with no room
  k1.15/1.3/1.5  the distance x 1.15 / 1.3 / 1.5 - further away
  imp0       beyond the impulse's start + the same room (the stop before 19.11)
The same $ risk per trade (a wider stop = a smaller size), MIN_RR as live (checked on the stop in use), FTMO's costs,
chosen on 30.04-31.12.2025, shown on 2026. Results only - nothing in the bot changes.
  python -m agents.tal_stop7_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from agents import tal_dirtrend_study as S

S.M.OUT = S.M.ROOT / "reports" / "agents" / "bot5" / "tal_stop7"
WAYS = {"now|base": {}, "st|k0.75": {"SSCALE": 0.75}, "st|k0.9": {"SSCALE": 0.9}, "st|nopad": {"IPAD": "c0"},
        "st|k1.15": {"SSCALE": 1.15}, "st|k1.3": {"SSCALE": 1.3}, "st|k1.5": {"SSCALE": 1.5},
        "st|imp0": {"STOPAT": "imp0"}}
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
