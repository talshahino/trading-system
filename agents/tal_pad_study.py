"""Bot 5's gap method - a little more room beyond the stop (user, 19.11.2026 17:46: 4-5 trades stopped out by a wick and
came back toward the target - "maybe it went to profit if we widened the stop a few more pips"). The stop stays beyond
the whole move (STOPAT move); the room beyond it (IPAD, now c4 = 4x spread+commission) widened: more multiples of the
costs, a share of the high timeframe's ATR, a share of the price. Same $ risk per trade (a wider stop = a smaller size),
MIN_RR as live. Every track as it runs, FTMO's costs, chosen on 2025, shown on 2026 (agents/tal_mgmt_study's runner).
Results only - nothing in the bot changes.
  python -m agents.tal_pad_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

from agents import tal_mgmt_study as M

M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_pad"
PADS = ("c6", "c8", "c12", "c20", "a0.05", "a0.1", "a0.2", "a0.3", "p0.03", "p0.06", "p0.1")


def ways():
    return {"now|base": {}, "now|walk": {"MGW": 1}, **{f"pad|{p}": {"IPAD": p} for p in PADS}}


M.ways = ways

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(M.plan()))
    elif a.mode == "run":
        M.run(a.sym)
    else:
        M.report()
