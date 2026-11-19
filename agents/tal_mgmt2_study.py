"""Bot 5's gap method - managing a live trade, from outside research (user, 19.11.2026: "research online how to manage
a trade that is already live - less risk, more profit - and what fits our strategies"; partial exits cut the
expectancy, a break-even stop helps only with a far target, the stop to the entry should cover the costs, a time stop
and a volatility stop are the untested tools). Every gap track as it runs now (the stop beyond the move, the bot's own
target and exit), FTMO's costs, chosen on 2025, shown on 2026 (agents/tal_mgmt_study's runner; agents/tal_gap.mg_walk).
  TSTOP  N:m  after N low-timeframe candles with the best excursion under m R - out (N 10 / 20 / 40, m 0.3 / 0.5)
  ATRTR  k:a  from +a R the stop trails k x the low timeframe's ATR behind the best price (k 2 / 3 / 4, a 1)
  BECR   a    at +a R the stop to the entry + the costs (a 0.5 / 1 / 1.5)
  python -m agents.tal_mgmt2_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

from agents import tal_mgmt_study as M

M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_mgmt2"


def ways():
    w = {"now|base": {}, "now|walk": {"MGW": 1}}
    for n in (10, 20, 40):
        for m in (0.3, 0.5):
            w[f"time|{n}:{m}"] = {"TSTOP": f"{n}:{m}"}
    for k in (2, 3, 4):
        w[f"atr|{k}"] = {"ATRTR": f"{k}:1"}
    for a in (0.5, 1, 1.5):
        w[f"becr|{a}"] = {"BECR": a}
    w["combo|t20:0.5+becr1"] = {"TSTOP": "20:0.5", "BECR": 1}
    w["combo|t20:0.5+atr3"] = {"TSTOP": "20:0.5", "ATRTR": "3:1"}
    return w


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
