"""Releases as a filter for bot 5's gap tracks (user, 18.11.2026: bot 7's own trades after FTMO's 2 minutes showed no
edge - "check both": the releases as a filter for the other bots). Each gap track as it runs now (the stop beyond the
impulse's start + room, live/bots/bot5.json) with:
  PROT   none / cal15 / cal60 (no entry and the waiting order cancelled in a red release's window of the instrument's
         currencies) / shock (cancelled after a sharp minute against the gap) / cal15shock
  NEWSX  none / 15 / 60 (no entry +-X minutes of a red release and an open trade closed X minutes before it)
FTMO's costs, chosen on 30.04-31.12.2025, shown on 2026 (agents/tal_mgmt_study's runner and report). Results only.

  python -m agents.tal_news_filter plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

from agents import tal_mgmt_study as M

M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_newsf"


def ways():
    w = {}
    for p in ("now", "none", "cal15", "cal60", "shock", "cal15shock"):
        for x in ("base", "nx15", "nx60"):
            c = {} if p == "now" else {"PROT": p}
            if x != "base":
                c["NEWSX"] = int(x[2:])
            w[f"{p}|{x}"] = c
    w["now|walk"] = {"MGW": 1}
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
