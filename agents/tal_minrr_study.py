"""Bot 5's gap method with the stop beyond the move and a lower minimum reward:risk (user, 19.11.2026: "who said it
has to be 1:2 or 1:1? I have no problem with a loss of 1 against a target of 0.5" - with the stop beyond the whole
move, MIN_RR 2 left EURUSD / USDJPY / USDCHF with no trade). Every track as it ran (MIN_RR as before) against MIN_RR
0.5 and 0, FTMO's costs, 2025 and 2026 apart (agents/tal_mgmt_study's runner). Results only.
  python -m agents.tal_minrr_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

from agents import tal_mgmt_study as M

M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_minrr"
BEFORE = {"t1h_15": 1.5, "t4h_1h": 0.0, "t4h_15": 0.0, "t4h_5": 0.5, "t1h_5@XAUUSD": 1.0, "t1h_15@NZDCHF": 1.0,
          "t1h_15@GBPNZD": 1.0}


def ways():
    return {"now|base": {}, "rr|0": {"MIN_RR": 0.0}, "rr|1": {"MIN_RR": 1.0}, "now|walk": {"MGW": 1}}


M.ways = ways
_run = M.run


def run(sym):
    """'before' = MIN_RR as before 19.11 (2 unless listed) - the live file now has 0.5 everywhere."""
    w0 = M.W.tracks

    def tracks():
        t = w0()
        for k, v in t.items():
            v = dict(v, cfg=dict(v["cfg"]))
            t[k] = v
        return t
    M.ways = lambda: {"now|base": {}, "before|base": {"MIN_RR": "__before__"}, "rr|0": {"MIN_RR": 0.0},
                      "rr|1": {"MIN_RR": 1.0}, "now|walk": {"MGW": 1}}
    import agents.tal_gap as TG
    n0 = TG._norm

    def norm(c):
        if c.get("MIN_RR") == "__before__":
            c = dict(c, MIN_RR=c.get("_before", 2.0))
        return n0(c)
    TG._norm = norm
    orig = M.W.tracks
    M.W.tracks = lambda: {k: dict(v, cfg=dict(v["cfg"], _before=BEFORE.get(k, 2.0))) for k, v in orig().items()}
    _run(sym)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(M.plan()))
    elif a.mode == "run":
        run(a.sym)
    else:
        M.ways = lambda: {"now|base": {}, "before|base": {}, "rr|0": {}, "rr|1": {}, "now|walk": {}}
        M.report()
