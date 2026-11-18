"""Bot 5's gap method - the stop beyond the extreme of the whole move (user, 18.11.2026: "usually beyond the extreme of
that move, even if it is above or below the impulse - even on the higher timeframe"; his 96 marked stops were on
average 2.35x farther than the impulse's start).

STOPAT "move" (agents/tal_gap._origins): back from the impulse on the high timeframe while the price stayed on the far
side of the impulse's end - the move's extreme over that span (or the impulse's start, whichever is farther).
  marks   how close "move" comes to the stops Tal placed (live/watch_own_state.json of the live branch, in high
          timeframe ATR) - the rule against his hand, on the full history (the app's charts stop at 200 candles)
  run     every gap track as it runs (now = imp0 + 4x spread) against move with room none / 4x spread / 0.1 ATR, and
          mover (MIN_RR on the rules' stop): FTMO's costs, chosen on 2025, shown on 2026 (tal_mgmt_study's runner)
  python -m agents.tal_move_study marks --state w5.json
  python -m agents.tal_move_study run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from agents import tal_gap as TG
from agents import tal_mgmt_study as M

M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_move"


def ways():
    return {"now|base": {}, "move|c4": {"STOPAT": "move", "IPAD": "c4"}, "move|none": {"STOPAT": "move", "IPAD": "none"},
            "move|a01": {"STOPAT": "move", "IPAD": "a0.1"}, "mover|c4": {"STOPAT": "mover", "IPAD": "c4"},
            "imp0|none": {"STOPAT": "imp0", "IPAD": "none"}, "sw2|base": {"STOPAT": "sw2", "IPAD": "none"},
            "now|walk": {"MGW": 1}}


M.ways = ways


def move_extreme(D, htf, d, c1_iso, imp1_px, imp0_px):
    ts, o, h, l, c, atr = D.tf(htf)
    k = int(np.searchsorted(ts, pd.Timestamp(c1_iso).value)) + 1          # candle 2
    if k < 3 or k >= len(c):
        return None, None
    j, lim = k, max(0, k - 3000)
    while j > lim and ((l[j - 1] > imp1_px) if d < 0 else (h[j - 1] < imp1_px)):
        j -= 1
    mv = float(h[j:k + 1].max()) if d < 0 else float(l[j:k + 1].min())
    mv = max(mv, imp0_px) if d < 0 else min(mv, imp0_px)
    a = float(atr[k - 2]) if np.isfinite(atr[k - 2]) else None
    return mv, a


def marks(state):
    import agent_bot5 as A5
    s = json.loads(open(state, encoding="utf-8").read())
    rows, cache = [], {}
    for t in s.get("watch") or []:
        m = t.get("mark") or {}
        g = t.get("geo") or {}
        if not t.get("tal") or str(t["tal"]).startswith("sc:") or m.get("stop") in (None, "") or not g.get("imp1") or not g.get("imp0"):
            continue
        sym = t["symbol"]
        if sym not in cache:
            try:
                cache[sym] = A5.tal_load(sym)
            except Exception as e:  # noqa: BLE001
                print(sym, e)
                cache[sym] = None
        D = cache[sym]
        if D is None:
            continue
        d, ts_, e = int(t["dir"]), float(m["stop"]), float(t["entry"])
        mv, a = move_extreme(D, g["htf"], d, g["c1"], float(g["imp1"][1]), float(g["imp0"][1]))
        if mv is None or not a:
            continue
        imp = float(g["imp0"][1])
        rows.append({"id": t["id"], "htf": g["htf"], "tal_atr": d * (e - ts_) / a, "move_atr": d * (e - mv) / a,
                     "imp0_atr": d * (e - imp) / a, "err_move": -d * (ts_ - mv) / a, "err_imp0": -d * (ts_ - imp) / a})
    df = pd.DataFrame(rows)
    out = {"n": int(len(df))}
    for k in ("err_move", "err_imp0"):
        x = df[k].to_numpy()
        out[k] = {"median_atr": round(float(np.median(x)), 2), "within_0.5": round(100 * float(np.mean(abs(x) <= .5)), 1),
                  "within_1": round(100 * float(np.mean(abs(x) <= 1)), 1), "tal_farther": round(100 * float(np.mean(x > .5)), 1),
                  "tal_closer": round(100 * float(np.mean(x < -.5)), 1)}
    out["by_htf"] = {h: {"n": int(len(g)), "err_move_median": round(float(g["err_move"].median()), 2),
                         "within_1": round(100 * float((g["err_move"].abs() <= 1).mean()), 1)} for h, g in df.groupby("htf")}
    out["rows"] = rows
    (M.OUT).mkdir(parents=True, exist_ok=True)
    (M.OUT / "marks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report", "marks"])
    ap.add_argument("--sym", default="")
    ap.add_argument("--state", default="w5.json")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(M.plan()))
    elif a.mode == "run":
        M.run(a.sym)
    elif a.mode == "marks":
        marks(a.state)
    else:
        M.report()
