"""Bot 5's gap method after 19.11.2026 (user, 19-20.11.2026: 13 longs on US500 / US100 / JP225 / US2000 against a falling
afternoon, -$3,866, while the morning's shorts were right; "check yesterday's trades again and draw conclusions"):
  1. the big picture's direction (TREND) on every gap track: as it runs, none, and the modes tal_gap has - the day's open,
     the timeframe above's EMA 20, EMA 9/20 of the high / entry timeframe, EMA 50/200, New York's cash open, premium /
     discount, the daily pivot
  2. a cap on trades open together in the same direction across the indices (and across everything) - from the
     trades as they run (rows with dir and minutes), in time order over all instruments and tracks
FTMO's costs, chosen on 30.04-31.12.2025, shown on 2026 (agents/tal_mgmt_study's runner). Results only.
  python -m agents.tal_dirtrend_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from agents import tal_gap as TG
from agents import tal_mgmt_study as M

M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_dirtrend"
MODES = ("none", "day", "up20", "e920", "e920l", "e50200", "rth", "pd20", "piv")
CAPS = (1, 2, 3, 4, 6, 8)
FX_METAL = lambda s: (len(s) == 6 and s.isalpha()) or s.startswith("XA")


def ways():
    return {"now|base": {}, **{f"tr|{m}": {"TREND": m} for m in MODES}}


M.ways = ways


def run(sym):
    """M.run with each trade's direction and minutes kept too (the cap needs them)."""
    import agent_bot5 as A
    M.OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out = {}
    t_end = time.time() + M.BUDGET_MIN * 60
    for tk, tv in M.W.tracks().items():
        if sym not in M.W.syms_of(tk):
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        base = TG._norm(dict(tv["cfg"]))
        gc = {}
        for w, extra in ways().items():
            if time.time() > t_end:
                out.setdefault(tk, {})[w] = {"error": "time budget"}
                continue
            try:
                t = TG.simulate(D, pair, TG._norm(dict(base, **extra)), gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(tk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            out.setdefault(tk, {})[w] = [{"time": str(a), "r": float(b), "why": str(y), "dir": int(d), "mins": int(m)}
                                         for a, b, y, d, m in zip(t["time"], t["r"], t["why"], t["dir"], t["mins"])]
        print(sym, tk, {w: len(x) if isinstance(x, list) else x for w, x in out[tk].items()}, flush=True)
        (M.OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")
    (M.OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")


def caps():
    """The trades as they run (now|base), every instrument and track in time order: a trade is skipped while `cap`
    trades in its direction are open in its group (indices / everything)."""
    rows = []
    for f in sorted(M.OUT.glob("*.json")):
        if f.name in ("report.json", "marks.json"):
            continue
        sym = f.stem
        for tk, wz in json.loads(f.read_text(encoding="utf-8")).items():
            for x in wz.get("now|base") or []:
                if isinstance(x, dict) and x.get("dir") is not None:
                    t = pd.Timestamp(x["time"])
                    if t >= M.FROM:
                        rows.append((t, t + pd.Timedelta(minutes=int(x["mins"]) + 1), int(x["dir"]), float(x["r"]), sym))
    rows.sort(key=lambda r: r[0])
    out = {}
    for grp in ("indices", "all"):
        for cap in (None,) + CAPS:
            opn, res = [], {"train": [], "unseen": []}
            for t0, t1, d, r, sym in rows:
                opn = [o for o in opn if o[0] > t0]
                ing = grp == "all" or not FX_METAL(sym)
                if cap is not None and ing and sum(1 for o in opn if o[1] == d and (grp == "all" or not FX_METAL(o[2]))) >= cap:
                    continue
                opn.append((t1, d, sym))
                res["unseen" if t0 >= TG.SPLIT else "train"].append(r)
            out[f"{grp}|{cap or 'none'}"] = {k: M._st(v) for k, v in res.items()}
    return out


def report():
    M.report()
    rep = json.loads((M.OUT / "report.json").read_text(encoding="utf-8"))
    rep["caps"] = caps()
    (M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in rep["caps"].items():
        print(f"cap {k:14} 2025 {v['train']['sum']:8} ({v['train']['n']}, dd {v['train']['dd']})  "
              f"2026 {v['unseen']['sum']:8} ({v['unseen']['n']}, dd {v['unseen']['dd']})")


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
        report()
