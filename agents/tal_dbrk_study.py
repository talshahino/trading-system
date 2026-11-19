"""Bot 5's gap method over the indices' short daily break (bot 5's smart agent, 19.11.2026, task b5reopen1911).
18.11.2026: the 5m/1m gap of 19:10 UTC gave 7 index sells; US100 / JP225 stopped 2 minutes after the daily reopening
(no candles 21:49-23:05), US500 / GER40 / UK100 filled at the reopening's first minute. A break under 2 hours is not a
break for WKND, so orders wait over it and trades are held through it. DBRK (agents/tal_gap.dbrk_of): c = cancel a
waiting order 10 minutes before an expected 30+ minute break, oN = no fill in the first N minutes after one, x = close an open
trade 10 minutes before one - alone and together, on every gap track of bot 5 as it runs, FTMO's costs. Chosen on 30.04-31.12.2025,
shown on 2026, with the time weights, the largest fall from the peak and the worst day. Results only.
  python -m agents.tal_dbrk_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from agents import agentkit as K
from agents import tal_gap as TG
from agents import tal_mgmt_study as M

M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_dbrk"
DB = ["c", "x", "c+x", "o10", "x+o10", "c+o10+x"]


def ways():
    return {"now|base": {}, **{f"db|{v}": {"DBRK": v} for v in DB}}


M.ways = ways


def run(sym):
    """M.run with the direction and the minutes held kept too (the same-move cluster check reads them)."""
    import time
    import agent_bot5 as A
    M.OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out, t_end = {}, time.time() + M.BUDGET_MIN * 60
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
            out.setdefault(tk, {})[w] = [{"time": str(a), "r": float(b), "why": str(y), "dir": int(d_), "mins": int(m_)}
                                         for a, b, y, d_, m_ in zip(t["time"], t["r"], t["why"], t["dir"], t["mins"])]
        print(sym, tk, {w: len(x) if isinstance(x, list) else x for w, x in out[tk].items()}, flush=True)
        (M.OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")
    (M.OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")


def _st(rows):
    if not rows:
        return {"n": 0, "sum": 0.0, "mean": None, "dd": 0.0, "worst": 0.0, "green": None, "wmean": None}
    t = pd.to_datetime([x["time"] for x in rows], utc=True, format="ISO8601")
    r = np.array([x["r"] for x in rows], float)
    o = np.argsort(t.asi8)
    t, r = t[o], r[o]
    eq = np.cumsum(r)
    day = pd.Series(r, index=t.normalize()).groupby(level=0).sum()
    ws = K.wstats(r, K.weights(t))
    return {"n": int(len(r)), "sum": round(float(r.sum()), 2), "mean": round(float(r.mean()), 3),
            "win": round(100 * float((r > 0).mean()), 1),
            "dd": round(float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()), 2),
            "worst": round(float(day.min()), 2), "green": round(100 * float((day > 0).mean()), 1),
            "wmean": ws["mean"], "t": ws["t"]}


def report():
    agg = {}
    for f in sorted(M.OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        for tk, wz in doc.items():
            if tk.startswith("_"):
                continue
            for w, rows in wz.items():
                a = agg.setdefault(tk, {}).setdefault(w, {"train": [], "unseen": [], "err": None})
                if isinstance(rows, dict):
                    a["err"] = rows.get("error")
                    continue
                for x in rows:
                    t = pd.Timestamp(x["time"])
                    if t < M.FROM:
                        continue
                    (a["unseen"] if t >= TG.SPLIT else a["train"]).append(x)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "tracks": {}, "all": {}}
    for tk, wz in agg.items():
        rows = {w: {"train": _st(v["train"]), "unseen": _st(v["unseen"]), "error": v["err"]} for w, v in wz.items()}
        ok = [w for w in rows if not rows[w]["error"]]
        best = max(ok, key=lambda w: rows[w]["train"]["sum"], default=None)
        now = rows.get("now|base")
        rep["tracks"][tk] = {"rows": rows, "best": best,
                             "passes": bool(best and now and best != "now|base"
                                            and rows[best]["train"]["sum"] > now["train"]["sum"]
                                            and rows[best]["unseen"]["sum"] > now["unseen"]["sum"])}
    for w in ways():
        tr, un = [], []
        for tk in agg:
            v = agg[tk].get(w) or {}
            tr += v.get("train", [])
            un += v.get("unseen", [])
        rep["all"][w] = {"train": _st(tr), "unseen": _st(un)}
    pick = {"train": [], "unseen": []}
    for tk, v in rep["tracks"].items():
        b = v["best"] or "now|base"
        pick["train"] += agg[tk][b]["train"]
        pick["unseen"] += agg[tk][b]["unseen"]
    rep["pick_per_track"] = {k: _st(x) for k, x in pick.items()}
    (M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for w, x in sorted(rep["all"].items(), key=lambda kv: -kv[1]["train"]["sum"]):
        print(f"{w:16} 2025 {x['train']['sum']:8} n{x['train']['n']} dd{x['train']['dd']} worst{x['train']['worst']}"
              f" | 2026 {x['unseen']['sum']:8} n{x['unseen']['n']} dd{x['unseen']['dd']} worst{x['unseen']['worst']}")
    print("pick per track:", {k: (v["sum"], v["dd"]) for k, v in rep["pick_per_track"].items()})
    for tk, v in rep["tracks"].items():
        r = v["rows"]
        print(tk, "best", v["best"], v["passes"], "now", r["now|base"]["train"]["sum"], r["now|base"]["unseen"]["sum"],
              "best", r[v["best"]]["train"]["sum"] if v["best"] else None, r[v["best"]]["unseen"]["sum"] if v["best"] else None)


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
