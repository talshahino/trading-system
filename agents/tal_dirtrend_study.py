"""Bot 5's gap method after 19.11.2026 (user, 19-20.11.2026: 13 longs on US500 / US100 / JP225 / US2000 against a falling
afternoon, -$3,866, while the morning's shorts were right; "check yesterday's trades again and draw conclusions"):
  1. the big picture's direction (TREND) on every gap track: as it runs, none, and the modes tal_gap has - the day's open,
     the timeframe above's EMA 20, EMA 9/20 of the high / entry timeframe, EMA 50/200, New York's cash open, premium /
     discount, the daily pivot
  2. a cap on trades open together in the same direction across the indices (and across everything) - from the
     trades as they run (rows with dir and minutes), in time order over all instruments and tracks
On every FTMO instrument that fits the method (user, 20.11.2026: "check it on all of FTMO's assets that fit the method"):
the 28 forex pairs, the 3 metals and the 13 cash indices - the general tracks (no "@") on each, the "@" tracks on their
own; "live" = the tracks x instruments bot 5 runs now. FTMO's costs, chosen on 30.04-31.12.2025, shown on 2026.
Results only.
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


def universe():
    import agent_bot5 as A
    from agents import tal_wide as TW
    return list(dict.fromkeys(list(A.TAL_SYMBOLS) + TW.FX28 + TW.METALS + TW.INDICES))


def live_pair(tk, sym):
    import agent_bot5 as A
    return (tk.split("@")[1] == sym) if "@" in tk else sym in A.TAL_SYMBOLS


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
        if "@" in tk and tk.split("@")[1] != sym:
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        base = TG._norm(dict(tv["cfg"]))
        gc = {}
        kk = f"{tk}#{sym}"
        for w, extra in ways().items():
            if time.time() > t_end:
                out.setdefault(kk, {})[w] = {"error": "time budget"}
                continue
            try:
                t = TG.simulate(D, pair, TG._norm(dict(base, **extra)), gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(kk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            out.setdefault(kk, {})[w] = [{"time": str(a), "r": float(b), "why": str(y), "dir": int(d), "mins": int(m)}
                                         for a, b, y, d, m in zip(t["time"], t["r"], t["why"], t["dir"], t["mins"])]
        print(kk, {w: len(x) if isinstance(x, list) else x for w, x in out[kk].items()}, flush=True)
        (M.OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")
    (M.OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")


def caps(scope):
    """The trades as they run (now|base), every instrument and track in time order: a trade is skipped while `cap`
    trades in its direction are open in its group (indices / everything)."""
    rows = []
    for f in sorted(M.OUT.glob("*.json")):
        if f.name in ("report.json", "marks.json"):
            continue
        sym = f.stem
        for kk, wz in json.loads(f.read_text(encoding="utf-8")).items():
            if scope == "live" and not live_pair(kk.split("#")[0], sym):
                continue
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


def trends():
    """Per scope (live / all) and way: 2025 / 2026 sums; and per track x instrument the way chosen on 2025."""
    rows = {}
    for f in sorted(M.OUT.glob("*.json")):
        if f.name in ("report.json", "marks.json"):
            continue
        for kk, wz in json.loads(f.read_text(encoding="utf-8")).items():
            for w, xs in wz.items():
                if not isinstance(xs, list):
                    continue
                a = {"train": [], "unseen": []}
                for x in xs:
                    t = pd.Timestamp(x["time"])
                    if t >= M.FROM:
                        a["unseen" if t >= TG.SPLIT else "train"].append(x["r"])
                rows.setdefault(kk, {})[w] = {k: M._st(v) for k, v in a.items()}
    out = {}
    for scope in ("live", "all"):
        keys = [k for k in rows if scope == "all" or live_pair(k.split("#")[0], k.split("#")[1])]
        agg = {w: {"train": round(sum(rows[k].get(w, {}).get("train", {}).get("sum", 0) for k in keys), 2),
                   "unseen": round(sum(rows[k].get(w, {}).get("unseen", {}).get("sum", 0) for k in keys), 2),
                   "n_unseen": sum(rows[k].get(w, {}).get("unseen", {}).get("n", 0) for k in keys)} for w in ways()}
        pick = {"train": 0.0, "unseen": 0.0}
        for k in keys:
            b = max(rows[k], key=lambda w: rows[k][w]["train"]["sum"])
            pick["train"] += rows[k][b]["train"]["sum"]
            pick["unseen"] += rows[k][b]["unseen"]["sum"]
        # instruments x tracks that earn on both periods as they run (now|base) - the method's fit, per instrument
        fit = sorted([k for k in keys if rows[k].get("now|base", {}).get("train", {}).get("sum", 0) > 0
                      and rows[k].get("now|base", {}).get("unseen", {}).get("sum", 0) > 0])
        out[scope] = {"ways": agg, "pick": {k: round(v, 2) for k, v in pick.items()}, "n": len(keys), "fit_both": fit}
    return out, rows


def report():
    tr, rows = trends()
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "trend": tr,
           "caps": {s: caps(s) for s in ("live", "all")}, "rows": rows}
    M.OUT.mkdir(parents=True, exist_ok=True)
    (M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for sc in ("live", "all"):
        print("==", sc, tr[sc]["n"], "pick", tr[sc]["pick"])
        for w, x in sorted(tr[sc]["ways"].items(), key=lambda kv: -kv[1]["train"]):
            print(f"  {w:12} 2025 {x['train']:9} 2026 {x['unseen']:9} ({x['n_unseen']})")
        for k, v in rep["caps"][sc].items():
            print(f"  cap {k:12} 2025 {v['train']['sum']:8} (dd {v['train']['dd']})  2026 {v['unseen']['sum']:8} ({v['unseen']['n']}, dd {v['unseen']['dd']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(universe()))
    elif a.mode == "run":
        run(a.sym)
    else:
        report()
