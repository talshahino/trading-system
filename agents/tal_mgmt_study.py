"""Bot 5's gap method - managing the trade (user, 18.11.2026, voice, task tal1811mgmt):
  1. the stop beyond the impulse's start with a small room, so a return exactly to the start does not stop the trade
     (STOPAT imp0 / imp0r + IPAD: a share of the high timeframe's ATR or a multiple of the spread + commission)
  2. the stop moved after every new break of structure in the trade's favour, beyond the new swing with room (BOS)
  3. at 60% of the way to the target the stop to the entry (EXIT be60)
  4. half out at 60% only when the volume toward the target is not strong; strong - only the stop to the entry (VOLP)
Each alone and together, on every gap track of bot 5 as it runs (live/bots/bot5.json tal_tracks), FTMO's costs.
Chosen on 30.04-31.12.2025, shown on 2026. Nothing in the bot changes here (agents/tal_gap.mg_walk).
"walk" = today's rules through the new minute-by-minute walk - must equal "now" (a check of the walk itself).

  python -m agents.tal_mgmt_study plan
  python -m agents.tal_mgmt_study run --sym US100
  python -m agents.tal_mgmt_study report
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents import tal_gap as TG
from agents import tal_wait_study as W

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "tal_mgmt"
FROM = pd.Timestamp("2025-04-30", tz="UTC")
BUDGET_MIN = 150
STOPS = {"now": {}, "imp0": {"STOPAT": "imp0"},
         "imp0_a10": {"STOPAT": "imp0", "IPAD": "a0.1"}, "imp0_a25": {"STOPAT": "imp0", "IPAD": "a0.25"},
         "imp0_a50": {"STOPAT": "imp0", "IPAD": "a0.5"}, "imp0_c2": {"STOPAT": "imp0", "IPAD": "c2"},
         "imp0_c4": {"STOPAT": "imp0", "IPAD": "c4"}, "imp0r_a25": {"STOPAT": "imp0r", "IPAD": "a0.25"}}
MGS = {"base": {}, "be60": {"EXIT": "be60"}, "p50_60": {"EXIT": "p50_60"}, "bos": {"BOS": 1},
       "be60_bos": {"EXIT": "be60", "BOS": 1}, "be60_vol15": {"EXIT": "be60", "VOLP": 1.5},
       "be60_vol12": {"EXIT": "be60", "VOLP": 1.2}, "be60_bos_vol15": {"EXIT": "be60", "BOS": 1, "VOLP": 1.5}}


def ways():
    w = {f"{s}|{m}": dict(sv, **mv) for s, sv in STOPS.items() for m, mv in MGS.items()}
    w["now|walk"] = {"MGW": 1}
    return w


def plan():
    return W.plan()


def run(sym):
    import agent_bot5 as A
    OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    v = D.v1
    out = {"_vol": {"real": bool(np.nanstd(v[-50000:]) > 0)}}
    t_end = time.time() + BUDGET_MIN * 60
    for tk, tv in W.tracks().items():
        if sym not in W.syms_of(tk):
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        base = TG._norm(dict(tv["cfg"]))
        gc = {}
        for w, extra in ways().items():
            if time.time() > t_end:
                out.setdefault(tk, {})[w] = {"error": "time budget"}
                continue
            c = TG._norm(dict(base, **extra))
            try:
                t = TG.simulate(D, pair, c, gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(tk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            out.setdefault(tk, {})[w] = [{"time": str(a), "r": float(b), "why": str(y)}
                                         for a, b, y in zip(t["time"], t["r"], t["why"])]
        print(sym, tk, {w: len(x) if isinstance(x, list) else x for w, x in out[tk].items()}, flush=True)
        (OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")
    (OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")


def _st(rs):
    if not rs:
        return {"n": 0, "sum": 0.0, "mean": None, "dd": 0.0}
    s = pd.Series(rs, dtype=float)
    eq = s.cumsum()
    return {"n": int(len(s)), "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 3),
            "win": round(100 * float((s > 0).mean()), 1), "dd": round(float((eq.cummax().clip(lower=0) - eq).max()), 2)}


def report():
    agg, novol = {}, []
    for f in sorted(OUT.glob("*.json")):
        if f.name in ("report.json", "marks.json"):
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        if not (doc.get("_vol") or {}).get("real", True):
            novol.append(f.stem)
        for tk, wz in doc.items():
            if tk.startswith("_"):
                continue
            for w, rows in wz.items():
                a = agg.setdefault(tk, {}).setdefault(w, {"train": [], "unseen": [], "err": None, "why": {}})
                if isinstance(rows, dict):
                    a["err"] = rows.get("error")
                    continue
                for x in sorted(rows, key=lambda r: r["time"]):
                    t = pd.Timestamp(x["time"])
                    if t < FROM:
                        continue
                    (a["unseen"] if t >= TG.SPLIT else a["train"]).append(x["r"])
                    if t >= TG.SPLIT:
                        a["why"][x["why"]] = a["why"].get(x["why"], 0) + 1
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "no_volume": novol, "tracks": {}, "all": {}}
    for tk, wz in agg.items():
        rows = {w: {"train": _st(v["train"]), "unseen": _st(v["unseen"]), "error": v["err"], "why26": v["why"]}
                for w, v in wz.items()}
        ok = [w for w in rows if not rows[w]["error"] and w != "now|walk"]
        best = max(ok, key=lambda w: rows[w]["train"]["sum"], default=None)
        now = rows.get("now|base")
        rep["tracks"][tk] = {"rows": rows, "best": best,
                             "passes": bool(best and now and best != "now|base"
                                            and rows[best]["train"]["sum"] > now["train"]["sum"]
                                            and rows[best]["unseen"]["sum"] > now["unseen"]["sum"])}
    for w in ways():
        tr = [rep["tracks"][t]["rows"].get(w, {}) for t in rep["tracks"]]
        rep["all"][w] = {"train_sum": round(sum(x.get("train", {}).get("sum", 0) for x in tr), 2),
                         "unseen_sum": round(sum(x.get("unseen", {}).get("sum", 0) for x in tr), 2),
                         "train_n": sum(x.get("train", {}).get("n", 0) for x in tr),
                         "unseen_n": sum(x.get("unseen", {}).get("n", 0) for x in tr)}
    # the best per track chosen on 2025, all together on 2026
    pick = {"train": 0.0, "unseen": 0.0}
    for tk, v in rep["tracks"].items():
        b = v["best"] or "now|base"
        pick["train"] += v["rows"][b]["train"]["sum"]
        pick["unseen"] += v["rows"][b]["unseen"]["sum"]
    rep["pick_per_track"] = {k: round(x, 2) for k, x in pick.items()}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for w, x in sorted(rep["all"].items(), key=lambda kv: -kv[1]["train_sum"]):
        print(f"{w:24} 2025 {x['train_sum']:8} ({x['train_n']})  2026 {x['unseen_sum']:8} ({x['unseen_n']})")
    print("pick per track:", rep["pick_per_track"], "no volume:", novol)
    for tk, v in rep["tracks"].items():
        r = v["rows"]
        print(tk, "best", v["best"], v["passes"], "now", r["now|base"]["train"]["sum"], r["now|base"]["unseen"]["sum"],
              "walk", r.get("now|walk", {}).get("train", {}).get("sum"), r.get("now|walk", {}).get("unseen", {}).get("sum"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(plan()))
    elif a.mode == "run":
        run(a.sym)
    else:
        report()
