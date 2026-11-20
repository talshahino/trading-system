"""Bot 5's gap method - a strategy of its own for every instrument (user, 20.11.2026 08:57: "a study on every asset
separately, every currency, everything... each one by a different target - swing 1 to 6 - or the stop in different
places, each separately, and a different strategy for each - what works best for each"). Every FTMO forex pair, metal
and cash index that fits the method (agents/tal_dirtrend_study's universe; "live" = what bot 5 runs now); every gap
track on it; every stop x target together:
  stops    now (beyond the move + 4x costs), x0.75, x0.9 (closer), x1.15, x1.3, x1.5 (further), imp0 (impulse start)
  targets  swing 1..6 beyond the entry at the fill (now: 2)
The same $ risk per trade, MIN_RR as live, FTMO's costs. Chosen on 30.04-31.12.2025 per instrument (one stop x target
for all its tracks) and per track x instrument, shown on 2026. Results only - nothing in the bot changes.
  python -m agents.tal_persym_study plan | run --sym US100-g1 | report
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from agents import tal_dirtrend_study as S
from agents import tal_gap as TG

M = S.M
M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_persym"
STOPS = {"now": {}, "k0.75": {"SSCALE": 0.75}, "k0.9": {"SSCALE": 0.9}, "k1.15": {"SSCALE": 1.15},
         "k1.3": {"SSCALE": 1.3}, "k1.5": {"SSCALE": 1.5}, "imp0": {"STOPAT": "imp0"}}
GROUPS = {"g1": (1, 2), "g2": (3, 4), "g3": (5, 6)}


def ways(g=None):
    out = {}
    for gk, ns in GROUPS.items():
        if g and gk != g:
            continue
        for n in ns:
            for sk, sv in STOPS.items():
                out[f"{sk}|sw{n}"] = dict(sv, **({"TSWN": n} if n != 2 else {}))
    return out


S.ways = ways


def plan():
    return [f"{s}-{g}" for s in S.universe() for g in GROUPS]


def run(job):
    import agent_bot5 as A
    sym, g = job.rsplit("-", 1)
    M.OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out, t_end = {}, time.time() + 160 * 60
    for tk, tv in M.W.tracks().items():
        if "@" in tk and tk.split("@")[1] != sym:
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        base = TG._norm(dict(tv["cfg"]))
        gc, kk = {}, f"{tk}#{sym}"
        for w, extra in ways(g).items():
            if time.time() > t_end:
                out.setdefault(kk, {})[w] = {"error": "time budget"}
                continue
            try:
                t = TG.simulate(D, pair, TG._norm(dict(base, **extra)), gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(kk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            out.setdefault(kk, {})[w] = [{"time": str(a), "r": float(b), "dir": int(d)}
                                         for a, b, d in zip(t["time"], t["r"], t["dir"])]
        print(kk, {w: len(x) if isinstance(x, list) else x for w, x in out[kk].items()}, flush=True)
        (M.OUT / f"{job}.json").write_text(json.dumps(out), encoding="utf-8")
    (M.OUT / f"{job}.json").write_text(json.dumps(out), encoding="utf-8")


def report():
    _, rows = S.trends()             # {track#sym: {way: {train, unseen}}}
    W = list(ways())
    sm = lambda d, w, p: d.get(w, {}).get(p, {}).get("sum", 0.0)
    syms = sorted({k.split("#")[1] for k in rows})
    per_sym, per_trk = {}, {}
    tot = {sc: {"now": [0.0, 0.0], "sym": [0.0, 0.0], "trk": [0.0, 0.0]} for sc in ("live", "all")}
    for s in syms:
        ks = [k for k in rows if k.endswith("#" + s)]
        for sc in ("live", "all"):
            kk = [k for k in ks if sc == "all" or S.live_pair(k.split("#")[0], s)]
            if not kk:
                continue
            agg = {w: [sum(sm(rows[k], w, "train") for k in kk), sum(sm(rows[k], w, "unseen") for k in kk),
                       sum(rows[k].get(w, {}).get("unseen", {}).get("n", 0) for k in kk)] for w in W}
            b = max(W, key=lambda w: agg[w][0])
            if sc == "all":
                per_sym[s] = {"best": b, "best_2025": round(agg[b][0], 1), "best_2026": round(agg[b][1], 1),
                              "n_2026": agg[b][2], "now_2025": round(agg["now|sw2"][0], 1),
                              "now_2026": round(agg["now|sw2"][1], 1), "tracks": len(kk),
                              "live": any(S.live_pair(k.split("#")[0], s) for k in ks)}
            tot[sc]["now"][0] += agg["now|sw2"][0]; tot[sc]["now"][1] += agg["now|sw2"][1]
            tot[sc]["sym"][0] += agg[b][0]; tot[sc]["sym"][1] += agg[b][1]
    for k, ws in rows.items():
        b = max(W, key=lambda w: sm(ws, w, "train"))
        per_trk[k] = {"best": b, "best_2025": round(sm(ws, b, "train"), 1), "best_2026": round(sm(ws, b, "unseen"), 1),
                      "now_2025": round(sm(ws, "now|sw2", "train"), 1), "now_2026": round(sm(ws, "now|sw2", "unseen"), 1)}
        for sc in ("live", "all"):
            if sc == "all" or S.live_pair(*k.split("#")):
                tot[sc]["trk"][0] += sm(ws, b, "train")
                tot[sc]["trk"][1] += sm(ws, b, "unseen")
    # the instruments whose own choice earns on both periods and beats "now" on both
    good = sorted([s for s, v in per_sym.items() if v["best_2025"] > 0 and v["best_2026"] > 0
                   and v["best_2025"] > v["now_2025"] and v["best_2026"] > v["now_2026"]])
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "totals": {sc: {k: [round(x, 1) for x in v] for k, v in d.items()} for sc, d in tot.items()},
           "per_symbol": per_sym, "per_track": per_trk, "good_symbols": good}
    (M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep["totals"]))
    for s, v in sorted(per_sym.items(), key=lambda kv: -kv[1]["best_2026"]):
        print(f"{s:7} {'*' if v['live'] else ' '} {v['best']:12} {v['best_2025']:7} {v['best_2026']:7} ({v['n_2026']})  now {v['now_2025']:7} {v['now_2026']:7}")
    print("good:", good)


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
