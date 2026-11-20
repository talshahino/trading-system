"""Bot 5's gap method - the order a little before the middle of the gap (user, 20.11.2026 17:07, task b5neartouch2011:
GER40 t5_1 and AUS200 t1h_5 shorts came within ~1 point of the middle and then fell hard - "it touched the middle
exactly and went down, why didn't it enter?").
  now     the order at the middle (as today)
  h5 / h10 / h15   5% / 10% / 15% of half the box before the middle (tal_gap ETOL)
  c1 / c2          one / two of FTMO's spread + commission before the middle
  lad2n            LADDER 2n: half at the near edge, half at the middle (the same total risk)
On bot 5 exactly as it runs (its instruments, its tracks, sym_cfg, the tracks off left out), FTMO's real spreads
(TAL_COSTX=1). Per track the way is CHOSEN on 2025 (time-weighted R, agentkit.weights) and shown on 2026; the account
replay (0.20% a trade, 4.5% a day / 9% in total) for each way on every track, and for the per-track choice.
Results only - nothing in the bot changes.
  TAL_COSTX=1 python -m agents.tal_near_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from agents import agentkit as K
from agents import tal_fine_study as F
from agents import tal_gap as TG
from agents import tal_ideas_study as I
from agents import tal_mgmt_study as M

OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_near"
WAYS = {"now": {}, "h5": {"ETOL": "h5"}, "h10": {"ETOL": "h10"}, "h15": {"ETOL": "h15"},
        "c1": {"ETOL": "c1"}, "c2": {"ETOL": "c2"}, "lad2n": {"LADDER": "2n"}}


def plan():
    return I.plan()


def run(sym):
    import agent_bot5 as A
    OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out, t_end = {}, time.time() + 160 * 60
    for tk, tv in I.tracks().items():
        if not I.live(tk, sym):
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        base = dict(tv["cfg"], **((tv.get("sym_cfg") or {}).get(sym) or {}))
        gc, kk = {}, f"{tk}#{sym}"
        for w, extra in WAYS.items():
            if time.time() > t_end:
                out.setdefault(kk, {})[w] = {"error": "time budget"}
                continue
            if w == "lad2n" and base.get("LADDER"):
                out.setdefault(kk, {})[w] = {"error": "the track already ladders"}
                continue
            try:
                t = TG.simulate(D, pair, TG._norm(dict(base, **extra)), gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(kk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            out.setdefault(kk, {})[w] = [{"time": str(a), "r": float(b), "mins": int(m)}
                                         for a, b, m in zip(t["time"], t["r"], t["mins"])]
        print(kk, {w: len(x) if isinstance(x, list) else x for w, x in out[kk].items()}, flush=True)
        (OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")
    (OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")


def _rows():
    rows = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name != "report.json":
            rows.update(json.loads(f.read_text(encoding="utf-8")))
    return rows


def _xs(wz, w):
    return wz.get(w) if isinstance(wz.get(w), list) else wz.get("now", [])


def _acct(rows, pick):
    tz, sums = {"2025": [], "2026": []}, {"2025": 0.0, "2026": 0.0}
    for kk, wz in rows.items():
        for x in _xs(wz, pick(kk)):
            t0 = pd.Timestamp(x["time"])
            if t0 < M.FROM:
                continue
            y = "2026" if t0 >= TG.SPLIT else "2025"
            tz[y].append((t0, t0 + pd.Timedelta(minutes=int(x["mins"]) + 1), x["r"]))
            sums[y] += x["r"]
    return {"R": {k: round(v, 1) for k, v in sums.items()}, **{y: F.account(v) for y, v in tz.items()}}


def report():
    rows = _rows()
    F.RISK = 0.002
    res = {w: _acct(rows, lambda kk, w=w: w) for w in WAYS}
    per, now_ts = {}, pd.Timestamp("2026-01-01", tz="UTC")
    for kk, wz in rows.items():
        row = {}
        for w in WAYS:
            xs = _xs(wz, w)
            tr = [x for x in xs if M.FROM <= pd.Timestamp(x["time"]) < TG.SPLIT]
            un = [x for x in xs if pd.Timestamp(x["time"]) >= TG.SPLIT]
            wt = K.weights(pd.DatetimeIndex([pd.Timestamp(x["time"]) for x in tr]), now=now_ts) if tr else np.array([])
            row[w] = {"n25": len(tr), "R25": round(sum(x["r"] for x in tr), 2),
                      "wR25": round(float(np.dot(wt, [x["r"] for x in tr])) if tr else 0.0, 2),
                      "n26": len(un), "R26": round(sum(x["r"] for x in un), 2)}
        ch = max(row, key=lambda w: (row[w]["wR25"], w == "now"))
        per[kk] = dict(row, chosen=ch, better26=row[ch]["R26"] > row["now"]["R26"])
    res["chosen"] = _acct(rows, lambda kk: per[kk]["chosen"])
    tracks = {}
    for kk, v in per.items():
        tk = kk.split("#")[0]
        tracks.setdefault(tk, {})[kk.split("#")[1]] = v
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "ways": res, "per": per}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for w, v in res.items():
        print(f"{w:8} R {v['R']}  2025 {v['2025']['ret']:7} dd {v['2025']['dd']:5} | 2026 {v['2026']['ret']:7} dd {v['2026']['dd']:5} n {v['2026']['taken']}")
    for kk, v in sorted(per.items()):
        if v["chosen"] != "now":
            print(kk, v["chosen"], "2025", v["now"]["R25"], "->", v[v["chosen"]]["R25"], "2026", v["now"]["R26"], "->", v[v["chosen"]]["R26"])


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
