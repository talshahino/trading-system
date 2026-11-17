"""When a waiting gap order of bot 5 is cancelled (user, 17.11.2026: "we wait for the correction - when is the wait
cancelled if it does not come? I don't want it to wait there forever"). Today only the time (EXPIRY: 8h / 24h / 120h
by the track) and a few events cancel it. Tested here on every gap track of bot 5 as it runs (live/bots/bot5.json
tal_tracks), with WAITX (agents/tal_gap.wait_cut):
  tgt        the move went on without a correction (beyond the impulse's end by as much again as gap-middle -> end)
  h12, h24   a cap of 12 / 24 candles of the high timeframe
  opp        a new impulse with a gap the other way
  tgt+h24    both
FTMO's real costs, chosen on the period to 2025-12-31, shown on 2026. Nothing in the bot changes here.

  python -m agents.tal_wait_study run --sym US100
  python -m agents.tal_wait_study report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agents import tal_gap as TG

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "tal_wait"
WAYS = ["now", "tgt", "h12", "h24", "opp", "tgt+h24"]


def tracks():
    s5 = json.loads((ROOT / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    return s5.get("tal_tracks") or {}


def syms_of(track):
    import agent_bot5 as A
    return [track.split("@")[1]] if "@" in track else list(A.TAL_SYMBOLS)


def plan():
    return sorted({s for t in tracks() for s in syms_of(t)})


def run(sym):
    import agent_bot5 as A
    OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out = {}
    for tk, v in tracks().items():
        if sym not in syms_of(tk):
            continue
        pair = next(p for p in TG.PAIRS if p[0] == v["pair"][0] and p[1] == v["pair"][1])
        base = TG._norm(dict(v["cfg"]))
        gc = {}
        for w in WAYS:
            c = TG._norm(dict(base, WAITX=w)) if w != "now" else base
            try:
                t = TG.simulate(D, pair, c, gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(tk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
            rows = t[["time", "r"]].assign(time=t["time"].astype(str)).to_dict("records")
            out.setdefault(tk, {})[w] = rows
        print(sym, tk, {w: len(x) if isinstance(x, list) else x for w, x in out[tk].items()})
    (OUT / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")


def _st(rs):
    if not rs:
        return {"n": 0, "sum": 0.0, "mean": None}
    s = pd.Series(rs)
    return {"n": int(len(s)), "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 3)}


def report():
    split = TG.SPLIT
    agg = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        for tk, ways in json.loads(f.read_text(encoding="utf-8")).items():
            for w, rows in ways.items():
                a = agg.setdefault(tk, {}).setdefault(w, {"train": [], "unseen": [], "err": None})
                if isinstance(rows, dict):
                    a["err"] = rows.get("error")
                    continue
                for x in rows:
                    (a["unseen"] if pd.Timestamp(x["time"]) >= split else a["train"]).append(x["r"])
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "tracks": {}, "all": {}}
    for tk, ways in agg.items():
        rows = {w: {"train": _st(v["train"]), "unseen": _st(v["unseen"]), "error": v["err"]} for w, v in ways.items()}
        now = rows.get("now")
        best = max((w for w in rows if w != "now" and not rows[w]["error"]), key=lambda w: rows[w]["train"]["sum"], default=None)
        rep["tracks"][tk] = {"rows": rows, "best": best,
                             "passes": bool(best and now and rows[best]["train"]["sum"] > now["train"]["sum"]
                                            and rows[best]["unseen"]["sum"] > now["unseen"]["sum"])}
    for w in WAYS:                         # every track together, each way
        tr = sum(rep["tracks"][t]["rows"].get(w, {}).get("train", {}).get("sum", 0) for t in rep["tracks"])
        un = sum(rep["tracks"][t]["rows"].get(w, {}).get("unseen", {}).get("sum", 0) for t in rep["tracks"])
        nn = sum(rep["tracks"][t]["rows"].get(w, {}).get("unseen", {}).get("n", 0) for t in rep["tracks"])
        rep["all"][w] = {"train_sum": round(tr, 2), "unseen_sum": round(un, 2), "unseen_n": nn}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep["all"], ensure_ascii=False, indent=1))
    for tk, v in rep["tracks"].items():
        print(tk, v["best"], v["passes"], {w: (r["train"]["sum"], r["unseen"]["sum"], r["unseen"]["n"]) for w, r in v["rows"].items()})


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
