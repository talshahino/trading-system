"""Bot 5's gap method + three ideas the network scout found on TikTok (user, 20.11.2026 19:13 "yes"):
  g3|fvg   only gaps whose candle 3 closes inside candle 2's range (a fair value gap - expect the return)
  g3|brk   only gaps whose candle 3 closes beyond candle 2's extreme (a breakaway gap) - the opposite, for reference
  swp|sw20 / swp|pdhl   only gaps after a sweep of a significant swing (the 20 candles' extreme / the previous day's)
  po3|m    the 4 hour candle's power of three: by the fill it already ran beyond its open against the trade
  po3|d    and the fill is on the manipulation's side of the 4h open
On bot 5 exactly as it runs since 20.11 (13 instruments, its tracks, sym_cfg, the tracks off left out), FTMO's real
spreads (TAL_COSTX=1), the account replay at 0.20% a trade with 4.5% a day / 9% in total. Chosen on 2025, shown on
2026. Results only.
  TAL_COSTX=1 python -m agents.tal_ideas_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from agents import tal_fine_study as F
from agents import tal_gap as TG
from agents import tal_mgmt_study as M

OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_ideas"
WAYS = {"now": {}, "g3|fvg": {"G3": "fvg"}, "g3|brk": {"G3": "brk"}, "swp|sw20": {"SWP": "sw20"},
        "swp|pdhl": {"SWP": "pdhl"}, "po3|m": {"PO3": "m"}, "po3|d": {"PO3": "d"}}


def tracks():
    return {k: v for k, v in M.W.tracks().items() if not v.get("off")}


def live(tk, sym):
    import agent_bot5 as A
    return (tk.split("@")[1] == sym) if "@" in tk else sym in A.TAL_SYMBOLS


def plan():
    import agent_bot5 as A
    return sorted({tk.split("@")[1] for tk in tracks() if "@" in tk} | set(A.TAL_SYMBOLS))


def run(sym):
    import agent_bot5 as A
    OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out, t_end = {}, time.time() + 160 * 60
    for tk, tv in tracks().items():
        if not live(tk, sym):
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        base = dict(tv["cfg"], **((tv.get("sym_cfg") or {}).get(sym) or {}))
        gc, kk = {}, f"{tk}#{sym}"
        for w, extra in WAYS.items():
            if time.time() > t_end:
                out.setdefault(kk, {})[w] = {"error": "time budget"}
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


def report():
    rows = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name != "report.json":
            rows.update(json.loads(f.read_text(encoding="utf-8")))
    F.RISK = 0.002
    res = {}
    for w in WAYS:
        tz, sums = {"2025": [], "2026": []}, {"2025": 0.0, "2026": 0.0}
        for kk, wz in rows.items():
            xs = wz.get(w) if isinstance(wz.get(w), list) else wz.get("now", [])
            for x in xs:
                t0 = pd.Timestamp(x["time"])
                if t0 < M.FROM:
                    continue
                y = "2026" if t0 >= TG.SPLIT else "2025"
                tz[y].append((t0, t0 + pd.Timedelta(minutes=int(x["mins"]) + 1), x["r"]))
                sums[y] += x["r"]
        res[w] = {"R": {k: round(v, 1) for k, v in sums.items()}, **{y: F.account(v) for y, v in tz.items()}}
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "ways": res}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for w, v in res.items():
        print(f"{w:10} R {v['R']}  2025 {v['2025']['ret']:7} dd {v['2025']['dd']:5} | 2026 {v['2026']['ret']:7} dd {v['2026']['dd']:5} n {v['2026']['taken']}")


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
