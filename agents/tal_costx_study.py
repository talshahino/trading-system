"""Bot 5's gap method with FTMO's real spreads (user, 20.11.2026 12:48: "check it counts FTMO's spread and commission
for every asset, by bot 1"). The backtest's spread is the Dukascopy history; on bot 1's FTMO account the real spread
was wider on some instruments (JP225 x1.83, N25 x2.1, US500 x1.16, US100 x1.11 - data/agents/ftmo_spread_x.json from
reports/costs_check.json). Here the tracks bot 5 runs, as they run, with those spreads (TAL_COSTX=1), and the
account replay (0.28%, 4.5% a day / 9% in total): before and after the 6 tracks taken off on 20.11, per year.
Results only.
  TAL_COSTX=1 python -m agents.tal_costx_study plan | run --sym US100 | report
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from agents import tal_dirtrend_study as S
from agents import tal_fine_study as F

M = S.M
M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_costx"
S.ways = lambda: {"now|base": {}}
OFF = {"t5_1@UK100", "t15_1@USDJPY", "t1h_5@SPN35", "t5_1@SPN35", "t1h_5@US2000", "t1h_1@EURUSD"}


def universe():
    import agent_bot5 as A
    syms = set(A.TAL_SYMBOLS)
    for tk in M.W.tracks():
        if "@" in tk:
            syms.add(tk.split("@")[1])
    return sorted(syms)


def report():
    rows = {}
    for f in sorted(M.OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        for kk, wz in json.loads(f.read_text(encoding="utf-8")).items():
            if isinstance(wz.get("now|base"), list):
                rows[kk] = wz["now|base"]
    out = {}
    for name, drop in (("before", set()), ("after_off", OFF)):
        tz = {"2025": [], "2026": []}
        per = {}
        for kk, xs in rows.items():
            tk, s = kk.split("#")
            if not S.live_pair(tk, s) or tk in drop:
                continue
            for x in xs:
                t0 = pd.Timestamp(x["time"])
                if t0 < M.FROM:
                    continue
                y = "2026" if t0 >= F.TG.SPLIT else "2025"
                tz[y].append((t0, t0 + pd.Timedelta(minutes=int(x["mins"]) + 1), x["r"]))
                per.setdefault(s, {"2025": 0.0, "2026": 0.0})[y] += x["r"]
        out[name] = {"account": {y: F.account(v) for y, v in tz.items()},
                     "per_symbol": {s: {k: round(v, 1) for k, v in d.items()} for s, d in sorted(per.items())}}
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "spread_x":
           json.loads((M.ROOT / "data" / "agents" / "ftmo_spread_x.json").read_text(encoding="utf-8"))["x"], **out}
    (M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k]["account"] for k in ("before", "after_off")}))
    print(json.dumps(rep["after_off"]["per_symbol"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(universe()))
    elif a.mode == "run":
        S.run(a.sym)
    else:
        report()
