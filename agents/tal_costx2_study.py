"""Bot 5 - the tuned stop / target of single instruments with FTMO's real spreads (user, 20.11.2026 13:20). Stage 2
(agents/tal_fine_study) chose per instrument on 2025; alone in the account four of them held on 2026: JP225 (stop x0.70,
swing 5), US500 (impulse start, swing 6), EU50 (swing 4), GBPNZD (stop x1.25). Here they run with FTMO's real spreads
(TAL_COSTX=1, data/agents/ftmo_spread_x.json) and the account replay (0.28%, 4.5% a day / 9% total) - today's tracks
from agents/tal_costx_study (the same spreads) with each one alone and the ones that hold together. Results only.
  TAL_COSTX=1 python -m agents.tal_costx2_study plan | run --sym JP225 | report
"""
from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

from agents import tal_dirtrend_study as S
from agents import tal_fine_study as F

M = S.M
BASE = M.ROOT / "reports" / "agents" / "bot5" / "tal_costx"
M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_costx2"
TUNED = {"JP225": {"SSCALE": 0.70, "TSWN": 5}, "US500": {"STOPAT": "imp0", "TSWN": 6},
         "EU50": {"TSWN": 4}, "GBPNZD": {"SSCALE": 1.25}}
OFF = {"t5_1@UK100", "t15_1@USDJPY", "t1h_5@SPN35", "t5_1@SPN35", "t1h_5@US2000", "t1h_1@EURUSD"}
_SYM = [None]


def _ways():
    return {"tuned": dict(TUNED[_SYM[0]])} if _SYM[0] else {"tuned": {}}


S.ways = _ways


def run(sym):
    _SYM[0] = sym
    S.run(sym)


def _rows(folder, way):
    out = {}
    for f in sorted(folder.glob("*.json")):
        if f.name == "report.json":
            continue
        for kk, wz in json.loads(f.read_text(encoding="utf-8")).items():
            if isinstance(wz.get(way), list):
                out[kk] = wz[way]
    return out


def report():
    base, tuned = _rows(BASE, "now|base"), _rows(M.OUT, "tuned")

    def acct(use):
        tz = {"2025": [], "2026": []}
        for kk in base:
            tk, s = kk.split("#")
            if not S.live_pair(tk, s) or tk in OFF:
                continue
            xs = tuned.get(kk, base[kk]) if s in use else base[kk]
            for x in xs:
                t0 = pd.Timestamp(x["time"])
                if t0 >= M.FROM:
                    tz["2026" if t0 >= F.TG.SPLIT else "2025"].append((t0, t0 + pd.Timedelta(minutes=int(x["mins"]) + 1), x["r"]))
        return {y: F.account(v) for y, v in tz.items()}
    res = {"now": acct(set())}
    for s in TUNED:
        res[s] = acct({s})
    for n in (2, 3, 4):
        for c in itertools.combinations(TUNED, n):
            res["+".join(c)] = acct(set(c))
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "tuned": TUNED, "accounts": res}
    (M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:28} 2025 {v['2025']['ret']:7} dd {v['2025']['dd']:5} | 2026 {v['2026']['ret']:7} dd {v['2026']['dd']:5} worst {v['2026']['worst_day']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(sorted(TUNED)))
    elif a.mode == "run":
        run(a.sym)
    else:
        report()
