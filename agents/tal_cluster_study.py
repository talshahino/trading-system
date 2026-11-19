"""Bot 5's gap tracks on the same move (bot 5's smart agent, 19.11.2026): on 19.11 15:14-15:31 UTC one US index move
gave 14 long entries on US500 / US100 / JP225 across t5_1, t15_1, t1h_15 and t1h_5 - all stopped together (-14R).
Each track checks only its own trades, so the same move is bought many times over. Tested here on the gap tracks'
trades as they run (reports/agents/bot5/tal_dbrk, way "now|base" or any other): a new entry is skipped while
  inst1   another track already holds a trade the same way on the same instrument
  idxK    K trades (K = 2 / 3 / 4 / 6) are already open the same way on the indices together (any track)
  allK    K trades are already open the same way on all the bot's instruments together
Real costs are inside each trade's R; chosen on 30.04-31.12.2025, shown on 2026, with the largest fall from the peak
(R), the worst day and the % of green days. Results only - nothing in the bot changes here.
  python -m agents.tal_cluster_study [way]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agents import tal_gap as TG
from agents.tal_dbrk_study import _st

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "agents" / "bot5" / "tal_dbrk"
OUT = ROOT / "reports" / "agents" / "bot5" / "tal_cluster.json"
FROM = pd.Timestamp("2025-04-30", tz="UTC")
INDICES = {"US100", "US500", "US30", "US2000", "GER40", "UK100", "EU50", "JP225", "AUS200", "N25", "SPN35"}
RULES = {"now": None, "inst1": ("inst", 1), "idx2": ("idx", 2), "idx3": ("idx", 3), "idx4": ("idx", 4),
         "idx6": ("idx", 6), "all3": ("all", 3), "all4": ("all", 4), "all6": ("all", 6)}


def trades(way="now|base"):
    rows = []
    for f in sorted(SRC.glob("*.json")):
        if f.name == "report.json":
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        for tk, wz in doc.items():
            xs = wz.get(way) if isinstance(wz, dict) else None
            if not isinstance(xs, list):
                continue
            for x in xs:
                if "dir" not in x:
                    continue
                t = pd.Timestamp(x["time"])
                rows.append({"t": t, "end": t + pd.Timedelta(minutes=int(x.get("mins") or 0)), "r": x["r"],
                             "dir": x["dir"], "sym": f.stem, "trk": tk, "time": x["time"]})
    return sorted(rows, key=lambda r: (r["t"], r["trk"], r["sym"]))


def apply(rows, rule):
    if rule is None:
        return rows
    kind, k = rule
    kept, open_ = [], []
    for x in rows:
        open_ = [o for o in open_ if o["end"] > x["t"]]
        same = [o for o in open_ if o["dir"] == x["dir"]]
        if kind == "inst":
            n = sum(o["sym"] == x["sym"] for o in same)
        elif kind == "idx":
            n = sum(o["sym"] in INDICES for o in same) if x["sym"] in INDICES else 0
        else:
            n = len(same)
        if n >= k:
            continue
        kept.append(x)
        open_.append(x)
    return kept


def main(way="now|base"):
    rows = [x for x in trades(way) if x["t"] >= FROM]
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "way": way, "rules": {}}
    for name, rule in RULES.items():
        k = apply(rows, rule)
        rep["rules"][name] = {"train": _st([x for x in k if x["t"] < TG.SPLIT]),
                              "unseen": _st([x for x in k if x["t"] >= TG.SPLIT])}
    now = rep["rules"]["now"]
    ok = [n for n, v in rep["rules"].items() if n != "now"
          and v["train"]["sum"] >= now["train"]["sum"] and v["train"]["dd"] <= now["train"]["dd"]]
    best = max(ok, key=lambda n: rep["rules"][n]["train"]["sum"] / max(rep["rules"][n]["train"]["dd"], 1e-9), default=None)
    rep["best"] = best
    rep["passes"] = bool(best and rep["rules"][best]["unseen"]["sum"] >= now["unseen"]["sum"]
                         and rep["rules"][best]["unseen"]["dd"] <= now["unseen"]["dd"])
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, v in rep["rules"].items():
        a, b = v["train"], v["unseen"]
        print(f"{n:6} 2025 {a['sum']:8} n{a['n']} dd{a['dd']} worst{a['worst']} green{a['green']} | "
              f"2026 {b['sum']:8} n{b['n']} dd{b['dd']} worst{b['worst']} green{b['green']}")
    print("best", best, "passes", rep["passes"])
    return rep


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "now|base")
