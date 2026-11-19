"""Bot 5's gap tracks: the same move bought many times - split the risk instead of skipping (bot 5's smart agent,
19.11.2026 night). On 19.11 15:14-15:31 UTC one US index move gave 14 longs across four tracks, all stopped (-14R,
the bot's day -2.6%). agents/tal_cluster_study.py skipped the extra entries - that cut most of 2026's profit, because
the move bought many times is usually also the winner. Here no entry is skipped: each one keeps its stop and target and
only its size changes, from what is already open at its fill (no look-ahead):
  instH       half size while another track holds a trade the same way on the same instrument
  grpH        half size while a trade is open the same way in the same group (US = US100/US500/US30/US2000,
              Europe = GER40/UK100/EU50/SPN35/N25, Asia = JP225/AUS200; others alone)
  grpS        size 1/(1+n), n = trades open the same way in the same group
  instU/grpU  the other way round - x1.5 when the move is already held (adding as it agrees, as Tal likes to add)
  idxB<K>     the indices' trades the same way share a budget of K R: a new one gets min(1, K - open risk), none under 0.25
  allB<K>     the same budget over all the bot's instruments
Each trade's R already holds FTMO's real costs (they scale with the size). Chosen on 30.04-31.12.2025, shown on 2026.
Because a smaller size lowers both the profit and the risk, every way is also shown at the same largest fall from the
peak as today (its total scaled by today's fall / its fall on 2025) - the fair question is whether it earns more for
the same risk. Results only - nothing in the bot changes here.
  python -m agents.tal_cluster_risk_study [way]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from agents import tal_gap as TG
from agents.tal_cluster_study import FROM, INDICES, trades
from agents.tal_dbrk_study import _st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "tal_cluster_risk.json"
GROUP = {**{s: "us" for s in ("US100", "US500", "US30", "US2000")},
         **{s: "eu" for s in ("GER40", "UK100", "EU50", "SPN35", "N25")},
         **{s: "asia" for s in ("JP225", "AUS200")}}
RULES = {"now": None, "instH": ("instH",), "grpH": ("grpH",), "grpS": ("grpS",), "instU": ("instU",), "grpU": ("grpU",),
         **{f"idxB{k}": ("idxB", k) for k in (2, 3, 4, 6)}, **{f"allB{k}": ("allB", k) for k in (3, 4, 6)}}


def size(x, same, rule):
    kind = rule[0]
    g = GROUP.get(x["sym"], x["sym"])
    if kind == "instH":
        return 0.5 if any(o["sym"] == x["sym"] for o in same) else 1.0
    if kind == "grpH":
        return 0.5 if any(GROUP.get(o["sym"], o["sym"]) == g for o in same) else 1.0
    if kind == "instU":
        return 1.5 if any(o["sym"] == x["sym"] for o in same) else 1.0
    if kind == "grpU":
        return 1.5 if any(GROUP.get(o["sym"], o["sym"]) == g for o in same) else 1.0
    if kind == "grpS":
        return 1.0 / (1 + sum(GROUP.get(o["sym"], o["sym"]) == g for o in same))
    k = rule[1]
    if kind == "idxB":
        if x["sym"] not in INDICES:
            return 1.0
        used = sum(o["w"] for o in same if o["sym"] in INDICES)
    else:
        used = sum(o["w"] for o in same)
    w = min(1.0, k - used)
    return w if w >= 0.25 else 0.0


def apply(rows, rule):
    out, open_ = [], []
    for x in rows:
        open_ = [o for o in open_ if o["end"] > x["t"]]
        w = 1.0 if rule is None else size(x, [o for o in open_ if o["dir"] == x["dir"]], rule)
        if w <= 0:
            continue
        y = dict(x, w=w, r=x["r"] * w)
        out.append(y)
        open_.append(y)
    return out


def main(way="now|base"):
    rows = [x for x in trades(way) if x["t"] >= FROM]
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "way": way, "rules": {}}
    for name, rule in RULES.items():
        k = apply(rows, rule)
        tr = [x for x in k if x["t"] < TG.SPLIT]
        un = [x for x in k if x["t"] >= TG.SPLIT]
        rep["rules"][name] = {"train": _st(tr), "unseen": _st(un),
                              "size_mean": {"train": round(sum(x["w"] for x in tr) / max(len(tr), 1), 3),
                                            "unseen": round(sum(x["w"] for x in un) / max(len(un), 1), 3)}}
    now = rep["rules"]["now"]
    for name, v in rep["rules"].items():
        f = now["train"]["dd"] / max(v["train"]["dd"], 1e-9)   # the scale chosen on 2025 only
        v["same_risk"] = {"scale": round(f, 3),
                          "train": {"sum": round(v["train"]["sum"] * f, 2), "dd": round(v["train"]["dd"] * f, 2),
                                    "worst": round(v["train"]["worst"] * f, 2)},
                          "unseen": {"sum": round(v["unseen"]["sum"] * f, 2), "dd": round(v["unseen"]["dd"] * f, 2),
                                     "worst": round(v["unseen"]["worst"] * f, 2)}}
    ok = [n for n in rep["rules"] if n != "now"
          and rep["rules"][n]["same_risk"]["train"]["sum"] > now["train"]["sum"]]
    best = max(ok, key=lambda n: rep["rules"][n]["same_risk"]["train"]["sum"], default=None)
    rep["best"] = best
    if best:
        b = rep["rules"][best]["same_risk"]
        rep["passes"] = bool(b["unseen"]["sum"] > now["unseen"]["sum"] and b["unseen"]["dd"] <= now["unseen"]["dd"]
                             and b["unseen"]["worst"] >= now["unseen"]["worst"])
    else:
        rep["passes"] = False
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, v in rep["rules"].items():
        a, b, s = v["train"], v["unseen"], v["same_risk"]
        print(f"{n:7} 2025 {a['sum']:7} dd{a['dd']:6} w{a['worst']:6} g{a['green']} | 2026 {b['sum']:7} dd{b['dd']:6}"
              f" w{b['worst']:6} g{b['green']} | same-risk x{s['scale']}: 2025 {s['train']['sum']} 2026 {s['unseen']['sum']}"
              f" dd{s['unseen']['dd']} w{s['unseen']['worst']}")
    print("best", best, "passes", rep["passes"])
    return rep


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "now|base")
