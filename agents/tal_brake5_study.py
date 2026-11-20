"""Bot 5's gap tracks - active brakes and time windows (bot 5's smart agent, 20.11.2026 night; loss report of 19.11,
task c80458266b8c, and Tal's 17.11 question "how do you not have risk-management strategies?", tal1711riskmgmt - the
bots 1-4 study agents/risk_brakes.py never ran on bot 5). On 19.11 26 of 39 losing trades were stopped without moving
for us; 14 index longs were filled 15:14-15:22 UTC (44-52 minutes after the US cash open, during one sell-off) and
all stopped. Tested on every gap trade as the tracks trade now (reports/agents/bot5/tal_dbrk, way "now|base"; real
FTMO costs already inside each R; 0.28% a trade, so 1% of the account ~ 3.6R):

  dayH<X> / dayS<X>  soft daily brake: once the day's closed result (FTMO day, Prague midnight) is -X R, new entries get
                     half size (H) or none (S) until tomorrow; X = 3.6 / 5.4 / 7.1 (-1% / -1.5% / -2%)
  red2H              after 2 losing days in a row: half size until a green day
  usoK               no fill in the first K minutes after the US cash open (09:30 New York), US indices only, K=30/60
  usoKall            the same on every index (one rule for the group)
  stopN              no new entry on an instrument (any track) the same day after N stops on it the same way, N=2/3

Nothing is chosen per trade; each rule is fixed in advance. Chosen on 30.04-31.12.2025, shown on 2026 (unseen),
time-weighted. A rule is a card only if it beats "as today" on BOTH periods (more profit, no larger fall), or smooths
the curve under agentkit.curve_ok. Results only - nothing in the bot changes here.
  python -m agents.tal_brake5_study   -> reports/agents/bot5/tal_brake5.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import tal_gap as TG
from agents.tal_cluster_study import FROM, trades
from agents.tal_dbrk_study import _st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "tal_brake5.json"
US = {"US100", "US500", "US30", "US2000"}
INDICES = US | {"GER40", "UK100", "EU50", "JP225", "AUS200", "N25", "SPN35"}
R_PER_PCT = 1 / 0.28
RULES = {"now": None,
         **{f"dayH{x}": ("day", x, 0.5) for x in (3.6, 5.4, 7.1)},
         **{f"dayS{x}": ("day", x, 0.0) for x in (3.6, 5.4, 7.1)},
         "red2H": ("red2", 0.5),
         "uso30": ("uso", 30, US), "uso60": ("uso", 60, US),
         "uso30all": ("uso", 30, INDICES), "uso60all": ("uso", 60, INDICES),
         "stop2": ("stopn", 2), "stop3": ("stopn", 3)}


def ftmo_day(t):
    return t.tz_convert("Europe/Prague").normalize()


def us_open(t):
    ny = t.tz_convert("America/New_York")
    return ny.normalize() + pd.Timedelta(hours=9, minutes=30)


def apply(rows, rule):
    """rows sorted by entry; returns rows with r scaled (0 = skipped). Uses only trades closed before each entry."""
    out = []
    if rule is None:
        return [dict(x) for x in rows]
    kind = rule[0]
    closed = []          # (end, r_scaled, day, sym, dir)
    for x in rows:
        t = x["t"]
        d = ftmo_day(t)
        w = 1.0
        if kind == "day":
            done = sum(c[1] for c in closed if c[0] <= t and c[2] == d)
            if done <= -rule[1]:
                w = rule[2]
        elif kind == "red2":
            tot = {}
            for c in closed:
                if c[0] <= t and c[2] < d:
                    tot[c[2]] = tot.get(c[2], 0.0) + c[1]
            days = sorted(tot)
            if len(days) >= 2 and tot[days[-1]] < 0 and tot[days[-2]] < 0:
                w = rule[1]
        elif kind == "uso":
            if x["sym"] in rule[2]:
                o = us_open(t)
                if o <= t.tz_convert("America/New_York") < o + pd.Timedelta(minutes=rule[1]):
                    w = 0.0
        elif kind == "stopn":
            n = sum(1 for c in closed if c[0] <= t and c[2] == d and c[3] == x["sym"] and c[4] == x["dir"]
                    and c[1] <= -0.9)
            if n >= rule[1]:
                w = 0.0
        y = dict(x)
        y["r"] = x["r"] * w
        y["w"] = w
        out.append(y)
        if w > 0:
            closed.append((x["end"], y["r"], ftmo_day(x["end"]), x["sym"], x["dir"]))
    return [y for y in out if y["w"] > 0]


def _curve(rows):
    if not rows:
        return {"sum": 0.0, "green": None, "dd": 0.0, "worst": 0.0, "sd": None}
    s = pd.Series([x["r"] for x in rows], index=pd.DatetimeIndex([x["t"] for x in rows]).normalize())
    day = s.groupby(level=0).sum()
    eq = np.cumsum([x["r"] for x in sorted(rows, key=lambda z: z["t"])])
    return {"sum": round(float(s.sum()), 2), "green": round(100 * float((day > 0).mean()), 1),
            "dd": round(float((np.maximum.accumulate(np.maximum(eq, 0)) - eq).max()), 2),
            "worst": round(float(day.min()), 2), "sd": round(float(day.std()), 3)}


def main():
    rows = [x for x in trades("now|base") if x["t"] >= FROM]
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "way": "now|base", "n": len(rows),
           "rules": {}, "curve": {}, "day_19_11": {}}
    for name, rule in RULES.items():
        k = apply(rows, rule)
        tr = [x for x in k if x["t"] < TG.SPLIT]
        un = [x for x in k if x["t"] >= TG.SPLIT]
        rep["rules"][name] = {"train": _st(tr), "unseen": _st(un)}
        rep["curve"][name] = {"train": _curve(tr), "unseen": _curve(un)}
        d19 = [x for x in k if ftmo_day(x["t"]) == pd.Timestamp("2026-11-19", tz="Europe/Prague")]
        rep["day_19_11"][name] = round(sum(x["r"] for x in d19), 2)
    now = rep["rules"]["now"]
    rep["passed"] = [n for n, v in rep["rules"].items() if n != "now"
                     and v["train"]["sum"] >= now["train"]["sum"] and v["unseen"]["sum"] >= now["unseen"]["sum"]
                     and v["train"]["dd"] <= now["train"]["dd"] and v["unseen"]["dd"] <= now["unseen"]["dd"]]
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, v in rep["rules"].items():
        print(f"{n:10s} 2025 {v['train']['sum']:8.1f}R dd {v['train']['dd']:6.1f} worst {v['train']['worst']:6.1f} "
              f"| 2026 {v['unseen']['sum']:8.1f}R dd {v['unseen']['dd']:6.1f} worst {v['unseen']['worst']:6.1f} "
              f"green {v['unseen']['green']} | 19.11 {rep['day_19_11'][n]}")
    print("passed:", rep["passed"])
    return rep


if __name__ == "__main__":
    main()
