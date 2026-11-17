"""Managing the bots on days like 16.11.2026 (user, 17.11.2026: "always there is something to do - chop is not an
answer"; on 16.11 46 of 64 closed trades lost, gold alone -$3,594 with up to three strategies of one bot selling it
at the same hour, and bot 5's SC071 US30 lost four buys in a row).

The ways tested on each bot as a whole (its strategies as they trade now, with the instrument's 5-minute bars - the
data of a chop_study run, agents/conflict_study.py run), with the fixed risk rule (portfolio_check.replay):
  asis          as today
  cap@N         at most N open trades of the bot on the same instrument in the same direction (N = 1, 2)
  streak@N      after N losing trades closed on an instrument the same day (UTC), no new trade on it that day (N = 2, 3)
  halfchop      a new trade while the 1-hour chart is choppy (Choppiness(14) > 61.8) opens at half its risk
  time@K        a trade that has not been +0.5R in its favour after K of its own bars leaves at that bar's close
                (K = 4, 8)
  trail@H       structure trailing stop: the stop follows the lowest low (a sell: highest high) of the last H closed
                hours, only forward, from the moment it is past the entry's stop (H = 3, 6)
Chosen on 30.04.2025-31.12.2025, shown on 2026. Nothing in the bots changes here.

  python -m agents.mgmt_study report --bot bot1,bot2,bot3,bot4,bot5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agents import chop_study as CH
from agents import conflict_study as CS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "mgmt_study.json"
WAYS = ["asis", "cap@1", "cap@2", "streak@2", "streak@3", "halfchop", "time@4", "time@8", "trail@3", "trail@6"]


def hour_extremes(b, n):
    """Per 5-minute bar: the lowest low / highest high of the last n CLOSED hours."""
    t = pd.to_datetime(b["m"], unit="m", utc=True)
    df = pd.DataFrame({"h": b["h"].to_numpy(), "l": b["l"].to_numpy()}, index=t)
    hr = df.resample("1h").agg({"h": "max", "l": "min"}).dropna()
    r = pd.DataFrame({"hi": hr["h"].rolling(n).max().to_numpy(), "lo": hr["l"].rolling(n).min().to_numpy()},
                     index=hr.index + pd.Timedelta(hours=1)).reindex(t, method="ffill")
    return r["hi"].to_numpy(), r["lo"].to_numpy()


def time_exit(x, bars, k):
    m, h, lo, c = bars
    i, j = np.searchsorted(m, x["a"]), np.searchsorted(m, x["b"] - 5)
    tcut = x["a"] + k * int(x["tf"])
    kk = np.searchsorted(m, tcut)
    if kk >= j or kk <= i:
        return x["r"], None
    d, e, rk = x["dir"], x["entry_px"], x["risk_px"]
    fav = ((h[i:kk] - e) if d > 0 else (e - lo[i:kk])) / rk
    if len(fav) and fav.max() >= 0.5:
        return x["r"], None
    return d * (c[kk] - e) / rk - x["cost_r"], int(m[kk])


def trail_exit(x, bars, ext):
    m, h, lo, c = bars
    i, j = np.searchsorted(m, x["a"]), np.searchsorted(m, x["b"] - 5)
    if j <= i:
        return x["r"], None
    d, e, rk, st = x["dir"], x["entry_px"], x["risk_px"], x["stop_px"]
    hi_, lo_ = ext
    stop = st
    for k in range(i, j):
        lvl = lo_[k] if d > 0 else hi_[k]
        if k > 0 and np.isfinite(lvl) and (lvl - stop) * d > 0 and (c[k - 1] - lvl) * d > 0:
            stop = lvl
        if (stop - st) * d > 0 and ((lo[k] <= stop) if d > 0 else (h[k] >= stop)):
            return d * (stop - e) / rk - x["cost_r"], int(m[k])
    return x["r"], None


def simulate(t, way, bars, flags, exts):
    out, opened = [], []
    losses = {}                                   # (sym, day) -> [exit minutes of losing trades]
    kind, _, arg = way.partition("@")
    for x in t.to_dict("records"):
        x["share"] = 1.0
        s = x["sym"]
        if kind == "cap":
            opened = [y for y in opened if y["b"] > x["a"]]
            if sum(1 for y in opened if y["sym"] == s and y["dir"] == x["dir"]) >= int(arg):
                continue
        if kind == "streak":
            day = x["a"] // 1440
            if sum(1 for b in losses.get((s, day), []) if b <= x["a"]) >= int(arg):
                continue
        if kind == "halfchop" and s in flags:
            i = np.searchsorted(bars[s][0], x["a"], side="right") - 1
            if i >= 0 and flags[s][1][i]:
                x["share"] = 0.5
        if kind == "time" and s in bars:
            r, b = time_exit(x, bars[s], int(arg))
            x["r"] = r
            if b:
                x["b"] = max(b, x["a"] + 1)
        if kind == "trail" and s in bars:
            r, b = trail_exit(x, bars[s], exts[s][int(arg)])
            x["r"] = r
            if b:
                x["b"] = max(b, x["a"] + 1)
        out.append(x)
        opened.append(x)
        if x["r"] < 0:
            losses.setdefault((s, x["b"] // 1440), []).append(x["b"])
    return out


def bot_report(bot):
    from agents import portfolio_check as PC
    import agent_portfolio as A
    t = CS.load(bot)
    bars, flags, exts = {}, {}, {}
    for f in CS.out_dir(bot).glob("*.bars.csv.gz"):
        b = pd.read_csv(f)
        if "c" not in b:
            continue
        s = f.name.split(".bars")[0]
        bars[s] = (b["m"].to_numpy(), b["h"].to_numpy(), b["l"].to_numpy(), b["c"].to_numpy())
        flags[s] = CH.chop_flags(b)
        exts[s] = {3: hour_extremes(b, 3), 6: hour_extremes(b, 6)}
    t = t[t["sym"].isin(bars)].copy()
    cfg = PC.bot_cfg(bot)
    if isinstance(cfg.get("risk"), dict):          # bot 5: a risk per strategy
        rk = cfg["risk"]
        risk = 1.0
        mults = {c: float(rk.get("scout:" + c) or rk.get(c) or 0.0018) for c in t["combo"].unique()}
    else:
        risk, mults = PC.bot_risk(bot), PC.mults(bot)
    split = int((pd.Timestamp(A.SPLIT, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1))
    per = {"train": t[t["a"] < split], "unseen": t[t["a"] >= split]}
    res = {"strategies": int(t["combo"].nunique()), "ways": {}}
    for way in WAYS:
        row = {}
        for p, tt in per.items():
            taken = simulate(tt, way, bars, flags, exts)
            rep = PC.replay(CS._book(taken, mults), risk)
            row[p] = {k: rep[k] for k in ("trades", "total", "worst_day", "max_dd")}
            # the losing days: the 10% worst days of "as today" - how much each way changes them
        res["ways"][way] = row
    a = res["ways"]["asis"]
    best = max((k for k in res["ways"] if k != "asis"), key=lambda k: res["ways"][k]["train"]["total"])
    res["best_train"] = {"way": best, "train": round(res["ways"][best]["train"]["total"] - a["train"]["total"], 2),
                         "unseen": round(res["ways"][best]["unseen"]["total"] - a["unseen"]["total"], 2),
                         "dd_unseen": round(res["ways"][best]["unseen"]["max_dd"] - a["unseen"]["max_dd"], 2)}
    return res


def report(bots):
    out = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "bots": {}}
    for b in bots:
        if CS.out_dir(b).exists():
            try:
                out["bots"][b] = bot_report(b)
            except Exception as e:  # noqa: BLE001
                out["bots"][b] = {"error": f"{type(e).__name__}: {e}"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report"])
    ap.add_argument("--bot", default="bot1,bot2,bot3,bot4,bot5")
    a = ap.parse_args()
    r = report([x for x in a.bot.split(",") if x])
    print(json.dumps({b: v.get("best_train", v.get("error")) for b, v in r["bots"].items()}, ensure_ascii=False, indent=1))
