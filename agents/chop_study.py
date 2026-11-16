"""Open trades while the instrument goes sideways (user, 16.11.2026: gold chopped all day, most trades lost - "maybe take
open trades out partially, knowing it is choppy, until the chart picks a direction").

Each bot's strategies as they trade now, with their prices (agents/conflict_study.py run - every instrument), and the
instrument's 5-minute bars. "Choppy" is read on the 1-hour chart, known at the hour's close (so from the next hour on):
  adx   ADX(14) under 20
  ci    Choppiness Index(14) over 61.8
While a trade is open and the chart is choppy:
  asis          as today
  no_new        no new trade opens while choppy (the bots' regime filters already try this per strategy - adx_range /
                adx_trend - this is the same thing on the whole bot)
  be@X          once the trade is X R in profit while choppy, its stop moves to the entry
  half@X        once the trade is X R in profit while choppy, half is taken there and the rest's stop moves to the entry
X = 0.3 / 0.5 / 1.0. After the change the trade ends at the entry if the price comes back there before the trade's own
exit (the exit's bar left out), otherwise as it ended; the costs stay its own. Whole bot with the fixed risk rule
(portfolio_check.replay); chosen on 30.04-31.12.2025, shown on 2026. Nothing in the bots changes here.

  python -m agents.chop_study plan --bot bot1,bot2,bot3,bot4,bot5
  python -m agents.chop_study report --bot bot1,bot2,bot3,bot4,bot5
(the run itself: python -m agents.conflict_study run --job bot:SYMBOL)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agents import conflict_study as CS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "chop_study.json"
TH = (0.3, 0.5, 1.0)
WAYS = ["asis", "no_new"] + [f"{k}@{x:g}" for k in ("be", "half") for x in TH]


def _adx(h, l, c, n=14):
    from agents.rules_agent import adx
    return adx(h, l, c, n)


def _ci(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h, pc) - np.minimum(l, pc)
    s = pd.Series(tr).rolling(n).sum()
    rng = pd.Series(h).rolling(n).max() - pd.Series(l).rolling(n).min()
    return (100 * np.log10(s / rng.replace(0, np.nan)) / np.log10(n)).to_numpy()


def chop_flags(b):
    """5-minute bars (m, h, l, c) -> per 5-minute bar: is the 1-hour chart choppy (by the last CLOSED hour)?"""
    t = pd.to_datetime(b["m"], unit="m", utc=True)
    df = pd.DataFrame({"h": b["h"].to_numpy(), "l": b["l"].to_numpy(), "c": b["c"].to_numpy()}, index=t)
    hr = df.resample("1h").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    a = _adx(hr["h"].to_numpy(), hr["l"].to_numpy(), hr["c"].to_numpy())
    ci = _ci(hr["h"].to_numpy(), hr["l"].to_numpy(), hr["c"].to_numpy())
    known = hr.index + pd.Timedelta(hours=1)                 # an hour's value is known at its close
    f = pd.DataFrame({"adx": a < 20, "ci": ci > 61.8}, index=known)
    f = f.reindex(t, method="ffill").fillna(False)
    return f["adx"].to_numpy(bool), f["ci"].to_numpy(bool)


def manage(x, bars, flag, kind, th):
    """One trade under a chop rule -> its new r."""
    m, h, lo = bars
    i, j = np.searchsorted(m, x["a"]), np.searchsorted(m, x["b"] - 5)
    if j <= i:
        return x["r"]
    d, e, rk = x["dir"], x["entry_px"], x["risk_px"]
    fav = ((h[i:j] - e) if d > 0 else (e - lo[i:j])) / rk
    trig = np.flatnonzero(flag[i:j] & (fav >= th))
    if not len(trig):
        return x["r"]
    k = i + trig[0] + 1                                       # from the next bar on
    back = bool((lo[k:j] <= e).any()) if d > 0 else bool((h[k:j] >= e).any())
    rest = 0.0 if back else x["r"] + x["cost_r"]              # the rest, before costs
    if kind == "be":
        return rest - x["cost_r"]
    return 0.5 * th + 0.5 * rest - x["cost_r"]


def simulate(t, way, det, flags, bars):
    out = []
    for x in t.to_dict("records"):
        x["share"] = 1.0
        s = x["sym"]
        if way != "asis" and s in flags:
            fl = flags[s][0 if det == "adx" else 1]
            if way == "no_new":
                i = np.searchsorted(bars[s][0], x["a"], side="right") - 1
                if i >= 0 and fl[i]:
                    continue
            else:
                kind, th = way.split("@")
                x["r"] = manage(x, bars[s], fl, kind, float(th))
        out.append(x)
    return out


def bot_report(bot):
    from agents import portfolio_check as PC
    import agent_portfolio as A
    t = CS.load(bot)
    bars, flags = {}, {}
    for f in CS.out_dir(bot).glob("*.bars.csv.gz"):
        b = pd.read_csv(f)
        if "c" not in b:
            continue
        s = f.name.split(".bars")[0]
        bars[s] = (b["m"].to_numpy(), b["h"].to_numpy(), b["l"].to_numpy())
        flags[s] = chop_flags(b)
    t = t[t["sym"].isin(bars)]
    risk, mults = PC.bot_risk(bot), PC.mults(bot)
    split = int((pd.Timestamp(A.SPLIT, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1))
    per = {"train": t[t["a"] < split], "unseen": t[t["a"] >= split]}
    res = {"strategies": int(t["combo"].nunique()), "symbols": sorted(bars), "ways": {}}
    chop_share = {s: round(100 * float(f[1].mean()), 1) for s, f in flags.items()}
    res["chop_share_ci"] = chop_share
    for det in ("adx", "ci"):
        for way in WAYS:
            if way == "asis" and det == "ci":
                continue
            key = way if way == "asis" else f"{det}:{way}"
            row = {}
            for p, tt in per.items():
                rep = PC.replay(CS._book(simulate(tt, way, det, flags, bars), mults), risk)
                row[p] = {k: rep[k] for k in ("trades", "total", "worst_day", "max_dd")}
            res["ways"][key] = row
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
            out["bots"][b] = bot_report(b)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "report"])
    ap.add_argument("--bot", default="bot1,bot2,bot3,bot4,bot5")
    a = ap.parse_args()
    bots = [x for x in a.bot.split(",") if x]
    if a.mode == "plan":
        json.dump([f"{b}:{s}" for b in bots for s in CS.plan(b, 1)], sys.stdout)
    else:
        r = report(bots)
        print(json.dumps({b: v["best_train"] for b, v in r["bots"].items()}, ensure_ascii=False, indent=1))
