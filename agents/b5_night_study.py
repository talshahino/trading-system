"""Bot 5's scout strategies - three questions of 16.11.2026 on the same trades (results only, nothing in the bot changes).

1. Stacking (manager, task mgr1611stack): 16.11 SC441 and SC402 bought XAUUSD in the same minute at the same price while
   SC042's buy was still open, and all lost. When a scout trade is already open on the same instrument in the same
   direction:  skip = the new one is not opened / half = the new one at half risk / cap2 = at most 2 open together.
2. SC071|15Min|US30 direction (user, task b5sc071dir1611: "a wrong read - it should have been a short"): the strategy
   only buys at the lower Keltner band. A higher-timeframe direction filter on its buys, from closed bars only:
   ema50h = buy only when the last closed 1-hour close is over the 1-hour EMA50; slope20h = only when the 1-hour EMA20
   is not falling (over its value 3 hours before); both = the two together; noreent = no buy again on the same day
   after one of its buys was stopped.
   The same trend filter (in the trade's direction: longs over / shorts under the 1-hour EMA50) on every scout strategy
   here, each one's pick chosen on 2025 and shown on 2026 (tal_thinking.md 16.11: a one-direction strategy is tested
   with a higher-timeframe direction filter before it keeps its risk).
3. The curve (user, task mgr1611curve: "many small wins every day rather than one big win"): every way reports beside
   the total the win rate, % green days, the fall from the peak, the worst day and the daily spread.

The trades: the strategies as bot 5 trades them, with prices, and the instrument's 5-minute bars - the artifacts of a
chop_study run (reports/agents/bot5/conflict, agents/conflict_study.run). The whole bot with the fixed risk rule
(portfolio_check.guarded, each strategy at its own risk). Chosen on 30.04-31.12.2025, shown on 2026.

  python -m agents.b5_night_study report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import conflict_study as CS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "night_study.json"
BOT = "bot5"
STACK = ("asis", "skip", "half", "cap2")
DIRW = ("asis", "ema50h", "slope20h", "both", "noreent")


def curve(t):
    """guarded trades (a, b, d, pnl, risk) -> the account's numbers in % of the opening balance."""
    if t.empty:
        return {"trades": 0, "total": 0.0}
    day = pd.to_datetime(t["b"], unit="m", utc=True).dt.tz_convert("Europe/Prague").dt.date
    daily = t.groupby(day)["pnl"].sum().sort_index()
    eq = np.r_[100.0, 100 + daily.cumsum().to_numpy()]
    return {"trades": int(len(t)), "total": round(float(daily.sum()), 2),
            "win": round(100 * float((t["pnl"] > 0).mean()), 1),
            "green_days": round(100 * float((daily > 0).mean()), 1), "days": int(len(daily)),
            "max_dd": round(float((np.maximum.accumulate(eq) - eq).max()), 2),
            "worst_day": round(float(min(0.0, daily.min())), 2), "day_sd": round(float(daily.std(ddof=0)), 3)}


def replay(rows, mults):
    from agents import portfolio_check as PC
    book = {}
    for x in rows:
        key = x["combo"] + ("#h" if x.get("share", 1.0) != 1.0 else "")
        book.setdefault(key, ([], float(mults.get(x["combo"], 0.0018)) * x.get("share", 1.0)))[0].append(
            [int(x["a"]), int(x["b"]), int(x["dir"]), float(x["r"])])
    t, g = PC.guarded(book, 1.0, True)
    out = curve(t)
    out.update({"guard_cut": g["cut"], "guard_skipped": g["skipped"]})
    return out


def stack(rows, way):
    """rows in time order -> the rows with the stacking rule (scout trades only; an open trade = entered before and
    not yet closed)."""
    if way == "asis":
        return [dict(x, share=1.0) for x in rows]
    out, open_ = [], []
    for x in rows:
        open_ = [o for o in open_ if o["b"] > x["a"]]
        same = [o for o in open_ if o["sym"] == x["sym"] and o["dir"] == x["dir"]]
        y = dict(x, share=1.0)
        if same:
            if way == "skip" or (way == "cap2" and len(same) >= 2):
                continue
            if way == "half":
                y["share"] = 0.5
        out.append(y)
        open_.append(y)
    return out


def hour_trend(bars):
    """5-minute bars (m, c) -> per 1-hour bar closing at minute M: close, EMA50, EMA20 and EMA20 three hours before."""
    m, c = bars["m"].to_numpy(), bars["c"].to_numpy()
    s = pd.Series(c, index=pd.to_datetime(m, unit="m", utc=True))
    h = s.resample("1h", label="left", closed="left").last().dropna()
    e50 = h.ewm(span=50, adjust=False).mean()
    e20 = h.ewm(span=20, adjust=False).mean()
    close_m = ((h.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1)).astype("int64") + 60
    return {"m": np.asarray(close_m), "c": h.to_numpy(), "e50": e50.to_numpy(), "e20": e20.to_numpy(),
            "e20p": e20.shift(3).to_numpy()}


def trend_at(ht, a):
    """The last 1-hour bar closed by minute a -> (close > EMA50, EMA20 not falling); None before enough bars."""
    i = int(np.searchsorted(ht["m"], a, side="right")) - 1
    if i < 50 or not np.isfinite(ht["e20p"][i]):
        return None
    return bool(ht["c"][i] > ht["e50"][i]), bool(ht["e20"][i] >= ht["e20p"][i])


def dir_filter(g, way, ht):
    """One strategy's trades (time order) under a direction way."""
    if way == "asis":
        return g
    keep, stopped = [], {}
    for x in g:
        if way == "noreent":
            day = pd.Timestamp(x["a"], unit="m", tz="UTC").tz_convert("America/New_York").date()
            if stopped.get((day, x["dir"]), 1e18) <= x["a"]:
                continue
            keep.append(x)
            if x["r"] < 0:
                stopped[(day, x["dir"])] = min(stopped.get((day, x["dir"]), 1e18), x["b"])
            continue
        tr = trend_at(ht, x["a"])
        if tr is None:
            keep.append(x)
            continue
        up, rising = tr
        okema = up if x["dir"] > 0 else not up
        okslope = rising if x["dir"] > 0 else not rising
        ok = {"ema50h": okema, "slope20h": okslope, "both": okema and okslope}[way]
        if ok:
            keep.append(x)
    return keep


def _sum(rows):
    r = [x["r"] * x.get("share", 1.0) for x in rows]
    return {"n": len(r), "sum": round(float(np.sum(r)), 2) if r else 0.0,
            "mean": round(float(np.mean(r)), 3) if r else None, "win": round(100 * float(np.mean(np.array(r) > 0)), 1) if r else None}


def report():
    from agents import portfolio_check as PC
    import agent_portfolio as A
    t = CS.load(BOT)
    bars = {f.name.split(".bars")[0]: pd.read_csv(f) for f in CS.out_dir(BOT).glob("*.bars.csv.gz")}
    t = t[t["sym"].isin(bars)]
    cfg = PC.bot_cfg(BOT)
    rk = cfg.get("risk") or {}
    mults = {c: float(rk.get("scout:" + c) or rk.get(c) or 0.0018) for c in t["combo"].unique()}
    split = int((pd.Timestamp(A.SPLIT, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1))
    rows = sorted(t.to_dict("records"), key=lambda x: (x["a"], x["b"], x["combo"]))
    per = {"train": [x for x in rows if x["a"] < split], "unseen": [x for x in rows if x["a"] >= split]}
    res = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "strategies": sorted(t["combo"].unique()),
           "symbols": sorted(bars), "risk": mults}
    # 1. stacking
    st = {}
    for way in STACK:
        st[way] = {p: replay(stack(rr, way), mults) for p, rr in per.items()}
        st[way]["stacked"] = {p: sum(1 for x in stack(rr, "half") if x["share"] != 1.0) for p, rr in per.items()} \
            if way == "asis" else None
    res["stack"] = st
    # 2. direction: SC071 and every scout strategy, each alone (R) and the whole bot with each one's 2025 pick
    hts = {s: hour_trend(b) for s, b in bars.items()}
    dirs, pick = {}, {}
    for c in sorted(t["combo"].unique()):
        sym = c.split("|")[2]
        g = {p: [x for x in rr if x["combo"] == c] for p, rr in per.items()}
        dirs[c] = {w: {p: _sum(dir_filter(g[p], w, hts[sym])) for p in per} for w in DIRW}
        dirs[c]["sides"] = {p: {"long": int(sum(1 for x in g[p] if x["dir"] > 0)), "short": int(sum(1 for x in g[p] if x["dir"] < 0))}
                            for p in per}
        best = max(DIRW, key=lambda w: dirs[c][w]["train"]["sum"])
        pick[c] = best if dirs[c][best]["train"]["sum"] > dirs[c]["asis"]["train"]["sum"] + 1e-9 else "asis"
        dirs[c]["pick"] = pick[c]
    res["direction"] = dirs
    whole = {}
    for label, choose in [("asis", lambda c: "asis"), ("pick_per_strategy", lambda c: pick[c]),
                          ("sc071_ema50h", lambda c: "ema50h" if c.startswith("SC071|") else "asis"),
                          ("sc071_both", lambda c: "both" if c.startswith("SC071|") else "asis"),
                          ("sc071_noreent", lambda c: "noreent" if c.startswith("SC071|") else "asis"),
                          ("all_ema50h", lambda c: "ema50h"), ("all_noreent", lambda c: "noreent"),
                          # the 2025 picks that also held on 2026 (night of 16.11: 3 of the 11 picks) - each alone and together
                          ("sc333_noreent", lambda c: "noreent" if c == "SC333|15Min|XAUUSD" else "asis"),
                          ("bk07us100_ema50h", lambda c: "ema50h" if c == "BK07|15Min|US100" else "asis"),
                          ("held_two", lambda c: {"SC333|15Min|XAUUSD": "noreent", "BK07|15Min|US100": "ema50h"}.get(c, "asis"))]:
        whole[label] = {}
        for p, rr in per.items():
            s = []
            for c in sorted({x["combo"] for x in rr}):
                s += dir_filter([x for x in rr if x["combo"] == c], choose(c), hts[c.split("|")[2]])
            whole[label][p] = replay(sorted([dict(x, share=1.0) for x in s], key=lambda x: (x["a"], x["b"])), mults)
    res["direction_whole"] = whole
    # 3. the census of 16.11 for the manager (the day's scout trades in this data, if the run reached it)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = report()
    print(json.dumps({"stack": r["stack"], "whole": r["direction_whole"],
                      "sc071": {k: v for k, v in r["direction"].items() if k.startswith("SC071")}}, ensure_ascii=False, indent=1))
