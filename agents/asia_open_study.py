"""Bot 5 - the Asian index's reopen gap after a big US session (bot 5's smart agent, 20.11.2026; the idea is bot 7's shock
agent's note in data/agents/strategies/tal_gap.md, 20.11 early morning: after Wall Street fell 2%+ on 19.11, JP225 reopened
-4% to -4.5% - the gap comes from the US session, not from news of the night). Results only - no card is built from here.

The exact rules (each knob in the grid below, chosen on the period to 2025-12-31 only, shown on 2026):
- the reopen: the first 1-minute bar of the Asian index (JP225 / AUS200) after a pause of 40 minutes or more in its bars
  (the daily break; a weekend reopen counts too - nothing is held over a break, the trade opens after it);
- the US move: the driver (US100 / US500) from its first bar at or after 09:30 New York of that day to its last bar
  before the Asian index's pause, in %; the event needs |US move| >= MOVE and the reopen gap (the reopen bar's open
  against the last close before the pause) at least GAP % of price, in the US move's direction ("same") or either way
  ("any");
- the entry: at the close of the K-th bar after the reopen (K = 1 / 5), not inside a news mask;
- "cont": in the gap's direction, the stop just beyond the last close before the pause (a filled gap proves it wrong, 2
  points of room), the target RR x the stop's distance;
- "fade": against the gap, the target the last close before the pause (the gap filled), the stop S x the target's
  distance on the other side;
- the exit: stop / target on 1-minute bars (both in one bar = the stop), or the bar the FTMO session / news mask closes
  at (agent_bot5.prep's stop_at: the next break, the session's end, a red release) - no time exit;
- costs: FTMO's recorded 1-minute spread + commission at the entry (TAL_COSTX=1: bot 1's real FTMO spreads), in R.
The account: agents/tal_fine_study.account (0.28% a trade, 4.5% a day / 9% total), per period.
  python -m agents.asia_open_study run | report
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import agent_bot5 as A
from agents import agentkit as K
from agents import tal_fine_study as F

OUT = Path("reports/agents/bot5/asia_open.json")
ASIA = ("JP225", "AUS200")
DRIVERS = ("US100", "US500")
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
PAUSE_MIN = 40
MOVES = (0.5, 1.0, 1.5, 2.0)              # |US move| in %
GAPS = (0.1, 0.3, 0.6)                    # |reopen gap| in % of price
SIDES = ("same", "any")
KS = (1, 5)
WAYS = [("cont", x) for x in (1.0, 1.5, 2.0)] + [("fade", x) for x in (0.5, 1.0, 1.5)]
ROOM = 2.0                                # points beyond the last close (index points)
MIN_N = 15                                # trades to 2025 for a version to be chosen


def _utc(ix):
    ix = pd.DatetimeIndex(ix)
    ix = ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")
    return ix.as_unit("ns")                 # the bars come in microseconds: asi8 below is in nanoseconds


def _load(s, end):
    d = A.prep(s, end)
    if d is None:
        return None
    df, cost, stop_at, noent = d
    df = df.copy()
    df.index = _utc(df.index)
    return df, np.asarray(cost, float), np.asarray(stop_at, bool), np.asarray(noent, bool)


def events(asia, drv):
    """[(i, pre, gap %, us move %)] - every reopen of the Asian index with the driver's New York session before it."""
    df, _, _, _ = asia
    t = df.index
    o, c = df["o"].to_numpy(float), df["c"].to_numpy(float)
    dt = np.r_[np.inf, np.diff(t.asi8) / 6e10]
    dd = drv[0]
    dny = dd.index.tz_convert("America/New_York")
    out = []
    for i in np.where(dt >= PAUSE_MIN)[0]:
        if i == 0 or i + 6 >= len(df):
            continue
        last = t[i - 1]
        ny = last.tz_convert("America/New_York")
        start = ny.normalize() + pd.Timedelta(hours=9, minutes=30)
        if ny < start:
            start -= pd.Timedelta(days=1)
        a = dny.searchsorted(start)
        b = dny.searchsorted(ny, side="right") - 1
        if a >= len(dd) or b <= a or (dny[b] - start) < pd.Timedelta(hours=3):
            continue
        mv = (dd["c"].iat[b] / dd["o"].iat[a] - 1) * 100
        pre = c[i - 1]
        out.append((int(i), float(pre), (o[i] / pre - 1) * 100, float(mv)))
    return out


def trade(asia, ev, way, x, k):
    """One trade on one event -> (t_open, t_close, r) or None."""
    df, cost, stop_at, noent = asia
    i, pre, gap, _ = ev
    h, lo, c = df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    e = i + k - 1
    if e >= len(df) - 1 or noent[e] or stop_at[e]:
        return None
    px = c[e]
    g = np.sign(gap)
    if way == "cont":
        d = g
        stop = pre - d * ROOM
        risk = (px - stop) * d
        if risk <= 0:
            return None                          # the gap already filled before the entry
        tgt = px + d * x * risk
    else:
        d = -g
        dist = (pre - px) * d
        if dist <= 0:
            return None                          # already filled
        tgt, risk = pre, x * dist
        stop = px - d * risk
    if risk <= 2 * cost[e]:
        return None                              # a stop inside the costs is not a trade
    ex, j = None, e + 1
    while j < len(df):
        hit_s = lo[j] <= stop if d > 0 else h[j] >= stop
        hit_t = h[j] >= tgt if d > 0 else lo[j] <= tgt
        if hit_s:
            ex = stop
            break
        if hit_t:
            ex = tgt
            break
        if stop_at[j]:
            ex = c[j]
            break
        j += 1
    if ex is None:
        return None
    r = (ex - px) * d / risk - cost[e] / risk
    return df.index[e], df.index[min(j, len(df) - 1)], float(r)


def run():
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t0 = time.time()
    data = {s: _load(s, end) for s in ASIA + DRIVERS}
    rows = {}
    for s, dv in itertools.product(ASIA, DRIVERS):
        if data[s] is None or data[dv] is None:
            rows[f"{s}<{dv}"] = {"err": "no data"}
            continue
        evs = events(data[s], data[dv])
        vers = []
        for mv, gp, sd, k, (way, x) in itertools.product(MOVES, GAPS, SIDES, KS, WAYS):
            tr = []
            for ev in evs:
                if abs(ev[3]) < mv or abs(ev[2]) < gp or (sd == "same" and np.sign(ev[2]) != np.sign(ev[3])):
                    continue
                z = trade(data[s], ev, way, x, k)
                if z:
                    tr.append(z)
            vers.append({"move": mv, "gap": gp, "side": sd, "k": k, "way": way, "x": x, "trades": tr})
        rows[f"{s}<{dv}"] = {"events": len(evs), "versions": vers}
        print(f"{s}<{dv}: {len(evs)} reopens, {len(vers)} versions, {time.time() - t0:.0f}s")
    return rows


def _st(tr):
    if not tr:
        return K.wstats([])
    tt = [a for a, _, _ in tr]
    st = K.wstats([r for _, _, r in tr], K.weights(tt))
    st["sum"] = round(float(sum(r for _, _, r in tr)), 2)
    return st


def report(rows):
    res = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "rules": __doc__.split("The account")[0].strip(),
           "pairs": {}}
    for key, x in rows.items():
        if "versions" not in x:
            res["pairs"][key] = x
            continue
        vs = []
        for v in x["versions"]:
            tr = [z for z in v["trades"] if z[0] < SPLIT]
            un = [z for z in v["trades"] if z[0] >= SPLIT]
            vs.append({**{k: v[k] for k in ("move", "gap", "side", "k", "way", "x")}, "train": _st(tr), "unseen": _st(un),
                       "acct": {"2025": F.account(tr), "2026": F.account(un)}})
        ok = [v for v in vs if (v["train"]["n"] or 0) >= MIN_N and v["train"]["mean"] is not None]
        best = max(ok, key=lambda v: v["train"]["mean"] * v["train"]["eff_n"]) if ok else None
        nb = []
        if best:
            for v in vs:
                diff = sum(v[k] != best[k] for k in ("move", "gap", "side", "k", "x")) + (v["way"] != best["way"]) * 9
                if diff == 1 and (v["train"]["n"] or 0) >= MIN_N:
                    nb.append(v)
        pos = [v for v in ok if (v["train"]["mean"] or 0) > 0 and (v["unseen"]["mean"] or 0) > 0]
        res["pairs"][key] = {"events": x["events"], "tested": len(vs), "with_n": len(ok), "chosen": best,
                             "neighbours_unseen_pos": (sum((v["unseen"]["mean"] or 0) > 0 for v in nb), len(nb)),
                             "both_periods_pos": len(pos),
                             "top_train": sorted(ok, key=lambda v: -v["train"]["mean"] * v["train"]["eff_n"])[:10]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for key, p in res["pairs"].items():
        b = p.get("chosen") or {}
        print(key, p.get("events"), "tested", p.get("tested"), "both+", p.get("both_periods_pos"),
              "chosen", {k: b.get(k) for k in ("move", "gap", "side", "k", "way", "x")},
              "train", b.get("train"), "unseen", b.get("unseen"), "nb", p.get("neighbours_unseen_pos"))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    a = ap.parse_args()
    report(run())
