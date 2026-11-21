"""A super bot (user, 20.11.2026 19:13 + 19:22 "yes"): one FTMO account that holds every strategy that already passed
its tests in the bots - bot 5's gap tracks (as they run since 20.11, FTMO's real spreads), bot 5ב's scout strategies,
the strategies of bots 1-4 and bot 7 - and divides the risk between them. Nothing here picks on 2026: every weight and
the account's scale are chosen on 30.04-31.12.2025 and the result is shown on 2026.

Streams: one per strategy x instrument (a combo, or a gap track x instrument); a combo held by two bots counts once.
Allocations (risk a trade = scale x weight; the scale = the largest that keeps 2025's deepest fall at 7% and its worst
day at 3% - FTMO's 10% / 5% with room):
  gap_alone       bot 5's gap tracks only (today's bot 5, for reference)
  equal           every stream the same risk
  bucket          every bot (gap, 5ב, bot 1, 2, 3, 4, 7) the same risk budget, split equally inside it
  bucket_vol      the budget by the bot's 2025 daily volatility (risk parity between the bots)
  pos2025         only streams that earned in 2025, equal
  bucket_pos2025  only streams that earned in 2025, the bucket split
Account replay: the bot's guards - a new trade risks at most the room under 4.5% a day / 9% in all (open risk counted),
under a quarter of its risk left -> skipped. Return, deepest fall, worst day, days near the limit, per year.
Results only - nothing in the bots changes.
  python -m agents.superbot_study
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import conflict_study as CS
from agents import tal_costx2_study as C2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "superbot.json"
FROM, SPLIT = pd.Timestamp("2025-04-30", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
DD_T, DAY_T = 7.0, 3.0
DAY, TOTAL = 0.045, 0.09


def streams():
    """[(t0, t1, r, stream, bucket)] over both periods."""
    rows = []
    # bot 5's gap tracks as they run (tal_costx base + the tuned JP225 / US500 / EU50, the tracks off left out)
    base, tuned = C2._rows(C2.BASE, "now|base"), C2._rows(C2.M.OUT, "tuned")
    for kk, xs in base.items():
        tk, s = kk.split("#")
        if not C2.S.live_pair(tk, s) or tk in C2.OFF:
            continue
        for x in tuned.get(kk, xs) if s in C2.TUNED else xs:
            t0 = pd.Timestamp(x["time"])
            rows.append((t0, t0 + pd.Timedelta(minutes=int(x["mins"]) + 1), float(x["r"]), f"gap:{kk}", "gap"))
    seen = set()
    for bot, bucket in (("bot5", "5b"), ("bot1", "bot1"), ("bot2", "bot2"), ("bot3", "bot3"), ("bot4", "bot4"),
                        ("bot7", "bot7")):
        t = CS.load(bot)
        for c, g in t.groupby("combo"):
            if c in seen:
                continue
            seen.add(c)
            a = pd.to_datetime(g["a"].to_numpy() * 60, unit="s", utc=True)
            b = pd.to_datetime(g["b"].to_numpy() * 60, unit="s", utc=True)
            for t0, t1, r in zip(a, b, g["r"].to_numpy(float)):
                rows.append((t0, max(t1, t0 + pd.Timedelta(minutes=1)), float(r), c, bucket))
    return [x for x in rows if x[0] >= FROM]


def _arr(tr):
    """Trades -> arrays: t0, t1 (ns), r, stream, the Prague day of t0 / t1."""
    t0 = pd.DatetimeIndex([x[0] for x in tr])
    t1 = pd.DatetimeIndex([x[1] for x in tr])
    d0 = t0.tz_convert("Europe/Prague").normalize().asi8
    d1 = t1.tz_convert("Europe/Prague").normalize().asi8
    return {"t0": t0.asi8, "t1": t1.asi8, "r": np.array([x[2] for x in tr]), "s": [x[3] for x in tr], "d0": d0, "d1": d1}


def account(A, w, scale):
    """A from _arr (sorted by t0); w[stream] weight; risk a trade = scale x weight (fraction of the balance)."""
    bal = peak = init = 100_000.0
    open_risk, heap = 0.0, []
    day, day_start, day_pnl, worst, near, dd, taken, skipped = None, bal, 0.0, 0.0, 0, 0.0, 0, 0
    t0s, t1s, rs, ss, d0s, d1s = A["t0"], A["t1"], A["r"], A["s"], A["d0"], A["d1"]
    for i in range(len(rs)):
        rk_w = w.get(ss[i], 0.0)
        if rk_w <= 0:
            continue
        t0 = t0s[i]
        while heap and heap[0][0] <= t0:
            _, _, pnl, rk, dy = heapq.heappop(heap)
            open_risk -= rk
            bal += pnl
            if dy == day:
                day_pnl += pnl
            if bal > peak:
                peak = bal
            dd = max(dd, (peak - bal) / peak)
        if d0s[i] != day:
            if day is not None:
                worst = min(worst, day_pnl / day_start)
                near += day_pnl / day_start <= -0.03
            day, day_start, day_pnl = d0s[i], bal, 0.0
        want = bal * scale * rk_w
        room = min(DAY * init + day_pnl - open_risk, bal - open_risk - init * (1 - TOTAL))
        if room < 0.25 * want:
            skipped += 1
            continue
        rk = min(want, room)
        open_risk += rk
        taken += 1
        heapq.heappush(heap, (t1s[i], i, rs[i] * rk, rk, d1s[i]))
    while heap:
        _, _, pnl, rk, dy = heapq.heappop(heap)
        bal += pnl
        if dy == day:
            day_pnl += pnl
        peak = max(peak, bal)
        dd = max(dd, (peak - bal) / peak)
    if day is not None:
        worst = min(worst, day_pnl / day_start)
    return {"ret": round(100 * (bal / init - 1), 1), "dd": round(100 * dd, 2), "worst_day": round(100 * worst, 2),
            "near_days": int(near), "taken": taken, "skipped": skipped}


def norm(w, A):
    """Weights scaled so the trade-weighted mean over 2025's trades is 1 (the scale = the mean risk a trade)."""
    v = [w.get(s, 0.0) for s in A["s"] if w.get(s, 0.0) > 0]
    m = float(np.mean(v)) if v else 1.0
    return {s: x / m for s, x in w.items()}


def fit_scale(tr25, w):
    """The largest scale with 2025's deepest fall <= DD_T and worst day >= -DAY_T (bisection)."""
    lo, hi = 0.0, 0.02
    for _ in range(14):
        mid = (lo + hi) / 2
        a = account(tr25, w, mid)
        if a["dd"] <= DD_T and a["worst_day"] >= -DAY_T:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    rows = sorted(streams(), key=lambda x: x[0])
    L25 = [x for x in rows if x[0] < SPLIT]
    L26 = [x for x in rows if x[0] >= SPLIT]
    tr25, tr26 = _arr(L25), _arr(L26)
    sb = {}
    for _, _, r, s, b in rows:
        sb[s] = b
    r25 = {}
    for _, _, r, s, _b in L25:
        r25[s] = r25.get(s, 0.0) + r
    buckets = sorted(set(sb.values()))
    by_b = {b: [s for s in sb if sb[s] == b] for b in buckets}
    # the buckets' 2025 daily R volatility (equal weight inside)
    vol = {}
    for b in buckets:
        d = {}
        for t0, t1, r, s, bb in L25:
            if bb == b:
                k = t1.tz_convert("Europe/Prague").date()
                d[k] = d.get(k, 0.0) + r / max(1, len(by_b[b]))
        vol[b] = float(np.std(list(d.values()))) if len(d) > 5 else None
    W = {}
    W["gap_alone"] = {s: 1.0 for s in sb if sb[s] == "gap"}
    W["equal"] = {s: 1.0 for s in sb}
    W["bucket"] = {s: 1.0 / len(by_b[sb[s]]) for s in sb}
    iv = {b: (1.0 / vol[b]) for b in buckets if vol[b]}
    W["bucket_vol"] = {s: iv.get(sb[s], 0.0) / len(by_b[sb[s]]) for s in sb}
    pos = {s for s in sb if r25.get(s, 0.0) > 0}
    W["pos2025"] = {s: 1.0 for s in pos}
    pb = {b: [s for s in by_b[b] if s in pos] for b in buckets}
    W["bucket_pos2025"] = {s: 1.0 / len(pb[sb[s]]) for s in pos if pb[sb[s]]}
    # the certainty of the edge (user, 21.11.2026: "the strategies with the highest certainty of profit"): each
    # stream's 2025 t (mean / its error over its trades); only t >= 1, the weight ~ t (capped at 4), and the same
    # inside the bucket split
    rr = {}
    for _, _, r, s, _b in L25:
        rr.setdefault(s, []).append(r)
    tt = {}
    for s, v in rr.items():
        if len(v) >= 20 and np.std(v) > 0:
            tt[s] = float(np.mean(v) / (np.std(v, ddof=1) / np.sqrt(len(v))))
    for th in (1.0, 2.0):
        W[f"t{th:g}"] = {s: min(4.0, t) for s, t in tt.items() if t >= th}
        tb = {b: sum(min(4.0, tt[s]) for s in by_b[b] if tt.get(s, 0) >= th) for b in buckets}
        W[f"t{th:g}_bucket"] = {s: min(4.0, tt[s]) / tb[sb[s]] for s in sb if tt.get(s, 0) >= th and tb[sb[s]] > 0}
        W[f"t{th:g}_bucket_vol"] = {s: iv.get(sb[s], 0.0) * min(4.0, tt[s]) / tb[sb[s]] for s in sb
                                    if tt.get(s, 0) >= th and tb[sb[s]] > 0}
    res = {}
    for name, w in W.items():
        w = norm(w, tr25)
        k = fit_scale(tr25, w)
        res[name] = {"scale": k, "streams": len(w), "2025": account(tr25, w, k), "2026": account(tr26, w, k),
                     "buckets": sorted({sb[s] for s in w})}
        print(name, res[name], flush=True)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "streams": {b: len(by_b[b]) for b in buckets}, "pos2025": {b: len(pb[b]) for b in buckets},
           "vol2025": vol, "allocations": res}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
