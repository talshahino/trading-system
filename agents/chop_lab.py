"""What is "choppy", how to see it early, and how to manage trend trades while it lasts (user, 17.11.2026: "check all the
measures and indicators that detect chop against the trends; understand what chop really is; how to manage the trend
trades during the chop, what is the most profitable, and find the most correct trade management").

Data: a chop_study run (every strategy of bots 1-5 as it trades now, with prices, and every instrument's 5-minute bars).

1. WHAT CHOP IS, measured: a stretch where the bots' trades do not work - the R the trades opened in it earn. Each
   detector (1-hour chart, closed hours only - nothing from the future) splits the trades into "opened while it said
   choppy" and the rest; a useful detector puts the losers on the choppy side and still leaves most of the time
   "not choppy". Detectors:
     adx      ADX(14) < 20                              ci       Choppiness Index(14) > 61.8
     er       Kaufman efficiency ratio(10) < 0.3        vhf      Vertical Horizontal Filter(28) < 0.35
     bbw      Bollinger(20,2) width in its lowest 20% of the last 100 hours
     don      Donchian(20) width < 3 ATR(14)            rsi      RSI(14) inside 40-60 for the last 6 hours
     atr      ATR(14) / ATR(100) < 0.8                  ma       EMA50 flat (|10-hour slope| < 0.1 ATR) and the close
                                                                 crossed the EMA20 3+ times in the last 12 hours
     fail     2+ failed breaks of the 24-hour high / low in the last 24 hours (broke it, closed back inside)
     adr      today's range so far < 60% of the 20-day average range (after 6 hours of the day)
     vote2 / vote3   2 / 3 of the best five on the choosing period at once
   Chosen on 30.04.2025-31.12.2025 by the gap in R per trade (not choppy - choppy) with at least 15% of the trades on
   each side; shown on 2026.
2. MANAGING TREND TRADES WHILE CHOPPY (with the chosen detector and with ci): per STRATEGY, not only per bot -
     be@X      stop to the entry once X R in profit while choppy           (0.3 / 0.5 / 1)
     half@X    half out at X R while choppy, the rest's stop to the entry  (0.3 / 0.5 / 1)
     shrink@F  the moment it turns choppy, the stop comes to F of its distance (0.5 / 0.7) - smaller loss, the trade
               can still break out
     fib@F / fibh@F   target the choppy 24-hour range's Fibonacci F (0.618 / 0.786), whole / half
     time@K    opened while choppy and not +0.5R after K of its bars -> out at that bar's close (4 / 8)
     trail@H   while choppy the stop follows the last H closed hours' low / high, only forward (3 / 6)
   Each strategy takes the way that earned the most on the choosing period (at least 20 trades and +1R or +5% over
   "as today", otherwise it stays as today); the bot with every strategy's choice is replayed whole with the fixed risk
   rule, on the choosing period and on 2026 (unseen), against "as today".
Nothing in the bots changes here.

  python -m agents.chop_lab report --bot bot1,bot2,bot3,bot4,bot5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import chop_study as CH
from agents import conflict_study as CS
from agents import mgmt_study as MG

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "chop_lab.json"
DETS = ["adx", "ci", "er", "vhf", "bbw", "don", "rsi", "atr", "ma", "fail", "adr"]
WAYS = ([f"be@{x:g}" for x in (0.3, 0.5, 1.0)] + [f"half@{x:g}" for x in (0.3, 0.5, 1.0)] + ["shrink@0.5", "shrink@0.7"]
        + ["fib@0.618", "fib@0.786", "fibh@0.618", "time@4", "time@8", "trail@3", "trail@6"])


# ---------------- detectors on the 1-hour chart ----------------
def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h, pc) - np.minimum(l, pc)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy(), tr


def _rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()


def hourly_detectors(b):
    """5-minute bars (m, h, l, c) -> {detector: bool per 5-minute bar}, each from CLOSED hours only."""
    t = pd.to_datetime(b["m"], unit="m", utc=True)
    df = pd.DataFrame({"h": b["h"].to_numpy(), "l": b["l"].to_numpy(), "c": b["c"].to_numpy()}, index=t)
    hr = df.resample("1h").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    h, l, c = (hr[x].to_numpy(dtype=float) for x in "hlc")
    S = lambda x: pd.Series(x)  # noqa: E731
    a14, tr = _atr(h, l, c, 14)
    a100, _ = _atr(h, l, c, 100)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["adx"] = CH._adx(h, l, c) < 20
        out["ci"] = CH._ci(h, l, c) > 61.8
        er = np.abs(c - np.r_[np.full(10, np.nan), c[:-10]]) / S(np.abs(np.diff(c, prepend=c[0]))).rolling(10).sum().to_numpy()
        out["er"] = er < 0.3
        hh28, ll28 = S(h).rolling(28).max().to_numpy(), S(l).rolling(28).min().to_numpy()
        vhf = (hh28 - ll28) / S(np.abs(np.diff(c, prepend=c[0]))).rolling(28).sum().to_numpy()
        out["vhf"] = vhf < 0.35
        mid = S(c).rolling(20).mean().to_numpy()
        sd = S(c).rolling(20).std(ddof=0).to_numpy()
        bw = 4 * sd / mid
        out["bbw"] = bw <= S(bw).rolling(100).quantile(0.2).to_numpy()
        out["don"] = (S(h).rolling(20).max().to_numpy() - S(l).rolling(20).min().to_numpy()) < 3 * a14
        r = _rsi(c, 14)
        out["rsi"] = S((r > 40) & (r < 60)).rolling(6).sum().to_numpy() >= 6
        out["atr"] = a14 / a100 < 0.8
        e50, e20 = _ema(c, 50), _ema(c, 20)
        flat = np.abs(e50 - np.r_[np.full(10, np.nan), e50[:-10]]) < 0.1 * a14
        side = np.sign(c - e20)
        crosses = S((side != np.r_[side[0], side[:-1]]).astype(float)).rolling(12).sum().to_numpy()
        out["ma"] = flat & (crosses >= 3)
        ph, pl = S(h).shift(1).rolling(24).max().to_numpy(), S(l).shift(1).rolling(24).min().to_numpy()
        failed = ((h > ph) & (c < ph)) | ((l < pl) & (c > pl))
        out["fail"] = S(failed.astype(float)).rolling(24).sum().to_numpy() >= 2
        day = hr.index.floor("D")
        dh, dl = S(h).groupby(day).cummax().to_numpy(), S(l).groupby(day).cummin().to_numpy()
        dr = pd.DataFrame({"h": h, "l": l}, index=hr.index).resample("1D").agg({"h": "max", "l": "min"}).dropna()
        adr = (dr["h"] - dr["l"]).rolling(20).mean().shift(1)
        adr_h = adr.reindex(day).to_numpy()
        hour_of_day = hr.index.hour.to_numpy()
        out["adr"] = (hour_of_day >= 6) & ((dh - dl) < 0.6 * adr_h)
    known = hr.index + pd.Timedelta(hours=1)
    f = pd.DataFrame({k: np.asarray(v, bool) for k, v in out.items()}, index=known).reindex(t, method="ffill")
    return {k: f[k].fillna(False).to_numpy(bool) for k in out}


# ---------------- managing while choppy ----------------
def shrink(x, bars, flag, f):
    m, h, lo, c = bars
    i, j = np.searchsorted(m, x["a"]), np.searchsorted(m, x["b"] - 5)
    if j <= i:
        return x["r"]
    on = np.flatnonzero(flag[i:j])
    if not len(on):
        return x["r"]
    k = i + on[0]
    d, e, rk = x["dir"], x["entry_px"], x["risk_px"]
    ns = e - d * f * rk
    if k > 0 and (c[k - 1] - ns) * d <= 0:                    # already beyond the smaller stop: out there
        return d * (c[k - 1] - e) / rk - x["cost_r"]
    hit = bool((lo[k:j] <= ns).any()) if d > 0 else bool((h[k:j] >= ns).any())
    return (-f - x["cost_r"]) if hit else x["r"]


def time_chop(x, bars, flag, k):
    m = bars[0]
    i = np.searchsorted(m, x["a"], side="right") - 1
    if i < 0 or not flag[i]:
        return x["r"]
    return MG.time_exit(x, bars, k)[0]


def trail_chop(x, bars, flag, ext):
    m, h, lo, c = bars
    i, j = np.searchsorted(m, x["a"]), np.searchsorted(m, x["b"] - 5)
    d, e, rk, st = x["dir"], x["entry_px"], x["risk_px"], x["stop_px"]
    hi_, lo_ = ext
    stop = st
    for k in range(i, j):
        lvl = lo_[k] if d > 0 else hi_[k]
        if k > 0 and flag[k] and np.isfinite(lvl) and (lvl - stop) * d > 0 and (c[k - 1] - lvl) * d > 0:
            stop = lvl
        if (stop - st) * d > 0 and ((lo[k] <= stop) if d > 0 else (h[k] >= stop)):
            return d * (stop - e) / rk - x["cost_r"]
    return x["r"]


def apply_way(x, way, bars, flag, rng, ext):
    kind, _, arg = way.partition("@")
    if kind in ("be", "half"):
        return CH.manage(x, bars[:3], flag, kind, float(arg))
    if kind in ("fib", "fibh"):
        return CH.fib_exit(x, bars, flag, rng, float(arg), kind == "fibh")
    if kind == "shrink":
        return shrink(x, bars, flag, float(arg))
    if kind == "time":
        return time_chop(x, bars, flag, int(arg))
    if kind == "trail":
        return trail_chop(x, bars, flag, ext[int(arg)])
    return x["r"]


# ---------------- the report ----------------
def bot_lab(bot):
    from agents import portfolio_check as PC
    import agent_portfolio as A
    t = CS.load(bot)
    bars, dets, rngs, exts = {}, {}, {}, {}
    for f in CS.out_dir(bot).glob("*.bars.csv.gz"):
        b = pd.read_csv(f)
        if "c" not in b:
            continue
        s = f.name.split(".bars")[0]
        bars[s] = (b["m"].to_numpy(), b["h"].to_numpy(), b["l"].to_numpy(), b["c"].to_numpy())
        dets[s] = hourly_detectors(b)
        rngs[s] = CH.ranges(b)
        exts[s] = {3: MG.hour_extremes(b, 3), 6: MG.hour_extremes(b, 6)}
    t = t[t["sym"].isin(bars)].copy()
    split = int((pd.Timestamp(A.SPLIT, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1))
    t["per"] = np.where(t["a"] < split, "train", "unseen")
    # each trade: which detectors said choppy at its entry
    for d in DETS:
        t[d] = False
    for s, (m, *_x) in bars.items():
        sel = t["sym"] == s
        idx = np.clip(np.searchsorted(m, t.loc[sel, "a"].to_numpy(), side="right") - 1, 0, len(m) - 1)
        for d in DETS:
            t.loc[sel, d] = dets[s][d][idx]
    # 1. detectors
    det_tab = {}
    for d in DETS:
        row = {}
        for p in ("train", "unseen"):
            g = t[t["per"] == p]
            ch, nc = g[g[d]], g[~g[d]]
            row[p] = {"share_choppy": round(100 * len(ch) / max(len(g), 1), 1), "r_choppy": round(float(ch["r"].mean()), 3) if len(ch) else None,
                      "r_other": round(float(nc["r"].mean()), 3) if len(nc) else None,
                      "gap": round(float(nc["r"].mean() - ch["r"].mean()), 3) if len(ch) and len(nc) else None}
        det_tab[d] = row
    ok = [d for d in DETS if det_tab[d]["train"]["gap"] is not None and 15 <= det_tab[d]["train"]["share_choppy"] <= 85]
    top = sorted(ok, key=lambda d: -det_tab[d]["train"]["gap"])[:5]
    for nv in (2, 3):
        key = f"vote{nv}"
        t[key] = t[top].sum(axis=1) >= nv if top else False
        for s in bars:
            dets[s][key] = np.sum([dets[s][d] for d in top], axis=0) >= nv if top else np.zeros(len(bars[s][0]), bool)
        row = {}
        for p in ("train", "unseen"):
            g = t[t["per"] == p]
            ch, nc = g[g[key]], g[~g[key]]
            row[p] = {"share_choppy": round(100 * len(ch) / max(len(g), 1), 1), "r_choppy": round(float(ch["r"].mean()), 3) if len(ch) else None,
                      "r_other": round(float(nc["r"].mean()), 3) if len(nc) else None,
                      "gap": round(float(nc["r"].mean() - ch["r"].mean()), 3) if len(ch) and len(nc) else None}
        det_tab[key] = row
    cands = [d for d in det_tab if det_tab[d]["train"]["gap"] is not None and 15 <= det_tab[d]["train"]["share_choppy"] <= 85]
    best_det = max(cands, key=lambda d: det_tab[d]["train"]["gap"]) if cands else "ci"
    # 2. management per strategy, with the chosen detector and with ci
    cfg = PC.bot_cfg(bot)
    if isinstance(cfg.get("risk"), dict):
        rk = cfg["risk"]
        risk = 1.0
        mults = {c: float(rk.get("scout:" + c) or rk.get(c) or 0.0018) for c in t["combo"].unique()}
    else:
        risk, mults = PC.bot_risk(bot), PC.mults(bot)
    recs = t.to_dict("records")
    for x in recs:
        x["share"] = 1.0
    asis_rep = {p: PC.replay(CS._book([x for x in recs if x["per"] == p], mults), risk) for p in ("train", "unseen")}
    mg = {"asis": {p: {k: asis_rep[p][k] for k in ("trades", "total", "worst_day", "max_dd")} for p in asis_rep}}
    for det in sorted({best_det, "ci"}):
        newr = {}
        for w in WAYS:
            newr[w] = np.array([apply_way(x, w, bars[x["sym"]], dets[x["sym"]][det], rngs[x["sym"]], exts[x["sym"]]) for x in recs])
        base = np.array([x["r"] for x in recs])
        combos = np.array([x["combo"] for x in recs])
        tr = np.array([x["per"] == "train" for x in recs])
        pick = {}
        for c in np.unique(combos):
            sel = (combos == c) & tr
            if sel.sum() < 20:
                continue
            b0 = base[sel].sum()
            gains = {w: newr[w][sel].sum() - b0 for w in WAYS}
            w = max(gains, key=gains.get)
            if gains[w] > max(1.0, 0.05 * abs(b0)):
                pick[c] = w
        # per-strategy choice, and every single way on the whole bot
        def run(choose):
            res = {}
            for p in ("train", "unseen"):
                taken = []
                for i, x in enumerate(recs):
                    if x["per"] != p:
                        continue
                    w = choose(x["combo"])
                    y = dict(x)
                    if w:
                        y["r"] = float(newr[w][i])
                    taken.append(y)
                rep = PC.replay(CS._book(taken, mults), risk)
                res[p] = {k: rep[k] for k in ("trades", "total", "worst_day", "max_dd")}
            return res
        mg[f"{det}:per_strategy"] = run(lambda c: pick.get(c))
        mg[f"{det}:per_strategy"]["picked"] = {w: sum(1 for v in pick.values() if v == w) for w in set(pick.values())}
        mg[f"{det}:per_strategy"]["strategies_changed"] = len(pick)
        for w in WAYS:
            mg[f"{det}:{w}"] = run(lambda c, w=w: w)
    return {"strategies": int(t["combo"].nunique()), "trades": int(len(t)), "detectors": det_tab, "top5": top,
            "best_detector": best_det, "management": mg}


def report(bots):
    out = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "bots": {}}
    for b in bots:
        if CS.out_dir(b).exists():
            try:
                out["bots"][b] = bot_lab(b)
            except Exception as e:  # noqa: BLE001
                import traceback
                out["bots"][b] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
            OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report"])
    ap.add_argument("--bot", default="bot1,bot2,bot3,bot4,bot5")
    a = ap.parse_args()
    r = report([x for x in a.bot.split(",") if x])
    for b, v in r["bots"].items():
        if "error" in v:
            print(b, v["error"])
            continue
        m = v["management"]
        print(b, "detector", v["best_detector"], {k: (round(x["train"]["total"] - m["asis"]["train"]["total"], 1),
                                                      round(x["unseen"]["total"] - m["asis"]["unseen"]["total"], 1))
                                                  for k, x in m.items() if k.endswith("per_strategy")})
