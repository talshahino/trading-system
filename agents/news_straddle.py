"""EV04 - the straddle before a release (the user's task tal1411strad, 14.11.2026: TikTok ExpertFx, gold on the PPI).

The idea: a minute before a red (high-impact) release put a Buy Stop above and a Sell Stop below the price, X away
(X = a multiple of the 5-minute ATR before the release). OCO: the first one filled cancels the other. Fixed risk per
trade (the bot's), never a martingale or a doubled position, one trade per release per instrument.
Grid (1-minute candles, Dukascopy, FTMO's spread at the fill minute + commission, agents/ftmo_costs.py):
  start    pre  - the orders go in at the open of the minute before the release (the original idea; INSIDE FTMO's
                  restricted window, so only for a virtual bot / the challenge phase)
           safe - the same orders at the open of minute 3 after the release (outside FTMO's 2 minutes; a pending order
                  filled inside the window is a violation on a funded account)
  X        0.5 / 1.0 / 1.5 x the 5-minute ATR(14) of the bars before the orders
  stop     opp  - the other order's level (risk = 2X + slippage)     atr1 - 1 x ATR from the fill
  exit     t1.5 / t2 (target in R) / be1t2 (stop to entry at +1R, target 2R) / be1 (stop to entry at +1R, no target)
  hold     out after 15 / 30 minutes from the fill
  cancel   not filled within 5 minutes after the release (safe: 5 minutes after the orders) -> no trade
  slippage the fill is worse than the order by half the minute's spread + 0.1 ATR (stop orders fill badly in news),
           on top of the spread the cost model charges; a minute that touches both orders counts as a full loss
           (filled and stopped in the same minute: the order inside a 1-minute candle is unknown)
  and in the study: which releases (USD only - the idea - or every red release of the instrument's currencies), a
  spread limit (no orders when the spread at the order minute is over 3 x the instrument's median), and at most
  1 / 2 / 3 trades per release over all instruments (EV01's lesson: one PPI = 8 stops on one dollar bet).
Fair method as the other bot 7 studies: chosen on 30.04-31.12.2025 by total R, shown on 2026 (never seen),
time-weighted (agents/agentkit.py); robustness = of the 10 best on 2025, how many made money in 2026. Also the gold
alone (the user's example) and each release type on its own.

Output: reports/agents/bot7/straddle_study.json. Cached per instrument (cache/news_straddle/<sym>.json), resumes
after a time budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7 (after the entry study).
  python -m agents.news_straddle [--minutes 10] [--force]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import rules_agent as R  # noqa: E402,F401  (first: rules_agent and event_strategies import each other)
from agents import event_strategies as E  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import poc_agent as P  # noqa: E402
from agents import bot7_trades as BT  # noqa: E402
from agents import news_straddle_live as NST  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "straddle_study.json"
CACHE = ROOT / "cache" / "news_straddle"
VERSION = 2                                # 2: the MANAGE exits and each row's mfe (16.11.2026)
STARTS = {"pre": -1, "safe": 3}            # the order minute relative to the release minute
XS = (0.5, 1.0, 1.5)
STOPS = ("opp", "atr1")
EXITS = ("t1.5", "t2", "be1t2", "be1")
# trade management on the live version only (pre, 1 ATR, stop at the other order, 30 minutes; task tal1611partial,
# 16.11.2026 - "most losses were in profit at some point, take them out partially"): stop to entry at +0.5R; half /
# 30% out at +0.5R / +1R and the rest's stop to the entry; stepped locking (from +S the stop at the entry, every
# further S locks S more). No target, out at the time limit, like be1.
MANAGE = ("be0.5", "h50at0.5", "h50at1", "h30at1", "lock1", "lock0.5")
MANAGE_ON = ("pre", 1.0, "opp", 30)
HOLDS = (15, 30)
CANCEL = 5                                 # minutes after the release (pre) / after the orders (safe)
SLIP_ATR = 0.1
CURS = ("usd", "own")
SPREAD_CAP = (0.0, 3.0)                    # x the instrument's median spread at the order minute (0 = no limit)
CAPS = (0, 1, 2, 3, 6, 9)               # 6 / 9 (15.11): the live bot counts every open stop - 9 x 0.5% = its 4.5% day
STUDY_VERSION = 3                          # the output (caps 6/9, day risk, slippage stress); the row cache keeps VERSION
STRESS_ATR = (0.2, 0.4)                    # extra slippage on the fill, x the 5-minute ATR (stop orders in news)
DAY_LIMIT_R = -9.0                         # 4.5% day stop / 0.5% risk per trade
TOP = 10
STALE_DAYS = 3
MIN_CAT_N = 8
BASE = "pre|x1|opp|t2|h30|cusd|sp0"        # the idea as told: 1 ATR, stop at the other order, 2R, 30 minutes
SPAN = 40                                  # minutes after the release that must be there without a gap


def vkey(st, x, stop, ex, hold):
    return f"{st}|x{x:g}|{stop}|{ex}|h{hold}"


def _atr5(d1):
    d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    a = R.atr(d5["h"].to_numpy(), d5["l"].to_numpy(), d5["c"].to_numpy(), 14)
    return pd.Series(a, index=d5.index)


def _exit(o, h, l, c, q0, end, dr, e, stop, tgt, be):
    """Walk 1-minute bars from the fill bar q0 (stop only in it: the order inside the candle is unknown), stop before
    target in every bar; be = the R at which the stop goes to entry (None = never). -> (exit price, exit bar)."""
    risk = abs(e - stop)
    s = stop
    if (l[q0] <= s) if dr > 0 else (h[q0] >= s):
        return s, q0
    moved = False
    for q in range(q0 + 1, end):
        if (l[q] <= s) if dr > 0 else (h[q] >= s):
            return s, q
        if tgt is not None and ((h[q] >= tgt) if dr > 0 else (l[q] <= tgt)):
            return tgt, q
        if be is not None and not moved and ((h[q] - e) if dr > 0 else (e - l[q])) >= be * risk:
            s, moved = e, True
    return c[end - 1], end - 1


def _manage(o, h, l, c, q0, end, dr, e, stop, kind):
    """One managed exit (MANAGE) from the fill bar q0, as _exit (stop only in the fill bar; in every bar the stop is
    checked before a new level; a level reached moves the stop from the next bar) -> (gross r, exit bar)."""
    risk = abs(e - stop)
    if (l[q0] <= stop) if dr > 0 else (h[q0] >= stop):
        return -1.0, q0
    if kind.startswith("be"):
        px, qx = _exit(o, h, l, c, q0, end, dr, e, stop, None, float(kind[2:]))
        return dr * (px - e) / risk, qx
    if kind.startswith("lock"):
        st_, s, best = float(kind[4:]), stop, -np.inf
        for q in range(q0 + 1, end):
            if (l[q] <= s) if dr > 0 else (h[q] >= s):
                return dr * (s - e) / risk, q
            best = max(best, ((h[q] - e) if dr > 0 else (e - l[q])) / risk)
            if best >= st_:
                lk = e + dr * (np.floor(best / st_) * st_ - st_) * risk
                s = max(s, lk) if dr > 0 else min(s, lk)
        return dr * (c[end - 1] - e) / risk, end - 1
    f, th = int(kind[1:3]) / 100.0, float(kind.split("at")[1])      # h50at1: half out at +1R
    s, took = stop, False
    for q in range(q0 + 1, end):
        if (l[q] <= s) if dr > 0 else (h[q] >= s):
            rest = dr * (s - e) / risk
            return (f * th + (1 - f) * rest if took else rest), q
        if not took and ((h[q] - e) if dr > 0 else (e - l[q])) >= th * risk:
            took, s = True, e
    rest = dr * (c[end - 1] - e) / risk
    return (f * th + (1 - f) * rest if took else rest), end - 1


def _mfe(h, l, q0, qx, dr, e, risk):
    """The furthest the trade went in its favour (R) from the bar after the fill to the bar before the exit."""
    if qx <= q0 + 1:
        return 0.0
    return float(max(0.0, ((h[q0 + 1:qx].max() - e) if dr > 0 else (e - l[q0 + 1:qx].min())) / risk))


def trades_sym(sym, rel):
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = P.bars(sym, "1Min", NF.LOAD_FROM, now)
    if d1 is None or d1.empty:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, NF.LOAD_FROM, now)
        sp = sp[~sp.index.duplicated()].sort_index() if sp is not None and len(sp) else None
    except Exception:  # noqa: BLE001
        sp = None
    idx = d1.index
    o, h, l, c = (d1[x].to_numpy(float) for x in "ohlc")
    a5 = _atr5(d1)
    spm = float(sp.median()) if sp is not None and len(sp) else 0.0
    own = set(E.events_for(sym))
    usd = {t for t, items in rel.items() if any(x[0] == "USD" for x in items)}   # red USD releases (the idea)
    times = sorted(t for t in (own | usd) if t >= NF.TEST_FROM)

    def spread_at(t):
        if sp is None:
            return 0.0
        k = sp.index.searchsorted(t, side="right") - 1
        return float(sp.iloc[k]) if k >= 0 else spm

    raw = []
    for t in times:
        k0 = idx.searchsorted(t)
        if k0 >= len(idx) or idx[k0] != t or k0 < 30 or k0 + SPAN >= len(idx) or idx[k0 + SPAN] - t != pd.Timedelta(minutes=SPAN):
            continue
        g = NF.tag(sym, t + pd.Timedelta(minutes=1), rel, [t])
        for st, off in STARTS.items():
            kp = k0 + off
            if idx[kp] - t != pd.Timedelta(minutes=off):
                continue
            ka = a5.index.searchsorted(idx[kp].floor("5min")) - 1       # the last 5-minute bar closed before the orders
            if ka < 14:
                continue
            atr = float(a5.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            ref = o[kp]
            spr = spread_at(idx[kp])
            last = (k0 + CANCEL) if off < 0 else (kp + CANCEL)       # the last bar the orders can fill in
            for x in XS:
                buy, sell = ref + x * atr, ref - x * atr
                q0, dr, both = None, 0, False
                for q in range(kp, last):
                    up, dn = h[q] >= buy, l[q] <= sell
                    if up or dn:
                        q0, dr, both = q, (1 if up else -1), bool(up and dn)
                        break
                if q0 is None:
                    continue
                slip = 0.5 * spread_at(idx[q0]) + SLIP_ATR * atr
                lvl = buy if dr > 0 else sell
                e = max(lvl, o[q0]) + slip if dr > 0 else min(lvl, o[q0]) - slip     # a gap through the order fills at the open
                for stop_k in STOPS:
                    stop = (sell if dr > 0 else buy) if stop_k == "opp" else e - dr * atr
                    risk = dr * (e - stop)
                    if risk <= 0:
                        continue
                    mng = [(m_, MANAGE_ON[3]) for m_ in MANAGE] if (st, x, stop_k) == MANAGE_ON[:3] else []
                    for ex, hold in list(itertools.product(EXITS, HOLDS)) + mng:
                        end = min(q0 + hold, len(idx))
                        if ex in MANAGE:
                            gr, qx = (-1.0, q0) if both else _manage(o, h, l, c, q0, end, dr, e, stop, ex)
                            raw.append({"v": vkey(st, x, stop_k, ex, hold), "time": idx[q0], "exit_time": idx[qx], "dir": dr,
                                        "entry_px": e, "stop_px": stop, "r": gr, "rel": t,
                                        "usd": int(t in usd), "own": int(t in own), "spr": spr / spm if spm > 0 else 0.0,
                                        "cat": (g or {}).get("cat", "other"), "both": int(both),
                                        "mfe": 0.0 if both else _mfe(h, l, q0, qx, dr, e, risk)})
                            continue
                        tg = {"t1.5": 1.5, "t2": 2.0, "be1t2": 2.0, "be1": None}[ex]
                        be = 1.0 if ex.startswith("be1") else None
                        tgt = None if tg is None else e + dr * tg * risk
                        if both:
                            px, qx = stop, q0
                        else:
                            px, qx = _exit(o, h, l, c, q0, end, dr, e, stop, tgt, be)
                        raw.append({"v": vkey(st, x, stop_k, ex, hold), "time": idx[q0], "exit_time": idx[qx], "dir": dr,
                                    "entry_px": e, "stop_px": stop, "r": dr * (px - e) / risk, "rel": t,
                                    "usd": int(t in usd), "own": int(t in own), "spr": spr / spm if spm > 0 else 0.0,
                                    "cat": (g or {}).get("cat", "other"), "both": int(both),
                                    "mfe": 0.0 if both else _mfe(h, l, q0, qx, dr, e, risk)})
    if not raw:
        return []
    tt = pd.DataFrame(raw)
    net = C.apply(tt, sym, sp)
    rows = []
    for x in net.itertuples():
        rows.append({"v": x.v, "sym": sym, "cls": U.CLASS.get(sym, "?"), "t": pd.Timestamp(x.time).isoformat(),
                     "dir": int(x.dir), "r": round(float(x.r), 4), "gross": round(float(tt.at[x.Index, "r"]), 4),
                     "cat": x.cat, "usd": int(x.usd), "own": int(x.own), "spr": round(float(x.spr), 2), "both": int(x.both),
                     "rel": pd.Timestamp(x.rel).isoformat(), "x": pd.Timestamp(x.exit_time).isoformat(),
                     "mfe": round(float(x.mfe), 3)})
    return rows


def capped(z, cap):
    """At most `cap` trades per release over all instruments: the first by fill time, then the bot's fixed order."""
    if not cap or z.empty:
        return z
    order = {s_: i for i, s_ in enumerate(NF.SYMBOLS)}
    z = z.assign(_o=z["sym"].map(order).fillna(99)).sort_values(["rel", "t", "_o"])
    return z[z.groupby("rel").cumcount() < cap].drop(columns="_o")


def study(df, base_key=BASE):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
    cand = []
    for v, y in df.groupby("v"):
        for cu in CURS:
            z0 = y[y["usd"] == 1] if cu == "usd" else y[y["own"] == 1]
            for spc in SPREAD_CAP:
                z1 = z0 if not spc else z0[z0["spr"] <= spc]
                for cap in CAPS:
                    z = capped(z1, cap)
                    tr, un = z[z.train], z[~z.train]
                    cand.append({"v": f"{v}|c{cu}|sp{spc:g}" + (f"|cap{cap}" if cap else ""),
                                 "train": NF._w(tr), "unseen": NF._w(un),
                                 "gross_unseen": round(float(un["gross"].sum()), 2) if len(un) else 0.0})
    # ranked by total R on 2025; a variant with too few 2025 trades to judge goes last (an empty one has sum 0)
    cand.sort(key=lambda c_: (c_["train"]["n"] < MIN_CAT_N, -c_["train"]["sum"]))
    top = cand[:TOP]
    base = next((c_ for c_ in cand if c_["v"] == base_key), None)
    by = {}
    for key, pat in (("start", r"^(pre|safe)\|"), ("x", r"\|x([\d.]+)\|"), ("stop", r"\|(opp|atr1)\|"),
                     ("exit", r"\|(t1\.5|t2|be1t2|be1|be0\.5|h50at0\.5|h50at1|h30at1|lock1|lock0\.5)\|"), ("hold", r"\|h(\d+)\|"), ("cur", r"\|c(usd|own)\|"),
                     ("spread", r"\|sp([\d.]+)"), ("cap", r"(?:\|cap(\d+))?$")):
        g = {}
        for c_ in cand:
            m = re.search(pat, c_["v"])
            if m:
                g.setdefault(m.group(1) or "0", []).append(c_)
        by[key] = {k_: {"variants": len(cs), "train_median_sum": round(float(np.median([c_["train"]["sum"] for c_ in cs])), 2),
                        "unseen_median_sum": round(float(np.median([c_["unseen"]["sum"] for c_ in cs])), 2),
                        "unseen_positive": int(sum(1 for c_ in cs if (c_["unseen"]["sum"] or 0) > 0))}
                   for k_, cs in g.items()}
    return {"baseline": base, "best": cand[0] if cand else None, "top": top,
            "robust": {"top_positive_unseen": int(sum(1 for c_ in top if (c_["unseen"]["sum"] or 0) > 0)), "top": len(top),
                       "top_better_than_base": int(sum(1 for c_ in top if base and (c_["unseen"]["sum"] or 0) > (base["unseen"]["sum"] or 0)))},
            "by": by, "tested": len(cand)}


def _rows_for(df, key):
    """The rows of one study key "<v>|c<cur>|sp<x>[|cap<n>]"."""
    m = re.match(r"^(.*?)\|c(usd|own)\|sp([\d.]+)(?:\|cap(\d+))?$", key)
    if not m:
        return df.iloc[0:0]
    v, cu, spc, cap = m.group(1), m.group(2), float(m.group(3)), int(m.group(4) or 0)
    z = df[(df["v"] == v) & (df["usd" if cu == "usd" else "own"] == 1)]
    if spc:
        z = z[z["spr"] <= spc]
    return capped(z, cap)


def risk_check(df, key):
    """For one version: the worst release / day in R (0.5% each: -9R = the bot's 4.5% day stop), how many days crossed
    it, and the 2026 total with extra slippage (STRESS_ATR x ATR on the fill: about s / (2X) R with the stop at the other
    order, s R with the 1-ATR stop)."""
    z = _rows_for(df, key)
    if z.empty:
        return None
    t = pd.to_datetime(z["t"], utc=True)
    un = z[t >= NF.SPLIT]
    day = z.groupby(t.dt.strftime("%Y-%m-%d"))["r"].sum()
    rel = z.groupby("rel")["r"].sum()
    m = re.search(r"\|x([\d.]+)\|(opp|atr1)\|", key)
    per = (lambda s_: s_ / (2 * float(m.group(1)))) if m and m.group(2) == "opp" else (lambda s_: s_)
    return {"worst_day_r": round(float(day.min()), 2), "days_over_limit": int((day <= DAY_LIMIT_R).sum()),
            "worst_release_r": round(float(rel.min()), 2), "max_per_release": int(z.groupby("rel").size().max()),
            "days": int(len(day)),
            "stress_unseen": {f"slip+{s_:g}atr": round(float((un["r"] - per(s_)).sum()), 2) for s_ in STRESS_ATR},
            "unseen_sum": round(float(un["r"].sum()), 2), "unseen_n": int(len(un))}


def deploy(df, res):
    """What the live bot can really run: chosen on 2025 among the versions with at most 9 trades per release (the open
    stops of one release must fit the 4.5% day), with the day risk and the slippage stress; plus the same for the
    overall best and the 10 best (chosen on 2025)."""
    cands = []
    for v, y in df.groupby("v"):
        for cu in CURS:
            z0 = y[y["usd"] == 1] if cu == "usd" else y[y["own"] == 1]
            for spc in SPREAD_CAP:
                z1 = z0 if not spc else z0[z0["spr"] <= spc]
                for cap in (3, 6, 9):
                    z = capped(z1, cap)
                    tr = z[pd.to_datetime(z["t"], utc=True) < NF.SPLIT]
                    cands.append((float(tr["r"].sum()), len(tr), f"{v}|c{cu}|sp{spc:g}|cap{cap}"))
    cands.sort(key=lambda x: (x[1] < MIN_CAT_N, -x[0]))
    out = {"best_capped": [], "top_check": []}
    for _, _, k in cands[:TOP]:
        z = _rows_for(df, k)
        t = pd.to_datetime(z["t"], utc=True)
        out["best_capped"].append({"v": k, "train": NF._w(z[t < NF.SPLIT]), "unseen": NF._w(z[t >= NF.SPLIT]),
                                   "risk": risk_check(df, k)})
    for c_ in (res.get("top") or [])[:TOP]:
        out["top_check"].append({"v": c_["v"], "risk": risk_check(df, c_["v"])})
    out["robust_capped"] = {"top_positive_unseen": int(sum(1 for c_ in out["best_capped"] if (c_["unseen"]["sum"] or 0) > 0)),
                            "top": len(out["best_capped"])}
    return out


def by_group(df, col):
    out = {}
    for k, y in df.groupby(col):
        r_ = study(y)
        top = [c_ for c_ in r_["top"] if c_["train"]["n"] >= MIN_CAT_N]
        if not top:
            continue
        out[str(k)] = {"best": top[0], "baseline": r_["baseline"],
                       "top_positive_unseen": int(sum(1 for c_ in top if (c_["unseen"]["sum"] or 0) > 0)), "top": len(top)}
    return out


def built_key(p):
    """The study key of a built EV04 plan (live/bots/bot7.json, card bfdb13a3a001)."""
    return (f"{p.get('start', 'pre')}|x{float(p.get('x_atr5', 1.0)):g}|{p.get('stop', 'opp')}|{p.get('exit', 'be1')}"
            f"|h{int(p.get('hold_min', 30))}|c{p.get('cur', 'own')}|sp{float(p.get('spread_cap', 0) or 0):g}"
            + (f"|cap{int(p['cap_per_release'])}" if p.get("cap_per_release") else ""))


def export(df):
    """The built version's trades for the whole-bot check (agents/bot7_trades.py). The cap per release counts the
    bot's EV04 instruments only, as the live bot does."""
    b = BT.built("EV04", "1Min")
    if df is None or not len(df) or not b:
        return {}
    syms = [s_ for s_ in b if s_ in set(df["sym"])]
    df = skipped_out(df, b)
    df = window_out(df, b)
    keys = {}
    for s_ in syms:
        keys.setdefault(built_key(b[s_]), []).append(s_)
    out = {}
    for k, ss in keys.items():
        z = _rows_for(df[df["sym"].isin(ss)], k)
        hold = {s_: int(b[s_].get("hold_min", 30)) + int(b[s_].get("cancel_min", CANCEL)) for s_ in ss}
        out.update(BT.write("EV04", "1Min", z, ss, hold, "agents/news_straddle.py", k))
    return out


def gaveback(df):
    """Task tal1611partial (16.11.2026): the live version's trades - how many losers first went +0.5R / +1R (mfe), and
    each MANAGE exit on the same fills (same skipped types, fill window, cap) against the built one, 2025 / 2026."""
    b = BT.built("EV04", "1Min")
    if df is None or not len(df) or not b or "mfe" not in df:
        return {}
    df = window_out(skipped_out(df, b), b)
    keys = {}
    for s_ in b:
        if s_ in set(df["sym"]):
            keys.setdefault(built_key(b[s_]), []).append(s_)

    def rows(ex):
        parts = []
        for k, ss in keys.items():
            kk = re.sub(r"^(pre\|x1\|opp)\|[^|]+\|h30\|", rf"\1|{ex}|h30|", k)
            if kk == k and ex != "built":
                continue
            parts.append(_rows_for(df[df["sym"].isin(ss)], k if ex == "built" else kk))
        z = pd.concat(parts) if parts else df.iloc[0:0]
        return z.assign(train=pd.to_datetime(z["t"], utc=True) < NF.SPLIT)

    base = rows("built")
    out = {"census": {}, "ways": {}}
    for per, z in (("train", base[base.train]), ("unseen", base[~base.train])):
        lo = z[z["r"] < 0]
        out["census"][per] = {"trades": int(len(z)), "losers": int(len(lo)), "lost_r": round(float(lo["r"].sum()), 1),
                              "mfe_0.5R": int((lo["mfe"] >= 0.5).sum()), "mfe_1R": int((lo["mfe"] >= 1).sum()),
                              "mfe_median": round(float(lo["mfe"].median()), 2) if len(lo) else None}
    day = lambda z: float(z.groupby(pd.to_datetime(z["t"], utc=True).dt.strftime("%Y-%m-%d"))["r"].sum().min()) if len(z) else 0.0  # noqa: E731
    for ex in ("built",) + MANAGE:
        z = base if ex == "built" else rows(ex)
        out["ways"][ex] = {per: {**NF._w(y), "worst_day_r": round(day(y), 2), **_curve(y)}
                           for per, y in (("train", z[z.train]), ("unseen", z[~z.train]))}
    bt = out["ways"]["built"]
    # task mgr1611curve (16.11.2026): a way that clearly smooths the curve on BOTH periods - 2+ points more green
    # days, a smaller fall from the peak, the worst day not worse - and keeps 80%+ of today's total (as bot 6)
    out["smoother"] = [ex for ex in MANAGE if ex in out["ways"] and all(
        None not in (out["ways"][ex][p]["dd"], bt[p]["dd"])
        and (out["ways"][ex][p]["green"] or 0) >= (bt[p]["green"] or 0) + 2.0 and out["ways"][ex][p]["dd"] < bt[p]["dd"]
        and out["ways"][ex][p]["worst_day_r"] >= bt[p]["worst_day_r"]
        and (out["ways"][ex][p]["sum"] or 0) >= 0.8 * max(bt[p]["sum"] or 0, 0) > 0 for p in ("train", "unseen"))]
    out["both"] = [ex for ex in MANAGE if ex in out["ways"]
                   and (out["ways"][ex]["train"]["sum"] or 0) > (bt["train"]["sum"] or 0)
                   and (out["ways"][ex]["unseen"]["sum"] or 0) > (bt["unseen"]["sum"] or 0)]
    out["best_train"] = max(MANAGE, key=lambda ex: out["ways"].get(ex, {}).get("train", {}).get("sum") or -1e9)
    return out


def _curve(z):
    """The curve's smoothness on one period (task mgr1611curve): % green days, the largest fall from the peak (R) and
    the daily spread; the win rate is NF._w's "win"."""
    if not len(z):
        return {"green": None, "dd": None, "sd": None}
    d = z.groupby(pd.to_datetime(z["t"], utc=True).dt.strftime("%Y-%m-%d"))["r"].sum()
    eq = d.cumsum()
    return {"green": round(float((d > 0).mean() * 100), 1), "dd": round(float((eq.cummax().clip(lower=0) - eq).max()), 2),
            "sd": round(float(d.std()), 2) if len(d) > 1 else 0.0}


def skipped_out(df, b):
    """The rows left after each instrument's skipped release types (plan "skip_groups", e.g. ["USD:gdp"]; bot 7 smart
    agent 16.11.2026) - before the per-release cap, as the live bot (agents/news_straddle_live.skip_events)."""
    sk = {s_: set(p.get("skip_groups") or []) for s_, p in b.items() if p.get("skip_groups")}
    if not sk or df is None or not len(df):
        return df
    G = NST.release_groups(json.loads(NF.HIST.read_text(encoding="utf-8")))
    keep = []
    for sym, rel in zip(df["sym"], df["rel"]):
        s_ = sk.get(sym)
        keep.append(not s_ or not (NST.groups_of(sym, pd.Timestamp(rel).tz_convert("UTC"), G, U.CURRENCIES.get(sym, [])) & s_))
    return df[np.array(keep, bool)]


def window_out(df, b):
    """The rows whose fill minute (after the release, -1 = the minute before) is inside each instrument's plan window:
    from "fill_from" (none = -1) to "cancel_min" - 1 (default CANCEL) - before the per-release cap, as the live bot
    (agents/news_straddle_live.live_straddle fill_from / cancel; bot 7 smart agent 16.11.2026)."""
    w = {s_: (p.get("fill_from"), p.get("cancel_min")) for s_, p in b.items()
         if p.get("fill_from") is not None or (p.get("cancel_min") not in (None, CANCEL))}
    if not w or df is None or not len(df):
        return df
    off = ((pd.to_datetime(df["t"], utc=True, format="ISO8601") - pd.to_datetime(df["rel"], utc=True, format="ISO8601"))
           // pd.Timedelta(minutes=1)).to_numpy()
    keep = np.ones(len(df), bool)
    for i, sym in enumerate(df["sym"]):
        if sym in w:
            ff, cn = w[sym]
            lo = -1 if ff is None else int(ff)
            hi = CANCEL if cn is None else int(cn)
            keep[i] = lo <= off[i] < hi
    return df[keep]


def _cached_rows():
    rows = []
    for f in sorted(CACHE.glob("*.json")) if CACHE.exists() else []:
        try:
            c_ = json.loads(f.read_text(encoding="utf-8"))
            if c_.get("version") == VERSION:
                rows += c_["rows"]
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(rows)


def _refresh_gaveback():
    """The fresh study without the curve stats (task mgr1611curve, 17.11.2026): only the gaveback part again, from the
    cached rows - minutes, not the whole 2,448-variant grid."""
    j = json.loads(OUT.read_text(encoding="utf-8"))
    g = (j.get("study") or {}).get("gaveback") or {}
    if "smoother" in g:
        return
    df = _cached_rows()
    if not len(df):
        return
    j["study"]["gaveback"] = gaveback(df)
    j["gaveback_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    OUT.write_text(json.dumps(j, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("EV04 gaveback refreshed (curve stats)")


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS) or j.get("partial", True)
                or j.get("version") != STUDY_VERSION)
    except Exception:  # noqa: BLE001
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args(argv)
    if not a.force and not stale():
        print("straddle study fresh - skipped")
        BT.safe(lambda: export(_cached_rows()))     # the whole-bot check's trades follow the plans every run
        BT.safe(_refresh_gaveback)
        return
    t_end = time.time() + a.minutes * 60
    rel = NF.releases()
    CACHE.mkdir(parents=True, exist_ok=True)
    syms = [s for s in a.symbols.split(",") if s] or ["XAUUSD"] + [s for s in NF.SYMBOLS if s != "XAUUSD"]   # the gold first
    rows, done = [], []
    for sym in syms:
        f = CACHE / f"{sym}.json"
        try:
            c_ = json.loads(f.read_text(encoding="utf-8"))
            if c_.get("version") == VERSION and pd.Timestamp.now(tz="UTC") - pd.Timestamp(c_["at"]) < pd.Timedelta(days=STALE_DAYS):
                rows += c_["rows"]
                done.append(sym)
                continue
        except Exception:  # noqa: BLE001
            pass
        if time.time() > t_end and done:
            print(f"EV04: time budget reached after {done}")
            continue
        try:
            r_ = trades_sym(sym, rel)
            rows += r_
            f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
            done.append(sym)
            print(f"EV04 {sym}: {len(r_)} rows ({len(rows)} so far)", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"EV04 {sym}: failed ({e})")
    df = pd.DataFrame(rows)
    res = study(df) if len(df) else {}
    BT.safe(export, df)
    if len(df):
        tr = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
        res["gold"] = study(df[df["sym"] == "XAUUSD"]) if (df["sym"] == "XAUUSD").any() else None
        res["by_cat"] = by_group(tr.drop(columns="train"), "cat")
        res["by_sym"] = by_group(tr.drop(columns="train"), "sym")
        res["both_share"] = round(float(df["both"].mean()), 3)
        try:
            res["gaveback"] = gaveback(df)
        except Exception as e:  # noqa: BLE001
            res["gaveback"] = {"error": str(e)}
        try:
            res["deploy"] = deploy(df, res)
        except Exception as e:  # noqa: BLE001 - the extra check never loses the study
            res["deploy"] = {"error": str(e)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": STUDY_VERSION,
                               "symbols": done, "partial": len(done) < len(NF.SYMBOLS), "rows": int(len(df)),
                               "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()), "base": BASE,
                               "study": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res:
        print("EV04", json.dumps({"base": res["baseline"], "best": res["best"], "robust": res["robust"]}, ensure_ascii=False, default=str))
        d_ = res.get("deploy") or {}
        print("EV04 deploy", json.dumps({"best_capped": (d_.get("best_capped") or [None])[0], "robust_capped": d_.get("robust_capped"),
                                         "best_risk": (d_.get("top_check") or [None])[0]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
