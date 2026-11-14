"""Backtest of rule-based strategies collected from TradingView (agents/rules_agent.py),
published parameters, no searching. Same test as the POC strategies:
  - 5 currency pairs + gold, silver, oil (Dukascopy), 1 / 5 / 15 minute and 1 hour charts
  - no trade opens or closes within 5 minutes of a high-impact news release
  - ONE $100,000 account per strategy under FTMO rules, opened one year ago, sampled weekly,
    stopped when burned; plus the chance to pass the FTMO 2-Step from different start dates.

  python run_rules.py [--demo]
"""
import argparse
import json
import html
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config
from agents import ftmo_agent as F
from agents import ftmo_costs as C
from agents import ftmo_universe as U
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP

SYMBOLS = RP.FX + RP.COMMOD
SWEEP_RISKS = (0.0025, 0.005, 0.0075, 0.01)   # TV strategies: Prophet at each of these risks per trade
LOW_RISK = 0.0075            # every other strategy: 25% below the 1% main risk
MAX_PROPHET_TRADES = 25000   # minute-by-minute replay is skipped above this (too slow, and such systems burn anyway)
_BARS = {}                   # 1-minute bars per symbol, loaded once per run
POOLED_SWEEP = False         # all-instruments-in-one-account sweep, replaced by per-instrument selection
SEL_MIN_TRADES, SEL_MIN_AVG_R, SEL_MIN_T = 30, 0.05, 1.5   # instrument+timeframe selection on the earlier period
PER_SYMBOL_RISKS = (0.005, 0.0075, 0.01)
PER_SYMBOL = []              # one row per (strategy, timeframe, instrument) that passed selection
SELECTED_AFTER = []          # their last-year trades, for the portfolio test
PORTFOLIO = {}
ERRORS = []   # tracebacks of strategies that crashed; shown at the end of the report
TIMEFRAMES = RP.TIMEFRAMES + ["4Hour"]
SPLIT, SPLIT_HE = RP.SPLIT, RP.SPLIT_HE
TEST_FROM_MODE = [False]
SPREADS = {}     # historical 1-minute spread per instrument
SPREAD_EST = set()   # instruments whose spread is estimated (no bid/ask history)
NC.CURRENCIES.update(U.CURRENCIES)
COMBO_TRADES = []
ALL_TRADES = []      # test mode: every combo's trades, passed or not (08.11.2026: combo_trades keeps only the
                     # combos that passed, so pooling those files over-states every strategy)
OPT_RESULTS = []
# trade-management grid: target (None = strategy's own, "none" = no target), breakeven fraction (0 = off), ATR-stop scale
OPT_TARGETS = [None, ("R", 1.0), ("R", 1.5), ("R", 2.0), ("R", 3.0), ("R", 4.0), ("R", 5.0), "none"]   # 4 / 5R: 12.11.2026 (QB11's 8R)
OPT_BE = [None]                 # None = the run's own --breakeven (the bots' rule)
OPT_STOP = [0.75, 1.0, 1.25, 1.5]
OPT_ENTRY = [0.0, 0.25, 0.5]     # limit order this many R better than the next open, valid 3 bars
OPT_ALL_TRADES = True            # keep every variant's trades (improve.py chooses, not the run)


def _variant(fn, target, stop_scale, pullback=0.0):
    def g(df):
        lo, sh, plan = fn(df)
        plan = dict(plan)
        if target == "none":
            plan["target"] = None
        elif target is not None:
            plan["target"] = target
        if stop_scale != 1.0:
            plan["stop_scale"] = stop_scale
        if pullback:
            plan["entry_pullback"] = (pullback, 3)
        return lo, sh, plan
    return g


def _stats(r):
    r = pd.Series(r, dtype="float64")
    if r.empty:
        return {"n": 0, "sum": 0.0, "avg": 0.0, "pf": 0.0}
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": int(len(r)), "sum": round(float(r.sum()), 2), "avg": round(float(r.mean()), 3),
            "pf": round(float(gain / loss), 2) if loss > 0 else 99.0}


def _regimes(df):
    """Market-regime masks (known at the bar's close). Each is (allow_long, allow_short)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    e200 = R.ema(c, 200)
    a = R.adx(h, l, c, 14)
    at = R.atr(h, l, c, 14) / c
    pr = pd.Series(at).rolling(500, min_periods=100).rank(pct=True).to_numpy()
    yes = np.ones(len(c), bool)
    return {
        "none": (yes, yes),
        "with_trend200": (c > e200, c < e200),
        "adx_trend": (a > 20, a > 20),
        "adx_range": (a < 20, a < 20),
        "vol_normal": ((pr > 0.2) & (pr < 0.9), (pr > 0.2) & (pr < 0.9)),
        "vol_high": (pr >= 0.5, pr >= 0.5),
        "vol_low": (pr < 0.5, pr < 0.5),
        **_extra_regimes(c),
    }


# the user's filters (14.11.2026, task tal1411ema): tried only with --filters-extra (bot 7 first); always known to
# _regimes so a built plan with one of them resolves
EXTRA_REGIMES = ("ema9_20", "bbw_p20")


def _extra_regimes(c):
    """ema9_20: EMA 9 above EMA 20 = longs only, below = shorts only (the bar's own timeframe).
    bbw_p20: Bollinger 20/2 width ((upper - lower) / middle) above its 20th percentile of the last 100 bars."""
    e9, e20 = R.ema(c, 9), R.ema(c, 20)
    s = pd.Series(c)
    mid, sd = s.rolling(20).mean(), s.rolling(20).std(ddof=0)
    w = (4 * sd / mid).to_numpy()
    lo = pd.Series(w).rolling(100, min_periods=50).quantile(0.2).to_numpy()
    ok = np.isfinite(w) & np.isfinite(lo) & (w > lo)
    return {"ema9_20": (e9 > e20, e9 < e20), "bbw_p20": (ok, ok)}


def filter_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block=None):
    """Each regime filter on one combo: choose on the training window, check on the later window - on top of the
    exits / trade management the bot trades now (--plans-file), like manage_combo (13.11.2026: the filters were tested
    on the strategy's own exits, so a filter card's baseline was not what runs - TV58|1Hour|USDCAD bot 3 got a filter
    and a new exit plan built together, the pair never tested)."""
    fn0 = R.STRATEGIES[sid][1]
    fn = fn0 if news_block is None else _blocked(fn0, news_block)
    t0, t1 = SPLIT, pd.Timestamp(args.filters, tz="UTC")
    pk = {k: v for k, v in plan_keys(PLANS.get(f"{sid}|{tf}|{s}") or {}).items()}
    regimes = _regimes(df)
    rows, keep = [], {}
    for name, (al, ash) in regimes.items():
        if name in EXTRA_REGIMES and not getattr(args, "filters_extra", False):
            continue
        def g(d, al=al, ash=ash):
            lo, sh, plan = fn(d)
            return np.asarray(lo, bool) & al, np.asarray(sh, bool) & ash, {**plan, **pk}
        t = R.run(df, g, cost, breakeven=be, session_exit=sess)
        t = drop_hole_trades(t, df, s, tf)
        if t.empty:
            continue
        if real:
            t = C.apply(t, s, SPREADS.get(s))
        if events is not None:
            t = t[~(NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events))]
        tr, va = t[(t["time"] >= t0) & (t["time"] < t1)], t[t["time"] >= t1]
        rows.append({"variant": name, "train": _stats(tr["r"]), "valid": _stats(va["r"])})
        keep[name] = t[t["time"] >= t0]
    if not rows:
        return
    ok = [x for x in rows if x["train"]["n"] >= 20]
    best = max(ok or rows, key=lambda x: (x["train"]["sum"], x["train"]["pf"]))
    base = next((x for x in rows if x["variant"] == "none"), None)
    OPT_RESULTS.append({"combo": f"{sid}|{tf}|{s}", "best": best, "base": base, "all": rows})
    for k in {best["variant"], "none"} & set(keep):
        COMBO_TRADES.append(keep[k].assign(sid=sid, tf=tf, symbol=s, variant=k))


# trade management per strategy (user, 12.11.2026: "every strategy moves its stop to break-even at some point, the
# timing tuned per strategy" + partial exits): on top of the combo's exits as the bot trades it now (--plans-file)
# never without a breakeven: every strategy moves its stop to the entry at some point (user, 09.11.2026) - only WHEN
MAN_BE = [0.4, 0.5, 0.6, 0.7, 0.8]                      # with a target: breakeven at this share of the way
MAN_BE_R = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]               # no target: breakeven once this many R in profit
# 0.75 / 1R and half at +0.5R (bots 1-4 smart agent, 14.11.2026): 22 live trades of bots 1-4 were +0.5R or more and
# ended at a full stop (-22R), 11 of them no-target strategies (FR01 GBPUSD +1.16R, TV20 XAUUSD +0.84R, SE02 USDJPY
# +0.56R x4, FR08 SPCX x3) - the grid never had a point under 1.5R for them. Chosen to 2025 / shown on 2026 as always.
MAN_PART = [None, (0.5, 0.25), (0.3, 0.5), (0.3, 0.7), (0.5, 0.5), (0.5, 0.7)]   # with a target: (share out, way); half at a quarter: 12.11.2026
MAN_PART_R = [None, (0.5, 0.5), (0.3, 1.0), (0.5, 1.0), (0.3, 2.0), (0.5, 2.0)]    # no target: (share out, at R)
# Fibonacci levels of the way from the entry to the target (user, 12.11.2026 - standing task for every bot): the stop to
# entry at each level, a part out (25% / 33% / 50%) at one level and the rest to breakeven / the target, two levels
# (a part out at 38.2%, breakeven at 61.8%). --manage-grid fib; strategies with no target have no way to measure.
FIB_BE = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
FIB_PART = [None] + [(sh, lv) for sh in (0.25, 0.33, 0.5) for lv in (0.382, 0.618, 0.786)]
PLANS = {}                                              # --plans-file: combo -> the bot's built plan (live/bots/<bot>.json)


def plan_keys(p):
    """The bot's built plan in the engine's words (the same as live/runner.py plan_of)."""
    out = {}
    if p.get("target") == "none":
        out["target"] = None
    elif p.get("target") is not None:
        out["target"] = ("R", float(p["target"]))
    if p.get("stop") is not None and float(p["stop"]) != 1.0:
        out["stop_scale"] = float(p["stop"])
    if p.get("limit"):
        out["entry_pullback"] = (float(p["limit"]), 3)
    for k in ("be_way", "be_r"):
        if p.get(k) is not None:
            out[k] = float(p[k])
    if p.get("partial"):
        out["partial"] = tuple(float(x) for x in p["partial"])
    if p.get("ladder"):
        out["ladder"] = (tuple(float(x) for x in p["ladder"]), int(p.get("ladder_bars") or LADDER_BARS))
    out.update(rule_keys(p, exits=not any(k in p for k in ("target", "stop", "limit"))))
    return out


def rule_keys(p, exits=True):
    """The morning review's live rules on a combo (agent_portfolio puts them in the plans file as "rules", 14.11.2026),
    in the engine's words - as live/runner.combo_rules: direction and hours always, the target / stop factors only
    when the combo has no built exit plan (the plan wins for the exits)."""
    r, out = p.get("rules") or {}, {}
    try:
        if r.get("dir") in (1, -1):
            out["rule_dir"] = int(r["dir"])
        if r.get("hours") and len(r["hours"]) == 2 and 0 <= int(r["hours"][0]) < int(r["hours"][1]) <= 24:
            out["rule_hours"] = (int(r["hours"][0]), int(r["hours"][1]))
        if exits and r.get("tp") is not None and 0.3 <= float(r["tp"]) < 1:
            out["rule_tp"] = float(r["tp"])
        if exits and r.get("sl") is not None and 0.5 <= float(r["sl"]) < 1:
            out["rule_sl"] = float(r["sl"])
    except (TypeError, ValueError):
        return {}
    return out


# laddered entries (user, 13.11.2026, standing task of the bot 1-4 agents): the same total risk split into equal parts,
# part k a limit order k x risk better than the signal's open (0 = at the open), valid LADDER_BARS bars, the same stop
# and target (rules_agent.run, plan "ladder"). "lad0" = one entry at the open, as the bots trade now.
LADDER_BARS = 3
LADDERS = {"lad0,0.5": (0.0, 0.5), "lad0,0.25,0.5": (0.0, 0.25, 0.5), "lad0,0.33,0.67": (0.0, 0.33, 0.67),
           "lad0,0.25,0.5,0.75": (0.0, 0.25, 0.5, 0.75), "lad0.25,0.5": (0.25, 0.5)}


def lad_key(levels):
    return "lad" + ",".join(f"{float(x):g}" for x in levels) if levels else "lad0"


def ladder_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block=None):
    """Every laddered entry of one combo, on top of its plan as the bot trades it now (exits, filter, management)."""
    fn0 = R.STRATEGIES[sid][1]
    fn = fn0 if news_block is None else _blocked(fn0, news_block)
    t0, t1 = SPLIT, pd.Timestamp(args.manage, tz="UTC")
    pk = plan_keys(PLANS.get(f"{sid}|{tf}|{s}") or {})
    if pk.get("entry_pullback"):
        return                                   # a built limit entry is already one ladder step
    plan0 = {**fn(df)[2], **pk}
    flt = (PLANS.get(f"{sid}|{tf}|{s}") or {}).get("filter")
    regime = _regimes(df).get(flt) if flt else None
    rows, keep = [], {}
    for key, lv in [("lad0", None)] + list(LADDERS.items()):
        extra = {**pk, **({"ladder": (lv, LADDER_BARS)} if lv else {})}

        def g(d, extra=extra):
            lo, sh, plan = fn(d)
            if regime is not None:
                lo, sh = np.asarray(lo, bool) & regime[0], np.asarray(sh, bool) & regime[1]
            return lo, sh, {**plan, **extra}
        t = R.run(df, g, cost, breakeven=be, session_exit=sess)
        t = drop_hole_trades(t, df, s, tf)
        if t.empty:
            continue
        if real:
            t = C.apply(t, s, SPREADS.get(s))
        if events is not None:
            t = t[~(NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events))]
        tr, va = t[(t["time"] >= t0) & (t["time"] < t1)], t[t["time"] >= t1]
        rows.append({"variant": key, "train": _stats(tr["r"]), "valid": _stats(va["r"])})
        keep[key] = t[t["time"] >= t0]
    if not rows:
        return
    ok = [x for x in rows if x["train"]["n"] >= 20]
    best = max(ok or rows, key=lambda x: (x["train"]["sum"], x["train"]["pf"]))
    base = next((x for x in rows if x["variant"] == "lad0"), None)
    OPT_RESULTS.append({"combo": f"{sid}|{tf}|{s}", "best": best, "base": base, "all": rows,
                        "has_target": plan0.get("target") is not None})
    for k in set(keep):
        COMBO_TRADES.append(keep[k].assign(sid=sid, tf=tf, symbol=s, variant=k))


# pyramid (user, 14.11.2026, task tal1411anti): add `frac` of a position at `at` R in profit, the whole stop moved so a
# stop still loses exactly 1R (rules_agent.run, plan "pyramid"). "pyr0" = no adding, as the bots trade now. No
# martingale: nothing is added to a losing trade and the risk never goes over the bot's 1R.
PYRS = {"pyr1@0.5": (1.0, 0.5), "pyr1.5@0.5": (1.5, 0.5), "pyr1@0.33": (1.0, 0.33), "pyr2@0.5": (2.0, 0.5)}


def pyr_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block=None):
    """Every pyramid variant of one combo, on top of its plan as the bot trades it now (exits, filter, management)."""
    fn0 = R.STRATEGIES[sid][1]
    fn = fn0 if news_block is None else _blocked(fn0, news_block)
    t0, t1 = SPLIT, pd.Timestamp(args.manage, tz="UTC")
    pk = plan_keys(PLANS.get(f"{sid}|{tf}|{s}") or {})
    plan0 = {**fn(df)[2], **pk}
    flt = (PLANS.get(f"{sid}|{tf}|{s}") or {}).get("filter")
    regime = _regimes(df).get(flt) if flt else None
    rows, keep = [], {}
    for key, pv in [("pyr0", None)] + list(PYRS.items()):
        extra = {**pk, **({"pyramid": pv} if pv else {})}

        def g(d, extra=extra):
            lo, sh, plan = fn(d)
            if regime is not None:
                lo, sh = np.asarray(lo, bool) & regime[0], np.asarray(sh, bool) & regime[1]
            return lo, sh, {**plan, **extra}
        t = R.run(df, g, cost, breakeven=be, session_exit=sess)
        t = drop_hole_trades(t, df, s, tf)
        if t.empty:
            continue
        if real:
            t = C.apply(t, s, SPREADS.get(s))
        if events is not None:
            t = t[~(NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events))]
        tr, va = t[(t["time"] >= t0) & (t["time"] < t1)], t[t["time"] >= t1]
        rows.append({"variant": key, "train": _stats(tr["r"]), "valid": _stats(va["r"])})
        keep[key] = t[t["time"] >= t0]
    if not rows:
        return
    ok = [x for x in rows if x["train"]["n"] >= 20]
    best = max(ok or rows, key=lambda x: (x["train"]["sum"], x["train"]["pf"]))
    base = next((x for x in rows if x["variant"] == "pyr0"), None)
    OPT_RESULTS.append({"combo": f"{sid}|{tf}|{s}", "best": best, "base": base, "all": rows,
                        "has_target": plan0.get("target") is not None})
    for k in set(keep):
        COMBO_TRADES.append(keep[k].assign(sid=sid, tf=tf, symbol=s, variant=k))


def man_key(has_target, be, part):
    b = (f"be{be:g}" if has_target else f"ber{be:g}")
    return f"{b}|part{'0' if part is None else f'{part[0]:g}@{part[1]:g}'}"


def manage_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block=None):
    """Every breakeven point x partial exit of one combo, on top of its exits as the bot trades it now."""
    fn0 = R.STRATEGIES[sid][1]
    fn = fn0 if news_block is None else _blocked(fn0, news_block)
    t0, t1 = SPLIT, pd.Timestamp(args.manage, tz="UTC")
    pk = plan_keys(PLANS.get(f"{sid}|{tf}|{s}") or {})
    plan0 = {**fn(df)[2], **pk}
    has_target = plan0.get("target") is not None
    flt = (PLANS.get(f"{sid}|{tf}|{s}") or {}).get("filter")
    regime = _regimes(df).get(flt) if flt else None
    rows, keep = [], {}
    be_list, part_list = (MAN_BE, MAN_PART) if has_target else (MAN_BE_R, MAN_PART_R)
    if getattr(args, "manage_grid", "std") == "fib":
        if not has_target:
            return
        # the combo as the bot trades it now stays in the grid (the card's baseline): its own breakeven / partial
        own_be = float(pk.get("be_way", 0.6))
        own_part = tuple(pk["partial"]) if pk.get("partial") else None
        be_list = sorted(set(FIB_BE) | {own_be})
        part_list = FIB_PART + ([own_part] if own_part and own_part not in FIB_PART else [])
    for bv in be_list:
        for part in part_list:
            extra = {**pk, **({"be_way": bv} if has_target else {"be_r": bv or None, "lock_r": (3.0, 1.0) if bv else None})}
            if part is not None:
                extra["partial"] = part

            def g(d, extra=extra):
                lo, sh, plan = fn(d)
                if regime is not None:
                    lo, sh = np.asarray(lo, bool) & regime[0], np.asarray(sh, bool) & regime[1]
                return lo, sh, {**plan, **{k: v for k, v in extra.items() if v is not None},
                                **({"be_r": None, "lock_r": None} if not has_target and not bv else {})}
            t = R.run(df, g, cost, breakeven=be, session_exit=sess)
            t = drop_hole_trades(t, df, s, tf)
            if t.empty:
                continue
            if real:
                t = C.apply(t, s, SPREADS.get(s))
            if events is not None:
                t = t[~(NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events))]
            tr, va = t[(t["time"] >= t0) & (t["time"] < t1)], t[t["time"] >= t1]
            key = man_key(has_target, bv, part)
            rows.append({"variant": key, "train": _stats(tr["r"]), "valid": _stats(va["r"])})
            keep[key] = t[t["time"] >= t0]
    if not rows:
        return
    base_key = man_key(has_target, 0.6 if has_target else 2.5, None)
    ok = [x for x in rows if x["train"]["n"] >= 20]
    best = max(ok or rows, key=lambda x: (x["train"]["sum"], x["train"]["pf"]))
    base = next((x for x in rows if x["variant"] == base_key), None)
    OPT_RESULTS.append({"combo": f"{sid}|{tf}|{s}", "best": best, "base": base, "all": rows, "has_target": has_target})
    for k in set(keep):
        COMBO_TRADES.append(keep[k].assign(sid=sid, tf=tf, symbol=s, variant=k))


def optimize_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block=None):
    """Every grid variant of one combo (target x stop distance x entry): stats on the training window and
    on the later window. All variants' trades are kept for improve.py, which decides."""
    fn0 = R.STRATEGIES[sid][1]
    fn1 = fn0 if news_block is None else _blocked(fn0, news_block)
    # the market-regime filter the bot trades now (--plans-file, 13.11.2026), so the exit grid is measured on the
    # entries that really run
    flt = (PLANS.get(f"{sid}|{tf}|{s}") or {}).get("filter")
    regime = _regimes(df).get(flt) if flt else None

    def fn(d):
        lo, sh, plan = fn1(d)
        if regime is not None:
            lo, sh = np.asarray(lo, bool) & regime[0], np.asarray(sh, bool) & regime[1]
        return lo, sh, plan
    t0, t1 = SPLIT, pd.Timestamp(args.optimize, tz="UTC")
    rows, keep = [], {}
    bf = be[0] if be else 0.0
    # the review's live rules (14.11.2026): direction / hours on every variant; its target / stop factors only on the
    # strategy's own exits (an approved exit plan replaces them in the live bot, live/runner.combo_rules)
    rk_all = rule_keys(PLANS.get(f"{sid}|{tf}|{s}") or {}, exits=True)
    rk_dir = {k: v for k, v in rk_all.items() if k in ("rule_dir", "rule_hours")}

    def with_rules(g, extra):
        if not extra:
            return g

        def h(d):
            lo, sh, plan = g(d)
            return lo, sh, {**plan, **extra}
        return h
    for tg in OPT_TARGETS:
        for sc in OPT_STOP:
            for pb in OPT_ENTRY:
                rk = rk_all if (tg is None and sc == 1.0 and not pb) else rk_dir
                t = R.run(df, with_rules(_variant(fn, tg, sc, pb), rk), cost, breakeven=be, session_exit=sess)
                t = drop_hole_trades(t, df, s, tf)
                if t.empty:
                    continue
                if real:
                    t = C.apply(t, s, SPREADS.get(s))
                if events is not None:
                    t = t[~(NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events))]
                tr, va = t[(t["time"] >= t0) & (t["time"] < t1)], t[t["time"] >= t1]
                key = f"{'own' if tg is None else ('none' if tg == 'none' else f'{tg[1]:g}R')}|be{bf:g}|stop{sc:g}|entry{pb:g}"
                rows.append({"variant": key, "train": _stats(tr["r"]), "valid": _stats(va["r"])})
                keep[key] = t[t["time"] >= t0]
    if not rows:
        return
    base_key = f"own|be{bf:g}|stop1|entry0"
    ok = [x for x in rows if x["train"]["n"] >= 20]
    best = max(ok or rows, key=lambda x: (x["train"]["sum"], x["train"]["pf"]))
    base = next((x for x in rows if x["variant"] == base_key), None)
    OPT_RESULTS.append({"combo": f"{sid}|{tf}|{s}", "best": best, "base": base, "all": rows})
    for k in (set(keep) if OPT_ALL_TRADES else {best["variant"], base_key} & set(keep)):
        COMBO_TRADES.append(keep[k].assign(sid=sid, tf=tf, symbol=s, variant=k))
VARIANT = [""]
MISSING = set()   # instruments with no history for a timeframe
HOLES = {}        # "sym tf" -> data holes found; trades held across one are dropped (09.11.2026: SOL/USD had no
                  # daily bars 09.2022-09.2024, so a short there was "stopped" two years late at the reopening price)
HOLE_DAYS = {"crypto": 4.0}   # longest normal pause between bars, calendar days (others: 6 = long weekend + holiday)


def drop_hole_trades(t, df, s, tf):
    """Drop trades whose holding period spans a data hole (a gap far longer than any market pause)."""
    if t.empty or len(df) < 2:
        return t
    lim = pd.Timedelta(days=HOLE_DAYS.get(U.CLASS.get(s), 6.0))
    idx = df.index
    gap = idx[1:] - idx[:-1]
    at = np.flatnonzero(gap > lim)
    if not len(at):
        return t
    HOLES[f"{s} {tf}"] = [f"{idx[i]:%Y-%m-%d}..{idx[i + 1]:%Y-%m-%d}" for i in at]
    bad = np.zeros(len(t), bool)
    ti, te = pd.DatetimeIndex(t["time"]), pd.DatetimeIndex(t["exit_time"])
    for i in at:
        bad |= (ti <= idx[i]) & (te >= idx[i + 1])
    return t[~bad]
NO_TRADES = []   # strategy/timeframe/instrument that produced no trade at all, so nothing to judge


def _with_be_r(fn, k):
    def g(df):
        L, S, plan = fn(df)
        return L, S, {**plan, "be_r": k}
    return g


def _with_plan(fn, **kw):
    def g(df):
        L, S, plan = fn(df)
        return L, S, {**plan, **kw}
    return g


def _shifted(fn, k):
    """Null version of a strategy: signals and every per-bar plan array moved k bars later (entries are no
    longer tied to the bars that produced them). Levels move with their signal, so stops/targets stay valid
    distances from that signal's price; the engine skips the ones now on the wrong side."""
    import numpy as _np

    def g(df):
        L, S, plan = fn(df)
        sh = lambda a: _np.r_[_np.zeros(k, dtype=_np.asarray(a).dtype), _np.asarray(a)[:-k]] if len(a) > k else _np.zeros_like(a)  # noqa: E731
        shf = lambda a: _np.r_[_np.full(k, _np.nan), _np.asarray(a, float)[:-k]] if len(a) > k else _np.full(len(a), _np.nan)  # noqa: E731
        p2 = dict(plan)
        for key in ("stop", "target"):
            v = plan.get(key)
            if v and v[0] == "level":
                p2[key] = ("level", shf(v[1]), shf(v[2]))
        for key in ("exit_long", "exit_short"):
            if plan.get(key) is not None:
                p2[key] = sh(_np.asarray(plan[key], bool))
        if plan.get("line") is not None:
            p2["line"] = shf(plan["line"])
        if plan.get("trail_after_px") is not None:
            p2["trail_after_px"] = shf(plan["trail_after_px"])
        return sh(_np.asarray(L, bool)), sh(_np.asarray(S, bool)), p2
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-29")
    ap.add_argument("--only", default="", help="comma separated strategy ids, e.g. TV01,TV03")
    ap.add_argument("--market", default="ftmo", choices=["forex", "ftmo"],
                    help="forex = 5 pairs + gold/silver/oil; ftmo = also indices, more commodities, crypto")
    ap.add_argument("--export", default="", help="strategy:timeframe pairs to export trade by trade, "
                    "e.g. TV01:4Hour,TV09:1Hour (survivors are always exported)")
    ap.add_argument("--timeframes", default="", help="comma separated subset of 1Min,5Min,15Min,1Hour,4Hour (empty = all)")
    ap.add_argument("--respect-tf", action="store_true",
                    help="only run a strategy on the timeframes its source states (default: every timeframe)")
    ap.add_argument("--symbols", default="", help="comma separated subset of the instrument universe (empty = all)")
    ap.add_argument("--news-exit", action="store_true",
                    help="FTMO calendar holidays (data/ftmo_calendar/history.json) close the affected markets for the day; "
                         "restricted events are applied minute by minute later (bots_compare.py --news)")
    ap.add_argument("--news-minute", action="store_true",
                    help="FTMO restricted events minute by minute on every trade, like the live bots (13.11.2026): a position "
                         "open 3 minutes before an event on its instrument is closed then at the 1-minute price, a signal "
                         "inside the window (3 minutes before to 3 after) is not taken")
    ap.add_argument("--break-exit", action="store_true",
                    help="FTMO Standard account: close every position before any market break longer than 2 hours (weekends too)")
    ap.add_argument("--stock-session-exit", action="store_true",
                    help="stocks: close every position at the end of the trading session (no overnight gaps)")
    ap.add_argument("--breakeven", type=float, default=0.0,
                    help="move the stop to entry + spread/commission once price covered this share of the way to the target (e.g. 0.6)")
    ap.add_argument("--be-r", type=float, default=0.0,
                    help="strategies with no target: stop to entry + spread/commission once this many R in profit (0 = off)")
    ap.add_argument("--trail-r", default="",
                    help="strategies with no target: 'start:dist' - from start R in profit, the stop trails dist R behind the best price")
    ap.add_argument("--lock-r", default="",
                    help="strategies with no target: 'a:b' - once a R in profit, the stop goes to entry + b R (locks b R)")
    ap.add_argument("--optimize", default="",
                    help="YYYY-MM-DD: with --combos-file, try a grid of trade-management values per combo, choose on "
                         "--test-from..this date, check on this date..end")
    ap.add_argument("--manage", default="",
                    help="YYYY-MM-DD: with --combos-file, try breakeven points x partial exits per combo (on top of its "
                         "--plans-file plan), choose on --test-from..this date, check on this date..end")
    ap.add_argument("--manage-grid", default="std", choices=["std", "fib", "ladder", "pyr"],
                    help="--manage values: std (40-80%% of the way, 30/50%% out), fib (Fibonacci levels of the way) or "
                         "ladder (the entry split into 2-4 limit orders, same total risk)")
    ap.add_argument("--plans-file", default="",
                    help="a bot's live/bots/<bot>.json: its built per-combo plans are the base of --manage")
    ap.add_argument("--filters", default="",
                    help="YYYY-MM-DD: with --combos-file, try market-regime filters per combo, choose on "
                         "--test-from..this date, check on this date..end")
    ap.add_argument("--filters-extra", action="store_true",
                    help="--filters also tries the user's EMA 9/20 direction and Bollinger-width filters (14.11.2026)")
    ap.add_argument("--combos-file", default="",
                    help="file with sid|timeframe|symbol lines: run only these and save their trades (no Prophet)")
    ap.add_argument("--stock-spread-mult", type=float, default=0.0,
                    help="stocks: spread x this (FTMO's live stock spreads were ~2.4x the history, 10.11.2026)")
    ap.add_argument("--est-spread", type=float, default=0.0,
                    help="spread (share of price) for stocks without bid/ask history (default 0.0003): a stress test")
    ap.add_argument("--null-shift", type=int, default=0,
                    help="NULL TEST: every strategy's signals (and their stop/target arrays) are moved K bars later, so "
                         "they no longer match the prices they were computed from: no edge left. What passes is luck.")
    ap.add_argument("--test-from", default="",
                    help="YYYY-MM-DD: judge every (strategy, timeframe, instrument) directly from this date, no earlier selection period")
    args = ap.parse_args()
    if args.news_minute and not args.demo:
        NEWS_MIN["on"] = True
        NEWS_MIN["start"], NEWS_MIN["end"] = args.start, args.end
        R.run = _news_run(R.run)
    if args.plans_file:
        import json as _json
        try:
            PLANS.update(_json.loads(Path(args.plans_file).read_text(encoding="utf-8")).get("plans") or {})
        except Exception as e:  # noqa: BLE001
            print(f"plans file not read: {e}")
    if args.est_spread:
        C.EST_SPREAD["stock"] = args.est_spread
    if args.stock_spread_mult:
        C.EST_SPREAD["stock"] = C.EST_SPREAD["stock"] * args.stock_spread_mult
    global SPLIT, SPLIT_HE
    if args.test_from:
        SPLIT = pd.Timestamp(args.test_from, tz="UTC")
        SPLIT_HE = SPLIT.strftime("%d.%m.%Y")
        TEST_FROM_MODE[0] = True
    VARIANT[0] = " · ".join(x for x in [
        "מניות נסגרות בסוף יום המסחר" if args.stock_session_exit else "",
        f"סטופ לנקודת איזון + ספרד ועמלה אחרי {args.breakeven:.0%} מהדרך ליעד" if args.breakeven else ""] if x) or "בלי ניהול עסקה נוסף"
    ids = [x for x in args.only.split(",") if x] or list(R.STRATEGIES)
    combos = None
    if args.combos_file:
        combos = {tuple(l.strip().split("|")) for l in open(args.combos_file, encoding="utf-8") if l.strip()}
        ids = sorted({c[0] for c in combos})
    events = None
    if not args.demo:
        events, real_end = NC.load()
    results, missing = [], set()
    tfs = [x for x in args.timeframes.split(",") if x] or TIMEFRAMES
    for tf in (["1Hour"] if args.demo else tfs):
        per_sid = {sid: [] for sid in ids}
        universe = U.SYMBOLS if args.market == "ftmo" else SYMBOLS
        if args.symbols:
            universe = [x for x in list(U.ALL) + SYMBOLS if x in args.symbols.split(",")]   # also research-only names
            universe = list(dict.fromkeys(universe))
        if combos is not None:   # every listed name, research-only ones included
            universe = [x for x in dict.fromkeys(list(U.ALL) + SYMBOLS) if any(c[1] == tf and c[2] == x for c in combos)]
        for i, s in enumerate(["DEMO1", "DEMO2"] if args.demo else universe):
            st = max(args.start, RP.MINUTE_START) if tf.endswith("Min") else args.start
            try:
                df = RP.demo_bars(i) if args.demo else P.bars(s, tf, st, args.end)
            except config.TimeBudgetReached:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  {s} {tf}: NO DATA ({str(e)[-200:]})")
                missing.add(s)
                continue
            if df is None or df.empty:
                print(f"  {s} {tf}: NO DATA (empty)")
                missing.add(s)
                continue
            df.attrs.update(symbol=s, tf=tf)              # strategies that use a partner instrument read these
            real = (not args.demo) and s in U.ALL       # FTMO's own costs, applied per trade below
            cost = 0.00005 if args.demo else (0.0 if real else RP.cost_of(s))
            if real and s not in SPREADS:
                try:
                    SPREADS[s] = P.spread_1m(s, (SPLIT - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), args.end)
                except config.TimeBudgetReached:
                    raise
                except Exception:  # noqa: BLE001
                    ERRORS.append(f"spread {s}:\n{traceback.format_exc()}")
                    SPREADS[s] = None
                if SPREADS[s] is None or len(SPREADS[s]) == 0:
                    SPREAD_EST.add(s)
                elif args.stock_spread_mult and U.CLASS.get(s) == "stock":
                    SPREADS[s] = SPREADS[s] * args.stock_spread_mult
            sess = session_mask(df, s, tf) if (args.stock_session_exit and not args.demo) else None
            if args.break_exit and not args.demo:
                bm = break_mask(df, tf)
                sess = bm if sess is None else (np.asarray(sess, dtype=bool) | bm)
            news_block = None
            if args.news_exit and not args.demo:
                nm, news_block = news_masks(df, s, tf)
                sess = nm if sess is None else (np.asarray(sess, dtype=bool) | nm)
            if args.stock_session_exit and not args.demo and U.CLASS.get(s) == "stock":
                # the live bots open no stock trade in the last 15 minutes of the session (live/runner.py), the
                # backtest did: entries at 15:45 / 15:50 / 15:55 New York held to the 16:00 close - in bots 1-4's
                # stock combos 2025-2026: 192 trades, -0.13R each in 2026 (13.11.2026) - blocked here the same way
                late = stock_late_mask(df, s)
                news_block = late if news_block is None else (np.asarray(news_block, dtype=bool) | late)
            be = None
            if args.breakeven:
                px = float(df["c"].median())
                off = 0.00005
                if real:
                    sp = SPREADS.get(s)
                    sp_px = float(sp.median()) if sp is not None and len(sp) else px * C.EST_SPREAD.get(U.CLASS[s], 0.0003)
                    spec = U.spec(s)
                    if spec["commissionType"] == "flat_USD":
                        comm = px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
                    else:
                        comm = px * 2 * float(spec["commission"]) / 100.0
                    off = (sp_px + comm) / px
                be = (args.breakeven, off)
            for sid in ids:
                if combos is not None and (sid, tf, s) not in combos:
                    continue
                if args.respect_tf and not R.timeframe_ok(sid, tf):
                    continue
                if args.manage and args.manage_grid == "pyr":
                    try:
                        pyr_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block)
                    except Exception:  # noqa: BLE001
                        ERRORS.append(f"pyr {sid} {tf} {s}:\n{traceback.format_exc()}")
                    continue
                if args.manage and args.manage_grid == "ladder":
                    try:
                        ladder_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block)
                    except Exception:  # noqa: BLE001
                        ERRORS.append(f"ladder {sid} {tf} {s}:\n{traceback.format_exc()}")
                    continue
                if args.manage:
                    try:
                        manage_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block)
                    except Exception:  # noqa: BLE001
                        ERRORS.append(f"manage {sid} {tf} {s}:\n{traceback.format_exc()}")
                    continue
                if args.filters:
                    try:
                        filter_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block)
                    except Exception:  # noqa: BLE001
                        ERRORS.append(f"filters {sid} {tf} {s}:\n{traceback.format_exc()}")
                    continue
                if args.optimize:
                    try:
                        optimize_combo(df, sid, tf, s, cost, sess, be, real, events, args, news_block)
                    except Exception:  # noqa: BLE001
                        ERRORS.append(f"optimize {sid} {tf} {s}:\n{traceback.format_exc()}")
                    continue
                try:
                    fn0 = R.STRATEGIES[sid][1]
                    if args.null_shift:
                        fn0 = _shifted(fn0, args.null_shift)
                    if args.be_r:
                        fn0 = _with_be_r(fn0, args.be_r)
                    if args.lock_r:
                        fn0 = _with_plan(fn0, lock_r=tuple(float(x) for x in args.lock_r.split(":")))
                    if args.trail_r:
                        fn0 = _with_plan(fn0, trail2_r=tuple(float(x) for x in args.trail_r.split(":")))
                    t = R.run(df, fn0 if news_block is None else _blocked(fn0, news_block), cost, breakeven=be, session_exit=sess)
                except Exception:  # noqa: BLE001 - one broken strategy must not kill the run
                    ERRORS.append(f"{sid} {tf} {s}:\n{traceback.format_exc()}")
                    continue
                if t.empty:
                    NO_TRADES.append(f"{sid} {RP.TF_HE[tf]} {s}")
                    continue
                t = drop_hole_trades(t, df, s, tf)
                if t.empty:
                    NO_TRADES.append(f"{sid} {RP.TF_HE[tf]} {s}")
                    continue
                if real:
                    t = C.apply(t, s, SPREADS.get(s))
                if events is not None:
                    bad = NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events)
                    t = t[~bad]
                per_sid[sid].append(t.assign(symbol=s))
            print(f"  {s} {tf}: {len(df):,} bars")
        split = (pd.Timestamp("2018-07-01", tz="UTC") if args.demo else SPLIT)
        if combos is not None:
            for sid, parts in per_sid.items():
                for t in parts:
                    a = t[t["time"] >= split]
                    if len(a):
                        COMBO_TRADES.append(a.assign(sid=sid, tf=tf))
            continue
        for sid, parts in per_sid.items():
            if not parts:
                continue
            try:
                if not args.test_from:  # pooled all-instruments summary only in the old mode
                    _summarise(sid, tf, parts, split, args, results)
            except Exception:  # noqa: BLE001
                ERRORS.append(f"{sid} {tf} summary:\n{traceback.format_exc()}")
            try:
                _per_symbol(sid, tf, parts, split, args)
            except Exception:  # noqa: BLE001
                ERRORS.append(f"{sid} {tf} per-symbol:\n{traceback.format_exc()}")
    try:
        if not args.test_from:
            _portfolio(args)
    except Exception:  # noqa: BLE001
        ERRORS.append(f"portfolio:\n{traceback.format_exc()}")
    MISSING.update(missing)
    if args.filters:
        tag = (args.symbols or "all").replace(",", "_").replace("/", "-")[:60]
        import json as _json
        (config.OUTPUT_DIR / f"filters_{tag}.json").write_text(_json.dumps(OPT_RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
        print("filtered", len(OPT_RESULTS), "combos")
    if args.manage:
        tag = (args.symbols or "all").replace(",", "_").replace("/", "-")[:60]
        import json as _json
        (config.OUTPUT_DIR / f"manage_{tag}.json").write_text(_json.dumps(OPT_RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
        print("managed", len(OPT_RESULTS), "combos")
    if args.optimize:
        tag = (args.symbols or "all").replace(",", "_").replace("/", "-")[:60]
        import json as _json
        (config.OUTPUT_DIR / f"optimize_{tag}.json").write_text(_json.dumps(OPT_RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
        print("optimized", len(OPT_RESULTS), "combos")
    if combos is not None or (args.test_from and COMBO_TRADES):
        cols = ["sid", "tf", "symbol", "time", "exit_time", "dir", "r", "entry_px", "stop_px", "exit_px", "target_px", "reason", "free_time",
                "mfe_r", "mae_r"]
        if args.optimize or args.filters or args.manage:
            cols = cols + ["variant"]
        out = pd.concat(COMBO_TRADES, ignore_index=True).reindex(columns=cols) if COMBO_TRADES else pd.DataFrame(columns=cols)
        tag = (args.symbols or "all").replace(",", "_").replace("/", "-")[:60]
        out.to_csv(config.OUTPUT_DIR / f"combo_trades_{tag}.csv.gz", index=False)
        print("saved", len(out), "trades for", out.groupby(["sid", "tf", "symbol"]).ngroups if len(out) else 0, "combos")
    if ALL_TRADES:
        cols = ["sid", "tf", "symbol", "time", "exit_time", "dir", "r", "entry_px", "stop_px", "exit_px", "target_px", "reason",
                "mfe_r", "mae_r", "passed"]
        tag = (args.symbols or "all").replace(",", "_").replace("/", "-")[:60]
        pd.concat(ALL_TRADES, ignore_index=True).reindex(columns=cols).to_csv(config.OUTPUT_DIR / f"all_trades_{tag}.csv.gz", index=False)
        print("saved every combo's trades:", sum(len(x) for x in ALL_TRADES))
    safe_report(results, args)


SESSION = {"US": ("America/New_York", 16 * 60), "EU": ("Europe/Berlin", 17 * 60 + 30)}


_CAL_HIST = None


def news_masks(df, s, tf):
    """(exit mask, no-entry mask) for FTMO restricted events and holidays: see ftmo_calendar.masks."""
    global _CAL_HIST
    from agents import ftmo_calendar as CAL
    if _CAL_HIST is None:
        _CAL_HIST = CAL.parse(json.load(open(Path(__file__).parent / "data" / "ftmo_calendar" / "history.json", encoding="utf-8")))
    return CAL.masks(df, s, RP.TF_MIN[tf], *_CAL_HIST)


# ---- FTMO restricted events, minute by minute (13.11.2026) ----
# The live bots close every position 3 minutes before a restricted event on its instrument and take no entry from 3
# minutes before to 3 after (live/runner.py, "restricted news ahead Standard"). The bar engine did not: a trade held
# through the release kept the move the release made (bot 2 QB11 USDJPY 9.11: the backtest +7.7R at the 8R target,
# the demo closed at 13:27 UTC before the US CPI at +2.8R; in 2026 13 of QB11's 124 trades were open over an event and
# made +21.5R of its +20.4R, TV58 USDCAD 9 of 37 made +29.9R of +23.9R).
NEWS_MIN = {"on": False}
_MIN1 = {}


def _minute_bars(s):
    if s not in _MIN1:
        try:
            _MIN1[s] = P.bars(s, "1Min", NEWS_MIN["start"], NEWS_MIN["end"])
        except config.TimeBudgetReached:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"  news: no 1-minute bars for {s} ({str(e)[-120:]}) - restricted events not applied")
            _MIN1[s] = None
    return _MIN1[s]


def _news_events(s):
    global _CAL_HIST
    from agents import ftmo_calendar as CAL
    if _CAL_HIST is None:
        _CAL_HIST = CAL.parse(json.load(open(Path(__file__).parent / "data" / "ftmo_calendar" / "history.json", encoding="utf-8")))
    return pd.DatetimeIndex(sorted({t for t, ins in _CAL_HIST[0] if CAL.restricted_affects(s, ins)}))


def news_minute(t, df, mb=None, ev=None):
    """The trades of one run with the live rule for restricted events applied. Each trade's exit happened somewhere in its
    exit bar; when the window start (event - 3 min) falls inside that bar, the 1-minute path decides whether the stop or
    the target came first. A trade that took a partial exit before the window is left as it is (its split is not known
    here - rare). mb / ev: 1-minute bars and event times (tests); else loaded."""
    if t is None or t.empty:
        return t
    s, tf = df.attrs.get("symbol"), df.attrs.get("tf")
    if not s or not tf:
        return t
    ev = _news_events(s) if ev is None else ev
    if not len(ev):
        return t
    mb = _minute_bars(s) if mb is None else mb
    if mb is None or mb.empty:
        return t
    bar = pd.Timedelta(minutes=RP.TF_MIN[tf])
    win_a, win_b = ev - pd.Timedelta(minutes=3), ev + pd.Timedelta(minutes=3)
    mi, mo, mh, ml = mb.index, mb["o"].to_numpy(float), mb["h"].to_numpy(float), mb["l"].to_numpy(float)
    keep, rows = [], t.to_dict("records")
    for r in rows:
        t0 = pd.Timestamp(r["time"])
        k = win_a.searchsorted(t0 - pd.Timedelta(minutes=6))          # first window that can touch the trade
        if k < len(ev) and win_a[k] <= t0 <= win_b[k]:
            continue                                                  # signal inside the window: not taken
        x0 = pd.Timestamp(r["exit_time"])
        j = win_a.searchsorted(t0, side="right")                      # first window starting after the entry
        if j >= len(ev) or win_a[j] >= x0 + bar:
            keep.append(r)
            continue
        a = win_a[j]
        if a > x0:                                                    # the window starts inside the exit bar:
            i0, i1 = mi.searchsorted(x0), mi.searchsorted(a)          # did the exit come before it?
            if i1 > i0:
                d, st, tg = r["dir"], r.get("stop_last", r["stop_px"]), r.get("target_px")
                lo, hi = ml[i0:i1].min(), mh[i0:i1].max()
                hit_stop = (lo <= st) if d > 0 else (hi >= st)
                hit_tg = tg == tg and tg is not None and ((hi >= tg) if d > 0 else (lo <= tg))
                if hit_stop or hit_tg or r.get("reason") in ("סיגנל יציאה", "סיגנל הפוך"):
                    keep.append(r)
                    continue
        pt = r.get("part_time")
        if pt is not None and pt == pt and pd.Timestamp(pt) <= a:
            keep.append(r)
            continue
        i = mi.searchsorted(a)
        if i >= len(mi) or mi[i] - a > pd.Timedelta(minutes=30):
            keep.append(r)
            continue
        e0, st0, d = r["entry_px"], r["stop_px"], r["dir"]
        risk = abs(e0 - st0)
        if risk <= 0:
            keep.append(r)
            continue
        cost_px = (d * (r["exit_px"] - e0) / risk - r["r"]) * risk      # the run's costs, in price, kept
        px = float(mo[i])
        r["exit_time"], r["exit_px"], r["reason"] = a, px, "נסגרה לפני הודעה מוגבלת"
        r["r"] = d * (px - e0) / risk - cost_px / risk
        r["held"] = int(max(0, (a - t0) / bar))
        ft = r.get("free_time")
        if ft is not None and ft == ft and pd.Timestamp(ft) > a:
            r["free_time"] = pd.NaT
        keep.append(r)
    return pd.DataFrame(keep, columns=t.columns)


def _news_run(orig):
    def run(df, fn, cost_side, breakeven=None, session_exit=None):
        t = orig(df, fn, cost_side, breakeven=breakeven, session_exit=session_exit)
        try:
            return news_minute(t, df)
        except config.TimeBudgetReached:
            raise
        except Exception:  # noqa: BLE001
            ERRORS.append(f"news minute {df.attrs.get('symbol')}:\n{traceback.format_exc()}")
            return t
    return run


def _blocked(fn, noent):
    from agents import ftmo_calendar as CAL
    return CAL.blocked(fn, noent)


def break_mask(df, tf):
    """True on the last bar before a market break longer than 2 hours (FTMO Standard account
    rule: no positions held through such breaks, weekends included)."""
    start = df.index.to_series()
    gap = start.shift(-1) - (start + pd.Timedelta(minutes=RP.TF_MIN[tf]))
    return (gap > pd.Timedelta(hours=2)).to_numpy()


def session_mask(df, s, tf):
    """Stocks only: True on the bars whose close is within 15 minutes of the exchange close,
    or that are the last bar of their trading day."""
    if U.CLASS.get(s) != "stock":
        return None
    tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[s] else SESSION["EU"]
    loc = df.index.tz_convert(tz)
    end = loc + pd.Timedelta(minutes=RP.TF_MIN[tf])
    mins = end.hour * 60 + end.minute + (end.normalize() > loc.normalize()) * 1440
    day = pd.Series(loc.normalize(), index=df.index)
    last = (day != day.shift(-1)).to_numpy()
    return (mins >= close_min - 15) | last


def stock_late_mask(df, s):
    """Stocks: True on the bars that open in the last 15 minutes of the exchange session - the live bots take no new
    stock trade then (live/runner.py, 'last 15 minutes of the stock session')."""
    tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[s] else SESSION["EU"]
    loc = df.index.tz_convert(tz)
    return np.asarray(loc.hour * 60 + loc.minute >= close_min - 15, dtype=bool)


def safe_report(results, args):
    """Write the report; a result that breaks the HTML is dropped and its traceback shown."""
    try:
        report(results, args)
        return
    except Exception:  # noqa: BLE001
        ERRORS.append(f"report:\n{traceback.format_exc()}")
    good = []
    for r in results:
        try:
            report([r], args)
            good.append(r)
        except Exception:  # noqa: BLE001
            ERRORS.append(f"report row {r.get('sid')} {r.get('tf')}:\n{traceback.format_exc()}")
    report(good, args)


def _per_symbol(sid, tf, parts, split, args):
    """Instrument-by-instrument: choose (strategy, timeframe, instrument) on the EARLIER period only
    (trades >= 30, avg >= +0.05R, t >= 1.5), then judge it on the last year with Trading Prophet,
    trading that one instrument alone. No last-year information is used for the choice."""
    for t in parts:
        if t.empty:
            continue
        sym = t["symbol"].iloc[0]
        before, after = t[t["time"] < split], t[t["time"] >= split]
        b = RP.stats(before["r"])
        a = RP.stats(after["r"])
        if args.test_from:
            pass  # no selection period and no skipping: every strategy/timeframe/instrument is judged on its own
        elif not (b["trades"] >= SEL_MIN_TRADES and b["avg_r"] >= SEL_MIN_AVG_R and (b["t"] or 0) >= SEL_MIN_T):
            continue
        row = {"sid": sid, "tf": tf, "symbol": sym, "before": b, "after": a, "tests": {}}
        if len(after) >= (1 if args.test_from else 5):
            for rk in PER_SYMBOL_RISKS:
                w2 = F.account_path(after, rk, start=split)
                led = w2[0].attrs.get("ledger") if len(w2[0]) else None
                row["tests"][rk] = {"account": w2, "prophet": prophet_check(led, args) if led is not None else None}
            passed = any(x["prophet"] and x["prophet"]["verdict"]["verdict"] == "PASS" for x in row["tests"].values())
            if passed and args.test_from:
                COMBO_TRADES.append(after.assign(sid=sid, tf=tf))
            if args.test_from:
                ALL_TRADES.append(after.assign(sid=sid, tf=tf, passed=passed))
            if not args.test_from:
                SELECTED_AFTER.append(after.assign(combo=f"{sid}|{tf}|{sym}"))
        PER_SYMBOL.append(row)


def _portfolio(args):
    """All selected (strategy, timeframe, instrument) combos traded together in ONE $100K account
    over the last year - the honest out-of-sample test of the whole selection."""
    if not SELECTED_AFTER:
        return
    tr = pd.concat(SELECTED_AFTER, ignore_index=True).sort_values("time").reset_index(drop=True)
    PORTFOLIO["combos"] = tr["combo"].nunique()
    PORTFOLIO["after"] = RP.stats(tr["r"])
    PORTFOLIO["tests"] = {}
    for rk in (0.0025, 0.005, 0.0075):
        w2 = F.account_path(tr, rk, start=SPLIT)
        led = w2[0].attrs.get("ledger") if len(w2[0]) else None
        PORTFOLIO["tests"][rk] = {"account": w2, "prophet": prophet_check(led, args) if led is not None else None}


def _per_symbol_html():
    if not PER_SYMBOL:
        return "<h2>התאמה לנכס ולגרף</h2><p class='mut'>אף שילוב של אסטרטגיה, גרף ונכס לא עבר את סינון התקופה הקודמת.</p>"
    def best(row):
        ok = [rk for rk, t in row["tests"].items() if t["prophet"] and t["prophet"]["verdict"]["verdict"] == "PASS"]
        return max(ok) if ok else None
    rows = sorted(PER_SYMBOL, key=lambda r: (best(r) is None, -(r["after"]["avg_r"] if r["after"]["avg_r"] == r["after"]["avg_r"] else -9)))
    body = ""
    for r in rows:
        bk = best(r)
        cells = "".join(f"<td>{_acc_text(t['account'])}<br>{_prophet_cell(t['prophet'])}</td>" for rk, t in sorted(r["tests"].items()))
        cells += "<td></td>" * (len(PER_SYMBOL_RISKS) - len(r["tests"]))
        a, b = r["after"], r["before"]
        f2 = lambda v: "—" if v != v else f"{v:+.2f}"
        body += (f"<tr><td>{'✅' if bk else '❌'}</td><td>{r['sid']}</td><td>{html.escape(R.STRATEGIES[r['sid']][0])}</td>"
                 f"<td>{RP.TF_HE[r['tf']]}</td><td>{r['symbol']}</td><td>{'—' if TEST_FROM_MODE[0] else f"{b['trades']} · {f2(b['avg_r'])}R · t={b['t']:.1f}"}</td>"
                 f"<td>{a['trades']} · {f2(a['avg_r'])}R</td><td>{'—' if bk is None else f'{bk * 100:g}%'}</td>{cells}</tr>")
    heads = "".join(f"<th>סיכון {rk * 100:g}%</th>" for rk in PER_SYMBOL_RISKS)
    port = ""
    if PORTFOLIO:
        prow = "".join(f"<tr><td>{rk * 100:g}%</td><td>{_acc_text(t['account'])}</td><td>{_prophet_cell(t['prophet'])}</td></tr>"
                       for rk, t in sorted(PORTFOLIO["tests"].items()))
        pa = PORTFOLIO["after"]
        port = (f"<h3>תיק משולב: כל {PORTFOLIO['combos']} השילובים שנבחרו, בחשבון אחד</h3><p class='mut'>השנה האחרונה: "
                f"{pa['trades']:,} עסקאות, ממוצע {pa['avg_r']:+.2f}R</p><div class='tw'><table><tr><th>סיכון</th><th>חשבון $100K</th>"
                f"<th>Trading Prophet</th></tr>{prow}</table></div>")
    n_ok = sum(1 for r in PER_SYMBOL if best(r))
    if TEST_FROM_MODE[0]:
        return (f"<h2>התאמה לנכס ולגרף</h2><p class='mut'>בלי תקופת בחירה: כל שילוב של אסטרטגיה, גרף ונכס נבדק ישירות ב־Trading Prophet "
                f"מ־{SPLIT_HE}, לבד בחשבון משלו (בלי דילוגים). "
                f"נבדקו {len(PER_SYMBOL)} שילובים, {n_ok} עברו. "
                f"{len(NO_TRADES)} שילובים לא פתחו אף עסקה בתקופה, אז אין מה לבדוק בהם.</p>{port}<div class='tw'><table><tr><th></th><th>#</th>"
                f"<th>אסטרטגיה</th><th>גרף</th><th>נכס</th><th>לפני (עסקאות · ממוצע · t)</th><th>תקופת הבדיקה</th>"
                f"<th>סיכון מקסימלי שעובר</th>{heads}</tr>{body}</table></div>")
    return (f"<h2>התאמה לנכס ולגרף</h2><p class='mut'>שלב 1 (בחירה, {SEL_MIN_TRADES}+ עסקאות, ממוצע ≥{SEL_MIN_AVG_R:+.2f}R, t≥{SEL_MIN_T}) "
            f"על התקופה שלפני {SPLIT_HE} בלבד. שלב 2: Trading Prophet על השנה האחרונה, כל שילוב לבד בחשבון משלו. "
            f"נבחרו {len(PER_SYMBOL)} שילובים, {n_ok} עברו את Prophet.</p>{port}<div class='tw'><table><tr><th></th><th>#</th>"
            f"<th>אסטרטגיה</th><th>גרף</th><th>נכס</th><th>לפני (עסקאות · ממוצע · t)</th><th>השנה האחרונה</th>"
            f"<th>סיכון מקסימלי שעובר</th>{heads}</tr>{body}</table></div>")


def _summarise(sid, tf, parts, split, args, results):
    """Account replay, FTMO challenge odds and Trading Prophet for one strategy/timeframe."""
    tr = pd.concat(parts, ignore_index=True).sort_values("time")
    before, after = tr[tr["time"] < split], tr[tr["time"] >= split]
    acc = {rk: F.account_path(after, rk, start=split) for rk in RP.RISKS} if len(after) >= 5 else {}
    ch = F.evaluate(after[["time", "exit_time", "r"]]) if len(after) >= 20 else None
    _, ch_best = F.verdict(ch)
    res = {"sid": sid, "name": R.STRATEGIES[sid][0], "tf": tf, "before": RP.stats(before["r"]),
           "after": RP.stats(after["r"]), "accounts": acc, "ftmo_best": ch_best,
           "by_sym": {s: RP.stats(q["r"]) for s, q in after.groupby("symbol")}}
    w = acc.get(RP.MAIN_RISK, (pd.DataFrame(), None, None))
    res["ok"] = bool(len(after) >= 100 and w[1] is None and len(w[0]) and w[0]["balance"].iloc[-1] > F.INITIAL)
    results.append(res)
    if (res["ok"] or f"{sid}:{tf}" in args.export.split(",")) and len(w[0]):
        export_trades(sid, tf, w[0].attrs.get("ledger"))
        res["prophet"] = prophet_check(w[0].attrs.get("ledger"), args)
    res["risk_tests"] = {}
    # lower risk cannot make a losing system pass, so the 4-level sweep runs only where the average trade is positive
    sweep = sid.startswith("TV") and len(after) and res["after"]["avg_r"] > 0
    for rk in (SWEEP_RISKS if (sweep and POOLED_SWEEP) else ()):
        if len(after) < 5:
            continue
        w2 = acc[rk] if rk in acc else F.account_path(after, rk, start=split)
        led = w2[0].attrs.get("ledger") if len(w2[0]) else None
        pr = prophet_check(led, args) if led is not None and len(after) <= MAX_PROPHET_TRADES else None
        res["risk_tests"][rk] = {"account": w2, "prophet": pr, "skipped": len(after) > MAX_PROPHET_TRADES}
    print(sid, tf, "trades", len(after), "avgR", round(res["after"]["avg_r"], 3) if len(after) else None)


def prophet_check(led, args):
    """Trading Prophet: replay the executed trades minute by minute (floating loss on equity,
    as FTMO measures it). The account replay above only checks limits when a trade closes."""
    if led is None or led.empty or args.demo:
        return None
    sys.path.insert(0, str(Path(__file__).parent / "skills" / "trading-prophet" / "scripts"))
    from ledger_adapter import evaluate_ledger
    frames = {}
    start = max(pd.Timestamp(led["time"].min()), pd.Timestamp(RP.MINUTE_START, tz="UTC"))
    for s in led["symbol"].unique():
        if s not in _BARS:
            try:
                _BARS[s] = P.bars(s, "1Min", (SPLIT - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), args.end)  # last year only: memory
            except Exception as e:  # noqa: BLE001
                print(f"  prophet: no 1-minute bars for {s}: {e}")
                _BARS[s] = None
        if _BARS[s] is not None:
            frames[s] = _BARS[s][_BARS[s].index >= start]
    try:
        res = evaluate_ledger(led, frames)
    except Exception as e:  # noqa: BLE001
        print(f"  prophet failed: {e}")
        return None
    print(f"  Trading Prophet: {res['verdict']['verdict']} {res['verdict']['reasons']}")
    return res


def _prophet_cell(p):
    """Verdict plus the reasons, so a failure says why (e.g. which FTMO limit broke and when)."""
    if not p:
        return "—"
    v = p["verdict"]
    why = "; ".join(v.get("reasons") or [])
    cont = p.get("continuous") or []
    if cont:
        last = cont[-1]
        why += f" | רציף: {last['phase']} {last['status']} {str(last['end'])[:10]}"
    roll = p.get("rolling") or {}
    if roll.get("completed"):
        why += f" | מעבר {roll.get('pass_rate_pct')}% פסילה {roll.get('breach_rate_pct')}% ({roll.get('completed')} דגימות)"
    return PROPHET_HE[v["verdict"]] + f"<div class='mut'>{html.escape(why)}</div>"


def _acc_text(w2):
    w, blown, _ = w2
    if blown is not None:
        return f"נשרף {pd.Timestamp(blown):%m/%Y}"
    return f"${w['balance'].iloc[-1]:,.0f}" if len(w) else "—"


def _risk_cell(r):
    t = (r.get("risk_tests") or {}).get(LOW_RISK)
    if not t:
        return "—"
    pr = "דולג (יותר מדי עסקאות)" if t["skipped"] else _prophet_cell(t["prophet"])
    return f"{_acc_text(t['account'])}<br>{pr}"


def _sweep_html(results):
    out = ""
    for r in results:
        if not r["sid"].startswith("TV") or not r.get("risk_tests"):
            continue
        rows = "".join(f"<tr><td>{rk * 100:g}%</td><td>{_acc_text(t['account'])}</td><td>{_prophet_cell(t['prophet'])}</td></tr>"
                       for rk, t in sorted(r["risk_tests"].items()))
        b = r["before"]
        oos_ok = b["trades"] >= 30 and b["avg_r"] == b["avg_r"] and b["avg_r"] > 0 and (b.get("t") or 0) > 1
        oos = (f"<p class='mut'>בדיקה מחוץ למדגם (לפני {SPLIT_HE}): {b['trades']:,} עסקאות, ממוצע "
               f"{b['avg_r']:+.2f}R, t={b.get('t', float('nan')):.1f} · "
               f"{'✅ רווחית גם בשנים שלא נבחרה עליהן' if oos_ok else '❌ לא רווחית באופן מובהק לפני כן'}</p>")
        out += (f"<h3>{r['sid']} · {RP.TF_HE[r['tf']]} · Trading Prophet לפי רמת סיכון</h3>{oos}<div class='tw'><table>"
                f"<tr><th>סיכון לעסקה</th><th>חשבון $100K</th><th>Trading Prophet</th></tr>{rows}</table></div>")
    return out


PROPHET_HE = {"PASS": "✅ עובר", "FAIL": "❌ נכשל", "PROVISIONAL": "🟡 בתנאי", "INSUFFICIENT": "⚪ חסר מידע"}


def export_trades(sid, tf, led):
    """Trade-by-trade table for one strategy/timeframe in the $100,000 account (1% risk)."""
    if led is None or led.empty:
        return
    ny = "Asia/Jerusalem"
    out = pd.DataFrame({
        "אסטרטגיה": f"{sid} {R.STRATEGIES[sid][0]}", "גרף": RP.TF_HE[tf], "נכס": led["symbol"],
        "כיוון": led["dir"].map({1: "קנייה", -1: "מכירה"}),
        "כניסה (שעון ישראל)": pd.to_datetime(led["time"]).dt.tz_convert(ny).dt.strftime("%d.%m.%Y %H:%M"),
        "מחיר כניסה": led["entry_px"].round(5), "סטופ": led["stop_px"].round(5), "יעד": led["target_px"].round(5),
        "יציאה (שעון ישראל)": pd.to_datetime(led["exit_time"]).dt.tz_convert(ny).dt.strftime("%d.%m.%Y %H:%M"),
        "מחיר יציאה": led["exit_px"].round(5), "סיבת יציאה": led["reason"], "תוצאה ב־R": led["r"].round(2),
        "בחשבון": led.get("taken"), "רווח/הפסד $": led.get("pnl_usd", pd.Series(dtype=float)).round(0),
        "יתרה אחרי": led.get("balance_after", pd.Series(dtype=float)).round(0)})
    path = config.OUTPUT_DIR / f"trades_{sid}_{tf}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  exported {len(out)} trades -> {path.name}")


def _data_html():
    sp = []
    for s, v in sorted(SPREADS.items()):
        if v is not None and len(v):
            sp.append(f"{html.escape(s)}: {float(v.median()):.6g}")
    return ("<h2>נתונים ועלויות</h2><p class='mut'>עלויות לפי FTMO (ftmo.com/en/symbols): עמלה וסוויפ לכל נכס, "
            "וספרד היסטורי של הדקה שבה נפתחה העסקה (Dukascopy, מחיר קנייה פחות מחיר מכירה).</p>"
            f"<p>ספרד חציוני (ביחידות מחיר): {', '.join(sp) or '—'}</p>"
            f"<p>ספרד משוער (אין היסטוריית קנייה/מכירה): {', '.join(sorted(SPREAD_EST)) or 'אין'}</p>"
            f"<p>נכסים בלי נתונים: {', '.join(sorted(MISSING)) or 'אין'}</p>"
            f"<p>חורים בנתונים (עסקאות שהוחזקו מעליהם הוסרו): "
            f"{html.escape('; '.join(f'{k}: ' + ', '.join(v) for k, v in sorted(HOLES.items()))) or 'אין'}</p>")


def report(results, args):
    f2 = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.2f}"
    rows = ""
    for r in results:
        rows += (f"<tr><td>{r['sid']}</td><td>{html.escape(r['name'])}</td><td>{RP.TF_HE[r['tf']]}</td>"
                 f"<td>{r['after']['trades']:,}</td><td>{f2(r['after']['avg_r'])}</td><td>{RP._wk(r)}</td>"
                 f"<td>{'✅' if r['ok'] else '❌'}</td><td>{'—' if r['ftmo_best'] is None else RP._pct(r['ftmo_best']['p_both'])}</td>"
                 f"<td>{_prophet_cell(r.get('prophet'))}</td><td>{_risk_cell(r)}</td></tr>")
    body = ""
    for r in results:
        sym = "".join(f"<tr><td>{s}</td><td>{q['trades']:,}</td><td>{f2(q['avg_r'])}</td><td>{f2(q['t'])}</td></tr>"
                      for s, q in r["by_sym"].items())
        b, a = r["before"], r["after"]
        body += (f"<h3>{r['sid']} · {html.escape(r['name'])} · {RP.TF_HE[r['tf']]}</h3>"
                 f"<p class='mut'>לפני {SPLIT_HE}: {b['trades']:,} עסקאות, ממוצע {f2(b['avg_r'])}R · "
                 f"השנה האחרונה: {a['trades']:,} עסקאות, ממוצע {f2(a['avg_r'])}R, הצלחה "
                 f"{'—' if a['win'] != a['win'] else f'{a['win'] * 100:.0f}%'}</p>"
                 + RP._weekly_html(r) +
                 "<details><summary>לפי נכס (השנה האחרונה)</summary><div class='tw'><table><tr><th>נכס</th><th>עסקאות</th>"
                 "<th>ממוצע R</th><th>t</th></tr>" + sym + "</table></div></details>")
    doc = "".join(f"<li><b>{sid}</b> {html.escape(R.STRATEGIES[sid][0])}: {html.escape((R.STRATEGIES[sid][1].__doc__ or '').strip())}</li>"
                  for sid in R.STRATEGIES)
    page = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TradingView Strategies</title>
<style>:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e3de}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--card:#1e1e1c;--line:#33332f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}main{{max-width:860px;margin:auto;padding:20px 16px 40px}}
.mut{{color:var(--mut);font-size:13px}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:start;white-space:nowrap}}h3{{margin-top:28px}}li{{margin:6px 0}}</style></head><body><main>
<h1>בקטסט · אסטרטגיות מ־TradingView</h1>
<p class='mut'>{VARIANT[0]}</p>
<p class="mut">{'דמו' if args.demo else 'Dukascopy'}: 5 צמדי מט״ח, זהב, כסף ונפט · גרפים: דקה, 5, 15 דקות ושעה · הפרמטרים כפי שפורסמו, בלי חיפוש ובלי התאמה ·
בלי עסקאות 5 דקות לפני ואחרי הודעות כלכליות חשובות · חשבון FTMO של $100,000 לכל אסטרטגיה מ־{SPLIT_HE}, נדגם כל שבוע, נעצר כשנשרף.</p>
{_per_symbol_html()}
<h2>שורה תחתונה</h2>
<div class='tw'><table><tr><th>#</th><th>אסטרטגיה</th><th>גרף</th><th>עסקאות בשנה האחרונה</th><th>ממוצע R</th>
<th>חשבון $100K מ־{SPLIT_HE} (סיכון 1%)</th><th>שרד ורווחי</th><th>עובר אתגר 2-Step</th><th>Trading Prophet (הפסד צף דקה־דקה)</th><th>סיכון 0.75%: חשבון + Prophet</th></tr>{rows}</table></div>
{_sweep_html(results)}
<h2>הכללים שנבדקו</h2><ul>{doc}</ul>
{body}
{_data_html()}
{"" if not ERRORS else "<h2>שגיאות בהרצה</h2><pre style='white-space:pre-wrap;font-size:12px'>" + html.escape(chr(10).join(ERRORS)) + "</pre>"}
<p class="mut">פרמטרים שהמחבר לא פרסם הושלמו בהנחות מקובלות, כמפורט בתיאור של כל אסטרטגיה. זה כלי מחקר, לא ייעוץ השקעות.</p>
</main></body></html>"""
    path = config.OUTPUT_DIR / "report_rules.html"
    path.write_text(page, encoding="utf-8")
    print(f"\n=== RESULT ===\n{path}")


if __name__ == "__main__":
    import sys
    from run_backtest import _explain
    try:
        main()
    except config.TimeBudgetReached:
        print("::notice::Time budget reached; run again to continue.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"::error::{_explain(e)}")
        raise
