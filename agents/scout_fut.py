"""Scout strategies on bot 6's own world (bot 6's smart agent, 13.11.2026).

The scout's strategies reach bot 6 through agent_portfolio.py, which tests them with the CFD bots' machine (FTMO CFD
costs, FTMO Standard rules, chosen on 30.04-31.12.2025). Five of them were approved for bot 6 on 12.11.2026 on that
test, and bot 6 trades none of them: its live loop (live/watch_own.py fut_step) ran only the user's gap method. Here:
  test   each approved strategy on bot 6's world - the index CFDs' 1-minute history sized as micro futures (a tick of
         spread + the commission + a tick of slippage per contract, live/fut_account.py), FTMO Futures sessions (no
         entry from 15:45 New York or into the close, everything out at 16:05, nothing over the weekend), no entry
         +-2 minutes around a high-impact release, and the user's breakeven rule (principle 7: with a target the stop
         to the entry at 60% of the way; without one, to the entry at 2.5R and +1R locked at 3R).
         Variants (the fix attempts the manager asked for, 13.11.2026): target as the source / 1 / 1.5 / 2 / 3R / none,
         stop x0.75 / 1 / 1.25 / 1.5, the source's time exit or none (the user rejects time exits), breakeven on / off.
         Chosen on 2024-01-01 .. 2025-12-31 only, shown on 2026 (never seen while choosing), time-weighted.
  live   the same function the test runs, on the live 1-minute bars of the last 14 days (live/watch_own.py fut_step),
         only for the strategies live/bots/bot6.json "scout" lists - written by agents/builder.py from an approved card
         of this test (never from the CFD test). Nothing in "scout" = nothing changes in the bot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TF_MIN = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60}
TARGETS = ("own", "none", 1.0, 1.5, 2.0, 3.0)
STOPS = (0.75, 1.0, 1.25, 1.5)
TIMES = ("own", "none")
BES = (True, False)
BASE = {"target": "own", "stop": 1.0, "time": "own", "be": True}


def vid(v):
    t = v["target"] if isinstance(v["target"], str) else f"{float(v['target']):g}R"
    return f"t{t}_s{float(v['stop']):g}_{'time' if v['time'] == 'own' else 'notime'}_{'be' if v['be'] else 'nobe'}"


def variants():
    return [{"target": t, "stop": s, "time": m, "be": b} for t in TARGETS for s in STOPS for m in TIMES for b in BES]


def he(v):
    """The variant in a few Hebrew words."""
    t = "יעד כמו במקור" if v["target"] == "own" else "בלי יעד" if v["target"] == "none" else f"יעד {float(v['target']):g}R"
    s = "" if float(v["stop"]) == 1.0 else f", סטופ x{float(v['stop']):g}"
    m = "" if v["time"] == "own" else ", בלי יציאה בזמן"
    b = "" if v["be"] else ", בלי איזון"
    return t + s + m + b


def resample(df1, tf):
    if tf == "1Min":
        return df1[["o", "h", "l", "c"]].copy()
    rule = f"{TF_MIN[tf]}min" if TF_MIN[tf] < 60 else "1h"
    agg = df1.resample(rule, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return agg


def masks(dft, tf, noent1, shut1, idx1):
    """On the strategy's bars: no signal whose entry (the next bar's open) is in a no-entry minute; a forced exit at
    the close of a bar during which the futures close (16:05 New York, the weekend)."""
    n = len(dft)
    s1 = pd.Series(np.asarray(noent1, bool), index=idx1)
    sh = pd.Series(np.asarray(shut1, bool), index=idx1)
    starts = dft.index
    nxt = starts[1:].append(starts[-1:] + pd.Timedelta(minutes=TF_MIN[tf]))
    # the entry minute = the next bar's first minute (a missing minute counts as no entry)
    ne = s1.reindex(nxt).fillna(True).to_numpy(bool)
    # a big hole before the next bar (a session break) = no entry either
    gap = np.asarray((nxt - starts) > pd.Timedelta(minutes=TF_MIN[tf] + 30))
    noent = ne | gap
    # any shut minute inside the bar, or the bar's end in a shut minute
    shut_in = sh.resample(f"{TF_MIN[tf]}min" if TF_MIN[tf] < 60 else "1h", label="left", closed="left").max()
    force = shut_in.reindex(starts).fillna(False).to_numpy(bool) | gap
    return noent[:n], force[:n]


def strategy(sid, v, noent):
    """The scout's function as bot 6 runs it: the variant's target / stop / time exit, signals in no-entry minutes off."""
    from agents.rules_agent import STRATEGIES
    fn0 = STRATEGIES[sid][1]

    def g(df):
        lo, sh, plan = fn0(df)
        plan = dict(plan)
        lo = np.asarray(lo, bool) & ~noent
        sh = np.asarray(sh, bool) & ~noent
        if v["target"] == "none":
            plan["target"] = None
        elif v["target"] != "own":
            plan["target"] = ("R", float(v["target"]))
        if float(v["stop"]) != 1.0:
            plan["stop_scale"] = float(v["stop"])
        if v["time"] == "none":
            plan.pop("max_bars", None)
        if v["be"] and plan.get("target") is None:          # no target: entry at 2.5R, +1R locked at 3R (principle 7)
            plan["be_r"], plan["lock_r"] = 2.5, (3.0, 1.0)
        return lo, sh, plan
    return g


def trades(df1, sym, sid, tf, v, noent1, shut1, cost_px):
    """Trades of one strategy on one futures contract (the CFD's prices), r net of the futures' costs."""
    from agents.rules_agent import run
    dft = resample(df1, tf)
    if len(dft) < 300:
        return pd.DataFrame()
    dft.attrs["symbol"] = sym
    noent, force = masks(dft, tf, noent1, shut1, df1.index)
    px = float(dft["c"].median())
    be = (0.6, cost_px / px) if v["be"] else None
    t = run(dft, strategy(sid, v, noent), 0.0, breakeven=be, session_exit=force)
    if t.empty:
        return t
    risk = (t["entry_px"] - t["stop_px"]).abs()
    t = t[risk > 0].copy()
    t["cost_r"] = cost_px / (t["entry_px"] - t["stop_px"]).abs()
    t["r"] = t["r"] - t["cost_r"]
    return t
