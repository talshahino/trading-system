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


# THE SESSION WINDOW AND THE BREAKEVEN POINT (bot 6's smart agent, 16.11.2026 night). Two optional keys of a variant,
# absent = as before (every existing variant id and every live entry in live/bots/bot6.json stays the same):
#   "sess"  entries only inside a New York window - the futures trade 23 hours, and a scout strategy built on the cash
#           session may lose in the thin evening / Asian hours (16.11 00:55 UTC = 19:55 New York, an hour after the
#           Sunday open: SC072|5Min|US500 stopped in 5 minutes, -$389, the day's only trade):
#           day = 03:00-15:45 (London + New York), ny = 08:00-15:45, rth = 09:30-15:45, noopen = not 18:00-19:59
#   "be"    True / False as before, or a number - the breakeven's own point (the user's principle 7, timing per strategy;
#           task 17, Fibonacci levels): below 1 = the share of the way to the target, 1 and above = R (no target; +1R
#           locked half an R later)
SESS = {"day": (3 * 60, 15 * 60 + 45), "ny": (8 * 60, 15 * 60 + 45), "rth": (9 * 60 + 30, 15 * 60 + 45),
        "noopen": (20 * 60, 18 * 60)}
SESS_HE = {"day": "כניסה רק 03:00-15:45 ניו יורק (לונדון וניו יורק)", "ny": "כניסה רק 08:00-15:45 ניו יורק",
           "rth": "כניסה רק בשעות המסחר הרגילות 09:30-15:45", "noopen": "בלי כניסה בשעתיים הראשונות אחרי 18:00"}
# "filt" (bot 6's smart agent, 17.11.2026 night - the shared methods of bots 5 and 6): bot 5's two scout filters, the
#           same rule as live/watch_own.scout5_filter (bot 5's live bot) - ema50h = longs only when the last CLOSED
#           1-hour close is over the 1-hour EMA50, shorts only under it (the user, 16.11.2026 19:49: "a long against a
#           falling market is a wrong read"); noreent = after a stopped trade, no new entry in the same direction that
#           New York day. The fix family "direction" for a scout strategy that lost (task 2072d266da96: SC072|5Min|US500
#           stopped at once on 16.11, MFE 0.14R). Absent = as before (every variant id stays the same).
FILT = ("ema50h", "noreent")
FILT_HE = {"ema50h": "רק בכיוון של EMA50 בגרף השעה (קניות מעליו, מכירות מתחתיו)",
           "noreent": "אחרי סטופ - בלי כניסה נוספת באותו כיוון באותו יום"}
BE_WAY = (0.382, 0.5, 0.618, 0.786, 0.886)      # with a target: the share of the way (Fibonacci, task 17)
BE_R = (1.5, 2.0, 3.0)                          # without one: R (2.5 = True, the rule of principle 7)


def _be_num(v):
    b = v.get("be")
    return None if isinstance(b, bool) or b is None else float(b)


def vid(v):
    t = v["target"] if isinstance(v["target"], str) else f"{float(v['target']):g}R"
    b = _be_num(v)
    be = f"be{b:g}" if b is not None else ("be" if v["be"] else "nobe")
    return (f"t{t}_s{float(v['stop']):g}_{'time' if v['time'] == 'own' else 'notime'}_{be}"
            + (f"_{v['sess']}" if v.get("sess") else "") + (f"_{v['filt']}" if v.get("filt") else ""))


def sess_ok(times, sess):
    """True where an entry at `times` is inside the New York window `sess` (None = always)."""
    if not sess:
        return np.ones(len(times), bool)
    a, b = SESS[sess]
    et = pd.DatetimeIndex(times).tz_convert("America/New_York")
    m = np.asarray(et.hour * 60 + et.minute)
    return (m >= a) & (m < b) if a < b else (m >= a) | (m < b)


def variants():
    return [{"target": t, "stop": s, "time": m, "be": b} for t in TARGETS for s in STOPS for m in TIMES for b in BES]


def he(v):
    """The variant in a few Hebrew words."""
    t = "יעד כמו במקור" if v["target"] == "own" else "בלי יעד" if v["target"] == "none" else f"יעד {float(v['target']):g}R"
    s = "" if float(v["stop"]) == 1.0 else f", סטופ x{float(v['stop']):g}"
    m = "" if v["time"] == "own" else ", בלי יציאה בזמן"
    bn = _be_num(v)
    b = (f", איזון ב־{bn:g}R" if bn >= 1 else f", איזון ב־{bn * 100:g}% מהדרך") if bn is not None else ("" if v["be"] else ", בלי איזון")
    return (t + s + m + b + (f", {SESS_HE[v['sess']]}" if v.get("sess") else "")
            + (f", {FILT_HE[v['filt']]}" if v.get("filt") in FILT_HE else ""))


def dir_filter(t, df1, way):
    """bot 5's scout filter (live/watch_own.scout5_filter, the same rule) on a strategy's trades (time order). df1: the
    1-minute bars. Nothing to filter / an unknown way = the trades as they are."""
    if way not in FILT or t.empty:
        return t
    utc = lambda v: (pd.Timestamp(v).tz_localize("UTC") if pd.Timestamp(v).tzinfo is None  # noqa: E731
                     else pd.Timestamp(v).tz_convert("UTC"))
    keep = []
    if way == "noreent":
        stopped = {}
        for i, x in enumerate(t.to_dict("records")):
            tm = utc(x["time"])
            key = (tm.tz_convert("America/New_York").date(), int(x["dir"]))
            if key in stopped and stopped[key] <= tm:
                continue
            keep.append(i)
            if x["reason"] != "סוף הנתונים" and pd.notna(x["r"]) and float(x["r"]) < 0 and pd.notna(x["exit_time"]):
                ex = utc(x["exit_time"])
                stopped[key] = min(stopped.get(key, ex), ex)
        return t.iloc[keep]
    c = df1["c"]
    if c.index.tz is None:
        c = c.tz_localize("UTC")
    h = c.resample("1h", label="left", closed="left").last().dropna()
    e50 = h.ewm(span=50, adjust=False).mean()
    closed = h.index + pd.Timedelta(hours=1)
    for i, x in enumerate(t.to_dict("records")):
        j = int(closed.searchsorted(utc(x["time"]), side="right")) - 1
        if j < 50 or bool(h.iloc[j] > e50.iloc[j]) == (int(x["dir"]) > 0):
            keep.append(i)
    return t.iloc[keep]


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


def strategy(sid, v, noent, tf=None):
    """The scout's function as bot 6 runs it: the variant's target / stop / time exit, signals in no-entry minutes off,
    and (with "sess") signals whose entry - the next bar's open - is outside the session window off."""
    from agents.rules_agent import STRATEGIES
    fn0 = STRATEGIES[sid][1]

    def g(df):
        lo, sh, plan = fn0(df)
        plan = dict(plan)
        lo = np.asarray(lo, bool) & ~noent
        sh = np.asarray(sh, bool) & ~noent
        if v.get("sess"):
            ok = sess_ok(df.index + pd.Timedelta(minutes=TF_MIN.get(tf, 0)), v["sess"])
            lo, sh = lo & ok, sh & ok
        if v["target"] == "none":
            plan["target"] = None
        elif v["target"] != "own":
            plan["target"] = ("R", float(v["target"]))
        if float(v["stop"]) != 1.0:
            plan["stop_scale"] = float(v["stop"])
        if v["time"] == "none":
            plan.pop("max_bars", None)
        if v["be"] and plan.get("target") is None:          # no target: entry at 2.5R, +1R locked at 3R (principle 7)
            bn = _be_num(v)
            r0 = bn if bn is not None and bn >= 1 else 2.5
            plan["be_r"], plan["lock_r"] = r0, (r0 + 0.5, 1.0)
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
    bn = _be_num(v)
    be = ((bn if bn is not None and bn < 1 else 0.6), cost_px / px) if v["be"] else None
    t = run(dft, strategy(sid, v, noent, tf), 0.0, breakeven=be, session_exit=force)
    if t.empty:
        return t
    risk = (t["entry_px"] - t["stop_px"]).abs()
    t = t[risk > 0].copy()
    t["cost_r"] = cost_px / (t["entry_px"] - t["stop_px"]).abs()
    t["r"] = t["r"] - t["cost_r"]
    if v.get("filt"):
        t = dir_filter(t.reset_index(drop=True), df1, v["filt"])
    return t
