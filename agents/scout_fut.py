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

import json
from pathlib import Path

import numpy as np
import pandas as pd

from live import fut_account as FA

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
        "noopen": (20 * 60, 18 * 60),
        # no10 (bot 6's smart agent, 18.11.2026 night): not 10:00-10:59 New York - the hour of the shadow cluster of
        # 16-17.11 (9 scout longs stopped 09:45-10:15 on two falling days, all three indices at once); on the scouts'
        # own trades the hour holds 35% of them at about +0.03R each (2024-2025 and 2026 alike), SC258|1Min|US100
        # -0.20R on 181 trades there to 2025. One more candidate of the hours family, the same fair test.
        "no10": (11 * 60, 10 * 60)}
SESS_HE = {"day": "כניסה רק 03:00-15:45 ניו יורק (לונדון וניו יורק)", "ny": "כניסה רק 08:00-15:45 ניו יורק",
           "rth": "כניסה רק בשעות המסחר הרגילות 09:30-15:45", "noopen": "בלי כניסה בשעתיים הראשונות אחרי 18:00",
           "no10": "בלי כניסה 10:00-10:59 ניו יורק (השעה שאחרי פתיחת המסחר הרגיל)"}
# "filt" (bot 6's smart agent, 17.11.2026 night - the shared methods of bots 5 and 6): bot 5's two scout filters, the
#           same rule as live/watch_own.scout5_filter (bot 5's live bot) - ema50h = longs only when the last CLOSED
#           1-hour close is over the 1-hour EMA50, shorts only under it (the user, 16.11.2026 19:49: "a long against a
#           falling market is a wrong read"); noreent = after a stopped trade, no new entry in the same direction that
#           New York day. The fix family "direction" for a scout strategy that lost (task 2072d266da96: SC072|5Min|US500
#           stopped at once on 16.11, MFE 0.14R). Absent = as before (every variant id stays the same).
FILT = ("ema50h", "noreent")
FILT_HE = {"ema50h": "רק בכיוון של EMA50 בגרף השעה (קניות מעליו, מכירות מתחתיו)",
           "noreent": "אחרי סטופ - בלי כניסה נוספת באותו כיוון באותו יום"}
# EVERY FIX FAMILY FOR A LOSING SCOUT STRATEGY (the user, 17.11.2026 17:26, task tal1711lossall: "you only manage the
# trade instead of also looking for different entry points and exit points - a different stop or target. I want all of
# it"). Optional keys of a variant, absent = as before (every variant id and every live entry stays the same), each the
# rules_agent.run mechanism the CFD bots' tests already use, so the live bot (live/watch_own.fut_scout -> trades) runs
# exactly what was tested:
#   "ent"   the entry: a limit order k x the risk better than the signal's open, valid 3 bars (pb25 / pb50), else no trade
#   "ssw"   the stop: beyond the 2nd / 3rd swing (3-bar pivots) on the stop's side + 0.1 ATR - the user's structural
#           stop (his gap method: beyond the 2nd swing; 16.11.2026: the 3rd when the other side is stronger)
#   "part"  the management: half out at half the way to the target (no target: at +1R), the rest's stop to the entry
#           (h50 - the user's rule for the fast charts, 13.11.2026)
#   "pyr"   the pyramid: half a position more at +1R, the whole stop moved so a stop still loses 1R (py1, tal1411anti)
ENT = {"pb25": (0.25, 3), "pb50": (0.5, 3)}
SSW = {"sw2": (2, 3, 0.1), "sw3": (3, 3, 0.1)}
PART = {"h50": (0.5, 0.5, 1.0)}                 # (share out, way with a target, R without one)
PYR = {"py1": (1.0, 0.5)}
FAM_HE = {"ent": {"pb25": "כניסה בלימיט 0.25R טוב יותר (עד 3 נרות)", "pb50": "כניסה בלימיט 0.5R טוב יותר (עד 3 נרות)"},
          "ssw": {"sw2": "סטופ מעבר לסווינג השני", "sw3": "סטופ מעבר לסווינג השלישי"},
          "part": {"h50": "חצי החוצה בחצי הדרך (בלי יעד: ב־+1R) וסטופ לכניסה"},
          "pyr": {"py1": "חצי פוזיציה נוספת ב־+1R (הסיכון נשאר 1R)"}}
FAMS = (("ent", ENT), ("ssw", SSW), ("part", PART), ("pyr", PYR))
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
            + (f"_{v['sess']}" if v.get("sess") else "") + (f"_{v['filt']}" if v.get("filt") else "")
            # "w": the pyramid on whole contracts (18.11.2026) - results kept from before it are not reused
            + "".join(f"_{v[k]}" + ("w" if k == "pyr" else "") for k, vals in FAMS if v.get(k) in vals))


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
            + (f", {FILT_HE[v['filt']]}" if v.get("filt") in FILT_HE else "")
            + "".join(f", {FAM_HE[k][v[k]]}" for k, vals in FAMS if v.get(k) in vals))


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
    # the volume column is kept (bot 6's smart agent, 17.11.2026 night): strategies that read it (SC071's Keltner reversal,
    # the VWAP ones) raised KeyError 'v' on every variant here, so they could never pass on bot 6's world
    cols = ["o", "h", "l", "c"] + (["v"] if "v" in df1 else [])
    if tf == "1Min":
        return df1[cols].copy()
    rule = f"{TF_MIN[tf]}min" if TF_MIN[tf] < 60 else "1h"
    how = {"o": "first", "h": "max", "l": "min", "c": "last"}
    if "v" in df1:
        how["v"] = "sum"
    agg = df1.resample(rule, label="left", closed="left").agg(how).dropna(subset=["o", "h", "l", "c"])
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


BOT6_FILE = Path(__file__).resolve().parents[1] / "live" / "bots" / "bot6.json"
_RISK_CACHE = [None, {}]


def risk_usd_of(combo):
    """The dollars one trade of this scout strategy risks on the account's opening balance (live/bots/bot6.json "risk":
    the strategy's own share, else the scouts' common one, else the file's absence: the bot's base risk)."""
    try:
        mt = BOT6_FILE.stat().st_mtime
        if _RISK_CACHE[0] != mt:
            _RISK_CACHE[0], _RISK_CACHE[1] = mt, json.loads(BOT6_FILE.read_text(encoding="utf-8")).get("risk") or {}
    except (OSError, ValueError):
        _RISK_CACHE[0], _RISK_CACHE[1] = None, {}
    rk = _RISK_CACHE[1]
    return FA.INITIAL * float(rk.get(f"scout:{combo}", rk.get("scout", FA.RISK_TAL)))


def add_gate(sym, risk_usd, frac):
    """The pyramid on whole micro contracts (bot 6's smart agent, 18.11.2026): the add is `frac` of the position, so it
    exists only when the contracts the trade opens with hold it exactly (half of 2, 4, 6...). One contract has no half:
    the trade then runs without the add and without its moved stop - in the test and in the live bot alike."""
    def ok(entry, stop):
        per = FA.per_contract(sym, entry, stop)
        n = int(risk_usd // per) if per > 0 else 0
        add = n * float(frac)
        return n >= 1 and add >= 1 and abs(add - round(add)) < 1e-9
    return ok


def strategy(sid, v, noent, tf=None, gate=None):
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
        if v.get("ent") in ENT:                             # tal1711lossall: the entry family
            plan["entry_pullback"] = ENT[v["ent"]]
        if v.get("ssw") in SSW:                             # the structural stop family
            plan["stop_swing"] = SSW[v["ssw"]]
        if v.get("part") in PART:                           # the partial exit family (on the target as set above)
            fr, way, r_ = PART[v["part"]]
            plan["partial"] = (fr, way if plan.get("target") is not None else r_)
        if v.get("pyr") in PYR:
            plan["pyramid"] = PYR[v["pyr"]]
            if gate is not None:
                plan["pyramid_gate"] = gate
        if v["be"] and plan.get("target") is None:          # no target: entry at 2.5R, +1R locked at 3R (principle 7)
            bn = _be_num(v)
            r0 = bn if bn is not None and bn >= 1 else 2.5
            plan["be_r"], plan["lock_r"] = r0, (r0 + 0.5, 1.0)
        return lo, sh, plan
    return g


def trades(df1, sym, sid, tf, v, noent1, shut1, cost_px, risk_usd=None):
    """Trades of one strategy on one futures contract (the CFD's prices), r net of the futures' costs. With a pyramid
    the add happens only where whole contracts hold it (add_gate; risk_usd: the live bot's, else live/bots/bot6.json's)
    and pays its own costs: "add_cost_r" is the part of cost_r that belongs to the added contracts."""
    from agents.rules_agent import run
    dft = resample(df1, tf)
    if len(dft) < 300:
        return pd.DataFrame()
    dft.attrs["symbol"] = sym
    noent, force = masks(dft, tf, noent1, shut1, df1.index)
    px = float(dft["c"].median())
    bn = _be_num(v)
    be = ((bn if bn is not None and bn < 1 else 0.6), cost_px / px) if v["be"] else None
    gate = None
    if v.get("pyr") in PYR:
        gate = add_gate(sym, risk_usd_of(f"{sid}|{tf}|{sym}") if risk_usd is None else float(risk_usd), PYR[v["pyr"]][1])
    t = run(dft, strategy(sid, v, noent, tf, gate), 0.0, breakeven=be, session_exit=force)
    if t.empty:
        return t
    risk = (t["entry_px"] - t["stop_px"]).abs()
    t = t[risk > 0].copy()
    base_r = cost_px / (t["entry_px"] - t["stop_px"]).abs()
    added = t["pyr_time"].notna().to_numpy(bool) if "pyr_time" in t else np.zeros(len(t), bool)
    t["add_cost_r"] = base_r * added * (PYR[v["pyr"]][1] if v.get("pyr") in PYR else 0.0)
    t["cost_r"] = base_r + t["add_cost_r"]
    t["r"] = t["r"] - t["cost_r"]
    if v.get("filt"):
        t = dir_filter(t.reset_index(drop=True), df1, v["filt"])
    return t
