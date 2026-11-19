"""Index futures' own intraday patterns (bot 6's smart agent, 17.11.2026 night; DISCOVER, the user's "repeating patterns
in the futures' intraday behaviour - session open, overnight gaps").

The index futures (MNQ / MES / MYM) trade almost 24 hours, but the cash session (09:30-16:00 New York) still sets the
day: the 09:30 open against the 16:00 close of the day before is the "overnight gap" every index trader watches. The
library's gap-fill strategy (SS08) measures the gap at the first bar of the UTC day and a bar's ATR - for a 23-hour
market that is not the cash gap. Here the gap is measured as the market sees it:

  gap   = the 09:30 bar's open - the last cash-session close (the bar that ends at 16:00 New York, the day before)
  size  = |gap| / the mean cash-session range (09:30-16:00 high - low) of the 10 days before

  FG01  small / medium gap that the first bar rejects (0.1-0.5 of the range; a gap up whose first bar closes under its
        open, a gap down whose first bar closes over it) -> trade back toward yesterday's close: target = that close,
        stop beyond the first bar's high / low, flat at 16:00.
  FG02  large gap that the first bar confirms (over 0.5 of the range, the first bar closes in the gap's direction)
        -> trade with it: stop beyond the first bar's other side, target 2R, flat at 16:00.

One trade a day at most, entry at the open of the bar after the first cash bar. Tested on bot 6's world like every
scout strategy (agents/scout_fut.py: 96 variants, chosen on 2024-2025, shown on 2026).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents.scout_strategies import _local


def _cash_gap(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    hs, he, _, _, day = _local(df, "America/New_York")
    rth = (hs >= 9.5) & (he <= 16.0)
    dk = pd.Series(day)
    close_d = pd.Series(np.where(rth & (he >= 15.99), c, np.nan)).groupby(dk).last()     # the 16:00 close of each day
    hi_d = pd.Series(np.where(rth, h, np.nan)).groupby(dk).max()
    lo_d = pd.Series(np.where(rth, l, np.nan)).groupby(dk).min()
    rng10 = (hi_d - lo_d).rolling(10, min_periods=5).mean().shift(1)                   # the 10 days before
    prev_close = close_d.ffill().shift(1)                                               # the last cash close before
    pc = dk.map(prev_close).to_numpy(float)
    rg = dk.map(rng10).to_numpy(float)
    first = rth & (np.abs(hs - 9.5) < 1e-9)
    gap = o - pc
    with np.errstate(invalid="ignore", divide="ignore"):
        size = np.abs(gap) / rg
    ok = first & np.isfinite(size) & (rg > 0)
    return o, h, l, c, he, pc, gap, size, ok


def s_cash_gap_fade(df):
    o, h, l, c, he, pc, gap, size, ok = _cash_gap(df)
    sel = ok & (size >= 0.1) & (size <= 0.5)
    short = sel & (gap > 0) & (c < o)
    long_ = sel & (gap < 0) & (c > o)
    return long_, short, {"stop": ("level", l, h), "target": ("level", pc, pc), "force_exit": he >= 16.0}


def s_cash_gap_go(df):
    o, h, l, c, he, pc, gap, size, ok = _cash_gap(df)
    sel = ok & (size > 0.5)
    long_ = sel & (gap > 0) & (c > o)
    short = sel & (gap < 0) & (c < o)
    return long_, short, {"stop": ("level", l, h), "target": ("R", 2.0), "force_exit": he >= 16.0}


# THE OVERNIGHT RANGE (bot 6's smart agent, 18.11.2026 night, the same DISCOVER): the futures' own night - the Globex
# session from the 18:00 reopen to 09:30 New York - builds a high and a low the cash open then tests. Known before 09:30.
#   FG03  the first cash-hour bar (09:30-11:30) that goes beyond the overnight high (low) and closes back inside ->
#         against it (a failed break / a sweep of the night's liquidity): stop beyond that bar, target the range's middle
#   FG04  the first cash-hour bar (09:30-11:30) that closes beyond the overnight high (low) -> with it (the night's
#         range broken): stop beyond the bar's other side, 2R
# One trade a day at most (the first event of the day), flat at 16:00.
def _overnight(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    hs, he, _, _, day = _local(df, "America/New_York")
    d = pd.to_datetime(pd.Series(day))
    tday = (d + pd.to_timedelta((hs >= 18.0).astype(int), unit="D")).dt.date      # 18:00 on = the next trading day
    night = (hs >= 18.0) | (he <= 9.5)
    on_h = pd.Series(np.where(night, h, np.nan)).groupby(tday.to_numpy()).transform("max").to_numpy(float)
    on_l = pd.Series(np.where(night, l, np.nan)).groupby(tday.to_numpy()).transform("min").to_numpy(float)
    win = (hs >= 9.5) & (he <= 11.5)
    return o, h, l, c, he, tday.to_numpy(), on_h, on_l, win & np.isfinite(on_h) & np.isfinite(on_l)


def _first_per_day(mask, tday):
    m = pd.Series(np.asarray(mask, bool))
    return m.to_numpy(bool) & (m.astype(int).groupby(tday).cumsum().to_numpy() == 1)


def s_on_sweep(df):
    o, h, l, c, he, tday, on_h, on_l, ok = _overnight(df)
    sh_ = ok & (h > on_h) & (c < on_h)
    lo_ = ok & (l < on_l) & (c > on_l)
    ev = _first_per_day(sh_ | lo_, tday)
    mid = (on_h + on_l) / 2.0
    return lo_ & ev & ~sh_, sh_ & ev & ~lo_, {"stop": ("level", l, h), "target": ("level", mid, mid), "force_exit": he >= 16.0}


def s_on_break(df):
    o, h, l, c, he, tday, on_h, on_l, ok = _overnight(df)
    up, dn = ok & (c > on_h), ok & (c < on_l)
    ev = _first_per_day(up | dn, tday)
    return up & ev, dn & ev, {"stop": ("level", l, h), "target": ("R", 2.0), "force_exit": he >= 16.0}


# INTRADAY MOMENTUM AND THE INITIAL BALANCE (bot 6's smart agent, 19.11.2026 night, the same DISCOVER):
#   FG05  the first half hour sets the last one (Gao, Han, Li, Zhou 2018, "Market intraday momentum", JFE - on SPY and
#         the index futures): the move from the last 16:00 close to 10:00 New York, when over 0.1 of the 10-day cash
#         range, is followed at 15:30 in its direction: entry at the 15:30 open (before FTMO Futures' 15:45 no-entry
#         rule), stop 1 ATR, 2R, flat 16:00 (the session rule)
#   FG06  the initial balance (the 09:30-10:30 range, the index futures traders' "IB"): the first 10:30-14:00 bar that
#         closes beyond it -> with it, stop at the IB's middle, target one IB range beyond the broken side, flat 16:00
# One trade a day at most.
def _day_level(mask, vals, day, how):
    s = pd.Series(np.where(mask, vals, np.nan))
    return s.groupby(pd.Series(day)).transform(how).to_numpy(float)


def s_intraday_mom(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    hs, he, _, _, day = _local(df, "America/New_York")
    rth = (hs >= 9.5) & (he <= 16.0)
    dk = pd.Series(day)
    close_d = pd.Series(np.where(rth & (he >= 15.99), c, np.nan)).groupby(dk).last()
    hi_d = pd.Series(np.where(rth, h, np.nan)).groupby(dk).max()
    lo_d = pd.Series(np.where(rth, l, np.nan)).groupby(dk).min()
    rg = dk.map((hi_d - lo_d).rolling(10, min_periods=5).mean().shift(1)).to_numpy(float)
    pc = dk.map(close_d.ffill().shift(1)).to_numpy(float)
    c10 = _day_level(rth & (np.abs(he - 10.0) < 1e-9), c, day, "max")
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = (c10 - pc) / rg
    sig = (np.abs(he - 15.5) < 1e-9) & np.isfinite(m1) & (rg > 0) & (np.abs(m1) >= 0.1)
    return sig & (m1 > 0), sig & (m1 < 0), {"stop": ("atr", 1.0), "target": ("R", 2.0), "force_exit": he >= 16.0}


def s_ib_break(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    hs, he, _, _, day = _local(df, "America/New_York")
    ib = (hs >= 9.5) & (he <= 10.5)
    ib_h = _day_level(ib, h, day, "max")
    ib_l = _day_level(ib, l, day, "min")
    ok = (hs >= 10.5) & (he <= 14.0) & np.isfinite(ib_h) & np.isfinite(ib_l) & (ib_h > ib_l)
    up, dn = ok & (c > ib_h), ok & (c < ib_l)
    ev = _first_per_day(up | dn, day)
    mid, w = (ib_h + ib_l) / 2.0, ib_h - ib_l
    return up & ev, dn & ev & ~up, {"stop": ("level", mid, mid), "target": ("level", ib_h + w, ib_l - w),
                                    "force_exit": he >= 16.0}


FUT_PATTERNS = {
    "FG05": ("Intraday momentum (Gao et al. 2018): the move from the last 16:00 close to 10:00 NY over 0.1 of the 10-day "
             "range -> with it at the 15:30 open, stop 1 ATR, 2R, flat 16:00", s_intraday_mom, ["1Min", "5Min", "15Min"]),
    "FG06": ("Initial balance break: the first 10:30-14:00 NY bar that closes beyond the 09:30-10:30 range -> with it, "
             "stop at the range's middle, target one range beyond, flat 16:00", s_ib_break, ["1Min", "5Min", "15Min"]),
    "FG03": ("Overnight range sweep: the first 09:30-11:30 bar beyond the Globex 18:00-09:30 high / low that closes back "
             "inside -> against it, stop beyond the bar, target the range's middle, flat 16:00", s_on_sweep, ["1Min", "5Min", "15Min"]),
    "FG04": ("Overnight range break: the first 09:30-11:30 bar that closes beyond the Globex 18:00-09:30 high / low -> with "
             "it, stop beyond the bar's other side, 2R, flat 16:00", s_on_break, ["1Min", "5Min", "15Min"]),
    "FG01": ("Cash-session gap fade: 09:30 open vs the last 16:00 close, 0.1-0.5 of the 10-day range, the first bar "
             "rejects -> to yesterday's close, stop beyond the first bar, flat 16:00", s_cash_gap_fade, ["1Min", "5Min", "15Min"]),
    "FG02": ("Cash-session gap and go: a gap over 0.5 of the 10-day range that the first bar confirms, stop beyond the "
             "first bar, 2R, flat 16:00", s_cash_gap_go, ["1Min", "5Min", "15Min"]),
}
