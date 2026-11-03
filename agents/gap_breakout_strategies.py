"""Price gaps (family 7) and all-time-high breakouts (family 2's untested corner). Same contract as
agents/rules_agent.py.

A "gap" here is a real reopening: the first bar after a pause of more than 2 hours in the data (the
shares' night, an index's daily break, the weekend for currencies/metals/energy), not a calendar day
change. (SS08 already tests the simpler first-bar-of-the-UTC-day gap fill.)
"""
import numpy as np
import pandas as pd

from agents.rules_agent import atr


def _reopen(df, hours=2):
    """True on the first bar after a pause longer than `hours`; and the pause length in hours."""
    t = df.index.to_numpy()
    gap_h = np.r_[0.0, (t[1:] - t[:-1]) / np.timedelta64(1, "h")]
    return gap_h > hours, gap_h


def s_gap_and_go(df):
    """Gap and go: the market reopens more than 1.5 ATR away from the last close and the first bar
    closes further in the gap's direction -> join it at the next bar. Stop at the first bar's other end,
    target 2R, at most 12 bars (shares are closed at the session end anyway)."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    ro, _ = _reopen(df)
    pc = np.r_[np.nan, c[:-1]]
    gap = o - pc
    up = ro & (gap > 1.5 * a) & (c > o)
    dn = ro & (gap < -1.5 * a) & (c < o)
    return up, dn, {"stop": ("level", l, h), "target": ("R", 2.0), "max_bars": 12}


def s_weekend_gap_fill(df):
    """Weekend gap fill (currencies, metals, energy, indices): the Sunday/Monday reopening is more than
    1 ATR away from Friday's close -> trade back toward Friday's close (the target). Stop 1.5 x the gap
    beyond the open, at most 24 bars."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    ro, gap_h = _reopen(df, hours=36)
    pc = np.r_[np.nan, c[:-1]]
    gap = o - pc
    # decide at the first bar's close, only if it has not already filled the gap
    long_ = ro & (gap < -a) & (c < pc)
    short = ro & (gap > a) & (c > pc)
    st_l, st_s = o - 1.5 * np.abs(gap), o + 1.5 * np.abs(gap)
    return long_, short, {"stop": ("level", st_l, st_s), "target": ("level", pc, pc), "max_bars": 24}


def s_failed_gap(df):
    """Failed gap: the market reopens more than 1.5 ATR away but the first bar closes back beyond its
    own midpoint against the gap -> fade it toward the last close. Stop beyond the first bar, target
    the last close, at most 12 bars."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    ro, _ = _reopen(df)
    pc = np.r_[np.nan, c[:-1]]
    gap = o - pc
    mid = (h + l) / 2
    short = ro & (gap > 1.5 * a) & (c < o) & (c < mid)
    long_ = ro & (gap < -1.5 * a) & (c > o) & (c > mid)
    return long_, short, {"stop": ("level", l - 0.1 * a, h + 0.1 * a), "target": ("level", pc, pc), "max_bars": 12}


def s_ath_retest(df):
    """All-time-high breakout and retest: a close above the highest high of all earlier data (at least
    500 bars of it) marks the old high as a level; within the next 30 bars price comes back to it
    (low within 0.25 ATR) and closes back above it -> buy. Mirror for all-time lows. Stop 1.5 ATR beyond
    the level, target 3R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    n = len(c)
    hh = pd.Series(h).cummax().shift(1).to_numpy()
    ll = pd.Series(l).cummin().shift(1).to_numpy()
    enough = np.arange(n) >= 500
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    st_l, st_s = np.full(n, np.nan), np.full(n, np.nan)
    lvl_up, until_up, lvl_dn, until_dn = np.nan, -1, np.nan, -1
    for i in range(1, n):
        if not enough[i] or not np.isfinite(a[i]):
            continue
        if c[i] > hh[i] and (not np.isfinite(lvl_up) or i > until_up):
            lvl_up, until_up = hh[i], i + 30
        elif np.isfinite(lvl_up) and i <= until_up and l[i] <= lvl_up + 0.25 * a[i] and c[i] > lvl_up:
            long_[i], st_l[i] = True, lvl_up - 1.5 * a[i]
            lvl_up = np.nan
        if c[i] < ll[i] and (not np.isfinite(lvl_dn) or i > until_dn):
            lvl_dn, until_dn = ll[i], i + 30
        elif np.isfinite(lvl_dn) and i <= until_dn and h[i] >= lvl_dn - 0.25 * a[i] and c[i] < lvl_dn:
            short[i], st_s[i] = True, lvl_dn + 1.5 * a[i]
            lvl_dn = np.nan
    return long_, short & ~long_, {"stop": ("level", st_l, st_s), "target": ("R", 3.0)}


def s_ath_breakout(df):
    """Straight all-time-high breakout (52-week-high effect, George & Hwang 2004): a close above the
    highest high of all earlier data (500+ bars) buys at once; mirror for all-time lows. Stop 2 ATR,
    ATR trailing stop 3."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    hh = pd.Series(h).cummax().shift(1).to_numpy()
    ll = pd.Series(l).cummin().shift(1).to_numpy()
    enough = np.arange(n) >= 500
    prev_c = np.r_[np.nan, c[:-1]]
    up = enough & (c > hh) & ~(prev_c > np.r_[np.nan, hh[:-1]])        # first close above, not every bar after
    dn = enough & (c < ll) & ~(prev_c < np.r_[np.nan, ll[:-1]])
    return up, dn, {"stop": ("atr", 2.0), "target": None, "trail_atr": 3.0}


# id: (name, fn, timeframes)
GAPS_BREAKOUTS = {
    "GP01": ("Gap and go after a reopening (>1.5 ATR), 2R", s_gap_and_go, ["15Min", "1Hour"]),
    "GP02": ("Weekend gap fill toward Friday's close", s_weekend_gap_fill, ["15Min", "1Hour"]),
    "GP03": ("Failed gap fade toward the last close", s_failed_gap, ["15Min", "1Hour"]),
    "HB01": ("All-time high/low breakout and retest, 3R", s_ath_retest, ["1Hour", "4Hour"]),
    "HB02": ("All-time high/low breakout, ATR trail", s_ath_breakout, ["1Hour", "4Hour"]),
}
