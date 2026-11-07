"""Swing strategies (research, 07.11.2026): classic daily-bar systems that hold for days, run on 1-hour and
4-hour bars. Same contract as agents/rules_agent.py. "Day" = the New York trading day built from the bars; a
signal known at a day's close enters on the next day's first bar, an exit known at a day's close leaves on the
next day's first bar. Stops are daily: 2-3 x the daily ATR(14) from yesterday's close. Under the FTMO Standard
rule the bots also close everything before market breaks over 2 hours, so these hold Monday..Friday at most.

SW01 Donchian 20-day breakout, out on the 10-day opposite extreme      (R. Donchian / the Turtles)
SW02 50/200-day golden / death cross, out on the opposite cross         (classic trend following)
SW03 Pullback in a trend: above the 50-day SMA and it is rising, two down days and a close under the 10-day EMA
     -> buy; out on a close above yesterday's high                     (L. Connors "Street Smarts" style)
SW04 Keltner breakout: a close above the 20-day EMA + 2 daily ATR, out on a close back under the EMA
SW05 Range expansion: a day's range over 1.8 x the daily ATR closing in its top quarter -> continue next day;
     out on a close under yesterday's low                              (T. Crabel, range expansion)
SW06 Daily MACD(12,26,9) cross in the 200-day trend, out on the opposite cross
SW07 Bollinger(20,2) daily mean reversion in the 200-day trend: a close under the lower band -> buy; out above
     the 20-day SMA
SW08 Momentum dip: up over the last 60 days but down over the last 5 -> buy; out on a 5-day high close
"""
import numpy as np
import pandas as pd

from agents.book_strategies import _days, _on_bars
from agents.rules_agent import ema, sma, rsi  # noqa: F401


def _datr(day, n=14):
    h, l, c = day["h"].to_numpy(), day["l"].to_numpy(), day["c"].to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr, index=day.index).ewm(alpha=1 / n, adjust=False).mean()


def _swing(df, L, S, XL, XS, k_stop=2.5):
    """Daily signals onto bars, with a daily-ATR stop measured from yesterday's close."""
    date, day, first, last = _days(df)
    prev = lambda s: _on_bars(date, pd.Series(s, index=day.index).shift(1).astype(float))  # noqa: E731
    enter_l, enter_s = prev(L) > 0, prev(S) > 0
    ex_l, ex_s = prev(XL) > 0, prev(XS) > 0
    xl, xs = np.r_[ex_l[1:], False], np.r_[ex_s[1:], False]
    a = _datr(day)
    pc, pa = _on_bars(date, day["c"].shift(1)), _on_bars(date, a.shift(1))
    stop_l, stop_s = pc - k_stop * pa, pc + k_stop * pa
    return first & enter_l, first & enter_s & ~enter_l, {
        "stop": ("level", stop_l, stop_s), "target": None,
        "exit_long": xl & np.r_[first[1:], False], "exit_short": xs & np.r_[first[1:], False]}


def s_donchian(df):
    _, day, _, _ = _days(df)
    h, l, c = (day[x].to_numpy() for x in "hlc")
    hi20, lo20 = pd.Series(h).shift(1).rolling(20).max().to_numpy(), pd.Series(l).shift(1).rolling(20).min().to_numpy()
    lo10, hi10 = pd.Series(l).shift(1).rolling(10).min().to_numpy(), pd.Series(h).shift(1).rolling(10).max().to_numpy()
    return _swing(df, c > hi20, c < lo20, c < lo10, c > hi10, 2.0)


def s_golden(df):
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    f, s = sma(c, 50), sma(c, 200)
    up, dn = (f > s) & (np.r_[np.nan, f[:-1]] <= np.r_[np.nan, s[:-1]]), (f < s) & (np.r_[np.nan, f[:-1]] >= np.r_[np.nan, s[:-1]])
    return _swing(df, up, dn, dn | (f < s), up | (f > s), 3.0)


def s_pullback(df):
    _, day, _, _ = _days(df)
    h, l, c = (day[x].to_numpy() for x in "hlc")
    s50, e10 = sma(c, 50), ema(c, 10)
    rising, falling = s50 > np.r_[np.full(5, np.nan), s50[:-5]], s50 < np.r_[np.full(5, np.nan), s50[:-5]]
    d1 = np.r_[False, c[1:] < c[:-1]]
    d2 = d1 & np.r_[False, d1[:-1]]
    u1 = np.r_[False, c[1:] > c[:-1]]
    u2 = u1 & np.r_[False, u1[:-1]]
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    return _swing(df, (c > s50) & rising & d2 & (c < e10), (c < s50) & falling & u2 & (c > e10), c > ph, c < pl, 2.0)


def s_keltner(df):
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    e, a = ema(c, 20), _datr(day).to_numpy()
    return _swing(df, c > e + 2 * a, c < e - 2 * a, c < e, c > e, 2.0)


def s_expansion(df):
    _, day, _, _ = _days(df)
    h, l, c = (day[x].to_numpy() for x in "hlc")
    a = _datr(day).shift(1).to_numpy()
    rng = h - l
    big = rng > 1.8 * a
    pos = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 0.5)
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    return _swing(df, big & (pos >= 0.75), big & (pos <= 0.25), c < pl, c > ph, 2.0)


def s_macd(df):
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    m = ema(c, 12) - ema(c, 26)
    sg = ema(m, 9)
    s200 = sma(c, 200)
    up = (m > sg) & (np.r_[np.nan, m[:-1]] <= np.r_[np.nan, sg[:-1]])
    dn = (m < sg) & (np.r_[np.nan, m[:-1]] >= np.r_[np.nan, sg[:-1]])
    return _swing(df, up & (c > s200), dn & (c < s200), dn, up, 2.5)


def s_bollinger(df):
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    m, sd = sma(c, 20), pd.Series(c).rolling(20).std().to_numpy()
    s200 = sma(c, 200)
    return _swing(df, (c > s200) & (c < m - 2 * sd), (c < s200) & (c > m + 2 * sd), c > m, c < m, 2.5)


def s_momdip(df):
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    r60 = c / np.r_[np.full(60, np.nan), c[:-60]] - 1
    r5 = c / np.r_[np.full(5, np.nan), c[:-5]] - 1
    hi5, lo5 = pd.Series(c).shift(1).rolling(5).max().to_numpy(), pd.Series(c).shift(1).rolling(5).min().to_numpy()
    return _swing(df, (r60 > 0) & (r5 < 0) & (c <= lo5), (r60 < 0) & (r5 > 0) & (c >= hi5), c >= hi5, c <= lo5, 2.5)


# id: (name, fn, timeframes)
SWING = {
    "SW01": ("Donchian 20-day breakout, 10-day exit (Turtles)", s_donchian, ["1Hour", "4Hour"]),
    "SW02": ("50/200-day golden / death cross", s_golden, ["1Hour", "4Hour"]),
    "SW03": ("Pullback in a rising 50-day trend, out above yesterday's high", s_pullback, ["1Hour", "4Hour"]),
    "SW04": ("Keltner daily breakout (EMA20 + 2 ATR)", s_keltner, ["1Hour", "4Hour"]),
    "SW05": ("Range expansion day continuation (Crabel)", s_expansion, ["1Hour", "4Hour"]),
    "SW06": ("Daily MACD cross in the 200-day trend", s_macd, ["1Hour", "4Hour"]),
    "SW07": ("Bollinger daily mean reversion in the 200-day trend", s_bollinger, ["1Hour", "4Hour"]),
    "SW08": ("Momentum dip: 60-day up, 5-day low", s_momdip, ["1Hour", "4Hour"]),
}
