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
from agents.rules_agent import ema, sma, rsi, adx  # noqa: F401


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


# ---- Round 2 (07.11.2026): published swing systems WITH their own trade management -------------------------
# Every rule as its source states it; where the source has no stop a wide 3 x daily-ATR safety stop is used
# (the bots never trade without one). The engine has no partial exits: "sell part, stop to breakeven" is
# done as "stop to breakeven once 1R in profit", the rest is managed by the source's own exit.
#   SW09 Connors RSI(2): close > 200-day SMA, RSI(2) < 10 -> buy; out on a close above the 5-day SMA
#        (L. Connors & C. Alvarez, "Short Term Trading Strategies That Work"; quantifiedstrategies, therobusttrader)
#   SW10 Connors Double Seven: close > 200-day SMA and a 7-day low close -> buy; out on a 7-day high close
#   SW11 Turtles System 2: close over the 55-day high -> buy; out on a close under the 20-day low; stop 2 N
#        (N = 20-day ATR)                                                         (R. Dennis / C. Faith)
#   SW12 Qullamaggie breakout: a stock in a rising 50-day trend, up over 20 days, a tight 10-day range
#        (under 2.5 daily ATR), a close above it -> buy; stop = the breakout day's low; stop to breakeven
#        at +1R (he sells 1/3-1/2 after 3-5 days and moves the stop to breakeven); the rest out on a close
#        under the 10-day SMA                                                    (K. Kullamagi, TradingView scripts)
#   SW13 Raschke Holy Grail: daily ADX(14) > 30, a day touching the 20-day SMA in an up trend and closing up
#        -> buy; stop = that day's low; target = the 20-day high (the recent swing high)
#                                                                                (L. Raschke, "Street Smarts")
#   SW14 20-EMA reversion: over the 200-day EMA, the low within 2% of the 20-day EMA, RSI(14) 40-55, an up
#        candle -> buy; stop = the nearer of the candle low and EMA20 - 1.5%; target = the recent swing high
#        (20-day high); out after 2 closes under the EMA20                       (TradeZella swing guide)
#   SW15 Chandelier trend: a 50-day high close with EMA50 > EMA200 -> buy; stop and trailing exit =
#        22-day high - 3 x ATR(22), out on a close under it                     (C. LeBeau, StockCharts)
#   SW16 Fibonacci pullback: up over 6 daily ATR in 25 days, back to 50-61.8% of that swing, an up candle
#        -> buy; stop under the 78.6% level; target = the swing high              (TradeZella swing guide)
# All have the mirror short (under the 200-day line, etc.).

def _swing2(df, L, S, XL, XS, stop_l, stop_s, tgt_l=None, tgt_s=None, be_r=None):
    """Daily signals onto bars; stops / targets are daily levels known at the signal day's close."""
    date, day, first, last = _days(df)
    prev = lambda s: _on_bars(date, pd.Series(np.asarray(s, dtype=float), index=day.index).shift(1))  # noqa: E731
    enter_l, enter_s = prev(L) > 0, prev(S) > 0
    ex_l, ex_s = prev(XL) > 0, prev(XS) > 0
    xl, xs = np.r_[ex_l[1:], False], np.r_[ex_s[1:], False]
    plan = {"stop": ("level", prev(stop_l), prev(stop_s)), "target": None,
            "exit_long": xl & np.r_[first[1:], False], "exit_short": xs & np.r_[first[1:], False]}
    if tgt_l is not None:
        plan["target"] = ("level", prev(tgt_l), prev(tgt_s))
    if be_r:
        plan["be_r"] = be_r
    return first & enter_l, first & enter_s & ~enter_l, plan


def _d(df):
    _, day, _, _ = _days(df)
    return day, *(day[x].to_numpy() for x in "ohlc")


def _roll(x, n, f):
    return getattr(pd.Series(x).rolling(n), f)().to_numpy()


def s_rsi2(df):
    day, o, h, l, c = _d(df)
    r, s200, s5, a = rsi(c, 2), sma(c, 200), sma(c, 5), _datr(day).to_numpy()
    return _swing2(df, (c > s200) & (r < 10), (c < s200) & (r > 90), c > s5, c < s5, c - 3 * a, c + 3 * a)


def s_double7(df):
    day, o, h, l, c = _d(df)
    s200, a = sma(c, 200), _datr(day).to_numpy()
    lo7, hi7 = _roll(c, 7, "min"), _roll(c, 7, "max")
    return _swing2(df, (c > s200) & (c <= lo7), (c < s200) & (c >= hi7), c >= hi7, c <= lo7, c - 3 * a, c + 3 * a)


def s_turtle55(df):
    day, o, h, l, c = _d(df)
    n = _datr(day, 20).to_numpy()
    hi55, lo55 = pd.Series(h).shift(1).rolling(55).max().to_numpy(), pd.Series(l).shift(1).rolling(55).min().to_numpy()
    lo20, hi20 = pd.Series(l).shift(1).rolling(20).min().to_numpy(), pd.Series(h).shift(1).rolling(20).max().to_numpy()
    return _swing2(df, c > hi55, c < lo55, c < lo20, c > hi20, c - 2 * n, c + 2 * n)


def s_qulla(df):
    day, o, h, l, c = _d(df)
    a, s50, s10 = _datr(day).to_numpy(), sma(c, 50), sma(c, 10)
    hi10, lo10 = pd.Series(h).shift(1).rolling(10).max().to_numpy(), pd.Series(l).shift(1).rolling(10).min().to_numpy()
    tight = (hi10 - lo10) < 2.5 * a
    up = (c > s50) & (s50 > np.r_[np.full(10, np.nan), s50[:-10]]) & (c > np.r_[np.full(20, np.nan), c[:-20]])
    dn = (c < s50) & (s50 < np.r_[np.full(10, np.nan), s50[:-10]]) & (c < np.r_[np.full(20, np.nan), c[:-20]])
    return _swing2(df, up & tight & (c > hi10), dn & tight & (c < lo10), c < s10, c > s10, l, h, be_r=1.0)


def s_holygrail(df):
    day, o, h, l, c = _d(df)
    s20, ax = sma(c, 20), adx(h, l, c, 14)
    hi20, lo20 = _roll(h, 20, "max"), _roll(l, 20, "min")
    pc = np.r_[np.nan, c[:-1]]
    L = (ax > 30) & (pc > np.r_[np.nan, s20[:-1]]) & (l <= s20) & (h >= s20) & (c > o)
    S = (ax > 30) & (pc < np.r_[np.nan, s20[:-1]]) & (l <= s20) & (h >= s20) & (c < o)
    z = np.zeros(len(c), dtype=bool)
    return _swing2(df, L, S, z, z, l, h, hi20, lo20)


def s_ema20rev(df):
    day, o, h, l, c = _d(df)
    e20, e200, r = ema(c, 20), ema(c, 200), rsi(c, 14)
    hi20, lo20 = _roll(h, 20, "max"), _roll(l, 20, "min")
    L = (c > e200) & (l <= e20 * 1.02) & (c >= e20) & (r >= 40) & (r <= 55) & (c > o)
    S = (c < e200) & (h >= e20 * 0.98) & (c <= e20) & (r >= 45) & (r <= 60) & (c < o)
    below2 = (c < e20) & np.r_[False, c[:-1] < e20[:-1]]
    above2 = (c > e20) & np.r_[False, c[:-1] > e20[:-1]]
    return _swing2(df, L, S, below2, above2, np.maximum(l, e20 * 0.985), np.minimum(h, e20 * 1.015), hi20, lo20)


def s_chandelier(df):
    day, o, h, l, c = _d(df)
    e50, e200, a22 = ema(c, 50), ema(c, 200), _datr(day, 22).to_numpy()
    cl, cs = _roll(h, 22, "max") - 3 * a22, _roll(l, 22, "min") + 3 * a22
    hi50, lo50 = pd.Series(c).shift(1).rolling(50).max().to_numpy(), pd.Series(c).shift(1).rolling(50).min().to_numpy()
    return _swing2(df, (c > hi50) & (e50 > e200), (c < lo50) & (e50 < e200), c < cl, c > cs, cl, cs)


def s_fib(df):
    day, o, h, l, c = _d(df)
    a = _datr(day).to_numpy()
    hh, ll = _roll(h, 25, "max"), _roll(l, 25, "min")
    rng = hh - ll
    hi_i = pd.Series(h).rolling(25).apply(np.argmax, raw=True).to_numpy()   # where in the window the extreme sits
    lo_i = pd.Series(l).rolling(25).apply(np.argmin, raw=True).to_numpy()
    upsw, dnsw = (rng > 6 * a) & (hi_i > lo_i), (rng > 6 * a) & (lo_i > hi_i)
    L = upsw & (c <= hh - 0.5 * rng) & (c >= hh - 0.618 * rng) & (c > o)
    S = dnsw & (c >= ll + 0.5 * rng) & (c <= ll + 0.618 * rng) & (c < o)
    stop_l, stop_s = hh - 0.786 * rng, ll + 0.786 * rng
    return _swing2(df, L, S, c < stop_l, c > stop_s, stop_l, stop_s, hh, ll)


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
    "SW09": ("Connors RSI(2) over the 200-day line, out above the 5-day SMA", s_rsi2, ["1Hour", "4Hour"]),
    "SW10": ("Connors Double Seven", s_double7, ["1Hour", "4Hour"]),
    "SW11": ("Turtles System 2: 55-day breakout, 20-day exit, 2N stop", s_turtle55, ["1Hour", "4Hour"]),
    "SW12": ("Qullamaggie breakout: tight range, low-of-day stop, breakeven at 1R, out under the 10-day SMA",
             s_qulla, ["1Hour", "4Hour"]),
    "SW13": ("Raschke Holy Grail: ADX>30 pullback to the 20-day SMA, target the swing high", s_holygrail, ["1Hour", "4Hour"]),
    "SW14": ("20-EMA reversion in the 200-day trend, target the swing high", s_ema20rev, ["1Hour", "4Hour"]),
    "SW15": ("Chandelier trend: 50-day high, 22-day high - 3 ATR trailing exit", s_chandelier, ["1Hour", "4Hour"]),
    "SW16": ("Fibonacci 50-61.8% pullback, stop under 78.6%, target the swing high", s_fib, ["1Hour", "4Hour"]),
}
