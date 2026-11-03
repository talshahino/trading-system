"""Strategies from classic trading books, coded from their published rules. Same contract as
agents/rules_agent.py. The books' rules are for daily bars; here "day" = the New York trading day built
from the intraday bars, and intraday rules use the day's own bars.

BK01 ConnorsRSI pullback           Connors & Alvarez, "An Introduction to ConnorsRSI" (2012)
BK02 R3 strategy                   Connors & Alvarez, "High Probability ETF Trading" (2009)
BK03 Multiple days down (MDD)      Connors & Alvarez, "High Probability ETF Trading" (2009)
BK04 Oops! (gap below prior low)   Larry Williams, "Long-Term Secrets to Short-Term Trading" (1999)
BK05 Volatility breakout           Larry Williams, same book
BK06 Buy-on-gap model              Ernest Chan, "Algorithmic Trading" (2013)
BK07 Opening-range breakout on NR4 Toby Crabel, "Day Trading with Short Term Price Patterns" (1990)
"""
import numpy as np
import pandas as pd

from agents.rules_agent import atr, rsi, sma

NY = "America/New_York"


def _days(df):
    """Per bar: its NY date; per day: open, high, low, close (indexed by date) and first/last flags."""
    t = df.index.tz_convert(NY)
    date = np.array([d.isoformat() for d in t.date])
    g = pd.DataFrame({"d": date, "o": df["o"].to_numpy(), "h": df["h"].to_numpy(), "l": df["l"].to_numpy(), "c": df["c"].to_numpy()})
    day = g.groupby("d", sort=False).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    first = np.r_[True, date[1:] != date[:-1]]
    last = np.r_[date[1:] != date[:-1], True]
    return date, day, first, last


def _on_bars(date, series):
    """A per-day series (indexed by date) spread onto every bar of that day."""
    return pd.Series(date).map(series).to_numpy(dtype=float)


def _daily_signals(df, long_day, short_day, exit_long_day, exit_short_day, stop_atr=3.0):
    """Daily-bar systems: a signal known at a day's close enters on the next day's first bar; an exit
    condition known at a day's close leaves on the next day's first bar."""
    date, day, first, last = _days(df)
    prev = lambda s: _on_bars(date, s.shift(1).astype(float))       # yesterday's value on today's bars  # noqa: E731
    enter_l, enter_s = prev(long_day) > 0, prev(short_day) > 0
    ex_l, ex_s = prev(exit_long_day) > 0, prev(exit_short_day) > 0
    # exits are checked at a bar's close and filled at the next open: put them on the previous bar
    xl, xs = np.r_[ex_l[1:], False], np.r_[ex_s[1:], False]
    return first & enter_l, first & enter_s & ~enter_l, {"stop": ("atr", stop_atr), "target": None,
                                                         "exit_long": xl & np.r_[first[1:], False],
                                                         "exit_short": xs & np.r_[first[1:], False]}


def connors_rsi(c, n_rsi=3, n_streak=2, n_rank=100):
    c = pd.Series(c, dtype=float)
    streak = np.zeros(len(c))
    d = np.sign(c.diff().fillna(0).to_numpy())
    for i in range(1, len(c)):
        streak[i] = (streak[i - 1] + d[i] if d[i] == np.sign(streak[i - 1]) and d[i] != 0 else d[i])
    roc = c.pct_change()
    rank = roc.rolling(n_rank + 1).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
    return (pd.Series(rsi(c.to_numpy(), n_rsi)) + pd.Series(rsi(streak, n_streak)) + rank.reset_index(drop=True)) / 3


def s_connors_rsi(df):
    """ConnorsRSI pullback: in an uptrend (close > 200-day SMA) a ConnorsRSI(3,2,100) under 10 buys the
    next day; out when the close is back above the 5-day SMA. Mirror for downtrends (CRSI over 90)."""
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    crsi = connors_rsi(c).to_numpy()
    s200, s5 = sma(c, 200), sma(c, 5)
    L = pd.Series((c > s200) & (crsi < 10), index=day.index)
    S = pd.Series((c < s200) & (crsi > 90), index=day.index)
    return _daily_signals(df, L, S, pd.Series(c > s5, index=day.index), pd.Series(c < s5, index=day.index))


def s_r3(df):
    """R3: above the 200-day SMA, RSI(2) falls three days in a row, the first day under 60 and the last
    under 10 -> buy; out when RSI(2) is over 70. Mirror below the SMA200."""
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    r = rsi(c, 2)
    s200 = sma(c, 200)
    r1, r2, r3 = np.r_[np.nan, r[:-1]], np.r_[np.nan, np.nan, r[:-2]], np.r_[np.nan, np.nan, np.nan, r[:-3]]
    fall = (r < r1) & (r1 < r2) & (r2 < r3)
    rise = (r > r1) & (r1 > r2) & (r2 > r3)
    L = pd.Series((c > s200) & fall & (r2 < 60) & (r < 10), index=day.index)
    S = pd.Series((c < s200) & rise & (r2 > 40) & (r > 90), index=day.index)
    return _daily_signals(df, L, S, pd.Series(r > 70, index=day.index), pd.Series(r < 30, index=day.index))


def s_mdd(df):
    """Multiple days down: above the 200-day SMA and below the 5-day SMA, with 4 of the last 5 days
    down -> buy; out on a close above the 5-day SMA. Mirror: multiple days up below the SMA200."""
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    s200, s5 = sma(c, 200), sma(c, 5)
    down = pd.Series(np.r_[False, c[1:] < c[:-1]]).rolling(5).sum().to_numpy()
    up = pd.Series(np.r_[False, c[1:] > c[:-1]]).rolling(5).sum().to_numpy()
    L = pd.Series((c > s200) & (c < s5) & (down >= 4), index=day.index)
    S = pd.Series((c < s200) & (c > s5) & (up >= 4), index=day.index)
    return _daily_signals(df, L, S, pd.Series(c > s5, index=day.index), pd.Series(c < s5, index=day.index))


def s_oops(df):
    """Larry Williams' Oops!: the day opens below yesterday's low (a gap down) and then trades back up
    through yesterday's low -> buy (the gap failed). Stop at the day's low so far, out at the day's end.
    Mirror for a gap above yesterday's high."""
    date, day, first, last = _days(df)
    h, l, c = (df[x].to_numpy() for x in "hlc")
    pl, ph = _on_bars(date, day["l"].shift(1)), _on_bars(date, day["h"].shift(1))
    dop = _on_bars(date, day["o"])
    lo_sofar = pd.Series(l).groupby(date).cummin().to_numpy()
    hi_sofar = pd.Series(h).groupby(date).cummax().to_numpy()
    gap_dn, gap_up = dop < pl, dop > ph
    xl = gap_dn & (c > pl) & (np.r_[np.nan, c[:-1]] <= pl) & ~first
    xs = gap_up & (c < ph) & (np.r_[np.nan, c[:-1]] >= ph) & ~first
    once_l = pd.Series(xl).groupby(date).cumsum().to_numpy() == 1
    once_s = pd.Series(xs).groupby(date).cumsum().to_numpy() == 1
    return xl & once_l, xs & once_s & ~xl, {"stop": ("level", lo_sofar, hi_sofar), "target": None, "force_exit": last}


def s_vol_breakout(df):
    """Larry Williams' volatility breakout: a close above today's open + 0.6 x yesterday's range buys,
    below the open - 0.6 x range sells (first break of the day). Stop at today's open, out at the day's
    end."""
    date, day, first, last = _days(df)
    c = df["c"].to_numpy()
    rng = _on_bars(date, (day["h"] - day["l"]).shift(1))
    dop = _on_bars(date, day["o"])
    up, dn = c > dop + 0.6 * rng, c < dop - 0.6 * rng
    k = pd.Series(up | dn).groupby(date).cumsum().to_numpy() == 1
    return up & k, dn & k & ~up, {"stop": ("level", dop, dop), "target": None, "force_exit": last}


def s_buy_on_gap(df):
    """Ernest Chan's buy-on-gap: the day opens more than one standard deviation (90-day close-to-close)
    below yesterday's low while yesterday's close is above its 20-day SMA -> buy after the first bar,
    out at the day's close. Mirror for gaps above yesterday's high below the SMA20. Stop 2 ATR."""
    date, day, first, last = _days(df)
    sd = day["c"].pct_change().rolling(90, min_periods=40).std()
    ma20 = day["c"].rolling(20).mean()
    pl, ph, pc = (_on_bars(date, day[x].shift(1)) for x in ("l", "h", "c"))
    sdv, ma = _on_bars(date, sd.shift(1)), _on_bars(date, ma20.shift(1))
    dop = _on_bars(date, day["o"])
    long_ = first & (dop < pl * (1 - sdv)) & (pc > ma)
    short = first & (dop > ph * (1 + sdv)) & (pc < ma)
    return long_, short, {"stop": ("atr", 2.0), "target": None, "force_exit": last}


def s_crabel_nr4(df):
    """Toby Crabel's opening-range breakout on an NR4 day: if yesterday had the narrowest range of the
    last 4 days, the first close above the first hour's high buys / below its low sells. Stop at the
    other side of the first hour, out at the day's end."""
    date, day, first, last = _days(df)
    h, l, c = (df[x].to_numpy() for x in "hlc")
    t = df.index.tz_convert(NY)
    rng = day["h"] - day["l"]
    nr4 = (rng <= rng.rolling(4).min()).shift(1)
    nr4_b = _on_bars(date, nr4.astype(float)) > 0
    mins = ((t - t.normalize()) / pd.Timedelta(minutes=1)).to_numpy()
    day_start = pd.Series(mins).groupby(date).transform("min").to_numpy()
    in_first = mins < day_start + 60
    oh = pd.Series(np.where(in_first, h, np.nan)).groupby(date).transform("max").to_numpy()
    ol = pd.Series(np.where(in_first, l, np.nan)).groupby(date).transform("min").to_numpy()
    after = ~in_first
    up, dn = after & nr4_b & (c > oh), after & nr4_b & (c < ol)
    k = pd.Series(up | dn).groupby(date).cumsum().to_numpy() == 1
    return up & k, dn & k & ~up, {"stop": ("level", ol, oh), "target": None, "force_exit": last}


# id: (name, fn, timeframes)
BOOKS = {
    "BK01": ("ConnorsRSI pullback (Connors & Alvarez)", s_connors_rsi, ["1Hour"]),
    "BK02": ("R3 strategy (Connors & Alvarez)", s_r3, ["1Hour"]),
    "BK03": ("Multiple days down/up (Connors & Alvarez)", s_mdd, ["1Hour"]),
    "BK04": ("Oops! failed gap through yesterday's low/high (L. Williams)", s_oops, ["15Min", "1Hour"]),
    "BK05": ("Volatility breakout 0.6 x yesterday's range (L. Williams)", s_vol_breakout, ["15Min", "1Hour"]),
    "BK06": ("Buy-on-gap model (E. Chan)", s_buy_on_gap, ["15Min", "1Hour"]),
    "BK07": ("Opening-range breakout after an NR4 day (T. Crabel)", s_crabel_nr4, ["15Min", "1Hour"]),
}
