"""Strategies from the Quantpedia library (quantpedia.com) that fit FTMO CFDs and are not covered by an
existing family. Same contract as agents/rules_agent.py.

QP01 intraday momentum      Gao, Han, Li & Zhou (2018), "Market intraday momentum"
QP02 time-series momentum   Moskowitz, Ooi & Pedersen (2012), weekly rebalanced, 3-month lookback
QP03 option-expiration week Stivers & Sun (2013), the monthly OpEx week is strong for US shares/indices
QP04 lunar cycle            Yuan, Zheng & Zhu (2006), returns are higher around the new moon
QP05 London 4pm fix reversal Melvin & Prins (2015) / Krohn et al.: the dollar move into the WM fix reverses
"""
import numpy as np
import pandas as pd

from agents.rules_agent import atr

NY, LDN = "America/New_York", "Europe/London"


def _local(df, tz):
    t = df.index.tz_convert(tz)
    return t, t.hour.to_numpy(), t.minute.to_numpy(), np.array([d.isoformat() for d in t.date])


def _day_flags(date):
    first = np.r_[True, date[1:] != date[:-1]]
    last = np.r_[date[1:] != date[:-1], True]
    return first, last


def s_intraday_momentum(df):
    """Market intraday momentum: the return from yesterday's close to 10:00 New York predicts the last
    half hour. At 15:30 enter in the direction of that first-half-hour move if it was bigger than 0.5 ATR
    of the 15-minute bars x 4; out at 16:00. Stop 1.5 ATR."""
    _, hour, minute, date = _local(df, NY)
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    first, last = _day_flags(date)
    close_by_day = pd.Series(c[last], index=date[last])
    prev_close = close_by_day.shift(1)
    pc = np.array([prev_close.get(d, np.nan) for d in date])
    at10 = (hour == 9) & (minute == 45)                 # the 9:45-10:00 bar closes at 10:00
    r1 = pd.Series(np.where(at10, c - pc, np.nan)).groupby(date).transform("max").to_numpy()
    sig_bar = (hour == 15) & (minute == 15)             # closes at 15:30 -> enter at 15:30
    up = sig_bar & (r1 > 2 * a)
    dn = sig_bar & (r1 < -2 * a)
    return up, dn, {"stop": ("atr", 1.5), "target": None, "force_exit": (hour == 15) & (minute == 45)}


def s_ts_momentum(df):
    """Time-series momentum: the sign of the instrument's own 63-trading-day return decides the week's
    direction (long if positive, short if negative). Entered on each day's first bar of the week, out on
    the week's last bar. Stop 6 ATR."""
    t, _, _, date = _local(df, NY)
    c = df["c"].to_numpy()
    first, last = _day_flags(date)
    daily = pd.Series(c[last], index=pd.to_datetime(date[last]))
    wk = daily.resample("W-FRI").last()
    sig = np.sign(daily.pct_change(63)).resample("W-FRI").last()       # known at the week's end
    week_of = t.tz_localize(None).to_period("W-FRI")
    prev_sig = {str(p): v for p, v in zip(sig.index.to_period("W-FRI") + 1, sig.to_numpy())}
    s = np.array([prev_sig.get(str(p), 0.0) for p in week_of])
    w = np.array([str(p) for p in week_of])
    week_last = np.r_[w[1:] != w[:-1], True]
    del wk
    return first & (s > 0), first & (s < 0), {"stop": ("atr", 6.0), "target": None, "force_exit": week_last}


def _third_friday(y, m):
    d = pd.Timestamp(year=y, month=m, day=15)
    return d + pd.Timedelta(days=(4 - d.dayofweek) % 7)


def s_opex_week(df):
    """Option-expiration week: long from Monday to Friday of the week with the month's third Friday
    (re-entered each morning, out at each day's / the week's end for shares). Stop 6 ATR."""
    t, _, _, date = _local(df, NY)
    first, last = _day_flags(date)
    months = {(x.year, x.month) for x in t}
    weeks = set()
    for y, m in months:
        f = _third_friday(y, m)
        for k in range(5):
            weeks.add((f - pd.Timedelta(days=4 - k)).date().isoformat())
    in_week = np.isin(date, list(weeks))
    fri = t.dayofweek.to_numpy() == 4
    return first & in_week, np.zeros(len(date), bool), {"stop": ("atr", 6.0), "target": None,
                                                         "force_exit": last & fri & in_week}


SYNODIC = 29.530588853
NEW_MOON_REF = pd.Timestamp("2000-01-06 18:14", tz="UTC")


def moon_age(ts):
    return ((ts - NEW_MOON_REF) / pd.Timedelta(days=1)) % SYNODIC


def s_lunar(df):
    """Lunar cycle: buy on each day's first bar during the new-moon half of the cycle (moon age under
    7.4 or over 22.1 days), sell during the full-moon half; out at the day's end. Stop 2 ATR."""
    _, _, _, date = _local(df, NY)
    first, last = _day_flags(date)
    age = moon_age(df.index)
    new = (age < 7.4) | (age > 22.1)
    return first & new, first & ~new, {"stop": ("atr", 2.0), "target": None, "force_exit": last}


def s_london_fix_reversal(df):
    """London 4pm fix reversal (currencies and gold/silver): the move from 15:00 to 16:00 London into the
    WM/Reuters fix tends to reverse. If that hour moved more than 1 ATR (x2 for 15-minute bars), enter
    against it at 16:00, target the 15:00 price, stop 1.5 x the move beyond, at most 2 hours."""
    _, hour, minute, date = _local(df, LDN)
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    at15 = (hour == 15) & (minute == 0)
    p15 = pd.Series(np.where(at15, o, np.nan)).groupby(date).transform("max").to_numpy()
    sig = (hour == 15) & (minute == 45)                 # closes at 16:00 -> enter at 16:00
    move = c - p15
    long_ = sig & (move < -2 * a)
    short = sig & (move > 2 * a)
    st_l, st_s = c + 1.5 * move, c + 1.5 * move         # move < 0 for longs: below; > 0 for shorts: above
    return long_, short, {"stop": ("level", st_l, st_s), "target": ("level", p15, p15), "max_bars": 8}


# id: (name, fn, timeframes)
QUANTPEDIA = {
    "QP01": ("Market intraday momentum: first half hour -> last half hour (Quantpedia)", s_intraday_momentum, ["15Min"]),
    "QP02": ("Time-series momentum, 3-month sign, weekly (Quantpedia)", s_ts_momentum, ["1Hour"]),
    "QP03": ("Option-expiration week long (Quantpedia)", s_opex_week, ["1Hour"]),
    "QP04": ("Lunar cycle: new moon long / full moon short (Quantpedia)", s_lunar, ["1Hour"]),
    "QP05": ("London 4pm fix reversal (Quantpedia)", s_london_fix_reversal, ["15Min"]),
}
