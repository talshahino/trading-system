"""Seasonality and timing (family 5 of the strategy map): returns that depend on the clock and the
calendar rather than on the price pattern. Same contract as agents/rules_agent.py: signals at a bar's
close, entry at the next bar's open, always a stop (FTMO), plus a time exit.

All clock rules use New York time (the US session drives indices, gold, the dollar and US shares), so
daylight-saving changes are handled. The adaptive ones (SE04, SE05) learn only from the past: at every
bar they look at the same hour / weekday over the previous weeks, never at the bar being traded.
"""
import numpy as np
import pandas as pd

NY = "America/New_York"

# FOMC rate decisions (statement 14:00 New York)
FOMC = {"2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"}
# NYSE full-day holidays
NYSE_HOLIDAYS = {"2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
                 "2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
                 "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"}


def _ny(df):
    t = df.index.tz_convert(NY)
    return t.hour.to_numpy(), np.array([d.isoformat() for d in t.date]), t.dayofweek.to_numpy()


def _day_ends(date):
    """True on the last bar of each New York date."""
    n = len(date)
    end = np.ones(n, bool)
    end[:-1] = date[1:] != date[:-1]
    return end


def _next_trading_day(date):
    """For every bar: the New York date of the next trading day in the data (None at the end)."""
    days = list(dict.fromkeys(date))
    nxt = {d: (days[i + 1] if i + 1 < len(days) else None) for i, d in enumerate(days)}
    return np.array([nxt[d] for d in date], dtype=object)


def s_overnight_long(df):
    """Overnight drift: most of the indices' long-run gain has come between the close and the next
    open (Cooper, Cliff & Gulen 2008). Buy at the New York close (16:00), sell at 09:00 before the
    open. Stop 2 ATR."""
    hour, _, _ = _ny(df)
    return hour == 15, np.zeros(len(hour), bool), {"stop": ("atr", 2.0), "target": None, "force_exit": hour == 8}


def s_ny_session_long(df):
    """The opposite slice: buy at 10:00 New York (after the opening half hour), sell at 16:00.
    Stop 1.5 ATR."""
    hour, _, _ = _ny(df)
    return hour == 9, np.zeros(len(hour), bool), {"stop": ("atr", 1.5), "target": None, "force_exit": hour == 15}


def s_turnaround_tuesday(df):
    """Turnaround Tuesday: if Monday closed below Friday's close, buy at Monday's 16:00 close and sell
    at Tuesday's close. Stop 2 ATR."""
    hour, date, dow = _ny(df)
    c = df["c"].to_numpy()
    ends = _day_ends(date)
    closes = pd.Series(c[ends], index=date[ends])
    prev_close = closes.shift(1)
    prev = np.array([prev_close.get(d, np.nan) for d in date])
    long_ = (dow == 0) & (hour == 15) & (c < prev)
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 2.0), "target": None, "force_exit": (dow == 1) & (hour == 15)}


def s_hour_of_day(df):
    """Adaptive hour-of-day: for the coming hour, look at the same hour on the previous 60 days. If its
    average return is clearly positive (t > 2) buy for that one hour, clearly negative (t < -2) sell.
    Stop 1.5 ATR, exit at that bar's close."""
    t = df.index.tz_convert(NY)
    hour = t.hour.to_numpy()
    c = df["c"].to_numpy()
    ret = pd.Series(c).pct_change().to_numpy()
    s = pd.DataFrame({"h": hour, "r": ret})
    g = s.groupby("h")["r"]
    mean = g.transform(lambda x: x.shift(1).rolling(60, min_periods=40).mean()).to_numpy()
    sd = g.transform(lambda x: x.shift(1).rolling(60, min_periods=40).std()).to_numpy()
    tstat = mean / sd * np.sqrt(60)
    nxt = np.r_[tstat[1:], np.nan]                    # the coming bar's hour, learned only from earlier days
    return nxt > 2, nxt < -2, {"stop": ("atr", 1.5), "target": None, "max_bars": 1}


def s_day_of_week(df):
    """Adaptive day-of-week: for the coming New York day, look at the same weekday over the previous 26
    weeks (close-to-close). t > 2: buy at today's close and hold the day; t < -2: sell. Stop 2 ATR,
    exit at that day's close."""
    _, date, dow = _ny(df)
    c = df["c"].to_numpy()
    ends = _day_ends(date)
    days = pd.DataFrame({"d": date[ends], "dow": dow[ends], "c": c[ends]})
    days["r"] = days["c"].pct_change()
    g = days.groupby("dow")["r"]
    mean = g.transform(lambda x: x.shift(1).rolling(26, min_periods=15).mean())
    sd = g.transform(lambda x: x.shift(1).rolling(26, min_periods=15).std())
    days["t"] = mean / sd * np.sqrt(26)
    t_by_day = dict(zip(days["d"], days["t"]))
    nxt = _next_trading_day(date)
    tt = np.array([t_by_day.get(d, np.nan) if d is not None else np.nan for d in nxt], dtype=float)
    long_, short = ends & (tt > 2), ends & (tt < -2)
    return long_, short, {"stop": ("atr", 2.0), "target": None, "force_exit": ends}


def s_pre_fomc(df):
    """Pre-FOMC drift (Lucca & Moench 2015): shares rise in the 24 hours before a Fed rate decision.
    Buy at 14:00 New York the day before, sell at 13:00 on decision day (before FTMO's restricted
    window around the 14:00 statement). Stop 2 ATR."""
    hour, date, _ = _ny(df)
    nxt = _next_trading_day(date)
    long_ = (hour == 13) & np.array([d in FOMC for d in nxt])
    return long_, np.zeros(len(hour), bool), {"stop": ("atr", 2.0), "target": None,
                                              "force_exit": (hour == 12) & np.isin(date, list(FOMC))}


def s_pre_holiday(df):
    """Pre-holiday effect (Ariel 1990): the trading day before a US market holiday tends to be strong.
    Buy at the close two days before, sell at the close of the last day before the holiday. Stop 2 ATR."""
    hour, date, _ = _ny(df)
    days = list(dict.fromkeys(date))
    pre = set()
    for d in days:
        if pd.Timestamp(str(d)).dayofweek >= 5:          # weekend bars (crypto, FX Sunday evening) are not a trading day
            continue
        nd = pd.Timestamp(str(d)) + pd.Timedelta(days=1)
        while nd.dayofweek >= 5:
            nd += pd.Timedelta(days=1)
        if nd.date().isoformat() in NYSE_HOLIDAYS:
            pre.add(d)
    nxt = _next_trading_day(date)
    long_ = (hour == 15) & np.array([d in pre for d in nxt])
    return long_, np.zeros(len(hour), bool), {"stop": ("atr", 2.0), "target": None,
                                              "force_exit": (hour == 15) & np.isin(date, list(pre))}


# id: (name, fn, timeframes)
SEASONAL = {
    "SE01": ("Overnight drift long (NY close -> 09:00)", s_overnight_long, ["1Hour"]),
    "SE02": ("NY session long (10:00 -> 16:00)", s_ny_session_long, ["1Hour"]),
    "SE03": ("Turnaround Tuesday (Monday down -> buy)", s_turnaround_tuesday, ["1Hour"]),
    "SE04": ("Adaptive hour-of-day (60-day t-stat)", s_hour_of_day, ["1Hour"]),
    "SE05": ("Adaptive day-of-week (26-week t-stat)", s_day_of_week, ["1Hour"]),
    "SE06": ("Pre-FOMC drift (day before -> 13:00)", s_pre_fomc, ["1Hour"]),
    "SE07": ("Pre-holiday effect (US market holidays)", s_pre_holiday, ["1Hour"]),
}
