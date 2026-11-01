"""Session-based and classic short-term systems (01.11.2026), coded from their published rules.
Same contract as agents/rules_agent.py: fn(df) -> (long, short, plan), signal at the bar close,
entry at the next bar's open. df is indexed in UTC with columns o, h, l, c, v.

SS01-SS07: trading-session systems (New York / London opening ranges, Asian-session fade,
           VWAP reversion and pullback, previous-day false breakout, opening-gap fill)
SS08-SS15: classic short-term systems (Connors, Raschke, Crabel-style contraction)
"""
import numpy as np
import pandas as pd

from agents.rules_agent import _prev_day_hlc, adx, atr, ema, rsi, sma


def _local(df, tz):
    loc = df.index.tz_convert(tz)
    mins = (loc.hour * 60 + loc.minute).to_numpy()
    day = pd.Series(loc.date).to_numpy()
    return mins, day


def _first(sig, day):
    """Only the first True of each day."""
    return sig & (pd.Series(sig).groupby(day).cumsum().to_numpy() == 1)


def _opening_range(df, tz, start_min, length_min, last_entry_min, close_min, target_r=2.0):
    """Opening-range breakout: the range of the first `length_min` minutes after `start_min`
    (local time). After it, the first close above/below the range enters (until last_entry_min);
    stop at the other side of the range, target `target_r` R, flat at close_min."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    mins, day = _local(df, tz)
    in_or = (mins >= start_min) & (mins < start_min + length_min)
    g = pd.DataFrame({"d": day, "h": np.where(in_or, h, np.nan), "l": np.where(in_or, l, np.nan)})
    rng = g.groupby("d").agg(oh=("h", "max"), ol=("l", "min"))
    m = g[["d"]].join(rng, on="d")
    oh, ol = m["oh"].to_numpy(), m["ol"].to_numpy()
    after = (mins >= start_min + length_min) & (mins < last_entry_min)
    up, dn = after & (c > oh), after & (c < ol)
    first = _first(up | dn, day)
    return up & first, dn & first & ~up, {"stop": ("level", ol, oh), "target": ("R", target_r),
                                          "force_exit": mins >= close_min}


def s_ny_orb15(df):
    """New York 15-minute opening-range breakout (9:30-9:45 NY), entries until 12:00 NY,
    stop at the other side of the range, target 2R, flat at 15:45 NY."""
    return _opening_range(df, "America/New_York", 570, 15, 720, 945)


def s_ny_orb30(df):
    """New York 30-minute opening-range breakout (9:30-10:00 NY), entries until 12:30 NY,
    stop at the other side of the range, target 1.5R, flat at 15:45 NY."""
    return _opening_range(df, "America/New_York", 570, 30, 750, 945, target_r=1.5)


def s_london_orb30(df):
    """London 30-minute opening-range breakout (8:00-8:30 London), entries until 11:00 London,
    stop at the other side of the range, target 2R, flat at 16:30 London."""
    return _opening_range(df, "Europe/London", 480, 30, 660, 990)


def s_asian_fade(df):
    """Asian-session range fade (forex): 22:00-06:00 UTC, a close outside the 20-bar Bollinger
    band (2 sd) that is not trending (ADX < 20) fades back to the middle band. Stop 1.5 ATR,
    target the middle band, flat at 07:00 UTC."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    hour = df.index.tz_convert("UTC").hour.to_numpy()
    mid = sma(c, 20)
    sd = pd.Series(c).rolling(20).std().to_numpy()
    calm = adx(h, l, c, 14) < 20
    asia = (hour >= 22) | (hour < 6)
    long_ = asia & calm & (c < mid - 2 * sd)
    short = asia & calm & (c > mid + 2 * sd)
    return long_, short, {"stop": ("atr", 1.5), "target": ("level", mid, mid),
                          "force_exit": (hour >= 7) & (hour < 22)}


def _vwap(df):
    """Session VWAP from 00:00 UTC, and the volume-weighted standard deviation around it."""
    h, l, c, v = (df[x].to_numpy(dtype=float) for x in "hlcv")
    tp = (h + l + c) / 3
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    v = np.where(v > 0, v, 1.0)
    cv = pd.Series(v).groupby(day).cumsum().to_numpy()
    cpv = pd.Series(tp * v).groupby(day).cumsum().to_numpy()
    cp2v = pd.Series(tp * tp * v).groupby(day).cumsum().to_numpy()
    vw = cpv / cv
    sd = np.sqrt(np.maximum(cp2v / cv - vw * vw, 0))
    return vw, sd


def s_vwap_revert(df):
    """VWAP 2-sd reversion: a close 2 standard deviations below the session VWAP with RSI(2) < 10
    buys (mirror above). Stop 1.5 ATR, target the VWAP, at most 24 bars."""
    c = df["c"].to_numpy()
    vw, sd = _vwap(df)
    r2 = rsi(c, 2)
    ok = sd > 0
    long_ = ok & (c < vw - 2 * sd) & (r2 < 10)
    short = ok & (c > vw + 2 * sd) & (r2 > 90)
    return long_, short, {"stop": ("atr", 1.5), "target": ("level", vw, vw), "max_bars": 24}


def s_vwap_pullback(df):
    """VWAP trend pullback: in an uptrend (close > EMA50 and EMA50 rising) the bar dips to the
    VWAP and closes back above it -> buy (mirror). Stop 1.5 ATR, target 2R."""
    l, h, c = (df[x].to_numpy() for x in "lhc")
    vw, _ = _vwap(df)
    e = ema(c, 50)
    rising = e > np.concatenate([[np.nan] * 5, e[:-5]])
    falling = e < np.concatenate([[np.nan] * 5, e[:-5]])
    long_ = rising & (c > e) & (l <= vw) & (c > vw)
    short = falling & (c < e) & (h >= vw) & (c < vw)
    return long_, short, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


def s_pdh_fakeout(df):
    """Previous-day high/low false breakout: price trades above yesterday's high but the bar
    closes back below it -> sell (mirror at the low), once per day each. Stop 1.5 ATR,
    target yesterday's midpoint."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    ph, pl, _ = _prev_day_hlc(df)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    short = _first((h > ph) & (c < ph), day)
    long_ = _first((l < pl) & (c > pl), day)
    mid = (ph + pl) / 2
    return long_, short & ~long_, {"stop": ("atr", 1.5), "target": ("level", mid, mid)}


def s_gap_fill(df):
    """Opening-gap fill (indices and stocks): the first bar of the UTC day opens more than
    0.5 ATR away from yesterday's close -> trade back toward that close. Stop 1 ATR, target
    yesterday's close, flat at the end of the day."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    _, _, pc = _prev_day_hlc(df)
    a = atr(h, l, c, 14)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    first_bar = np.concatenate([[False], day[1:] != day[:-1]])
    gap = o - pc
    long_ = first_bar & (gap < -0.5 * a) & (c < pc)
    short = first_bar & (gap > 0.5 * a) & (c > pc)
    last_bar = np.concatenate([day[1:] != day[:-1], [True]])
    return long_, short, {"stop": ("atr", 1.0), "target": ("level", pc, pc), "force_exit": last_bar}


def s_double7(df):
    """Connors 'Double 7s': above the 200 SMA, a close at a 7-bar low buys; exit at a close at a
    7-bar high (mirror for shorts below the 200 SMA). Stop 2 ATR as the risk unit."""
    c = df["c"].to_numpy()
    s200 = sma(c, 200)
    lo7 = pd.Series(c).rolling(7).min().to_numpy()
    hi7 = pd.Series(c).rolling(7).max().to_numpy()
    return (c > s200) & (c <= lo7), (c < s200) & (c >= hi7), {
        "stop": ("atr", 2.0), "target": None, "exit_long": c >= hi7, "exit_short": c <= lo7}


def s_cum_rsi(df):
    """Connors cumulative RSI: above the 200 SMA, RSI(2) of the last 2 bars summed < 35 buys;
    exit when RSI(2) > 65 (mirror: sum > 165 below the 200 SMA, exit < 35). Stop 2 ATR."""
    c = df["c"].to_numpy()
    r2 = rsi(c, 2)
    cum = r2 + np.concatenate([[np.nan], r2[:-1]])
    s200 = sma(c, 200)
    return (c > s200) & (cum < 35), (c < s200) & (cum > 165), {
        "stop": ("atr", 2.0), "target": None, "exit_long": r2 > 65, "exit_short": r2 < 35}


def s_holy_grail(df):
    """Raschke 'Holy Grail': ADX(14) > 30 and rising, price pulls back to the 20 EMA; the first
    close back above the previous bar's high buys (mirror). Stop at the pullback low, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    e = ema(c, 20)
    a = adx(h, l, c, 14)
    strong = (a > 30) & (a > np.concatenate([[np.nan] * 3, a[:-3]]))
    ph, pl = np.concatenate([[np.nan], h[:-1]]), np.concatenate([[np.nan], l[:-1]])
    touched_up = pd.Series(l <= e).rolling(3).max().to_numpy() > 0
    touched_dn = pd.Series(h >= e).rolling(3).max().to_numpy() > 0
    up_trend, dn_trend = c > ema(c, 50), c < ema(c, 50)
    low3 = pd.Series(l).rolling(3).min().to_numpy()
    high3 = pd.Series(h).rolling(3).max().to_numpy()
    long_ = strong & up_trend & touched_up & (c > ph)
    short = strong & dn_trend & touched_dn & (c < pl)
    return long_, short & ~long_, {"stop": ("level", low3, high3), "target": ("R", 2.0)}


def s_8020(df):
    """Raschke 80-20: the previous bar opened in the top 20% of its range and closed in the
    bottom 20%; this bar trades below that low and closes back above it -> buy (mirror).
    Stop at this bar's low, target 2R."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    rng = np.where(h - l > 0, h - l, np.nan)
    top_open = (o - l) / rng >= 0.8
    bot_close = (c - l) / rng <= 0.2
    bot_open = (o - l) / rng <= 0.2
    top_close = (c - l) / rng >= 0.8
    sh = lambda x: np.concatenate([[False if x.dtype == bool else np.nan], x[:-1]])  # noqa: E731
    pl, ph = sh(l), sh(h)
    long_ = sh(top_open & bot_close) & (l < pl) & (c > pl)
    short = sh(bot_open & top_close) & (h > ph) & (c < ph)
    return long_, short & ~long_, {"stop": ("level", l, h), "target": ("R", 2.0)}


def s_turtle_soup(df):
    """Raschke 'Turtle Soup': a new 20-bar low whose previous 20-bar low is at least 4 bars old,
    then a close back above that previous low -> buy (mirror). Stop at the new low, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    prev_lo = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    prev_hi = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    age_lo = pd.Series(l).rolling(20).apply(np.argmin, raw=True).shift(1).to_numpy()
    age_hi = pd.Series(h).rolling(20).apply(np.argmax, raw=True).shift(1).to_numpy()
    old_lo, old_hi = age_lo <= 15, age_hi <= 15            # index within the window: <=15 -> 4+ bars old
    long_ = old_lo & (l < prev_lo) & (c > prev_lo)
    short = old_hi & (h > prev_hi) & (c < prev_hi)
    return long_, short & ~long_, {"stop": ("level", l, h), "target": ("R", 2.0)}


def s_vol_contraction(df):
    """Volatility contraction breakout (Crabel-style): ATR(5) below 60% of ATR(50), then a close
    above the 10-bar high buys / below the 10-bar low sells. Stop 1.5 ATR(14), target 2.5R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    quiet = atr(h, l, c, 5) < 0.6 * atr(h, l, c, 50)
    q = np.concatenate([[False], quiet[:-1]])
    hh = pd.Series(h).rolling(10).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(10).min().shift(1).to_numpy()
    return q & (c > hh), q & (c < ll), {"stop": ("atr", 1.5), "target": ("R", 2.5)}


def s_three_lower_highs(df):
    """Connors 3-bar pullback: above the 200 SMA, three lower highs and lower lows in a row
    buys; exit on a close above the previous bar's high (mirror). Stop 2 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    s200 = sma(c, 200)
    lh = (h < np.roll(h, 1)) & (l < np.roll(l, 1))
    hl = (h > np.roll(h, 1)) & (l > np.roll(l, 1))
    three_dn = pd.Series(lh).rolling(3).sum().to_numpy() == 3
    three_up = pd.Series(hl).rolling(3).sum().to_numpy() == 3
    ph, pl = np.concatenate([[np.nan], h[:-1]]), np.concatenate([[np.nan], l[:-1]])
    return (c > s200) & three_dn, (c < s200) & three_up, {
        "stop": ("atr", 2.0), "target": None, "exit_long": c > ph, "exit_short": c < pl}


SESSION = {
    "SS01": ("NY 15-min opening-range breakout", s_ny_orb15),
    "SS02": ("NY 30-min opening-range breakout", s_ny_orb30),
    "SS03": ("London 30-min opening-range breakout", s_london_orb30),
    "SS04": ("Asian-session Bollinger fade", s_asian_fade),
    "SS05": ("VWAP 2-sd reversion + RSI(2)", s_vwap_revert),
    "SS06": ("VWAP trend pullback", s_vwap_pullback),
    "SS07": ("Previous-day high/low false breakout", s_pdh_fakeout),
    "SS08": ("Opening-gap fill", s_gap_fill),
    "SS09": ("Connors Double 7s", s_double7),
    "SS10": ("Connors cumulative RSI(2)", s_cum_rsi),
    "SS11": ("Raschke Holy Grail (ADX + EMA20 pullback)", s_holy_grail),
    "SS12": ("Raschke 80-20 reversal", s_8020),
    "SS13": ("Raschke Turtle Soup", s_turtle_soup),
    "SS14": ("Volatility contraction breakout", s_vol_contraction),
    "SS15": ("Connors 3-bar pullback", s_three_lower_highs),
}
