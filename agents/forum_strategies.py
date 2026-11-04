"""Strategies posted by traders with exact rules: Reddit r/algotrading (FR01-FR06) and trading forums
(BabyPips, Forex Factory, ProRealCode: FR07-FR13). Same contract as agents/rules_agent.py.

Translations (ours, so the same rule works on every FTMO instrument): fixed pip distances become ATR
multiples of the bar's timeframe (FX-typical sizes); buy-stop / sell-stop entries become "the first bar
that closes beyond the level, filled at the next open"; where the post has no stop, a 3 ATR stop (or
the noted level) is ours. Three entries are posted as FAILED by their authors (FR06, FR12, FR13).
"""
import numpy as np
import pandas as pd

from agents.book_strategies import _daily_signals, _days, _on_bars
from agents.rules_agent import atr, cross_down, cross_up, ema, macd, rsi, sma

NY, LDN, CET = "America/New_York", "Europe/London", "Europe/Paris"


def _s(x):
    return pd.Series(np.asarray(x, dtype=float))


def _first_of_day(m, date):
    m = np.asarray(m, bool)
    return m & (pd.Series(m).groupby(date).cumsum().to_numpy() == 1)


# ---------------- FR01 IBS + band below the 10-day high (r/algotrading) ----------------
def s_ibs_band(df):
    """Daily: close below (10-day highest high - 2.5 x 25-day average range) and IBS < 0.3 -> long next
    day; out the day after a close above the previous day's high. Stop 3 ATR (ours)."""
    _, day, _, _ = _days(df)
    rng = day["h"] - day["l"]
    band = day["h"].rolling(10).max() - 2.5 * rng.rolling(25).mean()
    ibs = (day["c"] - day["l"]) / rng.replace(0, np.nan)
    lo = (day["c"] < band) & (ibs < 0.3)
    ex = day["c"] > day["h"].shift(1)
    f = pd.Series(False, index=day.index)
    return _daily_signals(df, lo, f, ex, f)


# ---------------- FR02 RSI(5) above 70 momentum hold (r/algotrading) ----------------
def s_rsi5_momentum(df):
    """RSI(5) crosses above 70 -> long, held while it stays above 70 (out on the first close under 70).
    Long only. Stop 3 ATR (ours)."""
    c = df["c"].to_numpy()
    r = rsi(c, 5)
    return cross_up(r, 70), np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": r < 70}


# ---------------- FR03 / FR04 Russ_CW reversal day trades (r/algotrading) ----------------
def _next_day_breakout(df, setup_day, level_day):
    date, day, first, last = _days(df)
    c = df["c"].to_numpy()
    arm = _on_bars(date, setup_day.shift(1).astype(float)) > 0
    lvl = _on_bars(date, level_day.shift(1))
    hit = arm & (c > lvl)
    lo = _first_of_day(hit, date) & ~last
    return lo, np.zeros(len(c), bool), {"stop": ("atr", 2.0), "target": None, "force_exit": last}


def s_lower_day_reversal(df):
    """A day with a lower high AND a lower low than the day before -> the next day, the first close
    above that day's high buys; out at that day's close. Stop 2 ATR (ours)."""
    _, day, _, _ = _days(df)
    setup = (day["h"] < day["h"].shift(1)) & (day["l"] < day["l"].shift(1))
    return _next_day_breakout(df, setup, day["h"])


def s_four_lower_highs(df):
    """Four lower daily highs in a row -> the next day, the first close above the last high buys; out at
    that day's close. Stop 2 ATR (ours)."""
    _, day, _, _ = _days(df)
    lh = day["h"] < day["h"].shift(1)
    setup = lh.rolling(4).sum() == 4
    return _next_day_breakout(df, setup, day["h"])


# ---------------- FR05 10-bar range breakout, counter-trend (r/algotrading) ----------------
def s_range10_countertrend(df):
    """Hourly: the range of the last 10 completed bars; 08:00-14:00 New York, a close above its high
    while below the EMA200 buys (counter-trend), a close below its low while above the EMA200 sells.
    Stop at the other end of the range, target 1.5 x the range height."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    hi, lo = _s(h).rolling(10).max().shift(1).to_numpy(), _s(l).rolling(10).min().shift(1).to_numpy()
    e = ema(c, 200)
    hr = df.index.tz_convert(NY).hour.to_numpy()
    win = (hr >= 8) & (hr < 14)
    long_ = win & (c > hi) & (c < e)
    short = win & (c < lo) & (c > e)
    ht = hi - lo
    return long_, short & ~long_, {"stop": ("level", lo, hi), "target": ("level", hi + 1.5 * ht, lo - 1.5 * ht)}


# ---------------- FR06 Triple Supertrend + StochRSI + EMA200 (YouTube, FAILED on reddit) ----------------
def _st_line(h, l, c, period, mult):
    a = atr(h, l, c, period)
    mid = (h + l) / 2
    up, dn = mid - mult * a, mid + mult * a
    fu, fd = up.copy(), dn.copy()
    tr = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        tr[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else tr[i - 1]
    return tr, np.where(tr > 0, fu, fd)


def s_triple_st_stochrsi(df):
    """Close above the EMA200, Stochastic RSI %K crosses up out of 20 and at least 2 of 3 Supertrends
    (10x1, 11x2, 12x3) up -> long; mirror short. Stop at the middle Supertrend, target 1.5R. (A reddit
    rebuild lost 23% over 16 years on EURUSD 1H.)"""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    r = _s(rsi(c, 14))
    st = (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min())
    k = (st.rolling(3).mean() * 100).to_numpy()
    t1, _ = _st_line(h, l, c, 10, 1)
    t2, line2 = _st_line(h, l, c, 11, 2)
    t3, _ = _st_line(h, l, c, 12, 3)
    votes = t1 + t2 + t3
    e = ema(c, 200)
    long_ = (c > e) & cross_up(k, 20) & (votes >= 1)
    short = (c < e) & cross_down(k, 80) & (votes <= -1)
    return long_, short & ~long_, {"stop": ("level", line2, line2), "target": ("R", 1.5)}


# ---------------- FR07 3 Ducks (BabyPips) ----------------
def s_three_ducks(df):
    """3 Ducks: the 4-hour and 1-hour closes above their SMA60 and the 15-minute close crossing above its
    SMA60 -> long; mirror short. Out when the 1-hour close crosses back. 30/30 pips -> stop 3 ATR,
    target 1R."""
    c = df["c"]
    step = df.index[1] - df.index[0]

    def htf(rule, minutes):
        b = c.resample(rule, label="left", closed="left").last().dropna()
        above = pd.Series((b > b.rolling(60).mean()).to_numpy(), index=b.index + pd.Timedelta(minutes=minutes))
        below = pd.Series((b < b.rolling(60).mean()).to_numpy(), index=b.index + pd.Timedelta(minutes=minutes))
        return (above.reindex(df.index + step, method="ffill").fillna(False).to_numpy(),
                below.reindex(df.index + step, method="ffill").fillna(False).to_numpy())
    a4, b4 = htf("4h", 240)
    a1, b1 = htf("1h", 60)
    cc = c.to_numpy()
    s60 = sma(cc, 60)
    long_ = a4 & a1 & cross_up(cc, s60)
    short = b4 & b1 & cross_down(cc, s60)
    return long_, short & ~long_, {"stop": ("atr", 3.0), "target": ("R", 1.0), "exit_long": b1, "exit_short": a1}


# ---------------- FR08 Kathy Lien 5-minute Momo (BabyPips) ----------------
def s_momo(df):
    """5-minute Momo: price was under the EMA20; a close above EMA20 + 1 ATR (her 10 pips) with the MACD
    histogram turned positive within the last 5 bars -> long; mirror short. Stop 2 ATR (her 20 pips),
    trail out on a close back under EMA20 - 1.5 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    e = ema(c, 20)
    m, sg, _ = macd(c, 12, 26, 9)
    hist = m - sg
    turned_up = _s(cross_up(hist, 0).astype(float)).rolling(5).max().to_numpy() > 0
    turned_dn = _s(cross_down(hist, 0).astype(float)).rolling(5).max().to_numpy() > 0
    was_below = _s((c < e).astype(float)).shift(1).rolling(10).max().to_numpy() > 0
    was_above = _s((c > e).astype(float)).shift(1).rolling(10).max().to_numpy() > 0
    up = cross_up(c, e + a) & turned_up & was_below
    dn = cross_down(c, e - a) & turned_dn & was_above
    return up, dn & ~up, {"stop": ("atr", 2.0), "target": None,
                          "exit_long": c < e - 1.5 * a, "exit_short": c > e + 1.5 * a}


# ---------------- FR09 NickB 4H swing-point break (BabyPips) ----------------
def s_swing_break(df):
    """4-hour swing points (a high above the 4 bars on each side, confirmed 4 bars later): a close above
    the latest swing high buys, below the latest swing low sells. 50/50 pips -> stop 1 ATR, target 1R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    sh, sl = np.full(n, np.nan), np.full(n, np.nan)
    lh, ll = np.nan, np.nan
    for i in range(8, n):
        j = i - 4
        if h[j] == h[j - 4:j + 5].max():
            lh = h[j]
        if l[j] == l[j - 4:j + 5].min():
            ll = l[j]
        sh[i], sl[i] = lh, ll
    up = cross_up(c, sh)
    dn = cross_down(c, sl)
    return up, dn & ~up, {"stop": ("atr", 1.0), "target": ("R", 1.0)}


# ---------------- FR10 Vegas tunnel (Forex Factory) ----------------
def s_vegas_tunnel(df):
    """Vegas tunnel: the hourly close crosses above the EMA144/169 tunnel with EMA12 also above it ->
    long; mirror short; no new entries 17:00-24:00 New York. Stop the other side of the tunnel, target
    2R (his 89 pips single-position version), reverses on the opposite signal."""
    c = df["c"].to_numpy()
    e1, e2, e12 = ema(c, 144), ema(c, 169), ema(c, 12)
    up_t, lo_t = np.maximum(e1, e2), np.minimum(e1, e2)
    hr = df.index.tz_convert(NY).hour.to_numpy()
    ok = hr < 17
    long_ = ok & cross_up(c, up_t) & (e12 > up_t)
    short = ok & cross_down(c, lo_t) & (e12 < lo_t)
    return long_, short & ~long_, {"stop": ("level", lo_t, up_t), "target": ("R", 2.0), "opposite": True}


# ---------------- FR11 Raschke "Anti" (Forex Factory) ----------------
def s_raschke_anti(df):
    """Raschke's Anti (stochastic 7-4-10): %D rising 4 bars, %K fell 3 bars toward it and now turns up
    -> long at the next open; mirror short. Stop the 5-bar low/high, out after 4 bars."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    hh, ll = _s(h).rolling(7).max().to_numpy(), _s(l).rolling(7).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100 * (c - ll) / (hh - ll)
    k = sma(raw, 4)
    d = sma(k, 10)
    dk, dd = np.diff(k, prepend=np.nan), np.diff(d, prepend=np.nan)
    d_up = _s((dd > 0).astype(float)).rolling(4).sum().to_numpy() == 4
    d_dn = _s((dd < 0).astype(float)).rolling(4).sum().to_numpy() == 4
    k_fell = _s((dk < 0).astype(float)).shift(1).rolling(3).sum().to_numpy() == 3
    k_rose = _s((dk > 0).astype(float)).shift(1).rolling(3).sum().to_numpy() == 3
    long_ = d_up & k_fell & (dk > 0)
    short = d_dn & k_rose & (dk < 0)
    return long_, short & ~long_, {"stop": ("lowest", 5), "target": None, "max_bars": 4}


# ---------------- FR12 M15 LowBuyHighSell (ProRealCode, FAILED) ----------------
def s_low_buy_high_sell(df):
    """15-minute, 08:00-21:00 Paris time: a close under the lowest low of the last 3 days buys (once a
    day), out on a close above yesterday's close; a close under the highest high of 13 days sells, out
    on a close under yesterday's close. Stops 0.4 / 1.0 x yesterday's range, flat at 22:00. (Users
    found it lost money 2020-2025.)"""
    date, day, first, last = _days(df)
    c = df["c"].to_numpy()
    t = df.index.tz_convert(CET)
    hr = t.hour.to_numpy()
    win = (hr >= 8) & (hr < 21)
    low3 = _on_bars(date, day["l"].rolling(3).min().shift(1))
    high13 = _on_bars(date, day["h"].rolling(13).max().shift(1))
    pc = _on_bars(date, day["c"].shift(1))
    pr = _on_bars(date, (day["h"] - day["l"]).shift(1))
    long_ = _first_of_day(win & (c < low3), date)
    short = _first_of_day(win & cross_down(c, high13), date)
    return long_, short & ~long_, {"stop": ("level", c - 0.4 * pr, c + 1.0 * pr), "target": None,
                                   "exit_long": c > pc, "exit_short": c < pc, "force_exit": hr == 21}


# ---------------- FR13 Hourly Camarilla breakout (BabyPips, FAILED) ----------------
def s_camarilla_hourly(df):
    """Camarilla levels from the previous hour: 07:00-16:00 London, the first 5-minute close above H2
    in the hour buys (stop L5, target H4); below L2 sells (stop H5, target L4). (A commenter reported
    very bad results.)"""
    h1 = df.resample("1h", label="left", closed="left").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    H, L, C = h1["h"], h1["l"], h1["c"]
    rg = H - L
    lv = pd.DataFrame({"H2": C + rg * 1.1 / 6, "H4": C + rg * 1.1 / 2, "H5": H / L * C,
                       "L2": C - rg * 1.1 / 6, "L4": C - rg * 1.1 / 2})
    lv["L5"] = C - (lv["H5"] - C)
    key = df.index.floor("1h") - pd.Timedelta(hours=1)
    g = lv.reindex(key)
    c = df["c"].to_numpy()
    hr = df.index.tz_convert(LDN).hour.to_numpy()
    win = (hr >= 7) & (hr < 16)
    hour_id = np.asarray(df.index.floor("1h").astype("int64"))
    up = win & (c > g["H2"].to_numpy())
    dn = win & (c < g["L2"].to_numpy())
    first_up = up & (pd.Series(up).groupby(hour_id).cumsum().to_numpy() == 1)
    first_dn = dn & (pd.Series(dn).groupby(hour_id).cumsum().to_numpy() == 1)
    return first_up, first_dn & ~first_up, {"stop": ("level", g["L5"].to_numpy(), g["H5"].to_numpy()),
                                            "target": ("level", g["H4"].to_numpy(), g["L4"].to_numpy())}


# id: (name, fn, timeframes)
FORUMS = {
    "FR01": ("IBS < 0.3 under the 10-day-high band, daily (r/algotrading)", s_ibs_band, ["1Hour"]),
    "FR02": ("RSI(5) above 70 momentum hold (r/algotrading)", s_rsi5_momentum, ["4Hour"]),
    "FR03": ("Lower-high lower-low day, next-day breakout, out at close (r/algotrading)", s_lower_day_reversal, ["1Hour", "15Min"]),
    "FR04": ("Four lower highs, next-day breakout, out at close (r/algotrading)", s_four_lower_highs, ["1Hour", "15Min"]),
    "FR05": ("10-hour range breakout against the EMA200, 1.5x range (r/algotrading)", s_range10_countertrend, ["1Hour"]),
    "FR06": ("Triple Supertrend + StochRSI + EMA200, 1.5R (YouTube; failed on reddit)", s_triple_st_stochrsi, ["1Hour"]),
    "FR07": ("3 Ducks: SMA60 on 4H, 1H and 15m (BabyPips)", s_three_ducks, ["15Min"]),
    "FR08": ("Kathy Lien 5-minute Momo: EMA20 + MACD flip (BabyPips)", s_momo, ["5Min"]),
    "FR09": ("NickB 4H swing-point break (BabyPips)", s_swing_break, ["4Hour"]),
    "FR10": ("Vegas tunnel EMA144/169 + EMA12 (Forex Factory)", s_vegas_tunnel, ["1Hour"]),
    "FR11": ("Raschke Anti stochastic hook, 4-bar exit (Forex Factory)", s_raschke_anti, ["1Hour"]),
    "FR12": ("M15 LowBuyHighSell (ProRealCode; reported failing)", s_low_buy_high_sell, ["15Min"]),
    "FR13": ("Hourly Camarilla H2/L2 breakout (BabyPips; reported failing)", s_camarilla_hourly, ["5Min"]),
}
