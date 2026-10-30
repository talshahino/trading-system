"""Classic, well-documented systems (no social-media source), coded from their textbook rules.
Same contract as agents/rules_agent.py. Higher timeframes only (1H, 4H): that is where the only
strategy that passed the FTMO bar so far (TV04) lives.
"""
import numpy as np
import pandas as pd

from agents.rules_agent import atr, ema, rsi, sma

HTF = ["1Hour", "4Hour"]


def _hh(x, n):
    return pd.Series(x).rolling(n).max().shift(1).to_numpy()


def _ll(x, n):
    return pd.Series(x).rolling(n).min().shift(1).to_numpy()


def s_turtle_20(df):
    """Turtle System 1 (Dennis/Faith): close above the 20-bar high buys, below the 20-bar low
    sells. Stop 2N (N = ATR20). Exit on the opposite 10-bar breakout. No target."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    return (c > _hh(h, 20)), (c < _ll(l, 20)), {"stop": ("atr", 2.0), "target": None,
                                                  "exit_long": c < _ll(l, 10), "exit_short": c > _hh(h, 10)}


def s_turtle_55(df):
    """Turtle System 2: 55-bar breakout entries, stop 2N, exit on the opposite 20-bar breakout."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    return (c > _hh(h, 55)), (c < _ll(l, 55)), {"stop": ("atr", 2.0), "target": None,
                                                  "exit_long": c < _ll(l, 20), "exit_short": c > _hh(h, 20)}


def s_keltner_breakout(df):
    """Keltner channel breakout (Chester Keltner / Linda Raschke version): close above
    EMA20 + 2 x ATR10 buys, below EMA20 - 2 x ATR10 sells. Stop 2 ATR, ATR trailing stop 3 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    mid, a = ema(c, 20), atr(h, l, c, 10)
    return c > mid + 2 * a, c < mid - 2 * a, {"stop": ("atr", 2.0), "target": None, "trail_atr": 3.0}


def s_golden_cross(df):
    """50/200 moving-average trend (golden/death cross): SMA50 crossing above SMA200 buys,
    below sells. Stop 3 ATR, ATR trailing stop 3 ATR, reverse on the opposite cross."""
    c = df["c"].to_numpy()
    f, s = sma(c, 50), sma(c, 200)
    up = np.zeros(len(c), bool)
    dn = np.zeros(len(c), bool)
    up[1:] = (f[1:] > s[1:]) & (f[:-1] <= s[:-1])
    dn[1:] = (f[1:] < s[1:]) & (f[:-1] >= s[:-1])
    return up, dn, {"stop": ("atr", 3.0), "target": None, "trail_atr": 3.0, "opposite": True}


def s_bb_squeeze(df):
    """Bollinger squeeze breakout (John Bollinger): band width (20, 2) at its lowest of the last
    120 bars marks a squeeze; within the next 10 bars a close outside the band enters in that
    direction. Stop 2 ATR, ATR trailing stop 2.5 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    mid = sma(c, 20)
    sd = pd.Series(c).rolling(20).std().to_numpy()
    width = 4 * sd / mid
    squeeze = width <= pd.Series(width).rolling(120).min().to_numpy()
    recent = pd.Series(squeeze).rolling(10).max().fillna(0).to_numpy().astype(bool)
    return recent & (c > mid + 2 * sd), recent & (c < mid - 2 * sd), {"stop": ("atr", 2.0), "target": None,
                                                                      "trail_atr": 2.5}


def s_inside_bar(df):
    """Inside-bar breakout: bar i-1 is inside bar i-2 (the mother bar); a close above the mother
    high buys, below the mother low sells. Stop at the other side of the mother bar, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    mh, ml = np.full(n, np.nan), np.full(n, np.nan)
    mh[2:], ml[2:] = h[:-2], l[:-2]
    inside = np.zeros(n, bool)
    inside[2:] = (h[1:-1] < h[:-2]) & (l[1:-1] > l[:-2])
    long_, short = inside & (c > mh), inside & (c < ml)
    return long_, short & ~long_, {"stop": ("level", ml, mh), "target": ("R", 2.0)}


def s_rsi_trend_pullback(df):
    """Trend pullback with RSI (Andrew Cardwell style): uptrend when EMA50 > EMA200; RSI(14)
    dips under 40 and crosses back above 40 -> buy (mirror at 60 in downtrends). Stop 2 ATR,
    target 3R."""
    c = df["c"].to_numpy()
    e50, e200, r = ema(c, 50), ema(c, 200), rsi(c, 14)
    up = np.zeros(len(c), bool)
    dn = np.zeros(len(c), bool)
    up[1:] = (e50[1:] > e200[1:]) & (r[1:] > 40) & (r[:-1] <= 40)
    dn[1:] = (e50[1:] < e200[1:]) & (r[1:] < 60) & (r[:-1] >= 60)
    return up, dn, {"stop": ("atr", 2.0), "target": ("R", 3.0)}


CLASSIC = {
    "TV13": ("Turtle System 1 (20/10 breakout)", s_turtle_20, HTF),
    "TV14": ("Turtle System 2 (55/20 breakout)", s_turtle_55, HTF),
    "TV15": ("Keltner channel breakout + ATR trail", s_keltner_breakout, HTF),
    "TV16": ("Golden cross 50/200 + ATR trail", s_golden_cross, HTF),
    "TV17": ("Bollinger squeeze breakout", s_bb_squeeze, HTF),
    "TV18": ("Inside-bar breakout, 2R", s_inside_bar, HTF),
    "TV19": ("RSI trend pullback (EMA50/200), 3R", s_rsi_trend_pullback, HTF),
}


# ---------------- batch 2: more classic systems ----------------
def _cross(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    up = np.zeros(len(a), bool)
    dn = np.zeros(len(a), bool)
    up[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    dn[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return up, dn


def s_ichimoku(df):
    """Ichimoku (Hosoda, 9/26/52): Tenkan crosses above Kijun while price is above the cloud ->
    buy (mirror below). Stop at Kijun (assumed via level), ATR trail 3, exit when price closes
    back inside/through the cloud."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    mid = lambda n: ((pd.Series(h).rolling(n).max() + pd.Series(l).rolling(n).min()) / 2).to_numpy()
    tk, kj = mid(9), mid(26)
    sa = pd.Series((tk + kj) / 2).shift(26).to_numpy()
    sb = pd.Series(mid(52)).shift(26).to_numpy()
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    up, dn = _cross(tk, kj)
    return up & (c > top), dn & (c < bot), {"stop": ("level", kj, kj), "target": None, "trail_atr": 3.0,
                                            "exit_long": c < bot, "exit_short": c > top}


def _psar(h, l, step=0.02, mx=0.2):
    n = len(h)
    sar = np.zeros(n)
    trend, af, ep = 1, step, h[0]
    sar[0] = l[0]
    for i in range(1, n):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if trend == 1:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                trend, s, ep, af = -1, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                trend, s, ep, af = 1, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, mx)
        sar[i] = s
    return sar


def s_psar_ema200(df):
    """Parabolic SAR (Wilder, 0.02/0.2) with EMA200 filter: SAR flips below price while price is
    above EMA200 -> buy (mirror). Stop trails on the SAR line."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    sar, e = _psar(h, l), ema(c, 200)
    below = c > sar
    up = np.zeros(len(c), bool)
    dn = np.zeros(len(c), bool)
    up[1:] = below[1:] & ~below[:-1]
    dn[1:] = ~below[1:] & below[:-1]
    return up & (c > e), dn & (c < e), {"stop": ("trail_line",), "line": sar, "target": None}


def s_macd_zero(df):
    """MACD (12/26/9) zero-line trend: MACD line crosses above zero with price above EMA200 ->
    buy (mirror). Stop 2 ATR, target 2R."""
    from agents.rules_agent import macd
    c = df["c"].to_numpy()
    m, _, _ = macd(c)
    up, dn = _cross(m, np.zeros(len(c)))
    e = ema(c, 200)
    return up & (c > e), dn & (c < e), {"stop": ("atr", 2.0), "target": ("R", 2.0)}


def s_triple_ema_pullback(df):
    """Triple EMA 8/21/55 stack: EMAs aligned (8>21>55), price dips to the 21 EMA and closes back
    above the 8 EMA -> buy (mirror). Stop below the 55 EMA (level), target 2R."""
    l, h, c = df["l"].to_numpy(), df["h"].to_numpy(), df["c"].to_numpy()
    e8, e21, e55 = ema(c, 8), ema(c, 21), ema(c, 55)
    upst, dnst = (e8 > e21) & (e21 > e55), (e8 < e21) & (e21 < e55)
    tl = pd.Series(l <= e21).rolling(3).max().fillna(0).to_numpy().astype(bool)
    th = pd.Series(h >= e21).rolling(3).max().fillna(0).to_numpy().astype(bool)
    return upst & tl & (c > e8), dnst & th & (c < e8), {"stop": ("level", e55, e55), "target": ("R", 2.0)}


def s_adx_di(df):
    """ADX/DI (Wilder): +DI crosses above -DI with ADX(14) > 25 -> buy (mirror). Stop 2 ATR,
    ATR trail 3."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    from agents.rules_agent import rma, adx
    up_m = np.diff(h, prepend=h[0])
    dn_m = -np.diff(l, prepend=l[0])
    pdm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    ndm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    a = atr(h, l, c, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, ndi = 100 * rma(pdm, 14) / a, 100 * rma(ndm, 14) / a
    strong = adx(h, l, c, 14) > 25
    up, dn = _cross(pdi, ndi)
    return up & strong, dn & strong, {"stop": ("atr", 2.0), "target": None, "trail_atr": 3.0}


def s_heikin_ashi(df):
    """Heikin-Ashi trend: first green HA candle with no lower wick, above EMA50 -> buy (mirror:
    red, no upper wick, below EMA50). Exit on the first opposite-colour HA candle. Stop 2 ATR."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    hc = (o + h + l + c) / 4
    ho = np.zeros(len(c))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    hl, hh = np.minimum(l, np.minimum(ho, hc)), np.maximum(h, np.maximum(ho, hc))
    green, red = hc > ho, hc < ho
    e = ema(c, 50)
    first_g = np.r_[False, green[1:] & ~green[:-1]]
    first_r = np.r_[False, red[1:] & ~red[:-1]]
    nolow = np.isclose(hl, np.minimum(ho, hc))
    noup = np.isclose(hh, np.maximum(ho, hc))
    return first_g & nolow & (c > e), first_r & noup & (c < e), {"stop": ("atr", 2.0), "target": None,
                                                                 "exit_long": red, "exit_short": green}


def s_nr7(df):
    """NR7 breakout (Toby Crabel): the bar with the narrowest range of the last 7; a close above
    its high on the next bar buys, below its low sells. Stop at the other side of the NR7 bar,
    target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    rng = h - l
    nr7 = rng <= pd.Series(rng).rolling(7).min().to_numpy()
    n = len(c)
    ph, pl, prev = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    ph[1:], pl[1:], prev[1:] = h[:-1], l[:-1], nr7[:-1]
    long_, short = prev & (c > ph), prev & (c < pl)
    return long_, short & ~long_, {"stop": ("level", pl, ph), "target": ("R", 2.0)}


def s_prev_day_breakout(df):
    """Previous-day high/low breakout: first close above yesterday's high buys, below
    yesterday's low sells (once per day each). Stop at yesterday's midpoint, target 2R."""
    from agents.rules_agent import _prev_day_hlc, _utc
    h, l, c = (df[x].to_numpy() for x in "hlc")
    ph, pl, _ = _prev_day_hlc(df)
    _, day = _utc(df)
    up, dn = c > ph, c < pl
    fu = pd.Series(up).groupby(day).cumsum().to_numpy() == 1
    fd = pd.Series(dn).groupby(day).cumsum().to_numpy() == 1
    mid = (ph + pl) / 2
    return up & fu, dn & fd & ~(up & fu), {"stop": ("level", mid, mid), "target": ("R", 2.0)}


def s_stoch_pullback(df):
    """Stochastic pullback in trend (Lane, 14/3/3): uptrend (close > EMA200), %K crosses above %D
    below 20 -> buy (mirror above 80 in downtrend). Stop 1.5 ATR, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    lo, hi = pd.Series(l).rolling(14).min(), pd.Series(h).rolling(14).max()
    k = (100 * (pd.Series(c) - lo) / (hi - lo)).rolling(3).mean().to_numpy()
    d = pd.Series(k).rolling(3).mean().to_numpy()
    up, dn = _cross(k, d)
    e = ema(c, 200)
    return up & (k < 20) & (c > e), dn & (k > 80) & (c < e), {"stop": ("atr", 1.5), "target": ("R", 2.0)}


def s_zscore_mr(df):
    """Z-score mean reversion: z = (close - SMA20) / stdev20. z < -2 buys, z > 2 sells; exit
    back at the mean (z crosses 0). Stop 2.5 ATR."""
    c = df["c"].to_numpy()
    m = sma(c, 20)
    sd = pd.Series(c).rolling(20).std().to_numpy()
    z = (c - m) / sd
    return z < -2, z > 2, {"stop": ("atr", 2.5), "target": None, "exit_long": z > 0, "exit_short": z < 0}


def s_ibs(df):
    """Internal Bar Strength (index mean reversion): IBS = (close - low) / (high - low).
    IBS < 0.2 with close above SMA200 buys; exit when IBS > 0.8 or at a 2.5 ATR stop. Long only."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    with np.errstate(divide="ignore", invalid="ignore"):
        ibs = (c - l) / (h - l)
    return (ibs < 0.2) & (c > sma(c, 200)), np.zeros(len(c), bool), {"stop": ("atr", 2.5), "target": None,
                                                                      "exit_long": ibs > 0.8}


def s_engulfing_swing(df):
    """Engulfing at a swing extreme: bullish engulfing whose low is the lowest of 20 bars ->
    buy (mirror at a 20-bar high). Stop beyond the pattern, target 2R."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    n = len(c)
    po, pc = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]]
    bull = (pc < po) & (c > o) & (c >= po) & (o <= pc)
    bear = (pc > po) & (c < o) & (c <= po) & (o >= pc)
    ll20 = pd.Series(l).rolling(20).min().to_numpy()
    hh20 = pd.Series(h).rolling(20).max().to_numpy()
    lo2 = np.fmin(l, np.r_[np.nan, l[:-1]])
    hi2 = np.fmax(h, np.r_[np.nan, h[:-1]])
    return bull & (lo2 <= ll20), bear & (hi2 >= hh20), {"stop": ("level", lo2, hi2), "target": ("R", 2.0)}


CLASSIC.update({
    "TV20": ("Ichimoku TK cross above/below cloud", s_ichimoku, HTF),
    "TV21": ("Parabolic SAR + EMA200", s_psar_ema200, HTF),
    "TV22": ("MACD zero-line cross + EMA200, 2R", s_macd_zero, HTF),
    "TV23": ("Triple EMA 8/21/55 pullback, 2R", s_triple_ema_pullback, HTF),
    "TV24": ("ADX>25 DI cross + ATR trail", s_adx_di, HTF),
    "TV25": ("Heikin-Ashi trend + EMA50", s_heikin_ashi, HTF),
    "TV26": ("NR7 breakout, 2R", s_nr7, HTF),
    "TV27": ("Previous-day high/low breakout, 2R", s_prev_day_breakout, ["15Min", "1Hour"]),
    "TV28": ("Stochastic pullback in trend, 2R", s_stoch_pullback, HTF),
    "TV29": ("Z-score mean reversion", s_zscore_mr, HTF),
    "TV30": ("IBS mean reversion (long only)", s_ibs, HTF),
    "TV31": ("Engulfing at 20-bar extreme, 2R", s_engulfing_swing, HTF),
})


# ---------------- batch 3 ----------------
def s_weekly_breakout(df):
    """Previous-week high/low breakout: first close above last week's high buys, below last
    week's low sells (once per week each). Stop at last week's midpoint, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    wk = df.index.tz_convert("UTC").tz_localize(None).to_period("W").astype(str).to_numpy()
    g = pd.DataFrame({"w": wk, "h": h, "l": l})
    prev = g.groupby("w", sort=False).agg(h=("h", "max"), l=("l", "min")).shift(1)
    m = g[["w"]].join(prev, on="w")
    ph, pl = m["h"].to_numpy(dtype=float), m["l"].to_numpy(dtype=float)
    up, dn = c > ph, c < pl
    fu = pd.Series(up).groupby(wk).cumsum().to_numpy() == 1
    fd = pd.Series(dn).groupby(wk).cumsum().to_numpy() == 1
    mid = (ph + pl) / 2
    return up & fu, dn & fd & ~(up & fu), {"stop": ("level", mid, mid), "target": ("R", 2.0)}


def s_keltner_fade(df):
    """Keltner fade: a close back inside EMA20 +/- 2.5 ATR after closing outside -> fade toward
    the EMA (exit at the EMA). Stop 1.5 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    mid, a = ema(c, 20), atr(h, l, c, 10)
    up_b, lo_b = mid + 2.5 * a, mid - 2.5 * a
    was_lo = np.r_[False, (c < lo_b)[:-1]]
    was_hi = np.r_[False, (c > up_b)[:-1]]
    return was_lo & (c > lo_b), was_hi & (c < up_b), {"stop": ("atr", 1.5), "target": None,
                                                      "exit_long": c >= mid, "exit_short": c <= mid}


def s_hull_turn(df):
    """Hull moving average (Alan Hull, 55) turning up with price above EMA200 -> buy (mirror).
    Stop 2 ATR, target 2R."""
    c = df["c"].to_numpy()
    def wma(x, n):
        w = np.arange(1, n + 1)
        return pd.Series(x).rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True).to_numpy()
    n = 55
    hull = wma(2 * wma(c, n // 2) - wma(c, n), int(np.sqrt(n)))
    slope = np.diff(hull, prepend=np.nan)
    up = np.r_[False, (slope[1:] > 0) & (slope[:-1] <= 0)]
    dn = np.r_[False, (slope[1:] < 0) & (slope[:-1] >= 0)]
    e = ema(c, 200)
    return up & (c > e), dn & (c < e), {"stop": ("atr", 2.0), "target": ("R", 2.0)}


def s_willr_pullback(df):
    """Williams %R (14) pullback in trend: close > EMA100 and %R rises back above -80 -> buy
    (mirror below -20 in downtrends). Stop 1.5 ATR, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    hi, lo = pd.Series(h).rolling(14).max().to_numpy(), pd.Series(l).rolling(14).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = -100 * (hi - c) / (hi - lo)
    e = ema(c, 100)
    up = np.r_[False, (wr[1:] > -80) & (wr[:-1] <= -80)]
    dn = np.r_[False, (wr[1:] < -20) & (wr[:-1] >= -20)]
    return up & (c > e), dn & (c < e), {"stop": ("atr", 1.5), "target": ("R", 2.0)}


def s_ema20_bounce(df):
    """EMA20 bounce in a strong trend: EMA20 > EMA50 and both rising, the bar touches EMA20 and
    closes above it as a bullish candle -> buy (mirror). Stop below the bar low - 0.5 ATR, 2R."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    e20, e50, a = ema(c, 20), ema(c, 50), atr(h, l, c, 14)
    rise = np.r_[False, (e20[1:] > e20[:-1]) & (e50[1:] > e50[:-1])]
    fall = np.r_[False, (e20[1:] < e20[:-1]) & (e50[1:] < e50[:-1])]
    up = (e20 > e50) & rise & (l <= e20) & (c > e20) & (c > o)
    dn = (e20 < e50) & fall & (h >= e20) & (c < e20) & (c < o)
    return up, dn, {"stop": ("level", l - 0.5 * a, h + 0.5 * a), "target": ("R", 2.0)}


def s_three_bar_reversal(df):
    """Three-bar reversal at a 20-bar extreme: bar i-1 makes the lowest low of 20 bars and bar i
    closes above the high of bar i-1 -> buy (mirror). Stop below bar i-1's low, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    ll20 = pd.Series(l).rolling(20).min().to_numpy()
    hh20 = pd.Series(h).rolling(20).max().to_numpy()
    pl, ph = np.r_[np.nan, l[:-1]], np.r_[np.nan, h[:-1]]
    pll, phh = np.r_[np.nan, ll20[:-1]], np.r_[np.nan, hh20[:-1]]
    up = (pl <= pll) & (c > ph)
    dn = (ph >= phh) & (c < pl)
    return up, dn & ~up, {"stop": ("level", pl, ph), "target": ("R", 2.0)}


def s_donchian_fade_rsi2(df):
    """Donchian extreme fade (Connors style): close at a new 10-bar low with RSI(2) < 5 and
    close above SMA200 -> buy; mirror. Exit when close crosses the 5-bar SMA. Stop 2.5 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    r2, s200, s5 = rsi(c, 2), sma(c, 200), sma(c, 5)
    up = (l <= pd.Series(l).rolling(10).min().to_numpy()) & (r2 < 5) & (c > s200)
    dn = (h >= pd.Series(h).rolling(10).max().to_numpy()) & (r2 > 95) & (c < s200)
    return up, dn, {"stop": ("atr", 2.5), "target": None, "exit_long": c > s5, "exit_short": c < s5}


CLASSIC.update({
    "TV32": ("Previous-week high/low breakout, 2R", s_weekly_breakout, HTF),
    "TV33": ("Keltner fade to the mean", s_keltner_fade, HTF),
    "TV34": ("Hull MA 55 turn + EMA200, 2R", s_hull_turn, HTF),
    "TV35": ("Williams %R pullback in trend, 2R", s_willr_pullback, HTF),
    "TV36": ("EMA20 bounce in strong trend, 2R", s_ema20_bounce, HTF),
    "TV37": ("Three-bar reversal at 20-bar extreme, 2R", s_three_bar_reversal, HTF),
    "TV38": ("Donchian 10 fade + RSI(2) (Connors)", s_donchian_fade_rsi2, HTF),
})


# ---------------- batch 4 ----------------
def s_chandelier_trend(df):
    """Chandelier exit trend (Chuck LeBeau): long line = 22-bar highest high - 3 ATR22, short
    line = 22-bar lowest low + 3 ATR22. Close crossing above the short line buys (trend flips up),
    below the long line sells. Stop trails on the chandelier line."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 22)
    lng = pd.Series(h).rolling(22).max().to_numpy() - 3 * a
    sht = pd.Series(l).rolling(22).min().to_numpy() + 3 * a
    up, _ = _cross(c, np.r_[np.nan, sht[:-1]])
    _, dn = _cross(c, np.r_[np.nan, lng[:-1]])
    line = np.where(c > sht, lng, sht)
    return up, dn & ~up, {"stop": ("trail_line",), "line": line, "target": None}


def s_roc_adx(df):
    """ROC momentum with ADX: ROC(20) crosses above 0 while ADX(14) > 20 and rising -> buy
    (mirror). Stop 2 ATR, ATR trail 3."""
    from agents.rules_agent import adx
    h, l, c = (df[x].to_numpy() for x in "hlc")
    roc = pd.Series(c).pct_change(20).to_numpy()
    ad = adx(h, l, c, 14)
    rising = np.r_[False, ad[1:] > ad[:-1]]
    up, dn = _cross(roc, np.zeros(len(c)))
    ok = (ad > 20) & rising
    return up & ok, dn & ok, {"stop": ("atr", 2.0), "target": None, "trail_atr": 3.0}


def s_aroon(df):
    """Aroon (Chande, 25): Aroon Up crosses above Aroon Down with Aroon Up > 70 -> buy (mirror).
    Stop 2 ATR, target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = 25
    au = pd.Series(h).rolling(n + 1).apply(lambda v: 100 * np.argmax(v) / n, raw=True).to_numpy()
    ad = pd.Series(l).rolling(n + 1).apply(lambda v: 100 * np.argmin(v) / n, raw=True).to_numpy()
    up, dn = _cross(au, ad)
    return up & (au > 70), dn & (ad > 70), {"stop": ("atr", 2.0), "target": ("R", 2.0)}


def s_cci_trend(df):
    """CCI(20) trend (Lambert): CCI crosses above +100 buys, below -100 sells; exit when CCI
    crosses back through 0. Stop 2 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    tp = (h + l + c) / 3
    m = pd.Series(tp).rolling(20).mean()
    md = pd.Series(tp).rolling(20).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)
    cci = ((pd.Series(tp) - m) / (0.015 * md)).to_numpy()
    up, _ = _cross(cci, np.full(len(c), 100.0))
    _, dn = _cross(cci, np.full(len(c), -100.0))
    return up, dn, {"stop": ("atr", 2.0), "target": None, "exit_long": cci < 0, "exit_short": cci > 0}


def s_donchian_adx(df):
    """Donchian 20 breakout filtered by ADX(14) > 25. Stop 2 ATR, exit on the opposite
    10-bar breakout."""
    from agents.rules_agent import adx
    h, l, c = (df[x].to_numpy() for x in "hlc")
    strong = adx(h, l, c, 14) > 25
    return (c > _hh(h, 20)) & strong, (c < _ll(l, 20)) & strong, {"stop": ("atr", 2.0), "target": None,
                                                                  "exit_long": c < _ll(l, 10), "exit_short": c > _hh(h, 10)}


def s_pin_bar(df):
    """Pin bar at a 20-bar extreme: lower wick >= 2/3 of the range and the low is the 20-bar low
    -> buy (mirror for upper wick at the high). Stop beyond the wick, target 2R."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    rng = np.where(h - l > 0, h - l, np.nan)
    lw = (np.minimum(o, c) - l) / rng
    uw = (h - np.maximum(o, c)) / rng
    up = (lw >= 2 / 3) & (l <= pd.Series(l).rolling(20).min().to_numpy())
    dn = (uw >= 2 / 3) & (h >= pd.Series(h).rolling(20).max().to_numpy())
    return up, dn & ~up, {"stop": ("level", l, h), "target": ("R", 2.0)}


def s_vortex(df):
    """Vortex indicator (Botes/Siepman, 14): VI+ crosses above VI- buys (mirror). Stop 2 ATR,
    ATR trail 3."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    pc, ph, pl = np.r_[c[0], c[:-1]], np.r_[h[0], h[:-1]], np.r_[l[0], l[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    s = lambda x: pd.Series(x).rolling(14).sum().to_numpy()
    vp, vm = s(abs(h - pl)) / s(tr), s(abs(l - ph)) / s(tr)
    up, dn = _cross(vp, vm)
    return up, dn, {"stop": ("atr", 2.0), "target": None, "trail_atr": 3.0}


def s_trix(df):
    """TRIX (Hutson, 15) crosses its 9-signal: above buys (mirror), only with price on the same
    side of EMA200. Stop 2 ATR, target 2R."""
    c = df["c"].to_numpy()
    t = ema(ema(ema(c, 15), 15), 15)
    trix = 100 * np.diff(t, prepend=np.nan) / np.r_[np.nan, t[:-1]]
    sig = pd.Series(trix).ewm(span=9, adjust=False).mean().to_numpy()
    up, dn = _cross(trix, sig)
    e = ema(c, 200)
    return up & (c > e), dn & (c < e), {"stop": ("atr", 2.0), "target": ("R", 2.0)}


def s_elder_impulse(df):
    """Elder impulse system: EMA13 rising and MACD histogram rising = green bar; the first green
    bar after a non-green one buys (mirror red). Exit when the impulse turns the other colour.
    Stop 2 ATR."""
    from agents.rules_agent import macd
    c = df["c"].to_numpy()
    e13 = ema(c, 13)
    _, _, hist = macd(c)
    green = np.r_[False, (e13[1:] > e13[:-1]) & (hist[1:] > hist[:-1])]
    red = np.r_[False, (e13[1:] < e13[:-1]) & (hist[1:] < hist[:-1])]
    fg = np.r_[False, green[1:] & ~green[:-1]]
    fr = np.r_[False, red[1:] & ~red[:-1]]
    return fg, fr, {"stop": ("atr", 2.0), "target": None, "exit_long": red, "exit_short": green}


def s_turn_of_month(df):
    """Turn-of-month (Ariel 1987; Lakonishok & Smidt): long indices from the last 2 trading days
    of the month; stop 2 ATR, target 2R (no time exit). Long only; applies to any instrument."""
    c = df["c"].to_numpy()
    idx = df.index.tz_convert("UTC")
    day = pd.Series(idx.date)
    month = idx.to_period("M") if idx.tz is None else idx.tz_localize(None).to_period("M")
    days = pd.DataFrame({"m": month.astype(str), "d": day}).drop_duplicates()
    days["rank_from_end"] = days.groupby("m").cumcount(ascending=False)
    last2 = set(days.loc[days["rank_from_end"] == 1, "d"])
    first_bar = np.r_[True, day.to_numpy()[1:] != day.to_numpy()[:-1]]
    sig = first_bar & day.isin(last2).to_numpy()
    return sig, np.zeros(len(c), bool), {"stop": ("atr", 2.0), "target": ("R", 2.0)}


CLASSIC.update({
    "TV39": ("Chandelier exit trend (LeBeau)", s_chandelier_trend, HTF),
    "TV40": ("ROC(20) momentum + rising ADX", s_roc_adx, HTF),
    "TV41": ("Aroon 25 cross, 2R", s_aroon, HTF),
    "TV42": ("CCI(20) +/-100 trend", s_cci_trend, HTF),
    "TV43": ("Donchian 20 + ADX>25", s_donchian_adx, HTF),
    "TV44": ("Pin bar at 20-bar extreme, 2R", s_pin_bar, HTF),
    "TV45": ("Vortex 14 cross + ATR trail", s_vortex, HTF),
    "TV46": ("TRIX 15/9 cross + EMA200, 2R", s_trix, HTF),
    "TV47": ("Elder impulse system", s_elder_impulse, HTF),
    "TV48": ("Turn-of-month long, 2R", s_turn_of_month, HTF),
})


# ---------------- variants of last-year Prophet passers (TV04, TV14, TV25, TV28) ----------------
def _with_target(base, target):
    """Same entries/stops as `base`, different take-profit (R multiple, or None = base exit only)."""
    def fn(df):
        lg, sh, plan = base(df)
        plan = dict(plan)
        plan["target"] = ("R", target) if target else None
        return lg, sh, plan
    fn.__doc__ = f"Variant of {base.__name__} with target {target}R. " + (base.__doc__ or "")
    return fn


def _tv04():
    from agents.rules_agent import STRATEGIES
    return STRATEGIES["TV04"][1]


def s_tv04_15(df):
    return _with_target(_tv04(), 1.5)(df)


def s_tv04_3(df):
    return _with_target(_tv04(), 3.0)(df)


CLASSIC.update({
    "TV49": ("TV04 Golden Edge, target 1.5R (variant)", s_tv04_15, HTF),
    "TV50": ("TV04 Golden Edge, target 3R (variant)", s_tv04_3, HTF),
    "TV51": ("TV14 Turtle 55/20 + 3R target (variant)", _with_target(s_turtle_55, 3.0), HTF),
    "TV52": ("TV25 Heikin-Ashi + 2R target (variant)", _with_target(s_heikin_ashi, 2.0), HTF),
    "TV53": ("TV25 Heikin-Ashi + 3R target (variant)", _with_target(s_heikin_ashi, 3.0), HTF),
    "TV54": ("TV28 Stochastic pullback, 1.5R (variant)", _with_target(s_stoch_pullback, 1.5), HTF),
    "TV55": ("TV28 Stochastic pullback, 3R (variant)", _with_target(s_stoch_pullback, 3.0), HTF),
})


# ---------------- batch 5 ----------------
def s_envelope_fade(df):
    """Moving-average envelope fade: close below SMA20 x (1 - 2.5 x ATR%) ... implemented as
    SMA20 -/+ 3% of price scaled by volatility: close outside SMA20 +/- 2.5 x ATR14 then back
    inside -> fade; exit at SMA20. Stop 1.5 ATR. (Envelope with ATR width instead of fixed %.)"""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    m, a = sma(c, 20), atr(h, l, c, 14)
    lo, hi = m - 2.5 * a, m + 2.5 * a
    was_lo, was_hi = np.r_[False, (c < lo)[:-1]], np.r_[False, (c > hi)[:-1]]
    return was_lo & (c > lo), was_hi & (c < hi), {"stop": ("atr", 1.5), "target": None,
                                                  "exit_long": c >= m, "exit_short": c <= m}


def s_double_bottom(df):
    """Double bottom/top neckline break: two swing lows (5-bar pivots) within 0.5 ATR of each
    other, 10-60 bars apart; a close above the highest high between them (neckline) buys.
    Stop below the lower bottom, target = neckline + pattern height. Mirror for tops."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    piv_l = [i for i in range(5, n - 5) if l[i] == l[i - 5:i + 6].min()]
    piv_h = [i for i in range(5, n - 5) if h[i] == h[i - 5:i + 6].max()]
    for lst, d in ((piv_l, 1), (piv_h, -1)):
        for j in range(1, len(lst)):
            i1, i2 = lst[j - 1], lst[j]
            if not 10 <= i2 - i1 <= 60:
                continue
            p1, p2 = (l[i1], l[i2]) if d == 1 else (h[i1], h[i2])
            if abs(p1 - p2) > 0.5 * a[i2]:
                continue
            neck = h[i1:i2].max() if d == 1 else l[i1:i2].min()
            for k in range(i2 + 5, min(n, i2 + 40)):
                if d == 1 and c[k] > neck:
                    long_[k], sl_l[k], tg_l[k] = True, min(p1, p2) - 0.1 * a[k], neck + (neck - min(p1, p2))
                    break
                if d == -1 and c[k] < neck:
                    short[k], sl_s[k], tg_s[k] = True, max(p1, p2) + 0.1 * a[k], neck - (max(p1, p2) - neck)
                    break
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("level", tg_l, tg_s)}


def s_kst(df):
    """Know Sure Thing (Pring): KST crosses its 9-SMA signal -> trade in that direction, with price
    on the same side of EMA200. Stop 2 ATR, target 2R."""
    c = pd.Series(df["c"].to_numpy())
    roc = lambda n: c.pct_change(n) * 100
    kst = (roc(10).rolling(10).mean() + 2 * roc(15).rolling(10).mean() + 3 * roc(20).rolling(10).mean()
           + 4 * roc(30).rolling(15).mean()).to_numpy()
    sig = pd.Series(kst).rolling(9).mean().to_numpy()
    up, dn = _cross(kst, sig)
    cc = c.to_numpy()
    e = ema(cc, 200)
    return up & (cc > e), dn & (cc < e), {"stop": ("atr", 2.0), "target": ("R", 2.0)}


CLASSIC.update({
    "TV56": ("ATR envelope fade to SMA20", s_envelope_fade, HTF),
    "TV57": ("Double bottom/top neckline break", s_double_bottom, HTF),
    "TV58": ("KST cross + EMA200, 2R", s_kst, HTF),
})
