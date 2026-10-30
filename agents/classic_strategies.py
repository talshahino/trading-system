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
