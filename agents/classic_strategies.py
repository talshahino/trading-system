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
