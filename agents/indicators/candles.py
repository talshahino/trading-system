"""Japanese candlestick patterns (scout 16.11.2026) - port of the TA-Lib pattern definitions
(https://ta-lib.org/functions/ , candle settings https://ta-lib.org/api/candle-settings/ : a 'long' body is longer than the
average body of the previous 10 bars, a 'short' body shorter than it, a doji body <= 10% of the average range of the
previous 10 bars) with the pattern statistics of Thomas Bulkowski, Encyclopedia of Candlestick Charts
(https://www.thepatternsite.com/CandlePerformers.html , https://www.thepatternsite.com/ThreeLineStrikeBull.html).

Every pattern ENDS on bar i and uses bars <= i only (+1 bullish, -1 bearish, 0 none). OUR OWN changes, stated:
- 24-hour forex / CFD feeds open each bar at the previous close, so real-body gaps almost never exist there. The star
  patterns (morning / evening star, doji star) therefore accept a star whose body sits beyond the first body's close OR
  at it (<=) instead of a strict body gap; the abandoned baby keeps Bulkowski's SHADOW gaps; the kicker keeps the open
  gap (it fires only on gapping markets - indices / shares at the cash open).
- marubozu / soldiers / crows use 'close in the top (bottom) 30% of the bar's range' instead of TA-Lib's
  'very short shadow' (10% of the 10-bar average range) - looser, so the patterns occur on noisy intraday bars.
Trend context (a reversal needs something to reverse) is left to the strategies.
"""
import numpy as np

from agents.indicators._core import S, cols


def _avg_prev(x, n=10):
    return S(x).shift(1).rolling(n, min_periods=3).mean().to_numpy()


def _b(x, k):
    """x shifted right by k (value k bars ago), NaN-filled."""
    x = np.asarray(x, dtype=float)
    return np.concatenate([np.full(k, np.nan), x[:-k]]) if k > 0 else x


def candle_patterns(df, avg_n=10):
    """dict of int8 arrays (+1 / -1 / 0): morning_star, abandoned_baby, doji_star, three_outside, three_inside, harami,
    piercing, three_soldiers, three_line_strike (TA-Lib sign = the direction of the 3 candles), kicker, tweezer,
    marubozu, three_methods, belt_hold, engulfing; plus 'pattern_low' / 'pattern_high' (the extreme of the last 3 bars,
    for stops)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    body = np.abs(c - o)
    rng = h - l
    top, bot = np.maximum(o, c), np.minimum(o, c)
    ab = _avg_prev(body, avg_n)
    ar = _avg_prev(rng, avg_n)
    white, black = c > o, c < o
    long_ = body > ab
    short = body < ab
    doji = body <= 0.1 * ar
    B = lambda x, k: _b(x, k)  # noqa: E731
    o1, h1, l1, c1, b1, t1, bo1 = B(o, 1), B(h, 1), B(l, 1), B(c, 1), B(body, 1), B(top, 1), B(bot, 1)
    o2, h2, l2, c2, b2, t2, bo2 = B(o, 2), B(h, 2), B(l, 2), B(c, 2), B(body, 2), B(top, 2), B(bot, 2)
    o3, c3 = B(o, 3), B(c, 3)
    o4, h4, l4, c4 = B(o, 4), B(h, 4), B(l, 4), B(c, 4)
    w1, k1, w2, k2, w3, k3, w4, k4 = (B(white, 1) > 0, B(black, 1) > 0, B(white, 2) > 0, B(black, 2) > 0,
                                      B(white, 3) > 0, B(black, 3) > 0, B(white, 4) > 0, B(black, 4) > 0)
    lg1, lg2, lg4 = B(long_, 1) > 0, B(long_, 2) > 0, B(long_, 4) > 0
    sh1 = B(short, 1) > 0
    dj1 = B(doji, 1) > 0
    out = {}
    with np.errstate(invalid="ignore"):
        # morning / evening star (TA-Lib penetration 0.3)
        bull = k2 & lg2 & sh1 & (t1 <= c2) & white & ~short & (c > c2 + 0.3 * b2)
        bear = w2 & lg2 & sh1 & (bo1 >= c2) & black & ~short & (c < c2 - 0.3 * b2)
        out["morning_star"] = np.where(bull, 1, np.where(bear, -1, 0))
        # abandoned baby (shadow gaps on both sides of the doji)
        bull = k2 & lg2 & dj1 & (h1 < l2) & white & (l > h1) & (c > c2 + 0.3 * b2)
        bear = w2 & lg2 & dj1 & (l1 > h2) & black & (h < l1) & (c < c2 - 0.3 * b2)
        out["abandoned_baby"] = np.where(bull, 1, np.where(bear, -1, 0))
        # doji star (2 bars): long body then a doji at / beyond its close
        bull = k1 & lg1 & doji & (top <= c1)
        bear = w1 & lg1 & doji & (bot >= c1)
        out["doji_star"] = np.where(bull, 1, np.where(bear, -1, 0))
        # engulfing (bodies), three outside (engulfing + a higher / lower close)
        bull_e = white & k1 & (c >= o1) & (o <= c1) & (body > b1)
        bear_e = black & w1 & (c <= o1) & (o >= c1) & (body > b1)
        out["engulfing"] = np.where(bull_e, 1, np.where(bear_e, -1, 0))
        be1, se1 = B(bull_e, 1) > 0, B(bear_e, 1) > 0
        out["three_outside"] = np.where(be1 & white & (c > c1), 1, np.where(se1 & black & (c < c1), -1, 0))
        # harami (2 bars) and three inside (harami + a close beyond the first body)
        hb = k1 & lg1 & short & white & (top <= o1) & (bot >= c1)
        hs = w1 & lg1 & short & black & (top <= c1) & (bot >= o1)
        out["harami"] = np.where(hb, 1, np.where(hs, -1, 0))
        hb1, hs1 = B(hb, 1) > 0, B(hs, 1) > 0
        out["three_inside"] = np.where(hb1 & white & (c > o2), 1, np.where(hs1 & black & (c < o2), -1, 0))
        # piercing line / dark cloud cover (opens beyond the prior extreme or at its close on no-gap feeds)
        mid1 = (o1 + c1) / 2
        bull = k1 & lg1 & white & long_ & (o <= c1) & (c > mid1) & (c < o1)
        bear = w1 & lg1 & black & long_ & (o >= c1) & (c < mid1) & (c > o1)
        out["piercing"] = np.where(bull, 1, np.where(bear, -1, 0))
        # three white soldiers / three black crows
        topq = c >= h - 0.3 * rng
        botq = c <= l + 0.3 * rng
        sw = white & topq & ~short
        sk = black & botq & ~short
        sw1, sw2, sk1, sk2 = B(sw, 1) > 0, B(sw, 2) > 0, B(sk, 1) > 0, B(sk, 2) > 0
        bull = sw & sw1 & sw2 & (c > c1) & (c1 > c2) & (o >= o1) & (o <= c1) & (o1 >= o2) & (o1 <= c2)
        bear = sk & sk1 & sk2 & (c < c1) & (c1 < c2) & (o <= o1) & (o >= c1) & (o1 <= o2) & (o1 >= c2)
        out["three_soldiers"] = np.where(bull, 1, np.where(bear, -1, 0))
        # three line strike: 3 candles one way (higher closes), a 4th opening beyond the 3rd close and closing past
        # the 1st open (TA-Lib sign = the 3 candles' direction; Bulkowski: the bullish one breaks DOWN most often)
        bull = w3 & w2 & w1 & (c2 > c3) & (c1 > c2) & black & (o >= c1) & (c < o3)
        bear = k3 & k2 & k1 & (c2 < c3) & (c1 < c2) & white & (o <= c1) & (c > o3)
        out["three_line_strike"] = np.where(bull, 1, np.where(bear, -1, 0))
        # kicker: marubozu-like opposite candles with an open gap
        mw = white & topq & (o <= l + 0.3 * rng) & long_
        mk = black & botq & (o >= h - 0.3 * rng) & long_
        mw1, mk1 = B(mw, 1) > 0, B(mk, 1) > 0
        out["kicker"] = np.where(mk1 & mw & (o > o1), 1, np.where(mw1 & mk & (o < o1), -1, 0))
        out["marubozu"] = np.where(mw & (body >= 0.7 * rng), 1, np.where(mk & (body >= 0.7 * rng), -1, 0))
        # tweezer bottom / top: equal lows (highs) within 5% of the average range, black then white
        tol = 0.05 * ar
        out["tweezer"] = np.where(k1 & white & (np.abs(l - l1) <= tol), 1,
                                  np.where(w1 & black & (np.abs(h - h1) <= tol), -1, 0))
        # rising / falling three methods (5 bars)
        sm = [B(short, k) > 0 for k in (1, 2, 3)]
        inside = np.ones(n, bool)
        for k in (1, 2, 3):
            inside &= (B(h, k) <= h4) & (B(l, k) >= l4)
        smalls = sm[0] & sm[1] & sm[2]
        bull = w4 & lg4 & smalls & inside & white & long_ & (c > c4)
        bear = k4 & lg4 & smalls & inside & black & long_ & (c < c4)
        out["three_methods"] = np.where(bull, 1, np.where(bear, -1, 0))
        # belt hold: a long candle opening at its extreme
        bull = white & long_ & (o - l <= 0.05 * rng)
        bear = black & long_ & (h - o <= 0.05 * rng)
        out["belt_hold"] = np.where(bull, 1, np.where(bear, -1, 0))
    for k in list(out):
        a = np.nan_to_num(np.asarray(out[k], dtype=float)).astype(np.int8)
        a[: avg_n + 4] = 0
        out[k] = a
    out["pattern_low"] = S(l).rolling(3, min_periods=1).min().to_numpy()
    out["pattern_high"] = S(h).rolling(3, min_periods=1).max().to_numpy()
    return out
