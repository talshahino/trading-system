"""ATR Fibonacci Trend Envelopes [BigBeluga]: https://www.tradingview.com/script/16toJPAD-ATR-Fibonacci-Trend-Envelopes-BigBeluga/
Ported from the ProBuilder translation (https://www.prorealcode.com/prorealtime-indicators/atr-fibonacci-trend-envelopes/):
basis = RMA(close, 100); bands = basis +/- 3 x ATR(100); trend +1 when close crosses over the upper band, -1 under
the lower; in an up trend the trail = lower band and the fib lines are lower + range x 0.5 / 0.382 / 0.214 (the port's
'0.5 / 0.618 / 0.786' lines), mirror in a down trend; the 'golden pocket' = between the 0.618 and 0.786 lines.
Entries are discretionary in the original; ours (for the bots): a pullback into the pocket in the trend direction.
The MTF dashboard and future projections are left out. No repaint."""
import numpy as np

from agents.indicators._core import atr, cols, cross_down, cross_up, rma


def atr_fib_envelopes(df, ma_len=100, atr_len=100, mult=3.0):
    """dict: basis, upper, lower, trend, trail, f50, f618, f786, in_pocket (low/high inside the golden pocket)."""
    o, h, l, c, v = cols(df)
    b = rma(c, ma_len)
    a = atr(h, l, c, atr_len)
    up, dn = b + mult * a, b - mult * a
    cu, cd = cross_up(c, up), cross_down(c, dn)
    t = np.zeros(len(c))
    for i in range(1, len(c)):
        t[i] = 1 if cu[i] else -1 if cd[i] else t[i - 1]
    t[:max(ma_len, atr_len)] = 0
    rg = up - dn
    bull = t > 0
    f50 = np.where(bull, dn + rg * 0.5, up - rg * 0.5)
    f618 = np.where(bull, dn + rg * 0.382, up - rg * 0.382)
    f786 = np.where(bull, dn + rg * 0.214, up - rg * 0.214)
    trail = np.where(bull, dn, up)
    lo_p, hi_p = np.minimum(f618, f786), np.maximum(f618, f786)
    inp = np.where(bull, (l <= hi_p) & (c >= lo_p), (h >= lo_p) & (c <= hi_p)) & (t != 0)
    return {"basis": b, "upper": up, "lower": dn, "trend": t, "trail": trail, "f50": f50, "f618": f618,
            "f786": f786, "in_pocket": inp}
