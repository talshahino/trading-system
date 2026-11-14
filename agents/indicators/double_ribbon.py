"""Double Ribbon [ChartPrime]: https://www.tradingview.com/script/OdtelUqS-Double-Ribbon-ChartPrime/
From the page's description (code not read - OUR RECONSTRUCTION): two sets of 11 SMAs - a gray ribbon of one fixed
length shown with growing offsets, and a coloured ribbon with step-wise growing lengths; trend score = for each pair
+1 when the coloured SMA is above its gray partner, -1 below, summed (-11 .. +11) and scaled to -1 .. +1.
Ours: base length 20, offsets 0..10 bars, coloured lengths 20 + 2k. Uses: score crossing 0 = trend change,
|score| = 1 = full alignment."""
import numpy as np

from agents.indicators._core import cols, shift, sma


def double_ribbon(df, length=20, step=2, lines=11):
    """dict: score (-1..+1), buy (score crosses above 0), sell (below 0)."""
    o, h, l, c, v = cols(df)
    base = sma(c, length)
    sc = np.zeros(len(c))
    for k in range(lines):
        g = shift(base, k)
        col = sma(c, length + step * k)
        sc += np.where(np.isfinite(g) & np.isfinite(col), np.sign(col - g), 0)
    sc = sc / lines
    p = np.concatenate([[0], sc[:-1]])
    return {"score": sc, "buy": (sc > 0) & (p <= 0), "sell": (sc < 0) & (p >= 0)}
