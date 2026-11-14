"""HalfTrend (everget, open source https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/) as used by 'HalfTrend Long/Short
Signal Engine [BigBeluga]' (open source https://www.tradingview.com/script/ZHGPnlAz-HalfTrend-Long-Short-Signal-Engine-BigBeluga/;
Trading Data's video 'This FREE Indicator Should Be Illegal', task tal1411ind). Port of everget's loop: amplitude 2,
channel deviation 2, atr2 = ATR(100) / 2; the trend turns down when SMA(high, 2) < the highest of the up-trend lows and
the close is under the previous low (mirror for up); the line ratchets; bands = line +/- deviation x atr2. LONG on a
bearish -> bullish flip, SHORT on the reverse. Closed bars only - no repaint."""
import numpy as np

from agents.indicators._core import atr, cols, highest, lowest, sma


def halftrend(df, amplitude=2, deviation=2.0):
    """dict: line, trend (0 up / 1 down), buy, sell, band_hi, band_lo."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a2 = atr(h, l, c, 100) / 2
    dev = deviation * a2
    hp, lp = highest(h, amplitude), lowest(l, amplitude)
    hma_, lma_ = sma(h, amplitude), sma(l, amplitude)
    trend = np.zeros(n, dtype=int)
    nxt = 0
    max_low = l[0] if n else 0.0
    min_high = h[0] if n else 0.0
    up = np.zeros(n)
    dn = np.zeros(n)
    line = np.full(n, np.nan)
    for t in range(1, n):
        trend[t] = trend[t - 1]
        if nxt == 1:
            max_low = max(lp[t], max_low)
            if hma_[t] < max_low and c[t] < l[t - 1]:
                trend[t], nxt, min_high = 1, 0, hp[t]
        else:
            min_high = min(hp[t], min_high)
            if lma_[t] > min_high and c[t] > h[t - 1]:
                trend[t], nxt, max_low = 0, 1, lp[t]
        if trend[t] == 0:
            up[t] = dn[t - 1] if trend[t - 1] != 0 else max(max_low, up[t - 1])
            dn[t] = dn[t - 1]
            line[t] = up[t]
        else:
            dn[t] = up[t - 1] if trend[t - 1] != 1 else min(min_high, dn[t - 1])
            up[t] = up[t - 1]
            line[t] = dn[t]
    prev = np.concatenate([[0], trend[:-1]])
    ok = np.arange(n) > 100
    return {"line": line, "trend": trend, "buy": (trend == 0) & (prev == 1) & ok, "sell": (trend == 1) & (prev == 0) & ok,
            "band_hi": line + dev, "band_lo": line - dev}
