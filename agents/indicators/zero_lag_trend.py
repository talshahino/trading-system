"""Zero Lag Trend Signals (MTF) [AlgoAlpha] - open-source Pine (MPL 2.0):
https://www.tradingview.com/script/tHK4O1Fi-Zero-Lag-Trend-Signals-MTF-AlgoAlpha/
Port of the published code (defaults length 70, band multiplier 1.2): lag = floor((length-1)/2);
ZLEMA = EMA(close + (close - close[lag]), length); volatility = highest(ATR(length), length*3) * mult;
trend turns +1 when close crosses over ZLEMA + volatility, -1 when it crosses under ZLEMA - volatility.
Signals: trend change (big arrows); 'Bullish Entry' = close crosses over ZLEMA while the trend is already +1 (and
was +1 on the bar before), 'Bearish Entry' mirror. The MTF table (other timeframes) is left out - each bot can
call this on resampled bars. No repaint."""
import numpy as np

from agents.indicators._core import atr, cols, cross_down, cross_up, ema, highest, shift


def zero_lag_trend(df, length=70, mult=1.2):
    """dict: zlema, upper, lower, trend (+1/-1/0), buy / sell (trend flips), entry_long / entry_short."""
    o, h, l, c, v = cols(df)
    lag = int((length - 1) // 2)
    src = c + (c - np.nan_to_num(shift(c, lag), nan=c[0] if len(c) else 0))
    z = ema(src, length)
    vol = highest(atr(h, l, c, length), length * 3) * mult
    up, dn = z + vol, z - vol
    cu, cd = cross_up(c, up), cross_down(c, dn)
    t = np.zeros(len(c))
    for i in range(1, len(c)):
        t[i] = 1 if cu[i] else -1 if cd[i] else t[i - 1]
    pt = np.concatenate([[0], t[:-1]])
    return {"zlema": z, "upper": up, "lower": dn, "trend": t,
            "buy": (t == 1) & (pt != 1), "sell": (t == -1) & (pt != -1),
            "entry_long": cross_up(c, z) & (t == 1) & (pt == 1),
            "entry_short": cross_down(c, z) & (t == -1) & (pt == -1)}
