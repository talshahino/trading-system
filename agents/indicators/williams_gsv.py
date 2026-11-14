"""Larry Williams 'Greatest Swing Value' breakout levels (Williams, 'Long-Term Secrets to Short-Term Trading'; MQL5
article https://www.mql5.com/en/articles/21913). Failure swings of the completed bars: on bearish bars (close < open)
the buy swing = high - open, on bullish bars the sell swing = open - low; the average of the last 4 positive swings
(searching back at most 500 bars). Levels for the current bar: buy_trig = open + 1.8 x avg buy swing, sell_trig =
open - 1.8 x avg sell swing; setup: bullish when close[1] < close[6] (5 bars of weakness), bearish when close[1] >
close[6]. Uses only completed bars and the bar's own open - no repaint."""
import numpy as np

from agents.indicators._core import cols, shift


def _avg_last(vals, mask, k, limit):
    n = len(vals)
    out = np.full(n, np.nan)
    buf = []
    for i in range(n):
        out[i] = np.mean(buf[-k:]) if len(buf) >= k else np.nan   # completed bars only (< i)
        if mask[i] and vals[i] > 0:
            buf.append(vals[i])
            buf = buf[-k:]
    return out


def williams_gsv(df, k=4, mult=1.8, setup=5):
    """dict: buy_trig, sell_trig, bull_setup, bear_setup."""
    o, h, l, c, v = cols(df)
    ab = _avg_last(h - o, c < o, k, 500)
    as_ = _avg_last(o - l, c > o, k, 500)
    c1, c6 = shift(c, 1), shift(c, 1 + setup)
    return {"buy_trig": o + mult * ab, "sell_trig": o - mult * as_,
            "bull_setup": np.nan_to_num(c1 < c6, nan=0).astype(bool), "bear_setup": np.nan_to_num(c1 > c6, nan=0).astype(bool)}
