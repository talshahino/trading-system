"""Trend / range regime gauges.

choppiness - Choppiness Index (E.W. Dreiss): CHOP = 100 log10(sum TR(n) / (highest high n - lowest low n)) / log10(n);
under 38.2 = trending, over 61.8 = choppy. Port of the StockSharp example
https://doc.stocksharp.com/en/api-examples/0045_Choppiness_Index_Breakout (TR of each bar, n = 14).

efficiency_ratio - Kaufman Efficiency Ratio, signed: (close - close[n]) / sum |close - close[1]| over n, -1..+1
(WH SelfInvest 'Kaufman Efficiency Ratio' https://whselfinvest.com/en/trading_strategies_33_Kaufman_Efficiency_Ratio.php,
zones +/- 0.6). The lookback is not stated there - ours n = 10 (Kaufman's own default for KAMA).

squeeze_pro - TTM Squeeze Pro (John Carter; open source https://www.tradingview.com/script/0drMdHsO-TTM-Squeeze-Pro/ and
LuxAlgo concept https://www.luxalgo.com/library/concept/ttm-squeeze/): BB(20, 2) against three Keltner channels
SMA20 +/- k x ATR(20), k = 2.0 (low), 1.5 (mid), 1.0 (high compression); momentum = linear-regression value (20) of
close - ((HH20 + LL20) / 2 + SMA20) / 2. level: 0 none, 1 low, 2 mid, 3 high; fire = mid squeeze on the previous
bar and none now. All on closed bars - no repaint."""
import numpy as np

from agents.indicators._core import S, atr, cols, highest, lowest, shift, sma, stdev, tr


def choppiness(df, n=14):
    """array 0..100."""
    o, h, l, c, v = cols(df)
    s = S(tr(h, l, c)).rolling(n).sum().to_numpy()
    w = S(h).rolling(n).max().to_numpy() - S(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ch = 100 * np.log10(s / w) / np.log10(n)
    return np.where((w > 0) & (s > 0), ch, np.nan)


def efficiency_ratio(df, n=10):
    """array -1..+1 (signed)."""
    o, h, l, c, v = cols(df)
    ch = c - shift(c, n)
    vol = S(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 0, ch / vol, 0.0)


def _linreg_last(x, n):
    """Value of the least-squares line of the last n points at the last point (Pine linreg(x, n, 0))."""
    xs = np.arange(n, dtype=float)
    xm = xs.mean()
    den = ((xs - xm) ** 2).sum()

    def f(w):
        b = ((xs - xm) * (w - w.mean())).sum() / den
        return w.mean() + b * (n - 1 - xm)
    return S(x).rolling(n).apply(f, raw=True).to_numpy()


def squeeze_pro(df, n=20, bb=2.0):
    """dict: level (0-3), fire (bool), mom, mom_up (momentum rising)."""
    o, h, l, c, v = cols(df)
    m = sma(c, n)
    sd = stdev(c, n)
    a = S(tr(h, l, c)).rolling(n).mean().to_numpy()   # Carter: SMA of the true range
    bu, bd = m + bb * sd, m - bb * sd
    level = np.zeros(len(c), dtype=int)
    for lev, k in ((1, 2.0), (2, 1.5), (3, 1.0)):
        inside = (bd > m - k * a) & (bu < m + k * a)
        level = np.where(inside, lev, level)
    mom = _linreg_last(c - ((highest(h, n) + lowest(l, n)) / 2 + m) / 2, n)
    prev = np.concatenate([[0], level[:-1]])
    fire = (prev >= 2) & (level < 2)
    return {"level": level, "fire": fire, "mom": mom, "mom_up": mom > shift(mom, 1)}
