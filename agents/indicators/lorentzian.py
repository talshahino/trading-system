"""Machine Learning: Lorentzian Classification (jdehorty) - open-source Pine (MPL 2.0):
https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/ (Python port by the same
community: https://pypi.org/project/advanced-ta/).

Port of the defaults: features f1 RSI(14, smooth 1), f2 WaveTrend(10, 11), f3 CCI(20, 1), f4 ADX(20, 2), f5 RSI(9, 1),
each normalised like the original (RSI and ADX rescaled 0..100 -> 0..1, WT and CCI min-max normalised over all
bars so far); training label at bar j = sign(close[j+4] - close[j]) (known only from bar j+4 on - so at bar i only
labels of bars j <= i-4 are used: no look-ahead); neighbours k = 8, max bars back 2000, every 4th bar (the original's
'i % 4' spacing); distance = sum log(1 + |f_now - f_then|) (Lorentzian). Prediction = sum of the k neighbours'
labels; signal = +1 when prediction > 0, -1 when < 0, else hold the previous signal.
Our reconstruction / simplification: the original's approximate nearest-neighbour loop (keeping a rolling
'lastDistance' at the 75th percentile) is replaced by the exact k nearest among the spaced candidates; the optional
filters (volatility, regime, ADX, EMA/SMA, kernel) are returned separately: kernel = Nadaraya-Watson rational
quadratic estimate (h 8, r 8, start bar 25) on closed bars only - the endpoint version, which does not repaint.
Exit rule of the original: 4 bars after entry (or an opposite signal)."""
import numpy as np

from agents.indicators._core import S, cci, cols, dmi, ema, rsi, sma


def _minmax_running(x):
    s = S(x)
    lo, hi = s.cummin().to_numpy(), s.cummax().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hi > lo, (x - lo) / (hi - lo), 0.5)


def _wt(h, l, c, n1=10, n2=11):
    hlc3 = (h + l + c) / 3
    e1 = ema(hlc3, n1)
    e2 = ema(np.abs(hlc3 - e1), n1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.nan_to_num((hlc3 - e1) / (0.015 * e2))
    wt1 = ema(ci, n2)
    return wt1 - sma(wt1, 4)


def rq_kernel(c, h=8.0, r=8.0, start=25):
    """Nadaraya-Watson rational quadratic kernel estimate at the current bar (endpoint, no repaint)."""
    n = len(c)
    out = np.full(n, np.nan)
    m = min(n, start + 1)
    k = np.arange(start + 1, dtype=float)
    w = (1 + k ** 2 / (h ** 2 * 2 * r)) ** (-r)
    for i in range(n):
        j = min(i + 1, start + 1)
        seg = c[i - j + 1:i + 1][::-1]
        out[i] = np.dot(seg, w[:j]) / w[:j].sum()
    _ = m
    return out


def lorentzian(df, neighbors=8, max_back=2000, spacing=4):
    """Returns dict: signal (+1/-1), prediction (sum of neighbour labels), buy / sell (signal changes),
    kernel (RQ kernel estimate), kernel_up (kernel rising)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    pdi, ndi, ad = dmi(h, l, c, 20)
    F = np.vstack([
        np.nan_to_num(rsi(c, 14)) / 100.0,
        _minmax_running(np.nan_to_num(_wt(h, l, c, 10, 11))),
        _minmax_running(np.nan_to_num(cci(h, l, c, 20))),
        np.nan_to_num(ema(ad, 2)) / 100.0,
        np.nan_to_num(rsi(c, 9)) / 100.0,
    ]).T
    lab = np.zeros(n)
    lab[:-4] = np.sign(c[4:] - c[:-4])
    pred = np.zeros(n)
    sig = np.zeros(n)
    s = 0.0
    for i in range(n):
        hi = i - 4
        if hi >= 50:
            lo = max(0, hi - max_back)
            idx = np.arange(hi, lo - 1, -spacing)
            d = np.log1p(np.abs(F[idx] - F[i])).sum(axis=1)
            kk = min(neighbors, len(idx))
            near = idx[np.argpartition(d, kk - 1)[:kk]]
            pred[i] = lab[near].sum()
        s = 1.0 if pred[i] > 0 else -1.0 if pred[i] < 0 else s
        sig[i] = s
    prev = np.concatenate([[0.0], sig[:-1]])
    ker = rq_kernel(c)
    kup = np.concatenate([[False], ker[1:] > ker[:-1]])
    return {"signal": sig, "prediction": pred, "buy": (sig == 1) & (prev != 1) & (np.arange(n) > 54),
            "sell": (sig == -1) & (prev != -1) & (np.arange(n) > 54), "kernel": ker, "kernel_up": kup}
