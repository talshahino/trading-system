"""Nadaraya-Watson Envelope [LuxAlgo] - open-source Pine (CC BY-NC-SA): https://www.tradingview.com/script/Iko0E2kL-Nadaraya-Watson-Envelope-LuxAlgo/
The user's 'Non Repaint' envelope with reversal dots (task tal1411ind: a paid ad, build our own version). The original
repaints by default (it smooths the past with future bars); we use only its NON-repainting 'endpoint' mode:
out[i] = sum_k src[i-k] w_k / sum_k w_k, w_k = exp(-k^2 / (2 h^2)), k = 0..499, h = 8; envelope = out +/- 3 x
SMA(|src - out|, 499) (the original's MAE). Signals as in the original: close crosses under the lower band = buy dot,
over the upper band = sell dot. Ours: window 500 (truncated at the start)."""
import numpy as np

from agents.indicators._core import cols, cross_down, cross_up, sma


def nwe_envelope(df, h=8.0, mult=3.0, window=500):
    """dict: mid, upper, lower, buy, sell."""
    o, hh, l, c, v = cols(df)
    n = len(c)
    k = np.arange(window, dtype=float)
    w = np.exp(-(k ** 2) / (2 * h * h))
    kk = int(min(window, np.searchsorted(-w, -1e-12)))  # weights beyond ~6h are 0 - cut for speed
    kk = max(kk, 1)
    w = w[:kk]
    mid = np.full(n, np.nan)
    cs = c[::-1]
    for i in range(n):
        j = min(i + 1, kk)
        seg = cs[n - 1 - i:n - 1 - i + j]
        mid[i] = np.dot(seg, w[:j]) / w[:j].sum()
    mae = sma(np.abs(c - mid), window - 1) * mult
    up, dn = mid + mae, mid - mae
    return {"mid": mid, "upper": up, "lower": dn, "buy": cross_down(c, dn), "sell": cross_up(c, up)}
