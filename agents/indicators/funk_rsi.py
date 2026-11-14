"""Funk RSI: Break the Rules [Almazov Trade]: https://www.tradingview.com/script/823fXWAN-funk-rsi-break-the-rules-almazov-trade/
Port of the published rules (the page's description; code not read): RSI(14) thresholds 30/70, swing width N = 1.
A swing high at bar i (high[i] >= every high of i-N..i+N) whose window has any RSI > 70 -> SHORT at the close of
bar i+N (the bar where the swing is confirmed - no back-dating), target = low[i], stop = high[j - 14] where j is the
first bar of the window whose RSI is beyond the threshold; skipped if bars i..i+N are all bullish. Swing low / RSI < 30
-> LONG, target high[i], stop low[j - 14], skipped if i..i+N are all bearish. A window qualifying both ways is skipped;
the trade needs stop < entry < target (long). OUR CHANGE: the page's stop (low / high of bar j - 14) lies on the
wrong side of the entry almost every time (the price was falling into the RSI < 30), so the stop here is the lowest
low (highest high) of bars j-14 .. i+N plus 0.25 x ATR(14) - beyond the whole RSI excursion. No repaint: everything
is known at bar i+N."""
import numpy as np

from agents.indicators._core import atr, cols, rsi


def funk_rsi(df, n_rsi=14, lo=30.0, hi=70.0, N=1):
    """dict: long, short (signal on the confirmation bar), stop_long, stop_short, tgt_long, tgt_short."""
    o, h, l, c, v = cols(df)
    n = len(c)
    r = rsi(c, n_rsi)
    a = atr(h, l, c, 14)
    L = np.zeros(n, bool)
    Sh = np.zeros(n, bool)
    sl, ss, tl, ts = (np.full(n, np.nan) for _ in range(4))
    for t in range(2 * N + n_rsi, n):
        i = t - N
        w = slice(i - N, t + 1)
        rw = r[w]
        cand_s = h[i] >= h[w].max() and np.any(rw > hi)
        cand_l = l[i] <= l[w].min() and np.any(rw < lo)
        if cand_s and cand_l:
            continue
        if cand_l and not np.all(c[i:t + 1] < o[i:t + 1]):
            j = i - N + int(np.argmax(rw < lo))
            if j - n_rsi >= 0:
                stop, tgt = l[j - n_rsi:t + 1].min() - 0.25 * a[t], h[i]
                if stop < c[t] < tgt:
                    L[t], sl[t], tl[t] = True, stop, tgt
        if cand_s and not np.all(c[i:t + 1] > o[i:t + 1]):
            j = i - N + int(np.argmax(rw > hi))
            if j - n_rsi >= 0:
                stop, tgt = h[j - n_rsi:t + 1].max() + 0.25 * a[t], l[i]
                if tgt < c[t] < stop:
                    Sh[t], ss[t], ts[t] = True, stop, tgt
    return {"long": L, "short": Sh, "stop_long": sl, "stop_short": ss, "tgt_long": tl, "tgt_short": ts}
