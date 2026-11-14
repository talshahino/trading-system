"""VuManChu indicators (task tal1411ind 12:47-12:50). Closed bars only - no repaint.

cipher_b - VuManChu Cipher B (v4, open source https://www.tradingview.com/script/Msm4SjwI-VuManChu-Cipher-B-Divergences/,
code mirror https://gist.github.com/itsXactlY/a7ddc99055e942006dd945d5c04dd759): WaveTrend of hlc3 (channel 9, average
12, signal SMA 3); green buy dot = wt1 / wt2 cross up with wt2 <= -53, red sell dot = cross down with wt2 >= 53;
RSI+MFI area = SMA(((close - open) / (high - low)) x 150, 60) - 2.5 (the v2 multiplier 150; one v4 copy uses 175);
WaveTrend divergences from 5-bar fractals of wt2 confirmed 2 bars later (top >= 45, bottom <= -65): bearish = a higher
price high with a lower wt2 than the previous top fractal, bullish mirror.

range_filter - VuManChu Swing Free = DonovanWall's Range Filter (https://in.tradingview.com/script/q23anHmP-VuManChu-Swing-Free):
smooth range = EMA(EMA(|close - close[1]|, n), 2n - 1) x mult (n 20, mult 3.5 - the Swing Free defaults); the filter
ratchets with the close; buy = close over the filter while it rises, after a sell state; sell mirror."""
import numpy as np

from agents.indicators._core import cols, cross_down, cross_up, ema, shift, sma


def cipher_b(df, ob=53.0, os_=-53.0, mf_mult=150.0):
    """dict: wt1, wt2, buy, sell (the dots), rsimfi, bull_div, bear_div."""
    o, h, l, c, v = cols(df)
    src = (h + l + c) / 3
    esa = ema(src, 9)
    de = ema(np.abs(src - esa), 9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(de > 0, (src - esa) / (0.015 * de), 0.0)
    wt1 = ema(ci, 12)
    wt2 = sma(wt1, 3)
    up = cross_up(wt1, wt2)
    dn = cross_down(wt1, wt2)
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.where(rng > 0, (c - o) / rng, 0.0)
    rsimfi = sma(cl * mf_mult, 60) - 2.5
    n = len(c)
    bull_div, bear_div = np.zeros(n, bool), np.zeros(n, bool)
    last_top = last_bot = None          # (wt2, high / low) of the previous fractal
    for t in range(4, n):
        w = wt2
        if not np.all(np.isfinite(w[t - 4:t + 1])):
            continue
        top = w[t - 4] < w[t - 2] and w[t - 3] < w[t - 2] and w[t - 2] > w[t - 1] and w[t - 2] > w[t]
        bot = w[t - 4] > w[t - 2] and w[t - 3] > w[t - 2] and w[t - 2] < w[t - 1] and w[t - 2] < w[t]
        if top:
            if last_top is not None and w[t - 2] >= 45 and h[t - 2] > last_top[1] and w[t - 2] < last_top[0]:
                bear_div[t] = True
            last_top = (w[t - 2], h[t - 2])
        if bot:
            if last_bot is not None and w[t - 2] <= -65 and l[t - 2] < last_bot[1] and w[t - 2] > last_bot[0]:
                bull_div[t] = True
            last_bot = (w[t - 2], l[t - 2])
    return {"wt1": wt1, "wt2": wt2, "buy": up & (wt2 <= os_), "sell": dn & (wt2 >= ob), "rsimfi": rsimfi,
            "bull_div": bull_div, "bear_div": bear_div}


def range_filter(df, n=20, mult=3.5):
    """dict: filt, hband, lband, buy, sell, state (+1 / -1)."""
    o, h, l, c, v = cols(df)
    x = c
    rng = ema(ema(np.abs(x - shift(x, 1)), n), 2 * n - 1) * mult
    rng = np.nan_to_num(rng)
    N = len(x)
    f = np.zeros(N)
    f[0] = x[0] if N else 0.0
    upw = np.zeros(N)
    dnw = np.zeros(N)
    st = np.zeros(N)
    buy, sell = np.zeros(N, bool), np.zeros(N, bool)
    for t in range(1, N):
        r = rng[t]
        if x[t] > f[t - 1]:
            f[t] = f[t - 1] if x[t] - r < f[t - 1] else x[t] - r
        else:
            f[t] = f[t - 1] if x[t] + r > f[t - 1] else x[t] + r
        upw[t] = upw[t - 1] + 1 if f[t] > f[t - 1] else (0 if f[t] < f[t - 1] else upw[t - 1])
        dnw[t] = dnw[t - 1] + 1 if f[t] < f[t - 1] else (0 if f[t] > f[t - 1] else dnw[t - 1])
        lc = x[t] > f[t] and upw[t] > 0
        sc = x[t] < f[t] and dnw[t] > 0
        st[t] = 1 if lc else (-1 if sc else st[t - 1])
        buy[t] = lc and st[t - 1] == -1
        sell[t] = sc and st[t - 1] == 1
    warm = min(N, 3 * n)
    buy[:warm] = False
    sell[:warm] = False
    return {"filt": f, "hband": f + rng, "lband": f - rng, "buy": buy, "sell": sell, "state": st}
