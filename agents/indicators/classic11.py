"""Scout 19.11.2026: indicators found on TradingView, in the NNFX community, in Larry Williams' and Oliver Velez's methods.

None of the TradingView pages below showed its Pine code to the scout, so every one of these is OUR OWN RECONSTRUCTION from the
author's published description (lengths and steps the page states are kept, the rest is marked OURS):
- tdfi: Trend Direction Force Index (the NNFX volume indicator; standard MT4 formula), used in
  https://www.tradingview.com/script/wmJpD8xR-No-Nonsense-NNFX-VP-Strategy-for-Back-Testing-jh .
- lwti: Larry Williams Large Trade Index as described in https://www.tradingview.com/script/LpO7kcS9 (differences over 25 bars,
  smoothed over 5). OURS: scaled by ATR around 50.
- velez_bars: elephant bar, tail bar, 180 and narrow state, https://br.tradingview.com/script/MTQNwtFZ-Oliver-Velez-System-Visual-Helper-CocoChoco/ .
  OURS: all the size thresholds (the page gives none), measured on candles of n bars.
- vcp: Minervini's Volatility Contraction Pattern, https://www.luxalgo.com/library/concept/volatility-contraction-pattern.md .
  OURS: 5-bar pivots, 3 legs each shallower than the last.
- frequency_momentum: https://www.tradingview.com/script/ciGgoWUr-Frequency-Momentum-Oscillator-QuantAlgo (presets 32 / 3, harmonics 1-8).
- precision_trend: Ehlers, TASC 2024.09, https://www.tradingview.com/script/XxSVTg0v-TASC-2024-09-Precision-Trend-Analysis (250 / 40).
  OURS: Ehlers' standard two-pole high-pass; ROC scaled by period / 6.28.
- adaptive_resonance: https://www.tradingview.com/script/gqtSuLQt-Adaptive-Resonance-Oscillator-AlgoAlpha/ . OURS: Ehlers' homodyne
  discriminator for the cycle, half the cycle as the RSI length, HMA 5 smoothing.
- trend_reversal_prob: https://tradingview.com/script/7gnQovYe-Trend-Reversal-Probability-Algoalpha . OURS: SMA 20 / 50 of the midpoint
  (lengths not published), the probability = the empirical share of the last 100 finished trends shorter than the current one.
- trend_pulse: https://it.tradingview.com/script/J5o0F0GR-Trend-Pulse-BigBeluga/ . OURS: pivot 10, SMA length 20 growing by 1 a bar
  in the bear phase up to 200, offset 1.5 ATR.
- radius_trend: https://www.tradingview.com/script/4KIHf36j-Radius-Trend-ChartPrime (step 0.15, distance 2). OURS: the band's slope = step x the 100-bar range.
- poly_keltner: https://www.tradingview.com/script/NZAHz4G1-Polynomial-Regression-Keltner-Channel-ChartPrime (100 bars, HLC3). OURS: degree 2,
  the END POINT of each window's fit only (no redrawing of the past), band step 0.5 ATR scaled by ATR14 / ATR100.
- vol_gaussian_bands: https://www.tradingview.com/script/ZRD7xSGy-Volatility-Gaussian-Bands-BigBeluga . OURS: a one-sided Gaussian (causal), bands 1.5 x the 100-bar average range.
- prime_bands: https://my.tradingview.com/script/99KbMjoz-Prime-Bands-ChartPrime/ . OURS: base = the middle of the n-bar high / low, 1-3 SD of the close around it.
- kalman_trend_osc: https://it.tradingview.com/script/PDi6enZR-Adaptive-Kalman-filter-Trend-Strength-Oscillator-Zeiierman/ . OURS: level + velocity
  filter, noise in ATR units, measurement noise scaled by the bar's range (the page's Parkinson model), velocity scaled to +/-100.
- dynamic_range_filter: https://in.tradingview.com/script/vzKdIrEh-Dynamic-Range-Filter-with-Trend-Candlesticks-Zeiierman (HMA, the 2.618 multiplier). OURS: a ratchet on the bands.
- adaptive_st_osc: https://pl.tradingview.com/script/ABYvsqNS-Adaptive-SuperTrend-Oscillator-QuantAlgo/ . OURS: SuperTrend 10 x 3, ATR 7 / 28 blended by the 20-bar efficiency ratio.
- aco: https://kr.tradingview.com/script/vEMWSCP3-Adaptive-Composite-Oscillator-ACO/ . OURS: all lengths and the ADX 25 regime line.
- wavetrend_x: https://www.tradingview.com/script/EsEtVFFX-WaveTrend-X/ (40 / 16, Kalman Q 0.02 R 0.8, entropy window 40, 8 bins, rank 100, 82 / 18).
- zl_flow: https://tradingview.com/script/hpuNvNCf-TREND-and-ZL-FLOW (T3 8 / 0.7, zero-lag WMA 9).

No repaint: every value at bar i uses only bars <= i. Pivots count from the bar that confirms them, candles of n bars from the bar that completes them.
"""
from bisect import bisect_left, bisect_right, insort

import numpy as np

from agents.indicators._core import S, atr, cols, dmi, ema, hma, pivots, rma, sma, wma
from agents.indicators.classic5 import t3 as _t3
from agents.indicators.classic10 import agg_candles


def _x_up(a, b=0.0):
    a, b = np.asarray(a, float), np.broadcast_to(np.asarray(b, float), np.shape(a))
    return np.concatenate([[False], (a[1:] > b[1:]) & (a[:-1] <= b[:-1])])


def _x_dn(a, b=0.0):
    a, b = np.asarray(a, float), np.broadcast_to(np.asarray(b, float), np.shape(a))
    return np.concatenate([[False], (a[1:] < b[1:]) & (a[:-1] >= b[:-1])])


def _flips(t):
    t = np.asarray(t, float)
    p = np.concatenate([[0.0], t[:-1]])
    return (t > 0) & (p <= 0), (t < 0) & (p >= 0)


def tdfi(df, n=13, trig=0.05):
    """Trend Direction Force Index, -1..1. dict: tdfi, up (over +trig), dn (under -trig)."""
    c = cols(df)[3]
    m = ema(c * 1000.0, n)
    s = ema(m, n)
    im, iz = np.diff(m, prepend=m[0]), np.diff(s, prepend=s[0])
    tdf = np.abs(m - s) * ((im + iz) / 2.0) ** 3
    top = S(np.abs(tdf)).rolling(3 * n, min_periods=1).max().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.nan_to_num(np.where(top > 0, tdf / top, 0.0))
    return {"tdfi": x, "up": x > trig, "dn": x < -trig}


def lwti(df, n=25, sm=5):
    """Larry Williams Large Trade Index around 50. dict: lwti, signal."""
    o, h, l, c, v = cols(df)
    d = c - np.concatenate([np.full(n, np.nan), c[:-n]])
    a = atr(h, l, c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(a > 0, sma(d, n) / a * 50.0 + 50.0, 50.0)
    x = np.nan_to_num(x, nan=50.0)
    return {"lwti": x, "signal": sma(x, sm)}


def velez_bars(df, n=3, avg=20, narrow_pct=0.15):
    """Oliver Velez bars on candles of n bars (+1 / -1 on the bar that completes the candle). dict: elephant, tail, one80, narrow,
    bar_h, bar_l (the signal candle's high / low, carried forward)."""
    N = len(df)
    co, ch, cl, cc, _, pos = agg_candles(df, n)
    out = {k: np.zeros(N) for k in ("elephant", "tail", "one80")}
    bh, bl = np.full(N, np.nan), np.full(N, np.nan)
    if len(cc) > avg + 2:
        rng = ch - cl
        body = np.abs(cc - co)
        ab = S(body).shift(1).rolling(avg, min_periods=5).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_c = np.where(rng > 0, (cc - cl) / rng, 0.5)
        up_w, lo_w = ch - np.maximum(co, cc), np.minimum(co, cc) - cl
        el = np.where((body >= 1.5 * ab) & (cc > co) & (pos_c >= 0.7), 1.0, np.where((body >= 1.5 * ab) & (cc < co) & (pos_c <= 0.3), -1.0, 0.0))
        ta = np.where((lo_w >= 2 * body) & (lo_w >= 0.5 * rng) & (pos_c >= 0.6), 1.0, np.where((up_w >= 2 * body) & (up_w >= 0.5 * rng) & (pos_c <= 0.4), -1.0, 0.0))
        po, pc = np.concatenate([[np.nan], co[:-1]]), np.concatenate([[np.nan], cc[:-1]])
        pb = np.abs(pc - po)
        bull = (pc < po) & (pb >= ab) & (cc > co) & (cc >= po - 0.25 * pb) & (pos_c >= 0.7)
        bear = (pc > po) & (pb >= ab) & (cc < co) & (cc <= po + 0.25 * pb) & (pos_c <= 0.3)
        out["elephant"][pos], out["tail"][pos], out["one80"][pos] = el, ta, np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
        bh[pos], bl[pos] = ch, cl
    c = cols(df)[3]
    s20, s200 = sma(c, 20), sma(c, 200)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["narrow"] = np.nan_to_num(np.abs(s20 - s200) / c * 100.0, nan=9.9) <= narrow_pct
    out["bar_h"], out["bar_l"] = S(bh).ffill().to_numpy(), S(bl).ffill().to_numpy()
    return out


def vcp(df, k=5, legs=3, vol_n=50):
    """Volatility Contraction Pattern. dict: contractions (shrinking legs in a row), pivot, base_low, ready, buy."""
    o, h, l, c, v = cols(df)
    N = len(c)
    ph_conf, pl_conf, ph_val, pl_val = pivots(h, l, k)[:4]
    cnt, piv, low = np.zeros(N), np.full(N, np.nan), np.full(N, np.nan)
    ready, buy = np.zeros(N, bool), np.zeros(N, bool)
    s50, vm = sma(c, 50), sma(v, vol_n)
    last_h, last_d, run, cur_p, cur_l, armed = np.nan, np.nan, 0, np.nan, np.nan, False
    for i in range(N):
        if ph_conf[i]:
            last_h = ph_val[i]
        if pl_conf[i] and last_h == last_h and last_h > 0:
            d = (last_h - pl_val[i]) / last_h
            if d > 0:
                run = run + 1 if (last_d == last_d and d < last_d) else 1
                last_d, cur_p, cur_l = d, last_h, pl_val[i]
                armed = run >= legs
        if armed and c[i] < cur_l:
            armed, run, last_d = False, 0, np.nan
        cnt[i], piv[i], low[i], ready[i] = run, cur_p, cur_l, armed
        if armed and c[i] > cur_p and c[i] > s50[i] and v[i] >= vm[i]:
            buy[i], armed, run, last_d = True, False, 0, np.nan
    return {"contractions": cnt, "pivot": piv, "base_low": low, "ready": ready, "buy": buy}


def frequency_momentum(df, n=32, sm=3):
    """Spectral balance (low minus high harmonics, % of power) x the momentum's sign. dict: fmo, low, high, buy, sell."""
    c = cols(df)[3]
    N = len(c)
    x = np.nan_to_num(c - sma(c, n))
    lo, hi = np.zeros(N), np.zeros(N)
    if N >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        t = 2.0 * np.pi * np.arange(n) / n
        pw = np.stack([(w @ np.cos(k * t)) ** 2 + (w @ np.sin(k * t)) ** 2 for k in range(1, 9)], axis=1)
        tot = pw.sum(axis=1)
        tot[tot == 0] = 1.0
        lo[n - 1:], hi[n - 1:] = 100.0 * pw[:, :2].sum(axis=1) / tot, 100.0 * pw[:, 4:].sum(axis=1) / tot
    d = c - np.concatenate([np.full(n, np.nan), c[:-n]])
    f = ema(np.nan_to_num((ema(lo, sm) - ema(hi, sm)) * np.sign(np.nan_to_num(d))), sm)
    return {"fmo": f, "low": lo, "high": hi, "buy": _x_up(f), "sell": _x_dn(f)}


def _hp2(x, p):
    r = 0.707 * 2.0 * np.pi / p
    a = (np.cos(r) + np.sin(r) - 1.0) / np.cos(r)
    b, c1, c2 = (1 - a / 2.0) ** 2, 2 * (1 - a), (1 - a) ** 2
    y = np.zeros(len(x))
    for i in range(2, len(x)):
        y[i] = b * (x[i] - 2 * x[i - 1] + x[i - 2]) + c1 * y[i - 1] - c2 * y[i - 2]
    return y


def precision_trend(df, slow=250, fast=40):
    """Ehlers Precision Trend Analysis. dict: trend (HP slow - HP fast), roc, up, dn."""
    c = cols(df)[3]
    t = _hp2(c, slow) - _hp2(c, fast)
    r = (fast / 6.28) * np.diff(t, prepend=t[0])
    return {"trend": t, "roc": r, "up": _x_up(r), "dn": _x_dn(r)}


def _homodyne(p):
    """Ehlers' homodyne dominant cycle (6..50 bars)."""
    N = len(p)
    sm, de, q1, i1, i2, q2, re, im, per = (np.zeros(N) for _ in range(9))
    per[:7] = 15.0
    ht = lambda a, i, k: (0.0962 * a[i] + 0.5769 * a[i - 2] - 0.5769 * a[i - 4] - 0.0962 * a[i - 6]) * k
    for i in range(N):
        if i >= 3:
            sm[i] = (4 * p[i] + 3 * p[i - 1] + 2 * p[i - 2] + p[i - 3]) / 10.0
        if i < 7:
            continue
        k = 0.075 * per[i - 1] + 0.54
        de[i] = ht(sm, i, k)
        q1[i], i1[i] = ht(de, i, k), de[i - 3]
        ji, jq = ht(i1, i, k), ht(q1, i, k)
        i2[i] = 0.2 * (i1[i] - jq) + 0.8 * i2[i - 1]
        q2[i] = 0.2 * (q1[i] + ji) + 0.8 * q2[i - 1]
        re[i] = 0.2 * (i2[i] * i2[i - 1] + q2[i] * q2[i - 1]) + 0.8 * re[i - 1]
        im[i] = 0.2 * (i2[i] * q2[i - 1] - q2[i] * i2[i - 1]) + 0.8 * im[i - 1]
        x = 2.0 * np.pi / np.arctan(im[i] / re[i]) if (im[i] != 0 and re[i] != 0) else per[i - 1]
        x = min(max(x, 0.67 * per[i - 1], 6.0), 1.5 * per[i - 1], 50.0)
        per[i] = 0.2 * x + 0.8 * per[i - 1]
    return per


def _rsi_var(c, ln):
    """Wilder RSI whose length changes bar by bar."""
    N = len(c)
    up, dn, r = 0.0, 0.0, np.full(N, 50.0)
    for i in range(1, N):
        d, a = c[i] - c[i - 1], 1.0 / max(ln[i], 2.0)
        up, dn = (1 - a) * up + a * max(d, 0.0), (1 - a) * dn + a * max(-d, 0.0)
        r[i] = 100.0 * up / (up + dn) if up + dn > 0 else 50.0
    return r


def adaptive_resonance(df, sm=5):
    """RSI tuned to half the dominant cycle. dict: aro, signal, cycle, up, dn."""
    o, h, l, c, v = cols(df)
    per = _homodyne((h + l) / 2.0)
    x = np.nan_to_num(hma(_rsi_var(c, per / 2.0), sm), nan=50.0)
    g = ema(x, 5)
    return {"aro": x, "signal": g, "cycle": per, "up": _x_up(x, g) & (x < 50), "dn": _x_dn(x, g) & (x > 50)}


def trend_reversal_prob(df, fast=20, slow=50, hist=100):
    """Probability the current trend is older than past trends. dict: prob 0..1, dir, age."""
    o, h, l, c, v = cols(df)
    m = (h + l) / 2.0
    d = np.sign(np.nan_to_num(sma(m, fast) - sma(m, slow)))
    N = len(c)
    pr, age = np.zeros(N), np.zeros(N)
    past, ordered, cur = [], [], 0
    for i in range(1, N):
        if d[i] != 0 and d[i - 1] != 0 and d[i] != d[i - 1]:
            insort(ordered, cur)
            past.append(cur)
            if len(past) > hist:
                ordered.pop(bisect_left(ordered, past.pop(0)))
            cur = 0
        cur += 1
        age[i] = cur
        pr[i] = bisect_right(ordered, cur) / len(ordered) if len(ordered) >= 5 else 0.0
    return {"prob": pr, "dir": d, "age": age}


def trend_pulse(df, piv=10, base=20, mult=1.5):
    """Pivot-low break = bear, a close over the widening band = bull. dict: trend, band, pivot_low, buy, sell."""
    o, h, l, c, v = cols(df)
    N = len(c)
    pl_conf, pl_val = pivots(h, l, piv)[1], pivots(h, l, piv)[3]
    a = atr(h, l, c, 14)
    cum = np.concatenate([[0.0], np.cumsum((h + l) / 2.0)])
    t, band, pl = np.ones(N), np.full(N, np.nan), np.full(N, np.nan)
    ln, cur = base, np.nan
    for i in range(1, N):
        if pl_conf[i]:
            cur = pl_val[i]
        pl[i], t[i] = cur, t[i - 1]
        ln = min(ln + 1, 200) if t[i] < 0 else base
        w = min(ln, i + 1)
        band[i] = (cum[i + 1] - cum[i + 1 - w]) / w + mult * a[i]
        if t[i] > 0 and cur == cur and c[i] < cur:
            t[i], ln = -1.0, base
        elif t[i] < 0 and c[i] > band[i]:
            t[i], ln = 1.0, base
    b, s = _flips(t)
    return {"trend": t, "band": band, "pivot_low": pl, "buy": b, "sell": s}


def radius_trend(df, step=0.15, dist=2.0, n=100):
    """A band climbing (falling) by step x the average range each bar, reset on a cross. dict: trend, band, buy, sell."""
    o, h, l, c, v = cols(df)
    N = len(c)
    r = S(h - l).rolling(n, min_periods=1).mean().to_numpy()
    t, b = np.ones(N), np.full(N, np.nan)
    if N:
        b[0] = c[0] - dist * r[0]
    for i in range(1, N):
        t[i], b[i] = t[i - 1], b[i - 1] + t[i - 1] * step * r[i] * 0.1
        if t[i] > 0 and c[i] < b[i]:
            t[i], b[i] = -1.0, c[i] + dist * r[i]
        elif t[i] < 0 and c[i] > b[i]:
            t[i], b[i] = 1.0, c[i] - dist * r[i]
    u, d = _flips(t)
    return {"trend": t, "band": b, "buy": u, "sell": d}


def poly_keltner(df, n=100, deg=2, step=0.5):
    """Polynomial regression end point + 4 ATR bands each side. dict: mid, slope, up1-up4, dn1-dn4."""
    o, h, l, c, v = cols(df)
    p = (h + l + c) / 3.0
    x = np.linspace(-1.0, 1.0, n)
    X = np.vander(x, deg + 1)
    w = np.linalg.pinv(X)
    wt = np.array([1.0 ** k for k in range(deg, -1, -1)]) @ w
    mid = np.full(len(c), np.nan)
    if len(c) >= n:
        mid[n - 1:] = np.lib.stride_tricks.sliding_window_view(p, n) @ wt
    a14, a100 = atr(h, l, c, 14), atr(h, l, c, 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.nan_to_num(a14 / a100, nan=1.0)
    out = {"mid": mid, "slope": np.diff(mid, prepend=np.nan)}
    for k in range(1, 5):
        out["up%d" % k], out["dn%d" % k] = mid + k * step * a14 * f, mid - k * step * a14 * f
    return out


def vol_gaussian_bands(df, n=20, mult=1.5, vol_n=100):
    """Causal Gaussian midline with volatility bands. dict: mid, up, lo, trend, buy, sell."""
    o, h, l, c, v = cols(df)
    N = len(c)
    g = np.exp(-0.5 * (np.arange(n)[::-1] / (n / 3.0)) ** 2)
    g /= g.sum()
    mid = np.full(N, np.nan)
    if N >= n:
        mid[n - 1:] = np.lib.stride_tricks.sliding_window_view(c, n) @ g
    r = S(h - l).rolling(vol_n, min_periods=10).mean().to_numpy()
    up, lo = mid + mult * r, mid - mult * r
    t = np.zeros(N)
    for i in range(1, N):
        t[i] = 1.0 if c[i] > up[i] else -1.0 if c[i] < lo[i] else t[i - 1]
    b, s = _flips(t)
    return {"mid": mid, "up": up, "lo": lo, "trend": t, "buy": b, "sell": s}


def prime_bands(df, n=50):
    """Mid of the n-bar range with 1-3 SD bands. dict: mid, u1-u3, l1-l3, rev_up, rev_dn (back inside the 2 SD band)."""
    o, h, l, c, v = cols(df)
    mid = (S(h).rolling(n).max().to_numpy() + S(l).rolling(n).min().to_numpy()) / 2.0
    sd = np.sqrt(S((c - mid) ** 2).rolling(n).mean().to_numpy())
    out = {"mid": mid}
    for k in (1, 2, 3):
        out["u%d" % k], out["l%d" % k] = mid + k * sd, mid - k * sd
    out["rev_up"], out["rev_dn"] = _x_up(c, out["l2"]), _x_dn(c, out["u2"])
    return out


def kalman_trend_osc(df, q_level=0.05, q_vel=0.01, r=1.0, look=10, sm=5):
    """Level + velocity Kalman filter. dict: osc (+/-100), strength, level, up, dn."""
    o, h, l, c, v = cols(df)
    N = len(c)
    a = np.nan_to_num(atr(h, l, c, 14), nan=0.0)
    ar = S(h - l).rolling(50, min_periods=1).mean().to_numpy()
    lv, ve = np.zeros(N), np.zeros(N)
    if N:
        lv[0] = c[0]
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    for i in range(1, N):
        u = max(a[i], 1e-12)
        x0, x1 = lv[i - 1] + ve[i - 1], ve[i - 1]
        P = np.array([[P[0, 0] + 2 * P[0, 1] + P[1, 1] + (q_level * u) ** 2, P[0, 1] + P[1, 1]],
                      [P[0, 1] + P[1, 1], P[1, 1] + (q_vel * u) ** 2]])
        rr = (r * u * max((h[i] - l[i]) / ar[i], 0.25) if ar[i] > 0 else r * u) ** 2
        s = P[0, 0] + rr
        k0, k1 = P[0, 0] / s, P[1, 0] / s
        e = c[i] - x0
        lv[i], ve[i] = x0 + k0 * e, x1 + k1 * e
        P = np.array([[(1 - k0) * P[0, 0], (1 - k0) * P[0, 1]], [P[1, 0] - k1 * P[0, 0], P[1, 1] - k1 * P[0, 1]]])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.clip(np.nan_to_num(ve / a) * 400.0, -100.0, 100.0)
    st = ema(x - np.concatenate([np.full(look, 0.0), x[:-look]]), sm)
    return {"osc": x, "strength": st, "level": lv, "up": _x_up(x), "dn": _x_dn(x)}


def dynamic_range_filter(df, n=50, mult=1.0):
    """HMA with a 2.618 x smoothed-range band and a ratcheting filter line. dict: filt, hi, lo, trend, buy, sell."""
    o, h, l, c, v = cols(df)
    N = len(c)
    m = hma(c, n)
    rg = ema(ema(h, n // 2) - ema(l, n // 2), n // 2) * 2.618 * mult / 2.0
    f, t = np.full(N, np.nan), np.zeros(N)
    for i in range(1, N):
        if m[i] != m[i] or rg[i] != rg[i]:
            continue
        p = f[i - 1] if f[i - 1] == f[i - 1] else m[i]
        f[i] = max(p, m[i] - rg[i]) if m[i] > p else min(p, m[i] + rg[i])
        t[i] = 1.0 if f[i] > p else -1.0 if f[i] < p else t[i - 1]
    b, s = _flips(t)
    return {"filt": f, "hi": f + rg, "lo": f - rg, "trend": t, "buy": b, "sell": s}


def adaptive_st_osc(df, n=10, mult=3.0, er_n=20, neutral=25.0):
    """Close minus SuperTrend in % of an efficiency-blended ATR. dict: score, line, trend, up2, dn2, buy, sell."""
    o, h, l, c, v = cols(df)
    N = len(c)
    a = atr(h, l, c, n)
    hl = (h + l) / 2.0
    up, dn, t, line = hl - mult * a, hl + mult * a, np.ones(N), np.full(N, np.nan)
    for i in range(1, N):
        up[i] = max(up[i], up[i - 1]) if c[i - 1] > up[i - 1] else up[i]
        dn[i] = min(dn[i], dn[i - 1]) if c[i - 1] < dn[i - 1] else dn[i]
        t[i] = 1.0 if c[i] > dn[i - 1] else -1.0 if c[i] < up[i - 1] else t[i - 1]
        line[i] = up[i] if t[i] > 0 else dn[i]
    d = np.abs(c - np.concatenate([np.full(er_n, np.nan), c[:-er_n]]))
    path = S(np.abs(np.diff(c, prepend=c[0]))).rolling(er_n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.clip(np.nan_to_num(d / path), 0.0, 1.0)
        sc = np.nan_to_num(100.0 * (c - line) / (er * atr(h, l, c, 7) + (1 - er) * atr(h, l, c, 28)))
    sd = S(sc).rolling(100, min_periods=20).std().to_numpy()
    return {"score": sc, "line": line, "trend": t, "up2": sc > 2 * sd, "dn2": sc < -2 * sd,
            "buy": _x_up(sc, neutral), "sell": _x_dn(sc, -neutral)}


def aco(df, n=14, z_n=50, adx_line=25.0):
    """Adaptive Composite Oscillator. dict: aco, z, adx, long, short."""
    o, h, l, c, v = cols(df)
    a, am = atr(h, l, c, 14), atr(h, l, c, 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.clip(np.nan_to_num(n * am / a, nan=float(n)), n / 2.0, n * 2.0)
    r = _rsi_var(c, ln)
    N = len(c)
    k = np.full(N, 50.0)
    fa, sl = 2.0 / 3.0, 2.0 / 31.0
    for i in range(10, N):
        vol = np.abs(np.diff(r[i - 10:i + 1])).sum()
        er = abs(r[i] - r[i - 10]) / vol if vol > 0 else 0.0
        k[i] = k[i - 1] + (er * (fa - sl) + sl) ** 2 * (r[i] - k[i - 1])
    m, s = S(k - 50.0).rolling(z_n).mean().to_numpy(), S(k - 50.0).rolling(z_n).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.nan_to_num((k - 50.0 - m) / s)
    p, q, ax = dmi(h, l, c, 14)
    rg, vm = ax < adx_line, v >= sma(v, 20)
    lg = (rg & _x_up(z, -2.0)) | (~rg & _x_up(z) & (p > q))
    sh = (rg & _x_dn(z, 2.0)) | (~rg & _x_dn(z) & (q > p))
    return {"aco": k, "z": z, "adx": ax, "long": lg & vm, "short": sh & vm}


def wavetrend_x(df, n=40, m=16, q=0.02, r=0.8, ent_n=40, bins=8, rank_n=100):
    """WaveTrend with a Kalman line and an entropy gate. dict: wt1, wt2, entropy, rank, cross_up, cross_dn."""
    o, h, l, c, v = cols(df)
    p = (h + l + c) / 3.0
    sd = S(p).rolling(n).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.nan_to_num((p - ema(p, n)) / sd)
    a, am = atr(h, l, c, 14), atr(h, l, c, 100)
    N = len(c)
    w1, pv = np.zeros(N), 1.0
    for i in range(1, N):
        qq = q * (a[i] / am[i] if am[i] > 0 and a[i] == a[i] else 1.0)
        pv += qq
        k = pv / (pv + r)
        w1[i] = w1[i - 1] + k * (ci[i] - w1[i - 1])
        pv *= 1 - k
    w2 = ema(w1, m)
    ret = np.diff(c, prepend=c[0])
    ent = np.ones(N)
    for i in range(ent_n, N):
        cnt = np.histogram(ret[i - ent_n + 1:i + 1], bins=bins)[0].astype(float)
        pr = cnt[cnt > 0] / cnt.sum()
        ent[i] = float(-(pr * np.log(pr)).sum() / np.log(bins))
    rk = S(w1).rolling(rank_n, min_periods=20).rank(pct=True).to_numpy() * 100.0
    return {"wt1": w1, "wt2": w2, "entropy": ent, "rank": rk, "cross_up": _x_up(w1, w2), "cross_dn": _x_dn(w1, w2)}


def zl_flow(df, t3_n=8, b=0.7, zl_n=9):
    """T3 slope + zero-lag WMA slope. dict: t3, zl, state (+1 both up, -1 both down, 0 mixed), buy, sell."""
    c = cols(df)[3]
    t = _t3(df, t3_n, b)
    w = wma(c, zl_n)
    z = 2.0 * w - wma(w, zl_n)
    st = np.where((np.diff(t, prepend=np.nan) > 0) & (np.diff(z, prepend=np.nan) > 0), 1.0,
                  np.where((np.diff(t, prepend=np.nan) < 0) & (np.diff(z, prepend=np.nan) < 0), -1.0, 0.0))
    u, d = _flips(st)
    return {"t3": t, "zl": z, "state": st, "buy": u, "sell": d}
