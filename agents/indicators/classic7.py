"""Indicators the library lacked (scout 17.11.2026, night). Each is a port of the published formula, or our own
reconstruction where the code / formula is closed (said so per indicator). No repaint: every value at bar i uses bars
<= i only (the cumulative mean of the Hull Butterfly levels is an expanding mean, never centred).

- voss_predict: Ehlers Voss Predictive Filter (TASC 2019.08 'A Peek Into The Future',
  https://www.traders.com/Documentation/FEEDbk_docs/2019/08/TradersTips.html ; open source
  https://www.tradingview.com/script/ekkb1vlg-Voss-Predictor-A-Peek-Into-the-Future-Dr-John-Ehlers ): two-pole band-pass
  (period 20, bandwidth 0.25) and the Voss predictor (predict 3 bars, order 9). Voss crossing Filt = early turn.
- super_passband: Ehlers Super Passband Filter (TASC 2016.07,
  https://www.traders.com/Documentation/FEEDbk_docs/2016/07/TradersTips.html ): difference of two zero-lag EMAs (40 / 60)
  in one recursion, with its 50-bar RMS band. Buy = PB crosses up through -RMS, sell = crosses down through +RMS.
- universal_osc: Ehlers Universal Oscillator (TASC 2015.01,
  https://www.traders.com/Documentation/FEEDbk_docs/2015/01/TradersTips.html ): SuperSmoother 20 of the 2-bar 'white
  noise' (c - c[2]) / 2, normalised by a fast-attack / slow-decay peak (0.991). -1..1, zero crosses.
- elegant_osc: Ehlers Elegant Oscillator (TASC 2022.02 'Inverse Fisher Transform Redux',
  https://www.traders.com/Documentation/FEEDbk_docs/2022/02/TradersTips.html ): 2-bar derivative / its 50-bar RMS, the
  inverse Fisher (tanh) of it, SuperSmoother 20. Turns out of -0.5 / +0.5 = cycle lows / highs.
- dsma: Ehlers Deviation-Scaled Moving Average (TASC 2018.07,
  https://www.traders.com/Documentation/FEEDbk_docs/2018/07/TradersTips.html ): EMA whose alpha = 5 x |scaled filter| /
  period (period 40); the filter = SuperSmoother (period / 2) of the 2-bar zeros, scaled by its RMS.
- trama: LuxAlgo Trend Regularity Adaptive Moving Average (open source,
  https://www.tradingview.com/script/p8wGCPi6-Trend-Regularity-Adaptive-Moving-Average-LuxAlgo ): tc = SMA(new highest
  high or new lowest low, 99)^2, ama += tc x (close - ama).
- impulse_macd: LazyBear Impulse MACD (open source https://www.tradingview.com/script/qt6xLfLi-Impulse-MACD-LazyBear ):
  hi / lo = SMMA 34 of highs / lows, mi = ZLEMA 34 of hlc3; md = mi - hi over the band, mi - lo under it, 0 inside;
  signal SMA 9, histogram md - signal.
- mad: Ehlers Moving Average Difference (TASC 2021.10 'MAD Moving Average Difference',
  https://www.traders.com/Documentation/FEEDbk_docs/2021/10/TradersTips.html ): 100 x (SMA 8 - SMA 23) / SMA 23.
- spearman: Spearman rank correlation of the last n closes with time (Ehlers, Cycle Analytics for Traders ch. 'Spearman
  indicator'; https://www.tradingview.com/script/rpKPpW51-Ehlers-Spearman-Rank-Indicator-CC ), n 20, -1..1.
- market_thermometer: Elder Market Thermometer (Come Into My Trading Room, 2002;
  https://www.incrediblecharts.com/indicators/elder_market_thermometer.php ): max(h - h[1], l[1] - l), 0 on inside bars;
  EMA 22; 'hot' = thermometer > 3 x EMA (take profit), 'quiet' = under the EMA for 5+ bars (breakout ahead).
- williams_ad: Larry Williams Accumulation / Distribution (https://www.metastock.com/customer/resources/taaz/?p=125 ):
  cumulative close minus the true low (up closes) / true high (down closes).
- fib_bollinger: Fibonacci Bollinger Bands (Rashad, open source
  https://in.tradingview.com/script/qIKR3tbN-Fibonacci-Bollinger-Bands ): VWMA 200 of hlc3 +/- 0.236 .. 1.0 x 3 stdev.
- hull_butterfly: Hull Butterfly Oscillator (LuxAlgo, https://www.luxalgo.com/library/indicator/hull-butterfly-oscillator/ ;
  our reconstruction - the page gives no formula): the HMA's own kernel weights (length 14) applied to the price minus
  the same weights mirrored in time; levels = expanding mean of |osc| x 1.5; dots = turns beyond the levels.
- range_detector: Range Detector (LuxAlgo, https://www.luxalgo.com/library/indicator/range-detector/ ; our
  reconstruction of the description): every close of the last 20 bars within SMA 20 +/- ATR(500) x 1.0 -> a range
  (top / bottom = SMA +/- width, extended while it holds, merged with an overlapping unbroken range); break = close beyond.
"""
import math

import numpy as np

from agents.indicators._core import S, atr, cols, ema, hma, onset, rma, shift, sma, stdev


def _super_smoother(x, period):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    a1 = math.exp(-1.414 * math.pi / period)
    c2 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    f = np.zeros(len(x))
    for t in range(len(x)):
        if t < 2:
            f[t] = x[t]
            continue
        f[t] = c1 * (x[t] + x[t - 1]) / 2 + c2 * f[t - 1] + c3 * f[t - 2]
    return f


def voss_predict(df, period=20, predict=3, bandwidth=0.25):
    _, _, _, c, _ = cols(df)
    n = len(c)
    order = 3 * predict
    f1 = math.cos(2 * math.pi / period)
    g1 = math.cos(bandwidth * 2 * math.pi / period)
    s1 = 1 / g1 - math.sqrt(1 / (g1 * g1) - 1)
    filt = np.zeros(n)
    voss = np.zeros(n)
    for t in range(n):
        if t >= 5:
            filt[t] = 0.5 * (1 - s1) * (c[t] - c[t - 2]) + f1 * (1 + s1) * filt[t - 1] - s1 * filt[t - 2]
        if t > order:
            sumc = sum(((k + 1) / order) * voss[t - (order - k)] for k in range(order))
            voss[t] = ((3 + order) / 2) * filt[t] - sumc
    warm = np.arange(n) < max(6 * period, 60)
    filt[warm] = np.nan
    voss[warm] = np.nan
    return {"filt": filt, "voss": voss}


def super_passband(df, p1=40, p2=60, rms_n=50):
    _, _, _, c, _ = cols(df)
    n = len(c)
    a1, a2 = 5.0 / p1, 5.0 / p2
    pb = np.zeros(n)
    for t in range(n):
        if t < 2:
            continue
        pb[t] = ((a1 - a2) * c[t] + (a2 * (1 - a1) - a1 * (1 - a2)) * c[t - 1]
                 + ((1 - a1) + (1 - a2)) * pb[t - 1] - (1 - a1) * (1 - a2) * pb[t - 2])
    rms = np.sqrt(S(pb * pb).rolling(rms_n).mean().to_numpy())
    pb = np.where(np.arange(n) < p2, np.nan, pb)
    return {"pb": pb, "rms": rms}


def universal_osc(df, band_edge=20):
    _, _, _, c, _ = cols(df)
    n = len(c)
    wn = (c - shift(c, 2)) / 2
    filt = _super_smoother(np.nan_to_num(wn), band_edge)
    peak = np.zeros(n)
    out = np.full(n, np.nan)
    for t in range(n):
        p = abs(filt[t])
        peak[t] = max(0.991 * peak[t - 1], p) if t else p
        if t >= 2 * band_edge and peak[t] > 0:
            out[t] = filt[t] / peak[t]
    return out


def elegant_osc(df, rms_n=50, smooth=20):
    _, _, _, c, _ = cols(df)
    d = np.nan_to_num(c - shift(c, 2))
    rms = np.sqrt(S(d * d).rolling(rms_n).mean().to_numpy())
    nd = np.where(rms > 0, d / rms, 0.0)
    ifish = np.tanh(np.nan_to_num(nd))
    eo = _super_smoother(ifish, smooth)
    eo[np.arange(len(c)) < rms_n + smooth] = np.nan
    return eo


def dsma(df, period=40):
    _, _, _, c, _ = cols(df)
    n = len(c)
    zeros = np.nan_to_num(c - shift(c, 2))
    filt = _super_smoother(zeros, max(period / 2.0, 2))
    rms = np.sqrt(S(filt * filt).rolling(period).mean().to_numpy())
    sf = np.where(rms > 0, filt / rms, 0.0)
    out = np.full(n, np.nan)
    prev = c[0] if n else np.nan
    for t in range(n):
        a = 5 * abs(np.nan_to_num(sf[t])) / period
        a = min(max(a, 0.0), 1.0)
        prev = a * c[t] + (1 - a) * prev
        out[t] = prev
    out[: period] = np.nan
    return out


def trama(df, length=99):
    _, h, l, c, _ = cols(df)
    n = len(c)
    hh = S(h).rolling(length, min_periods=1).max().to_numpy()
    ll = S(l).rolling(length, min_periods=1).min().to_numpy()
    up = np.nan_to_num(np.sign(np.diff(hh, prepend=hh[:1]))) > 0
    dn = np.nan_to_num(np.sign(np.diff(ll, prepend=ll[:1]))) < 0
    tc = S((up | dn).astype(float)).rolling(length).mean().to_numpy() ** 2
    out = np.full(n, np.nan)
    ama = c[0] if n else np.nan
    for t in range(n):
        k = 0.0 if not np.isfinite(tc[t]) else tc[t]
        ama = ama + k * (c[t] - ama)
        out[t] = ama
    out[: length] = np.nan
    return {"trama": out, "tc": tc}


def impulse_macd(df, length=34, signal=9):
    _, h, l, c, _ = cols(df)
    src = (h + l + c) / 3
    hi = rma(h, length)                     # SMMA = Wilder / RMA
    lo = rma(l, length)
    e1 = ema(src, length)
    e2 = ema(e1, length)
    mi = e1 + (e1 - e2)                     # ZLEMA as LazyBear codes it (DEMA-style)
    md = np.where(mi > hi, mi - hi, np.where(mi < lo, mi - lo, 0.0))
    md[: 2 * length] = np.nan
    sb = sma(md, signal)
    return {"md": md, "signal": sb, "hist": md - sb, "mi": mi, "hi": hi, "lo": lo}


def mad(df, fast=8, slow=23):
    _, _, _, c, _ = cols(df)
    s = sma(c, slow)
    return 100 * (sma(c, fast) - s) / s


def spearman(df, n=20):
    _, _, _, c, _ = cols(df)
    t = np.arange(n, dtype=float)
    tr = t - t.mean()

    def _rho(v):
        rk = S(v).rank().to_numpy() - (n + 1) / 2
        den = math.sqrt((rk * rk).sum() * (tr * tr).sum())
        return float((rk * tr).sum() / den) if den > 0 else 0.0
    return S(c).rolling(n).apply(_rho, raw=True).to_numpy()


def market_thermometer(df, n=22, hot=3.0, quiet_bars=5):
    _, h, l, _, _ = cols(df)
    ph, pl = shift(h, 1), shift(l, 1)
    inside = (h <= ph) & (l >= pl)
    temp = np.where(inside, 0.0, np.maximum(np.nan_to_num(h - ph), np.nan_to_num(pl - l)))
    temp = np.maximum(temp, 0.0)
    e = ema(temp, n)
    below = (temp < e).astype(float)
    quiet = S(below).rolling(quiet_bars).sum().to_numpy() >= quiet_bars
    return {"temp": temp, "ema": e, "hot": temp > hot * e, "quiet": quiet}


def williams_ad(df):
    _, h, l, c, _ = cols(df)
    pc = shift(c, 1)
    trh = np.fmax(h, pc)
    trl = np.fmin(l, pc)
    ad = np.where(c > pc, c - trl, np.where(c < pc, c - trh, 0.0))
    return np.cumsum(np.nan_to_num(ad))


def fib_bollinger(df, length=200, mult=3.0):
    _, h, l, c, v = cols(df)
    src = (h + l + c) / 3
    basis = (S(src * v).rolling(length).sum() / S(v).rolling(length).sum()).to_numpy()
    dev = mult * stdev(src, length)
    out = {"basis": basis}
    for f in (0.236, 0.382, 0.5, 0.618, 0.764, 1.0):
        k = str(f).replace("0.", "").ljust(3, "0") if f < 1 else "1000"
        out["up_" + k] = basis + f * dev
        out["dn_" + k] = basis - f * dev
    return out


def _hma_kernel(length):
    """Impulse response of HMA(length) (most recent weight first)."""
    m = length + max(int(np.sqrt(length)), 1) + 2
    imp = np.zeros(4 * m)
    imp[2 * m] = 1.0
    r = hma(imp, length)
    w = r[2 * m: 2 * m + m]
    w = np.nan_to_num(w)
    return w[: np.max(np.nonzero(np.abs(w) > 1e-12)) + 1]


def hull_butterfly(df, length=14, mult=1.5):
    _, _, _, c, _ = cols(df)
    n = len(c)
    w = _hma_kernel(length)
    wr = w[::-1]
    k = len(w)
    osc = np.full(n, np.nan)
    for t in range(k - 1, n):
        win = c[t - k + 1: t + 1][::-1]        # most recent first
        osc[t] = float(np.dot(w, win) - np.dot(wr, win))
    absm = S(np.abs(osc)).expanding(min_periods=50).mean().to_numpy() * mult
    o1, o2 = shift(osc, 1), shift(osc, 2)
    turn_dn = (osc < o1) & (o1 >= o2) & (o1 > absm)
    turn_up = (osc > o1) & (o1 <= o2) & (o1 < -absm)
    return {"osc": osc, "upper": absm, "lower": -absm, "dot_sell": turn_dn, "dot_buy": turn_up}


def range_detector(df, length=20, mult=1.0, atr_len=500):
    _, h, l, c, _ = cols(df)
    n = len(c)
    a = atr(h, l, c, atr_len)
    a = np.where(np.isfinite(a), a, atr(h, l, c, 14)) * mult
    ma = sma(c, length)
    top = np.full(n, np.nan)
    btm = np.full(n, np.nan)
    in_rng = np.zeros(n, bool)
    brk_up = np.zeros(n, bool)
    brk_dn = np.zeros(n, bool)
    rlen = np.zeros(n)
    cur_t = cur_b = np.nan
    start = -1
    for t in range(length, n):
        if not (np.isfinite(ma[t]) and np.isfinite(a[t])):
            continue
        win = c[t - length + 1: t + 1]
        ok = np.all(np.abs(win - ma[t]) <= a[t])
        if np.isfinite(cur_t):
            if c[t] > cur_t:
                brk_up[t] = True
                cur_t = cur_b = np.nan
            elif c[t] < cur_b:
                brk_dn[t] = True
                cur_t = cur_b = np.nan
        if ok and not (brk_up[t] or brk_dn[t]):
            nt, nb = ma[t] + a[t], ma[t] - a[t]
            if np.isfinite(cur_t) and nb <= cur_t and nt >= cur_b:     # overlaps the live range -> extend it
                cur_t, cur_b = max(cur_t, nt), min(cur_b, nb)
            elif not np.isfinite(cur_t):
                cur_t, cur_b, start = nt, nb, t - length + 1
        if np.isfinite(cur_t):
            top[t], btm[t], in_rng[t] = cur_t, cur_b, True
            rlen[t] = t - start + 1
    return {"top": top, "bottom": btm, "mid": (top + btm) / 2, "in_range": in_rng, "break_up": brk_up,
            "break_dn": brk_dn, "range_len": rlen}
