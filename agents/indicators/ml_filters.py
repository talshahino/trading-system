"""Adaptive / 'AI' trend filters from the user's videos (task tal1411ind, 12:43-12:56). Closed bars only - no repaint.

knn_trend_navigator - AI Trend Navigator [K-Neighbor] (Zeiierman, open source
https://www.tradingview.com/script/30aLnPNh-AI-Trend-Navigator-K-Neighbor/; steps as in the ProRealCode port
https://www.prorealcode.com/prorealtime-indicators/ai-trend-navigator/): value = SMA(hl2, 5), target = RMA(close, 5);
knnMA = mean of the K=3 past values (of the last 30) closest to the current target; knnLine = WMA(knnMA, 5) (rising =
up); classifier over the last 10 bars of mid = (knnMA + close) / 2 votes on RMA(knnMA[1], 50) < RMA(knnMA, 50),
pred = WMA(+1/-1, 3).

gaussian_channel - Gaussian Channel [DW] (DonovanWall, MPL-2.0, https://www.tradingview.com/script/WpVY7GKW-Gaussian-Channel-DW/):
Ehlers N-pole Gaussian filter (N=4, sampling period 144) of hlc3 and of the true range; bands = filter +/- 1.414 x
filtered TR; beta = (1 - cos(2 pi / per)) / (1.414^(2/N) - 1), alpha = -beta + sqrt(beta^2 + 2 beta);
f = alpha^N s - sum_k C(N,k) (-(1-alpha))^k f[k]. Plain mode (no reduced lag / fast response).

kalman_supertrend - OUR RECONSTRUCTION of 'Pro Scalper' (profitprotrading, invite-only
https://tradingview.com/script/BEhT6Syv-Pro-Scalper; ProTrippa's scalping list): the description says a SuperTrend
run on a Kalman-smoothed price with VWMA bands of the highest high / lowest low as overbought / oversold. Ours: a
1-D Kalman filter of close (process / measurement noise ratio q/r = 0.05), SuperTrend ATR 10 x 3 around the filtered
price, bands = VWMA(highest high 20, 20) / VWMA(lowest low 20, 20).

trend_magnet - OUR RECONSTRUCTION of BigBeluga 'Trend Magnet' (closed, Market Waves Pro: a trend band with Scalp /
Normal / Macro sensitivity and low-momentum zones): mid = EMA(hlc3, L) with L 20 / 50 / 100, band = mid +/- 1.5 ATR(L);
the trend flips up on a close over the upper band of the previous bar, down under the lower; the line ratchets
(up trend: max(lower band, line[1])); low momentum = ADX(14) < 20.

volume_mapping - OUR RECONSTRUCTION of BigBeluga 'Volume Mapping' (closed, Nautilus oscillator: green bars = buying
volume dominant): bull = v on up candles, bear = v on down candles; osc = (EMA(bull, 12) - EMA(bear, 12)) / SMA(v, 12).

sniper_ribbon - OUR RECONSTRUCTION of 'Sniper Entry/Exit with SL&TP by KhanSaab V.02' (3 / 5 / 15 min,
https://www.tradingview.com/script/HBDcHSB7-Sniper-Entry-Exit-with-SL-TP-by-KhanSaab-V-02/): EMA9 / EMA21 ribbon;
a fresh signal = the 9 / 21 cross; a 'retest' (orange candle) = in the trend the low (high) touches the ribbon and the
close holds beyond EMA21; dashboard bull score = share of [close > session VWAP, RSI14 > 50, MACD histogram > 0,
ADX14 > 20 with +DI > -DI]; the source's filter: buy only with bull score > 60%."""
import math

import numpy as np
import pandas as pd

from agents.indicators._core import S, atr, cols, cross_down, cross_up, dmi, ema, rma, rsi, shift, sma, tr, wma


def knn_trend_navigator(df, k=3, window=30, ma_len=5, tgt_len=5, smooth=50):
    """dict: knn_ma, knn_line, up (knnLine rising), pred (-1..1), buy (line turns up with pred > 0), sell."""
    o, h, l, c, v = cols(df)
    n = len(c)
    val = sma((h + l) / 2, ma_len)
    tgt = rma(c, tgt_len)
    W = max(k, window)
    knn = np.full(n, np.nan)
    for t in range(W + ma_len, n):
        past = val[t - W:t][::-1]                 # value_in[1..W]
        d = np.abs(tgt[t] - past)
        idx = np.argsort(d, kind="stable")[:k]
        knn[t] = past[idx].mean()
    line = np.array(wma(np.nan_to_num(knn, nan=c), 5), dtype=float)
    line[:W + ma_len + 5] = np.nan
    up = line > shift(line, 1)
    mid = (knn + c) / 2
    o_r, c_r = rma(np.nan_to_num(knn, nan=c), smooth), rma(np.nan_to_num(shift(knn, 1), nan=c), smooth)
    raw = np.zeros(n)
    for t in range(10, n):
        if not np.isfinite(mid[t]):
            continue
        mn, pos, neg = np.inf, 0, 0
        for j in range(1, 11):
            dd = abs(mid[t - j] - mid[t]) if np.isfinite(mid[t - j]) else np.inf
            if dd < mn:
                mn = dd
                if c_r[t - j] < o_r[t - j]:
                    pos += 1
                elif c_r[t - j] > o_r[t - j]:
                    neg += 1
        raw[t] = 1.0 if pos > neg else -1.0
    pred = wma(raw, 3)
    turn_up = up & ~np.concatenate([[False], up[:-1]])
    dn = line < shift(line, 1)
    turn_dn = dn & ~np.concatenate([[False], dn[:-1]])
    return {"knn_ma": knn, "knn_line": line, "up": up, "pred": pred,
            "buy": turn_up & (pred > 0), "sell": turn_dn & (pred < 0)}


def _gauss(x, alpha, N):
    n = len(x)
    f = np.zeros(n)
    xx = 1 - alpha
    coef = [math.comb(N, kk) * (-xx) ** kk for kk in range(1, N + 1)]
    a = alpha ** N
    x = np.nan_to_num(x)
    for t in range(n):
        s = a * x[t]
        for kk in range(1, N + 1):
            if t - kk >= 0:
                s -= coef[kk - 1] * f[t - kk]
        f[t] = s
    return f


def gaussian_channel(df, poles=4, per=144, mult=1.414):
    """dict: filt, upper, lower, rising."""
    o, h, l, c, v = cols(df)
    src = (h + l + c) / 3
    beta = (1 - math.cos(2 * math.pi / per)) / (1.414 ** (2.0 / poles) - 1)
    alpha = -beta + math.sqrt(beta * beta + 2 * beta)
    f = _gauss(src, alpha, poles)
    ft = _gauss(tr(h, l, c), alpha, poles)
    warm = min(len(c), 2 * per)
    f[:warm] = np.nan
    ft[:warm] = np.nan
    return {"filt": f, "upper": f + mult * ft, "lower": f - mult * ft, "rising": f > shift(f, 1)}


def _supertrend_on(src, h, l, c, n=10, mult=3.0):
    a = atr(h, l, c, n)
    up0, dn0 = src - mult * a, src + mult * a
    N = len(c)
    up, dn, tr_ = up0.copy(), dn0.copy(), np.ones(N)
    for t in range(1, N):
        up[t] = max(up0[t], up[t - 1]) if c[t - 1] > up[t - 1] else up0[t]
        dn[t] = min(dn0[t], dn[t - 1]) if c[t - 1] < dn[t - 1] else dn0[t]
        if tr_[t - 1] < 0 and c[t] > dn[t - 1]:
            tr_[t] = 1
        elif tr_[t - 1] > 0 and c[t] < up[t - 1]:
            tr_[t] = -1
        else:
            tr_[t] = tr_[t - 1]
    line = np.where(tr_ > 0, up, dn)
    return tr_, line


def kalman_supertrend(df, q_r=0.05, n=10, mult=3.0, band_len=20):
    """dict: kf (filtered close), trend (+1/-1), line, buy, sell, ob (upper VWMA band), os (lower)."""
    o, h, l, c, v = cols(df)
    N = len(c)
    kf = np.zeros(N)
    p = 1.0
    x = c[0] if N else 0.0
    for t in range(N):
        p = p + q_r
        g = p / (p + 1.0)
        x = x + g * (c[t] - x)
        p = (1 - g) * p
        kf[t] = x
    tr_, line = _supertrend_on(kf, h, l, c, n, mult)
    tr_[:n + 1] = 1
    prev = np.concatenate([[1], tr_[:-1]])
    vw = lambda x_: S(x_ * v).rolling(band_len).sum().to_numpy() / S(v).rolling(band_len).sum().to_numpy()  # noqa: E731
    return {"kf": kf, "trend": tr_, "line": line, "buy": (tr_ > 0) & (prev < 0), "sell": (tr_ < 0) & (prev > 0),
            "ob": vw(S(h).rolling(band_len).max().to_numpy()), "os": vw(S(l).rolling(band_len).min().to_numpy())}


def trend_magnet(df, mode="normal", k=1.5):
    """dict: line, trend (+1/-1), buy, sell, low_mom (ADX < 20)."""
    o, h, l, c, v = cols(df)
    L = {"scalp": 20, "normal": 50, "macro": 100}[mode]
    mid = ema((h + l + c) / 3, L)
    a = atr(h, l, c, L)
    ub, lb = mid + k * a, mid - k * a
    N = len(c)
    tr_ = np.zeros(N)
    line = np.full(N, np.nan)
    for t in range(1, N):
        tr_[t] = tr_[t - 1]
        if tr_[t] <= 0 and c[t] > ub[t - 1]:
            tr_[t] = 1
        elif tr_[t] >= 0 and c[t] < lb[t - 1]:
            tr_[t] = -1
        if tr_[t] > 0:
            line[t] = lb[t] if tr_[t - 1] <= 0 or not np.isfinite(line[t - 1]) else max(lb[t], line[t - 1])
        elif tr_[t] < 0:
            line[t] = ub[t] if tr_[t - 1] >= 0 or not np.isfinite(line[t - 1]) else min(ub[t], line[t - 1])
    prev = np.concatenate([[0], tr_[:-1]])
    _, _, adx_ = dmi(h, l, c, 14)
    return {"line": line, "trend": tr_, "buy": (tr_ > 0) & (prev <= 0) & (np.arange(N) > L),
            "sell": (tr_ < 0) & (prev >= 0) & (np.arange(N) > L), "low_mom": adx_ < 20}


def volume_mapping(df, n=12):
    """array: buy-minus-sell volume oscillator (normalised by average volume)."""
    o, h, l, c, v = cols(df)
    bull = np.where(c > o, v, 0.0)
    bear = np.where(c < o, v, 0.0)
    vm = sma(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vm > 0, (ema(bull, n) - ema(bear, n)) / vm, 0.0)


def sniper_ribbon(df):
    """dict: e9, e21, buy, sell (9/21 cross), retest_long, retest_short, bull_score, bear_score (0..1)."""
    o, h, l, c, v = cols(df)
    e9, e21 = ema(c, 9), ema(c, 21)
    t = df.index.tz_convert("UTC")
    day = pd.Series(t.date).to_numpy()
    tp = (h + l + c) / 3
    vw = (pd.Series(tp * v).groupby(day).cumsum() / pd.Series(v).groupby(day).cumsum()).to_numpy()
    r = rsi(c, 14)
    hist = (ema(c, 12) - ema(c, 26)) - ema(ema(c, 12) - ema(c, 26), 9)
    pdi, ndi, adx_ = dmi(h, l, c, 14)
    bull = np.vstack([c > vw, r > 50, hist > 0, (adx_ > 20) & (pdi > ndi)]).mean(axis=0)
    bear = np.vstack([c < vw, r < 50, hist < 0, (adx_ > 20) & (ndi > pdi)]).mean(axis=0)
    up = e9 > e21
    return {"e9": e9, "e21": e21, "buy": cross_up(e9, e21), "sell": cross_down(e9, e21),
            "retest_long": up & (l <= np.fmax(e9, e21)) & (c > e21) & (c > o),
            "retest_short": ~up & (h >= np.fmin(e9, e21)) & (c < e21) & (c < o),
            "bull_score": bull, "bear_score": bear}
