"""Indicators the library lacked (scout 18.11.2026 run). Each is a port of the published formula / open-source code, or
our own reconstruction where the code is closed or the page does not give it (said so per indicator). No repaint: every
value at bar i uses bars <= i only (swing pivots enter on the bar that confirms them).

- damiani_volatmeter: Damiani Volatmeter (Giorgos Damianos, MT4; open-source Pine port by loxx
  https://www.tradingview.com/script/z95IdM4a-Damiani-Volatmeter-loxx ): vol = ATR13 / ATR40 + 0.5 (vol[1] - vol[3])
  (lag suppressor), anti = StDev20 / StDev100 of the close, trade-on when vol > 1.4 - anti (the 'volatility on' state
  in which trend / breakout systems are allowed).
- vpci: Volume Price Confirmation Indicator (Buff Dormeier, TASC 2007.07,
  https://www.traders.com/Documentation/FEEDbk_docs/2007/07/Abstracts_new/DORMEIER/dormeier.html ; LuxAlgo concept
  https://www.luxalgo.com/library/concept/vpci/ ): VPC = VWMA20 - SMA20, VPR = VWMA5 / SMA5, VM = SMA5(v) / SMA20(v),
  VPCI = VPC x VPR x VM. Signal line: SMA 8 of VPCI (the signal length is our choice; the article smooths with a short
  volume-weighted average). Tick volume is used as volume.
- nrtr: Nick Rypock Trailing Reverse (MQL5 article https://www.mql5.com/en/articles/3690 ; CodeBase
  https://www.mql5.com/en/code/606 ): in an up trend the line trails the highest close since the trend began minus
  k x ATR(n), it only rises; a close under it reverses to the down trend (the line = lowest close + k x ATR, only
  falls). The article's percentage version is replaced by the ATR version (k 2, ATR 40) so it fits every instrument.
- jma: Jurik Moving Average - Jurik's code is closed; this is the widely used open-source approximation (everget
  'Jurik Moving Average' https://www.tradingview.com/script/gwzRz6tI-Jurik-Moving-Average ): length 7, phase 50,
  power 2 (three-stage adaptive filter e0 / e1 / e2).
- autocorr_periodogram: Ehlers Autocorrelation Periodogram (Cycle Analytics for Traders ch. 8; open-source Pine
  https://in.tradingview.com/script/df3Ofxvr-Ehlers-Autocorrelation-Periodogram-Loxx ): roofing filter (HP 48 +
  SuperSmoother 10), Pearson correlation of the filter with its lag 0..48 over 3 samples, DFT of the correlations for
  periods 10..48, power smoothed 0.2 / 0.8, normalised by a 0.995-decaying max; dominant cycle = the power-weighted
  centre of the periods with normalised power >= 0.5. Returns dc (bars) and the adaptive RSI with length dc / 2
  (Ehlers' own use of it).
- chandelier_exit: Chandelier Exit as in everget's open-source script (https://tradingview.com/script/AqXxNS7j-Chandelier-Exit ):
  22 / 3 ATR from the highest / lowest CLOSE, the long stop only rises while the close holds over it (the short stop only
  falls), direction flips when the close crosses the other side's previous stop - buy / sell on the flip. (The library's
  TV39 uses the plain LeBeau lines on highs / lows without the ratchet.)
- elder_impulse: Elder Impulse System (Alexander Elder; open source
  https://www.tradingview.com/script/oCbEFfpg-Indicator-Elder-Impulse-System ): +1 (green) when EMA13 and the MACD
  12/26/9 histogram both rise, -1 (red) when both fall, 0 (blue) otherwise. Elder's rule: no shorts on green, no longs
  on red.
- livermore_pivots: the Livermore market-key pivots as Oxford Capital Strategies tests them
  (https://oxfordstrat.com/trading-strategies/livermore-system-1/ and /livermore-system-2/): mode "atr" = a supply
  pivot is a high followed by a fall of swing x ATR20 (demand: a low followed by a rise of swing x ATR20), known on the
  bar that completes the fall / rise; mode "bars" = DeMark pivots (size bars lower highs / higher lows on each side),
  known `size` bars later. Returns the last two supply pivots, the last two demand pivots and the smaller of the last
  two swings (the page's noise base). The swing pairing in "bars" mode is our simplification.
- range_rank: Toby Crabel's narrow / wide range N-bar test (Oxford Capital Strategies 'Wide Range N-Day Pattern'
  https://oxfordstrat.com/trading-strategies/wide-range-pattern/ ): the N-bar high-low range against every N-bar range
  of the last `lookback` bars - wr = the widest, nr = the narrowest, plus its percentile rank.
"""
import numpy as np

from agents.indicators._core import S, atr, cols, ema, highest, lowest, pivots, rsi, shift, sma, stdev


def damiani_volatmeter(df, vis_atr=13, vis_std=20, sed_atr=40, sed_std=100, threshold=1.4, lag_k=0.5):
    """dict vol, anti (the moving threshold = threshold - anti), on (bool trade-on)."""
    o, h, l, c, v = cols(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = atr(h, l, c, vis_atr) / atr(h, l, c, sed_atr)
        anti = stdev(c, vis_std) / stdev(c, sed_std)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0)
    vol = np.zeros(len(c))
    for i in range(len(c)):
        s1 = vol[i - 1] if i >= 1 else 0.0
        s3 = vol[i - 3] if i >= 3 else 0.0
        vol[i] = base[i] + lag_k * (s1 - s3)
    t = threshold - np.nan_to_num(anti, nan=threshold)
    vol[:sed_std] = np.nan
    return {"vol": vol, "thr": t, "on": np.nan_to_num(vol, nan=-1.0) > t}


def vpci(df, short=5, long=20, sig=8):
    """dict vpci, signal."""
    o, h, l, c, v = cols(df)
    v = np.where(v > 0, v, 1.0)
    cv = c * v
    with np.errstate(divide="ignore", invalid="ignore"):
        vwl = S(cv).rolling(long).sum().to_numpy() / S(v).rolling(long).sum().to_numpy()
        vws = S(cv).rolling(short).sum().to_numpy() / S(v).rolling(short).sum().to_numpy()
        vpc = vwl - sma(c, long)
        vpr = vws / sma(c, short)
        vm = sma(v, short) / sma(v, long)
    x = vpc * vpr * vm
    return {"vpci": x, "signal": sma(np.nan_to_num(x), sig)}


def nrtr(df, n=40, k=2.0):
    """dict line, trend (+1 / -1), flip_up, flip_dn."""
    o, h, l, c, v = cols(df)
    a = atr(h, l, c, n)
    N = len(c)
    line = np.full(N, np.nan)
    tr_ = np.zeros(N, int)
    d, ext, ln = 1, c[0], np.nan
    for i in range(N):
        if not np.isfinite(a[i]) or i < n:
            ext = c[i]
            continue
        if d > 0:
            ext = max(ext, c[i])
            cand = ext - k * a[i]
            ln = cand if not np.isfinite(ln) else max(ln, cand)
            if c[i] < ln:
                d, ext = -1, c[i]
                ln = ext + k * a[i]
        else:
            ext = min(ext, c[i])
            cand = ext + k * a[i]
            ln = cand if not np.isfinite(ln) else min(ln, cand)
            if c[i] > ln:
                d, ext = 1, c[i]
                ln = ext - k * a[i]
        line[i], tr_[i] = ln, d
    prev = np.concatenate([[0], tr_[:-1]])
    return {"line": line, "trend": tr_, "flip_up": (tr_ == 1) & (prev == -1), "flip_dn": (tr_ == -1) & (prev == 1)}


def jma(df, length=7, phase=50, power=2, src=None):
    """Jurik MA approximation (array)."""
    o, h, l, c, v = cols(df)
    x = c if src is None else np.asarray(src, float)
    pr = 0.5 if phase < -100 else (2.5 if phase > 100 else phase / 100.0 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = beta ** power
    out = np.full(len(x), np.nan)
    e0 = e1 = e2 = 0.0
    j = np.nan
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        if not np.isfinite(j):
            e0, e1, e2, j = x[i], 0.0, 0.0, x[i]
            out[i] = j
            continue
        e0 = (1 - alpha) * x[i] + alpha * e0
        e1 = (x[i] - e0) * (1 - beta) + beta * e1
        e2 = (e0 + pr * e1 - j) * (1 - alpha) ** 2 + alpha ** 2 * e2
        j = e2 + j
        out[i] = j
    return out


def autocorr_periodogram(df, hp=48, ss=10, avg=3, lo=10, hi=48):
    """dict dc (dominant cycle in bars), rsi_dc (RSI with length round(dc / 2), 2..24)."""
    o, h, l, c, v = cols(df)
    N = len(c)
    q = 0.707 * 2 * np.pi / hp
    a1 = (np.cos(q) + np.sin(q) - 1) / np.cos(q)
    hpf = np.zeros(N)
    for i in range(2, N):
        hpf[i] = ((1 - a1 / 2) ** 2 * (c[i] - 2 * c[i - 1] + c[i - 2]) + 2 * (1 - a1) * hpf[i - 1]
                  - (1 - a1) ** 2 * hpf[i - 2])
    aa = np.exp(-1.414 * np.pi / ss)
    b1 = 2 * aa * np.cos(1.414 * np.pi / ss)
    k2, k3 = b1, -aa * aa
    k1 = 1 - k2 - k3
    f = np.zeros(N)
    for i in range(2, N):
        f[i] = k1 * (hpf[i] + hpf[i - 1]) / 2 + k2 * f[i - 1] + k3 * f[i - 2]
    corr = np.zeros((N, hi + 1))
    fx = S(f)
    for lag in range(hi + 1):
        y = S(shift(f, lag))
        sx, sy = fx.rolling(avg).sum(), y.rolling(avg).sum()
        sxx, syy, sxy = (fx * fx).rolling(avg).sum(), (y * y).rolling(avg).sum(), (fx * y).rolling(avg).sum()
        den = (avg * sxx - sx * sx) * (avg * syy - sy * sy)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ((avg * sxy - sx * sy) / np.sqrt(den)).to_numpy()
        corr[:, lag] = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    periods = np.arange(lo, hi + 1)
    lags = np.arange(3, hi + 1)
    cosm = np.cos(2 * np.pi * np.outer(lags, 1.0 / periods))
    sinm = np.sin(2 * np.pi * np.outer(lags, 1.0 / periods))
    cp, sp = corr[:, lags] @ cosm, corr[:, lags] @ sinm
    sq = cp * cp + sp * sp
    R = np.zeros_like(sq)
    dc = np.full(N, np.nan)
    mx = 0.0
    for i in range(N):
        R[i] = 0.2 * sq[i] ** 2 + (0.8 * R[i - 1] if i else 0.0)
        mx = max(0.995 * mx, R[i].max())
        if i < hp + hi or mx <= 0:
            continue
        p = R[i] / mx
        w = p >= 0.5
        if w.any():
            dc[i] = float((periods[w] * p[w]).sum() / p[w].sum())
    dcf = S(dc).ffill().to_numpy()
    ln = np.clip(np.round(np.nan_to_num(dcf, nan=28.0) / 2), 2, 24).astype(int)
    rs = {k: rsi(c, k) for k in np.unique(ln)}
    rdc = np.array([rs[ln[i]][i] for i in range(N)])
    rdc[np.isnan(dcf)] = np.nan
    return {"dc": dcf, "rsi_dc": rdc}


def chandelier_exit(df, n=22, mult=3.0, use_close=True):
    """dict long_stop, short_stop, dir (+1 / -1), buy, sell (the flips)."""
    o, h, l, c, v = cols(df)
    a = mult * atr(h, l, c, n)
    hh = highest(c if use_close else h, n)
    ll = lowest(c if use_close else l, n)
    N = len(c)
    ls, ss_, d = np.full(N, np.nan), np.full(N, np.nan), np.zeros(N, int)
    for i in range(N):
        x, y = hh[i] - a[i], ll[i] + a[i]
        if i and np.isfinite(ls[i - 1]) and c[i - 1] > ls[i - 1]:
            x = max(x, ls[i - 1])
        if i and np.isfinite(ss_[i - 1]) and c[i - 1] < ss_[i - 1]:
            y = min(y, ss_[i - 1])
        ls[i], ss_[i] = x, y
        dd = d[i - 1] if i else 1
        if i and np.isfinite(ss_[i - 1]) and c[i] > ss_[i - 1]:
            dd = 1
        elif i and np.isfinite(ls[i - 1]) and c[i] < ls[i - 1]:
            dd = -1
        d[i] = dd if dd else 1
    prev = np.concatenate([[d[0]], d[:-1]])
    return {"long_stop": ls, "short_stop": ss_, "dir": d, "buy": (d == 1) & (prev == -1), "sell": (d == -1) & (prev == 1)}


def elder_impulse(df, n=13, fast=12, slow=26, sig=9):
    """array +1 green / -1 red / 0 blue."""
    o, h, l, c, v = cols(df)
    e = ema(c, n)
    m = ema(c, fast) - ema(c, slow)
    hist = m - ema(m, sig)
    de, dh = e - shift(e), hist - shift(hist)
    out = np.where((de > 0) & (dh > 0), 1, np.where((de < 0) & (dh < 0), -1, 0))
    out[:slow + sig] = 0
    return out


def livermore_pivots(df, mode="atr", swing=4.0, atr_n=20, size=20):
    """dict sup1, sup2 (last / previous supply pivot), dem1, dem2 (last / previous demand pivot), swing_min (smaller of
    the last two swings), new_sup / new_dem (True on the confirming bar)."""
    o, h, l, c, v = cols(df)
    N = len(c)
    sup1, sup2, dem1, dem2 = (np.full(N, np.nan) for _ in range(4))
    ns, nd = np.zeros(N, bool), np.zeros(N, bool)
    s1 = s2 = d1 = d2 = np.nan
    if mode == "atr":
        a = atr(h, l, c, atr_n)
        d = 0
        hi_, lo_ = h[0], l[0]
        for i in range(N):
            if np.isfinite(a[i]) and i >= atr_n:
                if d >= 0:
                    if h[i] >= hi_:
                        hi_ = h[i]
                    elif l[i] <= hi_ - swing * a[i]:
                        s2, s1 = s1, hi_
                        ns[i] = True
                        d, lo_ = -1, l[i]
                if d < 0 and not ns[i]:
                    if l[i] <= lo_:
                        lo_ = l[i]
                    elif h[i] >= lo_ + swing * a[i]:
                        d2, d1 = d1, lo_
                        nd[i] = True
                        d, hi_ = 1, h[i]
                if d == 0:
                    hi_, lo_ = max(hi_, h[i]), min(lo_, l[i])
            else:
                hi_, lo_ = max(hi_, h[i]), min(lo_, l[i])
            sup1[i], sup2[i], dem1[i], dem2[i] = s1, s2, d1, d2
    else:
        phc, plc, phv, plv, _, _ = pivots(h, l, size)
        for i in range(N):
            if phc[i]:
                s2, s1 = s1, phv[i]
                ns[i] = True
            if plc[i]:
                d2, d1 = d1, plv[i]
                nd[i] = True
            sup1[i], sup2[i], dem1[i], dem2[i] = s1, s2, d1, d2
    sw = np.fmin(np.abs(sup1 - dem1), np.abs(sup2 - dem2))
    return {"sup1": sup1, "sup2": sup2, "dem1": dem1, "dem2": dem2, "swing_min": sw, "new_sup": ns, "new_dem": nd}


def range_rank(df, n=2, lookback=20):
    """dict rng (N-bar range), wr (widest of the lookback), nr (narrowest), rank (0..1 percentile in the lookback)."""
    o, h, l, c, v = cols(df)
    r = highest(h, n) - lowest(l, n)
    rs = S(r)
    mx = rs.rolling(lookback).max().to_numpy()
    mn = rs.rolling(lookback).min().to_numpy()
    rank = rs.rolling(lookback).rank(pct=True).to_numpy()
    return {"rng": r, "wr": np.isfinite(mx) & (r >= mx), "nr": np.isfinite(mn) & (r <= mn), "rank": rank}
