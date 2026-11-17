"""Indicators the library lacked (scout 17.11.2026, afternoon run). Each is a port of the published formula, or our own
reconstruction where the code / formula is closed or not on the page (said so per indicator). No repaint: every value
at bar i uses bars <= i only (pivots enter on their confirmation bar; cumulative sums only look back).

- uniema: TASC 2026.11 'The UniEMA' (PineCodersTASC, https://www.tradingview.com/script/1ZLcORQR-TASC-2026-11-The-UniEMA/ ).
  The page lists the four knobs (poles 1-3, order = repeated passes, weight factor 1 = Wilder / 2 = EMA, damping 0..1)
  but no formula - OUR reconstruction: alpha = wf / (length + wf - 1) (wf 1 -> 1/L Wilder, wf 2 -> 2/(L+1) EMA);
  1 pole = EMA, 2 poles = e1 + d (e1 - e2) (DEMA at d = 1), 3 poles = e1 + d (2 e1 - 3 e2 + e3) (TEMA at d = 1);
  'order' repeats the whole filter on its own output.
- range_vol: range-based volatility estimators - Parkinson (1980), Garman-Klass (1980), Rogers-Satchell (1991),
  Yang-Zhang (2000) (formulas: https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/ ),
  per bar (not annualised), window 20, plus the 100-bar percentile rank of Yang-Zhang (compression / expansion).
- vqi: Volatility Quality Index (Thomas Stridsman, TASC 2002; LazyBear open source
  https://www.tradingview.com/script/MbAO4zo0-Indicator-Volatility-Quality-Index-VQI ): bar quality
  ((c - c1) / TR + (c - o) / (h - l)) / 2, times the bar's move ((c - c1) + (c - o)) / 2, cumulated; SMA 9 / SMA 200.
  The page tonight did not show the code; the formula is the published Stridsman / LazyBear one as we know it.
- didi_index: Didi Index (Odir 'Didi' Aguiar; open source https://www.tradingview.com/script/qt2X6wrd-Didi-Index ):
  SMA 3 and SMA 20 normalised by SMA 8 (ratio - 1). 'Agulhada' (needle) = the fast line crosses up through 0 and the
  slow one down through 0 within 3 bars (buy), mirror (sell) - our reading of the needle rule.
- pivot_supertrend: Pivot Point SuperTrend (LonesomeTheBlue, open source
  https://www.tradingview.com/script/L0AIiLvH-Pivot-Point-Supertrend/ ): pivots 2, centre = (2 centre + pivot) / 3 on
  every new (confirmed) pivot, bands centre +/- 3 x ATR 10, SuperTrend trailing logic.
- psy_line: Psychological Line (https://www.luxalgo.com/library/concept/psychological-line/ ): % of up closes in 12 bars.
- adxr: Wilder's Average Directional Movement Rating: (ADX + ADX 14 bars ago) / 2, with +DI / -DI and ADX.
- gapo: Gopalakrishnan Range Index (TASC 2001; https://lightningchart.com/blog/trader/gopalakrishnan-range-index/ ):
  log(highest high - lowest low over n) / log(n), n 5.
- asi: Wilder's Accumulative Swing Index (New Concepts in Technical Trading Systems, 1978;
  https://www.investopedia.com/terms/a/asi.asp ): cumulative SI = 50 (c - c1 + 0.5 (c - o) + 0.25 (c1 - o1)) / R x K / T.
  T (the futures limit move) has no meaning for CFDs / FX - OUR choice: T = 3 x ATR 14.
- cfo: Chande Forecast Oscillator: 100 x (close - linear-regression forecast of the close, 14) / close.
- r_squared: R-squared of the 14-bar linear regression of closes on time, with the slope in ATRs per bar.
- dynamic_momentum: Chande & Kroll Dynamic Momentum Index (The New Technical Trader, 1994;
  https://www.investopedia.com/terms/d/dynamicmomentumindex.asp ): RSI whose length = 14 / (stdev5 / SMA10(stdev5)),
  clipped 5..30 (simple sums of gains / losses over that length).
- stoch_cci: Stochastic CCI (stochastic 14 of CCI 20, %K SMA 3, %D SMA 3), 0..100 (common TradingView / MT build).
"""
import math

import numpy as np

from agents.indicators._core import S, atr, cci, cols, cross_down, cross_up, dmi, pivots, shift, sma, stdev


def _ema_alpha(x, alpha):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = prev
            continue
        prev = v if np.isnan(prev) else prev + alpha * (v - prev)
        out[i] = prev
    return out


def uniema(df, length=20, poles=2, order=1, wf=2.0, damp=1.0):
    """dict uniema, slope (+1 / -1 / 0)."""
    o, h, l, c, v = cols(df)
    alpha = float(wf) / (float(length) + float(wf) - 1.0)
    x = c
    for _ in range(max(int(order), 1)):
        e1 = _ema_alpha(x, alpha)
        if poles <= 1:
            y = e1
        else:
            e2 = _ema_alpha(e1, alpha)
            if poles == 2:
                y = e1 + damp * (e1 - e2)
            else:
                e3 = _ema_alpha(e2, alpha)
                y = e1 + damp * (2 * e1 - 3 * e2 + e3)
        x = y
    slope = np.sign(np.nan_to_num(np.diff(x, prepend=x[0])))
    return {"uniema": x, "slope": slope}


def range_vol(df, n=20, rank_n=100):
    """dict parkinson, garman_klass, rogers_satchell, yang_zhang (per bar, log units), yz_rank (0..1)."""
    o, h, l, c, v = cols(df)
    pos = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    sf = lambda a: np.where(pos, a, np.nan)  # noqa: E731
    o, h, l, c = sf(o), sf(h), sf(l), sf(c)
    c1 = shift(c, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(h / l)
        co = np.log(c / o)
        oc = np.log(o / c1)
        rs_t = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    park = np.sqrt(sma(hl ** 2, n) / (4 * math.log(2)))
    gk = np.sqrt(np.clip(sma(0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2, n), 0, None))
    rs = np.sqrt(np.clip(sma(rs_t, n), 0, None))
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_o = S(oc).rolling(n).var(ddof=1).to_numpy()
    var_c = S(co).rolling(n).var(ddof=1).to_numpy()
    yz = np.sqrt(np.clip(var_o + k * var_c + (1 - k) * sma(rs_t, n), 0, None))
    rank = S(yz).rolling(rank_n).apply(lambda w: np.mean(w <= w[-1]), raw=True).to_numpy()
    return {"parkinson": park, "garman_klass": gk, "rogers_satchell": rs, "yang_zhang": yz, "yz_rank": rank}


def vqi(df, fast=9, slow=200):
    """dict vqi (cumulative), sma_fast, sma_slow."""
    o, h, l, c, v = cols(df)
    c1 = np.concatenate([[c[0]], c[:-1]])
    trr = np.maximum(h, c1) - np.minimum(l, c1)
    rng = h - l
    ok = (trr != 0) & (rng != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(ok, ((c - c1) / trr + (c - o) / rng) * 0.5, np.nan)
    q = S(q).ffill().fillna(0.0).to_numpy()
    val = np.abs(q) * ((c - c1) + (c - o)) * 0.5
    tot = np.cumsum(np.nan_to_num(val))
    return {"vqi": tot, "sma_fast": sma(tot, fast), "sma_slow": sma(tot, slow)}


def didi_index(df, fast=3, mid=8, slow=20, window=3):
    """dict fast, slow (ratios to SMA 8 minus 1), needle_buy, needle_sell."""
    o, h, l, c, v = cols(df)
    m = sma(c, mid)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = sma(c, fast) / m - 1.0
        s = sma(c, slow) / m - 1.0
    f0, s0 = np.nan_to_num(f), np.nan_to_num(s)
    rec = lambda a: S(a.astype(float)).rolling(window, min_periods=1).max().to_numpy() > 0  # noqa: E731
    nb = rec(cross_up(f0, 0.0)) & rec(cross_down(s0, 0.0)) & (f0 > 0) & (s0 < 0)
    ns = rec(cross_down(f0, 0.0)) & rec(cross_up(s0, 0.0)) & (f0 < 0) & (s0 > 0)
    on = lambda a: a & ~np.concatenate([[False], a[:-1]])  # noqa: E731
    return {"fast": f, "slow": s, "needle_buy": on(nb), "needle_sell": on(ns)}


def pivot_supertrend(df, prd=2, factor=3.0, atr_n=10):
    """dict line (trailing stop), trend (+1 / -1), centre, flip_up, flip_dn."""
    o, h, l, c, v = cols(df)
    n = len(c)
    phc, plc, phv, plv, _, _ = pivots(h, l, prd)
    a = atr(h, l, c, atr_n)
    centre = np.full(n, np.nan)
    tup = np.full(n, np.nan)
    tdn = np.full(n, np.nan)
    trend = np.ones(n)
    cen = np.nan
    for i in range(n):
        pp = phv[i] if phc[i] else (plv[i] if plc[i] else np.nan)
        if not np.isnan(pp):
            cen = pp if np.isnan(cen) else (cen * 2 + pp) / 3
        centre[i] = cen
        if np.isnan(cen):
            continue
        up, dn = cen - factor * a[i], cen + factor * a[i]
        if i > 0 and not np.isnan(tup[i - 1]):
            tup[i] = max(up, tup[i - 1]) if c[i - 1] > tup[i - 1] else up
            tdn[i] = min(dn, tdn[i - 1]) if c[i - 1] < tdn[i - 1] else dn
            trend[i] = 1 if c[i] > tdn[i - 1] else (-1 if c[i] < tup[i - 1] else trend[i - 1])
        else:
            tup[i], tdn[i] = up, dn
            trend[i] = trend[i - 1] if i > 0 else 1
    line = np.where(trend > 0, tup, tdn)
    prev = np.concatenate([[trend[0]], trend[:-1]])
    return {"line": line, "trend": trend, "centre": centre, "flip_up": (trend > 0) & (prev < 0),
            "flip_dn": (trend < 0) & (prev > 0)}


def psy_line(df, n=12):
    o, h, l, c, v = cols(df)
    up = np.concatenate([[0.0], (c[1:] > c[:-1]).astype(float)])
    return 100.0 * S(up).rolling(n).mean().to_numpy()


def adxr(df, n=14):
    """dict adxr, adx, pdi, ndi."""
    o, h, l, c, v = cols(df)
    p, m, a = dmi(h, l, c, n)
    return {"adxr": (a + shift(a, n)) / 2.0, "adx": a, "pdi": p, "ndi": m}


def gapo(df, n=5):
    o, h, l, c, v = cols(df)
    rng = S(h).rolling(n).max().to_numpy() - S(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rng > 0, np.log(rng) / math.log(n), np.nan)


def asi(df, t_atr=3.0):
    """dict asi (cumulative), si."""
    o, h, l, c, v = cols(df)
    c1 = np.concatenate([[c[0]], c[:-1]])
    o1 = np.concatenate([[o[0]], o[:-1]])
    a, b, d = np.abs(h - c1), np.abs(l - c1), np.abs(h - l)
    e = np.abs(c1 - o1)
    r = np.where((a >= b) & (a >= d), a - 0.5 * b + 0.25 * e, np.where((b >= a) & (b >= d), b - 0.5 * a + 0.25 * e,
                                                                         d + 0.25 * e))
    k = np.maximum(a, b)
    t = t_atr * atr(h, l, c, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.where((r > 0) & (t > 0), 50.0 * (c - c1 + 0.5 * (c - o) + 0.25 * (c1 - o1)) / r * k / t, 0.0)
    si = np.nan_to_num(si)
    return {"asi": np.cumsum(si), "si": si}


def _linreg(y, n):
    """Rolling OLS of y on time over n bars: endpoint, slope, r2 (vectorised, bars <= i)."""
    y = np.asarray(y, dtype=float)
    idx = np.arange(len(y), dtype=float)
    sy = S(y).rolling(n).sum().to_numpy()
    syy = S(y * y).rolling(n).sum().to_numpy()
    siy = S(idx * y).rolling(n).sum().to_numpy()
    start = idx - n + 1
    sxy = siy - start * sy                   # x = 0..n-1 inside the window
    sx = n * (n - 1) / 2.0
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den
    inter = (sy - slope * sx) / n
    end = inter + slope * (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (n * sxy - sx * sy) ** 2 / (den * (n * syy - sy * sy))
    return end, slope, np.clip(np.nan_to_num(r2, nan=0.0), 0, 1)


def cfo(df, n=14):
    o, h, l, c, v = cols(df)
    end, _, _ = _linreg(c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (c - end) / c


def r_squared(df, n=14):
    """dict r2 (0..1), slope_atr (regression slope per bar in ATR 14 units)."""
    o, h, l, c, v = cols(df)
    _, slope, r2 = _linreg(c, n)
    a = atr(h, l, c, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"r2": r2, "slope_atr": np.where(a > 0, slope / a, 0.0)}


def dynamic_momentum(df, base=14, lo=5, hi=30):
    """dict dmi (0..100), length."""
    o, h, l, c, v = cols(df)
    sd = stdev(c, 5)
    av = sma(sd, 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln = np.where(sd > 0, np.round(base / (sd / av)), base)
    ln = np.clip(np.nan_to_num(ln, nan=base), lo, hi).astype(int)
    d = np.diff(c, prepend=c[0])
    cu = np.concatenate([[0.0], np.cumsum(np.clip(d, 0, None))])
    cd = np.concatenate([[0.0], np.cumsum(np.clip(-d, 0, None))])
    i = np.arange(len(c))
    j = np.clip(i + 1 - ln, 0, None)
    U, D = cu[i + 1] - cu[j], cd[i + 1] - cd[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(U + D > 0, 100.0 * U / (U + D), 50.0)
    out[: hi + 10] = np.nan
    return {"dmi": out, "length": ln}


def stoch_cci(df, n_cci=20, n_st=14, ks=3, ds=3):
    """dict k, d (0..100)."""
    o, h, l, c, v = cols(df)
    x = cci(h, l, c, n_cci)
    hh, ll = S(x).rolling(n_st).max().to_numpy(), S(x).rolling(n_st).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(hh > ll, 100.0 * (x - ll) / (hh - ll), 50.0)
    k = sma(raw, ks)
    return {"k": k, "d": sma(k, ds)}
