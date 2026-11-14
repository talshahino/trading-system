"""Classic oscillators, bands and trailing stops the library did not have yet (scout, 15.11.2026).

Every one is a port of the published formula (sources in agents/indicators/__init__.py and below); all are causal -
the value at bar i uses bars <= i only (the Detrended Price Oscillator is the non-centred TradingView version, the
centred one looks ahead).

tsi             True Strength Index (William Blau, 'Momentum, Direction and Divergence'): 100 x EMA(EMA(mom, 25), 13) /
                EMA(EMA(|mom|, 25), 13), signal EMA 7; with r=20, s=5, sig=5 it is Blau's SMI Ergodic.
                https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/true-strength-index
cmo             Chande Momentum Oscillator 9: 100 x (sum up - sum down) / (sum up + sum down) of close changes.
dpo             Detrended Price Oscillator 20, non-centred: close - SMA(20) of n/2 + 1 bars ago (TradingView 'isCentered=false').
rvi             Relative Vigor Index 10 (Ehlers): SMA of the 4-bar weighted (c - o) over SMA of the weighted (h - l), signal
                = 4-bar weighted RVI.
ppo             Percentage Price Oscillator 12/26/9 (MACD in % of the slow EMA).
force_index     Elder Force Index: EMA13 and EMA2 of (c - c[1]) x tick volume.
chaikin_osc     Chaikin Oscillator: EMA3 - EMA10 of the Accumulation/Distribution line (tick volume).
nvi_pvi         Negative / Positive Volume Index (Fosback) with their 255-bar EMA.
starc           STARC bands (Manning Stoller): SMA6 +/- 2 x ATR15.
rwi             Random Walk Index (E. Michael Poulos, TASC 9:2): max over k = 2..14 of (h - l[k]) / (ATR(k) x sqrt(k)), and the low side.
vhf             Vertical Horizontal Filter 28 (Adam White): (highest close - lowest close) / sum |close change|.
bop             Balance of Power (Igor Livshin): SMA14 of (c - o) / (h - l).
qstick          QStick 8 (Tushar Chande): SMA of (c - o).
tii             Trend Intensity Index (M. H. Pee, TASC 2002): share of the positive close-SMA60 deviations over 30 bars, 0..100.
chande_kroll    Chande Kroll Stop (Chande & Kroll, 'The New Technical Trader'): p 10, x 1, q 9.
volatility_stop TradingView 'Volatility Stop' (VStop): close-based, 20 x 2 ATR from the extreme close since the flip.
"""
import numpy as np

from agents.indicators._core import S, atr, cols, cross_down, cross_up, ema, highest, lowest, sma, tr


def _sh(x, k=1):
    x = np.asarray(x, dtype=float)
    return np.concatenate([np.full(k, np.nan), x[:-k]]) if k > 0 else x


def _w4(x):
    """Ehlers' 4-bar weighting (x + 2 x1 + 2 x2 + x3) / 6."""
    return (x + 2 * _sh(x) + 2 * _sh(x, 2) + _sh(x, 3)) / 6.0


def tsi(df, r=25, s=13, sig=7):
    """dict: tsi, signal, up (TSI crosses over the signal), dn."""
    c = cols(df)[3]
    m = np.diff(c, prepend=np.nan)
    num = ema(ema(np.nan_to_num(m), r), s)
    den = ema(ema(np.abs(np.nan_to_num(m)), r), s)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(den > 0, 100 * num / den, 0.0)
    t[: r + s] = np.nan
    sg = np.array(ema(np.nan_to_num(t), sig))
    sg[: r + s] = np.nan
    return {"tsi": t, "signal": sg, "up": cross_up(t, sg), "dn": cross_down(t, sg)}


def cmo(df, n=9):
    c = cols(df)[3]
    d = np.diff(c, prepend=c[0] if len(c) else 0.0)
    su = S(np.maximum(d, 0)).rolling(n).sum().to_numpy()
    sd = S(np.maximum(-d, 0)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(su + sd > 0, 100 * (su - sd) / (su + sd), 0.0)


def dpo(df, n=20):
    c = cols(df)[3]
    return c - _sh(sma(c, n), n // 2 + 1)


def rvi(df, n=10):
    """dict: rvi, signal, up, dn."""
    o, h, l, c, v = cols(df)
    num, den = sma(_w4(c - o), n), sma(_w4(h - l), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    sg = _w4(r)
    return {"rvi": r, "signal": sg, "up": cross_up(r, sg), "dn": cross_down(r, sg)}


def ppo(df, fast=12, slow=26, sig=9):
    """dict: ppo, signal, hist (in % of the slow EMA)."""
    c = cols(df)[3]
    es = ema(c, slow)
    p = 100 * (ema(c, fast) - es) / es
    sg = ema(p, sig)
    return {"ppo": p, "signal": sg, "hist": p - sg}


def force_index(df, n=13):
    """dict: fi (EMA n), fi2 (EMA2), raw."""
    o, h, l, c, v = cols(df)
    raw = np.nan_to_num(np.diff(c, prepend=c[0] if len(c) else 0.0) * v)
    return {"fi": ema(raw, n), "fi2": ema(raw, 2), "raw": raw}


def chaikin_osc(df, fast=3, slow=10):
    """dict: osc, adl."""
    o, h, l, c, v = cols(df)
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where(rng > 0, ((c - l) - (h - c)) / rng, 0.0)
    adl = np.cumsum(mfm * v)
    return {"osc": ema(adl, fast) - ema(adl, slow), "adl": adl}


def nvi_pvi(df, n=255):
    """dict: nvi, pvi, nvi_ema, pvi_ema (start 1000)."""
    o, h, l, c, v = cols(df)
    k = len(c)
    nv, pv = np.full(k, 1000.0), np.full(k, 1000.0)
    for t in range(1, k):
        r = c[t] / c[t - 1] if c[t - 1] else 1.0
        nv[t] = nv[t - 1] * (r if v[t] < v[t - 1] else 1.0)
        pv[t] = pv[t - 1] * (r if v[t] > v[t - 1] else 1.0)
    return {"nvi": nv, "pvi": pv, "nvi_ema": ema(nv, n), "pvi_ema": ema(pv, n)}


def starc(df, n=6, atr_n=15, k=2.0):
    """dict: mid, upper, lower."""
    o, h, l, c, v = cols(df)
    m, a = sma(c, n), atr(h, l, c, atr_n)
    return {"mid": m, "upper": m + k * a, "lower": m - k * a}


def rwi(df, n=14):
    """dict: hi, lo (Poulos RWI, max over lookbacks 2..n)."""
    o, h, l, c, v = cols(df)
    t = tr(h, l, c)
    hi = np.full(len(c), np.nan)
    lo = np.full(len(c), np.nan)
    for k in range(2, n + 1):
        a = sma(t, k) * np.sqrt(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            rh = (h - _sh(l, k)) / a
            rl = (_sh(h, k) - l) / a
        hi = np.fmax(hi, rh)
        lo = np.fmax(lo, rl)
    return {"hi": hi, "lo": lo}


def vhf(df, n=28):
    c = cols(df)[3]
    num = S(c).rolling(n).max().to_numpy() - S(c).rolling(n).min().to_numpy()
    den = S(np.abs(np.diff(c, prepend=np.nan))).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def bop(df, n=14):
    o, h, l, c, v = cols(df)
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(rng > 0, (c - o) / rng, 0.0)
    return sma(raw, n)


def qstick(df, n=8):
    o, h, l, c, v = cols(df)
    return sma(c - o, n)


def tii(df, ma=60, n=30):
    c = cols(df)[3]
    dev = c - sma(c, ma)
    pos = S(np.maximum(np.nan_to_num(dev), 0)).rolling(n).sum().to_numpy()
    neg = S(np.maximum(-np.nan_to_num(dev), 0)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(pos + neg > 0, 100 * pos / (pos + neg), 50.0)
    out[: ma + n] = np.nan
    return out


def chande_kroll(df, p=10, x=1.0, q=9):
    """dict: stop_short (the upper line), stop_long (the lower line), up (close crosses over the upper line), dn."""
    o, h, l, c, v = cols(df)
    a = atr(h, l, c, p)
    fh = S(h).rolling(p).max().to_numpy() - x * a
    fl = S(l).rolling(p).min().to_numpy() + x * a
    ss = S(fh).rolling(q).max().to_numpy()
    sl = S(fl).rolling(q).min().to_numpy()
    return {"stop_short": ss, "stop_long": sl, "up": cross_up(c, ss), "dn": cross_down(c, sl)}


def volatility_stop(df, n=20, mult=2.0):
    """dict: stop, trend (+1 / -1), flip_up, flip_dn."""
    o, h, l, c, v = cols(df)
    a = atr(h, l, c, n) * mult
    k = len(c)
    stop, trend = np.full(k, np.nan), np.zeros(k)
    if k == 0:
        return {"stop": stop, "trend": trend, "flip_up": np.zeros(0, bool), "flip_dn": np.zeros(0, bool)}
    up, mx, mn, st = True, c[0], c[0], c[0] - a[0]
    for t in range(k):
        if np.isnan(a[t]):
            trend[t] = 1
            continue
        mx, mn = max(mx, c[t]), min(mn, c[t])
        st = max(st, mx - a[t]) if up else min(st, mn + a[t])
        nu = c[t] - st >= 0
        if nu != up:
            mx = mn = c[t]
            st = mx - a[t] if nu else mn + a[t]
            up = nu
        stop[t], trend[t] = st, 1 if up else -1
    pt = _sh(trend)
    return {"stop": stop, "trend": trend, "flip_up": (trend == 1) & (pt == -1), "flip_dn": (trend == -1) & (pt == 1)}
