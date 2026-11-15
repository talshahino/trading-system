"""Classic indicators the library did not have yet (scout, 15.11.2026 night 2). Ports of the published formulas
(sources below and in agents/indicators/__init__.py); where the source code is closed, our own reconstruction is
marked. All causal: the value at bar i uses bars <= i only (pivots / fractals are shifted to their confirmation bar).

stc            Schaff Trend Cycle 10 / 23 / 50, factor 0.5 (LazyBear's open Pine port of Doug Schaff's formula).
               https://www.tradingview.com/script/dbxXeuw2-Indicator-Schaff-Trend-Cycle-STC/
connors_rsi    Connors RSI 3 / 2 / 100 = mean of RSI3(close), RSI2(up/down streak), percent rank of the 1-bar ROC over
               100 bars (Connors Research). https://www.luxalgo.com/library/concept/connors-rsi/
smi_blau       Stochastic Momentum Index (William Blau): 100 x EMA(EMA(c - mid(HL, 10), 3), 3) / (0.5 x EMA(EMA(H10 -
               L10, 3), 3)), signal EMA 10. https://ta-lib.org/functions/smi.html
demarker       DeMarker 14 (Tom DeMark; MT4 standard): SMA(DeMax) / (SMA(DeMax) + SMA(DeMin)), 0..1.
               https://www.tradingview.com/script/CF1kNGYo-DeMarker-DeM/
rvi_vol        Relative Volatility Index (Donald Dorsey, TASC 1993, refined 1995): RSI-style ratio of the 10-bar stdev on
               up vs down closes, Wilder-smoothed 14; 'refined' = mean of the high- and low-based versions.
               https://www.luxalgo.com/library/concept/relative-volatility-index/
inertia        Dorsey Inertia: 20-bar least-squares (linreg) value of the refined RVI.
               https://www.tradingview.com/script/bzxmXFGd-Dorsey-Inertia/
safezone       Elder SafeZone stop (Come into my Trading Room): long stop = low[1] - 2.5 x mean of the down-penetrations
               of the last 10 bars, held at its 3-bar max; short mirror. https://www.luxalgo.com/library/concept/elder-safezone-stop/
kase_peak      Kase Peak Oscillator - OUR RECONSTRUCTION of the closed Kase formula from the MT4 public versions
               (log(h / l[k]) / sqrt(k) over k = 8..65, normalised by the 30-bar mean of the 9-bar stdev of log returns);
               peak line = mean + 1.5 sd of |KPO| over 50 bars. https://www.mql5.com/en/forum/173722
pfe            Polarized Fractal Efficiency 10 / 5 (Hans Hannula, TASC 1994): signed 100 x straight distance / path
               length, EMA 5. https://www.luxalgo.com/library/concept/polarized-fractal-efficiency/
ttf            Trend Trigger Factor 15 (M. H. Pee, TASC Dec 2004): 100 x (buy power - sell power) / (0.5 x (bp + sp)).
               https://www.traders.com/Documentation/FEEDbk_docs/2004/12/Abstracts_new/Pee/pee.html
gann_hilo      Gann HiLo Activator 3: close over SMA3(high) -> up (line = SMA3(low)), under SMA3(low) -> down.
               https://www.tradingview.com/script/PyFdxzcj-Gann-HiLo-Activator-Strategy/
pgo            Pretty Good Oscillator 89 (Mark Johnson): (close - SMA89) / EMA89(true range).
               https://www.luxalgo.com/library/concept/pretty-good-oscillator/
imi            Intraday Momentum Index 14 (Tushar Chande): 100 x sum(c - o | c > o) / (sum up + sum down).
               https://www.luxalgo.com/library/concept/intraday-momentum-index/
ewo            Elliott Wave Oscillator: SMA5(hl2) - SMA35(hl2) (the 5/35 oscillator).
               https://www.luxalgo.com/library/concept/5-35-oscillator/
kdj            KDJ 9 / 3 / 3: K, D = 1/3-smoothed RSV, J = 3K - 2D. https://www.kucoin.com/blog/What-is-KDJ-Indicator
vidya          VIDYA (Tushar Chande): EMA14 whose alpha is scaled by |CMO9|. https://www.tradingview.com/script/kSf8O7HZ-VIDYA-Chande/
bw_ao          Bill Williams Awesome Oscillator SMA5 - SMA34 of hl2 and Accelerator AC = AO - SMA5(AO).
               https://www.babypips.com/forexpedia/awesome-oscillator
bw_mfi         Bill Williams Market Facilitation Index (h - l) / tick volume with the 4 bar types (green, fade, fake,
               squat). https://www.luxalgo.com/library/concept/bill-williams-market-facilitation-index/
alligator      Bill Williams Alligator SMMA 13/8/5 of hl2 shifted 8/5/3 (shift = the value of bars ago -> causal) and
               Williams fractals confirmed 2 bars late. https://www.babypips.com/forexpedia/alligator-indicator
chop_zone      TradingView Chop Zone: angle (degrees) of EMA34 over a 30-bar high-low normalised scale, 9 colour bins.
               https://www.tradingview.com/support/solutions/43000589111-chop-zone-chop/
pvt            Price Volume Trend: cumulative ROC x tick volume, signal EMA 21.
vortex         Vortex Indicator 14 (Botes & Siepman, TASC 2010): VI+ / VI-.
psar           Parabolic SAR 0.02 / 0.02 / 0.2 (Wilder).
kst            Know Sure Thing (Martin Pring): ROC 10/15/20/30 smoothed 10/10/10/15, weights 1-4, signal SMA9.
stretch        Toby Crabel 'stretch': 10-day mean of min(high - open, open - low) of the previous days (completed days).
               https://store.traders.com/-v06-c09-playing-pdf.html
td_sequential  Tom DeMark TD Sequential setup (9 closes vs close[4]) and countdown (13 closes vs low / high[2]); a
               countdown is cancelled by an opposite setup (our simplification: no TDST / recycle rules).
"""
import numpy as np
import pandas as pd

from agents.indicators._core import (S, atr, cols, cross_down, cross_up, ema, highest, lowest, rma, rsi, shift, sma,
                                     stdev, tr)


def _stoch_raw(x, n):
    lo, hi = S(x).rolling(n).min().to_numpy(), S(x).rolling(n).max().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hi - lo > 0, (x - lo) / (hi - lo) * 100.0, np.nan)


def _ffill_nan(x):
    return S(x).ffill().to_numpy()


def stc(df, fast=23, slow=50, cycle=10, factor=0.5):
    """dict stc (0..100), up (cross over 25), dn (cross under 75). LazyBear's Pine: two passes of a stochastic of the
    MACD each smoothed with 'factor' (an EMA with alpha 0.5)."""
    o, h, l, c, v = cols(df)
    m = ema(c, fast) - ema(c, slow)
    n = len(c)
    f1 = _stoch_raw(m, cycle)
    pf = np.full(n, np.nan)
    for i in range(n):
        x = f1[i] if np.isfinite(f1[i]) else (pf[i - 1] if i else np.nan)
        pf[i] = x if (i == 0 or not np.isfinite(pf[i - 1])) else pf[i - 1] + factor * (x - pf[i - 1])
    f2 = _stoch_raw(pf, cycle)
    out = np.full(n, np.nan)
    for i in range(n):
        x = f2[i] if np.isfinite(f2[i]) else (out[i - 1] if i else np.nan)
        out[i] = x if (i == 0 or not np.isfinite(out[i - 1])) else out[i - 1] + factor * (x - out[i - 1])
    return {"stc": out, "up": cross_up(out, 25), "dn": cross_down(out, 75)}


def connors_rsi(df, n_rsi=3, n_streak=2, n_rank=100):
    """array 0..100."""
    o, h, l, c, v = cols(df)
    n = len(c)
    streak = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    roc = np.concatenate([[0.0], np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])])
    rank = S(roc).rolling(n_rank + 1).apply(lambda w: (w[:-1] < w[-1]).mean() * 100.0, raw=True).to_numpy()
    return (rsi(c, n_rsi) + rsi(streak, n_streak) + rank) / 3.0


def smi_blau(df, n=10, s1=3, s2=3, sig=10):
    """dict smi (-100..100), signal, up / dn (signal crosses)."""
    o, h, l, c, v = cols(df)
    hh, ll = S(h).rolling(n).max().to_numpy(), S(l).rolling(n).min().to_numpy()
    rel = c - (hh + ll) / 2.0
    rng = hh - ll
    num = ema(np.nan_to_num(ema(np.nan_to_num(rel), s1)), s2)
    den = ema(np.nan_to_num(ema(np.nan_to_num(rng), s1)), s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(den > 0, 100.0 * num / (0.5 * den), 0.0)
    k[: n - 1] = np.nan
    sg = ema(np.nan_to_num(k), sig).copy()
    sg[: n - 1] = np.nan
    return {"smi": k, "signal": sg, "up": cross_up(k, sg), "dn": cross_down(k, sg)}


def demarker(df, n=14):
    o, h, l, c, v = cols(df)
    dmax = np.clip(h - shift(h), 0, None)
    dmin = np.clip(shift(l) - l, 0, None)
    a, b = sma(np.nan_to_num(dmax), n), sma(np.nan_to_num(dmin), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a + b > 0, a / (a + b), 0.5)


def _rvi_side(x, n=10, sm=14):
    sd = stdev(x, n)
    d = np.diff(x, prepend=x[0])
    up = rma(np.where(d > 0, np.nan_to_num(sd), 0.0), sm)
    dn = rma(np.where(d < 0, np.nan_to_num(sd), 0.0), sm)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(up + dn > 0, 100.0 * up / (up + dn), 50.0)


def rvi_vol(df, n=10, sm=14):
    """dict rvi (close-based, 1993) and refined (mean of the high- and low-based ones, 1995), 0..100."""
    o, h, l, c, v = cols(df)
    return {"rvi": _rvi_side(c, n, sm), "refined": (_rvi_side(h, n, sm) + _rvi_side(l, n, sm)) / 2.0}


def _linreg_val(x, n):
    xs = np.arange(n, dtype=float)
    xm, den = xs.mean(), ((xs - np.arange(n).mean()) ** 2).sum()

    def f(w):
        ym = w.mean()
        b = ((xs - xm) * (w - ym)).sum() / den
        return ym + b * (n - 1 - xm)
    return S(x).rolling(n).apply(f, raw=True).to_numpy()


def inertia(df, n=20):
    return _linreg_val(rvi_vol(df)["refined"], n)


def safezone(df, n=10, k=2.5, hold=3):
    """dict long_stop / short_stop (the trailing SafeZone levels known at the close of bar i)."""
    o, h, l, c, v = cols(df)
    dn_pen = np.clip(shift(l) - l, 0, None)
    up_pen = np.clip(h - shift(h), 0, None)
    dn_cnt = S((dn_pen > 0).astype(float)).rolling(n).sum().to_numpy()
    up_cnt = S((up_pen > 0).astype(float)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        dn_avg = np.where(dn_cnt > 0, S(np.nan_to_num(dn_pen)).rolling(n).sum().to_numpy() / dn_cnt, 0.0)
        up_avg = np.where(up_cnt > 0, S(np.nan_to_num(up_pen)).rolling(n).sum().to_numpy() / up_cnt, 0.0)
    ls = l - k * dn_avg
    ss = h + k * up_avg
    return {"long_stop": S(ls).rolling(hold, min_periods=1).max().to_numpy(),
            "short_stop": S(ss).rolling(hold, min_periods=1).min().to_numpy()}


def kase_peak(df, k1=8, k2=65, dev_n=9, avg_n=30, peak_n=50, mult=1.5):
    """dict kpo, peak (the peak-out line), up_peak / dn_peak (KPO beyond +/- peak and turning back). Our
    reconstruction (closed formula) - see the module docstring."""
    o, h, l, c, v = cols(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(c / shift(c))
        lh, ll = np.log(h), np.log(l)
    dev = stdev(np.nan_to_num(lr), dev_n)
    avg = sma(dev, avg_n)
    n = len(c)
    mx = np.full(n, -np.inf)
    mn = np.full(n, np.inf)
    for k in range(k1, k2 + 1):
        sk = np.sqrt(k)
        mx = np.fmax(mx, (lh - shift(ll, k)) / sk)
        mn = np.fmin(mn, (ll - shift(lh, k)) / sk)
    with np.errstate(divide="ignore", invalid="ignore"):
        kpo = np.where(avg > 0, (mx + mn) / avg, np.nan)
    kpo[~np.isfinite(kpo)] = np.nan
    a = np.abs(kpo)
    peak = S(a).rolling(peak_n).mean().to_numpy() + mult * S(a).rolling(peak_n).std(ddof=0).to_numpy()
    pk1, kp1 = shift(peak), shift(kpo)
    up_peak = (kp1 > pk1) & (kpo < kp1)     # a top peak-out: KPO was over the line and turned down
    dn_peak = (kp1 < -pk1) & (kpo > kp1)
    return {"kpo": kpo, "peak": peak, "up_peak": np.nan_to_num(up_peak).astype(bool),
            "dn_peak": np.nan_to_num(dn_peak).astype(bool)}


def pfe(df, n=10, sm=5):
    o, h, l, c, v = cols(df)
    d = c - shift(c, n)
    straight = np.sqrt(d ** 2 + n ** 2)
    path = S(np.sqrt(np.diff(c, prepend=c[0]) ** 2 + 1.0)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(path > 0, np.sign(d) * 100.0 * straight / path, 0.0)
    out = ema(np.nan_to_num(raw), sm).copy()
    out[:n] = np.nan
    return out


def ttf(df, n=15):
    o, h, l, c, v = cols(df)
    hh, ll = highest(h, n), lowest(l, n)
    bp = hh - shift(ll, n)
    sp = shift(hh, n) - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bp + sp != 0, 100.0 * (bp - sp) / (0.5 * (bp + sp)), 0.0)


def gann_hilo(df, n=3):
    """dict line (the activator), trend (+1 / -1), flip_up / flip_dn."""
    o, h, l, c, v = cols(df)
    hi, lo = sma(h, n), sma(l, n)
    hi1, lo1 = shift(hi), shift(lo)
    t = np.zeros(len(c))
    for i in range(1, len(c)):
        if np.isfinite(hi1[i]) and c[i] > hi1[i]:
            t[i] = 1
        elif np.isfinite(lo1[i]) and c[i] < lo1[i]:
            t[i] = -1
        else:
            t[i] = t[i - 1]
    line = np.where(t > 0, lo, hi)
    tp = shift(t)
    return {"line": line, "trend": t, "flip_up": (t == 1) & (tp != 1), "flip_dn": (t == -1) & (tp != -1)}


def pgo(df, n=89):
    o, h, l, c, v = cols(df)
    d = ema(tr(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, (c - sma(c, n)) / d, np.nan)


def imi(df, n=14):
    o, h, l, c, v = cols(df)
    up = S(np.where(c > o, c - o, 0.0)).rolling(n).sum().to_numpy()
    dn = S(np.where(c < o, o - c, 0.0)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(up + dn > 0, 100.0 * up / (up + dn), 50.0)


def ewo(df, fast=5, slow=35):
    o, h, l, c, v = cols(df)
    m = (h + l) / 2.0
    return sma(m, fast) - sma(m, slow)


def kdj(df, n=9, m1=3, m2=3):
    """dict k, d, j."""
    o, h, l, c, v = cols(df)
    hh, ll = S(h).rolling(n).max().to_numpy(), S(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsv = np.where(hh - ll > 0, (c - ll) / (hh - ll) * 100.0, 50.0)
    k = np.full(len(c), 50.0)
    d = np.full(len(c), 50.0)
    for i in range(len(c)):
        r = rsv[i] if np.isfinite(rsv[i]) else 50.0
        kp, dp = (k[i - 1], d[i - 1]) if i else (50.0, 50.0)
        k[i] = (m1 - 1) / m1 * kp + r / m1
        d[i] = (m2 - 1) / m2 * dp + k[i] / m2
    k[: n - 1] = np.nan
    d[: n - 1] = np.nan
    return {"k": k, "d": d, "j": 3 * k - 2 * d}


def vidya(df, n=14, cmo_n=9):
    o, h, l, c, v = cols(df)
    d = np.diff(c, prepend=c[0])
    up = S(np.clip(d, 0, None)).rolling(cmo_n).sum().to_numpy()
    dn = S(np.clip(-d, 0, None)).rolling(cmo_n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.abs(np.where(up + dn > 0, (up - dn) / (up + dn), 0.0))
    alpha = 2.0 / (n + 1)
    out = np.full(len(c), np.nan)
    for i in range(len(c)):
        if not np.isfinite(k[i]):
            continue
        prev = out[i - 1] if i and np.isfinite(out[i - 1]) else c[i]
        out[i] = alpha * k[i] * c[i] + (1 - alpha * k[i]) * prev
    return out


def bw_ao(df):
    """dict ao, ac, ao_green (AO rising), ac_green, saucer_up / saucer_dn (Williams saucer: AO on one side of zero,
    two bars against, then one bar with), twin_up / twin_dn (two AO troughs under zero, the second higher, no zero
    cross between, then a rising bar; mirror for peaks)."""
    o, h, l, c, v = cols(df)
    m = (h + l) / 2.0
    ao = sma(m, 5) - sma(m, 34)
    ac = ao - sma(ao, 5)
    g = ao > shift(ao)
    acg = ac > shift(ac)
    a1, a2, a3 = shift(ao), shift(ao, 2), shift(ao, 3)
    with np.errstate(invalid="ignore"):
        sau_up = (ao > 0) & (a1 > 0) & (a2 > 0) & (a2 < a3) & (a1 < a2) & (ao > a1)
        sau_dn = (ao < 0) & (a1 < 0) & (a2 < 0) & (a2 > a3) & (a1 > a2) & (ao < a1)
    n = len(c)
    tw_up, tw_dn = np.zeros(n, bool), np.zeros(n, bool)
    t1 = p1 = None    # last AO trough under zero / peak over zero since the last zero cross
    for i in range(2, n):
        if not (np.isfinite(ao[i]) and np.isfinite(ao[i - 1]) and np.isfinite(ao[i - 2])):
            continue
        if (ao[i] >= 0) != (ao[i - 1] >= 0):
            t1 = p1 = None
        # a trough at bar i-1 is known at bar i (AO turned up)
        if ao[i] < 0 and ao[i - 1] < ao[i - 2] and ao[i] > ao[i - 1]:
            if t1 is not None and ao[i - 1] > t1:
                tw_up[i] = True
            t1 = ao[i - 1]
        if ao[i] > 0 and ao[i - 1] > ao[i - 2] and ao[i] < ao[i - 1]:
            if p1 is not None and ao[i - 1] < p1:
                tw_dn[i] = True
            p1 = ao[i - 1]
    return {"ao": ao, "ac": ac, "ao_green": g, "ac_green": acg, "saucer_up": np.asarray(sau_up, bool),
            "saucer_dn": np.asarray(sau_dn, bool), "twin_up": tw_up, "twin_dn": tw_dn}


def bw_mfi(df):
    """dict mfi, green (mfi up, vol up), fade (both down), fake (mfi up, vol down), squat (mfi down, vol up)."""
    o, h, l, c, v = cols(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(v > 0, (h - l) / v, np.nan)
    mu, vu = m > shift(m), v > shift(v)
    md, vd = m < shift(m), v < shift(v)
    return {"mfi": m, "green": mu & vu, "fade": md & vd, "fake": mu & vd, "squat": md & vu}


def _smma(x, n):
    return rma(x, n)


def alligator(df):
    """dict jaw / teeth / lips (SMMA 13/8/5 of hl2, as plotted shifted forward 8/5/3 bars -> on bar i the value
    computed at bar i - shift), fractal_hi / fractal_lo (the last Williams fractal, known 2 bars after its bar)."""
    o, h, l, c, v = cols(df)
    m = (h + l) / 2.0
    jaw, teeth, lips = shift(_smma(m, 13), 8), shift(_smma(m, 8), 5), shift(_smma(m, 5), 3)
    n = len(c)
    fh, fl = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(4, n):
        j = i - 2
        if h[j] > max(h[j - 2], h[j - 1], h[j + 1], h[j + 2]):
            fh[i] = h[j]
        if l[j] < min(l[j - 2], l[j - 1], l[j + 1], l[j + 2]):
            fl[i] = l[j]
    return {"jaw": jaw, "teeth": teeth, "lips": lips, "fractal_hi": _ffill_nan(fh), "fractal_lo": _ffill_nan(fl)}


def chop_zone(df, n=30, ema_n=34):
    """dict angle (degrees) and zone (+4 .. -4: +4 = darkest turquoise, angle >= 5; 0 = yellow, |angle| < 1)."""
    o, h, l, c, v = cols(df)
    avg = (h + l + c) / 3.0
    hh, ll = S(h).rolling(n).max().to_numpy(), S(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = 25.0 / (hh - ll) * ll
    e = ema(c, ema_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        x1, y1 = 0.0, 0.0
        x2, y2 = 1.0, ((shift(e) - e) / avg) * rng
        cc = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        ang0 = np.round(180.0 * np.arccos((x2 - x1) / cc) / np.pi)
    ang = np.where(y2 > 0, -ang0, ang0)
    zone = np.select([ang >= 5, ang >= 3.57, ang >= 2.14, ang >= 0.71, ang <= -5, ang <= -3.57, ang <= -2.14,
                      ang <= -0.71], [4, 3, 2, 1, -4, -3, -2, -1], 0).astype(float)
    zone[~np.isfinite(ang)] = np.nan
    return {"angle": ang, "zone": zone}


def pvt(df, sig=21):
    o, h, l, c, v = cols(df)
    roc = np.concatenate([[0.0], np.diff(c) / np.where(c[:-1] == 0, np.nan, c[:-1])])
    p = np.cumsum(np.nan_to_num(roc) * v)
    s = ema(p, sig)
    return {"pvt": p, "signal": s, "up": cross_up(p, s), "dn": cross_down(p, s)}


def vortex(df, n=14):
    o, h, l, c, v = cols(df)
    vmp = np.abs(h - shift(l))
    vmm = np.abs(l - shift(h))
    t = S(tr(h, l, c)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = S(np.nan_to_num(vmp)).rolling(n).sum().to_numpy() / t
        vm = S(np.nan_to_num(vmm)).rolling(n).sum().to_numpy() / t
    return {"vi_plus": vp, "vi_minus": vm, "up": cross_up(vp, vm), "dn": cross_down(vp, vm)}


def psar(df, start=0.02, step=0.02, mx=0.2):
    """dict sar (the dot for the NEXT bar computed at bar i's close -> known at i), trend +1 / -1, flip_up / flip_dn."""
    o, h, l, c, v = cols(df)
    n = len(c)
    sar = np.full(n, np.nan)
    tr_ = np.zeros(n)
    if n < 2:
        return {"sar": sar, "trend": tr_, "flip_up": np.zeros(n, bool), "flip_dn": np.zeros(n, bool)}
    up = c[1] >= c[0]
    ep = h[1] if up else l[1]
    s = l[0] if up else h[0]
    af = start
    for i in range(1, n):
        if up:
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], start
            else:
                if h[i] > ep:
                    ep, af = h[i], min(af + step, mx)
        else:
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], start
            else:
                if l[i] < ep:
                    ep, af = l[i], min(af + step, mx)
        nxt = s + af * (ep - s)
        if up:
            nxt = min(nxt, l[i], l[i - 1])
        else:
            nxt = max(nxt, h[i], h[i - 1])
        s = nxt
        sar[i] = s
        tr_[i] = 1 if up else -1
    tp = shift(tr_)
    return {"sar": sar, "trend": tr_, "flip_up": (tr_ == 1) & (tp == -1), "flip_dn": (tr_ == -1) & (tp == 1)}


def kst(df):
    o, h, l, c, v = cols(df)

    def roc(k):
        return 100.0 * (c / shift(c, k) - 1.0)
    k = sma(roc(10), 10) + 2 * sma(roc(15), 10) + 3 * sma(roc(20), 10) + 4 * sma(roc(30), 15)
    s = sma(k, 9)
    return {"kst": k, "signal": s, "up": cross_up(k, s), "dn": cross_down(k, s)}


def stretch(df, n=10, tz="America/New_York"):
    """dict day_open (the session's first open in tz), stretch (10-day mean of min(h - o, o - l) of the PREVIOUS
    completed days), up_lvl / dn_lvl = open +/- stretch."""
    o, h, l, c, v = cols(df)
    loc = df.index.tz_convert(tz)
    day = pd.Series(loc.date)
    g = pd.DataFrame({"d": day, "o": o, "h": h, "l": l})
    dd = g.groupby("d").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"))
    st = np.minimum(dd["h"] - dd["o"], dd["o"] - dd["l"]).rolling(n).mean().shift(1)
    m = g[["d"]].join(st.rename("st"), on="d")
    dopen = g.groupby("d")["o"].transform("first").to_numpy()
    s = m["st"].to_numpy(dtype=float)
    return {"day_open": dopen, "stretch": s, "up_lvl": dopen + s, "dn_lvl": dopen - s}


def td_sequential(df):
    """dict buy_setup / sell_setup (count 0..9, 9 = completed this bar), buy_cd / sell_cd (countdown 0..13,
    13 = completed this bar)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    bs, ss = np.zeros(n), np.zeros(n)
    bcd, scd = np.zeros(n), np.zeros(n)
    b_on = s_on = False
    bc = sc = 0
    for i in range(4, n):
        bs[i] = bs[i - 1] + 1 if c[i] < c[i - 4] and bs[i - 1] < 9 else (1 if c[i] < c[i - 4] else 0)
        ss[i] = ss[i - 1] + 1 if c[i] > c[i - 4] and ss[i - 1] < 9 else (1 if c[i] > c[i - 4] else 0)
        if bs[i] == 9:
            b_on, bc, s_on, sc = True, 0, False, 0
        if ss[i] == 9:
            s_on, sc, b_on, bc = True, 0, False, 0
        if b_on and c[i] <= l[i - 2]:
            bc += 1
            bcd[i] = bc
            if bc >= 13:
                b_on, bc = False, 0
        if s_on and c[i] >= h[i - 2]:
            sc += 1
            scd[i] = sc
            if sc >= 13:
                s_on, sc = False, 0
    return {"buy_setup": bs, "sell_setup": ss, "buy_cd": bcd, "sell_cd": scd}
