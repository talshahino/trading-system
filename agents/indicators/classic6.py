"""Indicators the library lacked (scout 16.11.2026, night). Each is a port of the published formula, or our own
reconstruction where the code is closed (said so per indicator). No repaint: every value at bar i uses bars <= i only;
pivots (pitchfork) are used from the bar where they are confirmed.

- correlation_trend: Ehlers Correlation Trend Indicator (TASC 2020.05,
  https://www.traders.com/Documentation/FEEDbk_docs/2020/05/TradersTips.html ; open source
  https://www.tradingview.com/script/azjhI4tC-Correlation-Trend-Indicator-Dr-John-Ehlers ): Pearson correlation of
  the last n closes with a straight rising line (n 20), -1..1; > 0.5 / < -0.5 = trend.
- ttm_trend: TTM Trend (John Carter; closed - ours from ProRealCode 'TTM trend on price'
  https://www.prorealcode.com/prorealtime-indicators/ttm-trend-price/ and the TOS description): the close against the
  average of the last 6 Heikin-Ashi bodies' midpoints; up / down bar colour.
- ttm_scalper: TTM Scalper Alert (Carter; closed - ours from the description of
  https://in.tradingview.com/script/CddpDBGA-TTM-scalper-indicator-Strategy ): a 3-bar reversal square marked when the
  third bar confirms - the swing bar's high (low) is the highest (lowest) of 5 bars and the two bars after it close
  lower (higher) each time. Known on that third bar.
- murrey_math: Murrey Math lines (LuxAlgo summary https://www.luxalgo.com/library/indicator/murrey-math-levels/ , the
  classic MT4 'Murrey_Math_Line_X' octave table): 64-bar frame, the range snapped to the octave of the price's
  'fractal' (the 100000 ... 0.1953125 table), 0/8 ... 8/8 lines. Ours: without the MT4 x1-x6 fine adjustments.
- chaikin_volatility: Chaikin Volatility (https://ta-lib.org/functions/cvi.html ,
  https://www.incrediblecharts.com/indicators/chaikin_volatility.php ): EMA10(high - low), its 10-bar % change.
- standard_error_bands: Jon Andersen's Standard Error Bands (TASC 1996.09,
  https://www.luxalgo.com/library/indicator/standard-error-bands/ ): linear-regression(21) endpoint +/- 2 standard
  errors of the estimate, each smoothed SMA 3; width = band distance / close.
- projection: Mel Widner's Projection Bands and Projection Oscillator (TASC 1995,
  https://www.luxalgo.com/library/concept/projection-bands/ ; https://www.prorealcode.com/prorealtime-indicators/projection-oscillator ):
  the last n highs (lows) projected forward along the regression slope of the highs (lows); upper = their max, lower
  = their min; PO = 100 x (close - lower) / (upper - lower), trigger EMA 5. n 14.
- wave_pm: Mark Whistler's Wave PM (https://www.prorealcode.com/prorealtime-indicators/mark-whistlers-wave-pm ,
  https://www.tradingview.com/script/l9Oqp36g-Wave-PM ): dev = StdDev(14) x 1.25; tanh(dev / RMS of dev over 100
  bars), 0..1 - low = volatility energy stored, rising through 0.5 = release.
- sentiment_zone: Walid Khalil's Sentiment Zone Oscillator (TASC 2012.05,
  https://www.traders.com/Documentation/FEEDbk_docs/2012/05/Khalil.html ): TEMA 14 of +1 / -1 (close up / not up),
  ours scaled x 100 (-100..100); Khalil's dynamic zones: over the last 30 bars, overbought = min + 95% of the range,
  oversold = min + 5%; trend EMA 60.
- rainbow: Mel Widner's Rainbow Oscillator and Rainbow Bandwidth (TASC 1997.07,
  https://www.luxalgo.com/library/concept/rainbow-oscillator/ , https://help.ctrader.com/indicators/built-in/oscillators/rainbow-oscillator/ ):
  10 recursive SMA 2 of the close; RO = 100 x (close - their mean) / (10-bar close range); RB = 100 x (max - min of
  the 10 averages) / that range.
- madrid_ribbon: Madrid Moving Average Ribbon (open source https://in.tradingview.com/script/WH01CtjA-Madrid-MA-Ribbon-Bar ):
  EMAs 5, 10 ... 90 coloured lime (rising, over EMA100), green (rising, under), maroon (falling, over), red (falling,
  under); score = (lime - red) / 18, -1..1.
- pitchfork: automatic Andrews' Pitchfork (https://www.luxalgo.com/library/indicator/andrews-pitchfork/ ,
  https://www.tradingsetupsreview.com/andrews-pitchfork-trading-strategy/ ): from the last three alternating swing
  pivots (5 bars each side, confirmed): median line from A through the middle of B-C, parallels through B and C.
- kase_devstop: Cynthia Kase's Dev-Stops (https://www.prorealcode.com/prorealtime-indicators/kase-dev-stop/ ,
  https://www.luxalgo.com/library/concept/kase-studies/ ): 2-bar true range DTR, its SMA and std 30; trend by SMA
  10 vs 21; long stops = 20-bar highest close - (avg + k x std), k 0 (warning), 1, 2.2, 3.6 (mirror for shorts). Ours:
  the 'highest close' anchor (Kase anchors at the trade's extreme).
"""
import numpy as np
import pandas as pd

from agents.indicators._core import S, atr, cols, ema, highest, lowest, pivots, sma


def correlation_trend(df, n=20):
    o, h, l, c, v = cols(df)
    t = pd.Series(np.arange(len(c), dtype=float))
    cti = S(c).rolling(n).corr(t).to_numpy()
    return {"cti": cti, "up": cti > 0.5, "down": cti < -0.5}


def ttm_trend(df, n=6):
    o, h, l, c, v = cols(df)
    hac = (o + h + l + c) / 4.0
    hao = np.empty(len(c))
    if len(c):
        hao[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        hao[i] = (hao[i - 1] + hac[i - 1]) / 2.0
    mid = sma((hao + hac) / 2.0, n)
    up = c > mid
    return {"mid": mid, "up": up, "trend": np.where(np.isnan(mid), 0, np.where(up, 1, -1))}


def ttm_scalper(df):
    o, h, l, c, v = cols(df)
    n = len(c)
    hs, ls, cs = S(h), S(l), S(c)
    h2, l2, c2, c1 = hs.shift(2), ls.shift(2), cs.shift(2), cs.shift(1)
    hmax = hs.rolling(5).max()
    lmin = ls.rolling(5).min()
    sell = ((h2 >= hmax) & (c1 < c2) & (cs < c1)).to_numpy()
    buy = ((l2 <= lmin) & (c1 > c2) & (cs > c1)).to_numpy()
    return {"buy": buy, "sell": sell, "pivot_low": np.where(buy, l2.to_numpy(), np.nan),
            "pivot_high": np.where(sell, h2.to_numpy(), np.nan)}


def _mm_fractal(v2):
    f = np.full(len(v2), 0.1953125)
    table = [(25000, 250000, 100000.0), (2500, 25000, 10000.0), (250, 2500, 1000.0), (25, 250, 100.0),
             (12.5, 25, 12.5), (6.25, 12.5, 12.5), (3.125, 6.25, 6.25), (1.5625, 3.125, 3.125),
             (0.390625, 1.5625, 1.5625)]
    for lo, hi, val in table:
        f = np.where((v2 > lo) & (v2 <= hi), val, f)
    return f


def murrey_math(df, n=64):
    o, h, l, c, v = cols(df)
    v1, v2 = lowest(l, n), highest(h, n)
    rng = v2 - v1
    with np.errstate(all="ignore"):
        fr = _mm_fractal(v2)
        s = np.floor(np.log(fr / rng) / np.log(2))
        octave = fr * 0.5 ** s
        mn = np.floor(v1 / octave) * octave
        mx = np.where(mn + octave > v2, mn + octave, mn + 2 * octave)
    step = (mx - mn) / 8.0
    lines = {f"l{k}": mn + k * step for k in range(9)}
    lines.update({"m2": mn - 2 * step, "p2": mx + 2 * step, "step": step})
    with np.errstate(all="ignore"):
        lines["pos"] = (c - mn) / step  # 0..8 inside the frame
    return lines


def chaikin_volatility(df, n=10, roc=10):
    o, h, l, c, v = cols(df)
    e = ema(h - l, n)
    prev = S(e).shift(roc).to_numpy()
    with np.errstate(all="ignore"):
        cv = 100.0 * (e - prev) / prev
    return {"cv": cv}


def _rolling_linreg(y, n):
    """Endpoint, slope and standard error of the estimate of the OLS line over the last n values (bar i uses i-n+1..i)."""
    y = np.asarray(y, float)
    m = len(y)
    end = np.full(m, np.nan)
    slope = np.full(m, np.nan)
    se = np.full(m, np.nan)
    if m < n:
        return end, slope, se
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(y, n)
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    ym = W.mean(axis=1)
    b = ((W - ym[:, None]) * (x - xm)).sum(axis=1) / sxx
    a = ym - b * xm
    fit = a[:, None] + b[:, None] * x
    res = ((W - fit) ** 2).sum(axis=1)
    end[n - 1:] = a + b * (n - 1)
    slope[n - 1:] = b
    se[n - 1:] = np.sqrt(res / max(n - 2, 1))
    return end, slope, se


def standard_error_bands(df, n=21, k=2.0, smooth=3):
    o, h, l, c, v = cols(df)
    end, slope, se = _rolling_linreg(c, n)
    mid = sma(end, smooth)
    up = sma(end + k * se, smooth)
    lo = sma(end - k * se, smooth)
    with np.errstate(all="ignore"):
        width = (up - lo) / c
    return {"mid": mid, "upper": up, "lower": lo, "slope": slope, "width": width}


def projection(df, n=14, trig=5):
    o, h, l, c, v = cols(df)
    _, mh, _ = _rolling_linreg(h, n)
    _, ml, _ = _rolling_linreg(l, n)
    hs, ls = S(h), S(l)
    up = np.full(len(c), -np.inf)
    lo = np.full(len(c), np.inf)
    for k in range(n):
        up = np.fmax(up, hs.shift(k).to_numpy() + k * mh)
        lo = np.fmin(lo, ls.shift(k).to_numpy() + k * ml)
    bad = np.isnan(mh) | np.isnan(ml)
    up[bad] = np.nan
    lo[bad] = np.nan
    with np.errstate(all="ignore"):
        po = 100.0 * (c - lo) / (up - lo)
    return {"upper": up, "lower": lo, "po": po, "trigger": ema(np.nan_to_num(po, nan=50.0), trig)}


def wave_pm(df, n=14, mult=1.25, length=100):
    o, h, l, c, v = cols(df)
    dev = S(c).rolling(n).std(ddof=0).to_numpy() * mult
    rms = np.sqrt(S(dev ** 2).rolling(length, min_periods=n).mean().to_numpy())
    with np.errstate(all="ignore"):
        t = np.where(rms > 0, dev / rms, 0.0)
    return {"wpm": np.tanh(t)}


def _tema(x, n):
    e1 = ema(x, n)
    e2 = ema(e1, n)
    return 3 * e1 - 3 * e2 + ema(e2, n)


def sentiment_zone(df, n=14, zone=30, pct=0.95, trend=60):
    o, h, l, c, v = cols(df)
    r = np.where(S(c).diff().to_numpy() > 0, 1.0, -1.0)
    szo = 100.0 * _tema(r, n)
    hi, lo = highest(szo, zone), lowest(szo, zone)
    rng = hi - lo
    return {"szo": szo, "ob": lo + pct * rng, "os": lo + (1 - pct) * rng, "trend_up": c > ema(c, trend)}


def rainbow(df, depth=10, look=10):
    o, h, l, c, v = cols(df)
    mas = []
    x = c
    for _ in range(depth):
        x = sma(x, 2)
        mas.append(x)
    M = np.vstack(mas)
    hh, ll = highest(c, look), lowest(c, look)
    rng = hh - ll
    with np.errstate(all="ignore"):
        ro = np.where(rng > 0, 100.0 * (c - M.mean(axis=0)) / rng, 0.0)
        rb = np.where(rng > 0, 100.0 * (M.max(axis=0) - M.min(axis=0)) / rng, 0.0)
    return {"ro": ro, "rb": rb, "signal": sma(ro, 9)}


def madrid_ribbon(df):
    o, h, l, c, v = cols(df)
    ref = ema(c, 100)
    lime = np.zeros(len(c))
    red = np.zeros(len(c))
    for p in range(5, 95, 5):
        e = ema(c, p)
        rising = e > S(e).shift(1).to_numpy()
        lime += rising & (e > ref)
        red += ~rising & (e < ref)
    return {"score": (lime - red) / 18.0, "lime": lime, "red": red}


def pitchfork(df, k=5):
    o, h, l, c, v = cols(df)
    n = len(c)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    mid = np.full(n, np.nan)
    up = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    d = np.zeros(n, int)
    piv = []  # alternating (idx, price, type +1 high / -1 low)
    fork = None
    for i in range(n):
        for conf, val, idx, typ in ((phc[i], phv[i], phi[i], 1), (plc[i], plv[i], pli[i], -1)):
            if not conf:
                continue
            if piv and piv[-1][2] == typ:
                if (typ == 1 and val > piv[-1][1]) or (typ == -1 and val < piv[-1][1]):
                    piv[-1] = (idx, val, typ)
            else:
                piv.append((idx, val, typ))
            if len(piv) >= 3:
                (ia, pa, ta), (ib, pb, _), (ic, pc, _) = piv[-3:]
                im, pm = (ib + ic) / 2.0, (pb + pc) / 2.0
                if im != ia:
                    s = (pm - pa) / (im - ia)
                    fork = (ia, pa, ib, pb, ic, pc, s, -ta)  # A low -> bull fork (+1)
        if fork is not None:
            ia, pa, ib, pb, ic, pc, s, dr = fork
            mid[i] = pa + s * (i - ia)
            a, b = pb + s * (i - ib), pc + s * (i - ic)
            up[i], lo[i] = max(a, b), min(a, b)
            d[i] = dr
    return {"median": mid, "upper": up, "lower": lo, "dir": d}


def kase_devstop(df, n=30, look=20):
    o, h, l, c, v = cols(df)
    hs, ls = S(h), S(l)
    dtr = (np.fmax(h, hs.shift(1).to_numpy()) - np.fmin(l, ls.shift(1).to_numpy()))
    avg = sma(dtr, n)
    sd = S(dtr).rolling(n).std(ddof=0).to_numpy()
    up = sma(c, 10) > sma(c, 21)
    hc, lc = highest(c, look), lowest(c, look)
    out = {"trend_up": up}
    for name, k in (("warn", 0.0), ("ds1", 1.0), ("ds2", 2.2), ("ds3", 3.6)):
        out[name + "_long"] = hc - (avg + k * sd)
        out[name + "_short"] = lc + (avg + k * sd)
    return out
