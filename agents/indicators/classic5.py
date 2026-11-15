"""Classic / community indicators the library lacked (scout 15.11.2026, night 3). Each is a port of the published
formula; the source of each is in its docstring and in INDICATORS. No repaint: rolling / recursive values at bar i use
bars <= i only; fractals are shifted to their confirmation bar (2 bars late).

- vix_fix: CM Williams Vix Fix (ChrisMoody, open source https://www.tradingview.com/script/og7JPrRA-CM-Williams-Vix-Fix-Finds-Market-Bottoms/;
  tops / inverse version https://www.tradingview.com/script/wYbPyvTV-Williams-Vix-Fix-Bottoms-and-Tops/): WVF =
  (highest close 22 - low) / highest close 22 x 100; alert when WVF >= its BB 20 / 2 upper band or >= 0.85 x its
  50-bar high. Top version: (high - lowest close 22) / lowest close 22 x 100.
- bbtrend: John Bollinger's BBTrend (2024, https://www.tradingview.com/support/solutions/43000726749/):
  (|lower20 - lower50| - |upper20 - upper50|) / middle20 x 100.
- vzo / pzo: Walid Khalil's Volume / Price Zone Oscillators (TASC 2011.05,
  https://www.traders.com/Documentation/FEEDbk_docs/2011/05/Khalil.html): 100 x EMA14(sign(c - c[1]) x v) / EMA14(v)
  (tick volume), PZO the same with the close instead of the volume.
- twiggs_mf: Twiggs Money Flow (https://www.incrediblecharts.com/indicators/twiggs_money_flow.php): true-range
  close location x volume, Wilder 21 smoothing of both.
- rmi: Relative Momentum Index (Altman, TASC 1993; https://www.luxalgo.com/library/concept/relative-momentum-index/):
  RSI of the m-bar momentum, n 14, m 5.
- pmo: DecisionPoint Price Momentum Oscillator (Carl Swenlin,
  https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/decisionpoint-price-momentum-oscillator-pmo):
  ROC1 smoothed with alpha 2/35, x 10, smoothed with alpha 2/20; signal EMA10.
- t3: Tillson T3 (TASC 1998; https://www.luxalgo.com/library/concept/t3/): six EMAs, volume factor 0.7.
- alma: Arnaud Legoux MA (window 9, offset 0.85, sigma 6).
- ash: Absolute Strength Histogram, RSI method (MQL5 codebase https://www.mql5.com/en/code/21429; backtrader recipe
  https://backtrader.com/recipes/indicators/ash/ash/): bulls / bears of the 1-bar change, SMA 9 then SMA 3 (the
  original's default MA is selectable - ours SMA).
- trendilo: Trendilo (dudeowns, open source https://www.tradingview.com/script/h5kMWewu-Trendilo-OPEN-SOURCE/):
  ALMA(50, 0.85, 6) of the 1-bar percent change against its RMS band (x 1.0).
- kairi: Kairi Relative Index = (close - SMA25) / SMA25 x 100 (Japanese classic).
- fractal_bands: Fractal Chaos Bands (Edward Dobson; https://www.tradingview.com/script/U1iz6Mxb-Fractal-Chaos-Bands/ -
  ours from the definition): the last confirmed 2/2 Williams fractal high / low, known 2 bars after the fractal bar."""
import numpy as np

from agents.indicators._core import S, cols, ema, last_pivots, rma, sma, stdev


def vix_fix(df, pd_=22, bbl=20, mult=2.0, lb=50, ph=0.85):
    """dict wvf, upper, range_hi, bottom (WVF alert = fear spike), wvf_top, top (the inverse alert)."""
    o, h, l, c, v = cols(df)
    hc = S(c).rolling(pd_, min_periods=1).max().to_numpy()
    lc = S(c).rolling(pd_, min_periods=1).min().to_numpy()
    wvf = (hc - l) / hc * 100
    up = sma(wvf, bbl) + mult * stdev(wvf, bbl)
    rh = S(wvf).rolling(lb, min_periods=1).max().to_numpy() * ph
    wt = (h - lc) / lc * 100
    upt = sma(wt, bbl) + mult * stdev(wt, bbl)
    rht = S(wt).rolling(lb, min_periods=1).max().to_numpy() * ph
    bottom = np.nan_to_num(wvf >= up) | (wvf >= rh)
    top = np.nan_to_num(wt >= upt) | (wt >= rht)
    return {"wvf": wvf, "upper": up, "range_hi": rh, "bottom": np.asarray(bottom, bool), "wvf_top": wt,
            "top": np.asarray(top, bool)}


def bbtrend(df, fast=20, slow=50, k=2.0):
    c = cols(df)[3]
    m1, s1 = sma(c, fast), stdev(c, fast)
    m2, s2 = sma(c, slow), stdev(c, slow)
    lo1, up1, lo2, up2 = m1 - k * s1, m1 + k * s1, m2 - k * s2, m2 + k * s2
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.abs(lo1 - lo2) - np.abs(up1 - up2)) / m1 * 100


def vzo(df, n=14):
    o, h, l, c, v = cols(df)
    sg = np.sign(np.diff(c, prepend=c[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(100 * ema(sg * v, n) / ema(v, n))


def pzo(df, n=14):
    c = cols(df)[3]
    sg = np.sign(np.diff(c, prepend=c[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(100 * ema(sg * c, n) / ema(c, n))


def twiggs_mf(df, n=21):
    o, h, l, c, v = cols(df)
    pc = np.concatenate([[c[0]], c[:-1]])
    trh, trl = np.maximum(h, pc), np.minimum(l, pc)
    rng = trh - trl
    with np.errstate(invalid="ignore", divide="ignore"):
        ad = np.where(rng > 0, ((c - trl) - (trh - c)) / rng * v, 0.0)
        return np.nan_to_num(rma(ad, n) / rma(v, n))


def rmi(df, n=14, m=5):
    c = cols(df)[3]
    pm = np.concatenate([np.full(m, c[0]), c[:-m]])
    d = c - pm
    up, dn = rma(np.clip(d, 0, None), n), rma(np.clip(-d, 0, None), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(up + dn > 0, 100 * up / (up + dn), 50.0)


def pmo(df, s1=35, s2=20, sig=10):
    """dict pmo, signal."""
    c = cols(df)[3]
    pc = np.concatenate([[c[0]], c[:-1]])
    roc = (c / pc - 1) * 100
    a = S(roc).ewm(alpha=2.0 / s1, adjust=False).mean().to_numpy() * 10
    p = S(a).ewm(alpha=2.0 / s2, adjust=False).mean().to_numpy()
    return {"pmo": p, "signal": ema(p, sig)}


def t3(df, n=5, vf=0.7, src=None):
    x = cols(df)[3] if src is None else np.asarray(src, float)
    e1 = ema(x, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    e4 = ema(e3, n)
    e5 = ema(e4, n)
    e6 = ema(e5, n)
    a = vf
    c1 = -a ** 3
    c2 = 3 * a * a + 3 * a ** 3
    c3 = -6 * a * a - 3 * a - 3 * a ** 3
    c4 = 1 + 3 * a + a ** 3 + 3 * a * a
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def _alma_arr(x, n=9, offset=0.85, sigma=6.0):
    m = offset * (n - 1)
    s = n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    w = w / w.sum()
    return S(x).rolling(n).apply(lambda z: float(np.dot(z, w)), raw=True).to_numpy()


def alma(df, n=9, offset=0.85, sigma=6.0):
    return _alma_arr(cols(df)[3], n, offset, sigma)


def ash(df, n=9, sm=3):
    """dict bulls, bears (smoothed), diff = bulls - bears."""
    c = cols(df)[3]
    d = np.diff(c, prepend=c[0])
    bu = 0.5 * (np.abs(d) + d)
    be = 0.5 * (np.abs(d) - d)
    b1, b2 = sma(sma(bu, n), sm), sma(sma(be, n), sm)
    return {"bulls": b1, "bears": b2, "diff": b1 - b2}


def trendilo(df, n=50, offset=0.85, sigma=6.0, bmult=1.0):
    """dict avpch (smoothed % change), rms (band), trend (+1 / -1 / 0)."""
    c = cols(df)[3]
    pc = np.concatenate([[c[0]], c[:-1]])
    pch = (c - pc) / c * 100
    av = _alma_arr(pch, n, offset, sigma)
    rms = bmult * np.sqrt(S(np.nan_to_num(av) ** 2).rolling(n).mean().to_numpy())
    tr = np.where(av > rms, 1, np.where(av < -rms, -1, 0))
    return {"avpch": av, "rms": rms, "trend": tr}


def kairi(df, n=25):
    c = cols(df)[3]
    m = sma(c, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (c - m) / m * 100


def fractal_bands(df, k=2):
    """dict upper (last confirmed fractal high), lower (last confirmed fractal low)."""
    o, h, l, c, v = cols(df)
    up, lo = last_pivots(h, l, k)
    return {"upper": up, "lower": lo}
