"""John F. Ehlers' 2026 TASC filters (scout 16.11.2026) - ports of the published Traders' Tips listings, closed bars only:

- reversion_index: "The Reversion Index" (TASC 2026.01, https://www.traders.com/Documentation/FEEDbk_docs/2026/01/TradersTips.html,
  Pine https://www.tradingview.com/script/V35NeC45-TASC-2026-01-The-Reversion-Index/): Ratio = sum(close - close[1], 20) /
  sum(|close - close[1]|, 20) in -1..1; Smooth = SuperSmoother(Ratio, 8), Trigger = SuperSmoother(Ratio, 4). Signals (the
  Wealth-Lab / Python listings): Trigger crosses over Smooth = buy, under = sell. Port of the EasyLanguage listing.
- adaptive_supersmoother: "Adaptive SuperSmoother" (TASC 2026.09,
  https://www.tradingview.com/script/FnlMn99W-TASC-2026-09-Adaptive-SuperSmoother/): SS(close, 20); ROC = SS - SS[1];
  scaled = |ROC| / RMS(ROC, 81) capped at 2; factor = (1 - 0.5 scaled)^2; adaptive period = max(20 x factor, 2); a second
  SuperSmoother with that per-bar period. Direction: long while the adaptive filter is over the fixed one. OUR OWN
  reconstruction from the published steps (the Pine source was not read): the absolute value of the scaled ROC is ours.
- autotune_filter: "The AutoTune Filter" (TASC 2026.05, https://www.tradingview.com/script/GqKlw3yv-TASC-2026-05-The-AutoTune-Filter/,
  steps in https://lavs9.github.io/quantwave/guides/indicators/native/autotune_filter/): 2-pole high-pass(close, 20); the
  Pearson correlation of the last 20 HP values with the same window `lag` bars earlier for lag 1..window; dominant cycle
  DC = 2 x the lag with the LOWEST correlation (clamped 6..2 x window); band-pass(HP, DC, bandwidth 0.25). OUR OWN
  reconstruction of the lag range, the clamp and the band-pass form (Ehlers' standard band-pass of Cycle Analytics).
"""
import math

import numpy as np

from agents.indicators._core import S, cols
from agents.indicators.ehlers2 import highpass2, super_smoother


def _warm(a, n):
    a = np.asarray(a, dtype=float).copy()
    a[:n] = np.nan
    return a


def reversion_index(df, length=20, smooth=8, trigger=4):
    """dict: ratio (-1..1), smooth (SS 8), trigger (SS 4)."""
    o, h, l, c, v = cols(df)
    d = np.concatenate([[0.0], np.diff(c)])
    num = S(d).rolling(length).sum().to_numpy()
    den = S(np.abs(d)).rolling(length).sum().to_numpy()
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    ratio[:length] = 0.0
    sm = super_smoother(ratio, smooth)
    tg = super_smoother(ratio, trigger)
    w = length + 5
    return {"ratio": _warm(ratio, w), "smooth": _warm(sm, w), "trigger": _warm(tg, w)}


def _ss_var(x, period):
    """SuperSmoother whose period changes per bar (coefficients from the bar's own period)."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    f = np.zeros(len(x))
    for t in range(len(x)):
        if t < 2:
            f[t] = x[t]
            continue
        p = max(float(period[t]), 2.0)
        a1 = math.exp(-1.414 * math.pi / p)
        c2 = 2 * a1 * math.cos(1.414 * math.pi / p)
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        f[t] = c1 * (x[t] + x[t - 1]) / 2 + c2 * f[t - 1] + c3 * f[t - 2]
    return f


def adaptive_supersmoother(df, base=20, rms_len=81):
    """dict: ss (fixed SuperSmoother base), ass (adaptive SuperSmoother), period (adaptive period), diff (ass - ss)."""
    o, h, l, c, v = cols(df)
    ss = super_smoother(c, base)
    roc = np.concatenate([[0.0], np.diff(ss)])
    rms = np.sqrt(S(roc ** 2).rolling(rms_len, min_periods=10).mean().to_numpy())
    sc = np.minimum(np.where(rms > 0, np.abs(roc) / np.where(rms > 0, rms, 1), 0.0), 2.0)
    sc = np.nan_to_num(sc)
    per = np.maximum(base * (1 - 0.5 * sc) ** 2, 2.0)
    ass = _ss_var(c, per)
    w = 30
    return {"ss": _warm(ss, w), "ass": _warm(ass, w), "period": per, "diff": _warm(ass - ss, w)}


def autotune_filter(df, window=20, bandwidth=0.25):
    """dict: bp (band-pass at the tuned cycle), dc (dominant cycle in bars), roc (bp - bp[1])."""
    o, h, l, c, v = cols(df)
    n = len(c)
    hp = highpass2(c, window)
    hs = S(hp)
    corr = np.full((window, n), np.nan)
    for lag in range(1, window + 1):
        corr[lag - 1] = hs.rolling(window).corr(hs.shift(lag)).to_numpy()
    cc = np.where(np.isfinite(corr), corr, np.inf)
    best = np.argmin(cc, axis=0) + 1
    ok = np.isfinite(corr).any(axis=0)
    dc = np.clip(np.where(ok, 2.0 * best, 2.0 * window), 6.0, 2.0 * window)
    bp = np.zeros(n)
    for t in range(2, n):
        p = dc[t]
        l1 = math.cos(2 * math.pi / p)
        g1 = math.cos(bandwidth * 2 * math.pi / p)
        s1 = 1 / g1 - math.sqrt(max(1 / (g1 * g1) - 1, 0.0))
        bp[t] = 0.5 * (1 - s1) * (hp[t] - hp[t - 2]) + l1 * (1 + s1) * bp[t - 1] - s1 * bp[t - 2]
    w = 3 * window
    roc = np.concatenate([[np.nan], np.diff(bp)])
    return {"bp": _warm(bp, w), "dc": dc, "roc": _warm(roc, w)}
