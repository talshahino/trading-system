"""Tom DeMark's TD Range Expansion Index and TD Pressure (scout 16.11.2026).

- td_rei: TD REI as published by Linn Software (https://linnsoft.com/techind/demark-range-expansion-index) and the
  fxcodebase port (https://docs.gehtsoftusa.com/fxcodebase-backup/backup/custom-indicators/t/tom-demark-range-expansion-index-td-rei/):
  per bar x = (H - H[2]) + (L - L[2]), counted only when the bar 'overlaps' (H >= L[5] or H >= L[6], and L <= H[5] or
  L <= H[6]) or the bar 2 ago overlaps the closes 7-8 ago (H[2] >= C[7] or C[8], and L[2] <= C[7] or C[8]); else 0.
  REI = 100 x sum(x, 5) / sum(|H - H[2]| + |L - L[2]|, 5), -100..100. DeMark: oversold < -40 (-45), overbought > +40;
  the 'duration' rule - a reading that stays beyond the level more than 6 bars is a strong trend, not a reversal.
- td_pressure: OUR OWN reconstruction (the DeMark formula is proprietary / closed): buying pressure = close - true low,
  selling pressure = true high - close, both x tick volume; TD Pressure = 100 x sum(buy, 5) / sum(buy + sell, 5)
  (0..100; > 75 strong demand, < 25 strong supply), described at https://www.luxalgo.com/library/concept/td-pressure/ .
"""
import numpy as np

from agents.indicators._core import S, cols


def _b(x, k):
    x = np.asarray(x, dtype=float)
    return np.concatenate([np.full(k, np.nan), x[:-k]])


def td_rei(df, n=5):
    """dict: rei (-100..100), below (bars in a row under -40), above (bars in a row over +40)."""
    o, h, l, c, v = cols(df)
    h2, l2 = _b(h, 2), _b(l, 2)
    with np.errstate(invalid="ignore"):
        c1 = ((h >= _b(l, 5)) | (h >= _b(l, 6))) & ((l <= _b(h, 5)) | (l <= _b(h, 6)))
        c2 = ((h2 >= _b(c, 7)) | (h2 >= _b(c, 8))) & ((l2 <= _b(c, 7)) | (l2 <= _b(c, 8)))
        x = np.where(c1 | c2, (h - h2) + (l - l2), 0.0)
    den = np.abs(h - h2) + np.abs(l - l2)
    num = S(np.nan_to_num(x)).rolling(n).sum().to_numpy()
    dd = S(den).rolling(n).sum().to_numpy()
    rei = np.where(dd > 0, 100 * num / np.where(dd > 0, dd, 1), 0.0)
    rei[: n + 8] = np.nan
    below = np.zeros(len(c))
    above = np.zeros(len(c))
    for i in range(1, len(c)):
        below[i] = below[i - 1] + 1 if rei[i] < -40 else 0
        above[i] = above[i - 1] + 1 if rei[i] > 40 else 0
    return {"rei": rei, "below": below, "above": above}


def td_pressure(df, n=5):
    """array 0..100 (NaN warm-up)."""
    o, h, l, c, v = cols(df)
    pc = np.concatenate([[c[0]], c[:-1]])
    tl, th = np.minimum(l, pc), np.maximum(h, pc)
    w = np.where(v > 0, v, 1.0)
    buy, sell = (c - tl) * w, (th - c) * w
    sb = S(buy).rolling(n).sum().to_numpy()
    st = S(buy + sell).rolling(n).sum().to_numpy()
    return np.where(st > 0, 100 * sb / np.where(st > 0, st, 1), 50.0)
