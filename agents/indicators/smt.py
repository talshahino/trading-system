"""SMT Divergence (ICT; simple_forex video, task tal1411ind 12:52: 'one makes a new high and the other does not', e.g.
US100 vs US500; open-source logic as in ClayeWeight 'SMT Divergence' https://it.tradingview.com/script/PH9fxHqo-SMT-Divergence
and TJR 'SMT Divergences' https://www.tradingview.com/script/7wG1zmCe). Pivots of the instrument (k bars each side,
known k bars later - no repaint); the partner's value at each pivot = its high (low) over the same 2k+1-bar window.
Bearish SMT at consecutive swing highs: one makes a higher high and the other an equal or lower high; bullish mirror at
swing lows. The partner's bars must be aligned to the instrument's index (columns h, l)."""
import numpy as np

from agents.indicators._core import S, cols, pivots


def smt_divergence(df, partner, k=3):
    """dict: bull, bear (on the confirmation bar), piv_hi, piv_lo (the instrument's pivot price of that signal)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    ph_c, pl_c, ph_v, pl_v, ph_i, pl_i = pivots(h, l, k)
    w = 2 * k + 1
    bh = S(partner["h"].to_numpy(float)).rolling(w).max().to_numpy()   # window ending at pivot + k
    bl = S(partner["l"].to_numpy(float)).rolling(w).min().to_numpy()
    out = {"bull": np.zeros(n, bool), "bear": np.zeros(n, bool), "piv_hi": np.full(n, np.nan), "piv_lo": np.full(n, np.nan)}
    prev_h = prev_l = None
    for i in range(n):
        if ph_c[i] and np.isfinite(bh[i]):
            cur = (ph_v[i], bh[i])
            if prev_h is not None:
                a_up, b_up = cur[0] > prev_h[0], cur[1] > prev_h[1]
                if a_up != b_up:
                    out["bear"][i], out["piv_hi"][i] = True, max(cur[0], h[i - k:i + 1].max())
            prev_h = cur
        if pl_c[i] and np.isfinite(bl[i]):
            cur = (pl_v[i], bl[i])
            if prev_l is not None:
                a_dn, b_dn = cur[0] < prev_l[0], cur[1] < prev_l[1]
                if a_dn != b_dn:
                    out["bull"][i], out["piv_lo"][i] = True, min(cur[0], l[i - k:i + 1].min())
            prev_l = cur
    both = out["bull"] & out["bear"]
    out["bull"] &= ~both
    out["bear"] &= ~both
    return out
