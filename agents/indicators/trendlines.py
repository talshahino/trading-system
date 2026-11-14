"""Automatic trend lines and chart-pattern detector (OUR BUILD; ideas: Pattern Detector [BigBeluga] - channels /
triangles and their breakouts, Dan-Chi downtrend-line touch, TheTrader's Villa 3-touch liquidity trend line, Market
Structure 'trend-line retest'; tasks tal1411set4, tal1411set6, tal1411ind, task 16).
Lines join the last two confirmed swing highs (falling = resistance line) and the last two confirmed swing lows
(rising = support line); pivots have k bars on each side and are known k bars late, the line's value at bar i is the
straight-line extension to i (only pivots known by i are used - no repaint). touches = pivots (of the last 6) within
tol x ATR of the line's extension - the two that define it plus earlier ones on the same line. Patterns from the two lines' slopes: symmetric triangle (falling highs, rising
lows), ascending (flat highs, rising lows), descending (falling highs, flat lows), channel up / down (parallel),
wedge (both the same direction, converging)."""
import numpy as np

from agents.indicators._core import atr, cols, pivots

PATTERNS = {0: "none", 1: "sym triangle", 2: "asc triangle", 3: "desc triangle", 4: "channel up", 5: "channel down",
            6: "rising wedge", 7: "falling wedge"}


def trendlines(df, k=5, tol=0.3, flat=0.05):
    """dict: res_line, sup_line (value at bar i), res_slope, sup_slope (per bar, in ATR units), res_touches,
    sup_touches, pattern (code, see PATTERNS), touch_res (high reaches the falling line, close under it),
    touch_sup, break_res (close crosses above the resistance line), break_sup, height (distance of the lines at the
    pattern's start)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    PH, PL = [], []
    keys = ["res_line", "sup_line", "res_slope", "sup_slope", "res_touches", "sup_touches", "pattern", "height"]
    out = {x: np.full(n, np.nan) for x in keys}
    for x in ["touch_res", "touch_sup", "break_res", "break_sup"]:
        out[x] = np.zeros(n, bool)
    prev_r = prev_s = np.nan
    for i in range(n):
        if phc[i]:
            PH.append((phi[i], phv[i]))
        if plc[i]:
            PL.append((pli[i], plv[i]))
        ai = a[i] if np.isfinite(a[i]) and a[i] > 0 else np.nan
        r = s = np.nan
        if len(PH) >= 2:
            (x1, y1), (x2, y2) = PH[-2], PH[-1]
            m = (y2 - y1) / (x2 - x1)
            r = y2 + m * (i - x2)
            out["res_slope"][i] = m / ai if ai == ai else np.nan
            out["res_touches"][i] = sum(1 for x, y in PH[-6:] if abs(y - (y1 + m * (x - x1))) <= tol * ai)
        if len(PL) >= 2:
            (x1, y1), (x2, y2) = PL[-2], PL[-1]
            m = (y2 - y1) / (x2 - x1)
            s = y2 + m * (i - x2)
            out["sup_slope"][i] = m / ai if ai == ai else np.nan
            out["sup_touches"][i] = sum(1 for x, y in PL[-6:] if abs(y - (y1 + m * (x - x1))) <= tol * ai)
        out["res_line"][i], out["sup_line"][i] = r, s
        if np.isfinite(r):
            out["touch_res"][i] = h[i] >= r - 0.1 * (ai if ai == ai else 0) and c[i] < r
            out["break_res"][i] = c[i] > r and (i == 0 or not (c[i - 1] > prev_r))
        if np.isfinite(s):
            out["touch_sup"][i] = l[i] <= s + 0.1 * (ai if ai == ai else 0) and c[i] > s
            out["break_sup"][i] = c[i] < s and (i == 0 or not (c[i - 1] < prev_s))
        prev_r, prev_s = r, s
        rs, ss = out["res_slope"][i], out["sup_slope"][i]
        if np.isfinite(rs) and np.isfinite(ss) and np.isfinite(r) and np.isfinite(s) and r > s:
            start = min(PH[-2][0], PL[-2][0])
            hr = (PH[-2][1] + (PH[-1][1] - PH[-2][1]) / (PH[-1][0] - PH[-2][0]) * (start - PH[-2][0]))
            hs = (PL[-2][1] + (PL[-1][1] - PL[-2][1]) / (PL[-1][0] - PL[-2][0]) * (start - PL[-2][0]))
            out["height"][i] = hr - hs
            conv = (r - s) < (hr - hs)
            if rs < -flat and ss > flat:
                p = 1
            elif abs(rs) <= flat and ss > flat:
                p = 2
            elif rs < -flat and abs(ss) <= flat:
                p = 3
            elif rs > flat and ss > flat:
                p = 6 if conv and ss > rs else 4
            elif rs < -flat and ss < -flat:
                p = 7 if conv and rs < ss else 5
            else:
                p = 0
            out["pattern"][i] = p
        else:
            out["pattern"][i] = 0
    return out
