"""Inversion Fair Value Gaps (iFVG) - LuxAlgo, open-source Pine (CC BY-NC-SA 4.0):
https://www.tradingview.com/script/8FHucjY6-Inversion-Fair-Value-Gaps-IFVG-LuxAlgo/ (task tal1411set7 (4); also the IFVG
step of TT03 and of the user's SMT / London-NY reversal videos, tal1411set7 (3) and (11)).
The Pine code was not readable from here; this is OUR BUILD of the published rules: only gaps wider than ATR x
multiplier (min_atr), inversion when price CLOSES beyond the gap against its bias, inverted bullish gap = resistance,
inverted bearish gap = support, the zone is deleted once mitigated, retest signal by wick (LuxAlgo's 'Signal
Preference' = Wicks; retest_mode="close" uses the close inside). Details: a bullish FVG = low[i] > high[i-2], bearish = high[i] < low[i-2] (optional ATR size filter);
a gap is 'inverted' when a candle BODY closes beyond its far side (LuxAlgo's 'close' signal source: the close
goes under a bullish gap's bottom / over a bearish gap's top). The inverted gap then acts the opposite way:
a bearish FVG closed through upward = bullish iFVG (support), a bullish FVG closed through downward = bearish iFVG
(resistance). Signal = price comes back to the inverted zone and closes away from it (LuxAlgo's 'retest' bubble).
The iFVG dies when a close goes back through it (mitigated) or after max_age bars. Ours: up to `keep` live gaps
per side, gaps older than `max_gap_age` bars before the inversion are dropped. No repaint (closed bars only)."""
import numpy as np

from agents.indicators._core import atr, cols


def ifvg(df, min_atr=0.25, keep=5, max_gap_age=100, max_age=60, retest_mode="wick"):
    """dict: inv_bull / inv_bear (the bar a gap inverts), bull_top / bull_bot (the latest live bullish iFVG - a
    bearish gap closed through upward), bear_top / bear_bot, retest_long (low into the live bullish iFVG, close
    back over its top, bar after the inversion), retest_short (mirror)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    out = {x: np.full(n, np.nan) for x in ["bull_top", "bull_bot", "bear_top", "bear_bot"]}
    for x in ["inv_bull", "inv_bear", "retest_long", "retest_short"]:
        out[x] = np.zeros(n, bool)
    g_bull, g_bear = [], []          # live (unfilled, not inverted) gaps: (top, bot, born)
    ib = is_ = None                  # live inverted zones: (top, bot, inverted_at)
    for i in range(n):
        # retests of the live inverted zones (from the bar after the inversion)
        if ib is not None:
            if c[i] < ib[1] or i - ib[2] > max_age:
                ib = None
            elif i > ib[2] and (l[i] if retest_mode == "wick" else min(o[i], c[i])) <= ib[0] and c[i] > ib[0]:
                out["retest_long"][i] = True
        if is_ is not None:
            if c[i] > is_[0] or i - is_[2] > max_age:
                is_ = None
            elif i > is_[2] and (h[i] if retest_mode == "wick" else max(o[i], c[i])) >= is_[1] and c[i] < is_[1]:
                out["retest_short"][i] = True
        # inversions of the gaps formed before this bar
        keep_b = []
        for g in g_bull:
            if i - g[2] > max_gap_age:
                continue
            if c[i] < g[1]:                       # body close under a bullish gap -> bearish iFVG
                is_ = (g[0], g[1], i)
                out["inv_bear"][i] = True
            else:
                keep_b.append(g)
        g_bull = keep_b[-keep:]
        keep_s = []
        for g in g_bear:
            if i - g[2] > max_gap_age:
                continue
            if c[i] > g[0]:                       # body close over a bearish gap -> bullish iFVG
                ib = (g[0], g[1], i)
                out["inv_bull"][i] = True
            else:
                keep_s.append(g)
        g_bear = keep_s[-keep:]
        # new gaps on this bar
        if i >= 2:
            if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_atr * a[i]:
                g_bull.append((l[i], h[i - 2], i))
            if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_atr * a[i]:
                g_bear.append((l[i - 2], h[i], i))
        if ib is not None:
            out["bull_top"][i], out["bull_bot"][i] = ib[0], ib[1]
        if is_ is not None:
            out["bear_top"][i], out["bear_bot"][i] = is_[0], is_[1]
    return out
