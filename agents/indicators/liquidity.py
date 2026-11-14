"""Liquidity pools: equal highs / lows, sweeps and 'Liquidity Entry Zones' (OUR BUILD from the concepts in the
user's videos - Orderflow Sniper 'liquidity as a magnet', sphula gold sweep, Liquidity Entry Zones; tasks
tal1411of (3), tal1411set2 (2), tal1411ind; LuxAlgo SMC's EQH/EQL definition
https://www.tradingview.com/script/CnB3fSph-Smart-Money-Concepts-SMC-LuxAlgo/ : two confirmed swing highs within
0.1 x ATR = equal highs). A sweep = a bar trades beyond the level and closes back inside (the wick took the stops).
Entry zone after a bullish sweep = the sweep bar's low .. its body bottom. Pivots are known k bars late (no repaint)."""
import numpy as np

from agents.indicators._core import atr, cols, pivots


def liquidity(df, k=5, tol=0.1, max_age=300):
    """dict: eqh, eql (latest live equal-high / equal-low level, NaN when none), sweep_bull (low under the last
    swing low or EQL and close back above), sweep_bear, sweep_eq_bull / sweep_eq_bear (of an equal level only),
    zone_top / zone_bot (latest bullish entry zone), zone_top_s / zone_bot_s (bearish)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, _, _ = pivots(h, l, k)
    highs, lows = [], []  # (price, conf bar)
    eqh = np.full(n, np.nan)
    eql = np.full(n, np.nan)
    sb = np.zeros(n, bool)
    ss = np.zeros(n, bool)
    seb = np.zeros(n, bool)
    ses = np.zeros(n, bool)
    zt, zb, zts, zbs = (np.full(n, np.nan) for _ in range(4))
    cur_eqh = cur_eql = np.nan
    zone = zone_s = None
    for t in range(n):
        tolp = tol * a[t] if np.isfinite(a[t]) else 0.0
        # sweeps against the levels known BEFORE this bar
        if highs:
            lvl = highs[-1][0]
            if h[t] > lvl and c[t] < lvl:
                ss[t] = True
        if lows:
            lvl = lows[-1][0]
            if l[t] < lvl and c[t] > lvl:
                sb[t] = True
        if np.isfinite(cur_eqh) and h[t] > cur_eqh:
            ses[t] = c[t] < cur_eqh
            cur_eqh = np.nan
        if np.isfinite(cur_eql) and l[t] < cur_eql:
            seb[t] = c[t] > cur_eql
            cur_eql = np.nan
        if sb[t] or seb[t]:
            zone = (min(o[t], c[t]), l[t], t)
        if ss[t] or ses[t]:
            zone_s = (h[t], max(o[t], c[t]), t)
        highs = [x for x in highs if h[t] <= x[0] and t - x[1] <= max_age]
        lows = [x for x in lows if l[t] >= x[0] and t - x[1] <= max_age]
        if phc[t]:
            p = phv[t]
            if any(abs(p - x[0]) <= tolp for x in highs):
                cur_eqh = max(p, max(x[0] for x in highs if abs(p - x[0]) <= tolp))
            highs.append((p, t))
        if plc[t]:
            p = plv[t]
            if any(abs(p - x[0]) <= tolp for x in lows):
                cur_eql = min(p, min(x[0] for x in lows if abs(p - x[0]) <= tolp))
            lows.append((p, t))
        eqh[t], eql[t] = cur_eqh, cur_eql
        if zone is not None and (c[t] < zone[1] or t - zone[2] > 50):
            zone = None
        if zone_s is not None and (c[t] > zone_s[0] or t - zone_s[2] > 50):
            zone_s = None
        if zone is not None:
            zt[t], zb[t] = zone[0], zone[1]
        if zone_s is not None:
            zts[t], zbs[t] = zone_s[0], zone_s[1]
    return {"eqh": eqh, "eql": eql, "sweep_bull": sb, "sweep_bear": ss, "sweep_eq_bull": seb,
            "sweep_eq_bear": ses, "zone_top": zt, "zone_bot": zb, "zone_top_s": zts, "zone_bot_s": zbs}
