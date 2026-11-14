"""Session VWAP with standard-deviation bands and Anchored VWAP (OUR BUILD of the standard definitions:
TradingView 'VWAP' / 'Anchored VWAP' built-ins, https://www.tradingview.com/support/solutions/43000502018-volume-weighted-average-price-vwap/ ;
the user's tasks tal1411ind: 'VWAP with standard deviation bands per session', 'Anchored VWAP').
VWAP = cumsum(hlc3 x v) / cumsum(v) from the anchor; bands = VWAP +/- k x sqrt(cumsum(v x hlc3^2)/cumsum(v) - VWAP^2).
CFDs: tick volume (no real volume - noted); a feed without volume -> equal weights. Anchors: 'session' (day in tz),
'week', or any boolean mask (e.g. the bar of a news release or a confirmed swing pivot - use the confirmation bar,
not the pivot bar, to stay free of look-ahead). No repaint."""
import numpy as np
import pandas as pd

from agents.indicators._core import cols, last_pivots, pivots, session_day


def _anchored(tp, w, grp):
    pv = pd.Series(tp * w).groupby(grp).cumsum().to_numpy()
    vv = pd.Series(w).groupby(grp).cumsum().to_numpy()
    p2 = pd.Series(tp * tp * w).groupby(grp).cumsum().to_numpy()
    vw = pv / vv
    sd = np.sqrt(np.maximum(p2 / vv - vw * vw, 0))
    return vw, sd


def vwap_bands(df, anchor="session", tz="UTC", start_hour=0, mults=(1.0, 2.0, 3.0)):
    """dict: vwap, sd, upper1..3, lower1..3. anchor: 'session' | 'week' | boolean mask (new anchor on True)."""
    o, h, l, c, v = cols(df)
    tp = (h + l + c) / 3
    w = np.where(v > 0, v, 1.0)
    if isinstance(anchor, str) and anchor == "session":
        grp = session_day(df, tz, start_hour)
    elif isinstance(anchor, str) and anchor == "week":
        t = df.index.tz_convert(tz)
        iso = t.isocalendar()
        grp = (iso.year.to_numpy() * 100 + iso.week.to_numpy())
    else:
        grp = np.cumsum(np.asarray(anchor, dtype=bool))
    vw, sd = _anchored(tp, w, grp)
    out = {"vwap": vw, "sd": sd}
    for k, m in enumerate(mults, 1):
        out[f"upper{k}"] = vw + m * sd
        out[f"lower{k}"] = vw - m * sd
    return out


def anchored_vwap_swing(df, k=10):
    """Two anchored VWAPs: from the last confirmed swing high and from the last confirmed swing low (anchored on the
    PIVOT bar, but each line only exists from the confirmation bar k bars later on - recomputed from the pivot
    using bars that were all closed by then, so no look-ahead). dict: avwap_high, avwap_low."""
    o, h, l, c, v = cols(df)
    n = len(c)
    tp = (h + l + c) / 3
    w = np.where(v > 0, v, 1.0)
    cpv, cw = np.cumsum(tp * w), np.cumsum(w)
    phc, plc, _, _, phi, pli = pivots(h, l, k)
    ah = np.full(n, np.nan)
    al = np.full(n, np.nan)
    jh = jl = -1
    for t in range(n):
        if phc[t]:
            jh = phi[t]
        if plc[t]:
            jl = pli[t]
        if jh >= 0:
            base_p = cpv[jh - 1] if jh > 0 else 0.0
            base_w = cw[jh - 1] if jh > 0 else 0.0
            ah[t] = (cpv[t] - base_p) / (cw[t] - base_w)
        if jl >= 0:
            base_p = cpv[jl - 1] if jl > 0 else 0.0
            base_w = cw[jl - 1] if jl > 0 else 0.0
            al[t] = (cpv[t] - base_p) / (cw[t] - base_w)
    _ = last_pivots
    return {"avwap_high": ah, "avwap_low": al}
