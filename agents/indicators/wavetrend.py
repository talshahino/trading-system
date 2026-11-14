"""WaveTrend Oscillator [LazyBear] - open-source Pine: https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
Port: ap = hlc3; esa = EMA(ap, 10); d = EMA(|ap - esa|, 10); ci = (ap - esa) / (0.015 d); wt1 = EMA(ci, 21);
wt2 = SMA(wt1, 4); levels OB 60 / 53, OS -60 / -53. The user's 4 uses (task tal1411ind 09:30): overbought / oversold,
crosses (wt1 over wt2 under OS2 = buy, under it over OB2 = sell), divergence and trend (wt1 > 0). Divergence (ours):
price makes a lower confirmed swing low (pivots of 5 bars, known 5 bars later) while wt1 at that low is higher than at
the previous one = bullish; mirror bearish. No repaint."""
import numpy as np

from agents.indicators._core import cols, cross_down, cross_up, ema, pivots, sma


def wavetrend(df, n1=10, n2=21, ob=60.0, ob2=53.0, os_=-60.0, os2=-53.0, k=5):
    """dict: wt1, wt2, buy (cross up under OS2), sell (cross down over OB2), bull_div, bear_div, trend (+1/-1)."""
    o, h, l, c, v = cols(df)
    ap = (h + l + c) / 3
    esa = ema(ap, n1)
    d = ema(np.abs(ap - esa), n1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.nan_to_num((ap - esa) / (0.015 * d))
    w1 = ema(ci, n2)
    w2 = sma(w1, 4)
    buy = cross_up(w1, w2) & (w1 < os2)
    sell = cross_down(w1, w2) & (w1 > ob2)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    n = len(c)
    bd = np.zeros(n, bool)
    sd = np.zeros(n, bool)
    last_l = last_h = None
    for t in range(n):
        if plc[t]:
            j = pli[t]
            if last_l is not None and l[j] < l[last_l] and w1[j] > w1[last_l]:
                bd[t] = True
            last_l = j
        if phc[t]:
            j = phi[t]
            if last_h is not None and h[j] > h[last_h] and w1[j] < w1[last_h]:
                sd[t] = True
            last_h = j
    return {"wt1": w1, "wt2": w2, "buy": buy, "sell": sell, "bull_div": bd, "bear_div": sd,
            "trend": np.sign(w1), "ob": w1 > ob, "os": w1 < os_}
