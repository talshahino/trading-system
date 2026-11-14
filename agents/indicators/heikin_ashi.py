"""Heikin Ashi candles with the swing-trading states of the user's video (task tal1411set7 (6), TikTok 4H / daily:
'enter when the candle changes colour and the second candle in a row has no opposite wick; exit on a candle of the
opposite colour'). Standard formula (public domain; e.g. https://www.investopedia.com/trading/heikin-ashi-better-candlestick/):
HA close = (o+h+l+c)/4, HA open = (prev HA open + prev HA close)/2 (seed (o+c)/2), HA high = max(h, HA open, HA close),
HA low = min(l, HA open, HA close). 'No opposite wick' = HA low == min(HA open, HA close) for a green candle (no lower
wick), HA high == max(...) for a red one (tolerance tol x the bar's range, ours). No repaint."""
import numpy as np

from agents.indicators._core import cols


def heikin_ashi(df, tol=0.0):
    """dict: ha_o, ha_h, ha_l, ha_c, green, red, strong_green (green, no lower wick), strong_red (red, no upper
    wick), flip_up (first green after red), flip_dn, run (signed count of same-colour candles: +3 = third green),
    confirm_up (2nd green in a row with no lower wick right after a flip), confirm_dn."""
    o, h, l, c, v = cols(df)
    n = len(c)
    hc = (o + h + l + c) / 4.0
    ho = np.empty(n)
    if n:
        ho[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
    hh = np.maximum(h, np.maximum(ho, hc))
    hl = np.minimum(l, np.minimum(ho, hc))
    rng = np.maximum(hh - hl, 1e-12)
    green, red = hc > ho, hc < ho
    sg = green & (np.minimum(ho, hc) - hl <= tol * rng)
    sr = red & (hh - np.maximum(ho, hc) <= tol * rng)
    run = np.zeros(n)
    for i in range(n):
        d = 1 if green[i] else -1 if red[i] else 0
        if i and d != 0 and np.sign(run[i - 1]) == d:
            run[i] = run[i - 1] + d
        else:
            run[i] = d
    pg = np.r_[False, green[:-1]]
    pr = np.r_[False, red[:-1]]
    return {"ha_o": ho, "ha_h": hh, "ha_l": hl, "ha_c": hc, "green": green, "red": red, "strong_green": sg,
            "strong_red": sr, "flip_up": green & pr, "flip_dn": red & pg, "run": run,
            "confirm_up": (run == 2) & sg, "confirm_dn": (run == -2) & sr}
