"""Impulse legs for Fibonacci confluence (the user's task tal1411gapplus (2), (3), (6): TradersNotes 'FVG + Order Block +
Fibonacci', Mr Albash 'SMC + Fibonacci'). OUR BUILD: the latest completed impulse leg from confirmed pivots (k bars
each side; a pivot is known only k bars after it - no repaint). Up leg = a pivot low followed by a higher pivot high;
down leg = a pivot high followed by a lower pivot low. For the latest leg of each side the module gives its ends,
the golden zone 0.5-0.618 retracement, the 0.618-0.786 zone, the extensions 1.272 / 1.414 / 2.0 / 2.618 of the leg
(measured from the leg's start: ext_x = start + x * (end - start)), the order block (the last opposite-colour candle
at or up to `ob_back` bars before the leg's start) and whether a same-direction FVG formed inside the leg overlaps
the golden zone (fvg_top / fvg_bot of the overlapping one, NaN when none)."""
import numpy as np

from agents.indicators._core import cols, pivots


def fib_legs(df, k=3, ob_back=5):
    """dict of arrays for the up leg (prefix u_) and the down leg (prefix d_): lo, hi, gz_top, gz_bot (0.5-0.618),
    dz_top, dz_bot (0.618-0.786), ext1272, ext1414, ext2, ext2618, ob_top, ob_bot, fvg_top, fvg_bot, age (bars since
    the leg was confirmed)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    keys = ["lo", "hi", "gz_top", "gz_bot", "dz_top", "dz_bot", "ext1272", "ext1414", "ext2", "ext2618",
            "ob_top", "ob_bot", "fvg_top", "fvg_bot", "age"]
    out = {p + x: np.full(n, np.nan) for p in ("u_", "d_") for x in keys}
    last_l = last_h = None          # (price, index)
    up = dn = None                  # dict of the current leg
    for i in range(n):
        if plc[i]:
            if last_h is not None and pli[i] > last_h[1] and plv[i] < last_h[0]:
                dn = _leg(o, h, l, c, last_h, (plv[i], pli[i]), -1, ob_back, i)
            last_l = (plv[i], pli[i])
        if phc[i]:
            if last_l is not None and phi[i] > last_l[1] and phv[i] > last_l[0]:
                up = _leg(o, h, l, c, last_l, (phv[i], phi[i]), 1, ob_back, i)
            last_h = (phv[i], phi[i])
        for p, leg in (("u_", up), ("d_", dn)):
            if leg is None:
                continue
            for x in keys[:-1]:
                out[p + x][i] = leg[x]
            out[p + "age"][i] = i - leg["at"]
    return out


def _leg(o, h, l, c, start, end, d, ob_back, at):
    s, si = start
    e, ei = end
    rng = e - s                    # > 0 up leg, < 0 down leg
    lv = {"lo": min(s, e), "hi": max(s, e), "at": at}
    r = lambda f: e - f * rng      # noqa: E731  retracement level
    lv["gz_top"], lv["gz_bot"] = max(r(0.5), r(0.618)), min(r(0.5), r(0.618))
    lv["dz_top"], lv["dz_bot"] = max(r(0.618), r(0.786)), min(r(0.618), r(0.786))
    for name, f in (("ext1272", 1.272), ("ext1414", 1.414), ("ext2", 2.0), ("ext2618", 2.618)):
        lv[name] = s + f * rng
    lv["ob_top"] = lv["ob_bot"] = np.nan
    for j in range(si, max(-1, si - ob_back - 1), -1):
        if (d > 0 and c[j] < o[j]) or (d < 0 and c[j] > o[j]):
            lv["ob_top"], lv["ob_bot"] = h[j], l[j]
            break
    lv["fvg_top"] = lv["fvg_bot"] = np.nan
    for j in range(si + 2, ei + 1):
        if d > 0 and l[j] > h[j - 2]:
            t, b = l[j], h[j - 2]
        elif d < 0 and h[j] < l[j - 2]:
            t, b = l[j - 2], h[j]
        else:
            continue
        if t >= lv["gz_bot"] and b <= lv["gz_top"]:
            lv["fvg_top"], lv["fvg_bot"] = t, b
            break
    return lv
