"""Supply & Demand zones (+ 'Demand Supply Zone Matrix' and Flux Charts 'Supply & Demand + Breakout') - OUR BUILD
from the standard base-and-departure definition (Sam Seiden style; Flux / Matrix code closed or not read).
A zone = the base candle(s) right before a departure candle whose body >= dep x ATR(14); demand (rally from the
base): zone = base low .. base body top; supply mirror. The base must have a small body (<= 0.5 x ATR). A zone is
known at the departure bar's close (no repaint), stays live until a close through its far edge (= breakout: a broken
demand zone signals a bearish breakout, a broken supply zone a bullish one), first_touch = the first return into a
fresh zone with a close back out of it. Tasks tal1411ind / tal1411set6 (3)."""
import numpy as np

from agents.indicators._core import atr, cols


def supply_demand(df, dep=1.5, base_body=0.5, expire=300):
    """dict: dem_top, dem_bot, sup_top, sup_bot (latest live zones), first_touch_long, first_touch_short,
    break_up (supply broken), break_dn (demand broken), zone_count (live zones, matrix view)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    body = np.abs(c - o)
    keys = ["dem_top", "dem_bot", "sup_top", "sup_bot", "zone_count"]
    out = {x: np.full(n, np.nan) for x in keys}
    for x in ["first_touch_long", "first_touch_short", "break_up", "break_dn"]:
        out[x] = np.zeros(n, bool)
    dz, sz = [], []  # [top, bot, born, touched]
    for i in range(1, n):
        ai = a[i] if np.isfinite(a[i]) else 0.0
        for z in dz:
            if not z[3] and i > z[2] and l[i] <= z[0] and c[i] > z[0]:
                out["first_touch_long"][i] = True
                z[3] = True
            elif i > z[2] and l[i] <= z[0]:
                z[3] = True
        for z in sz:
            if not z[3] and i > z[2] and h[i] >= z[1] and c[i] < z[1]:
                out["first_touch_short"][i] = True
                z[3] = True
            elif i > z[2] and h[i] >= z[1]:
                z[3] = True
        if any(c[i] < z[1] for z in dz):
            out["break_dn"][i] = True
        if any(c[i] > z[0] for z in sz):
            out["break_up"][i] = True
        dz = [z for z in dz if c[i] >= z[1] and i - z[2] <= expire]
        sz = [z for z in sz if c[i] <= z[0] and i - z[2] <= expire]
        j = i - 1
        if ai > 0 and body[i] >= dep * ai and body[j] <= base_body * ai:
            if c[i] > o[i]:
                dz.append([max(o[j], c[j]), l[j], i, False])
            elif c[i] < o[i]:
                sz.append([h[j], min(o[j], c[j]), i, False])
        dz, sz = dz[-10:], sz[-10:]
        if dz:
            out["dem_top"][i], out["dem_bot"][i] = dz[-1][0], dz[-1][1]
        if sz:
            out["sup_top"][i], out["sup_bot"][i] = sz[-1][0], sz[-1][1]
        out["zone_count"][i] = len(dz) + len(sz)
    return out
