"""Impulse Zones | Flux Charts - open-source Pine: https://www.tradingview.com/script/Iv9MOWJ1-Impulse-Zones-Flux-Charts/
As described on the page: an impulse bar has (high - low) > ATR x size sensitivity (ours: the
true range, so a gap bar counts) AND volume > average volume x
volume sensitivity; the zone = that bar's high-low, extended right; retest = price comes back into the zone and
moves away; break = close beyond the far side (invalidation 'Close'). Defaults are not on the page - ours: ATR 14,
size 1.5, volume SMA 20 x 1.5, zones expire after 200 bars. With tick volume (CFD) the volume test is on tick volume;
a feed without real volume (constant) passes the volume test. The user (task tal1411ind 09:30): close to his gap
method - test as a gap filter. No repaint (zone known at the impulse bar's close)."""
import numpy as np

from agents.indicators._core import atr, cols, sma, stdev, tr


def impulse_zones(df, atr_len=14, size=1.5, vol_len=20, vol_mult=1.5, expire=200):
    """dict: impulse_up / impulse_dn (impulse bars), bull_top/bull_bot (latest live bullish zone), bear_top/bear_bot,
    retest_long (low enters the bullish zone, close above it), retest_short (mirror)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, atr_len)
    vm = sma(v, vol_len)
    flat = stdev(v, vol_len) == 0
    big = tr(h, l, c) > a * size  # true range (ours: a gap bar counts)
    vok = (v > vm * vol_mult) | flat
    iu = big & vok & (c > o)
    idn = big & vok & (c < o)
    bt, bb, st, sb = (np.full(n, np.nan) for _ in range(4))
    rl = np.zeros(n, bool)
    rs = np.zeros(n, bool)
    zb = zs = None
    for i in range(n):
        if zb is not None:
            if c[i] < zb[1] or i - zb[2] > expire:
                zb = None
            elif l[i] <= zb[0] and c[i] > zb[0] and i > zb[2]:
                rl[i] = True
        if zs is not None:
            if c[i] > zs[0] or i - zs[2] > expire:
                zs = None
            elif h[i] >= zs[1] and c[i] < zs[1] and i > zs[2]:
                rs[i] = True
        if iu[i]:
            zb = (h[i], l[i], i)
        if idn[i]:
            zs = (h[i], l[i], i)
        if zb is not None:
            bt[i], bb[i] = zb[0], zb[1]
        if zs is not None:
            st[i], sb[i] = zs[0], zs[1]
    return {"impulse_up": iu, "impulse_dn": idn, "bull_top": bt, "bull_bot": bb, "bear_top": st, "bear_bot": sb,
            "retest_long": rl, "retest_short": rs}
