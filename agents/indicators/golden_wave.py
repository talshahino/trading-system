"""Fibonacci Golden Wave | Flux Charts - open-source Pine (MPL 2.0):
https://www.tradingview.com/script/SWe51Ddl-Fibonacci-Golden-Wave-Flux-Charts (code as quoted in
https://www.prorealcode.com/topic/conversion-from-trading-view-to-prt-fibonacci-golden-wave-indicator-2/).
Port: top = highest(high, 20), bottom = lowest(low, 20) (current bar included), golden zone between
top - (top-bottom) x 0.5 and top - (top-bottom) x 0.618. The original draws no signals; ours for the bots: a close
back above the zone after a low inside it (long) when the 20-bar range is up-trending (top made after bottom),
mirror for short. No repaint (trailing window)."""
import numpy as np

from agents.indicators._core import S, cols


def golden_wave(df, rng=20, f1=0.5, f2=0.618):
    """dict: top, bottom, z_hi, z_lo, up_leg (the high came after the low), touch_long / touch_short."""
    o, h, l, c, v = cols(df)
    hs, ls = S(h).rolling(rng, min_periods=1), S(l).rolling(rng, min_periods=1)
    top, bot = hs.max().to_numpy(), ls.min().to_numpy()
    ah = S(h).rolling(rng, min_periods=1).apply(np.argmax, raw=True).to_numpy()
    al = S(l).rolling(rng, min_periods=1).apply(np.argmin, raw=True).to_numpy()
    d = top - bot
    z1, z2 = top - d * f1, top - d * f2
    zh, zl = np.maximum(z1, z2), np.minimum(z1, z2)
    up = ah > al
    # mirror zone for a down leg: measured from the bottom
    zh_s, zl_s = bot + d * f2, bot + d * f1
    tl = up & (l <= zh) & (l >= zl) & (c > zh)
    ts = ~up & (h >= zl_s) & (h <= zh_s) & (c < zl_s)
    return {"top": top, "bottom": bot, "z_hi": zh, "z_lo": zl, "up_leg": up, "touch_long": tl, "touch_short": ts}
