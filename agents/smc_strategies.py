"""Smart-money style setups from social media (research, 08.11.2026). Same contract as agents/rules_agent.py.

SM01 "Market structure + liquidity + FVG" (TheTrader's Villa, Facebook reel, sent by the user 08.11.2026;
     rules as the user confirmed them):
  1. impulse: price rises at least 3 x ATR(14) within 10 bars and leaves a fair value gap (a bar whose low is
     above the high of the bar two before it); the gap is the zone [that high, that low]
  2. channel: at least 10 bars follow in which price stays above the gap (a close under the gap cancels)
  3. breakout: a close under the lowest low of those channel bars (the liquidity under the channel is taken)
  4. entry: within 20 bars of the breakout price trades back into the gap -> buy (next bar's open)
  5. stop: the bottom of the gap - 0.25 ATR
  6. target: the size of the first leg (impulse low -> top of the channel), measured from the gap top
  Mirror for shorts. A setup with no breakout within 60 bars of the impulse is dropped.
"""
import numpy as np

from agents.rules_agent import atr


def _side(o, h, l, c, a, d):
    """Signals, stop and target for one direction (d=+1 long; shorts run on the mirrored prices)."""
    n = len(c)
    sig = np.zeros(n, dtype=bool)
    stop, tgt = np.full(n, np.nan), np.full(n, np.nan)
    state, j = 0, 0
    g_lo = g_hi = imp_low = top = brk_t = ch_low = None
    for t in range(12, n):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        if state == 0:
            if l[t] > h[t - 2]:
                lo10 = l[t - 9:t + 1].min()
                if c[t] - lo10 >= 3 * a[t]:
                    state, j = 1, t
                    g_lo, g_hi, imp_low, top, ch_low = h[t - 2], l[t], lo10, h[t], np.inf
            continue
        if state == 1:
            if c[t] < g_lo or t - j > 60:
                state = 0
                continue
            if t - j > 10 and c[t] < ch_low:          # the channel's low is broken (its bars: j+1 .. t-1)
                state, brk_t = 2, t
            else:
                top = max(top, h[t])
                ch_low = min(ch_low, l[t])
                continue
        if state == 2:
            if l[t] <= g_hi:                           # back in the gap: buy at the next open
                sig[t] = True
                stop[t] = g_lo - 0.25 * a[t]
                tgt[t] = g_hi + (top - imp_low)
                state = 0
            elif t - brk_t > 20:
                state = 0
    return sig, stop, tgt


def s_smc_fvg(df):
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in "ohlc")
    a = atr(h, l, c, 14)
    L, sl, tl = _side(o, h, l, c, a, 1)
    S, ss, ts = _side(-o, -l, -h, -c, a, -1)           # mirrored prices: highs become lows
    return L, S & ~L, {"stop": ("level", sl, -ss), "target": ("level", tl, -ts)}


SMC = {
    "SM01": ("Market structure + liquidity sweep + FVG entry, measured-move target (TheTrader's Villa)",
             s_smc_fvg, ["15Min", "1Hour", "4Hour", "1Day"]),
}
