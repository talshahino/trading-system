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


def _pivots(h, l, k=3):
    """Swing highs / lows: a bar higher (lower) than the k bars on each side; known only k bars later."""
    n = len(h)
    ph, pl = [], []
    for p in range(k, n - k):
        if h[p] == h[p - k:p + k + 1].max() and h[p] > h[p - k:p].max():
            ph.append(p)
        if l[p] == l[p - k:p + k + 1].min() and l[p] < l[p - k:p].min():
            pl.append(p)
    return ph, pl


def _falling_wedge(h, l, c, a, k=3, fresh=30):
    """Falling wedge breakout (long). Mirror the prices for a rising wedge."""
    n = len(c)
    sig = np.zeros(n, dtype=bool)
    stop, tgt = np.full(n, np.nan), np.full(n, np.nan)
    ph, pl = _pivots(h, l, k)
    known_h = {p + k: p for p in ph}          # bar at which a pivot becomes known -> the pivot bar
    known_l = {p + k: p for p in pl}
    H, Lw = [], []
    wedge, used = None, set()
    for t in range(n):
        new = False
        if t in known_h:
            H.append(known_h[t])
            new = True
        if t in known_l:
            Lw.append(known_l[t])
            new = True
        if new and len(H) >= 3 and len(Lw) >= 3:
            h3, l3 = H[-3:], Lw[-3:]
            wedge = None
            if (h[h3[0]] > h[h3[1]] > h[h3[2]] and l[l3[0]] > l[l3[1]] > l[l3[2]] and h3[0] < l3[-1]
                    and (h3[0], h3[2]) not in used):
                sr = (h[h3[2]] - h[h3[0]]) / (h3[2] - h3[0])      # resistance slope (per bar)
                ss = (l[l3[2]] - l[l3[0]]) / (l3[2] - l3[0])      # support slope
                res = lambda x, p=h3[0], s=sr: h[p] + s * (x - p)  # noqa: E731
                sup = lambda x, p=l3[0], s=ss: l[p] + s * (x - p)  # noqa: E731
                ok = sr < ss < 0 and abs(h[h3[1]] - res(h3[1])) < 0.75 * a[t] and abs(l[l3[1]] - sup(l3[1])) < 0.75 * a[t]
                if ok and res(t) - sup(t) < res(h3[0]) - sup(h3[0]):   # converging
                    wedge = (h3, l3, res, sup)
        if wedge is None or not np.isfinite(a[t]):
            continue
        h3, l3, res, sup = wedge
        if t - max(h3[-1], l3[-1]) > fresh or c[t] < sup(t) - 0.5 * a[t]:   # stale, or broken the wrong way
            wedge = None
            continue
        if c[t] > res(t):                          # breakout over the resistance line: buy at the next open
            sig[t] = True
            stop[t] = min(l[l3[-1]], l[max(0, t - 5):t + 1].min()) - 0.1 * a[t]   # under the last swing low
            tgt[t] = h[h3[0]]                      # the wedge's starting high
            used.add((h3[0], h3[2]))
            wedge = None
    return sig, stop, tgt


def s_falling_wedge(df):
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    a = atr(h, l, c, 14)
    sig, st, tg = _falling_wedge(h, l, c, a)
    z = np.zeros(len(c), dtype=bool)
    return sig, z, {"stop": ("level", st, st), "target": ("level", tg, tg)}


def s_rising_wedge(df):
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    a = atr(h, l, c, 14)
    sig, st, tg = _falling_wedge(-l, -h, -c, a)    # mirrored prices: a rising wedge becomes a falling one
    z = np.zeros(len(c), dtype=bool)
    return z, sig, {"stop": ("level", -st, -st), "target": ("level", -tg, -tg)}


# SM02 / SM03 (Trader Dhana reels, sent by the user 08.11.2026):
#   falling wedge: 3 lower swing highs and 3 lower swing lows, both lines falling, the resistance steeper
#   (converging); a close over the resistance line -> buy; stop under the last swing low; target = the
#   wedge's starting high. Rising wedge = the mirror -> sell short; stop over the last swing high; target =
#   the wedge's starting low.
SMC = {
    "SM02": ("Falling wedge breakout, target the wedge start (Trader Dhana)", s_falling_wedge,
             ["15Min", "1Hour", "4Hour", "1Day"]),
    "SM03": ("Rising wedge breakdown, short, target the wedge start (Trader Dhana)", s_rising_wedge,
             ["15Min", "1Hour", "4Hour", "1Day"]),
    "SM01": ("Market structure + liquidity sweep + FVG entry, measured-move target (TheTrader's Villa)",
             s_smc_fvg, ["15Min", "1Hour", "4Hour", "1Day"]),
}
