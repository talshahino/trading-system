"""Smart Money Concepts [LuxAlgo] - open-source Pine (CC BY-NC-SA 4.0):
https://www.tradingview.com/script/CnB3fSph-Smart-Money-Concepts-SMC-LuxAlgo/ (the board in the user's thaa.fx gold
video, task tal1411ind 10:34; also 'Order Blocks' and 'Breaker Blocks' of Flux Charts).
Port of the core logic: swing pivots (structure length 50 in the original 'swing' mode, 5 for 'internal' - ours
`k`, confirmed k bars late); a CLOSE above the last unbroken swing high = BOS if the trend was already up, CHoCH if it
was down (mirror for lows); bullish order block on a bullish break = the bar with the lowest low between the broken
swing high and the break (its high/low), mitigated when a close goes under its low (the original's 'Close'
mitigation); FVG = low[i] > high[i-2] (bullish) / high[i] < low[i-2] (bearish), filled when price trades through;
premium / discount = position of the close inside the last swing high - swing low range (0 = low, 1 = high; > 0.5
premium, < 0.5 discount; OTE = 0.618 - 0.79 retracement of the leg). Breaker block (ours, Flux's definition):
a bullish OB that gets mitigated turns into a BEARISH breaker (resistance), and vice versa; retest = price comes
back to it and closes away. No repaint (closed bars, pivots shifted to their confirmation)."""
import numpy as np

from agents.indicators._core import cols, pivots


def smc(df, k=5):
    """dict: trend (+1/-1), bos_up, bos_dn, choch_up, choch_dn, ob_bull_top/bot, ob_bear_top/bot (latest live),
    ob_retest_long (low into the live bull OB, close above its top) / ob_retest_short, fvg_bull_top/bot,
    fvg_bear_top/bot (latest unfilled), pd_pos (0..1 inside the swing range), swing_hi, swing_lo,
    brk_bull_top/bot (bullish breaker = mitigated bearish OB), brk_bear_top/bot, brk_retest_long / brk_retest_short."""
    o, h, l, c, v = cols(df)
    n = len(c)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    keys = ["ob_bull_top", "ob_bull_bot", "ob_bear_top", "ob_bear_bot", "fvg_bull_top", "fvg_bull_bot",
            "fvg_bear_top", "fvg_bear_bot", "pd_pos", "swing_hi", "swing_lo", "brk_bull_top", "brk_bull_bot",
            "brk_bear_top", "brk_bear_bot"]
    out = {x: np.full(n, np.nan) for x in keys}
    for x in ["bos_up", "bos_dn", "choch_up", "choch_dn", "ob_retest_long", "ob_retest_short", "brk_retest_long",
              "brk_retest_short"]:
        out[x] = np.zeros(n, bool)
    trend = np.zeros(n)
    t = 0
    sh = sl = None  # (price, index, crossed)
    last_hi = last_lo = np.nan
    obs_b, obs_s, brk_b, brk_s = [], [], [], []  # (top, bot)
    fvb = fvs = None
    for i in range(n):
        if phc[i]:
            sh = [phv[i], phi[i], False]
            last_hi = phv[i]
        if plc[i]:
            sl = [plv[i], pli[i], False]
            last_lo = plv[i]
        if sh is not None and not sh[2] and c[i] > sh[0]:
            (out["bos_up"] if t == 1 else out["choch_up"])[i] = True
            sh[2] = True
            t = 1
            j0 = sh[1]
            seg = l[j0:i + 1]
            j = j0 + int(np.argmin(seg))
            obs_b.append((h[j], l[j]))
        if sl is not None and not sl[2] and c[i] < sl[0]:
            (out["bos_dn"] if t == -1 else out["choch_dn"])[i] = True
            sl[2] = True
            t = -1
            j0 = sl[1]
            seg = h[j0:i + 1]
            j = j0 + int(np.argmax(seg))
            obs_s.append((h[j], l[j]))
        # mitigation -> breakers
        keep = []
        for ob in obs_b:
            if c[i] < ob[1]:
                brk_s.append(ob)
            else:
                keep.append(ob)
        obs_b = keep[-5:]
        keep = []
        for ob in obs_s:
            if c[i] > ob[0]:
                brk_b.append(ob)
            else:
                keep.append(ob)
        obs_s = keep[-5:]
        brk_b = [b for b in brk_b if c[i] >= b[1]][-5:]
        brk_s = [b for b in brk_s if c[i] <= b[0]][-5:]
        if obs_b:
            top, bot = obs_b[-1]
            out["ob_bull_top"][i], out["ob_bull_bot"][i] = top, bot
            out["ob_retest_long"][i] = l[i] <= top and c[i] > top
        if obs_s:
            top, bot = obs_s[-1]
            out["ob_bear_top"][i], out["ob_bear_bot"][i] = top, bot
            out["ob_retest_short"][i] = h[i] >= bot and c[i] < bot
        if brk_b:
            top, bot = brk_b[-1]
            out["brk_bull_top"][i], out["brk_bull_bot"][i] = top, bot
            out["brk_retest_long"][i] = l[i] <= top and c[i] > top
        if brk_s:
            top, bot = brk_s[-1]
            out["brk_bear_top"][i], out["brk_bear_bot"][i] = top, bot
            out["brk_retest_short"][i] = h[i] >= bot and c[i] < bot
        if i >= 2:
            if l[i] > h[i - 2]:
                fvb = (l[i], h[i - 2])
            if h[i] < l[i - 2]:
                fvs = (l[i - 2], h[i])
        if fvb is not None and l[i] < fvb[1]:
            fvb = None
        if fvs is not None and h[i] > fvs[0]:
            fvs = None
        if fvb is not None:
            out["fvg_bull_top"][i], out["fvg_bull_bot"][i] = fvb
        if fvs is not None:
            out["fvg_bear_top"][i], out["fvg_bear_bot"][i] = fvs
        out["swing_hi"][i], out["swing_lo"][i] = last_hi, last_lo
        if np.isfinite(last_hi) and np.isfinite(last_lo) and last_hi > last_lo:
            out["pd_pos"][i] = (c[i] - last_lo) / (last_hi - last_lo)
        trend[i] = t
    out["trend"] = trend
    return out


def premium_discount(df, k=10):
    """Premium / discount and OTE (ICT; Forex Bull Signals video, task tal1411set6 (9)): the dealing range = last
    confirmed swing high and low; pos 0..1; discount = pos < 0.5; long OTE = pos 0.21-0.382 (= a 61.8-79% retracement
    of an up leg, measured from the high), short OTE = pos 0.618-0.79. dict: pos, discount, premium, ote_long,
    ote_short, up_leg (the last swing low came before the last swing high)."""
    o, h, l, c, v = cols(df)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    n = len(c)
    hi = lo = np.nan
    hi_at = lo_at = -1
    pos = np.full(n, np.nan)
    up = np.zeros(n, bool)
    for i in range(n):
        if phc[i]:
            hi, hi_at = phv[i], phi[i]
        if plc[i]:
            lo, lo_at = plv[i], pli[i]
        if np.isfinite(hi) and np.isfinite(lo) and hi > lo:
            pos[i] = (c[i] - lo) / (hi - lo)
            up[i] = lo_at < hi_at
    with np.errstate(invalid="ignore"):
        return {"pos": pos, "discount": pos < 0.5, "premium": pos > 0.5, "ote_long": (pos >= 0.21) & (pos <= 0.382),
                "ote_short": (pos >= 0.618) & (pos <= 0.79), "up_leg": up}
