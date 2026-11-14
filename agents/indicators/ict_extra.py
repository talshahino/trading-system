"""More ICT arrays (scout, 15.11.2026): Unicorn (breaker + FVG overlap), Rejection blocks, Balanced Price Range,
New Day / New Week Opening Gaps and the Power-of-Three (AMD) frame around the New York midnight open.
Sources (concept pages; the definitions are public, the code is OURS):
- Unicorn: https://www.luxalgo.com/library/concept/unicorn/ (sweep -> displacement that breaks structure and leaves an
  FVG -> the breaker; only the slice where breaker and FVG overlap is the zone; entry on the retrace into it, stop
  beyond the breaker's far side / the swept swing, target the opposing liquidity).
- Rejection block: https://www.luxalgo.com/library/concept/rejection-block/ (a swing extreme with a long wick; the
  zone runs from the body edge to the wick extreme; traded on the revisit, stop beyond the wick, invalid on a body
  close beyond the wick).
- BPR: https://www.tradingview.com/script/zPbYfcE8-ICT-Balance-Price-Range-UAlgo/ (not readable from here - OUR BUILD
  of the standard ICT definition: a bullish and a bearish FVG within a few bars whose price ranges overlap; the
  overlap is the zone, its bias = the later gap's direction).
- NDOG / NWOG: https://www.luxalgo.com/library/concept/opening-gap/ and /new-week-opening-gap/ (the span between the
  previous close and the reopen; CE = its 50%; the last 5 kept). On a 24-hour feed: NWOG = Friday's last close vs the
  week's first open; NDOG = the close of the bar before the daily break (17:00 New York) vs the first open after it.
- PO3 / AMD: https://www.luxalgo.com/library/concept/accumulation-manipulation-distribution/ (anchor = the New York
  midnight open; manipulation = a run beyond the open that takes liquidity, distribution = displacement back through
  the open).
No repaint: every value at bar i uses bars <= i only; pivots are shifted to their confirmation bar."""
import numpy as np
import pandas as pd

from agents.indicators._core import atr, cols, pivots


def rejection_blocks(df, k=3, wick_ratio=1.5, max_age=150):
    """dict: bull_top / bull_bot (latest live bullish rejection block: the wick below the body of a confirmed swing
    low whose lower wick >= wick_ratio x body; top = the body bottom, bot = the low), bear_top / bear_bot (mirror),
    retest_long (low into the live bullish block, close back over its top), retest_short."""
    o, h, l, c, v = cols(df)
    n = len(c)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    out = {x: np.full(n, np.nan) for x in ["bull_top", "bull_bot", "bear_top", "bear_bot"]}
    out["retest_long"], out["retest_short"] = np.zeros(n, bool), np.zeros(n, bool)
    bu = be = None
    for i in range(n):
        if plc[i]:
            j = int(pli[i])
            body = abs(c[j] - o[j])
            wick = min(o[j], c[j]) - l[j]
            if wick >= wick_ratio * max(body, 1e-12) and wick > 0:
                bu = (min(o[j], c[j]), l[j], i)
        if phc[i]:
            j = int(phi[i])
            body = abs(c[j] - o[j])
            wick = h[j] - max(o[j], c[j])
            if wick >= wick_ratio * max(body, 1e-12) and wick > 0:
                be = (h[j], max(o[j], c[j]), i)
        if bu is not None:
            if min(o[i], c[i]) < bu[1] or i - bu[2] > max_age:       # body accepted under the wick = invalid
                bu = None
            else:
                out["bull_top"][i], out["bull_bot"][i] = bu[0], bu[1]
                out["retest_long"][i] = i > bu[2] and l[i] <= bu[0] and c[i] > bu[0]
        if be is not None:
            if max(o[i], c[i]) > be[0] or i - be[2] > max_age:
                be = None
            else:
                out["bear_top"][i], out["bear_bot"][i] = be[0], be[1]
                out["retest_short"][i] = i > be[2] and h[i] >= be[1] and c[i] < be[1]
    return out


def bpr(df, window=5, max_age=60):
    """Balanced Price Range. dict: bull_top / bull_bot (live bullish BPR: a bearish FVG followed within `window` bars
    by a bullish FVG overlapping it), bear_top / bear_bot (bullish FVG then an overlapping bearish one),
    retest_long / retest_short (return into the zone, close back out in its direction)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    out = {x: np.full(n, np.nan) for x in ["bull_top", "bull_bot", "bear_top", "bear_bot"]}
    out["retest_long"], out["retest_short"] = np.zeros(n, bool), np.zeros(n, bool)
    gb, gs = [], []        # recent gaps (top, bot, i)
    zb = zs = None
    for i in range(n):
        if zb is not None:
            if c[i] < zb[1] or i - zb[2] > max_age:
                zb = None
            elif i > zb[2] and l[i] <= zb[0] and c[i] > zb[0]:
                out["retest_long"][i] = True
        if zs is not None:
            if c[i] > zs[0] or i - zs[2] > max_age:
                zs = None
            elif i > zs[2] and h[i] >= zs[1] and c[i] < zs[1]:
                out["retest_short"][i] = True
        if i >= 2:
            if l[i] > h[i - 2]:
                top, bot = l[i], h[i - 2]
                for g in gs:
                    if i - g[2] <= window and min(top, g[0]) > max(bot, g[1]):
                        zb = (min(top, g[0]), max(bot, g[1]), i)
                gb.append((top, bot, i))
            if h[i] < l[i - 2]:
                top, bot = l[i - 2], h[i]
                for g in gb:
                    if i - g[2] <= window and min(top, g[0]) > max(bot, g[1]):
                        zs = (min(top, g[0]), max(bot, g[1]), i)
                gs.append((top, bot, i))
            gb = [g for g in gb if i - g[2] <= window]
            gs = [g for g in gs if i - g[2] <= window]
        if zb is not None:
            out["bull_top"][i], out["bull_bot"][i] = zb[0], zb[1]
        if zs is not None:
            out["bear_top"][i], out["bear_bot"][i] = zs[0], zs[1]
    return out


def opening_gaps(df, tz="America/New_York", break_hour=17, min_gap_atr=0.05):
    """New Week / New Day Opening Gaps. dict: nwog_top / nwog_bot / nwog_ce (the latest NWOG: the week's first
    open vs the previous week's last close), ndog_top / ndog_bot / ndog_ce (the latest NDOG across the 17:00 New York
    daily break), gap bars (new_nwog / new_ndog). Gaps smaller than min_gap_atr x ATR are skipped."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    t = df.index.tz_convert(tz)
    sess = (t - pd.Timedelta(hours=break_hour)).normalize()          # trading day key (starts 17:00 NY)
    wk = (t - pd.Timedelta(hours=break_hour)).tz_localize(None).to_period("W-FRI")     # week ends Friday 17:00 NY
    sk = np.asarray(sess.asi8)
    wkk = np.asarray(wk.asi8) if hasattr(wk, "asi8") else np.asarray([p.ordinal for p in wk])
    out = {x: np.full(n, np.nan) for x in ["nwog_top", "nwog_bot", "nwog_ce", "ndog_top", "ndog_bot", "ndog_ce"]}
    out["new_nwog"], out["new_ndog"] = np.zeros(n, bool), np.zeros(n, bool)
    wg = dg = None
    for i in range(1, n):
        if wkk[i] != wkk[i - 1]:
            top, bot = max(c[i - 1], o[i]), min(c[i - 1], o[i])
            if top - bot >= min_gap_atr * (a[i - 1] if np.isfinite(a[i - 1]) else 0):
                wg = (top, bot)
                out["new_nwog"][i] = True
        elif sk[i] != sk[i - 1]:
            top, bot = max(c[i - 1], o[i]), min(c[i - 1], o[i])
            if top - bot >= min_gap_atr * (a[i - 1] if np.isfinite(a[i - 1]) else 0):
                dg = (top, bot)
                out["new_ndog"][i] = True
        if wg is not None:
            out["nwog_top"][i], out["nwog_bot"][i], out["nwog_ce"][i] = wg[0], wg[1], (wg[0] + wg[1]) / 2
        if dg is not None:
            out["ndog_top"][i], out["ndog_bot"][i], out["ndog_ce"][i] = dg[0], dg[1], (dg[0] + dg[1]) / 2
    return out


def po3_midnight(df, tz="America/New_York"):
    """Power of Three around the New York midnight open. dict: mo (today's 00:00 NY open, from the first bar at or
    after midnight), asia_hi / asia_lo (running 20:00-00:00 NY range of the previous evening - ICT's Asian range),
    day_lo / day_hi (running extremes since midnight), manip_dn (price has traded below the midnight open AND under
    the Asian low today), manip_up (mirror), dist_long (manip_dn, then a close back over the midnight open - the
    distribution leg starts), dist_short (mirror)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    t = df.index.tz_convert(tz)
    hr = t.hour + t.minute / 60.0
    day = np.asarray(t.normalize().asi8)
    eve = np.asarray((t + pd.Timedelta(hours=4)).normalize().asi8)   # 20:00 NY belongs to the next day's key
    out = {x: np.full(n, np.nan) for x in ["mo", "asia_hi", "asia_lo", "day_hi", "day_lo"]}
    for x in ["manip_dn", "manip_up", "dist_long", "dist_short"]:
        out[x] = np.zeros(n, bool)
    mo = ah = al = dh = dl = np.nan
    cur_day = None
    a_key, a_hi, a_lo = None, np.nan, np.nan
    md = mu = False
    for i in range(n):
        if hr[i] >= 20.0:                                   # evening session builds the next day's Asian range
            if a_key != eve[i]:
                a_key, a_hi, a_lo = eve[i], h[i], l[i]
            else:
                a_hi, a_lo = max(a_hi, h[i]), min(a_lo, l[i])
        if day[i] != cur_day:
            cur_day = day[i]
            mo = o[i]
            dh, dl = h[i], l[i]
            ah, al = (a_hi, a_lo) if a_key == day[i] else (np.nan, np.nan)
            md = mu = False
        else:
            dh, dl = max(dh, h[i]), min(dl, l[i])
        if hr[i] < 20.0:
            if np.isfinite(al) and dl < mo and dl < al:
                if md and c[i] > mo and i and c[i - 1] <= mo:
                    out["dist_long"][i] = True
                md = True
            if np.isfinite(ah) and dh > mo and dh > ah:
                if mu and c[i] < mo and i and c[i - 1] >= mo:
                    out["dist_short"][i] = True
                mu = True
            out["manip_dn"][i], out["manip_up"][i] = md, mu
        out["mo"][i], out["asia_hi"][i], out["asia_lo"][i], out["day_hi"][i], out["day_lo"][i] = mo, ah, al, dh, dl
    return out


def unicorn(df, k=3, max_age=40):
    """ICT Unicorn: a breaker block (from agents.indicators.smc - a mitigated order block that flipped) that overlaps a
    fair value gap formed in the same displacement. dict: bull_top / bull_bot (live bullish unicorn zone = overlap of
    the bullish breaker and a bullish FVG of the break), bear_top / bear_bot, retest_long / retest_short (the bar
    trades into the zone and closes back out of it in the zone's direction), stop_long / stop_short (the breaker's far
    side)."""
    from agents.indicators.smc import smc
    o, h, l, c, v = cols(df)
    n = len(c)
    sm = smc(df, k)
    out = {x: np.full(n, np.nan) for x in ["bull_top", "bull_bot", "bear_top", "bear_bot", "stop_long", "stop_short"]}
    out["retest_long"], out["retest_short"] = np.zeros(n, bool), np.zeros(n, bool)
    zb = zs = None
    for i in range(n):
        if zb is not None:
            if c[i] < zb[3] or i - zb[2] > max_age:
                zb = None
            elif i > zb[2] and l[i] <= zb[0] and c[i] > zb[0]:
                out["retest_long"][i] = True
        if zs is not None:
            if c[i] > zs[3] or i - zs[2] > max_age:
                zs = None
            elif i > zs[2] and h[i] >= zs[1] and c[i] < zs[1]:
                out["retest_short"][i] = True
        # a bullish FVG on this bar overlapping the live bullish breaker
        bt, bb = sm["brk_bull_top"][i], sm["brk_bull_bot"][i]
        if i >= 2 and l[i] > h[i - 2] and np.isfinite(bt):
            top, bot = min(l[i], bt), max(h[i - 2], bb)
            if top > bot:
                zb = (top, bot, i, bb)
        st, sb = sm["brk_bear_top"][i], sm["brk_bear_bot"][i]
        if i >= 2 and h[i] < l[i - 2] and np.isfinite(st):
            top, bot = min(l[i - 2], st), max(h[i], sb)
            if top > bot:
                zs = (top, bot, i, st)
        if zb is not None:
            out["bull_top"][i], out["bull_bot"][i], out["stop_long"][i] = zb[0], zb[1], zb[3]
        if zs is not None:
            out["bear_top"][i], out["bear_bot"][i], out["stop_short"][i] = zs[0], zs[1], zs[3]
    return out
