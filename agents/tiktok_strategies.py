"""Strategies harvested from TikTok (strategies/catalog.csv, ids TT..), coded as stated in
the video. Where the video leaves something open, the assumption is written in the
docstring. Same contract as agents/rules_agent.py: fn(df) -> (long, short, exit plan),
signals at the bar close, entry at the next bar's open.
"""
import numpy as np
import pandas as pd

from agents.rules_agent import atr, ema

NY = "America/New_York"


def _ny(df):
    idx = df.index.tz_convert(NY)
    return idx.hour.to_numpy(), pd.Series(idx.date).to_numpy()


def s_first4h_fakeout(df):
    """TT01 First 4H candle false breakout (@tradingdata2). New York time. Range = high/low
    of the day's first 4H candle (assumed 00:00-04:00 NY), usable once it has closed and
    until the NY day ends. On the 5-minute chart: a candle BODY closes outside the range,
    then a candle closes back inside -> fade it (broke up = short, broke down = long).
    Stop at the extreme of the breakout move, target 2R. Several trades per day allowed.
    Not modelled: moving the stop to a 'nearer key level' when the breakout is very wide."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    hour, day = _ny(df)
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    cur, rh, rl, state, ext = None, np.nan, np.nan, 0, np.nan
    for i in range(n):
        if day[i] != cur:
            cur, rh, rl, state, ext = day[i], -np.inf, np.inf, 0, np.nan
        if hour[i] < 4:
            rh, rl = max(rh, h[i]), min(rl, l[i])
            continue
        if not np.isfinite(rh) or not np.isfinite(rl):
            continue
        body_hi, body_lo = max(o[i], c[i]), min(o[i], c[i])
        if state == 0:
            if body_lo > rh:
                state, ext = 1, h[i]
            elif body_hi < rl:
                state, ext = -1, l[i]
        elif state == 1:
            ext = max(ext, h[i])
            if c[i] <= rh and c[i] >= rl:
                short[i], sl_s[i], state = True, ext, 0
        elif state == -1:
            ext = min(ext, l[i])
            if c[i] >= rl and c[i] <= rh:
                long_[i], sl_l[i], state = True, ext, 0
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


def _dealing_range(o, h, l, c, a, d, look=20, max_wait=60):
    """Signals for one direction of TT02 (d=+1 long). Mirror by negating prices."""
    n = len(c)
    sig = np.zeros(n, bool)
    stop, target = np.full(n, np.nan), np.full(n, np.nan)
    i = look
    while i < n - 1:
        cons_lo = l[i - look:i].min()
        cons_hi = h[i - look:i].max()
        # 1) false breakout below consolidation: wick below, close back inside within 3 bars
        if not (l[i] < cons_lo):
            i += 1
            continue
        lo, j, back = l[i], i, False
        for k in range(i, min(n, i + 3)):
            lo = min(lo, l[k])
            if c[k] > cons_lo:
                back, j = True, k
                break
        if not back:
            i += 1
            continue
        # 2) expansion: break of the consolidation high
        hi, k, expanded = -np.inf, j, False
        while k < min(n, j + max_wait):
            hi = max(hi, h[k])
            if c[k] > cons_hi:
                expanded = True
            if l[k] < lo:
                break
            # 3-5) after expansion, retrace to the 25% level overlapping a bullish FVG
            if expanded:
                lvl25 = lo + 0.25 * (hi - lo)
                if l[k] <= lvl25 and c[k] > lo:
                    fvg = any(l[m] > h[m - 2] and h[m - 2] <= lvl25 <= l[m]
                              for m in range(max(j + 2, 2), k))
                    if fvg:
                        sig[k], stop[k], target[k] = True, lo - 0.1 * a[k], hi
                    break
            k += 1
        i = max(k, i + 1)
    return sig, stop, target


def s_dealing_range(df):
    """TT02 Type 1 dealing range (@ali_khanict). False breakout below a 20-bar consolidation
    (close back inside within 3 bars), then expansion above the consolidation high. Dealing
    range = false-break low to expansion high. Buy when price retraces to the 25% level and
    that level sits inside a bullish fair value gap of the expansion. Targets in the video:
    50% level and range high -> modelled as target = range high with the stop moved to
    break-even at the 50% level. Stop below the range low (not stated; assumed). Mirror for
    shorts (assumed)."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    lg, sl_l, tg_l = _dealing_range(o, h, l, c, a, 1)
    sg, sl_s, tg_s = _dealing_range(-o, -l, -h, -c, a, -1)
    return lg, sg & ~lg, {"stop": ("level", sl_l, -sl_s), "target": ("level", tg_l, -tg_s),
                          "breakeven": (1 / 3, 0.0)}


def _htf_fvgs(df, rule="15min"):
    """Higher-timeframe FVG zones as (time_known, bottom, top, kind)."""
    g = df[["o", "h", "l", "c"]].resample(rule, label="left", closed="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    H, L = g["h"].to_numpy(), g["l"].to_numpy()
    known = g.index + pd.Timedelta(rule)
    zones = []
    for m in range(2, len(g)):
        if L[m] > H[m - 2]:
            zones.append((known[m], H[m - 2], L[m], 1))
        elif H[m] < L[m - 2]:
            zones.append((known[m], H[m], L[m - 2], -1))
    return zones


def s_ict_4conditions(df):
    """TT03 Sweep + HTF POI + IFVG + CISD (@alawadtrd). All four, in order (short shown,
    long mirrored): (1) price sweeps a major high (above the 20-bar high); (2) the sweep
    trades into a bearish 15-minute FVG (higher-timeframe point of interest); (3) a bullish
    FVG on the entry chart is then closed through (it inverts = IFVG); (4) CISD: a close below
    the open of the latest run of up-candles. Enter on the CISD close, stop above the IFVG
    top, target 2R (stated 1:2). Window: 40 bars after the sweep (assumed)."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    n = len(c)
    a = atr(h, l, c, 14)
    t = df.index
    zones = _htf_fvgs(df)
    zt = np.array([z[0].value for z in zones]) if zones else np.array([], dtype=np.int64)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    hi20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    lo20 = pd.Series(l).rolling(20).min().shift(1).to_numpy()

    def in_poi(i, kind, price):
        k = np.searchsorted(zt, t[i].value, side="right")
        for z in zones[max(0, k - 60):k]:
            if z[3] == kind and z[1] <= price <= z[2]:
                return True
        return False

    i = 22
    while i < n - 2:
        for d in (-1, 1):
            swept = (h[i] > hi20[i]) if d == -1 else (l[i] < lo20[i])
            if not swept or not in_poi(i, -1 if d == -1 else 1, h[i] if d == -1 else l[i]):
                continue
            ifvg_top = None
            for k in range(i + 1, min(n, i + 40)):
                if ifvg_top is None:
                    # an opposite FVG formed after the sweep, then closed through
                    for m in range(i + 2, k):
                        if d == -1 and l[m] > h[m - 2] and c[k] < h[m - 2]:
                            ifvg_top = l[m]
                            break
                        if d == 1 and h[m] < l[m - 2] and c[k] > l[m - 2]:
                            ifvg_top = h[m]
                            break
                    continue
                # CISD: close through the open of the latest opposite-candle run
                j = k - 1
                while j > i and (c[j] > o[j] if d == -1 else c[j] < o[j]):
                    j -= 1
                run_open = o[j + 1] if j + 1 < k else np.nan
                if np.isfinite(run_open) and ((d == -1 and c[k] < run_open) or (d == 1 and c[k] > run_open)):
                    if d == -1:
                        short[k], sl_s[k] = True, ifvg_top + 0.1 * a[k]
                    else:
                        long_[k], sl_l[k] = True, ifvg_top - 0.1 * a[k]
                    break
        i += 1
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


def s_cup_handle(df):
    """TT04 Cup and handle (@smart.forex.pips). Edge = a swing high (5 bars each side). Price
    falls at least 2 ATR, bottoms, and comes back to within 0.25 x depth of the edge within
    20-150 bars (the cup; bottom in the middle 60% of the cup = 'rounded', assumed). Then a
    pullback under the edge no deeper than half the cup (the handle). Entry: a candle closes
    above the edge. Stop under the handle low. Target = edge + cup depth. Long only (as in
    the video)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    a = atr(h, l, c, 14)
    long_ = np.zeros(n, bool)
    sl, tg = np.full(n, np.nan), np.full(n, np.nan)
    piv = [i for i in range(5, n - 5) if h[i] == h[i - 5:i + 6].max()]
    used_until = -1
    for e in piv:
        if e < used_until:
            continue
        edge = h[e]
        end = min(n, e + 6 + 150)
        k = e + 6  # pivot is confirmed 5 bars later
        bottom, b_i = np.inf, None
        returned = None
        while k < end:
            if h[k] > edge:
                break
            if l[k] < bottom:
                bottom, b_i = l[k], k
            depth = edge - bottom
            if depth >= 2 * a[k] and h[k] >= edge - 0.25 * depth and k - e >= 20:
                frac = (b_i - e) / (k - e)
                if 0.2 <= frac <= 0.8:
                    returned = k
                break
            k += 1
        if returned is None:
            continue
        depth = edge - bottom
        hl = np.inf
        for m in range(returned + 1, min(n, returned + max(5, (returned - e) // 3))):
            hl = min(hl, l[m])
            if edge - hl > 0.5 * depth:
                break
            if c[m] > edge and hl < edge:
                long_[m], sl[m], tg[m] = True, hl - 0.1 * a[m], edge + depth
                used_until = m
                break
    return long_, np.zeros(n, bool), {"stop": ("level", sl, sl), "target": ("level", tg, tg)}


def _ny_minutes(df):
    idx = df.index.tz_convert(NY)
    return (idx.hour * 60 + idx.minute).to_numpy(), pd.Series(idx.date).to_numpy()


def _opening_range(df, start_min=570):
    """High/low of the bar that starts at 9:30 NY (start_min) each day, known from the next bar."""
    h, l = df["h"].to_numpy(), df["l"].to_numpy()
    mins, day = _ny_minutes(df)
    n = len(h)
    rh, rl, known = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    cur, ch, cl, seen = None, np.nan, np.nan, False
    for i in range(n):
        if day[i] != cur:
            cur, ch, cl, seen = day[i], np.nan, np.nan, False
        if mins[i] == start_min:
            ch, cl, seen = h[i], l[i], True
            continue
        if seen:
            rh[i], rl[i], known[i] = ch, cl, True
    return rh, rl, known, mins


def s_ny_orb_15m(df):
    """TT11 NY opening-range breakout, 15 minutes (@_itskinglouis, NAS100). Mark the high/low
    of the 9:30 New York 15-minute candle. First full candle close above (below) the range ->
    enter at the next candle. Stop below (above) the recent low (high) = extreme of the last 3
    candles (assumed). Target 2.5R. One trade per day; only until 16:00 NY (assumed)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    rh, rl, known, mins = _opening_range(df)
    _, day = _ny_minutes(df)
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    lo3 = pd.Series(l).rolling(3).min().to_numpy()
    hi3 = pd.Series(h).rolling(3).max().to_numpy()
    done = set()
    for i in range(n):
        if not known[i] or mins[i] >= 960 or day[i] in done:
            continue
        if c[i] > rh[i]:
            long_[i] = True
            done.add(day[i])
        elif c[i] < rl[i]:
            short[i] = True
            done.add(day[i])
    return long_, short, {"stop": ("level", lo3, hi3), "target": ("R", 2.5)}


def s_orb_retest_5m(df):
    """TT13 Opening range breakout and retest, 5 minutes (@thecrypteller, also @cyforex2).
    Range = first 5-minute candle after the 9:30 NY open. Wait for a close beyond the range,
    then the first candle that taps the broken level and closes back on the breakout side ->
    enter in the breakout direction. Stop beyond the other side of the range. Target 2R.
    One trade per day."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    rh, rl, known, mins = _opening_range(df)
    _, day = _ny_minutes(df)
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    state, cur, done = 0, None, False
    for i in range(n):
        if day[i] != cur:
            cur, state, done = day[i], 0, False
        if not known[i] or done or mins[i] >= 960:
            continue
        if state == 0:
            state = 1 if c[i] > rh[i] else -1 if c[i] < rl[i] else 0
        elif state == 1 and l[i] <= rh[i] and c[i] > rh[i]:
            long_[i], done = True, True
        elif state == -1 and h[i] >= rl[i] and c[i] < rl[i]:
            short[i], done = True, True
    a = atr(h, l, c, 14)
    return long_, short, {"stop": ("level", rl - 0.1 * a, rh + 0.1 * a), "target": ("R", 2.0)}


def s_prev_ny_session_fade(df):
    """TT16 Previous New York session high/low fade (@forextristan, US30). Mark yesterday's
    NY session (9:30-16:00 NY) high and low. Today, during the NY session, the first touch of
    either level is faded: short at the high, long at the low. Stop slightly beyond the level
    (0.1 ATR, assumed). Target 4R (stated)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    mins, day = _ny_minutes(df)
    a = atr(h, l, c, 14)
    sess = (mins >= 570) & (mins < 960)
    g = pd.DataFrame({"d": day, "h": np.where(sess, h, np.nan), "l": np.where(sess, l, np.nan)})
    daily = g.groupby("d").agg(sh=("h", "max"), sl=("l", "min")).shift(1)
    m = g[["d"]].join(daily, on="d")
    ph, pl = m["sh"].to_numpy(), m["sl"].to_numpy()
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    cur, used = None, False
    for i in range(n):
        if day[i] != cur:
            cur, used = day[i], False
        if used or not sess[i] or not np.isfinite(ph[i]):
            continue
        if h[i] >= ph[i] and c[i] < ph[i]:
            short[i], used = True, True
        elif l[i] <= pl[i] and c[i] > pl[i]:
            long_[i], used = True, True
    return long_, short, {"stop": ("level", pl - 0.1 * a, ph + 0.1 * a), "target": ("R", 4.0)}


def s_fib_golden_zone(df):
    """TT12 Fibonacci golden zone pullback, 5 minutes (@il.gufo.notturno). Trend up when price
    breaks the recent swing high (20-bar high). Fib from the swing low before the break to the
    new high. When price pulls back into 0.618-0.79 and a bullish candle closes there, buy.
    Stop just below the 0.79 level (stated), target 1:2 (stated). Mirror for downtrends."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    hi20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    lo20 = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    i = 21
    while i < n - 1:
        d = 1 if h[i] > hi20[i] else -1 if l[i] < lo20[i] else 0
        if d == 0:
            i += 1
            continue
        base = lo20[i] if d == 1 else hi20[i]
        ext = h[i] if d == 1 else l[i]
        k = i + 1
        while k < min(n, i + 60):
            if d == 1:
                if h[k] > ext and l[k] > ext - 0.618 * (ext - base):
                    ext = h[k]
                rng = ext - base
                z1, z2 = ext - 0.618 * rng, ext - 0.79 * rng
                if l[k] < base:
                    break
                if l[k] <= z1 and c[k] >= z2 and c[k] > o[k]:
                    long_[k], sl_l[k] = True, z2 - 0.05 * a[k]
                    break
            else:
                if l[k] < ext and h[k] < ext + 0.618 * (base - ext):
                    ext = l[k]
                rng = base - ext
                z1, z2 = ext + 0.618 * rng, ext + 0.79 * rng
                if h[k] > base:
                    break
                if h[k] >= z1 and c[k] <= z2 and c[k] < o[k]:
                    short[k], sl_s[k] = True, z2 + 0.05 * a[k]
                    break
            k += 1
        i = k + 1
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


def s_15m_candle_sweep_1m(df):
    """TT19 15-minute candle range sweep, executed on 1 minute (@protradingschool0). Range =
    the previous completed 15-minute candle. On the 1-minute chart, price breaks the range and
    a candle closes back inside -> enter at that close, against the break. Stop beyond the
    sweep's extreme. Target: 'next key level' in the video -> 2R assumed."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    g = df[["h", "l"]].resample("15min", label="left", closed="left").agg({"h": "max", "l": "min"})
    prev = g.shift(1).reindex(df.index, method="ffill")
    rh, rl = prev["h"].to_numpy(), prev["l"].to_numpy()
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    block = df.index.floor("15min")
    state, ext, cur = 0, np.nan, None
    for i in range(n):
        if block[i] != cur:
            cur, state, ext = block[i], 0, np.nan
        if not np.isfinite(rh[i]):
            continue
        if state == 0:
            if h[i] > rh[i]:
                state, ext = 1, h[i]
            elif l[i] < rl[i]:
                state, ext = -1, l[i]
        if state == 1:
            ext = max(ext, h[i])
            if c[i] < rh[i]:
                short[i], sl_s[i], state = True, ext, 2
        elif state == -1:
            ext = min(ext, l[i])
            if c[i] > rl[i]:
                long_[i], sl_l[i], state = True, ext, 2
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


LDN = "Europe/London"


def _local(df, tz):
    idx = df.index.tz_convert(tz)
    return (idx.hour * 60 + idx.minute).to_numpy(), pd.Series(idx.date).to_numpy()


def _fvg_retest(o, h, l, c, a, d, start, end):
    """First FVG in direction d formed in [start, end) and retested before end.
    Returns (entry_bar, stop) or (None, None). Long: gap = h[m-2] .. l[m]; entry when price dips
    into it and closes above its bottom; stop under the gap."""
    for m in range(max(start, 2), end):
        if d == 1 and l[m] > h[m - 2]:
            bot, top = h[m - 2], l[m]
            for k in range(m + 1, end):
                if c[k] < bot:
                    break
                if l[k] <= top and c[k] > bot:
                    return k, bot - 0.1 * a[k]
        if d == -1 and h[m] < l[m - 2]:
            bot, top = h[m], l[m - 2]
            for k in range(m + 1, end):
                if c[k] > top:
                    break
                if h[k] >= bot and c[k] < top:
                    return k, top + 0.1 * a[k]
    return None, None


def _range_sweep_fvg(df, tz, r_start, r_end, w_end, sweep_stop):
    """Shared engine: daily range [r_start, r_end) local minutes; after it, until w_end, a sweep of
    one side then an FVG retest in the reversal direction -> trade to the opposite end."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    mins, day = _local(df, tz)
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    ends = np.r_[starts[1:], n]
    for s0, e0 in zip(starts, ends):
        rng = np.flatnonzero((mins[s0:e0] >= r_start) & (mins[s0:e0] < r_end)) + s0
        win = np.flatnonzero((mins[s0:e0] >= r_end) & (mins[s0:e0] < w_end)) + s0
        if len(rng) == 0 or len(win) == 0:
            continue
        rh, rl = h[rng].max(), l[rng].min()
        w0, w1 = win[0], win[-1] + 1
        for i in range(w0, w1):
            if h[i] > rh or l[i] < rl:
                d = -1 if h[i] > rh else 1
                k, st = _fvg_retest(o, h, l, c, a, d, i, w1)
                if k is None:
                    break
                ext = h[i:k + 1].max() if d == -1 else l[i:k + 1].min()
                if d == 1:
                    long_[k], tg_l[k] = True, rh
                    sl_l[k] = (ext - 0.1 * a[k]) if sweep_stop else st
                else:
                    short[k], tg_s[k] = True, rl
                    sl_s[k] = (ext + 0.1 * a[k]) if sweep_stop else st
                break
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("level", tg_l, tg_s)}


def s_london_1h_crt(df):
    """TT35 London opening 1-hour candle range (@wesleytfxc, @primeswingtrader). Range = the
    08:00-09:00 London candle. A wick through one side (the body back inside) -> on the lower
    timeframe enter at the first fair value gap retest in the reversal direction. Stop just
    beyond the FVG, target the opposite end of the 1-hour candle. Window until 12:00 London
    (assumed)."""
    return _range_sweep_fvg(df, LDN, 480, 540, 720, sweep_stop=False)


def s_london_twilight(df):
    """TT36 'London Twilight Model' (@massimotsr, 5 videos). Range = the 2:00 AM New York 1-hour
    candle. Sweep of its high (low), then on 1 minute an inverted FVG / order block in the
    reversal direction -> enter. Stop at the swing extreme of the sweep, target the opposite
    end of the 2 AM range. Window 03:00-08:00 NY (assumed). The optional higher-timeframe key
    level tap is not modelled."""
    return _range_sweep_fvg(df, NY, 120, 180, 480, sweep_stop=True)


def s_silver_bullet(df):
    """TT39 ICT Silver Bullet (@blosefx, @fluxchart, @the.trading.geek1). Only 10:00-11:00 NY.
    A sweep of the prior 60-bar high (low), then an FVG in the reversal direction forms and is
    retested -> enter. Stop beyond the sweep extreme (@solidtrader85's correction), target 2R
    (stated minimum)."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    mins, day = _ny_minutes(df)
    n = len(c)
    hi = pd.Series(h).rolling(60).max().shift(1).to_numpy()
    lo = pd.Series(l).rolling(60).min().shift(1).to_numpy()
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    inwin = (mins >= 600) & (mins < 660)
    done = set()
    for i in np.flatnonzero(inwin):
        if day[i] in done or not np.isfinite(hi[i]):
            continue
        d = -1 if h[i] > hi[i] else 1 if l[i] < lo[i] else 0
        if d == 0:
            continue
        end = i
        while end < n and inwin[end] and day[end] == day[i]:
            end += 1
        k, _ = _fvg_retest(o, h, l, c, a, d, i, end)
        if k is None:
            continue
        done.add(day[i])
        if d == 1:
            long_[k], sl_l[k] = True, l[i:k + 1].min() - 0.1 * a[k]
        else:
            short[k], sl_s[k] = True, h[i:k + 1].max() + 0.1 * a[k]
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


def s_opr_vwap(df):
    """TT40 Opening range + VWAP (@horiizonfx, gold). Range 15:00-15:30 Paris = 09:00-09:30 NY.
    A close beyond it counts only if the session VWAP (from NY midnight) is on the other side of
    price (break down -> VWAP above). Wait for a retest of the broken edge that closes back
    outside -> enter. Stop just beyond the VWAP, target 1:3. One trade per day."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    v = df["v"].to_numpy() if "v" in df else np.ones(len(c))
    v = np.where(v > 0, v, 1.0)
    a = atr(h, l, c, 14)
    mins, day = _ny_minutes(df)
    tp = (h + l + c) / 3
    g = pd.Series(day)
    vwap = (pd.Series(tp * v).groupby(g).cumsum() / pd.Series(v).groupby(g).cumsum()).to_numpy()
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    cur, rh, rl, state, done = None, -np.inf, np.inf, 0, False
    for i in range(n):
        if day[i] != cur:
            cur, rh, rl, state, done = day[i], -np.inf, np.inf, 0, False
        if 540 <= mins[i] < 570:
            rh, rl = max(rh, h[i]), min(rl, l[i])
            continue
        if done or mins[i] < 570 or mins[i] >= 960 or not np.isfinite(rh):
            continue
        if state == 0:
            if c[i] > rh and vwap[i] < c[i]:
                state = 1
            elif c[i] < rl and vwap[i] > c[i]:
                state = -1
        elif state == 1 and l[i] <= rh and c[i] > rh:
            long_[i], sl_l[i], done = True, min(vwap[i], rl) - 0.1 * a[i], True
        elif state == -1 and h[i] >= rl and c[i] < rl:
            short[i], sl_s[i], done = True, max(vwap[i], rh) + 0.1 * a[i], True
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 3.0)}


def s_ny_8am_retest(df):
    """TT41 New York 8:00 AM 15-minute candle break and retest (@rp_profits). Zone = the 08:00
    ET 15-minute candle. From 09:00 ET the first close beyond the zone sets the direction; the
    retest of the zone that closes back on the breakout side is the entry. Stop slightly beyond
    the other side of the zone, target 1:3 (stated). One trade per day, until 12:00 ET."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    mins, day = _ny_minutes(df)
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    cur, zh, zl, state, done = None, np.nan, np.nan, 0, False
    for i in range(n):
        if day[i] != cur:
            cur, zh, zl, state, done = day[i], np.nan, np.nan, 0, False
        if mins[i] == 480:
            zh, zl = h[i], l[i]
            continue
        if done or not np.isfinite(zh) or mins[i] < 540 or mins[i] >= 720:
            continue
        if state == 0:
            state = 1 if c[i] > zh else -1 if c[i] < zl else 0
        elif state == 1 and l[i] <= zh and c[i] > zh:
            long_[i], sl_l[i], done = True, zl - 0.1 * a[i], True
        elif state == -1 and h[i] >= zl and c[i] < zl:
            short[i], sl_s[i], done = True, zh + 0.1 * a[i], True
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 3.0)}


def s_ema_9_21_1h(df):
    """TT45 EMA 9/21 crossover on 1 hour (@whitehattraders27). Bullish cross buys, bearish cross
    sells. Stop below/above the recent swing (lowest/highest of 10 bars, assumed). Target 1:2."""
    c = df["c"].to_numpy()
    e9, e21 = ema(c, 9), ema(c, 21)
    up = np.zeros(len(c), bool)
    dn = np.zeros(len(c), bool)
    up[1:] = (e9[1:] > e21[1:]) & (e9[:-1] <= e21[:-1])
    dn[1:] = (e9[1:] < e21[1:]) & (e9[:-1] >= e21[:-1])
    return up, dn, {"stop": ("lowest", 10), "target": ("R", 2.0)}


def s_break_retest(df):
    """TT56 Break and retest of a swing level (@user1239509627, @tarzantrader, @realnolanivey,
    @wchampfx, @savvybtrades). Level = the 20-bar high (low). A close through it, then within 10
    bars a retest that touches the level and closes back on the breakout side -> enter. Stop
    beyond the retested level (0.3 ATR, assumed), target 2:1 (stated by @realnolanivey)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    hi = pd.Series(h).rolling(20).max().shift(1).to_numpy()
    lo = pd.Series(l).rolling(20).min().shift(1).to_numpy()
    n = len(c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    i = 21
    while i < n - 1:
        d = 1 if c[i] > hi[i] else -1 if c[i] < lo[i] else 0
        if d == 0:
            i += 1
            continue
        lvl = hi[i] if d == 1 else lo[i]
        k = i + 1
        while k < min(n, i + 11):
            if d == 1 and c[k] < lvl - 0.3 * a[k]:
                break
            if d == -1 and c[k] > lvl + 0.3 * a[k]:
                break
            if d == 1 and l[k] <= lvl and c[k] > lvl:
                long_[k], sl_l[k] = True, lvl - 0.3 * a[k]
                break
            if d == -1 and h[k] >= lvl and c[k] < lvl:
                short[k], sl_s[k] = True, lvl + 0.3 * a[k]
                break
            k += 1
        i = k + 1
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


# id: (name, fn, timeframes stated in the video or None = every timeframe)
TIKTOK = {
    "TT01": ("First 4H candle false breakout (TikTok)", s_first4h_fakeout, ["5Min"]),
    "TT02": ("Type 1 dealing range 25% + FVG (TikTok)", s_dealing_range, None),
    "TT03": ("ICT sweep+POI+IFVG+CISD (TikTok)", s_ict_4conditions, ["1Min", "5Min"]),
    "TT04": ("Cup and handle breakout (TikTok)", s_cup_handle, None),
    "TT11": ("NY 9:30 opening range breakout 15m, 2.5R (TikTok)", s_ny_orb_15m, ["15Min"]),
    "TT12": ("Fibonacci golden zone pullback 5m (TikTok)", s_fib_golden_zone, ["5Min"]),
    "TT13": ("Opening range break and retest 5m (TikTok)", s_orb_retest_5m, ["5Min"]),
    "TT16": ("Previous NY session high/low fade, 4R (TikTok)", s_prev_ny_session_fade, ["5Min", "15Min"]),
    "TT19": ("15m candle range sweep on 1m (TikTok)", s_15m_candle_sweep_1m, ["1Min"]),
    "TT35": ("London 1H opening candle sweep + FVG (TikTok)", s_london_1h_crt, ["5Min"]),
    "TT36": ("London Twilight Model 2AM range (TikTok)", s_london_twilight, ["1Min"]),
    "TT39": ("ICT Silver Bullet 10-11 NY (TikTok)", s_silver_bullet, ["1Min", "5Min"]),
    "TT40": ("Opening range + VWAP filter, 1:3 (TikTok)", s_opr_vwap, ["5Min"]),
    "TT41": ("NY 8AM 15m candle break and retest, 1:3 (TikTok)", s_ny_8am_retest, ["15Min"]),
    "TT45": ("EMA 9/21 crossover 1H, 1:2 (TikTok)", s_ema_9_21_1h, ["1Hour"]),
    "TT56": ("Break and retest of swing level, 2:1 (TikTok)", s_break_retest, ["15Min", "1Hour"]),
}
