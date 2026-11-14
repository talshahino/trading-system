"""ICT session models built from their published definitions (OUR BUILDS; closed bars only - no repaint).

judas_open_range - Judas swing of the opening range with a CISD confirmation (TradingFinder 'ICT Setup 03 Judas Swing
NY 9:30am + CHoCH/FVG', open source https://www.tradingview.com/script/mRmLRVSW-ICT-Setup-03-TradingFinder-Judas-Swing-NY-9-30am-CHoCH-FVG/;
the user's task tal1411gapplus: 'Judas Swing: range 9:30-9:45 New York'). Range = high / low of the bars starting in
[range_start, range_end) local time; in [range_end, win_end) a sweep of one side, then CISD = a close back beyond the
open of the run of candles that made the sweep leg (the change in state of delivery). TradingFinder's windows: stocks
09:15-09:30 range / 09:30-09:45 move, forex 08:15-08:30 / 08:30-08:45; the user: range 09:30-09:45. One signal per day.

silver_bullet - ICT Silver Bullet (LuxAlgo concept https://www.luxalgo.com/library/concept/silver-bullet/): windows
03:00-04:00, 10:00-11:00 and 14:00-15:00 New York. Liquidity pools mapped at the window start = highest high / lowest
low of the previous `look_min` minutes (ours: 180 minutes - the session before the window); inside the window a sweep of one pool, then displacement the other way
leaving an FVG formed inside the window; entry when price trades back into that gap and closes out of it in the
trade's direction; stop beyond the sweep extreme; target the opposing pool. One per window."""
import numpy as np
import pandas as pd

from agents.indicators._core import atr, cols


def _clock(df, tz):
    t = df.index.tz_convert(tz)
    return (t.hour * 60 + t.minute).to_numpy(), pd.Series(t.date).to_numpy()


def judas_open_range(df, tz="America/New_York", range_start=570, range_end=585, win_end=660):
    """Times in minutes after local midnight (defaults 09:30, 09:45, 11:00 New York). dict: rng_hi, rng_lo, bull, bear,
    sweep_lo, sweep_hi (extreme of the sweep on the signal bar), cisd (the level the close crossed)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    mins, day = _clock(df, tz)
    out = {x: np.full(n, np.nan) for x in ["rng_hi", "rng_lo", "sweep_lo", "sweep_hi", "cisd"]}
    out["bull"] = np.zeros(n, bool)
    out["bear"] = np.zeros(n, bool)
    cur, rh, rl, done = None, -np.inf, np.inf, False
    swl = swh = np.nan
    lvl_b = lvl_s = np.nan
    for i in range(n):
        if day[i] != cur:
            cur, rh, rl, done = day[i], -np.inf, np.inf, False
            swl = swh = lvl_b = lvl_s = np.nan
        m = mins[i]
        if range_start <= m < range_end:
            rh, rl = max(rh, h[i]), min(rl, l[i])
            continue
        if not (range_end <= m < win_end) or not np.isfinite(rh) or not np.isfinite(rl) or done:
            continue
        out["rng_hi"][i], out["rng_lo"][i] = rh, rl
        if l[i] < rl and (not np.isfinite(swl) or l[i] < swl):
            swl = l[i]
            j, lv = i, np.nan
            while j > 0 and c[j] < o[j]:      # the run of bearish candles of the sweep leg
                lv = o[j] if not np.isfinite(lv) else max(lv, o[j])
                j -= 1
            if not np.isfinite(lv):
                lv = o[i] if c[i - 1] >= o[i - 1] else max(o[i], o[i - 1])
            lvl_b = lv
        if h[i] > rh and (not np.isfinite(swh) or h[i] > swh):
            swh = h[i]
            j, lv = i, np.nan
            while j > 0 and c[j] > o[j]:
                lv = o[j] if not np.isfinite(lv) else min(lv, o[j])
                j -= 1
            if not np.isfinite(lv):
                lv = o[i] if c[i - 1] <= o[i - 1] else min(o[i], o[i - 1])
            lvl_s = lv
        if np.isfinite(swl) and np.isfinite(lvl_b) and c[i] > lvl_b and c[i] > o[i]:
            out["bull"][i], out["sweep_lo"][i], out["cisd"][i] = True, swl, lvl_b
            done = True
        elif np.isfinite(swh) and np.isfinite(lvl_s) and c[i] < lvl_s and c[i] < o[i]:
            out["bear"][i], out["sweep_hi"][i], out["cisd"][i] = True, swh, lvl_s
            done = True
    return out


def silver_bullet(df, look_min=180, windows=((180, 240), (600, 660), (840, 900)), grace=60):
    """dict: long, short (entry bars), stop_l, stop_s (beyond the sweep), tgt_l, tgt_s (opposing pool), win (in a window).
    The sweep and the gap's displacement (middle) candle must be inside the window; the return into the gap may come up
    to `grace` minutes after the window (ours - on 15-minute bars a window has only four bars)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    mins, day = _clock(df, "America/New_York")
    bm = float(pd.Series(df.index).diff().median().total_seconds() / 60.0) if n > 1 else 5.0
    look = max(int(round(look_min / max(bm, 1.0))), 3)
    wid = np.full(n, -1)
    inw = np.zeros(n, bool)
    for k, (s, e) in enumerate(windows):
        wid = np.where((mins >= s) & (mins < e + grace), k, wid)
        inw |= (mins >= s) & (mins < e)
    out = {x: np.full(n, np.nan) for x in ["stop_l", "stop_s", "tgt_l", "tgt_s"]}
    out["long"], out["short"] = np.zeros(n, bool), np.zeros(n, bool)
    out["win"] = inw
    key = None
    for i in range(look + 2, n):
        if wid[i] < 0:
            key = None
            continue
        if (day[i], wid[i]) != key:
            if not inw[i]:
                continue
            key = (day[i], wid[i])
            ph, pl = h[i - look:i].max(), l[i - look:i].min()
            swh = swl = np.nan
            gap_b = gap_s = None
            done = False
        if done:
            continue
        if inw[i]:
            if h[i] > ph:
                swh = h[i] if not np.isfinite(swh) else max(swh, h[i])
            if l[i] < pl:
                swl = l[i] if not np.isfinite(swl) else min(swl, l[i])
        if inw[i - 1]:     # the displacement (middle) candle of the gap inside the window
            if np.isfinite(swh) and h[i] < l[i - 2]:
                gap_s = (l[i - 2], h[i])
                continue
            if np.isfinite(swl) and l[i] > h[i - 2]:
                gap_b = (l[i], h[i - 2])
                continue
        if gap_s is not None and h[i] >= gap_s[1] and c[i] < gap_s[0] and c[i] < o[i]:
            st = swh + 0.1 * a[i]
            if c[i] < st and pl < c[i]:
                out["short"][i], out["stop_s"][i], out["tgt_s"][i] = True, st, pl
                done = True
                continue
        if gap_b is not None and l[i] <= gap_b[0] and c[i] > gap_b[1] and c[i] > o[i]:
            st = swl - 0.1 * a[i]
            if c[i] > st and ph > c[i]:
                out["long"][i], out["stop_l"][i], out["tgt_l"][i] = True, st, ph
                done = True
    return out
