"""Auction-market 'unfinished business' levels (scout 16.11.2026) - OUR OWN builds from the Market Profile definitions:

- naked_poc: the point of control of a completed session that price has not traded back to since (LuxAlgo 'Naked POC
  As Level', https://www.luxalgo.com/library/concept/naked-poc-as-level/ ; NexusFi
  https://nexusfi.com/a/market-structure/naked-vpoc-futures-trading). Session POC from agents/indicators/volume_profile
  (tick volume). A naked POC is retired on the first bar whose range covers it. Uses: magnet targets and first-touch
  reaction levels.
- poor_extremes: poor high / poor low - a session extreme made by two or more 30-minute TPO periods at (almost) the same
  price, a flat ledge without a tail (LuxAlgo 'Poor High / Poor Low', https://www.luxalgo.com/library/concept/poor-high-poor-low/ ;
  marketcalls https://www.marketcalls.in/market-profile/poor-high-and-poor-low-market-profile-structure-explained.html).
  'Almost the same' = within 0.1 ATR(14) (ours). Repaired (retired) when a later bar trades 0.1 ATR beyond it.
- single_prints: price zones inside a completed session's range touched by only ONE 30-minute TPO period, away from both
  extremes (LuxAlgo 'Single Prints', https://www.luxalgo.com/library/concept/single-prints/). Price grid = 1/40 of the
  session range (ours); a zone needs 2+ adjacent single bins. Retired once a later bar has traded through all of it.
Everything comes from COMPLETED sessions and the bar's own range: the value at bar i uses bars <= i.
"""
import numpy as np

from agents.indicators._core import atr, cols, session_day
from agents.indicators.volume_profile import volume_profile


def _nearest(levels, x):
    if not levels:
        return np.nan, np.nan
    arr = np.asarray(levels, float)
    up, dn = arr[arr > x], arr[arr < x]
    return (up.min() if len(up) else np.nan), (dn.max() if len(dn) else np.nan)


def naked_poc(df, tz="UTC", start_hour=0, max_age=30):
    """dict: above / below (nearest naked POC over / under the close after this bar's touches), touched (the level a
    bar retired, NaN elsewhere), touch_dir (+1 the level was touched from above = support test, -1 from below), count."""
    o, h, l, c, v = cols(df)
    n = len(c)
    poc = volume_profile(df, tz, start_hour)["poc"]
    day = session_day(df, tz, start_hour)
    out = {k: np.full(n, np.nan) for k in ("above", "below", "touched", "touch_dir", "count")}
    live = []  # [level, session number]
    sess = -1
    for i in range(n):
        if i == 0 or day[i] != day[i - 1]:
            sess += 1
            if np.isfinite(poc[i]):
                live.append([float(poc[i]), sess])
            live = [x for x in live if sess - x[1] <= max_age]
        pc = c[i - 1] if i else o[i]
        keep, hit = [], []
        for x in live:
            if l[i] <= x[0] <= h[i]:
                hit.append(x[0])
            else:
                keep.append(x)
        live = keep
        if hit:
            lv = min(hit, key=lambda y: abs(y - o[i]))
            out["touched"][i] = lv
            out["touch_dir"][i] = 1.0 if pc > lv else -1.0
        out["above"][i], out["below"][i] = _nearest([x[0] for x in live], c[i])
        out["count"][i] = len(live)
    return out


def _sessions(df, tz, start_hour):
    """{first bar of a session: (first, last) bar of the session BEFORE it} - a session counts as completed only
    once the next one has started (no peeking at where a session will end)."""
    day = session_day(df, tz, start_hour)
    starts = np.r_[0, np.where(day[1:] != day[:-1])[0] + 1]
    return {int(starts[k + 1]): (int(starts[k]), int(starts[k + 1]) - 1) for k in range(len(starts) - 1)}


def _periods(df, a, b, minutes=30):
    t = df.index[a:b + 1]
    m = (t.hour * 60 + t.minute) // minutes
    return np.asarray(m)


def poor_extremes(df, tz="UTC", start_hour=0, tol_atr=0.1, max_age=20):
    """dict: poor_high / poor_low (nearest unrepaired poor high over / poor low under the close), new_high / new_low
    (the poor high / low of the session that just completed, on the next session's first bar; NaN elsewhere)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    out = {k: np.full(n, np.nan) for k in ("poor_high", "poor_low", "new_high", "new_low")}
    done = _sessions(df, tz, start_hour)
    ph, pl = [], []  # [level, tol, age]
    for i in range(n):
        tol_i = tol_atr * (a[i] if np.isfinite(a[i]) else (h[i] - l[i]))
        if i in done:
            s, e = done[i]
            if e > s:
                per = _periods(df, s, e)
                hh, ll = h[s:e + 1], l[s:e + 1]
                sh, sl = hh.max(), ll.min()
                at_h = np.unique(per[hh >= sh - tol_i])
                at_l = np.unique(per[ll <= sl + tol_i])
                if len(at_h) >= 2:
                    out["new_high"][i] = sh
                    ph.append([sh, tol_i, 0])
                if len(at_l) >= 2:
                    out["new_low"][i] = sl
                    pl.append([sl, tol_i, 0])
                for x in ph + pl:
                    x[2] += 1
                ph = [x for x in ph if x[2] <= max_age]
                pl = [x for x in pl if x[2] <= max_age]
        ph = [x for x in ph if not h[i] > x[0] + x[1]]
        pl = [x for x in pl if not l[i] < x[0] - x[1]]
        up = [x[0] for x in ph if x[0] > c[i]]
        dn = [x[0] for x in pl if x[0] < c[i]]
        out["poor_high"][i] = min(up) if up else np.nan
        out["poor_low"][i] = max(dn) if dn else np.nan
    return out


def single_prints(df, tz="UTC", start_hour=0, grid=40, minutes=30):
    """dict: zone_top / zone_bot of the nearest live single-print zone over the close (above_*) and under it
    (below_*), and entered (+1 a bar entered a zone from below, -1 from above, 0 none) with that zone's far edge in
    'far_edge' (NaN elsewhere)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    keys = ("above_bot", "above_top", "below_bot", "below_top", "far_edge")
    out = {k: np.full(n, np.nan) for k in keys}
    out["entered"] = np.zeros(n)
    done = _sessions(df, tz, start_hour)
    zones = []  # [bot, top, lo_reached, hi_reached]
    for i in range(n):
        pc = c[i - 1] if i else o[i]
        if i in done:
            s, e = done[i]
            if e - s >= 3:
                sh, sl = h[s:e + 1].max(), l[s:e + 1].min()
                if sh > sl:
                    step = (sh - sl) / grid
                    per = _periods(df, s, e, minutes)
                    cnt = np.zeros(grid)
                    for p in np.unique(per):
                        m = per == p
                        ph_, pl_ = h[s:e + 1][m].max(), l[s:e + 1][m].min()
                        b0 = int(np.clip(np.floor((pl_ - sl) / step), 0, grid - 1))
                        b1 = int(np.clip(np.ceil((ph_ - sl) / step) - 1, 0, grid - 1))
                        cnt[b0:b1 + 1] += 1
                    one = cnt == 1
                    k = 0
                    while k < grid:
                        if one[k]:
                            j = k
                            while j + 1 < grid and one[j + 1]:
                                j += 1
                            if k > 0 and j < grid - 1 and j - k + 1 >= 2:
                                zones.append([sl + k * step, sl + (j + 1) * step])
                            k = j + 1
                        else:
                            k += 1
                    zones = zones[-12:]
        keep = []
        for z in zones:
            bot, top = z[0], z[1]
            if l[i] <= bot and h[i] >= top:
                continue  # traded through: filled
            if out["entered"][i] == 0 and h[i] > bot and l[i] < top:
                if pc <= bot:
                    out["entered"][i], out["far_edge"][i] = 1, top
                elif pc >= top:
                    out["entered"][i], out["far_edge"][i] = -1, bot
            keep.append(z)
        zones = keep
        ab = [z for z in zones if z[0] > c[i]]
        be = [z for z in zones if z[1] < c[i]]
        if ab:
            z = min(ab, key=lambda y: y[0])
            out["above_bot"][i], out["above_top"][i] = z[0], z[1]
        if be:
            z = max(be, key=lambda y: y[1])
            out["below_bot"][i], out["below_top"][i] = z[0], z[1]
    return out
