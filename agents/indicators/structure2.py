"""Price-structure indicators the library lacked (scout 15.11.2026, night 3). No repaint: every pattern is known at the
bar where its last confirming bar has CLOSED (pivots k bars late, Darvas sides 3 bars late, chart reversals on the
closing bar).

- darvas_box: Nicolas Darvas' box ("How I Made $2,000,000 in the Stock Market", 1960; open-source ports e.g.
  https://www.tradingview.com/script/hZV01YX6-Advanced-Intraday-Darvas-Box/): top = a 20-bar high not exceeded for the
  next 3 bars; bottom = the lowest low after the top not broken for the next 3 bars; breakout = a close over the top
  of a complete box, breakdown = a close under its bottom.
- vsa: Volume Spread Analysis bar classes (Tom Williams, "Master the Markets"; open-source ports
  https://www.tradingview.com/script/OBV1KONK-No-Supply-No-Demand-NSND-Volume-Spread-Analysis-Tool/,
  https://www.prorealcode.com/prorealtime-indicators/tac-vsa-volume-spread-analysis-2): no demand / no supply (narrow
  spread = true range under its 20-bar average, volume under each of the two previous bars), upthrust / spring (a new
  20-bar extreme that closes back inside the broken extreme, volume > 1.5 x average), stopping volume (down bar,
  volume > 2 x average, close in the upper half of the true range). Spread and close location use the TRUE range
  (a gap from the previous close counts, as in Twiggs' money flow).
  Tick volume stands in for volume; a feed without volume information (constant v) passes the volume tests
  (library convention).
- ross_123: Joe Ross 1-2-3 reversal and Ross hook (https://www.luxalgo.com/library/concept/1-2-3-reversal/,
  https://fxopen.com/blog/en/how-to-trade-with-a-ross-hook-pattern): confirmed swings (k 3) low 1 (a 30-bar low),
  high 2, higher low 3; signal on the close through 2; the hook = the first bar after the break that fails to make a
  new high, entry on the close through the hook high. Mirror for tops.
- three_line_break: Three Line Break chart (Steve Nison; https://www.tradingview.com/scripts/threelinebreak/): a new
  line in the trend on a close beyond the last line, a reversal only on a close beyond the extreme of the last 3 lines.
- point_figure: 3-box-reversal Point & Figure on closes, box = 0.5 x ATR(50) taken when a column starts (ours: the
  classic fixed box would need the whole history; https://www.luxalgo.com/library/concept/point-and-figure/);
  double-top buy = an X column over the previous X column's top, double-bottom sell = mirror.
- swing_grade: explorer idea TT-s-swing-significance (itsnathanoreilly, TikTok: "not every swing high is equally
  important for a liquidity sweep") - OUR BUILD: grade of each unswept confirmed swing (k 3) = 1 + (it was the highest
  of the 30 bars before it) + (equal highs: another unswept swing within 0.1 ATR) + (it is within 0.1 ATR of the
  previous UTC day's high) + (it has held >= 50 bars); on the bar that sweeps it the highest swept grade is reported.
- gap_type: explorer idea TT-s-fvg-vs-breakaway (Adem OnChain, TikTok) - candle 3 of a gap closes inside candle 2's
  range -> fair value gap (expect a retest), beyond candle 2's extreme -> breakaway gap (expect no retest).
- valid_ob: explorer idea TT-s-obvalid-ictchartist (ICT & SMC CHARTIST, TikTok) - an order block counts only if the
  OB candle swept the previous swing (k 3) on its side, the move away left an FVG, and price closed through the last
  opposite swing (BMS) within 15 bars of the OB. Levels for the three entries: near wick, open (body), 50%."""
import numpy as np

from agents.indicators._core import S, atr, cols, last_pivots, pivots, sma


def darvas_box(df, look=20, hold=3):
    """dict top, bot (the complete box, NaN while forming), brk_up, brk_dn (close through it; the box is then spent)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    top_a, bot_a = np.full(n, np.nan), np.full(n, np.nan)
    up, dn = np.zeros(n, bool), np.zeros(n, bool)
    hh = S(h).rolling(look, min_periods=1).max().to_numpy()
    phase, top, ti, bot = 0, np.nan, -1, np.nan
    for i in range(n):
        if phase == 2:
            top_a[i], bot_a[i] = top, bot
            if c[i] > top:
                up[i], phase = True, 0
            elif c[i] < bot:
                dn[i], phase = True, 0
            continue
        if phase == 1 and h[i] > top:
            phase = 0
        j = i - hold
        if j < 0:
            continue
        if phase == 0:
            if h[j] >= hh[j] and h[j] >= h[j + 1:i + 1].max():
                top, ti, phase = h[j], j, 1
        elif phase == 1 and j > ti:
            if l[j] <= l[ti:j + 1].min() and l[j] < l[j + 1:i + 1].min():
                bot, phase = l[j], 2
                top_a[i], bot_a[i] = top, bot
    return {"top": top_a, "bot": bot_a, "brk_up": up, "brk_dn": dn}


def vsa(df, n=20):
    """dict no_demand, no_supply, upthrust, spring, stopping (bool arrays)."""
    o, h, l, c, v = cols(df)
    sp = h - l
    asp, av = sma(sp, n), sma(v, n)
    pc = np.concatenate([[c[0]], c[:-1]])
    v1 = np.concatenate([[np.inf], v[:-1]])
    v2 = np.concatenate([[np.inf, np.inf], v[:-2]])
    trh, trl = np.maximum(h, pc), np.minimum(l, pc)
    trs = trh - trl
    atrs = sma(trs, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(trs > 0, (c - trl) / trs, 0.5)
    narrow = np.nan_to_num(trs < atrs).astype(bool)
    novol = np.nan_to_num(S(v).rolling(n, min_periods=2).std().to_numpy(), nan=0.0) <= 1e-9 * (np.abs(v) + 1)  # no volume info
    low_v = ((v < v1) & (v < v2)) | novol
    hh = np.concatenate([[np.inf], S(h).rolling(n, min_periods=1).max().to_numpy()[:-1]])
    ll = np.concatenate([[-np.inf], S(l).rolling(n, min_periods=1).min().to_numpy()[:-1]])
    big = np.nan_to_num(v > 1.5 * av).astype(bool) | novol
    out = {"no_demand": (c > pc) & narrow & low_v, "no_supply": (c < pc) & narrow & low_v,
           "upthrust": (h > hh) & (c < hh) & big, "spring": (l < ll) & (c > ll) & big,
           "stopping": (c < pc) & (np.nan_to_num(v > 2 * av).astype(bool) | novol) & (pos > 0.5)}
    return {k: np.asarray(x, bool) for k, x in out.items()}


def ross_123(df, k=3, look=30):
    """dict long_123 / short_123 (close through point 2), p3_l / p3_s (point 3 = stop level), long_hook / short_hook
    (close through the hook), hook_stop_l / hook_stop_s (lowest low / highest high since the hook)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    ll = S(l).rolling(look, min_periods=1).min().to_numpy()
    hh = S(h).rolling(look, min_periods=1).max().to_numpy()
    out = {x: np.zeros(n, bool) for x in ("long_123", "short_123", "long_hook", "short_hook")}
    for x in ("p3_l", "p3_s", "hook_stop_l", "hook_stop_s"):
        out[x] = np.full(n, np.nan)
    piv = []       # (kind, price, idx, was_extreme)
    bull = bear = None      # pending 1-2-3: [p2, p3]
    hk_l = hk_s = None      # [state, hi, hook_hi, low_since]
    for i in range(n):
        if plc[i]:
            piv.append((-1, plv[i], pli[i], plv[i] <= ll[pli[i]]))
        if phc[i]:
            piv.append((1, phv[i], phi[i], phv[i] >= hh[phi[i]]))
        if (plc[i] or phc[i]) and len(piv) >= 3:
            a, b, d = piv[-3], piv[-2], piv[-1]
            if a[0] == -1 and b[0] == 1 and d[0] == -1 and a[3] and d[1] > a[1] and d[1] < b[1]:
                bull = [b[1], d[1]]
            if a[0] == 1 and b[0] == -1 and d[0] == 1 and a[3] and d[1] < a[1] and d[1] > b[1]:
                bear = [b[1], d[1]]
        if bull is not None:
            if c[i] < bull[1]:
                bull = None
            elif c[i] > bull[0]:
                out["long_123"][i], out["p3_l"][i] = True, bull[1]
                hk_l = [0, h[i], np.nan, l[i]]
                bull = None
        if bear is not None:
            if c[i] > bear[1]:
                bear = None
            elif c[i] < bear[0]:
                out["short_123"][i], out["p3_s"][i] = True, bear[1]
                hk_s = [0, l[i], np.nan, h[i]]
                bear = None
        if hk_l is not None and not out["long_123"][i]:
            if hk_l[0] == 0:
                if h[i] > hk_l[1]:
                    hk_l[1] = h[i]
                else:
                    hk_l = [1, hk_l[1], hk_l[1], l[i]]
            else:
                hk_l[3] = min(hk_l[3], l[i])
                if c[i] > hk_l[2]:
                    out["long_hook"][i], out["hook_stop_l"][i] = True, hk_l[3]
                    hk_l = None
        if hk_s is not None and not out["short_123"][i]:
            if hk_s[0] == 0:
                if l[i] < hk_s[1]:
                    hk_s[1] = l[i]
                else:
                    hk_s = [1, hk_s[1], hk_s[1], h[i]]
            else:
                hk_s[3] = max(hk_s[3], h[i])
                if c[i] < hk_s[2]:
                    out["short_hook"][i], out["hook_stop_s"][i] = True, hk_s[3]
                    hk_s = None
    return out


def three_line_break(df, lines=3):
    """dict trend (+1 / -1), flip_up, flip_dn."""
    c = cols(df)[3]
    n = len(c)
    tr = np.zeros(n, int)
    fu, fd = np.zeros(n, bool), np.zeros(n, bool)
    bl = []      # (lo, hi) of each line
    last = c[0]
    t = 0
    for i in range(1, n):
        x = c[i]
        if not bl:
            if x != last:
                bl.append((min(last, x), max(last, x)))
                t = 1 if x > last else -1
                last = x
        else:
            top, bot = bl[-1][1], bl[-1][0]
            rec = bl[-lines:]
            if t == 1:
                if x > top:
                    bl.append((top, x))
                elif x < min(r[0] for r in rec):
                    bl.append((x, bot))
                    t, fd[i] = -1, True
            else:
                if x < bot:
                    bl.append((x, bot))
                elif x > max(r[1] for r in rec):
                    bl.append((top, x))
                    t, fu[i] = 1, True
            if len(bl) > 50:
                bl = bl[-10:]
        tr[i] = t
    return {"trend": tr, "flip_up": fu, "flip_dn": fd}


def point_figure(df, box_mult=0.5, atr_n=50, rev=3):
    """dict col (+1 X / -1 O), dbl_top (X column over the previous X top), dbl_bot, box (current box size),
    col_lo / col_hi (the current column's extreme opposite the move, for stops)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, atr_n)
    col = np.zeros(n, int)
    dt, db = np.zeros(n, bool), np.zeros(n, bool)
    boxa, clo, chi = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    d, ext, start, box = 0, c[0], c[0], np.nan
    prev_x_top = prev_o_bot = np.nan
    fired = False
    for i in range(1, n):
        if not np.isfinite(box) or box <= 0:
            box = box_mult * a[i]
            ext = start = c[i]
            continue
        x = c[i]
        if d >= 0 and x >= ext + box:
            if d == 0:
                d = 1
            ext = ext + np.floor((x - ext) / box) * box
        elif d <= 0 and x <= ext - box:
            if d == 0:
                d = -1
            ext = ext - np.floor((ext - x) / box) * box
        if d == 1 and x <= ext - rev * box:
            prev_x_top, d = ext, -1
            start, box = ext, box_mult * a[i]
            ext = start - np.floor((start - x) / box) * box
            fired = False
        elif d == -1 and x >= ext + rev * box:
            prev_o_bot, d = ext, 1
            start, box = ext, box_mult * a[i]
            ext = start + np.floor((x - start) / box) * box
            fired = False
        if d == 1 and np.isfinite(prev_x_top) and ext > prev_x_top and not fired:
            dt[i], fired = True, True
        if d == -1 and np.isfinite(prev_o_bot) and ext < prev_o_bot and not fired:
            db[i], fired = True, True
        col[i], boxa[i] = d, box
        clo[i] = start if d == 1 else ext
        chi[i] = ext if d == 1 else start
    return {"col": col, "dbl_top": dt, "dbl_bot": db, "box": boxa, "col_lo": clo, "col_hi": chi}


def _prev_day_hl(df, h, l):
    import pandas as pd
    day = pd.Series(df.index.tz_convert("UTC").date)
    g = pd.DataFrame({"d": day, "h": h, "l": l})
    dd = g.groupby("d").agg(h=("h", "max"), l=("l", "min")).shift(1)
    m = g[["d"]].join(dd, on="d")
    return m["h"].to_numpy(float), m["l"].to_numpy(float)


def swing_grade(df, k=3, look=30, eq_tol=0.1, keep=30):
    """dict sweep_hi_grade / sweep_lo_grade (0 = no sweep; else the best grade 1..5 of the swings swept by this bar's
    wick), sweep_hi_lvl / sweep_lo_lvl (that swing's price)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    pdh, pdl = _prev_day_hl(df, h, l)
    hh = S(h).rolling(look, min_periods=1).max().to_numpy()
    ll = S(l).rolling(look, min_periods=1).min().to_numpy()
    gh, gl = np.zeros(n), np.zeros(n)
    lh, lo_ = np.full(n, np.nan), np.full(n, np.nan)
    H, L = [], []     # [price, idx, ext]
    for i in range(n):
        if phc[i]:
            H.append([phv[i], phi[i], phv[i] >= hh[phi[i]]])
            H = H[-keep:]
        if plc[i]:
            L.append([plv[i], pli[i], plv[i] <= ll[pli[i]]])
            L = L[-keep:]
        tol = eq_tol * a[i]
        best, lvl, rest = 0, np.nan, []
        for s in H:
            if h[i] > s[0] and s[1] < i:
                g = 1 + int(s[2]) + int(any(abs(t[0] - s[0]) <= tol for t in H if t is not s))
                g += int(np.isfinite(pdh[i]) and abs(pdh[i] - s[0]) <= tol) + int(i - s[1] >= 50)
                if g > best:
                    best, lvl = g, s[0]
            else:
                rest.append(s)
        H = rest
        gh[i], lh[i] = best, lvl
        best, lvl, rest = 0, np.nan, []
        for s in L:
            if l[i] < s[0] and s[1] < i:
                g = 1 + int(s[2]) + int(any(abs(t[0] - s[0]) <= tol for t in L if t is not s))
                g += int(np.isfinite(pdl[i]) and abs(pdl[i] - s[0]) <= tol) + int(i - s[1] >= 50)
                if g > best:
                    best, lvl = g, s[0]
            else:
                rest.append(s)
        L = rest
        gl[i], lo_[i] = best, lvl
    return {"sweep_hi_grade": gh, "sweep_lo_grade": gl, "sweep_hi_lvl": lh, "sweep_lo_lvl": lo_}


def gap_type(df):
    """dict bull_fvg, bull_brk, bear_fvg, bear_brk (on candle 3), top, bot (the gap zone)."""
    o, h, l, c, v = cols(df)
    h1, l1 = np.concatenate([[np.nan], h[:-1]]), np.concatenate([[np.nan], l[:-1]])
    h2, l2 = np.concatenate([[np.nan] * 2, h[:-2]]), np.concatenate([[np.nan] * 2, l[:-2]])
    bull = np.nan_to_num(l - h2, nan=-1) > 0
    bear = np.nan_to_num(l2 - h, nan=-1) > 0
    bb = bull & (np.nan_to_num(c - h1, nan=-1) > 0)
    sb = bear & (np.nan_to_num(l1 - c, nan=-1) > 0)
    return {"bull_fvg": bull & ~bb, "bull_brk": bb, "bear_fvg": bear & ~sb, "bear_brk": sb,
            "top": np.where(bull, l, np.where(bear, l2, np.nan)), "bot": np.where(bull, h2, np.where(bear, h, np.nan))}


def valid_ob(df, k=3, back=15):
    """dict bull / bear (a valid order block is confirmed on this bar: the BMS close), and for the latest valid block
    ob_*_hi, ob_*_lo, ob_*_open, ob_*_mid (NaN when none is live; a block dies on a close through its far side)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    lph, lpl = last_pivots(h, l, k)
    lph1 = np.concatenate([[np.nan], lph[:-1]])
    lpl1 = np.concatenate([[np.nan], lpl[:-1]])
    out = {"bull": np.zeros(n, bool), "bear": np.zeros(n, bool)}
    for s in ("bull", "bear"):
        for x in ("hi", "lo", "open", "mid"):
            out[f"ob_{s}_{x}"] = np.full(n, np.nan)
    cur_b = cur_s = None
    for i in range(3, n):
        if cur_b is not None and c[i] < cur_b[1]:
            cur_b = None
        if cur_s is not None and c[i] > cur_s[0]:
            cur_s = None
        # bearish: close under the last swing low (BMS down)
        if np.isfinite(lpl1[i]) and c[i] < lpl1[i] and c[i - 1] >= lpl1[i]:
            j0 = max(1, i - back)
            seg = np.arange(j0, i)
            ups = seg[c[seg] > o[seg]]
            if len(ups):
                j = int(ups[np.argmax(h[ups])])
                if h[j] >= h[j:i + 1].max() and np.isfinite(lph1[j]) and h[j] > lph1[j]:
                    t = np.arange(j + 2, i + 1)
                    if len(t) and np.any(h[t] < l[t - 2]):
                        cur_s = (h[j], l[j], o[j], (h[j] + l[j]) / 2)
                        out["bear"][i] = True
        if np.isfinite(lph1[i]) and c[i] > lph1[i] and c[i - 1] <= lph1[i]:
            j0 = max(1, i - back)
            seg = np.arange(j0, i)
            dns = seg[c[seg] < o[seg]]
            if len(dns):
                j = int(dns[np.argmin(l[dns])])
                if l[j] <= l[j:i + 1].min() and np.isfinite(lpl1[j]) and l[j] < lpl1[j]:
                    t = np.arange(j + 2, i + 1)
                    if len(t) and np.any(l[t] > h[t - 2]):
                        cur_b = (h[j], l[j], o[j], (h[j] + l[j]) / 2)
                        out["bull"][i] = True
        if cur_b is not None:
            out["ob_bull_hi"][i], out["ob_bull_lo"][i], out["ob_bull_open"][i], out["ob_bull_mid"][i] = cur_b
        if cur_s is not None:
            out["ob_bear_hi"][i], out["ob_bear_lo"][i], out["ob_bear_open"][i], out["ob_bear_mid"][i] = cur_s
    return out
