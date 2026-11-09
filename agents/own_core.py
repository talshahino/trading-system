"""The user's own impulse strategy, shared by the backtest (own15.py) and the live watch bot (live/watch_own.py),
so the bot runs exactly what was tested. Only numpy / pandas / the bots' own agents modules (installed on the bots'
computer). Rules: see own15.py.
"""
import numpy as np
import pandas as pd

from agents import ftmo_universe as U
from agents import rules_agent as R

# grid (the user, 06.11.2026): target at the 1st / 2nd / 3rd correction beyond the entry (then the impulse's top), stop at the
# 1st / 2nd / 3rd 15m swing beyond the entry or at the impulse's origin - every combination from the same entries
VARIANTS = [(2.0, False, '1m', 'grid')]
PIP = {"JPY": 0.01}
BODY_X = 2.5        # the leg's average body vs the 50 candles before it
MIN_GAP = 0.33      # the gap's size vs the average range of the 30 candles before the leg (user: tiny gaps are no gaps)


def pip_of(s):
    if len(s) == 6 and s.isalpha() and s[3:] in ("USD", "JPY", "CHF", "CAD", "GBP", "NZD", "AUD", "EUR"):
        return 0.01 if s.endswith("JPY") else 0.0001
    return None


def resample(df, rule):
    return df.resample(rule, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()


def swings(h, l, w=2):
    """fractal highs / lows (confirmed w bars later): lists of (confirm index, price)"""
    n = len(h)
    hi, lo = [], []
    for i in range(w, n - w):
        if h[i] == h[i - w:i + w + 1].max():
            hi.append((i + w, h[i]))
        if l[i] == l[i - w:i + w + 1].min():
            lo.append((i + w, l[i]))
    return hi, lo


def comm_px(s, px):
    """FTMO's round-trip commission in price units"""
    spec = U.spec(s)
    if spec["commissionType"] == "flat_USD":
        return px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
    return px * 2 * float(spec["commission"]) / 100.0


def run_symbol(s, df, cost, stop_at, noent, variants, live=False):
    """live=True (the watch bot): also setups still waiting for the touch (status 'pending') and trades not yet closed
    ('open'), so the last minutes of data are reported as they stand."""
    o1, h1, l1, c1 = (df[x].to_numpy(float) for x in "ohlc")
    t1 = df.index
    n1 = len(t1)
    m = resample(df, "15min")
    o, h, l, c = (m[x].to_numpy(float) for x in "ohlc")
    t15 = m.index
    n = len(c)
    a15 = R.atr(h, l, c, 14)
    body = np.abs(c - o)
    avgb = pd.Series(body).rolling(50, min_periods=30).mean().to_numpy()
    h60 = resample(df, "60min")
    e20, e50 = R.ema(h60["c"].to_numpy(), 20), R.ema(h60["c"].to_numpy(), 50)
    tr60 = pd.Series(np.sign(e20 - e50), index=h60.index + pd.Timedelta(minutes=60))
    trend = tr60.reindex(t15 + pd.Timedelta(minutes=15), method="ffill").to_numpy()
    sh, sl_ = swings(h, l, 2)
    SH = {i - 2: p for i, p in sh}            # swing at candle index (confirmed 2 candles later)
    SL = {i - 2: p for i, p in sl_}
    sh_idx = np.array(sorted(SH)); sl_idx = np.array(sorted(SL))
    pip = pip_of(s)
    # typical move over n candles (5 days ~ 480 candles before), for n = 8..30
    typ = {L: pd.Series(np.abs(c - np.r_[np.full(L, np.nan), c[:-L]])).rolling(480, min_periods=200).median().shift(1).to_numpy()
           for L in range(8, 31)}
    out = []
    # 1-minute swing highs / lows (5 minutes each side), for the target on the 1-minute chart
    w1 = 5
    w10 = 10
    sh10 = np.flatnonzero(h1 == pd.Series(h1).rolling(2 * w10 + 1, center=True).max().to_numpy())
    sl10 = np.flatnonzero(l1 == pd.Series(l1).rolling(2 * w10 + 1, center=True).min().to_numpy())
    sh1 = np.flatnonzero(h1 == pd.Series(h1).rolling(2 * w1 + 1, center=True).max().to_numpy())
    sl1 = np.flatnonzero(l1 == pd.Series(l1).rolling(2 * w1 + 1, center=True).min().to_numpy())
    for K, tfil, emode, tmode in variants:
        busy_until = -1
        used_peaks = set()
        setups = []
        for d in (1, -1):
            peaks = sh_idx if d > 0 else sl_idx
            for pk in peaks:
                conf = pk + 2                                     # the swing is known 2 candles later
                if conf >= n or pk in used_peaks:
                    continue
                # origin: the most extreme opposite swing 8-30 candles before the peak
                lo_i = None
                cands = (sl_idx if d > 0 else sh_idx)
                cands = cands[(cands >= pk - 30) & (cands <= pk - 8)]
                if not len(cands):
                    continue
                if d > 0:
                    lo_i = cands[np.argmin([SL[i] for i in cands])]
                    origin, peak = SL[lo_i], SH[pk]
                else:
                    lo_i = cands[np.argmax([SH[i] for i in cands])]
                    origin, peak = SH[lo_i], SL[pk]
                L = pk - lo_i
                if not (8 <= L <= 30):
                    continue
                size = d * (peak - origin)
                ty = typ[L][pk]
                if not (np.isfinite(ty) and ty > 0 and size >= K * ty):
                    continue
                ab = avgb[lo_i]
                if not (np.isfinite(ab) and ab > 0 and body[lo_i:pk + 1].mean() >= BODY_X * ab):
                    continue
                if d > 0 and (h[lo_i:pk + 1].max() > peak or l[lo_i:pk + 1].min() < origin):
                    continue
                if d < 0 and (l[lo_i:pk + 1].min() < peak or h[lo_i:pk + 1].max() > origin):
                    continue
                if tfil and trend[pk] != d:
                    continue
                # the first fair-value gap of the leg (in time)
                gap = None
                ar = (h - l)[max(0, lo_i - 30):lo_i].mean() if lo_i > 0 else np.nan
                for k in range(lo_i + 1, pk):
                    if d > 0 and l[k + 1] > h[k - 1] and l[k + 1] - h[k - 1] >= MIN_GAP * ar:
                        gap, gk = (l[k + 1], h[k - 1]), k; break
                    if d < 0 and h[k + 1] < l[k - 1] and l[k - 1] - h[k + 1] >= MIN_GAP * ar:
                        gap, gk = (h[k + 1], l[k - 1]), k; break
                if gap is None:
                    continue
                entry = gap[0]                                   # near edge: touched first on the way back
                if emode == "1m":
                    # the user (06.11.2026): inside the 15m gap, the 1-minute gap of the same move; enter at its near edge
                    a_ = t1.searchsorted(t15[gk - 1]); b_ = t1.searchsorted(t15[gk + 1] + pd.Timedelta(minutes=15))
                    zlo, zhi = min(gap), max(gap)
                    best1, e1 = 0.0, None
                    for jj in range(a_ + 1, b_ - 1):
                        if d > 0 and l1[jj + 1] > h1[jj - 1]:
                            lo_, hi_ = h1[jj - 1], l1[jj + 1]
                        elif d < 0 and h1[jj + 1] < l1[jj - 1]:
                            lo_, hi_ = h1[jj + 1], l1[jj - 1]
                        else:
                            continue
                        ov = min(hi_, zhi) - max(lo_, zlo)
                        if ov > 0 and hi_ - lo_ > best1:
                            best1, e1 = hi_ - lo_, (hi_ if d > 0 else lo_)
                    if e1 is None:
                        continue
                    entry = e1
                if emode in ("gc", "go", "fc", "fo"):
                    # research (user, 09.11.2026): enter deeper, at the body of the candle that leads the move -
                    # g = the impulse candle of the first gap, f = the leg's first candle; c = its close, o = its open
                    kk = gk if emode[0] == "g" else lo_i + 1
                    entry = c[kk] if emode[1] == "c" else o[kk]
                    if d * (entry - origin) <= 0:
                        continue
                buf = 4 * pip if pip else 0.1 * a15[pk]
                tbuf = 2 * pip if pip else 0.05 * a15[pk]
                stop = origin - d * buf
                used_peaks.add(pk)
                setups.append((conf, d, pk, lo_i, peak, entry, stop, tbuf, gap, L, tmode))
        for conf, d, pk, lo_i, peak, entry, stop, tbuf, gap, L, tmode in sorted(setups, key=lambda x: x[0]):
            if True:
                # walk the 1-minute bars from the confirmation for 8 hours
                s0 = t1.searchsorted(t15[conf])
                if s0 >= n1 or s0 <= busy_until:
                    continue
                s1 = min(n1, t1.searchsorted(t15[conf] + pd.Timedelta(hours=8)))
                fill = None
                for q in range(s0, s1):
                    if stop_at[q]:
                        break
                    if (d > 0 and h1[q] > peak) or (d < 0 and l1[q] < peak):
                        break                                    # the leg went on: setup gone
                    if (d > 0 and l1[q] <= stop) or (d < 0 and h1[q] >= stop):
                        break
                    if (d > 0 and l1[q] <= entry) or (d < 0 and h1[q] >= entry):
                        if noent[q]:
                            break
                        fill = q
                        break
                else:
                    if live and s1 >= n1:                         # still waiting for the touch at the end of the data
                        out.append({"symbol": s, "variant": f"{tmode}", "status": "pending", "dir": d, "entry": entry, "stop": stop,
                                    "setup_t": t15[conf], "expires_t": t15[conf] + pd.Timedelta(hours=8), "leg": int(L),
                                    "leg_start": t15[lo_i], "leg_top": t15[pk], "peak": peak, "gap_lo": min(gap), "gap_hi": max(gap)})
                if fill is None:
                    continue
                px = min(entry, o1[fill]) if d > 0 else max(entry, o1[fill])
                # target: the nearest 15m swing high above the entry known at the fill (after the origin)
                tq = t15.searchsorted(t1[fill], side="right") - 1        # 15m candle of the fill
                known = [i for i in (sh_idx if d > 0 else sl_idx) if lo_i <= i and i + 2 <= tq]
                lv = [(SH if d > 0 else SL)[i] for i in known]
                lv = [x for x in lv if d * (x - px) > 0]
                if tmode == "15p":
                    # the user (06.11.2026): the first high the price meets going up after the entry - a real 15m swing
                    # (the price turned down from it by at least half an ATR), not the impulse's top, not a 1-minute wiggle;
                    # among the 15m swings since the origin known at the fill, the lowest above the entry
                    lv2 = []
                    for i in (sh_idx if d > 0 else sl_idx):
                        if not (lo_i <= i and i + 2 <= tq):
                            continue
                        pv = (SH if d > 0 else SL)[i]
                        nxt = l[i + 1:i + 3].min() if d > 0 else h[i + 1:i + 3].max()
                        if d * (pv - px) > 0 and d * (pv - nxt) >= 0.5 * a15[i]:
                            lv2.append(pv)
                    lv = lv2
                if tmode in ("struct", "structx"):
                    q0 = t1.searchsorted(t15[pk])
                    sw = sh1 if d > 0 else sl1
                    arr = h1 if d > 0 else l1
                    sw = [i for i in sw if q0 <= i and i + w1 <= fill and d * (arr[i] - px) > 0]
                    lv = []
                    for i in reversed(sw):
                        if d > 0:
                            ok = l1[i:fill + 1].min() <= l1[max(0, i - 30):i].min() - 0.4 * a15[pk]
                        else:
                            ok = h1[i:fill + 1].max() >= h1[max(0, i - 30):i].max() + 0.4 * a15[pk]
                        if ok:
                            lv = [arr[i]]
                            break
                if tmode in ("last5", "last5x"):
                    # the last 1-minute swing before the fill (5 minutes each side, confirmed by the fill) beyond the entry:
                    # for a buy the last swing high above the entry, for a sell the last swing low below it
                    sw = sh1 if d > 0 else sl1
                    sw = sw[(sw >= lo_i_q) & (sw + w1 <= fill)] if (lo_i_q := t1.searchsorted(t15[lo_i])) >= 0 else sw
                    arr = h1 if d > 0 else l1
                    cand = [i for i in sw if d * (arr[i] - px) > 0]
                    lv = [arr[cand[-1]]] if cand else []
                if tmode in ("n10", "n10x"):
                    # the user (06.11.2026, two marked charts): the nearest 1-minute high (10 minutes each side lower)
                    # above the entry, made after the impulse's top and known at the fill, at least half the stop away
                    q0 = t1.searchsorted(t15[pk])
                    sw = sh10 if d > 0 else sl10
                    sw = sw[(sw >= q0) & (sw + w10 <= fill)]
                    rk0 = d * (px - stop)
                    lv = [(h1 if d > 0 else l1)[i] for i in sw]
                    lv = [x for x in lv if d * (x - px) >= 0.5 * rk0]
                if tmode == "1m":
                    # nearest 1-minute swing high (low) above (below) the entry, made after the leg's top, known at the fill
                    q0 = t1.searchsorted(t15[pk])
                    sw = sh1 if d > 0 else sl1
                    sw = sw[(sw >= q0) & (sw + w1 <= fill)]
                    lv1 = [(h1 if d > 0 else l1)[i] for i in sw]
                    lv1 = [x for x in lv1 if d * (x - px) > 0]
                    if lv1:
                        lv = lv1
                if tmode == "grid":
                    q0 = t1.searchsorted(t15[pk])
                    arr = h1 if d > 0 else l1
                    sw = [i for i in (sh1 if d > 0 else sl1) if q0 <= i and i + w1 <= fill and d * (arr[i] - px) > 0]
                    corr = []
                    for i in reversed(sw):
                        if d > 0:
                            ok = l1[i:fill + 1].min() <= l1[max(0, i - 30):i].min() - 0.4 * a15[pk]
                        else:
                            ok = h1[i:fill + 1].max() >= h1[max(0, i - 30):i].max() + 0.4 * a15[pk]
                        if ok:
                            corr.append(arr[i])
                    corr = sorted(set(corr), key=lambda x: d * (x - px))
                    if d * (peak - px) > 0 and (not corr or d * (peak - corr[-1]) > 0):
                        corr.append(peak)                               # the impulse's top is the last high
                    buf = 4 * pip if pip else 0.1 * a15[pk]
                    sws = [(SL if d > 0 else SH)[i] for i in (sl_idx if d > 0 else sh_idx) if lo_i <= i and i + 2 <= tq]
                    sws = sorted({x for x in sws if d * (px - x) > 0}, key=lambda x: d * (px - x))
                    stops = {f"s{k + 1}": sws[k] - d * buf for k in range(min(3, len(sws)))}
                    stops["so"] = stop
                    ends = []
                    for tn in range(min(3, len(corr))):
                        tgt = corr[tn] - d * tbuf
                        for sname, stp in stops.items():
                            rk = d * (px - stp)
                            if rk <= 0 or d * (tgt - px) <= 0:
                                continue
                            ex_px = ex_q = why = None
                            for q in range(fill, n1):
                                if (d > 0 and l1[q] <= stp) or (d < 0 and h1[q] >= stp):
                                    ex_px, ex_q, why = ((min(stp, o1[q]) if d > 0 else max(stp, o1[q])) if q > fill else stp), q, "stop"
                                    break
                                if q > fill and ((d > 0 and h1[q] >= tgt) or (d < 0 and l1[q] <= tgt)):
                                    ex_px, ex_q, why = (max(tgt, o1[q]) if d > 0 else min(tgt, o1[q])), q, "target"
                                    break
                                if stop_at[q]:
                                    ex_px, ex_q, why = c1[q], q, "forced"
                                    break
                            if ex_px is None:
                                continue
                            ends.append(ex_q)
                            out.append({"symbol": s, "variant": f"t{tn + 1}|{sname}", "time": t1[fill], "exit_time": t1[ex_q],
                                        "dir": d, "entry": px, "stop": stp, "target": tgt, "exit": ex_px, "reason": why,
                                        "rr": round(d * (tgt - px) / rk, 2), "cost_r": round(cost[fill] / rk, 3),
                                        "r": round((d * (ex_px - px) - cost[fill]) / rk, 4), "leg": int(L),
                                        "leg_start": t15[lo_i], "leg_top": t15[pk], "gap_lo": min(gap), "gap_hi": max(gap)})
                    if ends:
                        busy_until = min(ends)                              # one setup at a time per instrument
                    continue
                if not lv:
                    continue
                target = (min(lv) if d > 0 else max(lv)) - d * tbuf
                risk = d * (px - stop)
                if risk <= 0 or (tmode not in ("n10x", "last5x", "structx") and d * (target - px) < risk) or d * (target - px) <= 0:
                    continue
                ex_px = ex_q = why = None
                for q in range(fill, n1):
                    if (d > 0 and l1[q] <= stop) or (d < 0 and h1[q] >= stop):
                        ex_px, ex_q, why = ((min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if q > fill else stop), q, "stop"
                        break
                    if q > fill and ((d > 0 and h1[q] >= target) or (d < 0 and l1[q] <= target)):
                        ex_px, ex_q, why = (max(target, o1[q]) if d > 0 else min(target, o1[q])), q, "target"
                        break
                    if stop_at[q]:
                        ex_px, ex_q, why = c1[q], q, "forced"
                        break
                if ex_px is None:
                    if live:
                        busy_until = n1
                        out.append({"symbol": s, "variant": f"{tmode}", "status": "open", "time": t1[fill], "dir": d, "entry": px,
                                    "stop": stop, "target": target, "rr": round(d * (target - px) / risk, 2),
                                    "cost_r": round(cost[fill] / risk, 3), "r_now": round((d * (c1[-1] - px) - cost[fill]) / risk, 4),
                                    "leg": int(L), "leg_start": t15[lo_i], "leg_top": t15[pk], "peak": peak,
                                    "gap_lo": min(gap), "gap_hi": max(gap)})
                    continue
                busy_until = ex_q
                out.append({"symbol": s, "variant": f"K{K:g}|e{emode}|t{tmode}", "status": "closed", "time": t1[fill], "exit_time": t1[ex_q],
                            "dir": d, "entry": px, "stop": stop, "target": target, "exit": ex_px, "reason": why,
                            "rr": round(d * (target - px) / risk, 2), "cost_r": round(cost[fill] / risk, 3),
                            "r": round((d * (ex_px - px) - cost[fill]) / risk, 4), "leg": int(L),
                            "leg_start": t15[lo_i], "leg_top": t15[pk], "gap_lo": min(gap), "gap_hi": max(gap)})
    return out
