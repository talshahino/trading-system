"""The user's strategy, v5: every gap of a strong move is a level (user, 09.11.2026, 24 marked charts).

  gap       a 15-minute fair-value gap (candle k-1's high below candle k+1's low, mirror for sells) made by a strong
            candle k: its body >= GBODY x the average body of the 50 candles before it, the gap >= MIN_GAP x the
            average range of the 30 candles before it. A move in steps gives one gap per step, each one a trade.
  armed     only after the move turned: a 15-minute swing high (2 candles each side) above the gap, made after it.
            If a 15-minute candle goes back into the gap before that, the gap closed during the move: no gap.
  wait      the gap is NOT cancelled when the price goes on to new highs (the peak just moves up); it waits up to
            EXPIRY hours from its making for the price to come back.
  entry     the first 1-minute touch of the gap's near edge after it is armed, in the gap's direction.
  gone      a 15-minute candle closes beyond the gap's far edge before the touch, or the expiry.
  stop      'gap': beyond candle k-1's low (the gap's base); 'sN': beyond the N-th 15m swing low back from the
            entry, each lower than the one after it; always + 4 pips (instruments without pips 0.1 ATR)
  target    the move's peak (the highest high between the gap and the touch) minus 2 pips; the trade is taken only
            when that is at least MIN_RR x the risk
  dominant  optional: the up-move that made the gap (from the lowest low of the 30 candles before it) must be at
            least the down-move into that low (from the highest high of the 30 candles before the low)
Fills and exits walk the 1-minute bars; costs as in own_core. One position at a time per instrument.
Rows have the same fields as own_core.run_symbol, so own15.py's partial-exit grid applies to them.
"""
import numpy as np
import pandas as pd

from agents import own_core as CORE
from agents import rules_agent as R

GBODY = 1.5
MIN_GAP = 0.3
EXPIRY = 24          # hours from the gap's making (on 15m; scales with the timeframe: 96 candles)
EDGE = "near"        # where in the gap to enter: near (first touch), mid, far (its other side)
NEST_ANY = False     # v7: the nested entry gap need not lie inside the big gap (the impulse's first small gap)
TARGET6 = "peak"     # v7: "swing1m" = the nearest 1-minute swing beyond the entry
NEST = 0             # 0 = enter at the big gap's edge; N = at the first N-minute gap inside it (two timeframes down)
TFM = 15             # v5's detection timeframe in minutes (user, 10.11.2026: does it work on 1h / 5m / 1m too?)
MIN_RR = 1.0
STOP = "s2"
DOMINANT = False
PART_FRAC = 0.0      # v5 in the watch bot: 0.3 out at PART_WAY of the way, the rest to break-even
PART_WAY = 0.9
V5_MAX_COST = None   # v5 in the watch bot: skip trades whose spread + commission are more than this share of the risk
DIAG = None          # a list -> (symbol, gap time, dir, why) for every gap that did not become a trade


def _swing_levels(arr, w=2, high=True):
    s = pd.Series(arr)
    ext = (s.rolling(2 * w + 1, center=True).max() if high else s.rolling(2 * w + 1, center=True).min()).to_numpy()
    return np.flatnonzero(arr == ext)


# which setups a trade blocks (10.11.2026): "fill" = only touches while it is open; "arm" = also setups that were already
# waiting when it opened (they are dropped when a trade opens, as the old backtest did, but in the order things happen)
SEL = "fill"


def run_gaps(s, df, cost, stop_at, noent, live=False, variant="gap"):
    o1, h1, l1, c1 = (df[x].to_numpy(float) for x in "ohlc")
    t1 = df.index
    n1 = len(t1)
    m = CORE.resample(df, f"{TFM}min")
    o, h, l, c = (m[x].to_numpy(float) for x in "ohlc")
    t15 = m.index
    n = len(c)
    if n < 100:
        return []
    a15 = R.atr(h, l, c, 14)
    body = np.abs(c - o)
    avgb = pd.Series(body).rolling(50, min_periods=30).mean().shift(1).to_numpy()
    avgr = pd.Series(h - l).rolling(30, min_periods=20).mean().shift(1).to_numpy()
    shi = _swing_levels(h, 2, True)          # 15m swing highs (index of the swing candle; known 2 candles later)
    slo = _swing_levels(l, 2, False)
    pip = CORE.pip_of(s)
    out = []

    def dg(k, d, why):
        if DIAG is not None:
            DIAG.append({"symbol": s, "gap_t": t15[k], "dir": d, "why": why})

    for k in range(31, n - 1):
        for d in (1, -1):
            if d > 0:
                if not l[k + 1] > h[k - 1]:
                    continue
                near, far = l[k + 1], h[k - 1]
            else:
                if not h[k + 1] < l[k - 1]:
                    continue
                near, far = h[k + 1], l[k - 1]
            if not (d * (c[k] - o[k]) > 0 and np.isfinite(avgb[k]) and body[k] >= GBODY * avgb[k]):
                continue                                            # no strong candle: no force
            if not (np.isfinite(avgr[k]) and abs(near - far) >= MIN_GAP * avgr[k]):
                continue
            if DOMINANT:
                w0 = max(0, k - 30)
                if d > 0:
                    lo_i = w0 + int(np.argmin(l[w0:k + 1])); top = max(h[k:k + 2].max(), h[k])
                    b0 = max(0, lo_i - 30); back = h[b0:lo_i + 1].max() - l[lo_i] if lo_i > b0 else 0.0
                    up = top - l[lo_i]
                else:
                    lo_i = w0 + int(np.argmax(h[w0:k + 1])); top = min(l[k:k + 2].min(), l[k])
                    b0 = max(0, lo_i - 30); back = h[lo_i] - l[b0:lo_i + 1].min() if lo_i > b0 else 0.0
                    up = h[lo_i] - top
                if up < back:
                    dg(k, d, "correction"); continue
            # arm: a swing beyond the gap made after it, with no 15m candle back into the gap before it
            end = min(n, k + 2 + int(EXPIRY * 4))  # 96 candles of the detection timeframe
            sw = shi if d > 0 else slo
            arm = None
            for j in range(k + 2, end):
                if (l[j] < near) if d > 0 else (h[j] > near):
                    break                                           # back into the gap during the move
                if j in set_sw(sw) and j + 2 < end:
                    arm = j
                    break
            if arm is None:
                dg(k, d, "closed_in_move" if j < end - 1 else "no_turn"); continue
            if NEST:
                # entry two timeframes down (user, 10.11.2026): the first gap of the NEST-minute chart, in the same
                # direction, made during the 3 candles of the big gap and lying inside it; enter at ITS near edge
                sub = df[(t1 >= t15[max(0, k - (2 if NEST_ANY else 1))]) & (t1 < t15[k + 1] + pd.Timedelta(minutes=TFM))]
                ms = CORE.resample(sub, f"{NEST}min") if NEST > 1 else sub
                sh_, sl_ = ms["h"].to_numpy(float), ms["l"].to_numpy(float)
                lo_g, hi_g = min(near, far), max(near, far)
                e_ = None
                for jj in range(1, len(ms) - 1):
                    if d > 0 and sl_[jj + 1] > sh_[jj - 1]:
                        top, bot = sl_[jj + 1], sh_[jj - 1]
                    elif d < 0 and sh_[jj + 1] < sl_[jj - 1]:
                        top, bot = sl_[jj - 1], sh_[jj + 1]
                    else:
                        continue
                    if NEST_ANY:                                    # v7: the impulse's first small gap, wherever it is
                        e_ = top if d > 0 else bot
                        break
                    if top <= lo_g or bot >= hi_g:
                        continue                                    # not inside the big gap
                    e_ = min(top, hi_g) if d > 0 else max(bot, lo_g)
                    break
                if e_ is None:
                    dg(k, d, "no_small_gap"); continue
                near = e_
            if EDGE == "far":                                   # user, 10.11.2026: enter deeper, at the gap's far edge
                near = far
            elif EDGE == "mid":
                near = (near + far) / 2
            # wait for the touch on the 1-minute bars, from the 15m candle that confirmed the swing
            q0 = t1.searchsorted(t15[arm + 2] + pd.Timedelta(minutes=TFM))
            q1 = min(n1, t1.searchsorted(t15[k] + pd.Timedelta(hours=EXPIRY * TFM / 15)))
            if q0 >= q1:
                dg(k, d, "expired"); continue
            peak = h[k + 1:arm + 1].max() if d > 0 else l[k + 1:arm + 1].min()
            fill, why = None, "expired"
            for q in range(q0, q1):
                if stop_at[q]:
                    why = "break_news"; break
                if (d > 0 and h1[q] > peak) or (d < 0 and l1[q] < peak):
                    peak = h1[q] if d > 0 else l1[q]                # the move went on: the peak moves, keep waiting
                if (d > 0 and l1[q] <= near) or (d < 0 and h1[q] >= near):
                    if noent[q]:
                        why = "news_entry"; break
                    fill = q
                    break
                # gone: a 15-minute close beyond the far edge
                if (t1[q].hour * 60 + t1[q].minute + 1) % TFM == 0 and ((d > 0 and c1[q] < far) or (d < 0 and c1[q] > far)):
                    why = "gap_broken"; break
            else:
                if live and q1 >= n1:
                    out.append({"symbol": s, "variant": variant, "status": "pending", "dir": d, "entry": near, "stop": float((l[k - 1] if d > 0 else h[k - 1]) - d * (4 * pip if pip else 0.1 * a15[k])),
                                "target": float(peak - d * (2 * pip if pip else 0.05 * a15[k])),
                                "setup_t": t15[arm + 2], "leg": int(arm - k), "leg_start": t15[k - 1], "leg_top": t15[arm],
                                "gap_lo": min(near, far), "gap_hi": max(near, far), "_q": n1})
                    continue
            if fill is None:
                dg(k, d, why); continue
            px = min(near, o1[fill]) if d > 0 else max(near, o1[fill])
            buf = 4 * pip if pip else 0.1 * a15[k]
            tbuf = 2 * pip if pip else 0.05 * a15[k]
            kq = t15.searchsorted(t1[fill], side="right") - 1
            if STOP in ("m2", "m3", "m1"):
                # v7 (user's marks, 10.11.2026): beyond the 2nd / 3rd swing back on the 1-minute chart
                nsw = int(STOP[1:])
                lo1 = max(0, fill - 600)
                arr = l1[lo1:fill] if d > 0 else h1[lo1:fill]
                sw1 = _swing_levels(arr, 3, d < 0)
                lv, last = [], px
                for i in sorted(sw1, reverse=True):
                    if i + 3 >= len(arr):
                        continue
                    v = arr[i]
                    if d * (last - v) > 0:
                        lv.append(v); last = v
                        if len(lv) == nsw:
                            break
                if len(lv) < nsw:
                    dg(k, d, "no_swing_stop"); continue
                base = lv[-1]
            elif STOP == "gap":
                base = l[k - 1] if d > 0 else h[k - 1]
            elif STOP == "top":                                 # "a bit further": beyond the start of the impulse
                base = l[max(0, k - 5):k + 1].min() if d > 0 else h[max(0, k - 5):k + 1].max()
            else:
                nsw = int(STOP[1:])
                lv, last = [], px
                for i in sorted((slo if d > 0 else shi), reverse=True):
                    if i + 2 > kq:
                        continue
                    if i < kq - 480:
                        break
                    v = l[i] if d > 0 else h[i]
                    if d * (last - v) > 0:
                        lv.append(v); last = v
                        if len(lv) == nsw:
                            break
                if len(lv) < nsw:
                    dg(k, d, "no_swing_stop"); continue
                base = lv[-1]
            stop = base - d * buf
            target = peak - d * tbuf
            if TARGET6 == "swing1m":
                # v7: the nearest 1-minute swing beyond the entry, made between the impulse and the touch
                a0 = t1.searchsorted(t15[k - 1])
                arr = h1[a0:fill] if d > 0 else l1[a0:fill]
                cand = [arr[i] for i in _swing_levels(arr, 3, d > 0) if i + 3 < len(arr) and d * (arr[i] - px) > 0] if len(arr) > 7 else []
                if cand:
                    target = (min(cand) if d > 0 else max(cand)) - d * tbuf
            risk = d * (px - stop)
            if risk <= 0 or d * (target - px) < MIN_RR * risk:
                dg(k, d, "rr")
                if live:   # touched but not taken: show it, with the reason (user, 10.11.2026: "why did it not enter?")
                    out.append({"symbol": s, "variant": variant, "status": "cancelled", "why": "rr", "dir": d, "entry": float(px), "stop": float(stop),
                                "target": float(target), "setup_t": t15[arm + 2], "touch_t": t1[fill], "leg": int(arm - k), "leg_start": t15[k - 1],
                                "leg_top": t15[arm], "gap_lo": min(near, far), "gap_hi": max(near, far), "rr": round(d * (target - px) / risk, 2) if risk > 0 else None,
                                "cost_r": round(cost[fill] / risk, 3) if risk > 0 else None, "_q": fill})
                continue
            if V5_MAX_COST is not None and cost[fill] / risk > V5_MAX_COST:
                dg(k, d, "cost")
                if live:   # touched but not taken: show it, with the reason (user, 10.11.2026: "why did it not enter?")
                    out.append({"symbol": s, "variant": variant, "status": "cancelled", "why": "cost", "dir": d, "entry": float(px), "stop": float(stop),
                                "target": float(target), "setup_t": t15[arm + 2], "touch_t": t1[fill], "leg": int(arm - k), "leg_start": t15[k - 1],
                                "leg_top": t15[arm], "gap_lo": min(near, far), "gap_hi": max(near, far), "rr": round(d * (target - px) / risk, 2) if risk > 0 else None,
                                "cost_r": round(cost[fill] / risk, 3) if risk > 0 else None, "_q": fill})
                continue
            ex_px = ex_q = rsn = None
            # PART (user, 10.11.2026, approved): PART_FRAC of the position out at PART_WAY of the way to the target,
            # the rest's stop to the entry (break-even)
            lvl = px + PART_WAY * (target - px) if PART_FRAC else None
            part_r, cur_stop = 0.0, stop
            for q in range(fill, n1):
                if (d > 0 and l1[q] <= cur_stop) or (d < 0 and h1[q] >= cur_stop):
                    ex_px, ex_q = ((min(cur_stop, o1[q]) if d > 0 else max(cur_stop, o1[q])) if q > fill else cur_stop), q
                    rsn = "breakeven" if cur_stop != stop else "stop"
                    break
                if q > fill and ((d > 0 and h1[q] >= target) or (d < 0 and l1[q] <= target)):
                    if lvl is not None and cur_stop == stop:
                        part_r = PART_FRAC * d * (lvl - px) / risk
                    ex_px, ex_q, rsn = (max(target, o1[q]) if d > 0 else min(target, o1[q])), q, "target"
                    break
                if q > fill and stop_at[q]:
                    ex_px, ex_q, rsn = c1[q], q, "forced"
                    break
                if lvl is not None and cur_stop == stop and q > fill and ((d > 0 and h1[q] >= lvl) or (d < 0 and l1[q] <= lvl)):
                    part_r, cur_stop = PART_FRAC * d * (lvl - px) / risk, px
            rest = 1.0 - (PART_FRAC if part_r else 0.0)
            rec = {"symbol": s, "variant": variant, "dir": d, "entry": px, "stop": stop, "target": target,
                   "rr": round(d * (target - px) / risk, 2), "cost_r": round(cost[fill] / risk, 3), "time": t1[fill],
                   "leg": int(arm - k), "leg_start": t15[k - 1], "leg_top": t15[arm], "gap_lo": min(near, far), "gap_hi": max(near, far)}
            if part_r:
                rec["part"] = round(part_r, 3)
            rec["_fill"], rec["_k"], rec["_q0"] = fill, k, q0
            if ex_q is None:
                rec.update(status="open", stop_now=cur_stop, r_now=round(part_r + rest * d * (c1[-1] - px) / risk, 3), _exq=n1)
            else:
                rec.update(status="closed", exit_time=t1[ex_q], exit=ex_px, reason=rsn, _exq=ex_q,
                           r=round(part_r + rest * d * (ex_px - px) / risk - cost[fill] / risk, 4))
            out.append(rec)
    # one trade at a time, in the order the trades happen (10.11.2026): the first touch is taken, and nothing else
    # while it is open. (Before, the gaps were taken in the order they formed, so a later touch of an older gap could
    # replace a trade that had really happened first - impossible live, and live showed overlapping extra trades.)
    trades = sorted([x for x in out if "_fill" in x], key=lambda x: (x["_fill"], x["_k"]))
    taken, last = [], -1
    for x in trades:
        if (x["_q0"] if SEL == "arm" else x["_fill"]) > last:
            taken.append(x); last = x["_exq"]
            if x["status"] == "closed":
                dg(x["_k"], x["dir"], "traded")
        else:
            dg(x["_k"], x["dir"], "busy")
    spans = [(x["_fill"], x["_exq"]) for x in taken]
    res = []
    for x in out:
        if "_fill" in x:
            if not any(x is y for y in taken):
                continue
            if x["status"] == "open" and not live:
                continue
        elif any(a_ <= x["_q"] <= b_ for a_, b_ in spans) or (x["status"] == "pending" and last >= n1):
            continue                                  # touched (or waiting) while a trade was open: never a candidate
        res.append({k_: v_ for k_, v_ in x.items() if not k_.startswith("_")})
    return res


_SW_CACHE = {}


def set_sw(sw):
    key = id(sw)
    if key not in _SW_CACHE:
        _SW_CACHE.clear()
        _SW_CACHE[key] = set(int(i) for i in sw)
    return _SW_CACHE[key]


# ---------------- v6 (user, 10.11.2026, marked v5 examples) ----------------
# 1. only the FIRST gap of a move (a move starts at the lowest low since the last confirmed 15m swing high; a staircase
#    with a turn between the steps gives a new first gap per step)
# 2. the FIRST touch is the entry (a wick counts) - one shot: after a touch the gap is dead; no waiting for a turn
# 3. the move must be the big one, checked at the touch: from its origin to the extreme reached so far >= the move into
#    the origin (from the highest high of the 30 candles before it); else it is a correction and the gap is dead
# 4. trades whose spread + commission are more than MAX_COST of the risk are skipped
MAX_COST = 0.10
MOVE_W = 2           # the swing size (candles each side) that ends a move; bigger = small pullbacks inside a move don't
                     # restart it (user, 10.11.2026: "you did not take the first gap of the move")
ARM6 = False         # test: the first touch counts only after the move turned (a 15m swing beyond the gap confirmed)
STOP6 = "gap"        # 'gap' (beyond the gap's base candle), 'origin' (beyond the move's start), 'sN'


def run_gaps6(s, df, cost, stop_at, noent, live=False, variant="g6"):
    o1, h1, l1, c1 = (df[x].to_numpy(float) for x in "ohlc")
    t1 = df.index
    n1 = len(t1)
    m = CORE.resample(df, "15min")
    o, h, l, c = (m[x].to_numpy(float) for x in "ohlc")
    t15 = m.index
    n = len(c)
    if n < 100:
        return []
    a15 = R.atr(h, l, c, 14)
    body = np.abs(c - o)
    avgb = pd.Series(body).rolling(50, min_periods=30).mean().shift(1).to_numpy()
    avgr = pd.Series(h - l).rolling(30, min_periods=20).mean().shift(1).to_numpy()
    shi = _swing_levels(h, 2, True)
    slo = _swing_levels(l, 2, False)
    bshi, bslo = _swing_levels(h, MOVE_W, True), _swing_levels(l, MOVE_W, False)
    pip = CORE.pip_of(s)
    # qualifying gaps (center candle k): +1 up, -1 down, 0 none
    g = np.zeros(n, int)
    for k in range(31, n - 1):
        for d in (1, -1):
            ok = (l[k + 1] > h[k - 1]) if d > 0 else (h[k + 1] < l[k - 1])
            if not ok:
                continue
            near, far = (l[k + 1], h[k - 1]) if d > 0 else (h[k + 1], l[k - 1])
            if d * (c[k] - o[k]) > 0 and np.isfinite(avgb[k]) and body[k] >= GBODY * avgb[k] \
                    and np.isfinite(avgr[k]) and abs(near - far) >= MIN_GAP * avgr[k]:
                g[k] = d
    out = []
    busy_until = -1

    def dg(k, d, why):
        if DIAG is not None:
            DIAG.append({"symbol": s, "gap_t": t15[k], "dir": d, "why": why})

    for k in np.flatnonzero(g):
        d = int(g[k])
        near, far = (l[k + 1], h[k - 1]) if d > 0 else (h[k + 1], l[k - 1])
        # the move's origin: the extreme since the last opposite swing confirmed before the gap
        opp = (shi if d > 0 else slo) if MOVE_W == 2 else (bshi if d > 0 else bslo)
        prev = opp[opp + MOVE_W <= k]
        a0 = int(prev[-1]) if len(prev) else max(0, k - 30)
        a0 = max(a0, k - 96)
        oi = a0 + int(np.argmin(l[a0:k + 1]) if d > 0 else np.argmax(h[a0:k + 1]))
        if np.any(g[oi + 1:k] == d):
            dg(k, d, "not_first"); continue
        origin = l[oi] if d > 0 else h[oi]
        b0 = max(0, oi - 30)
        back = (h[b0:oi + 1].max() - origin) if d > 0 else (origin - l[b0:oi + 1].min())
        q0 = t1.searchsorted(t15[k + 1] + pd.Timedelta(minutes=15))
        q1 = min(n1, t1.searchsorted(t15[k] + pd.Timedelta(hours=EXPIRY)))
        if q0 >= q1:
            continue
        peak = h[k + 1] if d > 0 else l[k + 1]
        fill, why = None, "expired"
        for q in range(q0, q1):
            if stop_at[q]:
                why = "break_news"; break
            if (d > 0 and l1[q] <= near) or (d < 0 and h1[q] >= near):
                fill = q
                break
            peak = max(peak, h1[q]) if d > 0 else min(peak, l1[q])
        if fill is None:
            if live and q1 >= n1 and why == "expired":
                out.append({"symbol": s, "variant": variant, "status": "pending", "dir": d, "entry": near, "stop": None,
                            "setup_t": t15[k + 1], "leg": int(k - oi), "leg_start": t15[oi], "leg_top": t15[k + 1],
                            "gap_lo": min(near, far), "gap_hi": max(near, far)})
            else:
                dg(k, d, why)
            continue
        if noent[fill]:
            dg(k, d, "news_entry"); continue
        if ARM6:
            swk = shi if d > 0 else slo
            after = swk[(swk > k + 1) & (swk + 2 < n)]
            if not len(after) or t1[fill] < t15[int(after[0]) + 2] + pd.Timedelta(minutes=15):
                dg(k, d, "before_turn"); continue
        if DOMINANT and d * (peak - origin) < back:
            dg(k, d, "correction"); continue
        if fill <= busy_until:
            dg(k, d, "busy"); continue
        px = min(near, o1[fill]) if d > 0 else max(near, o1[fill])
        buf = 4 * pip if pip else 0.1 * a15[k]
        tbuf = 2 * pip if pip else 0.05 * a15[k]
        kq = t15.searchsorted(t1[fill], side="right") - 1
        if STOP6 == "gap":
            base = l[k - 1] if d > 0 else h[k - 1]
        elif STOP6 == "origin":
            base = origin
        else:
            nsw = int(STOP6[1:])
            lv, last = [], px
            for i in sorted((slo if d > 0 else shi), reverse=True):
                if i + 2 > kq:
                    continue
                if i < kq - 480:
                    break
                v = l[i] if d > 0 else h[i]
                if d * (last - v) > 0:
                    lv.append(v); last = v
                    if len(lv) == nsw:
                        break
            if len(lv) < nsw:
                dg(k, d, "no_swing_stop"); continue
            base = lv[-1]
        stop = base - d * buf
        target = peak - d * tbuf
        risk = d * (px - stop)
        if risk <= 0 or d * (target - px) < MIN_RR * risk:
            dg(k, d, "rr"); continue
        if cost[fill] / risk > MAX_COST:
            dg(k, d, "cost"); continue
        ex_px = ex_q = rsn = None
        for q in range(fill, n1):
            if (d > 0 and l1[q] <= stop) or (d < 0 and h1[q] >= stop):
                ex_px, ex_q, rsn = ((min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if q > fill else stop), q, "stop"
                break
            if q > fill and ((d > 0 and h1[q] >= target) or (d < 0 and l1[q] <= target)):
                ex_px, ex_q, rsn = (max(target, o1[q]) if d > 0 else min(target, o1[q])), q, "target"
                break
            if q > fill and stop_at[q]:
                ex_px, ex_q, rsn = c1[q], q, "forced"
                break
        rec = {"symbol": s, "variant": variant, "dir": d, "entry": px, "stop": stop, "target": target,
               "rr": round(d * (target - px) / risk, 2), "cost_r": round(cost[fill] / risk, 3), "time": t1[fill],
               "leg": int(k - oi), "leg_start": t15[oi], "leg_top": t15[k + 1], "gap_lo": min(near, far), "gap_hi": max(near, far)}
        if ex_q is None:
            if live:
                rec.update(status="open", r_now=round(d * (c1[-1] - px) / risk, 3))
                out.append(rec)
            continue
        dg(k, d, "traded")
        busy_until = ex_q
        rec.update(status="closed", exit_time=t1[ex_q], exit=ex_px, reason=rsn,
                   r=round(d * (ex_px - px) / risk - cost[fill] / risk, 4))
        out.append(rec)
    return out
