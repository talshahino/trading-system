"""The user's own strategy, version 3 (user, 06.11.2026, drawn and marked on charts): identified on the 15-minute chart;
entry on the 1-minute gap inside the 15-minute gap (variant), target the nearest 1-minute swing high (variant).

  impulse   a leg of 8-30 fifteen-minute candles from a swing low to a swing high (mirror for sells) that is
            clearly bigger than the recent history: its size >= K x the typical move over the same number of
            candles in the previous ~5 days (median of |close[t] - close[t-n]|), and its average body >= 1.5 x
            the average body of the 50 candles before it ("a mass of buyers")
  entry     the FIRST fair-value gap of the leg in time (candle k-1's high below candle k+1's low); when the price
            comes back and TOUCHES the gap's near edge -> buy (the impulse's direction)
  stop      the swing low the leg started from, minus 4 pips (instruments without pips: 0.1 x ATR)
  target    the nearest swing high above the entry (usually a lower high made during the pullback), minus a
            little (2 pips / 0.05 x ATR) to close safely
  rules     risk:reward at least 1:1; the setup lives 8 hours after the leg's top is confirmed; it is gone if the
            price goes above the leg's top first, or below the stop; FTMO restricted news: no entry 2 minutes
            either side, positions closed 2 minutes before; positions closed before market breaks; one position
            at a time per instrument
  filter    variant: only when the 1-hour trend (EMA20 vs EMA50) agrees with the impulse, or not filtered
Fills and exits are walked on the 1-minute bars (to know which came first); costs = the real spread at the fill
minute + FTMO commission. TRAIN 2024-01-01..2025-04-30, VALID after.

  python own15.py --symbols EURUSD,GBPUSD
Writes output/own15_<tag>.csv.gz.
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
import run_rules as RR
from own_strategy import resample, swings
from scalp_map import comm_px

# (K, 1h-trend filter, entry: '15m' gap edge / '1m' gap inside the 15m gap, target: '15m' / '1m' nearest swing high)
# last5: the last 1-minute turning point before the entry (5 minutes each side) beyond it (the user's marks, 06.11.2026);
# last5x: the same without the 1:1 minimum
# struct: the last real correction before the entry (the user, 06.11.2026): the last 1-minute turning point beyond the
# entry after which the price made a new low below the low before it (a high in a sideways pause does not count);
# structx: the same without the 1:1 minimum
VARIANTS = [(2.0, False, '1m', t) for t in ('15p', 'struct', 'structx')]
PIP = {"JPY": 0.01}
BODY_X = 2.5        # the leg's average body vs the 50 candles before it
MIN_GAP = 0.33      # the gap's size vs the average range of the 30 candles before the leg (user: tiny gaps are no gaps)


def pip_of(s):
    if len(s) == 6 and s.isalpha() and s[3:] in ("USD", "JPY", "CHF", "CAD", "GBP", "NZD", "AUD", "EUR"):
        return 0.01 if s.endswith("JPY") else 0.0001
    return None


def run_symbol(s, df, cost, stop_at, noent, variants):
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
                    continue
                busy_until = ex_q
                out.append({"symbol": s, "variant": f"K{K:g}|e{emode}|t{tmode}", "time": t1[fill], "exit_time": t1[ex_q],
                            "dir": d, "entry": px, "stop": stop, "target": target, "exit": ex_px, "reason": why,
                            "rr": round(d * (target - px) / risk, 2), "cost_r": round(cost[fill] / risk, 3),
                            "r": round((d * (ex_px - px) - cost[fill]) / risk, 4), "leg": int(L),
                            "leg_start": t15[lo_i], "leg_top": t15[pk], "gap_lo": min(gap), "gap_hi": max(gap)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tag = a.symbols.replace(",", "_").replace("/", "-")[:60]
    ev, _ = NC.load()
    rows = []
    for s in [x for x in a.symbols.split(",") if x]:
        t0 = time.time()
        try:
            df = P.bars(s, "1Min", max(a.start, RP.MINUTE_START), end)
            sp = P.spread_1m(s, a.start, end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-120:]}")
            continue
        if df is None or df.empty or sp is None or len(sp) == 0:
            print(f"{s}: no data / no real spread - skipped")
            continue
        if "v" not in df:
            df["v"] = 1.0
        c = df["c"].to_numpy(float)
        cost = np.abs(sp.sort_index().reindex(df.index, method="ffill").to_numpy(float)) + comm_px(s, c)
        cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
        stop_at = RR.break_mask(df, "1Min")
        sm = RR.session_mask(df, s, "1Min")
        if sm is not None:
            stop_at = stop_at | np.asarray(sm, bool)
        nm, noent = RR.news_masks(df, s, "1Min")
        blk = NC.blocked(pd.Series(df.index), s, ev, minutes=2).to_numpy()
        noent = np.asarray(noent, bool) | blk
        stop_at = stop_at | np.asarray(nm, bool) | np.r_[blk[1:], False]
        rows += run_symbol(s, df, cost, stop_at, noent, VARIANTS)
        print(f"{s}: {sum(1 for x in rows if x['symbol'] == s)} trades (all variants), {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / f"own15_{tag}.csv.gz", index=False)
    print("trades:", len(rows))


if __name__ == "__main__":
    main()
