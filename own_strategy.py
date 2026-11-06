"""The user's own strategy (memory 31.10.2026; scalping bot, 06.11.2026) as a backtest on 1-minute data.

  direction  1-hour chart: EMA20 above (below) EMA50 on completed hours -> only longs (shorts)
  impulse    15-minute chart: bodies suddenly much bigger than the recent history: 3-5 candles moving >= K x
             their count x the average body of the 30 before, two of them >= 1.5 x it (a small one between is
             fine); volume rule: the window's volume >= 2 x the median (the "buyer / seller mass")
  gap        1-minute chart, inside the FIRST candle of the impulse (else the second): the largest fair-value gap (bar j-1's high below
             bar j+1's low for a long); entry = its near edge (bar j+1's low), on the pullback
  stop       the SECOND 15-minute swing low (high) back below (above) the entry, not the first
  target     the nearest high (low): the nearest existing 15m swing high (low) beyond the impulse's extreme
  rules      risk:reward at least 1:1; the setup lives 8 hours; cancelled if, before the entry, the price covers 60% of the
             way from where it was when the setup appeared to the target; no entry around FTMO restricted news and holidays, positions closed before them and before
             market breaks; one position at a time per instrument
Costs: the real spread at the entry minute (Dukascopy ask - bid) + FTMO commission, in R.
Variants (to see what each rule is worth): K 1.5 / 2.5, volume rule on / off, stop at the second / first swing.
TRAIN 2024-01-01..2025-04-30, VALID after.

  python own_strategy.py --symbols EURUSD,GBPUSD
Writes output/own_trades_<tag>.csv.gz (every trade of every variant).
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
import run_rules as RR
from scalp_map import comm_px

VARIANTS = [(k, vol, sw) for k in (1.5, 2.5) for vol in (True, False) for sw in (2, 1)]


def resample(df, rule):
    x = df.resample(rule, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    return x


def swings(h, l, w=2):
    """fractal highs / lows (confirmed w bars later): arrays of (confirm index, price)"""
    n = len(h)
    hi, lo = [], []
    for i in range(w, n - w):
        if h[i] == h[i - w:i + w + 1].max():
            hi.append((i + w, h[i]))
        if l[i] == l[i - w:i + w + 1].min():
            lo.append((i + w, l[i]))
    return hi, lo


def run_symbol(s, df, cost, stop_at, noent, variants):
    o1, h1, l1, c1 = (df[x].to_numpy(float) for x in "ohlc")
    t1 = df.index
    n1 = len(t1)
    m15 = resample(df, "15min")
    h60 = resample(df, "60min")
    e20, e50 = R.ema(h60["c"].to_numpy(), 20), R.ema(h60["c"].to_numpy(), 50)
    tr60 = pd.Series(np.sign(e20 - e50), index=h60.index + pd.Timedelta(minutes=60))   # known when the hour closes
    o, h, l, c, v = (m15[x].to_numpy(float) for x in "ohlcv")
    t15 = m15.index
    trend = tr60.reindex(t15 + pd.Timedelta(minutes=15), method="ffill").to_numpy()     # at each 15m candle's close
    rng = h - l
    med = pd.Series(rng).rolling(96, min_periods=48).median().shift(1).to_numpy()
    medv = pd.Series(v).rolling(96, min_periods=48).median().shift(1).to_numpy()
    body = c - o
    sh, sl_ = swings(h, l)
    sh_i, sh_p = np.array([x[0] for x in sh]), np.array([x[1] for x in sh])
    sl_i, sl_p = np.array([x[0] for x in sl_]), np.array([x[1] for x in sl_])
    out = []
    for K, vol_rule, which in variants:
        # impulse (user, 06.11.2026): candle bodies suddenly much bigger than the recent history "seen by eye" -
        # a window of 3-5 candles whose move in the trend's direction is >= K x (its length) x the average body
        # of the 30 candles before it, with at least two candles of 1.5 x that average in the direction; a small
        # candle in the middle is allowed
        avgb = pd.Series(np.abs(body)).rolling(30, min_periods=20).mean().shift(1).to_numpy()
        n15 = len(c)
        busy_until = -1
        i = 0
        while i < n15:
            d = trend[i]
            if not (d in (1, -1)) or not np.isfinite(avgb[i]) or avgb[i] <= 0:
                i += 1
                continue
            hit = None
            for w in (3, 4, 5):
                j = i + w - 1
                if j >= n15 or np.any(trend[i:j + 1] != d):
                    break
                base_b = avgb[i]
                net = d * (c[j] - o[i])
                bigs = int(np.sum(d * body[i:j + 1] >= 1.5 * base_b))
                if d * body[i] > 0 and net >= K * w * base_b and bigs >= 2:
                    hit = j
                    break
            if hit is None:
                i += 1
                continue
            first, last = i, hit
            i = hit + 1
            if vol_rule and not (v[first:last + 1].mean() >= 2 * medv[first] if np.isfinite(medv[first]) and medv[first] > 0 else False):
                continue
            setup_t = t15[last] + pd.Timedelta(minutes=15)          # the run is known when its last candle closes
            # the largest 1-minute fair-value gap inside the first impulse candle (if none, the second)
            best, edge = 0.0, None
            for cand_i in (first, first + 1):
                if edge is not None or cand_i > last:
                    break
                a = t1.searchsorted(t15[cand_i])
                b = t1.searchsorted(t15[cand_i] + pd.Timedelta(minutes=15))
                for jj in range(a + 1, b - 1):
                    if d > 0 and l1[jj + 1] > h1[jj - 1] and l1[jj + 1] - h1[jj - 1] > best:
                        best, edge = l1[jj + 1] - h1[jj - 1], l1[jj + 1]
                    if d < 0 and h1[jj + 1] < l1[jj - 1] and l1[jj - 1] - h1[jj + 1] > best:
                        best, edge = l1[jj - 1] - h1[jj + 1], h1[jj + 1]
            if edge is None:
                continue
            # target: the nearest existing 15m swing high (low) beyond the impulse's extreme (the next resistance)
            ext = h[first:last + 1].max() if d > 0 else l[first:last + 1].min()
            if d > 0:
                k = (sh_i <= last) & (sh_p > ext)
                target = sh_p[k].min() if k.any() else np.nan
            else:
                k = (sl_i <= last) & (sl_p < ext)
                target = sl_p[k].max() if k.any() else np.nan
            if not np.isfinite(target):
                continue
            # stop: the second (or first) confirmed 15m swing back, beyond the entry
            if d > 0:
                k = (sl_i <= last) & (sl_p < edge)
                cand = sl_p[k]
            else:
                k = (sh_i <= last) & (sh_p > edge)
                cand = sh_p[k]
            if len(cand) < which:
                continue
            stop = cand[-which]
            risk = d * (edge - stop)
            if risk < 0.25 * med[last] or risk <= 0 or d * (target - edge) < risk:   # a real stop (>= 1/4 of a normal 15m range), at least 1:1
                continue
            s0 = t1.searchsorted(setup_t)
            if s0 >= n1 or s0 <= busy_until:
                continue
            if d * (c1[s0 - 1] - edge) <= 0:                      # already through the entry: no pullback entry
                continue
            s1 = min(n1, t1.searchsorted(setup_t + pd.Timedelta(hours=8)))
            fill, base = None, c1[s0 - 1]          # 60% rule measured from the price when the setup appears
            for q in range(s0, s1):
                if stop_at[q]:
                    break
                if d * (c1[q] - base) >= 0.6 * d * (target - base) or ((d > 0 and h1[q] >= target) or (d < 0 and l1[q] <= target)):
                    break                                          # went 60% of the way to the target first: setup gone
                if (d > 0 and l1[q] <= edge) or (d < 0 and h1[q] >= edge):
                    if noent[q]:
                        break
                    fill = q
                    break
            if fill is None:
                continue
            px = min(edge, o1[fill]) if d > 0 else max(edge, o1[fill])
            if d * (px - stop) <= 0:
                continue
            ex_px, ex_q, why = None, None, None
            for q in range(fill, n1):
                if (d > 0 and l1[q] <= stop) or (d < 0 and h1[q] >= stop):
                    ex_px, ex_q, why = (min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if q > fill else stop, q, "stop"
                    break
                if q > fill and ((d > 0 and h1[q] >= target) or (d < 0 and l1[q] <= target)):
                    ex_px, ex_q, why = (max(target, o1[q]) if d > 0 else min(target, o1[q])), q, "target"
                    break
                if stop_at[q]:
                    ex_px, ex_q, why = c1[q], q, "forced"
                    break
            if ex_px is None:
                continue
            rk = d * (px - stop)
            r = (d * (ex_px - px) - cost[fill]) / rk
            busy_until = ex_q
            out.append({"symbol": s, "variant": f"K{K:g}|vol{int(vol_rule)}|sw{which}", "time": t1[fill], "exit_time": t1[ex_q],
                        "dir": int(d), "entry": px, "stop": stop, "target": target, "exit": ex_px, "reason": why,
                        "rr": round(d * (target - px) / rk, 2), "cost_r": round(cost[fill] / rk, 3), "r": round(r, 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tag = a.symbols.replace(",", "_").replace("/", "-")[:60]
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
        stop_at = stop_at | np.asarray(nm, bool)
        # FTMO restricted events: out 2 minutes before, no entry within 2 minutes either side
        from agents import news_calendar as NC
        ev, _ = NC.load()
        blk = NC.blocked(pd.Series(df.index), s, ev, minutes=2).to_numpy()
        noent = np.asarray(noent, bool) | blk
        stop_at = stop_at | np.r_[blk[1:], False]
        rows += run_symbol(s, df, cost, stop_at, noent, VARIANTS)
        print(f"{s}: {len(df):,} bars, {sum(1 for x in rows if x['symbol'] == s)} trades (all variants), {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / f"own_trades_{tag}.csv.gz", index=False)
    print("trades:", len(rows))


if __name__ == "__main__":
    main()
