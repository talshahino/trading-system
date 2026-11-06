"""Entry scan on the 1-minute chart (user, 06.11.2026: "find when to enter first, then test the exits").

An exit can only harvest a move the entry gives. So before any stop or target, each entry rule is judged by
the price path after it: the signed move from the entry (next bar's open) to the close k bars later, for
k = 5, 15, 30, 60, 120, 240, in units of the instrument's round-trip trading cost (FTMO spread + commission).
An entry is worth exit tests only when that move beats 1 cost unit, on TRAIN and again on VALID.
Also kept: the furthest move for and against within 60 and 240 bars (in ATR), to shape the exits later.

Rules: decided on a closed 1-minute bar, entered at the next open; no entry around FTMO restricted news;
the path is cut at forced exits (stocks at the session end, market breaks, news). Signals of the same rule and
direction closer than 30 bars to the previous one are dropped (one move, one trade).

Entries (Israel-independent, exchange times):
  E01 NY opening range breakout     close beyond the 09:30-09:45 New York range, 09:45-11:30, first break only
  E02 London opening range breakout close beyond the 08:00-08:15 London range, 08:15-10:30, first break only
  E03 Asian range breakout          close beyond the 00:00-07:00 London range, 07:00-10:00 London, first only
  E04 previous-day sweep and reclaim  a bar trades beyond yesterday's high (low) and closes back inside: fade
  E05 previous-day breakout         close beyond yesterday's high (low): follow
  E06 Bollinger 3-sigma fade        close outside the 20-bar 3-sigma band: fade
  E07 RSI(2) pullback in 1h trend   1-hour EMA50 rising (falling) and 1-minute RSI(2) < 5 (> 95)
  E08 impulse bar follow            bar range > 3 x ATR(14) and body > 70% of range: follow
  E09 impulse bar fade              same bar: fade
  E10 VWAP 2-sigma fade             New York hours, close beyond the day's VWAP +/- 2 sigma: fade
  E11 squeeze breakout              20-bar Bollinger width in its lowest 10% of the last 1000 bars, then close
                                    beyond the 30-bar range
  E12 1h trend + 60-bar breakout    1-hour EMA20 > EMA50 and close above the last 60 bars' high (mirror short)

  python entry_scan.py --symbols EURUSD,XAUUSD [--start 2024-01-01]
Writes output/entry_scan_<tag>.json.
"""
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

import config
from agents import ftmo_universe as U
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
import run_rules as RR
from scalp_opt import be_offset

H = [5, 15, 30, 60, 120, 240]
TRAIN_FROM, VALID_FROM = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-04-30", tz="UTC")
GAP = 30


def _sh(x, k=1):
    x = np.asarray(x, float)
    return np.r_[np.full(k, np.nan), x[:-k]]


def local(df, tz):
    t = df.index.tz_convert(tz)
    return t, np.asarray(t.hour * 60 + t.minute), pd.Series(t.date).to_numpy()


def range_break(df, tz, a, b, until, c, h, l):
    """first close beyond the [a, b) minutes range of the local day, while b <= minute < until"""
    t, m, day = local(df, tz)
    inr = (m >= a) & (m < b)
    s = pd.DataFrame({"d": day, "h": np.where(inr, h, np.nan), "l": np.where(inr, l, np.nan)})
    hi = s.groupby("d")["h"].transform("max").to_numpy()
    lo = s.groupby("d")["l"].transform("min").to_numpy()
    win = (m >= b) & (m < until) & np.isfinite(hi)
    up, dn = win & (c > hi), win & (c < lo)
    first = lambda x: x & (pd.Series(x.astype(int)).groupby(day).cumsum().to_numpy() == 1)  # noqa: E731
    return first(up), first(dn)


def prev_day(df):
    t, m, day = local(df, "UTC")
    s = pd.DataFrame({"d": day, "h": df["h"].to_numpy(), "l": df["l"].to_numpy()})
    g = s.groupby("d").agg(h=("h", "max"), l=("l", "min"))
    g[["ph", "pl"]] = g[["h", "l"]].shift(1)
    return s["d"].map(g["ph"]).to_numpy(float), s["d"].map(g["pl"]).to_numpy(float)


def htf_ema(df, n_fast, n_slow):
    c = pd.Series(df["c"].to_numpy(float), index=df.index)
    c1 = c.resample("60min").last().dropna()
    f = pd.Series(R.ema(c1.to_numpy(), n_fast), index=c1.index)
    s = pd.Series(R.ema(c1.to_numpy(), n_slow), index=c1.index) if n_slow else None
    for x in (f, s):
        if x is not None:
            x.index = x.index + pd.Timedelta(minutes=59)   # known at the hour's last minute
    ff = f.reindex(df.index, method="ffill").to_numpy()
    ss = s.reindex(df.index, method="ffill").to_numpy() if s is not None else None
    return ff, ss, f


def entries(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    a = R.atr(h, l, c, 14)
    E = {}
    E["E01"] = range_break(df, "America/New_York", 570, 585, 690, c, h, l)
    E["E02"] = range_break(df, "Europe/London", 480, 495, 630, c, h, l)
    E["E03"] = range_break(df, "Europe/London", 0, 420, 600, c, h, l)
    ph, pl = prev_day(df)
    E["E04"] = ((l < pl) & (c > pl), (h > ph) & (c < ph))
    E["E05"] = ((c > ph) & (_sh(c) <= _sh(ph)), (c < pl) & (_sh(c) >= _sh(pl)))
    m20 = pd.Series(c).rolling(20).mean().to_numpy()
    sd20 = pd.Series(c).rolling(20).std().to_numpy()
    E["E06"] = (c < m20 - 3 * sd20, c > m20 + 3 * sd20)
    e50, _, e50h = htf_ema(df, 50, None)
    slope = e50 - pd.Series(e50h.shift(1).reindex(df.index, method="ffill")).to_numpy()
    r2 = R.rsi(c, 2)
    E["E07"] = ((slope > 0) & (r2 < 5), (slope < 0) & (r2 > 95))
    rng, body = h - l, np.abs(c - o)
    imp = (rng > 3 * _sh(a)) & (body > 0.7 * rng)
    E["E08"] = (imp & (c > o), imp & (c < o))
    E["E09"] = (imp & (c < o), imp & (c > o))
    v = df["v"].to_numpy(float) if "v" in df and np.nansum(df["v"].to_numpy(float)) > 0 else np.ones(len(c))
    t, m, day = local(df, "America/New_York")
    tp = (h + l + c) / 3
    g = pd.Series(day)
    cv = pd.Series(v).groupby(g).cumsum().to_numpy()
    vw = pd.Series(tp * v).groupby(g).cumsum().to_numpy() / np.where(cv > 0, cv, np.nan)
    var = pd.Series(tp * tp * v).groupby(g).cumsum().to_numpy() / np.where(cv > 0, cv, np.nan) - vw ** 2
    sdv = np.sqrt(np.clip(var, 0, None))
    ny = (m >= 600) & (m < 930)
    E["E10"] = (ny & (c < vw - 2 * sdv), ny & (c > vw + 2 * sdv))
    bw = 4 * sd20 / m20
    q = pd.Series(bw).rolling(1000, min_periods=500).quantile(0.1).to_numpy()
    sq = pd.Series(_sh(bw) <= _sh(q)).rolling(10, min_periods=1).max().to_numpy().astype(bool)
    hi30, lo30 = _sh(pd.Series(h).rolling(30).max().to_numpy()), _sh(pd.Series(l).rolling(30).min().to_numpy())
    E["E11"] = (sq & (c > hi30), sq & (c < lo30))
    f20, s50, _ = htf_ema(df, 20, 50)
    hi60, lo60 = _sh(pd.Series(h).rolling(60).max().to_numpy()), _sh(pd.Series(l).rolling(60).min().to_numpy())
    E["E12"] = ((f20 > s50) & (c > hi60), (f20 < s50) & (c < lo60))
    return E, a


def thin(idx, gap=GAP):
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def paths(df, sig_idx, d, a, cost_frac, stop_at):
    """signed forward moves per horizon (cost units and ATR units), MFE/MAE (ATR) within 60 and 240 bars"""
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    n = len(c)
    st = np.r_[np.flatnonzero(stop_at), n - 1]
    k_ = np.searchsorted(st, np.arange(n))                 # next forced exit at or after each bar
    nxt_stop = np.where(k_ < len(st), st[np.minimum(k_, len(st) - 1)], n - 1)
    rows = []
    for i in sig_idx:
        e_t = i + 1
        if e_t >= n - 1 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        e = o[e_t]
        cost = cost_frac * e
        end = nxt_stop[e_t]
        r = {"i": i, "d": d}
        for k in H:
            x = min(e_t + k - 1, end, n - 1)
            mv = d * (c[x] - e)
            r[f"c{k}"] = (mv - cost) / cost          # net of one round-trip cost, in cost units
            r[f"a{k}"] = mv / a[i]
        for k in (60, 240):
            x = min(e_t + k - 1, end, n - 1)
            hh, ll = h[e_t:x + 1].max(), l[e_t:x + 1].min()
            r[f"mfe{k}"] = (hh - e) / a[i] if d > 0 else (e - ll) / a[i]
            r[f"mae{k}"] = (e - ll) / a[i] if d > 0 else (hh - e) / a[i]
        r["cost_atr"] = cost / a[i]
        rows.append(r)
    return rows


def summarise(x):
    if len(x) == 0:
        return {"n": 0}
    out = {"n": int(len(x)), "cost_atr": round(float(x["cost_atr"].median()), 3)}
    for k in H:
        v = x[f"c{k}"]
        out[f"net{k}"] = round(float(v.mean()), 3)                       # mean net move, cost units
        out[f"t{k}"] = round(float(v.mean() / v.std() * np.sqrt(len(v))), 2) if len(v) > 1 and v.std() > 0 else 0.0
        out[f"atr{k}"] = round(float(x[f"a{k}"].mean()), 3)
    for k in (60, 240):
        out[f"mfe{k}"] = round(float(x[f"mfe{k}"].median()), 2)
        out[f"mae{k}"] = round(float(x[f"mae{k}"].median()), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    args = ap.parse_args()
    end = args.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tag = args.symbols.replace(",", "_").replace("/", "-")[:60]
    res, errors = [], []
    for s in [x for x in args.symbols.split(",") if x]:
        t0 = time.time()
        try:
            df = P.bars(s, "1Min", max(args.start, RP.MINUTE_START), end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-150:]}")
            continue
        if df is None or df.empty:
            continue
        real = s in U.ALL
        if real and s not in RR.SPREADS:
            try:
                RR.SPREADS[s] = P.spread_1m(s, (VALID_FROM - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), end)
            except Exception:  # noqa: BLE001
                RR.SPREADS[s] = None
        cost_frac = be_offset(df, s, real) if real else 2 * RP.cost_of(s)
        stop_at = RR.break_mask(df, "1Min")
        sm = RR.session_mask(df, s, "1Min")
        if sm is not None:
            stop_at = stop_at | np.asarray(sm, bool)
        nm, noent = RR.news_masks(df, s, "1Min")
        stop_at = stop_at | np.asarray(nm, bool)
        noent = np.asarray(noent, bool)
        try:
            E, a = entries(df)
        except Exception:  # noqa: BLE001
            errors.append(f"{s}:\n{traceback.format_exc()}")
            continue
        tm = df.index
        for eid, (L, S) in E.items():
            rows = []
            for d, sig in ((1, L), (-1, S)):
                sig = np.asarray(sig, bool) & ~np.r_[noent[1:], False]
                rows += paths(df, thin(np.flatnonzero(sig)), d, a, cost_frac, stop_at)
            if not rows:
                continue
            x = pd.DataFrame(rows)
            x["t"] = tm[x["i"].to_numpy()]
            tr, va = x[(x.t >= TRAIN_FROM) & (x.t < VALID_FROM)], x[x.t >= VALID_FROM]
            res.append({"eid": eid, "symbol": s, "cost_pct": round(100 * cost_frac, 4), "train": summarise(tr), "valid": summarise(va),
                        "valid_long": summarise(va[va.d > 0]), "valid_short": summarise(va[va.d < 0])})
        print(f"{s}: {len(df):,} bars, {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / f"entry_scan_{tag}.json").write_text(json.dumps({"rows": res, "errors": errors}, ensure_ascii=False), encoding="utf-8")
    print("rows:", len(res), "errors:", len(errors))
    for e in errors[:5]:
        print(e)


if __name__ == "__main__":
    main()
