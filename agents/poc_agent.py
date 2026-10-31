"""Volume-profile POC pullback strategy (from the TikTok videos the user sent), as exact rules.

As explained in the videos:
  1. Trend: two rising swing highs and two rising swing lows = bullish -> only buys
     (mirror for bearish -> only sells).
  2. Volume profile from the most recent swing low up to the new swing high.
  3. Its POC (point of control) = the price with the most volume in that move.
  4. Wait for price to come back to the POC and "react", then buy.
The videos never say where the stop and the target are, or what a "reaction" is, so a
few reasonable readings are tested side by side (fixed in advance):
  entry   touch    = limit order at the POC (filled when price trades there)
          reaction = after touching the POC, a bar closes back above it -> buy next open
  stop    swing    = just beyond the swing low the profile starts from
          atr      = 1 ATR(14) beyond the entry
  target  high     = the swing high (the top of the move)
          2R / 3R  = 2 or 3 times the stop distance
Swings are "fractal" pivots: a high with k lower highs on each side (k = 3 and 5 tested);
a pivot is only known k bars after it happens, and the setup starts only then.

Every real trade is also compared with RANDOM trades in the same stock and timeframe with
the same direction, stop % and target %: if the POC entry adds nothing, the two match.
"""
import bisect
import time

import numpy as np
import pandas as pd
import requests

import config
from agents.news_agent import _headers

NY = "America/New_York"
BINS = 40
MAX_WAIT = 50       # bars to wait for the pullback after the swing high is confirmed
MAX_HOLD = 100      # bars before a time exit
ENTRIES = ["touch", "reaction"]
EXITS = [("swing", "high"), ("swing", "2R"), ("swing", "3R"), ("atr", "high"), ("atr", "2R"), ("atr", "3R")]


# ---------------- data ----------------
def _get(url, params):
    out, token, fails = [], None, 0
    while True:
        config.check_time_budget()
        p = dict(params)
        if token:
            p["page_token"] = token
        try:
            r = requests.get(url, headers=_headers(), params=p, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            fails += 1
            if fails > 20:
                raise
            time.sleep(10)
            continue
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        d = r.json()
        out.extend((d.get("bars") or {}).get(params["symbols"], []))
        token = d.get("next_page_token")
        if not token:
            return out
        time.sleep(0.3)


# Forex and commodities come from Dukascopy (a Swiss bank's free historical feed) through
# the open-source dukascopy-node tool. Forex has no central exchange, so "volume" there is
# tick volume (how many price updates), which is also what TradingView shows on forex.
DUKA = {"EURUSD": "eurusd", "GBPUSD": "gbpusd", "USDJPY": "usdjpy", "AUDUSD": "audusd",
        "USDCAD": "usdcad", "XAUUSD": "xauusd", "XAGUSD": "xagusd", "WTI": "lightcmdusd",
        # FTMO's other markets: stock indices, more commodities, crypto (Dukascopy CFDs)
        "US30": "usa30idxusd", "US100": "usatechidxusd", "US500": "usa500idxusd",
        "GER40": "deuidxeur", "UK100": "gbridxgbp", "JP225": "jpnidxjpy",
        "BRENT": "brentcmdusd", "NATGAS": "gascmdusd", "COPPER": "coppercmdusd",
        "BTCUSD": "btcusd", "ETHUSD": "ethusd"}
DUKA_TF = {"1Min": "m1", "1Hour": "h1", "1Day": "d1"}
RESAMPLE = {"5Min": "5min", "15Min": "15min"}      # built locally from 1-minute bars


def _duka_year(symbol, timeframe, y, end, price="bid"):
    import glob
    import os
    import shutil
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp()
    to = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp(end)).strftime("%Y-%m-%d")
    cmd = ["npx", "--yes", "dukascopy-node@1.51.1", "-i", DUKA[symbol], "-from", f"{y}-01-01", "-to", to,
           "-t", DUKA_TF[timeframe], "-p", price, "-v", "-vu", "units", "-f", "csv", "-dir", tmp, "-s",
           "-r", "3", "-bs", "5", "-bp", "500", "--cache", "--cache-path", str(config.CACHE_DIR / "duka_raw")]
    chrome = os.environ.get("CHROME_PATH") or shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        cmd += ["--chrome-path", chrome]
    for attempt in range(3):
        config.check_time_budget()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        files = glob.glob(os.path.join(tmp, "*.csv"))
        if files:
            try:
                df = pd.read_csv(files[0])
            except pd.errors.EmptyDataError:   # no bars for this symbol/year (e.g. before listing)
                shutil.rmtree(tmp, ignore_errors=True)
                return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
            shutil.rmtree(tmp, ignore_errors=True)
            df = df.rename(columns={"timestamp": "t", "open": "o", "high": "h", "low": "l",
                                    "close": "c", "volume": "v"})
            df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            return df[["t", "o", "h", "l", "c", "v"]]
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"Dukascopy download failed for {symbol} {timeframe} {y}: "
                       f"{(r.stderr or r.stdout)[-400:]}")


def spread_1m(symbol, start, end):
    """Historical spread per minute (ask close - bid close) from Dukascopy, in price units.
    Empty Series when the instrument has no Dukascopy history (spread then estimated)."""
    if symbol not in DUKA:
        return pd.Series(dtype="float64")
    bid = bars(symbol, "1Min", start, end)
    frames = []
    now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        f = config.CACHE_DIR / f"poc_{symbol.replace('/', '-')}_1Min_ask_{y}.csv.gz"
        df = None
        if f.exists():
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                f.unlink()
        if df is None:
            e = min(pd.Timestamp(f"{y}-12-31 23:59", tz="UTC"), now)
            if e <= max(pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(start, tz="UTC")):
                continue
            df = _duka_year(symbol, "1Min", y, e.strftime("%Y-%m-%d"), price="ask")
            if y < now.year:
                df.to_csv(f, index=False)
        frames.append(df)
    if not frames or bid.empty:
        return pd.Series(dtype="float64")
    ask = pd.concat(frames, ignore_index=True)
    if ask.empty:
        return pd.Series(dtype="float64")
    ask["t"] = pd.to_datetime(ask["t"], utc=True)
    ask = ask.drop_duplicates("t").set_index("t")["c"]
    sp = (ask - bid["c"]).dropna()
    return sp[sp >= 0].astype("float32")


def bars(symbol, timeframe, start, end):
    """OHLCV for a stock/ETF ('SPY'), crypto ('BTC/USD') or forex/commodity ('XAUUSD'),
    cached per year."""
    crypto = "/" in symbol
    duka = symbol in DUKA
    if timeframe == "4Hour":
        h1 = bars(symbol, "1Hour", start, end)
        if h1.empty:
            return h1
        agg = h1.resample("4h", label="left", closed="left").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
        return agg[agg["v"] > 0]
    if duka and timeframe in RESAMPLE:
        m1 = bars(symbol, "1Min", start, end)
        if m1.empty:
            return m1
        agg = m1.resample(RESAMPLE[timeframe], label="left", closed="left").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
        return agg[agg["v"] > 0]
    frames = []
    now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        f = config.CACHE_DIR / f"poc_{symbol.replace('/', '-')}_{timeframe}_{y}.csv.gz"
        cached = None
        if f.exists():
            try:
                cached = pd.read_csv(f)
            except pd.errors.EmptyDataError:   # a zero-byte cache file: drop it and download again
                f.unlink()
        if cached is not None:
            df = cached
        else:
            s = max(pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(start, tz="UTC"))
            e = min(pd.Timestamp(f"{y}-12-31 23:59", tz="UTC"), now)
            if e <= s:
                continue
            p = {"symbols": symbol, "timeframe": timeframe, "start": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "end": e.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000, "sort": "asc"}
            if duka:
                df = _duka_year(symbol, timeframe, y, e.strftime("%Y-%m-%d"))
                if y < now.year:
                    df.to_csv(f, index=False)
                frames.append(df)
                print(f"    {symbol} {timeframe} {y}: {len(df):,} bars")
                continue
            if crypto:
                raw = _get(f"{config.ALPACA_DATA_URL}/v1beta3/crypto/us/bars", p)
            else:
                p.update({"adjustment": "all", "feed": config.BAR_FEED})
                raw = _get(f"{config.ALPACA_DATA_URL}/v2/stocks/bars", p)
            df = pd.DataFrame(raw, columns=["t", "o", "h", "l", "c", "v", "n", "vw"])
            if y < now.year:
                df.to_csv(f, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.drop_duplicates("t").sort_values("t").set_index("t")[["o", "h", "l", "c", "v"]]
    if not crypto and not duka and timeframe != "1Day":
        ny = df.index.tz_convert(NY)
        mins = ny.hour * 60 + ny.minute
        df = df[(mins >= 570) & (mins < 960)]          # regular hours only
    return df[df["v"] > 0]


# ---------------- strategy ----------------
def _compact(df):
    """Small dtypes: 1-minute runs produce millions of rows."""
    if df.empty:
        return df
    for c in ("stop_pct", "tgt_pct", "r", "ret"):
        df[c] = df[c].astype("float32")
    for c in ("bar", "held"):
        df[c] = df[c].astype("int32")
    for c in ("dir", "k"):
        df[c] = df[c].astype("int8")
    for c in ("entry", "exit"):
        df[c] = df[c].astype("category")
    return df



def _pivots(h, l, k):
    """Indices of swing highs / lows: higher than the k bars before (strictly) and at least as
    high as the k bars after (vectorized, so 1-minute data stays fast)."""
    hs, ls = pd.Series(h), pd.Series(l)
    left_h = hs.shift(1).rolling(k).max()
    right_h = hs[::-1].shift(1).rolling(k).max()[::-1]
    left_l = ls.shift(1).rolling(k).min()
    right_l = ls[::-1].shift(1).rolling(k).min()[::-1]
    ph = np.flatnonzero(((hs > left_h) & (hs >= right_h)).to_numpy()).tolist()
    pl = np.flatnonzero(((ls < left_l) & (ls <= right_l)).to_numpy()).tolist()
    return ph, pl


def _poc(h, l, v, a, b):
    """POC of the volume profile over bars a..b (inclusive), volume spread evenly over each
    bar's high-low range."""
    lo, hi = l[a:b + 1].min(), h[a:b + 1].max()
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, BINS + 1)
    prof = np.zeros(BINS)
    for i in range(a, b + 1):
        rng = max(h[i] - l[i], 1e-12)
        ov = np.clip(np.minimum(edges[1:], h[i]) - np.maximum(edges[:-1], l[i]), 0, None)
        prof += v[i] * ov / rng if h[i] > l[i] else v[i] * ((edges[:-1] <= h[i]) & (edges[1:] >= l[i]))
    j = int(prof.argmax())
    return (edges[j] + edges[j + 1]) / 2


def _atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(n).mean().to_numpy()
    return atr


def _setups(df, k):
    """Bullish and bearish setups: (direction, confirm_bar, poc, swing_extreme_start, swing_extreme_end)."""
    h, l, v = df["h"].to_numpy(), df["l"].to_numpy(), df["v"].to_numpy()
    ph, pl = _pivots(h, l, k)
    out = []
    # bullish: new pivot high above the previous one, with the last two pivot lows rising
    for idx in range(1, len(ph)):
        i_h, i_prev = ph[idx], ph[idx - 1]
        if h[i_h] <= h[i_prev]:
            continue
        m = bisect.bisect_left(pl, i_h)
        if m < 2:
            continue
        j_last, j_prev = pl[m - 1], pl[m - 2]
        if not (i_prev < j_last < i_h) or l[j_last] <= l[j_prev]:
            continue
        poc = _poc(h, l, v, j_last, i_h)
        if poc is not None:
            out.append((1, i_h + k, poc, l[j_last], h[i_h]))
    # bearish: mirror
    for idx in range(1, len(pl)):
        i_l, i_prev = pl[idx], pl[idx - 1]
        if l[i_l] >= l[i_prev]:
            continue
        m = bisect.bisect_left(ph, i_l)
        if m < 2:
            continue
        j_last, j_prev = ph[m - 1], ph[m - 2]
        if not (i_prev < j_last < i_l) or h[j_last] >= h[j_prev]:
            continue
        poc = _poc(h, l, v, j_last, i_l)
        if poc is not None:
            out.append((-1, i_l + k, poc, h[j_last], l[i_l]))
    return sorted(out, key=lambda s: s[1])


def _walk(o, h, l, c, start, d, entry, stop, target, same_bar_entry):
    """Exit of a trade entered at bar `start`: returns (R multiple gross, return gross, bars)."""
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    for t in range(start, min(start + MAX_HOLD, len(c))):
        if t > start or same_bar_entry:
            if t > start and ((d > 0 and o[t] <= stop) or (d < 0 and o[t] >= stop)):
                px = o[t]                                       # gap through the stop
                return d * (px - entry) / risk, d * (px / entry - 1), t - start
            hit_stop = l[t] <= stop if d > 0 else h[t] >= stop
            # On the fill bar of a limit order the order of high/low is unknown: only the
            # stop can count there (pessimistic), never the target.
            hit_tgt = (h[t] >= target if d > 0 else l[t] <= target) and t > start
            if hit_stop:                                        # same bar: assume the stop first
                return d * (stop - entry) / risk, d * (stop / entry - 1), t - start
            if hit_tgt:
                px = max(target, o[t]) if d > 0 else min(target, o[t])
                return d * (px - entry) / risk, d * (px / entry - 1), t - start
    t = min(start + MAX_HOLD, len(c)) - 1
    return d * (c[t] - entry) / risk, d * (c[t] / entry - 1), t - start


def trades(df, k, cost_side):
    """All trades for every entry/exit reading. One row per (setup, entry, exit)."""
    if len(df) < 200:
        return pd.DataFrame()
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    atr = _atr(h, l, c).tolist()
    setups = _setups(df, k)
    o, h, l, c = o.tolist(), h.tolist(), l.tolist(), c.tolist()
    n = len(c)
    rows, busy = [], {}
    for d, t0, poc, sw_start, sw_end in setups:
        if t0 >= n - 2:
            continue
        for entry_kind in ENTRIES:
            fill_t, fill_px, touched = None, None, False
            for t in range(t0 + 1, min(t0 + 1 + MAX_WAIT, n - 1)):
                touch = l[t] <= poc if d > 0 else h[t] >= poc
                # A resting limit order fills as soon as price trades at the POC, even if the
                # same bar then falls through the stop (checked on the fill bar, pessimistic).
                if entry_kind == "touch" and touch:
                    fill_t = t
                    fill_px = min(o[t], poc) if d > 0 else max(o[t], poc)
                    break
                # Setup dies if price breaks the swing the profile started from.
                if (d > 0 and l[t] < sw_start) or (d < 0 and h[t] > sw_start):
                    break
                if entry_kind == "reaction":
                    touched = touched or touch
                    if touched and ((d > 0 and c[t] > poc) or (d < 0 and c[t] < poc)):
                        fill_t, fill_px = t + 1, o[t + 1]
                        break
            if fill_t is None or not np.isfinite(atr[fill_t - 1]):
                continue
            for stop_kind, tgt_kind in EXITS:
                vkey = (entry_kind, stop_kind, tgt_kind)
                if fill_t <= busy.get(vkey, -1):
                    continue        # already in a trade with this variant on this instrument
                stop = sw_start if stop_kind == "swing" else fill_px - d * atr[fill_t - 1]
                risk = d * (fill_px - stop)
                # A stop closer than a quarter of a typical bar is not a real stop (it would
                # also make R explode on tiny distances): skip.
                if risk < 0.25 * atr[fill_t - 1]:
                    continue
                target = (sw_end if tgt_kind == "high" else
                          fill_px + d * risk * (2 if tgt_kind == "2R" else 3))
                if d * (target - fill_px) <= 0:
                    continue
                res = _walk(o, h, l, c, fill_t, d, fill_px, stop, target, entry_kind == "touch")
                if res is None:
                    continue
                r_mult, ret, held = res
                busy[vkey] = fill_t + held
                cost_r = 2 * cost_side * fill_px / risk
                rows.append({"time": df.index[fill_t], "bar": fill_t, "dir": d, "entry": entry_kind,
                             "exit": f"{stop_kind}/{tgt_kind}", "k": k,
                             "stop_pct": abs(fill_px - stop) / fill_px, "tgt_pct": abs(target - fill_px) / fill_px,
                             "r": r_mult - cost_r, "ret": ret - 2 * cost_side, "held": held})
    return _compact(pd.DataFrame(rows))


def random_baseline(df, tr, cost_side, draws=20, seed=1):
    """Same direction, same stop % and target %, random entry bars: average net R per trade."""
    if tr.empty:
        return tr.assign(r_rand=[])
    rng = np.random.default_rng(seed)
    o, h, l, c = (df[x].to_numpy().tolist() for x in "ohlc")
    n = len(c)
    draws = max(2, min(draws, 300_000 // max(len(tr), 1)))   # keep huge 1-minute sets fast
    out = []
    for row in tr.itertuples(index=False):
        vals = []
        for _ in range(draws):
            t = int(rng.integers(20, n - 2))
            e = o[t]
            stop = e * (1 - row.dir * row.stop_pct)
            tgt = e * (1 + row.dir * row.tgt_pct)
            res = _walk(o, h, l, c, t, row.dir, e, stop, tgt, False)
            if res:
                vals.append(res[0] - 2 * cost_side * e / abs(e - stop))
        out.append(np.mean(vals) if vals else np.nan)
    return tr.assign(r_rand=out)


# ---------------- more setups from the other videos ----------------
RANGE_N = 20          # bars that define a consolidation
RANGE_ATR = 4.0       # consolidation = the last 20 bars span at most 4 ATR
RANGE_EXITS = [("range", "2R"), ("range", "3R"), ("range", "measured"),
               ("atr", "2R"), ("atr", "3R"), ("atr", "measured")]


def _profile(h, l, v, a, b):
    lo, hi = l[a:b + 1].min(), h[a:b + 1].max()
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, BINS + 1)
    prof = np.zeros(BINS)
    for i in range(a, b + 1):
        if h[i] > l[i]:
            ov = np.clip(np.minimum(edges[1:], h[i]) - np.maximum(edges[:-1], l[i]), 0, None)
            prof += v[i] * ov / (h[i] - l[i])
        else:
            prof[min(int((h[i] - lo) / (hi - lo) * BINS), BINS - 1)] += v[i]
    return edges, prof


def _value_area(edges, prof, share=0.70):
    """POC, value area high and low (the 70% of volume around the POC)."""
    j = int(prof.argmax())
    lo_i, hi_i, tot = j, j, prof[j]
    total = prof.sum()
    while tot < share * total and (lo_i > 0 or hi_i < len(prof) - 1):
        down = prof[lo_i - 1] if lo_i > 0 else -1
        up = prof[hi_i + 1] if hi_i < len(prof) - 1 else -1
        if up >= down:
            hi_i += 1
            tot += prof[hi_i]
        else:
            lo_i -= 1
            tot += prof[lo_i]
    return (edges[j] + edges[j + 1]) / 2, edges[hi_i + 1], edges[lo_i]


def _range_setups(df, sweep):
    """Consolidation -> (optional sweep of one side) -> breakout: (dir, breakout_bar, poc, lo, hi)."""
    h, l, c, v = (df[x].to_numpy() for x in "hlcv")
    atr = _atr(h, l, c)
    n = len(c)
    span = (pd.Series(h).rolling(RANGE_N).max() - pd.Series(l).rolling(RANGE_N).min()).to_numpy()
    ok = np.flatnonzero(np.isfinite(atr) & (span <= RANGE_ATR * atr))
    out = []
    t = RANGE_N
    while t < n - 1:
        j = bisect.bisect_left(ok, t)
        if j >= len(ok):
            break
        t = int(ok[j])
        if t >= n - 1:
            break
        a = t - RANGE_N + 1
        lo, hi = l[a:t + 1].min(), h[a:t + 1].max()
        start, swept_lo, swept_hi, setup = a, False, False, None
        u = t + 1
        while u < min(n, t + 100):
            if sweep:
                if l[u] < lo:
                    swept_lo = True
                if h[u] > hi:
                    swept_hi = True
                if c[u] > hi and swept_lo and not (h[u] > hi and l[u] < lo):
                    setup = (1, u)
                    break
                if c[u] < lo and swept_hi and not (h[u] > hi and l[u] < lo):
                    setup = (-1, u)
                    break
                if c[u] > hi + (hi - lo) or c[u] < lo - (hi - lo):
                    break                               # ran away without the pattern
            else:
                if c[u] > hi:
                    setup = (1, u)
                    break
                if c[u] < lo:
                    setup = (-1, u)
                    break
            u += 1
        if setup:
            prof = _profile(h, l, v, start, u - 1)
            if prof:
                poc = _value_area(*prof)[0]
                out.append((setup[0], setup[1], poc, lo, hi))
        t = max(u, t + 1) + 1
    return out


def trades_range(df, sweep, cost_side):
    """Consolidation breakout (or sweep then breakout), pullback to the range's POC."""
    if len(df) < 200:
        return pd.DataFrame()
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    atr = _atr(h, l, c).tolist()
    setups = _range_setups(df, sweep)
    o, h, l, c = o.tolist(), h.tolist(), l.tolist(), c.tolist()
    n = len(c)
    rows, busy = [], {}
    for d, b, poc, lo, hi in setups:
        opp = lo if d > 0 else hi
        for entry_kind in ENTRIES:
            fill_t, fill_px, touched = None, None, False
            for t in range(b + 1, min(b + 1 + MAX_WAIT, n - 1)):
                touch = l[t] <= poc if d > 0 else h[t] >= poc
                if entry_kind == "touch" and touch:
                    fill_t, fill_px = t, (min(o[t], poc) if d > 0 else max(o[t], poc))
                    break
                if (d > 0 and c[t] < opp) or (d < 0 and c[t] > opp):
                    break                               # back through the whole range: failed
                if entry_kind == "reaction":
                    touched = touched or touch
                    if touched and ((d > 0 and c[t] > poc) or (d < 0 and c[t] < poc)):
                        fill_t, fill_px = t + 1, o[t + 1]
                        break
            if fill_t is None or not np.isfinite(atr[fill_t - 1]):
                continue
            for stop_kind, tgt_kind in RANGE_EXITS:
                vkey = (entry_kind, stop_kind, tgt_kind)
                if fill_t <= busy.get(vkey, -1):
                    continue        # already in a trade with this variant on this instrument
                stop = opp if stop_kind == "range" else fill_px - d * atr[fill_t - 1]
                risk = d * (fill_px - stop)
                # A stop closer than a quarter of a typical bar is not a real stop (it would
                # also make R explode on tiny distances): skip.
                if risk < 0.25 * atr[fill_t - 1]:
                    continue
                target = ((hi + (hi - lo)) if d > 0 else (lo - (hi - lo))) if tgt_kind == "measured" else \
                    fill_px + d * risk * (2 if tgt_kind == "2R" else 3)
                if d * (target - fill_px) <= 0:
                    continue
                res = _walk(o, h, l, c, fill_t, d, fill_px, stop, target, entry_kind == "touch")
                if res is None:
                    continue
                r_mult, ret, held = res
                busy[vkey] = fill_t + held
                rows.append({"time": df.index[fill_t], "bar": fill_t, "dir": d, "entry": entry_kind,
                             "exit": f"{stop_kind}/{tgt_kind}", "k": 0,
                             "stop_pct": abs(fill_px - stop) / fill_px, "tgt_pct": abs(target - fill_px) / fill_px,
                             "r": r_mult - 2 * cost_side * fill_px / risk, "ret": ret - 2 * cost_side, "held": held})
    return _compact(pd.DataFrame(rows))


def trades_fade(df, cost_side, crypto=False):
    """Fade the value-area extremes of the PREVIOUS session back to its POC (intraday only).
    Long: price trades below yesterday's value-area low and the bar closes back above it ->
    buy next open, target yesterday's POC. Short: mirror at the value-area high.
    Exit at the target, the stop, or the end of the session."""
    if len(df) < 200:
        return pd.DataFrame()
    o, h, l, c, v = (df[x].to_numpy() for x in "ohlcv")
    atr = _atr(h, l, c)
    idx = df.index if crypto else df.index.tz_convert(NY)
    sess = pd.Series(idx.date)
    bounds = sess.ne(sess.shift()).to_numpy().nonzero()[0].tolist() + [len(c)]
    rows = []
    for s in range(1, len(bounds) - 1):
        pa, pb, a, b = bounds[s - 1], bounds[s] - 1, bounds[s], bounds[s + 1] - 1
        prof = _profile(h, l, v, pa, pb)
        if not prof or b - a < 3:
            continue
        poc, vah, val = _value_area(*prof)
        done = {1: False, -1: False}
        for t in range(a, b):
            for d, edge in ((1, val), (-1, vah)):
                if done[d]:
                    continue
                if d > 0 and not (l[t] < val < c[t] < poc):
                    continue
                if d < 0 and not (h[t] > vah > c[t] > poc):
                    continue
                done[d] = True
                e_t, e = t + 1, o[t + 1]
                if not np.isfinite(atr[t]) or d * (poc - e) <= 0:
                    continue
                ext = l[a:t + 1].min() if d > 0 else h[a:t + 1].max()
                for stop_kind, stop in (("atr", e - d * atr[t]), ("extreme", ext - d * 0.1 * atr[t])):
                    risk = d * (e - stop)
                    if risk < 0.25 * atr[t]:
                        continue
                    # walk until target, stop or the session's last bar
                    r_mult = None
                    u = b
                    for u in range(e_t, b + 1):
                        if (d > 0 and l[u] <= stop) or (d < 0 and h[u] >= stop):
                            px = o[u] if ((d > 0 and o[u] <= stop) or (d < 0 and o[u] >= stop)) and u > e_t else stop
                            r_mult = d * (px - e) / risk
                            break
                        if (d > 0 and h[u] >= poc) or (d < 0 and l[u] <= poc):
                            r_mult = d * (poc - e) / risk
                            break
                    if r_mult is None:
                        r_mult = d * (c[b] - e) / risk
                    rows.append({"time": df.index[e_t], "bar": e_t, "dir": d, "entry": "reaction",
                                 "exit": f"{stop_kind}/poc", "k": 0, "stop_pct": risk / e,
                                 "tgt_pct": abs(poc - e) / e, "r": r_mult - 2 * cost_side * e / risk,
                                 "ret": 0.0, "held": (u if r_mult is not None else b) - e_t})
    return _compact(pd.DataFrame(rows))


# the 94 FTMO instruments chosen by the user (agents/ftmo_universe.py)
from agents import ftmo_universe as _U  # noqa: E402

for _k, _v in _U.DUKA.items():
    DUKA.setdefault(_k, _v)
