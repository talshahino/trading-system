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


def bars(symbol, timeframe, start, end):
    """OHLCV for a stock/ETF ('SPY') or crypto ('BTC/USD'), cached per year."""
    crypto = "/" in symbol
    frames = []
    now = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        f = config.CACHE_DIR / f"poc_{symbol.replace('/', '-')}_{timeframe}_{y}.csv.gz"
        if f.exists():
            df = pd.read_csv(f)
        else:
            s = max(pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(start, tz="UTC"))
            e = min(pd.Timestamp(f"{y}-12-31 23:59", tz="UTC"), now)
            if e <= s:
                continue
            p = {"symbols": symbol, "timeframe": timeframe, "start": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "end": e.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000, "sort": "asc"}
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
    if not crypto and timeframe != "1Day":
        ny = df.index.tz_convert(NY)
        mins = ny.hour * 60 + ny.minute
        df = df[(mins >= 570) & (mins < 960)]          # regular hours only
    return df[df["v"] > 0]


# ---------------- strategy ----------------
def _pivots(h, l, k):
    n = len(h)
    ph, pl = [], []
    for i in range(k, n - k):
        win_h = h[i - k:i + k + 1]
        win_l = l[i - k:i + k + 1]
        if h[i] == win_h.max() and (win_h == h[i]).sum() == 1:
            ph.append(i)
        if l[i] == win_l.min() and (win_l == l[i]).sum() == 1:
            pl.append(i)
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
    atr = _atr(h, l, c)
    n = len(c)
    rows = []
    for d, t0, poc, sw_start, sw_end in _setups(df, k):
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
            if fill_t is None or not np.isfinite(atr[min(fill_t, n - 1)]):
                continue
            for stop_kind, tgt_kind in EXITS:
                stop = sw_start if stop_kind == "swing" else fill_px - d * atr[fill_t - 1]
                risk = d * (fill_px - stop)
                if risk <= 0:
                    continue
                target = (sw_end if tgt_kind == "high" else
                          fill_px + d * risk * (2 if tgt_kind == "2R" else 3))
                if d * (target - fill_px) <= 0:
                    continue
                res = _walk(o, h, l, c, fill_t, d, fill_px, stop, target, entry_kind == "touch")
                if res is None:
                    continue
                r_mult, ret, held = res
                cost_r = 2 * cost_side * fill_px / risk
                rows.append({"time": df.index[fill_t], "bar": fill_t, "dir": d, "entry": entry_kind,
                             "exit": f"{stop_kind}/{tgt_kind}", "k": k,
                             "stop_pct": abs(fill_px - stop) / fill_px, "tgt_pct": abs(target - fill_px) / fill_px,
                             "r": r_mult - cost_r, "ret": ret - 2 * cost_side, "held": held})
    return pd.DataFrame(rows)


def random_baseline(df, tr, cost_side, draws=20, seed=1):
    """Same direction, same stop % and target %, random entry bars: average net R per trade."""
    if tr.empty:
        return tr.assign(r_rand=[])
    rng = np.random.default_rng(seed)
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    n = len(c)
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
