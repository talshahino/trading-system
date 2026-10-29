"""Latency agent: how much of the move is lost for every second we wait?

Minute bars can't answer this (they only say where the price was at the start of each
minute), so for the strongest news items this agent downloads every single trade in a
short window around the publication time and reads the price at:
  0 s  = the last trade at the moment the news was published (theoretical best: nobody
         can trade at this price after reading the news; it is the upper bound)
  1, 3, 10, 30, 60 s after publication = first trade at or after that moment.
All latencies exit at the SAME clock time (publication + 60 s + h minutes, i.e. the exit of
the main backtest, from minute bars), so the only difference between rows is how late we
came in.

Only prices are cached (one line per event), never the raw trades.
"""
import json
import time

import numpy as np
import pandas as pd
import requests

import config
from agents.news_agent import _headers

NY = "America/New_York"
LATENCIES = [0, 1, 3, 10, 30, 60]
EXIT_MIN = [1, 5, 15, 60]
_CACHE = config.CACHE_DIR / "latency_prices.jsonl"


def _load():
    out = {}
    if _CACHE.exists():
        for line in open(_CACHE, encoding="utf-8"):
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec["px"]
            except (ValueError, KeyError):
                continue
    return out


def _trades(symbol, start, end):
    """All trades between two UTC timestamps -> (times as int ns, prices)."""
    ts, px, token, fails = [], [], None, 0
    while True:
        params = {"symbols": symbol, "start": start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                  "end": end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "limit": 10000,
                  "feed": config.BAR_FEED, "sort": "asc"}
        if token:
            params["page_token"] = token
        try:
            r = requests.get(f"{config.ALPACA_DATA_URL}/v2/stocks/trades",
                             headers=_headers(), params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            fails += 1
            if fails > 10:
                raise
            time.sleep(5)
            continue
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        data = r.json()
        for t in (data.get("trades") or {}).get(symbol, []):
            ts.append(pd.Timestamp(t["t"]).value)
            px.append(float(t["p"]))
        token = data.get("next_page_token")
        if not token:
            break
        time.sleep(0.3)
    return np.array(ts, dtype=np.int64), np.array(px)


def _entry_prices(symbol, t_news):
    """{latency: price} or None if there was no trading around the news."""
    t_news = pd.Timestamp(t_news).tz_convert("UTC")
    ts, px = _trades(symbol, t_news - pd.Timedelta(seconds=120), t_news + pd.Timedelta(seconds=75))
    if len(ts) == 0:
        return None
    out = {}
    t0 = t_news.value
    i = np.searchsorted(ts, t0, side="right") - 1          # last trade at or before publication
    if i >= 0:
        out[0] = px[i]
    for lat in LATENCIES[1:]:
        j = np.searchsorted(ts, t0 + lat * 1_000_000_000, side="left")
        # Must actually trade within 15 s of the target moment, otherwise the fill is unknown.
        if j < len(ts) and ts[j] - (t0 + lat * 1_000_000_000) <= 15_000_000_000:
            out[lat] = px[j]
    return out or None


def _price_at(bars, when):
    """Close of the last minute bar that has fully finished by `when` (same session only)."""
    k = bars.index.searchsorted(when - pd.Timedelta(minutes=1), side="right") - 1
    if k < 0:
        return np.nan
    t = bars.index[k].tz_convert(NY)
    w = when.tz_convert(NY)
    if t.date() != w.date() or w.hour * 60 + w.minute > 16 * 60:
        return np.nan
    return bars["c"].iat[k]


def latency_study(results, bars_by_symbol, min_abs_score=0.3):
    """results = intraday results (one row per event). Returns (table, info)."""
    d = results[(results["skip_reason"].isna()) & (results["score"].abs() >= min_abs_score)]
    d = d.reindex(d["score"].abs().sort_values(ascending=False).index)   # strongest first
    cache = _load()
    budget_end = time.time() + config.LATENCY_MAX_MINUTES * 60
    fetched, stopped = 0, False
    rows = []
    with open(_CACHE, "a", encoding="utf-8") as f:
        for n, ev in enumerate(d.itertuples(index=False), 1):
            key = f"{ev.id}|{ev.symbol}"
            if key not in cache:
                if time.time() > budget_end:
                    stopped = True
                    continue
                try:
                    px = _entry_prices(ev.symbol, ev.time)
                except requests.HTTPError as e:
                    print(f"    latency: {ev.symbol} skipped ({e})")
                    continue
                cache[key] = {str(k): v for k, v in px.items()} if px else None
                f.write(json.dumps({"key": key, "px": cache[key]}) + "\n")
                fetched += 1
                if fetched % 200 == 0:
                    f.flush()
                    print(f"    latency: {n}/{len(d)} events")
            px = cache.get(key)
            bars = bars_by_symbol.get(ev.symbol)
            if not px or bars is None or bars.empty:
                continue
            t_news = pd.Timestamp(ev.time).tz_convert("UTC")
            direction = np.sign(ev.score)
            cost = ev.cost_bps / 1e4
            for h in EXIT_MIN:
                exit_px = _price_at(bars, t_news + pd.Timedelta(minutes=h) + pd.Timedelta(seconds=60))
                if not np.isfinite(exit_px):
                    continue
                for lat in LATENCIES:
                    p = px.get(str(lat))
                    if p:
                        rows.append({"lat": lat, "h": h, "strong": abs(ev.score) >= 0.5,
                                     "gross": direction * (exit_px / p - 1), "cost": cost})
    keys = [f"{e.id}|{e.symbol}" for e in d.itertuples(index=False)]
    info = {"events": len(d), "measured": sum(1 for k in keys if cache.get(k)),
            "stopped": stopped, "min_abs_score": min_abs_score}
    if not rows:
        return pd.DataFrame(), info
    df = pd.DataFrame(rows)
    out = []
    for (lat, h), g in df.groupby(["lat", "h"]):
        net = g["gross"] - g["cost"]
        s = g[g["strong"]]
        sd = net.std(ddof=1)
        out.append({"lat": lat, "h": h, "events": len(g),
                    "gross_bps": g["gross"].mean() * 1e4, "net_bps": net.mean() * 1e4,
                    "t_stat": net.mean() / (sd / np.sqrt(len(g))) if sd > 0 else np.nan,
                    "strong_events": len(s),
                    "strong_net_bps": (s["gross"] - s["cost"]).mean() * 1e4 if len(s) else np.nan})
    return pd.DataFrame(out), info
