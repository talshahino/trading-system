"""Market-data agent: pulls 1-minute bars from Alpaca, one cached file per symbol per month."""
import time
import requests
import pandas as pd

import config
from agents.news_agent import _headers


def _fetch_range(symbol, start, end, timeframe="1Min", adjustment="raw"):
    bars, token = [], None
    while True:
        config.check_time_budget()
        params = {
            "symbols": symbol, "timeframe": timeframe,
            "start": start, "end": end, "limit": 10000,
            "adjustment": adjustment, "feed": config.BAR_FEED, "sort": "asc",
        }
        if token:
            params["page_token"] = token
        try:
            r = requests.get(f"{config.ALPACA_DATA_URL}/v2/stocks/bars",
                         headers=_headers(), params=params, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            net_fail = locals().get('net_fail', 0) + 1
            if net_fail > 30:  # ~5 minutes of no network: give up with a clear error
                raise
            time.sleep(10)
            continue
        if r.status_code == 429:
            time.sleep(3)
            continue
        r.raise_for_status()
        data = r.json()
        bars.extend((data.get("bars") or {}).get(symbol, []))
        token = data.get("next_page_token")
        if not token:
            break
        time.sleep(0.35)
    return bars


def _clip_end(iso):
    """The free data plan refuses SIP data from the last 15 minutes (HTTP 403), so never ask
    for anything newer than 20 minutes ago (this also covers end dates in the future)."""
    latest = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    return min(pd.Timestamp(iso), latest).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_bars(symbol, start, end):
    """Return DataFrame indexed by UTC minute with columns o,h,l,c,v."""
    months = pd.period_range(start=start, end=end, freq="M")
    frames = []
    for m in months:
        f = config.CACHE_DIR / f"bars_{symbol}_{m}_{config.BAR_FEED}.csv.gz"
        # Only months that are completely over are cached; the current month is re-fetched.
        month_over = m.end_time.tz_localize("UTC") < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
        if f.exists() and month_over:
            df = pd.read_csv(f, parse_dates=["t"])
        else:
            m_start = m.start_time.strftime("%Y-%m-%dT00:00:00Z")
            m_end = _clip_end((m.end_time.floor("D")).strftime("%Y-%m-%dT23:59:59Z"))
            if pd.Timestamp(m_end) <= pd.Timestamp(m_start):
                continue
            raw = _fetch_range(symbol, m_start, m_end)
            df = pd.DataFrame(raw, columns=["t", "o", "h", "l", "c", "v", "n", "vw"])
            if month_over:
                df.to_csv(f, index=False)
            df["t"] = pd.to_datetime(df["t"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df.drop_duplicates("t").sort_values("t").set_index("t")[["o", "h", "l", "c", "v"]]


def get_daily_bars(symbol, start, end):
    """Daily bars, split/dividend adjusted, indexed by New York session date (no time zone)."""
    f = config.CACHE_DIR / f"daily_{symbol}_{start}_{end}_{config.BAR_FEED}.csv.gz"
    if f.exists():
        df = pd.read_csv(f)
    else:
        raw = _fetch_range(symbol, f"{start}T00:00:00Z", _clip_end(f"{end}T23:59:59Z"), "1Day", "all")
        df = pd.DataFrame(raw, columns=["t", "o", "h", "l", "c", "v", "n", "vw"])
        df.to_csv(f, index=False)
    if df.empty:
        return df
    df["t"] = (pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
               .dt.normalize().dt.tz_localize(None))
    return df.drop_duplicates("t").sort_values("t").set_index("t")[["o", "h", "l", "c", "v"]]
