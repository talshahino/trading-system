"""News agent: pulls historical news (Benzinga feed via Alpaca) and caches it locally."""
import json
import time
import requests
import pandas as pd

import config


def _headers():
    if not (config.ALPACA_KEY and config.ALPACA_SECRET):
        raise RuntimeError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY in .env")
    return {"APCA-API-KEY-ID": config.ALPACA_KEY, "APCA-API-SECRET-KEY": config.ALPACA_SECRET}


def fetch_news(symbols, start, end, use_cache=True):
    """Return a DataFrame: id, time (UTC), headline, summary, source, symbols(list).
    symbols=None pulls ALL news in the period (used to build a survivorship-free universe)."""
    tag = "ALL" if not symbols else '-'.join(sorted(symbols))[:80]
    key = f"news_{tag}_{start}_{end}.json"
    cache_file = config.CACHE_DIR / key
    if use_cache and cache_file.exists():
        items = json.loads(cache_file.read_text())
    else:
        items, token = [], None
        while True:
            params = {
                "start": f"{start}T00:00:00Z",
                "end": f"{end}T23:59:59Z",
                "limit": 50,
                "sort": "asc",
                "include_content": "false",
            }
            if symbols:
                params["symbols"] = ",".join(symbols)
            if token:
                params["page_token"] = token
            try:
                r = requests.get(f"{config.ALPACA_DATA_URL}/v1beta1/news",
                             headers=_headers(), params=params, timeout=30)
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
            items.extend(data.get("news", []))
            token = data.get("next_page_token")
            print(f"  news: {len(items)} items so far", end="\r")
            if not token:
                break
            time.sleep(0.35)  # stay under the free-plan rate limit (200/min)
        print()
        cache_file.write_text(json.dumps(items))

    if not items:
        return pd.DataFrame(columns=["id", "time", "headline", "summary", "source", "symbols"])
    df = pd.DataFrame(items)
    df["time"] = pd.to_datetime(df["created_at"], utc=True)
    df["summary"] = df.get("summary", "").fillna("")
    return df[["id", "time", "headline", "summary", "source", "symbols"]]


def explode_events(news_df, universe):
    """One row per (news, symbol) inside our universe, dropping roundups and duplicates."""
    df = news_df.copy()
    df["n_symbols"] = df["symbols"].apply(len)
    df = df[df["n_symbols"].between(1, config.MAX_SYMBOLS_PER_NEWS)]
    df = df.explode("symbols").rename(columns={"symbols": "symbol"})
    df = df[df["symbol"].isin(universe)]
    # Same headline on the same ticker within 24 hours = one event (keep the earliest
    # publication: re-posts by other outlets carry no new information).
    df["hl_norm"] = df["headline"].str.lower().str.strip()
    df = df.sort_values("time")
    gap = df.groupby(["symbol", "hl_norm"])["time"].diff()
    df = df[gap.isna() | (gap > pd.Timedelta(hours=24))]
    return df.drop(columns=["hl_norm"]).reset_index(drop=True)


def universe_from_news(news_df, min_news=5, max_symbols=300):
    """Survivorship-free universe: every ticker that had stock-specific news IN THE PERIOD,
    including companies that were later delisted, acquired or went bankrupt.
    A list of today's tickers would silently drop exactly those losers."""
    df = news_df[news_df["symbols"].apply(len).between(1, config.MAX_SYMBOLS_PER_NEWS)]
    counts = df.explode("symbols")["symbols"].value_counts()
    counts = counts[counts.index.str.fullmatch(r"[A-Z]{1,5}")]  # plain US tickers
    counts = counts[counts >= min_news]
    return list(counts.head(max_symbols).index)


def inactive_symbols(symbols):
    """Which of these tickers are no longer tradable today (delisted / merged).
    Returns None if the check could not run."""
    try:
        r = requests.get(f"{config.ALPACA_TRADING_URL}/v2/assets", headers=_headers(),
                         params={"status": "inactive", "asset_class": "us_equity"}, timeout=60)
        r.raise_for_status()
        inactive = {a["symbol"] for a in r.json()}
        return sorted(set(symbols) & inactive)
    except Exception as e:  # noqa: BLE001 - informational only
        print(f"  (could not check delisted tickers: {e})")
        return None
