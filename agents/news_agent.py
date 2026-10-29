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


def _fetch_range(symbols, start_iso, end_iso, label):
    """All news between two ISO timestamps, following page tokens."""
    items, token, net_fail = [], None, 0
    while True:
        params = {"start": start_iso, "end": end_iso, "limit": 50, "sort": "asc",
                  "include_content": "false"}
        if symbols:
            params["symbols"] = ",".join(symbols)
        if token:
            params["page_token"] = token
        try:
            r = requests.get(f"{config.ALPACA_DATA_URL}/v1beta1/news",
                             headers=_headers(), params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            net_fail += 1
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
        if len(items) % 5000 < 50:
            print(f"  news {label}: {len(items)} items so far", flush=True)
        if not token:
            return items
        time.sleep(0.3)  # free plan allows 200 requests a minute


def _migrate_legacy(tag):
    """Older versions cached one big file per date range. Split any such file into the
    monthly files used now, so that work is never downloaded twice."""
    for f in config.CACHE_DIR.glob(f"news_{tag}_????-??-??_????-??-??.json"):
        try:
            _, _, a, b = f.stem.split("_", 3)
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            items = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        by_month = {}
        for it in items:
            by_month.setdefault(it["created_at"][:7], []).append(it)
        for m in pd.period_range(a, b, freq="M"):
            covered = m.start_time >= a and m.end_time.normalize() <= b
            out = config.CACHE_DIR / f"news_{tag}_{m}.json"
            if covered and not out.exists():
                out.write_text(json.dumps(by_month.get(str(m), [])))


def fetch_news(symbols, start, end, use_cache=True):
    """Return a DataFrame: id, time (UTC), headline, summary, source, symbols(list).
    symbols=None pulls ALL news in the period (used to build a survivorship-free universe).
    Cached one file per calendar month, so a later run with a newer end date only
    downloads the months it has not seen."""
    tag = "ALL" if not symbols else "-".join(sorted(symbols))[:80]
    if use_cache:
        _migrate_legacy(tag)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    items = []
    for m in pd.period_range(start_ts, end_ts, freq="M"):
        m_start, m_end = m.start_time, m.end_time.normalize()
        full_month = start_ts <= m_start and m_end <= end_ts and m_end < today
        f = config.CACHE_DIR / f"news_{tag}_{m}.json"
        if use_cache and full_month and f.exists():
            month_items = json.loads(f.read_text())
            print(f"  news {m}: {len(month_items)} items (cached)", flush=True)
        else:
            a, b = max(m_start, start_ts), min(m_end, end_ts)
            month_items = _fetch_range(symbols, f"{a:%Y-%m-%d}T00:00:00Z", f"{b:%Y-%m-%d}T23:59:59Z", str(m))
            print(f"  news {m}: {len(month_items)} items (downloaded)", flush=True)
            if full_month:
                f.write_text(json.dumps(month_items))
        items.extend(month_items)

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
