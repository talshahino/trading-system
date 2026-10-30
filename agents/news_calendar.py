"""High-impact economic news calendar, so strategies can be kept out of the market around
news releases (FTMO restricts trading around high-impact news on funded accounts; the user's
rule: no trading from 5 minutes before to 5 minutes after).

Source: a public ForexFactory calendar history (Hugging Face dataset Tropstan/Forex_Factory_Calendar,
2007 to April 2025, MIT license, research use). After the dataset ends, the two biggest US
releases are added by rule: Non-Farm Payrolls (first Friday, 08:30 New York) and FOMC
decisions (14:00 New York, published schedule). The report says how far real coverage goes.
"""
import io

import pandas as pd
import requests

import config

HF = "https://huggingface.co"
REPO = "datasets/Tropstan/Forex_Factory_Calendar"
CACHE = config.CACHE_DIR / "news_calendar_high.csv"
FOMC = ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29",
        "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16",
        "2026-10-28", "2026-12-09"]
CURRENCIES = {"EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"], "USDJPY": ["USD", "JPY"],
              "AUDUSD": ["AUD", "USD"], "USDCAD": ["USD", "CAD"], "XAUUSD": ["USD"], "XAGUSD": ["USD"],
              "WTI": ["USD", "CAD"], "US30": ["USD"], "US100": ["USD"], "US500": ["USD"],
              "GER40": ["EUR"], "UK100": ["GBP"], "JP225": ["JPY"], "BRENT": ["USD"], "NATGAS": ["USD"],
              "COPPER": ["USD", "CNY"], "BTCUSD": ["USD"], "ETHUSD": ["USD"]}


def _download():
    files = requests.get(f"{HF}/api/{REPO}", timeout=60).json().get("siblings", [])
    names = [f["rfilename"] for f in files if f["rfilename"].lower().endswith(".csv")]
    if not names:
        raise RuntimeError("news calendar: no CSV in the dataset")
    frames = []
    for n in names:
        r = requests.get(f"{HF}/{REPO}/resolve/main/{n}", timeout=300)
        r.raise_for_status()
        frames.append(pd.read_csv(io.BytesIO(r.content)))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Impact"].astype(str).str.contains("High", case=False)]
    out = pd.DataFrame({"time": pd.to_datetime(df["DateTime"], utc=True, errors="coerce"),
                        "currency": df["Currency"].astype(str).str.upper().str.strip(),
                        "event": df["Event"].astype(str)}).dropna(subset=["time"])
    return out.drop_duplicates()


def _rule_based(start):
    """NFP and FOMC after the dataset ends (US, high impact)."""
    rows = []
    for m in pd.period_range(start, pd.Timestamp.now(), freq="M"):
        d = m.start_time
        first_fri = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
        rows.append({"time": (first_fri + pd.Timedelta(hours=8, minutes=30)).tz_localize("America/New_York"),
                     "currency": "USD", "event": "Non-Farm Employment Change (rule)"})
    for f in FOMC:
        if pd.Timestamp(f) >= pd.Timestamp(start):
            rows.append({"time": (pd.Timestamp(f) + pd.Timedelta(hours=14)).tz_localize("America/New_York"),
                         "currency": "USD", "event": "FOMC Statement (rule)"})
    df = pd.DataFrame(rows)
    df["time"] = df["time"].dt.tz_convert("UTC")
    return df


def load():
    """(events DataFrame [time UTC, currency, event], date where real data ends)."""
    if CACHE.exists():
        df = pd.read_csv(CACHE, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
    else:
        try:
            df = _download()
            df.to_csv(CACHE, index=False)
        except Exception as e:  # noqa: BLE001 - the backtest still runs, with rule-based news only
            print(f"  news calendar download failed ({e}); using rule-based US news only")
            df = pd.DataFrame(columns=["time", "currency", "event"])
            df["time"] = pd.to_datetime(df["time"], utc=True)
    real_end = df["time"].max() if len(df) else pd.Timestamp("2016-01-01", tz="UTC")
    extra = _rule_based(real_end.tz_convert(None).normalize() + pd.Timedelta(days=1))
    return pd.concat([df, extra], ignore_index=True).sort_values("time"), real_end


def blocked(times, symbol, events, minutes=5):
    """Boolean mask: which timestamps fall within +-minutes of a high-impact release for one of
    the instrument's currencies."""
    cur = CURRENCIES.get(symbol)
    if not cur or events is None or events.empty:
        return pd.Series(False, index=times.index)
    ev = events[events["currency"].isin(cur)]["time"].sort_values().to_numpy()
    if len(ev) == 0:
        return pd.Series(False, index=times.index)
    t = pd.to_datetime(times, utc=True).to_numpy()
    import numpy as np
    idx = np.searchsorted(ev, t)
    w = np.timedelta64(minutes, "m")
    after = np.where(idx < len(ev), ev[np.minimum(idx, len(ev) - 1)] - t, np.timedelta64(10**6, "m"))
    before = np.where(idx > 0, t - ev[np.maximum(idx - 1, 0)], np.timedelta64(10**6, "m"))
    return pd.Series((after <= w) | (before <= w), index=times.index)
