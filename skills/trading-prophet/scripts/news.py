#!/usr/bin/env python3
"""High-impact news blackout check (the user's rule: no trading N minutes
before/after a high-impact release on a currency the instrument depends on).

News CSV columns: ``time`` (UTC, ``YYYY-MM-DD HH:MM``), ``currency`` (USD, EUR...),
``impact`` (high/medium/low), optional ``title``. The fxmacrodata-calendar or
economic-calendar-fetcher skills from the same repo can produce this file.
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from mt5_deals import Trade, parse_ts

_IMPACT_RANK = {"low": 1, "medium": 2, "high": 3}
_INDEX_CCY = {
    "US30": "USD", "US100": "USD", "US500": "USD", "NAS": "USD", "SPX": "USD", "DJ": "USD",
    "USTEC": "USD", "USOIL": "USD", "WTI": "USD", "UKOIL": "USD", "BRENT": "USD",
    "GER": "EUR", "DE40": "EUR", "DAX": "EUR", "EU50": "EUR", "FRA": "EUR",
    "UK100": "GBP", "FTSE": "GBP", "JP225": "JPY", "JPN": "JPY", "AUS200": "AUD",
}


@dataclass
class NewsEvent:
    ts: dt.datetime
    currency: str
    impact: str
    title: str = ""


def symbol_currencies(symbol: str) -> set[str]:
    s = re.sub(r"[^A-Z0-9]", "", symbol.upper().split(".")[0])
    if len(s) >= 6 and s[:6].isalpha():
        return {s[:3], s[3:6]}  # EURUSD, XAUUSD -> {XAU, USD}
    for prefix, ccy in _INDEX_CCY.items():
        if s.startswith(prefix):
            return {ccy}
    return {"USD"}  # unknown instrument: assume USD news matters (conservative)


def load_news(path: str | Path, min_impact: str = "high", tz: str = "UTC") -> list[NewsEvent]:
    zone = ZoneInfo(tz)
    need = _IMPACT_RANK[min_impact]
    out = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if _IMPACT_RANK.get(r.get("impact", "").strip().lower(), 0) < need:
                continue
            ts = parse_ts(r["time"].replace("-", "."), zone)
            if ts:
                out.append(NewsEvent(ts, r["currency"].strip().upper(),
                                     r["impact"].strip().lower(), r.get("title", "")))
    return sorted(out, key=lambda e: e.ts)


def violations(trades: list[Trade], events: list[NewsEvent], minutes: int) -> list[dict]:
    """Trades whose open or close falls within +-minutes of a relevant event."""
    win = dt.timedelta(minutes=minutes)
    times = [e.ts for e in events]
    found = []
    for t in trades:
        ccys = symbol_currencies(t.symbol)
        for when, label in ((t.open_time, "open"), (t.close_time, "close")):
            lo = bisect.bisect_left(times, when - win)
            hi = bisect.bisect_right(times, when + win)
            for e in events[lo:hi]:
                if e.currency in ccys:
                    found.append({"symbol": t.symbol, "action": label, "trade_time": when,
                                  "event_time": e.ts, "currency": e.currency, "title": e.title})
                    break
    return found
