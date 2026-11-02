"""FTMO's own economic calendar (ftmo.com/en/calendar, API gw2.ftmo.com/public-api/v1/economic-calendar).

- Restricted events ("restriction": true): on an FTMO Standard account no trade may be opened or
  closed from 2 minutes before to 2 minutes after the release, on the instruments the event names
  ("CAD", "Crude Oil", "USD + US Indices + XAUUSD + DXY", ...).
- Holidays ("impact": "holiday", e.g. "German Bank Holiday" / EUR): that country's exchanges are
  closed, so its indices and stocks (and, for USD, energy and metals) stop trading for the day.
"""
from __future__ import annotations

import pandas as pd

from agents import ftmo_universe as U

API = "https://gw2.ftmo.com/public-api/v1/economic-calendar"
US_INDICES = {"US30", "US100", "US500", "US2000"}
OIL = {"WTI", "BRENT"}


def _fx_currencies(sym):
    code = (U.spec(sym) or {}).get("code", "")
    return set(code.split("/")) if "/" in code else set()


def restricted_affects(sym, instrument: str) -> bool:
    """Does a restricted event on `instrument` (FTMO's text) restrict trading `sym`?"""
    for tok in (x.strip() for x in str(instrument).split("+")):
        if len(tok) == 3 and tok.isupper() and U.CLASS.get(sym) == "fx" and tok in _fx_currencies(sym):
            return True
        if tok == "US Indices" and sym in US_INDICES:
            return True
        if tok == "Crude Oil" and sym in OIL:
            return True
        if tok.replace(" ", "").upper() == sym.upper():
            return True
    return False


def holiday_affects(sym, currency: str, title: str = "") -> bool:
    """Does a bank holiday of `currency` close the market of `sym`? (FX keeps trading.)"""
    if "Daylight" in title:
        return False
    cls = U.CLASS.get(sym)
    if cls in ("index", "stock"):
        return currency in U.CURRENCIES.get(sym, [])
    if cls in ("energy", "metal"):
        return currency == "USD"
    return False


def parse(items):
    """API items -> (restricted [(utc time, instrument)], holidays [(utc date, currency, title)])."""
    restricted, holidays = [], []
    for i in items:
        t = pd.Timestamp(i["date"]).tz_convert("UTC")
        if i.get("restriction"):
            restricted.append((t, i.get("instrument", "")))
        if i.get("impact") == "holiday":
            holidays.append((t.normalize(), i.get("instrument", ""), i.get("title", "")))
    return restricted, holidays


def fetch(date_from, date_to):
    import requests
    r = requests.get(API, timeout=20, params={
        "dateFrom": pd.Timestamp(date_from).strftime("%Y-%m-%dT00:00:00Z"),
        "dateTo": pd.Timestamp(date_to).strftime("%Y-%m-%dT23:59:59Z"), "timezone": "UTC"})
    r.raise_for_status()
    return r.json().get("items", [])
