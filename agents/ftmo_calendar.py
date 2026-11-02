"""FTMO's own economic calendar (ftmo.com/en/calendar, API gw2.ftmo.com/public-api/v1/economic-calendar).

- Restricted events ("restriction": true): on an FTMO Standard account no trade may be opened or
  closed from 2 minutes before to 2 minutes after the release, on the instruments the event names
  ("CAD", "Crude Oil", "USD + US Indices + XAUUSD + DXY", ...).
- Holidays ("impact": "holiday", e.g. "German Bank Holiday" / EUR): that country's exchanges are
  closed, so its indices and stocks (and, for USD, energy and metals) stop trading for the day.
"""
from __future__ import annotations

import numpy as np
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


def masks(df, sym, tf_minutes, restricted, holidays):
    """(exit mask, no-entry mask) over df's bars, the same in the backtest and in the live replay.
    Restricted event at t: an open position closes at the close of the last bar ending by t - 3.5 min,
    and no trade enters on a bar that overlaps t - 3.5 min .. t + 3 min.
    Holiday of the instrument's market (UTC day D): close at the last bar ending by D - 30 min, no entries on D."""
    start = df.index
    end = start + pd.Timedelta(minutes=tf_minutes)
    ex = np.zeros(len(df), bool)
    noent = np.zeros(len(df), bool)
    spans = [(t - pd.Timedelta(seconds=210), t + pd.Timedelta(minutes=3)) for t, ins in restricted if restricted_affects(sym, ins)]
    spans += [(d - pd.Timedelta(minutes=30), d + pd.Timedelta(days=1)) for d, cur, ti in holidays if holiday_affects(sym, cur, ti)]
    for a, b in spans:
        if b < start[0] or a > end[-1]:
            continue
        k = end.searchsorted(a, side="right") - 1          # last bar ending by a
        if 0 <= k < len(df) and start[min(k + 1, len(df) - 1)] < b:
            ex[k] = True
        noent[end.searchsorted(a, side="right"):start.searchsorted(b, side="left")] = True
    return ex, noent


def blocked(fn, noent):
    """The strategy, but no signal whose entry bar (the next one) is in `noent`."""
    def g(df):
        long_, short, plan = fn(df)
        blk = np.zeros(len(df), bool)
        m = min(len(noent), len(df)) - 1
        blk[:m] = noent[1:m + 1]
        return np.asarray(long_, bool) & ~blk, np.asarray(short, bool) & ~blk, plan
    return g
