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
EU_COUNTRY = ("German", "French", "Italian", "Spanish", "Dutch")
EU_HOME = {"GER40": "German", "EU50": "German", "ADSGn": "German", "ALVG": "German", "BAYGn": "German",
           "DBKGn": "German", "SIEGn": "German", "VOWG_p": "German", "BMW": "German", "MBG": "German",
           "FRA40": "French", "AIRF": "French", "LVMH": "French", "TTE": "French",
           "RACE": "Italian", "SPN35": "Spanish", "IBE": "Spanish", "SAN": "Spanish", "N25": "Dutch", "ASML": "Dutch"}


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
    """Does a bank holiday of `currency` alone close the market of `sym`? (FX: see closed_days.)"""
    if "Daylight" in title:
        return False
    cls = U.CLASS.get(sym)
    if cls in ("index", "stock"):
        if currency not in U.CURRENCIES.get(sym, []):
            return False
        country = next((c for c in EU_COUNTRY if title.startswith(c)), None)
        if currency == "EUR" and country:          # "German Bank Holiday" closes German markets only
            return EU_HOME.get(sym) == country
        return True
    if cls in ("energy", "metal"):
        return currency == "USD"
    return False


def closed_days(sym, holidays):
    """UTC days on which `sym`'s market is closed for a holiday, from FTMO's calendar.
    Indices/stocks: a holiday of their country. Energy/metals: a US holiday.
    FX pairs: both currencies on holiday the same day (Christmas, New Year, Good Friday for EURGBP...)."""
    by_day, titles = {}, {}
    for d, cur, title in holidays:
        if "Daylight" not in title:
            by_day.setdefault(d, set()).add(cur)
            titles.setdefault(d, []).append((cur, title))
    out = []
    fx = _fx_currencies(sym) if U.CLASS.get(sym) == "fx" else set()
    for d, curs in sorted(by_day.items()):
        if fx:
            if fx <= curs:
                out.append(d)
        elif any(holiday_affects(sym, c, t) for c, t in titles[d]):
            out.append(d)
    return out


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
    """(exit mask, no-entry mask) over df's bars for HOLIDAYS, the same in the backtest and in the
    live replay: the instrument's market is closed on UTC day D -> close at the last bar ending by
    D - 30 min, no entries on D. (Restricted events are handled minute by minute: apply_news.)"""
    start = df.index
    end = start + pd.Timedelta(minutes=tf_minutes)
    ex = np.zeros(len(df), bool)
    noent = np.zeros(len(df), bool)
    for d in closed_days(sym, holidays):
        a, b = d - pd.Timedelta(minutes=30), d + pd.Timedelta(days=1)
        if b < start[0] or a > end[-1]:
            continue
        k = end.searchsorted(a, side="right") - 1          # last bar ending by a
        if 0 <= k < len(df) and start[min(k + 1, len(df) - 1)] < b:
            ex[k] = True
        noent[end.searchsorted(a, side="right"):start.searchsorted(b, side="left")] = True
    return ex, noent


BEFORE = pd.Timedelta(minutes=3)      # the user's rule: no open position from 3 minutes before
AFTER = pd.Timedelta(minutes=3)       # ... to 3 minutes after a restricted event on the instrument


def apply_news(trades, minute_open, restricted):
    """Restricted events, minute by minute, on a trade list from the backtest engine:
    - a trade open 3 minutes before an event on its instrument is closed then (1-minute price) and
      is not reopened;
    - a trade whose entry falls between 3 minutes before and 3 minutes after the event waits: it
      opens 3 minutes after the event, if it is still running (exit later) and the price is still
      between its stop and its target; otherwise it is dropped.
    minute_open(symbol, utc time) -> the 1-minute open price at that time (or the next minute), or None.
    The R of a changed trade is recomputed with the same stop and target; its costs (in price) are kept."""
    out = []
    for r in trades.to_dict("records"):
        ev = sorted(t for t, ins in restricted if restricted_affects(r["symbol"], ins))
        d = r["dir"]
        e0, st, tg = r["entry_px"], r["stop_px"], r["target_px"]
        gross = d * (r["exit_px"] - e0) / abs(e0 - st) if abs(e0 - st) > 0 else 0.0
        cost_px = (gross - r["r"]) * abs(e0 - st)
        drop = False
        for t in ev:
            a, b = t - BEFORE, t + AFTER
            if r["exit_time"] <= a or r["time"] > b:
                continue
            if a <= r["time"] <= b:                       # entry inside the window: wait for b
                if r["exit_time"] <= b:
                    drop = True
                    break
                px = minute_open(r["symbol"], b)
                if px is None or not (min(st, tg) < px < max(st, tg)):
                    drop = True
                    break
                r["time"], r["entry_px"], r["reason_news"] = b, px, "כניסה נדחתה בגלל הודעה"
                e0 = px
                continue
            px = minute_open(r["symbol"], a)              # open at a: close 3 minutes before the event
            if px is None:
                continue
            r["exit_time"], r["exit_px"], r["reason"] = a, px, "נסגרה לפני הודעה מוגבלת"
            if r.get("free_time") is not None and r["free_time"] == r["free_time"] and r["free_time"] > a:
                r["free_time"] = None
            break
        if drop:
            continue
        risk = abs(r["entry_px"] - st)
        if risk > 0:
            r["r"] = d * (r["exit_px"] - r["entry_px"]) / risk - cost_px / risk
        out.append(r)
    return pd.DataFrame(out, columns=list(trades.columns) + (["reason_news"] if "reason_news" not in trades.columns else []))


def blocked(fn, noent):
    """The strategy, but no signal whose entry bar (the next one) is in `noent`."""
    def g(df):
        long_, short, plan = fn(df)
        blk = np.zeros(len(df), bool)
        m = min(len(noent), len(df)) - 1
        blk[:m] = noent[1:m + 1]
        return np.asarray(long_, bool) & ~blk, np.asarray(short, bool) & ~blk, plan
    return g
