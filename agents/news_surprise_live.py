"""EV08 live - trading the release's surprise (actual vs forecast) in bot 7 (card 00553aab28e5, 15.11.2026).

The rule chosen on 2025 in agents/news_surprise.py ("dew", Dewonggo's rule): a US Unemployment Claims / home sales
number (the leading US number of its minute, the only US number with actual + forecast in that minute, the surprise
|actual - forecast| at least the title's median on 2025) worse for the US economy than expected = buy BTC / the US
indices / gold (beta +1), better = sell; entered D minutes after the release, stop S x the 5-minute ATR(14) before the
release, out after H minutes, no target.

Two things the backtest knows and the live bot must not assume:
  * the actual must have been in the bot's calendar BEFORE the entry - `seen` maps (release minute, title) to the
    time the bot first read the item with its actual (kept on disk by live/news_bot.py, so a restart never makes a
    past release look known in time); a number read later -> no trade (never a trade entered in the past);
  * the entry candle's open is not known when the signal is given (the bot reads closed candles), so the stop is
    measured from the close of the candle before it (the open that follows is the same price up to a tick).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import news_surprise as NS
from agents import rules_agent as R

ROOT = Path(__file__).resolve().parent.parent
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
BETA = {"BTCUSD": 1, "US100": 1, "US500": 1, "US30": 1, "XAUUSD": 1, "EURUSD": 1, "USDJPY": -1, "WTI": -1}
_MED = None


def medians():
    """The median |actual - forecast| of each US title on 2025 only (as the study), from the full calendar."""
    global _MED
    if _MED is None:
        _MED = {}
        try:
            items = json.loads(NS.CAL_FULL.read_text(encoding="utf-8")).get("items") or []
        except Exception:  # noqa: BLE001
            items = []
        m = {}
        for i in items:
            if str(i.get("instrument", "")).split(" ")[0] not in ("USD", "Crude"):
                continue
            a, f = NS.NF._num(i.get("actual")), NS.NF._num(i.get("forecast"))
            if a is None or f is None or a == f:
                continue
            if pd.Timestamp(i["date"]).tz_convert("UTC") < SPLIT:
                m.setdefault(i["title"], []).append(abs(a - f))
        _MED = {k: float(np.median(v)) for k, v in m.items() if v}
    return _MED


def releases(items):
    """Release minute -> {group, title, bad, big, alone} exactly as news_surprise.us_releases, the medians from 2025."""
    med = medians()
    by_t = {}
    for i in items:
        if str(i.get("instrument", "")).split(" ")[0] not in ("USD", "Crude"):
            continue
        a, f = NS.NF._num(i.get("actual")), NS.NF._num(i.get("forecast"))
        if a is None or f is None or a == f:
            continue
        t = pd.Timestamp(i["date"]).tz_convert("UTC")
        by_t.setdefault(t, []).append({"title": i["title"], "group": NS.group(i["title"]),
                                       "bad": NS.bad_sign(i["title"], a, f), "abs": abs(a - f), "impact": i.get("impact")})
    imp = {"high": 0, "medium": 1, "low": 2}
    out = {}
    for t, xs in by_t.items():
        lead = min(xs, key=lambda x: (imp.get(x["impact"], 3), NS.GROUP_PRIO.index(x["group"])))
        m = med.get(lead["title"])
        out[t] = {"group": lead["group"], "title": lead["title"], "bad": lead["bad"],
                  "big": int(m is not None and lead["abs"] >= m), "alone": int(len(xs) == 1)}
    return out


def seen_key(t, title):
    return f"{pd.Timestamp(t).tz_convert('UTC').isoformat()}|{title}"


def live_surprise(sym, items, seen, D=10, S=1.5, H=120, groups=("claims", "home"), size="big", alone=True, side=1):
    """The strategy for agents/rules_agent.run on 1-minute candles. items / seen: callables returning the bot's calendar
    items and {seen_key: first time the actual was read (ISO)}."""
    beta = BETA.get(sym, 1)

    def fn(df):
        o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
        n, idx = len(c), df.index
        long_, short = np.zeros(n, bool), np.zeros(n, bool)
        sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
        if n < 200:
            return long_, short, {"stop": ("level", sl_l, sl_s), "target": None, "max_bars": int(H)}
        d5 = df.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        a5 = pd.Series(R.atr(d5["h"].to_numpy(), d5["l"].to_numpy(), d5["c"].to_numpy(), 14), index=d5.index)
        sn = seen() if callable(seen) else (seen or {})
        for t, x in releases(items() if callable(items) else (items or [])).items():
            if x["group"] not in groups or (size == "big" and not x["big"]) or (alone and not x["alone"]):
                continue
            s_at = sn.get(seen_key(t, x["title"]))
            if not s_at or pd.Timestamp(s_at) > t + pd.Timedelta(minutes=D, seconds=65):
                continue                                    # the number was not in the calendar before the entry
            ka = a5.index.searchsorted(t) - 1                  # the last 5-minute candle closed before the release
            if ka < 14:
                continue
            atr = float(a5.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            ke = idx.searchsorted(t + pd.Timedelta(minutes=D))
            if ke < 1 or ke > n or (ke < n and idx[ke] - t > pd.Timedelta(minutes=D + 2)):
                continue
            dr = int(x["bad"] * beta * side)
            stop = c[ke - 1] - dr * S * atr
            if dr > 0:
                long_[ke - 1], sl_l[ke - 1] = True, stop
            elif dr < 0:
                short[ke - 1], sl_s[ke - 1] = True, stop
        return long_, short, {"stop": ("level", sl_l, sl_s), "target": None, "max_bars": int(H)}
    return fn
