"""Economic-release trading (family 6 of the strategy map): what the market does right after a
high-impact release (rates, jobs, inflation, PMI ...). Same contract as agents/rules_agent.py.

Events: FTMO's own economic calendar (data/ftmo_calendar/history.json, from April 2025): every high-impact
or restricted release that concerns the instrument (its currencies, or FTMO's restriction list).
FTMO rule kept: no trade from 3 minutes before to 3 minutes after a restricted release, and the backtest's
own news filter blocks entries/exits within 5 minutes of a release. So the earliest entry here is the
open of the 5-minute bar that starts 10 minutes after the release.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import ftmo_universe as U
from agents.rules_agent import atr

HIST = Path(__file__).resolve().parent.parent / "data" / "ftmo_calendar" / "history.json"
_EVENTS = {}


def events_for(sym):
    """Release times (UTC) that concern this instrument: high impact or FTMO-restricted."""
    if sym in _EVENTS:
        return _EVENTS[sym]
    from agents import ftmo_calendar as CAL
    try:
        items = json.loads(HIST.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        items = []
    cur = set(U.CURRENCIES.get(sym, []))
    out = set()
    for i in items:
        if i.get("impact") not in ("high",) and not i.get("restriction"):
            continue
        ins = str(i.get("instrument", ""))
        if (i.get("restriction") and CAL.restricted_affects(sym, ins)) or ins in cur:
            out.add(pd.Timestamp(i["date"]).tz_convert("UTC"))
    _EVENTS[sym] = sorted(out)
    return _EVENTS[sym]


def _event_bars(df):
    """Index of the bar that starts exactly at each release (5-minute bars), for this instrument."""
    sym = df.attrs.get("symbol")
    if not sym:
        return []
    idx = df.index
    ks = []
    for t in events_for(sym):
        k = idx.searchsorted(t)
        if k < len(idx) and idx[k] == t:
            ks.append(int(k))
    return ks


def _empty(n):
    z = np.zeros(n, bool)
    return z, z, np.full(n, np.nan), np.full(n, np.nan)


def s_release_momentum(df):
    """Follow the release: if the first 10 minutes after it moved more than 1.5 ATR (5-minute ATR before
    the release), enter in that direction at the 10-minute mark. Stop beyond the opposite end of those 10
    minutes, target 2R, out after an hour."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    long_, short, sl_l, sl_s = _empty(len(c))
    for k0 in _event_bars(df):
        k1 = k0 + 1
        if k1 + 1 >= len(c) or k0 < 20 or df.index[k1] - df.index[k0] != pd.Timedelta(minutes=5):
            continue
        move, ref = c[k1] - o[k0], a[k0 - 1]
        if not np.isfinite(ref) or ref <= 0 or abs(move) < 1.5 * ref:
            continue
        if move > 0:
            long_[k1], sl_l[k1] = True, min(l[k0], l[k1])
        else:
            short[k1], sl_s[k1] = True, max(h[k0], h[k1])
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0), "max_bars": 12}


def s_release_fade(df):
    """Fade an overreaction: the first 5 minutes jumped more than 2.5 ATR and the next 5 minutes already
    gave back at least a third of it -> enter against the jump at the 10-minute mark, stop just beyond the
    spike, target the price before the release, out after 90 minutes."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    long_, short, sl_l, sl_s = _empty(len(c))
    tg_l, tg_s = np.full(len(c), np.nan), np.full(len(c), np.nan)
    for k0 in _event_bars(df):
        k1 = k0 + 1
        if k1 + 1 >= len(c) or k0 < 20 or df.index[k1] - df.index[k0] != pd.Timedelta(minutes=5):
            continue
        jump, ref = c[k0] - o[k0], a[k0 - 1]
        if not np.isfinite(ref) or ref <= 0 or abs(jump) < 2.5 * ref:
            continue
        back = c[k0] - c[k1] if jump > 0 else c[k1] - c[k0]
        if back < abs(jump) / 3:
            continue
        if jump > 0:            # spiked up, now fading: sell
            short[k1], sl_s[k1], tg_s[k1] = True, max(h[k0], h[k1]) + 0.1 * ref, o[k0]
        else:
            long_[k1], sl_l[k1], tg_l[k1] = True, min(l[k0], l[k1]) - 0.1 * ref, o[k0]
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("level", tg_l, tg_s), "max_bars": 18}


def s_release_range_break(df):
    """Post-release range breakout: the first 15 minutes after the release set a range; the first close
    outside it during the next hour enters in that direction. Stop at the other side, target 1.5R, out
    after two hours."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    long_, short, sl_l, sl_s = _empty(len(c))
    for k0 in _event_bars(df):
        k2 = k0 + 2
        if k2 + 13 >= len(c) or df.index[k2] - df.index[k0] != pd.Timedelta(minutes=10):
            continue
        hi, lo = h[k0:k2 + 1].max(), l[k0:k2 + 1].min()
        for k in range(k2 + 1, k2 + 13):
            if c[k] > hi:
                long_[k], sl_l[k] = True, lo
                break
            if c[k] < lo:
                short[k], sl_s[k] = True, hi
                break
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 1.5), "max_bars": 24}


# id: (name, fn, timeframes)
EVENTS = {
    "EV01": ("Release momentum: follow the first 10 minutes, 2R", s_release_momentum, ["5Min"]),
    "EV02": ("Release fade: spike that already gives back, target pre-release price", s_release_fade, ["5Min"]),
    "EV03": ("Post-release 15-minute range breakout, 1.5R", s_release_range_break, ["5Min"]),
}
