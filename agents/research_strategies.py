"""Strategies with long published evidence (research only, 08.11.2026; the user chose all three).
Rules, universes and the judging line were written here and in GATE.md 7. BEFORE any test; not to be tuned.
Same contract as agents/rules_agent.py: signal at the bar close, entry at the next bar's open. FTMO needs
a stop on every trade, so each has a wide safety stop the sources do not have.

RS01  Time-series momentum (Moskowitz, Ooi & Pedersen 2012, "Time Series Momentum"): at every month end,
      long if the close is above the close of 12 months earlier, short if below; held to the next month
      end, closed when the sign turns. Universe: every FTMO instrument except single shares.
      Safety stop 5 x ATR(14).
RS02  Cross-sectional stock momentum (Jegadeesh & Titman 1993): at every month end rank the FTMO shares on
      the return from 12 months to 1 month ago; hold the top 10 long to the next month end, out when a
      share leaves the top 10. Universe: the FTMO shares. Safety stop 5 x ATR(14).
RS03  RSI(2) pullback on stock indices (Connors & Alvarez, "Short Term Trading Strategies That Work", 2008):
      close above its 200-day average and RSI(2) under 10 -> buy; out on a close above the 5-day average.
      Universe: the FTMO stock indices. Safety stop 3 x ATR(14).
All on daily bars.
"""
import numpy as np
import pandas as pd

from agents.rules_agent import rsi, sma
from agents import ftmo_universe as U

DAY = ["1Day"]
TOP = 10


def _month_end(df):
    m = (df.index.tz_convert("UTC").tz_localize(None) if df.index.tz is not None else df.index).to_period("M").to_numpy()
    me = np.zeros(len(df), bool)
    me[:-1] = m[1:] != m[:-1]
    return me


def _close_ago(df, days):
    """Close of the last bar at least `days` calendar days before each bar (NaN when there is none)."""
    c = df["c"].to_numpy(dtype=float)
    idx = df.index
    pos = idx.searchsorted(idx - pd.Timedelta(days=days), side="right") - 1
    out = np.where(pos >= 0, c[np.clip(pos, 0, len(c) - 1)], np.nan)
    out[(idx - idx[0]) < pd.Timedelta(days=days)] = np.nan
    return out


def _none(n):
    z = np.zeros(n, bool)
    return z, z, {"stop": ("atr", 5.0), "target": None}


def s_tsmom12(df):
    sym = df.attrs.get("symbol")
    if sym and U.CLASS.get(sym) == "stock":
        return _none(len(df))
    c = df["c"].to_numpy(dtype=float)
    past = _close_ago(df, 365)
    me = _month_end(df) & np.isfinite(past)
    up, dn = c > past, c < past
    # a sign change closes the trade and opens the other side at the same open ("opposite")
    return me & up, me & dn & ~up, {"stop": ("atr", 5.0), "target": None, "opposite": True}


_RANK = {}


def _ranks(start, end):
    """Month-end rank table of every FTMO share on its 12-1 month return (only closed bars are used)."""
    key = (start, end)
    if key in _RANK:
        return _RANK[key]
    from agents import intermarket_strategies as IM
    mom = {}
    for s in U.GROUPS["stock"]:
        try:
            d = IM.PARTNER_BARS(s, "1Day", start, end)
        except Exception:  # noqa: BLE001 - a share without data is simply not ranked
            continue
        if d is None or len(d) < 300:
            continue
        r = pd.Series(_close_ago(d, 30) / _close_ago(d, 365) - 1, index=d.index)
        r = r[_month_end(d)]
        mom[s] = r.groupby(r.index.tz_convert("UTC").tz_localize(None).to_period("M")).last()
    tab = pd.DataFrame(mom)
    _RANK[key] = tab.rank(axis=1, ascending=False)
    return _RANK[key]


def s_xsmom(df):
    sym = df.attrs.get("symbol")
    if not sym or U.CLASS.get(sym) != "stock":
        return _none(len(df))
    start, end = df.index[0].strftime("%Y-%m-%d"), (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tab = _ranks(start, end)
    if sym not in tab:
        return _none(len(df))
    me = _month_end(df)
    per = df.index.tz_convert("UTC").tz_localize(None).to_period("M")
    rk = tab[sym].reindex(per).to_numpy()
    top = me & (rk <= TOP)
    return top, np.zeros(len(df), bool), {"stop": ("atr", 5.0), "target": None, "exit_long": me & ~(rk <= TOP)}


def s_rsi2_index(df):
    sym = df.attrs.get("symbol")
    if sym and U.CLASS.get(sym) != "index":
        z = np.zeros(len(df), bool)
        return z, z, {"stop": ("atr", 3.0), "target": None}
    c = df["c"].to_numpy(dtype=float)
    L = (c > sma(c, 200)) & (rsi(c, 2) < 10)
    return L, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": c > sma(c, 5)}


RESEARCH = {
    "RS01": ("Time-series momentum 12 months, monthly (Moskowitz, Ooi & Pedersen 2012)", s_tsmom12, DAY),
    "RS02": ("Share momentum 12-1 months, top 10, monthly (Jegadeesh & Titman 1993)", s_xsmom, DAY),
    "RS03": ("RSI(2) pullback on indices above the 200-day average (Connors & Alvarez 2008)", s_rsi2_index, DAY),
}
