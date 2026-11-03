"""Relations between instruments (family 8 of the strategy map). Each strategy trades ONE instrument,
using a related one (its "partner") as information: gold with silver, an index with its sister index,
a share with its home index, a currency pair with its twin.

Same contract as agents/rules_agent.py. The instrument and timeframe come from df.attrs ("symbol",
"tf"), set by run_rules.py; the partner's bars come from PARTNER_BARS (the backtest uses the normal
data loader; the live bot can plug in MetaTrader). The partner is aligned on the instrument's own bar
times and only bars already closed are used.
"""
import numpy as np
import pandas as pd

from agents.rules_agent import atr, ema

US_STOCKS_INDEX, EU_STOCKS_INDEX = "US100", "EU50"
GER = {"ADSGn", "ALVG", "BAYGn", "BMW", "DBKGn", "MBG", "SIEGn", "VOWG_p"}
EU_OTHER = {"AIRF", "IBE", "SAN", "LVMH", "RACE", "TTE", "AZN"}
PARTNER = {
    # indices
    "US100": "US500", "US30": "US500", "US2000": "US500", "US500": "US100", "AUS200": "US500", "JP225": "US100",
    "HK50": "US100", "GER40": "EU50", "EU50": "GER40", "FRA40": "EU50", "SPN35": "EU50", "UK100": "EU50", "N25": "EU50",
    "DXY": "EURUSD",
    # currencies
    "EURUSD": "GBPUSD", "GBPUSD": "EURUSD", "AUDUSD": "NZDUSD", "NZDUSD": "AUDUSD", "USDCHF": "EURUSD", "USDCAD": "WTI",
    "USDJPY": "EURJPY", "EURJPY": "USDJPY", "GBPJPY": "EURJPY", "EURGBP": "EURUSD",
    # metals, energy, crypto
    "XAUUSD": "XAGUSD", "XAGUSD": "XAUUSD", "XPTUSD": "XAUUSD", "COPPER": "US500",
    "WTI": "BRENT", "BRENT": "WTI", "NATGAS": "WTI",
    "BTCUSD": "ETHUSD", "ETHUSD": "BTCUSD", "SOL/USD": "ETHUSD", "XRP/USD": "BTCUSD",
}


def partner_of(sym):
    if sym in PARTNER:
        return PARTNER[sym]
    if sym in GER:
        return "GER40"
    if sym in EU_OTHER:
        return EU_STOCKS_INDEX
    return US_STOCKS_INDEX            # every other share: the Nasdaq 100


_CACHE = {}


def _default_loader(sym, tf, start, end):
    from agents import poc_agent as P
    return P.bars(sym, tf, start, end)


PARTNER_BARS = _default_loader


def _partner_close(df):
    """The partner's close on the instrument's bar times (NaN where the partner has no bar yet)."""
    sym, tf = df.attrs.get("symbol"), df.attrs.get("tf")
    if not sym or not tf:
        return None
    p = partner_of(sym)
    start, end = df.index[0].strftime("%Y-%m-%d"), (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    key = (p, tf, start, end)
    if key not in _CACHE:
        try:
            _CACHE[key] = PARTNER_BARS(p, tf, start, end)
        except Exception:  # noqa: BLE001 - no partner data: the strategy simply gives no signals
            _CACHE[key] = None
    pdf = _CACHE[key]
    if pdf is None or len(pdf) == 0:
        return None
    pc = pdf["c"].reindex(df.index.union(pdf.index)).ffill(limit=3).reindex(df.index)
    return pc.to_numpy(dtype=float)


def _none(n):
    z = np.zeros(n, bool)
    return z, z, {"stop": ("atr", 2.0), "target": None}


def s_pair_spread_revert(df):
    """Spread reversion (pairs logic, one leg): spread = log(price) - beta x log(partner price), beta
    from the last 200 bars. When the spread is 2 standard deviations (100 bars) below its mean the
    instrument is cheap against its partner: buy; 2 above: sell. Exit when the spread is back at its
    mean, stop 2.5 ATR, at most 60 bars."""
    c = df["c"].to_numpy(dtype=float)
    pc = _partner_close(df)
    if pc is None:
        return _none(len(c))
    lx, ly = pd.Series(np.log(c)), pd.Series(np.log(pc))
    beta = lx.rolling(200, min_periods=150).cov(ly) / ly.rolling(200, min_periods=150).var()
    spread = lx - beta * ly
    z = ((spread - spread.rolling(100, min_periods=80).mean()) / spread.rolling(100, min_periods=80).std()).to_numpy()
    corr = lx.diff().rolling(200, min_periods=150).corr(ly.diff()).to_numpy()
    related = np.abs(corr) > 0.5                      # only while the two really move together
    long_, short = related & (z < -2), related & (z > 2)
    return long_, short, {"stop": ("atr", 2.5), "target": None, "max_bars": 60,
                          "exit_long": z >= 0, "exit_short": z <= 0}


def s_lead_lag(df):
    """Lead-lag catch-up: the partner just made a big move (bar return over 1.5 standard deviations)
    while this instrument barely moved (under 0.3 sd): enter in the partner's direction, expecting it
    to follow. Only when the two are positively correlated. Stop 1.5 ATR, target 1.5R, at most 3 bars."""
    c = df["c"].to_numpy(dtype=float)
    pc = _partner_close(df)
    if pc is None:
        return _none(len(c))
    rx, ry = pd.Series(c).pct_change(), pd.Series(pc).pct_change()
    sx, sy = rx.rolling(100, min_periods=80).std().shift(1), ry.rolling(100, min_periods=80).std().shift(1)
    corr = rx.rolling(200, min_periods=150).corr(ry).to_numpy()
    big_up, big_dn = (ry > 1.5 * sy).to_numpy(), (ry < -1.5 * sy).to_numpy()
    calm = (rx.abs() < 0.3 * sx).to_numpy()
    ok = corr > 0.5
    return ok & big_up & calm, ok & big_dn & calm, {"stop": ("atr", 1.5), "target": ("R", 1.5), "max_bars": 3}


def s_relative_strength(df):
    """Relative strength against the partner: the ratio price / partner crosses above its 50-bar EMA
    while the instrument itself is above its EMA50 -> buy (it starts to beat its market); the mirror
    sells. Stop 2 ATR, ATR trailing stop 3, exit on the opposite signal."""
    c = df["c"].to_numpy(dtype=float)
    pc = _partner_close(df)
    if pc is None:
        return _none(len(c))
    ratio = c / pc
    r_ok = np.isfinite(ratio)
    re = ema(np.where(r_ok, ratio, np.nan), 50)
    e50 = ema(c, 50)
    above = ratio > re
    up, dn = np.zeros(len(c), bool), np.zeros(len(c), bool)
    up[1:] = above[1:] & ~above[:-1] & (c[1:] > e50[1:])
    dn[1:] = ~above[1:] & above[:-1] & (c[1:] < e50[1:])
    return up & r_ok, dn & r_ok, {"stop": ("atr", 2.0), "target": None, "trail_atr": 3.0, "opposite": True}


# id: (name, fn, timeframes)
INTERMARKET = {
    "IM01": ("Spread reversion vs partner (pairs, one leg)", s_pair_spread_revert, ["1Hour", "4Hour"]),
    "IM02": ("Lead-lag catch-up after partner's big bar", s_lead_lag, ["15Min", "1Hour"]),
    "IM03": ("Relative strength vs partner (ratio EMA50 cross)", s_relative_strength, ["1Hour", "4Hour"]),
}
