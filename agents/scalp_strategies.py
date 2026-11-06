"""Fast (scalping) strategies on the 1-minute chart: in and out within minutes, never seconds.

Why not seconds: FTMO's terms forbid high-frequency trading and strategies that open and close within
seconds; and at that speed the spread and commission eat the move. These hold for up to 30-60 one-minute bars.
Same contract as agents/rules_agent.py: decided on a closed bar, filled at the next bar's open; the backtest
charges FTMO's own spread and commission on every trade.

SC01 Keltner breakout  two closes outside the Keltner channel (EMA20 +/- 1.5 ATR20), RSI(14) crossed 50 in
                       the last 5 bars; stop 1 ATR, target 1.5 R, out after 30 bars
SC02 VWAP pullback     the day's VWAP rising (falling) and price on its side for 30 bars; a bar dips to the
                       VWAP and closes back on the trend side; New York hours only; stop 1.5 ATR, target 2 R,
                       out after 45 bars
SC03 5m trend, 1m entry  5-minute EMA20 above (below) EMA50 on completed 5-minute bars; the 1-minute close
                       comes back above (below) its EMA9 after at least two bars on the other side;
                       stop 1.2 ATR, target 2 R, out after 30 bars
SC04 EMA + ATR         EMA9 crosses EMA21 with the EMA50 sloping the same way, only while ATR(14) is above its
                       median of the last 100 bars (the market is moving); stop 1.5 ATR, target 2 ATR,
                       out after 60 bars (written for gold, run on every instrument)
"""
import numpy as np
import pandas as pd

from agents.rules_agent import atr, ema, rsi

NY = "America/New_York"


def _sh(x, k=1):
    x = np.asarray(x, float)
    return np.r_[np.full(k, np.nan), x[:-k]]


def _idx(df):
    t = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["t"], utc=True)
    return t.tz_localize("UTC") if t.tz is None else t


def s_keltner_breakout(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    mid = ema(c, 20)
    a20 = atr(h, l, c, 20)
    up, dn = mid + 1.5 * a20, mid - 1.5 * a20
    r = rsi(c, 14)
    up_x = (r > 50) & (_sh(r) <= 50)
    dn_x = (r < 50) & (_sh(r) >= 50)
    up5 = pd.Series(up_x).rolling(5, min_periods=1).max().to_numpy().astype(bool)
    dn5 = pd.Series(dn_x).rolling(5, min_periods=1).max().to_numpy().astype(bool)
    two_up = (c > up) & (_sh(c) > _sh(up))
    two_dn = (c < dn) & (_sh(c) < _sh(dn))
    fresh_up = two_up & ~(_sh(two_up.astype(float)) > 0)
    fresh_dn = two_dn & ~(_sh(two_dn.astype(float)) > 0)
    return fresh_up & up5, fresh_dn & dn5, {"stop": ("atr", 1.0), "target": ("r", 1.5), "max_bars": 30}


def s_vwap_pullback(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    v = df["v"].to_numpy(float) if "v" in df and np.nansum(df["v"].to_numpy(float)) > 0 else np.ones(len(c))
    t = _idx(df).tz_convert(NY)
    day = pd.Series(t.date, index=range(len(c)))
    tp = (h + l + c) / 3
    pv = pd.Series(tp * v).groupby(day.values).cumsum().to_numpy()
    vv = pd.Series(v).groupby(day.values).cumsum().to_numpy()
    vw = pv / np.where(vv > 0, vv, np.nan)
    slope = vw - _sh(vw, 10)
    above = pd.Series(_sh(c) > _sh(vw)).rolling(30).min().to_numpy() == 1
    below = pd.Series(_sh(c) < _sh(vw)).rolling(30).min().to_numpy() == 1
    mins = t.hour * 60 + t.minute
    ny = (mins >= 9 * 60 + 45) & (mins < 15 * 60 + 30)
    L = ny & above & (slope > 0) & (l <= vw) & (c > vw)
    S = ny & below & (slope < 0) & (h >= vw) & (c < vw)
    return L, S, {"stop": ("atr", 1.5), "target": ("r", 2.0), "max_bars": 45}


def s_5m_trend_1m_entry(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    t = _idx(df)
    s = pd.Series(c, index=t)
    c5 = s.resample("5min").last().dropna()
    e20, e50 = pd.Series(ema(c5.to_numpy(), 20), index=c5.index), pd.Series(ema(c5.to_numpy(), 50), index=c5.index)
    trend = np.sign(e20 - e50)
    # a 5-minute bar is known only once it is complete: at its last 1-minute bar (open + 4 minutes)
    trend.index = trend.index + pd.Timedelta(minutes=4)
    tr = trend.reindex(t, method="ffill").to_numpy()
    e9 = ema(c, 9)
    below2 = (_sh(c) < _sh(e9)) & (_sh(c, 2) < _sh(e9, 2))
    above2 = (_sh(c) > _sh(e9)) & (_sh(c, 2) > _sh(e9, 2))
    L = (tr > 0) & below2 & (c > e9)
    S = (tr < 0) & above2 & (c < e9)
    return L, S, {"stop": ("atr", 1.2), "target": ("r", 2.0), "max_bars": 30}


def s_ema_atr(df):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    a = atr(h, l, c, 14)
    active = a > pd.Series(a).rolling(100).median().to_numpy()
    up = (e9 > e21) & (_sh(e9) <= _sh(e21))
    dn = (e9 < e21) & (_sh(e9) >= _sh(e21))
    L = up & (e50 > _sh(e50, 5)) & active
    S = dn & (e50 < _sh(e50, 5)) & active
    return L, S, {"stop": ("atr", 1.5), "target": ("atr", 2.0), "max_bars": 60}


# id: (name, fn, timeframes)
SCALP = {
    "SC01": ("Keltner breakout with RSI 50 cross, 1-minute scalp", s_keltner_breakout, ["1Min"]),
    "SC02": ("VWAP trend pullback, New York hours, 1-minute scalp", s_vwap_pullback, ["1Min"]),
    "SC03": ("5-minute EMA trend, 1-minute EMA9 re-entry scalp", s_5m_trend_1m_entry, ["1Min"]),
    "SC04": ("EMA 9/21 cross with EMA50 slope and active ATR, 1-minute scalp", s_ema_atr, ["1Min"]),
}
