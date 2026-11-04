"""Strategies from QuantConnect (Strategy Library, Lean alpha models / examples, documentation examples,
forum), coded from their published rules. Same contract as agents/rules_agent.py.

Stops: where the source has no stop (most of them) a 3 ATR stop is OUR addition (needed for FTMO risk
sizing). Percentage stops/targets written for shares (1% / 2% on 5-minute bars) are translated to
ATR: stop 1.5 ATR, target 2R (same 1:2 shape). Everything else as published.
"""
import numpy as np
import pandas as pd

from agents.book_strategies import _daily_signals, _days, _on_bars
from agents.rules_agent import adx, atr, cross_down, cross_up, ema, macd, rsi, sma

NY = "America/New_York"


def _s(x):
    return pd.Series(np.asarray(x, dtype=float))


def _first(m):
    m = np.asarray(m, bool)
    return m & ~np.r_[False, m[:-1]]


# ---------------- QC01 Dynamic Breakout II ----------------
def s_dynamic_breakout(df):
    """Dynamic Breakout II (daily): the lookback N (20..60) grows/shrinks with the change of 30-day
    volatility; long when yesterday's close was above the upper Bollinger band (EMA middle, N, 2) and
    today's close breaks the N-day high; mirror short. Out when the close crosses the N-day average.
    Stop 3 ATR (ours)."""
    _, day, _, _ = _days(df)
    c, h, l = day["c"].to_numpy(), day["h"].to_numpy(), day["l"].to_numpy()
    n = len(c)
    N = np.full(n, 20)
    L, S, XL, XS = (np.zeros(n, bool) for _ in range(4))
    cur = 20
    for t in range(31, n):
        tv, yv = np.std(c[t - 29:t + 1]), np.std(c[t - 30:t])
        if tv > 0:
            cur = int(min(60, max(20, round(cur * (1 + (tv - yv) / tv)))))
        N[t] = cur
        if t < cur + 2:
            continue
        w = c[t - cur:t]                                   # yesterday's band uses closes up to yesterday
        mid = pd.Series(w).ewm(span=cur, adjust=False).mean().iloc[-1]
        sd = np.std(w)
        up_y, lo_y = mid + 2 * sd, mid - 2 * sd
        hh, ll = h[t - cur:t].max(), l[t - cur:t].min()
        L[t] = c[t - 1] > up_y and c[t] > hh
        S[t] = c[t - 1] < lo_y and c[t] < ll
        avg = c[t - cur + 1:t + 1].mean()
        XL[t], XS[t] = c[t] < avg, c[t] > avg
    ix = day.index
    return _daily_signals(df, pd.Series(L, ix), pd.Series(S & ~L, ix), pd.Series(XL, ix), pd.Series(XS, ix))


# ---------------- QC02 Dual Thrust (daily range, hourly checks, always in) ----------------
def s_dual_thrust_daily(df):
    """Dual Thrust (QuantConnect library): range = max(HH - LC, HC - LL) of the last 4 completed days;
    a close above today's open + 0.5 x range goes long, below open - 0.5 x range short; always in the
    market, reversing on the other line. Stop 3 ATR (ours)."""
    date, day, first, last = _days(df)
    c = df["c"].to_numpy()
    hh, ll = day["h"].rolling(4).max().shift(1), day["l"].rolling(4).min().shift(1)
    hc, lc = day["c"].rolling(4).max().shift(1), day["c"].rolling(4).min().shift(1)
    rng = _on_bars(date, np.maximum(hh - lc, hc - ll))
    dop = _on_bars(date, day["o"])
    up, dn = c >= dop + 0.5 * rng, c <= dop - 0.5 * rng
    return _first(up), _first(dn) & ~up, {"stop": ("atr", 3.0), "target": None, "opposite": True}


# ---------------- QC03 Dual Thrust intraday (30-minute bars) ----------------
def s_dual_thrust_30m(df):
    """Dual Thrust intraday (Lean alpha): on 30-minute bars, range = max(HH - LC, HC - LL) of the last 20
    bars; lines = last 30m close +/- 0.63 x range, fixed until the next 30m close; long above the upper
    line, short below the lower; reverses; out after 5 days. Source stop 3% of price -> 3 ATR."""
    b = df.resample("30min", label="left", closed="left").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    hh, ll = b["h"].rolling(20).max(), b["l"].rolling(20).min()
    hc, lc = b["c"].rolling(20).max(), b["c"].rolling(20).min()
    rng = np.maximum(hh - lc, hc - ll)
    up_l, dn_l = (b["c"] + 0.63 * rng), (b["c"] - 0.63 * rng)
    # a 30m bar's lines are known when it closes: use them from the next 30m block on
    key = df.index.floor("30min") - pd.Timedelta(minutes=30)
    ul = up_l.reindex(key).to_numpy()
    dl = dn_l.reindex(key).to_numpy()
    c = df["c"].to_numpy()
    up, dn = c > ul, c < dl
    bars_5d = int(round(5 * 24 * 60 / max(1, (df.index[1] - df.index[0]).total_seconds() / 60)))
    return _first(up), _first(dn) & ~up, {"stop": ("atr", 3.0), "target": None, "opposite": True, "max_bars": bars_5d}


# ---------------- QC04 Hodrick-Prescott trend momentum ----------------
def _hp_one_sided(y, lam=100.0):
    """One-sided Hodrick-Prescott trend: the HP filter's value at each day using only data up to that
    day, computed as the Kalman filter of the local linear trend model with noise ratio 1/lambda (the
    textbook equivalence; avoids re-solving an 1800-point system every day)."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = np.array([[0.0, 0.0], [0.0, 1.0 / lam]])
    x = np.array([y[0], 0.0])
    P = np.eye(2) * 1e4
    out = np.zeros(len(y))
    for i, v in enumerate(y):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H + 1.0
        K = P @ H / S
        x = x + K * (v - H @ x)
        P = P - np.outer(K, H) @ P
        out[i] = x[0]
    return out


def s_hp_trend(df):
    """Hodrick-Prescott trend momentum (QuantConnect library): every day, the HP trend (lambda 100) using
    only the data up to that day (one-sided); the trend turning up -> long, turning down -> short, always
    in. Stop 3 ATR (ours)."""
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    tr = _hp_one_sided(np.log(c))
    d1 = np.diff(tr, prepend=np.nan)
    d0 = np.r_[np.nan, d1[:-1]]
    L, S = (d1 > 0) & (d0 < 0), (d1 < 0) & (d0 > 0)
    L[:60] = S[:60] = False
    ix = day.index
    return _daily_signals(df, pd.Series(L, ix), pd.Series(S, ix), pd.Series(S, ix), pd.Series(L, ix))


# ---------------- QC05 Intraday FX reversal vs SMA5 ----------------
def s_fx_intraday_reversal(df):
    """Intraday reversal (Lean alpha, currencies): on hourly bars closing 10:00-15:00 New York, below
    SMA5 x 1.001 -> long, otherwise short, acting only when the direction changes; everything flat at
    15:00. Stop 3 ATR (ours)."""
    c = df["c"].to_numpy()
    t = df.index.tz_convert(NY)
    hr = t.hour.to_numpy()
    s5 = sma(c, 5)
    window = (hr >= 9) & (hr <= 13)            # bars starting 09:00-13:00 close 10:00-14:00; 14:00 bar = exit
    want = np.where(c < s5 * 1.001, 1, -1)
    state, L, S = 0, np.zeros(len(c), bool), np.zeros(len(c), bool)
    for i in range(len(c)):
        if not window[i] or not np.isfinite(s5[i]):
            continue
        if want[i] != state:
            state = want[i]
            L[i], S[i] = state == 1, state == -1
    return L, S, {"stop": ("atr", 3.0), "target": None, "opposite": True, "force_exit": hr == 14}


# ---------------- QC06 Lunch-break reversion ----------------
def s_lunch_reversion(df):
    """Lunch-break mean reversion (Lean alpha): at 12:00 New York, if the price is below the previous
    session's 16:00 close -> long, above -> short; out after one hour. Stop 2 ATR (ours)."""
    c = df["c"].to_numpy()
    t = df.index.tz_convert(NY)
    hr = t.hour.to_numpy()
    date = np.array([d.isoformat() for d in t.date])
    at_close = hr == 15                         # the 15:00 bar closes at 16:00
    cl = pd.Series(np.where(at_close, c, np.nan)).groupby(date).transform("max")
    prev = pd.Series(cl.groupby(date).first()).shift(1)
    pc = pd.Series(date).map(prev).to_numpy(dtype=float)
    sig = hr == 11                              # the 11:00 bar closes at 12:00
    r = c / pc - 1
    return sig & (r <= 0), sig & (r > 0), {"stop": ("atr", 2.0), "target": None, "max_bars": 1}


# ---------------- QC07 Noise-area intraday momentum ----------------
def s_noise_area(df):
    """Noise-area momentum (Zarattini, Aziz & Barbon; QuantConnect version), US session: for every
    half-hour slot, the average absolute move from the 09:30 open over the last 14 sessions makes a band
    around today's open; a half-hour close outside the band enters in its direction, a close back inside
    exits; flat at the close. Stop 3 ATR (ours)."""
    c = df["c"].to_numpy()
    t = df.index.tz_convert(NY)
    step = (df.index[1] - df.index[0]).total_seconds() / 60
    end = t + pd.Timedelta(minutes=step)
    em = (end.hour * 60 + end.minute).to_numpy()
    date = np.array([d.isoformat() for d in t.date])
    sess = (em > 570) & (em <= 960)
    slot_end = sess & (em % 30 == 0)
    op = pd.Series(np.where(sess & (em - step <= 570) & (em > 570), df["o"].to_numpy(), np.nan)).groupby(date).transform("max").to_numpy()
    mv = np.where(slot_end, np.abs(c / op - 1), np.nan)
    tab = pd.DataFrame({"d": date, "k": em, "m": mv})[slot_end]
    piv = tab.pivot_table(index="d", columns="k", values="m")
    sig = piv.rolling(14, min_periods=10).mean().shift(1)
    sg = np.array([sig.at[d, k] if (d in sig.index and k in sig.columns) else np.nan for d, k in zip(date, em)])
    ub, lb = op * (1 + sg), op * (1 - sg)
    long_ = slot_end & (c > ub)
    short = slot_end & (c < lb)
    flat = sess & (em >= 960 - step)
    return long_, short & ~long_, {"stop": ("atr", 3.0), "target": None,
                                   "exit_long": slot_end & (c < ub), "exit_short": slot_end & (c > lb),
                                   "force_exit": flat | ~sess}


# ---------------- QC08 Displaced SMA ribbon ----------------
def s_sma_ribbon(df):
    """Displaced SMA ribbon (Lean example), daily: SMA15 delayed by 5, 10, ... 30 days all in rising order
    -> long; all in falling order -> out. Long only. Stop 3 ATR (ours)."""
    _, day, _, _ = _days(df)
    s = day["c"].rolling(15).mean()
    D = [s.shift(k) for k in (5, 10, 15, 20, 25, 30)]
    up = pd.Series(True, index=day.index)
    dn = pd.Series(True, index=day.index)
    for a, b in zip(D[:-1], D[1:]):
        up &= a >= b
        dn &= a <= b
    up &= D[-1].notna()
    lo = up & ~up.shift(1, fill_value=False)
    return _daily_signals(df, lo, pd.Series(False, index=day.index), dn, pd.Series(False, index=day.index))


# ---------------- QC09 MACD trend with tolerance ----------------
def s_macd_trend(df):
    """MACD trend (Lean example), daily: (MACD - signal) / EMA12 above +0.0025 -> long, below -0.0025
    -> out. Long only. Stop 3 ATR (ours)."""
    _, day, _, _ = _days(df)
    c = day["c"].to_numpy()
    m, sgl, _ = macd(c, 12, 26, 9)
    delta = pd.Series((m - sgl) / ema(c, 12), index=day.index)
    lo = (delta > 0.0025) & ~(delta.shift(1) > 0.0025)
    return _daily_signals(df, lo, pd.Series(False, index=day.index), delta < -0.0025, pd.Series(False, index=day.index))


# ---------------- QC10 RSI oversold inside a 15m MACD uptrend ----------------
def s_rsi_in_macd_trend(df):
    """5-minute RSI(14) under 30 while the last completed 15-minute MACD histogram is positive -> long;
    out when that histogram turns negative. Stop 3 ATR (ours)."""
    c = df["c"].to_numpy()
    r = rsi(c, 14)
    b = df["c"].resample("15min", label="left", closed="left").last().dropna()
    m, sgl, _ = macd(b.to_numpy(), 12, 26, 9)
    hist = pd.Series(m - sgl, index=b.index + pd.Timedelta(minutes=15))      # known at the 15m bar's end
    step = df.index[1] - df.index[0]
    hb = hist.reindex(df.index + step, method="ffill").to_numpy()
    long_ = (r < 30) & (hb > 0)
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": hb < 0}


# ---------------- QC11 Hammer / inverted hammer ----------------
def s_hammers(df):
    """Hammer or inverted hammer (Lean / TA-Lib candle definitions) on 5-minute bars -> long. Source
    1% stop / 2% target -> stop 1.5 ATR, target 2R."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    body, rng = np.abs(c - o), h - l
    up_sh, lo_sh = h - np.maximum(o, c), np.minimum(o, c) - l
    ab = _s(body).shift(1).rolling(10).mean().to_numpy()
    ar10 = _s(rng).shift(1).rolling(10).mean().to_numpy()
    ar5 = _s(rng).shift(2).rolling(5).mean().to_numpy()
    pl = np.r_[np.nan, l[:-1]]
    hammer = (body < ab) & (lo_sh > body) & (up_sh < 0.1 * ar10) & (np.minimum(o, c) <= pl + 0.2 * ar5)
    pmin = np.r_[np.nan, np.minimum(o, c)[:-1]]
    inv = (body < ab) & (up_sh > body) & (lo_sh < 0.1 * ar10) & (np.maximum(o, c) < pmin)
    return hammer | inv, np.zeros(len(c), bool), {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- QC12 Three rising candles, accelerating ----------------
def s_three_rising(df):
    """Three rising candles, the third closing more than 2 bodies above the second's close -> long.
    Source 1% / 2% -> stop 1.5 ATR, target 2R."""
    o, c = df["o"].to_numpy(), df["c"].to_numpy()
    c1, c2, o1, o2 = (np.r_[np.full(k, np.nan), x[:-k]] for k, x in ((1, c), (2, c), (1, o), (2, o)))
    long_ = ((c > c1) & (c1 > c2) & (c2 > o2) & (c1 > o1) & (c > o) & ((c2 - o2) > (c1 - o1))
             & (c > c1 + 2 * (c1 - o1)))
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- QC13 EMA200 fade, always in ----------------
def s_ema200_fade(df):
    """Always-in fade of the 200-bar EMA (QuantConnect docs, EURUSD 5m): at or below the EMA -> long,
    above -> short. Stop 3 ATR (ours)."""
    c = df["c"].to_numpy()
    e = ema(c, 200)
    below = c <= e
    return cross_down(c, e), cross_up(c, e) & ~below, {"stop": ("atr", 3.0), "target": None, "opposite": True}


# ---------------- QC14 ADX / volatility regime switch ----------------
def s_regime_switch(df):
    """Regime switch (QuantConnect forum template): trend regime (ATR or stdev over 0.2% of price and
    ADX > 25) trades EMA10 vs EMA30 (0.05% band); otherwise RSI(14) 30/70 fades. A regime change closes
    the other leg. Stop 3 ATR (ours)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a, x = atr(h, l, c, 14), adx(h, l, c, 14)
    sd = _s(c).rolling(14).std().to_numpy()
    mom = ((a / c > 0.002) | (sd / c > 0.002)) & (x > 25)
    e10, e30 = ema(c, 10), ema(c, 30)
    r = rsi(c, 14)
    up_m, dn_m = mom & (e10 > e30 * 1.0005), mom & (e10 < e30 * 0.9995)
    up_r, dn_r = ~mom & (r < 30), ~mom & (r > 70)
    L, S = _first(up_m) | _first(up_r), _first(dn_m) | _first(dn_r)
    switch = mom != np.r_[mom[:1], mom[:-1]]
    neutral = mom & ~up_m & ~dn_m
    return L, S & ~L, {"stop": ("atr", 3.0), "target": None, "opposite": True,
                       "exit_long": switch | neutral, "exit_short": switch | neutral}


# id: (name, fn, timeframes)
QUANTCONNECT = {
    "QC01": ("Dynamic Breakout II, adaptive lookback (QuantConnect)", s_dynamic_breakout, ["4Hour"]),
    "QC02": ("Dual Thrust, 4-day range from today's open, always in (QuantConnect)", s_dual_thrust_daily, ["1Hour"]),
    "QC03": ("Dual Thrust on 30-minute bars, 5-day exit (Lean alpha)", s_dual_thrust_30m, ["15Min"]),
    "QC04": ("Hodrick-Prescott trend turn, daily (QuantConnect)", s_hp_trend, ["4Hour"]),
    "QC05": ("Intraday FX reversal vs SMA5, NY 10-15 (Lean alpha)", s_fx_intraday_reversal, ["1Hour"]),
    "QC06": ("Lunch-break mean reversion, 1 hour hold (Lean alpha)", s_lunch_reversion, ["1Hour"]),
    "QC07": ("Noise-area intraday momentum (Zarattini et al.)", s_noise_area, ["15Min"]),
    "QC08": ("Displaced SMA15 ribbon, daily (Lean example)", s_sma_ribbon, ["4Hour"]),
    "QC09": ("MACD trend with tolerance, daily (Lean example)", s_macd_trend, ["4Hour"]),
    "QC10": ("RSI14 < 30 on 5m inside a 15m MACD uptrend (QC docs)", s_rsi_in_macd_trend, ["5Min"]),
    "QC11": ("Hammer / inverted hammer, 1.5 ATR stop, 2R (QC docs)", s_hammers, ["5Min"]),
    "QC12": ("Three rising candles with acceleration, 2R (QC docs)", s_three_rising, ["5Min"]),
    "QC13": ("Always-in EMA200 fade (QC docs)", s_ema200_fade, ["5Min"]),
    "QC14": ("ADX / volatility regime switch: EMA trend vs RSI fade (QC forum)", s_regime_switch, ["5Min", "15Min"]),
}
