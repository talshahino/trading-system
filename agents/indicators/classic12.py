"""Indicators the library lacked (scout 19.11.2026 afternoon run). Each is a port of the published formula, or our own
reconstruction where the code is closed / not readable or the page gives only a description (said so per indicator).
No repaint: every value at bar i uses bars <= i only (swing pivots enter on the bar that confirms them; a session's
open type is known only from the bar that closes its classification window).

- aroon: Aroon Up / Down / Oscillator (Tushar Chande 1995; formula as Oxford Capital Strategies' 'Aroon Indicator
  Breakout' https://oxfordstrat.com/trading-strategies/aroon-indicator-breakout-3/ and StockCharts ChartSchool):
  up = 100 x (n - bars since the highest high of the last n + 1 bars) / n, down the same with the lowest low.
- open_type: Market Profile open types (Dalton; NexusFi 'Open Type Classification'
  https://nexusfi.com/a/concepts/open-type-classification and The Right Trader 'Opening Drive' / 'Trend Day'
  https://therighttrader.com/learn/trading-strategies/opening-drive ). The sources classify by eye; the numeric rules are
  OURS: over the first `window` minutes of the session (default 09:30-10:00 New York on 1-5 minute bars, 09:30-10:15
  on 15-minute bars) with R = the window's range:
  +/-1 open-drive (the other side never went more than 0.1 R past the open, close in the far 30% of R),
  +/-2 open-test-drive (first a test 0.1-0.4 R the other way, then the close in the far 30%), +/-3 open-rejection-reverse
  (first an extension >= 0.4 R the other way, then a close back beyond the open), 4 open-auction
  (everything else). Also: the session open, the initial balance (first `ib_min` minutes) high / low, known after it
  closes, and its ratio to the mean of the previous 20 sessions' IB ranges (narrow IB = trend-day candidate).
- vol_gated_trend: Volatility-Gated Trend Oscillator [QuantAlgo] https://www.tradingview.com/script/NPpLwEi1 - open
  source on TradingView but the code was not readable to us; OUR reconstruction from its description: baseline EMA 20,
  deviation = close - baseline, noise floor = 1.0 x the 20-bar mean absolute deviation; the state turns +1 when the
  deviation breaks over +floor and stays until it breaks under -floor (and mirror); inside the neutral zone the locked
  deviation decays x 0.9 a bar; oscillator = WMA 4 of locked deviation / ATR 14.
- l1_trend_swing: [blackcat] L1 Trend Swing Oscillator https://www.tradingview.com/script/jtgTVb4r - code not readable to
  us; OUR reconstruction: weighted price (2c + h + l + o) / 5, histogram = weighted price - its EMA 10; trend line = EMA 8
  of the close's position (0-100) between the 11-bar low and the 25-bar high; ratio differential = EMA 3 of
  100 x ((c - LL21) - (HH21 - c)) / (HH21 - LL21). Buy = the trend line is its lowest of 5 bars and under `lo` while the
  close is the lowest of 10 bars; sell = mirror over `hi`. The author's thresholds 3 / 34 are on his own (unpublished)
  scale; ours are 10 / 90 on the 0-100 scale. Strong buy / sell = ratio differential under -45 / over +45.
- true_trend_osc: True Trend Oscillator (MYNAMEISBRANDON) https://www.tradingview.com/script/egNqhlKM - stated formulas
  bull = sqrt(RSI x StochATR), bear = sqrt((100 - RSI) x StochATR); the stochastic of ATR 14 over 50 bars and its
  causal Rational Quadratic kernel smoothing (lookback 8, relative weight 1) are OUR choices (the page does not give the
  defaults). Active trend = the weaker line over 25.
- trend_bias_osc: Trend Bias Oscillator (tgregg75) https://www.tradingview.com/script/Un7inhjq - the first published
  version, from its description: the close's position between the last confirmed swing high and swing low (pivots 20
  left / 5 right, known 5 bars late) scaled to -100..+100; the structure bias flips +1 on a close over the last swing
  high and -1 on a close under the last swing low; signal = EMA 5. The later 'conviction engine' version's constants are
  not published.
"""
import numpy as np
import pandas as pd

from agents.indicators._core import S, atr, bar_minutes, cols, ema, highest, lowest, rsi, wma


def aroon(df, n=25):
    """dict up, down, osc (up - down), each 0..100 / -100..100."""
    o, h, l, c, v = cols(df)
    hs, ls = S(h), S(l)
    since_h = hs.rolling(n + 1).apply(lambda x: n - int(np.argmax(x)), raw=True).to_numpy()
    since_l = ls.rolling(n + 1).apply(lambda x: n - int(np.argmin(x)), raw=True).to_numpy()
    up = 100.0 * (n - since_h) / n
    dn = 100.0 * (n - since_l) / n
    return {"up": up, "down": dn, "osc": up - dn}


def open_type(df, tz="America/New_York", start=9.5, window=None, ib_min=60, end=16.0):
    """dict code (signed open type, known from the bar closing the window to the session end, else 0), sopen,
    shi / slo (session high / low so far), ib_hi / ib_lo (known after the IB closes), ib_ratio, in_session,
    t_lo_first (+1 the window's low came before its high, -1 the high first)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    st = df.index.tz_convert(tz)
    bm = bar_minutes(df)
    if window is None:                       # 30 minutes on 1-5 minute bars, 45 (three bars) on 15-minute bars
        window = 30 if bm <= 5 else 45
    hs = st.hour.to_numpy() + st.minute.to_numpy() / 60.0
    he = hs + bm / 60.0
    day = pd.Series(st.date).to_numpy()
    w_end = start + window / 60.0
    ib_end = start + ib_min / 60.0
    code = np.zeros(n)
    sopen = np.full(n, np.nan)
    shi = np.full(n, np.nan)
    slo = np.full(n, np.nan)
    ib_hi = np.full(n, np.nan)
    ib_lo = np.full(n, np.nan)
    ib_ratio = np.full(n, np.nan)
    insess = (hs >= start - 1e-9) & (hs < end - 1e-9)
    ib_hist = []
    cur_day, op, hi, lo, k, ihi, ilo, wd = None, np.nan, -np.inf, np.inf, 0, -np.inf, np.inf, None
    t_hi = t_lo = -1
    ib_known = False
    for i in range(n):
        if not insess[i]:
            continue
        if day[i] != cur_day:
            if cur_day is not None and ib_known:
                ib_hist.append(ihi - ilo)
            cur_day, op, hi, lo, k, ihi, ilo, ib_known = day[i], o[i], -np.inf, np.inf, 0, -np.inf, np.inf, False
            if hs[i] > start + 1e-9:          # session joined late (data gap) -> no open type today
                op = np.nan
            t_hi = t_lo = -1
            k_val = 0.0
        if h[i] > hi:
            hi, t_hi = h[i], i
        if l[i] < lo:
            lo, t_lo = l[i], i
        sopen[i], shi[i], slo[i] = op, hi, lo
        if he[i] <= ib_end + 1e-9:
            ihi, ilo = max(ihi, h[i]), min(ilo, l[i])
            if abs(he[i] - ib_end) < 1e-6:
                ib_known = True
        if ib_known:
            ib_hi[i], ib_lo[i] = ihi, ilo
            if len(ib_hist) >= 5:
                m = float(np.mean(ib_hist[-20:]))
                ib_ratio[i] = (ihi - ilo) / m if m > 0 else np.nan
        if abs(he[i] - w_end) < 1e-6 and np.isfinite(op):
            R = hi - lo
            if R > 0:
                up_x, dn_x = hi - op, op - lo
                top = c[i] >= hi - 0.3 * R
                bot = c[i] <= lo + 0.3 * R
                lo_first = t_lo < t_hi
                if dn_x <= 0.1 * R and top:
                    k_val = 1.0
                elif up_x <= 0.1 * R and bot:
                    k_val = -1.0
                elif lo_first and 0.1 * R < dn_x < 0.4 * R and top:
                    k_val = 2.0
                elif (not lo_first) and 0.1 * R < up_x < 0.4 * R and bot:
                    k_val = -2.0
                elif lo_first and dn_x >= 0.4 * R and c[i] > op:
                    k_val = 3.0
                elif (not lo_first) and up_x >= 0.4 * R and c[i] < op:
                    k_val = -3.0
                else:
                    k_val = 4.0
        if he[i] >= w_end - 1e-9 and np.isfinite(op):
            code[i] = k_val
    return {"code": code, "sopen": sopen, "shi": shi, "slo": slo, "ib_hi": ib_hi, "ib_lo": ib_lo,
            "ib_ratio": ib_ratio, "in_session": insess}


def vol_gated_trend(df, n=20, mult=1.0, decay=0.9, smooth=4):
    """dict osc, state (+1 / -1 / 0), buy / sell (state turns), floor."""
    o, h, l, c, v = cols(df)
    base = ema(c, n)
    dev = c - base
    floor = mult * S(np.abs(dev)).rolling(n).mean().to_numpy()
    a = atr(h, l, c, 14)
    nn = len(c)
    state = np.zeros(nn)
    lock = np.zeros(nn)
    s, lk = 0.0, 0.0
    for i in range(nn):
        f = floor[i]
        if np.isfinite(f):
            if dev[i] > f:
                s, lk = 1.0, dev[i]
            elif dev[i] < -f:
                s, lk = -1.0, dev[i]
            else:
                lk = lk * decay
        state[i], lock[i] = s, lk
    with np.errstate(divide="ignore", invalid="ignore"):
        osc = wma(np.nan_to_num(lock / np.where(a > 0, a, np.nan)), smooth)
    prev = np.concatenate([[0.0], state[:-1]])
    return {"osc": osc, "state": state, "buy": (state > 0) & (prev <= 0), "sell": (state < 0) & (prev >= 0),
            "floor": floor}


def l1_trend_swing(df, lo=10.0, hi=90.0):
    """dict trend, hist, ratio, buy, sell, strong_buy, strong_sell."""
    o, h, l, c, v = cols(df)
    wp = (2 * c + h + l + o) / 5.0
    hist = wp - ema(wp, 10)
    ll11, hh25 = lowest(l, 11), highest(h, 25)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = 100.0 * (c - ll11) / (hh25 - ll11)
    trend = ema(np.nan_to_num(pos, nan=50.0), 8)
    ll21, hh21 = lowest(l, 21), highest(h, 21)
    with np.errstate(divide="ignore", invalid="ignore"):
        rd = 100.0 * ((c - ll21) - (hh21 - c)) / (hh21 - ll21)
    ratio = ema(np.nan_to_num(rd), 3)
    tmin = S(trend).rolling(5).min().to_numpy()
    tmax = S(trend).rolling(5).max().to_numpy()
    cmin = S(c).rolling(10).min().to_numpy()
    cmax = S(c).rolling(10).max().to_numpy()
    buy = (trend <= tmin) & (trend < lo) & (c <= cmin)
    sell = (trend >= tmax) & (trend > hi) & (c >= cmax)
    return {"trend": trend, "hist": hist, "ratio": ratio, "buy": buy, "sell": sell & ~buy,
            "strong_buy": buy & (ratio < -45), "strong_sell": sell & ~buy & (ratio > 45)}


def _rq_kernel(x, lookback=8, weight=1.0):
    """Causal Rational Quadratic kernel average of the current and the previous `lookback` x 3 values."""
    x = np.asarray(x, float)
    m = lookback * 3
    j = np.arange(m + 1)
    w = (1.0 + j ** 2 / (2.0 * weight * lookback ** 2)) ** (-weight)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        k = min(i, m)
        seg = x[i - k:i + 1][::-1]
        ww = w[:k + 1]
        ok = np.isfinite(seg)
        if ok.any():
            out[i] = float(np.sum(seg[ok] * ww[ok]) / np.sum(ww[ok]))
    return out


def true_trend_osc(df, rsi_n=14, atr_n=14, stoch_n=50, lookback=8, active=25.0):
    """dict bull, bear, active (weaker line over `active`), stoch_atr."""
    o, h, l, c, v = cols(df)
    r = rsi(c, rsi_n)
    a = atr(h, l, c, atr_n)
    amin, amax = lowest(a, stoch_n), highest(a, stoch_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        sa = 100.0 * (a - amin) / (amax - amin)
    sa = _rq_kernel(np.nan_to_num(sa, nan=50.0), lookback)
    sa = np.clip(sa, 0, 100)
    bull = np.sqrt(np.clip(r, 0, 100) * sa)
    bear = np.sqrt(np.clip(100 - r, 0, 100) * sa)
    return {"bull": bull, "bear": bear, "active": np.minimum(bull, bear) > active, "stoch_atr": sa}


def trend_bias_osc(df, left=20, right=5, sig=5):
    """dict osc (-100..100), bias (+1 / -1 / 0), signal, swing_hi, swing_lo (last confirmed)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    hs, ls = S(h), S(l)
    ph = (hs == hs.rolling(left + right + 1).max().shift(-right)).to_numpy()   # pivot with left / right bars
    pl = (ls == ls.rolling(left + right + 1).min().shift(-right)).to_numpy()
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    j = np.where(ph)[0]
    j = j[j + right < n]
    sh[j + right] = h[j]
    j = np.where(pl)[0]
    j = j[j + right < n]
    sl[j + right] = l[j]
    sh, sl = S(sh).ffill().to_numpy(), S(sl).ffill().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        osc = np.clip(200.0 * (c - sl) / (sh - sl) - 100.0, -150, 150)
    bias = np.zeros(n)
    b = 0.0
    for i in range(1, n):
        if np.isfinite(sh[i - 1]) and c[i] > sh[i - 1]:
            b = 1.0
        elif np.isfinite(sl[i - 1]) and c[i] < sl[i - 1]:
            b = -1.0
        bias[i] = b
    osc = np.nan_to_num(osc)
    return {"osc": osc, "bias": bias, "signal": ema(osc, sig), "swing_hi": sh, "swing_lo": sl}
