"""Strategies from open-source GitHub projects (freqtrade-strategies, jesse, backtrader examples, a
LazyBear Pine port) and one QuanterLab study, coded from their source. Same contract as
agents/rules_agent.py.

Stops: most of these come from crypto bots with percentage stops of 5-30% (useless for FX/indices) or
none at all. Where the source has no usable stop, a 3 ATR stop is OUR addition (needed for FTMO risk
sizing); everything else (entries, exits, parameters) is as written in the source.
"""
import numpy as np
import pandas as pd

from agents.rules_agent import adx, atr, cross_down, cross_up, ema, rsi, sma

NY = "America/New_York"


def _s(x):
    return pd.Series(np.asarray(x, dtype=float))


# ---------------- GH01 OTT (freqtrade futures/FOttStrategy) ----------------
def _ott(c, pds=2, cmo_len=9, percent=1.4):
    n = len(c)
    d = np.diff(c, prepend=c[0])
    ud, dd = np.clip(d, 0, None), np.clip(-d, 0, None)
    UD, DD = _s(ud).rolling(cmo_len).sum().to_numpy(), _s(dd).rolling(cmo_len).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cmo = np.nan_to_num(np.abs((UD - DD) / (UD + DD)))
    alpha = 2 / (pds + 1)
    var = np.zeros(n)
    for i in range(1, n):
        var[i] = alpha * cmo[i] * c[i] + (1 - alpha * cmo[i]) * var[i - 1]
    fark = var * percent / 100
    nl, ns = var - fark, var + fark
    ls, ss, mt = nl.copy(), ns.copy(), np.zeros(n)
    dirn = 1
    for i in range(1, n):
        ls[i] = max(nl[i], ls[i - 1]) if var[i] > ls[i - 1] else nl[i]
        ss[i] = min(ns[i], ss[i - 1]) if var[i] < ss[i - 1] else ns[i]
        if dirn == -1 and var[i] > ss[i - 1]:
            dirn = 1
        elif dirn == 1 and var[i] < ls[i - 1]:
            dirn = -1
        mt[i] = ls[i] if dirn == 1 else ss[i]
    ott = np.where(var > mt, mt * (200 + percent) / 200, mt * (200 - percent) / 200)
    ott = np.r_[np.full(2, np.nan), ott[:-2]]          # OTT shifted 2 bars
    var[:50] = np.nan
    return var, ott


def s_ott(df):
    """OTT: VIDYA (VAR) crosses above the Optimized Trend Tracker -> long (the source never takes its
    shorts). Exit when ADX(14) > 60. Stop 3 ATR (ours)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    var, ott = _ott(c)
    a = adx(h, l, c, 14)
    return cross_up(var, ott), np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None,
                                                         "exit_long": a > 60}


# ---------------- GH02 three Supertrends (freqtrade FSupertrend) ----------------
def _st_dir(h, l, c, period, mult):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    a = _s(tr).rolling(period).mean().to_numpy()
    mid = (h + l) / 2
    bu, bl = mid + mult * a, mid - mult * a
    fu, fl = bu.copy(), bl.copy()
    st = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if not np.isfinite(a[i]):
            continue
        fu[i] = bu[i] if (not np.isfinite(fu[i - 1]) or bu[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = bl[i] if (not np.isfinite(fl[i - 1]) or bl[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        prev = st[i - 1]
        if not np.isfinite(prev) or prev == fu[i - 1]:
            st[i] = fu[i] if c[i] <= fu[i] else fl[i]
        else:
            st[i] = fl[i] if c[i] >= fl[i] else fu[i]
    return np.where(np.isfinite(st), np.where(c < st, -1, 1), 0)


def s_triple_supertrend(df):
    """All three 'buy' Supertrends (8x4, 9x7, 8x1) up -> long; exit when the 'sell' Supertrend 18x3 is
    down. Long only as the source runs it. Stop 3 ATR (ours)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    up = (_st_dir(h, l, c, 8, 4) == 1) & (_st_dir(h, l, c, 9, 7) == 1) & (_st_dir(h, l, c, 8, 1) == 1)
    entry = up & ~np.r_[False, up[:-1]]
    return entry, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None,
                                           "exit_long": _st_dir(h, l, c, 18, 3) == -1}


# ---------------- GH03 HLHB (freqtrade hlhb) ----------------
def s_hlhb(df):
    """HLHB: on the same bar RSI(10) of (open+close)/2 crosses above 50, EMA5 crosses above EMA10 and
    ADX(14) > 25 -> long; exit on the mirror (RSI under 50, EMA5 under EMA10, ADX > 25). Trailing 1.17%
    after +1.86% in the source -> trail 2 ATR here; stop 3 ATR (ours)."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    r = rsi((o + c) / 2, 10)
    e5, e10 = ema(c, 5), ema(c, 10)
    a = adx(h, l, c, 14)
    long_ = cross_up(r, 50) & cross_up(e5, e10) & (a > 25)
    ex = cross_down(r, 50) & cross_down(e5, e10) & (a > 25)
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": ex, "trail_atr": 2.0}


# ---------------- GH04 TD Sequential setup 9 with perfection ----------------
def _td(c, l, cmp=lambda a, b: a < b):
    n = len(c)
    cnt = np.zeros(n, int)
    for i in range(4, n):
        cnt[i] = cnt[i - 1] + 1 if cmp(c[i], c[i - 4]) else 0
    return cnt


def s_td_sequential(df):
    """TD Sequential buy setup: the 9th straight close below the close 4 bars earlier, 'perfected' (the
    low of bar 8 or 9 under the low of bar 6 or 7) -> long. Exit on a perfected sell setup 9. Source
    stop 5% -> 3 ATR here."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    nb, ns = _td(c, l, lambda a, b: a < b), _td(c, h, lambda a, b: a > b)
    lo6, lo7 = np.r_[np.full(3, np.nan), l[:-3]], np.r_[np.full(2, np.nan), l[:-2]]
    hi6, hi7 = np.r_[np.full(3, np.nan), h[:-3]], np.r_[np.full(2, np.nan), h[:-2]]
    perf_b = (l < np.minimum(lo6, lo7)) | (np.r_[np.nan, l[:-1]] < np.r_[np.full(4, np.nan), l[:-4]])
    perf_s = (h > np.maximum(hi6, hi7)) | (np.r_[np.nan, h[:-1]] > np.r_[np.full(4, np.nan), h[:-4]])
    long_ = (nb == 9) & perf_b
    ex = (ns >= 9) & perf_s
    return long_, np.zeros(n, bool), {"stop": ("atr", 3.0), "target": None, "exit_long": ex}


# ---------------- GH05 CofiBit (freqtrade) ----------------
def s_cofibit(df):
    """CofiBit: open under the EMA5 of the lows, fast stochastic K crosses above D with both under 25,
    ADX > 25 -> long. Exit when open >= EMA5 of the highs or fast K / D cross above 75. Stop 3 ATR
    (source 25%), its 5-10% ROI targets never reach on CFDs -> no target."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    hh, ll = _s(h).rolling(5).max().to_numpy(), _s(l).rolling(5).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100 * (c - ll) / (hh - ll)
    d = sma(k, 3)
    a = adx(h, l, c, 14)
    long_ = (o < ema(l, 5)) & cross_up(k, d) & (k < 25) & (d < 25) & (a > 25)
    ex = (o >= ema(h, 5)) | cross_up(k, 75) | cross_up(d, 75)
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": ex}


# ---------------- GH06 Volatility System (freqtrade futures) ----------------
def s_volatility_system(df):
    """Volatility System: on 3-hour bars, a close-to-close move bigger than 2 x ATR(14) of the 3-hour
    bars -> follow it; stop-and-reverse on the opposite signal. Signal at the close of each 3-hour
    block. Stop 3 ATR (ours; the source has none)."""
    c = df["c"]
    b3 = df.resample("3h", label="left", closed="left").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    a3 = atr(b3["h"].to_numpy(), b3["l"].to_numpy(), b3["c"].to_numpy(), 14)
    chg = b3["c"].diff().to_numpy()
    up3 = pd.Series(chg > 2 * np.r_[np.nan, a3[:-1]], index=b3.index)
    dn3 = pd.Series(chg < -2 * np.r_[np.nan, a3[:-1]], index=b3.index)
    block = df.index.floor("3h")
    last = np.r_[block[1:] != block[:-1], True]
    up = last & up3.reindex(block).fillna(False).to_numpy()
    dn = last & dn3.reindex(block).fillna(False).to_numpy()
    return up, dn & ~up, {"stop": ("atr", 3.0), "target": None, "opposite": True}


# ---------------- GH07 Strategy002 (freqtrade) ----------------
def _psar(h, l, af0=0.02, afmax=0.2):
    n = len(h)
    sar = np.zeros(n)
    up, af, ep = True, af0, h[0]
    sar[0] = l[0]
    for i in range(1, n):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(afmax, af + af0)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(afmax, af + af0)
        sar[i] = s
    return sar


def s_strategy002(df):
    """Strategy002: RSI(14) < 30, slow stochastic K < 20, close under the lower Bollinger band (20, 2)
    of the typical price and a hammer candle -> long. Exit when the parabolic SAR is above the close and
    the Fisher RSI > 0.3. Stop 3 ATR (source 10%)."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    r = rsi(c, 14)
    hh, ll = _s(h).rolling(5).max().to_numpy(), _s(l).rolling(5).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = sma(sma(100 * (c - ll) / (hh - ll), 3), 1)
    tp = (h + l + c) / 3
    lower = sma(tp, 20) - 2 * _s(tp).rolling(20).std(ddof=0).to_numpy()
    body = np.abs(c - o)
    rng = h - l
    avg_body = _s(body).shift(1).rolling(10).mean().to_numpy()
    avg_rng10 = _s(rng).shift(1).rolling(10).mean().to_numpy()
    avg_rng5 = _s(rng).shift(1).rolling(5).mean().to_numpy()
    hammer = ((body < avg_body) & (np.minimum(o, c) - l > body) & (h - np.maximum(o, c) < 0.1 * avg_rng10)
              & (np.minimum(o, c) <= np.r_[np.nan, l[:-1]] + 0.2 * avg_rng5))
    long_ = (r < 30) & (k < 20) & (c < lower) & hammer
    fisher = np.tanh(0.1 * (r - 50))
    ex = (_psar(h, l) > c) & (fisher > 0.3)
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": ex}


# ---------------- GH08 Dual Thrust (jesse) ----------------
def s_dual_thrust(df):
    """Dual Thrust: range = max(HC - LL, HH - LC) over the last 21 bars; long when the close is above
    today's open + 0.71 x range, short below today's open - 0.67 x range (classic form; jesse's
    low/high slip on the short side fixed). Reverses on the opposite signal, stop 2 ATR as in jesse."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    hh, ll = _s(h).rolling(21).max().to_numpy(), _s(l).rolling(21).min().to_numpy()
    hc, lc = _s(c).rolling(21).max().to_numpy(), _s(c).rolling(21).min().to_numpy()
    rng = np.maximum(hc - ll, hh - lc)
    t = df.index.tz_convert(NY)
    day = np.array([d.isoformat() for d in t.date])
    dop = pd.Series(o).groupby(day).transform("first").to_numpy()
    up = c > dop + 0.71 * rng
    dn = c < dop - 0.67 * rng
    return up & ~np.r_[False, up[:-1]], dn & ~np.r_[False, dn[:-1]] & ~up, {"stop": ("atr", 2.0), "target": None,
                                                                             "opposite": True}


# ---------------- GH09 Squeeze Momentum (LazyBear) ----------------
def _linreg_last(y, n):
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sx = ((x - xm) ** 2).sum()

    def f(w):
        if np.isnan(w).any():
            return np.nan
        b = ((x - xm) * (w - w.mean())).sum() / sx
        return w.mean() + b * (n - 1 - xm)
    return _s(y).rolling(n).apply(f, raw=True).to_numpy()


def s_squeeze(df):
    """Squeeze Momentum: the Bollinger band (21, 1.5 x stdev - LazyBear's own quirk) was inside the
    Keltner channel (20, 1.5 x true range) and has just come out of it -> trade in the momentum's
    direction (linear-regression value > 0 long, < 0 short). Out when the momentum histogram turns
    'gray' (stops growing). Stop 3 ATR (ours)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    basis, dev = sma(c, 21), 1.5 * _s(c).rolling(21).std(ddof=0).to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    ma, rm = sma(c, 20), sma(tr, 20)
    sqz_on = (basis - dev > ma - 1.5 * rm) & (basis + dev < ma + 1.5 * rm)
    sqz_off = (basis - dev < ma - 1.5 * rm) & (basis + dev > ma + 1.5 * rm)
    hh, ll = _s(h).rolling(20).max().to_numpy(), _s(l).rolling(20).min().to_numpy()
    val = _linreg_last(c - ((hh + ll) / 2 + ma) / 2, 20)
    rel = np.r_[False, sqz_on[:-1]] & ~sqz_on & sqz_off
    pv = np.r_[np.nan, val[:-1]]
    gray = ~((val > 0) & (val > pv)) & ~((val < 0) & (val < pv))
    return rel & (val > 0), rel & (val < 0), {"stop": ("atr", 3.0), "target": None, "exit_long": gray, "exit_short": gray}


# ---------------- GH10 Momentum Ignition (backtrader example) ----------------
def s_momentum_ignition(df):
    """Momentum Ignition: price above its 30-bar average and the 7-bar rate of change breaks one
    standard deviation above its own 30-bar average -> long (mirror short). Chandelier trail 1 ATR, no
    target."""
    c = df["c"].to_numpy()
    roc = c / np.r_[np.full(7, np.nan), c[:-7]] - 1
    m, sd = sma(roc, 30), _s(roc).rolling(30).std(ddof=0).to_numpy()
    calm = _s(c).rolling(30).std(ddof=0).to_numpy() / c < 0.10
    s30 = sma(c, 30)
    up = calm & (c > s30) & (roc > m + sd)
    dn = calm & (c < s30) & (roc < m - sd)
    return up & ~np.r_[False, up[:-1]], dn & ~np.r_[False, dn[:-1]], {"stop": ("atr", 1.0), "target": None,
                                                                       "trail_atr": 1.0}


# ---------------- GH11 Ornstein-Uhlenbeck z-score ----------------
def s_ou_zscore(df):
    """Ornstein-Uhlenbeck mean reversion: fit dX = a + b X over the last 30 log prices; z = distance of
    the price from the fitted equilibrium in equilibrium standard deviations. z < -1 above the 30-bar
    average -> long, z > 1 below it -> short; out when z crosses 0. Stop 3 ATR (ours)."""
    c = df["c"].to_numpy()
    x = np.log(c)
    xl, dx = _s(np.r_[np.nan, x[:-1]]), _s(np.r_[np.nan, np.diff(x)])
    n = 29
    mx, my = xl.rolling(n).mean(), dx.rolling(n).mean()
    cov = (xl * dx).rolling(n).mean() - mx * my
    var = (xl * xl).rolling(n).mean() - mx * mx
    b = (cov / var).to_numpy()
    a = (my - (cov / var) * mx).to_numpy()
    res_var = (dx.rolling(n).var(ddof=0) - (cov / var) ** 2 * var).clip(lower=0).to_numpy()
    theta = -b
    ok = theta > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(ok, a / theta, _s(x).rolling(30).mean().to_numpy())
        eq = np.where(ok, np.sqrt(res_var) / np.sqrt(2 * theta), np.sqrt(res_var))
        z = (x - mu) / eq
    s30 = sma(c, 30)
    long_ = (z < -1) & (c > s30)
    short = (z > 1) & (c < s30)
    return long_, short & ~long_, {"stop": ("atr", 3.0), "target": None, "exit_long": z > 0, "exit_short": z < 0}


# ---------------- GH12 Relative momentum acceleration ----------------
def _kama(c, n=30, fast=2, slow=30):
    out = np.full(len(c), np.nan)
    fsc, ssc = 2 / (fast + 1), 2 / (slow + 1)
    vol = _s(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    for i in range(n, len(c)):
        er = abs(c[i] - c[i - n]) / vol[i] if vol[i] > 0 else 0
        sc = (er * (fsc - ssc) + ssc) ** 2
        prev = out[i - 1] if np.isfinite(out[i - 1]) else c[i - 1]
        out[i] = prev + sc * (c[i] - prev)
    return out


def s_thrust(df):
    """Relative momentum acceleration: thrust = (EMA7 - KAMA30) / KAMA30; thrust above the upper
    Bollinger band of itself (7, 1 stdev) -> long, below the lower -> short. Chandelier trail 3 ATR."""
    c = df["c"].to_numpy()
    k = _kama(c)
    th = (ema(c, 7) - k) / (k + 1e-6)
    m, sd = sma(th, 7), _s(th).rolling(7).std(ddof=0).to_numpy()
    up, dn = th > m + sd, th < m - sd
    return up & ~np.r_[False, up[:-1]], dn & ~np.r_[False, dn[:-1]], {"stop": ("atr", 3.0), "target": None,
                                                                       "trail_atr": 3.0}


# ---------------- GH13 Ichimoku 7/14/30 ----------------
def s_ichimoku_fast(df):
    """Ichimoku 7/14/30: close above both cloud spans (drawn 14 bars ahead), Tenkan above Kijun and the
    close above the high of 14 bars ago (the Chikou rule without look-ahead) -> long; mirror short. The
    source trails 2% of price -> 3 ATR trail here; initial stop 3 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    mid = lambda n: (_s(h).rolling(n).max() + _s(l).rolling(n).min()).to_numpy() / 2  # noqa: E731
    tk, kj = mid(7), mid(14)
    sa = _s((tk + kj) / 2).shift(14).to_numpy()
    sb = _s(mid(30)).shift(14).to_numpy()
    h14, l14 = _s(h).shift(14).to_numpy(), _s(l).shift(14).to_numpy()
    up = (c > np.fmax(sa, sb)) & (tk > kj) & (c > h14)
    dn = (c < np.fmin(sa, sb)) & (tk < kj) & (c < l14)
    return up & ~np.r_[False, up[:-1]], dn & ~np.r_[False, dn[:-1]], {"stop": ("atr", 3.0), "target": None,
                                                                       "trail_atr": 3.0}


# ---------------- GH14 Kalman trend slope (QuanterLab) ----------------
def _kalman_slope(y, q=1e-8, r=0.03):
    xk = np.array([y[0], 0.0])
    P = np.eye(2) * 1.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = np.eye(2) * q
    out = np.zeros(len(y))
    for i, v in enumerate(y):
        xk = F @ xk
        P = F @ P @ F.T + Q
        S = H @ P @ H + r
        K = P @ H / S
        xk = xk + K * (v - H @ xk)
        IKH = np.eye(2) - np.outer(K, H)
        P = IKH @ P @ IKH.T + np.outer(K, K) * r
        out[i] = xk[1]
    return out


def s_kalman_trend(df):
    """Kalman local-linear-trend filter on the log of the daily close (QuanterLab study): slope > 0 at
    the daily close -> long from the next day's open, slope <= 0 -> out. Long only. Stop 3 ATR (ours),
    250 days burn-in."""
    from agents.book_strategies import _daily_signals, _days
    _, day, _, _ = _days(df)
    y = np.log(day["c"].to_numpy())
    sl = _kalman_slope(y)
    sl[:250] = np.nan
    s = pd.Series(sl, index=day.index)
    long_d = pd.Series((s > 0) & ~(s.shift(1) > 0), index=day.index)
    return _daily_signals(df, long_d, pd.Series(False, index=day.index), pd.Series(s <= 0, index=day.index),
                          pd.Series(False, index=day.index))


# ---------------- GH15 MMM "Maximas e Minimas" (jesse) ----------------
def s_mmm(df):
    """MMM: close above its 30-bar average and below the 3-bar average of the lows -> long; out when
    the close is above the 3-bar average of the highs. Stop 3 ATR (ours)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    long_ = (c > sma(c, 30)) & (c < sma(l, 3))
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": c > sma(h, 3)}


# id: (name, fn, timeframes)
GITHUB = {
    "GH01": ("OTT trend tracker, ADX>60 exit (freqtrade)", s_ott, ["1Hour"]),
    "GH02": ("Three Supertrends agree (freqtrade)", s_triple_supertrend, ["1Hour"]),
    "GH03": ("HLHB: RSI10 50-cross + EMA5/10 + ADX (freqtrade)", s_hlhb, ["4Hour", "1Hour"]),
    "GH04": ("TD Sequential setup 9, perfected (freqtrade)", s_td_sequential, ["1Hour", "4Hour"]),
    "GH05": ("CofiBit stochastic cross under 25 (freqtrade)", s_cofibit, ["5Min", "15Min"]),
    "GH06": ("Volatility System, 3h move > 2 ATR, stop-and-reverse (freqtrade)", s_volatility_system, ["1Hour"]),
    "GH07": ("Strategy002: RSI + stoch + BB + hammer, SAR exit (freqtrade)", s_strategy002, ["5Min", "15Min"]),
    "GH08": ("Dual Thrust (jesse)", s_dual_thrust, ["1Hour", "4Hour"]),
    "GH09": ("Squeeze Momentum release (LazyBear)", s_squeeze, ["1Hour", "4Hour"]),
    "GH10": ("Momentum Ignition: ROC breaks its 1-sigma band (backtrader)", s_momentum_ignition, ["1Hour", "4Hour"]),
    "GH11": ("Ornstein-Uhlenbeck z-score reversion (backtrader)", s_ou_zscore, ["1Hour", "4Hour"]),
    "GH12": ("EMA7-KAMA30 thrust beyond its band (backtrader)", s_thrust, ["1Hour", "4Hour"]),
    "GH13": ("Ichimoku 7/14/30 with Chikou check (backtrader)", s_ichimoku_fast, ["4Hour", "1Hour"]),
    "GH14": ("Kalman trend slope, daily (QuanterLab)", s_kalman_trend, ["1Hour"]),
    "GH15": ("MMM: under the 3-bar low average in an uptrend (jesse)", s_mmm, ["1Hour"]),
}
