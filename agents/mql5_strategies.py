"""Expert advisors from open MQL4/MQL5 source code on GitHub (geraked/metatrader5, giovabiancia/trading,
matthewkastor/Metatrader, EarnForex). The MQL5 Code Base itself does not let its source be downloaded,
so these come from repositories that publish theirs. Same contract as agents/rules_agent.py.

Port rules (ours): closed bars only (no reading of the forming bar); grids / martingale switched off;
geraked's "IgnoreSL" stops used as real stops; one position at a time; buy-stop / sell-stop entries
become "the first bar that closes beyond the level, filled at the next open". Fixed pip sizes are
scaled to every instrument as a share of the daily ATR (1 pip = daily ATR / 70, EURUSD-typical).
Broker-clock windows are moved to London time (what the authors meant: the London session).

MQ01 2MOMEMA  Momentum(10) crosses 100 inside a Momentum(34) + EMA10/34 regime        (geraked)
MQ02 2MAAOS   Andean Oscillator signal cross, SMA50/200 trend, pullback limit          (geraked)
MQ03 3MAF     Fractal pullback into an EMA 60/350/600 stack                            (geraked)
MQ04 CEZLSMA  Chandelier Exit flip on Heikin Ashi, ZLSMA(50) side filter               (geraked)
MQ05 DHLAOS   Andean signal cross after a move beyond the previous day's range          (geraked)
MQ06 2MACDSTO Stochastic out of an extreme while MACD(34,144) and MACD(13,21) disagree (geraked)
MQ07 AO3B     Hughes: two counter bars against the Awesome Oscillator, stop entry       (giovabiancia)
MQ08 Tolman   Envelopes(800, 2%) fade toward the SMA800, 4-week extreme filter          (giovabiancia)
MQ09 FirstMv  Fade the first 15-pip pullback from the session extreme, flat at 13:00    (giovabiancia)
MQ10 Keltner  Pullback below the 30-bar midpoint and a rising EMA30, +/-3 ATR bracket   (matthewkastor)
MQ11 HA Naive Inverted Heiken Ashi: fade a growing wickless HA candle                   (EarnForex)
MQ12 Straddle 08:00 London straddle beyond R1/S1, stop at the mid-pivot, target R2/S2   (giovabiancia)
MQ13 PinBar   Pin bar with the body inside the prior bar, ATR trail after +30 pips      (giovabiancia)
"""
import numpy as np
import pandas as pd

from agents.book_strategies import _days, _on_bars
from agents.rules_agent import atr, ema, sma

LDN = "Europe/London"


def _sh(x, k):
    x = np.asarray(x, float)
    return np.r_[np.full(k, np.nan), x[:-k]] if k else x


def _pip(df):
    """One EURUSD-pip equivalent per bar: yesterday's daily ATR(14) / 70."""
    date, day, _, _ = _days(df)
    pc = day["c"].shift(1)
    tr = pd.concat([day["h"] - day["l"], (day["h"] - pc).abs(), (day["l"] - pc).abs()], axis=1).max(axis=1)
    return _on_bars(date, tr.rolling(14).mean().shift(1)) / 70.0


def _roll_min(x, n):
    return pd.Series(x).rolling(n).min().to_numpy()


def _roll_max(x, n):
    return pd.Series(x).rolling(n).max().to_numpy()


def _swing_stop(df, n, dev_pips):
    h, l = df["h"].to_numpy(), df["l"].to_numpy()
    p = _pip(df)
    return _roll_min(l, n) - dev_pips * p, _roll_max(h, n) + dev_pips * p


def andean(o, c, n=50, sig=9):
    """Andean Oscillator (alexgrover): bull / bear 'volatility' of up and down envelopes, plus an
    EMA(9) signal line of the larger of the two."""
    a = 2.0 / (n + 1)
    N = len(c)
    up1, up2, dn1, dn2 = (np.zeros(N) for _ in range(4))
    up1[0] = dn1[0] = c[0]
    up2[0] = dn2[0] = c[0] ** 2
    for i in range(1, N):
        ci, oi = c[i], o[i]
        up1[i] = max(ci, oi, up1[i - 1] - (up1[i - 1] - ci) * a)
        up2[i] = max(ci * ci, oi * oi, up2[i - 1] - (up2[i - 1] - ci * ci) * a)
        dn1[i] = min(ci, oi, dn1[i - 1] + (ci - dn1[i - 1]) * a)
        dn2[i] = min(ci * ci, oi * oi, dn2[i - 1] + (ci * ci - dn2[i - 1]) * a)
    bull = np.sqrt(np.maximum(dn2 - dn1 * dn1, 0))
    bear = np.sqrt(np.maximum(up2 - up1 * up1, 0))
    signal = ema(np.maximum(bull, bear), sig)
    bull[:n], bear[:n] = np.nan, np.nan
    return bull, bear, signal


def heikin(df):
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    hc = (o + h + l + c) / 4
    ho = np.empty(len(c))
    ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    hh = np.maximum.reduce([h, ho, hc])
    hl = np.minimum.reduce([l, ho, hc])
    return ho, hh, hl, hc


def _linreg(x, n):
    """End-point of the least-squares line over the last n values (MT/TradingView 'linreg', offset 0)."""
    x = pd.Series(np.asarray(x, float))
    idx = np.arange(n, dtype=float)
    w = idx - idx.mean()
    den = (w * w).sum()

    def f(v):
        b = (w * (v - v.mean())).sum() / den
        return v.mean() + b * (n - 1 - idx.mean())
    return x.rolling(n).apply(f, raw=True).to_numpy()


# ---------------------------------------------------------------- MQ01
def s_2momema(df):
    """Momentum(10) crosses up through 100 while, on each of the last 9 bars, Momentum(34) > 100 and
    EMA10 > EMA34 -> long (mirror short). Stop: 10-bar low - 6 pips; target 1R."""
    c = df["c"].to_numpy()
    m10 = 100 * c / _sh(c, 10)
    m34 = 100 * c / _sh(c, 34)
    e10, e34 = ema(c, 10), ema(c, 34)
    up9 = pd.Series((m34 > 100) & (e10 > e34)).rolling(9).sum().to_numpy() == 9
    dn9 = pd.Series((m34 < 100) & (e10 < e34)).rolling(9).sum().to_numpy() == 9
    L = (m10 > 100) & (_sh(m10, 1) <= 100) & up9
    S = (m10 < 100) & (_sh(m10, 1) >= 100) & dn9
    sl, ss = _swing_stop(df, 10, 6)
    return L, S, {"stop": ("level", sl, ss), "target": ("r", 1.0)}


# ---------------------------------------------------------------- MQ02
def s_2maaos(df):
    """Andean bull line crosses above its signal, bull > bear, SMA50 > SMA200, price not deeper than
    halfway from SMA50 back to SMA200 -> long (mirror short). Stop: 10-bar low - 10 pips; no target;
    a 0.5R trailing stop once 0.5R in profit."""
    o, c = df["o"].to_numpy(), df["c"].to_numpy()
    bu, be, sg = andean(o, c)
    s50, s200 = sma(c, 50), sma(c, 200)
    half = 0.5 * np.abs(s50 - s200)
    L = (_sh(bu, 1) <= _sh(sg, 1)) & (bu > sg) & (bu > be) & (s50 > s200) & (c > s50 - half)
    S = (_sh(be, 1) <= _sh(sg, 1)) & (be > sg) & (bu < be) & (s50 < s200) & (c < s50 + half)
    sl, ss = _swing_stop(df, 10, 10)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": None, "trail_r": 0.5}


# ---------------------------------------------------------------- MQ03
def s_3maf(df):
    """EMA60 > EMA350 > EMA600, the bar dipped under EMA60 but stayed above EMA600, and a down fractal
    (5-bar) is confirmed at the bar two back -> long. Stop: EMA350 (EMA600 if the bar went below the
    EMA350); 0.5R trailing stop. Mirror short."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    e1, e2, e3 = ema(c, 60), ema(c, 350), ema(c, 600)
    fd = (_sh(l, 2) < _sh(l, 3)) & (_sh(l, 2) < _sh(l, 4)) & (_sh(l, 2) < _sh(l, 1)) & (_sh(l, 2) < l)
    fu = (_sh(h, 2) > _sh(h, 3)) & (_sh(h, 2) > _sh(h, 4)) & (_sh(h, 2) > _sh(h, 1)) & (_sh(h, 2) > h)
    L = (e1 > e2) & (e2 > e3) & (l > e3) & (l < e1) & fd
    S = (e3 > e2) & (e2 > e1) & (h < e3) & (h > e1) & fu
    sl = np.where(l > e2, e2, e3)
    ss = np.where(h < e2, e2, e3)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": None, "trail_r": 0.5}


# ---------------------------------------------------------------- MQ04
def s_cezlsma(df):
    """Chandelier Exit (ATR 1, x0.75) on Heikin Ashi flips up and the HA close is above ZLSMA(50) ->
    long (mirror short). Stop: the chandelier line - 65 pips; out when the HA close crosses back over
    the ZLSMA; 0.5R trailing stop."""
    ho, hh, hl, hc = heikin(df)
    n = len(hc)
    tr = np.maximum.reduce([hh - hl, np.abs(hh - _sh(hc, 1)), np.abs(hl - _sh(hc, 1))])
    tr[0] = hh[0] - hl[0]
    ls, ss_ = hc - 0.75 * tr, hc + 0.75 * tr
    d = np.ones(n)
    for i in range(1, n):
        if hc[i - 1] > ls[i - 1]:
            ls[i] = max(ls[i], ls[i - 1])
        if hc[i - 1] < ss_[i - 1]:
            ss_[i] = min(ss_[i], ss_[i - 1])
        d[i] = 1 if hc[i] > ss_[i - 1] else (-1 if hc[i] < ls[i - 1] else d[i - 1])
    lr = _linreg(hc, 50)
    zl = lr + (lr - _linreg(lr, 50))
    L = (d == 1) & (_sh(d, 1) == -1) & (hc > zl)
    S = (d == -1) & (_sh(d, 1) == 1) & (hc < zl)
    p = _pip(df)
    xl = (_sh(hc, 1) >= _sh(zl, 1)) & (hc < zl)
    xs = (_sh(hc, 1) <= _sh(zl, 1)) & (hc > zl)
    return L, S, {"stop": ("level", ls - 65 * p, ss_ + 65 * p), "target": None, "trail_r": 0.5,
                  "exit_long": xl, "exit_short": xs}


# ---------------------------------------------------------------- MQ05
def s_dhlaos(df):
    """Andean bull line crosses above its signal for the first time since the bull line overtook the
    bear line, and within the 50 bars before that overtaking a bar traded wholly below yesterday's low
    while bearish -> long (mirror short). Stop: 7-bar low - 6 pips; 0.5R trailing stop."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    bu, be, sg = andean(o, c)
    date, day, _, _ = _days(df)
    pdl, pdh = _on_bars(date, day["l"].shift(1)), _on_bars(date, day["h"].shift(1))
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    below = (h < pdl) & (bu < be)
    above = (l > pdh) & (bu > be)
    last_bull_x, last_bear_x = -1, -1      # bars where bull crossed above / below bear
    crossed_l, crossed_s = False, False    # a signal-line cross already happened since then
    for i in range(1, n):
        if not (np.isfinite(bu[i - 1]) and np.isfinite(sg[i - 1])):
            continue
        if bu[i - 1] <= be[i - 1] and bu[i] > be[i]:
            last_bull_x, crossed_l = i, False
        if bu[i - 1] >= be[i - 1] and bu[i] < be[i]:
            last_bear_x, crossed_s = i, False
        x_bull_sig = bu[i - 1] <= sg[i - 1] and bu[i] > sg[i]
        x_bear_sig = be[i - 1] <= sg[i - 1] and be[i] > sg[i]
        if x_bull_sig and bu[i] > be[i] and last_bull_x > 0 and not crossed_l:
            j = last_bull_x
            if below[max(0, j - 50):j + 1].any():
                L[i] = True
        if x_bear_sig and be[i] > bu[i] and last_bear_x > 0 and not crossed_s:
            j = last_bear_x
            if above[max(0, j - 50):j + 1].any():
                S[i] = True
        if x_bull_sig or (bu[i - 1] >= sg[i - 1] and bu[i] < sg[i]):
            crossed_l = True
        if x_bear_sig or (be[i - 1] >= sg[i - 1] and be[i] < sg[i]):
            crossed_s = True
    sl, ss = _swing_stop(df, 7, 6)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": None, "trail_r": 0.5}


# ---------------------------------------------------------------- MQ06
def s_2macdsto(df):
    """MACD(34,144) > 0 (up) while MACD(13,21) < 0 (pullback), and Stochastic(7,3,3) %K crosses above
    %D from under 20 -> long (mirror short). Stop: 7-bar low - 6 pips; 0.5R trailing stop."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    m1 = ema(c, 13) - ema(c, 21)
    m2 = ema(c, 34) - ema(c, 144)
    ll, hh = _roll_min(l, 7), _roll_max(h, 7)
    raw = 100 * (c - ll) / np.where(hh > ll, hh - ll, np.nan)
    k = sma(raw, 3)
    d = sma(k, 3)
    L = (_sh(m2, 1) > 0) & (_sh(m1, 1) < 0) & (_sh(k, 1) < 20) & (_sh(k, 1) <= _sh(d, 1)) & (k > d)
    S = (_sh(m2, 1) < 0) & (_sh(m1, 1) > 0) & (_sh(k, 1) > 80) & (_sh(k, 1) >= _sh(d, 1)) & (k < d)
    sl, ss = _swing_stop(df, 7, 6)
    return L, S, {"stop": ("level", sl, ss), "target": None, "trail_r": 0.5}


# ---------------------------------------------------------------- MQ07
def s_ao3b(df):
    """London 07:30-17:30: Awesome Oscillator > 0 and two bearish bars, the second closing under the
    first's low -> long on a close above that bar's high + 3 pips on the next bar. Hughes' PMOption 1
    (one position): stop 30 pips, target 50 pips, out after 8 bars. Mirror short (1 pip)."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    mid = (h + l) / 2
    ao = sma(mid, 5) - sma(mid, 34)
    p = _pip(df)
    t = df.index.tz_convert(LDN)
    mins = (t.hour * 60 + t.minute).to_numpy()
    ok = (mins >= 450) & (mins <= 1050)
    lset = (ao > 0) & (c < o) & (_sh(c, 1) < _sh(o, 1)) & (c < _sh(l, 1))
    sset = (ao < 0) & (c > o) & (_sh(c, 1) > _sh(o, 1)) & (c > _sh(h, 1))
    L = _sh(lset, 1).astype(bool) & ok & (c > _sh(h, 1) + 3 * p)
    S = _sh(sset, 1).astype(bool) & ok & (c < _sh(l, 1) - 1 * p)
    return L, S & ~L, {"stop": ("level", c - 30 * p, c + 30 * p),
                       "target": ("level", c + 50 * p, c - 50 * p), "max_bars": 8}


# ---------------------------------------------------------------- MQ08
def s_tolman(df):
    """The bar opens above the lower envelope (SMA800 - 2%) and trades below it, while the lowest low
    of the last 4 weeks has reached the envelope -> long; out back at the SMA800 (LetItRide = 0).
    Stop 1 daily ATR. Mirror short at the upper envelope."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    s = _sh(sma(c, 800), 1)
    lo_env, hi_env = s * 0.98, s * 1.02
    date, day, _, _ = _days(df)
    lo4 = _on_bars(date, day["l"].rolling(20).min().shift(1))
    hi4 = _on_bars(date, day["h"].rolling(20).max().shift(1))
    L = (o > lo_env) & (l < lo_env) & (lo4 <= lo_env)
    S = (o < hi_env) & (h > hi_env) & (hi4 >= hi_env)
    dat = 70 * _pip(df)
    return L, S & ~L, {"stop": ("level", c - dat, c + dat), "target": None,
                       "exit_long": c >= s, "exit_short": c <= s}


# ---------------------------------------------------------------- MQ09
def s_first_move(df):
    """London 05:00-13:00: the session high so far (from the second bar after 05:00); the first bar
    whose low reaches 15 pips under it -> long (mirror short from the session low); one trade a day.
    Stop 15 pips, target 40 pips, stop to entry after +15 pips, flat at 13:00."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    p = _pip(df)
    t = df.index.tz_convert(LDN)
    step = df.index.to_series().diff().median()
    mins = (t.hour * 60 + t.minute).to_numpy()
    end = ((df.index + step).tz_convert(LDN))
    end_m = (end.hour * 60 + end.minute).to_numpy()
    dkey = np.array([d.isoformat() for d in t.date])
    win = (mins >= 300 + int(step / pd.Timedelta(minutes=1))) & (mins < 780)
    hh = pd.Series(np.where(win, h, np.nan)).groupby(dkey).cummax().to_numpy()
    ll = pd.Series(np.where(win, l, np.nan)).groupby(dkey).cummin().to_numpy()
    ph, pl = _sh(hh, 1), _sh(ll, 1)
    same = np.r_[False, dkey[1:] == dkey[:-1]]
    lo = win & same & (l <= ph - 15 * p) & (end_m < 780)
    sh = win & same & (h >= pl + 15 * p) & (end_m < 780)
    first = pd.Series(lo | sh).groupby(dkey).cumsum().to_numpy() == 1
    L, S = lo & first, sh & first & ~lo
    return L, S, {"stop": ("level", c - 15 * p, c + 15 * p), "target": ("level", c + 40 * p, c - 40 * p),
                  "breakeven": (0.375, 0.0), "force_exit": end_m == 780}


# ---------------------------------------------------------------- MQ10
def s_keltner_pullback(df):
    """Both now and 30 bars ago: price at or below the 30-bar range midpoint, and the EMA30 of the
    typical price rising over 30 bars with price at or below it -> long (mirror short). Stop and target
    3 x the smaller of ATR30 now / 30 bars ago (1:1); an opposite signal closes the trade."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    tp = (h + l + c) / 3
    e = ema(tp, 30)
    mid = (_roll_max(h, 30) + _roll_min(l, 30)) / 2
    a30 = atr(h, l, c, 30)
    L = np.ones(len(c), bool)
    S = np.ones(len(c), bool)
    for s in (0, 30):
        ms, es, es2 = _sh(mid, s), _sh(e, s), _sh(e, s + 30)
        L &= (c <= ms) & (es > es2) & (c <= es)
        S &= (c >= ms) & (es < es2) & (c >= es)
    k = 3 * np.minimum(a30, _sh(a30, 30))
    return L, S & ~L, {"stop": ("level", c - k, c + k), "target": ("level", c + k, c - k), "opposite": True}


# ---------------------------------------------------------------- MQ11
def s_ha_naive(df):
    """Inverted Heiken Ashi Naive: a bearish HA candle with no upper wick and a body bigger than the
    previous (bearish) one -> LONG; the bullish mirror -> SHORT. Out on the opposite close pattern or an
    opposite signal. Stop 1 ATR (the EA's sizing stop)."""
    ho, hh, hl, hc = heikin(df)
    tol = 1e-9 * np.abs(hc)
    body = hc - ho
    bear1, bull1 = body < 0, body > 0
    bear2, bull2 = _sh(body, 1) < 0, _sh(body, 1) > 0
    grow = np.abs(body) > np.abs(_sh(body, 1))
    no_up = np.abs(hh - ho) <= tol
    no_dn = np.abs(hl - ho) <= tol
    L = bear1 & grow & bear2 & no_up
    S = bull1 & grow & bull2 & no_dn
    xl = bull1 & no_dn & bull2
    xs = bear1 & no_up & bear2
    return L, S, {"stop": ("atr", 1.0), "target": None, "exit_long": xl, "exit_short": xs, "opposite": True}


# ---------------------------------------------------------------- MQ12
def s_straddle_pivots(df):
    """Pivots of the 24 hours to 08:00 London. If 08:00 opens between S1 and R1: the first close above
    R1 + 3 pips buys / below S1 - 3 pips sells (one trade a day, until 20:00). Stop at the pivot-R1 /
    pivot-S1 midpoint -/+ 3 pips; target R2 / S2; stop to entry halfway."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    p = _pip(df)
    t = df.index.tz_convert(LDN)
    mins = (t.hour * 60 + t.minute).to_numpy()
    shifted = (t - pd.Timedelta(hours=8))
    key = np.array([d.isoformat() for d in shifted.date])      # the "day" runs 08:00 -> 08:00
    g = pd.DataFrame({"k": key, "h": h, "l": l, "c": c}).groupby("k", sort=False)
    dh, dl, dc = g["h"].max(), g["l"].min(), g["c"].last()
    P = (dh + dl + dc) / 3
    R1, S1, R2, S2 = 2 * P - dl, 2 * P - dh, P + (dh - dl), P - (dh - dl)
    prev = lambda s: pd.Series(key).map(s.shift(1)).to_numpy(dtype=float)  # noqa: E731
    p_, r1, s1, r2, s2 = (prev(x) for x in (P, R1, S1, R2, S2))
    open8 = pd.Series(np.where(mins == 480, o, np.nan)).groupby(key).transform("max").to_numpy()
    armed = (open8 >= s1) & (open8 <= r1) & (mins >= 480) & (mins < 1200)
    up = armed & (c > r1 + 3 * p)
    dn = armed & (c < s1 - 3 * p)
    first = pd.Series(up | dn).groupby(key).cumsum().to_numpy() == 1
    m3, m2 = (p_ + r1) / 2, (p_ + s1) / 2
    return up & first, dn & first & ~up, {"stop": ("level", m3 - 3 * p, m2 + 3 * p),
                                          "target": ("level", r2, s2), "breakeven": (0.5, 0.0)}


# ---------------------------------------------------------------- MQ13
def s_pin_bar(df):
    """Bullish pin: nose > 60% of the range, body < 40%, the body inside the prior bar's range and the
    low under the prior low; long on the first close above the pin's high within 4 bars (16 hours on
    4H). Stop at the pin's low, target 200 pips; once +30 pips the stop trails 1 ATR. Mirror short."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    n = len(c)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        body = np.abs(o - c) / rng
        nose_l = (np.minimum(o, c) - l) / rng
        nose_s = (h - np.maximum(o, c)) / rng
    inside = (np.maximum(o, c) < _sh(h, 1)) & (np.minimum(o, c) > _sh(l, 1))
    pl = (rng > 0) & (nose_l > 0.6) & (body < 0.4) & inside & (l < _sh(l, 1))
    ps = (rng > 0) & (nose_s > 0.6) & (body < 0.4) & inside & (h > _sh(h, 1))
    p = _pip(df)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stl, sts = np.full(n, np.nan), np.full(n, np.nan)
    for t in np.flatnonzero(pl | ps):
        for k in range(t + 1, min(n, t + 5)):
            if pl[t] and c[k] > h[t]:
                L[k], stl[k] = True, l[t]
                break
            if ps[t] and c[k] < l[t]:
                S[k], sts[k] = True, h[t]
                break
    return L, S & ~L, {"stop": ("level", stl, sts), "target": ("level", c + 200 * p, c - 200 * p),
                       "trail_atr": 1.0, "trail_after_px": 30 * p}


# id: (name, fn, timeframes)
MQL5 = {
    "MQ01": ("2MOMEMA momentum cross in an EMA regime (geraked EA)", s_2momema, ["1Hour"]),
    "MQ02": ("2MAAOS Andean Oscillator + SMA50/200 (geraked EA)", s_2maaos, ["1Hour", "4Hour"]),
    "MQ03": ("3MAF fractal pullback into EMA 60/350/600 (geraked EA)", s_3maf, ["15Min"]),
    "MQ04": ("CEZLSMA Chandelier on Heikin Ashi + ZLSMA (geraked EA)", s_cezlsma, ["15Min"]),
    "MQ05": ("DHLAOS Andean cross after a prior-day range excursion (geraked EA)", s_dhlaos, ["15Min"]),
    "MQ06": ("2MACDSTO Stochastic with opposing MACDs (geraked EA)", s_2macdsto, ["1Hour", "4Hour"]),
    "MQ07": ("Hughes AO3B Awesome Oscillator pullback stop entry (EA)", s_ao3b, ["1Hour"]),
    "MQ08": ("Tolman Envelopes(800,2%) reversion (EA)", s_tolman, ["1Hour"]),
    "MQ09": ("First Move: fade the first 15-pip pullback, London (EA)", s_first_move, ["15Min"]),
    "MQ10": ("Keltner pullback trader, 2-window concurrence (EA)", s_keltner_pullback, ["1Hour"]),
    "MQ11": ("Heiken Ashi Naive, inverted (EarnForex EA)", s_ha_naive, ["4Hour"]),
    "MQ12": ("2ndSkies 08:00 London pivot straddle (EA)", s_straddle_pivots, ["15Min"]),
    "MQ13": ("Pin bar trader with ATR trail (EA)", s_pin_bar, ["4Hour"]),
}
