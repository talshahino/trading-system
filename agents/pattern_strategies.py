"""Quantified Strategies tests and the classic pattern / system books, coded from their published rules.
Same contract as agents/rules_agent.py.

QB01 Bullish reversal day + RSI(5)<35, 20-day hold          Quantified Strategies (gold)
QB02 Bearish engulfing bought, out on a close > prior high   Quantified Strategies (indices)
QB03 Bearish three-line strike -> upward breakout            Bulkowski, Encyclopedia of Candlestick Charts
QB04 Bullish three-line strike -> downward breakout          Bulkowski
QB05 Morning / evening star breakout                         Bulkowski
QB06 Two black gapping candles breakout                      Bulkowski
QB07 Head-and-shoulders bottom / top, neckline break         Bulkowski, Encyclopedia of Chart Patterns
QB08 Three rising valleys / three falling peaks              Bulkowski
QB09 Rectangle breakout                                      Bulkowski
QB10 Kaufman Adaptive Moving Average + std filter            Kaufman, Trading Systems and Methods
QB11 Pivot-break top / bottom reversal, 8R target            Crabel / Wyckoff (Oxfordstrat codification)
QB12 Modified Hikkake, 20-bar hold                           D. Chapman (Oxfordstrat codification)
QB13 Hidden Smash Day (type A), 3-day hold                   Larry Williams
QB14 65sma-3cc trend following                               Chande, Beyond Technical Analysis
QB15 Three black crows, upward breakout only                 Bulkowski

Daily patterns use the New York trading day built from the intraday bars (book_strategies._days): a
signal known at a day's close enters on the next day's first bar (inside the instrument's own trading
hours), and a day-count exit leaves at the first bar of the day after the hold. Where the source has no
stop, ours is in daily ATRs and noted. Bulkowski's "engine assumptions" (prior trend = close 10 bars
earlier, tall candle = range above its 22-bar average) are his researchers' usual reading.
"""
import numpy as np
import pandas as pd

from agents.book_strategies import _days, _on_bars
from agents.rules_agent import atr, rsi, sma

NONE = None


# ---------------------------------------------------------------- helpers
def _datr(day, n=14):
    pc = day["c"].shift(1)
    tr = pd.concat([day["h"] - day["l"], (day["h"] - pc).abs(), (day["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _f(day):
    return pd.Series(False, index=day.index)


def _nan(day):
    return pd.Series(np.nan, index=day.index)


def _daily_trade(df, L, S, sl_l, sl_s, tg_l=None, tg_s=None, hold=None, ex_l=None, ex_s=None, opposite=False):
    """Day-level signals (known at the day's close) -> bar-level plan. Entry at the next day's first bar;
    stop / target levels fixed from the signal day; exit after `hold` days or on an exit condition known
    at a day's close (left at the next day's first bar)."""
    date, day, first, last = _days(df)
    prev = lambda s: _on_bars(date, pd.Series(s, index=day.index).astype(float).shift(1))  # noqa: E731
    el = first & (prev(L) > 0)
    es = first & (prev(S) > 0) & ~el
    plan = {"stop": ("level", prev(sl_l), prev(sl_s)), "target": None}
    if tg_l is not None:
        plan["target"] = ("level", prev(tg_l), prev(tg_s))
    xl = np.zeros(len(df), bool)
    xs = np.zeros(len(df), bool)
    nf = np.r_[first[1:], False]          # the bar before a day's first bar: exit there -> filled at the open
    if hold:
        xl |= np.r_[(prev(pd.Series(L, index=day.index).shift(hold)) > 0)[1:], False] & nf
        xs |= np.r_[(prev(pd.Series(S, index=day.index).shift(hold)) > 0)[1:], False] & nf
    if ex_l is not None:
        xl |= np.r_[(prev(ex_l) > 0)[1:], False] & nf
    if ex_s is not None:
        xs |= np.r_[(prev(ex_s) > 0)[1:], False] & nf
    plan["exit_long"], plan["exit_short"] = xl, xs
    if opposite:
        plan["opposite"] = True
    return el, es, plan


def _breakouts(day, pat_l, pat_s, hi, lo, window=10):
    """Bulkowski breakout: after a pattern completes, the first close above its high (up) or below its
    low (down) within `window` days. Returns per-day up / down flags and the pattern's high / low."""
    c = day["c"].to_numpy()
    n = len(c)
    up, dn = np.zeros(n, bool), np.zeros(n, bool)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    pat_l, pat_s = np.asarray(pat_l, bool), np.asarray(pat_s, bool)
    hi, lo = np.asarray(hi, float), np.asarray(lo, float)
    for t in np.flatnonzero(pat_l | pat_s):
        for k in range(t + 1, min(n, t + 1 + window)):
            if pat_l[t] and c[k] > hi[t]:
                up[k], ph[k], pl[k] = True, hi[t], lo[t]
                break
            if pat_s[t] and c[k] < lo[t]:
                dn[k], ph[k], pl[k] = True, hi[t], lo[t]
                break
    return up, dn, ph, pl


def _bracket_breakout(df, day, up, dn, ph, pl, hold=10, stop_buf=0.0):
    """Long on an up breakout (stop = pattern low, target = high + height), short on a down breakout
    (stop = pattern high, target = low - height); time exit after `hold` days."""
    da = _datr(day).to_numpy()
    hgt = ph - pl
    idx = day.index
    S = lambda x: pd.Series(x, index=idx)  # noqa: E731
    return _daily_trade(df, S(up), S(dn), S(pl - stop_buf * da), S(ph + stop_buf * da),
                        S(ph + hgt), S(pl - hgt), hold=hold)


def _ohlc(day):
    return (day[x].to_numpy() for x in "ohlc")


def _sh(x, k):
    x = np.asarray(x, float)
    return np.r_[np.full(k, np.nan), x[:-k]] if k else x


def _trend(c, k_before, up=True):
    """Prior trend: the close of the bar before the pattern vs. the close 10 bars before that."""
    a, b = _sh(c, k_before), _sh(c, k_before + 10)
    return a > b if up else a < b


def _tall(h, l, k):
    """Tall candle at shift k: range above the average range of the 22 bars before the pattern."""
    r = h - l
    avg = pd.Series(r).rolling(22).mean().shift(4).to_numpy()
    return _sh(r, k) > avg


# ---------------------------------------------------------------- QB01 / QB02 Quantified Strategies
def s_reversal_rsi5(df):
    """Daily: a lower low but a higher close than yesterday, with RSI(5) under 35 -> long; held 20 days.
    No stop in the test; ours: the signal day's low - 2 daily ATRs."""
    _, day, _, _ = _days(df)
    o, h, l, c = _ohlc(day)
    sig = (l < _sh(l, 1)) & (c > _sh(c, 1)) & (rsi(c, 5) < 35)
    sl = day["l"] - 2 * _datr(day)
    return _daily_trade(df, pd.Series(sig, index=day.index), _f(day), sl, _nan(day), hold=20)


def s_engulf_buy(df):
    """Daily bearish engulfing -> buy at the close (next day's first bar); out on the first close above
    the previous day's high. No stop in the test; ours: the pattern low - 1 daily ATR, 20-day cap."""
    _, day, _, _ = _days(df)
    o, h, l, c = _ohlc(day)
    o1, c1 = _sh(o, 1), _sh(c, 1)
    eng = (c1 > o1) & (c < o) & (o >= c1) & (c <= o1) & ((o - c) > (c1 - o1))
    ex = pd.Series(c > _sh(h, 1), index=day.index)
    sl = pd.Series(np.minimum(l, _sh(l, 1)), index=day.index) - _datr(day)
    return _daily_trade(df, pd.Series(eng, index=day.index), _f(day), sl, _nan(day), hold=20, ex_l=ex)


# ---------------------------------------------------------------- QB03-QB06, QB15 Bulkowski candles
def _candle_bo(df, fn, hold=10):
    _, day, _, _ = _days(df)
    o, h, l, c = _ohlc(day)
    pl_, ps_, n = fn(o, h, l, c)
    hi = pd.Series(h).rolling(n).max().to_numpy()
    lo = pd.Series(l).rolling(n).min().to_numpy()
    up, dn, ph, pl = _breakouts(day, pl_, ps_, hi, lo)
    return _bracket_breakout(df, day, up, dn, ph, pl, hold)


def _bear_strike(o, h, l, c):
    b = lambda k: _sh(c, k) < _sh(o, k)  # noqa: E731
    p = (_trend(c, 4, up=False) & b(3) & b(2) & b(1) & (_sh(l, 2) < _sh(l, 3)) & (_sh(l, 1) < _sh(l, 2))
         & (c > o) & (o < _sh(c, 1)) & (c > _sh(o, 3)))
    return p, np.zeros(len(c), bool), 4


def _bull_strike(o, h, l, c):
    w = lambda k: _sh(c, k) > _sh(o, k)  # noqa: E731
    p = (_trend(c, 4, up=True) & w(3) & w(2) & w(1) & (_sh(c, 2) > _sh(c, 3)) & (_sh(c, 1) > _sh(c, 2))
         & (c < o) & (o > _sh(c, 1)) & (c < _sh(o, 3)))
    return np.zeros(len(c), bool), p, 4


def _stars(o, h, l, c):
    o2, c2, o1, c1 = _sh(o, 2), _sh(c, 2), _sh(o, 1), _sh(c, 1)
    small = np.abs(c1 - o1) <= 0.5 * np.abs(c2 - o2)
    tall2 = _tall(h, l, 2)
    morning = (_trend(c, 3, up=False) & tall2 & (c2 < o2) & small & (np.maximum(o1, c1) < np.minimum(o2, c2))
               & (c > o) & (c > (o2 + c2) / 2))
    evening = (_trend(c, 3, up=True) & tall2 & (c2 > o2) & small & (np.minimum(o1, c1) > np.maximum(o2, c2))
               & (c < o) & (c < (o2 + c2) / 2))
    return morning, evening, 3


def _two_black_gap(o, h, l, c):
    p = (_trend(c, 2, up=False) & (_sh(h, 1) < _sh(l, 2)) & (_sh(c, 1) < _sh(o, 1)) & (c < o) & (h < _sh(h, 1)))
    return p, p, 2          # both breakouts are traded (Bulkowski: down = continuation, up = best 10-day result)


def _three_crows(o, h, l, c):
    blk = lambda k: _sh(c, k) < _sh(o, k)  # noqa: E731
    near = lambda k: (_sh(c, k) - _sh(l, k)) <= 0.25 * (_sh(h, k) - _sh(l, k))  # noqa: E731
    p = (_trend(c, 3, up=True) & blk(2) & blk(1) & blk(0) & _tall(h, l, 2) & _tall(h, l, 1) & _tall(h, l, 0)
         & (_sh(o, 1) <= _sh(o, 2)) & (_sh(o, 1) >= _sh(c, 2)) & (o <= _sh(o, 1)) & (o >= _sh(c, 1))
         & near(2) & near(1) & near(0) & (_sh(l, 1) < _sh(l, 2)) & (l < _sh(l, 1)))
    return p, np.zeros(len(c), bool), 3


def s_bear_strike(df):
    """Bearish three-line strike (3 black candles, lower lows, then a white one closing above the first
    open) -> long on the first close above the pattern high within 10 days. Stop pattern low, target the
    pattern's height, 10-day time exit."""
    return _candle_bo(df, _bear_strike)


def s_bull_strike(df):
    """Bullish three-line strike -> short on the first close below the pattern low within 10 days (mirror)."""
    return _candle_bo(df, _bull_strike)


def s_stars(df):
    """Morning star -> long on a close above the pattern high; evening star -> short below its low."""
    return _candle_bo(df, _stars)


def s_two_black_gap(df):
    """Two black gapping candles after a decline -> trade the breakout either way (pattern height target)."""
    return _candle_bo(df, _two_black_gap)


def s_three_crows(df):
    """Three black crows in an up-trend -> long on an upward breakout only (Bulkowski: only up breakouts pay)."""
    return _candle_bo(df, _three_crows)


# ---------------------------------------------------------------- swing points
def _swings(h, l, k):
    """Swing highs / lows with strength k (extreme of +/- k bars), known k bars later."""
    hs = pd.Series(h)
    ls = pd.Series(l)
    is_h = (hs == hs.rolling(2 * k + 1, center=True).max()).to_numpy()
    is_l = (ls == ls.rolling(2 * k + 1, center=True).min()).to_numpy()
    return is_h, is_l


# ---------------------------------------------------------------- QB07 head and shoulders
def s_head_shoulders(df):
    """Head-and-shoulders bottom (top mirror) on daily swing points (strength 5): head lowest, shoulders
    within 25% of the height of each other, time symmetry within 2:1, prior trend down. First close above
    the neckline within 30 days of the right shoulder -> long; stop at the right-shoulder low, target
    0.71 x height from the breakout (Bulkowski's measure rule); 60-day time exit."""
    _, day, _, _ = _days(df)
    o, h, l, c = _ohlc(day)
    n = len(c)
    K = 5
    is_h, is_l = _swings(h, l, K)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    lows = np.flatnonzero(is_l)
    highs = np.flatnonzero(is_h)

    def scan(piv_a, piv_b, ext, oth, sign):
        # sign=+1: bottom (lows are piv_a, the armpits are highs)
        for j in range(2, len(piv_a)):
            ls_, hd, rs = piv_a[j - 2], piv_a[j - 1], piv_a[j]
            a1 = piv_b[(piv_b > ls_) & (piv_b < hd)]
            a2 = piv_b[(piv_b > hd) & (piv_b < rs)]
            if not len(a1) or not len(a2):
                continue
            a1 = a1[np.argmax(sign * oth[a1])]
            a2 = a2[np.argmax(sign * oth[a2])]
            e_ls, e_hd, e_rs = ext[ls_], ext[hd], ext[rs]
            if not (sign * (e_ls - e_hd) > 0 and sign * (e_rs - e_hd) > 0):
                continue
            slope = (oth[a2] - oth[a1]) / (a2 - a1)
            neck_h = oth[a1] + slope * (hd - a1)
            height = sign * (neck_h - e_hd)
            if height <= 0 or abs(e_ls - e_rs) > 0.25 * height:
                continue
            r = (hd - ls_) / max(1, rs - hd)
            if not 0.5 <= r <= 2.0:
                continue
            if ls_ < 60 or not (sign * (c[ls_ - 60] - c[ls_]) > 0):
                continue
            for t in range(rs + K, min(n, rs + 30)):
                neck = oth[a1] + slope * (t - a1)
                if sign > 0 and slope < 0:
                    neck = oth[a2]
                if sign < 0 and slope > 0:
                    neck = oth[a2]
                if sign * (c[t] - neck) > 0:
                    if sign > 0:
                        L[t], sl_l[t], tg_l[t] = True, e_rs, c[t] + 0.71 * height
                    else:
                        S[t], sl_s[t], tg_s[t] = True, e_rs, c[t] - 0.71 * height
                    break

    scan(lows, highs, l, h, +1)
    scan(highs, lows, h, l, -1)
    ix = day.index
    P = lambda x: pd.Series(x, index=ix)  # noqa: E731
    return _daily_trade(df, P(L), P(S), P(sl_l), P(sl_s), P(tg_l), P(tg_s), hold=60)


# ---------------------------------------------------------------- QB08 three rising valleys
def s_three_valleys(df):
    """Three rising valleys (swing lows, strength 5, each above the last; spans and depths within 2:1):
    the first close above the highest peak -> long; stop under the third valley - 0.1 daily ATR, target
    0.57 x (highest peak - lowest valley) above the peak; 60-day time exit. Three falling peaks mirror."""
    _, day, _, _ = _days(df)
    o, h, l, c = _ohlc(day)
    n = len(c)
    K = 5
    da = _datr(day).to_numpy()
    is_h, is_l = _swings(h, l, K)
    lows, highs = np.flatnonzero(is_l), np.flatnonzero(is_h)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))

    def scan(va, pk, vx, px, sign):
        for j in range(2, len(va)):
            v1, v2, v3 = va[j - 2], va[j - 1], va[j]
            if not (sign * (vx[v2] - vx[v1]) > 0 and sign * (vx[v3] - vx[v2]) > 0):
                continue
            p1 = pk[(pk > v1) & (pk < v2)]
            p2 = pk[(pk > v2) & (pk < v3)]
            if not len(p1) or not len(p2):
                continue
            p1, p2 = p1[np.argmax(sign * px[p1])], p2[np.argmax(sign * px[p2])]
            s1, s2 = v2 - v1, v3 - v2
            d1, d2 = sign * (px[p1] - vx[v1]), sign * (px[p2] - vx[v2])
            if min(s1, s2) <= 0 or max(s1, s2) / min(s1, s2) > 2 or min(d1, d2) <= 0 or max(d1, d2) / min(d1, d2) > 2:
                continue
            nxt = va[j + 1] if j + 1 < len(va) else n
            top = max(sign * px[p1], sign * px[p2]) * sign
            for t in range(v3 + K, min(n, nxt + K, v3 + 60)):
                top = (max(top, h[t - 1]) if sign > 0 else min(top, l[t - 1])) if t - 1 > v3 else top
                if sign * (c[t] - top) > 0:
                    height = sign * (top - vx[v1])
                    if sign > 0:
                        L[t], sl_l[t], tg_l[t] = True, vx[v3] - 0.1 * da[t], top + 0.57 * height
                    else:
                        S[t], sl_s[t], tg_s[t] = True, vx[v3] + 0.1 * da[t], top - 0.57 * height
                    break

    scan(lows, highs, l, h, +1)
    scan(highs, lows, h, l, -1)
    ix = day.index
    P = lambda x: pd.Series(x, index=ix)  # noqa: E731
    return _daily_trade(df, P(L), P(S), P(sl_l), P(sl_s), P(tg_l), P(tg_s), hold=60)


# ---------------------------------------------------------------- QB09 rectangle
def s_rectangle(df):
    """Rectangle over the last 40 days: top = median of the swing highs (strength 3), bottom = median of
    the swing lows; at least 3 touches of one line and 2 of the other (within 15% of the height), every
    close inside the band. First close above the top -> long, target 0.78 x height; below the bottom ->
    short, target 0.54 x height. Stop at the opposite line, exit after 40 days."""
    _, day, _, _ = _days(df)
    o, h, l, c = _ohlc(day)
    n = len(c)
    N, K = 40, 3
    is_h, is_l = _swings(h, l, K)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    t = N + K
    while t < n:
        a, z = t - N, t - K          # swing points confirmed by day t-1
        hi_i = np.flatnonzero(is_h[a:z]) + a
        lo_i = np.flatnonzero(is_l[a:z]) + a
        if len(hi_i) >= 2 and len(lo_i) >= 2:
            T, B = np.median(h[hi_i]), np.median(l[lo_i])
            hgt = T - B
            if hgt > 0:
                nt = int(np.sum(np.abs(h[hi_i] - T) <= 0.15 * hgt))
                nb = int(np.sum(np.abs(l[lo_i] - B) <= 0.15 * hgt))
                inside = np.all((c[a:t] >= B - 0.15 * hgt) & (c[a:t] <= T + 0.15 * hgt))
                if inside and ((nt >= 3 and nb >= 2) or (nt >= 2 and nb >= 3)):
                    if c[t] > T + 0.15 * hgt:
                        L[t], sl_l[t], tg_l[t] = True, B, T + 0.78 * hgt
                        t += N // 2
                        continue
                    if c[t] < B - 0.15 * hgt:
                        S[t], sl_s[t], tg_s[t] = True, T, B - 0.54 * hgt
                        t += N // 2
                        continue
        t += 1
    ix = day.index
    P = lambda x: pd.Series(x, index=ix)  # noqa: E731
    return _daily_trade(df, P(L), P(S), P(sl_l), P(sl_s), P(tg_l), P(tg_s), hold=40)


# ---------------------------------------------------------------- QB10 Kaufman AMA (bar timeframe)
def kama(c, n=10, fast=2, slow=30):
    c = np.asarray(c, float)
    ch = np.abs(c - _sh(c, n))
    vol = pd.Series(np.abs(np.diff(c, prepend=np.nan))).rolling(n).sum().to_numpy()
    er = np.where(vol > 0, ch / vol, 0)
    fs, ss = 2 / (fast + 1), 2 / (slow + 1)
    sc = (er * (fs - ss) + ss) ** 2
    out = np.full(len(c), np.nan)
    start = n
    if len(c) <= start:
        return out
    out[start] = c[start]
    for i in range(start + 1, len(c)):
        k = sc[i] if np.isfinite(sc[i]) else ss ** 2
        out[i] = out[i - 1] + k * (c[i] - out[i - 1])
    return out


def s_kama(df):
    """AMA(10,2,30); filter = 0.1 x std of the AMA's 20 last changes. Long when the AMA has risen more
    than the filter above its last trough, short when it has fallen more than the filter below its last
    peak. Stop-and-reverse; 6 x ATR protective stop."""
    c = df["c"].to_numpy()
    a = kama(c)
    da = np.diff(a, prepend=np.nan)
    filt = 0.1 * pd.Series(da).rolling(20).std().to_numpy()
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    lo, hi, pos = np.nan, np.nan, 0
    for i in range(2, n):
        if not np.isfinite(a[i - 2]) or not np.isfinite(filt[i]):
            continue
        if a[i - 1] < a[i - 2] and a[i] > a[i - 1]:
            lo = a[i - 1]
        if a[i - 1] > a[i - 2] and a[i] < a[i - 1]:
            hi = a[i - 1]
        if pos <= 0 and np.isfinite(lo) and a[i] > a[i - 1] and a[i] - lo > filt[i]:
            L[i], pos = True, 1
        elif pos >= 0 and np.isfinite(hi) and a[i] < a[i - 1] and hi - a[i] > filt[i]:
            S[i], pos = True, -1
    return L, S, {"stop": ("atr", 6.0), "target": None, "opposite": True}


# ---------------------------------------------------------------- QB11 Crabel pivot-break reversal
def s_pivot_reversal(df):
    """A pivot low (5 higher lows each side) is broken; within 5 bars a bar closes above the 2 prior
    closes, above the pivot, above its open and its mid-range, with a range wider than the prior bar's ->
    long at the next open. Stop just under the setup bar's low, target 8R. Mirror for pivot highs."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    n = len(c)
    K = 5
    is_h, is_l = _swings(h, l, K)
    a = atr(h, l, c, 14)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stl, sts = np.full(n, np.nan), np.full(n, np.nan)
    rng = h - l
    mid = (h + l) / 2

    def scan(piv, sign):
        for p in np.flatnonzero(piv):
            lvl = l[p] if sign > 0 else h[p]
            b = None
            for k in range(p + K + 1, min(n, p + 200)):
                if (sign > 0 and l[k] < lvl) or (sign < 0 and h[k] > lvl):
                    b = k
                    break
            if b is None:
                continue
            for s in range(b, min(n, b + 5)):
                if s < 2:
                    continue
                if sign > 0:
                    ok = (c[s] > max(c[s - 1], c[s - 2]) and c[s] > lvl and c[s] > o[s] and c[s] > mid[s]
                          and rng[s] > rng[s - 1])
                else:
                    ok = (c[s] < min(c[s - 1], c[s - 2]) and c[s] < lvl and c[s] < o[s] and c[s] < mid[s]
                          and rng[s] > rng[s - 1])
                if ok:
                    if sign > 0:
                        L[s], stl[s] = True, l[s] - 0.05 * a[s]
                    else:
                        S[s], sts[s] = True, h[s] + 0.05 * a[s]
                    break

    scan(is_l, +1)
    scan(is_h, -1)
    return L, S & ~L, {"stop": ("level", stl, sts), "target": ("r", 8.0)}


# ---------------------------------------------------------------- QB12 modified Hikkake
def s_hikkake(df):
    """Inside bar, then a bar closing below the inside bar's close (a false break down) -> long on the
    first close above the two bars' high within 3 bars; mirror for shorts. Stop beyond the pattern's
    extreme, out after 20 bars."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    n = len(c)
    inside = (_sh(h, 1) < _sh(h, 2)) & (_sh(l, 1) > _sh(l, 2))
    ls = inside & (c < _sh(c, 1))
    ss = inside & (c > _sh(c, 1))
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stl, sts = np.full(n, np.nan), np.full(n, np.nan)
    for t in np.flatnonzero(ls | ss):
        top, bot = max(h[t - 1], h[t]), min(l[t - 1], l[t])
        lo3, hi3 = l[t - 2:t + 1].min(), h[t - 2:t + 1].max()
        for k in range(t + 1, min(n, t + 4)):
            lo3, hi3 = min(lo3, l[k]), max(hi3, h[k])
            if ls[t] and c[k] > top:
                L[k], stl[k] = True, lo3
                break
            if ss[t] and c[k] < bot:
                S[k], sts[k] = True, hi3
                break
    return L, S & ~L, {"stop": ("level", stl, sts), "target": None, "max_bars": 20}


# ---------------------------------------------------------------- QB13 Larry Williams hidden smash day
def s_smash_day(df):
    """Daily: an up close that sits in the bottom 25% of the day's range, in an up-trend (close above
    the close 20 days earlier) -> the next day, the first close above that day's high buys. Stop at the
    smash day's low, out at the close of the 3rd day. Mirror for shorts."""
    date, day, first, last = _days(df)
    o, h, l, c = _ohlc(day)
    pos = (c - l) / np.where(h > l, h - l, np.nan)
    ls = (c > _sh(c, 1)) & (pos <= 0.25) & (c > _sh(c, 20))
    ss = (c < _sh(c, 1)) & (pos >= 0.75) & (c < _sh(c, 20))
    ix = day.index
    prev = lambda x: _on_bars(date, pd.Series(x, index=ix).astype(float).shift(1))  # noqa: E731
    bc = df["c"].to_numpy()
    arm_l, arm_s = prev(ls) > 0, prev(ss) > 0
    ph, pl = prev(h), prev(l)
    hit_l = arm_l & (bc > ph) & ~last
    hit_s = arm_s & (bc < pl) & ~last
    k = pd.Series(hit_l | hit_s).groupby(date).cumsum().to_numpy() == 1
    # exit at the close of the 2nd day after the entry day (3 days in the trade)
    entry_day = pd.Series((hit_l | hit_s) & k).groupby(date).transform("max").to_numpy()
    ed = pd.Series(entry_day.astype(float), index=date).groupby(level=0).first().reindex(ix).fillna(0)
    fe = (_on_bars(date, ed.shift(2)) > 0) & last
    return hit_l & k, hit_s & k & ~hit_l, {"stop": ("level", pl, ph), "target": None, "force_exit": fe}


# ---------------------------------------------------------------- QB14 Chande 65sma-3cc
def s_sma65_3cc(df):
    """Daily: three closes in a row above the 65-day SMA -> long (reverse a short); three below ->
    short. Stop-and-reverse; protective stop 3 x ATR(40) daily from the signal close."""
    _, day, _, _ = _days(df)
    o, h, l, c = _ohlc(day)
    ab = c > sma(c, 65)
    be = c < sma(c, 65)
    up3 = ab & _sh(ab, 1).astype(bool) & _sh(ab, 2).astype(bool)
    dn3 = be & _sh(be, 1).astype(bool) & _sh(be, 2).astype(bool)
    Ls = up3 & ~(_sh(up3, 1) > 0)
    Ss = dn3 & ~(_sh(dn3, 1) > 0)
    a40 = _datr(day, 40)
    ix = day.index
    cs = pd.Series(c, index=ix)
    return _daily_trade(df, pd.Series(Ls, index=ix), pd.Series(Ss, index=ix), cs - 3 * a40, cs + 3 * a40,
                        ex_l=pd.Series(Ss, index=ix), ex_s=pd.Series(Ls, index=ix), opposite=True)


# id: (name, fn, timeframes)
PATTERNS = {
    "QB01": ("Bullish reversal day + RSI(5)<35, 20-day hold (Quantified Strategies)", s_reversal_rsi5, ["1Hour"]),
    "QB02": ("Bearish engulfing bought, out above the prior high (Quantified Strategies)", s_engulf_buy, ["1Hour"]),
    "QB03": ("Bearish three-line strike, upward breakout (Bulkowski)", s_bear_strike, ["1Hour"]),
    "QB04": ("Bullish three-line strike, downward breakout (Bulkowski)", s_bull_strike, ["1Hour"]),
    "QB05": ("Morning / evening star breakout (Bulkowski)", s_stars, ["1Hour"]),
    "QB06": ("Two black gapping candles breakout (Bulkowski)", s_two_black_gap, ["1Hour"]),
    "QB07": ("Head-and-shoulders bottom / top, 0.71 measure rule (Bulkowski)", s_head_shoulders, ["1Hour"]),
    "QB08": ("Three rising valleys / falling peaks (Bulkowski)", s_three_valleys, ["1Hour"]),
    "QB09": ("Rectangle breakout, 0.78 / 0.54 targets (Bulkowski)", s_rectangle, ["1Hour"]),
    "QB10": ("Kaufman Adaptive Moving Average + std filter (Kaufman)", s_kama, ["1Hour", "4Hour"]),
    "QB11": ("Pivot-break top / bottom reversal, 8R (Crabel / Wyckoff)", s_pivot_reversal, ["1Hour", "4Hour"]),
    "QB12": ("Modified Hikkake, 20-bar hold (Chapman)", s_hikkake, ["1Hour", "4Hour"]),
    "QB13": ("Hidden Smash Day, 3-day hold (L. Williams)", s_smash_day, ["15Min", "1Hour"]),
    "QB14": ("65sma-3cc trend following (Chande)", s_sma65_3cc, ["1Hour"]),
    "QB15": ("Three black crows, upward breakout only (Bulkowski)", s_three_crows, ["1Hour"]),
}
