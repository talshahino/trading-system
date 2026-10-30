"""Generic rule-based strategy engine for strategies collected from TradingView (and later
other sources). Each strategy = a function that turns OHLCV bars into long/short signals at
the bar close, plus an exit plan. Entry is always at the NEXT bar's open (no peeking).

Exit plan keys:
  stop        ("atr", m) | ("lowest", n) | ("trail_line",)   initial stop
  target      ("atr", m) | ("R", x) | None
  trail_atr   m or None        ATR trailing stop (moves only in the trade's favour)
  breakeven   (fraction_of_target, offset_pct) or None
  max_bars    n or None        time exit at that bar's close
  opposite    True/False       exit at next open when the opposite signal fires
Same-bar stop and target: the stop counts first (pessimistic). Gaps fill at the open.
"""
import numpy as np
import pandas as pd

# ---------------- indicators ----------------
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()


def rma(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up, dn = rma(np.clip(d, 0, None), n), rma(np.clip(-d, 0, None), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + up / dn)


def atr(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return rma(tr, n)


def macd(c, fast=12, slow=26, sig=9):
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, sig)
    return m, s, m - s


def adx(h, l, c, n=14):
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(h, l, c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(pdm, n) / a
        ndi = 100 * rma(ndm, n) / a
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    return rma(np.nan_to_num(dx), n)


def cross_up(a, b):
    a, b = np.asarray(a, float), np.broadcast_to(np.asarray(b, float), np.shape(a))
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_down(a, b):
    return cross_up(-np.asarray(a, float), -np.broadcast_to(np.asarray(b, float), np.shape(a)))


def ut_bot(h, l, c, key=1.0, period=10):
    """UT Bot alerts: an ATR trailing line; buy when price crosses above it, sell below."""
    a = atr(h, l, c, period) * key
    line = np.zeros(len(c))
    for i in range(1, len(c)):
        prev = line[i - 1]
        if c[i] > prev and c[i - 1] > prev:
            line[i] = max(prev, c[i] - a[i])
        elif c[i] < prev and c[i - 1] < prev:
            line[i] = min(prev, c[i] + a[i])
        elif c[i] > prev:
            line[i] = c[i] - a[i]
        else:
            line[i] = c[i] + a[i]
    return cross_up(c, line), cross_down(c, line), line


def supertrend(h, l, c, period=10, mult=3.0):
    a = atr(h, l, c, period)
    mid = (h + l) / 2
    up, dn = mid - mult * a, mid + mult * a
    trend = np.ones(len(c))
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        trend[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else trend[i - 1]
    flip = np.diff(trend, prepend=trend[0])
    return flip > 0, flip < 0


# ---------------- strategies (as published) ----------------
def s_gold_pro(df):
    """TV 'Gold Pro Strategy' (zackseal): EMA50 vs EMA200 trend + price on that side of EMA200,
    MACD(12,26,9) crossover, RSI(14) 50-70 for longs / 30-50 for shorts. Stop 3 ATR(14),
    trailing. No fixed target."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    e50, e200 = ema(c, 50), ema(c, 200)
    m, s, _ = macd(c)
    r = rsi(c)
    long_ = (e50 > e200) & (c > e200) & cross_up(m, s) & (r > 50) & (r < 70)
    short = (e50 < e200) & (c < e200) & cross_down(m, s) & (r > 30) & (r < 50)
    return long_, short, {"stop": ("atr", 3.0), "trail_atr": 3.0, "target": None}


def s_dch10(df):
    """TV 'AsliGold XAUUSD DCH10 Trend Breakout': close crosses above the previous 10-bar high,
    ADX(14) > 20, long only. ATR(14) stop and target (multipliers not published: 2 / 4
    assumed), time stop (20 bars assumed)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    hh = pd.Series(h).shift(1).rolling(10).max().to_numpy()
    long_ = cross_up(c, hh) & (adx(h, l, c) > 20)
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 2.0), "target": ("atr", 4.0), "max_bars": 20}


def s_golden_master(df):
    """TV 'Golden Master Strategy v1': close above EMA200, MACD(12,26,9) crosses above signal
    below zero, histogram rising. Stop = lowest low of the last 4 bars, target 1:3,
    break-even at 50% of the target. Long only."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    m, s, hist = macd(c)
    long_ = (c > ema(c, 200)) & cross_up(m, s) & (m < 0) & (hist > np.roll(hist, 1))
    return long_, np.zeros(len(c), bool), {"stop": ("lowest", 4), "target": ("R", 3.0), "breakeven": (0.5, 0.0)}


def s_golden_edge(df):
    """TV 'Golden Edge Pro': MACD(16,26,9) histogram crosses above 0, EMA6(vol) > EMA8(vol),
    volume > 50% of the previous bar. Target ATR(14) x 2 (not published, assumed); stop not
    published (1.5 ATR assumed); break-even at 69% of target -> entry + 0.48%. Long only."""
    h, l, c, v = (df[x].to_numpy() for x in "hlcv")
    _, _, hist = macd(c, 16, 26, 9)
    long_ = cross_up(hist, 0) & (ema(v, 6) - ema(v, 8) > 0) & (v > 0.5 * np.roll(v, 1))
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 1.5), "target": ("atr", 2.0), "breakeven": (0.69, 0.0048)}


def s_ut_scalper(df):
    """TV 'control Scalper': UT Bot signals (key 1, ATR 10 - common defaults, not published),
    only when ADX(14) is between 16 and 20, RSI(14) > 55 for longs / < 45 for shorts. Exit on
    the opposite signal; the UT trailing line is the stop."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    buy, sell, line = ut_bot(h, l, c)
    a, r = adx(h, l, c), rsi(c)
    ok = (a > 16) & (a < 20)
    return buy & ok & (r > 55), sell & ok & (r < 45), {"stop": ("trail_line",), "line": line,
                                                       "target": None, "opposite": True}


def s_supertrend(df):
    """TV classic 'Supertrend Strategy' (ATR 10, factor 3): long on an up flip, short on a down
    flip, always in the market (exit = opposite flip); stop = 3 ATR as the risk unit."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    up, dn = supertrend(h, l, c)
    return up, dn, {"stop": ("atr", 3.0), "target": None, "opposite": True}


STRATEGIES = {
    "TV01": ("Gold Pro Strategy (zackseal)", s_gold_pro),
    "TV02": ("AsliGold DCH10 breakout", s_dch10),
    "TV03": ("Golden Master v1 (EMA200 + MACD)", s_golden_master),
    "TV04": ("Golden Edge Pro (MACD hist + volume)", s_golden_edge),
    "TV05": ("control Scalper (UT Bot + ADX/RSI)", s_ut_scalper),
    "TV06": ("Supertrend 10/3", s_supertrend),
}


# ---------------- engine ----------------
def run(df, fn, cost_side):
    """Trades for one strategy on one instrument/timeframe (one position at a time)."""
    if len(df) < 300:
        return pd.DataFrame()
    long_, short, plan = fn(df)
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    a = atr(h, l, c, 14)
    line = plan.get("line")
    n = len(c)
    sig = np.where(long_, 1, np.where(short, -1, 0))
    rows, i = [], 0
    while i < n - 2:
        d = sig[i]
        if d == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            i += 1
            continue
        e_t, e = i + 1, o[i + 1]
        st = plan["stop"]
        if st[0] == "atr":
            stop = e - d * st[1] * a[i]
        elif st[0] == "lowest":
            stop = (l[max(0, i - st[1] + 1):i + 1].min() if d > 0 else h[max(0, i - st[1] + 1):i + 1].max())
        else:
            stop = line[i]
        risk = d * (e - stop)
        if risk < 0.25 * a[i]:
            i += 1
            continue
        tg = plan.get("target")
        target = None if tg is None else (e + d * tg[1] * a[i] if tg[0] == "atr" else e + d * tg[1] * risk)
        be = plan.get("breakeven")
        trail = plan.get("trail_atr")
        best = e
        exit_px, exit_t, reverse_at = None, None, None
        for t in range(e_t, min(n, e_t + (plan.get("max_bars") or 10**9))):
            if t > e_t and ((d > 0 and o[t] <= stop) or (d < 0 and o[t] >= stop)):
                exit_px, exit_t = o[t], t
                break
            if (d > 0 and l[t] <= stop) or (d < 0 and h[t] >= stop):
                exit_px, exit_t = stop, t
                break
            if target is not None and t > e_t and ((d > 0 and h[t] >= target) or (d < 0 and l[t] <= target)):
                exit_px, exit_t = (max(target, o[t]) if d > 0 else min(target, o[t])), t
                break
            # updates at the bar close, used from the next bar on
            best = max(best, h[t]) if d > 0 else min(best, l[t])
            if trail:
                ts = best - d * trail * a[t]
                stop = max(stop, ts) if d > 0 else min(stop, ts)
            if st[0] == "trail_line":
                stop = max(stop, line[t]) if d > 0 else min(stop, line[t])
            if be and target is not None and d * (best - e) >= be[0] * d * (target - e):
                lvl = e * (1 + d * be[1])
                stop = max(stop, lvl) if d > 0 else min(stop, lvl)
            if plan.get("opposite") and sig[t] == -d and t + 1 < n:
                exit_px, exit_t = o[t + 1], t + 1
                reverse_at = t          # the opposite signal can open the next trade at the same open
                break
        if exit_px is None:
            exit_t = min(n - 1, e_t + (plan.get("max_bars") or 10**9) - 1)
            exit_px = c[exit_t]
        r_mult = d * (exit_px - e) / risk - 2 * cost_side * e / risk
        rows.append({"time": df.index[e_t], "exit_time": df.index[exit_t], "dir": d, "r": float(r_mult),
                     "held": int(exit_t - e_t)})
        i = reverse_at if reverse_at is not None else max(exit_t, i + 1)
    return pd.DataFrame(rows)
