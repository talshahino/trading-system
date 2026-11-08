"""Crypto strategies from published research and popular posts (research only, 08.11.2026).
Same contract as agents/rules_agent.py. Long only unless stated (as published).

CR01-CR03 Donchian trend (Zarattini, Pagani & Barbon 2025, "Catching Crypto Trends"; rules as summarised
          by CXO Advisory): daily close above the highest close of the previous N days -> buy; trailing
          stop = the midpoint of the highest and lowest close of the last N days, never lowered.
          The paper averages N = 5..360; here three of its lookbacks on their own: 20, 60, 150.
CR04      RSI(5) momentum (QuantifiedStrategies, Bitcoin RSI article): RSI(5) crosses above 50 -> buy,
          crosses back under 50 -> sell. No stop in the source: a 3 x ATR safety stop.
CR05      Bitcoin intraday seasonality (Quantpedia): buy at 22:00 UTC, sell at 00:00 UTC. 1 hour bars.
CR06      Bitcoin intraday time-series momentum (Shen, Urquhart & Wang 2022), rule 3: return 22:00 (previous
          day) -> 14:30 UTC up AND return 21:00 -> 21:30 UTC down -> buy 21:30-22:00; the reverse -> sell
          short. 15 minute bars. Safety stop 1.5 x ATR.
CR07      EMA20 / SMA50 cross (popular BTC post): EMA20 crosses above SMA50 -> buy, crosses under -> sell.
          Safety stop 3 x ATR.
CR08      Bollinger breakout (QuantifiedStrategies): close above the upper band (20, 2) -> buy, close under
          the 20 SMA -> sell. Safety stop 3 x ATR.
"""
import numpy as np
import pandas as pd

from agents.rules_agent import ema, sma, rsi


def _roll(x, n, f):
    s = pd.Series(x)
    return (s.rolling(n).max() if f == "max" else s.rolling(n).min()).to_numpy()


def _donchian(n):
    def s(df):
        c = df["c"].to_numpy(dtype=float)
        hi_prev = np.r_[np.nan, _roll(c, n, "max")[:-1]]          # highest close of the previous n days
        mid = (_roll(c, n, "max") + _roll(c, n, "min")) / 2
        L = c > hi_prev
        z = np.zeros(len(c), dtype=bool)
        return L, z, {"stop": ("trail_line",), "line": mid, "target": None}
    return s


def _cross_up(a, b):
    return (a > b) & (np.r_[np.nan, a[:-1]] <= np.r_[np.nan, b[:-1]])


def s_rsi5(df):
    c = df["c"].to_numpy(dtype=float)
    r = rsi(c, 5)
    L = _cross_up(r, np.full(len(c), 50.0))
    X = _cross_up(np.full(len(c), 50.0), r)
    z = np.zeros(len(c), dtype=bool)
    return L, z, {"stop": ("atr", 3.0), "target": None, "exit_long": X}


def _utc(df):
    return df.index.tz_convert("UTC")


def s_btc_22utc(df):
    t = _utc(df)
    hm = t.hour * 60 + t.minute
    L = np.asarray(hm == 21 * 60)                    # the 21:00 bar closes at 22:00 -> buy at 22:00
    X = np.asarray(hm == 23 * 60)                    # the 23:00 bar closes at 00:00 -> sell at 00:00
    z = np.zeros(len(df), dtype=bool)
    return L, z, {"stop": ("atr", 2.0), "target": None, "exit_long": X}


def s_intraday_tsmom(df):
    t = _utc(df)
    o, c = df["o"].to_numpy(dtype=float), df["c"].to_numpy(dtype=float)
    hm = np.asarray(t.hour * 60 + t.minute)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    p22 = p1430 = p21 = np.nan
    for i in range(n):
        m = hm[i]
        if m == 22 * 60:
            p22 = o[i]                               # previous day's 22:00
        elif m == 14 * 60 + 15:
            p1430 = c[i]                             # close of the 14:15 bar = 14:30
        elif m == 21 * 60:
            p21 = o[i]
        elif m == 21 * 60 + 15 and np.isfinite(p22) and np.isfinite(p1430) and np.isfinite(p21):
            r1, r2 = p1430 / p22 - 1, c[i] / p21 - 1   # c[i] = 21:30
            if r1 > 0 and r2 < 0:
                L[i] = True
            elif r1 <= 0 and r2 >= 0:
                S[i] = True
            p1430 = p21 = np.nan
    X = hm == 21 * 60 + 45                           # the 21:45 bar closes at 22:00 -> out at 22:00
    return L, S, {"stop": ("atr", 1.5), "target": None, "exit_long": X, "exit_short": X}


def s_ema20_sma50(df):
    c = df["c"].to_numpy(dtype=float)
    e, s = ema(c, 20), sma(c, 50)
    z = np.zeros(len(c), dtype=bool)
    return _cross_up(e, s), z, {"stop": ("atr", 3.0), "target": None, "exit_long": _cross_up(s, e)}


def s_bb_breakout(df):
    c = df["c"].to_numpy(dtype=float)
    m = sma(c, 20)
    sd = pd.Series(c).rolling(20).std().to_numpy()
    L = c > m + 2 * sd
    z = np.zeros(len(c), dtype=bool)
    return L, z, {"stop": ("atr", 3.0), "target": None, "exit_long": c < m}


def _macd(c):
    m = ema(c, 12) - ema(c, 26)
    return m, ema(m, 9)


def s_macd200(df):
    """CR09 Trading Rush (YouTube, "BEST Bitcoin Trading Strategy EVER with PROOF tested 100 TIMES"):
    buy when MACD crosses above its signal line under the zero line with the candle above the 200 MA; sell on
    the mirror. Stop beyond the pullback (lowest low / highest high of the last 10 bars), target 1.5 x risk."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    m, s = _macd(c)
    e = ema(c, 200)
    L = _cross_up(m, s) & (m < 0) & (c > e)
    S = _cross_up(s, m) & (m > 0) & (c < e)
    lo, hi = _roll(l, 10, "min"), _roll(h, 10, "max")
    return L, S & ~L, {"stop": ("level", lo, hi), "target": ("R", 1.5)}


def _swings(h, l, k=5):
    """Last two confirmed swing highs / lows at every bar (a swing is known k bars after it)."""
    n = len(h)
    sh1 = sh2 = sl1 = sl2 = np.nan
    H1, H2, L1, L2 = (np.full(n, np.nan) for _ in range(4))
    for t in range(2 * k, n):
        p = t - k
        if h[p] == h[p - k:t + 1].max():
            sh2, sh1 = sh1, h[p]
        if l[p] == l[p - k:t + 1].min():
            sl2, sl1 = sl1, l[p]
        H1[t], H2[t], L1[t], L2[t] = sh1, sh2, sl1, sl2
    return H1, H2, L1, L2


def s_structure(df):
    """CR10 saya crypto (YouTube, "I EXPOSE my BEST Crypto Trading Strategy", 352K views): trade with the higher
    time-frame structure; enter when the lower time-frame structure breaks that way. Here on 1 hour bars:
    the 50-day average sets the big trend; uptrend = higher swing high and higher swing low; buy on a close over
    the last swing high; stop under the last swing low; target 2 x risk, stop to entry at +1 R. Mirror for shorts."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    H1, H2, L1, L2 = _swings(h, l, 5)
    big = sma(c, 50 * 24 if df.attrs.get("tf") == "1Hour" else 50 * 6)
    prev = np.r_[np.nan, c[:-1]]
    up = (H1 > H2) & (L1 > L2) & (c > big)
    dn = (H1 < H2) & (L1 < L2) & (c < big)
    L = up & (c > H1) & (prev <= H1)
    S = dn & (c < L1) & (prev >= L1)
    return L, S & ~L, {"stop": ("level", L1, H1), "target": ("R", 2.0), "breakeven": (0.5, 0.0)}


TF = ["4Hour", "1Day"]
CRYPTO = {
    "CR01": ("Donchian trend 20 days, midpoint trailing stop (Zarattini et al. 2025)", _donchian(20), TF),
    "CR02": ("Donchian trend 60 days, midpoint trailing stop (Zarattini et al. 2025)", _donchian(60), TF),
    "CR03": ("Donchian trend 150 days, midpoint trailing stop (Zarattini et al. 2025)", _donchian(150), ["1Day"]),
    "CR04": ("RSI(5) momentum: cross above / below 50 (QuantifiedStrategies)", s_rsi5, TF),
    "CR05": ("Bitcoin seasonality: long 22:00-00:00 UTC (Quantpedia)", s_btc_22utc, ["1Hour"]),
    "CR06": ("Bitcoin intraday momentum, last half hour (Shen, Urquhart & Wang)", s_intraday_tsmom, ["15Min"]),
    "CR07": ("EMA20 / SMA50 cross", s_ema20_sma50, TF),
    "CR08": ("Bollinger breakout 20/2, exit under the mid band (QuantifiedStrategies)", s_bb_breakout, TF),
    "CR09": ("MACD cross under/over zero + 200 MA, target 1.5R (Trading Rush, YouTube)", s_macd200,
             ["15Min", "1Hour", "4Hour"]),
    "CR10": ("Market structure break with the trend, target 2R (saya crypto, YouTube)", s_structure,
             ["1Hour", "4Hour"]),
}


def _mirror(df):
    m = df.copy()
    m["o"], m["h"], m["l"], m["c"] = -df["o"], -df["l"], -df["h"], -df["c"]
    return m


def _long_short(fn):
    """The same rules both ways (08.11.2026, the user's request): the short side is the long rule on the
    mirrored prices. Levels and the trailing line are mirrored back (the Donchian midpoint is the same line)."""
    def g(df):
        L, _, p = fn(df)
        S, _, pm = fn(_mirror(df))
        S = np.asarray(S, bool) & ~np.asarray(L, bool)
        plan = dict(p)
        if p.get("exit_long") is not None:
            plan["exit_short"] = pm["exit_long"]
        if p.get("line") is not None:
            plan["line"] = p["line"]
        if p["stop"][0] == "level":
            plan["stop"] = ("level", p["stop"][1], -np.asarray(pm["stop"][1]))
        return L, S, plan
    return g


CRYPTO.update({
    "CS01": ("Donchian trend 20, long and short", _long_short(_donchian(20)), TF),
    "CS02": ("Donchian trend 60, long and short", _long_short(_donchian(60)), TF),
    "CS03": ("Donchian trend 150, long and short", _long_short(_donchian(150)), ["1Day"]),
    "CS04": ("RSI(5) momentum, long and short", _long_short(s_rsi5), TF),
    "CS07": ("EMA20 / SMA50 cross, long and short", _long_short(s_ema20_sma50), TF),
    "CS08": ("Bollinger breakout, long and short", _long_short(s_bb_breakout), TF),
})
