"""Level methods from the user's videos (task tal1411ind 12:47: MSNR and CRT, JP225 2-hour). Closed bars only.

msnr - Malaysian SnR (LuxAlgo concept https://www.luxalgo.com/library/concept/malaysian-snr/, rules
https://www.keenbase-trading.com/msnr-trading-strategy/, open-source check https://www.tradingview.com/script/v5zEBKFJ-Malaysian-SnR-Levels-UAlgo/):
levels from the close line only - A-level (resistance) = close of a bull candle followed by a bear candle, V-level
(support) = close of a bear candle followed by a bull candle, gap level = close[i-1] when two same-colour candles have
open[i] != close[i-1]. A level is fresh until a wick touches it; a body close through it flips its role and makes it
fresh again (RBS / SBR). Known one bar after it forms. Outputs the nearest fresh support under / resistance over the
close and the first-touch events (a wick reaches a fresh level and the close stays on its side).

crt - Candle Range Theory (LuxAlgo concept https://www.luxalgo.com/library/concept/candle-range-theory/): on the higher
timeframe, candle 1 = the range (high / low, EQ = middle); candle 2 sweeps one side and closes back inside the range
-> expect candle 3 to deliver to the other side. Set on the last base bar of candle 2 (its close is then known)."""
import numpy as np
import pandas as pd

from agents.indicators._core import cols


def msnr(df, keep=30):
    """dict: sup, res (nearest fresh levels, NaN if none), touch_sup, touch_res (first wick touch held), lvl (the level
    of that touch)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    bull, bear = c > o, c < o
    sups, ress = [], []          # fresh levels
    out = {"sup": np.full(n, np.nan), "res": np.full(n, np.nan), "lvl": np.full(n, np.nan)}
    out["touch_sup"], out["touch_res"] = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(1, n):
        # touches / breaks of existing fresh levels by bar i
        for s in list(sups):
            if c[i] < s:                       # body close through support -> fresh resistance (SBR)
                sups.remove(s)
                ress.append(s)
            elif l[i] <= s:
                sups.remove(s)                 # touched: no longer fresh
                if not out["touch_sup"][i] or s > out["lvl"][i]:
                    out["touch_sup"][i], out["lvl"][i] = True, s
        for r in list(ress):
            if c[i] > r:
                ress.remove(r)
                sups.append(r)
            elif h[i] >= r:
                ress.remove(r)
                if not out["touch_res"][i] and not out["touch_sup"][i]:
                    out["touch_res"][i], out["lvl"][i] = True, r
        if out["touch_sup"][i] and out["touch_res"][i]:
            out["touch_sup"][i] = out["touch_res"][i] = False
        # new levels formed by bars i-1, i (known now)
        if bull[i - 1] and bear[i]:
            ress.append(c[i - 1])
        if bear[i - 1] and bull[i]:
            sups.append(c[i - 1])
        if (bull[i - 1] and bull[i]) or (bear[i - 1] and bear[i]):
            if abs(o[i] - c[i - 1]) > 1e-12:
                (sups if bull[i] else ress).append(c[i - 1])
        sups, ress = sups[-keep:], ress[-keep:]
        below = [s for s in sups if s < c[i]]
        above = [r for r in ress if r > c[i]]
        out["sup"][i] = max(below) if below else np.nan
        out["res"][i] = min(above) if above else np.nan
    return out


def crt(df, rule="4h"):
    """dict on the last base bar of each higher-timeframe candle 2: bull, bear, c1_hi, c1_lo, eq, c2_hi, c2_lo."""
    o, h, l, c, v = cols(df)
    n = len(c)
    out = {x: np.full(n, np.nan) for x in ["c1_hi", "c1_lo", "eq", "c2_hi", "c2_lo"]}
    out["bull"], out["bear"] = np.zeros(n, bool), np.zeros(n, bool)
    if n < 3:
        return out
    key = df.index.floor(rule)
    k = pd.Series(np.arange(n), index=df.index).groupby(key)
    g = pd.DataFrame({"h": pd.Series(h, index=df.index).groupby(key).max(), "l": pd.Series(l, index=df.index).groupby(key).min(),
                      "c": pd.Series(c, index=df.index).groupby(key).last(), "last": k.max(), "cnt": k.count()})
    bm = float(pd.Series(df.index).diff().median().total_seconds() / 60.0)
    exp_ = pd.Timedelta(rule).total_seconds() / 60.0 / max(bm, 1e-9)
    full = g["cnt"] >= 0.5 * exp_                     # skip stub candles (session edges)
    H, L, C, last = g["h"].to_numpy(), g["l"].to_numpy(), g["c"].to_numpy(), g["last"].to_numpy()
    ok = full.to_numpy()
    for j in range(1, len(g)):
        if not (ok[j] and ok[j - 1]):
            continue
        i = int(last[j])
        if i >= n - 1 and j == len(g) - 1:
            continue                                   # the last candle may still be forming
        bl = L[j] < L[j - 1] and L[j - 1] < C[j] < H[j - 1]
        br = H[j] > H[j - 1] and L[j - 1] < C[j] < H[j - 1]
        if bl == br:
            continue
        out["bull"][i], out["bear"][i] = bl, br
        out["c1_hi"][i], out["c1_lo"][i], out["eq"][i] = H[j - 1], L[j - 1], (H[j - 1] + L[j - 1]) / 2
        out["c2_hi"][i], out["c2_lo"][i] = H[j], L[j]
    return out
