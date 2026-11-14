"""Trend Duration Forecast [ChartPrime] - open-source Pine: https://www.tradingview.com/script/L3SJFfAQ/
The page describes it: a Hull MA decides bull / bear; the bars of every completed bullish and bearish phase are
counted; the average of the last N phases of the same side is the 'Probable Length' forecast; the running phase is the
'Real Length'. Defaults are not on the page - ours: HMA length 50 ('Trend Detection Sensitivity'), sample 10 phases.
Also the user's 'expected trend length' exit (task tal1411set2 (4), Trading Data): median of past trend lengths -
returned too. No repaint: a phase's length enters the average only once it has ended."""
import numpy as np

from agents.indicators._core import cols, hma


def trend_duration(df, length=50, samples=10):
    """dict: dir (+1/-1), age (bars in the running phase), forecast (mean of the last `samples` completed phases of
    the same side), median (their median), ratio = age / forecast (>= 1 = the trend is older than usual)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    m = hma(c, length)
    d = np.zeros(n)
    for i in range(1, n):
        d[i] = 1 if m[i] > m[i - 1] else -1 if m[i] < m[i - 1] else d[i - 1]
    age = np.zeros(n)
    fc = np.full(n, np.nan)
    med = np.full(n, np.nan)
    hist = {1: [], -1: []}
    run = 0
    for i in range(n):
        if i > 0 and d[i] != d[i - 1] and d[i - 1] != 0:
            hist[int(d[i - 1])].append(run)
            run = 0
        run += 1
        age[i] = run
        hs = hist.get(int(d[i]), [])[-samples:] if d[i] != 0 else []
        if hs:
            fc[i] = float(np.mean(hs))
            med[i] = float(np.median(hs))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = age / fc
    return {"dir": d, "age": age, "forecast": fc, "median": med, "ratio": ratio}
