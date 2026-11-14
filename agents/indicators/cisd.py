"""Change in State of Delivery (CISD) - ICT concept; LuxAlgo open-source version:
https://www.tradingview.com/script/gHHRpuHC-Change-in-State-of-Delivery-CISD-LuxAlgo (also the 'CISD Finder' of
Day Trade Setup in the user's task tal1411ind, 10:33). Classic method as described there: a bullish CISD = a close
above the OPEN of the first candle of the latest run of bearish candles; bearish CISD = a close below the open of the
first candle of the latest run of bullish candles. The liquidity-sweep method: a sweep of an unbroken swing low
(wick under, close back above) arms a bullish CISD at the sweep leg's start. Our reconstruction for both:
the level is armed when the run ends and fires once, valid for `max_age` bars. Trend = side of the last CISD;
a CISD against the previous trend = reversal, with it = continuation. No repaint (closed bars)."""
import numpy as np

from agents.indicators._core import cols


def cisd(df, max_age=30):
    """dict: bull, bear (CISD bars), level_bull / level_bear (armed level), trend (+1/-1), reversal_bull /
    reversal_bear (a CISD that flips the trend)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    bull = np.zeros(n, bool)
    bear = np.zeros(n, bool)
    lb = np.full(n, np.nan)
    ls = np.full(n, np.nan)
    trend = np.zeros(n)
    run_dir, run_open = 0, np.nan
    arm_b, arm_b_at = np.nan, -1
    arm_s, arm_s_at = np.nan, -1
    t = 0
    for i in range(n):
        d = 1 if c[i] > o[i] else -1 if c[i] < o[i] else 0
        if np.isfinite(arm_b) and c[i] > arm_b and i - arm_b_at <= max_age and d >= 0:
            bull[i] = True
            arm_b = np.nan
            t = 1
        elif np.isfinite(arm_s) and c[i] < arm_s and i - arm_s_at <= max_age and d <= 0:
            bear[i] = True
            arm_s = np.nan
            t = -1
        if d != 0 and d != run_dir:
            if run_dir == -1:
                arm_b, arm_b_at = run_open, i
            elif run_dir == 1:
                arm_s, arm_s_at = run_open, i
            run_dir, run_open = d, o[i]
        lb[i], ls[i] = arm_b, arm_s
        trend[i] = t
    pt = np.concatenate([[0], trend[:-1]])
    return {"bull": bull, "bear": bear, "level_bull": lb, "level_bear": ls, "trend": trend,
            "reversal_bull": bull & (pt == -1), "reversal_bear": bear & (pt == 1)}
