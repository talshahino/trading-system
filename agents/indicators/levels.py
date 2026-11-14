"""Daily / weekly pivot points (Traditional) and higher-timeframe structural levels - standard formulas
(https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/ ; Ezekiel Chew and the
'daily pivot points as levels' video of the user, Meks Trades 'HTF structural levels: previous day / week high-low';
tasks tal1411set6 (7), tal1411ind). Traditional: P = (H+L+C)/3 of the previous period, R1 = 2P-L, S1 = 2P-H,
R2 = P+(H-L), S2 = P-(H-L), R3 = H+2(P-L), S3 = L-2(H-P). Periods in `tz` (default UTC; New York 17:00 rollover:
tz='America/New_York', start_hour=17). Each value uses only the COMPLETED previous period (no repaint)."""
import numpy as np
import pandas as pd

from agents.indicators._core import cols, session_day


def _prev_period(df, grp):
    o, h, l, c, v = cols(df)
    g = pd.DataFrame({"g": grp, "h": h, "l": l, "c": c})
    agg = g.groupby("g", sort=False).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    prev = agg.shift(1)
    m = g[["g"]].join(prev, on="g")
    return m["h"].to_numpy(float), m["l"].to_numpy(float), m["c"].to_numpy(float)


def pivot_points(df, period="day", tz="UTC", start_hour=0):
    """dict: P, R1, R2, R3, S1, S2, S3, above_p (close > P = bullish bias)."""
    if period == "week":
        t = df.index.tz_convert(tz) - pd.Timedelta(hours=start_hour)
        iso = t.isocalendar()
        grp = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    else:
        grp = session_day(df, tz, start_hour)
    H, L, C = _prev_period(df, grp)
    P = (H + L + C) / 3
    c = df["c"].to_numpy(float)
    return {"P": P, "R1": 2 * P - L, "S1": 2 * P - H, "R2": P + (H - L), "S2": P - (H - L),
            "R3": H + 2 * (P - L), "S3": L - 2 * (H - P), "above_p": c > P}


def htf_levels(df, tz="UTC", start_hour=0):
    """dict: pdh, pdl (previous day high/low), pwh, pwl (previous week), dh, dl (today's high / low so far)."""
    o, h, l, c, v = cols(df)
    day = session_day(df, tz, start_hour)
    pdh, pdl, _ = _prev_period(df, day)
    t = df.index.tz_convert(tz) - pd.Timedelta(hours=start_hour)
    iso = t.isocalendar()
    wk = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    pwh, pwl, _ = _prev_period(df, wk)
    dh = pd.Series(h).groupby(day).cummax().to_numpy()
    dl = pd.Series(l).groupby(day).cummin().to_numpy()
    return {"pdh": pdh, "pdl": pdl, "pwh": pwh, "pwl": pwl, "dh": dh, "dl": dl}
