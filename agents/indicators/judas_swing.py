"""ICT Judas Swing (Flux Charts 'Judas Swing' / ICT concept; OUR BUILD from the definition - the user's task
tal1411set6 (13)): the session's first move sweeps the Asian range high (or low) or the previous day's extreme and
reverses. Asian range = 19:00-24:00 New York time of the evening before (the ICT Asian range); the Judas window = London 02:00-05:00 NY and New York 08:30-11:00 NY.
A bearish Judas = in the window the high trades above the Asian high (or PDH) and a later bar of the window closes
back under the Asian high; bullish mirror. The order block of the move = the last opposite candle before the
reversal close (its high / low). No repaint."""
import numpy as np
import pandas as pd

from agents.indicators._core import cols


def judas_swing(df, tz="America/New_York"):
    """dict: asia_hi, asia_lo, bull (reversal close after a sweep of the low), bear, sweep_lo, sweep_hi (the
    extreme of the sweep), ob_top / ob_bot (order block of the signal: last bearish candle before a bullish signal,
    last bullish before a bearish)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    t = df.index.tz_convert(tz) + pd.Timedelta(hours=5)  # shifted clock: 19:00 NY = 00:00 of the trading day
    hr = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
    day = pd.Series(t.date).to_numpy()
    ina = hr < 5.0
    ah = pd.Series(np.where(ina, h, np.nan)).groupby(day).cummax().to_numpy()
    al = pd.Series(np.where(ina, l, np.nan)).groupby(day).cummin().to_numpy()
    ah = pd.Series(ah).groupby(day).ffill().to_numpy()
    al = pd.Series(al).groupby(day).ffill().to_numpy()
    win = ((hr >= 7.0) & (hr < 10.0)) | ((hr >= 13.5) & (hr < 16.0))  # 02-05 and 08:30-11 NY
    out = {x: np.full(n, np.nan) for x in ["asia_hi", "asia_lo", "sweep_lo", "sweep_hi", "ob_top", "ob_bot"]}
    out["bull"] = np.zeros(n, bool)
    out["bear"] = np.zeros(n, bool)
    cur = None
    swl = swh = np.nan
    done = False
    for i in range(n):
        if day[i] != cur:
            cur, swl, swh, done = day[i], np.nan, np.nan, False
        if hr[i] >= 5.0 or not ina[i]:
            out["asia_hi"][i], out["asia_lo"][i] = ah[i], al[i]
        if not win[i] or done or not np.isfinite(ah[i]) or ina[i]:
            continue
        if l[i] < al[i]:
            swl = l[i] if not np.isfinite(swl) else min(swl, l[i])
        if h[i] > ah[i]:
            swh = h[i] if not np.isfinite(swh) else max(swh, h[i])
        if np.isfinite(swl) and c[i] > al[i] and c[i] > o[i]:
            out["bull"][i], out["sweep_lo"][i] = True, swl
            j = i - 1
            while j > 0 and c[j] >= o[j] and i - j < 20:
                j -= 1
            out["ob_top"][i], out["ob_bot"][i] = h[j], l[j]
            done = True
        elif np.isfinite(swh) and c[i] < ah[i] and c[i] < o[i]:
            out["bear"][i], out["sweep_hi"][i] = True, swh
            j = i - 1
            while j > 0 and c[j] <= o[j] and i - j < 20:
                j -= 1
            out["ob_top"][i], out["ob_bot"][i] = h[j], l[j]
            done = True
    return out
