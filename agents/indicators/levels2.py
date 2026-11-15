"""Time and level indicators the library lacked (scout 15.11.2026, night 3). No repaint: daily / session values are
known only after the session closed (shifted to the next session), running values use bars <= i.

- round_levels: psychological round numbers (BabyPips https://www.babypips.com/learn/forex/psychological-level,
  LuxAlgo https://www.luxalgo.com/library/concept/round-numbers/). Step = 10^(floor(log10(price)) - 2): EURUSD 1.08
  -> 0.01 (the '00' levels), USDJPY 150 -> 1.00, gold 2650 -> 10, US100 20000 -> 100; half steps = the '50' levels.
  OUR translation of the forex '00 / 50 pips' rule to every instrument by price magnitude.
- adr: Average Daily Range 14 (FXStreet https://www.fxstreet.com/education/how-to-use-the-average-daily-range-when-trading-forex-201610051529):
  mean high-low of the last 14 COMPLETED UTC days; today's running high / low / range and used = range / ADR.
- cbdr: ICT Central Bank Dealers Range (14:00-20:00 New York; https://www.tradingview.com/script/DtiFgMfL-CBDR-Deviations-M1D/):
  the range's high / low and standard-deviation projections (+/- k x the range height), valid from 20:00 NY until
  the next day's 14:00 NY.
- rvol: Relative Volume at time of day (https://www.tradingview.com/script/8ePzgnzt-Relative-Volume-at-Time/):
  the session's cumulative tick volume up to this bar's time slot / the mean of the same slot over the previous 20
  sessions (New York days); bar_rvol = this bar's volume / the same slot's mean.
- ict_macros: ICT algorithmic 'macro' windows, New York time (LuxAlgo ICT Macros
  https://www.tradingview.com/script/aLTywnJ0-ICT-Macros-LuxAlgo/): 02:33-03:00, 04:03-04:30, 08:50-09:10,
  09:50-10:10, 10:50-11:10, 11:50-12:10, 13:10-13:40, 15:15-15:45.
- po3_4h: explorer idea TT-s-po3-4h-ttrades (TTrades 'Trading the 4 Hour Power Of Three - OHLC',
  https://briefread.pages.dev/en/summary/trading-the-4-hour-power-of): New-York-aligned 4H candles (00/04/08/...
  NY); per bar the 4H open, the running 4H high / low and the previous 4H candle's high / low."""
import numpy as np
import pandas as pd

from agents.indicators._core import S, cols


def round_levels(df):
    """dict step, below / above (nearest major round level under / over the close), half_below / half_above."""
    c = cols(df)[3]
    with np.errstate(invalid="ignore", divide="ignore"):
        step = 10.0 ** (np.floor(np.log10(np.abs(c))) - 2)
    below = np.floor(c / step) * step
    hs = step / 2
    hb = np.floor(c / hs) * hs
    return {"step": step, "below": below, "above": below + step, "half_below": hb, "half_above": hb + hs}


def adr(df, n=14):
    """dict adr, day_hi, day_lo (running, this UTC day), used (running range / ADR), day_open."""
    o, h, l, c, v = cols(df)
    day = pd.Series(df.index.tz_convert("UTC").date)
    g = pd.DataFrame({"d": day, "h": h, "l": l, "o": o})
    rng = g.groupby("d").agg(h=("h", "max"), l=("l", "min"))
    r = (rng["h"] - rng["l"]).shift(1).rolling(n, min_periods=max(3, n // 2)).mean()
    a = g[["d"]].join(r.rename("adr"), on="d")["adr"].to_numpy(float)
    dh = g.groupby("d")["h"].cummax().to_numpy()
    dl = g.groupby("d")["l"].cummin().to_numpy()
    do = g.groupby("d")["o"].transform("first").to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        used = (dh - dl) / a
    return {"adr": a, "day_hi": dh, "day_lo": dl, "used": used, "day_open": do}


def cbdr(df, start=14, end=20, tz="America/New_York"):
    """dict hi, lo, sd (range height) of the last completed CBDR, NaN inside its own window and before the first."""
    o, h, l, c, v = cols(df)
    n = len(c)
    t = df.index.tz_convert(tz)
    hr = t.hour.to_numpy()
    inwin = (hr >= start) & (hr < end)
    day = pd.Series(t.date).to_numpy()
    hi, lo = np.full(n, np.nan), np.full(n, np.nan)
    cur_h = cur_l = np.nan
    run_h, run_l, run_d = -np.inf, np.inf, None
    for i in range(n):
        if inwin[i]:
            if run_d != day[i]:
                run_h, run_l, run_d = -np.inf, np.inf, day[i]
                cur_h = cur_l = np.nan          # the previous range expires when the new window opens
            run_h, run_l = max(run_h, h[i]), min(run_l, l[i])
        else:
            if run_d is not None and np.isfinite(run_h):
                cur_h, cur_l = run_h, run_l
                run_h, run_l, run_d = -np.inf, np.inf, None
            hi[i], lo[i] = cur_h, cur_l
    return {"hi": hi, "lo": lo, "sd": hi - lo}


def rvol(df, days=20, tz="America/New_York"):
    """dict rvol (cumulative, time-of-day), bar_rvol."""
    o, h, l, c, v = cols(df)
    t = df.index.tz_convert(tz)
    d = pd.DataFrame({"day": t.date, "slot": t.hour * 60 + t.minute, "v": v})
    d["cum"] = d.groupby("day")["v"].cumsum()
    prev_cum = d.groupby("slot")["cum"].transform(lambda x: x.shift(1).rolling(days, min_periods=3).mean())
    prev_v = d.groupby("slot")["v"].transform(lambda x: x.shift(1).rolling(days, min_periods=3).mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"rvol": (d["cum"] / prev_cum).to_numpy(float), "bar_rvol": (d["v"] / prev_v).to_numpy(float)}


MACROS = [(2, 33, 3, 0), (4, 3, 4, 30), (8, 50, 9, 10), (9, 50, 10, 10), (10, 50, 11, 10), (11, 50, 12, 10),
          (13, 10, 13, 40), (15, 15, 15, 45)]


def ict_macros(df, tz="America/New_York"):
    """dict in_macro (bar opens inside a window), macro_id (1..8, 0 outside)."""
    t = df.index.tz_convert(tz)
    m = (t.hour * 60 + t.minute).to_numpy()
    mid = np.zeros(len(m), int)
    for k, (h0, m0, h1, m1) in enumerate(MACROS, 1):
        mid[(m >= h0 * 60 + m0) & (m < h1 * 60 + m1)] = k
    return {"in_macro": mid > 0, "macro_id": mid}


def po3_4h(df, tz="America/New_York"):
    """dict open4, hi4 / lo4 (running, this 4H candle), prev_hi / prev_lo (the previous 4H candle), block (id),
    last_bar (the bar that closes the 4H candle, known from the timestamps)."""
    o, h, l, c, v = cols(df)
    t = df.index.tz_convert(tz)
    blk = pd.Series(pd.to_datetime(t.floor("h")).tz_localize(None)).dt.floor("4h")
    g = pd.DataFrame({"b": blk.to_numpy(), "o": o, "h": h, "l": l})
    open4 = g.groupby("b")["o"].transform("first").to_numpy()
    hi4 = g.groupby("b")["h"].cummax().to_numpy()
    lo4 = g.groupby("b")["l"].cummin().to_numpy()
    agg = g.groupby("b").agg(h=("h", "max"), l=("l", "min")).shift(1)
    j = g[["b"]].join(agg, on="b")
    bid = pd.factorize(g["b"])[0]
    bm = float(pd.Series(df.index).diff().median().total_seconds() / 60.0) if len(df) > 1 else 5.0
    end = pd.Series(pd.to_datetime((t + pd.Timedelta(minutes=bm)).floor("h")).tz_localize(None)).dt.floor("4h")
    last = end.to_numpy() != blk.to_numpy()
    return {"open4": open4, "hi4": hi4, "lo4": lo4, "prev_hi": j["h"].to_numpy(float), "prev_lo": j["l"].to_numpy(float),
            "block": bid, "last_bar": last}
