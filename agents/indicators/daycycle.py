"""Day-level structure indicators (scout 16.11.2026):

- taylor_cycle: George Douglass Taylor, 'The Taylor Trading Technique' (1950) - the 3-day buy day / sell day / sell
  short day rhythm (StoneX summary https://futures.stonex.com/blog/the-taylor-trading-technique ; LBR 3-day cycle notes
  https://c.mql5.com/forextsd/forum/118/lss_3_day_cycle_method.pdf). Per bar: the previous day's high / low / close,
  the number of consecutive down (up) daily closes up to yesterday, whether yesterday was an inside day, and today's
  running high / low (bars <= i). Ours: days = sessions of `tz` starting `start_hour` (default UTC midnight).
- noise_area: Zarattini, Aziz & Barbon, 'Beat the Market - An Effective Intraday Momentum Strategy for S&P500 ETF
  (SPY)' (2024, https://concretumgroup.com/beat-the-market-an-effective-intraday-momentum-strategy-for-sp500-etf-spy/ ;
  SSRN / SFI paper 24-97). sigma(t) = mean over the previous 14 sessions of |close at time-of-day t / that session's
  open - 1|; UB = max(open, previous close) x (1 + sigma), LB = min(open, previous close) x (1 - sigma); plus the
  session VWAP. Session = New York 09:30-16:00. Ours: on 24h feeds the same NY cash window is used, the 'open' is the
  first bar at / after 09:30.
"""
import numpy as np
import pandas as pd

from agents.indicators._core import cols, session_day


def taylor_cycle(df, tz="UTC", start_hour=0):
    """dict: pdh, pdl, pdc, down_days, up_days (consecutive lower / higher daily closes ending yesterday), inside
    (yesterday inside the day before: 1/0), day_high, day_low (today so far), day_open, new_day (first bar of a day)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    day = session_day(df, tz, start_hour)
    g = pd.DataFrame({"d": day, "h": h, "l": l, "c": c, "o": o})
    daily = g.groupby("d", sort=False).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), o=("o", "first"))
    dc = daily["c"].to_numpy()
    dn = np.zeros(len(daily))
    up = np.zeros(len(daily))
    for k in range(1, len(daily)):
        dn[k] = dn[k - 1] + 1 if dc[k] < dc[k - 1] else 0
        up[k] = up[k - 1] + 1 if dc[k] > dc[k - 1] else 0
    dh, dl = daily["h"].to_numpy(), daily["l"].to_numpy()
    ins = np.r_[0, ((dh[1:] < dh[:-1]) & (dl[1:] > dl[:-1])).astype(float)]
    prev = pd.DataFrame({"pdh": dh, "pdl": dl, "pdc": dc, "down_days": dn, "up_days": up, "inside": ins},
                        index=daily.index).shift(1)
    m = g[["d"]].join(prev, on="d")
    out = {k: m[k].to_numpy(dtype=float) for k in prev.columns}
    s = pd.Series(h).groupby(day).cummax()
    out["day_high"] = s.to_numpy(dtype=float)
    out["day_low"] = pd.Series(l).groupby(day).cummin().to_numpy(dtype=float)
    out["day_open"] = pd.Series(o).groupby(day).transform("first").to_numpy(dtype=float)
    nd = np.r_[True, day[1:] != day[:-1]]
    out["new_day"] = nd
    # a completed day's values reach the next day only (shift(1) above); the running values use bars <= i
    return out


def noise_area(df, lookback=14):
    """dict: ub, lb (noise-area bounds), vwap (session VWAP), in_rth (bar inside 09:30-16:00 NY), check (the bar closes
    on a :00 / :30 New York half hour, the paper's decision times), minutes (minutes from 09:30 to the bar's close)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    ny = df.index.tz_convert("America/New_York")
    step = float(pd.Series(df.index).diff().median().total_seconds() / 60.0) if n > 1 else 5.0
    mo = ny.hour.to_numpy() * 60 + ny.minute.to_numpy()      # bar open, minutes of the NY day
    mc = mo + step                                             # bar close
    rth = (mo >= 570) & (mc <= 960)
    date = pd.Series(ny.date).to_numpy()
    key = np.where(rth, date, None)
    out = {k: np.full(n, np.nan) for k in ("ub", "lb", "vwap", "minutes")}
    out["in_rth"] = rth
    out["check"] = rth & (np.mod(mc, 30) == 0) & (mc >= 600)
    if not rth.any():
        return out
    # session open and previous session close
    sess = pd.Series(np.arange(n))[rth].groupby(pd.Series(key)[rth])
    first = sess.min()
    last = sess.max()
    days = list(first.index)
    sopen = {d: o[int(first[d])] for d in days}
    sclose = {d: c[int(last[d])] for d in days}
    prev_close = {}
    for k, d in enumerate(days):
        prev_close[d] = sclose[days[k - 1]] if k > 0 else np.nan
    slot = np.where(rth, ((mc - 570) // step).astype(int), -1)
    move = np.full(n, np.nan)
    for i in np.where(rth)[0]:
        move[i] = abs(c[i] / sopen[key[i]] - 1)
    tab = pd.DataFrame({"d": key[rth], "s": slot[rth], "m": move[rth]}).pivot_table(index="d", columns="s", values="m")
    tab = tab.reindex(days)
    sig = tab.shift(1).rolling(lookback, min_periods=max(3, lookback // 2)).mean()
    tp = (h + l + c) / 3
    w = np.where(v > 0, v, 1.0)
    cum_pv = pd.Series(np.where(rth, tp * w, 0.0)).groupby(pd.Series(key).fillna("x")).cumsum().to_numpy()
    cum_v = pd.Series(np.where(rth, w, 0.0)).groupby(pd.Series(key).fillna("x")).cumsum().to_numpy()
    for i in np.where(rth)[0]:
        d = key[i]
        s_ = slot[i]
        sg = sig.at[d, s_] if s_ in sig.columns else np.nan
        op, pc = sopen[d], prev_close[d]
        if np.isfinite(sg):
            hi = max(op, pc) if np.isfinite(pc) else op
            lo = min(op, pc) if np.isfinite(pc) else op
            out["ub"][i] = hi * (1 + sg)
            out["lb"][i] = lo * (1 - sg)
        out["vwap"][i] = cum_pv[i] / cum_v[i] if cum_v[i] > 0 else np.nan
        out["minutes"][i] = mc[i] - 570
    return out
