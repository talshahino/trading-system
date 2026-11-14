"""Order-flow approximations from OHLCV + tick volume (OUR BUILD - there is no order book / footprint in our data,
so gamma levels, heat maps and footprints of the videos are NOT built). Sources of the ideas: Orderflow Sniper
(absorption, CVD divergence), ProTrippa 'Forget MACD & RSI' buyer/seller pressure oscillator, Meks Trades 'big
trades' (tick-volume spikes), Flux Charts 'Easy Strategy' (buy-volume % of the breakout bar) - the user's tasks
tal1411of / tal1411set3 / tal1411set6 (8) / tal1411ind. Reference for the buy/sell split: the common
'Volume Delta (approx.)' idea, buy = v x (close-low)/(high-low), sell = v x (high-close)/(high-low) (ours: on the true range)
(e.g. https://www.tradingview.com/script/1TpX6kXo-Buying-Selling-Volume/ ). No repaint."""
import numpy as np
import pandas as pd

from agents.indicators._core import S, atr, cols, ema, pivots, session_day, sma, stdev


def _split(h, l, c, v):
    """Buy / sell split by the close's location in the bar's TRUE range (previous close included, so a gap counts as
    pressure; on gap-free bars this is the usual (close-low)/(high-low))."""
    pc = np.concatenate([[c[0]], c[:-1]])
    hi, lo = np.maximum(h, pc), np.minimum(l, pc)
    rng = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(rng > 0, (c - lo) / rng, 0.5)
    return v * share, v * (1 - share), share


def buy_sell_pressure(df, n=14, sig=9):
    """ProTrippa-style pressure oscillator (ours): (EMA(buy, n) - EMA(sell, n)) / EMA(v, n) in -1..1, signal EMA(sig).
    dict: pressure, signal, buy_flip (crosses above 0), sell_flip, buy_pct (bar's buy share 0..1)."""
    o, h, l, c, v = cols(df)
    b, s, share = _split(h, l, c, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.nan_to_num((ema(b, n) - ema(s, n)) / ema(v, n))
    sg = ema(p, sig)
    pp = np.concatenate([[0], p[:-1]])
    return {"pressure": p, "signal": sg, "buy_flip": (p > 0) & (pp <= 0), "sell_flip": (p < 0) & (pp >= 0),
            "buy_pct": share}


def cvd(df, tz="UTC", start_hour=0, k=5):
    """Approximate cumulative volume delta, reset every session. Divergence (ours): a lower confirmed swing low with a
    higher CVD at it = bullish, a higher swing high with a lower CVD = bearish (signal on the confirmation bar).
    dict: cvd, delta, bull_div, bear_div."""
    o, h, l, c, v = cols(df)
    b, s, _ = _split(h, l, c, v)
    d = b - s
    grp = session_day(df, tz, start_hour)
    cv = pd.Series(d).groupby(grp).cumsum().to_numpy()
    phc, plc, _, _, phi, pli = pivots(h, l, k)
    n = len(c)
    bd = np.zeros(n, bool)
    sd = np.zeros(n, bool)
    lj = hj = None
    for t in range(n):
        if plc[t]:
            j = pli[t]
            if lj is not None and l[j] < l[lj] and cv[j] > cv[lj] and grp[j] == grp[lj]:
                bd[t] = True
            lj = j
        if phc[t]:
            j = phi[t]
            if hj is not None and h[j] > h[hj] and cv[j] < cv[hj] and grp[j] == grp[hj]:
                sd[t] = True
            hj = j
    return {"cvd": cv, "delta": d, "bull_div": bd, "bear_div": sd}


def absorption(df, vol_len=20, vol_mult=2.0, body_frac=0.35, trend_len=10):
    """Absorption (ours; a feed with constant volume passes the volume test): tick volume >= vol_mult x its average with a small body (<= body_frac of the ATR) at the end
    of a move (close at a trend_len-bar low = buying absorbs selling -> bullish; at a high -> bearish).
    dict: bull, bear, high, low (the absorption bar's high / low)."""
    o, h, l, c, v = cols(df)
    vm = np.concatenate([[np.nan], sma(v, vol_len)[:-1]])
    flat = stdev(v, vol_len) == 0
    a = atr(h, l, c, 14)
    spike = (v >= vol_mult * vm) | flat  # a feed without real volume (constant) passes
    small = np.abs(c - o) <= body_frac * a
    lo_n = S(l).rolling(trend_len).min().to_numpy()
    hi_n = S(h).rolling(trend_len).max().to_numpy()
    bull = spike & small & (l <= lo_n)
    bear = spike & small & (h >= hi_n)
    return {"bull": bull, "bear": bear & ~bull, "high": h, "low": l}


def big_trades(df, vol_len=50, z=3.0):
    """'Big trades' approximation: tick volume z-score >= z on the bar. dict: big, big_buy (close in the top half),
    big_sell."""
    o, h, l, c, v = cols(df)
    m = np.concatenate([[np.nan], sma(v, vol_len)[:-1]])
    sd = np.concatenate([[np.nan], stdev(v, vol_len)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        zz = np.where(sd > 0, (v - m) / sd, 0.0)
    big = zz >= z
    mid = (h + l) / 2
    return {"big": big, "big_buy": big & (c > mid), "big_sell": big & (c < mid), "z": np.nan_to_num(zz)}
