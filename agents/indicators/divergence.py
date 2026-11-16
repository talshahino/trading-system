"""Regular and hidden divergence between price swings and an oscillator (scout 16.11.2026).

Port of the logic of 'Divergence for Many Indicators v4' by LonesomeTheBlue (open-source Pine,
https://www.tradingview.com/script/n8AGnIZd-Divergence-for-Many-Indicators-v4/): pivot highs / lows of price (k bars each
side), the oscillator compared at the two last pivots - regular bullish = price lower low + oscillator higher low,
hidden bullish = price higher low + oscillator lower low (mirror for bearish); 'many' = how many of RSI 14, MACD
histogram 12/26/9, CCI 20, Momentum 10, OBV, MFI 14, Stochastic 14 agree at that pivot. Hidden-divergence continuation
rules: https://therighttrader.com/learn/divergence/rsi-hidden-bullish-divergence .
No repaint: a pivot at bar j is known at bar j + k (agents/indicators/_core.pivots) and every flag is set on that
confirmation bar. Ours: pivots further apart than max_gap bars are not compared (LonesomeTheBlue: 'maximum bars to
check' 100); the oscillator value is read AT the pivot bar.
"""
import numpy as np

from agents.indicators._core import S, cci, cols, ema, pivots, rsi


def _oscillators(df):
    o, h, l, c, v = cols(df)
    m = ema(c, 12) - ema(c, 26)
    mh = m - ema(m, 9)
    mom = c - np.concatenate([np.full(10, np.nan), c[:-10]])
    obv = np.cumsum(np.sign(np.concatenate([[0.0], np.diff(c)])) * v)
    tp = (h + l + c) / 3
    mf = tp * v
    up = np.where(np.concatenate([[False], np.diff(tp) > 0]), mf, 0.0)
    dn = np.where(np.concatenate([[False], np.diff(tp) < 0]), mf, 0.0)
    su, sd = S(up).rolling(14).sum().to_numpy(), S(dn).rolling(14).sum().to_numpy()
    mfi = np.where(su + sd > 0, 100 * su / np.where(su + sd > 0, su + sd, 1), 50.0)
    ll, hh = S(l).rolling(14).min().to_numpy(), S(h).rolling(14).max().to_numpy()
    stoch = S(np.where(hh > ll, 100 * (c - ll) / np.where(hh > ll, hh - ll, 1), 50.0)).rolling(3).mean().to_numpy()
    return {"rsi": rsi(c, 14), "macd": mh, "cci": cci(h, l, c, 20), "mom": mom, "obv": obv, "mfi": mfi,
            "stoch": stoch}


def divergence(df, k=5, max_gap=100, src="all"):
    """dict (arrays len(df), set on the pivot's confirmation bar):
      bull_reg / bear_reg / bull_hid / bear_hid   number of oscillators showing that divergence (0..7, or 0/1 for one src)
      rsi_bull_reg ... per oscillator flags for 'rsi' and 'macd' (bool)
      piv_low / piv_high                           the price of the pivot just confirmed (NaN elsewhere)"""
    o, h, l, c, v = cols(df)
    n = len(c)
    osc = _oscillators(df)
    if src != "all":
        osc = {src: osc[src]}
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    keys = ["bull_reg", "bear_reg", "bull_hid", "bear_hid"]
    out = {x: np.zeros(n) for x in keys}
    for nm in ("rsi", "macd"):
        for x in keys:
            out[nm + "_" + x] = np.zeros(n, bool)
    out["piv_low"] = np.where(plc, plv, np.nan)
    out["piv_high"] = np.where(phc, phv, np.nan)
    prev_l = None
    for t in np.where(plc)[0]:
        j = int(pli[t])
        if prev_l is not None and j - prev_l <= max_gap:
            for nm, x in osc.items():
                a, b = x[j], x[prev_l]
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                if l[j] < l[prev_l] and a > b:
                    out["bull_reg"][t] += 1
                    if nm in ("rsi", "macd"):
                        out[nm + "_bull_reg"][t] = True
                elif l[j] > l[prev_l] and a < b:
                    out["bull_hid"][t] += 1
                    if nm in ("rsi", "macd"):
                        out[nm + "_bull_hid"][t] = True
        prev_l = j
    prev_h = None
    for t in np.where(phc)[0]:
        j = int(phi[t])
        if prev_h is not None and j - prev_h <= max_gap:
            for nm, x in osc.items():
                a, b = x[j], x[prev_h]
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                if h[j] > h[prev_h] and a < b:
                    out["bear_reg"][t] += 1
                    if nm in ("rsi", "macd"):
                        out[nm + "_bear_reg"][t] = True
                elif h[j] < h[prev_h] and a > b:
                    out["bear_hid"][t] += 1
                    if nm in ("rsi", "macd"):
                        out[nm + "_bear_hid"][t] = True
        prev_h = j
    return out
