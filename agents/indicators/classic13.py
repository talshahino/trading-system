"""Indicators the library lacked (scout 20.11.2026 night run). Each is a port of the published formula, or our own
reconstruction where the code is closed / not readable or the page gives only a description (said so per indicator).
No repaint: every value at bar i uses bars <= i only.

- volatility_regime: Volatility Regime Switches [LuxAlgo] https://www.luxalgo.com/library/indicator/volatility-regime-switches/
  - the page gives the method and the defaults, the code was not readable to us, so the arithmetic is OUR reconstruction:
  score = ATR 14 / its SMA 100 ("Ratio to Average"); Stressed when the score reaches 1.25, Quiet at 0.80 or less,
  hysteresis 0.10 (Stressed is left only under 1.15, Quiet only over 0.90), a new state must hold 2 closed bars
  (confirmation) before it is committed. state: 0 quiet, 1 normal, 2 stressed; switch = +1 / -1 on the committing bar.
  mode="pct" = the page's 'Percentile Rank' normalization: score = ATR's percentile rank over the 100-bar baseline
  (0-100), Stressed >= 80, Quiet <= 20, hysteresis 10.
- hmm_regime: Hidden Markov Model Market Regimes [LuxAlgo] https://www.tradingview.com/script/USWQEica - open source on
  TradingView but the code was not readable to us; OUR reconstruction of the described method: a two-state (trending /
  ranging) forward filter on two observations - the 10-bar log-return drift in units of its own noise
  (|sum of 10 log returns| / (stdev of log returns over `n` x sqrt 10)) and the short / long volatility ratio
  (stdev 10 / stdev n). Gaussian emission heuristics (ours): trending drift ~ N(1.6, 0.8), ranging ~ N(0.4, 0.5);
  trending vol ratio ~ N(1.15, 0.35), ranging ~ N(0.9, 0.3). Sticky transition (stay 0.95). The newest posterior is
  blended with the previous one by the learning rate (0.3). p_trend + p_range = 1; |p_trend - 0.5| < 0.1 = neutral.
- vol_waterfall: Multi-Horizon Volatility Waterfall [LuxAlgo] https://www.tradingview.com/script/QuYs1iMf - open source,
  code not readable to us; OUR reconstruction of the description: ATR over ten horizons (step 10: 10, 20 .. 100 bars),
  each as its percentile rank over the last 200 bars, EMA 3 smoothed = the ten 'heat' ribbons (0..1); aggregate heat =
  their mean; regime +1 expansion (aggregate over 0.70), -1 compression (under 0.30), 0 neutral; fast / slow ratio =
  mean heat of the 3 fastest / mean heat of the 3 slowest horizons.
- pratik_trend: 'Trend indicator' (Pratik_4Clover) https://www.tradingview.com/script/lbuVV8u4-Trend-indicator/ - the
  description gives the formula: smoothed = EMA(close, n); oscillator = RMA(smoothed - SMA(close, n), n);
  signal = SMA(oscillator, n); buy / sell = the oscillator crossing 0. The default length is not on the page - ours 20.
"""
import numpy as np
import pandas as pd

from agents.indicators._core import S, atr, cols, cross_down, cross_up, ema, rma, sma


def volatility_regime(df, n=14, base=100, hi=None, lo=None, hyst=None, confirm=2, mode="ratio"):
    o, h, l, c, v = cols(df)
    a = atr(h, l, c, n)
    if mode == "pct":
        score = 100.0 * S(a).rolling(base, min_periods=base).rank(pct=True).to_numpy()
        hi, lo, hyst = (80.0 if hi is None else hi), (20.0 if lo is None else lo), (10.0 if hyst is None else hyst)
    else:
        hi, lo, hyst = (1.25 if hi is None else hi), (0.80 if lo is None else lo), (0.10 if hyst is None else hyst)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = a / S(a).rolling(base, min_periods=base).mean().to_numpy()
    state = np.ones(len(c), dtype=int)
    sw = np.zeros(len(c), dtype=int)
    cur, cand, cnt = 1, 1, 0
    for i in range(len(c)):
        s = score[i]
        if not np.isfinite(s):
            state[i] = cur
            continue
        if cur == 2:
            raw = 2 if s >= hi - hyst else (0 if s <= lo else 1)
        elif cur == 0:
            raw = 0 if s <= lo + hyst else (2 if s >= hi else 1)
        else:
            raw = 2 if s >= hi else (0 if s <= lo else 1)
        if raw == cur:
            cand, cnt = cur, 0
        else:
            cnt = cnt + 1 if raw == cand else 1
            cand = raw
            if cnt >= confirm:
                sw[i] = 1 if raw > cur else -1
                cur, cnt = raw, 0
        state[i] = cur
    return {"score": score, "state": state, "switch": sw,
            "quiet": state == 0, "stressed": state == 2}


def _npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / s


def hmm_regime(df, n=50, k=10, lr=0.3, stay=0.95):
    o, h, l, c, v = cols(df)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr_ = np.log(c / np.concatenate([[np.nan], c[:-1]]))
    r = S(lr_)
    sd_n = r.rolling(n, min_periods=n).std().to_numpy()
    sd_k = r.rolling(k, min_periods=k).std().to_numpy()
    drift = np.abs(r.rolling(k, min_periods=k).sum().to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        z = drift / (sd_n * np.sqrt(k))
        vr = sd_k / sd_n
    pt = np.full(len(c), np.nan)
    p = 0.5
    for i in range(len(c)):
        if not (np.isfinite(z[i]) and np.isfinite(vr[i])):
            continue
        prior_t = p * stay + (1 - p) * (1 - stay)
        et = _npdf(z[i], 1.6, 0.8) * _npdf(vr[i], 1.15, 0.35)
        er = _npdf(z[i], 0.4, 0.5) * _npdf(vr[i], 0.9, 0.3)
        den = prior_t * et + (1 - prior_t) * er
        post = prior_t * et / den if den > 0 else prior_t
        p = (1 - lr) * p + lr * post
        pt[i] = p
    return {"p_trend": pt, "p_range": 1 - pt, "neutral": np.abs(pt - 0.5) < 0.1}


def vol_waterfall(df, step=10, horizons=10, pct_n=200, smooth=3):
    o, h, l, c, v = cols(df)
    heats = []
    for j in range(1, horizons + 1):
        a = atr(h, l, c, step * j)
        pr = S(a).rolling(pct_n, min_periods=pct_n).rank(pct=True).to_numpy()
        heats.append(S(pr).ewm(span=smooth, adjust=False).mean().to_numpy() if smooth > 1 else pr)
    H = np.vstack(heats)
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = np.mean(H, axis=0)
        fs = np.mean(H[:3], axis=0) / np.mean(H[-3:], axis=0)
    agg = np.where(np.all(np.isfinite(H), axis=0), agg, np.nan)
    reg = np.where(agg > 0.70, 1, np.where(agg < 0.30, -1, 0))
    reg = np.where(np.isfinite(agg), reg, 0)
    return {"heat": H, "aggregate": agg, "regime": reg, "fast_slow": fs}


def pratik_trend(df, n=20):
    o, h, l, c, v = cols(df)
    sm = ema(c, n)
    osc = rma(np.nan_to_num(sm - sma(c, n), nan=0.0), n)
    osc = np.where(np.arange(len(c)) >= n - 1, osc, np.nan)
    sig = sma(osc, n)
    return {"osc": osc, "signal": sig, "buy": cross_up(osc, np.zeros(len(c))),
            "sell": cross_down(osc, np.zeros(len(c)))}
