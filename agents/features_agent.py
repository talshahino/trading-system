"""Features agent: gives every event continuous 1-100 scores instead of hard categories.

All scores are point-in-time: they only use the 20 sessions BEFORE the news day, so the
backtest never peeks into the future. Scales are absolute (log), not relative to the
sample, so a score of 70 means the same thing in every run.

  size_score       1 = ~$50M market cap,         100 = ~$1T+   (needs market cap in universe file;
                                                                otherwise = liquidity_score)
  liquidity_score  1 = ~$1M traded per day,      100 = ~$10B+ per day
  volatility_score 1 = ~0.5% typical daily move, 100 = ~10%+ per day
  sentiment_score  0 = very negative, 50 = neutral, 100 = very positive
  news_strength    0-100 = |sentiment score| * 100
  news_confidence  0-100 = classifier confidence * 100
"""
import numpy as np
import pandas as pd

import config

NY = "America/New_York"
LOOKBACK = 20
MIN_DAYS = 5


def _log_score(x, lo, hi):
    """Map x on a log scale so lo -> 1 and hi -> 100, clipped."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1 + 99 * (np.log10(x) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return np.clip(s, 1, 100)


def daily_profile(bars):
    """Per NY trading day: dollar volume and close-to-close return."""
    if bars.index.tz is None:
        # Already daily bars (multi-day tracks), indexed by session date.
        daily = pd.DataFrame({"dv": bars["c"] * bars["v"], "close": bars["c"]}, index=bars.index)
    else:
        ny = bars.index.tz_convert(NY)
        d = pd.DataFrame({"dv": bars["c"] * bars["v"], "c": bars["c"]}, index=ny)
        daily = d.groupby(d.index.date).agg(dv=("dv", "sum"), close=("c", "last"))
        daily.index = pd.to_datetime(daily.index)
    daily["ret"] = daily["close"].pct_change()
    # Shift by one so the value on day D only uses days before D.
    roll = daily.rolling(LOOKBACK, min_periods=MIN_DAYS)
    daily["avg_dv"] = roll["dv"].mean().shift(1)
    daily["daily_vol"] = daily["ret"].rolling(LOOKBACK, min_periods=MIN_DAYS).std().shift(1)
    return daily[["avg_dv", "daily_vol"]]


def add_scores(events, bars_by_symbol, market_caps=None):
    market_caps = market_caps or {}
    profiles = {s: daily_profile(b) for s, b in bars_by_symbol.items() if b is not None and not b.empty}
    avg_dv, vol = [], []
    for ev in events.itertuples(index=False):
        p = profiles.get(ev.symbol)
        day = pd.Timestamp(ev.time.tz_convert(NY).date())
        # Last session on or before the news day (weekend / after-hours news uses the
        # most recent session; each row only holds data from before its own day).
        prior = p.loc[:day] if p is not None else None
        if prior is None or prior.empty:
            avg_dv.append(np.nan)
            vol.append(np.nan)
            continue
        avg_dv.append(prior["avg_dv"].iat[-1])
        vol.append(prior["daily_vol"].iat[-1])

    out = events.copy()
    out["avg_dollar_volume"] = avg_dv
    out["daily_vol"] = vol
    out["liquidity_score"] = _log_score(out["avg_dollar_volume"], 1e6, 1e10)
    out["volatility_score"] = _log_score(out["daily_vol"], 0.005, 0.10)
    caps = out["symbol"].map(market_caps)
    out["size_score"] = np.where(caps.notna(), _log_score(caps.fillna(1), 5e7, 1e12),
                                 out["liquidity_score"])
    out["sentiment_score"] = ((out["score"] + 1) * 50).round(1)
    out["news_strength"] = (out["score"].abs() * 100).round(1)
    out["news_confidence"] = (out["confidence"] * 100).round(1)
    for c in ["liquidity_score", "volatility_score", "size_score"]:
        out[c] = out[c].round(1)

    # Cost model: thin stocks cost more to get in and out of.
    # Linear from COST_BPS_ILLIQUID at liquidity 1 to COST_BPS_LIQUID at liquidity 100.
    liq = out["liquidity_score"].fillna(1)
    out["cost_bps"] = config.COST_BPS_ILLIQUID + (config.COST_BPS_LIQUID - config.COST_BPS_ILLIQUID) * (liq - 1) / 99
    return out


SCORE_COLUMNS = {
    "size_score": "גודל מניה",
    "liquidity_score": "נזילות",
    "volatility_score": "תנודתיות",
    "news_strength": "עוצמת החדשה",
    "news_confidence": "ביטחון המסווג",
}
