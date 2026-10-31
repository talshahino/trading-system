"""Real FTMO trading costs per trade, applied to trades produced with zero cost.

- Spread: historical Dukascopy ask-bid at the entry minute (paid once per round trip).
  Instruments without Dukascopy history (Alpaca) use an estimate, flagged in the report.
- Commission: FTMO's own spec (ftmo.com/en/symbols):
    flat_USD  -> $ per lot, taken as round turn (verify on the demo account)
    percent   -> % of the traded value per side (x2 for the round trip)
- Swap: FTMO swapLong / swapShort for every server-midnight rollover the trade is held over
  (server = Europe/Athens time). Weekdays only, Wednesday night x3; crypto every night.
    points     -> swap x one point (10^-digits) of price per unit
    percentage -> annual % of the traded value / 360 per night
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents import ftmo_universe as U

SERVER_TZ = "Europe/Athens"
EST_SPREAD = {"stock": 0.0003, "crypto": 0.001}   # share of price, only where no bid/ask history


def _nights(entry, exit_, crypto):
    """Rollover counts per trade (vectorised over two DatetimeIndex-like arrays)."""
    a = pd.DatetimeIndex(entry).tz_convert(SERVER_TZ).tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
    b = pd.DatetimeIndex(exit_).tz_convert(SERVER_TZ).tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
    b = np.maximum(a, b)
    lo, hi = a + 1, b + 1                      # date changes into days (a, b]
    if crypto:
        return (b - a).astype("int64").astype("float64")
    # rollover into Tue..Sat = Mon..Fri night; into Thursday (Wednesday night) counts 3
    out = np.busday_count(lo, hi, weekmask="0111110") + 2 * np.busday_count(lo, hi, weekmask="0001000")
    return out.astype("float64")


def apply(t: pd.DataFrame, symbol: str, spread: pd.Series | None) -> pd.DataFrame:
    """t: trades from rules_agent.run(..., cost_side=0). Returns t with r net of FTMO costs and
    cost columns (spread_px, comm_px, swap_px, spread_est)."""
    if t.empty:
        return t
    sp = U.spec(symbol)
    cls = U.CLASS[symbol]
    e = t["entry_px"].to_numpy(dtype="float64")
    risk = np.abs(e - t["stop_px"].to_numpy(dtype="float64"))
    risk = np.where(risk > 0, risk, np.nan)
    d = t["dir"].to_numpy()

    # spread at the entry minute
    est = spread is None or len(spread) == 0
    if est:
        spread_px = e * EST_SPREAD.get(cls, 0.0003)
    else:
        s = spread.sort_index()
        idx = s.index.searchsorted(pd.DatetimeIndex(t["time"]), side="right") - 1
        med = float(s.median())
        vals = s.to_numpy(dtype="float64")
        spread_px = np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], med)
        spread_px = np.where(np.isfinite(spread_px), spread_px, med)

    # commission, round trip, in price units per unit traded
    if sp["commissionType"] == "flat_USD":
        base = sp["code"].split("/")[0]
        frac = float(sp["commission"]) / (float(sp["contractSize"]) * U.BASE_USD.get(base, 1.0))
        comm_px = e * frac
    else:
        comm_px = e * 2 * float(sp["commission"]) / 100.0

    # swap
    nights = _nights(t["time"], t["exit_time"], cls == "crypto")
    sw = np.where(d > 0, float(sp["swapLong"]), float(sp["swapShort"]))
    if sp["swapType"] == "percentage":
        swap_px = e * sw / 100.0 / 360.0 * nights
    else:
        swap_px = sw * 10.0 ** (-int(sp["digits"])) * nights

    out = t.copy()
    out["r"] = (t["r"].to_numpy(dtype="float64") - (spread_px + comm_px) / risk + swap_px / risk)
    out["r"] = out["r"].fillna(t["r"])
    out["spread_px"], out["comm_px"], out["swap_px"] = spread_px, comm_px, swap_px
    out["spread_est"] = est
    return out
