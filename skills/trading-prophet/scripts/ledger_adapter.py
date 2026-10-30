#!/usr/bin/env python3
"""Adapter: run Trading Prophet on a trade ledger from this repo's backtests
(agents/ftmo_agent.account_path -> w.attrs["ledger"]).

The repo's own FTMO account replay checks the loss limits only when a trade CLOSES.
FTMO checks equity every tick, so a trade that dips 5% intraday and closes green
still fails the challenge. This adapter replays the same executed trades with
1-minute bars to catch exactly that.

Ledger columns used: time, exit_time, symbol, dir (+1/-1), entry_px, stop_px,
exit_px, taken ("בוצעה" = executed), risk_usd, pnl_usd.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from equity import Bars, build_from_trades  # noqa: E402
from mt5_deals import Trade  # noqa: E402
import prophet  # noqa: E402

EXECUTED = "בוצעה"


def ledger_to_trades(led) -> list[Trade]:
    out = []
    rows = led[led["taken"] == EXECUTED] if "taken" in led else led
    for r in rows.itertuples(index=False):
        sp, risk = getattr(r, "stop_px", None), getattr(r, "risk_usd", None)
        stop_dist = abs(float(r.entry_px) - float(sp)) if sp is not None and sp == sp else 0.0
        k = float(risk) / stop_dist if stop_dist and risk is not None and risk == risk else None
        pnl = float(r.pnl_usd)
        out.append(Trade(symbol=str(r.symbol), direction=int(r.dir),
                         open_time=r.time.to_pydatetime(), close_time=r.exit_time.to_pydatetime(),
                         open_price=float(r.entry_px), close_price=float(r.exit_px),
                         volume=1.0, profit=pnl, k=k))
    return out


def bars_from_frames(frames: dict) -> Bars:
    """frames: {symbol: DataFrame indexed by UTC time with columns h, l (or high, low)}."""
    b = Bars()
    for sym, df in frames.items():
        if df is None or df.empty:
            continue
        hi = df["h"] if "h" in df else df["high"]
        lo = df["l"] if "l" in df else df["low"]
        b.add(sym, [(ts.to_pydatetime(), float(h), float(l)) for ts, h, l in zip(df.index, hi, lo)])
    return b


def evaluate_ledger(led, frames: dict | None, preset: str = "ftmo_2step_100k",
                    initial: float = 100_000.0, policy_overrides: dict | None = None) -> dict:
    """Full Prophet result for one strategy's ledger. News is assumed filtered upstream."""
    rules, policy = prophet.load_config(preset)
    policy = {**policy, **(policy_overrides or {})}
    trades = ledger_to_trades(led)
    bars = bars_from_frames(frames) if frames else None
    series = build_from_trades(trades, initial, bars)
    return prophet.evaluate_series(series, trades, rules, policy, preset, "fixed",
                                   warnings=["news blackout applied upstream by the backtest"])
