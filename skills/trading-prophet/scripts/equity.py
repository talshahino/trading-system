#!/usr/bin/env python3
"""Build the account equity timeline that the FTMO simulator checks.

FTMO measures loss limits on EQUITY (closed balance + floating P&L, incl.
commissions and swaps), not on closed trades only. The MT5 tester report has no
equity-per-minute series, so we rebuild it:

  trades (from the deals table) + 1-minute bars per symbol
    -> for every minute a position is open, the WORST floating P&L inside that bar
       (long: bar low, short: bar high), using money-per-price-unit derived from
       each trade's own realized profit (no contract specs needed).

Without bars only closed balance is known; the result is then flagged
``quality="closed_only"`` and the verdict must be treated as provisional.
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from mt5_deals import Trade, parse_ts


@dataclass
class Point:
    ts: dt.datetime  # UTC
    balance: float
    equity_low: float
    open_count: int


@dataclass
class EquitySeries:
    points: list[Point]
    initial: float
    trading_dates_utc: list[dt.datetime]  # open times, for trading-day counting
    quality: str  # "full" | "partial" | "closed_only"
    notes: list[str]


class Bars:
    """1-minute (or any timeframe) OHLC bars per symbol: start_ts -> (high, low)."""

    def __init__(self) -> None:
        self._ts: dict[str, list[dt.datetime]] = {}
        self._hl: dict[str, list[tuple[float, float]]] = {}

    def add(self, symbol: str, rows: list[tuple[dt.datetime, float, float]]) -> None:
        rows = sorted(rows)
        self._ts[symbol] = [r[0] for r in rows]
        self._hl[symbol] = [(r[1], r[2]) for r in rows]

    def has(self, symbol: str) -> bool:
        return symbol in self._ts

    def window(self, symbol: str, start: dt.datetime, end: dt.datetime):
        """Bars whose start is in [floor_minute(start), end)."""
        ts = self._ts[symbol]
        lo = bisect.bisect_left(ts, start.replace(second=0, microsecond=0))
        hi = bisect.bisect_left(ts, end)
        for i in range(lo, hi):
            yield ts[i], self._hl[symbol][i]

    @classmethod
    def from_dir(cls, folder: str | Path, bars_tz: str = "Europe/Athens") -> "Bars":
        """Load every ``<SYMBOL>*.csv`` in a folder.

        Accepts the MT5 "Export bars" format (``<DATE> <TIME> <OPEN> <HIGH> <LOW> ...``,
        tab separated) or a plain CSV with ``time,open,high,low,close``.
        The symbol is the file name up to the first ``_`` (e.g. ``EURUSD_M1.csv``).
        """
        tz = ZoneInfo(bars_tz)
        out = cls()
        for f in sorted(Path(folder).glob("*.csv")):
            symbol = f.stem.split("_")[0]
            text = f.read_text(encoding="utf-8-sig", errors="replace")
            delim = "\t" if "\t" in text.splitlines()[0] else ","
            reader = csv.reader(text.splitlines(), delimiter=delim)
            header = [h.strip().strip("<>").lower() for h in next(reader)]
            idx = {h: i for i, h in enumerate(header)}
            rows = []
            for r in reader:
                if not r:
                    continue
                if "date" in idx and "time" in idx:
                    cell = f"{r[idx['date']]} {r[idx['time']]}"
                else:
                    cell = r[idx.get("time", 0)]
                ts = parse_ts(cell.replace("-", "."), tz)
                if ts is None:
                    continue
                rows.append((ts, float(r[idx["high"]]), float(r[idx["low"]])))
            if rows:
                out.add(symbol, rows)
        return out


def _symbol_key(symbol: str, bars: Bars) -> str | None:
    """Match 'EURUSD.cash' / 'EURUSDm' to a bars file named 'EURUSD'."""
    if bars.has(symbol):
        return symbol
    base = symbol.split(".")[0]
    if bars.has(base):
        return base
    for known in bars._ts:
        if symbol.upper().startswith(known.upper()):
            return known
    return None


def build_from_trades(trades: list[Trade], initial: float, bars: Bars | None) -> EquitySeries:
    notes: list[str] = []
    floats: dict[dt.datetime, float] = defaultdict(float)
    uncovered = 0
    for t in trades:
        key = _symbol_key(t.symbol, bars) if bars else None
        k = t.value_per_price_unit
        if key is None:
            uncovered += 1
            continue
        for ts, (high, low) in bars.window(key, t.open_time, t.close_time):
            worst = low if t.direction == 1 else high
            f = t.direction * (worst - t.open_price) * k if k else 0.0
            floats[ts] += min(f, 0.0) + t.commission  # commission charged up front: conservative
    if bars is None:
        quality = "closed_only"
        notes.append("no bars supplied: floating losses unknown, only closed balance checked")
    elif uncovered:
        quality = "partial"
        notes.append(f"{uncovered}/{len(trades)} trades had no bars for their symbol")
    else:
        quality = "full"

    closes = sorted(trades, key=lambda t: t.close_time)
    opens = sorted(t.open_time for t in trades)
    close_times = [t.close_time for t in closes]
    cum = [0.0]
    for t in closes:
        cum.append(cum[-1] + t.net)

    timeline = sorted(set(floats) | set(close_times))
    points: list[Point] = []
    for ts in timeline:
        n_closed = bisect.bisect_right(close_times, ts)
        n_opened = bisect.bisect_right(opens, ts)
        bal = initial + cum[n_closed]
        eq = bal + floats.get(ts, 0.0)
        points.append(Point(ts, bal, min(eq, bal) if ts in floats else bal, n_opened - n_closed))
    return EquitySeries(points, initial, opens, quality, notes)


def build_closed_only(points: list[tuple[dt.datetime, float]], initial: float | None) -> EquitySeries:
    pts = [Point(ts, b, b, 0) for ts, b in sorted(points)]
    init = initial if initial is not None else (pts[0].balance if pts else 0.0)
    return EquitySeries(pts, init, [p.ts for p in pts[1:]], "closed_only",
                        ["report lists closed balance only: floating losses unknown"])


def load_equity_csv(path: str | Path, tz: str = "UTC") -> EquitySeries:
    """Direct equity export: columns time,balance,equity[,open_count]."""
    zone = ZoneInfo(tz)
    pts: list[Point] = []
    with open(path, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            ts = parse_ts(r["time"].replace("-", "."), zone)
            if ts is None:
                continue
            bal, eq = float(r["balance"]), float(r["equity"])
            pts.append(Point(ts, bal, min(bal, eq), int(r.get("open_count") or 0)))
    pts.sort(key=lambda p: p.ts)
    init = pts[0].balance if pts else 0.0
    dates = [p.ts for i, p in enumerate(pts) if i and p.open_count > pts[i - 1].open_count]
    return EquitySeries(pts, init, dates, "full", [])
