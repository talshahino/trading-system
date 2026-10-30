#!/usr/bin/env python3
"""Parse the deals table of an MT5 Strategy Tester report into round-trip trades.

Pure standard library. Reuses the encoding/number/table helpers from the vendored
mt5-robot-tester (MIT, TraderMonty).

A *Trade* is one opened-then-closed lot on one symbol, with the timestamps in UTC.
Partial closes are split FIFO per symbol, so one MT5 position can become several
Trade objects with proportional profit/commission/swap.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mt5_tester" / "scripts"))
from mt5_common import html_table_rows, read_text, to_float  # noqa: E402

# Header aliases (English / Spanish / a few common builds) -> canonical key.
_ALIASES = {
    "time": ("time", "hora", "fecha/hora", "tiempo", "fecha"),
    "deal": ("deal", "operación", "operacion", "trato"),
    "symbol": ("symbol", "símbolo", "simbolo"),
    "type": ("type", "tipo"),
    "direction": ("direction", "dirección", "direccion"),
    "volume": ("volume", "volumen"),
    "price": ("price", "precio"),
    "commission": ("commission", "comisión", "comision"),
    "swap": ("swap",),
    "profit": ("profit", "beneficio", "ganancia"),
    "balance": ("balance",),
}

_TS_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?")
_EPS = 1e-9


@dataclass
class Trade:
    symbol: str
    direction: int  # +1 long, -1 short
    open_time: dt.datetime  # UTC
    close_time: dt.datetime  # UTC
    open_price: float
    close_price: float
    volume: float
    profit: float  # gross price P&L (MT5 "Profit" column)
    commission: float = 0.0
    swap: float = 0.0

    @property
    def net(self) -> float:
        return self.profit + self.commission + self.swap

    @property
    def value_per_price_unit(self) -> float | None:
        """Money per 1.0 price move for this lot, derived from the deal itself.

        profit = direction * (close - open) * k  ->  k = profit / (direction * move).
        This avoids needing contract size / tick value / account currency.
        """
        move = self.direction * (self.close_price - self.open_price)
        if abs(move) < _EPS or abs(self.profit) < _EPS:
            return None
        k = self.profit / move
        return k if k > 0 else None


@dataclass
class ParsedReport:
    trades: list[Trade]
    initial_deposit: float | None
    mode: str  # "full" (in/out deals paired) or "closed_only"
    warnings: list[str] = field(default_factory=list)
    closed_only_points: list[tuple[dt.datetime, float]] = field(default_factory=list)


def parse_ts(cell: str, tz: ZoneInfo) -> dt.datetime | None:
    m = _TS_RE.search(cell or "")
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) if g else 0 for g in m.groups())
    try:
        return dt.datetime(y, mo, d, h, mi, s, tzinfo=tz).astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _find_header(rows: list[list[str]]) -> tuple[int, dict[str, int]] | None:
    for i, cells in enumerate(rows):
        lower = [c.strip().lower() for c in cells]
        if "balance" not in lower:
            continue
        cols: dict[str, int] = {}
        for key, names in _ALIASES.items():
            for j, c in enumerate(lower):
                if c in names:
                    cols[key] = j
                    break
        if "time" in cols and "balance" in cols and "type" in cols:
            return i, cols
    return None


def _cell(row: list[str], cols: dict[str, int], key: str) -> str:
    j = cols.get(key)
    return row[j].strip() if j is not None and j < len(row) else ""


def parse_report(path: str | Path, report_tz: str = "Europe/Athens") -> ParsedReport:
    rows = html_table_rows(read_text(path))
    return parse_rows(rows, report_tz)


def parse_rows(rows: list[list[str]], report_tz: str = "Europe/Athens") -> ParsedReport:
    tz = ZoneInfo(report_tz)
    found = _find_header(rows)
    if not found:
        return ParsedReport([], None, "closed_only", ["deals table not found"])
    hdr, cols = found
    deals = []
    for row in rows[hdr + 1:]:
        ts = parse_ts(_cell(row, cols, "time"), tz)
        if ts is None:
            continue
        deals.append({
            "ts": ts,
            "symbol": _cell(row, cols, "symbol"),
            "type": _cell(row, cols, "type").lower(),
            "dir": _cell(row, cols, "direction").lower(),
            "volume": to_float(_cell(row, cols, "volume")) or 0.0,
            "price": to_float(_cell(row, cols, "price")),
            "commission": to_float(_cell(row, cols, "commission")) or 0.0,
            "swap": to_float(_cell(row, cols, "swap")) or 0.0,
            "profit": to_float(_cell(row, cols, "profit")) or 0.0,
            "balance": to_float(_cell(row, cols, "balance")),
        })

    initial = next((d["balance"] for d in deals if d["type"] == "balance"), None)
    warnings: list[str] = []

    if "direction" not in cols:
        # Simplified report without in/out: only realized balance steps are usable.
        pts = [(d["ts"], d["balance"]) for d in deals if d["balance"] is not None]
        warnings.append("no Direction column: only closed-trade balance is available")
        return ParsedReport([], initial, "closed_only", warnings, pts)

    trades: list[Trade] = []
    open_lots: dict[str, list[dict]] = {}
    for d in deals:
        if d["type"] not in ("buy", "sell"):
            continue
        side = 1 if d["type"] == "buy" else -1
        is_in = d["dir"] in ("in", "entrada")
        is_out = d["dir"] in ("out", "salida")
        is_inout = d["dir"] in ("in/out", "entrada/salida")
        if is_in:
            open_lots.setdefault(d["symbol"], []).append({
                "ts": d["ts"], "side": side, "vol": d["volume"], "price": d["price"],
                "commission": d["commission"], "orig_vol": d["volume"],
            })
            continue
        if not (is_out or is_inout):
            continue
        remaining = d["volume"]
        total_out = d["volume"] or 1.0
        queue = open_lots.get(d["symbol"], [])
        while remaining > _EPS and queue:
            lot = queue[0]
            if lot["side"] == side:  # an out deal closes the opposite side
                warnings.append(f"unexpected same-side out deal at {d['ts']} on {d['symbol']}")
                break
            take = min(lot["vol"], remaining)
            frac_out = take / total_out
            frac_in = take / (lot["orig_vol"] or 1.0)
            trades.append(Trade(
                symbol=d["symbol"], direction=lot["side"], open_time=lot["ts"],
                close_time=d["ts"], open_price=lot["price"], close_price=d["price"],
                volume=take, profit=d["profit"] * frac_out,
                commission=lot["commission"] * frac_in + d["commission"] * frac_out,
                swap=d["swap"] * frac_out,
            ))
            lot["vol"] -= take
            remaining -= take
            if lot["vol"] <= _EPS:
                queue.pop(0)
        if is_inout and remaining > _EPS:
            open_lots.setdefault(d["symbol"], []).append({
                "ts": d["ts"], "side": side, "vol": remaining, "price": d["price"],
                "commission": 0.0, "orig_vol": remaining,
            })
        elif remaining > _EPS:
            warnings.append(f"out deal with no matching open lot at {d['ts']} on {d['symbol']}")

    leftover = sum(len(q) for q in open_lots.values())
    if leftover:
        warnings.append(f"{leftover} position(s) still open at end of report (ignored)")
    trades.sort(key=lambda t: (t.close_time, t.open_time))
    return ParsedReport(trades, initial, "full", warnings)
