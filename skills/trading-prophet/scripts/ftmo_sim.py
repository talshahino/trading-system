#!/usr/bin/env python3
"""FTMO challenge simulator - the core of "Trading Prophet".

Given an equity timeline (see equity.py) it answers three questions:

1. continuous   - one account started on day 1: does it pass Challenge + Verification
                  without breaching, and how does it look at every weekly checkpoint
                  for up to a year (stops at the first breach)?
2. rolling      - a new challenge started every week across the whole history:
                  pass rate, breach rate, days to pass. This removes the luck of the
                  start date.
3. verdict      - the user's bar on top of FTMO rules (config ``prophet_policy``).

Rules modelled (configurable, see config/ftmo_presets.json):
  * Max Daily Loss on EQUITY: equity must stay above (balance at midnight
    Europe/Prague) - X% of the INITIAL balance.
  * Max Loss on EQUITY: static floor initial*(1-Y%) (or EOD-trailing for 1-Step).
  * Profit target on closed balance, optionally only when flat.
  * Minimum trading days (distinct Prague dates with a position opened).
"""

from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from equity import EquitySeries


@dataclass
class PhaseResult:
    phase: str
    status: str  # passed | breach_daily | breach_max | timeout | incomplete
    start: dt.datetime
    end: dt.datetime
    end_idx: int
    trading_days: int
    calendar_days: float
    worst_daily_use_pct: float  # worst % of the daily allowance consumed
    worst_total_use_pct: float  # worst % of the max-loss allowance consumed
    final_pnl_pct: float


@dataclass
class ChallengeResult:
    start: dt.datetime
    phases: list[PhaseResult] = field(default_factory=list)

    @property
    def outcome(self) -> str:
        last = self.phases[-1].status
        if last == "passed":
            return "passed"
        return last

    @property
    def days_to_pass(self) -> float | None:
        if self.outcome != "passed":
            return None
        return (self.phases[-1].end - self.start).total_seconds() / 86400


class Simulator:
    def __init__(self, series: EquitySeries, rules: dict, scale_mode: str = "relative"):
        self.s = series
        self.p = series.points
        self.r = rules
        self.A = float(rules["account_size"])
        self.tz = ZoneInfo(rules.get("day_boundary_tz", "Europe/Prague"))
        self.scale_mode = scale_mode
        self.opens = sorted(series.trading_dates_utc)
        self.open_dates = [o.astimezone(self.tz).date() for o in self.opens]
        self.local_dates = [pt.ts.astimezone(self.tz).date() for pt in self.p]

    # -- one phase -----------------------------------------------------------
    def phase(self, start_idx: int, name: str, target_pct: float) -> PhaseResult:
        r, A, pts = self.r, self.A, self.p
        b0 = pts[start_idx].balance
        if self.scale_mode == "fixed":
            scale = A / self.s.initial if self.s.initial else 1.0
        else:
            scale = A / b0 if b0 else 1.0
        mdl = r["max_daily_loss_pct"] / 100.0
        ml = r["max_loss_pct"] / 100.0
        trailing = r.get("max_loss_type") == "eod_trailing"
        target = A * (1 + target_pct / 100.0)
        min_days = int(r.get("min_trading_days", 0))
        max_cal = r.get("max_calendar_days")
        need_flat = r.get("target_requires_flat", True)

        start_ts = pts[start_idx].ts
        day = self.local_dates[start_idx]
        day_start_bal = A
        prev_bal = A
        peak_eod = A
        oi = bisect.bisect_right(self.opens, start_ts)
        days: set = set()
        worst_daily = worst_total = 0.0

        def result(status: str, i: int) -> PhaseResult:
            end = pts[i].ts
            return PhaseResult(name, status, start_ts, end, i, len(days),
                               (end - start_ts).total_seconds() / 86400,
                               round(worst_daily, 1), round(worst_total, 1),
                               round((prev_bal - A) / A * 100, 2))

        for i in range(start_idx + 1, len(pts)):
            pt = pts[i]
            d = self.local_dates[i]
            if d != day:  # midnight Prague passed: freeze the day's starting balance
                day_start_bal = prev_bal
                peak_eod = max(peak_eod, prev_bal)
                day = d
            while oi < len(self.opens) and self.opens[oi] <= pt.ts:
                days.add(self.open_dates[oi])
                oi += 1
            bal = A + (pt.balance - b0) * scale
            eq = A + (pt.equity_low - b0) * scale
            base = A if r.get("daily_loss_base", "initial") == "initial" else day_start_bal
            daily_allow = mdl * base
            daily_floor = day_start_bal - daily_allow
            total_floor = (peak_eod if trailing else A) - ml * A
            worst_daily = max(worst_daily, (day_start_bal - eq) / daily_allow * 100)
            worst_total = max(worst_total, ((peak_eod if trailing else A) - eq) / (ml * A) * 100)
            prev_bal = bal
            if eq < daily_floor:
                return result("breach_daily", i)
            if eq < total_floor:
                return result("breach_max", i)
            if max_cal and (pt.ts - start_ts).days > max_cal:
                return result("timeout", i)
            if bal >= target and len(days) >= min_days and (pt.open_count == 0 or not need_flat):
                return result("passed", i)
        return result("incomplete", len(pts) - 1)

    # -- full challenge (all phases) ----------------------------------------
    def challenge(self, start_idx: int) -> ChallengeResult:
        res = ChallengeResult(self.p[start_idx].ts)
        idx = start_idx
        for ph in self.r["phases"]:
            pr = self.phase(idx, ph["name"], ph["profit_target_pct"])
            res.phases.append(pr)
            if pr.status != "passed":
                break
            idx = pr.end_idx
        return res

    # -- continuous account with weekly checkpoints ----------------------------
    def weekly_track(self, horizon_weeks: int = 52) -> list[dict]:
        """One account from day 1, loss limits only (no target), checkpoint each week."""
        if not self.p:
            return []
        no_target = {**self.r, "min_trading_days": 0}
        sim = Simulator(self.s, no_target, self.scale_mode)
        pr = sim.phase(0, "track", 10_000.0)  # unreachable target -> runs to breach/end
        start = self.p[0].ts
        breach_ts = pr.end if pr.status.startswith("breach") else None
        A, b0 = self.A, self.p[0].balance
        scale = A / self.s.initial if self.scale_mode == "fixed" and self.s.initial else A / b0
        out = []
        times = [q.ts for q in self.p]
        for w in range(1, horizon_weeks + 1):
            ts = start + dt.timedelta(weeks=w)
            if ts > self.p[-1].ts:
                break
            j = bisect.bisect_right(times, ts) - 1
            if breach_ts and ts >= breach_ts:
                out.append({"week": w, "status": pr.status, "pnl_pct": None})
                break
            bal = A + (self.p[j].balance - b0) * scale
            out.append({"week": w, "status": "alive", "pnl_pct": round((bal - A) / A * 100, 2)})
        return out

    # -- rolling starts ---------------------------------------------------------
    def rolling(self, every_days: int = 7) -> list[ChallengeResult]:
        if not self.p:
            return []
        out = []
        times = [q.ts for q in self.p]
        t = self.p[0].ts
        while t < self.p[-1].ts:
            i = bisect.bisect_left(times, t)
            if i >= len(self.p) - 1:
                break
            out.append(self.challenge(i))
            t += dt.timedelta(days=every_days)
        return out


def summarize_rolling(results: list[ChallengeResult]) -> dict:
    passed = [r for r in results if r.outcome == "passed"]
    breached = [r for r in results if r.outcome.startswith("breach")]
    completed = len(passed) + len(breached)
    days = sorted(r.days_to_pass for r in passed)
    reasons: dict[str, int] = {}
    for r in breached:
        reasons[r.outcome] = reasons.get(r.outcome, 0) + 1
    return {
        "samples": len(results),
        "completed": completed,
        "passed": len(passed),
        "breached": len(breached),
        "incomplete": len(results) - completed,
        "pass_rate_pct": round(len(passed) / completed * 100, 1) if completed else None,
        "breach_rate_pct": round(len(breached) / completed * 100, 1) if completed else None,
        "median_days_to_pass": round(days[len(days) // 2], 1) if days else None,
        "breach_reasons": reasons,
    }


def verdict(series: EquitySeries, cont: ChallengeResult, track: list[dict], roll: dict,
            news_hits: int, policy: dict) -> dict:
    """Fail-closed decision. Returns {'verdict': PASS|FAIL|PROVISIONAL|INSUFFICIENT, 'reasons': [...]}"""
    reasons, warn = [], []
    if not series.points:
        return {"verdict": "INSUFFICIENT", "reasons": ["no equity data"], "warnings": []}
    if policy.get("require_continuous_no_breach", True) and cont.outcome.startswith("breach"):
        reasons.append(f"continuous account breached ({cont.outcome}) in phase {cont.phases[-1].phase}")
    if cont.outcome != "passed" and not cont.outcome.startswith("breach"):
        warn.append(f"continuous account did not finish the challenge ({cont.outcome})")
    if roll["completed"] < policy.get("rolling_min_completed_samples", 8):
        return {"verdict": "INSUFFICIENT",
                "reasons": reasons + [f"only {roll['completed']} completed rolling samples"],
                "warnings": warn}
    if roll["pass_rate_pct"] < policy.get("rolling_min_pass_rate_pct", 50):
        reasons.append(f"rolling pass rate {roll['pass_rate_pct']}% below bar")
    if roll["breach_rate_pct"] > policy.get("rolling_max_breach_rate_pct", 25):
        reasons.append(f"rolling breach rate {roll['breach_rate_pct']}% above bar")
    if track and track[-1]["status"] == "alive" and (track[-1]["pnl_pct"] or 0) <= 0:
        reasons.append("continuous account not profitable at last weekly checkpoint")
    if news_hits and policy.get("news_policy", "fail") == "fail":
        reasons.append(f"{news_hits} trade action(s) inside the news blackout window")
    elif news_hits:
        warn.append(f"{news_hits} trade action(s) inside the news blackout window")
    if reasons:
        return {"verdict": "FAIL", "reasons": reasons, "warnings": warn}
    if series.quality != "full" and policy.get("require_full_equity_data", True):
        return {"verdict": "PROVISIONAL", "reasons": [f"equity data quality is '{series.quality}'"],
                "warnings": warn + series.notes}
    return {"verdict": "PASS", "reasons": [], "warnings": warn}
