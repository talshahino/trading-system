"""FTMO filter: would a strategy pass an FTMO $100,000 evaluation and then keep the account?

Rules (FTMO 2-Step, from ftmo.com, checked 2026-10-30):
  Phase 1 (Challenge)    profit target 10%, then Phase 2 (Verification) target 5%.
  Maximum Daily Loss     equity may not fall more than 5% of the INITIAL balance ($5,000)
                         below the day's starting balance (day = CE(S)T midnight).
  Maximum Loss           equity may never fall below 90% of the initial balance ($90,000).
  Minimum trading days   4 per phase. No time limit.
  Funded account         same loss limits, no target, 90% of the profit is paid out.
FTMO 1-Step (also simulated): target 10%, daily loss 3%, maximum loss 10% trailing from
the highest end-of-day balance (never above the initial balance... per FTMO's example it
trails upward), best day <= 50% of the profit.

The simulated bot follows its own guard so it never knowingly breaks a rule: it does not
open a trade whose full stop, added to what is already open, could breach the daily or the
overall limit. Only gaps through a stop can still break a limit.

Each trade needs: time (entry), exit_time, r (net result in R; -1 = full stop).
"""
import numpy as np
import pandas as pd

INITIAL = 100_000.0
CET = "Europe/Prague"
MAX_OPEN = 3


def _days(ts):
    return pd.DatetimeIndex(ts).tz_convert(CET).normalize()


def run_account(trades, risk_pct, target_pct, daily_pct, max_pct, trailing=False, min_days=4,
                stop_at_target=True, start_balance=INITIAL):
    """Replays trades in time order. Returns dict(outcome, days, end_balance, trading_days)."""
    bal = start_balance
    floor = INITIAL * (1 - max_pct)
    day, day_start = None, bal
    open_pos = []                    # (exit_time, risk_usd, r)
    traded_days = set()
    best_eod = bal
    first_day = None
    for t in trades.itertuples(index=False):
        now = t.time
        # close everything that ended before this entry
        for p in sorted([p for p in open_pos if p[0] <= now], key=lambda p: p[0]):
            bal += p[1] * p[2]
            open_pos.remove(p)
            if bal <= floor or bal < day_start - INITIAL * daily_pct:
                return {"outcome": "fail", "days": _span(first_day, p[0]), "end": bal, "tdays": len(traded_days)}
        d = t.day
        if d != day:
            if day is not None:
                best_eod = max(best_eod, bal)
                if trailing:
                    floor = max(floor, best_eod - INITIAL * max_pct)
            day, day_start = d, bal
        if stop_at_target and bal >= start_balance * (1 + target_pct) and len(traded_days) >= min_days and not open_pos:
            return {"outcome": "pass", "days": _span(first_day, now), "end": bal, "tdays": len(traded_days)}
        risk = INITIAL * risk_pct if not trailing else bal * risk_pct
        open_risk = sum(p[1] for p in open_pos)
        # bot guard: never open a trade that could breach a limit at its stop
        if len(open_pos) >= MAX_OPEN:
            continue
        if bal - open_risk - risk <= floor or bal - open_risk - risk < day_start - INITIAL * daily_pct * 0.9:
            continue
        open_pos.append((t.exit_time, risk, t.r))
        traded_days.add(d)
        first_day = first_day or now
    for p in sorted(open_pos, key=lambda p: p[0]):
        bal += p[1] * p[2]
    if stop_at_target and bal >= start_balance * (1 + target_pct) and len(traded_days) >= min_days:
        return {"outcome": "pass", "days": _span(first_day, trades["time"].iloc[-1]), "end": bal, "tdays": len(traded_days)}
    return {"outcome": "fail" if bal <= floor else "open", "days": _span(first_day, trades["time"].iloc[-1] if len(trades) else None),
            "end": bal, "tdays": len(traded_days)}


def _span(a, b):
    if a is None or b is None:
        return np.nan
    return (pd.Timestamp(b) - pd.Timestamp(a)).days


def evaluate(trades, risk_levels=(0.0025, 0.005, 0.01, 0.02), starts_every_days=7, max_starts=40):
    """Rolling start dates over the period (no reshuffling: the real order of trades).
    For each risk level: chance to pass Phase 1, then Phase 2, days needed, and how the
    funded account does over the next 12 months."""
    tr = trades.sort_values("time").reset_index(drop=True)
    if len(tr) < 20:
        return None
    tr["day"] = _days(tr["time"])
    t0, t1 = tr["time"].iloc[0], tr["time"].iloc[-1]
    starts = pd.date_range(t0, t1 - pd.Timedelta(days=60), freq=f"{starts_every_days}D")
    if len(starts) > max_starts:
        starts = starts[np.linspace(0, len(starts) - 1, max_starts).astype(int)]
    out = []
    for risk in risk_levels:
        p1 = p2 = n = 0
        days = []
        funded_ret, funded_lost = [], 0
        one_step = 0
        for s in starts:
            seg = tr[tr["time"] >= s]
            if len(seg) < 10:
                continue
            n += 1
            c = run_account(seg, risk, 0.10, 0.03, 0.10, trailing=True, min_days=1)
            one_step += c["outcome"] == "pass"
            a = run_account(seg, risk, 0.10, 0.05, 0.10)
            if a["outcome"] != "pass":
                continue
            p1 += 1
            seg2 = seg[seg["time"] > s + pd.Timedelta(days=a["days"] or 0)]
            b = run_account(seg2, risk, 0.05, 0.05, 0.10) if len(seg2) >= 10 else {"outcome": "open"}
            if b["outcome"] == "pass":
                p2 += 1
                days.append((a["days"] or 0) + (b["days"] or 0))
                seg3 = seg2[(seg2["time"] > seg2["time"].iloc[0] + pd.Timedelta(days=b["days"] or 0))]
                seg3 = seg3[seg3["time"] <= seg3["time"].iloc[0] + pd.Timedelta(days=365)] if len(seg3) else seg3
                if len(seg3) >= 5:
                    f = run_account(seg3, risk, 10.0, 0.05, 0.10, stop_at_target=False)
                    if f["outcome"] == "fail":
                        funded_lost += 1
                    funded_ret.append((f["end"] - INITIAL) / INITIAL)
        if n == 0:
            continue
        out.append({"risk": risk, "starts": n, "p_phase1": p1 / n, "p_both": p2 / n,
                    "median_days": float(np.median(days)) if days else np.nan,
                    "p_1step": one_step / n,
                    "funded_lost": funded_lost / len(funded_ret) if funded_ret else np.nan,
                    "funded_year_ret": float(np.median(funded_ret)) if funded_ret else np.nan,
                    "funded_payout": float(np.median(funded_ret)) * INITIAL * 0.9 if funded_ret else np.nan})
    return pd.DataFrame(out)


def verdict(df):
    """Our filter, in FTMO terms: at the best risk level, most start dates pass both phases
    and most funded accounts survive the following year."""
    if df is None or df.empty:
        return False, None
    best = df.sort_values(["p_both", "funded_year_ret"], ascending=False).iloc[0]
    ok = best["p_both"] >= 0.5 and (np.isnan(best["funded_lost"]) or best["funded_lost"] <= 0.3)
    return bool(ok), best


def weekly_track(trades, risk_levels=(0.0025, 0.005, 0.01, 0.02), weeks=52, daily_pct=0.05, max_pct=0.10):
    """The user's main test: an account under FTMO loss limits ($100,000, daily loss 5% of the
    initial balance from the day's start, never below $90,000) starts on every week of the
    period and trades the strategy for up to a year. Its result is sampled after week 1,
    2, ... 52. Once a limit is broken, the account stops (blown) for good.
    Returns {risk: DataFrame(week, alive, profitable, median_pnl_pct)} and the start count."""
    tr = trades.sort_values("time").reset_index(drop=True)
    if len(tr) < 20:
        return None, 0
    tr["day"] = _days(tr["time"])
    t0, t1 = tr["time"].iloc[0], tr["time"].iloc[-1]
    starts = pd.date_range(t0, t1 - pd.Timedelta(weeks=weeks), freq="7D")
    if len(starts) == 0:
        return None, 0
    times = pd.DatetimeIndex(tr["time"])
    out = {}
    for risk in risk_levels:
        snap = np.full((len(starts), weeks), np.nan)
        for si, s in enumerate(starts):
            end = s + pd.Timedelta(weeks=weeks)
            a, b = times.searchsorted(s), times.searchsorted(end)
            seg = tr.iloc[a:b]
            bal, floor = INITIAL, INITIAL * (1 - max_pct)
            day, day_start, open_pos, blown = None, bal, [], False
            week_i = 0
            marks = [s + pd.Timedelta(weeks=w + 1) for w in range(weeks)]

            def close_until(when):
                nonlocal bal, blown
                for p in sorted([p for p in open_pos if p[0] <= when], key=lambda p: p[0]):
                    bal += p[1] * p[2]
                    open_pos.remove(p)
                    if bal <= floor or bal < day_start - INITIAL * daily_pct:
                        blown = True
                        return

            for t in seg.itertuples(index=False):
                while week_i < weeks and marks[week_i] <= t.time and not blown:
                    close_until(marks[week_i])
                    snap[si, week_i] = bal
                    week_i += 1
                if blown:
                    break
                close_until(t.time)
                if blown:
                    break
                if t.day != day:
                    day, day_start = t.day, bal
                risk_usd = INITIAL * risk
                open_risk = sum(p[1] for p in open_pos)
                if len(open_pos) >= MAX_OPEN or bal - open_risk - risk_usd <= floor or \
                        bal - open_risk - risk_usd < day_start - INITIAL * daily_pct * 0.9:
                    continue
                open_pos.append((t.exit_time, risk_usd, t.r))
            if not blown:
                close_until(end)
            while week_i < weeks:
                snap[si, week_i] = -1.0 if blown else bal     # -1 marks a blown account
                week_i += 1
        rows = []
        for w in range(weeks):
            col = snap[:, w]
            alive = col > 0
            rows.append({"week": w + 1, "alive": alive.mean(),
                         "profitable": (alive & (col > INITIAL)).mean(),
                         "median_pnl_pct": float(np.median(np.where(alive, col / INITIAL - 1, -max_pct))) * 100})
        out[risk] = pd.DataFrame(rows)
    return out, len(starts)


def weekly_verdict(track, min_profitable=0.6):
    """Good = at the best risk level, at least 60% of the year-long accounts are alive AND in
    profit after 52 weeks."""
    if not track:
        return False, None
    best_risk = max(track, key=lambda r: (track[r]["profitable"].iloc[-1], track[r]["median_pnl_pct"].iloc[-1]))
    last = track[best_risk].iloc[-1]
    return bool(last["profitable"] >= min_profitable), best_risk
