"""Risk brakes across bots 1-4 (user, 17.11.2026 16:15, task tal1711riskmgmt): "how do you not have risk-management
strategies?" after two red days in a row in every bot. Each brake is replayed on the whole bot (and on the four bots
together) exactly like agents/portfolio_check.replay - every combo's own trades as it trades now, the live fixed risk
rule (open trades count as stopped, 4.5% a day / 9% in all) - once on the choosing period (30.04.2025-31.12.2025,
"tr25") and once on 2026 ("tr", unseen). Nothing is optimised per trade.

Brakes (all are below the fixed 5% / 10% limits, which never change):
  a) soft daily brake: once the day's realised result is -1 / -1.5 / -2% of the day's starting balance, new trades get
     half risk ("half") or none until tomorrow ("stop")
  b) after 2 losing days in a row: risk x0.75 / x0.5 until a green day
  c) no new entry 30 / 60 / 120 minutes after a high-impact USD release (FTMO's calendar, data/ftmo_calendar)
  d) a strategy that runs in more than one bot: its risk split between them (x 1/n), or kept only in the bot where its
     2025 result is best - measured on the four bots together (sum of the four accounts) and their daily correlation
Measures per run (user, 16.11.2026, mgr1611curve): total %, max fall from the peak %, % green days, worst day %, daily
spread %. A brake is a card only when it beats "as today" on BOTH periods: more (or equal) profit and no larger fall,
or a clearly smoother curve that still earns (the manager decides).

  python -m agents.risk_brakes        -> reports/agents/risk_brakes.json
"""
from __future__ import annotations

import heapq
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents import portfolio_check as P

ROOT = Path(__file__).resolve().parent.parent
BOTS = ("bot1", "bot2", "bot3", "bot4")
OUT = ROOT / "reports" / "agents" / "risk_brakes.json"


def usd_news():
    """High-impact USD releases (minutes since the epoch, UTC)."""
    try:
        ev = json.loads((ROOT / "data" / "ftmo_calendar" / "history.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return np.array([], dtype=np.int64)
    ts = sorted({int(pd.Timestamp(e["date"]).timestamp() // 60) for e in ev
                 if str(e.get("instrument") or "").startswith("USD") and e.get("impact") == "high" and e.get("date")})
    return np.array(ts, dtype=np.int64)


NEWS = usd_news()


def brake_run(book, risk, day_thr=None, day_mode="half", streak_mult=None, news_min=None, fit=0.0, start=100.0):
    """portfolio_check.guarded with the brakes on top -> DataFrame (a, b, d, pnl, risk). fit: the bot's own "dd_fit"
    (live/runner.Account.allowed_risk: risk x room left to the 9% limit / fit), in every run as it is live."""
    rows = [(a, b, d, r, risk * 100 * m) for combo, (tr, m) in book.items() if tr and m > 0
            for a, b, d, r in P._minutes(tr)]
    cols = ["a", "b", "d", "pnl", "risk"]
    if not rows:
        return pd.DataFrame(rows, columns=cols)
    rows.sort(key=lambda x: (x[0], x[1]))
    ent = pd.to_datetime([x[0] for x in rows], unit="m", utc=True).tz_convert("Europe/Prague")
    ext = pd.to_datetime([x[1] for x in rows], unit="m", utc=True).tz_convert("Europe/Prague")
    eday = [x.date() for x in ent]
    xday = [x.date() for x in ext]
    st = {"bal": start, "ds": start, "cur": None, "losing": 0}
    open_r, heap, out = 0.0, [], []

    def new_day(d):
        if st["cur"] is None or d > st["cur"]:
            if st["cur"] is not None:
                res = st["bal"] - st["ds"]
                st["losing"] = st["losing"] + 1 if res < -1e-9 else (0 if res > 1e-9 else st["losing"])
            st["ds"], st["cur"] = st["bal"], d

    for i, (a, b, d, r, k) in enumerate(rows):
        while heap and heap[0][0] <= a:
            _, j, pnl, rk = heapq.heappop(heap)
            new_day(xday[j])
            st["bal"] += pnl
            open_r -= rk
        new_day(eday[i])
        if news_min and len(NEWS):
            j = np.searchsorted(NEWS, a, side="right") - 1
            if j >= 0 and a - NEWS[j] <= news_min:
                continue
        kk = k
        if 0.5 <= fit <= 9:
            kk *= min(1.0, max(0.0, (st["bal"] - (100 - P.GUARD_ALL)) / fit))
        if day_thr is not None and st["bal"] - st["ds"] <= -day_thr:
            if day_mode == "stop":
                continue
            kk = k * 0.5
        if streak_mult is not None and st["losing"] >= 2:
            kk = kk * streak_mult
        left = st["bal"] - open_r
        room = min(left - (100 - P.GUARD_ALL), left - (st["ds"] - P.GUARD_DAY))
        use = min(kk, P.GRADE_SHARE * room) if room < 2 * kk else kk
        if use < P.GRADE_MIN * kk:
            continue
        open_r += use
        heapq.heappush(heap, (b, i, r * use, use))
        out.append((a, b, d, r * use, use))
    return pd.DataFrame(out, columns=cols)


def daily_of(t):
    if t.empty:
        return pd.Series(dtype=float)
    day = pd.to_datetime(t["b"], unit="m", utc=True).dt.tz_convert("Europe/Prague").dt.date
    return t.groupby(day)["pnl"].sum().sort_index()


def measures(daily):
    if daily.empty:
        return {"total": 0.0, "max_dd": 0.0, "green": 0.0, "worst_day": 0.0, "spread": 0.0, "days": 0}
    eq = np.r_[100.0, 100 + daily.cumsum().to_numpy()]
    return {"total": round(float(daily.sum()), 2),
            "max_dd": round(float((np.maximum.accumulate(eq) - eq).max()), 2),
            "green": round(float((daily > 0).mean() * 100), 1),
            "worst_day": round(float(daily.min()), 2),
            "spread": round(float(daily.std()), 2), "days": int(len(daily))}


VARIANTS = [("as_today", {})]
VARIANTS += [(f"day{t}_{m}", {"day_thr": t, "day_mode": m}) for t in (1.0, 1.5, 2.0) for m in ("half", "stop")]
VARIANTS += [(f"streak2_x{s}", {"streak_mult": s}) for s in (0.75, 0.5)]
VARIANTS += [(f"usdnews{n}", {"news_min": n}) for n in (30, 60, 120)]


def beats(v, base):
    """Better on a period: at least the profit and no larger fall, or a clearly smoother curve (more green days,
    smaller fall, better worst day) that keeps 90%+ of the profit."""
    more = v["total"] >= base["total"] - 0.1 and v["max_dd"] <= base["max_dd"] + 0.05
    smoother = (v["max_dd"] < base["max_dd"] - 0.3 and v["worst_day"] >= base["worst_day"] and v["green"] >= base["green"]
                and v["total"] >= 0.9 * base["total"])
    return bool(more or smoother)


def shared_combos():
    """Combos that run in more than one of bots 1-4 -> {combo: [bots]}."""
    where = {}
    for b in BOTS:
        for c in P.bot_cfg(b).get("combos") or []:
            where.setdefault(c, []).append(b)
    return {c: bs for c, bs in where.items() if len(bs) > 1}


def run():
    books = {b: {k: P.book_now(b, key=k)[0] for k in ("tr25", "tr")} for b in BOTS}
    risk = {b: P.bot_risk(b) for b in BOTS}
    fit = {b: float(P.bot_cfg(b).get("dd_fit") or 0) for b in BOTS}
    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bots": {}, "all4": {},
           "note": "realised result for the daily brake (live equity also counts open trades); 2025 = 30.04-31.12.2025"}
    dailies = {}
    for b in BOTS:
        res = {}
        for name, kw in VARIANTS:
            res[name] = {}
            for per, key in (("2025", "tr25"), ("2026", "tr")):
                d = daily_of(brake_run(books[b][key], risk[b], fit=fit[b], **kw))
                dailies[(b, name, per)] = d
                res[name][per] = measures(d)
        for name, _ in VARIANTS[1:]:
            res[name]["wins_both"] = all(beats(res[name][p], res["as_today"][p]) for p in ("2025", "2026"))
        out["bots"][b] = res
    # the four bots together, same brake in every bot
    for name, _ in VARIANTS:
        out["all4"][name] = {}
        for per in ("2025", "2026"):
            s = pd.concat([dailies[(b, name, per)] for b in BOTS], axis=1).fillna(0.0)
            out["all4"][name][per] = measures(s.sum(axis=1) / len(BOTS))
        if name != "as_today":
            out["all4"][name]["wins_both"] = all(beats(out["all4"][name][p], out["all4"]["as_today"][p])
                                                 for p in ("2025", "2026"))
    # d) the same strategy in several bots
    sh = shared_combos()
    best = {}
    for c, bs in sh.items():
        sc = {}
        for b in bs:
            tr = (books[b]["tr25"].get(c) or (None, 0))[0] or []
            sc[b] = sum(x[3] for x in tr)
        best[c] = max(sc, key=sc.get)
    dup = {}
    for mode in ("split", "best_only"):
        dup[mode] = {}
        per_bot = {}
        for per, key in (("2025", "tr25"), ("2026", "tr")):
            ds = []
            for b in BOTS:
                bk = {}
                for c, (tr, m) in books[b][key].items():
                    if c in sh:
                        m = m / len(sh[c]) if mode == "split" else (m if best[c] == b else 0.0)
                    bk[c] = (tr, m)
                d = daily_of(brake_run(bk, risk[b], fit=fit[b]))
                per_bot.setdefault(b, {})[per] = measures(d)
                ds.append(d)
            s = pd.concat(ds, axis=1).fillna(0.0)
            dup[mode][per] = measures(s.sum(axis=1) / len(BOTS))
            dup[mode][per]["corr"] = round(float(np.nanmean(s.corr().to_numpy()[np.triu_indices(len(BOTS), 1)])), 3)
        dup[mode]["per_bot"] = per_bot
        dup[mode]["wins_both"] = all(beats(dup[mode][p], out["all4"]["as_today"][p]) for p in ("2025", "2026"))
    for per in ("2025", "2026"):
        s = pd.concat([dailies[(b, "as_today", per)] for b in BOTS], axis=1).fillna(0.0)
        out["all4"]["as_today"][per]["corr"] = round(float(np.nanmean(s.corr().to_numpy()[np.triu_indices(len(BOTS), 1)])), 3)
    dup["shared"] = len(sh)
    out["shared"] = dup
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    o = run()
    for b, res in o["bots"].items():
        print(b)
        for n, v in res.items():
            print(f"  {n:14s} 25 {v['2025']}  26 {v['2026']}  {'WIN' if v.get('wins_both') else ''}")
    print("all4")
    for n, v in o["all4"].items():
        print(f"  {n:14s} 25 {v['2025']}  26 {v['2026']}  {'WIN' if v.get('wins_both') else ''}")
    for m in ("split", "best_only"):
        v = o["shared"][m]
        print(f"  dup {m:10s} 25 {v['2025']}  26 {v['2026']}  {'WIN' if v.get('wins_both') else ''}")
