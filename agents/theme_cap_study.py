"""Bots 1-4: less loss on the bad days (user, 18.11.2026: "run a check and come with conclusions on what can be done to
cut the losses and grow the profits at the end of the day" - after 12, 17 and 18.11, when a third of bot 1's loss came
in three bursts of 10-20 trades, most of them one bet that risk goes up / down: indices, stocks, USDJPY ...).

Each bot as it trades now (its combos' own trades: 2025 = tr25 chooses, 2026 = tr only shows, agents/portfolio_check),
its fixed risk guard on, with one rule at a time:
  theme cap k    the open risk on one theme in one direction (risk-on: long indices / stocks / USDJPY, AUDUSD ...;
                 risk-off the opposite - agents/live_gap_study.theme) at most k trades' risk; a new trade is cut to fit,
                 under a quarter - skipped (k = 2, 3, 4, 6)
  theme half n   from the n-th open trade on the same theme and direction, half the risk (n = 2, 3)
  day stop x     no new trade after the day (FTMO's, Prague) has lost x% realised (x = 1, 1.5, 2)
  day lock x     no new trade after the day has made x% realised (x = 1.5, 2, 3)
  ccy cap 1.5    the existing currency rule (portfolio_check.currency_rule) for comparison
A rule is good when on 2025 it makes more and does not fall deeper; the 2026 numbers are shown, not chosen on.
  python -m agents.theme_cap_study
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agents import live_gap_study as LG
from agents import portfolio_check as P

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "theme_cap.json"
BOTS = ("bot1", "bot2", "bot3", "bot4")


def _th(c, d):
    return LG.theme(c.split("|")[2], "buy" if d > 0 else "sell")


def rule_rows(rows, mm, risk, rule, arg):
    held, out = [], {}
    day_pnl, cur = 0.0, None
    closed = []                                   # (exit, pnl%) not yet booked
    for a, b, d, r, c in rows:
        day = pd.Timestamp(a * 60, unit="s", tz="UTC").tz_convert("Europe/Prague").date()
        if day != cur:
            cur, day_pnl = day, 0.0
            closed = [x for x in closed if x[0] > a]
        keep = []
        for x in closed:
            if x[0] <= a:
                if pd.Timestamp(x[0] * 60, unit="s", tz="UTC").tz_convert("Europe/Prague").date() == cur:
                    day_pnl += x[1]
            else:
                keep.append(x)
        closed = keep
        held = [h for h in held if h[0] > a]
        m0 = float(mm.get(c, 1.0))
        m = m0
        th = _th(c, d)
        same = [h for h in held if th and h[1] == th]
        if rule == "theme_cap" and th:
            m = min(m, max(0.0, arg - sum(h[2] for h in same)))
            if m < 0.25 * m0:
                m = 0.0
        elif rule == "theme_half" and th and len(same) >= arg - 1:
            m *= 0.5
        elif rule == "day_stop" and day_pnl <= -arg:
            m = 0.0
        elif rule == "day_lock" and day_pnl >= arg:
            m = 0.0
        if m <= 1e-9:
            continue
        held.append((b, th, m))
        closed.append((b, r * risk * 100 * m))
        out.setdefault(c, [])
        out[c].append([a, b, d, r * m / (m0 or 1.0)])
    return {c: (tr, float(mm.get(c, 1.0))) for c, tr in out.items()}


RULES = [("now", None)] + [("theme_cap", k) for k in (2, 3, 4, 6)] + [("theme_half", n) for n in (2, 3)] + \
        [("day_stop", x) for x in (1.0, 1.5, 2.0)] + [("day_lock", x) for x in (1.5, 2.0, 3.0)] + [("ccy", "cap1.5")]


def main():
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "bots": {}}
    for bot in BOTS:
        cfg = P.bot_cfg(bot)
        risk, mm = P.bot_risk(bot, cfg), P.mults(bot)
        res = {}
        for per, key in (("y2025", "tr25"), ("y2026", "tr")):
            book, _ = P.book_now(bot, cfg, key)
            rows = sorted((a, b, d, r, c) for c, (tr, m) in book.items() if tr and m > 0 for a, b, d, r in P._minutes(tr))
            for name, arg in RULES:
                k = name if arg is None else f"{name}:{arg}"
                if name == "now":
                    x = P.replay(book, risk)
                elif name == "ccy":
                    x = P.replay(P.currency_rule(rows, mm, arg), risk)
                else:
                    x = P.replay(rule_rows(rows, mm, risk, name, arg), risk)
                res.setdefault(k, {})[per] = x
        now = res["now"]["y2025"]
        good = [k for k, v in res.items() if k != "now" and v["y2025"]["total"] > now["total"] and v["y2025"]["max_dd"] <= now["max_dd"] + P.EPS]
        best = max(good, key=lambda k: res[k]["y2025"]["total"], default=None)
        rep["bots"][bot] = {"rules": res, "good_2025": good, "best": best}
        print(bot, "best", best)
        for k, v in res.items():
            a, b = v["y2025"], v["y2026"]
            print(f"  {k:16} 2025 {a['total']:7} dd {a['max_dd']:5} wd {a['worst_day']:6} n {a['trades']:5} | 2026 {b['total']:7} dd {b['max_dd']:5} wd {b['worst_day']:6} n {b['trades']:5}")
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
