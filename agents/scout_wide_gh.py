"""Every new scout strategy on every bot's instruments, NOW, on many GitHub machines in parallel (user, 14.11.2026:
"when is the analysis of all the videos?" -> "yes, run it in parallel"). The same test the bots' agents run a slice at
a time (agent_portfolio.test_new: the strategy's own timeframes, live rules, FTMO costs, chosen on the period to
2025-12-31, shown on 2026) - one job per bot x instrument. Results are the same per-combo files the agents read
(reports/agents/<bot>/exp/new_*.json), so the bots' agents and the manager make the cards from them on their next run.

  python -m agents.scout_wide_gh plan                          # the job list (JSON) for the workflow's matrix
  python -m agents.scout_wide_gh run --job bot1:XAUUSD --minutes 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time

BOTS = ["bot5", "bot6", "bot7", "bot1", "bot2", "bot3", "bot4"]     # the user's priority: the scalping bots first


def plan():
    import agent_portfolio as A
    jobs = []
    for b in BOTS:
        syms = A.NEW_ONLY.get(b) or A.bot_symbols(b)
        jobs += [f"{b}:{s}" for s in syms]
    return jobs


def run(job, minutes):
    import agent_portfolio as A
    bot, sym = job.split(":", 1)
    A.BOT_NOW["bot"] = bot
    t0 = time.time()
    n = A.test_new(bot, minutes * 60, "all", symbols=[sym])
    print(f"{job}: {n} combos tested in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run"])
    ap.add_argument("--job", default="")
    ap.add_argument("--minutes", type=float, default=90)
    a = ap.parse_args()
    if a.mode == "plan":
        json.dump(plan(), sys.stdout)
    else:
        run(a.job, a.minutes)
