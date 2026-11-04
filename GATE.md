# The research gate (registered 04.11.2026, before any new candidate is judged)

Written after the external review: ~1% of everything tested passes the old test even with no edge, and
every later check (halves, leave-one-out, bot replay, Monte Carlo) used the same period the combos were
chosen from. The search never ends, so without this gate the bots keep collecting lucky winners.

## 0. Freeze
No new combo enters a bot until this gate is in force (user's decision, 04.11.2026). The five combos
added on 04.11.2026 stay at full size (user's decision).

## 1. One test, declared in advance
- Rule set: the live one only (`--stock-session-exit --breakeven 0.6 --respect-tf --break-exit --news-exit`).
- One risk level: the bot's own. Other rule sets or risks are diagnostics, never a reason to accept.
- Judged on 2025-04-30 .. 2026-10-31.
- `research_gate.py` counts every evaluation ever run (N). A combo is accepted only if its trade-level t
  is at least the expected best t of N tries with no edge (N = 52,908 different combos on 04.11.2026:
  t >= 4.26). Halves, LOO and Monte Carlo stay as stability diagnostics only.

## 2. Hold-out
The demo record from 2026-11-01 on is never used to choose, tune or rank anything. It is only the judge.

## 3. Trial size, then full size
A combo that clears 1. starts at one tenth of the bot's risk. It goes to full size only after a sequential
test (SPRT) on its own demo trades: H0 mean R = 0 vs H1 mean R = half its backtest mean, alpha = beta = 0.1.
Accept H1 -> full size; accept H0 -> removed.

## 4. Fixed removal rule (all combos, including the existing ones)
Removed, with no second chance, when its own demo record, cumulative R since 2026-11-01, falls below
-2 x sqrt(trades) x (its backtest standard deviation of R) after at least 6 trades: a fall a no-loss-in-
expectation combo makes less than 2.5% of the time.

## 5. Luck rate, monthly
Once a month the whole pipeline runs on null strategies (`--null-shift 37`: every signal moved 37 bars
away from the prices it came from). How many of those "pass" is our luck rate; it is reported next to
every new result.
