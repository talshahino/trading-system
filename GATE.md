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
Proposed for removal (shown in the app under ניהול בוטים, no notification; removed only after the user approves - user, 05.11.2026), when its own demo record, cumulative R since 2026-11-01, falls below
-2 x sqrt(trades) x (its backtest standard deviation of R) after at least 6 trades: a fall a no-loss-in-
expectation combo makes less than 2.5% of the time.

## 5. Luck rate, monthly
Once a month the whole pipeline runs on null strategies (`--null-shift 37`: every signal moved 37 bars
away from the prices it came from). How many of those "pass" is our luck rate; it is reported next to
every new result.

## 6. Fair gate and portfolio test (user's approval 08.11.2026, written BEFORE these tests were run)
The per-combo gate in 1. cannot be passed by a daily or 4-hour strategy whatever its quality (18 months of
one instrument gives too few trades). Added, not replacing 1.:

A. Family test. A family = one strategy on one time frame, pooled over every instrument it was run on
   (for each family the run with the most instruments; base rules, no optimised variant). The bar is the
   expected best t of N families with no edge, N = every family ever tried (695 on 08.11.2026 -> t >= 3.17),
   on the test window 2025-04-30 .. 2026-10-31, and its average R must be above 0 in both halves of it.

B. Portfolio test (one test, decided now). Selection window 2025-04-30 .. 2025-12-31, test window
   2026-01-01 .. 2026-10-31 (never used to choose). The portfolio = every family with at least 30 trades and
   pooled t >= 2.0 in the selection window, equal weight (1 R per trade). It passes when, in the test window,
   its per-trade t >= 2.0 AND its daily result t >= 2.0. No second try with other thresholds.

Either pass makes a candidate only: it then goes through 3. (one tenth of the risk on the demo, SPRT)
and enters a bot only with the user's approval.

## 7. Long-history families (08.11.2026, written BEFORE RS01-RS03 were run)
For monthly and daily strategies (RS01 time-series momentum, RS02 share momentum, RS03 RSI(2) on indices,
rules in agents/research_strategies.py) 18 months hold too few trades. They are judged as families
(pooled over their declared universe, EVERY instrument's trades from all_trades_*.csv.gz) on
2019-01-01 .. 2026-10-31: t >= the expected best t of N families (N = 698 -> 3.17), and average R above 0
on 2025-04-30 .. 2026-10-31. Rule set: --respect-tf only (the stock session exit would close a monthly
hold every evening). No second version.
