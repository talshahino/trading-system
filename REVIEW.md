# For an external reviewer

Read-only review of the whole system. Do not push, open issues or PRs.

- Bots: `live/runner.py` (one process per MetaTrader 5 demo account), `live/manager.py` (starts/watches bots, installs approved updates, app API), configs `live/bots/bot1.json`..`bot4.json` (combos `strategy|timeframe|instrument`, risk per trade).
- Strategy engine shared by backtest and bots: `agents/rules_agent.py` (`run()`), families in `agents/*_strategies.py`.
- Backtests (GitHub Actions): `run_rules.py`, `bots_compare.py` (whole-bot FTMO replay), `montecarlo.py`, `loo.py`, `improve.py`, `bot4.py`; FTMO simulator `skills/trading-prophet/scripts/ftmo_sim.py`, `agents/ftmo_agent.py`; results in `reports/`.
- App: `docs/index.html`, `docs/sw.js`. Live account state: branch `live`, `live/*_state.json`.
- Rules: risk 0.375-0.45% per trade, pre-entry risk check, guards 4.5% daily / 9% total with graded risk, emergency stop at 4.5% daily, Prague-midnight day, close before market breaks > 2 h, FTMO restricted-news rule, shares closed at session end, breakeven at 60% to target, demo servers only.

Please check, most severe first, with file:line and a concrete failing scenario:
1. Look-ahead / survivorship bias in strategies and engine.
2. Engine realism (next-open entry, stop-before-target in a bar, gaps, costs, breakeven).
3. Selection bias: thousands of combos tested, ~1% kept - were the later checks really out of sample?
4. Backtest vs live: does `live/runner.py` feed strategies exactly what `run_rules.py` does?
5. Risk sizing and FTMO rules (floating equity in the simulator).
6. Failure handling: any path that leaves a position without a stop.
7. Security: PIN hashing, read-only code, encrypted manager address, self-update approval.
