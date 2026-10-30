---
name: trading-prophet
description: Trading Prophet – judge any trading strategy or MT5 robot against exact FTMO prop-firm rules ($100K 2-Step by default). Use when the user wants to know if a strategy/EA/backtest would pass FTMO, batch-test MT5 robots with an FTMO gate, or test strategies collected from videos/social media.
---

# Trading Prophet

Decides whether a strategy would survive and pass an FTMO challenge — not whether it
makes the most money. Fail-closed: missing data never turns into a pass.

## Pipeline (stage 1 – built)

```
MT5 robot / strategy ──► MT5 Strategy Tester report (.htm)
                         + 1-minute bars per symbol (floating loss)
                         + high-impact news calendar
                    ──► scripts/prophet.py ──► Hebrew report + JSON verdict
```

Verdicts: `PASS` (ready for demo) · `FAIL` (strategy stops) · `PROVISIONAL`
(closed trades only, no floating-loss data) · `INSUFFICIENT` (too little history).

## What is checked

1. **FTMO rules** (`config/ftmo_presets.json`, verify on ftmo.com before buying):
   Challenge +10%, Verification +5%, max daily loss 5% **on equity** measured from the
   balance at midnight Europe/Prague, max loss 10% static, min 4 trading days,
   target counted only when flat.
2. **Continuous account** from day 1 through both phases.
3. **Rolling starts** – a new challenge every 7 days over the whole history
   (pass rate, breach rate, days to pass) so a lucky start date can't fake a pass.
4. **Weekly checkpoints up to 52 weeks** on one account; stops at the first breach.
5. **News blackout** – any open/close within ±5 min of a high-impact release on a
   currency of the instrument (default policy: fail).

Pass bar (`prophet_policy`): no breach on the continuous account, ≥8 completed rolling
samples, pass rate ≥50%, breach rate ≤25%, profitable at the last weekly checkpoint,
no news violations, full equity data.

## Run one strategy

```bash
python scripts/prophet.py --report "Bot_EURUSD.htm" --bars bars/ --news news.csv \
    --out verdict.md --json verdict.json
# or a direct equity export:
python scripts/prophet.py --equity equity.csv
```

- `--report-tz` – timezone of the MT5 server clock in the report (FTMO servers:
  `Europe/Athens`, i.e. GMT+2/+3).
- `--scale relative|fixed` – P&L scaled as % of current balance (risk-% EAs) or
  of the starting deposit (fixed-lot EAs).
- Bars folder: one CSV per symbol named `SYMBOL_*.csv`, MT5 "Export bars" format
  or `time,open,high,low,close`. Without bars the verdict is at best `PROVISIONAL`.
- News CSV: `time,currency,impact,title` in UTC (the fxmacrodata-calendar /
  economic-calendar-fetcher skills from tradermonty/claude-trading-skills can feed it).

## Inside this repo's rules backtest (run_rules.py)

Every surviving (or `--export`ed) strategy is replayed by `scripts/ledger_adapter.py`
on 1-minute Dukascopy bars. The repo's own account replay checks limits only when a
trade closes; Prophet adds the intraday floating loss. Position size per trade comes
from `risk_usd / |entry − stop|`. Result appears as the "Trading Prophet" column in
`report_rules.html`.

## Batch-test MT5 robots with the FTMO gate

`mt5_tester/` is the vendored mt5-robot-tester (MIT, TraderMonty) with one change:
with `"gates": {"mode": "ftmo"}` the finalist decision is the Prophet verdict instead
of "profit ≥4× deposit, DD ≤12%". Use `config/pipeline_config.ftmo.json` as the
template; a `<report>.prophet.md` is written next to each final report. Requires
Windows + MetaTrader 5 at run time (see `mt5_tester/SKILL.md`).

## How the floating loss is rebuilt

For every minute a position is open: worst price in the bar (long → low, short → high),
times money-per-price-unit derived from that trade's own realized profit
(`profit / (close − open)`), minus its commission up front. No contract specs needed.
Conservative by design; short trades ignore the spread (noted limitation).

## Known limits (stage 1)

- 1-Step preset rules are uncertain across sources; Best Day rule not modelled.
- Rolling pass rate ignores samples that ran out of history (shown as "incomplete").
- MT5 Round 3 still optimizes for max profit; FTMO-aware optimization is a later stage.

## Next stages

2. News/strategy catalog: videos → rules → backtest → Prophet verdict.
3. Live/demo drift monitor + daily equity circuit breaker on the FTMO account.
