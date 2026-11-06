"""The user's own strategy, version 3 (user, 06.11.2026, drawn and marked on charts): identified on the 15-minute chart;
entry on the 1-minute gap inside the 15-minute gap (variant), target the nearest 1-minute swing high (variant).

  impulse   a leg of 8-30 fifteen-minute candles from a swing low to a swing high (mirror for sells) that is
            clearly bigger than the recent history: its size >= K x the typical move over the same number of
            candles in the previous ~5 days (median of |close[t] - close[t-n]|), and its average body >= 1.5 x
            the average body of the 50 candles before it ("a mass of buyers")
  entry     the FIRST fair-value gap of the leg in time (candle k-1's high below candle k+1's low); when the price
            comes back and TOUCHES the gap's near edge -> buy (the impulse's direction)
  stop      the swing low the leg started from, minus 4 pips (instruments without pips: 0.1 x ATR)
  target    the nearest swing high above the entry (usually a lower high made during the pullback), minus a
            little (2 pips / 0.05 x ATR) to close safely
  rules     risk:reward at least 1:1; the setup lives 8 hours after the leg's top is confirmed; it is gone if the
            price goes above the leg's top first, or below the stop; FTMO restricted news: no entry 2 minutes
            either side, positions closed 2 minutes before; positions closed before market breaks; one position
            at a time per instrument
  filter    variant: only when the 1-hour trend (EMA20 vs EMA50) agrees with the impulse, or not filtered
Fills and exits are walked on the 1-minute bars (to know which came first); costs = the real spread at the fill
minute + FTMO commission. TRAIN 2024-01-01..2025-04-30, VALID after.

  python own15.py --symbols EURUSD,GBPUSD
Writes output/own15_<tag>.csv.gz.
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
import run_rules as RR
from agents import own_core as CORE
from agents.own_core import VARIANTS, pip_of, resample, swings, comm_px, run_symbol  # noqa: F401

# (K, 1h-trend filter, entry: '15m' gap edge / '1m' gap inside the 15m gap, target: '15m' / '1m' nearest swing high)
# last5: the last 1-minute turning point before the entry (5 minutes each side) beyond it (the user's marks, 06.11.2026);
# last5x: the same without the 1:1 minimum
# struct: the last real correction before the entry (the user, 06.11.2026): the last 1-minute turning point beyond the
# entry after which the price made a new low below the low before it (a high in a sideways pause does not count);
# structx: the same without the 1:1 minimum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tag = a.symbols.replace(",", "_").replace("/", "-")[:60]
    ev, _ = NC.load()
    rows = []
    for s in [x for x in a.symbols.split(",") if x]:
        t0 = time.time()
        try:
            df = P.bars(s, "1Min", max(a.start, RP.MINUTE_START), end)
            sp = P.spread_1m(s, a.start, end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-120:]}")
            continue
        if df is None or df.empty or sp is None or len(sp) == 0:
            print(f"{s}: no data / no real spread - skipped")
            continue
        if "v" not in df:
            df["v"] = 1.0
        c = df["c"].to_numpy(float)
        cost = np.abs(sp.sort_index().reindex(df.index, method="ffill").to_numpy(float)) + comm_px(s, c)
        cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
        stop_at = RR.break_mask(df, "1Min")
        sm = RR.session_mask(df, s, "1Min")
        if sm is not None:
            stop_at = stop_at | np.asarray(sm, bool)
        nm, noent = RR.news_masks(df, s, "1Min")
        blk = NC.blocked(pd.Series(df.index), s, ev, minutes=2).to_numpy()
        noent = np.asarray(noent, bool) | blk
        stop_at = stop_at | np.asarray(nm, bool) | np.r_[blk[1:], False]
        rows += run_symbol(s, df, cost, stop_at, noent, VARIANTS)
        print(f"{s}: {sum(1 for x in rows if x['symbol'] == s)} trades (all variants), {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / f"own15_{tag}.csv.gz", index=False)
    print("trades:", len(rows))


if __name__ == "__main__":
    main()
