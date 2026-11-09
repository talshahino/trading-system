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


# name -> (K impulse size vs typical, BODY_X, MIN_GAP, entry mode); 'base' = exactly the watch bot
CONFIGS = {"base": (2.0, 2.5, 0.33, "1m"),
           "k15": (1.5, 2.5, 0.33, "1m"),     # smaller impulses count
           "b18": (2.0, 1.8, 0.33, "1m"),     # less strong candle bodies
           "g0": (2.0, 2.5, 0.0, "1m"),       # any gap size
           "e15": (2.0, 2.5, 0.33, "15m"),    # enter at the 15-minute gap's edge, no 1-minute gap needed
           "all": (1.5, 1.8, 0.15, "15m")}    # all of them looser together


# user (09.11.2026): "half profit and stop to break-even" - at +pl x the risk close half, move the stop to the entry,
# the other half goes on to the same target (or back to the entry). Research only, computed from the same 1-minute path.
PARTIALS = (0.5, 0.75, 1.0)


def partial_exit(x, df, stop_at, pl):
    if x.get("status") != "closed":
        return None
    d, px, stop, tgt = x["dir"], x["entry"], x["stop"], x["target"]
    risk = d * (px - stop)
    if risk <= 0:
        return None
    h1, l1, o1, c1 = (df[k].to_numpy(float) for k in ("h", "l", "o", "c"))
    i0 = df.index.get_loc(pd.Timestamp(x["time"]))
    cost_r = x["cost_r"]
    lvl = px + d * pl * risk
    half = None                       # bar where the half was taken
    for q in range(i0, len(df)):
        hit_stop = (l1[q] <= stop) if d > 0 else (h1[q] >= stop)
        if half is None:
            if hit_stop:                                  # same bar as the level: count the stop (the worse case)
                ex = (min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if q > i0 else stop
                r = d * (ex - px) / risk
                break
            if q > i0 and stop_at[q]:
                r = d * (c1[q] - px) / risk
                break
            if q > i0 and ((h1[q] >= lvl) if d > 0 else (l1[q] <= lvl)):
                if d * (tgt - px) <= pl * risk:           # the target comes first: plain trade
                    return None
                half, first = q, pl
            continue
        if (l1[q] <= px) if d > 0 else (h1[q] >= px):     # back to the entry: the second half at 0
            r = 0.5 * first + 0.5 * d * ((min(px, o1[q]) if d > 0 else max(px, o1[q])) - px) / risk
            break
        if (h1[q] >= tgt) if d > 0 else (l1[q] <= tgt):
            r = 0.5 * first + 0.5 * d * ((max(tgt, o1[q]) if d > 0 else min(tgt, o1[q])) - px) / risk
            break
        if stop_at[q]:
            r = 0.5 * first + 0.5 * d * (c1[q] - px) / risk
            break
    else:
        return None
    y = dict(x)
    y["variant"] = f"{x['variant']}|p{pl:g}"
    y["exit_time"] = df.index[q]
    y["reason"] = ("half+" if half is not None else "") + x["reason"]
    y["r"] = round(r - cost_r, 4)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--configs", default="", help="looser variants to test, e.g. base,k15,b18 (see CONFIGS)")
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
        if a.configs:
            # research only (user, 09.11.2026: more trades?): the bot's 'struct' rules with one filter loosened at a time.
            # own_core's module constants are set for each run and restored after, the bot's file is not changed.
            keep = (CORE.BODY_X, CORE.MIN_GAP)
            for name in a.configs.split(","):
                K, bx, mg, em = CONFIGS[name]
                CORE.BODY_X, CORE.MIN_GAP = bx, mg
                for x in run_symbol(s, df, cost, stop_at, noent, [(K, False, em, "struct")]):
                    x["variant"] = name
                    rows.append(x)
                    for pl in PARTIALS:
                        y = partial_exit(x, df, stop_at, pl)
                        if y is not None:
                            rows.append(y)
            CORE.BODY_X, CORE.MIN_GAP = keep
        else:
            rows += run_symbol(s, df, cost, stop_at, noent, VARIANTS)
        print(f"{s}: {sum(1 for x in rows if x['symbol'] == s)} trades (all variants), {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / f"own15_{tag}.csv.gz", index=False)
    print("trades:", len(rows))


if __name__ == "__main__":
    main()
