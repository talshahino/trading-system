"""Scalping optimisation (user, 06.11.2026): every 1-minute scalping strategy (SC01-SC04) on its own, with a
grid of target x stop distance x holding time, to find each strategy's best setting, then combine them.

Grid (the dollar risk stays the same; a wider stop means a smaller position, so the costs weigh less):
    target   1, 1.5, 2, 3, 4, 6 R
    stop     x1, x1.5, x2, x3 the strategy's own distance
    time     out after 30, 60, 120, 240 one-minute bars
Same trade rules as the bots' backtests: next bar's open, FTMO's own spread and commission per trade,
stop to breakeven + costs after 60% of the way to the target, stocks closed at the end of the session,
everything closed before market breaks, no trades around FTMO's restricted news.

Choosing and checking are separate: a setting is chosen on TRAIN (2024-01-01 .. 2025-04-30) and judged on
VALID (from 2025-04-30), which the choice never saw.

  python scalp_opt.py --symbols EURUSD,XAUUSD [--start 2024-01-01] [--end YYYY-MM-DD]
Writes output/scalp_opt_<tag>.json (stats of every variant) and output/scalp_daily_<tag>.csv.gz (daily R of
every variant, for the combination step in scalp_combine.py).
"""
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

import config
from agents import ftmo_costs as C
from agents import ftmo_universe as U
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
import run_rules as RR

SIDS = ["SC01", "SC02", "SC03", "SC04"]
TARGETS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
STOPS = [1.0, 1.5, 2.0, 3.0]
BARS = [30, 60, 120, 240]
TRAIN_FROM, VALID_FROM = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-04-30", tz="UTC")
BE = 0.6


def stats(t):
    r = t["r"]
    if r.empty:
        return {"n": 0}
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    hold = (pd.to_datetime(t["exit_time"], utc=True) - pd.to_datetime(t["time"], utc=True)).dt.total_seconds() / 60
    return {"n": int(len(r)), "sum": round(float(r.sum()), 2), "avg": round(float(r.mean()), 4),
            "sd": round(float(r.std()), 3) if len(r) > 1 else 0.0,
            "win": round(float((r > 0).mean()), 3), "pf": round(float(gain / loss), 3) if loss > 0 else 99.0,
            "hold_min": round(float(hold.median()), 1)}


def frozen(L, S, plan, target, stop, bars):
    p = dict(plan, target=("r", target), max_bars=bars)
    if stop != 1.0:
        p["stop_scale"] = stop
    return lambda df: (L, S, p)


def be_offset(df, s, real):
    px = float(df["c"].median())
    if not real:
        return 0.00005
    sp = RR.SPREADS.get(s)
    sp_px = float(sp.median()) if sp is not None and len(sp) else px * C.EST_SPREAD.get(U.CLASS[s], 0.0003)
    spec = U.spec(s)
    if spec["commissionType"] == "flat_USD":
        comm = px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
    else:
        comm = px * 2 * float(spec["commission"]) / 100.0
    return (sp_px + comm) / px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    args = ap.parse_args()
    end = args.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    events, _ = NC.load()
    tag = args.symbols.replace(",", "_").replace("/", "-")[:60]
    res, daily, errors = [], [], []
    for s in [x for x in args.symbols.split(",") if x]:
        t0 = time.time()
        try:
            df = P.bars(s, "1Min", max(args.start, RP.MINUTE_START), end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-150:]}")
            continue
        if df is None or df.empty:
            print(f"{s}: NO DATA")
            continue
        df.attrs.update(symbol=s, tf="1Min")
        real = s in U.ALL
        cost = 0.0 if real else RP.cost_of(s)
        if real and s not in RR.SPREADS:
            try:
                RR.SPREADS[s] = P.spread_1m(s, (VALID_FROM - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), end)
            except Exception:  # noqa: BLE001
                RR.SPREADS[s] = None
        sess = RR.session_mask(df, s, "1Min")
        sess = np.asarray(sess, dtype=bool) | RR.break_mask(df, "1Min") if sess is not None else RR.break_mask(df, "1Min")
        nm, news_block = RR.news_masks(df, s, "1Min")
        sess = np.asarray(sess, dtype=bool) | nm
        be = (BE, be_offset(df, s, real))
        for sid in SIDS:
            try:
                fn = RR._blocked(R.STRATEGIES[sid][1], news_block) if news_block is not None else R.STRATEGIES[sid][1]
                L, S, plan = fn(df)
            except Exception:  # noqa: BLE001
                errors.append(f"{sid} {s}:\n{traceback.format_exc()}")
                continue
            for tg in TARGETS:
                for sc in STOPS:
                    for nb in BARS:
                        key = f"t{tg:g}|s{sc:g}|b{nb}"
                        try:
                            t = R.run(df, frozen(L, S, plan, tg, sc, nb), cost, breakeven=be, session_exit=sess)
                            if t.empty:
                                continue
                            if real:
                                t = C.apply(t, s, RR.SPREADS.get(s))
                            t = t[~(NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events))]
                        except Exception:  # noqa: BLE001
                            errors.append(f"{sid} {s} {key}:\n{traceback.format_exc()}")
                            continue
                        tm = pd.to_datetime(t["time"], utc=True)
                        tr, va = t[(tm >= TRAIN_FROM) & (tm < VALID_FROM)], t[tm >= VALID_FROM]
                        res.append({"sid": sid, "symbol": s, "variant": key, "target": tg, "stop": sc, "bars": nb,
                                    "train": stats(tr), "valid": stats(va)})
                        x = t[tm >= TRAIN_FROM]
                        if len(x):
                            d = x.groupby(pd.to_datetime(x["time"], utc=True).dt.strftime("%Y-%m-%d"))["r"].agg(["sum", "size"])
                            daily.append(pd.DataFrame({"sid": sid, "symbol": s, "variant": key, "day": d.index,
                                                       "r": d["sum"].round(3).values, "n": d["size"].values}))
            base = next((x for x in res if x["sid"] == sid and x["symbol"] == s and x["variant"] == "t1.5|s1|b30"), None)
            print(f"  {s} {sid}: done ({time.time() - t0:.0f}s)", flush=True)
        print(f"{s}: {len(df):,} bars, {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / f"scalp_opt_{tag}.json").write_text(json.dumps({"rows": res, "errors": errors}, ensure_ascii=False),
                                                             encoding="utf-8")
    if daily:
        pd.concat(daily, ignore_index=True).to_csv(config.OUTPUT_DIR / f"scalp_daily_{tag}.csv.gz", index=False)
    print("variants:", len(res), "errors:", len(errors))
    for e in errors[:5]:
        print(e)


if __name__ == "__main__":
    main()
