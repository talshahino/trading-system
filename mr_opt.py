"""Mean-reversion scalp on the 1-minute chart (user, 06.11.2026). The entry scan (entry_scan.py, E06) found one
entry whose price path beats chance in both periods: a close outside the Bollinger band, faded - strongest on
the currency crosses (EURGBP, GBPJPY, USDCHF, AUDUSD), worth about one trading cost per trade before exits.
Here the entry is refined and an exit fitted to it, every setting with FTMO's own costs:

  entry  band  2.5, 3, 3.5, 4 standard deviations      lookback 20, 60 bars
         hours all / Asia 22-06 UTC / Europe 07-12 UTC / America 13-20 UTC
  exit   target: back to the band's middle (as it stood at the signal), or half of the way there
         stop 1.5, 3, 5 ATR(14)      out after 30, 120, 240 bars
Chosen on TRAIN (2024-01-01 .. 2025-04-30), judged on VALID (from 2025-04-30).

  python mr_opt.py --symbols EURGBP,GBPJPY
Writes output/mr_opt_<tag>.json and output/mr_daily_<tag>.csv.gz (daily R of every setting).
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
from scalp_opt import TRAIN_FROM, VALID_FROM, stats

SIGMA = [2.5, 3.0, 3.5, 4.0]
LOOK = [20, 60]
HOURS = {"all": None, "asia": (22, 6), "eu": (7, 12), "us": (13, 20)}
TGT = [1.0, 0.5]           # share of the way back to the middle
STOP = [1.5, 3.0, 5.0]
BARS = [30, 120, 240]


def hour_mask(df, w):
    if w is None:
        return np.ones(len(df), bool)
    h = df.index.hour.to_numpy()
    a, b = w
    return (h >= a) | (h < b) if a > b else (h >= a) & (h < b)


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
            continue
        real = s in U.ALL
        cost = 0.0 if real else RP.cost_of(s)
        if real and s not in RR.SPREADS:
            try:
                RR.SPREADS[s] = P.spread_1m(s, (VALID_FROM - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), end)
            except Exception:  # noqa: BLE001
                RR.SPREADS[s] = None
        sess = RR.break_mask(df, "1Min")
        sm = RR.session_mask(df, s, "1Min")
        if sm is not None:
            sess = sess | np.asarray(sm, bool)
        nm, noent = RR.news_masks(df, s, "1Min")
        sess = sess | np.asarray(nm, bool)
        noent = np.r_[np.asarray(noent, bool)[1:], False]       # the entry bar is the next one
        c = df["c"].to_numpy(float)
        for lk in LOOK:
            mid = pd.Series(c).rolling(lk).mean().to_numpy()
            sd = pd.Series(c).rolling(lk).std().to_numpy()
            for sg in SIGMA:
                lo_x = (c < mid - sg * sd)
                hi_x = (c > mid + sg * sd)
                for hn, w in HOURS.items():
                    hm = hour_mask(df, w) & ~noent
                    L, S = lo_x & hm, hi_x & hm
                    if L.sum() + S.sum() < 20:
                        continue
                    for tg in TGT:
                        lvl_l = c + tg * (mid - c)          # long: target above the close, toward the middle
                        lvl_s = c - tg * (c - mid)
                        for st in STOP:
                            for nb in BARS:
                                key = f"b{lk}|s{sg:g}|{hn}|t{tg:g}|x{st:g}|m{nb}"
                                plan = {"stop": ("atr", st), "target": ("level", lvl_l, lvl_s), "max_bars": nb}
                                try:
                                    t = R.run(df, lambda _d, L=L, S=S, plan=plan: (L, S, plan), cost, breakeven=None, session_exit=sess)
                                    if t.empty:
                                        continue
                                    if real:
                                        t = C.apply(t, s, RR.SPREADS.get(s))
                                    t = t[~(NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events))]
                                except Exception:  # noqa: BLE001
                                    errors.append(f"{s} {key}:\n{traceback.format_exc()}")
                                    continue
                                tm = pd.to_datetime(t["time"], utc=True)
                                tr, va = t[(tm >= TRAIN_FROM) & (tm < VALID_FROM)], t[tm >= VALID_FROM]
                                res.append({"symbol": s, "variant": key, "look": lk, "sigma": sg, "hours": hn, "target": tg,
                                            "stop": st, "bars": nb, "train": stats(tr), "valid": stats(va)})
                                x = t[tm >= TRAIN_FROM]
                                if len(x):
                                    dd = x.groupby(pd.to_datetime(x["time"], utc=True).dt.strftime("%Y-%m-%d"))["r"].sum()
                                    daily.append(pd.DataFrame({"symbol": s, "variant": key, "day": dd.index, "r": dd.round(3).values}))
            print(f"  {s} look {lk}: {time.time() - t0:.0f}s", flush=True)
        print(f"{s}: {len(df):,} bars, {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / f"mr_opt_{tag}.json").write_text(json.dumps({"rows": res, "errors": errors}, ensure_ascii=False), encoding="utf-8")
    if daily:
        pd.concat(daily, ignore_index=True).to_csv(config.OUTPUT_DIR / f"mr_daily_{tag}.csv.gz", index=False)
    print("variants:", len(res), "errors:", len(errors))
    for e in errors[:5]:
        print(e)


if __name__ == "__main__":
    main()
