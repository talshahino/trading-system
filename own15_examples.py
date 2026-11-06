"""Examples of own strategy v2 (own15.py) for the user's review by eye: N trades (mixed winners and losers) with
their 15-minute candles. Writes output/own15_examples.json.

  python own15_examples.py --symbols EURUSD,US30,XAUUSD --n 3 [--seed 1]
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
from agents import news_calendar as NC
from agents import poc_agent as P
import run_poc as RP
import run_rules as RR
import own15 as O
from own_strategy import resample
from scalp_map import comm_px

VARIANT = (3.0, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--body", type=float, default=1.5)
    ap.add_argument("--min-gap", type=float, default=0.0)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ev, _ = NC.load()
    O.BODY_X, O.MIN_GAP = a.body, a.min_gap
    variant = (a.k, False)
    allt, data = [], {}
    for s in [x for x in a.symbols.split(",") if x]:
        try:
            df = P.bars(s, "1Min", max(a.start, RP.MINUTE_START), end)
            sp = P.spread_1m(s, a.start, end)
        except Exception as e:  # noqa: BLE001
            print(s, "no data", e)
            continue
        if df is None or df.empty or sp is None or len(sp) == 0:
            continue
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
        tr = O.run_symbol(s, df, cost, stop_at, noent, [variant])
        allt += tr
        data[s] = df
        print(s, len(tr), "trades", flush=True)
    t = pd.DataFrame(allt)
    rng = np.random.default_rng(a.seed)
    win, loss = t[t.r > 0], t[t.r <= 0]
    k = (a.n + 1) // 2
    pick = pd.concat([win.iloc[rng.choice(len(win), min(k, len(win)), replace=False)],
                      loss.iloc[rng.choice(len(loss), min(a.n - k, len(loss)), replace=False)]]).sample(frac=1, random_state=a.seed)
    out = []
    for n_, (_, x) in enumerate(pick.iterrows(), 1):
        m15 = resample(data[x.symbol], "15min")
        w = m15[(m15.index >= x.leg_start - pd.Timedelta(hours=8)) & (m15.index <= x.exit_time + pd.Timedelta(hours=3))]
        pip = O.pip_of(x.symbol)
        out.append({"id": f"T{n_:02d}", "symbol": x.symbol, "dir": int(x.dir), "r": round(float(x.r), 2), "rr": float(x.rr),
                    "reason": x.reason, "entry": float(x.entry), "stop": float(x.stop), "target": float(x.target),
                    "point_c": float(x.target + x.dir * (2 * pip if pip else 0)), "exit": float(x.exit),
                    "time": int(x.time.timestamp()), "exit_time": int(x.exit_time.timestamp()),
                    "leg_start": int(x.leg_start.timestamp()), "leg_top": int(x.leg_top.timestamp()),
                    "gap_lo": float(x.gap_lo), "gap_hi": float(x.gap_hi), "leg": int(x.leg),
                    "m15": [[int(ti.timestamp()), round(float(r.o), 6), round(float(r.h), 6), round(float(r.l), 6), round(float(r.c), 6)]
                            for ti, r in w.iterrows()]})
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "own15_examples.json").write_text(json.dumps(out), encoding="utf-8")
    print("examples:", len(out), "of", len(t))


if __name__ == "__main__":
    main()
