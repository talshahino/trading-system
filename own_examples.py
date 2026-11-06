"""Examples of the own-strategy backtest for the user to review by eye (06.11.2026): 20 trades of the
closest-to-the-rules variant (volume rule on, stop at the first swing), winners and losers, with the 15-minute
candles around them and the 1-minute candles of the gap. Writes output/own_examples.json.

  python own_examples.py --symbols EURUSD,GBPUSD,... [--n 20]
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
import own_strategy as OS
from scalp_map import comm_px

VARIANT = (1.5, True, 1)


def bars_json(df):
    return [[int(t.timestamp()), round(float(r.o), 6), round(float(r.h), 6), round(float(r.l), 6), round(float(r.c), 6)]
            for t, r in df.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    allt, data = [], {}
    ev, _ = NC.load()
    for s in [x for x in a.symbols.split(",") if x]:
        try:
            df = P.bars(s, "1Min", max(a.start, RP.MINUTE_START), end)
            sp = P.spread_1m(s, a.start, end)
        except Exception as e:  # noqa: BLE001
            print(s, "no data", e)
            continue
        if df is None or df.empty or sp is None or len(sp) == 0:
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
        tr = OS.run_symbol(s, df, cost, stop_at, noent, [VARIANT])
        allt += tr
        data[s] = df
        print(s, len(tr), "trades", flush=True)
    t = pd.DataFrame(allt)
    rng = np.random.default_rng(7)
    win, loss = t[t.r > 0], t[t.r <= 0]
    k = a.n // 2
    pick = pd.concat([win.iloc[rng.choice(len(win), min(k, len(win)), replace=False)],
                      loss.iloc[rng.choice(len(loss), min(a.n - k, len(loss)), replace=False)]])
    pick = pick.sample(frac=1, random_state=3)
    out = []
    for n_, (_, x) in enumerate(pick.iterrows(), 1):
        df = data[x.symbol]
        m15 = OS.resample(df, "15min")
        w = m15[(m15.index >= x.imp_start - pd.Timedelta(hours=18)) & (m15.index <= x.exit_time + pd.Timedelta(hours=3))]
        g = df[(df.index >= x.gap_t - pd.Timedelta(minutes=20)) & (df.index <= x.gap_t + pd.Timedelta(minutes=40))]
        out.append({"id": f"T{n_:02d}", "symbol": x.symbol, "dir": int(x.dir), "r": round(float(x.r), 2), "rr": float(x.rr),
                    "reason": x.reason, "entry": float(x.entry), "stop": float(x.stop), "target": float(x.target), "exit": float(x.exit),
                    "time": int(x.time.timestamp()), "exit_time": int(x.exit_time.timestamp()),
                    "imp_start": int(x.imp_start.timestamp()), "imp_end": int(x.imp_end.timestamp()),
                    "gap_t": int(x.gap_t.timestamp()), "gap_near": float(x.gap_near), "gap_far": float(x.gap_far),
                    "m15": bars_json(w), "m1": bars_json(g)})
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "own_examples.json").write_text(json.dumps(out), encoding="utf-8")
    print("examples:", len(out), "of", len(t), "trades")


if __name__ == "__main__":
    main()
