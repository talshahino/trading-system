"""1-minute precision entry for the bots' own strategies (user, 06.11.2026: "כן").

The bots enter at the open after a signal on their own chart (5 / 15 minutes, 1 / 4 hours). Instead: a limit
order k x R better than that price (R = entry to stop), waiting W minutes on the 1-minute chart. Filled ->
same stop price, same exit (time and price) as the real trade, so the stop is nearer, the position bigger for
the same dollar risk, and the trade's R larger; the trade's own cost (spread + commission, the same in price)
weighs more. Not filled before the window ends (or before the real exit) -> no trade.

    k  0.1, 0.2, 0.3, 0.5 R        W  5, 15, 30, 60 minutes
Input: the live-rules trades of every live combo (run 674). Output: output/precision_<tag>.csv.gz, one row per
trade with the new R of every (k, W) (empty = not filled).

  python precision_entry.py --symbols EURUSD,XAUUSD [--trades reports/2026-11-05_tv_run674/combo_trades_all.csv.gz]
"""
import argparse

import numpy as np
import pandas as pd

import config
from agents import poc_agent as P
import run_poc as RP

KS = [0.1, 0.2, 0.3, 0.5]
WS = [5, 15, 30, 60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--trades", default="reports/2026-11-05_tv_run674/combo_trades_all.csv.gz")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    T = pd.read_csv(a.trades)
    T["time"] = pd.to_datetime(T["time"], utc=True)
    T["exit_time"] = pd.to_datetime(T["exit_time"], utc=True)
    out = []
    for s in [x for x in a.symbols.split(",") if x]:
        t = T[T.symbol == s].copy()
        if t.empty:
            continue
        st = (t.time.min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        try:
            df = P.bars(s, "1Min", max(st, RP.MINUTE_START), end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-120:]}")
            continue
        if df is None or df.empty:
            print(f"{s}: NO DATA")
            continue
        ix = df.index
        o, h, l = (df[x].to_numpy(float) for x in "ohl")
        d = t["dir"].to_numpy()
        e, sl, ex = (t[x].to_numpy(float) for x in ("entry_px", "stop_px", "exit_px"))
        risk = d * (e - sl)
        gross = d * (ex - e) / risk
        cost_px = (gross - t["r"].to_numpy(float)) * risk          # the trade's own cost, in price
        i0 = ix.searchsorted(t["time"].to_numpy(), side="left")
        i_exit = ix.searchsorted(t["exit_time"].to_numpy(), side="left")
        res = {"idx": t.index.to_numpy(), "covered": (i0 < len(ix)) & (np.abs((ix[np.minimum(i0, len(ix) - 1)] - t["time"]).dt.total_seconds().to_numpy()) <= 300)}
        for k in KS:
            lim = e - d * k * risk
            for w in WS:
                col = np.full(len(t), np.nan)
                for j in range(len(t)):
                    if not res["covered"][j] or risk[j] <= 0:
                        continue
                    a0, a1 = i0[j], min(i0[j] + w, i_exit[j] + 1, len(ix))
                    if a1 <= a0:
                        continue
                    if d[j] > 0:
                        hit = np.flatnonzero(l[a0:a1] <= lim[j])
                    else:
                        hit = np.flatnonzero(h[a0:a1] >= lim[j])
                    if not len(hit):
                        continue
                    b = a0 + hit[0]
                    fill = min(lim[j], o[b]) if d[j] > 0 else max(lim[j], o[b])
                    nr = d[j] * (fill - sl[j])
                    if nr <= 0:
                        continue
                    col[j] = (d[j] * (ex[j] - fill) - cost_px[j]) / nr
                res[f"k{k:g}_w{w}"] = col
        out.append(pd.DataFrame(res))
        print(f"{s}: {len(t)} trades, covered {int(res['covered'].sum())}", flush=True)
    if out:
        tag = a.symbols.replace(",", "_").replace("/", "-")[:60]
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat(out, ignore_index=True).round(4).to_csv(config.OUTPUT_DIR / f"precision_{tag}.csv.gz", index=False)


if __name__ == "__main__":
    main()
