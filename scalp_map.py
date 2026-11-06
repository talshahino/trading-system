"""Scalping bot, step 1 (user, 06.11.2026): where and when can a 1-minute scalp work at all?

For every instrument and every hour of the day (UTC), with the REAL cost of every minute (Dukascopy ask - bid
at that minute + FTMO commission; never an average - an average made spread blow-outs look like edges):
  cost      median round-trip cost
  move      median absolute move over 15 / 30 / 60 minutes, in cost units (how much room a trade has)
  direction does the last L minutes' move carry on over the next H minutes? For a "strong" move
            (|move over L| above that hour's 75th percentile on TRAIN), the signed forward move from the next
            open, in units of that minute's cost:  follow = gross - 1,  fade = -gross - 1  (net of the cost)
            L = 5, 15, 60;  H = 15, 30, 60;  samples at least H minutes apart
Periods: TRAIN 2024-01-01 .. 2025-04-30, VALID from 2025-04-30.

  python scalp_map.py --symbols EURUSD,XAUUSD
Writes output/scalp_map_<tag>.json.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import config
from agents import ftmo_universe as U
from agents import poc_agent as P
import run_poc as RP
import run_rules as RR

TRAIN_FROM, VALID_FROM = pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-04-30", tz="UTC")
LS, HS = [5, 15, 60], [15, 30, 60]


def comm_px(s, px):
    spec = U.spec(s)
    if spec["commissionType"] == "flat_USD":
        return px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
    return px * 2 * float(spec["commission"]) / 100.0


def thin(idx, gap):
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= gap:
            out.append(i)
            last = i
    return np.array(out, dtype=int)


def st(x):
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": int(len(x))}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 3), "t": round(float(x.mean() / x.std() * np.sqrt(len(x))), 2) if x.std() > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tag = a.symbols.replace(",", "_").replace("/", "-")[:60]
    res = []
    for s in [x for x in a.symbols.split(",") if x]:
        t0 = time.time()
        try:
            df = P.bars(s, "1Min", max(a.start, RP.MINUTE_START), end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-120:]}")
            continue
        if df is None or df.empty or s not in U.ALL:
            print(f"{s}: skipped")
            continue
        try:
            sp = P.spread_1m(s, a.start, end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: no spread history ({str(e)[-80:]})")
            sp = pd.Series(dtype="float64")
        if sp is None or len(sp) == 0:
            print(f"{s}: no real spread - skipped (an estimate is what misled before)")
            continue
        o, c = df["o"].to_numpy(float), df["c"].to_numpy(float)
        spr = sp.sort_index().reindex(df.index, method="ffill").to_numpy(float)
        cost = np.abs(spr) + comm_px(s, c)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, np.nan)
        n = len(c)
        # no sample across a market break / session end / news: the forward window must not contain one
        stop = RR.break_mask(df, "1Min")
        sm = RR.session_mask(df, s, "1Min")
        if sm is not None:
            stop = stop | np.asarray(sm, bool)
        nm, noent = RR.news_masks(df, s, "1Min")
        stop = stop | np.asarray(nm, bool)
        cs = np.r_[0, np.cumsum(stop)]
        hour = df.index.hour.to_numpy()
        tm = df.index
        trn = (tm >= TRAIN_FROM) & (tm < VALID_FROM)
        val = tm >= VALID_FROM
        row = {"symbol": s, "hours": {}}
        for hr in range(24):
            hm = hour == hr
            if hm.sum() < 2000:
                continue
            cell = {}
            for per, pm in (("train", trn), ("valid", val)):
                m = hm & pm
                if m.sum() < 1000:
                    continue
                cc = cost[m]
                cell[per] = {"cost_pct": round(float(np.nanmedian(cc / c[m]) * 100), 5)}
                for hh in HS:
                    j = np.flatnonzero(m)
                    j = j[j + hh < n]
                    mv = np.abs(c[j + hh] - c[j]) / cost[j]
                    cell[per][f"move{hh}"] = round(float(np.nanmedian(mv)), 2)
            for L in LS:
                past = np.r_[np.full(L, np.nan), c[L:] - c[:-L]]
                thr = np.nanquantile(np.abs(past[hm & trn]), 0.75) if (hm & trn).sum() > 1000 else np.nan
                if not np.isfinite(thr) or thr <= 0:
                    continue
                sig = hm & (np.abs(past) >= thr) & ~np.r_[np.asarray(noent, bool)[1:], False]
                for hh in HS:
                    idx = np.flatnonzero(sig)
                    idx = idx[idx + 1 + hh < n]
                    # forward window [i+1, i+hh] free of forced exits
                    ok = (cs[idx + 1 + hh] - cs[idx + 1]) == 0
                    idx = thin(idx[ok], hh)
                    if not len(idx):
                        continue
                    d = np.sign(past[idx])
                    e = o[idx + 1]
                    g = d * (c[idx + hh] - e) / cost[idx + 1]
                    for per, pm in (("train", trn), ("valid", val)):
                        k = pm[idx]
                        cell.setdefault(per, {})[f"L{L}_H{hh}"] = {"follow": st(g[k] - 1), "fade": st(-g[k] - 1)}
            row["hours"][hr] = cell
        res.append(row)
        print(f"{s}: {n:,} bars, {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / f"scalp_map_{tag}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("symbols:", len(res))


if __name__ == "__main__":
    main()
