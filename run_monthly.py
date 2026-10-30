"""Month-by-month result of each surviving strategy in its $100,000 FTMO account, with the
highest and lowest FLOATING equity of every month (open trades marked to market minute by
minute on 1-minute bars, the way FTMO measures equity).

  python run_monthly.py [--demo]      -> output/monthly_<id>.csv + output/monthly_all.csv
"""
import argparse

import numpy as np
import pandas as pd

import config
from agents import ftmo_agent as F
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
from run_sltp import variant

SYMBOLS = RP.FX + RP.COMMOD
# (label, strategy, timeframe, stop multiplier, target)
COMBOS = [("TV01 Gold Pro 4H", "TV01", "4Hour", 1.0, "orig"),
          ("TV01 Gold Pro 4H סטופ x0.75", "TV01", "4Hour", 0.75, "orig"),
          ("TV02 AsliGold 4H", "TV02", "4Hour", 1.0, "orig"),
          ("TV04 Golden Edge 4H", "TV04", "4Hour", 1.0, "orig"),
          ("TV04 Golden Edge 4H סטופ x1.25 יעד 3R", "TV04", "4Hour", 1.25, 3.0),
          ("TV09 RSI(2) 1H", "TV09", "1Hour", 1.0, "orig"),
          ("TV09 RSI(2) 4H", "TV09", "4Hour", 1.0, "orig")]
BAR = {"1Hour": pd.Timedelta(hours=1), "4Hour": pd.Timedelta(hours=4)}


def ledger_for(sid, tf, sm, tg, frames, events, split):
    parts = []
    for s, df in frames[tf].items():
        t = R.run(df, variant(R.STRATEGIES[sid][1], sm, tg), RP.cost_of(s))
        if t.empty:
            continue
        if events is not None:
            bad = NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events)
            t = t[~bad]
        parts.append(t.assign(symbol=s))
    tr = pd.concat(parts, ignore_index=True).sort_values("time")
    after = tr[tr["time"] >= split]
    w, blown, _ = F.account_path(after, RP.MAIN_RISK, start=split)
    led = w.attrs["ledger"]
    return led[led["taken"] == "בוצעה"].copy(), blown


def floating(led, minutes, tf, blown, split, end):
    """Minute equity: closed balance + open trades at their worst (low) / best (high) price."""
    idx = pd.date_range(split.floor("min"), end, freq="1min", tz="UTC")
    lo_u, hi_u = np.zeros(len(idx)), np.zeros(len(idx))
    realized = np.zeros(len(idx))
    for t in led.itertuples():
        m = minutes.get(t.symbol)
        if m is None:
            continue
        units = t.risk_usd / abs(t.entry_px - t.stop_px)
        seg = m.loc[t.time: t.exit_time + BAR[tf] - pd.Timedelta(minutes=1)]
        if len(seg):
            last = seg.loc[t.exit_time:]
            hit = last[(last["l"] <= t.exit_px) & (last["h"] >= t.exit_px)]
            stop_at = hit.index[0] if len(hit) else seg.index[-1]
            seg = seg.loc[:stop_at]
            adverse = (seg["l"] if t.dir > 0 else seg["h"]).to_numpy()
            fav = (seg["h"] if t.dir > 0 else seg["l"]).to_numpy()
            a = t.dir * (adverse - t.entry_px) * units
            b = t.dir * (fav - t.entry_px) * units
            close_at = stop_at
        else:
            close_at = t.exit_time
        # hold the last value across minutes without bars (weekend / gaps) until the close
        i0 = idx.searchsorted(t.time)
        i1 = idx.searchsorted(close_at)
        if len(seg) and i1 > i0:
            pos = idx[i0:i1]
            sa = pd.Series(a, index=seg.index).reindex(pos, method="ffill").fillna(0).to_numpy()
            sb = pd.Series(b, index=seg.index).reindex(pos, method="ffill").fillna(0).to_numpy()
            lo_u[i0:i1] += sa
            hi_u[i0:i1] += sb
        if i1 < len(idx):
            realized[i1] += t.pnl_usd
    bal = F.INITIAL + np.cumsum(realized)
    eq = pd.DataFrame({"balance": bal, "low": bal + lo_u, "high": bal + hi_u}, index=idx)
    if blown is not None:
        eq = eq.loc[:pd.Timestamp(blown)]
    return eq


def monthly(eq, led, label, blown):
    rows = []
    il = eq.tz_convert("Asia/Jerusalem")
    prev = F.INITIAL
    for per, g in il.groupby(il.index.to_period("M")):
        n = int(((led["exit_time"].dt.tz_convert("Asia/Jerusalem").dt.to_period("M")) == per).sum())
        end_bal = float(g["balance"].iloc[-1])
        rows.append({"אסטרטגיה": label, "חודש": per.strftime("%m/%Y"), "יתרה בתחילת החודש $": round(prev),
                     "יתרה בסוף החודש $": round(end_bal), "רווח/הפסד בחודש $": round(end_bal - prev),
                     "רווח/הפסד בחודש %": round((end_bal - prev) / F.INITIAL * 100, 2),
                     "הכי גבוה (צף) $": round(float(g["high"].max())), "הכי נמוך (צף) $": round(float(g["low"].min())),
                     "עסקאות שנסגרו": n,
                     "מצב": "נשרף" if blown is not None and pd.Timestamp(blown).tz_convert("Asia/Jerusalem").to_period("M") == per else ""})
        prev = end_bal
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-10-29")
    args = ap.parse_args()
    events = NC.load()[0]
    split = RP.SPLIT
    frames = {tf: {} for tf in BAR}
    minutes = {}
    for s in SYMBOLS:
        for tf in BAR:
            try:
                frames[tf][s] = P.bars(s, tf, args.start, args.end)
            except Exception as e:  # noqa: BLE001
                print(f"  {s} {tf}: NO DATA ({e})")
        try:
            m = P.bars(s, "1Min", (split - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), args.end)
            minutes[s] = m[["h", "l"]].astype("float32")
        except Exception as e:  # noqa: BLE001
            print(f"  {s} 1Min: NO DATA ({e})")
        print(" ", s, "loaded")
    end = max(m.index.max() for m in minutes.values())
    out = []
    for label, sid, tf, sm, tg in COMBOS:
        led, blown = ledger_for(sid, tf, sm, tg, frames, events, split)
        eq = floating(led, minutes, tf, blown, split, end)
        mo = monthly(eq, led, label, blown)
        out.append(mo)
        print(label, "final", mo["יתרה בסוף החודש $"].iloc[-1], "low", mo["הכי נמוך (צף) $"].min())
    pd.concat(out).to_csv(config.OUTPUT_DIR / "monthly_all.csv", index=False, encoding="utf-8-sig")
    print("written")


if __name__ == "__main__":
    main()
