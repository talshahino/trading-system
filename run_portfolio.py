"""ONE $100,000 FTMO account opened on 1.1.2026 that runs several strategies together.
Every combination of the candidate strategies is tested: shared loss limits (daily 5%, total
10%), the shared risk rule (no new trade if today's loss + all open stops + the new stop could
pass 90% of the daily limit), 1% risk per trade, news filter, and a minute-by-minute check of
FLOATING equity against the daily limit (from midnight Prague time) and the total limit.

  python run_portfolio.py   -> output/portfolio_monthly.csv, output/portfolio_summary.csv
"""
import argparse
import itertools

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
START = pd.Timestamp("2026-01-01", tz="Asia/Jerusalem").tz_convert("UTC")
CANDIDATES = {"TV01 Gold Pro (סטופ x0.75)": ("TV01", "4Hour", 0.75, "orig"),
              "TV02 AsliGold": ("TV02", "4Hour", 1.0, "orig"),
              "TV04 Golden Edge": ("TV04", "4Hour", 1.0, "orig"),
              "TV09 RSI(2) 4H": ("TV09", "4Hour", 1.0, "orig"),
              "TV09 RSI(2) 1H": ("TV09", "1Hour", 1.0, "orig")}
BAR = {"1Hour": pd.Timedelta(hours=1), "4Hour": pd.Timedelta(hours=4)}


def trades_for(sid, tf, sm, tg, frames, events):
    parts = []
    for s, df in frames[tf].items():
        t = R.run(df, variant(R.STRATEGIES[sid][1], sm, tg), RP.cost_of(s))
        if t.empty:
            continue
        bad = NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events)
        parts.append(t[~bad].assign(symbol=s, tf=tf))
    if not parts:
        return pd.DataFrame()
    tr = pd.concat(parts, ignore_index=True)
    return tr[tr["time"] >= START]


def floating(led, minutes, end):
    idx = pd.date_range(START.floor("min"), end, freq="1min", tz="UTC")
    lo_u, hi_u, realized = np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx))
    for t in led.itertuples():
        m = minutes.get(t.symbol)
        units = t.risk_usd / abs(t.entry_px - t.stop_px)
        close_at = t.exit_time
        if m is not None:
            seg = m.loc[t.time: t.exit_time + BAR[t.tf] - pd.Timedelta(minutes=1)]
            if len(seg):
                last = seg.loc[t.exit_time:]
                hit = last[(last["l"] <= t.exit_px) & (last["h"] >= t.exit_px)]
                close_at = hit.index[0] if len(hit) else seg.index[-1]
                seg = seg.loc[:close_at]
                a = t.dir * ((seg["l"] if t.dir > 0 else seg["h"]).to_numpy() - t.entry_px) * units
                b = t.dir * ((seg["h"] if t.dir > 0 else seg["l"]).to_numpy() - t.entry_px) * units
                i0, i1 = idx.searchsorted(t.time), idx.searchsorted(close_at)
                if i1 > i0:
                    pos = idx[i0:i1]
                    lo_u[i0:i1] += pd.Series(a, index=seg.index).reindex(pos, method="ffill").fillna(0).to_numpy()
                    hi_u[i0:i1] += pd.Series(b, index=seg.index).reindex(pos, method="ffill").fillna(0).to_numpy()
        i1 = idx.searchsorted(close_at)
        if i1 < len(idx):
            realized[i1] += t.pnl_usd
    bal = F.INITIAL + np.cumsum(realized)
    return pd.DataFrame({"balance": bal, "low": bal + lo_u, "high": bal + hi_u}, index=idx)


def breach(eq):
    """First minute where floating equity breaks the daily (from Prague midnight) or total limit."""
    prague = eq.index.tz_convert("Europe/Prague")
    day_start = eq["balance"].groupby(prague.date).transform("first")
    bad = (eq["low"] < day_start - F.INITIAL * 0.05) | (eq["low"] <= F.INITIAL * 0.90)
    worst_day = float((eq["low"] - day_start).min())
    return (eq.index[bad.to_numpy()][0] if bad.any() else None), worst_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-10-29")
    args = ap.parse_args()
    events = NC.load()[0]
    frames, minutes = {tf: {} for tf in BAR}, {}
    for s in SYMBOLS:
        for tf in BAR:
            try:
                frames[tf][s] = P.bars(s, tf, args.start, args.end)
            except Exception as e:  # noqa: BLE001
                print(f"  {s} {tf}: NO DATA ({e})")
        try:
            minutes[s] = P.bars(s, "1Min", "2025-12-25", args.end)[["h", "l"]].astype("float32")
        except Exception as e:  # noqa: BLE001
            print(f"  {s} 1Min: NO DATA ({e})")
        print(" ", s, "loaded")
    end = max(m.index.max() for m in minutes.values())
    pool = {k: trades_for(*v, frames, events).assign(strategy=k) for k, v in CANDIDATES.items()}
    F.MAX_OPEN = 99            # the shared risk rule decides how many trades can be open together
    months, summary = [], []
    names = [k for k in CANDIDATES if len(pool[k])]
    for n in range(1, len(names) + 1):
        for combo in itertools.combinations(names, n):
            if "TV09 RSI(2) 4H" in combo and "TV09 RSI(2) 1H" in combo:
                continue
            label = " + ".join(c.split(" ")[0] + (" 1H" if c.endswith("1H") else "") for c in combo)
            tr = pd.concat([pool[c] for c in combo], ignore_index=True).sort_values("time").reset_index(drop=True)
            w, blown, _ = F.account_path(tr, RP.MAIN_RISK, start=START)
            led = w.attrs["ledger"]
            led = led[led["taken"] == "בוצעה"]
            eq = floating(led, minutes, end)
            hit, worst_day = breach(eq)
            stop = min([x for x in (blown, hit) if x is not None], default=None)
            if stop is not None:
                eq = eq.loc[:pd.Timestamp(stop)]
            il = eq.tz_convert("Asia/Jerusalem")
            prev = F.INITIAL
            for per, g in il.groupby(il.index.tz_localize(None).to_period("M")):
                e = float(g["balance"].iloc[-1])
                months.append({"שילוב": label, "חודש": per.strftime("%m/%Y"), "יתרה בסוף החודש $": round(e),
                               "רווח/הפסד בחודש $": round(e - prev), "הכי גבוה (צף) $": round(float(g["high"].max())),
                               "הכי נמוך (צף) $": round(float(g["low"].min()))})
                prev = e
            summary.append({"שילוב": label, "אסטרטגיות": len(combo), "יתרה סופית $": round(float(eq["balance"].iloc[-1])),
                            "שיא (צף) $": round(float(eq["high"].max())), "שפל (צף) $": round(float(eq["low"].min())),
                            "יום הכי גרוע (צף) $": round(worst_day), "עסקאות": len(led),
                            "דולגו בגלל הסיכון המשותף": int((w.attrs["ledger"]["taken"] == "דולג: שומר מגבלות ההפסד").sum()),
                            "נשרף": "" if stop is None else pd.Timestamp(stop).tz_convert("Asia/Jerusalem").strftime("%d.%m.%Y")})
            print(summary[-1])
    pd.DataFrame(months).to_csv(config.OUTPUT_DIR / "portfolio_monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(config.OUTPUT_DIR / "portfolio_summary.csv", index=False, encoding="utf-8-sig")
    print("written")


if __name__ == "__main__":
    main()
