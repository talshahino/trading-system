"""Month -> week profit/loss of each portfolio in one $100K account, with the lowest floating
equity of every week (minute by minute, from 1-minute bars) and the worst intraday drop vs the
day's starting balance (what FTMO's 5% daily rule measures).

  python portfolio_breakdown.py --trades '<globs>' --portfolios data/portfolio/portfolios_B_2026-10-31.json \
      --configs 1:0.005:1,2:0.006:1,3:0.005:0 --end 2026-10-30 --out output/breakdown.json
configs: rank:risk:free-risk-at-breakeven(1/0)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from agents import ftmo_agent as F
from agents import poc_agent as P
import portfolio_search as PS

sys.path.insert(0, str(Path(__file__).parent / "skills" / "trading-prophet" / "scripts"))
from equity import build_from_trades  # noqa: E402
from ledger_adapter import bars_from_frames, ledger_to_trades  # noqa: E402

F.MAX_OPEN = 99
TZ = "Europe/Prague"
_BARS = {}


def series(led, start, end):
    frames = {}
    for s in led["symbol"].unique():
        if s not in _BARS:
            try:
                _BARS[s] = P.bars(s, "1Min", (start - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), end)
            except Exception as e:  # noqa: BLE001
                print("no bars", s, e)
                _BARS[s] = None
        if _BARS[s] is not None:
            frames[s] = _BARS[s][_BARS[s].index >= start]
    ex = led[led["taken"] == "בוצעה"]
    es = build_from_trades(ledger_to_trades(ex), F.INITIAL, bars_from_frames(frames))
    df = pd.DataFrame([(p.ts, p.balance, p.equity_low, p.open_count) for p in es.points],
                      columns=["ts", "balance", "eq_low", "open"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index(), es.quality


def breakdown(df, start):
    loc = df.index.tz_convert(TZ)
    df = df.assign(day=loc.normalize().tz_localize(None), week=(loc.normalize() - pd.to_timedelta(loc.weekday, unit="D")).tz_localize(None))
    # balance at the start of each day / week = last balance before it
    prev = df["balance"].shift(1).fillna(F.INITIAL)
    df["day_start_bal"] = prev.groupby(df["day"]).transform("first")
    df["week_start_bal"] = prev.groupby(df["week"]).transform("first")
    df["intraday_dd"] = df["eq_low"] - df["day_start_bal"]
    weeks = []
    for wk, g in df.groupby("week"):
        wsb, web = float(g["week_start_bal"].iloc[0]), float(g["balance"].iloc[-1])
        low_i = g["eq_low"].idxmin()
        dd_i = g["intraday_dd"].idxmin()
        weeks.append({"week": wk.strftime("%Y-%m-%d"), "month": wk.strftime("%Y-%m"),
                      "start_bal": round(wsb), "end_bal": round(web), "pnl": round(web - wsb),
                      "pnl_pct": round((web - wsb) / F.INITIAL * 100, 2),
                      "low_equity": round(float(g["eq_low"].min())),
                      "low_vs_week_start_pct": round((float(g["eq_low"].min()) - wsb) / F.INITIAL * 100, 2),
                      "low_when": low_i.tz_convert("Asia/Jerusalem").strftime("%d.%m %H:%M"),
                      "worst_intraday_pct": round(float(g["intraday_dd"].min()) / F.INITIAL * 100, 2),
                      "worst_intraday_when": dd_i.tz_convert("Asia/Jerusalem").strftime("%d.%m %H:%M"),
                      "max_open": int(g["open"].max())})
    return weeks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--portfolios", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--start", default="2025-04-30")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    t = PS.load(args.trades)
    ports = {p["rank"]: p for p in json.load(open(args.portfolios, encoding="utf-8"))}
    out = []
    for cfg in args.configs.split(","):
        rank, risk, free = cfg.split(":")
        p = ports[int(rank)]
        F.FREE_RISK_AT_BREAKEVEN = free == "1"
        tr = t[t["combo"].isin([c["combo"] for c in p["combos"]])].sort_values("time").reset_index(drop=True)
        w = F.account_path(tr, float(risk), start=start)
        led = w[0].attrs["ledger"]
        df, quality = series(led, start, args.end)
        weeks = breakdown(df, start)
        out.append({"rank": int(rank), "risk": float(risk), "free": free == "1", "combos": len(p["combos"]),
                    "blown": str(w[1]) if w[1] is not None else None, "quality": quality,
                    "trades": int((led["taken"] == "בוצעה").sum()), "skipped": int((led["taken"] != "בוצעה").sum()),
                    "weeks": weeks})
        print(rank, risk, free, "weeks", len(weeks), "end", weeks[-1]["end_bal"] if weeks else None)
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
