"""The 3 live bots under FTMO Swing vs Standard account rules, from 2025-04-30.

Swing:    positions may be held over weekends and market breaks; leverage = FTMO's leverageSwing.
Standard: every position is closed before any market break longer than 2 hours (weekends too)
          -> trades from a backtest run with --break-exit; leverage = FTMO's leverageStandard.

For each bot and account type: the $100K account replay with the bot's risk and guard, the
Trading Prophet verdict (minute by minute), profit, worst intraday drop, and margin: the margin
the open positions need at every moment (notional / leverage, FTMO's per-symbol leverage), its peak,
and how many trades the platform would have refused for lack of free margin.

  python bots_compare.py --swing 'reports/X/combo_trades_*.csv.gz' --standard 'reports/Y/combo_trades_*.csv.gz' \
      --end 2026-10-30 --out output/bots_compare.json
"""
import argparse
import json
import sys
import types
from pathlib import Path

import pandas as pd

from agents import ftmo_agent as F
from agents import ftmo_universe as U
import portfolio_breakdown as PB
import portfolio_search as PS
import run_rules as RR

F.MAX_OPEN = 99
sys.path.insert(0, str(Path(__file__).parent / "skills" / "trading-prophet" / "scripts"))
import ftmo_sim  # noqa: E402

ftmo_sim.CLOSE_ALL_AT_TARGET = True
ROOT = Path(__file__).parent


def margin_check(led, kind):
    """Margin the open positions need over time, and the trades refused for lack of free margin."""
    ex = led[led["taken"] == "בוצעה"].copy()
    if ex.empty:
        return {}
    lev = {s: float(U.spec(s)[f"leverage{kind}"]) for s in ex["symbol"].unique()}
    dist = (ex["entry_px"] - ex["stop_px"]).abs() / ex["entry_px"]
    ex["notional"] = ex["risk_usd"] / dist.clip(lower=1e-6)
    ex["margin"] = ex["notional"] / ex["symbol"].map(lev)
    ex["bal"] = ex["balance_after"].ffill().fillna(F.INITIAL)
    ev = sorted([(t, 1, i) for i, t in ex["time"].items()] + [(t, 0, i) for i, t in ex["exit_time"].items()])
    used, peak, peak_t, refused, open_ids = 0.0, 0.0, None, [], set()
    bal = F.INITIAL
    for t, kind_ev, i in ev:
        if kind_ev == 0:
            if i in open_ids:
                open_ids.discard(i)
                used -= ex.at[i, "margin"]
                b = ex.at[i, "balance_after"]
                bal = b if b == b else bal
            continue
        need = ex.at[i, "margin"]
        if used + need > bal:              # free margin (equity - used) smaller than this order needs
            refused.append({"combo": ex.at[i, "combo"], "time": str(t), "need": round(need), "used": round(used)})
        if ex.at[i, "exit_time"] <= t:     # opened and closed within the same moment
            continue
        open_ids.add(i)                    # counted anyway, as in the backtest, to see how far over it goes
        used += need
        if used > peak:
            peak, peak_t = used, t
    by_class = ex.groupby(ex["symbol"].map(U.CLASS))["margin"].median().round().to_dict()
    return {"peak_margin": round(peak), "peak_when": str(peak_t), "peak_pct": round(peak / F.INITIAL * 100, 1),
            "refused": len(refused), "refused_examples": refused[:8],
            "median_margin_per_trade_by_class": by_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swing", required=True)
    ap.add_argument("--standard", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--start", default="2025-04-30")
    ap.add_argument("--out", required=True)
    ap.add_argument("--risk", default="", help="override a bot's risk, e.g. 3:0.0025")
    ap.add_argument("--bots", default="1,2,3")
    ap.add_argument("--news", action="store_true",
                    help="apply FTMO restricted events minute by minute to the Standard trades (ftmo_calendar.apply_news)")
    ap.add_argument("--graded", action="store_true", help="risk shrinks near the bot's limits instead of skipping")
    ap.add_argument("--guard", default="", help="bot limits daily,total, e.g. 0.045,0.09")
    ap.add_argument("--add", default="", help="file of candidate combos: each bot is also run with all of them added "
                                               "and with each one added alone (variant column)")
    args = ap.parse_args()
    if args.graded:
        F.GRADED = True
    if args.guard:
        F.GUARD_DAILY, F.GUARD_TOTAL = (float(x) for x in args.guard.split(","))
    start = pd.Timestamp(args.start, tz="UTC")
    RR.SPLIT = start
    cands = [l.strip() for l in open(args.add, encoding="utf-8") if l.strip()] if args.add else []
    variants = [("bot", [])] + ([("+all", cands)] + [("+" + c, [c]) for c in cands] if cands else [])
    ns = types.SimpleNamespace(demo=False, end=args.end)
    trades = {"Swing": PS.load(args.swing), "Standard": PS.load(args.standard)} if args.swing != "none" else \
        {"Standard": PS.load(args.standard)}
    if args.news:
        from agents import ftmo_calendar as CAL
        restricted, _ = CAL.parse(json.load(open(ROOT / "data" / "ftmo_calendar" / "history.json", encoding="utf-8")))

        def minute_open(sym, t):
            if sym not in PB._BARS:
                try:
                    PB._BARS[sym] = PB.P.bars(sym, "1Min", (start - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), args.end)
                except Exception as e:  # noqa: BLE001
                    print("no bars", sym, e)
                    PB._BARS[sym] = None
            b = PB._BARS[sym]
            if b is None or b.empty:
                return None
            i = b.index.searchsorted(t)
            if i >= len(b) or b.index[i] - t > pd.Timedelta(minutes=30):
                return None
            return float(b["o"].iloc[i])
        want = {c for b in (1, 2, 3) for c in json.load(open(ROOT / "live" / "bots" / f"bot{b}.json", encoding="utf-8"))["combos"]}
        want |= set(cands)
        st = trades["Standard"]
        st = st[st["combo"].isin(want)]
        before = len(st)
        st = CAL.apply_news(st, minute_open, restricted)
        for c in ("time", "exit_time"):
            st[c] = pd.to_datetime(st[c], utc=True)
        st["free_time"] = pd.to_datetime(st["free_time"], utc=True, errors="coerce")
        print(f"news rule: {before} trades -> {len(st)}; closed before an event: {(st['reason'] == 'נסגרה לפני הודעה מוגבלת').sum()}, "
              f"signals in the window, not taken: {before - len(st)}")
        orig = trades["Standard"].set_index(["combo", "time"])["r"]
        st = st.reset_index(drop=True)
        chg = st.join(orig.rename("r_before"), on=["combo", "time"])
        chg = chg[(chg["r"] - chg["r_before"]).abs() > 1e-9]
        print("changed trades:", len(chg), "sum R before", round(chg["r_before"].sum(), 1), "after", round(chg["r"].sum(), 1),
              "| NaN r:", int(st["r"].isna().sum()))
        print(chg.sort_values("r").head(15)[["combo", "time", "exit_time", "dir", "entry_px", "stop_px", "exit_px", "r_before", "r"]].to_string())
        st.to_csv(Path(args.out).parent / "news_trades.csv.gz", index=False)
        trades["Standard"] = st
    out = []
    over = {int(k): float(v) for k, v in (x.split(":") for x in args.risk.split(",") if x)}
    for b in (int(x) for x in args.bots.split(",")):
        cfg = json.load(open(ROOT / "live" / "bots" / f"bot{b}.json", encoding="utf-8"))
        cfg["risk"] = over.get(b, cfg["risk"])
        F.FREE_RISK_AT_BREAKEVEN = bool(cfg["free_risk_at_breakeven"])
        for (variant, extra), (kind, t) in ((v, k) for v in variants for k in trades.items()):
            combos = list(cfg["combos"]) + [c for c in extra if c not in cfg["combos"]]
            tr = t[t["combo"].isin(combos)].sort_values("time").reset_index(drop=True)
            w = F.account_path(tr, cfg["risk"], start=start)
            led = w[0].attrs["ledger"]
            pr = RR.prophet_check(led, ns)
            df, quality = PB.series(led, start, args.end)
            weeks = PB.breakdown(df, start)
            ex = led[led["taken"] == "בוצעה"]
            roll = (pr or {}).get("rolling") or {}
            row = {"bot": b, "variant": variant, "account": kind, "risk": cfg["risk"], "combos_with_trades": int(tr["combo"].nunique()),
                   "trades": int(len(ex)), "skipped_by_guard": int((led["taken"] != "בוצעה").sum()),
                   "end_balance": round(float(df["balance"].iloc[-1])) if len(df) else None,
                   "blown": str(w[1]) if w[1] is not None else None,
                   "worst_intraday_pct": min(x["worst_intraday_pct"] for x in weeks) if weeks else None,
                   "lowest_equity": min(x["low_equity"] for x in weeks) if weeks else None,
                   "losing_weeks": sum(x["pnl"] < 0 for x in weeks), "weeks": len(weeks),
                   "prophet": (pr or {}).get("verdict", {}).get("verdict"),
                   "prophet_reasons": (pr or {}).get("verdict", {}).get("reasons"),
                   "pass_rate_pct": roll.get("pass_rate_pct"), "breach_rate_pct": roll.get("breach_rate_pct"),
                   "samples": roll.get("completed"),
                   "exit_reasons": ex["reason"].value_counts().head(8).to_dict(),
                   "margin": margin_check(led, kind), "quality": quality}
            out.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("margin", "exit_reasons")}, ensure_ascii=False))
            print("  margin:", json.dumps(row["margin"], ensure_ascii=False, default=str)[:600])
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
