"""Orchestrator for the insider-purchase track (SEC Form 4).

  python run_insider.py --start 2016-01-01 --end 2026-10-28
  python run_insider.py --demo            (synthetic data, no keys, checks the pipeline)

Same machinery as the news backtest: point-in-time scores, smart exits, walk-forward
split, luck test, borrow/short rules (unused here: we only buy), risk simulation.
"""
import argparse

import numpy as np
import pandas as pd

import config
from agents import backtest_agent, features_agent, insider_agent, report_agent

TRACK = "insider"


def demo_data(seed=5):
    """Synthetic purchases + daily bars with a planted drift after clusters."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", "2025-12-31")
    import string
    syms = ["Q" + string.ascii_uppercase[i // 26] + string.ascii_uppercase[i % 26] for i in range(60)]
    bars, rows = {}, []
    for s in syms:
        r = rng.normal(0, 0.025, len(days))
        for _ in range(rng.integers(3, 10)):
            k = int(rng.integers(30, len(days) - 60))
            n = int(rng.choice([1, 1, 1, 2, 2, 3, 4]))
            for j in range(n):
                fd = days[k + j * 3]
                rows.append({"accession": f"{s}{k}{j}", "filing_date": fd, "trans_date": fd - pd.Timedelta(days=2),
                             "symbol": s, "issuer_cik": s, "owner": f"{s}-o{j}",
                             "shares": 10000, "price": 20.0, "owned_after": 50000,
                             "officer": j == 0, "director": j > 0, "ten_pct": False, "ceo_cfo": j == 0,
                             "value": float(rng.choice([30000, 150000, 800000]))})
            if n >= 2:
                r[k + 1:k + 21] += 0.004 * n / 2   # planted: clusters drift up for 20 days
        c = 20 * np.exp(np.cumsum(r))
        o = np.concatenate([[c[0]], c[:-1]]) * (1 + rng.normal(0, 0.003, len(c)))
        bars[s] = pd.DataFrame({"o": o, "h": np.maximum(o, c) * 1.01, "l": np.minimum(o, c) * 0.99,
                                "c": c, "v": rng.integers(200_000, 2_000_000, len(c))}, index=days)
    mkt_c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(days))))
    mkt = pd.DataFrame({"o": np.concatenate([[mkt_c[0]], mkt_c[:-1]]), "c": mkt_c}, index=days)
    return pd.DataFrame(rows), bars, {"SPY": mkt, "IWM": mkt}


def reaction_by_insiders(results):
    d = results.dropna(subset=["entry_px"]).copy()
    d["band"] = d["n_insiders"].clip(upper=4).map({1: "1", 2: "2", 3: "3", 4: "4+"})
    out = []
    for b, g in d.groupby("band"):
        row = {"band": b, "events": len(g)}
        for h in config.TRACKS[TRACK]["horizons"]:
            row[f"h{h}_avg_bps"] = g[backtest_agent.rc(h)].mean() * 1e4
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-28")
    args = ap.parse_args()
    backtest_agent.set_track(TRACK)
    tr = config.TRACKS[TRACK]

    print("[1] insider agent: SEC Form 4 purchases")
    if args.demo:
        purchases, bars, market = demo_data()
        mode, period = "מצב דמו (נתונים סינתטיים)", "2018–2025"
    else:
        if config.KEY_PROBLEMS:
            for problem in config.KEY_PROBLEMS:
                print(f"::error::{problem}")
            raise SystemExit(1)
        from agents import market_data_agent
        purchases = insider_agent.fetch_purchases(args.start, args.end)
        mode, period = "נתונים אמיתיים (SEC Form 4 + Alpaca)", f"{args.start} עד {args.end}"
    print(f"    {len(purchases):,} open-market purchases")
    events = insider_agent.build_events(purchases)
    print(f"    {len(events):,} events (stock + filing day), {events['symbol'].nunique():,} stocks")

    if not args.demo:
        symbols = sorted(events["symbol"].unique())
        pre = (pd.Timestamp(args.start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        post = (pd.Timestamp(args.end) + pd.Timedelta(days=int(max(tr["horizons"]) * 1.6) + 10)).strftime("%Y-%m-%d")
        print(f"[2] market data: daily bars for {len(symbols):,} stocks")
        bars = {}
        for i, s in enumerate(symbols, 1):
            if i % 100 == 0:
                print(f"    {i}/{len(symbols)}")
            try:
                bars[s] = market_data_agent.get_daily_bars(s, pre, post)
            except Exception as e:  # noqa: BLE001 - one bad ticker must not stop the run
                print(f"    {s}: no bars ({str(e)[:80]})")
                bars[s] = None
        market = {m: market_data_agent.get_daily_bars(m, pre, post)
                  for m in (config.MARKET_SYMBOL, config.SMALL_MARKET_SYMBOL)}
    no_bars = sorted(s for s in events["symbol"].unique() if bars.get(s) is None or bars[s].empty)

    print("[3] features + backtest")
    ev = features_agent.add_scores(events, bars, {})
    results = backtest_agent.measure_reactions_daily(ev, bars, market)
    results["year"] = pd.to_datetime(results["time"]).dt.year
    react = reaction_by_insiders(results)
    cats = backtest_agent.category_table(results, threshold=0.3)
    if not cats.empty:
        cats["category"] = cats["category"].map(insider_agent.ROLE_HE).fillna(cats["category"])
    wf = backtest_agent.walk_forward(results)
    bands, details = {}, {}
    if wf and wf.get("best"):
        for col in ("size_score", "liquidity_score", "volatility_score"):
            bands[col] = backtest_agent.band_table(results, wf["best"], col)
        details = {"לפי שנה (יציבות לאורך זמן)": backtest_agent.group_table(results, wf["best"], "year"),
                   "גידור מול": backtest_agent.group_table(results, wf["best"], "hedge")}
        if not details["לפי שנה (יציבות לאורך זמן)"].empty:
            details["לפי שנה (יציבות לאורך זמן)"] = details["לפי שנה (יציבות לאורך זמן)"].sort_values("group")

    sv = ("רשימת המניות כוללת כל חברה שבה מנהל קנה מניות בתקופה, גם כאלה שנמחקו מאז. "
          f"מניות בלי נתוני מחיר ב־Alpaca ({len(no_bars):,}, בעיקר מניות שנמחקו או קטנות מאוד) לא נכללו, "
          "ולכן התוצאות עלולות להיות מעט אופטימיות מדי.")
    checks = {"survivorship": sv}
    print("[4] report")
    path, _ = report_agent.write_report(results, wf, react, cats, mode, period, bands, checks, details=details)
    status, text = report_agent.verdict(wf)
    print(f"\n=== RESULT ===\n{TRACK} {status.upper()} {path}\n{text}")


if __name__ == "__main__":
    import sys
    from run_backtest import _explain
    try:
        main()
    except config.TimeBudgetReached:
        print("::notice::Time budget reached. Everything downloaded so far is saved; "
              "run the workflow again and it will continue from here.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"::error::{_explain(e)}")
        raise
