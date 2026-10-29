"""Orchestrator: news agent -> market-data agent -> classifier agent -> backtest agent -> report.

Examples:
  python run_backtest.py --demo                    (all three tracks, synthetic data)
  python run_backtest.py --universe all --start 2025-10-01 --end 2026-10-27
  python run_backtest.py --track swing --universe all --start 2025-10-01 --end 2026-10-27
  python run_backtest.py --symbols AAPL,TSLA,NVDA,AMD,PLTR --start 2025-01-01 --end 2026-09-30
  python run_backtest.py --universe universe.txt --start 2025-06-01 --end 2026-09-30
"""
import argparse
import sys

import config
import pandas as pd

from agents import backtest_agent, classifier_agent, features_agent, report_agent


def read_universe(path):
    """Lines like 'AAPL' or 'AAPL,3500000000000' (optional market cap in USD)."""
    symbols, caps = [], {}
    for line in open(path, encoding="utf-8"):
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts or parts[0].lower() == "symbol":
            continue
        sym = parts[0].upper()
        symbols.append(sym)
        if len(parts) > 1:
            try:
                caps[sym] = float(parts[1])
            except ValueError:
                pass
    return symbols, caps


def _minute_to_daily(bars):
    ny = bars.index.tz_convert("America/New_York")
    d = bars.set_index(ny)
    out = d.groupby(d.index.date).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                                      c=("c", "last"), v=("v", "sum"))
    out.index = pd.to_datetime(out.index)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, no keys needed")
    ap.add_argument("--track", default="all", choices=list(config.TRACKS) + ["all"],
                    help="intraday = minutes, swing = up to a week, position = two weeks")
    ap.add_argument("--symbols", help="comma separated tickers")
    ap.add_argument("--universe", help="text file with one ticker per line, or 'all' to build the "
                    "list from every ticker that had news in the period (includes later-delisted ones)")
    ap.add_argument("--min-news", type=int, default=5, help="with --universe all: min news items per ticker")
    ap.add_argument("--max-symbols", type=int, default=300, help="with --universe all: cap on tickers")
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-09-30")
    args = ap.parse_args()
    tracks = list(config.TRACKS) if args.track == "all" else [args.track]

    # ---- 1. News (shared by all tracks) ----
    if args.demo:
        from agents.demo_data import make_demo
        print("[1] generating synthetic demo data")
        universe, news, minute_bars, caps = make_demo()
        events = news.explode("symbols").rename(columns={"symbols": "symbol"}).reset_index(drop=True)
        mode, period = "מצב דמו (נתונים סינתטיים)", "90 ימי מסחר"
        checks = {"survivorship": "לא רלוונטי בנתונים סינתטיים."}
    else:
        if config.KEY_PROBLEMS:
            for problem in config.KEY_PROBLEMS:
                print(f"::error::{problem}")
            sys.exit(1)
        from agents import market_data_agent, news_agent
        caps = {}
        if args.universe == "all":
            print(f"[1] news agent: ALL news {args.start} → {args.end} (this can take a while the first time)")
            news = news_agent.fetch_news(None, args.start, args.end)
            universe = news_agent.universe_from_news(news, args.min_news, args.max_symbols)
            print(f"    universe built from the news itself: {len(universe)} tickers")
        else:
            if args.universe:
                universe, caps = read_universe(args.universe)
            elif args.symbols:
                universe = [s.strip().upper() for s in args.symbols.split(",")]
            else:
                sys.exit("Give --symbols, --universe or --demo")
            print(f"[1] news agent: {len(universe)} symbols, {args.start} → {args.end}")
            news = news_agent.fetch_news(universe, args.start, args.end)
        events = news_agent.explode_events(news, universe)
        print(f"    {len(news)} news items → {len(events)} stock-specific events")
        if args.universe == "all":
            sv = "רשימת המניות נבנתה מהחדשות של התקופה עצמה, ולכן כוללת גם חברות שנמחקו מאז."
        else:
            sv = ("<b>אזהרה:</b> רשימת המניות נקבעה מראש לפי מה שמוכר היום, ולכן חסרות בה חברות שקרסו. "
                  "התוצאות עלולות להיראות טובות מדי. להרצה נקייה: <code>--universe all</code>.")
        inactive = news_agent.inactive_symbols(universe)
        if inactive is not None:
            sv += f" מניות במדגם שכבר לא נסחרות היום: {len(inactive)}."
        checks = {"survivorship": sv}
        mode = "נתונים אמיתיים (Alpaca/Benzinga)" + (" + Claude" if config.ANTHROPIC_KEY else " + מילות מפתח")
        period = f"{args.start} עד {args.end}"

    # ---- 2. Classification (shared, paid once) ----
    print("[2] classifier agent")
    events = classifier_agent.classify_events(events)
    checks["cost"] = classifier_agent.cost_summary(events)
    symbols = sorted(set(events["symbol"]))
    pre = (pd.Timestamp(args.start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    # Multi-day tracks need prices up to two weeks after the last news item.
    post = (pd.Timestamp(args.end) + pd.Timedelta(days=21)).strftime("%Y-%m-%d")

    summary = []
    for name in tracks:
        backtest_agent.set_track(name)
        tr = config.TRACKS[name]
        print(f"\n[3] track '{name}' ({tr['unit']}): market data")
        if args.demo:
            if tr["unit"] == "m":
                bars, market = minute_bars, None
            else:
                bars = {s: _minute_to_daily(b) for s, b in minute_bars.items()}
                # Synthetic "market" = equal-weight average of the demo stocks.
                rel = pd.concat([b["c"] / b["c"].iloc[0] for b in bars.values()], axis=1).mean(axis=1)
                market = pd.DataFrame({"o": rel.shift(1).fillna(rel.iloc[0]), "c": rel})
        else:
            bars, market = {}, None
            for i, s in enumerate(symbols, 1):
                print(f"    {i}/{len(symbols)} {s}", end="\r")
                if tr["unit"] == "m":
                    bars[s] = market_data_agent.get_bars(s, pre, args.end)
                else:
                    bars[s] = market_data_agent.get_daily_bars(s, pre, post)
            print()
            if tr["unit"] == "d":
                market = market_data_agent.get_daily_bars(config.MARKET_SYMBOL, pre, post)
            no_bars = sorted(s for s, b in bars.items() if b is None or b.empty)
            if no_bars:
                checks["survivorship_" + name] = no_bars

        tr_checks = dict(checks)
        nb = checks.get("survivorship_" + name)
        if nb:
            tr_checks["survivorship"] = (checks["survivorship"] +
                f" מניות בלי נתוני מחיר ({len(nb)}) לא נכללו: {', '.join(nb[:15])}{'…' if len(nb) > 15 else ''}.")

        print(f"[4] track '{name}': features + backtest")
        ev = features_agent.add_scores(events, bars, caps)
        if tr["unit"] == "m":
            results = backtest_agent.measure_reactions(ev, bars)
        else:
            results = backtest_agent.measure_reactions_daily(ev, bars, market)
        react = backtest_agent.reaction_table(results)
        cats = backtest_agent.category_table(results)
        wf = backtest_agent.walk_forward(results)
        bands = {}
        if wf and wf.get("best"):
            for col in features_agent.SCORE_COLUMNS:
                bands[col] = backtest_agent.band_table(results, wf["best"], col)
        latency = None
        if tr["unit"] == "m" and not args.demo:
            from agents import latency_agent
            print(f"    latency study: trade-by-trade prices around strong news (up to {config.LATENCY_MAX_MINUTES:g} min)")
            latency = latency_agent.latency_study(results, bars)
        print(f"[5] track '{name}': report")
        path, _ = report_agent.write_report(results, wf, react, cats, mode, period, bands, tr_checks,
                                            latency=latency)
        status, text = report_agent.verdict(wf)
        summary.append((name, status, text, path))

    print("\n=== RESULTS ===")
    for name, status, text, path in summary:
        print(f"{name:9s} {status.upper():7s} {path}")
        print(f"          {text}")


def _explain(e):
    """One plain line about what went wrong, without ever printing a key."""
    import requests
    if isinstance(e, requests.HTTPError) and e.response is not None:
        code, url = e.response.status_code, e.response.url.split("?")[0]
        body = e.response.text[:300]
        if "alpaca" in url and code in (401, 403):
            if "subscription" in body.lower() or "sip" in body.lower():
                return f"Alpaca: the free data plan does not allow this request ({code}): {body}"
            return (f"Alpaca refused the request ({code}) at {url}: {body}. If this is about the keys, "
                    "check ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY (Paper account keys, no spaces).")
        if "anthropic" in url and code in (401, 403):
            return f"Claude API refused the key ({code}). Check ANTHROPIC_API_KEY."
        if "anthropic" in url and code == 400:
            return f"Claude API rejected the request (400): {e.response.text[:300]}"
        return f"HTTP {code} from {url}: {e.response.text[:300]}"
    return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    try:
        main()
    except config.TimeBudgetReached:
        print("::notice::Time budget reached. Everything downloaded and classified so far is saved; "
              "run the workflow again and it will continue from here.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        msg = _explain(e)
        # "::error::" makes GitHub Actions show the message on the run's summary page.
        print(f"::error::{msg}")
        raise
