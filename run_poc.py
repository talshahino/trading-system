"""Backtest of the volume-profile POC pullback strategy from the user's TikTok videos.

  python run_poc.py --start 2016-01-01 --end 2026-10-29
  python run_poc.py --demo

For each timeframe (15 min, 1 hour, 1 day) the reading of the rules (entry / stop / target /
swing size) is chosen on data before 2026 and checked untouched on 2026, and every trade is
compared with random trades that have the same direction, stop and target.
"""
import argparse
import html
import math

import numpy as np
import pandas as pd

import config
from agents import ftmo_agent as F
from agents import news_calendar as NC
from agents import poc_agent as P

ETFS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLE", "XLF", "SMH"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "AMD", "JPM", "NFLX"]
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD"]
FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
COMMOD = ["XAUUSD", "XAGUSD", "WTI"]
INDICES = ["US30", "US100", "US500", "GER40", "UK100", "JP225"]
COMMOD2 = ["BRENT", "NATGAS", "COPPER"]
CRYPTO_CFD = ["BTCUSD", "ETHUSD"]
FTMO_ALL = FX + COMMOD + INDICES + COMMOD2 + CRYPTO_CFD
# Per side, as a share of price: typical retail spread + slippage.
# EURUSD ~0.6 pip spread = ~0.006% round trip; gold ~$0.30 on ~$2,500 = ~0.012%.
COST = {"etf": 0.0002, "stock": 0.0003, "crypto": 0.002, "fx": 0.00005, "XAUUSD": 0.0001,
        "XAGUSD": 0.0003, "WTI": 0.0003, "index": 0.00005, "BRENT": 0.0003, "NATGAS": 0.0006,
        "COPPER": 0.0003, "BTCUSD": 0.0005, "ETHUSD": 0.0007}
TIMEFRAMES = ["1Min", "5Min", "15Min", "1Hour"]
RISKS = (0.005, 0.01, 0.02)     # risk per trade, % of the $100,000
MAIN_RISK = 0.01
TF_MIN = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "4Hour": 240, "1Day": 1440}
TF_HE = {"4Hour": "4 שעות", "1Min": "דקה", "5Min": "5 דקות", "15Min": "15 דקות", "1Hour": "שעה", "1Day": "יום"}
# 1-minute history is huge; minute timeframes start in 2021 (train 2021-2025, test 2026).
MINUTE_START = "2021-01-01"
# Rules are chosen on everything before the last 12 months; the FTMO account opens exactly
# one year ago and runs to today on data the choice never saw.
SPLIT = pd.Timestamp("2025-10-30", tz="UTC")
SPLIT_HE = "30.10.2025"
# Closest reading to what each video shows.
LITERAL = {"trend_poc": ("reaction", "swing/high", 5), "range_breakout": ("reaction", "range/measured", 0),
           "range_sweep": ("reaction", "range/measured", 0), "va_fade": ("reaction", "extreme/poc", 0)}
FAMILY_SHORT = {"trend_poc": "מגמה + POC", "range_breakout": "פריצת דשדוש", "range_sweep": "מניפולציה + פריצה",
                "va_fade": "דחייה מאזור הערך"}
FAMILY_HE = {"trend_poc": "סרטונים 1–2: מגמה, פרופיל משפל לשיא, חזרה ל־POC",
             "range_breakout": "סרטון 3: דשדוש, פריצה, חזרה ל־POC של הדשדוש",
             "range_sweep": "סרטון 5: דשדוש, \"מניפולציה\" (שבירה שקרית של צד אחד), התפרצות לצד השני, חזרה ל־POC",
             "va_fade": "סרטון 4: נגיעה בקצה אזור הערך של היום הקודם ודחייה, יעד ה־POC של אתמול"}
ENTRY_HE = {"touch": "נגיעה ב־POC (פקודה ממתינה)", "reaction": "תגובה: נר שסוגר בחזרה מעבר ל־POC"}
EXIT_HE = {"range/2R": "סטופ בצד השני של הדשדוש, יעד 1:2", "range/3R": "סטופ בצד השני של הדשדוש, יעד 1:3",
           "range/measured": "סטופ בצד השני של הדשדוש, יעד בגובה הדשדוש מעל הפריצה",
           "atr/measured": "סטופ ATR אחד, יעד בגובה הדשדוש מעל הפריצה",
           "atr/poc": "סטופ ATR אחד, יעד POC של אתמול", "extreme/poc": "סטופ מעבר לקיצון של היום, יעד POC של אתמול",
           "swing/high": "סטופ מעבר לשפל/שיא, יעד בקצה התנועה", "swing/2R": "סטופ מעבר לשפל/שיא, יעד 1:2",
           "swing/3R": "סטופ מעבר לשפל/שיא, יעד 1:3", "atr/high": "סטופ ATR אחד, יעד בקצה התנועה",
           "atr/2R": "סטופ ATR אחד, יעד 1:2", "atr/3R": "סטופ ATR אחד, יעד 1:3"}


def asset_class(s):
    if s in FX:
        return "fx"
    if s in COMMOD or s in COMMOD2:
        return "commodity"
    if s in INDICES:
        return "index"
    if s in CRYPTO_CFD:
        return "crypto"
    return "crypto" if "/" in s else "etf" if s in ETFS else "stock"


def cost_of(s):
    return COST.get(s) or COST[asset_class(s)]


def demo_bars(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=8000, freq="h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    o = np.concatenate([[c[0]], c[:-1]])
    spread = np.abs(rng.normal(0, 0.004, len(idx)))
    return pd.DataFrame({"o": o, "h": np.maximum(o, c) * (1 + spread), "l": np.minimum(o, c) * (1 - spread),
                         "c": c, "v": rng.integers(1000, 5000, len(idx))}, index=idx)


def stats(r):
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 2:
        return {"trades": n, "win": np.nan, "avg_r": np.nan, "t": np.nan}
    sd = r.std(ddof=1)
    return {"trades": n, "win": float((r > 0).mean()), "avg_r": float(r.mean()),
            "t": float(r.mean() / (sd / math.sqrt(n))) if sd > 0 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-29")
    ap.add_argument("--download-only", action="store_true", help="only fetch and cache the price data")
    ap.add_argument("--market", default="forex", choices=["forex", "stocks", "ftmo"],
                    help="forex = 5 currency pairs + gold, silver, oil (Dukascopy); stocks = ETFs, stocks, crypto (Alpaca)")
    args = ap.parse_args()
    symbols = (["DEMO1", "DEMO2", "DEMO3"] if args.demo else FX + COMMOD if args.market == "forex"
               else FTMO_ALL if args.market == "ftmo" else ETFS + STOCKS + CRYPTO)
    results, news_info, news_tot = [], None, [0, 0]
    events = real_end = None
    if not args.demo and args.market == "forex":
        events, real_end = NC.load()
    for tf in (["1Hour"] if args.demo else TIMEFRAMES):
        all_tr, data = [], {}
        for i, s in enumerate(symbols):
            st = max(args.start, MINUTE_START) if tf.endswith("Min") else args.start
            df = demo_bars(i) if args.demo else P.bars(s, tf, st, args.end)
            if df is None or df.empty:
                print(f"  {s} {tf}: no data")
                continue
            if args.download_only:
                print(f"  {s} {tf}: {len(df):,} bars (downloaded)")
                continue
            data[(s, tf)] = df
            cost = COST["etf"] if args.demo else cost_of(s)
            cls = "etf" if args.demo else asset_class(s)
            parts = [("trend_poc", P.trades(df, k, cost)) for k in (3, 5)]
            parts += [("range_breakout", P.trades_range(df, False, cost)),
                      ("range_sweep", P.trades_range(df, True, cost))]
            if tf != "1Day":
                # Forex/commodities/crypto trade around the clock: a "session" is the UTC day.
                parts.append(("va_fade", P.trades_fade(df, cost, crypto=cls in ("crypto", "fx", "commodity"))))
            for fam, t in parts:
                if not t.empty:
                    all_tr.append(t.assign(symbol=s, tf=tf, cls=cls, cost=cost, family=fam))
            print(f"  {s} {tf}: {len(df):,} bars")
        if args.download_only or not all_tr:
            continue
        tr = pd.concat(all_tr, ignore_index=True)
        del all_tr
        tr["exit_time"] = tr["time"] + pd.to_timedelta(tr["held"] * tr["tf"].map(TF_MIN), unit="min")
        # News rule: no trade may open or close from 5 minutes before to 5 minutes after a
        # high-impact release for one of the instrument's currencies.
        if not args.demo and args.market == "forex":
            drop = pd.Series(False, index=tr.index)
            for sym, idx in tr.groupby("symbol").groups.items():
                sub = tr.loc[idx]
                drop.loc[idx] = (NC.blocked(sub["time"], sym, events) | NC.blocked(sub["exit_time"], sym, events)).to_numpy()
            news_tot[0] += int(drop.sum()); news_tot[1] += len(tr)
            news_info = {"dropped": news_tot[0], "total": news_tot[1], "real_end": real_end, "events": len(events)}
            print(f"news filter: {drop.sum():,} of {len(tr):,} trade rows touch a +-5 min news window -> removed")
            tr = tr[~drop]
        split = tr["time"].quantile(0.6) if args.demo else SPLIT
        tr["period"] = np.where(tr["time"] < split, "train", "test")
        print(f"{len(tr):,} trade rows")

        for (fam, _tf), g in tr.groupby(["family", "tf"], observed=True):
            train = g[g["period"] == "train"]
            grid = []
            for (e, x, k), gg in train.groupby(["entry", "exit", "k"]):
                st = stats(gg["r"])
                grid.append({"entry": e, "exit": x, "k": k, **st})
            grid = pd.DataFrame(grid).sort_values("t", ascending=False)
            best = tuple(grid.iloc[0][["entry", "exit", "k"]])
            for label, rule in (("best", best), ("literal", LITERAL[fam])):
                sel = g[(g["entry"] == rule[0]) & (g["exit"] == rule[1]) & (g["k"] == rule[2])]
                rows = {}
                for per in ("train", "test"):
                    part = sel[sel["period"] == per]
                    # random comparison: same symbol/timeframe, same direction, stop and target
                    rnd = []
                    for (s, tf2), q in part.groupby(["symbol", "tf"]):
                        rnd.append(P.random_baseline(data[(s, tf2)], q, q["cost"].iloc[0], draws=10))
                    part = pd.concat(rnd) if rnd else part.assign(r_rand=[])
                    st = stats(part["r"])
                    diff = (part["r"] - part["r_rand"]).dropna()
                    st.update({"rand_r": float(part["r_rand"].mean()) if len(part) else np.nan,
                               "edge_t": stats(diff)["t"],
                               "per_year": len(part) / max((part["time"].max() - part["time"].min()).days / 365, 0.5)
                               if len(part) else 0})
                    by_cls = {c: stats(q["r"]) for c, q in part.groupby("cls")}
                    rows[per] = (st, by_cls)
                    if per == "test" and len(part) >= 20:
                        pt = part[["time", "exit_time", "r"]]
                        ftmo_tbl = F.evaluate(pt)
                        ftmo_ok, ftmo_best = F.verdict(ftmo_tbl)
                        # The user's test: one $100,000 FTMO-rules account per strategy, opened on the
                        # first day of the test period, sampled weekly, stopped when burned.
                        accounts = {rk: F.account_path(pt, rk, start=split) for rk in RISKS}
                        w_main, blown_main, _ = accounts[MAIN_RISK]
                        week_ok = (blown_main is None and len(w_main) and w_main["balance"].iloc[-1] > F.INITIAL
                                   and len(part) >= 100)
                        ftmo_ok = ftmo_ok and len(part) >= 100
                    elif per == "test":
                        ftmo_tbl, ftmo_ok, ftmo_best = None, False, None
                        accounts, week_ok = {}, False
                results.append({"family": fam, "tf": tf, "label": label, "rule": rule, "rows": rows, "n_rules": len(grid),
                                "grid": grid, "ftmo": ftmo_tbl, "ftmo_ok": ftmo_ok, "ftmo_best": ftmo_best,
                                "accounts": accounts, "week_ok": week_ok})
                print(fam, tf, label, rule, {p: (round(v[0]["avg_r"], 3), v[0]["trades"]) for p, v in rows.items()})
        del data, tr
        import gc
        gc.collect()
    if args.download_only:
        print("download only: done")
        return
    report(results, args, news_info)


def _curve(w, blown):
    if w is None or len(w) < 2:
        return ""
    y = w["balance"].tolist()
    lo, hi = min(y + [90000]), max(y + [110000])
    W, H = 640, 160
    pts = " ".join(f"{i / (len(y) - 1) * W:.1f},{H - (v - lo) / (hi - lo) * H:.1f}" for i, v in enumerate(y))
    ystart = H - (100000 - lo) / (hi - lo) * H
    yfloor = H - (90000 - lo) / (hi - lo) * H
    col = "#c0392b" if blown is not None else ("#1f8a4c" if y[-1] > 100000 else "#b7791f")
    return (f"<svg viewBox='0 0 {W} {H}' preserveAspectRatio='none' style='width:100%;height:160px;background:var(--card)'>"
            f"<line x1='0' x2='{W}' y1='{ystart:.1f}' y2='{ystart:.1f}' stroke='#999' stroke-dasharray='4'/>"
            f"<line x1='0' x2='{W}' y1='{yfloor:.1f}' y2='{yfloor:.1f}' stroke='#c0392b' stroke-dasharray='2'/>"
            f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2'/></svg>"
            "<p class='mut'>היתרה בכל שבוע. קו מקווקו אפור = $100,000, אדום = $90,000 (נשרף).</p>")


def _weekly_html(res):
    acc = res.get("accounts") or {}
    if MAIN_RISK not in acc:
        return "<p class='mut'>חשבון FTMO: מעט מדי עסקאות.</p>"
    w, blown, yearly = acc[MAIN_RISK]
    if w is None or w.empty:
        return "<p class='mut'>חשבון FTMO: אין עסקאות בתקופת המבחן.</p>"
    status = (f"❌ נשרף ב־{pd.Timestamp(blown):%d.%m.%Y}" if blown is not None else
              ("✅ שרד ורווחי" if w["balance"].iloc[-1] > F.INITIAL else "⚠️ שרד, אבל בהפסד"))
    yrows = "".join(f"<tr><td>{int(r.year)}</td><td>${r.start:,.0f}</td><td>${r.end:,.0f}</td><td>${r.peak:,.0f}</td>"
                    f"<td>${r.low:,.0f}</td><td>{'כן' if r.survived else 'לא'}</td></tr>" for r in yearly.itertuples())
    wrows = "".join(f"<tr><td>{r.week_end:%d.%m.%Y}</td><td>${r.balance:,.0f}</td><td>{r.change:+,.0f}</td><td>${r.peak:,.0f}</td></tr>"
                    for r in w.itertuples())
    others = " · ".join(
        f"סיכון {rk * 100:g}%: " + (f"נשרף ב־{pd.Timestamp(acc[rk][1]):%m/%Y}" if acc[rk][1] is not None else
                                    f"${acc[rk][0]['balance'].iloc[-1]:,.0f}" if len(acc[rk][0]) else "—")
        for rk in RISKS if rk != MAIN_RISK and rk in acc)
    return (f"<p><b>חשבון FTMO של $100,000, נפתח ב־{SPLIT_HE}, סיכון {MAIN_RISK * 100:g}% לעסקה:</b> {status} · "
            f"יתרה אחרונה ${w['balance'].iloc[-1]:,.0f} · שיא ${w['peak'].max():,.0f}</p>" + _curve(w, blown) +
            "<div class='tw'><table><tr><th>שנה</th><th>פתיחה</th><th>סוף</th><th>שיא</th><th>שפל</th><th>שרד את השנה</th></tr>"
            + yrows + "</table></div>"
            f"<p class='mut'>לשם השוואה, אותו חשבון ב{others}.</p>"
            f"<details><summary>דגימה שבועית ({len(w)} שבועות)</summary><div class='tw'><table>"
            "<tr><th>סוף שבוע</th><th>יתרה</th><th>שינוי בשבוע</th><th>שיא עד כה</th></tr>" + wrows + "</table></div></details>")


def _pct(v):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.0f}%"


def _ftmo_html(res):
    t = res.get("ftmo")
    if t is None or t.empty:
        return "<p class='mut'>FTMO: מעט מדי עסקאות בתקופת המבחן.</p>"
    rows = "".join(
        f"<tr><td>{r.risk * 100:g}%</td><td>{_pct(r.p_phase1)}</td><td><b>{_pct(r.p_both)}</b></td>"
        f"<td>{'—' if r.median_days != r.median_days else int(r.median_days)}</td><td>{_pct(r.p_1step)}</td>"
        f"<td>{_pct(r.funded_lost)}</td><td>{'—' if r.funded_payout != r.funded_payout else f'${r.funded_payout:,.0f}'}</td></tr>"
        for r in t.itertuples())
    mark = "✅ עובר את מסנן FTMO" if res.get("ftmo_ok") else "❌ לא עובר את מסנן FTMO"
    return ("<p><b>חשבון FTMO של $100,000 (תקופת המבחן):</b> " + mark + "</p>"
            "<div class='tw'><table><tr><th>סיכון לעסקה</th><th>עובר שלב 1</th><th>עובר את שני השלבים</th>"
            "<th>ימים (חציון)</th><th>עובר 1-Step</th><th>חשבון ממומן נשרף תוך שנה</th><th>תשלום שנתי לך (חציון)</th></tr>"
            + rows + "</table></div>")


def _news_html(n):
    if not n:
        return ""
    return (f"<p class='mut'>הודעות כלכליות: {n['events']:,} הודעות בהשפעה גבוהה. נתונים מלאים עד {n['real_end']:%Y-%m-%d}, "
            "ואחר כך רק דוח התעסוקה האמריקאי והחלטות הריבית של הפד. "
            f"{n['dropped']:,} מתוך {n['total']:,} עסקאות נפתחו או נסגרו בחלון של 5 דקות סביב הודעה, והוצאו.</p>")


def _wk(r):
    acc = r.get("accounts") or {}
    if MAIN_RISK not in acc or acc[MAIN_RISK][0] is None or acc[MAIN_RISK][0].empty:
        return "—"
    w, blown, _ = acc[MAIN_RISK]
    if blown is not None:
        return f"נשרף {pd.Timestamp(blown):%m/%Y}"
    return f"${w['balance'].iloc[-1]:,.0f} (שיא ${w['peak'].max():,.0f})"


def _summary_html(results):
    rows = ""
    for r in results:
        st = r["rows"]["test"][0]
        b = r.get("ftmo_best")
        rows += (f"<tr><td>{html.escape(FAMILY_SHORT.get(r['family'], r['family']))}</td><td>{TF_HE[r['tf']]}</td>"
                 f"<td>{'טובה ביותר' if r['label'] == 'best' else 'כמו בסרטון'}</td><td>{st['trades']:,}</td>"
                 f"<td>{'—' if st['avg_r'] != st['avg_r'] else f'{st['avg_r']:+.2f}'}</td>"
                 f"<td>{_wk(r)}</td><td>{'✅' if r.get('week_ok') else '❌'}</td>"
                 f"<td>{'—' if b is None else _pct(b['p_both'])}</td></tr>")
    return ("<h2>שורה תחתונה: איזו אסטרטגיה טובה בתנאי FTMO?</h2>"
            "<p class='mut'>המסנן הראשי: לכל אסטרטגיה חשבון אחד של $100,000 בחוקי FTMO (הפסד יומי עד 5%, הפסד כולל עד 10%, "
            f"בלי מסחר 5 דקות לפני ואחרי הודעות כלכליות חשובות), שנפתח ב־{SPLIT_HE}, שנה אחורה, ורץ עד היום. "
            "היתרה נדגמת כל שבוע, וחשבון שנשרף נעצר. <b>טוב = שרד עד היום ורווחי</b>, עם לפחות 100 עסקאות. "
            "בעמודה האחרונה, לידיעה: הסיכוי לעבור את אתגר ה־2-Step (יעד 10% ואז 5%) מתאריכי התחלה שונים.</p>"
            "<div class='tw'><table><tr><th>אסטרטגיה</th><th>גרף</th><th>קריאה</th><th>עסקאות במבחן</th><th>ממוצע R</th>"
            f"<th>חשבון $100K מ־{SPLIT_HE} (סיכון 1%)</th><th>מסנן</th><th>עובר אתגר 2-Step</th></tr>" + rows + "</table></div>")


def report(results, args, news_info=None):
    f = lambda v, d=2: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.{d}f}"
    body = ""
    for fam in FAMILY_HE:
      if any(r["family"] == fam for r in results):
        body += f"<h2>{html.escape(FAMILY_HE[fam])}</h2>"
      for tf in (["1Hour"] if args.demo else TIMEFRAMES):
        for res in [r for r in results if r["tf"] == tf and r["family"] == fam]:
            e, x, k = res["rule"]
            title = ("הקריאה הטובה ביותר על הנתונים שלפני השנה האחרונה" if res["label"] == "best"
                     else "הקריאה הכי קרובה לסרטון")
            rows = ""
            for per, he in (("train", f"עד {SPLIT_HE} (בחירה)"), ("test", "השנה האחרונה (מבחן)")):
                st, by = res["rows"][per]
                rows += (f"<tr><td>{he}</td><td>{st['trades']:,}</td><td>{'—' if st['win'] != st['win'] else f"{st['win'] * 100:.0f}%"}</td>"
                         f"<td><b>{f(st['avg_r'])}</b></td><td>{f(st['t'], 1)}</td><td>{f(st['rand_r'])}</td>"
                         f"<td>{f(st['avg_r'] - st['rand_r'] if st['rand_r'] == st['rand_r'] else np.nan)}</td><td>{f(st['edge_t'], 1)}</td></tr>")
            cls_rows = "".join(
                f"<tr><td>{ {'etf': 'קרנות סל', 'stock': 'מניות', 'crypto': 'קריפטו', 'fx': 'מט״ח', 'commodity': 'סחורות', 'index': 'מדדים'}[c] }</td><td>{s['trades']:,}</td>"
                f"<td>{f(s['avg_r'])}</td><td>{f(s['t'], 1)}</td></tr>"
                for c, s in res["rows"]["test"][1].items())
            st_test = res["rows"]["test"][0]
            yearly = (st_test["avg_r"] * st_test["per_year"]) if st_test["avg_r"] == st_test["avg_r"] else np.nan
            picked = f" · נבחרה מתוך {res['n_rules']} קריאות" if res["label"] == "best" else ""
            body += (f"<h3>{TF_HE[tf]} · {title}</h3><p class='mut'>{html.escape(ENTRY_HE[e])} · {html.escape(EXIT_HE[x])} · "
                     f"פיבוט של {k} נרות{picked}</p>"
                     "<div class='tw'><table><tr><th>תקופה</th><th>עסקאות</th><th>הצלחה</th><th>ממוצע R נטו</th><th>t</th>"
                     "<th>R אקראי</th><th>יתרון על אקראי</th><th>t של היתרון</th></tr>" + rows + "</table></div>"
                     "<p class='mut'>במבחן לפי סוג נכס:</p><div class='tw'><table><tr><th>נכס</th><th>עסקאות</th><th>ממוצע R</th><th>t</th></tr>"
                     + cls_rows + "</table></div>"
                     + _weekly_html(res) + _ftmo_html(res))
    page = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>POC Strategy Backtest</title>
<style>:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e3de}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--card:#1e1e1c;--line:#33332f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}main{{max-width:820px;margin:auto;padding:20px 16px 40px}}
.mut{{color:var(--mut);font-size:13px}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:start;white-space:nowrap}}h3{{margin-top:28px}}</style></head><body><main>
<h1>בקטסט · אסטרטגיית POC מהסרטונים</h1>
<p class="mut">{'דמו (נתונים סינתטיים)' if args.demo else 'נתונים אמיתיים (' + ('Dukascopy' if args.market == 'forex' else 'Alpaca') + ')'} · שעה: {args.start} עד {args.end}, דקה / 5 / 15 דקות: {max(args.start, MINUTE_START)} עד {args.end} · חלוקה: עד {SPLIT_HE} בחירה, השנה האחרונה מבחן ·
{'5 צמדי מט״ח (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD), זהב, כסף ונפט (Dukascopy, ווליום = מספר עדכוני מחיר) · עלות לצד: מט״ח 0.5 bps, זהב 1 bps, כסף ונפט 3 bps' if args.market == 'forex' else '10 קרנות סל, 10 מניות גדולות, ביטקוין/את׳ריום/סולנה · עלות לצד: קרנות 2 bps, מניות 3 bps, קריפטו 20 bps'}</p>
<p><b>R</b> = יחידת סיכון: ‎+1R פירושו שהרווחת בדיוק את מה שסיכנת, ‎−1R שהגעת לסטופ. ממוצע R חיובי אחרי עלויות = רווחי.
<b>R אקראי</b> = אותן עסקאות (כיוון, סטופ ויעד) בנקודות כניסה אקראיות. אם ה־POC עובד, היתרון על האקראי צריך להיות חיובי, ו־t של היתרון 2 ומעלה.
ההשוואה לאקראי הוגנת במיוחד בכניסות "תגובה" (כניסה בפתיחת הנר הבא). בכניסות "נגיעה" (פקודה ממתינה) הנר שבו נכנסים מחושב בצורה פסימית: רק הסטופ יכול להיפגע בו.</p>
{_summary_html(results)}
{_news_html(news_info)}
<h2>אסטרטגיית FTMO: איך נבדק</h2>
<p class="mut">כל אסטרטגיה מקבלת חשבון של $100,000 שמתחיל בתאריכים שונים לאורך תקופת המבחן (40 תאריכי התחלה), עם העסקאות בסדר האמיתי שלהן.
הבוט לא פותח עסקה שהסטופ שלה עלול לשבור את מגבלת ההפסד היומית או הכוללת, ועד 3 עסקאות פתוחות. רק פער מחיר מעבר לסטופ יכול עדיין לשבור כלל.
"תשלום שנתי" = 90% מהרווח של חשבון ממומן בשנה שאחרי המעבר (חציון).</p>
{body}
<p class="mut">הסרטונים לא מגדירים סטופ, יעד ו"תגובה", ולכן נבדקו כמה קריאות לכל אסטרטגיה ומסגרת זמן. הקריאה הטובה נבחרה על הנתונים שלפני {SPLIT_HE} בלבד.
הגדרות קבועות: דשדוש = 20 נרות שהטווח שלהם עד 4 ATR; אזור ערך = 70% מהמחזור סביב ה־POC; "תגובה" = נר שסוגר בחזרה מעבר לרמה. אישור "order flow" מסרטון 4 לא נבדק (אין נתונים).
זה כלי מחקר, לא ייעוץ השקעות.</p></main></body></html>"""
    path = config.OUTPUT_DIR / "report_poc.html"
    path.write_text(page, encoding="utf-8")
    print(f"\n=== RESULT ===\n{path}")


if __name__ == "__main__":
    import sys
    from run_backtest import _explain
    try:
        main()
    except config.TimeBudgetReached:
        print("::notice::Time budget reached; downloaded data is saved, run again to continue.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"::error::{_explain(e)}")
        raise
