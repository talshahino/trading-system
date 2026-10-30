"""Backtest of the volume-profile POC pullback strategy from the user's TikTok videos.

  python run_poc.py --start 2016-01-01 --end 2026-10-29
  python run_poc.py --demo

For each timeframe (15 min, 1 hour, 1 day) the reading of the rules (entry / stop / target /
swing size) is chosen on 2016-2022 and checked untouched on 2023 onward, and every trade is
compared with random trades that have the same direction, stop and target.
"""
import argparse
import html
import math

import numpy as np
import pandas as pd

import config
from agents import poc_agent as P

ETFS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLE", "XLF", "SMH"]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "AMD", "JPM", "NFLX"]
CRYPTO = ["BTC/USD", "ETH/USD", "SOL/USD"]
COST = {"etf": 0.0002, "stock": 0.0003, "crypto": 0.002}     # per side
TIMEFRAMES = ["15Min", "1Hour", "1Day"]
TF_HE = {"15Min": "15 דקות", "1Hour": "שעה", "1Day": "יום"}
SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
LITERAL = ("reaction", "swing/high", 5)       # closest to what the video shows
ENTRY_HE = {"touch": "נגיעה ב־POC (פקודה ממתינה)", "reaction": "תגובה: נר שסוגר בחזרה מעבר ל־POC"}
EXIT_HE = {"swing/high": "סטופ מעבר לשפל/שיא, יעד בקצה התנועה", "swing/2R": "סטופ מעבר לשפל/שיא, יעד 1:2",
           "swing/3R": "סטופ מעבר לשפל/שיא, יעד 1:3", "atr/high": "סטופ ATR אחד, יעד בקצה התנועה",
           "atr/2R": "סטופ ATR אחד, יעד 1:2", "atr/3R": "סטופ ATR אחד, יעד 1:3"}


def asset_class(s):
    return "crypto" if "/" in s else "etf" if s in ETFS else "stock"


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
    args = ap.parse_args()
    symbols = ["DEMO1", "DEMO2", "DEMO3"] if args.demo else ETFS + STOCKS + CRYPTO
    all_tr, data = [], {}
    for tf in (["1Hour"] if args.demo else TIMEFRAMES):
        for i, s in enumerate(symbols):
            df = demo_bars(i) if args.demo else P.bars(s, tf, args.start, args.end)
            if df is None or df.empty:
                print(f"  {s} {tf}: no data")
                continue
            data[(s, tf)] = df
            cost = COST["etf"] if args.demo else COST[asset_class(s)]
            for k in (3, 5):
                tr = P.trades(df, k, cost)
                if not tr.empty:
                    all_tr.append(tr.assign(symbol=s, tf=tf, cls="etf" if args.demo else asset_class(s), cost=cost))
            print(f"  {s} {tf}: {len(df):,} bars")
    tr = pd.concat(all_tr, ignore_index=True)
    split = tr["time"].quantile(0.6) if args.demo else SPLIT
    tr["period"] = np.where(tr["time"] < split, "train", "test")
    print(f"{len(tr):,} trade rows")

    results = []
    for tf, g in tr.groupby("tf"):
        train = g[g["period"] == "train"]
        grid = []
        for (e, x, k), gg in train.groupby(["entry", "exit", "k"]):
            st = stats(gg["r"])
            grid.append({"entry": e, "exit": x, "k": k, **st})
        grid = pd.DataFrame(grid).sort_values("t", ascending=False)
        best = tuple(grid.iloc[0][["entry", "exit", "k"]])
        for label, rule in (("best", best), ("literal", LITERAL)):
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
            results.append({"tf": tf, "label": label, "rule": rule, "rows": rows, "n_rules": len(grid),
                            "grid": grid})
            print(tf, label, rule, {p: (round(v[0]["avg_r"], 3), v[0]["trades"]) for p, v in rows.items()})
    report(results, args)


def report(results, args):
    f = lambda v, d=2: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.{d}f}"
    body = ""
    for tf in (["1Hour"] if args.demo else TIMEFRAMES):
        for res in [r for r in results if r["tf"] == tf]:
            e, x, k = res["rule"]
            title = ("הקריאה הטובה ביותר על 2016–2022" if res["label"] == "best"
                     else "הקריאה הכי קרובה לסרטון")
            rows = ""
            for per, he in (("train", "2016–2022 (בחירה)"), ("test", "2023 ואילך (מבחן)")):
                st, by = res["rows"][per]
                rows += (f"<tr><td>{he}</td><td>{st['trades']:,}</td><td>{'—' if st['win'] != st['win'] else f"{st['win'] * 100:.0f}%"}</td>"
                         f"<td><b>{f(st['avg_r'])}</b></td><td>{f(st['t'], 1)}</td><td>{f(st['rand_r'])}</td>"
                         f"<td>{f(st['avg_r'] - st['rand_r'] if st['rand_r'] == st['rand_r'] else np.nan)}</td><td>{f(st['edge_t'], 1)}</td></tr>")
            cls_rows = "".join(
                f"<tr><td>{ {'etf': 'קרנות סל', 'stock': 'מניות', 'crypto': 'קריפטו'}[c] }</td><td>{s['trades']:,}</td>"
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
                     f"<p>בחשבון של $10,000 עם סיכון של 1.5% לעסקה: בערך <b>{f(yearly * 1.5 if yearly == yearly else np.nan, 1)}%</b> בשנה "
                     f"({st_test['per_year']:.0f} עסקאות בשנה × {f(st_test['avg_r'])}R × 1.5%), בלי להתחשב בחפיפה בין עסקאות.</p>")
    page = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>POC Strategy Backtest</title>
<style>:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e3de}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--card:#1e1e1c;--line:#33332f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}main{{max-width:820px;margin:auto;padding:20px 16px 40px}}
.mut{{color:var(--mut);font-size:13px}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:start;white-space:nowrap}}h3{{margin-top:28px}}</style></head><body><main>
<h1>בקטסט · אסטרטגיית POC מהסרטונים</h1>
<p class="mut">{'דמו (נתונים סינתטיים)' if args.demo else 'נתונים אמיתיים (Alpaca)'} · {args.start} עד {args.end} ·
10 קרנות סל, 10 מניות גדולות, ביטקוין/את׳ריום/סולנה · עלות לצד: קרנות 2 bps, מניות 3 bps, קריפטו 20 bps</p>
<p><b>R</b> = יחידת סיכון: ‎+1R פירושו שהרווחת בדיוק את מה שסיכנת, ‎−1R שהגעת לסטופ. ממוצע R חיובי אחרי עלויות = רווחי.
<b>R אקראי</b> = אותן עסקאות (כיוון, סטופ ויעד) בנקודות כניסה אקראיות. אם ה־POC עובד, היתרון על האקראי צריך להיות חיובי, ו־t של היתרון 2 ומעלה.</p>
{body}
<p class="mut">הסרטונים לא מגדירים סטופ, יעד ו"תגובה", ולכן נבדקו 24 קריאות לכל מסגרת זמן (2 כניסות × 6 יציאות × 2 גדלי פיבוט). הקריאה נבחרה על 2016–2022 בלבד.
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
