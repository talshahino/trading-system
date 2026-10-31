"""Trading Prophet on whole portfolios (data/portfolio/portfolios.json from portfolio_search.py):
all combos of one portfolio trade together in ONE $100K FTMO account, as the demo robot will,
replayed minute by minute (floating loss on equity). Writes output/report_portfolios.html.

  python verify_portfolio.py --trades 'reports/*/combo_trades_*.csv.gz' --portfolios data/portfolio/portfolios.json --end 2026-10-30
"""
import argparse
import html
import json
import types

import pandas as pd

import config
from agents import ftmo_agent as F
import portfolio_search as PS
import run_poc as RP
import run_rules as RR

F.MAX_OPEN = 99            # the portfolio robot caps by risk, not by a count of 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True)
    ap.add_argument("--portfolios", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--test-from", default="2025-04-30")
    args = ap.parse_args()
    RR.SPLIT = pd.Timestamp(args.test_from, tz="UTC")
    t = PS.load(args.trades)
    ports = json.load(open(args.portfolios, encoding="utf-8"))
    ns = types.SimpleNamespace(demo=False, end=args.end)
    blocks = []
    for p in ports:
        names = [c["combo"] for c in p["combos"]]
        tr = t[t["combo"].isin(names)].sort_values("time").reset_index(drop=True)
        res = {}
        for rk in sorted({p["risk"], 0.005, 0.0075, 0.01, 0.0025}):
            w = F.account_path(tr, rk, start=RR.SPLIT)
            led = w[0].attrs.get("ledger") if len(w[0]) else None
            res[rk] = (w, RR.prophet_check(led, ns) if led is not None else None)
            print(p["rank"], rk, "done")
        rows = "".join(
            f"<tr><td>{html.escape(c['combo'].split('|')[0])}</td><td>{RP.TF_HE[c['combo'].split('|')[1]]}</td>"
            f"<td>{html.escape(c['combo'].split('|')[2])}</td><td>{c['trades']}</td><td>{c['avg_r']:+.2f}R</td></tr>"
            for c in p["combos"])
        rk_rows = "".join(f"<tr><td>{rk * 100:g}%{' ← נבחר' if rk == p['risk'] else ''}</td>"
                          f"<td>{RR._acc_text(w)}</td><td>{RR._prophet_cell(pr)}</td></tr>"
                          for rk, (w, pr) in sorted(res.items()))
        m = p["metrics"]
        blocks.append(
            f"<h2>תיק {p['rank']}: {len(names)} צירופים, סיכון {p['risk'] * 100:g}% לעסקה</h2>"
            f"<p class='mut'>חיפוש (עסקאות סגורות): רווח ${m['profit']:,.0f} · היום הגרוע ${m['worst_day']:,.0f} · "
            f"היתרה הנמוכה ${m['min_bal']:,.0f}</p>"
            f"<div class='tw'><table><tr><th>סיכון</th><th>חשבון $100K</th><th>Trading Prophet (דקה־דקה)</th></tr>{rk_rows}</table></div>"
            f"<div class='tw'><table><tr><th>אסטרטגיה</th><th>גרף</th><th>נכס</th><th>עסקאות</th><th>ממוצע</th></tr>{rows}</table></div>")
    page = (f"<!doctype html><html lang='he' dir='rtl'><head><meta charset='utf-8'><title>תיקי צירופים</title>"
            f"<style>body{{font-family:sans-serif;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px}}"
            f".mut{{color:#666}}.tw{{overflow-x:auto;margin:8px 0}}</style></head><body><h1>תיקי צירופים ל־FTMO</h1>"
            f"<p class='mut'>כל הצירופים של תיק סוחרים יחד בחשבון $100K אחד מ־{args.test_from}, כל אחד רק בנכס ובגרף שלו. "
            f"בחיפוש: היום הגרוע לא מתחת ל־{PS.DAILY_CAP:.0%}- והיתרה לא מתחת ל־{1 - PS.MAX_DD:.0%}. "
            f"הבדיקה כאן: Trading Prophet עם ההפסד הצף דקה אחרי דקה, ושומר המגבלות של הבוט פעיל.</p>"
            + "".join(blocks) + "</body></html>")
    (config.OUTPUT_DIR / "report_portfolios.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
