"""Backtest of rule-based strategies collected from TradingView (agents/rules_agent.py),
published parameters, no searching. Same test as the POC strategies:
  - 5 currency pairs + gold, silver, oil (Dukascopy), 1 / 5 / 15 minute and 1 hour charts
  - no trade opens or closes within 5 minutes of a high-impact news release
  - ONE $100,000 account per strategy under FTMO rules, opened one year ago, sampled weekly,
    stopped when burned; plus the chance to pass the FTMO 2-Step from different start dates.

  python run_rules.py [--demo]
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
from agents import rules_agent as R
import run_poc as RP

SYMBOLS = RP.FX + RP.COMMOD
TIMEFRAMES = RP.TIMEFRAMES + ["4Hour"]
SPLIT, SPLIT_HE = RP.SPLIT, RP.SPLIT_HE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-29")
    ap.add_argument("--only", default="", help="comma separated strategy ids, e.g. TV01,TV03")
    args = ap.parse_args()
    ids = [x for x in args.only.split(",") if x] or list(R.STRATEGIES)
    events = None
    if not args.demo:
        events, real_end = NC.load()
    results = []
    for tf in (["1Hour"] if args.demo else TIMEFRAMES):
        per_sid = {sid: [] for sid in ids}
        for i, s in enumerate(["DEMO1", "DEMO2"] if args.demo else SYMBOLS):
            st = max(args.start, RP.MINUTE_START) if tf.endswith("Min") else args.start
            df = RP.demo_bars(i) if args.demo else P.bars(s, tf, st, args.end)
            if df is None or df.empty:
                continue
            cost = 0.00005 if args.demo else RP.cost_of(s)
            for sid in ids:
                t = R.run(df, R.STRATEGIES[sid][1], cost)
                if t.empty:
                    continue
                if events is not None:
                    bad = NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events)
                    t = t[~bad]
                per_sid[sid].append(t.assign(symbol=s))
            print(f"  {s} {tf}: {len(df):,} bars")
        split = (pd.Timestamp("2018-07-01", tz="UTC") if args.demo else SPLIT)
        for sid, parts in per_sid.items():
            if not parts:
                continue
            tr = pd.concat(parts, ignore_index=True).sort_values("time")
            before, after = tr[tr["time"] < split], tr[tr["time"] >= split]
            acc = {rk: F.account_path(after[["time", "exit_time", "r"]], rk, start=split) for rk in RP.RISKS} \
                if len(after) >= 5 else {}
            ch = F.evaluate(after[["time", "exit_time", "r"]]) if len(after) >= 20 else None
            _, ch_best = F.verdict(ch)
            res = {"sid": sid, "name": R.STRATEGIES[sid][0], "tf": tf, "before": RP.stats(before["r"]),
                   "after": RP.stats(after["r"]), "accounts": acc, "ftmo_best": ch_best,
                   "by_sym": {s: RP.stats(q["r"]) for s, q in after.groupby("symbol")}}
            w = acc.get(RP.MAIN_RISK, (pd.DataFrame(), None, None))
            res["ok"] = bool(len(after) >= 100 and w[1] is None and len(w[0]) and w[0]["balance"].iloc[-1] > F.INITIAL)
            results.append(res)
            print(sid, tf, "trades", len(after), "avgR", round(res["after"]["avg_r"], 3) if len(after) else None)
    report(results, args)


def report(results, args):
    f2 = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.2f}"
    rows = ""
    for r in results:
        rows += (f"<tr><td>{r['sid']}</td><td>{html.escape(r['name'])}</td><td>{RP.TF_HE[r['tf']]}</td>"
                 f"<td>{r['after']['trades']:,}</td><td>{f2(r['after']['avg_r'])}</td><td>{RP._wk(r)}</td>"
                 f"<td>{'✅' if r['ok'] else '❌'}</td><td>{'—' if r['ftmo_best'] is None else RP._pct(r['ftmo_best']['p_both'])}</td></tr>")
    body = ""
    for r in results:
        sym = "".join(f"<tr><td>{s}</td><td>{q['trades']:,}</td><td>{f2(q['avg_r'])}</td><td>{f2(q['t'])}</td></tr>"
                      for s, q in r["by_sym"].items())
        b, a = r["before"], r["after"]
        body += (f"<h3>{r['sid']} · {html.escape(r['name'])} · {RP.TF_HE[r['tf']]}</h3>"
                 f"<p class='mut'>לפני {SPLIT_HE}: {b['trades']:,} עסקאות, ממוצע {f2(b['avg_r'])}R · "
                 f"השנה האחרונה: {a['trades']:,} עסקאות, ממוצע {f2(a['avg_r'])}R, הצלחה "
                 f"{'—' if a['win'] != a['win'] else f'{a['win'] * 100:.0f}%'}</p>"
                 + RP._weekly_html(r) +
                 "<details><summary>לפי נכס (השנה האחרונה)</summary><div class='tw'><table><tr><th>נכס</th><th>עסקאות</th>"
                 "<th>ממוצע R</th><th>t</th></tr>" + sym + "</table></div></details>")
    doc = "".join(f"<li><b>{sid}</b> {html.escape(R.STRATEGIES[sid][0])}: {html.escape((R.STRATEGIES[sid][1].__doc__ or '').strip())}</li>"
                  for sid in R.STRATEGIES)
    page = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TradingView Strategies</title>
<style>:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e3de}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--card:#1e1e1c;--line:#33332f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}main{{max-width:860px;margin:auto;padding:20px 16px 40px}}
.mut{{color:var(--mut);font-size:13px}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:start;white-space:nowrap}}h3{{margin-top:28px}}li{{margin:6px 0}}</style></head><body><main>
<h1>בקטסט · אסטרטגיות מ־TradingView</h1>
<p class="mut">{'דמו' if args.demo else 'Dukascopy'}: 5 צמדי מט״ח, זהב, כסף ונפט · גרפים: דקה, 5, 15 דקות ושעה · הפרמטרים כפי שפורסמו, בלי חיפוש ובלי התאמה ·
בלי עסקאות 5 דקות לפני ואחרי הודעות כלכליות חשובות · חשבון FTMO של $100,000 לכל אסטרטגיה מ־{SPLIT_HE}, נדגם כל שבוע, נעצר כשנשרף.</p>
<h2>שורה תחתונה</h2>
<div class='tw'><table><tr><th>#</th><th>אסטרטגיה</th><th>גרף</th><th>עסקאות בשנה האחרונה</th><th>ממוצע R</th>
<th>חשבון $100K מ־{SPLIT_HE} (סיכון 1%)</th><th>שרד ורווחי</th><th>עובר אתגר 2-Step</th></tr>{rows}</table></div>
<h2>הכללים שנבדקו</h2><ul>{doc}</ul>
{body}
<p class="mut">פרמטרים שהמחבר לא פרסם הושלמו בהנחות מקובלות, כמפורט בתיאור של כל אסטרטגיה. זה כלי מחקר, לא ייעוץ השקעות.</p>
</main></body></html>"""
    path = config.OUTPUT_DIR / "report_rules.html"
    path.write_text(page, encoding="utf-8")
    print(f"\n=== RESULT ===\n{path}")


if __name__ == "__main__":
    import sys
    from run_backtest import _explain
    try:
        main()
    except config.TimeBudgetReached:
        print("::notice::Time budget reached; run again to continue.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"::error::{_explain(e)}")
        raise
