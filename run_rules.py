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
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config
from agents import ftmo_agent as F
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP

SYMBOLS = RP.FX + RP.COMMOD
SWEEP_RISKS = (0.0025, 0.005, 0.0075, 0.01)   # TV strategies: Prophet at each of these risks per trade
LOW_RISK = 0.0075            # every other strategy: 25% below the 1% main risk
MAX_PROPHET_TRADES = 25000   # minute-by-minute replay is skipped above this (too slow, and such systems burn anyway)
_BARS = {}                   # 1-minute bars per symbol, loaded once per run
ERRORS = []   # tracebacks of strategies that crashed; shown at the end of the report
TIMEFRAMES = RP.TIMEFRAMES + ["4Hour"]
SPLIT, SPLIT_HE = RP.SPLIT, RP.SPLIT_HE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-29")
    ap.add_argument("--only", default="", help="comma separated strategy ids, e.g. TV01,TV03")
    ap.add_argument("--market", default="ftmo", choices=["forex", "ftmo"],
                    help="forex = 5 pairs + gold/silver/oil; ftmo = also indices, more commodities, crypto")
    ap.add_argument("--export", default="", help="strategy:timeframe pairs to export trade by trade, "
                    "e.g. TV01:4Hour,TV09:1Hour (survivors are always exported)")
    args = ap.parse_args()
    ids = [x for x in args.only.split(",") if x] or list(R.STRATEGIES)
    events = None
    if not args.demo:
        events, real_end = NC.load()
    results, missing = [], set()
    for tf in (["1Hour"] if args.demo else TIMEFRAMES):
        per_sid = {sid: [] for sid in ids}
        universe = RP.FTMO_ALL if args.market == "ftmo" else SYMBOLS
        for i, s in enumerate(["DEMO1", "DEMO2"] if args.demo else universe):
            st = max(args.start, RP.MINUTE_START) if tf.endswith("Min") else args.start
            try:
                df = RP.demo_bars(i) if args.demo else P.bars(s, tf, st, args.end)
            except config.TimeBudgetReached:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  {s} {tf}: NO DATA ({str(e)[-200:]})")
                missing.add(s)
                continue
            if df is None or df.empty:
                continue
            cost = 0.00005 if args.demo else RP.cost_of(s)
            for sid in ids:
                if not args.demo and not R.timeframe_ok(sid, tf):
                    continue
                try:
                    t = R.run(df, R.STRATEGIES[sid][1], cost)
                except Exception:  # noqa: BLE001 - one broken strategy must not kill the run
                    ERRORS.append(f"{sid} {tf} {s}:\n{traceback.format_exc()}")
                    continue
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
            try:
                _summarise(sid, tf, parts, split, args, results)
            except Exception:  # noqa: BLE001
                ERRORS.append(f"{sid} {tf} summary:\n{traceback.format_exc()}")
    safe_report(results, args)


def safe_report(results, args):
    """Write the report; a result that breaks the HTML is dropped and its traceback shown."""
    try:
        report(results, args)
        return
    except Exception:  # noqa: BLE001
        ERRORS.append(f"report:\n{traceback.format_exc()}")
    good = []
    for r in results:
        try:
            report([r], args)
            good.append(r)
        except Exception:  # noqa: BLE001
            ERRORS.append(f"report row {r.get('sid')} {r.get('tf')}:\n{traceback.format_exc()}")
    report(good, args)


def _summarise(sid, tf, parts, split, args, results):
    """Account replay, FTMO challenge odds and Trading Prophet for one strategy/timeframe."""
    tr = pd.concat(parts, ignore_index=True).sort_values("time")
    before, after = tr[tr["time"] < split], tr[tr["time"] >= split]
    acc = {rk: F.account_path(after, rk, start=split) for rk in RP.RISKS} if len(after) >= 5 else {}
    ch = F.evaluate(after[["time", "exit_time", "r"]]) if len(after) >= 20 else None
    _, ch_best = F.verdict(ch)
    res = {"sid": sid, "name": R.STRATEGIES[sid][0], "tf": tf, "before": RP.stats(before["r"]),
           "after": RP.stats(after["r"]), "accounts": acc, "ftmo_best": ch_best,
           "by_sym": {s: RP.stats(q["r"]) for s, q in after.groupby("symbol")}}
    w = acc.get(RP.MAIN_RISK, (pd.DataFrame(), None, None))
    res["ok"] = bool(len(after) >= 100 and w[1] is None and len(w[0]) and w[0]["balance"].iloc[-1] > F.INITIAL)
    results.append(res)
    if (res["ok"] or f"{sid}:{tf}" in args.export.split(",")) and len(w[0]):
        export_trades(sid, tf, w[0].attrs.get("ledger"))
        res["prophet"] = prophet_check(w[0].attrs.get("ledger"), args)
    res["risk_tests"] = {}
    # lower risk cannot make a losing system pass, so the 4-level sweep runs only where the average trade is positive
    sweep = sid.startswith("TV") and len(after) and res["after"]["avg_r"] > 0
    for rk in (SWEEP_RISKS if sweep else (LOW_RISK,)):
        if len(after) < 5:
            continue
        w2 = acc[rk] if rk in acc else F.account_path(after, rk, start=split)
        led = w2[0].attrs.get("ledger") if len(w2[0]) else None
        pr = prophet_check(led, args) if led is not None and len(after) <= MAX_PROPHET_TRADES else None
        res["risk_tests"][rk] = {"account": w2, "prophet": pr, "skipped": len(after) > MAX_PROPHET_TRADES}
    print(sid, tf, "trades", len(after), "avgR", round(res["after"]["avg_r"], 3) if len(after) else None)


def prophet_check(led, args):
    """Trading Prophet: replay the executed trades minute by minute (floating loss on equity,
    as FTMO measures it). The account replay above only checks limits when a trade closes."""
    if led is None or led.empty or args.demo:
        return None
    sys.path.insert(0, str(Path(__file__).parent / "skills" / "trading-prophet" / "scripts"))
    from ledger_adapter import evaluate_ledger
    frames = {}
    start = max(pd.Timestamp(led["time"].min()), pd.Timestamp(RP.MINUTE_START, tz="UTC"))
    for s in led["symbol"].unique():
        if s not in _BARS:
            try:
                _BARS[s] = P.bars(s, "1Min", (SPLIT - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), args.end)  # last year only: memory
            except Exception as e:  # noqa: BLE001
                print(f"  prophet: no 1-minute bars for {s}: {e}")
                _BARS[s] = None
        if _BARS[s] is not None:
            frames[s] = _BARS[s][_BARS[s].index >= start]
    try:
        res = evaluate_ledger(led, frames)
    except Exception as e:  # noqa: BLE001
        print(f"  prophet failed: {e}")
        return None
    print(f"  Trading Prophet: {res['verdict']['verdict']} {res['verdict']['reasons']}")
    return res


def _prophet_cell(p):
    """Verdict plus the reasons, so a failure says why (e.g. which FTMO limit broke and when)."""
    if not p:
        return "—"
    v = p["verdict"]
    why = "; ".join(v.get("reasons") or [])
    cont = p.get("continuous") or []
    if cont:
        last = cont[-1]
        why += f" | רציף: {last['phase']} {last['status']} {str(last['end'])[:10]}"
    roll = p.get("rolling") or {}
    if roll.get("completed"):
        why += f" | מעבר {roll.get('pass_rate_pct')}% פסילה {roll.get('breach_rate_pct')}% ({roll.get('completed')} דגימות)"
    return PROPHET_HE[v["verdict"]] + f"<div class='mut'>{html.escape(why)}</div>"


def _acc_text(w2):
    w, blown, _ = w2
    if blown is not None:
        return f"נשרף {pd.Timestamp(blown):%m/%Y}"
    return f"${w['balance'].iloc[-1]:,.0f}" if len(w) else "—"


def _risk_cell(r):
    t = (r.get("risk_tests") or {}).get(LOW_RISK)
    if not t:
        return "—"
    pr = "דולג (יותר מדי עסקאות)" if t["skipped"] else _prophet_cell(t["prophet"])
    return f"{_acc_text(t['account'])}<br>{pr}"


def _sweep_html(results):
    out = ""
    for r in results:
        if not r["sid"].startswith("TV") or not r.get("risk_tests"):
            continue
        rows = "".join(f"<tr><td>{rk * 100:g}%</td><td>{_acc_text(t['account'])}</td><td>{_prophet_cell(t['prophet'])}</td></tr>"
                       for rk, t in sorted(r["risk_tests"].items()))
        b = r["before"]
        oos_ok = b["trades"] >= 30 and b["avg_r"] == b["avg_r"] and b["avg_r"] > 0 and (b.get("t") or 0) > 1
        oos = (f"<p class='mut'>בדיקה מחוץ למדגם (לפני {SPLIT_HE}): {b['trades']:,} עסקאות, ממוצע "
               f"{b['avg_r']:+.2f}R, t={b.get('t', float('nan')):.1f} · "
               f"{'✅ רווחית גם בשנים שלא נבחרה עליהן' if oos_ok else '❌ לא רווחית באופן מובהק לפני כן'}</p>")
        out += (f"<h3>{r['sid']} · {RP.TF_HE[r['tf']]} · Trading Prophet לפי רמת סיכון</h3>{oos}<div class='tw'><table>"
                f"<tr><th>סיכון לעסקה</th><th>חשבון $100K</th><th>Trading Prophet</th></tr>{rows}</table></div>")
    return out


PROPHET_HE = {"PASS": "✅ עובר", "FAIL": "❌ נכשל", "PROVISIONAL": "🟡 בתנאי", "INSUFFICIENT": "⚪ חסר מידע"}


def export_trades(sid, tf, led):
    """Trade-by-trade table for one strategy/timeframe in the $100,000 account (1% risk)."""
    if led is None or led.empty:
        return
    ny = "Asia/Jerusalem"
    out = pd.DataFrame({
        "אסטרטגיה": f"{sid} {R.STRATEGIES[sid][0]}", "גרף": RP.TF_HE[tf], "נכס": led["symbol"],
        "כיוון": led["dir"].map({1: "קנייה", -1: "מכירה"}),
        "כניסה (שעון ישראל)": pd.to_datetime(led["time"]).dt.tz_convert(ny).dt.strftime("%d.%m.%Y %H:%M"),
        "מחיר כניסה": led["entry_px"].round(5), "סטופ": led["stop_px"].round(5), "יעד": led["target_px"].round(5),
        "יציאה (שעון ישראל)": pd.to_datetime(led["exit_time"]).dt.tz_convert(ny).dt.strftime("%d.%m.%Y %H:%M"),
        "מחיר יציאה": led["exit_px"].round(5), "סיבת יציאה": led["reason"], "תוצאה ב־R": led["r"].round(2),
        "בחשבון": led.get("taken"), "רווח/הפסד $": led.get("pnl_usd", pd.Series(dtype=float)).round(0),
        "יתרה אחרי": led.get("balance_after", pd.Series(dtype=float)).round(0)})
    path = config.OUTPUT_DIR / f"trades_{sid}_{tf}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  exported {len(out)} trades -> {path.name}")


def report(results, args):
    f2 = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.2f}"
    rows = ""
    for r in results:
        rows += (f"<tr><td>{r['sid']}</td><td>{html.escape(r['name'])}</td><td>{RP.TF_HE[r['tf']]}</td>"
                 f"<td>{r['after']['trades']:,}</td><td>{f2(r['after']['avg_r'])}</td><td>{RP._wk(r)}</td>"
                 f"<td>{'✅' if r['ok'] else '❌'}</td><td>{'—' if r['ftmo_best'] is None else RP._pct(r['ftmo_best']['p_both'])}</td>"
                 f"<td>{_prophet_cell(r.get('prophet'))}</td><td>{_risk_cell(r)}</td></tr>")
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
<th>חשבון $100K מ־{SPLIT_HE} (סיכון 1%)</th><th>שרד ורווחי</th><th>עובר אתגר 2-Step</th><th>Trading Prophet (הפסד צף דקה־דקה)</th><th>סיכון 0.75%: חשבון + Prophet</th></tr>{rows}</table></div>
{_sweep_html(results)}
<h2>הכללים שנבדקו</h2><ul>{doc}</ul>
{body}
{"" if not ERRORS else "<h2>שגיאות בהרצה</h2><pre style='white-space:pre-wrap;font-size:12px'>" + html.escape(chr(10).join(ERRORS)) + "</pre>"}
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
