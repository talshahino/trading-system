"""Stop-loss / take-profit sweep for the strategies that survived the FTMO test.

For each strategy + timeframe, the published stop is scaled (x0.5 ... x1.5) and the target is
replaced by a fixed R multiple (1R ... 5R, or the original rule). To avoid fitting to the last
year, the best setting is CHOSEN on 2016 -> 30.10.2025 only (return / drawdown in R), and only
then checked on the last year in the $100,000 FTMO account (1% risk, news filter).

  python run_sltp.py [--demo]
"""
import argparse
import html

import numpy as np
import pandas as pd

import config
from agents import ftmo_agent as F
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP

SYMBOLS = RP.FX + RP.COMMOD
COMBOS = [("TV01", "4Hour"), ("TV02", "4Hour"), ("TV04", "4Hour"), ("TV09", "1Hour"), ("TV09", "4Hour")]
STOP_MULT = [0.5, 0.75, 1.0, 1.25, 1.5]
TARGETS = ["orig", None, 1.0, 1.5, 2.0, 3.0, 5.0]   # None = no fixed target, number = R multiple
SPLIT, SPLIT_HE = RP.SPLIT, RP.SPLIT_HE


def variant(fn, sm, tg):
    def f(df):
        lo, sh, plan = fn(df)
        plan = dict(plan)
        st = plan["stop"]
        if st[0] == "atr":
            plan["stop"] = ("atr", st[1] * sm)
        elif st[0] == "lowest":                       # not used by the survivors
            plan["stop"] = st
        if plan.get("trail_atr"):
            plan["trail_atr"] = plan["trail_atr"] * sm
        if tg != "orig":
            plan["target"] = None if tg is None else ("R", tg)
            if tg is None:
                plan.pop("breakeven", None)
        return lo, sh, plan
    return f


def tlabel(t):
    return "יעד מקורי" if t == "orig" else ("בלי יעד" if t is None else f"יעד {t:g}R")


def is_stats(r):
    """In-sample: R per year, max drawdown in R (trades in exit order), return/drawdown."""
    if len(r) < 30:
        return None
    eq = r.cumsum()
    dd = float((eq.cummax() - eq).max())
    years = max((r.index.max() - r.index.min()).days / 365.25, 0.5)
    per_year = float(r.sum()) / years
    return {"n": len(r), "avg": float(r.mean()), "per_year": per_year, "dd": dd,
            "score": per_year / max(dd, 3.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-29")
    args = ap.parse_args()
    events = None if args.demo else NC.load()[0]
    rows = []
    for tf in sorted({tf for _, tf in COMBOS}):
        sids = [s for s, t in COMBOS if t == tf]
        trades = {}
        for i, s in enumerate(["DEMO1", "DEMO2"] if args.demo else SYMBOLS):
            try:
                df = RP.demo_bars(i) if args.demo else P.bars(s, tf, args.start, args.end)
            except config.TimeBudgetReached:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  {s} {tf}: NO DATA ({str(e)[-200:]})")
                continue
            if df is None or df.empty:
                continue
            cost = 0.00005 if args.demo else RP.cost_of(s)
            for sid in sids:
                for sm in STOP_MULT:
                    for tg in TARGETS:
                        t = R.run(df, variant(R.STRATEGIES[sid][1], sm, tg), cost)
                        if t.empty:
                            continue
                        if events is not None:
                            bad = NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events)
                            t = t[~bad]
                        trades.setdefault((sid, sm, tg), []).append(t.assign(symbol=s))
            print(f"  {s} {tf}: {len(df):,} bars")
        split = pd.Timestamp("2018-07-01", tz="UTC") if args.demo else SPLIT
        for (sid, sm, tg), parts in trades.items():
            tr = pd.concat(parts, ignore_index=True).sort_values("time")
            before, after = tr[tr["time"] < split], tr[tr["time"] >= split]
            ins = is_stats(before.set_index("exit_time")["r"].sort_index())
            res = {"sid": sid, "tf": tf, "sm": sm, "tg": tg, "is": ins, "oos_n": len(after),
                   "oos_avg": float(after["r"].mean()) if len(after) else np.nan}
            if len(after) >= 5:
                w, blown, _ = F.account_path(after, RP.MAIN_RISK, start=split)
                res["final"] = float(w["balance"].iloc[-1]) if len(w) else F.INITIAL
                res["peak"] = float(w["balance"].max()) if len(w) else F.INITIAL
                res["low"] = float(w["balance"].min()) if len(w) else F.INITIAL
                res["blown"] = blown
            rows.append(res)
        print(tf, "done")
    report(rows, args)


def money(v):
    return f"${v:,.0f}"


def report(rows, args):
    df = pd.DataFrame(rows)
    df.drop(columns=["is"]).to_csv(config.OUTPUT_DIR / "sltp_grid.csv", index=False, encoding="utf-8-sig")
    summary, blocks = "", ""
    for sid, tf in COMBOS:
        g = [r for r in rows if r["sid"] == sid and r["tf"] == tf]
        if not g:
            continue
        name = R.STRATEGIES[sid][0]
        base = next((r for r in g if r["sm"] == 1.0 and r["tg"] == "orig"), None)
        cand = [r for r in g if r["is"] and r["is"]["avg"] > 0]
        pick = max(cand, key=lambda r: r["is"]["score"]) if cand else None

        def acct(r):
            if r is None or "final" not in r:
                return "—"
            if r.get("blown") is not None:
                return f"<span class='bad'>נשרף {pd.Timestamp(r['blown']).strftime('%m/%Y')}</span>"
            cls = "good" if r["final"] > F.INITIAL else "bad"
            return f"<span class='{cls}'>{money(r['final'])}</span> <span class='mut'>(שיא {money(r['peak'])}, שפל {money(r['low'])})</span>"

        def setting(r):
            return f"סטופ ×{r['sm']:g}, {tlabel(r['tg'])}" if r else "—"
        best_oos = max((r for r in g if "final" in r and r.get("blown") is None), key=lambda r: r["final"], default=None)
        summary += (f"<tr><td>{sid} {html.escape(name)}</td><td>{RP.TF_HE[tf]}</td><td>{acct(base)}</td>"
                    f"<td>{setting(pick)}</td><td>{acct(pick)}</td>"
                    f"<td class='mut'>{setting(best_oos)}<br>{acct(best_oos)}</td></tr>")
        # full grid: rows = stop multiplier, columns = target
        head = "".join(f"<th>{tlabel(t)}</th>" for t in TARGETS)
        body = ""
        for sm in STOP_MULT:
            cells = ""
            for t in TARGETS:
                r = next((x for x in g if x["sm"] == sm and x["tg"] == t), None)
                if r is None:
                    cells += "<td>—</td>"
                    continue
                i = r["is"]
                ist = (f"{i['per_year']:+.1f}R/שנה · DD {i['dd']:.0f}R" if i else "מעט עסקאות")
                mark = " sel" if r is pick else (" base" if r is base else "")
                cells += f"<td class='{mark.strip()}'><div class='mut'>{ist}</div>{acct(r)}</td>"
            body += f"<tr><th>סטופ ×{sm:g}</th>{cells}</tr>"
        blocks += (f"<h2>{sid} {html.escape(name)} · {RP.TF_HE[tf]}</h2>"
                   f"<div class='scroll'><table class='grid'><tr><th></th>{head}</tr>{body}</table></div>")
    page = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>סטופ ויעד – בדיקה</title>
<style>body{{font-family:system-ui,Arial;margin:16px;background:#fff;color:#111}}table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:5px 7px;font-size:13px;vertical-align:top}}th{{background:#f1f3f6}}
.good{{color:#0a7d32;font-weight:600}}.bad{{color:#b3261e}}.mut{{color:#666;font-size:12px}}
td.sel{{background:#e6f4ea;outline:2px solid #0a7d32}}td.base{{background:#fff6d6}}.scroll{{overflow-x:auto}}</style></head><body>
<h1>האם סטופ קטן יותר / יעד גדול יותר משפרים?</h1>
<p>הסטופ המקורי הוכפל ב־0.5 עד 1.5, והיעד הוחלף ביעד קבוע ביחס לסיכון (1R = רווח בגודל הסטופ). 8 נכסים (5 צמדים, זהב, כסף, נפט), עלויות מסחר, בלי מסחר 5 דק' סביב חדשות.</p>
<p><b>כדי לא "לכייל על התשובה":</b> ההגדרה הנבחרת (ירוק) נבחרה רק לפי השנים {args.start[:4]} עד {SPLIT_HE} — הכי הרבה רווח ביחס לירידה. רק אחר כך נבדקה על השנה האחרונה בחשבון FTMO של $100,000 (סיכון 1%). צהוב = ההגדרה המקורית.</p>
<table><tr><th>אסטרטגיה</th><th>גרף</th><th>מקורי – חשבון בשנה האחרונה</th><th>הגדרה שנבחרה מראש</th><th>הנבחרת – חשבון בשנה האחרונה</th><th>הכי טוב בדיעבד (לא אמין – רק להשוואה)</th></tr>{summary}</table>
<p class='mut'>בכל משבצת בטבלאות: למעלה – רווח ממוצע לשנה וירידה מקסימלית (ב־R) בשנים שלפני; למטה – החשבון בשנה האחרונה.</p>
{blocks}</body></html>"""
    (config.OUTPUT_DIR / "report_sltp.html").write_text(page, encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
