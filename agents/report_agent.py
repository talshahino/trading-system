"""Report agent: writes a one-page Hebrew HTML verdict plus the raw CSV."""
import html
import math
import pandas as pd

import config


def _fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "int":
        return f"{int(v):,}"
    return f"{v:,.1f}"


def _table(df, cols, kinds=None):
    kinds = kinds or {}
    head = "".join(f"<th>{html.escape(h)}</th>" for h in cols.values())
    body = ""
    for _, r in df.iterrows():
        body += "<tr>" + "".join(
            f"<td>{_fmt(r[c], kinds.get(c, 'num')) if not isinstance(r[c], str) else html.escape(r[c])}</td>"
            for c in cols) + "</tr>"
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _curve_svg(curve):
    if curve is None or len(curve) < 2:
        return ""
    y = curve.values.tolist()
    lo, hi = min(y + [0]), max(y + [0])
    span = (hi - lo) or 1
    w, h = 600, 180
    pts = " ".join(f"{i / (len(y) - 1) * w:.1f},{h - (v - lo) / span * h:.1f}" for i, v in enumerate(y))
    zero = h - (0 - lo) / span * h
    color = "var(--good)" if y[-1] > 0 else "var(--bad)"
    return (f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='none' class='curve'>"
            f"<line x1='0' x2='{w}' y1='{zero:.1f}' y2='{zero:.1f}' class='zero'/>"
            f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'/></svg>"
            f"<p class='cap'>רווח מצטבר בתקופת המבחן בדולרים, אחרי עלויות: ${y[-1]:,.0f}</p>")


def TRACK_NAME():
    from agents.backtest_agent import TRACK
    return next(k for k, v in config.TRACKS.items() if v is TRACK)


def verdict(wf):
    if not wf or not wf.get("best"):
        return "no-data", "אין מספיק עסקאות כדי לבחור כלל. צריך יותר מניות או תקופה ארוכה יותר."
    t = wf["test"]
    if wf.get("uses_llm") and t["trades"] < config.MIN_TRADES_FOR_RULE:
        return "weak", (f"אין מספיק עסקאות אחרי תאריך האימון של המודל ({config.MODEL_KNOWLEDGE_CUTOFF}) "
                        f"כדי לבדוק בצורה נקייה ({t['trades']} עסקאות, צריך לפחות {config.MIN_TRADES_FOR_RULE}). "
                        "הפתרון: יותר מניות ברשימה, או לחכות לעוד חודשים של נתונים.")
    if t["trades"] < config.MIN_TRADES_FOR_RULE:
        return "weak", f"הכלל נבחר, אבל בתקופת המבחן היו רק {t['trades']} עסקאות. מוקדם לקבוע."
    luck = wf.get("luck")
    lucky = luck is not None and luck["p"] > config.MAX_LUCK_P
    if t["avg_bps"] > 0 and (t["t_stat"] or 0) >= 2 and lucky:
        return "weak", (f"הכלל הרוויח בתקופת המבחן, אבל בבדיקת המזל {luck['p'] * 100:.0f}% מהחיפושים על נתונים "
                        "אקראיים מצאו כלל טוב לפחות כמוהו. לא מספיק כדי לעבור לדמו.")
    if t["avg_bps"] > 0 and (t["t_stat"] or 0) >= 2:
        return "pass", "הכלל שנבחר על העבר המשיך להרוויח גם בתקופה שלא ראה, אחרי עלויות. מועמד לשלב הדמו."
    if t["avg_bps"] > 0:
        return "weak", "רווחי בתקופת המבחן, אבל לא מספיק מובהק סטטיסטית (t<2). ייתכן שזה מזל."
    return "fail", "הכלל שעבד על העבר לא החזיק בתקופת המבחן. לא עובר לדמו."


def _bands_html(bands):
    from agents.features_agent import SCORE_COLUMNS
    out = ""
    for col, (tbl, slope) in bands.items():
        if tbl is None or tbl.empty:
            continue
        name = SCORE_COLUMNS.get(col, col)
        if isinstance(slope, float) and not math.isnan(slope):
            word = "עולה" if slope > 0 else "יורד"
            line = f"כל +10 נקודות ב{name}: הרווח הממוצע לעסקה {word} ב־{abs(slope):.1f} bps."
        else:
            line = "אין מספיק עסקאות לחישוב מגמה."
        out += (f"<h3>{name} (0–100)</h3><p class='mut'>{line}</p>" +
                _table(tbl, {"band": "ציון", "trades": "עסקאות", "avg_bps": "ממוצע bps נטו",
                             "win_rate": "הצלחה", "t_stat": "t"},
                       {"trades": "int", "win_rate": "pct"}))
    return out


def _checks_html(wf, checks):
    rows = []
    # 1. Model knowledge cutoff
    if wf and wf.get("uses_llm"):
        line = (f"תקופת המבחן מתחילה אחרי {config.MODEL_KNOWLEDGE_CUTOFF}, תאריך סוף נתוני האימון של המודל. "
                + ("נקודת החלוקה הוזזה לשם כדי למנוע תוצאות מנופחות." if wf.get("split_moved") else ""))
        c = wf.get("contamination")
        if c and c["before"]["trades"] and c["after"]["trades"]:
            b, a = c["before"]["avg_bps"], c["after"]["avg_bps"]
            line += (f" אותו כלל: {b:,.1f} bps לעסקה לפני התאריך ({c['before']['trades']} עסקאות), "
                     f"{a:,.1f} bps אחריו ({c['after']['trades']} עסקאות).")
            if b > 0 and a < b / 2:
                line += " <b>ירידה חדה אחרי התאריך: ייתכן שהמודל \"זכר\" חלק מהתוצאות. סמוך רק על התקופה שאחרי.</b>"
        rows.append(("הטיית זיכרון של המודל", line))
    else:
        rows.append(("הטיית זיכרון של המודל", "לא רלוונטי: הסיווג נעשה במילות מפתח ולא ב־Claude."))
    # 1b. Multiple testing
    luck = (wf or {}).get("luck")
    if luck:
        p = luck["p"]
        word = ("סביר שזה לא מזל." if p <= config.MAX_LUCK_P else
                "<b>ייתכן מאוד שזה מזל: חיפוש על נתונים אקראיים מוצא כללים כאלה לעתים קרובות.</b>")
        rows.append(("בדיקת מזל (ריבוי בדיקות)", (
            f"נבדקו {wf['configs_tried']} כללים, ולכן הטוב ביניהם ייראה טוב גם במקרה. כל החיפוש חזר על עצמו "
            f"{luck['runs']} פעמים על נתוני האימון כשכיוון כל חדשה הוגרל באקראי. "
            f"ב־{luck['hits']} מהן נמצא כלל טוב לפחות כמו האמיתי: <b>סיכוי למזל {p * 100:.1f}%</b> "
            f"(הסף: {config.MAX_LUCK_P * 100:g}%). {word}")))
    # 2. Survivorship
    sv = checks.get("survivorship")
    if sv:
        rows.append(("מניות שנמחקו", sv))
    # 3. Cost
    cost = checks.get("cost")
    if cost:
        rows.append(("עלות Claude", (
            f"{cost['calls']:,} קריאות, ממוצע {cost['avg_input_tokens']:,.0f} טוקנים נכנסים ו־"
            f"{cost['avg_output_tokens']:,.0f} יוצאים. ${cost['avg_cost_per_call'] * 1000:,.3f} לאלף ידיעות. "
            f"בקצב של {cost['events_per_month']:,.0f} ידיעות בחודש: <b>כ־${cost['cost_per_month']:,.2f} לחודש</b>"
            f"{' (פחות מדולר)' if cost['cost_per_month'] < 1 else ''}. "
            + (f"זמן תגובה חציוני {cost['median_latency_ms']:,.0f} מ״ש (90% מתחת ל־{cost['p90_latency_ms']:,.0f}), "
               "וזה חלק מההשהיה בין הפרסום לכניסה."
               if not math.isnan(cost['median_latency_ms']) else "")
        )))
    else:
        rows.append(("עלות Claude", "לא נמדדה בהרצה הזו (מילות מפתח, או שהכול הגיע מהקאש)."))
    return "".join(f"<div class='chk'><b>{html.escape(k)}</b><p>{v}</p></div>" for k, v in rows)


def _latency_html(latency):
    if not latency:
        return ""
    tbl, info = latency
    head = "<h2>כמה מהר צריך להיות?</h2>"
    if tbl is None or tbl.empty:
        return head + "<p class='mut'>אין עדיין מדידות של עסקאות בודדות.</p>"
    note = (f"לכל חדשה עם |ציון| ≥ {info['min_abs_score']} נמדד המחיר מעסקאות בודדות ברגע הפרסום ואחרי 1–60 שניות. "
            "כל השורות יוצאות באותה שעה, כך שההבדל ביניהן הוא רק כמה מאוחר נכנסנו. "
            "<b>0 שנ׳ = המחיר ברגע הפרסום עצמו: גבול עליון תיאורטי</b>, אי אפשר באמת לקנות בו אחרי שקוראים את החדשה. "
            f"נמדדו {info['measured']:,} מתוך {info['events']:,} חדשות"
            + (" (השאר יימדדו בהרצה הבאה)." if info.get("stopped") else "."))
    out = head + f"<p class='mut'>{note}</p>"
    for h in sorted(tbl["h"].unique()):
        t = tbl[tbl["h"] == h].sort_values("lat").copy()
        t["lat_he"] = t["lat"].map(lambda x: f"{int(x)} שנ׳")
        out += (f"<h3>יציאה {h} דק׳ אחרי הכניסה המקורית</h3>" +
                _table(t, {"lat_he": "כניסה אחרי", "events": "חדשות", "gross_bps": "תנועה bps",
                           "net_bps": "נטו bps", "t_stat": "t", "strong_events": "חזקות (≥0.5)",
                           "strong_net_bps": "נטו חזקות bps"},
                       {"events": "int", "strong_events": "int"}))
    return out


def write_report(results, wf, react, cats, mode_label, period, bands=None, checks=None, latency=None):
    from agents.backtest_agent import TRACK
    csv_path = config.OUTPUT_DIR / f"events_{TRACK_NAME()}.csv"
    results.to_csv(csv_path, index=False)

    status, text = verdict(wf)
    traded = results["entry_px"].notna().sum()
    skips = results["skip_reason"].value_counts().to_dict()
    skip_txt = " · ".join(f"{k}: {v}" for k, v in skips.items())

    rule_html = ""
    if wf and wf.get("best"):
        b = wf["best"]
        rule_html = (f"<p>הכלל שנבחר: כניסה בכיוון הסנטימנט כש־|ציון| ≥ {b['threshold']}, "
                     f"ביטחון ≥ {b['min_conf']}, {'רק מידע חדש, ' if b['require_new'] else ''}"
                     f"{('רק חדשות מסוג ' + html.escape(b['category']) + ', ') if b.get('category') else ''}"
                     f"{config.EXIT_LABELS_HE.get(b.get('exit', 'time'), '')}, "
                     f"החזקה מקסימלית {b['horizon']} {TRACK['unit_he']}. נבדקו {wf['configs_tried']} שילובים.</p>")
        cmp = pd.DataFrame([{"period": "אימון (בחירת הכלל)", **wf["train"]},
                            {"period": "מבחן (לא נראה)", **wf["test"]}])
        rule_html += _table(cmp, {"period": "תקופה", "trades": "עסקאות", "avg_bps": "ממוצע bps",
                                  "win_rate": "אחוז הצלחה", "t_stat": "t", "total_pct": "סה״כ %",
                                  "worst_bps": "הגרועה bps"},
                            {"trades": "int", "win_rate": "pct"})
        if wf.get("exit_compare") is not None:
            ec = wf["exit_compare"]
            rule_html += ("<p class='mut'>אותו כלל בתקופת המבחן, בכל אחת משיטות היציאה:</p>" +
                          _table(ec, {"exit_he": "יציאה", "trades": "עסקאות", "avg_bps": "ממוצע bps",
                                      "win_rate": "הצלחה", "t_stat": "t", "worst_bps": "הגרועה bps"},
                                 {"trades": "int", "win_rate": "pct"}))
        u = wf["test_usd"]
        halt = (f" <b>המסחר נעצר ב־{u['halted_on']} אחרי ירידה של {config.MAX_DRAWDOWN_PCT * 100:g}% מיתרת הפתיחה.</b>"
                if u.get("halted_on") else "")
        rule_html += (f"<p>חשבון דמו של ${config.EQUITY:,.0f} לפי כללי הסיכון (סיכון בסיס {config.RISK_PCT * 100:g}% לעסקה, "
                      f"עצירה יומית ב־{config.DAILY_LOSS_PCT * 100:g}%, עצירה כללית בהפסד של {config.MAX_DRAWDOWN_PCT * 100:g}% מיתרת הפתיחה): "
                      f"רווח/הפסד בתקופת המבחן <b>${u['pnl_usd']:,.0f}</b> "
                      f"(יתרה ${u['end_equity']:,.0f}), ירידה מקסימלית ${u['max_drawdown_usd']:,.0f}, "
                      f"{u['trades_taken']} עסקאות, גודל ממוצע ${u['avg_size_usd']:,.0f}, "
                      f"ימים שנעצרו במגבלה היומית: {u['days_stopped']}.{halt}</p>")
        rule_html += _curve_svg(u.get("curve_usd"))

    react_cols = {"band": "סנטימנט 0–100", "events": "אירועים"}
    from agents.backtest_agent import TRACK
    react_cols.update({f"h{h}_avg_bps": f"{h} {TRACK['unit_he']}" for h in TRACK["horizons"]})
    cat_html = _table(cats, {"category": "קטגוריה", "trades": "עסקאות", "avg_bps": "ממוצע bps",
                             "win_rate": "הצלחה", "t_stat": "t"},
                      {"trades": "int", "win_rate": "pct"}) if not cats.empty else ""

    page = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>News Backtest {TRACK['unit']}</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e3de;--good:#1f8a4c;--bad:#c0392b;--warn:#b7791f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--card:#1e1e1c;--line:#33332f;--good:#4cc27e;--bad:#ef6b5b;--warn:#e0a84a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}
main{{max-width:760px;margin:auto;padding:20px 16px 40px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} h3{{font-size:15px;margin:18px 0 2px}}
.mut,.cap{{color:var(--mut);font-size:13px}}
.v{{border-radius:10px;padding:14px 16px;margin:16px 0;background:var(--card);border-inline-start:5px solid var(--warn)}}
.v.pass{{border-color:var(--good)}} .v.fail{{border-color:var(--bad)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:start;white-space:nowrap}}
.chk{{background:var(--card);border-radius:8px;padding:10px 14px;margin:8px 0}} .chk p{{margin:4px 0 0;font-size:14px}}
.curve{{width:100%;height:180px;background:var(--card);margin-top:10px}} .zero{{stroke:var(--line)}}
</style></head><body><main>
<h1>בקטסט סיגנלי חדשות · מסלול {html.escape(TRACK['label'])}</h1>
<p class="mut">{html.escape(mode_label)} · {html.escape(period)} · השהיית כניסה {config.ENTRY_LATENCY_SEC} שנ׳ · עלות {config.COST_BPS_LIQUID:g}–{config.COST_BPS_ILLIQUID:g} bps לפי נזילות</p>
<div class="v {status}"><b>שורה תחתונה:</b> {text}</div>
<h2>בדיקות אמינות</h2>
{_checks_html(wf, checks or {})}
<h2>האם המניות זזות לפי הסנטימנט?</h2>
<p class="mut">0 = שלילי מאוד, 50 = ניטרלי, 100 = חיובי מאוד. תנועה ממוצעת אחרי הכניסה, בלי עלויות{' (ביחס לשוק)' if TRACK['unit'] == 'd' else ''}. אם יש סיגנל, המספרים יעלו ככל שהציון עולה.</p>
{_table(react, react_cols, {"events": "int"})}
{_latency_html(latency)}
<h2>כלל מסחר: אימון מול מבחן</h2>
{rule_html or "<p>אין מספיק נתונים.</p>"}
<h2>איפה הכלל עובד? פילוח לפי ציונים</h2>
<p class="mut">כל העסקאות של הכלל שנבחר, על כל התקופה, מחולקות לפי ציון 0–100. 100 = גדולה/נזילה/תנודתית/חזקה ביותר.</p>
{_bands_html(bands or {})}
<h2>לפי סוג חדשה ({TRACK['default_h']} {TRACK['unit_he']}, ציון ≥ 0.5)</h2>
{cat_html or "<p class='mut'>אין פילוח קטגוריות במצב מילות מפתח.</p>"}
<h2>נתונים</h2>
<p class="mut">{len(results):,} אירועים, {traded:,} נמדדו. דולגו: {html.escape(skip_txt) or "—"}.</p>
</main></body></html>"""
    path = config.OUTPUT_DIR / f"report_{TRACK_NAME()}.html"
    path.write_text(page, encoding="utf-8")
    return path, csv_path
