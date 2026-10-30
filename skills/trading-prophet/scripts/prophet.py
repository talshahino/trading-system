#!/usr/bin/env python3
"""Trading Prophet - run an FTMO verdict on one strategy.

Examples:
  python prophet.py --report Bot_EURUSD.htm --bars bars/ --news news.csv
  python prophet.py --equity equity.csv --preset ftmo_2step_100k
Outputs a Hebrew Markdown report (stdout or --out) and a JSON verdict (--json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from equity import Bars, build_closed_only, build_from_trades, load_equity_csv  # noqa: E402
from ftmo_sim import Simulator, summarize_rolling, verdict  # noqa: E402
from mt5_deals import parse_report  # noqa: E402
from news import load_news, violations  # noqa: E402

PRESETS = HERE.parent / "config" / "ftmo_presets.json"

_HE_STATUS = {
    "passed": "עבר ✅", "breach_daily": "נפסל – הפסד יומי ❌", "breach_max": "נפסל – הפסד כולל ❌",
    "timeout": "נגמר הזמן ⏱", "incomplete": "לא הסתיים (נגמר המידע)", "alive": "חי",
}
_HE_VERDICT = {
    "PASS": "✅ עובר – מוכן לשלב הדמו",
    "FAIL": "❌ נכשל – האסטרטגיה נעצרת",
    "PROVISIONAL": "🟡 עובר בתנאי – חסר מידע על הפסד צף (צריך נרות דקה)",
    "INSUFFICIENT": "⚪ אין מספיק מידע להחלטה",
}


def load_config(preset: str) -> tuple[dict, dict]:
    cfg = json.loads(PRESETS.read_text(encoding="utf-8"))
    return cfg[preset], cfg["prophet_policy"]


def evaluate(report=None, equity_csv=None, bars_dir=None, news_csv=None,
             preset="ftmo_2step_100k", report_tz="Europe/Athens", scale_mode="relative",
             policy_overrides: dict | None = None) -> dict:
    rules, policy = load_config(preset)
    policy = {**policy, **(policy_overrides or {})}
    trades, warnings = [], []
    if equity_csv:
        series = load_equity_csv(equity_csv)
    else:
        parsed = parse_report(report, report_tz)
        warnings += parsed.warnings
        trades = parsed.trades
        if parsed.mode == "closed_only":
            series = build_closed_only(parsed.closed_only_points, parsed.initial_deposit)
        else:
            bars = Bars.from_dir(bars_dir, report_tz) if bars_dir else None
            initial = parsed.initial_deposit or 0.0
            series = build_from_trades(trades, initial, bars)
    hits = []
    if news_csv and trades:
        hits = violations(trades, load_news(news_csv, policy["news_min_impact"]),
                          policy["news_blackout_minutes"])
    elif not news_csv:
        warnings.append("no news calendar supplied: news blackout rule not checked")
    return evaluate_series(series, trades, rules, policy, preset, scale_mode, hits, warnings)


def evaluate_series(series, trades, rules, policy, preset, scale_mode="relative",
                    hits=None, warnings=None) -> dict:
    """Core: run the simulator on an already-built equity series."""
    hits, warnings = hits or [], list(warnings or [])
    sim = Simulator(series, rules, scale_mode)
    cont = sim.challenge(0) if series.points else None
    track = sim.weekly_track(policy["continuous_horizon_weeks"])
    roll = summarize_rolling(sim.rolling(policy["rolling_start_every_days"]))
    v = verdict(series, cont, track, roll, len(hits), policy) if cont else \
        {"verdict": "INSUFFICIENT", "reasons": ["no data"], "warnings": []}
    v["warnings"] = warnings + v.get("warnings", [])
    return {
        "preset": preset, "data_quality": series.quality, "trades": len(trades),
        "verdict": v, "rolling": roll, "weekly_track": track,
        "continuous": [p.__dict__ for p in cont.phases] if cont else [],
        "news_violations": hits[:50], "news_violation_count": len(hits),
    }


def render_he(res: dict, name: str) -> str:
    v, roll = res["verdict"], res["rolling"]
    out = [f"# Trading Prophet – {name}", "", f"## פסק דין: {_HE_VERDICT[v['verdict']]}", ""]
    if v["reasons"]:
        out += ["**סיבות:**"] + [f"- {r}" for r in v["reasons"]] + [""]
    out += ["## חשבון רציף מהיום הראשון", "", "| שלב | תוצאה | ימי מסחר | ימים קלנדריים | ניצול הפסד יומי מקס' | ניצול הפסד כולל מקס' | רווח |",
            "|---|---|---|---|---|---|---|"]
    for p in res["continuous"]:
        out.append(f"| {p['phase']} | {_HE_STATUS[p['status']]} | {p['trading_days']} | "
                   f"{p['calendar_days']:.0f} | {p['worst_daily_use_pct']}% | {p['worst_total_use_pct']}% | {p['final_pnl_pct']}% |")
    out += ["", "## אתגר חדש בכל שבוע (בדיקת מזל תאריך ההתחלה)", "",
            f"- דגימות: {roll['samples']} (הסתיימו: {roll['completed']}, לא הסתיימו: {roll['incomplete']})",
            f"- אחוז מעבר: **{roll['pass_rate_pct']}%**",
            f"- אחוז פסילה: **{roll['breach_rate_pct']}%** {roll['breach_reasons'] or ''}",
            f"- חציון ימים עד מעבר: {roll['median_days_to_pass']}", ""]
    track = res["weekly_track"]
    if track:
        out += ["## דגימה שבועית – חשבון אחד עד שנה", "", "| שבוע | מצב | רווח מצטבר |", "|---|---|---|"]
        for w in track:
            pnl = "—" if w["pnl_pct"] is None else f"{w['pnl_pct']}%"
            out.append(f"| {w['week']} | {_HE_STATUS[w['status']]} | {pnl} |")
        out.append("")
    out.append(f"**חדשות:** {res['news_violation_count']} פעולות בתוך חלון 5 הדקות סביב חדשות חשובות")
    out.append(f"**איכות נתונים:** {res['data_quality']} · **עסקאות:** {res['trades']}")
    if v["warnings"]:
        out += ["", "**אזהרות:**"] + [f"- {w}" for w in v["warnings"]]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--report", help="MT5 Strategy Tester report (.htm)")
    src.add_argument("--equity", help="CSV time,balance,equity[,open_count]")
    ap.add_argument("--bars", help="folder of 1-minute bar CSVs per symbol")
    ap.add_argument("--news", help="news CSV time,currency,impact[,title] (UTC)")
    ap.add_argument("--preset", default="ftmo_2step_100k")
    ap.add_argument("--report-tz", default="Europe/Athens", help="timezone of MT5 server times")
    ap.add_argument("--scale", choices=["relative", "fixed"], default="relative")
    ap.add_argument("--out", help="write Hebrew Markdown here")
    ap.add_argument("--json", help="write full JSON result here")
    a = ap.parse_args(argv)
    res = evaluate(a.report, a.equity, a.bars, a.news, a.preset, a.report_tz, a.scale)
    name = Path(a.report or a.equity).stem
    md = render_he(res, name)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(res, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if res["verdict"]["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
