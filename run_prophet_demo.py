"""Trading Prophet gate on the EXACT configuration the demo robot runs:
TV01 Gold Pro (stop x0.75, trailing x0.75) + TV04 Golden Edge, 4-hour UTC bars, 5 pairs + gold,
silver, oil, 1% risk per trade, shared risk rule in one account, news filter.
Also each strategy alone, for comparison.

Prophet replays the executed trades on 1-minute bars (floating equity, FTMO daily limit from
Prague midnight), runs one continuous account and a new challenge every 7 days.

  python run_prophet_demo.py   -> output/prophet_demo.md, output/prophet_demo.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import config
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
from run_sltp import variant

sys.path.insert(0, str(Path(__file__).parent / "skills" / "trading-prophet" / "scripts"))
from ledger_adapter import evaluate_ledger  # noqa: E402
import prophet  # noqa: E402

SYMBOLS = RP.FX + RP.COMMOD
DEMO = {"TV01": ("4Hour", 0.75, "orig"), "TV04": ("4Hour", 1.0, "orig")}
RISK_USD, DAILY_GUARD = 1000.0, 5000.0 * 0.9


def trades_for(sid, frames, events, start):
    tf, sm, tg = DEMO[sid]
    parts = []
    for s, df in frames.items():
        t = R.run(df, variant(R.STRATEGIES[sid][1], sm, tg), RP.cost_of(s))
        if t.empty:
            continue
        bad = NC.blocked(t["time"], s, events) | NC.blocked(t["exit_time"], s, events)
        parts.append(t[~bad].assign(symbol=s, strategy=sid))
    tr = pd.concat(parts, ignore_index=True)
    return tr[tr["time"] >= start]


def shared_ledger(tr):
    """The robot's entry rule, never stopping: skip a trade when today's loss + all open stops +
    the new stop could pass 90% of the daily limit (day = Prague calendar day)."""
    tr = tr.sort_values("time").reset_index(drop=True)
    bal, open_pos, day, day_start, rows = 100_000.0, [], None, 100_000.0, []
    for t in tr.itertuples(index=False):
        for p in sorted([p for p in open_pos if p[0] <= t.time], key=lambda p: p[0]):
            bal += p[1]
            open_pos.remove(p)
        d = t.time.tz_convert("Europe/Prague").date()
        if d != day:
            day, day_start = d, bal
        if (day_start - bal) + RISK_USD * (len(open_pos) + 1) > DAILY_GUARD:
            continue
        pnl = t.r * RISK_USD
        open_pos.append((t.exit_time, pnl))
        rows.append({**t._asdict(), "taken": "בוצעה", "risk_usd": RISK_USD, "pnl_usd": pnl})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01", help="first entry checked by Prophet (1-minute bars from here)")
    ap.add_argument("--end", default="2026-10-29")
    args = ap.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    events = NC.load()[0]
    frames, minutes = {}, {}
    for s in SYMBOLS:
        try:
            frames[s] = P.bars(s, "4Hour", "2023-01-01", args.end)
            minutes[s] = P.bars(s, "1Min", args.start, args.end)[["h", "l"]]
        except Exception as e:  # noqa: BLE001
            print(f"  {s}: NO DATA ({e})")
        print(" ", s, "loaded")
    tv = {sid: trades_for(sid, frames, events, start) for sid in DEMO}
    configs = {"TV01 + TV04 (מה שרץ בדמו)": pd.concat(tv.values(), ignore_index=True),
               "TV01 לבד (סטופ x0.75)": tv["TV01"], "TV04 לבד": tv["TV04"]}
    md, js = [], {}
    for name, tr in configs.items():
        led = shared_ledger(tr)
        frames_used = {s: minutes[s] for s in led["symbol"].unique() if s in minutes}
        res = evaluate_ledger(led, frames_used)
        js[name] = res
        md.append(prophet.render_he(res, name))
        print(name, res["verdict"]["verdict"], res["verdict"]["reasons"], "rolling", res["rolling"])
    (config.OUTPUT_DIR / "prophet_demo.md").write_text("\n\n".join(md), encoding="utf-8")
    (config.OUTPUT_DIR / "prophet_demo.json").write_text(json.dumps(js, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print("written")


if __name__ == "__main__":
    main()
