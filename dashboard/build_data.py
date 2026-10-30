"""Turn the robot's account state (branch live, live/state.json) into the dashboard's data files.

  python dashboard/build_data.py <state.json> <out_dir>

Writes status.json, equity.json, trades.json, positions.json, monthly.json, journal.json
(arrays of row objects). Times are shown in Israel time.
"""
import csv
import io
import json
import sys
from pathlib import Path

import pandas as pd

IL = "Asia/Jerusalem"
RISK_USD = 1000.0          # 1% of $100,000 per trade
DAILY, TOTAL = 0.05, 0.10


def il(ts):
    if not ts:
        return ""
    return pd.Timestamp(ts).tz_convert(IL).strftime("%Y-%m-%d %H:%M")


def main(src, out):
    s = json.loads(Path(src).read_text(encoding="utf-8"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    init = float(s.get("initial") or 100000)
    bal, eq, ds = float(s["balance"]), float(s["equity"]), float(s.get("day_start") or s["balance"])
    trades = s.get("trades") or []
    pos = s.get("positions") or []
    status = [{
        "updated_utc": s.get("updated_utc") or "",
        "updated_il": il(s.get("updated_utc")),
        "balance": round(bal, 2), "equity": round(eq, 2),
        "profit": round(bal - init, 2), "profit_pct": round((bal - init) / init * 100, 2),
        "open_profit": round(eq - bal, 2),
        "day_loss": round(max(0.0, ds - eq), 2), "day_limit": init * DAILY,
        "total_room": round(eq - init * (1 - TOTAL), 2),
        "open_positions": len(pos), "closed_trades": len(trades),
        "state": "stopped" if s.get("halted_forever") else ("paused today" if s.get("halted_today") else "running"),
        "strategies": s.get("strategies", ""),
    }]
    eqc = [{"time": il(p["t"]), "balance": p["balance"], "equity": p["equity"]} for p in s.get("equity_curve") or []]
    tr = [{"id": str(t["id"]), "strategy": t["strategy"], "symbol": t["symbol"],
           "dir": "קנייה" if t["dir"] == "buy" else "מכירה", "open_il": il(t["open_utc"]),
           "close_il": il(t["close_utc"]), "open_price": t["open_price"], "close_price": t["close_price"],
           "volume": t["volume"], "profit": round(float(t["profit"]), 2),
           "r": round(float(t["profit"]) / RISK_USD, 2)} for t in trades]
    tr.sort(key=lambda r: r["close_il"], reverse=True)
    ps = [{"id": str(p["id"]), "strategy": p["strategy"], "symbol": p["symbol"],
           "dir": "קנייה" if p["dir"] == "buy" else "מכירה", "open_il": il(p["open_utc"]),
           "open_price": p["open_price"], "stop": p["stop"], "target": p["target"] or None,
           "volume": p["volume"], "profit": round(float(p["profit"]), 2)} for p in pos]
    months = {}
    for t in tr:
        m = t["close_il"][:7]
        r = months.setdefault(m, {"month": m, "profit": 0.0, "trades": 0, "wins": 0, "TV01": 0.0, "TV04": 0.0})
        r["profit"] += t["profit"]
        r["trades"] += 1
        r["wins"] += t["profit"] > 0
        if t["strategy"] in ("TV01", "TV04"):
            r[t["strategy"]] += t["profit"]
    monthly = [{**r, "profit": round(r["profit"], 2), "TV01": round(r["TV01"], 2), "TV04": round(r["TV04"], 2)}
               for r in sorted(months.values(), key=lambda r: r["month"])]
    jr = []
    for line in s.get("journal") or []:
        row = next(csv.reader(io.StringIO(line)), [])
        if len(row) < 9:
            continue
        note = row[8]
        kind = ("נפתחה" if note == "opened" else "דולגה: חדשות" if "news" in note else
                "דולגה: חוק הסיכון המשותף" if "shared risk" in note else
                "דולגה: המסחר נעצר" if "stopped" in note else
                "עצירה" if note.startswith("STOP") else note)
        t = row[0].replace(".", "-")
        jr.append({"time": il(pd.Timestamp(t, tz="UTC")) if t else "", "strategy": row[1], "symbol": row[2],
                   "dir": {"1": "קנייה", "-1": "מכירה"}.get(row[3], ""), "event": kind})
    jr.reverse()
    for i, r in enumerate(jr):
        r["n"] = str(len(jr) - i)
    for name, rows in [("status", status), ("equity", eqc), ("trades", tr), ("positions", ps),
                       ("monthly", monthly), ("journal", jr)]:
        (out / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print("built", {k: len(v) for k, v in [("equity", eqc), ("trades", tr), ("positions", ps), ("journal", jr)]})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
