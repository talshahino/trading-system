"""Bot 5's gap method by track (pair of timeframes) x instrument (user, 18.11.2026: "the agent should tell apart the
kinds of instruments and the strategies by timeframe - 5 minutes against 1, an hour against 15 or against 1 - against
the instruments too, and know statistically which instrument is good for which track and its entries; there we lower
what is weaker and raise what is better").

Per (track, instrument):
  backtest  the track as it runs, FTMO's costs (agents/tal_wait_study "now" rows): 30.04-31.12.2025 and 2026 apart
  live      bot 5's own closed trades of the combination (counted in the account or not), and Tal's answers
            ("היית נכנס?" yes / no) on its setups
  verdict   decided by the LIVE trades against the history (user, 18.11.2026: "raising and lowering happen when live
            trades succeed against the history and against the live statistics"): until 20 live trades - "wait";
            then the live mean against the backtest's (2025 + 2026 together), z = (live - backtest) / (live sd / sqrt n):
            raise (x1.5) = live > 0, the backtest > 0 on both periods and live not below it (z >= -1);
            lower (x0.5) = live < 0 and clearly below the backtest (z <= -2); else "hold".
The suggestion is a card for Tal - nothing changes by itself. (A choice on the backtest alone barely carried from 2025
into 2026: R per unit of risk 0.131 -> 0.137, the drawdown up.)

  python -m agents.tal_combo [--state live/watch_own_state.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WAIT = ROOT / "reports" / "agents" / "bot5" / "tal_wait"
OUT = ROOT / "reports" / "agents" / "bot5" / "combo_stats.json"
LIVE_MIN = 20
SPLIT, FROM = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2025-04-30", tz="UTC")
PAIR_HE = {"t15_1": "15 דק' / דקה", "t15_5": "15 דק' / 5 דק'", "t5_1": "5 דק' / דקה", "t1h_15": "שעה / 15 דק'",
           "t1h_5": "שעה / 5 דק'", "t4h_1h": "4 שעות / שעה", "t4h_15": "4 שעות / 15 דק'", "t4h_5": "4 שעות / 5 דק'"}


def _st(s):
    s = pd.Series(s, dtype=float)
    return {"n": int(len(s)), "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 3) if len(s) else None,
            "win": round(100 * float((s > 0).mean()), 1) if len(s) else None}


def main(state_path=None):
    rows = []
    for f in sorted(WAIT.glob("*.json")):
        if f.name == "report.json":
            continue
        for tk, w in json.loads(f.read_text(encoding="utf-8")).items():
            for x in (w.get("now") if isinstance(w.get("now"), list) else []):
                rows.append({"trk": tk.split("@")[0], "sym": f.stem, "t": pd.Timestamp(x["time"]), "r": float(x["r"])})
    bt = pd.DataFrame(rows)
    bt = bt[bt["t"] >= FROM]
    live, marks = {}, {}
    try:
        st = json.loads(Path(state_path or ROOT / "live" / "watch_own_state.json").read_text(encoding="utf-8"))
        for t in st.get("watch") or []:
            if not t.get("tal") or str(t["tal"]).startswith("sc:"):
                continue
            k = (str(t["tal"]).split("@")[0], t["symbol"])
            if t.get("status") == "closed" and t.get("r") is not None:
                live.setdefault(k, []).append(float(t["r"]))
            m = (t.get("mark") or {}).get("take")
            if m in ("yes", "no"):
                marks.setdefault(k, []).append(m)
    except Exception:  # noqa: BLE001
        pass
    out = []
    for (trk, sym), g in bt.groupby(["trk", "sym"]):
        a, b = _st(g[g["t"] < SPLIT]["r"]), _st(g[g["t"] >= SPLIT]["r"])
        lv = _st(live.get((trk, sym), []))
        mk = marks.get((trk, sym), [])
        ls = pd.Series(live.get((trk, sym), []), dtype=float)
        btm = float(g["r"].mean()) if len(g) else 0.0
        z = None
        if len(ls) >= 2 and ls.std() > 0:
            z = round(float((ls.mean() - btm) / (ls.std() / len(ls) ** 0.5)), 2)
        ready = lv["n"] >= LIVE_MIN
        strong = ready and lv["mean"] > 0 and (a["mean"] or 0) > 0 and (b["mean"] or 0) > 0 and z is not None and z >= -1
        weak = ready and lv["mean"] < 0 and z is not None and z <= -2
        out.append({"track": trk, "pair": PAIR_HE.get(trk, trk), "sym": sym, "y2025": a, "y2026": b, "live": lv,
                    "tal_yes": mk.count("yes"), "tal_no": mk.count("no"),
                    "bt_mean": round(btm, 3), "z": z,
                    "verdict": "strong" if strong else "weak" if weak else ("hold" if ready else "wait"),
                    "suggest": 1.5 if strong else 0.5 if weak else 1.0})
    out.sort(key=lambda x: -((x["y2025"]["mean"] or 0) + (x["y2026"]["mean"] or 0)))
    doc = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "rows": out,
           "check": {"note": "מגדילים או מקטינים לפי העסקאות החיות מול ההיסטוריה: אחרי 20 עסקאות חיות בצירוף. "
                             "הגדלה: החי ברווח ולא נופל מההיסטוריה. הקטנה: החי בהפסד ונמוך בבירור מההיסטוריה."}}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    for x in out:
        print(f"{x['track']:7} {x['sym']:7} 25:{x['y2025']['n']:3} {x['y2025']['mean']} 26:{x['y2026']['n']:3} {x['y2026']['mean']} live {x['live']['n']} {x['live']['mean']} {x['verdict']}")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="")
    main(ap.parse_args().state or None)
