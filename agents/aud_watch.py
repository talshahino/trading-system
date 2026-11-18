"""Bot 7 - the Australian dollar's day after its releases, watched on the side with no money (user, 18.11.2026: "add"
- agents/news_drift.py: the one way that held on both periods - AUD's high-impact releases, 30 minutes after the
release with the number, stop 1 x the daily ATR(14), out after 24 hours: 2025 +7.4R / 112, 2026 +36.6R / 206; 72
hours also positive on both). Every AUD release from 18.11.2026 on every AUD instrument, exactly the study's rules
and FTMO's costs (spread + commission + swap), the open ones at their price now. Written to
reports/agents/bot7/aud_watch.json - shown in bot 7's page. Nothing trades.

  python -m agents.aud_watch
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import ftmo_costs as C  # noqa: E402
from agents import news_fast as F  # noqa: E402
from agents import news_surprise as NS  # noqa: E402
from agents import poc_agent as P  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "aud_watch.json"
FROM = pd.Timestamp("2026-11-18", tz="UTC")
HOLDS = (24, 72)
BACKTEST = {"24": {"y2025": {"n": 112, "sum": 7.41}, "y2026": {"n": 206, "sum": 36.6}},
            "72": {"y2025": {"n": 112, "sum": 12.4}, "y2026": {"n": 206, "sum": 25.77}}}


def main():
    try:
        NS.CAL_DAYS = 0                                    # the newest actuals
        NS.calendar_full(fetch=True)
    except Exception as e:  # noqa: BLE001
        print("calendar:", e)
    rel = {t: x for t, x in (F.releases().get("AUD") or {}).items() if t >= FROM}
    syms = [s for s in F.plan() if "AUD" in F.betas(s)]
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for sym in syms:
        b = F.betas(sym)["AUD"]
        try:
            d1 = P.bars(sym, "1Min", (FROM - pd.Timedelta(days=40)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
            sp = P.spread_1m(sym, (FROM - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        except Exception as e:  # noqa: BLE001
            print(sym, e)
            continue
        if d1 is None or d1.empty:
            continue
        d1 = d1[~d1.index.duplicated()].sort_index()
        idx = d1.index
        o, h, l, c = (d1[x].to_numpy(float) for x in "ohlc")
        dd = d1.resample("1D").agg({"h": "max", "l": "min", "c": "last"}).dropna()
        tr = pd.concat([dd["h"] - dd["l"], (dd["h"] - dd["c"].shift()).abs(), (dd["l"] - dd["c"].shift()).abs()], axis=1).max(axis=1)
        atr_d = tr.rolling(14).mean()
        raw = []
        for t, x in sorted(rel.items()):
            ka = atr_d.index.searchsorted(t.floor("D")) - 1
            ke = idx.searchsorted(t + pd.Timedelta(minutes=30))
            if ka < 14 or ke >= len(idx) or idx[ke] - t > pd.Timedelta(minutes=35):
                continue
            atr = float(atr_d.iloc[ka])
            dr = int(x["g"] * b) or 1
            e = o[ke]
            stop = e - dr * atr
            for hh in HOLDS:
                end = idx.searchsorted(idx[ke] + pd.Timedelta(hours=hh))
                done = end < len(idx)
                end = min(end, len(idx))
                seg = l[ke:end] if dr > 0 else h[ke:end]
                hit = np.flatnonzero((seg <= stop) if dr > 0 else (seg >= stop))
                if len(hit):
                    q = ke + int(hit[0])
                    ex, why, done = (stop if q == ke else (min(stop, o[q]) if dr > 0 else max(stop, o[q]))), "stop", True
                else:
                    q, ex, why = end - 1, c[end - 1], ("time" if done else "open")
                raw.append({"hold": hh, "time": idx[ke], "exit_time": idx[q], "dir": dr, "entry_px": e, "stop_px": stop,
                            "r": dr * (ex - e) / atr, "why": why, "rel": t, "title": x["title"], "good": int(x["g"]),
                            "big": int(x["big"])})
        if not raw:
            continue
        tt = pd.DataFrame(raw)
        net = C.apply(tt.copy(), sym, sp if sp is not None and len(sp) else None)
        for k, z in enumerate(net.itertuples()):
            rows.append({"sym": sym, "hold": int(z.hold), "release": pd.Timestamp(z.rel).isoformat(), "title": z.title,
                         "good": z.good, "big": z.big, "dir": int(z.dir), "entry_t": pd.Timestamp(z.time).isoformat(),
                         "exit_t": pd.Timestamp(z.exit_time).isoformat(), "entry": round(float(z.entry_px), 6),
                         "stop": round(float(z.stop_px), 6), "r": round(float(z.r), 3), "status": z.why})
    tot = {}
    for hh in HOLDS:
        xs = [r for r in rows if r["hold"] == hh]
        cl = [r["r"] for r in xs if r["status"] != "open"]
        tot[str(hh)] = {"n": len(xs), "closed": len(cl), "sum": round(sum(cl), 2),
                        "open_r": round(sum(r["r"] for r in xs if r["status"] == "open"), 2),
                        "mean": round(float(np.mean(cl)), 3) if cl else None,
                        "win": round(100 * float(np.mean(np.array(cl) > 0)), 1) if cl else None,
                        "bt_mean": round(BACKTEST[str(hh)]["y2026"]["sum"] / BACKTEST[str(hh)]["y2026"]["n"], 3)}
    doc = {"at": now.isoformat(timespec="seconds"), "from": FROM.isoformat(), "releases": len(rel), "symbols": syms,
           "rules": "AUD, פרסום בהשפעה גבוהה: 30 דקות אחרי הפרסום בכיוון הנתון, סטופ ATR יומי אחד, יציאה אחרי 24 / 72 שעות",
           "backtest": BACKTEST, "total": tot, "trades": sorted(rows, key=lambda r: r["entry_t"], reverse=True)}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tot), len(rel), "releases")


if __name__ == "__main__":
    sys.exit(main())
