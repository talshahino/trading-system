"""Bots 1-2's crypto before FTMO's Saturday maintenance (user, 18.11.2026: "yes" - bot 1's QC02 ETHUSD was closed
14.11 05:45 "market break ahead" and the backtest held it to +8.1R: FTMO's crypto stops Saturday 08:00-12:00 server
time, the 24/7 data the backtest uses has no such gap, so the backtest never closed there - live/runner.break_close
does). Every crypto combo of bots 1 and 2 through the bots' backtest machine (run_rules.py, the live rules):
  hold       as the backtest is now (no Saturday close - what the bots' choice was made on)
  close      closed before the Saturday break as live (--crypto-break)
  noentryH   closed there and no new trade H hours before it (H = 4, 8, 12, 24)
Per combo: 30.04-31.12.2025 and 2026 apart. "close" is the honest number of the bot as it trades; the noentry rows say
whether a trade opened just before the break is worth taking.
  python -m agents.crypto_break_study
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent_portfolio as AP  # noqa: E402

OUT = ROOT / "reports" / "agents" / "crypto_break.json"
WAYS = {"hold": [], "close": ["--crypto-break"]} | {f"noentry{h}": ["--crypto-break", "--crypto-noentry", str(h)] for h in (4, 8, 12, 24)}


def combos():
    from agents import ftmo_universe as U
    out = {}
    for b in ("bot1", "bot2"):
        cfg = json.loads((ROOT / "live" / "bots" / f"{b}.json").read_text(encoding="utf-8"))
        for c in cfg.get("combos") or []:
            if U.CLASS.get(c.split("|")[2]) == "crypto":
                out.setdefault(c.split("|")[2], []).append((b, c))
    return out


def run(sym, cs, extra):
    od = ROOT / "output"
    od.mkdir(exist_ok=True)
    tag = sym.replace("/", "-")
    f = od / f"combo_trades_{tag}.csv.gz"
    f.unlink(missing_ok=True)
    cf = od / f"cb_{tag}.txt"
    cf.write_text("\n".join(sorted({c for _, c in cs})), encoding="utf-8")
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [sys.executable, "-u", "run_rules.py", "--market", "ftmo", "--start", AP.START, "--end", end, "--test-from",
           AP.TEST_FROM, "--combos-file", str(cf), "--symbols", sym] + AP.LIVE_RULES + extra
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, MAX_RUN_MINUTES="60"), capture_output=True, text=True)
    if not f.exists():
        print(sym, extra, "no trades file", (r.stdout or "")[-500:], (r.stderr or "")[-500:], flush=True)
        return None
    return pd.read_csv(f)


def _st(s):
    return {"n": int(len(s)), "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 3) if len(s) else None}


def main():
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "combos": {}, "total": {}}
    for sym, cs in combos().items():
        for w, extra in WAYS.items():
            t = run(sym, cs, extra)
            if t is None or t.empty:
                continue
            tt = pd.to_datetime(t["time"], utc=True, format="ISO8601")
            for b, c in cs:
                sid, tf, _ = c.split("|")
                g = t[(t["sid"].astype(str) == sid) & (t["tf"].astype(str) == tf)]
                if "variant" in g and len(g) and g["variant"].nunique() > 1:
                    g = g[g["variant"] == g["variant"].mode().iloc[0]]
                gt = tt[g.index]
                a = g[(gt >= pd.Timestamp(AP.TEST_FROM, tz="UTC")) & (gt < AP.T_SPLIT)]["r"]
                z = g[gt >= AP.T_SPLIT]["r"]
                k = f"{b}:{c}"
                rep["combos"].setdefault(k, {})[w] = {"y2025": _st(a), "y2026": _st(z)}
                T = rep["total"].setdefault(b, {}).setdefault(w, {"y2025": 0.0, "y2026": 0.0})
                T["y2025"] = round(T["y2025"] + float(a.sum()), 2)
                T["y2026"] = round(T["y2026"] + float(z.sum()), 2)
            print(sym, w, {k: v.get(w) for k, v in rep["combos"].items() if sym in k}, flush=True)
            OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep["total"], indent=1))


if __name__ == "__main__":
    main()
