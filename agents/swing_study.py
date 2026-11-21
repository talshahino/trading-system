"""The bots as an FTMO Swing account (user, 21.11.2026 21:48: "everything" - the improvement checks, #5): Swing has no
news restriction and no weekend / 2+ hour break restriction, at leverage 1:30 (FX) / 1:15 (indices, metals) / 1:1
(stocks, crypto) - the margin check (21.11) left bots 1 and 4 out (their stocks need more margin than the account has
about a fifth / a fifteenth of the time), so bots 2, 3, 5b (scout) and 7 here.
Every combo of the bot through the bots' backtest machine (run_rules.py) two ways:
  funded   the live rules as now (agent_portfolio.LIVE_RULES - out before a break, out before a restricted event)
  swing    the same without --break-exit, --stock-session-exit, --news-exit, --news-minute (FTMO's crypto Saturday
           maintenance stays - it applies to Swing too)
The trades (time, exit, r) per way are stored; the account replay (4.5% / 9% from the initial amount) is done after.
Results only - nothing in the bots changes.
  python -m agents.swing_study plan | run --job bot2:EURUSD | report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent_portfolio as AP  # noqa: E402

OUT = ROOT / "reports" / "agents" / "swing"
DROP = {"--break-exit", "--stock-session-exit", "--news-exit", "--news-minute"}
BOTS = ("bot2", "bot3", "bot5", "bot7")


def combos():
    out = {}
    for b in BOTS:
        cfg = json.loads((ROOT / "live" / "bots" / f"{b}.json").read_text(encoding="utf-8"))
        cs = list(cfg.get("scout") or {}) if b == "bot5" else list(cfg.get("combos") or [])
        for c in cs:
            out.setdefault((b, c.split("|")[2]), []).append(c)
    return out


def plan():
    return [f"{b}:{s}" for (b, s) in sorted(combos())]


def _run(b, sym, cs, rules):
    od = ROOT / "output"
    od.mkdir(exist_ok=True)
    tag = sym.replace("/", "-")
    f = od / f"combo_trades_{tag}.csv.gz"
    f.unlink(missing_ok=True)
    cf = od / f"sw_{tag}.txt"
    cf.write_text("\n".join(sorted(set(cs))), encoding="utf-8")
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [sys.executable, "-u", "run_rules.py", "--market", "ftmo", "--start", AP.START, "--end", end, "--test-from",
           AP.TEST_FROM, "--combos-file", str(cf), "--symbols", sym] + rules
    AP.BOT_NOW["bot"] = b
    pf = AP.plans_file(b, od)
    if pf is not None:
        cmd += ["--plans-file", str(pf)]
    r = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, MAX_RUN_MINUTES="70"), capture_output=True, text=True)
    if not f.exists():
        print(b, sym, "no trades file", (r.stdout or "")[-400:], (r.stderr or "")[-400:], flush=True)
        return None
    return pd.read_csv(f)


def run(job):
    b, sym = job.split(":")
    cs = combos()[(b, sym)]
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {}
    for way, rules in (("funded", AP.LIVE_RULES), ("swing", [x for x in AP.LIVE_RULES if x not in DROP])):
        t = _run(b, sym, cs, list(rules))
        if t is None or t.empty:
            continue
        for c in cs:
            sid, tf, _ = c.split("|")
            g = t[(t["sid"].astype(str) == sid) & (t["tf"].astype(str) == tf)]
            if "variant" in g and len(g) and g["variant"].nunique() > 1:
                g = g[g["variant"] == g["variant"].mode().iloc[0]]
            doc.setdefault(c, {})[way] = [[str(a), str(z), float(r)] for a, z, r in
                                          zip(g["time"], g["exit_time"].fillna(g["time"]), g["r"])]
        print(job, way, {c: len(v.get(way, [])) for c, v in doc.items()}, flush=True)
    (OUT / f"{b}_{sym.replace('/', '-')}.json").write_text(json.dumps({"bot": b, "combos": doc}), encoding="utf-8")


def report():
    tot = {}
    for f in sorted(OUT.glob("bot*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for c, ws in d["combos"].items():
            for w, xs in ws.items():
                for a, z, r in xs:
                    t = pd.Timestamp(a)
                    t = t.tz_localize("UTC") if t.tzinfo is None else t
                    if t < pd.Timestamp(AP.TEST_FROM, tz="UTC"):
                        continue
                    y = "2026" if t >= AP.T_SPLIT else "2025"
                    k = tot.setdefault(d["bot"], {}).setdefault(w, {}).setdefault(y, {"n": 0, "sum_r": 0.0})
                    k["n"] += 1
                    k["sum_r"] = round(k["sum_r"] + r, 2)
    (OUT / "report.json").write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                                 "total": tot}, indent=1), encoding="utf-8")
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--job", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(plan()))
    elif a.mode == "run":
        run(a.job)
    else:
        report()
