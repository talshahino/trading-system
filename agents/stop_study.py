"""Tight stops and costs in bots 5 and 6 (user, 16.11.2026: SC326|5Min|GBPUSD in bot 5 has a 3-pip stop, the spread +
commission are ~23% of its risk - "check every tight-stop strategy in bots 5 and 6, whether a slightly wider stop
leaves more profit after the costs").

Each live scout strategy of the bot at stop x0.75 / x1 (as it trades) / x1.25 / x1.5 of its own stop, everything else as
it trades (its own target in price, breakeven at 60% of the way + costs, entry at the next open, FTMO's costs, the live
rules). The risk in dollars stays the same per trade, so R summed is money: a wider stop = a smaller position, the same
costs in dollars become a smaller share of the risk, the same target becomes fewer R.
  bot 5: run_rules' exit grid (agent_portfolio.run_rules, "exits") - on GitHub, one machine per instrument
  bot 6: its own futures test already holds the stop grid (reports/agents/bot6/scout_fut.json) - read only
Chosen on 30.04.2025-31.12.2025 (bot 6: 2024-2025), shown on 2026. Nothing in the bots changes here.

  python -m agents.stop_study plan
  python -m agents.stop_study run --sym XAUUSD --minutes 100
  python -m agents.stop_study report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "stop_study"
STOPS = (0.75, 1.0, 1.25, 1.5)


def scout(bot):
    return (json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8")).get("scout") or {})


def plan():
    return sorted({c.split("|")[2] for c in scout("bot5")})


def _stats(g):
    if g.empty:
        return {"n": 0, "sum": 0.0, "mean": None, "win": None}
    return {"n": int(len(g)), "sum": round(float(g["r"].sum()), 2), "mean": round(float(g["r"].mean()), 3),
            "win": round(100 * float((g["r"] > 0).mean()), 1)}


def run(sym, minutes):
    import agent_portfolio as A
    A.BOT_NOW["bot"] = "bot5"
    combos = [c for c in scout("bot5") if c.split("|")[2] == sym]
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    one = [c for c in combos if c.split("|")[1] == "1Min"]
    for grp in [[c for c in combos if c not in one]] + [[c] for c in one]:
        if not grp:
            continue
        try:
            t, _ = A.run_rules(grp, "exits", minutes)
        except Exception as e:  # noqa: BLE001
            for c in grp:
                out[c] = {"error": str(e)[-300:]}
            continue
        tt = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        t = t[tt >= pd.Timestamp(A.TEST_FROM, tz="UTC")].copy()
        tt = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        risk = (t["entry_px"] - t["stop_px"]).abs()
        t["cost_r"] = (t["dir"] * (t["exit_px"] - t["entry_px"]) / risk - t["r"]).clip(lower=0)
        t["per"] = np.where(tt < A.T_SPLIT, "train", "unseen")
        for c in grp:
            sid, tf, s = c.split("|")
            tc = t[(t["sid"] == sid) & (t["tf"] == tf) & (t["symbol"] == s)]
            row = {}
            for sc in STOPS:
                g = tc[tc["variant"] == f"own|be0.6|stop{sc:g}|entry0"]
                row[f"{sc:g}"] = {"train": _stats(g[g["per"] == "train"]), "unseen": _stats(g[g["per"] == "unseen"]),
                                  "cost_r": round(float(g["cost_r"].median()), 3) if len(g) else None}
            out[c] = row
    (OUT / f"{sym}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False)[:1500])


def _pick(rows):
    """The stop chosen on the choosing period: the most R summed (at least 10 trades), x1 when none is better."""
    ok = {k: v for k, v in rows.items() if (v["train"]["n"] or 0) >= 10}
    if not ok:
        return "1"
    return max(ok, key=lambda k: (ok[k]["train"]["sum"], k == "1"))


def report():
    out = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "bot5": {}, "bot6": {}}
    for f in sorted(OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        for c, rows in json.loads(f.read_text(encoding="utf-8")).items():
            if "error" in rows:
                out["bot5"][c] = rows
                continue
            k = _pick(rows)
            out["bot5"][c] = {"stops": rows, "pick": k,
                              "gain_unseen": round(rows[k]["unseen"]["sum"] - rows["1"]["unseen"]["sum"], 2)}
    # bot 6: its own test's stop grid around the variant it trades
    sf = json.loads((ROOT / "reports" / "agents" / "bot6" / "scout_fut.json").read_text(encoding="utf-8"))
    for c, live in scout("bot6").items():
        v = dict((live or {}).get("v") or {})
        d = (sf.get(c) or {}).get("variants") or {}
        rows = {}
        for x in d.values():
            xv = x.get("v") or {}
            if all(xv.get(k) == v.get(k) for k in v if k != "stop") and all(xv.get(k) == v.get(k) for k in xv if k != "stop"):
                tr, un = x.get("train") or {}, x.get("unseen") or {}
                rows[f"{float(xv['stop']):g}"] = {
                    "train": {"n": tr.get("n"), "sum": round((tr.get("n") or 0) * (tr.get("mean") or 0), 2), "mean": tr.get("mean")},
                    "unseen": {"n": un.get("n"), "sum": x.get("unseen_sum"), "mean": un.get("mean")}, "cost_r": x.get("cost_r")}
        if "1" not in rows and rows:
            out["bot6"][c] = {"stops": rows, "note": "the traded stop is not x1"}
        elif rows:
            k = _pick(rows)
            out["bot6"][c] = {"stops": rows, "pick": k, "now": f"{float(v.get('stop', 1)):g}",
                              "gain_unseen": round((rows[k]["unseen"]["sum"] or 0) - (rows[f"{float(v.get('stop', 1)):g}"]["unseen"]["sum"] or 0), 2)}
        else:
            out["bot6"][c] = {"missing": True}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# the structural stop (user, 16.11.2026, task b5stop1611): the scout trades that were stopped and then went on to the
# target - "the stop beyond the third swing, the sellers are stronger" - on exactly those strategies, the stop beyond
# the 2nd / 3rd swing of the strategy's own timeframe (+0 / 0.1 ATR), the target in price as it trades
# (run_rules --manage --manage-grid swing, rules_agent.swing_stop); chosen on 30.04-31.12.2025, shown on 2026.
SWING_COMBOS = ["SC077|5Min|US30", "SC326|5Min|GBPUSD", "SC326|15Min|USDCHF", "SC326|15Min|USDJPY"]
SWING_OUT = OUT / "swing"


def plan_swing():
    return sorted({c.split("|")[2] for c in SWING_COMBOS if c in scout("bot5")})


def run_swing(sym, minutes):
    import agent_portfolio as A
    A.BOT_NOW["bot"] = "bot5"
    grp = [c for c in SWING_COMBOS if c.split("|")[2] == sym and c in scout("bot5")]
    SWING_OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    try:
        t, _ = A.run_rules(grp, "manage", minutes, extra=["--manage-grid", "swing"])
    except Exception as e:  # noqa: BLE001
        out = {c: {"error": str(e)[-300:]} for c in grp}
        t = None
    if t is not None:
        tt = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        t = t[tt >= pd.Timestamp(A.TEST_FROM, tz="UTC")].copy()
        tt = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        t["per"] = np.where(tt < A.T_SPLIT, "train", "unseen")
        for c in grp:
            sid, tf, s = c.split("|")
            tc = t[(t["sid"] == sid) & (t["tf"] == tf) & (t["symbol"] == s)]
            base = tc[tc["variant"] == "sw0"]
            # stopped in the entry's own bar (held 0), and how the same entries ended with each structural stop
            same = base[(base["r"] < 0) & (base.get("held", pd.Series(1, index=base.index)) == 0)]
            row = {"same_bar_stops": {"n": int(len(same)), "r": round(float(same["r"].sum()), 2)}}
            for v, g in tc.groupby("variant"):
                row[v] = {"train": _stats(g[g["per"] == "train"]), "unseen": _stats(g[g["per"] == "unseen"])}
                if v != "sw0" and len(same):
                    m = g[g["time"].isin(same["time"])]
                    row[v]["same_bar_then"] = {"n": int(len(m)), "r": round(float(m["r"].sum()), 2),
                                               "win": int((m["r"] > 0).sum())}
            out[c] = row
    (SWING_OUT / f"{sym}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False)[:1500])


def report_swing():
    out = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "combos": {}}
    for f in sorted(SWING_OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        for c, rows in json.loads(f.read_text(encoding="utf-8")).items():
            if "error" in rows:
                out["combos"][c] = rows
                continue
            vs = {k: v for k, v in rows.items() if k.startswith("sw")}
            ok = {k: v for k, v in vs.items() if (v["train"]["n"] or 0) >= 10}
            pick = max(ok, key=lambda k: (ok[k]["train"]["sum"], k == "sw0")) if ok else "sw0"
            b = vs.get("sw0") or {"train": {"sum": 0}, "unseen": {"sum": 0}}
            out["combos"][c] = {"rows": rows, "pick": pick,
                                "gain_train": round(vs[pick]["train"]["sum"] - b["train"]["sum"], 2) if pick in vs else 0.0,
                                "gain_unseen": round(vs[pick]["unseen"]["sum"] - b["unseen"]["sum"], 2) if pick in vs else 0.0}
    SWING_OUT.mkdir(parents=True, exist_ok=True)
    (SWING_OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report", "plan_swing", "run_swing", "report_swing"])
    ap.add_argument("--sym", default="")
    ap.add_argument("--minutes", type=float, default=100)
    a = ap.parse_args()
    if a.mode == "plan":
        json.dump(plan(), sys.stdout)
    elif a.mode == "run":
        run(a.sym, a.minutes)
    elif a.mode == "plan_swing":
        json.dump(plan_swing(), sys.stdout)
    elif a.mode == "run_swing":
        run_swing(a.sym, a.minutes)
    elif a.mode == "report_swing":
        print(json.dumps(report_swing(), ensure_ascii=False, indent=1)[:6000])
    else:
        print(json.dumps(report(), ensure_ascii=False, indent=1)[:6000])
