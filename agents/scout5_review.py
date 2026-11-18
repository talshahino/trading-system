"""Bot 5ב from the start (user, 18.11.2026: "a re-check now of all of 5ב's strategies from the start, then each on its
own and all together with combinations, and then we talk again" - after the whole-bot replay at 0.4% a strategy fell
53% from its peak in 2026).

The 30 scout strategies' trades as 5ב trades them (agents/conflict_study run for bot5: entry, exit, direction, R with
the bot's management and costs). Periods: 30.04-31.12.2025 (chooses) and 2026 (only shows).
  each      per strategy: trades, R a trade, win %, total R, its own deepest fall in R, and whether 2025 and 2026 agree
  together  all of them in one account, risk scale-free: total / deepest fall (both in R units of one trade's risk), the
            worst day, and the risk a trade that keeps the deepest fall at 8% and the worst day at 3% (FTMO 10% / 5%)
  combos    a set chosen on 2025 only: strategies added one by one while the account's total / deepest fall (2025)
            improves (positive on 2025 only) - shown on 2026 at the risk that keeps 2025's fall at 8%
  python -m agents.scout5_review
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import conflict_study as CS
import agent_portfolio as A

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "scout5_review.json"
DD_CAP, DAY_CAP = 8.0, 3.0


def _mi(s):
    return int((pd.Timestamp(s, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1))


def acct(t):
    """Trades (a, b, r) -> scale-free account in R: total, deepest fall, worst day (Prague day of the exit)."""
    if t.empty:
        return {"n": 0, "total": 0.0, "dd": 0.0, "worst_day": 0.0, "ratio": None}
    day = pd.to_datetime(t["b"] * 60, unit="s", utc=True).dt.tz_convert("Europe/Prague").dt.date
    d = t.groupby(day)["r"].sum().sort_index()
    eq = np.r_[0.0, d.cumsum().to_numpy()]
    dd = float((np.maximum.accumulate(eq) - eq).max())
    tot = float(d.sum())
    return {"n": int(len(t)), "total": round(tot, 1), "dd": round(dd, 1), "worst_day": round(float(min(0.0, d.min())), 1),
            "ratio": round(tot / dd, 2) if dd > 0 else None}


def risk_for(a):
    """The risk a trade (in %) that keeps the deepest fall at DD_CAP and the worst day at DAY_CAP."""
    caps = [DD_CAP / a["dd"] if a["dd"] > 0 else 1.0, DAY_CAP / -a["worst_day"] if a["worst_day"] < 0 else 1.0]
    return round(min(caps), 3)


def main():
    t = CS.load("bot5")
    sp, fr = _mi(A.SPLIT), _mi(A.TEST_FROM)
    t = t[t["a"] >= fr].copy()
    tr, un = t[t["a"] < sp], t[t["a"] >= sp]
    each = []
    for c, g in t.groupby("combo"):
        a, b = g[g["a"] < sp], g[g["a"] >= sp]
        ea = {"combo": c, "sym": c.split("|")[2], "y2025": acct(a), "y2026": acct(b),
              "mean25": round(float(a["r"].mean()), 3) if len(a) else None,
              "mean26": round(float(b["r"].mean()), 3) if len(b) else None,
              "win25": round(100 * float((a["r"] > 0).mean()), 1) if len(a) else None,
              "win26": round(100 * float((b["r"] > 0).mean()), 1) if len(b) else None}
        ea["agree"] = bool((ea["mean25"] or 0) > 0 and (ea["mean26"] or 0) > 0)
        each.append(ea)
    each.sort(key=lambda x: -(x["y2025"]["total"]))
    allx = {"y2025": acct(tr), "y2026": acct(un)}
    allx["risk_pct"] = risk_for(allx["y2025"])
    allx["y2026_at_risk"] = {k: round(v * allx["risk_pct"], 2) if isinstance(v, float) and k != "ratio" else v
                             for k, v in allx["y2026"].items()}
    # greedy combination on 2025
    pool = [x["combo"] for x in each if x["y2025"]["total"] > 0]
    chosen, best = [], None
    while True:
        cand = None
        for c in pool:
            if c in chosen:
                continue
            a = acct(tr[tr["combo"].isin(chosen + [c])])
            if a["ratio"] is not None and (best is None or a["ratio"] > best["ratio"]):
                cand, best_c = c, a
        if cand is None:
            break
        chosen.append(cand)
        best = best_c
        if len(chosen) >= 30:
            break
    combo = {"strategies": chosen, "y2025": acct(tr[tr["combo"].isin(chosen)]), "y2026": acct(un[un["combo"].isin(chosen)])}
    combo["risk_pct"] = risk_for(combo["y2025"])
    combo["y2026_at_risk"] = {k: round(v * combo["risk_pct"], 2) if isinstance(v, float) and k != "ratio" else v
                              for k, v in combo["y2026"].items()}
    # the strategies positive on 2025 (no greedy) - a simpler set
    pos = [x["combo"] for x in each if (x["mean25"] or 0) > 0]
    posx = {"strategies": pos, "y2025": acct(tr[tr["combo"].isin(pos)]), "y2026": acct(un[un["combo"].isin(pos)])}
    posx["risk_pct"] = risk_for(posx["y2025"])
    posx["y2026_at_risk"] = {k: round(v * posx["risk_pct"], 2) if isinstance(v, float) and k != "ratio" else v
                             for k, v in posx["y2026"].items()}
    # per instrument together
    by_sym = {s: {"strategies": int(g["combo"].nunique()), "y2025": acct(g[g["a"] < sp]), "y2026": acct(g[g["a"] >= sp])}
              for s, g in t.groupby(t["combo"].str.split("|").str[2])}
    # risk shared by instrument (14 of the 30 are on gold - their losses come on the same days): every instrument one
    # budget split among its strategies (per_sym), 1/sqrt(n), gold at half / a third - the ratio on 2025 chooses
    t["sym"] = t["combo"].str.split("|").str[2]
    nsym = t.groupby("sym")["combo"].nunique()
    weights = {"equal": lambda s_: 1.0, "per_sym": lambda s_: 1.0 / nsym[s_], "sqrt": lambda s_: 1.0 / nsym[s_] ** 0.5,
               "gold_half": lambda s_: 0.5 if s_ == "XAUUSD" else 1.0, "gold_third": lambda s_: 1 / 3 if s_ == "XAUUSD" else 1.0}
    split_ = {}
    for nm, wf in weights.items():
        u = t.copy()
        u["r"] = u["r"] * u["sym"].map(wf)
        a, b = acct(u[u["a"] < sp]), acct(u[u["a"] >= sp])
        rk = risk_for(a)
        split_[nm] = {"y2025": a, "y2026": b, "unit_risk_pct": rk,
                      "y2025_pct": round(a["total"] * rk, 1), "y2026_pct": round(b["total"] * rk, 1),
                      "y2026_dd_pct": round(b["dd"] * rk, 1), "y2026_worst_day_pct": round(b["worst_day"] * rk, 1)}
    best_split = max(split_, key=lambda k: split_[k]["y2025"]["ratio"] or 0)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "strategies": len(each), "each": each,
           "risk_split": split_, "risk_split_best_2025": best_split,
           "all": allx, "greedy": combo, "positive_2025": posx, "by_symbol": by_sym}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for x in each:
        print(f"{x['combo']:22} 25: n{x['y2025']['n']:4} m{x['mean25']} tot{x['y2025']['total']:7} dd{x['y2025']['dd']:6} | 26: n{x['y2026']['n']:4} m{x['mean26']} tot{x['y2026']['total']:7} dd{x['y2026']['dd']:6} {'OK' if x['agree'] else ''}")
    for k in ("all", "greedy", "positive_2025"):
        v = rep[k]
        print(k, len(v.get("strategies", [])) or "", "25", v["y2025"], "26", v["y2026"], "risk%", v["risk_pct"], "26@risk", v["y2026_at_risk"])
    for s, v in by_sym.items():
        print(s, v)


if __name__ == "__main__":
    main()
