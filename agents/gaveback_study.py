"""Trades that were in profit and ended in a loss - managing them in EVERY market (user, 16.11.2026: "almost every trade
in bot 5 today was in profit at some point inside the trade's range, and in the end almost all lost"; "I am not
willing to have such a day again - trades that reach 50-60% of the target and leave at a loss"; "maybe take them out
partially, as I keep telling you". Tasks tal1611gaveback (bot 5) and tal1611partial (bots 1-4)).

agents/chop_study.py tested the same management only while the 1-hour chart was choppy, and it lost. Here the rule
acts in every market, on the same trades (the strategies as each bot trades them now, agents/conflict_study.py run,
from 30.04.2025, with their prices and the instrument's 5-minute bars - the artifacts of a chop_study run):
  asis          as today (the bots' own management already inside: break-even at 60% of the way, or at 2.5R with
                +1R locked at 3R when there is no target)
  be@X          once the trade is X R in profit, its stop moves to the entry                       X = 0.5 / 1 / 1.5 / 2
  half@X        once X R in profit, half is taken there and the rest's stop moves to the entry    X = 0.5 / 1 / 1.5 / 2
  bew@W         a trade with a target: the stop to the entry at W of the way to the target         W = 0.5 / 0.6
  halfw@W       a trade with a target: half out at W of the way, the rest's stop to the entry      W = 0.5 / 0.6
  lock@S        stepped locking: from +S R in profit the stop is at the entry, and every further S R it locks S R
                more (S = 1: +2R locks +1R, +3R locks +2R ...; S = 0.5: +1R locks +0.5R ...)
The rule acts from the 5-minute bar after the one that reached the level; the trade's exit bar is left out (the worse
case); the costs stay the trade's own. Whole bot with the fixed risk rule (portfolio_check.replay); chosen on
30.04-31.12.2025, shown on 2026; per strategy too, and "each strategy its best way on 2025" shown on 2026 (the honest
test of a per-strategy pick). The census: how many losing trades had first reached +0.5R / +1R / half the way.
Nothing in the bots changes here.

  python -m agents.gaveback_study report --bot bot1,bot2,bot3,bot4,bot5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import conflict_study as CS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "gaveback_study.json"
WAYS = (["asis"] + [f"{k}@{x:g}" for k in ("be", "half") for x in (0.5, 1, 1.5, 2)]
        + [f"{k}@{x:g}" for k in ("bew", "halfw") for x in (0.5, 0.6)] + ["lock@1", "lock@0.5"])


def _span(x, m):
    return np.searchsorted(m, x["a"]), np.searchsorted(m, x["b"] - 5)


def manage(x, bars, kind, th):
    """One trade under one rule, in every market -> its new r."""
    m, h, lo = bars
    i, j = _span(x, m)
    if j <= i:
        return x["r"]
    d, e, rk = x["dir"], x["entry_px"], x["risk_px"]
    if kind in ("bew", "halfw"):
        tg = x["target_px"]
        if not np.isfinite(tg) or d * (tg - e) <= 0:
            return x["r"]
        th = th * d * (tg - e) / rk                     # W of the way, in R
        kind = "be" if kind == "bew" else "half"
    fav = ((h[i:j] - e) if d > 0 else (e - lo[i:j])) / rk
    if kind == "lock":
        adv = ((lo[i:j] - e) if d > 0 else (e - h[i:j])) / rk     # the bar's worst point, in R
        best, lock = -np.inf, None
        for q in range(j - i):
            if lock is not None and adv[q] <= lock:
                return lock - x["cost_r"]
            best = max(best, fav[q])
            if best >= th:
                lock = np.floor(best / th) * th - th
        return x["r"]
    trig = np.flatnonzero(fav >= th)
    if not len(trig):
        return x["r"]
    k = i + trig[0] + 1
    back = bool((lo[k:j] <= e).any()) if d > 0 else bool((h[k:j] >= e).any())
    rest = 0.0 if back else x["r"] + x["cost_r"]
    if kind == "be":
        return rest - x["cost_r"]
    return 0.5 * th + 0.5 * rest - x["cost_r"]


def apply(t, way, bars):
    out = []
    for x in t.to_dict("records"):
        x["share"] = 1.0
        if way != "asis" and x["sym"] in bars:
            kind, th = way.split("@")
            x["r"] = manage(x, bars[x["sym"]], kind, float(th))
        out.append(x)
    return out


def census(t):
    """The losing trades: how many first went +0.5R / +1R / half the way to the target in their favour (mfe_r)."""
    lo = t[t["r"] < 0]
    if lo.empty:
        return {"losers": 0}
    way = np.where(np.isfinite(lo["target_px"]), (lo["dir"] * (lo["target_px"] - lo["entry_px"]) / lo["risk_px"]), np.nan)
    mf = lo["mfe_r"].astype(float)
    w = mf / way
    return {"trades": int(len(t)), "losers": int(len(lo)), "lost_r": round(float(lo["r"].sum()), 1),
            "mfe_0.5R": round(100 * float((mf >= 0.5).mean()), 1), "mfe_1R": round(100 * float((mf >= 1).mean()), 1),
            "with_target": int(np.isfinite(way).sum()),
            "half_way": round(100 * float((w[np.isfinite(w)] >= 0.5).mean()), 1) if np.isfinite(w).any() else None,
            "mfe_median": round(float(mf.median()), 2)}


def bot_report(bot):
    from agents import portfolio_check as PC
    import agent_portfolio as A
    t = CS.load(bot)
    bars = {}
    for f in CS.out_dir(bot).glob("*.bars.csv.gz"):
        b = pd.read_csv(f)
        bars[f.name.split(".bars")[0]] = (b["m"].to_numpy(), b["h"].to_numpy(), b["l"].to_numpy())
    t = t[t["sym"].isin(bars)]
    cfg = PC.bot_cfg(bot)
    if isinstance(cfg.get("risk"), dict):          # bot 5: a risk per strategy ("scout:<combo>")
        rk = cfg["risk"]
        risk = 1.0
        mults = {c: float(rk.get("scout:" + c) or rk.get(c) or 0.0018) for c in t["combo"].unique()}
    else:
        risk, mults = PC.bot_risk(bot), PC.mults(bot)
    split = int((pd.Timestamp(A.SPLIT, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1))
    per = {"train": t[t["a"] < split], "unseen": t[t["a"] >= split]}
    res = {"strategies": int(t["combo"].nunique()), "symbols": sorted(bars),
           "census": {p: census(tt) for p, tt in per.items()}, "ways": {}, "per_strategy": {}}
    sims = {}
    for way in WAYS:
        row = {}
        for p, tt in per.items():
            s = apply(tt, way, bars)
            sims[(way, p)] = s
            rep = PC.replay(CS._book(s, mults), risk)
            row[p] = {k: rep[k] for k in ("trades", "total", "worst_day", "max_dd")}
        res["ways"][way] = row
    # per strategy: R in each period under each way, and each strategy's own best way on the choosing period
    ps = {}
    for way in WAYS:
        for p in per:
            for x in sims[(way, p)]:
                d = ps.setdefault(x["combo"], {}).setdefault(way, {"train": [0, 0.0], "unseen": [0, 0.0]})[p]
                d[0] += 1
                d[1] += x["r"]
    pick = {}
    for c, ws in ps.items():
        best = max(WAYS, key=lambda w: ws.get(w, {}).get("train", [0, -1e9])[1])
        pick[c] = best if ws[best]["train"][1] > ws["asis"]["train"][1] + 1e-9 else "asis"
        res["per_strategy"][c] = {"pick": pick[c], **{w: {p: [v[0], round(v[1], 2)] for p, v in ws[w].items()}
                                                      for w in ("asis", pick[c])}}
    row = {}
    for p, tt in per.items():
        s = []
        for c, g in tt.groupby("combo"):
            s += apply(g, pick.get(c, "asis"), bars)
        rep = PC.replay(CS._book(sorted(s, key=lambda x: (x["a"], x["b"])), mults), risk)
        row[p] = {k: rep[k] for k in ("trades", "total", "worst_day", "max_dd")}
    res["ways"]["pick_per_strategy"] = row
    a = res["ways"]["asis"]
    res["vs_asis"] = {w: {p: {"total": round(v[p]["total"] - a[p]["total"], 2), "max_dd": round(v[p]["max_dd"] - a[p]["max_dd"], 2)}
                          for p in ("train", "unseen")} for w, v in res["ways"].items() if w != "asis"}
    best = max(res["vs_asis"], key=lambda w: res["vs_asis"][w]["train"]["total"])
    res["best_train"] = {"way": best, **res["vs_asis"][best]}
    res["both"] = [w for w, v in res["vs_asis"].items() if v["train"]["total"] > 0 and v["unseen"]["total"] > 0
                   and v["unseen"]["max_dd"] <= 0.25]
    return res


def report(bots):
    out = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "ways": list(WAYS), "bots": {}}
    for b in bots:
        if CS.out_dir(b).exists():
            try:
                out["bots"][b] = bot_report(b)
            except Exception as e:  # noqa: BLE001 - one bot's failure must not lose the others
                out["bots"][b] = {"error": f"{type(e).__name__}: {e}"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report"])
    ap.add_argument("--bot", default="bot1,bot2,bot3,bot4,bot5")
    a = ap.parse_args()
    r = report([x for x in a.bot.split(",") if x])
    print(json.dumps({b: {"best_train": v.get("best_train"), "both": v.get("both"), "census": v.get("census")}
                      if "error" not in v else v for b, v in r["bots"].items()}, ensure_ascii=False, indent=1))
