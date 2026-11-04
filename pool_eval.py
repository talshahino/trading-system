"""Would the new pool's candidates help a bot? Each candidate is added, alone, to each bot (1-4), and the
bot's own $100K replay (loo.replay: graded risk, 4.5% / 9% guards) is compared with and without it, in the
two halves (2025-04-30..2025-12-31, 2026-01-01..end). A candidate helps a bot only if, in BOTH halves,
the end balance is higher and the deepest low is not deeper. Bot 4 trades only what the MetaQuotes demo
has (FX, metals, US shares).

  python pool_eval.py --cands data/portfolio/combos_candidates3.txt --cand-trades 'reports/X/combo_trades_*.csv.gz'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

import loo as LO
from agents import ftmo_universe as U

ROOT = Path(__file__).resolve().parent
BOTS13 = "reports/2026-11-04_bots_compare_run14/news_trades.csv.gz"
BOT4 = "reports/2026-11-04_tv_run61[2-9]/combo_trades_*.csv.gz"


def load(pattern):
    fs = glob.glob(pattern)
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if "combo" not in t:
        t["combo"] = t["sid"] + "|" + t["tf"] + "|" + t["symbol"]
    for c in ("time", "exit_time", "free_time"):
        t[c] = pd.to_datetime(t[c], utc=True, errors="coerce")
    return t


def on_metaquotes(c):
    s = c.split("|")[2]
    k = U.CLASS.get(s)
    return k in ("fx", "metal") or (k == "stock" and "USD" in U.CURRENCIES.get(s, []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--cand-trades", required=True)
    args = ap.parse_args()
    cands = [l.strip() for l in open(args.cands, encoding="utf-8") if l.strip()]
    ct = load(args.cand_trades)
    ct = ct[ct["combo"].isin(cands)]
    base = pd.concat([load(BOTS13), load(BOT4)], ignore_index=True)
    out = []
    for b in (1, 2, 3, 4):
        cfg = json.load(open(ROOT / "live" / "bots" / f"bot{b}.json", encoding="utf-8"))
        LO.F.FREE_RISK_AT_BREAKEVEN = bool(cfg["free_risk_at_breakeven"])
        mine = base[base["combo"].isin(cfg["combos"])].drop_duplicates(["combo", "time"])
        halves = [mine[(mine["time"] >= pd.Timestamp(a, tz="UTC")) & (mine["time"] < pd.Timestamp(z, tz="UTC"))]
                  for a, z in LO.HALVES]
        full = [LO.replay(h.sort_values("time").reset_index(drop=True), cfg["risk"]) for h in halves]
        print("bot", b, full)
        for c in cands:
            if c in cfg["combos"] or (b == 4 and not on_metaquotes(c)):
                continue
            x = ct[ct["combo"] == c]
            if x.empty:
                continue
            res = []
            for i, (a, z) in enumerate(LO.HALVES):
                xa = x[(x["time"] >= pd.Timestamp(a, tz="UTC")) & (x["time"] < pd.Timestamp(z, tz="UTC"))]
                w = pd.concat([halves[i], xa], ignore_index=True).sort_values("time").reset_index(drop=True)
                r = LO.replay(w, cfg["risk"])
                res.append({"end_change": r["end"] - full[i]["end"], "low_change": r["low"] - full[i]["low"],
                            "trades": int(len(xa)), "sum_r": round(float(xa["r"].sum()), 1), "blown": r["blown"]})
            helps = all(q["end_change"] > 0 and q["low_change"] >= 0 and not q["blown"] for q in res)
            out.append({"bot": b, "combo": c, "halves": res, "helps": helps})
            print(b, c, [(q["trades"], q["sum_r"], q["end_change"], q["low_change"]) for q in res], "HELPS" if helps else "")
    p = ROOT / "reports" / "pool_eval" / "result.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
