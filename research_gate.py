"""The research gate (GATE.md): how many tries we have made, and what a new combo must beat.

Every combo that was ever evaluated is a try. With N tries and no edge at all, the best trade-level t
statistic still lands near the expected maximum of N standard normals (about 3.3 at 1,000 tries, 4.3 at
50,000). A combo is only evidence when its t clears that line for the N tried so far.

  python research_gate.py                         -> N, the line, and every live combo's t
  python research_gate.py --cands 'reports/X/combo_trades_*.csv.gz'   -> which candidates clear it
Writes reports/gate/status.json.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEST_FROM = "2025-04-30"
HOLDOUT_FROM = "2026-11-01"      # the demo record: never used to choose anything (GATE.md)


def tries():
    rows, uniq = 0, set()
    for f in glob.glob(str(ROOT / "reports" / "*_run*" / "report_rules.html")):
        h = open(f, encoding="utf-8").read()
        for tr in re.findall(r"<tr>(.*?)</tr>", h, re.S):
            tds = [html.unescape(re.sub("<[^>]+>", " ", x)).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(tds) >= 8 and re.match(r"^[A-Z]{2}\d\d$", tds[1]):
                rows += 1
                uniq.add((tds[1], tds[3], tds[4]))
    return rows, len(uniq)


def expected_max_t(n):
    """Expected maximum of n independent standard normals (Gumbel approximation)."""
    a = np.sqrt(2 * np.log(n))
    return float(a - (np.log(np.log(n)) + np.log(4 * np.pi)) / (2 * a) + 0.5772 / a)


def t_stats(trades):
    t = trades[(trades["time"] >= pd.Timestamp(TEST_FROM, tz="UTC")) & (trades["time"] < pd.Timestamp(HOLDOUT_FROM, tz="UTC"))]
    g = t.groupby("combo")["r"].agg(["count", "mean", "std"])
    g["t"] = g["mean"] / g["std"] * np.sqrt(g["count"])
    return g


def load(pattern):
    t = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)], ignore_index=True)
    if "combo" not in t:
        t["combo"] = t["sid"] + "|" + t["tf"] + "|" + t["symbol"]
    t["time"] = pd.to_datetime(t["time"], utc=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="")
    args = ap.parse_args()
    rows, uniq = tries()
    line = expected_max_t(uniq)
    out = {"evaluations": rows, "unique_combos": uniq, "t_line": round(line, 2)}
    print(f"tries so far: {rows:,} evaluations, {uniq:,} different combos -> a combo needs t >= {line:.2f}")
    import pool_eval as P
    live = pd.concat([P.load(P.BOTS13), P.load(P.BOT4)]).drop_duplicates(["combo", "time"])
    g = t_stats(live)
    out["bots"] = {}
    for b in (1, 2, 3, 4):
        c = json.load(open(ROOT / "live" / "bots" / f"bot{b}.json", encoding="utf-8"))["combos"]
        x = g.reindex(c)["t"]
        out["bots"][b] = {"combos": len(c), "median_t": round(float(x.median()), 2), "max_t": round(float(x.max()), 2),
                          "clear_line": int((x >= line).sum())}
        print(f"bot {b}: {len(c)} combos, median t {x.median():.2f}, best {x.max():.2f}, clearing the line {int((x >= line).sum())}")
    if args.cands:
        c = t_stats(load(args.cands))
        c["clears"] = c["t"] >= line
        out["candidates"] = c.round(3).reset_index().to_dict("records")
        print(c.sort_values("t", ascending=False).round(2).to_string())
    p = ROOT / "reports" / "gate" / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
