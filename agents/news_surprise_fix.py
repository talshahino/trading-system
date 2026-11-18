"""EV08: where does the release-surprise trade lose, and is there a fix chosen on 2025? (bot 7 smart agent, 18.11.2026)

The first live release of EV08 (Pending Home Sales 18.11, -4.7% vs -2%: three index buys, up +0.5R to +1R, all three
stopped within 30 minutes, -3.3R) asks what the card never split: which US number, which side, which index, which exit.
Only the rows the surprise study already keeps (cache/news_surprise, agents/news_surprise.py) are read: a minute's work,
no candles, FTMO's costs already in r.

  split    the built version (live/bots/bot7.json) by title / group / side (a bad number = buy, a good one = sell) / index
  skip     the rule already approved for EV04 (card 4b152c858fb7): drop a title or a side that LOST on 2025 with at least
           MIN_N trades there; nothing is chosen on 2026
  exits    every delay x stop x hold the study holds, on the built releases: the best total on 2025 against the built one
Chosen on 30.04-31.12.2025, shown on 2026 (time-weighted statistics, agents/agentkit.py). Results only, nothing in the bot
changes here. Output: reports/agents/bot7/surprise_fix.json
  python -m agents.news_surprise_fix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import agentkit as AK  # noqa: E402
from agents import news_surprise as NS  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "surprise_fix.json"
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MIN_N = 8                    # as the skip rule of EV04 (card 4b152c858fb7)
STALE_DAYS = 3


def _plan():
    """The built plan of EV08: {symbol: plan}, one version over the instruments."""
    b = json.loads((ROOT / "live" / "bots" / "bot7.json").read_text(encoding="utf-8")).get("plans") or {}
    return {k.split("|")[2]: v for k, v in b.items() if k.startswith("EV08|1Min|") and not v.get("off_from")}


def _stats(z):
    """Sum and time-weighted statistics of a set of rows."""
    if not len(z):
        return {"n": 0, "sum": 0.0}
    r = z["r"].to_numpy(float)
    return dict(AK.wstats(r, AK.weights(pd.to_datetime(z["t"], utc=True))), sum=round(float(r.sum()), 2))


def _two(z):
    return {"y2025": _stats(z[z["_t"] < SPLIT]), "y2026": _stats(z[z["_t"] >= SPLIT])}


def _split(z, col):
    return {str(k): _two(g) for k, g in z.groupby(col)}


def built_rows(df, plans):
    """The releases the bot trades (groups / big / alone of the plan), every variant of the study."""
    p = next(iter(plans.values()))
    z = df[df["sym"].isin(plans)]
    if p.get("groups"):
        z = z[z["group"].isin(p["groups"])]
    if p.get("size") == "big":
        z = z[z["big"] == 1]
    if p.get("alone"):
        z = z[z["alone"] == 1]
    return z.copy()


def skip_rule(z, col):
    """What the 2025 rule drops: a value of `col` that lost on 2025 with at least MIN_N trades."""
    t = z[z["_t"] < SPLIT]
    g = t.groupby(col)["r"].agg(["count", "sum"])
    return sorted(str(k) for k, x in g.iterrows() if x["count"] >= MIN_N and x["sum"] < 0)


def analyse(df, titles, plans):
    p = next(iter(plans.values()))
    key = NS.built_key(p)
    z = built_rows(df, plans)
    z["_t"] = pd.to_datetime(z["t"], utc=True)
    z["title"] = z["rel"].map(titles).fillna("?")
    z["side"] = np.where(z["dir"] > 0, "buy", "sell")
    now = z[z["v"] == key]
    out = {"built": key, "symbols": sorted(plans), "now": _two(now),
           "by_title": _split(now, "title"), "by_group": _split(now, "group"),
           "by_side": _split(now, "side"), "by_sym": _split(now, "sym")}
    drop_t, drop_s = skip_rule(now, "title"), skip_rule(now, "side")
    kept = now[~now["title"].isin(drop_t) & ~now["side"].isin(drop_s)]
    out["skip"] = {"titles": drop_t, "sides": drop_s, "result": _two(kept),
                   "changed": bool(drop_t or drop_s)}
    rows = []
    for v, g in z[z["v"].str.endswith("|p1")].groupby("v"):
        two = _two(g)
        rows.append({"v": v, **two})
    rows.sort(key=lambda x: -x["y2025"]["sum"])
    best = rows[0] if rows else None
    out["exits"] = {"tested": len(rows), "best_2025": best, "top": rows[:8],
                    "share_top_positive_2026": round(float(np.mean([x["y2026"]["sum"] > 0 for x in rows[:8]])), 2) if rows else None}
    b, n = (best or {}), out["now"]
    out["verdict"] = {
        "skip_helps": bool(out["skip"]["changed"] and out["skip"]["result"]["y2025"]["sum"] > n["y2025"]["sum"]
                           and out["skip"]["result"]["y2026"]["sum"] > n["y2026"]["sum"]),
        "exit_helps": bool(best and b["v"] != key and b["y2025"]["sum"] > n["y2025"]["sum"]
                           and b["y2026"]["sum"] > n["y2026"]["sum"])}
    return out


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS)
    except Exception:  # noqa: BLE001
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5)          # the rows are in memory in seconds; kept for the round
    ap.parse_args(argv)
    df = NS._cached_rows()
    plans = _plan()
    if not len(df) or not plans or not set(plans) <= set(df["sym"]):
        print("EV08 fix: the surprise study's rows are not here yet")
        return False
    rel = NS.us_releases(NS.calendar_full(fetch=False)["items"])
    titles = {pd.Timestamp(k).isoformat(): v["title"] for k, v in rel.items()}
    res = analyse(df, titles, plans)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "split": str(SPLIT.date()),
                               "min_n": MIN_N, **res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("EV08 fix:", json.dumps({"now": res["now"], "skip": res["skip"]["titles"] + res["skip"]["sides"],
                                   "verdict": res["verdict"]}, ensure_ascii=False, default=str))
    return True


if __name__ == "__main__":
    main()
