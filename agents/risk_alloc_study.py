"""Sharing the risk between a bot's strategies by their edge and diversification - the step Tal left for later (13.11.2026,
tal_thinking case 41). Bots 1-4, the whole bot as it trades now, with the live guard (portfolio_check.replay).

Every way is set ONLY from the choosing period (30.04-31.12.2025, the "tr25" trades) and shown on 2026 (unseen).
Tal's red line holds: no trade above the bot's risk - a way may only lower a strategy, or raise one that is below x1 up to
x1 at most. A way counts only if on BOTH periods the bot earns more with no larger fall and no worse day.

  python -m agents.risk_alloc_study     -> reports/agents/risk_alloc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agents import portfolio_check as P  # noqa: E402

BOTS = ("bot1", "bot2", "bot3", "bot4")
MIN_N = 15            # fewer trades in the choosing period: the strategy keeps its risk
MIN_LIVE = 0.01       # strategies held at the minimum size stay there (the edge rule put them there)
EPS = P.EPS


def stats(tr):
    rows = P._minutes(tr or [])
    r = np.array([x[3] for x in rows], dtype=float)
    if len(r) < MIN_N or r.std() == 0:
        return None
    return {"n": len(r), "mean": float(r.mean()), "t": float(r.mean() / r.std() * np.sqrt(len(r)))}


def daily(tr):
    rows = P._minutes(tr or [])
    if not rows:
        return pd.Series(dtype=float)
    d = pd.to_datetime([x[1] for x in rows], unit="m", utc=True).tz_convert("Europe/Prague").date
    return pd.Series([x[3] for x in rows]).groupby(pd.Index(d)).sum()


def ways(train):
    """name -> {combo: factor}, every factor from the choosing period only."""
    st = {c: stats(tr) for c, (tr, m) in train.items() if m > MIN_LIVE}
    st = {c: s for c, s in st.items() if s}
    pos = [s["mean"] for s in st.values() if s["mean"] > 0]
    ref = float(np.median(pos)) if pos else 0.0
    out = {}
    for floor in (0.5, 0.7):
        out[f"edge_floor{floor}"] = {c: float(np.clip(s["mean"] / ref, floor, 1)) for c, s in st.items()} if ref > 0 else {}
        for t0 in (0.0, 0.5, 1.0):
            out[f"weak_t{t0}_x{floor}"] = {c: floor for c, s in st.items() if s["t"] < t0}
    days = pd.DataFrame({c: daily(train[c][0]) * train[c][1] for c in st}).fillna(0.0)
    if len(days.columns) > 2:
        tot = days.sum(axis=1)
        rho = {c: float(np.corrcoef(days[c], tot - days[c])[0, 1]) for c in days.columns if days[c].std() > 0}
        for r0 in (0.15, 0.25):
            for floor in (0.5, 0.7):
                out[f"crowded_r{r0}_x{floor}"] = {c: floor for c, v in rho.items() if v > r0}
    for t0 in (1.5, 2.0):
        out[f"strong_t{t0}_to1"] = {c: 1.25 for c, s in st.items() if s["t"] >= t0}
    return out


def apply(book, f):
    out = {}
    for c, (tr, m) in book.items():
        k = f.get(c, 1.0)
        nm = m * k
        if k > 1:
            nm = max(m, min(1.0, nm))          # the red line: never above the bot's risk
        out[c] = (tr, nm)
    return out


def better(a, b):
    return a["total"] > b["total"] and a["max_dd"] <= b["max_dd"] + EPS and a["worst_day"] >= b["worst_day"] - EPS


def main():
    res = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "bots": {}}
    for bot in BOTS:
        cfg = P.bot_cfg(bot)
        risk = P.bot_risk(bot, cfg)
        b26, miss26 = P.book_now(bot, cfg, "tr")
        b25, miss25 = P.book_now(bot, cfg, "tr25")
        base25, base26 = P.replay(b25, risk), P.replay(b26, risk)
        rows = []
        for name, f in ways(b25).items():
            if not f:
                continue
            a25, a26 = P.replay(apply(b25, f), risk), P.replay(apply(b26, f), risk)
            rows.append({"way": name, "changed": len(f), "train": a25, "unseen": a26,
                         "train_ok": better(a25, base25), "unseen_ok": better(a26, base26)})
        ok = [r for r in rows if r["train_ok"]]
        pick = max(ok, key=lambda r: r["train"]["total"]) if ok else None
        res["bots"][bot] = {"combos_train": len(b25), "combos_2026": len(b26), "missing_train": len(miss25),
                            "missing_2026": len(miss26), "base_train": base25, "base_2026": base26, "tested": len(rows),
                            "passed_train": len(ok), "passed_both": [r["way"] for r in ok if r["unseen_ok"]],
                            "picked_on_train": pick, "ways": rows}
        print(bot, "train", base25["total"], base25["max_dd"], "| 2026", base26["total"], base26["max_dd"],
              "| tested", len(rows), "train ok", len(ok), "both", res["bots"][bot]["passed_both"], flush=True)
        if pick:
            print("   pick", pick["way"], pick["changed"], "train", pick["train"]["total"], pick["train"]["max_dd"],
                  pick["train"]["worst_day"], "| 2026", pick["unseen"]["total"], pick["unseen"]["max_dd"],
                  pick["unseen"]["worst_day"], flush=True)
    (ROOT / "reports" / "agents" / "risk_alloc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
