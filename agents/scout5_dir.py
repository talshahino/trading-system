"""Bot 5b (the scout account) - the user's question of 18.11.2026 19:03 (task b5scdir1811): "what did you learn from
today - some trades very profitable, some less - could we have seen it coming and cut it?". On 18.11 four gold buys
opened together (13:40-14:00) against open winning gold sells and lost -$1,696. Tested here on ALL the scout strategies
of the 5b account together (live/bots/bot5.json "scout", the stored backtests reports/agents/bot5/exp new_*: FTMO's
costs inside each trade's R), sized as live/watch_own.py sizes them (one budget per instrument split among its
strategies) and replayed with the account rule of the live bot (TG.account: 4.5% day / 9% total, every open trade at
its full stop):
  opp      no new trade against a trade still open on the same instrument (any open trade - the stored trades carry no
           path, so "only when the open one is in profit" cannot be replayed from them; that variant needs the bars)
  cap1/2   at most 1 / 2 entries in the same direction on the same instrument inside 15 minutes (the later ones skipped)
  split    the same cluster kept, its risk split among the trades of the cluster (each 1/n)
  opp+cap1 both
  ema1d / ema4h / day  (needs the bars - the bots' computer): only with the direction - the last CLOSED daily / 4-hour
           bar's close above (buy) / below (sell) its EMA20, or the day so far (price vs the day's open, New York day)
Chosen on 2025 (to 2025-12-31), shown on 2026 (unseen). Time-weighted R (agentkit) beside the account's %: total,
the fall from the peak, the worst day, % green days, trades. Numbers only; the card is written by the nightly agent.

  python -m agents.scout5_dir [--dirs]   (agent_bot5.py runs it with --dirs on the bots' computer every pass)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import agentkit as K
from agents import portfolio_check as PCK
from agents import tal_gap as TG

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "scout_dir.json"
WIN = pd.Timedelta(minutes=15)


def book():
    s5 = json.loads((ROOT / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    rk = s5.get("risk") or {}
    combos = list(s5.get("scout") or {})
    psb = float(rk.get("scout_per_symbol") or 0)
    per_sym = pd.Series([c.split("|")[-1] for c in combos]).value_counts().to_dict()
    rows, risk, missing = [], {}, []
    for combo in combos:
        d = PCK.exp("bot5", "new", combo) or {}
        k = f"scout:{combo}"
        risk[k] = 100 * (float(rk[k]) if k in rk else
                         (psb / per_sym[combo.split("|")[-1]] if 0 < psb <= 0.02 else float(rk.get("scout", 0.004))))
        got = False
        for key in ("tr25", "tr"):
            tr = PCK._minutes(((d.get(key) or {}).get("new")) or [])
            if not tr:
                continue
            got = True
            a = np.array(tr, float)
            rows.append(pd.DataFrame({"time": pd.to_datetime(a[:, 0] * 60, unit="s", utc=True),
                                      "out": pd.to_datetime(a[:, 1] * 60 + 60, unit="s", utc=True),
                                      "dir": a[:, 2].astype(int), "r": a[:, 3], "trk": k, "sym": combo.split("|")[-1]}))
        if not got:
            missing.append(combo)
    t = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    return t, risk, missing


def opp_mask(t):
    """True = keep: no trade on the same instrument in the other direction is open at this entry (only kept trades count)."""
    keep = np.ones(len(t), bool)
    open_ = {}                                    # sym -> [(out, dir)]
    for i, (tm, out, d, s) in enumerate(zip(t["time"], t["out"], t["dir"], t["sym"])):
        lst = [x for x in open_.get(s, []) if x[0] > tm]
        open_[s] = lst
        if any(x[1] == -d for x in lst):
            keep[i] = False
            continue
        lst.append((out, d))
    return keep


def clusters(t):
    """Per trade: its index inside its cluster (same instrument + direction, entries within 15 min of the cluster's first)
    and the cluster's size."""
    pos, size = np.zeros(len(t), int), np.ones(len(t), int)
    first = {}                                    # (sym, dir) -> (time of the first, [indices])
    for i, (tm, d, s) in enumerate(zip(t["time"], t["dir"], t["sym"])):
        k = (s, d)
        f = first.get(k)
        if f is None or tm - f[0] > WIN:
            first[k] = (tm, [i])
        else:
            pos[i] = len(f[1])
            f[1].append(i)
            for j in f[1]:
                size[j] = len(f[1])
    return pos, size


def dir_keep(t, kind):
    """True = keep: the trade goes with the higher timeframe's direction at its entry (closed bars only). None when the
    bars are missing (the cloud has no price data - this runs on the bots' computer)."""
    from agents import poc_agent as P
    keep = np.ones(len(t), bool)
    for sym in t["sym"].unique():
        m = (t["sym"] == sym).to_numpy()
        tt = t.loc[m, "time"]
        try:
            # 15-minute bars from the 1-minute cache the bots' computer already keeps (no new download), as 1 hour
            h1 = P.bars(sym, "15Min", (tt.min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d"), tt.max().strftime("%Y-%m-%d"))
        except Exception:  # noqa: BLE001
            return None
        if h1 is None or h1.empty:
            return None
        h1.index = pd.to_datetime(h1.index, utc=True)
        h1 = h1.resample("1h", label="left", closed="left").agg({"o": "first", "c": "last"}).dropna()
        if kind == "day":
            ny = h1.index.tz_convert("America/New_York")
            dopen = h1["o"].groupby(ny.date).transform("first")
            ref = pd.DataFrame({"x": h1["c"] - dopen}, index=h1.index + pd.Timedelta(hours=1))   # known at the hour's close
        else:
            rule = "1D" if kind == "ema1d" else "4h"
            b = h1["c"].resample(rule, label="left", closed="left").last().dropna()
            ema = b.ewm(span=20, adjust=False).mean()
            ref = pd.DataFrame({"x": b - ema}, index=b.index + pd.Timedelta(rule))              # known at the bar's close
        ref = ref.sort_index()
        # keep the tz (tt.values dropped it -> "incompatible merge keys") and map back to the trades' own order
        q = pd.DataFrame({"time": pd.to_datetime(tt, utc=True).to_numpy(), "pos": np.arange(len(tt))})
        q["time"] = pd.to_datetime(q["time"], utc=True)
        ref.index = pd.to_datetime(ref.index, utc=True)
        q = pd.merge_asof(q.sort_values("time"), ref, left_on="time", right_index=True).sort_values("pos")
        x = q["x"]
        sgn = np.sign(x.to_numpy(float))
        d = t.loc[m, "dir"].to_numpy()
        keep[np.where(m)[0]] = ~(np.isfinite(sgn) & (sgn != 0) & (sgn != d))
    return keep


def stats(t, risk):
    out = {}
    for per in ("train", "unseen"):
        m = (t["time"] >= TG.SPLIT) if per == "unseen" else (t["time"] < TG.SPLIT)
        p = t[m].copy()
        a = TG.account(p, risk) or {}
        day = (p["r"] * p["trk"].map(risk).fillna(0) * p.get("mult", 1.0)).groupby(p["out"].dt.strftime("%Y-%m-%d")).sum()
        ks = K.wstats(p["r"].to_numpy(float), K.weights(p["time"])) if len(p) else None
        out[per] = {"n": int(len(p)), "total_pct": a.get("total_pct"), "max_fall": a.get("max_fall_pct"),
                    "worst_day": a.get("worst_day_pct"), "green": round(100 * float((day > 0).mean()), 1) if len(day) else None,
                    "stopped": a.get("stopped"), "room_skipped": a.get("room_skipped"), "k": ks,
                    "wsum_r": round(float((p["r"] * K.weights(p["time"])).sum()), 1) if len(p) else 0.0}
    return out


def main(dirs=False):
    t, risk, missing = book()
    keep_opp = opp_mask(t)
    pos, size = clusters(t)
    ways = {"now": t.assign(mult=1.0),
            "opp": t[keep_opp].assign(mult=1.0),
            "cap1": t[pos < 1].assign(mult=1.0),
            "cap2": t[pos < 2].assign(mult=1.0)}
    sp = t.copy()
    sp["r"] = sp["r"] / size                         # the cluster's risk split among its trades (each 1/n)
    ways["split"] = sp.assign(mult=1.0)
    t2 = t[keep_opp].reset_index(drop=True)
    p2, _ = clusters(t2)
    ways["opp+cap1"] = t2[p2 < 1].assign(mult=1.0)
    no_bars = []
    if dirs:
        for kind in ("ema1d", "ema4h", "day"):
            k_ = dir_keep(t, kind)
            if k_ is None:
                no_bars.append(kind)
                continue
            ways[kind] = t[k_].assign(mult=1.0)
    res = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "split": str(TG.SPLIT.date()),
           "strategies": int(t["trk"].nunique()), "missing": missing,
           "clustered_share": round(float((size > 1).mean()) * 100, 1), "no_bars": no_bars,
           "ways": {w: stats(x, risk) for w, x in ways.items()}}
    tr = {w: (v["train"]["total_pct"] or -1e9) for w, v in res["ways"].items()}
    res["chosen_on_train"] = max(tr, key=tr.get)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for w, v in res["ways"].items():
        print(w, {p: {k: v[p][k] for k in ("n", "total_pct", "max_fall", "worst_day", "green", "stopped")} for p in v})
    print("chosen on 2025:", res["chosen_on_train"])
    return res


if __name__ == "__main__":
    import sys
    main(dirs="--dirs" in sys.argv)
