"""Bot 5ב: could a day like 18.11.2026 be told apart at entry? (task b5scoutday1811, the user 18.11 19:03: gold fell all
day, 5 buys lost -5R, 13 sells earned +13R - "could we have anticipated it and reduced it?")

The rule tested: a trade AGAINST the day's move so far - the move from the day's start (Prague day, the last close before
it) to the last closed 5-minute bar before the entry, at least TH x the average daily range of the 14 days before.
Action: skip the trade, or take it at half risk. Scope: one rule for all, per instrument, per strategy.
The trades are 5ב's own (agents/conflict_study run for bot5: live rules, management, FTMO's costs); the account is
tal_gap.account (the live fixed risk rule: 4.5% a day, 9% from the start) with 0.4% per instrument split among its
strategies, as live. Chosen on 30.04-31.12.2025 only, shown on 2026.
  python -m agents.scout5_day
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from agents import agentkit as K, conflict_study as CS, tal_gap as TG

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "scout_day.json"
CONF = ROOT / "reports" / "agents" / "bot5" / "conflict"
FROM = pd.Timestamp("2025-04-30", tz="UTC")
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
THS = (0.5, 0.75, 1.0)
ACTS = ("skip", "half")
BUDGET = 0.4                # % of the balance per instrument, split among its strategies (bot5.json scout_per_symbol)


def day_context(sym, mode="first"):
    """For each 5-minute bar: the day's start price and the 14-day average range (only days already finished).
    mode "first" = the close of the day's first 5-minute bar (the bars keep no open), "prev" = the last close before the day."""
    b = pd.read_csv(CONF / f"{sym}.bars.csv.gz").dropna().sort_values("m").reset_index(drop=True)
    ts = pd.to_datetime(b["m"].to_numpy() * 60, unit="s", utc=True)
    b["day"] = ts.tz_convert("Europe/Prague").strftime("%Y-%m-%d")
    g = b.groupby("day", sort=True)
    d = pd.DataFrame({"hi": g["h"].max(), "lo": g["l"].min(), "first": g["c"].first(), "last": g["c"].last()})
    d["adr"] = (d["hi"] - d["lo"]).rolling(14, min_periods=10).mean().shift(1)
    d["open"] = d["first"] if mode == "first" else d["last"].shift(1)
    b["adr"], b["open"] = b["day"].map(d["adr"]), b["day"].map(d["open"])
    b["closed"] = b["m"] + 5        # a bar is known only when it has closed
    return b[["closed", "c", "adr", "open", "day"]].rename(columns={"day": "bar_day"})


def add_move(t, mode="first"):
    out = []
    for sym, g in t.groupby("sym"):
        ctx = day_context(sym, mode)
        g = pd.merge_asof(g.sort_values("a"), ctx, left_on="a", right_on="closed", direction="backward")
        out.append(g)
    t = pd.concat(out, ignore_index=True)
    # + = with the day's move, - = against it, in units of the average daily range
    t["x"] = t["dir"] * (t["c"] - t["open"]) / t["adr"]
    today = t["time"].dt.tz_convert("Europe/Prague").dt.strftime("%Y-%m-%d")
    t.loc[(t["bar_day"] != today).to_numpy(), "x"] = 0.0      # no closed bar of today yet: no move to read
    return t


def book(t, act=None, mask=None):
    """The trades as the account takes them under a rule (mask = the trades the rule catches)."""
    u = t.copy()
    u["trk"] = u["combo"]
    if mask is not None and act == "skip":
        u = u[~mask.to_numpy()]
    elif mask is not None and act == "half":
        u.loc[mask.to_numpy(), "trk"] = u.loc[mask.to_numpy(), "combo"] + "#h"
    return u


def result(u, risk):
    a = TG.account(u, risk) or {}
    d = u.groupby(u["out"].dt.strftime("%Y-%m-%d")).apply(lambda g: float((g["r"] * g["trk"].map(risk)).sum()), include_groups=False)
    return {"n": int(len(u)), "total": round(float(a.get("total_pct") or 0), 2), "fall": round(float(a.get("max_fall_pct") or 0), 2),
            "worst": round(float(a.get("worst_day_pct") or 0), 2), "green": round(100 * float((d > 0).mean()), 1) if len(d) else None,
            "stopped": a.get("stopped"), "skipped_room": a.get("room_skipped")}


def both(t, risk, act=None, mask=None):
    u = book(t, act, mask)
    return {"train": result(u[u["time"] < SPLIT], risk), "unseen": result(u[u["time"] >= SPLIT], risk)}


def caught(t, mask):
    """What the rule catches: R a trade of those trades (time-weighted), per period, with its t."""
    r = {}
    for p, m in (("train", t["time"] < SPLIT), ("unseen", t["time"] >= SPLIT)):
        g = t[mask & m]
        s = K.wstats(g["r"], K.weights(g["time"].astype(str))) if len(g) else {"n": 0, "mean": None, "t": None}
        r[p] = {k: s.get(k) for k in ("n", "mean", "t", "win")}
    return r


def better(v, b):
    return v["total"] >= b["total"] and v["fall"] <= b["fall"] and v["worst"] >= b["worst"]


def pick_by(t, risk, key):
    """Per instrument / per strategy: each group takes the rule that added most on the choosing period alone (or none)."""
    tr = t[t["time"] < SPLIT]
    chosen = {}
    for k, g in tr.groupby(key):
        best, gain = None, 0.0
        for th in THS:
            m = g["x"] <= -th
            if m.sum() < 15:
                continue
            for act in ACTS:
                add = float(-g.loc[m, "r"].sum() * (1.0 if act == "skip" else 0.5))
                if add > gain:
                    best, gain = (th, act), add
        if best:
            chosen[k] = {"th": best[0], "act": best[1], "train_gain_r": round(gain, 1), "train_n": int((g["x"] <= -best[0]).sum())}
    u = t.copy()
    u["trk"] = u["combo"]
    keep = np.ones(len(u), bool)
    for k, c in chosen.items():
        m = (u[key] == k).to_numpy() & (u["x"] <= -c["th"]).to_numpy()
        if c["act"] == "skip":
            keep &= ~m
        else:
            u.loc[m, "trk"] = u.loc[m, "combo"] + "#h"
    u = u[keep]
    return chosen, {"train": result(u[u["time"] < SPLIT], risk), "unseen": result(u[u["time"] >= SPLIT], risk)}


def main():
    t = CS.load("bot5")
    t["time"], t["out"] = pd.to_datetime(t["time"], utc=True, format="ISO8601"), pd.to_datetime(t["exit_time"], utc=True, format="ISO8601")
    t0 = t[t["time"] >= FROM].copy()
    t = add_move(t0, "first").dropna(subset=["x"]).reset_index(drop=True)
    nsym = json.loads((ROOT / "live/bots/bot5.json").read_text(encoding="utf-8"))["scout"]
    cnt = pd.Series([c.split("|")[2] for c in nsym]).value_counts()
    risk = {c: BUDGET / cnt[c.split("|")[2]] for c in t["combo"].unique()}
    risk.update({c + "#h": v / 2 for c, v in list(risk.items())})
    base = both(t, risk)
    rows = []
    for th in THS:
        m = t["x"] <= -th
        for act in ACTS:
            rows.append({"th": th, "act": act, **both(t, risk, act, m), "caught": caught(t, m)})
    # the choice: only the period to 2025 - the best total among the rules not worse there on the fall and the worst day
    fair = [r for r in rows if better(r["train"], base["train"]) and r["train"]["total"] > base["train"]["total"]]
    pick = max(fair, key=lambda r: r["train"]["total"]) if fair else None
    tested = len(rows) + 2 * len(THS) * len(ACTS)
    z_need = float(stats.norm.ppf(1 - 0.05 / tested))
    sym_pick, sym_res = pick_by(t, risk, "sym")
    combo_pick, combo_res = pick_by(t, risk, "combo")
    # the day itself
    d18 = t[(t["time"] >= "2026-11-18") & (t["time"] < "2026-11-19")]
    day = {"n": int(len(d18)), "sum_r": round(float(d18["r"].sum()), 2),
           "by_th": {str(th): {"caught": int((d18["x"] <= -th).sum()), "caught_r": round(float(d18.loc[d18["x"] <= -th, "r"].sum()), 2)} for th in THS}}
    # the same rules with the other reading of "the day's start": a rule that lives on one reading is not a rule
    t2 = add_move(t0, "prev").dropna(subset=["x"]).reset_index(drop=True)
    sym_prev_pick, sym_prev_res = pick_by(t2, risk, "sym")
    other = [{"th": th, "act": act, **both(t2, risk, act, t2["x"] <= -th), "caught": caught(t2, t2["x"] <= -th)} for th in THS for act in ACTS]
    verdict = bool(pick and better(pick["unseen"], base["unseen"]) and pick["unseen"]["total"] > base["unseen"]["total"])
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "task": "b5scoutday1811", "trades": int(len(t)),
           "strategies": int(t["combo"].nunique()), "not_in_data": sorted(set(nsym) - set(t["combo"])), "base": base, "rules": rows, "rules_prev_close": other,
           "pick_on_2025": None if not pick else {k: pick[k] for k in ("th", "act")}, "pick_holds_2026": verdict,
           "tested": tested, "t_needed_after_correction": round(z_need, 2),
           "per_symbol": {"chosen": sym_pick, **sym_res}, "per_symbol_prev_close": {"chosen": sym_prev_pick, **sym_prev_res}, "per_strategy": {"chosen": combo_pick, **combo_res}, "day_18_11": day}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("BASE", base)
    for r in rows:
        print(r["th"], r["act"], "25", r["train"], "26", r["unseen"], "caught", r["caught"])
    for r in other:
        print("prev", r["th"], r["act"], "25", r["train"]["total"], r["train"]["fall"], "26", r["unseen"]["total"], r["unseen"]["fall"])
    print("PICK", rep["pick_on_2025"], "holds 2026:", verdict, "| t needed", rep["t_needed_after_correction"], "| tested", tested)
    print("PER SYMBOL", sym_pick, sym_res)
    print("PER STRATEGY", len(combo_pick), combo_res)
    print("DAY", day, "missing", rep["not_in_data"])


if __name__ == "__main__":
    main()
