"""Opposite trades on the same instrument inside one bot (user, 16.11.2026, bot 3 gold: two buys and one sell open
together - "if the buy reaches its target the sell surely reaches its stop, and the other way round. Long and short
on the same instrument are fine on different timeframes, but it has to be calculated so we don't enter trades that
surely lose"; and "maybe a closer target, so that both directions get a better chance to close in profit, even if
lower").

A CONFLICT: a new trade while another strategy of the same bot holds the opposite side of the same instrument AND the
two ranges overlap - the new one's target is beyond the held one's stop (one winning = the other surely losing), or
the held one's target is beyond the new one's stop (a trade with no target counts as reaching anything).

Ways compared on the whole bot, with the fixed risk rule (portfolio_check.replay - 4.5% a day / 9% in all):
  asis        as today, every trade opens
  skip        the new conflicting trade does not open
  stronger    the stronger strategy (R a trade on the choosing period) stays: a stronger new one closes the held one at
              the new one's entry price; a weaker new one does not open
  half        the new conflicting trade opens at half its risk
  tight_new   the new one's target moves to just before the held one's stop (0.1R short of it); under 0.3R -> no trade
  tight_short the shorter-timeframe trade of the two gets that closer target (the new one or the held one)
  tight_both  both get it: each one's target just before the other's stop (the user's idea in full)
A closer target is reached when the trade went at least that far in its favour while open (mfe_r: the furthest it went
before its own exit - so before its stop too); the costs stay the trade's own. Approximations, said in the report: the
held trade's mfe may come from before the conflict; the exit time of a trade closed earlier stays its own.

Chosen on the choosing period (30.04.2025-31.12.2025), shown on 2026 - like every other proposal. Nothing in the bot
changes here.

  python -m agents.conflict_study plan --bot bot3              # instruments with 2+ strategies (JSON, the workflow)
  python -m agents.conflict_study run --bot bot3 --sym XAUUSD --minutes 100
  python -m agents.conflict_study report --bot bot3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TF_MIN = {"1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30, "1Hour": 60, "4Hour": 240, "1Day": 1440}
COLS = ["combo", "time", "exit_time", "dir", "r", "entry_px", "stop_px", "exit_px", "target_px", "mfe_r"]
GAP_R, MIN_TIGHT_R = 0.1, 0.3


def out_dir(bot):
    return ROOT / "reports" / "agents" / bot / "conflict"


def combos_by_sym(bot):
    from agents import portfolio_check as PC
    cfg = PC.bot_cfg(bot)
    off = PC.off_server(bot, cfg)
    by = {}
    for c in cfg.get("combos") or []:
        if c not in off and not ((cfg.get("plans") or {}).get(c) or {}).get("off_from"):
            by.setdefault(c.split("|")[2], []).append(c)
    return by


def plan(bot):
    return [s for s, cs in combos_by_sym(bot).items() if len(cs) >= 2]


def run(bot, sym, minutes):
    """The instrument's strategies as the bot trades them now (each in the family its built plan lives in, the built
    variant's trades), with prices - from 30.04.2025."""
    import agent_portfolio as A
    A.BOT_NOW["bot"] = bot
    cfg = json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
    plans = cfg.get("plans") or {}
    groups = {}
    for c in combos_by_sym(bot).get(sym, []):
        p = plans.get(c) or {}
        fam = "pyr" if p.get("pyramid") else A.replay_family(p)
        one = c.split("|")[1] == "1Min"
        groups.setdefault((fam, c if one else ""), []).append(c)
    t0, parts, missing = time.time(), [], []
    for (fam, _), cs in sorted(groups.items(), key=lambda x: x[0][1] != ""):
        left = minutes * 60 - (time.time() - t0)
        if left < 120:
            missing += cs
            continue
        try:
            t, rows = A.run_rules(cs, fam, left / 60)
        except Exception as e:  # noqa: BLE001
            print(f"{sym} {fam}: failed - {str(e)[-300:]}")
            missing += cs
            continue
        for c in cs:
            sid, tf, s = c.split("|")
            tc = t[(t["sid"] == sid) & (t["tf"] == tf) & (t["symbol"] == s)]
            if "variant" in tc:
                doc = {"has_target": (rows.get(c) or {}).get("has_target"),
                       "variants": {v: 1 for v in tc["variant"].dropna().unique()}}
                v = A.base_variant(bot, c, fam, doc)
                tc = tc[tc["variant"] == v]
            if tc.empty:
                missing.append(c)
                continue
            parts.append(tc.assign(combo=c))
        print(f"{sym} {fam}: {len(cs)} strategies ({time.time() - t0:.0f}s)")
    d = out_dir(bot)
    d.mkdir(parents=True, exist_ok=True)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    tt = pd.to_datetime(df["time"], utc=True, format="ISO8601") if len(df) else pd.Series([], dtype="datetime64[ns, UTC]")
    df = df[tt >= pd.Timestamp(A.TEST_FROM, tz="UTC")] if len(df) else df
    df.reindex(columns=COLS).to_csv(d / f"{sym}.csv.gz", index=False)
    (d / f"{sym}.json").write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                              "trades": int(len(df)), "missing": missing}, ensure_ascii=False), encoding="utf-8")
    print(f"{sym}: {len(df)} trades, missing {missing}")


# ---------- the comparison ----------

def _minute(x):
    ep, mn = pd.Timestamp("1970-01-01", tz="UTC"), pd.Timedelta(minutes=1)
    return ((pd.to_datetime(x, utc=True, format="ISO8601") - ep) // mn).astype("int64")


def load(bot):
    d = out_dir(bot)
    parts = [pd.read_csv(f) for f in sorted(d.glob("*.csv.gz"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=COLS + ["a", "b", "sym", "tf", "risk_px", "cost_r"])
    t = pd.concat(parts, ignore_index=True).dropna(subset=["entry_px", "stop_px", "r"])
    t["exit_time"] = t["exit_time"].fillna(t["time"])
    t["a"], t["b"] = _minute(t["time"]), _minute(t["exit_time"])
    t["sym"] = t["combo"].str.split("|").str[2]
    t["tf"] = t["combo"].str.split("|").str[1].map(TF_MIN).fillna(60)
    t["risk_px"] = (t["entry_px"] - t["stop_px"]).abs()
    t = t[t["risk_px"] > 0]
    gross = t["dir"] * (t["exit_px"] - t["entry_px"]) / t["risk_px"]
    t["cost_r"] = (gross - t["r"]).clip(lower=0).fillna(0)
    return t.sort_values(["a", "b"]).reset_index(drop=True)


def _beyond_stop(x, y):
    """x's target reaches y's stop (x and y on opposite sides). No target = reaches anything."""
    if not np.isfinite(x["target_px"]):
        return True
    return x["target_px"] >= y["stop_px"] if x["dir"] > 0 else x["target_px"] <= y["stop_px"]


def conflict(n, h):
    return n["dir"] != h["dir"] and (_beyond_stop(n, h) or _beyond_stop(h, n))


def _tight_r(x, other):
    """The R of a target just before the other's stop (GAP_R short of it); None if it would not change x's target."""
    lvl = other["stop_px"] - x["dir"] * GAP_R * x["risk_px"]
    r = x["dir"] * (lvl - x["entry_px"]) / x["risk_px"]
    if np.isfinite(x["target_px"]) and r >= x["dir"] * (x["target_px"] - x["entry_px"]) / x["risk_px"] - 1e-9:
        return None
    return r


def _tighten(x, r):
    """x with a target r R away: reached if it went that far in its favour while open."""
    x = dict(x)
    if np.isfinite(x["mfe_r"]) and x["mfe_r"] >= r:
        x["r"] = r - x["cost_r"]
    x["target_px"] = x["entry_px"] + x["dir"] * r * x["risk_px"]
    x["tight"] = True
    return x


def simulate(t, way, strength):
    """-> the trades taken under one way (list of dicts with a, b, r, mult_share, combo) + counts."""
    taken, held = [], []        # held: indices into taken still open
    n_conf = n_changed = 0
    for row in t.to_dict("records"):
        row["share"] = 1.0
        held = [i for i in held if taken[i]["b"] > row["a"]]
        opp = [i for i in held if taken[i]["sym"] == row["sym"] and taken[i]["combo"] != row["combo"]
               and conflict(row, taken[i])]
        if not opp:
            held.append(len(taken))
            taken.append(row)
            continue
        n_conf += 1
        h = taken[opp[0]]
        if way == "asis":
            pass
        elif way == "skip":
            n_changed += 1
            continue
        elif way == "stronger":
            if strength.get(row["combo"], -9) > strength.get(h["combo"], -9):
                for i in opp:                                   # close the weaker held one(s) at this price
                    x = taken[i]
                    if x["b"] > row["a"]:
                        x["r"] = x["dir"] * (row["entry_px"] - x["entry_px"]) / x["risk_px"] - x["cost_r"]
                        x["b"] = row["a"]
                n_changed += 1
                held = [i for i in held if i not in opp]
            else:
                n_changed += 1
                continue
        elif way == "half":
            row["share"] = 0.5
            n_changed += 1
        elif way in ("tight_new", "tight_short", "tight_both"):
            mine = way == "tight_new" or way == "tight_both" or row["tf"] <= h["tf"]
            if mine:
                r = _tight_r(row, h)
                if r is not None:
                    if r < MIN_TIGHT_R:
                        n_changed += 1
                        continue
                    row = _tighten(row, r)
                    n_changed += 1
            if way == "tight_both" or (way == "tight_short" and h["tf"] < row["tf"]):
                for i in opp:
                    x = taken[i]
                    r = _tight_r(x, row)
                    if r is not None and r >= MIN_TIGHT_R and not x.get("tight"):
                        taken[i] = _tighten(x, r)
                        n_changed += 1
        held.append(len(taken))
        taken.append(row)
    return taken, {"conflicts": n_conf, "changed": n_changed}


def _book(taken, mults):
    book = {}
    for x in taken:
        key = x["combo"] + ("#h" if x["share"] != 1.0 else "")
        m = float(mults.get(x["combo"], 1.0)) * x["share"]
        book.setdefault(key, ([], m))[0].append([int(x["a"]), int(x["b"]), int(x["dir"]), float(x["r"])])
    return book


WAYS = ["asis", "skip", "stronger", "half", "tight_new", "tight_short", "tight_both"]
WAY_HE = {"asis": "כמו היום", "skip": "לא לפתוח את המתנגשת", "stronger": "רק החזקה נשארת", "half": "המתנגשת בחצי סיכון",
          "tight_new": "יעד צר לחדשה", "tight_short": "יעד צר לטיים פריים הקצר", "tight_both": "יעד צר לשתיהן"}


def report(bot):
    from agents import portfolio_check as PC
    import agent_portfolio as A
    t = load(bot)
    risk = PC.bot_risk(bot)
    mults = PC.mults(bot)
    split = int((pd.Timestamp(A.SPLIT, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(minutes=1))
    train, unseen = t[t["a"] < split], t[t["a"] >= split]
    strength = train.groupby("combo")["r"].mean().to_dict()
    out = {"bot": bot, "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "strategies": int(t["combo"].nunique()),
           "missing": sorted({c for f in out_dir(bot).glob("*.json") if f.name != "report.json"
                              for c in json.loads(f.read_text(encoding="utf-8")).get("missing", [])}),
           "ways": {}}
    for way in WAYS:
        row = {}
        for per, tt in (("train", train), ("unseen", unseen)):
            taken, cnt = simulate(tt, way, strength)
            rep = PC.replay(_book(taken, mults), risk)
            cf = [x for x in taken]
            row[per] = {**cnt, **{k: rep[k] for k in ("trades", "total", "worst_day", "max_dd", "guard_cut", "guard_skipped")}}
        out["ways"][way] = row
    # the conflicts themselves (as today), by instrument, and what the conflicting new trades earned
    by = {}
    for per, tt in (("train", train), ("unseen", unseen)):
        taken, _ = simulate(tt, "asis", strength)
        held = []
        for i, x in enumerate(taken):
            held = [j for j in held if taken[j]["b"] > x["a"]]
            if any(taken[j]["sym"] == x["sym"] and taken[j]["combo"] != x["combo"] and conflict(x, taken[j]) for j in held):
                s = by.setdefault(x["sym"], {}).setdefault(per, {"n": 0, "r": 0.0})
                s["n"] += 1
                s["r"] = round(s["r"] + float(x["r"]), 2)
            held.append(i)
    out["by_symbol"] = by
    (out_dir(bot) / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--bot", default="bot3")
    ap.add_argument("--sym", default="")
    ap.add_argument("--minutes", type=float, default=100)
    a = ap.parse_args()
    if a.mode == "plan":
        json.dump(plan(a.bot), sys.stdout)
    elif a.mode == "run":
        run(a.bot, a.sym, a.minutes)
    else:
        print(json.dumps(report(a.bot), ensure_ascii=False, indent=1))
