"""The improvement agent of bot 5 (scalping, virtual) - nightly (user, 11.11.2026; plan "סוכני שיפור וסוכן מנהל").

What it does every night, within a time budget (GitHub minutes are limited):
 1. health    the bot's own virtual trades (live/watch_own_state.json on branch live) per track, against the backtest
              of the same rules, every number time-weighted (agents/agentkit.py: recent trades count more)
 2. experiments  one change at a time to the live rules of each gap track (v5 on 15 minutes, the 1-minute track):
              partial exit (share / where), stop, entry edge, minimum reward:risk, strong-candle and gap filters, expiry,
              detection timeframe, first gap only. Each experiment is a full backtest (2024 ->) with FTMO's real costs;
              results are kept (reports/agents/bot5/exp/) and a variant is re-run only after RERUN_DAYS.
 3. judge     a variant is chosen on the train period (to 2025-12-31) and shown on the unseen one (2026 ->); its
              neighbours (the other values of the same knob) give the robustness; the baseline is the live rules
 4. cards     every variant that beats the live rules on the unseen period -> a card for the manager
              (reports/agents/bot5/cards.json). Nothing changes in the bot: approved cards are built into it later,
              as a normal bot update the user approves with the code.

  python agent_bot5.py [--minutes 45] [--symbols US100,US30,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import own15 as O15
from agents import agentkit as K
from agents import own_gaps as OG

BOT = "bot5"
OUT = Path("reports/agents/bot5")
EXP = OUT / "exp"
START = "2024-01-01"
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
RERUN_DAYS = 14
TRACKS = {
    "v5": {"symbols": ["US100", "US30", "US500", "GER40", "JP225"], "risk_pct": 0.4,
           "base": {"STOP": "gap", "DOMINANT": False, "V5_MAX_COST": 0.10, "PART_FRAC": 0.3, "PART_WAY": 0.9, "TFM": 15,
                    "EDGE": "near", "NEST": 0, "MIN_RR": 1.0, "GBODY": 1.5, "MIN_GAP": 0.3, "EXPIRY": 24, "FIRST_ONLY": False}},
    "v1m": {"symbols": ["US100", "US30", "US500", "GER40"], "risk_pct": 0.075,
            "base": {"STOP": "gap", "DOMINANT": False, "V5_MAX_COST": 0.10, "PART_FRAC": 0.3, "PART_WAY": 0.9, "TFM": 1,
                     "EDGE": "near", "NEST": 0, "MIN_RR": 1.0, "GBODY": 1.5, "MIN_GAP": 0.3, "EXPIRY": 24, "FIRST_ONLY": False}},
}
# one knob at a time; the live value is in the base
KNOBS = {
    "PART_FRAC": [0.0, 0.3, 0.5, 0.7],
    "PART_WAY": [0.5, 0.7, 0.9],
    "STOP": ["gap", "s2", "s3", "top"],
    "EDGE": ["near", "mid", "far"],
    "MIN_RR": [0.8, 1.0, 1.5, 2.0],
    "GBODY": [1.2, 1.5, 2.0],
    "MIN_GAP": [0.2, 0.3, 0.5],
    "EXPIRY": [8, 24, 48],
    "FIRST_ONLY": [False, True],
    "TFM@v5": [5, 15, 60],
}
NAMES = {"PART_FRAC": "חלק שיוצא ברווח", "PART_WAY": "איפה יוצא החלק (מהדרך ליעד)", "STOP": "מיקום הסטופ",
         "EDGE": "נקודת הכניסה בפער", "MIN_RR": "יחס סיכוי/סיכון מינימלי", "GBODY": "עוצמת הנר החזק",
         "MIN_GAP": "גודל פער מינימלי", "EXPIRY": "תוקף הפער (שעות)", "FIRST_ONLY": "רק הפער הראשון של מהלך",
         "TFM": "טיים פריים לזיהוי (דקות)"}
TRK_NAME = {"v5": "מסלול 15 דקות (v5)", "v1m": "מסלול הדקה"}


def vid(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def variants():
    out = []
    for trk, T in TRACKS.items():
        out.append((trk, "base", None, dict(T["base"])))
        for knob, vals in KNOBS.items():
            k, _, only = knob.partition("@")
            if only and only != trk:
                continue
            for v in vals:
                if T["base"][k] == v:
                    continue
                if k == "PART_WAY" and T["base"]["PART_FRAC"] == 0:
                    continue
                cfg = dict(T["base"], **{k: v})
                out.append((trk, k, v, cfg))
    return out


def prep(s, end):
    df = O15.P.bars(s, "1Min", max(START, O15.RP.MINUTE_START), end)
    sp = O15.P.spread_1m(s, START, end)
    if df is None or df.empty or sp is None or len(sp) == 0:
        return None
    if "v" not in df:
        df["v"] = 1.0
    c = df["c"].to_numpy(float)
    cost = np.abs(sp.sort_index().reindex(df.index, method="ffill").to_numpy(float)) + O15.comm_px(s, c)
    cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
    stop_at = O15.RR.break_mask(df, "1Min")
    sm = O15.RR.session_mask(df, s, "1Min")
    if sm is not None:
        stop_at = stop_at | np.asarray(sm, bool)
    ev, _ = O15.NC.load()
    nm, noent = O15.RR.news_masks(df, s, "1Min")
    blk = O15.NC.blocked(pd.Series(df.index), s, ev, minutes=2).to_numpy()
    noent = np.asarray(noent, bool) | blk
    stop_at = stop_at | np.asarray(nm, bool) | np.r_[blk[1:], False]
    return df, cost, stop_at, noent


def run_one(s, data, cfg):
    df, cost, stop_at, noent = data
    keep = {k: getattr(OG, k) for k in cfg}
    try:
        for k, v in cfg.items():
            setattr(OG, k, v)
        rows = OG.run_gaps(s, df, cost, stop_at, noent, variant="agent")
    finally:
        for k, v in keep.items():
            setattr(OG, k, v)
    return pd.DataFrame([{"time": x["time"], "r": x["r"], "cost_r": x.get("cost_r")} for x in rows if x.get("status") == "closed"])


def experiments(budget_s, only_syms=None):
    """Runs what is missing or older than RERUN_DAYS, baseline first, until the time budget ends."""
    EXP.mkdir(parents=True, exist_ok=True)
    idx_f = EXP / "index.json"
    idx = json.loads(idx_f.read_text()) if idx_f.exists() else {}
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    todo = []
    for trk, knob, val, cfg in variants():
        for s in TRACKS[trk]["symbols"]:
            if only_syms and s not in only_syms:
                continue
            key = f"{trk}|{vid(cfg)}|{s}"
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(idx[key]["at"])).days if key in idx else 999
            if age >= RERUN_DAYS:
                todo.append((0 if knob == "base" else 1, -age, s, trk, knob, val, cfg, key))
    todo.sort(key=lambda x: (x[0], x[2], x[1]))                    # baselines, then the oldest; grouped by symbol
    t0, data, done = time.time(), {}, 0
    for _, _, s, trk, knob, val, cfg, key in todo:
        if time.time() - t0 > budget_s:
            break
        if s not in data:
            data.clear()                                             # one symbol's minutes in memory at a time
            try:
                data[s] = prep(s, end)
            except Exception as e:  # noqa: BLE001
                print(f"{s}: no data ({e})")
                data[s] = None
        if data[s] is None:
            continue
        ts = time.time()
        t = run_one(s, data[s], cfg)
        f = EXP / f"{trk}_{vid(cfg)}_{s}.csv.gz"
        t.to_csv(f, index=False)
        idx[key] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "trk": trk, "knob": knob, "val": val,
                    "cfg": cfg, "sym": s, "file": f.name, "n": len(t)}
        done += 1
        print(f"  {trk} {knob}={val} {s}: {len(t)} trades ({time.time() - ts:.0f}s)")
    idx_f.write_text(json.dumps(idx, indent=1, default=str))
    print(f"experiments: {done} run, {len(todo) - done} left for the next nights")
    return idx


def trades_of(idx, trk, cfg):
    fs = [EXP / v["file"] for k, v in idx.items() if v["trk"] == trk and v["cfg"] == cfg]
    syms = {v["sym"] for v in idx.values() if v["trk"] == trk and v["cfg"] == cfg}
    if syms != set(TRACKS[trk]["symbols"]):
        return None                                                  # not every instrument done yet
    t = pd.concat([pd.read_csv(f) for f in fs if f.exists()], ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    return t


def stats(t):
    w = K.weights(t["time"])
    tr, un = t[t["time"] < SPLIT], t[t["time"] >= SPLIT]
    return {"train": K.wstats(tr["r"], K.weights(tr["time"])), "unseen": K.wstats(un["r"], K.weights(un["time"])),
            "whole": K.wstats(t["r"], w), "halves": K.halves(t["time"], t["r"]),
            "cost_r": round(float(t["cost_r"].mean()), 3) if "cost_r" in t and len(t) else None}


def health():
    """The bot's own virtual trades (branch live) per track."""
    try:
        subprocess.run(["git", "fetch", "-q", "origin", "live"], check=False)
        s = json.loads(subprocess.run(["git", "show", "origin/live:live/watch_own_state.json"], capture_output=True, text=True).stdout)
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for t in s.get("watch", []) + s.get("trades_all", []):
        if not isinstance(t, dict) or t.get("status") != "closed" or t.get("r") is None:
            continue
        trk = "v5" if t.get("v5") else "v1m" if t.get("v1m") else None
        if trk:
            out.setdefault(trk, []).append((t.get("exit_time") or t.get("time"), float(t["r"])))
    return {k: K.wstats([r for _, r in v], K.weights([x for x, _ in v])) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    idx = experiments(a.minutes * 60, set(x for x in a.symbols.split(",") if x) or None)
    live = health()
    cards, summary = [], {}
    tested = 0
    for trk, T in TRACKS.items():
        base_t = trades_of(idx, trk, T["base"])
        if base_t is None or base_t.empty:
            summary[trk] = {"status": "הבדיקה הבסיסית עוד רצה"}
            continue
        base = stats(base_t)
        summary[trk] = {"backtest": base, "live": live.get(trk), "erosion": K.erosion(base_t["time"], base_t["r"])}
        if summary[trk]["erosion"]["fading"]:
            e = summary[trk]["erosion"]
            cards.append(K.card(BOT, "info", f"{TRK_NAME[trk]}: היתרון נשחק",
                                f"בחצי השנה האחרונה {e['recent_mean']:+.2f}R לעסקה, לעומת {e['before_mean']:+.2f}R בשנה וחצי שלפני",
                                base["train"], base["unseen"], whole=base["whole"], halves_=base["halves"], detail={"erosion": e}))
        by_knob = {}
        for trk_, knob, val, cfg in variants():
            if trk_ != trk or knob == "base":
                continue
            t = trades_of(idx, trk, cfg)
            if t is None or t.empty:
                continue
            tested += 1
            by_knob.setdefault(knob, []).append((val, cfg, stats(t)))
        for knob, lst in by_knob.items():
            # choose on the train period only, then show the unseen one
            best = max(lst, key=lambda x: (x[2]["train"]["mean"] or -9))
            if (best[2]["train"]["mean"] or -9) <= (base["train"]["mean"] or 0):
                continue
            pos = [x for x in lst if (x[2]["unseen"]["mean"] or -9) > (base["unseen"]["mean"] or 0)]
            val, cfg, st = best
            nm = NAMES.get(knob, knob)
            cards.append(K.card(
                BOT, "improve", f"{TRK_NAME[trk]}: {nm} {T['base'][knob]} ← {val}",
                f"לשנות את '{nm}' מ־{T['base'][knob]} ל־{val}. נבחר על 2024-2025 בלבד, ומוצג על 2026 שלא נראתה.",
                st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
                neighbours={"values": [x[0] for x in lst], "share_positive": round(len(pos) / len(lst), 2)},
                tested=len(variants()), risk_pct=T["risk_pct"], baseline=base["unseen"],
                change={"track": trk, "knob": knob, "from": T["base"][knob], "to": val}))
    for c in cards:
        c["tier"], c["verdict"] = K.judge(c)
    K.write_cards(BOT, cards, {"summary": summary, "tested": tested,
                               "done": sum(1 for v in idx.values()), "planned": sum(len(TRACKS[t]["symbols"]) for t, *_ in variants())})
    print(json.dumps({"cards": len(cards), "summary": {k: {kk: vv for kk, vv in v.items() if kk != "backtest"} for k, v in summary.items()}},
                     ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
