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
from agents import repo_io as RIO
from agents import own_gaps as OG
from agents import tal_gap as TG
from agents import tal_wide as TW

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
TAL_SYMBOLS = ["US100", "US30", "US500", "GER40", "JP225"]     # Tal's gap method: bot 5's instruments


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


# Tal's gap method on bot 5's world (12.11.2026: the engine's API is the one bot 6 uses - TG.run / summarize /
# write_results / cards; the old TG.experiments call failed every night, so bot 5 had no results)
TAL_DIR = OUT / "tal"
TAL_WIDE_DIR = TAL_DIR / "wide"      # under tal/: the bots computer keeps its run cache unpublished (agents_host.snapshot); the method on every FTMO forex pair, the metals, the other indices (13.11.2026)


TREND_MODES = ("day", "up20")
FIX_FILE = Path("data/agents/fix_attempts.json")


def _live_tracks():
    """{track: (pair_name, cfg, risk)} - the index tracks bot 5 counts now (risk > 0, no 'symbols')."""
    try:
        s5 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    rk = s5.get("risk") or {}
    return {k: (f"{v['pair'][0]}/{v['pair'][1]}", TG._norm(dict(v["cfg"])), float(rk.get(k, 0.0028) or 0))
            for k, v in (s5.get("tal_tracks") or {}).items() if not v.get("symbols") and float(rk.get(k, 0.0028) or 0) > 0}


def trend_runs(budget_s):
    """The fix tried for the loss of 12.11 (13.11.2026 night): every running track's own rules with TREND = day / up20,
    on the five indices - so the fix is tested against exactly what runs (the search tries TREND only around its own
    best). Saved in the gap method's index like a search configuration (tal_gap.summarize shows them)."""
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in idx_dir.glob("*.json")}
    t0, done = time.time(), 0
    want = [(pn, TG._norm(dict(cfg, TREND=m))) for pn, cfg, _ in _live_tracks().values() for m in TREND_MODES]
    for sym in TAL_SYMBOLS:
        todo = [(pn, c) for pn, c in want if f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}" not in have]
        if not todo or time.time() - t0 > budget_s:
            continue
        D, gc = tal_load(sym), {}
        if D is None:
            continue
        for pn, c in todo:
            key = f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}"
            t = TG.simulate(D, pairs[pn], c, gc.setdefault(pn.split("/")[0], {}))
            t.to_csv(TAL_DIR / f"{key}.csv.gz", index=False)
            (idx_dir / f"{key}.json").write_text(json.dumps(
                {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": pairs[pn][2],
                 "knob": "search", "val": TG.vid(c), "cfg": c, "sym": sym, "file": f"{key}.csv.gz", "n": len(t)}, default=str))
            done += 1
    print(f"tal trend fix: {done} run")
    return done


def sym_runs(budget_s):
    """The instrument tracks (13.11.2026 evening, the manager's task f8643b16aa57): every track in "tal_tracks" with
    "symbols" and a risk > 0 (t1h_5@US30, t5_1@EU50, ... - chosen by agents/tal_pool_gh.py on GitHub) run with exactly
    its rules on its instrument here, saved in the gap method's index - so the whole-bot test (TG.portfolio) takes
    them together with the index tracks, as the account does."""
    try:
        s5 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    rk = s5.get("risk") or {}
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in idx_dir.glob("*.json")}
    want = {}
    for k, v in (s5.get("tal_tracks") or {}).items():
        pn = f"{v['pair'][0]}/{v['pair'][1]}"
        if not v.get("symbols") or float(rk.get(k, 0) or 0) <= 0 or pn not in pairs:
            continue
        c = TG._norm(dict(v["cfg"]))
        for sym in v["symbols"]:
            if f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}" not in have:
                want.setdefault(sym, []).append((pn, c))
    t0, done = time.time(), 0
    for sym, todo in want.items():
        if time.time() - t0 > budget_s:
            break
        D, gc = tal_load(sym), {}
        if D is None:
            print(f"tal instrument tracks {sym}: no data")
            continue
        for pn, c in todo:
            key = f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}"
            t = TG.simulate(D, pairs[pn], c, gc.setdefault(pn.split("/")[0], {}))
            t.to_csv(TAL_DIR / f"{key}.csv.gz", index=False)
            (idx_dir / f"{key}.json").write_text(json.dumps(
                {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": pairs[pn][2],
                 "knob": "track", "val": TG.vid(c), "cfg": c, "sym": sym, "file": f"{key}.csv.gz", "n": len(t)}, default=str))
            done += 1
    print(f"tal instrument tracks: {done} run, {sum(len(v) for v in want.values()) - done} left")
    return done


STAB_RR = (0.5, 1.0)


def stab_runs(budget_s):
    """Stability (bot 5's smart agent, 13.11.2026 night; Tal 5.11: "many small wins and one big loss = not stable"):
    a running track with a low minimum target / stop (live 12.11: t4h_15 took US100 / US500 at 0.22-0.28 - a win
    +0.22R, a loss -1R) with MIN_RR raised to 0.5 / 1.0, on the five indices - the same rules otherwise, saved in the
    gap method's index like trend_runs()."""
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in idx_dir.glob("*.json")}
    t0, done = time.time(), 0
    want = [(pn, TG._norm(dict(cfg, MIN_RR=x))) for pn, cfg, _ in _live_tracks().values()
            for x in STAB_RR if float(cfg.get("MIN_RR", 0) or 0) < x]
    for sym in TAL_SYMBOLS:
        todo = [(pn, c) for pn, c in want if f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}" not in have]
        if not todo or time.time() - t0 > budget_s:
            continue
        D, gc = tal_load(sym), {}
        if D is None:
            continue
        for pn, c in todo:
            key = f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}"
            t = TG.simulate(D, pairs[pn], c, gc.setdefault(pn.split("/")[0], {}))
            t.to_csv(TAL_DIR / f"{key}.csv.gz", index=False)
            (idx_dir / f"{key}.json").write_text(json.dumps(
                {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": pairs[pn][2],
                 "knob": "search", "val": TG.vid(c), "cfg": c, "sym": sym, "file": f"{key}.csv.gz", "n": len(t)}, default=str))
            done += 1
    print(f"tal stability: {done} run")
    return done


# the user's notes on live trades (13.11.2026, data/agents/replies.json "_tasks"): each idea tested on exactly the
# track's rules as they run (live/bots/bot5.json, even a track at risk 0), on the five indices, one knob moved -
# chosen on 2024-2025 (the time-weighted total), shown on 2026 (unseen), a fixed risk per trade
NOTE_TESTS = [
    # t4h_5, 13.11 15:21: US500 stopped 0.37% under the entry in a sharp drop, US100 from the same impulse (stop 1.3%
    # away) stayed open and came back - "a slightly wider stop would have kept us in?"
    {"id": "t4h_5_spad", "track": "t4h_5", "knob": "SPAD", "vals": [0.001, 0.002, 0.004],
     "he": "סטופ רחב יותר מעבר לסווינג (% מהמחיר)"},
    # the fast charts (13.11, GER40 t15_5): half out at half the way, the rest's stop to the entry
    {"id": "t15_5_p5050", "track": "t15_5", "knob": "EXIT", "vals": ["p50_50"], "he": "50% החוצה ב־50% מהדרך + סטופ לכניסה"},
    {"id": "t5_1_p5050", "track": "t5_1", "knob": "EXIT", "vals": ["p50_50"], "he": "50% החוצה ב־50% מהדרך + סטופ לכניסה"},
    # bot 5's smart agent, 13.11.2026 night, from the live record: 3 of the day's 5 losses were filled within 40 minutes
    # of New York's open (14:30 UTC: t15_5 US500 / JP225 14:37-14:38, t4h_5 US500 15:10) - the open's first half hour
    # off (tal_gap.SESS_NY "xopen", as bot 6 tests on the futures)
    {"id": "t15_5_xopen", "track": "t15_5", "knob": "SESS", "vals": ["xopen"], "he": "בלי כניסות ב־09:30-10:00 ניו יורק"},
    {"id": "t15_1_xopen", "track": "t15_1", "knob": "SESS", "vals": ["xopen"], "he": "בלי כניסות ב־09:30-10:00 ניו יורק"},
    {"id": "t4h_5_xopen", "track": "t4h_5", "knob": "SESS", "vals": ["xopen"], "he": "בלי כניסות ב־09:30-10:00 ניו יורק"},
]


def _note_cfgs():
    """[(test, pair name, cfg now, [(val, cfg)])] from the tracks as they run."""
    try:
        s5 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    tt, out = s5.get("tal_tracks") or {}, []
    for T in NOTE_TESTS:
        v = tt.get(T["track"])
        if not v or v.get("symbols"):
            continue
        now = TG._norm(dict(v["cfg"]))
        out.append((T, f"{v['pair'][0]}/{v['pair'][1]}", now,
                    [(x, TG._norm(dict(now, **{T["knob"]: x}))) for x in T["vals"] if now.get(T["knob"]) != x]))
    return out


def note_runs(budget_s):
    """The runs of NOTE_TESTS (the track's rules now and each value), saved in the gap method's index like trend_runs()."""
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in idx_dir.glob("*.json")}
    want = []
    for T, pn, now, vs in _note_cfgs():
        if pn in pairs:
            want += [(pn, now)] + [(pn, c) for _, c in vs]
    t0, done = time.time(), 0
    for sym in TAL_SYMBOLS:
        todo = [(pn, c) for pn, c in want if f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}" not in have]
        if not todo or time.time() - t0 > budget_s:
            continue
        D, gc = tal_load(sym), {}
        if D is None:
            continue
        for pn, c in todo:
            key = f"{pn.replace('/', '_')}_{TG.vid(c)}_{sym}"
            if key in have:
                continue
            t = TG.simulate(D, pairs[pn], c, gc.setdefault(pn.split("/")[0], {}))
            t.to_csv(TAL_DIR / f"{key}.csv.gz", index=False)
            (idx_dir / f"{key}.json").write_text(json.dumps(
                {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": pairs[pn][2],
                 "knob": "note", "val": TG.vid(c), "cfg": c, "sym": sym, "file": f"{key}.csv.gz", "n": len(t)}, default=str))
            have.add(key)
            done += 1
    print(f"tal note tests: {done} run")
    return done


def note_cards(tidx):
    """Per NOTE_TESTS entry: the value CHOSEN on 2024-2025 (time-weighted total R), an 'improve' card only when it beats
    the rules that run on 2024-2025 AND on 2026 in total, and its worst day on 2026 is not worse (the main goal: more
    profit, not more risk). Returns (cards, results) - the results go to tal_gap_results.json (bot5.notes) for the
    trade conversation and bot 6's agent."""
    out, res = [], {}
    for T, pn, now, vs in _note_cfgs():
        base = _trades(tidx, pn, now)
        if base is None:
            res[T["id"]] = {"state": "running"}
            continue
        bv, bs = _risk_view(base), TG._stats(base)
        row = {"track": T["track"], "pair": pn, "knob": T["knob"], "now": {"val": now.get(T["knob"]), **bv}}
        best = None
        for x, c in vs:
            t = _trades(tidx, pn, c)
            if t is None:
                continue
            v = _risk_view(t)
            row[str(x)] = v
            if v["train"] and bv["train"] and v["train"]["total"] > bv["train"]["total"] \
                    and (best is None or v["train"]["total"] > best[1]["train"]["total"]):
                best = (x, v, c, t)                      # chosen on 2024-2025 only
        if len(row) - 4 < len(vs):
            row["state"] = "running"
            res[T["id"]] = row
            continue
        row["state"] = "done"
        win = bool(best and bv["unseen"] and best[1]["unseen"] and best[1]["unseen"]["total"] > bv["unseen"]["total"]
                   and best[1]["unseen"]["worst_day"] >= bv["unseen"]["worst_day"])
        row["chosen"], row["better"] = (best[0] if best else None), win
        res[T["id"]] = row
        if not win:
            continue
        x, v, c, t = best
        st = TG._stats(t)
        u, bu, tr, btr = v["unseen"], bv["unseen"], v["train"], bv["train"]
        months = max(1.0, (pd.Timestamp.now(tz="UTC") - SPLIT).days / 30.4)
        try:
            s5 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
            rk = float((s5.get("risk") or {}).get(T["track"], 0.0028) or 0)     # a track at risk 0 (t5_1) is worth 0 now
        except Exception:  # noqa: BLE001
            rk = 0.0028
        worth = round((u["total"] - bu["total"]) / months * rk * 100, 3)
        hi, lo = pn.split("/")
        out.append(K.card(
            BOT, "improve", f"שיטת הגאפ · {TG.TF_HE[hi]} / {TG.TF_HE[lo]}: {T['he']} ({x}) - מההערה של טל",
            f"מההערה של טל על עסקה במסלול {T['track']}. הכללים שרצים, רק {T['knob']} = {x}. נבחר על 2024-2025 (סה\"כ משוקלל "
            f"{btr['total']:+.1f}R ← {tr['total']:+.1f}R), 2026 (לא נראתה): סה\"כ {bu['total']:+.1f}R ← {u['total']:+.1f}R, "
            f"היום הגרוע {bu['worst_day']:+.1f}R ← {u['worst_day']:+.1f}R, ירידה {bu['max_fall']:.1f}R ← {u['max_fall']:.1f}R, "
            f"{u['n']} עסקאות. שווי: {worth:+.2f}% מהחשבון בחודש בסיכון של המסלול.",
            st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
            tested=len(vs), risk_pct=round(rk * 100, 3), baseline=bs["unseen"],
            neighbours={"values": [y for y, _ in vs], "share_positive": round(float(np.mean(
                [bool((row.get(str(y)) or {}).get("unseen")) and row[str(y)]["unseen"]["total"] > bu["total"] for y, _ in vs])), 2)},
            detail={"note_test": row, "worth_month_pct": worth},
            change={"method": "tal_gap", "pair": pn, "cfg": c, "track": f"tal_gap|{pn}", "knob": "cfg"}))
    return out, res


def _risk_view(t):
    """Per period: the total R (time-weighted on 2024-2025), the deepest fall and the worst day in R, and the total
    per unit of deepest fall - profit and risk together (the main goal: more profit AND less risk)."""
    out = {}
    for nm, part in (("train", t[t["time"] < SPLIT]), ("unseen", t[t["time"] >= SPLIT])):
        part = part.sort_values("time")
        if len(part) < 20:
            out[nm] = None
            continue
        tot = TG.score(part) if nm == "train" else float(part["r"].sum())
        cum = part["r"].cumsum()
        fall = float((cum.cummax() - cum).max())
        day = part.groupby(part["time"].dt.strftime("%Y-%m-%d"))["r"].sum()
        out[nm] = {"n": int(len(part)), "total": round(tot, 1), "max_fall": round(fall, 1),
                   "worst_day": round(float(day.min()), 2), "per_fall": round(tot / max(fall, 1.0), 3),
                   "small_wins": round(float(((part["r"] > 0) & (part["r"] < 0.5)).mean()) * 100, 1)}
    return out


def stab_cards(tidx):
    """Per running track with a low MIN_RR: the version CHOSEN on 2024-2025 by total per deepest fall (profit and risk
    together), an 'improve' card only when on 2026 (unseen) it is better per deepest fall, its worst day is not worse
    and its average R per trade is not lower. Returns (cards, results)."""
    out, res = [], {}
    for trk, (pn, cfg, rk) in _live_tracks().items():
        vs = [x for x in STAB_RR if float(cfg.get("MIN_RR", 0) or 0) < x]
        if not vs:
            continue
        base = _trades(tidx, pn, cfg)
        if base is None:
            continue
        bv = _risk_view(base)
        row = {"now": {"MIN_RR": cfg.get("MIN_RR"), **bv}}
        best = None
        for x in vs:
            c = TG._norm(dict(cfg, MIN_RR=x))
            t = _trades(tidx, pn, c)
            if t is None:
                continue
            v = _risk_view(t)
            row[str(x)] = v
            if v["train"] and bv["train"] and v["train"]["per_fall"] > bv["train"]["per_fall"] \
                    and (best is None or v["train"]["per_fall"] > best[1]["train"]["per_fall"]):
                best = (x, v, c, t)                       # chosen on 2024-2025 only
        res[trk] = row
        if not best or not bv["unseen"] or not best[1]["unseen"]:
            continue
        x, v, c, t = best
        u, bu = v["unseen"], bv["unseen"]
        st, bs = TG._stats(t), TG._stats(base)
        if u["per_fall"] <= bu["per_fall"] or u["worst_day"] < bu["worst_day"] \
                or (st["unseen"]["mean"] or -9) < (bs["unseen"]["mean"] or 0):
            continue
        hi, lo = pn.split("/")
        tr, btr = v["train"], bv["train"]
        out.append(K.card(
            BOT, "improve", f"שיטת הגאפ · {TG.TF_HE[hi]} / {TG.TF_HE[lo]}: יחס מינימלי {cfg.get('MIN_RR')} ← {x} (יציבות)",
            f"המסלול {trk} עם הכללים שרצים, רק עם יחס מינימלי יעד/סטופ {x} במקום {cfg.get('MIN_RR')}. בחי, 12.11: עסקאות ביחס "
            f"0.22-0.28 (רווח +0.2R מול הפסד -1R) - טל, 5.11: הרבה רווחים קטנים והפסד גדול אחד זה לא יציב. נבחר על 2024-2025 "
            f"לפי רווח לכל R של ירידה מהשיא ({btr['per_fall']:.2f} ← {tr['per_fall']:.2f}; סה\"כ משוקלל {btr['total']:+.1f}R ← "
            f"{tr['total']:+.1f}R, ירידה {btr['max_fall']:.1f}R ← {tr['max_fall']:.1f}R). 2026 (לא נראתה): סה\"כ {bu['total']:+.1f}R ← "
            f"{u['total']:+.1f}R, ירידה {bu['max_fall']:.1f}R ← {u['max_fall']:.1f}R, היום הגרוע {bu['worst_day']:+.1f}R ← "
            f"{u['worst_day']:+.1f}R, {st['unseen']['mean']:+.2f}R לעסקה (היום {bs['unseen']['mean']:+.2f}R), "
            f"רווחים קטנים מ־0.5R: {bu['small_wins']}% ← {u['small_wins']}% מהעסקאות.",
            st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
            tested=len(vs), risk_pct=round(rk * 100, 3), baseline=bs["unseen"],
            neighbours={"values": list(vs), "share_positive": round(float(np.mean(
                [bool((row.get(str(y)) or {}).get("unseen")) and row[str(y)]["unseen"]["per_fall"] > bu["per_fall"] for y in vs])), 2)},
            detail={"stability": row},
            change={"method": "tal_gap", "pair": pn, "cfg": c, "track": f"tal_gap|{pn}", "knob": "cfg"}))
    return out, res


def _trades(tidx, pn, cfg):
    fs = [v for v in tidx.values() if v["pair"] == pn and TG._norm(dict(v["cfg"])) == cfg and v["sym"] in TAL_SYMBOLS]
    if {v["sym"] for v in fs} != set(TAL_SYMBOLS):
        return None
    t = pd.concat([pd.read_csv(TAL_DIR / v["file"]) for v in fs], ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    return t


def trend_fix(tidx, pf, rule_cards):
    """Per running track: its rules with TREND against the rules alone - chosen on 2024-2025 (the time-weighted total),
    shown on 2026 (the total). An 'improve' card where TREND is better on both; data/agents/fix_attempts.json updated
    from the result (the manager approves a risk cut only after the fixes failed - user, 12.11.2026)."""
    out, res, pending = [], {}, False
    for trk, (pn, cfg, rk) in _live_tracks().items():
        base = _trades(tidx, pn, cfg)
        if base is None:
            pending = True
            continue
        b_tr, b_un = TG.score(base), float(base.loc[base["time"] >= SPLIT, "r"].sum())
        row = {"live": {"train_w": round(b_tr, 1), "unseen": round(b_un, 1)}}
        best = None
        for m in TREND_MODES:
            c = TG._norm(dict(cfg, TREND=m))
            t = _trades(tidx, pn, c)
            if t is None:
                pending = True
                continue
            tr, un = TG.score(t), float(t.loc[t["time"] >= SPLIT, "r"].sum())
            row[m] = {"train_w": round(tr, 1), "unseen": round(un, 1), "n": int(len(t))}
            if tr > b_tr and (best is None or tr > best[1]):
                best = (m, tr, un, c, t)                 # chosen on the train period only
        res[trk] = row
        if best and best[2] > b_un:
            m, tr, un, c, t = best
            st, bs = TG._stats(t), TG._stats(base)
            hi, lo = pn.split("/")
            out.append(K.card(BOT, "improve", f"שיטת הגאפ · {TG.TF_HE[hi]} / {TG.TF_HE[lo]}: רק בכיוון התמונה הגדולה ({m})",
                              f"המסלול {trk} עם הכללים שרצים היום, ועוד מסנן TREND={m} ({TG.NAMES['TREND']}). מהפסד 12.11: מכירות "
                              f"במדדים שעלו כל אחר הצהריים. נבחר על 2024-2025 (סה\"כ משוקלל {b_tr:+.1f}R ← {tr:+.1f}R), מוצג על 2026: "
                              f"סה\"כ {b_un:+.1f}R ← {un:+.1f}R, {st['unseen']['n']} עסקאות, {st['unseen']['mean']:+.2f}R לעסקה "
                              f"(היום {bs['unseen']['mean']:+.2f}R על {bs['unseen']['n']}).",
                              st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
                              tested=len(TREND_MODES), risk_pct=round(rk * 100, 3), baseline=bs["unseen"],
                              neighbours={"values": list(TREND_MODES), "share_positive": round(np.mean(
                                  [(row.get(x) or {}).get("unseen", -1e9) > b_un for x in TREND_MODES]), 2)},
                              change={"method": "tal_gap", "pair": pn, "cfg": c, "track": f"tal_gap|{pn}", "knob": "cfg"}))
    if pending:
        return out, res
    try:
        fx = json.loads(FIX_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        fx = {}
    at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    fixed = {c["change"]["pair"] for c in out}
    for trk, (pn, _, _) in _live_tracks().items():
        k = f"bot5|{trk}"
        if k not in fx:
            continue                                     # only a track that has a fix record (a loss / a weak track)
        line = f"{at[:10]}: TREND (day / up20) על הכללים שרצים: " + ", ".join(
            f"{m} {res[trk].get(m, {}).get('train_w', '-')}R/{res[trk].get(m, {}).get('unseen', '-')}R" for m in TREND_MODES) + \
            f" מול {res[trk]['live']['train_w']}R/{res[trk]['live']['unseen']}R (2024-2025 משוקלל / 2026)"
        if line not in fx[k].get("tried", []):
            fx[k].setdefault("tried", []).append(line)
        if pn in fixed:
            fx[k].update(status="fixed", at=at)
        elif fx[k].get("status") == "trying":
            fx[k].update(status="failed", at=at)
    # the bot as a whole: the caps / the one-per-side rule / the hour filter (portfolio) and TREND on every track
    if pf and not rule_cards and not out:
        fx["bot5|tal_all"] = {"status": "failed", "at": at, "by": "bot5 agent", "tried": [
            "תקרות על הבוט השלם (מסלול אחד לכל כיוון ומכשיר, עד 1/2/3 באותו כיוון בכל המדדים, מסנן שעות) - אף אחת לא מורידה "
            "את הסיכון בשתי התקופות בלי להוריד את הרווח של 2026", "TREND (day / up20) על כל המסלולים שרצים - לא משפר בשתי התקופות"]}
    elif out or rule_cards:
        fx["bot5|tal_all"] = {"status": "fixed" if out else "trying", "at": at, "by": "bot5 agent",
                              "tried": ["נמצא תיקון במסלולים / בכללי הבוט השלם - כרטיס"]}
    FIX_FILE.write_text(json.dumps(fx, ensure_ascii=False, indent=1), encoding="utf-8")
    return out, res


def wide_chosen():
    """{pair: [the indices' chosen cfg, the cfg bot 5 runs on that pair]} - phase 1 of agents/tal_wide.py."""
    out = {}
    try:
        doc = json.loads(TG.RESULTS.read_text(encoding="utf-8")).get("bot5") or {}
        for pn, x in (doc.get("pairs") or {}).items():
            if (x.get("chosen") or {}).get("cfg"):
                out.setdefault(pn, []).append(x["chosen"]["cfg"])
    except Exception:  # noqa: BLE001
        pass
    try:
        s5 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
        for v in (s5.get("tal_tracks") or {}).values():
            if not v.get("symbols"):
                out.setdefault(f"{v['pair'][0]}/{v['pair'][1]}", []).append(v["cfg"])
    except Exception:  # noqa: BLE001
        pass
    return out
TAL_WORLD = {"symbols": TAL_SYMBOLS, "costs": "FTMO's recorded 1-minute spread + commission",
             "sessions": "FTMO Standard: market breaks and session ends closed, news masks", "stop_buffer": "2 points",
             "split": "train to 2025-12-31, unseen 2026"}


def tal_load(s):
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = prep(s, end)
    if data is None:
        return None
    df, cost, stop_at, noent = data
    ev, _ = O15.NC.load()
    times = pd.Series(df.index)
    cal = {f"cal{m}": O15.NC.blocked(times, s, ev, minutes=m).to_numpy() for m in (15, 60)}
    return TG.Data(s, df, cost, stop_at, noent, TG.stop_buffer(s), cal)   # 2 points on an index, 2 pips elsewhere


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


IDX = EXP / "idx"          # one small file per experiment: several runs (one per instrument) can work in parallel


def load_index():
    IDX.mkdir(parents=True, exist_ok=True)
    idx = {}
    old = EXP / "index.json"                                  # the first nights' single index
    if old.exists():
        idx.update(json.loads(old.read_text()))
    for f in IDX.glob("*.json"):
        v = json.loads(f.read_text())
        idx[f"{v['trk']}|{vid(v['cfg'])}|{v['sym']}"] = v
    return idx


def plan():
    """The manager's instructions to this agent (data/agents/plan.json): order of the tracks, knobs to skip."""
    try:
        return json.loads(Path("data/agents/plan.json").read_text(encoding="utf-8")).get(BOT, {})
    except Exception:  # noqa: BLE001
        return {}


def experiments(budget_s, only_syms=None):
    """Runs what is missing or older than RERUN_DAYS, baseline first, until the time budget ends."""
    EXP.mkdir(parents=True, exist_ok=True)
    idx = load_index()
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
    # baselines first; the 15-minute track before the 1-minute one (one 1-minute run takes ~8 minutes, a v5 run ~6
    # seconds - 11.11.2026), then the oldest; grouped by symbol so its minutes are loaded once
    pl = plan()
    order = pl.get("track_order", ["v5", "v1m"])
    skip = set(pl.get("skip_knobs", []))
    todo = [x for x in todo if x[4] not in skip]
    todo.sort(key=lambda x: (x[0], order.index(x[3]) if x[3] in order else 9, x[2], x[1]))
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
                    "cfg": cfg, "sym": s, "file": f.name, "n": len(t), "secs": round(time.time() - ts, 1)}
        (IDX / f.name.replace(".csv.gz", ".json")).write_text(json.dumps(idx[key], default=str))
        done += 1
        print(f"  {trk} {knob}={val} {s}: {len(t)} trades ({time.time() - ts:.0f}s)")
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
    s = RIO.live_json("live/watch_own_state.json")
    if not s:
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
    ap.add_argument("--minutes", type=float, default=None, help="default: the manager's budget (data/agents/plan.json)")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--no-cards", action="store_true", help="experiments only (a parallel run for some instruments)")
    a = ap.parse_args()
    if a.minutes is None:
        a.minutes = float(plan().get("minutes", 30))
    t_start = time.time()
    only = set(x for x in a.symbols.split(",") if x) or None
    idx = experiments(a.minutes * 60, only)
    # Tal's gap method (user, 11.11.2026; data/agents/strategies/tal_gap.md): the rest of the budget
    tal_syms = [x for x in TAL_SYMBOLS if not only or x in only]
    tal_left = a.minutes * 60 - (time.time() - t_start)
    # only bot 5's own run: bot 6 imports this module (BOT = "bot6") and runs the method on its own world first -
    # without this guard bot 5's world was written into bot 6's results (12.11.2026)
    if tal_left > 60 and BOT == "bot5" and not only:
        # the user's task of 13.11.2026 (high priority): the method on every FTMO forex pair, metals, other indices -
        # half of what is left, the indices' never-ending search the rest
        try:
            sym_runs(tal_left * 0.3)                # the instrument tracks, so the whole bot is tested together (first)
        except Exception as e:  # noqa: BLE001
            print("tal instrument track runs failed:", e)
        try:
            trend_runs(tal_left * 0.25)             # the fix of the 12.11 loss against exactly what runs, first
        except Exception as e:  # noqa: BLE001
            print("tal trend runs failed:", e)
        try:
            note_runs(tal_left * 0.2)               # the user's notes on live trades (13.11.2026): wider stop, half at half
        except Exception as e:  # noqa: BLE001
            print("tal note runs failed:", e)
        try:
            stab_runs(tal_left * 0.1)               # stability: the low-MIN_RR tracks with MIN_RR 0.5 / 1.0 (13.11.2026 night)
        except Exception as e:  # noqa: BLE001
            print("tal stability runs failed:", e)
        tal_left = a.minutes * 60 - (time.time() - t_start)
        try:
            TW.run(tal_load, TAL_WIDE_DIR, tal_left * 0.5, wide_chosen())
        except Exception as e:  # noqa: BLE001
            print("tal wide experiments failed:", e)
        tal_left = a.minutes * 60 - (time.time() - t_start)
    if tal_left > 60 and BOT == "bot5":
        try:
            TG.run(BOT, tal_syms, tal_load, TAL_DIR, tal_left)
        except Exception as e:  # noqa: BLE001
            print("tal gap experiments failed:", e)
    if a.no_cards:
        return
    idx = load_index()
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
        lv = live.get(trk)
        if lv and lv.get("eff_n", 0) >= 30 and lv["mean"] is not None and lv["mean"] < (base["whole"]["mean"] or 0) - 0.2:
            cards.append(K.card(BOT, "risk", f"{TRK_NAME[trk]}: בפועל גרוע בהרבה מהבדיקה",
                                f"בחשבון הווירטואלי {lv['n']} עסקאות, {lv['mean']:+.2f}R לעסקה, לעומת {base['whole']['mean']:+.2f}R בבדיקה "
                                f"על נתוני העבר. כדאי לבדוק אם משהו בביצוע שונה מהבדיקה, או להוריד את הסיכון של המסלול.",
                                base["train"], base["unseen"], whole=base["whole"], halves_=base["halves"],
                                detail={"live": lv}, change={"track": trk, "kind": "review"}))
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
    tal = {}
    try:
        if BOT != "bot5":
            # bot 6 runs the gap method itself (agent_bot6.tal) - not an error, so nothing is printed as "failed"
            # (manager 13.11.2026: the ops view showed it as a failure every night)
            raise StopIteration
        tidx = {f.stem: json.loads(f.read_text()) for f in (TAL_DIR / "idx").glob("*.json")}
        pairs, answers = TG.summarize(tidx, TAL_SYMBOLS, TAL_DIR)
        planned = len(TG.PAIRS) * len(TG.variants()) * len(TAL_SYMBOLS)
        doc = TG.write_results(BOT, pairs, answers, TAL_WORLD, [len(tidx), planned], out_dir=TAL_DIR)
        # what bot 5 trades now (live/bots/bot5.json "tal_tracks", 13.11.2026): a card on a running pair carries it as
        # the baseline, so a version that earns less on 2026 in total is not approved over it
        s5 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
        # an instrument's own track ("symbols", agents/tal_wide.py) is not the indices' track of that pair
        live5 = {f"{v['pair'][0]}/{v['pair'][1]}": v["cfg"] for v in (s5.get("tal_tracks") or {}).values() if not v.get("symbols")}
        rk5 = [float(s5["risk"][k]) for k in (s5.get("tal_tracks") or {}) if k in (s5.get("risk") or {})]
        cards += TG.cards(BOT, pairs, doc.get("bot6"), "בוט 6", risk_pct=round(min(rk5) * 100, 3) if rk5 else 0.4, live=live5)
        tal = {"done": len(tidx), "planned": planned}
        # only the tracks the account counts: a track at risk 0 (t5_1, out since 12.11 night) still runs and is shown,
        # but its trades are not the account's - counted here they made the cap / doubled-risk test look worse or better
        # than the bot is (13.11.2026)
        rkd = s5.get("risk") or {}
        # each track's real risk (times the scale that runs, "tal_risk_scale"): the account in % with the bot's limits
        cur_sc = float(s5.get("tal_risk_scale", 1.0) or 1.0)
        pf = TG.portfolio(tidx, TAL_SYMBOLS, TAL_DIR, {k: (f"{v['pair'][0]}/{v['pair'][1]}", v["cfg"])
                                                       for k, v in (s5.get("tal_tracks") or {}).items()
                                                       if float(rkd.get(k, 0.0028) or 0) > 0 and not v.get("symbols")},
                          risk={k: float(rkd.get(k, 0.0028)) * cur_sc for k in (s5.get("tal_tracks") or {})})
        # every FTMO forex pair, metals, the other indices (agents/tal_wide.py): per instrument and pair
        try:
            widx = {f.stem: json.loads(f.read_text()) for f in (TAL_WIDE_DIR / "idx").glob("*.json")}
            if widx:
                summ = TW.summarize(widx, TAL_WIDE_DIR)
                try:
                    wst = json.loads((TAL_WIDE_DIR / TW.STATUS).read_text())
                except Exception:  # noqa: BLE001
                    wst = {}
                doc["bot5"]["wide"] = {"status": wst, "table": TW.table(summ), "pairs": summ}
                live_sym = {(sy, f"{v['pair'][0]}/{v['pair'][1]}") for v in (s5.get("tal_tracks") or {}).values()
                            for sy in (v.get("symbols") or [])}
                rk_ = [float(s5["risk"][k]) for k in (s5.get("tal_tracks") or {}) if float((s5.get("risk") or {}).get(k, 0) or 0) > 0]
                cards += TW.cards(BOT, summ, round(min(rk_) * 100, 3) if rk_ else 0.28, len(widx), live_sym)
                TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print("tal wide summary failed:", e)
        # the whole bot (13.11.2026 evening, the manager's task f8643b16aa57): the index tracks AND the instruments' own
        # tracks together, each at its real risk - the rule cards and the risk factor are chosen on this one when
        # every track's runs are there (sym_runs); until then on the index tracks as before
        try:
            live_all = {k: (f"{v['pair'][0]}/{v['pair'][1]}", v["cfg"]) + ((list(v["symbols"]),) if v.get("symbols") else ())
                        for k, v in (s5.get("tal_tracks") or {}).items() if float(rkd.get(k, 0.0028) or 0) > 0}
            tidx_all = {f.stem: json.loads(f.read_text()) for f in (TAL_DIR / "idx").glob("*.json")}
            pf_all = TG.portfolio(tidx_all, TAL_SYMBOLS, TAL_DIR, live_all,
                                  risk={k: float(rkd.get(k, 0.0028)) * cur_sc for k in (s5.get("tal_tracks") or {})})
        except Exception as e:  # noqa: BLE001
            print("tal whole bot failed:", e)
            pf_all = None
        if pf_all:
            doc["bot5"]["portfolio_all"] = pf_all
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        else:
            print("tal whole bot: an instrument track's runs are missing - the index tracks only")
        if pf:
            doc["bot5"]["portfolio"] = pf
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            rk_live = [x for x in rk5 if x > 0]
            pfc = pf_all or pf
            rule_cards = [pc for pc in (TG.portfolio_card(BOT, pfc, round(min(rk_live) * 100, 3) if rk_live else 0.4),
                                        TG.hours_card(BOT, pfc, round(min(rk_live) * 100, 3) if rk_live else 0.4)) if pc]
            cards += rule_cards
            try:
                tcards, tres = trend_fix(tidx, pf, rule_cards)
                cards += tcards
                doc["bot5"]["trend_fix"] = tres
                TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                print("tal trend fix failed:", e)
            try:
                scards, sres = stab_cards(tidx)
                cards += scards
                doc["bot5"]["stability"] = sres
                TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                print("tal stability failed:", e)
            try:
                ncards, nres = note_cards(tidx)
                cards += ncards
                doc["bot5"]["notes"] = nres
                TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                print("tal note tests failed:", e)
            sc_card = TG.scale_card(BOT, pf_all or pf, cur_sc)
            if sc_card:
                cards.append(sc_card)
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        print("tal gap summary failed:", e)
    for c in cards:
        c["tier"], c["verdict"] = K.judge(c)
    K.write_cards(BOT, cards, {"summary": summary, "tested": tested,
                               "done": sum(1 for v in idx.values()), "planned": sum(len(TRACKS[t]["symbols"]) for t, *_ in variants()),
                               "tal": {"done": tal.get("done"), "planned": tal.get("planned")}})
    print(json.dumps({"cards": len(cards), "summary": {k: {kk: vv for kk, vv in v.items() if kk != "backtest"} for k, v in summary.items()}},
                     ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
