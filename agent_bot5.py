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
from agents import ftmo_universe as U
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
    blk = O15.NC.blocked(pd.Series(df.index), s, ev, minutes=2, currencies=U.CURRENCIES.get(s)).to_numpy()
    noent = np.asarray(noent, bool) | blk
    stop_at = stop_at | np.asarray(nm, bool) | np.r_[blk[1:], False]
    return df, cost, stop_at, noent


# Tal's gap method on bot 5's world (12.11.2026: the engine's API is the one bot 6 uses - TG.run / summarize /
# write_results / cards; the old TG.experiments call failed every night, so bot 5 had no results)
TAL_DIR = OUT / "tal"
TAL_WIDE_DIR = OUT / "tal_wide"      # the method on every FTMO forex pair, the metals, the other indices (13.11.2026)


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
    cal = {f"cal{m}": O15.NC.blocked(times, s, ev, minutes=m, currencies=U.CURRENCIES.get(s)).to_numpy() for m in (15, 60)}
    return TG.Data(s, df, cost, stop_at, noent, 2.0, cal)


# ---------------------------------------------------------------- the rest of FTMO (user, 13.11.2026)
# "why does bot 5 not use every currency pair on FTMO?" - the method was tested only on the 5 indices, though the user
# built it on gold and EURUSD. Every FTMO forex pair (28), the two metals and the other cash indices, each on its own:
# its own grid on every pair of timeframes (the same 34 versions, management and Fibonacci exits included), chosen on
# its own period to 2025-12-31 and shown on 2026. Its own folder under tal/ (the bots' computer's cache, never published),
# so an instrument's choice never sees another's trades. Every instrument's grid first, then its own search.
TAL_X = [s for s in list(U.FX) + list(U.FX_MORE) + ["XAUUSD", "XAGUSD", "UK100", "FRA40", "EU50", "AUS200", "HK50",
                                                     "SPN35", "N25", "US2000"] if s not in TAL_SYMBOLS]
TAL_X_DIR = TAL_DIR / "x"
TAL_X_SHARE = 0.7          # of the gap method's minutes while the indices' search only refines (13.11.2026: 0 left)


def tal_x(budget_s):
    t0 = time.time()
    left = {}
    for s in TAL_X:                                     # 1) every instrument's grid
        rest = budget_s - (time.time() - t0)
        if rest < 60:
            break
        try:
            _, left[s] = TG.run(BOT, [s], tal_load, TAL_X_DIR / s, rest, search_ok=False)
        except Exception as e:  # noqa: BLE001
            print(f"tal_x {s} failed: {e}")
    done = [s for s in TAL_X if left.get(s) == 0]
    # 2) the search, the same minutes for each instrument whose grid is done (they take turns night after night)
    if len(done) == len(TAL_X):
        k0 = int(time.time() // 3600) % len(done)
        for s in done[k0:] + done[:k0]:
            rest = budget_s - (time.time() - t0)
            if rest < 120:
                break
            try:
                TG.run(BOT, [s], tal_load, TAL_X_DIR / s, min(rest, max(300.0, rest / 4)))
            except Exception as e:  # noqa: BLE001
                print(f"tal_x search {s} failed: {e}")
    print(f"tal_x: {len(done)} of {len(TAL_X)} instruments' grids done")


def _slim(r):
    return None if not r else {k: r.get(k) for k in ("knob", "val", "cfg", "train", "unseen", "whole", "halves", "cost_r",
                                                       "target_pct", "rr")}


def tal_x_summary():
    """Per instrument and pair of timeframes: the base and the version chosen on 2024-2025, shown on 2026."""
    out = {}
    for s in TAL_X:
        d = TAL_X_DIR / s
        tidx = {f.stem: json.loads(f.read_text()) for f in (d / "idx").glob("*.json")} if (d / "idx").exists() else {}
        if not tidx:
            out[s] = {"status": "missing", "runs": 0}
            continue
        try:
            pairs, _ = TG.summarize(tidx, [s], d)
        except Exception as e:  # noqa: BLE001
            out[s] = {"status": f"error: {e}"[:200], "runs": len(tidx)}
            continue
        out[s] = {"status": "ok", "runs": len(tidx), "planned": len(TG.PAIRS) * len(TG.variants()),
                  "pairs": {pn: {"name": x["name"], "done": x["done"], "planned": x["planned"], "searched": x["searched"],
                                 "base": _slim(x.get("base")), "chosen": _slim(x.get("chosen"))} for pn, x in pairs.items()}}
    return out


def tal_x_cards(ux, risk_pct, live_syms):
    """A 'new' card per instrument and pair whose version chosen on 2024-2025 earned on both periods: positive to 2025,
    positive on 2026 with t >= 1, and at least the manager's +3R on 2026 in total (agentkit.NEW_MIN_TOTAL) - a new
    track must add to the bot as a whole (principle 11). The best pair per instrument only: two pairs on the same
    instrument are mostly the same trades (12.11.2026, the US500 short taken twice)."""
    out = []
    for s, x in ux.items():
        if s in live_syms:
            continue
        best = None
        for pn, v in (x.get("pairs") or {}).items():
            ch = v.get("chosen")
            if not ch or v["done"] < v["planned"]:
                continue
            tr, un = ch["train"] or {}, ch["unseen"] or {}
            if (tr.get("mean") or 0) <= 0 or (tr.get("eff_n") or 0) < K.MIN_EFF_N or (un.get("mean") or 0) <= 0 \
                    or (un.get("t") or 0) < 1.0 or (un.get("mean") or 0) * (un.get("n") or 0) < K.NEW_MIN_TOTAL:
                continue
            if best is None or un["mean"] * un["n"] > best[1]["unseen"]["mean"] * best[1]["unseen"]["n"]:
                best = (pn, ch, v)
        if not best:
            continue
        pn, ch, v = best
        tr, un, cfg = ch["train"], ch["unseen"], ch["cfg"]
        knob = "הבסיס" if ch["knob"] == "base" else ", ".join(f"{k}={vv}" for k, vv in cfg.items() if TG.BASE.get(k, "none") != vv)
        what = (f"שיטת הגאפ של טל על {s} (לא נבדקה עליו עד היום), {v['name']}: {knob or 'הבסיס'}. "
                f"נבחר על 2024-2025 בלבד ({tr['mean']:+.2f}R לעסקה, {tr['n']} עסקאות), מוצג על 2026: {un['mean']:+.2f}R לעסקה, "
                f"{un['n']} עסקאות, {un['win']}% הצלחה, סה\"כ {un['mean'] * un['n']:+.1f}R. עלות FTMO {ch['cost_r']}R לעסקה. "
                f"מסלול חדש שסופר רק את {s}, בסיכון של מסלולי הגאפ.")
        out.append(K.card(BOT, "new", f"שיטת הגאפ של טל · {s} · {v['name']}", what, tr, un, whole=ch["whole"],
                          halves_=ch["halves"], cost_r=ch["cost_r"], tested=len(TG.PAIRS) * len(TG.variants()), risk_pct=risk_pct,
                          detail={"symbol": s, "pair": pn, "base": (v.get("base") or {}).get("unseen")},
                          change={"method": "tal_gap", "pair": pn, "cfg": cfg, "symbol": s,
                                  "track": f"tal_gap|{pn}|{s}", "knob": "cfg"}))
    return out


REVIEW_MIN_N = 15          # trades on 2026 on the instrument before what runs there is judged


def symbol_review(tidx, s5, risk_pct):
    """Each counted track on each of its instruments alone (the loss reports name a strategy as track + instrument, e.g.
    TAL|15Min|US500 - 13.11.2026). Where what runs loses on that instrument in 2024-2025 AND in 2026: first a fix - the
    version of the same pair that earns most on that instrument to 2025 (every run of the grid and the search), shown on
    2026; it must earn on both periods and beat what runs on 2026 in total -> a card that trades it there with its own
    rules. No fix -> the instrument leaves the track (its open trades followed to the exit). Returns (cards, attempts)."""
    cards_, att = [], {}
    rk = s5.get("risk") or {}
    for trk, v in (s5.get("tal_tracks") or {}).items():
        own = isinstance(v.get("symbols"), list) and len(v["symbols"]) == 1 and trk.endswith("_" + v["symbols"][0])
        if float(rk.get(trk, 0) or 0) <= 0 or own:
            continue                                   # out of the account, or already an instrument's own track
        pn = f"{v['pair'][0]}/{v['pair'][1]}"
        syms = v.get("symbols") if isinstance(v.get("symbols"), list) else TAL_SYMBOLS
        for s in syms:
            runs = [x for x in tidx.values() if x["pair"] == pn and x["sym"] == s]
            cur = next((x for x in runs if TG.vid(TG._norm(dict(x["cfg"]))) == TG.vid(TG._norm(dict(v["cfg"])))), None)
            if not cur:
                continue
            st = lambda x: TG._stats(_rd(x))  # noqa: E731
            c = st(cur)
            ct, cu = c["train"], c["unseen"]
            if (ct.get("mean") or 0) >= 0 or (cu.get("mean") or 0) >= 0 or (cu.get("n") or 0) < REVIEW_MIN_N:
                continue
            best, bs = None, None
            for x in runs:
                t = _rd(x)
                sc = TG.score(t)
                if best is None or sc > bs:
                    best, bs = x, sc
            b = TG._stats(_rd(best))
            bt, bu = b["train"], b["unseen"]
            key = f"bot5|TAL|{_combo_tf(trk, v['pair'])}|{s}"          # the account's combo name (live/watch_own.tal5_tf)
            base = f"{trk} על {s}: מה שרץ {ct['mean']:+.2f}R לעסקה ב־2024-2025 ({ct['n']}), {cu['mean']:+.2f}R ב־2026 ({cu['n']})"
            ok = ((bt.get("mean") or 0) > 0 and (bt.get("eff_n") or 0) >= K.MIN_EFF_N and (bu.get("mean") or 0) > 0
                  and (bu["mean"] * bu["n"]) > (cu["mean"] * cu["n"]))
            tried = (f"{len(runs)} גרסאות של {pn} על {s} לבד (כל הגריד והחיפוש: דחף, גאפ, סווינג, יחס, תוקף, הגנה, "
                     f"ניהול ופיבונאצ'י); הטובה עד 2025: {bt['mean']:+.2f}R ({bt['n']}), ב־2026 {bu['mean']:+.2f}R ({bu['n']})")
            # two tracks share a combo name (t15_1 and t15_5 are both TAL|15Min|...): one entry, fixed if either is
            prev = att.get(key) or {"status": "failed", "tried": []}
            att[key] = {"status": "fixed" if ok or prev["status"] == "fixed" else "failed",
                        "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "by": "bot5 smart agent",
                        "tried": prev["tried"] + [base + ". " + tried]}
            if ok:
                cards_.append(K.card(BOT, "improve", f"שיטת הגאפ · {s} במסלול {pn}: חוקים משלו",
                                     base + f". התיקון: {s} נסחר במסלול הזה בחוקים משלו - " + ", ".join(
                                         f"{k}={vv}" for k, vv in best["cfg"].items()) + f". נבחר על 2024-2025 בלבד, מוצג על 2026. {tried}.",
                                     bt, bu, whole=b["whole"], halves_=b["halves"], cost_r=b["cost_r"], tested=len(runs),
                                     risk_pct=round(float(rk.get(trk)) * 100, 3), baseline=cu,
                                     detail={"track": trk, "symbol": s},
                                     change={"method": "tal_gap", "pair": pn, "cfg": best["cfg"], "symbol": s, "from_track": trk,
                                             "track": f"tal_gap|{pn}|{s}", "knob": "cfg"}))
            else:
                cards_.append(K.card(BOT, "risk", f"שיטת הגאפ · {s} יוצא מהמסלול {pn}",
                                     base + f" - מפסיד בשתי התקופות. ניסיון התיקון נכשל: {tried}. לכן {s} יוצא מהמסלול הזה "
                                     f"(שאר המכשירים נשארים; עסקה פתוחה נשארת עד היציאה). אם גרסה תרוויח עליו - כרטיס חדש.",
                                     ct, cu, whole=c["whole"], halves_=c["halves"], cost_r=c["cost_r"], tested=len(runs),
                                     risk_pct=0.0, detail={"track": trk, "symbol": s},
                                     change={"method": "tal_gap", "pair": pn, "symbol": s, "from_track": trk,
                                             "track": trk, "knob": "SYMBOLS"}))
    return cards_, att


_TAL5_TF = {"t15_1": "15Min", "t1h_15": "1Hour", "t4h_1h": "4Hour", "t15_5": "15Min", "t4h_15": "4Hour15"}
_TFC = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "4h": "4Hour", "1d": "1Day"}


def _combo_tf(trk, pair):
    return _TAL5_TF.get(trk) or f"{_TFC.get(pair[0], pair[0])}{pair[1]}"


def _rd(x):
    t = pd.read_csv(TAL_DIR / x["file"])
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    return t


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
            trend_runs(tal_left * 0.25)             # the fix of the 12.11 loss against exactly what runs, first
        except Exception as e:  # noqa: BLE001
            print("tal trend runs failed:", e)
        tal_left = a.minutes * 60 - (time.time() - t_start)
        try:
            TW.run(tal_load, TAL_WIDE_DIR, tal_left * 0.5, wide_chosen())
        except Exception as e:  # noqa: BLE001
            print("tal wide experiments failed:", e)
        tal_left = a.minutes * 60 - (time.time() - t_start)
    if tal_left > 60 and BOT == "bot5":
        # the rest of FTMO first while its grids are not done (13.11.2026: the indices' search stood at 0 left on all
        # 12 pairs and only refines), then the indices
        x_s = tal_left * TAL_X_SHARE if not only else 0
        if x_s > 60:
            try:
                tal_x(x_s)
            except Exception as e:  # noqa: BLE001
                print("tal_x failed:", e)
        try:
            TG.run(BOT, tal_syms, tal_load, TAL_DIR, a.minutes * 60 - (time.time() - t_start))
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
        if pf:
            doc["bot5"]["portfolio"] = pf
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            rk_live = [x for x in rk5 if x > 0]
            for pc in (TG.portfolio_card(BOT, pf, round(min(rk_live) * 100, 3) if rk_live else 0.4),
                       TG.hours_card(BOT, pf, round(min(rk_live) * 100, 3) if rk_live else 0.4)):
                if pc:
                    cards.append(pc)
        # each track on each instrument alone (the loss reports' strategies), and the rest of FTMO (13.11.2026)
        rk_pos = sorted(x for x in rk5 if x > 0)
        rk_mid = round(rk_pos[len(rk_pos) // 2] * 100, 3) if rk_pos else 0.28
        try:
            sc_, att_ = symbol_review(tidx, s5, rk_mid)
            cards += sc_
            if att_:
                fa_f = Path("data/agents/fix_attempts.json")
                fa = json.loads(fa_f.read_text(encoding="utf-8")) if fa_f.exists() else {}
                fa.update(att_)
                fa_f.write_text(json.dumps(fa, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print("symbol review failed:", e)
        try:
            ux = tal_x_summary()
            doc = json.loads(TG.RESULTS.read_text(encoding="utf-8"))
            doc["bot5"]["universe"] = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                       "symbols": ux, "note": "כל מכשיר לבד: נבחר על 2024-2025, מוצג על 2026, עלויות FTMO"}
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            live_x = {x for v in (s5.get("tal_tracks") or {}).values() if isinstance(v.get("symbols"), list)
                      for x in v["symbols"] if x not in TAL_SYMBOLS}
            cards += tal_x_cards(ux, rk_mid, live_x)
            tal["universe"] = {"instruments": len(TAL_X), "grids_done": sum(
                1 for x in ux.values() if x.get("pairs") and all(p["done"] >= p["planned"] for p in x["pairs"].values())
                and len(x["pairs"]) == len(TG.PAIRS))}
        except Exception as e:  # noqa: BLE001
            print("tal_x summary failed:", e)
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        print("tal gap summary failed:", e)
    for c in cards:
        c["tier"], c["verdict"] = K.judge(c)
    K.write_cards(BOT, cards, {"summary": summary, "tested": tested,
                               "done": sum(1 for v in idx.values()), "planned": sum(len(TRACKS[t]["symbols"]) for t, *_ in variants()),
                               "tal": {"done": tal.get("done"), "planned": tal.get("planned"), "universe": tal.get("universe")}})
    print(json.dumps({"cards": len(cards), "summary": {k: {kk: vv for kk, vv in v.items() if kk != "backtest"} for k, v in summary.items()}},
                     ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
