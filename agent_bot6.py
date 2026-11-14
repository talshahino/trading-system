"""The improvement agent of bot 6 (CME micro index futures, virtual; user, 10-11.11.2026).

Bot 6 trades the scalping bot's methods (gap tracks v5 on 15 minutes and the 1-minute track) on MNQ / MES / MYM, so its
agent is bot 5's agent (agent_bot5.py: the same experiments, knobs, fair split and cards) with bot 6's own world:
  instruments  the index CFDs the bot reads (US100 / US500 / US30), sized and costed as micro futures:
               a tick of spread + the commission and a tick of slippage per contract (live/fut_account.py)
  sessions     FTMO Futures: everything closed 16:05-18:00 New York and over the weekend, no new entry from 15:45
  live record  the bot's virtual trades (live/watch_fut_state.json on branch live)
A method one of the two agents finds profitable reaches the other by itself: both test the same knobs, each on its own
instruments and costs (user, 11.11.2026), and every bot 6 card says what bot 5's agent found for the same change.

The user's gap method (data/agents/strategies/tal_gap.md, agents/tal_gap.py) is tested here too, on bot 6's world, up to
TAL_SHARE of each run until every pair and variant is done (then again every 14 days); results next to bot 5's in
data/agents/strategies/tal_gap_results.json and as cards in cards.json.

  python agent_bot6.py [--minutes 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import agent_bot5 as A
from agents import agentkit as K
from agents import repo_io as RIO
from agents import tal_gap as TG
from live import fut_account as FA

A.BOT = "bot6"
A.OUT = Path("reports/agents/bot6")
A.EXP = A.OUT / "exp"
A.IDX = A.EXP / "idx"
A.TRACKS = {
    "v5": {"symbols": ["US100", "US500", "US30"], "risk_pct": FA.RISK_V5 * 100, "base": dict(A.TRACKS["v5"]["base"])},
    "v1m": {"symbols": ["US100", "US500", "US30"], "risk_pct": FA.RISK_V1M * 100, "base": dict(A.TRACKS["v1m"]["base"])},
}
A.TRK_NAME = {"v5": "חוזים · מסלול 15 דקות (v5)", "v1m": "חוזים · מסלול הדקה"}
TAL_DIR = A.OUT / "tal"
TAL_SHARE = 0.5                 # of each run's minutes, while the user's method still has untested pairs
WORLD = {"symbols": ["US100", "US500", "US30"], "contracts": "MNQ / MES / MYM (micro)",
         "costs": f"tick spread + ${FA.COMM:g} commission + {FA.SLIP_TICKS:g} tick slippage per contract",
         "sessions": "FTMO Futures: closed 16:05-18:00 New York and weekends (forced exit), no new entry from 15:45",
         "stop_buffer": "2 ticks", "split": "train to 2025-12-31, unseen 2026"}


def prep(s, end):
    O15 = A.O15
    df = O15.P.bars(s, "1Min", max(A.START, O15.RP.MINUTE_START), end)
    if df is None or df.empty:
        return None
    if "v" not in df:
        df["v"] = 1.0
    _, pv, tick = FA.SPEC[s]
    cost = np.full(len(df), tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv)    # price units, per contract
    shut, late = FA.session_masks(df)
    ev, _ = O15.NC.load()
    blk = O15.NC.blocked(pd.Series(df.index), s, ev, minutes=2).to_numpy()
    noent = np.asarray(late, bool) | blk
    stop_at = np.asarray(shut, bool) | np.r_[blk[1:], False]
    return df, cost, stop_at, noent


def health():
    s = RIO.live_json("live/watch_fut_state.json")
    out = {}
    for t in s.get("closed", []) or []:
        try:
            risk = abs(float(t["open_px"]) - float(t["sl0"]))
            if risk <= 0:
                continue
            side = 1 if t.get("side") == "buy" else -1
            r = (float(t["close_px"]) - float(t["open_px"])) * side / risk
        except Exception:  # noqa: BLE001
            continue
        trk = "v1m" if "|1Min|" in (t.get("combo") or "") else "v5"
        out.setdefault(trk, []).append((t.get("close_t"), r))
    return {k: K.wstats([r for _, r in v], K.weights([x for x, _ in v])) for k, v in out.items()}


def bot5_view():
    """What bot 5's agent found for each change (its cards), shown on bot 6's card of the same change."""
    try:
        d = json.loads(Path("reports/agents/bot5/cards.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {(c.get("change") or {}).get("track", "") + "|" + str((c.get("change") or {}).get("knob")) + "|" + str((c.get("change") or {}).get("to")):
            {"tier": c.get("tier"), "unseen": (c.get("unseen") or {}).get("mean")} for c in d.get("cards", []) if c.get("kind") == "improve"}


A.prep = prep
A.health = health


def tal_load(s):
    """Bot 6's world for the user's gap method: micro-futures costs, the futures session, the calendar masks."""
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = A.O15.P.bars(s, "1Min", max(A.START, A.O15.RP.MINUTE_START), end)
    if df is None or df.empty:
        return None
    _, pv, tick = FA.SPEC[s]
    cost = np.full(len(df), tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv)
    shut, late = FA.session_masks(df)
    ev, _ = A.O15.NC.load()
    times = pd.Series(df.index)
    cal = {f"cal{m}": A.O15.NC.blocked(times, s, ev, minutes=m).to_numpy() for m in (15, 60)}
    return TG.Data(s, df, cost, shut, late, 2 * tick, cal)


def live_tracks():
    """What bot 6 trades now: {track: (pair name, cfg)} - the main track ("tal") and the others ("tal_tracks", the
    code's tracks where the file has none), as live/watch_own.py reads them."""
    out = {k: (f"{p[0]}/{p[1]}", dict(c)) for k, (p, c) in FA.TAL_TRACKS.items()}
    try:
        s6 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8"))
        tal = s6.get("tal") or {}
        if tal.get("pair") and isinstance(tal.get("cfg"), dict):
            out["tal"] = (f"{tal['pair'][0]}/{tal['pair'][1]}", {**FA.TAL_CFG, **tal["cfg"]})
        for k, v in (s6.get("tal_tracks") or {}).items():
            if k != "tal" and (v or {}).get("pair") and isinstance(v.get("cfg"), dict):
                out[k] = (f"{v['pair'][0]}/{v['pair'][1]}", dict(v["cfg"]))
    except Exception as e:  # noqa: BLE001
        print(f"bot6.json not read: {e}")
    main = out.get("tal", (None,))[0]
    return {k: v for k, v in out.items() if k == "tal" or v[0] != main}


def shared(idx, budget_s):
    """Shared methods (user, 11.11.2026): what bot 5's agent chose for each pair of the gap method, and what bot 5 trades
    now, run on bot 6's own world too - as 'search' rows, so they compete on bot 6's train period like any other version
    (never chosen on 2026)."""
    try:
        doc = json.loads(TG.RESULTS.read_text(encoding="utf-8"))
        s5 = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return 0
    want = [(pn, (x.get("chosen") or {}).get("cfg")) for pn, x in ((doc.get("bot5") or {}).get("pairs") or {}).items()]
    want += [(f"{v['pair'][0]}/{v['pair'][1]}", v.get("cfg")) for v in (s5.get("tal_tracks") or {}).values()]
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    t0, done, data, gcs = time.time(), 0, {}, {}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    for pn, cfg in want:
        if pn not in pairs or not isinstance(cfg, dict) or not all(k in cfg for k in TG.BASE):
            continue
        p, cfg = pairs[pn], TG._norm(dict(cfg))
        for sym in WORLD["symbols"]:
            key = f"{p[0]}_{p[1]}_{TG.vid(cfg)}_{sym}"
            if key in idx or time.time() - t0 > budget_s:
                continue
            if sym not in data:
                try:
                    data[sym] = tal_load(sym)
                except Exception:  # noqa: BLE001
                    data[sym] = None
            if data[sym] is None:
                continue
            t = TG.simulate(data[sym], p, cfg, gcs.setdefault((sym, p[0]), {}))
            f = TAL_DIR / f"{key}.csv.gz"
            t.to_csv(f, index=False)
            idx[key] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": p[2],
                        "knob": "search", "val": TG.vid(cfg), "cfg": cfg, "sym": sym, "file": f.name, "n": len(t), "from": "bot5"}
            (idx_dir / f"{key}.json").write_text(json.dumps(idx[key], default=str))
            done += 1
            print(f"  tal_gap bot 5's {pn} on bot 6's world {sym}: {len(t)} trades")
    return done


# TREND on what runs (bot 6's smart agent, 13.11.2026): bot 5's fix of its 12.11 loss (day = only in the direction of the
# UTC day's open, up20 = the timeframe above against its 20-candle EMA), shared, plus the futures' own picture (rth =
# against the New York cash open 09:30, agents/tal_gap.Data.trend) - each on bot 6's running tracks with exactly their
# rules, so the fix is measured head to head against what trades (the search tries TREND only around its own best).
TREND6 = ("day", "up20", "rth")


def _t6(idx, pn, cfg):
    cfg = TG._norm(dict(cfg))
    fs = [v for v in idx.values() if v["pair"] == pn and TG._norm(dict(v["cfg"])) == cfg and v["sym"] in WORLD["symbols"]]
    if {v["sym"] for v in fs} != set(WORLD["symbols"]):
        return None
    t = pd.concat([pd.read_csv(TAL_DIR / v["file"]) for v in fs], ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    return t


def trend_runs(idx, budget_s):
    """Every running track's own rules, alone and with TREND = day / up20 / rth, on bot 6's three indices - saved in the
    gap method's index as 'search' rows (from "bot6_trend"), like shared()."""
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    want = []
    for pn, cfg in live_tracks().values():
        if pn in pairs:
            c0 = TG._norm({k: v for k, v in cfg.items() if k != "TREND"})
            want += [(pn, c0)] + [(pn, TG._norm(dict(c0, TREND=m))) for m in TREND6]
    t0, done = time.time(), 0
    for sym in WORLD["symbols"]:
        todo = [(pn, c) for pn, c in want if f"{pairs[pn][0]}_{pairs[pn][1]}_{TG.vid(c)}_{sym}" not in idx]
        if not todo or time.time() - t0 > budget_s:
            continue
        try:
            D = tal_load(sym)
        except Exception as e:  # noqa: BLE001
            print(f"tal trend {sym}: no data ({e})")
            continue
        if D is None:
            continue
        gcs = {}
        for pn, c in todo:
            if time.time() - t0 > budget_s:
                break
            p = pairs[pn]
            key = f"{p[0]}_{p[1]}_{TG.vid(c)}_{sym}"
            t = TG.simulate(D, p, c, gcs.setdefault(p[0], {}))
            f = TAL_DIR / f"{key}.csv.gz"
            t.to_csv(f, index=False)
            idx[key] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": p[2],
                        "knob": "search", "val": TG.vid(c), "cfg": c, "sym": sym, "file": f.name, "n": len(t),
                        "from": "bot6_trend"}
            (idx_dir / f"{key}.json").write_text(json.dumps(idx[key], default=str))
            done += 1
    print(f"tal trend (bot 6): {done} run")
    return done


def _day_stats(t):
    tr, un = t[t["time"] < A.SPLIT], t[t["time"] >= A.SPLIT]
    wd = lambda x: round(float(x.groupby(x["time"].dt.strftime("%Y-%m-%d"))["r"].sum().min()), 2) if len(x) else None  # noqa: E731
    return {"train_w": round(TG.score(t), 1), "unseen": round(float(un["r"].sum()), 1), "n_unseen": int(len(un)),
            "mean_unseen": round(float(un["r"].mean()), 3) if len(un) else None,
            "worst_day_train": wd(tr), "worst_day_unseen": wd(un)}


def trend_cards(idx, b5_res=None):
    """Per running track: TREND (day / up20 / rth) against its rules alone. CHOSEN on 2024-2025 (the time-weighted
    total, the search's score); an 'improve' card only when the chosen mode is also better on 2026 in total AND per
    trade and its worst day of 2026 is not worse (principle: more profit, no more risk). Returns (cards, results)."""
    out, res = [], {}
    rk = {**FA.RISK}
    try:
        rk.update(json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("risk") or {})
    except Exception:  # noqa: BLE001
        pass
    b5 = "; ".join(f"{k}: מה שרץ {v.get('live', {}).get('train_w')}R/{v.get('live', {}).get('unseen')}R, " + ", ".join(
        f"{m} {(v.get(m) or {}).get('train_w', '-')}R/{(v.get(m) or {}).get('unseen', '-')}R" for m in ("day", "up20"))
        for k, v in (b5_res or {}).items()) or "TREND לא שיפר בשתי התקופות באף מסלול שנבדק (15 דק'/דקה: לא עדיף ב־2024-2025)"
    for trk, (pn, cfg) in live_tracks().items():
        c0 = TG._norm({k: v for k, v in cfg.items() if k != "TREND"})
        base = _t6(idx, pn, c0)
        if base is None:
            continue
        row = {"live": _day_stats(base)}
        best = None
        for m in TREND6:
            c = TG._norm(dict(c0, TREND=m))
            t = _t6(idx, pn, c)
            if t is None:
                continue
            row[m] = _day_stats(t)
            if row[m]["train_w"] > row["live"]["train_w"] and (best is None or row[m]["train_w"] > row[best[0]]["train_w"]):
                best = (m, c, t)                         # chosen on the train period only
        res[trk] = row
        if len(row) < len(TREND6) + 1 or not best:
            continue
        m, c, t = best
        lv, x = row["live"], row[m]
        if not (x["unseen"] > lv["unseen"] and (x["mean_unseen"] or -9) >= (lv["mean_unseen"] or -9)
                and (x["worst_day_unseen"] or -99) >= (lv["worst_day_unseen"] or -99)):
            continue
        st, bs = TG._stats(t), TG._stats(base)
        hi, lo = pn.split("/")
        mh = {"day": "פתיחת היום", "up20": "הטיים פריים שמעל מול ממוצע 20", "rth": "פתיחת ניו יורק 09:30"}[m]
        out.append(K.card("bot6", "improve", f"שיטת הגאפ · {TG.TF_HE[hi]} / {TG.TF_HE[lo]}: רק בכיוון {mh} ({m})",
                          f"המסלול {trk} עם הכללים שרצים, ועוד מסנן TREND={m}. נבחר על 2024-2025 בלבד (סה\"כ משוקלל "
                          f"{lv['train_w']:+.1f}R ← {x['train_w']:+.1f}R, מתוך day / up20 / rth); 2026 (לא נראתה): סה\"כ "
                          f"{lv['unseen']:+.1f}R ← {x['unseen']:+.1f}R, {x['n_unseen']} עסקאות, {x['mean_unseen']:+.2f}R לעסקה "
                          f"(היום {lv['mean_unseen']:+.2f}R על {lv['n_unseen']}), היום הגרוע {lv['worst_day_unseen']}R ← "
                          f"{x['worst_day_unseen']}R. בסיכון {float(rk.get(trk, FA.RISK_TAL)) * 100:g}%. בבוט 5: {b5}.",
                          st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
                          tested=len(TREND6), risk_pct=round(float(rk.get(trk, FA.RISK_TAL)) * 100, 3), baseline=bs["unseen"],
                          neighbours={"values": list(TREND6), "share_positive": round(float(np.mean(
                              [(row.get(q) or {}).get("unseen", -1e9) > lv["unseen"] for q in TREND6])), 2)},
                          detail={"world": "bot6", "trend": row, "bot5": b5_res},
                          change={"method": "tal_gap", "pair": pn, "cfg": c, "track": f"tal_gap|{pn}", "knob": "cfg"}))
    return out, res


# FILTERS on what runs (bot 6's smart agent, 13.11.2026 night) - shared with bot 5, plus the futures' own clock:
#   MAXC   bot 5's cost filter (13.11.2026, tal_pool_gh): skip a trade whose cost is more than this share of its risk.
#          Micro futures pay a fixed commission per contract, so on the short pairs (5m/1m, 15m/5m) a tight stop can
#          cost a big share of 1R - the same thing that killed forex in bot 5.
#   SESS   bot 5's London / New York hours (lnny, ny) and the futures' own (tal_gap.SESS_NY, New York time): rth = the
#          cash session, nyam = its first two hours (the open's move), xopen = everything but the open's first half hour.
#   EXIT   tp30 - the user's idea (13.11.2026: "take 100% of the profit at 30% of the way to the target"), tested in
#          bot 5 on the wide set; here on bot 6's running tracks.
# Each on every running track with exactly its rules, head to head; chosen on 2024-2025 only (the time-weighted total);
# an 'improve' card only when the chosen one is also better on 2026 in total AND per trade and its worst day of 2026
# is not worse (more profit, no more risk) - so the builder builds it at once.
FILT6 = (("MAXC", 0.1), ("MAXC", 0.2), ("MAXC", 0.3), ("SESS", "lnny"), ("SESS", "ny"), ("SESS", "rth"),
         ("SESS", "nyam"), ("SESS", "xopen"), ("SESS", "xreopen"), ("SESS", "xlate"), ("EXIT", "tp30"), ("EXIT", "p50_50"))
FILT_HE = {"MAXC": lambda v: f"בלי עסקה שהעלות שלה מעל {int(v * 100)}% מהסיכון (MAXC {v})",
           "SESS": lambda v: {"lnny": "רק בשעות לונדון + ניו יורק (07-20 UTC)", "ny": "רק בשעות ניו יורק (12-20 UTC)",
                              "rth": "רק בשעות המסחר של ניו יורק 09:30-16:00", "nyam": "רק בשעתיים הראשונות של ניו יורק 09:30-11:30",
                              "xopen": "בלי חצי השעה הראשונה של ניו יורק 09:30-10:00",
                              "xreopen": "בלי כניסה בשעה הראשונה אחרי ההפסקה היומית 18:00-19:00 ניו יורק",
                              "xlate": "בלי כניסה ב־45 הדקות האחרונות 15:00-15:45 ניו יורק"}[v] + f" (SESS {v})",
           "EXIT": lambda v: {"tp30": "כל העסקה יוצאת ב־30% מהדרך ליעד (tp30, הרעיון של טל)",
                              "p50_50": "חצי החוצה בחצי הדרך ליעד וסטופ לכניסה (p50_50, הבקשה של טל)"}.get(v, f"EXIT {v}")}


# LADDERED ENTRIES (the user's task of 13.11.2026, shared with bot 5 - tal_gap.LADDERS, the same in simulate and in
# live_trades, which bot 6's account trades): 2 / 3 / 3m / 4 entries across the gap's box with the same total risk,
# stop and target, the commission paid per entry. A group of its own, so the answer to the user's task is per ladder
# and not hidden behind a filter that happened to be better on 2024-2025.
LAD6 = (("LADDER", "2"), ("LADDER", "3"), ("LADDER", "3m"), ("LADDER", "4"))
FILT_HE["LADDER"] = lambda v: {"2": "כניסה מדורגת: 2 כניסות לאורך הגאפ", "3": "כניסה מדורגת: 3 כניסות לאורך הגאפ",
                               "3m": "כניסה מדורגת: 3 כניסות, יותר באמצע", "4": "כניסה מדורגת: 4 כניסות לאורך הגאפ"
                               }.get(str(v), f"LADDER {v}") + f" (LADDER {v}, אותו סיכון כולל)"


def _filt_variants(cfg, group=FILT6):
    c0 = TG._norm(dict(cfg))
    return c0, [(k, v, TG._norm(dict(c0, **{k: v}))) for k, v in group if c0.get(k) != v]


# A STOP BY THE MARKET'S VOLATILITY (bot 5's agent, 14.11.2026 - tal_gap.min_stop, the same in the backtest and in
# live_trades, which bot 6's account trades): the stop at least MINSTOP x the ATR(14) of the high timeframe's last
# closed candle from the entry, a wider stop left as it is, the same risk per trade (fewer contracts). From the user's
# question on the sibling trades of 13.11 (a tight stop hit, a wide one held). A group of its own on every running track.
MIN6 = (("MINSTOP", 0.25), ("MINSTOP", 0.5), ("MINSTOP", 1.0))
FILT_HE["MINSTOP"] = lambda v: f"סטופ לפחות {v:g} × ATR של הטיים פריים הגבוה מהכניסה (MINSTOP {v:g}, סטופ לפי התנודתיות)"
# THE USER'S OWN TASKS FIRST (bot 6's smart agent, 14.11.2026): the laddered entries, half at half (p50_50) and the
# volatility stop were queued behind the broad filters and the search, and on 13.11 the bots' computer ran over its
# minutes before reaching them (no result for a day). Their runs now go first, with a budget of their own.
USER6 = LAD6 + (("EXIT", "p50_50"),) + MIN6


def filt_runs(idx, budget_s, group=None):
    """Every running track's own rules, alone and with each FILT6 variant (or the group given), on bot 6's three
    indices - 'search' rows in the gap method's index (from "bot6_filt"), resumed where the last run stopped."""
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    want = []
    for pn, cfg in live_tracks().values():
        if pn in pairs:
            c0, vs = _filt_variants(cfg, group or (FILT6 + LAD6 + MIN6))
            want += [(pn, c0)] + [(pn, c) for _, _, c in vs]
    t0, done = time.time(), 0
    for sym in WORLD["symbols"]:
        todo = [(pn, c) for pn, c in want if f"{pairs[pn][0]}_{pairs[pn][1]}_{TG.vid(c)}_{sym}" not in idx]
        if not todo or time.time() - t0 > budget_s:
            continue
        try:
            D = tal_load(sym)
        except Exception as e:  # noqa: BLE001
            print(f"tal filt {sym}: no data ({e})")
            continue
        if D is None:
            continue
        gcs = {}
        for pn, c in todo:
            if time.time() - t0 > budget_s:
                break
            p = pairs[pn]
            key = f"{p[0]}_{p[1]}_{TG.vid(c)}_{sym}"
            t = TG.simulate(D, p, c, gcs.setdefault(p[0], {}))
            f = TAL_DIR / f"{key}.csv.gz"
            t.to_csv(f, index=False)
            idx[key] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": p[2],
                        "knob": "search", "val": TG.vid(c), "cfg": c, "sym": sym, "file": f.name, "n": len(t),
                        "from": "bot6_filt"}
            (idx_dir / f"{key}.json").write_text(json.dumps(idx[key], default=str))
            done += 1
    print(f"tal filt (bot 6): {done} run, {len(want) * len(WORLD['symbols'])} wanted")
    return done


def _bot5_pool():
    """What bot 5's agent chose for the indices with MAXC / SESS / TREND (its pooled group test, 13.11.2026), per pair."""
    try:
        d = json.loads(Path("reports/agents/bot5/cards_pool.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for c in d.get("cards", []):
        ch = c.get("change") or {}
        if ch.get("pair") and isinstance(ch.get("cfg"), dict) and ch["pair"] not in out:
            m = (c.get("unseen") or {}).get("mean")
            f = ", ".join(f"{k} {ch['cfg'][k]}" for k in ("MAXC", "SESS", "TREND", "EXIT") if ch["cfg"].get(k)) or "בלי מסננים"
            out[ch["pair"]] = f"במדדים של בוט 5 (הגדרת קבוצה): {f}" + (f", 2026 {m:+.2f}R לעסקה" if m is not None else "")
    return out


def filt_cards(idx, group=FILT6):
    """Per running track: each variant of the group (FILT6 / LAD6) against its rules alone; chosen on 2024-2025; a card
    only if 2026 is better in total, per trade and on the worst day. Returns (cards, results)."""
    out, res = [], {}
    rk = {**FA.RISK}
    try:
        rk.update(json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("risk") or {})
    except Exception:  # noqa: BLE001
        pass
    b5 = _bot5_pool()
    for trk, (pn, cfg) in live_tracks().items():
        c0, vs = _filt_variants(cfg, group)
        base = _t6(idx, pn, c0)
        if base is None:
            continue
        row, best = {"live": _day_stats(base)}, None
        for k, v, c in vs:
            t = _t6(idx, pn, c)
            if t is None:
                continue
            row[f"{k}={v}"] = _day_stats(t)
            if row[f"{k}={v}"]["train_w"] > row["live"]["train_w"] and (best is None or row[f"{k}={v}"]["train_w"] > row[best[0]]["train_w"]):
                best = (f"{k}={v}", k, v, c, t)           # chosen on the train period only
        res[trk] = row
        if len(row) < len(vs) + 1 or not best or row["live"]["train_w"] <= -1e8:
            continue                                     # not every variant ran yet / none better on 2024-2025 / too few
        name, k, v, c, t = best
        lv, x = row["live"], row[name]
        if not ((x["mean_unseen"] or 0) > 0 and x["unseen"] > lv["unseen"] and (x["mean_unseen"] or -9) >= (lv["mean_unseen"] or -9)
                and (x["worst_day_unseen"] or -99) >= (lv["worst_day_unseen"] or -99)):
            continue
        st, bs = TG._stats(t), TG._stats(base)
        hi, lo = pn.split("/")
        r_ = float(rk.get(trk, FA.RISK_TAL))
        better = sum(1 for q in row if q != "live" and row[q]["train_w"] > lv["train_w"])
        out.append(K.card("bot6", "improve", f"שיטת הגאפ · {TG.TF_HE[hi]} / {TG.TF_HE[lo]}: {FILT_HE[k](v)}",
                          f"המסלול {trk} עם הכללים שרצים, ועוד: {FILT_HE[k](v)}. נבחר על 2024-2025 בלבד מתוך {len(vs)} מסננים "
                          f"({better} מהם טובים מהיום ב־2024-2025; סה\"כ משוקלל {lv['train_w']:+.1f}R ← {x['train_w']:+.1f}R). "
                          f"2026 (לא נראתה): סה\"כ {lv['unseen']:+.1f}R ← {x['unseen']:+.1f}R, {x['n_unseen']} עסקאות, "
                          f"{x['mean_unseen']:+.2f}R לעסקה (היום {lv['mean_unseen']:+.2f}R על {lv['n_unseen']}), היום הגרוע "
                          f"{lv['worst_day_unseen']}R ← {x['worst_day_unseen']}R. בסיכון {r_ * 100:g}%. בבוט 5: "
                          + (_b5_ladder() if k == "LADDER" else f"{b5.get(pn, 'לזוג הזה אין הגדרת קבוצה עם מסננים')}."),
                          st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
                          tested=len(vs), risk_pct=round(r_ * 100, 3), baseline=bs["unseen"],
                          neighbours={"values": [q for q in row if q != "live"], "share_positive": round(float(np.mean(
                              [row[q]["unseen"] > lv["unseen"] for q in row if q != "live"])), 2)},
                          detail={"world": "bot6", "filters": row, "bot5": b5.get(pn)},
                          change={"method": "tal_gap", "pair": pn, "cfg": c, "track": f"tal_gap|{pn}", "knob": "cfg"}))
    return out, res


# THE USER'S EXPLICIT REQUEST (13.11.2026, twice, notes on a t5_1 US30 trade - "I want it nevertheless"): on the fast
# charts (5m/1m, 15m/5m) take half out at half the way to the target and move the stop to the entry (EXIT p50_50).
# data/agents/tal_thinking.md: a request repeated after 'no change' is a conscious risk preference - no arguing again,
# a card to the manager with the exact numbers and the manager decides. So unlike filt_cards (a card only for the best
# on 2024-2025), every fast running track gets its head-to-head card with full stats, better or worse, and the
# manager's fixed rules judge it (kept / rejected with the numbers the user sees).
ASK6 = {"EXIT": "p50_50"}
ASK_PAIRS = ("5m/1m", "15m/5m")


def ask_cards(idx):
    out, res = [], {}
    rk = {**FA.RISK}
    try:
        rk.update(json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("risk") or {})
    except Exception:  # noqa: BLE001
        pass
    for trk, (pn, cfg) in live_tracks().items():
        if pn not in ASK_PAIRS:
            continue
        c0 = TG._norm(dict(cfg))
        c = TG._norm(dict(c0, **ASK6))
        if c == c0:
            continue
        base, t = _t6(idx, pn, c0), _t6(idx, pn, c)
        if base is None or t is None:
            continue
        lv, x = _day_stats(base), _day_stats(t)
        res[trk] = {"live": lv, "p50_50": x, "exit_now": c0.get("EXIT", "none")}
        st, bs = TG._stats(t), TG._stats(base)
        hi, lo = pn.split("/")
        r_ = float(rk.get(trk, FA.RISK_TAL))
        mt = lambda s_: f"{s_['mean']:+.3f}R" if s_ and s_.get("mean") is not None else "-"  # noqa: E731
        out.append(K.card("bot6", "improve", f"בוט 6 · {TG.TF_HE[hi]} / {TG.TF_HE[lo]} ({trk}): חצי החוצה בחצי הדרך וסטופ לכניסה (הבקשה של טל)",
                          f"טל ביקש פעמיים (13.11, עסקת US30 במסלול t5_1): בגרפים המהירים - ב־50% מהדרך ליעד יוצאים בחצי "
                          f"ומעבירים סטופ לכניסה. נבדק על ההגדרות המדויקות של המסלול {trk} (יציאה היום: {c0.get('EXIT', 'none')}), "
                          f"3 המדדים בעולם של בוט 6. 2024-2025: סה\"כ משוקלל {lv['train_w']:+.1f}R ← {x['train_w']:+.1f}R, "
                          f"לעסקה {mt(bs['train'])} ← {mt(st['train'])}, היום הגרוע {lv['worst_day_train']}R ← {x['worst_day_train']}R. "
                          f"2026 (לא נראתה): סה\"כ {lv['unseen']:+.1f}R ← {x['unseen']:+.1f}R, {x['n_unseen']} עסקאות, "
                          f"לעסקה {mt(bs['unseen'])} ← {mt(st['unseen'])}, היום הגרוע {lv['worst_day_unseen']}R ← {x['worst_day_unseen']}R. "
                          f"בסיכון {r_ * 100:g}%. ההחלטה למנהל לפי הכללים שלו (טל: בקשה מודעת לניהול סיכון). "
                          f"בבוט 5 אותה בקשה נבדקת על t15_5 ו־t5_1 (agents/tal_fix_run).",
                          st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
                          tested=1, risk_pct=round(r_ * 100, 3), baseline=bs["unseen"],
                          detail={"world": "bot6", "by": "טל (הערה על עסקה, פעמיים, 13.11)", "ask": res[trk]},
                          change={"method": "tal_gap", "pair": pn, "cfg": c, "track": f"tal_gap|{pn}", "knob": "cfg"}))
    return out, res


def _b5_ladder():
    """What bot 5's agent found for the laddered entries (its LADDER cards, 13.11.2026)."""
    try:
        cs = [c for nm in ("cards.json", "cards_notes.json", "cards_smart.json") if Path(f"reports/agents/bot5/{nm}").exists()
              for c in json.loads(Path(f"reports/agents/bot5/{nm}").read_text(encoding="utf-8")).get("cards", [])
              if "LADDER" in json.dumps(c.get("change") or {}) and c.get("kind") == "improve"]
    except Exception:  # noqa: BLE001
        cs = []
    if not cs:
        return "הכניסה המדורגת נבדקת על 6 המסלולים שלו, עוד אין כרטיס שיפור."
    return "; ".join(f"{c['title'][:70]} ({(c.get('unseen') or {}).get('mean', 0):+.2f}R לעסקה ב־2026)" for c in cs[:3]) + "."


# THE SHARED EXIT (the user's task of 13.11.2026, shared with bot 5 - tal_gap.trade_paths / shared_exit / shared_test):
# bot 6 as it trades now (every gap track at risk > 0, its own risk, the same entries, each trade keeps its own stop and
# target), every open trade closed at once when the open trades together reach the target (+0.5..2R of the open risk,
# or +0.25..1% of the account, at least 2-3 open), against the bot as it is - with bot 6's own account (FTMO Futures:
# 1.5% a day, 4.5% trailing, fut_lim). Chosen on 2024-2025 (more total %, no deeper fall), shown on 2026.
SHARED6_DIR = TAL_DIR / "shared"


def _shared6_live():
    import hashlib
    rk = {**FA.RISK}
    try:
        rk.update(json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("risk") or {})
    except Exception:  # noqa: BLE001
        pass
    live = {k: (pn, TG._norm(dict(c))) for k, (pn, c) in live_tracks().items() if float(rk.get(k, FA.RISK_TAL) or 0) > 0}
    risk = {k: float(rk.get(k, FA.RISK_TAL)) * 100 for k in live}
    h = hashlib.sha1(json.dumps({k: [v[0], v[1], risk[k]] for k, v in sorted(live.items())}, sort_keys=True,
                                default=str).encode()).hexdigest()[:10]
    return h, live, risk


def shared6_runs(budget_s):
    import pickle
    h, live, _ = _shared6_live()
    if not live:
        return 0
    SHARED6_DIR.mkdir(parents=True, exist_ok=True)
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    t0, done = time.time(), 0
    for sym in WORLD["symbols"]:
        f = SHARED6_DIR / f"{h}_{sym}.pkl"
        if f.exists() or time.time() - t0 > budget_s:
            continue
        D, gc = tal_load(sym), {}
        if D is None:
            print(f"tal shared exit (bot 6) {sym}: no data")
            continue
        rows = []
        for trk, (pn, cfg) in live.items():
            if pn not in pairs:
                continue
            keep = []
            t = TG.simulate(D, pairs[pn], cfg, gc.setdefault(pn.split("/")[0], {}), keep=keep)
            if len(keep) != len(t):                      # a laddered track keeps no path - the test would be wrong
                print(f"tal shared exit (bot 6) {trk}: no minute paths ({len(keep)} of {len(t)})")
                continue
            for (tt, vv), x in zip(TG.trade_paths(D, keep), t.itertuples()):
                rows.append({"trk": trk, "sym": sym, "time": x.time, "r": float(x.r), "mins": int(x.mins), "t": tt, "v": vv})
        f.write_bytes(pickle.dumps(rows))
        done += 1
    print(f"tal shared exit (bot 6): {done} instruments run")
    return done


def shared6_cards():
    """(cards, results) once the three indices' paths are there."""
    import pickle
    h, live, risk = _shared6_live()
    if not live:
        return [], {"state": "no tracks"}
    miss = [sy for sy in WORLD["symbols"] if not (SHARED6_DIR / f"{h}_{sy}.pkl").exists()]
    if miss:
        return [], {"state": "running", "missing": miss}
    rows = []
    for sy in WORLD["symbols"]:
        rows += pickle.loads((SHARED6_DIR / f"{h}_{sy}.pkl").read_bytes())
    if not rows:
        return [], {"state": "no trades"}
    t = pd.DataFrame([{k: v for k, v in r.items() if k not in ("t", "v")} for r in rows])
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    t["out"] = t["time"] + pd.to_timedelta(t["mins"].astype(float) + 1, unit="min")
    order = np.argsort(t["time"].to_numpy(), kind="stable")
    t = t.iloc[order].reset_index(drop=True)
    items = [{"t": rows[i]["t"], "v": rows[i]["v"], "r": rows[i]["r"]} for i in order]
    res = TG.shared_test(t, items, risk, lim=fut_lim())
    res.update(state="done", tracks=len(live), instruments=len(WORLD["symbols"]), n=int(len(t)))
    try:            # what bot 5's agent found (its shared-exit result in tal_gap_results.json or its card)
        b5 = ((json.loads(TG.RESULTS.read_text(encoding="utf-8")).get("bot5") or {}).get("shared_exit") or {})
        b5t = ("בבוט 5: " + (f"נבחר {b5.get('chosen')}, " if b5.get("chosen") else "אף יעד לא עדיף ב־2024-2025, ")
               + ("עדיף גם ב־2026" if b5.get("better") else "לא עדיף ב־2026")) if b5.get("state") == "done" else "בבוט 5 הבדיקה עוד רצה"
    except Exception:  # noqa: BLE001
        b5t = "בבוט 5 הבדיקה עוד רצה"
    res["bot5"] = b5t
    if not res["better"]:
        return [], res
    k = res["chosen"]
    a, b = res["rows"]["now"], res["rows"][k]
    mode, rest = ("r", k[1:]) if k.startswith("r") else ("pct", k[3:])
    x, m = rest.split("_m")
    what = f"+{x}R מהסיכון הפתוח" if mode == "r" else f"+{x}% מהחשבון"
    ws = t["trk"].map(risk).fillna(0).to_numpy(float) / 100
    u = t.copy()
    u["r"] = [r_ for r_, _ in TG.shared_exit([dict(it, w=w_) for it, w_ in zip(items, ws)], mode, float(x), int(m))]
    u["cost_r"], t["cost_r"] = 0.0, 0.0
    st, bs = A.stats(u), A.stats(t)
    card = K.card(
        "bot6", "improve", f"בוט 6 · יציאה משותפת: לסגור את כל העסקאות הפתוחות ב־{what} (לפחות {m} פתוחות)",
        f"הבקשה של טל מ־13.11. נבדק על בוט 6 כפי שהוא ({len(live)} מסלולים של שיטת הגאפ, 3 המדדים, הסיכון האמיתי, "
        f"חשבון FTMO Futures: 1.5% ביום, 4.5% מהשיא), כל עסקה שומרת על הסטופ והיעד שלה. נבחר על 2024-2025 מתוך "
        f"{len(TG.SHARED_VARIANTS)} יעדים (סה\"כ {a['train']['total_pct']:+.1f}% ← {b['train']['total_pct']:+.1f}%, ירידה "
        f"{a['train']['max_fall_pct']:.1f}% ← {b['train']['max_fall_pct']:.1f}%). 2026 (לא נראתה): סה\"כ {a['unseen']['total_pct']:+.1f}% ← "
        f"{b['unseen']['total_pct']:+.1f}%, היום הגרוע {a['unseen']['worst_day_pct']:+.2f}% ← {b['unseen']['worst_day_pct']:+.2f}%, "
        f"ירידה {a['unseen']['max_fall_pct']:.1f}% ← {b['unseen']['max_fall_pct']:.1f}%. נסגרו מוקדם {b['closed_early_unseen']} "
        f"עסקאות ב־2026. {b5t}. צריך קוד בבוט החי (live/watch_own.py, נבנה אחרי אישור).",
        st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], baseline=bs["unseen"],
        tested=len(TG.SHARED_VARIANTS), risk_pct=round(max(risk.values()), 3) if risk else None,
        detail={"world": "bot6", "shared_exit": res},
        change={"track": "tal_all", "knob": "SHARED_EXIT", "to": {"mode": mode, "x": float(x), "min_open": int(m)}, "code": True})
    return [card], res


def fut_lim():
    """Bot 6's own account limits for the whole-bot test, exactly as live/fut_account.py keeps the account (13.11.2026
    night, bot 5's task b5to6room1311): FTMO Futures - a day's loss of DAILY_LOSS, a trailing maximum loss MAX_LOSS
    under the best end-of-day balance (it stops rising at the start), ROOM_USE of the room left with every open trade's
    full risk inside it, the futures' trading day from 18:00 New York; a day past the daily loss fails the account.
    Before this the test used bot 5's 4.5% / 9% - three times the daily room bot 6 really has."""
    return {"day": FA.DAILY_LOSS / FA.INITIAL * 100, "all": FA.MAX_LOSS / FA.INITIAL * 100, "trail": True,
            "use": FA.ROOM_USE, "day_hard": True, "dayf": FA.tday}


def scale_cards(pf, rk):
    """Bot 5's account-in-% rule (13.11.2026 night), shared: one factor on every gap track's risk, chosen on 2024-2025
    so the account stays within 80% of the bot's limits (tal_gap.risk_scale). Bot 6 keeps a risk per track in
    live/bots/bot6.json, so the factor is one 'risk' card per track (the builder builds each at once)."""
    a = (pf or {}).get("account") or {}
    sc = a.get("scale") or {}
    k = sc.get("chosen", 1.0)
    lm = sc.get("limits") or {}
    if not sc or k >= 1.0 or str(k) not in (sc.get("rows") or {}):
        return []
    nt, nu = a["now"]["train"] or {}, a["now"]["unseen"] or {}
    bt, bu = sc["rows"][str(k)].get("train") or {}, sc["rows"][str(k)].get("unseen") or {}
    line = lambda x: (f"סה\"כ {x.get('total_pct', 0):+.1f}%, היום הגרוע {x.get('worst_day_pct', 0):+.1f}%, "  # noqa: E731
                      f"הירידה הגדולה {x.get('max_fall_pct', 0):.1f}%, עסקאות שדולגו / הוקטנו בגלל המקום "
                      f"{x.get('room_skipped', 0)} / {x.get('room_cut', 0)}" + (f", החשבון נכשל ב־{str(x['stopped'])[:10]}"
                                                                              if x.get("stopped") else ""))
    try:            # what bot 5's agent found with the same rule (its RISK_SCALE card)
        b5 = next((c for c in json.loads(Path("reports/agents/bot5/cards.json").read_text(encoding="utf-8")).get("cards", [])
                   if (c.get("change") or {}).get("knob") == "RISK_SCALE"), None)
        b5t = f"בבוט 5 אותו כלל נתן × {b5['change']['to']} (כרטיס {b5['id']})" if b5 else "בבוט 5 אין כרגע כרטיס מקדם"
    except Exception:  # noqa: BLE001
        b5t = "בבוט 5 אותו כלל נבדק על המסלולים שלו"
    fail = bool(nt.get("stopped") or nu.get("stopped") or nu.get("total_pct", 0) <= -0.9 * lm.get("all", 4.5))
    out = []
    for trk, r in rk.items():
        to = round(float(r) * k, 5)
        out.append(K.card("bot6", "risk", f"שיטת הגאפ בבוט 6, מסלול {trk}: סיכון {float(r) * 100:g}% ← {to * 100:g}% (× {k})",
                          f"כל מסלולי שיטת הגאפ של בוט 6 יחד, בסיכון האמיתי של כל מסלול ועם המגבלות של החשבון כמו בחי "
                          f"(FTMO Futures: {lm.get('day', 1.5):g}% ביום, {lm.get('all', 4.5):g}% מתחת לשיא של סוף יום, "
                          f"עסקה חדשה רק בתוך 90% מהמקום שנשאר כולל הסיכון של העסקאות הפתוחות). היום: 2024-2025: {line(nt)}; "
                          f"2026: {line(nu)}. הכלל של בוט 5 (משותף): המקדם הגדול ביותר שבו 2024-2025 נשאר ב־80% מהמגבלות "
                          f"(ירידה ≤ {0.8 * lm.get('all', 4.5):.1f}%, יום ≥ -{0.8 * lm.get('day', 1.5):.1f}%), נבחר רק על 2024-2025 → × {k}. "
                          f"עם המקדם: 2024-2025: {line(bt)}; 2026 (לא נראתה): {line(bu)}. אסטרטגיות הסייר (0.28% כל אחת) "
                          f"מעל זה - לא בחישוב. " + ("" if sc.get("fits") else "גם במקדם הזה 2024-2025 לא נכנס ב־80% מהמגבלות - "
                          "צריך גם פחות מסלולים או מסלולים חזקים יותר. ") + "חוזה מיקרו אחד הוא המינימום: עסקה שחוזה אחד שלה "
                          f"מסכן יותר מ־{to * 100:g}% מהחשבון לא נלקחת (בעיקר שעה / 5 דק' על US30 עם סטופ רחב), אז הרווח בחי "
                          f"יהיה קצת מתחת לבדיקה. {b5t}.",
                          None, None, tested=1, risk_pct=round(to * 100, 4),
                          # account_fit (13.11.2026 night): the risk the account's own fixed limits allow, not a cut of a
                          # losing strategy - agents/autodecide does not hold it for "fix first"
                          detail={"world": "bot6", "account": a, "account_fit": True, "account_fails_now": fail},
                          change={"track": trk, "knob": f"RISK_{trk}", "from": float(r), "to": to}))
    return out


STEPS = {}                 # this run's gap-method steps: {name: {done | err, at, sec, budget_sec}} -> results bot6.steps


def tal(budget_s):
    t0 = time.time()
    idx0 = {f.stem: json.loads(f.read_text()) for f in (TAL_DIR / "idx").glob("*.json")} if (TAL_DIR / "idx").exists() else {}
    # each step's outcome into the results (bot6.steps, 14.11.2026, as bot 5): the bots' computer keeps its logs to
    # itself, so a step that failed or found no time was invisible to the nightly smart agent in the cloud
    for nm, fn, share in (("user_tasks", lambda b: filt_runs(idx0, b, USER6), 0.4),   # the user's tasks first (14.11)
                          ("trend_fix", lambda b: trend_runs(idx0, b), 0.15),        # TREND against what runs (13.11)
                          ("filters", lambda b: filt_runs(idx0, b), 0.2)):           # cost / hours / clock filters (13.11 night)
        t1 = time.time()
        try:
            STEPS[nm] = {"done": fn(budget_s * share)}
        except Exception as e:  # noqa: BLE001
            STEPS[nm] = {"err": str(e)[:300]}
            print(f"tal {nm} failed: {e}")
        STEPS[nm].update(at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), sec=round(time.time() - t1),
                         budget_sec=round(budget_s * share))
    t1 = time.time()
    idx, left = TG.run("bot6", WORLD["symbols"], tal_load, TAL_DIR, budget_s * 0.8 - (time.time() - t0))
    STEPS["search"] = {"done": left, "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                       "sec": round(time.time() - t1), "note": "done = runs left"}
    shared(idx, budget_s - (time.time() - t0))
    pairs, answers = TG.summarize(idx, WORLD["symbols"], TAL_DIR)
    planned = len(TG.PAIRS) * len(TG.variants()) * len(WORLD["symbols"])
    doc = TG.write_results("bot6", pairs, answers, WORLD, [planned - left, planned], out_dir=TAL_DIR)
    run = live_tracks()
    pcards = []
    try:            # the bot as a whole, the way the account takes its tracks (only tracks at risk > 0, as bot 5)
        rk = {**FA.RISK, **(json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json")
                                       .read_text(encoding="utf-8")).get("risk") or {})}
        counted = {k: v for k, v in run.items() if float(rk.get(k, FA.RISK_TAL) or 0) > 0}
        # the hour filter and the direction cap are bot 5's methods (13.11.2026), shared: chosen on 2024-2025 only.
        # Bot 6 has ~600 gap trades to 2025 against bot 5's ~5,000, so an hour needs 30 trades here (100 there),
        # with the same t <= -1.5
        # the account in % with each track's real risk (bot 5's rule, 13.11.2026 night, shared): the bot's limits
        # in % - not in R - and one factor on the tracks' risk chosen on 2024-2025 (scale_cards)
        pf = TG.portfolio(idx, WORLD["symbols"], TAL_DIR, counted, hour_min_n=30,
                          risk={k: float(rk.get(k, FA.RISK_TAL)) for k in counted}, lim=fut_lim())
        if pf:
            try:
                pcards += scale_cards(pf, {k: float(rk.get(k, FA.RISK_TAL)) for k in counted})
            except Exception as e:  # noqa: BLE001
                print(f"tal_gap scale failed: {e}")
            doc["bot6"]["portfolio"] = pf
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            pf5 = (doc.get("bot5") or {}).get("portfolio") or {}
            for pc in (TG.portfolio_card("bot6", pf, FA.RISK_TAL * 100), TG.hours_card("bot6", pf, FA.RISK_TAL * 100)):
                if not pc:
                    continue
                h5 = (pf5.get("hours_off") or {}).get("hours")
                pc["what"] += (" · בבוט 5: " + (("שעות " + ", ".join(f"{x:02d}" for x in h5)) if h5 else
                                                "אין שעה שהפסידה מספיק" if "hours_off" in pf5 else "עוד לא נבדק")
                               if pc["change"]["knob"] == "HOURS_OFF" else " · בבוט 5 אותו כלל נבדק על המסלולים שלו")
                pc["detail"]["bot5"] = {k: pf5.get(k) for k in ("hours_off", "max_dir", "one_per_side")}
                pcards.append(pc)
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap portfolio failed: {e}")
    try:            # the volatility stop on what runs, head to head (bot 5's MINSTOP, 14.11.2026)
        mc, mres = filt_cards(idx, MIN6)
        pcards += mc
        if mres:
            doc["bot6"]["minstop_fix"] = mres
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap minstop cards failed: {e}")
    doc["bot6"]["steps"] = STEPS
    TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    try:            # TREND on what runs, head to head (13.11.2026)
        tc, tres = trend_cards(idx, (doc.get("bot5") or {}).get("trend_fix"))
        pcards += tc
        if tres:
            doc["bot6"]["trend_fix"] = tres
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap trend cards failed: {e}")
    try:            # the filters on what runs, head to head (13.11.2026 night)
        fc, fres = filt_cards(idx)
        pcards += fc
        if fres:
            doc["bot6"]["filt_fix"] = fres
            doc["bot6"]["filt_bot5"] = _bot5_pool()
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap filt cards failed: {e}")
    try:            # laddered entries on what runs, head to head (the user's task, 13.11.2026)
        lc, lres = filt_cards(idx, LAD6)
        pcards += lc
        if lres:
            doc["bot6"]["ladder_fix"] = lres
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap ladder cards failed: {e}")
    try:            # the shared exit on the whole bot (the user's task, 13.11.2026)
        t1 = time.time()
        STEPS["shared_exit"] = {"done": shared6_runs(budget_s * 0.3), "sec": round(time.time() - t1),
                                "budget_sec": round(budget_s * 0.3)}
        sc_, sres = shared6_cards()
        pcards += sc_
        doc["bot6"]["shared_exit"] = sres
        TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap shared exit failed: {e}")
    try:            # the user's explicit request on the fast charts, head to head with full numbers (13.11.2026 night)
        ac, ares = ask_cards(idx)
        pcards += ac
        if ares:
            doc["bot6"]["ask_p50_50"] = ares
            TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap ask cards failed: {e}")
    # a card on a pair bot 6 already trades carries what runs as its baseline (approved only if 2026 is better per
    # trade AND in total); a new pair must have earned on 2024-2025 and on 2026 (it must add to the bot as a whole)
    return TG.cards("bot6", pairs, doc.get("bot5"), "בוט 5", risk_pct=FA.RISK_TAL * 100,
                    live={pn: c for pn, c in run.values()}) + pcards


SCOUT_DIR = A.OUT / "scout_fut"
SCOUT_SHARE = 0.25              # of each run's minutes (13.11.2026: the approved scout strategies, on bot 6's world)
SCOUT_RERUN_DAYS = 14


# DISCOVER (bot 6's smart agent, 13.11.2026 night): the scout's strategies built for the index futures' New York session
# (the cash open 09:30, its first candles / range, VWAP, the daily pivot) - none was approved for bot 6 on the CFD test,
# so none was ever run on bot 6's own world. Same fair test as the approved ones (96 variants, chosen on 2024-2025,
# shown on 2026); one that passes is a 'new' card like any other, one that does not is only recorded here (it is not
# a fix attempt: the bot never traded it). Newest first; the run resumes where the last one stopped.
# 13.11.2026 morning: the scout's index strategies of 13.11 join - the session-open ones first (SC073 the first 5-minute
# bar's breakout, SC077 the 15-minute opening range with a tight stop, SC078 the NQ Keltner pulse), then the
# Bollinger / Keltner ones (SC068, SC070, SC071). SC058: no variant earned on 2024-2025 and on 2026, on any index.
# 13.11.2026 late night: SC104 (the scout's US30 1-minute Parabolic SAR flip after 13:00 London, 25 / 50 points,
# marked for bots 5-6) first - built for the Dow on one minute, the bot's own instrument and the fastest track.
DISCOVER = [f"{sid}|{tf}|{sym}" for sid, tfs in (("SC104", ("1Min",)), ("SC058", ("5Min", "15Min", "1Min")), ("SC063", ("5Min",)),
                                                 ("SC073", ("5Min",)), ("SC077", ("5Min", "1Min")), ("SC078", ("15Min",)),
                                                 ("SC011", ("1Min",)), ("SC052", ("1Min",)), ("SC003", ("1Min",)),
                                                 ("SC047", ("1Min",)), ("SC021", ("5Min",)), ("SC028", ("5Min",)),
                                                 ("SC068", ("5Min",)), ("SC070", ("5Min", "15Min")), ("SC071", ("5Min", "15Min")))
            for tf in tfs for sym in ("US100", "US500", "US30")]


def _approved_scout():
    """The scout strategies the manager approved for bot 6 on the CFD test (change {"combo", "kind": "add"}), not undone,
    plus every bot 6 combo with a fix task (data/agents/fix_attempts.json) - each is tested on bot 6's world here."""
    out = {}
    try:
        dec = json.loads(Path("reports/agents/decisions.json").read_text(encoding="utf-8")).get("decisions", [])
        cards = {}
        for f in A.OUT.glob("cards*.json"):
            for c in json.loads(f.read_text(encoding="utf-8")).get("cards", []):
                cards[c.get("id")] = c
        undone = set(RIO.live_json("live/agent_decisions_state.json").get("undo") or [])
        for d in dec:
            c = cards.get(d.get("card")) or {}
            ch = c.get("change") or {}
            if d.get("bot") == "bot6" and d.get("decision") == "approved" and d.get("id") not in undone \
                    and ch.get("kind") == "add" and str(ch.get("combo", "")).count("|") == 2:
                out[ch["combo"]] = d.get("card")
        fx = json.loads(Path("data/agents/fix_attempts.json").read_text(encoding="utf-8"))
        for k in fx:
            if k.startswith("bot6|") and k.count("|") == 3:
                out.setdefault(k.split("|", 1)[1], None)
    except Exception as e:  # noqa: BLE001
        print(f"scout_fut: approved list not read: {e}")
    for combo in DISCOVER:
        out.setdefault(combo, "discover")
    return out


def scout_world(budget_s):
    """Every approved scout strategy on bot 6's own world (agents/scout_fut.py): 96 variants each, chosen on
    2024-2025, shown on 2026. Writes reports/agents/bot6/scout_fut.json, the fix attempts, and returns cards."""
    from agents import scout_fut as SF
    t0 = time.time()
    SCOUT_DIR.mkdir(parents=True, exist_ok=True)
    fres = A.OUT / "scout_fut.json"
    try:
        res = json.loads(fres.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        res = {}
    combos = _approved_scout()
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = {}
    now = pd.Timestamp.now(tz="UTC")
    for combo, card_id in combos.items():
        sid, tf, sym = combo.split("|")
        old = res.get(combo) or {}
        if old.get("at") and (now - pd.Timestamp(old["at"])).days < SCOUT_RERUN_DAYS and old.get("done"):
            continue
        if sym not in FA.SPEC or tf not in SF.TF_MIN or time.time() - t0 > budget_s:
            continue
        if sym not in data:
            try:
                p = prep(sym, end)
                if p is None:
                    data[sym] = None
                else:
                    df, _, _, noent = p
                    shut, _ = FA.session_masks(df)
                    _, pv, tick = FA.SPEC[sym]
                    data[sym] = (df, noent, shut, tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv)
            except Exception as e:  # noqa: BLE001
                print(f"scout_fut {sym}: no data ({e})")
                data[sym] = None
        if data[sym] is None:
            continue
        df, noent, shut, cost_px = data[sym]
        rows = dict(old.get("variants") or {})
        for v in SF.variants():
            k = SF.vid(v)
            if k in rows:
                continue
            if time.time() - t0 > budget_s:
                break
            try:
                t = SF.trades(df, sym, sid, tf, v, noent, shut, cost_px)
            except Exception as e:  # noqa: BLE001
                rows[k] = {"v": v, "error": str(e)[:200]}
                continue
            if not t.empty:     # a trade whose one micro contract risks more than the bot's risk is never taken live
                _, pv_, tick_ = FA.SPEC[sym]
                per = (t["entry_px"] - t["stop_px"]).abs() * pv_ + FA.COMM + FA.SLIP_TICKS * tick_ * pv_
                t = t[per <= FA.INITIAL * FA.RISK_TAL]
            if t.empty:
                rows[k] = {"v": v, "train": K.wstats([]), "unseen": K.wstats([])}
                continue
            tt = pd.to_datetime(t["time"], utc=True)
            tr, un = t[tt < A.SPLIT], t[tt >= A.SPLIT]
            rows[k] = {"v": v, "train": K.wstats(tr["r"], K.weights(tr["time"])) if len(tr) else K.wstats([]),
                       "unseen": K.wstats(un["r"], K.weights(un["time"])) if len(un) else K.wstats([]),
                       "unseen_sum": round(float(un["r"].sum()), 2), "cost_r": round(float(t["cost_r"].median()), 3),
                       "halves": K.halves(tr["time"], tr["r"]) if len(tr) else [None, None]}
            if k == SF.vid(SF.BASE):
                t.to_csv(SCOUT_DIR / f"{combo.replace('|', '_')}_base.csv.gz", index=False)
        done = all(SF.vid(v) in rows for v in SF.variants())
        ok = [x for x in rows.values() if (x.get("train") or {}).get("mean") is not None and (x["train"].get("eff_n") or 0) >= 20]
        best = max((x for x in ok if x["train"]["mean"] > 0), key=lambda x: x["train"]["mean"] * np.sqrt(x["train"]["eff_n"]),
                   default=None)
        res[combo] = {"at": now.isoformat(timespec="seconds"), "card": card_id, "done": done, "variants": rows,
                      "base": rows.get(SF.vid(SF.BASE)), "chosen": best,
                      "share_positive_train": round(sum(1 for x in ok if x["train"]["mean"] > 0) / len(ok), 2) if ok else None}
        print(f"  scout_fut {combo}: {len(rows)} variants, chosen {SF.vid(best['v']) if best else None}")
    fres.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return scout_cards(res)


def scout_cards(res):
    """A strategy that earns on bot 6's world in 2024-2025 with a variant chosen there (and stays positive on 2026) ->
    a 'new' card the builder builds into live/bots/bot6.json "scout"; none -> the fix attempt failed (an info card)."""
    from agents import scout_fut as SF
    cards, fx_f = [], Path("data/agents/fix_attempts.json")
    try:
        fx = json.loads(fx_f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        fx = {}
    try:
        b5 = {(c.get("change") or {}).get("combo"): c for c in json.loads(Path("reports/agents/bot5/cards_new.json").read_text(encoding="utf-8")).get("cards", [])}
    except Exception:  # noqa: BLE001
        b5 = {}
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    for combo, x in res.items():
        if not x.get("done"):
            continue
        ch, base = x.get("chosen"), x.get("base") or {}
        c5 = b5.get(combo)
        s5 = (f" · בבוט 5 (CFD): {(c5.get('unseen') or {}).get('mean') or 0:+.2f}R ב־2026, {c5.get('tier')}" if c5
              else " · בבוט 5 לא נבדקה על המדד הזה")
        bt, bu = base.get("train") or {}, base.get("unseen") or {}
        fmt = lambda s: f"{s.get('mean') or 0:+.2f}R על {s.get('n', 0)} עסקאות"  # noqa: E731
        win = ch and (ch["unseen"].get("mean") or 0) > 0
        key = f"bot6|{combo}"
        disc = x.get("card") == "discover"
        if disc and not win:
            continue            # never traded by the bot: nothing to fix, no card (the result stays in scout_fut.json)
        if not disc and (key in fx or not win):
            e = fx.setdefault(key, {"status": "trying", "tried": [], "at": today})
            e["tried"] = [t for t in e.get("tried", []) if t.get("what") != "bot6_world"] + [{
                "what": "bot6_world", "at": today, "variants": len(x["variants"]),
                "chosen": SF.vid(ch["v"]) if ch else None, "train": (ch or {}).get("train"), "unseen": (ch or {}).get("unseen")}]
            e["status"], e["at"] = ("fixed" if win else "failed"), today
        if win:
            v = ch["v"]
            cards.append(K.card("bot6", "new", f"אסטרטגיה חדשה בעולם של בוט 6: {combo} · {SF.he(v)}",
                                f"נבדקה על החוזים (עלויות מיקרו, שעות FTMO Futures, בלי כניסה סביב הודעות, איזון לפי הכלל), "
                                f"{len(x['variants'])} גרסאות. נבחרה על 2024-2025 בלבד: {fmt(ch['train'])}; ב־2026 (לא נראתה): "
                                f"{fmt(ch['unseen'])}. כמו במקור: עד 2025 {fmt(bt)}, ב־2026 {fmt(bu)}. "
                                f"עלות {ch.get('cost_r')}R לעסקה. בסיכון של הבוט (0.4%).{s5}",
                                ch["train"], ch["unseen"], halves_=ch.get("halves"), tested=len(x["variants"]),
                                neighbours={"share_positive": x.get("share_positive_train")}, cost_r=ch.get("cost_r"),
                                risk_pct=FA.RISK_TAL * 100, change={"family": "scout_fut", "combo": combo, "v": v},
                                detail={"world": "bot6", "approved_cfd_card": x.get("card"), "base": base, "bot5": (c5 or {}).get("unseen")}))
        else:
            cards.append(K.card("bot6", "info", f"{combo} בעולם של בוט 6: אין גרסה שמרוויחה גם עד 2025 וגם ב־2026",
                                f"{len(x['variants'])} גרסאות (יעד, סטופ, בלי יציאה בזמן, איזון) על החוזים בשעות FTMO Futures. "
                                f"כמו במקור: עד 2025 {fmt(bt)}, ב־2026 {fmt(bu)}. "
                                + (f"הטובה עד 2025 ({SF.he(ch['v'])}): {fmt(ch['train'])}, אבל ב־2026 {fmt(ch['unseen'])}. " if ch else "אף גרסה לא הרוויחה עד 2025. ")
                                + f"ניסיון התיקון נכשל (fix_attempts: failed) - לא נבנית בבוט 6.{s5}",
                                None, None, tested=len(x["variants"]),
                                detail={"world": "bot6", "approved_cfd_card": x.get("card"), "base": base, "chosen": ch}))
    fx_f.write_text(json.dumps(fx, ensure_ascii=False, indent=1), encoding="utf-8")
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=None)
    a, _ = ap.parse_known_args()
    mins = a.minutes if a.minutes is not None else float(A.plan().get("minutes", 30))
    t0, tcards = time.time(), []
    try:
        tcards = tal(TAL_SHARE * mins * 60)
    except Exception as e:  # noqa: BLE001 - the user's method must not stop the nightly experiments
        print(f"tal_gap failed: {e}")
    scards = []
    try:            # the scout strategies approved for bot 6, on bot 6's own world (13.11.2026)
        scards = scout_world(SCOUT_SHARE * mins * 60)
    except Exception as e:  # noqa: BLE001
        print(f"scout_fut failed: {e}")
    rest = max(1.0, mins - (time.time() - t0) / 60)
    argv = [x for i, x in enumerate(sys.argv) if x != "--minutes" and (i == 0 or sys.argv[i - 1] != "--minutes")]
    sys.argv = argv + ["--minutes", f"{rest:.1f}"]
    A.main()
    f = A.OUT / "cards.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    for c in tcards + scards:
        c["tier"], c["verdict"] = K.judge(c)
    d["cards"] = [c for c in d.get("cards", []) if (c.get("change") or {}).get("method") != "tal_gap"
                  and (c.get("change") or {}).get("track") != "tal_all"
                  and (c.get("change") or {}).get("family") != "scout_fut" and (c.get("detail") or {}).get("world") != "bot6"] + tcards + scards
    # one card per id (bot 6's smart agent, 13.11.2026 late night): the same change from two tests (the search's best
    # pair version and the head-to-head filter test, both "t5_1 + MAXC 0.2") has the same id, and the manager approved
    # the first copy and then withdrew it for the second, lower-tier copy (38531db8b5eb, 13.11 16:50) - the change
    # was never built. Keep the copy with the better tier (the first among equals).
    rank = {"strong": 0, "regular": 1, "watch": 2, "rejected": 3}
    best = {}
    for i, c in enumerate(d["cards"]):
        k = c.get("id")
        if k not in best or rank.get(c.get("tier"), 4) < rank.get(d["cards"][best[k]].get("tier"), 4):
            best[k] = i
    d["cards"] = [c for i, c in enumerate(d["cards"]) if best.get(c.get("id")) == i]
    view = bot5_view()
    for c in d.get("cards", []):
        ch = c.get("change") or {}
        b5 = view.get(f"{ch.get('track')}|{ch.get('knob')}|{ch.get('to')}")
        if b5:
            c["detail"]["bot5"] = b5
            c["what"] += f" · בבוט 5 אותו שינוי: {b5['tier']}, {b5['unseen']:+.2f}R בתקופה שלא נראתה" if b5.get("unseen") is not None else ""
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
