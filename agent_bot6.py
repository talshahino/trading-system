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
    # and each one with FTMO's break rule on, the way it would trade live here (bot 6's smart agent, 15.11.2026:
    # live/watch_own.py trades TG.ftmo_cfg(cfg), so a shared method measured without the rule is not what would run)
    want = [(pn, c) for pn, cfg in want if isinstance(cfg, dict) for c in (cfg, TG.ftmo_cfg(cfg))]
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


# THE GAP METHOD'S PAIR CARDS ON THE RULES AS THEY TRADE (bot 6's smart agent, 15.11.2026): the search (TG.run) and the
# shared rows choose each pair's version without FTMO's break rule, but live/watch_own.py trades TG.ftmo_cfg(cfg) and
# the builder REPLACES a track's rules with the card's cfg - so (1) a pair card showed 2026 numbers the live bot cannot
# have (the rule took 1h/5m from +45.6R to +20.1R and 4h/1h from +31.1R to +2.3R before REARM), and (2) an approved
# search card on 1h/5m or 4h/1h would have dropped the REARM the manager approved on 15.11 (3ced409700a5, 4a76ea371e30).
# Now every pair's train-chosen version is run with the rule on, alone and with REARM, and the pair's card is chosen
# only among versions with the rule on (by the train total, as summarize) - what the card shows is what would trade.
def ftmo_runs(idx, budget_s):
    """Each pair's chosen version of the last results (and the live tracks' pairs first) with FTMO's break rule on,
    alone and + REARM, on the three indices - 'search' rows from "bot6_ftmo", resumed where the last run stopped."""
    try:
        prev = ((json.loads(TG.RESULTS.read_text(encoding="utf-8")).get("bot6") or {}).get("pairs") or {})
    except Exception:  # noqa: BLE001
        return 0
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    live = {pn for pn, _ in live_tracks().values()}
    order = []
    for pn, x in prev.items():
        ch = x.get("chosen") or {}
        tr, un = ch.get("train") or {}, ch.get("unseen") or {}
        if pn not in pairs or not isinstance(ch.get("cfg"), dict) or (tr.get("mean") or 0) <= 0:
            continue
        order.append((pn not in live, -(tr.get("mean") or 0) * (tr.get("eff_n") or 0), pn, ch["cfg"]))
    want = []
    for _, _, pn, cfg in sorted(order, key=lambda z: z[:2]):
        for c in (TG.ftmo_cfg(cfg), TG.ftmo_cfg(dict(cfg, REARM=True))):
            if (pn, c) not in want:
                want.append((pn, c))
    t0, done, data, gcs = time.time(), 0, {}, {}
    idx_dir = TAL_DIR / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    for pn, c in want:
        p = pairs[pn]
        for sym in WORLD["symbols"]:
            key = f"{p[0]}_{p[1]}_{TG.vid(c)}_{sym}"
            if key in idx or time.time() - t0 > budget_s:
                continue
            if sym not in data:
                try:
                    data[sym] = tal_load(sym)
                except Exception:  # noqa: BLE001
                    data[sym] = None
            if data[sym] is None:
                continue
            t = TG.simulate(data[sym], p, c, gcs.setdefault((sym, p[0]), {}))
            f = TAL_DIR / f"{key}.csv.gz"
            t.to_csv(f, index=False)
            idx[key] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pn, "gap": p[2],
                        "knob": "search", "val": TG.vid(c), "cfg": c, "sym": sym, "file": f.name, "n": len(t),
                        "from": "bot6_ftmo"}
            (idx_dir / f"{key}.json").write_text(json.dumps(idx[key], default=str))
            done += 1
    print(f"tal ftmo (bot 6): {done} run, {len(want) * len(WORLD['symbols'])} wanted")
    return done


def ftmo_pairs(idx, pairs):
    """The pairs for TG.cards with each pair's 'chosen' replaced by the best train total among its versions that run
    with FTMO's break rule on (WKND), on all three indices; a pair with none yet gets no card ('chosen' None) until
    measured. Returns (pairs, {pair: what was chosen before / now})."""
    out, note = {}, {}
    for pn, x in pairs.items():
        rows, seen = [], set()
        for v in idx.values():
            c = TG._norm(dict(v["cfg"]))
            if v["pair"] != pn or not TG.wknd(c) or TG.vid(c) in seen:
                continue
            seen.add(TG.vid(c))
            t = _t6(idx, pn, c)
            if t is not None and len(t):
                rows.append({"knob": "search", "val": TG.vid(c), "cfg": c, **TG._stats(t)})
        ok = [r for r in rows if (r["train"]["eff_n"] or 0) >= K.MIN_EFF_N and r["train"]["mean"] is not None]
        best = max(ok, key=lambda r: r["train"]["mean"] * r["train"]["eff_n"]) if ok else None
        y = dict(x, chosen=best)
        if best and not TG._find(x, best["cfg"]):     # its row among the pair's variants too (neighbours / baseline)
            y["variants"] = list(x.get("variants") or []) + [{k: best[k] for k in ("knob", "val", "train", "unseen", "whole")}]
        out[pn] = y
        before = (x.get("chosen") or {}).get("cfg")
        note[pn] = {"before": before, "before_unseen": ((x.get("chosen") or {}).get("unseen") or {}).get("mean"),
                    "ftmo": best["cfg"] if best else None, "ftmo_train": best["train"] if best else None,
                    "ftmo_unseen": best["unseen"] if best else None, "ftmo_versions": len(rows)}
    return out, note


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
            c0 = TG.ftmo_cfg({k: v for k, v in cfg.items() if k != "TREND"})   # as the live bot runs it (b5to6brk)
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


def _curve(x):
    """The equity curve's smoothness on one period (the user, 16.11.2026 20:28, task mgr1611curve: "many small wins
    every day rather than one big win"): win rate, % green days, the largest fall from the peak (R), the daily spread."""
    if not len(x):
        return {"win": None, "green": None, "dd": None, "sd": None}
    d = x.groupby(x["time"].dt.strftime("%Y-%m-%d"))["r"].sum()
    eq = d.cumsum()
    return {"win": round(float((x["r"] > 0).mean() * 100), 1), "green": round(float((d > 0).mean() * 100), 1),
            "dd": round(float((eq.cummax().clip(lower=0) - eq).max()), 2), "sd": round(float(d.std()), 2) if len(d) > 1 else 0.0}


def _day_stats(t):
    tr, un = t[t["time"] < A.SPLIT], t[t["time"] >= A.SPLIT]
    wd = lambda x: round(float(x.groupby(x["time"].dt.strftime("%Y-%m-%d"))["r"].sum().min()), 2) if len(x) else None  # noqa: E731
    return {"train_w": round(TG.score(t), 1), "unseen": round(float(un["r"].sum()), 1), "n_unseen": int(len(un)),
            "mean_unseen": round(float(un["r"].mean()), 3) if len(un) else None,
            "worst_day_train": wd(tr), "worst_day_unseen": wd(un),
            "train_sum": round(float(tr["r"].sum()), 1), "curve_train": _curve(tr), "curve_unseen": _curve(un)}


def _smoother(x, lv, periods=("train", "unseen")):
    """mgr1611curve: a way that clearly smooths the curve - more green days and a smaller fall from the peak, the
    worst day not worse - and still earns (at least CURVE_KEEP of today's total), on each period given. The total may
    be somewhat lower; the manager decides with the full comparison."""
    for p in periods:
        tot, w = ("train_sum", "worst_day_train") if p == "train" else ("unseen", "worst_day_unseen")
        a, b = (x.get(f"curve_{p}") or {}), (lv.get(f"curve_{p}") or {})
        if None in (a.get("green"), b.get("green"), a.get("dd"), b.get("dd")):
            return False
        if not (a["green"] >= b["green"] + CURVE_GREEN and a["dd"] < b["dd"] and (x.get(w) or -99) >= (lv.get(w) or -99)):
            return False
        if (x.get(tot) or 0) <= 0 or (x.get(tot) or 0) < CURVE_KEEP * (lv.get(tot) or 0):
            return False
    return True


CURVE_GREEN = 2.0     # percentage points more green days, on each period
CURVE_KEEP = 0.8      # at least 80% of today's total, on each period


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
        c0 = TG.ftmo_cfg({k: v for k, v in cfg.items() if k != "TREND"})       # as the live bot runs it (b5to6brk)
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
    # the rules exactly as the live bot runs them (bot 5's task b5to6brk, 14.11.2026 night): live/watch_own.py trades
    # TG.ftmo_cfg(cfg) (FTMO's 2h+ break rule on, WKND 10), so every head-to-head is measured on it - only the rule's
    # own test (the WKND group) keeps the rules without it as its "now". Before this the tests ran without the rule.
    c0 = TG._norm(dict(cfg)) if all(k == "WKND" for k, _ in group) else TG.ftmo_cfg(cfg)
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
# THE USER'S TASKS OF 14.11.2026 (replies.json "_tasks" tal1411ema / tal1411strad / tal1411anti), shared with bot 5 -
# the same tal_gap knobs (bot 5's commit 9de242565), the same in simulate and in live_trades except PYR: on every
# running track of bot 6, one knob moved on its rules as they are, each group answered on its own (not hidden behind
# another group's best): (1) EMA 9/20 direction on the high / the entry timeframe (TREND e920 / e920l - replaces the
# track's TREND), (2) no entry while the Bollinger 20/2 width is under its 20th percentile of 100 candles, on the entry
# / the high timeframe (BBW l20 / h20), (3) the news guard: no entry +-15 / 60 minutes around a red release and an open
# trade closed 15 / 60 minutes before it (NEWSX; the live bot 6 has the calendar since 14.11.2026), (4) the pyramid:
# half a position more at +1R, the whole stop moved so a stop still loses 1R (PYR; the test only - its card is
# tal_gap_code, built by the smart agent after an approval). Queued after the 13.11 tasks inside the user's step.
EMA6 = (("TREND", "e920"), ("TREND", "e920l"))
BBW6 = (("BBW", "l20"), ("BBW", "h20"))
NEWS6 = (("NEWSX", 15), ("NEWSX", 60))
PYR6 = (("PYR", 1.0),)
# FTMO'S BREAK RULE (the user, 14.11.2026, tal1411wkend / bot 5's b5to6wk1411): no waiting order over a market break of
# 2+ hours. Bot 6's trades were already out at 16:05 New York and the futures' daily break (16:05-18:00) is under 2 hours,
# but a gap order waited over the weekend. The live bot 6 runs its tracks with tal_gap.ftmo_cfg (WKND 10) since
# 14.11.2026 night; this measures, per track, what the rule costs against the rules as tested (first in line).
WKND6 = (("WKND", 10),)
# THE VIDEO FILTERS OF 14.11.2026 (bot 5's b5to6vid1411, tal_gap.VIDEO_TREND): RSI 50 side, RSI in the trend against
# EMA 200, ADX14 >= 20, SuperTrend 10/3, EMA 50 / 200, the order of 5 EMAs - each one as the track's TREND, head to head
VID6 = tuple(("TREND", v) for v in TG.VIDEO_TREND)
# PREMIUM / DISCOUNT AND THE DAILY PIVOT (the user's videos of 14.11.2026, task tal1411set6 (7) / (9); bot 5's commit
# 912f549ba, tal_gap.Data.trend - closed candles only, the same in simulate and in live_trades, which bot 6's account
# trades): longs only under the middle of the last 20 / 50 high-timeframe candles' range and shorts only over it
# (pd20 / pd50), or longs only above the previous day's pivot (H+L+C)/3 and shorts only below it (piv) - as the track's
# TREND, head to head on bot 6's four running tracks and three indices.
PDP6 = (("TREND", "pd20"), ("TREND", "pd50"), ("TREND", "piv"))
# CLOSE TO THE GAP METHOD (the user's videos of 14.11.2026 12:48-14:26, task tal1411gapplus; bot 5's commit 17b23b710,
# tal_gap.gap_keep / SESS_NY - the high timeframe's closed candles only, the same in simulate and in live_trades, which
# bot 6's account trades): G3 = the gap's quality by candle 3 (body / wicks against the ATR), SWP = only a gap whose
# impulse swept the previous day's / the 20 candles' low or high, SESS sb / sb10 = fills only in the Silver Bullet
# windows (New York 03-04, 10-11, 14-15 / 10-11 - inside the futures' session), and TREND dbias = the Daily Bias of the
# last closed day (bot 6's smart agent, 14.11.2026 afternoon, tal_gap.Data.trend). One knob each, head to head on the
# four running tracks and three indices; a card only if better on 2026 too.
GQ6 = tuple(("G3", v) for v in TG.G3_VALS) + tuple(("SWP", v) for v in TG.SWP_VALS) + (("SESS", "sb"), ("SESS", "sb10"))
DB6 = (("TREND", "dbias"),)
# The sweep's stricter forms (bot 5's task for bot 6, b5to6swp2, 14.11.2026 evening; tal_gap.SWP2_VALS, the user's videos
# of 14:23-14:26): sw20r / pdhlr = the 20 candles' / yesterday's low (high) swept and the sweeping candle closed back
# inside (a rejection), eql = equal lows (highs) swept. One knob each, head to head on the four running tracks and the
# three indices, like GQ6; a card only if better on 2026 too, naming bot 5's result for the same value.
SWP26 = tuple(("SWP", v) for v in TG.SWP2_VALS)
# FIBONACCI EMAS AND HEIKIN ASHI (the user's videos of 14.11.2026 17:37-18:03, task tal1411set7 (6) / (13); bot 5's commit
# 80d94c8ac, tal_gap.Data.trend - closed candles only, the same in simulate and in live_trades, which bot 6's account
# trades): fib5 = EMA 8/13/21/34/55 of 15 minutes all in order = only that side, else both; ha = the high timeframe's
# last Heikin Ashi colour; ha2 = two of the same colour, the last without a wick against it, else no entry. As the
# track's TREND, head to head on the four running tracks and three indices; a card only if better on 2026 too.
SET76 = (("TREND", "fib5"), ("TREND", "ha"), ("TREND", "ha2"))
# LAZYBEAR'S SQUEEZE MOMENTUM AND YESTERDAY'S HIGH / LOW BREAKOUT (bot 5's task for bot 6, b5to6sqz, 14.11.2026 evening;
# the user's videos tal1411set7 (12) / (16), bot 5's e90e104ec, tal_gap.Data.trend - closed candles only, the same in
# simulate and in live_trades): sqzm = the momentum's sign on the high timeframe, sqz = the same and no entry while the
# squeeze is on, pdbrk = over yesterday's high longs only / under its low shorts only. Head to head on the four running
# tracks and three indices, like SET76; a card only if better on 2026 too, naming bot 5's result.
SQZ6 = (("TREND", "sqzm"), ("TREND", "sqz"), ("TREND", "pdbrk"))
# THE FIX FOR THE BREAK RULE'S COST (bot 5's task b5to6brk, 14.11.2026 night). On bot 6 the rule (WKND 10) took 2026 from
# +45.6R to +20.1R on 1h/5m and from +31.1R to +2.3R on 4h/1h: with EXPIRY 120 hours a gap order placed on Monday-Thursday
# still waited on Friday and was cancelled for good. Head to head on the rules as the live bot runs them (WKND 10):
# BRKH 1 / 3 / 6 (bot 5's fix - no order from X hours before the break) and REARM (bot 6's - the cancelled order is
# placed again at the reopening while the gap is within EXPIRY; no order waits over the break, so the rule is kept).
# First in the user's step. tal_gap.order_ok / rearm, the same in simulate and live_trades.
BRK6 = (("REARM", True), ("BRKH", 1), ("BRKH", 3), ("BRKH", 6))
FILT_HE["REARM"] = lambda v: "הוראת גאפ שבוטלה לפני סוף שבוע / חג חוזרת בפתיחה, כל עוד הגאפ בתוקף (REARM)"
FILT_HE["BRKH"] = lambda v: f"בלי הוראת גאפ חדשה {v:g} שעות לפני הפסקה של שעתיים ומעלה (BRKH {v:g})"
# PARTIAL EXITS IN EVERY MARKET (the user, 16.11.2026 17:18, task tal1611partial: "most of today's losses were in profit at
# some point - take them out partially, as I keep telling you"; bot 5's tal1611gaveback, the same values on its tracks).
# Head to head on bot 6's four running tracks and three indices, in place of the track's own break-even (EXIT): half /
# a third out at 50% / 61.8% of the way with the rest's stop to the entry, the stop to the entry at 50%, and in R of the
# trade's risk (tal_gap.exit_mode "...r", the same in simulate and live_trades): the stop to the entry / half out at
# +1R / +2R. Bot 6's live losses of 13-16.11 reached +0.58R / +0.64R / +0.14R before the stop (none +1R, none half the
# way). p50_50 alone was tested on 13.11 (ask_p50_50: worse on 2024-2025 on all four tracks). First in the user's step.
GAVE6 = tuple(("EXIT", v) for v in ("p50_618", "p33_618", "be50", "be1r", "p50_1r", "p30_1r", "be2r", "p50_2r"))
FILT_HE["EXIT"] = (lambda f0: lambda v: {
    "p50_618": "חצי החוצה ב־61.8% מהדרך ליעד וסטופ לכניסה (p50_618)",
    "p33_618": "שליש החוצה ב־61.8% מהדרך ליעד וסטופ לכניסה (p33_618)",
    "be50": "סטופ לכניסה בחצי הדרך ליעד (be50)",
    "be1r": "סטופ לכניסה ב־+1R (be1r)", "be2r": "סטופ לכניסה ב־+2R (be2r)",
    "p50_1r": "חצי החוצה ב־+1R וסטופ לכניסה (p50_1r)", "p30_1r": "30% החוצה ב־+1R וסטופ לכניסה (p30_1r)",
    "p50_2r": "חצי החוצה ב־+2R וסטופ לכניסה (p50_2r)"}.get(v) or f0(v))(FILT_HE["EXIT"])
# BOT 5'S FOLLOW-THROUGH RULE ON BOT 6'S TRACKS (the user, 16.11.2026, task b5follow1611 - "strong candles in the
# trade's direction right after the entry = more confidence"; bot 5's agent added tal_gap FOLLOW xN, 61e63edf8, and
# tests it on its 6 index tracks): at the close of the N-th low-timeframe candle after the fill, a trade closing against
# its entry leaves there. Shared methods (bots 5 and 6): the same values head to head on bot 6's four running tracks and
# three indices. FOLLOW is in simulate only, not in live_trades yet - so a winning card is a code card (the smart agent
# builds it into the live bot after an approval), like PYR.
FOL6 = (("FOLLOW", "x1"), ("FOLLOW", "x2"), ("FOLLOW", "x3"))
FILT_HE["FOLLOW"] = lambda v: f"בלי המשך אחרי {str(v)[1:]} נרות של הכניסה (סגירה נגד הכניסה) - יציאה בסגירה (FOLLOW {v})"
CODE6 = ("PYR", "FOLLOW")     # knobs live_trades does not run yet: an approved card is built by the smart agent
TASK6 = {"gaveback": GAVE6, "follow": FOL6, "brk": BRK6, "wknd": WKND6, "ema920": EMA6, "bbw": BBW6, "newsx": NEWS6, "pyr": PYR6, "video": VID6, "pdpiv": PDP6,
         "gapq": GQ6, "dbias": DB6, "swp2": SWP26, "set7": SET76, "set7b": SQZ6}
USER6 = GAVE6 + FOL6 + BRK6 + WKND6 + USER6 + EMA6 + BBW6 + NEWS6 + PYR6 + VID6 + PDP6 + GQ6 + DB6 + SWP26 + SET76 + SQZ6
TASK_TREND = set(TG.VIDEO_TREND) | {"pd20", "pd50", "piv", "dbias", "fib5", "ha", "ha2", "sqzm", "sqz", "pdbrk"}
FILT_HE["G3"] = lambda v: {"weak": "רק גאפ שנר 3 שלו חלש (גוף ופתילים קטנים מחצי ATR)",
                           "nostrong": "בלי גאפ שנר 3 שלו חזק (גוף מעל ATR)",
                           "nowick": "בלי גאפ שלנר 3 שלו פתיל גדול",
                           "nobad": "בלי גאפ שנר 3 שלו חזק או עם פתיל גדול"}.get(str(v), f"G3 {v}") + f" (G3 {v})"
FILT_HE["SWP"] = lambda v: {"pdhl": "רק גאפ אחרי גריפה של השפל / השיא של אתמול",
                            "sw20": "רק גאפ אחרי גריפה של השפל / השיא של 20 הנרות",
                            "sw20r": "רק גאפ אחרי גריפה של 20 הנרות שהנר שלה נסגר בחזרה בפנים (דחייה)",
                            "pdhlr": "רק גאפ אחרי גריפה של אתמול שהנר שלה נסגר בחזרה בפנים (דחייה)",
                            "eql": "רק גאפ אחרי גריפה של שפלים / שיאים שווים"}.get(str(v), f"SWP {v}") + f" (SWP {v})"
_sess_he = FILT_HE["SESS"]
FILT_HE["SESS"] = lambda v: ({"sb": "רק בחלונות Silver Bullet (ניו יורק 03-04, 10-11, 14-15)",
                              "sb10": "רק בחלון Silver Bullet 10:00-11:00 ניו יורק"}[v] + f" (SESS {v})") \
    if v in ("sb", "sb10") else _sess_he(v)
FILT_HE["TREND"] = lambda v: {"e920": "רק בכיוון EMA 9/20 של הטיים פריים הגבוה", "e920l": "רק בכיוון EMA 9/20 של טיים פריים הכניסה",
                              "day": "רק בכיוון פתיחת היום", "up20": "רק בכיוון EMA 20 של הטיים פריים שמעל",
                              "rsi50": "רק בצד של RSI 50", "rsix": "RSI במגמה מול EMA 200", "adx": "רק כש־ADX14 מעל 20",
                              "st": "רק בכיוון סופרטרנד 10/3", "e50200": "רק בכיוון EMA 50 מול 200",
                              "ema5": "רק כש־5 הממוצעים מסודרים בכיוון",
                              "pd20": "קנייה רק בדיסקאונט / מכירה רק בפרמיה של 20 הנרות בטיים פריים הגבוה",
                              "pd50": "קנייה רק בדיסקאונט / מכירה רק בפרמיה של 50 הנרות בטיים פריים הגבוה",
                              "piv": "רק בכיוון הפיבוט היומי (מעל P קנייה, מתחת מכירה)",
                              "dbias": "רק בכיוון ההטיה היומית (Daily Bias של היום הקודם)",
                              "fib5": "רק כשממוצעי פיבונאצ'י 8/13/21/34/55 (15 דק') מסודרים בכיוון",
                              "ha": "רק בכיוון נר הייקין אשי האחרון של הטיים פריים הגבוה",
                              "ha2": "רק אחרי שני נרות הייקין אשי באותו צבע, האחרון בלי פתיל נגד",
                              "sqzm": "רק בכיוון מומנטום הסקוויז של LazyBear בטיים פריים הגבוה",
                              "sqz": "רק בכיוון מומנטום הסקוויז, ובלי כניסה בזמן סקוויז",
                              "pdbrk": "מעל השיא של אתמול רק קניות, מתחת לשפל רק מכירות"}.get(str(v), f"TREND {v}") + f" (TREND {v})"
FILT_HE["BBW"] = lambda v: ("בלי כניסה כשרוחב בולינגר 20/2 מתחת לאחוזון 20 של 100 הנרות, טיים פריים "
                            + ("הכניסה" if str(v)[0] == "l" else "גבוה") + f" (BBW {v})")
FILT_HE["NEWSX"] = lambda v: f"הגנת חדשות: בלי כניסה ±{int(v)} דק' סביב חדשה אדומה, וסגירה {int(v)} דק' לפניה (NEWSX {int(v)})"
FILT_HE["WKND"] = lambda v: f"כלל FTMO: ביטול הוראה ממתינה וסגירה {int(v)} דק' לפני הפסקה של שעתיים ומעלה (WKND {int(v)})"
FILT_HE["PYR"] = lambda v: f"פירמידה: חצי פוזיציה נוספת ב־{v:g}R, הסטופ של הכל זז כך שהסיכון נשאר 1R (PYR {v:g})"


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
            want += [(pn, c0), (pn, TG._norm(dict(cfg)))] + [(pn, c) for _, _, c in vs]   # + the rules without WKND
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
        if len(row) < len(vs) + 1 or row["live"]["train_w"] <= -1e8:
            continue                                     # not every variant ran yet / too few
        lv, curve = row["live"], False
        if best:
            name, k, v, c, t = best
            x = row[name]
        if not best or not ((x["mean_unseen"] or 0) > 0 and x["unseen"] > lv["unseen"]
                            and (x["mean_unseen"] or -9) >= (lv["mean_unseen"] or -9)
                            and (x["worst_day_unseen"] or -99) >= (lv["worst_day_unseen"] or -99)):
            # mgr1611curve (the user, 16.11.2026 20:28): no card on the total - a way that clearly smooths the curve and
            # still earns comes as a card too. CHOSEN on 2024-2025 only (smoother there; the most green days among
            # them), then it must be smoother on 2026 as well.
            sm = [q for q in row if q != "live" and _smoother(row[q], lv, ("train",))]
            pick = max(sm, key=lambda q: row[q]["curve_train"]["green"], default=None)
            if pick is None or not _smoother(row[pick], lv, ("unseen",)):
                continue
            k, v, c = next((k_, v_, c_) for k_, v_, c_ in vs if f"{k_}={v_}" == pick)
            t, name, x, curve = _t6(idx, pn, c), pick, row[pick], True
            if t is None:
                continue
        st, bs = TG._stats(t), TG._stats(base)
        hi, lo = pn.split("/")
        r_ = float(rk.get(trk, FA.RISK_TAL))
        better = sum(1 for q in row if q != "live" and row[q]["train_w"] > lv["train_w"])
        cv = lambda p_: (lambda a, b: f"ימים ירוקים {b.get('green')}% ← {a.get('green')}%, עסקאות מנצחות {b.get('win')}% ← {a.get('win')}%, "  # noqa: E731
                                     f"ירידה מהשיא {b.get('dd')}R ← {a.get('dd')}R, פיזור יומי {b.get('sd')}R ← {a.get('sd')}R")(
            x.get(f"curve_{p_}") or {}, lv.get(f"curve_{p_}") or {})
        out.append(K.card("bot6", "improve", ("עקומה חלקה יותר · " if curve else "") + f"שיטת הגאפ · {TG.TF_HE[hi]} / {TG.TF_HE[lo]}: {FILT_HE[k](v)}",
                          (f"עקומת הון חלקה יותר בשתי התקופות (בקשה של טל מ־16.11: הרבה ניצחונות קטנים), גם אם הסכום קצת נמוך יותר - "
                           f"המנהל מחליט. עד 2025: {cv('train')}, סה\"כ {lv.get('train_sum')}R ← {x.get('train_sum')}R. "
                           f"2026: {cv('unseen')}. " if curve else "")
                          + f"המסלול {trk} עם הכללים שרצים, ועוד: {FILT_HE[k](v)}. נבחר על 2024-2025 בלבד מתוך {len(vs)} מסננים "
                          f"({better} מהם טובים מהיום ב־2024-2025; סה\"כ משוקלל {lv['train_w']:+.1f}R ← {x['train_w']:+.1f}R). "
                          f"2026 (לא נראתה): סה\"כ {lv['unseen']:+.1f}R ← {x['unseen']:+.1f}R, {x['n_unseen']} עסקאות, "
                          f"{x['mean_unseen']:+.2f}R לעסקה (היום {lv['mean_unseen']:+.2f}R על {lv['n_unseen']}), היום הגרוע "
                          f"{lv['worst_day_unseen']}R ← {x['worst_day_unseen']}R. בסיכון {r_ * 100:g}%. "
                          + (f"משימה של טל מ־14.11. " if k in ("BBW", "NEWSX", "PYR", "WKND", "G3", "SWP", "BRKH", "REARM") or str(v).startswith("e920") or v in TASK_TREND or v in ("sb", "sb10") else "")
                          + ("משימה של טל מ־16.11 (הפסדים שהיו ברווח - יציאה חלקית). " if (k, v) in GAVE6 else "")
                          + ("משימה של טל מ־16.11 (נרות חזקים אחרי הכניסה - b5follow1611). " if k == "FOLLOW" else "")
                          + "בבוט 5: " + ("עוד לא נבדק - שיטה של בוט 6, נשלחה לבוט 5 כמשימה (b6to5rearm)." if k == "REARM" else
                                          _b5_gave(v) if (k, v) in GAVE6 else
                                          _b5_gave(v, ("_follow",)) if k == "FOLLOW" else
                                          _b5_ladder(k, v) if k in ("LADDER", "BBW", "NEWSX", "PYR", "MINSTOP", "WKND", "G3", "SWP", "BRKH") or str(v).startswith("e920") or v in TASK_TREND or v in ("sb", "sb10")
                                          else f"{b5.get(pn, 'לזוג הזה אין הגדרת קבוצה עם מסננים')}."),
                          st["train"], st["unseen"], whole=st["whole"], halves_=st["halves"], cost_r=st["cost_r"],
                          tested=len(vs), risk_pct=round(r_ * 100, 3), baseline=bs["unseen"],
                          neighbours={"values": [q for q in row if q != "live"], "share_positive": round(float(np.mean(
                              [row[q]["unseen"] > lv["unseen"] for q in row if q != "live"])), 2)},
                          detail={"world": "bot6", "filters": row, "bot5": b5.get(pn)},
                          # PYR is not in live_trades yet: a code card (the smart agent builds it after an approval)
                          change=({"method": "tal_gap_code", "pair": pn, "cfg": c, "track": trk, "knob": k, "to": v,
                                   **({"test": "curve"} if curve else {})}
                                  if k in CODE6 else
                                  # "test" (14.11.2026): the head-to-head card must not share its id with the search's
                                  # card of the same pair and rules - 38531db8b5eb (t5_1 + MAXC 0.2, the fix of the 13.11
                                  # loss) was withdrawn as its own duplicate and an id once decided is never decided again
                                  {"method": "tal_gap", "pair": pn, "cfg": c, "track": f"tal_gap|{pn}", "knob": "cfg",
                                   "test": "curve" if curve else "head2head"})))
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
        c0 = TG.ftmo_cfg(cfg)                            # as the live bot runs it (b5to6brk)
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


def _b5_gave(val, ends=("_gaveback", "_gavebackr")):
    """What bot 5's agent found for the same partial exit on its tracks (tal_gap_results.json bot5.notes, its
    tal1611gaveback tests: <track>_gaveback / <track>_gavebackr; FOLLOW: <track>_follow, b5follow1611)."""
    try:
        n = (json.loads(TG.RESULTS.read_text(encoding="utf-8")).get("bot5") or {}).get("notes") or {}
    except Exception:  # noqa: BLE001
        n = {}
    got = []
    for k, r in n.items():
        if not k.endswith(tuple(ends)) or not isinstance(r, dict):
            continue
        if r.get("state") != "done":
            got.append(f"{r.get('track', k.split('_')[0])} עוד רץ")
            continue
        x, lv = r.get(str(val)) or {}, (r.get("now") or {})
        if x.get("unseen") and lv.get("unseen"):
            got.append(f"{r['track']}: 2026 {lv['unseen']['total']:+.1f}R ← {x['unseen']['total']:+.1f}R"
                       + (" (נבחר)" if r.get("chosen") == val and r.get("better") else ""))
    return ("; ".join(got[:6]) if got else "אותה בדיקה רצה על המסלולים של בוט 5 (" + ("b5follow1611" if "_follow" in ends else "tal1611gaveback") + ")") + "."


def _b5_ladder(knob="LADDER", val=None):
    """What bot 5's agent found for the same knob (its improve cards: LADDER 13.11.2026; BBW / NEWSX / PYR / TREND
    e920 / MINSTOP 14.11.2026)."""
    pat = (f'"{val}"' if val in TASK_TREND else '"e920') if knob == "TREND" else \
        (f'"{val}"' if knob == "SESS" else f'"{knob}"')
    try:
        cs = [c for nm in ("cards.json", "cards_notes.json", "cards_smart.json") if Path(f"reports/agents/bot5/{nm}").exists()
              for c in json.loads(Path(f"reports/agents/bot5/{nm}").read_text(encoding="utf-8")).get("cards", [])
              if pat in json.dumps(c.get("change") or {}) and c.get("kind") == "improve"]
    except Exception:  # noqa: BLE001
        cs = []
    if not cs:
        return ("הכניסה המדורגת" if knob == "LADDER" else f"אותו שינוי ({knob})") + " נבדק על 6 המסלולים שלו, עוד אין כרטיס שיפור."
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
    # whole micro contracts as live/fut_account.size takes them (14.11.2026): at the small risk the account's limits
    # allow (x 0.2-0.4 of 0.4%), one MNQ / MES / MYM is often more than a trade may risk - live skips such a trade
    return {"day": FA.DAILY_LOSS / FA.INITIAL * 100, "all": FA.MAX_LOSS / FA.INITIAL * 100, "trail": True,
            "use": FA.ROOM_USE, "day_hard": True, "dayf": FA.tday,
            "contracts": {"syms": {s: (pv, tick) for s, (_, pv, tick) in FA.SPEC.items()}, "comm": FA.COMM,
                          "slip": FA.SLIP_TICKS, "initial": FA.INITIAL, "max_n": FA.MAX_MICROS}}


# THE FIXED RISK RULE (the user, 14.11.2026 22:16, task tal1411risk: "a fixed rule that never changes - the daily and the
# overall risk; if such a risk is opened, it is not right"). The live account (live/fut_account.size) counts every open
# trade at its stop and gives a new trade only the room left; the whole-bot test (TG.account with fut_lim) does the same.
# What the gap tracks' test did not have: the scout strategies bot 6 also trades (live/bots/bot6.json "scout", their own
# risk) - so the whole bot here = the gap tracks (as the live bot runs them) + every scout strategy with its live
# variant, and the card reports how many trades the room rule cut / skipped in each period (a lot = fix the overlap).
GUARD_KEEP_DAYS = 7
GUARD_DIR = Path("reports/agents/bot6/scout_fut/guard")


def _guard_file(combo, v):
    """The kept trades of a scout strategy in its live variant (guard6), if fresh enough."""
    from agents import scout_fut as SF
    try:
        f = GUARD_DIR / f"{combo.replace('|', '_')}_{SF.vid(v)}.csv.gz"
    except Exception:  # noqa: BLE001
        return None
    if f.exists() and time.time() - f.stat().st_mtime < GUARD_KEEP_DAYS * 86400:
        return f
    return None


def guard6(idx, counted, rk, budget_s):
    from agents import scout_fut as SF
    t0, rows = time.time(), []
    for trk, (pn, c) in counted.items():
        c = TG._norm(dict(c))
        fs = list({v["sym"]: v for v in idx.values() if v["pair"] == pn and TG._norm(dict(v["cfg"])) == c
                   and v["sym"] in WORLD["symbols"]}.values())
        if {v["sym"] for v in fs} != set(WORLD["symbols"]):
            return None
        for v in fs:
            t = pd.read_csv(TAL_DIR / v["file"])
            t["trk"], t["sym"] = trk, v["sym"]
            rows.append(t[["time", "dir", "r", "cost_r", "mins", "trk", "sym"]])
    gap = pd.concat(rows, ignore_index=True)
    gap["time"] = pd.to_datetime(gap["time"], utc=True, format="ISO8601")
    gap["out"] = gap["time"] + pd.to_timedelta(gap["mins"].fillna(0).astype(float) + 1, unit="min")
    try:
        sc = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("scout") or {}
    except Exception:  # noqa: BLE001
        sc = {}
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data, srows, missing = {}, [], []
    # each scout's trades in its LIVE variant are kept on disk between rounds (bot 6's smart agent, 16.11.2026): the
    # 15.11 23:06 round reached 1 of the 10 scouts in its 12% of the time (one symbol's minutes + one strategy), so
    # the whole-bot test and the scouts' factor (scout_scale6) never saw them. A kept file is used for GUARD_KEEP_DAYS,
    # the ones not kept yet are run first, and each round adds what its time allows until all ten are in.
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    order = sorted(sc.items(), key=lambda kv: _guard_file(kv[0], (kv[1] or {}).get("v") or {}) is not None)
    for combo, e in order:
        try:
            sid, tf, sym = combo.split("|")
            v = (e or {}).get("v") or {}
            if sym not in FA.SPEC:
                missing.append(combo)
                continue
            kept = _guard_file(combo, v)
            if kept is not None:
                t = pd.read_csv(kept)
            else:
                if time.time() - t0 > budget_s:
                    missing.append(combo)
                    continue
                if sym not in data:
                    pr = prep(sym, end)
                    data[sym] = None if pr is None else pr
                if data[sym] is None:
                    missing.append(combo)
                    continue
                df, _, _, noent = data[sym]
                shut, _ = FA.session_masks(df)
                _, pv, tick = FA.SPEC[sym]
                t = SF.trades(df, sym, sid, tf, v, noent, shut, tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv)
                t.to_csv(GUARD_DIR / f"{combo.replace('|', '_')}_{SF.vid(v)}.csv.gz", index=False)
            if t.empty:
                continue
            k = f"scout:{combo}" if f"scout:{combo}" in rk else "scout"
            x = pd.DataFrame({"time": pd.to_datetime(t["time"], utc=True), "dir": t["dir"].astype(int), "r": t["r"].astype(float),
                              "cost_r": t["cost_r"].astype(float), "trk": k, "sym": sym})
            x["out"] = pd.to_datetime(t["exit_time"], utc=True) + pd.Timedelta(minutes=1)
            x["mins"] = (x["out"] - x["time"]).dt.total_seconds() / 60
            srows.append(x)
        except Exception as ex:  # noqa: BLE001 - one strategy must not stop the test
            print(f"guard6 {combo}: {ex}")
            missing.append(combo)
    allt = pd.concat([gap] + srows, ignore_index=True)
    risk = {k: float(rk.get(k, FA.RISK_TAL)) * 100 for k in set(allt["trk"])}
    lim = fut_lim()
    res = {"tracks": sorted(counted), "scouts": sorted(sc), "missing": missing, "risk_pct": risk}
    for nm, t in (("gap_only", gap), ("whole", allt)):
        res[nm] = {per: TG.account(t[(t["time"] < TG.SPLIT) == (per == "train")].copy(), risk, lim) for per in ("train", "unseen")}
        for per in ("train", "unseen"):
            a = res[nm][per] or {}
            a["n"] = int(((t["time"] < TG.SPLIT) == (per == "train")).sum())
            a["room_only_skipped"] = int((a.get("room_skipped") or 0) - (a.get("contract_skipped") or 0))
    try:            # one factor on the scouts' risk, chosen on 2024-2025 (bot 6's smart agent, 15.11.2026 night)
        res["scout_scale"] = scout_scale6(allt, risk, lim)
    except Exception as ex:  # noqa: BLE001
        print(f"guard6 scout scale failed: {ex}")
    return res


# THE SCOUTS' RISK ON THE WHOLE BOT (bot 6's smart agent, 15.11.2026 night; principle 11). The gap tracks run at
# x 0.3 of the bot's 0.4% since 14.11 (scale_cards: the largest factor with 2024-2025 inside 80% of FTMO Futures'
# limits) - but the ten scout strategies were added on top at 0.28-0.4% each, never in that test. The PC round of
# 15.11 14:36 (risk_guard, 6 of the 10 scouts): gap tracks alone 2026 +11.4%, worst day -0.43%, deepest fall 1.74%;
# with the scouts 2026 +30.7% but a fall of 11.7% and 4 days stopped at the daily limit; on 2024-2025 the whole bot
# ended at -4.46% - its lowest point, at the trailing limit - and skipped 2,873 of 2,955 trades for lack of room
# (the gap tracks alone: +20.9%, fall 2.8%). Bot 5's agent found the same that night (fall 9.5% -> 18% after 8 new
# scouts, x 0.45 on them). The rule, shared with the gap tracks (tal_gap.risk_scale): the gap tracks keep their
# approved risk, the scouts get ONE factor - the largest in steps of 0.05 with which 2024-2025 of the whole bot stays
# inside MARGIN of the limits and the account never fails; then shown on 2026. A cut to fit the account, not a cut of
# a losing strategy (account_fit) - so "fix first" does not hold it.
def scout_scale6(allt, risk, lim):
    sk = [k for k in risk if str(k).startswith("scout")]
    tr, un = allt[allt["time"] < TG.SPLIT], allt[allt["time"] >= TG.SPLIT]
    if not sk or tr.empty:
        return None
    L = TG._lim(lim)

    def fits(b):
        return bool(b) and b["max_fall_pct"] <= TG.MARGIN * L["all"] + 1e-9 \
            and b["worst_day_pct"] >= -TG.MARGIN * L["day"] - 1e-9 and not b["stopped"]

    def rk(k):
        return {x: (v * k if x in sk else v) for x, v in risk.items()}
    tried, k, b = [], 1.0, None
    while True:
        b = TG.account(tr.copy(), rk(k), lim)
        tried.append({"k": k, "fall": (b or {}).get("max_fall_pct"), "day": (b or {}).get("worst_day_pct"),
                      "total": (b or {}).get("total_pct"), "stopped": (b or {}).get("stopped")})
        if fits(b) or k <= TG.SCALE_FLOOR + 1e-9:
            break
        k = max(TG.SCALE_FLOOR, round(k - 0.05, 2))
    gap = allt[~allt["trk"].astype(str).str.startswith("scout")]
    gtr, gun = gap[gap["time"] < TG.SPLIT], gap[gap["time"] >= TG.SPLIT]
    try:
        edge = scout_edge6(tr, un, risk, sk, lim, b, TG.account(un.copy(), rk(k), lim), fits)
    except Exception as ex:  # noqa: BLE001 - the uniform factor stands without it
        print(f"scout edge split failed: {ex}")
        edge = None
    return {"chosen": round(float(k), 2), "fits": fits(b), "margin": TG.MARGIN, "edge": edge,
            "limits": {"day": L["day"], "all": L["all"], "trail": L["trail"], "use": L["use"]},
            "scouts": sorted(set(allt.loc[allt["trk"].astype(str).str.startswith("scout"), "trk"].astype(str))),
            "now": {"train": TG.account(tr.copy(), risk, lim), "unseen": TG.account(un.copy(), risk, lim)},
            "with": {"train": b, "unseen": TG.account(un.copy(), rk(k), lim)},
            "gap_only": {"train": TG.account(gtr.copy(), risk, lim), "unseen": TG.account(gun.copy(), risk, lim)},
            "tried": tried}


# THE SCOUTS' RISK BY EACH ONE'S EDGE (bot 6's smart agent, 16.11.2026 night; principles 2, 8, 11 and case 41's "next
# stage": the risk shared among the strategies by their edge). The uniform factor (x 0.15 on 16.11) gives a weak scout
# (SC072: +0.06R a trade to 2025, t 0.16) the same share as a strong one, and at 0.042-0.06% one micro contract is more
# than most scout trades may risk: 1,597 of the 2024-2025 trades and 1,114 of 2026's were skipped for that alone - the
# scouts that still trade are the ones with the tightest stops, where the costs weigh most. The variant: each scout's
# risk = its approved risk x k x w, w = its own time-weighted R a trade on 2024-2025 / the best scout's (continuous,
# never under EDGE_FLOOR - no edge -> the minimum, so the live data keeps coming - and never over the bot's 0.4%);
# k = the largest (from the strongest at 0.4%, down by 10%) with which 2024-2025 of the whole bot is NO riskier than with the uniform factor (its
# deepest fall and worst day) and inside the account's margin. Chosen only on 2024-2025; a card only when it earns more
# on 2024-2025 AND on 2026, with 2026 still inside the margin. Bot 5 runs one uniform factor (x 0.45) - not tested there.
EDGE_FLOOR = 0.1
EDGE_MIN_N = 20


def scout_edge6(tr, un, risk, sk, lim, b_uni, u_uni, fits):
    if not b_uni or not u_uni or tr.empty:
        return None
    e = {}
    for x in sk:
        t_ = tr[tr["trk"] == x]
        e[x] = None if len(t_) < EDGE_MIN_N else float(np.average(t_["r"].astype(float), weights=K.weights(t_["time"], now=TG.SPLIT)))
    best = max([v for v in e.values() if v is not None and v > 0], default=None)
    if best is None:
        return None
    w = {x: (EDGE_FLOOR if v is None or v <= 0 else max(EDGE_FLOOR, v / best)) for x, v in e.items()}
    cap = 0.4                     # % - the bot's own risk a trade (principle: no trade risks more than the bot's risk)

    def rk(k):
        return {x: (min(cap, v * k * w[x]) if x in sk else v) for x, v in risk.items()}

    def same_risk(bb):
        return fits(bb) and bb["max_fall_pct"] <= b_uni["max_fall_pct"] + 1e-9 and bb["worst_day_pct"] >= b_uni["worst_day_pct"] - 1e-9
    # the scouts' risk in bot6.json is already the uniform factor's (x 0.15 since 16.11), so k may go above 1: from the
    # k that puts the strongest scout at the bot's own 0.4%, down in steps of 10%
    top = max((risk[x] * w[x] for x in sk), default=0.0)
    if top <= 0:
        return None
    k, bt, tried = cap / top, None, []
    while k >= 0.02:
        bt = TG.account(tr.copy(), rk(k), lim)
        tried.append({"k": round(k, 3), "fall": (bt or {}).get("max_fall_pct"), "day": (bt or {}).get("worst_day_pct"),
                      "total": (bt or {}).get("total_pct")})
        if bt and same_risk(bt):
            break
        k, bt = k * 0.9, None
    if bt is None:
        return {"better": False, "why": "no k as safe as the uniform factor", "edge_r": e, "w": w, "tried": tried}
    bu = TG.account(un.copy(), rk(k), lim)
    L = TG._lim(lim)
    better = bool(bu and bt["total_pct"] > b_uni["total_pct"] + 1e-9 and bu["total_pct"] > u_uni["total_pct"] + 1e-9
                  and fits(bu))
    return {"k": round(k, 3), "edge_r": {x: (None if v is None else round(v, 4)) for x, v in e.items()},
            "w": {x: round(v, 3) for x, v in w.items()}, "risk": {x: round(v, 5) for x, v in rk(k).items() if x in sk},
            "train": bt, "unseen": bu, "uniform": {"train": b_uni, "unseen": u_uni}, "better_train": bool(bt["total_pct"] > b_uni["total_pct"] + 1e-9),
            "better": better, "limits": {"day": L["day"], "all": L["all"]}, "tried": tried}


def scout_scale_cards(g, rk):
    """One risk card per scout strategy of bot 6 (live/bots/bot6.json "scout") with the factor of scout_scale6 - also
    the scouts the test could not run in time (the same factor: never more risk than the tested ones)."""
    s = (g or {}).get("scout_scale") or {}
    k = s.get("chosen", 1.0)
    if s and (s.get("edge") or {}).get("better") and k >= 1.0:
        return scout_edge_cards(s, rk)
    if not s or k >= 1.0:
        return []
    gt0 = (s.get("gap_only") or {}).get("train") or {}
    lm0 = s.get("limits") or {}
    if not gt0 or gt0.get("stopped") or gt0.get("max_fall_pct", 0) > s.get("margin", 0.8) * lm0.get("all", 4.5) + 1e-9 \
            or gt0.get("worst_day_pct", 0) < -s.get("margin", 0.8) * lm0.get("day", 1.5) - 1e-9:
        return []           # the gap tracks alone do not fit - not the scouts' factor to fix (scale_cards is)
    lm = s.get("limits") or {}
    nt, nu = (s["now"].get("train") or {}), (s["now"].get("unseen") or {})
    wt, wu = (s["with"].get("train") or {}), (s["with"].get("unseen") or {})
    gt, gu = (s["gap_only"].get("train") or {}), (s["gap_only"].get("unseen") or {})
    line = lambda x: (f"סה\"כ {x.get('total_pct', 0):+.1f}%, היום הגרוע {x.get('worst_day_pct', 0):+.2f}%, "  # noqa: E731
                      f"הירידה הגדולה {x.get('max_fall_pct', 0):.1f}%, ימים שנעצרו במגבלה היומית {x.get('daily_stops', 0)}, "
                      f"עסקאות שדולגו בגלל המקום {max(0, (x.get('room_skipped') or 0) - (x.get('contract_skipped') or 0))}"
                      + (f" (ועוד {x['contract_skipped']} כי חוזה מיקרו אחד גדול מהסיכון)" if x.get("contract_skipped") else "")
                      + (f", החשבון נכשל ב־{str(x['stopped'])[:10]}" if x.get("stopped") else ""))
    try:            # what bot 5's agent found with the same rule (15.11.2026 night)
        b5 = [c for c in json.loads(Path("reports/agents/bot5/cards_smart.json").read_text(encoding="utf-8")).get("cards", [])
              if str((c.get("change") or {}).get("knob", "")).startswith("RISK_scout:")]
        f5 = (b5[-1].get("detail") or {}).get("factor") if b5 else None
        b5t = (f"בבוט 5 אותה בעיה: אחרי 8 אסטרטגיות סייר חדשות הירידה ב־2026 עלתה מ־9.5% ל־18%, והסוכן שם ×{f5} עליהן "
               f"({len(b5)} כרטיסים)" if f5 else "בבוט 5 אין כרגע כרטיס דומה")
    except Exception:  # noqa: BLE001
        b5t = "בבוט 5 אותו כלל נבדק על האסטרטגיות שלו"
    try:
        sc = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("scout") or {}
    except Exception:  # noqa: BLE001
        sc = {}
    tested = {str(x)[6:] for x in s.get("scouts", []) if str(x).startswith("scout:")}
    gen = "scout" in s.get("scouts", [])
    out = []
    for combo in sorted(sc):
        r0 = float(rk.get(f"scout:{combo}", rk.get("scout", FA.RISK_TAL)) or 0)
        if r0 <= 0:
            continue
        to = round(r0 * k, 5)
        inn = combo in tested or (gen and f"scout:{combo}" not in rk)
        out.append(K.card("bot6", "risk", f"בוט 6: {combo} - סיכון {r0 * 100:g}% ← {to * 100:g}% כדי שהבוט כולו יישאר במגבלות החשבון",
                          f"הבוט כולו קודם (עיקרון 11): מסלולי הגאפ רצים ב־×0.3 מאז 14.11 כדי להיכנס במגבלות של FTMO Futures "
                          f"({lm.get('day', 1.5):g}% ביום, {lm.get('all', 4.5):g}% נגרר מתחת לשיא של סוף יום), אבל אסטרטגיות הסייר נוספו מעל זה "
                          f"בסיכון מלא ולא נכנסו לבדיקה הזאת. הבוט כולו היום - 2024-2025: {line(nt)}; 2026: {line(nu)}. "
                          f"רק מסלולי הגאפ - 2024-2025: {line(gt)}; 2026: {line(gu)}. הכלל (אותו כלל של מסלולי הגאפ ושל בוט 5): "
                          f"מקדם אחד על הסיכון של אסטרטגיות הסייר, הגדול ביותר שבו 2024-2025 של הבוט כולו נשאר ב־{s.get('margin', 0.8):.0%} "
                          f"מהמגבלות (ירידה ≤ {s.get('margin', 0.8) * lm.get('all', 4.5):.1f}%, יום ≥ -{s.get('margin', 0.8) * lm.get('day', 1.5):.1f}%) "
                          f"והחשבון לא נכשל - נבחר רק על 2024-2025 → ×{k}" + ("" if s.get("fits") else " (גם כך לא נכנס - המינימום)")
                          + f". עם המקדם - 2024-2025: {line(wt)}; 2026 (לא נראתה): {line(wu)}. "
                          + (f"עם המקדם הסייר עדיין מוסיף לבוט: 2026 {gu.get('total_pct', 0):+.1f}% ← {wu.get('total_pct', 0):+.1f}%. "
                             if wu.get("total_pct", 0) > gu.get("total_pct", 0) else
                             f"שימו לב: עם המקדם הסייר לא מוסיף לבוט ב־2026 ({gu.get('total_pct', 0):+.1f}% בלעדיו, "
                             f"{wu.get('total_pct', 0):+.1f}% איתו) - הצעד הבא: פחות אסטרטגיות סייר, החזקות בלבד (קודם מתקנים). ")
                          + ("" if inn else "האסטרטגיה הזאת לא נכנסה לבדיקה בזמן - מקבלת את אותו מקדם, כדי לא להוסיף סיכון מעל הנבדקות. ")
                          + f"לא הקטנה של מפסידה - התאמה למגבלות החשבון; הבדיקה חוזרת בכל סבב במחשב הבוטים. {b5t}.",
                          None, None, tested=len(s.get("tried") or [1]), risk_pct=round(to * 100, 4),
                          detail={"world": "bot6", "task": "scout_scale", "account_fit": True, "factor": k, "tested": inn,
                                  "scale": {kk: s.get(kk) for kk in ("chosen", "fits", "limits", "now", "with", "gap_only", "tried")}},
                          change={"track": f"scout:{combo}", "knob": f"RISK_scout:{combo}", "from": r0, "to": to}))
    return out


def scout_edge_cards(s, rk):
    """The scouts' risk by each one's edge (scout_edge6): one risk card per tested scout whose risk changes."""
    e = s["edge"]
    lm = e.get("limits") or {}
    ut, uu = (e["uniform"].get("train") or {}), (e["uniform"].get("unseen") or {})
    et, eu = e.get("train") or {}, e.get("unseen") or {}
    line = lambda x: (f"סה\"כ {x.get('total_pct', 0):+.1f}%, היום הגרוע {x.get('worst_day_pct', 0):+.2f}%, "  # noqa: E731
                      f"הירידה הגדולה {x.get('max_fall_pct', 0):.1f}%"
                      + (f", {x['contract_skipped']} עסקאות דולגו כי חוזה מיקרו אחד גדול מהסיכון" if x.get("contract_skipped") else ""))
    rows = sorted(((x, e["w"].get(x), e["edge_r"].get(x)) for x in e.get("risk", {})), key=lambda r: -(r[1] or 0))
    table = " · ".join(f"{x[6:]}: {'' if er is None else f'{er:+.2f}R, '}משקל {w_:.2f}" for x, w_, er in rows)
    out = []
    for x, to_pct in sorted(e.get("risk", {}).items()):
        if not str(x).startswith("scout:"):
            continue
        r0 = float(rk.get(x, rk.get("scout", FA.RISK_TAL)) or 0)
        to = round(float(to_pct) / 100, 6)
        if r0 <= 0 or abs(to - r0) < 1e-6:
            continue
        out.append(K.card("bot6", "risk", f"בוט 6: {x[6:]} - סיכון {r0 * 100:g}% ← {to * 100:g}% לפי היתרון שלה (חלוקת הסיכון של הסייר)",
                          f"היום כל אסטרטגיות הסייר מקבלות אותו מקדם (×0.15 מ־16.11) כדי שהבוט כולו יישאר במגבלות של FTMO Futures "
                          f"({lm.get('day', 1.5):g}% ביום, {lm.get('all', 4.5):g}% נגרר). כך חלשה מקבלת כמו חזקה, ובסיכון כזה חוזה מיקרו אחד "
                          f"גדול מהסיכון של רוב העסקאות - הן מדולגות. החלוקה החדשה (עיקרון 8 ומקרה 41): לכל אחת סיכון לפי היתרון שלה "
                          f"עד 2025 (R לעסקה, משוקלל בזמן), ביחס לחזקה ביותר; בלי יתרון - משקל המינימום ({EDGE_FLOOR:g}; עסקה נלקחת רק כשחוזה מיקרו אחד נכנס בסיכון הזה); "
                          f"אף עסקה לא מעל 0.4%. המקדם הכללי נבחר רק עד 2025 כך שהבוט כולו לא מסוכן יותר מהיום (אותה ירידה ואותו יום גרוע). "
                          f"היתרון והמשקל: {table}. הבוט כולו - היום: 2024-2025 {line(ut)}; 2026 {line(uu)}. "
                          f"עם החלוקה: 2024-2025 {line(et)}; 2026 (לא נראתה) {line(eu)}. "
                          f"בבוט 5 עדיין מקדם אחד לכל הסייר (×0.45) - החלוקה לפי היתרון לא נבדקה שם.",
                          None, None, tested=len(e.get("tried") or [1]), risk_pct=round(to * 100, 4),
                          detail={"world": "bot6", "task": "scout_edge", "account_fit": True, "edge": e},
                          change={"track": x, "knob": f"RISK_{x}", "from": r0, "to": to}))
    return out


def guard6_card(g):
    """The info card of the fixed risk rule (tal1411risk): how many trades the room rule cut / skipped, per period."""
    if not g:
        return None
    w, wt, gu = g["whole"]["unseen"] or {}, g["whole"]["train"] or {}, g["gap_only"]["unseen"] or {}
    many = (w.get("room_cut", 0) + w.get("room_only_skipped", 0)) > 0.05 * max(1, w.get("n", 1))
    what = (f"כלל הסיכון הקבוע: כל עסקה פתוחה נספרת עד הסטופ, ועסקה חדשה מקבלת רק את המקום שנשאר (בבוט 6: {FA.ROOM_USE:.0%} מהמקום "
            f"עד {FA.DAILY_LOSS / FA.INITIAL:.1%} ביום ו־{FA.MAX_LOSS / FA.INITIAL:.1%} נגרר). הבוט כולו - {len(g['tracks'])} מסלולי הגאפ כפי שהם רצים "
            f"ו־{len(g['scouts'])} אסטרטגיות הסייר, כל אחת בסיכון שלה. 2026: מתוך {w.get('n', 0)} עסקאות {w.get('room_cut', 0)} הוקטנו ו־"
            f"{w.get('room_only_skipped', 0)} דולגו בגלל המקום (ועוד {w.get('contract_skipped', 0)} כי חוזה מיקרו אחד גדול מהסיכון של העסקה); "
            f"הסיכון הפתוח הגבוה בלי הכלל {w.get('open_peak_pct')}%, היום הגרוע {w.get('worst_day_pct')}%, הירידה הגדולה {w.get('max_fall_pct')}%. "
            f"2024-2025: {wt.get('room_cut', 0)} הוקטנו, {wt.get('room_only_skipped', 0)} דולגו מתוך {wt.get('n', 0)}. "
            f"רק מסלולי הגאפ (2026): {gu.get('room_cut', 0)} / {gu.get('room_only_skipped', 0)}. "
            + ("זה הרבה - המשימה הבאה: לתקן את החפיפה בין האסטרטגיות (קודם מתקנים)." if many else
               "מעט - אין חפיפה שצריך לתקן; שום שינוי לא יאושר אם הוא פותח סיכון מעבר למקום הזה.")
            + (f" לא נבדקו: {', '.join(g['missing'])}." if g["missing"] else ""))
    return K.card("bot6", "info", f"בוט 6: כלל הסיכון הקבוע - ב־2026 {w.get('room_cut', 0)} עסקאות הוקטנו ו־"
                  f"{w.get('room_only_skipped', 0)} דולגו בגלל המקום שנשאר (מתוך {w.get('n', 0)})", what, None, None,
                  tested=1, detail={"world": "bot6", "task": "tal1411risk", "guard": g})


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
                      f"{x.get('room_skipped', 0)} / {x.get('room_cut', 0)}" + (f" (מהן {x['contract_skipped']} כי חוזה מיקרו אחד גדול מהסיכון)"
                                                                              if x.get("contract_skipped") else "")
                      + (f", החשבון נכשל ב־{str(x['stopped'])[:10]}"
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
                          f"מסכן יותר מ־{to * 100:g}% מהחשבון לא נלקחת, וכמות תמיד מעוגלת למטה - הבדיקה עושה בדיוק כמו החי "
                          f"(חוזים שלמים, עד {FA.MAX_MICROS} פתוחים). {b5t}.",
                          None, None, tested=1, risk_pct=round(to * 100, 4),
                          # account_fit (13.11.2026 night): the risk the account's own fixed limits allow, not a cut of a
                          # losing strategy - agents/autodecide does not hold it for "fix first"
                          detail={"world": "bot6", "account": a, "account_fit": True, "account_fails_now": fail},
                          change={"track": trk, "knob": f"RISK_{trk}", "from": float(r), "to": to}))
    return out


# ONE MICRO UP TO (bot 6's smart agent, 14.11.2026 night): at the risk the account's limits allow (0.12% a track since
# the x 0.3 card of 14.11), one MNQ / MES / MYM is often more than a trade may risk - the account test of that card
# skipped 352 trades on 2024-2025 and 250 on 2026 for that alone (live/fut_account.size "small"). The variant: such a
# trade is taken with ONE contract when it risks at most 0.2 / 0.3 / 0.4% of the balance (never over the bot's own
# 0.4%) and fits the room left to the account's limits. Chosen on 2024-2025: the most total %, the train period still
# inside 80% of the limits (a fall <= 3.6%, a day >= -1.2%) and no account failure; a card only if 2026 is also better
# in total, its worst day and deepest fall stay inside the same margin. Bot 5 trades CFDs in fractions of a lot - it has
# no such skip, so this is bot 6's own.
ONE6 = (0.002, 0.003, 0.004)


def one6_cards(pf):
    a = (pf or {}).get("account") or {}
    om, lm = a.get("one_micro") or {}, ((a.get("scale") or {}).get("limits") or {})
    now = a.get("now") or {}
    nt, nu = now.get("train") or {}, now.get("unseen") or {}
    if not om or not nt or not nu:
        return [], {}
    L = fut_lim()
    dl, al = 0.8 * L["day"], 0.8 * L["all"]
    fits = lambda x: bool(x) and not x.get("stopped") and x.get("max_fall_pct", 99) <= al + 1e-9 \
        and x.get("worst_day_pct", -99) >= -dl - 1e-9        # noqa: E731
    # the main goal (data/agents/tal_thinking.md): more profit AND no more risk - more total %, the worst day and the
    # deepest fall no worse than the bot as it runs, inside 80% of the limits, no failure
    better = lambda x, b: fits(x) and x.get("total_pct", -99) > b.get("total_pct", -99) \
        and x.get("worst_day_pct", -99) >= b.get("worst_day_pct", 0) - 1e-9 \
        and x.get("max_fall_pct", 99) <= b.get("max_fall_pct", 0) + 1e-9      # noqa: E731
    best = None
    for k, row in om.items():
        tr = row.get("train") or {}
        if better(tr, nt) and (best is None or tr["total_pct"] > om[best]["train"]["total_pct"]):
            best = k
    res = {"now": now, "variants": om, "chosen": best}
    if best is None:
        return [], res
    bt, bu = om[best]["train"], om[best]["unseen"] or {}
    if not better(bu, nu):
        return [], res
    line = lambda x: (f"סה\"כ {x.get('total_pct', 0):+.1f}%, היום הגרוע {x.get('worst_day_pct', 0):+.1f}%, "   # noqa: E731
                      f"הירידה הגדולה {x.get('max_fall_pct', 0):.1f}%, דולגו {x.get('room_skipped', 0)}"
                      + (f" (מהן {x['contract_skipped']} כי חוזה אחד גדול מהסיכון)" if x.get("contract_skipped") else "")
                      + (f", בחוזה אחד {x['one_micro']}" if x.get("one_micro") else ""))
    to = float(best)
    card = K.card("bot6", "risk", f"בוט 6 · שיטת הגאפ: עסקה שחוזה מיקרו אחד שלה גדול מהסיכון - בחוזה אחד, עד {to * 100:g}%",
                  f"היום עסקה שחוזה MNQ / MES / MYM אחד שלה מסכן יותר מהסיכון של המסלול (0.12% מאז כרטיס ×0.3) לא נלקחת. "
                  f"ההצעה: לקחת אותה בחוזה אחד, אם הוא מסכן עד {to * 100:g}% מהחשבון (לא מעל הסיכון של הבוט, 0.4%) ויש מקום "
                  f"במגבלות של החשבון. הבוט כולו, כמו בחי (FTMO Futures {L['day']:g}% ביום, {L['all']:g}% מתחת לשיא): "
                  f"2024-2025: היום {line(nt)} ← {line(bt)}; 2026 (לא נראתה): היום {line(nu)} ← {line(bu)}. "
                  f"נבחר על 2024-2025 בלבד מתוך {len(om)} ערכים (0.2 / 0.3 / 0.4%): יותר רווח, והיום הגרוע והירידה הגדולה לא גרועים מהיום, בשתי התקופות. "
                  "בבוט 5 אין דילוג כזה (CFD בשברי לוט) - זה של בוט 6 בלבד.",
                  None, None, tested=len(om), risk_pct=round(to * 100, 3),
                  detail={"world": "bot6", "account": a, "one_micro": res},
                  change={"track": "tal_all", "knob": "ONE_MICRO", "from": 0.0, "to": to})
    return [card], res


# SIZE UP AFTER A WIN (the user's task tal1411anti, 14.11.2026, every bot; bot 5's anti_cards, shared): (a) the next
# trade of the same track at x1.25 the risk after a winning trade, up to x2, back to x1 after a loss; (b) the pyramid
# (PYR, per track in TASK6, and here on the whole bot). On bot 6 as the account takes it: every running track at its
# real risk, the day's total in % of the balance, a trade's multiplier only from the track's trades CLOSED before it.
# Fair against more risk (the main goal): the variant is scaled down so its deepest fall on 2024-2025 equals the bot's
# as it runs, then compared on both periods - and within bot 6's own limits (FTMO Futures, fut_lim), not bot 5's.
def anti6_cards(idx, b5=None):
    rk = {**FA.RISK}
    try:
        rk.update(json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("risk") or {})
    except Exception:  # noqa: BLE001
        pass
    rows, prow, miss, pmiss = [], [], [], []
    for trk, (pn, cfg) in live_tracks().items():
        r_ = float(rk.get(trk, FA.RISK_TAL) or 0) * 100
        if r_ <= 0:
            continue
        c0 = TG._norm(dict(cfg))
        t = _t6(idx, pn, c0)
        if t is None:
            miss.append(trk)
            continue
        t = t.sort_values("time").reset_index(drop=True)
        mins = t["mins"] if "mins" in t else pd.Series(0.0, index=t.index)
        t["out"] = t["time"] + pd.to_timedelta(mins.fillna(0).astype(float) + 1, unit="min")
        t["base"] = t["r"] * r_
        t["anti"] = t["r"] * r_ * A._anti_mult(t)
        rows.append(t[["time", "base", "anti"]])
        pt = _t6(idx, pn, TG._norm(dict(c0, PYR=1.0)))
        if pt is None:
            pmiss.append(trk)
        else:
            prow.append(pd.DataFrame({"time": pt["time"], "pyr": pt["r"] * r_}))
    if not rows:
        return [], {"state": "running", "missing": miss}
    t = pd.concat(rows, ignore_index=True)
    lm = fut_lim()
    res = {"state": "done" if not miss else "partial", "missing": miss, "pyr_missing": pmiss, "rule": A.ANTI,
           "limits": {"day": lm["day"], "all": lm["all"]}, "now": A._pct_view(t, "base"), "anti": A._pct_view(t, "anti")}
    if prow and not pmiss and not miss:
        res["pyr"] = A._pct_view(pd.concat(prow, ignore_index=True), "pyr")
    b5t = "בבוט 5: " + ("; ".join(f"{nm} {'עדיף' if (b5.get(nm) or {}).get('better') else 'לא עדיף'}" for nm in ("anti", "pyr") if b5.get(nm))
                        if b5 and b5.get("state") in ("done", "partial") else "הבדיקה עוד רצה")
    res["bot5"] = b5t
    cards, b = [], res["now"]
    for nm in ("anti", "pyr"):
        v = res.get(nm)
        if not (v and v.get("train") and v.get("unseen") and b.get("train") and b.get("unseen")) or miss:
            continue
        k = min(1.0, b["train"]["max_fall_pct"] / max(v["train"]["max_fall_pct"], 1e-9))   # the same risk as now
        vt, vu = v["train"]["total_pct"] * k, v["unseen"]["total_pct"] * k
        res[nm]["fair_k"] = round(k, 3)
        res[nm]["fair"] = {"train_total_pct": round(vt, 2), "unseen_total_pct": round(vu, 2),
                           "unseen_worst_day_pct": round(v["unseen"]["worst_day_pct"] * k, 2),
                           "unseen_max_fall_pct": round(v["unseen"]["max_fall_pct"] * k, 2)}
        win = (vt > b["train"]["total_pct"] and vu > b["unseen"]["total_pct"]
               and v["unseen"]["worst_day_pct"] * k >= b["unseen"]["worst_day_pct"]
               and v["unseen"]["max_fall_pct"] * k <= b["unseen"]["max_fall_pct"]
               and v["unseen"]["worst_day_pct"] * k > -0.8 * lm["day"] and v["unseen"]["max_fall_pct"] * k < 0.8 * lm["all"])
        res[nm]["better"] = bool(win)
        if not win:
            continue
        he = "x1.25 אחרי עסקה מנצחת באותו מסלול, עד x2, חזרה לבסיס אחרי הפסד" if nm == "anti" else \
             "חצי פוזיציה נוספת ב־1R, הסטופ של הכל כך שהסיכון נשאר 1R"
        cards.append(K.card(
            "bot6", "improve", "בוט 6 · שיטת הגאפ, כל המסלולים: " + ("הגדלה אחרי ניצחון" if nm == "anti" else "פירמידה"),
            f"משימה של טל מ־14.11. {he}. הבוט כולו (המסלולים שרצים בסיכון שלהם), מוקטן ל־x{k:.2f} כדי שהירידה הגדולה "
            f"ב־2024-2025 תהיה כמו היום. 2024-2025: {b['train']['total_pct']:+.1f}% ← {vt:+.1f}%; 2026 (לא נראתה): "
            f"{b['unseen']['total_pct']:+.1f}% ← {vu:+.1f}%, היום הגרוע {b['unseen']['worst_day_pct']:+.2f}% ← "
            f"{v['unseen']['worst_day_pct'] * k:+.2f}%, ירידה {b['unseen']['max_fall_pct']:.1f}% ← {v['unseen']['max_fall_pct'] * k:.1f}% "
            f"(המגבלות של החשבון: {lm['day']:g}% ביום, {lm['all']:g}% מהשיא). {b5t}. צריך קוד בבוט החי (נבנה אחרי אישור).",
            None, None, tested=2, risk_pct=None, detail={"world": "bot6", "anti": res},
            change={"method": "tal_gap_code", "track": "tal_all", "knob": "ANTI" if nm == "anti" else "PYR",
                    "to": A.ANTI if nm == "anti" else 1.0, "scale": round(k, 3)}))
    return cards, res


# THE STRADDLE BEFORE A RED RELEASE (the user's task tal1411strad, 14.11.2026: bot 7 builds EV04 in
# agents/news_straddle.py, bots 5-6 use the same code and cache) on bot 6's three indices. The study's costs are
# FTMO's CFD spread at the fill minute + the extra news slippage - wider than a micro future's tick + commission, so a
# variant that holds here holds with room on the futures; only the FTMO-safe start (minute 3) may be proposed.
def straddle6_runs(budget_s):
    keep = A.STRAD_SYMS
    try:
        A.STRAD_SYMS = list(WORLD["symbols"])
        return A.straddle_runs(budget_s)
    finally:
        A.STRAD_SYMS = keep


def straddle6_cards(b5=None):
    from agents import news_straddle as NS
    rows, have = [], []
    for sym in WORLD["symbols"]:
        try:
            c_ = json.loads((NS.CACHE / f"{sym}.json").read_text(encoding="utf-8"))
            if c_.get("version") == NS.VERSION:
                rows += c_["rows"]
                have.append(sym)
        except Exception:  # noqa: BLE001
            pass
    if not rows:
        return [], {"state": "running", "symbols": have}
    df = pd.DataFrame(rows)
    st = NS.study(df)
    safe = NS.study(df[df["v"].astype(str).str.startswith("safe|")])
    res = {"state": "done" if len(have) == len(WORLD["symbols"]) else "partial", "symbols": have,
           "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "all": {k: st.get(k) for k in ("baseline", "best", "robust", "tested")},
           "safe": {k: safe.get(k) for k in ("baseline", "best", "robust", "tested")}}
    b5b = ((b5 or {}).get("safe") or {}).get("best") or {}
    res["bot5"] = (f"בבוט 5 (המדדים והזהב): {b5b.get('v')}, 2026 {((b5b.get('unseen') or {}).get('sum') or 0):+.1f}R"
                   if b5b else "בבוט 5 הבדיקה עוד רצה")
    b = safe.get("best")
    rob = safe.get("robust") or {}
    if res["state"] != "done" or not b or not b["train"]["sum"] or b["train"]["sum"] <= 0 or (b["unseen"]["sum"] or 0) <= 0 \
            or rob.get("top_positive_unseen", 0) * 2 < rob.get("top", 10):
        return [], res
    return [K.card("bot6", "new", f"בוט 6 · סטרדל לפני חדשה אדומה (EV04) על 3 המדדים: {b['v']}",
                   f"משימה של טל מ־14.11. Buy Stop + Sell Stop אחרי החדשה (דקה 3, מחוץ לחלון של FTMO), OCO, סיכון קבוע, בלי "
                   f"הכפלה. נבחר על 2025, 2026 (לא נראתה): {b['unseen']['n']} עסקאות, {b['unseen']['sum']:+.1f}R. מתוך 10 הטובות "
                   f"ב־2025, {rob.get('top_positive_unseen')} מרוויחות ב־2026. עלויות CFD של FTMO (רחבות מחוזה מיקרו). "
                   f"מסלול נפרד בסיכון של מסלולי הגאפ. {res['bot5']}.",
                   b["train"], b["unseen"], tested=safe.get("tested") or 1, risk_pct=0.28,
                   detail={"world": "bot6", "straddle": res},
                   change={"method": "straddle_code", "v": b["v"], "symbols": have})], res


STEPS = {}                 # this run's gap-method steps: {name: {done | err, at, sec, budget_sec}} -> results bot6.steps


def census6(budget_s):
    """Task tal1611partial (1): of the losing trades of each running track (the rules as the live bot runs them, the
    three indices), how many first reached +0.5R / +1R / half the way to the target (tal_gap.simulate's mfe) - to 2025
    and on 2026. Kept a week (the tracks' rules change rarely); a census, no choice is made on it."""
    try:
        old = (json.loads(TG.RESULTS.read_text(encoding="utf-8")).get("bot6") or {}).get("gaveback_census") or {}
        if old.get("at") and pd.Timestamp.now(tz="UTC") - pd.Timestamp(old["at"]) < pd.Timedelta(days=7) \
                and old.get("tracks") and all(isinstance(v, dict) and "unseen" in v for v in old["tracks"].values()):
            return old
    except Exception:  # noqa: BLE001
        pass
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    t0, got = time.time(), {}
    run = {k: (pn, TG.ftmo_cfg(c)) for k, (pn, c) in live_tracks().items() if pn in pairs}
    for sym in WORLD["symbols"]:
        if time.time() - t0 > budget_s:
            break
        D = tal_load(sym)
        if D is None:
            continue
        gcs = {}
        for trk, (pn, c) in run.items():
            t = TG.simulate(D, pairs[pn], c, gcs.setdefault(pairs[pn][0], {}))
            if len(t):
                got.setdefault(trk, []).append(t)
    out = {}
    for trk, ts in got.items():
        t = pd.concat(ts, ignore_index=True)
        t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        row = {"pair": run[trk][0], "symbols": len(ts)}
        for nm, x in (("train", t[t["time"] < A.SPLIT]), ("unseen", t[t["time"] >= A.SPLIT])):
            lo = x[(x["r"] < 0) & x["mfe"].notna()]
            n = max(len(lo), 1)
            row[nm] = {"trades": int(len(x)), "losses": int(len(lo)),
                       "to_05r": round(100 * float((lo["mfe"] >= 0.5).sum()) / n, 1),
                       "to_1r": round(100 * float((lo["mfe"] >= 1.0).sum()) / n, 1),
                       "to_half": round(100 * float((lo["mfe"] >= 0.5 * lo["rr"]).sum()) / n, 1),
                       "r_losses": round(float(lo["r"].sum()), 1)}
        out[trk] = row
    return {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "tracks": out,
            "note": "% of the losing trades that first reached +0.5R / +1R / half the way to the target"}


def tal(budget_s):
    t0 = time.time()
    idx0 = {f.stem: json.loads(f.read_text()) for f in (TAL_DIR / "idx").glob("*.json")} if (TAL_DIR / "idx").exists() else {}
    # each step's outcome into the results (bot6.steps, 14.11.2026, as bot 5): the bots' computer keeps its logs to
    # itself, so a step that failed or found no time was invisible to the nightly smart agent in the cloud
    # the user's tasks get the search's minutes once the search has nothing left (bot 6's smart agent, 14.11.2026
    # night): the last round's search ran 0 new runs in 744 s while the user's 14.11 queue (about 30 knobs x 4 tracks
    # x 3 indices, ~6.5 s a run) needed several rounds at 0.4 - "no waiting" (principle 12)
    try:
        _srch = (json.loads(TG.RESULTS.read_text(encoding="utf-8")).get("bot6") or {}).get("steps", {}).get("search", {})
        ushare = 0.6 if _srch.get("done") == 0 else 0.4
    except Exception:  # noqa: BLE001
        ushare = 0.4
    for nm, fn, share in (("user_tasks", lambda b: filt_runs(idx0, b, USER6), ushare),   # the user's tasks first (14.11)
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
    try:            # each pair's chosen version with FTMO's break rule on, alone and + REARM (15.11.2026)
        STEPS["ftmo_pairs"] = {"done": ftmo_runs(idx0, budget_s * 0.1)}
    except Exception as e:  # noqa: BLE001
        STEPS["ftmo_pairs"] = {"err": str(e)[:300]}
        print(f"tal ftmo runs failed: {e}")
    STEPS["ftmo_pairs"].update(at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), sec=round(time.time() - t1),
                               budget_sec=round(budget_s * 0.1))
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
        # the whole bot on the rules exactly as the live bot runs them (b5to6brk, 14.11.2026 night): every track with
        # FTMO's break rule on (TG.ftmo_cfg) once its runs exist on the three indices, else as before (noted)
        def _has(pn_, c_):
            c_ = TG._norm(dict(c_))
            return {v["sym"] for v in idx.values() if v["pair"] == pn_ and TG._norm(dict(v["cfg"])) == c_} >= set(WORLD["symbols"])
        counted = {k: ((pn, TG.ftmo_cfg(c)) if _has(pn, TG.ftmo_cfg(c)) else (pn, c)) for k, (pn, c) in counted.items()}
        STEPS["portfolio_rules"] = {k: ("ftmo" if "WKND" in c else "without WKND (runs missing)") for k, (_, c) in counted.items()}
        # the hour filter and the direction cap are bot 5's methods (13.11.2026), shared: chosen on 2024-2025 only.
        # Bot 6 has ~600 gap trades to 2025 against bot 5's ~5,000, so an hour needs 30 trades here (100 there),
        # with the same t <= -1.5
        # the account in % with each track's real risk (bot 5's rule, 13.11.2026 night, shared): the bot's limits
        # in % - not in R - and one factor on the tracks' risk chosen on 2024-2025 (scale_cards)
        pf = TG.portfolio(idx, WORLD["symbols"], TAL_DIR, counted, hour_min_n=30,
                          risk={k: float(rk.get(k, FA.RISK_TAL)) for k in counted}, lim=fut_lim(), one_ups=ONE6)
        if pf:
            try:
                pcards += scale_cards(pf, {k: float(rk.get(k, FA.RISK_TAL)) for k in counted})
            except Exception as e:  # noqa: BLE001
                print(f"tal_gap scale failed: {e}")
            try:        # one micro up to 0.2-0.4% (14.11.2026 night)
                oc_, ores_ = one6_cards(pf)
                pcards += oc_
                doc["bot6"]["one_micro"] = ores_
            except Exception as e:  # noqa: BLE001
                print(f"tal_gap one micro failed: {e}")
            doc["bot6"]["portfolio"] = pf
            try:        # the fixed risk rule on the whole bot, scouts included (tal1411risk, 14.11.2026 night)
                # 0.05 of the run left 4 of the 10 scouts out of the 15.11 round (time) - the fixed rule needs them all
                g6 = guard6(idx, counted, rk, budget_s * 0.12)
                if g6:
                    doc["bot6"]["risk_guard"] = g6
                    gc6 = guard6_card(g6)
                    if gc6:
                        pcards.append(gc6)
                    pcards += scout_scale_cards(g6, rk)     # the scouts' risk on the whole bot (15.11.2026 night)
            except Exception as e:  # noqa: BLE001
                print(f"tal_gap risk guard failed: {e}")
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
    try:            # the losses that were in profit first (tal1611partial (1), 16.11.2026)
        t1 = time.time()
        doc["bot6"]["gaveback_census"] = census6(budget_s * 0.08)
        STEPS["gaveback_census"] = {"done": len(doc["bot6"]["gaveback_census"].get("tracks") or {}),
                                    "sec": round(time.time() - t1), "budget_sec": round(budget_s * 0.08)}
    except Exception as e:  # noqa: BLE001
        STEPS["gaveback_census"] = {"err": str(e)[:300]}
        print(f"tal_gap census failed: {e}")
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
    for gk, grp in TASK6.items():   # the user's tasks of 14.11.2026, each group head to head on what runs
        try:
            gc, gres = filt_cards(idx, grp)
            pcards += gc
            if gres:
                doc["bot6"].setdefault("tasks1411", {})[gk] = gres
                TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"tal_gap {gk} cards failed: {e}")
    try:            # size up after a win / the pyramid on the whole bot (tal1411anti)
        ac_, ares_ = anti6_cards(idx, (doc.get("bot5") or {}).get("anti"))
        pcards += ac_
        doc["bot6"]["anti"] = ares_
        TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap anti failed: {e}")
    try:            # EV04 on the three indices (tal1411strad) - the shared cache, so bot 5 / 7's runs cost nothing
        t1 = time.time()
        STEPS["straddle"] = {"done": straddle6_runs(budget_s * 0.1), "sec": round(time.time() - t1),
                             "budget_sec": round(budget_s * 0.1)}
        sc6, sres6 = straddle6_cards((doc.get("bot5") or {}).get("straddle"))
        pcards += sc6
        doc["bot6"]["straddle"] = sres6
        TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"tal_gap straddle failed: {e}")
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
    # ... chosen only among the versions with FTMO's break rule on - what the card shows is what would trade (15.11.2026)
    cpairs = pairs
    try:
        cpairs, fnote = ftmo_pairs(idx, pairs)
        doc["bot6"]["ftmo_pairs"] = fnote
        TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - no pair card on the rules without the break rule
        print(f"tal ftmo pairs failed: {e}")
        cpairs = {pn: dict(x, chosen=None) for pn, x in pairs.items()}
    return TG.cards("bot6", cpairs, doc.get("bot5"), "בוט 5", risk_pct=FA.RISK_TAL * 100,
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
# 14.11.2026 afternoon: the scout's strategies from the user's videos of 14.11 that are close to his gap method and built
# for the New York session - SC258 (the Judas swing of the 09:30-09:45 range + CISD), SC253 (the Silver Bullet's three
# windows), SC250 (an FVG with a weak candle 3), SC251 (an FVG with displacement), SC286 (Candle Range Theory) - first.
# 15.11.2026 night 3 (weekend, no new run yet): SC325 first - the edgeful power hour (a new day high / low 15:00-15:50 NY,
# out at 16:00), a pure cash-session index pattern, at most one trade a day (quick to run; the 15:45 no-entry rule of
# FTMO Futures cuts its last 5 minutes, as live). After SC286: the correlated-index ones, natural for MNQ / MES / MYM -
# SC292 (SMT divergence -> iFVG -> CISD), SC291 (the FVG return only on the stronger index), SC303 (NY sweeps London +
# iFVG) and SC307 (Power of Three around the NY midnight open).
DISCOVER = [f"{sid}|{tf}|{sym}" for sid, tfs in (("SC325", ("5Min", "15Min")), ("SC258", ("5Min", "1Min")),
                                                 ("SC253", ("5Min", "1Min")),
                                                 ("SC250", ("5Min", "1Min")), ("SC251", ("5Min",)), ("SC286", ("15Min",)),
                                                 ("SC292", ("5Min",)), ("SC291", ("5Min",)), ("SC303", ("5Min", "1Min")),
                                                 ("SC307", ("5Min",)),
                                                 ("SC104", ("1Min",)), ("SC058", ("5Min", "15Min", "1Min")), ("SC063", ("5Min",)),
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


# THE SCOUTS' SESSION WINDOW AND BREAKEVEN POINT (bot 6's smart agent, 16.11.2026 night; DISCOVER + principle 7).
# The futures trade 23 hours; the scout strategies were chosen on all of them. 16.11 00:55 UTC (19:55 New York, an hour
# after the Sunday open) SC072|5Min|US500 stopped in 5 minutes at 0.4%, -$389 - the day's only trade. Half of the live
# scouts run with no breakeven at all (the variant test had one point only), against the user's principle 7. Here, per
# scout strategy bot 6 trades, its LIVE variant against one change at a time: a New York session window
# (agents/scout_fut.SESS) or the breakeven's own point (BE_WAY with a target, BE_R without). Chosen on 2024-2025 only
# (the most time-weighted R, at least SESS_MIN_GAIN R over the live variant there, eff_n >= 20), shown on 2026; an
# 'improve' card with the live variant as the baseline (the manager's judge: more per trade and no less in total in
# 2026), the change {"family": "scout_fut", "combo", "v"} the builder builds at once. Rerun every SCOUT_RERUN_DAYS.
SESS_SHARE = 0.15
SESS_MIN_GAIN = 1.0


def _sess_cands(v):
    from agents import scout_fut as SF
    out = [dict(v, sess=x) for x in SF.SESS]
    own_tg = v.get("target") != "none"
    for b in (SF.BE_WAY if own_tg else SF.BE_R):
        out.append(dict(v, be=b))
    return [c for c in out if SF.vid(c) != SF.vid(v)]


def sess6_runs(budget_s):
    from agents import scout_fut as SF
    t0, f = time.time(), A.OUT / "scout_sess.json"
    try:
        res = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        res = {}
    try:
        sc = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("scout") or {}
    except Exception:  # noqa: BLE001
        sc = {}
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    now, data = pd.Timestamp.now(tz="UTC"), {}

    def stats(t):
        if t.empty:
            return None
        tt = pd.to_datetime(t["time"], utc=True)
        tr, un = t[tt < A.SPLIT], t[tt >= A.SPLIT]
        day = lambda x: round(float(x.groupby(pd.to_datetime(x["time"], utc=True).map(FA.tday))["r"].sum().min()), 2) if len(x) else None  # noqa: E731
        return {"train": K.wstats(tr["r"], K.weights(tr["time"])) if len(tr) else K.wstats([]),
                "unseen": K.wstats(un["r"], K.weights(un["time"])) if len(un) else K.wstats([]),
                "train_sum": round(float(tr["r"].sum()), 2), "unseen_sum": round(float(un["r"].sum()), 2),
                "train_wsum": round(float((tr["r"] * K.weights(tr["time"])).sum()), 2) if len(tr) else 0.0,
                "worst_day_train": day(tr), "worst_day_unseen": day(un),
                "halves": K.halves(tr["time"], tr["r"]) if len(tr) else [None, None]}
    for combo, e in sc.items():
        v = dict((e or {}).get("v") or {})
        old = res.get(combo) or {}
        if old.get("done") and old.get("live_vid") == SF.vid(v) and (now - pd.Timestamp(old["at"])).days < SCOUT_RERUN_DAYS:
            continue
        sid, tf, sym = combo.split("|")
        if sym not in FA.SPEC or tf not in SF.TF_MIN or time.time() - t0 > budget_s:
            continue
        if sym not in data:
            try:
                pr = prep(sym, end)
                if pr is None:
                    data[sym] = None
                else:
                    df, _, _, noent = pr
                    shut, _ = FA.session_masks(df)
                    _, pv, tick = FA.SPEC[sym]
                    data[sym] = (df, noent, shut, tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv)
            except Exception as ex:  # noqa: BLE001
                print(f"scout_sess {sym}: no data ({ex})")
                data[sym] = None
        if data[sym] is None:
            continue
        df, noent, shut, cost_px = data[sym]
        _, pv_, tick_ = FA.SPEC[sym]
        rows = dict(old.get("rows") or {}) if old.get("live_vid") == SF.vid(v) else {}
        for c in [v] + _sess_cands(v):
            k = SF.vid(c)
            if k in rows:
                continue
            if time.time() - t0 > budget_s:
                break
            try:
                t = SF.trades(df, sym, sid, tf, c, noent, shut, cost_px)
                if not t.empty:     # one micro contract over the bot's risk is never taken live
                    per = (t["entry_px"] - t["stop_px"]).abs() * pv_ + FA.COMM + FA.SLIP_TICKS * tick_ * pv_
                    t = t[per <= FA.INITIAL * FA.RISK_TAL]
                rows[k] = {"v": c, **(stats(t) or {})}
            except Exception as ex:  # noqa: BLE001
                rows[k] = {"v": c, "error": str(ex)[:200]}
        done = all(SF.vid(c) in rows for c in [v] + _sess_cands(v))
        res[combo] = {"at": now.isoformat(timespec="seconds"), "live_vid": SF.vid(v), "done": done, "rows": rows}
        print(f"  scout_sess {combo}: {len(rows)} variants{'' if done else ' (more next run)'}")
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def sess6_cards(res):
    from agents import scout_fut as SF
    try:
        sc = json.loads((Path(__file__).resolve().parent / "live" / "bots" / "bot6.json").read_text(encoding="utf-8")).get("scout") or {}
    except Exception:  # noqa: BLE001
        sc = {}
    fmt = lambda x: f"{(x or {}).get('mean') or 0:+.2f}R על {(x or {}).get('n', 0)} עסקאות"  # noqa: E731
    cards, summ = [], []
    for combo, x in (res or {}).items():
        e = sc.get(combo)
        if not x.get("done") or not e or x.get("live_vid") != SF.vid(e.get("v") or {}):
            continue
        rows = x["rows"]
        base = rows.get(x["live_vid"]) or {}
        if not base.get("train"):
            continue
        ok = [r for k, r in rows.items() if k != x["live_vid"] and (r.get("train") or {}).get("eff_n", 0) >= 20
              and r.get("train_wsum", -1e9) >= base.get("train_wsum", 0) + SESS_MIN_GAIN]
        best = max(ok, key=lambda r: r["train_wsum"], default=None)
        if not best:
            summ.append(f"{combo}: אין שינוי שעדיף עד 2025")
            continue
        bt, bu, ct, cu = base["train"], base["unseen"], best["train"], best["unseen"]
        hold = (cu.get("mean") or 0) > (bu.get("mean") or 0) and best.get("unseen_sum", 0) >= base.get("unseen_sum", 0)
        summ.append(f"{combo}: {SF.he(best['v'])} - 2026 {best.get('unseen_sum', 0):+.1f}R מול {base.get('unseen_sum', 0):+.1f}R"
                    + ("" if hold else " (לא מחזיק ב־2026)"))
        if not hold:
            continue
        what = (f"{combo} כפי שהוא רץ עכשיו ({SF.he(e['v'])}) מול שינוי אחד בכל פעם - חלון שעות ניו יורק (4) או נקודת האיזון "
                f"(עיקרון 7: {'5 רמות פיבונאצ׳י של הדרך ליעד' if e['v'].get('target') != 'none' else '1.5 / 2 / 3R'}), "
                f"{len(rows) - 1} שינויים. נבחר על 2024-2025 בלבד: {fmt(ct)}, סה\"כ {best.get('train_sum', 0):+.1f}R מול {base.get('train_sum', 0):+.1f}R היום; "
                f"ב־2026 (לא נראתה): {fmt(cu)}, סה\"כ {best.get('unseen_sum', 0):+.1f}R מול {base.get('unseen_sum', 0):+.1f}R ({fmt(bu)}). "
                f"היום הגרוע של האסטרטגיה ב־2026: {best.get('worst_day_unseen')}R מול {base.get('worst_day_unseen')}R. "
                f"חוזי מיקרו, עלויות אמיתיות, שעות FTMO Futures. בבוט 5 אין עדיין בדיקת שעות / איזון לאסטרטגיות הסייר (רק למסלולי הגאפ) - "
                f"התוצאה עוברת לסוכן של בוט 5.")
        cards.append(K.card("bot6", "improve", f"בוט 6: {combo} · {SF.he(best['v'])}", what, ct, cu, baseline=bu,
                            halves_=best.get("halves"), tested=len(rows) - 1, cost_r=None,
                            change={"family": "scout_fut", "combo": combo, "v": best["v"]},
                            detail={"world": "bot6", "task": "scout_sess", "live": base, "chosen": best}))
    if summ:
        cards.append(K.card("bot6", "info", f"בוט 6: חלון שעות ונקודת איזון לאסטרטגיות הסייר - {len(cards)} שיפורים מחזיקים",
                            "לכל אסטרטגיית סייר שרצה: שינוי אחד בכל פעם, נבחר עד 2025, מוצג על 2026. " + " · ".join(summ),
                            None, None, tested=len(summ), detail={"world": "bot6", "task": "scout_sess"}))
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
    try:            # the scouts' session window / breakeven point (16.11.2026 night)
        scards += sess6_cards(sess6_runs(SESS_SHARE * mins * 60))
    except Exception as e:  # noqa: BLE001
        print(f"scout_sess failed: {e}")
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
