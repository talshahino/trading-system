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


def tal(budget_s):
    t0 = time.time()
    idx, left = TG.run("bot6", WORLD["symbols"], tal_load, TAL_DIR, budget_s * 0.8)
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
        pf = TG.portfolio(idx, WORLD["symbols"], TAL_DIR, counted, hour_min_n=30)
        if pf:
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
DISCOVER = [f"{sid}|{tf}|{sym}" for sid, tfs in (("SC058", ("5Min", "15Min", "1Min")), ("SC063", ("5Min",)),
                                                 ("SC011", ("1Min",)), ("SC052", ("1Min",)), ("SC003", ("1Min",)),
                                                 ("SC047", ("1Min",)), ("SC021", ("5Min",)), ("SC028", ("5Min",)))
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
