"""Bot 7's release / shock strategies' trades for the whole-bot check (bot 7 smart agent, 15.11.2026).

Why: the whole-bot check (agents/portfolio_check.book_now) knows a combo's trades only from the stored experiments
(reports/agents/<bot>/exp/<family>_<combo>.json: [entry minute, exit minute, direction, R]). Bot 7's own strategies
are measured in their own studies (EV02 / EV03 / SH01 in agents/news_regime.py, EV04 in agents/news_straddle.py, EV08 in
agents/news_surprise.py), which kept no trade list - so 46 of bot 7's 55 live combos were "missing" in the replay, the
manager judged every bot 7 card on SC001 / SC086 alone, and the guard counts (task tal1411risk, part 3) left out the
straddle's 447 trades of 2026. Each study now writes, for every built combo, the trades of the version the live bot
runs (its plan in live/bots/bot7.json): exp/study_<sid>_<tf>_<sym>.json {"tr": {"built": [...]}, "tr25": {...}}, with
the plan it was made from ("plan_sig"), so a later card that changes the plan makes it unknown until the study runs
again (portfolio_check.current_trades).

Exit minute: the studies' rows carry "x" (the exit time) from 15.11.2026; rows cached before have none, and their exit
is put at the end of the holding time (entry + hold minutes - the longest the trade can be open, so the open-risk
count of the guard errs on the safe side) until the cache is rebuilt (each study: every STALE_DAYS days).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "reports" / "agents" / "bot7" / "exp"
PLANS_F = ROOT / "live" / "bots" / "bot7.json"
TEST_FROM, SPLIT = pd.Timestamp("2025-04-30", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_SKIP = ("card", "why", "added", "on_from", "prev", "risk_mult", "note")
REPORT = ROOT / "reports" / "agents" / "bot7" / "trades_export.json"
MODS = ("news_regime", "news_straddle", "news_surprise")
STUDIES = (("EV02", "5Min"), ("EV03", "5Min"), ("SH01", "1Min"), ("EV04", "1Min"), ("EV08", "1Min"))


def plans():
    try:
        return json.loads(PLANS_F.read_text(encoding="utf-8")).get("plans") or {}
    except Exception:  # noqa: BLE001
        return {}


def combos():
    try:
        return list(json.loads(PLANS_F.read_text(encoding="utf-8")).get("combos") or [])
    except Exception:  # noqa: BLE001
        return []


def plan_sig(p):
    """What decides a combo's trades in its plan (not the card ids, the reasons or the risk multiplier)."""
    p = p or {}
    keep = {k: v for k, v in p.items() if not (k in _SKIP or k.endswith(("_card", "_why")) or k == "off_from")}
    return json.dumps(keep, sort_keys=True, ensure_ascii=False, default=str)


def built(sid, tf):
    """The live combos of one strategy -> {symbol: plan} (removed ones left out)."""
    pl = plans()
    out = {}
    for c in combos():
        s, t_, sym = (c.split("|") + ["", "", ""])[:3]
        if s == sid and t_ == tf and not (pl.get(c) or {}).get("off_from"):
            out[sym] = pl.get(c) or {}
    return out


def _minutes(ts):
    return ((ts - _EPOCH) // pd.Timedelta(minutes=1)).astype("int64")


def write(sid, tf, z, syms, hold_min, source, variant):
    """z: the built version's rows (sym, t, dir, r and, when known, x) of every symbol in syms -> one exp file each.
    hold_min: {symbol: minutes} or a number - the exit when a row has no "x". Returns {symbol: (2025 n, 2026 n)}."""
    from agent_portfolio import COSTS             # the cost model stamp the whole-bot check requires (stale_source)
    EXP.mkdir(parents=True, exist_ok=True)
    pl = built(sid, tf)
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    out = {}
    for sym in syms:
        y = z[z["sym"] == sym] if len(z) else z
        tr, tr25, approx = [], [], 0
        if len(y):
            t = pd.to_datetime(y["t"], utc=True, format="ISO8601")
            hm = float(hold_min.get(sym, 60) if isinstance(hold_min, dict) else hold_min)
            if "x" in y:
                x = pd.to_datetime(y["x"], utc=True, format="ISO8601", errors="coerce")
            else:
                x = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns, UTC]")
            approx = int(x.isna().sum())
            x = x.fillna(t + pd.Timedelta(minutes=hm))
            x = x.where(x > t, t + pd.Timedelta(minutes=1))
            a, b = _minutes(t), _minutes(x)
            for ti, ai, bi, d, r in zip(t, a, b, y["dir"], y["r"]):
                row = [int(ai), int(bi), int(d), round(float(r), 4)]
                if ti >= SPLIT:
                    tr.append(row)
                elif ti >= TEST_FROM:
                    tr25.append(row)
            tr.sort()
            tr25.sort()
        combo = f"{sid}|{tf}|{sym}"
        doc = {"at": now, "combo": combo, "family": "study", "source": source, "variant": variant,
               "plan_sig": plan_sig(pl.get(sym)), "costs": COSTS, "exit_approx": approx, "no_rows": not len(y),
               "tr": {"built": tr}, "tr25": {"built": tr25}}
        (EXP / f"study_{sid}_{tf}_{sym}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        out[sym] = (len(tr25), len(tr))
    return out


def status():
    """Every live combo of the studies: is the whole-bot check able to use its trades, and if not - why."""
    pl, out = plans(), {}
    for sid, tf in STUDIES:
        for sym, p_ in built(sid, tf).items():
            combo = f"{sid}|{tf}|{sym}"
            try:
                d = json.loads((EXP / f"study_{sid}_{tf}_{sym}.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                out[combo] = {"state": "no file"}
                continue
            ok = d.get("plan_sig") == plan_sig(pl.get(combo) or p_)
            out[combo] = {"state": "ok" if ok else "old plan", "at": d.get("at"), "n25": len((d.get("tr25") or {}).get("built") or []),
                          "n26": len((d.get("tr") or {}).get("built") or []), "no_rows": bool(d.get("no_rows"))}
    return out


def export_all(when="round"):
    """All three studies' exports from their caches, and a report of what is still unknown (the manager's task
    mgr1811b7blind, 18.11.2026: the export sat at the end of a long round, so a round that was stopped left the trades
    of an older plan and the whole-bot check judged bot 7 without 23 of its 57 combos)."""
    import importlib
    mods = {}
    for mod in MODS:
        try:
            m_ = importlib.import_module(f"agents.{mod}")
            df = m_._cached_rows()
            r = safe(m_.export, df)
            mods[mod] = {"cached_rows": int(len(df)), "written": len(r or {})}
        except Exception as e:  # noqa: BLE001 - the export never stops a round
            mods[mod] = {"error": str(e)[-200:]}
    st = status()
    doc = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "when": when, "studies": mods,
           "known": sum(1 for v in st.values() if v["state"] == "ok"),
           "unknown": {k: v["state"] for k, v in st.items() if v["state"] != "ok"},
           "zero_trades": [k for k, v in st.items() if v.get("no_rows")], "combos": st}
    try:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"bot7 trades report failed ({e})")
    print(f"bot7 trades export ({when}): {doc['known']} known, {len(doc['unknown'])} unknown {list(doc['unknown'])[:12]}")
    return doc


def safe(fn, *a, **k):
    """The export never stops a study."""
    try:
        r = fn(*a, **k)
        if r:
            print("bot7 trades for the whole-bot check:", {k_: v for k_, v in (r.items() if isinstance(r, dict) else [])})
        return r
    except Exception as e:  # noqa: BLE001
        print(f"bot7 trades export failed ({str(e)[-200:]})")
        return None


if __name__ == "__main__":
    export_all("by hand")
