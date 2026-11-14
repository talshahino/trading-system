"""Bot 7's market-regime filters on its release and shock strategies (the user's task tal1411indstrat, 14.11.2026:
"bot 7: only filters on EV / SH - volatility regime, ADX, Choppiness - not new strategies from indicators").

Why pooled: a release strategy has 2-15 trades a year per instrument, so a filter chosen per instrument is noise
(EV03 GBPJPY 12.11.2026). Here each filter is summed over all 16 instruments of a strategy (EV02, EV03), and for the
shock fade SH01 per instrument (US100, GER40 - the strategy's own instruments, 50-80 trades a year each).

What is tested - exactly what the live bot trades now (live/bots/bot7.json plans: EV03's 2R target + 0.5R limit entry,
SH01's K / D / H), with the live engine (agents/rules_agent.run, breakeven at 60% of the way + costs as
live/news_bot.trades_of) and FTMO's costs (Dukascopy's spread at the entry minute + commission, agents/ftmo_costs),
with one filter at a time on the strategy's own candles (5 minutes for EV, 1 minute for SH01) - the masks are
live/news_bot.regime itself, so what is chosen here is what the live bot would apply:
  none, with_trend200, adx_trend / adx_range (ADX14 above / below 20), vol_normal / vol_high / vol_low (ATR14 % rank of
  500 bars), ema9_20, bbw_p20, rsi_trend, chop_trend / chop_range (Choppiness Index 14 below / from 50, new here)
Fair method: chosen on 30.04-31.12.2025 only (most total R with >= MIN_N trades, and above the unfiltered total), shown
on 2026; time-weighted (agents/agentkit.py). The bar corrected for the number of filters tried: the pick's 2025 t must
reach sqrt(2 ln m) (the expected best of m tries by chance), m = filters tried. Robustness: of the filters better than
none in 2025, how many are better in 2026 too.

Per-instrument rows are cached (cache/news_regime/<sym>.json, VERSION), so a run cut by its time budget resumes.
Output: reports/agents/bot7/regime_study.json (the nightly smart agent turns it into cards). Runs on the bots'
computer inside bot 7's portfolio agent (agent_portfolio.py --bot bot7), at most once every STALE_DAYS days.
  python -m agents.news_regime [--minutes 10] [--force]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import rules_agent as R  # noqa: E402,F401  (first: rules_agent and event_strategies import each other)
from agents import agentkit as K  # noqa: E402
from agents import event_strategies as E  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import poc_agent as P  # noqa: E402
from agents import shocks as SH  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "regime_study.json"
CACHE = ROOT / "cache" / "news_regime"
PLANS_F = ROOT / "live" / "bots" / "bot7.json"
EV_SYMBOLS = ["US100", "US500", "US30", "GER40", "UK100", "JP225", "XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
              "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY"]          # live/news_bot.SYMBOLS
EV_STRATS = ("EV02", "EV03")                                             # EV01 is off since 14.11.2026
SH_SYMBOLS = ("GER40", "US100")                                          # SH01's instruments (GER40 loses in 2026)
FILTERS = ("none", "with_trend200", "adx_trend", "adx_range", "vol_normal", "vol_high", "vol_low", "ema9_20", "bbw_p20",
           "rsi_trend", "chop_trend", "chop_range")
LOAD_FROM, TEST_FROM, SPLIT = "2025-03-01", pd.Timestamp("2025-04-30", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
VERSION = 2                 # 2: SH01 with the plan's p / side / US100 check (card bd7be6c8ae8a, 14.11.2026)
STALE_DAYS = 3
MIN_N = {"EV02": 8, "EV03": 20, "SH01": 20}      # trades on 2025 a filter must keep to be picked


def _plans():
    try:
        return json.loads(PLANS_F.read_text(encoding="utf-8")).get("plans") or {}
    except Exception:  # noqa: BLE001
        return {}


def _masked(base, al, ash):
    L, S, plan = base

    def g(_d):
        return np.asarray(L, bool) & al, np.asarray(S, bool) & ash, plan
    return g


def _run_filters(d, sym, tf, sid, fn0, p_, sp):
    """Every filter on one strategy x instrument -> rows {sid, sym, f, t, r}."""
    from live import news_bot as NB                    # the live masks and plan keys (same code as the bot)
    pk = NB.plan_keys(p_)
    L, S, plan = fn0(d)                                 # the signals once (SH01's detect() is the slow part)
    base = (L, S, {**plan, **pk} if pk else plan)
    px = float(d["c"].median())
    spec = U.spec(sym)
    comm = (px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
            if spec["commissionType"] == "flat_USD" else px * 2 * float(spec["commission"]) / 100.0)
    spm = float(sp.median()) if sp is not None and len(sp) else 0.0
    fe, ne = NB.break_exit(d, tf)                       # no position through a 2h+ market break (as live)
    rows = []
    for f in FILTERS:
        if f == "none":
            al = ash = np.ones(len(d), bool)
        else:
            al, ash = NB.regime(d, f)
            al, ash = np.asarray(al, bool), np.asarray(ash, bool)
        if ne.any():
            al, ash = al & ~ne, ash & ~ne
        t = R.run(d, _masked(base, al, ash), 0.0, breakeven=(0.6, (spm + comm) / px),
                  session_exit=fe if fe.any() else None)
        if t is None or not len(t):
            continue
        t = t[t["reason"] != "סוף הנתונים"]
        if not len(t):
            continue
        net = C.apply(t.reset_index(drop=True), sym, sp)
        for x in net.itertuples():
            tt = pd.Timestamp(x.time)
            tt = tt.tz_localize("UTC") if tt.tzinfo is None else tt
            if tt < TEST_FROM:
                continue
            rows.append({"sid": sid, "sym": sym, "f": f, "t": tt.isoformat(), "dir": int(x.dir), "r": round(float(x.r), 4)})
    return rows


def rows_sym(sym, plans, rel_sh):
    end = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = P.bars(sym, "1Min", LOAD_FROM, end)
    if d1 is None or d1.empty:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, LOAD_FROM, end)
    except Exception:  # noqa: BLE001
        sp = None
    rows = []
    if sym in EV_SYMBOLS:
        d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
        d5 = d5.dropna(subset=["o"])
        d5.attrs.update(symbol=sym, tf="5Min")
        E.events_for(sym)
        for sid in EV_STRATS:
            p_ = plans.get(f"{sid}|5Min|{sym}") or {}
            if p_.get("off_from"):
                continue
            rows += _run_filters(d5, sym, "5Min", sid, E.EVENTS[sid][1], p_, sp)
    if sym in SH_SYMBOLS:
        p_ = plans.get(f"SH01|1Min|{sym}") or {}
        try:
            k, dd, hh, pp = int(p_.get("K", 8)), int(p_.get("D", 1)), int(p_.get("H", 60)), float(p_.get("p", 0.25))
        except (TypeError, ValueError):
            k, dd, hh, pp = 8, 1, 60, 0.25
        cs = p_.get("co_sym")                       # SH01 GER40 since card bd7be6c8ae8a: only when US100 had no shock
        co_d = None
        if cs:
            co_d = P.bars(cs, "1Min", LOAD_FROM, end)
            co_d = co_d[~co_d.index.duplicated()].sort_index() if co_d is not None and len(co_d) else None
        d1.attrs.update(symbol=sym, tf="1Min")
        rows += _run_filters(d1, sym, "1Min", "SH01", SH.live_fade(sym, k, dd, hh, rel=rel_sh, p=pp,
                                                                   side=p_.get("side", "all"), co=p_.get("co", "all"),
                                                                   co_sym=cs, co_df=co_d), p_, sp)
    return rows


def _w(df):
    if df.empty:
        return {"n": 0, "eff_n": 0.0, "mean": None, "t": None, "win": None, "sum": 0.0}
    s = K.wstats(df["r"].to_numpy(), K.weights(df["t"]))
    s["sum"] = round(float(df["r"].sum()), 2)
    return s


def study_group(x, min_n):
    x = x.assign(train=pd.to_datetime(x["t"], utc=True, format="ISO8601") < SPLIT)
    res = {f: {"train": _w(y[y.train]), "unseen": _w(y[~y.train])} for f, y in x.groupby("f")}
    base = res.get("none")
    if not base:
        return {"filters": res}
    tried = [f for f in res if f != "none"]
    bar = round(math.sqrt(2 * math.log(max(2, len(tried)))), 2)
    ok = [f for f in tried if res[f]["train"]["n"] >= min_n and res[f]["train"]["sum"] > base["train"]["sum"]]
    pick = max(ok, key=lambda f: res[f]["train"]["sum"]) if ok else None
    better_tr = [f for f in tried if res[f]["train"]["sum"] > base["train"]["sum"]]
    better_both = [f for f in better_tr if res[f]["unseen"]["sum"] > base["unseen"]["sum"]]
    out = {"filters": dict(sorted(res.items(), key=lambda kv: -kv[1]["train"]["sum"])), "none": base,
           "tried": len(tried), "t_bar": bar, "pick": pick,
           "robust": {"better_on_2025": len(better_tr), "also_better_2026": len(better_both), "which": better_both}}
    if pick:
        p = res[pick]
        out["pick_stats"] = {**p, "passes_bar": bool((p["train"]["t"] or 0) >= bar),
                             "gain_2026": round(p["unseen"]["sum"] - base["unseen"]["sum"], 2)}
    return out


def study(df):
    out = {}
    for sid in EV_STRATS:
        x = df[df["sid"] == sid]
        if len(x):
            out[sid] = study_group(x, MIN_N[sid])
            out[sid]["by_symbol_none"] = {s: {"train": _w(y[y["t"] < SPLIT.isoformat()]), "unseen": _w(y[y["t"] >= SPLIT.isoformat()])}
                                          for s, y in x[x["f"] == "none"].groupby("sym")}
    for sym in SH_SYMBOLS:
        x = df[(df["sid"] == "SH01") & (df["sym"] == sym)]
        if len(x):
            out[f"SH01|{sym}"] = study_group(x, MIN_N["SH01"])
    return out


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS) or j.get("partial")
                or j.get("version") != VERSION)
    except Exception:  # noqa: BLE001
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.force and not stale():
        print("regime study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    plans = _plans()
    rel = SH.releases()
    syms = list(SH_SYMBOLS) + [s for s in EV_SYMBOLS if s not in SH_SYMBOLS]     # SH01 GER40 (a loser) first
    rows, done = [], []
    CACHE.mkdir(parents=True, exist_ok=True)
    for sym in syms:
        f = CACHE / f"{sym}.json"
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
            if c.get("version") == VERSION and pd.Timestamp.now(tz="UTC") - pd.Timestamp(c["at"]) < pd.Timedelta(days=STALE_DAYS):
                rows += c["rows"]
                done.append(sym)
                continue
        except Exception:  # noqa: BLE001
            pass
        if time.time() > t_end and done:
            print(f"time budget reached after {done}")
            break
        ts = time.time()
        try:
            r_ = rows_sym(sym, plans, rel)
            rows += r_
            f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}),
                         encoding="utf-8")
            done.append(sym)
            print(f"{sym}: {len(r_)} rows ({time.time() - ts:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"{sym}: failed ({str(e)[-200:]})")
    df = pd.DataFrame(rows)
    res = study(df) if len(df) else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "symbols": done, "partial": len(done) < len(syms), "rows": int(len(df)),
                               "split": str(SPLIT.date()), "test_from": str(TEST_FROM.date()), "filters": list(FILTERS),
                               "study": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({g: {"pick": r.get("pick"), "none_2026": (r.get("none") or {}).get("unseen", {}).get("sum"),
                          "pick_2026": (r.get("pick_stats") or {}).get("unseen", {}).get("sum")} for g, r in res.items()}))


if __name__ == "__main__":
    main()
