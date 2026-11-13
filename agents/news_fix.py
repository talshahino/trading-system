"""Bot 7's fix study (user, 12.11.2026: "first fix and improve the strategies, only then lower the risk or remove").

Why: bot 7's three release strategies (agents/event_strategies.py EV01-EV03) trade every high-impact release on all
16 instruments the same way. Per instrument there are only 5-45 trades, so a fix per combo is noise. Here the trades
of every instrument are pooled and split by what can explain a loss:
  release type   rates / infl / jobs / gdp / oil / other (news_research.category, the release the trade followed)
  instrument     fx / index / metal (agents/ftmo_universe.CLASS)
  own currency   the release's currency is one of the instrument's (else: an FTMO restriction that only touches it)
  surprise       the trade's direction with / against actual-vs-forecast (diagnostic only: FTMO's live 'actual' can be
                 blank, info card 285679fa716f)
and the exits are varied: target (own / 1 / 1.5 / 2 / 3R / none) x stop (x0.5 / x0.75 / x1) x time exit (own / half /
double) x entry (at once / a limit 0.25R or 0.5R better, the first pullback, valid 3 bars - 13.11.2026: exits and
release cells alone did not fix EV01/EV03), with the live bot's management (breakeven at 60% of the way + costs) and FTMO's costs (Dukascopy's spread at
the entry minute - news spreads are wide, and that is in it - and FTMO commission, agents/ftmo_costs.py).

Fair method: chosen on 30.04.2025-31.12.2025 only (the release cells kept = those that made money in that period with
4+ trades; the exit variant = the most total R in that period), shown on 2026 (never seen while choosing),
time-weighted (agents/agentkit.py). Robustness: of the 20 best variants on 2025, how many made money in 2026.
The objective (data/agents/objective.md) is the TOTAL R, so every result carries its sum, not only the mean.

Each instrument's rows are cached (cache/news_fix/<sym>.json, GRID version), so a run cut by its time budget
resumes with the instruments still missing instead of starting over (12.11.2026: 13 of 16 every time).

Output: reports/agents/bot7/fix_study.json (the nightly smart agent turns it into cards). Runs on the bots' computer
inside bot 7's portfolio agent (agent_portfolio.py --bot bot7), at most once every 3 days.
  python -m agents.news_fix [--minutes 20]
"""
from __future__ import annotations

import argparse
import itertools
import json
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

OUT = ROOT / "reports" / "agents" / "bot7" / "fix_study.json"
HIST = ROOT / "data" / "ftmo_calendar" / "history.json"
SYMBOLS = ["US100", "US500", "US30", "GER40", "UK100", "JP225", "XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
           "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY"]          # live/news_bot.SYMBOLS
LOAD_FROM, TEST_FROM, SPLIT = "2025-03-01", pd.Timestamp("2025-04-30", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
STRATS = ("EV01", "EV02", "EV03")
TARGETS = ("own", 1.0, 1.5, 2.0, 3.0, "none")
STOPS = (1.0, 0.75, 0.5)
HOLDS = ("own", 0.5, 2.0)
LIMITS = (0.0, 0.25, 0.5)        # entry on the first pullback: a limit k x risk better than the signal's close, 3 bars
# the user's Fibonacci task (12.11.2026): the breakeven at a Fibonacci share of the way to the target instead of the
# live bot's 60% (0 = no breakeven); with the own stop / time exit and the at-once entry, on these targets (GRID 3)
FIB_BE = (0.0, 0.382, 0.5, 0.618, 0.786)
FIB_TARGETS = ("own", 1.5, 2.0, 3.0)
GRID = 3                         # cache version: bump when the variants or the costs change
EXITS_TOP = 10                   # robustness of the exits-only pick: of its 10 best on 2025, how many made money in 2026
CACHE = ROOT / "cache" / "news_fix"
PRIO = ("rates", "infl", "jobs", "gdp", "oil", "other")
MIN_CELL_N = 4
STALE_DAYS = 3


def category(title):
    t = title.lower()
    if "oil" in t:
        return "oil"
    if any(k in t for k in ("rate", "fomc", "monetary", "policy", "mpc", "statement", "minutes", "inflation letter")):
        return "rates"
    if any(k in t for k in ("cpi", "ppi", "rpi")):
        return "infl"
    if any(k in t for k in ("employment", "unemployment", "payroll", "claims", "labor")):
        return "jobs"
    if "gdp" in t:
        return "gdp"
    return "other"


def _num(x):
    import re
    if x in (None, ""):
        return None
    m = re.search(r"-?\d+(\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None


def releases():
    """Release time -> [(currency, category, surprise +1/-1/None, title)] of the important ones."""
    out = {}
    for i in json.loads(HIST.read_text(encoding="utf-8")):
        if i.get("impact") != "high" and not i.get("restriction"):
            continue
        ins = str(i.get("instrument", ""))
        cur = "Crude Oil" if ins.startswith("Crude") else (ins.split(" ")[0] if ins else "")
        a, f = _num(i.get("actual")), _num(i.get("forecast"))
        s = None
        if a is not None and f is not None and a != f:
            s = 1 if a > f else -1
            tl = i["title"].lower()
            if "unemployment" in tl or "claims" in tl or category(i["title"]) == "oil":
                s = -s
        out.setdefault(pd.Timestamp(i["date"]).tz_convert("UTC"), []).append((cur, category(i["title"]), s, i["title"]))
    return out


def tag(sym, t, rel, evs):
    """The release a trade followed (the instrument's latest release before the entry, within 3 hours)."""
    ts = [x for x in evs if x <= t and t - x <= pd.Timedelta(hours=3)]
    if not ts:
        return None
    k = max(ts)
    items = rel.get(k, [])
    mine = [x for x in items if x[0] in U.CURRENCIES.get(sym, [])]
    use = mine or items
    if not use:
        return {"cat": "other", "own": False, "surp": None, "cur": "", "title": "", "rel": k}
    best = min(use, key=lambda x: PRIO.index(x[1]))
    # the surprise in the instrument's direction: + when the instrument should rise
    s = best[2]
    if s is not None:
        curs = U.CURRENCIES.get(sym, [])
        if U.CLASS.get(sym) == "fx" and len(curs) == 2 and best[0] == curs[1]:
            s = -s                                     # the quote currency strengthened: the pair falls
        elif U.CLASS.get(sym) in ("metal",) and best[0] == "USD":
            s = -s
        elif U.CLASS.get(sym) == "index":
            s = None                                   # a strong number can lift or sink an index
    return {"cat": best[1], "own": bool(mine), "surp": s, "cur": best[0], "title": best[3], "rel": k}


def variant_fn(fn, tg, st, hd, lim=0.0, bw=None):
    def g(d):
        L, S, plan = fn(d)
        p = dict(plan)
        if tg == "none":
            p["target"] = None
        elif tg != "own":
            p["target"] = ("R", float(tg))
        if st != 1.0:
            p["stop_scale"] = float(st)
        if hd != "own":
            p["max_bars"] = max(2, int(round((plan.get("max_bars") or 12) * float(hd))))
        if lim:
            p["entry_pullback"] = (float(lim), 3)
        if bw is not None:
            p["be_way"] = float(bw)               # rules_agent.run: the breakeven's own point of the way (0 = none)
        return L, S, p
    return g


def vkey(tg, st, hd, lim=0.0, bw=None):
    return f"t{tg}|s{st:g}|h{hd}" + (f"|l{lim:g}" if lim else "") + (f"|b{bw:g}" if bw is not None else "")


def grid():
    """54 exit variants at once, the same targets x stops with the two limit entries (own time exit): 90; and the
    Fibonacci breakeven points on 4 targets (own stop / time exit, at-once entry): 20 more - 110."""
    g = [(tg, st, hd, 0.0, None) for tg, st, hd in itertools.product(TARGETS, STOPS, HOLDS)]
    g += [(tg, st, "own", lim, None) for lim in LIMITS if lim for tg, st in itertools.product(TARGETS, STOPS)]
    return g + [(tg, 1.0, "own", 0.0, bw) for tg, bw in itertools.product(FIB_TARGETS, FIB_BE)]


def session(t):
    """The release's session by its UTC hour: asia (<7), europe (7-12), us (12+)."""
    h = pd.Timestamp(t).hour
    return "asia" if h < 7 else ("europe" if h < 12 else "us")


def trades_sym(sym, rel):
    d1 = P.bars(sym, "1Min", LOAD_FROM, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    if d1 is None or d1.empty:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, LOAD_FROM, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    except Exception:  # noqa: BLE001
        sp = None
    d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    d5 = d5.dropna(subset=["o"])
    d5.attrs.update(symbol=sym, tf="5Min")
    evs = E.events_for(sym)
    spec = U.spec(sym)
    rows = []
    for sid in STRATS:
        base = E.EVENTS[sid][1]
        for tg, st, hd, lim, bw in grid():
            fn = variant_fn(base, tg, st, hd, lim, bw)
            px = float(d5["c"].median())
            comm = (px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
                    if spec["commissionType"] == "flat_USD" else px * 2 * float(spec["commission"]) / 100.0)
            spm = float(sp.median()) if sp is not None and len(sp) else 0.0
            t = R.run(d5, fn, 0.0, breakeven=(0.6, (spm + comm) / px))
            if t is None or not len(t):
                continue
            t = t[t["reason"] != "סוף הנתונים"]
            if not len(t):
                continue
            net = C.apply(t.reset_index(drop=True), sym, sp)
            for x in net.itertuples():
                tt = pd.Timestamp(x.time)
                if tt < TEST_FROM:
                    continue
                g = tag(sym, tt, rel, evs)
                if g is None:
                    continue
                agree = None if g["surp"] is None else int(np.sign(g["surp"]) == np.sign(x.dir))
                rows.append({"sid": sid, "v": vkey(tg, st, hd, lim, bw), "sym": sym, "cls": U.CLASS.get(sym, "?"), "t": tt.isoformat(),
                             "dir": int(x.dir), "r": float(x.r), "cat": g["cat"], "own": g["own"], "agree": agree,
                             "rel": g["rel"].isoformat(), "title": g["title"]})
    return rows


def _w(df):
    if df.empty:
        return {"n": 0, "eff_n": 0.0, "mean": None, "t": None, "win": None, "sum": 0.0}
    s = K.wstats(df["r"].to_numpy(), K.weights(df["t"]))
    s["sum"] = round(float(df["r"].sum()), 2)
    return s


def cells_of(df):
    return list(zip(df["cat"], df["cls"]))


def pick_cells(train):
    g = train.groupby(["cat", "cls"])["r"].agg(["sum", "count"])
    return sorted([k for k, v in g.iterrows() if v["count"] >= MIN_CELL_N and v["sum"] > 0])


def keep(df, cells):
    s = set(map(tuple, cells))
    return df[[c in s for c in cells_of(df)]] if len(df) else df


def study(df):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < SPLIT)
    out = {}
    base_v = vkey("own", 1.0, "own")
    for sid in STRATS:
        x = df[df["sid"] == sid]
        if x.empty:
            continue
        b = x[x["v"] == base_v]
        res = {"baseline": {"train": _w(b[b.train]), "unseen": _w(b[~b.train])}}
        # where it loses: release type x instrument class, own currency, surprise agreement (as it trades now)
        tab = {}
        for keys, y in b.groupby(["cat", "cls"]):
            tab["|".join(keys)] = {"train": _w(y[y.train]), "unseen": _w(y[~y.train])}
        res["by_cell"] = tab
        res["by_own"] = {str(k): {"train": _w(y[y.train]), "unseen": _w(y[~y.train])} for k, y in b.groupby("own")}
        res["by_surprise"] = {str(k): {"train": _w(y[y.train]), "unseen": _w(y[~y.train])}
                              for k, y in b.assign(agree=b["agree"].fillna(-1)).groupby("agree")}
        res["by_session"] = {k: {"train": _w(y[y.train]), "unseen": _w(y[~y.train])}
                             for k, y in b.assign(ses=b["rel"].map(session)).groupby("ses")}
        lim = x.assign(lim=x["v"].str.extract(r"\|l([\d.]+)$")[0].fillna("0"))
        res["by_limit"] = {k: {"train": _w(y[y.train & (y["v"].str.startswith("town|s1|"))]),
                               "unseen": _w(y[~y.train & (y["v"].str.startswith("town|s1|"))])}
                           for k, y in lim[lim["v"].str.match(r"^town\|s1\|hown(\|l[\d.]+)?$")].groupby("lim")}
        # the Fibonacci breakeven on the own target (the live bot: 60% of the way)
        fib = x[x["v"].str.match(r"^town\|s1\|hown\|b[\d.]+$")]
        res["by_fib_be"] = {k: {"train": _w(y[y.train]), "unseen": _w(y[~y.train])}
                            for k, y in fib.assign(bw=fib["v"].str.extract(r"\|b([\d.]+)$")[0]).groupby("bw")}
        res["by_symbol"] = {k: {"train": _w(y[y.train]), "unseen": _w(y[~y.train])} for k, y in b.groupby("sym")}
        cand = []
        for v, y in x.groupby("v"):
            tr, un = y[y.train], y[~y.train]
            cells = pick_cells(tr)
            trk, unk = keep(tr, cells), keep(un, cells)
            cand.append({"v": v, "cells": [list(c) for c in cells], "train_all": _w(tr), "unseen_all": _w(un),
                         "train": _w(trk), "unseen": _w(unk)})
        by_all = sorted(cand, key=lambda c: -c["train_all"]["sum"])
        by_cells = sorted(cand, key=lambda c: -c["train"]["sum"])
        bc = next(c for c in cand if c["v"] == base_v) if any(c["v"] == base_v for c in cand) else None
        res["exits_only"] = {"v": by_all[0]["v"], "train": by_all[0]["train_all"], "unseen": by_all[0]["unseen_all"]}
        # is the exits-only pick a lone lucky point? its 10 best neighbours on 2025, shown on 2026 (13.11.2026: EV03's
        # pick, limit 0.5R + 3R, turned -38R into +0.2R in 2026 - kept only if its neighbours hold too)
        et = by_all[:EXITS_TOP]
        res["exits_top"] = [{"v": c["v"], "train_sum": c["train_all"]["sum"], "train_n": c["train_all"]["n"],
                             "unseen_sum": c["unseen_all"]["sum"], "unseen_n": c["unseen_all"]["n"]} for c in et]
        res["exits_robust"] = {"top_positive_unseen": int(sum(1 for c in et if (c["unseen_all"]["sum"] or 0) > 0)),
                               "top": len(et), "base_unseen_sum": res["baseline"]["unseen"]["sum"],
                               "top_better_than_base": int(sum(1 for c in et if (c["unseen_all"]["sum"] or 0)
                                                               > (res["baseline"]["unseen"]["sum"] or 0)))}
        if bc:
            res["cells_only"] = {"v": base_v, "cells": bc["cells"], "train": bc["train"], "unseen": bc["unseen"]}
        res["best"] = {k: by_cells[0][k] for k in ("v", "cells", "train", "unseen")}
        top = by_cells[:20]
        res["robust"] = {"top20_positive_unseen": int(sum(1 for c in top if (c["unseen"]["sum"] or 0) > 0)),
                         "top20_mean_unseen_sum": round(float(np.mean([c["unseen"]["sum"] for c in top])), 2)}
        res["tested"] = len(cand)
        out[sid] = res
    return out


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS) or j.get("partial")
                or j.get("grid") != GRID)
    except Exception:  # noqa: BLE001
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.force and not stale():
        print("fix study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    rel = releases()
    rows, done = [], []
    CACHE.mkdir(parents=True, exist_ok=True)
    for sym in SYMBOLS:
        f = CACHE / f"{sym}.json"
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
            if c.get("grid") == GRID and pd.Timestamp.now(tz="UTC") - pd.Timestamp(c["at"]) < pd.Timedelta(days=STALE_DAYS):
                rows += c["rows"]
                done.append(sym)
                print(f"{sym}: from the cache ({len(c['rows'])} rows)", flush=True)
                continue
        except Exception:  # noqa: BLE001
            pass
        if time.time() > t_end and done:
            print(f"time budget reached after {done}")
            continue
        try:
            r_ = trades_sym(sym, rel)
            rows += r_
            f.write_text(json.dumps({"grid": GRID, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
            done.append(sym)
            print(f"{sym}: {len(rows)} rows so far", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"{sym}: failed ({e})")
    df = pd.DataFrame(rows)
    res = study(df) if len(df) else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    partial = len(done) < len(SYMBOLS)
    doc = json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "grid": GRID, "symbols": done,
                      "partial": partial, "rows": int(len(df)), "split": str(SPLIT.date()),
                      "test_from": str(TEST_FROM.date()), "study": res}, ensure_ascii=False, indent=1, default=str)
    # a partial run of a new grid never overwrites the last complete study (13.11.2026: the cards are made from it);
    # it is kept beside it, and the next runs go on from the cache until every instrument is in
    try:
        prev_complete = not json.loads(OUT.read_text(encoding="utf-8")).get("partial")
    except Exception:  # noqa: BLE001
        prev_complete = False
    (OUT.with_name("fix_study_partial.json") if partial and prev_complete else OUT).write_text(doc, encoding="utf-8")
    print(json.dumps({s: {"base_unseen": r["baseline"]["unseen"]["sum"], "best_unseen": r["best"]["unseen"]["sum"]}
                      for s, r in res.items()}))


if __name__ == "__main__":
    main()
