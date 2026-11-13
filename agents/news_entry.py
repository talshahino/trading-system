"""Bot 7's entry-timing study (fix attempt 6 for EV01, 13.11.2026).

Why: EV01 (follow the first 10 minutes after a release when they moved 1.5 ATR) makes money from 30.04.2025 to
31.12.2025 and loses in 2026 (11 instruments, GRID 3 of agents/news_fix.py: +7.1R / -27.8R). Five attempts on the
exits, release types, instruments, market filters, the first-pullback entry and the Fibonacci breakeven did not fix
it. What was never varied is WHEN and ON WHAT the entry happens (the user's list: "minutes after the release",
the release's surprise, the spike size):
  window     the move measured over the first 10 / 15 / 20 / 30 minutes (entry at that minute; FTMO: never inside
             the 3 minutes around a restricted release, so 10 is the earliest)
  spike      the move at least 0.75 / 1.5 / 2.5 ATR (5-minute ATR before the release)
  side       with the move (EV01) or against it (a fade) - chosen on 2025 like every other knob, never on 2026
  surprise   all trades / only when the side agrees with actual-vs-forecast for the instrument (FX and gold only:
             an index has no fixed direction for a strong number) / only without a usable surprise
  entry      at once / a limit 0.5R better (the first pullback, valid 3 bars)
  cap        at most 1 / 2 / 3 trades per release, or no cap (attempt 7, 13.11.2026: one PPI = 8 stops)
  target     2R / 3R; stop beyond the window's other end; out after an hour (the live bot's breakeven at 60%)
Costs: FTMO's spread at the entry minute (Dukascopy) and commission (agents/ftmo_costs.py), as agents/news_fix.py.
Fair method: chosen on 30.04.2025-31.12.2025 by total R, shown on 2026, time-weighted (agents/agentkit.py);
robustness: of the 10 best on 2025, how many made money in 2026.

Output: reports/agents/bot7/entry_study.json. Cached per instrument (cache/news_entry/<sym>.json), resumes after a
time budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7, after the fix study.
  python -m agents.news_entry [--minutes 10]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import rules_agent as R  # noqa: E402,F401  (first: rules_agent and event_strategies import each other)
from agents import event_strategies as E  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import poc_agent as P  # noqa: E402
from agents.rules_agent import atr  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "entry_study.json"
CACHE = ROOT / "cache" / "news_entry"
VERSION = 1
WINDOWS = (2, 3, 4, 6)            # 5-minute bars after the release: entry at minute 10 / 15 / 20 / 30
SPIKES = (0.75, 1.5, 2.5)
SIDES = (1, -1)                   # +1 with the move, -1 against it
LIMITS = (0.0, 0.5)
TARGETS = (2.0, 3.0)
SURPRISE = ("all", "agree", "none")
CAPS = (0, 1, 2, 3)               # most trades per release (0 = no cap) - fix attempt 7
TOP = 10
STALE_DAYS = 3


def strat(w, k, side):
    def fn(df):
        o, h, l, c = (df[x].to_numpy() for x in "ohlc")
        a = atr(h, l, c, 14)
        n = len(c)
        long_, short = np.zeros(n, bool), np.zeros(n, bool)
        sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
        for k0 in E._event_bars(df):
            k1 = k0 + w - 1
            if k1 + 1 >= n or k0 < 20 or df.index[k1] - df.index[k0] != pd.Timedelta(minutes=5 * (w - 1)):
                continue
            move, ref = c[k1] - o[k0], a[k0 - 1]
            if not np.isfinite(ref) or ref <= 0 or abs(move) < k * ref:
                continue
            hi, lo = h[k0:k1 + 1].max(), l[k0:k1 + 1].min()
            d = int(np.sign(move)) * side
            if d > 0:
                long_[k1], sl_l[k1] = True, lo if side > 0 else lo - 0.1 * ref
            else:
                short[k1], sl_s[k1] = True, hi if side > 0 else hi + 0.1 * ref
        return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0), "max_bars": 12}
    return fn


def grid():
    return list(itertools.product(WINDOWS, SPIKES, SIDES, LIMITS, TARGETS))


def vkey(w, k, side, lim, tg):
    return f"m{5 * w}|k{k:g}|{'with' if side > 0 else 'fade'}|l{lim:g}|t{tg:g}"


def trades_sym(sym, rel):
    d1 = P.bars(sym, "1Min", NF.LOAD_FROM, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    if d1 is None or d1.empty:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, NF.LOAD_FROM, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    except Exception:  # noqa: BLE001
        sp = None
    d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    d5 = d5.dropna(subset=["o"])
    d5.attrs.update(symbol=sym, tf="5Min")
    evs = E.events_for(sym)
    spec = U.spec(sym)
    px = float(d5["c"].median())
    comm = (px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
            if spec["commissionType"] == "flat_USD" else px * 2 * float(spec["commission"]) / 100.0)
    spm = float(sp.median()) if sp is not None and len(sp) else 0.0
    rows = []
    for w, k, side, lim, tg in grid():
        base = strat(w, k, side)

        def fn(d, base=base, lim=lim, tg=tg):
            L, S, p = base(d)
            p = dict(p, target=("R", float(tg)))
            if lim:
                p["entry_pullback"] = (float(lim), 3)
            return L, S, p
        t = R.run(d5, fn, 0.0, breakeven=(0.6, (spm + comm) / px))
        if t is None or not len(t):
            continue
        t = t[t["reason"] != "סוף הנתונים"]
        if not len(t):
            continue
        net = C.apply(t.reset_index(drop=True), sym, sp)
        for x in net.itertuples():
            tt = pd.Timestamp(x.time)
            if tt < NF.TEST_FROM:
                continue
            g = NF.tag(sym, tt, rel, evs)
            if g is None:
                continue
            agree = None if g["surp"] is None else int(np.sign(g["surp"]) == np.sign(x.dir))
            rows.append({"v": vkey(w, k, side, lim, tg), "sym": sym, "cls": U.CLASS.get(sym, "?"), "t": tt.isoformat(),
                         "dir": int(x.dir), "r": float(x.r), "cat": g["cat"], "agree": agree, "rel": g["rel"].isoformat()})
    return rows


def capped(z, cap):
    """Fix attempt 7 (13.11.2026): at most `cap` trades per release. Live on 13.11 one US PPI release opened 8 EV01
    longs at minute 10 (EURUSD, GBPUSD, AUDUSD, NZDUSD, US30, US100, US500 ...) and all 8 hit the stop together
    (-9.3R): the instruments are one bet on the dollar, not 8. The kept trades are the first by entry time, then by
    the fixed instrument order of the bot (NF.SYMBOLS) - never by result."""
    if not cap or z.empty:
        return z
    order = {s_: i for i, s_ in enumerate(NF.SYMBOLS)}
    z = z.assign(_o=z["sym"].map(order).fillna(99)).sort_values(["rel", "t", "_o"])
    return z[z.groupby("rel").cumcount() < cap].drop(columns="_o")


def study(df):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
    cand = []
    for v, y in df.groupby("v"):
        for sf in SURPRISE:
            z0 = y if sf == "all" else (y[y["agree"] == 1] if sf == "agree" else y[y["agree"].isna()])
            for cap in CAPS:
                z = capped(z0, cap)
                tr, un = z[z.train], z[~z.train]
                cand.append({"v": f"{v}|s{sf}" + (f"|c{cap}" if cap else ""), "train": NF._w(tr), "unseen": NF._w(un)})
    cand.sort(key=lambda c: -c["train"]["sum"])
    top = cand[:TOP]
    base = next((c for c in cand if c["v"] == "m10|k1.5|with|l0|t2|sall"), None)   # EV01 as it trades now
    by = {}
    for key, pat in (("window", r"^m(\d+)\|"), ("spike", r"\|k([\d.]+)\|"), ("side", r"\|(with|fade)\|"),
                     ("surprise", r"\|s([a-z]+)"), ("cap", r"(?:\|c(\d+))?$")):
        g = {}
        for c in cand:
            m = re.search(pat, c["v"])
            if m:
                g.setdefault(m.group(1) or "0", []).append(c)
        # the median variant of each value (not the best: a whole family of variants either holds or not)
        by[key] = {k_: {"variants": len(cs), "train_median_sum": round(float(np.median([c["train"]["sum"] for c in cs])), 2),
                        "unseen_median_sum": round(float(np.median([c["unseen"]["sum"] for c in cs])), 2),
                        "unseen_positive": int(sum(1 for c in cs if (c["unseen"]["sum"] or 0) > 0))}
                   for k_, cs in g.items()}
    return {"baseline": base, "best": cand[0] if cand else None, "top": top,
            "robust": {"top_positive_unseen": int(sum(1 for c in top if (c["unseen"]["sum"] or 0) > 0)), "top": len(top),
                       "top_better_than_base": int(sum(1 for c in top if base and (c["unseen"]["sum"] or 0) > (base["unseen"]["sum"] or 0)))},
            "by": by, "tested": len(cand)}


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
        print("entry study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    rel = NF.releases()
    rows, done = [], []
    CACHE.mkdir(parents=True, exist_ok=True)
    for sym in NF.SYMBOLS:
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
            continue
        try:
            r_ = trades_sym(sym, rel)
            rows += r_
            f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
            done.append(sym)
            print(f"{sym}: {len(rows)} rows so far", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"{sym}: failed ({e})")
    df = pd.DataFrame(rows)
    res = study(df) if len(df) else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "symbols": done, "partial": len(done) < len(NF.SYMBOLS), "rows": int(len(df)),
                               "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()), "study": res},
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res:
        print(json.dumps({"base": res["baseline"], "best": res["best"], "robust": res["robust"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
