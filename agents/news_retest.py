"""EV06 - re-entries on retests of the broken levels after a release (the user's task tal1411fade, 14.11.2026 09:17:
TikTok sphula, gold on the CPI, 15 minutes: during the release's fall he sold three times, each time on the retest of a
level that had just broken - a staircase - with one shared target), and EV07 - absorption in the minutes after a release
(the user's task tal1411of, 14.11.2026 09:29, Orderflow Sniper: "bot 7: absorption in the minutes after a release").

EV06, the staircase (the move's direction, never against it):
  move    the first W minutes after the release (W = 5 / 10 / 15): the close of minute W against the release minute's
          open, at least M x the 5-minute ATR(14) before the release (M = 1 / 1.5 / 2.5); its sign is the trade side
  level   the first level is the extreme of those W minutes (the low of a fall). When the price breaks it, the broken
          level waits for a retest: a limit order T x ATR on the near side of it (T = 0.1 / 0.25) - a sell limit just
          under a broken low. Filled -> one entry, its own stop X x ATR beyond the level (X = 0.5 / 1.0). The next level
          is the new extreme made between that break and the retest - the next step of the staircase.
  entries at most N per release (N = 1 / 2 / 3), each sized 1/N of the bot's risk - the whole release never risks more
          than 1R (the user: "total risk at most 1R of the bot for the whole release"); retests only until minute 60
  target  one shared target for all the entries: leg - the first move once more beyond the first level; r2 - 2R per
          entry; none - only the time exit
  hold    everything out at minute 60 / 120 after the release
EV07, absorption (against the move):
  bar     in minutes 0-14 after a move of at least 1 ATR, a 1-minute bar with tick volume >= V x the average of the
          20 bars before the release (V = 2 / 3), a range <= R x the 5-minute ATR (R = 0.5 / 1.0), its far end within
          0.1 ATR of the move's extreme so far = someone absorbs the move (the first such bar of the release only)
  confirm the next bar closes back against the move; entry at the open of the bar after (start: never before minute 3)
  stop    Y x ATR beyond the absorption bar's far end (Y = 0.25 / 0.5)
  target  pre - the price before the release; r2 - 2R; hold 30 / 60 minutes
  no order book here: tick volume from the 1-minute bars stands in for the real flow (said in every card).
Costs: FTMO's spread at the fill minute + commission (agents/ftmo_costs.py) for every entry on its own risk; a limit fill
bar that also touches the stop counts as stopped (the order inside a 1-minute candle is unknown); fixed risk, never
sized up, no martingale. Fair method as the other bot 7 studies: chosen on 30.04-31.12.2025 by total R, shown on 2026;
robustness = of the 10 best on 2025, how many made money in 2026; at most 1 / 2 / 3 releases-trades over all instruments
(EV01's lesson: one PPI = 8 stops on one bet); also by release type, instrument and the move with / against the surprise.

Output: reports/agents/bot7/retest_study.json. Cached per instrument (cache/news_retest/<sym>.json), resumes after a
time budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7, once the straddle and fade studies
are complete (it takes their slot - the round's time does not grow).
  python -m agents.news_retest [--minutes 10] [--force] [--symbols XAUUSD]
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
from agents import event_strategies as E  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import news_straddle as NS  # noqa: E402
from agents import poc_agent as P  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "retest_study.json"
CACHE = ROOT / "cache" / "news_retest"
VERSION = 1
# EV06
WINS = (5, 10, 15)
MULTS = (1.0, 1.5, 2.5)
TOLS = (0.1, 0.25)
XSTOP = (0.5, 1.0)
NMAX = (1, 2, 3)
TARGETS = ("leg", "r2", "none")
HOLDS = (60, 120)
LAST_RETEST = 60
# EV07
VOLS = (2.0, 3.0)
RANGES = (0.5, 1.0)
YSTOP = (0.25, 0.5)
A_TARGETS = ("pre", "r2")
A_HOLDS = (30, 60)
A_LAST = 15
A_START = 3
CAPS = (0, 1, 2, 3)
TOP = 10
STALE_DAYS = 3
MIN_N = 8
SPAN = 125
BASE6 = "w10|m1.5|t0.1|x0.5|n3|leg|h60"     # the video: three entries on the retests, one shared target
BASE7 = "v2|q0.5|y0.25|pre|h30"


def k6(w, m, t, x, n, tg, hd):
    return f"w{w}|m{m:g}|t{t:g}|x{x:g}|n{n}|{tg}|h{hd}"


def k7(v, q, y, tg, hd):
    return f"v{v:g}|q{q:g}|y{y:g}|{tg}|h{hd}"


def _walk(o, h, l, c, q0, end, dr, e, stop, tgt):
    """1-minute bars from the fill bar q0 (stop only in it), stop before target -> (exit price, exit bar)."""
    return NS._exit(o, h, l, c, q0, end, dr, e, stop, tgt, None)


def stairs(o, h, l, c, k0, w, dr, atr, tol, x, n):
    """The staircase's fills: [(fill bar, entry price, stop)] - at most n."""
    end_r = k0 + LAST_RETEST
    seg = slice(k0, k0 + w)
    level = float(l[seg].min()) if dr < 0 else float(h[seg].max())
    out, q, broken, ext = [], k0 + w, False, None
    while q <= end_r and len(out) < n:
        if not broken:
            if (l[q] < level) if dr < 0 else (h[q] > level):
                broken, ext = True, (l[q] if dr < 0 else h[q])
            q += 1
            continue
        # a sell limit just under a broken low (dr < 0) / a buy limit just over a broken high (dr > 0)
        lim = level - tol * atr if dr < 0 else level + tol * atr
        hit = (h[q] >= lim) if dr < 0 else (l[q] <= lim)
        if hit:
            e = max(lim, o[q]) if dr < 0 else min(lim, o[q])     # a gap through the limit fills at the open (better)
            stop = level + x * atr if dr < 0 else level - x * atr
            if dr * (e - stop) < 0:                              # a sane order: the stop on the far side of the entry
                out.append((q, float(e), float(stop)))
            level, broken = float(ext), False                    # the next step: the extreme made since the break
            q += 1
            continue
        ext = min(ext, l[q]) if dr < 0 else max(ext, h[q])
        q += 1
    return out


def trades_sym(sym, rel):
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = P.bars(sym, "1Min", NF.LOAD_FROM, now)
    if d1 is None or d1.empty:
        return [], []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, NF.LOAD_FROM, now)
        sp = sp[~sp.index.duplicated()].sort_index() if sp is not None and len(sp) else None
    except Exception:  # noqa: BLE001
        sp = None
    idx = d1.index
    o, h, l, c = (d1[x].to_numpy(float) for x in "ohlc")
    v = d1["v"].to_numpy(float) if "v" in d1 else np.zeros(len(d1))
    a5 = NS._atr5(d1)
    cur = set(U.CURRENCIES.get(sym, []))
    own = sorted(t for t in set(E.events_for(sym)) | {t for t, it in rel.items() if any(x_[0] in cur for x_ in it)}
                 if t >= NF.TEST_FROM)
    raw6, raw7 = [], []
    for t in own:
        k0 = idx.searchsorted(t)
        if k0 >= len(idx) or idx[k0] != t or k0 < 30 or k0 + SPAN >= len(idx) or idx[k0 + SPAN] - t != pd.Timedelta(minutes=SPAN):
            continue
        ka = a5.index.searchsorted(t.floor("5min")) - 1
        if ka < 14:
            continue
        atr = float(a5.iloc[ka])
        if not np.isfinite(atr) or atr <= 0:
            continue
        g = NF.tag(sym, t + pd.Timedelta(minutes=1), rel, [t]) or {}
        surp = g.get("surp")
        ref = o[k0]
        base = {"rel": t, "cat": g.get("cat", "other")}
        # ---- EV06: the staircase, the move's direction
        for w in WINS:
            mv = c[k0 + w - 1] - ref
            if abs(mv) < min(MULTS) * atr:
                continue
            dr = 1 if mv > 0 else -1
            mcls = next(m for m in sorted(MULTS, reverse=True) if abs(mv) >= m * atr)
            side = "none" if surp is None or surp == 0 else ("with" if np.sign(surp) == dr else "against")
            seg = slice(k0, k0 + w)
            lvl0 = float(l[seg].min()) if dr < 0 else float(h[seg].max())
            leg_t = lvl0 + dr * abs(lvl0 - ref)                   # the first move once more beyond the first level
            for tol, x in itertools.product(TOLS, XSTOP):
                fills = stairs(o, h, l, c, k0, w, dr, atr, tol, x, max(NMAX))
                if not fills:
                    continue
                for n, tg, hd in itertools.product(NMAX, TARGETS, HOLDS):
                    end = min(k0 + hd, len(idx))
                    for j, (q0, e, stop) in enumerate(fills[:n]):
                        if q0 >= end:
                            continue
                        risk = abs(e - stop)
                        tgt = leg_t if tg == "leg" else (e + dr * 2.0 * risk if tg == "r2" else None)
                        if tgt is not None and dr * (tgt - e) <= 0:
                            tgt = None if tg == "leg" else tgt            # the shared target already passed: time exit
                        px, qx = _walk(o, h, l, c, q0, end, dr, e, stop, tgt)
                        raw6.append({**base, "v": k6(w, mcls, tol, x, n, tg, hd), "time": idx[q0], "exit_time": idx[qx],
                                     "dir": dr, "entry_px": e, "stop_px": stop, "r": dr * (px - e) / risk, "wt": 1.0 / n,
                                     "side": side, "leg": j, "min": int(q0 - k0)})
        # ---- EV07: absorption at the move's extreme, against it
        pre = slice(k0 - 20, k0)
        v_avg = float(np.mean(v[pre])) if v[pre].size else 0.0
        if v_avg <= 0:
            continue
        hi, lo, taken = h[k0], l[k0], set()
        for q in range(k0, k0 + A_LAST):
            hi, lo = max(hi, h[q]), min(lo, l[q])
            mdir = 1 if (hi - ref) >= (ref - lo) else -1                # the move so far
            far = h[q] if mdir > 0 else l[q]
            ext = hi if mdir > 0 else lo
            if abs(ext - ref) < 1.0 * atr or abs(far - ext) > 0.1 * atr:
                continue
            ratio, rng = v[q] / v_avg, (h[q] - l[q]) / atr
            qc, qe = q + 1, q + 2
            if qe - k0 < A_START or qe >= len(idx):
                continue
            if not ((c[qc] < o[qc]) if mdir > 0 else (c[qc] > o[qc])):  # the confirmation closes against the move
                continue
            dr = -mdir
            e = o[qe]
            side = "none" if surp is None or surp == 0 else ("with" if np.sign(surp) == dr else "against")
            for vv, rr in itertools.product(VOLS, RANGES):
                if (vv, rr) in taken or ratio < vv or rng > rr:
                    continue
                taken.add((vv, rr))                                     # each variant: its first absorption bar only
                for y, tg, hd in itertools.product(YSTOP, A_TARGETS, A_HOLDS):
                    stop = far + mdir * y * atr
                    risk = dr * (e - stop)
                    if risk <= 0:
                        continue
                    tgt = ref if tg == "pre" else e + dr * 2.0 * risk
                    if dr * (tgt - e) <= 0:
                        continue
                    end = min(qe + hd, len(idx))
                    px, qx = _walk(o, h, l, c, qe, end, dr, e, stop, tgt)
                    raw7.append({**base, "v": k7(vv, rr, y, tg, hd), "time": idx[qe], "exit_time": idx[qx], "dir": dr,
                                 "entry_px": e, "stop_px": stop, "r": dr * (px - e) / risk, "wt": 1.0, "side": side,
                                 "leg": 0, "min": int(qe - k0)})

    def net(raw):
        if not raw:
            return []
        tt = pd.DataFrame(raw)
        nt = C.apply(tt, sym, sp)
        tt["net"] = nt["r"].to_numpy(float)
        rows = []
        for (vk, rl), y_ in tt.groupby(["v", "rel"]):              # one row per release: the entries' sum, 1/N each
            rows.append({"v": vk, "sym": sym, "cls": U.CLASS.get(sym, "?"), "t": pd.Timestamp(y_["time"].min()).isoformat(),
                         "rel": pd.Timestamp(rl).isoformat(), "r": round(float((y_["net"] * y_["wt"]).sum()), 4),
                         "gross": round(float((y_["r"] * y_["wt"]).sum()), 4), "entries": int(len(y_)),
                         "cat": y_["cat"].iloc[0], "side": y_["side"].iloc[0], "min": int(y_["min"].min())})
        return rows
    return net(raw6), net(raw7)


def expand(df):
    """A move of 2.5 ATR is also one of 1.5 and of 1 ATR (EV06 only)."""
    out = []
    for m in MULTS:
        z = df[df["v"].str.extract(r"\|m([\d.]+)\|")[0].astype(float) >= m]
        if len(z):
            out.append(z.assign(v=z["v"].str.replace(r"\|m[\d.]+\|", f"|m{m:g}|", regex=True)))
    return pd.concat(out, ignore_index=True) if out else df


def study(df, base_key, keys):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
    cand = []
    for v, y in df.groupby("v"):
        for cap in CAPS:
            z = NS.capped(y, cap)
            tr, un = z[z.train], z[~z.train]
            cand.append({"v": v + (f"|cap{cap}" if cap else ""), "train": NF._w(tr), "unseen": NF._w(un),
                         "gross_unseen": round(float(un["gross"].sum()), 2) if len(un) else 0.0,
                         "entries_unseen": int(un["entries"].sum()) if len(un) else 0})
    cand.sort(key=lambda c_: (c_["train"]["n"] < MIN_N, -c_["train"]["sum"]))
    top = cand[:TOP]
    base = next((c_ for c_ in cand if c_["v"] == base_key), None)
    by = {}
    for pos, key in enumerate(keys):
        g = {}
        for c_ in cand:
            g.setdefault(c_["v"].split("|")[pos], []).append(c_)
        by[key] = {k_: {"variants": len(cs), "train_median_sum": round(float(np.median([c_["train"]["sum"] for c_ in cs])), 2),
                        "unseen_median_sum": round(float(np.median([c_["unseen"]["sum"] for c_ in cs])), 2),
                        "unseen_positive": int(sum(1 for c_ in cs if (c_["unseen"]["sum"] or 0) > 0))}
                   for k_, cs in g.items()}
    return {"baseline": base, "best": cand[0] if cand else None, "top": top,
            "robust": {"top_positive_unseen": int(sum(1 for c_ in top if (c_["unseen"]["sum"] or 0) > 0)), "top": len(top),
                       "top_better_than_base": int(sum(1 for c_ in top if base and (c_["unseen"]["sum"] or 0) > (base["unseen"]["sum"] or 0)))},
            "by": by, "tested": len(cand)}


def by_group(df, col, base_key, keys):
    out = {}
    for k, y in df.groupby(col):
        r_ = study(y, base_key, keys)
        top = [c_ for c_ in r_["top"] if c_["train"]["n"] >= MIN_N]
        if not top:
            continue
        out[str(k)] = {"best": top[0], "baseline": r_["baseline"],
                       "top_positive_unseen": int(sum(1 for c_ in top if (c_["unseen"]["sum"] or 0) > 0)), "top": len(top)}
    return out


def full(df, base_key, keys):
    if not len(df):
        return {}
    res = study(df, base_key, keys)
    res["gold"] = study(df[df["sym"] == "XAUUSD"], base_key, keys) if (df["sym"] == "XAUUSD").any() else None
    res["by_cat"] = by_group(df, "cat", base_key, keys)
    res["by_sym"] = by_group(df, "sym", base_key, keys)
    res["by_side"] = by_group(df, "side", base_key, keys)
    return res


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS) or j.get("partial", True)
                or j.get("version") != VERSION)
    except Exception:  # noqa: BLE001
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--symbols", default="")
    a = ap.parse_args(argv)
    if not a.force and not stale():
        print("retest study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    rel = NF.releases()
    CACHE.mkdir(parents=True, exist_ok=True)
    syms = [s for s in a.symbols.split(",") if s] or ["XAUUSD"] + [s for s in NF.SYMBOLS if s != "XAUUSD"]   # the gold first
    r6, r7, done = [], [], []
    for sym in syms:
        f = CACHE / f"{sym}.json"
        try:
            c_ = json.loads(f.read_text(encoding="utf-8"))
            if c_.get("version") == VERSION and pd.Timestamp.now(tz="UTC") - pd.Timestamp(c_["at"]) < pd.Timedelta(days=STALE_DAYS):
                r6 += c_["ev06"]
                r7 += c_["ev07"]
                done.append(sym)
                continue
        except Exception:  # noqa: BLE001
            pass
        if time.time() > t_end and done:
            print(f"EV06/EV07: time budget reached after {done}")
            continue
        try:
            a6, a7 = trades_sym(sym, rel)
            r6 += a6
            r7 += a7
            f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "ev06": a6, "ev07": a7}),
                         encoding="utf-8")
            done.append(sym)
            print(f"EV06 {sym}: {len(a6)} rows, EV07: {len(a7)} rows", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"EV06/EV07 {sym}: failed ({e})")
    d6 = expand(pd.DataFrame(r6)) if r6 else pd.DataFrame()
    d7 = pd.DataFrame(r7) if r7 else pd.DataFrame()
    res = {"ev06": full(d6, BASE6, ("win", "mult", "tol", "stop", "entries", "target", "hold")),
           "ev07": full(d7, BASE7, ("vol", "range", "stop", "target", "hold"))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "symbols": done, "partial": len(done) < len(NF.SYMBOLS), "rows": {"ev06": int(len(d6)), "ev07": int(len(d7))},
                               "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()),
                               "base": {"ev06": BASE6, "ev07": BASE7}, "note": "tick volume, no order book",
                               "study": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k_, r_ in res.items():
        if r_:
            print(k_, json.dumps({"base": r_["baseline"], "best": r_["best"], "robust": r_["robust"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
