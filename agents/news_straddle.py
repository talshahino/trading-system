"""EV04 - the straddle before a release (the user's task tal1411strad, 14.11.2026: TikTok ExpertFx, gold on the PPI).

The idea: a minute before a red (high-impact) release put a Buy Stop above and a Sell Stop below the price, X away
(X = a multiple of the 5-minute ATR before the release). OCO: the first one filled cancels the other. Fixed risk per
trade (the bot's), never a martingale or a doubled position, one trade per release per instrument.
Grid (1-minute candles, Dukascopy, FTMO's spread at the fill minute + commission, agents/ftmo_costs.py):
  start    pre  - the orders go in at the open of the minute before the release (the original idea; INSIDE FTMO's
                  restricted window, so only for a virtual bot / the challenge phase)
           safe - the same orders at the open of minute 3 after the release (outside FTMO's 2 minutes; a pending order
                  filled inside the window is a violation on a funded account)
  X        0.5 / 1.0 / 1.5 x the 5-minute ATR(14) of the bars before the orders
  stop     opp  - the other order's level (risk = 2X + slippage)     atr1 - 1 x ATR from the fill
  exit     t1.5 / t2 (target in R) / be1t2 (stop to entry at +1R, target 2R) / be1 (stop to entry at +1R, no target)
  hold     out after 15 / 30 minutes from the fill
  cancel   not filled within 5 minutes after the release (safe: 5 minutes after the orders) -> no trade
  slippage the fill is worse than the order by half the minute's spread + 0.1 ATR (stop orders fill badly in news),
           on top of the spread the cost model charges; a minute that touches both orders counts as a full loss
           (filled and stopped in the same minute: the order inside a 1-minute candle is unknown)
  and in the study: which releases (USD only - the idea - or every red release of the instrument's currencies), a
  spread limit (no orders when the spread at the order minute is over 3 x the instrument's median), and at most
  1 / 2 / 3 trades per release over all instruments (EV01's lesson: one PPI = 8 stops on one dollar bet).
Fair method as the other bot 7 studies: chosen on 30.04-31.12.2025 by total R, shown on 2026 (never seen),
time-weighted (agents/agentkit.py); robustness = of the 10 best on 2025, how many made money in 2026. Also the gold
alone (the user's example) and each release type on its own.

Output: reports/agents/bot7/straddle_study.json. Cached per instrument (cache/news_straddle/<sym>.json), resumes
after a time budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7 (after the entry study).
  python -m agents.news_straddle [--minutes 10] [--force]
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

OUT = ROOT / "reports" / "agents" / "bot7" / "straddle_study.json"
CACHE = ROOT / "cache" / "news_straddle"
VERSION = 1
STARTS = {"pre": -1, "safe": 3}            # the order minute relative to the release minute
XS = (0.5, 1.0, 1.5)
STOPS = ("opp", "atr1")
EXITS = ("t1.5", "t2", "be1t2", "be1")
HOLDS = (15, 30)
CANCEL = 5                                 # minutes after the release (pre) / after the orders (safe)
SLIP_ATR = 0.1
CURS = ("usd", "own")
SPREAD_CAP = (0.0, 3.0)                    # x the instrument's median spread at the order minute (0 = no limit)
CAPS = (0, 1, 2, 3)
TOP = 10
STALE_DAYS = 3
MIN_CAT_N = 8
BASE = "pre|x1|opp|t2|h30|cusd|sp0"        # the idea as told: 1 ATR, stop at the other order, 2R, 30 minutes
SPAN = 40                                  # minutes after the release that must be there without a gap


def vkey(st, x, stop, ex, hold):
    return f"{st}|x{x:g}|{stop}|{ex}|h{hold}"


def _atr5(d1):
    d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    a = R.atr(d5["h"].to_numpy(), d5["l"].to_numpy(), d5["c"].to_numpy(), 14)
    return pd.Series(a, index=d5.index)


def _exit(o, h, l, c, q0, end, dr, e, stop, tgt, be):
    """Walk 1-minute bars from the fill bar q0 (stop only in it: the order inside the candle is unknown), stop before
    target in every bar; be = the R at which the stop goes to entry (None = never). -> (exit price, exit bar)."""
    risk = abs(e - stop)
    s = stop
    if (l[q0] <= s) if dr > 0 else (h[q0] >= s):
        return s, q0
    moved = False
    for q in range(q0 + 1, end):
        if (l[q] <= s) if dr > 0 else (h[q] >= s):
            return s, q
        if tgt is not None and ((h[q] >= tgt) if dr > 0 else (l[q] <= tgt)):
            return tgt, q
        if be is not None and not moved and ((h[q] - e) if dr > 0 else (e - l[q])) >= be * risk:
            s, moved = e, True
    return c[end - 1], end - 1


def trades_sym(sym, rel):
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = P.bars(sym, "1Min", NF.LOAD_FROM, now)
    if d1 is None or d1.empty:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, NF.LOAD_FROM, now)
        sp = sp[~sp.index.duplicated()].sort_index() if sp is not None and len(sp) else None
    except Exception:  # noqa: BLE001
        sp = None
    idx = d1.index
    o, h, l, c = (d1[x].to_numpy(float) for x in "ohlc")
    a5 = _atr5(d1)
    spm = float(sp.median()) if sp is not None and len(sp) else 0.0
    own = set(E.events_for(sym))
    usd = {t for t, items in rel.items() if any(x[0] == "USD" for x in items)}   # red USD releases (the idea)
    times = sorted(t for t in (own | usd) if t >= NF.TEST_FROM)

    def spread_at(t):
        if sp is None:
            return 0.0
        k = sp.index.searchsorted(t, side="right") - 1
        return float(sp.iloc[k]) if k >= 0 else spm

    raw = []
    for t in times:
        k0 = idx.searchsorted(t)
        if k0 >= len(idx) or idx[k0] != t or k0 < 30 or k0 + SPAN >= len(idx) or idx[k0 + SPAN] - t != pd.Timedelta(minutes=SPAN):
            continue
        g = NF.tag(sym, t + pd.Timedelta(minutes=1), rel, [t])
        for st, off in STARTS.items():
            kp = k0 + off
            if idx[kp] - t != pd.Timedelta(minutes=off):
                continue
            ka = a5.index.searchsorted(idx[kp].floor("5min")) - 1       # the last 5-minute bar closed before the orders
            if ka < 14:
                continue
            atr = float(a5.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            ref = o[kp]
            spr = spread_at(idx[kp])
            last = (k0 + CANCEL) if off < 0 else (kp + CANCEL)       # the last bar the orders can fill in
            for x in XS:
                buy, sell = ref + x * atr, ref - x * atr
                q0, dr, both = None, 0, False
                for q in range(kp, last):
                    up, dn = h[q] >= buy, l[q] <= sell
                    if up or dn:
                        q0, dr, both = q, (1 if up else -1), bool(up and dn)
                        break
                if q0 is None:
                    continue
                slip = 0.5 * spread_at(idx[q0]) + SLIP_ATR * atr
                lvl = buy if dr > 0 else sell
                e = max(lvl, o[q0]) + slip if dr > 0 else min(lvl, o[q0]) - slip     # a gap through the order fills at the open
                for stop_k in STOPS:
                    stop = (sell if dr > 0 else buy) if stop_k == "opp" else e - dr * atr
                    risk = dr * (e - stop)
                    if risk <= 0:
                        continue
                    for ex, hold in itertools.product(EXITS, HOLDS):
                        tg = {"t1.5": 1.5, "t2": 2.0, "be1t2": 2.0, "be1": None}[ex]
                        be = 1.0 if ex.startswith("be1") else None
                        tgt = None if tg is None else e + dr * tg * risk
                        end = min(q0 + hold, len(idx))
                        if both:
                            px, qx = stop, q0
                        else:
                            px, qx = _exit(o, h, l, c, q0, end, dr, e, stop, tgt, be)
                        raw.append({"v": vkey(st, x, stop_k, ex, hold), "time": idx[q0], "exit_time": idx[qx], "dir": dr,
                                    "entry_px": e, "stop_px": stop, "r": dr * (px - e) / risk, "rel": t,
                                    "usd": int(t in usd), "own": int(t in own), "spr": spr / spm if spm > 0 else 0.0,
                                    "cat": (g or {}).get("cat", "other"), "both": int(both)})
    if not raw:
        return []
    tt = pd.DataFrame(raw)
    net = C.apply(tt, sym, sp)
    rows = []
    for x in net.itertuples():
        rows.append({"v": x.v, "sym": sym, "cls": U.CLASS.get(sym, "?"), "t": pd.Timestamp(x.time).isoformat(),
                     "dir": int(x.dir), "r": round(float(x.r), 4), "gross": round(float(tt.at[x.Index, "r"]), 4),
                     "cat": x.cat, "usd": int(x.usd), "own": int(x.own), "spr": round(float(x.spr), 2), "both": int(x.both),
                     "rel": pd.Timestamp(x.rel).isoformat()})
    return rows


def capped(z, cap):
    """At most `cap` trades per release over all instruments: the first by fill time, then the bot's fixed order."""
    if not cap or z.empty:
        return z
    order = {s_: i for i, s_ in enumerate(NF.SYMBOLS)}
    z = z.assign(_o=z["sym"].map(order).fillna(99)).sort_values(["rel", "t", "_o"])
    return z[z.groupby("rel").cumcount() < cap].drop(columns="_o")


def study(df, base_key=BASE):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
    cand = []
    for v, y in df.groupby("v"):
        for cu in CURS:
            z0 = y[y["usd"] == 1] if cu == "usd" else y[y["own"] == 1]
            for spc in SPREAD_CAP:
                z1 = z0 if not spc else z0[z0["spr"] <= spc]
                for cap in CAPS:
                    z = capped(z1, cap)
                    tr, un = z[z.train], z[~z.train]
                    cand.append({"v": f"{v}|c{cu}|sp{spc:g}" + (f"|cap{cap}" if cap else ""),
                                 "train": NF._w(tr), "unseen": NF._w(un),
                                 "gross_unseen": round(float(un["gross"].sum()), 2) if len(un) else 0.0})
    # ranked by total R on 2025; a variant with too few 2025 trades to judge goes last (an empty one has sum 0)
    cand.sort(key=lambda c_: (c_["train"]["n"] < MIN_CAT_N, -c_["train"]["sum"]))
    top = cand[:TOP]
    base = next((c_ for c_ in cand if c_["v"] == base_key), None)
    by = {}
    for key, pat in (("start", r"^(pre|safe)\|"), ("x", r"\|x([\d.]+)\|"), ("stop", r"\|(opp|atr1)\|"),
                     ("exit", r"\|(t1\.5|t2|be1t2|be1)\|"), ("hold", r"\|h(\d+)\|"), ("cur", r"\|c(usd|own)\|"),
                     ("spread", r"\|sp([\d.]+)"), ("cap", r"(?:\|cap(\d+))?$")):
        g = {}
        for c_ in cand:
            m = re.search(pat, c_["v"])
            if m:
                g.setdefault(m.group(1) or "0", []).append(c_)
        by[key] = {k_: {"variants": len(cs), "train_median_sum": round(float(np.median([c_["train"]["sum"] for c_ in cs])), 2),
                        "unseen_median_sum": round(float(np.median([c_["unseen"]["sum"] for c_ in cs])), 2),
                        "unseen_positive": int(sum(1 for c_ in cs if (c_["unseen"]["sum"] or 0) > 0))}
                   for k_, cs in g.items()}
    return {"baseline": base, "best": cand[0] if cand else None, "top": top,
            "robust": {"top_positive_unseen": int(sum(1 for c_ in top if (c_["unseen"]["sum"] or 0) > 0)), "top": len(top),
                       "top_better_than_base": int(sum(1 for c_ in top if base and (c_["unseen"]["sum"] or 0) > (base["unseen"]["sum"] or 0)))},
            "by": by, "tested": len(cand)}


def by_group(df, col):
    out = {}
    for k, y in df.groupby(col):
        r_ = study(y)
        top = [c_ for c_ in r_["top"] if c_["train"]["n"] >= MIN_CAT_N]
        if not top:
            continue
        out[str(k)] = {"best": top[0], "baseline": r_["baseline"],
                       "top_positive_unseen": int(sum(1 for c_ in top if (c_["unseen"]["sum"] or 0) > 0)), "top": len(top)}
    return out


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
        print("straddle study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    rel = NF.releases()
    CACHE.mkdir(parents=True, exist_ok=True)
    syms = [s for s in a.symbols.split(",") if s] or ["XAUUSD"] + [s for s in NF.SYMBOLS if s != "XAUUSD"]   # the gold first
    rows, done = [], []
    for sym in syms:
        f = CACHE / f"{sym}.json"
        try:
            c_ = json.loads(f.read_text(encoding="utf-8"))
            if c_.get("version") == VERSION and pd.Timestamp.now(tz="UTC") - pd.Timestamp(c_["at"]) < pd.Timedelta(days=STALE_DAYS):
                rows += c_["rows"]
                done.append(sym)
                continue
        except Exception:  # noqa: BLE001
            pass
        if time.time() > t_end and done:
            print(f"EV04: time budget reached after {done}")
            continue
        try:
            r_ = trades_sym(sym, rel)
            rows += r_
            f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
            done.append(sym)
            print(f"EV04 {sym}: {len(r_)} rows ({len(rows)} so far)", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"EV04 {sym}: failed ({e})")
    df = pd.DataFrame(rows)
    res = study(df) if len(df) else {}
    if len(df):
        tr = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
        res["gold"] = study(df[df["sym"] == "XAUUSD"]) if (df["sym"] == "XAUUSD").any() else None
        res["by_cat"] = by_group(tr.drop(columns="train"), "cat")
        res["by_sym"] = by_group(tr.drop(columns="train"), "sym")
        res["both_share"] = round(float(df["both"].mean()), 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "symbols": done, "partial": len(done) < len(NF.SYMBOLS), "rows": int(len(df)),
                               "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()), "base": BASE,
                               "study": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res:
        print("EV04", json.dumps({"base": res["baseline"], "best": res["best"], "robust": res["robust"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
