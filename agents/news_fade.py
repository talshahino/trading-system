"""EV05 - fading the release spike early (the user's task tal1411fade, 14.11.2026: TikTok IMPERATOR, gold on the PPI,
5 minutes: gold jumped ~$20 on the release and fell straight back; he sold near the top of the spike, stop above the
high, target the low from before the release, ~$24).

EV02 (bot 7's fade today) enters only at minute 10 at the market. EV05 enters earlier, on the turn:
  spike   the first W minutes after the release (W = 1 / 3 / 5): the bigger move from the release minute's open, at
          least M x the 5-minute ATR(14) before the release (M = 1 / 1.5 / 2.5); up or down - the fade is the other way
  entry   bar  - a stop order just beyond the peak bar's other end (the spike bar's low for an up spike)
          r25  - a stop order 25% of the spike back from the peak
          m10  - EV02's timing for the comparison: at the market at the open of minute 10 (if the spike gave back a third)
          the orders live from the end of the spike window to minute 10; not filled -> no trade
  start   1 - orders from minute 1 (the original idea; inside FTMO's restricted window - only a virtual bot / challenge)
          3 - nothing fills before minute 3 (outside FTMO's 2 minutes, so a funded account may take it)
  stop    X x ATR beyond the spike's peak (X = 0.25 / 0.5 / 1.0)
  target  pre   - the release minute's open (the price before the release)
          half  - half the way from the peak back to it
          low   - the other end of the 15 minutes before the release (the video: the low before the spike)
          r2    - 2R
  hold    out after 15 / 30 / 60 minutes from the fill
  costs   FTMO's spread at the fill minute + commission (agents/ftmo_costs.py), the stop order fills worse by half the
          minute's spread + 0.1 ATR; a fill bar that also touches the stop counts as stopped (the order inside a
          1-minute candle is unknown); fixed risk per trade, never sized up
  and in the study: the release type (PPI / CPI = infl, NFP = jobs, rates, gdp...), the instrument's own releases only,
  and at most 1 / 2 / 3 trades per release over all instruments (EV01's lesson: one PPI = 8 stops on one bet).
Fair method as the other bot 7 studies: chosen on 30.04-31.12.2025 by total R, shown on 2026 (never seen);
robustness = of the 10 best on 2025, how many made money in 2026. Also the gold alone (the user's example), each
release type, each instrument and the spike against / with the surprise vs the forecast.

Output: reports/agents/bot7/fade_study.json. Cached per instrument (cache/news_fade/<sym>.json), resumes after a time
budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7 (after the straddle study).
  python -m agents.news_fade [--minutes 10] [--force] [--symbols XAUUSD]
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

OUT = ROOT / "reports" / "agents" / "bot7" / "fade_study.json"
CACHE = ROOT / "cache" / "news_fade"
VERSION = 1
WINS = (1, 3, 5)                  # spike window, minutes after the release
MULTS = (1.0, 1.5, 2.5)           # spike size, x ATR(5-minute, 14)
ENTRIES = ("bar", "r25", "m10")
STARTS = (1, 3)
XSTOP = (0.25, 0.5, 1.0)
TARGETS = ("pre", "half", "low", "r2")
HOLDS = (15, 30, 60)
LAST = 10                         # the orders live until minute 10 after the release
SLIP_ATR = 0.1
CAPS = (0, 1, 2, 3)
TOP = 10
STALE_DAYS = 3
MIN_N = 8
BASE = "w5|m2.5|m10|s3|x0.5|pre|h60"   # nearest to EV02 (2.5 ATR, minute 10, target the price before the release)
SPAN = 70


def vkey(w, m, en, st, x, tg, hd):
    return f"w{w}|m{m:g}|{en}|s{st}|x{x:g}|{tg}|h{hd}"


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
    a5 = NS._atr5(d1)
    spm = float(sp.median()) if sp is not None and len(sp) else 0.0
    cur = set(U.CURRENCIES.get(sym, []))                      # the instrument's releases + every red one of its currencies
    own = sorted(t for t in set(E.events_for(sym)) | {t for t, it in rel.items() if any(x[0] in cur for x in it)}
                 if t >= NF.TEST_FROM)

    def spread_at(t):
        if sp is None:
            return 0.0
        k = sp.index.searchsorted(t, side="right") - 1
        return float(sp.iloc[k]) if k >= 0 else spm

    raw = []
    for t in own:
        k0 = idx.searchsorted(t)
        if k0 >= len(idx) or idx[k0] != t or k0 < 30 or k0 + SPAN >= len(idx) or idx[k0 + SPAN] - t != pd.Timedelta(minutes=SPAN):
            continue
        ka = a5.index.searchsorted(t.floor("5min")) - 1           # the last 5-minute bar closed before the release
        if ka < 14:
            continue
        atr = float(a5.iloc[ka])
        if not np.isfinite(atr) or atr <= 0:
            continue
        g = NF.tag(sym, t + pd.Timedelta(minutes=1), rel, [t]) or {}
        ref = o[k0]
        pre_hi, pre_lo = float(h[k0 - 15:k0].max()), float(l[k0 - 15:k0].min())
        for w in WINS:
            seg_h, seg_l = h[k0:k0 + w], l[k0:k0 + w]
            up, dn = float(seg_h.max()) - ref, ref - float(seg_l.min())
            sdir = 1 if up >= dn else -1                           # the spike's direction; the fade is -sdir
            size = max(up, dn)
            if size < min(MULTS) * atr:
                continue
            kp = k0 + int(np.argmax(seg_h) if sdir > 0 else np.argmin(seg_l))
            peak = h[kp] if sdir > 0 else l[kp]
            dr = -sdir
            surp = g.get("surp")
            side = "none" if surp is None or surp == 0 else ("against" if np.sign(surp) != sdir else "with")
            for en, st in itertools.product(ENTRIES, STARTS):
                first = max(k0 + w, k0 + st)
                q0, e = None, None
                if en == "m10":
                    q = k0 + LAST
                    back = (peak - o[q]) * sdir                    # how much of the spike it gave back by minute 10
                    if back >= size / 3.0 and q >= first:
                        q0, e = q, o[q]
                else:
                    lvl = (l[kp] if sdir > 0 else h[kp]) if en == "bar" else peak - sdir * 0.25 * size
                    for q in range(first, k0 + LAST + 1):
                        if (l[q] <= lvl) if dr < 0 else (h[q] >= lvl):
                            q0 = q
                            px0 = min(lvl, o[q]) if dr < 0 else max(lvl, o[q])     # a gap through the order: the open
                            e = px0 + dr * (0.5 * spread_at(idx[q]) + SLIP_ATR * atr)
                            break
                if q0 is None:
                    continue
                for x in XSTOP:
                    stop = peak + sdir * x * atr
                    risk = dr * (e - stop)
                    if risk <= 0:
                        continue
                    for tg, hd in itertools.product(TARGETS, HOLDS):
                        if tg == "pre":
                            tgt = ref
                        elif tg == "half":
                            tgt = peak - sdir * 0.5 * abs(peak - ref)
                        elif tg == "low":
                            tgt = pre_lo if sdir > 0 else pre_hi
                        else:
                            tgt = e + dr * 2.0 * risk
                        if dr * (tgt - e) <= 0:                    # the target is already behind the entry
                            continue
                        end = min(q0 + hd, len(idx))
                        px, qx = NS._exit(o, h, l, c, q0, end, dr, e, stop, tgt, None)
                        raw.append({"v": vkey(w, next(m for m in sorted(MULTS, reverse=True) if size >= m * atr), en, st, x, tg, hd),
                                    "size": size / atr, "time": idx[q0], "exit_time": idx[qx], "dir": dr, "entry_px": e,
                                    "stop_px": stop, "r": dr * (px - e) / risk, "rel": t, "cat": g.get("cat", "other"),
                                    "side": side, "min": int(q0 - k0)})
    if not raw:
        return []
    tt = pd.DataFrame(raw)
    net = C.apply(tt, sym, sp)
    rows = []
    for x in net.itertuples():
        rows.append({"v": x.v, "sym": sym, "cls": U.CLASS.get(sym, "?"), "t": pd.Timestamp(x.time).isoformat(),
                     "dir": int(x.dir), "r": round(float(x.r), 4), "gross": round(float(tt.at[x.Index, "r"]), 4),
                     "size": round(float(x.size), 2), "cat": x.cat, "side": x.side, "min": int(x.min),
                     "rel": pd.Timestamp(x.rel).isoformat()})
    return rows


def expand(df):
    """A spike of 2.5 ATR is also one of 1.5 and of 1 ATR: the rows carry the biggest size class, each variant with
    m = M takes every spike of at least M."""
    out = []
    for m in MULTS:
        z = df[df["size"] >= m]
        if len(z):
            out.append(z.assign(v=z["v"].str.replace(r"\|m[\d.]+\|", f"|m{m:g}|", regex=True)))
    return pd.concat(out, ignore_index=True) if out else df


def study(df, base_key=BASE):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
    cand = []
    for v, y in df.groupby("v"):
        for cap in CAPS:
            z = NS.capped(y, cap)
            tr, un = z[z.train], z[~z.train]
            cand.append({"v": v + (f"|cap{cap}" if cap else ""), "train": NF._w(tr), "unseen": NF._w(un),
                         "gross_unseen": round(float(un["gross"].sum()), 2) if len(un) else 0.0})
    cand.sort(key=lambda c_: (c_["train"]["n"] < MIN_N, -c_["train"]["sum"]))
    top = cand[:TOP]
    base = next((c_ for c_ in cand if c_["v"] == base_key), None)
    by = {}
    for pos, key in enumerate(("win", "mult", "entry", "start", "stop", "target", "hold")):
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


def by_group(df, col):
    out = {}
    for k, y in df.groupby(col):
        r_ = study(y)
        top = [c_ for c_ in r_["top"] if c_["train"]["n"] >= MIN_N]
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
        print("fade study fresh - skipped")
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
            print(f"EV05: time budget reached after {done}")
            continue
        try:
            r_ = trades_sym(sym, rel)
            rows += r_
            f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
            done.append(sym)
            print(f"EV05 {sym}: {len(r_)} rows ({len(rows)} so far)", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"EV05 {sym}: failed ({e})")
    df = expand(pd.DataFrame(rows)) if rows else pd.DataFrame()
    res = study(df) if len(df) else {}
    if len(df):
        res["gold"] = study(df[df["sym"] == "XAUUSD"]) if (df["sym"] == "XAUUSD").any() else None
        res["by_cat"] = by_group(df, "cat")
        res["by_sym"] = by_group(df, "sym")
        res["by_side"] = by_group(df, "side")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "symbols": done, "partial": len(done) < len(NF.SYMBOLS), "rows": int(len(df)),
                               "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()), "base": BASE,
                               "study": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res:
        print("EV05", json.dumps({"base": res["baseline"], "best": res["best"], "robust": res["robust"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
