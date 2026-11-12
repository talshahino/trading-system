"""Shocks without a calendar release - bot 7's smart layer (user, 11.11.2026; data/agents/strategies/shocks.md).

The definition of a shock is the agent's (user: "the agent decides what a shock is"). Version 1 (12.11.2026):
  base     the median 1-minute range (high - low) of the instrument over the previous 1,440 minutes it traded
  shock    at the close of minute t: the 1-minute body |c - o| >= K x base, or the 3-minute move (c[t] - o[t-2])
           >= 1.5 K x base; and at least the class floor in percent (so a quiet Asian hour cannot make one);
           the first minute that fires starts the shock, the next 30 minutes are the same shock
  skipped  the 15 minutes after a break of 30+ minutes (or 30 flat minutes), the market's clock (SESSIONS: daily
           rollover / reopen 16:55-18:20 New York, US cash open / close, Xetra open / close, NYMEX settlement - v2,
           12.11.2026), and every shock from 10 minutes before to 60 after an
           important (high impact or FTMO-restricted) release of a currency the instrument moves on
Why so: a ratio to the instrument's own recent median adapts to every instrument and regime (a 0.1% minute is a
shock on EURUSD at 3am and noise on BTC); the median ignores the shock itself; the floor keeps it to moves
someone would call news. K is chosen from the data below (the study's best K on 2025) and reported.

Measured per shock (1-minute candles): the exact start minute, the size (x base, %), how long the move went on,
the retrace inside the first 3 minutes, Dukascopy's ask-bid spread per minute (t-1 .. t+5), the first correcting
minute, and the move after 1 / 2 / 3 / 5 / 15 / 60 / 240 minutes (in shock units, + = continuation).

Strategies (entry D minutes after the shock minute closes, at the open; FTMO costs: the spread at the entry minute -
wide during shocks, it is in the data - and FTMO commission):
  CONT  with the shock; stop at the shock's origin (s=1) or half of it (s=0.5); out after H minutes or at 2R
  PULL  a limit at a 50% pull-back of the shock within 15 minutes; stop at the origin; target the shock's extreme
        or out after H minutes
  FADE  against the shock; stop 25% of the shock beyond its extreme; target a 50% give-back or out after H
Chosen on 30.04-31.12.2025 (the FTMO calendar starts April 2025, so earlier shocks cannot be told from releases),
shown on 2026 (unseen), time-weighted (agents/agentkit.py).

Outputs: data/agents/shocks_scan.json (the shocks of the last 30 days, grouped across instruments - the nightly
agent adds the cause from the news into data/agents/shocks.json), reports/agents/bot7/shocks_study.json.
  python agents/shocks.py [--minutes N] [--days 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import agentkit as K  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import poc_agent as P  # noqa: E402
from agents import repo_io as RIO  # noqa: E402

INSTR = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US100", "US500", "US30", "GER40", "WTI", "BTCUSD"]
CURS = {"EURUSD": ["USD", "EUR"], "GBPUSD": ["USD", "GBP"], "USDJPY": ["USD", "JPY"], "XAUUSD": ["USD"],
        "US100": ["USD"], "US500": ["USD"], "US30": ["USD"], "GER40": ["EUR", "USD"], "WTI": ["USD", "Crude Oil", "CAD"],
        "BTCUSD": ["USD"]}
FLOOR = {"EURUSD": 0.10, "GBPUSD": 0.10, "USDJPY": 0.10, "XAUUSD": 0.20, "US100": 0.20, "US500": 0.18, "US30": 0.18,
         "GER40": 0.18, "WTI": 0.35, "BTCUSD": 0.60}           # % of price, the smallest move called a shock
KS = (6, 8, 10, 12)
AFTER = (1, 2, 3, 5, 15, 60, 240)
START, SPLIT = pd.Timestamp("2025-04-30", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
SCAN = ROOT / "data" / "agents" / "shocks_scan.json"
LOG = ROOT / "data" / "agents" / "shocks.json"
OUT = ROOT / "reports" / "agents" / "bot7" / "shocks_study.json"


def releases():
    """Important releases (high impact or FTMO-restricted): FTMO's history + the live calendar (branch live)."""
    items = json.loads((ROOT / "data" / "ftmo_calendar" / "history.json").read_text(encoding="utf-8"))
    items += (RIO.live_json("live/calendar.json") or {}).get("items", [])
    out = {}
    for i in items:
        if i.get("impact") != "high" and not i.get("restriction"):
            continue
        ins = str(i.get("instrument", ""))
        cur = "Crude Oil" if ins.startswith("Crude") else ins.split(" ")[0]
        out.setdefault(cur, {})[pd.Timestamp(i["date"]).tz_convert("UTC")] = i.get("title", "")
    return {k: (pd.DatetimeIndex(sorted(v)), [v[t] for t in sorted(v)]) for k, v in out.items()}


def near_release(t, sym, rel):
    """The title of an important release from 10 minutes after t back to 60 minutes before it, else None."""
    for cur in CURS[sym]:
        ix, titles = rel.get(cur, (None, None))
        if ix is None or not len(ix):
            continue
        k = ix.searchsorted(t - pd.Timedelta(minutes=60))
        if k < len(ix) and ix[k] <= t + pd.Timedelta(minutes=10):
            return f"{titles[k]} ({cur}, {ix[k].strftime('%Y-%m-%d %H:%M')} UTC)"
    return None


# scheduled minutes that move markets every day without news (local exchange time, DST-proof): daily rollover /
# futures reopen, cash opens and closes, settlement. Shocks there are the market's clock, not news (night of 12.11.2026:
# the first scan's US30 "shocks" were mostly the 09:30 New York open, GER40's the 09:00 Xetra open).
SESSIONS = {"*": [("America/New_York", "16:55", "18:20")],
            "US100": [("America/New_York", "09:28", "09:45"), ("America/New_York", "15:58", "16:16")],
            "GER40": [("Europe/Berlin", "08:58", "09:15"), ("Europe/Berlin", "17:28", "17:40")],
            "WTI": [("America/New_York", "14:27", "14:33")]}
SESSIONS["US500"] = SESSIONS["US30"] = SESSIONS["US100"]


def session_minute(idx, sym):
    out = np.zeros(len(idx), bool)
    for tz, a, b in SESSIONS["*"] + SESSIONS.get(sym, []):
        loc = idx.tz_convert(tz)
        m = loc.hour * 60 + loc.minute
        lo, hi = (int(x[:2]) * 60 + int(x[3:]) for x in (a, b))
        out |= (m >= lo) & (m <= hi)
    return out


def detect(d, sym, k_mult, rel):
    """Shock minutes of one instrument for one K. Returns a list of dicts (index position, direction, size)."""
    o, h, l, c = (d[x].to_numpy(float) for x in "ohlc")
    idx = d.index
    rng = pd.Series(h - l)
    base = rng.where(rng > 0).rolling(1440, min_periods=300).median().shift(1).to_numpy()
    gap = np.r_[np.inf, np.diff(idx.asi8) / 6e10]                 # minutes since the previous candle
    since_break = np.full(len(c), 1e9)
    last, flat = -1e9, 0
    for i in range(len(c)):
        flat = flat + 1 if h[i] - l[i] <= 0 else 0
        if gap[i] > 30 or flat >= 30:                          # a break, or a market that sat still 30 minutes
            last = i
        since_break[i] = i - last
    body = c - o
    m3 = c - np.r_[o[:2], o[:-2]]
    mins = idx.hour * 60 + idx.minute
    floor = FLOOR[sym] / 100 * c
    fire1 = (np.abs(body) >= k_mult * base) & (np.abs(body) >= floor)
    fire3 = (np.abs(m3) >= 1.5 * k_mult * base) & (np.abs(m3) >= 1.5 * floor)
    ok = np.isfinite(base) & (base > 0) & (since_break > 15) & ~session_minute(idx, sym)
    cand = np.flatnonzero((fire1 | fire3) & ok)
    out, nxt = [], -1
    for i in cand:
        if i < nxt or i < 3 or i + 1 >= len(c):
            continue
        one = bool(fire1[i] and abs(body[i]) >= abs(m3[i]) / 1.5)
        move, org = (body[i], o[i]) if one else (m3[i], o[i - 2])
        nxt = i + 30
        if near_release(idx[i], sym, rel):
            continue
        out.append({"i": int(i), "dir": int(np.sign(move)), "move": float(abs(move)), "origin": float(org),
                    "x": float(abs(move) / base[i]), "pct": float(abs(move) / c[i] * 100), "base": float(base[i])})
    return out


def _cost_r(sym, sp, t, e_px, stop, dr):
    tt = pd.DataFrame({"r": [0.0], "entry_px": [e_px], "stop_px": [stop], "dir": [dr], "time": [t], "exit_time": [t]})
    x = C.apply(tt, sym, sp)
    return float((x["spread_px"].iloc[0] + x["comm_px"].iloc[0]) / abs(e_px - stop))


def _walk(o, h, l, c, k, end, dr, entry, stop, tgt):
    """R of a trade from candle k (entered at entry) to candle end-1: stop first (worst case), then target."""
    risk = abs(entry - stop)
    for q in range(k, end):
        if (l[q] <= stop) if dr > 0 else (h[q] >= stop):
            return -1.0
        if tgt is not None and ((h[q] >= tgt) if dr > 0 else (l[q] <= tgt)):
            return dr * (tgt - entry) / risk
    return dr * (c[end - 1] - entry) / risk


def trades(d, sp, sym, shocks, k_mult):
    o, h, l, c = (d[x].to_numpy(float) for x in "ohlc")
    idx, n = d.index, len(d)
    rows = []
    for s in shocks:
        i, dr, mv, org = s["i"], s["dir"], s["move"], s["origin"]
        if i + 241 >= n:
            continue
        ext = h[i - 2:i + 1].max() if dr > 0 else l[i - 2:i + 1].min()
        base = {"sym": sym, "t": idx[i], "K": k_mult, "x": s["x"]}
        for D in (1, 2, 3):
            ke = i + D
            if (idx[ke] - idx[i]) > pd.Timedelta(minutes=D + 2):
                continue
            e = o[ke]
            for sf in (1.0, 0.5):                                 # CONT
                stop = c[i] - dr * sf * mv
                if dr * (e - stop) <= 0.15 * mv:
                    continue
                cr = _cost_r(sym, sp, idx[ke], e, stop, dr)
                for H in (5, 15, 60):
                    for tg in (None, 2.0):
                        tgt = None if tg is None else e + dr * tg * abs(e - stop)
                        rows.append({**base, "strat": "CONT", "D": D, "p": sf, "H": H, "tg": tg or 0,
                                     "r": _walk(o, h, l, c, ke, min(ke + H, n), dr, e, stop, tgt) - cr, "cost_r": cr})
            stop = ext + dr * 0.25 * mv                           # FADE
            if dr * (stop - e) > 0.15 * mv:
                cr = _cost_r(sym, sp, idx[ke], e, stop, -dr)
                for H in (15, 60):
                    for tg in (None, 0.5):
                        tgt = None if tg is None else c[i] - dr * tg * mv
                        if tgt is not None and dr * (e - tgt) <= 0:
                            continue
                        rows.append({**base, "strat": "FADE", "D": D, "p": 0.25, "H": H, "tg": tg or 0,
                                     "r": _walk(o, h, l, c, ke, min(ke + H, n), -dr, e, stop, tgt) - cr, "cost_r": cr})
        lim, stop = c[i] - dr * 0.5 * mv, org                     # PULL
        for q in range(i + 1, min(i + 16, n)):
            if (l[q] <= lim) if dr > 0 else (h[q] >= lim):
                if (l[q] <= stop) if dr > 0 else (h[q] >= stop):
                    r0 = -1.0                                     # filled and stopped in the same minute
                else:
                    r0 = None
                cr = _cost_r(sym, sp, idx[q], lim, stop, dr)
                for H in (15, 60):
                    for tg in (None, 1.0):
                        tgt = None if tg is None else lim + dr * tg * abs(lim - stop)
                        r = r0 if r0 is not None else _walk(o, h, l, c, q + 1, min(q + 1 + H, n), dr, lim, stop, tgt)
                        rows.append({**base, "strat": "PULL", "D": 0, "p": 0.5, "H": H, "tg": tg or 0, "r": r - cr, "cost_r": cr})
                break
    return rows


def describe(d, sp, sym, s):
    """One shock as the log keeps it."""
    o, h, l, c = (d[x].to_numpy(float) for x in "ohlc")
    idx, i, dr, mv = d.index, s["i"], s["dir"], s["move"]
    n = len(c)
    start = i
    for j in (i - 1, i - 2):
        if j >= 0 and dr * (c[j] - o[j]) > 2 * s["base"]:
            start = j
        else:
            break
    dur = 1 + (i - start)
    for j in range(i + 1, min(i + 30, n)):
        if dr * (c[j] - o[j]) > 2 * s["base"]:
            dur += 1
        else:
            break
    fc = next((j - i for j in range(i + 1, min(i + 60, n)) if dr * (c[j] - c[j - 1]) < 0), None)
    back3 = (max(0.0, -dr * ((l[i + 1:i + 4].min() if dr > 0 else h[i + 1:i + 4].max()) - c[i])) / mv) if i + 4 < n else None
    after = {}
    for a in AFTER:
        if i + a < n and idx[i + a] - idx[i] <= pd.Timedelta(minutes=a * 1.5 + 5):
            after[str(a)] = {"x": round(dr * (c[i + a] - c[i]) / mv, 2), "pct": round(dr * (c[i + a] - c[i]) / c[i] * 100, 3)}
    spr = []
    if sp is not None and len(sp):
        for j in range(i - 1, min(i + 6, n)):
            v = sp.get(idx[j])
            spr.append(None if v is None or v != v else round(float(v) / c[j] * 1e4, 2))   # in basis points
    return {"sym": sym, "t": idx[start].isoformat(), "t_signal": idx[i].isoformat(), "dir": "up" if dr > 0 else "down",
            "x": round(s["x"], 1), "pct": round(s["pct"] * dr, 3), "px": round(float(c[i]), 5), "minutes": dur,
            "first_correction_min": fc, "retrace_3min": None if back3 is None else round(back3, 2),
            "spread_bp": spr, "after": after}


def group(evs):
    """Shocks of several instruments within 5 minutes are one event."""
    evs = sorted(evs, key=lambda e: e["t"])
    out = []
    for e in evs:
        if out and pd.Timestamp(e["t"]) - pd.Timestamp(out[-1]["t"]) <= pd.Timedelta(minutes=5):
            out[-1]["legs"].append(e)
            continue
        out.append({"t": e["t"], "legs": [e]})
    for g in out:
        g["id"] = pd.Timestamp(g["t"]).strftime("%Y%m%d%H%M")
        g["instruments"] = sorted({x["sym"] for x in g["legs"]})
        g["main"] = max(g["legs"], key=lambda x: x["x"])["sym"]
    return out


def stats(x, now):
    w = K.weights(x["t"].astype(str), now)
    return K.wstats(x["r"].to_numpy(), w)


def study(df, causes, now):
    keys = ["K", "strat", "D", "p", "H", "tg"]
    tr, un = df[df["t"] < SPLIT], df[df["t"] >= SPLIT]
    rows = []
    for key, x in df.groupby(keys):
        a, b = tr.loc[tr.index.intersection(x.index)], un.loc[un.index.intersection(x.index)]
        rows.append({**dict(zip(keys, [float(v) if isinstance(v, (np.floating, float)) else v for v in key])),
                     "train": stats(a, now), "unseen": stats(b, now), "whole": stats(x, now),
                     "halves": K.halves(x["t"].astype(str), x["r"]), "cost_r": round(float(x["cost_r"].mean()), 3)})
    ok = [r for r in rows if (r["train"]["eff_n"] or 0) >= 30 and r["train"]["mean"] is not None]
    ok.sort(key=lambda r: -(r["train"]["t"] or -99))
    best = ok[0] if ok else None
    if best:
        nb = [r for r in rows if r["strat"] == best["strat"] and r is not best and
              sum(r[k] != best[k] for k in keys) == 1 and r["unseen"]["mean"] is not None]
        best["neighbours"] = {"values": len(nb), "share_positive": round(float(np.mean([r["unseen"]["mean"] > 0 for r in nb])), 2) if nb else None}
    per_strat = {}
    for s in ("CONT", "PULL", "FADE"):
        c = [r for r in ok if r["strat"] == s]
        if c:
            per_strat[s] = c[0]
    by_cat = {}
    if causes:
        df = df.assign(cat=[causes.get((r.sym, r.t.strftime("%Y%m%d%H%M"))) for r in df.itertuples()])
        for (cat, strat), x in df.dropna(subset=["cat"]).groupby(["cat", "strat"]):
            by_cat.setdefault(cat, {})[strat] = {"n": int(len(x)), "mean": round(float(x["r"].mean()), 3)}
    per_k = {}
    for kk, x in df[(df["strat"] == "CONT")].groupby("K"):
        per_k[str(kk)] = {"shocks_train": int(x[x["t"] < SPLIT].drop_duplicates(["sym", "t"]).shape[0]),
                          "shocks_unseen": int(x[x["t"] >= SPLIT].drop_duplicates(["sym", "t"]).shape[0])}
    return {"tested": len(rows), "best": best, "per_strategy": per_strat, "top": ok[:15], "per_K": per_k, "by_category": by_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    t_end = time.time() + a.minutes * 60
    now = pd.Timestamp.now(tz="UTC")
    rel = releases()
    causes = {}
    try:
        for e in json.loads(LOG.read_text(encoding="utf-8")).get("shocks", []):
            for leg in e.get("legs", []):
                if e.get("category"):
                    causes[(leg["sym"], pd.Timestamp(leg["t_signal"]).strftime("%Y%m%d%H%M"))] = e["category"]
    except Exception:  # noqa: BLE001
        pass
    all_rows, recent, done = [], [], []
    for sym in INSTR:
        if time.time() > t_end and done:
            print(f"time budget reached after {done}")
            break
        try:
            d = P.bars(sym, "1Min", (START - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
            sp = P.spread_1m(sym, (START - pd.Timedelta(days=3)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: no data ({e})")
            continue
        if d is None or d.empty:
            continue
        d = d[~d.index.duplicated()].sort_index()
        d = d[d.index >= START - pd.Timedelta(days=2)]
        sp = sp[~sp.index.duplicated()] if sp is not None and len(sp) else sp
        for kk in KS:
            sh = [s for s in detect(d, sym, kk, rel) if d.index[s["i"]] >= START]
            all_rows += trades(d, sp, sym, sh, kk)
            if kk == 8:                                            # the log's definition (version 1)
                recent += [describe(d, sp, sym, s) for s in sh if d.index[s["i"]] >= now - pd.Timedelta(days=a.days)]
            print(f"{sym} K={kk}: {len(sh)} shocks")
        done.append(sym)
    df = pd.DataFrame(all_rows)
    res = study(df, causes, now) if len(df) else {"tested": 0}
    groups = group(recent)
    SCAN.parent.mkdir(parents=True, exist_ok=True)
    SCAN.write_text(json.dumps({"updated": now.isoformat(timespec="seconds"), "definition": "v1 K=8 (agents/shocks.py)",
                                "days": a.days, "instruments": done, "shocks": groups}, ensure_ascii=False, indent=1, default=str),
                    encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"updated": now.isoformat(timespec="seconds"), "instruments": done, "train": [str(START.date()), "2025-12-31"],
                               "unseen": ["2026-01-01", str(now.date())], **res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"{len(groups)} shock events in the last {a.days} days; {res.get('tested')} variants tested")
    if res.get("best"):
        b = res["best"]
        print("best on 2025:", {k: b[k] for k in ("K", "strat", "D", "p", "H", "tg")}, "train", b["train"], "unseen", b["unseen"])


if __name__ == "__main__":
    main()
