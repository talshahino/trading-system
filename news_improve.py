"""Improving the three best news candidates (user, 10.11.2026). Research only.

Fair method: every variant is chosen on 2025 only (train) and then shown on 2026 (never seen) - the 2026 number is
the honest one. Same data and costs as news_research.py (1-minute Dukascopy, FTMO spread at the entry minute +
FTMO commission; DRIFT also pays the overnight swap).

  DRIFT  buy an index before a central-bank rate decision, out 5 minutes before it
         variants: start 24h / 6h / at the US (or local) cash open the day before; the index of the bank's country
         (Fed: US500 US100 US30; ECB: GER40; BOE: UK100; BOJ: JP225; RBA: AUS200); stop 30 or 60 x 1-min ATR
  BOX    jobs releases: the box before the release, first 1-minute close outside it from minute D
         variants: box 10/15/30 minutes, D 2/3, target 1.5/2/3R or none (out after 30/60), only when the break agrees
         with the surprise (actual vs forecast), instrument kept only if it made money in 2025
  SURP   rates releases with a forecast: trade the surprise from minute D
         variants: D 2/3/5, out after 30/60/120/240 minutes, also only when the first D minutes agree
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import news_research as NR
from agents import ftmo_costs as C
from agents import poc_agent as P

SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
BANK_INDEX = {"USD": ["US500", "US100", "US30"], "EUR": ["GER40"], "GBP": ["UK100"], "AUD": ["AUS200"], "JPY": ["JP225"]}
OPEN = {"US500": ("America/New_York", 9, 30), "US100": ("America/New_York", 9, 30), "US30": ("America/New_York", 9, 30),
        "GER40": ("Europe/Berlin", 9, 0), "UK100": ("Europe/London", 8, 0), "AUS200": ("Australia/Sydney", 10, 0),
        "JP225": ("Asia/Tokyo", 9, 0)}
DATA = {}


def load(sym, t0, t1):
    if sym not in DATA:
        try:
            d = P.bars(sym, "1Min", t0, t1)
            sp = P.spread_1m(sym, t0, t1)
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: no data ({e})")
            DATA[sym] = None
            return None
        if d is None or d.empty:
            DATA[sym] = None
            return None
        d = d[~d.index.duplicated()].sort_index()
        o, h, l, c = (d[k].to_numpy(float) for k in "ohlc")
        pc = np.r_[c[0], c[:-1]]
        tr = np.maximum(np.maximum(h - l, np.abs(pc - h)), np.abs(pc - l))
        DATA[sym] = (d.index, o, h, l, c, pd.Series(tr).rolling(60).mean().to_numpy(), sp)
    return DATA[sym]


def cost_r(sym, sp, t, e, stop, dr, t_out=None):
    tt = pd.DataFrame({"r": [0.0], "entry_px": [e], "stop_px": [stop], "dir": [dr], "time": [t], "exit_time": [t_out or t]})
    x = C.apply(tt, sym, sp)
    return float((x["spread_px"].iloc[0] + x["comm_px"].iloc[0] - x["swap_px"].iloc[0]) / abs(e - stop))


def walk(h, l, c, k, end, dr, stop, tgt):
    for q in range(k, end):
        if (l[q] <= stop) if dr > 0 else (h[q] >= stop):
            return -1.0, q
        if tgt is not None and ((h[q] >= tgt) if dr > 0 else (l[q] <= tgt)):
            return "T", q
    return None, end - 1


def drift(evs, t0, t1):
    rows = []
    for e in evs:
        if e["cat"] != "rates" or not any(k in e["title"].lower() for k in ("rate", "fomc statement", "funds")):
            continue
        cur = "JPY" if "boj" in e["title"].lower() else e["cur"]
        for sym in BANK_INDEX.get(cur, []):
            D = load(sym, t0, t1)
            if D is None:
                continue
            idx, o, h, l, c, atr, sp = D
            kx = idx.searchsorted(e["t"] - pd.Timedelta(minutes=5)) - 1
            if kx < 1500 or idx[kx] < e["t"] - pd.Timedelta(minutes=30):
                continue
            tz, hh, mm = OPEN[sym]
            day_before = (e["t"].tz_convert(tz) - pd.Timedelta(days=1)).normalize()
            starts = {"24h": e["t"] - pd.Timedelta(hours=24), "6h": e["t"] - pd.Timedelta(hours=6),
                      "open": (day_before + pd.Timedelta(hours=hh, minutes=mm)).tz_convert("UTC")}
            for sn, ts in starts.items():
                ks = idx.searchsorted(ts)
                if ks >= kx or idx[ks] - ts > pd.Timedelta(hours=3):
                    continue
                a = atr[ks - 1]
                if not np.isfinite(a) or a <= 0:
                    continue
                for sm in (30, 60):
                    stop = o[ks] - sm * a
                    r, q = walk(h, l, c, ks, kx + 1, 1, stop, None)
                    if r is None:
                        r = (c[kx] - o[ks]) / (o[ks] - stop)
                    r -= cost_r(sym, sp, idx[ks], o[ks], stop, 1, idx[kx])
                    rows.append({"strat": "DRIFT", "var": f"{sn}|stop{sm}", "sym": sym, "t": e["t"], "r": r})
    return rows


def box(evs, t0, t1):
    rows = []
    for e in evs:
        if e["cat"] != "jobs":
            continue
        for sym in NR.BY_CUR[e["cur"]]:
            D_ = load(sym, t0, t1)
            if D_ is None:
                continue
            idx, o, h, l, c, atr, sp = D_
            k0 = idx.searchsorted(e["t"])
            if k0 >= len(idx) or idx[k0] != e["t"] or k0 < 100 or k0 + 130 >= len(idx):
                continue
            a = atr[k0 - 1]
            if not np.isfinite(a) or a <= 0:
                continue
            for span in (10, 15, 30):
                bh, bl = h[k0 - span:k0].max(), l[k0 - span:k0].min()
                mid = (bh + bl) / 2
                for D in (2, 3):
                    for q in range(k0 + D - 1, k0 + 30):
                        dr = 1 if c[q] > bh else (-1 if c[q] < bl else 0)
                        if dr == 0:
                            continue
                        ent = o[q + 1]
                        stop = mid if dr * (ent - mid) > 3 * a else ent - dr * 3 * a
                        risk = abs(ent - stop)
                        cr = cost_r(sym, sp, idx[q + 1], ent, stop, dr)
                        agree = None
                        if e["surp"] is not None and (e["cur"], sym) in NR.SIGN:
                            agree = e["surp"] * NR.SIGN[(e["cur"], sym)] == dr
                        for tg in (1.5, 2.0, 3.0, None):
                            for H in (30, 60):
                                r, qx = walk(h, l, c, q + 1, min(q + 1 + H, len(idx)), dr, stop,
                                             None if tg is None else ent + dr * tg * risk)
                                r = tg if r == "T" else (dr * (c[qx] - ent) / risk if r is None else r)
                                rows.append({"strat": "BOX", "var": f"box{span}|D{D}|tg{tg or 0}|H{H}", "sym": sym, "t": e["t"],
                                             "r": r - cr, "agree": agree})
                        break
    return rows


def surp(evs, t0, t1):
    rows = []
    for e in evs:
        if e["cat"] != "rates" or e["surp"] is None:
            continue
        for sym in NR.BY_CUR[e["cur"]]:
            if (e["cur"], sym) not in NR.SIGN:
                continue
            D_ = load(sym, t0, t1)
            if D_ is None:
                continue
            idx, o, h, l, c, atr, sp = D_
            k0 = idx.searchsorted(e["t"])
            if k0 >= len(idx) or idx[k0] != e["t"] or k0 < 100 or k0 + 250 >= len(idx):
                continue
            a = atr[k0 - 1]
            if not np.isfinite(a) or a <= 0:
                continue
            dr = e["surp"] * NR.SIGN[(e["cur"], sym)]
            for D in (2, 3, 5):
                ke = k0 + D
                ent = o[ke]
                first = np.sign(c[ke - 1] - o[k0])
                lo, hi = l[k0:ke].min(), h[k0:ke].max()
                stop = lo if dr > 0 else hi
                if dr * (ent - stop) < 3 * a:
                    stop = ent - dr * 3 * a
                risk = abs(ent - stop)
                cr = cost_r(sym, sp, idx[ke], ent, stop, dr)
                for H in (30, 60, 120, 240):
                    r, qx = walk(h, l, c, ke, min(ke + H, len(idx)), dr, stop, None)
                    if r is None:
                        r = dr * (c[qx] - ent) / risk
                    rows.append({"strat": "SURP", "var": f"D{D}|H{H}", "sym": sym, "t": e["t"], "r": r - cr,
                                 "agree": bool(first == dr)})
    return rows


def judge(df, name):
    """Variants chosen on 2025; for BOX/SURP also the agreement filter and the 2025-positive instruments."""
    out = []
    df = df.copy()
    df["train"] = df["t"] < SPLIT
    variants = []
    for v, g in df.groupby("var"):
        variants.append((v, "all", g))
        if "agree" in g and g["agree"].notna().any():
            variants.append((v, "agree", g[g["agree"] == True]))  # noqa: E712
    for v, filt, g in variants:
        tr, va = g[g["train"]], g[~g["train"]]
        if len(tr) < 10:
            continue
        good = tr.groupby("sym")["r"].mean()
        good = list(good[good > 0].index)
        for sel in ("all", "train+"):
            a = tr if sel == "all" else tr[tr["sym"].isin(good)]
            b = va if sel == "all" else va[va["sym"].isin(good)]
            if len(a) < 10:
                continue
            ra, rb = a["r"].to_numpy(), b["r"].to_numpy()
            out.append({"strat": name, "var": v, "filter": filt, "syms": sel if sel == "all" else ",".join(good),
                        "n25": len(ra), "mean25": round(float(ra.mean()), 3),
                        "t25": round(float(ra.mean() / ra.std() * np.sqrt(len(ra))), 2) if ra.std() > 0 else None,
                        "n26": len(rb), "mean26": round(float(rb.mean()), 3) if len(rb) else None,
                        "t26": round(float(rb.mean() / rb.std() * np.sqrt(len(rb))), 2) if len(rb) > 2 and rb.std() > 0 else None})
    o = pd.DataFrame(out).sort_values("t25", ascending=False)
    return o


def main():
    evs = NR.events()
    t0 = (min(e["t"] for e in evs) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    t1 = (max(e["t"] for e in evs) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    res = []
    for name, fn in (("DRIFT", drift), ("BOX", box), ("SURP", surp)):
        rows = fn(evs, t0, t1)
        df = pd.DataFrame(rows)
        if df.empty:
            print(name, "no trades")
            continue
        j = judge(df, name)
        pd.set_option("display.width", 220)
        print(f"\n=== {name}: best 12 on 2025 (train), with 2026 (unseen)")
        print(j.head(12).to_string())
        top = j.iloc[0]
        print(f"chosen on 2025: {top['var']} {top['filter']} {top['syms']} -> 2026: {top['n26']} trades, {top['mean26']}R, t {top['t26']}")
        res.append(j.head(30).assign(chosen=[True] + [False] * (min(30, len(j)) - 1)))
    Path("reports").mkdir(exist_ok=True)
    pd.concat(res).to_csv("reports/news_improve.csv", index=False)


if __name__ == "__main__":
    main()
