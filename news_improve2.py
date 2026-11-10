"""Round 2 of improving the news candidates (user, 10.11.2026: "more ways to improve - timeframes, entry, stop, TP").
Research only. Same data and costs as news_improve.py.

JOBS BOX (jobs releases, the box before the release, breakout after it). Grid:
  box        10 / 15 / 20 / 30 minutes before the release
  tf         the breakout is a close outside the box of a 1 / 3 / 5-minute candle (from minute 3: FTMO-safe)
  entry      market (the next minute's open) / retest (a limit at the box edge, within 20 minutes of the breakout)
  stop       box middle / the box's other side / 3 x or 6 x the 1-minute ATR(60)
  target     1 / 1.5 / 2 / 3 R, or 1 x the box height beyond the edge, or none
  hold       30 / 60 / 120 minutes
  breakeven  none / stop to entry (+ costs) at 1R
RATES SURPRISE (rates releases with a forecast). Grid:
  D 2 / 3 / 5, agreement filter on / off, stop: first-minutes extreme / 3 x / 6 x ATR, target 2 / 3R / none,
  hold 60 / 120 / 240 / to the US close (16:00 New York)
Fair method: every variant is ranked on 2025 only; reported with 2026 (never seen). Also the robustness check:
of the 20 best variants on 2025, how many made money in 2026 and their average (if the edge is real most do).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import news_improve as NI
import news_research as NR

SPLIT = NI.SPLIT


def jobs_box(evs, t0, t1):
    rows = []
    for e in evs:
        if e["cat"] != "jobs":
            continue
        for sym in NR.BY_CUR[e["cur"]]:
            D_ = NI.load(sym, t0, t1)
            if D_ is None:
                continue
            idx, o, h, l, c, atr, sp = D_
            k0 = idx.searchsorted(e["t"])
            if k0 >= len(idx) or idx[k0] != e["t"] or k0 < 100 or k0 + 200 >= len(idx):
                continue
            a = atr[k0 - 1]
            if not np.isfinite(a) or a <= 0:
                continue
            for span, tf in itertools.product((10, 15, 20, 30), (1, 3, 5)):
                bh, bl = h[k0 - span:k0].max(), l[k0 - span:k0].min()
                height, mid = bh - bl, (bh + bl) / 2
                # first candle of tf minutes (aligned to the release) closing outside, ending at minute >= 3
                kb, dr = None, 0
                for start in range(k0, k0 + 30, tf):
                    end = start + tf - 1
                    if end - k0 + 1 < 3:
                        continue
                    if c[end] > bh:
                        kb, dr = end, 1
                        break
                    if c[end] < bl:
                        kb, dr = end, -1
                        break
                if kb is None:
                    continue
                edge = bh if dr > 0 else bl
                for entry in ("market", "retest"):
                    if entry == "market":
                        ke, ent = kb + 1, o[kb + 1]
                    else:
                        ke = None
                        for q in range(kb + 1, min(kb + 21, len(idx))):
                            if (l[q] <= edge) if dr > 0 else (h[q] >= edge):
                                ke = q
                                break
                        if ke is None:
                            continue
                        ent = min(o[ke], edge) if dr > 0 else max(o[ke], edge)
                    for stn in ("mid", "other", "atr3", "atr6"):
                        stop = {"mid": mid, "other": bl if dr > 0 else bh,
                                "atr3": ent - dr * 3 * a, "atr6": ent - dr * 6 * a}[stn]
                        if dr * (ent - stop) < 2 * a:
                            stop = ent - dr * 2 * a
                        risk = abs(ent - stop)
                        cr = NI.cost_r(sym, sp, idx[ke], ent, stop, dr)
                        for tgn in ("1", "1.5", "2", "3", "boxH", "none"):
                            tg = {"none": None, "boxH": edge + dr * height}.get(tgn)
                            if tgn not in ("none", "boxH"):
                                tg = ent + dr * float(tgn) * risk
                            if tg is not None and dr * (tg - ent) <= 0:
                                continue
                            for H in (30, 60, 120):
                                for be in (False, True):
                                    r = sim(h, l, c, ke, min(ke + H, len(idx)), dr, ent, stop, tg, risk, be, cr)
                                    rows.append({"var": f"box{span}|tf{tf}|{entry}|{stn}|tg{tgn}|H{H}|be{int(be)}",
                                                 "sym": sym, "t": e["t"], "r": r})
    return rows


def sim(h, l, c, k, end, dr, ent, stop, tg, risk, be, cr):
    st = stop
    for q in range(k, end):
        if (l[q] <= st) if dr > 0 else (h[q] >= st):
            return dr * (st - ent) / risk - cr
        if tg is not None and ((h[q] >= tg) if dr > 0 else (l[q] <= tg)):
            return dr * (tg - ent) / risk - cr
        if be and dr * ((h[q] if dr > 0 else l[q]) - ent) >= risk:
            st = ent + dr * cr * risk                      # entry + costs
    return dr * (c[end - 1] - ent) / risk - cr


def rates_surp(evs, t0, t1):
    rows = []
    for e in evs:
        if e["cat"] != "rates" or e["surp"] is None:
            continue
        for sym in NR.BY_CUR[e["cur"]]:
            if (e["cur"], sym) not in NR.SIGN:
                continue
            D_ = NI.load(sym, t0, t1)
            if D_ is None:
                continue
            idx, o, h, l, c, atr, sp = D_
            k0 = idx.searchsorted(e["t"])
            if k0 >= len(idx) or idx[k0] != e["t"] or k0 < 100 or k0 + 300 >= len(idx):
                continue
            a = atr[k0 - 1]
            if not np.isfinite(a) or a <= 0:
                continue
            dr = e["surp"] * NR.SIGN[(e["cur"], sym)]
            ny_close = (e["t"].tz_convert("America/New_York").normalize() + pd.Timedelta(hours=16)).tz_convert("UTC")
            kc = idx.searchsorted(ny_close)
            for D in (2, 3, 5):
                ke = k0 + D
                ent = o[ke]
                agree = bool(np.sign(c[ke - 1] - o[k0]) == dr)
                for stn in ("first", "atr3", "atr6"):
                    stop = {"first": l[k0:ke].min() if dr > 0 else h[k0:ke].max(),
                            "atr3": ent - dr * 3 * a, "atr6": ent - dr * 6 * a}[stn]
                    if dr * (ent - stop) < 2 * a:
                        stop = ent - dr * 2 * a
                    risk = abs(ent - stop)
                    cr = NI.cost_r(sym, sp, idx[ke], ent, stop, dr)
                    for tgn in ("2", "3", "none"):
                        tg = None if tgn == "none" else ent + dr * float(tgn) * risk
                        for H in ("60", "120", "240", "close"):
                            end = (kc if kc > ke + 5 else ke + 240) if H == "close" else ke + int(H)
                            r = sim(h, l, c, ke, min(end, len(idx)), dr, ent, stop, tg, risk, False, cr)
                            for filt in ("all", "agree"):
                                if filt == "agree" and not agree:
                                    continue
                                rows.append({"var": f"D{D}|{filt}|{stn}|tg{tgn}|H{H}", "sym": sym, "t": e["t"], "r": r})
    return rows


def judge(df, name, min_n=15):
    df = df.copy()
    df["train"] = df["t"] < SPLIT
    out = []
    for v, g in df.groupby("var"):
        tr, va = g[g["train"]], g[~g["train"]]
        good = tr.groupby("sym")["r"].mean()
        good = sorted(good[good > 0].index)
        for sel in ("all", "train+"):
            a = tr if sel == "all" else tr[tr["sym"].isin(good)]
            b = va if sel == "all" else va[va["sym"].isin(good)]
            if len(a) < min_n:
                continue
            ra, rb = a["r"].to_numpy(), b["r"].to_numpy()
            out.append({"var": v, "syms": "all" if sel == "all" else ",".join(good), "n25": len(ra),
                        "mean25": round(float(ra.mean()), 3),
                        "t25": round(float(ra.mean() / ra.std() * np.sqrt(len(ra))), 2) if ra.std() > 0 else 0,
                        "n26": len(rb), "mean26": round(float(rb.mean()), 3) if len(rb) else None,
                        "t26": round(float(rb.mean() / rb.std() * np.sqrt(len(rb))), 2) if len(rb) > 2 and rb.std() > 0 else None})
    o = pd.DataFrame(out).sort_values("t25", ascending=False)
    top = o.head(20)
    print(f"\n=== {name}: {len(o)} variants. Best 15 on 2025, with 2026 (unseen)")
    print(o.head(15).to_string())
    print(f"robustness: of the 20 best on 2025, {int((top['mean26'] > 0).sum())} made money in 2026; "
          f"their 2026 average {top['mean26'].mean():+.3f}R (all variants' 2026 average {o['mean26'].mean():+.3f}R)")
    c = o.iloc[0]
    print(f"chosen on 2025: {c['var']} [{c['syms']}] -> 2026: {c['n26']} trades, {c['mean26']}R, t {c['t26']}")
    return o


def main():
    evs = NR.events()
    t0 = (min(e["t"] for e in evs) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    t1 = (max(e["t"] for e in evs) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pd.set_option("display.width", 230)
    a = judge(pd.DataFrame(jobs_box(evs, t0, t1)), "JOBS BOX")
    b = judge(pd.DataFrame(rates_surp(evs, t0, t1)), "RATES SURPRISE", min_n=8)
    pd.concat([a.head(50).assign(strat="JOBS BOX"), b.head(50).assign(strat="RATES SURPRISE")]).to_csv(
        "reports/news_improve2.csv", index=False)


if __name__ == "__main__":
    main()
