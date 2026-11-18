"""Bot 7 - five ways to use a release AFTER FTMO's 2 restricted minutes (user, 18.11.2026, voice: "what happens after 2
minutes? how can we use the momentum, the big volume after a release, in the minute FTMO lets us in?"; the plain
momentum entry at 1-3 minutes earned ~0 on 2025 and lost on 2026 - agents/news_fast.py). No entry before minute 2.
  pb     the pullback: the first 2 minutes' move sets the side; a limit order at 38.2 / 50 / 61.8% back of the move
         (until minute 30), the stop beyond the release minute's open + 0.25 ATR, out after 30 / 60 minutes
  fade   against an overreaction: the first 2 minutes moved at least K x the 5-minute ATR (K 2 / 3 / 4) per unit of
         surprise (the move / the surprise's size against its title's 2025 median, at least 0.5); entry at minute 2
         against it, the stop beyond the 2 minutes' extreme + 0.5 ATR, target = half way back, out after 15 / 30 / 60
  orb    the range breakout: the high / low of the first 5 / 15 minutes; a stop order beyond them (until minute 60),
         the stop at the range's other side / its middle, out after 30 / 60 / 120 minutes
  mixed  releases with numbers pointing both ways (e.g. jobs vs wages): at minute 5 / 15 in the side the price chose,
         stop 1.5 ATR, out after 30 / 60
  day    the big surprise's day: at minute 2 / 5 with the number, stop 3 / 6 ATR, out after 4 / 8 hours
Every way also against (p-1) and, for pb / orb, only when the price's side agrees with the number (agree). Index side
of a "good" number chosen by the p knob. Costs: FTMO's spread at the entry minute + commission (r) and the spread x2
(r2: the minutes after a release). Chosen per currency x way on 30.04-31.12.2025, shown on 2026. Results only.

  python -m agents.news_after plan | run --sym EURUSD | report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import ftmo_costs as C  # noqa: E402
from agents import news_fast as F  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import news_surprise as NS  # noqa: E402
from agents import poc_agent as P  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "news_after"
WAYS = ("pb", "fade", "orb", "mixed", "day")


def plan():
    return F.plan()


def run(sym):
    OUT.mkdir(parents=True, exist_ok=True)
    rel_all, bt = F.releases(), F.betas(sym)
    zs = _zs()
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = P.bars(sym, "1Min", NF.LOAD_FROM, now)
    if d1 is None or d1.empty:
        raise SystemExit(f"no bars for {sym}")
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, NF.LOAD_FROM, now)
        sp = sp[~sp.index.duplicated()].sort_index() if sp is not None and len(sp) else None
    except Exception:  # noqa: BLE001
        sp = None
    idx = d1.index
    o, h, l, c = (d1[x].to_numpy(float) for x in "ohlc")
    a5 = NS._atr5(d1)
    n = len(idx)
    raw = []

    def at(t, m):
        k = idx.searchsorted(t + pd.Timedelta(minutes=m))
        return k if k < n and idx[k] - t <= pd.Timedelta(minutes=m + 2) else None

    def walk(ke, e, dr, stop, end, tgt=None):
        """From bar ke (filled at e) to bar end (exclusive): (exit px, exit bar)."""
        end = min(end, n)
        for q in range(ke, end):
            if (l[q] <= stop) if dr > 0 else (h[q] >= stop):
                if q == ke:
                    return stop, q
                return (min(stop, o[q]) if dr > 0 else max(stop, o[q])), q
            if tgt is not None and q > ke and ((h[q] >= tgt) if dr > 0 else (l[q] <= tgt)):
                return tgt, q
        return c[end - 1], end - 1

    def add(v, ke, e, dr, stop, ex, qx, t, cur, x):
        if dr * (e - stop) <= 0:
            return
        raw.append({"v": v, "time": idx[ke], "exit_time": idx[qx], "dir": dr, "entry_px": e, "stop_px": stop,
                    "r": dr * (ex - e) / abs(e - stop), "cur": cur, "big": x["big"], "mixed": x["mixed"],
                    "title": x["title"], "rel": t})

    for cur, b in bt.items():
        for t, x in sorted((rel_all.get(cur) or {}).items()):
            if t < F.TEST_FROM:
                continue
            k0, k2 = at(t, 0), at(t, 2)
            if k0 is None or k2 is None or k0 < 30 or k0 + 500 >= n:
                continue
            ka = a5.index.searchsorted(t) - 1
            if ka < 14:
                continue
            atr = float(a5.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            num = int(x["g"] * (b or 1))                          # the number's side on this instrument
            o0 = o[k0]
            m2 = c[k2 - 1] - o0                                   # the first 2 minutes' move
            sd = 1 if m2 > 0 else -1 if m2 < 0 else 0
            for p in (1, -1):
                # pb
                if sd:
                    ext = h[k0:k2].max() if sd > 0 else l[k0:k2].min()
                    for fr in (0.382, 0.5, 0.618):
                        lvl = ext - sd * fr * abs(ext - o0)
                        for agree in (0, 1):
                            if agree and sd != num:
                                continue
                            dr = sd * p
                            if p < 0:
                                continue                          # a pullback is taken with the move only
                            ke = None
                            for q in range(k2, min(n, k2 + 28)):
                                if (l[q] <= lvl) if sd > 0 else (h[q] >= lvl):
                                    ke = q
                                    break
                            if ke is None:
                                continue
                            e = min(lvl, o[ke]) if dr > 0 else max(lvl, o[ke])
                            stop = o0 - dr * 0.25 * atr
                            for hm in (30, 60):
                                ex, qx = walk(ke, e, dr, stop, ke + hm)
                                add(f"pb|f{fr}|a{agree}|h{hm}", ke, e, dr, stop, ex, qx, t, cur, x)
                # fade
                if sd and p > 0:
                    z = max(0.5, zs.get((cur, x["title"], t), 1.0))
                    over = abs(m2) / atr / z
                    ext = h[k0:k2].max() if sd > 0 else l[k0:k2].min()
                    for K in (2, 3, 4):
                        if over < K:
                            continue
                        dr, e = -sd, o[k2]
                        stop = ext + sd * 0.5 * atr
                        tgt = e + dr * 0.5 * abs(e - o0)
                        for hm in (15, 30, 60):
                            ex, qx = walk(k2, e, dr, stop, k2 + hm, tgt)
                            add(f"fade|k{K}|h{hm}", k2, e, dr, stop, ex, qx, t, cur, x)
                # orb
                for N in (5, 15):
                    kn = at(t, N)
                    if kn is None:
                        continue
                    hi, lo = h[k0:kn].max(), l[k0:kn].min()
                    ke, dr = None, 0
                    for q in range(kn, min(n, kn + 60 - N)):
                        up, dn = h[q] > hi, l[q] < lo
                        if up and dn:
                            break
                        if up or dn:
                            ke, dr = q, (1 if up else -1)
                            break
                    if ke is None:
                        continue
                    for agree in (0, 1):
                        if agree and dr != num:
                            continue
                        d_ = dr * p
                        e = (max(hi, o[ke]) if dr > 0 else min(lo, o[ke]))
                        for sk in ("side", "mid"):
                            stop = (lo if d_ > 0 else hi) if sk == "side" else (hi + lo) / 2
                            if p < 0:
                                stop = e - d_ * abs(hi - lo) / 2
                            for hm in (30, 60, 120):
                                ex, qx = walk(ke, e, d_, stop, ke + hm)
                                add(f"orb|n{N}|{sk}|a{agree}|p{p}|h{hm}", ke, e, d_, stop, ex, qx, t, cur, x)
                # mixed
                if x["mixed"]:
                    for N in (5, 15):
                        kn = at(t, N)
                        if kn is None:
                            continue
                        mv = c[kn - 1] - o0
                        if not mv:
                            continue
                        dr = (1 if mv > 0 else -1) * p
                        e = o[kn]
                        stop = e - dr * 1.5 * atr
                        for hm in (30, 60):
                            ex, qx = walk(kn, e, dr, stop, kn + hm)
                            add(f"mixed|n{N}|p{p}|h{hm}", kn, e, dr, stop, ex, qx, t, cur, x)
                # day
                for N in (2, 5):
                    kn = at(t, N)
                    if kn is None:
                        continue
                    dr = num * p
                    e = o[kn]
                    for s in (3, 6):
                        stop = e - dr * s * atr
                        for hh in (4, 8):
                            end = idx.searchsorted(idx[kn] + pd.Timedelta(hours=hh))
                            ex, qx = walk(kn, e, dr, stop, max(end, kn + 1))
                            add(f"day|n{N}|s{s}|p{p}|h{hh}", kn, e, dr, stop, ex, qx, t, cur, x)
    if not raw:
        (OUT / f"{sym}.json").write_text("[]", encoding="utf-8")
        return
    tt = pd.DataFrame(raw)
    net = C.apply(tt.copy(), sym, sp)
    sp2 = (sp * 2) if sp is not None and len(sp) else None
    net2 = C.apply(tt.copy(), sym, sp2) if sp2 is not None else net.assign(r=tt["r"] - 2 * (tt["r"] - net["r"]))
    rows = [{"v": x.v, "sym": sym, "cur": x.cur, "t": pd.Timestamp(x.time).isoformat(), "r": round(float(x.r), 4),
             "r2": round(float(net2["r"].iloc[k]), 4), "big": int(x.big), "title": x.title}
            for k, x in enumerate(net.itertuples())]
    (OUT / f"{sym}.json").write_text(json.dumps(rows), encoding="utf-8")
    print(sym, len(rows), "rows", flush=True)


def _zs():
    """(currency, title, time) -> the lead number's surprise against its title's 2025 median."""
    items = json.loads((ROOT / "reports" / "agents" / "bot7" / "calendar_full.json").read_text(encoding="utf-8"))["items"]
    rows = []
    for i in items:
        cur = str(i.get("instrument", "")).split(" ")[0]
        a, f = NF._num(i.get("actual")), NF._num(i.get("forecast"))
        if cur not in F.CURS or i.get("impact") != "high" or a is None or f is None or a == f:
            continue
        rows.append((cur, i["title"], pd.Timestamp(i["date"]).tz_convert("UTC"), abs(a - f)))
    med = {}
    for cur, ti, t, v in rows:
        if t < F.SPLIT:
            med.setdefault((cur, ti), []).append(v)
    med = {k: float(np.median(v)) for k, v in med.items()}
    return {(cur, ti, t): v / med[(cur, ti)] for cur, ti, t, v in rows if med.get((cur, ti))}


def _st(s):
    return {"n": int(len(s)), "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 3) if len(s) else None}


def report():
    rows = []
    for f in sorted(OUT.glob("*.json")):
        if f.name != "report.json":
            rows += json.loads(f.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "rows": int(len(df))}
    if not len(df):
        (OUT / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        return
    df["train"] = pd.to_datetime(df["t"], utc=True) < F.SPLIT
    df["way"] = df["v"].str.split("|").str[0]
    out, tot = {}, {}
    for (way, cur), g in df.groupby(["way", "cur"]):
        for cost in ("r", "r2"):
            best = None
            for v, y in g.groupby("v"):
                for big in (0, 1):
                    z = y[y["big"] == 1] if big else y
                    tr, un = z[z.train][cost], z[~z.train][cost]
                    if len(tr) < 20:
                        continue
                    k = {"v": v, "big": big, "train": _st(tr), "unseen": _st(un)}
                    if best is None or k["train"]["sum"] > best["train"]["sum"]:
                        best = k
            if not best:
                continue
            out.setdefault(way, {}).setdefault(cur, {})[cost] = best
            t_ = tot.setdefault(way, {}).setdefault(cost, {"train": 0.0, "unseen": 0.0, "n_unseen": 0, "pos26": 0, "cur": 0})
            t_["train"] = round(t_["train"] + best["train"]["sum"], 2)
            t_["unseen"] = round(t_["unseen"] + best["unseen"]["sum"], 2)
            t_["n_unseen"] += best["unseen"]["n"]
            t_["pos26"] += int(best["unseen"]["sum"] > 0)
            t_["cur"] += 1
    rep["by_way"], rep["all"] = out, tot
    # per release type: the best version of each way on that type, chosen on 2025 (n >= 10), shown on 2026
    typ = {}
    for (cur, title), g in df.groupby(["cur", "title"]):
        for v, y in g.groupby("v"):
            tr, un = y[y.train]["r"], y[~y.train]["r"]
            if len(tr) < 10:
                continue
            k = typ.get(f"{cur} {title}")
            if k is None or _st(tr)["sum"] > k["train"]["sum"]:
                typ[f"{cur} {title}"] = {"v": v, "train": _st(tr), "unseen": _st(un)}
    rep["by_type"] = typ
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tot, indent=1))
    for way, cs in out.items():
        for cur, b in cs.items():
            x = b["r"]
            print(way, cur, x["v"], x["big"], x["train"], x["unseen"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(plan()))
    elif a.mode == "run":
        run(a.sym)
    else:
        report()
