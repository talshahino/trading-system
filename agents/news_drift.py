"""Bot 7 - the drift of days after a big surprise (user, 18.11.2026: "check both" - after the release's own minutes
showed no edge past FTMO's 2 minutes: a horizon where speed does not matter). The releases as agents/news_fast (FTMO's
calendar history, high impact, the lead number by its surprise; big = at least its title's 2025 median).
  entry  at minute 5 / 30 / 60 after the release, with the number (p1) or against it (p-1; an index: the side by p)
  stop   1 / 2 x the daily ATR(14) before the release
  out    after 1 / 2 / 3 / 5 days (24 / 48 / 72 / 120 hours)
FTMO's costs with the swap of every night held (r) and the spread x2 (r2). Chosen per currency on 30.04-31.12.2025
(all releases / big ones only), shown on 2026. Results only.

  python -m agents.news_drift plan | run --sym EURUSD | report
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

OUT = ROOT / "reports" / "agents" / "bot7" / "news_drift"


def plan():
    return F.plan()


def run(sym):
    OUT.mkdir(parents=True, exist_ok=True)
    rel_all, bt = F.releases(), F.betas(sym)
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
    dd = d1.resample("1D").agg({"h": "max", "l": "min", "c": "last"}).dropna()
    tr = pd.concat([dd["h"] - dd["l"], (dd["h"] - dd["c"].shift()).abs(), (dd["l"] - dd["c"].shift()).abs()], axis=1).max(axis=1)
    atr_d = tr.rolling(14).mean()
    n = len(idx)
    raw = []
    for cur, b in bt.items():
        for t, x in sorted((rel_all.get(cur) or {}).items()):
            if t < F.TEST_FROM:
                continue
            ka = atr_d.index.searchsorted(t.floor("D")) - 1
            if ka < 14:
                continue
            atr = float(atr_d.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            num = int(x["g"] * (b or 1)) or 1
            for N in (5, 30, 60):
                ke = idx.searchsorted(t + pd.Timedelta(minutes=N))
                if ke >= n or idx[ke] - t > pd.Timedelta(minutes=N + 5):
                    continue
                e = o[ke]
                for p in (1, -1):
                    dr = num * p
                    for s in (1, 2):
                        stop = e - dr * s * atr
                        for hh in (24, 48, 72, 120):
                            end = idx.searchsorted(idx[ke] + pd.Timedelta(hours=hh))
                            if end >= n:
                                continue
                            seg = l[ke:end] if dr > 0 else h[ke:end]
                            hit = np.flatnonzero((seg <= stop) if dr > 0 else (seg >= stop))
                            if len(hit):
                                q = ke + int(hit[0])
                                ex = stop if q == ke else (min(stop, o[q]) if dr > 0 else max(stop, o[q]))
                            else:
                                q, ex = end - 1, c[end - 1]
                            raw.append({"v": f"n{N}|p{p}|s{s}|h{hh}", "time": idx[ke], "exit_time": idx[q], "dir": dr,
                                        "entry_px": e, "stop_px": stop, "r": dr * (ex - e) / (s * atr), "cur": cur,
                                        "big": x["big"], "mixed": x["mixed"], "title": x["title"], "rel": t})
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
    df["way"] = "h" + df["v"].str.extract(r"\|h(\d+)$")[0]
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
