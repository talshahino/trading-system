"""How fast to enter after a release, on every currency (user, 17.11.2026: "enter in the release's own seconds - in
the direction the number points: worse for the dollar = buy EURUSD at 15:30:02", then "not only a few instruments -
every pair of the yen, the pound, the dollar, the euro, the franc ... and the countries' indices", and two ways:
a regular broker (no news rule: enter at once) and FTMO's rules (no trade inside the 2 minutes of a restricted
release: enter after 2 minutes)).

The releases: FTMO's calendar history with the actuals (reports/agents/bot7/calendar_full.json, from 03.2025),
high impact, with an actual and a forecast that differ. Per currency (USD EUR GBP JPY CHF CAD AUD NZD) and minute:
the lead number = the biggest surprise against its title's median surprise on 2025; "good" = better for that
economy than expected (claims / unemployment / inventories: lower is better); "big" = at least the title's 2025
median; "mixed" = numbers of the same minute pointing both ways.
The instruments: every FTMO currency pair holding the currency (base: with its strength, quote: against), gold for
the dollar (against), and the country's indices (side chosen on 2025: good numbers can lift or sink an index).
Entry at the open of minute 0 (the release's own minute - an UPPER BOUND: no one knows the number at its first
tick; if this does not earn, nothing faster will), 1, 2 or 3 after it; stop 1.5 / 3 x the 5-minute ATR before the
release; out after 15 / 30 / 60 minutes; side p1 (with the number) / p-1 (against).
Costs: FTMO's (Dukascopy's spread at the entry minute + commission, agents/ftmo_costs.py) and "broker" = the same
with the spread x3 at the entry (the spread around a release is several times its usual).
Chosen on 30.04.2025-31.12.2025, shown on 2026. Nothing in the bots changes here.

  python -m agents.news_fast plan
  python -m agents.news_fast run --sym EURUSD
  python -m agents.news_fast report
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

from agents import rules_agent as R  # noqa: E402,F401
from agents import ftmo_costs as C  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import news_surprise as NS  # noqa: E402
from agents import poc_agent as P  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "news_fast"
CURS = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "EURCHF",
      "AUDJPY", "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "GBPAUD", "GBPCAD", "GBPCHF", "AUDCAD", "AUDNZD", "NZDJPY",
      "AUDCHF", "CADCHF", "EURNZD", "GBPNZD"]
INDEX = {"USD": ["US100", "US500", "US30"], "EUR": ["GER40", "EU50"], "GBP": ["UK100"], "JPY": ["JP225"], "AUD": ["AUS200"]}
DELAYS = (0, 1, 2, 3)
STOPS = (1.5, 3.0)
HOLDS = (15, 30, 60)
POLS = (1, -1)
SPAN = 70
SPLIT, TEST_FROM = NF.SPLIT, NF.TEST_FROM


def betas(sym):
    """currency -> +1 (the instrument goes up when the currency is strong) / -1 / 0 (index: the side is chosen)."""
    if sym in FX:
        return {sym[:3]: 1, sym[3:]: -1}
    if sym == "XAUUSD":
        return {"USD": -1}
    return {c: 1 for c, xs in INDEX.items() if sym in xs}


def plan():
    return FX + ["XAUUSD"] + sorted({s for xs in INDEX.values() for s in xs})


def releases():
    items = json.loads((ROOT / "reports" / "agents" / "bot7" / "calendar_full.json").read_text(encoding="utf-8"))["items"]
    by = {}
    for i in items:
        cur = str(i.get("instrument", "")).split(" ")[0]
        if cur not in CURS or i.get("impact") != "high":
            continue
        a, f = NF._num(i.get("actual")), NF._num(i.get("forecast"))
        if a is None or f is None or a == f:
            continue
        t = pd.Timestamp(i["date"]).tz_convert("UTC")
        g = -NS.bad_sign(i["title"], a, f)                     # +1: better for the economy than expected
        by.setdefault((cur, t), []).append({"title": i["title"], "g": g, "abs": abs(a - f)})
    med = {}
    for (cur, t), xs in by.items():
        if t < SPLIT:
            for x in xs:
                med.setdefault((cur, x["title"]), []).append(x["abs"])
    med = {k: float(np.median(v)) for k, v in med.items() if v}
    out = {}
    for (cur, t), xs in by.items():
        for x in xs:
            m = med.get((cur, x["title"]))
            x["z"] = x["abs"] / m if m else 1.0
        lead = max(xs, key=lambda x: x["z"])
        out.setdefault(cur, {})[t] = {"title": lead["title"], "g": lead["g"], "big": int(lead["z"] >= 1.0),
                                      "mixed": int(len({x["g"] for x in xs}) > 1)}
    return out


def run(sym):
    OUT.mkdir(parents=True, exist_ok=True)
    rel_all, bt = releases(), betas(sym)
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
    raw, moves = [], []
    for cur, b in bt.items():
        for t, x in sorted((rel_all.get(cur) or {}).items()):
            if t < TEST_FROM:
                continue
            k0 = idx.searchsorted(t)
            if k0 >= len(idx) or k0 < 30 or k0 + SPAN >= len(idx) or idx[k0 + SPAN] - idx[k0] > pd.Timedelta(minutes=SPAN + 5):
                continue
            ka = a5.index.searchsorted(t) - 1
            if ka < 14:
                continue
            atr = float(a5.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            # the release's impact on this instrument (user, 17.11.2026: "the impact of each type of release"): the
            # move from the release minute's open after 1 / 5 / 15 / 60 minutes in 5-minute ATRs, signed so that + is
            # the way the number points (the currency's strength; an index: + = up after a good number), and the
            # biggest move either way in the first 15 minutes
            e0 = o[k0]
            mv = {}
            for m in (1, 5, 15, 60):
                km = min(len(idx) - 1, idx.searchsorted(t + pd.Timedelta(minutes=m)) - 1)
                mv[f"m{m}"] = round(float(x["g"] * (b or 1) * (c[km] - e0) / atr), 3)
            k15 = max(k0 + 1, min(len(idx), idx.searchsorted(t + pd.Timedelta(minutes=15))))
            mv["range15"] = round(float((h[k0:k15].max() - l[k0:k15].min()) / atr), 3)
            moves.append({"sym": sym, "cur": cur, "title": x["title"], "t": t.isoformat(), "big": x["big"], **mv})
            for dl in DELAYS:
                ke = idx.searchsorted(t + pd.Timedelta(minutes=dl))
                if ke >= len(idx) or idx[ke] - t > pd.Timedelta(minutes=dl + 2):
                    continue
                e = o[ke]
                for p in POLS:
                    dr = int(x["g"] * (b or 1) * p)
                    for s in STOPS:
                        stop = e - dr * s * atr
                        for hm in HOLDS:
                            end = min(max(idx.searchsorted(idx[ke] + pd.Timedelta(minutes=hm)), ke + 1), len(idx))
                            px, qx = c[end - 1], end - 1
                            for q in range(ke, end):
                                if (l[q] <= stop) if dr > 0 else (h[q] >= stop):
                                    px, qx = stop, q
                                    break
                            raw.append({"v": f"d{dl}|s{s:g}|h{hm}|p{p}", "time": idx[ke], "exit_time": idx[qx], "dir": dr,
                                        "entry_px": e, "stop_px": stop, "r": dr * (px - e) / (s * atr), "rel": t, "cur": cur,
                                        "idx": int(b == 0 or sym not in FX and sym != "XAUUSD"), "big": x["big"], "mixed": x["mixed"],
                                        "title": x["title"]})
    (OUT / f"{sym}.moves.json").write_text(json.dumps(moves), encoding="utf-8")
    if not raw:
        (OUT / f"{sym}.json").write_text("[]", encoding="utf-8")
        return
    tt = pd.DataFrame(raw)
    net = C.apply(tt.copy(), sym, sp)
    sp3 = (sp * 3) if sp is not None and len(sp) else None
    net3 = C.apply(tt.copy(), sym, sp3) if sp3 is not None else None
    if net3 is None:                                           # no spread history: the estimate x3 by hand
        cost = tt["r"] - net["r"]
        net3 = net.assign(r=tt["r"] - 3 * cost)
    rows = [{"v": x.v, "sym": sym, "cur": x.cur, "t": pd.Timestamp(x.time).isoformat(), "r": round(float(x.r), 4),
             "r3": round(float(net3["r"].iloc[k]), 4), "gross": round(float(tt["r"].iloc[k]), 4), "big": int(x.big),
             "mixed": int(x.mixed), "title": x.title} for k, x in enumerate(net.itertuples())]
    (OUT / f"{sym}.json").write_text(json.dumps(rows), encoding="utf-8")
    print(sym, len(rows), "rows")


def _st(s):
    return {"n": int(len(s)), "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 3) if len(s) else None}


def report():
    rows = []
    for f in sorted(OUT.glob("*.json")):
        if f.name == "report.json" or f.name.endswith(".moves.json"):
            continue
        try:
            rows += json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    df = pd.DataFrame(rows)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "rows": int(len(df)),
           "instruments": sorted(df["sym"].unique().tolist()) if len(df) else []}
    if not len(df):
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    df["train"] = pd.to_datetime(df["t"], utc=True) < SPLIT
    df["d"] = df["v"].str.extract(r"^d(\d+)")[0].astype(int)
    # per currency x delay x cost: the best version (stop / hold / side / only big surprises / without mixed minutes)
    # chosen on 2025 by total R, and how it did on 2026
    out = {}
    for (cur, dl), g in df.groupby(["cur", "d"]):
        for cost in ("r", "r3"):
            best = None
            for v, y in g.groupby("v"):
                for big in (0, 1):
                    for nomix in (0, 1):
                        z = y[(y["big"] == 1) if big else y.index == y.index]
                        z = z[z["mixed"] == 0] if nomix else z
                        tr, un = z[z.train][cost], z[~z.train][cost]
                        if len(tr) < 20:
                            continue
                        k = {"v": v, "big": big, "nomix": nomix, "train": _st(tr), "unseen": _st(un)}
                        if best is None or k["train"]["sum"] > best["train"]["sum"]:
                            best = k
            out.setdefault(cur, {}).setdefault(f"d{dl}", {})["ftmo" if cost == "r" else "broker"] = best
    rep["by_currency"] = out
    # every currency together, per delay and cost: each currency's chosen version summed (the choice on 2025 only)
    tot = {}
    for cur, ds in out.items():
        for dk, cs in ds.items():
            for ck, b in cs.items():
                if not b:
                    continue
                t_ = tot.setdefault(dk, {}).setdefault(ck, {"train": 0.0, "unseen": 0.0, "n_unseen": 0, "positive_2026": 0, "currencies": 0})
                t_["train"] = round(t_["train"] + b["train"]["sum"], 2)
                t_["unseen"] = round(t_["unseen"] + b["unseen"]["sum"], 2)
                t_["n_unseen"] += b["unseen"]["n"]
                t_["positive_2026"] += int((b["unseen"]["sum"] or 0) > 0)
                t_["currencies"] += 1
    rep["all"] = tot
    # every type of release (user, 17.11.2026: "how the strategies behave by each type of release, and the impact of
    # each type"): its impact (moves.json) and how each entry delay did on it - the version chosen for its currency
    mv = []
    for f in sorted(OUT.glob("*.moves.json")):
        try:
            mv += json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    mdf = pd.DataFrame(mv)
    types = {}
    if len(mdf):
        for (cur, title), g in mdf.groupby(["cur", "title"]):
            types[f"{cur} {title}"] = {
                "releases": int(g["t"].nunique()),
                "impact_range15_atr": round(float(g["range15"].median()), 2),
                "with_number_5m_pct": round(100 * float((g["m5"] > 0).mean()), 1),
                "move_atr": {m: round(float(g[m].median()), 2) for m in ("m1", "m5", "m15", "m60")}}
    for key, t_ in types.items():
        cur, title = key.split(" ", 1)
        for dl in DELAYS:
            for ck, cost in (("ftmo", "r"), ("broker", "r3")):
                b = ((out.get(cur) or {}).get(f"d{dl}") or {}).get(ck)
                if not b:
                    continue
                z = df[(df["cur"] == cur) & (df["title"] == title) & (df["d"] == dl) & (df["v"] == b["v"])]
                z = z[z["big"] == 1] if b["big"] else z
                z = z[z["mixed"] == 0] if b["nomix"] else z
                t_.setdefault("by_delay", {}).setdefault(f"d{dl}", {})[ck] = {"train": _st(z[z.train][cost]), "unseen": _st(z[~z.train][cost])}
    rep["by_type"] = dict(sorted(types.items(), key=lambda kv: -kv[1]["impact_range15_atr"]))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(tot, ensure_ascii=False, indent=1))
    for cur, ds in out.items():
        print(cur, {dk: {ck: (b["v"], b["big"], b["nomix"], b["train"]["sum"], b["unseen"]["sum"], b["unseen"]["n"]) if b else None
                         for ck, b in cs.items()} for dk, cs in ds.items()})


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
