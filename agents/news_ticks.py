"""Entering a release within seconds, on real ticks (user, 17.11.2026: "enter at 15:30:02 in the direction of the
number"; agents/news_fast.py showed every R is in the release's first minute, but 1-minute bars cannot tell what
2 seconds would get; "only ticks around the releases, not the whole time").

Dukascopy's free tick history (bid + ask of every price change), only the hour of each release and the next one
(datafeed.dukascopy.com/<INSTRUMENT>/<year>/<month-1>/<day>/<hour>h_ticks.bi5). The releases and their direction
exactly as news_fast (FTMO's calendar history, high impact, the lead number by surprise size; + = better for the
economy than expected), on the instruments that moved most: AUD, CAD, NZD, GBP and the big USD releases.
  entry     the first tick at least S seconds after the release minute (S = 1, 2, 5, 10, 30, 60), a buy at the ask,
            a sell at the bid - the real spread of that second is paid
  stop      1.5 / 3 x the 5-minute ATR(14) before the release, from the entry; touched by the bid (a buy) / ask (a sell)
  exit      after 15 / 30 / 60 minutes at the bid / ask of that moment
  side      with the number (p1)
Costs: the ticks' own spread + FTMO's commission. No latency of the order itself (an order sent at S may fill later -
the result is still a bound, a tighter one). Chosen on 30.04.2025-31.12.2025, shown on 2026. Results only.

  python -m agents.news_ticks plan
  python -m agents.news_ticks run --sym EURUSD
  python -m agents.news_ticks report
"""
from __future__ import annotations

import argparse
import json
import lzma
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import news_fast as F  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import news_surprise as NS  # noqa: E402
from agents import poc_agent as P  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "news_ticks"
CACHE = ROOT / "cache" / "duka_ticks"
SYMS = {"AUDUSD": "AUDUSD", "AUDJPY": "AUDJPY", "EURAUD": "EURAUD", "USDCAD": "USDCAD", "CADJPY": "CADJPY",
        "NZDUSD": "NZDUSD", "GBPUSD": "GBPUSD", "EURGBP": "EURGBP", "GBPJPY": "GBPJPY", "EURUSD": "EURUSD",
        "USDJPY": "USDJPY", "XAUUSD": "XAUUSD", "US100": "USATECHIDXUSD", "US500": "USA500IDXUSD"}
SECS = (1, 2, 5, 10, 30, 60)
STOPS = (1.5, 3.0)
HOLDS = (15, 30, 60)


def plan():
    return list(SYMS)


def hour_ticks(inst, h):
    """One hour of ticks: DataFrame(t, ask, bid) with raw integer prices (scaled later), cached."""
    f = CACHE / inst / f"{h.strftime('%Y%m%d%H')}.npy"
    if f.exists():
        a = np.load(f)
        return a
    url = f"https://datafeed.dukascopy.com/datafeed/{inst}/{h.year}/{h.month - 1:02d}/{h.day:02d}/{h.hour:02d}h_ticks.bi5"
    data = b""
    for k in range(4):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                data = r.content
                break
            if r.status_code == 404:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1 + 2 * k)
    rows = np.zeros((0, 3), dtype=np.float64)
    if data:
        try:
            raw = lzma.decompress(data)
            n = len(raw) // 20
            recs = struct.iter_unpack(">IIIff", raw[:n * 20])
            rows = np.array([(h.value / 1e6 + ms, ask, bid) for ms, ask, bid, _, _ in recs], dtype=np.float64)  # ms since epoch
        except Exception:  # noqa: BLE001
            rows = np.zeros((0, 3), dtype=np.float64)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, rows)
    return rows


def run(sym):
    OUT.mkdir(parents=True, exist_ok=True)
    inst = SYMS[sym]
    rel_all, bt = F.releases(), F.betas(sym)
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = P.bars(sym, "1Min", NF.LOAD_FROM, now)
    d1 = d1[~d1.index.duplicated()].sort_index()
    a5 = NS._atr5(d1)
    sp = U.spec(sym)
    rows, scale = [], None
    for cur, b in bt.items():
        for t, x in sorted((rel_all.get(cur) or {}).items()):
            if t < F.TEST_FROM:
                continue
            ka = a5.index.searchsorted(t) - 1
            if ka < 14:
                continue
            atr = float(a5.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            h0 = t.floor("h")
            tk = np.concatenate([hour_ticks(inst, h0), hour_ticks(inst, h0 + pd.Timedelta(hours=1))])
            if len(tk) < 50:
                continue
            if scale is None:                                  # the raw integers' scale, from the 1-minute bars
                k = d1.index.searchsorted(t) - 1
                ref = float(d1["c"].iloc[k])
                scale = 10.0 ** round(np.log10(tk[0, 2] / ref))
            ts, ask, bid = tk[:, 0], tk[:, 1] / scale, tk[:, 2] / scale
            t0 = t.value / 1e6
            pre = np.searchsorted(ts, t0) - 1
            if pre < 0:
                continue
            spr0 = float(ask[pre] - bid[pre])
            dr = int(x["g"] * (b or 1))
            comm = sp["commission"]
            for S in SECS:
                i = np.searchsorted(ts, t0 + S * 1000)
                if i >= len(ts) or ts[i] - (t0 + S * 1000) > 20_000:
                    continue
                e = ask[i] if dr > 0 else bid[i]
                spr = float(ask[i] - bid[i])
                cpx = (e * float(comm) / (float(sp["contractSize"]) * U.BASE_USD.get(sp["code"].split("/")[0], 1.0))
                       if sp["commissionType"] == "flat_USD" else e * 2 * float(comm) / 100.0)
                for s in STOPS:
                    risk = s * atr
                    stop = e - dr * risk
                    for hm in HOLDS:
                        j = np.searchsorted(ts, ts[i] + hm * 60_000)
                        if j >= len(ts):
                            continue
                        exit_side = bid if dr > 0 else ask
                        seg = exit_side[i:j]
                        hit = np.nonzero((seg <= stop) if dr > 0 else (seg >= stop))[0]
                        px = float(seg[hit[0]]) if len(hit) else float(exit_side[j])
                        r = dr * (px - e) / risk - cpx / risk
                        rows.append({"cur": cur, "title": x["title"], "t": t.isoformat(), "S": S, "s": s, "h": hm,
                                     "r": round(float(r), 4), "spread": round(spr, 6), "spread0": round(spr0, 6),
                                     "move_px": round(float(dr * (e - (ask[pre] + bid[pre]) / 2)), 6), "big": x["big"]})
    (OUT / f"{sym}.json").write_text(json.dumps(rows), encoding="utf-8")
    print(sym, len(rows), "rows, scale", scale)


def _st(s):
    return {"n": int(len(s)), "sum": round(float(s.sum()), 2), "mean": round(float(s.mean()), 3) if len(s) else None}


def report():
    rows = []
    for f in sorted(OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        for x in json.loads(f.read_text(encoding="utf-8")):
            x["sym"] = f.stem
            rows.append(x)
    df = pd.DataFrame(rows)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "rows": int(len(df))}
    if len(df):
        df["train"] = pd.to_datetime(df["t"], utc=True) < F.SPLIT
        # per seconds: each currency's stop / hold chosen on 2025 (only big surprises or all), summed; and the spread
        by_s = {}
        for S, g in df.groupby("S"):
            tot = {"train": 0.0, "unseen": 0.0, "n_unseen": 0, "cur": {}}
            for cur, y in g.groupby("cur"):
                best = None
                for (s, h), z in y.groupby(["s", "h"]):
                    for big in (0, 1):
                        w = z[z["big"] == 1] if big else z
                        tr, un = w[w.train]["r"], w[~w.train]["r"]
                        if len(tr) < 15:
                            continue
                        k = {"s": s, "h": h, "big": big, "train": _st(tr), "unseen": _st(un)}
                        if best is None or k["train"]["sum"] > best["train"]["sum"]:
                            best = k
                if best:
                    tot["cur"][cur] = best
                    tot["train"] = round(tot["train"] + best["train"]["sum"], 2)
                    tot["unseen"] = round(tot["unseen"] + best["unseen"]["sum"], 2)
                    tot["n_unseen"] += best["unseen"]["n"]
            tot["spread_x_normal"] = round(float((g["spread"] / g["spread0"].replace(0, np.nan)).median()), 2)
            by_s[f"{S}s"] = tot
        rep["by_seconds"] = by_s
        # per release type at 2 / 5 / 10 seconds (stop 1.5, 30 minutes): the plain version, no choice
        typ = {}
        for (cur, title), g in df[(df["s"] == 1.5) & (df["h"] == 30)].groupby(["cur", "title"]):
            typ[f"{cur} {title}"] = {f"{S}s": {"train": _st(y[y.train]["r"]), "unseen": _st(y[~y.train]["r"])}
                                     for S, y in g.groupby("S") if S in (2, 5, 10, 30)}
        rep["by_type"] = typ
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cur"} for k, v in rep.get("by_seconds", {}).items()}, ensure_ascii=False, indent=1))


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
