"""News-release strategies by event type (user, 10.11.2026: "maybe it depends on the kind of release, categorise
the strategies, maybe enter 2 minutes after the release"). Research only.

Events: FTMO's own calendar (data/ftmo_calendar/history.json, April 2025 ->), high impact or restricted, grouped:
  rates   central-bank decisions, statements, minutes, policy reports
  infl    CPI / PPI / RPI variants
  jobs    NFP, employment change, unemployment rate, claims
  gdp     GDP
  oil     crude oil inventories (WTI only)
Instruments per release: the ones its currency moves (USD: EURUSD GBPUSD USDJPY XAUUSD US100 US500 US30; GBP: GBPUSD
GBPJPY UK100; EUR: EURUSD GER40; CAD: USDCAD; AUD: AUDUSD; NZD: NZDUSD; CHF: USDCHF; crude: WTI).

Strategies (1-minute candles, entry at the open of minute D after the release, D = 2, 3, 5):
  MOM  follow the first D minutes (release-minute open -> close of minute D-1)
  FADE against them
  SURP the surprise: actual vs forecast (unemployment and oil inventories: higher = weaker), into the instrument's
       direction (indices: skipped - a strong US number can lift or sink them)
  stop: beyond the first D minutes' extreme on the other side, at least 0.5 x the 1-minute ATR(60) x 6
  exit: after H minutes (H = 15, 30, 60) or at the stop; a second version also takes 2R as a target
Costs: FTMO's spread at the entry minute (Dukascopy 1-minute ask-bid - news spreads are wide, and that is in it),
FTMO commission (agents/ftmo_costs.py). No swap (all trades close within the hour).
Judged per category x strategy x D x exit: trades, mean R, t, and the two halves (2025 / 2026) apart.
FTMO note: restricted events allow no trade from 2 minutes before to 2 after (FTMO Standard funded accounts); entries
at D = 2 sit on that edge, so D = 3 is the first fully safe one. FTMO Swing accounts have no news restriction.
Writes reports/news_research.json and .csv.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from agents import ftmo_costs as C
from agents import ftmo_universe as U
from agents import poc_agent as P

HIST = Path("data/ftmo_calendar/history.json")
BY_CUR = {"USD": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US100", "US500", "US30"], "GBP": ["GBPUSD", "GBPJPY", "UK100"],
          "EUR": ["EURUSD", "GER40"], "CAD": ["USDCAD"], "AUD": ["AUDUSD"], "NZD": ["NZDUSD"], "CHF": ["USDCHF"],
          "Crude Oil": ["WTI"]}
# +1 when the instrument rises as the release's currency strengthens
SIGN = {("USD", "EURUSD"): -1, ("USD", "GBPUSD"): -1, ("USD", "USDJPY"): 1, ("USD", "XAUUSD"): -1, ("GBP", "GBPUSD"): 1,
        ("GBP", "GBPJPY"): 1, ("EUR", "EURUSD"): 1, ("CAD", "USDCAD"): -1, ("AUD", "AUDUSD"): 1, ("NZD", "NZDUSD"): 1,
        ("CHF", "USDCHF"): -1, ("Crude Oil", "WTI"): 1}
DELAYS, HOLDS = (2, 3, 5), (15, 30, 60)


def category(title):
    t = title.lower()
    if "oil" in t:
        return "oil"
    if any(k in t for k in ("rate", "fomc", "monetary", "policy", "mpc", "statement", "minutes", "inflation letter")):
        return "rates"
    if any(k in t for k in ("cpi", "ppi", "rpi")):
        return "infl"
    if any(k in t for k in ("employment", "unemployment", "payroll", "claims", "labor")):
        return "jobs"
    if "gdp" in t:
        return "gdp"
    return "other"


def number(x):
    if x in (None, ""):
        return None
    m = re.search(r"-?\d+(\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None


def currency(ins):
    if ins.startswith("Crude"):
        return "Crude Oil"
    return ins.split(" ")[0] if ins else ""


def events():
    out = []
    for i in json.loads(HIST.read_text(encoding="utf-8")):
        if i.get("impact") != "high" and not i.get("restriction"):
            continue
        cur = currency(str(i.get("instrument", "")))
        if cur not in BY_CUR:
            continue
        a, f = number(i.get("actual")), number(i.get("forecast"))
        surp = None
        if a is not None and f is not None and a != f:
            surp = 1 if a > f else -1
            if "unemployment" in i["title"].lower() or "claims" in i["title"].lower() or category(i["title"]) == "oil":
                surp = -surp                       # more unemployed / more oil in storage = weaker
        out.append({"t": pd.Timestamp(i["date"]).tz_convert("UTC"), "title": i["title"], "cat": category(i["title"]),
                    "cur": cur, "surp": surp, "restricted": bool(i.get("restriction"))})
    return out


def main():
    evs = events()
    t0, t1 = min(e["t"] for e in evs) - pd.Timedelta(days=3), max(e["t"] for e in evs) + pd.Timedelta(days=1)
    syms = sorted({s for e in evs for s in BY_CUR[e["cur"]]})
    rows = []
    for sym in syms:
        try:
            d = P.bars(sym, "1Min", t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
            sp = P.spread_1m(sym, t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d"))
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: no data ({e})")
            continue
        if d is None or d.empty:
            continue
        d = d[~d.index.duplicated()].sort_index()
        o, h, l, c = (d[k].to_numpy(float) for k in "ohlc")
        tr = np.maximum(h - l, np.abs(np.r_[c[0], c[:-1]] - h), np.abs(np.r_[c[0], c[:-1]] - l))
        atr = pd.Series(tr).rolling(60).mean().to_numpy()
        idx = d.index
        spec = U.spec(sym)
        print(f"{sym}: {len(d):,} minutes")
        for e in evs:
            if sym not in BY_CUR[e["cur"]]:
                continue
            k0 = idx.searchsorted(e["t"])
            if k0 >= len(idx) or idx[k0] != e["t"] or k0 < 61 or k0 + 70 >= len(idx):
                continue
            a = atr[k0 - 1]
            if not np.isfinite(a) or a <= 0:
                continue
            for D in DELAYS:
                ke = k0 + D                                 # entry at the open of minute D
                if idx[ke] - idx[k0] != pd.Timedelta(minutes=D):
                    continue
                first = c[ke - 1] - o[k0]
                hi, lo = h[k0:ke].max(), l[k0:ke].min()
                e_px = o[ke]
                dirs = {"MOM": np.sign(first), "FADE": -np.sign(first)}
                if e["surp"] is not None and (e["cur"], sym) in SIGN:
                    dirs["SURP"] = e["surp"] * SIGN[(e["cur"], sym)]
                for st, dr in dirs.items():
                    if dr == 0:
                        continue
                    stop = lo if dr > 0 else hi
                    if dr * (e_px - stop) < 3 * a:          # at least 3 x the 1-minute ATR away
                        stop = e_px - dr * 3 * a
                    risk = abs(e_px - stop)
                    # costs at entry (FTMO's spread at that minute + commission), in R
                    tt = pd.DataFrame({"r": [0.0], "entry_px": [e_px], "stop_px": [stop], "dir": [dr],
                                       "time": [idx[ke]], "exit_time": [idx[ke]]})
                    cost = C.apply(tt, sym, sp)
                    cost_r = float((cost["spread_px"].iloc[0] + cost["comm_px"].iloc[0]) / risk)
                    for H in HOLDS:
                        for tgt in (None, 2.0):
                            r = None
                            for q in range(ke, min(ke + H, len(idx))):
                                if (l[q] <= stop) if dr > 0 else (h[q] >= stop):
                                    r = -1.0 if q > ke or True else -1.0
                                    break
                                if tgt and ((h[q] >= e_px + dr * tgt * risk) if dr > 0 else (l[q] <= e_px + dr * tgt * risk)):
                                    r = tgt
                                    break
                            if r is None:
                                q = min(ke + H, len(idx)) - 1
                                r = dr * (c[q] - e_px) / risk
                            rows.append({"sym": sym, "cat": e["cat"], "title": e["title"], "t": e["t"], "restricted": e["restricted"],
                                         "strat": st, "D": D, "H": H, "tgt": tgt or 0, "r": r - cost_r, "cost_r": cost_r})
    df = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/news_research_trades.csv.gz", index=False)
    g = []
    for key, x in df.groupby(["cat", "strat", "D", "H", "tgt"]):
        r = x["r"].to_numpy()
        a = x[x["t"] < pd.Timestamp("2026-01-01", tz="UTC")]["r"]
        b = x[x["t"] >= pd.Timestamp("2026-01-01", tz="UTC")]["r"]
        g.append(dict(zip(["cat", "strat", "D", "H", "tgt"], key), n=len(r), mean=round(float(r.mean()), 3),
                      t=round(float(r.mean() / r.std() * np.sqrt(len(r))), 2) if len(r) > 2 and r.std() > 0 else None,
                      h2025=round(float(a.mean()), 3) if len(a) else None, h2026=round(float(b.mean()), 3) if len(b) else None,
                      cost=round(float(x["cost_r"].mean()), 3)))
    s = pd.DataFrame(g).sort_values("t", ascending=False)
    pd.set_option("display.width", 200)
    print(s.head(40).to_string())
    print(s.groupby(["cat", "strat"])["mean"].max().round(3).to_string())
    s.to_csv("reports/news_research.csv", index=False)
    Path("reports/news_research.json").write_text(s.to_json(orient="records"))


if __name__ == "__main__":
    main()
