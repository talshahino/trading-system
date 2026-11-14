"""EV08 - trading the release's surprise (actual vs forecast) by its direction (the user's task, 14.11.2026, item 41:
Dewonggo - "Initial Jobless Claims worse than expected = buy (better = sell), Existing Home Sales the same way, crude
oil inventories the other way; out at a fixed time" - on BTC / the US indices / gold).

Why a new calendar first: FTMO's calendar history the other bot 7 studies use (data/ftmo_calendar/history.json) has
the actual and the forecast for NO US release at all (0 of 20 NFP, 0 of 52 CPI, no PPI; only crude oil inventories
and other currencies' releases have numbers), and only high-impact / restricted items, so jobless claims and home
sales are not in it. Every "surprise" knob of the earlier studies (news_entry.py "agree", news_fix.py) measured the
US releases as "no usable surprise" (bot 7 smart agent, 14.11.2026). Here, on the bots' computer:
  1. calendar_full(): FTMO's calendar API (agents/ftmo_calendar.fetch, 10 days a request) from 01.03.2025 to today,
     every impact, merged with the live bot's calendar (live/calendar.json: it has the US numbers since 02.11.2026),
     kept in reports/agents/bot7/calendar_full.json with the coverage of actual+forecast by currency and year - so
     what the API really holds is known, and the history grows every week from the live feed either way.
  2. the study, on US releases with an actual and a forecast (actual == forecast: no trade):
     group     claims (Unemployment Claims) / home (Existing, New, Pending Home Sales) / oil (Crude Oil Inventories)
               / jobs / infl / activity / other - the release that leads its minute (several at 08:30: the most
               important one; "alone" = the only US number in that minute)
     "bad"     the number is worse for the US economy than expected (claims / unemployment higher, the rest lower;
               crude inventories: a bigger build = bad for oil)
     side      p1: bad US number -> BTC / indices / gold / EURUSD up, USDJPY down, WTI down (Dewonggo: rate-cut
               hopes); p-1 the other way (growth fear / strong dollar). Dewonggo's own rule is p1.
     entry     at the open of minute 3 / 5 / 10 after the release (FTMO: never inside the 2 minutes of a restricted
               release, so 3 is the earliest)
     stop      1.5 / 3 x the 5-minute ATR(14) before the release        exit  out after 15 / 30 / 60 / 120 minutes
     size      all / big (|actual - forecast| at least the title's median on 2025 - the median from 2025 only)
     alone     all / only when no other US number came out in the same minute
  "dew" = the rule as told (claims + home, side p1, BTC / US100 / US500 / US30 / XAUUSD), only entry / stop / exit /
  size / alone chosen; every group on its own with its side chosen too.
Costs: FTMO's spread at the entry minute (Dukascopy) and commission (agents/ftmo_costs.py), as the other bot 7 studies.
Fair method: chosen on 30.04.2025-31.12.2025 by total R, shown on 2026, time-weighted (agents/agentkit.py);
robustness = of the 10 best on 2025, how many made money in 2026. A card only from a version chosen on 2025 that
also earns in 2026.

Output: reports/agents/bot7/surprise_study.json. Cached per instrument (cache/news_surprise/<sym>.json), resumes
after a time budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7.
  python -m agents.news_surprise [--minutes 10] [--force] [--no-fetch]
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
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import poc_agent as P  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "surprise_study.json"
CAL_FULL = ROOT / "reports" / "agents" / "bot7" / "calendar_full.json"
LIVE_CAL = ROOT / "live" / "calendar.json"
HIST = ROOT / "data" / "ftmo_calendar" / "history.json"
CACHE = ROOT / "cache" / "news_surprise"
VERSION = 1
CAL_DAYS = 7                         # the API calendar is fetched again after a week
STALE_DAYS = 3
SYMBOLS = ["BTCUSD", "US100", "US500", "US30", "XAUUSD", "EURUSD", "USDJPY", "WTI"]
DEW_SYMS = {"BTCUSD", "US100", "US500", "US30", "XAUUSD"}
# +1: this instrument rises when a US number is bad under side p1 (rate-cut hopes: risk up, dollar down)
BETA = {"BTCUSD": 1, "US100": 1, "US500": 1, "US30": 1, "XAUUSD": 1, "EURUSD": 1, "USDJPY": -1, "WTI": -1}
DELAYS = (3, 5, 10)
STOPS = (1.5, 3.0)
HOLDS = (15, 30, 60, 120)
POLS = (1, -1)
SIZES = ("all", "big")
ALONES = ("all", "alone")
GROUP_PRIO = ("jobs", "infl", "rates", "activity", "claims", "home", "oil", "other")
BASE = "d3|s3|h60|p1|zall|aall"      # Dewonggo's rule as told: right after the release, out at a fixed time
TOP = 10
MIN_N = 8
SPAN = 125                           # minutes after the release that must be there


def group(title):
    t = title.lower()
    if "claims" in t:
        return "claims"
    if "home sales" in t or "housing starts" in t or "building permits" in t:
        return "home"
    if "crude oil" in t or "inventories" in t and "oil" in t:
        return "oil"
    if "non-farm" in t or "nonfarm" in t or "unemployment rate" in t or "adp" in t or "average hourly" in t or "jolts" in t:
        return "jobs"
    if "cpi" in t or "ppi" in t or "pce" in t or "inflation" in t:
        return "infl"
    if "federal funds" in t or "rate" in t and "fed" in t:
        return "rates"
    if any(k in t for k in ("retail sales", "ism", "pmi", "gdp", "durable", "industrial production", "consumer confidence",
                            "sentiment", "philly", "empire state")):
        return "activity"
    return "other"


def bad_sign(title, a, f):
    """+1 when the number is worse for the US economy (crude inventories: for oil) than the forecast, -1 better."""
    s = 1 if a > f else -1
    t = title.lower()
    worse_if_higher = "claims" in t or "unemployment" in t or "inventories" in t
    return s if worse_if_higher else -s


def _items_key(i):
    return (str(i.get("date"))[:16], i.get("title"), i.get("instrument"))


def _merge(base, more):
    """One item per (minute, title, instrument); a version with the actual wins."""
    out = {_items_key(i): i for i in base}
    for i in more:
        k = _items_key(i)
        if k not in out or (out[k].get("actual") in (None, "") and i.get("actual") not in (None, "")):
            out[k] = i
    return sorted(out.values(), key=lambda i: str(i.get("date")))


def coverage(items):
    cov = {}
    for i in items:
        cur = str(i.get("instrument", "")).split(" ")[0] or "?"
        y = str(i.get("date"))[:4]
        c = cov.setdefault(f"{cur} {y}", {"items": 0, "with_numbers": 0})
        c["items"] += 1
        c["with_numbers"] += int(NF._num(i.get("actual")) is not None and NF._num(i.get("forecast")) is not None)
    return dict(sorted(cov.items()))


def calendar_full(fetch=True):
    """FTMO's calendar of every impact since 01.03.2025 (the API, weekly) + the live feed + the old history."""
    old = {}
    try:
        old = json.loads(CAL_FULL.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    items = old.get("items") or []
    fresh = old.get("api_at") and pd.Timestamp.now(tz="UTC") - pd.Timestamp(old["api_at"]) < pd.Timedelta(days=CAL_DAYS)
    api_at, api_err = old.get("api_at"), None
    if fetch and not fresh:
        try:
            from agents import ftmo_calendar as CAL
            got = CAL.fetch("2025-03-01", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
            if got:
                items = _merge(items, got)
                api_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        except Exception as e:  # noqa: BLE001 - without the API the live feed and the history still count
            api_err = str(e)[:200]
    for f in (HIST, LIVE_CAL):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
            items = _merge(items, j.get("items", []) if isinstance(j, dict) else j)
        except Exception:  # noqa: BLE001
            pass
    keep = [{k: i.get(k) for k in ("title", "impact", "instrument", "restriction", "date", "forecast", "previous", "actual")}
            for i in items]
    out = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "api_at": api_at, "api_error": api_err,
           "coverage": coverage(keep), "items": keep}
    CAL_FULL.parent.mkdir(parents=True, exist_ok=True)
    CAL_FULL.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return out


def us_releases(items):
    """Release minute -> the leading US number {group, title, bad, z_abs, alone}. Only items with actual and forecast."""
    by_t = {}
    for i in items:
        if str(i.get("instrument", "")).split(" ")[0] not in ("USD", "Crude"):
            continue
        a, f = NF._num(i.get("actual")), NF._num(i.get("forecast"))
        if a is None or f is None or a == f:
            continue
        t = pd.Timestamp(i["date"]).tz_convert("UTC")
        by_t.setdefault(t, []).append({"title": i["title"], "group": group(i["title"]), "bad": bad_sign(i["title"], a, f),
                                       "abs": abs(a - f), "impact": i.get("impact")})
    # the median surprise of each title on 2025 only (no look at 2026)
    med = {}
    for t, xs in by_t.items():
        if t < NF.SPLIT:
            for x in xs:
                med.setdefault(x["title"], []).append(x["abs"])
    med = {k: float(np.median(v)) for k, v in med.items() if v}
    imp = {"high": 0, "medium": 1, "low": 2}
    out = {}
    for t, xs in by_t.items():
        lead = min(xs, key=lambda x: (imp.get(x["impact"], 3), GROUP_PRIO.index(x["group"])))
        m = med.get(lead["title"])
        out[t] = {"group": lead["group"], "title": lead["title"], "bad": lead["bad"],
                  "big": int(m is not None and lead["abs"] >= m), "alone": int(len(xs) == 1)}
    return out


def vkey(d, s, h, p):
    return f"d{d}|s{s:g}|h{h}|p{p}"


def _atr5(d1):
    d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    a = R.atr(d5["h"].to_numpy(), d5["l"].to_numpy(), d5["c"].to_numpy(), 14)
    return pd.Series(a, index=d5.index)


def simulate(idx, o, h, l, c, a5, rel, sym):
    """Raw trades (before costs) of every variant at every release, from 1-minute bid bars."""
    raw = []
    for t, x in sorted(rel.items()):
        if t < NF.TEST_FROM:
            continue
        k0 = idx.searchsorted(t)
        if k0 >= len(idx) or k0 < 30 or k0 + SPAN >= len(idx) or idx[k0 + SPAN] - idx[k0] > pd.Timedelta(minutes=SPAN + 5):
            continue
        ka = a5.index.searchsorted(t) - 1                      # the last 5-minute bar closed before the release
        if ka < 14:
            continue
        atr = float(a5.iloc[ka])
        if not np.isfinite(atr) or atr <= 0:
            continue
        for d in DELAYS:
            ke = idx.searchsorted(t + pd.Timedelta(minutes=d))
            if ke >= len(idx) or idx[ke] - t > pd.Timedelta(minutes=d + 2):
                continue
            e = o[ke]
            for p in POLS:
                dr = int(x["bad"] * BETA[sym] * p)
                for s in STOPS:
                    stop = e - dr * s * atr
                    for hm in HOLDS:
                        end = idx.searchsorted(idx[ke] + pd.Timedelta(minutes=hm))
                        end = min(max(end, ke + 1), len(idx))
                        px, qx = c[end - 1], end - 1
                        for q in range(ke, end):
                            if (l[q] <= stop) if dr > 0 else (h[q] >= stop):
                                px, qx = stop, q
                                break
                        raw.append({"v": vkey(d, s, hm, p), "time": idx[ke], "exit_time": idx[qx], "dir": dr,
                                    "entry_px": e, "stop_px": stop, "r": dr * (px - e) / (s * atr), "rel": t,
                                    "group": x["group"], "big": x["big"], "alone": x["alone"]})
    return raw


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
    o, h, l, c = (d1[x].to_numpy(float) for x in "ohlc")
    raw = simulate(d1.index, o, h, l, c, _atr5(d1), rel, sym)
    if not raw:
        return []
    tt = pd.DataFrame(raw)
    net = C.apply(tt, sym, sp)
    return [{"v": x.v, "sym": sym, "t": pd.Timestamp(x.time).isoformat(), "dir": int(x.dir), "r": round(float(x.r), 4),
             "gross": round(float(tt.at[x.Index, "r"]), 4), "group": x.group, "big": int(x.big), "alone": int(x.alone),
             "rel": pd.Timestamp(x.rel).isoformat()} for x in net.itertuples()]


def _cands(df, fixed_pol=None):
    cand = []
    for v, y in df.groupby("v"):
        if fixed_pol is not None and not v.endswith(f"|p{fixed_pol}"):
            continue
        for sz, al in itertools.product(SIZES, ALONES):
            z = y if sz == "all" else y[y["big"] == 1]
            z = z if al == "all" else z[z["alone"] == 1]
            tr, un = z[z.train], z[~z.train]
            cand.append({"v": f"{v}|z{sz}|a{al}", "train": NF._w(tr), "unseen": NF._w(un),
                         "gross_unseen": round(float(un["gross"].sum()), 2) if len(un) else 0.0})
    cand.sort(key=lambda c_: (c_["train"]["n"] < MIN_N, -c_["train"]["sum"]))
    return cand


def _summary(cand, base_key=BASE):
    top = cand[:TOP]
    base = next((c_ for c_ in cand if c_["v"] == base_key), None)
    by = {}
    for key, pat in (("delay", r"^d(\d+)\|"), ("stop", r"\|s([\d.]+)\|"), ("hold", r"\|h(\d+)\|"), ("side", r"\|p(-?1)\|"),
                     ("size", r"\|z(all|big)\|"), ("alone", r"\|a(all|alone)$")):
        g = {}
        for c_ in cand:
            m = re.search(pat, c_["v"])
            if m:
                g.setdefault(m.group(1), []).append(c_)
        by[key] = {k_: {"variants": len(cs), "train_median_sum": round(float(np.median([c_["train"]["sum"] for c_ in cs])), 2),
                        "unseen_median_sum": round(float(np.median([c_["unseen"]["sum"] for c_ in cs])), 2),
                        "unseen_positive": int(sum(1 for c_ in cs if (c_["unseen"]["sum"] or 0) > 0))}
                   for k_, cs in g.items()}
    return {"baseline": base, "best": cand[0] if cand else None, "top": top,
            "robust": {"top_positive_unseen": int(sum(1 for c_ in top if (c_["unseen"]["sum"] or 0) > 0)), "top": len(top)},
            "by": by, "tested": len(cand)}


def study(df):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
    res = {}
    dew = df[df["group"].isin(["claims", "home"]) & df["sym"].isin(DEW_SYMS)]
    if len(dew):
        res["dew"] = _summary(_cands(dew, fixed_pol=1))
        best = res["dew"]["best"]
        if best:                          # the chosen version on each instrument (no new choice there)
            v0 = best["v"].split("|z")[0]
            res["dew"]["by_sym"] = {s_: {"train": NF._w(y[y.train]), "unseen": NF._w(y[~y.train])}
                                    for s_, y in dew[dew["v"] == v0].groupby("sym")}
    res["groups"] = {}
    for g_, y in df.groupby("group"):
        cand = _cands(y)
        if not cand or cand[0]["train"]["n"] < MIN_N:
            continue
        s_ = _summary(cand)
        res["groups"][g_] = {"best": s_["best"], "baseline": s_["baseline"], "robust": s_["robust"], "by": s_["by"],
                             "releases_train": int(y[y.train]["rel"].nunique()), "releases_unseen": int(y[~y.train]["rel"].nunique())}
    return res


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
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args(argv)
    if not a.force and not stale():
        print("surprise study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    cal = calendar_full(fetch=not a.no_fetch)
    rel = us_releases(cal["items"])
    n_tr = sum(1 for t in rel if NF.TEST_FROM <= t < NF.SPLIT)
    n_un = sum(1 for t in rel if t >= NF.SPLIT)
    print(f"EV08: {len(rel)} US releases with numbers (2025: {n_tr}, 2026: {n_un}); API {cal.get('api_at')} {cal.get('api_error') or ''}")
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, done = [], []
    sig = f"{VERSION}|{len(rel)}"                     # a calendar with more numbers makes the cache old
    if n_tr >= MIN_N:
        for sym in SYMBOLS:
            f = CACHE / f"{sym}.json"
            try:
                c_ = json.loads(f.read_text(encoding="utf-8"))
                if c_.get("sig") == sig and pd.Timestamp.now(tz="UTC") - pd.Timestamp(c_["at"]) < pd.Timedelta(days=STALE_DAYS):
                    rows += c_["rows"]
                    done.append(sym)
                    continue
            except Exception:  # noqa: BLE001
                pass
            if time.time() > t_end and done:
                print(f"EV08: time budget reached after {done}")
                continue
            try:
                r_ = trades_sym(sym, rel)
                rows += r_
                f.write_text(json.dumps({"sig": sig, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
                done.append(sym)
                print(f"EV08 {sym}: {len(r_)} rows ({len(rows)} so far)", flush=True)
            except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
                print(f"EV08 {sym}: failed ({e})")
    df = pd.DataFrame(rows)
    res = study(df) if len(df) else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "symbols": done, "partial": n_tr >= MIN_N and len(done) < len(SYMBOLS), "rows": int(len(df)),
                               "releases": {"train": n_tr, "unseen": n_un,
                                            "by_group": pd.Series([x["group"] for x in rel.values()]).value_counts().to_dict()
                                            if rel else {}},
                               "calendar": {"api_at": cal.get("api_at"), "api_error": cal.get("api_error"),
                                            "coverage_usd": {k: v for k, v in cal["coverage"].items() if k.startswith(("USD", "Crude"))}},
                               "too_few": n_tr < MIN_N, "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()),
                               "base": BASE, "study": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res.get("dew"):
        print("EV08 dew", json.dumps({"base": res["dew"]["baseline"], "best": res["dew"]["best"],
                                      "robust": res["dew"]["robust"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
