"""EV04 on the releases no study has seen: FTMO's HIGH-impact items that are NOT restricted (bot 7 smart agent,
17.11.2026 evening).

Why: every EV study (agents/news_straddle.py and the others) took its releases from data/ftmo_calendar/history.json,
which holds only FTMO's restricted items (and holidays). The live bot read the whole calendar, so until 17.11 evening
EV04 also traded on the high-impact items that are not restricted - 682 of them in 30.04.2025-14.11.2026 (US Retail
Sales, Unemployment Claims, ISM PMIs, PPI m/m, Core PCE, UK GDP m/m, BOE Gov Bailey speaks, ...). On 17.11 the US
Retail Sales filled 7 EV04 straddles, 6 lost. Since 17.11 evening the live bot trades only the restricted items
(live/news_bot.py EV_RESTRICTED_ONLY), exactly the tested set. This study asks the other question fairly: is there
a release type among the extra ones where the built EV04 version earns? If so it comes back as a card ("add_titles").

How: the same simulation as the study (agents/news_straddle.trades_sym: 1-minute candles, FTMO's spread at the fill
minute + commission, the same slippage), on the extra release minutes of each instrument's own currencies only
(a minute that also has a restricted item is left out - it is in the main study), the built plan's rows only
(variant, own currencies, fill window, the cap per release over the instruments). By release title (the minute's
first high-impact title of the instrument's currencies): 30.04-31.12.2025 vs 2026. The pick is made on 2025 only:
titles with at least MIN_N trades and a positive sum on 2025; shown on 2026.

Output reports/agents/bot7/straddle_extra_study.json; cache cache/news_straddle_extra/<sym>.json; resumes after a
time budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7, right after the straddle study.
  python -m agents.news_straddle_extra [--minutes 10] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import rules_agent as R  # noqa: E402,F401  (first: rules_agent and event_strategies import each other)
from agents import event_strategies as E  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import news_straddle as NS  # noqa: E402
from agents import bot7_trades as BT  # noqa: E402

FULL = ROOT / "reports" / "agents" / "bot7" / "calendar_full.json"
OUT = ROOT / "reports" / "agents" / "bot7" / "straddle_extra_study.json"
CACHE = ROOT / "cache" / "news_straddle_extra"
VERSION = 1
STALE_DAYS = 7
MIN_N = 8


def _cur(ins):
    ins = str(ins or "")
    return "Crude Oil" if ins.startswith("Crude") else (ins.split(" ")[0] if ins else "")


def extra_items():
    """High-impact, not restricted items whose minute has no restricted item."""
    items = json.loads(FULL.read_text(encoding="utf-8")).get("items", [])
    restricted = {pd.Timestamp(i["date"]).tz_convert("UTC") for i in items if i.get("restriction")}
    out = []
    for i in items:
        if i.get("impact") != "high" or i.get("restriction"):
            continue
        t = pd.Timestamp(i["date"]).tz_convert("UTC")
        if t in restricted:
            continue
        out.append((t, _cur(i.get("instrument")), i.get("title", "")))
    return out


def rel_of(items):
    """NF.releases() format: time -> [(currency, category, surprise, title)]."""
    rel = {}
    for t, cur, title in items:
        rel.setdefault(t, []).append((cur, NF.category(title), None, title))
    return rel


def title_of(sym, t, rel):
    curs = U.CURRENCIES.get(sym, [])
    mine = sorted({x[3] for x in rel.get(t, []) if x[0] in curs})
    return " + ".join(mine[:2]) if mine else "?"


def rows_sym(sym, items, rel):
    curs = set(U.CURRENCIES.get(sym, []))
    own = sorted({t for t, cur, _ in items if cur in curs})
    if not own:
        return []
    keep = E._EVENTS.get(sym)
    E._EVENTS[sym] = own
    try:
        mine = {t: v for t, v in rel.items() if any(x[0] in curs for x in v)}   # "usd" = own here, never another's release
        r_ = NS.trades_sym(sym, mine)
    finally:
        if keep is None:
            E._EVENTS.pop(sym, None)
        else:
            E._EVENTS[sym] = keep
    for x in r_:
        x["title"] = title_of(sym, pd.Timestamp(x["rel"]), rel)
    return r_


def built_rows(df):
    b = BT.built("EV04", "1Min")
    if df is None or not len(df) or not b:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    df = NS.window_out(df, b)                       # the fill window of the built plan; no skip groups (other types)
    keys = {}
    for s_ in b:
        if s_ in set(df["sym"]):
            keys.setdefault(NS.built_key(b[s_]), []).append(s_)
    parts = [NS._rows_for(df[df["sym"].isin(ss)], k) for k, ss in keys.items()]
    z = pd.concat(parts) if parts else df.iloc[0:0]
    return z.assign(train=pd.to_datetime(z["t"], utc=True) < NF.SPLIT)


def summary(z):
    by = {}
    for ti, y in z.groupby("title"):
        by[ti] = {"train": NF._w(y[y.train]), "unseen": NF._w(y[~y.train])}
    pick = sorted(t for t, v in by.items() if v["train"]["n"] >= MIN_N and (v["train"]["sum"] or 0) > 0)
    zp = z[z["title"].isin(pick)]
    by_sym = {s_: {"train": NF._w(y[y.train]), "unseen": NF._w(y[~y.train])} for s_, y in z.groupby("sym")}
    return {"all": {"train": NF._w(z[z.train]), "unseen": NF._w(z[~z.train])},
            "pick_rule": f"titles with n>={MIN_N} and sum>0 on 30.04-31.12.2025", "pick": pick,
            "picked": {"train": NF._w(zp[zp.train]), "unseen": NF._w(zp[~zp.train])},
            "by_title": dict(sorted(by.items(), key=lambda kv: -(kv[1]["train"]["n"] + kv[1]["unseen"]["n"]))),
            "by_sym": by_sym}


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
    a = ap.parse_args(argv)
    if not a.force and not stale():
        print("straddle extra study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    items = extra_items()
    rel = rel_of(items)
    CACHE.mkdir(parents=True, exist_ok=True)
    syms = list(NF.SYMBOLS)
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
            print(f"EV04 extra: time budget reached after {done}")
            continue
        try:
            r_ = rows_sym(sym, items, rel)
            rows += r_
            f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
            done.append(sym)
            print(f"EV04 extra {sym}: {len(r_)} rows", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"EV04 extra {sym}: failed ({e})")
    df = pd.DataFrame(rows)
    res = summary(built_rows(df)) if len(df) else {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "symbols": done, "partial": len(done) < len(syms), "extra_items": len(items),
                               "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()), "study": res},
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res:
        print("EV04 extra", json.dumps({"all": res["all"], "pick": res["pick"], "picked": res["picked"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
