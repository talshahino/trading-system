"""The actuals FTMO's calendar history lacks for the biggest US releases (user, 17.11.2026: "and all the dollar's
releases?" - NFP, CPI, the Fed's rate, advance GDP: the forecast is there, the actual is not), from the official
numbers as they were first published: ALFRED (the St. Louis Fed's archive of every vintage, no key) - the series as
it stood on the release day, so a later revision never changes the surprise. When ALFRED does not answer, FRED's
current series (marked "revised" - CPI y/y not seasonally adjusted is never revised; NFP is).

  title                       series        how
  Non-Farm Employment Change  PAYEMS        change of the last month, thousands ("K")
  CPI m/m / Core CPI m/m      CPIAUCSL / CPILFESL   % change of the last month, 1 decimal
  CPI y/y / Core CPI y/y      CPIAUCNS / CPILFENS   % change over 12 months, 1 decimal
  Federal Funds Rate          DFEDTARU      the target's upper bound the day after
  Advance GDP q/q             A191RL1Q225SBEA       the last quarter, annualised %, 1 decimal

Writes the actuals into reports/agents/bot7/calendar_full.json (field "actual_src" says where it came from).
  python -m agents.us_actuals
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "reports" / "agents" / "bot7" / "calendar_full.json"
SERIES = {"Non-Farm Employment Change": ("PAYEMS", "chg"), "CPI m/m": ("CPIAUCSL", "mom"), "Core CPI m/m": ("CPILFESL", "mom"),
          "CPI y/y": ("CPIAUCNS", "yoy"), "Core CPI y/y": ("CPILFENS", "yoy"), "Federal Funds Rate": ("DFEDTARU", "level"),
          "Advance GDP q/q": ("A191RL1Q225SBEA", "level")}
UA = {"User-Agent": "Mozilla/5.0"}


def _csv(url):
    for k in range(3):
        try:
            r = requests.get(url, timeout=40, headers=UA)
            if r.status_code == 200 and r.text[:200].upper().find("DATE") >= 0:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = ["date", "v"] + list(df.columns[2:])
                df["date"] = pd.to_datetime(df["date"])
                df["v"] = pd.to_numeric(df["v"], errors="coerce")
                return df.dropna(subset=["v"])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 + 3 * k)
    return None


def as_of(sid, day):
    """The series as it stood on `day` (ALFRED vintage), else today's FRED series cut at `day` (revised)."""
    d = pd.Timestamp(day).strftime("%Y-%m-%d")
    df = _csv(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={sid}&vintage_date={d}")
    if df is not None and len(df):
        return df, "ALFRED"
    df = _csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
    if df is not None and len(df):
        return df, "FRED revised"
    return None, None


def value(df, how, rel):
    if how == "level":
        return float(df["v"].iloc[-1]) if len(df) else None
    v = df["v"].to_numpy()
    if len(v) < 13:
        return None
    if how == "chg":
        return round(float(v[-1] - v[-2]), 0)
    if how == "mom":
        return round(100 * float(v[-1] / v[-2] - 1), 1)
    if how == "yoy":
        return round(100 * float(v[-1] / v[-13] - 1), 1)
    return None


def main():
    doc = json.loads(CAL.read_text(encoding="utf-8"))
    n, src_n = 0, {}
    cache = {}
    for i in doc["items"]:
        if not str(i.get("instrument", "")).startswith("USD") or i.get("title") not in SERIES:
            continue
        if i.get("actual") not in (None, "") or i.get("forecast") in (None, ""):
            continue
        rel = pd.Timestamp(i["date"]).tz_convert("UTC")
        if rel > pd.Timestamp.now(tz="UTC"):
            continue
        sid, how = SERIES[i["title"]]
        day = (rel + pd.Timedelta(days=1)) if sid == "DFEDTARU" else rel      # the rate: in force the day after
        key = (sid, day.strftime("%Y-%m-%d"))
        if key not in cache:
            cache[key] = as_of(sid, day)
        df, src = cache[key]
        if df is None:
            continue
        if src == "FRED revised":                     # cut at what was published by then (the last month before)
            last = (rel.tz_localize(None) if rel.tzinfo is None else rel.tz_convert(None)).to_period("M").to_timestamp() - pd.offsets.MonthBegin(1)
            if sid == "DFEDTARU":
                df = df[df["date"] <= day.tz_convert(None)]
            elif "GDP" in i["title"]:
                df = df[df["date"] <= last - pd.offsets.MonthBegin(2)]
            else:
                df = df[df["date"] <= last]
        v = value(df, how, rel)
        if v is None:
            continue
        unit = "K" if how == "chg" else "%"
        i["actual"] = f"{v:g}{unit}" if unit == "K" else f"{v:.2f} %" if sid == "DFEDTARU" else f"{v:.1f} %"
        i["actual_src"] = src
        n += 1
        src_n[src] = src_n.get(src, 0) + 1
        print(i["date"], i["title"], "forecast", i.get("forecast"), "actual", i["actual"], src, flush=True)
    doc["us_actuals"] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "filled": n, "by_source": src_n}
    CAL.write_text(json.dumps(doc, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(doc["us_actuals"]))


if __name__ == "__main__":
    sys.exit(main())
