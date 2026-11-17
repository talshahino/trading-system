"""The actuals FTMO's calendar history lacks for the biggest US releases (user, 17.11.2026: "and all the dollar's
releases?" - CPI m/m, CPI y/y, Core CPI m/m: the forecast is there, the actual is not). From the BLS API (v1, no
key). FRED / ALFRED (NFP's first print) do not answer GitHub's machines - NFP is left for the bots' computer.
Writes the actuals into reports/agents/bot7/calendar_full.json (field "actual_src").
  python -m agents.us_actuals
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CAL = ROOT / "reports" / "agents" / "bot7" / "calendar_full.json"
SERIES = {"CPI m/m": ("CUSR0000SA0", "mom"), "Core CPI m/m": ("CUSR0000SA0L1E", "mom"),
          "CPI y/y": ("CUUR0000SA0", "yoy"), "Core CPI y/y": ("CUUR0000SA0L1E", "yoy")}
# NFP (PAYEMS): its first print is not in the BLS API (revised only) and FRED / ALFRED do not answer GitHub's machines
# (17.11.2026) - left for the bots' computer. CPI: not seasonally adjusted y/y is never revised; the adjusted m/m only
# by the yearly seasonal factors (rarely the first decimal).
UA = {"User-Agent": "Mozilla/5.0", "Content-type": "application/json"}


def bls(ids, y0, y1):
    r = requests.post("https://api.bls.gov/publicAPI/v1/timeseries/data/", headers=UA, timeout=40,
                      data=json.dumps({"seriesid": ids, "startyear": str(y0), "endyear": str(y1)}))
    j = r.json()
    out = {}
    for s_ in (j.get("Results") or {}).get("series") or []:
        rows = [(pd.Timestamp(f"{x['year']}-{x['period'][1:]}-01"), float(x["value"])) for x in s_["data"]
                if x["period"].startswith("M") and x["period"] != "M13" and x["value"] not in ("-", "")]
        out[s_["seriesID"]] = pd.Series(dict(rows)).sort_index()
    print("BLS:", j.get("status"), j.get("message"), {k: len(v) for k, v in out.items()}, flush=True)
    return out


def main():
    doc = json.loads(CAL.read_text(encoding="utf-8"))
    ser = bls(sorted({v[0] for v in SERIES.values()}), 2024, pd.Timestamp.now().year)
    n = 0
    for i in doc["items"]:
        if not str(i.get("instrument", "")).startswith("USD") or i.get("title") not in SERIES:
            continue
        if i.get("actual") not in (None, "") or i.get("forecast") in (None, ""):
            continue
        rel = pd.Timestamp(i["date"]).tz_convert("UTC")
        if rel > pd.Timestamp.now(tz="UTC"):
            continue
        sid, how = SERIES[i["title"]]
        x = ser.get(sid)
        if x is None:
            continue
        m = (rel.tz_convert(None).to_period("M") - 1).to_timestamp()          # the month the release is about
        prev = m - pd.DateOffset(months=1 if how == "mom" else 12)
        if m not in x.index or prev not in x.index:
            continue
        v = round(100 * (x[m] / x[prev] - 1), 1)
        i["actual"], i["actual_src"] = f"{v:.1f} %", "BLS"
        n += 1
        print(i["date"], i["title"], "forecast", i.get("forecast"), "actual", i["actual"], flush=True)
    doc["us_actuals"] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "filled": n, "source": "BLS API"}
    CAL.write_text(json.dumps(doc, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(doc["us_actuals"]))


if __name__ == "__main__":
    sys.exit(main())
