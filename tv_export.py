"""The scalping watch bot's trades as a TradingView indicator (user, 09.11.2026: "I want to see every position on my
TradingView - entry long/short, where the stop is, where the target is - and review whether the timing is right").

TradingView cannot read data from outside, so the trades are written INTO the indicator's code:
  * the backtest trades of the bot's two tracks (reports/*own15*: 'base' = the original track, 'b18' = the wide one)
  * the watch bot's forward trades, read from live/watch_own_state.json on branch 'live' (also waiting setups)
Each trade is drawn like TradingView's long/short position tool: green box entry -> target, red box entry -> stop,
from the entry to the exit, with the impulse (dotted line) and the 15-minute gap (grey box) it came from.

  python tv_export.py [--days 365] [--state path/to/watch_own_state.json]
Writes tradingview/scalping_trades.pine (paste it into TradingView's Pine editor).
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tradingview" / "scalping_trades.pine"
CORE = ["US100", "US30", "US500", "GER40", "FRA40", "JP225", "UK100", "XAUUSD", "EURUSD", "EURJPY", "USDJPY", "GBPUSD",
        "GBPJPY", "USDCHF", "AUDUSD", "NZDUSD"]
EXTRA = ["WTI", "XAGUSD", "USDCAD", "EURGBP", "EU50", "AUS200"]
# names the same instrument has on TradingView's feeds (the chart's ticker without exchange prefix and '.cash')
ALIASES = {"US100": ["NAS100", "NAS100USD", "USTEC", "NDX", "NQ1!", "US100"], "US30": ["US30", "US30USD", "DJI", "YM1!", "WS30"],
           "US500": ["SPX500", "SPX500USD", "US500", "SPX", "ES1!", "SP500"], "GER40": ["GER40", "DE40", "DE30EUR", "DAX", "GER30", "DEU40", "FDAX1!"],
           "FRA40": ["FRA40", "FR40", "FRA40EUR", "CAC40", "CAC", "FCE1!"], "JP225": ["JP225", "JPN225", "JP225USD", "NI225", "NKD1!"],
           "UK100": ["UK100", "UK100GBP", "FTSE100", "UKX", "FTSE"], "WTI": ["USOIL", "WTI", "WTICOUSD", "XTIUSD", "CL1!", "USCRUDE"],
           "EU50": ["EU50", "EUSTX50", "EU50EUR", "SX5E", "STOXX50", "STOXX50E", "FESX1!"], "AUS200": ["AUS200", "AU200AUD", "ASX200", "XJO", "AUS200AUD"]}


def dec(sym):
    if sym in ("XAUUSD", "XAGUSD", "WTI"):
        return 3
    if len(sym) == 6 and sym.isalpha():
        return 3 if sym.endswith("JPY") else 5
    return 2


def backtest(days):
    latest = {}
    for f in sorted(glob.glob(str(ROOT / "reports" / "*own15*" / "*.csv.gz"))):
        latest[os.path.basename(f)] = f                     # the newest report of each instrument group
    d = pd.concat([pd.read_csv(f) for f in latest.values()])
    d = d[d.variant.isin(["base", "b18"]) & (d.status == "closed")].copy()
    d = d.drop_duplicates(["symbol", "variant", "leg_top", "dir", "time"])   # an instrument tested in two report groups
    for c in ("time", "exit_time", "leg_start", "leg_top"):
        d[c] = pd.to_datetime(d[c], utc=True)
    d = d[d.time >= d.time.max() - pd.Timedelta(days=days)]
    key = lambda x: (x.symbol, x.leg_top, x.dir)  # noqa: E731
    strict = {key(x) for x in d[d.variant == "base"].itertuples()}
    rows = []
    for x in d[d.variant == "b18"].itertuples():            # b18 holds every setup of both tracks
        rows.append({"symbol": x.symbol, "dir": int(x.dir), "leg_start": x.leg_start, "leg_top": x.leg_top, "time": x.time,
                     "exit_time": x.exit_time, "entry": x.entry, "stop": x.stop, "target": x.target, "exit": x.exit,
                     "r": x.r, "gap_lo": x.gap_lo, "gap_hi": x.gap_hi, "wide": key(x) not in strict, "src": "B",
                     "status": "closed", "reason": x.reason})
    return rows


def forward(state_path):
    try:
        if state_path:
            s = json.loads(Path(state_path).read_text(encoding="utf-8"))
        else:
            r = requests.get("https://api.github.com/repos/talshahino/trading-system/contents/live/watch_own_state.json",
                             params={"ref": "live"}, headers={"Accept": "application/vnd.github+json"}, timeout=30)
            j = r.json()
            raw = base64.b64decode(j["content"]) if j.get("content") else requests.get(j["download_url"], timeout=60).content
            s = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        print(f"forward trades not read ({e}); backtest only")
        return []
    rows = []
    for t in s.get("watch") or []:
        if t.get("status") not in ("pending", "open", "closed"):
            continue
        rows.append({"symbol": t["symbol"], "dir": int(t["dir"]), "leg_start": pd.Timestamp(t["leg_start"]),
                     "leg_top": pd.Timestamp(t["leg_top"]), "time": pd.Timestamp(t["time"]) if t.get("time") else None,
                     "exit_time": pd.Timestamp(t["exit_time"]) if t.get("exit_time") else None, "entry": t["entry"],
                     "stop": t["stop"], "target": t.get("target"), "exit": t.get("exit"), "r": t.get("r"),
                     "gap_lo": t["gap_lo"], "gap_hi": t["gap_hi"], "wide": bool(t.get("wide")), "src": "F",
                     "status": t["status"], "reason": t.get("reason"), "setup_t": pd.Timestamp(t["setup_t"]) if t.get("setup_t") else None})
    return rows


def enc(x):
    """one trade: src,dir,flags,leg_start,leg_top,entry_t,exit_t,entry,stop,target,exit,r,gap_lo,gap_hi,reason (times in minutes)"""
    k = dec(x["symbol"])
    m = lambda t: "" if t is None or pd.isna(t) else str(int(pd.Timestamp(t).timestamp() // 60))  # noqa: E731
    p = lambda v: "" if v is None or pd.isna(v) else f"{float(v):.{k}f}"  # noqa: E731
    st = {"pending": "P", "open": "O", "closed": "C"}[x["status"]]
    rs = {"target": "T", "stop": "S", "forced": "F"}.get(x.get("reason") or "", "")
    t_in = x["time"] if x["time"] is not None else x.get("setup_t")
    return ",".join([x["src"], str(x["dir"]), ("W" if x["wide"] else "N") + st, m(x["leg_start"]), m(x["leg_top"]), m(t_in),
                     m(x["exit_time"]), p(x["entry"]), p(x["stop"]), p(x["target"]), p(x["exit"]),
                     "" if x["r"] is None or pd.isna(x["r"]) else f"{float(x['r']):.2f}", p(x["gap_lo"]), p(x["gap_hi"]), rs])


def chunks(s, n=3500):
    return [s[i:i + n] for i in range(0, len(s), n)] or [""]


def pine(rows, days):
    by = {}
    for x in sorted(rows, key=lambda x: (x["time"] or x.get("setup_t") or x["leg_top"])):
        by.setdefault(x["symbol"], []).append(enc(x))
    syms = CORE + EXTRA
    gen = pd.Timestamp.now(tz="Asia/Jerusalem").strftime("%d.%m.%Y %H:%M")
    n_b = sum(r["src"] == "B" for r in rows)
    n_f = sum(r["src"] == "F" for r in rows)
    L = []
    w = L.append
    w("//@version=6")
    w(f"// Taloona Trade - scalping watch bot trades (generated {gen}, Israel time)")
    w(f"// backtest: last {days} days ({n_b} trades) + forward watch-bot trades ({n_f}). Regenerate: python tv_export.py")
    w('indicator("Taloona Trade · סקלפינג", overlay=true, max_boxes_count=500, max_lines_count=500, max_labels_count=500)')
    w("")
    w('symIn   = input.string("אוטומטי", "מכשיר", options=["אוטומטי", ' + ", ".join(f'"{s}"' for s in syms) + "])")
    w('showB   = input.bool(true, "עסקאות מהבדיקה ההיסטורית")')
    w('showF   = input.bool(true, "עסקאות אמת מהבוט (צפייה)")')
    w('showN   = input.bool(true, "מסלול מקורי")')
    w('showW   = input.bool(true, "מסלול רחב")')
    w('showLeg = input.bool(true, "להראות את הדחף והפער")')
    w('oneDay  = input.bool(false, "רק יום אחד")')
    w('dayT    = input.time(timestamp("2026-11-09"), "היום להצגה")')
    w('maxN    = input.int(150, "מקסימום עסקאות על הגרף", minval=1, maxval=160)')
    w("")
    w("norm(string t) =>")
    w('    s = str.replace_all(str.replace_all(str.upper(t), ".CASH", ""), ".", "")')
    w("    s")
    w("autoSym() =>")
    w("    t = norm(syminfo.ticker)")
    w('    string r = ""')
    for s, al in ALIASES.items():
        cond = " or ".join(f't == "{a}"' for a in al)
        w(f'    r := r == "" and ({cond}) ? "{s}" : r')
    fx = [s for s in syms if s not in ALIASES]
    w("    if r == \"\"")
    w("        for s in array.from(" + ", ".join(f'"{s}"' for s in fx) + ")")
    w("            if t == s")
    w("                r := s")
    w("    r")
    w("")
    w('sym = symIn == "אוטומטי" ? autoSym() : symIn')
    w("")
    w("dataOf(string s) =>")
    w('    string d = ""')
    for s in syms:
        recs = by.get(s, [])
        if not recs:
            continue
        body = ";".join(recs)
        parts = " + ".join(f'"{c}"' for c in chunks(body))
        w(f'    if s == "{s}"')
        w(f"        d := {parts}")
    w("    d")
    w("")
    w("f(array<string> a, int i) => array.get(a, i)")
    w("num(string v) => v == \"\" ? float(na) : str.tonumber(v)")
    w("ms(string v) => v == \"\" ? int(na) : int(str.tonumber(v)) * 60000")
    w("")
    w("if barstate.islast")
    w('    if sym == ""')
    w('        label.new(bar_index, high, "Taloona Trade: המכשיר בגרף לא מזוהה - בחר מכשיר בהגדרות", color=color.orange, textcolor=color.white, style=label.style_label_down)')
    w("    else")
    w("        recs = str.split(dataOf(sym), \";\")")
    w("        int shown = 0")
    w("        int n = array.size(recs)")
    w("        for k = n - 1 to 0")
    w("            rec = array.get(recs, k)")
    w("            if rec == \"\" or shown >= maxN")
    w("                continue")
    w("            a = str.split(rec, \",\")")
    w("            src = f(a, 0)")
    w("            dir = int(num(f(a, 1)))")
    w("            flg = f(a, 2)")
    w('            wide = str.substring(flg, 0, 1) == "W"')
    w("            st = str.substring(flg, 1, 2)")
    w("            tIn = ms(f(a, 5))")
    w("            okSrc = (src == \"B\" and showB) or (src == \"F\" and showF)")
    w("            okTrk = (wide and showW) or (not wide and showN)")
    w("            okDay = not oneDay or (tIn >= dayT and tIn < dayT + 86400000)")
    w("            if okSrc and okTrk and okDay")
    w("                shown += 1")
    w("                legS = ms(f(a, 3))")
    w("                legT = ms(f(a, 4))")
    w("                tOut0 = ms(f(a, 6))")
    w("                entry = num(f(a, 7))")
    w("                stop = num(f(a, 8))")
    w("                tgt = num(f(a, 9))")
    w("                ex = num(f(a, 10))")
    w("                r = num(f(a, 11))")
    w("                gLo = num(f(a, 12))")
    w("                gHi = num(f(a, 13))")
    w("                why = f(a, 14)")
    w("                tOut = na(tOut0) ? time + 15 * 60000 : tOut0")
    w("                pend = st == \"P\"")
    w("                if showLeg")
    w("                    line.new(legS, stop, legT, dir > 0 ? math.max(entry, gHi) : math.min(entry, gLo), xloc=xloc.bar_time, color=color.new(color.blue, 30), style=line.style_dotted, width=2)")
    w("                    box.new(legS, gHi, tIn, gLo, xloc=xloc.bar_time, bgcolor=color.new(color.gray, 85), border_color=color.new(color.gray, 50))")
    w("                if not na(tgt)")
    w("                    box.new(tIn, math.max(entry, tgt), tOut, math.min(entry, tgt), xloc=xloc.bar_time, bgcolor=color.new(color.green, pend ? 88 : 75), border_color=color.new(color.green, 30))")
    w("                box.new(tIn, math.max(entry, stop), tOut, math.min(entry, stop), xloc=xloc.bar_time, bgcolor=color.new(color.red, pend ? 88 : 75), border_color=color.new(color.red, 30))")
    w("                line.new(tIn, entry, tOut, entry, xloc=xloc.bar_time, color=color.white, width=1)")
    w('                side = dir > 0 ? "לונג" : "שורט"')
    w('                trk = wide ? "רחב" : "מקורי"')
    w('                orig = src == "F" ? "בוט" : "בדיקה"')
    w('                res = pend ? "מחכה לכניסה" : st == "O" ? "פתוחה" : why == "T" ? "יעד" : why == "S" ? "סטופ" : "סגירה כפויה"')
    w('                rTxt = na(r) ? "" : " · " + str.tostring(r, "#.##") + "R"')
    w('                txt = side + " · " + trk + " · " + orig + "\\n" + res + rTxt + "\\nכניסה " + str.tostring(entry, format.mintick) + "\\nסטופ " + str.tostring(stop, format.mintick) + (na(tgt) ? "" : "\\nיעד " + str.tostring(tgt, format.mintick))')
    w("                lc = pend ? color.gray : na(r) ? color.blue : r > 0 ? color.green : color.red")
    w("                label.new(tIn, dir > 0 ? math.min(entry, stop) : math.max(entry, stop), txt, xloc=xloc.bar_time, color=lc, textcolor=color.white, size=size.small, style=dir > 0 ? label.style_label_up : label.style_label_down)")
    w("                if not na(ex) and not pend")
    w("                    label.new(tOut, ex, \"\", xloc=xloc.bar_time, color=lc, style=label.style_circle, size=size.tiny)")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--state", default="")
    a = ap.parse_args()
    rows = backtest(a.days) + forward(a.state)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(pine(rows, a.days), encoding="utf-8")
    print(f"{OUT}: {len(rows)} trades, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
