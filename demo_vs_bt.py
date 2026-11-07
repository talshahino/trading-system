"""Demo vs backtest, trade by trade (read only, nothing in the bots changes).

For every live bot and every combo it trades, the backtest of the same combo with the live rules (a fresh
run_rules.py run on data up to yesterday) says which trades SHOULD have happened since the bot started.
Each one is matched to the demo account's real trade (same combo, same direction, opened within 2 bars):
    matched   both have it: entry slippage (R), exit difference (R)
    missed    the backtest has it, the demo does not (risk guard, connection, a skip rule, a bug...)
    extra     the demo has it, the backtest does not (a signal that the backtest does not see)
Approved per-combo rules (dir, hours) are applied to the backtest first, so a trade skipped on purpose is not
counted as missed; combos with an approved tp/sl change are marked, their R differs from the backtest by design.

  python demo_vs_bt.py --states <dir with bot*_state.json> --bt 'reports/X/combo_trades_*.csv.gz'
Writes reports/demo_vs_bt/summary.json and reports/demo_vs_bt/trades.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "demo_vs_bt"
TF_MIN = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "4Hour": 240}


def load_bt(pattern):
    parts = [pd.read_csv(f) for f in glob.glob(pattern)]
    if not parts:
        raise SystemExit(f"no backtest files for {pattern}")
    d = pd.concat(parts, ignore_index=True)
    if "variant" in d:      # base variant only, when the file holds several
        base = d["variant"].astype(str).str.contains(r"\|stop1\|entry0$", regex=True)
        if base.any():
            d = d[base]
    if "combo" not in d:
        d["combo"] = d["sid"] + "|" + d["tf"] + "|" + d["symbol"]
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d["exit_time"] = pd.to_datetime(d["exit_time"], utc=True)
    return d.drop_duplicates(["combo", "time", "dir"])


def live_closed(s):
    rows = []
    for c in s.get("closed") or []:
        try:
            side = 1 if c["side"] == "buy" else -1
            o, sl0 = float(c["open_px"]), float(c.get("sl0") or 0)
            risk = side * (o - sl0) if sl0 else np.nan
            rows.append({"id": str(c["id"]), "combo": c["combo"], "dir": side, "open_t": pd.Timestamp(c["open_t"]),
                         "close_t": pd.Timestamp(c["close_t"]), "open_px": o, "sl0": sl0,
                         "r": side * (float(c["close_px"]) - o) / risk if risk and risk > 0 else np.nan,
                         "net": c.get("net"), "why": c.get("why") or c.get("reason") or ""})
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def israel_hour(ts):
    return ts.tz_convert("Asia/Jerusalem").hour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--bt", required=True)
    a = ap.parse_args()
    bt = load_bt(a.bt)
    bt_end = bt["time"].max()
    out_rows, per_combo = [], []
    added_f = ROOT / "data" / "portfolio" / "combo_added.json"     # when each combo entered each bot (git history)
    added = json.loads(added_f.read_text(encoding="utf-8")) if added_f.exists() else {}
    for f in sorted(Path(a.states).glob("bot*_state.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        bot = str(s.get("bot") or f.stem.replace("_state", ""))
        start = pd.Timestamp(s.get("started") or "2026-11-01", tz="UTC") if not str(s.get("started", "")).endswith("+00:00") \
            else pd.Timestamp(s["started"])
        upd = pd.Timestamp(s.get("updated_utc") or pd.Timestamp.now(tz="UTC"))
        if upd.tzinfo is None:
            upd = upd.tz_localize("UTC")
        until = min(upd, bt_end + pd.Timedelta(hours=1))
        lv = live_closed(s)
        rules = s.get("combo_rules") or {}
        mult = s.get("risk_mult") or {}
        for combo in s.get("combos") or []:
            tf = combo.split("|")[1]
            tol = pd.Timedelta(minutes=2 * TF_MIN.get(tf, 60) + 2)
            r = rules.get(combo) or {}
            c_start = start
            if (ad := (added.get(bot.split("_")[0]) or {}).get(combo)):
                c_start = max(start, pd.Timestamp(ad).tz_convert("UTC"))
            b = bt[(bt["combo"] == combo) & (bt["time"] >= c_start + pd.Timedelta(minutes=2)) & (bt["time"] <= until)].copy()
            if mult.get(combo, 1) == 0:
                b = b.iloc[0:0]
            if r.get("dir"):
                b = b[b["dir"] == r["dir"]]
            if r.get("hours"):
                h = b["time"].map(israel_hour)
                b = b[~((h >= r["hours"][0]) & (h < r["hours"][1]))]
            l = lv[(lv["combo"] == combo) & (lv["open_t"] <= until)] if len(lv) else lv
            used = set()
            for _, x in b.iterrows():
                m = None
                if len(l):
                    cand = l[(l["dir"] == x["dir"]) & ((l["open_t"] - x["time"]).abs() <= tol) & ~l["id"].isin(used)]
                    if len(cand):
                        m = cand.iloc[(cand["open_t"] - x["time"]).abs().argsort().iloc[0]]
                risk_bt = abs(float(x["entry_px"]) - float(x["stop_px"]))
                row = {"bot": bot, "combo": combo, "dir": int(x["dir"]), "bt_time": x["time"], "bt_entry": x["entry_px"],
                       "bt_r": round(float(x["r"]), 3), "bt_exit_time": x["exit_time"], "bt_reason": x.get("reason", ""),
                       "rule_change": ",".join(k for k in ("tp", "sl") if k in r)}
                if m is None:
                    # a trade still open in the backtest at the bot's last update cannot be judged yet
                    row["status"] = "missed" if x["time"] < upd - pd.Timedelta(minutes=10) else "too new"
                else:
                    used.add(m["id"])
                    row.update({"status": "matched", "live_id": m["id"], "live_time": m["open_t"], "live_entry": m["open_px"],
                                "live_r": round(float(m["r"]), 3) if pd.notna(m["r"]) else None, "live_net": m["net"],
                                "live_why": m["why"],
                                "slip_r": round(float(x["dir"]) * (float(x["entry_px"]) - float(m["open_px"])) / risk_bt, 3)
                                if risk_bt else None,
                                "delay_min": round((m["open_t"] - x["time"]).total_seconds() / 60, 1)})
                out_rows.append(row)
            for _, m in (l[~l["id"].isin(used)].iterrows() if len(l) else []):
                if m["open_t"] < c_start:
                    continue
                out_rows.append({"bot": bot, "combo": combo, "dir": int(m["dir"]), "status": "extra", "live_id": m["id"],
                                 "live_time": m["open_t"], "live_entry": m["open_px"],
                                 "live_r": round(float(m["r"]), 3) if pd.notna(m["r"]) else None, "live_net": m["net"],
                                 "live_why": m["why"], "rule_change": ",".join(k for k in ("tp", "sl") if k in r)})
    t = pd.DataFrame(out_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "trades.csv", index=False)
    res = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "backtest_until": str(bt_end), "bots": {}}
    for bot, g in t.groupby("bot"):
        mt = g[g["status"] == "matched"]
        clean = mt[mt["rule_change"].fillna("") == ""]
        res["bots"][bot] = {
            "backtest_trades": int((g["status"].isin(["matched", "missed"])).sum()),
            "matched": int(len(mt)), "missed": int((g["status"] == "missed").sum()), "extra": int((g["status"] == "extra").sum()),
            "avg_slip_r": round(float(mt["slip_r"].mean()), 3) if len(mt) else None,
            "avg_delay_min": round(float(mt["delay_min"].mean()), 1) if len(mt) else None,
            "matched_live_r": round(float(clean["live_r"].mean()), 3) if len(clean) else None,
            "matched_bt_r": round(float(clean["bt_r"].mean()), 3) if len(clean) else None,
            "missed_bt_r_sum": round(float(g.loc[g["status"] == "missed", "bt_r"].sum()), 2),
            "extra_live_r_sum": round(float(g.loc[g["status"] == "extra", "live_r"].fillna(0).sum()), 2)}
    for (bot, combo), g in t.groupby(["bot", "combo"]):
        mt = g[g["status"] == "matched"]
        per_combo.append({"bot": bot, "combo": combo, "bt": int(g["status"].isin(["matched", "missed"]).sum()),
                          "matched": int(len(mt)), "missed": int((g["status"] == "missed").sum()),
                          "extra": int((g["status"] == "extra").sum()),
                          "live_r": round(float(mt["live_r"].dropna().sum()), 2), "bt_r": round(float(mt["bt_r"].sum()), 2),
                          "slip_r": round(float(mt["slip_r"].mean()), 3) if len(mt) else None,
                          "rule_change": g["rule_change"].fillna("").iloc[0]})
    res["combos"] = sorted(per_combo, key=lambda x: -(x["missed"] + x["extra"]))
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res["bots"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
