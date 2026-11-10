"""Real trading costs on the bots' accounts vs the costs the backtests charge (user, 10.11.2026).

For every closed trade of the 4 bots: the backtest's spread at the entry minute (Dukascopy, or the estimate for
instruments without bid/ask history), commission and swap (FTMO spec, agents/ftmo_costs.py), against what the
account really paid: commission and swap from MetaTrader's deals, spread from the bot's own samples of the live
spread for that instrument (median). All in dollars per trade (units traded = gross / price move).
Writes reports/costs_check.json and prints a table. Read-only for the bots.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from agents import ftmo_costs as C
from agents import ftmo_universe as U
from agents import poc_agent as P

BOTS = {"bot1": "בוט 1", "bot2_5058299955": "בוט 2", "bot3_10013442846": "בוט 3", "bot4_115118875": "בוט 4"}


ALIAS = {"USOIL": "WTI", "UKOIL": "BRENT", "N25": "NL25", "DXY": "DXY"}


def norm(s):
    b = str(s).split(".")[0]
    return b if b in U.CLASS else ALIAS.get(b, b)


def state(name):
    subprocess.run(["git", "fetch", "-q", "origin", "live"], check=False)
    raw = subprocess.run(["git", "show", f"origin/live:live/{name}_state.json"], capture_output=True, text=True).stdout
    return json.loads(raw)


def main():
    out, rows = {}, []
    sp_cache = {}
    for name, label in BOTS.items():
        d = state(name)
        c = pd.DataFrame(d.get("closed", []))
        if c.empty:
            continue
        live_sp = {s: float(v["median"]) if isinstance(v, dict) else float(np.median(v))
                   for s, v in (d.get("spreads") or {}).items() if v}
        c["symbol"] = c["symbol"].map(norm)
        c = c[c["symbol"].isin(U.CLASS.keys())].copy()
        c["dir"] = np.where(c["side"] == "buy", 1, -1)
        c["time"] = pd.to_datetime(c["open_t"], utc=True)
        c["exit_time"] = pd.to_datetime(c["close_t"], utc=True)
        move = c["dir"] * (c["close_px"] - c["open_px"])
        c["units"] = np.where(move.abs() > 0, c["gross"] / move.replace(0, np.nan), np.nan)
        for sym, g in c.groupby("symbol"):
            g = g[g["units"] > 0]
            if g.empty:
                continue
            if sym not in sp_cache:
                try:
                    sp_cache[sym] = P.spread_1m(sym, (g["time"].min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                                                (pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%d"))
                except Exception as e:  # noqa: BLE001
                    print(f"{sym}: no spread history ({e})")
                    sp_cache[sym] = pd.Series(dtype="float64")
            t = pd.DataFrame({"r": 0.0, "entry_px": g["open_px"].astype(float), "stop_px": g["sl0"].astype(float),
                              "dir": g["dir"], "time": g["time"], "exit_time": g["exit_time"]})
            m = C.apply(t, sym, sp_cache[sym])
            u = g["units"].to_numpy(float)
            bt_spread = float((m["spread_px"].to_numpy(float) * u).sum())
            bt_comm = float((m["comm_px"].to_numpy(float) * u).sum())
            bt_swap = float((m["swap_px"].to_numpy(float) * u).sum())      # + = paid to us
            lv = live_sp.get(sym)
            live_spread = float(lv * u.sum()) if lv is not None else None
            rows.append({"bot": label, "server": d.get("server"), "symbol": sym, "trades": int(len(g)),
                         "bt_spread": round(bt_spread, 2), "live_spread": None if live_spread is None else round(live_spread, 2),
                         "bt_spread_est": bool(m["spread_est"].iloc[0]),
                         "bt_comm": round(bt_comm, 2), "live_comm": round(float(-g["comm"].sum()), 2),
                         "bt_swap": round(bt_swap, 2), "live_swap": round(float(g["swap"].sum()), 2),
                         "gross": round(float(g["gross"].sum()), 2)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string())
    for label, g in df.groupby("bot"):
        out[label] = {k: round(float(g[k].fillna(0).sum()), 2) for k in
                      ("trades", "bt_spread", "live_spread", "bt_comm", "live_comm", "bt_swap", "live_swap", "gross")}
        out[label]["server"] = g["server"].iloc[0]
    print(json.dumps(out, ensure_ascii=False, indent=1))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/costs_check.json").write_text(json.dumps({"by_bot": out, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
