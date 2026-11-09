"""Scalping watch bot (user, 06.11.2026): the user's own impulse strategy, live, WITHOUT money.

Every minute it reads the last days of 1-minute candles of the low-cost instruments from bot 1's MetaTrader
(bot 1's local candle server - read-only, nothing is ever sent to a broker) and runs exactly the tested rules
(agents/own_core.py, variant 'struct'):
  impulse on the 15-minute chart -> the first gap of the move -> the 1-minute gap inside it -> entry on the touch
  stop 4 pips beyond the impulse's start, target 2 pips before the last real correction before the entry
Each setup is followed: waiting for the touch ('pending'), virtual trade open ('open'), closed at the target /
stop ('closed'). The phone gets a notification at each step, and the state (with the candles of every trade for
its charts: 15 minutes for the impulse and the entry, 1 minute for the rest) goes to live/watch_own_state.json.
The user marks each trade in the app (would take / would not, and why): the manager keeps the marks in
live/watch_marks.json and this bot copies them into the state. Forward data to tune the rules.

  started by the manager (env SOURCE_PORT = bot 1's local port, MANAGER_KEY)
  python live/watch_own.py [--once]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import ftmo_calendar as CAL  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import own_core as CORE  # noqa: E402
from live import reporter as REP  # noqa: E402

NAME = "watch_own"
LABEL = "בוט סקלפינג · צפייה בלבד"
LIVE = ROOT / "live"
STATE_F = LIVE / f"{NAME}_state.json"
MARKS_F = LIVE / "watch_marks.json"
VARIANT = (2.0, False, "1m", "struct")
# wide track (user, 09.11.2026: "I want more trades, it's scalping"): the same rules with a weaker candle-body filter
# (1.8 instead of 2.5 x the average body) - about 4x the setups in the backtest (3.6 a week on these 16 instruments
# since 2024, vs 0.9), but they lost a little there (-0.09R a trade vs -0.03R). Shown apart in the app ('wide').
WIDE_BODY_X = 1.8
DAYS = 8                      # history kept per instrument (the impulse rules look back up to ~5 days)
# the instruments where spread + commission are the smallest share of a trade's risk (optimisation, 06.11.2026)
SYMBOLS = {"US100": ["US100.cash", "US100"], "US30": ["US30.cash", "US30"], "US500": ["US500.cash", "US500"],
           "GER40": ["GER40.cash", "GER40"], "FRA40": ["FRA40.cash", "FRA40"], "JP225": ["JP225.cash", "JP225"],
           "UK100": ["UK100.cash", "UK100"], "XAUUSD": ["XAUUSD", "XAU/USD"], "EURUSD": ["EURUSD"], "EURJPY": ["EURJPY"],
           "USDJPY": ["USDJPY"], "GBPUSD": ["GBPUSD"], "GBPJPY": ["GBPJPY"], "USDCHF": ["USDCHF"], "AUDUSD": ["AUDUSD"],
           "NZDUSD": ["NZDUSD"]}
KEEP = 60                     # closed trades kept in the state (with their candles)
# virtual demo account (user, 06.11.2026): $100,000 like the other demo accounts, risk per trade as bots 2 and 4,
# the bots' own daily stop (4.5%, FTMO's limit is 5%) and FTMO's 10% maximum loss
INITIAL, RISK, DAILY_STOP, MAX_LOSS = 100_000.0, 0.0045, 0.045, 0.10


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LIVE / f"{NAME}.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


class Source:
    """1-minute candles from bot 1's MetaTrader, through its local candle server (GET /candles)."""

    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}/candles"
        self.broker = {}
        self.df = {}

    def get(self, sym, t0, t1):
        r = requests.get(self.url, params={"symbol": sym, "tf": "1m", "from": int(t0), "to": int(t1)}, timeout=20)
        j = r.json()
        if j.get("error"):
            raise RuntimeError(j["error"])
        b = j.get("bars") or []
        if not b:
            return pd.DataFrame(columns=list("ohlcv"))
        a = np.array(b, dtype=float)
        return pd.DataFrame({"o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": 1.0},
                            index=pd.to_datetime(a[:, 0].astype("int64"), unit="s", utc=True))

    def resolve(self, name):
        now = time.time()
        for cand in SYMBOLS[name]:
            try:
                d = self.get(cand, now - 5 * 86400, now)
                if len(d):
                    return cand
            except Exception:  # noqa: BLE001
                continue
        return None

    def update(self, name):
        if name not in self.df:
            b = self.broker.get(name) or self.resolve(name)
            if not b:
                return None
            self.broker[name] = b
            parts, now, step = [], time.time(), 2 * 86400       # the server returns at most 3000 candles a call
            for k in range(DAYS // 2, 0, -1):
                parts.append(self.get(b, now - k * step, now - (k - 1) * step))   # bot 1 busy: retried next minute
            self.df[name] = pd.concat(parts)
        else:
            last = self.df[name].index[-1].timestamp() if len(self.df[name]) else time.time() - DAYS * 86400
            new = self.get(self.broker[name], last - 180, time.time())
            self.df[name] = pd.concat([self.df[name], new])
        d = self.df[name]
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d = d[d.index >= d.index[-1] - pd.Timedelta(days=DAYS)] if len(d) else d
        # the candle still forming is left out: the rules decide on closed candles only
        if len(d) and d.index[-1] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1):
            d = d.iloc[:-1]
        self.df[name] = d
        return d


class Calendar:
    """FTMO's restricted events (2 minutes either side) and holidays, refreshed every hour."""

    def __init__(self):
        self.t, self.restricted, self.holidays = 0, [], []

    def refresh(self):
        if time.time() - self.t < 3600:
            return
        try:
            now = pd.Timestamp.now(tz="UTC")
            self.restricted, self.holidays = CAL.parse(CAL.fetch((now - pd.Timedelta(days=DAYS + 1)).strftime("%Y-%m-%d"),
                                                                 (now + pd.Timedelta(days=2)).strftime("%Y-%m-%d")))
            self.t = time.time()
        except Exception as e:  # noqa: BLE001
            log(f"FTMO calendar not read ({e}); trying again in 10 minutes")
            self.t = time.time() - 3000

    def masks(self, df, name):
        n = len(df)
        idx = df.index
        start = idx.to_series()
        gap = start.shift(-1) - (start + pd.Timedelta(minutes=1))
        stop_at = np.array((gap > pd.Timedelta(hours=2)).to_numpy(), dtype=bool)      # a writable copy
        noent = np.zeros(n, bool)
        try:
            ex, ne = CAL.masks(df, name, 1, self.restricted, self.holidays)
            stop_at = stop_at | np.asarray(ex, bool)
            noent = noent | np.asarray(ne, bool)
        except Exception:  # noqa: BLE001
            pass
        for t, ins in self.restricted:
            if not CAL.restricted_affects(name, ins):
                continue
            a, b = idx.searchsorted(t - pd.Timedelta(minutes=2)), idx.searchsorted(t + pd.Timedelta(minutes=2), side="right")
            noent[max(0, a - 1):b] = True                    # no entry from 2 minutes before to 2 after
            if a - 1 >= 0:
                stop_at[a - 1] = True                         # out 2 minutes before
        return stop_at, np.r_[noent[1:], False]


def cost_of(name, df):
    c = df["c"].to_numpy(float)
    cls = U.CLASS.get(name, "fx")
    return c * C.EST_SPREAD.get(cls, 0.0002) + CORE.comm_px(name, c)


def bars(df, t0, t1, rule):
    x = df[(df.index >= t0) & (df.index <= t1)]
    if rule != "1min":
        x = CORE.resample(x, rule)
    return [[int(t.timestamp()), round(float(r.o), 6), round(float(r.h), 6), round(float(r.l), 6), round(float(r.c), 6)]
            for t, r in x.iterrows()]


def tid(x):
    return f"{x['symbol']}|{pd.Timestamp(x['leg_top']).strftime('%Y%m%d%H%M')}|{int(x['dir'])}"


def iso(t):
    return pd.Timestamp(t).isoformat() if t is not None and not (isinstance(t, float) and np.isnan(t)) else None


def fmt(name, p):
    return f"{p:.2f}" if not (len(name) == 6 and name.isalpha()) else (f"{p:.3f}" if name.endswith("JPY") else f"{p:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    port = os.environ.get("SOURCE_PORT")
    if not port:
        log("no SOURCE_PORT (bot 1's local port): the manager starts this bot")
        return
    src, cal = Source(port), Calendar()
    try:
        state = json.loads(STATE_F.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        state = {}
    state.update({"bot": NAME, "label": LABEL, "watch_only": True, "symbols": list(SYMBOLS),
                  "rules": "דחף ב־15 דקות · כניסה בפער הדקתי בתוך הפער הראשון של המהלך · סטופ 4 פיפס מעבר לתחילת הדחף · "
                           "יעד 2 פיפס לפני התיקון האמיתי האחרון לפני הכניסה · יחס מינימום 1:1 · מסלול רחב: אותם כללים עם נרות חזקים פחות (פי 1.8 מהממוצע במקום 2.5), בערך פי 4 עסקאות"})
    trades = {t["id"]: t for t in (state.get("watch") or [])}
    acc = state.get("vacct") or {"initial": INITIAL, "risk_pct": round(RISK * 100, 3), "ledger": [], "breached": None}

    def balance():
        return acc["initial"] + sum(x["pnl"] for x in acc["ledger"])

    def prague_day(ts):
        return pd.Timestamp(ts).tz_convert("Europe/Prague").strftime("%Y-%m-%d")

    def day_pnl(day):
        return sum(x["pnl"] for x in acc["ledger"] if prague_day(x["exit_time"]) == day)
    REP.remember_alerts(LIVE / f"{NAME}_alerts.json")
    last_push = 0
    while True:
        t_cycle = time.time()
        cal.refresh()
        seen_now, errors = set(), []
        for name in SYMBOLS:
            try:
                df = src.update(name)
                if df is None or len(df) < 3000:
                    continue
                stop_at, noent = cal.masks(df, name)
                cst = cost_of(name, df)
                res = CORE.run_symbol(name, df, cst, stop_at, noent, [VARIANT], live=True)
                strict = {tid(x) for x in res}
                keep_bx = CORE.BODY_X
                try:
                    CORE.BODY_X = WIDE_BODY_X
                    wide = CORE.run_symbol(name, df, cst, stop_at, noent, [VARIANT], live=True)
                finally:
                    CORE.BODY_X = keep_bx
                for x in wide:
                    if tid(x) not in strict:              # a setup both tracks find is the original one
                        x["wide"] = True
                        res.append(x)
                cutoff = df.index[-1] - pd.Timedelta(days=2)       # only what happened recently is reported
                for x in res:
                    if pd.Timestamp(x.get("time") or x.get("setup_t")) < cutoff:
                        continue
                    k = tid(x)
                    seen_now.add(k)
                    old = trades.get(k, {})
                    rec = {"id": k, "symbol": name, "broker": src.broker.get(name), "dir": int(x["dir"]), "status": x["status"],
                           "entry": float(x["entry"]), "stop": float(x["stop"]), "target": float(x["target"]) if x.get("target") else None,
                           "rr": x.get("rr"), "cost_r": x.get("cost_r"), "leg": x.get("leg"),
                           "leg_start": iso(x["leg_start"]), "leg_top": iso(x["leg_top"]), "gap_lo": float(x["gap_lo"]), "gap_hi": float(x["gap_hi"]),
                           "setup_t": iso(x.get("setup_t")) or old.get("setup_t"), "time": iso(x.get("time")), "exit_time": iso(x.get("exit_time")),
                           "exit": float(x["exit"]) if x.get("exit") is not None else None, "reason": x.get("reason"),
                           "r": x.get("r"), "r_now": x.get("r_now"), "mark": old.get("mark"), "wide": bool(x.get("wide"))}
                    if x["status"] != "pending" or "m15" not in old or old.get("status") != x["status"]:
                        ls = pd.Timestamp(x["leg_start"])
                        t_in = pd.Timestamp(x.get("time")) if x.get("time") is not None else df.index[-1]
                        t_out = pd.Timestamp(x["exit_time"]) if x.get("exit_time") is not None else df.index[-1]
                        rec["m15"] = bars(df, ls - pd.Timedelta(hours=6), t_in + pd.Timedelta(hours=1), "15min")
                        rec["m1"] = bars(df, t_in - pd.Timedelta(minutes=45), t_out + pd.Timedelta(minutes=10), "1min")
                    else:
                        rec["m15"], rec["m1"] = old.get("m15"), old.get("m1")
                    prev = old.get("status")
                    # the virtual account: the risk is fixed when the trade opens, from the balance at that moment
                    for f in ("risk_usd", "skipped"):
                        if f in old:
                            rec[f] = old[f]
                    if rec["status"] in ("open", "closed") and "risk_usd" not in rec and "skipped" not in rec:
                        bal, day = balance(), prague_day(rec["time"])
                        start = bal - day_pnl(day)
                        if acc.get("breached"):
                            rec["skipped"] = "breach"
                        elif day_pnl(day) <= -DAILY_STOP * start:
                            rec["skipped"] = "daily"
                        else:
                            rec["risk_usd"] = round(bal * RISK, 2)
                    if rec.get("risk_usd"):
                        if rec["status"] == "closed" and rec.get("r") is not None:
                            rec["pnl"] = round(rec["r"] * rec["risk_usd"], 2)
                            if not any(x["id"] == k for x in acc["ledger"]):
                                acc["ledger"].append({"id": k, "exit_time": rec["exit_time"], "pnl": rec["pnl"]})
                                if balance() <= acc["initial"] * (1 - MAX_LOSS) and not acc.get("breached"):
                                    acc["breached"] = rec["exit_time"]
                                    REP.notify("סקלפינג (צפייה): החשבון הווירטואלי נשרף", f"היתרה ירדה ל־${balance():,.0f} (10% הפסד). לא נפתחות עוד עסקאות וירטואליות.",
                                               tags=["eyes"], key="wbreach", every=86400 * 30)
                        elif rec["status"] == "open" and rec.get("r_now") is not None:
                            rec["pnl_now"] = round(rec["r_now"] * rec["risk_usd"], 2)
                    trades[k] = rec
                    side = ("קנייה" if rec["dir"] > 0 else "מכירה") + (" · מסלול רחב" if rec["wide"] else "")
                    if prev is None and rec["status"] == "pending":
                        REP.notify(f"סקלפינג (צפייה): {name} מחכה לכניסה", f"{side} · דחף של {rec['leg']} נרות · כניסה בנגיעה ב־{fmt(name, rec['entry'])}\n"
                                   f"סטופ {fmt(name, rec['stop'])} · לא נשלחת פקודה", tags=["eyes"], key=f"wp:{k}", every=86400)
                    if prev in (None, "pending") and rec["status"] in ("open", "closed"):
                        REP.notify(f"סקלפינג (צפייה): {name} עסקה וירטואלית נפתחה", f"{side} @ {fmt(name, rec['entry'])} · סטופ {fmt(name, rec['stop'])} · "
                                   f"יעד {fmt(name, rec['target'])} · יחס 1:{rec['rr']}\nפתח את האפליקציה וסמן אם היית לוקח אותה",
                                   tags=["eyes"], key=f"wo:{k}", every=86400)
                    if prev != "closed" and rec["status"] == "closed":
                        word = {"target": "ביעד", "stop": "בסטופ", "forced": "בסגירה כפויה"}.get(rec["reason"], "")
                        usd = f" · {'+' if rec.get('pnl', 0) >= 0 else '−'}${abs(rec.get('pnl', 0)):,.0f} · יתרה ${balance():,.0f}" if rec.get("pnl") is not None else ""
                        REP.notify(f"סקלפינג (צפייה): {name} נסגרה {word}", f"{side} · {'+' if (rec['r'] or 0) >= 0 else ''}{rec['r']:.2f} סיכון{usd} (וירטואלי)",
                                   tags=["eyes"], key=f"wc:{k}", every=86400)
            except Exception:  # noqa: BLE001 - one instrument must not stop the others
                errors.append(f"{name}: {traceback.format_exc()[-300:]}")
        # a pending setup that no longer appears (cancelled: the leg went on, expired, stop first) is dropped
        for k, t in list(trades.items()):
            if t["status"] == "pending" and k not in seen_now and t["symbol"] in src.df:
                t["status"] = "cancelled"
        try:
            marks = json.loads(MARKS_F.read_text(encoding="utf-8"))
            for k, m in marks.items():
                if k in trades:
                    trades[k]["mark"] = m
        except Exception:  # noqa: BLE001
            pass
        closed = sorted([t for t in trades.values() if t["status"] == "closed"], key=lambda t: t["time"] or "")[-KEEP:]
        live_ = [t for t in trades.values() if t["status"] in ("pending", "open")]
        cancelled = sorted([t for t in trades.values() if t["status"] == "cancelled"], key=lambda t: t.get("setup_t") or "")[-20:]
        trades = {t["id"]: t for t in closed + live_ + cancelled}
        rs = [t["r"] for t in closed if t.get("r") is not None]
        rs_strict = [t["r"] for t in closed if t.get("r") is not None and not t.get("wide")]
        rs_wide = [t["r"] for t in closed if t.get("r") is not None and t.get("wide")]
        state.update({"updated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                      "watch": sorted(trades.values(), key=lambda t: t.get("time") or t.get("setup_t") or "", reverse=True),
                      "summary": {"closed": len(rs), "sum_r": round(float(np.sum(rs)), 2) if rs else 0.0,
                                  "avg_r": round(float(np.mean(rs)), 3) if rs else None, "win_pct": round(100 * float(np.mean(np.array(rs) > 0)), 1) if rs else None,
                                  "open": sum(t["status"] == "open" for t in live_), "pending": sum(t["status"] == "pending" for t in live_),
                                  "strict": {"closed": len(rs_strict), "sum_r": round(float(np.sum(rs_strict)), 2) if rs_strict else 0.0},
                                  "wide": {"closed": len(rs_wide), "sum_r": round(float(np.sum(rs_wide)), 2) if rs_wide else 0.0}},
                      "instruments": {k: src.broker.get(k) for k in SYMBOLS}, "errors": errors[-5:]})
        bal = balance()
        eq = bal + sum(t.get("pnl_now") or 0 for t in live_ if t["status"] == "open")
        today = prague_day(pd.Timestamp.now(tz="UTC"))
        curve, run = [[None, acc["initial"]]], acc["initial"]
        for x in sorted(acc["ledger"], key=lambda x: x["exit_time"] or ""):
            run += x["pnl"]
            curve.append([x["exit_time"], round(run, 2)])
        acc.update({"balance": round(bal, 2), "equity": round(eq, 2), "profit": round(bal - acc["initial"], 2),
                    "today": round(day_pnl(today), 2), "curve": curve[-300:]})
        # the same fields as a real bot's state, so the app shows this account like the demo accounts
        now_iso = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        ec = acc.setdefault("eq_curve", [])
        if not ec or ec[-1][1] != round(bal, 2) or ec[-1][2] != round(eq, 2) or now_iso[:15] != ec[-1][0][:15]:
            ec.append([now_iso, round(bal, 2), round(eq, 2)])
            del ec[:-4000]
        acc.setdefault("started", now_iso)
        pip = lambda n: 0.01 if n.endswith("JPY") else 0.0001 if (len(n) == 6 and n.isalpha()) else 1.0  # noqa: E731
        combo = lambda n: f"OWN|15Min|{n}"  # noqa: E731
        counted = [t for t in closed if t.get("risk_usd") and t.get("pnl") is not None]
        state.update({"account": "וירטואלי", "server": "צפייה בלבד · בלי כסף", "initial": acc["initial"], "risk": RISK,
                      "balance": round(bal, 2), "equity": round(eq, 2), "day_start": round(bal - day_pnl(today), 2),
                      "halted_until": None, "combos": [combo(n) for n in SYMBOLS], "started": acc["started"], "withdrawals": [],
                      "equity_curve": ec, "lows": {},
                      "positions": [{"ticket": t["id"], "side": "buy" if t["dir"] > 0 else "sell", "dir": "קנייה" if t["dir"] > 0 else "מכירה",
                                     "combo": combo(t["symbol"]), "symbol": t["symbol"], "lots": "וירטואלי", "open": t["entry"],
                                     "now": t["entry"] + t["dir"] * (t.get("r_now") or 0) * abs(t["entry"] - t["stop"]),
                                     "sl": t["stop"], "tp": t["target"], "sl0": t["stop"], "tp0": t["target"],
                                     "sl_usd": -(t.get("risk_usd") or 0), "tp_usd": (t.get("risk_usd") or 0) * (t.get("rr") or 0),
                                     "pnl": t.get("pnl_now") or 0, "since": t["time"], "pip": pip(t["symbol"]), "free": False,
                                     "mae": None, "mfe": None} for t in live_ if t["status"] == "open" and t.get("risk_usd")],
                      "trades": [{"time": t["exit_time"], "combo": combo(t["symbol"]), "symbol": t["symbol"],
                                  "dir": "קנייה" if t["dir"] > 0 else "מכירה", "lots": "וירטואלי", "price": t["exit"], "pnl": t["pnl"],
                                  "why": t.get("reason")} for t in counted],
                      "closed": [{"id": t["id"], "combo": combo(t["symbol"]), "symbol": t["symbol"], "side": "buy" if t["dir"] > 0 else "sell",
                                  "lots": "וירטואלי", "open_t": t["time"], "open_px": t["entry"], "close_t": t["exit_time"], "close_px": t["exit"],
                                  "sl0": t["stop"], "tp0": t["target"], "sl": t["stop"], "tp": t["target"], "mae": None, "mfe": None,
                                  "gross": t["pnl"], "swap": 0, "comm": 0, "net": t["pnl"], "why": t.get("reason") or "",
                                  "reason": {"target": "יעד", "stop": "סטופ", "forced": "סגירה כפויה"}.get(t.get("reason"), ""),
                                  "pip": pip(t["symbol"])} for t in counted]})
        state["vacct"] = acc
        try:
            STATE_F.write_text(json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log(f"state not saved: {e}")
        if time.time() - last_push > 120:
            err = REP.push_state(NAME, state)
            if err:
                log(err)
            last_push = time.time()
        if errors:
            log("; ".join(e.splitlines()[-1] for e in errors)[:400])
        if a.once:
            return
        time.sleep(max(5.0, 60 - (time.time() - t_cycle)))


if __name__ == "__main__":
    main()
