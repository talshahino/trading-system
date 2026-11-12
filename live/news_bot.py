"""Bot 7 (user, 10.11.2026): economic releases, live, WITHOUT money - a virtual $100,000 account.

Why a virtual bot: the three release strategies (agents/event_strategies.py) could not be judged on history -
FTMO's calendar only goes back to April 2025, so each instrument had 5-45 trades (runs 405-407, 03.11.2026).
This bot collects those trades forward, from now on, with FTMO's real costs.

Every minute it reads the last days of 1-minute candles from bot 1's MetaTrader (bot 1's local candle server,
read-only, nothing is ever sent to a broker), builds 5-minute candles and runs exactly the backtest's rules:
  EV01 release momentum  - the first 10 minutes moved > 1.5 ATR: follow them, stop beyond them, target 2R, max 1 hour
  EV02 release fade      - a spike > 2.5 ATR already giving back a third: against it, target the pre-release price
  EV03 range breakout    - the first 15 minutes' range, the first close outside it within the hour, target 1.5R
Releases: FTMO's own economic calendar, high impact or FTMO-restricted, concerning the instrument (read every hour).
Entries are 10+ minutes after the release, so FTMO's restricted window (3 minutes either side) is never touched.
Trade management as the live bots: one position per strategy and instrument, breakeven at 60% of the way to the
target (+ costs). Costs: spread (bot 1's own live samples of FTMO's spread), FTMO commission and swap.
Virtual account: $100,000, RISK of the balance per trade, the bots' daily stop (4.5%) and FTMO's 10% maximum loss.

  started by the manager (env SOURCE_PORT = bot 1's local port, MANAGER_KEY)
  python live/news_bot.py [--once]
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

from agents import rules_agent as R  # noqa: E402,F401  (import first: rules_agent and event_strategies import each other)
from agents import event_strategies as E  # noqa: E402
from agents import ftmo_calendar as CAL  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from live import reporter as REP  # noqa: E402

NAME = "news_bot"
LABEL = "בוט 7 · פרסומים כלכליים (וירטואלי)"
LIVE = ROOT / "live"
STATE_F = LIVE / f"{NAME}_state.json"
DAYS = 6                       # candles kept (the strategies need ~2 hours after a release and 20 bars before it)
SYMBOLS = {"US100": ["US100.cash", "US100"], "US500": ["US500.cash", "US500"], "US30": ["US30.cash", "US30"],
           "GER40": ["GER40.cash", "GER40"], "UK100": ["UK100.cash", "UK100"], "JP225": ["JP225.cash", "JP225"],
           "XAUUSD": ["XAUUSD", "XAU/USD"], "EURUSD": ["EURUSD"], "GBPUSD": ["GBPUSD"], "USDJPY": ["USDJPY"],
           "AUDUSD": ["AUDUSD"], "USDCAD": ["USDCAD"], "USDCHF": ["USDCHF"], "NZDUSD": ["NZDUSD"], "EURJPY": ["EURJPY"],
           "GBPJPY": ["GBPJPY"]}
STRATS = ["EV01", "EV02", "EV03"]
NAMES = {"EV01": "הליכה עם הפרסום", "EV02": "נגד הקפיצה", "EV03": "פריצת טווח"}
END = "סוף הנתונים"            # rules_agent's reason for a trade still open at the last candle
INITIAL, RISK, DAILY_STOP, MAX_LOSS = 100_000.0, 0.005, 0.045, 0.10
ACCT_V = 1
KEEP = 300
PLANS_F = LIVE / "bots" / "bot7.json"   # approved cards, built (agent_portfolio.built_variant reads the same "plans")


def plans():
    """Per combo ("EV01|5Min|GBPUSD"): {"filter": "with_trend200"} - market-regime filters from approved cards."""
    try:
        return json.loads(PLANS_F.read_text(encoding="utf-8")).get("plans") or {}
    except Exception:  # noqa: BLE001
        return {}


def regime(d, name):
    """The same masks as run_rules._regimes, known at the bar's close: (allow_long, allow_short)."""
    h, l, c = (d[x].to_numpy() for x in "hlc")
    if name == "with_trend200":
        e = R.ema(c, 200)
        return c > e, c < e
    if name in ("adx_trend", "adx_range"):
        a = R.adx(h, l, c, 14)
        m = a > 20 if name == "adx_trend" else a < 20
        return m, m
    at = R.atr(h, l, c, 14) / c
    pr = pd.Series(at).rolling(500, min_periods=100).rank(pct=True).to_numpy()
    m = {"vol_normal": (pr > 0.2) & (pr < 0.9), "vol_high": pr >= 0.5, "vol_low": pr < 0.5}[name]
    return m, m


def filtered(fn, flt):
    if not flt:
        return fn

    def g(d):
        L, S, plan = fn(d)
        al, ash = regime(d, flt)
        return np.asarray(L, bool) & al, np.asarray(S, bool) & ash, plan
    return g


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
        self.broker, self.df = {}, {}

    def get(self, sym, t0, t1):
        j = requests.get(self.url, params={"symbol": sym, "tf": "1m", "from": int(t0), "to": int(t1)}, timeout=20).json()
        if j.get("error"):
            raise RuntimeError(j["error"])
        b = j.get("bars") or []
        if not b:
            return pd.DataFrame(columns=list("ohlcv"))
        a = np.array(b, dtype=float)
        return pd.DataFrame({"o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "v": 1.0},
                            index=pd.to_datetime(a[:, 0].astype("int64"), unit="s", utc=True))

    def update(self, name):
        if name not in self.df:
            b = self.broker.get(name)
            if not b:
                for cand in SYMBOLS[name]:
                    try:
                        if len(self.get(cand, time.time() - 5 * 86400, time.time())):
                            b = cand
                            break
                    except Exception:  # noqa: BLE001
                        continue
            if not b:
                return None
            self.broker[name] = b
            now, step = time.time(), 2 * 86400                  # the server returns at most 3000 candles a call
            self.df[name] = pd.concat([self.get(b, now - k * step, now - (k - 1) * step) for k in range(DAYS // 2, 0, -1)])
        else:
            last = self.df[name].index[-1].timestamp() if len(self.df[name]) else time.time() - DAYS * 86400
            self.df[name] = pd.concat([self.df[name], self.get(self.broker[name], last - 180, time.time())])
        d = self.df[name]
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d = d[d.index >= d.index[-1] - pd.Timedelta(days=DAYS)] if len(d) else d
        if len(d) and d.index[-1] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=1):
            d = d.iloc[:-1]                                     # the forming candle is left out
        self.df[name] = d
        return d


def five(d1):
    """5-minute candles from the 1-minute ones; a 5-minute candle counts only once all its minutes closed."""
    d = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    d = d.dropna(subset=["o"])
    if len(d) and d.index[-1] + pd.Timedelta(minutes=5) > d1.index[-1] + pd.Timedelta(minutes=1):
        d = d.iloc[:-1]
    return d


class Calendar:
    def __init__(self):
        self.t, self.items = 0, []

    def refresh(self):
        if time.time() - self.t < 3600:
            return
        try:
            now = pd.Timestamp.now(tz="UTC")
            self.items = CAL.fetch((now - pd.Timedelta(days=DAYS + 1)).strftime("%Y-%m-%d"), (now + pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
            for s in SYMBOLS:
                E._EVENTS[s] = E.events_from(self.items, s)
            self.t = time.time()
        except Exception as e:  # noqa: BLE001
            log(f"FTMO calendar not read ({e}); trying again in 10 minutes")
            self.t = time.time() - 3000

    def upcoming(self, n=8):
        now = pd.Timestamp.now(tz="UTC")
        out = []
        for i in self.items:
            t = pd.Timestamp(i["date"]).tz_convert("UTC")
            if t >= now - pd.Timedelta(hours=2) and (i.get("impact") == "high" or i.get("restriction")):
                out.append({"t": t.isoformat(), "title": i.get("title", ""), "instrument": i.get("instrument", ""),
                            "restricted": bool(i.get("restriction"))})
        return sorted(out, key=lambda x: x["t"])[:n]


def spreads():
    """FTMO's live spread per instrument, as bot 1 samples it (median of its last samples), from bot 1's state."""
    try:
        s = json.loads((LIVE / "bot1_state.json").read_text(encoding="utf-8")).get("spreads") or {}
        return {k: float(v["median"] if isinstance(v, dict) else np.median(v)) for k, v in s.items() if v}
    except Exception:  # noqa: BLE001
        return {}


def trades_of(name, d5, spread_now):
    """Every trade of the three strategies on this instrument in the candles held, net of FTMO costs (r in R)."""
    out = []
    sp = pd.Series([spread_now], index=[d5.index[0]]) if spread_now else None
    pl = plans()
    for sid in STRATS:
        fn = filtered(E.EVENTS[sid][1], (pl.get(f"{sid}|5Min|{name}") or {}).get("filter"))
        px = float(d5["c"].iloc[-1])
        spec = U.spec(name)
        comm = (px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
                if spec["commissionType"] == "flat_USD" else px * 2 * float(spec["commission"]) / 100.0)
        be = (0.6, ((spread_now or 0.0) + comm) / px)
        t = R.run(d5, fn, 0.0, breakeven=be)
        if t is None or not len(t):
            continue
        net = C.apply(t, name, sp)
        for k in range(len(t)):
            x, n_ = t.iloc[k], net.iloc[k]
            open_ = x["reason"] == END
            risk = abs(float(x["entry_px"]) - float(x["stop_px"]))
            cost_r = float(n_["spread_px"] + n_["comm_px"] - n_["swap_px"]) / risk if risk > 0 else 0.0
            r_gross = float(x["r"])
            if open_:                                           # still open: r at the last close, costs included
                r_gross = float(x["dir"]) * (px - float(x["entry_px"])) / risk if risk > 0 else 0.0
            out.append({"id": f"{sid}|{name}|{pd.Timestamp(x['time']).strftime('%Y%m%d%H%M')}", "sid": sid, "symbol": name,
                        "time": pd.Timestamp(x["time"]).isoformat(), "dir": int(x["dir"]), "entry": float(x["entry_px"]),
                        "stop": float(x["stop_px"]), "target": None if x["target_px"] != x["target_px"] else float(x["target_px"]),
                        "status": "open" if open_ else "closed", "exit_time": None if open_ else pd.Timestamp(x["exit_time"]).isoformat(),
                        "exit": None if open_ else float(x["exit_px"]), "r": round(r_gross - cost_r, 4),
                        "reason": "" if open_ else str(x["reason"])})
    return out


class Account:
    def __init__(self, a):
        if not a or a.get("v") != ACCT_V:
            a = {"v": ACCT_V, "initial": INITIAL, "started": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                 "ledger": {}, "eq_curve": [], "dead": False}
        self.a = a

    def balance(self):
        return self.a["initial"] + sum(x["pnl"] for x in self.a["ledger"].values() if x.get("pnl") is not None)

    def day_start(self, day):
        return self.a["initial"] + sum(x["pnl"] for x in self.a["ledger"].values() if x.get("pnl") is not None and x["closed_day"] < day)


def israel_day(ts):
    return pd.Timestamp(ts).tz_convert("Europe/Prague").strftime("%Y-%m-%d")      # FTMO's day (midnight Prague)


def book(acc, t, notify):
    """Puts a trade in the virtual account the first time it is seen (from the account's start), and books its
    result when it closes. Returns True when something changed."""
    if t["time"] < acc.a["started"]:
        return False
    led = acc.a["ledger"]
    side = "קנייה" if t["dir"] > 0 else "מכירה"
    nm = f"{NAMES[t['sid']]} ({t['sid']}) · {t['symbol']}"
    if t["id"] not in led:
        bal = acc.balance()
        day = israel_day(pd.Timestamp.now(tz="UTC"))
        stop_day = acc.day_start(day) * (1 - DAILY_STOP)
        if acc.a.get("dead") or bal <= INITIAL * (1 - MAX_LOSS) or bal <= stop_day:
            led[t["id"]] = {"skipped": "loss limit", "pnl": None, "closed_day": day}
            return True
        led[t["id"]] = {"risk_usd": round(bal * RISK, 2), "pnl": None, "closed_day": "9999"}
        if notify:
            REP.notify(f"בוט 7: נפתחה {side}", f"{nm}\nכניסה {t['entry']:.5g} · סטופ {t['stop']:.5g}"
                       + (f" · יעד {t['target']:.5g}" if t["target"] else "") + f" · סיכון ${led[t['id']]['risk_usd']:,.0f} (וירטואלי)",
                       tags=["newspaper"])
    e = led[t["id"]]
    if t["status"] == "closed" and e.get("pnl") is None and "skipped" not in e:
        e["pnl"] = round(e["risk_usd"] * t["r"], 2)
        e["r"] = t["r"]
        e["closed_day"] = israel_day(t["exit_time"])
        if notify:
            REP.notify(f"בוט 7: נסגרה {side} {'+' if e['pnl'] >= 0 else '-'}${abs(e['pnl']):,.0f}",
                       f"{nm}\n{t['r']:+.2f} סיכון · יתרה ${acc.balance():,.0f} (וירטואלי)", tags=["newspaper"])
        if acc.balance() <= INITIAL * (1 - MAX_LOSS) and not acc.a.get("dead"):
            acc.a["dead"] = True
            REP.notify("בוט 7: החשבון הווירטואלי נשרף", f"היתרה ירדה ל־${acc.balance():,.0f} (10% הפסד). לא נפתחות עוד עסקאות.", priority=4)
        return True
    return t["id"] not in led


def state_of(acc, trades, cal, src):
    now = pd.Timestamp.now(tz="UTC")
    led = acc.a["ledger"]
    mine = [t for t in trades.values() if t["id"] in led and "skipped" not in led[t["id"]]]
    closed = sorted([t for t in mine if t["status"] == "closed"], key=lambda t: t["exit_time"])[-KEEP:]
    opened = [t for t in mine if t["status"] == "open"]
    bal = acc.balance()
    pnl_now = lambda t: round(led[t["id"]]["risk_usd"] * t["r"], 2)  # noqa: E731
    eq = bal + sum(pnl_now(t) for t in opened)
    ni = now.isoformat(timespec="seconds")
    ec = acc.a["eq_curve"]
    if not ec or ec[-1][1] != round(bal, 2) or ec[-1][2] != round(eq, 2) or ni[:15] != ec[-1][0][:15]:
        ec.append([ni, round(bal, 2), round(eq, 2)])
        del ec[:-4000]
    combo = lambda t: f"{t['sid']}|5Min|{t['symbol']}"  # noqa: E731
    side = lambda t: "buy" if t["dir"] > 0 else "sell"  # noqa: E731
    by = {s: [led[t["id"]]["r"] for t in closed if t["sid"] == s] for s in STRATS}
    rules = ("בוט 7 · פרסומים כלכליים · חשבון וירטואלי, בלי כסף · שלוש אסטרטגיות על כל פרסום בהשפעה גבוהה בלוח של FTMO: "
             "EV01 הליכה עם 10 הדקות הראשונות (תנועה מעל 1.5 ATR, יעד 2R, עד שעה) · "
             "EV02 נגד קפיצה של מעל 2.5 ATR שכבר מחזירה שליש (יעד: המחיר שלפני הפרסום, עד שעה וחצי) · "
             "EV03 פריצת הטווח של 15 הדקות הראשונות (יעד 1.5R, עד שעתיים) · נרות 5 דקות · כניסה 10 דקות ומעלה אחרי הפרסום · "
             f"סטופ לנקודת הכניסה ב־60% מהדרך ליעד · סיכון {RISK * 100:g}% מהיתרה לעסקה · עלויות FTMO אמיתיות · "
             f"עצירה יומית {DAILY_STOP * 100:g}%, הפסד מקסימלי {MAX_LOSS * 100:g}% · "
             f"{len(SYMBOLS)} נכסים: {', '.join(SYMBOLS)}")
    return {"bot": NAME, "label": LABEL, "watch_only": True, "rules": rules, "account": "וירטואלי · פרסומים כלכליים",
            "server": "צפייה בלבד · בלי כסף", "initial": acc.a["initial"], "risk": RISK, "balance": round(bal, 2),
            "equity": round(eq, 2), "day_start": round(acc.day_start(israel_day(now)), 2), "halted_until": None,
            "started": acc.a["started"], "withdrawals": [], "combos": [f"{s}|5Min|{n}" for s in STRATS for n in SYMBOLS],
            "equity_curve": ec, "lows": {}, "updated_utc": ni,
            "summary": {s: {"closed": len(v), "sum_r": round(float(np.sum(v)), 2) if v else 0.0,
                            "win_pct": round(100 * float(np.mean(np.array(v) > 0)), 1) if v else None} for s, v in by.items()},
            "events": cal.upcoming(), "instruments": {n: src.broker.get(n) for n in SYMBOLS},
            "positions": [{"ticket": t["id"], "side": side(t), "dir": "קנייה" if t["dir"] > 0 else "מכירה", "combo": combo(t),
                           "symbol": t["symbol"], "lots": "וירטואלי", "open": t["entry"],
                           "now": t["entry"] + t["dir"] * (t["r"]) * abs(t["entry"] - t["stop"]),
                           "sl": t["stop"], "tp": t["target"] or 0.0, "sl0": t["stop"], "tp0": t["target"] or 0.0,
                           "sl_usd": -led[t["id"]]["risk_usd"], "tp_usd": None, "pnl": pnl_now(t), "since": t["time"],
                           "pip": 1.0, "free": False, "mae": None, "mfe": None} for t in opened],
            "trades": [{"time": t["exit_time"], "combo": combo(t), "symbol": t["symbol"], "dir": "קנייה" if t["dir"] > 0 else "מכירה",
                        "lots": "וירטואלי", "price": t["exit"], "pnl": led[t["id"]]["pnl"], "why": t["reason"]} for t in closed],
            "closed": [{"id": t["id"], "combo": combo(t), "symbol": t["symbol"], "side": side(t), "lots": "וירטואלי",
                        "open_t": t["time"], "open_px": t["entry"], "close_t": t["exit_time"], "close_px": t["exit"],
                        "sl0": t["stop"], "tp0": t["target"] or 0, "sl": t["stop"], "tp": t["target"] or 0, "mae": None, "mfe": None,
                        "gross": led[t["id"]]["pnl"], "swap": 0, "comm": 0, "net": led[t["id"]]["pnl"], "why": t["reason"],
                        "reason": "", "pip": 1.0} for t in closed],
            "vacct": {k: v for k, v in acc.a.items() if k != "eq_curve"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    port = os.environ.get("SOURCE_PORT")
    if not port:
        sys.exit("SOURCE_PORT not set (the manager starts this bot)")
    src, cal = Source(port), Calendar()
    try:
        saved = json.loads(STATE_F.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        saved = {}
    acc = Account(saved.get("vacct_full"))
    trades = {t["id"]: t for t in saved.get("all_trades", [])}
    REP.remember_alerts(LIVE / f"{NAME}_alerts.json")
    first, last_push = True, 0
    log(f"news bot started (account from {acc.a['started']})")
    while True:
        try:
            cal.refresh()
            sp = spreads()
            seen = {}
            for name in SYMBOLS:
                try:
                    d1 = src.update(name)
                except Exception as e:  # noqa: BLE001
                    log(f"{name}: candles not read ({e})")
                    continue
                if d1 is None or len(d1) < 600:
                    continue
                d5 = five(d1)
                d5.attrs.update(symbol=name, tf="5Min")
                for t in trades_of(name, d5, sp.get(name)):
                    seen[t["id"]] = t
            for k, t in seen.items():
                old = trades.get(k)
                if old and old["status"] == "closed":
                    continue                                    # a closed trade never changes
                trades[k] = t
                book(acc, t, notify=not first)
            # a trade still open in the account but no longer in the candles held (bot restarted long after): left open
            first = False
            keep = sorted(trades.values(), key=lambda t: t["time"])[-2000:]
            trades = {t["id"]: t for t in keep}
            state = state_of(acc, trades, cal, src)
            STATE_F.write_text(json.dumps({**state, "vacct_full": acc.a, "all_trades": keep}, ensure_ascii=False), encoding="utf-8")
            if time.time() - last_push > 120:
                err = REP.push_state(NAME, state)
                if err:
                    log(f"state not pushed: {err}")
                last_push = time.time()
        except Exception:  # noqa: BLE001
            log("step failed:\n" + traceback.format_exc())
        if args.once:
            break
        time.sleep(60 - time.time() % 60 + 5)


if __name__ == "__main__":
    main()
