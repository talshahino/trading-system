"""Live demo robot: runs one portfolio of (strategy, timeframe, instrument) combos on one MT5
terminal, with exactly the code that was backtested (agents/rules_agent.py).

How it decides (every 20 s it checks for newly closed bars):
  1. fetch the combo's bars from MT5, convert to UTC like the backtest (4H = UTC-aligned from 1H),
     append the bar that just opened with open=high=low=close (nothing from the future)
  2. replay the strategy with the backtest engine (same breakeven at 60%, stocks closed at session end)
  3. the replay's last trade tells what to do now:
       - a trade entering on the bar that just opened -> open it (same stop and target)
       - our open trade still open in the replay      -> move the stop to the replay's current stop
       - our open trade closed in the replay           -> close it (exit signal, time exit, session end)
     Stops and targets sit on the broker, so intrabar hits happen there, as in the backtest.
Account guard (same as the backtest): no new trade if today's loss + every open stop + the new
stop could pass 90% of the 5% daily limit or the 10% total limit. Emergency: floating daily loss
4.5% -> close everything until Prague midnight.
News: no entries within 5 minutes of a high-impact release (ForexFactory weekly calendar).

  python live\\runner.py --config live\\bots\\bot2.json --terminal "C:\\FTMO2\\terminal64.exe" [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agents import ftmo_universe as U  # noqa: E402
from agents import rules_agent as R  # noqa: E402

import MetaTrader5 as mt5  # noqa: E402  (pip install MetaTrader5, Windows only)

SYM_BY_MAGIC = {}
TF_MIN = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "4Hour": 240}
MT5_TF = {"1Min": mt5.TIMEFRAME_M1, "5Min": mt5.TIMEFRAME_M5, "15Min": mt5.TIMEFRAME_M15,
          "1Hour": mt5.TIMEFRAME_H1, "4Hour": mt5.TIMEFRAME_H1}
BARS = {"1Min": 3000, "5Min": 3000, "15Min": 3000, "1Hour": 3000, "4Hour": 8000}
SESSION = {"US": ("America/New_York", 16 * 60), "EU": ("Europe/Berlin", 17 * 60 + 30)}
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def mt5_name(sym):
    return U.ALL[sym][0].replace("/", "")


# ---------------- time ----------------
def server_offset_hours(probe):
    tick = mt5.symbol_info_tick(probe)
    if tick is None:
        return 0
    return int(round((tick.time - time.time()) / 3600.0))


def bars(sym, tf, off):
    rates = mt5.copy_rates_from_pos(mt5_name(sym), MT5_TF[tf], 0, BARS[tf])
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"] - off * 3600, unit="s", utc=True)
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "tick_volume": "v"})[["o", "h", "l", "c", "v"]]
    if tf == "4Hour":
        df = df.resample("4h", label="left", closed="left").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    return df


def session_mask(df, sym, tf):
    if U.CLASS.get(sym) != "stock":
        return None
    tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[sym] else SESSION["EU"]
    loc = df.index.tz_convert(tz)
    end = loc + pd.Timedelta(minutes=TF_MIN[tf])
    mins = end.hour * 60 + end.minute + (end.normalize() > loc.normalize()) * 1440
    day = pd.Series(loc.normalize(), index=df.index)
    return (mins >= close_min - 15) | (day != day.shift(-1)).to_numpy()


def prague_midnight_utc():
    now = pd.Timestamp.now(tz="Europe/Prague")
    return now.normalize().tz_convert("UTC")


# ---------------- news ----------------
_NEWS = {"t": 0, "ev": []}


def news_near(sym, minutes=5):
    if time.time() - _NEWS["t"] > 3 * 3600:
        try:
            import requests
            js = requests.get(NEWS_URL, timeout=20).json()
            _NEWS["ev"] = [(pd.Timestamp(e["date"]).tz_convert("UTC"), e["country"]) for e in js
                           if str(e.get("impact", "")).lower() == "high"]
            _NEWS["t"] = time.time()
        except Exception as e:  # noqa: BLE001
            log(f"news calendar not available ({e}); entries are not blocked")
            _NEWS["t"] = time.time()
    cur = set(U.CURRENCIES.get(sym, []))
    now = pd.Timestamp.now(tz="UTC")
    return any(c in cur and abs((t - now).total_seconds()) <= minutes * 60 for t, c in _NEWS["ev"])


# ---------------- account ----------------
class Account:
    def __init__(self, cfg, dry):
        self.cfg, self.dry = cfg, dry
        self.initial = cfg["initial"]
        self.risk_usd = self.initial * cfg["risk"]
        self.free = cfg["free_risk_at_breakeven"]
        self.halt_until = None

    def positions(self):
        return [p for p in (mt5.positions_get() or []) if p.magic in MAGICS]

    def day_start_balance(self):
        since = prague_midnight_utc().to_pydatetime()
        deals = mt5.history_deals_get(since, dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)) or []
        realized = sum(d.profit + d.commission + d.swap for d in deals if d.entry in (1, 3))
        return mt5.account_info().balance - realized

    def open_risk(self):
        tot = 0.0
        for p in self.positions():
            if self.free and p.sl and ((p.type == 0 and p.sl >= p.price_open) or (p.type == 1 and p.sl <= p.price_open)):
                continue
            tot += STATE["risk"].get(str(p.ticket), self.risk_usd)
        return tot

    def allows(self):
        ai = mt5.account_info()
        ds = self.day_start_balance()
        floor_total = self.initial * (1 - 0.10)
        new = self.open_risk() + self.risk_usd
        return (ai.balance - new > floor_total) and (ai.balance - new >= ds - self.initial * 0.05 * 0.9)

    def guard(self):
        """Emergency stop: floating daily loss reaches 4.5% -> close everything until Prague midnight."""
        if self.halt_until and pd.Timestamp.now(tz="UTC") < self.halt_until:
            return True
        ai = mt5.account_info()
        if ai.equity <= self.day_start_balance() - self.initial * 0.045 or ai.equity <= self.initial * 0.91:
            log(f"EMERGENCY equity {ai.equity:,.0f}: closing everything until Prague midnight")
            for p in self.positions():
                close(p, "emergency")
            self.halt_until = prague_midnight_utc() + pd.Timedelta(days=1)
            return True
        return False


# ---------------- orders ----------------
def lots_for(sym, entry, sl, risk_usd):
    info = mt5.symbol_info(mt5_name(sym))
    if info is None or info.trade_tick_size == 0:
        return 0.0
    per_lot = abs(entry - sl) / info.trade_tick_size * info.trade_tick_value
    if per_lot <= 0:
        return 0.0
    lots = risk_usd / per_lot
    step = info.volume_step or 0.01
    lots = np.floor(lots / step + 1e-9) * step
    digits = max(0, len(f"{step:.8f}".rstrip("0").split(".")[1]))
    lots = round(float(min(max(lots, 0.0), info.volume_max)), digits)
    return lots if lots >= info.volume_min else 0.0


def send(req, what):
    if ACC.dry:
        log(f"[dry-run] {what}: {req}")
        return None
    r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"FAILED {what}: {None if r is None else (r.retcode, r.comment)}")
        return None
    log(f"OK {what}: {r.order}")
    return r


def open_trade(combo, sym, d, sl, tp, entry_time):
    name = mt5_name(sym)
    tick = mt5.symbol_info_tick(name)
    px = tick.ask if d > 0 else tick.bid
    if d * (px - sl) <= 0 or (tp == tp and tp is not None and d * (tp - px) <= 0):
        log(f"skip {combo}: price {px} already beyond stop/target")
        return
    lots = lots_for(sym, px, sl, ACC.risk_usd)
    if lots <= 0:
        log(f"skip {combo}: lot size too small")
        return
    info = mt5.symbol_info(name)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": name, "volume": lots,
           "type": mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL, "price": px,
           "sl": round(sl, info.digits), "tp": round(tp, info.digits) if tp == tp and tp is not None else 0.0,
           "deviation": 20, "magic": MAGICS_BY_COMBO[combo], "comment": combo[:31],
           "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    r = send(req, f"open {combo} {'buy' if d > 0 else 'sell'} {lots} @ {px} sl {sl:.5g} tp {tp}")
    if r is not None:
        STATE["entry"][combo] = str(entry_time)
        STATE["risk"][str(r.order)] = lots * abs(px - sl) / info.trade_tick_size * info.trade_tick_value
        save_state()


def close(p, why):
    tick = mt5.symbol_info_tick(p.symbol)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "position": p.ticket,
           "type": mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY,
           "price": tick.bid if p.type == 0 else tick.ask, "deviation": 20, "magic": p.magic,
           "comment": why[:31], "type_filling": mt5.ORDER_FILLING_IOC}
    send(req, f"close {p.comment} ({why})")


def move_stop(p, new_sl):
    info = mt5.symbol_info(p.symbol)
    new_sl = round(new_sl, info.digits)
    better = (p.type == 0 and new_sl > p.sl + info.point) or (p.type == 1 and (p.sl == 0 or new_sl < p.sl - info.point))
    tick = mt5.symbol_info_tick(p.symbol)
    if better and ((p.type == 0 and new_sl >= tick.bid) or (p.type == 1 and new_sl <= tick.ask)):
        close(p, "stop passed")                        # price already beyond the new stop
        return
    if better:
        send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": p.symbol, "sl": new_sl, "tp": p.tp},
             f"stop {p.comment} -> {new_sl}")


# ---------------- strategy step ----------------
def be_offset(sym, px):
    spec = U.spec(sym)
    tick = mt5.symbol_info_tick(mt5_name(sym))
    spread = (tick.ask - tick.bid) if tick else 0.0
    if spec["commissionType"] == "flat_USD":
        comm = px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
    else:
        comm = px * 2 * float(spec["commission"]) / 100.0
    return (spread + comm) / px


def stock_session_close(now):
    """Stocks: close every position 15 minutes before the exchange closes (no overnight gaps)."""
    for p in ACC.positions():
        sym = SYM_BY_MAGIC.get(p.magic)
        if sym is None or U.CLASS.get(sym) != "stock":
            continue
        tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[sym] else SESSION["EU"]
        loc = now.tz_convert(tz)
        if close_min - 15 <= loc.hour * 60 + loc.minute < close_min + 60:
            close(p, "session end")


def step(combo, sid, tf, sym, off):
    df = bars(sym, tf, off)
    if df is None:
        return
    # The newest MT5 bar is still forming: keep only its open (what the backtest knows at entry),
    # and add one flat bar after it, because the engine enters on the bar AFTER a signal.
    forming = df.index[-1]
    last_closed = df.index[-2]
    if pd.Timestamp.now(tz="UTC") - forming >= pd.Timedelta(minutes=TF_MIN[tf]):
        return          # the newest bar is not in progress: market closed (weekend/holiday/session break)
    df.iloc[-1, df.columns.get_indexer(["h", "l", "c"])] = df["o"].iloc[-1]
    dummy = df.iloc[[-1]].copy()
    dummy.index = [forming + pd.Timedelta(minutes=TF_MIN[tf])]
    df = pd.concat([df, dummy])
    be = (0.6, be_offset(sym, float(df["c"].iloc[-3])))
    tr = R.run(df, R.STRATEGIES[sid][1], 0.0, breakeven=be, session_exit=session_mask(df, sym, tf))
    pos = [p for p in ACC.positions() if p.magic == MAGICS_BY_COMBO[combo]]

    def stop_now():
        """The stop as the backtest has it after the last CLOSED bar (the helper bars must not move it)."""
        tc = R.run(df.iloc[:-2], R.STRATEGIES[sid][1], 0.0, breakeven=be, session_exit=session_mask(df.iloc[:-2], sym, tf))
        m = tc[tc["time"].astype(str) == STATE["entry"].get(combo, "")] if len(tc) else tc
        return float(m.iloc[-1]["stop_last"]) if len(m) and m.iloc[-1]["reason"] == "סוף הנתונים" else None

    if pos:
        p = pos[0]
        mine = tr[tr["time"].astype(str) == STATE["entry"].get(combo, "")] if len(tr) else tr
        if mine.empty:
            return                                     # not in the replay (data edge): broker SL/TP manage it
        t = mine.iloc[-1]
        reason = str(t["reason"])
        if reason == "יעד" or "סטופ" in reason:
            if t["exit_time"] > forming and (sn := stop_now()) is not None:   # only the helper bars "hit" it
                move_stop(p, sn)
            elif t["exit_time"] == forming and "פער פתיחה" in reason:
                close(p, "stop (gap)")                 # the new stop is already past the price: out at market
            return                                     # real stop/target: the broker closes it at that price
        if reason == "סוף הנתונים" or not (t["exit_time"] <= last_closed or (
                t["exit_time"] == forming and reason in ("סיגנל יציאה", "סיגנל הפוך"))):
            if (sn := stop_now()) is not None:
                move_stop(p, sn)
            return
        close(p, reason)                               # exit signal / opposite signal / time / session end
        # an opposite signal may open the next trade on this same bar: fall through
    if not len(tr):
        return
    t = tr[tr["time"] == forming]
    if t.empty:
        return
    t = t.iloc[-1]
    if STATE["entry"].get(combo) == str(t["time"]):
        return                                         # already handled this entry
    if U.CLASS.get(sym) == "stock":
        tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[sym] else SESSION["EU"]
        loc = pd.Timestamp.now(tz="UTC").tz_convert(tz)
        if loc.hour * 60 + loc.minute >= close_min - 15:
            return                                     # no new stock trades in the last 15 minutes
    if news_near(sym):
        log(f"skip {combo}: high-impact news within 5 minutes")
        return
    if not ACC.allows():
        log(f"skip {combo}: daily/total loss guard")
        return
    open_trade(combo, sym, int(t["dir"]), float(t["stop_px"]), float(t["target_px"]), t["time"])


# ---------------- main ----------------
def save_state():
    STATE_FILE.write_text(json.dumps(STATE, indent=1), encoding="utf-8")


def main():
    global ACC, STATE, STATE_FILE, LOG_FILE, MAGICS, MAGICS_BY_COMBO
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--terminal", required=True, help="path of terminal64.exe of the MT5 logged into this bot's demo account")
    ap.add_argument("--dry-run", action="store_true", help="log the orders instead of sending them")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    name = Path(args.config).stem
    LOG_FILE = ROOT / "live" / f"{name}.log"
    STATE_FILE = ROOT / "live" / f"{name}_state.json"
    STATE = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {"entry": {}, "risk": {}}
    if not mt5.initialize(path=args.terminal, portable=True):
        sys.exit(f"MT5 did not start: {mt5.last_error()}  (open the terminal and log in first)")
    ai = mt5.account_info()
    log(f"{name}: account {ai.login} {ai.server} balance {ai.balance:,.0f} | {len(cfg['combos'])} combos, "
        f"risk {cfg['risk'] * 100:g}% | {'DRY RUN' if args.dry_run else 'LIVE ON THIS ACCOUNT'}")
    if "demo" not in ai.server.lower() and "trial" not in ai.server.lower() and not args.dry_run:
        sys.exit(f"server {ai.server} does not look like a demo: refusing to trade")
    combos = []
    for c in cfg["combos"]:
        sid, tf, sym = c.split("|")
        if not mt5.symbol_select(mt5_name(sym), True):
            log(f"WARNING: symbol {mt5_name(sym)} not available on this account - {c} is off")
            continue
        combos.append((c, sid, tf, sym))
    MAGICS_BY_COMBO = {c: cfg["magic_base"] + i for i, c in enumerate(cfg["combos"])}
    MAGICS = set(MAGICS_BY_COMBO.values())
    SYM_BY_MAGIC.update({MAGICS_BY_COMBO[c]: c.split("|")[2] for c in cfg["combos"]})
    ACC = Account(cfg, args.dry_run)
    done = {}
    probe = mt5_name(combos[0][3])
    while True:
        try:
            if not ACC.guard():
                off = server_offset_hours(probe)
                now = pd.Timestamp.now(tz="UTC")
                stock_session_close(now)
                for combo, sid, tf, sym in combos:
                    bar_start = now.floor(f"{TF_MIN[tf]}min")
                    if done.get(combo) == bar_start or (now - bar_start).total_seconds() < 3:
                        continue
                    try:
                        step(combo, sid, tf, sym, off)
                    except Exception:  # noqa: BLE001
                        log(f"error {combo}:\n{traceback.format_exc()}")
                    done[combo] = bar_start
        except Exception:  # noqa: BLE001
            log(f"loop error:\n{traceback.format_exc()}")
            if not mt5.initialize(path=args.terminal, portable=True):
                log("MT5 connection lost; retrying in 60 s")
                time.sleep(60)
        time.sleep(20)


if __name__ == "__main__":
    main()
