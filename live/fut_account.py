"""Bot 6 (user, 10.11.2026): a virtual futures account - the scalping bot's gap methods (v5 on 15 minutes and the
1-minute track) on CME micro index futures, under FTMO Futures rules. No money, no orders: the watch bot feeds it.

Prices: the watch bot reads FTMO's index CFDs (US100 / US500 / US30) from bot 1's MetaTrader; they follow the
futures (MNQ / MES / MYM) closely, so the trades are found on them and sized and costed as micro contracts.
The levels shown are the CFD's, not the futures' own quotes.

FTMO Futures rules used (ftmo.com, futures trading objectives, read 10.11.2026), plan PLAN:
  * trailing maximum loss, end of day: the highest end-of-day balance minus MAX_LOSS; it stops rising at the
    initial balance. Equity must never touch it.
  * daily loss limit (Pro, hard): the day's starting balance minus DAILY_LOSS.
  * trading day 18:00 -> 16:10 New York; everything closed by 16:10, nothing over the weekend.
  * contracts: at most MAX_MICROS micros open (10 minis = 100 micros on the $100K account).
  * the evaluation: +TARGET with the best day at most CONSISTENCY of the total profit.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

# CFD name -> (micro contract, $ per point, tick)
SPEC = {"US100": ("MNQ", 2.0, 0.25), "US500": ("MES", 5.0, 0.25), "US30": ("MYM", 0.5, 1.0)}
COMM = 1.50          # $ per micro, round turn (assumed: typical prop-firm cost; FTMO does not publish it on the rules page)
SLIP_TICKS = 1.0     # one tick lost per trade on average (stop exits, the session close)
PLAN = "Pro $100K"
INITIAL, TARGET, MAX_LOSS, DAILY_LOSS, MAX_MICROS, CONSISTENCY = 100_000.0, 6_000.0, 4_500.0, 1_500.0, 100, 0.5
RISK_V5, RISK_V1M = 0.004, 0.0             # share of the balance risked per trade (set from the simulation)
# the user's gap method, 1 hour / 5 minutes (card abb187377383, approved 12.11.2026): the only track since then
RISK_TAL = 0.004
# the method's tracks (each approved by the manager on its own backtest, 12.11.2026):
#   tal    1 hour / 5 minutes   card abb187377383  2026 unseen +0.18R / 254 trades
#   tal4h  4 hours / 1 hour     card 6694a91a9563  2026 unseen +0.07R / 35 trades (user: "maybe there are more pairs")
TAL_TRACKS = {"tal": (("1h", "5m", 2), {"GBODY": 1.5, "MIN_GAP": 0.3, "SW": 3, "MIN_RR": 1.0, "EXPIRY": 120, "PROT": "none"}),
              "tal4h": (("4h", "1h", 1), {"GBODY": 1.5, "MIN_GAP": 0.6, "SW": 3, "MIN_RR": 1.0, "EXPIRY": 48, "PROT": "none"})}
TAL_PAIR, TAL_CFG = TAL_TRACKS["tal"]
TAL_NAME = {"tal": "שעה / 5 דק'", "tal4h": "4 שעות / שעה"}
TAL_COMBO = {"tal": "TAL|1Hour", "tal4h": "TAL|4Hour"}
RISK = {"v5": 0.0, "v1m": 0.0, "tal": RISK_TAL, "tal4h": RISK_TAL}   # the old gap tracks are replaced (12.11.2026)
# the 1-minute track is stopped (card 9a0bc4200f6d, 12.11.2026): it loses in bot 6's world (backtest -0.08R, 2026 -0.04R)
ROOM_USE = 0.9                              # open risk may use at most 90% of the room left to either limit
VERSION = 3                 # 2 (10.11.2026): one name per gap; 3 (12.11.2026): the user's gap method - a fresh $100K account


def session_masks(df):
    """(shut, late) per 1-minute candle: shut = the futures are closed (16:05-18:00 New York, the weekend) - every
    trade is closed then; late = no new entry (from 15:45, and while shut)."""
    et = df.index.tz_convert("America/New_York")
    mins = np.asarray(et.hour * 60 + et.minute)
    wd = np.asarray(et.weekday)
    shut = ((mins >= 16 * 60 + 5) & (mins < 18 * 60)) | (wd == 5) | ((wd == 6) & (mins < 18 * 60)) | ((wd == 4) & (mins >= 16 * 60 + 5))
    late = ((mins >= 15 * 60 + 45) & (mins < 18 * 60)) | shut
    return shut, late


def tday(ts):
    """The futures trading day (18:00 New York starts the next one)."""
    return (pd.Timestamp(ts).tz_convert("America/New_York") + pd.Timedelta(hours=6)).strftime("%Y-%m-%d")


def per_contract(symbol, entry, stop):
    _, pv, tick = SPEC[symbol]
    return abs(entry - stop) * pv + COMM + SLIP_TICKS * tick * pv


def pnl_of(symbol, n, entry, stop, r_gross):
    _, pv, tick = SPEC[symbol]
    return n * (pv * abs(entry - stop) * r_gross - COMM - SLIP_TICKS * tick * pv)


class Account:
    def __init__(self, acc: dict | None):
        acc = acc or {}
        if acc.get("v") != VERSION:
            acc = {"v": VERSION, "plan": PLAN, "initial": INITIAL, "ledger": [], "breached": None, "passed": None,
                   "started": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
        self.a = acc

    # ---- balances
    def balance(self):
        return self.a["initial"] + sum(x["pnl"] for x in self.a["ledger"])

    def day_pnl(self, day):
        return sum(x["pnl"] for x in self.a["ledger"] if x["day"] == day)

    def day_start(self, day):
        return self.a["initial"] + sum(x["pnl"] for x in self.a["ledger"] if x["day"] < day)

    def level(self, day):
        """The trailing maximum-loss level on trading day `day`: the best end-of-day balance before it, minus MAX_LOSS,
        never above the initial balance."""
        bal, peak = self.a["initial"], self.a["initial"]
        for d in sorted({x["day"] for x in self.a["ledger"] if x["day"] < day}):
            bal = self.a["initial"] + sum(x["pnl"] for x in self.a["ledger"] if x["day"] <= d)
            peak = max(peak, bal)
        return min(peak - MAX_LOSS, self.a["initial"])

    # ---- a new trade: how many contracts, or why none
    def size(self, symbol, trk, entry, stop, t, open_recs):
        if self.a.get("breached"):
            return 0, "breach"
        if RISK.get(trk, 0.0) <= 0:
            return 0, "off"                     # the track is stopped
        day = tday(t)
        bal = self.balance()
        room = min(bal - self.level(day), bal - (self.day_start(day) - DAILY_LOSS))
        room = ROOM_USE * room - sum(r.get("risk_usd") or 0 for r in open_recs)
        per = per_contract(symbol, entry, stop)
        want = bal * RISK.get(trk, 0.0)
        n = int(min(want, room) // per)
        n = min(n, MAX_MICROS - sum(r.get("n") or 0 for r in open_recs))
        if n < 1:
            return 0, "room" if room < want else "small"
        return n, None

    def close(self, rec):
        """A closed trade enters the ledger once; checks the limits and the evaluation."""
        if any(x["id"] == rec["id"] for x in self.a["ledger"]):
            return []
        day = tday(rec["exit_time"])
        self.a["ledger"].append({"id": rec["id"], "exit_time": rec["exit_time"], "day": day, "pnl": rec["pnl"]})
        events = []
        bal = self.balance()
        if not self.a.get("breached"):
            if bal <= self.level(day):
                self.a["breached"] = {"t": rec["exit_time"], "why": "max"}
                events.append("breach_max")
            elif bal <= self.day_start(day) - DAILY_LOSS:
                self.a["breached"] = {"t": rec["exit_time"], "why": "daily"}
                events.append("breach_daily")
        profit = bal - self.a["initial"]
        if not self.a.get("passed") and not self.a.get("breached") and profit >= TARGET:
            days = {}
            for x in self.a["ledger"]:
                days[x["day"]] = days.get(x["day"], 0) + x["pnl"]
            if max(days.values()) <= CONSISTENCY * profit:
                self.a["passed"] = rec["exit_time"]
                events.append("passed")
        return events

    def status(self, now):
        day = tday(now)
        bal = self.balance()
        lvl = self.level(day)
        days = {}
        for x in self.a["ledger"]:
            days[x["day"]] = days.get(x["day"], 0) + x["pnl"]
        profit = bal - self.a["initial"]
        return {"plan": PLAN, "balance": round(bal, 2), "profit": round(profit, 2), "target": TARGET,
                "max_loss_level": round(lvl, 2), "room_max": round(bal - lvl, 2),
                "day": day, "day_start": round(self.day_start(day), 2), "today": round(days.get(day, 0.0), 2),
                "daily_level": round(self.day_start(day) - DAILY_LOSS, 2),
                "best_day": round(max(days.values()), 2) if days else 0.0,
                "consistency_ok": bool(profit > 0 and days and max(days.values()) <= CONSISTENCY * profit),
                "breached": self.a.get("breached"), "passed": self.a.get("passed"),
                "risk_v5_pct": RISK["v5"] * 100, "risk_v1m_pct": RISK["v1m"] * 100, "risk_tal_pct": RISK_TAL * 100, "max_micros": MAX_MICROS}
