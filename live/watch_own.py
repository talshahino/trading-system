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
from agents import own_gaps as OG  # noqa: E402
from agents import tal_gap as TG  # noqa: E402
from live import fut_account as FA  # noqa: E402
from live import reporter as REP  # noqa: E402

NAME = "watch_own"
LABEL = "בוט סקלפינג · צפייה בלבד"
LIVE = ROOT / "live"
STATE_F = LIVE / f"{NAME}_state.json"
MARKS_F = LIVE / "watch_marks.json"
FUT_NAME = "watch_fut"          # bot 6: the virtual futures account (live/fut_account.py)
FUT_F = LIVE / f"{FUT_NAME}_state.json"
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
# data-collection instruments (user, 09.11.2026: "more trades"): all six lost in the backtest (2024-2026, both tracks),
# so they are followed and reported like the others, marked 'extra', but kept OUT of the virtual account and the summary
EXTRA = {"WTI": ["USOIL.cash", "USOIL", "WTI"], "XAGUSD": ["XAGUSD", "XAG/USD"], "USDCAD": ["USDCAD"], "EURGBP": ["EURGBP"],
         "EU50": ["EU50.cash", "EU50"], "AUS200": ["AUS200.cash", "AUS200"]}
SYMBOLS.update(EXTRA)
# v4 (user, 09.11.2026, after marking charts and the backtests): the impulse is the big move (a move smaller than the
# opposite move before it is a correction), the first gap that stayed open during the move, stop beyond the 3rd swing
# back, the whole position out at 80% of the way to the target. Best of the grid on these 12 low-cost instruments
# (+0.11R a trade, positive in both halves, ~0.7 trades a week). Shown as its own track ('v4'), its own summary.
V4_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "EURJPY", "US100", "US30", "GER40", "US500"]
# v5 (user, 10.11.2026: "run v5"): every gap of a strong 15m candle, entry on the touch after the move turned, stop beyond
# the gap's base, target the move's peak, trades with costs above 10% of the risk skipped. The backtest's best result:
# the 4 indices, +0.10R a trade, positive in both halves, ~0.9 trades a day per index. Its own track ('v5').
V5_SYMBOLS = ["US100", "US30", "GER40", "US500"]
# the 15-minute track also on JP225 (user, 10.11.2026): +0.11R a trade in both halves, ~0.9 a day; not on 1 minute (lost in 2024)
V5_15_SYMBOLS = V5_SYMBOLS + ["JP225"]
V4 = {"BODY_X": 1.8, "DOMINANT": True, "GAP_OPEN": True, "STOP_SW": 3, "TARGET_FRAC": 0.8}
KEEP = 60                     # closed trades kept in the state (with their candles)
# virtual demo account (user, 06.11.2026): $100,000 like the other demo accounts, risk per trade as bots 2 and 4,
# the bots' own daily stop (4.5%, FTMO's limit is 5%) and FTMO's 10% maximum loss
INITIAL, RISK, DAILY_STOP, MAX_LOSS = 100_000.0, 0.0045, 0.045, 0.10
# the virtual account, v2 (user, 10.11.2026: "as if it were a real demo account"): a fresh $100,000 from the moment this
# version starts, only the two gap tracks count, each with its own risk: v5 (15m) 0.4%, the 1-minute track 0.075%.
# FTMO rules: no new trades after a 4.5% loss in the day, the account burns at a 10% loss from the start.
# v3 (10.11.2026, the review before the user's calibration): a fresh start with the corrected trade selection
ACCT_V, RISK_V5, RISK_V1M = 3, 0.004, 0.00075
# The settings bots 5 and 6 trade with live in live/bots/bot5.json and bot6.json (12.11.2026: the virtual bots are run
# as if they were real accounts - an approved change goes in at once, like bots 1-4). Read every cycle; a file that is
# missing or odd keeps the values above.
SET5 = {"v5": {"STOP": "gap", "DOMINANT": False, "V5_MAX_COST": 0.10, "PART_FRAC": 0.3, "PART_WAY": 0.9},
        "v1m": {"STOP": "gap", "DOMINANT": False, "V5_MAX_COST": 0.10, "PART_FRAC": 0.3, "PART_WAY": 0.9, "TFM": 1}}
_SET = {"m5": None, "m6": None}


def load_settings():
    global RISK_V5, RISK_V1M
    for n, key in (("bot5", "m5"), ("bot6", "m6")):
        f = ROOT / "live" / "bots" / f"{n}.json"
        try:
            m = f.stat().st_mtime
            if m == _SET[key]:
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            _SET[key] = m
        except Exception:  # noqa: BLE001
            continue
        try:
            if n == "bot5":
                for trk in ("v5", "v1m"):
                    s = (d.get("tracks") or {}).get(trk) or {}
                    SET5[trk] = {k: v for k, v in s.items() if hasattr(OG, k)}
                rk = d.get("risk") or {}
                RISK_V5 = float(rk.get("v5", RISK_V5)) if 0 <= float(rk.get("v5", RISK_V5)) <= 0.02 else RISK_V5
                RISK_V1M = float(rk.get("v1m", RISK_V1M)) if 0 <= float(rk.get("v1m", RISK_V1M)) <= 0.02 else RISK_V1M
            else:
                tal = d.get("tal") or {}
                if tal.get("pair") and tuple(tal["pair"]) in {tuple(p_) for p_ in TG.PAIRS}:
                    FA.TAL_PAIR = tuple(tal["pair"])
                if isinstance(tal.get("cfg"), dict):
                    FA.TAL_CFG = {**FA.TAL_CFG, **tal["cfg"]}
                for trk, v in (d.get("risk") or {}).items():
                    if trk in FA.RISK and 0 <= float(v) <= 0.02:
                        FA.RISK[trk] = float(v)
                FA.RISK_TAL = FA.RISK.get("tal", FA.RISK_TAL)
            print(f"settings of {n} loaded: {json.dumps(d.get('tracks') or d.get('tal'), default=str)[:300]}")
        except Exception as e:  # noqa: BLE001
            print(f"settings of {n} not loaded: {e}")


def _set_og(trk):
    """Put a track's settings on agents/own_gaps.py; returns what to put back after."""
    keep = {k: getattr(OG, k) for k in SET5[trk]}
    for k, v in SET5[trk].items():
        setattr(OG, k, v)
    return keep


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


# the real cost (spread + commission, in price) of the gap tracks' instruments: the median of the backtest, which used
# FTMO's recorded 1-minute spreads. The class estimate below was ~4x too high for the indices (GER40 5.3 vs 1.4 points),
# so the 10%-of-risk cost filter skipped almost every 1-minute trade (found 10.11.2026)
REAL_COST = {"GER40": 1.42, "US100": 1.22, "US30": 2.13, "US500": 0.51, "JP225": 6.63}


def cost_of(name, df):
    c = df["c"].to_numpy(float)
    if name in REAL_COST:
        return np.full(len(c), REAL_COST[name])
    cls = U.CLASS.get(name, "fx")
    return c * C.EST_SPREAD.get(cls, 0.0002) + CORE.comm_px(name, c)


def bars(df, t0, t1, rule):
    x = df[(df.index >= t0) & (df.index <= t1)]
    if rule != "1min":
        x = CORE.resample(x, rule)
    return [[int(t.timestamp()), round(float(r.o), 6), round(float(r.h), 6), round(float(r.l), 6), round(float(r.c), 6)]
            for t, r in x.iterrows()]


def tid(x):
    base = f"{x['symbol']}|{pd.Timestamp(x['leg_top']).strftime('%Y%m%d%H%M')}|{int(x['dir'])}" + ("|v4" if x.get("v4") else "|v5" if x.get("v5") else "|v1m" if x.get("v1m") else "")
    if x.get("v5") or x.get("v1m"):
        # the gap tracks: several gaps of one move share the move's peak, so the gap's own candle is part of the name
        # (found 10.11.2026: two trades with one name overwrote each other's risk and notifications)
        base += "|g" + pd.Timestamp(x["leg_start"]).strftime("%Y%m%d%H%M")
    return base


def gap_id_fix(trades, acc):
    """Once: the gap-track trades saved under the old name get the new one (tid), the ledger follows."""
    ren = {}
    for k, t in list(trades.items()):
        if (t.get("v5") or t.get("v1m")) and "|g" not in k and t.get("leg_start"):
            nk = k + "|g" + pd.Timestamp(t["leg_start"]).strftime("%Y%m%d%H%M")
            t["id"], t["old_id"] = nk, k
            trades[nk] = trades.pop(k)
            ren[k] = nk
    for x in acc.get("ledger") or []:
        x["id"] = ren.get(x["id"], x["id"])
    return trades


def follow(t, df, stop_at, part=0.3, way=0.9):
    """An open trade that the strategy run no longer reports (its setup changed with new candles): it is followed to
    its exit here with the same rules - stop, target, 30% out at 90% of the way and the rest to break-even, forced exit.
    Returns the fields that changed, or {} while it is still open."""
    o1, h1, l1, c1 = (df[x].to_numpy(float) for x in "ohlc")
    d, px, stop, target = t["dir"], t["entry"], t["stop"], t["target"]
    risk = d * (px - stop)
    if not t.get("time") or risk <= 0 or target is None:
        return {}
    fill = int(df.index.searchsorted(pd.Timestamp(t["time"])))
    lvl = px + way * (target - px)
    part_r, cur = 0.0, stop
    for q in range(fill, len(c1)):
        ex = rsn = None
        if (d > 0 and l1[q] <= cur) or (d < 0 and h1[q] >= cur):
            ex = (min(cur, o1[q]) if d > 0 else max(cur, o1[q])) if q > fill else cur
            rsn = "breakeven" if cur != stop else "stop"
        elif q > fill and ((d > 0 and h1[q] >= target) or (d < 0 and l1[q] <= target)):
            if cur == stop:
                part_r = part * d * (lvl - px) / risk
            ex, rsn = (max(target, o1[q]) if d > 0 else min(target, o1[q])), "target"
        elif q > fill and stop_at[q]:
            ex, rsn = c1[q], "forced"
        if ex is not None:
            rest = 1.0 - (part if part_r else 0.0)
            return {"status": "closed", "exit_time": iso(df.index[q]), "exit": float(ex), "reason": rsn,
                    "r": round(part_r + rest * d * (ex - px) / risk - (t.get("cost_r") or 0), 4), "followed": True}
        if cur == stop and q > fill and ((d > 0 and h1[q] >= lvl) or (d < 0 and l1[q] <= lvl)):
            part_r, cur = part * d * (lvl - px) / risk, px
    rest = 1.0 - (part if part_r else 0.0)
    return {"r_now": round(part_r + rest * d * (c1[-1] - px) / risk, 3), "followed": True}


def iso(t):
    return pd.Timestamp(t).isoformat() if t is not None and not (isinstance(t, float) and np.isnan(t)) else None


def fmt(name, p):
    return f"{p:.2f}" if not (len(name) == 6 and name.isalpha()) else (f"{p:.3f}" if name.endswith("JPY") else f"{p:.5f}")


FUT_WORD = {"target": "יעד", "stop": "סטופ", "forced": "סגירה כפויה", "breakeven": "ברייק (30% יצאו)"}


def fut_step(name, df, cst, stop_at, noent, fa, ftrades, fseen):
    """Bot 6 (12.11.2026): the user's gap method on 1 hour / 5 minutes (agents/tal_gap.py, card abb187377383) - an
    impulse on the hour chart, the gap between candle 1 and candle 3 that stayed open, a limit at its middle, the stop
    2 ticks beyond the 2nd 5-minute swing beyond the entry, the target at the 2nd swing on the other side, no management.
    With the futures session (no entry from 15:45 New York, everything out at 16:05, nothing over the weekend), sized
    in micro contracts on the virtual FTMO Futures account. It replaced the scalping bot's old gap tracks."""
    shut, late = FA.session_masks(df)
    tick = FA.SPEC[name][2]
    D = TG.Data(name, df[["o", "h", "l", "c"]], np.full(len(df), tick), np.asarray(stop_at | shut, bool),
                np.asarray(noent | late, bool), 2 * tick, {})
    contract = FA.SPEC[name][0]
    cutoff = (df.index[-1] - pd.Timedelta(days=2)).isoformat()
    for x in TG.live_trades(D, FA.TAL_PAIR, FA.TAL_CFG):
        if x["status"] == "cancelled" and (x.get("time") or x["setup_t"]) < cutoff:
            continue
        k = f"{name}|{x['id_key']}"
        fseen.add(k)
        old = ftrades.get(k, {})
        rec = {"id": k, "symbol": name, "contract": contract, "trk": "tal", "dir": x["dir"], "status": x["status"],
               "entry": x["entry"], "stop": x.get("stop"), "target": x.get("target"), "rr": x.get("rr"), "cost_r": 0.0,
               "setup_t": x["setup_t"], "time": x.get("time"), "exit_time": x.get("exit_time"), "exit": x.get("exit"),
               "reason": x.get("reason"), "r": x.get("r"), "r_now": x.get("r_now"), "gap": x.get("gap")}
        fut_rec(rec, old, fa, ftrades)
    o1, h1, l1, c1 = (df[x].to_numpy(float) for x in "ohlc")
    for k, t in list(ftrades.items()):        # an open trade the run no longer reports: followed to its exit here
        if t["symbol"] != name or t["status"] != "open" or k in fseen or not t.get("time") or t.get("stop") is None:
            continue
        fseen.add(k)
        d, px, st, tg = t["dir"], t["entry"], t["stop"], t["target"]
        risk = d * (px - st)
        f = int(df.index.searchsorted(pd.Timestamp(t["time"])))
        upd = {"r_now": round(d * (c1[-1] - px) / risk, 3)} if risk > 0 else {}
        for q in range(f + 1, len(c1)):
            hit_s = (l1[q] <= st) if d > 0 else (h1[q] >= st)
            hit_t = tg is not None and ((h1[q] >= tg) if d > 0 else (l1[q] <= tg))
            if hit_s or hit_t or shut[q]:
                ex, why = (st, "stop") if hit_s else (tg, "target") if hit_t else (o1[q], "forced")
                upd = {"status": "closed", "exit": float(ex), "exit_time": df.index[q].isoformat(), "reason": why,
                       "r": round(d * (ex - px) / risk, 4)}
                break
        fut_rec(dict(t, **upd), t, fa, ftrades)


def fut_rec(rec, old, fa, ftrades):
    """Bot 6: size a trade when it opens, book it when it closes, one notification each."""
    name, k, contract = rec["symbol"], rec["id"], rec["contract"]
    for f in ("n", "risk_usd", "skipped", "sent_open", "sent_close"):
        if f in old:
            rec[f] = old[f]
    if rec["status"] in ("open", "closed") and "n" not in rec and "skipped" not in rec:
        if rec["time"] < fa.a["started"]:
            rec["skipped"] = "before"
        else:
            opn = [t for t in ftrades.values() if t["status"] == "open" and t.get("n") and t["id"] != k]
            n, why = fa.size(name, rec["trk"], rec["entry"], rec["stop"], rec["time"], opn)
            if n:
                rec["n"], rec["risk_usd"] = n, round(n * FA.per_contract(name, rec["entry"], rec["stop"]), 2)
            else:
                rec["skipped"] = why
    if rec.get("n"):
        # notifications (user, 10.11.2026): only bot 6 trades that open and close - each one once, nothing else
        side = "קנייה" if rec["dir"] > 0 else "מכירה"
        tf = {"v5": "15 דק'", "v1m": "דקה", "tal": "שיטת הגאפ, שעה / 5 דק'"}.get(rec["trk"], rec["trk"])
        lots = f"{rec['n']} {contract}"
        if not rec.get("sent_open"):
            rec["sent_open"] = True
            if rec["status"] == "open":
                REP.notify(f"בוט 6: נפתחה {side} {lots}", f"{name} ({tf}) · כניסה {fmt(name, rec['entry'])} · סטופ {fmt(name, rec['stop'])}"
                           f"{' · יעד ' + fmt(name, rec['target']) if rec['target'] else ''} · סיכון ${rec['risk_usd']:,.0f}", tags=["eyes"])
        if rec["status"] == "closed" and rec.get("r") is not None:
            rec["pnl"] = round(FA.pnl_of(name, rec["n"], rec["entry"], rec["stop"], rec["r"] + (rec.get("cost_r") or 0)), 2)
            events = fa.close(rec)
            if not rec.get("sent_close"):
                rec["sent_close"] = True
                extra = {"breach_max": " · החשבון נשרף (הפסד מקסימלי)", "breach_daily": " · נגע במגבלה היומית, החשבון נשרף",
                         "passed": " · עבר את ההערכה ✓"}
                word = FUT_WORD.get(rec.get("reason"), "")
                REP.notify(f"בוט 6: נסגרה {side} {lots} {'+' if rec['pnl'] >= 0 else '-'}${abs(rec['pnl']):,.0f}",
                           f"{name} ({tf}) · {word} · יתרה ${fa.balance():,.0f}" + "".join(extra[e] for e in events), tags=["eyes"])
        elif rec["status"] == "open" and rec.get("r_now") is not None:
            rec["pnl_now"] = round(FA.pnl_of(name, rec["n"], rec["entry"], rec["stop"], rec["r_now"]), 2)
    ftrades[k] = rec


def fut_state(fa, ftrades, fseen, src):
    """live/watch_fut_state.json: bot 6 in the same shape as a real bot's state, so the app shows it like the others."""
    for k, t in list(ftrades.items()):
        if t["status"] == "pending" and k not in fseen and t["symbol"] in src.df:
            t["status"] = "cancelled"
    closed = sorted([t for t in ftrades.values() if t["status"] == "closed"], key=lambda t: t["time"] or "")[-200:]
    live_ = [t for t in ftrades.values() if t["status"] in ("pending", "open")]
    cancelled = sorted([t for t in ftrades.values() if t["status"] == "cancelled"], key=lambda t: t.get("setup_t") or "")[-20:]
    ftrades.clear()
    ftrades.update({t["id"]: t for t in closed + live_ + cancelled})
    now = pd.Timestamp.now(tz="UTC")
    st = fa.status(now)
    bal = st["balance"]
    eq = bal + sum(t.get("pnl_now") or 0 for t in live_ if t["status"] == "open" and t.get("n"))
    now_iso = now.isoformat(timespec="seconds")
    ec = fa.a.setdefault("eq_curve", [])
    if not ec or ec[-1][1] != round(bal, 2) or ec[-1][2] != round(eq, 2) or now_iso[:15] != ec[-1][0][:15]:
        ec.append([now_iso, round(bal, 2), round(eq, 2)])
        del ec[:-4000]
    counted = [t for t in closed if t.get("n") and t.get("pnl") is not None]
    combo = lambda t: (f"TAL|1Hour|{t['symbol']}" if t.get("trk") == "tal"  # noqa: E731
                       else f"FUT|{'1Min' if t.get('trk') == 'v1m' else '15Min'}|{t['symbol']}")
    lots = lambda t: f"{t['n']} {t['contract']}"  # noqa: E731
    rs5 = [t["r"] for t in closed if t.get("r") is not None and t["trk"] == "v5"]
    rs1 = [t["r"] for t in closed if t.get("r") is not None and t["trk"] == "v1m"]
    rst = [t["r"] for t in closed if t.get("r") is not None and t["trk"] == "tal"]
    rules = (f"בוט 6 · חוזים עתידיים מיקרו (MNQ, MES, MYM) · שיטת הגאפ של טל (מ־12.11.2026): דחף בגרף השעה, הגאפ בין נר 1 לנר 3 "
             f"שנשאר פתוח, כניסה באמצע הגאפ, סטופ 2 טיקים מעבר לסווינג השני של 5 דקות, יעד בסווינג השני בצד השני, בלי ניהול, "
             f"תוקף 120 שעות · "
             f"חוקי FTMO Futures, {FA.PLAN}: הפסד מקסימלי נגרר ${FA.MAX_LOSS:,.0f} לפי סוף יום, מגבלה יומית ${FA.DAILY_LOSS:,.0f}, "
             f"יעד ${FA.TARGET:,.0f}, היום הטוב ביותר עד {FA.CONSISTENCY:.0%} מהרווח, עד {FA.MAX_MICROS} חוזי מיקרו · "
             f"אין כניסה אחרי 15:45 שעון ניו יורק, הכול נסגר ב־16:05, לא מחזיקים בסוף שבוע · "
             f"סיכון {FA.RISK_TAL * 100:g}% לעסקה, מעוגל לחוזים שלמים · "
             f"עמלה ${FA.COMM:g} לחוזה הלוך־חזור + טיק החלקה · המחירים מה־CFD של FTMO, לא מהחוזה עצמו")
    return {"bot": FUT_NAME, "label": "בוט 6 · חוזים עתידיים (וירטואלי)", "watch_only": True, "rules": rules,
            "account": f"וירטואלי · חוזים עתידיים · FTMO {FA.PLAN}", "server": "צפייה בלבד · בלי כסף",
            "initial": fa.a["initial"], "risk": FA.RISK_TAL, "balance": round(bal, 2), "equity": round(eq, 2),
            "day_start": st["day_start"], "halted_until": None, "started": fa.a["started"], "withdrawals": [],
            "combos": [f"TAL|1Hour|{n}" for n in FA.SPEC], "equity_curve": ec, "lows": {},
            "updated_utc": now_iso, "futures": st,
            "summary": {"v5": {"closed": len(rs5), "sum_r": round(float(np.sum(rs5)), 2) if rs5 else 0.0},
                        "v1m": {"closed": len(rs1), "sum_r": round(float(np.sum(rs1)), 2) if rs1 else 0.0},
                        "tal": {"closed": len(rst), "sum_r": round(float(np.sum(rst)), 2) if rst else 0.0},
                        "skipped": sum(1 for t in closed if t.get("skipped") and t["skipped"] != "before")},
            "positions": [{"ticket": t["id"], "side": "buy" if t["dir"] > 0 else "sell", "dir": "קנייה" if t["dir"] > 0 else "מכירה",
                           "combo": combo(t), "symbol": t["symbol"], "lots": lots(t), "open": t["entry"],
                           "now": t["entry"] + t["dir"] * (t.get("r_now") or 0) * abs(t["entry"] - t["stop"]),
                           "sl": t["stop"], "tp": t["target"], "sl0": t["stop"], "tp0": t["target"],
                           "sl_usd": -(t.get("risk_usd") or 0), "tp_usd": (t.get("risk_usd") or 0) * (t.get("rr") or 0),
                           "pnl": t.get("pnl_now") or 0, "since": t["time"], "pip": 1.0, "free": False, "mae": None, "mfe": None}
                          for t in live_ if t["status"] == "open" and t.get("n")],
            "trades": [{"time": t["exit_time"], "combo": combo(t), "symbol": t["symbol"], "dir": "קנייה" if t["dir"] > 0 else "מכירה",
                        "lots": lots(t), "price": t["exit"], "pnl": t["pnl"], "why": t.get("reason")} for t in counted],
            "closed": [{"id": t["id"], "combo": combo(t), "symbol": t["symbol"], "side": "buy" if t["dir"] > 0 else "sell",
                        "lots": lots(t), "open_t": t["time"], "open_px": t["entry"], "close_t": t["exit_time"], "close_px": t["exit"],
                        "sl0": t["stop"], "tp0": t["target"], "sl": t["stop"], "tp": t["target"], "mae": None, "mfe": None,
                        "gross": t["pnl"], "swap": 0, "comm": 0, "net": t["pnl"], "why": t.get("reason") or "",
                        "reason": FUT_WORD.get(t.get("reason"), ""), "pip": 1.0} for t in counted],
            "watch": sorted(ftrades.values(), key=lambda t: t.get("time") or t.get("setup_t") or "", reverse=True),
            "vacct": fa.a}


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
                           "יעד 2 פיפס לפני התיקון האמיתי האחרון לפני הכניסה · יחס מינימום 1:1 · מסלול פערים בדקה: אותם כללים בגרף הדקה, המתנה של 96 דקות לנגיעה · מסלול פערים (v5): כל פער של נר חזק במדדים (גם JP225), כניסה בנגיעה אחרי שהמהלך פנה, סטופ מתחת לבסיס הפער, יעד בשיא המהלך, 30% יוצאים ב־90% מהדרך והשאר לברייק · מסלול מתוקן (v4): הדחף הוא המהלך הגדול, הפער הראשון שנשאר פתוח, סטופ מעבר לסווינג השלישי, יציאה ב־80% מהדרך ליעד, 12 מכשירים · מסלול רחב: אותם כללים עם נרות חזקים פחות (פי 1.8 מהממוצע במקום 2.5), בערך פי 4 עסקאות · "
                           "איסוף נתונים: נפט, כסף, USDCAD, ‏EURGBP, ‏EU50, ‏AUS200 - הפסידו בבדיקה, מוצגים בנפרד ולא נכנסים לחשבון הווירטואלי"})
    trades = {t["id"]: t for t in (state.get("watch") or [])}
    acc = state.get("vacct") or {}
    trades = gap_id_fix(trades, acc)
    if acc.get("v") != ACCT_V:
        # a new account: nothing that happened before counts
        acc = {"v": ACCT_V, "initial": INITIAL, "risk_pct": round(RISK_V5 * 100, 3), "risk_v1m_pct": round(RISK_V1M * 100, 4),
               "ledger": [], "breached": None, "started": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
        for t in trades.values():
            for f in ("risk_usd", "skipped", "pnl", "pnl_now"):
                t.pop(f, None)

    def balance():
        return acc["initial"] + sum(x["pnl"] for x in acc["ledger"])

    def prague_day(ts):
        return pd.Timestamp(ts).tz_convert("Europe/Prague").strftime("%Y-%m-%d")

    def day_pnl(day):
        return sum(x["pnl"] for x in acc["ledger"] if prague_day(x["exit_time"]) == day)
    try:
        fstate = json.loads(FUT_F.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        fstate = {}
    fa = FA.Account(fstate.get("vacct"))
    ftrades = {t["id"]: t for t in (fstate.get("watch") or [])} if (fstate.get("vacct") or {}).get("v") == FA.VERSION else {}
    REP.remember_alerts(LIVE / f"{NAME}_alerts.json")
    last_push = last_fpush = 0
    while True:
        t_cycle = time.time()
        load_settings()
        cal.refresh()
        seen_now, errors, fseen = set(), [], set()
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
                if name in V5_15_SYMBOLS:
                    keep5 = _set_og("v5")    # live/bots/bot5.json (from 10.11.2026: 30% out at 90% of the way, rest to BE)
                    try:
                        for x in OG.run_gaps(name, df, cst, stop_at, noent, live=True, variant="v5"):
                            if x.get("status") == "pending" and pd.Timestamp(x["setup_t"]) < df.index[-1] - pd.Timedelta(hours=OG.EXPIRY):
                                continue
                            x["v5"] = True
                            res.append(x)
                    finally:
                        for k_, v_ in keep5.items():
                            setattr(OG, k_, v_)
                if name in V5_SYMBOLS:
                    # v5 on 1-minute candles (user, 10.11.2026: "if it is profitable, why not add it"): the same rules,
                    # gaps found on the 1-minute chart, waiting 96 minutes for the touch
                    keep1 = _set_og("v1m")
                    try:
                        for x in OG.run_gaps(name, df, cst, stop_at, noent, live=True, variant="v1m"):
                            if x.get("status") == "pending" and pd.Timestamp(x["setup_t"]) < df.index[-1] - pd.Timedelta(minutes=96):
                                continue
                            x["v1m"] = True
                            res.append(x)
                    finally:
                        for k_, v_ in keep1.items():
                            setattr(OG, k_, v_)
                if name in FA.SPEC:
                    fut_step(name, df, cst, stop_at, noent, fa, ftrades, fseen)
                if name in V4_SYMBOLS:
                    keep4 = {k_: getattr(CORE, k_) for k_ in V4}
                    try:
                        for k_, v_ in V4.items():
                            setattr(CORE, k_, v_)
                        for x in CORE.run_symbol(name, df, cst, stop_at, noent, [VARIANT], live=True):
                            x["v4"] = True
                            res.append(x)
                    finally:
                        for k_, v_ in keep4.items():
                            setattr(CORE, k_, v_)
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
                           "r": x.get("r"), "r_now": x.get("r_now"), "mark": old.get("mark"), "wide": bool(x.get("wide")),
                           "why": x.get("why"), "touch_t": iso(x.get("touch_t")),
                           "extra": name in EXTRA and not x.get("v4"), "v4": bool(x.get("v4")), "v5": bool(x.get("v5")), "v1m": bool(x.get("v1m"))}
                    if x["status"] != "closed" or "m15" not in old or old.get("status") != x["status"] or old.get("chart_v") != 2:
                        rec["chart_v"] = 2
                        ls = pd.Timestamp(x["leg_start"])
                        t_in = pd.Timestamp(x.get("time")) if x.get("time") is not None else df.index[-1]
                        t_out = pd.Timestamp(x["exit_time"]) if x.get("exit_time") is not None else df.index[-1]
                        rec["m15"] = bars(df, ls - pd.Timedelta(hours=6), t_in + pd.Timedelta(hours=1), "15min")
                        # the whole move on the 1-minute chart (user, 10.11.2026): from before the impulse to after the exit,
                        # at most the last 12 hours of that window
                        m1_end = t_out + pd.Timedelta(minutes=10)
                        m1_start = max(min(ls - pd.Timedelta(minutes=30), t_in - pd.Timedelta(minutes=120)), m1_end - pd.Timedelta(hours=12))
                        rec["m1"] = bars(df, m1_start, m1_end, "1min")
                    else:
                        rec["m15"], rec["m1"], rec["chart_v"] = old.get("m15"), old.get("m1"), old.get("chart_v")
                    prev = old.get("status")
                    # the virtual account: the risk is fixed when the trade opens, from the balance at that moment
                    for f in ("risk_usd", "skipped"):
                        if f in old:
                            rec[f] = old[f]
                    if "risk_usd" not in rec and "skipped" not in rec:
                        if not (rec["v5"] or rec["v1m"]):
                            rec["skipped"] = "v4" if rec["v4"] else "extra" if rec["extra"] else "old"   # only the gap tracks count
                        elif rec["status"] in ("open", "closed") and rec["time"] and rec["time"] < acc["started"]:
                            rec["skipped"] = "before"                                       # before the account opened
                    if rec["status"] in ("open", "closed") and "risk_usd" not in rec and "skipped" not in rec:
                        bal, day = balance(), prague_day(rec["time"])
                        start = bal - day_pnl(day)
                        if acc.get("breached"):
                            rec["skipped"] = "breach"
                        elif day_pnl(day) <= -DAILY_STOP * start:
                            rec["skipped"] = "daily"
                        else:
                            rec["risk_usd"] = round(bal * (RISK_V5 if rec["v5"] else RISK_V1M), 2)
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
                    side = ("קנייה" if rec["dir"] > 0 else "מכירה") + (" · פערים בדקה" if rec["v1m"] else " · פערים (v5)" if rec["v5"] else " · מתוקן (v4)" if rec["v4"] else " · מסלול רחב" if rec["wide"] else "") + (" · איסוף נתונים בלבד" if rec["extra"] else "")
                    # notifications (user, 10.11.2026: "I'm bombarded"): only trades of the virtual account - opened and closed.
                    # Waiting setups and touched-but-skipped ones stay in the app, without a push.
                    quiet = not rec.get("risk_usd")
                    if False and rec["status"] == "cancelled" and rec.get("why") and prev != "cancelled":
                        why = {"rr": f"היעד קרוב מדי (יחס 1:{rec.get('rr')})", "cost": f"העלות {round(100 * (rec.get('cost_r') or 0))}% מהסיכון"}.get(rec["why"], rec["why"])
                        REP.notify(f"סקלפינג (צפייה): {name} נגע ולא נכנס", f"{side} · נגיעה ב־{fmt(name, rec['entry'])} · לא נכנס: {why}",
                                   tags=["eyes"], key=f"wx:{k}", every=86400)
                    if False and prev is None and rec["status"] == "pending":
                        REP.notify(f"סקלפינג (צפייה): {name} מחכה לכניסה", f"{side} · דחף של {rec['leg']} נרות · כניסה בנגיעה ב־{fmt(name, rec['entry'])}\n"
                                   f"סטופ {fmt(name, rec['stop'])} · לא נשלחת פקודה", tags=["eyes"], key=f"wp:{k}", every=86400)
                    if not quiet and prev in (None, "pending") and rec["status"] in ("open", "closed"):
                        REP.notify(f"סקלפינג (צפייה): {name} עסקה וירטואלית נפתחה", f"{side} @ {fmt(name, rec['entry'])} · סטופ {fmt(name, rec['stop'])} · "
                                   f"יעד {fmt(name, rec['target'])} · יחס 1:{rec['rr']}\nפתח את האפליקציה וסמן אם היית לוקח אותה",
                                   tags=["eyes"], key=f"wo:{k}", every=86400)
                    if not quiet and prev != "closed" and rec["status"] == "closed":
                        word = {"target": "ביעד", "stop": "בסטופ", "forced": "בסגירה כפויה", "breakeven": "בברייק אחרי יציאה חלקית"}.get(rec["reason"], "")
                        usd = f" · {'+' if rec.get('pnl', 0) >= 0 else '−'}${abs(rec.get('pnl', 0)):,.0f} · יתרה ${balance():,.0f}" if rec.get("pnl") is not None else ""
                        REP.notify(f"סקלפינג (צפייה): {name} נסגרה {word}", f"{side} · {'+' if (rec['r'] or 0) >= 0 else ''}{rec['r']:.2f} סיכון{usd} (וירטואלי)",
                                   tags=["eyes"], key=f"wc:{k}", every=86400)
                for k, t in list(trades.items()):
                    if t["symbol"] == name and t["status"] == "open" and k not in seen_now and (t.get("v5") or t.get("v1m")):
                        seen_now.add(k)
                        t.update(follow(t, df, stop_at))
                        if t.get("risk_usd"):
                            if t["status"] == "closed":
                                t["pnl"] = round(t["r"] * t["risk_usd"], 2)
                                if not any(x["id"] == k for x in acc["ledger"]):
                                    acc["ledger"].append({"id": k, "exit_time": t["exit_time"], "pnl": t["pnl"]})
                                word = {"target": "ביעד", "stop": "בסטופ", "forced": "בסגירה כפויה", "breakeven": "בברייק אחרי יציאה חלקית"}.get(t["reason"], "")
                                REP.notify(f"סקלפינג (צפייה): {name} נסגרה {word}", f"{'+' if t['r'] >= 0 else ''}{t['r']:.2f} סיכון · "
                                           f"{'+' if t['pnl'] >= 0 else '−'}${abs(t['pnl']):,.0f} · יתרה ${balance():,.0f} (וירטואלי)",
                                           tags=["eyes"], key=f"wc:{k}", every=86400)
                            elif t.get("r_now") is not None:
                                t["pnl_now"] = round(t["r_now"] * t["risk_usd"], 2)
            except Exception:  # noqa: BLE001 - one instrument must not stop the others
                errors.append(f"{name}: {traceback.format_exc()[-300:]}")
        # a pending setup that no longer appears (cancelled: the leg went on, expired, stop first) is dropped
        for k, t in list(trades.items()):
            if t["status"] == "pending" and k not in seen_now and t["symbol"] in src.df:
                t["status"] = "cancelled"
        try:
            marks = json.loads(MARKS_F.read_text(encoding="utf-8"))
            olds = {t["old_id"]: k for k, t in trades.items() if t.get("old_id")}
            for k, m in marks.items():
                k = k if k in trades else olds.get(k)
                if k in trades:
                    trades[k]["mark"] = m
        except Exception:  # noqa: BLE001
            pass
        closed = sorted([t for t in trades.values() if t["status"] == "closed"], key=lambda t: t["time"] or "")[-KEEP:]
        live_ = [t for t in trades.values() if t["status"] in ("pending", "open")]
        cancelled = sorted([t for t in trades.values() if t["status"] == "cancelled"], key=lambda t: t.get("setup_t") or "")[-20:]
        trades = {t["id"]: t for t in closed + live_ + cancelled}
        core_closed = [t for t in closed if not t.get("extra") and not t.get("v4") and not t.get("v5") and not t.get("v1m")]
        rs_v1m = [t["r"] for t in closed if t.get("r") is not None and t.get("v1m")]
        rs_v5 = [t["r"] for t in closed if t.get("r") is not None and t.get("v5")]
        rs_v4 = [t["r"] for t in closed if t.get("r") is not None and t.get("v4")]
        rs = [t["r"] for t in core_closed if t.get("r") is not None]
        rs_strict = [t["r"] for t in core_closed if t.get("r") is not None and not t.get("wide")]
        rs_wide = [t["r"] for t in core_closed if t.get("r") is not None and t.get("wide")]
        rs_extra = [t["r"] for t in closed if t.get("r") is not None and t.get("extra")]
        state.update({"updated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                      "watch": sorted(trades.values(), key=lambda t: t.get("time") or t.get("setup_t") or "", reverse=True),
                      "summary": {"closed": len(rs), "sum_r": round(float(np.sum(rs)), 2) if rs else 0.0,
                                  "avg_r": round(float(np.mean(rs)), 3) if rs else None, "win_pct": round(100 * float(np.mean(np.array(rs) > 0)), 1) if rs else None,
                                  "open": sum(t["status"] == "open" and not any(t.get(k_) for k_ in ("extra", "v4", "v5", "v1m")) for t in live_),
                                  "pending": sum(t["status"] == "pending" and not any(t.get(k_) for k_ in ("extra", "v4", "v5", "v1m")) for t in live_),
                                  "strict": {"closed": len(rs_strict), "sum_r": round(float(np.sum(rs_strict)), 2) if rs_strict else 0.0},
                                  "wide": {"closed": len(rs_wide), "sum_r": round(float(np.sum(rs_wide)), 2) if rs_wide else 0.0},
                                  "v1m": {"closed": len(rs_v1m), "sum_r": round(float(np.sum(rs_v1m)), 2) if rs_v1m else 0.0,
                                          "win_pct": round(100 * float(np.mean(np.array(rs_v1m) > 0)), 1) if rs_v1m else None,
                                          "open": sum(t["status"] == "open" and t.get("v1m") for t in live_),
                                          "pending": sum(t["status"] == "pending" and t.get("v1m") for t in live_)},
                                  "v5": {"closed": len(rs_v5), "sum_r": round(float(np.sum(rs_v5)), 2) if rs_v5 else 0.0,
                                         "win_pct": round(100 * float(np.mean(np.array(rs_v5) > 0)), 1) if rs_v5 else None,
                                         "open": sum(t["status"] == "open" and t.get("v5") for t in live_),
                                         "pending": sum(t["status"] == "pending" and t.get("v5") for t in live_)},
                                  "v4": {"closed": len(rs_v4), "sum_r": round(float(np.sum(rs_v4)), 2) if rs_v4 else 0.0,
                                         "win_pct": round(100 * float(np.mean(np.array(rs_v4) > 0)), 1) if rs_v4 else None,
                                         "open": sum(t["status"] == "open" and t.get("v4") for t in live_),
                                         "pending": sum(t["status"] == "pending" and t.get("v4") for t in live_)},
                                  "extra": {"closed": len(rs_extra), "sum_r": round(float(np.sum(rs_extra)), 2) if rs_extra else 0.0,
                                            "symbols": list(EXTRA)}},
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
        combo = lambda n, t=None: f"OWN|{'1Min' if (t or {}).get('v1m') else '15Min'}|{n}"  # noqa: E731
        counted = [t for t in closed if t.get("risk_usd") and t.get("pnl") is not None]
        state.update({"account": "וירטואלי · פערים v5 + דקה", "server": "צפייה בלבד · בלי כסף · סיכון 0.4% / 0.075%", "initial": acc["initial"], "risk": RISK_V5,
                      "balance": round(bal, 2), "equity": round(eq, 2), "day_start": round(bal - day_pnl(today), 2),
                      "halted_until": None, "combos": [combo(n) for n in V5_15_SYMBOLS] + [combo(n, {"v1m": True}) for n in V5_SYMBOLS], "started": acc["started"], "withdrawals": [],
                      "equity_curve": ec, "lows": {},
                      "positions": [{"ticket": t["id"], "side": "buy" if t["dir"] > 0 else "sell", "dir": "קנייה" if t["dir"] > 0 else "מכירה",
                                     "combo": combo(t["symbol"], t), "symbol": t["symbol"], "lots": "וירטואלי", "open": t["entry"],
                                     "now": t["entry"] + t["dir"] * (t.get("r_now") or 0) * abs(t["entry"] - t["stop"]),
                                     "sl": t["stop"], "tp": t["target"], "sl0": t["stop"], "tp0": t["target"],
                                     "sl_usd": -(t.get("risk_usd") or 0), "tp_usd": (t.get("risk_usd") or 0) * (t.get("rr") or 0),
                                     "pnl": t.get("pnl_now") or 0, "since": t["time"], "pip": pip(t["symbol"]), "free": False,
                                     "mae": None, "mfe": None} for t in live_ if t["status"] == "open" and t.get("risk_usd")],
                      "trades": [{"time": t["exit_time"], "combo": combo(t["symbol"], t), "symbol": t["symbol"],
                                  "dir": "קנייה" if t["dir"] > 0 else "מכירה", "lots": "וירטואלי", "price": t["exit"], "pnl": t["pnl"],
                                  "why": t.get("reason")} for t in counted],
                      "closed": [{"id": t["id"], "combo": combo(t["symbol"], t), "symbol": t["symbol"], "side": "buy" if t["dir"] > 0 else "sell",
                                  "lots": "וירטואלי", "open_t": t["time"], "open_px": t["entry"], "close_t": t["exit_time"], "close_px": t["exit"],
                                  "sl0": t["stop"], "tp0": t["target"], "sl": t["stop"], "tp": t["target"], "mae": None, "mfe": None,
                                  "gross": t["pnl"], "swap": 0, "comm": 0, "net": t["pnl"], "why": t.get("reason") or "",
                                  "reason": {"target": "יעד", "stop": "סטופ", "forced": "סגירה כפויה", "breakeven": "ברייק (30% יצאו)"}.get(t.get("reason"), ""),
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
        try:
            fstate = fut_state(fa, ftrades, fseen, src)
            FUT_F.write_text(json.dumps(fstate, ensure_ascii=False, default=str), encoding="utf-8")
            if time.time() - last_fpush > 120:
                err = REP.push_state(FUT_NAME, fstate)
                if err:
                    log(err)
                last_fpush = time.time()
        except Exception:  # noqa: BLE001
            errors.append(f"bot 6: {traceback.format_exc()[-300:]}")
        if errors:
            log("; ".join(e.splitlines()[-1] for e in errors)[:400])
        if a.once:
            return
        time.sleep(max(5.0, 60 - (time.time() - t_cycle)))


if __name__ == "__main__":
    main()
