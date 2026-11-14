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
from agents import scout_fut as SF  # noqa: E402
from agents import tal_gap as TG  # noqa: E402
from live import fut_account as FA  # noqa: E402
from live import reporter as REP  # noqa: E402

NAME = "watch_own"
LABEL = "בוט 5 - סקאלפינג · צפייה בלבד"
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
DAYS = 14                     # history kept per instrument (the 4-hour gaps stay valid 5 days; 12.11.2026)
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
BOT_MAX_LOSS = 0.09     # the bots' own total limit (data/agents/tal_thinking.md: 4.5% a day, 9% in total)
# the virtual account, v2 (user, 10.11.2026: "as if it were a real demo account"): a fresh $100,000 from the moment this
# version starts, only the two gap tracks count, each with its own risk: v5 (15m) 0.4%, the 1-minute track 0.075%.
# FTMO rules: no new trades after a 4.5% loss in the day, the account burns at a 10% loss from the start.
# v3 (10.11.2026, the review before the user's calibration): a fresh start with the corrected trade selection
ACCT_V, RISK_V5, RISK_V1M = 4, 0.004, 0.00075     # 4 (12.11.2026): the user's gap method - a fresh $100K account
# the user's gap method on bot 5 (agents/tal_gap.py; cards approved by the manager 12.11.2026, each "maybe luck" -> risk
# x0.7 of the bot's 0.4%). Since then the only tracks of bot 5's virtual account; the old gap tracks (v5 / 1-minute)
# are still shown but no longer counted.
TAL5_SYMBOLS = ["US100", "US30", "US500", "GER40", "JP225"]
TAL5 = {"t15_1": (("15m", "1m", 2), {"GBODY": 1.5, "MIN_GAP": 0.3, "SW": 3, "MIN_RR": 1.0, "EXPIRY": 48, "PROT": "shock"}),
        "t1h_15": (("1h", "15m", 1), {"GBODY": 1.5, "MIN_GAP": 0.3, "SW": 3, "MIN_RR": 0.5, "EXPIRY": 48, "PROT": "none"}),
        # 12.11.2026 18:22 (user: yes - the agent's newest search, train and 2026 agree): 4h/1h's new settings
        # (+0.16R a trade on 2026, 189 trades, was +0.08R) and 4h/15m with the stop to entry at 70% of the way to the
        # target (be70: +0.175R a trade on 2026, 311 trades, 46% winners)
        "t4h_1h": (("4h", "1h", 1), {"GBODY": 0.0, "MIN_GAP": 0.0, "SW": 3, "MIN_RR": 1.0, "EXPIRY": 120, "PROT": "none"}),
        "t4h_15": (("4h", "15m", 2), {"GBODY": 0.0, "MIN_GAP": 0.0, "SW": 5, "MIN_RR": 0.0, "EXPIRY": 120, "PROT": "cal15", "EXIT": "be70"}),
        "t15_5": (("15m", "5m", 1), {"GBODY": 1.5, "MIN_GAP": 0.3, "SW": 3, "MIN_RR": 1.0, "EXPIRY": 120, "PROT": "none"})}
TAL5_NAME = {"t15_1": "15 דק' / דקה", "t1h_15": "שעה / 15 דק'", "t4h_1h": "4 שעות / שעה", "t15_5": "15 דק' / 5 דק'",
             "t4h_15": "4 שעות / 15 דק' (סטופ לכניסה ב־70%)"}
# t15_5 was "15Min" too, like t15_1 - two tracks under one name, so the loss report of 12.11 could not tell them apart
# (13.11.2026 night): from now "15Min5m"; t15_1 keeps "15Min" (its history)
TAL5_TF = {"t15_1": "15Min", "t1h_15": "1Hour", "t4h_1h": "4Hour", "t15_5": "15Min5m", "t4h_15": "4Hour15"}
RISK_TAL5 = 0.004 * 0.7
# 13.11.2026 (bot 5's smart agent): the tracks above are also settings in live/bots/bot5.json ("tal_tracks": {track:
# {"pair": [high, low], "cfg": {...}}}, risk per track in "risk"), so an approved card of the gap method is built by
# agents/builder.py at once, like bot 6. A file that is missing or odd keeps the tracks in the code.
TAL5_RISK = {}
# an instrument's own gap track (13.11.2026, agents/tal_wide.py: the method on every FTMO forex pair, the metals, the
# other indices): "symbols" in its "tal_tracks" entry - {track: [symbol]}; a track without it trades TAL5_SYMBOLS.
# Such an instrument is added to SYMBOLS (and EXTRA: the old tracks' results on it stay out of their summary).
TAL5_SYMS = {}
TAL5_ONE = [False]     # "tal_one_per_side": a track does not enter a symbol / direction another track holds (off until approved)
TAL5_MAXDIR = [0]      # "tal_max_dir" (13.11.2026): at most K counted trades open in one direction on all the indices together (0 = off until approved)
TAL5_SCALE = [1.0]      # "tal_risk_scale" (13.11.2026 night): one factor on every gap track's risk (1 until approved)
TAL5_HOURS_OFF = set()  # "tal_hours_off" (13.11.2026 night): UTC hours of entry where no new counted trade opens (empty until approved)
# bot 6 (13.11.2026 night, the shared methods with bot 5): the same three rules over bot 6's gap tracks, from the same keys
# in live/bots/bot6.json - only the gap method's tracks, not the scout strategies (chosen on the gap tracks together)
TAL6_ONE = [False]
TAL6_MAXDIR = [0]
TAL6_HOURS_OFF = set()
_TAL5_CODE = {k: (p, dict(c)) for k, (p, c) in TAL5.items()}


def tal5_name(trk):
    if scout5_combo(trk):
        sid_, tf_, _ = scout5_combo(trk).split("|")
        return f"{sid_} · {TG.TF_HE.get({'1Min': '1m', '5Min': '5m', '15Min': '15m', '1Hour': '1h'}.get(tf_, tf_), tf_)}"
    if trk in TAL5_NAME:
        return TAL5_NAME[trk]
    p = (TAL5.get(trk) or _TAL5_CODE.get(trk) or ((trk, trk),))[0]
    return f"{TG.TF_HE.get(p[0], p[0])} / {TG.TF_HE.get(p[1], p[1])}"


def tal5_tf(trk):
    if trk in TAL5_TF:
        return TAL5_TF[trk]
    p = (TAL5.get(trk) or ((trk, ""),))[0]
    return f"{TF_COMBO.get(p[0], p[0])}{p[1]}"
# The settings bots 5 and 6 trade with live in live/bots/bot5.json and bot6.json (12.11.2026: the virtual bots are run
# as if they were real accounts - an approved change goes in at once, like bots 1-4). Read every cycle; a file that is
# missing or odd keeps the values above.
SET5 = {"v5": {"STOP": "gap", "DOMINANT": False, "V5_MAX_COST": 0.10, "PART_FRAC": 0.3, "PART_WAY": 0.9},
        "v1m": {"STOP": "gap", "DOMINANT": False, "V5_MAX_COST": 0.10, "PART_FRAC": 0.3, "PART_WAY": 0.9, "TFM": 1}}
_SET = {"m5": None, "m6": None}
_TAL0 = dict(FA.TAL_TRACKS)          # the tracks in the code; the file's main track is laid over them on every load
TF_COMBO = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "4h": "4Hour", "1d": "1Day"}
# bot 6's scout strategies (13.11.2026, agents/scout_fut.py): "scout" in live/bots/bot6.json - {combo: {"v", "since"}},
# written by agents/builder.py only from an approved card of bot 6's own test. Empty = none trades.
SCOUT6 = {}
# bot 5's scout strategies (14.11.2026, bot 5's smart agent - the manager approved SC042 and SC069 on XAUUSD 15Min, the
# first scout strategies of bot 5): "scout" in live/bots/bot5.json - {combo "SID|TF|SYM": {"since", "card", "spread"?}},
# written by agents/builder.py from an approved card of agent_portfolio.py's test (the CFD bots' backtest machine:
# run_rules with the live rules - break exit, holidays, FTMO's restricted events minute by minute, breakeven at 60% of
# the way + costs, FTMO's costs). The same functions run here on the live bars (scout5_trades). Risk per trade:
# "scout:<combo>" in "risk" (the manager's star: x0.7), else "scout", else the bot's 0.4%. Empty = none trades.
SCOUT5 = {}
# FTMO's live spread per instrument (bot 1's MetaTrader quotes, sampled once a minute while the market is open): the
# scout strategies' cost. "spread" in the strategy's entry overrides it; no sample yet -> FTMO's class estimate.
SPREAD5 = {}
RISK_SCOUT5 = 0.004


def scout5_combo(trk):
    return str(trk)[3:] if str(trk).startswith("sc:") else None


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
                tt = d.get("tal_tracks")
                if isinstance(tt, dict) and tt:
                    pairs_ = {(p_[0], p_[1]): tuple(p_) for p_ in TG.PAIRS}
                    new_ = {}
                    syms_ = {}
                    for trk, v in tt.items():
                        pr = pairs_.get(tuple((v or {}).get("pair") or [])[:2])
                        cf = (v or {}).get("cfg")
                        if pr and isinstance(cf, dict) and all(k in cf for k in ("GBODY", "MIN_GAP", "SW", "MIN_RR", "EXPIRY", "PROT")):
                            sy = (v or {}).get("symbols")
                            if sy is not None:
                                sy = [x for x in sy if isinstance(x, str) and x in U.ALL] if isinstance(sy, list) else []
                                if not sy:
                                    continue                  # a track for instruments we do not know: not traded
                                syms_[str(trk)] = sy
                            new_[str(trk)] = (pr, dict(cf))
                    if new_:
                        TAL5.clear()
                        TAL5.update(new_)
                        TAL5_SYMS.clear()
                        TAL5_SYMS.update(syms_)
                        for sy in {x for v_ in syms_.values() for x in v_}:
                            if sy not in SYMBOLS:
                                code_ = U.ALL[sy][0]
                                cand_ = list(dict.fromkeys([code_.replace("/", ""), code_, sy]))
                                SYMBOLS[sy] = cand_
                                EXTRA[sy] = cand_
                TAL5_ONE[0] = bool(d.get("tal_one_per_side", False))
                try:
                    TAL5_MAXDIR[0] = max(0, int(d.get("tal_max_dir", 0) or 0))
                except (TypeError, ValueError):
                    TAL5_MAXDIR[0] = 0
                TAL5_HOURS_OFF.clear()
                try:
                    TAL5_HOURS_OFF.update(int(h_) for h_ in (d.get("tal_hours_off") or []) if 0 <= int(h_) <= 23)
                except (TypeError, ValueError):
                    TAL5_HOURS_OFF.clear()
                try:
                    sc_ = float(d.get("tal_risk_scale", 1.0) or 1.0)
                    TAL5_SCALE[0] = sc_ if 0.1 <= sc_ <= 1.0 else 1.0
                except (TypeError, ValueError):
                    TAL5_SCALE[0] = 1.0
                TAL5_RISK.clear()
                for trk in TAL5:
                    v = rk.get(trk, rk.get("tal"))
                    if v is not None and 0 <= float(v) <= 0.02:
                        TAL5_RISK[trk] = float(v)
                SCOUT5.clear()
                from agents import rules_agent as R_
                for combo_, e_ in (d.get("scout") or {}).items():
                    try:
                        sid_, tf_, sym_ = str(combo_).split("|")
                        e_ = e_ or {}
                        if sid_ in R_.STRATEGIES and tf_ in SF.TF_MIN and sym_ in U.ALL and e_.get("since"):
                            sp_ = e_.get("spread")
                            SCOUT5[str(combo_)] = {"sid": sid_, "tf": tf_, "sym": sym_, "since": str(e_["since"]),
                                                   "spread": float(sp_) if sp_ is not None and float(sp_) > 0 else None}
                            v = rk.get(f"scout:{combo_}", rk.get("scout", RISK_SCOUT5))
                            TAL5_RISK[f"sc:{combo_}"] = float(v) if 0 <= float(v) <= 0.02 else RISK_SCOUT5
                            if sym_ not in SYMBOLS:
                                code_ = U.ALL[sym_][0]
                                SYMBOLS[sym_] = list(dict.fromkeys([code_.replace("/", ""), code_, sym_]))
                    except Exception:  # noqa: BLE001 - an odd entry is skipped, the rest trade
                        continue
            else:
                tal = d.get("tal") or {}
                if tal.get("pair") and tuple(tal["pair"]) in {tuple(p_) for p_ in TG.PAIRS}:
                    FA.TAL_PAIR = tuple(tal["pair"])
                if isinstance(tal.get("cfg"), dict):    # over the code's rules (not the last load's: a key the
                    FA.TAL_CFG = {**_TAL0["tal"][1], **tal["cfg"]}    # file dropped, e.g. EXIT, is dropped here too)
                # the pair / rules the builder wrote are what the main track trades (12.11.2026: before this, fut_step
                # read FA.TAL_TRACKS only, so an approved card changed the file and not the trades). Another track on
                # the same pair stops taking new trades (its open ones are still followed to their exit).
                FA.TAL_TRACKS.clear()
                FA.TAL_TRACKS.update(_TAL0)
                FA.TAL_TRACKS["tal"] = (FA.TAL_PAIR, dict(FA.TAL_CFG))
                FA.TAL_NAME["tal"] = f"{TG.TF_HE[FA.TAL_PAIR[0]]} / {TG.TF_HE[FA.TAL_PAIR[1]]}"
                FA.TAL_COMBO["tal"] = "TAL|" + TF_COMBO.get(FA.TAL_PAIR[0], FA.TAL_PAIR[0])
                # the other tracks are settings too (13.11.2026, bot 6's smart agent - like bot 5's "tal_tracks"): an
                # approved card on another pair replaces that track's rules or adds a track, built by agents/builder.py.
                # The main track stays in "tal" above; a "tal" entry here is ignored. Odd entries keep the code's track.
                tt = d.get("tal_tracks")
                if isinstance(tt, dict):
                    pairs_ = {(p_[0], p_[1]): tuple(p_) for p_ in TG.PAIRS}
                    for trk_, v_ in tt.items():
                        pr = pairs_.get(tuple((v_ or {}).get("pair") or [])[:2])
                        cf = (v_ or {}).get("cfg")
                        if str(trk_) == "tal" or not pr or not isinstance(cf, dict) \
                                or not all(k in cf for k in ("GBODY", "MIN_GAP", "SW", "MIN_RR", "EXPIRY", "PROT")):
                            continue
                        trk_ = str(trk_)
                        FA.TAL_TRACKS[trk_] = (pr, dict(cf))
                        FA.TAL_NAME[trk_] = f"{TG.TF_HE[pr[0]]} / {TG.TF_HE[pr[1]]}"
                        if trk_ not in FA.TAL_COMBO:
                            FA.TAL_COMBO[trk_] = f"TAL|{TF_COMBO.get(pr[0], pr[0])}{pr[1]}"
                        FA.RISK.setdefault(trk_, FA.RISK_TAL)
                for trk_ in [k_ for k_, (p_, _) in FA.TAL_TRACKS.items() if k_ != "tal" and tuple(p_) == tuple(FA.TAL_PAIR)]:
                    FA.TAL_TRACKS.pop(trk_)
                SCOUT6.clear()
                for combo_, e_ in (d.get("scout") or {}).items():
                    try:
                        sid_, tf_, sym_ = str(combo_).split("|")
                        v_ = (e_ or {}).get("v") or {}
                        if sym_ in FA.SPEC and tf_ in SF.TF_MIN and all(k_ in v_ for k_ in SF.BASE) and (e_ or {}).get("since"):
                            SCOUT6[str(combo_)] = {"sid": sid_, "tf": tf_, "sym": sym_, "v": dict(v_), "since": str(e_["since"])}
                    except Exception:  # noqa: BLE001 - an odd entry is skipped, the rest trade
                        continue
                TAL6_ONE[0] = bool(d.get("tal_one_per_side", False))
                try:
                    TAL6_MAXDIR[0] = max(0, int(d.get("tal_max_dir", 0) or 0))
                except (TypeError, ValueError):
                    TAL6_MAXDIR[0] = 0
                TAL6_HOURS_OFF.clear()
                try:
                    TAL6_HOURS_OFF.update(int(h_) for h_ in (d.get("tal_hours_off") or []) if 0 <= int(h_) <= 23)
                except (TypeError, ValueError):
                    TAL6_HOURS_OFF.clear()
                FA.RISK.setdefault("scout", FA.RISK_TAL)
                # a scout strategy's own risk (13.11.2026 night, bot 6's smart agent: "maybe luck" - one chosen out of
                # 96 variants on few trades goes in at a lowered risk, as the manager's star does in bots 1-4):
                # "scout:<combo>" in "risk", over the scout track's risk; gone from the file -> the track's risk again
                for trk in [k_ for k_ in FA.RISK if str(k_).startswith("scout:")]:
                    FA.RISK.pop(trk)
                for trk, v in (d.get("risk") or {}).items():
                    if (trk in FA.RISK or trk in FA.TAL_TRACKS or (str(trk).startswith("scout:") and str(trk)[6:] in SCOUT6)) \
                            and 0 <= float(v) <= 0.02:
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

    def spreads(self, names):
        """FTMO's live bid / ask (bot 1's MetaTrader, GET /quotes; read-only) -> SPREAD5 samples. The first ask only
        registers the symbols with the server; the samples come from the next minute on. Quiet on any failure."""
        try:
            j = requests.get(self.url.rsplit("/", 1)[0] + "/quotes", params={"symbols": ",".join(names)}, timeout=10).json()
            now = time.time()
            for n, q in (j.get("quotes") or {}).items():
                bid, ask, at = float(q[0]), float(q[1]), float(q[2])
                if bid > 0 and ask > bid and now - at < 300 and (ask - bid) / bid < 0.01:     # a fresh, sane quote
                    s = SPREAD5.setdefault(n, [])
                    s.append(round(ask - bid, 6))
                    del s[:-2000]
        except Exception:  # noqa: BLE001
            pass

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
        self.t, self.restricted, self.holidays, self.high = 0, [], [], []

    def refresh(self):
        if time.time() - self.t < 3600:
            return
        try:
            now = pd.Timestamp.now(tz="UTC")
            items = CAL.fetch((now - pd.Timedelta(days=DAYS + 1)).strftime("%Y-%m-%d"), (now + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
            self.restricted, self.holidays = CAL.parse(items)
            # every high-impact release by currency: the gap method's calendar protection (PROT cal15 / cal60), the
            # same "+-N minutes of a high-impact release of the instrument's currency" as the backtest
            self.high = [(pd.Timestamp(i["date"]).tz_convert("UTC"), str(i.get("instrument") or i.get("currency") or "").upper())
                         for i in items if str(i.get("impact") or "").lower() == "high"]
            self.t = time.time()
        except Exception as e:  # noqa: BLE001
            log(f"FTMO calendar not read ({e}); trying again in 10 minutes")
            self.t = time.time() - 3000

    def cal_masks(self, df, name):
        """{"cal15": mask, "cal60": mask} on the 1-minute candles: within +-15 / 60 minutes of a high-impact release."""
        from agents import news_calendar as NC
        cur = set(NC.CURRENCIES.get(name) or [])
        ev = np.array(sorted(pd.Timestamp(t).as_unit("ns").value for t, c in self.high if c in cur or any(x in c for x in cur)), dtype=np.int64)
        t = pd.DatetimeIndex(df.index).as_unit("ns").asi8         # pandas 3 keeps microseconds: compare in ns
        out = {}
        for m in (15, 60):
            w = m * 60 * 10**9
            if not len(ev):
                out[f"cal{m}"] = np.zeros(len(t), bool)
                continue
            i = np.searchsorted(ev, t)
            after = np.where(i < len(ev), ev[np.minimum(i, len(ev) - 1)] - t, 10**18)
            before = np.where(i > 0, t - ev[np.maximum(i - 1, 0)], 10**18)
            out[f"cal{m}"] = (after <= w) | (before <= w)
        return out

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
        # FTMO's rule (user, 14.11.2026): no position over a market break of 2+ hours - known in advance from the same
        # weekday a week earlier (tal_gap.pre_break_mask), so every track closes WKND_MIN minutes before the weekend /
        # a daily close and opens nothing in that window (the break in the data is seen only after it, too late)
        try:
            pre = TG.pre_break_mask(pd.DatetimeIndex(idx).as_unit("ns").asi8, TG.WKND_MIN)
            stop_at = stop_at | pre
            noent = noent | pre
        except Exception as e:  # noqa: BLE001 - the masks as before rather than no masks
            log(f"{name}: the pre-break window not computed ({e})")
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


RULE = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
MINS = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def tal_chart(df, geo, t_in, t_out):
    """The two charts of a gap-method trade (user, 13.11.2026): the high timeframe from before the impulse to after the
    entry (the impulse arrow, candles 1-3, the gap box), the low timeframe from the swings that set the stop and the
    target to after the exit. Returns (high bars, low bars) - at most 120 / 480 candles."""
    hm, lm = MINS.get(geo["htf"], 15), MINS.get(geo["ltf"], 1)
    end_h = max(t_in, pd.Timestamp(geo["c3"])) + pd.Timedelta(minutes=3 * hm)
    st_h = max(pd.Timestamp(geo["imp0"][0]) - pd.Timedelta(minutes=6 * hm), end_h - pd.Timedelta(minutes=120 * hm))
    pts = [pd.Timestamp(geo["c1"])] + [pd.Timestamp(geo[k][0]) for k in ("sw_stop", "sw_tgt") if geo.get(k)]
    end_l = t_out + pd.Timedelta(minutes=10 * lm)
    st_l = max(min(pts) - pd.Timedelta(minutes=5 * lm), end_l - pd.Timedelta(minutes=480 * lm))
    return bars(df, st_h, end_h, RULE.get(geo["htf"], "15min")), bars(df, st_l, end_l, RULE.get(geo["ltf"], "1min"))


def tid(x):
    if x.get("tal"):
        return f"{x['symbol']}|{x['tal_key']}|{x['tal']}"
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


def fut_step(name, df, cst, stop_at, noent, fa, ftrades, fseen, calm=None):
    """Bot 6 (12.11.2026): the user's gap method on 1 hour / 5 minutes (agents/tal_gap.py, card abb187377383) - an
    impulse on the hour chart, the gap between candle 1 and candle 3 that stayed open, a limit at its middle, the stop
    2 ticks beyond the 2nd 5-minute swing beyond the entry, the target at the 2nd swing on the other side, no management.
    With the futures session (no entry from 15:45 New York, everything out at 16:05, nothing over the weekend), sized
    in micro contracts on the virtual FTMO Futures account. It replaced the scalping bot's old gap tracks."""
    shut, late = FA.session_masks(df)
    tick = FA.SPEC[name][2]
    # calm (14.11.2026, bot 6's smart agent): the high-impact calendar masks, as bot 5's gap tracks get them - so a
    # track with PROT cal15 / cal60 or the news guard NEWSX (the user's task tal1411strad) trades live exactly as tested
    # (before this bot 6's live Data had no calendar and those rules were silently off). No track uses them today.
    D = TG.Data(name, df[["o", "h", "l", "c"]], np.full(len(df), tick), np.asarray(stop_at | shut, bool),
                np.asarray(noent | late, bool), 2 * tick, calm or {})
    contract = FA.SPEC[name][0]
    cutoff = (df.index[-1] - pd.Timedelta(days=2)).isoformat()
    for trk, (pair, cfg) in FA.TAL_TRACKS.items():
      gcache = {}
      # FTMO's break rule (the user, 14.11.2026, as bot 5): no waiting gap order over a 2+ hour break (the weekend, a
      # holiday) - cancelled WKND_MIN minutes before it; the trades were already out at 16:05 New York
      for x in TG.live_trades(D, pair, TG.ftmo_cfg(cfg), gcache):
        if x["status"] == "cancelled" and (x.get("time") or x["setup_t"]) < cutoff:
            continue
        k = f"{name}|{x['id_key']}" + ("" if trk == "tal" else f"|{trk}")
        old = ftrades.get(k, {})
        if x["status"] == "cancelled" and old.get("status") in ("open", "closed"):
            # a trade already taken stays as it was (the break rule of 14.11.2026 counts from now on): a closed one is
            # kept, an open one is followed to its stop / target / 16:05 by the loop below (not marked seen)
            if old["status"] == "closed":
                fseen.add(k)
            continue
        fseen.add(k)
        rec = {"id": k, "symbol": name, "contract": contract, "trk": trk, "dir": x["dir"], "status": x["status"],
               "entry": x["entry"], "stop": x.get("stop"), "target": x.get("target"), "rr": x.get("rr"), "cost_r": 0.0,
               "setup_t": x["setup_t"], "time": x.get("time"), "exit_time": x.get("exit_time"), "exit": x.get("exit"),
               "reason": x.get("reason"), "r": x.get("r"), "r_now": x.get("r_now"), "gap": x.get("gap"),
               "geo": x.get("geo") or old.get("geo")}
        if rec["gap"]:
            rec["gap_lo"], rec["gap_hi"] = min(rec["gap"]), max(rec["gap"])
        if rec["geo"] and rec["status"] in ("open", "closed"):
            # the two charts of the method (user, 13.11.2026)
            if rec["status"] != "closed" or old.get("chart_v") != 3 or old.get("status") != "closed":
                t_in = pd.Timestamp(rec["time"]) if rec.get("time") else df.index[-1]
                t_out = pd.Timestamp(rec["exit_time"]) if rec.get("exit_time") else df.index[-1]
                rec["m15"], rec["m1"] = tal_chart(df, rec["geo"], t_in, t_out)
            else:
                rec["m15"], rec["m1"] = old.get("m15"), old.get("m1")
            rec["chart_v"] = 3
        fut_rec(rec, old, fa, ftrades)
    for combo_, sc in SCOUT6.items():          # the scout strategies approved on bot 6's own test (13.11.2026)
        if sc["sym"] != name:
            continue
        try:
            fut_scout(name, df, np.asarray(noent, bool) | np.asarray(late, bool), shut, sc, combo_, contract, tick, fa, ftrades, fseen)
        except Exception:  # noqa: BLE001 - one strategy must not stop the account
            print(f"bot 6 scout {combo_}: {traceback.format_exc()[-300:]}")
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


def fut_scout(name, df, noent, shut, sc, combo, contract, tick, fa, ftrades, fseen):
    """One scout strategy on the live 1-minute bars, exactly as agents/scout_fut.py tests it; trades from the day it
    was built ("since") on. A trade still running at the last bar is open; the account books the closed ones."""
    _, pv, _ = FA.SPEC[name]
    cost_px = tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv
    t = SF.trades(df[["o", "h", "l", "c"]], name, sc["sid"], sc["tf"], sc["v"], np.asarray(noent, bool), np.asarray(shut, bool), cost_px)
    if t.empty:
        return
    since = pd.Timestamp(sc["since"])
    for _, x in t.iterrows():
        tm = pd.Timestamp(x["time"])
        if tm < since:
            continue
        risk = abs(float(x["entry_px"]) - float(x["stop_px"]))
        r_live = float(x["r"]) + float(x["cost_r"]) - tick / risk     # the spread here; commission + slippage in pnl_of
        live_now = x["reason"] == "סוף הנתונים"
        k = f"{name}|{sc['sid']}|{sc['tf']}|{tm.isoformat()}"
        fseen.add(k)
        old = ftrades.get(k, {})
        tg = None if pd.isna(x["target_px"]) else float(x["target_px"])
        rec = {"id": k, "symbol": name, "contract": contract, "trk": "scout", "combo": combo, "dir": int(x["dir"]),
               "status": "open" if live_now else "closed", "entry": float(x["entry_px"]), "stop": float(x["stop_px"]),
               "stop_now": float(x["stop_last"]), "target": tg, "rr": round(abs(tg - float(x["entry_px"])) / risk, 2) if tg else None,
               "cost_r": 0.0, "setup_t": tm.isoformat(), "time": tm.isoformat(),
               "exit_time": None if live_now else pd.Timestamp(x["exit_time"]).isoformat(),
               "exit": None if live_now else float(x["exit_px"]), "reason": None if live_now else x["reason"],
               "r": None if live_now else round(r_live, 4), "r_now": round(r_live, 3) if live_now else None}
        fut_rec(rec, old, fa, ftrades)


def spread5(name, px, sc):
    """The spread a scout strategy of bot 5 pays (price units): the entry's own "spread", else the median of FTMO's live
    quotes sampled here (at least 20), else FTMO's class estimate (agents/ftmo_costs.apply's default: 0.03% of the price)."""
    if sc.get("spread"):
        return float(sc["spread"])
    s = SPREAD5.get(name) or []
    if len(s) >= 20:
        return float(np.median(s))
    return px * C.EST_SPREAD.get(U.CLASS.get(name), 0.0003)


_RR = []


def _run_rules():
    """run_rules.py (news_minute, drop_hole_trades), imported once. Its import adds FTMO's currencies of more indices to
    agents/news_calendar.CURRENCIES (EU50, SPN35, AUS200, N25, US2000) - the gap tracks' calendar protection here reads
    that table, so it is put back as it was: importing for the scout strategies changes nothing in the gap tracks."""
    if not _RR:
        from agents import news_calendar as NC
        keep = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in NC.CURRENCIES.items()}
        import run_rules as RR
        NC.CURRENCIES.clear()
        NC.CURRENCIES.update(keep)
        _RR.append(RR)
    return _RR[0]


def scout5_trades(name, df, sc, cal):
    """Bot 5 (14.11.2026): one scout strategy on the live 1-minute bars, exactly as agent_portfolio.py tested it with
    run_rules.py and the live rules (LIVE_RULES): the strategy's timeframe, no position over a break of 2+ hours (the
    weekend), the holidays (no entry, out before), no entry from 3 minutes before to 3 after a restricted event of the
    instrument and out 3 minutes before it (run_rules.news_minute, here with the live calendar and 1-minute bars), the
    stop to the entry + costs at 60% of the way to the target, FTMO's costs (spread: spread5, commission, swap).
    Signals only on closed bars (rules_agent.run enters at the next bar's open, the forming bar only exits).
    Returns the trades from "since" on, in the records' shape of the gap tracks (tal = "sc:<combo>")."""
    from agents import rules_agent as R
    RR = _run_rules()
    tf, tfm = sc["tf"], SF.TF_MIN[sc["tf"]]
    rule = f"{tfm}min" if tfm < 60 else "1h"
    x1 = df[["o", "h", "l", "c"]]
    d = x1.resample(rule, label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    if len(d) < 300:
        return []
    d["v"] = df["v"].resample(rule, label="left", closed="left").sum().reindex(d.index).fillna(0.0) if "v" in df else 1.0
    d.attrs.update(symbol=name, tf=tf)
    start = d.index.to_series()
    gap = start.shift(-1) - (start + pd.Timedelta(minutes=tfm))
    sess = np.asarray((gap > pd.Timedelta(hours=2)).to_numpy(), dtype=bool)
    ex, ne = CAL.masks(d, name, tfm, cal.restricted, cal.holidays)
    sess = sess | np.asarray(ex, bool)
    fn = CAL.blocked(R.STRATEGIES[sc["sid"]][1], np.asarray(ne, bool))
    px = float(d["c"].median())
    sp = spread5(name, px, sc)
    spec = U.spec(name)
    if spec["commissionType"] == "flat_USD":
        comm = px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
    else:
        comm = px * 2 * float(spec["commission"]) / 100.0
    t = R.run(d, fn, 0.0, breakeven=(0.6, (sp + comm) / px), session_exit=sess)
    if t.empty:
        return []
    ev = pd.DatetimeIndex(sorted({ts for ts, ins in cal.restricted if CAL.restricted_affects(name, ins)}))
    if len(ev):
        t = RR.news_minute(t, d, mb=x1, ev=ev)
    t = RR.drop_hole_trades(t, d, name, tf)
    if t.empty:
        return []
    t = C.apply(t, name, pd.Series([sp], index=d.index[:1]))
    since = pd.Timestamp(sc["since"])
    since = since.tz_localize("UTC") if since.tzinfo is None else since
    combo = f"{sc['sid']}|{tf}|{name}"
    out = []
    for _, x in t.iterrows():
        tm = pd.Timestamp(x["time"])
        if tm < since:
            continue
        e0, st0 = float(x["entry_px"]), float(x["stop_px"])
        risk = abs(e0 - st0)
        if risk <= 0:
            continue
        live_now = x["reason"] == "סוף הנתונים"
        tg = None if pd.isna(x["target_px"]) else float(x["target_px"])
        why = "target" if x["reason"] == "יעד" else "breakeven" if "איזון" in str(x["reason"]) else \
            "stop" if "סטופ" in str(x["reason"]) else "forced"
        out.append({"symbol": name, "dir": int(x["dir"]), "status": "open" if live_now else "closed", "entry": e0,
                    "stop": st0, "target": tg, "rr": round(abs(tg - e0) / risk, 2) if tg else None, "cost_r": 0.0,
                    "leg": None, "leg_start": tm, "leg_top": tm, "gap_lo": e0, "gap_hi": e0, "setup_t": tm, "time": tm,
                    "exit_time": None if live_now else pd.Timestamp(x["exit_time"]),
                    "exit": None if live_now else float(x["exit_px"]), "reason": None if live_now else why,
                    "r": None if live_now else round(float(x["r"]), 4), "r_now": round(float(x["r"]), 3) if live_now else None,
                    "why": None, "tal": f"sc:{combo}", "tal_key": f"{sc['sid']}|{tf}|{tm.strftime('%Y%m%d%H%M')}", "geo": None})
    return out


def tal6_rule(rec, opn):
    """Bot 6's gap tracks: the approved rules against concentrated risk (agents/tal_gap.portfolio, chosen on 2024-2025).
    Returns why the trade is not taken, or None."""
    ts_ = pd.Timestamp(rec["time"])
    if TAL6_HOURS_OFF and (ts_.tz_convert("UTC") if ts_.tzinfo else ts_.tz_localize("UTC")).hour in TAL6_HOURS_OFF:
        return "hours"           # an hour the gap tracks lost in together on 2024-2025
    tal_ = [o for o in opn if o.get("trk") in FA.TAL_TRACKS]
    if TAL6_ONE[0] and any(o["trk"] != rec["trk"] and o["symbol"] == rec["symbol"] and o["dir"] == rec["dir"] for o in tal_):
        return "side"            # another gap track already holds this index and direction
    if TAL6_MAXDIR[0] > 0 and sum(1 for o in tal_ if o["dir"] == rec["dir"]) >= TAL6_MAXDIR[0]:
        return "dir"             # K gap trades already open in this direction on the indices together
    return None


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
            why = tal6_rule(rec, opn) if rec["trk"] in FA.TAL_TRACKS else None
            rk_ = f"scout:{rec.get('combo')}" if rec["trk"] == "scout" and f"scout:{rec.get('combo')}" in FA.RISK else rec["trk"]
            n, why = (0, why) if why else fa.size(name, rk_, rec["entry"], rec["stop"], rec["time"], opn)
            if n:
                rec["n"], rec["risk_usd"] = n, round(n * FA.per_contract(name, rec["entry"], rec["stop"]), 2)
            else:
                rec["skipped"] = why
    if rec.get("n"):
        # notifications (user, 10.11.2026): only bot 6 trades that open and close - each one once, nothing else
        side = "קנייה" if rec["dir"] > 0 else "מכירה"
        tf = {"v5": "15 דק'", "v1m": "דקה"}.get(rec["trk"]) or (rec.get("combo") if rec["trk"] == "scout" else
                                                              f"שיטת הגאפ, {FA.TAL_NAME.get(rec['trk'], rec['trk'])}")
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
    combo = lambda t: (t["combo"] if t.get("trk") == "scout" and t.get("combo") else  # noqa: E731
                       f"{FA.TAL_COMBO[t['trk']]}|{t['symbol']}" if t.get("trk") in FA.TAL_COMBO
                       else f"FUT|{'1Min' if t.get('trk') == 'v1m' else '15Min'}|{t['symbol']}")
    lots = lambda t: f"{t['n']} {t['contract']}"  # noqa: E731
    rs5 = [t["r"] for t in closed if t.get("r") is not None and t["trk"] == "v5"]
    rs1 = [t["r"] for t in closed if t.get("r") is not None and t["trk"] == "v1m"]
    rst = [t["r"] for t in closed if t.get("r") is not None and t["trk"] in FA.TAL_COMBO]
    rules = (f"בוט 6 · חוזים עתידיים מיקרו (MNQ, MES, MYM) · שיטת הגאפ של טל (מ־12.11.2026): דחף בגרף הגבוה, הגאפ בין נר 1 לנר 3 "
             f"שנשאר פתוח, כניסה באמצע הגאפ, סטופ 2 טיקים מעבר לסווינג השני, יעד בסווינג השני בצד השני, בלי ניהול · "
             f"שני מסלולים: שעה / 5 דק' (תוקף 120 שעות) ו־4 שעות / שעה (תוקף 48 שעות) · "
             f"חוקי FTMO Futures, {FA.PLAN}: הפסד מקסימלי נגרר ${FA.MAX_LOSS:,.0f} לפי סוף יום, מגבלה יומית ${FA.DAILY_LOSS:,.0f}, "
             f"יעד ${FA.TARGET:,.0f}, היום הטוב ביותר עד {FA.CONSISTENCY:.0%} מהרווח, עד {FA.MAX_MICROS} חוזי מיקרו · "
             f"אין כניסה אחרי 15:45 שעון ניו יורק, הכול נסגר ב־16:05, לא מחזיקים בסוף שבוע · "
             f"סיכון {FA.RISK_TAL * 100:g}% לעסקה, מעוגל לחוזים שלמים · "
             f"עמלה ${FA.COMM:g} לחוזה הלוך־חזור + טיק החלקה · המחירים מה־CFD של FTMO, לא מהחוזה עצמו")
    return {"bot": FUT_NAME, "label": "בוט 6 · חוזים עתידיים (וירטואלי)", "watch_only": True, "rules": rules,
            "account": f"וירטואלי · חוזים עתידיים · FTMO {FA.PLAN}", "server": "צפייה בלבד · בלי כסף",
            "initial": fa.a["initial"], "risk": FA.RISK_TAL, "balance": round(bal, 2), "equity": round(eq, 2),
            "day_start": st["day_start"], "halted_until": None, "started": fa.a["started"], "withdrawals": [],
            "combos": [f"{c}|{n}" for n in FA.SPEC for c in FA.TAL_COMBO.values()] + sorted(SCOUT6), "equity_curve": ec, "lows": {},
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
            # the charts only on the newest 60 trades (the state stays small for the app)
            "watch": [t if i < 60 else {k_: v_ for k_, v_ in t.items() if k_ not in ("m15", "m1")} for i, t in
                      enumerate(sorted(ftrades.values(), key=lambda t: t.get("time") or t.get("setup_t") or "", reverse=True))],
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
                  "rules": "מ־12.11.2026 החשבון הווירטואלי סופר רק את שיטת הגאפ של טל: דחף בגרף הגבוה, הגאפ בין נר 1 לנר 3 שנשאר פתוח, "
                           "כניסה באמצע הגאפ, סטופ 2 נקודות מעבר לסווינג השני, יעד בסווינג השני בצד השני · 5 צמדים: 15 דק'/דקה (עם הגנה מזעזועים), "
                           "שעה/15 דק', 4 שעות/שעה, 4 שעות/15 דק' (סטופ לנקודת הכניסה כשעברו 70% מהדרך ליעד, בלי כניסה ±15 דק' מפרסום חשוב), 15 דק'/5 דק' · סיכון 0.28% לעסקה (⭐ אולי מזל: x0.7) · המסלולים הישנים מוצגים ולא נספרים: "
                           "דחף ב־15 דקות · כניסה בפער הדקתי בתוך הפער הראשון של המהלך · סטופ 4 פיפס מעבר לתחילת הדחף · "
                           "יעד 2 פיפס לפני התיקון האמיתי האחרון לפני הכניסה · יחס מינימום 1:1 · מסלול פערים בדקה: אותם כללים בגרף הדקה, המתנה של 96 דקות לנגיעה · מסלול פערים (v5): כל פער של נר חזק במדדים (גם JP225), כניסה בנגיעה אחרי שהמהלך פנה, סטופ מתחת לבסיס הפער, יעד בשיא המהלך, 30% יוצאים ב־90% מהדרך והשאר לברייק · מסלול מתוקן (v4): הדחף הוא המהלך הגדול, הפער הראשון שנשאר פתוח, סטופ מעבר לסווינג השלישי, יציאה ב־80% מהדרך ליעד, 12 מכשירים · מסלול רחב: אותם כללים עם נרות חזקים פחות (פי 1.8 מהממוצע במקום 2.5), בערך פי 4 עסקאות · "
                           "איסוף נתונים: נפט, כסף, USDCAD, ‏EURGBP, ‏EU50, ‏AUS200 - הפסידו בבדיקה, מוצגים בנפרד ולא נכנסים לחשבון הווירטואלי"})
    trades = {t["id"]: t for t in (state.get("watch") or [])}
    acc = state.get("vacct") or {}
    trades = gap_id_fix(trades, acc)
    for n_, s_ in (state.get("spreads5") or {}).items():
        SPREAD5[n_] = [float(v) for v in (s_ or [])][-2000:]
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
        if SCOUT5:
            src.spreads(sorted({v["sym"] for v in SCOUT5.values()}))
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
                    try:
                        calm = cal.cal_masks(df, name)
                    except Exception as e:  # noqa: BLE001
                        calm = {}
                        log(f"bot 6 calendar masks failed for {name}: {e}")
                    fut_step(name, df, cst, stop_at, noent, fa, ftrades, fseen, calm=calm)
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
                if name in TAL5_SYMBOLS or any(name in v_ for v_ in TAL5_SYMS.values()):
                    D5 = TG.Data(name, df[["o", "h", "l", "c"]], np.asarray(cst, float), np.asarray(stop_at, bool),
                                 np.asarray(noent, bool), TG.stop_buffer(name), cal.cal_masks(df, name))
                    for trk, (pair, cfg) in TAL5.items():
                        if name not in (TAL5_SYMS.get(trk) or TAL5_SYMBOLS):
                            continue                      # an instrument's own track / the indices' tracks
                        for x in TG.live_trades(D5, pair, TG.ftmo_cfg(cfg)):   # FTMO: closed / cancelled before a 2h+ break
                            g_ = x.get("gap") or [x["entry"], x["entry"]]
                            res.append({"symbol": name, "dir": x["dir"], "status": x["status"], "entry": x["entry"],
                                        "stop": x.get("stop"), "target": x.get("target"), "rr": x.get("rr"), "cost_r": 0.0,
                                        "leg": None, "leg_start": x["setup_t"], "leg_top": x["setup_t"],
                                        "gap_lo": min(g_), "gap_hi": max(g_), "setup_t": x["setup_t"], "time": x.get("time"),
                                        "exit_time": x.get("exit_time"), "exit": x.get("exit"), "reason": x.get("reason"),
                                        "r": x.get("r"), "r_now": x.get("r_now"),
                                        "why": x.get("reason") if x["status"] == "cancelled" else None,
                                        "tal": trk, "tal_key": x["id_key"], "geo": x.get("geo")})
                for combo_, sc_ in list(SCOUT5.items()):
                    if sc_["sym"] != name:
                        continue
                    try:
                        res.extend(scout5_trades(name, df, sc_, cal))
                    except Exception:  # noqa: BLE001 - one strategy must not stop the instrument's other tracks
                        errors.append(f"{name} {combo_}: {traceback.format_exc()[-300:]}")
                cutoff = df.index[-1] - pd.Timedelta(days=2)       # only what happened recently is reported
                for x in res:
                    # ... but a trade the state holds as open / waiting is always followed to its exit (14.11.2026: a
                    # gap trade open for 2+ days - the 4-hour and 1-hour tracks, or any trade over the weekend, whose
                    # exit at Friday's close is known only on Monday - was skipped here and never booked)
                    if pd.Timestamp(x.get("time") or x.get("setup_t")) < cutoff \
                            and (trades.get(tid(x)) or {}).get("status") not in ("open", "pending"):
                        continue
                    k = tid(x)
                    seen_now.add(k)
                    old = trades.get(k, {})
                    rec = {"id": k, "symbol": name, "broker": src.broker.get(name), "dir": int(x["dir"]), "status": x["status"],
                           "entry": float(x["entry"]), "stop": float(x["stop"]) if x.get("stop") is not None else None,
                           "target": float(x["target"]) if x.get("target") else None,
                           "rr": x.get("rr"), "cost_r": x.get("cost_r"), "leg": x.get("leg"), "tal": x.get("tal"), "trk": x.get("tal"),
                           "leg_start": iso(x["leg_start"]), "leg_top": iso(x["leg_top"]), "gap_lo": float(x["gap_lo"]), "gap_hi": float(x["gap_hi"]),
                           "setup_t": iso(x.get("setup_t")) or old.get("setup_t"), "time": iso(x.get("time")), "exit_time": iso(x.get("exit_time")),
                           "exit": float(x["exit"]) if x.get("exit") is not None else None, "reason": x.get("reason"),
                           "r": x.get("r"), "r_now": x.get("r_now"), "mark": old.get("mark"), "wide": bool(x.get("wide")),
                           "why": x.get("why"), "touch_t": iso(x.get("touch_t")),
                           "extra": name in EXTRA and not x.get("v4"), "v4": bool(x.get("v4")), "v5": bool(x.get("v5")), "v1m": bool(x.get("v1m")),
                           "geo": x.get("geo") or old.get("geo")}
                    if rec["geo"] and rec["status"] in ("open", "closed"):
                        # gap-method trades: the two charts of the method (high / low timeframe), user 13.11.2026
                        if x["status"] != "closed" or old.get("chart_v") != 3 or old.get("status") != "closed":
                            t_in = pd.Timestamp(x["time"]) if x.get("time") else df.index[-1]
                            t_out = pd.Timestamp(x["exit_time"]) if x.get("exit_time") else df.index[-1]
                            rec["m15"], rec["m1"] = tal_chart(df, rec["geo"], t_in, t_out)
                            rec["chart_v"] = 3
                        else:
                            rec["m15"], rec["m1"], rec["chart_v"] = old.get("m15"), old.get("m1"), 3
                    elif x["status"] != "closed" or "m15" not in old or old.get("status") != x["status"] or old.get("chart_v") != 2:
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
                        if not rec.get("tal"):
                            # only the user's gap method counts since 12.11.2026 (the old tracks are shown, not counted)
                            rec["skipped"] = "v4" if rec["v4"] else "extra" if rec["extra"] else "old"
                        elif rec["status"] in ("open", "closed") and rec["time"] and rec["time"] < acc["started"]:
                            rec["skipped"] = "before"                                       # before the account opened
                    if rec["status"] in ("open", "closed") and "risk_usd" not in rec and "skipped" not in rec:
                        bal, day = balance(), prague_day(rec["time"])
                        start = bal - day_pnl(day)
                        # the gap tracks' own rules (chosen on the gap tracks together) - not on a scout strategy
                        gap_ = not scout5_combo(rec["tal"])
                        side_ = gap_ and TAL5_ONE[0] and any(o.get("tal") and not scout5_combo(o.get("tal")) and o.get("tal") != rec["tal"]
                                                             and o.get("risk_usd") and o.get("status") == "open"
                                                             and o.get("symbol") == rec["symbol"] and o.get("dir") == rec["dir"]
                                                             for o in trades.values())
                        dir_ = gap_ and TAL5_MAXDIR[0] > 0 and sum(1 for o in trades.values() if o.get("tal") and not scout5_combo(o.get("tal"))
                                                                   and o.get("risk_usd") and o.get("status") == "open"
                                                                   and o.get("dir") == rec["dir"] and o.get("id", o) != rec.get("id")) >= TAL5_MAXDIR[0]
                        ts_ = pd.Timestamp(rec["time"])
                        hour_ = gap_ and bool(TAL5_HOURS_OFF) and (ts_.tz_convert("UTC") if ts_.tzinfo else ts_).hour in TAL5_HOURS_OFF
                        if acc.get("breached"):
                            rec["skipped"] = "breach"
                        elif hour_:
                            rec["skipped"] = "hours"         # an hour the tracks lost in together on 2024-2025
                        elif side_:
                            rec["skipped"] = "side"          # another track already holds this symbol and direction
                        elif dir_:
                            rec["skipped"] = "dir"           # K trades already open in this direction on the indices
                        elif day_pnl(day) <= -DAILY_STOP * start:
                            rec["skipped"] = "daily"
                        else:
                            # the red lines hold for the open trades too (bot 5's smart agent, 13.11.2026 evening; the
                            # manager's task f8643b16aa57: 23 gap tracks x 0.2-0.28% = 5.4% that can be open at once,
                            # more than the 5% daily line): a new trade risks at most the room left - the day's loss so
                            # far plus every open trade's full risk stays within the bot's 4.5% a day and 9% in total.
                            # Near a limit the risk shrinks (Tal: lower gradually, not block); under a quarter of the
                            # trade's risk left -> not taken.
                            # tal_risk_scale is the gap tracks' factor; a scout strategy has its own risk
                            want_ = bal * TAL5_RISK.get(rec["tal"], RISK_TAL5) * (TAL5_SCALE[0] if gap_ else 1.0)
                            open_ = sum(float(o.get("risk_usd") or 0) for kk_, o in trades.items()
                                        if kk_ != k and o.get("status") == "open" and o.get("risk_usd"))
                            room_ = min(DAILY_STOP * start + day_pnl(day) - open_,
                                        bal - open_ - acc["initial"] * (1 - BOT_MAX_LOSS))
                            if room_ < 0.25 * want_:
                                rec["skipped"] = "room"
                            else:
                                rec["risk_usd"] = round(min(want_, room_), 2)
                    if rec.get("risk_usd"):
                        if rec["status"] == "closed" and rec.get("r") is not None:
                            rec["pnl"] = round(rec["r"] * rec["risk_usd"], 2)
                            if not any(x["id"] == k for x in acc["ledger"]):
                                acc["ledger"].append({"id": k, "exit_time": rec["exit_time"], "pnl": rec["pnl"]})
                                if balance() <= acc["initial"] * (1 - MAX_LOSS) and not acc.get("breached"):
                                    acc["breached"] = rec["exit_time"]
                                    REP.notify("בוט 5 - סקאלפינג: החשבון הווירטואלי נשרף", f"היתרה ירדה ל־${balance():,.0f} (10% הפסד). לא נפתחות עוד עסקאות וירטואליות.",
                                               tags=["eyes"], key="wbreach", every=86400 * 30)
                        elif rec["status"] == "open" and rec.get("r_now") is not None:
                            rec["pnl_now"] = round(rec["r_now"] * rec["risk_usd"], 2)
                    trades[k] = rec
                    side = ("קנייה" if rec["dir"] > 0 else "מכירה") + (f" · אסטרטגיה {tal5_name(rec['tal'])}" if scout5_combo(rec.get("tal")) else
                                                                      f" · שיטת הגאפ {tal5_name(rec['tal'])}" if rec.get("tal") else " · פערים בדקה" if rec["v1m"] else " · פערים (v5)" if rec["v5"] else " · מתוקן (v4)" if rec["v4"] else " · מסלול רחב" if rec["wide"] else "") + (" · איסוף נתונים בלבד" if rec["extra"] else "")
                    # notifications (user, 10.11.2026: "I'm bombarded"): only trades of the virtual account - opened and closed.
                    # Waiting setups and touched-but-skipped ones stay in the app, without a push.
                    quiet = not rec.get("risk_usd")
                    if False and rec["status"] == "cancelled" and rec.get("why") and prev != "cancelled":
                        why = {"rr": f"היעד קרוב מדי (יחס 1:{rec.get('rr')})", "cost": f"העלות {round(100 * (rec.get('cost_r') or 0))}% מהסיכון"}.get(rec["why"], rec["why"])
                        REP.notify(f"בוט 5 - סקאלפינג: {name} נגע ולא נכנס", f"{side} · נגיעה ב־{fmt(name, rec['entry'])} · לא נכנס: {why}",
                                   tags=["eyes"], key=f"wx:{k}", every=86400)
                    if False and prev is None and rec["status"] == "pending":
                        REP.notify(f"בוט 5 - סקאלפינג: {name} מחכה לכניסה", f"{side} · דחף של {rec['leg']} נרות · כניסה בנגיעה ב־{fmt(name, rec['entry'])}\n"
                                   f"סטופ {fmt(name, rec['stop'])} · לא נשלחת פקודה", tags=["eyes"], key=f"wp:{k}", every=86400)
                    if not quiet and prev in (None, "pending") and rec["status"] in ("open", "closed"):
                        REP.notify(f"בוט 5 - סקאלפינג: {name} עסקה וירטואלית נפתחה", f"{side} @ {fmt(name, rec['entry'])} · סטופ {fmt(name, rec['stop'])} · "
                                   f"יעד {fmt(name, rec['target'])} · יחס 1:{rec['rr']}\nפתח את האפליקציה וסמן אם היית לוקח אותה",
                                   tags=["eyes"], key=f"wo:{k}", every=86400)
                    if not quiet and prev != "closed" and rec["status"] == "closed":
                        word = {"target": "ביעד", "stop": "בסטופ", "forced": "בסגירה כפויה", "breakeven": "בברייק אחרי יציאה חלקית"}.get(rec["reason"], "")
                        usd = f" · {'+' if rec.get('pnl', 0) >= 0 else '−'}${abs(rec.get('pnl', 0)):,.0f} · יתרה ${balance():,.0f}" if rec.get("pnl") is not None else ""
                        REP.notify(f"בוט 5 - סקאלפינג: {name} נסגרה {word}", f"{side} · {'+' if (rec['r'] or 0) >= 0 else ''}{rec['r']:.2f} סיכון{usd} (וירטואלי)",
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
                                REP.notify(f"בוט 5 - סקאלפינג: {name} נסגרה {word}", f"{'+' if t['r'] >= 0 else ''}{t['r']:.2f} סיכון · "
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
                      "instruments": {k: src.broker.get(k) for k in SYMBOLS}, "errors": errors[-5:],
                      "spreads5": {n_: s_[-500:] for n_, s_ in SPREAD5.items()},
                      "scout5": {c_: {"spread": round(spread5(v_["sym"], 1.0, v_), 6) if (v_.get("spread") or len(SPREAD5.get(v_["sym"]) or []) >= 20) else None,
                                      "samples": len(SPREAD5.get(v_["sym"]) or []), "since": v_["since"]} for c_, v_ in SCOUT5.items()}})
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
        combo = lambda n, t=None: (scout5_combo(t["tal"]) if scout5_combo((t or {}).get("tal"))  # noqa: E731
                                   else f"TAL|{tal5_tf(t['tal'])}|{n}" if (t or {}).get("tal")
                                   else f"OWN|{'1Min' if (t or {}).get('v1m') else '15Min'}|{n}")
        counted = [t for t in closed if t.get("risk_usd") and t.get("pnl") is not None]
        state.update({"account": "וירטואלי · שיטת הגאפ של טל", "server": f"צפייה בלבד · בלי כסף · סיכון {RISK_TAL5 * 100:.2f}% לעסקה", "initial": acc["initial"], "risk": RISK_TAL5,
                      "balance": round(bal, 2), "equity": round(eq, 2), "day_start": round(bal - day_pnl(today), 2),
                      "halted_until": None, "combos": [f"TAL|{tal5_tf(k)}|{n}" for k in TAL5 for n in (TAL5_SYMS.get(k) or TAL5_SYMBOLS)] + sorted(SCOUT5), "started": acc["started"], "withdrawals": [],
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
