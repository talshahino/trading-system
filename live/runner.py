"""Live demo robot: runs one portfolio of (strategy, timeframe, instrument) combos on one MT5
terminal, with exactly the code that was backtested (agents/rules_agent.py).

How it decides (every 5 s it checks for newly closed bars):
  1. fetch the combo's bars from MT5, convert to UTC like the backtest (4H = UTC-aligned from 1H),
     append the bar that just opened with open=high=low=close (nothing from the future)
  2. replay the strategy with the backtest engine (same breakeven at 60%; no-target strategies: entry at 2.5R, +1R at 3R; stocks closed at session end; with close_before_breaks, every position closed before market breaks over 2 hours)
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
import os
import datetime as dt
import json
import re
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
from agents import ftmo_calendar as CAL  # noqa: E402

import MetaTrader5 as mt5  # noqa: E402  (pip install MetaTrader5, Windows only)
from live import reporter as REP  # noqa: E402
from live import tunnel as TUN  # noqa: E402
from live import updater as UPD  # noqa: E402
from collections import deque  # noqa: E402

JOURNAL = deque(maxlen=300)
TF_HE = {"1Min": "דקה", "5Min": "5 דק׳", "15Min": "15 דק׳", "1Hour": "שעה", "4Hour": "4 שעות"}


def pretty(combo):
    sid, tf, sym = combo.split("|")
    return f"{sid} · {TF_HE.get(tf, tf)} · {sym}"

SYM_BY_MAGIC = {}
TF_MIN = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "4Hour": 240}
MT5_TF = {"1Min": mt5.TIMEFRAME_M1, "5Min": mt5.TIMEFRAME_M5, "15Min": mt5.TIMEFRAME_M15,
          "1Hour": mt5.TIMEFRAME_H1, "4Hour": mt5.TIMEFRAME_H1}
BARS = {"1Min": 3000, "5Min": 3000, "15Min": 3000, "1Hour": 3000, "4Hour": 8000}
SESSION = {"US": ("America/New_York", 16 * 60), "EU": ("Europe/Berlin", 17 * 60 + 30)}
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    if sys.stdout is not None:          # pythonw (hidden, no window) has no console to print to
        print(line, flush=True)
    JOURNAL.append(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if msg.startswith("skip "):
        note_skip(msg)


def note_skip(msg):
    """Every skipped signal goes into the state file too (07.11.2026, approved by the user: logging only), so the
    daily demo-vs-backtest check knows why a backtest trade is missing. The same reason for the same combo is
    kept once per 30 minutes; the newest 300 are published."""
    try:
        st = globals().get("STATE")
        if not isinstance(st, dict):
            return
        combo, _, why = msg[5:].partition(": ")
        now = pd.Timestamp.now(tz="UTC")
        key = re.sub(r"[0-9.]+", "#", why)
        sk = st.setdefault("skips", [])
        for x in reversed(sk[-50:]):
            if x.get("combo") == combo and x.get("key") == key and now - pd.Timestamp(x["t"]) < pd.Timedelta(minutes=30):
                return
        sk.append({"t": now.isoformat(timespec="seconds"), "combo": combo, "why": why[:200], "key": key})
        del sk[:-300]
    except Exception:  # noqa: BLE001 - logging must never stop the bot
        pass


SYMMAP = {}          # instrument -> this server's symbol name, when it differs from FTMO's (see live/symbols.py)


def mt5_name(sym):
    return SYMMAP.get(sym) or U.ALL[sym][0].replace("/", "")


def map_symbol(sym):
    """FTMO's name if this server has it; otherwise the same instrument under another name (another broker's demo)."""
    name = U.ALL[sym][0].replace("/", "")
    info = mt5.symbol_info(name)
    if info is not None and info.trade_mode != 0:
        return name
    from live import symbols as SY
    hit = SY.resolve(sym, U.ALL[sym][0], mt5.symbols_get() or [])
    if hit is None:
        return None
    SYMMAP[sym] = hit.name
    return hit.name


# ---------------- time ----------------
def server_offset_hours(probe):
    tick = mt5.symbol_info_tick(probe)
    if tick is None:
        return 0
    return int(round((tick.time - time.time()) / 3600.0))


SERVER_CLOCK = {"tz": "NY+7"}     # FTMO: New York + 7 h. MetaQuotes: Eastern European time (EU daylight rules)


def pick_server_clock(server, probe):
    """The two clocks differ only in the 3-4 weeks a year when the US and the EU change clocks on
    different dates. Default by server name; when the last tick is fresh, the measured offset decides."""
    tz = "Europe/Athens" if "metaquotes" in str(server).lower() else "NY+7"
    try:
        tick = mt5.symbol_info_tick(probe)
        now = pd.Timestamp.now(tz="UTC").floor("s")
        if tick is not None and abs(tick.time - time.time() - 3600 * round((tick.time - time.time()) / 3600)) < 120:
            off = round((tick.time - time.time()) / 3600)
            ny7 = round((now.tz_convert("America/New_York").utcoffset().total_seconds() + 7 * 3600) / 3600)
            eet = round(now.tz_convert("Europe/Athens").utcoffset().total_seconds() / 3600)
            if ny7 != eet and off in (ny7, eet):
                tz = "NY+7" if off == ny7 else "Europe/Athens"
    except Exception:  # noqa: BLE001
        pass
    SERVER_CLOCK["tz"] = tz
    return tz


def server_to_utc(seconds):
    """MT5 bar/deal times are the server's wall clock. FTMO's server is New York time + 7 hours (GMT+2 in
    winter, GMT+3 in summer, US daylight dates); MetaQuotes' demo is Eastern European time (EU dates).
    Every time is converted with the rule of its own date. (The offset can't be read from the last tick
    alone: on weekends that tick is Friday's.)"""
    naive = pd.to_datetime(pd.Series(seconds).to_numpy(), unit="s")
    if SERVER_CLOCK["tz"] == "NY+7":
        naive = naive - pd.Timedelta(hours=7)
        tz = "America/New_York"
    else:
        tz = SERVER_CLOCK["tz"]
    loc = pd.DatetimeIndex(naive).tz_localize(tz, ambiguous=np.zeros(len(naive), bool), nonexistent="shift_forward")
    return loc.tz_convert("UTC")


def utc_to_server(ts):
    """A UTC moment as the server clock shows it, in the form MT5's history functions expect."""
    if SERVER_CLOCK["tz"] == "NY+7":
        t = pd.Timestamp(ts).tz_convert("America/New_York").tz_localize(None) + pd.Timedelta(hours=7)
    else:
        t = pd.Timestamp(ts).tz_convert(SERVER_CLOCK["tz"]).tz_localize(None)
    return t.tz_localize("UTC").to_pydatetime()


def server_ts(seconds):
    return server_to_utc([seconds])[0]


def bars(sym, tf, off):
    rates = mt5.copy_rates_from_pos(mt5_name(sym), MT5_TF[tf], 0, BARS[tf])
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df.index = server_to_utc(df["time"])
    df = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "tick_volume": "v"})[["o", "h", "l", "c", "v"]]
    if U.CLASS.get(sym) == "stock" and not U.DUKA.get(sym):
        # the backtest of these stocks uses Alpaca bars of the regular session only, labelled by their start:
        # the 09:00 hour bar (09:30-10:00) does not exist there, so it must not exist here either
        ny = df.index.tz_convert("America/New_York")
        m = ny.hour * 60 + ny.minute
        df = df[(m >= 570) & (m < 960)]
        if len(df) < 50:
            return None
    if tf == "4Hour":
        df = df.resample("4h", label="left", closed="left").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    return df


_PARTNER_OK = {}


def partner_bars(sym, tf, start, end):
    """Bars of a related instrument for the inter-market strategies (agents/intermarket_strategies), from MT5."""
    if sym not in _PARTNER_OK:
        name = map_symbol(sym)
        if name is None:                        # only its prices are needed: a symbol closed for trading will do
            from live import symbols as SY
            hit = SY.resolve(sym, U.ALL[sym][0], mt5.symbols_get() or [], tradable_only=False)
            if hit is not None:
                name = SYMMAP[sym] = hit.name
        _PARTNER_OK[sym] = bool(name and mt5.symbol_select(name, True))
        if not _PARTNER_OK[sym]:
            log(f"WARNING: partner {sym} is not available on this server - its inter-market combos give no signals")
    if not _PARTNER_OK[sym]:
        return None
    return bars(sym, tf, None)


def session_mask(df, sym, tf):
    if U.CLASS.get(sym) != "stock":
        return None
    tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[sym] else SESSION["EU"]
    loc = df.index.tz_convert(tz)
    end = loc + pd.Timedelta(minutes=TF_MIN[tf])
    mins = end.hour * 60 + end.minute + (end.normalize() > loc.normalize()) * 1440
    day = pd.Series(loc.normalize(), index=df.index)
    return (mins >= close_min - 15) | (day != day.shift(-1)).to_numpy()


CLOSE_BREAKS = False          # FTMO Standard account: no positions through market breaks over 2 hours
BREAKS = {}                   # symbol -> {(server weekday, server minute of day)} where such a break starts
_BREAKS_AT = {}


def break_mask(df, tf):
    """Same rule as the backtest (run_rules.break_mask): the last bar before a gap over 2 hours."""
    start = df.index.to_series()
    gap = start.shift(-1) - (start + pd.Timedelta(minutes=TF_MIN[tf]))
    return (gap > pd.Timedelta(hours=2)).to_numpy()


def exit_mask(df, sym, tf):
    m = session_mask(df, sym, tf)
    if CLOSE_BREAKS:
        b = break_mask(df, tf) | cal_masks(df, sym, tf)[0]
        m = b if m is None else (np.asarray(m, dtype=bool) | b)
    return m


def cal_masks(df, sym, tf):
    """FTMO restricted events + holidays, the same masks the backtest uses (--news-exit)."""
    c = _calendar()
    return CAL.masks(df, sym, TF_MIN[tf], c["restricted"], c["holidays"])


# Strategies with no target (user, 10.11.2026): once 2.5R in profit the stop goes to the entry (+ costs), once 3R in
# profit to entry + 1R. Tested on the 50 live no-target combos (runs 798/809/812/813): last 1.5 years +1,312R vs
# +1,311R, drawdowns 438R vs 445R, trades that reached 2.5R and still lost 4 instead of 42. Strategies with a target
# keep the 60%-of-the-way breakeven; the engine applies both only when the trade has no target.
PROTECT = {"be_r": 2.5, "lock_r": (3.0, 1.0)}


# Approved per-combo changes from the improvement agents' cards (live/bots/<bot>.json "plans"; user, 11-12.11.2026),
# exactly the variants the agents tested (agent_portfolio.py -> run_rules --optimize / --filters):
#   target   "none" = no target (the no-target protection above applies) | k = k x the stop distance (the "kR" variants)
#   stop     x the strategy's stop distance (the same dollar risk: the lots follow the stop)
#   limit    a limit order k R better than the next open, valid 3 bars (run_rules OPT_ENTRY)
#   filter   a market-regime filter of run_rules._regimes (only with the 200 trend, ADX, volatility)
PLANS = {}
LIMIT_BARS = 3


def plan_of(combo):
    p = PLANS.get(combo) or {}
    out = {}
    try:
        if p.get("target") == "none":
            out["target"] = None
        elif p.get("target") is not None and 0.5 <= float(p["target"]) <= 5:
            out["target"] = ("R", float(p["target"]))
        if p.get("stop") is not None and 0.5 <= float(p["stop"]) <= 2 and float(p["stop"]) != 1.0:
            out["stop_scale"] = float(p["stop"])
        if p.get("limit") is not None and 0 < float(p["limit"]) <= 1:
            out["entry_pullback"] = (float(p["limit"]), LIMIT_BARS)
            out["emit_pending"] = 1                    # step() adds one flat helper bar after the forming bar
        if p.get("filter") in REGIMES:
            out["filter"] = p["filter"]
    except (TypeError, ValueError):
        return {}
    return out


def _regime(df, name):
    """The same masks as run_rules._regimes (known at the bar's close): (allow_long, allow_short)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    if name == "with_trend200":
        e200 = R.ema(c, 200)
        return c > e200, c < e200
    if name in ("adx_trend", "adx_range"):
        a = R.adx(h, l, c, 14)
        m = a > 20 if name == "adx_trend" else a < 20
        return m, m
    at = R.atr(h, l, c, 14) / c
    pr = pd.Series(at).rolling(500, min_periods=100).rank(pct=True).to_numpy()
    m = {"vol_normal": (pr > 0.2) & (pr < 0.9), "vol_high": pr >= 0.5, "vol_low": pr < 0.5}[name]
    return m, m


REGIMES = ("with_trend200", "adx_trend", "adx_range", "vol_normal", "vol_high", "vol_low")


def strat(sid, df, sym, tf):
    fn = R.STRATEGIES[sid][1]
    fn = CAL.blocked(fn, cal_masks(df, sym, tf)[1]) if CLOSE_BREAKS else fn
    pl = plan_of(f"{sid}|{tf}|{sym}")
    flt = pl.pop("filter", None)

    def g(d):
        L, S, plan = fn(d)
        if flt:
            al, ash = _regime(d, flt)
            L, S = np.asarray(L, bool) & al, np.asarray(S, bool) & ash
        return L, S, {**plan, **pl, **PROTECT}
    return g


def learn_breaks(sym):
    """When this instrument's breaks over 2 hours start, in server time, from ~5 weeks of MT5
    1-minute bars. A break must have been seen at least twice (a one-off holiday is not a pattern).
    Start minutes are rounded down to a quarter hour, so the close comes 15-30 minutes before."""
    rates = mt5.copy_rates_from_pos(mt5_name(sym), mt5.TIMEFRAME_M1, 0, 60000)
    if rates is None or len(rates) < 1000:
        return BREAKS.get(sym, set())
    t = pd.to_datetime(pd.Series(rates["time"]), unit="s")
    nxt = t.shift(-1)
    starts = (t + pd.Timedelta(minutes=1))[(nxt - t) > pd.Timedelta(hours=2)]
    keys = pd.Series([(x.weekday(), (x.hour * 60 + x.minute) // 15 * 15) for x in starts])
    cnt = keys.value_counts()
    return {k for k, n in cnt.items() if n >= 2}


def break_soon(sym, now, minutes=15):
    """True when a market break over 2 hours starts within `minutes` (server-time schedule)."""
    if not CLOSE_BREAKS:
        return False
    if sym not in BREAKS or time.time() - _BREAKS_AT.get(sym, 0) > 6 * 3600:
        try:
            BREAKS[sym] = learn_breaks(sym)
        except Exception as e:  # noqa: BLE001
            log(f"break schedule {sym}: {e}")
            BREAKS.setdefault(sym, set())
        _BREAKS_AT[sym] = time.time()
    srv = pd.Timestamp(utc_to_server(now)).tz_localize(None)
    mod = srv.weekday() * 1440 + srv.hour * 60 + srv.minute
    for wd, m in BREAKS[sym]:
        ahead = (wd * 1440 + m - mod + 15) % (7 * 1440) - 15   # m is rounded down: the real start is m..m+15
        if -15 < ahead <= minutes:
            return True
    return False


_CAL = {"t": 0, "restricted": [], "holidays": [], "warned": False}


def _calendar():
    """FTMO's calendar (restricted events + holidays), refreshed every hour."""
    if time.time() - _CAL["t"] > 3600:
        now = pd.Timestamp.now(tz="UTC")
        try:
            try:      # ~200 days back, so the replay knows the past events too; else just the coming days
                items = CAL.fetch(now - pd.Timedelta(days=200), now + pd.Timedelta(days=9))
            except Exception:  # noqa: BLE001
                items = CAL.fetch(now - pd.Timedelta(days=2), now + pd.Timedelta(days=9))
            _CAL["restricted"], _CAL["holidays"] = CAL.parse(items)
            _CAL["warned"] = False
        except Exception as e:  # noqa: BLE001
            log(f"FTMO calendar not available ({e}); trying again in 10 minutes")
            if not _CAL["warned"]:
                REP.notify(f"{BOT}: אין יומן FTMO", "לא הצלחתי לקרוא את יומן האירועים של FTMO. אנסה שוב כל 10 דקות.",
                           priority=4, tags=["warning"])
                _CAL["warned"] = True
            _CAL["t"] = time.time() - 3000
            return _CAL
        _CAL["t"] = time.time()
    return _CAL


def news_window(sym, now=None):
    """The user's rule around FTMO restricted events on this instrument: no open position and no new
    trade from 3 minutes before to 3 minutes after the release.
    "close": 3 to 2 minutes before - close what is open (FTMO's own ban starts 2 minutes before);
    "hold":  2 minutes before to 3 minutes after - no opening, closing or stop moves; None otherwise."""
    if not CLOSE_BREAKS:
        return None
    now = now or pd.Timestamp.now(tz="UTC")
    phase = None
    for t, ins in _calendar()["restricted"]:
        d = (now - t).total_seconds()
        if -180 <= d <= 180 and CAL.restricted_affects(sym, ins):
            if d < -120:
                return "close"
            phase = "hold"
    return phase


NEWS_RETRY = set()            # combos waiting for a restricted-event window to end


def holiday_block(sym, now=None, before_min=30):
    """Standard rule: the instrument's market is closed for a holiday today (UTC), or will be within
    `before_min` minutes. Returns the holiday name or None."""
    if not CLOSE_BREAKS:
        return None
    now = now or pd.Timestamp.now(tz="UTC")
    hol = _calendar()["holidays"]
    for day in CAL.closed_days(sym, hol):
        if day - pd.Timedelta(minutes=before_min) <= now < day + pd.Timedelta(days=1):
            return ", ".join(sorted({f"{t} ({c})" for d, c, t in hol if d == day and "Daylight" not in t}))
    return None


def break_close(now):
    """Standard account: close every position 15-30 minutes before a break over 2 hours (weekends too)."""
    if not CLOSE_BREAKS:
        return
    for p in ACC.positions():
        sym = SYM_BY_MAGIC.get(p.magic)
        if sym is None:
            continue
        nw = news_window(sym, now)
        if nw == "close":
            close(p, "restricted news ahead (Standard rule)")
            continue
        if nw == "hold":
            continue                                   # FTMO's ban window: no closing now
        if break_soon(sym, now):
            close(p, "market break ahead (Standard rule)")
        elif (h := holiday_block(sym, now)):
            close(p, f"holiday ahead: {h} (Standard rule)")


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


# ---------------- per-combo risk (approved by the user in the app: "ניהול בוטים") ----------------
RISK_FILE = ROOT / "live" / "risk.json"      # written only by the manager after the user approves; never in git
_RM = {"mtime": None, "bot": {}}


def bot_base():
    return os.environ.get("BOT_BASE") or str(MON["name"]).split("_")[0]


def _risk_section():
    try:
        m = RISK_FILE.stat().st_mtime
        if m != _RM["mtime"]:
            d = json.loads(RISK_FILE.read_text(encoding="utf-8"))
            _RM.update(mtime=m, bot=d.get(bot_base()) or {})
    except FileNotFoundError:
        _RM.update(mtime=None, bot={})
    except Exception as e:  # noqa: BLE001
        log(f"risk.json unreadable, keeping the last values: {e}")
    return _RM["bot"]


def risk_mult(combo):
    """The combo's risk multiplier: 1 = the bot's risk; 0.5..1.5 after an approved change; 0 = no new trades
    (removal approved). Unknown or unreadable -> 1, so a missing file never changes anything."""
    try:
        v = float((_risk_section().get("mult") or {}).get(combo, 1.0))
    except (TypeError, ValueError):
        return 1.0
    return v if v == 0 or 0.5 <= v <= 1.5 else 1.0


def combo_rules(combo):
    """The other approved changes of a combo (each checked; anything odd is ignored):
    dir 1/-1 = buy only / sell only; tp f = target at f x its distance; sl f = stop at f x its distance (same
    dollar risk); hours [a, b) = no new trades in that Israel-time block; vol {max_pct|min_pct} = no new trades
    when the stop is farther / nearer than that % of the price."""
    s, out = _risk_section(), {}
    try:
        if (v := (s.get("dir") or {}).get(combo)) in (1, -1):
            out["dir"] = v
        if (v := (s.get("tp") or {}).get(combo)) is not None and 0.3 <= float(v) < 1:
            out["tp"] = float(v)
        if (v := (s.get("sl") or {}).get(combo)) is not None and 0.5 <= float(v) < 1:
            out["sl"] = float(v)
        if (v := (s.get("hours") or {}).get(combo)) and len(v) == 2 and 0 <= int(v[0]) < int(v[1]) <= 24:
            out["hours"] = [int(v[0]), int(v[1])]
        if (v := (s.get("vol") or {}).get(combo)) and isinstance(v, dict) and ({"max_pct", "min_pct"} & set(v)):
            out["vol"] = {k: float(x) for k, x in v.items() if k in ("max_pct", "min_pct")}
    except (TypeError, ValueError):
        pass
    return out


# ---------------- account ----------------
GUARD_DAILY, GUARD_TOTAL = 0.045, 0.09          # the bot's own loss limits (FTMO's are 5% / 10%)
GRADE_SHARE, GRADE_MIN = 0.5, 0.10              # near a limit: risk = half the room left; skip under 10% of normal


class Account:
    def __init__(self, cfg, dry):
        self.cfg, self.dry = cfg, dry
        self.initial = cfg["initial"]
        self.risk_usd = self.initial * cfg["risk"]
        self.free = cfg["free_risk_at_breakeven"]
        self.halt_until = None
        self._ds = None              # (Prague midnight, day-start balance): the last good reading
        self._halt_retry = 0.0

    def positions(self):
        return [p for p in (mt5.positions_get() or []) if p.magic in MAGICS]

    def day_start_balance(self):
        since = utc_to_server(prague_midnight_utc())
        deals = mt5.history_deals_get(since, utc_to_server(pd.Timestamp.now(tz="UTC")) + dt.timedelta(days=1))
        if deals is None:
            # no answer from MT5 is NOT "no deals today": an empty list would hide today's realized loss from the
            # guard and the emergency stop. Use today's last good value, or stop this loop (MT5 is re-initialized).
            if self._ds and self._ds[0] == since:
                return self._ds[1]
            raise RuntimeError(f"history_deals_get failed: {mt5.last_error()}")
        # everything that changed the balance since midnight: closed profit/loss, and the commission/swap/fees
        # of every trade deal, also of positions opened today that are still open (FTMO: the day's limit is
        # 5% of the account size below the balance at the start of the day)
        realized = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in deals if d.type in (0, 1))
        v = mt5.account_info().balance - realized
        self._ds = (since, v)
        return v

    def open_risk(self):
        tot = 0.0
        for p in self.positions():
            if self.free and p.sl and ((p.type == 0 and p.sl >= p.price_open) or (p.type == 1 and p.sl <= p.price_open)):
                continue
            # what this position really loses at its stop now (MetaTrader's calculation); a position without a
            # stop, or when the calculation fails, counts with the larger of the stored risk and the bot's risk
            loss = None
            if p.sl:
                try:
                    v = mt5.order_calc_profit(p.type, p.symbol, p.volume, p.price_open, p.sl)
                    loss = max(0.0, -v) if v is not None else None
                except Exception:  # noqa: BLE001
                    loss = None
            tot += loss if loss is not None else max(STATE["risk"].get(str(p.ticket), 0.0), self.risk_usd)
        held = {p.magic for p in self.positions()}
        for combo, pk in STATE.get("pending", {}).items():   # a waiting limit order counts as if filled and stopped
            if MAGICS_BY_COMBO.get(combo) not in held:
                tot += float(pk.get("risk") or self.risk_usd)
        return tot

    def allowed_risk(self, mult=1.0):
        """The risk (dollars) a new trade may take now. The bot's limits: 4.5% below the day's starting
        balance and 9% below the account size, counting every open position as if it hit its stop.
        Far from a limit: the normal risk (times the combo's approved multiplier). Near it: half of the room
        that is left, so the stops can never cross the limit; under 10% of the normal risk the trade is skipped."""
        base = self.risk_usd * mult
        ai = mt5.account_info()
        ds = self.day_start_balance()
        left = ai.balance - self.open_risk()
        room = min(left - self.initial * (1 - GUARD_TOTAL), left - (ds - self.initial * GUARD_DAILY))
        r = min(base, GRADE_SHARE * room)
        return r if r >= GRADE_MIN * base else 0.0

    def guard(self):
        """Emergency stop: floating daily loss reaches 4.5% -> close everything until Prague midnight."""
        if self.halt_until and pd.Timestamp.now(tz="UTC") < self.halt_until:
            # a close that failed at the emergency (market closed, no quote, requote...) is tried again every minute
            if time.time() - self._halt_retry >= 60:
                self._halt_retry = time.time()
                left = self.positions()
                if left:
                    log(f"emergency stop: {len(left)} position(s) still open - closing again")
                    for p in left:
                        close(p, "emergency")
            return True
        ai = mt5.account_info()
        if ai.equity <= self.day_start_balance() - self.initial * 0.045 or ai.equity <= self.initial * 0.91:
            log(f"EMERGENCY equity {ai.equity:,.0f}: closing everything until Prague midnight")
            REP.notify(f"{BOT}: עצירת חירום", f"ההון ירד ל־${ai.equity:,.0f}. כל העסקאות נסגרות והבוט לא נכנס לעסקאות עד חצות (שעון פראג).",
                       priority=5, tags=["rotating_light"])
            for p in self.positions():
                close(p, "emergency")
            pending_guard(pd.Timestamp.now(tz="UTC"), everything=True)
            self._halt_retry = time.time()
            self.halt_until = (pd.Timestamp.now(tz="Europe/Prague").normalize() + pd.DateOffset(days=1)).tz_convert("UTC")
            return True
        return False


# ---------------- orders ----------------
def lots_for(sym, entry, sl, risk_usd):
    info = mt5.symbol_info(mt5_name(sym))
    if info is None or info.trade_tick_size == 0:
        return 0.0
    # the loss of 1 lot at the stop as MetaTrader itself calculates it (contract sizes differ between servers:
    # on MetaQuotes-Demo 1 lot of silver is 5,000 oz and the tick value it reports did not match that)
    per_lot = 0.0
    try:
        v = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if sl < entry else mt5.ORDER_TYPE_SELL, info.name, 1.0, entry, sl)
        per_lot = abs(v) if v else 0.0
    except Exception:  # noqa: BLE001
        per_lot = 0.0
    tick_way = abs(entry - sl) / info.trade_tick_size * info.trade_tick_value
    if per_lot <= 0:
        per_lot = tick_way
    elif tick_way > 0 and abs(per_lot - tick_way) / per_lot > 0.2:
        log(f"{sym}: lot value check - MetaTrader's profit calculator says ${per_lot:,.2f} per lot at the stop, the tick value says ${tick_way:,.2f}; using the larger (safer)")
        per_lot = max(per_lot, tick_way)
    if per_lot <= 0:
        return 0.0
    lots = risk_usd / per_lot
    step = info.volume_step or 0.01
    lots = np.floor(lots / step + 1e-9) * step
    digits = max(0, len(f"{step:.8f}".rstrip("0").split(".")[1]))
    lots = round(float(min(max(lots, 0.0), info.volume_max)), digits)
    return lots if lots >= info.volume_min else 0.0


def _fill_modes(symbol):
    """The filling modes this server allows for the symbol, the allowed ones first (other brokers' demos
    often refuse IOC; MetaTrader then answers with an error or with nothing at all)."""
    info = mt5.symbol_info(symbol)
    fm = getattr(info, "filling_mode", 0) or 0
    allowed = [m for bit, m in ((2, mt5.ORDER_FILLING_IOC), (1, mt5.ORDER_FILLING_FOK)) if fm & bit]
    return allowed + [m for m in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN) if m not in allowed]


def _comment(text):
    return re.sub(r"[^A-Za-z0-9 |_.-]", "", str(text))[:27]


def send(req, what):
    if ACC.dry:
        log(f"[dry-run] {what}: {req}")
        return None
    req = {**req, "comment": _comment(req.get("comment", ""))}
    r, err = None, None
    if req.get("action") in (mt5.TRADE_ACTION_DEAL, mt5.TRADE_ACTION_PENDING):
        modes = _fill_modes(req["symbol"])
        if req.get("action") == mt5.TRADE_ACTION_PENDING:     # a waiting limit order: "return" first
            modes = [mt5.ORDER_FILLING_RETURN] + [m for m in modes if m != mt5.ORDER_FILLING_RETURN]
        for fill in modes:
            r = mt5.order_send({**req, "type_filling": fill})
            if r is None:
                err = mt5.last_error()
                continue
            if r.retcode != 10030:                      # 10030 = this filling mode is not allowed: try the next
                break
    else:
        r = mt5.order_send(req)
        if r is None:
            err = mt5.last_error()
    # a market deal that filled only partly (10010) or was accepted for execution (10008) is a live position:
    # treat it as done, so the bot records and manages it (otherwise it would sit on the account unmanaged)
    ok = r is not None and (r.retcode == mt5.TRADE_RETCODE_DONE or
                            (req.get("action") == mt5.TRADE_ACTION_DEAL and r.retcode in (10008, 10010)) or
                            (req.get("action") == mt5.TRADE_ACTION_PENDING and r.retcode == 10008))
    if r is not None and r.retcode in (10008, 10010) and req.get("action") != mt5.TRADE_ACTION_PENDING:
        log(f"{what}: accepted with code {r.retcode} ({r.comment}) - managed as open")
    if not ok:
        MON["last_fail"] = (None, str(err)) if r is None else (r.retcode, r.comment)
        log(f"FAILED {what}: {('no answer', err) if r is None else (r.retcode, r.comment)}")
        if r is not None and r.retcode == 10027 and os.environ.get("MANAGED"):
            pass                                    # the manager switches Algo Trading on and tells the user
        elif r is not None and r.retcode == 10027:
            # the Algo Trading button of this account's MetaTrader is off: only the user can turn it on
            REP.notify(f"{BOT}: המסחר האוטומטי כבוי", "בחלון MetaTrader של החשבון הזה הכפתור Algo Trading כבוי, ולכן "
                       "הבוט לא יכול לפתוח או לסגור עסקאות. הצג את החלון מהאפליקציה ולחץ על Algo Trading (שיהיה ירוק).",
                       priority=5, tags=["warning"], key="algo_off", every=3600)
        elif r is None or r.retcode not in (10018,):           # 10018 = market closed: not worth a message
            REP.notify(f"{BOT}: פקודה נכשלה", f"{what}\n{('MetaTrader לא ענה', err) if r is None else (r.retcode, r.comment)}",
                       priority=4, tags=["warning"], key="fail:" + what[:20], every=1800)
        return None
    log(f"OK {what}: {r.order}")
    return r


# Fills far from the price the size was worked out on (user, 10.11.2026: FR08 SPCX on bot 4 sized 412 shares at
# ~142.8, filled at 144.84 -> $1,289 at the stop instead of $460). Two guards:
#  1. the price is read again right before sending, and the size is worked out on that fresh price;
#  2. each instrument's slippage is remembered (last 20 fills, against us only). Only for an instrument where a fill
#     has already put the loss at the stop over the risk (+10%), the size is worked out as if the fill will be the
#     usual bad case (80th percentile) worse - fewer lots, never more. Other instruments are sized as before
#     (user, 10.11.2026: "only for instruments that go over the stop amount").
SLIP_KEEP = 20


# until a bot has its own fills of an instrument: SPCX's 4 full-size entries on bot 4 filled about $2.0-$4.2 worse
# than the price their size was worked out on (worked back from lots, stops and the $460 risk; 05.-10.11.2026)
SLIP_SEED = {"SPCX": [2.80, 2.96, 4.21, 2.01]}


def slip_guess(sym):
    if sym not in SLIP_SEED and sym not in STATE.get("slip_over", {}):
        return 0.0
    q = [x for x in STATE.get("slip", {}).get(sym, SLIP_SEED.get(sym, [])) if x > 0]
    return float(np.percentile(q, 80)) if len(q) >= 3 else (max(q) if q else 0.0)


def note_slip(sym, d, asked, filled):
    if not (asked and filled):
        return
    x = d * (float(filled) - float(asked))            # > 0 = filled worse than asked
    q = STATE.setdefault("slip", {}).setdefault(sym, [])
    q.append(round(x, 6))
    del q[:-SLIP_KEEP]


def open_trade(combo, sym, d, sl, tp, entry_time, risk=None, mult=1.0, rules=None):
    risk = ACC.risk_usd * mult if risk is None else risk
    base = ACC.risk_usd * mult
    rules = rules or {}
    name = mt5_name(sym)
    tick = mt5.symbol_info_tick(name)
    px = tick.ask if d > 0 else tick.bid
    if d * (px - sl) <= 0 or (tp == tp and tp is not None and d * (tp - px) <= 0):
        log(f"skip {combo}: price {px} already beyond stop/target")
        return
    # approved exit changes: nearer stop (the lots follow it: the same dollar risk), nearer target
    if rules.get("sl"):
        sl = px - d * rules["sl"] * abs(px - sl)
    if rules.get("tp") and tp == tp and tp is not None:
        tp = px + d * rules["tp"] * abs(tp - px)
    sl = round(sl, mt5.symbol_info(name).digits)          # size the trade for the stop exactly as it will be sent
    fresh = mt5.symbol_info_tick(name)                  # guard 1: the price right before sending
    if fresh:
        px2 = fresh.ask if d > 0 else fresh.bid
        if px2 and abs(px2 - px) > 0:
            log(f"{combo}: price moved {px} -> {px2} before sending; sizing on {px2}")
            px = px2
    if d * (px - sl) <= 0:
        log(f"skip {combo}: price {px} already beyond the stop")
        return
    slip = slip_guess(sym)                              # guard 2: this instrument's usual bad fill
    size_px = px + d * slip
    if slip > 0:
        log(f"{combo}: sizing as if filled {slip:g} worse ({size_px:g}), from the last fills of {sym}")
    lots = lots_for(sym, size_px, sl, risk)
    if lots <= 0:
        log(f"skip {combo}: lot size too small")
        return
    info = mt5.symbol_info(name)
    # last check before sending: the loss at the stop (MetaTrader's own calculation, for this exact size and
    # the stop as it will be sent) may never be above the risk set for this bot. Otherwise: no trade + alert.
    sl_r = round(sl, info.digits)
    try:
        loss = abs(mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL, name, lots, size_px, sl_r) or 0.0)
    except Exception:  # noqa: BLE001
        loss = 0.0
    if not loss:
        loss = lots * abs(size_px - sl_r) / info.trade_tick_size * info.trade_tick_value
    if loss > risk + 0.01:
        log(f"BLOCKED {combo}: {lots} lots would lose ${loss:,.2f} at the stop, above the allowed risk ${risk:,.2f}")
        REP.notify(f"{BOT}: עסקה נחסמה", f"{pretty(combo)}\nבסטופ היא הייתה מפסידה ${loss:,.0f}, יותר מהסיכון המותר (${risk:,.0f}). לא נפתחה.",
                   priority=4, tags=["no_entry"], key="block:" + combo, every=3600)
        return
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": name, "volume": lots,
           "type": mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL, "price": px,
           "sl": sl_r, "tp": round(tp, info.digits) if tp == tp and tp is not None else 0.0,
           "deviation": 20, "magic": MAGICS_BY_COMBO[combo], "comment": combo[:31],
           "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    r = send(req, f"open {combo} {'buy' if d > 0 else 'sell'} {lots} @ {px} sl {sl:.5g} tp {tp}")
    if r is not None and not ACC.dry:
        REP.notify(f"{BOT}: עסקה נפתחה", f"{pretty(combo)}\n{'קנייה' if d > 0 else 'מכירה'} {lots} @ {px:g}\n"
                   f"סטופ {sl:.5g} · יעד {'—' if not (tp == tp and tp is not None) else f'{tp:.5g}'} · הפסד בסטופ ${loss:,.0f}"
                   + (f"\nסיכון לאסטרטגיה: ×{mult:g} מהרגיל" if abs(mult - 1) > 1e-9 else "")
                   + (f"\nסטופ ×{rules['sl']:g}" if rules.get("sl") else "") + (f" · יעד ×{rules['tp']:g}" if rules.get("tp") else "")
                   + (f"\nסיכון מוקטן: קרוב לגבול ההפסד (רגיל ${base:,.0f})" if risk < base - 0.5 else ""),
                   tags=["chart_with_upwards_trend" if d > 0 else "chart_with_downwards_trend"])
    if r is not None:
        fill = getattr(r, "price", 0) or 0
        note_slip(sym, d, px, fill)
        if fill and d * (fill - size_px) > 0:
            try:
                real = abs(mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL, name, lots, fill, sl_r) or 0.0)
            except Exception:  # noqa: BLE001
                real = 0.0
            if real > risk * 1.1:
                log(f"{combo}: filled at {fill} (asked {px}) - ${real:,.0f} at the stop, above the risk ${risk:,.0f}")
                REP.notify(f"{BOT}: החלקה בכניסה", f"{pretty(combo)}\nבוצע ב-{fill:g} במקום {px:g}. בסטופ: ${real:,.0f} (מותר ${risk:,.0f}).\nמעכשיו הבוט יקטין את הכמות בכניסות הבאות בנכס הזה לפי ההחלקה הרגילה שלו.",
                           priority=4, tags=["warning"], key="slip:" + combo, every=3600)
                loss = real
                STATE.setdefault("slip_over", {})[sym] = str(pd.Timestamp.now(tz="UTC"))
        STATE["entry"][combo] = str(entry_time)
        STATE["risk"][str(r.order)] = loss
        save_state()


def place_limit(combo, sym, d, lim, sl, tp, sig, until, risk, mult=1.0, rules=None):
    """An approved limit entry (plans "limit"): the order the backtest places, at its price, stop and target,
    sized so that a fill at the limit loses `risk` at the stop. Kept while the backtest's order waits (3 bars)."""
    rules = rules or {}
    name = mt5_name(sym)
    info, tick = mt5.symbol_info(name), mt5.symbol_info_tick(name)
    if info is None or tick is None:
        return
    px = tick.ask if d > 0 else tick.bid
    if d * (px - lim) <= 0 or d * (lim - sl) <= 0:
        log(f"skip {combo}: price {px} already at / past the limit {lim:g} - not taken")
        STATE["entry"][combo] = str(sig)
        save_state()
        return
    if rules.get("sl"):
        sl = lim - d * rules["sl"] * abs(lim - sl)
    if rules.get("tp") and tp == tp:
        tp = lim + d * rules["tp"] * abs(tp - lim)
    lim, sl = round(lim, info.digits), round(sl, info.digits)
    lots = lots_for(sym, lim, sl, risk)
    if lots <= 0:
        log(f"skip {combo}: lot size too small")
        return
    kind = mt5.ORDER_TYPE_BUY if d > 0 else mt5.ORDER_TYPE_SELL
    try:
        loss = abs(mt5.order_calc_profit(kind, name, lots, lim, sl) or 0.0)
    except Exception:  # noqa: BLE001
        loss = 0.0
    if not loss:
        loss = lots * abs(lim - sl) / info.trade_tick_size * info.trade_tick_value
    if loss > risk + 0.01:
        log(f"BLOCKED {combo}: {lots} lots would lose ${loss:,.2f} at the stop, above the allowed risk ${risk:,.2f}")
        return
    req = {"action": mt5.TRADE_ACTION_PENDING, "symbol": name, "volume": lots,
           "type": mt5.ORDER_TYPE_BUY_LIMIT if d > 0 else mt5.ORDER_TYPE_SELL_LIMIT, "price": lim,
           "sl": sl, "tp": round(tp, info.digits) if tp == tp and tp is not None else 0.0,
           "magic": MAGICS_BY_COMBO[combo], "comment": combo[:31], "type_time": mt5.ORDER_TIME_GTC}
    r = send(req, f"limit {combo} {'buy' if d > 0 else 'sell'} {lots} @ {lim} sl {sl:.5g} tp {tp} until {until:%d.%m %H:%M} UTC")
    if r is None:
        return
    STATE.setdefault("pending", {})[combo] = {"order": int(r.order), "sig": str(sig), "until": str(until), "sym": sym,
                                              "dir": d, "lim": lim, "risk": loss}
    STATE["entry"][combo] = str(sig)
    STATE["risk"][str(r.order)] = loss
    save_state()


def order_alive(pk):
    return bool(mt5.orders_get(ticket=int(pk["order"])))


def cancel_pending(combo, why):
    """Remove the combo's waiting limit order. True = nothing is waiting any more (and no position came of it)."""
    pk = STATE.get("pending", {}).get(combo)
    if not pk:
        return True
    if order_alive(pk):
        if ACC.dry or send({"action": mt5.TRADE_ACTION_REMOVE, "order": int(pk["order"])}, f"cancel limit {combo} ({why})") is None:
            return False
    elif any(p.magic == MAGICS_BY_COMBO.get(combo) for p in ACC.positions()):
        return False                                   # it filled: step() takes the position over
    STATE["pending"].pop(combo, None)
    save_state()
    return True


def pending_guard(now, everything=False):
    """Every loop: a waiting limit order is removed when its 3 bars are over, and before what would block a new
    entry (a market break or holiday, the last 15 minutes of a stock session, an FTMO restricted event)."""
    for combo, pk in list(STATE.get("pending", {}).items()):
        sym = pk["sym"]
        why = "emergency" if everything else None
        if why is None and now >= pd.Timestamp(pk["until"]):
            why = "3 bars over"
        elif why is None and (break_soon(sym, now) or holiday_block(sym, now)):
            why = "market break / holiday ahead"
        elif why is None and news_window(sym, now) == "close":
            why = "restricted news ahead"
        elif why is None and U.CLASS.get(sym) == "stock":
            tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[sym] else SESSION["EU"]
            loc = now.tz_convert(tz)
            if loc.hour * 60 + loc.minute >= close_min - 15:
                why = "stock session ends"
        if why:
            cancel_pending(combo, why)


# the replay's exit reasons are Hebrew; MT5 comments keep only ASCII (_comment), so they reached the trade history
# as blanks (23 of bot 1's first 118 trades) - sent in English instead
WHY_EN = {"סיגנל יציאה": "exit signal", "סיגנל הפוך": "opposite signal", "סגירה בשעה קבועה": "fixed-time exit",
          "יציאת זמן": "time exit", "יעד": "target", "סטופ": "stop"}


def close(p, why):
    why = WHY_EN.get(why, why)
    tick = mt5.symbol_info_tick(p.symbol)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume, "position": p.ticket,
           "type": mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY,
           "price": tick.bid if p.type == 0 else tick.ask, "deviation": 20, "magic": p.magic,
           "comment": why[:31], "type_filling": mt5.ORDER_FILLING_IOC}
    return send(req, f"close {p.comment} ({why})")


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
SPREADS = {}


def typical_spread(sym):
    """Median of the spreads this bot has seen for the symbol (sampled at every step, last 300), so the
    rollover's wide spread never moves a breakeven level."""
    tick = mt5.symbol_info_tick(mt5_name(sym))
    if tick and tick.ask > tick.bid:
        q = SPREADS.setdefault(sym, deque(STATE.get("spreads", {}).get(sym, []), maxlen=300))
        q.append(float(tick.ask - tick.bid))
        if len(q) % 20 == 0:
            STATE.setdefault("spreads", {})[sym] = list(q)          # kept across restarts
    q = SPREADS.get(sym)
    if not q:
        return 0.0
    return float(np.median(q))


def be_offset(sym, px):
    spec = U.spec(sym)
    spread = typical_spread(sym)
    if spec["commissionType"] == "flat_USD":
        comm = px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
    else:
        comm = px * 2 * float(spec["commission"]) / 100.0
    return (spread + comm) / px


def stock_session_close(now):
    """Stocks: close every position 15 minutes before the exchange closes (no overnight gaps)."""
    for p in ACC.positions():
        sym = SYM_BY_MAGIC.get(p.magic)
        if sym is None or U.CLASS.get(sym) != "stock" or news_window(sym, now) == "hold":
            continue
        tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[sym] else SESSION["EU"]
        loc = now.tz_convert(tz)
        if close_min - 15 <= loc.hour * 60 + loc.minute < close_min + 60:
            close(p, "session end")


DIAG = {}


def step(combo, sid, tf, sym, off):
    df = bars(sym, tf, off)
    now = pd.Timestamp.now(tz="UTC")
    if df is None:
        DIAG[combo] = {"at": now.isoformat(timespec="seconds"), "bars": 0}
        return False
    # The newest MT5 bar is still forming: keep only its open (what the backtest knows at entry),
    # and add one flat bar after it, because the engine enters on the bar AFTER a signal.
    forming = df.index[-1]
    last_closed = df.index[-2]
    DIAG[combo] = {"at": now.isoformat(timespec="seconds"), "bars": len(df), "forming": forming.isoformat(),
                   "age_min": round((now - forming).total_seconds() / 60, 1)}
    # the symbol's own clock (its last tick), not the wall clock: some feeds (MetaQuotes' shares) arrive late,
    # and a bar is "current" when it is the newest bar of its own feed
    tick = mt5.symbol_info_tick(mt5_name(sym))
    feed_now = server_ts(tick.time) if tick is not None and tick.time else now
    feed_now = min(now, max(feed_now, forming))
    DIAG[combo]["feed_lag_min"] = round((now - feed_now).total_seconds() / 60, 1)
    if feed_now - forming >= pd.Timedelta(minutes=TF_MIN[tf]) or now - forming >= pd.Timedelta(minutes=TF_MIN[tf] + 30):
        DIAG[combo]["stale"] = True
        return False    # the newest bar has not appeared yet (just started, or market closed): try again soon
    late = feed_now - forming > pd.Timedelta(minutes=min(5.0, TF_MIN[tf] / 4))
    df.iloc[-1, df.columns.get_indexer(["h", "l", "c"])] = df["o"].iloc[-1]
    dummy = df.iloc[[-1]].copy()
    dummy.index = [forming + pd.Timedelta(minutes=TF_MIN[tf])]
    df = pd.concat([df, dummy])
    df.attrs.update(symbol=sym, tf=tf)          # strategies that need the instrument (inter-market, events)
    if sid.startswith("IM"):
        from agents import intermarket_strategies as IM
        IM.PARTNER_BARS = partner_bars
        IM._CACHE.clear()                       # fresh partner bars every time, never yesterday's
    be = (0.6, be_offset(sym, float(df["c"].iloc[-3])))
    tr = R.run(df, strat(sid, df, sym, tf), 0.0, breakeven=be, session_exit=exit_mask(df, sym, tf))
    limit = "entry_pullback" in plan_of(combo)
    pend = tr.iloc[0:0]
    if len(tr):
        pend = tr[tr["reason"] == "pending"]
        tr = tr[tr["reason"] != "pending"]
    if len(tr):
        DIAG[combo]["last_signal"] = f"{tr.iloc[-1]['time']} {'buy' if tr.iloc[-1]['dir'] > 0 else 'sell'} {tr.iloc[-1]['reason']}"
        DIAG[combo]["trades_in_window"] = int(len(tr))
    pos = [p for p in ACC.positions() if p.magic == MAGICS_BY_COMBO[combo]]
    nw = news_window(sym, now)
    if nw:
        # FTMO restricted event: no position from 3 minutes before to 3 minutes after it.
        # An open one is closed before FTMO's ban starts; a new entry of this bar is dropped.
        if nw == "close" and pos:
            close(pos[0], "restricted news ahead (Standard rule)")   # closed for the news: not reopened
        new = tr[tr["time"] == forming] if len(tr) else tr
        if len(new) and STATE["entry"].get(combo) != str(new.iloc[-1]["time"]):
            STATE["entry"][combo] = str(new.iloc[-1]["time"])       # a signal inside the window is not taken, not later either
            save_state()
            log(f"skip {combo}: signal within 3 min of an FTMO restricted event - not taken (Standard rule)")
        DIAG[combo]["news_window"] = nw
        NEWS_RETRY.add(combo)
        return False
    NEWS_RETRY.discard(combo)
    pk = STATE.get("pending", {}).get(combo)
    if pk:
        if pos:
            # the limit order filled: from the replay's trade of that order on, managed like every other trade
            m = tr[tr["time"] >= pd.Timestamp(pk["sig"])] if len(tr) else tr
            if m.empty:
                return                                 # the fill bar is still forming: the broker's stop / target hold it
            STATE["entry"][combo] = str(m.iloc[0]["time"])
            STATE["pending"].pop(combo, None)
            save_state()
            log(f"{combo}: limit order {pk['order']} filled - trade of {m.iloc[0]['time']}")
            if not ACC.dry:
                REP.notify(f"{BOT}: עסקה נפתחה (לימיט)", f"{pretty(combo)}\n{'קנייה' if pk['dir'] > 0 else 'מכירה'} בלימיט {pk['lim']:g} · הפסד בסטופ ${pk['risk']:,.0f}",
                           tags=["chart_with_upwards_trend" if pk["dir"] > 0 else "chart_with_downwards_trend"])
        elif len(pend) and (pend["time"].astype(str) == pk["sig"]).any() and order_alive(pk):
            return                                     # the order is still waiting, as in the backtest
        elif not cancel_pending(combo, "the backtest's order is no longer waiting"):
            return

    def stop_now():
        """The stop as the backtest has it after the last CLOSED bar (the helper bars must not move it)."""
        tc = R.run(df.iloc[:-2], strat(sid, df.iloc[:-2], sym, tf), 0.0, breakeven=be, session_exit=exit_mask(df.iloc[:-2], sym, tf))
        tc = tc[tc["reason"] != "pending"] if len(tc) else tc
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
        if close(p, reason) is None:                   # exit signal / opposite signal / time / session end
            return                                     # the close failed: never open a new trade next to it
        # an opposite signal may open the next trade on this same bar: fall through
    src = pend if limit else tr                       # limit combos enter only through a waiting limit order
    if not len(src):
        return
    t = src[src["time"] == forming]
    if t.empty:
        return
    t = t.iloc[-1]
    if STATE["entry"].get(combo) == str(t["time"]):
        return                                         # already handled this entry
    if late:
        # the bar is already older than a quarter of it (max 5 min): a restart, a reconnect or a late feed.
        # The backtest entered at the bar's open; entering now at another price changes the stop distance
        # and every level in R, so this entry is skipped (and remembered as handled).
        log(f"skip {combo}: entry signal seen {(feed_now - forming).total_seconds() / 60:.1f} min into the bar")
        STATE["entry"][combo] = str(t["time"])
        save_state()
        return
    if U.CLASS.get(sym) == "stock":
        tz, close_min = SESSION["US"] if "USD" in U.CURRENCIES[sym] else SESSION["EU"]
        loc = pd.Timestamp.now(tz="UTC").tz_convert(tz)
        if loc.hour * 60 + loc.minute >= close_min - 15:
            note_skip(f"skip {combo}: last 15 minutes of the stock session - no new trades")
            return                                     # no new stock trades in the last 15 minutes
    if (h := holiday_block(sym)):
        log(f"skip {combo}: market closed for {h} (Standard rule)")
        return
    if break_soon(sym, pd.Timestamp.now(tz="UTC")):
        log(f"skip {combo}: market break over 2 hours starts within 15 minutes (Standard rule)")
        return
    if not CLOSE_BREAKS and news_near(sym):
        log(f"skip {combo}: high-impact news within 5 minutes")
        return
    mult, rules = risk_mult(combo), combo_rules(combo)
    why = None
    if mult <= 0:
        why = "removed by the user (no new trades)"
    elif rules.get("dir") and int(t["dir"]) != rules["dir"]:
        why = f"{'buy' if rules['dir'] > 0 else 'sell'} only (approved by the user)"
    elif rules.get("hours") and rules["hours"][0] <= pd.Timestamp.now(tz="Asia/Jerusalem").hour < rules["hours"][1]:
        why = f"no trades {rules['hours'][0]:02d}:00-{rules['hours'][1]:02d}:00 Israel (approved by the user)"
    elif rules.get("vol"):
        e0 = float(t["entry_px"]) if t.get("entry_px") == t.get("entry_px") and t.get("entry_px") else None
        sp = abs(e0 - float(t["stop_px"])) / e0 * 100 if e0 else None
        v = rules["vol"]
        if sp is not None and (("max_pct" in v and sp > v["max_pct"]) or ("min_pct" in v and sp <= v["min_pct"])):
            why = f"volatility filter: stop {sp:.3f}% of the price (approved by the user)"
    if why:
        log(f"skip {combo}: {why}")
        STATE["entry"][combo] = str(t["time"])         # handled: this entry is not taken
        save_state()
        return
    risk = ACC.allowed_risk(mult)
    if risk <= 0:
        log(f"skip {combo}: too close to the daily 4.5% / total 9% limit")
        return
    if limit:
        until = forming + pd.Timedelta(minutes=TF_MIN[tf] * int(t["bars_left"]))
        place_limit(combo, sym, int(t["dir"]), float(t["entry_px"]), float(t["stop_px"]), float(t["target_px"]), t["time"], until, risk, mult, rules)
        return
    open_trade(combo, sym, int(t["dir"]), float(t["stop_px"]), float(t["target_px"]), t["time"], risk, mult, rules)


# ---------------- monitoring: notifications + dashboard ----------------
BOT = "bot"
MON = {"open": {}, "free": set(), "day": None, "warned": set(), "push": 0, "eq": 0}


def deals_since(days):
    now = pd.Timestamp.now(tz="UTC")
    return [d for d in (mt5.history_deals_get(utc_to_server(now - pd.Timedelta(days=days)),
                                              utc_to_server(now) + dt.timedelta(days=1)) or [])
            if d.magic in MAGICS]


def combo_of(magic):
    for c, m in (STATE.get("magic_map") or MAGICS_BY_COMBO).items():
        if m == magic:
            return c
    return str(magic)


def status_line(ai):
    risk = ACC.open_risk() / ACC.initial * 100
    return f"יתרה ${ai.balance:,.0f} · הון ${ai.equity:,.0f} · סיכון פתוח {risk:.2f}%"


def _equity_at(t):
    """Equity at UTC time t from the 5-minute history (the last point before it), else None."""
    best = None
    for ts, _bal, eq in STATE.get("equity", []):
        if pd.Timestamp(ts) <= t:
            best = eq
        else:
            break
    return best


REPORT_KINDS = (("day", "דוח יומי", "היום"), ("week", "דוח שבועי", "השבוע"), ("month", "דוח חודשי", "החודש"))


def reports(ai, now=None):
    """Israel time: daily at 22:00; weekly on Friday at 23:30; monthly on the month's last weekday at
    23:30 (the last hour before the close). Profit/loss is against the account size, from the previous
    report of the same kind (the first time: from the start of the day/week/month)."""
    il = (now or pd.Timestamp.now(tz="UTC")).tz_convert("Asia/Jerusalem")
    last_wd = (il + pd.offsets.MonthEnd(0)).normalize()
    while last_wd.weekday() >= 5:
        last_wd -= pd.Timedelta(days=1)
    due = {"day": il.hour == 22 and il.minute < 30,
           "week": il.weekday() == 4 and (il.hour, il.minute) >= (23, 30),
           "month": il.normalize() == last_wd and (il.hour, il.minute) >= (23, 30)}
    starts = {"day": il.normalize(), "week": il.normalize() - pd.Timedelta(days=il.weekday()), "month": il.normalize().replace(day=1)}
    rep = STATE.setdefault("reports", {})
    for kind, title, word in REPORT_KINDS:
        key = il.strftime("%Y-%m-%d")
        if not due[kind] or rep.get(kind, {}).get("key") == key:
            continue
        prev = rep.get(kind, {})
        base = prev.get("eq") if prev.get("eq") is not None else _equity_at(starts[kind].tz_convert("UTC"))
        base = base if base is not None else ACC.initial
        pnl = ai.equity - base
        pct = pnl / ACC.initial * 100
        sign = "רווח" if pnl >= 0 else "הפסד"
        REP.notify(f"{BOT} · {title}", f"{sign} {word}: {'+' if pnl >= 0 else '−'}${abs(pnl):,.0f} ({pct:+.2f}% מהחשבון)\n{status_line(ai)}",
                   tags=["bar_chart"])
        rep[kind] = {"key": key, "eq": ai.equity, "t": pd.Timestamp.now(tz="UTC").isoformat()}
        save_state()


def monitor():
    now = pd.Timestamp.now(tz="UTC")
    ai = mt5.account_info()
    pos = {p.ticket: p for p in ACC.positions()}
    # closed positions (stop, target, bot close, emergency)
    for tk, p in list(MON["open"].items()):
        if tk not in pos:
            ds = [d for d in deals_since(10) if d.position_id == tk and d.entry in (1, 3)]
            pnl = sum(d.profit + d.commission + d.swap for d in ds)
            why = ds[-1].comment if ds else ""
            REP.notify(f"{BOT}: עסקה נסגרה {'ברווח' if pnl >= 0 else 'בהפסד'} ${pnl:+,.0f}",
                       f"{pretty(combo_of(p.magic))}\n{why}\nיתרה ${ai.balance:,.0f}",
                       tags=["white_check_mark" if pnl >= 0 else "x"])
            MON["open"].pop(tk)
            MON["free"].discard(tk)
    seeded = MON.get("seeded", False)              # first pass after a start: remember, do not announce
    sls = MON.setdefault("sl", {})
    for tk, p in pos.items():
        MON["open"][tk] = p
        free = bool(p.sl) and ((p.type == 0 and p.sl >= p.price_open) or (p.type == 1 and p.sl <= p.price_open))
        if free and tk not in MON["free"]:
            MON["free"].add(tk)
            if seeded:
                REP.notify(f"{BOT}: סטופ בנקודת איזון", f"{pretty(combo_of(p.magic))}\nהעסקה כבר לא יכולה להפסיד", tags=["shield"])
                sls[tk] = (p.sl, time.time())
        old = sls.get(tk)
        if old is None:
            sls[tk] = (p.sl, 0.0 if seeded else time.time())
        elif p.sl and abs(p.sl - old[0]) > 1e-12 and seeded:
            if time.time() - old[1] >= 1800:
                info = mt5.symbol_info(p.symbol)
                d = info.digits if info else 5
                REP.notify(f"{BOT}: הסטופ הוזז", f"{pretty(combo_of(p.magic))}\nסטופ חדש {p.sl:.{d}f} (היה {old[0]:.{d}f})",
                           tags=["arrow_up" if p.type == 0 else "arrow_down"])
                sls[tk] = (p.sl, time.time())
            else:
                sls[tk] = (p.sl, old[1])
    for tk in [k for k in sls if k not in pos]:
        sls.pop(tk)
    MON["seeded"] = True
    # daily loss warnings and the daily summary (Prague day)
    day = pd.Timestamp.now(tz="Europe/Prague").normalize()
    ds_bal = ACC.day_start_balance()
    MON["ds_bal"] = ds_bal
    if time.time() - MON.get("full_t", 0) >= 15:
        MON["full_t"] = time.time()
        TUN.SNAP["full"] = dashboard_state(ai, pos, ds_bal, now)
    wk = STATE.setdefault("warned", {})
    if wk.get("day") != str(day.date()):
        STATE["warned"] = wk = {"day": str(day.date()), "levels": []}
    MON["warned"] = set(wk["levels"])
    MON["day"] = day
    reports(ai)
    dd = (ds_bal - ai.equity) / ACC.initial * 100
    for lvl in (2, 3, 4):
        if dd >= lvl and lvl not in MON["warned"]:
            MON["warned"].add(lvl)
            wk["levels"] = sorted(MON["warned"])
            save_state()
            REP.notify(f"{BOT}: הפסד יומי {dd:.1f}%", f"ההון ${ai.equity:,.0f} · רף FTMO היומי 5% · הבוט עוצר הכל ב־4.5%",
                       priority=4 if lvl < 4 else 5, tags=["warning"])
    # equity history every 5 minutes, dashboard every 3 minutes
    if time.time() - MON.get("diag_log", 0) >= 1800 and DIAG:
        MON["diag_log"] = time.time()
        vals = list(DIAG.values())
        log(f"status: {len(vals)} combos checked | no data {sum(1 for v in vals if not v.get('bars'))} | "
            f"market closed {sum(1 for v in vals if v.get('stale'))} | open positions {len(pos)}")
    if time.time() - MON["eq"] >= 300:
        MON["eq"] = time.time()
        STATE.setdefault("equity", []).append([now.isoformat(timespec="minutes"), round(ai.balance, 2), round(ai.equity, 2)])
        STATE["equity"] = STATE["equity"][-12000:]
        save_state()
    if time.time() - MON["push"] >= 180:
        MON["push"] = time.time()
        err = REP.push_state(MON["name"], dashboard_state(ai, pos, ds_bal, now))
        if not MON.get("pushed_once"):
            MON["pushed_once"] = True
            log(f"dashboard update: {err or 'OK'}")
        MON["push_fail"] = MON.get("push_fail", 0) + 1 if err else 0
        if err:
            log(err)
            if MON["push_fail"] >= 3:                 # a single failed save is retried at the next one (3 min)
                REP.notify(f"{BOT}: הנתונים לא נשמרים ב־GitHub", f"כבר {MON['push_fail']} ניסיונות נכשלו. האפליקציה עדיין מתעדכנת בזמן אמת דרך המחשב.\n{err[:120]}",
                           priority=3, key="push", every=6 * 3600)


def _at(p, price):
    """Profit/loss in dollars if the position closes at price (MT5's own calculation, before commission)."""
    if not price:
        return None
    try:
        v = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if p.type == 0 else mt5.ORDER_TYPE_SELL, p.symbol, p.volume, p.price_open, price)
        return round(v, 2) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def positions_doc(pos):
    meta = STATE.get("pos_meta", {})
    return [{"ticket": p.ticket, "sl_usd": _at(p, p.sl), "tp_usd": _at(p, p.tp), "side": "buy" if p.type == 0 else "sell", "sl0": meta.get(str(p.ticket), {}).get("sl0") or p.sl,
             "tp0": meta.get(str(p.ticket), {}).get("tp0") or p.tp, "mae": meta.get(str(p.ticket), {}).get("mae"),
             "mfe": meta.get(str(p.ticket), {}).get("mfe"), "pip": pip_size(p.symbol),
             "combo": combo_of(p.magic), "symbol": p.symbol, "dir": "קנייה" if p.type == 0 else "מכירה",
             "lots": p.volume, "open": p.price_open, "now": p.price_current, "sl": p.sl, "tp": p.tp,
             "pnl": round(p.profit + p.swap, 2), "since": server_ts(p.time).isoformat(),
             "free": bool(p.sl and ((p.type == 0 and p.sl >= p.price_open) or (p.type == 1 and p.sl <= p.price_open)))}
            for p in pos.values()]


def refresh_live():
    """The real-time snapshot the dashboard reads every second through the tunnel."""
    ai = mt5.account_info()
    if ai is None:
        return
    pos = {p.ticket: p for p in ACC.positions()}
    track_live(ai, pos)
    TUN.SNAP["live"] = {"bot": MON["name"], "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "balance": ai.balance,
                        "equity": ai.equity, "day_start": MON.get("ds_bal", ai.balance), "lows": STATE.get("lows", {}),
                        "halted_until": str(ACC.halt_until) if ACC.halt_until else None, "positions": positions_doc(pos),
                        "algo": bool(getattr(mt5.terminal_info(), "trade_allowed", True))}   # MetaTrader's Algo Trading button


# ---------------- app data: closed trades, the lowest floating equity, candles ----------------
def pip_size(sym):
    s = sym.upper().split(".")[0]
    if s.startswith("XAU"):
        return 0.1
    if s.startswith("XAG"):
        return 0.01
    if len(s) == 6 and s.isalpha():
        return 0.01 if "JPY" in s else 0.0001
    return 1.0


def track_live(ai, pos):
    """Every quarter second: the day's and the all-time lowest equity (FTMO's loss limits count open
    trades too), and each open position's first stop/target and its worst/best price so far."""
    day = pd.Timestamp.now(tz="Europe/Prague").strftime("%Y-%m-%d")
    lw = STATE.setdefault("lows", {})
    if "all_low" not in lw:
        lw["all_low"] = min([e for _t, _b, e in STATE.get("equity", [])] + [ai.equity])
    if lw.get("day") != day:
        lw.update(day=day, day_low=ai.equity)
    lw["day_low"] = min(lw["day_low"], ai.equity)
    lw["all_low"] = min(lw["all_low"], ai.equity)
    meta = STATE.setdefault("pos_meta", {})
    for tk, p in pos.items():
        px, buy = p.price_current, p.type == 0
        m = meta.setdefault(str(tk), {"sl0": p.sl, "tp0": p.tp, "mae": px, "mfe": px})
        if not m.get("sl0") and p.sl:
            m["sl0"] = p.sl
        if not m.get("tp0") and p.tp:
            m["tp0"] = p.tp
        m["mae"] = min(m["mae"], px) if buy else max(m["mae"], px)
        m["mfe"] = max(m["mfe"], px) if buy else min(m["mfe"], px)
        m["sl"], m["tp"] = p.sl, p.tp
    if len(meta) > 800:
        for k in sorted(meta, key=int)[:len(meta) - 800]:
            meta.pop(k)


REASON = {4: "sl", 5: "tp", 6: "so"}


def closed_doc(days=60):
    """Every closed position (open + close deals put together), cached once it is final."""
    by = {}
    for d in deals_since(days):
        by.setdefault(d.position_id, []).append(d)
    cache = STATE.setdefault("closed", {})
    meta = STATE.get("pos_meta", {})
    for pid, ds in by.items():
        key = str(pid)
        ins = [d for d in ds if d.entry == 0]
        outs = [d for d in ds if d.entry in (1, 3)]
        if not ins or not outs or key in cache:
            continue
        o = ins[0]
        vol = sum(d.volume for d in outs) or o.volume
        if vol + 1e-9 < o.volume:                      # partly closed: wait for the rest
            continue
        m = meta.get(key, {})
        sl0, tp0 = m.get("sl0"), m.get("tp0")
        if sl0 is None:
            try:
                od = [x for x in (mt5.history_orders_get(position=pid) or []) if x.position_id == pid]
                if od:
                    sl0, tp0 = od[0].sl, od[0].tp
            except Exception:  # noqa: BLE001
                pass
        cache[key] = {
            "id": pid, "combo": combo_of(o.magic), "symbol": o.symbol, "side": "buy" if o.type == 0 else "sell",
            "lots": o.volume, "open_t": server_ts(o.time).isoformat(), "open_px": o.price,
            "close_t": server_ts(outs[-1].time).isoformat(), "close_px": round(sum(d.price * d.volume for d in outs) / vol, 6),
            "sl0": sl0 or 0, "tp0": tp0 or 0, "sl": m.get("sl", sl0) or 0, "tp": m.get("tp", tp0) or 0,
            "mae": m.get("mae"), "mfe": m.get("mfe"),
            "gross": round(sum(d.profit for d in ds), 2), "swap": round(sum(d.swap for d in ds), 2),
            "comm": round(sum(d.commission + getattr(d, "fee", 0.0) for d in ds), 2),
            "net": round(sum(d.profit + d.swap + d.commission + getattr(d, "fee", 0.0) for d in ds), 2),
            "why": outs[-1].comment, "reason": REASON.get(outs[-1].reason, ""), "pip": pip_size(o.symbol)}
    if len(cache) > 3000:
        for k in sorted(cache, key=lambda k: cache[k]["close_t"])[:len(cache) - 3000]:
            cache.pop(k)
    return sorted(cache.values(), key=lambda x: x["close_t"])


def withdrawals_doc(days=730):
    """Money taken out of the account (MetaTrader balance operations with a negative amount, e.g. a payout)."""
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for d in mt5.history_deals_get(utc_to_server(now - pd.Timedelta(days=days)), utc_to_server(now) + dt.timedelta(days=1)) or []:
        if d.type == mt5.DEAL_TYPE_BALANCE and d.profit < 0:
            t = server_ts(d.time)
            out.append({"id": str(d.ticket), "amount": round(-d.profit, 2), "date": t.tz_convert("Asia/Jerusalem").strftime("%Y-%m-%d"),
                        "time": t.isoformat(), "note": d.comment or ""})
    return out


import queue  # noqa: E402
import threading  # noqa: E402
CANDLE_Q = queue.Queue()
CANDLE_TF = {"1m": "M1", "5m": "M5", "15m": "M15", "1h": "H1", "4h": "H4", "1d": "D1", "1w": "W1"}


def _candles_now(q):
    sym, tf = q.get("symbol", ""), CANDLE_TF.get(q.get("tf", ""))
    if not tf or mt5.symbol_info(sym) is None:
        return {"error": "unknown symbol or timeframe"}
    t0 = pd.Timestamp(int(q["from"]), unit="s", tz="UTC")
    t1 = pd.Timestamp(int(q.get("to") or time.time()), unit="s", tz="UTC")
    rates = mt5.copy_rates_range(sym, getattr(mt5, "TIMEFRAME_" + tf), utc_to_server(t0), utc_to_server(t1))
    if rates is None or not len(rates):
        return {"symbol": sym, "tf": q["tf"], "bars": []}
    rates = rates[-3000:]
    ts = [int(x.timestamp()) for x in server_to_utc(rates["time"])]
    return {"symbol": sym, "tf": q["tf"], "digits": mt5.symbol_info(sym).digits,
            "bars": [[t, float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])] for t, r in zip(ts, rates)]}


def candles(q):
    """Called from the web server's thread: MT5 is only touched from the main loop, so hand it over."""
    box = {"ev": threading.Event()}
    CANDLE_Q.put((q, box))
    if not box["ev"].wait(12):
        return {"error": "busy, try again"}
    return box["doc"]


def serve_candles():
    while not CANDLE_Q.empty():
        q, box = CANDLE_Q.get_nowait()
        try:
            box["doc"] = _candles_now(q)
        except Exception as e:  # noqa: BLE001
            box["doc"] = {"error": str(e)}
        box["ev"].set()


TUN.ROUTES["/candles"] = candles

FAIL_HE = {10018: "השוק של הנכס הזה סגור עכשיו", 10027: "המסחר האוטומטי (Algo Trading) כבוי ב־MetaTrader", 10026: "המסחר האוטומטי חסום בשרת",
           10004: "המחיר זז, נסה שוב", 10021: "אין מחיר כרגע", 10030: "סוג הביצוע לא נתמך", 10031: "אין חיבור לשרת", 10017: "המסחר בנכס הזה חסום"}

# manual close from the app (manager → this bot's local port, with the manager's key): done in the main loop
ACTION_Q = queue.Queue()


def manual_close(d):
    box = {"ev": threading.Event()}
    ACTION_Q.put((d, box))
    if not box["ev"].wait(20):
        return {"error": "הבוט עסוק, נסה שוב"}
    return box["doc"]


def serve_actions():
    while not ACTION_Q.empty():
        d, box = ACTION_Q.get_nowait()
        try:
            pos = ACC.positions()
            if d.get("all"):
                pick = pos
            else:
                pick = [p for p in pos if str(p.ticket) == str(d.get("ticket"))]
                if not pick:
                    box["doc"] = {"error": "העסקה כבר לא פתוחה"}
                    box["ev"].set()
                    continue
            ok, bad, why = [], [], []
            for p in pick:
                if close(p, "app close"):
                    ok.append(p.ticket)
                else:
                    bad.append(p.ticket)
                    code, text = MON.get("last_fail", (None, ""))
                    why.append(f"{p.symbol}: {FAIL_HE.get(code, '')} (MetaTrader: {text} {code or ''})".replace(":  (", ": ("))
            log(f"manual close from the app: closed {ok}, failed {bad} {why}")
            if ok:
                REP.notify(f"{BOT}: נסגר ידנית מהאפליקציה", f"{len(ok)} עסקאות נסגרו" if len(ok) > 1 else "עסקה אחת נסגרה", tags=["hand"])
            box["doc"] = {"ok": not bad, "closed": ok, "failed": bad,
                          **({"error": "לא נסגר: " + " · ".join(why)} if bad else {})}
        except Exception as e:  # noqa: BLE001
            box["doc"] = {"error": str(e)}
        box["ev"].set()


TUN.POST_ROUTES["/close"] = manual_close


def dashboard_state(ai, pos, ds_bal, now):
    trades = []
    for d in deals_since(60):
        if d.entry not in (1, 3):
            continue
        trades.append({"time": server_ts(d.time).isoformat(), "combo": combo_of(d.magic), "symbol": d.symbol,
                       "dir": "קנייה" if d.type == 1 else "מכירה", "lots": d.volume, "price": d.price,
                       "pnl": round(d.profit + d.commission + d.swap, 2), "why": d.comment})
    return {
        "bot": MON["name"], "updated_utc": now.isoformat(), "account": ai.login, "server": ai.server,
        "initial": ACC.initial, "risk": MON["cfg"]["risk"], "free_risk_at_breakeven": ACC.free,
        "balance": ai.balance, "equity": ai.equity, "day_start": ds_bal,
        "halted_until": str(ACC.halt_until) if ACC.halt_until else None, "combos": MON["combos"],
        "positions": positions_doc(pos), "live_url": TUN.URL["url"],
        "trades": trades[-400:], "equity_curve": STATE.get("equity", [])[-4000:],   # the bot's log stays on the computer only
        "closed": closed_doc()[-500:], "skips": [{k: v for k, v in x.items() if k != "key"} for x in STATE.get("skips", [])[-300:]],
        "lows": STATE.get("lows", {}), "label": BOT, "withdrawals": withdrawals_doc(),
        # typical spread per symbol as this broker quotes it (median of what the bot saw): checks the backtest's
        # cost estimate for the shares that have no bid/ask history
        "spreads": {k: {"median": float(np.median(q)), "n": len(q)} for k, q in SPREADS.items() if q},
        "app_push": REP.PUSH_STAT, "managed": MANAGED,
        "risk_mult": {c: risk_mult(c) for c in MON["combos"] if risk_mult(c) != 1.0},
        "combo_rules": {c: r for c in MON["combos"] if (r := combo_rules(c))},
        "started": (STATE.get("equity") or [[now.isoformat()]])[0][0],
    }


# ---------------- main ----------------
def save_state():
    """Written to a temporary file and then swapped in, so a bot killed mid-write never leaves a broken
    state file behind; the previous good copy is kept as .bak."""
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(STATE, indent=1), encoding="utf-8")
    if STATE_FILE.exists():
        try:
            os.replace(STATE_FILE, STATE_FILE.with_suffix(".json.bak"))
        except OSError:
            pass
    os.replace(tmp, STATE_FILE)


def load_state(path):
    for f in (path, path.with_suffix(".json.bak")):
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                log(f"state file {f.name} is damaged - trying the backup")
                REP.notify(f"{path.stem}: קובץ מצב פגום", "הבוט עלה מגיבוי של קובץ המצב. העסקאות הפתוחות מנוהלות כרגיל.",
                           priority=4, tags=["warning"])
    return {"entry": {}, "risk": {}}


def magic_map(cfg):
    """One fixed MT5 magic number per combo, kept in the state file: removing or reordering combos in the
    config never moves an open position to another combo. A combo seen for the first time gets the next
    free number (the old rule, base + list position, is kept for the combos that already had it)."""
    mm = STATE.setdefault("magic_map", {})
    if not mm:
        mm.update({c: cfg["magic_base"] + i for i, c in enumerate(cfg["combos"])})
    for c in cfg["combos"]:
        if c not in mm:
            mm[c] = max(list(mm.values()) + [cfg["magic_base"] + len(cfg["combos"]) - 1]) + 1
    return mm


def no_quick_edit():
    """Windows console: clicking inside the window ("QuickEdit") pauses the program until a key is
    pressed - the bot would silently freeze. Turn it off for this window."""
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)
    except Exception:  # noqa: BLE001 - not Windows / no console
        pass


MANAGED = os.environ.get("MANAGED") == "1"


def mt5_init(terminal):
    """Start/attach MetaTrader. Started by the manager for an account added in the app: log in with
    the account it passes (the password never touches a file here)."""
    if os.environ.get("MT5_LOGIN"):
        return mt5.initialize(path=terminal, portable=True, login=int(os.environ["MT5_LOGIN"]),
                              password=os.environ.get("MT5_PASSWORD", ""), server=os.environ.get("MT5_SERVER", ""))
    return mt5.initialize(path=terminal, portable=True)


def main():
    no_quick_edit()
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
    STATE = load_state(STATE_FILE)
    for attempt in range(10):           # at Windows start-up MT5 may need a while
        if mt5_init(args.terminal):
            break
        log(f"MT5 did not start ({mt5.last_error()}), trying again in 60 s")
        time.sleep(60)
    else:
        REP.notify(f"{name}: לא עלה", f"לא הצלחתי להתחבר ל־MT5 ({args.terminal}). פתח את MT5, התחבר לחשבון והפעל את הבוט שוב.",
                   priority=5, tags=["warning"])
        sys.exit("MT5 did not start")
    ai = mt5.account_info()
    log(f"{name}: account {ai.login} {ai.server} balance {ai.balance:,.0f} | {len(cfg['combos'])} combos, "
        f"risk {cfg['risk'] * 100:g}% | {'DRY RUN' if args.dry_run else 'LIVE ON THIS ACCOUNT'}")
    if "demo" not in ai.server.lower() and "trial" not in ai.server.lower() and not args.dry_run:
        REP.notify(f"{name}: סירב לסחור", f"השרת {ai.server} לא נראה כמו חשבון דמו.", priority=5, tags=["warning"])
        sys.exit(f"server {ai.server} does not look like a demo: refusing to trade")
    log("checking the 39 instruments in MT5...")
    combos = []
    for c in cfg["combos"]:
        sid, tf, sym = c.split("|")
        sname = map_symbol(sym)
        if sname is None or not mt5.symbol_select(sname, True):
            log(f"WARNING: {sym} ({U.ALL[sym][0]}) cannot be traded on this server - {c} is off")
            continue
        if sname != U.ALL[sym][0].replace("/", ""):
            log(f"{sym}: on this server it is {sname}")
        combos.append((c, sid, tf, sym))
    if combos:
        tzname = pick_server_clock(ai.server, mt5_name(combos[0][3]))
        log(f"server clock: {'New York + 7 h (FTMO)' if tzname == 'NY+7' else tzname + ' (EU daylight dates)'}")
    mm = magic_map(cfg)
    save_state()
    MAGICS_BY_COMBO = {c: mm[c] for c in cfg["combos"]}
    MAGICS = set(mm.values())                     # also combos removed since: their positions stay guarded and closed
    SYM_BY_MAGIC.update({m: c.split("|")[2] for c, m in mm.items()})
    stray = [p for p in (mt5.positions_get() or []) if p.magic not in MAGICS
             and cfg["magic_base"] <= p.magic < cfg["magic_base"] + 10000]
    if stray:
        REP.notify(f"{name}: עסקה בלי שיוך", f"{len(stray)} עסקאות פתוחות עם מספר של הבוט שלא שייך לאף אסטרטגיה: "
                   + ", ".join(f"{p.symbol} #{p.ticket}" for p in stray), priority=4, tags=["warning"])
    ACC = Account(cfg, args.dry_run)
    PLANS.update({c: p for c, p in (cfg.get("plans") or {}).items() if isinstance(p, dict)})
    for c in PLANS:
        log(f"approved plan {c}: {PLANS[c]} -> {plan_of(c)}")
    global CLOSE_BREAKS
    CLOSE_BREAKS = bool(cfg.get("close_before_breaks"))
    if CLOSE_BREAKS:
        for c, sid, tf, sym in combos:
            if sym not in BREAKS:
                BREAKS[sym] = learn_breaks(sym)
                _BREAKS_AT[sym] = time.time()
        days = "Mon Tue Wed Thu Fri Sat Sun".split()
        log("Standard account rule: every position closes 15-30 min before a market break over 2 hours (server time):")
        for s_, b in sorted(BREAKS.items()):
            log(f"  {s_}: " + (", ".join(f"{days[d]} {m // 60:02d}:{m % 60:02d}" for d, m in sorted(b)) or "no breaks found"))
        cal = _calendar()
        syms = sorted({c[3] for c in combos})
        il = lambda t: t.tz_convert("Asia/Jerusalem").strftime("%a %d.%m %H:%M")  # noqa: E731
        ev = [(t, ins, [x for x in syms if CAL.restricted_affects(x, ins)]) for t, ins in cal["restricted"] if t > pd.Timestamp.now(tz="UTC")]
        log(f"FTMO calendar: {len(cal['restricted'])} restricted events, {len(cal['holidays'])} holidays loaded. Coming up for this bot (Israel time):")
        for t, ins, hit in sorted(ev):
            if hit:
                log(f"  {il(t)}  restricted: {ins} -> {', '.join(hit)}")
        days_hit = {}
        for x in syms:
            for d in CAL.closed_days(x, cal["holidays"]):
                if d >= pd.Timestamp.now(tz="UTC").normalize():
                    days_hit.setdefault(d, []).append(x)
        for d, hit in sorted(days_hit.items()):
            names = ", ".join(sorted({f"{t} ({c})" for dd, c, t in cal["holidays"] if dd == d and "Daylight" not in t}))
            log(f"  {d.strftime('%a %d.%m')}  holiday: {names} -> market closed for {', '.join(hit)}")
    global BOT
    tmpl = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4"}
    bot_label = cfg.get("bot_label") or tmpl.get(name) or name
    nick = os.environ.get("BOT_NICKNAME") or cfg.get("nickname")
    BOT = f"{nick} · {bot_label}" if nick else bot_label
    MON.update(name=name, cfg=cfg, dry=args.dry_run, combos=[c[0] for c in combos])
    REP.remember_alerts(ROOT / "live" / f"{name}_alerts.json")
    if not args.dry_run and not MANAGED:
        UPD.startup(log, BOT)
        UPD.announce(log, BOT, STATE, save_state)
    if not args.dry_run:
        port = int(os.environ.get("BOT_PORT") or 8760 + int("".join(ch for ch in name if ch.isdigit())[:1] or 0))
        try:
            TUN.serve(port)                     # also the guard against running the same bot twice
        except OSError:
            log(f"another copy of {name} is already running (port {port} is taken): this one stops")
            sys.exit(4)
    if not args.dry_run and not MANAGED and cfg.get("live_tunnel", True):
        try:

            def on_url(u):
                log(f"real-time dashboard feed: {u}")
                MON["push"] = 0                     # tell the dashboard the new address right away
            TUN.tunnel(port, log, on_url, name)
            log(f"real-time feed: local server on port {port}, starting the tunnel...")
        except Exception as e:  # noqa: BLE001
            log(f"real-time feed not started: {e}")
    if not args.dry_run:
        # under the manager (updates, restarts) the manager sends one "update installed" message instead
        err = None if os.environ.get("MANAGED") else REP.notify(f"{BOT} התחיל לעבוד", status_line(ai), tags=["robot"])
        log(f"phone notifications: {err or 'OK (topic ' + REP._CFG['topic'] + ')'}")
        log(f"dashboard: {'GitHub token found' if REP.token() else 'NO GitHub token - create live/github_token.txt'}")
    done, tried, tried_bar = {}, {}, {}
    probe = mt5_name(combos[0][3])
    import faulthandler
    hang_file = open(ROOT / "live" / f"{name}_hang.txt", "w")
    log("running: checking the strategies every 5 seconds")
    while True:
        # if one loop takes over 5 minutes, write where the program is stuck to live/<bot>_hang.txt
        faulthandler.dump_traceback_later(300, repeat=True, file=hang_file)
        try:
            if not ACC.guard():
                off = server_offset_hours(probe)
                now = pd.Timestamp.now(tz="UTC")
                stock_session_close(now)
                break_close(now)
                pending_guard(now)
                for combo, sid, tf, sym in combos:
                    bar_start = now.floor(f"{TF_MIN[tf]}min")
                    if done.get(combo) == bar_start or (now - bar_start).total_seconds() < 3:
                        continue
                    # a bar that MT5 has not produced yet is retried, at most once a minute, until it appears
                    wait = 5 if (now - bar_start).total_seconds() < 120 or combo in NEWS_RETRY else 60
                    if tried_bar.get(combo) == bar_start and (now - tried[combo]).total_seconds() < wait:
                        continue
                    handled = True
                    try:
                        handled = step(combo, sid, tf, sym, off) is not False
                    except Exception:  # noqa: BLE001
                        log(f"error {combo}:\n{traceback.format_exc()}")
                        REP.notify(f"{BOT}: שגיאה", f"{pretty(combo)}\n{traceback.format_exc()[-300:]}", priority=4,
                                   tags=["warning"], key="err:" + combo, every=3600)
                    if handled:
                        done[combo] = bar_start
                    else:
                        tried[combo], tried_bar[combo] = now, bar_start
        except Exception:  # noqa: BLE001
            log(f"loop error:\n{traceback.format_exc()}")
            if not mt5_init(args.terminal):
                log("MT5 connection lost; retrying in 60 s")
                REP.notify(f"{BOT}: אין חיבור ל־MT5", "בדוק ש־MT5 פתוח ומחובר לחשבון. הבוט מנסה שוב כל דקה.",
                           priority=5, tags=["warning"], key="mt5lost", every=1800)
                time.sleep(60)
        try:
            if not args.dry_run:
                monitor()
        except Exception:  # noqa: BLE001
            log(f"monitor error:\n{traceback.format_exc()}")
        try:
            if not args.dry_run and not MANAGED:
                UPD.tick(log, BOT, STATE, save_state, lambda: len(ACC.positions()), lambda: UPD.restart_self(log))
        except Exception:  # noqa: BLE001
            log(f"update error:\n{traceback.format_exc()}")
        for _ in range(20):                       # 5 seconds, refreshing the live dashboard feed 4 times a second
            try:
                if not args.dry_run:
                    serve_candles()
                    serve_actions()
                    refresh_live()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)


if __name__ == "__main__":
    main()
