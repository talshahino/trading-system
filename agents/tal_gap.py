"""The user's gap method (data/agents/strategies/tal_gap.md, 11.11.2026) - one engine for bots 5 and 6 (shared methods,
user 11.11.2026; not bot 7). Each bot tests it on its own world (instruments, costs, sessions) through a `world`.

  high timeframe (HTF)  the impulse: an up move -> longs only, a down move -> shorts only. A gap = the space between
                        candle 1's high and candle 3's low, wicks included (mirror for shorts), made by a strong candle 2
                        (body >= GBODY x the average body of the 20 candles before; the gap >= MIN_GAP x their average range).
                        The impulse ends at the first HTF swing (2 candles each side) beyond the gap; the gap counts only if
                        the price did not come back to its middle until that swing was confirmed.
  entry                 a limit order at the gap's middle, placed at the impulse's end, valid for EXPIRY hours;
                        filled on the 1-minute bars (gap-through fills at the open).
  stop                  counting back in time on the low timeframe (LTF): swing lows below the entry (SW candles each side,
                        confirmed before the fill), each one beyond the one counted before by at least 0.3 LTF-ATR
                        (small, close swings count as one); the stop is BUF beyond the 2nd one. Mirror for shorts.
  target                the 2nd swing on the other side counted the same way (longs: the 2nd high above the entry).
                        Taken only when target / stop >= MIN_RR.
  no management         to the stop or the target; only the world's session close forces an exit.
  one trade at a time   per instrument; a touch while a trade is open is dropped.
  protection (PROT)     none | cal15 / cal60: no entry within +-15 / 60 minutes of a high-impact release of one of the
                        instrument's currencies, and a waiting order is cancelled when such a window starts |
                        shock: the agent's shock (below) against the trade cancels every waiting gap of that direction;
                        no fill in a shock minute or in the 5 minutes after one; no fill when the fill minute runs
                        through the whole gap | cal15shock: both.
  shock (the agent's definition, 12.11.2026 - from the minute chart, as the user asked: "the agent decides")
                        a 1-minute candle whose range is >= SHOCK_X x the average range of the 60 minutes before it and
                        whose body is >= half its range, or 3 minutes that together move >= 1.5 x SHOCK_X of that average.
                        Its direction is its body's. It is tuned later from what protects best (bot 7 studies shocks).
Results: chosen on the period to 2025-12-31 only, shown on 2026 (unseen), time-weighted (agents/agentkit.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents import agentkit as K

TF = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
TF_HE = {"1m": "דקה", "5m": "5 דק'", "15m": "15 דק'", "1h": "שעה", "4h": "4 שעות", "1d": "יום"}
# the user's pairs (tal_gap.md): two steps apart first, then one, then three
PAIRS = [("15m", "1m", 2), ("1h", "5m", 2), ("4h", "15m", 2), ("1d", "1h", 2),
         ("5m", "1m", 1), ("15m", "5m", 1), ("1h", "15m", 1), ("4h", "1h", 1), ("1d", "4h", 1),
         ("1h", "1m", 3), ("4h", "5m", 3), ("1d", "15m", 3)]
BASE = {"GBODY": 1.5, "MIN_GAP": 0.3, "SW": 3, "MIN_RR": 1.0, "EXPIRY": 48, "PROT": "none"}
KNOBS = {"GBODY": [0.0, 1.0, 2.0], "MIN_GAP": [0.0, 0.6], "SW": [2, 5], "MIN_RR": [0.0, 0.5, 1.5, 2.0],
         "EXPIRY": [8, 24, 120], "PROT": ["cal15", "cal60", "shock", "cal15shock"],
         # trade management (user, 12.11.2026: "partial exit? stop to break-even? the smart agent should optimise it"):
         # beW = the stop to the entry once the price went W of the way to the target; pF_W = F of the trade out at W of
         # the way, the rest's stop to the entry. Not in BASE (the base stays "none"), so earlier runs keep their names.
         # Fibonacci levels of the way (user, 12.11.2026, every bot): 3 digits = per mille (be618 = 61.8% of the way)
         "EXIT": ["be50", "be70", "p30_50", "p50_50", "p50_70", "p70_70",
                  "be382", "be618", "be786", "be886", "p50_382", "p50_618", "p33_618", "p50_786"],
         # one loss per impulse (bot 6's smart agent, 13.11.2026, from bot 6's live record: on 5.11 three US30 longs from
         # gaps of the same 1-hour impulse were stopped one after another, -3R, and two US500 longs of another, -2R):
         # after a stop, the impulse's other gaps in that direction are cancelled. Not in BASE (off), like EXIT.
         "IMP1": [True],
         # the direction of the bigger picture (bot 5's smart agent, 13.11.2026 night, from the loss of 12.11: nine
         # index shorts were stopped one after the other while the indices rose all afternoon - each short was in its
         # own impulse's direction but against the day's). day = only longs while the price is above the day's open
         # (UTC) and only shorts below it; up20 = only in the direction of the next timeframe up (at most 4 hours)
         # against its 20-candle EMA. Off by default (not in BASE), like EXIT / IMP1. Both work on the live bot's
         # 14 days of minutes (live_trades applies the same rule).
         "TREND": ["day", "up20"],
         # laddered entries across the gap's box, the same total risk (user, 13.11.2026; LADDERS above simulate). Off
         # by default (not in BASE), like EXIT / IMP1 / TREND; the live bot runs the same levels (live_trades).
         "LADDER": ["2", "3", "3m", "4"]}
NAMES = {"GBODY": "עוצמת הדחף (גוף הנר האמצעי מול הממוצע)", "MIN_GAP": "גודל גאפ מינימלי (מול הטווח הממוצע)",
         "SW": "סווינג בטיים פריים הנמוך (נרות מכל צד)", "MIN_RR": "יחס מינימלי יעד/סטופ", "EXPIRY": "תוקף הגאפ (שעות)",
         "PROT": "הגנה מהודעות וזעזועים", "EXIT": "ניהול העסקה: יציאה חלקית / סטופ לנקודת הכניסה",
         "IMP1": "אחרי סטופ - בלי עוד כניסות מאותו דחף",
         "TREND": "רק בכיוון התמונה הגדולה (פתיחת היום / טיים פריים מעל)",
         "LADDER": "כניסה מדורגת לאורך הגאפ (אותו סיכון כולל)"}
UP = {"1m": "5m", "5m": "15m", "15m": "1h", "1h": "4h", "4h": "4h", "1d": "4h"}   # TREND up20: the timeframe above (<= 4h)
PROT_HE = {"none": "בלי הגנה", "cal15": "יומן ±15 דק'", "cal60": "יומן ±60 דק'", "shock": "זעזועים",
           "cal15shock": "יומן ±15 דק' + זעזועים"}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
SHOCK_X = 4.0
MERGE_ATR = 0.3
CALM_MIN = 5
RERUN_DAYS = 14
RESULTS = Path("data/agents/strategies/tal_gap_results.json")


def stop_buffer(sym):
    """The stop's 2 pips / points beyond the 2nd swing (tal_gap.md): an index 2 points (as since 11.11); forex, metals
    and the dollar index 2 pips - 10 x FTMO's last digit (EURUSD 0.0002, USDJPY 0.02, gold 0.2, silver 0.02)."""
    try:
        from agents import ftmo_universe as U
        if U.CLASS.get(sym) == "index" and sym != "DXY":
            return 2.0
        return 2 * 10.0 ** -(int(U.spec(sym)["digits"]) - 1)
    except Exception:  # noqa: BLE001 - an unknown name: the indices' rule
        return 2.0


def pair_name(p):
    return f"{p[0]}/{p[1]}"


def variants():
    out = [("base", None, dict(BASE))]
    for k, vals in KNOBS.items():
        for v in vals:
            if BASE.get(k, "none") != v:
                out.append((k, v, dict(BASE, **{k: v})))
    return out


def vid(cfg):
    import hashlib
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def pre_break_mask(ns, minutes):
    """Data.pre_break on any sorted 1-minute bar times (int64 ns, UTC) - also the live bot's own masks
    (live/watch_own.py), so every track of bot 5 closes before a break, not only the gap tracks."""
    ns = np.asarray(ns, np.int64)
    n = len(ns)
    m = np.zeros(n, bool)
    if n < 2:
        return m
    step = 60_000_000_000
    brk = np.flatnonzero((ns[1:] - ns[:-1] - step) > 2 * 3600 * 10**9)
    if not len(brk):
        return m
    c = pd.DatetimeIndex(ns[brk] + step).tz_localize("UTC")
    best = None
    for z in ("UTC", "America/New_York", "Europe/London", "Europe/Berlin", "Australia/Sydney"):
        loc = c.tz_convert(z).tz_localize(None) + pd.Timedelta(days=7)
        p = loc.tz_localize(z, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
        v = p.as_unit("ns").asi8.astype(float)
        v[np.asarray(p.isna())] = np.inf
        best = v if best is None else np.minimum(best, v)
    p = best[np.isfinite(best)].astype(np.int64)
    a = np.searchsorted(ns, p - int(minutes) * step)
    b = np.searchsorted(ns, p + 2 * 3600 * 10**9)
    dm = np.zeros(n + 1, int)
    np.add.at(dm, a, 1)
    np.add.at(dm, b, -1)
    return np.cumsum(dm[:-1]) > 0


# ---------------------------------------------------------------- data per instrument (shared by all variants)
class Data:
    """1-minute bars + the world's masks, and every timeframe resampled once."""

    def __init__(self, s, df, cost, shut, late, buf, cal):
        self.s, self.df = s, df
        self.t1 = df.index
        self.ns = self.t1.as_unit("ns").asi8
        self.o1, self.h1, self.l1, self.c1 = (df[x].to_numpy(float) for x in "ohlc")
        self.cost, self.shut, self.late, self.buf = np.asarray(cost, float), np.asarray(shut, bool), np.asarray(late, bool), buf
        self.cal = cal                                    # {"cal15": mask, "cal60": mask}
        r = self.h1 - self.l1
        avg = pd.Series(r).rolling(60, min_periods=30).mean().shift(1).to_numpy()
        body = self.c1 - self.o1
        one = (r >= SHOCK_X * avg) & (np.abs(body) >= 0.5 * r)
        mv3 = self.c1 - np.r_[np.full(2, np.nan), self.o1[:-2]]
        three = np.abs(mv3) >= 1.5 * SHOCK_X * avg
        self.shock_up = np.nan_to_num((one & (body > 0)) | (three & (mv3 > 0))).astype(bool)
        self.shock_dn = np.nan_to_num((one & (body < 0)) | (three & (mv3 < 0))).astype(bool)
        anyk = (self.shock_up | self.shock_dn).astype(int)
        cs = np.cumsum(anyk)
        self.recent_shock = (cs - np.r_[np.zeros(CALM_MIN + 1, int), cs[:-(CALM_MIN + 1)]]) > 0   # this or the 5 before
        self.cs_up, self.cs_dn = np.cumsum(self.shock_up), np.cumsum(self.shock_dn)
        self.cal_cs = {k: np.cumsum(v) for k, v in cal.items()}
        self.cal_start = {k: np.cumsum(v & ~np.r_[False, v[:-1]]) for k, v in cal.items()}
        self.bars, self.sw = {}, {}

    def pre_break(self, minutes):
        """WKND (FTMO rule, user 14.11.2026): per 1-minute bar, True from `minutes` before the market's next break of 2+
        hours (the weekend, a daily close, a holiday) - known IN ADVANCE, from the same weekday a week earlier: each
        break seen in the data (the last bar's end) is carried 7 days forward on every clock an exchange may follow
        (UTC, New York, London, Frankfurt, Sydney - summer / winter time), and the earliest is taken, so a clock
        change never makes it late. The window runs to 2 hours after that time (the break itself). The same on the
        backtest and on the live bot's 14 days of minutes; a holiday's break also closes early the week after (careful)."""
        key = ("prebrk", int(minutes))
        if key not in self.sw:
            self.sw[key] = pre_break_mask(self.ns, minutes)
        return self.sw[key]

    def cancel_at(self, minutes):
        """WKND: the bars where every waiting gap order is cancelled - the pre-break window (pre_break) and the last bar
        before a 2+ hour break in the data (on the live bot that one is known only after the break, when pre_break has
        already cancelled it). Sorted bar positions."""
        key = ("cancel", int(minutes))
        if key not in self.sw:
            ns = self.ns
            last = np.r_[(ns[1:] - ns[:-1] - 60_000_000_000) > 2 * 3600 * 10**9, False] if len(ns) else np.zeros(0, bool)
            self.sw[key] = np.flatnonzero(self.pre_break(minutes) | last)
        return self.sw[key]

    def tf(self, name):
        if name not in self.bars:
            m = self.df if name == "1m" else self.df.resample(f"{TF[name]}min", label="left", closed="left").agg(
                {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
            o, h, l, c = (m[x].to_numpy(float) for x in "ohlc")
            pc = np.r_[c[0], c[:-1]]
            tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
            atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
            self.bars[name] = (m.index.as_unit("ns").asi8, o, h, l, c, atr)
        return self.bars[name]

    def trend(self, htf, mode, ltf=None):
        """TREND: per 1-minute bar, +1 / -1 / 0 = the side the bigger picture allows (known before the bar opens), or
        None when off. day: the previous minute's close against the UTC day's first open; up20: the last CLOSED candle
        of the timeframe above HTF against its 20-candle EMA; e920 / e920l (user, 14.11.2026): the EMA 9 against the
        EMA 20 of the last CLOSED candle of the high timeframe (e920) / the entry timeframe (e920l) - 9 above 20 =
        longs only, below = shorts only."""
        if not mode or mode == "none":
            return None
        key = ("trend", htf, mode, ltf if mode == "e920l" else None)
        if key not in self.sw:
            prev = np.r_[self.c1[0], self.c1[:-1]]
            if mode in ("e920", "e920l"):
                tfn = ltf if (mode == "e920l" and ltf) else htf
                ts, o, h, l, c, _ = self.tf(tfn)
                e9 = pd.Series(c).ewm(span=9, adjust=False).mean().to_numpy()
                e20 = pd.Series(c).ewm(span=20, adjust=False).mean().to_numpy()
                j = np.searchsorted(ts, self.ns - TF[tfn] * 60_000_000_000, side="right") - 1
                jj = np.maximum(j, 0)
                s = np.where(j >= 0, np.sign(e9[jj] - e20[jj]), 0)
            elif mode == "day":
                day0 = pd.Series(self.o1, index=self.t1).groupby(self.t1.floor("1D")).transform("first").to_numpy(float)
                s = np.sign(prev - day0)
            elif mode == "rth":
                # bot 6's smart agent (13.11.2026): the index futures' own bigger picture - from the New York cash open
                # (09:30) to 18:00 New York, the side of the price against that day's 09:30 open; the rest of the
                # futures day falls back to "day". Not in KNOBS (bot 6 tests it on its running tracks only).
                ny = self.t1.tz_convert("America/New_York")
                mins = np.asarray(ny.hour * 60 + ny.minute)
                rth = (mins >= 570) & (mins < 1080)
                day = ny.normalize()
                day0 = pd.Series(self.o1, index=self.t1).groupby(self.t1.floor("1D")).transform("first").to_numpy(float)
                o930 = pd.Series(np.where(rth, self.o1, np.nan)).groupby(np.asarray(day)).transform("first").to_numpy(float)
                ref = np.where(rth & ~np.isnan(o930), o930, day0)
                s = np.sign(prev - ref)
            else:
                up = UP.get(htf, "4h")
                ts, o, h, l, c, _ = self.tf(up)
                ema = pd.Series(c).ewm(span=int(mode[2:]), adjust=False).mean().to_numpy()
                j = np.searchsorted(ts, self.ns - TF[up] * 60_000_000_000, side="right") - 1
                jj = np.maximum(j, 0)
                s = np.where(j >= 0, np.sign(c[jj] - ema[jj]), 0)
            self.sw[key] = np.nan_to_num(s).astype(np.int8)
        return self.sw[key]

    def bbw_ok(self, tfn, pct):
        """BBW (user, 14.11.2026): per 1-minute bar, True when the Bollinger 20/2 width ((upper - lower) / middle) of
        the last CLOSED candle of timeframe tfn is at least its pct-quantile over the 100 candles before it - no entry
        in a squeeze (a quiet market). Known before the bar opens."""
        key = ("bbw", tfn, pct)
        if key not in self.sw:
            ts, o, h, l, c, _ = self.tf(tfn)
            cs = pd.Series(c)
            mid = cs.rolling(20).mean()
            w = (4 * cs.rolling(20).std(ddof=0) / mid).to_numpy()
            q = pd.Series(w).rolling(100, min_periods=50).quantile(pct).shift(1).to_numpy()
            ok_c = ~(w < q)                                   # unknown (warm-up) = allowed
            j = np.searchsorted(ts, self.ns - TF[tfn] * 60_000_000_000, side="right") - 1
            self.sw[key] = np.where(j >= 0, ok_c[np.maximum(j, 0)], True).astype(bool)
        return self.sw[key]

    def swings(self, name, w):
        """LTF swing highs / lows: (confirm time ns [end of the confirming candle], value, atr at the swing)."""
        key = (name, w)
        if key not in self.sw:
            ts, o, h, l, c, atr = self.tf(name)
            step = TF[name] * 60_000_000_000
            out = []
            for arr, hi in ((h, True), (l, False)):
                s = pd.Series(arr)
                ext = (s.rolling(2 * w + 1, center=True).max() if hi else s.rolling(2 * w + 1, center=True).min()).to_numpy()
                idx = np.flatnonzero(arr == ext)
                idx = idx[idx + w < len(arr)]
                out.append((ts[idx + w] + step, arr[idx], atr[idx]))
            self.sw[key] = tuple(out)
        return self.sw[key]


# ---------------------------------------------------------------- the impulse's gaps on the high timeframe
def gaps(D, htf, gbody, min_gap, cache):
    key = (htf, gbody, min_gap)
    if key in cache:
        return cache[key]
    ts, o, h, l, c, _ = D.tf(htf)
    n = len(c)
    step = TF[htf] * 60_000_000_000
    body = np.abs(c - o)
    avgb = pd.Series(body).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    avgr = pd.Series(h - l).rolling(20, min_periods=10).mean().shift(1).to_numpy()
    s_hi = np.zeros(n, bool)
    s_lo = np.zeros(n, bool)
    for i in range(2, n - 2):
        s_hi[i] = h[i] >= h[i - 2:i + 3].max()
        s_lo[i] = l[i] <= l[i - 2:i + 3].min()
    out, geo = [], []
    for k in range(1, n - 1):
        if not (np.isfinite(avgb[k]) and body[k] >= gbody * avgb[k]):
            continue
        d = 1 if c[k] > o[k] else -1 if c[k] < o[k] else 0
        if d > 0 and l[k + 1] > h[k - 1]:
            far, near = h[k - 1], l[k + 1]
        elif d < 0 and h[k + 1] < l[k - 1]:
            far, near = l[k - 1], h[k + 1]
        else:
            continue
        if abs(near - far) < min_gap * avgr[k]:
            continue
        mid = (near + far) / 2
        arm = None
        for j in range(k + 1, min(n - 2, k + 40)):
            if j >= k + 2 and ((l[j] <= mid) if d > 0 else (h[j] >= mid)):
                break                                   # back to the middle during the impulse: not a gap
            if (s_hi[j] if d > 0 else s_lo[j]):
                if all((l[x] > mid) if d > 0 else (h[x] < mid) for x in (j + 1, j + 2)):
                    arm = j
                break
        if arm is None:
            continue
        out.append((d, mid, near, far, int(ts[k + 1] + step), int(ts[arm + 2] + step)))   # made, armed (ns)
        # the chart of the trade (user, 13.11.2026): where the impulse started, its top, and candles 1-3 of the gap
        a0 = max(0, k - 6)
        i0 = a0 + int(np.argmin(l[a0:k + 1]) if d > 0 else np.argmax(h[a0:k + 1]))
        geo.append({"c1": int(ts[k - 1]), "c3": int(ts[k + 1]), "imp0": [int(ts[i0]), float(l[i0] if d > 0 else h[i0])],
                    "imp1": [int(ts[arm]), float(h[arm] if d > 0 else l[arm])]})
    cache[key] = out
    cache[key + ("geo",)] = geo
    return out


def _count_back(conf, val, atr, t_ns, px, d, side):
    """The 2nd swing beyond px counting back from t_ns. side=-1: below px (lows), +1: above px (highs)."""
    j = int(np.searchsorted(conf, t_ns, side="right")) - 1
    got, last = [], px
    lo = max(-1, j - 400)
    while j > lo:
        v = val[j]
        if side * (v - last) > 0 and (not got or side * (v - last) >= MERGE_ATR * atr[j]):
            got.append(v)
            last = v
            if len(got) == 2:
                return v
        j -= 1
    return None


def _count_back_t(conf, val, atr, t_ns, px, d, side, step_ns, w):
    """_count_back's swing and the time of its candle (confirmed w candles after it): for the trade's chart."""
    j = int(np.searchsorted(conf, t_ns, side="right")) - 1
    got, last = [], px
    lo = max(-1, j - 400)
    while j > lo:
        v = val[j]
        if side * (v - last) > 0 and (not got or side * (v - last) >= MERGE_ATR * atr[j]):
            got.append(v)
            last = v
            if len(got) == 2:
                return int(conf[j] - (w + 1) * step_ns)
        j -= 1
    return None


def exit_mode(cfg):
    """(fraction out, way to the target) for the EXIT knob; (0, way) = only the stop to the entry; None = no management."""
    m = cfg.get("EXIT", "none")
    if m == "none" or m.startswith("tp"):                 # "tp30": the whole position out at 30% (tp_way), no management
        return None
    way = lambda x: int(x) / (1000 if len(x) == 3 else 100)   # noqa: E731 - "618" = 61.8%, "70" = 70%
    if m.startswith("be"):
        return 0.0, way(m[2:])
    f, w = m[1:].split("_")
    return int(f) / 100, way(w)


# (user, 13.11.2026: "find a way to pass most of them"): what kills the method on forex is the cost - spread +
# commission as a share of the risk - and the quiet hours. MAXC = skip a trade whose cost is more than this share of
# its risk; SESS = only entries in these UTC hours ("ln" London 07-16, "ny" 12-20, "lnny" 07-20). Off unless set.
SESS_H = {"ln": (7, 16), "ny": (12, 20), "lnny": (7, 20)}
# the index futures' own clock (bot 6's smart agent, 13.11.2026 night), in New York time so summer / winter time do not
# move it - minutes from midnight New York: rth = the cash session 09:30-16:00, nyam = its first two hours 09:30-11:30
# (the open's move), xopen = every hour but the open's first half hour 09:30-10:00 (its noise). Off unless set.
SESS_NY = {"rth": [(570, 960)], "nyam": [(570, 690)], "xopen": [(0, 570), (600, 1440)],
           # (bot 6's smart agent, 13.11.2026 late night, from bot 6's numbers) xreopen = no entry in the first hour
           # after the daily halt, 18:00-19:00 New York (the gap tracks' trades in 23:00 UTC lost in both periods,
           # -0.19R / -0.37R a trade, while 22:00 UTC earned); xlate = no entry in the last 45 minutes before the
           # 15:45 cutoff, 15:00-15:45 (on 13.11 a 5m/1m long at 15:25 was stopped at 15:40, -1R) - little time left
           # before the forced exit at 16:05 and the close's own swings
           "xreopen": [(0, 1080), (1140, 1440)], "xlate": [(0, 900), (945, 1440)]}


def filt_ok(cfg, cost, risk, t_ns):
    mc = cfg.get("MAXC")
    if mc and cost / risk > float(mc):
        return False
    se = cfg.get("SESS")
    if se and se in SESS_NY:
        t = pd.Timestamp(int(t_ns), tz="UTC").tz_convert("America/New_York")
        m = t.hour * 60 + t.minute
        if not any(a <= m < b for a, b in SESS_NY[se]):
            return False
    if se and se in SESS_H:
        h = pd.Timestamp(int(t_ns), tz="UTC").hour
        a, b = SESS_H[se]
        if not (a <= h < b):
            return False
    return True


def stop_pad(cfg, px):
    """SPAD (the user's note on a trade, 13.11.2026: "maybe a slightly wider stop in such a sharp drop would have kept
    us in"): extra room beyond the 2nd swing as a share of the entry price (0.002 = 0.2%), on top of the 2 points /
    pips. Off unless set (0 / absent = the rules as before, the same configuration names)."""
    x = cfg.get("SPAD")
    return float(x) * abs(float(px)) if x else 0.0


def min_stop(D, cfg, htf, t_ns, px, stop, d):
    """MINSTOP (bot 5's smart agent, 14.11.2026, from the user's question on sibling trades: on 13.11 US500 and US100
    took the same 4-hour impulse; US500's stop beyond the swing was 0.37% away and was hit, US100's 1.3% held - "should
    the tight one have had room?"): the stop at least MINSTOP x the ATR(14) of the high timeframe's last closed candle
    away from the entry, by the market's own volatility (the user's rule: stops by volatility and structure) - a stop
    already wider is left as it is. The risk per trade stays the same (a smaller size), MIN_RR is checked after it.
    Off unless set (0 / absent = the rules as before, the same configuration names); the live bot runs the same."""
    k = cfg.get("MINSTOP")
    if not k:
        return stop
    ts, _, _, _, _, atr = D.tf(htf)
    j = int(np.searchsorted(ts, int(t_ns) - TF[htf] * 60_000_000_000, side="right")) - 1
    if j < 0 or not np.isfinite(atr[j]):
        return stop
    room = float(k) * float(atr[j])
    return px - d * room if d * (px - stop) < room else stop


def bbw_mask(D, cfg, htf, ltf):
    """BBW "l20" / "h20": the Bollinger-width filter on the entry (l) / high (h) timeframe at the 20th percentile; None
    when off (absent = the rules as before, the same configuration names)."""
    b = cfg.get("BBW")
    if not b or b == "none":
        return None
    return D.bbw_ok(ltf if str(b)[0] == "l" else htf, int(str(b)[1:]) / 100)


def news_x(D, cfg):
    """NEWSX 15 / 60 (user, 14.11.2026, the news guard for the gap): no entry within +-X minutes of a red release of
    the instrument's currencies, and an open trade is closed at the open of the window's first minute (X before the
    release) - the gap trade is not left open into the news. None when off or the calendar is missing."""
    x = cfg.get("NEWSX")
    return D.cal.get(f"cal{int(x)}") if x else None


def shut_of(D, cfg, f, e):
    """The forced-close mask from bar f to e (the world's own closes, NEWSX's window starts, WKND's pre-break window)."""
    sh = D.shut[f:e].copy()
    nx = news_x(D, cfg)
    if nx is not None:
        sh |= nx[f:e]
    wk = wknd(cfg)
    if wk:
        sh |= D.pre_break(wk)[f:e]
    return sh


# FTMO's rule (user, 14.11.2026: "it breaks FTMO's rules"): in a funded Standard account no position is held and no order
# waits over a market break of 2+ hours (the weekend, a holiday, a daily close). WKND = minutes before the break: every
# gap trade is closed at the open of the window's first minute, every waiting gap order is cancelled (not placed again
# after the break), and no entry inside the window. Bot 5 runs it on every gap track (ftmo_cfg, the default below);
# WKND 0 / absent = the rules as before (the same configuration names, so the old runs are the "without" side).
WKND_MIN = 10


def wknd(cfg):
    x = cfg.get("WKND")
    return int(x) if x else 0


def ftmo_cfg(cfg, minutes=WKND_MIN):
    """The track's rules with FTMO's break rule on (WKND), unless the rules set WKND themselves."""
    c = dict(cfg)
    if "WKND" not in c:
        c["WKND"] = int(minutes)
    return _norm(c)


def cancel_bar(D, cfg, q0):
    """WKND: the first bar at or after q0 where a waiting order is cancelled (len(ns) when none / off)."""
    wk = wknd(cfg)
    if not wk:
        return len(D.ns)
    ca = D.cancel_at(wk)
    i = int(np.searchsorted(ca, q0))
    return int(ca[i]) if i < len(ca) else len(D.ns)


def tp_way(cfg):
    """EXIT "tp30" (user, 13.11.2026: "take 100% of the profit at 30% of the way to the target"): the target moves to
    30% of the way; None for every other EXIT."""
    m = cfg.get("EXIT", "none")
    return int(m[2:]) / 100 if m.startswith("tp") else None


# laddered entries (user, 13.11.2026: "instead of one entry in the middle of the gap - several small entries across the
# gap's box, the same total risk, the same stop and target"). LADDER = the levels as a share of the box from its near
# edge (0, the side the price reaches first) to its far edge (1), and each level's share of the track's risk; a level
# beyond the stop is dropped and the shares of the rest scale up to the whole risk. The trade starts at the first
# level's touch; the stop, the target, MIN_RR and the filters are those of the middle of the gap (the trade of
# LADDER none), so only the fills differ. A deeper level fills only while the trade runs and before the management
# level (EXIT) is reached - then the waiting levels are cancelled; each filled level has its own size (its share of
# the risk over its own distance to the stop), its own part out and its own stop to ITS entry. Each fill pays its own
# spread + commission (FTMO's real cost per unit, so the total cost is the same per unit traded). Off unless set.
LADDERS = {"2": ([0.25, 0.75], [0.5, 0.5]), "2n": ([0.0, 0.5], [0.5, 0.5]),
           "3": ([0.0, 0.5, 1.0], [1 / 3, 1 / 3, 1 / 3]), "3m": ([0.0, 0.5, 1.0], [0.25, 0.5, 0.25]),
           "4": ([0.0, 1 / 3, 2 / 3, 1.0], [0.25, 0.25, 0.25, 0.25]), "4m": ([0.0, 1 / 3, 2 / 3, 1.0], [0.2, 0.3, 0.3, 0.2])}


def ladder_of(cfg):
    m = cfg.get("LADDER")
    return LADDERS.get(str(m)) if m and m != "none" else None


def ladder_legs(lad, d, near, far, stop, risk_ref):
    """[(price, share of the risk)] of the levels on the right side of the stop (at least a tenth of the middle's
    distance from it), the shares scaled up to 1; [] when none is left."""
    legs = [(near + x * (far - near), w) for x, w in zip(*lad)]
    legs = [(p, w) for p, w in legs if d * (p - stop) >= 0.1 * risk_ref]
    tot = sum(w for _, w in legs)
    return [(p, w / tot) for p, w in legs] if tot > 0 else []


def ladder_trade(D, f, d, legs, stop, tgt, ref, cfg, e):
    """One laddered trade from its first fill at bar f, looked at up to bar e (exclusive). The whole trade ends at the
    first stop / target / forced close (the same bars as the single entry); each level's result in R of the track's
    whole risk. Returns None while it is still open at e when e is the data's end handled by the caller - the dict has
    end (bar offset of the last exit, or None if open), r (closed) / r_now (open), why, entry (the filled levels'
    average price, by share), filled, cost_r, moved (a part was taken / the stops moved)."""
    ns, o1, h1, l1, c1 = D.ns, D.o1, D.h1, D.l1, D.c1
    hs, ls = h1[f:e], l1[f:e]
    big = e - f + 1
    st = (ls <= stop) if d > 0 else (hs >= stop)
    tg = (hs >= tgt) if d > 0 else (ls <= tgt)
    tg[0] = False
    sh = shut_of(D, cfg, f, e)
    sh[0] = False
    i_st = int(np.argmax(st)) if st.any() else big
    i_tg = int(np.argmax(tg)) if tg.any() else big
    i_sh = int(np.argmax(sh)) if sh.any() else big
    i = min(i_st, i_tg, i_sh)
    em = exit_mode(cfg)
    i_lv, lvl, frac = big, None, 0.0
    if em is not None:
        frac, way = em
        lvl = ref + way * (tgt - ref)
        lv = (hs >= lvl) if d > 0 else (ls <= lvl)
        lv[0] = False
        i_lv = int(np.argmax(lv)) if lv.any() else big
        if i_lv >= i:
            i_lv = big
    out = {"r": 0.0, "r_now": 0.0, "cost_r": 0.0, "filled": 0, "end": 0, "open": False, "moved": i_lv < big,
           "why": None, "entry": 0.0, "whys": []}
    wsum = 0.0
    for j, (lp, w) in enumerate(legs):
        if j == 0:
            fi = 0
        else:
            hit = (ls <= lp) if d > 0 else (hs >= lp)
            if not hit.any():
                continue
            fi = int(np.argmax(hit))
            # a level fills while the trade runs: before the management level, and before the end - on the stop's bar
            # it fills (the price went through it to the stop: careful), on the target's / a forced close's not
            if fi >= i_lv or fi > i or (fi == i and i != i_st) or D.late[f + fi]:
                continue
        q = f + fi
        px = (min(lp, o1[q]) if d > 0 else max(lp, o1[q])) if fi else (min(lp, o1[f]) if d > 0 else max(lp, o1[f]))
        risk = d * (px - stop)
        if risk <= 0:
            continue
        cost = float(D.cost[q]) / risk
        part, ie, why = 0.0, i, None
        if i_lv < big and fi < i_lv:
            part = frac * d * (lvl - px) / risk
            be = (ls[i_lv + 1:] <= px) if d > 0 else (hs[i_lv + 1:] >= px)
            i_be = i_lv + 1 + int(np.argmax(be)) if be.any() else big
            if i_be < big and i_be <= i:              # not reached (big) while the trade is open at e: still open
                ie, why = i_be, "breakeven"
                ex = min(px, o1[f + ie]) if d > 0 else max(px, o1[f + ie])
        keep = 1 - frac if part else 1.0
        if why is None:
            if i >= big:
                out["open"] = True
                out["r_now"] += w * (part + keep * d * (c1[e - 1] - px) / risk - cost)
                out["filled"] += 1
                out["entry"] += w * px
                wsum += w
                out["cost_r"] += w * cost
                continue
            qx = f + i
            if i == i_st:
                ex, why = ((min(stop, o1[qx]) if d > 0 else max(stop, o1[qx])) if i else stop), "stop"
            elif i == i_tg:
                ex, why = (max(tgt, o1[qx]) if d > 0 else min(tgt, o1[qx])), "target"
            else:
                ex, why = o1[qx], "session"
        r_leg = w * (part + keep * d * (ex - px) / risk - cost)
        out["r"] += r_leg
        out["r_now"] += r_leg
        out["filled"] += 1
        out["entry"] += w * px
        wsum += w
        out["cost_r"] += w * cost
        out["end"] = max(out["end"], ie)
        out["whys"].append(why)
    if not wsum:
        return None
    out["entry"] /= wsum
    w_ = out["whys"]
    out["why"] = None if out["open"] else ("breakeven" if w_ and all(x == "breakeven" for x in w_) else
                                           next((x for x in w_ if x != "breakeven"), "breakeven"))
    if out["open"]:
        out["end"] = None
    return out


def simulate(D, pair, cfg, gcache, keep=None):
    htf, ltf, _ = pair
    g = gaps(D, htf, cfg["GBODY"], cfg["MIN_GAP"], gcache)
    hi_sw, lo_sw = D.swings(ltf, cfg["SW"])
    ns, o1, h1, l1, c1 = D.ns, D.o1, D.h1, D.l1, D.c1
    n1 = len(ns)
    exp_ns = int(cfg["EXPIRY"] * 3600 * 1e9)
    prot = cfg["PROT"]
    calk = "cal60" if prot == "cal60" else "cal15" if prot in ("cal15", "cal15shock") else None
    shock = prot in ("shock", "cal15shock")
    trend = D.trend(htf, cfg.get("TREND", "none"), ltf)
    bbw, newsx = bbw_mask(D, cfg, htf, ltf), news_x(D, cfg)
    lad = ladder_of(cfg)
    cands = []
    for d, mid, near, far, made, armed in g:
        q0 = int(np.searchsorted(ns, armed))
        q1 = min(n1, int(np.searchsorted(ns, armed + exp_ns)))      # from the impulse's end (the order's placing)
        q1 = min(q1, cancel_bar(D, cfg, q0))                          # WKND: cancelled before a 2+ hour break
        if q0 >= q1:
            continue
        lv0 = (near + lad[0][0] * (far - near)) if lad else mid       # LADDER: the first level's touch starts the trade
        seg = (l1[q0:q1] <= lv0) if d > 0 else (h1[q0:q1] >= lv0)
        if not seg.any():
            continue
        f = q0 + int(np.argmax(seg))
        if D.late[f]:
            continue                                     # the world's no-entry time (session)
        if trend is not None and trend[f] != d:
            continue                                     # TREND: against the bigger picture
        if bbw is not None and not bbw[f]:
            continue                                     # BBW: a squeeze (Bollinger width under its percentile)
        if newsx is not None and newsx[f]:
            continue                                     # NEWSX: inside a red release's window
        if calk:
            if D.cal[calk][f] or D.cal_start[calk][f] - (D.cal_start[calk][q0 - 1] if q0 else 0) > 0:
                continue                                 # a release window: no entry, the waiting order was cancelled
        if shock:
            cs = D.cs_dn if d > 0 else D.cs_up
            if cs[f] - (cs[q0 - 1] if q0 else 0) > 0:
                continue                                 # a shock against the gap while it waited: cancelled
            if D.recent_shock[f] or ((l1[f] < far) if d > 0 else (h1[f] > far)):
                continue                                 # not a calm correction
        px = (min(mid, o1[f]) if d > 0 else max(mid, o1[f])) if not lad else mid   # LADDER: the middle's geometry
        sw_st, sw_tg = (lo_sw, hi_sw) if d > 0 else (hi_sw, lo_sw)
        s2 = _count_back(*sw_st, ns[f], px, d, -d)
        t2 = _count_back(*sw_tg, ns[f], px, d, d)
        if s2 is None or t2 is None:
            continue
        stop = min_stop(D, cfg, htf, ns[f], px, s2 - d * (D.buf + stop_pad(cfg, px)), d)
        risk = d * (px - stop)
        if risk <= 0 or d * (t2 - px) / risk < cfg["MIN_RR"]:
            continue
        if not filt_ok(cfg, D.cost[f], risk, ns[f]):
            continue
        if tp_way(cfg):
            t2 = px + tp_way(cfg) * (t2 - px)
        legs = ladder_legs(lad, d, near, far, stop, risk) if lad else None
        if lad and not legs:
            continue
        cands.append((f, d, px, stop, t2, risk, armed, legs))
    cands.sort(key=lambda x: x[:7])
    rows, busy, dead, imp1 = [], -1, set(), cfg.get("IMP1") is True
    for f, d, px, stop, tgt, risk, armed, legs in cands:
        if f < busy or (imp1 and (armed, d) in dead):
            continue
        if legs:
            for span in (4 * 1440, 40 * 1440):
                e = min(n1, f + span)
                lt = ladder_trade(D, f, d, legs, stop, tgt, px, cfg, e)
                if lt is None or not lt["open"] or e >= n1:
                    break
            if lt is None:
                continue
            if lt["open"]:
                break                                    # still open at the data's end
            if lt["why"] == "stop":
                dead.add((armed, d))
            rows.append({"time": pd.Timestamp(ns[f], tz="UTC").isoformat(), "dir": d, "r": round(lt["r"], 4),
                         "cost_r": round(lt["cost_r"], 4), "rr": round(d * (tgt - px) / risk, 2), "why": lt["why"],
                         "mins": lt["end"]})
            busy = f + lt["end"] + 1
            continue
        for span in (4 * 1440, 40 * 1440):              # most trades end within a day; the rest within weeks
            e = min(n1, f + span)
            hs, ls = h1[f:e], l1[f:e]
            st = (ls <= stop) if d > 0 else (hs >= stop)
            tg = (hs >= tgt) if d > 0 else (ls <= tgt)
            tg[0] = False
            sh = shut_of(D, cfg, f, e)
            sh[0] = False
            big = e - f + 1
            i_st = int(np.argmax(st)) if st.any() else big
            i_tg = int(np.argmax(tg)) if tg.any() else big
            i_sh = int(np.argmax(sh)) if sh.any() else big
            i = min(i_st, i_tg, i_sh)
            if i < big or e >= n1:
                break
        if i >= big:
            break                                        # still open at the data's end
        em = exit_mode(cfg)
        part_r, frac = 0.0, 0.0
        # PYR (user, 14.11.2026: "add to a winning trade"): at +PYR R the trade adds half its size (a stop order, filled
        # at the level or the minute's open if worse, its own spread + commission) and the stop of the whole position
        # moves to entry - R/3, so a stop then loses exactly the original 1R (1 x -1/3 + 0.5 x -4/3) - the risk never
        # grows. Only when the target is beyond the add level, and only when the add comes before EXIT's management
        # level (then the management of EXIT is not used - the pyramid replaces it). Off unless set; the test only
        # (the live bot is built after an approval).
        pyr = None
        pk = cfg.get("PYR")
        if pk and d * (tgt - px) > float(pk) * risk:
            la = px + d * float(pk) * risk
            ad = (hs >= la) if d > 0 else (ls <= la)
            ad[0] = False
            i_ad = int(np.argmax(ad)) if ad.any() else big
            i_lv0 = big
            if em is not None:
                lv0 = px + em[1] * (tgt - px)
                lvm = (hs >= lv0) if d > 0 else (ls <= lv0)
                lvm[0] = False
                i_lv0 = int(np.argmax(lvm)) if lvm.any() else big
            if i_ad < i and i_ad <= i_lv0:
                pa = max(la, o1[f + i_ad]) if d > 0 else min(la, o1[f + i_ad])
                nstop = px - d * risk / 3
                st2 = (ls <= nstop) if d > 0 else (hs >= nstop)
                st2[:i_ad] = False                        # from the add's own minute (careful)
                i_st = int(np.argmax(st2)) if st2.any() else big
                i = min(i_st, i_tg, i_sh)
                stop, pyr, em = nstop, (pa, float(D.cost[f + i_ad]) / risk), None
        if em is not None:
            frac, way = em
            lvl = px + way * (tgt - px)
            lv = (hs >= lvl) if d > 0 else (ls <= lvl)
            lv[0] = False
            i_lv = int(np.argmax(lv)) if lv.any() else big
            if i_lv < i:                                  # the level came first: part out, the rest's stop to the entry
                part_r = frac * way * d * (tgt - px) / risk
                be = (ls[i_lv + 1:] <= px) if d > 0 else (hs[i_lv + 1:] >= px)
                i_be = i_lv + 1 + int(np.argmax(be)) if be.any() else big
                if i_be <= i:                             # same minute as the stop / target: the entry first (careful)
                    i, i_st = i_be, i_be
                    stop = px                             # the rest leaves at the entry (or the bar's open if worse)
        q = f + i
        if i == i_st:
            ex, why = ((min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if i else stop), "stop"
        elif i == i_tg:
            ex, why = (max(tgt, o1[q]) if d > 0 else min(tgt, o1[q])), "target"
        else:
            ex, why = o1[q], "session"
        if part_r and why == "stop" and stop == px:
            why = "breakeven"
        if why == "stop":
            dead.add((armed, d))                          # IMP1: the impulse's other gaps are cancelled
        cost_r = D.cost[f] / risk
        r_rest = d * (ex - px) / risk
        r_all = (part_r + (1 - frac) * r_rest) if part_r else r_rest
        if pyr is not None:                               # PYR: the added half from its own fill
            r_all += 0.5 * d * (ex - pyr[0]) / risk
            cost_r += 0.5 * pyr[1]
        rows.append({"time": pd.Timestamp(ns[f], tz="UTC").isoformat(), "dir": d, "r": round(r_all - cost_r, 4),
                     "cost_r": round(cost_r, 4), "rr": round(d * (tgt - px) / risk, 2), "why": why, "mins": i})
        if keep is not None:                              # the trade's path (shared_exit): where it was each minute
            keep.append({"f": int(f), "q": int(q), "d": int(d), "px": float(px), "risk": float(risk), "cost": float(cost_r),
                         "lv": int(f + i_lv) if part_r else None, "part": float(part_r), "frac": float(frac)})
        busy = q + 1
    return pd.DataFrame(rows, columns=["time", "dir", "r", "cost_r", "rr", "why", "mins"])


# ---------------------------------------------------------------- the nightly run, cached per (pair, variant, instrument)
def run(bot, symbols, load, out_dir, budget_s, end=None):
    """load(symbol) -> Data or None. Runs what is missing or older than RERUN_DAYS until the budget ends.
    Returns the index of the runs."""
    out_dir = Path(out_dir)
    idx_dir = out_dir / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx = {f.stem: json.loads(f.read_text()) for f in idx_dir.glob("*.json")}
    now = pd.Timestamp.now(tz="UTC")
    todo = []
    for pi, p in enumerate(PAIRS):
        for knob, val, cfg in variants():
            for s in symbols:
                key = f"{p[0]}_{p[1]}_{vid(cfg)}_{s}"
                age = (now - pd.Timestamp(idx[key]["at"])).days if key in idx else 999
                if age >= RERUN_DAYS:
                    todo.append((s, 0 if knob == "base" else 1, pi, -age, p, knob, val, cfg, key))
    # one instrument's minutes in memory at a time; within it the bases first, then the user's order of pairs
    todo.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    t0, cur, D, gc, done = time.time(), None, None, {}, 0
    for s, _, _, _, p, knob, val, cfg, key in todo:
        if time.time() - t0 > budget_s:
            break
        if s != cur:
            cur, gc = s, {}
            try:
                D = load(s)
            except Exception as e:  # noqa: BLE001
                print(f"tal_gap {s}: no data ({e})")
                D = None
        if D is None:
            continue
        ts = time.time()
        t = simulate(D, p, cfg, gc)
        f = out_dir / f"{key}.csv.gz"
        t.to_csv(f, index=False)
        idx[key] = {"at": now.isoformat(timespec="seconds"), "pair": pair_name(p), "gap": p[2], "knob": knob, "val": val,
                    "cfg": cfg, "sym": s, "file": f.name, "n": len(t), "secs": round(time.time() - ts, 1)}
        (idx_dir / f"{key}.json").write_text(json.dumps(idx[key], default=str))
        done += 1
        print(f"  tal_gap {pair_name(p)} {knob}={val} {s}: {len(t)} trades ({time.time() - ts:.1f}s)")
    left = len(todo) - done
    if left == 0 and time.time() - t0 < budget_s:
        done += search(symbols, load, out_dir, idx, idx_dir, budget_s - (time.time() - t0))
    print(f"tal_gap: {done} run, {left} left")
    return idx, left


# ---------------------------------------------------------------- the never-ending search (user, 12.11.2026)
# "the agents optimise every strategy, timeframe and partial profit, every time trying to raise the profit, all the
# time - even if it means entering every trade with a gap; what matters is more profit". So once the planned grid is
# done, every run keeps searching: from the best configuration of a pair (by the TOTAL profit on the train period -
# more trades that each earn a little beat few that earn a lot), it tries one neighbour after another (one knob moved,
# combinations included) and keeps going from whichever earns more. The unseen period is never used to choose - it
# only shows the result, so the search cannot fit 2026.
SEARCH_LOG = "search.json"


def score(t):
    """Total time-weighted R on the train period (what the search maximises)."""
    tr = t[pd.to_datetime(t["time"], utc=True, format="ISO8601") < SPLIT]
    if len(tr) < 30:
        return -1e9
    w = K.weights(tr["time"])
    return float(np.sum(w * tr["r"].to_numpy(float)))


def _cfg_runs(p, cfg, symbols, idx):
    return [v for v in idx.values() if v["pair"] == pair_name(p) and v["cfg"] == cfg and v["sym"] in symbols]


def _norm(c):
    for k in ("EXIT", "IMP1", "TREND", "LADDER", "BBW", "NEWSX", "PYR"):
        if c.get(k) == "none":
            c.pop(k)                                    # "none" is the base: the same configuration, the same name
    if "WKND" in c and not c["WKND"]:
        c.pop("WKND")                                   # WKND 0 = off: the rules as before, the same name
    return c


def _neighbours(best, keys):
    """One knob moved (every knob, every value), then two knobs moved together."""
    vals = {k: KNOBS[k] + [BASE.get(k, "none")] for k in keys}
    for k in keys:
        for v in vals[k]:
            yield _norm(dict(best, **{k: v}))
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            for v1 in vals[k1]:
                if v1 == best.get(k1, "none"):
                    continue
                for v2 in vals[k2]:
                    if v2 == best.get(k2, "none"):
                        continue
                    yield _norm(dict(best, **{k1: v1, k2: v2}))


# beyond the grid (manager, 13.11.2026: on 13.11 every one of the 12 pairs of bot 5 reached a best whose 450 one- and
# two-knob neighbours were all tried - "1758 run, 0 left" - and the search stood still; the user: no agent stops).
# The best sits at the grid's edge on most pairs (EXPIRY 120, MIN_RR 2.0, SW 5), so the numbers go further, and
# in-between values are tried; then random three-knob moves around the best (seeded by the best, so a run that is
# stopped goes on with the same order). Only numeric knobs widen - the live bot reads them as numbers.
WIDE = {"GBODY": [0.5, 2.5, 3.0], "MIN_GAP": [0.15, 0.45, 1.0], "SW": [1, 4, 8, 12],
        "MIN_RR": [0.25, 0.75, 1.25, 2.5, 3.0, 4.0], "EXPIRY": [4, 72, 240, 480]}
RANDOM_MOVES = 3000


def _moves(best, keys):
    """The grid's neighbours, then one / two knobs moved to the wider values, then random three-knob moves."""
    yield from _neighbours(best, keys)
    allv = {k: KNOBS[k] + [BASE.get(k, "none")] + WIDE.get(k, []) for k in keys}
    wk = [k for k in keys if k in WIDE]
    for k in wk:
        for v in WIDE[k]:
            yield _norm(dict(best, **{k: v}))
    for i, k1 in enumerate(wk):
        for k2 in keys:
            if k2 == k1 or (k2 in wk and keys.index(k2) < i):
                continue
            for v1 in WIDE[k1]:
                for v2 in allv[k2]:
                    if v2 == best.get(k2, "none"):
                        continue
                    yield _norm(dict(best, **{k1: v1, k2: v2}))
    import random
    rng = random.Random(vid(best))
    for _ in range(RANDOM_MOVES):
        c = dict(best)
        for k in rng.sample(keys, 3):
            c[k] = rng.choice(allv[k])
        yield _norm(c)


def search(symbols, load, out_dir, idx, idx_dir, budget_s):
    out_dir = Path(out_dir)
    t0, done = time.time(), 0
    logf = out_dir / SEARCH_LOG
    try:
        log = json.loads(logf.read_text())
    except Exception:  # noqa: BLE001
        log = {}
    data, gcs = {}, {}

    def total(p, cfg):
        """Runs what is missing for (pair, cfg) on every instrument; returns the train total, or None (out of time)."""
        nonlocal done
        frames = []
        for s in symbols:
            key = f"{p[0]}_{p[1]}_{vid(cfg)}_{s}"
            if key not in idx:
                if time.time() - t0 > budget_s:
                    return None
                if s not in data:
                    try:
                        data[s] = load(s)
                    except Exception:  # noqa: BLE001
                        data[s] = None
                if data[s] is None:
                    continue
                ts = time.time()
                t = simulate(data[s], p, cfg, gcs.setdefault((s, p[0]), {}))
                f = out_dir / f"{key}.csv.gz"
                t.to_csv(f, index=False)
                idx[key] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "pair": pair_name(p), "gap": p[2],
                            "knob": "search", "val": vid(cfg), "cfg": cfg, "sym": s, "file": f.name, "n": len(t),
                            "secs": round(time.time() - ts, 1)}
                (idx_dir / f"{key}.json").write_text(json.dumps(idx[key], default=str))
                done += 1
            frames.append(pd.read_csv(out_dir / idx[key]["file"]))
        t = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "r"])
        return score(t)

    keys = list(KNOBS)
    while time.time() - t0 < budget_s:
        moved = False
        for p in PAIRS:
            st = log.setdefault(pair_name(p), {"best": dict(BASE), "score": None, "tried": []})
            if st["score"] is None:
                sc = total(p, st["best"])
                if sc is None:
                    break
                st["score"] = sc
            # the next untried neighbour of the best: one knob moved, round robin; once every one-knob neighbour was
            # tried, two knobs moved together (manager, 12.11.2026: the one-knob search reached a local best on
            # pairs and stopped - the user: no agent stops searching)
            nb = None
            tried = set(st["tried"])
            for c in _moves(st["best"], keys):
                if c != st["best"] and vid(c) not in tried:
                    nb = c
                    break
            if nb is None:
                continue                               # a local best: every neighbour tried
            sc = total(p, nb)
            if sc is None:
                break
            st["tried"].append(vid(nb))
            # better by more than the drift of the time weights (they move with the clock, so the same trades scored
            # minutes later score a hair higher: 13.11.2026 the search swung between two equal versions for ever);
            # the version it leaves counts as tried, so it never swings back
            if sc > st["score"] + 1e-3 + 1e-3 * abs(st["score"]):
                old = vid(st["best"])
                st["best"], st["score"], st["tried"] = nb, sc, [old]   # better: search around it from now on
                print(f"  tal_gap search {pair_name(p)}: better total {sc:.1f}R with {nb}")
            moved = True
            logf.write_text(json.dumps(log, default=str))
        if not moved:
            break
    logf.write_text(json.dumps(log, default=str))
    return done


def _stats(t):
    tr, un = t[t["time"] < SPLIT], t[t["time"] >= SPLIT]
    return {"train": K.wstats(tr["r"], K.weights(tr["time"])), "unseen": K.wstats(un["r"], K.weights(un["time"])),
            "whole": K.wstats(t["r"], K.weights(t["time"])), "halves": K.halves(t["time"], t["r"]),
            "cost_r": round(float(t["cost_r"].mean()), 3) if len(t) else None,
            "target_pct": round(float((t["why"] == "target").mean()) * 100, 1) if len(t) else None,
            "rr": round(float(t["rr"].median()), 2) if len(t) else None}


def summarize(idx, symbols, out_dir):
    """Per pair: the base, every variant, the one chosen on the train period, shown on the unseen one; and the open
    questions of tal_gap.md answered from the data (the knob values that win on train across the pairs)."""
    out_dir = Path(out_dir)

    def trades(p, cfg):
        fs = [v for v in idx.values() if v["pair"] == pair_name(p) and v["cfg"] == cfg]
        if {v["sym"] for v in fs} != set(symbols):
            return None
        t = pd.concat([pd.read_csv(out_dir / v["file"]) for v in fs], ignore_index=True)
        t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        return t
    pairs, votes = {}, {}
    for p in PAIRS:
        rows = []
        for knob, val, cfg in variants():
            t = trades(p, cfg)
            if t is None:
                continue
            rows.append({"knob": knob, "val": val, "cfg": cfg, **_stats(t)})
        seen = {vid(r["cfg"]) for r in rows}
        for v in idx.values():                          # the never-ending search's configurations
            if v.get("knob") == "search" and v["pair"] == pair_name(p) and vid(v["cfg"]) not in seen:
                t = trades(p, v["cfg"])
                seen.add(vid(v["cfg"]))
                if t is not None:
                    rows.append({"knob": "search", "val": vid(v["cfg"]), "cfg": v["cfg"], **_stats(t)})
        if not rows:
            continue
        base = next((r for r in rows if r["knob"] == "base"), None)
        ok = [r for r in rows if (r["train"]["eff_n"] or 0) >= K.MIN_EFF_N and r["train"]["mean"] is not None]
        # by the TOTAL profit on the train period (user, 12.11.2026: "what matters is more profit")
        chosen = max(ok, key=lambda r: r["train"]["mean"] * r["train"]["eff_n"]) if ok else None
        prot = {r["val"] if r["knob"] == "PROT" else "none": {"train": r["train"]["mean"], "unseen": r["unseen"]["mean"],
                                                             "n": r["whole"]["n"]}
                for r in rows if r["knob"] in ("PROT", "base")}
        pairs[pair_name(p)] = {"gap": p[2], "name": f"{TF_HE[p[0]]} / {TF_HE[p[1]]}", "done": min(len(rows), len(variants())),
                               "planned": len(variants()), "searched": sum(1 for r in rows if r["knob"] == "search"), "base": base, "chosen": chosen, "protection": prot,
                               "variants": [{k: r[k] for k in ("knob", "val", "train", "unseen", "whole")} for r in rows]}
        if base and len([r for r in rows if r["knob"] != "search"]) == len(variants()):
            for knob in KNOBS:
                lst = [r for r in rows if r["knob"] == knob] + [dict(base, val=BASE.get(knob, "none"))]
                lst = [r for r in lst if r["train"]["mean"] is not None]
                if lst:
                    b = max(lst, key=lambda r: r["train"]["mean"])
                    votes.setdefault(knob, []).append((b["val"], b["unseen"]["mean"]))
    answers = {}
    for knob, vs in votes.items():
        cnt = pd.Series([str(v) for v, _ in vs]).value_counts()
        answers[knob] = {"name": NAMES[knob], "best_on_train": cnt.index[0], "pairs_voting": int(cnt.iloc[0]), "pairs": len(vs)}
    return pairs, answers


def search_status(out_dir):
    """Per pair: the search's best total (train), its configuration's changes from the base, neighbours tried / left -
    so the manager sees from the repo whether the search still moves (13.11.2026: search.json stays on the computer)."""
    try:
        log = json.loads((Path(out_dir) / SEARCH_LOG).read_text())
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for pn, st in log.items():
        best = st.get("best") or {}
        tried = set(st.get("tried", []))
        left = len({vid(c) for c in _moves(best, list(KNOBS)) if c != best and vid(c) not in tried})
        out[pn] = {"score": None if st.get("score") is None else round(float(st["score"]), 2),
                   "changes": {k: v for k, v in best.items() if BASE.get(k, "none") != v},
                   "tried": len(st.get("tried", [])), "left": left}
    return out


def write_results(bot, pairs, answers, world, progress, out_dir=None):
    """This bot's section of data/agents/strategies/tal_gap_results.json, next to the other bot's (read just before
    writing, so the other agent's section is kept)."""
    try:
        doc = json.loads(RESULTS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        doc = {}
    doc["note"] = ("שיטת הגאפ של טל (tal_gap.md) - תוצאות לכל בוט בעולם שלו. נבחר על התקופה עד 31.12.2025, מוצג על 2026 "
                   "שלא נראתה, משוקלל לפי זמן. R לעסקה אחרי עלויות.")
    doc[bot] = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "world": world, "base": BASE,
                "knobs": KNOBS, "progress": progress, "answers": answers, "pairs": pairs}
    if out_dir is not None:
        doc[bot]["search"] = search_status(out_dir)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return doc


def _find(x, cfg):
    """The row of a pair's results that is exactly this configuration (a one-knob variant, the base or a searched one)."""
    want = vid(_norm(dict(cfg)))
    for v in x.get("variants") or []:
        if v["knob"] == "search":
            if v["val"] == want:
                return v
            continue
        c = dict(BASE) if v["knob"] == "base" else dict(BASE, **{v["knob"]: v["val"]})
        if vid(_norm(c)) == want:
            return v
    return None


def cards(bot, pairs, other=None, other_name="בוט 5", risk_pct=None, live=None):
    """A 'new' card per pair whose train-chosen variant made money on the unseen period. live = {pair: cfg} of the
    tracks the bot trades now (13.11.2026): such a pair's card is an 'improve' card with what runs now as its baseline,
    so the manager approves it only if 2026 is better per trade AND in total (agentkit.judge); no card when the chosen
    configuration is already the one running."""
    out = []
    live = live or {}
    for pn, x in pairs.items():
        ch = x.get("chosen")
        if not ch or x["done"] < x["planned"]:
            continue
        u = ch["unseen"]
        if u["mean"] is None or u["mean"] <= 0:
            continue
        cur, base_row = live.get(pn), None
        # a new pair must add to the bot as a whole, not only risk (tal_thinking.md principle 11; 13.11.2026: 5m/1m lost
        # on the train period in every version, 4h/5m earned +0.007R a trade on 2026 - both were approved and add nothing)
        if live and cur is None and ((ch["train"].get("mean") or 0) <= 0 or (u.get("t") or 0) < 1.0):
            continue
        if cur is not None:
            if vid(_norm(dict(cur))) == vid(_norm(dict(ch["cfg"]))):
                continue
            base_row = _find(x, cur)
        try:
            ob = ((other or {}).get("pairs") or {}).get(pn, {}).get("chosen") or {}
            om = (ob.get("unseen") or {}).get("mean")
        except Exception:  # noqa: BLE001 - the other agent's file in another shape
            ob, om = {}, None
        cfg = ch["cfg"]
        same = f" · ב{other_name}: {om:+.2f}R בתקופה שלא נראתה ({ob.get('knob')}={ob.get('val')})" if om is not None \
            else f" · ב{other_name}: עוד לא נבדק"
        knob = ("הבסיס" if ch["knob"] == "base" else "" if ch["knob"] == "PROT" else "חיפוש: " + ", ".join(f"{k}={v}" for k, v in cfg.items())
                if ch["knob"] == "search" else f"{NAMES[ch['knob']]} = {ch['val']}")
        what = (f"שיטת הגאפ של טל, {x['name']} (פער {x['gap']} דרגות): {knob + ', ' if knob else ''}{PROT_HE.get(cfg['PROT'])}. "
                f"נבחר על 2024-2025, מוצג על 2026: {u['mean']:+.2f}R לעסקה, {u['n']} עסקאות, {u['win']}% הצלחה, "
                f"{ch['target_pct']}% הגיעו ליעד, יחס חציוני {ch['rr']}, עלות {ch['cost_r']}R{same}")
        what += f" · סה\"כ על 2026: {u['mean'] * u['n']:+.1f}R"
        if cur is not None:
            bu = (base_row or {}).get("unseen") or {}
            what += (f" · מה שרץ היום בזוג הזה: {bu['mean']:+.2f}R לעסקה, {bu['n']} עסקאות, סה\"כ {bu['mean'] * bu['n']:+.1f}R על 2026"
                     if bu.get("mean") is not None else " · מה שרץ היום בזוג הזה עוד לא נבדק באותה בדיקה")
        c = K.card(bot, "new" if cur is None else "improve", f"שיטת הגאפ של טל · {x['name']}", what, ch["train"], u, whole=ch["whole"],
                   baseline=None if cur is None else ((base_row or {}).get("unseen") or {"mean": None, "n": 0}),
                   halves_=ch["halves"], cost_r=ch["cost_r"], tested=len(PAIRS) * len(variants()), risk_pct=risk_pct,
                   neighbours={"values": [v["val"] for v in x["variants"] if v["knob"] == ch["knob"]],
                               "share_positive": round(np.mean([(v["unseen"]["mean"] or -1) > 0 for v in x["variants"]]), 2)},
                   detail={"other_bot": {"name": other_name, "unseen": om, "cfg": ob.get("cfg")}, "protection": x["protection"]},
                   change={"method": "tal_gap", "pair": pn, "cfg": cfg, "track": f"tal_gap|{pn}", "knob": "cfg"})
        out.append(c)
    return out


# ---------------------------------------------------------------- the bot as a whole (13.11.2026)
DAY_LIM, ALL_LIM = 4.5, 9.0          # the bot's own limits (% of the account; FTMO's 5% / 10% stay above them)


def _lim(lim):
    """The account's limits: bot 5 by default (4.5% a day, 9% from the start, the full room); bot 6 passes its own
    (FTMO Futures, live/fut_account.py: a day loss of DAILY_LOSS, a trailing maximum loss from the best end-of-day
    balance that stops rising at the start, ROOM_USE of the room, the futures' trading day from 18:00 New York, and a
    day that passes the daily loss fails the account - FTMO's hard daily limit)."""
    d = {"day": DAY_LIM, "all": ALL_LIM, "trail": False, "use": 1.0, "dayf": None, "day_hard": False, "contracts": None}
    d.update(lim or {})
    return d


def _contract_risk(part, spec):
    """Bot 6 (14.11.2026): the trade's risk and result per ONE micro contract, in $ - the way live/fut_account.size
    takes it (per_contract = the stop distance x point value + commission + slippage; whole contracts only).
    The trades' files keep no stop distance, but they keep cost_r = the cost in price / the stop distance in price,
    and the cost in price is the symbol's constant (agent_bot6.tal_load), so the distance = cost / cost_r.
    spec = {"syms": {symbol: (point value, tick)}, "comm", "slip"}. A row without a usable cost_r -> NaN (no rounding)."""
    syms, comm, slip = spec["syms"], float(spec["comm"]), float(spec["slip"])
    per, pnl1 = np.full(len(part), np.nan), np.full(len(part), np.nan)
    for i, (sym, cr, r_) in enumerate(zip(part["sym"], part["cost_r"], part["r"])):
        if sym not in syms or not cr or not np.isfinite(cr) or cr <= 0:
            continue
        pv, tick = syms[sym]
        dist = (tick + (comm + slip * tick * pv) / pv) / float(cr)
        per[i] = dist * pv + comm + slip * tick * pv
        pnl1[i] = dist * pv * float(r_)
    return per, pnl1


def account(part, risk_pct, lim=None):
    """The account the way the virtual bot keeps it (live/watch_own.py), in % of the starting balance (no compounding -
    careful): each trade risks its track's % (risk_pct = {track: %}); a result counts when the trade closes ("out").
    After the day's closed loss reaches DAY_LIM no new trade opens that day; and (13.11.2026 evening, the live rule of
    bot 5) a new trade risks at most the room left - the day's closed loss plus every open trade's full risk stays
    within DAY_LIM, and the fall from the start plus the open risk within ALL_LIM; under a quarter of its risk left it is
    not taken. Once the fall from the start reaches ALL_LIM the account stops. Returns total, worst day, deepest fall
    from the peak, the fall from the start, days the daily limit stopped trading, trades cut / skipped by the room rule,
    the largest open risk at once (and the days it passed DAY_LIM before this rule), and when the account stopped.
    lim["contracts"] (bot 6, 14.11.2026): whole micro contracts exactly as live/fut_account.size - see _contract_risk."""
    if part.empty:
        return None
    L = _lim(lim)
    day_lim, all_lim, use_k = float(L["day"]), float(L["all"]), float(L["use"])
    dayf = L["dayf"] or (lambda x: x.strftime("%Y-%m-%d"))
    part = part.sort_values("time")
    rk = part["trk"].map(risk_pct).fillna(0).to_numpy(float)
    rr = part["r"].to_numpy(float)
    times = list(part["time"])
    outs = list(part["out"]) if "out" in part else times
    # whole micro contracts (bot 6, 14.11.2026): at a small risk one contract can be more than the trade may risk - the
    # live account then skips it (fut_account.size "small"), and a size is always rounded down. Without this the test
    # took fractions of a contract, so a factor on the tracks' risk looked better than the account could trade.
    ct = L.get("contracts")
    if ct and {"sym", "cost_r"} <= set(part.columns):
        per1, pnl1 = _contract_risk(part, ct)
        init, max_n = float(ct.get("initial", 100_000.0)), int(ct.get("max_n", 10 ** 9))
    else:
        ct, per1, pnl1 = None, np.full(len(part), np.nan), np.full(len(part), np.nan)
    small = 0
    eq, peak, fall, low, dead = 0.0, 0.0, 0.0, 0.0, None
    day_real, stopped_days, cut, skipped = {}, set(), 0, 0
    opened = []                       # (out, risk %, pnl %, contracts) of the trades taken and not booked yet
    raw_open, raw_peak, raw_days = [], 0.0, set()
    eod = {"day": None, "eq": 0.0, "peak": 0.0}     # the trailing level (bot 6): the best end-of-day balance so far

    def roll(d_):
        if eod["day"] is not None and d_ != eod["day"]:
            eod["peak"] = max(eod["peak"], eod["eq"])
        eod["day"] = d_

    def level():
        return min(eod["peak"] - all_lim, 0.0) if L["trail"] else -all_lim

    def book(upto):
        nonlocal eq, peak, fall, low, dead
        for o in sorted([o for o in opened if o[0] <= upto], key=lambda o: o[0]):
            opened.remove(o)
            d_ = dayf(o[0])
            roll(d_)
            day_real[d_] = day_real.get(d_, 0.0) + o[2]
            eq += o[2]
            eod["eq"] = eq
            peak = max(peak, eq)
            fall = max(fall, peak - eq)
            low = min(low, eq)
            if (eq <= level() or (L["day_hard"] and day_real[d_] <= -day_lim - 1e-9)) and dead is None:
                dead = o[0].isoformat()

    for r_, w_, t_, o_, pc_, p1_ in zip(rr, rk, times, outs, per1, pnl1):
        book(t_)
        # the open risk with no room rule (what the tracks would hold at once): how often alone it passes DAY_LIM
        raw_open = [x for x in raw_open if x[0] > t_] + [(o_, w_)]
        ro = sum(x[1] for x in raw_open)
        raw_peak = max(raw_peak, ro)
        if ro > day_lim:
            raw_days.add(dayf(t_))
        if dead is not None or w_ <= 0:
            continue
        d_ = dayf(t_)
        roll(d_)
        dreal = day_real.get(d_, 0.0)
        if dreal <= -use_k * day_lim:
            stopped_days.add(d_)
            continue
        open_ = sum(o[1] for o in opened)
        room = use_k * min(day_lim + dreal, eq - level()) - open_
        if ct is not None and np.isfinite(pc_):
            # live/fut_account.size: n = min(balance x risk, room) // one contract's risk, at most MAX_MICROS open
            want_usd = init * (1 + eq / 100) * w_ / 100
            n = int(min(want_usd, room * init / 100) // pc_) if room > 0 else 0
            n = min(n, max_n - sum(o[3] for o in opened))
            if n < 1:
                skipped += 1
                small += room * init / 100 >= want_usd      # skipped for the contract's size, not for the room
                continue
            use = n * pc_ / init * 100
            cut += min(want_usd, room * init / 100) < want_usd
            opened.append((o_, use, n * p1_ / init * 100, n))
            continue
        if room < 0.25 * w_:
            skipped += 1
            continue
        use = min(w_, room)
        cut += use < w_
        opened.append((o_, use, r_ * use, 0))
    book(pd.Timestamp.max.tz_localize("UTC"))
    worst = min([0.0] + list(day_real.values()))
    return {"total_pct": round(float(eq), 2), "worst_day_pct": round(float(worst), 2), "max_fall_pct": round(float(fall), 2),
            "low_pct": round(float(low), 2), "daily_stops": len(stopped_days), "room_cut": int(cut), "room_skipped": int(skipped),
            "open_peak_pct": round(float(raw_peak), 2), "open_days_over_day_lim": len(raw_days), "stopped": dead,
            **({"contract_skipped": int(small)} if ct is not None else {})}


MARGIN = 0.8          # the train period must stay within 80% of the bot's limits (a fall <= 7.2%, a day >= -3.6%)
SCALE_FLOOR = 0.1     # the smallest factor (the builder takes 0.1-1.0 for bot 5's "tal_risk_scale")


def risk_scale(t, risk_pct, lim=None):
    """One factor on every track's risk, CHOSEN on 2024-2025 only: the largest (steps of 0.05, at least SCALE_FLOOR) with which
    the train period's deepest fall and worst day stay inside MARGIN of the bot's limits - then shown on 2026.
    A fall is measured from the peak: an account may start on any day, the worst one included."""
    tr, un = t[t["time"] < SPLIT], t[t["time"] >= SPLIT]
    L = _lim(lim)
    a = account(tr, risk_pct, lim)
    if not a:
        return {"chosen": 1.0, "fits": True, "rows": {}}
    ks = [1.0]
    if a["max_fall_pct"] > 0:
        ks.append(MARGIN * L["all"] / a["max_fall_pct"])
    if a["worst_day_pct"] < 0:
        ks.append(MARGIN * L["day"] / -a["worst_day_pct"])
    k = max(SCALE_FLOOR, np.floor(min(ks) * 20) / 20)

    def _fits(b_):
        return (b_["max_fall_pct"] <= MARGIN * L["all"] + 1e-9 and b_["worst_day_pct"] >= -MARGIN * L["day"] - 1e-9
                and not b_["stopped"])
    # an account that failed (or was stuck at its limit, skipping trades) shows a fall no deeper than the limit itself,
    # so the factor from that fall is too big (bot 6, 13.11.2026: x 0.8 chosen, "fits": false, and with it the account
    # still fell to its limit in 2026). Step down by 0.05 until the train period really fits (bot 6's smart agent,
    # 13.11.2026 night) - a factor that fits at once (bot 5) does not change.
    while True:
        rk = {x: v * k for x, v in risk_pct.items()}
        b = account(tr, rk, lim)
        fits = _fits(b)
        if fits or k <= SCALE_FLOOR + 1e-9:
            break
        k = max(SCALE_FLOOR, round(k - 0.05, 2))
    return {"chosen": round(float(k), 2), "fits": bool(fits), "limits": {"day": L["day"], "all": L["all"], "trail": L["trail"], "use": L["use"]},
            "rows": {str(round(float(k), 2)): {"train": b, "unseen": account(un, rk, lim)}}}


def portfolio(idx, symbols, out_dir, live, hour_min_n=100, risk=None, lim=None):
    """All the tracks the bot runs together (live = {track: (pair_name, cfg)}), the way the account takes them: the
    total time-weighted R, the worst day and the deepest fall (in R of one trade), on 2024-2025 and on 2026 - and the
    same with one rule against doubled risk: a track does not enter a symbol in a direction another track already holds
    (12.11.2026: t15_1 and t15_5 took the same US500 short at 15:51, both stopped - one idea, twice the loss).
    Returns None while a track's runs are missing."""
    out_dir, rows = Path(out_dir), []
    for trk, lv in live.items():
        # (pair, cfg) trades on `symbols`; (pair, cfg, [symbols]) is an instrument's own track (13.11.2026 evening:
        # the manager's task f8643b16aa57 - the instrument tracks were outside the whole-bot test)
        pn, cfg = lv[0], lv[1]
        syms_ = list(lv[2]) if len(lv) > 2 and lv[2] else list(symbols)
        fs = [v for v in idx.values() if v["pair"] == pn and _norm(dict(v["cfg"])) == _norm(dict(cfg)) and v["sym"] in syms_]
        fs = list({v["sym"]: v for v in fs}.values())
        if {v["sym"] for v in fs} != set(syms_):
            return None
        for v in fs:
            t = pd.read_csv(out_dir / v["file"])
            t["trk"], t["sym"] = trk, v["sym"]
            rows.append(t)
    if not rows:
        return None
    t = pd.concat(rows, ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    t["out"] = t["time"] + pd.to_timedelta(t["mins"].fillna(0).astype(float) + 1, unit="min")
    t = t.sort_values(["time", "trk"]).reset_index(drop=True)
    keep, held = [], {}
    for i, x in enumerate(t.itertuples()):
        k = (x.sym, int(x.dir))
        held[k] = [e for e in held.get(k, []) if e > x.time]
        keep.append(not held[k])
        if keep[-1]:
            held[k].append(x.out)
    t["solo"] = keep
    # one direction across all the bot's indices (bot 5's smart agent, 13.11.2026): the indices move together - on
    # 12.11 between 14:41 and 16:19 UTC nine shorts of four tracks on US500 / US100 / JP225 were stopped one after the
    # other (-2.6% in under two hours, more than half of the 4.5% daily limit). The rule: at most K trades open in the
    # same direction on all the instruments together; a trade beyond them is not taken.
    caps = {}
    for cap in (1, 2, 3):
        keep, opened = [], {1: [], -1: []}
        for x in t.itertuples():
            d_ = int(x.dir)
            opened[d_] = [e for e in opened[d_] if e > x.time]
            keep.append(len(opened[d_]) < cap)
            if keep[-1]:
                opened[d_].append(x.out)
        caps[cap] = keep

    def stats(d):
        out = {}
        for name, part in (("train", d[d["time"] < SPLIT]), ("unseen", d[d["time"] >= SPLIT])):
            if part.empty:
                out[name] = None
                continue
            w = K.weights(part["time"])
            day = part.groupby(part["time"].dt.strftime("%Y-%m-%d"))["r"].sum()
            cum = part["r"].cumsum()
            out[name] = {"n": int(len(part)), "total_w": round(float(np.sum(w * part["r"].to_numpy(float))), 1),
                         "total": round(float(part["r"].sum()), 1), "worst_day": round(float(day.min()), 2),
                         "days_below_3r": int((day <= -3).sum()), "max_fall": round(float((cum.cummax() - cum).max()), 1)}
        return out
    doubled = t[~t["solo"]]
    # an hour filter (bot 5's smart agent, 13.11.2026 night): the UTC hours of entry where all the tracks together LOST
    # on 2024-2025 (time-weighted mean < 0, t <= -1.5, at least 100 trades) - chosen on that period only - and the
    # account without them on both periods. 2026 only shows the result.
    tr_ = t[t["time"] < SPLIT]
    hrs = []
    for h_, part in tr_.groupby(tr_["time"].dt.hour):
        if len(part) < hour_min_n:      # bot 6 runs far fewer gap trades (2 tracks, 3 indices): it passes 30
            continue
        ws_ = K.wstats(part["r"].to_numpy(float), K.weights(part["time"]))
        if ws_["mean"] is not None and ws_["mean"] < 0 and (ws_["t"] or 0) <= -1.5:
            hrs.append(int(h_))
    hours_off = {"hours": hrs, **(stats(t[~t["time"].dt.hour.isin(hrs)]) if hrs else {})}
    # the same direction at once (13.11.2026 evening): the most trades open together in one direction, per period
    same = {}
    for name, part in (("train", t[t["time"] < SPLIT]), ("unseen", t[t["time"] >= SPLIT])):
        mx = 0
        for d_ in (1, -1):
            ev = sorted([(x, 1) for x in part.loc[part["dir"] == d_, "time"]] + [(x, -1) for x in part.loc[part["dir"] == d_, "out"]],
                        key=lambda e: (e[0], e[1]))
            c_ = 0
            for _, e in ev:
                c_ += e
                mx = max(mx, c_)
        same[name] = mx
    # the account in % with each track's real risk (13.11.2026 night): R totals hide that the tracks risk different
    # shares, and that the account has limits - a deepest fall of 75R at 0.28% a trade is 21%, beyond the 10% line
    acct = None
    if risk:
        rk = {k: float(v) * 100 for k, v in risk.items() if k in live}
        acct = {"risk_pct": rk,
                "now": {"train": account(t[t["time"] < SPLIT], rk, lim), "unseen": account(t[t["time"] >= SPLIT], rk, lim)},
                "scale": risk_scale(t, rk, lim)}
    return {"account": acct, "tracks": {k: v[0] + (f" {','.join(v[2])}" if len(v) > 2 and v[2] else "") for k, v in live.items()},
            "same_dir_max": same, "now": stats(t), "one_per_side": stats(t[t["solo"]]),
            "max_dir": {str(c): stats(t[pd.Series(k_, index=t.index)]) for c, k_ in caps.items()},
            "hours_off": hours_off,
            # R by the hour of entry (UTC) and by weekday, all tracks together (13.11.2026: material for a time filter
            # chosen on 2024-2025 by the smart agent - the losses of 12.11 were all between 14:41 and 16:19 UTC)
            "by_hour": _by(t, t["time"].dt.hour), "by_weekday": _by(t, t["time"].dt.weekday),
            "doubled": {"n": int(len(doubled)), "r": round(float(doubled["r"].sum()), 1),
                        "unseen_n": int((doubled["time"] >= SPLIT).sum()),
                        "unseen_r": round(float(doubled.loc[doubled["time"] >= SPLIT, "r"].sum()), 1)},
            "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}


# ---------------------------------------------------------------- a shared exit (user, 13.11.2026, an idea from a video)
# "When several trades are open together - close all of them at once when the open trades' total profit reaches a
# target." Tested on the bot as it is: every track at its real risk, the same entries, each trade keeps its own stop and
# target; each minute the open trades are marked at the minute's close (net of their cost), and when their total
# reaches the target - in R of the open risk, or in % of the account - with at least MIN open, all of them close at
# that close. Careful: a track that was closed early by it does not take the next trade sooner (the entries stay the
# same), so the test cannot show more trades than the bot as it is.
SHARED_VARIANTS = [("r", x, m) for x in (0.5, 1.0, 1.5, 2.0) for m in (2, 3)] + \
                  [("pct", x, 2) for x in (0.25, 0.5, 1.0)]


def trade_paths(D, keep):
    """Per trade of simulate(..., keep=[]): the minutes' times and the open result in R (net of its cost) each minute
    before its own exit."""
    out = []
    for k in keep:
        f, q = k["f"], k["q"]
        if q <= f:
            out.append((D.ns[f:f], np.zeros(0)))
            continue
        v = k["d"] * (D.c1[f:q] - k["px"]) / k["risk"]
        if k["lv"] is not None and k["lv"] < q:
            j = k["lv"] - f
            v[j:] = k["part"] + (1 - k["frac"]) * v[j:]
        out.append((D.ns[f:q], (v - k["cost"]).astype(np.float32)))
    return out


def shared_exit(items, mode, x, min_open):
    """items: [{"w": risk as a share of the account, "t": minute times (ns), "v": open R each minute, "r": the trade's
    own result}] -> [(r, closed early at ns or None)] in the same order."""
    ts = [it["t"] for it in items if len(it["t"])]
    if not ts:
        return [(it["r"], None) for it in items]
    G = np.unique(np.concatenate(ts))
    n = len(G)
    P, W, N = np.zeros(n), np.zeros(n), np.zeros(n, np.int32)
    span = []
    for it in items:
        if not len(it["t"]):
            span.append(None)
            continue
        a = int(np.searchsorted(G, it["t"][0]))
        b = int(np.searchsorted(G, it["t"][-1], side="right"))
        vg = it["v"][np.searchsorted(it["t"], G[a:b], side="right") - 1].astype(float)
        P[a:b] += it["w"] * vg
        W[a:b] += it["w"]
        N[a:b] += 1
        span.append((a, b, vg))

    def cond(lo, hi):
        if mode == "r":
            return (N[lo:hi] >= min_open) & (P[lo:hi] >= x * W[lo:hi] - 1e-12)
        return (N[lo:hi] >= min_open) & (P[lo:hi] * 100 >= x - 1e-12)
    C = cond(0, n)
    A = np.array([s_[0] if s_ else -1 for s_ in span])
    B = np.array([s_[1] if s_ else -1 for s_ in span])
    closed = np.zeros(len(items), bool)
    res = [(it["r"], None) for it in items]
    pos = 0
    while pos < n:
        k = int(np.argmax(C[pos:]))
        if not C[pos + k]:
            break
        t0 = pos + k
        hit = np.where((A <= t0) & (B > t0) & ~closed)[0]
        hi = t0 + 1
        for j in hit:
            a, b, vg = span[j]
            res[j] = (float(vg[t0 - a]), int(G[t0]))
            closed[j] = True
            if b > t0 + 1:
                P[t0 + 1:b] -= items[j]["w"] * vg[t0 + 1 - a:]
                W[t0 + 1:b] -= items[j]["w"]
                N[t0 + 1:b] -= 1
                hi = max(hi, b)
        C[t0 + 1:hi] = cond(t0 + 1, hi)
        pos = t0 + 1
    return res


def shared_test(t, items, risk_pct, lim=None):
    """t: the bot's trades (time, out, r, trk) in the order of items. The account (account()) as it is and with each
    SHARED_VARIANTS, on 2024-2025 and on 2026; chosen on 2024-2025 only: the most total % whose deepest fall is not
    deeper than the bot's as it is (more profit, not more risk)."""
    t = t.copy().reset_index(drop=True)
    rows = {"now": {"train": account(t[t["time"] < SPLIT], risk_pct, lim), "unseen": account(t[t["time"] >= SPLIT], risk_pct, lim)}}
    ws = t["trk"].map(risk_pct).fillna(0).to_numpy(float) / 100
    for mode, x, m in SHARED_VARIANTS:
        its = [dict(it, w=w_) for it, w_ in zip(items, ws)]
        res = shared_exit(its, mode, x, m)
        u = t.copy()
        u["r"] = [r_ for r_, _ in res]
        early = [e_ for _, e_ in res]
        u["out"] = [pd.Timestamp(e_, tz="UTC") + pd.Timedelta(minutes=1) if e_ is not None else o_ for e_, o_ in zip(early, t["out"])]
        key = f"{mode}{x}_m{m}"
        rows[key] = {"train": account(u[u["time"] < SPLIT], risk_pct, lim), "unseen": account(u[u["time"] >= SPLIT], risk_pct, lim),
                     "closed_early": int(sum(e_ is not None for e_ in early)),
                     "closed_early_unseen": int(sum(e_ is not None for e_, tt in zip(early, t["time"]) if tt >= SPLIT))}
    now = rows["now"]["train"]
    best = None
    for k_, v_ in rows.items():
        if k_ == "now" or not v_["train"] or not now:
            continue
        if v_["train"]["max_fall_pct"] <= now["max_fall_pct"] + 1e-9 and v_["train"]["total_pct"] > now["total_pct"] \
                and (best is None or v_["train"]["total_pct"] > rows[best]["train"]["total_pct"]):
            best = k_
    better = bool(best and rows[best]["unseen"] and rows["now"]["unseen"]
                  and rows[best]["unseen"]["total_pct"] > rows["now"]["unseen"]["total_pct"]
                  and rows[best]["unseen"]["worst_day_pct"] >= rows["now"]["unseen"]["worst_day_pct"])
    return {"rows": rows, "chosen": best, "better": better, "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}


def _by(t, key):
    out = {}
    for k_, part in t.groupby(key):
        tr, un = part[part["time"] < SPLIT], part[part["time"] >= SPLIT]
        out[str(k_)] = {"train_n": int(len(tr)), "train_mean": round(float(tr["r"].mean()), 3) if len(tr) else None,
                        "unseen_n": int(len(un)), "unseen_mean": round(float(un["r"].mean()), 3) if len(un) else None}
    return out


def portfolio_card(bot, pf, risk_pct):
    """A card for one rule against concentrated risk - one track per symbol and direction, or at most K trades open in
    one direction on all the indices together (13.11.2026). The rule is CHOSEN on 2024-2025 only (the lowest deepest
    fall among those that keep >= 95% of the time-weighted total), then shown on 2026: a card only when it lowers the
    risk on both periods without lowering 2026's total."""
    a = pf["now"]
    if not a.get("unseen") or not a.get("train"):
        return None
    rules = [("side", pf.get("one_per_side") or {})] + [(f"dir{c}", v) for c, v in sorted((pf.get("max_dir") or {}).items())]
    at, au = a["train"], a["unseen"]
    ok = [(nm, r) for nm, r in rules if r.get("train") and r.get("unseen") and r["train"]["total_w"] >= at["total_w"] * 0.95
          and r["train"]["max_fall"] <= at["max_fall"] and r["train"]["worst_day"] >= at["worst_day"]]
    if not ok:
        return None
    nm, b = min(ok, key=lambda x: (x[1]["train"]["max_fall"], -x[1]["train"]["total_w"]))
    bu, bt = b["unseen"], b["train"]
    if bu["max_fall"] > au["max_fall"] or bu["worst_day"] < au["worst_day"] or bu["total"] < au["total"]:
        return None
    if nm == "side":
        rule = "מסלול לא נכנס למכשיר ולכיוון שמסלול אחר כבר מחזיק"
        change = {"track": "tal_all", "knob": "ONE_PER_SIDE", "from": False, "to": True}
        title = "שיטת הגאפ: בלי סיכון כפול על אותו רעיון"
    else:
        k = int(nm[3:])
        rule = f"לכל היותר {k} עסקאות פתוחות באותו כיוון בכל המדדים יחד (המדדים זזים יחד)"
        change = {"track": "tal_all", "knob": "MAX_DIR", "from": 0, "to": k}
        title = f"שיטת הגאפ: עד {k} עסקאות באותו כיוון בכל המדדים"
    what = (f"כל מסלולי שיטת הגאפ יחד, כמו שהחשבון לוקח אותם. 12.11: 9 מכירות של 4 מסלולים ב־US500 / US100 / JP225 נעצרו "
            f"תוך פחות משעתיים (-2.6%). הכלל: {rule}. נבחר על 2024-2025, מוצג על 2026. "
            f"2026: סה\"כ {au['total']:+.1f}R ← {bu['total']:+.1f}R, היום הגרוע {au['worst_day']:+.1f}R ← {bu['worst_day']:+.1f}R, "
            f"הירידה הגדולה {au['max_fall']:.1f}R ← {bu['max_fall']:.1f}R. 2024-2025: ירידה {at['max_fall']:.1f}R ← "
            f"{bt['max_fall']:.1f}R, סה\"כ משוקלל {at['total_w']:+.1f}R ← {bt['total_w']:+.1f}R.")
    return K.card(bot, "risk", title, what, None, None, tested=len(rules), risk_pct=risk_pct, detail=pf, change=change)


def scale_card(bot, pf, cur=1.0):
    """A card for risk_scale(): when the factor chosen on 2024-2025 is below 1 - every gap track's risk times it
    (change {"track": "tal_all", "knob": "RISK_SCALE"}). The bot as a whole first (principle 11): the profit in % falls
    with the factor, the risk of touching the limits falls more."""
    a = (pf or {}).get("account") or {}
    sc = a.get("scale") or {}
    if not sc or sc.get("chosen", 1.0) >= 1.0:
        return None
    k = sc["chosen"]
    now, new = a["now"], sc["rows"][str(k)]
    nt, nu, bt, bu = now["train"], now["unseen"], new["train"], new["unseen"]
    if not (nt and nu and bt and bu):
        return None

    def line(x):
        return (f"סה\"כ {x['total_pct']:+.1f}%, היום הגרוע {x['worst_day_pct']:+.1f}%, הירידה הגדולה {x['max_fall_pct']:.1f}%"
                + (f", נעצר ב־{x['stopped'][:10]}" if x.get("stopped") else "") + f", ימים שנעצרו במגבלה היומית {x['daily_stops']}")
    what = (f"כל מסלולי שיטת הגאפ יחד, בסיכון האמיתי של כל מסלול ועם המגבלות של הבוט (4.5% ביום, 9% בסך הכל). "
            f"היום: 2024-2025: {line(nt)}. 2026: {line(nu)}. "
            f"הכלל: כל הסיכונים × {k} - המקדם הגדול ביותר שבו 2024-2025 נשאר ב־80% מהמגבלות (יום ≥ -3.6%, ירידה מהשיא ≤ 7.2%), "
            f"נבחר רק על התקופה הזאת. עם המקדם: 2024-2025: {line(bt)}. 2026 (לא נראתה): {line(bu)}."
            + ("" if sc.get("fits") else " גם במקדם 0.2 2024-2025 לא נכנס במגבלות - צריך גם תיקון במסלולים עצמם."))
    to = round(float(cur) * k, 3)            # the risks given are already times the scale that runs (cur)
    return K.card(bot, "risk", f"שיטת הגאפ: סיכון כל המסלולים × {k}", what, None, None, tested=1,
                  risk_pct=round(min(a["risk_pct"].values()) * k, 3) if a.get("risk_pct") else None,
                  detail={"account": a}, change={"track": "tal_all", "knob": "RISK_SCALE", "from": cur, "to": to})


def hours_card(bot, pf, risk_pct):
    """A card for the hour filter of portfolio() (13.11.2026): the hours are chosen on 2024-2025 only; a card only when,
    without them, 2026's total does not fall and its deepest fall and worst day are not worse."""
    a, h = pf.get("now") or {}, pf.get("hours_off") or {}
    if not h.get("hours") or not a.get("unseen") or not h.get("unseen") or not h.get("train"):
        return None
    au, bu, at, bt = a["unseen"], h["unseen"], a["train"], h["train"]
    if bu["total"] < au["total"] or bu["max_fall"] > au["max_fall"] or bu["worst_day"] < au["worst_day"]:
        return None
    hs = ", ".join(f"{x:02d}:00" for x in h["hours"])
    what = (f"כל מסלולי שיטת הגאפ יחד. בשעות {hs} (UTC) כל המסלולים יחד הפסידו ב־2024-2025 (נבחר רק על התקופה הזאת). "
            f"הכלל: לא נכנסים בשעות האלה. 2026 (לא נראתה): סה\"כ {au['total']:+.1f}R ← {bu['total']:+.1f}R, "
            f"היום הגרוע {au['worst_day']:+.1f}R ← {bu['worst_day']:+.1f}R, הירידה הגדולה {au['max_fall']:.1f}R ← "
            f"{bu['max_fall']:.1f}R, עסקאות {au['n']} ← {bu['n']}. 2024-2025: ירידה {at['max_fall']:.1f}R ← {bt['max_fall']:.1f}R.")
    return K.card(bot, "risk", "שיטת הגאפ: בלי כניסות בשעות שהפסידו", what, None, None, tested=24, risk_pct=risk_pct,
                  detail={"hours_off": h, "now": a}, change={"track": "tal_all", "knob": "HOURS_OFF", "from": [], "to": h["hours"]})


# ---------------------------------------------------------------- live (12.11.2026): the same rules on the running bot
def live_trades(D, pair, cfg, gcache=None):
    """The method on the live 1-minute history in D (the last days): every gap of the high timeframe that is armed,
    with its state now - pending (waiting for the touch, within EXPIRY), open, closed, or cancelled (expired /
    no valid stop or target / below MIN_RR). Same rules as simulate(); one trade at a time per instrument.
    Returns dicts: id_key, dir, status, entry, stop, target, rr, setup_t, time, exit_time, exit, reason, r, r_now."""
    gcache = {} if gcache is None else gcache
    htf, ltf, _ = pair
    g = gaps(D, htf, cfg["GBODY"], cfg["MIN_GAP"], gcache)
    hi_sw, lo_sw = D.swings(ltf, cfg["SW"])
    ns, o1, h1, l1, c1 = D.ns, D.o1, D.h1, D.l1, D.c1
    n1 = len(ns)
    if not n1:
        return []
    exp_ns = int(cfg["EXPIRY"] * 3600 * 1e9)
    ts = lambda q: pd.Timestamp(int(ns[q]), tz="UTC").isoformat()   # noqa: E731
    prot = cfg.get("PROT", "none")
    calk = "cal60" if prot == "cal60" else "cal15" if prot in ("cal15", "cal15shock") else None
    shock = prot in ("shock", "cal15shock")
    trend = D.trend(htf, cfg.get("TREND", "none"), ltf)
    bbw, newsx = bbw_mask(D, cfg, htf, ltf), news_x(D, cfg)
    lad = ladder_of(cfg)
    out, cands = [], []
    # pass 1: every gap -> pending / cancelled, or a candidate fill (exactly simulate()'s checks, protection included)
    geo = gcache.get((htf, cfg["GBODY"], cfg["MIN_GAP"], "geo")) or []
    iso_ = lambda v: pd.Timestamp(int(v), tz="UTC").isoformat()   # noqa: E731
    for gi, (d, mid, near, far, made, armed) in enumerate(g):
        key = f"tal|{int(d)}|g{pd.Timestamp(made, tz='UTC').strftime('%Y%m%d%H%M')}"
        base = {"id_key": key, "dir": int(d), "setup_t": pd.Timestamp(armed, tz="UTC").isoformat(), "gap": [float(far), float(near)]}
        if gi < len(geo):
            ge = geo[gi]
            base["geo"] = {"htf": htf, "ltf": ltf, "c1": iso_(ge["c1"]), "c3": iso_(ge["c3"]),
                           "imp0": [iso_(ge["imp0"][0]), ge["imp0"][1]], "imp1": [iso_(ge["imp1"][0]), ge["imp1"][1]]}
        q0 = int(np.searchsorted(ns, armed))
        q1 = min(n1, int(np.searchsorted(ns, armed + exp_ns)))
        if q0 >= n1:
            continue
        qc = cancel_bar(D, cfg, q0)                                   # WKND: cancelled before a 2+ hour break (as simulate)
        if qc < q1:
            seg_c = (l1[q0:qc] <= ((near + lad[0][0] * (far - near)) if lad else mid)) if d > 0 else \
                (h1[q0:qc] >= ((near + lad[0][0] * (far - near)) if lad else mid))
            if not seg_c.any():
                out.append(dict(base, status="cancelled", entry=float(mid), stop=None, target=None, reason="break"))
                continue
            q1 = qc
        lv0 = (near + lad[0][0] * (far - near)) if lad else mid       # LADDER: the first level's touch (as simulate)
        seg = (l1[q0:q1] <= lv0) if d > 0 else (h1[q0:q1] >= lv0)
        if calk and calk in D.cal and D.cal_start[calk][q1 - 1] - (D.cal_start[calk][q0 - 1] if q0 else 0) > 0 and not seg.any():
            out.append(dict(base, status="cancelled", entry=float(mid), stop=None, target=None, reason="news"))
            continue
        if shock and not seg.any():
            cs = D.cs_dn if d > 0 else D.cs_up
            if cs[q1 - 1] - (cs[q0 - 1] if q0 else 0) > 0:
                out.append(dict(base, status="cancelled", entry=float(mid), stop=None, target=None, reason="shock"))
                continue
        if not seg.any():
            expired = armed + exp_ns <= ns[-1]
            out.append(dict(base, status="cancelled" if expired else "pending", entry=float(mid), stop=None, target=None,
                            reason="expired" if expired else None))
            continue
        f = q0 + int(np.argmax(seg))
        why = None
        if D.late[f]:
            why = "session"
        elif trend is not None and trend[f] != d:
            why = "trend"
        elif bbw is not None and not bbw[f]:
            why = "bbw"
        elif newsx is not None and newsx[f]:
            why = "news"
        elif calk and calk in D.cal and (D.cal[calk][f] or D.cal_start[calk][f] - (D.cal_start[calk][q0 - 1] if q0 else 0) > 0):
            why = "news"
        elif shock:
            cs = D.cs_dn if d > 0 else D.cs_up
            if cs[f] - (cs[q0 - 1] if q0 else 0) > 0 or D.recent_shock[f] or ((l1[f] < far) if d > 0 else (h1[f] > far)):
                why = "shock"
        if why:
            out.append(dict(base, status="cancelled", entry=float(mid), stop=None, target=None, reason=why))
            continue
        px = (min(mid, o1[f]) if d > 0 else max(mid, o1[f])) if not lad else mid
        sw_st, sw_tg = (lo_sw, hi_sw) if d > 0 else (hi_sw, lo_sw)
        s2 = _count_back(*sw_st, ns[f], px, d, -d)
        t2 = _count_back(*sw_tg, ns[f], px, d, d)
        if s2 is None or t2 is None:
            out.append(dict(base, status="cancelled", entry=float(px), stop=None, target=None, reason="no swings"))
            continue
        if "geo" in base:
            st_ns, w_ = TF[ltf] * 60_000_000_000, cfg["SW"]
            ts_s = _count_back_t(*sw_st, ns[f], px, d, -d, st_ns, w_)
            ts_t = _count_back_t(*sw_tg, ns[f], px, d, d, st_ns, w_)
            base = dict(base, geo=dict(base["geo"], sw_stop=[iso_(ts_s), float(s2)] if ts_s else None,
                                       sw_tgt=[iso_(ts_t), float(t2)] if ts_t else None))
        stop = min_stop(D, cfg, htf, ns[f], px, s2 - d * (D.buf + stop_pad(cfg, px)), d)
        risk = d * (px - stop)
        if risk <= 0 or d * (t2 - px) / risk < cfg["MIN_RR"]:
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="rr"))
            continue
        if not filt_ok(cfg, D.cost[f], risk, ns[f]):
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="filter"))
            continue
        if tp_way(cfg):
            t2 = px + tp_way(cfg) * (t2 - px)
        legs = ladder_legs(lad, d, near, far, stop, risk) if lad else None
        if lad and not legs:
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="rr"))
            continue
        base = dict(base, legs=[[float(p_), round(float(w_), 4)] for p_, w_ in legs]) if legs else base
        cands.append((f, d, px, stop, t2, risk, base))
    # pass 2: in the order of the fills, one trade at a time per instrument (as simulate())
    busy, dead, imp1 = -1, set(), cfg.get("IMP1") is True
    # the same order as simulate() (two gaps filled in the same minute: the same one is taken - backtest = live)
    for f, d, px, stop, t2, risk, base in sorted(cands, key=lambda x: x[:6]):
        if f < busy:
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="busy"))
            continue
        if imp1 and (base["setup_t"], d) in dead:      # IMP1: a gap of this impulse was already stopped
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="impulse"))
            continue
        if base.get("legs"):                             # LADDER: the same levels and fills as simulate()
            lt = ladder_trade(D, f, d, [tuple(x) for x in base["legs"]], stop, t2, px, cfg, n1)
            if lt is None:
                continue
            rec = dict(base, entry=float(lt["entry"]), stop=float(stop), target=float(t2), rr=round(d * (t2 - px) / risk, 2),
                       cost_r=0.0, time=ts(f), part=None, filled=lt["filled"])
            if lt["moved"]:
                rec["stop_now"] = float(lt["entry"])
            if lt["open"]:
                rec.update(status="open", r_now=round(lt["r_now"], 3))
                busy = n1
            else:
                q = f + lt["end"]
                rec.update(status="closed", exit=float(lt["entry"] if lt["why"] == "breakeven" else (stop if lt["why"] == "stop" else t2 if lt["why"] == "target" else o1[q])),
                           exit_time=ts(q), reason=lt["why"] if lt["why"] != "session" else "forced", r=round(lt["r"], 4))
                if lt["why"] == "stop":
                    dead.add((base["setup_t"], d))
                busy = q + 1
            out.append(rec)
            continue
        cost_r = float(D.cost[f]) / risk
        hs, ls = h1[f:], l1[f:]
        st = (ls <= stop) if d > 0 else (hs >= stop)
        tg = (hs >= t2) if d > 0 else (ls <= t2)
        tg[0] = False
        sh = shut_of(D, cfg, f, n1)
        sh[0] = False
        big = n1 - f + 1
        i_st = int(np.argmax(st)) if st.any() else big
        i_tg = int(np.argmax(tg)) if tg.any() else big
        i_sh = int(np.argmax(sh)) if sh.any() else big
        i = min(i_st, i_tg, i_sh)
        em, part_r, frac, moved = exit_mode(cfg), 0.0, 0.0, False
        if em is not None:                               # the same management as simulate()
            frac, way = em
            lvl = px + way * (t2 - px)
            lv = (hs >= lvl) if d > 0 else (ls <= lvl)
            lv[0] = False
            i_lv = int(np.argmax(lv)) if lv.any() else big
            if i_lv < i:
                part_r, moved = frac * way * d * (t2 - px) / risk, True
                be = (ls[i_lv + 1:] <= px) if d > 0 else (hs[i_lv + 1:] >= px)
                i_be = i_lv + 1 + int(np.argmax(be)) if be.any() else big
                if i_be <= i:
                    i, i_st = i_be, i_be
        rec = dict(base, entry=float(px), stop=float(stop), target=float(t2), rr=round(d * (t2 - px) / risk, 2),
                   cost_r=0.0, time=ts(f), part=round(frac, 2) if moved else None)
        if moved:
            rec["stop_now"] = float(px)                   # the rest's stop is at the entry
        if i >= big:
            rest_now = d * (c1[-1] - px) / risk
            rec.update(status="open", r_now=round((part_r + (1 - frac) * rest_now if moved else rest_now) - cost_r, 3))
            busy = n1
        else:
            q = f + i
            if i == i_st and moved:
                ex, why = ((min(px, o1[q]) if d > 0 else max(px, o1[q])) if i else px), "breakeven"
            elif i == i_st:
                ex, why = ((min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if i else stop), "stop"
            elif i == i_tg:
                ex, why = (max(t2, o1[q]) if d > 0 else min(t2, o1[q])), "target"
            else:
                ex, why = o1[q], "forced"
            rest = d * (ex - px) / risk
            if why == "stop":
                dead.add((base["setup_t"], d))
            rec.update(status="closed", exit=float(ex), exit_time=ts(q), reason=why,
                       r=round((part_r + (1 - frac) * rest if moved else rest) - cost_r, 4))
            busy = q + 1
        out.append(rec)
    return out
