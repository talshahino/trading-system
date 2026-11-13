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
         "TREND": ["day", "up20"]}
NAMES = {"GBODY": "עוצמת הדחף (גוף הנר האמצעי מול הממוצע)", "MIN_GAP": "גודל גאפ מינימלי (מול הטווח הממוצע)",
         "SW": "סווינג בטיים פריים הנמוך (נרות מכל צד)", "MIN_RR": "יחס מינימלי יעד/סטופ", "EXPIRY": "תוקף הגאפ (שעות)",
         "PROT": "הגנה מהודעות וזעזועים", "EXIT": "ניהול העסקה: יציאה חלקית / סטופ לנקודת הכניסה",
         "IMP1": "אחרי סטופ - בלי עוד כניסות מאותו דחף",
         "TREND": "רק בכיוון התמונה הגדולה (פתיחת היום / טיים פריים מעל)"}
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

    def trend(self, htf, mode):
        """TREND: per 1-minute bar, +1 / -1 / 0 = the side the bigger picture allows (known before the bar opens), or
        None when off. day: the previous minute's close against the UTC day's first open; up20: the last CLOSED candle
        of the timeframe above HTF against its 20-candle EMA."""
        if not mode or mode == "none":
            return None
        key = ("trend", htf, mode)
        if key not in self.sw:
            prev = np.r_[self.c1[0], self.c1[:-1]]
            if mode == "day":
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
SESS_NY = {"rth": [(570, 960)], "nyam": [(570, 690)], "xopen": [(0, 570), (600, 1440)]}


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


def tp_way(cfg):
    """EXIT "tp30" (user, 13.11.2026: "take 100% of the profit at 30% of the way to the target"): the target moves to
    30% of the way; None for every other EXIT."""
    m = cfg.get("EXIT", "none")
    return int(m[2:]) / 100 if m.startswith("tp") else None


def simulate(D, pair, cfg, gcache):
    htf, ltf, _ = pair
    g = gaps(D, htf, cfg["GBODY"], cfg["MIN_GAP"], gcache)
    hi_sw, lo_sw = D.swings(ltf, cfg["SW"])
    ns, o1, h1, l1, c1 = D.ns, D.o1, D.h1, D.l1, D.c1
    n1 = len(ns)
    exp_ns = int(cfg["EXPIRY"] * 3600 * 1e9)
    prot = cfg["PROT"]
    calk = "cal60" if prot == "cal60" else "cal15" if prot in ("cal15", "cal15shock") else None
    shock = prot in ("shock", "cal15shock")
    trend = D.trend(htf, cfg.get("TREND", "none"))
    cands = []
    for d, mid, near, far, made, armed in g:
        q0 = int(np.searchsorted(ns, armed))
        q1 = min(n1, int(np.searchsorted(ns, armed + exp_ns)))      # from the impulse's end (the order's placing)
        if q0 >= q1:
            continue
        seg = (l1[q0:q1] <= mid) if d > 0 else (h1[q0:q1] >= mid)
        if not seg.any():
            continue
        f = q0 + int(np.argmax(seg))
        if D.late[f]:
            continue                                     # the world's no-entry time (session)
        if trend is not None and trend[f] != d:
            continue                                     # TREND: against the bigger picture
        if calk:
            if D.cal[calk][f] or D.cal_start[calk][f] - (D.cal_start[calk][q0 - 1] if q0 else 0) > 0:
                continue                                 # a release window: no entry, the waiting order was cancelled
        if shock:
            cs = D.cs_dn if d > 0 else D.cs_up
            if cs[f] - (cs[q0 - 1] if q0 else 0) > 0:
                continue                                 # a shock against the gap while it waited: cancelled
            if D.recent_shock[f] or ((l1[f] < far) if d > 0 else (h1[f] > far)):
                continue                                 # not a calm correction
        px = min(mid, o1[f]) if d > 0 else max(mid, o1[f])
        sw_st, sw_tg = (lo_sw, hi_sw) if d > 0 else (hi_sw, lo_sw)
        s2 = _count_back(*sw_st, ns[f], px, d, -d)
        t2 = _count_back(*sw_tg, ns[f], px, d, d)
        if s2 is None or t2 is None:
            continue
        stop = s2 - d * (D.buf + stop_pad(cfg, px))
        risk = d * (px - stop)
        if risk <= 0 or d * (t2 - px) / risk < cfg["MIN_RR"]:
            continue
        if not filt_ok(cfg, D.cost[f], risk, ns[f]):
            continue
        if tp_way(cfg):
            t2 = px + tp_way(cfg) * (t2 - px)
        cands.append((f, d, px, stop, t2, risk, armed))
    cands.sort()
    rows, busy, dead, imp1 = [], -1, set(), cfg.get("IMP1") is True
    for f, d, px, stop, tgt, risk, armed in cands:
        if f < busy or (imp1 and (armed, d) in dead):
            continue
        for span in (4 * 1440, 40 * 1440):              # most trades end within a day; the rest within weeks
            e = min(n1, f + span)
            hs, ls = h1[f:e], l1[f:e]
            st = (ls <= stop) if d > 0 else (hs >= stop)
            tg = (hs >= tgt) if d > 0 else (ls <= tgt)
            tg[0] = False
            sh = D.shut[f:e].copy()
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
        rows.append({"time": pd.Timestamp(ns[f], tz="UTC").isoformat(), "dir": d, "r": round(r_all - cost_r, 4),
                     "cost_r": round(cost_r, 4), "rr": round(d * (tgt - px) / risk, 2), "why": why, "mins": i})
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
    for k in ("EXIT", "IMP1", "TREND"):
        if c.get(k) == "none":
            c.pop(k)                                    # "none" is the base: the same configuration, the same name
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
    d = {"day": DAY_LIM, "all": ALL_LIM, "trail": False, "use": 1.0, "dayf": None, "day_hard": False}
    d.update(lim or {})
    return d


def account(part, risk_pct, lim=None):
    """The account the way the virtual bot keeps it (live/watch_own.py), in % of the starting balance (no compounding -
    careful): each trade risks its track's % (risk_pct = {track: %}); a result counts when the trade closes ("out").
    After the day's closed loss reaches DAY_LIM no new trade opens that day; and (13.11.2026 evening, the live rule of
    bot 5) a new trade risks at most the room left - the day's closed loss plus every open trade's full risk stays
    within DAY_LIM, and the fall from the start plus the open risk within ALL_LIM; under a quarter of its risk left it is
    not taken. Once the fall from the start reaches ALL_LIM the account stops. Returns total, worst day, deepest fall
    from the peak, the fall from the start, days the daily limit stopped trading, trades cut / skipped by the room rule,
    the largest open risk at once (and the days it passed DAY_LIM before this rule), and when the account stopped."""
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
    eq, peak, fall, low, dead = 0.0, 0.0, 0.0, 0.0, None
    day_real, stopped_days, cut, skipped = {}, set(), 0, 0
    opened = []                       # (out, risk %, pnl %) of the trades taken and not booked yet
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

    for r_, w_, t_, o_ in zip(rr, rk, times, outs):
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
        if room < 0.25 * w_:
            skipped += 1
            continue
        use = min(w_, room)
        cut += use < w_
        opened.append((o_, use, r_ * use))
    book(pd.Timestamp.max.tz_localize("UTC"))
    worst = min([0.0] + list(day_real.values()))
    return {"total_pct": round(float(eq), 2), "worst_day_pct": round(float(worst), 2), "max_fall_pct": round(float(fall), 2),
            "low_pct": round(float(low), 2), "daily_stops": len(stopped_days), "room_cut": int(cut), "room_skipped": int(skipped),
            "open_peak_pct": round(float(raw_peak), 2), "open_days_over_day_lim": len(raw_days), "stopped": dead}


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
    trend = D.trend(htf, cfg.get("TREND", "none"))
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
        seg = (l1[q0:q1] <= mid) if d > 0 else (h1[q0:q1] >= mid)
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
        elif calk and calk in D.cal and (D.cal[calk][f] or D.cal_start[calk][f] - (D.cal_start[calk][q0 - 1] if q0 else 0) > 0):
            why = "news"
        elif shock:
            cs = D.cs_dn if d > 0 else D.cs_up
            if cs[f] - (cs[q0 - 1] if q0 else 0) > 0 or D.recent_shock[f] or ((l1[f] < far) if d > 0 else (h1[f] > far)):
                why = "shock"
        if why:
            out.append(dict(base, status="cancelled", entry=float(mid), stop=None, target=None, reason=why))
            continue
        px = min(mid, o1[f]) if d > 0 else max(mid, o1[f])
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
        stop = s2 - d * (D.buf + stop_pad(cfg, px))
        risk = d * (px - stop)
        if risk <= 0 or d * (t2 - px) / risk < cfg["MIN_RR"]:
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="rr"))
            continue
        if not filt_ok(cfg, D.cost[f], risk, ns[f]):
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="filter"))
            continue
        if tp_way(cfg):
            t2 = px + tp_way(cfg) * (t2 - px)
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
        cost_r = float(D.cost[f]) / risk
        hs, ls = h1[f:], l1[f:]
        st = (ls <= stop) if d > 0 else (hs >= stop)
        tg = (hs >= t2) if d > 0 else (ls <= t2)
        tg[0] = False
        sh = D.shut[f:].copy()
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
