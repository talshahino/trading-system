"""Tal's gap method (user, 11.11.2026; data/agents/strategies/tal_gap.md) - the backtest, for bots 5 and 6 (not bot 7).

Exact rules (the smart agent's reading of the file, 12.11.2026):
  timeframes  a pair H/L from the ladder 1m, 5m, 15m, 1h, 4h, 1d, one / two / three steps apart (PAIRS, 12 pairs).
              H finds the setup, L places the stop and the target; fills and exits always walk the 1-minute bars.
  impulse     on H: from the last swing low (2 candles each side) before the gap to the first swing high after it
              (mirror for down). Its size >= IMP x ATR14(H). Up = longs only, down = shorts only.
  gap         candles k-1, k, k+1 of the impulse: long when low(k+1) > high(k-1) (wicks included), candle k bullish;
              size >= MIN_GAP x ATR14(H). It must stay open until the impulse ends: no H candle from k+2 to the
              impulse's top goes back to the gap's middle.
  armed       when the impulse's end is known (the swing high confirmed, 2 H candles later). If the price already
              came back to the middle during those 2 candles the touch was before anyone could know: skipped.
  entry       a limit order at the gap's middle; the first 1-minute touch fills it (at the open when the bar opens
              beyond it). Order valid EXP x DEF_EXP[H] hours from the gap's making; cancelled when an H candle
              closes beyond the whole box (the gap's far edge).
  stop        counting back in time on L the swing lows below the entry (SW candles each side, known before the
              fill), each one at least MERGE x ATR14(L) below the one counted before it (small close swings = one);
              the stop is BUF_PIPS pips (index point = 1) beyond the 2nd. Mirror for shorts.
  target      the same on the other side: the 2nd swing high above the entry, counted back in time.
  min R:R     the order is dropped when target / stop distance < MIN_RR (0 = Tal's rule as drawn).
  management  none: stop or target. The bot's market rules stay: out before a market break > 2 hours.
Protection (the file's mechanism, PROT):
  NEWS = X    no fill from X minutes before to X after a high-impact release of the instrument's currencies (the
              pending order is paused, not lost); and bot 5's own rules (FTMO restricted events, out 1 minute
              before a high-impact release). X = 0 -> no news rule at all.
  SHOCK       a 5-minute candle with a range >= 3 x the average of the 100 before it, against the gap's direction,
              cancels the gap and every older gap of that direction (a new impulse makes new ones); a 5-minute
              candle that crosses the whole box cancels that gap; and a fill only in a calm pullback (no 1-minute
              range >= 3 x average in the 10 minutes before the touch).
Costs: FTMO's real spread (Dukascopy ask-bid at the fill minute) + commission + swap per night held.
Every trade carries the features the file's open questions need (impulse size / candles, gap size, R:R, hour,
minutes to the nearest release), so the answers come from the trades of the period to 2025-12-31 and are shown
on 2026.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents import ftmo_costs as FC
from agents import ftmo_universe as U
from agents import news_calendar as NC
from agents import rules_agent as R

V = 1                                   # raise when the rules change: every result is re-run
LADDER = [1, 5, 15, 60, 240, 1440]
TF_NAME = {1: "1Min", 5: "5Min", 15: "15Min", 60: "1Hour", 240: "4Hour", 1440: "1Day"}
TF_HE = {1: "דקה", 5: "5 דק'", 15: "15 דק'", 60: "שעה", 240: "4 שעות", 1440: "יום"}
DEF_EXP = {5: 8, 15: 24, 60: 48, 240: 120, 1440: 480}      # hours a gap waits (x EXP)
BASE = {"V": V, "IMP": 1.0, "MIN_GAP": 0.0, "SW": 2, "MERGE": 0.3, "MIN_RR": 0.0, "EXP": 1.0, "NEWS": 0, "SHOCK": False,
        "BUF_PIPS": 2}
PROT = {"NEWS": 30, "SHOCK": True}
HW = 2                                   # H swings: 2 candles each side
SHOCK_X, CALM_X = 3.0, 3.0


def pairs():
    """[(H, L, steps)] - the file's 12 pairs, two steps first (Tal's own), then one, then three."""
    out = []
    for st in (2, 1, 3):
        for i, h in enumerate(LADDER):
            if i - st >= 0:
                out.append((h, LADDER[i - st], st))
    return out


def pair_name(h, l_):
    return f"{TF_NAME[h]}/{TF_NAME[l_]}"


def pip(s):
    from agents import own_core as CORE
    return CORE.pip_of(s) or (0.1 if s.startswith("XA") else 1.0)


def tf_bars(df, m):
    if m == 1:
        return df
    kw = {"offset": "22h"} if m == 1440 else {}
    rule = "24h" if m == 1440 else f"{m}min"
    b = df.resample(rule, label="left", closed="left", **kw).agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    return b


def _swing_idx(arr, w, high):
    s = pd.Series(arr)
    ext = (s.rolling(2 * w + 1, center=True).max() if high else s.rolling(2 * w + 1, center=True).min()).to_numpy()
    return np.flatnonzero(arr == ext)


class Data:
    """One instrument's minutes and everything derived from them, built once and shared by every config."""

    def __init__(self, s, df, cost, brk, nm, noent_ftmo, events):
        self.s, self.df, self.cost = s, df, cost
        self.t1 = df.index
        self.o1, self.h1, self.l1, self.c1 = (df[x].to_numpy(float) for x in "ohlc")
        self.n1 = len(df)
        self.brk = np.asarray(brk, bool)
        self.nm = np.asarray(nm, bool)
        self.noent_ftmo = np.asarray(noent_ftmo, bool)
        self.events = events
        self._tf, self._sw, self._news, self._shock = {}, {}, {}, None
        cur = NC.CURRENCIES.get(s) or []
        ev = events[events["currency"].isin(cur)]["time"] if events is not None and len(events) else pd.Series([], dtype="datetime64[ns, UTC]")
        self.ev = np.sort(pd.DatetimeIndex(pd.to_datetime(ev, utc=True)).tz_localize(None).to_numpy().astype("datetime64[ns]"))
        self.t64 = self.t1.tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[ns]")
        r1 = self.h1 - self.l1
        self.avg1 = pd.Series(r1).rolling(100, min_periods=30).mean().shift(1).to_numpy()
        self.maxprev10 = pd.Series(r1).rolling(10, min_periods=1).max().shift(1).to_numpy()
        self.spec = U.spec(s)

    def tf(self, m):
        if m not in self._tf:
            b = tf_bars(self.df, m)
            o, h, l_, c = (b[x].to_numpy(float) for x in "ohlc")
            end = b.index + pd.Timedelta(minutes=m)
            if m == 1440:
                end = b.index + pd.Timedelta(days=1)
            q_end = self.t1.searchsorted(end)                   # first minute at / after the candle's end
            self._tf[m] = {"t": b.index, "o": o, "h": h, "l": l_, "c": c, "atr": R.atr(h, l_, c, 14), "end": end, "q_end": q_end,
                           "shi": _swing_idx(h, HW, True), "slo": _swing_idx(l_, HW, False)}
        return self._tf[m]

    def sw(self, m, w):
        if (m, w) not in self._sw:
            b = self.tf(m)
            self._sw[(m, w)] = (_swing_idx(b["h"], w, True), _swing_idx(b["l"], w, False))
        return self._sw[(m, w)]

    def news(self, x):
        """minutes within +-x of a release of the instrument's currencies"""
        if x not in self._news:
            if x <= 0 or len(self.ev) == 0:
                self._news[x] = np.zeros(self.n1, bool)
            else:
                t = self.t64
                i = np.searchsorted(self.ev, t)
                w = np.timedelta64(int(x), "m")
                big = np.timedelta64(10 ** 6, "m")
                after = np.where(i < len(self.ev), self.ev[np.minimum(i, len(self.ev) - 1)] - t, big)
                before = np.where(i > 0, t - self.ev[np.maximum(i - 1, 0)], big)
                self._news[x] = (after <= w) | (before <= w)
        return self._news[x]

    def near_event_min(self, q):
        if len(self.ev) == 0:
            return None
        t = self.t64[q]
        i = np.searchsorted(self.ev, t)
        d = [abs((self.ev[j] - t) / np.timedelta64(1, "m")) for j in (i - 1, i) if 0 <= j < len(self.ev)]
        return round(float(min(d)), 1) if d else None

    def shock(self):
        """5-minute shock candles: (1m index at the candle's end for bearish shocks, for bullish ones), and the
        5-minute arrays for the box-crossing test"""
        if self._shock is None:
            b = self.tf(5)
            rng = b["h"] - b["l"]
            avg = pd.Series(rng).rolling(100, min_periods=30).mean().shift(1).to_numpy()
            big = np.isfinite(avg) & (rng >= SHOCK_X * avg)
            bear = big & (b["c"] < b["o"])
            bull = big & (b["c"] > b["o"])
            self._shock = {"bear": b["q_end"][bear], "bull": b["q_end"][bull], "avg": avg}
        return self._shock


def masks(D, cfg):
    """(no-fill mask, forced-exit mask) for a config"""
    if cfg["NEWS"] > 0:
        blk2 = D.news(2)
        stop_at = D.brk | D.nm | np.r_[blk2[1:], False]
        nofill = D.noent_ftmo | blk2 | D.news(cfg["NEWS"])
    else:
        stop_at = D.brk.copy()
        nofill = np.zeros(D.n1, bool)
    nofill = nofill | D.brk
    if cfg["SHOCK"]:
        calm = ~(np.isfinite(D.avg1) & (D.maxprev10 >= CALM_X * D.avg1))
        nofill = nofill | ~calm
    return nofill, stop_at


def _exit(D, fill, d, stop, target, stop_at, chunk=20000):
    l1, h1, o1, c1 = D.l1, D.h1, D.o1, D.c1
    if (d > 0 and l1[fill] <= stop) or (d < 0 and h1[fill] >= stop):
        return fill, stop, "stop"
    a = fill + 1
    while a < D.n1:
        e = min(D.n1, a + chunk)
        hs = (l1[a:e] <= stop) if d > 0 else (h1[a:e] >= stop)
        ht = (h1[a:e] >= target) if d > 0 else (l1[a:e] <= target)
        any_ = hs | ht | stop_at[a:e]
        if any_.any():
            i = int(np.argmax(any_))
            q = a + i
            if hs[i]:
                return q, (min(stop, o1[q]) if d > 0 else max(stop, o1[q])), "stop"
            if ht[i]:
                return q, (max(target, o1[q]) if d > 0 else min(target, o1[q])), "target"
            return q, c1[q], "forced"
        a = e
    return None, None, None


def _swap_px(D, d, px, t_in, t_out):
    sp = D.spec
    nights = float(FC._nights(pd.DatetimeIndex([t_in]), pd.DatetimeIndex([t_out]), False)[0])
    if nights <= 0:
        return 0.0
    sw = float(sp["swapLong"] if d > 0 else sp["swapShort"])
    if sp["swapType"] == "percentage":
        return px * sw / 100.0 / 360.0 * nights
    return sw * 10.0 ** (-int(sp["digits"])) * nights / FC.PRICE_SCALE.get(D.s, 1.0)


def run(D, H, L, cfg, diag=None):
    """All trades of one instrument for one pair and config (one position at a time). Returns a DataFrame."""
    Hb = D.tf(H)
    o, h, l_, c, atr, qend = Hb["o"], Hb["h"], Hb["l"], Hb["c"], Hb["atr"], Hb["q_end"]
    tH = Hb["t"]
    n = len(c)
    if n < 40:
        return pd.DataFrame()
    shi, slo = Hb["shi"], Hb["slo"]
    Lb = D.tf(L)
    lh, ll, latr, lend = Lb["h"], Lb["l"], Lb["atr"], Lb["q_end"]
    lshi, lslo = D.sw(L, int(cfg["SW"]))
    nofill, stop_at = masks(D, cfg)
    sh = D.shock() if cfg["SHOCK"] else None
    buf = cfg["BUF_PIPS"] * pip(D.s)
    exp_td = pd.Timedelta(hours=DEF_EXP[H] * cfg["EXP"])
    t1 = D.t1
    cand = []

    def dg(why):
        if diag is not None:
            diag[why] = diag.get(why, 0) + 1

    for k in range(15, n - HW - 2):
        for d in (1, -1):
            if d > 0:
                if not (l_[k + 1] > h[k - 1] and c[k] > o[k]):
                    continue
                bot, top = h[k - 1], l_[k + 1]
            else:
                if not (h[k + 1] < l_[k - 1] and c[k] < o[k]):
                    continue
                bot, top = h[k + 1], l_[k - 1]
            if not np.isfinite(atr[k]) or atr[k] <= 0:
                continue
            gap = top - bot
            mid = (top + bot) / 2
            if gap < cfg["MIN_GAP"] * atr[k]:
                dg("gap_small"); continue
            ends = shi if d > 0 else slo
            jj = np.searchsorted(ends, k + 1)
            if jj >= len(ends):
                continue
            j = int(ends[jj])
            arm = j + HW
            if arm >= n - 1:
                continue
            # the gap stays open (the middle untouched) until the impulse ends
            if j >= k + 2:
                if (d > 0 and l_[k + 2:j + 1].min() <= mid) or (d < 0 and h[k + 2:j + 1].max() >= mid):
                    dg("closed_in_impulse"); continue
            starts = slo if d > 0 else shi
            ii = np.searchsorted(starts, k - 1, side="right") - 1
            i0 = int(starts[ii]) if ii >= 0 and k - 1 - starts[ii] <= 100 else max(0, k - 5)
            i0 = min(i0, k - 1)
            size = (h[j] - l_[i0:k + 1].min()) if d > 0 else (h[i0:k + 1].max() - l_[j])
            if size < cfg["IMP"] * atr[k]:
                dg("impulse_small"); continue
            if (d > 0 and l_[j + 1:arm + 1].min() <= mid) or (d < 0 and h[j + 1:arm + 1].max() >= mid):
                dg("touched_before_known"); continue
            q0 = int(qend[arm])
            q1 = min(D.n1, int(t1.searchsorted(tH[k] + exp_td)))
            # cancelled: an H close beyond the whole box
            far = bot if d > 0 else top
            kx = min(n, int(tH.searchsorted(tH[k] + exp_td)) + 1)
            cc = c[arm + 1:kx]
            br = (cc < far) if d > 0 else (cc > far)
            if br.any():
                q1 = min(q1, int(qend[arm + 1 + int(np.argmax(br))]))
            if sh is not None:
                arr = sh["bear"] if d > 0 else sh["bull"]
                p_ = np.searchsorted(arr, q0)
                if p_ < len(arr):
                    q1 = min(q1, int(arr[p_]))
                # a 5-minute candle across the whole box
                b5 = D.tf(5)
                a5, e5 = np.searchsorted(b5["q_end"], q0), np.searchsorted(b5["q_end"], q1)
                if e5 > a5:
                    cross = (b5["h"][a5:e5] >= top) & (b5["l"][a5:e5] <= bot)
                    if cross.any():
                        q1 = min(q1, int(b5["q_end"][a5 + int(np.argmax(cross))]))
            if q0 >= q1:
                dg("expired"); continue
            touch = (D.l1[q0:q1] <= mid) if d > 0 else (D.h1[q0:q1] >= mid)
            ok = touch & ~nofill[q0:q1]
            if not ok.any():
                dg("blocked" if touch.any() else "no_touch"); continue
            fill = q0 + int(np.argmax(ok))
            px = min(mid, D.o1[fill]) if d > 0 else max(mid, D.o1[fill])
            # stop and target from the L swings known before the fill
            nL = int(np.searchsorted(lend, fill, side="right"))        # L candles ended by the fill's minute
            hiL = nL - 1 - int(cfg["SW"])
            if hiL < 5:
                continue
            a_ = latr[nL - 1] if np.isfinite(latr[nL - 1]) else 0.0
            mg = cfg["MERGE"] * a_

            # back in time as far as the impulse's start (and 200 L candles before it), at least 400 L candles
            lo_ = min(hiL - 400, int(Lb["t"].searchsorted(tH[i0])) - 200)

            def nth(idx, vals, beyond):
                ix = idx[np.searchsorted(idx, lo_):np.searchsorted(idx, hiL, side="right")][::-1]
                last, cnt = px, 0
                for i in ix:
                    v = vals[i]
                    if beyond * (v - last) > (mg if cnt else 0):
                        cnt += 1; last = v
                        if cnt == 2:
                            return v
                return None
            s2 = nth(lslo, ll, -1) if d > 0 else nth(lshi, lh, 1)
            t2 = nth(lshi, lh, 1) if d > 0 else nth(lslo, ll, -1)
            if s2 is None or t2 is None:
                dg("no_swings"); continue
            stop = s2 - d * buf
            target = t2
            risk = d * (px - stop)
            rew = d * (target - px)
            if risk <= 0 or rew <= 0 or rew < cfg["MIN_RR"] * risk:
                dg("rr"); continue
            cand.append({"fill": fill, "d": d, "px": px, "stop": stop, "target": target, "risk": risk, "rr": rew / risk,
                         "imp_atr": size / atr[k], "imp_n": int(j - i0), "gap_atr": gap / atr[k], "q0": q0})
    cand.sort(key=lambda x: x["fill"])
    out, last = [], -1
    for x in cand:
        if x["fill"] <= last:
            dg("busy"); continue
        q, ex, why = _exit(D, x["fill"], x["d"], x["stop"], x["target"], stop_at)
        if q is None:
            continue                                     # still open at the data's end
        last = q
        d, px, risk = x["d"], x["px"], x["risk"]
        swp = _swap_px(D, d, px, t1[x["fill"]], t1[q])
        cost_r = D.cost[x["fill"]] / risk
        r = d * (ex - px) / risk - cost_r + swp / risk
        out.append({"time": t1[x["fill"]], "exit_time": t1[q], "dir": d, "r": round(float(r), 4), "cost_r": round(float(cost_r), 4),
                    "swap_r": round(float(swp / risk), 4), "rr": round(float(x["rr"]), 2), "reason": why,
                    "imp_atr": round(float(x["imp_atr"]), 2), "imp_n": x["imp_n"], "gap_atr": round(float(x["gap_atr"]), 3),
                    "wait_min": int(x["fill"] - x["q0"]), "hold_min": int(q - x["fill"]),
                    "news_min": D.near_event_min(x["fill"])})
        dg("traded")
    return pd.DataFrame(out)


# ---------------------------------------------------------------- the experiments (the bot agents run them)
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
START = "2024-01-01"
RERUN_DAYS = 30
RESULTS = Path("data/agents/strategies/tal_gap_results.json")
# (name, change from BASE, tier): 0 = Tal's rules without / with the protection, 1 = the protection's parts,
# 2 = the open questions (impulse, gap, swing, R:R, expiry)
CONFIGS = [("base", {}, 0), ("prot", dict(PROT), 0),
           ("news2", {"NEWS": 2}, 1), ("news30", {"NEWS": 30}, 1), ("shock", {"SHOCK": True}, 1),
           ("news15s", {"NEWS": 15, "SHOCK": True}, 1), ("news60s", {"NEWS": 60, "SHOCK": True}, 1),
           ("imp2", {"IMP": 2.0}, 2), ("imp3", {"IMP": 3.0}, 2), ("gap25", {"MIN_GAP": 0.25}, 2), ("gap50", {"MIN_GAP": 0.5}, 2),
           ("rr1", {"MIN_RR": 1.0}, 2), ("rr15", {"MIN_RR": 1.5}, 2), ("sw1", {"SW": 1}, 2), ("sw3", {"SW": 3}, 2),
           ("sw5", {"SW": 5}, 2), ("merge0", {"MERGE": 0.0}, 2), ("merge1", {"MERGE": 1.0}, 2),
           ("exp033", {"EXP": 0.33}, 2), ("exp2", {"EXP": 2.0}, 2)]
CFG_HE = {"base": "החוקים כפי שהם, בלי הגנה", "prot": "עם ההגנה (פרסומים ±30 דק' + זעזועים)", "news2": "הגנת פרסומים של בוט 5 היום (±2 דק')",
          "news30": "פרסומים ±30 דק' בלי זעזועים", "shock": "רק הגנת זעזועים", "news15s": "פרסומים ±15 דק' + זעזועים",
          "news60s": "פרסומים ±60 דק' + זעזועים", "imp2": "דחף לפחות 2 ATR", "imp3": "דחף לפחות 3 ATR", "gap25": "גאפ לפחות 0.25 ATR",
          "gap50": "גאפ לפחות 0.5 ATR", "rr1": "יחס יעד/סטופ לפחות 1", "rr15": "יחס יעד/סטופ לפחות 1.5", "sw1": "סווינג: נר אחד מכל צד",
          "sw3": "סווינג: 3 נרות מכל צד", "sw5": "סווינג: 5 נרות מכל צד", "merge0": "בלי איחוד סווינגים צמודים",
          "merge1": "סווינגים במרחק לפחות 1 ATR", "exp033": "גאפ בתוקף שליש מהזמן", "exp2": "גאפ בתוקף כפול זמן"}


def cfg_of(name):
    ch = next(c for n_, c, _ in CONFIGS if n_ == name)
    return dict(BASE, **ch)


def vid(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def prep_data(s, end):
    """bot 5's data: FTMO's real cost per minute, market breaks, FTMO restricted events, the news calendar"""
    import own15 as O15
    df = O15.P.bars(s, "1Min", max(START, O15.RP.MINUTE_START), end)
    sp = O15.P.spread_1m(s, START, end)
    if df is None or df.empty or sp is None or len(sp) == 0:
        return None
    if "v" not in df:
        df["v"] = 1.0
    c = df["c"].to_numpy(float)
    cost = np.abs(sp.sort_index().reindex(df.index, method="ffill").to_numpy(float)) + O15.comm_px(s, c)
    cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
    brk = O15.RR.break_mask(df, "1Min")
    sm = O15.RR.session_mask(df, s, "1Min")
    if sm is not None:
        brk = brk | np.asarray(sm, bool)
    ev, _ = NC.load()
    nm, noent = O15.RR.news_masks(df, s, "1Min")
    return Data(s, df, cost, brk, nm, noent, ev)


def _dir(bot):
    return Path("reports/agents") / bot / "tal"


def load_index(bot):
    d = _dir(bot) / "idx"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in d.glob("*.json"):
        v = json.loads(f.read_text())
        out[v["key"]] = v
    return out


def experiments(bot, symbols, budget_s, prep=prep_data):
    """Runs what is missing (or older than RERUN_DAYS), most important first, until the budget ends.
    One small index file per run, so parallel runs (one per instrument) never clash."""
    out = _dir(bot)
    out.mkdir(parents=True, exist_ok=True)
    idx = load_index(bot)
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    todo = []
    for po, (H, L, st) in enumerate(pairs()):
        for name, _, tier in CONFIGS:
            cfg = cfg_of(name)
            for s in symbols:
                key = f"{H}_{L}|{vid(cfg)}|{s}"
                age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(idx[key]["at"])).days if key in idx else 999
                if age >= RERUN_DAYS:
                    todo.append((tier, s, po, H, L, name, cfg, key))
    todo.sort(key=lambda x: (x[0], x[1], x[2]))
    t0, data, done = time.time(), {}, 0
    for tier, s, po, H, L, name, cfg, key in todo:
        if time.time() - t0 > budget_s:
            break
        if s not in data:
            data.clear()
            try:
                data[s] = prep(s, end)
            except Exception as e:  # noqa: BLE001
                print(f"tal {s}: no data ({e})")
                data[s] = None
        if data[s] is None:
            continue
        ts = time.time()
        diag = {}
        try:
            t = run(data[s], H, L, cfg, diag)
        except Exception as e:  # noqa: BLE001
            print(f"tal {pair_name(H, L)} {name} {s}: failed ({e})")
            continue
        f = out / f"{H}_{L}_{vid(cfg)}_{s}.csv.gz"
        t.to_csv(f, index=False)
        rec = {"key": key, "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "H": H, "L": L, "name": name,
               "cfg": cfg, "sym": s, "file": f.name, "n": len(t), "secs": round(time.time() - ts, 1), "diag": diag}
        idx[key] = rec
        (out / "idx" / f.name.replace(".csv.gz", ".json")).write_text(json.dumps(rec, default=str))
        done += 1
        print(f"  tal {pair_name(H, L)} {name} {s}: {len(t)} trades ({time.time() - ts:.0f}s)")
    print(f"tal gap: {done} runs, {len(todo) - done} left for the next runs")
    return {"done": done, "left": len(todo) - done}


def trades_of(bot, idx, H, L, cfg, symbols):
    rs = [v for v in idx.values() if v["H"] == H and v["L"] == L and v["cfg"] == cfg]
    if {v["sym"] for v in rs} != set(symbols):
        return None
    fs = []
    for v in rs:
        f = _dir(bot) / v["file"]
        if f.exists() and v["n"] > 0:
            x = pd.read_csv(f)
            x["sym"] = v["sym"]
            fs.append(x)
    if not fs:
        return pd.DataFrame(columns=["time", "r"])
    t = pd.concat(fs, ignore_index=True)
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    return t.sort_values("time").reset_index(drop=True)


def stats(t):
    from agents import agentkit as K
    if t is None or t.empty:
        z = K.wstats([])
        return {"train": z, "unseen": z, "whole": z, "halves": [None, None], "cost_r": None, "per_day": 0.0}
    tr, un = t[t["time"] < SPLIT], t[t["time"] >= SPLIT]
    days = max(1.0, (t["time"].max() - t["time"].min()).days * 5 / 7)
    return {"train": K.wstats(tr["r"], K.weights(tr["time"])), "unseen": K.wstats(un["r"], K.weights(un["time"])),
            "whole": K.wstats(t["r"], K.weights(t["time"])), "halves": K.halves(t["time"], t["r"]),
            "cost_r": round(float(t["cost_r"].mean() + (-t["swap_r"].mean() if "swap_r" in t else 0)), 3),
            "per_day": round(len(t) / days, 2),
            "unseen_by_sym": {s: round(float(g["r"].mean()), 3) for s, g in un.groupby("sym")} if "sym" in un else {}}


def _thr(t, col, vals, ge=True):
    """the threshold chosen on the train period (time-weighted mean, >= 30 effective trades), shown on 2026"""
    from agents import agentkit as K
    tr, un = t[t["time"] < SPLIT], t[t["time"] >= SPLIT]
    rows = []
    for v in vals:
        a = tr[tr[col] >= v] if ge else tr[tr[col] <= v]
        b = un[un[col] >= v] if ge else un[un[col] <= v]
        sa, sb = K.wstats(a["r"], K.weights(a["time"])), K.wstats(b["r"], K.weights(b["time"]))
        rows.append({"value": v, "train": sa, "unseen": sb})
    ok = [x for x in rows if x["train"]["eff_n"] >= 30 and x["train"]["mean"] is not None]
    best = max(ok, key=lambda x: x["train"]["mean"]) if ok else None
    return {"rows": rows, "best": None if best is None else best["value"],
            "best_train": None if best is None else best["train"]["mean"], "best_unseen": None if best is None else best["unseen"]["mean"],
            "best_unseen_n": None if best is None else best["unseen"]["n"]}


def _hours(t):
    """hour-of-day filter (Israel time): the hours positive on the train period, shown on 2026"""
    from agents import agentkit as K
    t = t.copy()
    t["hr"] = t["time"].dt.tz_convert("Asia/Jerusalem").dt.hour
    tr, un = t[t["time"] < SPLIT], t[t["time"] >= SPLIT]
    g = tr.groupby("hr")["r"].agg(["mean", "count"])
    good = sorted(int(h) for h, x in g.iterrows() if x["count"] >= 15 and x["mean"] > 0)
    b = un[un["hr"].isin(good)]
    return {"hours_il": good, "unseen": K.wstats(b["r"], K.weights(b["time"])) if len(b) else None,
            "unseen_all": K.wstats(un["r"], K.weights(un["time"])) if len(un) else None}


def summarize(bot, symbols):
    """Results per pair and config (complete ones only), the open questions' answers, the best config per pair
    (chosen on the train period). Written to data/agents/strategies/tal_gap_results.json under the bot."""
    idx = load_index(bot)
    res = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "symbols": symbols, "split": "2026-01-01",
           "rules_version": V, "pairs": {}, "planned": len(pairs()) * len(CONFIGS) * len(symbols),
           "done": sum(1 for v in idx.values() if v["sym"] in symbols and v["cfg"].get("V") == V)}
    pooled = []
    for H, L, st in pairs():
        pn = pair_name(H, L)
        P = {"H": H, "L": L, "steps": st, "configs": {}}
        trades = {}
        for name, _, _ in CONFIGS:
            t = trades_of(bot, idx, H, L, cfg_of(name), symbols)
            if t is None:
                continue
            trades[name] = t
            P["configs"][name] = stats(t)
        if "base" in trades and len(trades["base"]):
            b = trades["base"]
            P["answers"] = {"impulse_atr": _thr(b, "imp_atr", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]),
                            "impulse_candles": _thr(b, "imp_n", [1, 2, 3, 5, 8]),
                            "gap_atr": _thr(b, "gap_atr", [0.0, 0.1, 0.25, 0.5, 1.0]),
                            "min_rr": _thr(b, "rr", [0.0, 0.5, 1.0, 1.5, 2.0]),
                            "hours": _hours(b)}
            sws = {n_: P["configs"][n_]["train"]["mean"] for n_ in ("sw1", "base", "sw3", "sw5") if n_ in P["configs"]}
            if len(sws) > 1:
                bs = max(sws, key=lambda k_: sws[k_] if sws[k_] is not None else -9)
                P["answers"]["swing_w"] = {"best": {"sw1": 1, "base": 2, "sw3": 3, "sw5": 5}[bs], "train": sws,
                                           "unseen": P["configs"][bs]["unseen"]["mean"]}
            if st == 2:
                pooled.append(b.assign(pair=pn))
        comp = {n_: v for n_, v in P["configs"].items() if v["train"]["mean"] is not None and v["train"]["eff_n"] >= 30}
        if comp:
            bn = max(comp, key=lambda k_: comp[k_]["train"]["mean"])
            pos = sum(1 for v in comp.values() if (v["unseen"]["mean"] or -9) > 0)
            P["best"] = {"config": bn, "cfg": cfg_of(bn), "tested": len(comp), "share_positive": round(pos / len(comp), 2), **comp[bn]}
            if bn in trades:
                P["best"]["risk_pct"] = risk_from_history(trades[bn])
        res["pairs"][pn] = P
    if pooled:
        b = pd.concat(pooled, ignore_index=True)
        res["answers_two_steps"] = {"impulse_atr": _thr(b, "imp_atr", [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]),
                                    "impulse_candles": _thr(b, "imp_n", [1, 2, 3, 5, 8]),
                                    "gap_atr": _thr(b, "gap_atr", [0.0, 0.1, 0.25, 0.5, 1.0]),
                                    "min_rr": _thr(b, "rr", [0.0, 0.5, 1.0, 1.5, 2.0])}
    try:
        allr = json.loads(RESULTS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        allr = {}
    allr.setdefault("note", "Tal's gap method (data/agents/strategies/tal_gap.md) - results per bot and timeframe pair; "
                            "the rules: agents/tal_gap.py. Chosen on the period to 2025-12-31, shown on 2026; time-weighted.")
    allr.setdefault("bots", {})[bot] = res
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(allr, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def risk_from_history(t, bot_max=0.4):
    """risk per trade (%) so that the worst day in the history costs <= 2.5% (half the 5% daily limit) and the
    deepest drawdown <= 5% (half the 10% total limit); at most the bot's own risk"""
    if t is None or t.empty:
        return None
    day = t.groupby(t["time"].dt.tz_convert("Europe/Prague").dt.date)["r"].sum()
    worst = max(0.5, -float(day.min()))
    eq = t["r"].cumsum()
    dd = max(0.5, float((eq.cummax() - eq).max()))
    r = min(bot_max, 2.5 / worst, 5.0 / dd)
    return round(max(0.025, np.floor(r / 0.025) * 0.025), 3)


def cards(bot, res, risk_max=0.4):
    """One card per pair (its config chosen on the train period, shown on 2026) and one with the answers."""
    from agents import agentkit as K
    out = []
    for pn, P in res["pairs"].items():
        b = P.get("best")
        if not b:
            continue
        H, L = P["H"], P["L"]
        nm = f"{TF_HE[H]}/{TF_HE[L]}"
        prot = P["configs"].get("prot", {}).get("unseen", {}).get("mean")
        base = P["configs"].get("base", {}).get("unseen", {}).get("mean")
        what = (f"שיטת הגאפ של טל: מזהים ב־{TF_HE[H]}, סטופ ויעד מסווינגים ב־{TF_HE[L]}. הגרסה שנבחרה על 2024-2025: "
                f"{CFG_HE.get(b['config'], b['config'])}. {b['per_day']} עסקאות ביום בממוצע. "
                f"ב־2026 (לא נראתה): בלי הגנה {base:+.2f}R, עם ההגנה {prot:+.2f}R לעסקה." if base is not None and prot is not None else
                f"שיטת הגאפ של טל: מזהים ב־{TF_HE[H]}, סטופ ויעד ב־{TF_HE[L]}. הגרסה שנבחרה על 2024-2025: {CFG_HE.get(b['config'], b['config'])}.")
        out.append(K.card(bot, "new", f"שיטת הגאפ של טל {nm}: {CFG_HE.get(b['config'], b['config'])}", what,
                          b["train"], b["unseen"], whole=b["whole"], halves_=b["halves"], cost_r=b["cost_r"],
                          neighbours={"values": b["tested"], "share_positive": b["share_positive"]}, tested=b["tested"],
                          risk_pct=b.get("risk_pct"), detail={"pair": pn, "steps": P["steps"], "per_day": b["per_day"],
                                                             "unseen_by_sym": b.get("unseen_by_sym")},
                          change={"track": "tal", "pair": pn, "H": H, "L": L, "config": b["config"], "cfg": b["cfg"],
                                  "symbols": res["symbols"], "reset_account": True}))
    return out
