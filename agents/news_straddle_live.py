"""EV04 live - the straddle before a release in bot 7 (card bfdb13a3a001, 15.11.2026; study agents/news_straddle.py).

The version chosen on 2025 among those with at most 9 trades per release ("pre|x1|opp|be1|h30|cown|sp0|cap9"):
  * at the open of the minute before a high-impact / FTMO-restricted release of the instrument's own currencies:
    Buy Stop at open + X x ATR and Sell Stop at open - X x ATR (X = 1, ATR(14) of the last 5-minute bar closed before
    that minute); OCO - the first filled cancels the other; not filled by the end of minute +4 -> no trade;
  * the fill is worse than the order by half the spread + 0.1 ATR (a gap through the order fills at the bar's open);
  * stop at the other order's level; at +1R the stop goes to the entry; no target; out after 30 minutes;
  * a minute that touches both orders counts as a full loss (the order inside a 1-minute candle is unknown);
  * at most 9 trades per release over all instruments (live/news_bot.py applies the cap: first by fill time).
Exactly agents/news_straddle.trades_sym + _exit, on the candles the bot holds. The levels are fixed before the release,
so reading closed candles after it never uses the future (a virtual account; inside FTMO's restricted window - never
for a real account). The spread the bot knows is bot 1's median, not the release minute's: SPREAD_X x it is charged
(the release minute's spread is several times the median) - more cost than the study, never less.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents import rules_agent as R

SPREAD_X = 3.0          # x bot 1's median spread, for the fill minute of a release

# The release types EV04 may skip (card "skip_groups" in the plan, bot 7 smart agent 16.11.2026): "<currency>:<type>",
# the type exactly agents/news_fix.category (copied here so the live bot does not load the study's data).
PRIO = ("rates", "infl", "jobs", "gdp", "oil", "other")


def category(title):
    t = str(title).lower()
    if "oil" in t:
        return "oil"
    if any(k in t for k in ("rate", "fomc", "monetary", "policy", "mpc", "statement", "minutes", "inflation letter")):
        return "rates"
    if any(k in t for k in ("cpi", "ppi", "rpi")):
        return "infl"
    if any(k in t for k in ("employment", "unemployment", "payroll", "claims", "labor")):
        return "jobs"
    if "gdp" in t:
        return "gdp"
    return "other"


def release_groups(items):
    """FTMO calendar items -> {release time (UTC): [(currency, type)]} of the high-impact / restricted ones (as
    agents/news_fix.releases)."""
    out = {}
    for i in items or []:
        if i.get("impact") != "high" and not i.get("restriction"):
            continue
        ins = str(i.get("instrument", ""))
        cur = "Crude Oil" if ins.startswith("Crude") else (ins.split(" ")[0] if ins else "")
        try:
            t = pd.Timestamp(i["date"]).tz_convert("UTC")
        except Exception:  # noqa: BLE001
            continue
        out.setdefault(t, []).append((cur, category(i.get("title", ""))))
    return out


def groups_of(sym, t, groups, curs):
    """The release's type(s) for this instrument, as agents/news_fix.tag: its own currencies' items first, then the
    most important type (PRIO) - every item of that type ({"USD:gdp", ...}; a tie keeps all, so the order of the
    calendar never decides)."""
    items = groups.get(t, [])
    mine = [x for x in items if x[0] in (curs or [])]
    use = mine or items
    if not use:
        return {"?:other"}
    k = min(PRIO.index(x[1]) for x in use)
    return {f"{c}:{g}" for c, g in use if PRIO.index(g) == k}


def skip_events(events, sym, groups, curs, skip):
    """The release times left after the skipped types (skip: ["USD:gdp", ...]; empty -> all)."""
    if not skip:
        return list(events)
    sk = set(skip)
    out = []
    for e in events:
        t = pd.Timestamp(e)
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        if not (groups_of(sym, t, groups, curs) & sk):
            out.append(e)
    return out


def _exit(o, h, l, c, q0, end, dr, e, stop, tgt, be):
    """agents/news_straddle._exit (copied, so the live bot does not import the study's data loaders)."""
    risk = abs(e - stop)
    s = stop
    if (l[q0] <= s) if dr > 0 else (h[q0] >= s):
        return s, q0
    moved = False
    for q in range(q0 + 1, end):
        if (l[q] <= s) if dr > 0 else (h[q] >= s):
            return s, q
        if tgt is not None and ((h[q] >= tgt) if dr > 0 else (l[q] <= tgt)):
            return tgt, q
        if be is not None and not moved and ((h[q] - e) if dr > 0 else (e - l[q])) >= be * risk:
            s, moved = e, True
    return c[end - 1], end - 1


def _atr5(d1):
    d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    a = R.atr(d5["h"].to_numpy(), d5["l"].to_numpy(), d5["c"].to_numpy(), 14)
    return pd.Series(a, index=d5.index)


def live_straddle(d1, events, spread_px=0.0, x=1.0, hold=30, be=1.0, cancel=5, slip_atr=0.1, start=-1, fill_from=None):
    """d1: the bot's 1-minute candles (o/h/l/c, UTC index); events: release times (UTC) of the instrument's currencies.
    fill_from (plan "fill_from", bot 7 smart agent 16.11.2026): the first minute after the release a fill may come in
    (0 = the release minute); an order touched earlier (the minute before the release) cancels both - no trade.
    -> list of dicts: time (fill minute), exit_time, dir, entry_px, stop_px, exit_px, r (gross, in R), rel, open."""
    if d1 is None or len(d1) < 120 or not events:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    idx = d1.index
    o, h, l, c = (d1[k].to_numpy(float) for k in "ohlc")
    a5 = _atr5(d1)
    sp = float(spread_px or 0.0) * SPREAD_X
    out = []
    for t in sorted(set(pd.Timestamp(e) for e in events)):
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        if t < idx[0] + pd.Timedelta(hours=2) or t > idx[-1]:
            continue
        k0 = idx.searchsorted(t)
        if k0 >= len(idx) or idx[k0] != t:
            continue
        kp = k0 + start
        if kp < 0 or idx[kp] - t != pd.Timedelta(minutes=start):
            continue
        ka = a5.index.searchsorted(idx[kp].floor("5min")) - 1
        if ka < 14:
            continue
        atr = float(a5.iloc[ka])
        if not np.isfinite(atr) or atr <= 0:
            continue
        ref = o[kp]
        buy, sell = ref + x * atr, ref - x * atr
        last = k0 + cancel
        if last > len(idx):                       # the cancel window is not over yet: wait for it
            if not any(h[q] >= buy or l[q] <= sell for q in range(kp, len(idx))):
                continue
        q0, dr, both = None, 0, False
        for q in range(kp, min(last, len(idx))):
            if idx[q] - t > pd.Timedelta(minutes=cancel):
                break
            up, dn = h[q] >= buy, l[q] <= sell
            if up or dn:
                q0, dr, both = q, (1 if up else -1), bool(up and dn)
                break
        if q0 is None:
            continue
        if fill_from is not None and q0 < k0 + int(fill_from):
            continue                                 # touched before the allowed minute: both orders cancelled
        slip = 0.5 * sp + slip_atr * atr
        lvl = buy if dr > 0 else sell
        e = max(lvl, o[q0]) + slip if dr > 0 else min(lvl, o[q0]) - slip
        stop = sell if dr > 0 else buy
        risk = dr * (e - stop)
        if risk <= 0:
            continue
        end_t = idx[q0] + pd.Timedelta(minutes=hold)
        end = int(idx.searchsorted(end_t))           # bars up to (not incl.) fill + hold
        done = end < len(idx) or idx[-1] >= end_t
        end = min(max(end, q0 + 1), len(idx))
        if both:
            px, qx, done = stop, q0, True
        else:
            px, qx = _exit(o, h, l, c, q0, end, dr, e, stop, None, be)
            if qx < end - 1 or (qx == end - 1 and px != c[end - 1]):
                done = True                          # stopped (at the stop or at the entry) before the time exit
        out.append({"time": idx[q0], "exit_time": idx[qx], "dir": dr, "entry_px": e, "stop_px": stop,
                    "exit_px": float(px), "r": dr * (px - e) / risk, "rel": t, "open": not done})
    return out
