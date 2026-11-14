"""Bot 7's entry-timing study (fix attempt 6 for EV01, 13.11.2026).

Why: EV01 (follow the first 10 minutes after a release when they moved 1.5 ATR) makes money from 30.04.2025 to
31.12.2025 and loses in 2026 (11 instruments, GRID 3 of agents/news_fix.py: +7.1R / -27.8R). Five attempts on the
exits, release types, instruments, market filters, the first-pullback entry and the Fibonacci breakeven did not fix
it. What was never varied is WHEN and ON WHAT the entry happens (the user's list: "minutes after the release",
the release's surprise, the spike size):
  window     the move measured over the first 10 / 15 / 20 / 30 minutes (entry at that minute; FTMO: never inside
             the 3 minutes around a restricted release, so 10 is the earliest)
  spike      the move at least 0.75 / 1.5 / 2.5 ATR (5-minute ATR before the release)
  side       with the move (EV01) or against it (a fade) - chosen on 2025 like every other knob, never on 2026
  surprise   all trades / only when the side agrees with actual-vs-forecast for the instrument (FX and gold only:
             an index has no fixed direction for a strong number) / only without a usable surprise
  entry      at once / a limit 0.5R better (the first pullback, valid 3 bars)
  cap        at most 1 / 2 / 3 trades per release, or no cap (attempt 7, 13.11.2026: one PPI = 8 stops)
  target     2R / 3R; stop beyond the window's other end; out after an hour (the live bot's breakeven at 60%)
Costs: FTMO's spread at the entry minute (Dukascopy) and commission (agents/ftmo_costs.py), as agents/news_fix.py.
Fair method: chosen on 30.04.2025-31.12.2025 by total R, shown on 2026, time-weighted (agents/agentkit.py);
robustness: of the 10 best on 2025, how many made money in 2026.

EV03 (fix attempt 5, 13.11.2026 night): the same timing question for the range breakout. EV03's loss in 2026 is all
in US releases and its built version (pullback 0.5R, target 3R) only stops the loss (+0.2R). Varied here:
  range      the first 10 / 15 / 20 / 30 minutes after the release set the range (the live bot: 15)
  side       the first 5-minute close outside it within the next hour: with the break, or against it (fade:
             stop beyond the break bar's extreme + 0.1 ATR)
  entry      at once / a limit 0.5R better (3 bars)    target 1.5R / 2R / 3R    surprise and cap as above
Kept in its own cache (cache/news_entry/<sym>_ev03.json) and its own result ("study_ev03"), so EV01's rows stay.

PRE (fix attempt 6 for EV03 and a wider sample for EV04, 14.11.2026): the only release strategy that held out of
sample so far is the jobs box (news_improve2.py: the 10 minutes BEFORE the release make the box, the breakout
after it; 2025 17 trades +0.32R, 2026 19 trades +0.23R) - EV03 draws its range AFTER the release and loses in US
releases. Same machinery as EV03, the range moved before the release:
  box        the 10 / 15 / 20 / 30 minutes before the release (2 / 3 / 4 / 6 five-minute bars)
  side       the first 5-minute close outside it, from the release bar (closes at minute 5) through minute 30:
             with the break (stop the box's other side) or against it (stop beyond the break bar + 0.1 ATR)
  entry      at once / a limit 0.5R better (3 bars)    target 1.5R / 2R / 3R    out after an hour
  surprise and cap as above; and every release type on its own ("by_cat": the best variant chosen on 2025 within
  that type, shown on 2026), so the jobs box can be checked on all 16 instruments and the other types beside it.
Own cache (cache/news_entry/<sym>_pre.json) and result ("study_pre").

Output: reports/agents/bot7/entry_study.json. Cached per instrument (cache/news_entry/<sym>.json), resumes after a
time budget cut. Runs on the bots' computer inside agent_portfolio.py --bot bot7, after the fix study.
  python -m agents.news_entry [--minutes 10]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents import rules_agent as R  # noqa: E402,F401  (first: rules_agent and event_strategies import each other)
from agents import event_strategies as E  # noqa: E402
from agents import ftmo_costs as C  # noqa: E402
from agents import ftmo_universe as U  # noqa: E402
from agents import news_fix as NF  # noqa: E402
from agents import poc_agent as P  # noqa: E402
from agents.rules_agent import atr  # noqa: E402

OUT = ROOT / "reports" / "agents" / "bot7" / "entry_study.json"
CACHE = ROOT / "cache" / "news_entry"
VERSION = 1
WINDOWS = (2, 3, 4, 6)            # 5-minute bars after the release: entry at minute 10 / 15 / 20 / 30
SPIKES = (0.75, 1.5, 2.5)
SIDES = (1, -1)                   # +1 with the move, -1 against it
LIMITS = (0.0, 0.5)
TARGETS = (2.0, 3.0)
SURPRISE = ("all", "agree", "none")
CAPS = (0, 1, 2, 3)               # most trades per release (0 = no cap) - fix attempt 7
TOP = 10
STALE_DAYS = 3
BOX_VERSION = 1
BOX_TARGETS = (1.5, 2.0, 3.0)
BOX_BASE = "b15|with|l0.5|t3|sall"     # EV03 as it trades now (card 340e3e2fcd30)
PRE_VERSION = 1
PRE_BASE = "p10|with|l0|t2|sall"       # the closest to the jobs box card faee9efdd86f (box 10, market entry)
MIN_CAT_N = 8                          # a release type needs this many 2025 trades to choose a variant in it


def strat(w, k, side):
    def fn(df):
        o, h, l, c = (df[x].to_numpy() for x in "ohlc")
        a = atr(h, l, c, 14)
        n = len(c)
        long_, short = np.zeros(n, bool), np.zeros(n, bool)
        sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
        for k0 in E._event_bars(df):
            k1 = k0 + w - 1
            if k1 + 1 >= n or k0 < 20 or df.index[k1] - df.index[k0] != pd.Timedelta(minutes=5 * (w - 1)):
                continue
            move, ref = c[k1] - o[k0], a[k0 - 1]
            if not np.isfinite(ref) or ref <= 0 or abs(move) < k * ref:
                continue
            hi, lo = h[k0:k1 + 1].max(), l[k0:k1 + 1].min()
            d = int(np.sign(move)) * side
            if d > 0:
                long_[k1], sl_l[k1] = True, lo if side > 0 else lo - 0.1 * ref
            else:
                short[k1], sl_s[k1] = True, hi if side > 0 else hi + 0.1 * ref
        return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0), "max_bars": 12}
    return fn


def strat_box(w, side):
    """EV03 with a variable range: the first `w` 5-minute bars after the release; the first close outside it in the
    next 12 bars enters with the break (side +1, stop the range's other end) or against it (side -1, stop beyond
    the break bar's extreme + 0.1 ATR)."""
    def fn(df):
        h, l, c = (df[x].to_numpy() for x in "hlc")
        a = atr(h, l, c, 14)
        n = len(c)
        long_, short = np.zeros(n, bool), np.zeros(n, bool)
        sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
        for k0 in E._event_bars(df):
            k2 = k0 + w - 1
            if k2 + 13 >= n or k0 < 20 or df.index[k2] - df.index[k0] != pd.Timedelta(minutes=5 * (w - 1)):
                continue
            ref = a[k0 - 1]
            if not np.isfinite(ref) or ref <= 0:
                continue
            hi, lo = h[k0:k2 + 1].max(), l[k0:k2 + 1].min()
            for k in range(k2 + 1, k2 + 13):
                up, dn = c[k] > hi, c[k] < lo
                if not (up or dn):
                    continue
                d = (1 if up else -1) * side
                if d > 0:
                    long_[k], sl_l[k] = True, lo if side > 0 else l[k] - 0.1 * ref
                else:
                    short[k], sl_s[k] = True, hi if side > 0 else h[k] + 0.1 * ref
                break
        return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 1.5), "max_bars": 24}
    return fn


def strat_pre(w, side):
    """The range moved before the release: the `w` 5-minute bars before it make the box; the first close outside it
    from the release bar through minute 30 enters with the break (stop the box's other side) or against it (stop
    beyond the break bar's extreme + 0.1 ATR). Entry at the next bar's open (minute 5 at the earliest: outside
    FTMO's 2 minutes around a restricted release)."""
    def fn(df):
        h, l, c = (df[x].to_numpy() for x in "hlc")
        a = atr(h, l, c, 14)
        n = len(c)
        long_, short = np.zeros(n, bool), np.zeros(n, bool)
        sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
        for k0 in E._event_bars(df):
            if k0 < 20 or k0 + 18 >= n or df.index[k0] - df.index[k0 - w] != pd.Timedelta(minutes=5 * w):
                continue
            ref = a[k0 - w - 1]
            if not np.isfinite(ref) or ref <= 0:
                continue
            hi, lo = h[k0 - w:k0].max(), l[k0 - w:k0].min()
            for k in range(k0, k0 + 6):
                if df.index[k] - df.index[k0] > pd.Timedelta(minutes=25):
                    break
                up, dn = c[k] > hi, c[k] < lo
                if not (up or dn):
                    continue
                d = (1 if up else -1) * side
                if d > 0:
                    long_[k], sl_l[k] = True, lo if side > 0 else l[k] - 0.1 * ref
                else:
                    short[k], sl_s[k] = True, hi if side > 0 else h[k] + 0.1 * ref
                break
        return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0), "max_bars": 12}
    return fn


def vkey_pre(w, side, lim, tg):
    return f"p{5 * w}|{'with' if side > 0 else 'fade'}|l{lim:g}|t{tg:g}"


def grid_box():
    return list(itertools.product(WINDOWS, SIDES, LIMITS, BOX_TARGETS))


def vkey_box(w, side, lim, tg):
    return f"b{5 * w}|{'with' if side > 0 else 'fade'}|l{lim:g}|t{tg:g}"


def grid():
    return list(itertools.product(WINDOWS, SPIKES, SIDES, LIMITS, TARGETS))


def vkey(w, k, side, lim, tg):
    return f"m{5 * w}|k{k:g}|{'with' if side > 0 else 'fade'}|l{lim:g}|t{tg:g}"


def trades_sym(sym, rel, family="EV01"):
    d1 = P.bars(sym, "1Min", NF.LOAD_FROM, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    if d1 is None or d1.empty:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, NF.LOAD_FROM, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    except Exception:  # noqa: BLE001
        sp = None
    d5 = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    d5 = d5.dropna(subset=["o"])
    d5.attrs.update(symbol=sym, tf="5Min")
    evs = E.events_for(sym)
    spec = U.spec(sym)
    px = float(d5["c"].median())
    comm = (px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
            if spec["commissionType"] == "flat_USD" else px * 2 * float(spec["commission"]) / 100.0)
    spm = float(sp.median()) if sp is not None and len(sp) else 0.0
    rows = []
    if family == "EV03":
        variants = [(vkey_box(w, side, lim, tg), strat_box(w, side), lim, tg) for w, side, lim, tg in grid_box()]
    elif family == "PRE":
        variants = [(vkey_pre(w, side, lim, tg), strat_pre(w, side), lim, tg) for w, side, lim, tg in grid_box()]
    else:
        variants = [(vkey(w, k, side, lim, tg), strat(w, k, side), lim, tg) for w, k, side, lim, tg in grid()]
    for vk, base, lim, tg in variants:

        def fn(d, base=base, lim=lim, tg=tg):
            L, S, p = base(d)
            p = dict(p, target=("R", float(tg)))
            if lim:
                p["entry_pullback"] = (float(lim), 3)
            return L, S, p
        t = R.run(d5, fn, 0.0, breakeven=(0.6, (spm + comm) / px))
        if t is None or not len(t):
            continue
        t = t[t["reason"] != "סוף הנתונים"]
        if not len(t):
            continue
        net = C.apply(t.reset_index(drop=True), sym, sp)
        for x in net.itertuples():
            tt = pd.Timestamp(x.time)
            if tt < NF.TEST_FROM:
                continue
            g = NF.tag(sym, tt, rel, evs)
            if g is None:
                continue
            agree = None if g["surp"] is None else int(np.sign(g["surp"]) == np.sign(x.dir))
            rows.append({"v": vk, "sym": sym, "cls": U.CLASS.get(sym, "?"), "t": tt.isoformat(),
                         "dir": int(x.dir), "r": float(x.r), "cat": g["cat"], "agree": agree, "rel": g["rel"].isoformat()})
    return rows


def capped(z, cap):
    """Fix attempt 7 (13.11.2026): at most `cap` trades per release. Live on 13.11 one US PPI release opened 8 EV01
    longs at minute 10 (EURUSD, GBPUSD, AUDUSD, NZDUSD, US30, US100, US500 ...) and all 8 hit the stop together
    (-9.3R): the instruments are one bet on the dollar, not 8. The kept trades are the first by entry time, then by
    the fixed instrument order of the bot (NF.SYMBOLS) - never by result."""
    if not cap or z.empty:
        return z
    order = {s_: i for i, s_ in enumerate(NF.SYMBOLS)}
    z = z.assign(_o=z["sym"].map(order).fillna(99)).sort_values(["rel", "t", "_o"])
    return z[z.groupby("rel").cumcount() < cap].drop(columns="_o")


def study(df, base_key="m10|k1.5|with|l0|t2|sall"):
    df = df.assign(train=pd.to_datetime(df["t"], utc=True) < NF.SPLIT)
    cand = []
    for v, y in df.groupby("v"):
        for sf in SURPRISE:
            z0 = y if sf == "all" else (y[y["agree"] == 1] if sf == "agree" else y[y["agree"].isna()])
            for cap in CAPS:
                z = capped(z0, cap)
                tr, un = z[z.train], z[~z.train]
                cand.append({"v": f"{v}|s{sf}" + (f"|c{cap}" if cap else ""), "train": NF._w(tr), "unseen": NF._w(un)})
    cand.sort(key=lambda c: -c["train"]["sum"])
    top = cand[:TOP]
    base = next((c for c in cand if c["v"] == base_key), None)   # the strategy as it trades now
    by = {}
    for key, pat in (("window", r"^[mbp](\d+)\|"), ("spike", r"\|k([\d.]+)\|"), ("side", r"\|(with|fade)\|"),
                     ("surprise", r"\|s([a-z]+)"), ("cap", r"(?:\|c(\d+))?$")):
        g = {}
        for c in cand:
            m = re.search(pat, c["v"])
            if m:
                g.setdefault(m.group(1) or "0", []).append(c)
        # the median variant of each value (not the best: a whole family of variants either holds or not)
        by[key] = {k_: {"variants": len(cs), "train_median_sum": round(float(np.median([c["train"]["sum"] for c in cs])), 2),
                        "unseen_median_sum": round(float(np.median([c["unseen"]["sum"] for c in cs])), 2),
                        "unseen_positive": int(sum(1 for c in cs if (c["unseen"]["sum"] or 0) > 0))}
                   for k_, cs in g.items()}
    return {"baseline": base, "best": cand[0] if cand else None, "top": top,
            "robust": {"top_positive_unseen": int(sum(1 for c in top if (c["unseen"]["sum"] or 0) > 0)), "top": len(top),
                       "top_better_than_base": int(sum(1 for c in top if base and (c["unseen"]["sum"] or 0) > (base["unseen"]["sum"] or 0)))},
            "by": by, "tested": len(cand)}


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS) or j.get("partial")
                or j.get("version") != VERSION or j.get("partial_ev03", True) or j.get("box_version") != BOX_VERSION
                or j.get("partial_pre", True) or j.get("pre_version") != PRE_VERSION)
    except Exception:  # noqa: BLE001
        return True


def by_cat(df):
    """Every release type on its own: the variant with the most total R on 2025 within that type (at least MIN_CAT_N
    trades there), shown on 2026; plus how many of the type's 10 best on 2025 made money in 2026."""
    out = {}
    for cat, y in df.groupby("cat"):
        r_ = study(y, PRE_BASE)
        top = [c for c in r_["top"] if c["train"]["n"] >= MIN_CAT_N]
        if not top:
            continue
        out[cat] = {"best": top[0], "baseline": r_["baseline"],
                    "top_positive_unseen": int(sum(1 for c in top if (c["unseen"]["sum"] or 0) > 0)), "top": len(top)}
    return out


def _family(fam, ver, suffix, rel, t_end):
    """Rows of one family over all instruments, per-instrument cache, resumes after a time budget cut."""
    rows, done = [], []
    for sym in NF.SYMBOLS:
        f = CACHE / f"{sym}{suffix}.json"
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
            if c.get("version") == ver and pd.Timestamp.now(tz="UTC") - pd.Timestamp(c["at"]) < pd.Timedelta(days=STALE_DAYS):
                rows += c["rows"]
                done.append(sym)
                continue
        except Exception:  # noqa: BLE001
            pass
        if time.time() > t_end:
            print(f"{fam}: time budget reached after {done}")
            continue
        try:
            r_ = trades_sym(sym, rel, fam)
            rows += r_
            f.write_text(json.dumps({"version": ver, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": r_}), encoding="utf-8")
            done.append(sym)
            print(f"{fam} {sym}: {len(rows)} rows so far", flush=True)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"{fam} {sym}: failed ({e})")
    return rows, done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.force and not stale():
        print("entry study fresh - skipped")
        return
    t_end = time.time() + a.minutes * 60
    rel = NF.releases()
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, done = _family("EV01", VERSION, "", rel, t_end)
    if not done:                      # always make progress on EV01 first (it was the first in the queue)
        rows, done = _family("EV01", VERSION, "", rel, time.time() + a.minutes * 60)
    rows_b, done_b = _family("EV03", BOX_VERSION, "_ev03", rel, t_end)
    df = pd.DataFrame(rows)
    res = study(df) if len(df) else {}
    df_b = pd.DataFrame(rows_b)
    res_b = study(df_b, BOX_BASE) if len(df_b) else {}
    rows_p, done_p = _family("PRE", PRE_VERSION, "_pre", rel, t_end) if len(done_b) == len(NF.SYMBOLS) else ([], [])
    df_p = pd.DataFrame(rows_p)
    res_p = study(df_p, PRE_BASE) if len(df_p) else {}
    if res_p:
        res_p["by_cat"] = by_cat(df_p.assign(train=pd.to_datetime(df_p["t"], utc=True) < NF.SPLIT))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "box_version": BOX_VERSION,
                               "symbols": done, "partial": len(done) < len(NF.SYMBOLS), "rows": int(len(df)),
                               "symbols_ev03": done_b, "partial_ev03": len(done_b) < len(NF.SYMBOLS), "rows_ev03": int(len(df_b)),
                               "split": str(NF.SPLIT.date()), "test_from": str(NF.TEST_FROM.date()), "study": res,
                               "study_ev03": res_b, "pre_version": PRE_VERSION, "symbols_pre": done_p,
                               "partial_pre": len(done_p) < len(NF.SYMBOLS), "rows_pre": int(len(df_p)), "study_pre": res_p},
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for name, r_ in (("EV01", res), ("EV03", res_b), ("PRE", res_p)):
        if r_:
            print(name, json.dumps({"base": r_["baseline"], "best": r_["best"], "robust": r_["robust"]}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
