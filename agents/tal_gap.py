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
KNOBS = {"GBODY": [1.0, 2.0], "MIN_GAP": [0.0, 0.6], "SW": [2, 5], "MIN_RR": [0.5, 1.5, 2.0],
         "EXPIRY": [8, 24, 120], "PROT": ["cal15", "cal60", "shock", "cal15shock"],
         # trade management (user, 12.11.2026: "partial exit? stop to break-even? the smart agent should optimise it"):
         # beW = the stop to the entry once the price went W of the way to the target; pF_W = F of the trade out at W of
         # the way, the rest's stop to the entry. Not in BASE (the base stays "none"), so earlier runs keep their names.
         "EXIT": ["be50", "be70", "p30_50", "p50_50", "p50_70", "p70_70"]}
NAMES = {"GBODY": "עוצמת הדחף (גוף הנר האמצעי מול הממוצע)", "MIN_GAP": "גודל גאפ מינימלי (מול הטווח הממוצע)",
         "SW": "סווינג בטיים פריים הנמוך (נרות מכל צד)", "MIN_RR": "יחס מינימלי יעד/סטופ", "EXPIRY": "תוקף הגאפ (שעות)",
         "PROT": "הגנה מהודעות וזעזועים", "EXIT": "ניהול העסקה: יציאה חלקית / סטופ לנקודת הכניסה"}
PROT_HE = {"none": "בלי הגנה", "cal15": "יומן ±15 דק'", "cal60": "יומן ±60 דק'", "shock": "זעזועים",
           "cal15shock": "יומן ±15 דק' + זעזועים"}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
SHOCK_X = 4.0
MERGE_ATR = 0.3
CALM_MIN = 5
RERUN_DAYS = 14
RESULTS = Path("data/agents/strategies/tal_gap_results.json")


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
    out = []
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
    cache[key] = out
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


def exit_mode(cfg):
    """(fraction out, way to the target) for the EXIT knob; (0, way) = only the stop to the entry; None = no management."""
    m = cfg.get("EXIT", "none")
    if m == "none":
        return None
    if m.startswith("be"):
        return 0.0, int(m[2:]) / 100
    f, w = m[1:].split("_")
    return int(f) / 100, int(w) / 100


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
        stop = s2 - d * D.buf
        risk = d * (px - stop)
        if risk <= 0 or d * (t2 - px) / risk < cfg["MIN_RR"]:
            continue
        cands.append((f, d, px, stop, t2, risk))
    cands.sort()
    rows, busy = [], -1
    for f, d, px, stop, tgt, risk in cands:
        if f < busy:
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
    print(f"tal_gap: {done} run, {len(todo) - done} left")
    return idx, len(todo) - done


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
        if not rows:
            continue
        base = next((r for r in rows if r["knob"] == "base"), None)
        ok = [r for r in rows if (r["train"]["eff_n"] or 0) >= K.MIN_EFF_N and r["train"]["mean"] is not None]
        chosen = max(ok, key=lambda r: r["train"]["mean"]) if ok else None
        prot = {r["val"] if r["knob"] == "PROT" else "none": {"train": r["train"]["mean"], "unseen": r["unseen"]["mean"],
                                                             "n": r["whole"]["n"]}
                for r in rows if r["knob"] in ("PROT", "base")}
        pairs[pair_name(p)] = {"gap": p[2], "name": f"{TF_HE[p[0]]} / {TF_HE[p[1]]}", "done": len(rows),
                               "planned": len(variants()), "base": base, "chosen": chosen, "protection": prot,
                               "variants": [{k: r[k] for k in ("knob", "val", "train", "unseen", "whole")} for r in rows]}
        if base and len(rows) == len(variants()):
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


def write_results(bot, pairs, answers, world, progress):
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
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return doc


def cards(bot, pairs, other=None, other_name="בוט 5", risk_pct=None):
    """A 'new' card per pair whose train-chosen variant made money on the unseen period."""
    out = []
    for pn, x in pairs.items():
        ch = x.get("chosen")
        if not ch or x["done"] < x["planned"]:
            continue
        u = ch["unseen"]
        if u["mean"] is None or u["mean"] <= 0:
            continue
        try:
            ob = ((other or {}).get("pairs") or {}).get(pn, {}).get("chosen") or {}
            om = (ob.get("unseen") or {}).get("mean")
        except Exception:  # noqa: BLE001 - the other agent's file in another shape
            ob, om = {}, None
        cfg = ch["cfg"]
        same = f" · ב{other_name}: {om:+.2f}R בתקופה שלא נראתה ({ob.get('knob')}={ob.get('val')})" if om is not None \
            else f" · ב{other_name}: עוד לא נבדק"
        knob = "הבסיס" if ch["knob"] == "base" else "" if ch["knob"] == "PROT" else f"{NAMES[ch['knob']]} = {ch['val']}"
        what = (f"שיטת הגאפ של טל, {x['name']} (פער {x['gap']} דרגות): {knob + ', ' if knob else ''}{PROT_HE.get(cfg['PROT'])}. "
                f"נבחר על 2024-2025, מוצג על 2026: {u['mean']:+.2f}R לעסקה, {u['n']} עסקאות, {u['win']}% הצלחה, "
                f"{ch['target_pct']}% הגיעו ליעד, יחס חציוני {ch['rr']}, עלות {ch['cost_r']}R{same}")
        c = K.card(bot, "new", f"שיטת הגאפ של טל · {x['name']}", what, ch["train"], u, whole=ch["whole"],
                   halves_=ch["halves"], cost_r=ch["cost_r"], tested=len(PAIRS) * len(variants()), risk_pct=risk_pct,
                   neighbours={"values": [v["val"] for v in x["variants"] if v["knob"] == ch["knob"]],
                               "share_positive": round(np.mean([(v["unseen"]["mean"] or -1) > 0 for v in x["variants"]]), 2)},
                   detail={"other_bot": {"name": other_name, "unseen": om, "cfg": ob.get("cfg")}, "protection": x["protection"]},
                   change={"method": "tal_gap", "pair": pn, "cfg": cfg, "track": f"tal_gap|{pn}", "knob": "cfg"})
        out.append(c)
    return out


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
    out, busy = [], -1
    for d, mid, near, far, made, armed in sorted(g, key=lambda x: x[5]):
        key = f"tal|{int(d)}|g{pd.Timestamp(made, tz='UTC').strftime('%Y%m%d%H%M')}"
        base = {"id_key": key, "dir": int(d), "setup_t": pd.Timestamp(armed, tz="UTC").isoformat(), "gap": [float(far), float(near)]}
        q0 = int(np.searchsorted(ns, armed))
        q1 = min(n1, int(np.searchsorted(ns, armed + exp_ns)))
        if q0 >= n1:
            continue                                     # armed in the future of the data (should not happen)
        seg = (l1[q0:q1] <= mid) if d > 0 else (h1[q0:q1] >= mid)
        if not seg.any():
            expired = armed + exp_ns <= ns[-1]
            out.append(dict(base, status="cancelled" if expired else "pending", entry=float(mid), stop=None, target=None,
                            reason="expired" if expired else None))
            continue
        f = q0 + int(np.argmax(seg))
        if f < busy or D.late[f]:
            out.append(dict(base, status="cancelled", entry=float(mid), stop=None, target=None,
                            reason="busy" if f < busy else "session"))
            continue
        px = min(mid, o1[f]) if d > 0 else max(mid, o1[f])
        sw_st, sw_tg = (lo_sw, hi_sw) if d > 0 else (hi_sw, lo_sw)
        s2 = _count_back(*sw_st, ns[f], px, d, -d)
        t2 = _count_back(*sw_tg, ns[f], px, d, d)
        if s2 is None or t2 is None:
            out.append(dict(base, status="cancelled", entry=float(px), stop=None, target=None, reason="no swings"))
            continue
        stop = s2 - d * D.buf
        risk = d * (px - stop)
        if risk <= 0 or d * (t2 - px) / risk < cfg["MIN_RR"]:
            out.append(dict(base, status="cancelled", entry=float(px), stop=float(stop), target=float(t2), reason="rr"))
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
        rec = dict(base, entry=float(px), stop=float(stop), target=float(t2), rr=round(d * (t2 - px) / risk, 2),
                   cost_r=0.0, time=ts(f))
        if i >= big:
            rec.update(status="open", r_now=round(d * (c1[-1] - px) / risk - cost_r, 3))
            busy = n1
        else:
            q = f + i
            if i == i_st:
                ex, why = ((min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if i else stop), "stop"
            elif i == i_tg:
                ex, why = (max(t2, o1[q]) if d > 0 else min(t2, o1[q])), "target"
            else:
                ex, why = o1[q], "forced"
            rec.update(status="closed", exit=float(ex), exit_time=ts(q), reason=why, r=round(d * (ex - px) / risk - cost_r, 4))
            busy = q + 1
        out.append(rec)
    return out
