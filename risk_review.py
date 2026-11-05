"""Strategy review per bot ("ניהול בוטים" in the app): every combo gets its numbers and recommendations.
Nothing changes until the user approves a recommendation in the app, one by one.

For each combo, in each bot, from the demo record since 2026-11-01 (and from its backtest with the live rules):

  profit      trades, success %, mean R, sum R, profit factor
  stability   0..1, the mean of three checks:
                - not one trade: still profitable without its single best trade
                - the largest loss against the average win: 1 at <= 1x, 0 at >= 3x
                - the deepest fall of its cumulative R against everything it won: 1 at none, 0 at all of it
              demo and backtest mixed by the weight of the demo record (n / (n + 20))
  risk        target multiplier = 1 + 0.5 * w * edge, where edge = clip((combo mean R - bot mean R) / 0.4, -1, 1);
              a positive edge counts only as far as the combo is stable (edge x stability), and an unstable
              combo loses up to 0.25 more (- 0.25 x (1 - stability)); limits 0.5..1.5, the bot's combos keep
              an average of 1; advice only when 0.05 or more away from the current multiplier
  direction   "buy only" / "sell only": the other side lost in the backtest in BOTH halves (>= 20 trades,
              mean <= -0.1 R) while the kept side made money, and the demo record does not say otherwise
  target/stop how much of the move the trades took (how often they came within 90% of the target and then
              lost); a nearer target (x0.5 .. x0.9) or a nearer stop (x0.6 .. x0.9, same dollar risk) is advised
              only when it adds >= 0.05 R per trade in BOTH backtest halves (>= 30 trades) and the demo record
              (when it has >= 6 trades) is not worse with it
  hours       a 4-hour block of the day (Israel time) that lost in BOTH backtest halves (>= 20 trades,
              mean <= -0.15 R) and whose removal helps; the demo record must not say otherwise
  market      volatility (the stop's distance against the price, the backtest's thirds): high-volatility or
              low-volatility trades skipped under the same both-halves test
  together    the bot's other combo whose daily results move most with this one (backtest correlation)
  removal     GATE.md rule 4 (trade_review.removal_rule): "remove" = no new trades

Recommendations are refreshed once a week (sample = the week's Monday); the numbers every evening. A declined
recommendation comes back only in a later sample, and only if the record still calls for it.

  python risk_review.py --states <dir with *_state.json> [--manager <manager_state.json>] [--backtest 'glob']
Writes reports/trade_review/risk.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import trade_review as TR

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "trade_review" / "risk.json"
HOLDOUT_FROM = TR.HOLDOUT_FROM
TEST_FROM, SPLIT = "2025-04-30", "2026-01-01"
# every live combo, live rules, with MFE/MAE (run 674); until it is there, the bots' own backtests (MFE/MAE only in part)
BT_DEFAULT = "reports/*_tv_run674/combo_trades_*.csv.gz"
BT_FALLBACK = ("reports/2026-11-04_tv_run6*/combo_trades_*.csv.gz", "reports/2026-11-04_bots_compare_run14/news_trades.csv.gz")
K_TRADES, FULL_EDGE_R, SPAN, UNSTABLE = 20, 0.4, 0.5, 0.25
LO, HI, STEP = 0.5, 1.5, 0.05
TP_F, SL_F = (0.5, 0.6, 0.7, 0.8, 0.9), (0.6, 0.75, 0.9)
BE_AT = 0.6                     # the bots move the stop to breakeven at 60% of the way to the target
MIN_GAIN, BT_MIN, LIVE_MIN = 0.05, 30, 6
TZ = "Asia/Jerusalem"


# ---------------- data ----------------
def demo_trades(states):
    rows = []
    for f in sorted(Path(states).glob("*_state.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        acc = f.stem.replace("_state", "")
        for c in s.get("closed") or []:
            try:
                side = 1 if c["side"] == "buy" else -1
                o, sl0 = float(c["open_px"]), float(c.get("sl0") or 0)
                risk = side * (o - sl0) if sl0 else 0
                ok = risk > 0
                rows.append({"account": acc, "bot": acc.split("_")[0], "id": str(c["id"]), "combo": c["combo"],
                             "time": c["open_t"], "dir": side, "net": float(c.get("net") or 0),
                             "r": side * (float(c["close_px"]) - o) / risk if ok else np.nan,
                             "mfe_r": max(0.0, side * (float(c["mfe"]) - o) / risk) if ok and c.get("mfe") is not None else np.nan,
                             "mae_r": max(0.0, side * (o - float(c["mae"])) / risk) if ok and c.get("mae") is not None else np.nan,
                             "tp0_r": side * (float(c["tp0"]) - o) / risk if ok and c.get("tp0") else np.nan,
                             "stop_pct": abs(o - sl0) / o * 100 if ok else np.nan})
            except Exception:  # noqa: BLE001
                continue
    t = pd.DataFrame(rows, columns=["account", "bot", "id", "combo", "time", "dir", "net", "r", "mfe_r", "mae_r", "tp0_r", "stop_pct"])
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t = t[t["time"] >= pd.Timestamp(HOLDOUT_FROM, tz="UTC")].drop_duplicates(["account", "id"])
    return t


def backtest(pattern):
    files = glob.glob(str(ROOT / pattern))
    if not files and pattern == BT_DEFAULT:
        files = [f for p in BT_FALLBACK for f in glob.glob(str(ROOT / p))]
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f))
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        return pd.DataFrame(columns=["combo", "time", "dir", "r", "mfe_r", "mae_r", "tp0_r", "stop_pct", "net"])
    d = pd.concat(parts, ignore_index=True)
    if "combo" not in d:
        d["combo"] = d["sid"] + "|" + d["tf"] + "|" + d["symbol"]
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d[(d["time"] >= pd.Timestamp(TEST_FROM, tz="UTC")) & (d["time"] < pd.Timestamp(HOLDOUT_FROM, tz="UTC"))]
    risk = (d["entry_px"] - d["stop_px"]).abs()
    d["tp0_r"] = (d["target_px"] - d["entry_px"]) * d["dir"] / risk
    d["stop_pct"] = risk / d["entry_px"] * 100
    d["mfe_r"] = d["mfe_r"].clip(lower=0)
    d["mae_r"] = d["mae_r"].clip(lower=0)
    d["net"] = d["r"]
    d = d.sort_values("mfe_r", na_position="last")           # a copy with MFE/MAE wins over one without
    return d.drop_duplicates(["combo", "time"]).sort_values("time").reset_index(drop=True)


def halves(d):
    return [d[d["time"] < pd.Timestamp(SPLIT, tz="UTC")], d[d["time"] >= pd.Timestamp(SPLIT, tz="UTC")]]


# ---------------- numbers ----------------
def profit(g):
    r = g["r"].dropna()
    n = len(g)
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(n), "win_pct": round(100 * float((g["net"] > 0).mean()), 1) if n else None,
            "mean_r": round(float(r.mean()), 3) if len(r) else None, "sum_r": round(float(r.sum()), 2) if len(r) else 0.0,
            "pf": round(float(wins / losses), 2) if losses > 0 else None}


def stability(g):
    r = g.sort_values("time")["r"].dropna()
    if len(r) < 3:
        return None
    win = r[r > 0]
    avg_win = float(win.mean()) if len(win) else 0.0
    max_loss = float(-r.min()) if r.min() < 0 else 0.0
    without_best = float(r.sum() - r.max())
    cum = r.cumsum()
    dd = float((cum.cummax().clip(lower=0) - cum).max())
    c1 = 1.0 if without_best > 0 else 0.0
    c2 = float(np.clip((3 - max_loss / avg_win) / 2, 0, 1)) if avg_win > 0 else 0.0
    c3 = float(np.clip(1 - dd / win.sum(), 0, 1)) if len(win) else 0.0
    return {"score": round((c1 + c2 + c3) / 3, 2), "without_best_r": round(without_best, 2),
            "max_loss_r": round(max_loss, 2), "avg_win_r": round(avg_win, 2), "drawdown_r": round(dd, 2),
            "sd_r": round(float(r.std()), 2), "checks": [c1, round(c2, 2), round(c3, 2)]}


def tp_value(g, f):
    """Each trade's R with the target moved to f x its target (exact for reaching it from the MFE; the
    breakeven move comes earlier with a nearer target: a loser that reached 60% of the new way exits at 0)."""
    tp = g["tp0_r"]
    hit = g["mfe_r"] >= f * tp
    be = (g["mfe_r"] >= BE_AT * f * tp) & (g["r"] < 0)
    return np.where(hit, f * tp, np.where(be, 0.0, g["r"]))


def sl_value(g, f):
    """Each trade's R in units of the new (nearer) stop, the same dollar risk: stopped if it went f R against
    us, else its own result made 1/f times bigger."""
    return np.where(g["mae_r"] >= f, -1.0, g["r"] / f)


def exits(bt, demo):
    """The target/stop advice. Returns (numbers, recommendation or None)."""
    b = bt.dropna(subset=["mfe_r", "mae_r", "r"])
    d = demo.dropna(subset=["mfe_r", "mae_r", "r"])
    info = {"bt_trades": int(len(b)), "demo_trades": int(len(d))}
    has_tp = b["tp0_r"].notna() & (b["tp0_r"] > 0)
    if has_tp.any():
        x = b[has_tp]
        info["near_miss_pct"] = round(100 * float(((x["mfe_r"] >= 0.9 * x["tp0_r"]) & (x["r"] < 0)).mean()), 1)
        info["reach_pct"] = round(100 * float((x["mfe_r"] >= x["tp0_r"] - 1e-9).mean()), 1)
        info["mfe_of_target_median_pct"] = round(100 * float((x["mfe_r"] / x["tp0_r"]).clip(upper=1).median()), 0)
    if len(b):
        info["mfe_median_r"] = round(float(b["mfe_r"].median()), 2)
        info["mae_winners_p90_r"] = round(float(b.loc[b["r"] > 0, "mae_r"].quantile(0.9)), 2) if (b["r"] > 0).any() else None
    if len(b) < BT_MIN:
        return info, None
    best = None
    variants = [("tp", f) for f in TP_F] if has_tp.mean() > 0.8 else []
    variants += [("sl", f) for f in SL_F]
    for kind, f in variants:
        fn = tp_value if kind == "tp" else sl_value
        gains = []
        for h in halves(b):
            if kind == "tp":
                h = h[h["tp0_r"].notna() & (h["tp0_r"] > 0)]
            if len(h) < 10:
                gains = None
                break
            gains.append(float(np.mean(fn(h, f)) - h["r"].mean()))
        if not gains or min(gains) < MIN_GAIN:
            continue
        dg = None
        dd = d[d["tp0_r"].notna() & (d["tp0_r"] > 0)] if kind == "tp" else d
        if len(dd) >= LIVE_MIN:
            dg = float(np.mean(fn(dd, f)) - dd["r"].mean())
            if dg < 0:
                continue
        score = sum(gains)
        if not best or score > best["score"]:
            best = {"kind": kind, "value": f, "score": score, "bt_gain_r": [round(g, 3) for g in gains],
                    "demo_gain_r": round(dg, 3) if dg is not None else None}
    if not best:
        return info, None
    best.pop("score")
    word = "יעד" if best["kind"] == "tp" else "סטופ"
    best["why"] = (f"{word} ×{best['value']:g}: בבקטסט +{best['bt_gain_r'][0]:.2f}R ו־+{best['bt_gain_r'][1]:.2f}R לעסקה בשני החצאים"
                   + (f", בדמו {best['demo_gain_r']:+.2f}R" if best["demo_gain_r"] is not None else ", בדמו עוד מעט עסקאות"))
    return info, best


def skip_test(bt, demo, mask_bt, mask_demo, label):
    """Shared both-halves test for skipping a slice of trades (a side, an hour block, a volatility third)."""
    out = []
    for h in halves(bt.assign(_skip=np.asarray(mask_bt, dtype=bool))):
        s = h[h["_skip"]]
        if len(s) < 10:
            return None
        out.append((len(s), float(s["r"].mean()), float(h["r"].sum()), float(h["r"].sum() - s["r"].sum())))
    n = sum(x[0] for x in out)
    if n < 20 or any(x[1] > -0.1 for x in out) or any(x[3] <= x[2] for x in out):
        return None
    dm = demo[mask_demo]
    if len(dm) >= 3 and dm["r"].mean() > 0:
        return None                          # the demo record says otherwise
    return {"bt_trades": n, "bt_mean_r": [round(x[1], 2) for x in out],
            "demo": {"trades": int(len(dm)), "mean_r": round(float(dm["r"].mean()), 2) if len(dm) else None},
            "why": f"{label}: בבקטסט {out[0][1]:+.2f}R ו־{out[1][1]:+.2f}R לעסקה בשני החצאים ({n} עסקאות)"
                   + (f", בדמו {dm['r'].mean():+.2f}R ב־{len(dm)}" if len(dm) else "")}


def side_info(g):
    return {s: {"trades": int((g["dir"] == v).sum()), "mean_r": round(float(g.loc[g["dir"] == v, "r"].mean()), 2) if (g["dir"] == v).any() else None}
            for s, v in (("buy", 1), ("sell", -1))}


def block_of(ts):
    return (ts.dt.tz_convert(TZ).dt.hour // 4) * 4


def direction(bt, demo):
    info = {"demo": side_info(demo), "backtest": side_info(bt)}
    for keep, drop, name in ((1, -1, "רק קנייה"), (-1, 1, "רק מכירה")):
        if not len(bt) or bt.loc[bt["dir"] == keep, "r"].sum() <= 0:
            continue
        t = skip_test(bt, demo, bt["dir"] == drop, demo["dir"] == drop, f"{'המכירות' if drop == -1 else 'הקניות'} מפסידות")
        if t:
            return info, dict(t, kind="dir", value=keep, label=name)
    return info, None


def hours(bt, demo):
    if not len(bt):
        return {}, None
    hb, hd = block_of(bt["time"]), block_of(demo["time"]) if len(demo) else pd.Series(dtype=float)
    info = {f"{int(b):02d}-{int(b) + 4:02d}": {"trades": int((hb == b).sum()), "mean_r": round(float(bt.loc[hb == b, "r"].mean()), 2)}
            for b in sorted(hb.unique())}
    best = None
    for b in sorted(hb.unique()):
        t = skip_test(bt, demo, hb == b, (hd == b) if len(demo) else demo["dir"] != demo["dir"], f"שעות {int(b):02d}:00-{int(b) + 4:02d}:00")
        if t and (not best or t["bt_mean_r"][0] + t["bt_mean_r"][1] < best["bt_mean_r"][0] + best["bt_mean_r"][1]):
            best = dict(t, kind="hours", value=[int(b), int(b) + 4], label=f"לא לסחור {int(b):02d}:00-{int(b) + 4:02d}:00")
    return info, best


def market(bt, demo):
    b = bt.dropna(subset=["stop_pct"])
    if len(b) < BT_MIN:
        return {}, None
    lo, hi = (float(x) for x in b["stop_pct"].quantile([1 / 3, 2 / 3]))
    third = lambda s: np.where(s <= lo, "low", np.where(s > hi, "high", "mid"))  # noqa: E731
    tb = third(b["stop_pct"])
    info = {"cut_pct": [round(lo, 3), round(hi, 3)],
            **{k: {"trades": int((tb == k).sum()), "mean_r": round(float(b.loc[tb == k, "r"].mean()), 2)} for k in ("low", "mid", "high")}}
    dt = third(demo["stop_pct"].fillna(-1)) if len(demo) else np.array([])
    for k, cut, name in (("high", {"max_pct": round(hi, 4)}, "לא לסחור בתנודתיות גבוהה"), ("low", {"min_pct": round(lo, 4)}, "לא לסחור בתנודתיות נמוכה")):
        t = skip_test(b, demo, tb == k, (dt == k) if len(demo) else demo["dir"] != demo["dir"],
                      f"תנודתיות {'גבוהה' if k == 'high' else 'נמוכה'} (סטופ {'מעל' if k == 'high' else 'עד'} {hi if k == 'high' else lo:.2f}% מהמחיר)")
        if t:
            return info, dict(t, kind="vol", value=cut, label=name)
    return info, None


def together(bt, combos):
    """For each combo: the bot's other combo whose daily backtest results move most with it."""
    x = bt[bt["combo"].isin(combos)]
    if not len(x):
        return {}
    day = x.assign(d=x["time"].dt.tz_convert(TZ).dt.date).groupby(["d", "combo"])["r"].sum().unstack(fill_value=0.0)
    day = day.loc[:, day.astype(bool).sum() >= 15]
    if day.shape[1] < 2:
        return {}
    cor = day.corr()
    out = {}
    for c in cor.columns:
        s = cor[c].drop(c).dropna()
        if len(s):
            p = s.idxmax()
            both = int(((day[c] < 0) & (day[p] < 0)).sum())
            out[c] = {"with": p, "corr": round(float(s.max()), 2), "lost_together_days": both}
    return out


# ---------------- risk ----------------
def targets(stats, bot_mean, now):
    raw = {}
    for c, s in stats.items():
        if now.get(c, 1.0) == 0:
            continue
        n = s["n_r"]
        w = n / (n + K_TRADES)
        rel = float(np.clip((s["mean_r"] - bot_mean) / FULL_EDGE_R, -1, 1)) if n else 0.0
        q = s["stab"]
        rel = (rel * q if rel > 0 else rel) - UNSTABLE * (1 - q)
        raw[c] = 1 + SPAN * w * float(np.clip(rel, -1, 1))
    if not raw:
        return {}
    v = np.array(list(raw.values()))
    for _ in range(50):
        v = np.clip(v / v.mean(), LO, HI)
    return {c: float(x) for c, x in zip(raw, v)}


def week_of(ts):
    d = ts.tz_convert(TZ).normalize()
    return (d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def decided_value(dec, kind, combo):
    return ((dec.get(kind) or {}) if kind != "mult" else (dec.get("mult") or {})).get(combo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--manager", default="")
    ap.add_argument("--backtest", default=BT_DEFAULT)
    args = ap.parse_args()
    now_ts = pd.Timestamp.now(tz="UTC")
    sample = week_of(now_ts)
    mgr = json.loads(Path(args.manager).read_text(encoding="utf-8")) if args.manager and Path(args.manager).exists() else {}
    decided = mgr.get("risk") or {}
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t = demo_trades(args.states)
    bt = backtest(args.backtest)
    if not len(bt):
        print("no backtest with MFE/MAE found:", args.backtest)
    removal = {}
    try:
        lt = TR.live_trades(args.states)
        if len(lt):
            removal = {r["combo"]: r for r in TR.removal_rule(lt)}
    except Exception as e:  # noqa: BLE001
        print("removal rule skipped:", e)
    out = {"updated": now_ts.isoformat(timespec="seconds"), "sample": sample, "since": HOLDOUT_FROM,
           "backtest": {"from": TEST_FROM, "to": HOLDOUT_FROM, "split": SPLIT, "trades": int(len(bt))},
           "rule": {"k_trades": K_TRADES, "full_edge_r": FULL_EDGE_R, "span": SPAN, "lo": LO, "hi": HI, "step": STEP},
           "bots": {}}
    for b in (1, 2, 3, 4):
        bot = f"bot{b}"
        cfg = json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
        mine = t[t["bot"] == bot]
        dec = decided.get(bot) or {}
        cur = {k: float(v) for k, v in (dec.get("mult") or {}).items()}
        rejected = dec.get("rejected") or {}
        bot_mean = float(mine["r"].mean()) if mine["r"].notna().any() else 0.0
        tog = together(bt, cfg["combos"])
        stats, detail = {}, {}
        for c in cfg["combos"]:
            g, gb = mine[mine["combo"] == c], bt[bt["combo"] == c]
            n_r = int(g["r"].notna().sum())
            sd, sb = stability(g), stability(gb)
            w = n_r / (n_r + K_TRADES)
            q = (w * sd["score"] if sd else 0) + (1 - w if sd else 1) * (sb["score"] if sb else 0.5)
            stats[c] = {"n_r": n_r, "mean_r": float(g["r"].mean()) if n_r else 0.0, "stab": q}
            ex_info, ex_rec = exits(gb, g)
            dir_info, dir_rec = direction(gb, g)
            hr_info, hr_rec = hours(gb, g)
            mk_info, mk_rec = market(gb, g)
            detail[c] = {"profit": {"demo": profit(g), "backtest": profit(gb)},
                         "stability": {"score": round(q, 2), "demo": sd, "backtest": sb},
                         "direction": dir_info, "exits": ex_info, "hours": hr_info, "market": mk_info,
                         "together": tog.get(c), "recs": [x for x in (dir_rec, ex_rec, hr_rec, mk_rec) if x]}
        tg = targets(stats, bot_mean, cur)
        old = {x["combo"]: x for x in ((prev.get("bots") or {}).get(bot) or {}).get("combos") or []}
        # a new backtest (more trades) is new evidence: the week's recommendations are worked out again
        same_week = prev.get("sample") == sample and (prev.get("backtest") or {}).get("trades") == len(bt)
        rows = []
        for c in cfg["combos"]:
            m_now = cur.get(c, 1.0)
            if same_week and c in old and "recs" in old[c]:
                recs = [r for r in old[c]["recs"] if r["kind"] != "remove"]     # frozen for the week
            else:
                recs = []
                if m_now > 0 and c in tg:
                    m = round(round(tg[c] / STEP) * STEP, 2)
                    if abs(m - m_now) >= STEP - 1e-9:
                        s = detail[c]
                        recs.append({"kind": "mult", "value": m, "label": f"סיכון ×{m:g}",
                                     "why": f"רווח ממוצע {stats[c]['mean_r']:+.2f}R מול {bot_mean:+.2f}R של הבוט, יציבות {s['stability']['score']:.2f}, {stats[c]['n_r']} עסקאות"})
                recs += detail[c]["recs"]
            rm = removal.get(c)
            if rm and rm.get("remove") and m_now > 0:
                recs = [{"kind": "remove", "value": 0, "label": "להסיר",
                         "why": f"{rm['cum_r']:+.1f}R ב־{rm['trades']} עסקאות, מתחת לקו {rm['line_r']:.1f}R (כלל ההסרה)"}]
            live_recs = []
            for r in recs:
                k = "mult" if r["kind"] in ("mult", "remove") else r["kind"]
                if decided_value(dec, k, c) == r["value"] or (r["kind"] == "mult" and abs(r["value"] - m_now) < 1e-9):
                    continue                              # already applied
                if rejected.get(f"{r['kind']}:{c}") == sample and r["kind"] != "remove":
                    r = dict(r, rejected=True)
                live_recs.append(r)
            p = detail[c]["profit"]["demo"]
            rows.append({"combo": c, "trades": p["trades"], "win_pct": p["win_pct"], "mean_r": p["mean_r"], "sum_r": p["sum_r"],
                         "mult": m_now, "risk_pct": round(cfg["risk"] * 100 * m_now, 4), "target": round(tg.get(c, m_now), 3),
                         "recs": live_recs, "detail": {k: v for k, v in detail[c].items() if k != "recs"}})
        out["bots"][bot] = {"risk_pct": cfg["risk"] * 100, "trades": int(len(mine)), "mean_r": round(bot_mean, 3), "combos": rows}
        print(bot, len(mine), "demo trades,", sum(len(r["recs"]) for r in rows), "recommendations:",
              {k: sum(1 for r in rows for x in r["recs"] if x["kind"] == k) for k in ("mult", "dir", "tp", "sl", "hours", "vol", "remove")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
