"""Why is a bot below its backtest in the demo? (manager's task mgr1811live, 18.11.2026)

Read only. For bots 1-4, every closed demo trade since the bot started is matched to the stored backtest trade of the same
combo as it trades now (agents/portfolio_check.current_trades: same direction, opened within 2 bars). The gap between the
demo and the expectation is split, in R, into parts that add up exactly:

  expected    the backtest's own mean R per trade on 2026 before the window x the backtest's trades in the window
  market      backtest in the window - expected: the same rules lost on those days without any execution at all
  execution   demo R - backtest R on the matched trades (slippage, spread, late fills, exits; "plan" = the combo's exit /
              filter plan was changed after the trade opened, so the difference is by design and not execution)
  missed      minus the backtest R of trades the demo did not take, by reason: the runner's own skip log (risk guard, an
              approved rule, a late signal), or "unexplained". Not counted at all: backtest trades from before the combo
              joined the bot, while the bot was off (no equity point for 90+ minutes), on an instrument the bot's server does
              not have, or of a combo paused by the manager (risk x0)
  extra       the demo R of trades the backtest does not have
  uncovered   demo trades after the combo's stored backtest ends or of a combo with no stored backtest: not judged

selection     the choosing period's mean R (30.04-31.12.2025) against 2026 before the window: was the edge inflated?
clusters      the bot's worst days: the losses that closed within one 3-hour span, and how many were one bet (same side
              of the same risk: stocks / indices / USDJPY long = risk on).

  python -m agents.live_gap_study      writes reports/agents/live_gap.json
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from agents import health as H
from agents import portfolio_check as P

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "live_gap.json"
BOTS = ("bot1", "bot2", "bot3", "bot4")
START = pd.Timestamp("2026-11-03", tz="UTC")
TF = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "4Hour": 240}
COMBO = re.compile(r"[A-Z]+\d+\|\w+\|[\w/.]+")
RISK_ON_FX = {"USDJPY", "AUDUSD", "NZDUSD", "GBPJPY", "EURJPY", "AUDJPY"}


def _mi(ts):
    return int(pd.Timestamp(ts).timestamp() // 60)


def theme(sym, side):
    """+1 = a bet that risk goes up, -1 = that it goes down, 0 = neither (gold, oil, the other currency pairs)."""
    from agents import ftmo_universe as U
    cls = U.CLASS.get(sym)
    s = 1 if side == "buy" else -1
    if cls in ("stock", "index") or sym in RISK_ON_FX:
        return s
    return 0


def decisions():
    try:
        return json.loads((ROOT / "reports/agents/decisions.json").read_text(encoding="utf-8"))["decisions"]
    except Exception:  # noqa: BLE001
        return []


def plan_changes(bot, rows):
    """{combo: last time an approved exit / filter / management change of the combo was decided}."""
    out = {}
    for d in rows:
        if d.get("bot") != bot or not str(d.get("decision")).startswith("approved") or d.get("undone") or d.get("kind") != "improve":
            continue
        c, at = d.get("combo") or (COMBO.search(str(d.get("title"))) or [None])[0], d.get("at")
        if c and at and pd.Timestamp(at) >= START:
            out[c] = max(out.get(c, ""), at)
    return out


def joined(bot):
    """{combo: when it first appeared in the bot's config} from the git history of live/bots/<bot>.json. Needs the full
    history: on a shallow clone every combo looks new and the study refuses to run rather than report a false gap (the
    first run of this study counted trades from before a combo joined as "missed": +29R in bot 1 that never existed)."""
    if subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "true":
        raise SystemExit("shallow clone: run git fetch --unshallow origin main first")
    log = subprocess.run(["git", "log", "--reverse", "--format=%H %cI", "--", f"live/bots/{bot}.json"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split("\n")
    out = {}
    for line in filter(str.strip, log):
        h, t = line.split()
        try:
            doc = json.loads(subprocess.run(["git", "show", f"{h}:live/bots/{bot}.json"], cwd=ROOT, capture_output=True, text=True).stdout)
        except Exception:  # noqa: BLE001
            continue
        for c in doc.get("combos") or []:
            out.setdefault(c, pd.Timestamp(t).tz_convert("UTC"))
    return out


def up_spans(states, bot):
    """The minutes the bot was running: its equity curves (all its accounts), split where no point came for 90+ minutes."""
    pts = []
    for n, s in states.items():
        if H._is_bot_file(bot, n):
            pts += [_mi(str(e[0])) for e in s.get("equity_curve") or []]
    pts = sorted(set(pts))
    spans, a = [], None
    for i, m in enumerate(pts):
        a = m if a is None else a
        if i == len(pts) - 1 or pts[i + 1] - m > 90:
            spans.append((a, m))
            a = None
    return spans


def skip_log(states, bot):
    """{combo: [(minute, reason)]} from the runner's skip log of all the bot's accounts."""
    out = {}
    for n, s in states.items():
        if H._is_bot_file(bot, n):
            for k in s.get("skips") or []:
                try:
                    out.setdefault(k["combo"], []).append((_mi(k["t"]), str(k.get("why") or "")))
                except Exception:  # noqa: BLE001
                    continue
    return out


def reason(why):
    w = why.lower()
    if "opposite side" in w:
        return "opposite_rule"
    if "into the bar" in w or "late" in w:
        return "late_signal"
    if "risk" in w or "room" in w or "limit" in w or "guard" in w:
        return "risk_guard"
    if "spread" in w:
        return "spread"
    if "news" in w:
        return "news"
    return "other"


def backtest_end(bot, combo, tr):
    """The last minute the stored backtest of this combo covers: the newest exit of any stored variant."""
    ends = [x[1] for x in tr]
    for fam in ("exits", "filters", "manage"):
        d = P.exp(bot, fam, combo) or {}
        for v in (d.get("tr") or {}).values():
            ends += [x[1] for x in (v or [])[-1:]]
    return max(ends) if ends else None


def match(live, bt, bar):
    """Nearest-first matching of the demo trades to the backtest's: same direction, within 2 bars and at least 30 minutes
    (the MetaQuotes server's stock prices come 15 minutes late). -> pairs, missed, extra."""
    pairs, used, extra = [], set(), []
    for lv in sorted(live, key=lambda x: x["mi"]):
        best, gap = None, max(2 * bar, 30) + 1
        for i, b in enumerate(bt):
            if i in used or b[2] != lv["dir"]:
                continue
            g = abs(b[0] - lv["mi"])
            if g < gap:
                best, gap = i, g
        if best is None:
            extra.append(lv)
        else:
            used.add(best)
            pairs.append((lv, bt[best]))
    return pairs, [b for i, b in enumerate(bt) if i not in used], extra


def clusters(rows, top=3):
    """The worst days: the 3-hour span that lost the most, its side split and how much of it was one risk bet."""
    t = pd.DataFrame(rows)
    out = []
    if t.empty:
        return out
    t["day"] = t["close"].dt.strftime("%Y-%m-%d")
    for day, g in sorted(t.groupby("day"), key=lambda kv: kv[1]["r"].sum())[:top]:
        g = g.sort_values("close")
        best, span = 0.0, None
        for _, a in g.iterrows():
            w = g[(g["close"] >= a["close"]) & (g["close"] <= a["close"] + pd.Timedelta(hours=3))]
            los = w[w["r"] < 0]
            if los["r"].sum() < best:
                best, span = float(los["r"].sum()), los
        one = {}
        if span is not None and len(span):
            on, off = span[span["theme"] > 0], span[span["theme"] < 0]
            big = on if on["r"].sum() <= off["r"].sum() else off
            one = {"from": span["close"].min().strftime("%H:%M"), "to": span["close"].max().strftime("%H:%M"),
                   "n": int(len(span)), "r": round(best, 2), "one_bet_n": int(len(big)), "one_bet_r": round(float(big["r"].sum()), 2),
                   "side": "risk_on" if big is on else "risk_off", "symbols": sorted(set(big["symbol"]))}
        out.append({"day": day, "n": int(len(g)), "r": round(float(g["r"].sum()), 2), "worst_span": one})
    return out


def study(bot, states):
    cfg = P.bot_cfg(bot)
    st = H.state_of(bot, states) or {}
    sides = {str(c.get("id")): c for c in st.get("closed") or []}
    tl = H.trades(st, bot)
    tl = tl[tl["time"] >= START]
    rows_d = decisions()
    changed, join, spans = plan_changes(bot, rows_d), joined(bot), up_spans(states, bot)
    skips, off = skip_log(states, bot), P.off_server(bot, cfg)
    mult = P.mults(bot)
    first = spans[0][0] if spans else _mi(START)
    tl = tl[tl["time"] >= pd.Timestamp(first, unit="m", tz="UTC")]
    is_up = lambda m: any(a <= m <= b for a, b in spans)
    why_missed, unexplained = {}, []
    res = {"live_n": int(len(tl)), "live_r": round(float(tl["r"].sum()), 2)}
    sums = dict(bt_n=0, bt_r=0.0, expected=0.0, exec_same=0.0, exec_plan=0.0, matched=0, matched_plan=0, missed_n=0, missed_r=0.0,
                extra_n=0, extra_r=0.0, unc_n=0, unc_r=0.0, flipped=0)
    train, before, per, rows = [], [], [], []
    for _, x in tl.iterrows():
        c = sides.get(str(x["id"])) or {}
        rows.append({"close": x["close"], "r": float(x["r"]), "symbol": c.get("symbol") or x["combo"].split("|")[-1],
                     "side": c.get("side"), "theme": theme(c.get("symbol") or x["combo"].split("|")[-1], c.get("side"))})
    for combo in sorted(set(cfg.get("combos") or []) | set(tl["combo"])):
        lv = tl[tl["combo"] == combo]
        tr = P.current_trades(bot, combo, cfg) if combo in (cfg.get("combos") or []) else None
        if not tr:
            sums["unc_n"] += len(lv)
            sums["unc_r"] += float(lv["r"].sum())
            continue
        tr = [list(x) for x in tr]
        end, s0 = backtest_end(bot, combo, tr), first
        if combo in join:
            s0 = max(s0, _mi(join[combo]))
        win = [x for x in tr if s0 <= x[0] and x[1] <= end]
        if combo in off or mult.get(combo) == 0:
            win = []                                   # never traded here: nothing to compare
        win = [x for x in win if is_up(x[0])]
        pre = [x[3] for x in tr if x[0] < first]
        mean = float(np.mean(pre)) if len(pre) >= 10 else 0.0
        before += pre
        t25 = P.current_trades(bot, combo, cfg, key="tr25")
        train += [x[3] for x in (t25 or [])]
        inside = lv[lv["close"] <= pd.Timestamp(end, unit="m", tz="UTC")]
        out = lv.drop(inside.index)
        sums["unc_n"] += len(out)
        sums["unc_r"] += float(out["r"].sum())
        live = [{"mi": _mi(r["time"]), "r": float(r["r"]), "time": r["time"],
                 "dir": 1 if (sides.get(str(r["id"])) or {}).get("side") == "buy" else -1} for _, r in inside.iterrows()]
        pairs, missed, extra = match(live, win, TF.get(combo.split("|")[1], 60))
        ex_same, ex_plan = 0.0, 0.0
        for a, b in pairs:
            d = a["r"] - b[3]
            if combo in changed and a["time"] < pd.Timestamp(changed[combo]):
                ex_plan += d
                sums["matched_plan"] += 1
            else:
                ex_same += d
            sums["flipped"] += int(b[3] > 0 >= a["r"])
        bar = TF.get(combo.split("|")[1], 60)
        for b in missed:
            hit = [w for m, w in skips.get(combo, []) if abs(m - b[0]) <= max(2 * bar, 30)]
            key = reason(hit[0]) if hit else "unexplained"
            e = why_missed.setdefault(key, {"n": 0, "r": 0.0})
            e["n"] += 1
            e["r"] = round(e["r"] + b[3], 2)
            if not hit:
                unexplained.append({"combo": combo, "time": str(pd.Timestamp(b[0], unit="m", tz="UTC"))[:16], "dir": b[2], "r": round(b[3], 2)})
        bt_r, miss_r, ext_r = sum(x[3] for x in win), sum(x[3] for x in missed), sum(x["r"] for x in extra)
        sums["bt_n"] += len(win)
        sums["bt_r"] += bt_r
        sums["expected"] += mean * len(win)
        sums["exec_same"] += ex_same
        sums["exec_plan"] += ex_plan
        sums["matched"] += len(pairs)
        sums["missed_n"] += len(missed)
        sums["missed_r"] += miss_r
        sums["extra_n"] += len(extra)
        sums["extra_r"] += ext_r
        if live or win:
            per.append({"combo": combo, "live_n": len(live), "live_r": round(sum(x["r"] for x in live), 2), "bt_n": len(win),
                        "bt_r": round(bt_r, 2), "exec_r": round(ex_same, 2), "plan_r": round(ex_plan, 2), "missed_n": len(missed),
                        "missed_r": round(miss_r, 2), "extra_n": len(extra), "extra_r": round(ext_r, 2)})
    covered = sums["bt_r"] + sums["exec_same"] + sums["exec_plan"] - sums["missed_r"] + sums["extra_r"]
    res.update({k: round(v, 2) if isinstance(v, float) else v for k, v in sums.items()})
    res["parts"] = {"expected": round(sums["expected"], 2), "market": round(sums["bt_r"] - sums["expected"], 2),
                    "execution": round(sums["exec_same"], 2), "plan_changes": round(sums["exec_plan"], 2),
                    "missed": round(-sums["missed_r"], 2), "extra": round(sums["extra_r"], 2), "uncovered": round(sums["unc_r"], 2)}
    res["check_sum"] = round(covered + sums["unc_r"] - float(tl["r"].sum()), 6)
    res["selection"] = {"train_n": len(train), "train_mean": round(float(np.mean(train)), 3) if train else None,
                        "before_n": len(before), "before_mean": round(float(np.mean(before)), 3) if before else None,
                        "window_mean": round(sums["bt_r"] / sums["bt_n"], 3) if sums["bt_n"] else None}
    res["never_live"] = sorted({u["combo"] for u in unexplained} - set(tl["combo"]))
    res["missed_why"] = why_missed
    res["unexplained_top"] = sorted(unexplained, key=lambda x: -abs(x["r"]))[:12]
    res["window_from"] = str(pd.Timestamp(first, unit="m", tz="UTC"))[:16]
    res["down_hours"] = round(sum(max(0, spans[i + 1][0] - spans[i][1]) for i in range(len(spans) - 1)) / 60, 1)
    res["worst_exec"] = sorted(per, key=lambda x: x["exec_r"])[:6]
    res["worst_market"] = sorted(per, key=lambda x: x["bt_r"])[:6]
    res["clusters"] = clusters(rows)
    return res


def main():
    states = H.live_states()
    out = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "since": str(START.date()),
           "bots": {b: study(b, states) for b in BOTS}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for b, r in out["bots"].items():
        print(b, r["live_n"], r["live_r"], json.dumps(r["parts"]), "check", r["check_sum"], json.dumps(r["selection"]), json.dumps(r["missed_why"]), r["down_hours"])


if __name__ == "__main__":
    main()
