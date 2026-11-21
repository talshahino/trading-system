"""Bot 5's gap method - the exact stop and target of every instrument, then which instruments to run together within
the risk allowed (user, 20.11.2026 09:08: "between the successes, tune the stop and target points of each asset to an
exact level; after you recommend which assets to put in, run a combination test against the risk we are allowed").

Stage 1 (agents/tal_persym_study): 7 stops x 6 targets per instrument.
Stage 2 (here, run): the instruments that run now and those whose stage-1 choice held on 2026 (tal_persym
"good_symbols"): the stop's distance from x0.6 to x1.6 in steps of 0.05 (and the impulse start) x the target swing
around the stage-1 choice (its swing, one less, one more), every gap track of the instrument.
  The choice (on 2025 only): per instrument one stop x target for all its tracks - the one whose 2025 result together
  with its two stop neighbours is best (a plateau, not a lone peak); shown on 2026.
Stage 3 (here, report): the account as bot 5 runs it - 0.28% risk per trade from the balance, a new trade risks at most
the room left under 4.5% a day (from the day's start, open risk counted) and 9% in total, under a quarter of its risk
left -> not taken - on (a) today's tracks as they run, (b) the same with each instrument's tuned stop x target, (c) (b)
plus the new instruments that held on both periods. Return, max drawdown, worst day, days stopped, per year.
Results only - nothing in the bot changes.
  python -m agents.tal_fine_study plan | run --sym US100-a | report
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from agents import tal_dirtrend_study as S
from agents import tal_gap as TG

M = S.M
M.OUT = M.ROOT / "reports" / "agents" / "bot5" / "tal_fine"
STAGE1 = M.ROOT / "reports" / "agents" / "bot5" / "tal_persym" / "report.json"
KS = [round(0.6 + 0.05 * i, 2) for i in range(21)]          # 0.60 .. 1.60
RISK, DAY, TOTAL = 0.0028, 0.045, 0.09


def stage1():
    return json.loads(STAGE1.read_text(encoding="utf-8"))


def symbols():
    r = stage1()
    return sorted({s for s, v in r["per_symbol"].items() if v["live"]} | set(r.get("good_symbols") or []))


def swings_of(sym):
    b = stage1()["per_symbol"][sym]["best"]               # "k0.9|sw4"
    n = int(b.split("|")[1][2:])
    return sorted({max(1, n - 1), n, min(6, n + 1), 2})   # 2 = today's, always as the reference


def ways_for(sym, part):
    """part a = the stops x0.60-1.05, b = x1.10-1.60 and the impulse start."""
    ks = [k for k in KS if (k <= 1.05) == (part == "a")]
    out = {}
    for n in swings_of(sym):
        for k in ks:
            out[f"k{k:.2f}|sw{n}"] = dict({} if k == 1.0 else {"SSCALE": k}, **({"TSWN": n} if n != 2 else {}))
        if part == "b":
            out[f"imp0|sw{n}"] = dict({"STOPAT": "imp0"}, **({"TSWN": n} if n != 2 else {}))
    return out


def plan():
    return [f"{s}-{p}" for s in symbols() for p in ("a", "b")]


def run(job):
    import agent_bot5 as A
    sym, part = job.rsplit("-", 1)
    M.OUT.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out, t_end = {}, time.time() + 160 * 60
    for tk, tv in M.W.tracks().items():
        if "@" in tk and tk.split("@")[1] != sym:
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        base = TG._norm(dict(tv["cfg"]))
        gc, kk = {}, f"{tk}#{sym}"
        for w, extra in ways_for(sym, part).items():
            if time.time() > t_end:
                out.setdefault(kk, {})[w] = {"error": "time budget"}
                continue
            try:
                t = TG.simulate(D, pair, TG._norm(dict(base, **extra)), gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(kk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            out.setdefault(kk, {})[w] = [{"time": str(a), "r": float(b), "dir": int(d), "mins": int(m)}
                                         for a, b, d, m in zip(t["time"], t["r"], t["dir"], t["mins"])]
        print(kk, {w: len(x) if isinstance(x, list) else x for w, x in out[kk].items()}, flush=True)
        (M.OUT / f"{job}.json").write_text(json.dumps(out), encoding="utf-8")
    (M.OUT / f"{job}.json").write_text(json.dumps(out), encoding="utf-8")


def _load():
    rows = {}
    for f in sorted(M.OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        for kk, wz in json.loads(f.read_text(encoding="utf-8")).items():
            for w, xs in wz.items():
                if isinstance(xs, list):
                    rows.setdefault(kk, {})[w] = xs
    return rows


def _sum(xs, unseen):
    return sum(x["r"] for x in xs if pd.Timestamp(x["time"]) >= M.FROM and (pd.Timestamp(x["time"]) >= TG.SPLIT) == unseen)


def dkey(t):
    """FTMO's day: midnight Prague."""
    return t.tz_convert("Europe/Prague").normalize() if t.tzinfo else t.normalize()


def account(trades):
    """The bot's virtual account on these trades: [(t_open, t_close, r)] -> per year return %, max DD %, worst day %."""
    ev = sorted(trades, key=lambda x: x[0])
    bal, init, peak, open_, out = 100_000.0, 100_000.0, 100_000.0, [], {}
    day, day_start, day_pnl, dd, worst, stopped, taken, skipped = None, bal, 0.0, 0.0, {}, set(), 0, 0
    closes = []                                   # (t_close, pnl, risk)

    def settle(upto):
        nonlocal bal, peak, dd, day_pnl
        closes.sort(key=lambda c: c[0])
        while closes and closes[0][0] <= upto:
            tc, pnl, rk = closes.pop(0)
            open_.remove(rk)
            bal += pnl
            if day is not None and dkey(tc) == day:
                day_pnl += pnl
            peak = max(peak, bal)
            dd = max(dd, (peak - bal) / peak)
    for t0, t1, r in ev:
        settle(t0)
        d0 = dkey(t0)
        if day is None or d0 != day:
            if day is not None:
                worst[day] = day_pnl / day_start
            day, day_start, day_pnl = d0, bal, 0.0
        want = bal * RISK
        room = min(DAY * init + day_pnl - sum(open_), bal - sum(open_) - init * (1 - TOTAL))
        if room < 0.25 * want:
            skipped += 1
            if day_pnl - sum(open_) <= -DAY * init * 0.9:
                stopped.add(day)
            continue
        rk = min(want, room)
        taken += 1
        open_.append(rk)
        closes.append((t1, r * rk, rk))
    settle(pd.Timestamp.max.tz_localize("UTC"))
    if day is not None:
        worst[day] = day_pnl / day_start
    return {"ret": round(100 * (bal / init - 1), 2), "dd": round(100 * dd, 2),
            "worst_day": round(100 * min(worst.values(), default=0), 2), "days_stopped": len(stopped),
            "taken": taken, "skipped": skipped}


def report():
    rows = _load()
    s1 = stage1()
    syms = sorted({k.split("#")[1] for k in rows})
    choice = {}
    for s in syms:
        ks = [k for k in rows if k.endswith("#" + s)]
        W = sorted({w for k in ks for w in rows[k]})
        tr = {w: sum(_sum(rows[k].get(w, []), False) for k in ks) for w in W}
        un = {w: sum(_sum(rows[k].get(w, []), True) for k in ks) for w in W}

        def plateau(w):
            st, sw = w.split("|")
            if not st.startswith("k"):
                return tr[w]
            k = float(st[1:])
            nb = [f"k{k + d:.2f}|{sw}" for d in (-0.05, 0.05)]
            return np.mean([tr[w]] + [tr[x] for x in nb if x in tr])
        b = max(W, key=plateau)
        now = "k1.00|sw2"
        choice[s] = {"best": b, "2025": round(tr[b], 1), "2026": round(un[b], 1),
                     "now_2025": round(tr.get(now, 0), 1), "now_2026": round(un.get(now, 0), 1),
                     "live": s1["per_symbol"].get(s, {}).get("live", False),
                     "holds": tr[b] > 0 and un[b] > 0}
    # the trades of each portfolio, per year
    def trades(sel):
        out = {"2025": [], "2026": []}
        for k, ws in rows.items():
            tk, s = k.split("#")
            if s not in sel:
                continue
            if choice[s]["live"] and not S.live_pair(tk, s):
                continue                          # a running instrument keeps its tracks; a new one takes them all
            w = "k1.00|sw2" if sel[s] == "now" else sel[s]
            for x in ws.get(w, []):
                t0 = pd.Timestamp(x["time"])
                if t0 < M.FROM:
                    continue
                out["2026" if t0 >= TG.SPLIT else "2025"].append((t0, t0 + pd.Timedelta(minutes=x["mins"] + 1), x["r"]))
        return out
    live_syms = [s for s, v in choice.items() if v["live"]]
    pf = {"a_now": {s: "now" for s in live_syms},
          "b_tuned": {s: choice[s]["best"] for s in live_syms},
          "c_tuned_plus_new": {**{s: choice[s]["best"] for s in live_syms},
                               **{s: choice[s]["best"] for s, v in choice.items() if not v["live"] and v["holds"]}},
          "d_tuned_holding_only": {s: choice[s]["best"] for s, v in choice.items() if v["holds"]}}
    # b on the live tracks only (the tuned stop x target, the same tracks x instruments as today)
    res = {}
    for name, sel in pf.items():
        tz = trades(sel)
        res[name] = {y: account(v) for y, v in tz.items()}
        res[name]["symbols"] = sorted(sel)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "choice": choice, "portfolios": res}
    (M.OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for s, v in sorted(choice.items(), key=lambda kv: -kv[1]["2026"]):
        print(f"{s:7} {'*' if v['live'] else ' '} {v['best']:12} {v['2025']:7} {v['2026']:7}  now {v['now_2025']:7} {v['now_2026']:7} {'holds' if v['holds'] else ''}")
    for n, v in res.items():
        print(n, v["2025"], v["2026"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(plan()))
    elif a.mode == "run":
        run(a.sym)
    else:
        report()
