"""The user's gap method on every FTMO forex pair, the metals and the other cash indices - bot 5's world (user,
13.11.2026: "why doesn't bot 5 use all of FTMO's currency pairs?" - the method was tested and run only on the five
indices, although the user built it on gold and EURUSD).

Each instrument on its own, every timeframe pair of tal_gap.PAIRS, FTMO's recorded 1-minute spread + commission, the
same sessions / news masks as the indices (agent_bot5.prep). Phase 1: per pair, the base, the configuration chosen
for the five indices and the one the bot runs on that pair; phase 2: every one-knob variant of tal_gap.variants()
(EXIT and the Fibonacci levels included). Chosen per instrument and pair on the period to 2025-12-31 only (the total
time-weighted R, as tal_gap.summarize), shown on 2026 (unseen). A pair that earns on both periods on an instrument is
a "new" card whose change adds a gap track for that instrument only (agents/builder.py family "tal5sym" ->
"tal_tracks" {"<track>@<symbol>": {"pair", "cfg", "symbols": [symbol]}} in live/bots/bot5.json, read by
live/watch_own.py). An instrument without minute data is listed as missing, not skipped silently.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents import agentkit as K
from agents import tal_gap as TG

FX28 = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF",
        "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD", "NZDCAD",
        "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
METALS = ["XAUUSD", "XAGUSD", "XPTUSD"]
INDICES = ["UK100", "FRA40", "EU50", "AUS200", "HK50", "N25", "SPN35", "US2000"]
# the shared pool (user, 14.11.2026, task tal1411pool: "test tal_gap on the instruments of bots 1-4 that fit it -
# indices, forex, gold, stocks with an opening gap - and report to bots 1-4 through the manager"): bots 1-4's forex,
# gold and indices are already above; their stocks run here in phase 1 only (the base and the indices' choice), and are
# a REPORT for bots 1-4 - never a card for bot 5 (an index scalping bot)
POOL14 = ["AAPL", "AMZN", "ARM", "ASML", "CSCO", "GME", "GOOG", "INTC", "KO", "MSTR", "NVDA", "PFE", "PLTR", "QCOM",
          "T", "TSLA", "WMT", "XOM", "AIRF", "LVMH", "RACE", "SIEG"]
REPORT_ONLY = set(POOL14)
# gold and EURUSD first: the user built the method on them (tal_gap.md)
SYMBOLS = ["XAUUSD", "EURUSD"] + [s for s in FX28 + METALS + INDICES if s not in ("XAUUSD", "EURUSD")] + POOL14
STATUS = "status.json"
MIN_UNSEEN_N = 30


def _cfgs(phase, chosen):
    """{pair_name: [cfg, ...]} for a phase. chosen = {pair_name: [cfg, ...]} (the indices' choice, what runs now)."""
    out = {}
    for p in TG.PAIRS:
        pn = TG.pair_name(p)
        if phase == 1:
            lst = [dict(TG.BASE)] + [TG._norm(dict(c)) for c in chosen.get(pn, []) if c]
        else:
            lst = [cfg for _, _, cfg in TG.variants()]
        seen, uniq = set(), []
        for c in lst:
            if TG.vid(c) not in seen:
                seen.add(TG.vid(c))
                uniq.append(c)
        out[pn] = uniq
    return out


def run(load, out_dir, budget_s, chosen, symbols=None):
    """Runs what is missing (or older than tal_gap.RERUN_DAYS), phase 1 on every instrument before phase 2, one
    instrument's minutes in memory at a time. Returns (index, status)."""
    out_dir = Path(out_dir)
    idx_dir = out_dir / "idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx = {f.stem: json.loads(f.read_text()) for f in idx_dir.glob("*.json")}
    try:
        st = json.loads((out_dir / STATUS).read_text())
    except Exception:  # noqa: BLE001
        st = {}
    missing = dict(st.get("missing") or {})
    now = pd.Timestamp.now(tz="UTC")
    t0, done = time.time(), 0
    pairs = {TG.pair_name(p): p for p in TG.PAIRS}
    symbols = symbols or SYMBOLS
    stopped = False
    for phase in (1, 2):
        plan = _cfgs(phase, chosen)
        for s in symbols:
            if phase == 2 and s in REPORT_ONLY:
                continue                                  # bots 1-4's stocks: phase 1 only (a report, not bot 5's)
            # a missing instrument is tried again after a day (a download may have failed once)
            if s in missing and (now - pd.Timestamp(missing[s]["at"])).total_seconds() < 86400:
                continue
            todo = []
            for pn, lst in plan.items():
                for cfg in lst:
                    key = f"{pn.replace('/', '_')}_{TG.vid(cfg)}_{s}"
                    age = (now - pd.Timestamp(idx[key]["at"])).days if key in idx else 999
                    if age >= TG.RERUN_DAYS:
                        todo.append((pn, cfg, key))
            if not todo:
                continue
            if time.time() - t0 > budget_s:
                stopped = True
                break
            try:
                D = load(s)
            except Exception as e:  # noqa: BLE001
                D, err = None, str(e)[:200]
            else:
                err = "no minute data / spread"
            if D is None:
                missing[s] = {"at": now.isoformat(timespec="seconds"), "why": err}
                print(f"tal_wide {s}: missing ({err})")
                continue
            missing.pop(s, None)
            gc = {}
            for pn, cfg, key in todo:
                if time.time() - t0 > budget_s:
                    stopped = True
                    break
                ts = time.time()
                t = TG.simulate(D, pairs[pn], cfg, gc)
                f = out_dir / f"{key}.csv.gz"
                t.to_csv(f, index=False)
                idx[key] = {"at": now.isoformat(timespec="seconds"), "pair": pn, "phase": phase, "cfg": cfg, "sym": s,
                            "file": f.name, "n": len(t), "secs": round(time.time() - ts, 1)}
                (idx_dir / f"{key}.json").write_text(json.dumps(idx[key], default=str))
                done += 1
            print(f"  tal_wide phase {phase} {s}: {len(todo)} planned ({time.time() - t0:.0f}s)")
            if stopped:
                break
        if stopped:
            break
    c1, c2 = _cfgs(1, chosen), _cfgs(2, chosen)
    planned = sum(len({TG.vid(c) for c in c1[pn] + c2[pn]}) for pn in c1) * len([x for x in symbols if x not in REPORT_ONLY]) \
        + sum(len({TG.vid(c) for c in c1[pn]}) for pn in c1) * len([x for x in symbols if x in REPORT_ONLY])
    st = {"at": now.isoformat(timespec="seconds"), "symbols": symbols, "missing": missing, "runs": len(idx),
          "planned": planned, "this_run": done}
    (out_dir / STATUS).write_text(json.dumps(st, ensure_ascii=False, default=str))
    print(f"tal_wide: {done} run, {len(idx)} / {planned} in all, missing {sorted(missing)}")
    return idx, st


def summarize(idx, out_dir):
    """{symbol: {pair: {"chosen": cfg + stats, "base": stats, "tried": n}}} - chosen on the train period only."""
    out_dir = Path(out_dir)
    by = {}
    for v in idx.values():
        by.setdefault((v["sym"], v["pair"]), []).append(v)
    out = {}
    for (s, pn), lst in sorted(by.items()):
        rows = []
        for v in lst:
            try:
                t = pd.read_csv(out_dir / v["file"])
            except Exception:  # noqa: BLE001
                continue
            t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
            rows.append({"cfg": v["cfg"], **TG._stats(t)})
        ok = [r for r in rows if (r["train"]["eff_n"] or 0) >= K.MIN_EFF_N and r["train"]["mean"] is not None]
        if not rows:
            continue
        ch = max(ok, key=lambda r: r["train"]["mean"] * r["train"]["eff_n"]) if ok else None
        base = next((r for r in rows if TG.vid(TG._norm(dict(r["cfg"]))) == TG.vid(dict(TG.BASE))), None)
        out.setdefault(s, {})[pn] = {"tried": len(rows), "chosen": ch,
                                     "base": {k: base[k] for k in ("train", "unseen")} if base else None,
                                     "share_positive": round(float(np.mean([(r["unseen"]["mean"] or -1) > 0 for r in rows])), 2)}
    return out


def table(summ):
    """One line per instrument for the app / the cards: the best pair chosen on train and what it did on 2026."""
    rows = []
    for s, pairs in summ.items():
        best = None
        for pn, x in pairs.items():
            ch = x.get("chosen")
            if not ch:
                continue
            sc = (ch["train"]["mean"] or 0) * (ch["train"]["eff_n"] or 0)
            if best is None or sc > best[0]:
                best = (sc, pn, ch)
        if best:
            _, pn, ch = best
            rows.append({"sym": s, "pair": pn, "train_mean": ch["train"]["mean"], "train_n": ch["train"]["n"],
                         "unseen_mean": ch["unseen"]["mean"], "unseen_n": ch["unseen"]["n"], "cost_r": ch["cost_r"],
                         **({"report": "bots1-4"} if s in REPORT_ONLY else {})})
    return sorted(rows, key=lambda r: -((r["unseen_mean"] or -9) * (r["unseen_n"] or 0)))


def cards(bot, summ, risk_pct, tested, live=None):
    """A "new" card per instrument and pair whose train-chosen configuration earns on both periods (train t >= 1.5,
    2026 t >= 1.0 on >= MIN_UNSEEN_N trades); live = {(symbol, pair)} already running (no card)."""
    out = []
    live = live or set()
    for s, pairs in summ.items():
        if s in REPORT_ONLY:
            continue                                      # bots 1-4's stocks: a report through the manager (pool14())
        for pn, x in pairs.items():
            ch = x.get("chosen")
            if not ch or (s, pn) in live:
                continue
            tr, u = ch["train"], ch["unseen"]
            if (tr["mean"] or 0) <= 0 or (tr.get("t") or 0) < 1.5 or (u["mean"] or 0) <= 0 or (u.get("t") or 0) < 1.0 \
                    or (u["n"] or 0) < MIN_UNSEEN_N:
                continue
            cfg = ch["cfg"]
            hi, lo = pn.split("/")
            name = f"{TG.TF_HE.get(hi, hi)} / {TG.TF_HE.get(lo, lo)}"
            what = (f"שיטת הגאפ של טל על {s}, {name}: " + ", ".join(f"{k}={v}" for k, v in cfg.items()) + ". "
                    f"נבחר על 2024-2025 בלבד ({tr['mean']:+.2f}R לעסקה, {tr['n']} עסקאות), מוצג על 2026: {u['mean']:+.2f}R לעסקה, "
                    f"{u['n']} עסקאות, {u['win']}% הצלחה, עלות {ch['cost_r']}R, סה\"כ {u['mean'] * u['n']:+.1f}R. "
                    f"נבדקו {x['tried']} גרסאות בזוג הזה ({x['share_positive']:.0%} מהן חיוביות ב־2026). "
                    f"מסלול חדש רק למכשיר הזה, בסיכון של מסלולי הגאפ.")
            out.append(K.card(bot, "new", f"שיטת הגאפ של טל · {s} · {name}", what, tr, u, whole=ch["whole"],
                              halves_=ch["halves"], cost_r=ch["cost_r"], tested=tested, risk_pct=risk_pct,
                              neighbours={"values": [], "share_positive": x["share_positive"]},
                              change={"method": "tal_gap_sym", "sym": s, "pair": pn, "cfg": cfg,
                                      "track": f"tal_gap|{pn}|{s}", "knob": "cfg"}))
    return out


def pool14(summ):
    """The report for bots 1-4 (task tal1411pool): per stock of theirs, the pair chosen on the period to 2025-12-31
    and what it did on 2026 - "holds" when it earns on both (train t >= 1.5, 2026 mean > 0 on >= MIN_UNSEEN_N)."""
    out = []
    for r in table({s: v for s, v in summ.items() if s in REPORT_ONLY}):
        ch = summ[r["sym"]][r["pair"]]["chosen"]
        out.append(dict(r, train_t=ch["train"].get("t"), unseen_t=ch["unseen"].get("t"), cfg=ch["cfg"],
                        holds=bool((ch["train"].get("t") or 0) >= 1.5 and (r["unseen_mean"] or 0) > 0
                                   and (r["unseen_n"] or 0) >= MIN_UNSEEN_N)))
    return out
