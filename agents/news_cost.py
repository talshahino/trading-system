"""Bot 7: what a tight stop really costs (manager's task mgr1811cost, bot 7 smart agent, 19.11.2026).

EV03 USDJPY lost -2.62R on 18.11 (-902$ against about 345$ planned): the stop was 3.45 pips and the costs alone were about
1.6R. The size is set from the stop's distance without the costs, so on a tight stop the real loss is far above the risk.

Every trade of the built version of EV02 / EV03 / EV04 / EV08 (the lists the whole-bot check reads, exp/study_*.json),
each with its own cost in R (c = spread + commission - swap, over the stop's distance):
  net        the size set on the net risk: the loss at the stop is 1R with the costs, every result is r / (1 + c)
  skip_c     no trade when its cost is above a threshold in R
  minstop    no trade when the stop is nearer than a share of the 5-minute ATR (EV02 / EV03, the only ones with a free stop)
  net_skip   both
One rule per scope (all the families together, and each family alone), chosen on 30.04-31.12.2025 by the total R, shown on
2026. "passes" = better than today on both periods. Results only, nothing in the bot changes here.
Output: reports/agents/bot7/cost_study.json
  python -m agents.news_cost --minutes 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "agents" / "bot7" / "cost_study.json"
EXP = ROOT / "reports" / "agents" / "bot7" / "exp"
CACHE = ROOT / "cache" / "news_cost"
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
SIDS = ("EV02", "EV03", "EV04", "EV08")
ENGINE_SIDS = ("EV02", "EV03")
SKIP_C = (0.3, 0.5, 0.75, 1.0, 1.5)
MIN_ATR = (0.25, 0.5, 0.75, 1.0)
MIN_TRADES = 20              # trades a scope must keep on 2025 for its rule to be picked
STALE_DAYS = 3
VERSION = 1


def rules():
    """Rule name -> (size on net risk, largest cost in R, nearest stop in 5-minute ATR)."""
    out = {"today": (False, None, None), "net": (True, None, None)}
    for k in SKIP_C:
        out[f"skip_c{k:g}"] = (False, k, None)
        out[f"net_skip_c{k:g}"] = (True, k, None)
    for k in MIN_ATR:
        out[f"minstop_atr{k:g}"] = (False, None, k)
    return out


def apply_rule(df, rule):
    """The result of every trade under a rule; NaN = the trade is not taken."""
    net, max_c, min_atr = rule
    r = df["r"].to_numpy(float).copy()
    c = np.maximum(df["c"].to_numpy(float), 0.0)
    if net:
        r = r / (1.0 + c)
    if max_c is not None:
        r[c > max_c] = np.nan
    if min_atr is not None:
        a = df["risk_atr"].to_numpy(float)
        r[np.isfinite(a) & (a < min_atr)] = np.nan
    return r


def _stats(r, r0, c):
    """Kept trades: count, total, mean, worst; what the rule dropped; the risk really carried (R at the stop, costs in)."""
    keep = np.isfinite(r)
    x = r[keep]
    return {"n": int(keep.sum()), "sum": round(float(x.sum()), 2) if len(x) else 0.0,
            "mean": round(float(x.mean()), 4) if len(x) else None, "worst": round(float(x.min()), 2) if len(x) else None,
            "skipped": int((~keep).sum()), "skipped_r": round(float(r0[~keep].sum()), 2),
            "over_1_5r": int((x < -1.5).sum()), "cost_mean": round(float(c[keep].mean()), 3) if len(x) else None}


def evaluate(df):
    """Every rule on one scope, on both periods, and the rule chosen on 2025."""
    t = pd.to_datetime(df["t"], utc=True)
    old, new = (t < SPLIT).to_numpy(), (t >= SPLIT).to_numpy()
    r0, c = df["r"].to_numpy(float), np.maximum(df["c"].to_numpy(float), 0.0)
    res = {}
    for name, rule in rules().items():
        r = apply_rule(df, rule)
        res[name] = {"y2025": _stats(r[old], r0[old], c[old]), "y2026": _stats(r[new], r0[new], c[new])}
    base = res["today"]
    pool = [k for k in res if k != "today" and res[k]["y2025"]["n"] >= MIN_TRADES]
    best = max(pool, key=lambda k: res[k]["y2025"]["sum"]) if pool else None
    better = bool(best and res[best]["y2025"]["sum"] > base["y2025"]["sum"])
    chosen = best if better else None
    return {"today": base, "chosen": chosen, "chosen_result": res[chosen] if chosen else None,
            "passes": bool(chosen and res[chosen]["y2026"]["sum"] > base["y2026"]["sum"]), "rules": res}


def analyse(df):
    out = {"all": evaluate(df)}
    for sid in SIDS:
        z = df[df["sid"] == sid]
        if len(z):
            out[sid] = evaluate(z.reset_index(drop=True))
    return out


def built_rows():
    """The trades the whole-bot check knows for the built version: exp/study_<sid>_<tf>_<sym>.json."""
    rows = []
    for sid in SIDS:
        for f in sorted(EXP.glob(f"study_{sid}_*.json")):
            j = json.loads(f.read_text(encoding="utf-8"))
            sym = j["combo"].split("|")[2]
            for k in ("tr25", "tr"):
                for x in (j.get(k) or {}).get("built") or []:
                    rows.append({"sid": sid, "sym": sym, "m": int(x[0]), "dir": int(x[2]), "r": float(x[3])})
    df = pd.DataFrame(rows, columns=["sid", "sym", "m", "dir", "r"])
    df["t"] = EPOCH + pd.to_timedelta(df["m"], unit="min")
    return df


def _minute(ts):
    return ((pd.to_datetime(ts, utc=True, format="ISO8601") - EPOCH) // pd.Timedelta(minutes=1)).astype("int64")


def study_costs(mod, sid):
    """EV04 / EV08: the studies keep the gross and the net result of each row -> {(sid, sym, minute, dir, r): cost in R}."""
    z = mod._cached_rows()
    if z is None or not len(z) or "gross" not in z:
        return {}
    m = _minute(z["t"]).to_numpy()
    c = (z["gross"].to_numpy(float) - z["r"].to_numpy(float))
    return {(sid, s, int(a), int(d), round(float(r), 4)): float(x)
            for s, a, d, r, x in zip(z["sym"], m, z["dir"], z["r"], c)}


def engine_rows(sym, plans):
    """EV02 / EV03 on one instrument as agents/news_regime.py runs them (the plan's filter, the live engine), with the
    cost of each trade in R and the stop's distance in 5-minute ATR."""
    from agents import event_strategies as E
    from agents import ftmo_costs as C
    from agents import ftmo_universe as U
    from agents import news_regime as NR
    from agents import poc_agent as P
    from agents import rules_agent as R
    from live import news_bot as NB
    end = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    d1 = P.bars(sym, "1Min", NR.LOAD_FROM, end)
    if d1 is None or d1.empty:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    try:
        sp = P.spread_1m(sym, NR.LOAD_FROM, end)
    except Exception:  # noqa: BLE001
        sp = None
    d = d1.resample("5min", label="left", closed="left").agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
    d = d.dropna(subset=["o"])
    d.attrs.update(symbol=sym, tf="5Min")
    E.events_for(sym)
    pc = d["c"].shift(1)
    tr = pd.concat([d["h"] - d["l"], (d["h"] - pc).abs(), (d["l"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)
    px = float(d["c"].median())
    spec = U.spec(sym)
    comm = (px * float(spec["commission"]) / (float(spec["contractSize"]) * U.BASE_USD.get(spec["code"].split("/")[0], 1.0))
            if spec["commissionType"] == "flat_USD" else px * 2 * float(spec["commission"]) / 100.0)
    spm = float(sp.median()) if sp is not None and len(sp) else 0.0
    fe, ne = NB.break_exit(d, "5Min")
    rows = []
    for sid in ENGINE_SIDS:
        p_ = plans.get(f"{sid}|5Min|{sym}") or {}
        if p_.get("off_from"):
            continue
        pk = NB.plan_keys(p_)
        L, S, plan = E.EVENTS[sid][1](d)
        f = p_.get("filter") or "none"
        al, ash = (np.ones(len(d), bool), np.ones(len(d), bool)) if f == "none" else NB.regime(d, f)
        al, ash = np.asarray(al, bool) & ~ne, np.asarray(ash, bool) & ~ne
        t = R.run(d, NR._masked((L, S, {**plan, **pk} if pk else plan), al, ash), 0.0, breakeven=(0.6, (spm + comm) / px),
                  session_exit=fe if fe.any() else None)
        if t is None or not len(t):
            continue
        t = t[t["reason"] != "סוף הנתונים"]
        if not len(t):
            continue
        net = C.apply(t.reset_index(drop=True), sym, sp)
        for x in net.itertuples():
            tt = pd.Timestamp(x.time)
            tt = tt.tz_localize("UTC") if tt.tzinfo is None else tt
            risk = abs(float(x.entry_px) - float(x.stop_px))
            if risk <= 0:
                continue
            a = atr.asof(tt.tz_convert(d.index.tz) if d.index.tz is not None else tt.tz_localize(None))
            rows.append({"sid": sid, "sym": sym, "m": int((tt - EPOCH) // pd.Timedelta(minutes=1)), "dir": int(x.dir),
                         "r": round(float(x.r), 4), "c": round(float(x.spread_px + x.comm_px - x.swap_px) / risk, 4),
                         "risk_atr": round(risk / float(a), 3) if a == a and a > 0 else None})
    return rows


def cached_engine(sym, plans, t_end):
    """(rows, done): one instrument's rows from the cache, or from the engine while the time lasts."""
    f = CACHE / f"{sym}.json"
    try:
        j = json.loads(f.read_text(encoding="utf-8"))
        if j.get("version") == VERSION and pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) < pd.Timedelta(days=STALE_DAYS):
            return j["rows"], True
    except Exception:  # noqa: BLE001
        pass
    if time.time() > t_end:
        return [], False
    rows = engine_rows(sym, plans)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"version": VERSION, "at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": rows}), encoding="utf-8")
    return rows, True


def with_costs(df, costs, eng):
    """The built trades with their cost; a trade no study can price is counted and left out, never guessed."""
    e = {(x["sid"], x["sym"], x["m"], x["dir"]): x for x in eng}
    c, a, miss = [], [], {}
    for x in df.itertuples():
        v, atr_ = None, None
        if x.sid in ENGINE_SIDS:
            y = e.get((x.sid, x.sym, x.m, x.dir))
            if y is not None and abs(y["r"] - x.r) <= 0.02:
                v, atr_ = y["c"], y.get("risk_atr")
        else:
            v = costs.get((x.sid, x.sym, x.m, x.dir, round(x.r, 4)))
        if v is None:
            miss[x.sid] = miss.get(x.sid, 0) + 1
        c.append(np.nan if v is None else v)
        a.append(np.nan if atr_ is None else atr_)
    out = df.assign(c=c, risk_atr=a)
    return out[out["c"].notna()].reset_index(drop=True), miss


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return (j.get("partial") or j.get("version") != VERSION
                or pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS))
    except Exception:  # noqa: BLE001
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    a = ap.parse_args(argv)
    from agents import news_straddle as NST
    from agents import news_surprise as NSU
    t_end = time.time() + a.minutes * 60
    df = built_rows()
    plans = json.loads((ROOT / "live" / "bots" / "bot7.json").read_text(encoding="utf-8")).get("plans") or {}
    costs = {**study_costs(NST, "EV04"), **study_costs(NSU, "EV08")}
    eng, left = [], []
    for sym in sorted(df.loc[df["sid"].isin(ENGINE_SIDS), "sym"].unique()):
        try:
            r_, ok = cached_engine(sym, plans, t_end)
        except Exception as e:  # noqa: BLE001 - one instrument without data does not stop the study
            print(f"{sym}: failed ({str(e)[-200:]})")
            r_, ok = [], False
        eng += r_
        if not ok:
            left.append(sym)
    z, miss = with_costs(df, costs, eng)
    if not len(z):
        print("cost study: no trade could be priced yet")
        return False
    res = analyse(z)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "version": VERSION,
                               "split": str(SPLIT.date()), "partial": bool(left), "left": left, "trades": int(len(z)),
                               "not_priced": miss, "scopes": res}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("cost study:", json.dumps({k: {"chosen": v["chosen"], "passes": v["passes"]} for k, v in res.items()}, ensure_ascii=False))
    return True


if __name__ == "__main__":
    main()
