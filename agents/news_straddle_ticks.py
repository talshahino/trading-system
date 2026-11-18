"""EV04 on real ticks: is the straddle's profit there at the prices a stop order really gets? (bot 7 smart agent, 18.11.2026)

EV04 is most of bot 7's backtest profit, and it is measured on 1-minute candles: a stop order "fills" at its level + half
the spread + 0.1 ATR (agents/news_straddle.py; a gap counts only when the minute's OPEN is beyond the level). In the
first seconds of a release the price can jump over the level inside the minute - the candle cannot show that, and the
tick study of 17.11 (agents/news_ticks.py) showed how much of a release's move is gone within 1-2 seconds.

Here every trade of the BUILT version (live/bots/bot7.json: the same releases, levels, window and cap) is replayed on
Dukascopy's ticks (bid + ask of every price change, the hours of the release only):
  trigger  a Buy Stop when the ASK reaches the level, a Sell Stop when the BID reaches it, in the release minute;
           touched in the minute before the release -> both cancelled, no trade (fill_from 0, as the live bot)
  fill     the first tick at least L ms after the trigger (L = 0 / 250 / 500: the order's own delay), at its ask / bid
  stop     the other order's level, touched by the bid (a buy) / ask (a sell), filled at that tick; at +1R the stop goes to
           the entry; out after 30 minutes at the bid / ask of that moment
  R        r = on the risk of the real fill (as the candle study); r_plan = on the risk planned before the fill (the lot
           is sized before the order fills, so a bad fill costs more than 1R on the stop) - r_plan is the honest number
Costs: the ticks' own spread + FTMO's commission. Only the trades the candle version took are replayed.

Results only - nothing in the bot changes here. Output: reports/agents/bot7/straddle_ticks.json (+ per-trade rows in
reports/agents/bot7/straddle_ticks/). Resumes after a time budget cut; runs on the bots' computer in bot 7's round.
  python -m agents.news_straddle_ticks [--minutes 30]
  python -m agents.news_straddle_ticks report
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

LATENCY_MS = (0, 250, 500)
HOLD_MS = 30 * 60_000
OUT = ROOT / "reports" / "agents" / "bot7" / "straddle_ticks.json"
ROWS = ROOT / "reports" / "agents" / "bot7" / "straddle_ticks"
STALE_DAYS = 7
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")


def replay(ts, ask, bid, t0, buy, sell, latency=0, cancel_ms=60_000, hold_ms=HOLD_MS, comm_px=0.0):
    """One straddle on ticks. ts: ms (sorted), t0: the release minute (ms). -> dict with "out": "trade" / "early" (touched
    before the release: both cancelled) / "nofill"; for a trade: dir, entry, r, r_plan, slip (entry beyond the level, in
    price, + = worse), exit_why."""
    a = np.searchsorted(ts, t0 - 60_000)
    b = np.searchsorted(ts, t0 + cancel_ms)
    up = np.nonzero(ask[a:b] >= buy)[0]
    dn = np.nonzero(bid[a:b] <= sell)[0]
    iu = a + int(up[0]) if len(up) else None
    id_ = a + int(dn[0]) if len(dn) else None
    if iu is None and id_ is None:
        return {"out": "nofill"}
    if iu is not None and (id_ is None or iu <= id_):
        i, dr, lvl, stop = iu, 1, buy, sell
    else:
        i, dr, lvl, stop = id_, -1, sell, buy
    if ts[i] < t0:
        return {"out": "early"}
    j = int(np.searchsorted(ts, ts[i] + latency))
    if j >= len(ts):
        return {"out": "nofill"}
    e = float(ask[j] if dr > 0 else bid[j])
    plan = abs(lvl - stop)
    risk = dr * (e - stop)
    side = bid if dr > 0 else ask
    end = max(int(np.searchsorted(ts, ts[j] + hold_ms, side="right")) - 1, j)      # the last tick inside the hold
    seg = side[j:end + 1]
    if risk <= 0:                                  # filled beyond the other order's level: out at once at the stop side
        px, why = float(seg[0]), "stop"
    else:
        hit = np.nonzero((seg <= stop) if dr > 0 else (seg >= stop))[0]
        first_stop = int(hit[0]) if len(hit) else None
        gain = np.nonzero((seg >= e + risk) if dr > 0 else (seg <= e - risk))[0]
        first_be = int(gain[0]) if len(gain) else None
        if first_be is not None and (first_stop is None or first_be < first_stop):
            rest = seg[first_be:]
            hit2 = np.nonzero((rest <= e) if dr > 0 else (rest >= e))[0]
            if len(hit2):
                px, why = float(rest[int(hit2[0])]), "be"
            else:
                px, why = float(seg[-1]), "time"
        elif first_stop is not None:
            px, why = float(seg[first_stop]), "stop"
        else:
            px, why = float(seg[-1]), "time"
    move = dr * (px - e) - comm_px
    return {"out": "trade", "dir": dr, "entry": e, "lvl": lvl, "slip": dr * (e - lvl), "spread": float(ask[j] - bid[j]),
            "r": move / risk if risk > 0 else move / plan, "r_plan": move / plan, "why": why, "ms": int(ts[i] - t0)}


def _built():
    """The built EV04 trades on candles, with the levels of each -> list of dicts, newest release first, instruments mixed
    (so a cut run is still a fair sample)."""
    from agents import news_straddle as NS
    from agents import bot7_trades as BT
    from agents import news_fix as NF
    from agents import poc_agent as P
    df = NS._cached_rows()
    b = BT.built("EV04", "1Min")
    if not len(df) or not b:
        return []
    df = NS.window_out(NS.skipped_out(df, b), b)
    keys = {}
    for s_ in b:
        keys.setdefault(NS.built_key(b[s_]), []).append(s_)
    z = pd.concat([NS._rows_for(df[df["sym"].isin(ss)], k) for k, ss in keys.items()], ignore_index=True)
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    out = []
    for sym, g in z.groupby("sym"):
        d1 = P.bars(sym, "1Min", NF.LOAD_FROM, now)
        d1 = d1[~d1.index.duplicated()].sort_index()
        a5 = NS._atr5(d1)
        idx, o = d1.index, d1["o"].to_numpy(float)
        p = b[sym]
        for x in g.itertuples():
            t = pd.Timestamp(x.rel).tz_convert("UTC")
            k0 = idx.searchsorted(t)
            if k0 >= len(idx) or idx[k0] != t or k0 < 1:
                continue
            ka = a5.index.searchsorted(idx[k0 - 1].floor("5min")) - 1
            if ka < 14:
                continue
            atr = float(a5.iloc[ka])
            if not np.isfinite(atr) or atr <= 0:
                continue
            X = float(p.get("x_atr5", 1.0))
            out.append({"sym": sym, "rel": t.isoformat(), "ref": float(o[k0 - 1]), "atr": atr, "x": X,
                        "cancel": int(p.get("cancel_min", 1)), "dir": int(x.dir), "r_bar": float(x.r), "gross_bar": float(x.gross)})
    out.sort(key=lambda r: r["rel"], reverse=True)
    return out


def _comm_px(sym, e):
    from agents import ftmo_universe as U
    sp = U.spec(sym)
    comm = float(sp["commission"])
    if sp["commissionType"] == "flat_USD":
        return e * comm / (float(sp["contractSize"]) * U.BASE_USD.get(sp["code"].split("/")[0], 1.0))
    return e * 2 * comm / 100.0


def _inst(sym):
    from agents import ftmo_universe as U
    return str(U.DUKA.get(sym) or sym).upper()


def _load(sym):
    f = ROWS / f"{sym}.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def run(minutes):
    from agents import news_ticks as NT
    ROWS.mkdir(parents=True, exist_ok=True)
    todo = _built()
    if not todo:
        print("EV04 ticks: no built rows (the straddle study's cache is missing)")
        return False
    have = {s_: _load(s_) for s_ in {x["sym"] for x in todo}}
    seen = {(s_, r["rel"]) for s_, rows in have.items() for r in rows}
    left = [x for x in todo if (x["sym"], x["rel"]) not in seen]
    print(f"EV04 ticks: {len(todo)} trades, {len(left)} left", flush=True)
    t_end = time.time() + minutes * 60
    scale = {}
    for n, x in enumerate(left):
        if time.time() > t_end:
            print(f"EV04 ticks: time budget, {len(left) - n} left", flush=True)
            break
        sym, t = x["sym"], pd.Timestamp(x["rel"])
        h0 = t.floor("h")
        hours = [h0 - pd.Timedelta(hours=1)] if t.minute == 0 else []     # the minute before the release may be the hour before
        hours += [h0, h0 + pd.Timedelta(hours=1)]
        tk = np.concatenate([NT.hour_ticks(_inst(sym), h) for h in hours])
        row = {"rel": x["rel"], "dir_bar": x["dir"], "r_bar": x["r_bar"], "gross_bar": x["gross_bar"], "atr": x["atr"]}
        if len(tk) < 50:
            row["out"] = "noticks"
        else:
            if sym not in scale:
                scale[sym] = 10.0 ** round(np.log10(tk[0, 2] / x["ref"]))
            ts, ask, bid = tk[:, 0], tk[:, 1] / scale[sym], tk[:, 2] / scale[sym]
            buy, sell = x["ref"] + x["x"] * x["atr"], x["ref"] - x["x"] * x["atr"]
            t0 = t.value / 1e6
            comm = _comm_px(sym, x["ref"])
            for L in LATENCY_MS:
                res = replay(ts, ask, bid, t0, buy, sell, latency=L, cancel_ms=x["cancel"] * 60_000, comm_px=comm)
                row[f"L{L}"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}
        have[sym].append(row)
        if n % 25 == 0:
            for s_, rows in have.items():
                (ROWS / f"{s_}.json").write_text(json.dumps(rows), encoding="utf-8")
    for s_, rows in have.items():
        (ROWS / f"{s_}.json").write_text(json.dumps(rows), encoding="utf-8")
    report(total=len(todo))
    return True


def _sum(rows, L, key):
    v = [r[f"L{L}"][key] for r in rows if r.get(f"L{L}", {}).get("out") == "trade"]
    return round(float(np.sum(v)), 2) if v else 0.0


def report(total=None):
    rows = []
    for f in sorted(ROWS.glob("*.json")) if ROWS.exists() else []:
        for r in json.loads(f.read_text(encoding="utf-8")):
            rows.append(dict(r, sym=f.stem))
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "rows": len(rows), "total": total,
           "partial": total is None or len(rows) < total, "periods": {}, "by_sym": {}}
    for name, g in (("2025", [r for r in rows if pd.Timestamp(r["rel"]) < SPLIT]),
                    ("2026", [r for r in rows if pd.Timestamp(r["rel"]) >= SPLIT])):
        p = {"n_bar": len(g), "r_bar": round(float(sum(r["r_bar"] for r in g)), 2),
             "noticks": sum(1 for r in g if r.get("out") == "noticks")}
        for L in LATENCY_MS:
            tr = [r for r in g if r.get(f"L{L}", {}).get("out") == "trade"]
            slip = [r[f"L{L}"]["slip"] / r["atr"] for r in tr]
            p[f"L{L}"] = {"n": len(tr), "r": _sum(g, L, "r"), "r_plan": _sum(g, L, "r_plan"),
                          "r_bar_same": round(float(sum(r["r_bar"] for r in tr)), 2),
                          "slip_atr_median": round(float(np.median(slip)), 3) if slip else None,
                          "slip_atr_p90": round(float(np.percentile(slip, 90)), 3) if slip else None,
                          "early": sum(1 for r in g if r.get(f"L{L}", {}).get("out") == "early"),
                          "nofill": sum(1 for r in g if r.get(f"L{L}", {}).get("out") == "nofill"),
                          "other_side": sum(1 for r in tr if r[f"L{L}"]["dir"] != r["dir_bar"])}
        rep["periods"][name] = p
    for s_ in sorted({r["sym"] for r in rows}):
        g = [r for r in rows if r["sym"] == s_ and pd.Timestamp(r["rel"]) >= SPLIT]
        rep["by_sym"][s_] = {"n_2026": len(g), "r_bar": round(float(sum(r["r_bar"] for r in g)), 2),
                             "r_plan_L250": _sum(g, 250, "r_plan")}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep["periods"], ensure_ascii=False, indent=1))
    return rep


def stale():
    try:
        j = json.loads(OUT.read_text(encoding="utf-8"))
        return j.get("partial", True) or pd.Timestamp.now(tz="UTC") - pd.Timestamp(j["at"]) > pd.Timedelta(days=STALE_DAYS)
    except Exception:  # noqa: BLE001
        return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=("run", "report"))
    ap.add_argument("--minutes", type=float, default=30)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        report()
    else:
        run(a.minutes)


if __name__ == "__main__":
    main()
