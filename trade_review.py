"""Review of every live trade, in every bot and every account: how could it have been better?

For each closed position (the bots' state files on branch `live`, `closed` list) in R units, R = the
distance from the entry to the first stop:
    exit_r   what it made
    mfe_r    the furthest it went in our favour while open (the bot tracks it live)
    mae_r    the furthest it went against us while open
What-ifs, exact from mfe/mae for targets (a target at k R is reached if and only if mfe_r >= k while the
trade was open), approximate for a nearer stop (the order of the high and the low is not known):
    tp_k     k if mfe_r >= k, else what it really made          (k = 1, 1.5, 2, 3)
    stop_75  -0.75 if mae_r >= 0.75, else what it really made
Accumulated per strategy (sid) and per combo across all accounts, with the same numbers for the
strategy's backtest trades (base variant of the newest optimize runs) next to them, so a live hint can be
checked against hundreds of past trades. A hint becomes a "candidate" only when live AND backtest agree;
improve.py then tests candidates with the whole bot.

  python trade_review.py --states <dir with *_state.json> [--backtest 'reports/*/combo_trades_*.csv.gz']
Writes reports/trade_review/live_trades.csv (all trades seen so far) and reports/trade_review/summary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "trade_review"
KS = (1.0, 1.5, 2.0, 3.0)
MIN_LIVE = 6            # live trades of a strategy before it gets a hint (user: 6)


def live_trades(states):
    rows = []
    for f in sorted(Path(states).glob("*_state.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        acc = str(s.get("bot") or f.stem.replace("_state", ""))
        for c in s.get("closed") or []:
            try:
                side = 1 if c["side"] == "buy" else -1
                o, sl0 = float(c["open_px"]), float(c["sl0"] or 0)
                if not sl0 or c.get("mfe") is None or c.get("mae") is None:
                    continue
                risk = side * (o - sl0)
                if risk <= 0:
                    continue
                rows.append({"account": acc, "id": str(c["id"]), "combo": c["combo"], "sid": c["combo"].split("|")[0],
                             "symbol": c["symbol"], "side": side, "open_t": c["open_t"], "close_t": c["close_t"],
                             "exit_r": side * (float(c["close_px"]) - o) / risk,
                             "mfe_r": max(0.0, side * (float(c["mfe"]) - o) / risk),
                             "mae_r": max(0.0, side * (o - float(c["mae"])) / risk),
                             "tp0_r": side * (float(c["tp0"]) - o) / risk if c.get("tp0") else np.nan,
                             "reason": c.get("reason") or "", "net": c.get("net")})
            except Exception:  # noqa: BLE001
                continue
    return pd.DataFrame(rows)


def what_ifs(t):
    t = t.copy()
    for k in KS:
        # an existing target below k ended the trade there: what came after is unknown, keep the real result
        capped = t["tp0_r"].notna() & (t["tp0_r"] < k) & (t["reason"].isin(["tp"]) | (t["exit_r"] >= t["tp0_r"] - 0.05))
        t[f"tp_{k:g}"] = np.where((t["mfe_r"] >= k) & ~capped, k, t["exit_r"])
    t["stop_75"] = np.where(t["mae_r"] >= 0.75, -0.75, t["exit_r"])
    return t


def summarise(t):
    if t.empty:
        return {"n": 0}
    out = {"n": int(len(t)), "avg_r": round(float(t["exit_r"].mean()), 3),
           "win_pct": round(100 * float((t["exit_r"] > 0).mean()), 1),
           "mfe_median": round(float(t["mfe_r"].median()), 2), "mae_median": round(float(t["mae_r"].median()), 2),
           "losers_reached_1r_pct": round(100 * float(((t["exit_r"] < 0) & (t["mfe_r"] >= 1)).sum() / max(1, (t["exit_r"] < 0).sum())), 1),
           "winners_mae_p90": round(float(t.loc[t["exit_r"] > 0, "mae_r"].quantile(0.9)), 2) if (t["exit_r"] > 0).any() else None}
    for k in KS:
        out[f"if_tp_{k:g}R"] = round(float(t[f"tp_{k:g}"].mean()), 3)
    out["if_stop_0.75"] = round(float(t["stop_75"].mean()), 3)
    return out


def hints(live, bt):
    """Candidates where live and backtest agree: a target or a nearer stop that beats what happened by
    0.05 R per trade or more in both."""
    if live.get("n", 0) < MIN_LIVE or not bt or bt.get("n", 0) < 30:
        return []
    out = []
    for key in [f"if_tp_{k:g}R" for k in KS] + ["if_stop_0.75"]:
        dl, db = live[key] - live["avg_r"], bt[key] - bt["avg_r"]
        if dl >= 0.05 and db >= 0.05:
            out.append({"change": key.replace("if_", ""), "live_gain_r": round(dl, 3), "backtest_gain_r": round(db, 3)})
    return sorted(out, key=lambda x: -(x["live_gain_r"] + x["backtest_gain_r"]))


def backtest_base(pattern):
    parts = []
    for f in glob.glob(pattern):
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if "mfe_r" not in d.columns or "variant" not in d.columns:
            continue
        d = d[d["variant"].str.contains(r"\|stop1\|entry0$", regex=True) & d["variant"].str.startswith("own|")]
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d["combo"] = d["sid"] + "|" + d["tf"] + "|" + d["symbol"]
    return pd.DataFrame({"combo": d["combo"], "sid": d["sid"], "exit_r": d["r"], "mfe_r": d["mfe_r"].clip(lower=0),
                         "mae_r": d["mae_r"].clip(lower=0), "tp0_r": (d["target_px"] - d["entry_px"]) * d["dir"]
                         / (d["entry_px"] - d["stop_px"]).abs(),
                         "reason": np.where(d["reason"] == "יעד", "tp", "")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--backtest", default=str(ROOT / "reports" / "*" / "combo_trades_*.csv.gz"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    new = live_trades(args.states)
    hist_f = OUT / "live_trades.csv"
    old = pd.read_csv(hist_f, dtype={"id": str}) if hist_f.exists() else pd.DataFrame()
    t = pd.concat([old, new], ignore_index=True).drop_duplicates(["account", "id"], keep="last") if len(new) or len(old) else new
    if t.empty:
        print("no live trades with stop and MFE/MAE yet")
        return
    t.to_csv(hist_f, index=False)
    t = what_ifs(t)
    bt = backtest_base(args.backtest)
    bt = what_ifs(bt) if len(bt) else bt
    res = {"updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "live_trades": int(len(t)),
           "accounts": sorted(t["account"].unique().tolist()), "strategies": [], "combos": []}
    for key, col in (("strategies", "sid"), ("combos", "combo")):
        for k, g in t.groupby(col):
            lv = summarise(g)
            bv = summarise(bt[bt[col] == k]) if len(bt) else {"n": 0}
            res[key].append({col: k, "live": lv, "backtest": bv, "candidates": hints(lv, bv),
                             # losing live after enough trades: check with loo.py whether the bot is better without it
                             "check_removal": bool(lv.get("n", 0) >= MIN_LIVE and lv.get("avg_r", 0) < 0)})
        res[key].sort(key=lambda x: -x["live"]["n"])
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"live trades: {len(t)} in {len(res['accounts'])} accounts | strategies: {len(res['strategies'])} | "
          f"with candidates: {sum(1 for x in res['strategies'] if x['candidates'])}")


if __name__ == "__main__":
    main()
