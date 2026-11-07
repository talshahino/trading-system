"""A smart filter over the live bots' signals (research only): can what is known at the moment of a signal
tell a better trade from a worse one?

Trades: the backtest of every live combo with the live rules (bots 1-3: news_trades of bots_compare run 14;
bot 4: tv run 674). Features known at the signal (nothing from the future):
    hour (Israel), weekday, timeframe, direction, family (TV/SS/TT/...), instrument class,
    stop distance as % of price and vs that combo's own past median (is the market wilder than usual?),
    the combo's last 5 and 20 closed results (R), all live combos' last 30 closed results (R),
    how many trades of the same instrument are open right now
Model: gradient boosting, fitted on 2025-04-30 .. 2025-12-31; the skip line (predicted R below x) is chosen
there too. Checked on 2026-01-01 .. end, which the model never saw: total R and R per trade of the trades it
keeps vs all trades, per bot. A filter is worth proposing only if it adds R in the check period.

  python metalabel.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import portfolio_risk as PR

ROOT = Path(__file__).resolve().parent
SPLIT = PR.SPLIT


def features(books):
    rows = []
    for n, (t, cfg) in books.items():
        x = t.copy()
        x["bot"] = n
        rows.append(x)
    d = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    d["stop_pct"] = (d["entry_px"] - d["stop_px"]).abs() / d["entry_px"] * 100
    loc = d["time"].dt.tz_convert("Asia/Jerusalem")
    d["hour"], d["wday"] = loc.dt.hour, loc.dt.weekday
    d["tfm"] = d["tf"].map({"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60, "4Hour": 240}).fillna(60)
    d["fam"] = d["sid"].str.extract(r"^([A-Z]+)")[0].astype("category").cat.codes
    d["cls"] = d["symbol"].map(lambda s: list(PR.factors(s, 1))[0]).astype("category").cat.codes
    # per combo: past closed results and stop distance vs its own past (only trades closed before this signal)
    f5, f20, rel = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), np.nan)
    for _, g in d.groupby(["bot", "combo"]):
        ex, r, sp = g["exit_time"].to_numpy(), g["r"].to_numpy(), g["stop_pct"].to_numpy()
        for j, (i, tm) in enumerate(zip(g.index, g["time"].to_numpy())):
            past = np.where(ex[:j] <= tm)[0]
            if len(past):
                f5[i] = r[past[-5:]].mean()
                f20[i] = r[past[-20:]].mean()
            if j >= 5:
                rel[i] = sp[j] / np.median(sp[max(0, j - 50):j])
    d["last5"], d["last20"], d["stop_rel"] = f5, f20, rel
    # all combos: last 30 closed results; open trades on the same instrument
    ex_sorted = d.sort_values("exit_time")
    ex_t, ex_r = ex_sorted["exit_time"].to_numpy(), ex_sorted["r"].to_numpy()
    k = np.searchsorted(ex_t, d["time"].to_numpy(), side="right")
    cs = np.r_[0, np.cumsum(ex_r)]
    d["all30"] = [(cs[i] - cs[max(0, i - 30)]) / max(1, min(30, i)) if i else np.nan for i in k]
    same = np.zeros(len(d))
    tm, et, sy = d["time"].to_numpy(), d["exit_time"].to_numpy(), d["symbol"].to_numpy()
    for i in range(len(d)):
        lo = max(0, i - 400)
        same[i] = np.sum((sy[lo:i] == sy[i]) & (et[lo:i] > tm[i]))
    d["same_open"] = same
    return d


COLS = ["hour", "wday", "tfm", "dir", "fam", "cls", "stop_pct", "stop_rel", "last5", "last20", "all30", "same_open"]


def main():
    d = features(PR.load())
    tr, te = d[d["time"] < SPLIT], d[d["time"] >= SPLIT]
    y = tr["r"].clip(-3, 5)
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=80, random_state=1)
    m.fit(tr[COLS], y)
    # the skip line, chosen on the training period with 4-fold time splits (out-of-fold predictions)
    oof = np.full(len(tr), np.nan)
    folds = np.array_split(np.arange(len(tr)), 4)
    for k in range(1, 4):
        fit_i = np.concatenate(folds[:k])
        mm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=80, random_state=1)
        mm.fit(tr.iloc[fit_i][COLS], y.iloc[fit_i])
        oof[folds[k]] = mm.predict(tr.iloc[folds[k]][COLS])
    ok = ~np.isnan(oof)
    best, rows = None, []
    for q in (0.05, 0.10, 0.15, 0.20, 0.30):
        line = np.quantile(oof[ok], q)
        keep = oof[ok] >= line
        gain = float(tr["r"].to_numpy()[ok][keep].sum() - tr["r"].to_numpy()[ok].sum())
        rows.append({"skip_share": q, "line": round(float(line), 3), "train_oof_gain_r": round(gain, 1)})
        if best is None or gain > best[1]:
            best = (q, gain, line)
    pt = m.predict(te[COLS])
    line_te = np.quantile(m.predict(tr[COLS]), best[0])
    keep = pt >= line_te
    res = {"train_trades": int(len(tr)), "check_trades": int(len(te)), "lines": rows, "chosen_skip_share": best[0],
           "check": {"all_r": round(float(te["r"].sum()), 1), "kept_r": round(float(te["r"][keep].sum()), 1),
                     "skipped_r": round(float(te["r"][~keep].sum()), 1), "skipped_n": int((~keep).sum()),
                     "avg_r_all": round(float(te["r"].mean()), 3), "avg_r_kept": round(float(te["r"][keep].mean()), 3),
                     "avg_r_skipped": round(float(te["r"][~keep].mean()), 3)},
           "per_bot": {}, "deciles_check": []}
    for n in (1, 2, 3, 4):
        s = (te["bot"] == n).to_numpy()
        res["per_bot"][f"bot{n}"] = {"all_r": round(float(te["r"][s].sum()), 1), "kept_r": round(float(te["r"][s & keep].sum()), 1),
                                     "skipped_n": int((s & ~keep).sum()), "n": int(s.sum())}
    q = pd.qcut(pt, 10, labels=False, duplicates="drop")
    for k, g in te.assign(q=q).groupby("q"):
        res["deciles_check"].append({"decile": int(k), "n": int(len(g)), "avg_r": round(float(g["r"].mean()), 3)})
    # feature effect, plain: average R in the check period by bucket of the strongest single features
    res["simple"] = {}
    for c in ("last5", "last20", "all30", "stop_rel", "same_open"):
        b = pd.qcut(te[c], 4, duplicates="drop")
        res["simple"][c] = {str(k): [int(len(g)), round(float(g["r"].mean()), 3)] for k, g in te.groupby(b, observed=True)}
    od = ROOT / "reports" / "metalabel"
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
