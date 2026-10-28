"""Backtest agent: for every classified news event, measures what the stock actually did
after a realistic delay, then picks trading rules on the early period and tests them,
untouched, on the later period."""
import numpy as np
import pandas as pd

import config

NY = "America/New_York"

# The active track (intraday / swing / position). Set once per run with set_track().
TRACK = config.TRACKS["intraday"]


def set_track(name):
    global TRACK
    TRACK = config.TRACKS[name]


def rc(h):
    """Return-column name for horizon h in the active track, e.g. ret_15m or ret_3d."""
    return f"ret_{h}{TRACK['unit']}"


def _in_rth(ts_utc):
    t = ts_utc.tz_convert(NY)
    minutes = t.hour * 60 + t.minute
    return t.weekday() < 5 and 9 * 60 + 30 <= minutes < 16 * 60


def measure_reactions(events, bars_by_symbol):
    """Adds entry price and raw returns ret_{h}m for each horizon."""
    lat = pd.Timedelta(seconds=config.ENTRY_LATENCY_SEC)
    rows = []
    for ev in events.itertuples(index=False):
        bars = bars_by_symbol.get(ev.symbol)
        rec = ev._asdict()
        rec.update({"entry_time": pd.NaT, "entry_px": np.nan, "skip_reason": None})
        for h in config.HORIZONS_MIN:
            rec[f"ret_{h}m"] = np.nan
        if bars is None or bars.empty:
            rec["skip_reason"] = "no bars"
            rows.append(rec)
            continue
        if config.RTH_ONLY and not _in_rth(ev.time):
            rec["skip_reason"] = "outside market hours"
            rows.append(rec)
            continue

        target = ev.time + lat
        # First bar that starts at or after target is the first one we could trade in.
        idx = bars.index.searchsorted(target.ceil("min"), side="left")
        if idx >= len(bars) or bars.index[idx] - target > pd.Timedelta(minutes=3):
            rec["skip_reason"] = "no trading right after news"
            rows.append(rec)
            continue
        entry_t = bars.index[idx]
        entry_px = bars["o"].iat[idx]
        day = entry_t.tz_convert(NY).date()
        rec["entry_time"], rec["entry_px"] = entry_t, entry_px

        for h in config.HORIZONS_MIN:
            # Bars are stamped by their start; the bar starting at entry+h-1 closes at entry+h.
            exit_target = entry_t + pd.Timedelta(minutes=h - 1)
            j = max(idx, bars.index.searchsorted(exit_target, side="right") - 1)
            exit_t = bars.index[j]
            if exit_t.tz_convert(NY).date() != day or exit_t.tz_convert(NY).hour >= 16:
                continue  # don't hold overnight
            rec[f"ret_{h}m"] = bars["c"].iat[j] / entry_px - 1
        rows.append(rec)
    return pd.DataFrame(rows)


def measure_reactions_daily(events, daily_by_symbol, market_daily):
    """Multi-day tracks. Entry at the first market open after the news (+ latency);
    exit at the close of day N, counting the entry day as day 1. Stored returns are
    EXCESS returns versus the market over the same days, so a trade only counts as a
    win if the stock beat the market."""
    lat = pd.Timedelta(seconds=config.ENTRY_LATENCY_SEC)
    mkt = market_daily
    rows = []
    for ev in events.itertuples(index=False):
        rec = ev._asdict()
        rec.update({"entry_time": pd.NaT, "entry_px": np.nan, "skip_reason": None})
        for h in TRACK["horizons"]:
            rec[rc(h)] = np.nan
            rec[f"raw_{h}d"] = np.nan
        bars = daily_by_symbol.get(ev.symbol)
        if bars is None or bars.empty:
            rec["skip_reason"] = "no bars"
            rows.append(rec)
            continue
        # Daily bars are stamped at the session date; the open is 09:30 New York.
        opens = bars.index + pd.Timedelta(hours=9, minutes=30)
        opens = opens.tz_localize(NY) if opens.tz is None else opens
        i = opens.searchsorted((ev.time + lat).tz_convert(NY), side="left")
        if i >= len(bars):
            rec["skip_reason"] = "no trading after news"
            rows.append(rec)
            continue
        entry_day = bars.index[i]
        entry_px = bars["o"].iat[i]
        rec["entry_time"], rec["entry_px"] = opens[i].tz_convert("UTC"), entry_px
        for h in TRACK["horizons"]:
            j = i + h - 1
            if j >= len(bars):
                continue
            exit_day = bars.index[j]
            raw = bars["c"].iat[j] / entry_px - 1
            m_ret = np.nan
            if mkt is not None and entry_day in mkt.index and exit_day in mkt.index:
                m_ret = mkt.at[exit_day, "c"] / mkt.at[entry_day, "o"] - 1
            rec[f"raw_{h}d"] = raw
            rec[rc(h)] = raw - m_ret if not np.isnan(m_ret) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _cost(sel):
    extra = config.HEDGE_COST_BPS if TRACK["unit"] == "d" else 0
    return (sel["cost_bps"] + extra) / 1e4


def _trade_stats(r):
    """r = net returns per trade (fractions)."""
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {"trades": 0, "avg_bps": np.nan, "median_bps": np.nan, "win_rate": np.nan,
                "t_stat": np.nan, "total_pct": np.nan, "worst_bps": np.nan}
    sd = r.std(ddof=1) if n > 1 else np.nan
    return {
        "trades": n,
        "avg_bps": r.mean() * 1e4,
        "median_bps": r.median() * 1e4,
        "win_rate": (r > 0).mean(),
        "t_stat": r.mean() / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan,
        "total_pct": r.sum() * 100,
        "worst_bps": r.min() * 1e4,
    }


def select(df, threshold, min_conf=0.0, require_new=False, category=None):
    sel = df[(df["score"].abs() >= threshold) & (df["confidence"] >= min_conf)]
    if require_new:
        sel = sel[sel["new_information"]]
    if category:
        sel = sel[sel["category"] == category]
    return sel


def strategy_returns(df, horizon, threshold, min_conf=0.0, require_new=False, category=None):
    """Trade in the direction of the score when |score| >= threshold (optionally only one
    type of news). Net of a per-stock cost that depends on its liquidity score."""
    sel = select(df, threshold, min_conf, require_new, category)
    return np.sign(sel["score"]) * sel[rc(horizon)] - _cost(sel)


def position_size(sel, horizon):
    """Dollars per trade. Volatile stocks get smaller positions, thin stocks are capped
    by how much actually trades in a minute."""
    days = horizon / 390 if TRACK["unit"] == "m" else horizon
    sigma_h = sel["daily_vol"].fillna(0.03) * np.sqrt(days)
    by_risk = config.EQUITY * config.RISK_PCT / np.maximum(2 * sigma_h, 1e-4)
    by_equity = config.EQUITY * config.MAX_POSITION_PCT
    # Entry happens within a minute (intraday) or at the opening auction / first minutes
    # (multi-day); either way we cap at a share of an average minute's dollar volume.
    by_liquidity = sel["avg_dollar_volume"].fillna(0) / 390 * config.MAX_PCT_OF_MINUTE_VOLUME
    return np.minimum(np.minimum(by_risk, by_equity), by_liquidity)


def dollar_results(df, rule):
    sel = select(df, rule["threshold"], rule["min_conf"], rule["require_new"], rule.get("category"))
    net = np.sign(sel["score"]) * sel[rc(rule["horizon"])] - _cost(sel)
    size = position_size(sel, rule["horizon"])
    pnl = (size * net).dropna()
    curve = pnl.cumsum()
    dd = (curve - curve.cummax()).min() if len(curve) else np.nan
    return {"pnl_usd": pnl.sum(), "max_drawdown_usd": dd, "avg_size_usd": size[pnl.index].mean(),
            "curve_usd": curve}


def _edges(width):
    # Last edge slightly above 100 so a score of exactly 100 lands in the top band.
    e = list(range(0, 100, width)) + [100.001]
    return e


def _label(interval):
    return f"{int(interval.left)}-{min(100, int(round(interval.right)))}"


def reaction_table(df, width=10):
    """Average raw move (no direction, no cost) by sentiment score 0-100."""
    d = df.dropna(subset=["entry_px"]).copy()
    d["band"] = pd.cut(d["sentiment_score"], _edges(width), include_lowest=True, right=False)
    out = []
    for b, g in d.groupby("band", observed=True):
        row = {"band": _label(b), "events": len(g)}
        for h in TRACK["horizons"]:
            row[f"h{h}_avg_bps"] = g[rc(h)].mean() * 1e4
        out.append(row)
    return pd.DataFrame(out)


def band_table(df, rule, column, width=10):
    """Performance of the chosen rule split by a 0-100 score, plus a slope:
    how many bps the average trade changes per +10 points of that score."""
    sel = select(df.dropna(subset=["entry_px"]), rule["threshold"], rule["min_conf"],
                 rule["require_new"], rule.get("category"))
    sel = sel.assign(net=np.sign(sel["score"]) * sel[rc(rule["horizon"])] - _cost(sel))
    sel = sel.dropna(subset=["net", column])
    rows = []
    bands = pd.cut(sel[column], _edges(width), include_lowest=True, right=False)
    for b, g in sel.groupby(bands, observed=True):
        s = _trade_stats(g["net"])
        s["band"] = _label(b)
        rows.append(s)
    slope = np.nan
    if len(sel) >= 20 and sel[column].std() > 0:
        slope = np.polyfit(sel[column], sel["net"] * 1e4, 1)[0] * 10
    return pd.DataFrame(rows), slope


def category_table(df, horizon=None, threshold=0.5):
    horizon = horizon or TRACK["default_h"]
    d = df.dropna(subset=["entry_px"])
    out = []
    for cat, g in d.groupby("category"):
        s = _trade_stats(strategy_returns(g, horizon, threshold))
        s["category"] = cat
        out.append(s)
    return pd.DataFrame(out).sort_values("trades", ascending=False) if out else pd.DataFrame()


def walk_forward(df):
    """Pick the best (horizon, threshold, confidence, new-only) on the train period,
    then report the SAME rule on the test period it never saw."""
    d = df.dropna(subset=["entry_px"]).sort_values("time")
    if d.empty:
        return None
    split = d["time"].quantile(config.TRAIN_FRACTION)
    cutoff = pd.Timestamp(config.MODEL_KNOWLEDGE_CUTOFF, tz="UTC") + pd.Timedelta(days=1)
    uses_llm = (d["model"] != "keywords").any()
    split_moved = False
    if uses_llm and split < cutoff:
        # The model may remember what happened after older news. Never test on that period.
        split, split_moved = cutoff, True
    train, test = d[d["time"] <= split], d[d["time"] > split]

    # Also try "only this type of news" for every type with enough history, because some
    # effects (e.g. drift after earnings) live in one category and drown in the average.
    cat_counts = train["category"].value_counts()
    categories = [None] + [c for c, n in cat_counts.items() if n >= config.MIN_TRADES_FOR_RULE]
    grid = []
    for h in TRACK["horizons"]:
        for th in [0.3, 0.5, 0.7, 0.85]:
            for conf in [0.0, 0.6, 0.8]:
                for new in [False, True]:
                    for cat in categories:
                        s = _trade_stats(strategy_returns(train, h, th, conf, new, cat))
                        if s["trades"] >= config.MIN_TRADES_FOR_RULE:
                            grid.append({"horizon": h, "threshold": th, "min_conf": conf,
                                         "new_only": new, "category": cat, **s})
    if not grid:
        return {"split": split, "split_moved": split_moved, "uses_llm": uses_llm,
                "grid": pd.DataFrame(), "best": None,
                "train": None, "test": None, "n_train": len(train), "n_test": len(test)}
    g = pd.DataFrame(grid)
    # Rank by t-stat, not raw average, so a rule built on 3 lucky trades can't win.
    best = g.sort_values("t_stat", ascending=False).iloc[0]
    rule = dict(horizon=int(best["horizon"]), threshold=float(best["threshold"]),
                min_conf=float(best["min_conf"]), require_new=bool(best["new_only"]),
                category=best["category"] if isinstance(best["category"], str) else None)
    contamination = None
    if uses_llm:
        # Same rule, before vs after the model's knowledge cutoff. A big drop after the
        # cutoff suggests the model was partly "remembering" the outcome.
        before = _trade_stats(strategy_returns(d[d["time"] < cutoff], **rule))
        after = _trade_stats(strategy_returns(d[d["time"] >= cutoff], **rule))
        contamination = {"before": before, "after": after}
    return {
        "split": split, "split_moved": split_moved, "uses_llm": uses_llm,
        "contamination": contamination, "best": rule, "grid": g,
        "train": _trade_stats(strategy_returns(train, **rule)),
        "test": _trade_stats(strategy_returns(test, **rule)),
        "test_usd": dollar_results(test, rule),
        "n_train": len(train), "n_test": len(test),
        "configs_tried": len(g),
    }
