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

def _sigma(daily_vol, h):
    days = h / 390 if TRACK["unit"] == "m" else h
    v = daily_vol if daily_vol == daily_vol and daily_vol else 0.03
    return v * np.sqrt(days)


def _stop(ks, sig):
    """Stop distance for an exit mode: ks[0] sigmas, never further than its cap (if any)."""
    st = ks[0] * sig
    return min(st, ks[3]) if len(ks) > 3 and ks[3] else st


def _path_exit(entry_px, o, hi, lo, c, direction, stop_pct, target_pct, gap_fill, trailing=False):
    """Walk bars in order; return (bar offset, exit price) of the first stop/target touch,
    or (None, None) if neither is hit. direction +1 long, -1 short. With trailing=True the
    stop follows the best price seen so far (updated after each bar, so a bar can't both
    set a new high and trigger the stop it created)."""
    best = entry_px
    target = None
    if target_pct is not None:
        target = entry_px * (1 + target_pct) if direction > 0 else entry_px * (1 - target_pct)
    for k in range(len(c)):
        stop = best * (1 - stop_pct) if direction > 0 else best * (1 + stop_pct)
        if gap_fill and k > 0:
            if direction > 0 and o[k] <= stop:
                return k, o[k]
            if direction < 0 and o[k] >= stop:
                return k, o[k]
        hit_stop = lo[k] <= stop if direction > 0 else hi[k] >= stop
        hit_tgt = target is not None and (hi[k] >= target if direction > 0 else lo[k] <= target)
        if hit_stop:
            return k, stop
        if hit_tgt:
            return k, target
        if trailing:
            best = max(best, hi[k]) if direction > 0 else min(best, lo[k])
    return None, None


def exit_col(h, mode, side):
    return f"x_{h}{TRACK['unit']}_{mode}_{side}"


def pos_ret(df, h, mode):
    """Position return (already signed) for each row, trading in the direction of its score."""
    long_ = np.sign(df["score"]) > 0
    if mode == "time":
        return np.sign(df["score"]) * df[rc(h)]
    return pd.Series(np.where(long_, df[exit_col(h, mode, "L")], df[exit_col(h, mode, "S")]),
                     index=df.index)


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

        O, H, L, C = (bars[k].to_numpy() for k in ("o", "h", "l", "c"))
        for h in config.HORIZONS_MIN:
            # Bars are stamped by their start; the bar starting at entry+h-1 closes at entry+h.
            exit_target = entry_t + pd.Timedelta(minutes=h - 1)
            j = max(idx, bars.index.searchsorted(exit_target, side="right") - 1)
            exit_t = bars.index[j]
            if exit_t.tz_convert(NY).date() != day or exit_t.tz_convert(NY).hour >= 16:
                continue  # don't hold overnight
            raw = C[j] / entry_px - 1
            rec[f"ret_{h}m"] = raw
            sig = _sigma(getattr(ev, "daily_vol", np.nan), h)
            for mode, ks in config.EXIT_MODES.items():
                if ks is None:
                    continue
                for side, d in (("L", 1), ("S", -1)):
                    k, px = _path_exit(entry_px, O[idx:j + 1], H[idx:j + 1], L[idx:j + 1], C[idx:j + 1],
                                       d, _stop(ks, sig), None if ks[1] is None else ks[1] * sig,
                                       gap_fill=False, trailing=ks[2])
                    rec[exit_col(h, mode, side)] = d * ((px if k is not None else C[j]) / entry_px - 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def measure_reactions_daily(events, daily_by_symbol, market_daily):
    """Multi-day tracks. Entry at the first market open after the news (+ latency);
    exit at the close of day N, counting the entry day as day 1. Stored returns are
    EXCESS returns versus the market over the same days, so a trade only counts as a
    win if the stock beat the market."""
    lat = pd.Timedelta(seconds=config.ENTRY_LATENCY_SEC)
    # market_daily: one DataFrame, or {symbol: DataFrame} to hedge small stocks with the
    # small-cap index and the rest with the S&P 500.
    markets = market_daily if isinstance(market_daily, dict) else {"_": market_daily}
    rows = []
    for ev in events.itertuples(index=False):
        rec = ev._asdict()
        rec.update({"entry_time": pd.NaT, "entry_px": np.nan, "skip_reason": None})
        for h in TRACK["horizons"]:
            rec[rc(h)] = np.nan
            rec[f"raw_{h}d"] = np.nan
        if len(markets) > 1:
            liq = getattr(ev, "liquidity_score", np.nan)
            small = liq == liq and liq < config.SMALL_BELOW_LIQUIDITY
            hedge = config.SMALL_MARKET_SYMBOL if small and config.SMALL_MARKET_SYMBOL in markets else config.MARKET_SYMBOL
        else:
            hedge = next(iter(markets))
        mkt = markets.get(hedge)
        rec["hedge"] = hedge
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
        O, H, L, C = (bars[k].to_numpy() for k in ("o", "h", "l", "c"))

        def mret(d0, d1):
            if mkt is not None and d0 in mkt.index and d1 in mkt.index:
                return mkt.at[d1, "c"] / mkt.at[d0, "o"] - 1
            return np.nan

        for h in TRACK["horizons"]:
            j = i + h - 1
            if j >= len(bars):
                continue
            exit_day = bars.index[j]
            raw = C[j] / entry_px - 1
            m_ret = mret(entry_day, exit_day)
            rec[f"raw_{h}d"] = raw
            rec[rc(h)] = raw - m_ret if not np.isnan(m_ret) else np.nan
            sig = _sigma(getattr(ev, "daily_vol", np.nan), h)
            for mode, ks in config.EXIT_MODES.items():
                if ks is None:
                    continue
                for side, d in (("L", 1), ("S", -1)):
                    k, px = _path_exit(entry_px, O[i:j + 1], H[i:j + 1], L[i:j + 1], C[i:j + 1],
                                       d, _stop(ks, sig), None if ks[1] is None else ks[1] * sig,
                                       gap_fill=True, trailing=ks[2])
                    end = j if k is None else i + k
                    px = C[j] if k is None else px
                    m = mret(entry_day, bars.index[end])
                    # Hedged: the stock's move minus the market's move over the days actually held.
                    rec[exit_col(h, mode, side)] = d * ((px / entry_px - 1) - m) if not np.isnan(m) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _cost(sel):
    extra = config.HEDGE_COST_BPS if TRACK["unit"] == "d" else 0
    return (sel["cost_bps"] + extra) / 1e4


def _borrow_rate(df):
    """Yearly fee for borrowing the shares of a short, per row (fraction). Easy-to-borrow
    stocks are cheap; hard-to-borrow ones (often small caps right after an offering) can
    cost tens of percent a year."""
    etb = df["easy_to_borrow"] if "easy_to_borrow" in df else pd.Series(True, index=df.index)
    return np.where(etb.fillna(False).astype(bool), config.BORROW_FEE_ETB_PCT, config.BORROW_FEE_HTB_PCT) / 100


def _hold_days(h):
    # Conservative: charge the borrow for the maximum holding period, even if a stop exits earlier.
    return h / 390 if TRACK["unit"] == "m" else h * 365 / 252


def net_ret(sel, h, mode):
    """Net return per trade: position return minus trading cost, hedge cost and, for
    shorts, the borrow fee."""
    short = np.sign(sel["score"]) < 0
    borrow = np.where(short, _borrow_rate(sel) * _hold_days(h) / 365, 0.0)
    return pos_ret(sel, h, mode) - _cost(sel) - borrow


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
    if "entry_px" in sel and config.MIN_PRICE:
        sel = sel[~(sel["entry_px"] < config.MIN_PRICE)]
    if "easy_to_borrow" in sel and config.SKIP_HTB_SHORTS:
        # Squeeze protection: hard-to-borrow usually means a crowded short.
        sel = sel[~((np.sign(sel["score"]) < 0) & ~sel["easy_to_borrow"].fillna(False).astype(bool))]
    if "shortable" in sel and not config.ALLOW_UNSHORTABLE:
        # A short the broker won't allow can't be traded.
        sel = sel[~((np.sign(sel["score"]) < 0) & ~sel["shortable"].fillna(False).astype(bool))]
    if require_new:
        sel = sel[sel["new_information"]]
    if category:
        sel = sel[sel["category"] == category]
    return sel


def strategy_returns(df, horizon, threshold, min_conf=0.0, require_new=False, category=None,
                     exit="time"):
    """Trade in the direction of the score when |score| >= threshold (optionally only one
    type of news). Net of a per-stock cost that depends on its liquidity score."""
    sel = select(df, threshold, min_conf, require_new, category)
    return net_ret(sel, horizon, exit)


def position_size(sel, horizon, exit="time"):
    """Dollars per trade. Volatile stocks get smaller positions, thin stocks are capped
    by how much actually trades in a minute. With a capped stop the risk distance is the
    cap (when closer than 2 sigma), so the position is bigger for the same risk."""
    days = horizon / 390 if TRACK["unit"] == "m" else horizon
    sigma_h = sel["daily_vol"].fillna(0.03) * np.sqrt(days)
    dist = 2 * sigma_h
    ks = config.EXIT_MODES.get(exit)
    if ks and len(ks) > 3 and ks[3]:
        dist = np.minimum(dist, ks[3])
    by_risk = config.EQUITY * config.RISK_PCT / np.maximum(dist, 1e-4)
    by_equity = config.EQUITY * config.MAX_POSITION_PCT
    # Entry happens within a minute (intraday) or at the opening auction / first minutes
    # (multi-day); either way we cap at a share of an average minute's dollar volume.
    by_liquidity = sel["avg_dollar_volume"].fillna(0) / 390 * config.MAX_PCT_OF_MINUTE_VOLUME
    return np.minimum(np.minimum(by_risk, by_equity), by_liquidity)


def risk_scale(equity):
    """Risk grows with the balance, slower than the balance (square root), and shrinks the
    same way after losses."""
    ratio = max(equity, 1) / config.EQUITY
    return float(np.clip(ratio ** config.RISK_SCALING_POWER, config.RISK_SCALE_MIN, config.RISK_SCALE_MAX))


def dollar_results(df, rule):
    """Replays the rule's trades in time order on an account that starts at EQUITY and
    obeys the risk rules: risk per trade and the daily loss limit scale with the balance,
    the day stops once its loss hits the limit, and everything halts once the balance
    falls MAX_DRAWDOWN_PCT below the starting balance."""
    sel = select(df, rule["threshold"], rule["min_conf"], rule["require_new"], rule.get("category"))
    sel = sel.assign(net=net_ret(sel, rule["horizon"], rule.get("exit", "time")))
    sel = sel.assign(base_size=position_size(sel, rule["horizon"], rule.get("exit", "time")))
    sel = sel.dropna(subset=["net"]).sort_values("entry_time")

    equity = peak = config.EQUITY
    day, day_pnl, day_start = None, 0.0, equity
    stopped_days, halted_on = set(), None
    pnls, sizes, times, fracs = [], [], [], []
    hold = (pd.Timedelta(minutes=rule["horizon"]) if TRACK["unit"] == "m"
            else pd.Timedelta(days=int(np.ceil(rule["horizon"] * 7 / 5))))
    open_pos, skipped_limits = [], 0                       # (exit time, size, is_short)
    for r in sel.itertuples(index=False):
        now = pd.Timestamp(r.entry_time)
        open_pos = [p for p in open_pos if p[0] > now]
        is_short = r.score < 0
        short_expo = sum(p[1] for p in open_pos if p[2])
        d = now.tz_convert(NY).date()
        if d != day:
            day, day_pnl, day_start = d, 0.0, equity
        scale = risk_scale(equity)
        if day_pnl <= -config.DAILY_LOSS_PCT * scale * day_start:
            stopped_days.add(d)
            continue
        size = r.base_size * scale * (equity / config.EQUITY)
        if len(open_pos) >= config.MAX_OPEN_POSITIONS or (
                is_short and short_expo + size > config.MAX_SHORT_EXPOSURE_PCT * equity):
            skipped_limits += 1
            continue
        open_pos.append((now + hold, size, is_short))
        pnl = size * r.net
        fracs.append(pnl / equity)
        equity += pnl
        day_pnl += pnl
        peak = max(peak, equity)
        pnls.append(pnl); sizes.append(size); times.append(r.entry_time)
        if equity <= config.EQUITY * (1 - config.MAX_DRAWDOWN_PCT):
            halted_on = d
            break

    curve = pd.Series(np.cumsum(pnls), index=times) if pnls else pd.Series(dtype=float)
    # Peak includes the starting balance, so a loss from day one counts as drawdown.
    dd = (curve - curve.cummax().clip(lower=0)).min() if len(curve) else np.nan
    return {"pnl_usd": float(np.sum(pnls)) if pnls else 0.0, "max_drawdown_usd": dd,
            "avg_size_usd": float(np.mean(sizes)) if sizes else np.nan, "curve_usd": curve,
            "end_equity": equity, "trades_taken": len(pnls),
            "days_stopped": len(stopped_days), "halted_on": halted_on,
            "trade_fracs": np.array(fracs), "times": times, "skipped_limits": skipped_limits}


def monte_carlo(test_usd, runs=None, seed=3):
    """Risk of the rule over one year: the test-period trades (as a share of the balance)
    are replayed in random order, at the test period's pace, many times. Returns the
    chance of hitting the overall stop (MAX_DRAWDOWN_PCT below the start) within a year,
    and the spread of yearly results and drawdowns."""
    runs = config.MONTE_CARLO_RUNS if runs is None else runs
    f, t = test_usd.get("trade_fracs"), test_usd.get("times")
    if f is None or len(f) < 10:
        return None
    span_days = max((pd.Timestamp(t[-1]) - pd.Timestamp(t[0])).days, 30)
    per_year = int(round(len(f) * 365 / span_days))
    rng = np.random.default_rng(seed)
    floor = 1 - config.MAX_DRAWDOWN_PCT
    ends, dds, halts = [], [], 0
    for _ in range(runs):
        path = np.cumprod(1 + rng.choice(f, size=per_year, replace=True))
        hit = np.nonzero(path <= floor)[0]
        if len(hit):
            halts += 1
            path = path[:hit[0] + 1]
        peak = np.maximum.accumulate(np.concatenate([[1.0], path]))
        dds.append(float((np.concatenate([[1.0], path]) / peak - 1).min()))
        ends.append(float(path[-1] - 1))
    ends, dds = np.array(ends), np.array(dds)
    return {"runs": runs, "trades_per_year": per_year, "halt_prob": halts / runs,
            "median_year": float(np.median(ends)), "p5_year": float(np.percentile(ends, 5)),
            "p95_year": float(np.percentile(ends, 95)), "loss_prob": float((ends < 0).mean()),
            "median_dd": float(np.median(dds)), "p5_dd": float(np.percentile(dds, 5))}


def _with_htb(test, rule):
    """Same rule in the test period if hard-to-borrow shorts were allowed (no squeeze filter)."""
    if not config.SKIP_HTB_SHORTS or "easy_to_borrow" not in test:
        return None
    config.SKIP_HTB_SHORTS = False
    try:
        return _trade_stats(strategy_returns(test, **rule))
    finally:
        config.SKIP_HTB_SHORTS = True


def side_split(df, rule):
    """The rule's trades split into longs and shorts, plus how many shorts the broker
    would not have allowed and how many are hard to borrow."""
    out = {}
    base = df[(df["score"].abs() >= rule["threshold"]) & (df["confidence"] >= rule["min_conf"])]
    if rule["require_new"]:
        base = base[base["new_information"]]
    if rule.get("category"):
        base = base[base["category"] == rule["category"]]
    shorts_all = base[np.sign(base["score"]) < 0]
    if "shortable" in base:
        out["blocked_shorts"] = int((~shorts_all["shortable"].fillna(False).astype(bool)).sum())
    sel = select(df, rule["threshold"], rule["min_conf"], rule["require_new"], rule.get("category"))
    net = net_ret(sel, rule["horizon"], rule.get("exit", "time"))
    is_short = np.sign(sel["score"]) < 0
    out["long"] = _trade_stats(net[~is_short])
    out["short"] = _trade_stats(net[is_short])
    if "easy_to_borrow" in sel:
        sh = sel[is_short]
        out["htb_shorts"] = int((~sh["easy_to_borrow"].fillna(False).astype(bool)).sum())
    return out


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
    sel = sel.assign(net=net_ret(sel, rule["horizon"], rule.get("exit", "time")))
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


def group_table(df, rule, column, bins=None, labels=None):
    """The chosen rule's trades (whole period) grouped by a detail column, e.g. offering
    type, or a number cut into bins."""
    if column not in df:
        return pd.DataFrame()
    sel = select(df.dropna(subset=["entry_px"]), rule["threshold"], rule["min_conf"],
                 rule["require_new"], rule.get("category"))
    sel = sel.assign(net=net_ret(sel, rule["horizon"], rule.get("exit", "time"))).dropna(subset=["net"])
    key = sel[column]
    if bins is not None:
        key = pd.cut(pd.to_numeric(key, errors="coerce"), bins, labels=labels, include_lowest=True)
        key = key.astype(object).where(key.notna(), "לא צוין")
    else:
        key = key.astype(object).where(key.notna(), "לא צוין")
    rows = []
    for k, g in sel.groupby(key, observed=True):
        st = _trade_stats(g["net"])
        st["group"] = str(k)
        rows.append(st)
    return pd.DataFrame(rows).sort_values("trades", ascending=False) if rows else pd.DataFrame()


def category_table(df, horizon=None, threshold=0.5):
    horizon = horizon or TRACK["default_h"]
    d = df.dropna(subset=["entry_px"])
    out = []
    for cat, g in d.groupby("category"):
        s = _trade_stats(strategy_returns(g, horizon, threshold))
        s["category"] = cat
        out.append(s)
    return pd.DataFrame(out).sort_values("trades", ascending=False) if out else pd.DataFrame()


THRESHOLDS = [0.3, 0.5, 0.7, 0.85]
CONFIDENCES = [0.0, 0.6, 0.8]


def _search_exits():
    return [m for m in config.EXIT_MODES if m != "time" or config.ALLOW_TIME_EXIT]


def luck_test(train, categories, real_best_t, runs=None, seed=11):
    """Multiple-testing correction. Hundreds of rules are tried, so the best one looks good
    even on pure noise. Here the ENTIRE search is repeated `runs` times on the training
    data with each event's direction flipped at random (same stocks, same days, same exits,
    but the sentiment no longer says anything). Returns the share of those random searches
    whose best t-stat was at least the real one: the chance the real rule is just luck."""
    runs = config.SHUFFLE_RUNS if runs is None else runs
    if runs <= 0 or train.empty or not np.isfinite(real_best_t):
        return None
    rng = np.random.default_rng(seed)
    score = train["score"].to_numpy()
    sgn = np.sign(score)
    absd = np.abs(score)
    conf = train["confidence"].to_numpy()
    new = train["new_information"].fillna(False).to_numpy().astype(bool)
    cat = train["category"].to_numpy()
    cost = _cost(train).to_numpy()
    borrow_rate = _borrow_rate(train)
    can_short = (train["shortable"].fillna(False).astype(bool).to_numpy()
                 if "shortable" in train and not config.ALLOW_UNSHORTABLE else np.ones(len(train), bool))
    if "easy_to_borrow" in train and config.SKIP_HTB_SHORTS:
        can_short = can_short & train["easy_to_borrow"].fillna(False).astype(bool).to_numpy()
    price_ok = ~(train["entry_px"].to_numpy() < config.MIN_PRICE) if config.MIN_PRICE else np.ones(len(train), bool)

    masks = []
    for th in THRESHOLDS:
        for cf in CONFIDENCES:
            for nw in (False, True):
                for c in categories:
                    m = (absd >= th) & (conf >= cf) & price_ok
                    if nw:
                        m &= new
                    if c:
                        m &= cat == c
                    masks.append(m)
    M = np.array(masks, dtype=np.float32)               # rules x events

    legs = []                                            # (long return, short return) per horizon/exit
    for h in TRACK["horizons"]:
        b = borrow_rate * _hold_days(h) / 365
        for ex in _search_exits():
            if ex == "time":
                r = train[rc(h)].to_numpy()
                L, S = r, -r
            else:
                L, S = train[exit_col(h, ex, "L")].to_numpy(), train[exit_col(h, ex, "S")].to_numpy()
            legs.append((L, np.where(can_short, S - b, np.nan)))

    def best_t(direction):
        best = -np.inf
        for L, S in legs:
            r = np.where(direction > 0, L, S) - cost
            r = np.where(sgn == 0, np.nan, r)            # the real rule never trades a 0 score
            ok = np.isfinite(r)
            rv = np.where(ok, r, 0.0).astype(np.float64)
            n = M @ ok.astype(np.float32)
            s1 = M @ rv
            s2 = M @ (rv * rv)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = s1 / n
                var = (s2 - n * mean * mean) / (n - 1)
                t = mean / np.sqrt(var / n)
            t = np.where((n >= config.MIN_TRADES_FOR_RULE) & (var > 0), t, -np.inf)
            best = max(best, float(np.nanmax(t)))
        return best

    hits = 0
    for _ in range(runs):
        flip = rng.choice([-1.0, 1.0], size=len(sgn))
        if best_t(sgn * flip) >= real_best_t:
            hits += 1
    return {"p": (hits + 1) / (runs + 1), "runs": runs, "hits": hits}


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
        for th in THRESHOLDS:
            for conf in CONFIDENCES:
                for new in [False, True]:
                    for cat in categories:
                        for ex in _search_exits():
                            s = _trade_stats(strategy_returns(train, h, th, conf, new, cat, ex))
                            if s["trades"] >= config.MIN_TRADES_FOR_RULE:
                                grid.append({"horizon": h, "threshold": th, "min_conf": conf,
                                             "new_only": new, "category": cat, "exit": ex, **s})
    if not grid:
        return {"split": split, "split_moved": split_moved, "uses_llm": uses_llm,
                "grid": pd.DataFrame(), "best": None,
                "train": None, "test": None, "n_train": len(train), "n_test": len(test)}
    g = pd.DataFrame(grid)
    # Rank by t-stat, not raw average, so a rule built on 3 lucky trades can't win.
    best = g.sort_values("t_stat", ascending=False).iloc[0]
    rule = dict(horizon=int(best["horizon"]), threshold=float(best["threshold"]),
                min_conf=float(best["min_conf"]), require_new=bool(best["new_only"]),
                category=best["category"] if isinstance(best["category"], str) else None,
                exit=str(best["exit"]))
    print(f"    luck test: repeating the search of {len(g)} rules {config.SHUFFLE_RUNS}x on shuffled directions")
    luck = luck_test(train, categories, float(best["t_stat"]))
    contamination = None
    if uses_llm:
        # Same rule, before vs after the model's knowledge cutoff. A big drop after the
        # cutoff suggests the model was partly "remembering" the outcome.
        before = _trade_stats(strategy_returns(d[d["time"] < cutoff], **rule))
        after = _trade_stats(strategy_returns(d[d["time"] >= cutoff], **rule))
        contamination = {"before": before, "after": after}
    exit_rows = []
    for ex in config.EXIT_MODES:
        st = _trade_stats(strategy_returns(test, **{**rule, "exit": ex}))
        exit_rows.append({"exit_he": config.EXIT_LABELS_HE.get(ex, ex), **st})
    return {
        "split": split, "split_moved": split_moved, "uses_llm": uses_llm,
        "exit_compare": pd.DataFrame(exit_rows),
        "contamination": contamination, "best": rule, "grid": g,
        "train": _trade_stats(strategy_returns(train, **rule)),
        "test": _trade_stats(strategy_returns(test, **rule)),
        "test_usd": (tu := dollar_results(test, rule)),
        "risk": monte_carlo(tu),
        "n_train": len(train), "n_test": len(test),
        "configs_tried": len(g),
        "luck": luck,
        "sides": {"train": side_split(train, rule), "test": side_split(test, rule)},
        "test_with_htb": _with_htb(test, rule),
    }


def candidate_check(df, rule, split, base_universe):
    """A fixed rule from an earlier run, no searching: whole period and the part after
    `split`, on all stocks / the stocks it was found on / the stocks added since, with and
    without the squeeze filter (hard-to-borrow shorts)."""
    d = df.dropna(subset=["entry_px"])
    r = {k: rule[k] for k in ("horizon", "threshold", "min_conf", "require_new", "category", "exit")}
    groups = [("כל המניות", d),
              (f"{len(base_universe)} המניות של הרצה 8", d[d["symbol"].isin(base_universe)]),
              ("המניות שנוספו (לא נראו מעולם)", d[~d["symbol"].isin(base_universe)])]
    rows = []
    for gname, g in groups:
        for htb in (False, True):
            old = config.SKIP_HTB_SHORTS
            config.SKIP_HTB_SHORTS = not htb
            try:
                for pname, part in (("כל התקופה", g), ("אחרי נקודת החלוקה", g[g["time"] > split])):
                    st = _trade_stats(strategy_returns(part, **r))
                    sel = select(part, r["threshold"], r["min_conf"], r["require_new"], r["category"])
                    st.update({"group": gname, "htb": "כן" if htb else "לא", "period": pname,
                               "short_share": float((np.sign(sel["score"]) < 0).mean()) if len(sel) else np.nan})
                    rows.append(st)
            finally:
                config.SKIP_HTB_SHORTS = old
    return pd.DataFrame(rows)
