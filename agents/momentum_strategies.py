"""Cross-sectional momentum (family 4 of the strategy map): instruments are ranked against each other
every week, and the bot trades this instrument only when it is among the strongest (buy) or the
weakest (sell) of its group. Same contract as agents/rules_agent.py.

Ranks use weekly closes (Friday) of every instrument in the FTMO universe, loaded with UNIVERSE_BARS
(the backtest's normal loader: every run downloads the whole market first). A position is opened on the first
bar of each day of the week from ranks of the weeks that already ended (no peeking) and closed on the
week's last bar (FTMO Standard closes before the weekend anyway).
"""
import numpy as np
import pandas as pd

from agents import ftmo_universe as U

NY = "America/New_York"
_WEEKLY = {}


def _default_loader(sym, tf, start, end):
    from agents import poc_agent as P
    return P.bars(sym, tf, start, end)


UNIVERSE_BARS = _default_loader


def weekly_closes(start, end):
    """Friday closes of every instrument (columns), weeks as rows. Cached per date range."""
    key = (start, end)
    if key not in _WEEKLY:
        cols = {}
        for s in U.SYMBOLS:
            try:
                b = UNIVERSE_BARS(s, "1Hour", start, end)
            except Exception:  # noqa: BLE001 - an instrument without data is left out of the ranking
                continue
            if b is None or len(b) < 50:
                continue
            ny = b["c"].copy()
            ny.index = ny.index.tz_convert(NY)
            cols[s] = ny.resample("W-FRI").last()
        _WEEKLY[key] = pd.DataFrame(cols).dropna(how="all")
    return _WEEKLY[key]


def _week_frame(df):
    """For every bar: its New York week (ending Friday), first/last bar of the week flags."""
    wk = df.index.tz_convert(NY).tz_localize(None).to_period("W-FRI")
    w = np.array([str(p) for p in wk])
    first = np.r_[True, w[1:] != w[:-1]]
    last = np.r_[w[1:] != w[:-1], True]
    return wk, first, last


def _signals(df, score_fn, group, top):
    """Shared engine: score_fn(weekly closes) -> scores per week x instrument (already known at that
    week's end); rank inside the group; top share buys, bottom share sells for the following week."""
    sym = df.attrs.get("symbol")
    n = len(df)
    z = np.zeros(n, bool)
    if not sym:
        return z, z, {"stop": ("atr", 2.5), "target": None}
    start = (df.index[0] - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end = (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    wc = weekly_closes(start, end)
    if sym not in wc.columns:
        return z, z, {"stop": ("atr", 2.5), "target": None}
    members = [s for s in wc.columns if group == "all" or U.CLASS.get(s) == U.CLASS.get(sym)]
    if len(members) < 3:
        return z, z, {"stop": ("atr", 2.5), "target": None}
    sc = score_fn(wc[members])
    k = max(1, int(round(top * len(members))))
    rank = sc.rank(axis=1, ascending=False)            # 1 = strongest
    cnt = sc.notna().sum(axis=1)
    is_top = (rank[sym] <= k) & (cnt >= 3)
    is_bot = (rank[sym] > cnt - k) & (cnt >= 3)
    wk, first, last = _week_frame(df)
    # enter on the first bar of every New York day of the week (shares are closed every evening, so they
    # re-enter each morning; 24-hour instruments simply stay in until the week's last bar)
    day = np.array([d.isoformat() for d in df.index.tz_convert(NY).date])
    first = np.r_[True, day[1:] != day[:-1]]
    prev = {str(p): str(p - 1) for p in set(wk)}         # ranks of the week that just ended
    top_w = {str(p): bool(v) for p, v in zip(sc.index.tz_localize(None).to_period("W-FRI"), is_top.to_numpy())}
    bot_w = {str(p): bool(v) for p, v in zip(sc.index.tz_localize(None).to_period("W-FRI"), is_bot.to_numpy())}
    w = [str(p) for p in wk]
    long_ = first & np.array([top_w.get(prev[x], False) for x in w])
    short = first & np.array([bot_w.get(prev[x], False) for x in w]) & ~long_
    return long_, short, {"stop": ("atr", 6.0), "target": None, "force_exit": last}


def s_mom_class(df):
    """12-week momentum inside its class (currencies, indices, metals, energy, crypto, shares): the top
    20% buy for the next week, the bottom 20% sell. Stop 6 ATR of the hourly bars (about a day's range: a weekly hold needs room), out on the week's last bar."""
    return _signals(df, lambda w: w / w.shift(12) - 1, "class", 0.2)


def s_mom_all_riskadj(df):
    """Risk-adjusted 12-week momentum across the whole universe (return / volatility of weekly returns):
    the 10 strongest buy, the 10 weakest sell for the next week."""
    def score(w):
        r = w.pct_change()
        return (w / w.shift(12) - 1) / (r.rolling(12, min_periods=8).std() * np.sqrt(12))
    return _signals(df, score, "all", 10 / 94)


def s_reversal_class(df):
    """Short-term reversal inside its class: last week's biggest losers (bottom 20%) buy, biggest winners
    sell, for one week (Jegadeesh 1990 / Lehmann 1990)."""
    return _signals(df, lambda w: -(w / w.shift(1) - 1), "class", 0.2)


# id: (name, fn, timeframes)
MOMENTUM = {
    "MO01": ("12-week momentum, top/bottom 20% of its class (weekly)", s_mom_class, ["1Hour"]),
    "MO02": ("Risk-adjusted 12-week momentum, top/bottom 10 of all (weekly)", s_mom_all_riskadj, ["1Hour"]),
    "MO03": ("1-week reversal, losers buy / winners sell, in class (weekly)", s_reversal_class, ["1Hour"]),
}
