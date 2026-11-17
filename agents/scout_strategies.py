"""Strategies the scout agent brings in (user, 11.11.2026: "a separate agent that only brings new strategies, from the
whole internet"). Same contract as agents/rules_agent.py: fn(df) -> (long, short, plan), ids SC001, SC002, ...

The scout (a scheduled Claude session, once a day) searches every source it can reach - YouTube, TikTok, Instagram,
Facebook, X, TradingView, BabyPips, Forex Factory, Reddit, QuantConnect, GitHub, MQL5, articles and forums - and adds a
strategy only when its source states the timeframe, the entry, the target and the stop (user, 30.10.2026), and only
when nothing in the library (agents/*_strategies.py) already trades the same idea. Every new one also gets a line in
data/agents/scout/inbox.json with the bots it suits; the bots' agents test it there, each on its own instruments with
its own rules, and send the manager a card. The scout itself tests nothing.
"""
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from agents.rules_agent import adx, atr, cross_down, cross_up, ema, macd, rma, rsi, sma  # noqa: F401



def _s(x):
    return pd.Series(np.asarray(x, dtype=float))


def _bar_minutes(df):
    if len(df.index) < 2:
        return 5.0
    return float(pd.Series(df.index).diff().median().total_seconds() / 60.0)


def _prev_day(df):
    """Previous UTC day's high, low, close on every bar (known at the bar)."""
    day = pd.Series(df.index.tz_convert("UTC").date)
    g = pd.DataFrame({"d": day, "h": df["h"].to_numpy(), "l": df["l"].to_numpy(), "c": df["c"].to_numpy()})
    daily = g.groupby("d").agg(h=("h", "max"), l=("l", "min"), c=("c", "last")).shift(1)
    m = g[["d"]].join(daily, on="d")
    return (m["h"].to_numpy(dtype=float), m["l"].to_numpy(dtype=float), m["c"].to_numpy(dtype=float))


def _release_bars(df):
    """Bar index of each release. With a symbol: the same FTMO high-impact calendar the EV family uses
    (agents/event_strategies.events_for) - that is what every real backtest and the bot use. Without a symbol
    (synthetic test data only): every weekday 08:30 New York, the slot of the US releases (jobs report,
    jobless claims, CPI, retail sales ...)."""
    times = []
    sym = df.attrs.get("symbol")
    if sym:
        try:
            from agents.event_strategies import events_for
            times = list(events_for(sym))
        except Exception:  # noqa: BLE001
            times = []
    if not times and len(df):
        ny = df.index.tz_convert("America/New_York")
        days = pd.Series(pd.to_datetime(ny.date)).drop_duplicates()
        fri = days[days.dt.weekday < 5]
        times = [pd.Timestamp(d.date()).tz_localize("America/New_York").replace(hour=8, minute=30)
                 .tz_convert("UTC") for d in fri]
    idx, out = df.index, []
    for t in times:
        k = int(idx.searchsorted(t))
        if 0 < k < len(idx) and idx[k] - t < pd.Timedelta(minutes=15):
            out.append(k)
    return out


# ---------------- SC001 RV 5 Min Scalping (TradingView) ----------------
def s_rv_ema20_scalp(df):
    """TradingView 'RV 5 Min Scalping Strategy': runs on the 1-minute chart. Trend = EMA200 of the 5-minute
    closes; long when the 1-minute close crosses above EMA20 and is above the 5-minute EMA200, short on the
    mirror. Stop 0.5% from entry, target 2x the stop (1%) - as published (a percentage, so no pip
    translation needed). Ours: the 5-minute EMA200 is taken from completed 5-minute bars only."""
    c = df["c"].to_numpy()
    s = df["c"]
    c5 = s.resample("5min", label="right", closed="left").last().dropna()
    e5 = pd.Series(ema(c5.to_numpy(), 200), index=c5.index)
    # a 5m bar labelled T (right edge) is complete once the 1m bar opening at T-1min has closed -> known from T
    e5_on = e5.reindex(df.index.union(e5.index)).ffill().reindex(df.index).shift(1).to_numpy()
    e20 = ema(c, 20)
    long_ = cross_up(c, e20) & (c > e5_on)
    short = cross_down(c, e20) & (c < e5_on)
    return long_, short & ~long_, {"stop": ("level", c * (1 - 0.005), c * (1 + 0.005)), "target": ("R", 2.0)}


# ---------------- SC002 Alligator + Fractals (RoboForex) ----------------
def s_alligator_fractal(df):
    """RoboForex 'Alligator + Fractals' (Bill Williams), M30/H1/H4: Alligator on the median price - jaw
    SMMA13 shifted 8, teeth SMMA8 shifted 5, lips SMMA5 shifted 3. Long: the Alligator is awake upward (lips
    > teeth > jaw), price above it, and a bar closes above the last valid up-fractal (5-bar fractal whose
    candle does not touch the Alligator lines). Stop at the last down-fractal low, target 1:3. Shorts mirror.
    Ours: a fractal counts from 2 bars after its centre (when it is confirmed)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    med = (h + l) / 2
    jaw = _s(rma(med, 13)).shift(8).to_numpy()
    teeth = _s(rma(med, 8)).shift(5).to_numpy()
    lips = _s(rma(med, 5)).shift(3).to_numpy()
    top = np.fmax(np.fmax(jaw, teeth), lips)
    bot = np.fmin(np.fmin(jaw, teeth), lips)
    hs, ls = _s(h), _s(l)
    upf = (hs == hs.rolling(5, center=True).max()).to_numpy()
    dnf = (ls == ls.rolling(5, center=True).min()).to_numpy()
    up_valid = upf & (l > top)            # buy fractal above the Alligator, not touching it
    dn_valid = dnf & (h < bot)
    # confirmed 2 bars later
    lvl_up = _s(np.where(up_valid, h, np.nan)).shift(2).ffill().to_numpy()
    lvl_dn = _s(np.where(dn_valid, l, np.nan)).shift(2).ffill().to_numpy()
    last_lo = _s(np.where(dnf, l, np.nan)).shift(2).ffill().to_numpy()
    last_hi = _s(np.where(upf, h, np.nan)).shift(2).ffill().to_numpy()
    awake_up = (lips > teeth) & (teeth > jaw) & (c > top)
    awake_dn = (lips < teeth) & (teeth < jaw) & (c < bot)
    long_ = awake_up & cross_up(c, lvl_up) & (last_lo < c)
    short = awake_dn & cross_down(c, lvl_dn) & (last_hi > c)
    return long_, short & ~long_, {"stop": ("level", last_lo, last_hi), "target": ("R", 3.0)}


# ---------------- SC003 Pavan CPR (TradingView / StockSharp) ----------------
def s_pavan_cpr(df):
    """'Pavan CPR Strategy' (TradingView, wealthcreation_trading; StockSharp port runs it on 1-minute
    candles): from the previous day P = (H+L+C)/3 and top CPR TC = (H+L)/2. Long only: the previous close
    below TC and this close above it. Stop at P, target 50 points above entry. Translation (ours): 50 points
    on the Nifty index (~24,000) = about 0.2% of price -> target 0.2% above the signal close; a signal only
    when TC is above P (otherwise the stop would sit above the entry)."""
    c = df["c"].to_numpy()
    ph, pl, pc = _prev_day(df)
    P = (ph + pl + pc) / 3
    TC = (ph + pl) / 2
    long_ = cross_up(c, TC) & (TC > P) & np.isfinite(P)
    z = np.zeros(len(c), bool)
    return long_, z, {"stop": ("level", P, P), "target": ("level", c * 1.002, c * 0.998)}


# ---------------- SC004 NFP pre-release 4-hour range straddle (BrokerXplorer) ----------------
def s_nfp_straddle(df):
    """BrokerXplorer 'Easy short-term NFP strategy' (5-minute chart, EUR/USD): 30 minutes before the
    release take the high and low of the past 4 hours; buy stop 5-10 pips above the high, sell stop 5-10
    pips below the low, the other order is the stop, target = the range (high - low) beyond the entry.
    Translations (ours): 5-10 pips ~= 1 ATR(14) of 5-minute EUR/USD before the release; the stop orders
    become 'the first close beyond the level within 2 hours after the release, filled at the next open';
    out after 3 hours. Releases: FTMO high-impact calendar for the symbol (else the NFP schedule)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    a = atr(h, l, c, 14)
    bm = _bar_minutes(df)
    pre, look, win = max(1, int(round(30 / bm))), max(2, int(round(240 / bm))), max(1, int(round(120 / bm)))
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    for k0 in _release_bars(df):
        ks = k0 - pre
        if ks - look < 15 or k0 + 1 >= n:
            continue
        hi, lo, ref = h[ks - look:ks].max(), l[ks - look:ks].min(), a[ks - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        buf = 1.0 * ref
        bl, sl_ = hi + buf, lo - buf
        rng = hi - lo
        for k in range(k0, min(n - 1, k0 + win)):
            if c[k] > bl:
                long_[k], sl_l[k], tg_l[k] = True, sl_, bl + rng
                break
            if c[k] < sl_:
                short[k], sl_s[k], tg_s[k] = True, bl, sl_ - rng
                break
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("level", tg_l, tg_s),
                          "max_bars": max(2, int(round(180 / bm)))}


# ---------------- SC005 NFP V-shaped reversal (FOREX.com) ----------------
def s_nfp_v_reversal(df):
    """FOREX.com 'Non-farm payrolls V-shaped reversal' (5- or 15-minute chart, EUR/USD, USD/JPY): do not
    trade the release; after the first move, wait for a reversal candlestick (the example: a hammer) and
    enter near its close, tight stop beyond the recent low/high, target 1:3. Translations (ours): the
    window is 10-120 minutes after the release; the first move = close vs the bar before the release; a
    reversal candle = a hammer (lower wick >= 2x body, upper wick <= 1/3 of the range), an outside
    reversal (new extreme then close beyond the previous bar's high/low) or a two-bar reversal (the previous
    bar set the extreme since the release and this bar closes beyond its high/low); stop 0.1 ATR beyond the lowest low
    (highest high) since the release; out after 3 hours."""
    o, h, l, c = (df[x].to_numpy() for x in "ohlc")
    n = len(c)
    a = atr(h, l, c, 14)
    bm = _bar_minutes(df)
    k_from, k_to = max(1, int(round(10 / bm))), max(2, int(round(120 / bm)))
    body = np.abs(c - o)
    rng = h - l
    lw = np.minimum(o, c) - l
    uw = h - np.maximum(o, c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for k0 in _release_bars(df):
        if k0 < 20 or k0 + k_from >= n - 1:
            continue
        base = c[k0 - 1]
        for k in range(k0 + k_from, min(n - 1, k0 + k_to + 1)):
            lo_, hi_ = l[k0:k + 1].min(), h[k0:k + 1].max()
            ref = a[k0 - 1]
            if c[k - 1] < base:     # first move down -> look for a bullish reversal
                hammer = (lw[k] >= 2 * body[k]) and (uw[k] <= rng[k] / 3)
                outside = l[k] < l[k - 1] and c[k] > h[k - 1]
                two_bar = l[k - 1] <= lo_ and c[k] > h[k - 1]
                if (hammer or outside or two_bar) and l[k] <= lo_ + 0.5 * ref and c[k] > lo_:
                    long_[k], sl_l[k] = True, lo_ - 0.1 * ref
                    break
            elif c[k - 1] > base:
                star = (uw[k] >= 2 * body[k]) and (lw[k] <= rng[k] / 3)
                outside = h[k] > h[k - 1] and c[k] < l[k - 1]
                two_bar = h[k - 1] >= hi_ and c[k] < l[k - 1]
                if (star or outside or two_bar) and h[k] >= hi_ - 0.5 * ref and c[k] < hi_:
                    short[k], sl_s[k] = True, hi_ + 0.1 * ref
                    break
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 3.0),
                          "max_bars": max(2, int(round(180 / bm)))}


# ---------------- SC006 Stochastic Hash Strat (TradingView, Hash Capital Research) ----------------
def s_stoch_hash(df):
    """TradingView 'Stochastic Hash Strat' (Hash Capital Research), 4H and higher (crude oil, gold, EUR/USD,
    crypto): slow stochastic length 16, smooth K 7, D 3. Long when K crosses above D below 25 (oversold),
    short when K crosses below D above 70 (overbought). Stop 2.2%, target 7% (3.18:1); exit on the opposite
    extreme (long out when K > 70, short out when K < 25). Translation (ours): the 2.2% stop is the author's
    'slightly above ATR' on crude 12H -> stop 1.2 ATR(14) of the bar's timeframe, target 3.18R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    hh, ll = _s(h).rolling(16).max().to_numpy(), _s(l).rolling(16).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100 * (c - ll) / (hh - ll)
    k = sma(np.nan_to_num(raw, nan=50.0), 7)
    d = sma(k, 3)
    long_ = cross_up(k, d) & (k < 25)
    short = cross_down(k, d) & (k > 70)
    return long_, short & ~long_, {"stop": ("atr", 1.2), "target": ("R", 3.18),
                                   "exit_long": k > 70, "exit_short": k < 25}


# ---------------- helpers added 2026-11-12 (second scout run) ----------------
def _open(df):
    """The bar's open. A feed without real opens (every open == close) falls back to the previous close."""
    o, c = df["o"].to_numpy(dtype=float), df["c"].to_numpy(dtype=float)
    if len(c) and np.allclose(o, c):
        return np.concatenate([[c[0]], c[:-1]])
    return o


def _htf(df, rule, fn):
    """fn(htf_df) -> array on completed `rule` bars, mapped back to every bar (a bar labelled T by its right
    edge is known from the bar opening at T on)."""
    g = df[["o", "h", "l", "c", "v"]].resample(rule, label="right", closed="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    if not len(g):
        return np.full(len(df), np.nan)
    vals = pd.Series(np.asarray(fn(g), dtype=float), index=g.index)
    k = g.index.searchsorted(df.index, side="right") - 1
    out = np.where(k >= 0, vals.to_numpy()[np.clip(k, 0, None)], np.nan)
    return out


def _vwap_day(df):
    """VWAP from 00:00 UTC (no volume -> equal weights)."""
    h, l, c, v = (df[x].to_numpy(dtype=float) for x in "hlcv")
    tp = (h + l + c) / 3
    v = np.where(v > 0, v, 1.0)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    return pd.Series(tp * v).groupby(day).cumsum().to_numpy() / pd.Series(v).groupby(day).cumsum().to_numpy()


def _stoch(h, l, c, n, ks, ds):
    hh, ll = _s(h).rolling(n).max().to_numpy(), _s(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100 * (c - ll) / (hh - ll)
    k = sma(np.nan_to_num(raw, nan=50.0), ks) if ks > 1 else np.nan_to_num(raw, nan=50.0)
    return k, sma(k, ds)


def _local(df, tz):
    """Start and end (start + bar length) of every bar as hour floats and weekdays in a time zone."""
    st = df.index.tz_convert(tz)
    en = (df.index + pd.Timedelta(minutes=_bar_minutes(df))).tz_convert(tz)
    hs = st.hour.to_numpy() + st.minute.to_numpy() / 60.0
    he = en.hour.to_numpy() + en.minute.to_numpy() / 60.0
    return hs, he, st.weekday.to_numpy(), en.weekday.to_numpy(), pd.Series(st.date).to_numpy()


def _swings(h, l, k):
    """Last confirmed swing high / low (k bars each side), known k bars after the pivot."""
    hs, ls = _s(h), _s(l)
    w = 2 * k + 1
    ph = (hs == hs.rolling(w, center=True).max()).to_numpy()
    pl = (ls == ls.rolling(w, center=True).min()).to_numpy()
    last_h = _s(np.where(ph, h, np.nan)).shift(k).ffill().to_numpy()
    last_l = _s(np.where(pl, l, np.nan)).shift(k).ffill().to_numpy()
    return last_h, last_l


def _supertrend_line(h, l, c, period, mult):
    a = atr(h, l, c, period)
    mid = (h + l) / 2
    up, dn = mid - mult * a, mid + mult * a
    fu, fd = up.copy(), dn.copy()
    trend = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        trend[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else trend[i - 1]
    return np.where(trend > 0, fu, fd)


# ---------------- SC007 Cowabunga (BabyPips) ----------------
def s_cowabunga(df):
    """BabyPips 'Cowabunga System' (GBP/USD): 4-hour trend, 15-minute entries. 15m: EMA5 crosses EMA10, RSI(9)
    above 50, Stochastic (10,3,3) heading up and not overbought (< 80), MACD (12,26,9) histogram turning
    positive or negative and rising; the 4-hour chart must trend the same way (ours: last completed 4H bar
    EMA5 > EMA10 and RSI(9) > 50). Stop at the most recent swing low (ours: lowest low of the last 10 bars),
    target 'the same pips as the risk' (1R; the round-number 50/00 target is not modelled). Entries 01:00 -
    15:45 New York; every trade closed at 16:00 New York. Shorts mirror."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    e5, e10 = ema(c, 5), ema(c, 10)
    r9 = rsi(c, 9)
    k, _ = _stoch(h, l, c, 10, 3, 3)
    _, _, hist = macd(c)
    hp = np.concatenate([[np.nan], hist[:-1]])
    kp = np.concatenate([[np.nan], k[:-1]])
    t_up = _htf(df, "4h", lambda g: (ema(g["c"].to_numpy(), 5) > ema(g["c"].to_numpy(), 10))
                & (rsi(g["c"].to_numpy(), 9) > 50)) > 0.5
    t_dn = _htf(df, "4h", lambda g: (ema(g["c"].to_numpy(), 5) < ema(g["c"].to_numpy(), 10))
                & (rsi(g["c"].to_numpy(), 9) < 50)) > 0.5
    hs, he, _, _, _ = _local(df, "America/New_York")
    win = (hs >= 1.0) & (he <= 15.75 + 1e-9) & (he > hs)
    m_up = ((hist > 0) & (hp <= 0)) | ((hist < 0) & (hist > hp))
    m_dn = ((hist < 0) & (hp >= 0)) | ((hist > 0) & (hist < hp))
    long_ = cross_up(e5, e10) & (r9 > 50) & (k > kp) & (k < 80) & m_up & t_up & win
    short = cross_down(e5, e10) & (r9 < 50) & (k < kp) & (k > 20) & m_dn & t_dn & win
    fe = (he >= 16.0) & (hs < 16.0)
    return long_, short & ~long_, {"stop": ("lowest", 10), "target": ("R", 1.0), "force_exit": fe}


# ---------------- SC008 EMA 5/13 + VWAP + ADX, 15m (TradingView) ----------------
def s_ema513_vwap_adx(df):
    """TradingView 'EMA Crossover Strategy [15M]' (lokeshmudavath224, XAUUSD): EMA5 crosses EMA13, longs only
    above VWAP and shorts only below, ADX above 20 (ADX length not stated -> 14). Stop at the EMA intersection
    point (ours: EMA13 at the cross bar), target 2R. The 'EMA separation' filter's minimum gap is not stated
    -> not used."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    e5, e13 = ema(c, 5), ema(c, 13)
    vw = _vwap_day(df)
    ax = adx(h, l, c, 14)
    long_ = cross_up(e5, e13) & (c > vw) & (ax > 20)
    short = cross_down(e5, e13) & (c < vw) & (ax > 20)
    return long_, short & ~long_, {"stop": ("level", e13, e13), "target": ("R", 2.0)}


# ---------------- SC009 Gold 15m pivot liquidity sweep + EMA200, 1:2 (TradingView) ----------------
def s_gold_pivot_sweep(df):
    """TradingView 'Gold 15m: Trend + S/R + Liquidity Sweep (RR 1:2)' (phanongkumsen): long when price breaks
    below the last pivot low and closes back above it, with a momentum candle, above EMA200; short mirror at
    the last pivot high below EMA200. Stop slightly beyond the pivot with an ATR buffer, target 2R. Not
    stated, ours: pivots 5 bars each side; momentum candle = body >= half the bar's range in the trade's
    direction; stop = the sweep bar's extreme -/+ 0.2 ATR(14)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    a = atr(h, l, c, 14)
    e200 = ema(c, 200)
    ph, pl = _swings(h, l, 5)
    rng = np.maximum(h - l, 1e-12)
    body = c - o
    long_ = (l < pl) & (c > pl) & (body >= 0.5 * rng) & (c > e200)
    short = (h > ph) & (c < ph) & (-body >= 0.5 * rng) & (c < e200)
    return long_, short & ~long_, {"stop": ("level", np.minimum(l, pl) - 0.2 * a, np.maximum(h, ph) + 0.2 * a),
                                   "target": ("R", 2.0)}


# ---------------- SC010 Toto Capital SBL reversal (FX Replay) ----------------
def s_toto_sbl(df):
    """FX Replay 'Toto Capital SBL reversal model', 15-minute, London and New York killzones: EMA 50/100/200
    clearly aligned (long: 50 > 100 > 200); after a sweep of sell-side liquidity (a low under the last swing
    low) a confirming candle closes up -> long; stop at the sweep level, target 2R. Shorts mirror. Ours: swing
    = 3 bars each side; the sweep within the last 3 bars; the confirming candle closes up and above the
    previous bar's high; killzones = the ICT London 02:00-05:00 and New York 07:00-10:00 New York time."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    e50, e100, e200 = ema(c, 50), ema(c, 100), ema(c, 200)
    ph, pl = _swings(h, l, 3)
    hs, _, _, _, _ = _local(df, "America/New_York")
    kz = ((hs >= 2) & (hs < 5)) | ((hs >= 7) & (hs < 10))
    ll3 = _s(l).rolling(3).min().to_numpy()
    hh3 = _s(h).rolling(3).max().to_numpy()
    pl3 = _s(pl).shift(2).to_numpy()   # the swing level as it was before the sweep window
    ph3 = _s(ph).shift(2).to_numpy()
    hp = np.concatenate([[np.nan], h[:-1]])
    lp = np.concatenate([[np.nan], l[:-1]])
    long_ = (e50 > e100) & (e100 > e200) & kz & (ll3 < pl3) & (c > pl3) & (c > o) & (c > hp)
    short = (e50 < e100) & (e100 < e200) & kz & (hh3 > ph3) & (c < ph3) & (c < o) & (c < lp)
    return long_, short & ~long_, {"stop": ("level", ll3, hh3), "target": ("R", 2.0)}


# ---------------- SC011 Casper SMC 1-minute opening-candle FVG retest (YouTube) ----------------
def s_casper_orb_fvg(df):
    """YouTube 'The 1 Minute Scalping Strategy (That Actually Works)' (Casper SMC): levels = high/low of the
    first 5-minute candle after 9:30 New York. On the 1-minute chart price moves through the high leaving a
    fair value gap (bar k low > bar k-2 high), price digs back into that gap, then an engulfing candle closes
    above the retest candle's body -> long. Stop one tick below the candles that dug into the gap (ours: their
    lowest low), target 3R. Shorts mirror through the low. Ours: one trade a day, setups until 12:00 New York."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    n = len(c)
    hs, he, _, _, day = _local(df, "America/New_York")
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for d in pd.unique(day):
        ix = np.flatnonzero(day == d)
        orb = ix[(hs[ix] >= 9.5) & (hs[ix] < 9.5 + 5 / 60 - 1e-9)]
        if not len(orb):
            continue
        hi, lo = h[orb].max(), l[orb].min()
        after = ix[(ix > orb[-1]) & (hs[ix] < 12.0)]
        state, fvg, dig = 0, None, None    # state 1 long setup, -1 short setup
        for k in after:
            if k < 2:
                continue
            if state == 0:
                if c[k] > hi and l[k] > h[k - 2]:
                    state, fvg, dig = 1, (h[k - 2], l[k]), None
                elif c[k] < lo and h[k] < l[k - 2]:
                    state, fvg, dig = -1, (h[k], l[k - 2]), None
                continue
            if state == 1:
                if dig is None:
                    if l[k] <= fvg[1]:
                        dig = l[k]
                    continue
                if c[k] > o[k] and c[k] > max(o[k - 1], c[k - 1]) and o[k] <= c[k - 1] + 1e-12:
                    long_[k], sl_l[k] = True, dig
                    break
                dig = min(dig, l[k])
                if c[k] < fvg[0]:
                    state = 0
            else:
                if dig is None:
                    if h[k] >= fvg[0]:
                        dig = h[k]
                    continue
                if c[k] < o[k] and c[k] < min(o[k - 1], c[k - 1]) and o[k] >= c[k - 1] - 1e-12:
                    short[k], sl_s[k] = True, dig
                    break
                dig = max(dig, h[k])
                if c[k] > fvg[1]:
                    state = 0
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 3.0)}


# ---------------- SC012 CPI anchored-VWAP reversion (FXNX) ----------------
def s_cpi_avwap(df):
    """FXNX 'CPI Volatility: AVWAP Reversion Strategy' (5- or 15-minute, FX): anchor a VWAP on the release
    candle, wait 5-10 minutes, after the spike and its exhaustion wait for price to come back to the AVWAP and
    enter on a confirmation candle there - after an up spike a bearish pin bar / engulfing -> short, after a
    down spike a hammer / bullish engulfing -> long. Stop just beyond the spike extreme, target 1:2 (one of
    the page's three targets). Ours: spike = the larger excursion from the pre-release close, at least 1
    ATR(14); the candle must touch the AVWAP; stop 0.1 ATR beyond the extreme; setups 10-120 minutes after the
    release, out after 3 hours. Releases: FTMO high-impact calendar (else 08:30 New York)."""
    h, l, c, v = (df[x].to_numpy(dtype=float) for x in "hlcv")
    o = _open(df)
    n = len(c)
    a = atr(h, l, c, 14)
    bm = _bar_minutes(df)
    tp = (h + l + c) / 3
    vv = np.where(v > 0, v, 1.0)
    k_from, k_to = max(1, int(round(10 / bm))), max(2, int(round(120 / bm)))
    body = np.abs(c - o)
    lw, uw = np.minimum(o, c) - l, h - np.maximum(o, c)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for k0 in _release_bars(df):
        if k0 < 20 or k0 + k_from >= n - 1:
            continue
        base, ref = c[k0 - 1], a[k0 - 1]
        if not np.isfinite(ref) or ref <= 0:
            continue
        cpv, cv = np.cumsum(tp[k0:] * vv[k0:]), np.cumsum(vv[k0:])
        for k in range(k0 + k_from, min(n - 1, k0 + k_to + 1)):
            av = cpv[k - k0] / cv[k - k0]
            hi_, lo_ = h[k0:k + 1].max(), l[k0:k + 1].min()
            up, dn = hi_ - base, base - lo_
            if max(up, dn) < ref or not (l[k] <= av <= h[k]):
                continue
            if up > dn:
                pin = uw[k] >= 2 * body[k] and c[k] <= o[k]
                eng = c[k] < o[k] and c[k] < min(o[k - 1], c[k - 1]) and o[k] >= max(o[k - 1], c[k - 1]) - 1e-12
                if pin or eng:
                    short[k], sl_s[k] = True, hi_ + 0.1 * ref
                    break
            else:
                pin = lw[k] >= 2 * body[k] and c[k] >= o[k]
                eng = c[k] > o[k] and c[k] > max(o[k - 1], c[k - 1]) and o[k] <= min(o[k - 1], c[k - 1]) + 1e-12
                if pin or eng:
                    long_[k], sl_l[k] = True, lo_ - 0.1 * ref
                    break
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0),
                          "max_bars": max(2, int(round(180 / bm)))}


# ---------------- SC013 4H Doji sandwich (TradingView) ----------------
def s_doji_sandwich(df):
    """TradingView idea 'GBPJPY 4H Doji Sandwich strategy': a large candle, a doji, then a third large candle
    in the first one's direction -> enter at the 4th candle's open in that direction. Stop below (above) the
    doji. Exit when the Stochastic (default 14,3,3) is overbought (long, > 80) / oversold (short, < 20); no
    fixed target. Ours: large = body >= 0.6 ATR(14); doji = body <= 10% of its range; the trailing under the
    50 SMA is not modelled."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    a = atr(h, l, c, 14)
    body = c - o
    rng = np.maximum(h - l, 1e-12)
    big_up = body >= 0.6 * a
    big_dn = -body >= 0.6 * a
    doji = np.abs(body) <= 0.1 * rng
    sh = lambda x, k: np.concatenate([np.zeros(k, bool), x[:-k]])  # noqa: E731
    long_ = sh(big_up, 2) & sh(doji, 1) & big_up
    short = sh(big_dn, 2) & sh(doji, 1) & big_dn
    lp = np.concatenate([[np.nan], l[:-1]])
    hp = np.concatenate([[np.nan], h[:-1]])
    k, _ = _stoch(h, l, c, 14, 3, 3)
    return long_, short & ~long_, {"stop": ("level", lp, hp), "target": None,
                                   "exit_long": k > 80, "exit_short": k < 20}


# ---------------- SC014 VWAP + MACD 1-minute (FXOpen) ----------------
def s_vwap_macd_1m(df):
    """FXOpen '1-minute scalping strategies' #1 VWAP + MACD: price closes through the VWAP while MACD
    (12,26,9) flips (MACD line crosses its signal) in the same direction - usually within 4-5 candles -> enter.
    Stop just beyond a recent swing (ours: extreme of the last 5 bars). Take profit when the MACD crosses back
    the other way (no fixed target)."""
    c = df["c"].to_numpy()
    vw = _vwap_day(df)
    m, sg, _ = macd(c)
    up5 = _s(cross_up(m, sg).astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    dn5 = _s(cross_down(m, sg).astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    long_ = cross_up(c, vw) & up5 & (m > sg)
    short = cross_down(c, vw) & dn5 & (m < sg)
    return long_, short & ~long_, {"stop": ("lowest", 5), "target": None,
                                   "exit_long": cross_down(m, sg), "exit_short": cross_up(m, sg)}


# ---------------- SC015 ALMA + Stochastic 1-minute (FXOpen) ----------------
def _alma(x, n=21, offset=0.85, sigma=6.0):
    x = np.asarray(x, dtype=float)
    m, s = offset * (n - 1), n / sigma
    w = np.exp(-((np.arange(n) - m) ** 2) / (2 * s * s))
    w /= w.sum()
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.lib.stride_tricks.sliding_window_view(x, n) @ w
    return out


def s_alma_stoch_1m(df):
    """FXOpen '1-minute scalping strategies' #3 ALMA + Stochastic: ALMA (21, offset 0.85, sigma 6),
    Stochastic (21,1,3). Long: close above the ALMA with the Stochastic crossing above 20; short: close below
    the ALMA with the Stochastic crossing below 80. Stop beyond the nearest swing (ours: extreme of the last 5
    bars). Take profit when the Stochastic reaches the opposite zone (long > 80, short < 20)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    al = _alma(c, 21, 0.85, 6.0)
    k, _ = _stoch(h, l, c, 21, 1, 3)
    long_ = (c > al) & cross_up(k, 20)
    short = (c < al) & cross_down(k, 80)
    return long_, short & ~long_, {"stop": ("lowest", 5), "target": None,
                                   "exit_long": k > 80, "exit_short": k < 20}


# ---------------- SC016 RSI(4) 80/20 + Bollinger, target the opposite band (FXOpen) ----------------
def s_rsi4_bb_1m(df):
    """FXOpen '1-minute scalping strategies' #4 RSI + Bollinger Bands: RSI(4) crosses back above 20 with price
    touching / breaching the lower band (20, 2) -> long; RSI(4) back below 80 at the upper band -> short. Stop
    just beyond the swing / outside the band (ours: extreme of the last 3 bars). Target the opposite band (as
    it stood at the signal); also out when RSI(4) reaches the opposite zone."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    r4 = rsi(c, 4)
    mid = sma(c, 20)
    sd = _s(c).rolling(20).std(ddof=0).to_numpy()
    up, lo = mid + 2 * sd, mid - 2 * sd
    lp = np.concatenate([[np.nan], l[:-1]])
    hp = np.concatenate([[np.nan], h[:-1]])
    long_ = cross_up(r4, 20) & (np.fmin(l, lp) <= lo)
    short = cross_down(r4, 80) & (np.fmax(h, hp) >= up)
    return long_, short & ~long_, {"stop": ("lowest", 3), "target": ("level", up, lo),
                                   "exit_long": r4 > 80, "exit_short": r4 < 20}


# ---------------- SC017 freqtrade ScalpingSimple, 1m (GitHub) ----------------
def s_ft_scalping_simple(df):
    """freqtrade 'ScalpingSimple' (strategy listing fts.naif.one), 1-minute, long only: RSI(14) < 30, close
    above SMA5, the bar up, volume > 0. Exit on RSI > 70, SMA5 crossing under SMA10, or a down bar under SMA5.
    Stoploss -0.5%. ROI table 1% at once, then 0.5% / 0.3% / 0.2% / 0.1% after 5 / 10 / 15 / 30 minutes ->
    ours: target 1%, out after 30 minutes (the falling ROI steps and the 0.2%/0.3% trailing are not modelled)."""
    c, v = df["c"].to_numpy(), df["v"].to_numpy()
    r = rsi(c, 14)
    s5, s10 = sma(c, 5), sma(c, 10)
    cp = np.concatenate([[np.nan], c[:-1]])
    long_ = (r < 30) & (c > s5) & (c > cp) & (v > 0)
    ex = (r > 70) | cross_down(s5, s10) | ((c < s5) & (c < cp))
    bm = _bar_minutes(df)
    return long_, np.zeros(len(c), bool), {"stop": ("level", c * 0.995, c * 1.005),
                                           "target": ("level", c * 1.01, c * 0.99),
                                           "exit_long": ex, "max_bars": max(1, int(round(30 / bm)))}


# ---------------- SC018 freqtrade MeanReversionScalper, 5m (GitHub) ----------------
def s_ft_mean_rev_scalper(df):
    """freqtrade 'MeanReversionScalper' (fts.naif.one), 5-minute, long only: close at/under the lower
    Bollinger band (20, 2), RSI(14) < 30, volume > 1.5x its 20-bar average, band width (upper-lower)/mid >
    0.005, and the 1-hour EMA50 not falling more than 0.3% over the last hour. Exit at the middle band.
    Stoploss -0.8%, first ROI 0.6% -> target 0.6%. Ours: without volume data (constant volume) the volume
    filter is off; the time-tightening custom stop and the trailing are not modelled."""
    c, v = df["c"].to_numpy(dtype=float), df["v"].to_numpy(dtype=float)
    mid = sma(c, 20)
    sd = _s(c).rolling(20).std(ddof=0).to_numpy()
    lo, up = mid - 2 * sd, mid + 2 * sd
    r = rsi(c, 14)
    vs = sma(v, 20)
    has_vol = np.nanstd(v) > 0
    vol_ok = (v > 1.5 * vs) if has_vol else np.ones(len(c), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        width = (up - lo) / mid
    e50 = _htf(df, "1h", lambda g: pd.Series(ema(g["c"].to_numpy(), 50)).pct_change().to_numpy())
    long_ = (c <= lo) & (r < 30) & vol_ok & (width > 0.005) & (e50 > -0.003)
    return long_, np.zeros(len(c), bool), {"stop": ("level", c * 0.992, c * 1.008),
                                           "target": ("level", c * 1.006, c * 0.994), "exit_long": c >= mid}


# ---------------- SC019 freqtrade RSI_BB futures, 15m (GitHub) ----------------
def s_ft_rsi_bb(df):
    """freqtrade 'RSI_BB' (pneumann1980, fts.naif.one), 15-minute futures, Bollinger on the typical price
    (20): long when the close is under the 1-SD lower band; short when RSI(14) > 56 and the close is over the
    3-SD upper band. Exit long on RSI > 56 with a close over the 3-SD upper band; exit short on a close under
    the 1-SD lower band. Stoploss -10% (a percentage, kept); the ROI table starts at 85% -> no fixed target."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    tp = (h + l + c) / 3
    mid = sma(tp, 20)
    sd = _s(tp).rolling(20).std(ddof=0).to_numpy()
    lo1, up3 = mid - sd, mid + 3 * sd
    r = rsi(c, 14)
    long_ = c < lo1
    short = (r > 56) & (c > up3)
    return long_, short & ~long_, {"stop": ("level", c * 0.90, c * 1.10), "target": None,
                                   "exit_long": short, "exit_short": long_}


# ---------------- SC020 DAX 15min false breakout + SuperTrend (ProRealCode) ----------------
def s_prc_dax_false_break(df):
    """ProRealCode 'DAX 15Min - False Breakout / SuperTrend' (DESCOUMA), long only: the previous bar made a new
    20-bar low (lowest 20 fell), this bar stays under the previous 20-bar high, closes up, its range is under
    half of the 20-bar range, and the close is above the SuperTrend (multiplier 1.5, period 18) -> buy at
    market. Stop 80 points, target 150 points. Translation (ours): DAX 15-minute ATR(14) was ~18 points when
    published (2016, DAX ~10,000) -> stop 4.5 ATR, target 8.4 ATR (same 1.875 ratio)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    a = atr(h, l, c, 14)
    amax, amin = _s(h).rolling(20).max().to_numpy(), _s(l).rolling(20).min().to_numpy()
    sh = lambda x, k: np.concatenate([np.full(k, np.nan), x[:-k]])  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        rng_ok = (h - l) / (amax - amin) < 0.5
    st = _supertrend_line(h, l, c, 18, 1.5)
    long_ = (sh(amin, 1) < sh(amin, 2)) & (h < sh(amax, 1)) & (c > o) & rng_ok & (c > st)
    return long_, np.zeros(len(c), bool), {"stop": ("level", c - 4.5 * a, c + 4.5 * a),
                                           "target": ("level", c + 8.4 * a, c - 8.4 * a)}


# ---------------- SC021 'Jump' 5-minute breakout (ProRealCode forum, NASDAQ) ----------------
def s_prc_jump(df):
    """ProRealCode forum 'Jump - a 5 min strategy' (NASDAQ): base = midpoint of the 16-bar high/low, offset =
    2 ATR(15). Long: close above the close 24 bars ago, 60-minute close above its 300-bar average, 30-minute
    ADX(14) >= 20, not Friday -> buy stop at base + offset. Short: close under the close 24 bars ago, 60-minute
    close under its average, 30-minute ADX <= 20, not Tuesday, std(3 closes)/close < 0.29% -> sell stop at
    base - offset. Stop 8 ATR(20), target 7 ATR(20). Ours: the stop order = the first close through the level;
    the extra 110-unit stop, the ATR trailing and the breakeven are not modelled."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    n = len(c)
    base = (_s(h).rolling(16).max().to_numpy() + _s(l).rolling(16).min().to_numpy()) / 2
    off = 2 * atr(h, l, c, 15)
    a20 = atr(h, l, c, 20)
    c24 = _s(c).shift(24).to_numpy()
    h60 = _htf(df, "60min", lambda g: g["c"].to_numpy() - sma(g["c"].to_numpy(), 300))
    ax30 = _htf(df, "30min", lambda g: adx(g["h"].to_numpy(), g["l"].to_numpy(), g["c"].to_numpy(), 14))
    wd = df.index.tz_convert("UTC").weekday.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = _s(c).rolling(3).std().to_numpy() / c * 100
    up_l, dn_l = base + off, base - off
    long_ = cross_up(c, up_l) & (c > c24) & (h60 > 0) & (ax30 >= 20) & (wd != 4)
    short = cross_down(c, dn_l) & (c < c24) & (h60 < 0) & (ax30 <= 20) & (wd != 1) & (vol < 0.29)
    return long_, short & ~long_, {"stop": ("level", c - 8 * a20, c + 8 * a20),
                                   "target": ("level", c + 7 * a20, c - 7 * a20)}


# ---------------- helpers added 2026-11-12 (third scout run) ----------------
def _sh(x, k=1):
    x = np.asarray(x, dtype=float)
    return np.concatenate([np.full(k, np.nan), x[:-k]]) if k > 0 else x


def _wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    return _s(x).rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True).to_numpy()


def _hma(x, n):
    return _wma(2 * _wma(x, n // 2) - _wma(x, n), int(np.sqrt(n)))


def _bb(c, n=20, k=2.0):
    m = sma(c, n)
    sd = _s(c).rolling(n).std(ddof=0).to_numpy()
    return m - k * sd, m, m + k * sd


def _cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    m = sma(tp, n)
    md = _s(tp).rolling(n).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (tp - m) / (0.015 * md)


def _di(h, l, c, n=14):
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(h, l, c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * rma(pdm, n) / a, 100 * rma(ndm, n) / a


def _linreg_end(x, n):
    """Value of the n-bar least-squares line at its last bar (Pine linreg(x, n, 0))."""
    t = np.arange(n, dtype=float)
    tm = t.mean()
    den = ((t - tm) ** 2).sum()

    def f(v):
        b = ((t - tm) * (v - v.mean())).sum() / den
        return v.mean() + b * (n - 1 - tm)
    return _s(x).rolling(n).apply(f, raw=True).to_numpy()


def _engulf(o, c):
    po, pc = _sh(o), _sh(c)
    bull = (c > o) & (pc < po) & (c >= po) & (o <= pc)
    bear = (c < o) & (pc > po) & (c <= po) & (o >= pc)
    return bull, bear


# ---------------- SC022 Simple 5 EMA high/low scalp (BabyPips forum) ----------------
def s_bp_5ema_hl_scalp(df):
    """BabyPips forum 'Simple 5 EMA High/Low Scalping Strategy', 5-minute, any pair (EUR/USD preferred): a 5 EMA
    of the highs and a 5 EMA of the lows. A candle that opens below the 5 EMA (low) -> buy, one that opens above the
    5 EMA high -> sell. Stop 8 pips, target 5 pips; trade the hour after the London open and the hour after the
    New York open. Ours: the next candle opens where this one closes, so the signal is a close crossing below the
    EMA-low (above the EMA-high). Translation: EUR/USD 5-minute ATR(14) ~4.5 pips -> stop 1.8 ATR, target 1.1 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    eh, el = ema(h, 5), ema(l, 5)
    lon, _, _, _, _ = _local(df, "Europe/London")
    ny, _, _, _, _ = _local(df, "America/New_York")
    win = ((lon >= 8) & (lon < 9)) | ((ny >= 8) & (ny < 9))
    long_ = cross_down(c, el) & win
    short = cross_up(c, eh) & win
    return long_, short & ~long_, {"stop": ("atr", 1.8), "target": ("atr", 1.1)}


# ---------------- SC023 Keltner Scalping Trading System, 1-minute (BabyPips forum) ----------------
def s_bp_keltner_scalp_1m(df):
    """BabyPips forum 'Keltner Scalping Trading System', EUR/USD 1-minute: SMA89, Keltner channel around a 22-period
    average, MACD(5,34,5), EMA445, Stochastic Momentum (5,3). Long: price above / testing the SMA89, price inside
    the Keltner channel, MACD(5,34) below zero, stochastic momentum near/below -40, and the MACD line turning up
    through its 5-period average -> buy the next open (short mirror). Stop 1 pip outside the channel, at most 12
    pips; target 6-12 pips. Ours: Keltner = EMA22 +/- 1.5 ATR(22) (the multiplier is not stated); the stochastic
    momentum 'approaching or below -40' = full stochastic (5,3,3) under 35 in the last 3 bars (over 65 for shorts); translation from EUR/USD 1-minute
    ATR(14) ~1.3 pips: 1 pip = 0.75 ATR, 12-pip cap = 9 ATR, target 9 pips (the middle of 6-12) = 7 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    s89 = sma(c, 89)
    mid = ema(c, 22)
    ka = atr(h, l, c, 22)
    ku, kl = mid + 1.5 * ka, mid - 1.5 * ka
    m, sig, _ = macd(c, 5, 34, 5)
    sig5 = sma(m, 5)
    k, _ = _stoch(h, l, c, 5, 3, 3)
    k_lo = _s(k).rolling(3).min().to_numpy() < 35    # 'approaching or below' the zone in the last 3 bars
    k_hi = _s(k).rolling(3).max().to_numpy() > 65
    long_ = ((c > s89) | (l <= s89)) & (c >= kl) & (c <= ku) & (m < 0) & k_lo & cross_up(m, sig5)
    short = ((c < s89) | (h >= s89)) & (c >= kl) & (c <= ku) & (m > 0) & k_hi & cross_down(m, sig5)
    stop_l = np.maximum(kl - 0.75 * a, c - 9 * a)
    stop_s = np.minimum(ku + 0.75 * a, c + 9 * a)
    return long_, short & ~long_, {"stop": ("level", stop_l, stop_s), "target": ("atr", 7.0)}


# ---------------- SC024 EURUSD 5 min TEMA / CCI (ProRealCode forum) ----------------
def s_prc_eurusd_tema_cci(df):
    """ProRealCode forum 'EURUSD 5 min trading strategy' (victormork): long when TEMA(21) > average(300),
    Bollinger band width(50) > 0.003 and CCI(20) crosses over -110; short when TEMA(21) < average(300), band width
    > 0.003 and CCI crosses under 160. Only 07:00-22:00, no weekend. Stop 25 pips; exits: long out when CCI crosses
    over 210, short out when CCI crosses under -210 (no fixed target). Ours: platform time = Paris; band width =
    (upper - lower) / middle of BB(50, 2); translation: EUR/USD 5-minute ATR(14) ~4.5 pips -> 25 pips = 5.5 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    e1 = ema(c, 21)
    e2 = ema(e1, 21)
    tema = 3 * e1 - 3 * e2 + ema(e2, 21)
    s300 = sma(c, 300)
    lo, mi, up = _bb(c, 50, 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = (up - lo) / mi
    cci = _cci(h, l, c, 20)
    hr, _, wd, _, _ = _local(df, "Europe/Paris")
    t_ok = (hr >= 7) & (hr < 22) & (wd < 5)
    long_ = (tema > s300) & (bw > 0.003) & cross_up(cci, np.full(len(c), -110.0)) & t_ok
    short = (tema < s300) & (bw > 0.003) & cross_down(cci, np.full(len(c), 160.0)) & t_ok
    return long_, short & ~long_, {"stop": ("atr", 5.5), "target": None,
                                   "exit_long": cross_up(cci, np.full(len(c), 210.0)),
                                   "exit_short": cross_down(cci, np.full(len(c), -210.0))}


# ---------------- SC025 Bollinger + RSI + Stochastic scalp, 5-minute (Forex Tester) ----------------
def s_fxt_bb_rsi_stoch(df):
    """Forex Tester blog 'Bollinger Bands RSI Stochastic scalping strategy', 5-minute, major pairs: BB(20, 2),
    RSI(14), Stochastic(5,3,3). Long at the close of the candle where price is below the lower band, RSI < 30 and
    stochastic < 20 (short: above the upper band, RSI > 70, stochastic > 80). Stop 10 pips beyond the piercing
    candle's extreme, target 20 pips; also out on the first close beyond the middle band. Translation: EUR/USD
    5-minute ATR(14) ~4.5 pips -> stop = candle low - 2.2 ATR, target 4.4 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    lo, mi, up = _bb(c, 20, 2.0)
    r = rsi(c, 14)
    k, _ = _stoch(h, l, c, 5, 3, 3)
    long_ = (l < lo) & (r < 30) & (k < 20)
    short = (h > up) & (r > 70) & (k > 80)
    return long_, short & ~long_, {"stop": ("level", l - 2.2 * a, h + 2.2 * a),
                                   "target": ("level", c + 4.4 * a, c - 4.4 * a),
                                   "exit_long": c > mi, "exit_short": c < mi}


# ---------------- SC026 Admiral Keltner scalping, M15 (Admirals blog) ----------------
def s_admiral_keltner_m15(df):
    """Admirals blog 'Scalping with the Admiral Keltner indicator', M15 only (EUR/USD, GBP/USD, USD/CHF,
    USD/JPY): two Keltner channels (50, 10, 2.75) and (50, 10, 3.75) and Stochastic (14,7,3). Long: a candle goes
    below either lower band and closes back inside the channel while the stochastic crosses upward far from 80
    (ideally near 20). Short mirror. Stop 8 pips + spread beyond the setup candle; target the middle line, at least
    10 pips. Ours: middle = EMA50, bands = middle +/- 2.75 ATR(10) (the inner band is the one price has to pierce);
    'far from 80' = %K under 50 (over 50 for shorts). Translation: EUR/USD 15-minute ATR(14) ~7 pips -> stop =
    candle low - 1.1 ATR, minimum target 1.4 ATR. The move of the stop to breakeven at the middle line is not
    modelled (the middle line is the target)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    mid = ema(c, 50)
    a10 = atr(h, l, c, 10)
    up, lo = mid + 2.75 * a10, mid - 2.75 * a10
    k, d = _stoch(h, l, c, 14, 7, 3)
    long_ = (l < lo) & (c > lo) & (c < mid) & cross_up(k, d) & (k < 50)
    short = (h > up) & (c < up) & (c > mid) & cross_down(k, d) & (k > 50)
    tl = np.maximum(mid, c + 1.4 * a)
    ts = np.minimum(mid, c - 1.4 * a)
    return long_, short & ~long_, {"stop": ("level", l - 1.1 * a, h + 1.1 * a), "target": ("level", tl, ts)}


# ---------------- SC027 Meta Scalper / simple range scalper (forexop) ----------------
def s_forexop_range_scalper(df):
    """forexop 'Meta Scalper - a simple low risk scalping strategy', M1/M5, any market, best in ranges: Bollinger
    bands set to hold ~95% of the highs/lows. Long when a bar touches the lower band and the next bar starts to
    retrace back toward the centre (short mirror at the upper band). Stop no more than half the band width; target
    about 5% of the band width. Ours: BB(20, 2) (~95% of a normal spread; the source gives no period); 'retrace' =
    an up close back above the lower band right after the touching bar. Already in band-width units, so no pip
    translation."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    lo, mi, up = _bb(c, 20, 2.0)
    w = up - lo
    long_ = (_sh(l) <= _sh(lo)) & (c > lo) & (c > o)
    short = (_sh(h) >= _sh(up)) & (c < up) & (c < o)
    return long_, short & ~long_, {"stop": ("level", c - w / 2, c + w / 2),
                                   "target": ("level", c + 0.05 * w, c - 0.05 * w)}


# ---------------- SC028 Intraday DAX first 5-minute bar (ProRealCode) ----------------
def s_prc_dax_first_bar(df):
    """ProRealCode 'Intraday DAX strategy mini-1EUR, timeframe 5 min' (Raul Vg): at 09:05 Frankfurt, if the first
    5-minute bar of the session (09:00-09:05) closed more than 1 point above its open -> buy, otherwise sell.
    Stop 5 points, target 10 points, flat at 17:30. Ours: the signal is that bar's close; the order-size
    martingale is not modelled. Translation: DAX 5-minute ATR(14) ~8 points when published (2017, DAX ~11,500) ->
    1 point = 0.12 ATR, stop 0.6 ATR, target 1.25 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    a = atr(h, l, c, 14)
    hs, he, wd, _, _ = _local(df, "Europe/Berlin")
    first = (np.abs(hs - 9.0) < 1e-6) & (wd < 5)
    up = c > o + 0.12 * a
    long_ = first & up
    short = first & ~up
    return long_, short & ~long_, {"stop": ("atr", 0.6), "target": ("atr", 1.25), "force_exit": hs >= 17.5}


# ---------------- SC029 Hull Suite + DI cross, 1-minute (Axiory) ----------------
def s_axiory_hull_di_1m(df):
    """Axiory 'One Minute Strategies' #1, 1-minute: Hull Suite (default HMA 55) and the +DI/-DI lines of ADX.
    Buy when +DI crosses above -DI while the Hull Suite is green (sell on the mirror). Stop slightly below the
    recent swing low (above the swing high), target 1:1 (1-1.5x the stop). Ours: green = HMA(55) above its value
    2 bars ago (the Hull Suite colouring); swing = last confirmed 2-bar swing, else the 10-bar extreme; 'slightly'
    = 0.1 ATR; target 1R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    hm = _hma(c, 55)
    green = hm > _sh(hm, 2)
    red = hm < _sh(hm, 2)
    pdi, ndi = _di(h, l, c, 14)
    sh_, sl_ = _swings(h, l, 2)
    ll, hh = _s(l).rolling(10).min().to_numpy(), _s(h).rolling(10).max().to_numpy()
    st_l = np.where(sl_ < c, sl_, ll) - 0.1 * a
    st_s = np.where(sh_ > c, sh_, hh) + 0.1 * a
    long_ = cross_up(pdi, ndi) & green
    short = cross_down(pdi, ndi) & red
    return long_, short & ~long_, {"stop": ("level", st_l, st_s), "target": ("R", 1.0)}


# ---------------- SC030 Alpha S/R Channel, gold 5m with a 15m EMA channel (TradingView) ----------------
def s_alpha_sr_channel(df):
    """TradingView 'Alpha S/R Channel Strategy' (Awab_Hassan, XAUUSD): a channel around the higher-timeframe EMA
    (15m EMA36, width 0.35% of price), entries on 5m. Long: the candle opens inside the channel and is a bullish
    engulfing (body 0.098%-0.55% of price), price above the 5m EMA36 and the 15m EMA, RSI(14) not above 70,
    Bollinger width not below 0.002, squeeze momentum value below 0. Short mirror. Stop at the channel band minus a
    buffer (500 ticks = $0.50 on gold), target 1:3; no entries 16:45-19:05 New York, out at 17:00 New York, no
    Friday. Ours: body-only engulfing mode; buffer = 0.02% of price (gold ~$2,500); squeeze = LazyBear linreg
    momentum (20); the pin-bar combo, the 2nd trade and the trade counter are not modelled."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    he_ = _htf(df, "15min", lambda g: ema(g["c"].to_numpy(), 36))
    e_ltf = ema(c, 36)
    up_ch, lo_ch = he_ * (1 + 0.00175), he_ * (1 - 0.00175)
    inside = (o >= lo_ch) & (o <= up_ch)
    bull, bear = _engulf(o, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        body = np.abs(c - o) / c * 100
        lo, mi, up = _bb(c, 20, 2.0)
        bw = (up - lo) / mi
    size_ok = (body >= 0.098) & (body <= 0.55)
    r = rsi(c, 14)
    hh, ll = _s(h).rolling(20).max().to_numpy(), _s(l).rolling(20).min().to_numpy()
    mom = _linreg_end(c - ((hh + ll) / 2 + sma(c, 20)) / 2, 20)
    hs, _, wd, _, _ = _local(df, "America/New_York")
    t_ok = ~((hs >= 16.75) & (hs < 19.0833)) & (wd != 4)
    long_ = inside & bull & size_ok & (c > e_ltf) & (c > he_) & (r <= 70) & (bw >= 0.002) & (mom < 0) & t_ok
    short = inside & bear & size_ok & (c < e_ltf) & (c < he_) & (r >= 30) & (bw >= 0.002) & (mom > 0) & t_ok
    return long_, short & ~long_, {"stop": ("level", lo_ch * (1 - 0.0002), up_ch * (1 + 0.0002)),
                                   "target": ("R", 3.0), "force_exit": (hs >= 17) & (hs < 18)}


# ---------------- SC031 freqtrade EMA_RSIStrategy, 5m long (GitHub) ----------------
def s_ft_ema_rsi(df):
    """GitHub freqtrade 'EMA_RSIStrategy' (gksdud6330, crypto, 5m with a 1h trend): long when 1h EMA50 >= 0.995 x
    1h EMA200, EMA20 > EMA50, close > EMA50, 40 < RSI(14) < 70, ADX(14) > 18 and volume > 0.8 x its 20-bar mean.
    Exit when the 1h trend fails, EMA20 < EMA50, RSI < 35 or close < EMA50. Stoploss -7%, ROI 4.5% (3% after 60
    min, 1.5% after 180 min, 0 after 360 min). Ours: target 4.5%, time exit after 360 minutes; the falling ROI
    steps and the 1.2%/3.5% trailing stop are not modelled. Percentages, so no pip translation."""
    h, l, c, v = (df[x].to_numpy() for x in "hlcv")
    e20, e50 = ema(c, 20), ema(c, 50)
    tr1h = _htf(df, "60min", lambda g: (ema(g["c"].to_numpy(), 50) >= 0.995 * ema(g["c"].to_numpy(), 200))
                .astype(float)) > 0.5
    r = rsi(c, 14)
    ax = adx(h, l, c, 14)
    vm = sma(v, 20)
    long_ = tr1h & (e20 > e50) & (c > e50) & (r > 40) & (r < 70) & (ax > 18) & (v > 0.8 * vm)
    ext = ~tr1h | (e20 < e50) | (r < 35) | (c < e50)
    mb = max(1, int(round(360 / max(_bar_minutes(df), 1.0))))
    return long_, np.zeros(len(c), bool), {"stop": ("level", c * 0.93, c * 1.07),
                                           "target": ("level", c * 1.045, c * 0.955),
                                           "max_bars": mb, "exit_long": ext}


# ---------------- SC032 London-open central pivot, 15-minute (TradingView) ----------------
def s_tv_london_pivot_15m(df):
    """TradingView idea 'EURUSD 15M Pivot Point day trading strategy': trade only the London open, 08:00 GMT.
    After the first 15 minutes, buy if price is above the daily central pivot (sell if below). Stop 5-10 pips
    beyond the pivot; TP1 at the first level, TP2 at the second (R1/R2 for buys, S1/S2 for sells); trail the stop
    to the pivot at R2. Ours: signal at the close of the bar ending 08:15 UTC; pivots of the previous UTC day;
    target R2 / S2 (TP2); the 'too close to S1' skip has no number and is left out. Translation: EUR/USD
    15-minute ATR(14) ~7.5 pips -> stop = pivot -/+ 1 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    ph, pl, pc = _prev_day(df)
    p = (ph + pl + pc) / 3
    r2, s2 = p + (ph - pl), p - (ph - pl)
    _, he, _, _, _ = _local(df, "UTC")
    at = np.abs(he - 8.25) < 1e-6
    long_ = at & (c > p) & (r2 > c)
    short = at & (c < p) & (s2 < c)
    return long_, short & ~long_, {"stop": ("level", p - a, p + a), "target": ("level", r2, s2)}


# ---------------- SC033 Crypto 1H EMA50 trend + EMA21 pullback, 24-bar target (TradingView) ----------------
def s_tv_crypto_1h_strict_rr(df):
    """TradingView 'Crypto 1H [Universal - Strict + R:R]' (ObiWan-Satoshi), 1-hour: trend = price above a rising
    EMA50; after price touches EMA21 the script waits at most 3 bars for a valid signal candle (body >= 0.15% of
    price). Stop at the lowest low of the last 5 bars minus a small ATR buffer; target the highest high of the last
    24 bars; the trade is skipped unless the target is at least 2x the stop. London and New York sessions only.
    Shorts mirror. Ours: signal candle = a close back above EMA21 in the trade direction; buffer = 0.2 ATR(14);
    sessions = 07:00-21:00 UTC."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    a = atr(h, l, c, 14)
    e50, e21 = ema(c, 50), ema(c, 21)
    touch_l = _s((l <= e21).astype(float)).rolling(4).max().to_numpy() > 0
    touch_s = _s((h >= e21).astype(float)).rolling(4).max().to_numpy() > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        body = np.abs(c - o) / c * 100
    stl = _s(l).rolling(5).min().to_numpy() - 0.2 * a
    sts = _s(h).rolling(5).max().to_numpy() + 0.2 * a
    tgl = _sh(_s(h).rolling(24).max().to_numpy())
    tgs = _sh(_s(l).rolling(24).min().to_numpy())
    hs, _, _, _, _ = _local(df, "UTC")
    sess = (hs >= 7) & (hs < 21)
    long_ = (c > e50) & (e50 > _sh(e50)) & touch_l & (c > o) & (c > e21) & (body >= 0.15) & \
        (tgl - c >= 2 * (c - stl)) & sess
    short = (c < e50) & (e50 < _sh(e50)) & touch_s & (c < o) & (c < e21) & (body >= 0.15) & \
        (c - tgs >= 2 * (sts - c)) & sess
    return long_, short & ~long_, {"stop": ("level", stl, sts), "target": ("level", tgl, tgs)}


# ---------------- SC034 ATR Trend + RSI pullback, long (TradingView) ----------------
def s_tv_atr_trend_rsi_pullback(df):
    """TradingView 'ATR Trend + RSI Pullback Strategy [Profit-Focused]', long only, tested on 15m/30m/1H (BTCUSD,
    NAS100, ETHUSD): price above a rising EMA200, RSI dips below 40 and ATR is above a minimum threshold -> buy.
    Stop 1.5 ATR, target 3 ATR; optional exit when RSI > 70. Ours: RSI(14) crossing under 40 (the first bar of the
    dip), ATR(14) above half of its 100-bar average (the threshold value is not stated)."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    e200 = ema(c, 200)
    r = rsi(c, 14)
    long_ = (c > e200) & (e200 > _sh(e200)) & cross_down(r, np.full(len(c), 40.0)) & (a > 0.5 * sma(a, 100))
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 1.5), "target": ("atr", 3.0), "exit_long": r > 70}


# ---------------- SC035 EMA 50/100 + Stochastic, 1-minute (The5ers) ----------------
def s_5ers_ema50_100_stoch_1m(df):
    """The5ers blog '1 Minute Scalping Strategy' (EUR/USD, GBP/USD): long when the 50 EMA has crossed above the
    100 EMA, price pulls back toward the EMAs and the stochastic rises above 20; short when the 50 EMA is below
    the 100 EMA and the stochastic falls below 80. Stop just beyond the latest swing low / high; target 8-12 pips.
    Ours: the cross within the last 60 bars, pullback = a low at/under EMA50 in the last 5 bars, stochastic
    (14,3,3) %K crossing 20 / 80; swing = last confirmed 3-bar swing (else the 10-bar extreme) -/+ 0.1 ATR.
    Translation: EUR/USD 1-minute ATR(14) ~1.5 pips -> target 10 pips = 6.5 ATR."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    a = atr(h, l, c, 14)
    e50, e100 = ema(c, 50), ema(c, 100)
    xup = _s(cross_up(e50, e100).astype(float)).rolling(60).max().to_numpy() > 0
    pull = _s((l <= e50).astype(float)).rolling(5).max().to_numpy() > 0
    k, _ = _stoch(h, l, c, 14, 3, 3)
    sh_, sl_ = _swings(h, l, 3)
    ll, hh = _s(l).rolling(10).min().to_numpy(), _s(h).rolling(10).max().to_numpy()
    st_l = np.where(sl_ < c, sl_, ll) - 0.1 * a
    st_s = np.where(sh_ > c, sh_, hh) + 0.1 * a
    long_ = (e50 > e100) & xup & pull & cross_up(k, np.full(len(c), 20.0))
    short = (e50 < e100) & cross_down(k, np.full(len(c), 80.0))
    return long_, short & ~long_, {"stop": ("level", st_l, st_s), "target": ("atr", 6.5)}


# ---------------- SC036 Gold 5m pullback to EMA50 with 4H EMA200 trend (XS blog) ----------------
def s_xs_gold_5m_pullback(df):
    """XS.com blog 'Gold (XAUUSD) scalping trading strategy 2026', 5-minute (preferred): trend from the 200 EMA on
    the 4-hour chart; long on a pullback to the 50 EMA with a bullish candle (hammer or bullish engulfing) and RSI
    bouncing from oversold back above 40 (MACD as extra confirmation, at least three confirmations); short
    mirror (bearish candle, RSI back under 60). Stop 5-10 pips beyond the recent swing, target at least 2:1;
    London-New York overlap 13:00-17:00 GMT. Ours: the 4H trend is required plus at least 2 of: EMA50 touch with a
    close back beyond it, engulfing or hammer, RSI(14) back over 40 after being under it in the last 5 bars, MACD
    (12,26,9) bullish cross in the last 3 bars (= three confirmations), on an up candle; translation: 7.5
    gold pips ($0.75) at 5-minute ATR(14) ~$3.5 -> 0.2 ATR beyond the 5-bar extreme; target 2R."""
    h, l, c = (df[x].to_numpy() for x in "hlc")
    o = _open(df)
    a = atr(h, l, c, 14)
    e50 = ema(c, 50)
    t4 = _htf(df, "240min", lambda g: g["c"].to_numpy() - ema(g["c"].to_numpy(), 200))
    bull, bear = _engulf(o, c)
    rng = np.where(h - l > 0, h - l, np.nan)
    with np.errstate(invalid="ignore"):
        hammer = (np.minimum(o, c) - l) >= 0.6 * rng
        star = (h - np.maximum(o, c)) >= 0.6 * rng
    r = rsi(c, 14)
    was_lo = _s((r < 40).astype(float)).rolling(5).max().to_numpy() > 0
    was_hi = _s((r > 60).astype(float)).rolling(5).max().to_numpy() > 0
    hs, _, _, _, _ = _local(df, "UTC")
    sess = (hs >= 13) & (hs < 17)
    stl = _s(l).rolling(5).min().to_numpy() - 0.2 * a
    sts = _s(h).rolling(5).max().to_numpy() + 0.2 * a
    m, sg, _ = macd(c)
    mx_up = _s(cross_up(m, sg).astype(float)).rolling(3).max().to_numpy() > 0
    mx_dn = _s(cross_down(m, sg).astype(float)).rolling(3).max().to_numpy() > 0
    n_l = ((l <= e50) & (c > e50)).astype(int) + (bull | (hammer & (c >= o))).astype(int) + \
        ((r > 40) & was_lo).astype(int) + mx_up.astype(int)
    n_s = ((h >= e50) & (c < e50)).astype(int) + (bear | (star & (c <= o))).astype(int) + \
        ((r < 60) & was_hi).astype(int) + mx_dn.astype(int)
    long_ = (t4 > 0) & sess & (n_l >= 2) & (c > o)
    short = (t4 < 0) & sess & (n_s >= 2) & (c < o)
    return long_, short & ~long_, {"stop": ("level", stl, sts), "target": ("R", 2.0)}


# ---------------- helpers added 2026-11-12 (fourth scout run: the user's Facebook reels) ----------------
def _zigzag_snapshots(h, l, k):
    """Alternating swing pivots (k bars each side), each known k bars after it. Returns, for every bar t, the
    list of pivots confirmed by t as (index, price, +1 high / -1 low); consecutive same-type pivots keep the
    more extreme one."""
    n = len(h)
    piv, snaps = [], []
    for t in range(n):
        j = t - k
        if j - k >= 0:
            wh, wl = h[j - k:t + 1], l[j - k:t + 1]
            for typ, val, ok in ((1, h[j], h[j] == wh.max()), (-1, l[j], l[j] == wl.min())):
                if not ok:
                    continue
                if piv and piv[-1][2] == typ:
                    if (typ == 1 and val > piv[-1][1]) or (typ == -1 and val < piv[-1][1]):
                        piv[-1] = (j, val, typ)
                else:
                    piv.append((j, val, typ))
        snaps.append(len(piv))
    return piv, snaps


def _butterfly(df, k=3, tol=0.05, target="A"):
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    n = len(c)
    piv, snaps = _zigzag_snapshots(h, l, k)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    st_l, st_s = np.full(n, np.nan), np.full(n, np.nan)
    tg_l, tg_s = np.full(n, np.nan), np.full(n, np.nan)
    lo_r = lambda x: x * (1 - tol)  # noqa: E731
    hi_r = lambda x: x * (1 + tol)  # noqa: E731
    for t in range(1, n):
        m = snaps[t]
        for end in (m, m - 1):              # the pattern may be the last 4 pivots, or the 4 before a minor swing
            if end < 4:
                continue
            (xi, X, xt), (ai, A, _), (bi, B, _), (ci, C, ct) = piv[end - 4:end]
            if ct != -xt:
                continue
            s = xt                           # +1: X high (bearish, sell at D); -1: X low (bullish, buy at D)
            XA = s * (X - A)
            AB = s * (B - A)
            BC = s * (B - C)
            if XA <= 0 or AB <= 0 or BC <= 0:
                continue
            if not (lo_r(0.786) <= AB / XA <= hi_r(0.786) and lo_r(0.382) <= BC / AB <= hi_r(0.886)):
                continue
            D = A + s * 1.27 * XA
            CD = s * (D - C)
            if not (lo_r(1.618) <= CD / BC <= hi_r(2.24)):
                continue
            Dt = D - s * tol * XA            # the D zone starts tol x XA before the exact 1.27 level
            seg = h[ci + 1:t] if s > 0 else l[ci + 1:t]
            if s > 0:
                if h[t] < Dt or (len(seg) and seg.max() >= Dt):
                    continue
                stop = A + 1.414 * XA
                tgt = A if target == "A" else D - 0.618 * (D - A)
                if c[t] < stop and tgt < c[t]:
                    short[t], st_s[t], tg_s[t] = True, stop, tgt
            else:
                if l[t] > Dt or (len(seg) and seg.min() <= Dt):
                    continue
                stop = A - 1.414 * XA
                tgt = A if target == "A" else D + 0.618 * (A - D)
                if c[t] > stop and tgt > c[t]:
                    long_[t], st_l[t], tg_l[t] = True, stop, tgt
            break
    return long_, short & ~long_, {"stop": ("level", st_l, st_s), "target": ("level", tg_l, tg_s)}


# ---------------- SC037 / SC038 Butterfly harmonic pattern (Trader Dhana, Facebook reel) ----------------
def s_butterfly_target_a(df):
    """Butterfly harmonic pattern (Trader Dhana, Facebook reel sent by the user; rules written down by the user
    12.11.2026), intraday 15m/30m: five swing points X-A-B-C-D. AB = 0.786 retracement of XA, BC = 0.382-0.886 of
    AB, CD = 1.618-2.24 of BC, D = 1.27 extension of XA (beyond X). Bearish: X high, A low, D a new high above X
    -> sell at D; bullish mirror -> buy at D. Stop a little beyond D (ours: the 1.414 XA extension); target the A
    level. Ours: 2-bar swings, +/-5% tolerance on every ratio, entry at the close of the first bar that reaches the
    D zone (the 1.27 level minus 5% of XA)."""
    return _butterfly(df, k=2, tol=0.05, target="A")


def s_butterfly_target_618(df):
    """SC037 Butterfly with the alternative target the user asked to test: the 0.618 retracement of AD (instead of
    the A level). Everything else as SC037."""
    return _butterfly(df, k=2, tol=0.05, target="618")


def _tl_fib(df, k=3, stop_fib=0.618, target="proj"):
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    n = len(c)
    a = atr(h, l, c, 14)
    piv, snaps = _zigzag_snapshots(h, l, k)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    st_l, st_s = np.full(n, np.nan), np.full(n, np.nan)
    tg_l, tg_s = np.full(n, np.nan), np.full(n, np.nan)
    setup = None    # dict(s, i1, p1, i2, p2, top, broken, key)
    used = set()
    for t in range(1, n):
        m = snaps[t]
        if m >= 4:
            (i1, p1, t1), (_, q1, _), (i2, p2, _), (_, q2, _) = piv[m - 4:m]
            s = -t1                          # L,H,L,H -> s = +1 (uptrend); H,L,H,L -> s = -1
            key = (i1, i2)
            if s * (p2 - p1) > 0 and s * (q2 - q1) > 0 and key not in used and \
                    (setup is None or setup["key"] != key):
                setup = {"s": s, "i1": i1, "p1": p1, "i2": i2, "p2": p2, "top": q2, "broken": False, "key": key}
        if setup is None:
            continue
        s, p1 = setup["s"], setup["p1"]
        line = setup["p2"] + (setup["p2"] - p1) / (setup["i2"] - setup["i1"]) * (t - setup["i2"])
        if not setup["broken"]:
            ext = h[t] if s > 0 else l[t]
            if s * (ext - setup["top"]) > 0:
                setup["top"] = ext          # the impulse is still extending
            if s * (c[t] - line) < 0:
                setup["broken"] = True      # the trendline under the lows (over the highs) breaks
            continue
        top = setup["top"]
        R = s * (top - p1)
        if R <= 0:
            setup = None
            continue
        z50, z618 = top - s * 0.5 * R, top - s * 0.618 * R
        z786 = top - s * 0.786 * R
        if s * (c[t] - z786) < 0 or (t - setup["i2"]) > 300:
            used.add(setup["key"])          # gone through the zone, or too old
            setup = None
            continue
        touch = (l[t] <= z50) if s > 0 else (h[t] >= z50)
        if touch and s * (c[t] - z618) >= 0:
            stop = top - s * stop_fib * R - s * 0.25 * a[t]
            tgt = c[t] + s * R if target == "proj" else top
            if s > 0:
                long_[t], st_l[t], tg_l[t] = True, stop, tgt
            else:
                short[t], st_s[t], tg_s[t] = True, stop, tgt
            used.add(setup["key"])
            setup = None
    return long_, short & ~long_, {"stop": ("level", st_l, st_s), "target": ("level", tg_l, tg_s)}


# ---------------- SC039 / SC040 Trendline break + Fibonacci 0.5-0.618 (Trade Master, Facebook reel) ----------------
def s_tl_break_fib(df):
    """'Trendline break + Fibonacci retracement' (Trade Master, Facebook reel 'Beginner to pro day 26', sent by the
    user; rules written down by the user 12.11.2026): an uptrend with breaks of structure (each high above the last)
    and a rising trendline under the lows; price breaks the trendline down (the liquidity under it is taken);
    Fibonacci on the whole impulse from the low at the start of the trend (1) to the high (0); buy in the 0.5-0.618
    zone. Stop a little under 0.618; target = the length of the whole impulse projected up from the entry. Bearish
    mirror (downtrend, line over the highs breaks up, sell 0.5-0.618). Ours: 3-bar swings; trend = the last two swing
    lows and highs both rising; trendline through those two lows; entry at the close of the first bar touching 0.5
    that is still above 0.618; stop = 0.618 - 0.25 ATR(14); the setup dies on a close beyond 0.786 or after 300
    bars. The source names no timeframe - the user asked for 15 minutes to 4 hours."""
    return _tl_fib(df, k=3, stop_fib=0.618, target="proj")


def s_tl_break_fib_786(df):
    """SC039 with the alternatives the user asked to test: stop under the 0.786 level (- 0.25 ATR) and the nearer
    target = the impulse high (0). Everything else as SC039."""
    return _tl_fib(df, k=3, stop_fib=0.786, target="top")


# id: (name, fn, timeframes the source states)


# ======================= scout night 2026-11-12 (second run): SC041-SC063 =======================
def _first_per_day(mask, day):
    """Only the first True of each day."""
    m = np.asarray(mask, dtype=bool)
    cs = pd.Series(m.astype(int)).groupby(day).cumsum().to_numpy()
    return m & (cs == 1)


def _ha(df):
    """Heikin-Ashi close and open."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    hc = (o + h + l + c) / 4
    ho = np.empty(len(c))
    if len(c):
        ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return hc, ho


def _range_levels(df, tz, start_h, end_h):
    """High / low of the bars that start inside [start_h, end_h) local time, on every later bar of that day
    (NaN before the range is complete). Also returns the local start hour, end hour, weekday and day."""
    h, l = df["h"].to_numpy(float), df["l"].to_numpy(float)
    hs, he, wd, _, day = _local(df, tz)
    inr = (hs >= start_h - 1e-9) & (hs < end_h - 1e-9)
    hi = pd.Series(np.where(inr, h, np.nan)).groupby(day).transform("max").to_numpy()
    lo = pd.Series(np.where(inr, l, np.nan)).groupby(day).transform("min").to_numpy()
    after = hs >= end_h - 1e-9
    return np.where(after, hi, np.nan), np.where(after, lo, np.nan), hs, he, wd, day


# ---------------- SC041 Fluid Money Asian-session late breakout (TradingView) ----------------
def s_fluid_asian_late_break(df):
    """TradingView idea 'NEW FLUID MONEY ASIAN SESSION BREAKOUT STRATEGY (15 Min chart)' (Remytrades): 15-minute
    charts only. Box = the Asian session high/low; in the last quarter of the box, a break of the session's high
    so far -> buy, of its low -> sell. Stop 5-7 pips, target no more than 20 pips (author: 10).
    Translation (ours): Asian session 00:00-08:00 UTC, last quarter 06:00-08:00 UTC; a 15-minute close beyond
    the high/low made earlier that session; stop 6 pips ~ 0.6 ATR(15m), target 10 pips ~ 1.0 ATR(15m) (a major
    pair's 15-minute ATR is about 10 pips). First signal of the day only."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    hs, _, _, _, day = _local(df, "UTC")
    asia = hs < 8.0
    g_h = pd.Series(np.where(asia, h, -np.inf)).groupby(day)
    g_l = pd.Series(np.where(asia, l, np.inf)).groupby(day)
    prev_hi = g_h.cummax().groupby(day).shift(1).to_numpy()
    prev_lo = g_l.cummin().groupby(day).shift(1).to_numpy()
    late = (hs >= 6.0) & (hs < 8.0)
    long_ = late & np.isfinite(prev_hi) & (c > prev_hi)
    short = late & np.isfinite(prev_lo) & (c < prev_lo) & ~long_
    sig = _first_per_day(long_ | short, day)
    return long_ & sig, short & sig, {"stop": ("atr", 0.6), "target": ("atr", 1.0)}


# ---------------- SC042 EMA 9/21 cross + RSI filter, swing stop (TradingView idea) ----------------
def s_ema921_rsi_swing(df):
    """TradingView idea 'Day Trading Strategy Using EMA Crossovers + RSI for Crypto' (Cryptobingbong), 5- or
    15-minute charts: long when EMA9 crosses above EMA21 and RSI(14) is above 50 but below 70; short when EMA9
    crosses below EMA21 and RSI is below 50. Stop just beyond the most recent swing low/high, target 2:1 (1.5:1
    also named). Ours: swing = last confirmed 2-bar pivot; differs from TT45 (EMA 9/21, 1H) by the RSI filter
    and the swing stop."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e9, e21, r = ema(c, 9), ema(c, 21), rsi(c, 14)
    sh, sl = _swings(h, l, 2)
    long_ = cross_up(e9, e21) & (r > 50) & (r < 70) & (sl < c)
    short = cross_down(e9, e21) & (r < 50) & (sh > c)
    return long_, short & ~long_, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


# ---------------- SC043 15MDbleBol double Bollinger reversal (BabyPips forum) ----------------
def s_double_bb_reversal(df):
    """BabyPips forum 'Scalping 15MDbleBol': 15M chart (also 30M-Daily). Two Bollinger Bands, 20 and 25 periods
    (deviation not stated - ours 2). A candle pokes outside a band, then a reversal candle changes colour and
    closes past the previous candle's body and back inside the bands -> enter against the poke (upper = short,
    lower = long). Stop beyond the outermost wick, target the middle band (author: 1:1-2:1, flexible).
    Ours: middle band = SMA20; outermost wick = extreme of the poke candle and the reversal candle."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    lo20, mid, up20 = _bb(c, 20, 2.0)
    lo25, _, up25 = _bb(c, 25, 2.0)
    up_in, lo_in = np.fmin(up20, up25), np.fmax(lo20, lo25)
    poke_up = _sh(h) > _sh(up_in)
    poke_dn = _sh(l) < _sh(lo_in)
    po, pc = _sh(o), _sh(c)
    short = poke_up & (pc > po) & (c < o) & (c < np.fmin(po, pc)) & (c < up_in)
    long_ = poke_dn & (pc < po) & (c > o) & (c > np.fmax(po, pc)) & (c > lo_in)
    hi2, lo2 = np.fmax(h, _sh(h)), np.fmin(l, _sh(l))
    return long_ & (c < mid), short & (c > mid) & ~long_, {"stop": ("level", lo2, hi2), "target": ("level", mid, mid)}


# ---------------- SC044 RSI range re-cross, half at 0.5R (Tradingpedia) ----------------
def s_tp_rsi_range(df):
    """Tradingpedia 'Squeezing Pips from Ranging Markets via RSI' (daily stated, example on 4-hour): RSI(14) below
    30, then a bullish bar closes with RSI back above 30 -> long at the next open; mirror at 70 with a bearish
    bar -> short. Stop several pips beyond the latest swing low/high (short: 5-10 pips). Half the position out
    when price has moved 50% of the risk, the stop of the rest to break-even; the rest exits when RSI goes over
    70 and back below (long) / when RSI enters oversold (short).
    Translation (ours): swing = lowest low / highest high of the last 5 bars; 'several pips' ~ 7 pips ~ 0.2 ATR
    on 4-hour bars."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    r = rsi(c, 14)
    a = atr(h, l, c, 14)
    long_ = cross_up(r, 30) & (c > o)
    short = cross_down(r, 70) & (c < o)
    sl_l = _s(l).rolling(5).min().to_numpy() - 0.2 * a
    sl_s = _s(h).rolling(5).max().to_numpy() + 0.2 * a
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": None, "partial": (0.5, 0.5),
                                   "exit_long": cross_down(r, 70), "exit_short": r < 30}


# ---------------- SC045 RSI(3) + Full Stochastic 6/3/3 + SMA150 (Tradingpedia) ----------------
def s_tp_rsi3_stoch_sma150(df):
    """Tradingpedia 'Combining RSI, Full Stochastic Oscillator and SMA' (daily best, example on the hourly chart):
    long when price is above the 150 SMA, RSI(3) is oversold and the Full Stochastic (6,3,3) lines cross bullish
    while still oversold; short mirror below the SMA. Stop 5-10 pips beyond the latest swing; exit when the
    Stochastic reaches the opposite extreme.
    Translation (ours): oversold RSI < 30 (on this or the previous bar) and %K < 30 at the cross (the source's
    zones are '30-20' / '70-80'); swing = 5-bar extreme; 7 pips ~ 0.6 ATR on 1-hour bars."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    s150, r3, a = sma(c, 150), rsi(c, 3), atr(h, l, c, 14)
    k, d = _stoch(h, l, c, 6, 3, 3)
    r_lo = np.fmin(r3, _sh(r3))
    r_hi = np.fmax(r3, _sh(r3))
    long_ = (c > s150) & (r_lo < 30) & cross_up(k, d) & (k < 30)
    short = (c < s150) & (r_hi > 70) & cross_down(k, d) & (k > 70)
    sl_l = _s(l).rolling(5).min().to_numpy() - 0.6 * a
    sl_s = _s(h).rolling(5).max().to_numpy() + 0.6 * a
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": None,
                                   "exit_long": k > 80, "exit_short": k < 20}


# ---------------- SC046 Supertrend + Stochastic RSI pullback (The Right Trader) ----------------
def _stochrsi(c, n_rsi=14, n_st=14, ks=3, ds=3):
    r = rsi(c, n_rsi)
    lo, hi = _s(r).rolling(n_st).min().to_numpy(), _s(r).rolling(n_st).max().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100 * (r - lo) / (hi - lo)
    k = sma(np.nan_to_num(raw, nan=50.0), ks)
    return k, sma(k, ds)


def s_trt_st_stochrsi(df):
    """The Right Trader 'Supertrend and Stochastic RSI Strategy' (daily swing; 15-minute day-trade variant):
    Supertrend (10, 3) line below the close and Stochastic RSI (14,14,3,3) %K crosses above %D from at/below 20
    -> long at that close; mirror above / from 80 -> short. Stop at the Supertrend value of the signal bar,
    target 2R. Also out when the Supertrend flips or %K crosses back through %D."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    line = _supertrend_line(h, l, c, 10, 3.0)
    k, d = _stochrsi(c)
    pk = _sh(k)
    long_ = (line < c) & cross_up(k, d) & (np.fmin(k, pk) <= 20)
    short = (line > c) & cross_down(k, d) & (np.fmax(k, pk) >= 80)
    return long_, short & ~long_, {"stop": ("level", line, line), "target": ("R", 2.0),
                                   "exit_long": (c < line) | cross_down(k, d), "exit_short": (c > line) | cross_up(k, d)}


# ---------------- SC047 1-minute VWAP + EMA 9/21 pullback break (The Right Trader) ----------------
def s_trt_vwap_ema_1m(df):
    """The Right Trader 'One-Minute Scalping': 1-minute execution, 5-minute bias. Long when the latest 5-minute
    close is above VWAP, EMA9 > EMA21 on the 1-minute chart, and a pullback touches or nears EMA9 without a
    1-minute close below VWAP -> buy the break of the pullback bar's high. Stop at the pullback low; skip if the
    stop is wider than 0.3% of price. Target 1.5R, or out on a 1-minute close back through VWAP. Flat before
    the session close. Shorts mirror.
    Ours: VWAP from 00:00 UTC; 'near' = within 0.1 ATR; the break = this bar closing beyond the pullback bar's
    extreme (entry next open); session = New York cash hours, entries 9:30-15:45, flat at 15:45 New York.
    Translation (ours): the 0.3%-of-price cap (a ~$50 share, whose 1-minute ATR is ~0.08%) -> stop at most
    3.5 x ATR(14) of the 1-minute bars, so it scales to every instrument."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    vw = _vwap_day(df)
    c5 = _htf(df, "5min", lambda g: g["c"].to_numpy())
    e9, e21, a = ema(c, 9), ema(c, 21), atr(h, l, c, 14)
    hs, _, _, _, _ = _local(df, "America/New_York")
    sess = (hs >= 9.5) & (hs < 15.75)
    pl, ph, pe9, pvw, pc = _sh(l), _sh(h), _sh(e9), _sh(vw), _sh(c)
    long_ = sess & (c5 > vw) & (e9 > e21) & (pl <= pe9 + 0.1 * _sh(a)) & (pc > pvw) & (c > ph) & ((c - pl) <= 3.5 * a)
    short = sess & (c5 < vw) & (e9 < e21) & (ph >= pe9 - 0.1 * _sh(a)) & (pc < pvw) & (c < pl) & ((ph - c) <= 3.5 * a)
    return long_, short & ~long_, {"stop": ("level", pl, ph), "target": ("R", 1.5),
                                   "exit_long": c < vw, "exit_short": c > vw, "force_exit": hs >= 15.75}


# ---------------- SC048 SuperTrade ST1: Supertrend flip over EMA200 (TradingView) ----------------
def s_supertrade_st1(df):
    """TradingView 'SuperTrade ST1 Strategy' (SupertradeOfficial): any timeframe, 1H / 4H / Daily 'more
    reliable'. Long only: the Supertrend flips from bearish to bullish, price closes above the line and above
    EMA200. Stop 1 x ATR below the entry, target 4 x ATR (1:4). Ours: Supertrend (10, 3) - TradingView's
    default, the page states no values - and ATR(14)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    line = _supertrend_line(h, l, c, 10, 3.0)
    up = c > line
    flip = up & ~np.concatenate([[True], up[:-1]])
    long_ = flip & (c > ema(c, 200))
    return long_, np.zeros(len(c), bool), {"stop": ("atr", 1.0), "target": ("atr", 4.0)}


# ---------------- SC049 ETH Signal 15m: Supertrend 12/2.76 + RSI 12 (TradingView) ----------------
def s_eth_st_rsi_15m(df):
    """TradingView 'ETH Signal 15m' (ltphau99), 15-minute ETHUSDT perpetual: Supertrend (ATR 12, factor 2.76)
    buy signal with RSI(12) below 70 -> long; sell signal with RSI above 30 -> short. Stop 4 x ATR(12); target
    2 x ATR(12) for longs, 2.237 x ATR(12) for shorts (as published). Ours: levels from the signal close."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    line = _supertrend_line(h, l, c, 12, 2.76)
    up = c > line
    pu = np.concatenate([[up[0] if len(up) else True], up[:-1]])
    r = rsi(c, 12)
    a = atr(h, l, c, 12)
    long_ = up & ~pu & (r < 70)
    short = ~up & pu & (r > 30)
    return long_, short, {"stop": ("level", c - 4 * a, c + 4 * a), "target": ("level", c + 2 * a, c - 2.237 * a)}


# ---------------- SC050 EMA50 + MACD histogram + RSI band, 1/2 ATR (FMZ) ----------------
def s_fmz_ema50_macd_rsi(df):
    """FMZ 'Multi-Indicator ATR Dynamic SL/TP' (15-minute): long when price is above EMA50, the MACD (12,26,9)
    histogram is positive and RSI(14) is above 50 and below 70; short when below EMA50, histogram negative and
    RSI below 50 and above 30. Stop 1 x ATR, target 2 x ATR. Ours: a signal on the bar the conditions become
    true together (not on every bar they stay true)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e50, r = ema(c, 50), rsi(c, 14)
    _, _, hist = macd(c)
    L = (c > e50) & (hist > 0) & (r > 50) & (r < 70)
    S = (c < e50) & (hist < 0) & (r < 50) & (r > 30)
    long_ = L & ~np.concatenate([[True], L[:-1]])
    short = S & ~np.concatenate([[True], S[:-1]])
    return long_, short, {"stop": ("atr", 1.0), "target": ("atr", 2.0)}


# ---------------- SC051 Engulfing / hammer + MACD cross, 1.5/2 ATR (FMZ, dev.to) ----------------
def s_fmz_candle_macd(df):
    """FMZ 'Multi-Level Trend Momentum Trading Strategy with ATR Risk Management' (dev.to, 15-minute): long on a
    bullish engulfing or a hammer together with the MACD (12,26,9) line crossing above its signal on the same
    bar; short on a bearish engulfing or shooting star with a MACD cross down. Stop 1.5 x ATR(14), target
    2 x ATR(14). Ours: hammer = lower wick >= 2 x body and upper wick <= body (the published code's hammer test
    is always true for green candles, so we use the textbook shape); shooting star mirrors it."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    body = np.abs(c - o)
    lw = np.fmin(o, c) - l
    uw = h - np.fmax(o, c)
    hammer = (body > 0) & (lw >= 2 * body) & (uw <= body)
    star = (body > 0) & (uw >= 2 * body) & (lw <= body)
    bull, bear = _engulf(o, c)
    m, s, _ = macd(c)
    long_ = (bull | hammer) & cross_up(m, s)
    short = (bear | star) & cross_down(m, s)
    return long_, short & ~long_, {"stop": ("atr", 1.5), "target": ("atr", 2.0)}


# ---------------- SC052 Three-Step 9:30 range, FVG break (TradingView) ----------------
def s_three_step_930_break(df):
    """TradingView 'Three-Step 9:30 Range Scalping' (nicolasnourgt): range = high/low of the first 5-minute
    candle after 9:30 New York; trade on the 1-minute chart 9:30-10:30 only. BREAK: a fair value gap (3-candle
    wick gap) where one of its 3 candles closes outside the range -> trade the break; stop at the low (long) /
    high (short) of the first candle that closed outside the range; target 2:1. Body closes, not wicks.
    Ours: only the BREAK pattern (the TRAP and REVERSAL patterns are not modelled); one trade a day."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    n = len(c)
    hi, lo, hs, _, _, day = _range_levels(df, "America/New_York", 9.5, 9.5 + 5 / 60)
    long_, short = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    win = np.flatnonzero(np.isfinite(hi) & (hs < 10.5))
    cur, first_up, first_dn, done = None, None, None, False
    for k in win:
        if day[k] != cur:
            cur, first_up, first_dn, done = day[k], None, None, False
        if done:
            continue
        if first_up is None and c[k] > hi[k]:
            first_up = k
        if first_dn is None and c[k] < lo[k]:
            first_dn = k
        if k < 2:
            continue
        if first_up is not None and l[k] > h[k - 2] and max(c[k - 2], c[k - 1], c[k]) > hi[k] and c[k] > hi[k]:
            long_[k], sl_l[k], done = True, l[first_up], True
        elif first_dn is not None and h[k] < l[k - 2] and min(c[k - 2], c[k - 1], c[k]) < lo[k] and c[k] < lo[k]:
            short[k], sl_s[k], done = True, h[first_dn], True
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


# ---------------- SC053 Macketings 1-minute EMA band scalp (TradingView) ----------------
def s_macketings_1m(df):
    """TradingView 'Macketings 1min Scalping' (Macketing1337), 1-minute: EMA 80/90 band above EMA 340/500 and
    EMA340 above EMA500, longs only above the SMA325 safety filter; price retests / dips into the 80/90 band,
    then a momentum breakout out of it -> long (shorts mirror). Stop 0.2%, target 2.5%; at 0.3% profit the stop
    moves to lock 0.2%. Out when EMA 80 or 90 crosses EMA340, or EMA340 crosses EMA500.
    Ours: 'Both' entry mode = the previous bar dipped into the band and this bar closes above the band and the
    previous high; break-even = 0.3/2.5 = 12% of the way to the target, offset 0.2% of price. Not modelled:
    the profit-gated band exit, the trailing stop, the 100-bar cooldown, the session toggles."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e80, e90, e340, e500, s325 = ema(c, 80), ema(c, 90), ema(c, 340), ema(c, 500), sma(c, 325)
    top, bot = np.fmax(e80, e90), np.fmin(e80, e90)
    up = (bot > np.fmax(e340, e500)) & (e340 > e500) & (c > s325)
    dn = (top < np.fmin(e340, e500)) & (e340 < e500) & (c < s325)
    long_ = up & (_sh(l) <= _sh(top)) & (c > top) & (c > _sh(h))
    short = dn & (_sh(h) >= _sh(bot)) & (c < bot) & (c < _sh(l))
    xl = cross_down(e80, e340) | cross_down(e90, e340) | cross_down(e340, e500)
    xs = cross_up(e80, e340) | cross_up(e90, e340) | cross_up(e340, e500)
    return long_, short & ~long_, {"stop": ("level", c * 0.998, c * 1.002),
                                   "target": ("level", c * 1.025, c * 0.975),
                                   "breakeven": (0.12, 0.002), "exit_long": xl, "exit_short": xs}


# ---------------- SC054 CCI + EMA 10/21/50 cross (RoboForex) ----------------
def s_robo_cci_ema(df):
    """RoboForex 'Trading with Scalping Forex Strategy CCI + EMA', M5 and higher (H charts also): EMA10 crosses up
    through both EMA21 and EMA50 with CCI above zero (ideally above 100) -> long; mirror -> short. Stop 10-15
    points beyond the entry (or behind the nearest pivot), target 2-3 times the stop.
    Translation (ours): CCI(14) (MetaTrader default); 12 points ~ 2.5 ATR on M5 bars (a major's M5 ATR is ~5
    pips), target 2.5R (middle of 2-3)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e10, e21, e50 = ema(c, 10), ema(c, 21), ema(c, 50)
    cc = _cci(h, l, c, 14)
    long_ = cross_up(e10, np.fmax(e21, e50)) & (cc > 0)
    short = cross_down(e10, np.fmin(e21, e50)) & (cc < 0)
    return long_, short & ~long_, {"stop": ("atr", 2.5), "target": ("R", 2.5)}


# ---------------- SC055 EMA21 + CCI20 agreement (The Right Trader) ----------------
def s_trt_ema21_cci(df):
    """The Right Trader 'EMA and CCI Strategy' (daily swing; 15-minute day-trade variant, same settings): long on
    the closed bar where the close is above EMA21 and CCI(20) is at/above 0 (the second of the two becoming
    true); short mirror. Stop below the swing low of the pullback (above the swing high). No fixed target: out
    when the close crosses back through EMA21 or CCI back through 0; the day variant is also flat at the
    session close. Ours: swing = 5-bar extreme; session close = 16:00 New York."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e21, cc = ema(c, 21), _cci(h, l, c, 20)
    L = (c > e21) & (cc >= 0)
    S = (c < e21) & (cc <= 0)
    long_ = L & ~np.concatenate([[True], L[:-1]])
    short = S & ~np.concatenate([[True], S[:-1]])
    _, he, _, _, _ = _local(df, "America/New_York")
    hs, _, _, _, _ = _local(df, "America/New_York")
    close_bar = (he >= 16.0) & (hs < 16.0)
    return long_, short, {"stop": ("lowest", 5), "target": None, "exit_long": ~L, "exit_short": ~S,
                          "force_exit": close_bar}


# ---------------- SC056 Basic CCI RSI two-bar extreme (StockSharp) ----------------
def s_ss_cci_rsi(df):
    """StockSharp API example 2998 'Basic CCI RSI' (1-hour candles): long when CCI(12) > 80 and RSI(15) > 75 on
    the current and the previous closed candle; short when CCI < -95 and RSI < 30 on both. Stop 125 pips,
    take profit 60 pips (trailing 5 / step 5 pips).
    Translation (ours): the text rule (the page's code shows only a CCI zero cross); a signal on the first bar
    of such a two-bar run; on 1-hour bars a major's ATR is ~12 pips -> stop 10 ATR, target 5 ATR; the trailing
    stop is not modelled."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    cc, r = _cci(h, l, c, 12), rsi(c, 15)
    up = (cc > 80) & (r > 75)
    dn = (cc < -95) & (r < 30)
    L = up & np.concatenate([[False], up[:-1]])
    S = dn & np.concatenate([[False], dn[:-1]])
    long_ = L & ~np.concatenate([[True], L[:-1]])
    short = S & ~np.concatenate([[True], S[:-1]])
    return long_, short, {"stop": ("atr", 10.0), "target": ("atr", 5.0)}


# ---------------- SC057 XS M15 EMA20/50 pullback + RSI50 + reversal candle ----------------
def s_xs_m15_ema_pullback(df):
    """XS '15 minute Scalping Strategy' (M15): price above EMA20 and EMA50, a pullback touches (or slightly dips
    toward) EMA20, RSI(14) above 50 and a bullish reversal candle (engulfing or pin bar) closes -> long at that
    close; mirror for shorts. Stop 2-3 pips beyond the recent swing low/high; target the next S/R, R:R 1:1.5 to
    1:2. London and New York sessions, Asian session avoided.
    Translation (ours): swing = 5-bar extreme, 2.5 pips ~ 0.25 ATR(15m); target 2R; sessions 07:00-20:00 UTC;
    pin bar = wick toward the trend >= 2 x body and close in the bar's far half."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    e20, e50, r, a = ema(c, 20), ema(c, 50), rsi(c, 14), atr(h, l, c, 14)
    body = np.abs(c - o)
    rng = np.where(h - l > 0, h - l, np.nan)
    pin_bull = ((np.fmin(o, c) - l) >= 2 * body) & ((c - l) / rng > 0.5)
    pin_bear = ((h - np.fmax(o, c)) >= 2 * body) & ((h - c) / rng > 0.5)
    bull, bear = _engulf(o, c)
    hs, _, _, _, _ = _local(df, "UTC")
    sess = (hs >= 7.0) & (hs < 20.0)
    touch_l = np.fmin(l, _sh(l)) <= e20
    touch_h = np.fmax(h, _sh(h)) >= e20
    long_ = sess & (c > e20) & (c > e50) & touch_l & (r > 50) & (bull | pin_bull)
    short = sess & (c < e20) & (c < e50) & touch_h & (r < 50) & (bear | pin_bear)
    sl_l = _s(l).rolling(5).min().to_numpy() - 0.25 * a
    sl_s = _s(h).rolling(5).max().to_numpy() + 0.25 * a
    return long_, short & ~long_, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


# ---------------- SC058 NY 9:30-10:00 range breakout, half at 2R, 3R (MQL5 blog) ----------------
def s_mql5_930_range_2r3r(df):
    """MQL5 blog 'The 09:30 Liquidity Expansion' (Volume Strike EA), M1/M5/M15: range = 09:30-10:00 New York
    high/low; buy when price reaches the range high (+ offset), sell at the range low (- offset). Stop at the
    opposite side of the range; TP1 at 2R closes 50% and the rest's stop goes to entry; TP2 at 3R. One trade a
    day. Ours: a close beyond the range (offset 0), entries until 16:00 New York; partial = half at 2/3 of the
    way to the 3R target. Differs from SS02 (same range, 1.5R target) by the exits and one trade a day."""
    c = df["c"].to_numpy(float)
    hi, lo, hs, _, _, day = _range_levels(df, "America/New_York", 9.5, 10.0)
    ok = np.isfinite(hi) & (hs < 16.0)
    L = ok & (c > hi)
    S = ok & (c < lo) & ~L
    sig = _first_per_day(L | S, day)
    return L & sig, S & sig, {"stop": ("level", lo, hi), "target": ("R", 3.0), "partial": (0.5, 2 / 3)}


# ---------------- SC059 Bollinger mid + MACD state, 3-bar stop (The Right Trader) ----------------
def s_trt_bb_macd_5m(df):
    """The Right Trader 'Bollinger Bands and MACD Strategy' (daily swing; 5-minute day-trade variant): long on the
    first close strictly above the middle band (BB 20, 2) with the MACD (12,26,9) line above its signal; short
    mirror. Stop at the 3-bar low / high; skip when that stop is wider than the middle-to-outer band gap. No
    fixed target: out on the first close that loses either condition; the day variant is flat at the session
    close. Ours: session close = 16:00 New York."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    lo, mid, up = _bb(c, 20, 2.0)
    m, s, _ = macd(c)
    L = (c > mid) & (m > s)
    S = (c < mid) & (m < s)
    sl_l = _s(l).rolling(3).min().to_numpy()
    sl_s = _s(h).rolling(3).max().to_numpy()
    long_ = L & ~np.concatenate([[True], L[:-1]]) & ((c - sl_l) <= (up - mid))
    short = S & ~np.concatenate([[True], S[:-1]]) & ((sl_s - c) <= (mid - lo))
    hs, he, _, _, _ = _local(df, "America/New_York")
    return long_, short, {"stop": ("level", sl_l, sl_s), "target": None, "exit_long": ~L, "exit_short": ~S,
                          "force_exit": (he >= 16.0) & (hs < 16.0)}


# ---------------- SC060 BTC 15m: SMA200 up + SMA5 cross of the BB mid (TradingView idea) ----------------
def s_tv_btc15_sma5_mid(df):
    """TradingView idea 'Scalping Strategy for Trading BTCUSD on 15min' (Michael_Harding), 15 min: price above the
    200 SMA and the 200 SMA sloping up, the 5 SMA crosses above the Bollinger midline (21 MA), bullish price
    action -> long. Stop on the fractal below the entry candle; close when price closes below the 21 MA midline.
    Long only (no sell setup given). Ours: slope = SMA200 above its value 10 bars ago; bullish price action =
    a green candle; fractal = last confirmed 5-bar down-fractal."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    s200, s21, s5 = sma(c, 200), sma(c, 21), sma(c, 5)
    _, fr_lo = _swings(h, l, 2)
    long_ = (c > s200) & (s200 > _sh(s200, 10)) & cross_up(s5, s21) & (c > o) & (fr_lo < c)
    return long_, np.zeros(len(c), bool), {"stop": ("level", fr_lo, fr_lo), "target": None,
                                           "exit_long": c < s21}


# ---------------- SC061 Gold 1h Bollinger + 53-bar high breakout, ATR targets (ProRealCode) ----------------
def s_prc_gold_bb_53(df):
    """ProRealCode 'Gold (1h): Intraday Bollinger Breakout with ATR Targets': long when the close is below the
    upper Bollinger band (15) and above the highest high of the previous 53 bars; stop 2.9 x ATR(9), target
    5.8 x ATR(9). Long only, one position at a time. Ours: band deviation 2 (ProRealTime default, not shown)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    _, _, up15 = _bb(c, 15, 2.0)
    hh53 = _s(h).rolling(53).max().shift(1).to_numpy()
    a9 = atr(h, l, c, 9)
    long_ = (c < up15) & (c > hh53)
    return long_, np.zeros(len(c), bool), {"stop": ("level", c - 2.9 * a9, c + 2.9 * a9),
                                           "target": ("level", c + 5.8 * a9, c - 5.8 * a9)}


# ---------------- SC062 XAU 55-MA high/low channel + Heikin-Ashi (TradingView idea) ----------------
def s_xau_ma55_ha(df):
    """TradingView idea 'XAU/USD - Scalping Strategy' (warsam), M1 and M5: channel of a 55 MA on the highs and a
    55 MA on the lows; price breaks above the high MA with a green Heikin-Ashi candle -> buy, below the low MA
    with a red one -> sell; no trades inside the channel. Stop below the previous candle's low (above its high),
    target 1:1.5. Ours: simple MAs (type not stated); the optional 200 MA filter and breakeven/partial at the
    target are not modelled."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    mh, ml = sma(h, 55), sma(l, 55)
    hc, ho = _ha(df)
    long_ = cross_up(c, mh) & (hc > ho)
    short = cross_down(c, ml) & (hc < ho)
    return long_, short & ~long_, {"stop": ("level", _sh(l), _sh(h)), "target": ("R", 1.5)}


# ---------------- SC063 ES 5-minute ORB, target half the range (Edgeful) ----------------
def s_edgeful_orb5_half(df):
    """Edgeful '5 minute opening range breakout on ES': range = 09:30-09:35 ET; break above -> long, below ->
    short; stop at the opposite edge of the range, target 50% of the range beyond the breakout level; skip days
    whose range is wider than 0.55% of the opening price; Tuesday breakouts off. Ours: a 5-minute close beyond
    the range, one trade a day, entries until 16:00 ET.
    Translation (ours): the 0.55%-of-price cap (ES, whose 5-minute ATR around the open is ~0.15%) -> range at
    most 3.5 x ATR(14) of the 5-minute bars, so it scales to every instrument; Tuesdays skipped both ways (the
    page is ambiguous whether only Tuesday longs are off)."""
    c = df["c"].to_numpy(float)
    hi, lo, hs, _, wd, day = _range_levels(df, "America/New_York", 9.5, 9.5 + 5 / 60)
    a = atr(df["h"].to_numpy(float), df["l"].to_numpy(float), c, 14)
    a_or = pd.Series(np.where((hs >= 9.5 - 1e-9) & (hs < 9.5 + 5 / 60 - 1e-9), a, np.nan)).groupby(day).transform("last").to_numpy()
    ok = np.isfinite(hi) & (hs < 16.0) & ((hi - lo) <= 3.5 * a_or) & (wd != 1)
    L = ok & (c > hi)
    S = ok & (c < lo) & ~L
    sig = _first_per_day(L | S, day)
    rng = hi - lo
    return L & sig, S & sig, {"stop": ("level", lo, hi), "target": ("level", hi + 0.5 * rng, lo - 0.5 * rng)}



# ======================= scout night 2026-11-13: SC064-SC078 =======================
def _day_vwap_local(df, tz):
    """VWAP from the start of the local day in `tz` (no volume -> equal weights)."""
    h, l, c, v = (df[x].to_numpy(dtype=float) for x in "hlcv")
    tp = (h + l + c) / 3
    v = np.where(v > 0, v, 1.0)
    day = pd.Series(df.index.tz_convert(tz).date).to_numpy()
    return pd.Series(tp * v).groupby(day).cumsum().to_numpy() / pd.Series(v).groupby(day).cumsum().to_numpy()


def _hammer(o, h, l, c):
    body = np.abs(c - o)
    lw = np.minimum(o, c) - l
    uw = h - np.maximum(o, c)
    return (lw >= 2 * body) & (uw <= np.maximum(body, 1e-12)) & (h > l)


def _since(mask, n):
    """True if `mask` was True on any of the previous n bars (not this one)."""
    return pd.Series(np.asarray(mask, dtype=float)).shift(1).rolling(n, min_periods=1).max().fillna(0).to_numpy() > 0


# ---------------- SC064 Double Bollinger 20/1 + 50/1 + RSI 50 (TradingView, tweakerID) ----------------
def s_tv_double_bollinger(df):
    """TradingView 'Double Bollinger Strategy' (tweakerID, from a Bitcoin book), recommended 1-minute or 5-minute:
    fast BB(20, 1 SD), slow BB(50, 1 SD), RSI(14). Long: RSI > 50, a candle closes above the fast band's top, then
    a later candle closes above the slow band's top (price above both) -> buy; short mirror under the bottoms with
    RSI < 50. Stop under the entry candle's low (over its high); no fixed target - out when a candle closes back
    below the fast band's top (short: back above the fast band's bottom). Ours: the 'earlier' fast-band close is
    looked for in the previous 10 bars; the signal is the first close above the slow top."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    flo, _, fup = _bb(c, 20, 1.0)
    slo, _, sup = _bb(c, 50, 1.0)
    r = rsi(c, 14)
    long_ = (r > 50) & cross_up(c, sup) & (c > fup) & _since(c > fup, 10)
    short = (r < 50) & cross_down(c, slo) & (c < flo) & _since(c < flo, 10)
    return long_, short & ~long_, {"stop": ("level", l, h), "target": None,
                                   "exit_long": c < fup, "exit_short": c > flo}


# ---------------- SC065 EMA 5/20 cross + Bollinger + MACD histogram, 5-minute (FXOpen) ----------------
def s_fxo_ema520_bb_macd(df):
    """FXOpen 'Three working 5-minute strategies' - #3 EMA, MACD and Bollinger Bands, 5-minute forex: EMA5 crosses
    above EMA20, price near the lower or the middle Bollinger band (20, 2), MACD(12,26,9) histogram above zero ->
    buy (mirror: upper/middle band, histogram below zero). Stop 7 pips beyond the closest support (resistance);
    close when the MACD histogram makes its new high (short: new low).
    Ours: 'near the lower or middle band' = close at or under the middle band + a quarter of the half-width (short
    mirror); support = lowest low of the last 10 bars; 7 pips on a major's 5-minute ATR(14) (~4.5 pips) -> 1.5 ATR
    beyond it; the 'new high' exit = the first bar the histogram turns down from its high."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e5, e20 = ema(c, 5), ema(c, 20)
    lo, mid, up = _bb(c, 20, 2.0)
    _, _, hist = macd(c, 12, 26, 9)
    a = atr(h, l, c, 14)
    long_ = cross_up(e5, e20) & (c <= mid + 0.25 * (up - mid)) & (c >= lo - 0.25 * (up - mid)) & (hist > 0)
    short = cross_down(e5, e20) & (c >= mid - 0.25 * (mid - lo)) & (c <= up + 0.25 * (mid - lo)) & (hist < 0)
    sup = _s(l).rolling(10).min().to_numpy() - 1.5 * a
    res = _s(h).rolling(10).max().to_numpy() + 1.5 * a
    ph = _sh(hist)
    return long_, short & ~long_, {"stop": ("level", sup, res), "target": None,
                                   "exit_long": (hist < ph) & (ph > 0), "exit_short": (hist > ph) & (ph < 0)}


# ---------------- SC066 EMA 10/21/50 midpoint pullback, 5/10 pips, 5-minute (IndicatorVault) ----------------
def s_iv_ema1021_mid_5m(df):
    """IndicatorVault 'The 5 Minute Scalping System', 5-minute EUR/USD (others possible): EMA10, EMA21, EMA50;
    uptrend = rising EMA50 and higher highs -> buy when price retraces to the midpoint between EMA10 and EMA21
    (downtrend mirror). Stop 5 pips (+spread), target 10 pips (+spread); after the London or New York open, not the
    Asian session. Ours: rising = EMA50 above its value 5 bars ago, higher highs = EMA10 > EMA21; the touch = the
    bar's low reaches the midpoint and it closes above it; window 08:00 London to 17:00 New York.
    Translation: EUR/USD 5-minute ATR(14) ~4.5 pips -> stop 1.1 ATR, target 2.2 ATR."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e10, e21, e50 = ema(c, 10), ema(c, 21), ema(c, 50)
    mid = (e10 + e21) / 2
    lon, _, _, _, _ = _local(df, "Europe/London")
    ny, _, _, _, _ = _local(df, "America/New_York")
    win = (lon >= 8) & (ny < 17)
    up = (e50 > _sh(e50, 5)) & (e10 > e21)
    dn = (e50 < _sh(e50, 5)) & (e10 < e21)
    long_ = up & (l <= mid) & (c > mid) & win
    short = dn & (h >= mid) & (c < mid) & win
    return long_, short & ~long_, {"stop": ("atr", 1.1), "target": ("atr", 2.2)}


# ---------------- SC067 5-minute intraday: SMA50 angle + ADX 13 + 21-EMA pullback (BabyPips) ----------------
def s_bp_5m_sma_angle(df):
    """BabyPips forum '5 Minute Intraday Trading System' (EUR/USD, GBP/USD, USD/JPY, EUR/JPY), 5-minute: EMA10,
    EMA21, SMA50 with a steep angle (21-EMA of the SMAangle outside the +/-0.1 no-trade zone), ADX(13) above 25 with
    +DI over -DI; price pulls back through the 10 EMA to the 21 EMA and the next candle pulls away from it -> buy
    (mirror). Stop 6 pips, break-even after 6 pips, target 10-15 pips.
    Ours: the SMAangle = the 1-bar change of SMA50 in ATR(14) units, smoothed by a 21 EMA, beyond +/-0.1; the
    pullback = previous low at/under the 21 EMA, this bar closes above it and above the previous close.
    Translation: 5-minute ATR(14) of these pairs ~4.5 pips -> stop 1.3 ATR, target 12.5 pips = 2.8 ATR,
    break-even at 6/12.5 ~ 0.5 of the way to the target."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e10, e21, s50 = ema(c, 10), ema(c, 21), sma(c, 50)
    a = atr(h, l, c, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        ang = ema(np.nan_to_num((s50 - _sh(s50)) / a), 21)
    ax = adx(h, l, c, 13)
    pdi, ndi = _di(h, l, c, 13)
    pc, pl, ph = _sh(c), _sh(l), _sh(h)
    pe = _sh(e21)
    long_ = (ang > 0.1) & (ax > 25) & (pdi > ndi) & (e10 > e21) & (pl <= pe) & (c > e21) & (c > pc)
    short = (ang < -0.1) & (ax > 25) & (ndi > pdi) & (e10 < e21) & (ph >= pe) & (c < e21) & (c < pc)
    return long_, short & ~long_, {"stop": ("atr", 1.3), "target": ("atr", 2.8), "breakeven": (0.5, 0.0)}


# ---------------- SC068 5-minute Bollinger squeeze + VWAP, middle-band stop, 2R, 60-minute stop (DayTradingToolkit) ----
def s_dtt_bb_squeeze_vwap(df):
    """DayTradingToolkit 'The Bollinger Band Squeeze Strategy', 5-minute chart, liquid US stocks / ETFs / index
    futures: BB(20, 2), band width (upper-lower)/middle at its lowest in ~100 bars = squeeze; a 5-minute close above
    the upper band on above-average volume, long only above a rising VWAP (short: close below the lower band under a
    falling VWAP) -> enter past that candle. Stop at the middle band; half at 2R, rest trailed; out if 2R is not
    reached within 60 minutes. Ours: the squeeze within the last 10 bars; volume at/above its 20-bar average;
    VWAP from 00:00 New York; target 2R on the whole position, time exit after 60 minutes of bars; the
    're-enters the bands within 3 bars' invalidation is not modelled."""
    h, l, c, v = (df[x].to_numpy(float) for x in "hlcv")
    lo, mid, up = _bb(c, 20, 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (up - lo) / mid
    sq = w <= _s(w).rolling(100).min().to_numpy() + 1e-12
    recent = _s(sq.astype(float)).rolling(10, min_periods=1).max().to_numpy() > 0
    vw = _day_vwap_local(df, "America/New_York")
    vol_ok = v >= sma(v, 20)
    long_ = recent & cross_up(c, up) & vol_ok & (c > vw) & (vw > _sh(vw))
    short = recent & cross_down(c, lo) & vol_ok & (c < vw) & (vw < _sh(vw))
    mb = max(1, int(round(60 / max(_bar_minutes(df), 1.0))))
    return long_, short & ~long_, {"stop": ("level", mid, mid), "target": ("R", 2.0), "max_bars": mb}


# ---------------- SC069 Gold: Asian range broken in the 12-16 GMT window, 1.5x range target (NYC Servers) ----------------
def s_nycs_gold_asian_ny(df):
    """NYC Servers 'Gold (XAUUSD) trading strategy' #5 session breakout, M15/M30: Asian range 00:00-07:00 GMT;
    trade window 12:00-16:00 GMT; a candle closes above the Asian high -> buy, below the Asian low -> sell. Stop at
    the opposite end of the range, target 1.5x to 2x the range width. Ours: target 1.5x the width beyond the
    broken edge; first break of the day only; the 200-pip cap for an 'exceptionally wide' range is not modelled."""
    c = df["c"].to_numpy(float)
    hi, lo, hs, _, _, day = _range_levels(df, "UTC", 0.0, 7.0)
    ok = np.isfinite(hi) & (hs >= 12.0) & (hs < 16.0)
    L = ok & (c > hi)
    S = ok & (c < lo) & ~L
    first = _first_per_day(L | S, day)
    w = hi - lo
    return L & first, S & first, {"stop": ("level", lo, hi), "target": ("level", hi + 1.5 * w, lo - 1.5 * w)}


# ---------------- SC070 Keltner trend pullback to the 20 EMA, 2 ATR target (JournalPlus) ----------------
def _htf_above_ema20(df):
    bm = _bar_minutes(df)
    rule = f"{int(max(1, round(4 * bm)))}min"
    return _htf(df, rule, lambda g: (g["c"].to_numpy(float) > ema(g["c"].to_numpy(float), 20)).astype(float)) > 0.5


def s_jp_keltner_pullback(df):
    """JournalPlus 'Keltner Channel Strategy' entry 1 (trend pullback), 5-minute / 15-minute (SPY, QQQ, ES): KC =
    EMA20 +/- 2 x ATR(14). Price closes above the upper band and holds at least one bar; the first later bar that
    touches the 20 EMA, confirmed by a smaller bar range or declining volume, -> buy. Stop below the prior swing low or 1 ATR
    below entry, whichever is tighter; target 2 ATR above entry; long only above the 20 EMA of the next higher
    timeframe; flat before the last 15 minutes of the session. Short side not stated (long only).
    Ours: the breakout = two closes above the band in a row within the last 30 bars; 'smaller range' = not larger
    than the previous bar's; swing low = lowest low of 5 bars; higher timeframe = 4x the bar; session close 16:00
    New York."""
    h, l, c, v = (df[x].to_numpy(float) for x in "hlcv")
    m = ema(c, 20)
    a = atr(h, l, c, 14)
    out = c > m + 2 * a
    held = out & (np.nan_to_num(_sh(out.astype(float))) > 0)
    n = len(c)
    long_ = np.zeros(n, bool)
    armed, since = False, 0
    rng = h - l
    for i in range(1, n):
        if held[i]:
            armed, since = True, 0
            continue
        since += 1
        if armed and since > 30:
            armed = False
        if armed and l[i] <= m[i]:
            if rng[i] <= rng[i - 1] or v[i] < v[i - 1]:
                long_[i] = True
            armed = False
    long_ &= _htf_above_ema20(df)
    ny, _, _, _, _ = _local(df, "America/New_York")
    stop = np.maximum(_s(l).rolling(5).min().to_numpy(), c - a)
    return long_, np.zeros(n, bool), {"stop": ("level", stop, stop), "target": ("level", c + 2 * a, c - 2 * a),
                                      "force_exit": (ny >= 15.75) & (ny < 16.0)}


# ---------------- SC071 Keltner lower-band reversal candle on light volume, to the 20 EMA (JournalPlus) ----------------
def s_jp_keltner_reversal(df):
    """JournalPlus 'Keltner Channel Strategy' entry 2 (mean reversion), 5-minute / 15-minute: price touches the
    lower band (EMA20 - 2 x ATR14) with a hammer or a bullish engulfing on volume below its 20-bar average -> buy at
    the candle's close. Stop 0.25 ATR beyond the wick that tagged the band; target the 20 EMA. Long only (short not
    stated), flat before the last 15 minutes. Ours: volume at/under the average (equal volume counts); session
    close 16:00 New York. Not applied: the page's 'above the higher timeframe's 20 EMA before any long' - a
    lower-band tag almost never sits above it, so with it this entry would have no trades (the bots can add it)."""
    o = _open(df)
    h, l, c, v = (df[x].to_numpy(float) for x in "hlcv")
    m = ema(c, 20)
    a = atr(h, l, c, 14)
    bull, _ = _engulf(o, c)
    pat = _hammer(o, h, l, c) | bull
    long_ = (l <= m - 2 * a) & pat & (c < m) & (v <= sma(v, 20))
    ny, _, _, _, _ = _local(df, "America/New_York")
    st = l - 0.25 * a
    return long_, np.zeros(len(c), bool), {"stop": ("level", st, st), "target": ("level", m, m),
                                           "force_exit": (ny >= 15.75) & (ny < 16.0)}


# ---------------- SC072 Oil: first test of yesterday's high/low + 3-candle gap (Purple Trading) ----------------
def s_purple_oil_level_gap(df):
    """Purple Trading 'Trading oil intraday - price action strategy' (WTI / Brent), H1 levels with 5-minute or
    15-minute entries: levels = previous days' highs and lows, only the first test counts; at support, a bullish
    3-candle gap (candle 1's high below candle 3's low) confirms the bounce -> buy at the gap's upper edge; short
    mirror at resistance. Stop below the low formation (above the high one); target the session high/low or the H1
    level reached. Ours: levels = yesterday's high/low (UTC day); the first touch of the day, the gap within the
    next 6 bars; entry at that close; stop = the lowest low since the touch (highest high); target = the opposite
    level (yesterday's high for a long, low for a short)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    pdh, pdl, _ = _prev_day(df)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    tl = np.nan_to_num(l <= pdl).astype(bool)
    th = np.nan_to_num(h >= pdh).astype(bool)
    ft_l = _first_per_day(tl, day)
    ft_h = _first_per_day(th, day)
    gap_up = l > _sh(h, 2)
    gap_dn = h < _sh(l, 2)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stL, stS = np.full(n, np.nan), np.full(n, np.nan)
    kl = ks = -10**9
    for i in range(n):
        if ft_l[i]:
            kl = i
        if ft_h[i]:
            ks = i
        if 0 < i - kl <= 6 and gap_up[i] and day[i] == day[kl] and c[i] < pdh[i]:
            L[i] = True
            stL[i] = l[kl:i + 1].min()
            kl = -10**9
        if 0 < i - ks <= 6 and gap_dn[i] and day[i] == day[ks] and c[i] > pdl[i]:
            S[i] = True
            stS[i] = h[ks:i + 1].max()
            ks = -10**9
    S &= ~L
    return L, S, {"stop": ("level", stL, stS), "target": ("level", pdh, pdl)}


# ---------------- SC073 First 5-minute bar breakout, measured-move target (The Right Trader) ----------------
def s_trt_first_bar_orb(df):
    """The Right Trader 'Five Minute Scalping' (liquid stocks, index futures, major pairs), 5-minute: range = the
    session's first 5-minute bar, its height between 0.15% and 0.8% of price; a 5-minute close above its high ->
    long, below its low -> short. Stop at the bar's other end; target the bar's height beyond entry; out early if
    a bar closes back inside the opening bar; flat at the session close. Ours: session = New York 09:30-16:00;
    one trade a day. Translation: the 0.15%-0.8% band (an ES/SPY first 5-minute bar averages ~0.25% ~ 1 ATR) ->
    height between 0.6 and 3.2 x ATR(14) of the bars, so it works on every instrument."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    hi, lo, hs, _, _, day = _range_levels(df, "America/New_York", 9.5, 9.5 + 5 / 60)
    a = atr(h, l, c, 14)
    a_or = pd.Series(np.where((hs >= 9.5 - 1e-9) & (hs < 9.5 + 5 / 60 - 1e-9), a, np.nan)).groupby(day).transform("last").to_numpy()
    rng = hi - lo
    ok = np.isfinite(hi) & (hs < 15.75) & (rng >= 0.6 * a_or) & (rng <= 3.2 * a_or)
    L = ok & (c > hi)
    S = ok & (c < lo) & ~L
    first = _first_per_day(L | S, day)
    return L & first, S & first, {"stop": ("level", lo, hi), "target": ("level", c + rng, c - rng),
                                  "exit_long": c < hi, "exit_short": c > lo,
                                  "force_exit": hs >= 16.0 - _bar_minutes(df) / 60.0 - 1e-9}


# ---------------- SC074 Bollinger-Keltner squeeze release past both middles (The Right Trader) ----------------
def s_trt_bb_kc_release(df):
    """The Right Trader 'Bollinger Bands and Keltner Channels Strategy' (daily swing; 15-minute day variant, same
    settings): BB(20, 2) around SMA20 (population SD), KC = EMA20 +/- 2 x ATR(20) (Wilder). Squeeze = BB inside KC
    on a closed bar; release = squeeze on the previous bar, off now. Long when the release closes above both middles
    (SMA20 and EMA20), short below both. Stop one tick beyond the squeeze's low (high), including the last inside
    bar; no fixed target - out on a close under either middle (short: over either); day variant flat at the
    session close (ours: 16:00 New York)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    blo, bmid, bup = _bb(c, 20, 2.0)
    km = ema(c, 20)
    a = atr(h, l, c, 20)
    sq = np.nan_to_num((bup < km + 2 * a) & (blo > km - 2 * a)).astype(bool)
    rel = _sh(sq.astype(float)) > 0.5
    rel = rel & ~sq
    run = np.cumsum(~sq)
    gid = np.where(sq, run, -1)
    lo_run = pd.Series(np.where(sq, l, np.nan)).groupby(gid).cummin().to_numpy()
    hi_run = pd.Series(np.where(sq, h, np.nan)).groupby(gid).cummax().to_numpy()
    sq_lo, sq_hi = _sh(lo_run), _sh(hi_run)
    L = rel & (c > bmid) & (c > km)
    S = rel & (c < bmid) & (c < km) & ~L
    ny, _, _, _, _ = _local(df, "America/New_York")
    return L, S, {"stop": ("level", sq_lo, sq_hi), "target": None,
                  "exit_long": (c < bmid) | (c < km), "exit_short": (c > bmid) | (c > km),
                  "force_exit": (ny >= 16.0 - _bar_minutes(df) / 60.0 - 1e-9) & (ny < 16.0)}


# ---------------- SC075 Crude oil initial-balance breakout and retest, target IB high + IB range (NexusFi) ----------------
def s_nfi_cl_ib_retest(df):
    """NexusFi 'Crude Oil (CL) Trading Strategies' #3 initial balance breakout with retest, 5-minute: IB = 09:00-10:00
    ET; a 5-minute close above the IB high, then a pullback that holds the IB high on a close (or the first bar
    testing it and closing bullish) -> buy. Stop below the pullback swing low + 10 ticks; TP1 = IB high + the IB
    range (runner trailed on the 15-minute EMA21 - not modelled); IB wider than 80 ticks is less valid (skipped).
    Long side only (short not specified). Translation: CL 5-minute ATR(14) ~ 18 ticks -> 10 ticks = 0.55 ATR,
    80 ticks = 4.5 ATR; swing low = lowest low of the last 3 bars; one trade a day, entries until 14:30 ET."""
    o = _open(df)
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    hi, lo, hs, _, _, day = _range_levels(df, "America/New_York", 9.0, 10.0)
    a = atr(h, l, c, 14)
    a_ib = pd.Series(np.where((hs >= 9.0) & (hs < 10.0), a, np.nan)).groupby(day).transform("last").to_numpy()
    ok = np.isfinite(hi) & ((hi - lo) <= 4.5 * a_ib) & (hs < 14.5)
    brk = np.nan_to_num(ok & (c > hi)).astype(bool)
    broke_before = pd.Series(brk.astype(int)).groupby(day).cumsum().to_numpy() - brk.astype(int) > 0
    L = ok & broke_before & (l <= hi) & (c > hi) & (c > o)
    L = L & _first_per_day(L, day)
    st = _s(l).rolling(3).min().to_numpy() - 0.55 * a
    rng = hi - lo
    return L, np.zeros(len(c), bool), {"stop": ("level", st, st), "target": ("level", hi + rng, lo - rng)}


# ---------------- SC076 EIA inventory fade back into the 09:00-10:15 range (NexusFi) ----------------
def s_nfi_eia_fade(df):
    """NexusFi 'Crude Oil (CL) Trading Strategies' #2 EIA inventory fade (Wednesdays 10:30 ET), 5-minute
    confirmation: pre-report range 09:00-10:15 ET; after the release spike beyond the range stalls, a 5-minute close
    back inside the range -> fade it (spike up = short, spike down = long). Stand aside 10:15-10:45 ET; stop beyond
    the spike extreme + 10-15 ticks; TP1 VWAP, TP2 the range midpoint; out if VWAP is not reclaimed within 30
    minutes. Ours: 'confirmed reversal' = a candle closing past the previous bar's extreme against the spike; entries 10:45-14:30 ET (the CL pit close) on Wednesdays, one per side a day; target = the range midpoint; 30-minute time
    exit. Translation: CL 5-minute ATR(14) ~ 18 ticks -> 12 ticks = 0.7 ATR beyond the spike."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    hi, lo, hs, _, wd, day = _range_levels(df, "America/New_York", 9.0, 10.25)
    a = atr(h, l, c, 14)
    post = (hs >= 10.5) & (wd == 2)
    sp_hi = pd.Series(np.where(post, h, np.nan)).groupby(day).cummax().to_numpy()
    sp_lo = pd.Series(np.where(post, l, np.nan)).groupby(day).cummin().to_numpy()
    win = (wd == 2) & (hs >= 10.75) & (hs < 14.5) & np.isfinite(hi)
    o = _open(df)
    rev_dn = (c < o) & (c < _sh(l))
    rev_up = (c > o) & (c > _sh(h))
    S = win & (sp_hi > hi) & ((c < hi) | rev_dn) & (c > lo)
    L = win & (sp_lo < lo) & ((c > lo) | rev_up) & (c < hi)
    L = _first_per_day(L, day)
    S = _first_per_day(S, day) & ~L
    mid = (hi + lo) / 2
    mb = max(1, int(round(30 / max(_bar_minutes(df), 1.0))))
    return L, S, {"stop": ("level", sp_lo - 0.7 * a, sp_hi + 0.7 * a), "target": ("level", mid, mid), "max_bars": mb}


# ---------------- SC077 15-minute ORB, tight stop past the level, OR-width target (NexusFi) ----------------
def s_nfi_orb15_tight(df):
    """NexusFi 'Breakout Trading Strategies' #1 opening range breakout (ES / NQ; 15-minute opening range from 09:30
    ET): a 1-minute close beyond the range high/low, a 5-minute close being the stronger confirmation; skip it when
    price is more than half the 5-period ATR past the edge (no chasing). Stop 1-2 ticks beyond the breakout level
    (ES: 1-2 points); target the OR width as a measured move. Ours: a close beyond the edge within 0.5 x ATR(5); one
    trade a day, entries until 12:00 ET; the 50%-at-1R partial, break-even after 0.8R and the 15-30 minute time
    exit are not modelled. Translation: ES 1.5 points ~ 0.3 x ATR(14) of 5-minute bars -> stop 0.3 ATR inside the
    broken edge."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    hi, lo, hs, _, _, day = _range_levels(df, "America/New_York", 9.5, 9.75)
    a5, a14 = atr(h, l, c, 5), atr(h, l, c, 14)
    ok = np.isfinite(hi) & (hs < 12.0)
    L = ok & (c > hi) & (c - hi <= 0.5 * a5)
    S = ok & (c < lo) & (lo - c <= 0.5 * a5) & ~L
    first = _first_per_day((ok & (c > hi)) | (ok & (c < lo)), day)
    L, S = L & first, S & first
    w = hi - lo
    return L, S, {"stop": ("level", hi - 0.3 * a14, lo + 0.3 * a14), "target": ("level", hi + w, lo - w)}


# ---------------- SC078 NQ Keltner Pulse: KC(20, 2.6) close + stop past yesterday's extreme (MQL5 Market) ----------------
def s_mq_nq_keltner_pulse(df):
    """MQL5 Market 'NQ Keltner Pulse M15 Strategy' (Nasdaq, M15): setup = a close above the upper Keltner Channel
    (20, 2.6); buy stop at the daily high reference + 0.1 x ATR(30), the order valid 46 bars (short mirror: close
    under the lower band, sell stop at the daily low - 0.1 ATR). Stop 2.5 x ATR(17), target 1.5% from entry, time
    exit after 30 bars, Friday exit 19:00. Ours: KC = EMA20 +/- 2.6 x ATR(20); daily reference = yesterday's
    high/low (UTC day); the stop order = a close through that level within 46 bars of the setup; Friday 19:00 UTC;
    the 100-point trailing stop (from 1.8 ATR in profit) is not modelled."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    m = ema(c, 20)
    a20, a30 = atr(h, l, c, 20), atr(h, l, c, 30)
    up_set = _s((c > m + 2.6 * a20).astype(float)).rolling(46, min_periods=1).max().to_numpy() > 0
    dn_set = _s((c < m - 2.6 * a20).astype(float)).rolling(46, min_periods=1).max().to_numpy() > 0
    pdh, pdl, _ = _prev_day(df)
    L = up_set & cross_up(c, pdh + 0.1 * a30)
    S = dn_set & cross_down(c, pdl - 0.1 * a30) & ~L
    a17 = atr(h, l, c, 17)
    hs, _, wd, _, _ = _local(df, "UTC")
    return L, S, {"stop": ("level", c - 2.5 * a17, c + 2.5 * a17), "target": ("level", c * 1.015, c * 0.985),
                  "max_bars": 30, "force_exit": (wd == 4) & (hs >= 19.0)}



def _onset(mask):
    """True only on the first bar of a run of True."""
    m = np.asarray(mask, dtype=bool)
    return m & ~np.concatenate([[False], m[:-1]])


def _pct_levels(c, sl_pct, tp_pct):
    """Fixed-percentage stop and target around the signal close: (stop_long, stop_short), (tgt_long, tgt_short)."""
    return ((c * (1 - sl_pct / 100), c * (1 + sl_pct / 100)), (c * (1 + tp_pct / 100), c * (1 - tp_pct / 100)))


def _range_filter(c, per=100, mult=3.0):
    """DonovanWise Range Filter: smoothed range = EMA(EMA(|dc|, per), 2*per-1) * mult; the filter moves only when the
    close leaves that band. Returns the filter and the up / down bar counters."""
    avr = ema(np.abs(np.diff(c, prepend=c[0])), per)
    r = ema(avr, per * 2 - 1) * mult
    f = np.empty(len(c))
    up, dn = np.zeros(len(c)), np.zeros(len(c))
    if not len(c):
        return f, up, dn
    f[0] = c[0]
    for i in range(1, len(c)):
        p = f[i - 1]
        f[i] = (p if c[i] - r[i] < p else c[i] - r[i]) if c[i] > p else (p if c[i] + r[i] > p else c[i] + r[i])
        up[i] = up[i - 1] + 1 if f[i] > p else 0 if f[i] < p else up[i - 1]
        dn[i] = dn[i - 1] + 1 if f[i] < p else 0 if f[i] > p else dn[i - 1]
    return f, up, dn


# ---------------- SC079 Range Filter + RSI 50 + ADX 25, 1 ATR stop, 3R, 5m (TradingView) ----------------
def s_tv_range_filter_rsi_adx(df):
    """TradingView 'Range Filter Buy and Sell 5min' (Forexbros22): long when price is above the Range Filter, its
    upward counter is positive, RSI > 50 and ADX > 25 (with supportive ATR volatility); short mirror (below the
    filter, downward counter, RSI < 50, ADX > 25). Stop 1 x ATR from entry, target 3 x the risk (3:1); out when ADX
    falls back under the threshold (ranging market). Ours: the source names no lengths -> Range Filter defaults of
    the original script (sampling period 100, multiplier 3), RSI 14, ADX 14, ATR 14; the 'ATR volatility' check =
    ATR(14) above its own 50-bar mean; a signal on the first bar all conditions hold."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    f, up, dn = _range_filter(c)
    r, ax, a = rsi(c, 14), adx(h, l, c, 14), atr(h, l, c, 14)
    vol = a > sma(a, 50)
    L = _onset((c > f) & (up > 0) & (r > 50) & (ax > 25) & vol)
    S = _onset((c < f) & (dn > 0) & (r < 50) & (ax > 25) & vol) & ~L
    return L, S, {"stop": ("atr", 1.0), "target": ("R", 3.0), "exit_long": ax < 25, "exit_short": ax < 25}


# ---------------- SC080 BVK 5m: EMA 9/21 cross + MACD cross within 5 bars + RSI 50-70, swing stop, 1.5R (TradingView) ----------------
def s_tv_bvk_ema_macd_rsi(df):
    """TradingView '5-Min EMA MACD RSI Scalping (MACD 5 Candles Confirm) - BVK', 5-minute chart (forex, indices,
    crypto, stocks): long at the candle close when EMA9 crosses above EMA21, the MACD(12,26,9) line crosses above its
    signal, and RSI(14) is above 50 but below 70; short mirror (RSI 30-50). Stop below / above the recent swing low /
    high (or 0.3-0.5%); target 1.5x to 2x the risk. Ours: the 'MACD 5 candles confirm' of the title = the MACD
    cross on this bar or one of the 4 before; swing = the lowest low / highest high of the last 5 bars; target 1.5R
    (the lower end; the EMA9 trailing stop is optional and not modelled). Differs from SC042 by the MACD cross."""
    c = df["c"].to_numpy(float)
    e9, e21 = ema(c, 9), ema(c, 21)
    m, s, _ = macd(c)
    r = rsi(c, 14)
    mu = _s(cross_up(m, s).astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    md = _s(cross_down(m, s).astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    L = cross_up(e9, e21) & mu & (r > 50) & (r < 70)
    S = cross_down(e9, e21) & md & (r < 50) & (r > 30) & ~L
    return L, S, {"stop": ("lowest", 5), "target": ("R", 1.5)}


# ---------------- SC081 EMA 9/21 cross over EMA50 + RSI 40-70, 0.5% / 1%, 15m (FMZ) ----------------
def s_fmz_ema9_21_50_rsi(df):
    """FMZ 'Multi-EMA Crossover with RSI Momentum Adaptive Scalping Strategy' (also on dev.to), 15-minute: long when
    EMA9 crosses above EMA21, the close is above EMA50 and 40 < RSI(14) < 70; short when EMA9 crosses below EMA21,
    the close is under EMA50 and 30 < RSI < 60. Stop 0.5% and target 1.0% from the signal close (the code's
    defaults; percentages, no pip translation)."""
    c = df["c"].to_numpy(float)
    e9, e21, e50, r = ema(c, 9), ema(c, 21), ema(c, 50), rsi(c, 14)
    L = cross_up(e9, e21) & (c > e50) & (r > 40) & (r < 70)
    S = cross_down(e9, e21) & (c < e50) & (r > 30) & (r < 60) & ~L
    (sl, ss), (tl, ts) = _pct_levels(c, 0.5, 1.0)
    return L, S, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- SC082 5m close over the SMA21-of-highs + VWAP + EMA50 + RSI 60, SMA21-low stop, 1.5R (FMZ) ----------------
def s_fmz_sma21hl_vwap_rsi(df):
    """FMZ '5-Min Call/Put Entry Strategy' (Five-minute trend breakout momentum), 5-minute: long when the close is
    above the 21-period SMA of the highs, above VWAP, above EMA50 and RSI > 60; short when the close is under the
    21-period SMA of the lows, under VWAP, under EMA50 and RSI < 40. The stop is the other SMA band (long out on a
    close under the SMA21 of the lows, short on a close over the SMA21 of the highs); target = 1.5 x that risk.
    Ours: RSI 14; VWAP from 00:00 UTC; a signal on the first bar all four hold."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    sh, sl = sma(h, 21), sma(l, 21)
    vw, e50, r = _vwap_day(df), ema(c, 50), rsi(c, 14)
    L = _onset((c > sh) & (c > vw) & (c > e50) & (r > 60))
    S = _onset((c < sl) & (c < vw) & (c < e50) & (r < 40)) & ~L
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 1.5), "exit_long": c < sl, "exit_short": c > sh}


# ---------------- SC083 EMA 9/21 cross + RSI 50, 0.5% / 1%, 09:15-15:30 IST, 1H (FMZ) ----------------
def s_fmz_ema921_rsi50_ist(df):
    """FMZ 'Scalping EMA + RSI Strategy (Long & Short)' (Short-term EMA-RSI bidirectional crossover), 1-hour (the
    script's period): long when EMA9 crosses above EMA21 and RSI > 50, short when EMA9 crosses below EMA21 and
    RSI < 50; stop 0.5%, target 1.0% from entry; trades only 09:15-15:30 Asian (Indian market) time. Ours: RSI 14;
    the window in Asia/Kolkata. Differs from TT45 (no RSI, 1:2) and SC042 (RSI 50-70, swing stop) by the exits."""
    c = df["c"].to_numpy(float)
    e9, e21, r = ema(c, 9), ema(c, 21), rsi(c, 14)
    hs, _, _, _, _ = _local(df, "Asia/Kolkata")
    win = (hs >= 9.25) & (hs < 15.5)
    L = cross_up(e9, e21) & (r > 50) & win
    S = cross_down(e9, e21) & (r < 50) & win & ~L
    (sl, ss), (tl, ts) = _pct_levels(c, 0.5, 1.0)
    return L, S, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- SC084 RSI < 40 while EMA9 > EMA21 (RSI > 60 under), 0.5% / 1%, 15m (FMZ) ----------------
def s_fmz_rsi40_ema_trend(df):
    """FMZ 'Aggressive RSI + EMA Strategy with TP/SL' (Trend-following RSI with EMA dual confirmation), 15-minute:
    long when RSI(14) is below 40 while EMA9 is above EMA21; short when RSI is above 60 while EMA9 is under EMA21.
    Stop 0.5%, target 1% from entry. Ours: a signal on the first bar of each such state."""
    c = df["c"].to_numpy(float)
    e9, e21, r = ema(c, 9), ema(c, 21), rsi(c, 14)
    L = _onset((r < 40) & (e9 > e21))
    S = _onset((r > 60) & (e9 < e21)) & ~L
    (sl, ss), (tl, ts) = _pct_levels(c, 0.5, 1.0)
    return L, S, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


def _trend_magic(h, l, c, cci_n=20, atr_n=5, mult=1.0, src=None):
    """Trend Magic: CCI >= 0 -> line = low - mult*ATR (never lowered), CCI < 0 -> high + mult*ATR (never raised).
    Returns the line and its colour (0 = CCI >= 0 / up, 1 = down)."""
    cc = _cci(h, l, c, cci_n) if src is None else _cci(src, src, src, cci_n)
    a = sma(np.maximum(h - l, np.maximum(abs(h - _sh(c)), abs(l - _sh(c)))), atr_n)
    line, col = np.full(len(c), np.nan), np.zeros(len(c))
    for i in range(len(c)):
        if not (np.isfinite(cc[i]) and np.isfinite(a[i])):
            continue
        p = line[i - 1] if i and np.isfinite(line[i - 1]) else None
        if cc[i] >= 0:
            v = l[i] - mult * a[i]
            line[i] = v if p is None or v > p or col[i - 1] == 1 else p
            col[i] = 0
        else:
            v = h[i] + mult * a[i]
            line[i] = v if p is None or v < p or col[i - 1] == 0 else p
            col[i] = 1
    return line, col


# ---------------- SC085 Perfect order EMA45 > SMA90 > SMA180 + Trend Magic flip, SMA90 stop, 1.5R, 5m (FMZ) ----------------
def s_fmz_perfect_order_trendmagic(df):
    """FMZ 'EMA, SMA, CCI, ATR, Perfect Order Moving Average Strategy with Trend Magic' (PakunFX), 5-minute: long when
    the moving averages are in perfect order (EMA45 > SMA90 > SMA180) and Trend Magic changes colour; short in the
    bearish order (EMA45 < SMA90 < SMA180). Stop at the SMA90 at entry, target 1.5 x the risk. Ours: Trend Magic with
    the indicator's usual defaults (CCI 20, ATR 5, multiplier 1); the colour change = to up for longs, to down for
    shorts; the close must be on the right side of the SMA90 so the stop is valid."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e45, s90, s180 = ema(c, 45), sma(c, 90), sma(c, 180)
    _, col = _trend_magic(h, l, c, 20, 5, 1.0)
    pc = _sh(col)
    L = (e45 > s90) & (s90 > s180) & (pc == 1) & (col == 0) & (c > s90)
    S = (e45 < s90) & (s90 < s180) & (pc == 0) & (col == 1) & (c < s90) & ~L
    return L, S, {"stop": ("level", s90, s90), "target": ("R", 1.5)}


# ---------------- SC086 Exp TrendMagic CCI 50 (median) + ATR 5 colour switch, 4H (StockSharp) ----------------
def s_ss_exp_trendmagic(df):
    """StockSharp 'Exp TrendMagic' (port of the MT5 Exp_TrendMagic EA), 4-hour candles by default: TrendMagic from
    CCI(50) on the median price and ATR(5): colour 0 when CCI >= 0 (line = low - ATR), colour 1 when CCI < 0
    (line = high + ATR). Signal bar 1: buy when the colour two bars ago is 0 and on the signal bar is 1, sell on the
    mirror (exactly as the port states it); each switch also closes the opposite position. Stop 1000 points, target
    2000 points. Translation: 1000 points of a 5-digit pair = 100 pips ~ 3.5 x ATR(14) of 4-hour bars (a major's 4H
    ATR is ~28 pips) -> stop 3.5 ATR, target 7 ATR. Ours: the signal is acted on at the close of the bar after the
    switch (signal bar 1)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    _, col = _trend_magic(h, l, c, 50, 5, 1.0, src=(h + l) / 2)
    c1, c2 = _sh(col, 1), _sh(col, 2)
    L = (c2 == 0) & (c1 == 1)
    S = (c2 == 1) & (c1 == 0)
    return L, S & ~L, {"stop": ("atr", 3.5), "target": ("atr", 7.0), "exit_long": S, "exit_short": L}


# ---------------- SC087 OneHrStocTrader: BB width band + Stochastic 5/3/5 turn at set hours, 95 / 200 pips, H1 (StockSharp) ----------------
def s_ss_one_hr_stoc(df):
    """StockSharp 'OneHrStocTrader', hourly candles: trade only when the Bollinger(20, 2) band spread is between 56
    and 158 pips. Long on the bar starting at hour 4 when Stochastic %K (5, %D 3, slowing 5) is below 36 and rising;
    short on the bar starting at hour 0 when %K is above 70 and falling; one position per side. Stop 95 pips, target
    200 pips (40-pip trailing stop not modelled). Translation: a major's 1-hour ATR(14) ~ 12 pips -> spread 4.7-13
    ATR, stop 8 ATR, target 16.7 ATR. Hours taken as UTC (the EA uses broker-server time)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    lo, _, up = _bb(c, 20, 2.0)
    a = atr(h, l, c, 14)
    w = (up - lo) / a
    ok = (w >= 4.7) & (w <= 13.0)
    k, _ = _stoch(h, l, c, 5, 5, 3)
    hs, _, _, _, _ = _local(df, "UTC")
    L = ok & (np.floor(hs) == 4) & (k < 36) & (k > _sh(k))
    S = ok & (np.floor(hs) == 0) & (k > 70) & (k < _sh(k)) & ~L
    return L, S, {"stop": ("atr", 8.0), "target": ("atr", 16.7)}


# ---------------- SC088 Williams AO zero cross + AC 3 bars + RSI 20, BB width filter, 60 / 90 pips, H1 (StockSharp) ----------------
def s_ss_williams_ao_ac(df):
    """StockSharp 'Williams AO + AC', hourly chart by default: Bollinger(20, 2) spread between 40 and 210 points;
    long when RSI(20) > 46, the Awesome Oscillator (11/40) crosses from below to above zero on this bar, and the
    Accelerator (AO - SMA5 of AO) is positive on the last three candles with the latest above the previous; short
    when RSI < 40, AO crosses below zero and AC is negative three candles with the latest lower; entries in the
    window from hour 0 for 20 hours. Stop 60 points, target 90 points (30-point trailing not modelled).
    Translation: points read as pips (a 0.1-pip point would make the 40-210 band filter narrower than any
    Bollinger band); a major's 1-hour ATR(14) ~ 12 pips -> spread 3.3-17.5 ATR, stop 5 ATR, target 7.5 ATR.
    Hours in UTC."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    lo, _, up = _bb(c, 20, 2.0)
    a = atr(h, l, c, 14)
    w = (up - lo) / a
    ok = (w >= 3.3) & (w <= 17.5)
    med = (h + l) / 2
    ao = sma(med, 11) - sma(med, 40)
    ac = ao - sma(ao, 5)
    r = rsi(c, 20)
    hs, _, _, _, _ = _local(df, "UTC")
    win = hs < 20
    acp = (ac > 0) & (_sh(ac) > 0) & (_sh(ac, 2) > 0) & (ac > _sh(ac))
    acn = (ac < 0) & (_sh(ac) < 0) & (_sh(ac, 2) < 0) & (ac < _sh(ac))
    L = ok & win & (r > 46) & cross_up(ao, 0.0) & acp
    S = ok & win & (r < 40) & cross_down(ao, 0.0) & acn & ~L
    return L, S, {"stop": ("atr", 5.0), "target": ("atr", 7.5)}


# ---------------- SC089 EMA 12/26 cross over EMA55 + RSI 35-65 + MACD rising, 0.5% / 1%, 5m (StockSharp Designer) ----------------
def s_ss_designer_ema_rsi_macd(df):
    """StockSharp Designer 'Scalping EMA Cross with RSI and MACD', 5-minute candles: long when EMA12 crosses above
    EMA26, the candle closes above the EMA55 trend line, RSI(14) is between 35 and 65 and the MACD line (12/26) is
    higher than one candle ago; short mirror (cross down, close under EMA55, RSI 35-65, MACD lower). Stop 0.5%,
    target 1% from the fill; no reversal while in a trade."""
    c = df["c"].to_numpy(float)
    e12, e26, e55, r = ema(c, 12), ema(c, 26), ema(c, 55), rsi(c, 14)
    m = e12 - e26
    mid = (r > 35) & (r < 65)
    L = cross_up(e12, e26) & (c > e55) & mid & (m > _sh(m))
    S = cross_down(e12, e26) & (c < e55) & mid & (m < _sh(m)) & ~L
    (sl, ss), (tl, ts) = _pct_levels(c, 0.5, 1.0)
    return L, S, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- SC090 First pullback to EMA50 after the EMA50/SMA200 cross, swing break, fib 161.8 target, H1 (Forex Factory / Metalsmine) ----------------
def s_ff_first_pullback_cross(df):
    """Forex Factory network thread 'First pullback after crossover' (H1; AUDUSD, GBPJPY, XAUUSD): wait for EMA50
    and SMA200 to cross, then the first touch of the EMA50 after the cross (a wick may touch the SMA200 but no
    close beyond it). Draw the Fibonacci over the move (0 = the pullback extreme, 100 = the swing before it): buy
    stop at the 100 level, stop beyond 0, first target 161.8 (break-even there; the rest may run - not modelled).
    Invalid when price closes beyond the SMA200 or the averages cross back. Ours: 100 = the highest high since the
    cross up to the touch (lowest low for sells), 0 = the lowest low since the touch; entry on a close through the
    100 level; target 100 + 0.618 x (100 - 0); one trade per cross."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e50, s200 = ema(c, 50), sma(c, 200)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stl, sts, tgl, tgs = (np.full(n, np.nan) for _ in range(4))
    d, state, ext, lev, pb = 0, 0, np.nan, np.nan, np.nan
    for i in range(1, n):
        if not (np.isfinite(s200[i]) and np.isfinite(s200[i - 1])):
            continue
        if e50[i] > s200[i] and e50[i - 1] <= s200[i - 1]:
            d, state, ext = 1, 1, h[i]
            continue
        if e50[i] < s200[i] and e50[i - 1] >= s200[i - 1]:
            d, state, ext = -1, 1, l[i]
            continue
        if state == 0 or d == 0:
            continue
        if (d > 0 and c[i] < s200[i]) or (d < 0 and c[i] > s200[i]):
            state = 0
            continue
        if state == 1:
            ext = max(ext, h[i]) if d > 0 else min(ext, l[i])
            if (d > 0 and l[i] <= e50[i]) or (d < 0 and h[i] >= e50[i]):
                state, lev, pb = 2, ext, (l[i] if d > 0 else h[i])
            continue
        pb = min(pb, l[i]) if d > 0 else max(pb, h[i])
        if d > 0 and c[i] > lev:
            L[i], stl[i], tgl[i] = True, pb, lev + 0.618 * (lev - pb)
            state = 0
        elif d < 0 and c[i] < lev:
            S[i], sts[i], tgs[i] = True, pb, lev - 0.618 * (pb - lev)
            state = 0
    return L, S, {"stop": ("level", stl, sts), "target": ("level", tgl, tgs)}


# ---------------- SC091 Heiken Ashi 2 candles + ADX 22 / DI + Stochastic 5/3/3 direction, 12.5 / 12.5 pips, M5 (RoboForex) ----------------
def s_robo_ha_adx_stoch(df):
    """RoboForex 'Heiken Ashi + ADX + Stochastic' (M5; EURUSD, GBPUSD): buy when price is above the middle of the
    Heiken Ashi candles with two rising candles in a row, the ADX main line is above 22 with +DI over -DI, and the
    Stochastic (5, 3, 3) is rising; sell when price is below with two falling candles, ADX below 22 (as written)
    with -DI over +DI and the Stochastic falling. Stop 10-15 points, take profit 10-15 points. Translation: 12.5
    pips ~ 2.5 x ATR(14) of 5-minute bars (a major's M5 ATR ~ 5 pips) -> stop 2.5 ATR, target 2.5 ATR. Ours: 'price
    above the middle' = close above the HA body midpoint; two rising candles = two bullish HA candles; ADX 14."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    hc, ho = _ha(df)
    bull, bear = hc > ho, hc < ho
    mid = (hc + ho) / 2
    ax = adx(h, l, c, 14)
    pdi, ndi = _di(h, l, c, 14)
    k, _ = _stoch(h, l, c, 5, 3, 3)
    L = _onset(bull & (_sh(bull) == 1) & (c > mid) & (ax > 22) & (pdi > ndi) & (k > _sh(k)))
    S = _onset(bear & (_sh(bear) == 1) & (c < mid) & (ax < 22) & (ndi > pdi) & (k < _sh(k))) & ~L
    return L, S, {"stop": ("atr", 2.5), "target": ("atr", 2.5)}


# ---------------- SC092 Donchian 20 upper-band crawl, buy the middle-band pullback, out end of day, 15m (TradingView) ----------------
def s_tsg_donchian_mid_pullback(df):
    """TradingView idea 'USDJPY 15M Donchian Channel Strategy' (Tradingstrategyguides), 15-minute: Donchian(20); when
    price crawls along the upper band without going below the middle band, buy the pullback to the middle band
    (second buy at the lower band - not modelled); stop below the lower band (its first flat part); take profit at
    the end of the day or when price breaks below the most recent flat lower band; sells in reverse. Ours: 'crawl' =
    a new 20-bar high in the last 10 bars with no close under the middle band in them; entry = the first touch of
    the middle band; stop = the lower band at entry; out at 21:00 UTC or on a close beyond the lower band."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    up = _s(h).rolling(20).max().to_numpy()
    dn = _s(l).rolling(20).min().to_numpy()
    mid = (up + dn) / 2
    roll = lambda m: _s(np.asarray(m, float)).rolling(10, min_periods=10).max().to_numpy() > 0  # noqa: E731
    crawl_u = roll(h >= up) & ~roll(c < mid)
    crawl_d = roll(l <= dn) & ~roll(c > mid)
    L = (_sh(crawl_u) == 1) & (l <= mid) & (c > dn)
    S = (_sh(crawl_d) == 1) & (h >= mid) & (c < up) & ~L
    hs, _, _, _, _ = _local(df, "UTC")
    pdn, pup = _sh(dn), _sh(up)
    return L, S, {"stop": ("level", dn, up), "target": None, "force_exit": hs >= 21.0,
                  "exit_long": c < pdn, "exit_short": c > pup}


# ---------------- SC093 LV-HP: daily candle colour + Donchian 20 touch, opposite band target, 1:1, 16-20 GMT, M1 (ForTraders) ----------------
def s_ft_lvhp_donchian_m1(df):
    """ForTraders 'LV-HP Scalping System' (M1; EURUSD, USDJPY, GBPUSD): after the London session closes (16:00-20:00
    GMT), in a flat market: when the current daily candle is green and price touches the lower Donchian boundary ->
    buy; when the daily candle is red and price touches the upper boundary -> sell. Take profit at the opposite
    channel boundary, stop at the same distance (1:1); after a stop wait five M1 candles (not modelled). Ours:
    Donchian 20 (the article's 'typically 20-period') of the previous bars; the daily candle = the UTC day's first
    open vs the close; the optional MA filter is left out."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    up = _sh(_s(h).rolling(20).max().to_numpy())
    dn = _sh(_s(l).rolling(20).min().to_numpy())
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    dopen = pd.Series(o).groupby(day).transform("first").to_numpy()
    hs, _, _, _, _ = _local(df, "UTC")
    win = (hs >= 16) & (hs < 20)
    L = win & (c > dopen) & (l <= dn) & (c > dn)
    S = win & (c < dopen) & (h >= up) & (c < up) & ~L
    return L, S, {"stop": ("level", c - (up - c), c + (c - dn)), "target": ("level", up, dn)}


# ---------------- SC094 MTF Ichimoku: over the cloud and the Kijun on 4H and 1H, 2% / 4%, 12-bar cooldown (TradingView) ----------------
def s_tv_mtf_ichimoku(df):
    """TradingView 'MTF Ichimoku Strategy (4H & 1H)' (Gus81): Ichimoku 9/26/52 computed on 4-hour and 1-hour bars;
    long when on both price is above the top of the cloud and above the Kijun-sen; short when on both it is below
    the bottom of the cloud and below the Kijun. Stop 2%, target 4% of the entry price; 12-bar cooldown between
    signals. Ours: the cloud displaced 26 bars as plotted; completed 1H / 4H bars; a signal when both agree for the
    first bar."""
    def ich(g, side):
        hh, ll, cc = g["h"].to_numpy(float), g["l"].to_numpy(float), g["c"].to_numpy(float)
        mx = lambda n: _s(hh).rolling(n).max().to_numpy()  # noqa: E731
        mn = lambda n: _s(ll).rolling(n).min().to_numpy()  # noqa: E731
        ten, kij = (mx(9) + mn(9)) / 2, (mx(26) + mn(26)) / 2
        sa, sb = _sh((ten + kij) / 2, 26), _sh((mx(52) + mn(52)) / 2, 26)
        if side > 0:
            return ((cc > np.fmax(sa, sb)) & (cc > kij)).astype(float)
        return ((cc < np.fmin(sa, sb)) & (cc < kij)).astype(float)
    c = df["c"].to_numpy(float)
    u1, u4 = _htf(df, "1h", lambda g: ich(g, 1)), _htf(df, "4h", lambda g: ich(g, 1))
    d1, d4 = _htf(df, "1h", lambda g: ich(g, -1)), _htf(df, "4h", lambda g: ich(g, -1))
    L0 = _onset((u1 == 1) & (u4 == 1))
    S0 = _onset((d1 == 1) & (d4 == 1))
    any_ = L0 | S0
    cool = _s(any_.astype(float)).shift(1).rolling(12, min_periods=1).max().fillna(0).to_numpy() > 0
    L, S = L0 & ~cool, S0 & ~cool & ~L0
    (sl, ss), (tl, ts) = _pct_levels(c, 2.0, 4.0)
    return L, S, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- helpers added 2026-11-13 (fourth scout run) ----------------
def _psar(h, l, step=0.02, mx=0.2):
    """Wilder's Parabolic SAR (TradingView defaults 0.02 / 0.02 / 0.2). Returns the SAR value and the trend (+1 / -1)
    of every bar (the SAR printed on bar i is computed from bars up to i-1)."""
    n = len(h)
    sar, tr = np.full(n, np.nan), np.zeros(n)
    if n < 3:
        return sar, tr
    up = h[1] >= h[0]
    ep = h[1] if up else l[1]
    s = l[0] if up else h[0]
    af = step
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(mx, af + step)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(mx, af + step)
        sar[i], tr[i] = s, 1 if up else -1
    return sar, tr


def _rmi(c, n=20, mom=5):
    """Relative Momentum Index: RSI computed on close - close[mom] (freqtrade technical.RMI, EMA smoothing)."""
    d = c - _sh(c, mom)
    d = np.nan_to_num(d)
    up, dn = ema(np.clip(d, 0, None), n), ema(np.clip(-d, 0, None), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(up + dn > 0, 100 * up / (up + dn), 50.0)


def _vol_ok(v, n, mult=1.0):
    """Volume above mult x its n-bar average; a feed without real volume (constant) passes."""
    v = np.asarray(v, dtype=float)
    vm = _sh(sma(v, n))
    flat = _s(v).rolling(n).std(ddof=0).to_numpy() == 0
    return (v > mult * vm) | flat


def _shb(x, k=1):
    """Boolean mask shifted k bars (False where unknown)."""
    return np.nan_to_num(_sh(np.asarray(x, dtype=float), k)) > 0


def _bars_since(mask):
    """Bars since `mask` was last True (0 on a True bar, large before the first)."""
    m = np.asarray(mask, dtype=bool)
    idx = np.where(m, np.arange(len(m)), -10**9)
    last = np.maximum.accumulate(idx)
    return np.arange(len(m)) - last


# ---------------- SC095 Trader DNA 5-8-13 EMA + Parabolic SAR, swing stop, 3R, 1H (Medium / YouTube) ----------------
def s_tdna_ema5813_psar(df):
    """YouTube (Trader DNA) '5-8-13 EMA + Parabolic SAR', as written up and backtested on BTC 1H in the Medium
    series 'Backtesting popular strategies from YouTube #2': long when the 8 EMA crosses above the 13 EMA with the
    5 EMA above both (the 5 EMA crossed within the last two candles) and the Parabolic SAR under the candle; short
    on the mirror. Stop at the recent swing low (high), target 3x the risk. Ours: PSAR 0.02/0.2 (not stated), the
    'recent swing' = lowest low / highest high of the last 10 bars (lookback not stated)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e5, e8, e13 = ema(c, 5), ema(c, 8), ema(c, 13)
    sar, tr = _psar(h, l)
    L = cross_up(e8, e13) & (e5 > e8) & (tr > 0)
    S = cross_down(e8, e13) & (e5 < e8) & (tr < 0) & ~L
    return L, S, {"stop": ("lowest", 10), "target": ("R", 3.0)}


# ---------------- SC096 Trade Pro 50 EMA wick rejection over the 200 EMA, 1 ATR stop, 1.5R, 15m (Medium / YouTube) ----------------
def s_tp_ema50_wick_reject(df):
    """YouTube (Trade Pro) '50 EMA' strategy as backtested on EURUSD 15M in the Medium series 'Backtesting popular
    YouTube strategies #4': price above the 200 MA, a candle's low wick pokes below the 50 MA (closing back above
    it) and the next candle stays above -> buy; the mirror under the 200 MA with an upper wick through the 50 MA ->
    sell. Stop 1 x ATR(14), target 1.5 x the risk. Ours: EMAs (the title says EMA), periods on the chart's bars."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e50, e200 = ema(c, 50), ema(c, 200)
    wl = (_sh(l) < _sh(e50)) & (_sh(c) > _sh(e50))
    wh = (_sh(h) > _sh(e50)) & (_sh(c) < _sh(e50))
    L = wl & (c > e50) & (c > e200)
    S = wh & (c < e50) & (c < e200) & ~L
    return L, S, {"stop": ("atr", 1.0), "target": ("R", 1.5)}


# ---------------- SC097 Keltner reversal + EMA 9/21/50 trend, 1.5 / 2 ATR (TradingView) ----------------
def s_tv_keltner_ema_dual(df):
    """TradingView 'Keltner Channel Strategy' (adkhomepros, 2025; gold, NQ, CL; 15m / 1H / 4H). Two entries:
    reversal - long when price crosses below the lower Keltner band, short when it crosses above the upper band;
    trend - long when EMA9 crosses above EMA21 with price above EMA50, short on the mirror. Stop 1.5 x ATR, target
    2 x ATR. Ours: Keltner EMA20 +/- 2 x ATR(10) (the TradingView defaults; lengths not stated); the extra signal
    exits (mid-band cross back, opposite EMA cross) are left to the fixed stop / target."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    mid = ema(c, 20)
    a = atr(h, l, c, 10)
    up, dn = mid + 2 * a, mid - 2 * a
    e9, e21, e50 = ema(c, 9), ema(c, 21), ema(c, 50)
    L = cross_down(c, dn) | (cross_up(e9, e21) & (c > e50))
    S = (cross_up(c, up) | (cross_down(e9, e21) & (c < e50))) & ~L
    return L, S, {"stop": ("atr", 1.5), "target": ("atr", 2.0)}


# ---------------- SC098 Williams %R Zone Scalper: -80 / -20 cross + EMA20 + Choppiness < 38.2 + volume, 5m (TradingView) ----------------
def s_tv_wr_zone_scalper(df):
    """TradingView 'Williams R Zone Scalper v1.0 [BullByte]' (2025, defaults for 5-minute BTC): long when Williams %R(14)
    crosses above -80, short when it crosses below -20, with the default filters: trend EMA20 (price on the trade's
    side), Choppiness Index(12) < 38.2 (trending), volume above its 50-bar average. Stop 1.5 x ATR(14), target 2 x
    ATR(14) (the script's ATR SL/TP inputs). Ours: the script does not say which side of the EMA20 a trade needs,
    so the trend-MA filter is left out (a %R cross out of -80 sits under a 20 EMA almost always); a feed without
    real volume skips the volume filter."""
    h, l, c, v = (df[x].to_numpy(float) for x in "hlcv")
    hh, ll = _s(h).rolling(14).max().to_numpy(), _s(l).rolling(14).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = -100 * (hh - c) / (hh - ll)
    pc = _sh(c)
    tr1 = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    h12, l12 = _s(h).rolling(12).max().to_numpy(), _s(l).rolling(12).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        chop = 100 * np.log10(_s(tr1).rolling(12).sum().to_numpy() / (h12 - l12)) / np.log10(12)
    ok = (chop < 38.2) & _vol_ok(v, 50)
    L = cross_up(wr, -80) & ok
    S = cross_down(wr, -20) & ok & ~L
    return L, S, {"stop": ("atr", 1.5), "target": ("atr", 2.0)}


# ---------------- SC099 Dynamic R:R BTC: EMA 20/50 cross + EMA200 + RSI 60/40, swing stop, 2R, 15m (TradingView) ----------------
def s_tv_dynamic_rr_btc(df):
    """TradingView 'Dynamic R:R Strategy - BTC Optimized' (wojtek_s730ir, 15-minute Bitcoin): long when EMA20 crosses
    above EMA50 with price above EMA200 and RSI above 60; short when EMA20 crosses below EMA50 under EMA200 with RSI
    below 40. Stop at the last swing low / high (or ATR), targets 1R / 2R / 3R. Ours: RSI(14); swing = last
    confirmed 3-bar pivot; one target at 2R (the middle of the three)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c, 14)
    sh_, sl_ = _swings(h, l, 3)
    L = cross_up(e20, e50) & (c > e200) & (r > 60)
    S = cross_down(e20, e50) & (c < e200) & (r < 40) & ~L
    return L, S, {"stop": ("level", sl_, sh_), "target": ("R", 2.0)}


# ---------------- SC100 XAUUSD GoldScalper Pro v8: EMA 8/21/55 + MACD cross + RSI 50-68 + ADX 20, 1 / 5 ATR, 5m (TradingView) ----------------
def s_tv_goldscalper_v8(df):
    """TradingView 'XAUUSD GoldScalper Pro v8+Session' (EMA + MACD + asymmetric R:R, 5-minute gold): long when EMA8 >
    EMA21 > EMA55, EMA55 higher than 10 bars ago, MACD golden cross, RSI 50-68, ADX >= 20 and a bullish candle; short
    on the mirror (RSI 32-50). Stop 1 x ATR, TP1 2 x ATR (half, then stop to entry), TP2 5 x ATR; out on the
    opposite EMA8/21 cross; no new entries 00:00-06:00 Seoul time. Ours: MACD 12/26/9, RSI(14), ADX(14), ATR(14);
    the zero-line filter (side not stated) is left out; the half at 2 ATR is modelled as the stop to break-even
    once 40% of the way to the 5-ATR target."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    e8, e21, e55 = ema(c, 8), ema(c, 21), ema(c, 55)
    m, sg, _ = macd(c)
    r, ax = rsi(c, 14), adx(h, l, c, 14)
    hs, _, _, _, _ = _local(df, "Asia/Seoul")
    ok = ~(hs < 6) & (ax >= 20)
    L = ok & (e8 > e21) & (e21 > e55) & (e55 > _sh(e55, 10)) & cross_up(m, sg) & (r >= 50) & (r <= 68) & (c > o)
    S = ok & (e8 < e21) & (e21 < e55) & (e55 < _sh(e55, 10)) & cross_down(m, sg) & (r >= 32) & (r <= 50) & (c < o) & ~L
    return L, S, {"stop": ("atr", 1.0), "target": ("atr", 5.0), "breakeven": (0.4, 0.0),
                  "exit_long": cross_down(e8, e21), "exit_short": cross_up(e8, e21)}


# ---------------- SC101 Happy Hunter: LHLL / HHHL reversal bar, stop entry past Bar1, 1.3 / 1.9 ATR(120), 1H (BabyPips) ----------------
def s_bp_happy_hunter(df):
    """BabyPips 'Happy Hunter Price Action Trading System' (Type 1, 1-hour chart, GBPNZD): the signal bar (SBar) makes a
    lower high and a lower low than the bar before it (Bar1) and its low is below the lows of the four candles before
    it -> buy stop at Bar1's high (+1 pip), cancelled after 12 hours (2 hours when Bar1 high - SBar low >= 1.5 x
    ATRH); short on the mirror (HHHL, sell stop at Bar1's low). ATRH = ATR(120). Stop entry -/+ 1.3 x ATRH, target
    level entry +/- 1.9 x ATRH (the system trails from there). Ours: the 1-pip filter and spread dropped; the
    signal is the bar whose high touches the buy-stop level (entry next open); the trailing after the target is not
    modelled."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    n = len(c)
    a = atr(h, l, c, 120)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    lvl_l, lvl_s = np.full(n, np.nan), np.full(n, np.nan)
    pend = None
    for i in range(5, n):
        if pend is not None:
            d, lv, exp = pend
            if i > exp:
                pend = None
            elif d > 0 and h[i] >= lv:
                L[i], lvl_l[i], pend = True, lv, None
            elif d < 0 and l[i] <= lv:
                S[i], lvl_s[i], pend = True, lv, None
        if not np.isfinite(a[i]):
            continue
        if h[i] < h[i - 1] and l[i] < l[i - 1] and l[i] < l[i - 5:i - 1].min():
            w = 2 if (h[i - 1] - l[i]) >= 1.5 * a[i] else 12
            pend = (1, h[i - 1], i + w)
        elif h[i] > h[i - 1] and l[i] > l[i - 1] and h[i] > h[i - 5:i - 1].max():
            w = 2 if (h[i] - l[i - 1]) >= 1.5 * a[i] else 12
            pend = (-1, l[i - 1], i + w)
    lv = np.where(L, lvl_l, lvl_s)
    return L, S, {"stop": ("level", lv - 1.3 * a, lv + 1.3 * a), "target": ("level", lv + 1.9 * a, lv - 1.9 * a)}


# ---------------- SC102 Simple XAUUSD scalp: EMA50/200 trend, pullback to EMA50 with RSI < 30 turning, swing stop, 2R (blog) ----------------
def s_pyro_xau_ema_rsi(df):
    """Blog 'Simple XAUUSD Scalping Strategy (Indicators)' (pyroco, 2025; 1-minute or 5-minute gold): after EMA50
    crosses above EMA200, price pulls back near EMA50 and RSI(14) drops below 30; buy when RSI starts rising (or a
    bullish candle). Short on the mirror (EMA50 under EMA200, RSI above 70, turning down). Stop under the recent swing
    low (over the swing high), target 1.5x or 2x the stop. Ours: 'near' = a low within 1 ATR(14) of EMA50 in the last
    5 bars (not defined), RSI under 30 in the last 5 bars and rising now; swing = last confirmed 3-bar pivot; 2R."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e50, e200 = ema(c, 50), ema(c, 200)
    a = atr(h, l, c, 14)
    r = rsi(c, 14)
    sh_, sl_ = _swings(h, l, 3)
    near_l = _s((np.abs(l - e50) <= a).astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    near_h = _s((np.abs(h - e50) <= a).astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    L = (e50 > e200) & near_l & _since(r < 30, 5) & (r > _sh(r))
    S = (e50 < e200) & near_h & _since(r > 70, 5) & (r < _sh(r)) & ~L
    return L, S, {"stop": ("level", sl_, sh_), "target": ("R", 2.0)}


# ---------------- SC103 Gold Time Precise Model: Asian range break in London's first 3 hours + big-candle patterns, 15m (EnergyExch) ----------------
def s_ee_gold_asian_patterns(df):
    """Forex Factory / EnergyExch 'Simple Gold Scalping Module: Time Precise Model' (XAUUSD, 15-minute): Asian session
    range up to the London open (13:00 Bangladesh = 07:00 UTC); trade only if price breaks the Asian high or low in
    the first 3 hours of London. Long: after the break, a big bull candle, small bear consolidation, then a bull
    candle engulfing the bear (after a break of either side: 'manipulation' or 'continuation'), or after a break of
    the Asian high two consecutive big bull candles; short on the mirror. Stop beyond the first candle of the setup;
    take profit +20 pips (or half at +10 and trail). Ours: Asian range 00:00-07:00 UTC, window 07:00-10:00 UTC; 'big'
    candle = body >= 1 x ATR(14) (not defined); 20 pips of gold = 0.5 x the 15m ATR (gold's 15m ATR was ~40 pips)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    a = atr(h, l, c, 14)
    hs, _, _, _, day = _local(df, "UTC")
    asia = hs < 7
    g = pd.DataFrame({"d": day, "h": np.where(asia, h, np.nan), "l": np.where(asia, l, np.nan)})
    ah = g.groupby("d")["h"].transform("max").to_numpy()
    al = g.groupby("d")["l"].transform("min").to_numpy()
    win = (hs >= 7) & (hs < 10)
    hw = pd.Series(np.where(hs >= 7, h, np.nan)).groupby(day).cummax().to_numpy()
    lw = pd.Series(np.where(hs >= 7, l, np.nan)).groupby(day).cummin().to_numpy()
    bh, bl = hw > ah, lw < al
    body = c - o
    bigbull, bigbear = body >= a, -body >= a
    bull, bear = body > 0, body < 0
    eng_bull, eng_bear = _engulf(o, c)
    pat_l = _shb(bigbull, 2) & _shb(bear) & eng_bull
    pat_s = _shb(bigbear, 2) & _shb(bull) & eng_bear
    two_l = bigbull & _shb(bigbull)
    two_s = bigbear & _shb(bigbear)
    L = win & (((bh | bl) & pat_l) | (bh & two_l))
    S = win & (((bh | bl) & pat_s) | (bl & two_s)) & ~L
    first_l = np.where(pat_l, _sh(l, 2), _sh(l))
    first_h = np.where(pat_s, _sh(h, 2), _sh(h))
    return L, S, {"stop": ("level", first_l, first_h), "target": ("atr", 0.5)}


# ---------------- SC104 US30 Parabolic SAR flip + close beyond the last opposite dot, EMA100, 1m (EnergyExch) ----------------
def s_ee_us30_psar_ema100(df):
    """Forex Factory / EnergyExch 'Parabolic SAR + EMA100' (US30, 1-minute): price above EMA100; when the SAR flips to
    buy, mark the last sell-SAR dot; when a candle closes above that line, enter on the next candle (short: under
    EMA100, SAR flips to sell, close below the last buy dot). Only after 13:00 London time (after the 13:30 news).
    Stop 25 pips, target 50 pips, break-even at 30-40 pips. Ours: PSAR 0.02/0.2; window 13:00-21:00 London; 25 / 50
    US30 points = 2 / 4 x the 1-minute ATR(14) (US30's 1m ATR ~12 points); break-even at 70% of the target."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    n = len(c)
    e100 = ema(c, 100)
    sar, tr = _psar(h, l)
    hs, _, _, _, _ = _local(df, "Europe/London")
    win = (hs >= 13) & (hs < 21)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    lvl, d = np.nan, 0
    for i in range(2, n):
        if tr[i] != tr[i - 1] and tr[i] != 0:
            d, lvl = int(tr[i]), sar[i - 1]
        if d > 0 and tr[i] > 0 and np.isfinite(lvl) and c[i] > lvl:
            if win[i] and c[i] > e100[i]:
                L[i] = True
            d = 0
        elif d < 0 and tr[i] < 0 and np.isfinite(lvl) and c[i] < lvl:
            if win[i] and c[i] < e100[i]:
                S[i] = True
            d = 0
    return L, S, {"stop": ("atr", 2.0), "target": ("atr", 4.0), "breakeven": (0.7, 0.0)}


# ---------------- SC105 WHS Z-Score: re-cross of -2 / +2 with the 100 MA's slope, target the 20 MA, 20-bar exit, 15m (WH SelfInvest) ----------------
def s_whs_zscore_trend(df):
    """WH SelfInvest 'Day trading with Z-Score averages' (15-minute, all markets): long when the 100 MA is rising and the
    Z-Score (deviation from the 20-period average in standard deviations) falls below -2.0 and rises back above it;
    short when the 100 MA is falling and the Z-Score rises above +2.0 and falls back below. Target the 20 MA, stop the
    low (high) of the entry candle (or ATR), out after 20 candles (five hours). Ours: SMAs, slope = MA100 above (below)
    its value one bar ago; the 20 MA target is the level at the signal."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    m = sma(c, 20)
    sd = _s(c).rolling(20).std(ddof=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (c - m) / sd
    m100 = sma(c, 100)
    L = (m100 > _sh(m100)) & cross_up(z, -2.0)
    S = (m100 < _sh(m100)) & cross_down(z, 2.0) & ~L
    return L, S, {"stop": ("level", l, h), "target": ("level", m, m), "max_bars": 20}


# ---------------- SC106 Pieper Histo Breakout: HVR < 0.5 + inside / narrow-range bar, close beyond within 4 bars, 2R, 1H (WH SelfInvest) ----------------
def s_whs_histo_breakout(df):
    """WH SelfInvest 'Histo Breakout' (David Pieper, Traders' Magazine; weekly, daily and hourly charts): with the Historic
    Volatility Ratio below 0.5, a signal candle is an inside candle (body within the previous candle's range) or a narrow
    range candle (range smaller than each of the three before); long when price closes above the signal candle's high
    within four candles, short when it closes below its low. Stop the other end of the signal candle, target 2 x the
    risk, break-even once the profit reaches 1R. Ours: HVR = 6-bar / 100-bar historical volatility of log returns
    (Pieper's usual 6/100; lookbacks not stated); break-even at half way to the 2R target."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    n = len(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(c / _sh(c))
    hvr = _s(lr).rolling(6).std().to_numpy() / _s(lr).rolling(100).std().to_numpy()
    rng = h - l
    inside = (np.maximum(o, c) <= _sh(h)) & (np.minimum(o, c) >= _sh(l))
    nr = (rng < _sh(rng, 1)) & (rng < _sh(rng, 2)) & (rng < _sh(rng, 3))
    sig = (hvr < 0.5) & (inside | nr)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stl, sts = np.full(n, np.nan), np.full(n, np.nan)
    k = -1
    for i in range(1, n):
        if k >= 0 and i - k <= 4:
            if c[i] > h[k]:
                L[i], stl[i], k = True, l[k], -1
            elif c[i] < l[k]:
                S[i], sts[i], k = True, h[k], -1
        if sig[i]:
            k = i
    return L, S, {"stop": ("level", stl, sts), "target": ("R", 2.0), "breakeven": (0.5, 0.0)}


# ---------------- SC107 News 1m: impulse candle, enter after the 3-4 minute correction, stop past it, target the impulse extreme (TradingView) ----------------
def s_tv_news_impulse_correction(df):
    """TradingView idea 'RIGHT WAY TO TRADE NEWS (1m chart)' (Marenno, EURUSD): on the 1-minute chart wait for the
    news impulse candle, then the short correction (usually 3-5 minutes); enter in the impulse direction after the
    3rd-4th minute's close; stop beyond the correction's extreme; target a new extreme. Ours: the impulse = the
    release bar (agents/event_strategies calendar; weekdays 08:30 New York without a symbol) with a body >= 0.5 x
    ATR(14); entry at the close of the 3rd or 4th bar after it (the first that qualifies) when that close is back
    against the impulse (a correction) while no close of the correction gave back the whole impulse candle; stop
    beyond the correction's extreme; target = the impulse's extreme (the new extreme starts there)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    o = _open(df)
    n = len(c)
    a = atr(h, l, c, 14)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    st, tg = np.full(n, np.nan), np.full(n, np.nan)
    for k in _release_bars(df):
        if k < 1 or not np.isfinite(a[k - 1]):
            continue
        body = c[k] - o[k]
        for j in (k + 3, k + 4):
            if j >= n:
                break
            if body >= 0.5 * a[k - 1] and c[j] < c[k] and c[k + 1:j + 1].min() > o[k]:
                L[j], st[j], tg[j] = True, l[k + 1:j + 1].min(), h[k:j + 1].max()
                break
            if -body >= 0.5 * a[k - 1] and c[j] > c[k] and c[k + 1:j + 1].max() < o[k]:
                S[j], st[j], tg[j] = True, h[k + 1:j + 1].max(), l[k:j + 1].min()
                break
    return L, S, {"stop": ("level", st, st), "target": ("level", tg, tg)}


# ---------------- SC108 News fade: release spike + reversal candle, break of its wick, stop the spike, target the pre-release close, 15-min exit, 5m (JournalPlus) ----------------
def s_jp_news_fade_reversal(df):
    """JournalPlus 'News Fade Trading Strategy' (macro releases NFP / CPI / FOMC on the 5-minute chart): the first candle
    after the release closes as a reversal pattern (shooting star / bearish engulfing / inside-bar rejection after a
    spike up; hammer / bullish engulfing after a spike down) on volume 3-5x its 20-bar average; enter on the break of
    the reversal candle's opposite wick; stop beyond the spike's extreme; T1 VWAP / nearby level, T2 the prior close
    (gap fill); exit at market if nothing happens within 15 minutes. Ours: the release bar (event calendar) is the
    spike (close beyond the pre-release close, volume 3x its 20-bar average when real volume exists); the next bar is
    the reversal candle: engulfing the spike's body, a shooting star / hammer (wick >= 2x body, close in the far
    third) or an inside bar closing against the spike; the close beyond its opposite end within the next 3 bars is
    the entry; the structural-level check is left out; target = the pre-release close; max 3 bars (15 minutes)."""
    h, l, c, v = (df[x].to_numpy(float) for x in "hlcv")
    o = _open(df)
    n = len(c)
    vm = _sh(sma(v, 20))
    flat = _s(v).rolling(20).std(ddof=0).to_numpy() == 0
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    st, tg = np.full(n, np.nan), np.full(n, np.nan)
    for k in _release_bars(df):
        if k < 1 or k + 1 >= n:
            continue
        if not (flat[k] or v[k] >= 3 * vm[k]):
            continue
        pre, r_ = c[k - 1], k + 1
        if r_ + 1 >= n:
            continue
        body, rng = c[r_] - o[r_], h[r_] - l[r_]
        uw, lw = h[r_] - max(o[r_], c[r_]), min(o[r_], c[r_]) - l[r_]
        inside = h[r_] <= h[k] and l[r_] >= l[k]
        if c[k] > pre:      # spike up -> look for a bearish reversal candle
            rev = body < 0 and (c[r_] <= min(o[k], c[k]) or inside
                                or (rng > 0 and uw >= 2 * abs(body) and c[r_] - l[r_] <= rng / 3))
            d = -1 if rev else 0
        elif c[k] < pre:
            rev = body > 0 and (c[r_] >= max(o[k], c[k]) or inside
                                or (rng > 0 and lw >= 2 * abs(body) and h[r_] - c[r_] <= rng / 3))
            d = 1 if rev else 0
        else:
            d = 0
        if d == 0:
            continue
        for j in range(r_ + 1, min(n, r_ + 4)):
            if d < 0 and c[j] < l[r_]:
                S[j], st[j], tg[j] = True, h[k:j + 1].max(), pre
                break
            if d > 0 and c[j] > h[r_]:
                L[j], st[j], tg[j] = True, l[k:j + 1].min(), pre
                break
    return L, S, {"stop": ("level", st, st), "target": ("level", tg, tg), "max_bars": 3}


# ---------------- SC109 CPI 08:30-09:00 ET range breakout, stop over the midpoint, target the prior day's level, 5m (DayTradingToolkit) ----------------
def s_dtt_cpi_range_break(df):
    """DayTradingToolkit 'The Trader's Playbook: How to Day Trade the CPI Report' (QQQ / SPY): flat into the 08:30 ET
    release, no trades 08:30-09:00, mark that half hour's high and low; trade the breakout of that range; stop just
    beyond the range midpoint; primary target a prior-day key level (support for a short, resistance for a long).
    Ours: 08:30 ET releases from the event calendar (weekdays 08:30 without a symbol), the first close beyond the
    range until 12:00 ET, one trade per release; target = the previous UTC day's low (short) / high (long); the
    source gives no chart interval for the entry (the range is clock-defined) - run on 5-minute bars."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    n = len(c)
    ph, pl, _ = _prev_day(df)
    ny = df.index.tz_convert("America/New_York")
    hr = ny.hour.to_numpy() + ny.minute.to_numpy() / 60.0
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    mid = np.full(n, np.nan)
    for k in _release_bars(df):
        if abs(hr[k] - 8.5) > 0.01:
            continue
        j0 = k
        while j0 < n and hr[j0] < 9.0 and ny[j0].date() == ny[k].date():
            j0 += 1
        if j0 >= n or j0 == k:
            continue
        rh, rl = h[k:j0].max(), l[k:j0].min()
        md = (rh + rl) / 2
        for j in range(j0, n):
            if ny[j].date() != ny[k].date() or hr[j] >= 12.0:
                break
            if c[j] > rh:
                L[j], mid[j] = True, md
                break
            if c[j] < rl:
                S[j], mid[j] = True, md
                break
    return L, S, {"stop": ("level", mid, mid), "target": ("level", ph, pl)}


# ---------------- SC110 freqtrade Hacklemore: RMI + MACD + SAR + BB-low breakout 2-10 bars ago, -8.5% / ROI 1.5%, 5m (GitHub) ----------------
def s_ft_hacklemore(df):
    """freqtrade 'Hacklemore' (5-minute crypto, long only): buy when the 12-bar high rose in the last 3 bars (up-trend),
    RMI(20, 5) >= 55 and at or above its 3-bar average, MACD above its signal with a positive histogram, the low touched
    the lower Bollinger band(20) 2-10 bars ago, two higher closes in a row, the SAR under the last two closes and volume
    under 30x its 24-bar average. Stoploss -8.5%; minimal_roi 15% at once, 2.5% after 5 minutes, 1.5% after 10
    minutes. Ours: the ROI table as a 1.5% target (it applies from the third 5-minute bar); the loss-cutting sell
    signal (down-trend + RMI(9,3) < 50 while losing) as exit_long."""
    h, l, c, v = (df[x].to_numpy(float) for x in "hlcv")
    n = len(c)
    rs, rf = _rmi(c, 20, 5), _rmi(c, 9, 3)
    m, sg, hist = macd(c)
    mx, mn = _s(h).rolling(12).max().to_numpy(), _s(l).rolling(12).min().to_numpy()
    upt = _s((mx > _sh(mx)).astype(float)).rolling(3, min_periods=1).sum().to_numpy() > 0
    dnt = _s((mn < _sh(mn)).astype(float)).rolling(3, min_periods=1).sum().to_numpy() > 0
    bl, _, _ = _bb(c, 20, 2.0)
    since = _bars_since(l <= bl)
    sar, tr = _psar(h, l)
    vm = _sh(_s(v).rolling(24).mean().to_numpy())
    L = (upt & (rs >= 55) & (hist > 0) & (since >= 2) & (since <= 10) & (v < vm * 30) & (m > sg)
         & (rs >= _s(rs).rolling(3).mean().to_numpy()) & (c > _sh(c)) & (_sh(c) > _sh(c, 2))
         & (sar < c) & (_sh(sar) < _sh(c)))
    (sl_, _), (tl, _) = _pct_levels(c, 8.5, 1.5)
    return L, np.zeros(n, bool), {"stop": ("level", sl_, sl_), "target": ("level", tl, tl),
                                  "exit_long": dnt & (rf < 50)}


# ---------------- pip -> ATR translation used by SC111-SC123 ----------------
_EU_ATR_PIPS = ((1.5, 1.5), (5.0, 3.0), (15.0, 5.5), (60.0, 11.0), (240.0, 22.0))


def _pips_atr(df, pips):
    """Fixed EUR/USD pips -> ATR(14) multiples of this bar's timeframe. Typical EUR/USD ATR(14): 1m ~1.5 pips,
    5m ~3, 15m ~5.5, 1h ~11, 4h ~22 (the nearest timeframe at or above the bar is used)."""
    bm = _bar_minutes(df)
    ref = _EU_ATR_PIPS[-1][1]
    for mins, p in _EU_ATR_PIPS:
        if bm <= mins + 1e-9:
            ref = p
            break
    return float(pips) / ref


# ---------------- SC111 Tricolor M1 (ForTraders) ----------------
def s_fort_tricolor_m1(df):
    """ForTraders 'Tricolor - Scalping Strategy for M1' (EUR/USD, GBP/USD, European session): EMA63 > EMA102 >
    EMA165, the slow Stochastic (30,6,18) has gone below 10 and the fast Stochastic (15,3,9) crosses 20 upward ->
    buy right after the cross; mirror (EMA63 < 102 < 165, slow stochastic above 90, fast crosses 80 down) -> sell.
    Stop 5 points + spread, target 10 points + spread; EMAs crossed / mixed -> no trade.
    Ours: 'has gone below 10' = within the last 40 bars; European session 06:00-17:00 UTC; points read as pips:
    5 pips ~ 3.3 ATR(14) of 1-minute EUR/USD, target 2R (10 pips)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e1, e2, e3 = ema(c, 63), ema(c, 102), ema(c, 165)
    ks, _ = _stoch(h, l, c, 30, 6, 18)
    kf, _ = _stoch(h, l, c, 15, 3, 9)
    hs = _local(df, "UTC")[0]
    sess = (hs >= 6.0) & (hs < 17.0)
    lo_seen = _since(ks < 10, 40) | (ks < 10)
    hi_seen = _since(ks > 90, 40) | (ks > 90)
    L = sess & (e1 > e2) & (e2 > e3) & lo_seen & cross_up(kf, np.full(len(c), 20.0))
    S = sess & (e1 < e2) & (e2 < e3) & hi_seen & cross_down(kf, np.full(len(c), 80.0))
    return L, S & ~L, {"stop": ("atr", _pips_atr(df, 5)), "target": ("R", 2.0)}


# ---------------- SC112 ADX crossing its MA9 + MACD sign (ForTraders) ----------------
def s_fort_adx_ma_macd(df):
    """ForTraders 'Forex Strategy Using ADX and MACD Indicators' (M15, EUR/USD): the ADX(14) line crosses its
    9-period moving average from below; MACD (5,25,9) above zero -> buy, below zero -> sell. Stop 30 points,
    target 60 points; an opposite signal closes the trade. Ours: MA = SMA9 of the ADX, MACD = its main line;
    points read as pips: 30 pips ~ 5.5 ATR(14) of 15-minute EUR/USD, target 2R; exit on the opposite signal."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    ax = adx(h, l, c, 14)
    am = sma(np.nan_to_num(ax, nan=0.0), 9)
    m, _, _ = macd(c, 5, 25, 9)
    trig = cross_up(ax, am)
    L = trig & (m > 0)
    S = trig & (m < 0)
    return L, S & ~L, {"stop": ("atr", _pips_atr(df, 30)), "target": ("R", 2.0), "exit_long": S, "exit_short": L}


# ---------------- SC113 '10 points' MACD + Stochastic M1 (Engineer the Forex) ----------------
def s_etf_10points_m1(df):
    """Engineer the Forex 'Scalping system 10 points' (M1, medium/high-volatility majors, European and American
    sessions): MACD (13,26,9) histogram above zero and the Stochastic (5,3,3) comes back above 20 after a dip
    under it -> buy; histogram below zero and the stochastic falls back under 80 -> sell. Stop one point beyond the
    nearest local low / high, target 10 points. Ours: local low / high = 5-bar extreme -0.2 / +0.2 ATR; sessions
    07:00-20:00 UTC; 10 points read as 10 pips ~ 6.5 ATR(14) of 1-minute EUR/USD."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    _, _, hist = macd(c, 13, 26, 9)
    k, _ = _stoch(h, l, c, 5, 3, 3)
    hs = _local(df, "UTC")[0]
    sess = (hs >= 7.0) & (hs < 20.0)
    L = sess & (hist > 0) & cross_up(k, np.full(len(c), 20.0))
    S = sess & (hist < 0) & cross_down(k, np.full(len(c), 80.0))
    lo5 = _s(l).rolling(5).min().to_numpy() - 0.2 * a
    hi5 = _s(h).rolling(5).max().to_numpy() + 0.2 * a
    t = _pips_atr(df, 10) * a
    return L, S & ~L, {"stop": ("level", lo5, hi5), "target": ("level", c + t, c - t)}


# ---------------- SC114 Ranger: Parabolic SAR + Stochastic 5/5/5 (Trading Guide) ----------------
def s_tg_ranger_sar_stoch(df):
    """Trading Guide (GitBook) 'Ranger', M15 and above: Parabolic SAR (0.04, 0.1) below price and the Stochastic
    (5,5,5) lines cross in the zone below 25 and turn up -> buy at the next candle's open; SAR above price and the
    lines turn down from above 75 -> sell. Stop at the latest local low / high, no more than 20 points; target 2-3x
    the stop; also out when the stochastic reaches the opposite zone (75 / 25).
    Ours: the stochastic cross in the zone may come up to 3 bars before the SAR side agrees; local low / high = 5-bar extreme; the 20-point cap read as 20 pips ~ 3.6 ATR(14) of 15-minute EUR/USD
    (a wider stop -> no trade); target 2.5R."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    sar, _ = _psar(h, l, 0.04, 0.1)
    k, d = _stoch(h, l, c, 5, 5, 5)
    lo5 = _s(l).rolling(5).min().to_numpy()
    hi5 = _s(h).rolling(5).max().to_numpy()
    cap = _pips_atr(df, 20) * a
    xu = cross_up(k, d) & (_sh(k) < 25) & (_sh(d) < 25)
    xd = cross_down(k, d) & (_sh(k) > 75) & (_sh(d) > 75)
    L = _onset((sar < c) & (xu | _since(xu, 3)) & (k > _sh(k))) & (c - lo5 <= cap) & (c > lo5)
    S = _onset((sar > c) & (xd | _since(xd, 3)) & (k < _sh(k))) & (hi5 - c <= cap) & (hi5 > c)
    return L, S & ~L, {"stop": ("level", lo5, hi5), "target": ("R", 2.5), "exit_long": k > 75, "exit_short": k < 25}


# ---------------- SC115 PinkFlint news straddle, 5-point target (ForTraders) ----------------
def s_fort_pinkflint_news(df):
    """ForTraders 'PinkFlint Scalping Trading Strategy', news approach (M5 chart; EUR/USD, GBP/USD, AUD/USD,
    EUR/JPY, USD/CAD, NZD/USD): 5 minutes before the release place a buy stop and a sell stop 20 points from the
    price; stop 30 points, take the profit at +5 points (break-even after +5).
    Ours: reference = the close 5 minutes before the release; pips -> ATR(14) of
    5-minute EUR/USD before the release (~3 pips): offset 6.7 ATR, stop 10 ATR, target 5/30 of the stop.
    The pending orders: the first bar whose high / low reaches a level within 3 hours fills it (entry next open).
    Releases: FTMO high-impact calendar for the symbol (else 08:30 New York weekdays)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    n = len(c)
    a = atr(h, l, c, 14)
    bm = _bar_minutes(df)
    pre, win = max(1, int(round(5 / bm))), max(1, int(round(180 / bm)))
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for k0 in _release_bars(df):
        ks = k0 - pre
        if ks < 20 or k0 >= n:
            continue
        ref, u = c[ks], a[ks - 1] * _pips_atr(df, 1)  # one pip in price, from the pre-release ATR
        if not np.isfinite(u) or u <= 0:
            continue
        up, dn = ref + 20 * u, ref - 20 * u
        for i in range(ks + 1, min(n, k0 + win)):
            if h[i] >= up and l[i] > dn:
                L[i], sl_l[i] = True, max(c[i], up) - 30 * u
                break
            if l[i] <= dn and h[i] < up:
                S[i], sl_s[i] = True, min(c[i], dn) + 30 * u
                break
            if h[i] >= up and l[i] <= dn:
                break
    return L, S, {"stop": ("level", sl_l, sl_s), "target": ("R", 5.0 / 30.0)}


# ---------------- SC116 Stochastic + MACD agreement, state exit (The Right Trader) ----------------
def s_trt_stoch_macd_state(df):
    """The Right Trader 'Stochastic and MACD Strategy: When Both Tools Agree' (daily swing; 15-minute day-trade
    variant): slow Stochastic (14,3,3) %K above %D with %K at or under 80, and the MACD (12,26,9) line above its
    signal - on the first close where all three hold -> long; mirror (%K under %D, %K at or over 20, MACD under
    signal) -> short. Stop beyond the lowest low / highest high of the entry bar and the two before it. Exit on the
    first close where either condition fails (the 80 / 20 caps block entries only), or at the session close.
    Ours: session close = end of the UTC day for the intraday variant."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    k, d = _stoch(h, l, c, 14, 3, 3)
    m, sg, _ = macd(c, 12, 26, 9)
    up_state = (k > d) & (m > sg)
    dn_state = (k < d) & (m < sg)
    L = _onset(up_state & (k <= 80))
    S = _onset(dn_state & (k >= 20))
    lo3 = _s(l).rolling(3).min().to_numpy()
    hi3 = _s(h).rolling(3).max().to_numpy()
    hs = _local(df, "UTC")[0]
    bm = _bar_minutes(df)
    eod = hs >= 24.0 - bm / 60.0 - 1e-9
    return L, S & ~L, {"stop": ("level", lo3, hi3), "target": None, "exit_long": ~up_state, "exit_short": ~dn_state,
                       "force_exit": eod}


# ---------------- SC117 EMA ribbon 2-15 vs 44-74 scalp (Engineer the Forex) ----------------
def s_etf_ema_ribbon_scalp(df):
    """Engineer the Forex 'Scalping strategy with EMA' (M1 recommended, also M5; EUR/USD, GBP/USD, AUD/USD,
    USD/CAD): pink ribbon = EMA 2..15, blue = EMA 17..41 (odd), green = EMA 44..74 (step 3). The pink ribbon crosses
    the green one -> trade in the direction of the cross. M1: stop 12 points, target 5-7 points; M5: stop 18,
    target 12-20. Adding on touches of the blue/green line is not modelled.
    Ours: ribbon = the mean of its EMAs; points read as pips; 1-minute: stop 12 pips ~ 8 ATR, target 6 pips
    (0.5R); 5-minute and slower: stop 18 pips ~ 6 ATR(5m), target 16 pips (0.89R)."""
    c = df["c"].to_numpy(float)
    pink = np.mean([ema(c, p) for p in range(2, 16)], axis=0)
    green = np.mean([ema(c, p) for p in range(44, 75, 3)], axis=0)
    L = cross_up(pink, green)
    S = cross_down(pink, green)
    if _bar_minutes(df) <= 1.5:
        stop, tr = _pips_atr(df, 12), 6.0 / 12.0
    else:
        stop, tr = _pips_atr(df, 18), 16.0 / 18.0
    return L, S & ~L, {"stop": ("atr", stop), "target": ("R", tr)}


# ---------------- SC118 RSI(7) extremes 20/80, out at RSI 50 (FXOpen) ----------------
def s_fxo_rsi7_extremes(df):
    """FXOpen 'Ultra-Fast Scalping Strategies in Forex' #2 RSI Extremes (1-minute): RSI(7) dips below 20 and
    moves back above -> buy; rises above 80 and falls back below -> sell. Stop a few pips beyond the entry candle;
    take the profit as the RSI approaches 50. Ours: stop = the signal candle's low / high -/+ 2 pips (~1.3 ATR of
    1-minute EUR/USD); exit when the RSI reaches 50 (48 / 52 = 'approaches')."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    r = rsi(c, 7)
    L = cross_up(r, np.full(len(c), 20.0))
    S = cross_down(r, np.full(len(c), 80.0))
    off = _pips_atr(df, 2) * a
    return L, S & ~L, {"stop": ("level", l - off, h + off), "target": None, "exit_long": r >= 48, "exit_short": r <= 52}


# ---------------- SC119 Stochastic 14/3/3 quick signal, out at the opposite extreme (FXOpen) ----------------
def s_fxo_stoch_quick(df):
    """FXOpen 'Ultra-Fast Scalping Strategies in Forex' #3 Stochastic Quick Signal (1-minute): %K crosses above %D
    with both under 20 -> buy; %K crosses below %D with both over 80 -> sell. Stop a few pips away / beyond the
    swing; take the profit when the stochastic reaches the opposite extreme. Ours: stop = 5-bar swing -/+ 2 pips
    (~1.3 ATR of 1-minute EUR/USD); exit when %K reaches 80 (long) / 20 (short); divergence exit not modelled."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    k, d = _stoch(h, l, c, 14, 3, 3)
    L = cross_up(k, d) & (k < 20) & (d < 20)
    S = cross_down(k, d) & (k > 80) & (d > 80)
    off = _pips_atr(df, 2) * a
    lo5 = _s(l).rolling(5).min().to_numpy() - off
    hi5 = _s(h).rolling(5).max().to_numpy() + off
    return L, S & ~L, {"stop": ("level", lo5, hi5), "target": None, "exit_long": k >= 80, "exit_short": k <= 20}


# ---------------- SC120 5m EMA20 pullback + Stochastic RSI + VWAP side (XS) ----------------
def s_xs_5m_ema20_stochrsi_vwap(df):
    """XS '5-minute Scalping Strategy' (Maki Miyai; forex, crypto, indices): trade the VWAP side (above -> buys,
    below -> sells); price holds above the 20 EMA, pulls back toward it, and a bullish candle closes back above it
    with the Stochastic RSI turning up from below 20 -> buy (mirror -> sell). Stop just beyond the recent swing,
    target 1:1 to 1:1.5; out at once on a clear close on the other side of the EMA; trade 08:00-12:00 New York.
    Ours: pullback = a low at / under the EMA20 in the last 3 bars after 3 closes above it before that; StochRSI
    (14,14,3,3) %K crosses %D with %D under 20 within the last 5 bars; swing = 5-bar low / high; target 1.25R;
    'clear close' = a close beyond the EMA by 0.25 ATR."""
    o, h, l, c = _open(df), *(df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    e = ema(c, 20)
    vw = _day_vwap_local(df, "America/New_York")
    k, d = _stochrsi(c)
    st_up = _since(cross_up(k, d) & (d < 20), 5) | (cross_up(k, d) & (d < 20))
    st_dn = _since(cross_down(k, d) & (d > 80), 5) | (cross_down(k, d) & (d > 80))
    hs = _local(df, "America/New_York")[0]
    sess = (hs >= 8.0) & (hs < 12.0)
    above_before = (_sh(c, 4) > _sh(e, 4)) & (_sh(c, 5) > _sh(e, 5)) & (_sh(c, 6) > _sh(e, 6))
    below_before = (_sh(c, 4) < _sh(e, 4)) & (_sh(c, 5) < _sh(e, 5)) & (_sh(c, 6) < _sh(e, 6))
    touch_dn = _since(l <= e, 3) | (l <= e)
    touch_up = _since(h >= e, 3) | (h >= e)
    L = sess & (c > vw) & above_before & touch_dn & (c > e) & (c > o) & st_up
    S = sess & (c < vw) & below_before & touch_up & (c < e) & (c < o) & st_dn
    lo5 = _s(l).rolling(5).min().to_numpy()
    hi5 = _s(h).rolling(5).max().to_numpy()
    return L, S & ~L, {"stop": ("level", lo5, hi5), "target": ("R", 1.25),
                       "exit_long": c < e - 0.25 * a, "exit_short": c > e + 0.25 * a}


# ---------------- SC121 MA Crossover ADX: rising EMA39 + ADX33 > 22 + DI (StockSharp / MQL) ----------------
def s_ss_ema39_adx33(df):
    """StockSharp 'MA Crossover ADX Strategy' (port of the MetaTrader EA MA_Crossover_ADX), 1-hour default: EMA39
    rising on the last two steps, the previous close above the previous EMA, ADX(33) > 22 and +DI > -DI -> buy;
    mirror -> sell; the opposite conditions close the trade. Stop 400 points, target 900 points.
    Ours: the documented rules (the sample code differs); 400 / 900 points on a 5-digit EUR/USD = 40 / 90 pips ->
    stop 40 pips ~ 3.6 ATR(14) of 1-hour EUR/USD, target 2.25R; entry only on the first bar of the condition."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e = ema(c, 39)
    ax = adx(h, l, c, 33)
    pdi, mdi = _di(h, l, c, 33)
    up = (e - _sh(e) > 0) & (_sh(e) - _sh(e, 2) > 0) & (_sh(c) > _sh(e)) & (ax > 22) & (pdi > mdi)
    dn = (e - _sh(e) < 0) & (_sh(e) - _sh(e, 2) < 0) & (_sh(c) < _sh(e)) & (ax > 22) & (pdi < mdi)
    L, S = _onset(up), _onset(dn)
    return L, S & ~L, {"stop": ("atr", _pips_atr(df, 40)), "target": ("R", 2.25), "exit_long": dn, "exit_short": up}


# ---------------- SC122 MA Price Cross 160, 1:3 (StockSharp / JBlanked) ----------------
def s_ss_ma160_price_cross(df):
    """StockSharp 'MA Price Cross Strategy' (from the MQL4 EA by JBlanked, 2023): the close crosses the 160-period
    moving average upward -> buy, downward -> sell (an opposite position is flattened first); stop 200 points,
    target 600 points; entries only 01:00-22:00. Ours: SMA160 on closes (the documented type); 200 / 600 points =
    20 / 60 pips on a 5-digit EUR/USD -> stop in ATR(14) of the bar's timeframe, target 3R; window in UTC."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    m = sma(c, 160)
    hs = _local(df, "UTC")[0]
    win = (hs >= 1.0) & (hs < 22.0)
    L = win & cross_up(c, m)
    S = win & cross_down(c, m)
    return L, S & ~L, {"stop": ("atr", _pips_atr(df, 20)), "target": ("R", 3.0), "exit_long": S, "exit_short": L}


# ---------------- SC123 Connors S&P momentum day trade: MA20 + RSI(3) pullback (TradingMarkets) ----------------
def s_connors_ma20_rsi3_intraday(df):
    """Larry Connors, TradingMarkets 'Three Indicators I Combine To Find Great Intraday Entry Points In The S&P'
    (20-minute bars; 5, 10 or 15-minute also; S&P futures, also bonds, currencies): bars above the 20-period MA and
    RSI(3) under 30 -> buy one tick above the previous bar's high on the next two bars only; mirror (below the MA,
    RSI(3) over 70, sell one tick under the previous low). Risk no more than the previous bar's size; exit: half
    off at 1R, stop to break-even, trail the rest.
    Ours: the buy stop = a close above the previous bar's high within the two bars after the setup bar (filled at
    the next open); stop = signal close minus the previous bar's range; break-even at 1R, target 2R (the trailed
    half approximated)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    m = sma(c, 20)
    r = rsi(c, 3)
    set_l = (c > m) & (r < 30)
    set_s = (c < m) & (r > 70)
    L = _since(set_l, 2) & (c > _sh(h)) & (c > m)
    S = _since(set_s, 2) & (c < _sh(l)) & (c < m)
    rng = _sh(h) - _sh(l)
    L, S = _onset(L), _onset(S) & ~L
    return L, S, {"stop": ("level", c - rng, c + rng), "target": ("R", 2.0), "breakeven": (0.5, 0.0)}



# ======================= scout night 2026-11-14: SC124-SC149 =======================
def _tema(x, n):
    e1 = ema(x, n)
    e2 = ema(e1, n)
    return 3 * e1 - 3 * e2 + ema(e2, n)


def _stoch_carry(x, n):
    """100 * (x - lowest(x, n)) / range, carrying the last valid value forward when the range is 0 (Pine STC style)."""
    s = _s(x)
    lo, hi = s.rolling(n).min(), s.rolling(n).max()
    rng = hi - lo
    raw = (100 * (s - lo) / rng.where(rng > 0)).ffill().fillna(50.0)
    return raw.to_numpy()


def _stc_ema(c, fast, slow, cycle):
    """Schaff Trend Cycle, EMA-3 smoothed version (FMZ 435160)."""
    m = ema(c, fast) - ema(c, slow)
    d = ema(_stoch_carry(m, cycle), 3)
    return np.clip(ema(_stoch_carry(d, cycle), 3), 0, 100)


def _stc_half(c, length, fast, slow):
    """Schaff Trend Cycle, classic 0.5 factor smoothing (FMZ 429472)."""
    m = ema(c, fast) - ema(c, slow)
    s1 = _stoch_carry(m, length)
    d = np.empty(len(c))
    for i in range(len(c)):
        d[i] = s1[i] if i == 0 else d[i - 1] + 0.5 * (s1[i] - d[i - 1])
    s2 = _stoch_carry(d, length)
    out = np.empty(len(c))
    for i in range(len(c)):
        out[i] = s2[i] if i == 0 else out[i - 1] + 0.5 * (s2[i] - out[i - 1])
    return out


def _ut_trail(c, a, mult):
    """UT-Bot ATR trailing stop line and the regime (+1 after a close crosses above it, -1 below, 0 before any)."""
    n = len(c)
    t = np.zeros(n)
    pos = np.zeros(n)
    nl = mult * np.nan_to_num(a)
    for i in range(n):
        if i == 0:
            t[i] = c[i] - nl[i]
            continue
        p = t[i - 1]
        if c[i] > p and c[i - 1] > p:
            t[i] = max(p, c[i] - nl[i])
        elif c[i] < p and c[i - 1] < p:
            t[i] = min(p, c[i] + nl[i])
        else:
            t[i] = c[i] - nl[i] if c[i] > p else c[i] + nl[i]
        if c[i - 1] < p and c[i] > t[i]:
            pos[i] = 1
        elif c[i - 1] > p and c[i] < t[i]:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]
    return t, pos


def _mfi(h, l, c, v, n):
    tp = (h + l + c) / 3
    mf = tp * np.where(v > 0, v, 1.0)
    dtp = np.diff(tp, prepend=tp[0])
    pos = _s(np.where(dtp > 0, mf, 0.0)).rolling(n).sum().to_numpy()
    neg = _s(np.where(dtp < 0, mf, 0.0)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pos + neg > 0, 100 * pos / (pos + neg), 50.0)


def _wpr(h, l, c, n):
    hh, ll = _s(h).rolling(n).max().to_numpy(), _s(l).rolling(n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hh > ll, -100 * (hh - c) / (hh - ll), -50.0)


def _kalman_hull(x, gain=10000.0):
    """FMZ 427921 'Kalman' smoothing of a line (dk*sqrt(gain/10000*2), velocity gain/10000)."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=np.nan)
    kf = np.full(len(x), np.nan)
    velo = 0.0
    g = gain / 10000.0
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        prev = kf[i - 1] if i > 0 and np.isfinite(kf[i - 1]) else x[i]
        dk = x[i] - prev
        smooth = prev + dk * np.sqrt(g * 2)
        velo = velo + g * dk
        kf[i] = smooth + velo
    return kf


def _wavetrend(h, l, c, n1=10, n2=21):
    ap = (h + l + c) / 3
    esa = ema(ap, n1)
    d = ema(np.abs(ap - esa), n1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.nan_to_num((ap - esa) / (0.015 * d))
    wt1 = ema(ci, n2)
    return wt1, sma(wt1, 4)


def _mid(h, l, n):
    return (_s(h).rolling(n).max().to_numpy() + _s(l).rolling(n).min().to_numpy()) / 2


def _alligator(h, l, jaw=(13, 8), teeth=(8, 5), lips=(5, 3)):
    """Williams Alligator: SMMA of the median price, each line shifted forward (value at t = SMMA[t - shift])."""
    md = (h + l) / 2
    return [_sh(rma(md, n), s) for n, s in (jaw, teeth, lips)]


def _rvi(o, h, l, c, n=10):
    co, hl = c - o, h - l
    num = (co + 2 * _sh(co, 1) + 2 * _sh(co, 2) + _sh(co, 3)) / 6
    den = (hl + 2 * _sh(hl, 1) + 2 * _sh(hl, 2) + _sh(hl, 3)) / 6
    with np.errstate(divide="ignore", invalid="ignore"):
        r = sma(num, n) / sma(den, n)
    sig = (r + 2 * _sh(r, 1) + 2 * _sh(r, 2) + _sh(r, 3)) / 6
    return r, sig


def _at_local_time(df, x, tz, hour, minute=0):
    """Value of x on the bar that STARTS at hour:minute local time, on every bar of that local day from then on."""
    st = df.index.tz_convert(tz)
    day = pd.Series(st.date).to_numpy()
    hit = (st.hour == hour) & (st.minute == minute)
    v = pd.Series(np.where(hit, np.asarray(x, dtype=float), np.nan))
    return v.groupby(day).ffill().to_numpy(), day


# ---------------- SC124 STC 23/50/10 + EMA35/EMA200 (FMZ) ----------------
def s_fmz_stc_ema35_200(df):
    """FMZ 435160 'Schaff Trend Cycle with Double Moving Average Crossover' (BTC_USDT perpetual, 3-hour backtest):
    STC = EMA3 of a 10-bar stochastic of EMA3 of a 10-bar stochastic of MACD(23, 50). Long when STC crosses above
    25 with EMA35 > EMA200 and close > EMA35; short when STC crosses below 75 with EMA35 < EMA200 and close < EMA35.
    Stop and target 400 ticks each (1:1); an opposite signal reverses. Translation (ours): the tick size is not
    stated, so the equal 1:1 stop / target becomes 1 ATR(14) each; 3-hour -> listed as 4Hour (nearest allowed)."""
    c = df["c"].to_numpy(float)
    stc = _stc_ema(c, 23, 50, 10)
    e35, e200 = ema(c, 35), ema(c, 200)
    L = cross_up(stc, 25) & (e35 > e200) & (c > e35)
    S = cross_down(stc, 75) & (e35 < e200) & (c < e35)
    return L, S & ~L, {"stop": ("atr", 1.0), "target": ("R", 1.0), "opposite": True}


# ---------------- SC125 STC turn + UT ATR trail regime + SMA200 (FMZ) ----------------
def s_fmz_stc_ut_sma200(df):
    """FMZ 429472 'STC MA ATR Integrated Trend Trading' (BTC_USDT, 4-hour): STC(12, 26, 50) with 0.5 smoothing; a
    UT-Bot trailing stop of 3.5 x ATR(5) sets the regime (+1 after a close crosses above it, -1 below). Long on
    the first bar the STC turns up, regime not -1, close >= SMA200; short on the first bar the STC turns down,
    regime -1, close < SMA200. Stop = the trailing-stop line at entry (fixed), target = entry +/- 2 x ATR(5);
    a long is closed when the regime turns -1 (short when +1). As published - no pip translation needed."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a5 = atr(h, l, c, 5)
    t, pos = _ut_trail(c, a5, 3.5)
    stc = _stc_half(c, 12, 26, 50)
    up = stc > _sh(stc)
    dn = stc < _sh(stc)
    lt = up & ~(_sh(up.astype(float)) > 0)
    st = dn & ~(_sh(dn.astype(float)) > 0)
    s200 = sma(c, 200)
    L = (pos != -1) & lt & (c >= s200)
    S = (pos == -1) & st & (c < s200)
    return L, S & ~L, {"stop": ("level", t, t), "target": ("level", c + 2 * a5, c - 2 * a5),
                       "exit_long": pos == -1, "exit_short": pos == 1}


# ---------------- SC126 Kijun 20 cross with a strong close + 1.5 ATR stop (FMZ) ----------------
def s_fmz_kijun20_wpr1(df):
    """FMZ 426847 'ATR Stop Loss Ichimoku Kijun Breakout' (15-minute, written for FX): Kijun = mid of the 20-bar
    high/low; long when a bar opens below the Kijun and closes above it in the top 25% of its range (Williams %R(1)
    >= -25), short on the mirror (%R(1) <= -75). Stop 1.5 x ATR(14), target 15 pips, out when the close crosses
    back over the Kijun. Translation: a major's 15-minute ATR(14) is ~9 pips -> 15 pips ~ 1.6 ATR. %R(1) is taken
    over the bar's true range (high / low extended to the previous close)."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    kj = _mid(h, l, 20)
    pc = _sh(c)
    th, tl = np.fmax(h, pc), np.fmin(l, pc)   # ours: %R(1) over the true range (a gap from the last close counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = np.where(th > tl, -100 * (th - c) / (th - tl), -50.0)
    L = (o < kj) & (c > kj) & (wr >= -25)
    S = (o > kj) & (c < kj) & (wr <= -75)
    return L, S & ~L, {"stop": ("atr", 1.5), "target": ("atr", 1.6), "exit_long": c < kj, "exit_short": c > kj}


# ---------------- SC127 Kijun 20 loopback, 1-hour (FMZ) ----------------
def s_fmz_kijun20_loopback(df):
    """FMZ 431282 'Kijun Loopback' (same author as 426847, 1-hour): a bar that opens below the 20-bar Kijun and
    closes above it -> long (mirror short), no %R filter. Stop 1.5 x ATR(14), target 10 pips, out when the close
    crosses back over the Kijun. Translation: a major's 1-hour ATR(14) is ~13 pips -> 10 pips ~ 0.8 ATR."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    kj = _mid(h, l, 20)
    L = (o < kj) & (c > kj)
    S = (o > kj) & (c < kj)
    return L, S & ~L, {"stop": ("atr", 1.5), "target": ("atr", 0.8), "exit_long": c < kj, "exit_short": c > kj}


# ---------------- SC128 Kalman Hull cross + WaveTrend cross (FMZ) ----------------
def s_fmz_kalman_hull_wt(df):
    """FMZ 427921 'Hull Moving Average and WT Cross' (BTC_USDT, 1-hour): two Hull lines - HMA(hl2, 24) smoothed
    by WMA 5 and a 3-WMA Hull of the close (4 / 6 / 12, WMA 12) - each Kalman-smoothed (gain 10000); WaveTrend
    10/21 (wt2 = SMA4 of wt1). Long when the fast Hull crosses above the slow one AND wt1 crosses above wt2 on the
    same bar, short on the mirror. Stop 5%, target 10% from entry; an opposite signal closes. As published."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    hl2 = (h + l) / 2
    hma = _wma(2 * _wma(hl2, 12) - _wma(hl2, 24), 5)
    hma3 = _wma(3 * _wma(c, 4) - _wma(c, 6) - _wma(c, 12), 12)
    a, b = _kalman_hull(hma), _kalman_hull(hma3)
    wt1, wt2 = _wavetrend(h, l, c)
    L = cross_up(b, a) & cross_up(wt1, wt2)
    S = cross_up(a, b) & cross_down(wt1, wt2)
    (sl, ss), (tl, ts) = _pct_levels(c, 5.0, 10.0)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": ("level", tl, ts), "opposite": True}


# ---------------- SC129 MFI(3) momentum bar, 5-minute (FMZ) ----------------
def s_fmz_mfi3_5m(df):
    """FMZ 439748 'Money Flow Index 5 Minute Strategy' (BTC_USDT, 5-minute): previous bar's MFI(3) = 100, this bar
    bullish with a body > 50% of its range and close > SMA80 -> long; previous MFI(3) = 0, bearish body > 50%,
    close < SMA80 -> short. Stop = the low of today's 00:00 UTC bar - 3% (short: its high + 3%), target 2.2R.
    As published (percentages); the optional stop modes / trailing / 60-minute exit are off by default."""
    o, h, l, c, v = _open(df), *(df[x].to_numpy(float) for x in "hlcv")
    m = _sh(_mfi(h, l, c, v, 3))
    s80 = sma(c, 80)
    body = np.abs(c - o) > 0.5 * (h - l)
    L = (m >= 99.999) & (c > o) & body & (c > s80)
    S = (m <= 0.001) & (c < o) & body & (c < s80)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    l0 = pd.Series(l).groupby(day).transform("first").to_numpy()
    h0 = pd.Series(h).groupby(day).transform("first").to_numpy()
    return L, S & ~L, {"stop": ("level", l0 * 0.97, h0 * 1.03), "target": ("R", 2.2)}


# ---------------- SC130 Asian 19:00-01:00 range fade (FMZ) ----------------
def s_fmz_asian_range_fade(df):
    """FMZ 446567 'Asian Session High/Low Breakout' (BTC_USDT, 1-hour; the code FADES the break): range = high / low
    of the 19:00-01:00 UTC bars; from 04:00 (session end + 3 h) until the next session, a close under the Asian low
    -> long, over the Asian high -> short, one trade per session. Stop and target 300 ticks each (1:1). Translation:
    the tick size is not stated -> 1 ATR(14) each; the levels are fixed at entry (the code's drifting exit is a bug)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    ts = df.index.tz_convert("UTC")
    hr = ts.hour.to_numpy()
    key = pd.Series((ts - pd.Timedelta(hours=19)).date).to_numpy()
    inr = (hr >= 19) | (hr < 1)
    hi = pd.Series(np.where(inr, h, np.nan)).groupby(key).transform("max").to_numpy()
    lo = pd.Series(np.where(inr, l, np.nan)).groupby(key).transform("min").to_numpy()
    win = (hr >= 4) & (hr < 19)
    L0 = win & (c < lo)
    S0 = win & (c > hi)
    first = _first_per_day(L0 | S0, key)
    return L0 & first, S0 & first & ~L0, {"stop": ("atr", 1.0), "target": ("R", 1.0)}


# ---------------- SC131 Sproule EMA100 (BabyPips) ----------------
def s_bp_sproule_ema100(df):
    """BabyPips forum 'The Sproule System' (1-hour, FX): long when price is above EMA100 and the previous bar's low
    is above it too (short: below, previous high below). Stop = the EMA100, moved to its new value every hour
    (trailing on the EMA); target 100 pips. Ours: entry on the first bar the condition becomes true.
    Translation: a major's 1-hour ATR(14) is ~13 pips -> 100 pips ~ 8 ATR."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    e = ema(c, 100)
    L = _onset((c > e) & (_sh(l) > _sh(e)))
    S = _onset((c < e) & (_sh(h) < _sh(e)))
    return L, S & ~L, {"stop": ("trail_line",), "line": e, "target": ("atr", 8.0)}


# ---------------- SC132 EURUSD SMA5/SMA100 cross confirmed, 150 / 300 pips (BabyPips) ----------------
def s_bp_sma5_100_150pips(df):
    """BabyPips forum 'Potential 150 pips ... swing trading system' (EURUSD, 4-hour and daily): SMA5 crosses above
    SMA100, wait one more candle and buy if SMA5 is still above (short mirror). Stop 150 pips, target 300 pips,
    nothing else. Translation: EURUSD 4-hour ATR(14) is ~30 pips -> stop 5 ATR, target 2R (10 ATR)."""
    c = df["c"].to_numpy(float)
    s5, s100 = sma(c, 5), sma(c, 100)
    L = (_sh(cross_up(s5, s100).astype(float)) > 0) & (s5 > s100)
    S = (_sh(cross_down(s5, s100).astype(float)) > 0) & (s5 < s100)
    return L, S & ~L, {"stop": ("atr", 5.0), "target": ("R", 2.0)}


# ---------------- SC133 Richbugger EMA60 median 4H + 1H (BabyPips) ----------------
def s_bp_richbugger(df):
    """BabyPips forum 'The Richbugger System' (4-hour and 1-hour, example GBPJPY): EMA60 of the median price on
    the 4-hour and the 1-hour chart. Long when price is at least 20 pips above the 4-hour EMA and above the 1-hour
    EMA (short mirror, never short above the 4H EMA). Stop 30 pips, then trailed on the 1-hour EMA; out when
    price crosses back through the 1-hour EMA; no fixed target. Differs from FR07 (3 Ducks SMA60 x3): EMA of
    HL/2, a 20-pip buffer, 30-pip stop and the 1H-EMA exit. Translation: GBPJPY 1-hour ATR(14) ~20 pips ->
    buffer 1.0 ATR, stop 1.5 ATR; the trail on the EMA = the exit on a close back through it."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    f = lambda g: ema(((g["h"] + g["l"]) / 2).to_numpy(), 60)  # noqa: E731
    e4, e1 = _htf(df, "4h", f), _htf(df, "1h", f)
    L = _onset((c > e4 + a) & (c > e1))
    S = _onset((c < e4 - a) & (c < e1))
    return L, S & ~L, {"stop": ("atr", 1.5), "target": None, "exit_long": c < e1, "exit_short": c > e1}


# ---------------- SC134 Centred 227-bar close extreme breakout, EURUSD 1H (ProRealCode) ----------------
def s_prc_fractal113_breakout(df):
    """ProRealCode 'Fractal breakout intraday strategy EURUSD 1H': level = close[113] when it is the highest
    (lowest) close of the 227-bar window centred on it (confirmed 113 bars later), held until the next one. Long
    when the close crosses over the high level, short under the low level, entries 10:00-23:00 platform time
    (CET). Stop = the 20-bar Donchian low (high) of the previous bar; a 5-point trailing stop on closes once 5
    points in profit; target 30 points. Translation: EURUSD 1-hour ATR(14) ~13 pips -> target 30 pips ~ 2.3 ATR,
    the 5-pip trail ~ 0.4 ATR, starting after 0.4 ATR of profit."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    cs = _s(c)
    c113 = cs.shift(113).to_numpy()
    hil = _s(np.where(c113 >= cs.rolling(227).max().to_numpy() - 1e-12, c113, np.nan)).ffill().to_numpy()
    lol = _s(np.where(c113 <= cs.rolling(227).min().to_numpy() + 1e-12, c113, np.nan)).ffill().to_numpy()
    hs, _, _, _, _ = _local(df, "Europe/Paris")
    win = (hs >= 10) & (hs < 23)
    L = cross_up(c, hil) & win
    S = cross_down(c, lol) & win
    dl = _sh(_s(l).rolling(20).min().to_numpy())
    dh = _sh(_s(h).rolling(20).max().to_numpy())
    return L, S & ~L, {"stop": ("level", dl, dh), "target": ("atr", 2.3), "trail_atr": 0.4,
                       "trail_after_px": 0.4 * a}


# ---------------- SC135 DAX 3 same-colour bars at 12:00 / 16:00 (ProRealCode) ----------------
def s_prc_dax_3bars_hour(df):
    """ProRealCode '3 bars patterns DAX intraday trading 1 hour' (DAX, 1-hour, CET): on the bar closing at 12:00,
    if it and the two before it all closed up -> buy; on the bar closing at 16:00, three down bars -> sell short.
    Stop 30 points, target 45 points, flat at 21:00. Translation: DAX 1-hour ATR(14) ~50 points -> stop 0.6 ATR,
    target 1.5R."""
    o, c = _open(df), df["c"].to_numpy(float)
    _, he, _, _, _ = _local(df, "Europe/Berlin")
    up, dn = c > o, c < o
    up3 = up & (_sh(up.astype(float)) > 0) & (_sh(up.astype(float), 2) > 0)
    dn3 = dn & (_sh(dn.astype(float)) > 0) & (_sh(dn.astype(float), 2) > 0)
    L = up3 & (np.abs(he - 12) < 1e-6)
    S = dn3 & (np.abs(he - 16) < 1e-6)
    return L, S & ~L, {"stop": ("atr", 0.6), "target": ("R", 1.5), "force_exit": he >= 21 - 1e-9}


# ---------------- SC136 20/200 pips open-difference at 18:00 GMT (MQL5) ----------------
def s_mql5_20_200_pips(df):
    """MQL5 CodeBase 214 '20/200 pips' (EURUSD H1): at the start of the 18:00 GMT bar, if Open[2] - Open[7] (the
    16:00 and 11:00 bar opens) > 7 pips -> buy, < -7 pips -> sell (direction as in the StockSharp port's text);
    one trade a day, target 20 pips, stop 200 pips. Translation: EURUSD 1-hour ATR(14) ~13 pips -> 7 pips ~ 0.5,
    20 pips ~ 1.5, 200 pips ~ 15 ATR of the 1-hour bars. Ours: the signal is the bar that ends at 18:00 UTC."""
    o = _open(df)
    a1 = _htf(df, "1h", lambda g: atr(g["h"].to_numpy(), g["l"].to_numpy(), g["c"].to_numpy(), 14))
    o16, _ = _at_local_time(df, o, "UTC", 16)
    o11, _ = _at_local_time(df, o, "UTC", 11)
    _, he, _, _, _ = _local(df, "UTC")
    at = np.abs(he - 18) < 1e-6
    dlt = o16 - o11
    c = df["c"].to_numpy(float)
    L = at & (dlt > 0.5 * a1)
    S = at & (dlt < -0.5 * a1)
    return L, S & ~L, {"stop": ("level", c - 15 * a1, c + 15 * a1), "target": ("level", c + 1.5 * a1, c - 1.5 * a1)}


# ---------------- SC137 freqtrade BinHV45 (GitHub) ----------------
def s_ft_binhv45(df):
    """freqtrade BinHV45 (berlinguyinca; fork TheoBrigitte), 1-minute crypto, long only: BB 40 / 2 sample SD;
    bbdelta = mid - lower > 0.8% of close, |close - close[1]| > 1.75% of close, close - low < 0.25 x bbdelta,
    close < the previous lower band and close <= close[1] -> buy. Stop -5%, target +1.25% (ROI). As published."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    mid = sma(c, 40)
    lower = mid - 2 * _s(c).rolling(40).std(ddof=1).to_numpy()
    bbd = np.abs(mid - lower)
    cd = np.abs(c - _sh(c))
    tail = np.abs(c - l)
    L = (_sh(lower) > 0) & (bbd > 0.008 * c) & (cd > 0.0175 * c) & (tail < 0.25 * bbd) & (c < _sh(lower)) & (c <= _sh(c))
    (sl, ss), (tl, ts) = _pct_levels(c, 5.0, 1.25)
    return L, np.zeros(len(c), bool), {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- SC138 DAX Reversal Catcher M30 (ProRealCode) ----------------
def s_prc_dax_reversal_catcher(df):
    """ProRealCode 'DAX Reversal Catcher M30' (ThibauldR, DAX 30-minute, long only): the previous bar's low under
    its BB(20, 1.5) lower band, this bar closes inside the bands, makes a higher high and higher low and closes over
    the previous high, TEMA21 > TEMA50, and the close crosses over the session VWAP -> buy. Stop = the previous
    bar's low; out when the close is above the upper band, or TEMA21 crosses under TEMA50, or at 15:00 CET.
    The source's 30-minute chart is not an allowed timeframe -> listed as 15Min and 1Hour."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    lo, _, up = _bb(c, 20, 1.5)
    t21, t50 = _tema(c, 21), _tema(c, 50)
    vw = _day_vwap_local(df, "Europe/Berlin")
    _, he, _, _, _ = _local(df, "Europe/Berlin")
    L = ((_sh(l) < _sh(lo)) & (c > lo) & (c < up) & (l > _sh(l)) & (h > _sh(h)) & (c > _sh(h)) & (t21 > t50)
         & cross_up(c, vw) & (he < 15))
    return L, np.zeros(len(c), bool), {"stop": ("level", _sh(l), _sh(h)), "target": None,
                                       "exit_long": (c > up) | cross_down(t21, t50), "force_exit": he >= 15 - 1e-9}


# ---------------- SC139 DAX H4 trend: over SMA200 + MACD > 0, 1% / 3% (ProRealCode) ----------------
def s_prc_dax_h4_sma200_macd(df):
    """ProRealCode 'Day trading DAX with the trend' (DAX, 4-hour): close > SMA200 and MACD(12, 26) main line > 0,
    from 09:00 -> buy (the code also shorts on the mirror). A state, not a cross: any qualifying bar while flat.
    Stop 1%, target 3%, flat at 21:00 CET. Differs from TV22 (MACD zero cross + EMA200, 2R) in the state entry,
    the SMA, the 1% / 3% and the 21:00 flat. As published (percentages)."""
    c = df["c"].to_numpy(float)
    m = ema(c, 12) - ema(c, 26)
    s200 = sma(c, 200)
    hs, he, _, _, _ = _local(df, "Europe/Berlin")
    tm = (hs >= 9) & (he < 21)
    L = tm & (c > s200) & (m > 0)
    S = tm & (c < s200) & (m < 0)
    (sl, ss), (tl, ts) = _pct_levels(c, 1.0, 3.0)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": ("level", tl, ts), "force_exit": he >= 21 - 1e-9}


# ---------------- SC140 MACD(5,13,1) overnight change, entry at the 08:00 open (investing.co.uk) ----------------
def s_inv_macd5131_overnight(df):
    """investing.co.uk 'MACD 4 hour trading strategy' (FX, 4-hour): MACD(5, 13, 1) on the 04:00 candle minus its
    value on the previous day's 20:00 candle; >= +50 (5-digit points) -> long at the 08:00 open, <= -50 -> short.
    Two positions, stop 30 pips each, targets 30 and 45 pips; at +30 pips half out and the stop to break-even.
    Translation: 50 points = 0.0005 ~ 0.15 ATR(14) of a major's 4-hour bar (~30 pips); stop 30 pips ~ 1 ATR;
    the two targets -> one target at 1.5R with break-even at 1R (2/3 of the way). Times taken as UTC."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    m = ema(c, 5) - ema(c, 13)
    st = df.index.tz_convert("UTC")
    day = pd.Series(st.date)
    at20 = (st.hour == 20) & (st.minute == 0)
    v20 = pd.Series(np.where(at20, m, np.nan)).groupby(day.to_numpy()).max().shift(1)  # the previous day's 20:00
    A = day.map(v20).to_numpy(dtype=float)
    # the bar that ends at 08:00 (4-hour bar opening 04:00, or the last lower-timeframe bar before 08:00)
    _, he, _, _, _ = _local(df, "UTC")
    sig = np.abs(he - 8) < 1e-6
    dlt = m - A
    L = sig & (dlt >= 0.15 * a)
    S = sig & (dlt <= -0.15 * a)
    return L, S & ~L, {"stop": ("atr", 1.0), "target": ("R", 1.5), "breakeven": (0.667, 0.0)}


# ---------------- SC141 SMA48 + ATR(7) crossing its 30-bar average, USDJPY H4 (Forex Tester) ----------------
def s_ftb_sma48_atr_cross(df):
    """Forex Tester blog 'List of strategies 3' (USD/JPY, 4-hour): price above SMA48 and ATR(7) crosses above its
    30-period average -> buy at the next bar; price below SMA48 and ATR crosses its average from above -> sell.
    Stop at least 25 points, break-even at 25-30 points, target 100-120 points, out on a reverse signal.
    Translation (points read as pips): USDJPY 4-hour ATR(14) ~35 pips -> stop 0.7 ATR, target 110 pips ~ 3.1 ATR,
    break-even at 27.5 pips ~ 0.25 of the way to the target."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    s48 = sma(c, 48)
    a7 = atr(h, l, c, 7)
    am = sma(a7, 30)
    L = (c > s48) & cross_up(a7, am)
    S = (c < s48) & cross_down(a7, am)
    return L, S & ~L, {"stop": ("atr", 0.7), "target": ("atr", 3.1), "breakeven": (0.25, 0.0), "opposite": True}


# ---------------- SC142 SSMA90 + CCI(12) hook + WPR(11) + ATR filter (Forex Tester) ----------------
def s_ftb_ssma90_cci_wpr(df):
    """Forex Tester blog 'List of strategies 3' (FX, 5-minute to 1-hour): price above the smoothed MA(90), WPR(11)
    under -85, CCI(12) turning up after a low under -100, ATR(14) > 0.002 -> buy at the signal close; mirror sell
    (WPR over -15, CCI turning down from over +100). Stop 10-15 points, target at least 50 points, out on a reverse
    signal. Translation (points read as pips, a major's 15-minute ATR ~8 pips): stop 12.5 pips ~ 1.5 ATR, target
    50 pips ~ 6 ATR; the fixed ATR > 0.002 filter -> ATR(14) above its 100-bar average."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    ss = rma(c, 90)
    cc = _cci(h, l, c, 12)
    w = _wpr(h, l, c, 11)
    act = a > sma(a, 100)
    c1, c2 = _sh(cc), _sh(cc, 2)
    L = (c > ss) & (w < -85) & (c1 < -100) & (cc > c1) & (c2 > c1) & act
    S = (c < ss) & (w > -15) & (c1 > 100) & (cc < c1) & (c2 < c1) & act
    return L, S & ~L, {"stop": ("atr", 1.5), "target": ("atr", 6.0), "opposite": True}


# ---------------- SC143 RVI(10) cross + rising RSI and StdDev (Forex Tester) ----------------
def s_ftb_rvi_rsi_std(df):
    """Forex Tester blog 'List of strategies 4' (majors, 1-hour or higher): RVI(10) crosses above its signal line
    while RSI(14) and StdDev(20) are both higher than on the previous bar -> buy at the next open; mirror sell.
    Stop at least 35 points, break-even after 30 points, target 100 points, out on a reverse signal.
    Translation (points read as pips, a major's 1-hour ATR ~13 pips): stop 2.7 ATR, target 7.7 ATR, break-even at
    0.3 of the way."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    r, sg = _rvi(o, h, l, c, 10)
    rs = rsi(c, 14)
    sd = _s(c).rolling(20).std(ddof=0).to_numpy()
    L = cross_up(r, sg) & (rs > _sh(rs)) & (sd > _sh(sd))
    S = cross_down(r, sg) & (rs < _sh(rs)) & (sd < _sh(sd))
    return L, S & ~L, {"stop": ("atr", 2.7), "target": ("atr", 7.7), "breakeven": (0.3, 0.0), "opposite": True}


# ---------------- SC144 Ichimoku cloud + two Alligators + CCI(50) + Stochastic (Forex Tester) ----------------
def s_ftb_ichi_two_alligators(df):
    """Forex Tester blog 'List of strategies 4' (USD/JPY and yen crosses, 1-hour or higher): a candle closes above
    the Kumo (9/26/52) and above every line of Alligator(13,8,5) and Alligator(39,24,15), CCI(50) > +50 and
    Stochastic(14,3,3) > 50 -> buy at the next candle; mirror sell. Stop 35-50 points, break-even after 30, target
    100-150; out at market when a candle closes beyond any Alligator line against the trade. Translation (pips,
    USDJPY 1-hour ATR ~15 pips): stop 42.5 pips ~ 2.8 ATR, target 125 pips ~ 3R, break-even at 0.24 of the way."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    tk, kj = _mid(h, l, 9), _mid(h, l, 26)
    sa, sb = _sh((tk + kj) / 2, 26), _sh(_mid(h, l, 52), 26)
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    lines = _alligator(h, l) + _alligator(h, l, (39, 24), (24, 15), (15, 9))
    lmax, lmin = np.max(np.vstack(lines), axis=0), np.min(np.vstack(lines), axis=0)
    cc = _cci(h, l, c, 50)
    k, _ = _stoch(h, l, c, 14, 3, 3)
    L = _onset((c > top) & (c > lmax) & (cc > 50) & (k > 50))
    S = _onset((c < bot) & (c < lmin) & (cc < -50) & (k < 50))
    return L, S & ~L, {"stop": ("atr", 2.8), "target": ("R", 3.0), "breakeven": (0.24, 0.0),
                       "exit_long": c < lmax, "exit_short": c > lmin}


# ---------------- SC145 SMA233 high / low channel breakout ladder (Forex Tester) ----------------
def s_ftb_sma233_channel(df):
    """Forex Tester blog 'List of strategies 3' (FX, 30-minute or higher): channel U = SMA233 of the highs, L =
    SMA233 of the lows; a break above U that closes above U + 15 points -> buy, below L - 15 -> sell. Targets a
    ladder at U + 89 / 144 / 233 / 362; stop 'the first level on the back side of the tunnel' (read: U itself).
    Translation (points read as pips, a major's 30-minute ATR ~7 pips): buffer 15 pips ~ 2 ATR; the first ladder
    target 89 pips ~ 12.7 ATR beyond the channel line (ours: one target, the first rung)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    U, D = sma(h, 233), sma(l, 233)
    L = _onset(c > U + 2 * a)
    S = _onset(c < D - 2 * a)
    return L, S & ~L, {"stop": ("level", U, D), "target": ("level", U + 12.7 * a, D - 12.7 * a)}


# ---------------- SC146 DAX 08:00-09:05 CET range breakout + EMA50 bias, 2R (Admirals) ----------------
def s_adm_dax_orb_0905_ema50(df):
    """Admirals traders' blog 'DAX 30 open range breakout strategy' (DAX, 5-minute): range = high / low of
    08:00-09:05 CET; a break of the range in the direction of the 5-minute EMA50 (long only above it, short only
    below) -> enter; stop the other side of the range, target 2R. Ours: first break of the day, until 17:30 CET.
    Differs from SS03 / SC028 in the 08:00-09:05 window and the EMA50 bias."""
    c = df["c"].to_numpy(float)
    hi, lo, hs, _, _, day = _range_levels(df, "Europe/Berlin", 8.0, 9.0 + 5 / 60)
    e50 = ema(c, 50)
    tm = hs < 17.5
    L0 = tm & (c > hi) & (c > e50)
    S0 = tm & (c < lo) & (c < e50)
    first = _first_per_day(L0 | S0, day)
    return L0 & first, S0 & first & ~L0, {"stop": ("level", lo, hi), "target": ("R", 2.0)}


# ---------------- SC147 Warrior Trading pullback: green ramp, 9-EMA / VWAP dip, first new high ----------------
def s_wt_ramp_pullback_9ema(df):
    """Warrior Trading 'Pull back trading strategy' (strong US stocks, 5-minute): a ramp of at least 5 green candles
    in a row, then a pullback to the 9-EMA or VWAP; buy the first 5-minute candle that makes a new high (over the
    previous candle's high). Stop = the low of the pullback; target = the high of the day (sell into the high-of-
    day squeeze). Ours: ramp within the last 12 bars, the EMA9 / VWAP touch within the last 3, pullback low = the
    3-bar low, high of day = before this bar. Long only."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    g = (c > o).astype(int)
    run = pd.Series(g).groupby((g == 0).cumsum()).cumsum().to_numpy()
    ramp = _since(run >= 5, 12)
    e9 = ema(c, 9)
    vw = _day_vwap_local(df, "America/New_York")
    touch = _since(l <= np.fmax(e9, vw), 3) | (l <= np.fmax(e9, vw))
    day = pd.Series(df.index.tz_convert("America/New_York").date).to_numpy()
    hod = _sh(pd.Series(h).groupby(day).cummax().to_numpy())
    pl = _s(l).rolling(3).min().to_numpy()
    L = ramp & touch & (c > _sh(h)) & ~(run >= 5) & (c < hod)
    L = _onset(L)
    return L, np.zeros(len(c), bool), {"stop": ("level", pl, pl), "target": ("level", hod, hod)}


# ---------------- SC148 EMA50 cross, 1 ADR stop, 2 ADR target (HowToTrade) ----------------
def s_htt_ema50_adr(df):
    """HowToTrade 'Average Daily Range (ADR)' method (EUR/USD example; M15, M30, H1): price crosses above EMA50
    -> buy, below -> sell; stop = 1 x ADR from entry, target = 2 x ADR. ADR = the 14-day average of the daily
    high - low (period not stated; 14 is the common default), from completed days. The 'avoid an hour around major
    news' filter is not modelled."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    day = pd.Series(df.index.tz_convert("UTC").date)
    g = pd.DataFrame({"d": day, "h": h, "l": l}).groupby("d").agg(h=("h", "max"), l=("l", "min"))
    adr = (g["h"] - g["l"]).rolling(14).mean().shift(1)
    A = day.map(adr).to_numpy(dtype=float)
    e50 = ema(c, 50)
    L = cross_up(c, e50)
    S = cross_down(c, e50)
    return L, S & ~L, {"stop": ("level", c - A, c + A), "target": ("level", c + 2 * A, c - 2 * A)}


# ---------------- SC149 EIA 10:15-10:30 ET box breakout, midpoint stop (DayTradingToolkit) ----------------
def s_dtt_eia_box_break(df):
    """DayTradingToolkit 'Trader's playbook: EIA oil report' (CL / USO): box = high / low of 10:15-10:30 ET on EIA
    Wednesdays; wait 30 seconds after 10:30, then a break above the box -> long, below -> short. Stop = the box
    midpoint (never moved further away), target 2:1. Ours: first close beyond the box from 10:30 until 12:00 ET on
    Wednesdays (the report day); on 1-minute bars the 10:30 bar closes after the 30-second wait."""
    c = df["c"].to_numpy(float)
    hi, lo, hs, _, wd, day = _range_levels(df, "America/New_York", 10.25, 10.5)
    mid = (hi + lo) / 2
    tm = (wd == 2) & (hs >= 10.5) & (hs < 12)
    L0 = tm & (c > hi)
    S0 = tm & (c < lo)
    first = _first_per_day(L0 | S0, day)
    return L0 & first, S0 & first & ~L0, {"stop": ("level", mid, mid), "target": ("R", 2.0)}


# ---------------- shared helpers for SC150-SC166 ----------------
def _star(o, h, l, c):
    """Shooting star: the mirror of _hammer (upper wick >= 2 bodies, lower wick <= the body)."""
    body = np.abs(c - o)
    lw = np.minimum(o, c) - l
    uw = h - np.maximum(o, c)
    return (uw >= 2 * body) & (lw <= np.maximum(body, 1e-12)) & (h > l)


def _lo(l, n=5):
    return _s(l).rolling(n).min().to_numpy()


def _hi(h, n=5):
    return _s(h).rolling(n).max().to_numpy()


# ---------------- SC150 Engulfing / pin bar breakout, candle stop, 5R long / 4R short (TradingView) ----------------
def s_tv_engulf_pin_breakout(df):
    """TradingView 'Engulfing & Pin Bar Breakout Strategy' (Punisher1on1; 1-hour chart, BTCUSD example): a hammer or
    bullish engulfing signal candle -> buy stop above its high on the next candle; a shooting star or bearish
    engulfing -> sell stop below its low. Stop = the other end of the signal candle; target 1:5 for longs, 1:4 for
    shorts (as published). Ours: pattern definitions are the library's (_hammer / _engulf, star = mirror hammer);
    the signal is the next bar whose high / low reaches the stop level (entry next open), targets measured from the
    stop level."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    be, se = _engulf(o, c)
    bull = _sh((_hammer(o, h, l, c) | be).astype(float)) > 0
    bear = _sh((_star(o, h, l, c) | se).astype(float)) > 0
    h1, l1 = _sh(h), _sh(l)
    L = bull & (h >= h1)
    S = bear & (l <= l1) & ~L
    rg = h1 - l1
    return L, S, {"stop": ("level", l1, h1), "target": ("level", h1 + 5 * rg, l1 - 4 * rg)}


# ---------------- SC151 1h liquidity swings, pivot 5 reclaim, SMA50, 0.5% buffer stop, 2R (TradingView) ----------------
def s_tv_liquidity_swings_1h(df):
    """TradingView '1h Liquidity Swings Strategy with 1:2 RR' (BTCUSD, EURUSD; 1-hour): support = the last pivot low
    (5 bars each side), resistance = the last pivot high. Uptrend + the low crosses back above support + close below
    resistance -> long; downtrend + the high crosses back under resistance + close above support -> short. Stop =
    support - 0.5% (resistance + 0.5%), target 2R; a long closes on a close under support (short: over resistance).
    Ours: the trend rule is not stated - read as close vs SMA50 (the script's trend filter)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    res, sup = _swings(h, l, 5)
    s50 = sma(c, 50)
    L = (c > s50) & cross_up(l, sup) & (c < res)
    S = (c < s50) & cross_down(h, res) & (c > sup) & ~L
    return L, S, {"stop": ("level", sup * 0.995, res * 1.005), "target": ("R", 2.0),
                  "exit_long": c < sup, "exit_short": c > res}


# ---------------- SC152 50 Pips A Day: 07:00 GMT hourly candle straddle, 15 / 50 pips (TradingView) ----------------
def s_tv_50pips_0700_straddle(df):
    """TradingView '50 Pips A Day' (Kaspricci; 1-hour, EURUSD / GBPUSD): range = the 07:00 GMT (London open) hourly
    candle; buy stop at its high, sell stop at its low (+/- a gap, default not stated -> 0), the other order cancelled
    once one fills; stop 15 pips, target 50 pips; Monday-Friday; open trades closed at the session end (time not
    stated -> ours 21:00 UTC). Translation: 15 / 50 EURUSD pips -> ATR(14) multiples of the bar's timeframe
    (_pips_atr; 1-hour: 1.4 / 4.5 ATR). The signal is the first bar after 08:00 whose high / low reaches the level."""
    h, l = df["h"].to_numpy(float), df["l"].to_numpy(float)
    hi, lo, hs, _, wd, day = _range_levels(df, "UTC", 7.0, 8.0)
    tm = (hs < 21.0) & (wd < 5)
    L0 = tm & (h >= hi)
    S0 = tm & (l <= lo)
    first = _first_per_day(L0 | S0, day)
    return L0 & first, S0 & first & ~L0, {"stop": ("atr", _pips_atr(df, 15)), "target": ("atr", _pips_atr(df, 50)),
                                          "force_exit": hs >= 21.0}


# ---------------- SC153 Bollinger 20/2 close breakout + RSI 50, 2% / 4%, 4H (TradingView) ----------------
def s_tv_bb_breakout_rsi50_4h(df):
    """TradingView 'Bollinger Breakout Strategy with Direction Control [4H crypto]' (BTC, also FX / stocks; 4-hour):
    close above the upper Bollinger band (20, 2) and RSI(14) > 50 -> long; close under the lower band and RSI < 50
    -> short. Stop 2% from entry, target 2:1 (4%). Differs from CR08 / CS08 (mid-band exit, no RSI)."""
    c = df["c"].to_numpy(float)
    lb, _, ub = _bb(c, 20, 2.0)
    r = rsi(c, 14)
    L = (c > ub) & (r > 50)
    S = (c < lb) & (r < 50)
    (sl, ss), (tl, ts) = _pct_levels(c, 2.0, 4.0)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- SC154 SMA10/30 cross + EMA5 filter, 1% / 2%, 5m (TradingView) ----------------
def s_tv_sma10_30_ema5(df):
    """TradingView 'MA Crossover Strategy with TP/SL (5 EMA Filter)' (5-minute): SMA10 crosses above SMA30 and the
    close is above EMA5 -> long at the close; SMA10 under SMA30 and close under EMA5 -> short. Stop 1%, target 2%,
    held until one is hit."""
    c = df["c"].to_numpy(float)
    f, s, e5 = sma(c, 10), sma(c, 30), ema(c, 5)
    L = cross_up(f, s) & (c > e5)
    S = cross_down(f, s) & (c < e5)
    (sl, ss), (tl, ts) = _pct_levels(c, 1.0, 2.0)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- SC155 Nifty 1m EMA 5/10 cross + Stochastic 14/3/3, 5 / 10 points (TradingView) ----------------
def s_tv_nifty_ema5_10_stoch(df):
    """TradingView 'Nifty 1m EMA Pullback Scalper' (rahul52das; Nifty, 1-minute): EMA5 crosses above EMA10 and the
    Stochastic (14,3,3) rises above 20 -> buy; EMA5 under EMA10 and the Stochastic falls under 80 -> sell. Stop 5
    index points, target 10 points; out when the Stochastic reaches overbought / oversold (80 / 20, not stated).
    Translation: Nifty's 1-minute ATR(14) ~10 points -> stop 0.5 ATR, target 1.0 ATR. 'Rises above 20' read as
    %K > 20 and rising (mirror: %K < 80 and falling)."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    k, _ = _stoch(h, l, c, 14, 3, 3)
    e5, e10 = ema(c, 5), ema(c, 10)
    L = cross_up(e5, e10) & (k > 20) & (k > _sh(k)) & (k < 80)
    S = cross_down(e5, e10) & (k < 80) & (k < _sh(k)) & (k > 20)
    return L, S & ~L, {"stop": ("atr", 0.5), "target": ("atr", 1.0), "exit_long": k >= 80, "exit_short": k <= 20}


# ---------------- SC156 Forex Master: EMA20(RSI10) / EMA30(RSI30) cross of 50, 50 / 50 pips, 1h (TradingView) ----------------
def s_tv_forex_master_rsi_ema(df):
    """TradingView 'Forex Master (EUR/USD)' (Stable_Camel; EURUSD, 1-hour): long when EMA20 of RSI(10) crosses above
    50; short when EMA30 of RSI(30) crosses below 50. Stop 50 pips, target 50 pips, closed on the opposite signal.
    Translation: 50 EURUSD pips -> _pips_atr (1-hour ~4.5 ATR)."""
    c = df["c"].to_numpy(float)
    el = ema(np.nan_to_num(rsi(c, 10), nan=50.0), 20)
    es = ema(np.nan_to_num(rsi(c, 30), nan=50.0), 30)
    L = cross_up(el, 50.0)
    S = cross_down(es, 50.0) & ~L
    m = _pips_atr(df, 50)
    return L, S, {"stop": ("atr", m), "target": ("atr", m), "opposite": True}


# ---------------- SC157 EMA9 breakaway candle 5 pips off the EMA, previous-candle stop, 2R, M15 (Forex Factory) ----------------
def s_ff_ema9_breakaway(df):
    """Forex Factory post (GBPUSD M15): the previous candle closes above EMA9, this candle's low is at least 5 pips
    above EMA9 and it closes above the previous candle's high -> buy; mirror sell. Stop = the previous candle's low
    - 1 pip - spread (short: its high + 1 pip), target 2 x the stop. Translation (_pips_atr, 15-minute): 5 pips ~
    0.9 ATR, 1 pip ~ 0.18 ATR; the spread is left to the bot's costs."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    a = atr(h, l, c, 14)
    e9 = ema(c, 9)
    g, p1 = _pips_atr(df, 5) * a, _pips_atr(df, 1) * a
    c1, e1, h1, l1 = _sh(c), _sh(e9), _sh(h), _sh(l)
    L = (c1 > e1) & (l >= e9 + g) & (c > h1)
    S = (c1 < e1) & (h <= e9 - g) & (c < l1)
    return L, S & ~L, {"stop": ("level", l1 - p1, h1 + p1), "target": ("R", 2.0)}


# ---------------- SC158 MACD 12/26/9 cross + confirming candle, 125 / 30 pips, 6-bar exit, H1 (Forex Factory) ----------------
def s_ff_macd_confirm_h1(df):
    """Forex Factory post (GBPUSD, 1-hour): the MACD line crosses over the signal line, the next hourly candle is
    bullish with the MACD still above the signal -> buy at that candle's close; mirror sell. Stop 125 pips, target
    30 pips, every trade closed at the end of six hourly candles. The author's counter-trend filter is undefined and
    left out. Translation (_pips_atr, 1-hour): 125 pips ~ 11.4 ATR, 30 pips ~ 2.7 ATR."""
    o, c = _open(df), df["c"].to_numpy(float)
    m, s, _ = macd(c, 12, 26, 9)
    xu, xd = _sh(cross_up(m, s).astype(float)) > 0, _sh(cross_down(m, s).astype(float)) > 0
    L = xu & (c > o) & (m > s)
    S = xd & (c < o) & (m < s)
    return L, S & ~L, {"stop": ("atr", _pips_atr(df, 125)), "target": ("atr", _pips_atr(df, 30)), "max_bars": 6}


# ---------------- SC159 Kathy Lien Pure Fade: 3-SD touch + RSI, close back in the 2-1 SD zone, 2R (book) ----------------
def s_kl_pure_fade_bb3(df):
    """Kathy Lien 'High Probability Trading Setups' - The Pure Fade Trade (FX, 1-hour only): Bollinger 20 at 1, 2
    and 3 SD. RSI < 30 and price touches the lower 3-SD band; then a candle closes fully inside the lower 2-SD /
    1-SD zone -> buy; mirror sell (RSI > 70, upper 3-SD touch, close inside the upper 1-2 SD zone). Stop 10 pips
    beyond the swing low / high of the move; half at 1R and the stop to break-even, the rest at the opposite 2-SD
    band (the book's no-watch variant: 2R - used here). Ours: the touch within the last 10 bars, 'fully inside' =
    the candle's open and close in the zone, swing = the 10-bar extreme, 10 pips via _pips_atr (1-hour ~0.9 ATR),
    break-even at 1R = half of the 2R target."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    a = atr(h, l, c, 14)
    l1, _, u1 = _bb(c, 20, 1.0)
    l2, _, u2 = _bb(c, 20, 2.0)
    l3, _, u3 = _bb(c, 20, 3.0)
    r = rsi(c, 14)
    tl = _since((l <= l3) & (r < 30), 10)
    ts = _since((h >= u3) & (r > 70), 10)
    inl = (np.minimum(o, c) >= l2) & (np.maximum(o, c) <= l1)
    ins = (np.maximum(o, c) <= u2) & (np.minimum(o, c) >= u1)
    L = _onset(tl & inl)
    S = _onset(ts & ins) & ~L
    p10 = _pips_atr(df, 10) * a
    return L, S, {"stop": ("level", _lo(l, 10) - p10, _hi(h, 10) + p10), "target": ("R", 2.0),
                  "breakeven": (0.5, 0.0)}


# ---------------- SC160 Steel Wolverine: Supertrend 10/3 + HMA20 + ROC(9) zero cross (tradingstrategyguides) ----------------
def s_tsg_steel_wolverine(df):
    """TradingStrategyGuides 'The Steel Wolverine' (EURUSD, GBPUSD, NQ, ES; 5m / 15m, 1h / 4h): Supertrend(10, 3)
    under price and HMA20 rising; price pulls back to the HMA20 with no close beyond the Supertrend and ROC(9) dips
    to about 0; trigger = a bullish engulfing or pin bar closing above the HMA20 while ROC(9) crosses above 0 on that
    bar -> buy at the close; mirror sell. Stop 2-3 pips beyond the pullback swing; target 2.5R (option A). London and
    NY opening sessions only (ours 07:00-10:00 and 13:00-16:00 UTC); skip an abnormally large trigger (ours > 2 ATR).
    Ours: the HMA touch on this bar or the 3 before, 2.5 pips via _pips_atr, swing = 5-bar extreme."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    a = atr(h, l, c, 14)
    stl = _supertrend_line(h, l, c, 10, 3.0)
    hm = _hma(c, 20)
    roc = 100 * (c / _sh(c, 9) - 1)
    be, se = _engulf(o, c)
    hr = df.index.tz_convert("UTC").hour.to_numpy()
    sess = ((hr >= 7) & (hr < 10)) | ((hr >= 13) & (hr < 16))
    ok = (h - l) <= 2 * a
    p = _pips_atr(df, 2.5) * a
    tb = (l <= hm) | _since(l <= hm, 3)
    ts = (h >= hm) | _since(h >= hm, 3)
    L = (c > stl) & (hm > _sh(hm)) & tb & (c > hm) & cross_up(roc, 0.0) & (be | _hammer(o, h, l, c)) & sess & ok
    S = (c < stl) & (hm < _sh(hm)) & ts & (c < hm) & cross_down(roc, 0.0) & (se | _star(o, h, l, c)) & sess & ok
    return L, S & ~L, {"stop": ("level", _lo(l) - p, _hi(h) + p), "target": ("R", 2.5)}


# ---------------- SC161 Velvet Hammer: HMA20 + Stochastic RSI < 20 + engulfing, 2R (tradingstrategyguides) ----------------
def s_tsg_velvet_hammer(df):
    """TradingStrategyGuides 'The Velvet Hammer' (index ETFs, crypto, FX; 15m, 1h, 4h): HMA20 rising and price above
    it; price touches or pierces the HMA20 without the HMA changing slope; Stochastic RSI (14,14,3,3) %K and %D both
    under 20; a bullish engulfing closes while the HMA still rises -> buy at the close; mirror sell (%K, %D over 80,
    bearish engulfing). Stop 2-3 pips beyond the lower of the pullback swing low and the engulfing low; target 2R.
    Ours: touch and the oversold Stoch RSI on this bar or the 5 before, the HMA rising for 4 bars, 2.5 pips via
    _pips_atr, swing = 5-bar extreme."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    a = atr(h, l, c, 14)
    hm = _hma(c, 20)
    k, d = _stochrsi(c, 14, 14, 3, 3)
    be, se = _engulf(o, c)
    rise = pd.Series((hm > _sh(hm)).astype(float)).rolling(4).min().to_numpy() > 0
    fall = pd.Series((hm < _sh(hm)).astype(float)).rolling(4).min().to_numpy() > 0
    os_, ob = (k < 20) & (d < 20), (k > 80) & (d > 80)
    p = _pips_atr(df, 2.5) * a
    L = rise & (c > hm) & ((l <= hm) | _since(l <= hm, 5)) & (os_ | _since(os_, 5)) & be
    S = fall & (c < hm) & ((h >= hm) | _since(h >= hm, 5)) & (ob | _since(ob, 5)) & se
    return L, S & ~L, {"stop": ("level", _lo(l) - p, _hi(h) + p), "target": ("R", 2.0)}


# ---------------- SC162 50 pips a day: 07:00 GMT candle's direction, stop past its other end, 50 pips (tradefx) ----------------
def s_tfx_50pips_0700_direction(df):
    """tradefx.co.za '50 Pips a Day Strategy' (EURUSD / GBPUSD, 1-hour): the 07:00 GMT hourly candle; bullish -> buy
    stop at its high, bearish -> sell stop at its low (one direction only, unlike SC152's straddle). Stop 5-10 pips
    beyond the candle's other end, target +50 pips, closed at the end of the day if neither is hit. Translation
    (_pips_atr): 7.5 pips beyond the other end, 50 pips from entry (1-hour ~4.5 ATR); end of day = 23:00 UTC."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    a = atr(h, l, c, 14)
    hi, lo, hs, _, wd, day = _range_levels(df, "UTC", 7.0, 8.0)
    inr = (hs >= 7.0) & (hs < 8.0)
    ro = pd.Series(np.where(inr, o, np.nan)).groupby(day).transform("first").to_numpy()
    rc = pd.Series(np.where(inr, c, np.nan)).groupby(day).transform("last").to_numpy()
    tm = (hs >= 8.0) & (hs < 23.0) & (wd < 5)
    L0 = tm & (rc > ro) & (h >= hi)
    S0 = tm & (rc < ro) & (l <= lo)
    first = _first_per_day(L0 | S0, day)
    b = _pips_atr(df, 7.5) * a
    return L0 & first, S0 & first & ~L0, {"stop": ("level", lo - b, hi + b), "target": ("atr", _pips_atr(df, 50)),
                                          "force_exit": hs >= 23.0}


# ---------------- SC163 EMA 5/20 cross over / under EMA50, 10 / 20 pips, BE at 10, 15m (TradingView idea) ----------------
def s_tv_ema5_20_50_20pips(df):
    """TradingView idea 'The Simplest Scalping Strategy With 3 EMAs For 20 Pips Per Day' (EURGBP, 15-minute): EMA5
    crosses above EMA20 with the close, EMA5 and EMA20 all above EMA50 -> buy; mirror sell. Stop 10 pips, target 20
    pips or out earlier when EMA5 crosses back over EMA20; stop to entry at +10 pips. Translation (_pips_atr,
    15-minute): 10 pips ~ 1.8 ATR, 20 pips ~ 3.6 ATR, break-even at half the target."""
    c = df["c"].to_numpy(float)
    e5, e20, e50 = ema(c, 5), ema(c, 20), ema(c, 50)
    L = cross_up(e5, e20) & (c > e50) & (e5 > e50) & (e20 > e50)
    S = cross_down(e5, e20) & (c < e50) & (e5 < e50) & (e20 < e50)
    return L, S & ~L, {"stop": ("atr", _pips_atr(df, 10)), "target": ("atr", _pips_atr(df, 20)),
                       "breakeven": (0.5, 0.0), "exit_long": cross_down(e5, e20), "exit_short": cross_up(e5, e20)}


# ---------------- SC164 PineMyths #2: ADX + Bollinger %B + Awesome Oscillator + 4 EMAs, 2% / 1% (Medium / YouTube) ----------------
def s_pm_adx_pctb_ao_ema(df):
    """Medium 'ADX + BB %B + AO + EMA strategy - PineMyths #2' (from a YouTube video; ETHUSDT, tested on 1-minute):
    EMA5 > EMA21, EMA50 > EMA200, Bollinger %B(20, 2) > 0.75, ADX(14) > 15 and Awesome Oscillator > 2 -> long;
    EMA5 < EMA21, EMA50 < EMA200, %B < 0.25, ADX > 15, AO < -2 -> short. Stop 2%, target 1% (as published).
    Translation: AO +/-2 is in ETH dollars (~0.07% of ETH's price) -> AO beyond +/-0.07% of the close."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    lb, _, ub = _bb(c, 20, 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pb = (c - lb) / (ub - lb)
    hl2 = (h + l) / 2
    ao = sma(hl2, 5) - sma(hl2, 34)
    ad = adx(h, l, c, 14)
    e5, e21, e50, e200 = ema(c, 5), ema(c, 21), ema(c, 50), ema(c, 200)
    th = 0.0007 * c
    L = (e5 > e21) & (e50 > e200) & (pb > 0.75) & (ad > 15) & (ao > th)
    S = (e5 < e21) & (e50 < e200) & (pb < 0.25) & (ad > 15) & (ao < -th)
    (sl, ss), (tl, ts) = _pct_levels(c, 2.0, 1.0)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": ("level", tl, ts)}


# ---------------- SC165 freqtrade MACD RSI EMA: histogram zero cross + RSI 50 + EMA5 > EMA20, 1h (GitHub) ----------------
def s_ft_macd_rsi_ema_1h(df):
    """freqtrade 'MACD RSI EMA Strategy' (crypto, 1-hour, long only): MACD(12,26,9) histogram crosses above 0, RSI(14)
    > 50 and EMA5 > EMA20 -> buy. Stop -8%; minimal ROI 5% at once, 3% after 30 min, 1% after 60 min, 0% after 120
    min; trailing 2% from +3%; sell when RSI < 40 or the histogram is under 0 two bars running. Ours: target 3%
    (the ROI from 30 minutes on), a time exit at 120 minutes, the two sell signals; the trailing is not modelled."""
    c = df["c"].to_numpy(float)
    _, _, hist = macd(c, 12, 26, 9)
    r = rsi(c, 14)
    L = cross_up(hist, 0.0) & (r > 50) & (ema(c, 5) > ema(c, 20))
    (sl, _), (tl, _) = _pct_levels(c, 8.0, 3.0)
    nb = max(1, int(round(120 / max(_bar_minutes(df), 1.0))))
    return L, np.zeros(len(c), bool), {"stop": ("level", sl, sl), "target": ("level", tl, tl), "max_bars": nb,
                                       "exit_long": (r < 40) | ((hist < 0) & (_sh(hist) < 0))}


# ---------------- SC166 E-News Lucky: 10:30 straddle +/-20 pips, 50 / 150 pips, flat 22:30, 1m (StockSharp) ----------------
def s_ss_enews_lucky_1030(df):
    """StockSharp 'E-News Lucky' (port of an MT4 EA; FX, 1-minute): at 10:30 a buy stop 20 pips above and a sell stop
    20 pips below the price; once one fills the other is cancelled (one trade a day); stop 50 pips, target 150
    pips, a 5-pip trailing stop; pending orders removed and positions closed at 22:30. Ours: times as UTC (the EA
    uses terminal time), the price = the 10:30 bar's open, the signal the first bar reaching a level, the trailing
    not modelled. Translation (_pips_atr, 1-minute): 20 / 50 / 150 pips -> ATR multiples of the bar's timeframe,
    the 20-pip distance with the ATR of the 10:30 bar."""
    o, h, l, c = _open(df), df["h"].to_numpy(float), df["l"].to_numpy(float), df["c"].to_numpy(float)
    a = atr(h, l, c, 14)
    ref, day = _at_local_time(df, o, "UTC", 10, 30)
    ar, _ = _at_local_time(df, a, "UTC", 10, 30)
    hs, _, _, _, _ = _local(df, "UTC")
    d = _pips_atr(df, 20) * ar
    tm = (hs >= 10.5) & (hs < 22.5)
    L0 = tm & (h >= ref + d)
    S0 = tm & (l <= ref - d)
    first = _first_per_day(L0 | S0, day)
    return L0 & first, S0 & first & ~L0, {"stop": ("atr", _pips_atr(df, 50)), "target": ("atr", _pips_atr(df, 150)),
                                          "force_exit": hs >= 22.5}


# ---------------- shared helpers for SC167-SC172 ----------------
def _roi_plan(df, c, sl_pct, tp_pct, minutes):
    """freqtrade stoploss / minimal_roi -> a fixed % stop, one % target and a time exit after `minutes`."""
    (sl, _), (tl, _) = _pct_levels(c, sl_pct, tp_pct)
    nb = max(1, int(round(minutes / max(_bar_minutes(df), 1.0))))
    return {"stop": ("level", sl, sl), "target": ("level", tl, tl), "max_bars": nb}


# ---------------- SC167 freqtrade Quickie: ADX 30 + TEMA9 under the mid band and rising, under SMA200, 5m (GitHub) ----------------
def s_ft_quickie(df):
    """freqtrade-strategies 'Quickie' (berlinguyinca; crypto spot, 5-minute, long only): ADX(14) > 30, TEMA(9) under
    the Bollinger (20, 2) middle band and rising, SMA200 over the close -> buy. Stop -25%; ROI 15% after 10 min,
    6% after 15, 3% after 30, 1% after 100; sell when ADX > 70 and TEMA is over the middle band and falling.
    Ours: one target 3% (the ROI from 30 minutes on) and a time exit at 100 minutes."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    ad = adx(h, l, c, 14)
    tm = _tema(c, 9)
    _, mid, _ = _bb(c, 20, 2.0)
    L = (ad > 30) & (tm < mid) & (tm > _sh(tm)) & (sma(c, 200) > c)
    p = _roi_plan(df, c, 25.0, 3.0, 100)
    p["exit_long"] = (ad > 70) & (tm > mid) & (tm < _sh(tm))
    return L, np.zeros(len(c), bool), p


# ---------------- SC168 freqtrade Simple: MACD over 0 and signal + widening upper band + RSI7 > 70, 5m (GitHub) ----------------
def s_ft_simple_macd_rsi7(df):
    """freqtrade-strategies 'Simple' (berlinguyinca, after the book 'The Simple Strategy'; crypto spot, 5-minute,
    long only): MACD(12,26,9) > 0 and over its signal, the upper Bollinger band (12, 2) higher than on the previous
    bar and RSI(7) > 70 -> buy. Stop -25%, ROI 1% from the start; sell when RSI(7) > 80."""
    c = df["c"].to_numpy(float)
    m, s, _ = macd(c, 12, 26, 9)
    _, _, ub = _bb(c, 12, 2.0)
    r = rsi(c, 7)
    L = (m > 0) & (m > s) & (ub > _sh(ub)) & (r > 70)
    (sl, _), (tl, _) = _pct_levels(c, 25.0, 1.0)
    return L, np.zeros(len(c), bool), {"stop": ("level", sl, sl), "target": ("level", tl, tl), "exit_long": r > 80}


# ---------------- SC169 freqtrade AdxSmas: ADX 25 + SMA3 crossing SMA6, -25% / 10%, 1h (GitHub) ----------------
def s_ft_adx_smas(df):
    """freqtrade-strategies 'AdxSmas' (berlinguyinca, port of Mynt AdxSmas; crypto spot, 1-hour, long only): ADX(14) >
    25 and SMA3 crosses over SMA6 -> buy. Stop -25%, ROI 10% from the start; sell when ADX < 25 and SMA6 crosses
    over SMA3."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    ad = adx(h, l, c, 14)
    s3, s6 = sma(c, 3), sma(c, 6)
    L = (ad > 25) & cross_up(s3, s6)
    (sl, _), (tl, _) = _pct_levels(c, 25.0, 10.0)
    return L, np.zeros(len(c), bool), {"stop": ("level", sl, sl), "target": ("level", tl, tl),
                                       "exit_long": (ad < 25) & cross_up(s6, s3)}


# ---------------- SC170 freqtrade BinHV27: RSI5-EMA5 <= 20/25 under EMA60/120 with -DI over its EMA25, 5m (GitHub) ----------------
def s_ft_binhv27(df):
    """freqtrade-strategies 'BinHV27' (berlinguyinca; crypto spot, 5-minute, long only): close under EMA120 and EMA60,
    -DI over its EMA25, RSI(5) not falling, and one of four regimes from SMA120 / SMA240 (bigup = SMA120 over SMA240
    by more than close/300, trend turning, SMA240 rising two bars) with ADX > 25-35 and EMA5 of RSI(5) <= 20 (25) ->
    buy. Stop -50%, ROI 100% (effectively signal exits); the five sell branches of the source are coded as published."""
    h, l, c = (df[x].to_numpy(float) for x in "hlc")
    r = np.nan_to_num(rsi(c, 5))
    er = ema(r, 5)
    ad = np.nan_to_num(adx(h, l, c, 14))
    pdi, ndi = _di(h, l, c, 14)
    pdi, ndi = np.nan_to_num(pdi), np.nan_to_num(ndi)
    ndie = ema(ndi, 25)
    lows, highs = ema(c, 60), ema(c, 120)
    fs, ss = np.nan_to_num(sma(c, 120)), np.nan_to_num(sma(c, 240))
    bigup = (fs > ss) & ((fs - ss) > c / 300)
    bigdown = ~bigup
    tr = fs - ss
    pct = tr > _sh(tr)
    pctc = pct & (_sh(tr) > _sh(tr, 2))
    cu = (ss > _sh(ss)) & (_sh(ss) > _sh(ss, 2))
    dl = fs - _sh(fs)
    slow = dl < _sh(dl)
    base = (ss > 0) & (c < highs) & (c < lows) & (ndi > ndie) & (r >= _sh(r))
    L = base & ((~pct & ~cu & (ad > 25) & bigdown & (er <= 20)) | (~pct & cu & (ad > 30) & bigdown & (er <= 20)) |
                (~cu & (ad > 35) & bigup & (er <= 20)) | (cu & (ad > 30) & bigup & (er <= 25)))
    ex = ((~pctc & ~cu & ((c > lows) | (c > highs)) & bigdown) |
          (~pctc & ~cu & (c > highs) & ((er >= 75) | (c > ss)) & bigdown) |
          (~pctc & (c > highs) & (ad > 30) & (er >= 80) & bigup) |
          (pctc & ~cu & slow & (er >= 75)) |
          (pctc & (ndi < pdi) & (c > lows)))
    (sl, _), (tl, _) = _pct_levels(c, 50.0, 100.0)
    return L, np.zeros(len(c), bool), {"stop": ("level", sl, sl), "target": ("level", tl, tl), "exit_long": ex}


# ---------------- SC171 freqtrade AverageStrategy: EMA 8/21 cross, -20% / 50%, 4h (GitHub) ----------------
def s_ft_average_ema8_21(df):
    """freqtrade-strategies 'AverageStrategy' (berlinguyinca; crypto spot, 4-hour, long only): EMA8 crosses over EMA21
    -> buy. Stop -20%, ROI 50% from the start; sell when EMA21 crosses over EMA8. Differs from TT45 / CR07 in the
    periods and the exits."""
    c = df["c"].to_numpy(float)
    e8, e21 = ema(c, 8), ema(c, 21)
    (sl, _), (tl, _) = _pct_levels(c, 20.0, 50.0)
    return cross_up(e8, e21), np.zeros(len(c), bool), {"stop": ("level", sl, sl), "target": ("level", tl, tl),
                                                       "exit_long": cross_up(e21, e8)}


# ---------------- SC172 freqtrade Strategy001: EMA 20/50 cross + green Heikin-Ashi over EMA20, 5m (GitHub) ----------------
def s_ft_strategy001(df):
    """freqtrade-strategies 'Strategy001' (crypto spot, 5-minute, long only): EMA20 crosses over EMA50, the
    Heikin-Ashi close over EMA20 and the HA candle green -> buy. Stop -10%; ROI 5% at once, 4% after 20 min, 3% after
    30, 1% after 60; sell when EMA50 crosses over EMA100 with the HA close under EMA20 and a red HA candle. Ours: one
    target 3% (the ROI from 30 minutes on) and a time exit at 60 minutes."""
    c = df["c"].to_numpy(float)
    hc, ho = _ha(df)
    e20, e50, e100 = ema(c, 20), ema(c, 50), ema(c, 100)
    L = cross_up(e20, e50) & (hc > e20) & (ho < hc)
    p = _roi_plan(df, c, 10.0, 3.0, 60)
    p["exit_long"] = cross_up(e50, e100) & (hc < e20) & (ho > hc)
    return L, np.zeros(len(c), bool), p



# =====================================================================================================================
# 14.11.2026 scout run - the user's TikTok / Facebook videos (replies.json "_tasks" 15, 16, 38, tal1411*) and the
# shared indicator library agents/indicators (user: "not compromising on indicators either"). Every strategy below is
# built from closed bars only; where a video left out the stop / target, a standard one is chosen and SAID so.
# =====================================================================================================================
from agents import indicators as IND  # noqa: E402


def _ohlcv(df):
    """o, h, l, c, v (no-open feeds -> the previous close as the open, as everywhere in the library)."""
    from agents.indicators._core import cols
    return cols(df)


def _plan_r(sl_l, sl_s, r=2.0, **kw):
    p = {"stop": ("level", np.asarray(sl_l, float), np.asarray(sl_s, float)), "target": ("R", float(r))}
    p.update(kw)
    return p


def _plan_lv(sl_l, sl_s, tg_l, tg_s, **kw):
    p = {"stop": ("level", np.asarray(sl_l, float), np.asarray(sl_s, float)),
         "target": ("level", np.asarray(tg_l, float), np.asarray(tg_s, float))}
    p.update(kw)
    return p


def _excl(L, S):
    L = np.asarray(L, bool)
    return L, np.asarray(S, bool) & ~L


# ---------------- SC173 / SC174 Diamond top / bottom (Trader Dhana, Facebook; task 15) ----------------
def _diamond(df, k=2, frac=1.0):
    """Swings (k bars each side, known k bars late): after a prior trend, an expanding part (a higher high AND a lower
    low than the first swings) followed by a contracting part (a lower high AND a higher low) - 4 lines, 2+ touches a
    side = at least 3 highs and 3 lows in the formation (h1 < h2 > h3, l1 > l2 < l3 alternating). Top: prior trend up
    (close 20 bars before the first pivot below it); sell on the first close under the lower-right line (the line
    through l2 and l3, extended); stop over the last swing high inside the diamond (h3) ; target = the diamond's height
    (h2 - l2) x frac projected from the break. Bottom = mirror (prior trend down, buy over the upper-right line)."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    from agents.indicators._core import pivots
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    H, Lw = [], []
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    used = set()
    for t in range(n):
        if phc[t]:
            H.append((phi[t], phv[t]))
        if plc[t]:
            Lw.append((pli[t], plv[t]))
        if len(H) < 3 or len(Lw) < 3:
            continue
        (a1, h1), (a2, h2), (a3, h3) = H[-3:]
        (b1, l1), (b2, l2), (b3, l3) = Lw[-3:]
        if not (h2 > h1 and h2 > h3 and l2 < l1 and l2 < l3):
            continue
        if not (min(a1, b1) < min(a2, b2) and max(a2, b2) < min(a3, b3) + 1 and t - max(a3, b3) <= 30):
            continue
        key = (a2, b2)
        if key in used:
            continue
        start = min(a1, b1)
        hgt = h2 - l2
        if hgt <= 0 or start < 20:
            continue
        m_lo = (l3 - l2) / (b3 - b2) if b3 != b2 else 0.0
        m_hi = (h3 - h2) / (a3 - a2) if a3 != a2 else 0.0
        lo_line = l3 + m_lo * (t - b3)
        hi_line = h3 + m_hi * (t - a3)
        prior_up = c[start - 20] < c[start]
        if prior_up and c[t] < lo_line and (t == 0 or c[t - 1] >= l3 + m_lo * (t - 1 - b3)):
            S[t], sl_s[t], tg_s[t] = True, h3, c[t] - frac * hgt
            used.add(key)
        elif not prior_up and c[t] > hi_line and (t == 0 or c[t - 1] <= h3 + m_hi * (t - 1 - a3)):
            L[t], sl_l[t], tg_l[t] = True, l3, c[t] + frac * hgt
            used.add(key)
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


def s_diamond_full(df):
    """Diamond top / bottom (Trader Dhana, Facebook reel 'Bearish Diamond Pattern', sent by the user - task 15),
    intraday 15m / 30m. After an up trend the range widens (higher highs, lower lows) and then narrows (lower highs,
    higher lows); sell on a close under the lower-right line, stop over the last swing high inside the diamond,
    target = the full height (highest high to lowest low) projected down from the break. Diamond bottom = mirror.
    Ours: 2-bar swings, the formation = the last 3 swing highs and 3 swing lows, prior trend = 20 bars."""
    return _diamond(df, k=2, frac=1.0)


def s_diamond_618(df):
    """SC173 diamond with the user's alternative target: 61.8% of the height (task 15: 'also test 50% / 61.8%')."""
    return _diamond(df, k=2, frac=0.618)


# ---------------- SC175 Liquidity trend-line sweep + FVG + BOS (TheTrader's Villa, Facebook; task 16) ----------------
def s_villa_tl_sweep_fvg(df):
    """TheTrader's Villa 'Liquidity Trendline Break + FVG' (Facebook reel, task 16), 5m - 1h. A falling trend line with
    3+ touches (resistance - buyers' stops above it); price pokes through it (the liquidity grab), then a strong
    bearish candle leaves a bearish FVG (high[i] < low[i-2]) and closes under the last swing low (BOS) -> sell when
    price comes back into the FVG (its middle); stop over the high of the grab; target 2R (the video: the next
    liquidity below / a fixed R). Bullish mirror (rising line with 3+ touches, sweep under it, bullish FVG + BOS over
    the last swing high, buy back in the FVG). Differs from the user's gap method (tal_gap) by the 3-touch line and the
    sweep before the gap. Ours: line = agents/indicators trendlines (pivots 3), touch tolerance 0.3 ATR, the FVG entry
    valid 10 bars, the grab within the 10 bars before the FVG."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    tl = IND.trendlines(df, k=3, tol=0.3)
    sh, sl = _swings(h, l, 3)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    arm_s = arm_l = None  # (fvg_top, fvg_bot, stop, born)
    for i in range(2, n):
        w0 = max(0, i - 10)
        grab_s = np.any((tl["res_touches"][w0:i + 1] >= 3) & (h[w0:i + 1] > tl["res_line"][w0:i + 1]))
        grab_l = np.any((tl["sup_touches"][w0:i + 1] >= 3) & (l[w0:i + 1] < tl["sup_line"][w0:i + 1]))
        if grab_s and h[i] < l[i - 2] and c[i] < o[i] and np.isfinite(sl[i - 1]) and c[i] < sl[i - 1]:
            arm_s = (l[i - 2], h[i], h[w0:i + 1].max(), i)
        if grab_l and l[i] > h[i - 2] and c[i] > o[i] and np.isfinite(sh[i - 1]) and c[i] > sh[i - 1]:
            arm_l = (l[i], h[i - 2], l[w0:i + 1].min(), i)
        if arm_s is not None:
            top, bot, st, b = arm_s
            if i - b > 10 or c[i] > st:
                arm_s = None
            elif i > b and h[i] >= (top + bot) / 2:
                S[i], sl_s[i] = True, st
                arm_s = None
        if arm_l is not None:
            top, bot, st, b = arm_l
            if i - b > 10 or c[i] < st:
                arm_l = None
            elif i > b and l[i] <= (top + bot) / 2:
                L[i], sl_l[i] = True, st
                arm_l = None
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC176 / SC177 VWAP + RSI(14) 30/70 re-cross with a reversal candle (The Trader's Guide; task 38) ------
def _vwap_rsi14(df, target):
    o, h, l, c, v = _ohlcv(df)
    vw = IND.vwap_bands(df)["vwap"]
    r = rsi(c, 14)
    a = atr(h, l, c, 14)
    rev_l = c > _sh(h)
    rev_s = c < _sh(l)
    L = (c < vw) & cross_up(r, 30.0) & rev_l
    S = (c > vw) & cross_down(r, 70.0) & rev_s
    sl_l = _s(l).rolling(5).min().to_numpy() - 0.2 * a
    sl_s = _s(h).rolling(5).max().to_numpy() + 0.2 * a
    L, S = _excl(L, S)
    if target == "vwap":
        return L, S, _plan_lv(sl_l, sl_s, vw, vw)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


def s_tg_vwap_rsi14_to_vwap(df):
    """The Trader's Guide (TikTok post, task 38), 5-15m intraday: buy when price is under the session VWAP, RSI(14) was
    under 30 and crosses back over 30, confirmed by a reversal candle (close over the previous candle's high); sell =
    mirror (over VWAP, RSI back under 70, close under the previous low). No stop / target in the post - ours: stop
    beyond the 5-bar low / high + 0.2 ATR, target = back to the VWAP. Session VWAP from 00:00 UTC on tick volume
    (CFDs have no real volume - noted). Differs from SS05 (RSI(2) + 2-SD) by RSI 14 with the reversal confirmation."""
    return _vwap_rsi14(df, "vwap")


def s_tg_vwap_rsi14_2r(df):
    """SC176 with the post's variant target 2R instead of the VWAP (task 38)."""
    return _vwap_rsi14(df, "2r")


# ---------------- SC178 RSI range-shift pullback (Trading Data, TikTok; task tal1411rsi (b)) ----------------
def s_td_rsi_range_shift(df):
    """Trading Data (TikTok, task tal1411rsi): in a trend RSI does not reach 30 / 70 - in a down trend it lives in
    ~20-60, in an up trend in ~40-80, so 'buy at 30' fails in a down trend. Rule (b): in an up trend buy when RSI(14)
    dips to 40-45 and turns up; in a down trend sell when RSI rises to 55-60 and turns down. Trend (ours, from the
    same day's Market Structure video): close over EMA50 and EMA21 over EMA50 (mirror). No stop / target in the video -
    ours: stop beyond the 10-bar swing extreme, target 2R. agents/indicators rsi_regime."""
    o, h, l, c, v = _ohlcv(df)
    rr = IND.rsi_regime(df)
    a = atr(h, l, c, 14)
    sl_l = _s(l).rolling(10).min().to_numpy() - 0.1 * a
    sl_s = _s(h).rolling(10).max().to_numpy() + 0.1 * a
    L, S = _excl(rr["pull_long"], rr["pull_short"])
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC179 / SC180 Market Structure 'RSI 60-40 setup' A (pullback) and B (breakout) ----------------
def s_ms_rsi40_pullback(df):
    """Market Structure (TikTok 'RSI 60-40 Setup', task tal1411rsi 08:58), 15m / 1h / 4h. Trend up = close over EMA21
    and EMA50 and EMA21 over EMA50. Entry A (pullback): RSI(14) near 40 (<= 45 on the previous bar) and turning up,
    price near EMA21/50 (the low within 0.5 ATR of EMA21 or under it, close over EMA50), on a confirmation candle
    (close over the previous high). Stop under EMA50 (ours: EMA50 - 0.2 ATR); target 1 = the last swing high (ours:
    the whole position at the last confirmed 5-bar swing high, at least 1R away), trail under EMA21 (exit on a close
    under EMA21). Short = mirror (RSI near 60 turning down)."""
    o, h, l, c, v = _ohlcv(df)
    e21, e50 = ema(c, 21), ema(c, 50)
    r = rsi(c, 14)
    a = atr(h, l, c, 14)
    sh, sl = _swings(h, l, 5)
    up = (c > e21) & (c > e50) & (e21 > e50)
    dn = (c < e21) & (c < e50) & (e21 < e50)
    pr = _sh(r)
    L = up & (pr <= 45) & (r > pr) & (l <= e21 + 0.5 * a) & (c > _sh(h))
    S = dn & (pr >= 55) & (r < pr) & (h >= e21 - 0.5 * a) & (c < _sh(l))
    stl, sts = e50 - 0.2 * a, e50 + 0.2 * a
    tgl = np.where(sh - c >= (c - stl), sh, c + (c - stl) * 2)
    tgs = np.where(c - sl >= (sts - c), sl, c - (sts - c) * 2)
    L, S = _excl(L, S)
    return L, S, _plan_lv(stl, sts, tgl, tgs, exit_long=c < e21, exit_short=c > e21)


def s_ms_rsi60_breakout(df):
    """Market Structure 'RSI 60-40 Setup' entry B (breakout), 15m / 1h / 4h: in the up trend (as SC179) RSI(14) over
    60 and a strong candle (body >= 0.6 of its true range, true range >= 1 ATR) closes over the last swing high -> buy; stop
    under EMA21; target 2-3R (ours: 2.5R) with the trail under EMA21 (exit on a close under it). Short = mirror
    (RSI under 40, close under the last swing low)."""
    o, h, l, c, v = _ohlcv(df)
    e21, e50 = ema(c, 21), ema(c, 50)
    r = rsi(c, 14)
    a = atr(h, l, c, 14)
    sh, sl = _swings(h, l, 5)
    up = (c > e21) & (c > e50) & (e21 > e50)
    dn = (c < e21) & (c < e50) & (e21 < e50)
    pc = _sh(c)
    rng = np.fmax(h, pc) - np.fmin(l, pc)  # true range (a gap counts)
    strong = (np.abs(c - o) >= 0.6 * rng) & (rng >= a)
    L = up & (r > 60) & strong & (c > o) & cross_up(c, sh)
    S = dn & (r < 40) & strong & (c < o) & cross_down(c, sl)
    L, S = _excl(L, S)
    return L, S, _plan_r(e21 - 0.1 * a, e21 + 0.1 * a, 2.5, exit_long=c < e21, exit_short=c > e21)


# ---------------- SC181 RSI divergence with the big trend + RSI 50 confirmation (The Trader's Guide) ----------------
def s_tg_rsi_div_trend50(df):
    """The Trader's Guide ('5 RSI ideas', task tal1411rsi 09:16): RSI divergence only in the direction of the big
    trend, confirmed by RSI crossing 50. Long: close over EMA200, a confirmed swing low (5 bars) lower than the previous
    one with a HIGHER RSI(14) at it (bullish divergence), then RSI crosses over 50 within 20 bars -> buy. Short =
    mirror (under EMA200, bearish divergence, RSI under 50). No stop / target in the video - ours: stop under the
    divergence low (over the high), target 2R."""
    o, h, l, c, v = _ohlcv(df)
    from agents.indicators._core import pivots
    r = rsi(c, 14)
    e200 = ema(c, 200)
    phc, plc, phv, plv, phi, pli = pivots(h, l, 5)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    lj = hj = None
    arm_l = arm_s = None
    c50u, c50d = cross_up(r, 50.0), cross_down(r, 50.0)
    for t in range(n):
        if plc[t]:
            j = pli[t]
            if lj is not None and l[j] < l[lj] and r[j] > r[lj]:
                arm_l = (t, l[j])
            lj = j
        if phc[t]:
            j = phi[t]
            if hj is not None and h[j] > h[hj] and r[j] < r[hj]:
                arm_s = (t, h[j])
            hj = j
        if arm_l is not None and t - arm_l[0] <= 20 and c50u[t] and c[t] > e200[t]:
            L[t], sl_l[t] = True, arm_l[1]
            arm_l = None
        if arm_s is not None and t - arm_s[0] <= 20 and c50d[t] and c[t] < e200[t]:
            S[t], sl_s[t] = True, arm_s[1]
            arm_s = None
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC182 VWAP + EMA close beyond both, exit back beyond both (The Trader's Guide) ----------------
def s_tg_vwap_ema_both(df):
    """The Trader's Guide (TikTok, task tal1411vwap (1)), intraday: sell when a candle closes under the session VWAP
    AND under the EMA; exit when a candle closes over both; buy = mirror. Ours: EMA 20 (the video does not say the
    length), the first close beyond both (onset), stop 1.5 ATR (no stop in the video), no fixed target - the exit
    rule. Tick volume VWAP (CFDs, noted)."""
    o, h, l, c, v = _ohlcv(df)
    vw = IND.vwap_bands(df)["vwap"]
    e = ema(c, 20)
    below = (c < vw) & (c < e)
    above = (c > vw) & (c > e)
    L, S = _excl(_onset(above), _onset(below))
    return L, S, {"stop": ("atr", 1.5), "target": None, "exit_long": below, "exit_short": above}


# ---------------- SC183 Fibonacci 0.5-0.786 pullback + EMA9 reclaim in the EMA50/200 trend (The Trader's Guide) ------
def s_tg_fib_ema9_reclaim(df):
    """The Trader's Guide (TikTok, task tal1411vwap (2)): in a trend (EMA50 over EMA200) wait for a pullback into the
    0.5-0.786 zone of the last move (confirmed 5-bar swing low -> swing high), enter only after the price closes back
    over EMA9; stop under the pullback low; target the swing high / 2R (ours: the swing high when it is >= 1R away,
    else 2R). Short = mirror. Differs from SW16 / TT12 by the EMA9-reclaim trigger and the 0.786 floor."""
    o, h, l, c, v = _ohlcv(df)
    from agents.indicators._core import pivots
    e9, e50, e200 = ema(c, 9), ema(c, 50), ema(c, 200)
    phc, plc, phv, plv, phi, pli = pivots(h, l, 5)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    lo = hi = None  # (idx, price)
    touched_l = touched_s = None
    for t in range(n):
        if plc[t]:
            lo = (pli[t], plv[t])
        if phc[t]:
            hi = (phi[t], phv[t])
        if lo and hi and lo[0] < hi[0] and e50[t] > e200[t]:
            mv = hi[1] - lo[1]
            z1, z2 = hi[1] - 0.5 * mv, hi[1] - 0.786 * mv
            if l[t] <= z1 and l[t] >= z2:
                touched_l = min(touched_l, l[t]) if touched_l is not None else l[t]
            if l[t] < z2:
                touched_l = None
            if t > 0 and touched_l is not None and c[t] > e9[t] and c[t - 1] <= e9[t - 1]:
                risk = c[t] - touched_l
                if risk > 0:
                    L[t], sl_l[t] = True, touched_l
                    tg_l[t] = hi[1] if hi[1] - c[t] >= risk else c[t] + 2 * risk
                touched_l = None
        else:
            touched_l = None
        if lo and hi and hi[0] < lo[0] and e50[t] < e200[t]:
            mv = hi[1] - lo[1]
            z1, z2 = lo[1] + 0.5 * mv, lo[1] + 0.786 * mv
            if h[t] >= z1 and h[t] <= z2:
                touched_s = max(touched_s, h[t]) if touched_s is not None else h[t]
            if h[t] > z2:
                touched_s = None
            if t > 0 and touched_s is not None and c[t] < e9[t] and c[t - 1] >= e9[t - 1]:
                risk = touched_s - c[t]
                if risk > 0:
                    S[t], sl_s[t] = True, touched_s
                    tg_s[t] = lo[1] if c[t] - lo[1] >= risk else c[t] - 2 * risk
                touched_s = None
        else:
            touched_s = None
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


# ---------------- SC184 Ichimoku: over the cloud + TK cross + Chikou over the price (Quant Tactics) ----------------
def s_qt_ichimoku_chikou(df):
    """Quant Tactics (TikTok, '2944% backtest' on crypto - probably exaggerated; task tal1411vwap (3)): buy when price
    is over the cloud, Tenkan crosses over Kijun and the Chikou span is over the price (= today's close over the close
    26 bars ago - no look-ahead); sell = mirror. Ichimoku 9/26/52. No stop / target in the video - ours: stop at the
    Kijun (or 1.5 ATR when the Kijun is closer than 0.5 ATR), exit on the opposite TK cross, no fixed target.
    Differs from TV20 (no Chikou rule) and GH13 (7/14/30)."""
    o, h, l, c, v = _ohlcv(df)
    ic = IND.ichimoku(df)
    a = atr(h, l, c, 14)
    L = ic["above_cloud"] & ic["tk_up"] & ic["chikou_up"]
    S = ic["below_cloud"] & ic["tk_dn"] & ic["chikou_dn"]
    kj = ic["kijun"]
    sl_l = np.where(c - kj >= 0.5 * a, kj, c - 1.5 * a)
    sl_s = np.where(kj - c >= 0.5 * a, kj, c + 1.5 * a)
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", sl_l, sl_s), "target": None, "exit_long": ic["tk_dn"], "exit_short": ic["tk_up"]}


# ---------------- SC185 EMA 9/21 cross + VWAP + ADX > 20, break of the confirmation high (The Trader's Guide) --------
def s_tg_ema9_21_vwap_adx_break(df):
    """The Trader's Guide 'ADX 80%' (TikTok, task tal1411set2 (1)): EMA9 crosses over EMA21, price over EMA9 and over
    the VWAP, ADX(14) over 20 on the confirmation candle -> buy on a break of the confirmation candle's high (within 3
    bars); stop under the confirmation candle (or EMA21, the lower), target 2R (the video: 1.5-2R or exit on the cross
    back - we also exit on EMA9 crossing back under EMA21). Sell = mirror. Differs from SC008 (EMA 5/13, entry at the
    cross) by EMA 9/21 and the stop-entry over the confirmation high."""
    o, h, l, c, v = _ohlcv(df)
    e9, e21 = ema(c, 9), ema(c, 21)
    vw = IND.vwap_bands(df)["vwap"]
    ad = adx(h, l, c, 14)
    n = len(c)
    conf_l = cross_up(e9, e21) & (c > e9) & (c > vw) & (ad > 20)
    conf_s = cross_down(e9, e21) & (c < e9) & (c < vw) & (ad > 20)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(1, n):
        for k in (1, 2, 3):
            j = t - k
            if j < 0:
                break
            if conf_l[j] and c[t] > h[j] and not np.any(c[j + 1:t] > h[j]):
                L[t], sl_l[t] = True, min(l[j], e21[t])
            if conf_s[j] and c[t] < l[j] and not np.any(c[j + 1:t] < l[j]):
                S[t], sl_s[t] = True, max(h[j], e21[t])
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0, exit_long=cross_down(e9, e21), exit_short=cross_up(e9, e21))


# ---------------- SC186 Gold 15m liquidity sweep and reclaim, target the previous swing (sphula) ----------------
def s_sphula_sweep_reclaim(df):
    """sphula (TikTok, gold 15m, task tal1411set2 (2)): price runs under a previous swing low and comes back over it -
    buy on the reclaim (the close back over the low) or on the retest of the zone; stop under the sweep low; target the
    previous swing high. Sell = mirror. Ours: swings of 5 bars (agents/indicators liquidity), entry at the reclaim
    close, target = the last confirmed swing high when it is >= 1R away (else 2R). Differs from SC151 (pivot-5 + SMA50
    filter, 2R) and TV11 (the previous DAY's low) by the swing-high target and no trend filter."""
    o, h, l, c, v = _ohlcv(df)
    lq = IND.liquidity(df, k=5)
    sh, sl = _swings(h, l, 5)
    L, S = lq["sweep_bull"], lq["sweep_bear"]
    risk_l, risk_s = c - l, h - c
    tgl = np.where(sh - c >= risk_l, sh, c + 2 * risk_l)
    tgs = np.where(c - sl >= risk_s, sl, c - 2 * risk_s)
    L, S = _excl(L, S)
    return L, S, _plan_lv(l, h, tgl, tgs)


# ---------------- SC187 Two EMAs + buyer / seller pressure flip (ProTrippa) ----------------
def s_pt_ma_pressure(df):
    """ProTrippa 'Forget MACD & RSI' (TikTok, task tal1411set3 (1)): two moving averages give the direction (ours:
    EMA20 over EMA50 and both rising = up); enter only when the buyers-vs-sellers pressure oscillator flips to the side
    of the trend (agents/indicators buy_sell_pressure: close location x tick volume - CFDs have no real volume, noted).
    No stop / target in the video - ours: stop beyond the 10-bar swing, target 2R, exit when the pressure flips back."""
    o, h, l, c, v = _ohlcv(df)
    e20, e50 = ema(c, 20), ema(c, 50)
    bp = IND.buy_sell_pressure(df)
    a = atr(h, l, c, 14)
    up = (e20 > e50) & (e20 > _sh(e20)) & (e50 > _sh(e50))
    dn = (e20 < e50) & (e20 < _sh(e20)) & (e50 < _sh(e50))
    L, S = _excl(up & bp["buy_flip"], dn & bp["sell_flip"])
    return L, S, _plan_r(_s(l).rolling(10).min().to_numpy() - 0.1 * a, _s(h).rolling(10).max().to_numpy() + 0.1 * a,
                         2.0, exit_long=bp["sell_flip"], exit_short=bp["buy_flip"])


# ---------------- SC188 MFI(14) out of 20 / 80 with the trend (Quant Tactics) ----------------
def s_qt_mfi14_trend(df):
    """Quant Tactics (TikTok, crypto results 300%+ probably exaggerated; task tal1411set3 (2)): Money Flow Index (14)
    - buy when it leaves the zone under 20 (crosses back over 20) in an up trend, sell when it crosses back under 80 in
    a down trend. Trend (ours) = EMA50 vs EMA200 (the dip under 20 often takes the price itself under a long average). Tick volume (CFDs). No stop / target - ours: stop beyond the 10-bar
    swing, target 2R. Differs from SC129 (MFI(3) = 100 / 0 momentum)."""
    o, h, l, c, v = _ohlcv(df)
    m = IND.mfi(df, 14)
    e50, e200 = ema(c, 50), ema(c, 200)
    a = atr(h, l, c, 14)
    L, S = _excl(m["exit_os"] & (e50 > e200), m["exit_ob"] & (e50 < e200))
    return L, S, _plan_r(_s(l).rolling(10).min().to_numpy() - 0.1 * a, _s(h).rolling(10).max().to_numpy() + 0.1 * a, 2.0)


# ---------------- SC189 / SC190 Support / resistance retest (Flux Charts S&R; Lima Signal) ----------------
def s_flux_sr_retest_confirm(df):
    """Flux Charts 'S&R Strategy' (TikTok, task tal1411set4 (1)): automatic S/R levels (swings tested 2+ times); enter
    on the retest of a level that broke, only with a second indicator's confirmation (ours: EMA50 slope in the trade's
    direction AND RSI(14) on the right side of 50); stop beyond the level; target the next level. Ours: agents/indicators
    sr_levels (pivots 5, zone 0.25 ATR), the level must have had 2+ tests before the break; stop = level -/+ 0.5 ATR;
    target = the next resistance (support) when >= 1R away, else 2R."""
    o, h, l, c, v = _ohlcv(df)
    sr = IND.sr_levels(df, k=5)
    a = atr(h, l, c, 14)
    e50 = ema(c, 50)
    r = rsi(c, 14)
    lvl = sr["retest_level"]
    tests = np.nan_to_num(sr["retest_tests"])
    L = sr["retest_long"] & (tests >= 2) & (e50 > _sh(e50)) & (r > 50)
    S = sr["retest_short"] & (tests >= 2) & (e50 < _sh(e50)) & (r < 50)
    sl_l, sl_s = lvl - 0.5 * a, lvl + 0.5 * a
    res, sup = sr["res"], sr["sup"]
    tgl = np.where(np.isfinite(res) & (res - c >= c - sl_l), res, c + 2 * (c - sl_l))
    tgs = np.where(np.isfinite(sup) & (c - sup >= sl_s - c), sup, c - 2 * (sl_s - c))
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tgl, tgs)


def s_lima_break_retest_3tests(df):
    """Lima Signal (TikTok, task tal1411set4 09:31) break and retest: a support that held 3 pullbacks; a candle closes
    under it; price comes back to it from below and is rejected -> sell; stop over the swing high before the break;
    target the established low (ours: 2R - the video's 'established low' is the next support, taken when it is >= 1R
    away). Buy = mirror (resistance held 3 times, broken up, retested from above). agents/indicators sr_levels."""
    o, h, l, c, v = _ohlcv(df)
    sr = IND.sr_levels(df, k=5)
    a = atr(h, l, c, 14)
    tests = np.nan_to_num(sr["retest_tests"])
    sw = sr["pre_break_swing"]
    lvl = sr["retest_level"]
    S = sr["retest_short"] & (tests >= 3)
    L = sr["retest_long"] & (tests >= 3)
    sl_s = np.where(np.isfinite(sw) & (sw > c), sw, lvl + 0.5 * a)
    sl_l = np.where(np.isfinite(sw) & (sw < c), sw, lvl - 0.5 * a)
    sup, res = sr["sup"], sr["res"]
    tgs = np.where(np.isfinite(sup) & (c - sup >= sl_s - c), sup, c - 2 * (sl_s - c))
    tgl = np.where(np.isfinite(res) & (res - c >= c - sl_l), res, c + 2 * (c - sl_l))
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tgl, tgs)


# ---------------- SC191 EMA 9/21/50/100/200 graded pullback (The Trader's Guide) ----------------
def s_tg_ema_stack_pullback(df):
    """The Trader's Guide (TikTok, task tal1411set4 (2)): the order of EMA 9/21/50/100/200 grades the trend - all five
    in order = strong trend, pullbacks go to EMA50/100; a normal trend (price over EMA200, 9 > 21 > 50) pulls back to
    EMA21. Buy when the low touches the EMA of the grade (strong: EMA50, normal: EMA21) and the candle closes back over
    it; sell = mirror. No stop / target - ours: stop 0.5 ATR beyond the next slower EMA (strong: EMA100, normal: EMA50),
    target 2R. agents/indicators ema_stack."""
    o, h, l, c, v = _ohlcv(df)
    es = IND.ema_stack(df)
    E = es["lines"]
    a = atr(h, l, c, 14)
    strong_u, strong_d = es["full_up"], es["full_dn"]
    norm_u = ~strong_u & (c > E[200]) & (E[9] > E[21]) & (E[21] > E[50])
    norm_d = ~strong_d & (c < E[200]) & (E[9] < E[21]) & (E[21] < E[50])
    Ls = strong_u & (l <= E[50]) & (c > E[50])
    Ln = norm_u & (l <= E[21]) & (c > E[21])
    Ss = strong_d & (h >= E[50]) & (c < E[50])
    Sn = norm_d & (h >= E[21]) & (c < E[21])
    sl_l = np.where(Ls, E[100] - 0.5 * a, E[50] - 0.5 * a)
    sl_s = np.where(Ss, E[100] + 0.5 * a, E[50] + 0.5 * a)
    L, S = _excl(Ls | Ln, Ss | Sn)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC192 Falling trend-line touch, tight stop, far target (Dan-Chi, US500 1m) ----------------
def s_danchi_tl_touch_far(df):
    """Dan-Chi (TikTok, US500 1-minute, task tal1411set4 09:37): sell at the touch of a falling trend line with a tight
    stop over it and a far target. Ours: automatic line through the last two lower confirmed swing highs (agents/
    indicators trendlines, pivots 5), the high reaches the line and the candle closes under it -> sell; stop = the line
    + 0.3 ATR; target 4R ('far' - ours). Buy = mirror on a rising support line."""
    o, h, l, c, v = _ohlcv(df)
    tl = IND.trendlines(df, k=5)
    a = atr(h, l, c, 14)
    S = tl["touch_res"] & (tl["res_slope"] < 0)
    L = tl["touch_sup"] & (tl["sup_slope"] > 0)
    L, S = _excl(L, S)
    return L, S, _plan_r(tl["sup_line"] - 0.3 * a, tl["res_line"] + 0.3 * a, 4.0)


# ---------------- SC193 EMA20/50 + RSI 50 pullback to EMA20, exit on the 20/50 cross back (Market Structure) ----------
def s_ms_ema20_50_rsi_pullback(df):
    """Market Structure 'pullbacks EMA20/50 + RSI' (TikTok, task tal1411set6 (2)): EMA20 over EMA50 and RSI(14) over
    50 -> buy when the price touches EMA20 (the low reaches it, the close stays over it) with RSI still over 50; exit
    when EMA20 closes under EMA50 or the price closes under EMA50. Sell = mirror. No stop in the video - ours: EMA50 -
    0.5 ATR, no fixed target. Differs from SC057 (needs an engulfing / pin bar and 2R)."""
    o, h, l, c, v = _ohlcv(df)
    e20, e50 = ema(c, 20), ema(c, 50)
    r = rsi(c, 14)
    a = atr(h, l, c, 14)
    L = (e20 > e50) & (r > 50) & (l <= e20) & (c > e20)
    S = (e20 < e50) & (r < 50) & (h >= e20) & (c < e20)
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", e50 - 0.5 * a, e50 + 0.5 * a), "target": None,
                  "exit_long": (e20 < e50) | (c < e50), "exit_short": (e20 > e50) | (c > e50)}


# ---------------- SC194 POC retest (Market Structure '8 types of retest'; task tal1411set6 (3)) ----------------
def s_ms_poc_retest(df):
    """Market Structure '8 types of retest' (TikTok, task tal1411set6 (3)) - the POC kind: once the price has closed
    over the previous session's POC (bullish POC), the first return to it that is rejected (the low reaches the POC,
    the close is back over it and over the open = rejection candle) -> buy; mirror for a bearish POC. Entry only after
    the confirmation, as the video says. Ours: previous-UTC-session POC from the tick-volume profile (agents/
    indicators volume_profile - CFD tick volume, noted), stop 0.5 ATR beyond the POC, target 2R, one per side a day."""
    o, h, l, c, v = _ohlcv(df)
    vp = IND.volume_profile(df)
    poc = vp["poc"]
    a = atr(h, l, c, 14)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    was_above = pd.Series((c > poc + 0.25 * a).astype(float)).groupby(day).cummax().to_numpy() > 0
    was_below = pd.Series((c < poc - 0.25 * a).astype(float)).groupby(day).cummax().to_numpy() > 0
    L = _shb(was_above) & (l <= poc) & (c > poc) & (c > o)
    S = _shb(was_below) & (h >= poc) & (c < poc) & (c < o)
    L, S = _excl(_first_per_day(L, day), _first_per_day(S, day))
    return L, S, _plan_r(poc - 0.5 * a, poc + 0.5 * a, 2.0)


# ---------------- SC195 RSI divergence, then a structure break with expanding volume (Market Structure) -------------
def s_ms_div_then_bos(df):
    """Market Structure 'RSI divergence is a warning, not an entry' (TikTok, task tal1411set6 (4)): after a bullish
    RSI(14) divergence (lower swing low, higher RSI - pivots of 5 bars, known 5 bars late) wait for a break of
    structure - a close over the last swing high - with expanding volume (tick volume over its 20-bar average; a feed
    without volume passes) -> buy; bearish mirror. Valid 30 bars after the divergence. No stop / target in the video -
    ours: stop under the divergence low, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    from agents.indicators._core import pivots
    r = rsi(c, 14)
    sh, sl = _swings(h, l, 5)
    vok = _vol_ok(v, 20, 1.0)
    phc, plc, phv, plv, phi, pli = pivots(h, l, 5)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    lj = hj = None
    arm_l = arm_s = None
    for t in range(1, n):
        if plc[t]:
            j = pli[t]
            if lj is not None and l[j] < l[lj] and r[j] > r[lj]:
                arm_l = (t, l[j])
            lj = j
        if phc[t]:
            j = phi[t]
            if hj is not None and h[j] > h[hj] and r[j] < r[hj]:
                arm_s = (t, h[j])
            hj = j
        if arm_l and t - arm_l[0] <= 30 and np.isfinite(sh[t]) and c[t] > sh[t] and c[t - 1] <= sh[t] and vok[t]:
            L[t], sl_l[t] = True, arm_l[1]
            arm_l = None
        if arm_s and t - arm_s[0] <= 30 and np.isfinite(sl[t]) and c[t] < sl[t] and c[t - 1] >= sl[t] and vok[t]:
            S[t], sl_s[t] = True, arm_s[1]
            arm_s = None
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC196 Bollinger poke back inside at a high-volume area (Flux 'Bollinger + Volume Profile') ---------
def s_flux_bb_vp(df):
    """Flux Charts 'Bollinger + Volume Profile' (TikTok, task tal1411set6 (5)): a candle outside the lower Bollinger
    band (20, 2) inside a high-volume area / the POC that closes back inside the bands -> buy; target the middle band
    (variant: the other band). Sell = mirror at the upper band. Ours: 'high-volume area' = within the previous
    session's value area (VAL..VAH) or within 0.5 ATR of its POC or an HVN (tick-volume profile, noted); entry at the
    close back inside; stop 0.3 ATR beyond the poke's extreme; target the middle band."""
    o, h, l, c, v = _ohlcv(df)
    lo, mid, up = _bb(c, 20, 2.0)
    vp = IND.volume_profile(df)
    a = atr(h, l, c, 14)
    in_va = (c >= vp["val"]) & (c <= vp["vah"])
    near = (np.abs(c - vp["poc"]) <= 0.5 * a) | (np.abs(c - vp["hvn_below"]) <= 0.5 * a) | \
           (np.abs(c - vp["hvn_above"]) <= 0.5 * a)
    hv = in_va | near
    L = hv & (_sh(l) < _sh(lo)) & (c > lo) & (_sh(c) < _sh(lo))
    S = hv & (_sh(h) > _sh(up)) & (c < up) & (_sh(c) > _sh(up))
    sl_l = np.fmin(l, _sh(l)) - 0.3 * a
    sl_s = np.fmax(h, _sh(h)) + 0.3 * a
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, mid, mid)


# ---------------- SC197 Triangle breakout (16 chart patterns video; task tal1411set6 (6)) ----------------
def s_triangle_breakout(df):
    """'16 chart patterns' (TikTok, task tal1411set6 (6)) - the triangles, which the library did not have: symmetric
    (falling highs + rising lows), ascending (flat highs + rising lows) and descending (falling highs + flat lows).
    Buy on a close over the upper line, sell on a close under the lower line; stop on the other side of the triangle
    at the breakout bar (the opposite line); target = the triangle's height at its start projected from the break
    (the classic measured move). Ours: lines + pattern from agents/indicators trendlines (pivots 4); ascending only
    long, descending only short, symmetric both."""
    o, h, l, c, v = _ohlcv(df)
    tl = IND.trendlines(df, k=4)
    p = np.nan_to_num(tl["pattern"])
    pp = _sh(p)
    hgt = tl["height"]
    L = tl["break_res"] & np.isin(pp, [1, 2]) & np.isfinite(hgt)
    S = tl["break_sup"] & np.isin(pp, [1, 3]) & np.isfinite(hgt)
    L, S = _excl(L, S)
    return L, S, _plan_lv(_sh(tl["sup_line"]), _sh(tl["res_line"]), c + hgt, c - hgt)


# ---------------- SC198 Triple top / bottom (16 chart patterns) ----------------
def s_triple_top_bottom(df):
    """'16 chart patterns' (TikTok, task tal1411set6 (6)): triple top = three swing highs within 1 ATR of each other
    with the two lows between them (the neckline = the lower of those lows); sell on a close under the neckline; stop
    over the highest of the three tops; target = the pattern's height under the neckline. Triple bottom = mirror.
    Ours: swings of 4 bars (known 4 bars late), tops within 1 ATR, the pattern must break within 40 bars of the
    third top."""
    o, h, l, c, v = _ohlcv(df)
    from agents.indicators._core import pivots
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, phi, pli = pivots(h, l, 4)
    n = len(c)
    H, Lw = [], []
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    arm_s = arm_l = None
    for t in range(n):
        if phc[t]:
            H.append((phi[t], phv[t]))
            if len(H) >= 3:
                (x1, y1), (x2, y2), (x3, y3) = H[-3:]
                lows = [p for (j, p) in Lw if x1 < j < x3]
                if len(lows) >= 2 and max(y1, y2, y3) - min(y1, y2, y3) <= 1.0 * a[t]:
                    neck = min(lows)
                    arm_s = (neck, max(y1, y2, y3), t)
        if plc[t]:
            Lw.append((pli[t], plv[t]))
            if len(Lw) >= 3:
                (x1, y1), (x2, y2), (x3, y3) = Lw[-3:]
                highs = [p for (j, p) in H if x1 < j < x3]
                if len(highs) >= 2 and max(y1, y2, y3) - min(y1, y2, y3) <= 1.0 * a[t]:
                    neck = max(highs)
                    arm_l = (neck, min(y1, y2, y3), t)
        if arm_s and t - arm_s[2] <= 40:
            neck, top, _ = arm_s
            if c[t] < neck:
                S[t], sl_s[t], tg_s[t] = True, top, neck - (top - neck)
                arm_s = None
            elif c[t] > top:
                arm_s = None
        if arm_l and t - arm_l[2] <= 40:
            neck, bot, _ = arm_l
            if c[t] > neck:
                L[t], sl_l[t], tg_l[t] = True, bot, neck + (neck - bot)
                arm_l = None
            elif c[t] < bot:
                arm_l = None
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


# ---------------- SC199 Bull / bear flag (16 chart patterns) ----------------
def s_flag_breakout(df):
    """'16 chart patterns' (TikTok, task tal1411set6 (6)) - flags: a pole (a move of >= 3 ATR within <= 10 bars) then a
    flag of 5-20 bars that retraces under 50% of the pole; buy on a close over the flag's high (bull flag), stop under
    the flag's low, target = the pole's length from the breakout (measured move). Bear flag = mirror. Ours: the flag =
    the bars since the pole's extreme; ATR(14) at the pole."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    for t in range(40, n):
        for f in range(5, 21):
            pk = t - f            # pole extreme bar (the flag is pk+1 .. t-1)
            if pk - 10 < 0:
                break
            fl_h, fl_l = h[pk + 1:t].max(), l[pk + 1:t].min()
            # bull: pole up into pk
            base = l[pk - 10:pk + 1].min()
            pole = h[pk] - base
            if h[pk] >= h[pk - 10:t].max() and pole >= 3 * a[pk] and fl_l >= h[pk] - 0.5 * pole and fl_h <= h[pk] \
                    and c[t] > fl_h and c[t - 1] <= fl_h:
                L[t], sl_l[t], tg_l[t] = True, fl_l, c[t] + pole
                break
            top = h[pk - 10:pk + 1].max()
            pole_d = top - l[pk]
            if l[pk] <= l[pk - 10:t].min() and pole_d >= 3 * a[pk] and fl_h <= l[pk] + 0.5 * pole_d \
                    and fl_l >= l[pk] and c[t] < fl_l and c[t - 1] >= fl_l:
                S[t], sl_s[t], tg_s[t] = True, fl_h, c[t] - pole_d
                break
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


# ---------------- SC200 Daily pivot bias + pullback to P (Ezekiel Chew) ----------------
def s_ec_daily_pivot_bias(df):
    """Ezekiel Chew (TikTok, task tal1411set6 (7)): daily pivot points as support / resistance and trend - price over
    P = bullish. Rules (ours, from the video's idea): after the price has closed over the day's P (traditional, from
    the previous UTC day - agents/indicators pivot_points), the first pullback that touches P and closes back over it
    -> buy, target R1 (R2 when R1 is under 1R away), stop under S1 or 1 ATR under P, the nearer. Mirror for shorts
    under P (target S1). One per side a day."""
    o, h, l, c, v = _ohlcv(df)
    pv = IND.pivot_points(df)
    a = atr(h, l, c, 14)
    P, R1, R2, S1, S2 = pv["P"], pv["R1"], pv["R2"], pv["S1"], pv["S2"]
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    ab = pd.Series((c > P + 0.2 * a).astype(float)).groupby(day).cummax().to_numpy() > 0
    be = pd.Series((c < P - 0.2 * a).astype(float)).groupby(day).cummax().to_numpy() > 0
    L = _shb(ab) & (l <= P) & (c > P)
    S = _shb(be) & (h >= P) & (c < P)
    sl_l = np.fmax(S1, P - a)
    sl_s = np.fmin(R1, P + a)
    tgl = np.where(R1 - c >= c - sl_l, R1, R2)
    tgs = np.where(c - S1 >= sl_s - c, S1, S2)
    L, S = _excl(_first_per_day(L, day), _first_per_day(S, day))
    return L, S, _plan_lv(sl_l, sl_s, tgl, tgs)


# ---------------- SC201 / SC202 Range-box breakout with buyer volume (Flux 'Easy Strategy') ----------------
def _box_break(df, mode):
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    bh, bl = _sh(_s(h).rolling(20).max().to_numpy()), _sh(_s(l).rolling(20).min().to_numpy())
    tight = (bh - bl) <= 5.0 * _sh(a)
    if mode == "pct":
        bp = IND.buy_sell_pressure(df)["buy_pct"]
        vl, vs = bp >= 0.8, bp <= 0.2
    else:
        bt = IND.big_trades(df, z=2.0)
        flat = _s(v).rolling(50).std(ddof=0).to_numpy() == 0
        vl = bt["big_buy"] | (flat & (c > _sh(c)))
        vs = bt["big_sell"] | (flat & (c < _sh(c)))
    L = tight & (c > bh) & vl
    S = tight & (c < bl) & vs
    mid = (bh + bl) / 2
    L, S = _excl(L, S)
    return L, S, _plan_lv(mid, mid, c + (bh - bl), c - (bh - bl))


def s_flux_box_break_buy_pct(df):
    """Flux Charts 'Easy Strategy, two free indicators' (TikTok, task tal1411set6 (8)): breakout from a range box
    confirmed by the buyers' share of the breakout candle's volume (83% buyers in the video). Ours: box = the last 20
    bars' high-low, tight = height <= 5 ATR; buy on a close over the box with the bar's buy share >= 80% (close location in the
    true range - no real volume / order flow, noted), sell mirror (<= 20%); stop at the box middle; target = the box
    height from the close."""
    return _box_break(df, "pct")


def s_meks_box_break_big_trade(df):
    """SC201's box breakout confirmed instead by a 'big trade' (Meks Trades, task tal1411ind 10:08): tick volume
    z-score >= 2 on the breakout candle closing in its upper (lower) half (agents/indicators big_trades; a feed with
    constant volume uses the bar's direction only). Same box, stop and target as SC201."""
    return _box_break(df, "big")


# ---------------- SC203 OTE in the discount (Forex Bull Signals, gold) ----------------
def s_fbs_ote_discount(df):
    """Forex Bull Signals (TikTok, gold, task tal1411set6 (9)): fib of the swing; buy only under 50% (discount), the
    entry zone is the 61.8-79% retracement (OTE). Ours: dealing range = the last confirmed 10-bar swing low -> swing
    high (up leg, agents/indicators premium_discount), the close in the OTE zone (21-38.2% of the range from the low)
    on a bullish candle (close over open and over the previous close) -> buy; stop under the swing low - 0.2 ATR;
    target the swing high. Sell = mirror in the premium of a down leg."""
    o, h, l, c, v = _ohlcv(df)
    pdz = IND.premium_discount(df, k=10)
    sm = IND.smc(df, k=10)
    a = atr(h, l, c, 14)
    lo, hi = sm["swing_lo"], sm["swing_hi"]
    L = pdz["up_leg"] & pdz["ote_long"] & (c > o) & (c > _sh(c))
    S = ~pdz["up_leg"] & pdz["ote_short"] & (c < o) & (c < _sh(c))
    L, S = _excl(L, S)
    return L, S, _plan_lv(lo - 0.2 * a, hi + 0.2 * a, hi, lo)


# ---------------- SC204 First touch of the previous session's POC, reverse (sixlots) ----------------
def s_sixlots_poc_first_touch(df):
    """sixlots (TikTok, task tal1411set6 (10)): the POC as a reversal level - a limit order at the previous session's
    POC, reverse entry at the FIRST touch. Coming from above -> buy at the POC, from below -> sell. Ours: the session
    opens on one side of the previous UTC session's POC (agents/indicators volume_profile, tick volume - noted); the
    first bar that reaches it enters at the close of that bar (a bar-close approximation of the limit), stop 1 ATR
    beyond the POC, target 2R; one a day."""
    o, h, l, c, v = _ohlcv(df)
    vp = IND.volume_profile(df)
    poc = vp["poc"]
    a = atr(h, l, c, 14)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    first_o = pd.Series(o).groupby(day).transform("first").to_numpy()
    touch = (l <= poc) & (h >= poc)
    ft = _first_per_day(touch, day)
    L = ft & (first_o > poc) & (c >= poc - 0.5 * a)
    S = ft & (first_o < poc) & (c <= poc + 0.5 * a)
    L, S = _excl(L, S)
    return L, S, _plan_r(poc - a, poc + a, 2.0)


# ---------------- SC205 ADX + DI cross + EMA50 (The Trader's Guide) ----------------
def s_tg_adx_di_ema50(df):
    """The Trader's Guide (TikTok, task tal1411set6 (11)): price over EMA50, +DI crosses over -DI, ADX(14) over 20 and
    rising -> buy (sell = mirror). Ours: the cross may come up to 3 bars before ADX is over 20 and rising (+DI still
    over -DI), the first bar where all hold. No stop / target in the video - ours: stop beyond the 10-bar swing, target 2R.
    Differs from TV24 (ADX > 25, ATR trail, no EMA50)."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.adx_di(df, 14)
    e50 = ema(c, 50)
    a = atr(h, l, c, 14)
    up_x = d["di_up"] | _since(d["di_up"], 3)
    dn_x = d["di_dn"] | _since(d["di_dn"], 3)
    L = _onset((c > e50) & up_x & (d["pdi"] > d["ndi"]) & (d["adx"] > 20) & d["adx_rising"])
    S = _onset((c < e50) & dn_x & (d["pdi"] < d["ndi"]) & (d["adx"] > 20) & d["adx_rising"])
    L, S = _excl(L, S)
    return L, S, _plan_r(_s(l).rolling(10).min().to_numpy() - 0.1 * a, _s(h).rolling(10).max().to_numpy() + 0.1 * a, 2.0)


# ---------------- SC206 ICT Judas Swing + order block re-entry (Flux; task tal1411set6 (13)) ----------------
def s_judas_ob(df):
    """ICT Judas Swing + Order Block (Flux Charts, TikTok, task tal1411set6 (13)): at the London / New York open the
    first move sweeps the Asian high / low (or the previous day's) and reverses; enter on the return to the order block
    that formed; stop beyond the sweep's extreme; target the other side of the range. Ours: agents/indicators
    judas_swing (Asian range 19:00-24:00 NY, windows 02-05 and 08:30-11 NY); buy when, within 12 bars after the
    bullish Judas signal, the low trades back into the order block (its top) and the bar closes over the OB's bottom;
    stop under the sweep low - 0.1 ATR; target the Asian high (mirror: Asian low). Fits bots 5-6 (opens)."""
    o, h, l, c, v = _ohlcv(df)
    j = IND.judas_swing(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    armL = armS = None
    for t in range(n):
        if j["bull"][t]:
            armL = (t, j["ob_top"][t], j["ob_bot"][t], j["sweep_lo"][t], j["asia_hi"][t])
        if j["bear"][t]:
            armS = (t, j["ob_top"][t], j["ob_bot"][t], j["sweep_hi"][t], j["asia_lo"][t])
        if armL and t > armL[0]:
            b, top, bot, sw, tgt = armL
            if t - b > 12 or c[t] < sw:
                armL = None
            elif l[t] <= top and c[t] > bot:
                L[t], sl_l[t], tg_l[t] = True, sw - 0.1 * a[t], tgt
                armL = None
        if armS and t > armS[0]:
            b, top, bot, sw, tgt = armS
            if t - b > 12 or c[t] > sw:
                armS = None
            elif h[t] >= bot and c[t] < top:
                S[t], sl_s[t], tg_s[t] = True, sw + 0.1 * a[t], tgt
                armS = None
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


# ---------------- SC207 - SC211 Volume-profile shapes (Dualis Flow / Olisa Flow / saint / nq tom; task tal1411vp) -----
def _vp_ctx(df, mode="volume"):
    o, h, l, c, v = _ohlcv(df)
    vp = IND.volume_profile(df, mode=mode)
    a = atr(h, l, c, 14)
    day = pd.Series(df.index.tz_convert("UTC").date).to_numpy()
    return o, h, l, c, vp, a, day


def _vp_d_fade(df, mode):
    o, h, l, c, vp, a, day = _vp_ctx(df, mode)
    D = vp["shape"] == 0
    L = D & (l <= vp["val"]) & (c > vp["val"])
    S = D & (h >= vp["vah"]) & (c < vp["vah"])
    L, S = _excl(_first_per_day(L, day), _first_per_day(S, day))
    return L, S, _plan_lv(vp["val"] - 0.5 * a, vp["vah"] + 0.5 * a, vp["poc"], vp["poc"])


def s_vp_d_edges_to_poc(df):
    """Dualis Flow / saint (TikTok, task tal1411vp): after a balanced session (D-shaped profile) trade edge to edge -
    buy at the lower edge of the value area (VAL), sell at the upper (VAH), target the POC; saint 'wait for the edge,
    do not trade the middle'. Ours: the previous UTC session's tick-volume profile (agents/indicators volume_profile -
    no real volume in CFDs, noted); a rejection at VAL (the low under it, the close back over it) ->
    buy; stop 0.5 ATR beyond the edge; target the POC; first per side a day."""
    return _vp_d_fade(df, "volume")


def s_tpo_d_edges_to_poc(df):
    """SC207 on the TPO (time-at-price) profile instead of volume (Meks Trades 'TPO profile', task tal1411ind) -
    agents/indicators tpo_profile; everything else as SC207."""
    return _vp_d_fade(df, "tpo")


def s_vp_pb_poc_pullback(df):
    """Dualis Flow (TikTok, task tal1411vp (b)): a P-shaped session (volume in the upper part - short covering) tends
    to continue up, a b-shaped one (volume low - long liquidation) down; enter on the pullback to the POC in that
    direction. Ours (with saint's 'trend day - one side
    dominates, do not trade against it': a trend-up session counts as P, trend-down as b): previous session P -> the first bar that trades down to its POC and closes over it -> buy, stop
    under the VAL - 0.2 ATR, target 2R; b -> mirror. One a day."""
    o, h, l, c, vp, a, day = _vp_ctx(df)
    P, b = np.isin(vp["shape"], [1, 2]), np.isin(vp["shape"], [-1, -2])
    L = P & (l <= vp["poc"]) & (c > vp["poc"])
    S = b & (h >= vp["poc"]) & (c < vp["poc"])
    L, S = _excl(_first_per_day(L, day), _first_per_day(S, day))
    return L, S, _plan_r(vp["val"] - 0.2 * a, vp["vah"] + 0.2 * a, 2.0)


def s_vp_lvn_to_hvn(df):
    """Olisa Flow (TikTok, LVN; task tal1411vp / tal1411of (2)): price moves fast through low-volume nodes - a break
    into an LVN targets the next high-volume node. Ours: the close crosses up through an LVN of the previous session
    (agents/indicators volume_profile: the bar's low under the LVN, its close over it) -> buy, target the next HVN above
    (must be >= 1 ATR away), stop 1 ATR under the LVN; mirror down. Do not place targets / stops inside an LVN is the
    filter idea for the bots (in_lvn)."""
    o, h, l, c, vp, a, day = _vp_ctx(df)
    lb, la = vp["lvn_below"], vp["lvn_above"]
    ha, hb = vp["hvn_above"], vp["hvn_below"]
    L = np.isfinite(lb) & (l <= lb) & (c > lb + 0.1 * a) & np.isfinite(ha) & (ha - c >= a)
    S = np.isfinite(la) & (h >= la) & (c < la - 0.1 * a) & np.isfinite(hb) & (c - hb >= a)
    L, S = _excl(L, S)
    return L, S, _plan_lv(lb - a, la + a, ha, hb)


def s_vp_b_void_break(df):
    """saint / Dualis Flow B-shape (two distributions, task tal1411vp): after a double-distribution session, a break
    out of the low-volume zone between the two peaks (the 'void' - Delta Void Profile idea, BigBeluga) goes to the
    other distribution. Ours: previous session B (agents/indicators volume_profile / delta_void_profile); a close
    crossing over the void -> buy, target the HVN above, stop 1 ATR under the void; crossing under -> sell to the HVN
    below."""
    o, h, l, c, vp, a, day = _vp_ctx(df)
    dv = IND.delta_void_profile(df)
    B = vp["shape"] == 3
    vd = dv["void"]
    L = B & cross_up(c, vd) & np.isfinite(dv["hvn_above"])
    S = B & cross_down(c, vd) & np.isfinite(dv["hvn_below"])
    L, S = _excl(L, S)
    return L, S, _plan_lv(vd - a, vd + a, dv["hvn_above"], dv["hvn_below"])


# ---------------- SC212 Absorption reversal (Orderflow Sniper; task tal1411of (1)) ----------------
def s_of_absorption_reversal(df):
    """Orderflow Sniper (TikTok, task tal1411of (1)): absorption = abnormal tick volume (2x+ the average) on a candle
    with a small range / body at the end of a move - someone is absorbing; enter the opposite way after a confirmation
    candle; stop beyond the absorption candle. Ours: agents/indicators absorption (body <= 0.35 ATR, at a 10-bar
    extreme; no order book - tick volume only, noted); the next 3 bars: a close over the absorption bar's high -> buy
    (mirror: under its low -> sell); stop 0.1 ATR beyond the absorption bar; target 2R. Bot 7: the same in the minutes
    after a release."""
    o, h, l, c, v = _ohlcv(df)
    ab = IND.absorption(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(1, n):
        for k in (1, 2, 3):
            j = t - k
            if j < 0:
                break
            if ab["bull"][j] and c[t] > h[j] and not np.any(c[j + 1:t] > h[j]):
                L[t], sl_l[t] = True, l[j] - 0.1 * a[t]
            if ab["bear"][j] and c[t] < l[j] and not np.any(c[j + 1:t] < l[j]):
                S[t], sl_s[t] = True, h[j] + 0.1 * a[t]
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC213 Equal highs / lows sweep reversal (Orderflow Sniper; task tal1411of (3)) ----------------
def s_of_equal_level_sweep(df):
    """Orderflow Sniper (TikTok, task tal1411of (3)): equal highs / lows (2+ touches) attract the price, and are bait
    before a reversal - a sweep of the level and a close back inside = enter the other way. Ours: agents/indicators
    liquidity (equal = two confirmed 5-bar swings within 0.1 ATR); entry at the sweep bar's close back inside; stop 0.2
    ATR beyond the sweep extreme; target = the opposite equal level when one is live and >= 1R away, else 2R
    (the 'magnet' half of the video)."""
    o, h, l, c, v = _ohlcv(df)
    lq = IND.liquidity(df, k=5)
    a = atr(h, l, c, 14)
    sl_l, sl_s = l - 0.2 * a, h + 0.2 * a
    eqh, eql = _sh(lq["eqh"]), _sh(lq["eql"])
    tgl = np.where(np.isfinite(eqh) & (eqh - c >= c - sl_l), eqh, c + 2 * (c - sl_l))
    tgs = np.where(np.isfinite(eql) & (c - eql >= sl_s - c), eql, c - 2 * (sl_s - c))
    L, S = _excl(lq["sweep_eq_bull"], lq["sweep_eq_bear"])
    return L, S, _plan_lv(sl_l, sl_s, tgl, tgs)


# ---------------- SC214 CVD divergence + reversal candle (Orderflow Sniper; task tal1411of (4)) ----------------
def s_of_cvd_divergence(df):
    """Orderflow Sniper (TikTok, task tal1411of (4)): approximate cumulative delta (tick volume x close location, reset
    each UTC session - no real order flow, noted) diverging from the price is a reversal warning. Ours: a bullish
    CVD divergence at a confirmed 5-bar swing low (agents/indicators cvd), then within 10 bars a close over the previous
    bar's high -> buy; stop under the divergence low - 0.1 ATR; target 2R. Mirror short."""
    o, h, l, c, v = _ohlcv(df)
    cv = IND.cvd(df, k=5)
    a = atr(h, l, c, 14)
    sh, sl = _swings(h, l, 5)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    arm_l = arm_s = None
    for t in range(1, n):
        if cv["bull_div"][t]:
            arm_l = (t, sl[t])
        if cv["bear_div"][t]:
            arm_s = (t, sh[t])
        if arm_l and t - arm_l[0] <= 10 and c[t] > h[t - 1]:
            L[t], sl_l[t] = True, arm_l[1] - 0.1 * a[t]
            arm_l = None
        if arm_s and t - arm_s[0] <= 10 and c[t] < l[t - 1]:
            S[t], sl_s[t] = True, arm_s[1] + 0.1 * a[t]
            arm_s = None
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC215 Far structural target with an early break-even (Ezekiel Chew; task tal1411set5 (3)) ----------
def s_ec_far_structural_target(df):
    """Ezekiel Chew (TikTok, US500, 1:5.7, task tal1411set5 (3)): the idea is a far structural target (a big swing)
    instead of 1:1, with an early break-even. Ours on a plain trend pullback so the bots can compare the exit: up trend
    (EMA50 over EMA200, close over EMA50), the low touches EMA50 and the close is back over it -> buy; stop under the
    10-bar low; target = the previous WEEK's high (agents/indicators htf_levels) when it is >= 3R away, else 3R;
    break-even at 1R (the breakeven rule: at 1/3 of the way to a 3R target). Mirror short to the previous week's low."""
    o, h, l, c, v = _ohlcv(df)
    e50, e200 = ema(c, 50), ema(c, 200)
    hl = IND.htf_levels(df)
    a = atr(h, l, c, 14)
    sl_l = _s(l).rolling(10).min().to_numpy() - 0.1 * a
    sl_s = _s(h).rolling(10).max().to_numpy() + 0.1 * a
    L = (e50 > e200) & (l <= e50) & (c > e50)
    S = (e50 < e200) & (h >= e50) & (c < e50)
    rl, rs = c - sl_l, sl_s - c
    tgl = np.where(hl["pwh"] - c >= 3 * rl, hl["pwh"], c + 3 * rl)
    tgs = np.where(c - hl["pwl"] >= 3 * rs, hl["pwl"], c - 3 * rs)
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tgl, tgs, breakeven=(0.33, 0.0))


# ---------------- strategies on the new indicators (one or more per indicator, so every bot's agent tests them) ------
def _swing_stops(h, l, a, n=10, buf=0.1):
    return _s(l).rolling(n).min().to_numpy() - buf * a, _s(h).rolling(n).max().to_numpy() + buf * a


def s_ind_supertrend_ai(df):
    """SuperTrend AI (Clustering) [LuxAlgo] / Flux 'AI SuperTrend' (task tal1411ind): buy when the adaptive SuperTrend
    flips up, sell when it flips down (the original's signals); stop at the trailing line (0.25 ATR beyond it when it
    is closer); exit on the opposite flip; no fixed target. agents/indicators supertrend_ai (ATR 10, factors 1-5,
    k-means 'best' cluster)."""
    o, h, l, c, v = _ohlcv(df)
    st = IND.supertrend_ai(df)
    a = atr(h, l, c, 14)
    ln = st["line"]
    sl_l = np.where(c - ln >= 0.25 * a, ln, c - 0.25 * a)
    sl_s = np.where(ln - c >= 0.25 * a, ln, c + 0.25 * a)
    L, S = _excl(st["buy"], st["sell"])
    return L, S, {"stop": ("level", sl_l, sl_s), "target": None, "exit_long": st["sell"], "exit_short": st["buy"]}


def s_ind_lorentzian(df):
    """Machine Learning: Lorentzian Classification (jdehorty, task tal1411ind): a new +1 / -1 classification with the
    kernel estimate in the same direction (the original's default kernel filter) -> buy / sell; exit after 4 bars (the
    original's hold) or on the opposite signal; stop 1.5 ATR (ours - the original has none). agents/indicators
    lorentzian."""
    o, h, l, c, v = _ohlcv(df)
    lz = IND.lorentzian(df)
    L = lz["buy"] & lz["kernel_up"]
    S = lz["sell"] & ~lz["kernel_up"]
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": None, "max_bars": 4, "exit_long": lz["sell"], "exit_short": lz["buy"]}


def s_ind_trend_duration_exit(df):
    """Trend Duration Forecast [ChartPrime] / 'expected trend length' (Trading Data, task tal1411set2 (4)): enter on an
    HMA(50) trend turn, exit when the running trend is as old as the average of the last 10 trends of that side
    (agents/indicators trend_duration ratio >= 1) or on the next turn; stop beyond the 10-bar swing; no target."""
    o, h, l, c, v = _ohlcv(df)
    td = IND.trend_duration(df)
    a = atr(h, l, c, 14)
    d = td["dir"]
    pd_ = _sh(d)
    L = (d == 1) & (pd_ == -1)
    S = (d == -1) & (pd_ == 1)
    old = np.nan_to_num(td["ratio"]) >= 1.0
    sl_l, sl_s = _swing_stops(h, l, a)
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", sl_l, sl_s), "target": None, "exit_long": old | (d == -1),
                  "exit_short": old | (d == 1)}


def s_ind_zero_lag_reentry(df):
    """Zero Lag Trend Signals [AlgoAlpha] (task tal1411ind): in an established trend (+1 for 2+ bars), the
    'Bullish Entry' = close crosses back over the ZLEMA -> buy; 'Bearish Entry' mirror; stop at the opposite band
    side (ZLEMA - volatility for longs), target 2R, exit when the trend flips. agents/indicators zero_lag_trend."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.zero_lag_trend(df)
    L, S = _excl(z["entry_long"], z["entry_short"])
    return L, S, _plan_r(z["lower"], z["upper"], 2.0, exit_long=z["sell"], exit_short=z["buy"])


def s_ind_zero_lag_flip(df):
    """Zero Lag Trend Signals [AlgoAlpha]: the big trend-flip arrows (close beyond ZLEMA +/- volatility) -> enter,
    exit on the opposite flip; stop at the ZLEMA (ours), no target."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.zero_lag_trend(df)
    a = atr(h, l, c, 14)
    zl = z["zlema"]
    L, S = _excl(z["buy"], z["sell"])
    return L, S, {"stop": ("level", np.fmin(zl, c - 0.5 * a), np.fmax(zl, c + 0.5 * a)), "target": None,
                  "exit_long": z["sell"], "exit_short": z["buy"]}


def s_ind_atr_fib_pocket(df):
    """ATR Fibonacci Trend Envelopes [BigBeluga] (task tal1411ind): in the envelope's trend, a pullback into the golden
    pocket (between the 0.618 and 0.786 lines) that closes back out on the trend side (close over open) -> buy; stop at
    the trailing band (the trend line), target 2R; mirror short. agents/indicators atr_fib_envelopes."""
    o, h, l, c, v = _ohlcv(df)
    e = IND.atr_fib_envelopes(df)
    up, dn = e["trend"] > 0, e["trend"] < 0
    L = up & e["in_pocket"] & (c > o)
    S = dn & e["in_pocket"] & (c < o)
    L, S = _excl(L, S)
    return L, S, _plan_r(e["trail"], e["trail"], 2.0)


def s_ind_double_ribbon(df):
    """Double Ribbon [ChartPrime] (task tal1411ind): the trend score turns positive -> buy, negative -> sell; exit on the
    opposite turn; stop 2 ATR (ours), no target. agents/indicators double_ribbon (our reconstruction)."""
    rb = IND.double_ribbon(df)
    L, S = _excl(rb["buy"], rb["sell"])
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": rb["sell"], "exit_short": rb["buy"]}


def s_ind_golden_wave(df):
    """Fibonacci Golden Wave [Flux Charts] (task tal1411ind - 'Fibonacci Golden Wave' video of Flux): in an up leg of
    the 20-bar range (its high made after its low), a low inside the 0.5-0.618 golden zone with a close back over the
    zone -> buy; stop under the 20-bar low (the 1.0 level); target the 20-bar high; mirror short. agents/indicators
    golden_wave."""
    o, h, l, c, v = _ohlcv(df)
    g = IND.golden_wave(df)
    a = atr(h, l, c, 14)
    L, S = _excl(g["touch_long"], g["touch_short"])
    return L, S, _plan_lv(g["bottom"] - 0.1 * a, g["top"] + 0.1 * a, g["top"], g["bottom"])


def s_ind_cisd_after_sweep(df):
    """CISD (ICT, LuxAlgo 'Change in State of Delivery' / CISD Finder, task tal1411ind 10:33) with its liquidity-sweep
    method: a sweep of a swing low (agents/indicators liquidity, 5-bar swings) and then, within 10 bars, a bullish CISD
    (close over the open of the last bearish run's first candle) -> buy; stop under the sweep low; target 2R. Mirror
    short. Differs from TT03 (sweep + POI + IFVG + CISD chain) - only the sweep and the CISD."""
    o, h, l, c, v = _ohlcv(df)
    lq = IND.liquidity(df, k=5)
    cs = IND.cisd(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    arm_l = arm_s = None
    for t in range(n):
        if lq["sweep_bull"][t]:
            arm_l = (t, l[t])
        if lq["sweep_bear"][t]:
            arm_s = (t, h[t])
        if arm_l and t - arm_l[0] <= 10 and cs["bull"][t]:
            L[t], sl_l[t] = True, min(arm_l[1], l[t]) - 0.1 * a[t]
            arm_l = None
        if arm_s and t - arm_s[0] <= 10 and cs["bear"][t]:
            S[t], sl_s[t] = True, max(arm_s[1], h[t]) + 0.1 * a[t]
            arm_s = None
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


def s_ind_impulse_zone_retest(df):
    """Impulse Zones [Flux Charts] (task tal1411ind 09:30 - 'close to the user's gap method'): a retest of the latest
    live bullish impulse zone (the low enters it, the close is back over its top) -> buy; stop under the zone's bottom
    - 0.1 ATR; target 2R; mirror short. agents/indicators impulse_zones (range > 1.5 ATR, tick volume > 1.5x avg)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.impulse_zones(df)
    a = atr(h, l, c, 14)
    L, S = _excl(z["retest_long"], z["retest_short"])
    return L, S, _plan_r(z["bull_bot"] - 0.1 * a, z["bear_top"] + 0.1 * a, 2.0)


def s_ind_funk_rsi(df):
    """Funk RSI: Break the Rules [Almazov Trade] (task tal1411ind 10:33): RSI(14) 30/70 swing reversal - a confirmed
    1-bar swing low with RSI under 30 in its window -> buy at the confirmation close, target the swing bar's high;
    mirror short. Stop: ours (the page's stop lies on the wrong side) - beyond the whole RSI excursion + 0.25 ATR.
    agents/indicators funk_rsi."""
    f = IND.funk_rsi(df)
    L, S = _excl(f["long"], f["short"])
    return L, S, _plan_lv(f["stop_long"], f["stop_short"], f["tgt_long"], f["tgt_short"])


def s_ind_wavetrend_zone_cross(df):
    """WaveTrend [LazyBear] (task tal1411ind 09:30, uses 1+2): wt1 crosses over wt2 under the oversold level -53 -> buy;
    crosses under over +53 -> sell; stop beyond the 10-bar swing; target 2R; exit at the opposite zone cross. Differs
    from SC128 (WT cross + Kalman Hull on the same bar)."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.wavetrend(df)
    a = atr(h, l, c, 14)
    sl_l, sl_s = _swing_stops(h, l, a)
    L, S = _excl(w["buy"], w["sell"])
    return L, S, _plan_r(sl_l, sl_s, 2.0, exit_long=w["sell"], exit_short=w["buy"])


def s_ind_wavetrend_divergence(df):
    """WaveTrend [LazyBear] uses 3+4 (divergence with the trend): a bullish wt1 divergence at a confirmed 5-bar swing low
    while wt1 is under 0, then wt1 crosses over wt2 within 10 bars -> buy; mirror short. Stop beyond the 10-bar swing,
    target 2R."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.wavetrend(df)
    a = atr(h, l, c, 14)
    cu, cd = cross_up(w["wt1"], w["wt2"]), cross_down(w["wt1"], w["wt2"])
    L = _since(w["bull_div"], 10) & cu & (w["wt1"] < 0)
    S = _since(w["bear_div"], 10) & cd & (w["wt1"] > 0)
    sl_l, sl_s = _swing_stops(h, l, a)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


def s_ind_nwe_reversal(df):
    """Nadaraya-Watson Envelope [LuxAlgo], non-repainting endpoint version (our version of the paid 'Non Repaint'
    envelope with reversal dots, task tal1411set2 / tal1411ind): the close crosses under the lower envelope = buy dot,
    over the upper = sell dot -> enter at the close; target the middle line; stop 1.5 ATR (ours). agents/indicators
    nwe_envelope (h 8, window 500; ours x2.5 instead of the original x3 - the x3 dots are very rare intraday)."""
    nw = IND.nwe_envelope(df, mult=2.5)
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    L, S = _excl(nw["buy"], nw["sell"])
    return L, S, _plan_lv(c - 1.5 * a, c + 1.5 * a, nw["mid"], nw["mid"])


def s_ind_avwap_swing_pullback(df):
    """Anchored VWAP (task tal1411ind): in an up trend (close over EMA50) a pullback to the VWAP anchored at the last
    confirmed swing low (10-bar pivots; the line exists only from the confirmation bar - no look-ahead) that closes back
    over it -> buy; mirror short on the VWAP anchored at the last swing high. Stop 1 ATR beyond the AVWAP, target 2R.
    agents/indicators anchored_vwap_swing."""
    o, h, l, c, v = _ohlcv(df)
    av = IND.anchored_vwap_swing(df, k=10)
    e50 = ema(c, 50)
    a = atr(h, l, c, 14)
    al, ah = av["avwap_low"], av["avwap_high"]
    L = (c > e50) & (l <= al) & (c > al)
    S = (c < e50) & (h >= ah) & (c < ah)
    L, S = _excl(L, S)
    return L, S, _plan_r(al - a, ah + a, 2.0)


def s_ind_smc_bos_ob(df):
    """Smart Money Concepts [LuxAlgo] (the thaa.fx gold board, task tal1411ind 10:34): continuation - while the
    structure trend is up (last break = BOS / CHoCH up), the first retest of the live bullish order block (the low
    enters the OB, the close is back over its top) -> buy; stop under the OB's bottom - 0.1 ATR; target 2R; mirror
    short. agents/indicators smc (pivots 5)."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df, k=5)
    a = atr(h, l, c, 14)
    L = (sm["trend"] == 1) & sm["ob_retest_long"]
    S = (sm["trend"] == -1) & sm["ob_retest_short"]
    L, S = _excl(_onset(L), _onset(S))
    return L, S, _plan_r(sm["ob_bull_bot"] - 0.1 * a, sm["ob_bear_top"] + 0.1 * a, 2.0)


def s_ind_smc_choch_fvg(df):
    """Smart Money Concepts [LuxAlgo]: reversal - after a CHoCH up, the first return into the live bullish FVG (within
    20 bars; the low enters the gap, the close stays over its bottom) -> buy; stop under the last swing low; target 2R;
    mirror after a CHoCH down. agents/indicators smc."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df, k=5)
    a = atr(h, l, c, 14)
    ch_u = _since(sm["choch_up"], 20) | sm["choch_up"]
    ch_d = _since(sm["choch_dn"], 20) | sm["choch_dn"]
    ft, fb = sm["fvg_bull_top"], sm["fvg_bull_bot"]
    st, sb = sm["fvg_bear_top"], sm["fvg_bear_bot"]
    L = ch_u & (sm["trend"] == 1) & np.isfinite(ft) & (l <= ft) & (c > fb)
    S = ch_d & (sm["trend"] == -1) & np.isfinite(sb) & (h >= sb) & (c < st)
    L, S = _excl(_onset(L), _onset(S))
    return L, S, _plan_r(sm["swing_lo"] - 0.1 * a, sm["swing_hi"] + 0.1 * a, 2.0)


def s_ind_breaker_retest(df):
    """Breaker Blocks (Flux Charts / LuxAlgo SMC, task tal1411ind): an order block that failed (a close through it)
    flips role - a bearish OB broken up becomes a bullish breaker; its retest (the low enters it, the close is back
    over its top) -> buy; stop under the breaker's bottom - 0.1 ATR; target 2R; mirror short. agents/indicators smc."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df, k=5)
    a = atr(h, l, c, 14)
    L, S = _excl(_onset(sm["brk_retest_long"]), _onset(sm["brk_retest_short"]))
    return L, S, _plan_r(sm["brk_bull_bot"] - 0.1 * a, sm["brk_bear_top"] + 0.1 * a, 2.0)


def s_ind_sd_breakout(df):
    """Supply & Demand + Breakout (Flux Charts / Demand Supply Zone Matrix, task tal1411ind): a close through a live
    supply zone (zone broken) -> buy (the breakout half of the indicator); stop under the broken zone's bottom - 0.2
    ATR (ours: 1.5 ATR when no zone level), target 2R; a close through a demand zone -> sell. agents/indicators
    supply_demand (base + departure >= 1.5 ATR). The first-touch half exists already as YT03."""
    o, h, l, c, v = _ohlcv(df)
    sd = IND.supply_demand(df)
    a = atr(h, l, c, 14)
    pst, psb = _sh(sd["sup_bot"]), _sh(sd["dem_top"])
    sl_l = np.where(np.isfinite(pst) & (pst < c), pst - 0.2 * a, c - 1.5 * a)
    sl_s = np.where(np.isfinite(psb) & (psb > c), psb + 0.2 * a, c + 1.5 * a)
    L, S = _excl(sd["break_up"], sd["break_dn"])
    return L, S, _plan_r(sl_l, sl_s, 2.0)


def s_ind_trendline_break(df):
    """Automatic trend-line breakout (BigBeluga Pattern Detector / Market Structure 'trend-line retest', task
    tal1411ind 09:39): a close over a falling resistance line that had 3+ touches -> buy; stop under the last confirmed
    swing low (5-bar), target 2R; mirror short through a rising support line with 3+ touches. agents/indicators
    trendlines."""
    o, h, l, c, v = _ohlcv(df)
    tl = IND.trendlines(df, k=5)
    sh, sl = _swings(h, l, 5)
    a = atr(h, l, c, 14)
    L = tl["break_res"] & (np.nan_to_num(_sh(tl["res_touches"])) >= 3) & (np.nan_to_num(tl["res_slope"]) < 0)
    S = tl["break_sup"] & (np.nan_to_num(_sh(tl["sup_touches"])) >= 3) & (np.nan_to_num(tl["sup_slope"]) > 0)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl - 0.1 * a, sh + 0.1 * a, 2.0)


def s_ind_ema9_20_bbw(df):
    """The user's two filters of 14.11 (task tal1411ema) as one strategy so the bots see them work together: EMA9
    crosses over EMA20 (buys only while 9 is over 20) and the Bollinger 20/2 width is NOT under its 20th percentile of
    the last 100 bars (agents/indicators bb_width_pct) -> buy; mirror; stop beyond the 10-bar swing; target 2R; exit on
    the cross back."""
    o, h, l, c, v = _ohlcv(df)
    e9, e20 = ema(c, 9), ema(c, 20)
    bw = IND.bb_width_pct(df)["width_pct"]
    a = atr(h, l, c, 14)
    ok = np.nan_to_num(bw, nan=0.0) >= 0.2
    sl_l, sl_s = _swing_stops(h, l, a)
    L, S = _excl(cross_up(e9, e20) & ok, cross_down(e9, e20) & ok)
    return L, S, _plan_r(sl_l, sl_s, 2.0, exit_long=cross_down(e9, e20), exit_short=cross_up(e9, e20))


# ---------------- strategies from tonight's web search (rules from the sources named in each docstring) ----------------
def s_qqe_cross(df):
    """QQE signals (StockSharp example https://doc.stocksharp.com/en/api-examples/0440_Qqe_Signals , M30): RSI 14
    smoothed by EMA5, QQE factor 4.238; buy when the smoothed RSI crosses over the QQE trailing line, exit when it
    crosses back. The source is long-only and has no stop - ours: the mirror short, stop 2 ATR. agents/indicators qqe."""
    q = IND.qqe(df)
    L, S = _excl(q["buy"], q["sell"])
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": q["sell"], "exit_short": q["buy"]}


def s_laguerre_rsi(df):
    """Laguerre RSI reversion (StockSharp https://doc.stocksharp.com/en/api-examples/0024_RSI_Laguerre , M5): gamma
    0.7; buy when it crosses up through 30, sell when it crosses down through 70; exit at 50 (or the opposite
    signal); stop 2% from the fill (a percentage - kept). agents/indicators laguerre_rsi."""
    o, h, l, c, v = _ohlcv(df)
    lr = IND.laguerre_rsi(df)
    x = lr["lrsi"]
    L, S = _excl(lr["buy"], lr["sell"])
    return L, S, {"stop": ("level", c * 0.98, c * 1.02), "target": None, "exit_long": x >= 50, "exit_short": x <= 50}


def s_fisher_cross(df):
    """Fisher Transform crossover (FMZ https://www.fmz.com/lang/en/strategy/503845 ): Fisher(9) crosses over its
    trigger (the previous value) while under 1 -> buy, crosses under while over 1 -> sell (as published... the short
    side's 'over 1' is the source's rule), exit on the opposite cross. No stop in the source - ours: 2 ATR.
    agents/indicators fisher_transform."""
    f = IND.fisher_transform(df)
    L, S = _excl(f["buy"], f["sell"])
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": cross_down(f["fisher"], f["trigger"]),
                  "exit_short": cross_up(f["fisher"], f["trigger"])}


def s_ssl_hma_wavetrend(df):
    """SSL Hybrid + WaveTrend (FMZ https://www.fmz.com/strategy/442097 - some parameters behind a login): long when the
    HMA baseline is bullish (close over HMA60 - ours), the SSL channel (10) flips up and WaveTrend crosses up (within
    the last 3 bars), and the candle is not oversized (range <= 2 ATR); short mirror. Stop = 1.5 ATR (the source: k x
    ATR, k ours), target 1R (the source's default 1:1). agents/indicators ssl_channel, wavetrend."""
    o, h, l, c, v = _ohlcv(df)
    ss = IND.ssl_channel(df, 10)
    w = IND.wavetrend(df)
    hm = _hma(c, 60)
    a = atr(h, l, c, 14)
    wu = cross_up(w["wt1"], w["wt2"])
    wd = cross_down(w["wt1"], w["wt2"])
    small = (h - l) <= 2 * a
    L = (c > hm) & ss["buy"] & (wu | _since(wu, 3)) & small
    S = (c < hm) & ss["sell"] & (wd | _since(wd, 3)) & small
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 1.0)}


def _harmonic(df, ab, bc, cd, d_xa, k=2, tol=0.05, stop_x=True, target="C"):
    """Generic XABCD harmonic on the zigzag of _zigzag_snapshots: AB/XA within ab, BC/AB within bc, CD/BC within cd,
    D = A + s x d_xa x XA (s = +1 for a high X); entry at the first bar reaching the D zone; stop beyond X (+5% XA);
    target C or A."""
    h, l, c = (df[x].to_numpy(dtype=float) for x in "hlc")
    n = len(c)
    piv, snaps = _zigzag_snapshots(h, l, k)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    st_l, st_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    for t in range(1, n):
        m = snaps[t]
        for end in (m, m - 1):
            if end < 4:
                continue
            (xi, X, xt), (ai, A, _), (bi, B, _), (ci, C, ct) = piv[end - 4:end]
            if ct != -xt:
                continue
            s = xt
            XA, AB, BC = s * (X - A), s * (B - A), s * (B - C)
            if XA <= 0 or AB <= 0 or BC <= 0:
                continue
            if not (ab[0] * (1 - tol) <= AB / XA <= ab[1] * (1 + tol) and bc[0] * (1 - tol) <= BC / AB <= bc[1] * (1 + tol)):
                continue
            D = A + s * d_xa * XA
            CD = s * (D - C)
            if CD <= 0 or not (cd[0] * (1 - tol) <= CD / BC <= cd[1] * (1 + tol)):
                continue
            Dt = D - s * tol * XA
            seg = h[ci + 1:t] if s > 0 else l[ci + 1:t]
            tgt = C if target == "C" else A
            if s > 0:
                if h[t] < Dt or (len(seg) and seg.max() >= Dt):
                    continue
                stop = X + 0.05 * XA if stop_x else D + 0.1 * XA
                if c[t] < stop and tgt < c[t]:
                    S[t], st_s[t], tg_s[t] = True, stop, tgt
            else:
                if l[t] > Dt or (len(seg) and seg.min() <= Dt):
                    continue
                stop = X - 0.05 * XA if stop_x else D - 0.1 * XA
                if c[t] > stop and tgt > c[t]:
                    L[t], st_l[t], tg_l[t] = True, stop, tgt
            break
    L, S = _excl(L, S)
    return L, S, _plan_lv(st_l, st_s, tg_l, tg_s)


def s_gartley_harmonic(df):
    """Gartley harmonic (The Right Trader https://therighttrader.com/learn/harmonic-patterns/gartley-pattern ), any
    timeframe: AB = 0.618 of XA, BC = 0.382-0.886 of AB, D = the 0.786 retracement of XA (CD = 1.13-1.618 of BC);
    enter at D; stop beyond X; target 1 = C (this one), target 2 = A (SC242). Bullish and bearish mirrors. Ours:
    2-bar zigzag swings, +/-5% tolerance, entry at the close of the first bar reaching the D zone."""
    return _harmonic(df, (0.618, 0.618), (0.382, 0.886), (1.13, 1.618), 0.786, target="C")


def s_gartley_target_a(df):
    """SC241 Gartley with the source's second target, the A level."""
    return _harmonic(df, (0.618, 0.618), (0.382, 0.886), (1.13, 1.618), 0.786, target="A")


def s_ross_gimmee_bar(df):
    """Joe Ross 'Gimmee bar' (LuxAlgo classic bar setups https://www.luxalgo.com/library/concept/classic-bar-setups/ ),
    sideways markets only: Bollinger 20/2; sell setup = the bar's high at / over the upper band and the close under its
    open -> sell stop 1 tick under that bar's low (next 2 bars); buy mirror at the lower band. Ours: sideways =
    ADX(14) < 20 (the source suggests an ADX filter); stop the setup bar's other extreme; target the middle band
    (no target in the source)."""
    o, h, l, c, v = _ohlcv(df)
    lo, mid, up = _bb(c, 20, 2.0)
    ad = adx(h, l, c, 14)
    n = len(c)
    setup_s = (h >= up) & (c < o) & (ad < 20)
    setup_l = (l <= lo) & (c > o) & (ad < 20)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(1, n):
        for k in (1, 2):
            j = t - k
            if j < 0:
                break
            if setup_s[j] and c[t] < l[j] and not np.any(c[j + 1:t] < l[j]):
                S[t], sl_s[t] = True, h[j]
            if setup_l[j] and c[t] > h[j] and not np.any(c[j + 1:t] > h[j]):
                L[t], sl_l[t] = True, l[j]
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, mid, mid)


def s_momentum_pinball(df):
    """Momentum Pinball (Raschke & Connors; WH SelfInvest
    https://www.whselfinvest.de/en-de/trading-platform/free-trading-strategies/tradingsystem/48-momentum-pinball-trading-indicator-raschke-linda-connors-larry ),
    daily signal, intraday entry: RSI(3) of the 1-day change closes under 30 -> the next day buy stop over the first
    hour's high (stop = the first hour's low); over 70 -> sell stop under the first hour's low. Exit: a loser is closed
    before the close; a winner is held overnight and closed the next morning. Ours on intraday bars (15m / 1h): UTC days,
    the first hour = 00:00-01:00 UTC; entry = the first close beyond that hour's range; time exit after 24 hours
    (= 'the next morning')."""
    o, h, l, c, v = _ohlcv(df)
    idx = df.index.tz_convert("UTC")
    day = pd.Series(idx.date).to_numpy()
    dc = pd.Series(c).groupby(day).last()
    ind = pd.Series(rsi(dc.diff().fillna(0).to_numpy(), 3), index=dc.index).shift(1)
    rv = pd.Series(day).map(ind).to_numpy(dtype=float)
    hr = idx.hour.to_numpy()
    fh = hr < 1
    fhh = pd.Series(np.where(fh, h, np.nan)).groupby(day).transform("max").to_numpy()
    fhl = pd.Series(np.where(fh, l, np.nan)).groupby(day).transform("min").to_numpy()
    after = ~fh
    L = after & (rv < 30) & (c > fhh)
    S = after & (rv > 70) & (c < fhl)
    L, S = _excl(_first_per_day(L, day), _first_per_day(S, day))
    nb = max(1, int(round(24 * 60 / max(_bar_minutes(df), 1.0))))
    return L, S, {"stop": ("level", fhl, fhh), "target": None, "max_bars": nb}


def s_elder_triple_screen(df):
    """Elder Triple Screen (forexop https://forexop.com/strategy/simple-triple-screen/ ), e.g. Daily / H4 / H1: screen 1
    = the trend from the MACD-histogram slope of a 4x higher timeframe (rising = long only); screen 2 = Force Index
    EMA(2) under 0 on this timeframe (a pullback); screen 3 = buy stop over the previous bar's high (Elder's rule) ->
    the first close over it; stop under the low of the signal / previous bar; target 2R (no fixed target in the
    source - ours). Short = mirror."""
    o, h, l, c, v = _ohlcv(df)
    bm = _bar_minutes(df)
    rule = f"{int(bm * 4)}min"
    mh = _htf(df, rule, lambda g: IND.macd_hist(g)["hist"])
    mh_prev = _htf(df, rule, lambda g: _sh(IND.macd_hist(g)["hist"]))
    up, dn = mh > mh_prev, mh < mh_prev
    fi = ema(np.diff(c, prepend=c[0]) * np.where(v > 0, v, 1.0), 2)
    L = up & (_sh(fi) < 0) & (c > _sh(h))
    S = dn & (_sh(fi) > 0) & (c < _sh(l))
    L, S = _excl(L, S)
    return L, S, _plan_r(np.fmin(l, _sh(l)), np.fmax(h, _sh(h)), 2.0)


def s_raschke_anti_310(df):
    """Raschke 'Anti', 3/10 oscillator version (MQL5 article https://www.mql5.com/en/articles/24280 , H4): fast = SMA3 -
    SMA10, slow = SMA16(fast); the slow line over 0 opens the long window; the fast line pulls back to / under the slow
    line and then crosses back over it (the first cross) with a close over the previous close -> buy at the next open;
    stop the lowest low of the pullback; exit when the fast line crosses back under. Short = mirror. Differs from FR11
    (the stochastic hook)."""
    o, h, l, c, v = _ohlcv(df)
    ro = IND.raschke_osc(df)
    f, s = ro["fast"], ro["slow"]
    n = len(c)
    cu, cd = cross_up(f, s), cross_down(f, s)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    pb_lo = pb_hi = None
    for t in range(1, n):
        if not np.isfinite(s[t]):
            continue
        if s[t] > 0:
            if cd[t]:
                pb_lo = l[t]
            elif pb_lo is not None and f[t] <= s[t]:
                pb_lo = min(pb_lo, l[t])
            if pb_lo is not None and cu[t] and c[t] > c[t - 1]:
                L[t], sl_l[t] = True, min(pb_lo, l[t])
                pb_lo = None
        else:
            pb_lo = None
        if s[t] < 0:
            if cu[t]:
                pb_hi = h[t]
            elif pb_hi is not None and f[t] >= s[t]:
                pb_hi = max(pb_hi, h[t])
            if pb_hi is not None and cd[t] and c[t] < c[t - 1]:
                S[t], sl_s[t] = True, max(pb_hi, h[t])
                pb_hi = None
        else:
            pb_hi = None
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", sl_l, sl_s), "target": None, "exit_long": cd, "exit_short": cu}


def s_lw_smash_day(df):
    """Larry Williams Smash Day (Rogue Quant https://roguequant.substack.com/p/i-backtested-larry-williams-trading ),
    daily: a bar that closes under the previous low AND is the lowest close of the last 8 bars -> buy stop over that
    bar's high on the next bar (the first close over it within 1 bar); sell = mirror. The source tests it as stop-and-
    reverse; Williams' own exits: a stop / the first profitable open - ours: stop at the smash bar's low (high), exit
    after 3 bars, no target. Differs from QB13 (the HIDDEN smash day)."""
    o, h, l, c, v = _ohlcv(df)
    ll8 = _s(c).rolling(8).min().to_numpy()
    hh8 = _s(c).rolling(8).max().to_numpy()
    sb = (c < _sh(l)) & (c <= ll8)
    ss = (c > _sh(h)) & (c >= hh8)
    L = _shb(sb) & (c > _sh(h))
    S = _shb(ss) & (c < _sh(l))
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", _sh(l), _sh(h)), "target": None, "max_bars": 3}


def s_weinstein_stage2(df):
    """Stan Weinstein stage 1 -> 2 transition (MQL5 https://www.mql5.com/en/articles/22746 ), daily: SMA150 for the
    30-week line; slope = (SMA - SMA 10 bars ago) / 10 / close, rising > +0.0001; stage 2 = rising SMA, price over it
    and the last 10 highs over the 10 before; long when stage 2 begins on tick volume > 1.8x its 20-bar average (a feed
    without volume passes) with RSI(14) >= 50; short = the stage 3 -> 4 mirror. Stop 2 ATR(14), target 3x the stop;
    early exit on a close back across the SMA."""
    o, h, l, c, v = _ohlcv(df)
    m = sma(c, 150)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (m - _sh(m, 10)) / 10 / c
    hh = _s(h).rolling(10).max().to_numpy()
    st2 = (slope > 1e-4) & (c > m) & (hh > _sh(hh, 10))
    st4 = (slope < -1e-4) & (c < m) & (_s(l).rolling(10).min().to_numpy() < _sh(_s(l).rolling(10).min().to_numpy(), 10))
    vok = _vol_ok(v, 20, 1.8)
    r = rsi(c, 14)
    L = _onset(st2) & vok & (r >= 50)
    S = _onset(st4) & vok & (r <= 50)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("atr", 6.0), "exit_long": c < m, "exit_short": c > m}


def s_avg_hl_ibs(df):
    """Average High-Low Range + IBS reversal (TradingView https://in.tradingview.com/script/BTwOiV1h-Average-High-Low-Range-IBS-Reversal-Strategy ),
    daily, long only: the close under (10-bar highest high - 2.5 x the 20-bar average of high - low) on 2 bars in a row
    and IBS = (close - low) / (high - low) under 0.2 -> buy; exit on a close over the previous bar's high. The source's
    'upper' is not defined - ours: the 10-bar highest high; IBS on the true range (a gap bar counts); no stop in the source - ours: 2 ATR."""
    o, h, l, c, v = _ohlcv(df)
    upper = _s(h).rolling(10).max().to_numpy()
    rng = sma(h - l, 20)
    under = c < upper - 2.5 * rng
    pc = _sh(c)
    hi, lo = np.fmax(h, pc), np.fmin(l, pc)
    with np.errstate(invalid="ignore", divide="ignore"):
        ibs = np.where(hi > lo, (c - lo) / (hi - lo), 0.5)
    L = under & _shb(under) & (ibs < 0.2)
    return L, np.zeros(len(c), bool), {"stop": ("atr", 2.0), "target": None, "exit_long": c > _sh(h)}



# ======================= 15.11.2026 night run: SC250-SC274 =======================
# tal1411gapplus (FVG quality, Silver Bullet, FVG + OB + Fibonacci, SMC + Fibonacci, Judas 09:30, CHoCH fake breakout),
# tal1411fade (EV06 stair-step retests), tal1411strad (ATR straddles), and the night's search (MQL5, StockSharp,
# WH SelfInvest, Ehlers, The Right Trader, TradeIQ, TTM Squeeze Pro, Kacher / Morales).

def _fvg_return(df, ok_bull, ok_bear, max_age=20, buf=0.1):
    """Entry on the return into the latest qualifying gap: the bar trades into the gap and closes back out of it in
    the gap's direction (edgeful's conservative entry: 'a close back beyond the gap'). The gap dies when a close goes
    through its far side or after max_age bars; one trade per gap. Stop buf x ATR beyond the gap."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.fvg_quality(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    gb = gs = None
    for i in range(n):
        if gb is not None:
            if c[i] < gb[1] or i - gb[2] > max_age:
                gb = None
            elif i > gb[2] and l[i] <= gb[0] and c[i] > gb[0] and c[i] > o[i]:
                L[i], sl_l[i] = True, gb[1] - buf * a[i]
                gb = None
        if gs is not None:
            if c[i] > gs[0] or i - gs[2] > max_age:
                gs = None
            elif i > gs[2] and h[i] >= gs[1] and c[i] < gs[1] and c[i] < o[i]:
                S[i], sl_s[i] = True, gs[0] + buf * a[i]
                gs = None
        if q["bull"][i] and ok_bull[i]:
            gb = (q["top"][i], q["bot"][i], i)
        if q["bear"][i] and ok_bear[i]:
            gs = (q["top"][i], q["bot"][i], i)
    return L, S & ~L, sl_l, sl_s


# ---------------- SC250 FVG with a weak third candle (thecrypteller, TikTok; task tal1411gapplus (4)) ----------------
def s_fvg_c3_weak(df):
    """thecrypteller (TikTok, sent by the user 14.11): 'most traders lose with FVGs because they trade the wrong ones' -
    judge the gap by its third candle: strong (big body) = price probably won't come back, big wick = don't trust it,
    weak (small body, small wicks) = a good gap. Only gaps with a weak candle 3 are traded: entry when price comes back
    into the gap and closes back out of it in the gap's direction, stop 0.1 ATR beyond the gap, 2R. The video gives no
    numbers - ours: weak = body < 0.5 ATR(14) and total wick < 0.5 ATR (agents/indicators fvg_quality); the entry /
    stop / 2R are the library's standard (TV10). Also a direct filter for the user's gap method (bots 5-6)."""
    q = IND.fvg_quality(df)
    L, S, sl, ss = _fvg_return(df, q["c3_weak"], q["c3_weak"])
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC251 FVG with significant displacement (SecondBell, TradingView; edgeful entry) ----------------
def s_fvg_displacement(df):
    """TradingView 'Fair Value Gaps - Significant Displacement' (SecondBell, open source): only gaps whose middle candle
    is a real displacement - relative volume >= 1.25x, body >= 0.70 ATR and >= 60% of its range, gap >= 0.10 ATR,
    candle in the gap's direction. The script gives no trade rules - entry / stop from edgeful's FVG guide (close back
    beyond the gap after a rejection, stop just beyond the gap); target 2R (ours; edgeful lists swing points / a
    measured move). Differs from TV10 (body > 1 ATR only, no volume, entry on a close above the gap's bottom)."""
    q = IND.fvg_quality(df)
    L, S, sl, ss = _fvg_return(df, q["displacement"], q["displacement"])
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC252 FVG with strong buying / selling volume inside (Flux / Volumatic idea; task (5)) ----------------
def s_fvg_buy_volume(df):
    """Flux Charts 'How to choose the correct FVG' and BigBeluga 'Volumatic Fair Value Gaps' (task tal1411gapplus (5)):
    trade a bullish gap only when buyers dominate the volume that made it, a bearish gap only when sellers do. Ours:
    buy share of the three candles' tick volume by close location (agents/indicators fvg_quality buy_share) >= 65%
    for a bullish gap, <= 35% for a bearish one (Flux's '75%' is per order block on real volume - our tick-volume
    split is softer). Entry back out of the gap, stop beyond it, 2R."""
    q = IND.fvg_quality(df)
    bs = np.nan_to_num(q["buy_share"], nan=0.5)
    L, S, sl, ss = _fvg_return(df, bs >= 0.65, bs <= 0.35)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC253 ICT Silver Bullet, three windows, target the opposing pool (LuxAlgo; task (1)) ----------------
def s_silver_bullet_3w(df):
    """ICT Silver Bullet as LuxAlgo defines it (task tal1411gapplus (1), Flux video): windows 03-04, 10-11 and 14-15
    New York; a sweep of the pool mapped before the window (ours: the high / low of the previous 3 hours), displacement the
    other way leaving an FVG inside the window, entry when price comes back into the gap and closes out of it, stop
    0.1 ATR beyond the sweep, target the opposing pool. Differs from TT39 (10-11 only, fixed 2R): three windows and the
    liquidity target. agents/indicators silver_bullet; on 15-minute bars the return may come up to 60 minutes after
    the window (ours)."""
    sb = IND.silver_bullet(df)
    return sb["long"], sb["short"] & ~sb["long"], _plan_lv(sb["stop_l"], sb["stop_s"], sb["tgt_l"], sb["tgt_s"])


def _leg_first(lo, hi, sig):
    """Only the first signal per leg (a leg = its (lo, hi) pair)."""
    key = np.nan_to_num(lo) * 1e6 + np.nan_to_num(hi)
    out = np.zeros(len(sig), bool)
    used = set()
    for i in np.where(sig)[0]:
        if key[i] not in used:
            used.add(key[i])
            out[i] = True
    return out


def _fib_ob_fvg(df, target):
    """TradersNotes (task (2)): the gap of the impulse must overlap the 0.5-0.618 zone AND the impulse's order block
    must reach into the 0.5-0.786 area (ours: 'overlaps the block' read as the block's top at or above the 0.786
    retracement); entry when price comes back into the overlap and closes up out of it."""
    o, h, l, c, v = _ohlcv(df)
    fl = IND.fib_legs(df)
    a = atr(h, l, c, 14)
    out = []
    for p, d in (("u_", 1), ("d_", -1)):
        ft, fb = fl[p + "fvg_top"], fl[p + "fvg_bot"]
        zt, zb = np.fmin(ft, fl[p + "gz_top"]), np.fmax(fb, fl[p + "gz_bot"])
        ok = np.isfinite(ft) & (fl[p + "age"] <= 60)
        if d > 0:
            obk = fl[p + "ob_top"] >= fl[p + "dz_bot"]
            sig = ok & obk & (l <= zt) & (c > zt) & (c > o) & (l > fl[p + "lo"])
            stop = np.fmin(fl[p + "ob_bot"], fl[p + "lo"]) - 0.1 * a
        else:
            obk = fl[p + "ob_bot"] <= fl[p + "dz_top"]
            sig = ok & obk & (h >= zb) & (c < zb) & (c < o) & (h < fl[p + "hi"])
            stop = np.fmax(fl[p + "ob_top"], fl[p + "hi"]) + 0.1 * a
        sig = _leg_first(fl[p + "lo"], fl[p + "hi"], np.nan_to_num(sig, nan=0).astype(bool))
        out.append((sig, stop, fl[p + target]))
    (L, sl, tl), (S, ss, ts) = out
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl, ss, tl, ts)


# ---------------- SC254 / SC255 FVG + order block + Fibonacci 0.5-0.618 (TradersNotes, 5m; task (2)) ----------------
def s_fib_ob_fvg_1272(df):
    """TradersNotes (TikTok, 5-minute; task tal1411gapplus (2)): enter in the gap only when it overlaps the order block
    before the impulse and the 0.5-0.618 zone of the impulse; stop under the block; targets the Fibonacci extensions
    1.272 / 1.414 / 2.0 / 2.618. This one: target 1.272 of the leg. Legs from confirmed pivots (k=3, agents/indicators
    fib_legs), first touch per leg, the leg at most 60 bars old (ours)."""
    return _fib_ob_fvg(df, "ext1272")


def s_fib_ob_fvg_2(df):
    """SC254 with the 2.0 extension as the target (TradersNotes' third target)."""
    return _fib_ob_fvg(df, "ext2")


def _smc_fib(df, target):
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df)
    fl = IND.fib_legs(df)
    a = atr(h, l, c, 14)
    bs_cu, bs_bu = _bars_since(sm["choch_up"]), _bars_since(sm["bos_up"])
    bs_cd, bs_bd = _bars_since(sm["choch_dn"]), _bars_since(sm["bos_dn"])
    seq_u = (bs_cu <= 80) & (bs_bu < bs_cu)            # CHoCH up, then a BOS up after it
    seq_d = (bs_cd <= 80) & (bs_bd < bs_cd)
    # the demand / supply zone = the live order block of that side; it must overlap the golden zone
    ovu = (sm["ob_bull_top"] >= fl["u_gz_bot"]) & (sm["ob_bull_bot"] <= fl["u_gz_top"])
    ovd = (sm["ob_bear_bot"] <= fl["d_gz_top"]) & (sm["ob_bear_top"] >= fl["d_gz_bot"])
    L = seq_u & ovu & (l <= fl["u_gz_top"]) & (c > fl["u_gz_bot"]) & (l > fl["u_lo"])
    S = seq_d & ovd & (h >= fl["d_gz_bot"]) & (c < fl["d_gz_top"]) & (h < fl["d_hi"])
    L = _leg_first(fl["u_lo"], fl["u_hi"], np.nan_to_num(L, nan=0).astype(bool))
    S = _leg_first(fl["d_lo"], fl["d_hi"], np.nan_to_num(S, nan=0).astype(bool))
    L, S = _excl(L, S)
    if target == "edge":
        tl, ts = fl["u_hi"], fl["d_lo"]
    else:
        tl, ts = fl["u_ext1272"], fl["d_ext1272"]
    return L, S, _plan_lv(fl["u_lo"] - 0.1 * a, fl["d_hi"] + 0.1 * a, tl, ts)


# ---------------- SC256 / SC257 SMC CHoCH -> BOS -> golden zone in demand (Mr Albash; task (3)) ----------------
def s_smc_fib_golden_edge(df):
    """Mr Albash (TikTok, 'SMC + Fibonacci'; task tal1411gapplus (3)): a CHoCH, then a BOS in the new direction; Fibonacci
    on that impulse; entry in the golden zone 0.5-0.618 where it overlaps a demand (supply) zone - ours: the live SMC
    order block of that side (agents/indicators smc, k=5) - a touch of the zone that closes inside or beyond it in the
    trade's direction (his limit entry in the zone; on bars: the close); stop beyond
    the leg's start; target the leg's end (the 'edge'). CHoCH within the last 80 bars (ours)."""
    return _smc_fib(df, "edge")


def s_smc_fib_golden_ext(df):
    """SC256 with the 1.272 extension of the impulse as the target (Mr Albash's second target)."""
    return _smc_fib(df, "ext")


def _judas_or(df, rs, re_, we, tz="America/New_York"):
    o, h, l, c, v = _ohlcv(df)
    j = IND.judas_open_range(df, tz=tz, range_start=rs, range_end=re_, win_end=we)
    a = atr(h, l, c, 14)
    sl, ss = j["sweep_lo"] - 0.1 * a, j["sweep_hi"] + 0.1 * a
    tl = np.fmax(j["rng_hi"], c + 1.5 * (c - sl))
    ts = np.fmin(j["rng_lo"], c - 1.5 * (ss - c))
    t = df.index.tz_convert(tz)
    mins = (t.hour * 60 + t.minute).to_numpy()
    L, S = _excl(j["bull"], j["bear"])
    return L, S, _plan_lv(sl, ss, tl, ts, force_exit=mins >= 15 * 60 + 45)


# ---------------- SC258 Judas swing of the NY 09:30-09:45 range + CISD (TradingFinder / the user; task (3) Judas) -------
def s_judas_ny0930(df):
    """ICT Judas swing of the New York opening range (the user's task tal1411gapplus: 'Judas Swing: range 9:30-9:45 New
    York'; TradingFinder 'ICT Setup 03 Judas Swing NY 9:30am + CHoCH/FVG'): range = 09:30-09:45 NY; until 11:00 a
    sweep of one side, then a CISD close (back beyond the open of the candles that made the sweep) -> enter the other
    way. Stop 0.1 ATR beyond the sweep; target the opposite edge of the range but at least 1.5R (ours - TradingFinder
    gives no target); flat 15:45 NY. One per day. agents/indicators judas_open_range."""
    return _judas_or(df, 570, 585, 660)


# ---------------- SC259 Judas swing of the forex 08:15-08:30 NY range + CISD (TradingFinder forex windows) --------------
def s_judas_fx0815(df):
    """TradingFinder's forex timing of the same model: range 08:15-08:30 New York, Judas move and CISD from 08:30
    (the source: 08:30-08:45; ours: confirmation allowed until 09:30 NY). Same stop / target / exit as SC258."""
    return _judas_or(df, 495, 510, 570)


# ---------------- SC260 CHoCH, fake breakout up into a bearish FVG at 0.618-0.786, extension target (TradersNotes 4H) ----
def s_choch_fakeout_fvg(df):
    """TradersNotes (TikTok, 4-hour; task tal1411gapplus (6)): after a bearish CHoCH, a fake breakout up (a sweep of the
    last minor high) into a bearish FVG that overlaps the 0.618-0.786 retracement of the down leg = sell; target a
    Fibonacci extension of the leg. Ours: CHoCH within 80 bars (agents/indicators smc), down leg from fib_legs (k=3),
    the live bearish FVG of smc overlapping 0.618-0.786, the sweep = the bar's high over the previous 3 bars' high and
    the close back under it, staying under the leg's start; stop 0.1 ATR over the bar's high;
    target the 1.272 extension. Mirror for longs."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df)
    fl = IND.fib_legs(df)
    a = atr(h, l, c, 14)
    ph5 = _s(h).rolling(3).max().shift(1).to_numpy()
    pl5 = _s(l).rolling(3).min().shift(1).to_numpy()
    ovd = (sm["fvg_bear_bot"] <= fl["d_dz_top"]) & (sm["fvg_bear_top"] >= fl["d_dz_bot"])
    ovu = (sm["fvg_bull_top"] >= fl["u_dz_bot"]) & (sm["fvg_bull_bot"] <= fl["u_dz_top"])
    S = ((_bars_since(sm["choch_dn"]) <= 80) & ovd & (h > ph5) & (c < ph5) & (h >= fl["d_dz_bot"])
         & (h < fl["d_hi"]))
    L = ((_bars_since(sm["choch_up"]) <= 80) & ovu & (l < pl5) & (c > pl5) & (l <= fl["u_dz_top"])
         & (l > fl["u_lo"]))
    L, S = _excl(np.nan_to_num(L, nan=0).astype(bool), np.nan_to_num(S, nan=0).astype(bool))
    return L, S, _plan_lv(l - 0.1 * a, h + 0.1 * a, fl["u_ext1272"], fl["d_ext1272"])


# ---------------- SC261 Release stair-step: re-entries on retests of broken levels (sphula, gold CPI; tal1411fade) -------
def s_release_stair_retests(df):
    """sphula (TikTok, gold CPI, 15m; the user's task tal1411fade, EV06 idea): during the release move he sold three times,
    each time on a retest of a level the move had broken (a staircase), with one common target. Ours: the release bar
    (_release_bars: the FTMO high-impact calendar with a symbol; synthetic data: 08:30 NY); direction = the first close
    within 3 bars at least 0.8 ATR from the pre-release close; a 'broken level' = the pre-release hour's high (low) once the
    move is through it, and every previous extreme of the move after the move paused 1+ bar and then made a new extreme; a retest = a bar that trades back to within 0.1 ATR of it
    and closes on the move's side with a candle in its direction -> entry. Stop 0.3 ATR beyond the level / the bar,
    common target = the first extreme + the initial move again (measured move), at most 3 entries in 3 hours (ours), each with
    its own stop (the bot's total risk per release <= 1R is the bot's rule, not modelled here)."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    a = atr(h, l, c, 14)
    bm = max(_bar_minutes(df), 1.0)
    W = max(int(round(180 / bm)), 6)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    for k in _release_bars(df):
        if k < 2 or k + 4 >= n:
            continue
        pre, a0 = c[k - 1], a[k - 1]
        d, t0 = 0, None
        for t in range(k, min(k + 3, n)):
            if abs(c[t] - pre) >= 0.8 * a0:
                d, t0 = (1 if c[t] > pre else -1), t
                break
        if d == 0:
            continue
        ext = (h[k:t0 + 1].max() if d > 0 else l[k:t0 + 1].min())
        tgt = ext + d * abs(ext - pre)
        H = max(int(round(60 / bm)), 1)
        lv0 = (h[max(0, k - H):k].max() if d > 0 else l[max(0, k - H):k].min())   # the pre-release hour's edge
        ext_t, levels, entries = t0, ([lv0] if d * (ext - lv0) > 0 else []), 0
        for t in range(t0 + 1, min(k + W, n)):
            if not levels and d * (ext - lv0) > 0 and lv0 not in levels and entries == 0:
                levels.append(lv0)
            if entries >= 3:
                break
            new = (h[t] > ext) if d > 0 else (l[t] < ext)
            if new:
                if t - ext_t >= 2:
                    levels.append(ext)
                ext, ext_t = (h[t] if d > 0 else l[t]), t
                continue
            for lv in list(levels):
                if d > 0 and l[t] <= lv + 0.1 * a0 and c[t] > lv and c[t] > o[t] and tgt > c[t]:
                    L[t], sl_l[t], tg_l[t] = True, min(lv, l[t]) - 0.3 * a0, tgt
                elif d < 0 and h[t] >= lv - 0.1 * a0 and c[t] < lv and c[t] < o[t] and tgt < c[t]:
                    S[t], sl_s[t], tg_s[t] = True, max(lv, h[t]) + 0.3 * a0, tgt
                else:
                    continue
                levels.remove(lv)
                entries += 1
                break
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s, max_bars=W)


# ---------------- SC262 Pre-release ATR straddle, OCO, opposite-order stop (ExpertFx, gold PPI; tal1411strad) ------------
def s_release_atr_straddle(df):
    """ExpertFx (TikTok, gold PPI; the user's task tal1411strad, bot 7 builds the full EV04): one minute before a red
    release a Buy Stop and a Sell Stop X from the price, OCO; ours from bars: X = 1.0 x ATR(14) of the bar before the
    release (the task: test 0.5 / 1 / 1.5), the side whose level a bar of the first 5 minutes reaches first (a bar that
    reaches both = skipped, unknown order), stop at the other order's level (the task's 'stop at the opposite order'),
    target 2R, out after 30 minutes. On bars the entry is the next bar's open after the touch - later and usually worse
    than the stop order (the task's extra news slippage is the bots' cost model). Differs from SC115 (fixed +/-20 pips,
    30 / 5 pips) and SC004 (4-hour range)."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    a = atr(h, l, c, 14)
    bm = max(_bar_minutes(df), 1.0)
    win = max(int(round(5 / bm)), 1)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for k in _release_bars(df):
        if k < 2:
            continue
        ref, x = c[k - 1], 1.0 * a[k - 1]
        up, dn = ref + x, ref - x
        for t in range(k, min(k + win, n)):
            hu, hd = h[t] >= up, l[t] <= dn
            if hu and hd:
                break
            if hu:
                L[t], sl_l[t] = True, dn
                break
            if hd:
                S[t], sl_s[t] = True, up
                break
    return L, S, _plan_r(sl_l, sl_s, 2.0, max_bars=max(int(round(30 / bm)), 1))


# ---------------- SC263 Session-open ATR straddle: London 08:00 and New York 09:30 (tal1411strad, 08:13) ----------------
def s_session_open_atr_straddle(df):
    """The user's extension of the ExpertFx straddle (task tal1411strad, 14.11 08:13: 'search the whole space, not only
    news' - London open, New York open): at 08:00 London and 09:30 New York a Buy Stop and a Sell Stop 0.5 ATR(14) from
    the last close before the open, OCO, valid 15 minutes; stop at the other order's level, 2R, out after 60 minutes.
    One per session. X (0.3-1.5 ATR) and the opens are the bots' sweep; 0.5 is ours."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    a = atr(h, l, c, 14)
    bm = max(_bar_minutes(df), 1.0)
    win = max(int(round(15 / bm)), 1)
    starts = []
    for tz, hh, mm in (("Europe/London", 8, 0), ("America/New_York", 9, 30)):
        t = df.index.tz_convert(tz)
        m = (t.hour == hh) & (t.minute == mm) & (t.weekday < 5)
        starts += list(np.where(m)[0])
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for k in sorted(starts):
        if k < 2:
            continue
        ref, x = c[k - 1], 0.5 * a[k - 1]
        up, dn = ref + x, ref - x
        for t in range(k, min(k + win, n)):
            hu, hd = h[t] >= up, l[t] <= dn
            if hu and hd:
                break
            if hu:
                L[t], sl_l[t] = True, dn
                break
            if hd:
                S[t], sl_s[t] = True, up
                break
    return L, S, _plan_r(sl_l, sl_s, 2.0, max_bars=max(int(round(60 / bm)), 1))


# ---------------- SC264 Tokyo-range breakout near the prior-day extreme, GBPUSD M15 (MQL5 article 24936) ----------------
def s_mql5_tokyo_range_pdh(df):
    """MQL5 article 'GBPUSD M15 Tokyo-range breakout EA, part 2' (https://www.mql5.com/en/articles/24936): range = 02:00-
    10:00 server time (ours: a UTC+2 server -> 00:00-08:00 UTC); within 48 bars after it a close beyond the edge by
    0.05 ATR(14) on a candle with body >= 80% of its range, and the edge no more than 50 points (5 pips) from the
    prior day's high (low) on that side; one per day. Asymmetric as published, R = the range width: up - stop the
    edge - 0.60R, target the breakout close + 0.85R; down - stop the edge + 1.50R, target - 0.30R; stop at least
    0.5 ATR away; out if a close comes back inside the edge by 0.25R, after 32 bars, or 21:45 server (19:45 UTC).
    Translation: 5 pips -> _pips_atr (15m: ~0.9 ATR)."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    rh, rl, hs, _, wd, day = _range_levels(df, "UTC", 0.0, 8.0)
    pdh, pdl, _ = _prev_day(df)
    R = rh - rl
    near = _pips_atr(df, 5) * a
    win = (hs >= 8.0) & (hs < 19.75) & (wd < 5)
    rng = np.where(h - l > 0, h - l, np.nan)
    body = np.abs(c - o) / rng >= 0.8
    upok = (pdh - rh <= near)
    dnok = (rl - pdl <= near)
    L0 = win & body & (c > rh + 0.05 * a) & upok
    S0 = win & body & (c < rl - 0.05 * a) & dnok
    first = _first_per_day(np.nan_to_num(L0 | S0, nan=0).astype(bool), day)
    L, S = _excl(L0 & first, S0 & first)
    sl_l = np.fmin(rh - 0.60 * R, c - 0.5 * a)
    sl_s = np.fmax(rl + 1.50 * R, c + 0.5 * a)
    return L, S, _plan_lv(sl_l, sl_s, c + 0.85 * R, c - 0.30 * R, max_bars=32, force_exit=hs >= 19.75,
                          exit_long=c < rh - 0.25 * R, exit_short=c > rl + 0.25 * R)


# ---------------- SC265 Gold London breakout of the 03-08 server range with range filter, 2.5R, BE 1.5R (MQL5 blog) -----
def s_mql5_gold_london_box(df):
    """MQL5 blog 'London Breakout on Gold: 3.7 Years of Backtests' (https://www.mql5.com/en/blogs/post/777540; XAUUSD,
    real ticks, 2023-2026, 2026 out-of-sample PF 1.61): range 03:00-08:00 server (ours: UTC+2 -> 01:00-06:00 UTC),
    from the range end a buy stop over the high / sell stop under the low (small buffer - ours 0.05 ATR), OCO, one a
    day; skip days whose range is < 0.30% or > 1.60% of the price; stop the other side of the range capped at 0.6% of
    the price; target 2.5R; break-even at 1.5R; all closed 21:00 server (19:00 UTC). The author's failed variants:
    mid-range stop, no range filter, holding overnight, 5R. Differs from TV07 / SC069 / SC103 (other hours, no filter).
    Translation (the percentages are gold's): gold's 15-minute ATR(14) is ~0.18% of the price, so 0.30% / 1.60% / 0.6%
    -> 1.7 / 9 / 3.3 ATR(14) of 15 minutes, scaled by sqrt(15 / bar minutes) on other timeframes."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    rh, rl, hs, _, wd, day = _range_levels(df, "UTC", 1.0, 6.0)
    k = np.sqrt(15.0 / max(_bar_minutes(df), 1.0))
    ra = (rh - rl) / (a * k)
    ok = (ra >= 1.7) & (ra <= 9.0) & (hs >= 6.0) & (hs < 19.0) & (wd < 5)
    buf = 0.05 * a
    L0 = ok & (h >= rh + buf) & ~(l <= rl - buf)
    S0 = ok & (l <= rl - buf) & ~(h >= rh + buf)
    first = _first_per_day(np.nan_to_num(L0 | S0, nan=0).astype(bool), day)
    L, S = _excl(L0 & first, S0 & first)
    cap = 3.3 * a * k
    sl_l = np.fmax(rl, rh + buf - cap)
    sl_s = np.fmin(rh, rl - buf + cap)
    return L, S, _plan_r(sl_l, sl_s, 2.5, breakeven=(0.6, 0.0), force_exit=hs >= 19.0)


# ---------------- SC266 Larry Williams Greatest Swing Value breakout (MQL5 article 21913) ----------------
def s_lw_greatest_swing(df):
    """Larry Williams 'Greatest Swing Value' (MQL5 article https://www.mql5.com/en/articles/21913): setup = close[1] <
    close[6] (bullish) / > (bearish); buy trigger = the bar's open + 1.8 x the average of the last 4 failure swings
    (high - open of bearish bars), sell = open - 1.8 x (open - low of bullish bars); entry when the close crosses the
    trigger inside the bar (the EA confirms on M1 closes - ours: the bar's own close beyond the trigger with the open
    on the other side); stop at the bar's own low (high) - 'today's extreme'; target 2R. agents/indicators williams_gsv."""
    o, h, l, c, v = _ohlcv(df)
    g = IND.williams_gsv(df)
    a = atr(h, l, c, 14)
    L = g["bull_setup"] & (c > g["buy_trig"]) & (o <= g["buy_trig"])
    S = g["bear_setup"] & (c < g["sell_trig"]) & (o >= g["sell_trig"])
    L, S = _excl(np.nan_to_num(L, nan=0).astype(bool), np.nan_to_num(S, nan=0).astype(bool))
    return L, S, _plan_r(l - 0.05 * a, h + 0.05 * a, 2.0)


# ---------------- SC267 Choppiness Index breakout over / under SMA20, out when choppy (StockSharp) ----------------
def s_ss_chop_breakout(df):
    """StockSharp 'Choppiness Index Breakout' (https://doc.stocksharp.com/en/api-examples/0045_Choppiness_Index_Breakout;
    5-minute): flat and CHOP(14) < 38.2 (trending) -> buy when the close is over SMA20, sell under it; close the position
    when CHOP rises over 61.8; stop 2% from the entry; no target. agents/indicators choppiness. Differs from SC098
    (CHOP only as a filter of a %R cross)."""
    o, h, l, c, v = _ohlcv(df)
    ch = IND.choppiness(df, 14)
    m = sma(c, 20)
    tr_ = np.nan_to_num(ch, nan=100) < 38.2
    L, S = _excl(tr_ & (c > m), tr_ & (c < m))
    (sl, ss), _ = _pct_levels(c, 2.0, 4.0)
    x = np.nan_to_num(ch) > 61.8
    return L, S, {"stop": ("level", sl, ss), "target": None, "exit_long": x, "exit_short": x}


# ---------------- SC268 Kaufman Efficiency Ratio zones +/-0.6, 10-bar extreme trailing stop (WH SelfInvest) ----------------
def s_whs_kaufman_er(df):
    """WH SelfInvest 'Trading strategy: Kaufman Efficiency Ratio' (https://whselfinvest.com/en/trading_strategies_33_Kaufman_Efficiency_Ratio.php;
    15-minute+ / daily): buy when the signed ER enters the upper zone (crosses over +0.6), sell when it crosses under
    -0.6; stop the 10-bar low (high), trailed to the 10-bar low as time passes (ours: exit on a close beyond the last
    10 bars' extreme); the page's profit target line has no stated level - none. ER lookback 10 (ours)."""
    o, h, l, c, v = _ohlcv(df)
    er = IND.efficiency_ratio(df, 10)
    L, S = _excl(cross_up(er, 0.6), cross_down(er, -0.6))
    lo10, hi10 = _s(l).rolling(10).min().to_numpy(), _s(h).rolling(10).max().to_numpy()
    return L, S, {"stop": ("level", lo10, hi10), "target": None,
                  "exit_long": c < _sh(lo10), "exit_short": c > _sh(hi10)}


# ---------------- SC269 MESA Adaptive MA: MAMA / FAMA cross (Ehlers / LazyBear, FMZ) ----------------
def s_mama_fama_cross(df):
    """Ehlers MESA Adaptive Moving Average strategy (LazyBear's Pine, FMZ https://fmz.com/strategy/428090; 30m crypto in
    the source): long when MAMA crosses above FAMA, short when it crosses under; always in (the opposite cross
    reverses); no stop in the source - ours 2 ATR(14). agents/indicators mama_fama (fast 0.5, slow 0.05, hl2)."""
    mf = IND.mama_fama(df)
    L, S = _excl(mf["up"], mf["dn"])
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


# ---------------- SC270 Ehlers Instantaneous Trendline trigger cross (blackcat1402, TradingView) ----------------
def s_ehlers_itrend_cross(df):
    """blackcat1402 'L2 Ehlers Instantaneous Trendline Trading Strategy' (TradingView, open source): Trigger (2 x ITrend -
    ITrend[2]) crosses above the Instantaneous Trendline (alpha 0.07, hl2) -> buy, under -> sell; guide stop 2 x ATR(14),
    guide target the reference x 1.015 (sell: / 1.015) - ours: 1.5% from the signal close; the opposite cross reverses
    (Ehlers' original reversal). agents/indicators ehlers_itrend."""
    o, h, l, c, v = _ohlcv(df)
    it = IND.ehlers_itrend(df)
    L, S = _excl(it["up"], it["dn"])
    return L, S, {"stop": ("atr", 2.0), "target": ("level", c * 1.015, c / 1.015), "opposite": True}


# ---------------- SC271 Chaikin Money Flow zero cross with the trend (The Right Trader, plan 1) ----------------
def s_trt_cmf_zero_trend(df):
    """The Right Trader 'Chaikin Money Flow' plan 1 (https://therighttrader.com/learn/indicators/chaikin-money-flow): only
    with the bigger trend up (ours: close over a rising EMA50), enter on the close that pushes CMF(20) above zero; stop
    under the swing low formed while CMF was negative (ours: the lowest low of that negative run, at most 30 bars);
    out on a close back under zero; target next resistance - ours 2R. Mirror short. agents/indicators cmf (tick volume)."""
    o, h, l, c, v = _ohlcv(df)
    cm = IND.cmf(df, 20)
    a = atr(h, l, c, 14)
    e = ema(c, 50)
    n = len(c)
    lo_run, hi_run = np.full(n, np.nan), np.full(n, np.nan)
    mn = mx = np.nan
    cnt_n = cnt_p = 0
    for i in range(n):
        if cm[i] < 0:
            mn = l[i] if cnt_n == 0 else min(mn, l[i])
            cnt_n += 1
        else:
            if i and cm[i - 1] < 0:
                lo_run[i] = mn if cnt_n <= 30 else _s(l[max(0, i - 30):i]).min()
            cnt_n = 0
        if cm[i] > 0:
            mx = h[i] if cnt_p == 0 else max(mx, h[i])
            cnt_p += 1
        else:
            if i and cm[i - 1] > 0:
                hi_run[i] = mx if cnt_p <= 30 else _s(h[max(0, i - 30):i]).max()
            cnt_p = 0
    up = (c > e) & (e > _sh(e, 5))
    dn = (c < e) & (e < _sh(e, 5))
    L = cross_up(cm, 0) & up
    S = cross_down(cm, 0) & dn
    L, S = _excl(L, S)
    return L, S, _plan_r(lo_run - 0.05 * a, hi_run + 0.05 * a, 2.0, exit_long=cm < 0, exit_short=cm > 0)


# ---------------- SC272 UT Bot with the volume filter, 2 ATR stop, reverse (TradeIQ) ----------------
def s_tiq_ut_bot_volume(df):
    """TradingView 'UT Bot Strategy with Risk Management [TradeIQ]' (4H crypto): UT Bot Alerts (key 3, ATR 10) buy / sell
    signals on the close, only when volume >= 1.2 x its 20-bar average (on by default); protective stop 2 ATR; the
    opposite signal reverses; targets optional (off). Differs from TV05 (UT Bot + ADX / RSI) and SC125 (UT trail + STC).
    agents/indicators ut_bot."""
    o, h, l, c, v = _ohlcv(df)
    u = IND.ut_bot(df, 3.0, 10)
    vo = _vol_ok(v, 20, 1.2)
    L, S = _excl(u["buy"] & vo, u["sell"] & vo)
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


# ---------------- SC273 TTM Squeeze Pro: fire after high compression with rising momentum (Carter) ----------------
def s_ttm_squeeze_pro_fire(df):
    """TTM Squeeze Pro (John Carter; LuxAlgo concept and the open-source 'TTM Squeeze Pro'): BB(20, 2) inside the Keltner
    channel 1.0 x ATR = high compression. Trade the fire (BB leaves the 1.5 channel) only when the squeeze reached high
    compression within the last 10 bars: long with momentum > 0 and rising, short < 0 and falling; out when the
    momentum histogram fades (turns against the trade); stop 2 ATR (ours). Differs from GH09 (any 1.5 squeeze, LazyBear
    1.5-SD band) and SC074 (2 / 2 channel, middle-band exit). agents/indicators squeeze_pro."""
    sp = IND.squeeze_pro(df)
    hc = _shb(_s((sp["level"] == 3).astype(float)).rolling(10, min_periods=1).max().to_numpy(), 1)
    mom = np.nan_to_num(sp["mom"])
    L = sp["fire"] & hc & (mom > 0) & sp["mom_up"]
    S = sp["fire"] & hc & (mom < 0) & ~sp["mom_up"]
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": ~sp["mom_up"], "exit_short": sp["mom_up"]}


# ---------------- SC274 Pocket Pivot near the 10-day line, out under it (Kacher & Morales / MarketSmith) ----------------
def s_pocket_pivot(df):
    """Pocket Pivot (Kacher & Morales 'Trade Like an O'Neil Disciple'; MarketSmith / LuxAlgo concept): an up bar whose
    volume beats every down bar's volume of the previous 10 bars, closing over the 50-bar line and near (not extended
    from) the 10-bar line -> buy; sell on a close under the 10-bar line. Their 7-8% maximum loss -> ours 2.5 ATR(14).
    Long only (stocks / indices / crypto). agents/indicators pocket_pivot."""
    o, h, l, c, v = _ohlcv(df)
    pp = IND.pocket_pivot(df)
    L = pp["pp"]
    return L, np.zeros(len(c), bool), {"stop": ("atr", 2.5), "target": None, "exit_long": c < sma(c, 10)}



# ======================= 14.11.2026 manual run: SC275-SC289 (task tal1411ind, the indicators still missing) =======================

# ---------------- SC275 AI Trend Navigator [K-Neighbor] turn with the classifier (Zeiierman) ----------------
def s_knn_trend_navigator(df):
    """AI Trend Navigator [K-Neighbor] (Zeiierman, open source; Flux / the user's 'AI indicators' list, task tal1411ind):
    buy when the KNN line turns up (green) and the classifier's prediction is positive, sell when it turns down with a
    negative prediction; the opposite turn reverses; no stop in the source - ours 2 ATR(14). agents/indicators
    knn_trend_navigator (K 3, window 30, SMA(hl2, 5) vs RMA(close, 5), smoothing 50)."""
    kn = IND.knn_trend_navigator(df)
    L, S = _excl(kn["buy"], kn["sell"])
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


# ---------------- SC276 Gaussian Channel breakout while the filter rises (DonovanWall) ----------------
def s_gaussian_channel_break(df):
    """Gaussian Channel [DW] (DonovanWall, open source) as its common strategy port trades it: the filter rising and the
    close crossing above the upper band -> long, out on a close back under the upper band; mirror short under the lower
    band while the filter falls (the port is long-only - the short side is ours); stop 2 ATR (ours). agents/indicators
    gaussian_channel (4 poles, period 144, x1.414)."""
    o, h, l, c, v = _ohlcv(df)
    g = IND.gaussian_channel(df)
    fall = g["filt"] < _sh(g["filt"])
    L = g["rising"] & cross_up(c, g["upper"])
    S = fall & cross_down(c, g["lower"])
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": c < g["upper"], "exit_short": c > g["lower"]}


# ---------------- SC277 Kalman SuperTrend flip ('Pro Scalper' reconstruction, ProTrippa) ----------------
def s_kalman_supertrend(df):
    """'Pro Scalper' (profitprotrading, invite-only; in ProTrippa's scalping list, task tal1411ind) - OUR reconstruction
    from its description: SuperTrend on a Kalman-smoothed price; buy / sell on the flips, skipping buys that start over
    the upper VWMA band (overbought) and sells under the lower band (oversold); stop at the SuperTrend line (at least
    0.25 ATR away), exit on the opposite flip. agents/indicators kalman_supertrend."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.kalman_supertrend(df)
    a = atr(h, l, c, 14)
    L = k["buy"] & ~(c > k["ob"])
    S = k["sell"] & ~(c < k["os"])
    L, S = _excl(L, S)
    ln = k["line"]
    return L, S, {"stop": ("level", np.fmin(ln, c - 0.25 * a), np.fmax(ln, c + 0.25 * a)), "target": None,
                  "opposite": True}


# ---------------- SC278 Trend Magnet flip outside low-momentum zones (BigBeluga reconstruction) ----------------
def s_trend_magnet(df):
    """BigBeluga 'Trend Magnet' (closed, Market Waves Pro; task tal1411ind) - OUR reconstruction: the band flips up / down
    (Scalp sensitivity - EMA20 of hlc3 +/- 1.5 ATR20 - on 1-15 minute bars, Normal - EMA50 / ATR50 - above); trade the flip only outside a low-momentum zone (ADX14 >= 20 - the
    toolkit's 'low momentum zones' filter); stop at the magnet line, exit on the opposite flip."""
    o, h, l, c, v = _ohlcv(df)
    tm = IND.trend_magnet(df, "scalp" if _bar_minutes(df) <= 15 else "normal")
    a = atr(h, l, c, 14)
    L, S = _excl(tm["buy"] & ~tm["low_mom"], tm["sell"] & ~tm["low_mom"])
    ln = tm["line"]
    return L, S, {"stop": ("level", np.fmin(ln, c - 0.25 * a), np.fmax(ln, c + 0.25 * a)), "target": None,
                  "opposite": True}


# ---------------- SC279 Volume Mapping zero cross with the EMA50 trend (BigBeluga reconstruction) ----------------
def s_volume_mapping_trend(df):
    """BigBeluga 'Volume Mapping' (closed, Nautilus; task tal1411ind: green = buying volume dominant) - OUR
    reconstruction: the buy-minus-sell tick-volume oscillator crosses above zero while the close is over a rising
    EMA50 -> long (mirror short); out when it crosses back; stop 1.5 ATR, 2R (ours). agents/indicators volume_mapping."""
    o, h, l, c, v = _ohlcv(df)
    vm = IND.volume_mapping(df)
    e = ema(c, 50)
    L = cross_up(vm, 0) & (c > e) & (e > _sh(e, 3))
    S = cross_down(vm, 0) & (c < e) & (e < _sh(e, 3))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": vm < 0, "exit_short": vm > 0}


# ---------------- SC280 / SC281 Sniper V.02: EMA 9/21 cross with the score, and the ribbon retest ----------------
def s_sniper_cross_score(df):
    """'Sniper Entry/Exit with SL&TP by KhanSaab V.02' (3 / 5 / 15 min; task tal1411ind 'Sniper V.02') - OUR
    reconstruction: the EMA9 / EMA21 cross (black candle) taken only with the dashboard's bull score > 60% (bear score for
    sells): close vs session VWAP, RSI14 50, MACD histogram, ADX14 > 20 with DI; stop 1.5 ATR, target 3 ATR (its ATR
    TP3 - multipliers not published, ours), trail exit on a close beyond EMA21."""
    o, h, l, c, v = _ohlcv(df)
    s = IND.sniper_ribbon(df)
    L, S = _excl(s["buy"] & (s["bull_score"] > 0.6), s["sell"] & (s["bear_score"] > 0.6))
    return L, S, {"stop": ("atr", 1.5), "target": ("atr", 3.0), "exit_long": c < s["e21"], "exit_short": c > s["e21"]}


def s_sniper_retest(df):
    """Sniper V.02's 'retest' (orange candle): in the EMA9 > EMA21 trend the low touches the ribbon and the candle closes
    up over EMA21 (mirror for sells), with the bull (bear) score > 60%; one per trend leg (the first retest after each
    cross); stop 1.5 ATR, target 3 ATR, trail exit beyond EMA21 (ours, as SC280)."""
    o, h, l, c, v = _ohlcv(df)
    s = IND.sniper_ribbon(df)
    bl, bs = _bars_since(s["buy"]), _bars_since(s["sell"])
    rl = s["retest_long"] & (s["bull_score"] > 0.6) & (bl < bs)
    rs = s["retest_short"] & (s["bear_score"] > 0.6) & (bs < bl)
    # first retest after each cross
    fl = rl & (_bars_since(_sh(rl.astype(float)) > 0) > bl)
    fs = rs & (_bars_since(_sh(rs.astype(float)) > 0) > bs)
    L, S = _excl(fl, fs)
    return L, S, {"stop": ("atr", 1.5), "target": ("atr", 3.0), "exit_long": c < s["e21"], "exit_short": c > s["e21"]}


# ---------------- SC282 VuManChu Cipher B green / red dot with the money flow turning (VuManChu) ----------------
def s_cipher_b_dot_mf(df):
    """VuManChu Cipher B (open source; task tal1411ind): the green dot (WaveTrend cross up under -53) only when the
    RSI+MFI money-flow area is turning up (higher than 3 bars ago) - or a WaveTrend bullish divergence within the last 10
    bars; red dot mirror. Stop beyond the 10-bar swing, 2R (ours - Cipher B has no exits). Differs from SC227 (the bare
    WaveTrend cross) by the money-flow / divergence condition. agents/indicators cipher_b."""
    o, h, l, c, v = _ohlcv(df)
    cb = IND.cipher_b(df)
    a = atr(h, l, c, 14)
    mf = cb["rsimfi"]
    bd, sd = _bars_since(cb["bull_div"]) <= 10, _bars_since(cb["bear_div"]) <= 10
    L = cb["buy"] & ((mf > _sh(mf, 3)) | bd)
    S = cb["sell"] & ((mf < _sh(mf, 3)) | sd)
    L, S = _excl(np.nan_to_num(L, nan=0).astype(bool), np.nan_to_num(S, nan=0).astype(bool))
    sl, ss = _swing_stops(h, l, a, 10, 0.1)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC283 VuManChu Swing Free (Range Filter 20 / 3.5) signals, band stop, reverse ----------------
def s_swing_free(df):
    """VuManChu Swing Free (= DonovanWall's Range Filter, swing period 20, multiplier 3.5; task tal1411ind): BUY / SELL on
    the filter's state changes; the opposite signal reverses; stop at the opposite range band (filter -/+ the smooth
    range, at least 0.5 ATR). Differs from SC079 (Range Filter 100 / 3 with RSI and ADX filters, 3R)."""
    o, h, l, c, v = _ohlcv(df)
    rf = IND.range_filter(df, 20, 3.5)
    a = atr(h, l, c, 14)
    L, S = _excl(rf["buy"], rf["sell"])
    return L, S, {"stop": ("level", np.fmin(rf["lband"], c - 0.5 * a), np.fmax(rf["hband"], c + 0.5 * a)),
                  "target": None, "opposite": True}


# ---------------- SC284 HalfTrend signal engine: flip, band stop, 3R, opposite exit (BigBeluga / Trading Data) ----------
def s_halftrend_engine(df):
    """'HalfTrend Long/Short Signal Engine [BigBeluga]' (open source; Trading Data 'This FREE Indicator Should Be Illegal',
    task tal1411ind): LONG on the HalfTrend bearish -> bullish flip, SHORT on the reverse; stop beyond the HalfTrend
    channel (line -/+ 2 x ATR(100)/2, at least 0.5 ATR); target 3R (the dashboard's 1:3; TP1 at 1:1 is the engine's
    'scratch' rule, not modelled); the opposite signal exits. agents/indicators halftrend."""
    o, h, l, c, v = _ohlcv(df)
    ht = IND.halftrend(df)
    a = atr(h, l, c, 14)
    L, S = _excl(ht["buy"], ht["sell"])
    return L, S, {"stop": ("level", np.fmin(ht["band_lo"], c - 0.5 * a), np.fmax(ht["band_hi"], c + 0.5 * a)),
                  "target": ("R", 3.0), "opposite": True}


# ---------------- SC285 MSNR: first touch of a fresh support in a bullish storyline, target the next fresh level -------
def s_msnr_fresh_touch(df):
    """Malaysian SnR (task tal1411ind 12:47; Keenbase / LuxAlgo rules): in a bullish storyline price reaches a fresh
    support (V-level / gap level / flipped resistance) and the candle does not close through it -> buy at the close;
    stop under the bar's low and the level - 0.1 ATR; target the nearest fresh resistance; skip unless the target is
    >= 1.5x the stop distance; out after 200 bars. Sells mirror. Ours: storyline = the close over / under EMA20 of the
    4x higher timeframe (the source reads it from D1 / H4 levels). agents/indicators msnr."""
    o, h, l, c, v = _ohlcv(df)
    m = IND.msnr(df)
    a = atr(h, l, c, 14)
    up = _htf_above_ema20(df)
    sl_l = np.fmin(l, m["lvl"]) - 0.1 * a
    sl_s = np.fmax(h, m["lvl"]) + 0.1 * a
    tl, ts = m["res"], m["sup"]
    okl = (tl - c) >= 1.5 * (c - sl_l)
    oks = (c - ts) >= 1.5 * (sl_s - c)
    L = m["touch_sup"] & (c > m["lvl"]) & up & okl
    S = m["touch_res"] & (c < m["lvl"]) & ~up & oks
    L, S = _excl(np.nan_to_num(L, nan=0).astype(bool), np.nan_to_num(S, nan=0).astype(bool))
    return L, S, _plan_lv(sl_l, sl_s, tl, ts, max_bars=200)


def _crt_rule(df):
    bm = _bar_minutes(df)
    return "4h" if bm < 240 else ("1D" if bm < 1440 else "7D")


def _crt(df, target):
    o, h, l, c, v = _ohlcv(df)
    k = IND.crt(df, _crt_rule(df))
    a = atr(h, l, c, 14)
    L, S = _excl(k["bull"], k["bear"])
    if target == "eq":
        tl = ts = k["eq"]
    else:
        tl, ts = k["c1_hi"], k["c1_lo"]
    return L, S, _plan_lv(k["c2_lo"] - 0.1 * a, k["c2_hi"] + 0.1 * a, tl, ts)


# ---------------- SC286 / SC287 Candle Range Theory on the 4-hour candle (LuxAlgo; JP225 video) ----------------
def s_crt_far_extreme(df):
    """Candle Range Theory (task tal1411ind 12:47, JP225 2-hour; LuxAlgo concept): higher-timeframe candle 2 sweeps candle
    1's low and closes back inside its range -> buy at candle 2's close (mirror: sweep of the high -> sell); stop 0.1 ATR
    beyond candle 2's extreme; target candle 1's other extreme (TP2). Higher timeframe: 4H for intraday bars (ours; the
    video's 2H is a sweep for the bots), 1D for 4H bars, a week for daily. Differs from TT19 / TT35 (one fixed session
    candle) - every higher-timeframe candle counts."""
    return _crt(df, "far")


def s_crt_eq(df):
    """SC286 with candle 1's 50% (EQ, LuxAlgo's TP1) as the target."""
    return _crt(df, "eq")


def _partner_hl(df):
    """The partner's high / low aligned to df (agents/intermarket_strategies partner and loader). Without a symbol
    (synthetic test data only) a stand-in partner = the 3-bar mean of the instrument's own high / low, so the code path
    runs; no real backtest uses it."""
    sym, tf = df.attrs.get("symbol"), df.attrs.get("tf")
    if not sym or not tf:
        return pd.DataFrame({"h": _s(df["h"].to_numpy(float)).rolling(3, min_periods=1).mean().to_numpy(),
                             "l": _s(df["l"].to_numpy(float)).rolling(3, min_periods=1).mean().to_numpy()}, index=df.index)
    from agents import intermarket_strategies as IM
    p = IM.partner_of(sym)
    start, end = df.index[0].strftime("%Y-%m-%d"), (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    key = ("hl", p, tf, start, end)
    if key not in IM._CACHE:
        try:
            IM._CACHE[key] = IM.PARTNER_BARS(p, tf, start, end)
        except Exception:  # noqa: BLE001 - no partner data: no signals
            IM._CACHE[key] = None
    pdf = IM._CACHE[key]
    if pdf is None or len(pdf) == 0:
        return None
    u = df.index.union(pdf.index)
    return pd.DataFrame({x: pdf[x].reindex(u).ffill(limit=3).reindex(df.index).to_numpy(float) for x in ("h", "l")},
                        index=df.index)


# ---------------- SC288 SMT divergence + structure shift, sweep stop, 2R (simple_forex; US100 vs US500) ----------------
def s_smt_divergence_mss(df):
    """SMT divergence (simple_forex video, task tal1411ind 12:52: US100 vs US500 - one makes a new high, the other does
    not; ICT usage): a bearish SMT at a confirmed swing high (pivots k=3) with the partner from
    agents/intermarket_strategies (US100 <-> US500, EURUSD <-> GBPUSD, gold <-> silver ...), then within 10 bars a close
    under the previous bar's low (the market-structure shift) -> sell; stop 0.1 ATR over the swing; target 2R (ours for
    'the opposing liquidity'). Bullish mirror. No partner data -> no signals. agents/indicators smt_divergence."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    part = _partner_hl(df)
    if part is None:
        z = np.zeros(n, bool)
        return z, z, {"stop": ("atr", 2.0), "target": None}
    sm = IND.smt_divergence(df, part, 3)
    a = atr(h, l, c, 14)
    bb, bs = _bars_since(sm["bull"]), _bars_since(sm["bear"])
    plo = _s(sm["piv_lo"]).ffill().to_numpy()
    phi = _s(sm["piv_hi"]).ffill().to_numpy()
    L = (bb <= 10) & (bb < bs) & (c > _sh(h)) & (c > plo)
    S = (bs <= 10) & (bs < bb) & (c < _sh(l)) & (c < phi)
    # once per divergence
    L = L & (_bars_since(_sh(L.astype(float)) > 0) > bb)
    S = S & (_bars_since(_sh(S.astype(float)) > 0) > bs)
    L, S = _excl(L, S)
    return L, S, _plan_r(plo - 0.1 * a, phi + 0.1 * a, 2.0)


# ---------------- SC289 Gaussian Channel as a filter for the user's gap entry: FVG return in the filter's direction -------
def s_fvg_gaussian_trend(df):
    """The indicator-as-filter test the user asks for (task tal1411ind: 'test each as a filter on the gap method'): the
    library's FVG return entry (SC250's mechanics, any gap) only in the Gaussian Channel's direction - bullish gaps while
    the filter rises and the close is over it, bearish mirror; gap stop, 2R."""
    o, h, l, c, v = _ohlcv(df)
    g = IND.gaussian_channel(df)
    up = g["rising"] & (c > g["filt"])
    dn = (g["filt"] < _sh(g["filt"])) & (c < g["filt"])
    L, S, sl, ss = _fvg_return(df, up, dn)
    return L, S, _plan_r(sl, ss, 2.0)


# =====================================================================================================================
# 15.11.2026 (night): task tal1411set7 (the user's TikTok videos 14.11 17:37-18:03) and tonight's web search.
# =====================================================================================================================
def _htf_prev_range(df, rule):
    """The previous COMPLETED higher-timeframe candle's high / low on every base bar (known from the current
    candle's first bar), and the bar's position in the current candle (key)."""
    o, h, l, c, v = _ohlcv(df)
    key = df.index.floor(rule)
    g = pd.DataFrame({"h": h, "l": l}, index=df.index).groupby(key)
    hh, ll = g["h"].max(), g["l"].min()
    ph, pl = hh.shift(1), ll.shift(1)
    return ph.reindex(key).to_numpy(float), pl.reindex(key).to_numpy(float), np.asarray(key.asi8)


def _run_ext(x, key, fn):
    """Running min / max of x inside each key group (bars <= i)."""
    s = pd.Series(np.asarray(x, float))
    return s.groupby(key).cummin().to_numpy() if fn == "min" else s.groupby(key).cummax().to_numpy()


def _once_since(mask, event):
    """mask, only the first True after each event."""
    m = np.asarray(mask, bool)
    be = _bars_since(event)
    out = np.zeros(len(m), bool)
    last_ev = -1
    taken = False
    for i in range(len(m)):
        if be[i] == 0:
            taken = False
        if m[i] and not taken and be[i] < 10**8:
            out[i] = True
            taken = True
    return out


# ---------------- SC290 CRT 4H live: sweep of the previous 4H candle, LTF MSS, FVG entry (thecrypteller; set7 (1), (14)) ----
def s_crt_live_mss_fvg(df):
    """thecrypteller (TikTok, task tal1411set7 (1); wyckoff.hq the same with a 5-minute entry, (14)): the previous 4H
    candle marks the range; the current 4H candle wicks beyond it (manipulation); on the 15 (5)-minute chart: the
    sweep, a market-structure shift back (agents/indicators smc k=3: BOS / CHoCH in the other direction) and the
    entry on the return into the FVG of that shift (the library's FVG return, SC250 mechanics); stop beyond the gap
    (0.1 ATR); target = the other side of the 4H range. Live version (ours): known bar by bar inside the 4H candle -
    SC286 needs the 4H candle CLOSED back inside and enters at its close. Higher timeframe: 4H for bars < 4H, 1D for
    4H bars, a week for daily."""
    o, h, l, c, v = _ohlcv(df)
    rule = _crt_rule(df)
    ph, pl, key = _htf_prev_range(df, rule)
    lo_run, hi_run = _run_ext(l, key, "min"), _run_ext(h, key, "max")
    sm = IND.smc(df, 3)
    mss_u = _bars_since(sm["bos_up"] | sm["choch_up"]) <= 6
    mss_d = _bars_since(sm["bos_dn"] | sm["choch_dn"]) <= 6
    ok_b = (lo_run < pl) & mss_u & (c > pl) & (c < ph)
    ok_s = (hi_run > ph) & mss_d & (c < ph) & (c > pl)
    L, S, sl, ss = _fvg_return(df, ok_b, ok_s, max_age=12)
    L = L & (c < ph)
    S = S & (c > pl)
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl, ss, ph, pl)


# ---------------- SC291 Trade only the strongest of a correlated pair (ictrules; set7 (2)) ----------------
def s_fvg_strongest_of_pair(df):
    """ictrules (TikTok, task tal1411set7 (2)): when EURUSD / GBPUSD (US100 / US500 / US30) give the same setup, take it
    only on the instrument with the cleaner, faster move. Ours: the setup = the FVG return of the library (SC250
    mechanics); 'faster' = the instrument's 10-bar displacement in ATRs is larger than the partner's
    (agents/intermarket_strategies partner; partner ATR = its 14-bar mean high-low) in the trade's direction; 'less
    chop' = the instrument's Kaufman efficiency ratio over 10 bars >= 0.3. Gap stop, 2R. No partner -> no signals.
    Compare with SC250 (every instrument)."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    part = _partner_hl(df)
    if part is None:
        z = np.zeros(n, bool)
        return z, z, {"stop": ("atr", 2.0), "target": None}
    a = atr(h, l, c, 14)
    pm = ((part["h"] + part["l"]) / 2).to_numpy(float)
    pa = (part["h"] - part["l"]).rolling(14, min_periods=5).mean().to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_own = (c - _sh(c, 10)) / a
        d_p = (pm - _sh(pm, 10)) / pa
    er = IND.efficiency_ratio(df)
    er = np.abs(er["er"] if isinstance(er, dict) else er)
    ok_b = (d_own > np.maximum(np.nan_to_num(d_p), 0)) & (er >= 0.3)
    ok_s = (d_own < np.minimum(np.nan_to_num(d_p), 0)) & (er >= 0.3)
    L, S, sl, ss = _fvg_return(df, ok_b, ok_s)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC292 / SC293 SMT confirmation models (The Creative Trader; CY FOREX; set7 (3)) ----------------
def _smt(df):
    part = _partner_hl(df)
    if part is None:
        return None
    return IND.smt_divergence(df, part, 3)


def s_smt_ifvg_cisd(df):
    """The Creative Trader (TikTok, task tal1411set7 (3)): SMT (NQ sweeps buy-side liquidity, ES does not - or EURUSD vs
    GBPUSD) -> the higher-timeframe FVG holds -> an FVG inverts (iFVG) -> CISD -> entry. Ours: a bearish SMT at a
    confirmed swing high (smt_divergence, k=3), then within 20 bars a bearish iFVG (agents/indicators ifvg: a bullish
    gap closed through) and then a bearish CISD close (agents/indicators cisd) within 10 bars of the inversion -> sell
    at the CISD close; stop 0.1 ATR over the SMT swing; target 2R (ours for 'the opposing liquidity'); the HTF FVG
    step is left to the bots' HTF filters. Bullish mirror. Differs from TT03 (sweep + POI, no SMT) and SC288 (SMT +
    a simple structure shift)."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    sm = _smt(df)
    if sm is None:
        z = np.zeros(n, bool)
        return z, z, {"stop": ("atr", 2.0), "target": None}
    iv, cs = IND.ifvg(df), IND.cisd(df)
    a = atr(h, l, c, 14)
    bb, bs = _bars_since(sm["bull"]), _bars_since(sm["bear"])
    ib, is_ = _bars_since(iv["inv_bull"]), _bars_since(iv["inv_bear"])
    plo, phi = _s(sm["piv_lo"]).ffill().to_numpy(), _s(sm["piv_hi"]).ffill().to_numpy()
    L = cs["bull"] & (bb <= 20) & (bb < bs) & (ib <= 10) & (ib <= bb)
    S = cs["bear"] & (bs <= 20) & (bs < bb) & (is_ <= 10) & (is_ <= bs)
    L, S = _once_since(L, sm["bull"]), _once_since(S, sm["bear"])
    L, S = _excl(L, S)
    return L, S, _plan_r(plo - 0.1 * a, phi + 0.1 * a, 2.0)


def s_smt_engulf(df):
    """CY FOREX (TikTok, task tal1411set7 (3)): SMT divergence + an engulfing candle. Ours: within 5 bars of a bearish
    SMT (smt_divergence k=3) a bearish engulfing candle -> sell at its close; stop 0.1 ATR over the SMT swing; 2R.
    Bullish mirror. Once per divergence."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    sm = _smt(df)
    if sm is None:
        z = np.zeros(n, bool)
        return z, z, {"stop": ("atr", 2.0), "target": None}
    a = atr(h, l, c, 14)
    eb, es = _engulf(o, c)
    bb, bs = _bars_since(sm["bull"]), _bars_since(sm["bear"])
    plo, phi = _s(sm["piv_lo"]).ffill().to_numpy(), _s(sm["piv_hi"]).ffill().to_numpy()
    L = _once_since(eb & (bb <= 5) & (bb < bs), sm["bull"])
    S = _once_since(es & (bs <= 5) & (bs < bb), sm["bear"])
    L, S = _excl(L, S)
    return L, S, _plan_r(plo - 0.1 * a, phi + 0.1 * a, 2.0)


# ---------------- SC294 Inversion FVG retest (LuxAlgo; set7 (4)) ----------------
def s_ifvg_retest(df):
    """Inversion Fair Value Gaps [LuxAlgo] (task tal1411set7 (4)): a gap closed through by a candle body flips role;
    the entry is the return into the inverted zone with a close back out (LuxAlgo's retest signal, wicks): long on a
    bullish iFVG (a bearish gap closed through upward), short mirror; stop 0.1 ATR beyond the zone's far side; 2R
    (ours). agents/indicators ifvg (gaps >= 0.25 ATR, ours)."""
    o, h, l, c, v = _ohlcv(df)
    iv = IND.ifvg(df)
    a = atr(h, l, c, 14)
    L, S = _excl(iv["retest_long"], iv["retest_short"])
    return L, S, _plan_r(iv["bull_bot"] - 0.1 * a, iv["bear_top"] + 0.1 * a, 2.0)


# ---------------- SC295 / SC296 CHoCH with confirmation vs the immediate entry (TradeMachine, LuxAlgo; set7 (5)) ---------
def s_choch_confirmed_retest(df):
    """TradeMachine / LuxAlgo (TikTok, task tal1411set7 (5)): do not enter on the CHoCH itself - wait for the return
    to the broken level / the order block and a confirmation candle. Ours: after a bullish CHoCH (agents/indicators
    smc k=5) the broken swing high is the level; within 30 bars the low comes back to the level + 0.2 ATR (or into
    the live bullish OB) and the bar confirms (bullish engulfing, or a bullish close over the previous high) -> buy;
    stop 0.1 ATR under the lower of the OB bottom and the pullback low; 2R. Disarmed by a close under the OB / the
    CHoCH leg's low. Bearish mirror. Compare with SC296 (enter on the CHoCH close)."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    sm = IND.smc(df, 5)
    a = atr(h, l, c, 14)
    eb, es = _engulf(o, c)
    conf_b = eb | ((c > o) & (c > _sh(h)))
    conf_s = es | ((c < o) & (c < _sh(l)))
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    ab = as_ = None            # (level, floor, armed_at, low_since)
    for i in range(n):
        if sm["choch_up"][i]:
            lv = _sh(sm["swing_hi"])[i] if i else np.nan
            lv = lv if np.isfinite(lv) else h[i]
            fl = sm["ob_bull_bot"][i] if np.isfinite(sm["ob_bull_bot"][i]) else l[i] - a[i]
            ab, as_ = [lv, fl, i, l[i]], None
            continue
        if sm["choch_dn"][i]:
            lv = _sh(sm["swing_lo"])[i] if i else np.nan
            lv = lv if np.isfinite(lv) else l[i]
            fl = sm["ob_bear_top"][i] if np.isfinite(sm["ob_bear_top"][i]) else h[i] + a[i]
            as_, ab = [lv, fl, i, h[i]], None
            continue
        if ab is not None:
            ab[3] = min(ab[3], l[i])
            if c[i] < ab[1] or i - ab[2] > 30:
                ab = None
            elif l[i] <= max(ab[0] + 0.2 * a[i], sm["ob_bull_top"][i] if np.isfinite(sm["ob_bull_top"][i]) else -np.inf) \
                    and conf_b[i]:
                L[i], sl_l[i] = True, min(ab[1], ab[3]) - 0.1 * a[i]
                ab = None
        if as_ is not None:
            as_[3] = max(as_[3], h[i])
            if c[i] > as_[1] or i - as_[2] > 30:
                as_ = None
            elif h[i] >= min(as_[0] - 0.2 * a[i], sm["ob_bear_bot"][i] if np.isfinite(sm["ob_bear_bot"][i]) else np.inf) \
                    and conf_s[i]:
                S[i], sl_s[i] = True, max(as_[1], as_[3]) + 0.1 * a[i]
                as_ = None
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


def s_choch_immediate(df):
    """The comparison for SC295 (task tal1411set7 (5): 'compare with the immediate entry'): buy on the bullish CHoCH
    close itself (agents/indicators smc k=5); stop 0.1 ATR under the last confirmed swing low; 2R. Mirror."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df, 5)
    a = atr(h, l, c, 14)
    L, S = _excl(sm["choch_up"], sm["choch_dn"])
    return L, S, _plan_r(sm["swing_lo"] - 0.1 * a, sm["swing_hi"] + 0.1 * a, 2.0)


# ---------------- SC297 Heikin Ashi swing: colour change + 2nd candle without the opposite wick (set7 (6)) ----------
def s_ha_swing_confirm(df):
    """Heikin Ashi swing (TikTok, task tal1411set7 (6), 4H / daily on gold, indices, bitcoin): the HA candle changes
    colour and the SECOND candle in a row has no opposite wick (green with no lower wick) -> buy at its close; exit on
    the first HA candle of the opposite colour. Stop (not in the video, ours): 0.1 ATR beyond the 5-bar extreme.
    Mirror short. agents/indicators heikin_ashi. Differs from TV25 (first green no-wick candle over EMA50, no colour
    exit)."""
    o, h, l, c, v = _ohlcv(df)
    ha = IND.heikin_ashi(df)
    a = atr(h, l, c, 14)
    L, S = _excl(ha["confirm_up"], ha["confirm_dn"])
    sl, ss = _swing_stops(h, l, a, 5)
    return L, S, {"stop": ("level", sl, ss), "target": None, "exit_long": ha["red"], "exit_short": ha["green"]}


# ---------------- SC298 / SC299 Internal vs external liquidity targets (Ali Khan; set7 (7)) ----------------
def s_erl_sweep_to_irl(df):
    """Ali Khan (TikTok, task tal1411set7 (7)): after external liquidity is taken (the dealing range's swing high /
    low), the target is the internal liquidity (an FVG inside the range). Ours: the bar sweeps the last confirmed
    swing high (agents/indicators smc k=5) and closes back under it -> sell; stop 0.1 ATR over the sweep high; target
    the top of the latest unfilled bullish FVG under the price (the IRL), else 2R. Mirror at the swing low."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df, 5)
    a = atr(h, l, c, 14)
    shi, slo = _sh(sm["swing_hi"]), _sh(sm["swing_lo"])
    S = (h > shi) & (c < shi) & (c < o)
    L = (l < slo) & (c > slo) & (c > o)
    sl_l, sl_s = l - 0.1 * a, h + 0.1 * a
    tg_s = np.where(np.isfinite(sm["fvg_bull_top"]) & (sm["fvg_bull_top"] < c), sm["fvg_bull_top"], c - 2 * (sl_s - c))
    tg_l = np.where(np.isfinite(sm["fvg_bear_bot"]) & (sm["fvg_bear_bot"] > c), sm["fvg_bear_bot"], c + 2 * (c - sl_l))
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


def s_irl_fill_to_erl(df):
    """Ali Khan (task tal1411set7 (7)), the other half: after the internal FVG is filled, the target is the external
    liquidity. Ours: structure trend up (smc k=5), the bar trades into the live bullish FVG and closes back over its
    top (bullish candle) -> buy; stop 0.1 ATR under the gap; target the last confirmed swing high (the ERL) - no trade
    when it is not above the price. Mirror short to the swing low."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df, 5)
    a = atr(h, l, c, 14)
    ft, fb = _sh(sm["fvg_bull_top"]), _sh(sm["fvg_bull_bot"])
    gt, gb = _sh(sm["fvg_bear_top"]), _sh(sm["fvg_bear_bot"])
    L = (sm["trend"] == 1) & (l <= ft) & (c > ft) & (c > o) & (sm["swing_hi"] > c)
    S = (sm["trend"] == -1) & (h >= gb) & (c < gb) & (c < o) & (sm["swing_lo"] < c)
    L, S = _excl(L, S)
    return L, S, _plan_lv(fb - 0.1 * a, gt + 0.1 * a, sm["swing_hi"], sm["swing_lo"])


# ---------------- SC300 Pullback to the last swing's POC (Swing Profile Analyzer, ChartPrime; set7 (8)) ----------------
def s_swing_poc_pullback(df):
    """Swing Profile Analyzer [ChartPrime] (task tal1411set7 (8): the POC of each swing as a filter / builder): after an
    up leg completes (agents/indicators swing_profile k=5, frequency-of-closes profile), the first pullback that trades
    to the leg's POC and closes back over it -> buy; stop 0.1 ATR under the leg's value-area low (at least 0.5 ATR under
    the POC); target the leg's high (no trade when it is not above the price). Mirror after a down leg."""
    o, h, l, c, v = _ohlcv(df)
    sp = IND.swing_profile(df)
    a = atr(h, l, c, 14)
    poc, d = sp["leg_poc"], sp["leg_dir"]
    L = (d == 1) & (l <= poc) & (c > poc) & (c < sp["leg_hi"])
    S = (d == -1) & (h >= poc) & (c < poc) & (c > sp["leg_lo"])
    L, S = _once_since(L, sp["new_leg"]), _once_since(S, sp["new_leg"])
    sl_l = np.minimum(sp["leg_val"], poc - 0.5 * a) - 0.1 * a
    sl_s = np.maximum(sp["leg_vah"], poc + 0.5 * a) + 0.1 * a
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, sp["leg_hi"], sp["leg_lo"])


# ---------------- SC301 SK System: structure + 0.5-0.618 retracement + extension target (TradersNotes; set7 (9)) ---------
def s_sk_system(df):
    """SK System (TradersNotes, TikTok, task tal1411set7 (9); 'SK Trading System' on TradingView is invite-only - rules
    from the video): structure gives the direction; the Fibonacci retracement of the last leg = the zone of the next
    HL (0.5-0.618); the extension = the target of the next HH (1.272 / 1.414 / 2.0). Ours: structure trend up
    (smc k=5), the latest up leg (agents/indicators fib_legs k=3), the low trades into its 0.5-0.618 zone and the bar
    closes back over the zone's top (bullish) -> buy; stop 0.1 ATR under the 0.786 level (the dz bottom); target the
    1.414 extension (the middle of the video's three). Mirror in a down trend. Differs from SC254 (needs the FVG + OB
    overlap) and SC256 (CHoCH -> BOS sequence)."""
    o, h, l, c, v = _ohlcv(df)
    fl = IND.fib_legs(df)
    sm = IND.smc(df, 5)
    a = atr(h, l, c, 14)
    L = (sm["trend"] == 1) & (l <= fl["u_gz_top"]) & (l >= fl["u_dz_bot"]) & (c > fl["u_gz_top"]) & (c > o)
    S = (sm["trend"] == -1) & (h >= fl["d_gz_bot"]) & (h <= fl["d_dz_top"]) & (c < fl["d_gz_bot"]) & (c < o)
    L, S = _excl(L, S)
    return L, S, _plan_lv(fl["u_dz_bot"] - 0.1 * a, fl["d_dz_top"] + 0.1 * a, fl["u_ext1414"], fl["d_ext1414"])


# ---------------- SC302 Double top / bottom: entry at the second top (set7 (10), the comparison for TV57) ----------------
def s_double_top_second_touch(df):
    """Task tal1411set7 (10): 'a double top is not an immediate short - wait for the neckline break' - the library has
    the neckline entry (TV57); this is the comparison the user asks for: the short AT the second top. Ours: the last
    confirmed swing high (k=3) at least 8 bars back; the bar's high comes within 0.3 ATR of it (not more than 0.3 ATR
    over), the bar is a bearish rejection (close under the open, upper wick >= the body), the neckline (lowest low
    between) at least 1 ATR under the top -> sell; stop 0.2 ATR over the higher top; target the neckline. Mirror for
    the double bottom."""
    o, h, l, c, v = _ohlcv(df)
    from agents.indicators._core import pivots
    n = len(c)
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, phi, pli = pivots(h, l, 3)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s, tg_l, tg_s = (np.full(n, np.nan) for _ in range(4))
    top = bot = None
    for i in range(n):
        if phc[i]:
            top = (phv[i], int(phi[i]))
        if plc[i]:
            bot = (plv[i], int(pli[i]))
        if not np.isfinite(a[i]):
            continue
        uw, lw, body = h[i] - max(o[i], c[i]), min(o[i], c[i]) - l[i], abs(c[i] - o[i])
        if top is not None and i - top[1] >= 8 and abs(h[i] - top[0]) <= 0.3 * a[i] and c[i] < o[i] and uw >= body:
            neck = l[top[1]:i].min()
            if top[0] - neck >= a[i]:
                S[i], sl_s[i], tg_s[i] = True, max(h[i], top[0]) + 0.2 * a[i], neck
                top = None
        if bot is not None and i - bot[1] >= 8 and abs(l[i] - bot[0]) <= 0.3 * a[i] and c[i] > o[i] and lw >= body:
            neck = h[bot[1]:i].max()
            if neck - bot[0] >= a[i]:
                L[i], sl_l[i], tg_l[i] = True, min(l[i], bot[0]) - 0.2 * a[i], neck
                bot = None
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


# ---------------- SC303 London / New York reversal with an IFVG (QuantFlowLabs, EURUSD 1m; set7 (11)) ----------------
def s_london_ny_ifvg_reversal(df):
    """QuantFlowLabs (TikTok, task tal1411set7 (11), EURUSD 1-minute): London sweeps the sell-side liquidity (the Asian
    low) and an IFVG forms upward -> long to the FVG / range targets; New York sweeps London's high and flips -> short.
    Ours (London time): Asia = 00:00-07:00; London window 07:00-10:00: the day's low since 07:00 under the Asian low,
    a bullish iFVG (agents/indicators ifvg) within the last 5 bars and the close back over the Asian low -> buy, target
    the Asian high; mirror for a sweep of the Asian high. New York window 13:30-16:00: the high since 13:30 over the
    London 07:00-12:00 high, a bearish iFVG within 5 bars, close back under the London high -> sell, target the London
    low (mirror long). Stop 0.1 ATR beyond the sweep extreme; one trade per window per day; no trade when the target
    is not on the right side."""
    o, h, l, c, v = _ohlcv(df)
    n = len(c)
    hs, he, _, _, day = _local(df, "Europe/London")
    a = atr(h, l, c, 14)
    iv = IND.ifvg(df)
    ib, is_ = _bars_since(iv["inv_bull"]) <= 5, _bars_since(iv["inv_bear"]) <= 5
    dk = pd.Series(day)

    def rng(m, x, fn):
        s = pd.Series(np.where(m, x, np.nan))
        r = s.groupby(dk).cummax() if fn == "max" else s.groupby(dk).cummin()
        return r.groupby(dk).ffill().to_numpy()
    asia = he <= 7.0
    a_hi, a_lo = rng(asia, h, "max"), rng(asia, l, "min")
    lon = (hs >= 7.0) & (he <= 12.0)
    l_hi, l_lo = rng(lon, h, "max"), rng(lon, l, "min")
    w1 = (hs >= 7.0) & (hs < 10.0)
    w2 = (hs >= 13.5) & (hs < 16.0)
    lo1, hi1 = rng(w1, l, "min"), rng(w1, h, "max")
    lo2, hi2 = rng(w2, l, "min"), rng(w2, h, "max")
    L1 = w1 & (lo1 < a_lo) & ib & (c > a_lo) & (a_hi > c)
    S1 = w1 & (hi1 > a_hi) & is_ & (c < a_hi) & (a_lo < c)
    S2 = w2 & (hi2 > l_hi) & is_ & (c < l_hi) & (l_lo < c)
    L2 = w2 & (lo2 < l_lo) & ib & (c > l_lo) & (l_hi > c)
    L = _first_per_day(L1, day) | _first_per_day(L2, day)
    S = _first_per_day(S1, day) | _first_per_day(S2, day)
    sl_l = np.where(w1, lo1, lo2) - 0.1 * a
    sl_s = np.where(w1, hi1, hi2) + 0.1 * a
    tg_l = np.where(w1, a_hi, l_hi)
    tg_s = np.where(w1, a_lo, l_lo)
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


# ---------------- SC304 LazyBear Squeeze Momentum: release in the histogram's colour, squeeze-range stop, 2R (set7 (12)) ---
def s_lb_squeeze_structural(df):
    """Squeeze Momentum [LazyBear] (task tal1411set7 (12): 'add LazyBear's version'; agents/indicators squeeze_momentum,
    exact port incl. the 1.5-SD quirk): the first bar the squeeze is off after >= 3 bars on, the histogram lime (> 0
    and rising) -> buy, red (< 0 and falling) -> sell; stop 0.1 ATR beyond the squeeze's extreme (the bars while it
    was on); target 2R; out early when the histogram loses strength (green for longs, maroon for shorts). Differs from
    GH09 (3 ATR stop, no target, any colour) and SC273 (TTM Squeeze Pro levels)."""
    o, h, l, c, v = _ohlcv(df)
    sq = IND.squeeze_momentum(df)
    a = atr(h, l, c, 14)
    fired = sq["fired"] & (_sh(sq["bars_on"]) >= 3)
    nb = np.nan_to_num(_sh(sq["bars_on"]), nan=1).astype(int)
    n = len(c)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for i in np.where(fired)[0]:
        j = max(0, i - max(nb[i], 1))
        sl_l[i], sl_s[i] = l[j:i + 1].min() - 0.1 * a[i], h[j:i + 1].max() + 0.1 * a[i]
    L, S = _excl(fired & sq["lime"], fired & sq["red"])
    return L, S, _plan_r(sl_l, sl_s, 2.0, exit_long=sq["green"], exit_short=sq["maroon"])


# ---------------- SC305 Fibonacci EMA stack 8/13/21/34/55: pullback to 21-34 (Trading Data; set7 (13)) ----------------
def s_fib_ema_stack_pullback(df):
    """Trading Data (TikTok, task tal1411set7 (13)): EMAs 8 / 13 / 21 / 34 / 55 (Fibonacci); when the stack is in order,
    enter on the pullback to the 21 / 34. Ours: full bullish order (agents/indicators ema_stack), the low reaches the
    EMA21 (or deeper, down to the EMA34 - 0.1 ATR... not under the EMA55), the close back over the EMA21 with a
    bullish candle -> buy; stop 0.1 ATR under the EMA55; target 2R. Mirror short."""
    o, h, l, c, v = _ohlcv(df)
    es = IND.ema_stack(df, (8, 13, 21, 34, 55))
    e21, e34, e55 = es["lines"][21], es["lines"][34], es["lines"][55]
    a = atr(h, l, c, 14)
    L = _shb(es["full_up"]) & (l <= e21) & (l > e55) & (c > e21) & (c > o)
    S = _shb(es["full_dn"]) & (h >= e21) & (h < e55) & (c < e21) & (c < o)
    L, S = _excl(L, S)
    return L, S, _plan_r(np.minimum(e55, e34) - 0.1 * a, np.maximum(e55, e34) + 0.1 * a, 2.0)


# ---------------- SC306 ICT Unicorn: breaker + FVG overlap retest (LuxAlgo concept; Flux Charts) ----------------
def s_ict_unicorn(df):
    """ICT Unicorn (https://www.luxalgo.com/library/concept/unicorn/ ; Flux Charts 'ICT Unicorn' open-source script):
    a sweep, a displacement that breaks structure and leaves an FVG, the breaker = the failed opposing order block;
    only where breaker and FVG overlap; limit entry on the retrace into the overlap, stop beyond the breaker's far side,
    target the opposing liquidity. Ours: agents/indicators unicorn (smc k=3 breakers); entry = the bar trades into the
    zone and closes back out of it; stop 0.1 ATR beyond the breaker; target the last swing extreme on the other side
    if it is >= 1.5R away, else 2R."""
    o, h, l, c, v = _ohlcv(df)
    u = IND.unicorn(df)
    sm = IND.smc(df, 3)
    a = atr(h, l, c, 14)
    sl_l, sl_s = u["stop_long"] - 0.1 * a, u["stop_short"] + 0.1 * a
    tg_l = np.where(sm["swing_hi"] - c >= 1.5 * (c - sl_l), sm["swing_hi"], c + 2 * (c - sl_l))
    tg_s = np.where(c - sm["swing_lo"] >= 1.5 * (sl_s - c), sm["swing_lo"], c - 2 * (sl_s - c))
    L, S = _excl(u["retest_long"], u["retest_short"])
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


# ---------------- SC307 Power of Three (AMD) around the NY midnight open (LuxAlgo concept / ICT) ----------------
def s_po3_midnight_open(df):
    """ICT Power of Three / AMD (https://www.luxalgo.com/library/concept/accumulation-manipulation-distribution/ ;
    FXOpen 'What is ICT PO3'): anchor = the New York midnight open; manipulation = a run beyond the open that takes
    liquidity (ours: under the open AND under the Asian 20:00-00:00 NY range low), distribution = displacement back
    through the open -> join it. Ours: the first close back over the midnight open after a bearish manipulation,
    between 00:00 and 11:00 NY -> buy; stop 0.1 ATR under the day's low (the manipulation extreme); target 2R, out by
    16:00 NY at the latest. Mirror short. agents/indicators po3_midnight. Differs from SC206 (Asian-range Judas + OB)
    and SC258 (09:30 range)."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.po3_midnight(df)
    a = atr(h, l, c, 14)
    hs, he, _, _, day = _local(df, "America/New_York")
    w = hs < 11.0
    L = _first_per_day(p["dist_long"] & w, day)
    S = _first_per_day(p["dist_short"] & w, day)
    L, S = _excl(L, S)
    return L, S, _plan_r(p["day_lo"] - 0.1 * a, p["day_hi"] + 0.1 * a, 2.0, force_exit=he >= 16.0)


# ---------------- SC308 ICT rejection block revisit (LuxAlgo concept) ----------------
def s_rejection_block(df):
    """Rejection block (https://www.luxalgo.com/library/concept/rejection-block/ ; UAlgo 'Rejection Blocks' open-source):
    a swing extreme with a long wick; the zone = body edge to wick extreme; traded on the revisit after price left;
    stop beyond the wick; invalid on body acceptance beyond it. Ours: agents/indicators rejection_blocks (pivots k=3,
    wick >= 1.5 x body); entry when the bar trades into the zone and closes back out of it; stop 0.1 ATR beyond the
    wick; 2R (no fixed target in the source)."""
    o, h, l, c, v = _ohlcv(df)
    rb = IND.rejection_blocks(df)
    a = atr(h, l, c, 14)
    L, S = _excl(rb["retest_long"], rb["retest_short"])
    return L, S, _plan_r(rb["bull_bot"] - 0.1 * a, rb["bear_top"] + 0.1 * a, 2.0)


# ---------------- SC309 NWOG / NDOG edge rejection (ICT opening gaps; LuxAlgo concept) ----------------
def s_opening_gap_edge(df):
    """ICT New Week / New Day Opening Gaps (https://www.luxalgo.com/library/concept/opening-gap/ ; /new-week-opening-gap/):
    the span between the previous close and the reopen; CE = its 50%; watch the first reaction at an edge. Ours
    (agents/indicators opening_gaps, 17:00 New York break): price comes down to the latest NWOG / NDOG from above,
    trades into it and closes back over its top -> buy (the edge held); from below at the bottom -> sell; stop 0.1 ATR
    beyond the gap's CE (the edge failed if price accepts past the middle), at least 0.5 ATR beyond the edge; 2R.
    Every reopen counts (min_gap_atr 0): on a feed without a gap the zone collapses to the reopen price itself (ICT's
    new day / week opening price) - the same edge test on that level."""
    o, h, l, c, v = _ohlcv(df)
    g = IND.opening_gaps(df, min_gap_atr=0.0)
    a = atr(h, l, c, 14)
    L = np.zeros(len(c), bool)
    S = np.zeros(len(c), bool)
    sl_l, sl_s = np.full(len(c), np.nan), np.full(len(c), np.nan)
    for pre in ("nwog", "ndog"):
        top, bot, ce = _sh(g[pre + "_top"]), _sh(g[pre + "_bot"]), _sh(g[pre + "_ce"])
        lo_ = (_sh(c) > top) & (l <= top) & (c > top) & (c > o) & ~L & ~S
        sh_ = (_sh(c) < bot) & (h >= bot) & (c < bot) & (c < o) & ~L & ~S
        sl_l = np.where(lo_, np.minimum(ce - 0.1 * a, top - 0.5 * a), sl_l)
        sl_s = np.where(sh_, np.maximum(ce + 0.1 * a, bot + 0.5 * a), sl_s)
        L, S = L | lo_, S | sh_
    L, S = _excl(L, S)
    return L, S, _plan_r(sl_l, sl_s, 2.0)


# ---------------- SC310 Balanced Price Range retest (ICT; UAlgo / TakingProphets scripts) ----------------
def s_bpr_retest(df):
    """ICT Balanced Price Range (open-source 'ICT Balance Price Range [UAlgo]' / 'BPR [TakingProphets]' on TradingView;
    our build of the standard definition): a bearish FVG and a bullish FVG within 5 bars whose ranges overlap - the
    overlap is the zone, biased to the later gap. Entry on the return into the zone and a close back out in its
    direction; stop 0.1 ATR beyond the zone; 2R (ours). agents/indicators bpr."""
    o, h, l, c, v = _ohlcv(df)
    b = IND.bpr(df)
    a = atr(h, l, c, 14)
    L, S = _excl(b["retest_long"], b["retest_short"])
    return L, S, _plan_r(b["bull_bot"] - 0.1 * a, b["bear_top"] + 0.1 * a, 2.0)


# ---------------- SC311 WH SelfInvest London Reversal System: fade the Asian extreme +/- ATR, 5 ATR / 5 ATR ----------------
def s_whs_london_reversal(df):
    """WH SelfInvest 'The London Reversal System' (free strategy 93,
    https://www.whselfinvest.de/en-de/trading-platform/free-trading-strategies/tradingsystem/93-london-reversal-system ):
    the previous session's high / low channel extended by a fraction of the ATR; within 1-2 hours of the new session's
    start (London 08:00) buy when the market crosses under the session low - the ATR buffer, sell when it crosses over
    the high + buffer; stop 5 x ATR(20), target 5 x ATR(20) (1:1); forex and indices. Ours: the previous session =
    00:00-08:00 London (Asia), the buffer = 0.25 x ATR(20) (the page gives no fraction); the window 08:00-10:00 London;
    the same at the New York transition (London 08:00-13:00 range, window 13:00-15:00 London) - 'any session
    transition'; one trade per window."""
    o, h, l, c, v = _ohlcv(df)
    hs, he, _, _, day = _local(df, "Europe/London")
    a20 = atr(h, l, c, 20)
    dk = pd.Series(day)

    def rng(m, x, fn):
        s = pd.Series(np.where(m, x, np.nan))
        r = s.groupby(dk).cummax() if fn == "max" else s.groupby(dk).cummin()
        return r.groupby(dk).ffill().to_numpy()
    as_ = he <= 8.0
    lo_s = (hs >= 8.0) & (he <= 13.0)
    w1, w2 = (hs >= 8.0) & (hs < 10.0), (hs >= 13.0) & (hs < 15.0)
    hi = np.where(w1, rng(as_, h, "max"), rng(lo_s, h, "max"))
    lo = np.where(w1, rng(as_, l, "min"), rng(lo_s, l, "min"))
    up, dn = hi + 0.25 * a20, lo - 0.25 * a20
    w = w1 | w2
    L = w & (c < dn) & (_sh(c) >= dn)
    S = w & (c > up) & (_sh(c) <= up)
    wk = np.where(w1, 1, 2)
    key = pd.Series(day).astype(str) + "_" + pd.Series(wk).astype(str)
    L, S = _first_per_day(L, key.to_numpy()), _first_per_day(S, key.to_numpy())
    L, S = _excl(L, S)
    return L, S, _plan_lv(c - 5 * a20, c + 5 * a20, c + 5 * a20, c - 5 * a20)


# ---------------- SC312 Big Ben: fade the pre-London breakout of the Asian body range (Tradingstrategyguides) ----------------
def s_big_ben_fade(df):
    """'Big Ben' breakout-fade (Tradingstrategyguides, TradingView idea EURAUD 15M,
    https://id.tradingview.com/chart/EURAUD/O8s3VFzb-EURAUD-15M-BIG-BEN-BREAKOUT-TRADING-STRATEGY ): the Asian range
    from candle BODIES; momentum starts an hour before the London open and breaks the range; after the open the price
    reverses back into it (V-shape) -> enter; target = the range's size projected from the opposite side; time stop:
    out if the first hour after the open has not fully reversed the breakout. Ours (London time): bodies 00:00-07:00;
    a close outside between 07:00 and 08:00; from 08:05 to 09:00 the first close back inside -> trade back through the
    range; target = the far side - the range size; time stop = 60 minutes (max_bars); a safety stop 0.1 ATR beyond the
    pre-open extreme (the bots need a price stop)."""
    o, h, l, c, v = _ohlcv(df)
    hs, he, _, _, day = _local(df, "Europe/London")
    a = atr(h, l, c, 14)
    dk = pd.Series(day)
    bt, bb = np.maximum(o, c), np.minimum(o, c)

    def rng(m, x, fn):
        s = pd.Series(np.where(m, x, np.nan))
        r = s.groupby(dk).cummax() if fn == "max" else s.groupby(dk).cummin()
        return r.groupby(dk).ffill().to_numpy()
    asia = he <= 7.0
    top, bot = rng(asia, bt, "max"), rng(asia, bb, "min")
    pre = (hs >= 7.0) & (he <= 8.0)
    pre_up = rng(pre & (c > top), np.ones(len(c)), "max") > 0
    pre_dn = rng(pre & (c < bot), np.ones(len(c)), "max") > 0
    ext_hi = rng((hs >= 7.0) & (hs < 9.0), h, "max")
    ext_lo = rng((hs >= 7.0) & (hs < 9.0), l, "min")
    w = (hs >= 8.0 + 5 / 60) & (hs < 9.0)
    S = _first_per_day(w & pre_up & (c < top) & (c > bot), day)
    L = _first_per_day(w & pre_dn & (c > bot) & (c < top), day)
    size = top - bot
    L, S = _excl(L, S)
    mb = max(int(round(60 / max(_bar_minutes(df), 1))), 1)
    return L, S, _plan_lv(ext_lo - 0.1 * a, ext_hi + 0.1 * a, top + size, bot - size, max_bars=mb)


# ---------------- SC313 Coppock Curve turn under zero (Coppock; StockCharts) ----------------
def s_coppock_turn(df):
    """Coppock Curve (StockCharts ChartSchool coppock-curve): WMA10 of ROC14 + ROC11; the buy signal = the curve turns
    up from below zero (monthly originally; here on the bots' charts, the test decides). Ours: also the mirror short
    (turns down from above zero); stop 2 ATR; out when the curve turns the other way. agents/indicators coppock."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.coppock(df)
    L, S = _excl(k["turn_up"] & (k["cc"] < 0), k["turn_dn"] & (k["cc"] > 0))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": k["turn_dn"], "exit_short": k["turn_up"]}


# ---------------- SC314 Ultimate Oscillator divergence, Williams' three rules ----------------
def s_ultimate_osc_divergence(df):
    """Ultimate Oscillator (Larry Williams; StockCharts ultimate-oscillator): buy = (1) a bullish divergence (price lower
    low, UO higher low), (2) the first low of the divergence under 30, (3) UO rises over the divergence's high (the
    highest UO between the two lows). Sell = the bearish mirror (first high over 70). Exit: UO over 70 (longs) / under
    30 (shorts) or the stop - ours 0.1 ATR beyond the second extreme. Pivots k=2, divergence armed 30 bars. Ours: the
    first extreme's zone is 35 / 65 (Williams' 30 / 70 is for daily bars; on intraday bars the 7/14/28 averages rarely
    reach 30 / 70 at a swing - the 30 / 70 exits stay)."""
    o, h, l, c, v = _ohlcv(df)
    from agents.indicators._core import pivots
    n = len(c)
    uo = IND.ultimate_oscillator(df)
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, phi, pli = pivots(h, l, 2)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    pl_prev = ph_prev = None
    ab = as_ = None
    for i in range(n):
        if plc[i]:
            j = int(pli[i])
            cur = (l[j], uo[j], j)
            if pl_prev is not None and cur[0] < pl_prev[0] and cur[1] > pl_prev[1] and pl_prev[1] < 35:
                ab = (np.nanmax(uo[pl_prev[2]:j + 1]), cur[0], i)
            pl_prev = cur
        if phc[i]:
            j = int(phi[i])
            cur = (h[j], uo[j], j)
            if ph_prev is not None and cur[0] > ph_prev[0] and cur[1] < ph_prev[1] and ph_prev[1] > 65:
                as_ = (np.nanmin(uo[ph_prev[2]:j + 1]), cur[0], i)
            ph_prev = cur
        if ab is not None:
            if i - ab[2] > 30 or l[i] < ab[1]:
                ab = None
            elif uo[i] > ab[0]:
                L[i], sl_l[i] = True, ab[1] - 0.1 * a[i]
                ab = None
        if as_ is not None:
            if i - as_[2] > 30 or h[i] > as_[1]:
                as_ = None
            elif uo[i] < as_[0]:
                S[i], sl_s[i] = True, as_[1] + 0.1 * a[i]
                as_ = None
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", sl_l, sl_s), "target": None, "exit_long": uo > 70, "exit_short": uo < 30}


# ---------------- SC315 Klinger Volume Oscillator signal cross with the trend (StockCharts) ----------------
def s_klinger_trend_cross(df):
    """Klinger Volume Oscillator (StockCharts klinger-volume-oscillator: signal-line crosses in the direction of the
    trend; tick volume here): KVO(34, 55) crosses over its 13-EMA signal while the close is over the EMA100 -> buy;
    mirror short; stop 2 ATR, target 2R (ours). agents/indicators klinger."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.klinger(df)
    e = ema(c, 100)
    L = cross_up(k["kvo"], k["signal"]) & (c > e) & (k["kvo"] < 0)
    S = cross_down(k["kvo"], k["signal"]) & (c < e) & (k["kvo"] > 0)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC316 Mass Index as a range-expansion filter on the EMA 9 / 21 cross (Dorsey; StockCharts) ----------------
def s_mass_index_filter_cross(df):
    """Mass Index (Donald Dorsey; StockCharts mass-index): a 'reversal bulge' (MI over 27, then under 26.5) warns of a
    trend reversal; Dorsey's own entry = the direction of the EMA9 after the bulge. Filter variant (ours): the EMA 9 /
    21 cross is traded only when NO bulge happened in the last 20 bars (no pending reversal), and the bulge itself
    flips the trade: an EMA9 turn within 10 bars after a bulge -> trade the turn. Stop 2 ATR, 2R. agents/indicators
    mass_index."""
    o, h, l, c, v = _ohlcv(df)
    mi = IND.mass_index(df)
    e9, e21 = ema(c, 9), ema(c, 21)
    recent = _bars_since(mi["bulge_end"]) <= 20
    t_up = (e9 > _sh(e9)) & (_sh(e9) <= _sh(e9, 2))
    t_dn = (e9 < _sh(e9)) & (_sh(e9) >= _sh(e9, 2))
    near = _bars_since(mi["bulge_end"]) <= 10
    L = (~recent & cross_up(e9, e21)) | (near & t_up)
    S = (~recent & cross_down(e9, e21)) | (near & t_dn)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC317 McGinley Dynamic threshold cross with cooldown (StockSharp 1038) ----------------
def s_mcginley_threshold(df):
    """McGinley Dynamic (Improved) - StockSharp example 1038
    (https://doc.stocksharp.com/en/api-examples/1038_McGinley_Dynamic_Improved ), 15-minute: MD(20, k 0.6); diff =
    (close - MD) / close x 100; long when diff crosses over +0.25%, short when it crosses under -0.25%; at least 10
    bars between entries; always in (reverse). Stop: none in the source - 2 ATR (ours). agents/indicators mcginley."""
    o, h, l, c, v = _ohlcv(df)
    md = IND.mcginley(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (c - md) / c * 100
    L0, S0 = cross_up(d, 0.25), cross_down(d, -0.25)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    last = -10**6
    for i in range(n):
        if (L0[i] or S0[i]) and i - last >= 10:
            L[i], S[i] = L0[i], S0[i] and not L0[i]
            last = i
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": S0, "exit_short": L0}


# ---------------- SC318 Elder Ray: EMA13 slope + rising bear power under zero (Elder) ----------------
def s_elder_ray(df):
    """Elder Ray (Alexander Elder, 'Trading for a Living'; https://www.luxalgo.com/library/concept/elder-ray/ ): buy
    when the EMA13 rises and bear power (low - EMA13) is negative but rising; sell when the EMA13 falls and bull power
    is positive but falling. Stop 0.1 ATR beyond the 5-bar extreme, target 2R (ours). agents/indicators elder_ray."""
    o, h, l, c, v = _ohlcv(df)
    er = IND.elder_ray(df)
    a = atr(h, l, c, 14)
    sl, ss = _swing_stops(h, l, a, 5)
    L, S = _excl(_onset(er["buy"]), _onset(er["sell"]))
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC319 Wilder's Volatility System: ATR(7) x 3 stop-and-reverse (Wilder 1978) ----------------
def s_wilder_volatility_system(df):
    """Welles Wilder's Volatility System (New Concepts in Technical Trading Systems, 1978; open-source
    https://www.tradingview.com/script/pjZKWnHl-Volatility-System-by-Wilder-LucF ): ARC = 3 x ATR(7); the SAR trails
    the most favourable CLOSE by the ARC; a close through it reverses the position (always in). Stop = the SAR; exit
    on the reversal. agents/indicators volatility_system."""
    o, h, l, c, v = _ohlcv(df)
    vs = IND.volatility_system(df)
    L, S = _excl(vs["flip_up"], vs["flip_dn"])
    return L, S, {"stop": ("level", vs["sar"], vs["sar"]), "target": None, "exit_long": vs["flip_dn"],
                  "exit_short": vs["flip_up"]}


# ---------------- SC320 FRAMA cross with its slope (Ehlers 2005) ----------------
def s_frama_cross(df):
    """FRAMA (John Ehlers, S&C Oct 2005; agents/indicators frama, period 16 on hl2): the close crosses over a RISING FRAMA
    -> buy, under a falling one -> sell; out on the cross back; stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    f = IND.frama(df)
    L = cross_up(c, f) & (f > _sh(f))
    S = cross_down(c, f) & (f < _sh(f))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": c < f, "exit_short": c > f}


# ---------------- SC321 Hurst-exponent mean reversion to the SMA20 (StockSharp 0228) ----------------
def s_hurst_reversion(df):
    """Hurst Exponent Reversion - StockSharp example 0228
    (https://doc.stocksharp.com/en/api-examples/0228_Hurst_Exponent_Reversion ), 5-minute: Hurst(100) < 0.7 (mean-
    reverting) and the close under the SMA20 -> buy; over -> sell; exit at the SMA20 or when Hurst > 0.7; stop 2% of
    the price (translated: kept as 2% - the source's own unit). agents/indicators hurst."""
    o, h, l, c, v = _ohlcv(df)
    hu = IND.hurst(df)
    m = sma(c, 20)
    rev = hu < 0.7
    L, S = _excl(_onset(rev & (c < m)), _onset(rev & (c > m)))
    return L, S, {"stop": ("level", c * 0.98, c * 1.02), "target": None,
                  "exit_long": (c >= m) | (hu > 0.7), "exit_short": (c <= m) | (hu > 0.7)}


# ---------------- SC322 Linear regression channel fade with the slope (StockSharp designer) ----------------
def s_linreg_channel_fade(df):
    """Linear Regression Channel - StockSharp designer gallery
    (https://doc.stocksharp.com/en/designer-gallery/Linear_Regression_Channel ), 5-minute: regression(50) +/- 1.5
    standard errors; slope > 0 and the close under the lower band -> buy; slope < 0 and over the upper band -> sell;
    exit at the regression line. Stop: none in the source - 2 ATR (ours). agents/indicators linreg_channel."""
    o, h, l, c, v = _ohlcv(df)
    lr = IND.linreg_channel(df)
    L = _onset((lr["slope"] > 0) & (c < lr["lower"]))
    S = _onset((lr["slope"] < 0) & (c > lr["upper"]))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": c >= lr["mid"], "exit_short": c <= lr["mid"]}


# ---------------- SC323 Renko trend reversal, StdDev bricks (StockSharp 1608) ----------------
def s_renko_reversal(df):
    """Renko Trend Reversal - StockSharp example 1608 (https://doc.stocksharp.com/en/api-examples/1608_Renko_Trend_Reversal ),
    4-hour: bricks of 0.5 x StdDev(14) on closes; the first up brick after a down trend -> buy, the first down brick
    after an up trend -> sell (reverse); stop 3% (the code's 0.97 / 1.03). agents/indicators renko_trend (bricks only
    on closed bars)."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.renko_trend(df)
    L, S = _excl(r["rev_up"], r["rev_dn"])
    return L, S, {"stop": ("level", c * 0.97, c * 1.03), "target": None, "exit_long": r["rev_dn"],
                  "exit_short": r["rev_up"]}


# ---------------- SC324 Woodies CCI zero-line reject in the CCI trend (Woodies CCI club) ----------------
def s_woodies_zlr(df):
    """Woodies CCI zero-line reject (https://www.luxalgo.com/library/concept/woodies-cci-conventions/ ; StockSharp 2263
    uses CCI 6 / 14): the CCI(14) trend = 6 bars on one side of zero; in an up trend the CCI pulls back toward zero
    (-50..+100) and turns up -> buy; mirror. Stop 0.1 ATR beyond the 5-bar extreme, target 2R (ours).
    agents/indicators woodies_cci."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.woodies_cci(df)
    a = atr(h, l, c, 14)
    sl, ss = _swing_stops(h, l, a, 5)
    L, S = _excl(w["zlr_long"], w["zlr_short"])
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC325 Power-hour breakout of the day's high / low (edgeful) ----------------
def s_power_hour_breakout(df):
    """Power hour breakout (edgeful, https://docs.edgeful.com/trade-using-edgeful/price-action-reports/power-hour-breakout ):
    a new high (low) of the day made between 15:00 and 16:00 New York -> trade in the breakout's direction (US index /
    stock session 09:30-16:00). Ours: the first close over the 09:30-15:00 high in 15:00-15:50 -> buy, under the low
    -> sell; stop 1 ATR; out at the 16:00 close; target 2R."""
    o, h, l, c, v = _ohlcv(df)
    hs, he, _, _, day = _local(df, "America/New_York")
    dk = pd.Series(day)
    m = (hs >= 9.5) & (he <= 15.0)
    hi = pd.Series(np.where(m, h, np.nan)).groupby(dk).cummax().groupby(dk).ffill().to_numpy()
    lo = pd.Series(np.where(m, l, np.nan)).groupby(dk).cummin().groupby(dk).ffill().to_numpy()
    w = (hs >= 15.0) & (hs < 15.0 + 50 / 60)
    L = _first_per_day(w & (c > hi), day)
    S = _first_per_day(w & (c < lo), day)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.0), "target": ("R", 2.0), "force_exit": he >= 16.0}


# ---------------- SC326 Smart Money Index divergence (Don Hays) ----------------
def s_smart_money_divergence(df):
    """Smart Money Index (Don Hays; https://www.luxalgo.com/library/concept/smart-money-index/ ): SMI = yesterday's SMI
    - the first 30 minutes' change + the last hour's change (US session); 'smart money' divergence: the index falls
    while the SMI rises -> bullish (mirror bearish). Ours (agents/indicators smart_money_index, completed days only):
    on the first bar after 10:00 NY, the previous close is under the close 5 sessions earlier while the SMI is over its
    value 5 sessions earlier -> buy; mirror sell; out at the 16:00 close; stop 1.5 ATR."""
    o, h, l, c, v = _ohlcv(df)
    hs, he, _, _, day = _local(df, "America/New_York")
    sm = IND.smart_money_index(df)
    sd = sm["smi_day"]
    dc = pd.Series(c).groupby(pd.Series(day)).last()
    dd = pd.Series(day)
    keys = np.array(list(dc.index), dtype="datetime64[D]")
    bars = dd.to_numpy().astype("datetime64[D]")
    pos = np.searchsorted(keys, bars, side="left") - 1           # yesterday's index in the daily closes
    dcv = dc.to_numpy(float)
    pc = np.where(pos >= 0, dcv[np.clip(pos, 0, None)], np.nan)
    pc5 = np.where(pos >= 5, dcv[np.clip(pos - 5, 0, None)], np.nan)
    sk = np.array(list(sd.index), dtype="datetime64[D]")
    sv = sd.to_numpy(float)
    sp = np.searchsorted(sk, bars, side="left") - 1
    s0 = np.where(sp >= 0, sv[np.clip(sp, 0, None)] if len(sv) else np.nan, np.nan)
    s5 = np.where(sp >= 5, sv[np.clip(sp - 5, 0, None)] if len(sv) else np.nan, np.nan)
    w = (hs >= 10.0) & (he <= 16.0)
    L = _first_per_day(w & (pc < pc5) & (s0 > s5), day)
    S = _first_per_day(w & (pc > pc5) & (s0 < s5), day)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": None, "force_exit": he >= 16.0}


# ---------------- SC327 Woodie pivots: pivot bounce in the open's direction, R1 / S1 targets (DailyFX) ----------------
def s_woodie_pivot_bounce(df):
    """Woodie's pivots (https://www.luxalgo.com/library/concept/woodie-pivots/ ; DailyFX 'Trading with Woodie's pivots'):
    PP = (prior H + L + 2 x today's open) / 4; above the PP the bias is long - buy pullbacks to the PP with R1 as the
    target, stop under S1; below it short to S1. Ours: the day opened over the PP, the bar's low touches the PP and the
    close is back over it (bullish) -> buy; stop 0.1 ATR under S1; target R1; first per day; mirror. UTC days."""
    o, h, l, c, v = _ohlcv(df)
    wp = IND.woodie_pivots(df)
    a = atr(h, l, c, 14)
    from agents.indicators.classic2 import _daily_hlc
    _, _, _, O = _daily_hlc(df)
    day = pd.Series(df.index.date).to_numpy()
    L = (O > wp["pp"]) & (l <= wp["pp"]) & (c > wp["pp"]) & (c > o) & (c < wp["r1"])
    S = (O < wp["pp"]) & (h >= wp["pp"]) & (c < wp["pp"]) & (c < o) & (c > wp["s1"])
    L, S = _first_per_day(L, day), _first_per_day(S, day)
    L, S = _excl(L, S)
    return L, S, _plan_lv(wp["s1"] - 0.1 * a, wp["r1"] + 0.1 * a, wp["r1"], wp["s1"])


# ---------------- SC328 OBV breakout ahead of price (Granville; StockCharts) ----------------
def s_obv_breakout(df):
    """On-Balance Volume (Joe Granville; StockCharts on-balance-volume: OBV moves before price - an OBV breakout while
    price has not broken out is bullish). Ours: OBV (tick volume) over its prior 20-bar high, the close NOT at a new
    20-bar high and over the EMA50 -> buy; mirror; stop 2 ATR, 2R. agents/indicators obv."""
    o, h, l, c, v = _ohlcv(df)
    ob = IND.obv(df)
    e = ema(c, 50)
    ph, pl = _sh(_s(h).rolling(20).max().to_numpy()), _sh(_s(l).rolling(20).min().to_numpy())
    L = _onset(ob["obv"] > ob["hi20"]) & (c < ph) & (c > e)
    S = _onset(ob["obv"] < ob["lo20"]) & (c > pl) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC329 Ease of Movement zero cross with the SMA50 trend (Arms; StockCharts) ----------------
def s_eom_zero_cross(df):
    """Ease of Movement (Richard Arms; StockCharts ease-of-movement-emv: EMV over zero = price rises easily) - EMV(14,
    tick volume) crosses over zero while the close is over the SMA50 -> buy; mirror; out on the cross back; stop 2
    ATR (ours). agents/indicators eom."""
    o, h, l, c, v = _ohlcv(df)
    e = IND.eom(df)
    m = sma(c, 50)
    L = cross_up(e, 0.0) & (c > m)
    S = cross_down(e, 0.0) & (c < m)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": e < 0, "exit_short": e > 0}


# ---------------- SC330 EMA 20 / 50 cross only in calm markets: Ulcer Index filter (Martin; StockCharts) ----------------
def s_ulcer_calm_cross(df):
    """Ulcer Index (Peter Martin; StockCharts ulcer-index: depth and duration of drawdowns) as a filter (ours): the EMA
    20 / 50 cross is traded only while the Ulcer Index(14) is under its 100-bar median (shallow drawdowns = a calm
    trend); stop 2 ATR, 2R. agents/indicators ulcer_index."""
    o, h, l, c, v = _ohlcv(df)
    u = IND.ulcer_index(df)
    calm = u <= _s(u).rolling(100, min_periods=20).median().to_numpy()
    e20, e50 = ema(c, 20), ema(c, 50)
    L, S = _excl(cross_up(e20, e50) & calm, cross_down(e20, e50) & calm)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


SCOUT = {
    "SC001": ("RV EMA20 cross over the 5m EMA200, 0.5% stop, 1:2, 1-minute (TradingView)", s_rv_ema20_scalp, ["1Min"]),
    "SC002": ("Alligator + Fractals breakout, fractal stop, 1:3 (RoboForex)", s_alligator_fractal, ["1Hour", "4Hour"]),
    "SC003": ("Pavan CPR: cross above top CPR, stop at pivot, long only (TradingView)", s_pavan_cpr, ["1Min"]),
    "SC004": ("NFP straddle of the 4-hour pre-release range, target = range (BrokerXplorer)", s_nfp_straddle, ["5Min"]),
    "SC005": ("NFP V-shaped reversal candle after the release, 1:3 (FOREX.com)", s_nfp_v_reversal, ["5Min", "15Min"]),
    "SC006": ("Stochastic Hash 16/7/3 cross at 25/70, 3.18R (TradingView)", s_stoch_hash, ["4Hour"]),
    "SC007": ("Cowabunga: 4H trend + 15m EMA5/10, RSI9, stoch, MACD, 1R, out 16:00 NY (BabyPips)", s_cowabunga, ["15Min"]),
    "SC008": ("EMA 5/13 cross + VWAP + ADX>20, stop at the cross, 2R (TradingView)", s_ema513_vwap_adx, ["15Min"]),
    "SC009": ("Pivot liquidity sweep + momentum candle + EMA200, 1:2 (TradingView)", s_gold_pivot_sweep, ["15Min"]),
    "SC010": ("Toto SBL: EMA 50/100/200 aligned, swing sweep in killzones, 2R (FX Replay)", s_toto_sbl, ["15Min"]),
    "SC011": ("NY first 5m candle break, 1m FVG retest + engulfing, 3R (YouTube Casper SMC)", s_casper_orb_fvg, ["1Min"]),
    "SC012": ("CPI anchored-VWAP retest after the spike, confirmation candle, 1:2 (FXNX)", s_cpi_avwap, ["5Min", "15Min"]),
    "SC013": ("Doji sandwich continuation, stop at the doji, stochastic exit (TradingView)", s_doji_sandwich, ["4Hour"]),
    "SC014": ("VWAP cross + MACD flip, exit on the MACD cross back, 1-minute (FXOpen)", s_vwap_macd_1m, ["1Min"]),
    "SC015": ("ALMA 21 + Stochastic 21/1/3 cross of 20/80, 1-minute (FXOpen)", s_alma_stoch_1m, ["1Min"]),
    "SC016": ("RSI(4) 80/20 + Bollinger touch, target the opposite band, 1-minute (FXOpen)", s_rsi4_bb_1m, ["1Min"]),
    "SC017": ("freqtrade ScalpingSimple: RSI<30 over SMA5, 0.5% stop, 1% target (GitHub)", s_ft_scalping_simple, ["1Min"]),
    "SC018": ("freqtrade MeanReversionScalper: BB + RSI + volume spike, out at mid band (GitHub)", s_ft_mean_rev_scalper, ["5Min"]),
    "SC019": ("freqtrade RSI_BB: 1-SD lower band long, 3-SD upper band short (GitHub)", s_ft_rsi_bb, ["15Min"]),
    "SC020": ("DAX false breakout of the 20-bar low + SuperTrend 18/1.5, long (ProRealCode)", s_prc_dax_false_break, ["15Min"]),
    "SC021": ("Jump: 16-bar mid +/- 2 ATR stop entry with 60m/30m filters, 5-minute (ProRealCode)", s_prc_jump, ["5Min"]),
    "SC022": ("5 EMA high/low fade scalp, 8-pip stop, 5-pip target, London/NY open hour (BabyPips)", s_bp_5ema_hl_scalp, ["5Min"]),
    "SC023": ("Keltner scalp: SMA89 + MACD 5/34/5 turn + stochastic, 1-minute (BabyPips)", s_bp_keltner_scalp_1m, ["1Min"]),
    "SC024": ("EURUSD 5m TEMA21 vs SMA300 + BB width + CCI -110/160, CCI 210 exit (ProRealCode)", s_prc_eurusd_tema_cci, ["5Min"]),
    "SC025": ("Bollinger + RSI 30/70 + Stochastic 5/3/3 scalp, mid-band exit, 5-minute (Forex Tester)", s_fxt_bb_rsi_stoch, ["5Min"]),
    "SC026": ("Admiral Keltner 50/10/2.75 re-entry + Stochastic 14/7/3, target the middle, M15 (Admirals)", s_admiral_keltner_m15, ["15Min"]),
    "SC027": ("Meta Scalper: Bollinger touch and retrace, half-width stop, 5% width target (forexop)", s_forexop_range_scalper, ["1Min", "5Min"]),
    "SC028": ("DAX first 5-minute bar direction at 09:05, 5/10 points, flat 17:30 (ProRealCode)", s_prc_dax_first_bar, ["5Min"]),
    "SC029": ("Hull Suite 55 + DI cross, swing stop, 1:1, 1-minute (Axiory)", s_axiory_hull_di_1m, ["1Min"]),
    "SC030": ("Alpha S/R: engulfing inside the 15m EMA36 channel, channel stop, 1:3, gold 5m (TradingView)", s_alpha_sr_channel, ["5Min"]),
    "SC031": ("freqtrade EMA_RSIStrategy: 1h trend + EMA20/50 + RSI 40-70 + ADX, -7%/4.5% (GitHub)", s_ft_ema_rsi, ["5Min"]),
    "SC032": ("London 08:15 GMT side of the central pivot, stop past the pivot, target R2/S2 (TradingView)", s_tv_london_pivot_15m, ["15Min"]),
    "SC033": ("Crypto 1H: EMA50 trend + EMA21 touch, 5-bar stop, 24-bar target, min 2R (TradingView)", s_tv_crypto_1h_strict_rr, ["1Hour"]),
    "SC034": ("ATR Trend + RSI<40 pullback over a rising EMA200, 1.5/3 ATR, long (TradingView)", s_tv_atr_trend_rsi_pullback, ["15Min", "1Hour"]),
    "SC035": ("EMA 50/100 cross + pullback + Stochastic 20/80, swing stop, 10-pip target, 1-minute (The5ers)", s_5ers_ema50_100_stoch_1m, ["1Min"]),
    "SC036": ("Gold 5m: 4H EMA200 trend, EMA50 pullback + engulfing/hammer + RSI 40, 2R, 13-17 GMT (XS)", s_xs_gold_5m_pullback, ["5Min"]),
    "SC037": ("Butterfly harmonic X-A-B-C-D, sell/buy at the 1.27 XA extension, target A (Trader Dhana, Facebook)", s_butterfly_target_a, ["15Min"]),
    "SC038": ("Butterfly harmonic, target 0.618 of AD (Trader Dhana, Facebook; user's variant)", s_butterfly_target_618, ["15Min"]),
    "SC039": ("Trendline break + Fibonacci 0.5-0.618 entry, stop under 0.618, impulse-length target (Trade Master, Facebook)", s_tl_break_fib, ["15Min", "1Hour", "4Hour"]),
    "SC040": ("Trendline break + Fibonacci 0.5-0.618, stop under 0.786, target the impulse high (Trade Master; user's variant)", s_tl_break_fib_786, ["15Min", "1Hour", "4Hour"]),
    "SC041": ("Fluid Money: Asian-session high/low break in its last quarter, 0.6/1.0 ATR, 15m (TradingView)", s_fluid_asian_late_break, ["15Min"]),
    "SC042": ("EMA 9/21 cross + RSI 50-70 filter, swing stop, 2R (TradingView)", s_ema921_rsi_swing, ["5Min", "15Min"]),
    "SC043": ("15MDbleBol: BB 20 + BB 25 poke, reversal candle back inside, target the middle band (BabyPips)", s_double_bb_reversal, ["15Min", "1Hour", "4Hour"]),
    "SC044": ("RSI 30/70 re-cross with a reversal bar, half at 0.5R + break-even, RSI exit (Tradingpedia)", s_tp_rsi_range, ["4Hour"]),
    "SC045": ("RSI(3) + Full Stochastic 6/3/3 cross over/under SMA150, exit at the opposite extreme (Tradingpedia)", s_tp_rsi3_stoch_sma150, ["1Hour"]),
    "SC046": ("Supertrend 10/3 + Stochastic RSI cross from 20/80, stop at the line, 2R (The Right Trader)", s_trt_st_stochrsi, ["15Min"]),
    "SC047": ("1-minute VWAP + EMA 9/21 pullback-bar break, 1.5R or VWAP exit (The Right Trader)", s_trt_vwap_ema_1m, ["1Min"]),
    "SC048": ("SuperTrade ST1: Supertrend flip above EMA200, 1 ATR stop, 4 ATR target, long (TradingView)", s_supertrade_st1, ["1Hour", "4Hour"]),
    "SC049": ("ETH Signal: Supertrend 12/2.76 + RSI 12, 4 ATR stop, 2 / 2.237 ATR target, 15m (TradingView)", s_eth_st_rsi_15m, ["15Min"]),
    "SC050": ("EMA50 + MACD histogram + RSI 50-70 / 30-50, 1 ATR stop, 2 ATR target, 15m (FMZ)", s_fmz_ema50_macd_rsi, ["15Min"]),
    "SC051": ("Engulfing / hammer + same-bar MACD cross, 1.5 ATR stop, 2 ATR target, 15m (FMZ)", s_fmz_candle_macd, ["15Min"]),
    "SC052": ("Three-Step 9:30 range: 5m-candle range, 1m FVG break, first-close stop, 2R (TradingView)", s_three_step_930_break, ["1Min"]),
    "SC053": ("Macketings 1m: EMA 80/90 band over 340/500 + SMA325, 0.2% stop, 2.5% target (TradingView)", s_macketings_1m, ["1Min"]),
    "SC054": ("CCI + EMA10 crossing EMA21/50, 12-point stop, 2.5R, M5+ (RoboForex)", s_robo_cci_ema, ["5Min", "15Min", "1Hour"]),
    "SC055": ("EMA21 + CCI20 zero agreement, swing stop, exit on the cross back, 15m (The Right Trader)", s_trt_ema21_cci, ["15Min"]),
    "SC056": ("CCI12 > 80 & RSI15 > 75 two bars (CCI < -95 & RSI < 30), 125/60 pips, 1H (StockSharp)", s_ss_cci_rsi, ["1Hour"]),
    "SC057": ("M15 EMA20/50 pullback + RSI 50 + engulfing / pin bar, swing stop, 2R, London-NY (XS)", s_xs_m15_ema_pullback, ["15Min"]),
    "SC058": ("NY 09:30-10:00 range breakout, half at 2R + break-even, rest 3R, one a day (MQL5 blog)", s_mql5_930_range_2r3r, ["1Min", "5Min", "15Min"]),
    "SC059": ("Bollinger middle + MACD signal state, 3-bar stop, exit when either flips, 5m (The Right Trader)", s_trt_bb_macd_5m, ["5Min"]),
    "SC060": ("BTC 15m: rising SMA200, SMA5 crosses the BB 21 midline, fractal stop, out under the midline (TradingView)", s_tv_btc15_sma5_mid, ["15Min"]),
    "SC061": ("Gold 1h: under the BB(15) top and over the 53-bar high, 2.9 / 5.8 ATR(9), long (ProRealCode)", s_prc_gold_bb_53, ["1Hour"]),
    "SC062": ("XAU 55-MA high/low channel break + Heikin-Ashi colour, previous-candle stop, 1.5R (TradingView)", s_xau_ma55_ha, ["1Min", "5Min"]),
    "SC063": ("ES 5-minute ORB, stop the other edge, target half the range, range <= 0.55%, no Tuesdays (Edgeful)", s_edgeful_orb5_half, ["5Min"]),
    "SC064": ("Double Bollinger 20/1 + 50/1 with RSI 50, entry-candle stop, out back under the fast band (TradingView)", s_tv_double_bollinger, ["1Min", "5Min"]),
    "SC065": ("EMA 5/20 cross near the lower/middle band + MACD histogram, 7-pip support stop, 5m (FXOpen)", s_fxo_ema520_bb_macd, ["5Min"]),
    "SC066": ("EMA 10/21/50: pullback to the 10/21 midpoint, 5-pip stop, 10-pip target, London-NY, 5m (IndicatorVault)", s_iv_ema1021_mid_5m, ["5Min"]),
    "SC067": ("SMA50 angle + ADX13 > 25 + 21-EMA pullback, 6-pip stop, break-even, 12.5-pip target, 5m (BabyPips)", s_bp_5m_sma_angle, ["5Min"]),
    "SC068": ("5m Bollinger squeeze breakout with VWAP slope, middle-band stop, 2R, 60-minute exit (DayTradingToolkit)", s_dtt_bb_squeeze_vwap, ["5Min"]),
    "SC069": ("Gold: Asian 00-07 GMT range broken 12-16 GMT, stop the other end, 1.5x range target (NYC Servers)", s_nycs_gold_asian_ny, ["15Min"]),
    "SC070": ("Keltner 20/2 breakout held, first 20-EMA touch, swing/1 ATR stop, 2 ATR target, long (JournalPlus)", s_jp_keltner_pullback, ["5Min", "15Min"]),
    "SC071": ("Keltner lower band + hammer/engulfing on light volume, 0.25 ATR wick stop, to the EMA20 (JournalPlus)", s_jp_keltner_reversal, ["5Min", "15Min"]),
    "SC072": ("Oil: first test of yesterday's high/low + 3-candle gap, formation stop, opposite level target (Purple Trading)", s_purple_oil_level_gap, ["5Min", "15Min"]),
    "SC073": ("First 5-minute bar breakout, 0.15-0.8% bar, stop the other end, target the bar height (The Right Trader)", s_trt_first_bar_orb, ["5Min"]),
    "SC074": ("BB 20/2 inside KC 20/2 release past both middles, squeeze-range stop, out under a middle (The Right Trader)", s_trt_bb_kc_release, ["15Min"]),
    "SC075": ("CL initial balance 09-10 ET breakout + retest, swing stop, target IB high + IB range (NexusFi)", s_nfi_cl_ib_retest, ["5Min"]),
    "SC076": ("EIA Wednesday fade back into the 09:00-10:15 ET range, spike stop, range-mid target (NexusFi)", s_nfi_eia_fade, ["5Min"]),
    "SC077": ("15-minute ORB without chasing, stop just inside the broken edge, OR-width target (NexusFi)", s_nfi_orb15_tight, ["1Min", "5Min"]),
    "SC078": ("NQ Keltner Pulse: KC 20/2.6 close, stop entry past yesterday's extreme, 2.5 ATR / 1.5%, M15 (MQL5)", s_mq_nq_keltner_pulse, ["15Min"]),
    "SC079": ("Range Filter + RSI 50 + ADX 25, 1 ATR stop, 3R, out under ADX 25, 5m (TradingView)", s_tv_range_filter_rsi_adx, ["5Min"]),
    "SC080": ("BVK: EMA 9/21 cross + MACD cross within 5 bars + RSI 50-70, swing stop, 1.5R, 5m (TradingView)", s_tv_bvk_ema_macd_rsi, ["5Min"]),
    "SC081": ("EMA 9/21 cross over EMA50 + RSI 40-70 / 30-60, 0.5% stop, 1% target, 15m (FMZ)", s_fmz_ema9_21_50_rsi, ["15Min"]),
    "SC082": ("Close over the SMA21 of highs + VWAP + EMA50 + RSI 60, SMA21-low stop, 1.5R, 5m (FMZ)", s_fmz_sma21hl_vwap_rsi, ["5Min"]),
    "SC083": ("EMA 9/21 cross + RSI 50, 0.5% / 1%, 09:15-15:30 IST, 1H (FMZ)", s_fmz_ema921_rsi50_ist, ["1Hour"]),
    "SC084": ("RSI < 40 while EMA9 > EMA21 (RSI > 60 under), 0.5% / 1%, 15m (FMZ)", s_fmz_rsi40_ema_trend, ["15Min"]),
    "SC085": ("Perfect order EMA45/SMA90/SMA180 + Trend Magic flip, SMA90 stop, 1.5R, 5m (FMZ)", s_fmz_perfect_order_trendmagic, ["5Min"]),
    "SC086": ("Exp TrendMagic CCI 50 median + ATR 5 colour switch, 3.5 / 7 ATR, 4H (StockSharp)", s_ss_exp_trendmagic, ["4Hour"]),
    "SC087": ("OneHrStocTrader: BB width band + Stochastic 5/3/5 turn at hours 4 / 0, 95 / 200 pips, H1 (StockSharp)", s_ss_one_hr_stoc, ["1Hour"]),
    "SC088": ("Williams AO 11/40 zero cross + AC 3 bars + RSI 20, BB width filter, 60 / 90 pips, H1 (StockSharp)", s_ss_williams_ao_ac, ["1Hour"]),
    "SC089": ("EMA 12/26 cross over EMA55 + RSI 35-65 + MACD rising, 0.5% / 1%, 5m (StockSharp Designer)", s_ss_designer_ema_rsi_macd, ["5Min"]),
    "SC090": ("First EMA50 pullback after the EMA50/SMA200 cross, swing-break entry, fib 161.8 target, H1 (Forex Factory)", s_ff_first_pullback_cross, ["1Hour"]),
    "SC091": ("Heiken Ashi 2 candles + ADX 22 / DI + Stochastic 5/3/3 direction, 12.5 / 12.5 pips, M5 (RoboForex)", s_robo_ha_adx_stoch, ["5Min"]),
    "SC092": ("Donchian 20 upper-band crawl, buy the middle-band pullback, lower-band stop, out end of day, 15m (TradingView)", s_tsg_donchian_mid_pullback, ["15Min"]),
    "SC093": ("LV-HP: daily candle colour + Donchian 20 touch, opposite band target, 1:1, 16-20 GMT, M1 (ForTraders)", s_ft_lvhp_donchian_m1, ["1Min"]),
    "SC094": ("MTF Ichimoku: over the cloud and the Kijun on 4H and 1H, 2% / 4%, 12-bar cooldown (TradingView)", s_tv_mtf_ichimoku, ["1Hour"]),
    "SC095": ("Trader DNA 5-8-13 EMA cross + Parabolic SAR, swing stop, 3R, 1H (YouTube via Medium)", s_tdna_ema5813_psar, ["1Hour"]),
    "SC096": ("Trade Pro 50 EMA wick rejection over the 200 EMA, 1 ATR stop, 1.5R, 15m (YouTube via Medium)", s_tp_ema50_wick_reject, ["15Min"]),
    "SC097": ("Keltner band reversal + EMA 9/21 cross over EMA50, 1.5 / 2 ATR (TradingView)", s_tv_keltner_ema_dual, ["15Min", "1Hour", "4Hour"]),
    "SC098": ("Williams %R Zone Scalper: -80 / -20 cross + EMA20 + Choppiness < 38.2 + volume, 1.5 / 2 ATR, 5m (TradingView)", s_tv_wr_zone_scalper, ["5Min"]),
    "SC099": ("Dynamic R:R BTC: EMA 20/50 cross + EMA200 + RSI 60/40, swing stop, 2R, 15m (TradingView)", s_tv_dynamic_rr_btc, ["15Min"]),
    "SC100": ("GoldScalper Pro v8: EMA 8/21/55 + MACD cross + RSI 50-68 + ADX 20, 1 / 5 ATR, break-even, 5m (TradingView)", s_tv_goldscalper_v8, ["5Min", "15Min"]),
    "SC101": ("Happy Hunter: LHLL / HHHL bar, stop entry past Bar1, 1.3 / 1.9 ATR(120), 1H (BabyPips)", s_bp_happy_hunter, ["1Hour"]),
    "SC102": ("XAUUSD EMA50/200 trend + pullback to EMA50 with RSI < 30 turning, swing stop, 2R (blog)", s_pyro_xau_ema_rsi, ["1Min", "5Min"]),
    "SC103": ("Gold Time Precise Model: Asian range break in London's first 3 hours + big-candle patterns, 20 pips, 15m (EnergyExch)", s_ee_gold_asian_patterns, ["15Min"]),
    "SC104": ("US30 Parabolic SAR flip + close beyond the last opposite dot, EMA100, 25 / 50 pips, after 13:00 London, 1m (EnergyExch)", s_ee_us30_psar_ema100, ["1Min"]),
    "SC105": ("WHS Z-Score: re-cross of -2 / +2 with the 100 MA slope, target the 20 MA, 20-bar exit, 15m (WH SelfInvest)", s_whs_zscore_trend, ["15Min"]),
    "SC106": ("Pieper Histo Breakout: HVR < 0.5 + inside / narrow-range bar, close beyond within 4 bars, 2R, 1H (WH SelfInvest)", s_whs_histo_breakout, ["1Hour"]),
    "SC107": ("News 1m: impulse candle, entry after the 3-minute correction, stop past it, target the impulse extreme (TradingView)", s_tv_news_impulse_correction, ["1Min"]),
    "SC108": ("News fade: release spike + reversal candle, wick break, stop the spike, target the pre-release close, 15-min exit, 5m (JournalPlus)", s_jp_news_fade_reversal, ["5Min"]),
    "SC109": ("CPI 08:30-09:00 ET range breakout, stop at the midpoint, target the prior day's high / low (DayTradingToolkit)", s_dtt_cpi_range_break, ["5Min"]),
    "SC110": ("freqtrade Hacklemore: RMI + MACD + SAR + lower-BB touch 2-10 bars ago, -8.5% / 1.5%, 5m (GitHub)", s_ft_hacklemore, ["5Min"]),
    "SC111": ("Tricolor M1: EMA 63/102/165 stack + Stochastic 30/6/18 under 10 + 15/3/9 crossing 20, 5 / 10 pips (ForTraders)", s_fort_tricolor_m1, ["1Min"]),
    "SC112": ("ADX(14) crossing its MA9 + MACD 5/25/9 zero side, 30 / 60 pips, opposite-signal exit, M15 (ForTraders)", s_fort_adx_ma_macd, ["15Min"]),
    "SC113": ("10 points: MACD 13/26/9 histogram side + Stochastic 5/3/3 back over 20 / under 80, local-extreme stop, 10 pips, M1 (Engineer the Forex)", s_etf_10points_m1, ["1Min"]),
    "SC114": ("Ranger: Parabolic SAR 0.04/0.1 + Stochastic 5/5/5 cross under 25 / over 75, local stop <= 20 pips, 2.5R, M15 (Trading Guide)", s_tg_ranger_sar_stoch, ["15Min"]),
    "SC115": ("PinkFlint news straddle: +/-20 pips from the price 5 min before, stop 30, target 5 pips, M5 (ForTraders)", s_fort_pinkflint_news, ["5Min"]),
    "SC116": ("Stochastic 14/3/3 %K>%D (<=80) + MACD over signal, 3-bar stop, exit when either fails, 15m (The Right Trader)", s_trt_stoch_macd_state, ["15Min"]),
    "SC117": ("EMA ribbon 2-15 crossing ribbon 44-74, M1 12 / 6 pips, M5 18 / 16 pips (Engineer the Forex)", s_etf_ema_ribbon_scalp, ["1Min", "5Min"]),
    "SC118": ("RSI(7) back over 20 / under 80, entry-candle stop, out at RSI 50, 1-minute (FXOpen)", s_fxo_rsi7_extremes, ["1Min"]),
    "SC119": ("Stochastic 14/3/3 cross under 20 / over 80, swing stop, out at the opposite extreme, 1-minute (FXOpen)", s_fxo_stoch_quick, ["1Min"]),
    "SC120": ("5m EMA20 pullback + Stochastic RSI turn from 20 on the VWAP side, swing stop, 1.25R, 08-12 NY (XS)", s_xs_5m_ema20_stochrsi_vwap, ["5Min"]),
    "SC121": ("MA Crossover ADX: EMA39 rising 2 bars + ADX33 > 22 + DI, 40 / 90 pips, H1 (StockSharp)", s_ss_ema39_adx33, ["1Hour"]),
    "SC122": ("MA Price Cross: close crosses SMA160, 20 / 60 pips (1:3), 01-22 (StockSharp / JBlanked)", s_ss_ma160_price_cross, ["1Min", "1Hour"]),
    "SC123": ("Connors S&P: over MA20 + RSI(3) < 30, break of the prior high within 2 bars, prior-bar-range stop, BE at 1R (TradingMarkets)", s_connors_ma20_rsi3_intraday, ["5Min", "15Min"]),
    "SC124": ("STC 23/50/10 cross of 25/75 + EMA35/EMA200 trend, 1:1, opposite reverses (FMZ)", s_fmz_stc_ema35_200, ["4Hour"]),
    "SC125": ("STC(12,26,50) turn + UT ATR(5)x3.5 trail regime + SMA200, trail stop, 2 ATR(5) target (FMZ)", s_fmz_stc_ut_sma200, ["4Hour"]),
    "SC126": ("Kijun 20 cross bar closing in its top / bottom 25%, 1.5 ATR stop, 15 pips, Kijun exit, 15m (FMZ)", s_fmz_kijun20_wpr1, ["15Min"]),
    "SC127": ("Kijun 20 loopback: open under / close over the Kijun, 1.5 ATR stop, 10 pips, Kijun exit, 1H (FMZ)", s_fmz_kijun20_loopback, ["1Hour"]),
    "SC128": ("Kalman-smoothed Hull cross + WaveTrend 10/21 cross on the same bar, 5% / 10%, 1H (FMZ)", s_fmz_kalman_hull_wt, ["1Hour"]),
    "SC129": ("MFI(3) = 100 / 0 + momentum body over / under SMA80, 00:00-bar stop -3%, 2.2R, 5m (FMZ)", s_fmz_mfi3_5m, ["5Min"]),
    "SC130": ("Asian 19:00-01:00 UTC range fade from 04:00, one per session, 1:1, 1H (FMZ)", s_fmz_asian_range_fade, ["1Hour"]),
    "SC131": ("Sproule: price and previous bar on one side of EMA100, EMA trailing stop, 100 pips, 1H (BabyPips)", s_bp_sproule_ema100, ["1Hour"]),
    "SC132": ("EURUSD SMA5/SMA100 cross confirmed one candle later, 150 / 300 pips, 4H (BabyPips)", s_bp_sma5_100_150pips, ["4Hour"]),
    "SC133": ("Richbugger: 20 pips past the 4H EMA60(HL2) and over the 1H EMA60, 30-pip stop, 1H-EMA exit (BabyPips)", s_bp_richbugger, ["1Hour"]),
    "SC134": ("Centred 227-bar close extreme breakout, Donchian-20 stop, 5-pip trail, 30 pips, 10-23 CET, EURUSD 1H (ProRealCode)", s_prc_fractal113_breakout, ["1Hour"]),
    "SC135": ("DAX 3 up bars closing 12:00 long / 3 down bars 16:00 short, 30 / 45 points, flat 21:00, 1H (ProRealCode)", s_prc_dax_3bars_hour, ["1Hour"]),
    "SC136": ("20/200 pips: Open[2]-Open[7] > 7 pips at 18:00 GMT, 20-pip target, 200-pip stop, EURUSD H1 (MQL5)", s_mql5_20_200_pips, ["1Hour"]),
    "SC137": ("freqtrade BinHV45: BB40 lower-band capitulation bar, -5% / +1.25%, 1m, long (GitHub)", s_ft_binhv45, ["1Min"]),
    "SC138": ("DAX Reversal Catcher: BB 1.5 lower poke + higher bar + TEMA21>50 + VWAP cross, upper-band exit, 15:00 flat (ProRealCode)", s_prc_dax_reversal_catcher, ["15Min", "1Hour"]),
    "SC139": ("DAX trend: over SMA200 + MACD > 0 state from 09:00, 1% / 3%, flat 21:00, H4 (ProRealCode)", s_prc_dax_h4_sma200_macd, ["4Hour"]),
    "SC140": ("MACD(5,13,1) change 20:00 -> 04:00 >= 50 points, entry at the 08:00 open, 30 / 30+45 pips, BE, 4H (investing.co.uk)", s_inv_macd5131_overnight, ["4Hour"]),
    "SC141": ("SMA48 side + ATR(7) crossing its 30-bar average, 25 / 110 points, BE, USDJPY H4 (Forex Tester)", s_ftb_sma48_atr_cross, ["4Hour"]),
    "SC142": ("SSMA90 + WPR(11) < -85 + CCI(12) hook from under -100 + ATR filter, 12.5 / 50 points (Forex Tester)", s_ftb_ssma90_cci_wpr, ["5Min", "15Min", "1Hour"]),
    "SC143": ("RVI(10) signal cross + rising RSI(14) and StdDev(20), 35 / 100 points, BE at 30, H1 (Forex Tester)", s_ftb_rvi_rsi_std, ["1Hour", "4Hour"]),
    "SC144": ("Close beyond the Kumo and two Alligators + CCI50 > 50 + Stoch > 50, Alligator exit, yen pairs H1 (Forex Tester)", s_ftb_ichi_two_alligators, ["1Hour", "4Hour"]),
    "SC145": ("SMA233 high / low channel close beyond +15 points, stop at the channel, first ladder target +89 (Forex Tester)", s_ftb_sma233_channel, ["1Hour", "4Hour"]),
    "SC146": ("DAX 08:00-09:05 CET range breakout with the EMA50 bias, range stop, 2R, 5m (Admirals)", s_adm_dax_orb_0905_ema50, ["5Min"]),
    "SC147": ("Warrior pullback: 5+ green ramp, dip to EMA9 / VWAP, first new-high candle, pullback-low stop, HOD target, 5m (Warrior Trading)", s_wt_ramp_pullback_9ema, ["5Min"]),
    "SC148": ("EMA50 cross, stop 1 x ADR(14), target 2 x ADR (HowToTrade)", s_htt_ema50_adr, ["15Min", "1Hour"]),
    "SC149": ("EIA Wednesday 10:15-10:30 ET box breakout after 10:30, midpoint stop, 2R (DayTradingToolkit)", s_dtt_eia_box_break, ["1Min", "5Min"]),
    "SC150": ("Engulfing / pin bar breakout, stop entry past the signal candle, candle stop, 5R long / 4R short, 1H (TradingView)", s_tv_engulf_pin_breakout, ["1Hour"]),
    "SC151": ("1h liquidity swings: low reclaims the pivot-5 low over SMA50, 0.5% buffer stop, 2R, close-through exit (TradingView)", s_tv_liquidity_swings_1h, ["1Hour"]),
    "SC152": ("50 Pips A Day: 07:00 GMT hourly candle straddle, 15 / 50 pips, Mon-Fri (TradingView, Kaspricci)", s_tv_50pips_0700_straddle, ["1Hour"]),
    "SC153": ("Bollinger 20/2 close breakout + RSI 50, 2% / 4%, 4H (TradingView)", s_tv_bb_breakout_rsi50_4h, ["4Hour"]),
    "SC154": ("SMA10/30 cross + EMA5 filter, 1% / 2%, 5m (TradingView)", s_tv_sma10_30_ema5, ["5Min"]),
    "SC155": ("Nifty EMA 5/10 cross + Stochastic 14/3/3 over 20 / under 80, 5 / 10 points, stoch exit, 1m (TradingView)", s_tv_nifty_ema5_10_stoch, ["1Min"]),
    "SC156": ("Forex Master: EMA20(RSI10) up / EMA30(RSI30) down through 50, 50 / 50 pips, opposite exit, 1H (TradingView)", s_tv_forex_master_rsi_ema, ["1Hour"]),
    "SC157": ("EMA9 breakaway: low 5 pips off the EMA9 + close over the prior high, prior-candle stop, 2R, M15 (Forex Factory)", s_ff_ema9_breakaway, ["15Min"]),
    "SC158": ("MACD 12/26/9 cross + confirming candle, 125 / 30 pips, 6-bar exit, H1 (Forex Factory)", s_ff_macd_confirm_h1, ["1Hour"]),
    "SC159": ("Kathy Lien Pure Fade: 3-SD touch + RSI 30/70, close back in the 2-1 SD zone, swing stop, BE 1R, 2R, 1H (book)", s_kl_pure_fade_bb3, ["1Hour"]),
    "SC160": ("Steel Wolverine: Supertrend 10/3 + rising HMA20 pullback + ROC(9) zero cross on a pin / engulfing, 2.5R, London-NY opens (TSG)", s_tsg_steel_wolverine, ["5Min", "15Min", "1Hour", "4Hour"]),
    "SC161": ("Velvet Hammer: rising HMA20 touch + Stoch RSI 14/14/3/3 under 20 + engulfing, swing stop, 2R (TSG)", s_tsg_velvet_hammer, ["15Min", "1Hour", "4Hour"]),
    "SC162": ("50 pips a day: 07:00 GMT candle's direction, stop entry at its extreme, 7.5 pips past the other end, 50 pips, EOD exit (tradefx)", s_tfx_50pips_0700_direction, ["1Hour"]),
    "SC163": ("EMA 5/20 cross with all over / under EMA50, 10 / 20 pips, BE at 10, cross-back exit, 15m (TradingView idea)", s_tv_ema5_20_50_20pips, ["15Min"]),
    "SC164": ("PineMyths #2: EMA 5/21 + 50/200 + BB %B 0.75 / 0.25 + ADX 15 + Awesome Oscillator, 2% / 1%, 1m (Medium / YouTube)", s_pm_adx_pctb_ao_ema, ["1Min"]),
    "SC165": ("freqtrade MACD RSI EMA: histogram zero cross + RSI 50 + EMA5 > EMA20, -8% / ROI 3%, RSI 40 exit, 1h (GitHub)", s_ft_macd_rsi_ema_1h, ["1Hour"]),
    "SC166": ("E-News Lucky: 10:30 straddle +/-20 pips, 50 / 150 pips, flat 22:30, 1m (StockSharp)", s_ss_enews_lucky_1030, ["1Min", "5Min"]),
    "SC167": ("freqtrade Quickie: ADX > 30 + TEMA9 rising under the BB mid + under SMA200, -25% / ROI 3%, 100-min exit, 5m (GitHub)", s_ft_quickie, ["5Min"]),
    "SC168": ("freqtrade Simple: MACD > 0 and over signal + rising BB(12) top + RSI7 > 70, -25% / 1%, RSI 80 exit, 5m (GitHub)", s_ft_simple_macd_rsi7, ["5Min"]),
    "SC169": ("freqtrade AdxSmas: ADX > 25 + SMA3 crosses SMA6, -25% / 10%, ADX < 25 cross-back exit, 1h (GitHub)", s_ft_adx_smas, ["1Hour"]),
    "SC170": ("freqtrade BinHV27: EMA5(RSI5) <= 20/25 under EMA60/120 + -DI over its EMA25, four SMA120/240 regimes, signal exits, 5m (GitHub)", s_ft_binhv27, ["5Min"]),
    "SC171": ("freqtrade AverageStrategy: EMA8 crosses EMA21, -20% / 50%, cross-back exit, 4h (GitHub)", s_ft_average_ema8_21, ["4Hour"]),
    "SC172": ("freqtrade Strategy001: EMA20/50 cross + green Heikin-Ashi over EMA20, -10% / ROI 3%, 60-min exit, 5m (GitHub)", s_ft_strategy001, ["5Min"]),
    "SC173": ('Diamond top / bottom: close through the right-side line, stop the last inner swing, full-height target, 15m/30m (Trader Dhana, Facebook)', s_diamond_full, ['15Min', '1Hour']),
    "SC174": ("Diamond top / bottom, target 61.8% of the height (Trader Dhana; user's variant)", s_diamond_618, ['15Min', '1Hour']),
    "SC175": ("3-touch trend-line liquidity grab + FVG + BOS, entry back in the FVG, grab stop, 2R (TheTrader's Villa, Facebook)", s_villa_tl_sweep_fvg, ['5Min', '15Min', '1Hour']),
    "SC176": ("Under VWAP + RSI14 back over 30 + reversal candle, 5-bar swing stop, target the VWAP (The Trader's Guide, TikTok)", s_tg_vwap_rsi14_to_vwap, ['5Min', '15Min']),
    "SC177": ("Under VWAP + RSI14 back over 30 + reversal candle, 5-bar swing stop, 2R (The Trader's Guide; variant)", s_tg_vwap_rsi14_2r, ['5Min', '15Min']),
    "SC178": ('RSI range shift: up trend RSI dips to 40-45 and turns / down trend 55-60, swing stop, 2R (Trading Data, TikTok)', s_td_rsi_range_shift, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC179": ('RSI 60-40 setup A: EMA21>50 trend, RSI ~40 turn near EMA21 + confirm candle, EMA50 stop, last-high target, EMA21 trail (Market Structure, TikTok)', s_ms_rsi40_pullback, ['15Min', '1Hour', '4Hour']),
    "SC180": ('RSI 60-40 setup B: RSI > 60 + strong candle through the swing high, EMA21 stop, 2.5R, EMA21 trail (Market Structure, TikTok)', s_ms_rsi60_breakout, ['15Min', '1Hour', '4Hour']),
    "SC181": ("RSI14 divergence with the EMA200 trend, then RSI 50 cross, divergence-extreme stop, 2R (The Trader's Guide, TikTok)", s_tg_rsi_div_trend50, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC182": ("Close beyond both the VWAP and EMA20, exit on a close back beyond both, 1.5 ATR stop (The Trader's Guide, TikTok)", s_tg_vwap_ema_both, ['5Min', '15Min']),
    "SC183": ("EMA50>200 trend, 0.5-0.786 pullback, entry on the EMA9 reclaim, pullback-low stop, swing-high / 2R (The Trader's Guide, TikTok)", s_tg_fib_ema9_reclaim, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC184": ('Ichimoku 9/26/52: over the cloud + TK cross + Chikou over the price, Kijun stop, TK-cross exit (Quant Tactics, TikTok)', s_qt_ichimoku_chikou, ['1Hour', '4Hour', '1Day']),
    "SC185": ("EMA9/21 cross over VWAP + ADX > 20, stop entry over the confirm high, candle stop, 2R (The Trader's Guide 'ADX 80%', TikTok)", s_tg_ema9_21_vwap_adx_break, ['5Min', '15Min']),
    "SC186": ('Swing-low sweep and reclaim, sweep-low stop, target the previous swing high, gold 15m (sphula, TikTok)', s_sphula_sweep_reclaim, ['5Min', '15Min']),
    "SC187": ('EMA20/50 direction + buyer/seller pressure flip (tick volume), swing stop, 2R (ProTrippa, TikTok)', s_pt_ma_pressure, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC188": ('MFI14 back over 20 above EMA200 / under 80 below, swing stop, 2R (Quant Tactics, TikTok)', s_qt_mfi14_trend, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC189": ('Broken S/R level (2+ tests) retest + EMA50 slope + RSI 50 confirmation, level stop, next-level target (Flux Charts, TikTok)', s_flux_sr_retest_confirm, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC190": ('Support held 3 tests, break, retest and rejection, pre-break swing stop, next level / 2R (Lima Signal, TikTok)', s_lima_break_retest_3tests, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC191": ("EMA 9/21/50/100/200 graded trend: strong -> EMA50 pullback, normal -> EMA21, 2R (The Trader's Guide, TikTok)", s_tg_ema_stack_pullback, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC192": ('Falling trend-line touch short, stop 0.3 ATR over the line, 4R, US500 1m (Dan-Chi, TikTok)', s_danchi_tl_touch_far, ['1Min', '5Min']),
    "SC193": ('EMA20>50 + RSI > 50, touch of EMA20, exit on 20<50 or close < EMA50 (Market Structure, TikTok)', s_ms_ema20_50_rsi_pullback, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC194": ("Previous-session POC retest after acceptance, rejection candle, 0.5 ATR stop, 2R (Market Structure '8 retests', TikTok)", s_ms_poc_retest, ['5Min', '15Min', '1Hour']),
    "SC195": ('RSI divergence, then a structure break on expanding volume, divergence-low stop, 2R (Market Structure, TikTok)', s_ms_div_then_bos, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC196": ('Bollinger 20/2 poke back inside within the value area / POC / HVN, target the middle band (Flux Charts, TikTok)', s_flux_bb_vp, ['5Min', '15Min', '1Hour']),
    "SC197": ('Symmetric / ascending / descending triangle breakout, opposite-line stop, height target (16 chart patterns, TikTok)', s_triangle_breakout, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC198": ('Triple top / bottom neckline break, extreme stop, height target (16 chart patterns, TikTok)', s_triple_top_bottom, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC199": ('Bull / bear flag: 3-ATR pole, 5-20 bar flag under 50%, flag break, pole target (16 chart patterns, TikTok)', s_flag_breakout, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC200": ('Daily pivot bias: over P, first pullback to P, stop S1 / 1 ATR, target R1 (Ezekiel Chew, TikTok)', s_ec_daily_pivot_bias, ['5Min', '15Min', '1Hour']),
    "SC201": ("20-bar box breakout with >= 80% buyer share of the bar's tick volume, mid stop, box-height target (Flux Charts, TikTok)", s_flux_box_break_buy_pct, ['5Min', '15Min', '1Hour']),
    "SC202": ("20-bar box breakout on a 'big trade' (tick-volume z >= 2), mid stop, box-height target (Meks Trades idea)", s_meks_box_break_big_trade, ['5Min', '15Min', '1Hour']),
    "SC203": ('ICT OTE in the discount: 61.8-79% of the up leg + bullish candle, swing-low stop, swing-high target, gold (Forex Bull Signals, TikTok)', s_fbs_ote_discount, ['5Min', '15Min', '1Hour']),
    "SC204": ("First touch of the previous session's POC, reverse entry, 1 ATR stop, 2R (sixlots, TikTok)", s_sixlots_poc_first_touch, ['5Min', '15Min']),
    "SC205": ("Over EMA50 + DI cross + ADX > 20 rising, swing stop, 2R (The Trader's Guide, TikTok)", s_tg_adx_di_ema50, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC206": ('ICT Judas swing of the Asian range at London / NY, order-block re-entry, sweep stop, Asian-range target (Flux Charts, TikTok)', s_judas_ob, ['5Min', '15Min']),
    "SC207": ('Balanced (D) previous session: VAL / VAH rejection, edge stop, target the POC (Dualis Flow / saint, TikTok)', s_vp_d_edges_to_poc, ['5Min', '15Min', '1Hour']),
    "SC208": ('Balanced (D) TPO profile: VAL / VAH rejection, target the TPO POC (Meks Trades TPO idea)', s_tpo_d_edges_to_poc, ['5Min', '15Min', '1Hour']),
    "SC209": ("P / b previous session: pullback to the POC in the shape's direction, VA-edge stop, 2R (Dualis Flow, TikTok)", s_vp_pb_poc_pullback, ['5Min', '15Min', '1Hour']),
    "SC210": ('Close through a previous-session LVN, target the next HVN, 1 ATR stop (Olisa Flow, TikTok)', s_vp_lvn_to_hvn, ['5Min', '15Min', '1Hour']),
    "SC211": ('Double-distribution (B) session: close through the void, target the other HVN (saint / Dualis Flow; Delta Void idea)', s_vp_b_void_break, ['5Min', '15Min', '1Hour']),
    "SC212": ('Absorption bar (2x tick volume, small body, at an extreme) + confirmation close, bar stop, 2R (Orderflow Sniper, TikTok)', s_of_absorption_reversal, ['1Min', '5Min', '15Min', '1Hour']),
    "SC213": ('Equal highs / lows sweep and close back, sweep stop, opposite equal level / 2R (Orderflow Sniper, TikTok)', s_of_equal_level_sweep, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC214": ('Approximate CVD divergence at a swing + reversal close, swing stop, 2R (Orderflow Sniper, TikTok)', s_of_cvd_divergence, ['5Min', '15Min', '1Hour']),
    "SC215": ("EMA50 pullback in the EMA50/200 trend, target the previous week's high (>= 3R), BE at 1R (Ezekiel Chew, TikTok)", s_ec_far_structural_target, ['15Min', '1Hour', '4Hour']),
    "SC216": ("SuperTrend AI (k-means factor) flips, line stop, flip exit (LuxAlgo / Flux 'AI SuperTrend')", s_ind_supertrend_ai, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC217": ('ML Lorentzian Classification signal + kernel direction, 4-bar hold, 1.5 ATR stop (jdehorty)', s_ind_lorentzian, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC218": ('HMA50 trend turn, exit when the trend reaches its average length (ChartPrime Trend Duration / Trading Data)', s_ind_trend_duration_exit, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC219": ('Zero Lag Trend re-entry (close back over ZLEMA in the trend), band stop, 2R, flip exit (AlgoAlpha)', s_ind_zero_lag_reentry, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC220": ('Zero Lag Trend flips (close beyond ZLEMA +/- volatility), ZLEMA stop, flip exit (AlgoAlpha)', s_ind_zero_lag_flip, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC221": ('ATR Fibonacci envelope trend: golden-pocket pullback, trail-band stop, 2R (BigBeluga)', s_ind_atr_fib_pocket, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC222": ('Double Ribbon trend score sign change, 2 ATR stop, opposite exit (ChartPrime, our reconstruction)', s_ind_double_ribbon, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC223": ('Fibonacci Golden Wave: 0.5-0.618 touch of the 20-bar up leg, range-low stop, range-high target (Flux Charts)', s_ind_golden_wave, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC224": ('Swing sweep then CISD within 10 bars, sweep stop, 2R (ICT / LuxAlgo CISD)', s_ind_cisd_after_sweep, ['1Min', '5Min', '15Min', '1Hour']),
    "SC225": ('Impulse Zone retest (1.5 ATR + 1.5x volume bar), zone stop, 2R (Flux Charts)', s_ind_impulse_zone_retest, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC226": ('Funk RSI 30/70 swing reversal, swing-bar target, our excursion stop (Almazov Trade)', s_ind_funk_rsi, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC227": ('WaveTrend cross under -53 / over +53, swing stop, 2R (LazyBear)', s_ind_wavetrend_zone_cross, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC228": ('WaveTrend divergence + cross within 10 bars, swing stop, 2R (LazyBear)', s_ind_wavetrend_divergence, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC229": ("Nadaraya-Watson envelope (non-repaint) band-cross dots, 1.5 ATR stop, mid target (LuxAlgo; our 'Non Repaint')", s_ind_nwe_reversal, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC230": ('Pullback to the VWAP anchored at the last swing low over EMA50, 1 ATR stop, 2R', s_ind_avwap_swing_pullback, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC231": ('SMC: order-block retest with the structure trend, OB stop, 2R (LuxAlgo)', s_ind_smc_bos_ob, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC232": ('SMC: CHoCH then the first FVG return, swing stop, 2R (LuxAlgo)', s_ind_smc_choch_fvg, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC233": ('Breaker block retest (failed order block), breaker stop, 2R (Flux / LuxAlgo)', s_ind_breaker_retest, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC234": ('Supply / demand zone breakout (close through the zone), zone stop, 2R (Flux / zone matrix)', s_ind_sd_breakout, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC235": ('Break of a 3-touch auto trend line, swing stop, 2R (BigBeluga Pattern Detector idea)', s_ind_trendline_break, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC236": ("EMA9/20 cross only when BB 20/2 width >= its 20th percentile, swing stop, 2R (user's filters tal1411ema)", s_ind_ema9_20_bbw, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC237": ('QQE 14/5/4.238 cross of its trailing line, cross-back exit, 2 ATR stop (StockSharp)', s_qqe_cross, ['15Min', '1Hour', '4Hour']),
    "SC238": ('Laguerre RSI 0.7 crosses 30 / 70, exit at 50, 2% stop (StockSharp)', s_laguerre_rsi, ['5Min', '15Min']),
    "SC239": ('Fisher Transform 9 trigger cross (under / over 1), opposite exit, 2 ATR stop (FMZ)', s_fisher_cross, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC240": ('SSL channel flip + HMA60 baseline + WaveTrend cross, 1.5 ATR stop, 1R (FMZ)', s_ssl_hma_wavetrend, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC241": ('Gartley harmonic XABCD (0.618 / 0.786), stop beyond X, target C (The Right Trader)', s_gartley_harmonic, ['15Min', '1Hour', '4Hour']),
    "SC242": ('Gartley harmonic, target A (The Right Trader; second target)', s_gartley_target_a, ['15Min', '1Hour', '4Hour']),
    "SC243": ('Joe Ross Gimmee bar: BB touch + opposite close in a sideways market (ADX < 20), bar stop, middle-band target (LuxAlgo)', s_ross_gimmee_bar, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC244": ('Momentum Pinball: RSI3 of the daily change < 30 / > 70, first-hour breakout, first-hour stop, next-morning exit (Raschke / Connors)', s_momentum_pinball, ['15Min', '1Hour']),
    "SC245": ('Elder Triple Screen: 4x-TF MACD-hist slope + Force Index EMA2 pullback + prior-high break, 2R (forexop)', s_elder_triple_screen, ['15Min', '1Hour', '4Hour']),
    "SC246": ('Raschke Anti 3/10 oscillator pullback re-cross, pullback-low stop, cross-back exit (MQL5)', s_raschke_anti_310, ['1Hour', '4Hour']),
    "SC247": ('Larry Williams Smash Day: close under the prior low at an 8-bar low close, buy over its high, 3-bar exit (Rogue Quant)', s_lw_smash_day, ['4Hour', '1Day']),
    "SC248": ('Weinstein stage 1->2 (SMA150 slope, 10-bar highs) on 1.8x volume + RSI >= 50, 2 / 6 ATR, SMA exit (MQL5)', s_weinstein_stage2, ['1Day', '4Hour']),
    "SC249": ('Close 2.5x avg range under the 10-bar high twice + IBS < 0.2, long, out over the prior high (TradingView)', s_avg_hl_ibs, ['1Day', '4Hour']),
    "SC250": ('FVG with a weak third candle (body and wicks < 0.5 ATR), return entry, gap stop, 2R (thecrypteller, TikTok)', s_fvg_c3_weak, ['1Min', '5Min', '15Min', '1Hour']),
    "SC251": ('FVG with significant displacement (1.25x volume, body >= 0.7 ATR and 60%), return entry, 2R (SecondBell / edgeful)', s_fvg_displacement, ['5Min', '15Min', '1Hour']),
    "SC252": ('FVG made on >= 65% buyer (<= 35%) tick volume, return entry, gap stop, 2R (Flux / Volumatic idea)', s_fvg_buy_volume, ['1Min', '5Min', '15Min', '1Hour']),
    "SC253": ('ICT Silver Bullet 03-04 / 10-11 / 14-15 NY: sweep, window FVG return, sweep stop, opposing-pool target (LuxAlgo)', s_silver_bullet_3w, ['1Min', '5Min', '15Min']),
    "SC254": ("FVG overlapping the impulse's order block and 0.5-0.618, block stop, 1.272 extension (TradersNotes, TikTok)", s_fib_ob_fvg_1272, ['5Min', '15Min', '1Hour']),
    "SC255": ('FVG + order block + 0.5-0.618 confluence, target the 2.0 extension (TradersNotes, TikTok)', s_fib_ob_fvg_2, ['5Min', '15Min', '1Hour']),
    "SC256": ('CHoCH then BOS, golden zone 0.5-0.618 inside the order block, leg-start stop, leg-end target (Mr Albash, TikTok)', s_smc_fib_golden_edge, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC257": ('CHoCH then BOS, golden zone in the order block, 1.272 extension target (Mr Albash, TikTok)', s_smc_fib_golden_ext, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC258": ('Judas swing of the NY 09:30-09:45 range + CISD close, sweep stop, opposite edge / 1.5R (TradingFinder / the user)', s_judas_ny0930, ['1Min', '5Min']),
    "SC259": ('Judas swing of the 08:15-08:30 NY range + CISD close, forex timing (TradingFinder)', s_judas_fx0815, ['1Min', '5Min']),
    "SC260": ('After a CHoCH, a fake breakout into an FVG at 0.618-0.786 of the leg, bar stop, 1.272 extension (TradersNotes 4H)', s_choch_fakeout_fvg, ['1Hour', '4Hour']),
    "SC261": ("Release staircase: up to 3 retests of broken levels in the move's direction, common measured-move target (sphula, TikTok)", s_release_stair_retests, ['1Min', '5Min', '15Min']),
    "SC262": ('Pre-release straddle +/-1 ATR, OCO, stop at the other order, 2R, 30-minute exit (ExpertFx, TikTok)', s_release_atr_straddle, ['1Min', '5Min']),
    "SC263": ("London 08:00 / NY 09:30 straddle +/-0.5 ATR, OCO, other-order stop, 2R, 60-minute exit (user's tal1411strad)", s_session_open_atr_straddle, ['1Min', '5Min', '15Min']),
    "SC264": ('Tokyo 00-08 UTC range breakout near the prior-day extreme, 80% body, asymmetric R stops / targets, GBPUSD M15 (MQL5)', s_mql5_tokyo_range_pdh, ['15Min']),
    "SC265": ('Gold 01-06 UTC box breakout, 0.3-1.6% range filter, stop <= 0.6%, 2.5R, BE 1.5R, flat 19 UTC (MQL5 blog)', s_mql5_gold_london_box, ['5Min', '15Min']),
    "SC266": ('Larry Williams Greatest Swing Value: open +/- 1.8 x failure swing after a 5-bar setup, bar-extreme stop, 2R (MQL5)', s_lw_greatest_swing, ['1Hour', '4Hour', '1Day']),
    "SC267": ('Choppiness < 38.2 with close over / under SMA20, out when CHOP > 61.8, 2% stop (StockSharp)', s_ss_chop_breakout, ['5Min', '15Min', '1Hour']),
    "SC268": ('Kaufman Efficiency Ratio crosses +/-0.6, 10-bar extreme stop and trailing exit (WH SelfInvest)', s_whs_kaufman_er, ['15Min', '1Hour', '1Day']),
    "SC269": ('Ehlers MAMA / FAMA cross, always in, 2 ATR stop (LazyBear / FMZ)', s_mama_fama_cross, ['15Min', '1Hour', '4Hour']),
    "SC270": ('Ehlers Instantaneous Trendline trigger cross, 2 ATR stop, 1.5% target, reverse (blackcat1402)', s_ehlers_itrend_cross, ['15Min', '1Hour', '4Hour']),
    "SC271": ('CMF(20) zero cross with a rising EMA50, negative-run low stop, out back under zero, 2R (The Right Trader)', s_trt_cmf_zero_trend, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC272": ('UT Bot (3, ATR 10) signals with volume >= 1.2x, 2 ATR stop, reverse (TradeIQ)', s_tiq_ut_bot_volume, ['1Hour', '4Hour']),
    "SC273": ('TTM Squeeze Pro fire after high compression (KC 1.0) with momentum, fade exit, 2 ATR stop (Carter)', s_ttm_squeeze_pro_fire, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC274": ('Pocket Pivot near the 10-bar line over SMA50, 2.5 ATR stop, out under the 10-bar line (Kacher / Morales)', s_pocket_pivot, ['1Hour', '4Hour', '1Day']),
    "SC275": ('AI Trend Navigator [K-Neighbor] line turn with the classifier, reverse, 2 ATR stop (Zeiierman)', s_knn_trend_navigator, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC276": ('Gaussian Channel [DW] upper-band breakout while the filter rises, out back under the band (DonovanWall)', s_gaussian_channel_break, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC277": ("Kalman SuperTrend flips outside the VWMA OB / OS bands, line stop ('Pro Scalper' reconstruction)", s_kalman_supertrend, ['1Min', '5Min', '15Min']),
    "SC278": ('Trend Magnet flip outside low-momentum zones (ADX >= 20), line stop (BigBeluga reconstruction)', s_trend_magnet, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC279": ('Volume Mapping buy-sell volume zero cross with a rising EMA50, 1.5 ATR, 2R (BigBeluga reconstruction)', s_volume_mapping_trend, ['5Min', '15Min', '1Hour']),
    "SC280": ('Sniper V.02: EMA 9/21 cross with bull score > 60%, 1.5 / 3 ATR, EMA21 trail (KhanSaab reconstruction)', s_sniper_cross_score, ['1Min', '5Min', '15Min']),
    "SC281": ('Sniper V.02: first ribbon retest after the cross with score > 60%, 1.5 / 3 ATR, EMA21 trail', s_sniper_retest, ['1Min', '5Min', '15Min']),
    "SC282": ('VuManChu Cipher B green / red dot with money flow turning or a WT divergence, swing stop, 2R', s_cipher_b_dot_mf, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC283": ('VuManChu Swing Free (Range Filter 20 / 3.5) state changes, band stop, reverse', s_swing_free, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC284": ('HalfTrend signal engine: flip, channel stop, 3R, opposite exit (BigBeluga / Trading Data)', s_halftrend_engine, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC285": ('MSNR: first touch of a fresh support / resistance in the storyline, next fresh level target >= 1.5R (Malaysian SnR)', s_msnr_fresh_touch, ['15Min', '1Hour', '4Hour']),
    "SC286": ('Candle Range Theory: HTF candle 2 sweeps candle 1 and closes inside, target the far extreme (LuxAlgo)', s_crt_far_extreme, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC287": ("Candle Range Theory, target candle 1's 50% (EQ)", s_crt_eq, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC288": ('SMT divergence vs the partner (US100 / US500 ...) + structure shift, swing stop, 2R (simple_forex / ICT)', s_smt_divergence_mss, ['1Min', '5Min', '15Min', '1Hour']),
    "SC289": ("FVG return entry only in the Gaussian Channel's direction, gap stop, 2R (filter test for the gap method)", s_fvg_gaussian_trend, ['5Min', '15Min', '1Hour']),
    "SC290": ("CRT 4H live: sweep of the previous 4H candle, LTF structure shift, FVG return, target the range's other side (thecrypteller / wyckoff.hq)", s_crt_live_mss_fvg, ['5Min', '15Min']),
    "SC291": ('FVG return only on the stronger instrument of a correlated pair (displacement vs the partner, ER >= 0.3), 2R (ictrules)', s_fvg_strongest_of_pair, ['5Min', '15Min']),
    "SC292": ('SMT divergence -> iFVG -> CISD entry, SMT-swing stop, 2R (The Creative Trader)', s_smt_ifvg_cisd, ['1Min', '5Min', '15Min']),
    "SC293": ('SMT divergence + engulfing within 5 bars, SMT-swing stop, 2R (CY FOREX)', s_smt_engulf, ['5Min', '15Min', '1Hour']),
    "SC294": ('Inversion FVG retest (body close through a gap flips it), zone stop, 2R (LuxAlgo)', s_ifvg_retest, ['1Min', '5Min', '15Min', '1Hour']),
    "SC295": ('CHoCH, then the return to the broken level / OB with a confirmation candle, 2R (TradeMachine / LuxAlgo)', s_choch_confirmed_retest, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC296": ('CHoCH immediate entry on the break close, swing stop, 2R (comparison for SC295)', s_choch_immediate, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC297": ('Heikin Ashi swing: colour change + 2nd no-opposite-wick candle, exit on the opposite colour', s_ha_swing_confirm, ['1Hour', '4Hour', '1Day']),
    "SC298": ('External liquidity sweep -> target the internal FVG (IRL), sweep stop (Ali Khan)', s_erl_sweep_to_irl, ['5Min', '15Min', '1Hour']),
    "SC299": ('Internal FVG fill in the trend -> target the external swing (ERL), gap stop (Ali Khan)', s_irl_fill_to_erl, ['5Min', '15Min', '1Hour']),
    "SC300": ("First pullback to the last swing's POC (Swing Profile Analyzer), value-area stop, leg-extreme target (ChartPrime)", s_swing_poc_pullback, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC301": ('SK System: structure + 0.5-0.618 retracement entry, 0.786 stop, 1.414 extension target (TradersNotes)', s_sk_system, ['5Min', '15Min', '4Hour']),
    "SC302": ('Double top / bottom: entry at the second touch with a rejection candle, neckline target (comparison for TV57)', s_double_top_second_touch, ['15Min', '1Hour', '4Hour']),
    "SC303": ('London sweeps Asia + iFVG -> Asian-range target; NY sweeps London + iFVG -> London-range target (QuantFlowLabs)', s_london_ny_ifvg_reversal, ['1Min', '5Min']),
    "SC304": ("LazyBear Squeeze release in the histogram's colour, squeeze-range stop, 2R, fade exit", s_lb_squeeze_structural, ['15Min', '1Hour', '4Hour']),
    "SC305": ('Fibonacci EMA stack 8/13/21/34/55 in order, pullback to the 21-34, EMA55 stop, 2R (Trading Data)', s_fib_ema_stack_pullback, ['5Min', '15Min', '1Hour', '4Hour']),
    "SC306": ('ICT Unicorn: breaker + FVG overlap retest, breaker stop, opposing swing or 2R (LuxAlgo / Flux)', s_ict_unicorn, ['1Min', '5Min', '15Min']),
    "SC307": ('ICT Power of Three: manipulation under the NY midnight open and the Asian low, close back over the open, 2R, out 16:00 NY', s_po3_midnight_open, ['5Min', '15Min']),
    "SC308": ('ICT rejection block revisit (long-wick swing), wick stop, 2R (LuxAlgo concept)', s_rejection_block, ['5Min', '15Min', '1Hour']),
    "SC309": ('ICT NWOG / NDOG edge rejection, CE stop, 2R (LuxAlgo concept)', s_opening_gap_edge, ['5Min', '15Min', '1Hour']),
    "SC310": ('ICT Balanced Price Range retest, zone stop, 2R (UAlgo / TakingProphets)', s_bpr_retest, ['1Min', '5Min', '15Min']),
    "SC311": ('WH SelfInvest London Reversal: fade the Asian (London) extreme +/- 0.25 ATR in the first 2 hours, 5 ATR stop / target', s_whs_london_reversal, ['5Min', '15Min']),
    "SC312": ('Big Ben: fade the pre-London breakout of the Asian body range, range-size target, 60-minute time stop', s_big_ben_fade, ['5Min', '15Min']),
    "SC313": ('Coppock Curve turn under / over zero, 2 ATR stop, opposite-turn exit (StockCharts)', s_coppock_turn, ['4Hour', '1Day']),
    "SC314": ("Ultimate Oscillator divergence, Williams' 3 rules (30 / 70), exit at 70 / 30", s_ultimate_osc_divergence, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC315": ('Klinger Volume Oscillator signal cross on the right side of zero with the EMA100, 2 ATR, 2R', s_klinger_trend_cross, ['15Min', '1Hour', '4Hour']),
    "SC316": ('EMA 9/21 cross unless a Mass Index reversal bulge is pending; after a bulge trade the EMA9 turn (Dorsey)', s_mass_index_filter_cross, ['1Hour', '4Hour', '1Day']),
    "SC317": ('McGinley Dynamic +/-0.25% threshold cross, 10-bar cooldown, reverse (StockSharp 1038)', s_mcginley_threshold, ['15Min']),
    "SC318": ('Elder Ray: rising EMA13 with rising negative bear power (mirror), swing stop, 2R (Elder)', s_elder_ray, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC319": ('Wilder Volatility System: 3 x ATR(7) close SAR, stop and reverse (Wilder 1978)', s_wilder_volatility_system, ['1Hour', '4Hour', '1Day']),
    "SC320": ('FRAMA cross with its slope, 2 ATR stop, cross-back exit (Ehlers 2005)', s_frama_cross, ['15Min', '1Hour', '4Hour']),
    "SC321": ('Hurst < 0.7 mean reversion to the SMA20, 2% stop (StockSharp 0228)', s_hurst_reversion, ['5Min', '15Min']),
    "SC322": ('Linear regression channel 50 / 1.5 SE fade with the slope, exit at the line (StockSharp)', s_linreg_channel_fade, ['5Min', '15Min', '1Hour']),
    "SC323": ('Renko (0.5 x StdDev14) trend reversal bricks, 3% stop, reverse (StockSharp 1608)', s_renko_reversal, ['1Hour', '4Hour']),
    "SC324": ('Woodies CCI zero-line reject in the CCI trend, swing stop, 2R', s_woodies_zlr, ['5Min', '15Min', '1Hour']),
    "SC325": ('Power hour: new day high / low after 15:00 NY, 1 ATR stop, 2R, out at 16:00 (edgeful)', s_power_hour_breakout, ['5Min', '15Min']),
    "SC326": ('Smart Money Index divergence vs the 5-session price change, day trade (Don Hays)', s_smart_money_divergence, ['5Min', '15Min', '1Hour']),
    "SC327": ("Woodie pivots: PP bounce in the open's direction, S1 stop, R1 target (DailyFX)", s_woodie_pivot_bounce, ['5Min', '15Min', '1Hour']),
    "SC328": ('OBV breakout ahead of price over the EMA50, 2 ATR, 2R (Granville)', s_obv_breakout, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC329": ('Ease of Movement zero cross with the SMA50, 2 ATR, cross-back exit (Arms)', s_eom_zero_cross, ['15Min', '1Hour', '4Hour', '1Day']),
    "SC330": ('EMA 20/50 cross only while the Ulcer Index is under its 100-bar median, 2 ATR, 2R', s_ulcer_calm_cross, ['15Min', '1Hour', '4Hour']),
}

# Indicator builder (task tal1411indstrat): IB### strategies = direction from one library indicator + entry from
# another (logical pairings), and a 4H direction + an entry-chart trigger; agents/indicators/builder.py.
from agents.indicators.builder import BUILT as _IB  # noqa: E402

SCOUT.update({k: (v[0], v[1], v[2]) for k, v in _IB.items()})


# =====================================================================================================================
# Scout 15.11.2026 (night): 34 new indicators (agents/indicators/classic3.py, ehlers2.py, tv_popular.py) and the
# strategies their sources publish, plus at least one strategy per new indicator so every bot's agent tests it.
# =====================================================================================================================

def _swing_lv(l, h, n=10):
    """Swing stop levels: lowest low / highest high of the last n bars (incl. the signal bar)."""
    return (pd.Series(l).rolling(n, min_periods=1).min().to_numpy(),
            pd.Series(h).rolling(n, min_periods=1).max().to_numpy())


# ---------------- SC331 Coral Trend Pullback (TradeIQ) ----------------
def s_coral_pullback(df):
    """TradingView 'Coral Trend Pullback Strategy (TradeIQ)', 1H: Coral Trend (LazyBear, 21 / 0.4) bullish; since the
    last close-cross over the line at least one candle had its low above the line; then a pullback closes under the
    line while the line stays bullish; the next close back over the line = long (mirror short). Stop = the swing low of
    the last 10 bars ('Local High/Low Lookback', value ours), target 2R (the script's R:R input, value ours). No
    optional confirmation (the author reports better results with confirmations off)."""
    o, h, l, c, v = _ohlcv(df)
    cr = IND.coral(df)
    line, up = cr["coral"], cr["up"]
    dn = cr["coral"] < _sh(cr["coral"])
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    st_l = st_s = 0          # 0 none, 1 clean (low above line seen), 2 pulled back
    for t in range(1, n):
        if np.isnan(line[t]) or np.isnan(line[t - 1]):
            continue
        if c[t] > line[t] and c[t - 1] <= line[t - 1] and st_l != 2:
            st_l = 0
        if c[t] < line[t] and c[t - 1] >= line[t - 1] and st_s != 2:
            st_s = 0
        # long track
        if not up[t]:
            st_l = 0
        elif st_l == 0 and l[t] > line[t]:
            st_l = 1
        elif st_l == 1 and c[t] < line[t]:
            st_l = 2
        elif st_l == 2 and c[t] > line[t]:
            L[t], st_l = True, 1
        # short track
        if not dn[t]:
            st_s = 0
        elif st_s == 0 and h[t] < line[t]:
            st_s = 1
        elif st_s == 1 and c[t] > line[t]:
            st_s = 2
        elif st_s == 2 and c[t] < line[t]:
            S[t], st_s = True, 1
    L, S = _excl(L, S)
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


# ---------------- SC332 Heiken Ashi TMS / TDI (ProRealCode, Elsborgtrading) ----------------
def s_ha_tms_tdi(df):
    """ProRealCode 'Heiken Ashi TMS/TDI strategy' (V1, Brent 2H): TDI on RSI 13 - signal line mab = SMA2, trade line
    mbb = SMA7; yellow MA = SMA5 of the typical price shifted 2 bars; Heikin Ashi high / low. Long: mab crosses over
    mbb, mab < 50, HA high > yellow MA. Short: mab crosses under mbb, mab > 50, HA low < yellow MA. Exit long: mab
    crosses under mbb with 50 < mab < 68 and HA low > yellow; exit short: mab crosses over mbb with 32 < mab < 50.
    The source has no stop or target -> ours: 2 ATR protective stop, no target, the source's exits."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.tdi(df)
    mab, mbb = t["price"], t["signal"]
    ha = IND.heikin_ashi(df)
    hh = np.asarray(ha["h"] if "h" in ha else ha["ha_h"], float)
    hl = np.asarray(ha["l"] if "l" in ha else ha["ha_l"], float)
    yel = _sh(sma((h + l + c) / 3, 5), 2)
    cu, cd = cross_up(mab, mbb), cross_down(mab, mbb)
    L = cu & (mab < 50) & (hh > yel)
    S = cd & (mab > 50) & (hl < yel)
    xl = cd & (mab < 68) & (mab > 50) & (hl > yel)
    xs = cu & (mab > 32) & (mab < 50) & (hh > yel)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": xl, "exit_short": xs}


# ---------------- SC333 Chande Kroll Trend (StockSharp 0611) ----------------
def s_chande_kroll_trend(df):
    """StockSharp example 0611 'Chande Kroll Trend' (1H, long only): highStop = Highest(high, 21) - 3 ATR(10),
    lowStop = Lowest(low, 21) + 3 ATR(10). Long when the previous close was <= the previous lowStop, the close is now
    over the lowStop and over the SMA21; exit when the close falls under the highStop. The source has no fixed stop ->
    ours: 3 ATR protective stop."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 10)
    hs = pd.Series(h).rolling(21).max().to_numpy() - 3 * a
    ls = pd.Series(l).rolling(21).min().to_numpy() + 3 * a
    L = (_sh(c) <= _sh(ls)) & (c > ls) & (c > sma(c, 21))
    return L, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": None, "exit_long": c < hs}


# ---------------- SC334 Chande Kroll Stop breakout (Chande & Kroll classic) ----------------
def s_chande_kroll_break(df):
    """Chande Kroll Stop (p 10, x 1, q 9; 'The New Technical Trader'; LuxAlgo concept page): a close over the upper
    (short-stop) line = long, a close under the lower (long-stop) line = short; the stop is the opposite line, exit on
    the opposite signal (reverse)."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.chande_kroll(df)
    L, S = _excl(k["up"], k["dn"])
    return L, S, {"stop": ("level", k["stop_long"], k["stop_short"]), "target": None, "opposite": True}


# ---------------- SC335 Ehlers Sinewave stop-and-reverse (WH SelfInvest system 09) ----------------
def s_ehlers_sinewave_sar(df):
    """WH SelfInvest free system 09 'Sinewave Market Cycles' (Ehlers): the Sine crossing over its LeadSine = long,
    under = short; always in the market (stop and reverse). The source runs it on half-day bars of the Eurostoxx 50
    and has no stop -> ours: 3 ATR protective stop, no target, reverse on the opposite cross."""
    o, h, l, c, v = _ohlcv(df)
    s = IND.sinewave(df)
    L, S = _excl(s["up"], s["dn"])
    return L, S, {"stop": ("atr", 3.0), "target": None, "opposite": True}


# ---------------- SC336 Sinewave Grube variant: MACD 11/66 trend filter, 10 ATR target ----------------
def s_sinewave_grube(df):
    """WH SelfInvest system 09, Grube variant (15-minute Eurostoxx 50): Sinewave crosses only in the direction of an
    EMA 11 / 66 MACD (> 0 long, < 0 short; the source computes it on 89 quarter-hour bars, ours on the chart's bars);
    profit target 10 x ATR(280) as a level, as published; stop 3 ATR (the page gives no stop value)."""
    o, h, l, c, v = _ohlcv(df)
    s = IND.sinewave(df)
    m = ema(c, 11) - ema(c, 66)
    a280 = atr(h, l, c, 280)
    L, S = _excl(s["up"] & (m > 0), s["dn"] & (m < 0))
    return L, S, {"stop": ("atr", 3.0), "target": ("level", c + 10 * a280, c - 10 * a280)}


# ---------------- SC337 Even Better Sinewave reversal ----------------
def s_ebsw_reversal(df):
    """Ehlers Even Better Sinewave (Cycle Analytics for Traders ch. 12; HP 40, SS 10): the wave crossing back over
    -0.85 = cycle low -> long, back under +0.85 = cycle high -> short; only while the VHF(28) says congestion (< its
    100-bar median - cycle mode, Ehlers uses the wave in cycle mode only; ours). Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.ebsw(df)
    vh = IND.vhf(df)
    cyc = vh < pd.Series(vh).rolling(100, min_periods=20).median().to_numpy()
    L, S = _excl(w["up"] & cyc, w["dn"] & cyc)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- SC338 Waddah Attar Explosion + EMA200 scalp ----------------
def s_wae_ema200(df):
    """Waddah Attar Explosion (LazyBear 20 / 40 / 20 / 2 / 150, dead zone 3.7 x RMA(TR, 100)), as the LuxAlgo concept
    page uses it: the first bar whose green histogram is over both the explosion line and the dead zone gives long
    permission (red = short), traded in the EMA200 direction (the common YouTube 1-5 minute scalp filter). Exit when
    the histogram falls back under the explosion line (the page's exit aid). Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.waddah_attar(df)
    e = ema(c, 200)
    lg, sg = w["long_go"], w["short_go"]
    L = lg & ~(_sh(lg.astype(float)) > 0) & (c > e)
    S = sg & ~(_sh(sg.astype(float)) > 0) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0),
                  "exit_long": w["up_power"] < w["explosion"], "exit_short": w["dn_power"] < w["explosion"]}


# ---------------- SC339 TDI classic (Dean Malone) ----------------
def s_tdi_classic(df):
    """Traders Dynamic Index, Dean Malone's standard long rule: green price line crosses over the red signal line while
    both are over the yellow market base and over 50, and the green is under the upper band (not overbought) -> long;
    mirror short; exit when green crosses back through red. Stop 1.5 ATR, target 2R (ours; the TDI guide has no
    fixed numbers)."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.tdi(df)
    L = t["up"] & (t["price"] > t["base"]) & (t["price"] > 50) & (t["price"] < t["upper"])
    S = t["dn"] & (t["price"] < t["base"]) & (t["price"] < 50) & (t["price"] > t["lower"])
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": t["dn"], "exit_short": t["up"]}


# ---------------- SC340 TDI shark fin (band re-entry) ----------------
def s_tdi_shark_fin(df):
    """TDI 'shark fin' (Forex Factory TDI Shark Fin thread; Dean Malone): the green line pokes outside the volatility
    band and the next cross back inside the band forms the fin -> fade it (back under the upper band = short, back
    over the lower = long). Stop beyond the 10-bar swing, target the market base line (yellow) projected as 2R min;
    ours: stop swing, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.tdi(df)
    L = cross_up(t["price"], t["lower"])
    S = cross_down(t["price"], t["upper"])
    L, S = _excl(L, S)
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


# ---------------- SC341 PMax (KivancOzbilgic) ----------------
def s_pmax_flip(df):
    """Profit Maximizer strategy (KivancOzbilgic, TradingView 7yn5dteO): EMA10 of hl2 crossing over the PMax line
    (ATR 10 x 3 SuperTrend on the MA) = long, under = short; always in (stop and reverse); stop at the PMax line."""
    p = IND.pmax(df)
    L, S = _excl(p["up"], p["dn"])
    return L, S, {"stop": ("level", p["pmax"], p["pmax"]), "target": None, "opposite": True}


# ---------------- SC342 PMax on RSI with T3 ----------------
def s_pmax_rsi_t3(df):
    """'PMax on Rsi w/T3 Strategy' (KivancOzbilgic, TradingView ZLcmWQtj): PMax run on T3(8) of RSI(14) (our reading:
    the RSI line's own range as ATR, x 3); the T3 crossing over PMax = long, under = short; stop 2 ATR of price,
    reverse on the opposite cross."""
    p = IND.pmax(df, source="rsi_t3")
    L, S = _excl(p["up"], p["dn"])
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


# ---------------- SC343 MACD-V rebound from the risk zone ----------------
def s_macdv_rebound(df):
    """MACD-V (Spiroglou; StockCharts / LuxAlgo): signal-line crosses are traded only outside the +/-50 neutral band -
    a cross over the signal while MACD-V < -50 (oversold rebound) -> long, a cross under while > +50 -> short. Stop 2
    ATR, target 2R (ours)."""
    m = IND.macd_v(df)
    L = cross_up(m["macdv"], m["signal"]) & (m["macdv"] < -50)
    S = cross_down(m["macdv"], m["signal"]) & (m["macdv"] > 50)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC344 MACD-V momentum (rally zone) ----------------
def s_macdv_momentum(df):
    """MACD-V momentum: crossing up through +50 with the histogram positive and under +150 (Spiroglou's 'rallying'
    zone, not yet overextended) -> long; mirror short through -50; exit when MACD-V crosses back through its signal or
    passes +/-150 (risk zone). Stop 2 ATR."""
    m = IND.macd_v(df)
    mv, sg = m["macdv"], m["signal"]
    L = cross_up(mv, 50) & (m["hist"] > 0) & (mv < 150)
    S = cross_down(mv, -50) & (m["hist"] < 0) & (mv > -150)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None,
                  "exit_long": cross_down(mv, sg) | (mv > 150), "exit_short": cross_up(mv, sg) | (mv < -150)}


# ---------------- SC345 VFI zero cross ----------------
def s_vfi_zero(df):
    """Volume Flow Indicator (Katsanos; LazyBear 130 / 0.2 / 2.5): VFI crossing over zero with VFI over its EMA5 signal
    and the close over the EMA50 -> long; mirror short; exit when VFI crosses back through zero. Stop 2 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    f = IND.vfi(df)
    e = ema(c, 50)
    L = cross_up(f["vfi"], 0) & (f["vfi"] > f["signal"]) & (c > e)
    S = cross_down(f["vfi"], 0) & (f["vfi"] < f["signal"]) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0),
                  "exit_long": cross_down(f["vfi"], 0), "exit_short": cross_up(f["vfi"], 0)}


# ---------------- SC346 Cyclic Smoothed RSI band turns ----------------
def s_crsi_bands(df):
    """Cyclic Smoothed RSI (Lars von Thienen, TASC 2017): cRSI crossing back over its lower dynamic band -> long,
    back under the upper band -> short; exit at the opposite band. Stop 1.5 ATR, target 2R (ours)."""
    r = IND.crsi(df)
    L, S = _excl(r["up"], r["dn"])
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0),
                  "exit_long": r["crsi"] > r["upper"], "exit_short": r["crsi"] < r["lower"]}


# ---------------- SC347 TSI signal cross outside +/-25 (Blau) ----------------
def s_tsi_signal(df):
    """True Strength Index 25 / 13, signal 7 (Blau; StockCharts): a signal-line cross from beyond the +/-25 zone (TSI
    under -25 crossing over its signal = long, over +25 crossing under = short); exit on the opposite cross. Stop 2
    ATR, target 2R."""
    t = IND.tsi(df)
    L = t["up"] & (t["tsi"] < -25)
    S = t["dn"] & (t["tsi"] > 25)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0), "exit_long": t["dn"], "exit_short": t["up"]}


# ---------------- SC348 SMI Ergodic (TSI 20 / 5 / 5) zero cross with EMA50 ----------------
def s_smi_ergodic(df):
    """Blau's SMI Ergodic (= TSI 20 / 5, signal 5): oscillator crossing over its signal on the positive side with the
    close over the EMA50 -> long; mirror short; exit on the opposite signal cross. Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.tsi(df, r=20, s=5, sig=5)
    e = ema(c, 50)
    L = t["up"] & (t["tsi"] > 0) & (c > e)
    S = t["dn"] & (t["tsi"] < 0) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": t["dn"], "exit_short": t["up"]}


# ---------------- SC349 CMO +/-50 reversal in the trend ----------------
def s_cmo_reversal(df):
    """Chande Momentum Oscillator 9: CMO crossing back over -50 while the close is over the EMA200 (oversold dip in an
    up trend) -> long; back under +50 under the EMA200 -> short; exit at the opposite +/-50. Stop 2 ATR, 2R."""
    o, h, l, c, v = _ohlcv(df)
    m = IND.cmo(df)
    e = ema(c, 200)
    L = cross_up(m, -50) & (c > e)
    S = cross_down(m, 50) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0), "exit_long": m > 50, "exit_short": m < -50}


# ---------------- SC350 DPO cycle turn with the trend ----------------
def s_dpo_cycle(df):
    """Detrended Price Oscillator 20 (non-centred; StockCharts): a cycle low = DPO crossing back over zero while the
    SMA50 rises -> long; mirror short; exit when DPO crosses back. Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.dpo(df)
    s50 = sma(c, 50)
    L = cross_up(d, 0) & (s50 > _sh(s50, 5))
    S = cross_down(d, 0) & (s50 < _sh(s50, 5))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": cross_down(d, 0), "exit_short": cross_up(d, 0)}


# ---------------- SC351 RVI cross with the EMA50 (no RSI / StdDev, variant of SC143) ----------------
def s_rvi_trend_cross(df):
    """Relative Vigor Index 10 (Ehlers): RVI crossing over its signal while under zero and the close over the EMA50 ->
    long; mirror short (variant of SC143 without its RSI / StdDev filters, with a trend filter and the zero-side
    rule). Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.rvi(df)
    e = ema(c, 50)
    L = r["up"] & (r["rvi"] < 0) & (c > e)
    S = r["dn"] & (r["rvi"] > 0) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- SC352 PPO cross only when meaningful (scale-free threshold) ----------------
def s_ppo_threshold(df):
    """Percentage Price Oscillator 12 / 26 / 9: signal crosses count only when |PPO| >= its 100-bar median absolute
    value (a cross near zero is noise; ours) and in the PPO's sign direction reversed (cross up while PPO < 0 = early
    turn) -> long; mirror short. Stop 2 ATR, target 2R."""
    p = IND.ppo(df)
    big = np.abs(p["ppo"]) >= pd.Series(np.abs(p["ppo"])).rolling(100, min_periods=20).median().to_numpy()
    L = cross_up(p["ppo"], p["signal"]) & (p["ppo"] < 0) & big
    S = cross_down(p["ppo"], p["signal"]) & (p["ppo"] > 0) & big
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC353 Force Index zero cross with the EMA22 slope (Elder) ----------------
def s_force_index_trend(df):
    """Elder Force Index (Trading for a Living): FI(13) crossing over zero while the EMA22 rises -> long; mirror short;
    exit when FI(13) crosses back. Stop 2 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    f = IND.force_index(df)["fi"]
    e = ema(c, 22)
    L = cross_up(f, 0) & (e > _sh(e))
    S = cross_down(f, 0) & (e < _sh(e))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0), "exit_long": cross_down(f, 0), "exit_short": cross_up(f, 0)}


# ---------------- SC354 EMA 20 / 50 cross confirmed by the Chaikin Oscillator side ----------------
def s_chaikin_osc_trend(df):
    """Chaikin Oscillator 3 / 10 (StockCharts: money flow confirms the trend) as a filter: an EMA 20 / 50 cross is
    traded only when the oscillator is on the same side of zero (>= 0 for a long, <= 0 for a short) and moving that
    way; exit when the oscillator crosses to the other side. Stop 2 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.chaikin_osc(df)["osc"]
    tol = 1e-9 * np.nanmax(np.abs(k)) if np.isfinite(k).any() else 0.0
    e20, e50 = ema(c, 20), ema(c, 50)
    L = cross_up(e20, e50) & (k >= -tol) & (k >= _sh(k) - tol)
    S = cross_down(e20, e50) & (k <= tol) & (k <= _sh(k) + tol)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0), "exit_long": cross_down(k, 0), "exit_short": cross_up(k, 0)}


# ---------------- SC355 SMA50 close cross filtered by the NVI over / under its 255 EMA (Fosback) ----------------
def s_nvi_trend(df):
    """Negative Volume Index (Fosback; StockCharts): NVI over its 255-bar EMA = the 'smart money' bull phase. As a
    filter: a close crossing over the SMA50 is a long only while NVI >= its EMA, a close under the SMA50 a short only
    while NVI <= its EMA (the short side is ours - Fosback uses the long side). Exit when NVI crosses its EMA the
    other way. Stop 2 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    x = IND.nvi_pvi(df)
    nv, ne = x["nvi"], x["nvi_ema"]
    s50 = sma(c, 50)
    L = cross_up(c, s50) & (nv >= ne * (1 - 1e-12))
    S = cross_down(c, s50) & (nv <= ne * (1 + 1e-12))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0),
                  "exit_long": cross_down(nv, ne), "exit_short": cross_up(nv, ne)}


# ---------------- SC356 STARC band fade to the middle ----------------
def s_starc_fade(df):
    """STARC bands (Manning Stoller; SMA6 +/- 2 ATR15): buy near the lower band when the trend (SMA50) is up - the low
    pierces the lower band and the bar closes back inside -> long, target the SMA6 middle (level), stop 1 ATR below
    the band; mirror short at the upper band in a down trend."""
    o, h, l, c, v = _ohlcv(df)
    b = IND.starc(df)
    s50 = sma(c, 50)
    a = atr(h, l, c, 14)
    L = (l < b["lower"]) & (c > b["lower"]) & (s50 > _sh(s50, 5)) & (c < b["mid"])
    S = (h > b["upper"]) & (c < b["upper"]) & (s50 < _sh(s50, 5)) & (c > b["mid"])
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", b["lower"] - a, b["upper"] + a), "target": ("level", b["mid"], b["mid"])}


# ---------------- SC357 Random Walk Index trend start (Poulos) ----------------
def s_rwi_trend(df):
    """Random Walk Index (Poulos, TASC 9:2): RWI-high crossing over RWI-low while RWI-high > 1 (a move bigger than a
    random walk) -> long; mirror short; exit when the lines cross back. Stop 2 ATR, target 2R."""
    r = IND.rwi(df)
    L = cross_up(r["hi"], r["lo"]) & (r["hi"] > 1)
    S = cross_up(r["lo"], r["hi"]) & (r["lo"] > 1)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0),
                  "exit_long": cross_up(r["lo"], r["hi"]), "exit_short": cross_up(r["hi"], r["lo"])}


# ---------------- SC358 Donchian 20 breakout only when the VHF says trend ----------------
def s_vhf_breakout(df):
    """Vertical Horizontal Filter 28 (Adam White): trade a 20-bar close breakout only when VHF is rising and over its
    100-bar median (trending phase). Stop 2 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    f = IND.vhf(df)
    tr_ = (f > _sh(f)) & (f > pd.Series(f).rolling(100, min_periods=20).median().to_numpy())
    hh = _sh(pd.Series(h).rolling(20).max().to_numpy())
    ll = _sh(pd.Series(l).rolling(20).min().to_numpy())
    L = (c > hh) & ~(_sh(c) > _sh(hh)) & tr_
    S = (c < ll) & ~(_sh(c) < _sh(ll)) & tr_
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC359 Balance of Power zero cross with the EMA50 ----------------
def s_bop_trend(df):
    """Balance of Power (Livshin): BOP(14) crossing over zero with the close over the EMA50 -> long; mirror short;
    exit when BOP crosses back. Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    b = IND.bop(df)
    e = ema(c, 50)
    L = cross_up(b, 0) & (c > e)
    S = cross_down(b, 0) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": cross_down(b, 0), "exit_short": cross_up(b, 0)}


# ---------------- SC360 QStick zero cross with its own trend ----------------
def s_qstick_cross(df):
    """QStick 8 (Chande): QStick crossing over zero after at least 5 bars under it, close over the EMA100 -> long;
    mirror short. Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.qstick(df)
    under = pd.Series((q < 0).astype(float)).shift(1).rolling(5).min().to_numpy() == 1
    over = pd.Series((q > 0).astype(float)).shift(1).rolling(5).min().to_numpy() == 1
    e = ema(c, 100)
    L = cross_up(q, 0) & under & (c > e)
    S = cross_down(q, 0) & over & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- SC361 Trend Intensity Index 80 / 20 ----------------
def s_tii_trend(df):
    """Trend Intensity Index 60 / 30 (M. H. Pee, TASC 2002): TII crossing over 80 = strong up trend -> long, under 20
    -> short; exit when TII crosses back through 50. Stop 2 ATR."""
    t = IND.tii(df)
    L, S = _excl(cross_up(t, 80), cross_down(t, 20))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 3.0), "exit_long": cross_down(t, 50), "exit_short": cross_up(t, 50)}


# ---------------- SC362 Volatility Stop flip ----------------
def s_vstop_flip(df):
    """TradingView Volatility Stop (20 x 2 ATR from the extreme close since the flip): a flip up = long, down = short,
    stop at the VStop line, reverse on the opposite flip."""
    s = IND.volatility_stop(df)
    L, S = _excl(s["flip_up"], s["flip_dn"])
    return L, S, {"stop": ("level", s["stop"], s["stop"]), "target": None, "opposite": True}


# ---------------- SC363 Coral Trend colour flip ----------------
def s_coral_flip(df):
    """Coral Trend (LazyBear 21 / 0.4) colour flip with the close on the same side of the line -> trade it; stop
    1.5 ATR, exit on the opposite flip (reverse)."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.coral(df)
    L, S = _excl(k["flip_up"] & (c > k["coral"]), k["flip_dn"] & (c < k["coral"]))
    return L, S, {"stop": ("atr", 1.5), "target": None, "opposite": True}


# ---------------- SC364 GMMA separation after compression (Guppy) ----------------
def s_gmma_breakout(df):
    """Guppy GMMA 'trading the breakout' (Daryl Guppy; BabyPips): the long group (30-60) compressed (width under its
    100-bar 25th percentile within the last 10 bars), then the whole short group (3-15) separates above the whole long
    group -> long; mirror short. Stop under the long group's lowest EMA (level), target 2R."""
    o, h, l, c, v = _ohlcv(df)
    g = IND.gmma(df)
    w = g["l_width"]
    q = pd.Series(w).rolling(100, min_periods=30).quantile(0.25).to_numpy()
    comp = pd.Series((w < q).astype(float)).rolling(10, min_periods=1).max().to_numpy() > 0
    bull, bear = g["bull"], g["bear"]
    L = bull & ~np.concatenate([[False], bull[:-1]]) & comp
    S = bear & ~np.concatenate([[False], bear[:-1]]) & comp
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", g["l_min"], g["l_max"]), "target": ("R", 2.0)}


# ---------------- SC365 Jurik RSX 30 / 70 with the EMA trend ----------------
def s_rsx_reversal(df):
    """Jurik RSX 14: RSX crossing back over 30 with the close over the EMA200 -> long; back under 70 under the EMA200
    -> short; exit at the opposite 70 / 30. Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.rsx(df)
    e = ema(c, 200)
    L = cross_up(r, 30) & (c > e)
    S = cross_down(r, 70) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": r > 70, "exit_short": r < 30}


# ---------------- SC366 Inverse Fisher RSI -0.5 / +0.5 ----------------
def s_ift_rsi(df):
    """Inverse Fisher Transform of RSI (Vervoort; TradingView Nw4PDAwY): IFT crossing over -0.5 -> long, under +0.5
    -> short; exit at the opposite +/-0.5 level. Stop 1.5 ATR, target 2R (ours)."""
    x = IND.ift_rsi(df)
    L, S = _excl(cross_up(x, -0.5), cross_down(x, 0.5))
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": cross_down(x, 0.5), "exit_short": cross_up(x, -0.5)}


# ---------------- SC367 Cyber Cycle trigger cross on the extreme side ----------------
def s_cyber_cycle(df):
    """Ehlers Cyber Cycle (alpha 0.07): the cycle crossing over its 1-bar trigger while under zero -> long, under the
    trigger while over zero -> short; exit on the opposite cross. Stop 1.5 ATR, target 2R."""
    k = IND.cyber_cycle(df)
    L = k["up"] & (k["cycle"] < 0)
    S = k["dn"] & (k["cycle"] > 0)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": k["dn"], "exit_short": k["up"]}


# ---------------- SC368 Center of Gravity cross with the EMA50 ----------------
def s_cg_cross(df):
    """Ehlers Center of Gravity oscillator 10: CG crossing over its trigger while the close is over the EMA50 -> long;
    mirror short. Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.center_of_gravity(df)
    e = ema(c, 50)
    L, S = _excl(k["up"] & (c > e) & (k["cg"] < 0), k["dn"] & (c < e) & (k["cg"] > 0))
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- SC369 Ehlers roofing stochastic (thinkorswim EhlersStoch) ----------------
def s_ehlers_stoch(df):
    """thinkorswim strategy 'EhlersStoch': Ehlers' stochastic of the roofing-filtered close (HP 48, SS 10, 20 bars)
    crossing over 0.2 -> buy, under 0.8 -> sell (stop and reverse in the source); ours adds a 2 ATR stop."""
    r = IND.roofing(df)
    L, S = _excl(r["up"], r["dn"])
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


# ---------------- SC370 Decycler oscillator cross (TASC Sep 2015) ----------------
def s_decycler_osc(df):
    """Ehlers Decycler Oscillators (TASC Sep 2015 'Decyclers'): the fast oscillator (HP 100) crossing over the slow one
    (HP 125) with the close over the Decycler line -> long; mirror short; reverse on the opposite cross. Stop 2 ATR."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.decycler(df)
    L, S = _excl(d["up"] & (c > d["decycler"]), d["dn"] & (c < d["decycler"]))
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


# ---------------- SC371 Cybernetic Oscillator zero cross (TASC 2025.06) ----------------
def s_cybernetic_osc(df):
    """Ehlers Cybernetic Oscillator (TASC June 2025; HP 30 -> SS 20, / RMS 100): zero cross in the direction of the
    EMA100 -> trade it; exit on the cross back. Stop 2 ATR, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.cybernetic_osc(df)
    e = ema(c, 100)
    L, S = _excl(k["up"] & (c > e), k["dn"] & (c < e))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0), "exit_long": k["dn"], "exit_short": k["up"]}


# ---------------- SC372 Heiken Ashi Smoothed flip with TDI (TMS core) ----------------
def s_tms_core(df):
    """Trading Made Simple (TMS, Forex Factory / ProRealCode topic): Heiken Ashi Smoothed turns blue (bull) + TDI green
    over red and over 50 + the close over the price action channel (EMA5 of the highs) -> long; mirror short (red HA,
    green under red and 50, close under the EMA5 of the lows). Stop = the swing (10 bars), target 2R; exit when the
    smoothed HA flips (TMS rule)."""
    o, h, l, c, v = _ohlcv(df)
    ha = IND.smoothed_ha(df)
    t = IND.tdi(df)
    pac_h, pac_l = ema(h, 5), ema(l, 5)
    bull = ha["bull"]
    L = bull & (t["price"] > t["signal"]) & (t["price"] > 50) & (c > pac_h)
    S = ~bull & (t["price"] < t["signal"]) & (t["price"] < 50) & (c < pac_l)
    L = L & ~np.concatenate([[False], L[:-1]])
    S = S & ~np.concatenate([[False], S[:-1]])
    L, S = _excl(L, S)
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0), "exit_long": ha["flip_dn"], "exit_short": ha["flip_up"]}


SCOUT.update({
    "SC331": ("Coral Trend Pullback: clean candle over the line, pullback close under, close back over, swing stop, 2R (TradeIQ)", s_coral_pullback, ["1Hour", "15Min", "4Hour"]),
    "SC332": ("Heiken Ashi TMS / TDI: TDI 2 / 7 cross on the other side of 50 + HA vs the shifted SMA5, source exits (ProRealCode)", s_ha_tms_tdi, ["1Hour", "4Hour"]),
    "SC333": ("Chande Kroll Trend: close back over Lowest(21) + 3 ATR above the SMA21, out under Highest(21) - 3 ATR, long only (StockSharp 0611)", s_chande_kroll_trend, ["1Hour", "4Hour"]),
    "SC334": ("Chande Kroll Stop 10 / 1 / 9 breakout through the opposite line, reverse (Chande & Kroll)", s_chande_kroll_break, ["15Min", "1Hour", "4Hour"]),
    "SC335": ("Ehlers Sinewave: Sine / LeadSine cross, stop and reverse (WH SelfInvest system 09)", s_ehlers_sinewave_sar, ["15Min", "1Hour", "4Hour"]),
    "SC336": ("Ehlers Sinewave + MACD 11 / 66 trend filter, 10 x ATR(280) target (WH SelfInvest, Grube variant)", s_sinewave_grube, ["15Min"]),
    "SC337": ("Even Better Sinewave back over -0.85 / under +0.85 in cycle mode (VHF low), 1.5 ATR, 2R (Ehlers)", s_ebsw_reversal, ["5Min", "15Min", "1Hour"]),
    "SC338": ("Waddah Attar Explosion first 'go' bar over the explosion line and dead zone, EMA200 side, out when it fades", s_wae_ema200, ["1Min", "5Min", "15Min"]),
    "SC339": ("TDI classic: green over red above the base and 50, under the upper band, out on the cross back (Dean Malone)", s_tdi_classic, ["5Min", "15Min", "1Hour", "4Hour"]),
    "SC340": ("TDI shark fin: green back inside the volatility band -> fade, swing stop, 2R (Forex Factory)", s_tdi_shark_fin, ["15Min", "1Hour", "4Hour"]),
    "SC341": ("PMax: EMA10(hl2) crosses the ATR 10 x 3 PMax line, stop and reverse (KivancOzbilgic)", s_pmax_flip, ["15Min", "1Hour", "4Hour"]),
    "SC342": ("PMax on RSI with T3: T3(RSI14) crosses its PMax, reverse (KivancOzbilgic)", s_pmax_rsi_t3, ["15Min", "1Hour", "4Hour"]),
    "SC343": ("MACD-V signal cross outside the +/-50 neutral band (rebound from the risk zone), 2 ATR, 2R (Spiroglou)", s_macdv_rebound, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC344": ("MACD-V momentum through +/-50, out on the signal cross or past +/-150 (Spiroglou)", s_macdv_momentum, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC345": ("Volume Flow Indicator zero cross over its signal on the EMA50 side, 2 ATR, 2R (Katsanos / LazyBear)", s_vfi_zero, ["15Min", "1Hour", "4Hour"]),
    "SC346": ("Cyclic Smoothed RSI back inside its dynamic bands, out at the opposite band (whentotrade)", s_crsi_bands, ["5Min", "15Min", "1Hour"]),
    "SC347": ("True Strength Index 25 / 13 signal cross beyond +/-25, 2 ATR, 2R (Blau)", s_tsi_signal, ["15Min", "1Hour", "4Hour"]),
    "SC348": ("SMI Ergodic (TSI 20 / 5 / 5) signal cross on the EMA50 side, 1.5 ATR, 2R (Blau)", s_smi_ergodic, ["5Min", "15Min", "1Hour"]),
    "SC349": ("Chande Momentum Oscillator back over -50 / under +50 on the EMA200 side, 2 ATR, 2R", s_cmo_reversal, ["5Min", "15Min", "1Hour"]),
    "SC350": ("Detrended Price Oscillator zero cross with the SMA50 slope, 1.5 ATR, 2R", s_dpo_cycle, ["15Min", "1Hour", "4Hour"]),
    "SC351": ("Relative Vigor Index signal cross on the far side of zero with the EMA50, 1.5 ATR, 2R (variant of SC143)", s_rvi_trend_cross, ["15Min", "1Hour", "4Hour"]),
    "SC352": ("PPO early-turn signal cross only when |PPO| is over its median, 2 ATR, 2R", s_ppo_threshold, ["15Min", "1Hour", "4Hour"]),
    "SC353": ("Elder Force Index(13) zero cross with the EMA22 slope, 2 ATR, 2R", s_force_index_trend, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC354": ("EMA 20 / 50 cross only on the Chaikin Oscillator's side of zero, out when it flips, 2 ATR, 2R (StockCharts)", s_chaikin_osc_trend, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC355": ("SMA50 close cross only while the Negative Volume Index is on its 255-EMA side (Fosback filter), 2 ATR, 2R", s_nvi_trend, ["1Hour", "4Hour", "1Day"]),
    "SC356": ("STARC band pierce and close back inside with the SMA50 trend, target the SMA6 middle (Stoller)", s_starc_fade, ["5Min", "15Min", "1Hour"]),
    "SC357": ("Random Walk Index high over low and > 1 = trend start, 2 ATR, 2R (Poulos)", s_rwi_trend, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC358": ("20-bar breakout only when the Vertical Horizontal Filter is rising and over its median, 2 ATR, 2R", s_vhf_breakout, ["15Min", "1Hour", "4Hour"]),
    "SC359": ("Balance of Power zero cross with the EMA50, 1.5 ATR, 2R (Livshin)", s_bop_trend, ["15Min", "1Hour", "4Hour"]),
    "SC360": ("QStick(8) zero cross after 5 bars on the other side, EMA100 side, 1.5 ATR, 2R (Chande)", s_qstick_cross, ["15Min", "1Hour", "4Hour"]),
    "SC361": ("Trend Intensity Index over 80 / under 20, out through 50, 2 ATR, 3R (Pee)", s_tii_trend, ["1Hour", "4Hour", "1Day"]),
    "SC362": ("Volatility Stop 20 x 2 flip, stop at the line, reverse (TradingView VStop)", s_vstop_flip, ["15Min", "1Hour", "4Hour"]),
    "SC363": ("Coral Trend colour flip with the close on its side, reverse (LazyBear)", s_coral_flip, ["15Min", "1Hour", "4Hour"]),
    "SC364": ("GMMA: short group separates over / under a compressed long group, long-group stop, 2R (Guppy)", s_gmma_breakout, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC365": ("Jurik RSX back over 30 / under 70 on the EMA200 side, out at 70 / 30", s_rsx_reversal, ["5Min", "15Min", "1Hour"]),
    "SC366": ("Inverse Fisher RSI over -0.5 / under +0.5, out at the opposite level (Vervoort)", s_ift_rsi, ["5Min", "15Min", "1Hour"]),
    "SC367": ("Ehlers Cyber Cycle trigger cross on the extreme side of zero, 1.5 ATR, 2R", s_cyber_cycle, ["5Min", "15Min", "1Hour"]),
    "SC368": ("Ehlers Center of Gravity trigger cross on the far side of zero with the EMA50, 1.5 ATR, 2R", s_cg_cross, ["5Min", "15Min", "1Hour"]),
    "SC369": ("Ehlers roofing stochastic over 0.2 / under 0.8, reverse (thinkorswim EhlersStoch)", s_ehlers_stoch, ["15Min", "1Hour", "4Hour"]),
    "SC370": ("Ehlers Decycler Oscillator 100 / 125 cross on the Decycler side, reverse (TASC 2015)", s_decycler_osc, ["1Hour", "4Hour", "1Day"]),
    "SC371": ("Ehlers Cybernetic Oscillator zero cross with the EMA100, 2 ATR, 2R (TASC 2025.06)", s_cybernetic_osc, ["15Min", "1Hour", "4Hour"]),
    "SC372": ("Trading Made Simple: smoothed HA + TDI over 50 + close over the EMA5 high channel, swing stop, 2R, HA-flip exit", s_tms_core, ["15Min", "1Hour", "4Hour"]),
})


# ======================= scout 15.11.2026 night 2 =======================
# TikTok explorer idea TT-7501017829412441374 (deltatrendtrading): ICT reversal setups only in the range regime, plus
# one strategy per new indicator of agents/indicators/classic4.py, plus Three Bar Play and Wolfe Waves.

def _onset_b(m):
    m = np.asarray(m, bool)
    return m & ~np.concatenate([[False], m[:-1]])


def _gate(sig, gate):
    L, S, plan = sig
    g = np.nan_to_num(np.asarray(gate, float)).astype(bool)
    return np.asarray(L, bool) & g, np.asarray(S, bool) & g, plan


# ---------------- SC373-SC375 ICT mean reversion only in the range regime (deltatrendtrading, TikTok) ----------------
def s_sweep_er_range(df):
    """deltatrendtrading (TikTok 7501017829412441374, explorer idea): ICT reversal setups (sweep + reversal, tight
    stop, wide target) work only in a range-bound regime - measure the regime and trade them only there. Ours: SC298
    (sweep of the last confirmed swing and a close back inside, stop 0.1 ATR past the sweep, target the internal FVG /
    2R) only while |Kaufman Efficiency Ratio(20)| < 0.25 (a range). The regime measure is ours (the video names
    efficiency ratio / choppiness / ADX as candidates); SC374 / SC375 are the other two."""
    er = np.abs(IND.efficiency_ratio(df, 20))
    return _gate(s_erl_sweep_to_irl(df), er < 0.25)


def s_first4h_fade_chop(df):
    """deltatrendtrading regime filter (TikTok 7501017829412441374) on TT01 (first New York 4H candle range: a 5m
    close outside, then a close back inside -> fade; stop at the breakout extreme, 2R; the same rules as explorer idea
    TT-7707251297434701076 from @mstructure): only while Choppiness(14) > 55 (range regime; threshold ours)."""
    from agents.tiktok_strategies import s_first4h_fakeout
    ch = IND.choppiness(df, 14)
    return _gate(s_first4h_fakeout(df), ch > 55)


def s_sweep_adx_range(df):
    """deltatrendtrading regime filter (TikTok 7501017829412441374) with ADX as the regime measure: SC298 sweep
    reversal only while ADX(14) < 20 (no trend)."""
    o, h, l, c, v = _ohlcv(df)
    return _gate(s_erl_sweep_to_irl(df), adx(h, l, c, 14) < 20)


# ---------------- SC376 Schaff Trend Cycle with VIDYA ----------------
def s_stc_vidya(df):
    """Schaff Trend Cycle 23/50/10 (LazyBear port; therighttrader / capital.com rules: buy when STC turns up through
    25, sell when it turns down through 75), confirmed by Chande's VIDYA(14) rising with the close over it (mirror).
    Exit when the STC crosses back through 75 / 25. Stop 2 ATR, target 2R (not stated; ours)."""
    o, h, l, c, v = _ohlcv(df)
    st = IND.stc(df)
    vd = IND.vidya(df)
    L = st["up"] & (c > vd) & (vd > _sh(vd))
    S = st["dn"] & (c < vd) & (vd < _sh(vd))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0), "exit_long": st["dn"], "exit_short": st["up"]}


# ---------------- SC377 Connors RSI (3/2/100) ----------------
def s_connors_crsi(df):
    """Connors RSI (Connors Research; LuxAlgo / therighttrader): close over the SMA200 and CRSI under 10 -> buy; exit
    when CRSI is over 70. Mirror short (under the SMA200, CRSI over 90, out under 30). Connors trades without a stop;
    ours: 3 ATR stop and a 10-bar time exit."""
    o, h, l, c, v = _ohlcv(df)
    cr = IND.connors_rsi(df)
    m = sma(c, 200)
    L = _onset_b((c > m) & (cr < 10))
    S = _onset_b((c < m) & (cr > 90))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 3.0), "target": None, "max_bars": 10, "exit_long": cr > 70, "exit_short": cr < 30}


# ---------------- SC378 Stochastic Momentum Index (Blau) ----------------
def s_smi_blau(df):
    """Stochastic Momentum Index 10/3/3 (Blau; barchart / tradingpedia rules): SMI crosses over its signal while under
    -40 -> buy, under its signal while over +40 -> sell; out on the opposite cross. Stop 1.5 ATR, target 2R (ours)."""
    k = IND.smi_blau(df)
    L, S = _excl(k["up"] & (k["smi"] < -40), k["dn"] & (k["smi"] > 40))
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": k["dn"], "exit_short": k["up"]}


# ---------------- SC379 DeMarker 0.3 / 0.7 with the trend ----------------
def s_demarker_trend(df):
    """DeMarker 14 (capital.com / StockSharp 'Simplest DeMarker'): back over 0.3 = oversold exhausted -> buy, under
    0.7 -> sell; ours only with the EMA100 side. Stop = the 10-bar swing, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    dm = IND.demarker(df)
    e = ema(c, 100)
    L, S = _excl(cross_up(dm, 0.3) & (c > e), cross_down(dm, 0.7) & (c < e))
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


# ---------------- SC380 20-bar breakout confirmed by the Relative Volatility Index ----------------
def s_rvi_vol_breakout(df):
    """Dorsey (TASC 1993): the Relative Volatility Index confirms other signals - take buys only with RVI over 50,
    sells only under 50. Ours: close over the 20-bar high with the refined RVI over 50 (mirror under 50). Stop 2 ATR,
    target 2R."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.rvi_vol(df)["refined"]
    hh, ll = _sh(pd.Series(h).rolling(20).max().to_numpy()), _sh(pd.Series(l).rolling(20).min().to_numpy())
    L, S = _excl(_onset_b(c > hh) & (r > 50), _onset_b(c < ll) & (r < 50))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC381 Dorsey Inertia 50 cross ----------------
def s_inertia_50(df):
    """Dorsey Inertia (linreg 20 of the refined RVI; TradingView open script): over 50 = positive inertia -> long, under
    50 -> short; out on the cross back. Stop 2 ATR (ours), no fixed target."""
    x = IND.inertia(df)
    up, dn = cross_up(x, 50), cross_down(x, 50)
    L, S = _excl(up, dn)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": dn, "exit_short": up}


# ---------------- SC382 Elder EMA22 trend pullback with the SafeZone stop ----------------
def s_safezone_pullback(df):
    """Elder (Come into my Trading Room): trade with the EMA22 slope (rising over 3 bars), enter when the close comes
    back over the fast EMA13 after a pullback; protective stop = the SafeZone long stop (10 x 2.5); exit when the close
    falls under the trailing SafeZone line; target 3R. Mirror short."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.safezone(df)
    e22, e13 = ema(c, 22), ema(c, 13)
    L = (e22 > _sh(e22, 3)) & cross_up(c, e13) & (z["long_stop"] < c)
    S = (e22 < _sh(e22, 3)) & cross_down(c, e13) & (z["short_stop"] > c)
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", z["long_stop"], z["short_stop"]), "target": ("R", 3.0),
                  "exit_long": c < _sh(z["long_stop"]), "exit_short": c > _sh(z["short_stop"])}


# ---------------- SC383 Kase Peak Oscillator peak-out reversal ----------------
def s_kase_peakout(df):
    """Kase Peak Oscillator (Cynthia Kase; our reconstruction): a 'peak out' - the KPO beyond its peak line turning back
    - marks momentum exhaustion -> trade the turn (top peak-out -> sell, bottom -> buy). Stop = the 10-bar swing,
    target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.kase_peak(df)
    L, S = _excl(k["dn_peak"] & (c > o), k["up_peak"] & (c < o))
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


# ---------------- SC384 Polarized Fractal Efficiency zero cross ----------------
def s_pfe_trend(df):
    """Polarized Fractal Efficiency 10/5 (Hannula; LuxAlgo / capital.com): PFE through zero = the efficient direction
    turns; ours with the EMA50 side; out on the cross back. Stop 2 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.pfe(df)
    e = ema(c, 50)
    up, dn = cross_up(p, 0), cross_down(p, 0)
    L, S = _excl(up & (c > e), dn & (c < e))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0), "exit_long": dn, "exit_short": up}


# ---------------- SC385 Trend Trigger Factor +/-100 stop and reverse ----------------
def s_ttf_reverse(df):
    """Trend Trigger Factor 15 (M. H. Pee, TASC Dec 2004): buy when the TTF rises over +100, sell when it falls under
    -100, stop and reverse. Ours adds a 2 ATR protective stop."""
    t = IND.ttf(df)
    L, S = _excl(cross_up(t, 100), cross_down(t, -100))
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


# ---------------- SC386 Gann HiLo Activator flip with the EMA200 ----------------
def s_gann_hilo_flip(df):
    """TradingView 'Gann HiLo Activator Strategy' (open source): long on the flip up of the HiLo Activator (close over
    the SMA3 of the highs), short on the flip down; ours only on the EMA200 side, stop at the activator line, 2R,
    exit on the opposite flip."""
    o, h, l, c, v = _ohlcv(df)
    g = IND.gann_hilo(df)
    e = ema(c, 200)
    L, S = _excl(g["flip_up"] & (c > e), g["flip_dn"] & (c < e))
    return L, S, {"stop": ("level", g["line"], g["line"]), "target": ("R", 2.0),
                  "exit_long": g["flip_dn"], "exit_short": g["flip_up"]}


# ---------------- SC387 Pretty Good Oscillator +/-3 breakout ----------------
def s_pgo_breakout(df):
    """Pretty Good Oscillator 89 (Mark Johnson; LuxAlgo / backtrader): over +3 = a long-term breakout -> buy, exit when
    it falls back under 0; under -3 -> sell, exit over 0. Stop 2 ATR (ours)."""
    p = IND.pgo(df)
    L, S = _excl(cross_up(p, 3.0), cross_down(p, -3.0))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": cross_down(p, 0.0), "exit_short": cross_up(p, 0.0)}


# ---------------- SC388 Intraday Momentum Index 30 / 70 with the EMA200 ----------------
def s_imi_reversal(df):
    """Intraday Momentum Index 14 (Chande; LuxAlgo / StockSharp): back over 30 = oversold candles exhausted -> buy, back
    under 70 -> sell; ours only on the EMA200 side. Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    m = IND.imi(df)
    e = ema(c, 200)
    L, S = _excl(cross_up(m, 30) & (c > e), cross_down(m, 70) & (c < e))
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- SC389 Elliott Wave Oscillator wave-5 entry ----------------
def s_ewo_wave5(df):
    """Elliott Wave Oscillator 5/35 (classic use, LuxAlgo '5/35 oscillator' / daytrading.com): the oscillator's highest
    peak marks wave 3; when it then pulls back to near zero (within 25% of that peak) and turns up, wave 4 is done ->
    buy for wave 5. Ours: peak = the 60-bar EWO high, at least 90% of the 200-bar max |EWO|; stop = the 10-bar swing,
    target 2R. Mirror for shorts."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.ewo(df)
    P = pd.Series(w).rolling(60).max().to_numpy()
    T = pd.Series(w).rolling(60).min().to_numpy()
    big = pd.Series(np.abs(w)).rolling(200).max().to_numpy()
    turn_up = (w > _sh(w)) & (_sh(w) <= _sh(w, 2))
    turn_dn = (w < _sh(w)) & (_sh(w) >= _sh(w, 2))
    with np.errstate(invalid="ignore"):
        L = (P > 0) & (P >= 0.9 * big) & (np.abs(w) <= 0.25 * P) & turn_up
        S = (T < 0) & (-T >= 0.9 * big) & (np.abs(w) <= -0.25 * T) & turn_dn
    L, S = _excl(L, S)
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


# ---------------- SC390 KDJ J-line extreme + K/D cross ----------------
def s_kdj_extreme(df):
    """KDJ 9/3/3 (KuCoin / changelly crypto guides): J under 0 = oversold; buy when K then crosses over D; J over 100
    and K crossing under D -> sell. Stop 1.5 ATR, target 2R (ours)."""
    k = IND.kdj(df)
    jp = _sh(k["j"])
    L, S = _excl((jp < 0) & cross_up(k["k"], k["d"]), (jp > 100) & cross_down(k["k"], k["d"]))
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- SC391 VIDYA close cross with a rising VIDYA ----------------
def s_vidya_cross(df):
    """Chande VIDYA 14 / CMO 9 (TradingView open 'VIDYA Chande' / 'VIDYA ProTrend'): the close crossing over a VIDYA
    that rises over 3 bars -> long, mirror short. Stop 2 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    vd = IND.vidya(df)
    L, S = _excl(cross_up(c, vd) & (vd > _sh(vd, 3)), cross_down(c, vd) & (vd < _sh(vd, 3)))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC392 / SC393 / SC394 Bill Williams AO saucer, twin peaks, AC ----------------
def s_ao_saucer(df):
    """Bill Williams (New Trading Dimensions; BabyPips / AvaTrade): AO saucer - AO over zero, two falling bars (the
    second lower) then a rising bar -> buy (mirror under zero). Stop = the 5-bar low / high, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    a = IND.bw_ao(df)
    L, S = _excl(a["saucer_up"], a["saucer_dn"])
    sl, sh = _swing_lv(l, h, 5)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


def s_ao_twin_peaks(df):
    """Bill Williams AO twin peaks: two AO troughs under zero, the second higher, no zero cross between, AO turning up ->
    buy (mirror over zero). Stop = the 10-bar swing, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    a = IND.bw_ao(df)
    L, S = _excl(a["twin_up"], a["twin_dn"])
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


def _run_len(m):
    m = np.asarray(m, bool)
    out = np.zeros(len(m), int)
    for i in range(len(m)):
        out[i] = out[i - 1] + 1 if m[i] and i else (1 if m[i] else 0)
    return out


def s_ac_bars(df):
    """Bill Williams Accelerator (AC = AO - SMA5(AO)): buy on the 2nd consecutive rising AC bar over zero or the 3rd
    under zero (mirror for sells) - Williams' rule; out after two AC bars against. Stop 1.5 ATR, target 2R (ours)."""
    a = IND.bw_ao(df)
    ac, g = a["ac"], np.nan_to_num(a["ac"] > _sh(a["ac"])).astype(bool)
    r = np.nan_to_num(a["ac"] < _sh(a["ac"])).astype(bool)
    gn, rn = _run_len(g), _run_len(r)
    with np.errstate(invalid="ignore"):
        L = ((ac > 0) & (gn == 2)) | ((ac < 0) & (gn == 3))
        S = ((ac < 0) & (rn == 2)) | ((ac > 0) & (rn == 3))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": rn == 2, "exit_short": gn == 2}


# ---------------- SC395 Market Facilitation Index: no breakouts on fake / fade bars ----------------
def s_mfi_no_fake_break(df):
    """Bill Williams Market Facilitation Index (MQL5 article 12172, LuxAlgo): 'fake' bars (MFI up, volume down) and 'fade'
    bars (both down) show no real participation - Williams does not trust moves on them. Ours: a close beyond the
    20-bar high / low, skipped when the breakout bar is a fake or fade bar (tick volume). Stop 1.5 ATR, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    m = IND.bw_mfi(df)
    hh, ll = _sh(pd.Series(h).rolling(20).max().to_numpy()), _sh(pd.Series(l).rolling(20).min().to_numpy())
    weak = np.asarray(m["fake"] | m["fade"], bool)
    L, S = _excl(_onset_b(c > hh) & ~weak, _onset_b(c < ll) & ~weak)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- SC396 Alligator wakes up ----------------
def s_alligator_wake(df):
    """Bill Williams Alligator (BabyPips): the sleeping alligator (jaw, teeth, lips intertwined) wakes when the lines
    fan out in order - lips over teeth over jaw with the close over the lips -> long; exit on a close under the teeth
    (Williams). Ours: 'sleeping' = the line spread under 0.4 ATR on 8 of the last 15 bars; awake = spread over 0.5
    ATR. Stop 2 ATR. Mirror short. (SC002 / SC144 trade fractals / the Kumo - not the awakening.)"""
    o, h, l, c, v = _ohlcv(df)
    g = IND.alligator(df)
    a = atr(h, l, c, 14)
    j, t, p = g["jaw"], g["teeth"], g["lips"]
    spread = np.fmax(np.fmax(j, t), p) - np.fmin(np.fmin(j, t), p)
    with np.errstate(invalid="ignore"):
        asleep = pd.Series((spread < 0.4 * a).astype(float)).rolling(15).sum().shift(1).to_numpy() >= 8
        L = _onset_b((p > t) & (t > j) & (c > p) & (spread > 0.5 * a)) & asleep
        S = _onset_b((p < t) & (t < j) & (c < p) & (spread > 0.5 * a)) & asleep
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": c < t, "exit_short": c > t}


# ---------------- SC397 Chop Zone turns darkest turquoise ----------------
def s_chop_zone_trend(df):
    """TradingView Chop Zone: the EMA34 angle zone turning to the strongest colour (+4, angle >= 5 deg) with the close over
    the EMA34 = trending up -> long; exit when the zone turns red side (<= -1). Mirror short. Stop 2 ATR, target 3R."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.chop_zone(df)["zone"]
    e = ema(c, 34)
    zp = _sh(z)
    with np.errstate(invalid="ignore"):
        L = (z == 4) & (zp < 4) & (c > e)
        S = (z == -4) & (zp > -4) & (c < e)
        xl, xs = z <= -1, z >= 1
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 3.0), "exit_long": xl, "exit_short": xs}


# ---------------- SC398 Price Volume Trend signal cross ----------------
def s_pvt_cross(df):
    """Price Volume Trend (StockCharts) crossing its EMA21 signal in the direction of the EMA50 (tick volume). Stop 2 ATR,
    target 2R."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.pvt(df)
    e = ema(c, 50)
    L, S = _excl(p["up"] & (c > e), p["dn"] & (c < e))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC399 Vortex over 1.10 ----------------
def s_vortex_strong(df):
    """Vortex Indicator 14 (Botes & Siepman; StockCharts): VI+ rising over 1.10 while over VI- = a strong up-trend
    starting -> long; VI- over 1.10 -> short (TV45 trades the plain VI cross). Stop 2 ATR, target 2R."""
    x = IND.vortex(df)
    vp, vm = x["vi_plus"], x["vi_minus"]
    with np.errstate(invalid="ignore"):
        L, S = _excl(cross_up(vp, 1.10) & (vp > vm), cross_up(vm, 1.10) & (vm > vp))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC400 Parabolic SAR flip with ADX and EMA50 ----------------
def s_psar_adx(df):
    """Parabolic SAR 0.02 / 0.2 flip only with ADX(14) over 20 and the close on the EMA50 side; stop at the SAR dot,
    target 2R, out on the opposite flip (TV21 uses the EMA200 alone, SC104 the last opposite dot)."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.psar(df)
    ad = adx(h, l, c, 14)
    e = ema(c, 50)
    L, S = _excl(p["flip_up"] & (ad > 20) & (c > e), p["flip_dn"] & (ad > 20) & (c < e))
    return L, S, {"stop": ("level", p["sar"], p["sar"]), "target": ("R", 2.0),
                  "exit_long": p["flip_dn"], "exit_short": p["flip_up"]}


# ---------------- SC401 KST signal cross on the far side of zero ----------------
def s_kst_far_side(df):
    """Know Sure Thing (Pring): the KST crossing over its signal while under zero = an early bottom (mirror over zero
    for tops) - TV58 trades every cross with the EMA200. Stop 2 ATR, target 2R."""
    k = IND.kst(df)
    with np.errstate(invalid="ignore"):
        L, S = _excl(k["up"] & (k["kst"] < 0), k["dn"] & (k["kst"] > 0))
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


# ---------------- SC402 Crabel stretch breakout ----------------
def s_crabel_stretch(df):
    """Toby Crabel (TASC 1988 'Playing the opening range breakout'; Day Trading with Short Term Price Patterns): buy
    when the price moves 'stretch' above the day's open (stretch = 10-day mean of min(high - open, open - low)), sell
    'stretch' below; first trigger per New York day; exit at the close of the day (16:00 NY, before the break). Ours:
    entry on the bar close beyond the level, stop at the day's open."""
    o, h, l, c, v = _ohlcv(df)
    s = IND.stretch(df)
    hs, he, wd, wde, day = _local(df, "America/New_York")
    L = _first_per_day(cross_up(c, s["up_lvl"]) & (hs < 15.5), day)
    S = _first_per_day(cross_down(c, s["dn_lvl"]) & (hs < 15.5), day)
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", s["day_open"], s["day_open"]), "target": None, "force_exit": he >= 16.0}


# ---------------- SC403 TD Sequential countdown 13 ----------------
def s_td_countdown13(df):
    """Tom DeMark TD Sequential: a completed buy countdown 13 (after a buy setup 9) = trend exhaustion -> buy; sell
    countdown 13 -> sell (GH04 trades the setup 9). Stop = the 10-bar swing, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.td_sequential(df)
    L, S = _excl(t["buy_cd"] == 13, t["sell_cd"] == 13)
    sl, sh = _swing_lv(l, h, 10)
    return L, S, {"stop": ("level", sl, sh), "target": ("R", 2.0)}


# ---------------- SC404 Three Bar Play ----------------
def s_three_bar_play(df):
    """Three Bar Play (daytradingtoolkit.com, momentum day traders): an igniting bar (true range >= 2 ATR14 - a gap counts, ours - close in
    the top 25%), then 1-3 resting bars each with a range under half of the igniter that hold in its upper half -> buy when a
    bar closes over the resting bars' high; stop under the resting low; target 2R. Mirror short."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    for j in range(15, n - 2):
        top, bot = max(h[j], c[j - 1]), min(l[j], c[j - 1])     # the igniter's true range (a gap counts)
        r = top - bot
        if not np.isfinite(a[j - 1]) or r < 2 * a[j - 1]:
            continue
        mid = bot + r / 2
        bull = c[j] >= top - 0.25 * r
        bear = c[j] <= bot + 0.25 * r
        if not (bull or bear):
            continue
        rh, rl = -np.inf, np.inf
        for i in range(j + 1, min(n, j + 5)):
            k = i - j  # resting bars j+1 .. i-1 (1..3)
            if k >= 2:
                if bull and c[i] > rh:
                    L[i], sl_l[i] = True, rl
                    break
                if bear and c[i] < rl:
                    S[i], sl_s[i] = True, rh
                    break
            if k > 3:
                break
            ri = h[i] - l[i]
            ok = ri < 0.5 * r and ((bull and l[i] >= mid) or (bear and h[i] <= mid))
            if not ok:
                break
            rh, rl = max(rh, h[i]), min(rl, l[i])
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", sl_l, sl_s), "target": ("R", 2.0)}


# ---------------- SC405 Wolfe Waves ----------------
def s_wolfe_wave(df, k=3):
    """Wolfe Waves (Bill Wolfe; LuxAlgo / ATAS / mql5 forextsd pdf): bullish = five alternating swings 1 (low), 2
    (high), 3 (low under 1), 4 (high under 2, over 1), 5 (low under 3, at / beyond the 1-3 line); buy when point 5 is
    confirmed (k = 3 bars later, no repaint) with a bullish close; stop 0.1 ATR under point 5; target = the 1-4 line
    at the entry bar (the 'EPA' line), when it is above the price (else 2R). Mirror for bearish."""
    from agents.indicators._core import pivots
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    tg_l, tg_s = np.full(n, np.nan), np.full(n, np.nan)
    piv = []  # (bar, price, +1 high / -1 low) in confirmation order, alternating
    for i in range(n):
        for conf, val, idx, kind in ((phc, phv, phi, 1), (plc, plv, pli, -1)):
            if conf[i]:
                if piv and piv[-1][2] == kind:
                    if (kind == 1 and val[i] > piv[-1][1]) or (kind == -1 and val[i] < piv[-1][1]):
                        piv[-1] = (int(idx[i]), float(val[i]), kind)
                else:
                    piv.append((int(idx[i]), float(val[i]), kind))
                if len(piv) >= 5 and piv[-1][0] == int(idx[i]):
                    (b1, p1, k1), (b2, p2, _), (b3, p3, _), (b4, p4, _), (b5, p5, k5) = piv[-5:]
                    if k5 == -1 and p3 < p1 and p1 < p4 < p2 and p5 < p3:
                        line13 = p1 + (p3 - p1) * (b5 - b1) / max(b3 - b1, 1)
                        if p5 <= line13 + 0.2 * a[i] and c[i] > o[i]:
                            t = p1 + (p4 - p1) * (i - b1) / max(b4 - b1, 1)
                            L[i], sl_l[i] = True, p5 - 0.1 * a[i]
                            tg_l[i] = t if t > c[i] else c[i] + 2 * (c[i] - sl_l[i])
                    if k5 == 1 and p3 > p1 and p2 < p4 < p1 and p5 > p3:
                        line13 = p1 + (p3 - p1) * (b5 - b1) / max(b3 - b1, 1)
                        if p5 >= line13 - 0.2 * a[i] and c[i] < o[i]:
                            t = p1 + (p4 - p1) * (i - b1) / max(b4 - b1, 1)
                            S[i], sl_s[i] = True, p5 + 0.1 * a[i]
                            tg_s[i] = t if t < c[i] else c[i] - 2 * (sl_s[i] - c[i])
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl_l, sl_s, tg_l, tg_s)


_F = ["5Min", "15Min", "1Hour"]
_M = ["15Min", "1Hour", "4Hour"]
SCOUT.update({
    "SC373": ("ICT sweep reversal (SC298) only in a range: |Efficiency Ratio 20| < 0.25 (deltatrendtrading regime filter)", s_sweep_er_range, ["5Min", "15Min"]),
    "SC374": ("First NY 4H candle fade (TT01) only while Choppiness 14 > 55 (deltatrendtrading regime filter)", s_first4h_fade_chop, ["5Min"]),
    "SC375": ("ICT sweep reversal (SC298) only while ADX 14 < 20 (deltatrendtrading regime filter)", s_sweep_adx_range, ["5Min", "15Min"]),
    "SC376": ("Schaff Trend Cycle through 25 / 75 with a rising VIDYA under / over the close, STC exit, 2 ATR, 2R", s_stc_vidya, _M),
    "SC377": ("Connors RSI 3/2/100 under 10 over the SMA200 (over 90 under it), out over 70 / under 30, 10 bars", s_connors_crsi, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC378": ("Stochastic Momentum Index 10/3/3 signal cross under -40 / over +40, 1.5 ATR, 2R (Blau)", s_smi_blau, _F),
    "SC379": ("DeMarker 14 back over 0.3 / under 0.7 on the EMA100 side, swing stop, 2R", s_demarker_trend, _F),
    "SC380": ("20-bar breakout confirmed by the refined Relative Volatility Index over / under 50, 2 ATR, 2R (Dorsey)", s_rvi_vol_breakout, _M),
    "SC381": ("Dorsey Inertia through 50, out on the cross back, 2 ATR", s_inertia_50, _M),
    "SC382": ("Elder: EMA22 slope + close back over the EMA13, SafeZone stop and trailing exit, 3R", s_safezone_pullback, _M),
    "SC383": ("Kase Peak Oscillator peak-out reversal, swing stop, 2R (our KPO reconstruction)", s_kase_peakout, _M),
    "SC384": ("Polarized Fractal Efficiency 10/5 zero cross on the EMA50 side, 2 ATR, 2R (Hannula)", s_pfe_trend, _M),
    "SC385": ("Trend Trigger Factor 15 over +100 / under -100, stop and reverse (Pee, TASC 2004)", s_ttf_reverse, _M),
    "SC386": ("Gann HiLo Activator 3 flip on the EMA200 side, stop at the line, 2R", s_gann_hilo_flip, ["5Min", "15Min", "1Hour", "4Hour"]),
    "SC387": ("Pretty Good Oscillator 89 over +3 / under -3, out through 0 (Mark Johnson)", s_pgo_breakout, ["1Hour", "4Hour", "1Day"]),
    "SC388": ("Intraday Momentum Index 14 back over 30 / under 70 on the EMA200 side, 1.5 ATR, 2R (Chande)", s_imi_reversal, _F),
    "SC389": ("Elliott Wave Oscillator wave-5 entry: pullback to near zero after the wave-3 peak, turn, swing stop, 2R", s_ewo_wave5, _M),
    "SC390": ("KDJ 9/3/3: J under 0 then K over D (J over 100 then K under D), 1.5 ATR, 2R", s_kdj_extreme, _F),
    "SC391": ("Close cross of a rising VIDYA 14 / CMO 9, 2 ATR, 2R (Chande)", s_vidya_cross, _M),
    "SC392": ("Bill Williams AO saucer over / under zero, 5-bar stop, 2R", s_ao_saucer, _M),
    "SC393": ("Bill Williams AO twin peaks, swing stop, 2R", s_ao_twin_peaks, _M),
    "SC394": ("Bill Williams Accelerator: 2nd rising AC bar over zero / 3rd under zero, out after 2 against, 1.5 ATR, 2R", s_ac_bars, _M),
    "SC395": ("20-bar close breakout skipped on Market Facilitation Index fake / fade bars (Williams), 1.5 ATR, 2R", s_mfi_no_fake_break, _F),
    "SC396": ("Alligator wakes up: lines fan out in order after sleeping, close-under-teeth exit, 2 ATR (Williams)", s_alligator_wake, _M),
    "SC397": ("Chop Zone turns darkest turquoise / red with the EMA34 side, out on the opposite colour, 2 ATR, 3R", s_chop_zone_trend, _M),
    "SC398": ("Price Volume Trend over / under its EMA21 on the EMA50 side, 2 ATR, 2R", s_pvt_cross, _M),
    "SC399": ("Vortex 14: VI+ (VI-) rising over 1.10 while dominant, 2 ATR, 2R", s_vortex_strong, _M),
    "SC400": ("Parabolic SAR flip with ADX > 20 and the EMA50 side, SAR stop, 2R, flip exit", s_psar_adx, ["5Min", "15Min", "1Hour", "4Hour"]),
    "SC401": ("KST signal cross under zero (over zero for shorts), 2 ATR, 2R (Pring)", s_kst_far_side, ["1Hour", "4Hour", "1Day"]),
    "SC402": ("Crabel stretch: close beyond the day open +/- 10-day stretch, first per NY day, open stop, out 16:00 NY", s_crabel_stretch, ["5Min", "15Min"]),
    "SC403": ("TD Sequential countdown 13 exhaustion reversal, swing stop, 2R (DeMark)", s_td_countdown13, ["1Hour", "4Hour", "1Day"]),
    "SC404": ("Three Bar Play: 2-ATR igniting bar + 1-3 tight resting bars in its upper half, break, resting stop, 2R", s_three_bar_play, ["1Min", "5Min", "15Min"]),
    "SC405": ("Wolfe Wave 5 swings (k 3, confirmed), stop past point 5, target the 1-4 line", s_wolfe_wave, ["15Min", "1Hour", "4Hour"]),
})


# =====================================================================================================================
# scout 15.11.2026 night 3 (Sunday, one weekly run): SC406-SC454 - Ehlers TASC 2025 filters, community classics,
# Raschke / Connors Street Smarts, Joe Ross, Darvas, VSA, chart types (3LB, P&F), round numbers, ADR, CBDR, RVOL ORB
# (Zarattini et al.), ICT macros, and the 5 new TikTok explorer ideas (TT-s-*). Indicators: agents/indicators
# ehlers3.py, classic5.py, structure2.py, levels2.py.
# =====================================================================================================================
from agents.indicators._core import dmi as _dmi  # noqa: E402


def _r(a, n=1):
    """Shift right by n with NaN fill (the value n bars ago)."""
    a = np.asarray(a, float)
    return np.concatenate([np.full(n, np.nan), a[:-n]]) if n > 0 else a


def _swl(h, l, a, n=5, buf=0.1):
    return _s(l).rolling(n, min_periods=1).min().to_numpy() - buf * a, _s(h).rolling(n, min_periods=1).max().to_numpy() + buf * a


def _atr_plan(c, a, k=2.0, r=2.0, **kw):
    return _plan_r(c - k * a, c + k * a, r, **kw) if r else {"stop": ("level", c - k * a, c + k * a), "target": None, **kw}


def _ny(df):
    t = df.index.tz_convert("America/New_York")
    return (t.hour.to_numpy() + t.minute.to_numpy() / 60.0), pd.Series(t.date).to_numpy()


# ---------------- SC406 CM Williams Vix Fix bottoms / tops ----------------
def s_vix_fix(df):
    """CM Williams Vix Fix (ChrisMoody, open source; Larry Williams' synthetic VIX): WVF = (highest close 22 - low) /
    highest close 22 x 100; a 'fear spike' when WVF >= its BB 20/2 upper band or >= 0.85 x its 50-bar high. Moody's
    use: buy when the spike bars END - ours: a spike in the last 3 bars, none on this bar, and a close over the prior
    bar's high. Tops: the inverse Vix Fix ('Williams Vix Fix Bottoms and Tops') mirrored for shorts. Stop 0.1 ATR beyond
    the 5-bar extreme, target 2R (the scripts are signals only - stop / target are the library standard)."""
    o, h, l, c, v = _ohlcv(df)
    vf = IND.vix_fix(df)
    a = atr(h, l, c, 14)
    b, t = vf["bottom"].astype(float), vf["top"].astype(float)
    rb = _s(_r(b)).rolling(3, min_periods=1).max().to_numpy() > 0
    rt = _s(_r(t)).rolling(3, min_periods=1).max().to_numpy() > 0
    L = rb & ~vf["bottom"] & (c > _r(h))
    S = rt & ~vf["top"] & (c < _r(l))
    sl, ss = _swl(h, l, a, 5)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC407 BBTrend zero cross ----------------
def s_bbtrend(df):
    """John Bollinger's BBTrend (2024): (|lower20 - lower50| - |upper20 - upper50|) / middle20 x 100 - positive = the
    fast bands expand downward less than upward (up trend). Bollinger: a reading crossing zero marks the trend change.
    Ours: long when BBTrend crosses over 0 with the close over the 20-bar middle band, out when it crosses back under 0
    (mirror for shorts); stop 2 ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    bt = np.nan_to_num(IND.bbtrend(df))
    a = atr(h, l, c, 14)
    m = sma(c, 20)
    L = cross_up(bt, 0) & (c > m)
    S = cross_down(bt, 0) & (c < m)
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=bt < 0, exit_short=bt > 0)


# ---------------- SC408-SC410 Khalil Volume / Price Zone Oscillators ----------------
def s_vzo_trend(df):
    """Walid Khalil, 'Trending or Nontrending?' (TASC 2011.05), Volume Zone Oscillator in a trend (ADX14 > 18, close
    over / under the EMA60): buy when the VZO, after a dip under -40 in the last 20 bars (ours), crosses back over 0; out when
    it falls under -5, or under +40 after having been over +60 (Khalil's zones +/-5, 40, 60). Mirror for shorts.
    Stop 2 ATR (ours - the article exits on the oscillator only)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.vzo(df)
    pdi, ndi, ax = _dmi(h, l, c, 14)
    e = ema(c, 60)
    a = atr(h, l, c, 14)
    dip = _s(z).rolling(20, min_periods=1).min().to_numpy() < -40
    pk = _s(z).rolling(20, min_periods=1).max().to_numpy() > 40
    L = (ax > 18) & (c > e) & dip & cross_up(z, 0)
    S = (ax > 18) & (c < e) & pk & cross_down(z, 0)
    hi60 = _s(z).rolling(20, min_periods=1).max().to_numpy() > 60
    lo60 = _s(z).rolling(20, min_periods=1).min().to_numpy() < -60
    xl = cross_down(z, -5) | (hi60 & cross_down(z, 40))
    xs = cross_up(z, 5) | (lo60 & cross_up(z, -40))
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=xl, exit_short=xs)


def _zone_range(df, z):
    o, h, l, c, v = _ohlcv(df)
    pdi, ndi, ax = _dmi(h, l, c, 14)
    a = atr(h, l, c, 14)
    L = (ax < 18) & cross_up(z, -40)
    S = (ax < 18) & cross_down(z, 40)
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=z > 40, exit_short=z < -40)


def s_vzo_range(df):
    """Khalil (TASC 2011.05), the non-trending market (ADX14 < 18): buy when the VZO crosses back over -40 (oversold),
    sell at +40; mirror for shorts (cross back under +40, cover at -40). Stop 2 ATR (ours)."""
    return _zone_range(df, IND.vzo(df))


def s_pzo_range(df):
    """Khalil's Price Zone Oscillator (TASC 2011 companion of the VZO; thinkorswim PriceZoneOscillatorLX/SE) in the
    non-trending market: ADX14 < 18, buy on the PZO back over -40, out at +40; mirror. Stop 2 ATR (ours)."""
    return _zone_range(df, IND.pzo(df))


# ---------------- SC411 Twiggs Money Flow ----------------
def s_twiggs_mf(df):
    """Twiggs Money Flow 21 (Colin Twiggs, incrediblecharts.com): a CMF built on the true range with Wilder smoothing.
    Twiggs trades the zero line in the trend's direction - ours: TMF crosses over 0 after 10+ bars under it while the
    close is over a rising EMA50 (mirror), stop 2 ATR, 2R. Differs from SC271 (CMF 20, plain high-low location,
    negative-run stop)."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.twiggs_mf(df)
    e = ema(c, 50)
    a = atr(h, l, c, 14)
    neg = _s(_r(t) < 0).rolling(10).min().to_numpy() > 0
    pos = _s(_r(t) > 0).rolling(10).min().to_numpy() > 0
    L = cross_up(t, 0) & neg & (c > e) & (e > _r(e, 5))
    S = cross_down(t, 0) & pos & (c < e) & (e < _r(e, 5))
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, 2.0)


# ---------------- SC412 Relative Momentum Index ----------------
def s_rmi_reversal(df):
    """Relative Momentum Index 14 / 5 (Roger Altman, TASC 1993): the RSI of the 5-bar momentum. Altman's overbought /
    oversold 70 / 30 - ours: back over 30 on the EMA200's up side (back under 70 under it), 1.5 ATR stop, 2R."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.rmi(df)
    e = ema(c, 200)
    a = atr(h, l, c, 14)
    L = cross_up(r, 30) & (c > e)
    S = cross_down(r, 70) & (c < e)
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 1.5, 2.0)


# ---------------- SC413 DecisionPoint PMO ----------------
def s_pmo_cross(df):
    """DecisionPoint Price Momentum Oscillator 35/20 with its EMA10 signal (Carl Swenlin, StockCharts ChartSchool):
    a PMO crossover of the signal line BELOW zero = buy (above zero for the sell), exit on the opposite crossover.
    Stop 2 ATR, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.pmo(df)
    a = atr(h, l, c, 14)
    up, dn = cross_up(p["pmo"], p["signal"]), cross_down(p["pmo"], p["signal"])
    L, S = up & (p["pmo"] < 0), dn & (p["pmo"] > 0)
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, 2.0, exit_long=dn, exit_short=up)


# ---------------- SC414 Tillson T3 cross ----------------
def s_t3_cross(df):
    """Tillson T3 (TASC 1998, volume factor 0.7) crossover, as in TradingView's 'T3 Averages Strategy': T3(8) crosses
    T3(21) - long over, short under, out on the cross back. Stop 2 ATR, no fixed target (the published script is
    always-in)."""
    o, h, l, c, v = _ohlcv(df)
    f, s = IND.t3(df, 8), IND.t3(df, 21)
    a = atr(h, l, c, 14)
    up, dn = cross_up(f, s), cross_down(f, s)
    L, S = _excl(up, dn)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=dn, exit_short=up)


# ---------------- SC415 Absolute Strength Histogram ----------------
def s_ash_cross(df):
    """Absolute Strength Histogram (RSI method, 9 / 3; MQL5 codebase / backtrader recipe): buy when the smoothed bulls
    cross over the smoothed bears, sell on the opposite cross (the original's signals), out on the reverse cross.
    Stop 2 ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    s = IND.ash(df)
    a = atr(h, l, c, 14)
    up, dn = cross_up(s["bulls"], s["bears"]), cross_down(s["bulls"], s["bears"])
    L, S = _excl(up, dn)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=dn, exit_short=up)


# ---------------- SC416 Trendilo ----------------
def s_trendilo(df):
    """Trendilo (dudeowns, open source): ALMA 50 of the bar % change against its RMS band - over the band = up trend,
    under minus the band = down trend. Ours: enter when the trend turns +1 / -1, out when the smoothed change crosses
    zero against the trade; stop 2 ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.trendilo(df)
    a = atr(h, l, c, 14)
    tr = t["trend"]
    L = (tr == 1) & (_r(tr) != 1)
    S = (tr == -1) & (_r(tr) != -1)
    av = np.nan_to_num(t["avpch"])
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=av < 0, exit_short=av > 0)


# ---------------- SC417 Darvas box ----------------
def s_darvas(df):
    """Nicolas Darvas' box (1960): top = a 20-bar high not exceeded for 3 bars, bottom = the lowest low after it not
    broken for 3 bars; buy the close over the top of a complete box, stop 0.1 ATR under the box bottom (Darvas' stop),
    target 2R (ours - Darvas trailed into the next box). Breakdown under the bottom = the short mirror (ours)."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.darvas_box(df)
    a = atr(h, l, c, 14)
    L, S = _excl(d["brk_up"], d["brk_dn"])
    return L, S, _plan_r(d["bot"] - 0.1 * a, d["top"] + 0.1 * a, 2.0)


# ---------------- SC418 / SC419 Volume Spread Analysis ----------------
def s_vsa_test(df):
    """VSA (Tom Williams, 'Master the Markets'): a no-supply bar (down bar, below-average true-range spread, volume under the
    two previous bars) inside an up trend = professionals not selling; buy when a following bar (within 2) closes over its high.
    Stop 0.1 ATR under the no-supply bar, 2R. Mirror: no-demand bar in a down trend. Trend = close over a rising EMA50
    (ours). Tick volume."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.vsa(df)
    e = ema(c, 50)
    a = atr(h, l, c, 14)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    up = (c > e) & (e > _r(e, 5))
    dn = (c < e) & (e < _r(e, 5))
    for i in range(2, n):
        for k in (1, 2):
            j = i - k
            if q["no_supply"][j] and up[j] and c[i] > h[j] and not L[i - 1]:
                L[i], sl[i] = True, l[j] - 0.1 * a[i]
            if q["no_demand"][j] and dn[j] and c[i] < l[j] and not S[i - 1]:
                S[i], ss[i] = True, h[j] + 0.1 * a[i]
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


def s_vsa_upthrust(df):
    """VSA upthrust / spring: a new 20-bar high on high volume (> 1.5 x average) that closes back under the broken high
    = upthrust; sell when the next bar closes under the upthrust bar's low, stop 0.1 ATR over its high, 2R. Spring
    (shakeout) = mirror for longs."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.vsa(df)
    a = atr(h, l, c, 14)
    ut, sp = _r(q["upthrust"].astype(float)) > 0, _r(q["spring"].astype(float)) > 0
    S = ut & (c < _r(l))
    L = sp & (c > _r(h))
    L, S = _excl(L, S)
    return L, S, _plan_r(_r(l) - 0.1 * a, _r(h) + 0.1 * a, 2.0)


# ---------------- SC420 / SC421 Round numbers ----------------
def s_round_reject(df):
    """Psychological round levels (BabyPips / DailyFX 'psychological levels and round numbers'): a touch of a '00'
    level that no close has gone through in the last 20 bars, rejected - low within 0.05 ATR of the level under the previous close, close back over
    it, a bullish rejection candle (lower wick >= the body and >= 30% of the range); stop 0.5 ATR under the level (or 0.1 ATR under the low
    if lower), 2R. Mirror at the level over. OUR translation of '00' to every instrument: step = 10^(digits - 2) of
    the price (EURUSD 0.01, USDJPY 1, gold 10, US100 100)."""
    o, h, l, c, v = _ohlcv(df)
    rl = IND.round_levels(df)
    a = atr(h, l, c, 14)
    lb, la = _r(rl["below"]), _r(rl["above"])
    rng = np.maximum(h - l, 1e-12)
    lw = (np.minimum(o, c) - l) / rng
    uw = (h - np.maximum(o, c)) / rng
    c20 = _r(_s(c).rolling(20, min_periods=1).min().to_numpy())     # level not closed through in the last 20 bars
    C20 = _r(_s(c).rolling(20, min_periods=1).max().to_numpy())
    body = np.abs(c - o)
    L = (l <= lb + 0.05 * a) & (c > lb) & (c > o) & (lw >= 0.3) & (lw * rng >= body) & (c20 > lb)
    S = (h >= la - 0.05 * a) & (c < la) & (c < o) & (uw >= 0.3) & (uw * rng >= body) & (C20 < la)
    sl = np.minimum(lb - 0.5 * a, l - 0.1 * a)
    ss = np.maximum(la + 0.5 * a, h + 0.1 * a)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


def s_round_retest(df):
    """Round level break and retest (BabyPips psychological levels; IG 'Psychological Levels & Round Numbers'): a close
    through a '00' level, then within 2-30 bars a pull back that touches it (0.1 ATR) and closes away in the break's
    direction; stop 0.5 ATR beyond the level, 2R. Same level step as SC420."""
    o, h, l, c, v = _ohlcv(df)
    rl = IND.round_levels(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    bu = bd = None     # (level, bar)
    for i in range(1, n):
        lv_up = rl["above"][i - 1]
        lv_dn = rl["below"][i - 1]
        if np.isfinite(lv_up) and c[i - 1] < lv_up <= c[i]:
            bu = (lv_up, i)
        if np.isfinite(lv_dn) and c[i - 1] > lv_dn >= c[i]:
            bd = (lv_dn, i)
        if bu is not None and i - bu[1] >= 2:
            if i - bu[1] > 30 or c[i] < bu[0] - 0.5 * a[i]:
                bu = None
            elif l[i] <= bu[0] + 0.1 * a[i] and c[i] > bu[0] and c[i] > o[i]:
                L[i], sl[i], bu = True, bu[0] - 0.5 * a[i], None
        if bd is not None and i - bd[1] >= 2:
            if i - bd[1] > 30 or c[i] > bd[0] + 0.5 * a[i]:
                bd = None
            elif h[i] >= bd[0] - 0.1 * a[i] and c[i] < bd[0] and c[i] < o[i]:
                S[i], ss[i], bd = True, bd[0] + 0.5 * a[i], None
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC422 / SC423 Average Daily Range ----------------
def s_adr_fade(df):
    """ADR exhaustion fade (FXStreet 'How to use the Average Daily Range'; forex-station / MQL5 ADR fade): once the day
    has used >= 100% of its 14-day ADR, a bar that makes the day's extreme and then a reversal close (back under the
    previous bar's low for a short) fades toward the day's middle. Stop 0.2 ATR beyond the day extreme, target the
    running day mid; flat at 21:00 UTC (the forex day end). Intraday, UTC days."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.adr(df)
    a = atr(h, l, c, 14)
    used = np.nan_to_num(d["used"])
    mid = (d["day_hi"] + d["day_lo"]) / 2
    newhi = _r(h) >= _r(d["day_hi"]) - 1e-12
    newlo = _r(l) <= _r(d["day_lo"]) + 1e-12
    hr = df.index.tz_convert("UTC").hour.to_numpy()
    S = (used >= 1.0) & newhi & (c < _r(l)) & (c < o) & (hr < 20)
    L = (used >= 1.0) & newlo & (c > _r(h)) & (c > o) & (hr < 20)
    L, S = _excl(L, S)
    return L, S, _plan_lv(d["day_lo"] - 0.2 * a, d["day_hi"] + 0.2 * a, mid, mid, force_exit=hr >= 21)


def s_orb_adr_room(df):
    """New York 15-minute opening range (09:30-09:45) breakout only while the day has used < 60% of its ADR at the break
    (the ADR 'room left' rule of FXStreet / NexusFi ADR guides): close through the range, stop at the other side, 2R,
    flat 16:00 NY, first break per day. Differs from SS01 by the ADR filter."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.adr(df)
    hr, day = _ny(df)
    n = len(c)
    bm = _bar_minutes(df)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    cur, rh, rl_, done = None, -np.inf, np.inf, False
    for i in range(n):
        if day[i] != cur:
            cur, rh, rl_, done = day[i], -np.inf, np.inf, False
        if 9.5 <= hr[i] and hr[i] + bm / 60.0 <= 9.75 + 1e-9:
            rh, rl_ = max(rh, h[i]), min(rl_, l[i])
            continue
        if done or not np.isfinite(rh) or hr[i] < 9.75 or hr[i] >= 15.5:
            continue
        room = np.isfinite(d["used"][i]) and d["used"][i] < 0.6
        if c[i] > rh and room:
            L[i], sl[i], done = True, rl_, True
        elif c[i] < rl_ and room:
            S[i], ss[i], done = True, rh, True
    return L, S, _plan_r(sl, ss, 2.0, force_exit=hr >= 16)


# ---------------- SC424 ICT CBDR standard-deviation fade ----------------
def s_cbdr_fade(df):
    """ICT Central Bank Dealers Range (Mentorship month 8; 'CBDR Deviations' open source): the 14:00-20:00 NY range
    projects the next day's extremes at 1-4 standard deviations (one SD = the range's height). Ours: from 20:00 NY
    (Asia) to 11:00 NY (London / NY morning), a bar that trades through the deepest of the +1 / +2 / +3 SD levels and
    closes back under it = short, stop 0.1 ATR over the next SD level or the bar high (the higher), target the CBDR
    high; mirror under the CBDR low. Flat 16:00 NY. CBDR range at least 0.25 ATR."""
    o, h, l, c, v = _ohlcv(df)
    cb = IND.cbdr(df)
    a = atr(h, l, c, 14)
    hr, _ = _ny(df)
    sd = cb["sd"]
    ok = ((hr >= 20) | (hr < 11)) & np.nan_to_num(sd > 0.25 * a)
    with np.errstate(invalid="ignore", divide="ignore"):
        ku = np.floor((h - cb["hi"]) / sd)
        kd = np.floor((cb["lo"] - l) / sd)
    ku, kd = np.nan_to_num(ku, nan=0.0), np.nan_to_num(kd, nan=0.0)
    u, d = cb["hi"] + ku * sd, cb["lo"] - kd * sd
    S = ok & (ku >= 1) & (ku <= 3) & (c < u)
    L = ok & (kd >= 1) & (kd <= 3) & (c > d)
    ss = np.maximum(cb["hi"] + (ku + 1) * sd, h) + 0.1 * a
    sl = np.minimum(cb["lo"] - (kd + 1) * sd, l) - 0.1 * a
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl, ss, cb["lo"], cb["hi"], force_exit=hr >= 16)


# ---------------- SC425 / SC428 Street Smarts (Raschke & Connors 1995) ----------------
def _rth_sessions(df):
    """New York cash sessions 09:30-16:00 on intraday bars: per-bar session number (-1 outside), and per session
    open / high / low / close and its first / last bar (a session is complete at its last bar's close)."""
    o, h, l, c, v = _ohlcv(df)
    hr, day = _ny(df)
    ins = (hr >= 9.5) & (hr < 16)
    sid = np.full(len(c), -1)
    keys, first, last = [], [], []
    for i in np.where(ins)[0]:
        if not keys or keys[-1] != day[i]:
            keys.append(day[i])
            first.append(i)
            last.append(i)
        sid[i] = len(keys) - 1
        last[-1] = i
    first, last = np.array(first, int), np.array(last, int)
    so = o[first] if len(first) else np.array([])
    sh = np.array([h[f:t + 1].max() for f, t in zip(first, last)])
    sl = np.array([l[f:t + 1].min() for f, t in zip(first, last)])
    sc = c[last] if len(last) else np.array([])
    return hr, sid, first, last, so, sh, sl, sc


def s_whiplash(df):
    """Whiplash (Raschke & Connors, 'Street Smarts' 1995, a daily S&P pattern; cryptodatadownload.com port): the session
    opens with a gap under the previous session's low and closes in the top half of its range (over its open) = buy
    the next session's open, sell at that session's close (as published); mirror for a gap over the previous high that
    closes in the lower half. Ours: sessions = New York cash hours 09:30-16:00 on intraday bars (the signal sits on the
    bar before the next 09:30 open, so the entry is that open), stop 0.1 ATR beyond the signal session's extreme,
    target 2R, flat 16:00 NY."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    hr, sid, first, last, so, sh, sl, sc = _rth_sessions(df)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stl, sts = np.full(n, np.nan), np.full(n, np.nan)
    for k in range(1, len(first) - 1):
        j = first[k + 1] - 1
        if j <= last[k]:
            continue
        rng = max(sh[k] - sl[k], 1e-12)
        pos = (sc[k] - sl[k]) / rng
        if so[k] < sl[k - 1] and pos > 0.5 and sc[k] > so[k]:
            L[j], stl[j] = True, sl[k] - 0.1 * a[j]
        elif so[k] > sh[k - 1] and pos < 0.5 and sc[k] < so[k]:
            S[j], sts[j] = True, sh[k] + 0.1 * a[j]
    return L, S, _plan_r(stl, sts, 2.0, force_exit=hr >= 16)


def s_turtle_soup_plus1(df):
    """Turtle Soup Plus One (Raschke & Connors, 'Street Smarts'): the previous bar CLOSED at a new 20-bar low, the
    prior 20-bar low having been set at least 3 bars before; this bar closes back over that earlier 20-bar low = buy;
    stop 0.1 ATR under the lower of the two bars' lows, target 2R (the book: trail, partial at 1-2 days). Mirror for
    the high. Differs from SS13 (Turtle Soup: same-bar re-entry, no close beyond required)."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(23, n):
        w = l[i - 21:i - 1]
        j = int(np.argmin(w)) + i - 21
        if c[i - 1] < w.min() and (i - 1) - j >= 3 and c[i] > w.min():
            L[i] = True
        w = h[i - 21:i - 1]
        j = int(np.argmax(w)) + i - 21
        if c[i - 1] > w.max() and (i - 1) - j >= 3 and c[i] < w.max():
            S[i] = True
    sl = np.minimum(l, _r(l)) - 0.1 * a
    ss = np.maximum(h, _r(h)) + 0.1 * a
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC429 / SC430 Joe Ross ----------------
def s_ross_123(df):
    """Joe Ross 1-2-3 reversal (tradingeducators.com; LuxAlgo '1-2-3 Reversal'): point 1 = a 30-bar swing low, 2 = the
    swing high after it, 3 = a higher swing low (all confirmed, k 3); buy on the close through point 2, stop 0.1 ATR
    under point 3, target 2R. Mirror for tops."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.ross_123(df)
    a = atr(h, l, c, 14)
    L, S = _excl(r["long_123"], r["short_123"])
    return L, S, _plan_r(r["p3_l"] - 0.1 * a, r["p3_s"] + 0.1 * a, 2.0)


def s_ross_hook(df):
    """Ross hook (Joe Ross; FXOpen 'How to Trade with a Ross Hook Pattern'): after the 1-2-3 break, the first bar that
    fails to make a new high forms the hook; buy on the close through the hook high, stop 0.1 ATR under the lowest low
    since the hook, target 2R. Mirror."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.ross_123(df)
    a = atr(h, l, c, 14)
    L, S = _excl(r["long_hook"], r["short_hook"])
    return L, S, _plan_r(r["hook_stop_l"] - 0.1 * a, r["hook_stop_s"] + 0.1 * a, 2.0)


# ---------------- SC431 / SC432 chart types ----------------
def s_three_line_break(df):
    """Three Line Break (Steve Nison, 'Beyond Candlesticks'; TradingView 'Three Line Break Backtest' open source): trade
    the reversal - a close beyond the extreme of the last 3 lines - in the EMA50's direction, out on the opposite
    reversal; stop 2 ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.three_line_break(df)
    e = ema(c, 50)
    a = atr(h, l, c, 14)
    L, S = _excl(t["flip_up"] & (c > e), t["flip_dn"] & (c < e))
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=t["flip_dn"], exit_short=t["flip_up"])


def s_pnf_double(df):
    """Point & Figure double-top buy / double-bottom sell (3-box reversal on closes, box 0.5 x ATR50 fixed when a column
    starts - our translation of a fixed box; LuxAlgo 'Point & Figure', StockCharts P&F): buy when an X column rises over
    the previous X column's top, stop 0.1 ATR under the current column's bottom (the column start), target 2R. Mirror."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.point_figure(df)
    a = atr(h, l, c, 14)
    L, S = _excl(p["dbl_top"], p["dbl_bot"])
    return L, S, _plan_r(p["col_lo"] - 0.1 * a, p["col_hi"] + 0.1 * a, 2.0)


# ---------------- SC433-SC440 Ehlers (TASC 2020 / 2025) ----------------
def s_ehlers_uo(df):
    """Ehlers 'The Ultimate Oscillator' (TASC 2025.04; Wealth-Lab rule in the Traders' Tips): buy when the UO crosses
    over zero, exit when it crosses back under; mirror for shorts (ours). Stop 2 ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    u = IND.ehlers_uo(df)["uo"]
    a = atr(h, l, c, 14)
    up, dn = cross_up(u, 0), cross_down(u, 0)
    L, S = _excl(up, dn)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=dn, exit_short=up)


def s_laguerre_filter(df):
    """Ehlers 'Laguerre Filters' (TASC 2025.07; Wealth-Lab rule): long when the UltimateSmoother(40) crosses over the
    Laguerre filter (gamma 0.8), exit when it crosses under; mirror for shorts (ours). Stop 2 ATR."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.laguerre_filter(df)
    a = atr(h, l, c, 14)
    up, dn = cross_up(q["us"], q["filt"]), cross_down(q["us"], q["filt"])
    L, S = _excl(up, dn)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=dn, exit_short=up)


def s_laguerre_osc(df):
    """Ehlers Laguerre oscillator (TASC 2025.07; Wealth-Lab rule): buy when the RMS-normalised oscillator turns up from
    under -2 (Wealth-Lab: between -2 and -3 SD), 3% stop as published, exit when the UltimateSmoother crosses under the
    Laguerre filter. Mirror (turn down from over +2) for shorts - ours."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.laguerre_filter(df)
    x = q["osc"]
    L = (_r(x) < -2) & (x > _r(x)) & (_r(x) <= _r(x, 2))
    S = (_r(x) > 2) & (x < _r(x)) & (_r(x) >= _r(x, 2))
    up, dn = cross_up(q["us"], q["filt"]), cross_down(q["us"], q["filt"])
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", c * 0.97, c * 1.03), "target": None, "exit_long": dn, "exit_short": up}


def s_continuation_index(df):
    """Ehlers 'The Continuation Index' (TASC 2025.09): CI over +0.5 = a strong trend that should continue. The article
    gives no trade rule - ours: buy when CI crosses over +0.5, out when it falls under 0; mirror under -0.5. Stop 2
    ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    ci = IND.continuation_index(df)
    a = atr(h, l, c, 14)
    L, S = _excl(cross_up(ci, 0.5), cross_down(ci, -0.5))
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=ci < 0, exit_short=ci > 0)


def s_one_euro(df):
    """Ehlers 'The One Euro Filter' (TASC 2025.12): the oscillator (filter of HP54) crossing zero while the One Euro
    filter of the close slopes the same way. No rule in the article - ours: 2 ATR stop, 2R target."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.one_euro(df)
    a = atr(h, l, c, 14)
    f = q["filt"]
    L = cross_up(q["osc"], 0) & (f > _r(f))
    S = cross_down(q["osc"], 0) & (f < _r(f))
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, 2.0)


def s_pma_turn(df):
    """Ehlers 'Removing Moving Average Lag' (TASC 2025.03; Wealth-Lab rule): enter when the projected moving average
    (PMA 20) turns up, exit when it turns down; mirror for shorts (ours). Stop 2 ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.pma_predict(df)["pma"]
    a = atr(h, l, c, 14)
    tu = (p > _r(p)) & (_r(p) <= _r(p, 2))
    td = (p < _r(p)) & (_r(p) >= _r(p, 2))
    L, S = _excl(tu, td)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=td, exit_short=tu)


def s_pma_predict(df):
    """Ehlers 'Removing Moving Average Lag' (TASC 2025.03; the Python listing's signal): long while Predict > PMA - ours:
    enter on the cross over, out on the cross under; mirror. Stop 2 ATR."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.pma_predict(df)
    a = atr(h, l, c, 14)
    up, dn = cross_up(q["predict"], q["pma"]), cross_down(q["predict"], q["pma"])
    L, S = _excl(up, dn)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=dn, exit_short=up)


def s_trendflex(df):
    """Ehlers Reflex / Trendflex (TASC 2020.02, 'Reflex: A New Zero-Lag Indicator'): Trendflex crosses zero (trend) while
    Reflex (cycle) is on the same side; out when Trendflex crosses back. Ehlers gives the zero crossings as the
    signals; the confirmation, 2 ATR stop and exit are ours."""
    o, h, l, c, v = _ohlcv(df)
    q = IND.reflex_trendflex(df)
    a = atr(h, l, c, 14)
    t, r = q["trendflex"], q["reflex"]
    L = cross_up(t, 0) & (r > 0)
    S = cross_down(t, 0) & (r < 0)
    L, S = _excl(L, S)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=t < 0, exit_short=t > 0)


# ---------------- SC441 / SC442 Zarattini ORB for stocks in play ----------------
def _zarattini_orb(df, min_rvol):
    o, h, l, c, v = _ohlcv(df)
    d = IND.adr(df)
    rv = IND.rvol(df)["bar_rvol"]
    hr, day = _ny(df)
    n = len(c)
    novol = np.nan_to_num(_s(v).rolling(20, min_periods=2).std().to_numpy(), nan=0.0) <= 1e-9 * (np.abs(v) + 1)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    cur, fb, done = None, None, False
    for i in range(n):
        if day[i] != cur:
            cur, fb, done = day[i], None, False
        if abs(hr[i] - 9.5) < 1e-9:
            ok = novol[i] or (np.isfinite(rv[i]) and rv[i] >= min_rvol)
            fb = (o[i], h[i], l[i], c[i]) if ok else None
            continue
        if fb is None or done or hr[i] >= 15.5 or not np.isfinite(d["adr"][i]):
            continue
        risk = 0.1 * d["adr"][i]
        if fb[3] > fb[0] and c[i] > fb[1]:
            L[i], sl[i], done = True, c[i] - risk, True
        elif fb[3] < fb[0] and c[i] < fb[2]:
            S[i], ss[i], done = True, c[i] + risk, True
    return L, S, {"stop": ("level", sl, ss), "target": None, "force_exit": hr >= 16}


def s_zarattini_orb(df):
    """Zarattini, Barbon & Aziz, 'A Profitable Day Trading Strategy for the U.S. Equity Market' (SSRN / Swiss Finance
    Institute 2024; QuantConnect 'Opening Range Breakout for Stocks in Play'): the first 5-minute candle from 09:30 NY
    sets the direction (bullish -> only longs, bearish -> only shorts); entry when price breaks its high (low) - ours on
    the close through it, next open; stop 10% of the 14-day ATR (ours: the 14-day average range, agents/indicators
    adr) from the entry; no target, flat at 16:00 NY. 'Stocks in play' = relative volume of that first candle >= 1
    (ours: the bar's volume vs the same 09:30 slot over 20 sessions; feeds without volume pass). Built for 5-minute
    bars (on other frames the 09:30 bar stands in for the 5-minute candle)."""
    return _zarattini_orb(df, 1.0)


def s_zarattini_orb_rv2(df):
    """SC441 with a strict 'in play' filter: first-candle relative volume >= 2 (the paper ranks by RVOL; the top-20
    names typically exceed 2x)."""
    return _zarattini_orb(df, 2.0)


# ---------------- SC443 ICT macro FVG ----------------
def s_macro_fvg(df):
    """ICT macros (LuxAlgo 'ICT Macros', TradingFinder 'ICT Macro Times'): the algorithm's 20-30 minute delivery windows
    (02:33-03:00, 04:03-04:30, 08:50-09:10, 09:50-10:10, 10:50-11:10, 11:50-12:10, 13:10-13:40, 15:15-15:45 NY). Only a
    fair value gap whose third candle closes inside a macro is traded: return into the gap and close back out of it,
    stop 0.1 ATR beyond the gap, 2R (the library's FVG entry, TV10 / SC250). Best on 1-5 minute bars."""
    m = IND.ict_macros(df)["in_macro"]
    L, S, sl, ss = _fvg_return(df, m, m)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC444 Fractal Chaos Bands ----------------
def s_fractal_bands(df):
    """Fractal Chaos Bands (Edward Dobson): the last confirmed fractal high / low as bands; a close over the upper band
    with the EMA50 rising = trend breakout (mirror). Stop at the lower band when it is 0.5-3 ATR away, else 2 ATR;
    target 2R. Differs from SC002 (no Alligator, fractal bands only)."""
    o, h, l, c, v = _ohlcv(df)
    fb = IND.fractal_bands(df)
    e = ema(c, 50)
    a = atr(h, l, c, 14)
    L = cross_up(c, _r(fb["upper"])) & (e > _r(e, 3))
    S = cross_down(c, _r(fb["lower"])) & (e < _r(e, 3))
    dl, ds = c - fb["lower"], fb["upper"] - c
    sl = np.where((dl >= 0.5 * a) & (dl <= 3 * a), fb["lower"], c - 2 * a)
    ss = np.where((ds >= 0.5 * a) & (ds <= 3 * a), fb["upper"], c + 2 * a)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- SC445 Kairi extreme reversion ----------------
def s_kairi_revert(df):
    """Kairi Relative Index 25 (Japanese classic; Investopedia 'Kairi'): an extreme deviation from the SMA25 reverts.
    Ours (the classic fixed % thresholds depend on the market): Kairi under -2.5 x its 200-bar standard deviation, then
    a close back over that line = buy, target the SMA25, stop 1.5 ATR; mirror."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.kairi(df)
    sd = _s(k).rolling(200, min_periods=50).std().to_numpy()
    m = sma(c, 25)
    a = atr(h, l, c, 14)
    L = cross_up(k, -2.5 * sd)
    S = cross_down(k, 2.5 * sd)
    L, S = _excl(L, S)
    return L, S, _plan_lv(c - 1.5 * a, c + 1.5 * a, m, m)


# ---------------- SC446-SC454 the new TikTok explorer ideas (data/agents/scout/tiktok_explore.json, 15.11) ----------------
def s_ob_twice_breakaway(df):
    """Explorer idea TT-s-gap-ictchartist (ICT & SMC CHARTIST, TikTok): a higher-timeframe order block touched twice,
    then a breakaway gap away from it plus an inverted FVG -> sell the bearish FVG (buy the bullish mirror), stop over
    the high, target the old low. Ours: the block = the latest live order block from swings k 5 (agents/indicators smc; run
    the strategy on the higher chart for the 'HTF' block), two separate touches (bars entering the block after leaving it), then a breakaway gap (candle 3 closes beyond
    candle 2, agents/indicators gap_type) in the direction away from the block with an iFVG inverted in that direction
    in the last 20 bars; entry on the return into that gap with a close back out, stop 0.1 ATR beyond the order block,
    target 2R (the 'old low' is not defined in the caption). Timeframes not stated: HTF 1h/4h -> ours one chart."""
    o, h, l, c, v = _ohlcv(df)
    sm = IND.smc(df, k=5)
    gt = IND.gap_type(df)
    iv = IND.ifvg(df)
    a = atr(h, l, c, 14)
    n = len(c)
    ib = _s(iv["inv_bull"].astype(float)).rolling(20, min_periods=1).max().to_numpy() > 0
    ibr = _s(iv["inv_bear"].astype(float)).rolling(20, min_periods=1).max().to_numpy() > 0
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    st_s = {"key": None, "t": 0, "in": False, "gap": None}
    st_b = {"key": None, "t": 0, "in": False, "gap": None}
    for i in range(n):
        bt, bb = sm["ob_bear_top"][i], sm["ob_bear_bot"][i]
        if np.isfinite(bt):
            if st_s["key"] != (bt, bb):
                st_s.update(key=(bt, bb), t=0, **{"in": False}, gap=None)
            inside = h[i] >= bb
            if inside and not st_s["in"]:
                st_s["t"] += 1
            st_s["in"] = inside
            g = st_s["gap"]
            if g is not None:
                if c[i] > g[0] or i - g[2] > 30:
                    st_s["gap"] = None
                elif i > g[2] and h[i] >= g[1] and c[i] < g[1] and c[i] < o[i]:
                    S[i], ss[i] = True, bt + 0.1 * a[i]
                    st_s["gap"] = None
            if st_s["t"] >= 2 and gt["bear_brk"][i] and gt["top"][i] < bb and ibr[i]:
                st_s["gap"] = (gt["top"][i], gt["bot"][i], i)
        else:
            st_s.update(key=None, t=0, gap=None)
        bt, bb = sm["ob_bull_top"][i], sm["ob_bull_bot"][i]
        if np.isfinite(bt):
            if st_b["key"] != (bt, bb):
                st_b.update(key=(bt, bb), t=0, **{"in": False}, gap=None)
            inside = l[i] <= bt
            if inside and not st_b["in"]:
                st_b["t"] += 1
            st_b["in"] = inside
            g = st_b["gap"]
            if g is not None:
                if c[i] < g[1] or i - g[2] > 30:
                    st_b["gap"] = None
                elif i > g[2] and l[i] <= g[0] and c[i] > g[0] and c[i] > o[i]:
                    L[i], sl[i] = True, bb - 0.1 * a[i]
                    st_b["gap"] = None
            if st_b["t"] >= 2 and gt["bull_brk"][i] and gt["bot"][i] > bt and ib[i]:
                st_b["gap"] = (gt["top"][i], gt["bot"][i], i)
        else:
            st_b.update(key=None, t=0, gap=None)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


def _valid_ob_entry(df, mode):
    o, h, l, c, v = _ohlcv(df)
    q = IND.valid_ob(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    used_b = used_s = None
    for i in range(1, n):
        bh, blo = q["ob_bear_hi"][i - 1], q["ob_bear_lo"][i - 1]
        if np.isfinite(bh) and (bh, blo) != used_s:
            lv = {"wick": blo, "body": q["ob_bear_open"][i - 1], "mid": q["ob_bear_mid"][i - 1]}[mode]
            if h[i] >= lv and c[i] < lv and c[i] <= bh:
                S[i], ss[i], used_s = True, bh + 0.1 * a[i], (bh, blo)
        bh, blo = q["ob_bull_hi"][i - 1], q["ob_bull_lo"][i - 1]
        if np.isfinite(bh) and (bh, blo) != used_b:
            lv = {"wick": bh, "body": q["ob_bull_open"][i - 1], "mid": q["ob_bull_mid"][i - 1]}[mode]
            if l[i] <= lv and c[i] > lv and c[i] >= blo:
                L[i], sl[i], used_b = True, blo - 0.1 * a[i], (bh, blo)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


def s_valid_ob_wick(df):
    """Explorer idea TT-s-obvalid-ictchartist (ICT & SMC CHARTIST, TikTok): an order block counts only if (1) it swept
    the previous swing high, (2) the move away left an FVG, (3) price closed below the last low (BMS) - mirror for
    bullish (agents/indicators valid_ob, swings k 3). Entry at the block's near wick (first touch), on a close back
    away from it; stop 0.1 ATR beyond the block's far wick; target 2R (not in the caption). One trade per block."""
    return _valid_ob_entry(df, "wick")


def s_valid_ob_body(df):
    """TT-s-obvalid-ictchartist, entry at the order-block candle's open (the 'body' entry); as SC447 otherwise."""
    return _valid_ob_entry(df, "body")


def s_valid_ob_mid(df):
    """TT-s-obvalid-ictchartist, entry at the block's 50% (mean threshold); as SC447 otherwise."""
    return _valid_ob_entry(df, "mid")


def s_fvg_true_gap(df):
    """Explorer idea TT-s-fvg-vs-breakaway (Adem OnChain, TikTok; TradingView 'FVG Breakaway/3rd Candle (Arjo)'): when
    candle 3 closes INSIDE candle 2's range the gap is a fair value gap and a retest comes first - only those gaps are
    traded on the return (close back out of the gap, stop 0.1 ATR beyond it, 2R, library FVG entry)."""
    gt = IND.gap_type(df)
    L, S, sl, ss = _fvg_return(df, gt["bull_fvg"], gt["bear_fvg"])
    return L, S, _plan_r(sl, ss, 2.0)


def s_breakaway_go(df):
    """Explorer idea TT-s-fvg-vs-breakaway: when candle 3 closes BEYOND candle 2's extreme it is a breakaway gap - a
    strong move without a retest, so do not wait: enter on the next open in the gap's direction, stop 0.1 ATR beyond
    candle 1 (the gap's far side), target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    gt = IND.gap_type(df)
    a = atr(h, l, c, 14)
    L, S = _excl(gt["bull_brk"], gt["bear_brk"])
    return L, S, _plan_r(gt["bot"] - 0.1 * a, gt["top"] + 0.1 * a, 2.0)


def _graded_sweep(df, g):
    o, h, l, c, v = _ohlcv(df)
    q = IND.swing_grade(df)
    a = atr(h, l, c, 14)
    S = (q["sweep_hi_grade"] >= g) & (c < q["sweep_hi_lvl"])
    L = (q["sweep_lo_grade"] >= g) & (c > q["sweep_lo_lvl"])
    L, S = _excl(L, S)
    return L, S, _plan_r(l - 0.1 * a, h + 0.1 * a, 2.0)


def s_sweep_grade2(df):
    """Explorer idea TT-s-swing-significance (itsnathanoreilly, TikTok: 'not every swing high is equally important for a
    liquidity sweep'): sweep-and-close-back reversal only when the swept swing grades >= 2 (agents/indicators
    swing_grade: +1 30-bar extreme, +1 equal highs, +1 prior-day level, +1 age 50+ bars); stop 0.1 ATR beyond the
    sweep bar, 2R."""
    return _graded_sweep(df, 2)


def s_sweep_grade3(df):
    """TT-s-swing-significance with the stricter threshold: grade >= 3."""
    return _graded_sweep(df, 3)


def s_po3_4h(df):
    """Explorer idea TT-s-po3-4h-ttrades (TTrades, 'Trading the 4 Hour Power Of Three - OHLC'): every New-York-aligned 4H
    candle = accumulation at its open, manipulation beyond the open that takes the previous 4H candle's low (high),
    distribution the other way. Entry on the lower chart: after the run under the 4H open has taken the previous 4H
    low, a close back over the 4H open = buy; stop 0.1 ATR under the manipulation low, target the previous 4H high
    when it is >= 1R away (else 2R - ours), flat on the 4H candle's last bar; one trade per 4H candle. Mirror.
    Differs from SC307 (daily PO3 on the NY midnight open)."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.po3_4h(df)
    a = atr(h, l, c, 14)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    tl, ts = np.full(n, np.nan), np.full(n, np.nan)
    done = -1
    for i in range(1, n):
        b = p["block"][i]
        if b == done or p["last_bar"][i] or not np.isfinite(p["prev_lo"][i]):
            continue
        op = p["open4"][i]
        if p["lo4"][i] <= p["prev_lo"][i] and p["lo4"][i] < op and c[i] > op and c[i - 1] <= op and p["block"][i - 1] == b:
            st = p["lo4"][i] - 0.1 * a[i]
            r = c[i] - st
            L[i], sl[i], done = True, st, b
            tl[i] = p["prev_hi"][i] if p["prev_hi"][i] - c[i] >= r else c[i] + 2 * r
        elif p["hi4"][i] >= p["prev_hi"][i] and p["hi4"][i] > op and c[i] < op and c[i - 1] >= op and p["block"][i - 1] == b:
            st = p["hi4"][i] + 0.1 * a[i]
            r = st - c[i]
            S[i], ss[i], done = True, st, b
            ts[i] = p["prev_lo"][i] if c[i] - p["prev_lo"][i] >= r else c[i] - 2 * r
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl, ss, tl, ts, force_exit=p["last_bar"])


SCOUT.update({
    "SC406": ("CM Williams Vix Fix: fear spike ends + close over the prior high (inverse for tops), 5-bar stop, 2R", s_vix_fix, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC407": ("Bollinger BBTrend crosses zero with the close past the 20 middle, out on the cross back, 2 ATR", s_bbtrend, _M),
    "SC408": ("Khalil VZO trend: ADX > 18 + EMA60 side, VZO back over 0 after < -40, zone exits, 2 ATR", s_vzo_trend, _M),
    "SC409": ("Khalil VZO range: ADX < 18, VZO back over -40 (under +40), out at the opposite 40, 2 ATR", s_vzo_range, _F),
    "SC410": ("Khalil Price Zone Oscillator range: ADX < 18, back over -40 / under +40, out at the opposite 40", s_pzo_range, _F),
    "SC411": ("Twiggs Money Flow 21 zero cross after 10 bars on the other side, rising EMA50 side, 2 ATR, 2R", s_twiggs_mf, _M),
    "SC412": ("Relative Momentum Index 14/5 back over 30 / under 70 on the EMA200 side, 1.5 ATR, 2R (Altman)", s_rmi_reversal, _F),
    "SC413": ("DecisionPoint PMO signal cross under zero (over zero for shorts), opposite-cross exit, 2 ATR, 2R", s_pmo_cross, _M),
    "SC414": ("Tillson T3(8) / T3(21) cross, out on the cross back, 2 ATR", s_t3_cross, _M),
    "SC415": ("Absolute Strength Histogram 9/3 bulls / bears cross, reverse-cross exit, 2 ATR", s_ash_cross, _M),
    "SC416": ("Trendilo ALMA-50 % change over / under its RMS band, out through zero, 2 ATR (dudeowns)", s_trendilo, _M),
    "SC417": ("Darvas box breakout / breakdown, stop past the box, 2R", s_darvas, ["15Min", "1Hour", "4Hour", "1Day"]),
    "SC418": ("VSA no-supply (no-demand) test in the trend, close past its high (low) within 2 bars, 2R", s_vsa_test, _F),
    "SC419": ("VSA upthrust / spring on high volume, next close past its low (high), bar stop, 2R", s_vsa_upthrust, _F),
    "SC420": ("Round '00' level rejection (unbroken 20 bars, wick >= body), stop 0.5 ATR past the level, 2R", s_round_reject, ["5Min", "15Min", "1Hour"]),
    "SC421": ("Round '00' level break and retest in 2-30 bars, stop 0.5 ATR past the level, 2R", s_round_retest, ["5Min", "15Min", "1Hour"]),
    "SC422": ("ADR exhaustion fade: >= 100% of ADR14 used, day-extreme reversal close, target the day mid, flat 21 UTC", s_adr_fade, ["5Min", "15Min"]),
    "SC423": ("NY 15-min ORB only while < 60% of ADR14 is used, range stop, 2R, flat 16:00 NY", s_orb_adr_room, ["5Min", "15Min"]),
    "SC424": ("ICT CBDR: 20-11 NY rejection of the deepest CBDR +/- 1-3 SD level, stop past the next SD, target the CBDR edge, flat 16:00", s_cbdr_fade, ["5Min", "15Min"]),
    "SC425": ("Street Smarts Whiplash (NY cash sessions): gap past the prior extreme, close in the other half, next session, 2R", s_whiplash, ["5Min", "15Min", "1Hour"]),
    "SC428": ("Street Smarts Turtle Soup Plus One: close at a new 20-bar low, next close back over the old low, 2R", s_turtle_soup_plus1, _M + ["1Day"]),
    "SC429": ("Joe Ross 1-2-3 reversal: close through point 2, stop under point 3, 2R", s_ross_123, _F + ["4Hour"]),
    "SC430": ("Joe Ross hook after the 1-2-3 break: close through the hook, stop under the pullback, 2R", s_ross_hook, _F + ["4Hour"]),
    "SC431": ("Three Line Break reversal on the EMA50 side, out on the opposite reversal, 2 ATR (Nison)", s_three_line_break, _M),
    "SC432": ("Point & Figure double-top buy / double-bottom sell, 3-box reversal, box 0.5 ATR50, column stop, 2R", s_pnf_double, _M),
    "SC433": ("Ehlers Ultimate Oscillator 2025 zero cross, out on the cross back, 2 ATR (TASC 2025.04)", s_ehlers_uo, _M),
    "SC434": ("Ehlers UltimateSmoother 40 cross of the Laguerre filter 0.8, cross-back exit, 2 ATR (TASC 2025.07)", s_laguerre_filter, _M),
    "SC435": ("Ehlers Laguerre oscillator turn from beyond -2 / +2 SD, 3% stop, filter-cross exit (TASC 2025.07)", s_laguerre_osc, _M),
    "SC436": ("Ehlers Continuation Index through +/-0.5, out through 0, 2 ATR (TASC 2025.09)", s_continuation_index, _M),
    "SC437": ("Ehlers One Euro oscillator zero cross with the filter slope, 2 ATR, 2R (TASC 2025.12)", s_one_euro, _M),
    "SC438": ("Ehlers projected MA 20 turns up / down, out on the opposite turn, 2 ATR (TASC 2025.03)", s_pma_turn, _M),
    "SC439": ("Ehlers PMA predict line crosses the PMA, cross-back exit, 2 ATR (TASC 2025.03)", s_pma_predict, _M),
    "SC440": ("Ehlers Trendflex zero cross confirmed by Reflex, out through zero, 2 ATR (TASC 2020.02)", s_trendflex, _M),
    "SC441": ("Zarattini ORB: first 5-min candle's direction, break of it, stop 10% of daily ATR, flat 16:00, RVOL >= 1", s_zarattini_orb, ["5Min"]),
    "SC442": ("Zarattini ORB with first-candle RVOL >= 2 (stocks in play), stop 10% of daily ATR, flat 16:00", s_zarattini_orb_rv2, ["5Min"]),
    "SC443": ("ICT macro FVG: gap formed inside a macro window, return entry, gap stop, 2R", s_macro_fvg, ["1Min", "5Min"]),
    "SC444": ("Fractal Chaos Bands breakout with the EMA50 slope, band / 2 ATR stop, 2R", s_fractal_bands, _M),
    "SC445": ("Kairi 25 beyond 2.5 sd of itself and back, target the SMA25, 1.5 ATR stop", s_kairi_revert, _F),
    "SC446": ("Order block touched twice -> breakaway gap + iFVG away -> trade the gap's return, OB stop, 2R (TikTok idea)", s_ob_twice_breakaway, ["5Min", "15Min"]),
    "SC447": ("Valid order block (sweep + FVG + BMS), entry at the near wick, far-wick stop, 2R (TikTok idea)", s_valid_ob_wick, ["5Min", "15Min", "1Hour"]),
    "SC448": ("Valid order block (sweep + FVG + BMS), entry at the OB open, 2R (TikTok idea)", s_valid_ob_body, ["5Min", "15Min", "1Hour"]),
    "SC449": ("Valid order block (sweep + FVG + BMS), entry at the 50%, 2R (TikTok idea)", s_valid_ob_mid, ["5Min", "15Min", "1Hour"]),
    "SC450": ("FVG return only when candle 3 closes inside candle 2 (a true FVG, not breakaway), 2R (TikTok idea)", s_fvg_true_gap, ["1Min", "5Min", "15Min"]),
    "SC451": ("Breakaway gap (candle 3 closes past candle 2): enter at once, stop past candle 1, 2R (TikTok idea)", s_breakaway_go, ["1Min", "5Min", "15Min"]),
    "SC452": ("Sweep reversal only of a significant swing (grade >= 2 of 5), sweep-bar stop, 2R (TikTok idea)", s_sweep_grade2, ["5Min", "15Min"]),
    "SC453": ("Sweep reversal only of a significant swing (grade >= 3 of 5), sweep-bar stop, 2R (TikTok idea)", s_sweep_grade3, ["5Min", "15Min"]),
    "SC454": ("TTrades 4H Power of Three: 4H open run takes the prior 4H extreme, close back over the open, flat at 4H end", s_po3_4h, ["5Min", "15Min"]),
})


# ---------------- 15.11.2026 forced weekend run: SC455-SC458 ----------------
def s_synthetic_osc(df):
    """Ehlers 'A Synthetic Oscillator' (TASC 2026.04, RealTest / Wealth-Lab rule in the Traders' Tips): ROC2 = 1-bar
    change of Hann(Synth, 4); long when ROC2 crosses over 0, out (and short, RealTest's mirror) when it crosses under.
    The article has no stop - ours 2 ATR, no fixed target."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.synthetic_osc(df)["roc2"]
    a = atr(h, l, c, 14)
    up, dn = cross_up(r, 0), cross_down(r, 0)
    L, S = _excl(up, dn)
    return L, S, _atr_plan(c, a, 2.0, None, exit_long=dn, exit_short=up)


def s_ultimate_channel(df):
    """Ehlers Ultimate Channel (TASC 2024.05): a close over the upper channel (UltimateSmoother 20 + 1 x smoothed true
    range) with a rising basis = breakout, stop at the basis (or 1 ATR if nearer than 0.5 ATR), 2R; mirror. The
    article gives no rules - Keltner-style breakout use is ours (it replaces Keltner's EMA / ATR)."""
    o, h, l, c, v = _ohlcv(df)
    u = IND.ultimate_channel(df)
    a = atr(h, l, c, 14)
    b = u["basis"]
    L = cross_up(c, u["ch_up"]) & (b > _r(b))
    S = cross_down(c, u["ch_dn"]) & (b < _r(b))
    sl = np.where(c - b >= 0.5 * a, b, c - a)
    ss = np.where(b - c >= 0.5 * a, b, c + a)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


def s_ultimate_bands(df):
    """Ehlers Ultimate Bands (TASC 2024.05): a close back inside the bands after a close outside (Bollinger-style
    reversion with the low-lag UltimateSmoother basis), target the basis, stop 0.1 ATR beyond the extreme of the
    last 3 bars. Rules ours (the article gives none)."""
    o, h, l, c, v = _ohlcv(df)
    u = IND.ultimate_channel(df, band_mult=2.0)
    a = atr(h, l, c, 14)
    L = (_r(c) < _r(u["bd_dn"])) & (c > u["bd_dn"])
    S = (_r(c) > _r(u["bd_up"])) & (c < u["bd_up"])
    sl, ss = _swl(h, l, a, 3)
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl, ss, u["basis"], u["basis"])


def s_buyable_gap_up(df):
    """Buyable Gap-Up (Kacher & Morales, 'Trade Like an O'Neil Disciple', 2010): the session opens at least 0.75 x the
    40-session ATR over the previous session's close (chartmill 'Buyable Gap Ups (ATR based)') on volume >= 1.5 x
    the 50-session average; buy (Kacher: on the
    gap day; ours: the next session's 09:30 open, once the gap day's volume is known), stop at the gap day's intraday
    low (their rule), target 2R (ours - they hold while it acts well). Sessions = New York cash hours 09:30-16:00;
    feeds without volume pass the volume test. Mirror (a sellable gap-down) for shorts - ours. Differs from GP01
    (reopening gap > 1.5 ATR, no volume, ATR stop)."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    hr, sid, first, last, so, sh, sl_, sc = _rth_sessions(df)
    n, m = len(c), len(first)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    stl, sts = np.full(n, np.nan), np.full(n, np.nan)
    if m < 5:
        return L, S, _plan_r(stl, sts, 2.0)
    sv = np.array([v[f:t + 1].sum() for f, t in zip(first, last)])
    pc = np.concatenate([[sc[0]], sc[:-1]])
    tr_ = np.maximum(sh, pc) - np.minimum(sl_, pc)
    satr = _s(tr_).rolling(40, min_periods=5).mean().to_numpy()
    avgv = _s(sv).shift(1).rolling(50, min_periods=5).mean().to_numpy()
    novol = np.nanstd(sv[max(0, m - 50):]) <= 1e-9 * (np.nanmean(sv) + 1)
    for k in range(1, m - 1):
        j = first[k + 1] - 1
        if j <= last[k] or not np.isfinite(satr[k - 1]):
            continue
        vol_ok = novol or (np.isfinite(avgv[k]) and sv[k] >= 1.5 * avgv[k])
        if so[k] >= sc[k - 1] + 0.75 * satr[k - 1] and vol_ok:
            L[j], stl[j] = True, sl_[k]
        elif so[k] <= sc[k - 1] - 0.75 * satr[k - 1] and vol_ok:
            S[j], sts[j] = True, sh[k]
    return L, S, _plan_r(stl, sts, 2.0)


SCOUT.update({
    "SC455": ("Ehlers Synthetic Oscillator: Hann(4) ROC crosses zero, out on the opposite cross, 2 ATR (TASC 2026.04)", s_synthetic_osc, _M),
    "SC456": ("Ehlers Ultimate Channel breakout with the basis slope, basis stop, 2R (TASC 2024.05)", s_ultimate_channel, _M),
    "SC457": ("Ehlers Ultimate Bands (2 x RMS dev) close back inside, target the basis, 3-bar stop (TASC 2024.05)", s_ultimate_bands, _F),
    "SC458": ("Kacher-Morales buyable gap-up (>= 0.75 ATR40, 1.5x volume), next session open, gap-day-low stop, 2R", s_buyable_gap_up, ["5Min", "15Min", "1Hour"]),
})


# =====================================================================================================================
# scout 16.11.2026 (night): SC459-SC481 - TA-Lib / Bulkowski candlestick patterns, divergence for many indicators
# (LonesomeTheBlue), Ehlers TASC 2026 (Reversion Index, Adaptive SuperSmoother, AutoTune), DeMark TD REI / TD Pressure,
# Market Profile unfinished business (naked POC, poor highs / lows, single prints), the Taylor 3-day cycle, the
# Zarattini-Aziz-Barbon noise-area intraday momentum and IPF pivot-stop breakouts. Indicators: agents/indicators
# candles.py, divergence.py, ehlers4.py, demark2.py, auction.py, daycycle.py.
# =====================================================================================================================
def _cdl_trend(c, k=3):
    """Prior trend for a reversal pattern starting k bars back: the close k bars ago under (over) the EMA20 then."""
    e = ema(c, 20)
    return _r(c, k) < _r(e, k), _r(c, k) > _r(e, k)


def _cdl_rev(df, name, k=3):
    o, h, l, c, v = _ohlcv(df)
    p = IND.candle_patterns(df)
    a = atr(h, l, c, 14)
    dn, up = _cdl_trend(c, k)
    L = (p[name] == 1) & dn
    S = (p[name] == -1) & up
    L, S = _excl(L, S)
    return L, S, _plan_r(p["pattern_low"] - 0.1 * a, p["pattern_high"] + 0.1 * a, 2.0)


def s_cdl_stars(df):
    """Morning star / evening star (TA-Lib CDLMORNINGSTAR / CDLEVENINGSTAR, penetration 0.3) or abandoned baby
    (CDLABANDONEDBABY, shadow gaps) after a move down (up): the close 3 bars ago under (over) its EMA20. Bulkowski
    ranks the evening star / abandoned babies among the most reliable reversals
    (https://www.thepatternsite.com/CandlePerformers.html). Stop 0.1 ATR beyond the 3-bar pattern, target 2R (the
    library standard - the books give no stop). No-gap feeds: the star only needs a body at / beyond the first close."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.candle_patterns(df)
    a = atr(h, l, c, 14)
    dn, up = _cdl_trend(c, 3)
    st = p["morning_star"] + p["abandoned_baby"]
    L, S = _excl((st > 0) & dn, (st < 0) & up)
    return L, S, _plan_r(p["pattern_low"] - 0.1 * a, p["pattern_high"] + 0.1 * a, 2.0)


def s_cdl_three_outside(df):
    """Three outside up / down (TA-Lib CDL3OUTSIDE: an engulfing candle and a third candle closing beyond it) after a
    move down (up), stop 0.1 ATR beyond the pattern, target 2R (ours). Bulkowski: three outside down / up are among his
    top-performing reversal candles. Differs from the plain engulfing strategies (needs the 3rd-candle confirmation)."""
    return _cdl_rev(df, "three_outside", 3)


def s_cdl_three_inside(df):
    """Three inside up / down (TA-Lib CDL3INSIDE: a long candle, a small opposite harami inside its body, a third
    candle closing beyond the first candle's open) after a move down (up); stop beyond the pattern, 2R (ours)."""
    return _cdl_rev(df, "three_inside", 3)


def s_cdl_piercing(df):
    """Piercing line / dark cloud cover (TA-Lib CDLPIERCING / CDLDARKCLOUDCOVER: a long candle, then a long opposite one
    opening at / beyond its close and closing past the middle of its body but not past its open) after a move down
    (up); stop beyond the 2 bars, target 2R (ours)."""
    return _cdl_rev(df, "piercing", 2)


def s_cdl_three_line_strike(df):
    """Three-line strike traded as Bulkowski measured it (https://www.thepatternsite.com/ThreeLineStrikeBull.html):
    three rising white candles 'struck' by a black candle that opens at / over the 3rd close and closes under the 1st
    open - the 'bullish' strike breaks DOWN most of the time, so we sell with the 4th candle (and buy the bearish
    strike's white 4th candle). Stop 0.1 ATR beyond the 4th candle's extreme, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.candle_patterns(df)
    a = atr(h, l, c, 14)
    L, S = _excl(p["three_line_strike"] == -1, p["three_line_strike"] == 1)
    return L, S, _plan_r(l - 0.1 * a, h + 0.1 * a, 2.0)


def s_cdl_tweezer_extreme(df):
    """Tweezer bottom / top (equal lows / highs within 5% of the 10-bar average range, a black then a white candle) that
    is the 20-bar low (high), confirmed by the next close over the tweezer's high (under its low) within 3 bars; stop
    0.1 ATR beyond the tweezer, target 2R (ours; Bulkowski: tweezers need the extreme location to work)."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.candle_patterns(df)
    a = atr(h, l, c, 14)
    lo2, hi2 = np.minimum(l, _r(l)), np.maximum(h, _r(h))
    tb = (p["tweezer"] == 1) & (lo2 <= _s(l).rolling(20, min_periods=5).min().to_numpy() + 1e-12)
    tt = (p["tweezer"] == -1) & (hi2 >= _s(h).rolling(20, min_periods=5).max().to_numpy() - 1e-12)
    n = len(c)
    L, S = np.zeros(n, bool), np.zeros(n, bool)
    sl, ss = np.full(n, np.nan), np.full(n, np.nan)
    for j in np.where(tb)[0]:
        hi, lo = max(h[j], h[j - 1]), min(l[j], l[j - 1])
        for t in range(j + 1, min(n, j + 4)):
            if l[t] < lo:
                break
            if c[t] > hi:
                L[t], sl[t] = True, lo - 0.1 * a[t]
                break
    for j in np.where(tt)[0]:
        hi, lo = max(h[j], h[j - 1]), min(l[j], l[j - 1])
        for t in range(j + 1, min(n, j + 4)):
            if h[t] > hi:
                break
            if c[t] < lo:
                S[t], ss[t] = True, hi + 0.1 * a[t]
                break
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


# ---------------- divergence (LonesomeTheBlue 'Divergence for Many Indicators') ----------------
def s_rsi_hidden_div(df):
    """RSI(14) HIDDEN bullish divergence (price higher low, RSI lower low at confirmed 5-bar pivots) with a rising EMA50
    under the close -> buy on the confirmation bar (trend continuation; therighttrader 'RSI hidden bullish divergence');
    bearish mirror. Stop 0.1 ATR beyond the pivot, target 2R (ours). New vs SC181 / SC195 (regular divergence)."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.divergence(df, k=5)
    a = atr(h, l, c, 14)
    e = ema(c, 50)
    L = d["rsi_bull_hid"] & (c > e) & (e > _r(e, 5))
    S = d["rsi_bear_hid"] & (c < e) & (e < _r(e, 5))
    pl = _s(d["piv_low"]).ffill().to_numpy()
    ph = _s(d["piv_high"]).ffill().to_numpy()
    L, S = _excl(L, S)
    return L, S, _plan_r(pl - 0.1 * a, ph + 0.1 * a, 2.0)


def s_macd_div(df):
    """MACD histogram (12/26/9) REGULAR divergence at confirmed 5-bar pivots (price lower low, histogram higher low) ->
    buy on the confirmation bar (5 bars after the pivot); bearish mirror. Stop 0.1 ATR beyond the pivot, target 2R
    (ours)."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.divergence(df, k=5)
    a = atr(h, l, c, 14)
    L, S = _excl(d["macd_bull_reg"], d["macd_bear_reg"])
    pl = _s(d["piv_low"]).ffill().to_numpy()
    ph = _s(d["piv_high"]).ffill().to_numpy()
    return L, S, _plan_r(pl - 0.1 * a, ph + 0.1 * a, 2.0)


def s_many_div(df):
    """Divergence for Many Indicators (LonesomeTheBlue v4): at a confirmed 5-bar pivot at least 3 of the 7 oscillators
    (RSI, MACD hist, CCI, momentum, OBV, MFI, Stochastic) show a regular divergence -> trade the reversal on the
    confirmation bar; stop 0.1 ATR beyond the pivot, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.divergence(df, k=5)
    a = atr(h, l, c, 14)
    L, S = _excl(d["bull_reg"] >= 3, d["bear_reg"] >= 3)
    pl = _s(d["piv_low"]).ffill().to_numpy()
    ph = _s(d["piv_high"]).ffill().to_numpy()
    return L, S, _plan_r(pl - 0.1 * a, ph + 0.1 * a, 2.0)


# ---------------- Ehlers TASC 2026 ----------------
def s_reversion_index(df):
    """Ehlers 'The Reversion Index' (TASC 2026.01): the SuperSmoother-4 trigger crosses over the SuperSmoother-8 line
    while the index is under zero -> buy (Wealth-Lab / Python listings: buy the trigger's cross over, sell the cross
    under; the under-zero / over-zero condition is ours - the article calls it a mean-reversion tool); out on the
    opposite cross. Stop 2 ATR, no target."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.reversion_index(df)
    up, dn = cross_up(r["trigger"], r["smooth"]), cross_down(r["trigger"], r["smooth"])
    L, S = _excl(up & (r["smooth"] < 0), dn & (r["smooth"] > 0))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": dn, "exit_short": up}


def s_adaptive_ss(df):
    """Ehlers Adaptive SuperSmoother (TASC 2026.09): the adaptive filter crosses over the fixed SuperSmoother(20) ->
    buy, under -> sell ('preferred direction long while the adaptive filter is above'); out on the opposite cross; stop
    2 ATR, no target (ours)."""
    r = IND.adaptive_supersmoother(df)
    up, dn = cross_up(r["ass"], r["ss"]), cross_down(r["ass"], r["ss"])
    L, S = _excl(up, dn)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": dn, "exit_short": up}


def s_autotune(df):
    """Ehlers AutoTune Filter (TASC 2026.05): the band-pass tuned to the dominant cycle crosses zero upward -> buy,
    downward -> sell (zero crossings are the cited signals); out on the opposite crossing; stop 1.5 ATR, target 2R
    (ours)."""
    r = IND.autotune_filter(df)
    up, dn = cross_up(r["bp"], 0.0), cross_down(r["bp"], 0.0)
    L, S = _excl(up, dn)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": dn, "exit_short": up}


# ---------------- DeMark ----------------
def s_td_rei_duration(df):
    """DeMark TD REI (5) with his duration rule: REI was under -40 for 1-6 bars in a row (not longer - a longer stay is
    a trend) and closes back over -40 -> buy; mirror over +40. Stop 0.1 ATR beyond the 5-bar extreme, target 2R (ours;
    DeMark pairs it with TD POQ - not modelled)."""
    o, h, l, c, v = _ohlcv(df)
    r = IND.td_rei(df)
    a = atr(h, l, c, 14)
    pb, pa = _r(r["below"]), _r(r["above"])
    L = (pb >= 1) & (pb <= 6) & (r["rei"] >= -40)
    S = (pa >= 1) & (pa <= 6) & (r["rei"] <= 40)
    sl, ss = _swl(h, l, a, 5)
    L, S = _excl(L, S)
    return L, S, _plan_r(sl, ss, 2.0)


def s_td_pressure_pullback(df):
    """TD Pressure (5, our build) as a trend pullback: with a rising EMA50 under the close, pressure dips under 35
    (supply) and crosses back over 35 -> buy; mirror (over 65 and back under, falling EMA50). Stop 1.5 ATR, target 2R
    (ours; the 35 / 65 pullback levels are ours - 25 / 75 mark the rare extremes)."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.td_pressure(df)
    e = ema(c, 50)
    L = cross_up(p, 35.0) & (c > e) & (e > _r(e, 5))
    S = cross_down(p, 65.0) & (c < e) & (e < _r(e, 5))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


# ---------------- Market Profile unfinished business ----------------
def s_npoc_touch_fade(df):
    """Naked POC first touch (LuxAlgo 'Naked POC As Level': 'fade into a first touch'): a bar touches an untouched
    previous-session POC coming from above and closes back over it -> buy (from below and closes under it -> sell).
    Stop 0.25 ATR beyond the bar's extreme, target 2R (ours - the source gives no numbers)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.naked_poc(df)
    a = atr(h, l, c, 14)
    t = z["touched"]
    L = (z["touch_dir"] == 1) & (c > t)
    S = (z["touch_dir"] == -1) & (c < t)
    L, S = _excl(L, S)
    return L, S, _plan_r(l - 0.25 * a, h + 0.25 * a, 2.0)


def s_npoc_magnet(df):
    """Naked POC as a magnet: the close crosses over the previous session's value-area high with a naked POC 1-6 ATR
    over it -> buy, target that naked POC; mirror under the value-area low toward a naked POC under. Stop 1 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.naked_poc(df)
    vp = IND.volume_profile(df)
    a = atr(h, l, c, 14)
    da, db = z["above"] - c, c - z["below"]
    L = cross_up(c, vp["vah"]) & (da >= a) & (da <= 6 * a)
    S = cross_down(c, vp["val"]) & (db >= a) & (db <= 6 * a)
    L, S = _excl(L, S)
    return L, S, _plan_lv(c - a, c + a, z["above"], z["below"])


def s_poor_extreme_revisit(df):
    """Poor high / poor low revisit (LuxAlgo 'Poor High / Poor Low': unfinished business, a candidate draw when price
    rotates back): the close crosses over the session open with an unrepaired poor high 0.5-5 ATR over it -> buy, target
    the poor high + 0.1 ATR (its repair); mirror toward a poor low. Stop 1 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.poor_extremes(df)
    t = IND.taylor_cycle(df)
    a = atr(h, l, c, 14)
    da, db = z["poor_high"] - c, c - z["poor_low"]
    L = cross_up(c, t["day_open"]) & (da >= 0.5 * a) & (da <= 5 * a)
    S = cross_down(c, t["day_open"]) & (db >= 0.5 * a) & (db <= 5 * a)
    L, S = _excl(L, S)
    return L, S, _plan_lv(c - a, c + a, z["poor_high"] + 0.1 * a, z["poor_low"] - 0.1 * a)


def s_single_print_fill(df):
    """Single prints fill (Market Profile: low-acceptance zones of one TPO period tend to be crossed quickly once
    re-entered): a bar enters a previous session's single-print zone from below -> buy, target the zone's far edge;
    from above -> sell. Stop 0.3 ATR beyond the bar's extreme (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.single_prints(df)
    a = atr(h, l, c, 14)
    fe = z["far_edge"]
    L = (z["entered"] == 1) & (fe > c)
    S = (z["entered"] == -1) & (fe < c)
    L, S = _excl(L, S)
    return L, S, _plan_lv(np.minimum(l, c - 0.5 * a) - 0.3 * a, np.maximum(h, c + 0.5 * a) + 0.3 * a, fe, fe)


# ---------------- Taylor 3-day cycle ----------------
def s_taylor_buy_day(df):
    """Taylor Trading Technique buy day / sell short day (StoneX summary): after 1-3 lower daily closes the market
    trades under the previous day's low and a close comes back over it -> buy (trapping the last sellers), target the
    previous day's high (the sell-day rally target; 2R if that is under 1R away); mirror after 1-3 higher closes over
    the previous high. First signal per day; stop 0.1 ATR beyond today's extreme (ours)."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.taylor_cycle(df)
    a = atr(h, l, c, 14)
    day = session_day_np(df)
    L = (t["down_days"] >= 1) & (t["down_days"] <= 3) & (t["day_low"] < t["pdl"]) & cross_up(c, t["pdl"])
    S = (t["up_days"] >= 1) & (t["up_days"] <= 3) & (t["day_high"] > t["pdh"]) & cross_down(c, t["pdh"])
    L, S = _first_per_day(L, day), _first_per_day(S, day)
    sl, ss = t["day_low"] - 0.1 * a, t["day_high"] + 0.1 * a
    tl = np.where(t["pdh"] - c >= c - sl, t["pdh"], c + 2 * (c - sl))
    ts = np.where(c - t["pdl"] >= ss - c, t["pdl"], c - 2 * (ss - c))
    L, S = _excl(L, S)
    return L, S, _plan_lv(sl, ss, tl, ts)


def s_taylor_inside_day(df):
    """Taylor inside day 'go with' trade (StoneX summary: on an inside day both its high and low are the references):
    the last inside day (yesterday or the day before - ours, so the levels live 2 days) -> the first close over its
    high buys, under its low sells, once per day; stop at the middle of the inside day's range, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    t = IND.taylor_cycle(df)
    day = session_day_np(df)
    nd = t["new_day"]
    ins = (t["inside"] == 1) & nd
    hi = _s(np.where(ins, t["pdh"], np.nan)).ffill().to_numpy()
    lo = _s(np.where(ins, t["pdl"], np.nan)).ffill().to_numpy()
    dcount = np.cumsum(nd)
    age = dcount - _s(np.where(ins, dcount, np.nan)).ffill().to_numpy()
    live = age <= 1
    L = _first_per_day(live & cross_up(c, hi), day)
    S = _first_per_day(live & cross_down(c, lo), day)
    mid = (hi + lo) / 2
    L, S = _excl(L, S)
    return L, S, _plan_r(mid, mid, 2.0)


def session_day_np(df):
    from agents.indicators._core import session_day
    return session_day(df, "UTC", 0)


# ---------------- Zarattini-Aziz-Barbon noise area ----------------
def s_noise_area_momentum(df):
    """'Beat the Market' intraday momentum (Zarattini, Aziz, Barbon 2024, SPY): at each :00 / :30 New York decision
    time from 10:00, a close over the noise area's upper bound (open x (1 + 14-day mean |move from the open| at that
    time), gap-adjusted with the previous close) buys, under the lower bound sells; the stop trails at max(UB, VWAP)
    for longs (min(LB, VWAP) for shorts), checked at the decision times; everything out at the 16:00 close. Ours: the
    initial hard stop = the trailing level (or 0.25 ATR under the entry close if that is over it); no target; the
    paper's volatility sizing is left to the bot."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.noise_area(df)
    a = atr(h, l, c, 14)
    ck = z["check"]
    lt = np.fmax(z["ub"], z["vwap"])
    st = np.fmin(z["lb"], z["vwap"])
    L = ck & (c > z["ub"])
    S = ck & (c < z["lb"])
    sl = np.where(lt < c, lt, c - 0.25 * a)
    ss = np.where(st > c, st, c + 0.25 * a)
    step = _bar_minutes(df)
    t = df.index.tz_convert("America/New_York")
    mc = t.hour.to_numpy() * 60 + t.minute.to_numpy() + step
    fe = (mc >= 960) & (mc < 960 + step) | ~z["in_rth"]
    L, S = _excl(L & ~fe, S & ~fe)
    return L, S, {"stop": ("level", sl, ss), "target": None, "exit_long": ck & (c < lt), "exit_short": ck & (c > st),
                  "force_exit": fe}


# ---------------- IPF pivot-stop breakouts ----------------
def s_ipf_pivot_break(df):
    """IPF Price Action EA (MQL5 blog https://www.mql5.com/en/blogs/post/777967; H4 / D1, no indicators): a confirmed
    pivot high (5 bars each side) that is the extreme of its area (the 30-bar high) and at least 1 ATR from the close
    when it is confirmed; a close over it buys (their buy-stop), mirror under a pivot low. The post's 'fake-breakout
    exit': out if the bar after the entry closes back under the level (ours: on the entry chart, not M1). Stop 1.5
    ATR, target 2R, break-even at half way (ours - the post gives no numbers)."""
    o, h, l, c, v = _ohlcv(df)
    from agents.indicators._core import pivots
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, phi, pli = pivots(h, l, 5)
    hi30, lo30 = _s(h).rolling(30, min_periods=10).max().to_numpy(), _s(l).rolling(30, min_periods=10).min().to_numpy()
    okh = phc & (phv >= hi30 - 1e-12) & (phv - c >= a)
    okl = plc & (plv <= lo30 + 1e-12) & (c - plv >= a)
    lh = _s(np.where(okh, phv, np.nan)).ffill().to_numpy()
    ll = _s(np.where(okl, plv, np.nan)).ffill().to_numpy()
    L = cross_up(c, lh)
    S = cross_down(c, ll)
    xl = (_r(L.astype(float)) > 0) & (c < _r(lh))
    xs = (_r(S.astype(float)) > 0) & (c > _r(ll))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "breakeven": (0.5, 0.0), "exit_long": xl,
                  "exit_short": xs}


SCOUT.update({
    "SC459": ("Morning / evening star or abandoned baby after a move (TA-Lib, Bulkowski), pattern stop, 2R", s_cdl_stars, _F + ["4Hour"]),
    "SC460": ("Three outside up / down after a move (TA-Lib CDL3OUTSIDE), pattern stop, 2R", s_cdl_three_outside, _F + ["4Hour"]),
    "SC461": ("Three inside up / down after a move (TA-Lib CDL3INSIDE), pattern stop, 2R", s_cdl_three_inside, _F + ["4Hour"]),
    "SC462": ("Piercing line / dark cloud cover after a move (TA-Lib), pattern stop, 2R", s_cdl_piercing, _F + ["4Hour"]),
    "SC463": ("Three-line strike traded the Bulkowski way (with the striking 4th candle), 2R", s_cdl_three_line_strike, _F + ["4Hour"]),
    "SC464": ("Tweezer bottom / top at the 20-bar extreme + confirmation close within 3 bars, 2R", s_cdl_tweezer_extreme, _F + ["4Hour"]),
    "SC466": ("RSI 14 hidden divergence with the EMA50 trend (continuation), pivot stop, 2R", s_rsi_hidden_div, _F + ["4Hour"]),
    "SC467": ("MACD histogram regular divergence at a confirmed 5-bar pivot, pivot stop, 2R", s_macd_div, _F + ["4Hour"]),
    "SC468": ("Divergence for Many: 3+ of 7 oscillators diverge at a pivot (LonesomeTheBlue), pivot stop, 2R", s_many_div, _F + ["4Hour"]),
    "SC469": ("Ehlers Reversion Index: SS4 trigger crosses SS8 on the far side of zero, out on the opposite cross (TASC 2026.01)", s_reversion_index, _F + ["4Hour"]),
    "SC470": ("Ehlers Adaptive SuperSmoother crosses the fixed SuperSmoother 20, stop and reverse-out, 2 ATR (TASC 2026.09)", s_adaptive_ss, _M),
    "SC471": ("Ehlers AutoTune band-pass zero cross, opposite-cross exit, 1.5 ATR, 2R (TASC 2026.05)", s_autotune, _F + ["4Hour"]),
    "SC472": ("DeMark TD REI back inside +/-40 after 1-6 bars beyond (duration rule), 5-bar stop, 2R", s_td_rei_duration, _F + ["4Hour"]),
    "SC473": ("TD Pressure back over 35 (under 65) with the EMA50 trend, 1.5 ATR, 2R", s_td_pressure_pullback, _F),
    "SC474": ("Naked POC first-touch fade (close back on the approach side), 0.25 ATR beyond the bar, 2R", s_npoc_touch_fade, ["5Min", "15Min"]),
    "SC475": ("Value-area breakout toward a naked POC 1-6 ATR away, target the naked POC, 1 ATR stop", s_npoc_magnet, ["5Min", "15Min"]),
    "SC476": ("Rotation over the session open toward an unrepaired poor high (poor low) 0.5-5 ATR away, target its repair, 1 ATR stop", s_poor_extreme_revisit, ["5Min", "15Min"]),
    "SC477": ("Single-print zone fill: entering a previous session's single prints, target the far edge", s_single_print_fill, ["5Min", "15Min"]),
    "SC478": ("Taylor buy day / sell short day: after 1-3 down (up) days, reclaim of the previous day's low (high), target its high", s_taylor_buy_day, ["5Min", "15Min", "1Hour"]),
    "SC479": ("Taylor inside day go-with: close beyond the last inside day's high / low (2 days), mid-range stop, 2R", s_taylor_inside_day, ["15Min", "1Hour"]),
    "SC480": ("Zarattini noise-area intraday momentum (SPY paper): :00 / :30 breaks of the noise area, VWAP trail, out 16:00 NY", s_noise_area_momentum, ["5Min", "15Min", "30Min"]),
    "SC481": ("IPF pivot-stop breakout of an area-extreme pivot 1+ ATR away, fake-breakout exit, 1.5 ATR, 2R, BE (MQL5 EA)", s_ipf_pivot_break, ["1Hour", "4Hour", "1Day"]),
})


# ================= scout 16.11.2026 night: ProRealCode DAX, StockSharp, Larry Williams (MQL5 series), new indicators =================
def _local_min(df, tz):
    t = df.index.tz_convert(tz)
    return t, t.hour.to_numpy() * 60 + t.minute.to_numpy()


def _daily_on_bars(df, lag=1):
    """Completed UTC days' o, h, l, c shifted by `lag` days, on every intraday bar (known at the bar), + the day's
    first-bar flag and today's open."""
    day = pd.Series(df.index.tz_convert("UTC").date)
    g = pd.DataFrame({"d": day, "o": df["o"].to_numpy(float), "h": df["h"].to_numpy(float),
                      "l": df["l"].to_numpy(float), "c": df["c"].to_numpy(float)})
    d = g.groupby("d").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    out = {}
    for k in range(lag, lag + 5):
        sh = d.shift(k)
        m = g[["d"]].join(sh, on="d")
        out[k] = {x: m[x].to_numpy(float) for x in "ohlc"}
    first = (day != day.shift(1)).to_numpy()
    today_o = g[["d"]].join(d[["o"]], on="d")["o"].to_numpy(float)
    return out, first, today_o, day


# ---------------- SC482 Trend Surfer DAX (ProRealCode) ----------------
def s_prc_trend_surfer_dax(df):
    """ProRealCode 'Trend Surfer DAX' (https://www.prorealcode.com/prorealtime-trading-strategies/trend-surfer-dax/ ;
    DAX 5-minute): RSI 14, priceline = SMA4(RSI), middle band = SMA29(RSI). Entries only 09:00-11:00 and 15:00-17:00
    Frankfurt. Long: priceline over the middle band, priceline > 35 and middle band > 40 (first bar of that state);
    short: priceline crosses under the middle band with priceline > 40. Out at 11:00 / 17:00. Source stop 250 points
    and target 50 points translated for DAX 5m (ATR14 ~ 20 points): stop 12 ATR, target 2.5 ATR."""
    o, h, l, c, v = _ohlcv(df)
    r = rsi(c, 14)
    pl, mb = sma(r, 4), sma(r, 29)
    t, mm = _local_min(df, "Europe/Berlin")
    step = _bar_minutes(df)
    win = ((mm >= 540) & (mm < 660)) | ((mm >= 900) & (mm < 1020))
    L = _onset((pl > mb) & (pl > 35) & (mb > 40)) & win
    S = cross_down(pl, mb) & (pl > 40) & win
    fe = ((mm >= 660) & (mm < 660 + step)) | ((mm >= 1020) & (mm < 1020 + step))
    L, S = _excl(L & ~fe, S & ~fe)
    a = atr(h, l, c, 14)
    return L, S, {"stop": ("atr", 12.0), "target": ("level", c + 2.5 * a, c - 2.5 * a), "force_exit": fe}


# ---------------- SC483 VTAD Turnaround Tuesday DAX ----------------
def s_vtad_turnaround_dax(df):
    """VTAD Turnaround Tuesday - DAX (ProRealCode https://www.prorealcode.com/prorealtime-trading-strategies/vtad-turnaround-tuesday-dax-15m/ ;
    15-minute, also 5-minute): on Monday at 17:45 Frankfurt buy if Monday's close is under the daily SMA34 of closes;
    out on Wednesday 09:00 Frankfurt. Stop 2.6% and target 2.3% (the poster's additions; VTAD has none). Ours: the
    17:45 bar's close stands for Monday's close, the SMA34 uses the 33 completed days before + that close. Differs
    from SE03 (Monday down vs Friday, out Tuesday's close, no SMA filter)."""
    o, h, l, c, v = _ohlcv(df)
    t, mm = _local_min(df, "Europe/Berlin")
    step = _bar_minutes(df)
    dow = t.weekday.to_numpy()
    dly, first, today_o, day = _daily_on_bars(df, 1)
    g = pd.DataFrame({"d": day, "c": c})
    dc = g.groupby("d")["c"].last()
    s33 = dc.rolling(33, min_periods=3).mean().shift(1) * 33.0  # short history: mean of what exists
    s33b = g[["d"]].join(s33.rename("s"), on="d")["s"].to_numpy(float)
    sma34 = (s33b + c) / 34.0
    sig = (dow == 0) & (mm + step == 17 * 60 + 45) & (c < sma34)
    fe = (dow == 2) & (mm >= 540) & (mm < 540 + step)
    sl, tp = c * (1 - 0.026), c * (1 + 0.023)
    return sig, np.zeros(len(c), bool), {"stop": ("level", sl, c * 1.026), "target": ("level", tp, c * 0.977),
                                         "force_exit": fe}


# ---------------- SC484 StockSharp Breaks and Retests ----------------
def s_ss_breaks_retests(df):
    """StockSharp 'Breaks and Retests' (https://doc.stocksharp.com/en/api-examples/0563_Breaks_and_Retests ; 5-minute):
    levels = highest / lowest CLOSE of the prior 20 bars; after a close over the resistance, a retest 2-4 bars later
    (the bar's low touches the broken level and it closes back over it) buys; mirror for shorts. Stop 2% from the
    entry (source). Source trail (after 5% profit, 1% under the best close) - ours: target 2R instead, the plan has
    no trailing."""
    o, h, l, c, v = _ohlcv(df)
    res = _r(_s(c).rolling(20).max().to_numpy())
    sup = _r(_s(c).rolling(20).min().to_numpy())
    bu, bd = c > res, c < sup
    L = np.zeros(len(c), bool)
    S = np.zeros(len(c), bool)
    for k in (2, 3, 4):
        lv_u, lv_d = _r(res, k), _r(sup, k)
        L |= (_r(bu.astype(float), k) > 0) & (l <= lv_u) & (c > lv_u)
        S |= (_r(bd.astype(float), k) > 0) & (h >= lv_d) & (c < lv_d)
    L, S = _excl(_onset(L), _onset(S))
    return L, S, _plan_r(c * 0.98, c * 1.02, 2.0)


# ---------------- SC485 StockSharp S/R breakout 55 + EMA500 ----------------
def s_ss_sr_break_ema500(df):
    """StockSharp 'Support & Resistance Breakout' (https://doc.stocksharp.com/en/api-examples/3998_Support_Resistance_Breakout ;
    1-hour default): resistance / support = the prior 55 bars' high / low; long on a close over the resistance with
    the close over EMA 500, short mirror; an in-profit close beyond the opposite band exits (ours: the close beyond
    the 55-bar opposite band). Source stop / target 500 points each - ours 3 ATR / 1R (1:1 like the source).
    Differs from the turtle 55/20 (TV14): EMA500 filter, 1:1 fixed exits."""
    o, h, l, c, v = _ohlcv(df)
    hi = _r(_s(h).rolling(55).max().to_numpy())
    lo = _r(_s(l).rolling(55).min().to_numpy())
    e = ema(c, 500)
    L = cross_up(c, hi) & (c > e)
    S = cross_down(c, lo) & (c < e)
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 3.0), "target": ("R", 1.0), "exit_long": c < lo, "exit_short": c > hi}


# ---------------- SC486-SC489 Larry Williams Market Secrets (MQL5 series) ----------------
def s_lw_swing_range_breakout(df):
    """Larry Williams Market Secrets Part 6 (https://www.mql5.com/en/articles/20862 ; daily): range = the larger of
    |high 4 days ago - low 1 day ago| and |high 2 days ago - low 4 days ago| (swing range); buy when price crosses
    today's open + 0.5 x range, sell under the open - 0.5 x range (first cross of the day, intraday bars); stop 0.5 x
    range from the entry level, target 4R. Differs from BK05 (yesterday's range, open stop, day-end exit)."""
    o, h, l, c, v = _ohlcv(df)
    d, first, to, day = _daily_on_bars(df, 1)
    rng = np.fmax(np.abs(d[4]["h"] - d[1]["l"]), np.abs(d[2]["h"] - d[4]["l"]))
    bl, sl_ = to + 0.5 * rng, to - 0.5 * rng
    up, dn = c > bl, c < sl_
    g = pd.DataFrame({"d": day, "u": up, "n": dn})
    fu = g.groupby("d")["u"].cumsum().to_numpy() == 1
    fd = g.groupby("d")["n"].cumsum().to_numpy() == 1
    L, S = _excl(up & fu, dn & fd)
    return L, S, _plan_r(bl - 0.5 * rng, sl_ + 0.5 * rng, 4.0)


def _lw_break(df, cond_l, cond_s, mult=0.5):
    """On the chart's own bars (a daily chart = the source; on intraday charts each bar stands for a day): the bar
    closes over its open + mult x the previous bar's range (under open - mult x range) after the pattern on the bars
    before it. Ours: the close instead of the source's stop-order cross inside the day."""
    o, h, l, c, v = _ohlcv(df)
    d = {k: {"o": _r(o, k), "h": _r(h, k), "l": _r(l, k), "c": _r(c, k)} for k in range(1, 5)}
    rng = d[1]["h"] - d[1]["l"]
    L = (c > o + mult * rng) & cond_l(d)
    S = (c < o - mult * rng) & cond_s(d)
    L, S = _excl(L, S)
    return L, S, _plan_r(d[1]["l"], d[1]["h"], 3.0, max_bars=3)


def s_lw_three_down_break(df):
    """Larry Williams Market Secrets Part 9 'Patterns to Profit' (https://www.mql5.com/en/articles/21063 ; daily):
    after three down closes in a row buy the move over the open + 0.5 x the previous bar's range (after three up
    closes sell under the open - 0.5 x range); stop the previous bar's low (high), target 3R, out after 3 bars."""
    def cl(d):
        return (d[1]["c"] < d[2]["c"]) & (d[2]["c"] < d[3]["c"]) & (d[3]["c"] < d[4]["c"])

    def cs(d):
        return (d[1]["c"] > d[2]["c"]) & (d[2]["c"] > d[3]["c"]) & (d[3]["c"] > d[4]["c"])
    return _lw_break(df, cl, cs)


def s_lw_outside_day_break(df):
    """Larry Williams Part 9 (https://www.mql5.com/en/articles/21063): the previous bar a bearish outside bar (higher
    high, lower low - or, ours, a body that engulfs the previous body - close under its open) -> buy the move over the open + 0.5 x its range; mirror after a bullish
    outside bar; stop its low (high), target 3R, out after 3 bars."""
    def out(d):  # outside by high / low, or by the body (feeds whose highs / lows are only close +/- spread)
        hl = (d[1]["h"] > d[2]["h"]) & (d[1]["l"] < d[2]["l"])
        bd = (np.fmax(d[1]["o"], d[1]["c"]) >= np.fmax(d[2]["o"], d[2]["c"])) & \
             (np.fmin(d[1]["o"], d[1]["c"]) <= np.fmin(d[2]["o"], d[2]["c"])) & (d[1]["o"] != d[1]["c"])
        return hl | bd

    def cl(d):
        return out(d) & (d[1]["c"] < d[1]["o"])

    def cs(d):
        return out(d) & (d[1]["c"] > d[1]["o"])
    return _lw_break(df, cl, cs)


def s_lw_short_term_swing(df):
    """Larry Williams Part 4 short-term swing points (https://www.mql5.com/en/articles/20716 ; daily, any chart):
    bar 2's low under the lows of bars 1 and 3 (bar 1 = the bar just closed), skipped when bar 2 is an outside bar of
    bar 3 or bar 1 is an inside bar of bar 2 -> buy; mirror with highs -> sell. Stop at bar 2's low (high); the
    source's default exit: hold one bar (out at the next bar)."""
    o, h, l, c, v = _ohlcv(df)
    h1, l1, h2, l2, h3, l3 = h, l, _r(h, 1), _r(l, 1), _r(h, 2), _r(l, 2)
    outside2 = (h2 > h3) & (l2 < l3)
    inside1 = (h1 < h2) & (l1 > l2)
    L = (l2 < l1) & (l2 < l3) & ~outside2 & ~inside1
    S = (h2 > h1) & (h2 > h3) & ~outside2 & ~inside1
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", l2, h2), "target": None, "max_bars": 1}


# ---------------- SC490-SC503 strategies on the new indicators (agents/indicators/classic6.py) ----------------
def s_cti_trend(df):
    """Ehlers Correlation Trend Indicator (TASC 2020.05, https://www.traders.com/Documentation/FEEDbk_docs/2020/05/TradersTips.html ):
    CTI 20 crosses over +0.5 (trend begins) buys, under -0.5 sells; out when it falls back under 0 (over 0); stop 2
    ATR (ours - the article gives no stop)."""
    o, h, l, c, v = _ohlcv(df)
    x = IND.correlation_trend(df)["cti"]
    L = cross_up(x, np.full(len(c), 0.5))
    S = cross_down(x, np.full(len(c), -0.5))
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": x < 0, "exit_short": x > 0}


def s_ttm_trend_flip(df):
    """TTM Trend (Carter; our build agents/indicators ttm_trend): the second trend bar of the new colour after 6+
    bars of the other colour, with the EMA50 slope agreeing; stop 1.5 ATR, target 2R, out on 2 bars of the other
    colour (Carter: 'stay in while the bars stay blue')."""
    o, h, l, c, v = _ohlcv(df)
    tr_ = IND.ttm_trend(df)["trend"]
    up, dn = tr_ == 1, tr_ == -1
    run_dn = _s(dn.astype(float)).rolling(6).sum().to_numpy()
    run_up = _s(up.astype(float)).rolling(6).sum().to_numpy()
    e = ema(c, 50)
    es = e > _r(e, 3)
    L = up & _r(up.astype(float)).astype(bool) & (_r(run_dn, 2) >= 6) & es
    S = dn & _r(dn.astype(float)).astype(bool) & (_r(run_up, 2) >= 6) & ~es
    L, S = _excl(L, S)
    ex_l = dn & _r(dn.astype(float)).astype(bool)
    ex_s = up & _r(up.astype(float)).astype(bool)
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0), "exit_long": ex_l, "exit_short": ex_s}


def s_ttm_scalper_trend(df):
    """TTM Scalper reversal squares (Carter; our build agents/indicators ttm_scalper) taken only with the trend: buy
    square while the close is over EMA 50, sell square under it; stop 0.2 ATR beyond the pivot bar, target 2R."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.ttm_scalper(df)
    a = atr(h, l, c, 14)
    e = ema(c, 50)
    L, S = _excl(z["buy"] & (c > e), z["sell"] & (c < e))
    return L, S, _plan_r(z["pivot_low"] - 0.2 * a, z["pivot_high"] + 0.2 * a, 2.0)


def s_murrey_extreme_fade(df):
    """Murrey Math (LuxAlgo https://www.luxalgo.com/library/indicator/murrey-math-levels/ : 0/8 ultimate support, 8/8
    ultimate resistance, 4/8 pivot): a bar dips under 0/8 and closes back over it -> buy, target 4/8, stop 1/8 under
    0/8; mirror at 8/8. Stop and target placement ours (the source names the roles only)."""
    o, h, l, c, v = _ohlcv(df)
    m = {k: _r(x) for k, x in IND.murrey_math(df).items()}  # the frame of the bar before (0/8 <= its own low)
    L = (l < m["l0"]) & (c > m["l0"])
    S = (h > m["l8"]) & (c < m["l8"])
    L, S = _excl(L, S)
    return L, S, _plan_lv(m["l0"] - m["step"], m["l8"] + m["step"], m["l4"], m["l4"])


def s_murrey_pivot_cross(df):
    """Murrey Math 4/8 pivot (LuxAlgo summary: 'closes above / below 4/8'): a close over 4/8 after 3 closes under it
    buys, target 6/8 (the next reversal rung), stop under 2/8; mirror. Placement ours."""
    o, h, l, c, v = _ohlcv(df)
    m = IND.murrey_math(df)
    below = _s((c < m["l4"]).astype(float)).rolling(3).sum().to_numpy()
    above = _s((c > m["l4"]).astype(float)).rolling(3).sum().to_numpy()
    L = (c > m["l4"]) & (_r(below) >= 3)
    S = (c < m["l4"]) & (_r(above) >= 3)
    L, S = _excl(L, S)
    return L, S, _plan_lv(m["l2"], m["l6"], m["l6"], m["l2"])


def s_chaikin_vol_breakout(df):
    """Chaikin Volatility (https://www.incrediblecharts.com/indicators/chaikin_volatility.php : expanding range
    confirms a breakout, contracting range warns it is weak): a close beyond the prior 20-bar high / low taken only
    while Chaikin Volatility 10/10 is >= 0 and not falling (ranges expanding); skipped when it is under 0 (ranges
    contracting); stop 2 ATR, target 2R. Combination ours."""
    o, h, l, c, v = _ohlcv(df)
    cv = np.nan_to_num(IND.chaikin_volatility(df)["cv"])
    ok = (cv >= 0) & (cv >= _r(cv) - 1e-12)
    hi = _r(_s(h).rolling(20).max().to_numpy())
    lo = _r(_s(l).rolling(20).min().to_numpy())
    L, S = _excl(cross_up(c, hi) & ok, cross_down(c, lo) & ok)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


def s_se_bands_squeeze_break(df):
    """Andersen Standard Error Bands (https://www.luxalgo.com/library/indicator/standard-error-bands/ : tight bands =
    a strong steady trend / coiling, widening after = the move): band width at its 50-bar low within the last 5 bars,
    then a close outside the band in the direction of the regression slope; out on a close back at the middle; stop
    2 ATR. Differs from SC322 (fade back inside a linreg / standard-error channel)."""
    o, h, l, c, v = _ohlcv(df)
    b = IND.standard_error_bands(df)
    w = b["width"]
    tight = _s((w <= _s(w).rolling(50).min().to_numpy() * 1.0001).astype(float)).rolling(5, min_periods=1).max().to_numpy() > 0
    L = cross_up(c, b["upper"]) & (b["slope"] > 0) & tight
    S = cross_down(c, b["lower"]) & (b["slope"] < 0) & tight
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": c < b["mid"], "exit_short": c > b["mid"]}


def s_projection_osc_turn(df):
    """Widner Projection Oscillator (https://www.prorealcode.com/prorealtime-indicators/projection-oscillator ;
    LuxAlgo concept https://www.luxalgo.com/library/concept/projection-bands/ ): PO 14 crosses over its EMA5 trigger
    with both under 20 -> buy, crosses under with both over 80 -> sell; stop 0.25 ATR beyond the projection band,
    target the other band."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.projection(df)
    po, tg = p["po"], p["trigger"]
    a = atr(h, l, c, 14)
    L = cross_up(np.nan_to_num(po, nan=50.0), tg) & (tg < 20) & (_r(po) < 20)
    S = cross_down(np.nan_to_num(po, nan=50.0), tg) & (tg > 80) & (_r(po) > 80)
    L, S = _excl(L, S)
    return L, S, _plan_lv(p["lower"] - 0.25 * a, p["upper"] + 0.25 * a, p["upper"], p["lower"])


def s_wave_pm_release(df):
    """Mark Whistler's Wave PM (https://www.tradingview.com/script/l9Oqp36g-Wave-PM ; Whistler: energy stored while the
    wave is low, released when it rises): Wave PM was under 0.5 within the last 10 bars and the close breaks the prior
    20-bar high (low) while Wave PM rises over 0.5 -> trade the break; stop 2 ATR, target 2R. Combination ours."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.wave_pm(df)["wpm"]
    low_ = _s((w < 0.5).astype(float)).rolling(10, min_periods=1).max().to_numpy() > 0
    rising = (w > 0.5) & (w > _r(w))
    hi = _r(_s(h).rolling(20).max().to_numpy())
    lo = _r(_s(l).rolling(20).min().to_numpy())
    L, S = _excl(cross_up(c, hi) & low_ & rising, cross_down(c, lo) & low_ & rising)
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


def s_szo_zone_trend(df):
    """Khalil Sentiment Zone Oscillator (TASC 2012.05, https://www.traders.com/Documentation/FEEDbk_docs/2012/05/Khalil.html ):
    in an uptrend (close over EMA 60) buy when SZO crosses back over the dynamic oversold zone; in a downtrend sell
    when it crosses back under the overbought zone; out on the opposite zone touch; stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.sentiment_zone(df)
    s = z["szo"]
    L = cross_up(s, z["os"]) & z["trend_up"]
    S = cross_down(s, z["ob"]) & ~z["trend_up"]
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": s >= z["ob"], "exit_short": s <= z["os"]}


def s_rainbow_osc_cross(df):
    """Widner Rainbow Oscillator (cTrader https://help.ctrader.com/indicators/built-in/oscillators/rainbow-oscillator/ :
    buy when RO crosses over its average, sell under; stop beyond the recent swing; out on the opposite cross), only
    while the Rainbow Bandwidth widens (over its 20-bar mean, Widner: wide rainbow = trend); stop the 5-bar swing +/-
    0.1 ATR, opposite-cross exit."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.rainbow(df)
    ro, sg, rb = z["ro"], z["signal"], z["rb"]
    wide = rb > sma(rb, 20)
    a = atr(h, l, c, 14)
    L = cross_up(ro, sg) & wide & (ro > 0)
    S = cross_down(ro, sg) & wide & (ro < 0)
    L, S = _excl(L, S)
    sl, ss = _swl(h, l, a, 5, 0.1)
    return L, S, {"stop": ("level", sl, ss), "target": None, "exit_long": cross_down(ro, sg),
                  "exit_short": cross_up(ro, sg)}


def s_madrid_ribbon_turn(df):
    """Madrid MA Ribbon (https://in.tradingview.com/script/WH01CtjA-Madrid-MA-Ribbon-Bar ): the ribbon turns all lime
    (score 1: all 18 EMAs rising over EMA100) after being under 0.5 within 10 bars -> buy; all red -> sell; out when
    the score crosses 0; stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    sc = IND.madrid_ribbon(df)["score"]
    was_lo = _s((sc < 0.5).astype(float)).rolling(10, min_periods=1).max().to_numpy() > 0
    was_hi = _s((sc > -0.5).astype(float)).rolling(10, min_periods=1).max().to_numpy() > 0
    L = _onset(sc >= 0.999) & was_lo
    S = _onset(sc <= -0.999) & was_hi
    L, S = _excl(L, S)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": sc < 0, "exit_short": sc > 0}


def s_pitchfork_lower_test(df):
    """Andrews' Pitchfork (Trading Setups Review https://www.tradingsetupsreview.com/andrews-pitchfork-trading-strategy/ ):
    in a bull fork (A a pivot low, rising median) price tests the lower parallel - the bar's low reaches it while its
    high stays over it - and closes as a bull bar over the line -> buy (source: a tick over that bar's high; ours: the
    close); stop under the bar's low (0.1 ATR), target the median line. Mirror at the upper parallel of a bear fork."""
    o, h, l, c, v = _ohlcv(df)
    p = IND.pitchfork(df)
    a = atr(h, l, c, 14)
    bull = (p["dir"] == 1) & (p["median"] > _r(p["median"]))
    bear = (p["dir"] == -1) & (p["median"] < _r(p["median"]))
    L = bull & (l <= p["lower"]) & (h > p["lower"]) & (c > o) & (c > p["lower"]) & (p["median"] > c)
    S = bear & (h >= p["upper"]) & (l < p["upper"]) & (c < o) & (c < p["upper"]) & (p["median"] < c)
    L, S = _excl(L, S)
    return L, S, _plan_lv(l - 0.1 * a, h + 0.1 * a, p["median"], p["median"])


def s_kase_devstop_trend(df):
    """Kase Dev-Stops (https://www.prorealcode.com/prorealtime-indicators/kase-dev-stop/ ): enter on the SMA 10 / 21
    cross; exit when the close crosses the Dev-Stop 2 (2.2 std) line; hard stop at Dev-Stop 3 (3.6 std). Entry rule
    ours (Kase sells the stops, the entry is the trend change)."""
    o, h, l, c, v = _ohlcv(df)
    k = IND.kase_devstop(df)
    f, s_ = sma(c, 10), sma(c, 21)
    L, S = _excl(cross_up(f, s_), cross_down(f, s_))
    sl = np.where(k["ds3_long"] < c, k["ds3_long"], c - 3 * atr(h, l, c, 14))
    ss = np.where(k["ds3_short"] > c, k["ds3_short"], c + 3 * atr(h, l, c, 14))
    return L, S, {"stop": ("level", sl, ss), "target": None, "exit_long": c < k["ds2_long"],
                  "exit_short": c > k["ds2_short"]}


SCOUT.update({
    "SC482": ("Trend Surfer DAX: RSI14 SMA4 vs SMA29 (35 / 40), 09-11 + 15-17 Frankfurt, out 11:00 / 17:00, 12 ATR / 2.5 ATR (ProRealCode)", s_prc_trend_surfer_dax, ["5Min"]),
    "SC484": ("Breaks and retests: 20-bar close extreme break, retest 2-4 bars later closing back, 2% stop, 2R (StockSharp)", s_ss_breaks_retests, ["5Min", "15Min"]),
    "SC485": ("S/R 55-bar breakout with the EMA500, 3 ATR 1:1, opposite-band exit (StockSharp)", s_ss_sr_break_ema500, ["1Hour", "4Hour"]),
    "SC486": ("Larry Williams swing-range breakout: open +/- 0.5 x max(|H4-L1|, |H2-L4|), range stop, 4R (MQL5 Part 6)", s_lw_swing_range_breakout, ["15Min", "1Hour"]),
    "SC487": ("Larry Williams 3 down (up) closes -> open + 0.5 x range break, prior-bar stop, 3R, 3 bars (MQL5 Part 9)", s_lw_three_down_break, ["1Hour", "4Hour", "1Day"]),
    "SC488": ("Larry Williams bearish (bullish) outside day -> open + 0.5 x range break, 3R, 3 bars (MQL5 Part 9)", s_lw_outside_day_break, ["1Hour", "4Hour", "1Day"]),
    "SC489": ("Larry Williams short-term swing low / high (no outside / inside bars), pivot-bar stop, hold 1 bar (MQL5 Part 4)", s_lw_short_term_swing, ["1Hour", "4Hour", "1Day"]),
    "SC490": ("Ehlers Correlation Trend 20 crosses +/-0.5, out at 0, 2 ATR (TASC 2020.05)", s_cti_trend, _F + ["4Hour"]),
    "SC491": ("TTM Trend flip: 2 bars of the new colour after 6+, EMA50 slope, 1.5 ATR, 2R, 2-bar colour exit (Carter)", s_ttm_trend_flip, _F + ["4Hour"]),
    "SC492": ("TTM Scalper reversal square with the EMA50 trend, pivot stop, 2R (Carter)", s_ttm_scalper_trend, _F),
    "SC493": ("Murrey Math 0/8 - 8/8 false break fade, target 4/8, 1/8 stop", s_murrey_extreme_fade, _F + ["4Hour"]),
    "SC494": ("Murrey Math 4/8 pivot cross after 3 closes on the other side, target 6/8 / 2/8", s_murrey_pivot_cross, _F + ["4Hour"]),
    "SC495": ("20-bar breakout only while Chaikin Volatility 10/10 is >= 0 and not falling (ranges expanding), 2 ATR, 2R", s_chaikin_vol_breakout, _F + ["4Hour"]),
    "SC496": ("Standard Error Bands squeeze (50-bar low width) then close outside with the slope, mid exit, 2 ATR (Andersen)", s_se_bands_squeeze_break, _F + ["4Hour"]),
    "SC497": ("Projection Oscillator crosses its trigger under 20 / over 80, band stop, other-band target (Widner)", s_projection_osc_turn, _F + ["4Hour"]),
    "SC498": ("Wave PM energy release: 20-bar break as Wave PM rises through 0.5 after compression, 2 ATR, 2R (Whistler)", s_wave_pm_release, _F + ["4Hour"]),
    "SC499": ("Sentiment Zone Oscillator back over the dynamic oversold (under overbought) with the EMA60 trend, zone exit (Khalil)", s_szo_zone_trend, _F + ["4Hour"]),
    "SC500": ("Rainbow Oscillator crosses its SMA9 while the rainbow widens, swing stop, opposite-cross exit (Widner)", s_rainbow_osc_cross, _F + ["4Hour"]),
    "SC501": ("Madrid MA Ribbon turns all lime / all red, out at score 0, 2 ATR", s_madrid_ribbon_turn, _F + ["4Hour"]),
    "SC502": ("Andrews' Pitchfork lower (upper) parallel test with a bull (bear) bar, bar stop, median-line target", s_pitchfork_lower_test, _F + ["4Hour", "1Day"]),
    "SC503": ("SMA 10/21 cross, Kase Dev-Stop 2 exit, Dev-Stop 3 hard stop (Kase)", s_kase_devstop_trend, _M + ["1Day"]),
})


# ---------------- scout 17.11.2026 night: TASC 2026.06 / .08 / .10, TradingView built-ins, Hoffman IRB, session
# setups (The Right Trader, edgeful, TradeZella) and one strategy per new indicator (agents/indicators/classic7.py) ----

def _wk_shift(df):
    """Trading-week clock: New York time + 7 h, so the FX week (Sunday 17:00 NY) and the cash week both start on a
    'Monday'. Returns weekday, hour float and week key per bar."""
    t = df.index.tz_convert("America/New_York") + pd.Timedelta(hours=7)
    wk = (pd.Series(t.isocalendar().year.to_numpy()) * 100 + pd.Series(t.isocalendar().week.to_numpy())).to_numpy()
    return t.weekday.to_numpy(), t.hour.to_numpy() + t.minute.to_numpy() / 60.0, wk


def s_tasc_one_pct_week(df):
    """TASC 2026.06 'One Percent A Week - Adaptive' (Dion Kurczek's variant; PineCoders
    https://www.tradingview.com/script/CSGmHoik-TASC-2026-06-One-Percent-A-Week-Adaptive/ ): buy at the week's open,
    hard stop 1.5%, target 7%. After Monday: open profit > 0.3% -> target x 1.011 (8.177%), profit <= 0 -> target 2.5%.
    Monday return > 2% and Tuesday return < 3% -> out after Tuesday. Out at the end of the week. Long only.
    Ours: the signal is the week's first bar (entry at the next bar's open, the reference = that first bar's close);
    targets are taken on a close at / over the level (exit mask) instead of an intrabar limit; the end of the week =
    Friday 15:00 New York (before the FTMO weekend rule); the week clock is New York + 7 h."""
    o, h, l, c, v = _ohlcv(df)
    wd, hr, wk = _wk_shift(df)
    first = np.r_[True, wk[1:] != wk[:-1]]
    ref = _s(np.where(first, c, np.nan)).groupby(wk).ffill().to_numpy()
    mon_c = _s(np.where(wd == 0, c, np.nan)).groupby(wk).ffill().to_numpy()
    tue_c = _s(np.where(wd == 1, c, np.nan)).groupby(wk).ffill().to_numpy()
    mon_ret = mon_c / ref - 1
    tue_ret = tue_c / ref - 1
    tmult = np.where(wd == 0, 1.07, np.where(mon_ret > 0.003, 1.07 * 1.011, np.where(mon_ret <= 0, 1.025, 1.07)))
    hit = c >= ref * tmult
    weak = (wd >= 2) & (mon_ret > 0.02) & (tue_ret < 0.03)
    fe = (wd == 4) & (hr >= 22.0)
    L = first & (wd <= 1)
    return L, np.zeros(len(c), bool), {"stop": ("level", c * 0.985, c * 1.015), "target": None,
                                        "exit_long": np.nan_to_num(hit | weak, nan=0).astype(bool), "force_exit": fe}


def s_tasc_ag_selling(df):
    """TASC 2026.08 'An Ag Selling Model' (Perry Kaufman; https://www.tradingview.com/script/VFzCk3HX-TASC-2026-08-An-Ag-Selling-Model/ ):
    sell when price reaches SMA 40 + 3 x ATR, never twice in the same rally. Short only (the model sells crops).
    Ours: ATR 20 (the length is not given); 'same rally' = no new sale until the close has been back under the SMA and
    at least 20 bars have passed; harvest has no equivalent -> cover on a close under the SMA 40 or after 120 bars;
    hard stop 2 ATR over the entry (Kaufman uses none)."""
    o, h, l, c, v = _ohlcv(df)
    m = sma(c, 40)
    a = atr(h, l, c, 20)
    lvl = m + 3.0 * a
    n = len(c)
    S = np.zeros(n, bool)
    last, reset = -10**9, True
    for i in range(n):
        if not (np.isfinite(lvl[i]) and np.isfinite(m[i])):
            continue
        if c[i] < m[i]:
            reset = True
        if c[i] >= lvl[i] and reset and i - last >= 20:
            S[i] = True
            last, reset = i, False
    return np.zeros(n, bool), S, {"stop": ("atr", 2.0), "target": None, "exit_short": c < m, "max_bars": 120}


def s_tasc_katsanos_pullback(df):
    """TASC 2026.10 'A Low-Risk ETF-Trading Strategy' (Markos Katsanos;
    https://www.tradingview.com/script/2S4BqQzQ-TASC-2026-10-A-Low-Risk-ETF-Trading-Strategy/ ), single-instrument
    version: RSMK 35 (EMA4 x 100 of the 35-bar log change) over its SMA 50, over +3, at a 4-bar high and under 25; RSI 10
    was under 35 within 20 bars and rose 3 bars in a row; EMA 5 over the SMA 15 of 5 bars ago -> buy. Out when RSMK
    crosses under -3, after 30 bars, or at +25%. Ours: RSMK on the instrument's own log price (the source measures it
    against SPY, which a single feed lacks), no SPY correlation / volume filters, hard stop 3 ATR (none published)."""
    o, h, l, c, v = _ohlcv(df)
    lc = np.log(c)
    rs = lc - _r(lc, 35)
    rk = ema(np.nan_to_num(rs), 4) * 100
    rk[:40] = np.nan
    r = rsi(c, 10)
    was_os = _s((r < 35).astype(float)).rolling(20, min_periods=1).max().to_numpy() > 0
    rising = (r > _r(r)) & (_r(r) > _r(r, 2)) & (_r(r, 2) > _r(r, 3))
    top4 = rk >= _s(rk).rolling(4).max().to_numpy()
    L = (rk > sma(rk, 50)) & (rk > 3) & top4 & (rk < 25) & was_os & rising & (ema(c, 5) > _r(sma(c, 15), 5))
    return L, np.zeros(len(c), bool), {"stop": ("atr", 3.0), "target": ("level", c * 1.25, c * 0.75),
                                        "exit_long": cross_down(np.nan_to_num(rk), -3.0), "max_bars": 30}


def s_tv_pivot_reversal(df):
    """TradingView built-in 'Pivot Reversal Strategy' (https://www.luxalgo.com/library/indicator/aqVvUNub-pivot-reversal-strategy/ , list https://www.tradingview.com/support/folders/43000587406-built-in-strategies/ ):
    pivot high / low with 4 bars left and 2 right; buy-stop a tick over the last pivot high, sell-stop under the last
    pivot low, stop and reverse. Ours: the stop entry as a close through the pivot (entry at the next open), the pivot
    used only after its 2 right bars have closed, a 2 ATR hard stop (the built-in has none)."""
    o, h, l, c, v = _ohlcv(df)
    h2, l2 = _r(h, 2), _r(l, 2)
    ph = h2 == _s(h).rolling(7).max().to_numpy()          # bar t-2 is the high of t-6..t (4 left, 2 right)
    pl = l2 == _s(l).rolling(7).min().to_numpy()
    last_h = _s(np.where(ph, _r(h, 2), np.nan)).ffill().to_numpy()
    last_l = _s(np.where(pl, _r(l, 2), np.nan)).ffill().to_numpy()
    L, S = _excl(cross_up(c, last_h), cross_down(c, last_l))
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


def s_tv_volty_expan_close(df):
    """TradingView / TradeStation 'Volty Expan Close' (https://ru.tradingview.com/support/solutions/43000599890 ,
    https://help.tradestation.com/09_01/tradestationhelp/Subsystems/elanalysis/signal/volty_expan_close_le_signal_.htm ):
    stop orders at the previous close +/- 0.75 x SMA 5 of the true range; stop and reverse. Ours: a close beyond the
    band (entry at the next open), 2 ATR hard stop."""
    o, h, l, c, v = _ohlcv(df)
    trn = np.maximum(h - l, np.maximum(np.abs(h - _r(c)), np.abs(l - _r(c))))
    band = 0.75 * _r(sma(np.nan_to_num(trn), 5))
    L, S = _excl(c > _r(c) + band, c < _r(c) - band)
    return _onset(L), _onset(S), {"stop": ("atr", 2.0), "target": None, "opposite": True}


def s_tv_consecutive_updown(df):
    """TradingView built-in 'Consecutive Up/Down Strategy' (https://www.tradingview.com/support/solutions/43000599836-consecutive-up-down-strategy/ ):
    buy after 3 closes in a row over the previous close, sell after 3 under; stop and reverse. Ours: 2 ATR hard stop."""
    o, h, l, c, v = _ohlcv(df)
    up = (c > _r(c)).astype(int)
    dn = (c < _r(c)).astype(int)
    ups = _s(up).groupby((up == 0).cumsum()).cumsum().to_numpy()
    dns = _s(dn).groupby((dn == 0).cumsum()).cumsum().to_numpy()
    L, S = _excl(ups == 3, dns == 3)
    return L, S, {"stop": ("atr", 2.0), "target": None, "opposite": True}


def s_hoffman_irb(df):
    """Rob Hoffman Inventory Retracement Bar (WH SelfInvest
    https://www.whselfinvest.de/en-de/trading-platform/trader-tools/technical-analysis/13-rob-hoffman-irb-inventory-retracement-bar ,
    open source https://www.tradingview.com/script/khxEdmoy-Noski-Rob-Hoffman-Inventory-Retracement-Bar ): in an
    uptrend, a bar that opens and closes at least 45% under its high (upper wick >= 45% of the range) is an IRB; buy when
    price trades over the IRB's high; a newer IRB replaces the old one. Mirror for downtrends. Ours: uptrend = close over
    an EMA 20 that rose at least 0.5 ATR over 5 bars (Hoffman's '45-degree' slope), the buy-stop as a close over the high
    within 20 bars, stop under the IRB's low (0.1 ATR), target 2R (not published)."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    e = ema(c, 20)
    up = (c > e) & (e - _r(e, 5) >= 0.5 * a)
    dn = (c < e) & (_r(e, 5) - e >= 0.5 * a)
    rng = h - l
    irb_l = up & (rng > 0) & (h - np.maximum(o, c) >= 0.45 * rng)
    irb_s = dn & (rng > 0) & (np.minimum(o, c) - l >= 0.45 * rng)
    n = len(c)
    L = np.zeros(n, bool)
    S = np.zeros(n, bool)
    sl = np.full(n, np.nan)
    ss = np.full(n, np.nan)
    bl = bs = None
    for i in range(n):
        if bl is not None and i - bl[0] <= 20 and c[i] > bl[1] and up[i]:
            L[i], sl[i] = True, bl[2] - 0.1 * a[i]
            bl = None
        if bs is not None and i - bs[0] <= 20 and c[i] < bs[2] and dn[i]:
            S[i], ss[i] = True, bs[1] + 0.1 * a[i]
            bs = None
        if irb_l[i]:
            bl = (i, h[i], l[i])
        if irb_s[i]:
            bs = (i, h[i], l[i])
    L, S = _excl(L, S)
    return L, S, {"stop": ("level", sl, ss), "target": ("R", 2.0)}


def _ny_rth(df):
    """New York regular session (09:30-16:00) per bar: in-session mask, session date, previous session's close /
    high / low, the session's running low / high before the bar, the session's open, and the bar-end hour."""
    hs, he, wd, _, day = _local(df, "America/New_York")
    o, h, l, c, v = _ohlcv(df)
    rth = (hs >= 9.5) & (he <= 16.0 + 1e-9) & (wd < 5)
    g = pd.DataFrame({"d": day, "h": np.where(rth, h, np.nan), "l": np.where(rth, l, np.nan),
                      "c": np.where(rth, c, np.nan), "o": np.where(rth, o, np.nan)})
    sess = g[rth].groupby("d").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), o=("o", "first"))
    prev = sess.shift(1)
    m = g[["d"]].join(prev, on="d")
    pdc, pdh, pdl = (m[k].to_numpy(dtype=float) for k in ("c", "h", "l"))
    sopen = g[["d"]].join(sess[["o"]], on="d")["o"].to_numpy(dtype=float)
    first_bar = rth & np.r_[True, day[1:] != day[:-1]] | (rth & ~np.r_[False, rth[:-1]])
    sopen = np.where(rth, _s(np.where(first_bar, o, np.nan)).groupby(day).ffill().to_numpy(), np.nan)
    lo_b = _s(np.where(rth, l, np.nan)).groupby(day).cummin().to_numpy()
    hi_b = _s(np.where(rth, h, np.nan)).groupby(day).cummax().to_numpy()
    return rth, day, pdc, pdh, pdl, _r(lo_b), _r(hi_b), sopen, he, sess


def s_rtr_red_to_green(df):
    """The Right Trader 'Red to Green Move' (https://therighttrader.com/learn/trading-strategies/red-to-green-move ;
    1-minute bars, 5-minute as the slower version, regular session): price traded under the previous session's close
    today; the first bar that closes back over it, on volume >= the previous bar's -> buy at the close; stop the signal
    bar's low; target the previous day's high, taken only when it is >= 2R away; skip when the risk is > 1% of price;
    out on a close back under the previous close or flat at the session close. Mirror 'green to red' short to the
    previous day's low. Ours: none beyond the bar timing (entry at the next bar's open)."""
    o, h, l, c, v = _ohlcv(df)
    rth, day, pdc, pdh, pdl, lo_b, hi_b, sopen, he, _ = _ny_rth(df)
    L = rth & (lo_b < pdc) & (c > pdc) & (v >= _r(v)) & (c - l <= 0.01 * c) & (pdh - c >= 2 * (c - l)) & (he < 15.9)
    S = rth & (hi_b > pdc) & (c < pdc) & (v >= _r(v)) & (h - c <= 0.01 * c) & (c - pdl >= 2 * (h - c)) & (he < 15.9)
    L = _first_per_day(np.nan_to_num(L).astype(bool), day)
    S = _first_per_day(np.nan_to_num(S).astype(bool), day)
    L, S = _excl(L, S)
    return L, S, {**_plan_lv(l, h, pdh, pdl), "exit_long": c < pdc, "exit_short": c > pdc,
                  "force_exit": he >= 16.0 - 1e-9}


def s_edgeful_streak_bias(df):
    """edgeful 'green and red streak' (https://www.edgeful.com/blog/posts/green-red-streak-trading-strategy ; YM, New
    York session): streaks average ~2 days, so after 2+ red sessions (close under the open) lean long, after 2+ green
    lean short - 'with price-action confirmation'. Ours (the source gives no trigger): the first 15 minutes of the
    session close with the bias (over the session open for longs) -> enter; stop the session's low (high) so far
    0.1 ATR beyond, target 2R, flat at 16:00 New York."""
    o, h, l, c, v = _ohlcv(df)
    rth, day, pdc, pdh, pdl, lo_b, hi_b, sopen, he, sess = _ny_rth(df)
    col = np.sign(sess["c"] - sess["o"]).to_numpy()
    red = np.zeros(len(col))
    grn = np.zeros(len(col))
    for k in range(len(col)):
        red[k] = (red[k - 1] + 1 if k else 1) if col[k] < 0 else 0
        grn[k] = (grn[k - 1] + 1 if k else 1) if col[k] > 0 else 0
    st = pd.DataFrame({"red": red, "grn": grn}, index=sess.index).shift(1)
    m = pd.DataFrame({"d": day}).join(st, on="d")
    rs, gs = m["red"].to_numpy(dtype=float), m["grn"].to_numpy(dtype=float)
    a = atr(h, l, c, 14)
    at15 = rth & (np.abs(he - 9.75) < 1e-6)
    lo_i = np.fmin(lo_b, l)
    hi_i = np.fmax(hi_b, h)
    L = at15 & (rs >= 2) & (c > sopen)
    S = at15 & (gs >= 2) & (c < sopen)
    L, S = _excl(np.nan_to_num(L).astype(bool), np.nan_to_num(S).astype(bool))
    return L, S, _plan_r(lo_i - 0.1 * a, hi_i + 0.1 * a, 2.0, force_exit=he >= 16.0 - 1e-9)


def s_voss_cross(df):
    """Ehlers Voss Predictive Filter (TASC 2019.08): the Voss predictor crosses over its band-pass filter while the filter
    is under zero (a cycle low turning) -> buy; mirror over zero -> sell; stop 1.5 ATR, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.voss_predict(df)
    vo, f = np.nan_to_num(z["voss"]), np.nan_to_num(z["filt"])
    L, S = _excl(cross_up(vo, f) & (f < 0), cross_down(vo, f) & (f > 0))
    return L, S, {"stop": ("atr", 1.5), "target": ("R", 2.0)}


def s_super_passband(df):
    """Ehlers Super Passband (TASC 2016.07): buy when the passband crosses up through -RMS, sell when it crosses down
    through +RMS; out on the opposite RMS cross (Ehlers' rules); stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.super_passband(df)
    pb, rm = np.nan_to_num(z["pb"]), np.nan_to_num(z["rms"])
    L, S = _excl(cross_up(pb, -rm), cross_down(pb, rm))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": cross_down(pb, rm), "exit_short": cross_up(pb, -rm)}


def s_universal_osc(df):
    """Ehlers Universal Oscillator (TASC 2015.01): zero crosses in the direction of the EMA 50 (ours), out on the
    opposite zero cross; stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    u = np.nan_to_num(IND.universal_osc(df))
    e = ema(c, 50)
    L, S = _excl(cross_up(u, 0.0) & (c > e), cross_down(u, 0.0) & (c < e))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": cross_down(u, 0.0), "exit_short": cross_up(u, 0.0)}


def s_elegant_osc(df):
    """Ehlers Elegant Oscillator (TASC 2022.02): buy when it turns up out of the -0.5 zone (crosses back over -0.5),
    sell when it crosses back under +0.5; stop beyond the 5-bar swing (0.1 ATR), target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    eo = np.nan_to_num(IND.elegant_osc(df))
    a = atr(h, l, c, 14)
    L, S = _excl(cross_up(eo, -0.5), cross_down(eo, 0.5))
    sl, ss = _swl(h, l, a, 5, 0.1)
    return L, S, _plan_r(sl, ss, 2.0)


def s_dsma_cross(df):
    """Ehlers Deviation-Scaled MA 40 (TASC 2018.07): close crosses the DSMA while the DSMA slopes the same way -> trade;
    out on the close back through it; stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    d = IND.dsma(df)
    dd = np.nan_to_num(d, nan=c[0] if len(c) else 0.0)
    L, S = _excl(cross_up(c, dd) & (d > _r(d)), cross_down(c, dd) & (d < _r(d)))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": c < d, "exit_short": c > d}


def s_trama_cross(df):
    """LuxAlgo TRAMA 99: close crosses the TRAMA while it slopes the same way and the trend regularity tc is over its
    100-bar median (ours: only regular trends); out on the close back through; stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.trama(df)
    t, tc = z["trama"], z["tc"]
    tt = np.nan_to_num(t, nan=c[0] if len(c) else 0.0)
    reg = tc >= _s(tc).rolling(100, min_periods=20).median().to_numpy()
    L, S = _excl(cross_up(c, tt) & (t > _r(t)) & reg, cross_down(c, tt) & (t < _r(t)) & reg)
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": c < t, "exit_short": c > t}


def s_impulse_macd(df):
    """LazyBear Impulse MACD 34 / 9: the impulse leaves zero upward (ZLEMA over the SMMA-high band) with the histogram
    positive -> buy; mirror -> sell; out when the impulse is back to zero (inside the band); stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.impulse_macd(df)
    md, hs = np.nan_to_num(z["md"]), np.nan_to_num(z["hist"])
    L, S = _excl(_onset(md > 0) & (hs > 0), _onset(md < 0) & (hs < 0))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": md <= 0, "exit_short": md >= 0}


def s_mad_zero(df):
    """Ehlers MAD 8 / 23 (TASC 2021.10): zero-line cross -> trade, opposite cross -> out; stop 2 ATR (ours)."""
    o, h, l, c, v = _ohlcv(df)
    m = np.nan_to_num(IND.mad(df))
    L, S = _excl(cross_up(m, 0.0), cross_down(m, 0.0))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": cross_down(m, 0.0), "exit_short": cross_up(m, 0.0)}


def s_spearman_pullback(df):
    """Spearman rank 20 (Ehlers): in an uptrend (close over EMA 200) the rank crosses back over -0.8 (a pullback ends)
    -> buy; mirror over +0.8 in a downtrend; stop beyond the 5-bar swing, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    sp = np.nan_to_num(IND.spearman(df))
    e = ema(c, 200)
    a = atr(h, l, c, 14)
    L, S = _excl(cross_up(sp, -0.8) & (c > e), cross_down(sp, 0.8) & (c < e))
    sl, ss = _swl(h, l, a, 5, 0.1)
    return L, S, _plan_r(sl, ss, 2.0)


def s_elder_thermometer(df):
    """Elder Market Thermometer: after 5+ quiet bars (thermometer under its EMA 22) a close over the prior 10-bar high
    (under the low) -> trade the breakout (Elder: quiet markets precede explosive moves); take profit on the first
    'hot' bar (thermometer > 3 x EMA, Elder's exit); stop 2 ATR, 20-bar time stop (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.market_thermometer(df)
    q = _r(z["quiet"].astype(float)) > 0
    hi = _r(_s(h).rolling(10).max().to_numpy())
    lo = _r(_s(l).rolling(10).min().to_numpy())
    L, S = _excl(q & (c > hi), q & (c < lo))
    return L, S, {"stop": ("atr", 2.0), "target": None, "exit_long": z["hot"], "exit_short": z["hot"], "max_bars": 20}


def s_williams_ad_div(df):
    """Larry Williams A/D divergence: the close makes a 20-bar low while Williams' A/D stays over its prior 20-bar low
    (no confirmation) -> buy; mirror at a 20-bar high; stop 1.5 ATR, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    w = IND.williams_ad(df)
    lo_c = c <= _s(c).rolling(20).min().to_numpy()
    hi_c = c >= _s(c).rolling(20).max().to_numpy()
    wl = _r(_s(w).rolling(20).min().to_numpy())
    wh = _r(_s(w).rolling(20).max().to_numpy())
    L, S = _excl(lo_c & (w > wl), hi_c & (w < wh))
    return _onset(L), _onset(S), {"stop": ("atr", 1.5), "target": ("R", 2.0)}


def s_fib_bollinger_fade(df):
    """Fibonacci Bollinger Bands (Rashad): the bar pierces the outer 1.0 band and closes back inside it as a reversal bar
    -> fade; stop beyond the bar (0.25 ATR); target the 0.382 band on the same side (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.fib_bollinger(df)
    a = atr(h, l, c, 14)
    L = (l <= z["dn_1000"]) & (c > z["dn_1000"]) & (c > o)
    S = (h >= z["up_1000"]) & (c < z["up_1000"]) & (c < o)
    L, S = _excl(L, S)
    return L, S, _plan_lv(l - 0.25 * a, h + 0.25 * a, z["dn_382"], z["up_382"])


def s_hull_butterfly(df):
    """Hull Butterfly Oscillator (LuxAlgo, our reconstruction): green dot (turn up under the lower level) -> buy, red
    dot -> sell; stop 2 ATR, target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.hull_butterfly(df)
    L, S = _excl(z["dot_buy"], z["dot_sell"])
    return L, S, {"stop": ("atr", 2.0), "target": ("R", 2.0)}


def s_range_detector_break(df):
    """Range Detector (LuxAlgo, our reconstruction): a close through the top (bottom) of a range that held 20+ bars ->
    trade the break; stop at the range's middle; target 2R (ours)."""
    o, h, l, c, v = _ohlcv(df)
    z = IND.range_detector(df)
    mid = _r(z["mid"])
    ok = _r(z["range_len"]) >= 20
    L, S = _excl(z["break_up"] & ok, z["break_dn"] & ok)
    return L, S, _plan_r(mid, mid, 2.0)


SCOUT.update({
    "SC504": ("One Percent A Week - Adaptive: buy the week's open, 1.5% stop, 7% target -> 8.177% / 2.5% after Monday, Tuesday weakness exit, out Friday (TASC 2026.06)", s_tasc_one_pct_week, ["15Min", "1Hour", "4Hour"]),
    "SC505": ("Kaufman Ag selling model: short at SMA40 + 3 ATR once per rally, cover under the SMA / 120 bars, 2 ATR stop (TASC 2026.08)", s_tasc_ag_selling, ["1Hour", "4Hour", "1Day"]),
    "SC506": ("Katsanos low-risk pullback: RSMK 35 strength + RSI10 oversold within 20 and rising 3 bars + EMA5 > SMA15[5], out RSMK < -3 / 30 bars / +25% (TASC 2026.10)", s_tasc_katsanos_pullback, ["4Hour", "1Day"]),
    "SC507": ("Pivot Reversal 4/2: close through the last confirmed pivot high / low, stop and reverse, 2 ATR (TradingView built-in)", s_tv_pivot_reversal, _F + ["4Hour"]),
    "SC508": ("Volty Expan Close: close beyond previous close +/- 0.75 x SMA5(TR), stop and reverse, 2 ATR (TradingView / TradeStation built-in)", s_tv_volty_expan_close, _F + ["4Hour"]),
    "SC509": ("Consecutive Up/Down 3/3: momentum after 3 closes in a row, stop and reverse, 2 ATR (TradingView built-in)", s_tv_consecutive_updown, _F + ["4Hour"]),
    "SC510": ("Rob Hoffman Inventory Retracement Bar: 45% wick bar in a steep EMA20 trend, close through its high within 20 bars, IRB stop, 2R", s_hoffman_irb, ["5Min", "15Min", "1Hour", "4Hour"]),
    "SC511": ("Red to Green / Green to Red: first close back over the previous RTH close on rising volume, bar stop, previous-day high if >= 2R (The Right Trader)", s_rtr_red_to_green, ["1Min", "5Min"]),
    "SC512": ("edgeful streak bias: after 2+ red (green) NY sessions, first 15 minutes closing with the bias, session-extreme stop, 2R, flat 16:00", s_edgeful_streak_bias, ["5Min", "15Min"]),
    "SC513": ("Ehlers Voss predictor crosses the band-pass filter at a cycle low / high, 1.5 ATR, 2R (TASC 2019.08)", s_voss_cross, _F + ["4Hour"]),
    "SC514": ("Ehlers Super Passband crosses -RMS / +RMS, opposite-RMS exit, 2 ATR (TASC 2016.07)", s_super_passband, _F + ["4Hour"]),
    "SC515": ("Ehlers Universal Oscillator zero cross with the EMA50, opposite-cross exit, 2 ATR (TASC 2015.01)", s_universal_osc, _F + ["4Hour"]),
    "SC516": ("Ehlers Elegant Oscillator back over -0.5 / under +0.5, swing stop, 2R (TASC 2022.02)", s_elegant_osc, _F + ["4Hour"]),
    "SC517": ("Close crosses the Ehlers Deviation-Scaled MA 40 with its slope, cross-back exit, 2 ATR (TASC 2018.07)", s_dsma_cross, _F + ["4Hour"]),
    "SC518": ("Close crosses LuxAlgo TRAMA 99 with its slope while trend regularity is high, cross-back exit, 2 ATR", s_trama_cross, _M + ["1Day"]),
    "SC519": ("LazyBear Impulse MACD leaves zero with the histogram, out back at zero, 2 ATR", s_impulse_macd, _F + ["4Hour"]),
    "SC520": ("Ehlers MAD 8/23 zero cross, opposite-cross exit, 2 ATR (TASC 2021.10)", s_mad_zero, _F + ["4Hour"]),
    "SC521": ("Spearman rank 20 back over -0.8 (under +0.8) with the EMA200 trend, swing stop, 2R (Ehlers)", s_spearman_pullback, _F + ["4Hour"]),
    "SC522": ("Elder Market Thermometer: 10-bar breakout after 5 quiet bars, out on the first hot bar / 20 bars, 2 ATR", s_elder_thermometer, _F + ["4Hour"]),
    "SC523": ("Williams A/D divergence at a 20-bar close low / high, 1.5 ATR, 2R", s_williams_ad_div, _F + ["4Hour"]),
    "SC524": ("Fibonacci Bollinger Bands: reversal bar back inside the 1.0 band, bar stop, 0.382-band target (Rashad)", s_fib_bollinger_fade, _F + ["4Hour"]),
    "SC525": ("Hull Butterfly Oscillator turn dots beyond the adaptive levels, 2 ATR, 2R (LuxAlgo, reconstruction)", s_hull_butterfly, _F + ["4Hour"]),
    "SC526": ("Range Detector: close through a 20+ bar range (SMA20 +/- ATR500), range-mid stop, 2R (LuxAlgo, reconstruction)", s_range_detector_break, _F + ["4Hour"]),
})


# ---------------- SC527-SC530 the range pack: trade only while the 1-hour chart is choppy (user, 16-17.11.2026) ----------
# "If it is choppy, there are strategies for chop - I did not see them act yesterday." Almost every strategy the bots
# trade looks for a trend; on 16.11 (gold sideways all day) nothing earned. These four trade the range ONLY while the
# 1-hour chart is choppy - Choppiness Index(14) over 61.8 or ADX(14) under 20, read on CLOSED hours (no look-ahead) -
# so they add trades on the days the trend strategies lose. Our own pack (classic range rules); the pool tests each on
# every bot's instruments and timeframes.
def _chop_1h(df):
    def ci(g):
        h, l, c = (g[x].to_numpy(dtype=float) for x in "hlc")
        pc = np.r_[c[0], c[:-1]]
        tr = np.maximum(h, pc) - np.minimum(l, pc)
        s = pd.Series(tr).rolling(14).sum()
        rng = pd.Series(h).rolling(14).max() - pd.Series(l).rolling(14).min()
        return (100 * np.log10(s / rng.replace(0, np.nan)) / np.log10(14)).to_numpy()

    def ax(g):
        return adx(g["h"].to_numpy(dtype=float), g["l"].to_numpy(dtype=float), g["c"].to_numpy(dtype=float), 14)

    with np.errstate(invalid="ignore"):
        return (_htf(df, "1h", ci) > 61.8) | (_htf(df, "1h", ax) < 20)


def s_range_bb_reentry(df):
    """Choppy 1-hour chart: a close back inside the Bollinger band (20, 2) after a close outside it - buy back above
    the lower band, sell back under the upper. Stop beyond the last 10 bars' extreme + 0.2 ATR, target the middle
    band (also an exit when the close reaches it)."""
    o, h, l, c, v = _ohlcv(df)
    mid = sma(c, 20)
    sd = _s(c).rolling(20).std(ddof=0).to_numpy()
    lo, up = mid - 2 * sd, mid + 2 * sd
    a = atr(h, l, c, 14)
    ch = _chop_1h(df)
    pc, plo, pup = np.r_[np.nan, c[:-1]], np.r_[np.nan, lo[:-1]], np.r_[np.nan, up[:-1]]
    L = ch & (pc < plo) & (c > lo) & (c < mid)
    S = ch & (pc > pup) & (c < up) & (c > mid)
    L, S = _excl(L, S)
    ll, hh = _s(l).rolling(10).min().to_numpy(), _s(h).rolling(10).max().to_numpy()
    return L, S, _plan_lv(ll - 0.2 * a, hh + 0.2 * a, mid, mid, exit_long=c >= mid, exit_short=c <= mid)


def s_range_edge_reject(df):
    """Choppy 1-hour chart, range = the last 24 bars' high / low (at least 2 ATR wide): a bar that reaches the lowest
    15% of the range and closes up in its upper half (a rejection) -> buy; the mirror at the top -> sell. Stop
    beyond the range edge + 0.25 ATR, target the range's 0.618 level (Fibonacci) from the entry side."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    hi = _s(h).shift(1).rolling(24).max().to_numpy()
    lo = _s(l).shift(1).rolling(24).min().to_numpy()
    rg = hi - lo
    ch = _chop_1h(df) & (rg >= 2 * a)
    mid_bar = (h + l) / 2
    L = ch & (l <= lo + 0.15 * rg) & (c > mid_bar) & (c > o) & (c < lo + 0.5 * rg)
    S = ch & (h >= hi - 0.15 * rg) & (c < mid_bar) & (c < o) & (c > lo + 0.5 * rg)
    L, S = _excl(L, S)
    return L, S, _plan_lv(np.minimum(lo, l) - 0.25 * a, np.maximum(hi, h) + 0.25 * a,
                          lo + 0.618 * rg, hi - 0.618 * rg)


def s_range_rsi2(df):
    """Choppy 1-hour chart: RSI(2) under 10 -> buy, over 90 -> sell (a short-term stretch inside a range). Stop 1.5
    ATR, target 1.5 ATR, out when the close crosses the 5-bar average back."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    r2 = rsi(c, 2)
    ch = _chop_1h(df)
    L, S = _excl(ch & (r2 < 10), ch & (r2 > 90))
    s5 = sma(c, 5)
    return L, S, _plan_lv(c - 1.5 * a, c + 1.5 * a, c + 1.5 * a, c - 1.5 * a, exit_long=c > s5, exit_short=c < s5)


def s_range_vwap_revert(df):
    """Choppy 1-hour chart: price 1.5 ATR or more under the day's VWAP and a bar that closes up -> buy back toward the
    VWAP; the mirror over it -> sell. Stop 1 ATR beyond the bar's extreme, target the VWAP."""
    o, h, l, c, v = _ohlcv(df)
    a = atr(h, l, c, 14)
    vw = _vwap_day(df)
    ch = _chop_1h(df)
    L = ch & (c <= vw - 1.5 * a) & (c > o)
    S = ch & (c >= vw + 1.5 * a) & (c < o)
    L, S = _excl(L, S)
    return L, S, _plan_lv(l - a, h + a, vw, vw, exit_long=c >= vw, exit_short=c <= vw)


SCOUT.update({
    "SC527": ("Range pack: Bollinger 20/2 re-entry only while the 1-hour chart is choppy (CI > 61.8 or ADX < 20), 10-bar extreme stop, middle-band target (ours, 17.11.2026)", s_range_bb_reentry, ["5Min", "15Min", "1Hour"]),
    "SC528": ("Range pack: 24-bar range edge rejection while the 1-hour chart is choppy, edge stop, Fibonacci 0.618 target (ours, 17.11.2026)", s_range_edge_reject, ["5Min", "15Min", "1Hour"]),
    "SC529": ("Range pack: RSI(2) under 10 / over 90 while the 1-hour chart is choppy, 1.5 ATR, out at the 5-bar average (ours, 17.11.2026)", s_range_rsi2, ["5Min", "15Min", "1Hour"]),
    "SC530": ("Range pack: 1.5 ATR away from the day's VWAP with a turning bar while the 1-hour chart is choppy, VWAP target (ours, 17.11.2026)", s_range_vwap_revert, ["5Min", "15Min", "1Hour"]),
})
