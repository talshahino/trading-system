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


# id: (name, fn, timeframes the source states)
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
}
