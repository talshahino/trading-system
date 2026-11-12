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
}
