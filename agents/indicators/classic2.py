"""Classic indicators the library did not have yet (scout, 14-15.11.2026). Public formulas - each function's
docstring names its source (StockCharts ChartSchool, the author's original publication, StockSharp, LuxAlgo concept
pages). Every value at bar i uses only bars <= i (closed bars); daily / session values are taken from completed
sessions only."""
import numpy as np
import pandas as pd

from agents.indicators._core import S, atr, cci, cols, ema, highest, lowest, rma, shift, sma, stdev, tr, wma

SC = "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/"


def _roc(c, n):
    p = shift(c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (c - p) / p


def coppock(df, r1=14, r2=11, w=10):
    """Coppock Curve (E. Coppock 1962; StockCharts coppock-curve): WMA(10) of ROC(14) + ROC(11) (monthly in the
    original; any bar here). dict: cc, turn_up (rises after falling, below zero = Coppock's buy), turn_dn."""
    o, h, l, c, v = cols(df)
    cc = wma(_roc(c, r1) + _roc(c, r2), w)
    p1, p2 = shift(cc, 1), shift(cc, 2)
    with np.errstate(invalid="ignore"):
        return {"cc": cc, "turn_up": (cc > p1) & (p1 <= p2), "turn_dn": (cc < p1) & (p1 >= p2)}


def ultimate_oscillator(df, n1=7, n2=14, n3=28):
    """Ultimate Oscillator (Larry Williams 1976; StockCharts ultimate-oscillator): BP = close - min(low, prev close),
    TR = max(high, prev close) - min(low, prev close); UO = 100 x (4 A7 + 2 A14 + A28) / 7, A = sum BP / sum TR."""
    o, h, l, c, v = cols(df)
    pc = np.r_[c[0], c[:-1]]
    bp = c - np.minimum(l, pc)
    t = np.maximum(h, pc) - np.minimum(l, pc)
    av = [S(bp).rolling(n).sum().to_numpy() / np.maximum(S(t).rolling(n).sum().to_numpy(), 1e-12) for n in (n1, n2, n3)]
    return 100.0 * (4 * av[0] + 2 * av[1] + av[2]) / 7.0


def klinger(df, fast=34, slow=55, sig=13):
    """Klinger Volume Oscillator (S. Klinger; StockCharts klinger-volume-oscillator): trend = sign of the change in
    hlc3, dm = h - l, cm accumulates dm while the trend holds (resets to prev dm + dm on a change), VF = V x |2 dm/cm
    - 1| x trend x 100; KVO = EMA34(VF) - EMA55(VF), signal EMA13. dict: kvo, signal."""
    o, h, l, c, v = cols(df)
    hlc = (h + l + c) / 3
    n = len(c)
    trend = np.ones(n)
    dm = h - l
    cm = np.zeros(n)
    vf = np.zeros(n)
    for i in range(n):
        if i:
            trend[i] = 1.0 if hlc[i] > hlc[i - 1] else -1.0 if hlc[i] < hlc[i - 1] else trend[i - 1]
            cm[i] = cm[i - 1] + dm[i] if trend[i] == trend[i - 1] else dm[i - 1] + dm[i]
        else:
            cm[i] = dm[i]
        vf[i] = v[i] * abs(2 * (dm[i] / cm[i] if cm[i] else 0.0) - 1) * trend[i] * 100
    kvo = ema(vf, fast) - ema(vf, slow)
    return {"kvo": kvo, "signal": ema(kvo, sig)}


def mass_index(df, n=9, s=25):
    """Mass Index (Donald Dorsey; StockCharts mass-index): sum over 25 bars of EMA9(h-l) / EMA9(EMA9(h-l)). dict:
    mi, bulge_end (the 'reversal bulge': MI went over 27 and now falls under 26.5)."""
    o, h, l, c, v = cols(df)
    e1 = ema(h - l, n)
    r = e1 / np.maximum(ema(e1, n), 1e-12)
    mi = S(r).rolling(s).sum().to_numpy()
    over = S((mi > 27).astype(float)).rolling(25, min_periods=1).max().to_numpy() > 0
    with np.errstate(invalid="ignore"):
        return {"mi": mi, "bulge_end": over & (mi < 26.5) & (shift(mi, 1) >= 26.5)}


def mcginley(df, n=20, k=0.6):
    """McGinley Dynamic (John McGinley 1990; StockSharp 1038 'McGinley Dynamic Improved' code: k 0.6, exponent 4):
    MD = MD[-1] + (c - MD[-1]) / (k x n x (c / MD[-1])^4)."""
    o, h, l, c, v = cols(df)
    md = np.empty(len(c))
    if len(c):
        md[0] = c[0]
    for i in range(1, len(c)):
        p = md[i - 1]
        md[i] = p + (c[i] - p) / max(k * n * (c[i] / p) ** 4, 1e-9) if p else c[i]
    return md


def elder_ray(df, n=13):
    """Elder Ray (Alexander Elder; https://www.luxalgo.com/library/concept/elder-ray/): EMA13, bull power = high - EMA,
    bear power = low - EMA. dict: ema, bull, bear, buy (EMA rising, bear power < 0 and rising - Elder's buy),
    sell (EMA falling, bull power > 0 and falling)."""
    o, h, l, c, v = cols(df)
    e = ema(c, n)
    bull, bear = h - e, l - e
    with np.errstate(invalid="ignore"):
        up, dn = e > shift(e, 1), e < shift(e, 1)
        return {"ema": e, "bull": bull, "bear": bear, "buy": up & (bear < 0) & (bear > shift(bear, 1)),
                "sell": dn & (bull > 0) & (bull < shift(bull, 1))}


def volatility_system(df, n=7, mult=3.0):
    """Welles Wilder's Volatility System (New Concepts in Technical Trading Systems, 1978; open-source port
    https://www.tradingview.com/script/pjZKWnHl-Volatility-System-by-Wilder-LucF ): ARC = ATR(7) x 3.0; long: SAR =
    highest close since entry - ARC, a close under it reverses to short; short: SAR = lowest close + ARC. dict: sar,
    dir (+1/-1), flip_up, flip_dn."""
    o, h, l, c, v = cols(df)
    arc = mult * rma(tr(h, l, c), n)
    m = len(c)
    sar = np.full(m, np.nan)
    d = np.zeros(m)
    if m == 0:
        return {"sar": sar, "dir": d, "flip_up": d > 0, "flip_dn": d > 0}
    t, ext = 1, c[0]
    for i in range(m):
        if not np.isfinite(arc[i]) or i < n:
            d[i] = 0
            ext = c[i]
            continue
        if t == 1:
            ext = max(ext, c[i])
            s = ext - arc[i]
            if c[i] < s:
                t, ext = -1, c[i]
                s = ext + arc[i]
        else:
            ext = min(ext, c[i])
            s = ext + arc[i]
            if c[i] > s:
                t, ext = 1, c[i]
                s = ext - arc[i]
        sar[i], d[i] = s, t
    pd_ = shift(d, 1)
    return {"sar": sar, "dir": d, "flip_up": (d == 1) & (pd_ == -1), "flip_dn": (d == -1) & (pd_ == 1)}


def frama(df, n=16, fc=1, sc=198):
    """Fractal Adaptive Moving Average (John Ehlers, 'FRAMA', Stocks & Commodities Oct 2005 - Traders' Tips
    https://www.traders.com/Documentation/FEEDbk_docs/2005/10/TradersTips/TradersTips.html ): fractal dimension D from
    the two half-window ranges vs the full window, alpha = exp(w (D - 1)) with w = ln(2 / (SC + 1)) (= -4.6 for SC 198, Ehlers' constant; the
    'modified FRAMA' form), limited to [2/(SC+1), 2/(FC+1)], FRAMA = alpha x price + (1 - alpha) x FRAMA[-1]; price = hl2."""
    o, h, l, c, v = cols(df)
    p = (h + l) / 2
    n = int(n) // 2 * 2
    hh = n // 2
    n1 = (highest(h, hh) - lowest(l, hh)) / hh
    n2 = shift(n1, hh)
    n3 = (highest(h, n) - lowest(l, n)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        dim = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    w = np.log(2.0 / (sc + 1))
    with np.errstate(invalid="ignore", over="ignore"):
        alpha = np.exp(w * (dim - 1))
    alpha = np.clip(np.nan_to_num(alpha, nan=2.0 / (sc + 1)), 2.0 / (sc + 1), 2.0 / (fc + 1))
    out = np.empty(len(c))
    if len(c):
        out[0] = p[0]
    for i in range(1, len(c)):
        out[i] = alpha[i] * p[i] + (1 - alpha[i]) * out[i - 1]
    return out


def hurst(df, n=100):
    """Rolling Hurst exponent by rescaled range over the last n log returns (StockSharp 0228 'Hurst Exponent
    Reversion': R/S over 100 bars, < threshold = mean-reverting). Single-window R/S estimate H = log(R/S) / log(n)
    (as in that example; noisy by nature - the StockSharp page uses 0.7 as the threshold)."""
    o, h, l, c, v = cols(df)
    r = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.nan)

    def f(x):
        x = x[np.isfinite(x)]
        if len(x) < 10:
            return np.nan
        y = np.cumsum(x - x.mean())
        rs = (y.max() - y.min()) / (x.std() or np.nan)
        return np.log(rs) / np.log(len(x)) if rs and np.isfinite(rs) and rs > 0 else np.nan
    return S(r).rolling(n, min_periods=n).apply(f, raw=True).to_numpy()


def linreg_channel(df, n=50, dev=1.5):
    """Linear Regression Channel (StockSharp designer gallery 'Linear Regression Channel': length 50, regression line
    +/- 1.5 standard errors (divided by n - 2), slope over 50). dict: mid (the line's value at bar i), slope, upper,
    lower."""
    o, h, l, c, v = cols(df)
    x = np.arange(n, dtype=float)
    xm, sx = x.mean(), ((x - x.mean()) ** 2).sum()
    m = len(c)
    mid, slope, se = np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan)
    if m >= n:
        win = np.lib.stride_tricks.sliding_window_view(c, n)
        ym = win.mean(axis=1)
        b = ((win - ym[:, None]) * (x - xm)).sum(axis=1) / sx
        a = ym - b * xm
        fit = a[:, None] + b[:, None] * x
        res = np.sqrt(((win - fit) ** 2).sum(axis=1) / max(n - 2, 1))
        mid[n - 1:], slope[n - 1:], se[n - 1:] = a + b * (n - 1), b, res
    return {"mid": mid, "slope": slope, "upper": mid + dev * se, "lower": mid - dev * se}


def renko_trend(df, n=14, mult=0.5):
    """Renko bricks on closes (StockSharp 1608 'Renko Trend Reversal' code: brick = StdDev(14) x 0.5, frozen when a
    brick forms; a new up brick when the close reaches the last brick's top + size, down when it reaches the bottom
    - size; ours: one brick per bar at most, bricks only on closed bars). dict: dir (+1/-1 last brick), new_up,
    new_dn, rev_up (an up brick after a down brick), rev_dn, top, bot."""
    o, h, l, c, v = cols(df)
    sd = stdev(c, n) * mult
    m = len(c)
    d = np.zeros(m)
    top, bot = np.full(m, np.nan), np.full(m, np.nan)
    nu, nd = np.zeros(m, bool), np.zeros(m, bool)
    t = 0
    hi = lo = np.nan
    size = np.nan
    for i in range(m):
        if not np.isfinite(hi):
            if np.isfinite(sd[i]) and sd[i] > 0:
                hi = lo = c[i]
                size = sd[i]
        else:
            if c[i] >= hi + size:
                lo, hi = hi, c[i]
                nu[i], t = True, 1
                size = sd[i] if np.isfinite(sd[i]) and sd[i] > 0 else size
            elif c[i] <= lo - size:
                hi, lo = lo, c[i]
                nd[i], t = True, -1
                size = sd[i] if np.isfinite(sd[i]) and sd[i] > 0 else size
        d[i], top[i], bot[i] = t, hi, lo
    pd_ = shift(d, 1)
    return {"dir": d, "new_up": nu, "new_dn": nd, "rev_up": nu & (pd_ == -1), "rev_dn": nd & (pd_ == 1),
            "top": top, "bot": bot}


def woodies_cci(df, fast=6, slow=14):
    """Woodies CCI (Ken Wood's CCI club; StockSharp 2263 'CCI Woodies': 'turbo' CCI 6 and CCI 14; zero-line reject
    and trend side per https://www.luxalgo.com/library/concept/woodies-cci-conventions/ ). dict: turbo, cci, trend
    (+1 after 6 bars over zero, -1 after 6 under), zlr_long (trend up, CCI dipped toward zero (0..+100 / -50) and
    turns up), zlr_short."""
    o, h, l, c, v = cols(df)
    t6, t14 = cci(h, l, c, fast), cci(h, l, c, slow)
    above = S((t14 > 0).astype(float)).rolling(6).min().to_numpy() > 0
    below = S((t14 < 0).astype(float)).rolling(6).min().to_numpy() > 0
    tr_ = np.zeros(len(c))
    cur = 0
    for i in range(len(c)):
        if above[i]:
            cur = 1
        elif below[i]:
            cur = -1
        tr_[i] = cur
    p = shift(t14, 1)
    with np.errstate(invalid="ignore"):
        zl = (tr_ == 1) & (p > -50) & (p < 100) & (t14 > p) & (p < shift(t14, 2))
        zs = (tr_ == -1) & (p < 50) & (p > -100) & (t14 < p) & (p > shift(t14, 2))
    return {"turbo": t6, "cci": t14, "trend": tr_, "zlr_long": zl, "zlr_short": zs}


def _daily_hlc(df, tz="UTC"):
    """The previous completed day's high, low, close and today's open per bar (local day in tz)."""
    o, h, l, c, v = cols(df)
    day = pd.Series(df.index.tz_convert(tz).date, index=df.index)
    g = pd.DataFrame({"h": h, "l": l, "c": c, "o": o}, index=df.index).groupby(day.to_numpy())
    dd = pd.DataFrame({"h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(), "o": g["o"].first()})
    prev = dd.shift(1)
    k = day.to_numpy()
    return (prev["h"].reindex(k).to_numpy(), prev["l"].reindex(k).to_numpy(), prev["c"].reindex(k).to_numpy(),
            dd["o"].reindex(k).to_numpy())


def woodie_pivots(df, tz="UTC"):
    """Woodie pivot points (https://www.luxalgo.com/library/concept/woodie-pivots/ ; DailyFX 'Trading with Woodie's
    pivots'): PP = (H + L + 2 x today's OPEN) / 4 from the previous day's H / L, R1 = 2 PP - L, S1 = 2 PP - H,
    R2 = PP + (H - L), S2 = PP - (H - L). Known from the day's first bar (today's open is that bar's open)."""
    H, L, C, O = _daily_hlc(df, tz)
    pp = (H + L + 2 * O) / 4
    return {"pp": pp, "r1": 2 * pp - L, "s1": 2 * pp - H, "r2": pp + (H - L), "s2": pp - (H - L)}


def obv(df, sig=20):
    """On-Balance Volume (Joe Granville; StockCharts on-balance-volume) on tick volume. dict: obv, ema (20),
    hi20 / lo20 (its prior 20-bar extremes, for breakouts)."""
    o, h, l, c, v = cols(df)
    ob = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v)
    return {"obv": ob, "ema": ema(ob, sig), "hi20": shift(highest(ob, 20), 1), "lo20": shift(lowest(ob, 20), 1)}


def eom(df, n=14, scale=1e4):
    """Ease of Movement (Richard Arms; StockCharts ease-of-movement-emv): distance = ((h + l)/2 - prior (h + l)/2),
    box ratio = (v / scale) / (h - l), EMV = distance / box ratio, smoothed SMA14 (tick volume)."""
    o, h, l, c, v = cols(df)
    m = (h + l) / 2
    dist = m - shift(m, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        box = (np.maximum(v, 1e-9) / scale) / np.where(h - l > 0, h - l, np.nan)
        e = dist / box
    return sma(np.nan_to_num(e), n)


def ulcer_index(df, n=14):
    """Ulcer Index (Peter Martin 1987; StockCharts ulcer-index): sqrt(mean over n of the squared % drawdown from the
    n-bar highest close)."""
    o, h, l, c, v = cols(df)
    dd = 100 * (c - highest(c, n)) / np.maximum(highest(c, n), 1e-12)
    return np.sqrt(S(dd ** 2).rolling(n).mean().to_numpy())


def smart_money_index(df, tz="America/New_York", open_h=9.5, close_h=16.0):
    """Smart Money Index (Don Hays; https://www.luxalgo.com/library/concept/smart-money-index/): SMI = yesterday's SMI
    - (the first 30 minutes' change) + (the last hour's change). Intraday bars of a US index / stock only; the value
    of a day is known from the day's last bar on, and is carried into the next day. dict: smi (per bar, last
    completed day's value), smi_day (pandas Series by day)."""
    o, h, l, c, v = cols(df)
    t = df.index.tz_convert(tz)
    hr = t.hour + t.minute / 60.0
    bm = float(pd.Series(df.index).diff().median().total_seconds() / 60.0) if len(df) > 1 else 5.0
    day = pd.Series(t.date, index=df.index)
    f = pd.DataFrame({"o": o, "c": c, "hr": hr, "d": day.to_numpy()}, index=df.index)
    first = f[(f.hr >= open_h) & (f.hr < open_h + 0.5)].groupby("d").agg(o=("o", "first"), c=("c", "last"))
    last = f[(f.hr >= close_h - 1.0) & (f.hr + bm / 60.0 <= close_h + 1e-9)].groupby("d").agg(o=("o", "first"), c=("c", "last"))
    j = first.join(last, lsuffix="_f", rsuffix="_l", how="inner")
    step = -(j["c_f"] - j["o_f"]) + (j["c_l"] - j["o_l"])
    smi_day = step.cumsum()
    # a day's value is usable from the next day on (the last hour ends at the close)
    keys = np.array(list(smi_day.index), dtype="datetime64[D]")
    dd = day.to_numpy().astype("datetime64[D]")
    pos = np.searchsorted(keys, dd, side="left") - 1          # the last completed day strictly before this bar's day
    vals = np.where(pos >= 0, smi_day.to_numpy(dtype=float)[np.clip(pos, 0, None)] if len(keys) else np.nan, np.nan)
    return {"smi": vals, "smi_day": smi_day}
