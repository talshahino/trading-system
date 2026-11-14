"""Classic indicators the user's videos of 14.11.2026 name, as shared library functions (standard public formulas;
each docstring has its reference). No repaint: bar i uses bars <= i (Ichimoku's Chikou is compared with the price
26 bars AGO, never with the future)."""
import numpy as np

from agents.indicators._core import (S, atr, cols, cross_down, cross_up, dmi, ema, highest, lowest, rma, rsi, shift,
                                     sma, stdev)


def ichimoku(df, tenkan=9, kijun=26, senkou=52, disp=26):
    """Ichimoku Kinko Hyo (https://www.tradingview.com/support/solutions/43000589152-ichimoku-cloud/ ; Quant Tactics
    video, task tal1411vwap (3)). The cloud at bar i = spans computed disp bars ago (that is what is drawn at i).
    Chikou check without look-ahead: close[i] vs the close disp bars ago. dict: tenkan, kijun, span_a, span_b,
    above_cloud, below_cloud, tk_up, tk_dn, chikou_up (close > close[disp]), chikou_dn."""
    o, h, l, c, v = cols(df)
    tk = (highest(h, tenkan) + lowest(l, tenkan)) / 2
    kj = (highest(h, kijun) + lowest(l, kijun)) / 2
    sa = shift((tk + kj) / 2, disp)
    sb = shift((highest(h, senkou) + lowest(l, senkou)) / 2, disp)
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    pc = shift(c, disp)
    return {"tenkan": tk, "kijun": kj, "span_a": sa, "span_b": sb, "above_cloud": c > top, "below_cloud": c < bot,
            "tk_up": cross_up(tk, kj), "tk_dn": cross_down(tk, kj), "chikou_up": c > pc, "chikou_dn": c < pc}


def mfi(df, n=14):
    """Money Flow Index (https://www.tradingview.com/support/solutions/43000502348-money-flow-index-mfi/ ; Quant
    Tactics MFI video, task tal1411set3 (2)) on tick volume. dict: mfi, exit_os (crosses back above 20),
    exit_ob (crosses back under 80)."""
    o, h, l, c, v = cols(df)
    tp = (h + l + c) / 3
    mf = tp * np.where(v > 0, v, 1.0)
    d = np.diff(tp, prepend=tp[0])
    pos = S(np.where(d > 0, mf, 0.0)).rolling(n).sum().to_numpy()
    neg = S(np.where(d < 0, mf, 0.0)).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(pos + neg > 0, 100 * pos / (pos + neg), 50.0)
    return {"mfi": m, "exit_os": cross_up(m, 20.0), "exit_ob": cross_down(m, 80.0)}


def adx_di(df, n=14):
    """ADX with +DI / -DI (Wilder; The Trader's Guide ADX + DI + EMA50, task tal1411set6 (11)). dict: pdi, ndi, adx,
    di_up (+DI crosses over -DI), di_dn, adx_rising."""
    o, h, l, c, v = cols(df)
    p, m, a = dmi(h, l, c, n)
    return {"pdi": p, "ndi": m, "adx": a, "di_up": cross_up(p, m), "di_dn": cross_down(p, m),
            "adx_rising": a > shift(a, 1)}


def bb_width_pct(df, n=20, k=2.0, look=100):
    """Bollinger Band width and its percentile rank over the last `look` bars (task tal1411ema (2): no entries when the
    width is under the 20th percentile). dict: upper, mid, lower, width, width_pct (0..1)."""
    o, h, l, c, v = cols(df)
    m = sma(c, n)
    sd = stdev(c, n)
    up, dn = m + k * sd, m - k * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (up - dn) / m
    pr = S(w).rolling(look, min_periods=20).apply(lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan,
                                                  raw=True).to_numpy()
    return {"upper": up, "mid": m, "lower": dn, "width": w, "width_pct": pr}


def ema_stack(df, lengths=(9, 21, 50, 100, 200)):
    """EMA ribbon order as a graded trend filter (The Trader's Guide 9/21/50/100/200, task tal1411set4 (2); also
    tal1411ema 9/20 and the 3-SMA stack of tal1411set2 (3) with other lengths). dict: score (-4..+4: +1 for every
    adjacent pair in bullish order), full_up (all in order), full_dn, above_slow (close over the slowest), lines
    (dict length -> EMA)."""
    o, h, l, c, v = cols(df)
    ls = {n: ema(c, n) for n in lengths}
    arr = [ls[n] for n in lengths]
    sc = np.zeros(len(c))
    for a, b in zip(arr[:-1], arr[1:]):
        sc += np.sign(a - b)
    m = len(lengths) - 1
    return {"score": sc, "full_up": sc == m, "full_dn": sc == -m, "above_slow": c > arr[-1], "lines": ls}


def rsi_regime(df, n=14, trend_len=50, bull=(40.0, 80.0), bear=(20.0, 60.0)):
    """RSI range shift (Trading Data / Market Structure '60-40', task tal1411rsi): in an up trend RSI lives in
    40-80, in a down trend 20-60. Trend = close vs EMA(trend_len) and EMA21 vs EMA50 (Market Structure's rule).
    dict: rsi, up (trend), dn, pull_long (RSI dipped to <= 45 in an up trend and turns up), pull_short (RSI rose to
    >= 55 in a down trend and turns down), mom_long (RSI crosses 60 up in an up trend), mom_short (crosses 40 down)."""
    o, h, l, c, v = cols(df)
    r = rsi(c, n)
    e21, e50, et = ema(c, 21), ema(c, 50), ema(c, trend_len)
    up = (c > et) & (e21 > e50)
    dn = (c < et) & (e21 < e50)
    pr, ppr = shift(r, 1), shift(r, 2)
    pull_l = up & (pr <= bull[0] + 5) & (r > pr) & (pr <= ppr)
    pull_s = dn & (pr >= bear[1] - 5) & (r < pr) & (pr >= ppr)
    return {"rsi": r, "up": up, "dn": dn, "pull_long": pull_l, "pull_short": pull_s,
            "mom_long": up & cross_up(r, 60.0), "mom_short": dn & cross_down(r, 40.0)}


def qqe(df, n=14, sf=5, q=4.238):
    """QQE (Quantitative Qualitative Estimation; StockSharp 'QQE signals'
    https://doc.stocksharp.com/en/api-examples/0440_Qqe_Signals , standard Pine 'QQE MOD' logic): RsiMa = EMA(RSI(n),
    sf); dar = EMA(EMA(|dRsiMa|, 2n-1), 2n-1) x q; trailing line follows RsiMa at dar. dict: rsima, trail, buy
    (RsiMa crosses over the trail), sell."""
    o, h, l, c, v = cols(df)
    rm = ema(rsi(c, n), sf)
    wl = 2 * n - 1
    dar = ema(ema(np.abs(np.diff(rm, prepend=rm[0])), wl), wl) * q
    tr = np.zeros(len(c))
    for i in range(1, len(c)):
        lb, sb = rm[i] - dar[i], rm[i] + dar[i]
        p = tr[i - 1]
        if rm[i] > p and rm[i - 1] > p:
            tr[i] = max(p, lb)
        elif rm[i] < p and rm[i - 1] < p:
            tr[i] = min(p, sb)
        elif rm[i] > p:
            tr[i] = lb
        else:
            tr[i] = sb
    return {"rsima": rm, "trail": tr, "buy": cross_up(rm, tr), "sell": cross_down(rm, tr)}


def laguerre_rsi(df, gamma=0.7):
    """Laguerre RSI (John Ehlers, 'Cybernetic Analysis'; StockSharp https://doc.stocksharp.com/en/api-examples/0024_RSI_Laguerre ).
    dict: lrsi (0..100), buy (crosses up 30), sell (crosses down 70)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    L0 = L1 = L2 = L3 = c[0] if n else 0.0
    out = np.zeros(n)
    for i in range(n):
        p0, p1, p2 = L0, L1, L2
        L0 = (1 - gamma) * c[i] + gamma * L0
        L1 = -gamma * L0 + p0 + gamma * L1
        L2 = -gamma * L1 + p1 + gamma * L2
        L3 = -gamma * L2 + p2 + gamma * L3
        cu = max(L0 - L1, 0) + max(L1 - L2, 0) + max(L2 - L3, 0)
        cd = max(L1 - L0, 0) + max(L2 - L1, 0) + max(L3 - L2, 0)
        out[i] = 100 * cu / (cu + cd) if cu + cd > 0 else (out[i - 1] if i else 50.0)
    return {"lrsi": out, "buy": cross_up(out, 30.0), "sell": cross_down(out, 70.0)}


def fisher_transform(df, n=9):
    """Ehlers Fisher Transform (standard Pine ta formula; FMZ strategy https://www.fmz.com/lang/en/strategy/503845 ).
    dict: fisher, trigger (previous bar's value), buy (crosses over the trigger under 1... as published: while
    fisher < 1), sell (crosses under while fisher > 1)."""
    o, h, l, c, v = cols(df)
    hl2 = (h + l) / 2
    hi, lo = highest(hl2, n), lowest(hl2, n)
    val = np.zeros(len(c))
    f = np.zeros(len(c))
    for i in range(len(c)):
        rng = hi[i] - lo[i]
        x = 0.66 * ((hl2[i] - lo[i]) / rng - 0.5) if rng > 0 else 0.0
        val[i] = np.clip(x + 0.67 * (val[i - 1] if i else 0.0), -0.999, 0.999)
        f[i] = 0.5 * np.log((1 + val[i]) / (1 - val[i])) + 0.5 * (f[i - 1] if i else 0.0)
    trig = shift(f, 1)
    return {"fisher": f, "trigger": trig, "buy": cross_up(f, trig) & (f < 1), "sell": cross_down(f, trig) & (f > 1)}


def ssl_channel(df, n=10):
    """SSL Channel (Erwin Beckers; standard Pine: SMA of highs / lows, the side flips when the close leaves the channel).
    dict: up, dn (the two lines), hlv (+1 / -1), buy (flips to +1), sell."""
    o, h, l, c, v = cols(df)
    sh, sl = sma(h, n), sma(l, n)
    hv = np.zeros(len(c))
    for i in range(1, len(c)):
        hv[i] = 1 if c[i] > sh[i] else -1 if c[i] < sl[i] else hv[i - 1]
    up = np.where(hv < 0, sl, sh)
    dn = np.where(hv < 0, sh, sl)
    p = np.concatenate([[0], hv[:-1]])
    return {"up": up, "dn": dn, "hlv": hv, "buy": (hv == 1) & (p == -1), "sell": (hv == -1) & (p == 1)}


def supertrend(df, n=10, mult=3.0):
    """Classic SuperTrend (ATR n, factor mult; Quant Tactics Supertrend 10/3 + EMA200 video, task tal1411set3 (2)).
    dict: line, trend (+1/-1), buy, sell."""
    o, h, l, c, v = cols(df)
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    up, dn = hl2 - mult * a, hl2 + mult * a
    fu, fd = up.copy(), dn.copy()
    t = np.ones(len(c))
    for i in range(1, len(c)):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if c[i - 1] < fd[i - 1] else dn[i]
        t[i] = 1 if c[i] > fd[i - 1] else -1 if c[i] < fu[i - 1] else t[i - 1]
    line = np.where(t > 0, fu, fd)
    p = np.concatenate([[t[0]], t[:-1]])
    return {"line": line, "trend": t, "buy": (t == 1) & (p == -1), "sell": (t == -1) & (p == 1)}


def macd_hist(df, fast=12, slow=26, sig=9):
    """MACD (Appel) - line, signal, histogram, crosses."""
    o, h, l, c, v = cols(df)
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, sig)
    return {"macd": m, "signal": s, "hist": m - s, "up": cross_up(m, s), "dn": cross_down(m, s)}


def raschke_osc(df):
    """Raschke 3/10 oscillator (SMA3 - SMA10, signal SMA16; MQL5 article https://www.mql5.com/en/articles/24280 ).
    dict: fast, slow."""
    o, h, l, c, v = cols(df)
    f = sma(c, 3) - sma(c, 10)
    return {"fast": f, "slow": sma(f, 16)}


def rma_atr(df, n=14):
    """ATR (Wilder RMA of the true range) - convenience."""
    o, h, l, c, v = cols(df)
    return atr(h, l, c, n)


_ = rma
