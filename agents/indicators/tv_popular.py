"""Popular TradingView / MT4 indicators the library did not have (scout, 15.11.2026). Ports of the open-source
scripts (links below); where a step of the original is a display trick or not reproducible, our version is said.
All causal: the value at bar i uses bars <= i only.

waddah_attar   Waddah Attar Explosion (LazyBear port of the MT4 indicator): t1 = (MACD(20, 40)[0] - MACD[1]) x 150,
               explosion = BB(20, 2) width, dead zone = RMA(TR, 100) x 3.7. https://www.tradingview.com/script/iu3kKWDI-Waddah-Attar-Explosion-LazyBear/
tdi            Traders Dynamic Index (Dean Malone): RSI 13, price line SMA2, signal SMA7, market base SMA34, volatility
               bands base +/- 1.6185 x StdDev34. https://www.luxalgo.com/library/concept/traders-dynamic-index/
macd_v         MACD-V (Alex Spiroglou, 2022): 100 x (EMA12 - EMA26) / ATR26, signal EMA9; +/-50 neutral, +/-150 extended.
               https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-v
vfi            Volume Flow Indicator (Markos Katsanos; LazyBear port): length 130, coef 0.2, vcoef 2.5, smoothing 3,
               signal EMA5, tick volume. https://www.tradingview.com/script/MhlDpfdS-Volume-Flow-Indicator-LazyBear/
coral          Coral Trend (LazyBear): 6-stage EMA cascade with cd 0.4 (a T3 relative), smoothing 21; colour = slope.
               https://www.luxalgo.com/library/concept/coral-trend/
t3             Tillson T3 (6 EMAs, volume factor 0.7) - helper.
pmax           Profit Maximizer (KivancOzbilgic): SuperTrend logic applied to a moving average (EMA10 of hl2, ATR10 x 3);
               the 'PMax on RSI w/ T3' variant runs it on T3(RSI) with the RSI's own range as ATR (our reading of the
               script). https://www.tradingview.com/script/7yn5dteO-Profit-Maximizer-Strategy , .../ZLcmWQtj-PMax-on-Rsi-w-T3-Strategy/
gmma           Guppy Multiple Moving Averages: short EMAs 3/5/8/10/12/15, long EMAs 30/35/40/45/50/60, group edges and
               the long group's width. https://www.babypips.com/learn/forex/guppy-multiple-moving-average
rsx            Jurik RSX (everget's open-source port of the published RSX algorithm), 0..100.
               https://www.tradingview.com/script/20TzvKPk-Jurik-RSX/
crsi           Cyclic Smoothed RSI (Lars von Thienen, whentotrade; TASC Oct 2017): dominant cycle 20, vibration 10,
               bands at the 10 % / 90 % levels of the last 40 values. Ours: the original finds the band levels by a
               step search; a rolling 10 / 90 percentile is the same level up to the step size.
               https://www.prorealcode.com/prorealtime-indicators/rsi-cyclic-smoothed/
ift_rsi        Inverse Fisher Transform of RSI (Sylvain Vervoort): 0.1 x (RSI5 - 50) -> WMA9 -> IFT, -1..1.
               https://www.tradingview.com/script/Nw4PDAwY-Inverse-Fisher-Transform-on-RSI-for-backtest-w-alerts/
smoothed_ha    Heiken Ashi Smoothed (MT4 'Heiken_Ashi_Smoothed'): SMMA6 of the OHLC -> Heikin Ashi -> LWMA2.
"""
import numpy as np
import pandas as pd

from agents.indicators._core import (S, atr, cols, cross_down, cross_up, ema, rma, rsi, sma, stdev, tr, wma)


def _sh(x, k=1):
    x = np.asarray(x, dtype=float)
    return np.concatenate([np.full(k, np.nan), x[:-k]]) if k > 0 else x


def waddah_attar(df, fast=20, slow=40, channel=20, mult=2.0, sens=150, dz=3.7):
    """dict: up_power, dn_power (>= 0 histograms), explosion, deadzone, long_go, short_go (bar above both lines)."""
    o, h, l, c, v = cols(df)
    m = ema(c, fast) - ema(c, slow)
    t1 = (m - _sh(m)) * sens
    bm, sd = sma(c, channel), stdev(c, channel)
    e1 = (bm + mult * sd) - (bm - mult * sd)
    dead = rma(tr(h, l, c), 100) * dz
    up = np.where(t1 >= 0, t1, 0.0)
    dn = np.where(t1 < 0, -t1, 0.0)
    return {"up_power": up, "dn_power": dn, "explosion": e1, "deadzone": dead,
            "long_go": (up > e1) & (up > dead), "short_go": (dn > e1) & (dn > dead)}


def tdi(df, rsi_n=13, price_n=2, sig_n=7, band_n=34, k=1.6185):
    """dict: price (green), signal (red), base (yellow), upper, lower, up (green over red), dn."""
    c = cols(df)[3]
    r = rsi(c, rsi_n)
    pr, sg = sma(r, price_n), sma(r, sig_n)
    base, sd = sma(r, band_n), stdev(r, band_n)
    return {"price": pr, "signal": sg, "base": base, "upper": base + k * sd, "lower": base - k * sd,
            "up": cross_up(pr, sg), "dn": cross_down(pr, sg)}


def macd_v(df, fast=12, slow=26, sig=9, atr_n=26):
    """dict: macdv, signal, hist."""
    o, h, l, c, v = cols(df)
    a = atr(h, l, c, atr_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        mv = np.where(a > 0, 100 * (ema(c, fast) - ema(c, slow)) / a, 0.0)
    mv[:slow] = np.nan
    sg = np.array(ema(np.nan_to_num(mv), sig))
    sg[:slow] = np.nan
    return {"macdv": mv, "signal": sg, "hist": mv - sg}


def vfi(df, length=130, coef=0.2, vcoef=2.5, smooth=3, sig=5):
    """dict: vfi, signal."""
    o, h, l, c, v = cols(df)
    typ = (h + l + c) / 3
    inter = np.log(typ) - np.log(_sh(typ))
    vinter = S(inter).rolling(30).std(ddof=0).to_numpy()
    cutoff = coef * vinter * c
    vave = _sh(sma(v, length))
    vc = np.minimum(v, vave * vcoef)
    mf = typ - _sh(typ)
    vcp = np.where(mf > cutoff, vc, np.where(mf < -cutoff, -vc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = S(vcp).rolling(length).sum().to_numpy() / vave
    out = sma(raw, smooth)
    return {"vfi": out, "signal": ema(np.nan_to_num(out), sig)}


def coral(df, sm=21, cd=0.4):
    """dict: coral, up (rising), flip_up, flip_dn."""
    c = cols(df)[3]
    di = (sm - 1.0) / 2.0 + 1.0
    c1 = 2 / (di + 1.0)
    c2 = 1 - c1
    c3 = 3.0 * (cd * cd + cd * cd * cd)
    c4 = -3.0 * (2.0 * cd * cd + cd + cd * cd * cd)
    c5 = 3.0 * cd + 1.0 + cd * cd * cd + 3.0 * cd * cd
    n = len(c)
    i = np.zeros((6, n))
    bf = np.zeros(n)
    for t in range(n):
        prev = i[:, t - 1] if t else np.full(6, c[0])
        x = c[t]
        for j in range(6):
            i[j, t] = c1 * x + c2 * prev[j]
            x = i[j, t]
        bf[t] = -cd ** 3 * i[5, t] + c3 * i[4, t] + c4 * i[3, t] + c5 * i[2, t]
    bf[: 3 * sm] = np.nan
    up = bf > _sh(bf)
    dn = bf < _sh(bf)
    return {"coral": bf, "up": up, "flip_up": up & (_sh(bf) < _sh(bf, 2)), "flip_dn": dn & (_sh(bf) > _sh(bf, 2))}


def t3(x, n=8, vf=0.7):
    def gd(y):
        e = ema(y, n)
        return e * (1 + vf) - ema(e, n) * vf
    return gd(gd(gd(np.asarray(x, dtype=float))))


def _pmax_core(ma, a, mult):
    k = len(ma)
    ls, ss, d = np.full(k, np.nan), np.full(k, np.nan), np.ones(k)
    for t in range(k):
        if np.isnan(ma[t]) or np.isnan(a[t]):
            continue
        l0, s0 = ma[t] - mult * a[t], ma[t] + mult * a[t]
        lp = ls[t - 1] if t and not np.isnan(ls[t - 1]) else l0
        sp = ss[t - 1] if t and not np.isnan(ss[t - 1]) else s0
        ls[t] = max(l0, lp) if ma[t] > lp else l0
        ss[t] = min(s0, sp) if ma[t] < sp else s0
        dp = d[t - 1] if t else 1
        d[t] = 1 if (dp == -1 and ma[t] > sp) else (-1 if (dp == 1 and ma[t] < lp) else dp)
    pm = np.where(d == 1, ls, ss)
    return {"ma": ma, "pmax": pm, "dir": d, "up": cross_up(ma, pm), "dn": cross_down(ma, pm)}


def pmax(df, ma_len=10, atr_n=10, mult=3.0, source="hl2"):
    """dict: ma, pmax, dir, up (MA crosses over PMax), dn. source 'hl2' (default) or 'rsi_t3' (PMax on RSI w/ T3:
    T3(8) of RSI(14), the RSI line's own true range as ATR)."""
    o, h, l, c, v = cols(df)
    if source == "rsi_t3":
        r = rsi(c, 14)
        ma = t3(np.nan_to_num(r, nan=50.0), ma_len if ma_len != 10 else 8)
        ma[:30] = np.nan
        a = rma(np.abs(np.diff(np.nan_to_num(r, nan=50.0), prepend=50.0)), atr_n)
    else:
        ma = ema((h + l) / 2, ma_len)
        a = atr(h, l, c, atr_n)
    return _pmax_core(ma, a, mult)


def gmma(df, short=(3, 5, 8, 10, 12, 15), long=(30, 35, 40, 45, 50, 60)):
    """dict: s_min, s_max, l_min, l_max, l_width (in % of price), bull (whole short group over the long group), bear."""
    c = cols(df)[3]
    sg = np.vstack([ema(c, n) for n in short])
    lg = np.vstack([ema(c, n) for n in long])
    smin, smax, lmin, lmax = sg.min(0), sg.max(0), lg.min(0), lg.max(0)
    w = 100 * (lmax - lmin) / c
    w[: max(long)] = np.nan
    return {"s_min": smin, "s_max": smax, "l_min": lmin, "l_max": lmax, "l_width": w,
            "bull": smin > lmax, "bear": smax < lmin}


def rsx(df, n=14):
    c = cols(df)[3]
    k = len(c)
    f18 = 3.0 / (n + 2.0)
    f20 = 1.0 - f18
    st = np.zeros(12)   # f28 f30 f38 f40 f48 f50 / f58 f60 f68 f70 f78 f80
    out = np.full(k, np.nan)
    f10 = c[0] * 100 if k else 0.0
    for t in range(k):
        f8 = 100 * c[t]
        v8 = f8 - f10
        f10 = f8
        res = []
        for base, val in ((0, v8), (6, abs(v8))):
            x = val
            for j in range(3):
                a, b = base + 2 * j, base + 2 * j + 1
                st[a] = f20 * st[a] + f18 * x
                st[b] = f18 * st[a] + f20 * st[b]
                x = st[a] * 1.5 - st[b] * 0.5
            res.append(x)
        if t >= n and res[1] != 0:
            out[t] = min(max((res[0] / res[1] + 1) * 50, 0.0), 100.0)
        elif t >= n:
            out[t] = 50.0
    return out


def crsi(df, domcycle=20, vibration=10, leveling=10.0):
    """dict: crsi, upper, lower, up (crosses over the lower band), dn (under the upper band)."""
    c = cols(df)[3]
    cl = max(domcycle // 2, 2)
    mem = domcycle * 2
    torque = 2.0 / (vibration + 1)
    lag = int((vibration - 1) / 2.0)
    d = np.diff(c, prepend=c[0] if len(c) else 0.0)
    up, dn = rma(np.maximum(d, 0), cl), rma(np.maximum(-d, 0), cl)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    k = len(c)
    cr = np.zeros(k)
    for t in range(k):
        rl = r[t - lag] if t >= lag else r[t]
        cr[t] = torque * (2 * r[t] - rl) + (1 - torque) * (cr[t - 1] if t else 50.0)
    cr[: mem] = np.nan
    lo = S(cr).rolling(mem).quantile(leveling / 100.0).to_numpy()
    hi = S(cr).rolling(mem).quantile(1 - leveling / 100.0).to_numpy()
    return {"crsi": cr, "upper": hi, "lower": lo, "up": cross_up(cr, lo), "dn": cross_down(cr, hi)}


def ift_rsi(df, n=5, w=9):
    c = cols(df)[3]
    x = wma(0.1 * (rsi(c, n) - 50), w)
    e = np.exp(2 * np.clip(x, -20, 20))
    return (e - 1) / (e + 1)


def smoothed_ha(df, n1=6, n2=2):
    """dict: o, h, l, c (smoothed HA), bull (ha close > ha open), flip_up, flip_dn."""
    o, h, l, c, v = cols(df)
    so, sh_, sl, sc = (rma(x, n1) for x in (o, h, l, c))
    k = len(c)
    hc = (so + sh_ + sl + sc) / 4
    ho = np.zeros(k)
    for t in range(k):
        ho[t] = (so[t] + sc[t]) / 2 if t == 0 else (ho[t - 1] + hc[t - 1]) / 2
    hh = np.maximum(sh_, np.maximum(ho, hc))
    hl = np.minimum(sl, np.minimum(ho, hc))
    lo_, lh, ll, lc = wma(ho, n2), wma(hh, n2), wma(hl, n2), wma(hc, n2)
    bull = lc > lo_
    pb = np.concatenate([[False], bull[:-1]])
    valid = ~np.isnan(lo_) & ~np.isnan(np.concatenate([[np.nan], lo_[:-1]]))
    return {"o": lo_, "h": lh, "l": ll, "c": lc, "bull": bull, "flip_up": bull & ~pb & valid,
            "flip_dn": ~bull & pb & valid}
