"""Shared building blocks for agents/indicators (self-contained: no import of agents.rules_agent, so the package can
be imported from any strategy module without a circular import). Every function uses only bars <= i at bar i."""
import numpy as np
import pandas as pd


def cols(df):
    """o, h, l, c, v as float arrays (a feed without real opens -> previous close as the open)."""
    o, h, l, c = (df[x].to_numpy(dtype=float) for x in "ohlc")
    v = df["v"].to_numpy(dtype=float) if "v" in df else np.ones(len(c))
    if len(c) and np.allclose(o, c):
        o = np.concatenate([[c[0]], c[:-1]])
    return o, h, l, c, v


def S(x):
    return pd.Series(np.asarray(x, dtype=float))


def ema(x, n):
    return S(x).ewm(span=n, adjust=False).mean().to_numpy()


def sma(x, n):
    return S(x).rolling(int(n)).mean().to_numpy()


def rma(x, n):
    return S(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def wma(x, n):
    n = int(n)
    w = np.arange(1, n + 1, dtype=float)
    return S(x).rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True).to_numpy()


def hma(x, n):
    return wma(2 * wma(x, max(n // 2, 1)) - wma(x, n), max(int(np.sqrt(n)), 1))


def stdev(x, n):
    return S(x).rolling(int(n)).std(ddof=0).to_numpy()


def highest(x, n):
    return S(x).rolling(int(n), min_periods=1).max().to_numpy()


def lowest(x, n):
    return S(x).rolling(int(n), min_periods=1).min().to_numpy()


def shift(x, k=1):
    x = np.asarray(x, dtype=float)
    if k <= 0:
        return x
    return np.concatenate([np.full(k, np.nan), x[:-k]])


def tr(h, l, c):
    pc = np.concatenate([[c[0]], c[:-1]])
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr(h, l, c, n=14):
    return rma(tr(h, l, c), n)


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up, dn = rma(np.clip(d, 0, None), n), rma(np.clip(-d, 0, None), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    return np.where(dn == 0, 100.0, r)


def cci(h, l, c, n=20):
    tp = (h + l + c) / 3
    m = sma(tp, n)
    md = S(tp).rolling(n).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((tp - m) / (0.015 * md))


def dmi(h, l, c, n=14):
    """(+DI, -DI, ADX) Wilder."""
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(h, l, c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.nan_to_num(100 * rma(pdm, n) / a)
        ndi = np.nan_to_num(100 * rma(ndm, n) / a)
        dx = np.nan_to_num(100 * np.abs(pdi - ndi) / (pdi + ndi))
    return pdi, ndi, rma(dx, n)


def cross_up(a, b):
    a = np.asarray(a, float)
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_down(a, b):
    a = np.asarray(a, float)
    return cross_up(-a, -np.broadcast_to(np.asarray(b, float), a.shape))


def pivots(h, l, k):
    """Swing pivots with k bars on each side. A pivot at bar j is KNOWN only at bar j + k (no repaint): returns
    ph_conf / pl_conf (True on the confirmation bar), ph_val / pl_val (the pivot price on that bar, else NaN) and
    the pivot's own index on the confirmation bar (ph_idx / pl_idx, -1 elsewhere)."""
    n = len(h)
    hs, ls = S(h), S(l)
    w = 2 * k + 1
    ph = (hs == hs.rolling(w, center=True).max()).to_numpy()
    pl = (ls == ls.rolling(w, center=True).min()).to_numpy()
    ph_conf = np.zeros(n, bool)
    pl_conf = np.zeros(n, bool)
    ph_val = np.full(n, np.nan)
    pl_val = np.full(n, np.nan)
    ph_idx = np.full(n, -1)
    pl_idx = np.full(n, -1)
    j = np.where(ph)[0]
    j = j[j + k < n]
    ph_conf[j + k] = True
    ph_val[j + k] = h[j]
    ph_idx[j + k] = j
    j = np.where(pl)[0]
    j = j[j + k < n]
    pl_conf[j + k] = True
    pl_val[j + k] = l[j]
    pl_idx[j + k] = j
    return ph_conf, pl_conf, ph_val, pl_val, ph_idx, pl_idx


def last_pivots(h, l, k):
    """Last confirmed swing high / low on every bar (ffill of pivots())."""
    _, _, phv, plv, _, _ = pivots(h, l, k)
    return S(phv).ffill().to_numpy(), S(plv).ffill().to_numpy()


def session_day(df, tz="UTC", start_hour=0):
    """Session key per bar: the local date in `tz` of (time - start_hour), so a session starting 17:00 New York
    is start_hour=17 with tz America/New_York."""
    t = df.index.tz_convert(tz) - pd.Timedelta(hours=start_hour)
    return pd.Series(t.date).to_numpy()


def bar_minutes(df):
    if len(df.index) < 2:
        return 5.0
    return float(pd.Series(df.index).diff().median().total_seconds() / 60.0)


def onset(mask):
    m = np.asarray(mask, dtype=bool)
    return m & ~np.concatenate([[False], m[:-1]])


def ffill(x):
    return S(x).ffill().to_numpy()
