"""Squeeze Momentum Indicator [LazyBear] - open-source Pine:
https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/ (task tal1411set7 (12); the library
had only TTM Squeeze Pro). Exact port: BB = SMA(close, 20) +/- 1.5 x stdev,
KC = SMA(close, 20) +/- 1.5 x SMA(true range, 20) (useTrueRange = true). LazyBear's quirk is kept: the published code
computes the BB deviation with multKC (1.5), not with the BB input mult (2.0) - bb_mult=None reproduces it. sqzOn = BB inside KC, sqzOff = BB outside
KC, noSqz = neither. val = linreg(close - avg(avg(highest(high,20), lowest(low,20)), SMA(close,20)), 20, 0).
Histogram colours: lime (val > 0 and rising), green (> 0 falling), red (< 0 falling), maroon (< 0 rising).
No repaint (closed bars)."""
import numpy as np

from agents.indicators._core import S, cols, highest, lowest, sma, stdev, tr


def _linreg(y, n):
    x = np.arange(n, dtype=float)
    xm = x.mean()
    sx = ((x - xm) ** 2).sum()

    def f(w):
        if np.isnan(w).any():
            return np.nan
        b = ((x - xm) * (w - w.mean())).sum() / sx
        return w.mean() + b * (n - 1 - xm)
    return S(y).rolling(n).apply(f, raw=True).to_numpy()


def squeeze_momentum(df, bb_len=20, bb_mult=None, kc_len=20, kc_mult=1.5):
    """dict: val (momentum histogram), sqz_on, sqz_off, no_sqz, fired (first sqz_off bar after an on bar),
    lime / green / red / maroon (histogram colours), bars_on (length of the squeeze that just ended / is running)."""
    o, h, l, c, v = cols(df)
    basis, dev = sma(c, bb_len), (kc_mult if bb_mult is None else bb_mult) * stdev(c, bb_len)
    ma, rr = sma(c, kc_len), sma(tr(h, l, c), kc_len)
    ubb, lbb, ukc, lkc = basis + dev, basis - dev, ma + kc_mult * rr, ma - kc_mult * rr
    on = (lbb > lkc) & (ubb < ukc)
    off = (lbb < lkc) & (ubb > ukc)
    val = _linreg(c - ((highest(h, kc_len) + lowest(l, kc_len)) / 2 + sma(c, kc_len)) / 2, kc_len)
    pv = np.r_[np.nan, val[:-1]]
    pon = np.r_[False, on[:-1]]
    run = np.zeros(len(c))
    for i in range(len(c)):
        run[i] = (run[i - 1] + 1) if (i and on[i]) else (1.0 if on[i] else (run[i - 1] if i and pon[i] else 0.0))
    with np.errstate(invalid="ignore"):
        return {"val": val, "sqz_on": on, "sqz_off": off, "no_sqz": ~on & ~off, "fired": off & pon,
                "lime": (val > 0) & (val > pv), "green": (val > 0) & (val <= pv),
                "red": (val < 0) & (val < pv), "maroon": (val < 0) & (val >= pv), "bars_on": run}
