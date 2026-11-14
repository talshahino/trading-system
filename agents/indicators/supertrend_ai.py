"""SuperTrend AI (Clustering) [LuxAlgo] - open-source Pine (CC BY-NC-SA 4.0):
https://www.tradingview.com/script/wP7WWjLL-SuperTrend-AI-Clustering-LuxAlgo/ (library page:
https://www.luxalgo.com/library/indicator/supertrend-ai-clustering/). Also covers Flux Charts 'AI SuperTrend' buy/sell
(the user's task tal1411ind, 10:08), which is the same idea.

Port (defaults of the original): ATR length 10, factors 1.0 .. 5.0 step 0.5, performance memory 10, cluster 'Best',
k-means with 3 clusters initialised at the performance quartiles, max 1000 iterations. Every factor runs its own
SuperTrend; its performance is an EMA (alpha 2/(memory+1)) of sign(close[1] - st[1]) * (close - close[1]); the k-means
on the factors' performances picks the best cluster; the final factor = the mean factor of that cluster; the final
SuperTrend uses that factor on the bar. Performance index = max(cluster mean perf, 0) / EMA(|close - close[1]|, memory).
Our own choices: the clustering is done on every bar from the values known at that bar (the original does the same
on the realtime bar); the 'Historical Bars Calculation' window is not used (all bars). No repaint: bar i uses bars <= i.
Signals: os flips to +1 = buy, to -1 = sell (the original's labels)."""
import numpy as np

from agents.indicators._core import atr, cols, ema


def _kmeans3(perf, factors, iters=50):
    q = np.percentile(perf, [25, 50, 75])
    cent = q.astype(float)
    lab = np.zeros(len(perf), int)
    for _ in range(iters):
        lab = np.argmin(np.abs(perf[:, None] - cent[None, :]), axis=1)
        new = np.array([perf[lab == j].mean() if np.any(lab == j) else cent[j] for j in range(3)])
        if np.allclose(new, cent):
            break
        cent = new
    return lab, cent


def supertrend_ai(df, length=10, fmin=1.0, fmax=5.0, step=0.5, memory=10, cluster="best"):
    """Returns dict: line (trailing stop), trend (+1 / -1), buy, sell (flip bars), factor, perf_idx."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, length)
    hl2 = (h + l) / 2
    facs = np.arange(fmin, fmax + 1e-9, step)
    m = len(facs)
    upper = hl2[0] * np.ones(m)
    lower = hl2[0] * np.ones(m)
    out = np.zeros(m)
    trend = np.zeros(m)
    perf = np.zeros(m)
    alpha = 2.0 / (memory + 1)
    den = ema(np.abs(np.diff(c, prepend=c[0])), memory)
    line = np.full(n, np.nan)
    tr_final = np.zeros(n)
    fac_final = np.full(n, np.nan)
    pidx = np.zeros(n)
    f_up, f_lo, f_os = hl2[0], hl2[0], 0
    pick = {"best": 2, "average": 1, "worst": 0}[cluster]
    for i in range(n):
        if not np.isfinite(a[i]):
            continue
        up = hl2[i] + a[i] * facs
        dn = hl2[i] - a[i] * facs
        if i > 0:
            prev_out = out.copy()
            trend = np.where(c[i] > upper, 1, np.where(c[i] < lower, 0, trend))
            upper = np.where(c[i - 1] < upper, np.minimum(up, upper), up)
            lower = np.where(c[i - 1] > lower, np.maximum(dn, lower), dn)
            out = np.where(trend == 1, lower, upper)
            diff = np.sign(c[i - 1] - prev_out)
            perf = perf + alpha * ((c[i] - c[i - 1]) * diff - perf)
        else:
            upper, lower, out = up.copy(), dn.copy(), dn.copy()
        if i < length:
            continue
        lab, cent = _kmeans3(perf, facs)
        order = np.argsort(cent)
        j = order[pick]
        sel = lab == j
        if not np.any(sel):
            sel = np.ones(m, bool)
        target = float(facs[sel].mean())
        fac_final[i] = target
        pidx[i] = max(float(perf[sel].mean()), 0.0) / den[i] if den[i] > 0 else 0.0
        up1 = hl2[i] + a[i] * target
        dn1 = hl2[i] - a[i] * target
        f_up = min(up1, f_up) if i > 0 and c[i - 1] < f_up else up1
        f_lo = max(dn1, f_lo) if i > 0 and c[i - 1] > f_lo else dn1
        f_os = 1 if c[i] > f_up else 0 if c[i] < f_lo else f_os
        line[i] = f_lo if f_os == 1 else f_up
        tr_final[i] = 1 if f_os == 1 else -1
    prev = np.concatenate([[0], tr_final[:-1]])
    buy = (tr_final == 1) & (prev == -1)
    sell = (tr_final == -1) & (prev == 1)
    return {"line": line, "trend": tr_final, "buy": buy, "sell": sell, "factor": fac_final, "perf_idx": pidx}
