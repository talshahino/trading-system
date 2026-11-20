"""Indicators brought by the scout on 20.11.2026 (day run).

iakf - Innovation Adaptive Kalman Filter Oscillator (jdehorty, TradingView open-source,
       https://www.tradingview.com/script/06GCx5qL-Innovation-Adaptive-Kalman-Filter-Oscillator/ ). The page text was read
       but the Pine code was not visible to us, so this is OUR RECONSTRUCTION from the description: a level + slope Kalman
       filter on the close whose prediction errors ("innovations") drive (1) a Surprise meter = short-run innovation
       variance vs its normal level, mapped 100 x (1 - 1 / ratio) so 0 = normal, 50 = twice, 75 = four times (the page's
       anchor points), (2) unusual moves = a trend-centred innovation larger than 2.05 x its typical size (major = 2 x
       that), (3) damping = each unusual move adds 0.05 x the squared innovation to the measurement noise of that bar,
       so the filter trusts it less; the Damping meter = 100 x (1 - R_base / R_bar), EMA smoothed. Volatility shift: on
       when both meters reach 30, off when both fall under 10 (the page's levels). Trend smoothness 20, noise lookback 100
       (defaults not stated on the page - ours). Causal recursion: the value at bar i uses only bars <= i.

raschke_sheet - Raschke Trade Sheet scanner (JoeRodTrades, TradingView open-source,
       https://www.tradingview.com/script/8MVLfKu4-Raschke-Trade-Sheet-scanner/ ; rules from the description, code not
       read): bias from the 3/10 oscillator (SMA3 - SMA10), its 16-bar slow line (SMA16 of it - ours) and the 2-period
       ROC (all three rising = +1, all falling = -1); flags 3bar (high under the two prior highs and low over the two
       prior lows), WR7 (widest range of 7), Coil (the last three bars overlap: lowest high > highest low), ExtSig (after
       7+ closes on one side of the SMA5 the first close across it; ours: B after closes UNDER the SMA, S after closes
       OVER - the page does not say which side maps to B / S), 2ROC new 30-bar high / low, 20-bar new high / low. Every
       flag uses closed bars only.
"""
import numpy as np

from agents.indicators._core import S, cols, sma


def iakf(df, smooth=20, lookback=100, thr=2.05, damp=0.05, fast=10):
    """dict est, slope, dir (+1 rising / -1 falling), innov, surprise 0..100, damping 0..100, unusual_up / unusual_dn,
    major_up / major_dn, shift (volatility shift on), settling (shift on and a meter under 20)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    out = {k: np.full(n, np.nan) for k in ("est", "slope", "innov", "surprise", "damping")}
    flags = {k: np.zeros(n, bool) for k in ("unusual_up", "unusual_dn", "major_up", "major_dn", "shift", "settling")}
    if n == 0:
        out["dir"] = np.zeros(0)
        out.update(flags)
        return out
    a_l = 2.0 / (lookback + 1.0)
    a_f = 2.0 / (fast + 1.0)
    a_s = 2.0 / (smooth + 1.0)
    q = a_s
    x = np.array([c[0], 0.0])
    P = np.eye(2)
    d0 = np.diff(c[: min(n, lookback)])
    V = float(np.var(d0)) if len(d0) > 1 and np.var(d0) > 0 else max(abs(c[0]) * 1e-4, 1e-12)
    Vf, m, dsm = V, 0.0, 0.0
    shift = False
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    for i in range(n):
        if i == 0:
            out["est"][0], out["slope"][0], out["innov"][0] = c[0], 0.0, 0.0
            out["surprise"][0] = out["damping"][0] = 0.0
            continue
        xp = F @ x
        Q = np.array([[q * V, 0.0], [0.0, q * q * V]])
        Pp = F @ P @ F.T + Q
        e = c[i] - xp[0]
        ec = e - m
        typ = np.sqrt(V)
        unusual = abs(ec) > thr * typ
        R = V
        if unusual:
            R = V + damp * ec * ec
        Sv = Pp[0, 0] + R
        K = Pp[:, 0] / Sv
        x = xp + K * e
        P = (np.eye(2) - np.outer(K, [1.0, 0.0])) @ Pp
        ratio = Vf / V if V > 0 else 1.0
        sur = float(np.clip(100.0 * (1.0 - 1.0 / ratio), 0.0, 100.0)) if ratio > 0 else 0.0
        dval = 100.0 * (1.0 - V / R) if R > 0 else 0.0
        dsm = dsm + a_f * (dval - dsm)
        out["est"][i], out["slope"][i], out["innov"][i] = x[0], x[1], e
        out["surprise"][i], out["damping"][i] = sur, dsm
        if unusual:
            up = ec > 0
            flags["unusual_up" if up else "unusual_dn"][i] = True
            if abs(ec) > 2 * thr * typ:
                flags["major_up" if up else "major_dn"][i] = True
        if not shift and sur >= 30 and dsm >= 30:
            shift = True
        elif shift and sur < 10 and dsm < 10:
            shift = False
        flags["shift"][i] = shift
        flags["settling"][i] = shift and (sur < 20 or dsm < 20)
        # learn the normal level after this bar (the next bar's test uses bars <= i only)
        V = V + a_l * (e * e - V)
        Vf = Vf + a_f * (e * e - Vf)
        m = m + a_s * (e - m)
    est = out["est"]
    d = np.zeros(n)
    d[1:] = np.sign(np.diff(est))
    out["dir"] = S(np.where(d == 0, np.nan, d)).ffill().fillna(0.0).to_numpy()
    out.update(flags)
    return out


def raschke_sheet(df, ext_n=7, roc_n=30, hl_n=20):
    """dict osc (SMA3 - SMA10), slow (SMA16 of osc), roc2, bias (+1 / -1 / 0), three_bar, wr7, coil, ext_buy, ext_sell,
    roc2_hi, roc2_lo, new_hi20, new_lo20."""
    o, h, l, c, v = cols(df)
    n = len(c)
    osc = sma(c, 3) - sma(c, 10)
    slow = sma(osc, 16)
    pc2 = np.concatenate([np.full(2, np.nan), c[:-2]]) if n > 2 else np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        roc2 = 100.0 * (c / pc2 - 1.0)

    def _up(x):
        x = np.asarray(x, float)
        r = np.zeros(n, bool)
        r[1:] = x[1:] > x[:-1]
        return r

    def _dn(x):
        x = np.asarray(x, float)
        r = np.zeros(n, bool)
        r[1:] = x[1:] < x[:-1]
        return r

    bias = np.where(_up(osc) & _up(slow) & _up(roc2), 1.0, np.where(_dn(osc) & _dn(slow) & _dn(roc2), -1.0, 0.0))
    hs, ls = S(h), S(l)
    h1, h2, l1, l2 = hs.shift(1).to_numpy(), hs.shift(2).to_numpy(), ls.shift(1).to_numpy(), ls.shift(2).to_numpy()
    with np.errstate(invalid="ignore"):
        three_bar = (h < np.fmin(h1, h2)) & (l > np.fmax(l1, l2))
        rg = h - l
        wr7 = rg >= S(rg).rolling(7).max().to_numpy()
        coil = (S(h).rolling(3).min().to_numpy() > S(l).rolling(3).max().to_numpy())
    m5 = sma(c, 5)
    above = np.nan_to_num(c > m5).astype(bool) & np.isfinite(m5)
    below = np.nan_to_num(c < m5).astype(bool) & np.isfinite(m5)

    def _run(mask):
        r = np.zeros(n)
        for i in range(n):
            r[i] = r[i - 1] + 1 if (mask[i] and i > 0) else (1.0 if mask[i] else 0.0)
        return r

    ra, rb = _run(above), _run(below)
    ra1 = np.concatenate([[0.0], ra[:-1]])
    rb1 = np.concatenate([[0.0], rb[:-1]])
    ext_buy = (rb1 >= ext_n) & above
    ext_sell = (ra1 >= ext_n) & below
    rs = S(roc2)
    with np.errstate(invalid="ignore"):
        roc2_hi = np.isfinite(roc2) & (roc2 >= rs.rolling(roc_n).max().to_numpy())
        roc2_lo = np.isfinite(roc2) & (roc2 <= rs.rolling(roc_n).min().to_numpy())
        new_hi = h > hs.shift(1).rolling(hl_n).max().to_numpy()
        new_lo = l < ls.shift(1).rolling(hl_n).min().to_numpy()
    return {"osc": osc, "slow": slow, "roc2": roc2, "bias": bias, "three_bar": three_bar, "wr7": wr7, "coil": coil,
            "ext_buy": ext_buy, "ext_sell": ext_sell, "roc2_hi": roc2_hi, "roc2_lo": roc2_lo,
            "new_hi20": new_hi, "new_lo20": new_lo}
