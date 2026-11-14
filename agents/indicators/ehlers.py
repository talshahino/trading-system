"""John Ehlers' adaptive trend lines.

mama_fama - MESA Adaptive Moving Average and its Following AMA (Ehlers, 'MAMA - The Mother of Adaptive Moving
Averages', https://mesasoftware.com/papers/papers/MAMA.pdf; LazyBear's Pine port, strategy https://fmz.com/strategy/428090).
Ported from the paper's EasyLanguage: Hilbert-transform homodyne period, phase rate of change -> alpha clamped to
[slow limit 0.05, fast limit 0.5], MAMA = alpha x price + (1 - alpha) x MAMA[1], FAMA = 0.5 alpha x MAMA + (1 - 0.5 alpha) x FAMA[1].

ehlers_itrend - Instantaneous Trendline with its Trigger (Ehlers, 'Cybernetic Analysis for Stocks and Futures', ch. 1;
blackcat1402 strategy https://www.tradingview.com/script/edDSj1kq-blackcat-L2-Ehlers-Instantaneous-Trendline-Trading-Strategy/):
ITrend = (a - a^2/4) p + 0.5 a^2 p[1] - (a - 0.75 a^2) p[2] + 2 (1 - a) ITrend[1] - (1 - a)^2 ITrend[2] (first 7
bars: (p + 2 p[1] + p[2]) / 4), Trigger = 2 ITrend - ITrend[2]; p = hl2, a = 0.07.
Both are recursive filters on closed bars only - no repaint."""
import math

import numpy as np

from agents.indicators._core import cols, cross_down, cross_up


def mama_fama(df, fast=0.5, slow=0.05):
    """dict: mama, fama, period (dominant cycle), up (MAMA crosses above FAMA), dn."""
    o, h, l, c, v = cols(df)
    p = (h + l) / 2
    n = len(p)
    sm, det, q1, i1, ji, jq, i2, q2, re, im, per, sper, ph, mama, fama = (np.zeros(n) for _ in range(15))
    for t in range(n):
        if t < 6:
            mama[t] = fama[t] = p[t]
            per[t] = 6.0
            continue
        sm[t] = (4 * p[t] + 3 * p[t - 1] + 2 * p[t - 2] + p[t - 3]) / 10
        k = 0.075 * per[t - 1] + 0.54
        det[t] = (0.0962 * sm[t] + 0.5769 * sm[t - 2] - 0.5769 * sm[t - 4] - 0.0962 * sm[t - 6]) * k
        q1[t] = (0.0962 * det[t] + 0.5769 * det[t - 2] - 0.5769 * det[t - 4] - 0.0962 * det[t - 6]) * k
        i1[t] = det[t - 3]
        ji[t] = (0.0962 * i1[t] + 0.5769 * i1[t - 2] - 0.5769 * i1[t - 4] - 0.0962 * i1[t - 6]) * k
        jq[t] = (0.0962 * q1[t] + 0.5769 * q1[t - 2] - 0.5769 * q1[t - 4] - 0.0962 * q1[t - 6]) * k
        a2, b2 = i1[t] - jq[t], q1[t] + ji[t]
        i2[t] = 0.2 * a2 + 0.8 * i2[t - 1]
        q2[t] = 0.2 * b2 + 0.8 * q2[t - 1]
        r = i2[t] * i2[t - 1] + q2[t] * q2[t - 1]
        m = i2[t] * q2[t - 1] - q2[t] * i2[t - 1]
        re[t] = 0.2 * r + 0.8 * re[t - 1]
        im[t] = 0.2 * m + 0.8 * im[t - 1]
        pr = per[t - 1]
        if im[t] != 0 and re[t] != 0:
            ang = math.degrees(math.atan(im[t] / re[t]))
            if ang != 0:
                pr = 360.0 / ang
        pr = min(pr, 1.5 * per[t - 1])
        pr = max(pr, 0.67 * per[t - 1])
        pr = min(max(pr, 6.0), 50.0)
        per[t] = 0.2 * pr + 0.8 * per[t - 1]
        sper[t] = 0.33 * per[t] + 0.67 * sper[t - 1]
        ph[t] = math.degrees(math.atan(q1[t] / i1[t])) if i1[t] != 0 else ph[t - 1]
        dp = ph[t - 1] - ph[t]
        if dp < 1:
            dp = 1.0
        al = min(max(fast / dp, slow), fast)
        mama[t] = al * p[t] + (1 - al) * mama[t - 1]
        fama[t] = 0.5 * al * mama[t] + (1 - 0.5 * al) * fama[t - 1]
    mama[:30] = np.nan  # warm-up
    fama[:30] = np.nan
    return {"mama": mama, "fama": fama, "period": sper, "up": cross_up(mama, fama), "dn": cross_down(mama, fama)}


def ehlers_itrend(df, alpha=0.07):
    """dict: itrend, trigger, up (trigger crosses above itrend), dn."""
    o, h, l, c, v = cols(df)
    p = (h + l) / 2
    n = len(p)
    it = np.zeros(n)
    a = alpha
    for t in range(n):
        if t < 7:
            it[t] = (p[t] + 2 * p[t - 1] + p[t - 2]) / 4 if t >= 2 else p[t]
            continue
        it[t] = ((a - a * a / 4) * p[t] + 0.5 * a * a * p[t - 1] - (a - 0.75 * a * a) * p[t - 2]
                 + 2 * (1 - a) * it[t - 1] - (1 - a) ** 2 * it[t - 2])
    trig = np.full(n, np.nan)
    trig[2:] = 2 * it[2:] - it[:-2]
    it[:20] = np.nan
    trig[:20] = np.nan
    return {"itrend": it, "trigger": trig, "up": cross_up(trig, it), "dn": cross_down(trig, it)}
