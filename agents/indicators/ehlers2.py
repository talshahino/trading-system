"""John Ehlers' cycle filters and oscillators (scout, 15.11.2026). Ported from the published EasyLanguage; every
filter is recursive on closed bars only - no repaint.

super_smoother    2-pole Super Smoother (Ehlers, 'Cycle Analytics for Traders' 2013, ch. 3).
roofing           Roofing filter: 2-pole high-pass 48 -> Super Smoother 10, and Ehlers' stochastic of it (20 bars, smoothed)
                  - the thinkorswim 'EhlersStoch' strategy buys over 0.2 and sells under 0.8.
                  https://toslc.thinkorswim.com/center/reference/Tech-Indicators/strategies/E-K/EhlersStoch
decycler          Decycler = close - 2-pole high-pass(60) and the Decycler Oscillators 100 x HP(decycle, P/2) / close for
                  P = 125 and 100 (TASC Sep 2015, 'Decyclers'). https://www.traders.com/Documentation/FEEDbk_docs/2015/09/TradersTips.html
cyber_cycle       Cyber Cycle (alpha 0.07) with its 1-bar trigger ('Cybernetic Analysis', ch. 4).
center_of_gravity CG oscillator 10 on hl2 with its 1-bar trigger (Ehlers 2002, 'The CG Oscillator').
ebsw              Even Better Sinewave: 1-pole HP 40 -> Super Smoother 10, 3-bar wave / sqrt(power) (Cycle Analytics ch. 12).
sinewave          Classic Hilbert sinewave: Sine and LeadSine (45 deg ahead) of the dominant-cycle phase ('Rocket Science
                  for Traders' ch. 9); the dominant period comes from the same homodyne discriminator as MAMA.
                  WH SelfInvest system 09: https://www.whselfinvest.de/en/trading-platform/free-trading-strategies/tradingsystem/09-sinewave-ehlers
cybernetic_osc    Cybernetic Oscillator (TASC June 2025): HP 30 -> Super Smoother 20, divided by its 100-bar RMS.
"""
import math

import numpy as np

from agents.indicators._core import cols, cross_down, cross_up


def super_smoother(x, period=10):
    x = np.asarray(x, dtype=float)
    a1 = math.exp(-1.414 * math.pi / period)
    c2 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    f = np.zeros(len(x))
    xx = np.nan_to_num(x)
    for t in range(len(x)):
        if t < 2:
            f[t] = xx[t]
            continue
        f[t] = c1 * (xx[t] + xx[t - 1]) / 2 + c2 * f[t - 1] + c3 * f[t - 2]
    return f


def highpass2(x, period=48):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    w = 0.707 * 2 * math.pi / period
    al = (math.cos(w) + math.sin(w) - 1) / math.cos(w)
    hp = np.zeros(len(x))
    for t in range(2, len(x)):
        hp[t] = ((1 - al / 2) ** 2 * (x[t] - 2 * x[t - 1] + x[t - 2]) + 2 * (1 - al) * hp[t - 1]
                 - (1 - al) ** 2 * hp[t - 2])
    return hp


def _warm(a, n):
    a = np.asarray(a, dtype=float).copy()
    a[:n] = np.nan
    return a


def roofing(df, hp=48, lp=10, stoch_n=20):
    """dict: filt, stoch (0..1, smoothed), up (stoch crosses over 0.2), dn (under 0.8)."""
    c = cols(df)[3]
    f = super_smoother(highpass2(c, hp), lp)
    import pandas as pd
    hh = pd.Series(f).rolling(stoch_n).max().to_numpy()
    ll = pd.Series(f).rolling(stoch_n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        st = np.where(hh - ll > 0, (f - ll) / (hh - ll), 0.5)
    st = super_smoother(np.nan_to_num(st, nan=0.5), lp)
    f, st = _warm(f, 2 * hp), _warm(st, 2 * hp)
    return {"filt": f, "stoch": st, "up": cross_up(st, 0.2), "dn": cross_down(st, 0.8)}


def decycler(df, period=60, fast=100, slow=125):
    """dict: decycler (trend line), osc_fast (P=100), osc_slow (P=125), up (fast osc crosses over slow), dn."""
    c = cols(df)[3]
    dec = c - highpass2(c, period)

    def osc(p):
        d = c - highpass2(c, p)
        return 100 * highpass2(d, 0.5 * p) / c

    of, os_ = _warm(osc(fast), 2 * slow), _warm(osc(slow), 2 * slow)
    return {"decycler": _warm(dec, 2 * period), "osc_fast": of, "osc_slow": os_, "up": cross_up(of, os_),
            "dn": cross_down(of, os_)}


def cyber_cycle(df, alpha=0.07):
    """dict: cycle, trigger, up, dn."""
    o, h, l, c, v = cols(df)
    p = (h + l) / 2
    n = len(p)
    sm, cy = np.zeros(n), np.zeros(n)
    for t in range(n):
        if t < 3:
            sm[t] = p[t]
            continue
        sm[t] = (p[t] + 2 * p[t - 1] + 2 * p[t - 2] + p[t - 3]) / 6
        if t < 7:
            cy[t] = (p[t] - 2 * p[t - 1] + p[t - 2]) / 4
        else:
            cy[t] = ((1 - 0.5 * alpha) ** 2 * (sm[t] - 2 * sm[t - 1] + sm[t - 2]) + 2 * (1 - alpha) * cy[t - 1]
                     - (1 - alpha) ** 2 * cy[t - 2])
    cy = _warm(cy, 20)
    tg = np.concatenate([[np.nan], cy[:-1]])
    return {"cycle": cy, "trigger": tg, "up": cross_up(cy, tg), "dn": cross_down(cy, tg)}


def center_of_gravity(df, n=10):
    """dict: cg, trigger, up, dn."""
    o, h, l, c, v = cols(df)
    p = (h + l) / 2
    k = len(p)
    cg = np.full(k, np.nan)
    w = np.arange(1, n + 1, dtype=float)
    for t in range(n - 1, k):
        win = p[t - n + 1: t + 1][::-1]          # win[0] = current bar
        den = win.sum()
        if den != 0:
            cg[t] = -np.dot(w, win) / den + (n + 1) / 2.0
    tg = np.concatenate([[np.nan], cg[:-1]])
    return {"cg": cg, "trigger": tg, "up": cross_up(cg, tg), "dn": cross_down(cg, tg)}


def ebsw(df, duration=40, ss=10):
    """dict: wave (-1..1), up (crosses over -0.85), dn (crosses under 0.85)."""
    c = np.nan_to_num(cols(df)[3])
    n = len(c)
    ang = 2 * math.pi / duration
    a1 = (1 - math.sin(ang)) / math.cos(ang)
    hp = np.zeros(n)
    for t in range(1, n):
        hp[t] = 0.5 * (1 + a1) * (c[t] - c[t - 1]) + a1 * hp[t - 1]
    f = super_smoother(hp, ss)
    f1 = np.concatenate([[0.0], f[:-1]])
    f2 = np.concatenate([[0.0, 0.0], f[:-2]])[:n]
    wave = (f + f1 + f2) / 3
    pwr = (f * f + f1 * f1 + f2 * f2) / 3
    with np.errstate(divide="ignore", invalid="ignore"):
        wv = np.where(pwr > 0, wave / np.sqrt(pwr), 0.0)
    wv = _warm(wv, 2 * duration)
    return {"wave": wv, "up": cross_up(wv, -0.85), "dn": cross_down(wv, 0.85)}


def sinewave(df):
    """dict: sine, lead, period, up (sine crosses over the lead sine), dn."""
    from agents.indicators.ehlers import mama_fama
    o, h, l, c, v = cols(df)
    p = (h + l) / 2
    n = len(p)
    per = np.nan_to_num(mama_fama(df)["period"], nan=15.0)
    sm = np.zeros(n)
    for t in range(3, n):
        sm[t] = (4 * p[t] + 3 * p[t - 1] + 2 * p[t - 2] + p[t - 3]) / 10
    ph = np.zeros(n)
    for t in range(n):
        dc = int(per[t] + 0.5)
        if dc < 6 or t < dc + 6:
            ph[t] = ph[t - 1] if t else 0.0
            continue
        k = np.arange(dc)
        win = sm[t - k]
        re = float(np.dot(np.sin(2 * math.pi * k / dc), win))
        im = float(np.dot(np.cos(2 * math.pi * k / dc), win))
        if abs(im) > 0.001:
            d = math.degrees(math.atan(re / im))
        else:
            d = 90.0 * (1 if re >= 0 else -1)
        d += 90
        if im < 0:
            d += 180
        if d > 315:
            d -= 360
        ph[t] = d
    s = np.sin(np.radians(ph))
    ld = np.sin(np.radians(ph + 45))
    s, ld = _warm(s, 60), _warm(ld, 60)
    return {"sine": s, "lead": ld, "period": per, "up": cross_up(s, ld), "dn": cross_down(s, ld)}


def cybernetic_osc(df, hp=30, lp=20, rms_n=100):
    """dict: osc (RMS-normalised), up (crosses over 0), dn."""
    import pandas as pd
    c = cols(df)[3]
    f = super_smoother(highpass2(c, hp), lp)
    rms = np.sqrt(pd.Series(f * f).rolling(rms_n).mean().to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        o_ = np.where(rms > 0, f / rms, 0.0)
    o_ = _warm(o_, rms_n)
    return {"osc": o_, "up": cross_up(o_, 0.0), "dn": cross_down(o_, 0.0)}
