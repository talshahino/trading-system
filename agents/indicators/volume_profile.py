"""Session Volume Profile / TPO profile - POC, VAH, VAL, HVN, LVN, profile shape, delta split and the void (OUR OWN
BUILD from the standard Market Profile / Volume Profile definitions; references: TradingView 'Session Volume Profile'
help https://www.tradingview.com/support/solutions/43000703072-volume-profile-indicators-basic-concepts/ ,
Delta Void Profile [BigBeluga] https://www.tradingview.com/script/hD0LaluR-Delta-Void-Profile-BigBeluga/ (described:
buy/sell delta per bin, the void = the bins farthest below the peak), Dualis Flow / Olisa Flow / saint / nq tom /
Orderflow Sniper / Meks Trades videos of the user, tasks tal1411vp, tal1411of, tal1411ind).

CFDs have no real volume: the profile uses the feed's tick volume per bar, spread evenly over the bins the bar's
high-low covers ('tpo' mode counts time instead - every bar = 1). A bar's buy share = (close-low)/(high-low) (delta
approximation, no order book). Sessions = calendar days in `tz` shifted by `start_hour`. Bin size is fixed per
session = 0.25 x ATR(14) at the session's first bar. Value area = the highest-volume bins up to 70% of the volume
(ours: not the contiguous two-bin expansion of CBOT - close in practice).
Shapes of a COMPLETED session (known only from the next session's first bar - no repaint): P = POC in the top 40%
of the range and 60%+ of the volume above the middle; b = mirror; B = two peaks (>= 60% of the POC) with a valley
under half of the smaller peak between them; trend = the range is 1.5x the 10-session average and the session
moved over 60% of its range open-to-close (2 up / -2 down); else D (balanced).
HVN = local peaks >= 50% of the POC, LVN = local valleys <= 30% of the POC (3-bin smoothed profile)."""
import numpy as np

from agents.indicators._core import atr, cols, session_day

SHAPES = {1: "P", -1: "b", 0: "D", 3: "B", 2: "trend up", -2: "trend down"}


def _va(hist, frac=0.7):
    if hist.sum() <= 0:
        return 0, len(hist) - 1
    order = np.argsort(-hist)
    cum = np.cumsum(hist[order])
    m = int(np.searchsorted(cum, frac * cum[-1])) + 1
    sel = order[:m]
    return int(sel.min()), int(sel.max())


def _shape(hist, lo, bin_, o0, c1, rng, avg_rng):
    if hist.sum() <= 0 or rng <= 0:
        return 0, [], [], np.nan
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    pk = sm.max()
    peaks = [i for i in range(1, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] >= 0.5 * pk]
    vall = [i for i in range(1, len(sm) - 1) if sm[i] <= sm[i - 1] and sm[i] <= sm[i + 1] and sm[i] <= 0.3 * pk]
    hvn = [lo + (i + 0.5) * bin_ for i in peaks]
    lvn = [lo + (i + 0.5) * bin_ for i in vall]
    poc = int(np.argmax(hist))
    p = (poc + 0.5) / len(hist)
    mid = len(hist) // 2
    up_share = hist[mid:].sum() / hist.sum()
    inner = sm[1:-1] if len(sm) > 2 else sm
    void = lo + (int(np.argmin(inner)) + 1.5) * bin_ if len(sm) > 2 else np.nan
    big = [i for i in peaks if sm[i] >= 0.6 * pk]
    if len(big) >= 2:
        a, b = big[0], big[-1]
        if b - a >= 3 and sm[a:b + 1].min() <= 0.5 * min(sm[a], sm[b]):
            return 3, hvn, lvn, void
    if avg_rng and rng > 1.5 * avg_rng and abs(c1 - o0) > 0.6 * rng:
        return (2 if c1 > o0 else -2), hvn, lvn, void
    if p >= 0.6 and up_share >= 0.6:
        return 1, hvn, lvn, void
    if p <= 0.4 and up_share <= 0.4:
        return -1, hvn, lvn, void
    return 0, hvn, lvn, void


def volume_profile(df, tz="UTC", start_hour=0, mode="volume", va=0.7):
    """dict of arrays:
      dev_poc, dev_vah, dev_val  developing profile of the running session (bars <= i)
      poc, vah, val              the PREVIOUS completed session
      shape                      previous session's shape code (1 P, -1 b, 0 D, 3 B, 2 / -2 trend)
      hvn_above, hvn_below, lvn_above, lvn_below   nearest HVN / LVN of the previous session around the close
      in_lvn                     close within half a bin of a previous-session LVN
      void                       previous session's deepest gap (least-volume bin)
      delta                      running session's buy-minus-sell share at the developing POC bin (-1..1)"""
    o, h, l, c, v = cols(df)
    n = len(c)
    w = np.ones(n) if mode == "tpo" else np.where(v > 0, v, 1.0)
    a = atr(h, l, c, 14)
    day = session_day(df, tz, start_hour)
    keys = ["dev_poc", "dev_vah", "dev_val", "poc", "vah", "val", "shape", "hvn_above", "hvn_below", "lvn_above",
            "lvn_below", "void", "delta"]
    out = {k: np.full(n, np.nan) for k in keys}
    out["in_lvn"] = np.zeros(n, bool)
    hist = None
    buyh = None
    lo0 = bin_ = None
    prev = None  # (poc, vah, val, shape, hvn, lvn, void, bin)
    sess_o = sess_h = sess_l = None
    rngs = []
    cur = None
    for i in range(n):
        if day[i] != cur:
            if hist is not None:
                tot_lo = lo0
                rng = sess_h - sess_l
                avg = float(np.mean(rngs[-10:])) if rngs else 0.0
                lo_i, hi_i = _va(hist, va)
                pk = int(np.argmax(hist))
                shp, hv, lv, vd = _shape(hist, tot_lo, bin_, sess_o, c[i - 1], rng, avg)
                prev = (tot_lo + (pk + 0.5) * bin_, tot_lo + (hi_i + 1) * bin_, tot_lo + lo_i * bin_, shp,
                        np.array(sorted(hv)), np.array(sorted(lv)), vd, bin_)
                rngs.append(rng)
            cur = day[i]
            bin_ = max(float(a[i]) * 0.25 if np.isfinite(a[i]) and a[i] > 0 else (h[i] - l[i]) or 1e-6, 1e-9)
            lo0 = l[i] - bin_ * 50
            hist = np.zeros(100)
            buyh = np.zeros(100)
            sess_o, sess_h, sess_l = o[i], h[i], l[i]
        sess_h, sess_l = max(sess_h, h[i]), min(sess_l, l[i])
        b0 = int(np.floor((l[i] - lo0) / bin_))
        b1 = int(np.floor((h[i] - lo0) / bin_))
        if b0 < 0:
            pad = -b0 + 20
            hist = np.concatenate([np.zeros(pad), hist])
            buyh = np.concatenate([np.zeros(pad), buyh])
            lo0 -= pad * bin_
            b0 += pad
            b1 += pad
        if b1 >= len(hist):
            pad = b1 - len(hist) + 21
            hist = np.concatenate([hist, np.zeros(pad)])
            buyh = np.concatenate([buyh, np.zeros(pad)])
        share = (c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.5
        cnt = b1 - b0 + 1
        hist[b0:b1 + 1] += w[i] / cnt
        buyh[b0:b1 + 1] += w[i] * share / cnt
        nz = np.nonzero(hist)[0]
        sub = hist[nz[0]:nz[-1] + 1]
        pk = int(np.argmax(sub)) + nz[0]
        lo_i, hi_i = _va(sub, va)
        out["dev_poc"][i] = lo0 + (pk + 0.5) * bin_
        out["dev_val"][i] = lo0 + (lo_i + nz[0]) * bin_
        out["dev_vah"][i] = lo0 + (hi_i + nz[0] + 1) * bin_
        out["delta"][i] = 2 * buyh[pk] / hist[pk] - 1 if hist[pk] > 0 else 0.0
        if prev is not None:
            poc, vah, val, shp, hv, lv, vd, pb = prev
            out["poc"][i], out["vah"][i], out["val"][i], out["shape"][i], out["void"][i] = poc, vah, val, shp, vd
            x = c[i]
            if len(hv):
                k = np.searchsorted(hv, x)
                out["hvn_above"][i] = hv[k] if k < len(hv) else np.nan
                out["hvn_below"][i] = hv[k - 1] if k > 0 else np.nan
            if len(lv):
                k = np.searchsorted(lv, x)
                out["lvn_above"][i] = lv[k] if k < len(lv) else np.nan
                out["lvn_below"][i] = lv[k - 1] if k > 0 else np.nan
                out["in_lvn"][i] = bool(np.min(np.abs(lv - x)) <= pb / 2)
    return out


def tpo_profile(df, tz="UTC", start_hour=0):
    """Market Profile by time (TPO): the same outputs as volume_profile, every bar weighs 1."""
    return volume_profile(df, tz, start_hour, mode="tpo")


def delta_void_profile(df, tz="UTC", start_hour=0):
    """Delta Void Profile (our build of BigBeluga's idea): previous session's void (deepest gap) and the developing
    POC's buy/sell delta; plus the HVN/LVN levels - a thin wrapper over volume_profile."""
    vp = volume_profile(df, tz, start_hour)
    return {"void": vp["void"], "delta": vp["delta"], "hvn_above": vp["hvn_above"], "hvn_below": vp["hvn_below"],
            "lvn_above": vp["lvn_above"], "lvn_below": vp["lvn_below"]}
