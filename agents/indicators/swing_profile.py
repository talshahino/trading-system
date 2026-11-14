"""Swing Profile Analyzer [ChartPrime] - task tal1411set7 (8): a profile for every swing leg, with the leg's POC.
Source: https://www.tradingview.com/script/8S40eYNi-Swing-Profile-Analyzer-ChartPrime/ (listed open-source, Feb 2025;
the code itself was not readable from here, so this is OUR RECONSTRUCTION of the published description): zigzag swing
legs; the profile of a leg is a FREQUENCY profile of CLOSES (how many closes fell in each price bin - ChartPrime's
'highest number of closes within a swing'), POC = the bin with the most closes; profiles appear only once the
swing is complete. source="volume" (our addition, the user's wording 'volume profile per swing') spreads each bar's
tick volume over the bins its high-low range covers instead.
Ours: pivots with k bars each side (known k bars late - no repaint); a leg = the bars from one confirmed pivot to
the next opposite pivot; `bins` rows between the leg's low and high; value area = the bins around the POC holding
70% (standard, our addition). The developing profile of the running leg (from the last confirmed pivot to
bar i) is recomputed on every bar from closed bars only (capped at `max_len` bars)."""
import numpy as np

from agents.indicators._core import cols, pivots


def _profile(h, l, v, bins, c=None):
    lo, hi = float(np.min(l)), float(np.max(h))
    if not np.isfinite(lo) or hi <= lo:
        m = (hi + lo) / 2.0
        return m, hi, lo
    edges = np.linspace(lo, hi, bins + 1)
    bl, bh = edges[:-1], edges[1:]
    if c is not None:                                     # frequency of closes (ChartPrime)
        prof = np.bincount(np.clip(np.searchsorted(edges, c, side="right") - 1, 0, bins - 1), minlength=bins).astype(float)
        return _va(prof, bl, bh, bins)
    ov = np.clip(np.minimum(h[:, None], bh[None, :]) - np.maximum(l[:, None], bl[None, :]), 0, None)
    rng = np.maximum(h - l, 1e-12)
    w = np.where((h - l)[:, None] > 0, ov / rng[:, None], 0.0)
    # zero-range bars: all volume in the bin holding the price
    z = (h - l) <= 0
    if z.any():
        idx = np.clip(np.searchsorted(edges, h[z], side="right") - 1, 0, bins - 1)
        w[np.where(z)[0], idx] = 1.0
    prof = (w * v[:, None]).sum(axis=0)
    return _va(prof, bl, bh, bins)


def _va(prof, bl, bh, bins):
    j = int(np.argmax(prof))
    tot = prof.sum()
    a, b, acc = j, j, prof[j]
    while tot > 0 and acc < 0.7 * tot and (a > 0 or b < bins - 1):
        up = prof[b + 1] if b < bins - 1 else -1
        dn = prof[a - 1] if a > 0 else -1
        if up >= dn:
            b += 1
            acc += prof[b]
        else:
            a -= 1
            acc += prof[a]
    return (bl[j] + bh[j]) / 2.0, bh[b], bl[a]


def swing_profile(df, k=5, bins=24, max_len=300, source="close"):
    """dict: leg_poc / leg_vah / leg_val / leg_dir (+1 up leg, -1 down leg) / leg_hi / leg_lo of the LAST COMPLETED
    leg (known when its end pivot is confirmed), dev_poc / dev_vah / dev_val (developing profile of the running leg
    from the last confirmed pivot to bar i), new_leg (True on the bar a leg completes)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    v = np.where(np.isfinite(v) & (v > 0), v, 1.0)
    cc = c if source == "close" else None

    def prof(s, e):
        return _profile(h[s:e], l[s:e], v[s:e], bins, None if cc is None else cc[s:e])
    phc, plc, phv, plv, phi, pli = pivots(h, l, k)
    keys = ["leg_poc", "leg_vah", "leg_val", "leg_dir", "leg_hi", "leg_lo", "dev_poc", "dev_vah", "dev_val"]
    out = {x: np.full(n, np.nan) for x in keys}
    out["new_leg"] = np.zeros(n, bool)
    last_piv = None                      # (index, kind) kind +1 high, -1 low
    cur = [np.nan] * 6
    for i in range(n):
        for conf, idx, kind in ((plc[i], pli[i], -1), (phc[i], phi[i], 1)):
            if not conf:
                continue
            if last_piv is not None and last_piv[1] != kind and idx > last_piv[0]:
                s, e = last_piv[0], idx
                s = max(s, e - max_len)
                poc, vah, val = prof(s, e + 1)
                cur = [poc, vah, val, float(kind), float(np.max(h[s:e + 1])), float(np.min(l[s:e + 1]))]
                out["new_leg"][i] = True
            if last_piv is None or last_piv[1] != kind or (kind == 1 and h[idx] >= h[last_piv[0]]) or \
                    (kind == -1 and l[idx] <= l[last_piv[0]]):
                last_piv = (int(idx), kind)
        for x, val_ in zip(keys[:6], cur):
            out[x][i] = val_
        if last_piv is not None:
            s = max(last_piv[0], i - max_len)
            if i - s >= 2:
                out["dev_poc"][i], out["dev_vah"][i], out["dev_val"][i] = prof(s, i + 1)
    return out
