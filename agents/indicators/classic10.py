"""Scout 18.11.2026 (evening): named price-action methods as indicators - Camarilla, Fisher's ACD, TheStrat, Weis Wave,
Better Volume, DeMark's short-term patterns, Kagi, Quasimodo, Al Brooks' bar counting, Ochoa's (Pivot Boss) reversals.

Sources and what is our own reconstruction:
- camarilla: formulas https://www.mql5.com/en/blogs/post/720680 (H4 = C + range x 1.1 / 2 ... L4 = C - range x 1.1 / 2).
- acd: https://www.luxalgo.com/library/concept/acd-method.md and https://nexusfi.com/a/strategies/acd-trading-method .
  OURS: A and C are set per market in the book, here 0.10 / 0.20 of the 14-day average daily range (TradingFinder's script
  also scales them by volatility); the opening range is 30 minutes from 09:30 New York; acceptance = half of it.
- the_strat: https://www.luxalgo.com/library/concept/the-strat.md and
  https://trendspider.com/learning-center/thestrat-candlestick-patterns-a-traders-guide/ .
- weis_wave: OURS, rebuilt from the descriptions of https://my.tradingview.com/script/HFGx4ote-Indicator-Weis-Wave-Volume-LazyBear
  (trend-detection length) and https://id.tradingview.com/script/32ymSRAQ-Weis-Wave-Oscillator/ (high effort = wave volume
  over 1.5 x the average completed wave); the Pine source was not readable.
- better_volume: 1:1 from the public-domain code at https://prorealcode.com/prorealtime-indicators/better-volume .
- td_patterns: conditions as implemented by https://4e3995d1.wickra-docs.pages.dev/Indicators/Indicator-TdClop (also
  -TdClopwin, -TdCamouflage); that library's reading, not checked against DeMark's books.
- kagi: https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-types/kagi-charts . OURS: reversal = 2 x ATR14.
- quasimodo: https://www.mql5.com/en/articles/23141 (5-bar pivots, 30-bar wait for the retest).
- brooks_count: https://nexusfi.com/a/strategies/al-brooks-price-action-method .
- pivot_boss: book notes https://paisaconsultant.blogspot.com/2020/07/book-notes-secret-of-pivot-boss-by.html . OURS: the
  20-candle average, 'about 2 x' read as 1.6 x, and the close in the top / bottom 35% of the wick candle (the notes' 'Close%' is undefined).

No repaint: every value at bar i uses only bars <= i. Candles of n bars count only on the bar that completes them, pivots
only from the bar that confirms them.
"""
import numpy as np
import pandas as pd

from agents.indicators._core import S, cols


def _atr(h, l, c, n=14):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return S(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def agg_candles(df, n=1):
    """Candles of n bars on fixed clock buckets. Returns (o, h, l, c, v, pos): pos[j] = the df bar that completes candle j.
    A bucket counts only if its last bar is there (the bar's own end time equals the bucket's end), so nothing looks ahead."""
    o, h, l, c, v = cols(df)
    if n <= 1 or len(c) < 2:
        return o, h, l, c, v, np.arange(len(c))
    ns = df.index.asi8
    bar = int(np.median(np.diff(ns)))
    size = bar * int(n)
    b = ns // size
    g = pd.DataFrame({"b": b, "o": o, "h": h, "l": l, "c": c, "v": v, "i": np.arange(len(c)), "t": ns})
    a = g.groupby("b", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
                                      i=("i", "last"), t=("t", "last"), m=("o", "size"))
    a = a[((a.index.to_numpy() + 1) * size == a["t"].to_numpy() + bar) & (a["m"].to_numpy() == int(n))]
    return (a["o"].to_numpy(), a["h"].to_numpy(), a["l"].to_numpy(), a["c"].to_numpy(), a["v"].to_numpy(),
            a["i"].to_numpy(dtype=int))


def _daily(df):
    """Previous UTC day's high, low, close and the 14-day average range, on every bar."""
    o, h, l, c, v = cols(df)
    d = pd.Series(df.index.tz_convert("UTC").date)
    g = pd.DataFrame({"d": d, "h": h, "l": l, "c": c})
    a = g.groupby("d").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    a["adr"] = (a["h"] - a["l"]).rolling(14, min_periods=3).mean()
    a = a.shift(1)
    m = g[["d"]].join(a, on="d")
    return tuple(m[x].to_numpy(dtype=float) for x in ("h", "l", "c", "adr"))


def camarilla(df):
    """Camarilla levels from the previous UTC day: dict h1..h4, l1..l4."""
    ph, pl, pc, _ = _daily(df)
    r = (ph - pl) * 1.1
    out = {}
    for name, div in (("1", 12.0), ("2", 6.0), ("3", 4.0), ("4", 2.0)):
        out["h" + name] = pc + r / div
        out["l" + name] = pc - r / div
    return out


def acd(df, or_minutes=30, a_mult=0.10, c_mult=0.20, days=30):
    """Mark Fisher's ACD on the 09:30 New York open. Dict: orh, orl, a_up, a_dn, c_up, c_dn, pr_hi, pr_lo (pivot range),
    a_long / a_short (A held for half the opening range), c_long / c_short (a failed A, then through the other side by C),
    number (sum of the last `days` finished days' scores: A +1 / -1, C after a failed A +2 / -2)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    ph, pl, pc, adr = _daily(df)
    p = (ph + pl + pc) / 3.0
    m = (ph + pl) / 2.0
    w = np.abs(p - m)
    ny = df.index.tz_convert("America/New_York")
    hr = ny.hour.to_numpy() + ny.minute.to_numpy() / 60.0
    dates = pd.Series(ny.date).to_numpy()
    bar_min = float(pd.Series(df.index).diff().median().total_seconds() / 60.0) if n > 1 else 5.0
    hold = max(int(round(or_minutes / 2.0 / bar_min)), 1)
    start, end = 9.5, 9.5 + or_minutes / 60.0
    orh, orl, aup, adn, cup, cdn = (np.full(n, np.nan) for _ in range(6))
    al, ash, cl, csh = (np.zeros(n, bool) for _ in range(4))
    number = np.full(n, np.nan)
    scores = []
    day, hi, lo, state, score = None, np.nan, np.nan, 0, 0
    for t in range(n):
        if dates[t] != day:
            if day is not None:
                scores.append(score)
            day, hi, lo, state, score = dates[t], np.nan, np.nan, 0, 0
        number[t] = float(np.sum(scores[-days:])) if scores else 0.0
        if start <= hr[t] < end:
            hi = np.nanmax([hi, h[t]])
            lo = np.nanmin([lo, l[t]])
            continue
        if hr[t] < end or not np.isfinite(hi) or not np.isfinite(adr[t]):
            continue
        a, cv = a_mult * adr[t], c_mult * adr[t]
        orh[t], orl[t], aup[t], adn[t], cup[t], cdn[t] = hi, lo, hi + a, lo - a, hi + cv, lo - cv
        if hr[t] >= 16.0:
            continue
        if state == 0:
            s = 1 if c[t] > aup[t] else -1 if c[t] < adn[t] else 0
            if s != 0 and t >= hold - 1:
                back = c[t - hold + 1:t + 1]
                if np.all(back > aup[t]) if s > 0 else np.all(back < adn[t]):
                    state, score = s, s
                    (al if s > 0 else ash)[t] = True
        elif state == 1 and c[t] < cdn[t]:
            state, score = 2, -2
            csh[t] = True
        elif state == -1 and c[t] > cup[t]:
            state, score = 2, 2
            cl[t] = True
    return {"orh": orh, "orl": orl, "a_up": aup, "a_dn": adn, "c_up": cup, "c_dn": cdn, "pr_hi": p + w, "pr_lo": p - w,
            "a_long": al, "a_short": ash, "c_long": cl, "c_short": csh, "number": number}


def the_strat(df, n=1):
    """TheStrat on candles of n bars. Dict (on the bar completing the candle): type (1 inside, 2 up, -2 down, 3 outside,
    0 elsewhere), and the combos as +1 / -1: r22 (2-2 reversal), s212, s312, s122 (RevStrat); stop = the other side of the
    trigger candle's base, tgt = the extreme of the candle before the inside bar."""
    o, h, l, c, v, pos = agg_candles(df, n)
    N, m = len(df), len(c)
    typ = np.zeros(N, int)
    r22, s212, s312, s122 = (np.zeros(N, int) for _ in range(4))
    stop, tgt = np.full(N, np.nan), np.full(N, np.nan)
    ty = np.zeros(m, int)
    for j in range(1, m):
        up, dn = h[j] > h[j - 1], l[j] < l[j - 1]
        ty[j] = 3 if up and dn else 2 if up else -2 if dn else 1
    for j in range(3, m):
        p = pos[j]
        typ[p] = ty[j]
        d = 1 if ty[j] == 2 else -1 if ty[j] == -2 else 0
        if d == 0:
            continue
        a, b = ty[j - 2], ty[j - 1]
        if b == 1 and abs(a) == 2:
            s212[p], stop[p], tgt[p] = d, l[j - 1] if d > 0 else h[j - 1], h[j - 2] if d > 0 else l[j - 2]
        elif b == 1 and a == 3:
            s312[p], stop[p], tgt[p] = d, l[j - 1] if d > 0 else h[j - 1], h[j - 2] if d > 0 else l[j - 2]
        elif a == 1 and b == -2 * d:
            s122[p], stop[p], tgt[p] = d, l[j - 1] if d > 0 else h[j - 1], h[j - 2] if d > 0 else l[j - 2]
        elif b == -2 * d:
            r22[p], stop[p], tgt[p] = d, l[j - 1] if d > 0 else h[j - 1], h[j - 2] if d > 0 else l[j - 2]
    return {"type": typ, "r22": r22, "s212": s212, "s312": s312, "s122": s122, "stop": stop, "tgt": tgt}


def weis_wave(df, n=2, factor=1.5, waves=20):
    """Weis Wave Volume. Dict: wave (+1 / -1), vol (volume of the running wave), avg (average of the last finished waves),
    high_effort (running wave over factor x avg)."""
    o, h, l, c, v = cols(df)
    N = len(c)
    wave, vol, avg = np.ones(N, int), np.zeros(N), np.full(N, np.nan)
    mov, trend = 0, 1
    done = []
    for t in range(N):
        if t == 0:
            vol[t] = v[t]
            continue
        m = int(np.sign(c[t] - c[t - 1]))
        if m != 0 and m != mov:
            trend = m
        mov = m
        run = t >= n and (np.all(np.diff(c[t - n:t + 1]) > 0) or np.all(np.diff(c[t - n:t + 1]) < 0))
        wave[t] = trend if (trend != wave[t - 1] and run) else wave[t - 1]
        if wave[t] == wave[t - 1]:
            vol[t] = vol[t - 1] + v[t]
        else:
            done.append(vol[t - 1])
            vol[t] = v[t]
        if done:
            avg[t] = float(np.mean(done[-waves:]))
    return {"wave": wave, "vol": vol, "avg": avg, "high_effort": np.isfinite(avg) & (vol > factor * avg)}


def better_volume(df, lookback=20):
    """Better Volume (one-bar set). Dict of bool arrays: low_vol, climax_up, climax_dn, churn, climax_churn."""
    o, h, l, c, v = cols(df)
    rng = np.where(h - l > 0, h - l, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        v1 = np.where(c > o, v * (rng / (2 * rng + o - c)), np.where(c < o, v * ((rng + c - o) / (2 * rng + c - o)), 0.5 * v))
    v2 = v - v1

    def top(x):
        return S(x).rolling(lookback).max().to_numpy()

    def bot(x):
        return S(x).rolling(lookback).min().to_numpy()

    def eq(x, y):
        return np.isfinite(x) & np.isfinite(y) & (np.abs(x - y) <= 1e-12 * (np.abs(y) + 1.0))

    a4, a5, a6, a7 = v1 * rng, (v1 - v2) * rng, v2 * rng, (v2 - v1) * rng
    a8, a9, a10, a11 = v1 / rng, (v1 - v2) / rng, v2 / rng, (v2 - v1) / rng
    a12 = v / rng
    up_bar, dn_bar = c > o, c < o
    cu = up_bar & (eq(a4, top(a4)) | eq(a5, top(a5)) | eq(a10, bot(a10)) | eq(a11, bot(a11)))
    cd = dn_bar & (eq(a6, top(a6)) | eq(a7, top(a7)) | eq(a8, bot(a8)) | eq(a9, bot(a9)))
    ch = eq(a12, top(a12))
    return {"low_vol": eq(v, bot(v)), "climax_up": cu, "climax_dn": cd, "churn": ch, "climax_churn": ch & (cu | cd)}


def td_patterns(df):
    """DeMark short-term patterns as +1 / -1 arrays: clop, clopwin, camouflage."""
    o, h, l, c, v = cols(df)
    po, pc, pl, ph = (np.concatenate([[x[0]], x[:-1]]) for x in (o, c, l, h))
    clop = np.where((o < po) & (o < pc) & (c > po) & (c > pc), 1, np.where((o > po) & (o > pc) & (c < po) & (c < pc), -1, 0))
    lo, hi = np.minimum(po, pc), np.maximum(po, pc)
    ins = (o >= lo) & (o <= hi) & (c >= lo) & (c <= hi)
    win = np.where(ins & (c >= o), 1, np.where(ins & (c < o), -1, 0))
    cam = np.where((c < pc) & (c > o) & (l < pl), 1, np.where((c > pc) & (c < o) & (h > ph), -1, 0))
    for x in (clop, win, cam):
        x[0] = 0
    return {"clop": clop, "clopwin": win, "camouflage": cam}


def kagi(df, atr_mult=2.0, atr_n=14):
    """Kagi on closes with an ATR reversal. Dict: line (+1 rising, -1 falling), yang (True thick / False thin), buy (thin
    turns thick: the close breaks the last shoulder), sell (thick turns thin: the close breaks the last waist)."""
    o, h, l, c, v = cols(df)
    N = len(c)
    a = _atr(h, l, c, atr_n)
    line, yang = np.ones(N, int), np.zeros(N, bool)
    buy, sell = np.zeros(N, bool), np.zeros(N, bool)
    d, end, shoulder, waist, thick = 1, c[0] if N else 0.0, np.nan, np.nan, False
    for t in range(1, N):
        rev = atr_mult * a[t]
        if d > 0:
            if c[t] > end:
                end = c[t]
            elif c[t] <= end - rev:
                shoulder, d, end = end, -1, c[t]
        else:
            if c[t] < end:
                end = c[t]
            elif c[t] >= end + rev:
                waist, d, end = end, 1, c[t]
        if not thick and np.isfinite(shoulder) and c[t] > shoulder:
            thick, buy[t] = True, True
        elif thick and np.isfinite(waist) and c[t] < waist:
            thick, sell[t] = False, True
        line[t], yang[t] = d, thick
    return {"line": line, "yang": yang, "buy": buy, "sell": sell}


def _pivots(h, l, k):
    """Alternating confirmed swings (index, price, +1 high / -1 low), each with the bar it is known on."""
    N = len(h)
    out = []
    for j in range(k, N - k):
        wh, wl = h[j - k:j + k + 1], l[j - k:j + k + 1]
        for typ, val, ok in ((1, h[j], h[j] == wh.max()), (-1, l[j], l[j] == wl.min())):
            if not ok:
                continue
            if out and out[-1][2] == typ:
                if (typ == 1 and val > out[-1][1]) or (typ == -1 and val < out[-1][1]):
                    out[-1] = (j, val, typ)
            else:
                out.append((j, val, typ))
    return out


def quasimodo(df, k=5, wait=30):
    """Quasimodo: shoulder, leg, a head beyond the shoulder, a close through the leg (break of structure), then the return
    to the shoulder within `wait` bars. Dict: sell / buy (touch of the QM line), sell_rej / buy_rej (pierced and closed back
    through it), line, head, bos."""
    o, h, l, c, v = cols(df)
    N = len(c)
    sell, buy, sr, br = (np.zeros(N, bool) for _ in range(4))
    line, head, bos = (np.full(N, np.nan) for _ in range(3))
    pv = _pivots(h, l, k)
    for m in range(2, len(pv)):
        (si, sp, st), (li, lp, _), (hi_, hp, _) = pv[m - 2], pv[m - 1], pv[m]
        s = st                                       # +1: bearish shape (highs), -1: bullish mirror
        if s * (hp - sp) <= 0 or s * (sp - lp) <= 0:
            continue
        t0 = hi_ + k
        brk = None
        for t in range(t0, min(N, t0 + wait)):
            if s * (c[t] - hp) > 0:
                break
            if s * (lp - c[t]) > 0:
                brk = t
                break
        if brk is None:
            continue
        for t in range(brk + 1, min(N, brk + wait + 1)):
            if s * (c[t] - hp) > 0:
                break
            touch = h[t] >= sp if s > 0 else l[t] <= sp
            if touch:
                if not np.isfinite(line[t]):
                    (sell if s > 0 else buy)[t] = True
                    line[t], head[t], bos[t] = sp, hp, lp
                    if s * (sp - c[t]) > 0:
                        (sr if s > 0 else br)[t] = True
                break
    return {"sell": sell, "buy": buy, "sell_rej": sr, "buy_rej": br, "line": line, "head": head, "bos": bos}


def brooks_count(df):
    """Al Brooks' pullback bar count. Dict: h1, h2 (first / second bar over the prior bar's high in a pullback), l1, l2
    (mirror), pb_low / pb_high (the pullback's extreme so far)."""
    o, h, l, c, v = cols(df)
    N = len(c)
    h1, h2, l1, l2 = (np.zeros(N, bool) for _ in range(4))
    pbl, pbh = np.full(N, np.nan), np.full(N, np.nan)
    top, low_pb, cu, in_u = (h[0], l[0], 0, False) if N else (0, 0, 0, False)
    bot, high_pb, cd, in_d = (l[0], h[0], 0, False) if N else (0, 0, 0, False)
    for t in range(1, N):
        if h[t] >= top:
            top, cu, in_u, low_pb = h[t], 0, False, l[t]
        else:
            low_pb = min(low_pb, l[t])
            if h[t] < h[t - 1]:
                in_u = True
            elif in_u and h[t] > h[t - 1]:
                cu += 1
                in_u = False
                h1[t], h2[t] = cu == 1, cu == 2
        pbl[t] = low_pb
        if l[t] <= bot:
            bot, cd, in_d, high_pb = l[t], 0, False, h[t]
        else:
            high_pb = max(high_pb, h[t])
            if l[t] > l[t - 1]:
                in_d = True
            elif in_d and l[t] < l[t - 1]:
                cd += 1
                in_d = False
                l1[t], l2[t] = cd == 1, cd == 2
        pbh[t] = high_pb
    return {"h1": h1, "h2": h2, "l1": l1, "l2": l2, "pb_low": pbl, "pb_high": pbh}


def pivot_boss(df, n=1, lookback=20):
    """Ochoa's reversal candles on candles of n bars, as +1 / -1 on the completing bar: wick, extreme, outside, doji;
    stop = the pattern's far extreme."""
    o, h, l, c, v, pos = agg_candles(df, n)
    N, m = len(df), len(c)
    wick, extreme, outside, doji = (np.zeros(N, int) for _ in range(4))
    stop = np.full(N, np.nan)
    r = h - l
    avg = S(r).rolling(lookback).mean().shift(1).to_numpy()
    sma10 = S(c).rolling(10).mean().to_numpy()
    lo10, hi10 = S(l).rolling(10).min().to_numpy(), S(h).rolling(10).max().to_numpy()
    for j in range(2, m):
        p = pos[j]
        if not np.isfinite(avg[j]) or r[j] <= 0:
            continue
        body = abs(c[j] - o[j])
        lw, uw = min(o[j], c[j]) - l[j], h[j] - max(o[j], c[j])
        if lw >= 2.5 * max(body, 1e-12) and c[j] >= l[j] + 0.65 * r[j] and l[j] <= lo10[j]:
            wick[p], stop[p] = 1, l[j]
        elif uw >= 2.5 * max(body, 1e-12) and c[j] <= h[j] - 0.65 * r[j] and h[j] >= hi10[j]:
            wick[p], stop[p] = -1, h[j]
        pr, pb = r[j - 1], abs(c[j - 1] - o[j - 1])
        if pr >= 1.6 * avg[j] and 0.5 * pr < pb <= 0.85 * pr:
            if c[j - 1] < o[j - 1] and c[j] > o[j]:
                extreme[p], stop[p] = 1, min(l[j], l[j - 1])
            elif c[j - 1] > o[j - 1] and c[j] < o[j]:
                extreme[p], stop[p] = -1, max(h[j], h[j - 1])
        if r[j] >= 1.05 * avg[j]:
            if l[j] < l[j - 1] and c[j] > h[j - 1]:
                outside[p], stop[p] = 1, l[j]
            elif h[j] > h[j - 1] and c[j] < l[j - 1]:
                outside[p], stop[p] = -1, h[j]
        for b in (1, 2):                              # the doji b candles ago, confirmed by this close
            q = j - b
            if q < 10 or r[q] <= 0 or abs(c[q] - o[q]) > 0.01 * r[q]:
                continue
            if h[q] < sma10[q] and c[j] > h[q]:
                doji[p], stop[p] = 1, min(l[q:j + 1])
            elif l[q] > sma10[q] and c[j] < l[q]:
                doji[p], stop[p] = -1, max(h[q:j + 1])
    return {"wick": wick, "extreme": extreme, "outside": outside, "doji": doji, "stop": stop}
