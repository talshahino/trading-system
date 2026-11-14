"""Fair Value Gap quality (user's tasks tal1411gapplus (4) and (5), 14.11.2026) - three published FVG filters in one place:

1. Candle-3 quality (thecrypteller, TikTok: "most traders lose with FVGs because they trade the wrong ones"): a strong
   third candle (big body) = price probably will not come back; a big wick on candle 3 = do not trust it; a weak
   candle 3 (small body, small wicks) = a good gap. OUR BUILD of the thresholds (the video gives none): body / wick
   measured in ATR(14) of the bar before the gap; weak = body < 0.5 ATR and total wick < 0.5 ATR; strong = body >= 1.0 ATR;
   big wick = total wick >= 0.5 ATR and >= the body.
2. Significant displacement (SecondBell, open source: https://www.tradingview.com/script/PsBN8tD2-Fair-Value-Gaps-Significant-Displacement/):
   middle candle relative volume >= 1.25 x its 20-bar average (the lookback is not stated - ours 20), gap >= 0.10 ATR,
   middle body >= 0.70 ATR and >= 60% of its range, middle candle in the gap's direction.
3. Buy / sell volume inside the gap (Volumatic Fair Value Gaps [BigBeluga] / Flux 'FVG with strong buying volume';
   BigBeluga samples a 10x lower timeframe - closed to us with OHLCV only, so OUR RECONSTRUCTION: each of the three
   candles' tick volume is split by its close location in the TRUE range (a gap from the previous close counts), buy =
   v x (c - min(l, c[1])) / (max(h, c[1]) - min(l, c[1])), sell = the rest; buy_share = buy / total of the three
   candles). A feed without volume information (constant v) passes the relative-volume test (library convention).

All values are set on candle 3 (bar i), when the gap is complete - no repaint. Bullish gap: l[i] > h[i-2] (zone
h[i-2]..l[i]); bearish: h[i] < l[i-2] (zone h[i]..l[i-2])."""
import numpy as np

from agents.indicators._core import S, atr, cols, shift, sma


def fvg_quality(df, atr_n=14, vol_n=20):
    """dict (every array len(df)): bull, bear (a new gap closes on this bar), top, bot (its zone), gap_atr,
    c3_body_atr, c3_wick_atr, c3_weak, c3_strong, c3_bigwick, displacement, buy_share (0..1, of the three candles)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = shift(atr(h, l, c, atr_n), 1)                      # ATR known before candle 3
    h2, l2 = shift(h, 2), shift(l, 2)
    bull = np.nan_to_num(l - h2, nan=-1) > 0
    bear = np.nan_to_num(l2 - h, nan=-1) > 0
    top = np.where(bull, l, np.where(bear, l2, np.nan))
    bot = np.where(bull, h2, np.where(bear, h, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_atr = (top - bot) / a
        body3 = np.abs(c - o) / a
        wick3 = ((h - l) - np.abs(c - o)) / a
        mb = np.abs(shift(c, 1) - shift(o, 1))
        mr = shift(h, 1) - shift(l, 1)
        rv = shift(v, 1) / shift(sma(v, vol_n), 2)
        flat = shift(S(v).rolling(vol_n).std(ddof=0).to_numpy(), 1) == 0   # no volume information -> passes
        disp = (((rv >= 1.25) | flat) & (gap_atr >= 0.10) & (mb / a >= 0.70) & (mb >= 0.60 * mr)
                & ((bull & (shift(c, 1) > shift(o, 1))) | (bear & (shift(c, 1) < shift(o, 1)))))
        pc = shift(c, 1)
        th, tl = np.fmax(h, pc), np.fmin(l, pc)                       # true range: a gap counts
        rng = np.where(th - tl > 0, th - tl, np.nan)
        buy = np.nan_to_num(v * (c - tl) / rng, nan=0.5 * np.nan_to_num(v))
        b3 = buy + shift(buy, 1) + shift(buy, 2)
        v3 = v + shift(v, 1) + shift(v, 2)
        share = np.where(v3 > 0, b3 / v3, np.nan)
    has = bull | bear
    return {
        "bull": bull, "bear": bear, "top": top, "bot": bot,
        "gap_atr": np.where(has, gap_atr, np.nan),
        "c3_body_atr": np.where(has, body3, np.nan), "c3_wick_atr": np.where(has, wick3, np.nan),
        "c3_weak": has & (np.nan_to_num(body3, nan=9) < 0.5) & (np.nan_to_num(wick3, nan=9) < 0.5),
        "c3_strong": has & (np.nan_to_num(body3) >= 1.0),
        "c3_bigwick": has & (np.nan_to_num(wick3) >= 0.5) & (np.nan_to_num(wick3) >= np.nan_to_num(body3)),
        "displacement": has & np.nan_to_num(disp, nan=0).astype(bool),
        "buy_share": np.where(has, share, np.nan),
    }
