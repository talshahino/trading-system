"""Volume-based gauges on tick volume (v).

cmf - Chaikin Money Flow: MFV = ((c - l) - (h - c)) / (h - l) x v; CMF = sum MFV(n) / sum v(n), n = 20 (The Right
Trader https://therighttrader.com/learn/indicators/chaikin-money-flow ; high = low -> MFV 0; zero volume -> 0). Default:
the close location in the true range (Twiggs' gap correction; on gap-free FX bars the same as Chaikin's).

pocket_pivot - Pocket Pivot (Kacher & Morales, 'Trade Like an O'Neil Disciple'; MarketSmith / LuxAlgo concept
https://www.luxalgo.com/library/concept/pocket-pivot/): an up bar (close > previous close) whose volume is above the
highest volume of every down bar of the previous 10 bars, closing above the 50-bar SMA and not more than 'ext' x ATR
over the 10-bar SMA (the authors: buy near the 10-day line, not extended - OUR translation of 'near' = within
1.5 ATR(14)). A feed without volume information (constant v) passes the volume test (library convention). On
closed bars - no repaint."""
import numpy as np

from agents.indicators._core import S, atr, cols, shift, sma


def cmf(df, n=20, true_range=True):
    """array -1..+1. true_range=True: the close location in the true range (high / low extended to the previous close -
    Colin Twiggs' gap correction); False = Chaikin's original bar range."""
    o, h, l, c, v = cols(df)
    if true_range:
        pc = np.concatenate([[c[0]], c[:-1]])
        h, l = np.fmax(h, pc), np.fmin(l, pc)
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where(rng > 0, ((c - l) - (h - c)) / rng, 0.0)
    mfv = S(mfm * v).rolling(n).sum().to_numpy()
    vs = S(v).rolling(n).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vs > 0, mfv / vs, 0.0)


def pocket_pivot(df, look=10, ext=1.5):
    """dict: pp (bool), down_vol_max (highest down-bar volume of the previous `look` bars)."""
    o, h, l, c, v = cols(df)
    pc = shift(c, 1)
    down_v = np.where(c < pc, v, 0.0)
    dvm = shift(S(down_v).rolling(look, min_periods=1).max().to_numpy(), 1)
    s10, s50 = sma(c, 10), sma(c, 50)
    a = atr(h, l, c, 14)
    flat = S(v).rolling(look + 1).std(ddof=0).to_numpy() == 0       # no volume information -> the volume test passes
    pp = (c > pc) & ((v > np.nan_to_num(dvm, nan=np.inf)) | flat) & (c > s50) & (c - s10 <= ext * a) & (c >= s10 - 0.5 * a)
    return {"pp": np.nan_to_num(pp, nan=0).astype(bool), "down_vol_max": dvm}
