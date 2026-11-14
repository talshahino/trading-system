"""UT Bot Alerts (QuantNomad / Yo_adriiiiaan, open source https://www.tradingview.com/script/n8ss8BID-UT-Bot-Alerts/;
TradeIQ strategy https://www.tradingview.com/script/pWT8wmAC-UT-Bot-Strategy-with-Risk-Management-TradeIQ/).
Port of the Pine: nLoss = key x ATR(atr_n); the trailing stop xATRTrailingStop ratchets with the source (src > prev
stop and src[1] > prev stop -> max(prev, src - nLoss); both under -> min(prev, src + nLoss); else flips);
buy = src crosses above the stop, sell = crosses below. src = close or the Heikin-Ashi close (o+h+l+c)/4
(heikin=True, TradeIQ's option). Closed bars only - no repaint."""
import numpy as np

from agents.indicators._core import atr, cols


def ut_bot(df, key=3.0, atr_n=10, heikin=False):
    """dict: trail, buy, sell, pos (+1 / -1 regime)."""
    o, h, l, c, v = cols(df)
    src = (o + h + l + c) / 4 if heikin else c
    nl = key * np.nan_to_num(atr(h, l, c, atr_n))
    n = len(c)
    t = np.zeros(n)
    buy, sell = np.zeros(n, bool), np.zeros(n, bool)
    pos = np.zeros(n)
    for i in range(n):
        if i == 0:
            t[i] = src[i] - nl[i]
            continue
        p = t[i - 1]
        if src[i] > p and src[i - 1] > p:
            t[i] = max(p, src[i] - nl[i])
        elif src[i] < p and src[i - 1] < p:
            t[i] = min(p, src[i] + nl[i])
        else:
            t[i] = src[i] - nl[i] if src[i] > p else src[i] + nl[i]
        buy[i] = src[i] > t[i] and src[i - 1] <= t[i - 1]
        sell[i] = src[i] < t[i] and src[i - 1] >= t[i - 1]
        pos[i] = 1 if buy[i] else (-1 if sell[i] else pos[i - 1])
    buy[:atr_n + 1] = False
    sell[:atr_n + 1] = False
    return {"trail": t, "buy": buy, "sell": sell, "pos": pos}
