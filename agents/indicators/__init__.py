"""Shared indicator library of Taloona Trade (user, 14.11.2026: "I don't want to compromise - not on strategies and
not on indicators that can help the strategies"). The scout ports every useful indicator here - from its open source
(TradingView open-source Pine, GitHub, MQL5) or, when the code is closed, as our own build from the description and
the chart - and every bot's own agent tests it as a filter / entry / exit on its own instruments.

Contract: fn(df, **params) with df columns o, h, l, c, v and a UTC DatetimeIndex; returns a numpy array or a dict of
numpy arrays, each len(df). No repaint: the value at bar i uses only bars <= i (pivots are shifted to the bar where
they are confirmed). Each module's docstring has the source link and what is our own reconstruction.

INDICATORS = {"name": (fn, "one-line description", "source url")}
"""
from agents.indicators.atr_fib_envelopes import atr_fib_envelopes
from agents.indicators.cisd import cisd
from agents.indicators.classic import (adx_di, bb_width_pct, ema_stack, fisher_transform, ichimoku, laguerre_rsi,
                                       macd_hist, mfi, qqe, raschke_osc, rsi_regime, ssl_channel, supertrend)
from agents.indicators.double_ribbon import double_ribbon
from agents.indicators.ehlers import ehlers_itrend, mama_fama
from agents.indicators.fib_legs import fib_legs
from agents.indicators.fvg_quality import fvg_quality
from agents.indicators.funk_rsi import funk_rsi
from agents.indicators.golden_wave import golden_wave
from agents.indicators.impulse_zones import impulse_zones
from agents.indicators.judas_swing import judas_swing
from agents.indicators.levels import htf_levels, pivot_points
from agents.indicators.liquidity import liquidity
from agents.indicators.lorentzian import lorentzian
from agents.indicators.nwe_envelope import nwe_envelope
from agents.indicators.regime import choppiness, efficiency_ratio, squeeze_pro
from agents.indicators.sessions_ict import judas_open_range, silver_bullet
from agents.indicators.order_flow import absorption, big_trades, buy_sell_pressure, cvd
from agents.indicators.smc import premium_discount, smc
from agents.indicators.sr_levels import sr_levels
from agents.indicators.supertrend_ai import supertrend_ai
from agents.indicators.supply_demand import supply_demand
from agents.indicators.trend_duration import trend_duration
from agents.indicators.trendlines import trendlines
from agents.indicators.ut_bot import ut_bot
from agents.indicators.volume_flow import cmf, pocket_pivot
from agents.indicators.volume_profile import delta_void_profile, tpo_profile, volume_profile
from agents.indicators.vwap_bands import anchored_vwap_swing, vwap_bands
from agents.indicators.wavetrend import wavetrend
from agents.indicators.williams_gsv import williams_gsv
from agents.indicators.zero_lag_trend import zero_lag_trend

TV = "https://www.tradingview.com/script/"

INDICATORS = {
    "supertrend_ai": (supertrend_ai, "SuperTrend AI: k-means picks the best ATR factor (1-5), flips = buy/sell (LuxAlgo; = Flux 'AI SuperTrend')",
                      TV + "wP7WWjLL-SuperTrend-AI-Clustering-LuxAlgo/"),
    "lorentzian": (lorentzian, "ML Lorentzian Classification: kNN (k 8) on RSI/WT/CCI/ADX features, 4-bar labels, + RQ kernel (jdehorty)",
                   TV + "WhBzgfDu-Machine-Learning-Lorentzian-Classification/"),
    "trend_duration": (trend_duration, "Trend Duration Forecast: HMA trend age vs the average / median length of past trends (ChartPrime)",
                       TV + "L3SJFfAQ/"),
    "zero_lag_trend": (zero_lag_trend, "Zero Lag Trend Signals: ZLEMA 70 +/- highest ATR x1.2, trend flips and re-entries (AlgoAlpha)",
                       TV + "tHK4O1Fi-Zero-Lag-Trend-Signals-MTF-AlgoAlpha/"),
    "atr_fib_envelopes": (atr_fib_envelopes, "ATR Fibonacci Trend Envelopes: RMA100 +/- 3 ATR trend + golden-pocket fib lines (BigBeluga)",
                          TV + "16toJPAD-ATR-Fibonacci-Trend-Envelopes-BigBeluga/"),
    "double_ribbon": (double_ribbon, "Double Ribbon: 11 offset SMAs vs 11 growing SMAs, trend score -1..1 (ChartPrime, our reconstruction)",
                      TV + "OdtelUqS-Double-Ribbon-ChartPrime/"),
    "golden_wave": (golden_wave, "Fibonacci Golden Wave: 0.5-0.618 zone of the 20-bar range, pullback touches (Flux Charts)",
                    TV + "SWe51Ddl-Fibonacci-Golden-Wave-Flux-Charts"),
    "cisd": (cisd, "Change in State of Delivery: close through the open of the last opposite candle run (ICT / LuxAlgo, = CISD Finder)",
             TV + "gHHRpuHC-Change-in-State-of-Delivery-CISD-LuxAlgo"),
    "impulse_zones": (impulse_zones, "Impulse Zones: big-range + high-volume bars become zones, retests (Flux Charts)",
                      TV + "Iv9MOWJ1-Impulse-Zones-Flux-Charts/"),
    "funk_rsi": (funk_rsi, "Funk RSI: RSI 30/70 swing reversals, target the swing, stop 14 bars back (Almazov Trade)",
                 TV + "823fXWAN-funk-rsi-break-the-rules-almazov-trade/"),
    "wavetrend": (wavetrend, "WaveTrend 10/21: OB/OS, crosses, divergences, trend (LazyBear)",
                  TV + "2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/"),
    "nwe_envelope": (nwe_envelope, "Nadaraya-Watson Envelope, non-repainting endpoint version, band-cross reversal dots (LuxAlgo; = the 'Non Repaint' ad)",
                     TV + "Iko0E2kL-Nadaraya-Watson-Envelope-LuxAlgo/"),
    "volume_profile": (volume_profile, "Session volume profile on tick volume: POC/VAH/VAL, HVN/LVN, P/b/D/B/trend shape (our build)",
                       "https://www.tradingview.com/support/solutions/43000703072-volume-profile-indicators-basic-concepts/"),
    "tpo_profile": (tpo_profile, "TPO (time-at-price) market profile: POC/VAH/VAL/shape by time (our build)",
                    "https://www.tradingview.com/support/solutions/43000713306-time-price-opportunity-tpo-indicator/"),
    "delta_void_profile": (delta_void_profile, "Delta Void Profile: previous session's void (deepest gap) + POC buy/sell delta (BigBeluga idea, our build)",
                           TV + "hD0LaluR-Delta-Void-Profile-BigBeluga/"),
    "vwap_bands": (vwap_bands, "Session / weekly / anchored VWAP with 1-2-3 standard-deviation bands (our build of the standard)",
                   "https://www.tradingview.com/support/solutions/43000502018-volume-weighted-average-price-vwap/"),
    "anchored_vwap_swing": (anchored_vwap_swing, "Anchored VWAPs from the last confirmed swing high and swing low",
                            "https://www.tradingview.com/support/solutions/43000652199-anchored-vwap/"),
    "buy_sell_pressure": (buy_sell_pressure, "Buyer vs seller pressure oscillator from the bar's close location x tick volume (ProTrippa idea, our build)",
                          TV + "1TpX6kXo-Buying-Selling-Volume/"),
    "cvd": (cvd, "Approximate cumulative volume delta per session + price/CVD divergences (our build)",
            TV + "1TpX6kXo-Buying-Selling-Volume/"),
    "absorption": (absorption, "Absorption: 2x tick volume with a small body at a 10-bar extreme (Orderflow Sniper idea, our build)",
                   "https://www.tiktok.com/ (Orderflow Sniper, task tal1411of)"),
    "big_trades": (big_trades, "'Big trades': tick-volume z-score >= 3 on the bar, buy / sell side by close location (Meks Trades idea)",
                   "https://www.tiktok.com/ (Meks Trades, task tal1411ind)"),
    "liquidity": (liquidity, "Equal highs / lows, liquidity sweeps and post-sweep entry zones (LuxAlgo EQH/EQL definition, our build)",
                  TV + "CnB3fSph-Smart-Money-Concepts-SMC-LuxAlgo/"),
    "smc": (smc, "Smart Money Concepts: BOS / CHoCH, order blocks, breaker blocks, FVG, premium/discount (LuxAlgo port, core)",
            TV + "CnB3fSph-Smart-Money-Concepts-SMC-LuxAlgo/"),
    "premium_discount": (premium_discount, "ICT premium / discount of the dealing range + OTE 61.8-79% zones",
                         TV + "CnB3fSph-Smart-Money-Concepts-SMC-LuxAlgo/"),
    "sr_levels": (sr_levels, "Support / resistance from merged pivots with test counts, hold %, breaks and flipped-level retests (Flux / BigBeluga ideas, our build)",
                  TV + "E8byEL6g-Adaptive-Support-and-Resistance-Zones-BigBeluga/"),
    "supply_demand": (supply_demand, "Supply & demand zones (base + departure), first touches and breakouts (zone matrix, our build)",
                      "https://www.tradingview.com/scripts/supplyanddemand/"),
    "trendlines": (trendlines, "Auto trend lines from the last two pivots, touches, breaks, triangle / channel / wedge detector (BigBeluga Pattern Detector idea, our build)",
                   "https://www.tradingview.com/scripts/chartpatterns/"),
    "pivot_points": (pivot_points, "Traditional daily / weekly pivot points P, R1-R3, S1-S3",
                     "https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/"),
    "htf_levels": (htf_levels, "Previous day / week high-low and today's high-low so far",
                   "https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/"),
    "judas_swing": (judas_swing, "ICT Judas Swing: London / NY window sweeps the Asian range and reverses, + its order block (our build)",
                    "https://www.tradingview.com/scripts/judasswing/"),
    "ichimoku": (ichimoku, "Ichimoku 9/26/52 cloud, TK cross, look-ahead-free Chikou check",
                 "https://www.tradingview.com/support/solutions/43000589152-ichimoku-cloud/"),
    "mfi": (mfi, "Money Flow Index 14 on tick volume, exits from 20 / 80",
            "https://www.tradingview.com/support/solutions/43000502348-money-flow-index-mfi/"),
    "adx_di": (adx_di, "ADX with +DI / -DI crosses and ADX rising",
               "https://www.tradingview.com/support/solutions/43000502250-average-directional-index-adx/"),
    "bb_width_pct": (bb_width_pct, "Bollinger 20/2 width and its percentile over 100 bars (volatility filter)",
                     "https://www.tradingview.com/support/solutions/43000501972-bollinger-bands-width-bbw/"),
    "ema_stack": (ema_stack, "EMA 9/21/50/100/200 order score as a graded trend filter",
                  "https://www.tradingview.com/support/solutions/43000592270-exponential-moving-average/"),
    "rsi_regime": (rsi_regime, "RSI range shift: 40-80 in up trends, 20-60 in down trends; trend pullback / momentum turns",
                   "https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/"),
    "qqe": (qqe, "QQE: smoothed RSI vs its ATR-of-RSI trailing line (factor 4.238)",
            "https://doc.stocksharp.com/en/api-examples/0440_Qqe_Signals"),
    "laguerre_rsi": (laguerre_rsi, "Ehlers Laguerre RSI (gamma 0.7), crosses of 30 / 70",
                     "https://doc.stocksharp.com/en/api-examples/0024_RSI_Laguerre"),
    "fisher_transform": (fisher_transform, "Ehlers Fisher Transform 9 with the previous-bar trigger",
                         "https://www.fmz.com/lang/en/strategy/503845"),
    "ssl_channel": (ssl_channel, "SSL Channel 10: SMA of highs / lows, side flips",
                    "https://www.tradingview.com/scripts/sslchannel/"),
    "supertrend": (supertrend, "Classic SuperTrend ATR 10 x 3", "https://www.tradingview.com/support/solutions/43000634738-supertrend/"),
    "macd_hist": (macd_hist, "MACD 12/26/9 line, signal, histogram, crosses",
                  "https://www.tradingview.com/support/solutions/43000502344-macd-moving-average-convergence-divergence/"),
    "raschke_osc": (raschke_osc, "Raschke 3/10 oscillator with its 16-bar signal line", "https://www.mql5.com/en/articles/24280"),
    "fvg_quality": (fvg_quality, "FVG quality: candle-3 weak / strong / big wick (thecrypteller), significant displacement (SecondBell), buy share of the gap's volume (Volumatic idea)",
                    "https://www.tradingview.com/script/PsBN8tD2-Fair-Value-Gaps-Significant-Displacement/"),
    "fib_legs": (fib_legs, "Latest up / down impulse leg: golden zone 0.5-0.618, 0.618-0.786, extensions 1.272-2.618, leg order block, FVG overlapping the zone",
                 "https://www.tradingview.com/script/C62RhZ7n/"),
    "silver_bullet": (silver_bullet, "ICT Silver Bullet 03-04 / 10-11 / 14-15 NY: pool sweep, FVG in the window, entry back from the gap, target the opposing pool",
                      "https://www.luxalgo.com/library/concept/silver-bullet/"),
    "judas_open_range": (judas_open_range, "Judas swing of the opening range (NY 09:30-09:45 or 08:15-08:30) with a CISD close (TradingFinder ICT Setup 03)",
                         "https://www.tradingview.com/script/mRmLRVSW-ICT-Setup-03-TradingFinder-Judas-Swing-NY-9-30am-CHoCH-FVG/"),
    "choppiness": (choppiness, "Choppiness Index 14: < 38.2 trending, > 61.8 choppy",
                   "https://doc.stocksharp.com/en/api-examples/0045_Choppiness_Index_Breakout"),
    "efficiency_ratio": (efficiency_ratio, "Kaufman Efficiency Ratio 10, signed -1..+1, zones +/- 0.6",
                         "https://whselfinvest.com/en/trading_strategies_33_Kaufman_Efficiency_Ratio.php"),
    "squeeze_pro": (squeeze_pro, "TTM Squeeze Pro: BB 20/2 inside KC 2.0 / 1.5 / 1.0 (low / mid / high compression), fire and linreg momentum",
                    "https://www.tradingview.com/script/0drMdHsO-TTM-Squeeze-Pro/"),
    "mama_fama": (mama_fama, "Ehlers MESA Adaptive MA (MAMA) and FAMA, limits 0.5 / 0.05, crosses",
                  "https://mesasoftware.com/papers/papers/MAMA.pdf"),
    "ehlers_itrend": (ehlers_itrend, "Ehlers Instantaneous Trendline (alpha 0.07, hl2) and its Trigger, crosses",
                      "https://www.tradingview.com/script/edDSj1kq-blackcat-L2-Ehlers-Instantaneous-Trendline-Trading-Strategy/"),
    "cmf": (cmf, "Chaikin Money Flow 20 on tick volume, zero-line side", "https://therighttrader.com/learn/indicators/chaikin-money-flow"),
    "pocket_pivot": (pocket_pivot, "Pocket Pivot: up bar with volume over every down bar's of 10, over SMA50, near SMA10 (Kacher / Morales)",
                     "https://www.luxalgo.com/library/concept/pocket-pivot/"),
    "ut_bot": (ut_bot, "UT Bot Alerts: ATR(10) x 3 trailing stop, close (or Heikin-Ashi close) crosses = buy / sell",
               "https://www.tradingview.com/script/n8ss8BID-UT-Bot-Alerts/"),
    "williams_gsv": (williams_gsv, "Larry Williams Greatest Swing Value: open +/- 1.8 x average failure swing, 5-bar setup",
                     "https://www.mql5.com/en/articles/21913"),
}

__all__ = ["INDICATORS"] + list(INDICATORS)
