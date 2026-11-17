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
from agents.indicators.halftrend import halftrend
from agents.indicators.golden_wave import golden_wave
from agents.indicators.impulse_zones import impulse_zones
from agents.indicators.judas_swing import judas_swing
from agents.indicators.levels import htf_levels, pivot_points
from agents.indicators.levels_ict import crt, msnr
from agents.indicators.liquidity import liquidity
from agents.indicators.lorentzian import lorentzian
from agents.indicators.ml_filters import (gaussian_channel, kalman_supertrend, knn_trend_navigator, sniper_ribbon,
                                          trend_magnet, volume_mapping)
from agents.indicators.nwe_envelope import nwe_envelope
from agents.indicators.regime import choppiness, efficiency_ratio, squeeze_pro
from agents.indicators.sessions_ict import judas_open_range, silver_bullet
from agents.indicators.order_flow import absorption, big_trades, buy_sell_pressure, cvd
from agents.indicators.smc import premium_discount, smc
from agents.indicators.smt import smt_divergence
from agents.indicators.sr_levels import sr_levels
from agents.indicators.supertrend_ai import supertrend_ai
from agents.indicators.supply_demand import supply_demand
from agents.indicators.trend_duration import trend_duration
from agents.indicators.trendlines import trendlines
from agents.indicators.ut_bot import ut_bot
from agents.indicators.volume_flow import cmf, pocket_pivot
from agents.indicators.vumanchu import cipher_b, range_filter
from agents.indicators.volume_profile import delta_void_profile, tpo_profile, volume_profile
from agents.indicators.vwap_bands import anchored_vwap_swing, vwap_bands
from agents.indicators.wavetrend import wavetrend
from agents.indicators.williams_gsv import williams_gsv
from agents.indicators.zero_lag_trend import zero_lag_trend
from agents.indicators.ifvg import ifvg
from agents.indicators.heikin_ashi import heikin_ashi
from agents.indicators.squeeze_momentum import squeeze_momentum
from agents.indicators.swing_profile import swing_profile
from agents.indicators.classic2 import (coppock, eom, elder_ray, frama, hurst, klinger, linreg_channel, mass_index,
                                        mcginley, obv, renko_trend, smart_money_index, ulcer_index, ultimate_oscillator,
                                        volatility_system, woodie_pivots, woodies_cci)
from agents.indicators.ict_extra import bpr, opening_gaps, po3_midnight, rejection_blocks, unicorn

from agents.indicators.classic3 import (bop, chaikin_osc, chande_kroll, cmo, dpo, force_index, nvi_pvi, ppo, qstick, rvi,
                                        rwi, starc, tii, tsi, vhf, volatility_stop)
from agents.indicators.ehlers2 import (center_of_gravity, cyber_cycle, cybernetic_osc, decycler, ebsw, roofing,
                                       sinewave)
from agents.indicators.tv_popular import (coral, crsi, gmma, ift_rsi, macd_v, pmax, rsx, smoothed_ha, tdi, vfi,
                                          waddah_attar)
from agents.indicators.classic4 import (alligator, bw_ao, bw_mfi, chop_zone, connors_rsi, demarker, ewo, gann_hilo, imi,
                                        inertia, kase_peak, kdj, kst, pfe, pgo, psar, pvt, rvi_vol, safezone, smi_blau,
                                        stc, stretch, td_sequential, ttf, vidya, vortex)
from agents.indicators.ehlers3 import (continuation_index, ehlers_uo, laguerre_filter, one_euro, pma_predict,
                                       reflex_trendflex, synthetic_osc, two_pole_predict, ultimate_channel)
from agents.indicators.classic5 import (alma, ash, bbtrend, fractal_bands, kairi, pmo, pzo, rmi, t3, trendilo,
                                        twiggs_mf, vix_fix, vzo)
from agents.indicators.structure2 import (darvas_box, gap_type, point_figure, ross_123, swing_grade, three_line_break,
                                          valid_ob, vsa)
from agents.indicators.levels2 import adr, cbdr, ict_macros, po3_4h, round_levels, rvol
from agents.indicators.ehlers4 import adaptive_supersmoother, autotune_filter, reversion_index
from agents.indicators.candles import candle_patterns
from agents.indicators.divergence import divergence
from agents.indicators.auction import naked_poc, poor_extremes, single_prints
from agents.indicators.demark2 import td_pressure, td_rei
from agents.indicators.daycycle import noise_area, taylor_cycle
from agents.indicators.classic6 import (chaikin_volatility, correlation_trend, kase_devstop, madrid_ribbon, murrey_math,
                                        pitchfork, projection, rainbow, sentiment_zone, standard_error_bands, ttm_scalper,
                                        ttm_trend, wave_pm)
from agents.indicators.classic7 import (dsma, elegant_osc, fib_bollinger, hull_butterfly, impulse_macd, mad,
                                        market_thermometer, range_detector, spearman, super_passband, trama,
                                        universal_osc, voss_predict, williams_ad)
TV = "https://www.tradingview.com/script/"
TT = "https://www.traders.com/Documentation/FEEDbk_docs/"
SC2 = "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/"

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
    "knn_trend_navigator": (knn_trend_navigator, "AI Trend Navigator [K-Neighbor]: KNN mean of past SMA(hl2) closest to RMA(close), WMA line + 10-bar classifier",
                            "https://www.tradingview.com/script/30aLnPNh-AI-Trend-Navigator-K-Neighbor/"),
    "gaussian_channel": (gaussian_channel, "Gaussian Channel [DW]: 4-pole Gaussian filter of hlc3 (period 144) +/- 1.414 x filtered TR",
                         "https://www.tradingview.com/script/WpVY7GKW-Gaussian-Channel-DW/"),
    "kalman_supertrend": (kalman_supertrend, "Pro Scalper reconstruction: SuperTrend 10 x 3 on a Kalman-filtered close + VWMA high / low bands",
                          "https://tradingview.com/script/BEhT6Syv-Pro-Scalper"),
    "trend_magnet": (trend_magnet, "Trend Magnet reconstruction (BigBeluga, closed): EMA(hlc3) +/- 1.5 ATR ratcheting band, flips, ADX<20 low momentum",
                     "https://www.tradingview.com/script/HNuao7jn-Market-Waves-BigBeluga/"),
    "volume_mapping": (volume_mapping, "Volume Mapping reconstruction (BigBeluga, closed): EMA of up-candle minus down-candle volume / average volume",
                       "https://docs.bigbeluga.com/oscillators/nautilus-tm/volume-mapping"),
    "sniper_ribbon": (sniper_ribbon, "Sniper V.02 reconstruction: EMA 9/21 ribbon cross, ribbon retests, bull / bear score (VWAP, RSI, MACD, ADX)",
                      "https://www.tradingview.com/script/HBDcHSB7-Sniper-Entry-Exit-with-SL-TP-by-KhanSaab-V-02/"),
    "cipher_b": (cipher_b, "VuManChu Cipher B: WaveTrend dots at +/-53, RSI+MFI area, fractal WaveTrend divergences",
                 "https://www.tradingview.com/script/Msm4SjwI-VuManChu-Cipher-B-Divergences/"),
    "range_filter": (range_filter, "VuManChu Swing Free = Range Filter (20, 3.5): ratcheting filter, buy / sell on state changes",
                     "https://in.tradingview.com/script/q23anHmP-VuManChu-Swing-Free"),
    "halftrend": (halftrend, "HalfTrend (everget; BigBeluga signal engine): amplitude 2, ATR(100)/2 bands, long / short on flips",
                  "https://www.tradingview.com/script/ZHGPnlAz-HalfTrend-Long-Short-Signal-Engine-BigBeluga/"),
    "msnr": (msnr, "Malaysian SnR: A / V / gap levels from closes, fresh until a wick touches, role flip on a close through",
             "https://www.luxalgo.com/library/concept/malaysian-snr/"),
    "crt": (crt, "Candle Range Theory: higher-timeframe candle 2 sweeps candle 1's high / low and closes back inside",
            "https://www.luxalgo.com/library/concept/candle-range-theory/"),
    "smt_divergence": (smt_divergence, "SMT divergence vs a correlated partner at confirmed swing highs / lows (needs the partner's bars)",
                       "https://it.tradingview.com/script/PH9fxHqo-SMT-Divergence"),
    # ---- 15.11.2026: task tal1411set7 and the night's search ----
    "ifvg": (ifvg, "Inversion FVG: a gap closed through by a body flips role; retest of the inverted zone (LuxAlgo, our build of the published rules)",
             TV + "8FHucjY6-Inversion-Fair-Value-Gaps-IFVG-LuxAlgo/"),
    "heikin_ashi": (heikin_ashi, "Heikin Ashi candles: colour, no-opposite-wick, colour runs, flip + 2nd no-wick confirmation",
                    "https://www.investopedia.com/trading/heikin-ashi-better-candlestick/"),
    "squeeze_momentum": (squeeze_momentum, "Squeeze Momentum [LazyBear]: BB(20, 1.5 quirk) vs KC(20, 1.5 TR), linreg momentum, lime/green/red/maroon",
                         TV + "nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/"),
    "swing_profile": (swing_profile, "Swing Profile Analyzer: per-swing frequency-of-closes profile (or volume), leg POC / value area, developing leg POC (ChartPrime, our reconstruction)",
                      TV + "8S40eYNi-Swing-Profile-Analyzer-ChartPrime/"),
    "coppock": (coppock, "Coppock Curve: WMA10 of ROC14 + ROC11, turns", SC2 + "coppock-curve"),
    "ultimate_oscillator": (ultimate_oscillator, "Ultimate Oscillator 7/14/28 (Larry Williams)", SC2 + "ultimate-oscillator"),
    "klinger": (klinger, "Klinger Volume Oscillator 34/55, signal 13 (tick volume)", SC2 + "klinger-volume-oscillator"),
    "mass_index": (mass_index, "Mass Index 9/25 and Dorsey's reversal bulge (27 -> 26.5)", SC2 + "mass-index"),
    "mcginley": (mcginley, "McGinley Dynamic (k 0.6, exponent 4)", "https://doc.stocksharp.com/en/api-examples/1038_McGinley_Dynamic_Improved"),
    "elder_ray": (elder_ray, "Elder Ray: bull / bear power vs EMA13, Elder's buy / sell conditions", "https://www.luxalgo.com/library/concept/elder-ray/"),
    "volatility_system": (volatility_system, "Wilder Volatility System: close-based SAR 3 x ATR(7), stop and reverse",
                          TV + "pjZKWnHl-Volatility-System-by-Wilder-LucF"),
    "frama": (frama, "Ehlers FRAMA (16, hl2), fractal-dimension alpha", "https://www.traders.com/Documentation/FEEDbk_docs/2005/10/TradersTips/TradersTips.html"),
    "hurst": (hurst, "Rolling Hurst exponent (R/S, 100 log returns) - < 0.5..0.7 mean-reverting", "https://doc.stocksharp.com/en/api-examples/0228_Hurst_Exponent_Reversion"),
    "linreg_channel": (linreg_channel, "Linear regression channel 50 +/- 1.5 standard errors, slope", "https://doc.stocksharp.com/en/designer-gallery/Linear_Regression_Channel"),
    "renko_trend": (renko_trend, "Renko bricks on closes, 0.5 x StdDev(14) size, reversals", "https://doc.stocksharp.com/en/api-examples/1608_Renko_Trend_Reversal"),
    "woodies_cci": (woodies_cci, "Woodies CCI: turbo 6 / CCI 14, 6-bar trend, zero-line reject", "https://www.luxalgo.com/library/concept/woodies-cci-conventions/"),
    "woodie_pivots": (woodie_pivots, "Woodie pivots: PP = (H + L + 2 x open) / 4, R1/S1/R2/S2", "https://www.luxalgo.com/library/concept/woodie-pivots/"),
    "obv": (obv, "On-Balance Volume (tick volume), EMA20, prior 20-bar extremes", SC2 + "on-balance-volume-obv"),
    "eom": (eom, "Ease of Movement (Arms) SMA14, tick volume", SC2 + "ease-of-movement-emv"),
    "ulcer_index": (ulcer_index, "Ulcer Index 14 (drawdown depth / duration)", SC2 + "ulcer-index"),
    "smart_money_index": (smart_money_index, "Smart Money Index (Don Hays): - first 30 min + last hour, completed US days", "https://www.luxalgo.com/library/concept/smart-money-index/"),
    "rejection_blocks": (rejection_blocks, "ICT rejection blocks: long-wick swing extremes, body edge to wick, revisit signals", "https://www.luxalgo.com/library/concept/rejection-block/"),
    "bpr": (bpr, "ICT Balanced Price Range: overlap of opposite FVGs within 5 bars, retests (our build)", TV + "zPbYfcE8-ICT-Balance-Price-Range-UAlgo/"),
    "opening_gaps": (opening_gaps, "ICT NWOG / NDOG: previous close vs reopen across the 17:00 NY break, CE", "https://www.luxalgo.com/library/concept/opening-gap/"),
    "po3_midnight": (po3_midnight, "ICT Power of Three around the NY midnight open: Asian range, manipulation, distribution back through the open",
                     "https://www.luxalgo.com/library/concept/accumulation-manipulation-distribution/"),
    "unicorn": (unicorn, "ICT Unicorn: breaker block overlapping an FVG of the break, retests, breaker stop", "https://www.luxalgo.com/library/concept/unicorn/"),
    # scout 15.11.2026: classic oscillators / bands / stops (agents/indicators/classic3.py)
    "tsi": (tsi, "True Strength Index 25/13, signal 7 (Blau); r 20 / s 5 / sig 5 = SMI Ergodic", SC2 + "true-strength-index"),
    "cmo": (cmo, "Chande Momentum Oscillator 9, -100..100", "https://www.luxalgo.com/library/concept/chande-momentum-oscillator/"),
    "dpo": (dpo, "Detrended Price Oscillator 20, non-centred (no look-ahead)", SC2 + "detrended-price-oscillator-dpo"),
    "rvi": (rvi, "Relative Vigor Index 10 with its 4-bar weighted signal (Ehlers)", "https://www.luxalgo.com/blog/relative-vigor-index-to-judge-bull-bear-balance/"),
    "ppo": (ppo, "Percentage Price Oscillator 12/26/9 (MACD in % of the slow EMA)", SC2 + "percentage-price-oscillator-ppo"),
    "force_index": (force_index, "Elder Force Index EMA13 / EMA2 (tick volume)", SC2 + "force-index"),
    "chaikin_osc": (chaikin_osc, "Chaikin Oscillator EMA3 - EMA10 of the A/D line (tick volume)", SC2 + "chaikin-oscillator"),
    "nvi_pvi": (nvi_pvi, "Negative / Positive Volume Index with the 255 EMA (Fosback)", SC2 + "negative-volume-index-nvi"),
    "starc": (starc, "STARC bands SMA6 +/- 2 x ATR15 (Manning Stoller)", "https://www.luxalgo.com/library/concept/starc-bands/"),
    "rwi": (rwi, "Random Walk Index high / low, lookbacks 2..14 (Poulos)", "https://www.luxalgo.com/library/indicator/random-walk-index/"),
    "vhf": (vhf, "Vertical Horizontal Filter 28: trending vs congested (Adam White)", "https://ta-lib.org/functions/vhf.html"),
    "bop": (bop, "Balance of Power: SMA14 of (c - o) / (h - l) (Livshin)", "https://www.luxalgo.com/library/concept/balance-of-power/"),
    "qstick": (qstick, "QStick 8: SMA of (close - open) (Chande)", "https://www.luxalgo.com/library/concept/qstick/"),
    "tii": (tii, "Trend Intensity Index 60/30: share of positive deviations from the SMA60, 0..100 (M. H. Pee)", "https://www.luxalgo.com/library/concept/trend-intensity-index/"),
    "chande_kroll": (chande_kroll, "Chande Kroll Stop p 10 / x 1 / q 9, upper and lower stop lines", "https://www.luxalgo.com/library/concept/chande-kroll-stop/"),
    "volatility_stop": (volatility_stop, "Volatility Stop 20 x 2 ATR from the extreme close since the flip (TradingView VStop)", "https://www.tradingview.com/support/solutions/43000594695-volatility-stop/"),
    # Ehlers cycle tools (agents/indicators/ehlers2.py)
    "roofing": (roofing, "Ehlers Roofing filter (HP 48 -> SS 10) and its stochastic, 0.2 / 0.8 crosses", "https://toslc.thinkorswim.com/center/reference/Tech-Indicators/strategies/E-K/EhlersStoch"),
    "decycler": (decycler, "Ehlers Decycler (close - HP60) and the Decycler Oscillators 100 / 125, crosses", "https://www.traders.com/Documentation/FEEDbk_docs/2015/09/TradersTips.html"),
    "cyber_cycle": (cyber_cycle, "Ehlers Cyber Cycle (alpha 0.07) with its 1-bar trigger", "https://www.luxalgo.com/library/concept/cyber-cycle/"),
    "center_of_gravity": (center_of_gravity, "Ehlers Center of Gravity oscillator 10 (hl2) with its 1-bar trigger", "https://www.luxalgo.com/library/concept/center-of-gravity/"),
    "ebsw": (ebsw, "Ehlers Even Better Sinewave (HP 40, SS 10), -1..1", "https://www.luxalgo.com/library/indicator/even-better-sinewave/"),
    "sinewave": (sinewave, "Ehlers Hilbert Sinewave: Sine vs LeadSine of the dominant-cycle phase", "https://www.whselfinvest.de/en/trading-platform/free-trading-strategies/tradingsystem/09-sinewave-ehlers"),
    "cybernetic_osc": (cybernetic_osc, "Ehlers Cybernetic Oscillator (HP 30 -> SS 20) / 100-bar RMS (TASC 2025.06)", "https://www.tradingview.com/scripts/highpass/"),
    # TradingView / MT4 favourites (agents/indicators/tv_popular.py)
    "waddah_attar": (waddah_attar, "Waddah Attar Explosion: MACD(20, 40) change x 150 vs BB width and an ATR dead zone", TV + "iu3kKWDI-Waddah-Attar-Explosion-LazyBear/"),
    "tdi": (tdi, "Traders Dynamic Index: RSI13 price line 2 / signal 7 / base 34 +/- 1.6185 SD (Dean Malone)", "https://www.luxalgo.com/library/concept/traders-dynamic-index/"),
    "macd_v": (macd_v, "MACD-V: 100 x MACD / ATR26, signal 9, +/-50 neutral, +/-150 extended (Spiroglou)", SC2 + "macd-v"),
    "vfi": (vfi, "Volume Flow Indicator 130 (Katsanos; LazyBear), signal EMA5, tick volume", TV + "MhlDpfdS-Volume-Flow-Indicator-LazyBear/"),
    "coral": (coral, "Coral Trend 21 / 0.4 (LazyBear): EMA-cascade line, colour = slope", "https://www.luxalgo.com/library/concept/coral-trend/"),
    "pmax": (pmax, "Profit Maximizer: SuperTrend on EMA10(hl2), ATR10 x 3; 'rsi_t3' = PMax on T3(RSI) (KivancOzbilgic)", TV + "7yn5dteO-Profit-Maximizer-Strategy/"),
    "gmma": (gmma, "Guppy Multiple MAs 3-15 / 30-60: group edges, long-group width, bull / bear separation", "https://www.babypips.com/learn/forex/guppy-multiple-moving-average"),
    "rsx": (rsx, "Jurik RSX 14 (everget's open port), smooth RSI 0..100", TV + "20TzvKPk-Jurik-RSX/"),
    "crsi": (crsi, "Cyclic Smoothed RSI (whentotrade): cycle 20, vibration 10, 10/90 % dynamic bands", "https://www.prorealcode.com/prorealtime-indicators/rsi-cyclic-smoothed/"),
    "ift_rsi": (ift_rsi, "Inverse Fisher Transform of RSI5 (WMA9), -1..1 (Vervoort)", TV + "Nw4PDAwY-Inverse-Fisher-Transform-on-RSI-for-backtest-w-alerts/"),
    "smoothed_ha": (smoothed_ha, "Heiken Ashi Smoothed (SMMA6 -> HA -> LWMA2, the MT4 one used by TMS)", "https://www.prorealcode.com/topic/trading-made-simple-tms-system"),
    # scout 15.11.2026 night 2: classic indicators the library lacked (agents/indicators/classic4.py)
    "stc": (stc, "Schaff Trend Cycle 23/50/10, factor 0.5; 25 / 75 crosses (LazyBear port)", TV + "dbxXeuw2-Indicator-Schaff-Trend-Cycle-STC/"),
    "connors_rsi": (connors_rsi, "Connors RSI 3/2/100: RSI3 + streak RSI2 + 100-bar percent rank of ROC1, / 3", "https://www.luxalgo.com/library/concept/connors-rsi/"),
    "smi_blau": (smi_blau, "Stochastic Momentum Index 10/3/3 (Blau), signal EMA10, -100..100", "https://ta-lib.org/functions/smi.html"),
    "demarker": (demarker, "DeMarker 14 (Tom DeMark), 0..1, 0.3 / 0.7 zones", TV + "CF1kNGYo-DeMarker-DeM/"),
    "rvi_vol": (rvi_vol, "Relative Volatility Index 10/14 (Dorsey), close-based and refined (high + low)", "https://www.luxalgo.com/library/concept/relative-volatility-index/"),
    "inertia": (inertia, "Dorsey Inertia: 20-bar linear-regression value of the refined RVI", TV + "bzxmXFGd-Dorsey-Inertia/"),
    "safezone": (safezone, "Elder SafeZone stop 10 x 2.5, held 3 bars (long / short stop lines)", "https://www.luxalgo.com/library/concept/elder-safezone-stop/"),
    "kase_peak": (kase_peak, "Kase Peak Oscillator (our reconstruction, k 8..65) with a mean + 1.5 sd peak-out line", "https://www.mql5.com/en/forum/173722"),
    "pfe": (pfe, "Polarized Fractal Efficiency 10 / EMA5 (Hannula), -100..100", "https://www.luxalgo.com/library/concept/polarized-fractal-efficiency/"),
    "ttf": (ttf, "Trend Trigger Factor 15 (M. H. Pee, TASC 2004), +/-100 triggers", "https://www.traders.com/Documentation/FEEDbk_docs/2004/12/Abstracts_new/Pee/pee.html"),
    "gann_hilo": (gann_hilo, "Gann HiLo Activator 3: SMA3 of the highs / lows, trend flips", TV + "PyFdxzcj-Gann-HiLo-Activator-Strategy/"),
    "pgo": (pgo, "Pretty Good Oscillator 89 (Mark Johnson): (close - SMA) / EMA(TR), +/-3 breakouts", "https://www.luxalgo.com/library/concept/pretty-good-oscillator/"),
    "imi": (imi, "Intraday Momentum Index 14 (Chande): RSI of the candle bodies, 0..100", "https://www.luxalgo.com/library/concept/intraday-momentum-index/"),
    "ewo": (ewo, "Elliott Wave Oscillator SMA5 - SMA35 of hl2", "https://www.luxalgo.com/library/concept/5-35-oscillator/"),
    "kdj": (kdj, "KDJ 9/3/3: K, D and J = 3K - 2D", "https://www.kucoin.com/blog/What-is-KDJ-Indicator"),
    "vidya": (vidya, "VIDYA 14 / CMO 9 (Chande's variable index dynamic average)", TV + "kSf8O7HZ-VIDYA-Chande/"),
    "bw_ao": (bw_ao, "Bill Williams Awesome / Accelerator Oscillators with saucer and twin-peaks signals", "https://www.babypips.com/forexpedia/awesome-oscillator"),
    "bw_mfi": (bw_mfi, "Bill Williams Market Facilitation Index (h - l) / tick volume: green / fade / fake / squat bars", "https://www.luxalgo.com/library/concept/bill-williams-market-facilitation-index/"),
    "alligator": (alligator, "Williams Alligator SMMA 13/8/5 shifted 8/5/3 + fractals confirmed 2 bars late", "https://www.babypips.com/forexpedia/alligator-indicator"),
    "chop_zone": (chop_zone, "TradingView Chop Zone: EMA34 angle in 9 colour zones (+4 .. -4)", "https://www.tradingview.com/support/solutions/43000589111-chop-zone-chop/"),
    "pvt": (pvt, "Price Volume Trend (tick volume) with an EMA21 signal", SC2 + "price-volume-trend-pvt"),
    "vortex": (vortex, "Vortex Indicator 14 VI+ / VI- (Botes & Siepman)", SC2 + "vortex-indicator"),
    "psar": (psar, "Parabolic SAR 0.02 / 0.02 / 0.2 (Wilder), trend and flips", SC2 + "parabolic-sar"),
    "kst": (kst, "Know Sure Thing (Pring) with its SMA9 signal", SC2 + "know-sure-thing-kst"),
    "stretch": (stretch, "Toby Crabel stretch: 10-day mean of min(h - o, o - l), levels open +/- stretch (NY day)", "https://store.traders.com/-v06-c09-playing-pdf.html"),
    "td_sequential": (td_sequential, "TD Sequential setup 9 and countdown 13 counts (DeMark; no TDST)", "https://www.tradingview.com/scripts/tdsequential/"),
    # scout 15.11.2026 night 3: Ehlers TASC 2025 (ehlers3.py), community classics (classic5.py), structures
    # (structure2.py, incl. 3 TikTok explorer ideas) and time / level tools (levels2.py)
    "ehlers_uo": (ehlers_uo, "Ehlers Ultimate Oscillator 2025: HP(40) - HP(20) of the close / RMS100 (not Williams' UO)", TT + "2025/04/TradersTips.html"),
    "laguerre_filter": (laguerre_filter, "Ehlers Laguerre filter (UltimateSmoother 40, gamma 0.8) + Laguerre oscillator (30, 0.5) / RMS", TT + "2025/07/TradersTips.html"),
    "continuation_index": (continuation_index, "Ehlers Continuation Index: tanh(2 (US20 - Laguerre8 of US40) / mean dev 40), -1..1", TV + "5ZrOut79-TASC-2025-09-The-Continuation-Index/"),
    "one_euro": (one_euro, "One Euro filter (Casiez 2012, Ehlers TASC 2025.12): adaptive EMA, cutoff 10 + 0.2 |change|; oscillator on HP54", TT + "2025/12/TradersTips.html"),
    "pma_predict": (pma_predict, "Ehlers projected MA 20 (SMA + OLS slope x L/2) and its predict line (TASC 2025.03)", TT + "2025/03/TradersTips.html"),
    "reflex_trendflex": (reflex_trendflex, "Ehlers Reflex / Trendflex 20 (TASC 2020.02): zero-lag cycle / trend oscillators", "https://www.prorealcode.com/prorealtime-indicators/reflex-and-trendflex-indicators-john-f-ehlers/"),
    "two_pole_predict": (two_pole_predict, "Ehlers simple 2-pole predictor of SS(HP(close, 15), 30), Q 0.35 (TASC 2025.01)", TT + "2025/01/TradersTips.html"),
    "vix_fix": (vix_fix, "CM Williams Vix Fix 22 with BB 20/2 and 0.85 x 50-bar high alerts; inverse version for tops", TV + "og7JPrRA-CM-Williams-Vix-Fix-Finds-Market-Bottoms/"),
    "bbtrend": (bbtrend, "John Bollinger BBTrend (2024): (|lower20 - lower50| - |upper20 - upper50|) / mid20 x 100", "https://www.tradingview.com/support/solutions/43000726749/"),
    "vzo": (vzo, "Volume Zone Oscillator 14 (Khalil, TASC 2011): 100 EMA(sign x tick volume) / EMA(volume)", TT + "2011/05/Khalil.html"),
    "pzo": (pzo, "Price Zone Oscillator 14 (Khalil): the VZO with the close instead of the volume", TT + "2011/05/Khalil.html"),
    "twiggs_mf": (twiggs_mf, "Twiggs Money Flow 21: true-range close location x volume, Wilder smoothing", "https://www.incrediblecharts.com/indicators/twiggs_money_flow.php"),
    "rmi": (rmi, "Relative Momentum Index 14 / 5 (Altman): RSI of the 5-bar momentum", "https://www.luxalgo.com/library/concept/relative-momentum-index/"),
    "pmo": (pmo, "DecisionPoint Price Momentum Oscillator 35/20 with EMA10 signal (Swenlin)", SC2 + "decisionpoint-price-momentum-oscillator-pmo"),
    "t3": (t3, "Tillson T3 moving average (6 EMAs, volume factor 0.7)", "https://www.luxalgo.com/library/concept/t3/"),
    "alma": (alma, "Arnaud Legoux moving average 9 / 0.85 / 6", "https://www.tradingview.com/support/solutions/43000594683-arnaud-legoux-moving-average/"),
    "ash": (ash, "Absolute Strength Histogram (RSI method) 9 / 3: smoothed bulls vs bears", "https://www.mql5.com/en/code/21429"),
    "trendilo": (trendilo, "Trendilo: ALMA 50 of the % change vs its RMS band, trend +1 / -1 / 0 (dudeowns)", TV + "h5kMWewu-Trendilo-OPEN-SOURCE/"),
    "kairi": (kairi, "Kairi Relative Index: % distance of the close from its SMA25", "https://www.investopedia.com/terms/k/kairi.asp"),
    "fractal_bands": (fractal_bands, "Fractal Chaos Bands: last confirmed 2/2 fractal high / low (2 bars late)", "https://www.tradingview.com/script/U1iz6Mxb-Fractal-Chaos-Bands/"),
    "darvas_box": (darvas_box, "Darvas box: 20-bar high held 3 bars, low held 3 bars; close-through breakout / breakdown", TV + "hZV01YX6-Advanced-Intraday-Darvas-Box/"),
    "vsa": (vsa, "Volume Spread Analysis bars: no demand / no supply / upthrust / spring / stopping volume (tick volume)", TV + "OBV1KONK-No-Supply-No-Demand-NSND-Volume-Spread-Analysis-Tool/"),
    "ross_123": (ross_123, "Joe Ross 1-2-3 reversal (confirmed swings k 3) and the Ross hook after the break of 2", "https://www.luxalgo.com/library/concept/1-2-3-reversal/"),
    "three_line_break": (three_line_break, "Three Line Break chart trend and reversals (Nison), on closes", "https://www.tradingview.com/scripts/threelinebreak/"),
    "point_figure": (point_figure, "Point & Figure 3-box reversal on closes, box 0.5 ATR50; double-top / double-bottom signals", "https://www.luxalgo.com/library/concept/point-and-figure/"),
    "swing_grade": (swing_grade, "Grade 1-5 of the swing a wick sweeps (30-bar extreme, equal highs, prior-day level, age 50+) - TikTok idea", "https://www.tiktok.com/search/video?q=liquidity%20sweep"),
    "gap_type": (gap_type, "FVG vs breakaway gap: candle 3 closes inside / beyond candle 2's range (Adem OnChain, TikTok)", TV + "KmrRBv0v-FVG-Breakaway-3rd-Candle-Arjo-MK/"),
    "valid_ob": (valid_ob, "Valid order block: swept the previous swing + left an FVG + BMS close (ICT & SMC CHARTIST, TikTok)", "https://www.tiktok.com/search/video?q=order%20block"),
    "round_levels": (round_levels, "Psychological round levels: step 10^(digits-2) of the price, plus half steps", "https://www.babypips.com/learn/forex/psychological-level"),
    "adr": (adr, "Average Daily Range 14 (completed UTC days), running day high / low and % of ADR used", "https://www.fxstreet.com/education/how-to-use-the-average-daily-range-when-trading-forex-201610051529"),
    "cbdr": (cbdr, "ICT Central Bank Dealers Range 14:00-20:00 NY: high / low / height for SD projections", TV + "DtiFgMfL-CBDR-Deviations-M1D/"),
    "rvol": (rvol, "Relative volume at time of day: cumulative and bar tick volume vs the same slot's 20-session mean", TV + "8ePzgnzt-Relative-Volume-at-Time/"),
    "ict_macros": (ict_macros, "ICT macro windows (NY): 02:33, 04:03, 08:50, 09:50, 10:50, 11:50, 13:10, 15:15", TV + "aLTywnJ0-ICT-Macros-LuxAlgo/"),
    "po3_4h": (po3_4h, "NY-aligned 4H candle: open, running high / low, previous 4H high / low (TTrades 4H Power of Three)", "https://briefread.pages.dev/en/summary/trading-the-4-hour-power-of"),
    "synthetic_osc": (synthetic_osc, "Ehlers Synthetic Oscillator (TASC 2026.04): cycle-phase sine, 15-25 bar band, Hann-smoothed ROC signals", TT + "2026/04/TradersTips.html"),
    "ultimate_channel": (ultimate_channel, "Ehlers Ultimate Channel (UltimateSmoother of TR) and Ultimate Bands (RMS deviation) around UltimateSmoother 20", TV + "4Plc7Zjk-TASC-2024-05-Ultimate-Channels-and-Ultimate-Bands/"),
    "reversion_index": (reversion_index, "Ehlers Reversion Index (TASC 2026.01): sum of changes / sum of |changes| 20, SuperSmoother 8 / 4 trigger", TV + "V35NeC45-TASC-2026-01-The-Reversion-Index/"),
    "adaptive_supersmoother": (adaptive_supersmoother, "Ehlers Adaptive SuperSmoother (TASC 2026.09): period 20 shrunk by the RMS-scaled ROC, vs the fixed SS 20", TV + "FnlMn99W-TASC-2026-09-Adaptive-SuperSmoother/"),
    "autotune_filter": (autotune_filter, "Ehlers AutoTune Filter (TASC 2026.05): band-pass at 2 x the least-autocorrelated lag of HP 20 (our build of the lag search)", TV + "GqKlw3yv-TASC-2026-05-The-AutoTune-Filter/"),
    "candle_patterns": (candle_patterns, "15 Japanese candle patterns (TA-Lib definitions; stars / soldiers adapted to no-gap feeds): stars, baby, 3 outside / inside, harami, piercing, soldiers, 3-line strike, kicker, tweezer, marubozu, 3 methods, belt hold, engulfing", "https://ta-lib.org/functions/"),
    "divergence": (divergence, "Regular + hidden divergence of 7 oscillators (RSI, MACD hist, CCI, MOM, OBV, MFI, Stoch) at confirmed 5-bar pivots (LonesomeTheBlue)", TV + "n8AGnIZd-Divergence-for-Many-Indicators-v4/"),
    "naked_poc": (naked_poc, "Naked (virgin) session POCs: nearest untouched previous-session POC over / under price, first-touch events", "https://www.luxalgo.com/library/concept/naked-poc-as-level/"),
    "poor_extremes": (poor_extremes, "Market Profile poor highs / lows: session extreme hit by 2+ 30-minute periods within 0.1 ATR, until repaired", "https://www.luxalgo.com/library/concept/poor-high-poor-low/"),
    "single_prints": (single_prints, "Market Profile single-print zones (one 30-minute TPO period, inside the range) of completed sessions, until filled", "https://www.luxalgo.com/library/concept/single-prints/"),
    "td_rei": (td_rei, "DeMark TD Range Expansion Index 5 (-100..100, +/-40 levels) with the bars-beyond-level duration count", "https://linnsoft.com/techind/demark-range-expansion-index"),
    "td_pressure": (td_pressure, "TD Pressure 5 (our build): volume-weighted close-vs-true-range buying share 0..100", "https://www.luxalgo.com/library/concept/td-pressure/"),
    "taylor_cycle": (taylor_cycle, "Taylor 3-day cycle inputs: previous day H/L/C, consecutive down / up days, inside day, today's running H/L", "https://futures.stonex.com/blog/the-taylor-trading-technique"),
    "noise_area": (noise_area, "Zarattini-Aziz-Barbon noise area: open x (1 +/- 14-session mean |move from the open| at the time of day), session VWAP", "https://concretumgroup.com/beat-the-market-an-effective-intraday-momentum-strategy-for-sp500-etf-spy/"),
    "correlation_trend": (correlation_trend, "Ehlers Correlation Trend Indicator (TASC 2020.05): correlation of 20 closes with a rising line, -1..1", TV + "azjhI4tC-Correlation-Trend-Indicator-Dr-John-Ehlers"),
    "ttm_trend": (ttm_trend, "TTM Trend (Carter, our build): close vs the 6-bar average of Heikin-Ashi body midpoints, up / down bars", "https://www.prorealcode.com/prorealtime-indicators/ttm-trend-price/"),
    "ttm_scalper": (ttm_scalper, "TTM Scalper (Carter, our build): 3-bar reversal square - 5-bar swing extreme then two lower (higher) closes", TV + "CddpDBGA-TTM-scalper-indicator-Strategy"),
    "murrey_math": (murrey_math, "Murrey Math lines 0/8-8/8 (+/-2/8) of the 64-bar frame snapped to the price octave (MT4 table, no fine adjustments)", "https://www.luxalgo.com/library/indicator/murrey-math-levels/"),
    "chaikin_volatility": (chaikin_volatility, "Chaikin Volatility: 10-bar % change of EMA10(high - low)", "https://ta-lib.org/functions/cvi.html"),
    "standard_error_bands": (standard_error_bands, "Andersen Standard Error Bands: linreg 21 endpoint +/- 2 standard errors, SMA 3; band width", "https://www.luxalgo.com/library/indicator/standard-error-bands/"),
    "projection": (projection, "Widner Projection Bands (highs / lows projected along their regression slope, 14) + Projection Oscillator with EMA5 trigger", "https://www.luxalgo.com/library/concept/projection-bands/"),
    "wave_pm": (wave_pm, "Whistler Wave PM: tanh(StdDev14 x 1.25 / its 100-bar RMS), 0..1 volatility energy", "https://www.prorealcode.com/prorealtime-indicators/mark-whistlers-wave-pm"),
    "sentiment_zone": (sentiment_zone, "Khalil Sentiment Zone Oscillator: TEMA14 of +/-1 closes, dynamic 30-bar 95% / 5% zones, EMA60 trend", TT + "2012/05/Khalil.html"),
    "rainbow": (rainbow, "Widner Rainbow Oscillator + Rainbow Bandwidth: 10 recursive SMA2 vs the 10-bar close range", "https://www.luxalgo.com/library/concept/rainbow-oscillator/"),
    "madrid_ribbon": (madrid_ribbon, "Madrid MA Ribbon: EMAs 5-90 lime / green / maroon / red vs EMA100, score (lime - red) / 18", "https://in.tradingview.com/script/WH01CtjA-Madrid-MA-Ribbon-Bar"),
    "pitchfork": (pitchfork, "Automatic Andrews' Pitchfork from the last three confirmed alternating 5-bar pivots: median, upper, lower, bull / bear", "https://www.luxalgo.com/library/indicator/andrews-pitchfork/"),
    "kase_devstop": (kase_devstop, "Kase Dev-Stops: 2-bar true range avg + 0 / 1 / 2.2 / 3.6 std under the 20-bar high close (over the low), SMA10/21 trend", "https://www.prorealcode.com/prorealtime-indicators/kase-dev-stop/"),
    "voss_predict": (voss_predict, "Ehlers Voss Predictive Filter (TASC 2019.08): band-pass 20 / 0.25 and the 3-bar Voss predictor, crosses = early turns", TT + "2019/08/TradersTips.html"),
    "super_passband": (super_passband, "Ehlers Super Passband (TASC 2016.07): zero-lag EMA 40 - 60 recursion with its 50-bar RMS band, crosses of -/+RMS", TT + "2016/07/TradersTips.html"),
    "universal_osc": (universal_osc, "Ehlers Universal Oscillator (TASC 2015.01): SuperSmoother 20 of the 2-bar white noise / fast-attack peak, -1..1", TT + "2015/01/TradersTips.html"),
    "elegant_osc": (elegant_osc, "Ehlers Elegant Oscillator (TASC 2022.02): SuperSmoother 20 of tanh(2-bar derivative / RMS 50), -0.5 / +0.5 turns", TT + "2022/02/TradersTips.html"),
    "dsma": (dsma, "Ehlers Deviation-Scaled MA 40 (TASC 2018.07): EMA with alpha = 5 x |RMS-scaled SuperSmoother of the zeros| / period", TT + "2018/07/TradersTips.html"),
    "trama": (trama, "LuxAlgo Trend Regularity Adaptive MA 99: alpha = SMA(new high / new low)^2", TV + "p8wGCPi6-Trend-Regularity-Adaptive-Moving-Average-LuxAlgo"),
    "impulse_macd": (impulse_macd, "LazyBear Impulse MACD 34 / 9: ZLEMA(hlc3) beyond the SMMA high / low band, 0 inside, SMA9 signal", TV + "qt6xLfLi-Impulse-MACD-LazyBear"),
    "mad": (mad, "Ehlers Moving Average Difference (TASC 2021.10): 100 x (SMA8 - SMA23) / SMA23", TT + "2021/10/TradersTips.html"),
    "spearman": (spearman, "Spearman rank correlation of the last 20 closes with time (Ehlers), -1..1", TV + "rpKPpW51-Ehlers-Spearman-Rank-Indicator-CC"),
    "market_thermometer": (market_thermometer, "Elder Market Thermometer: bar extension beyond the prior bar, EMA22, hot (3x) / quiet (5 bars under) states", "https://www.incrediblecharts.com/indicators/elder_market_thermometer.php"),
    "williams_ad": (williams_ad, "Larry Williams Accumulation / Distribution: cumulative close vs true low (up) / true high (down)", "https://www.metastock.com/customer/resources/taaz/?p=125"),
    "fib_bollinger": (fib_bollinger, "Fibonacci Bollinger Bands: VWMA200(hlc3) +/- 0.236 .. 1.0 x 3 stdev (Rashad)", TV + "qIKR3tbN-Fibonacci-Bollinger-Bands"),
    "hull_butterfly": (hull_butterfly, "Hull Butterfly Oscillator (LuxAlgo, our reconstruction): HMA14 kernel minus its mirror, expanding |osc| x 1.5 levels, turn dots", "https://www.luxalgo.com/library/indicator/hull-butterfly-oscillator/"),
    "range_detector": (range_detector, "Range Detector (LuxAlgo, our reconstruction): 20 closes within SMA20 +/- ATR500 -> range top / bottom, close breaks", "https://www.luxalgo.com/library/indicator/range-detector/"),
}

__all__ = ["INDICATORS"] + list(INDICATORS)
