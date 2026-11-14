"""The TikTok explorer's search list (user, 14.11.2026: "from now on the scout looks at videos by itself - it scans
through the bots' computer thousands of search combinations on the level of strategies, trading, stocks, forex,
currencies, indices, everything; every new strategy or indicator missing from the library that gives new information,
for learning or for building a new strategy").

Thousands of searches = (topic) x (market / instrument) x (style / timeframe), plus every indicator and concept alone.
The explorer (a scheduled task driving Chrome on the bots' computer) takes the next ~20 not-done searches each run,
slowly, and records what it saw in data/agents/scout/tiktok_explore.json.

  python -m agents.tiktok_queries            # writes / extends data/agents/scout/tiktok_explore.json "queue"
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "agents" / "scout" / "tiktok_explore.json"

CONCEPTS = [
    "trading strategy", "scalping strategy", "day trading strategy", "swing trading strategy", "price action strategy",
    "breakout strategy", "pullback strategy", "reversal strategy", "trend following strategy", "mean reversion strategy",
    "opening range breakout", "gap strategy", "fair value gap", "inverse fvg", "order block", "breaker block",
    "liquidity sweep", "stop hunt", "market structure shift", "change of character", "break of structure",
    "smart money concepts", "ict concepts", "silver bullet ict", "judas swing", "power of 3", "candle range theory",
    "smt divergence", "turtle soup", "swing failure pattern", "supply and demand zones", "support resistance flip",
    "trendline break retest", "double top strategy", "head and shoulders trading", "flag pattern trading",
    "wedge breakout", "triangle breakout", "harmonic pattern", "elliott wave trading", "wyckoff accumulation",
    "fibonacci retracement strategy", "fibonacci extension targets", "ote ict", "premium discount ict",
    "asian session range", "london breakout", "new york open strategy", "london reversal", "killzone trading",
    "news trading", "nfp trading", "cpi trading", "fomc trading", "earnings trading", "first candle strategy",
    "inside bar strategy", "engulfing candle strategy", "pin bar strategy", "heikin ashi strategy", "renko strategy",
    "volume profile trading", "vwap strategy", "anchored vwap", "order flow trading", "footprint chart",
    "cumulative delta", "market profile tpo", "point of control", "initial balance", "multi timeframe analysis",
    "risk management trading", "position sizing", "trailing stop strategy", "partial take profit", "break even rule",
    "prop firm strategy", "ftmo challenge strategy", "funded account strategy", "backtested strategy",
    "algorithmic trading strategy", "trading bot strategy", "pine script strategy", "mt5 expert advisor",
    "best tradingview indicator", "free tradingview indicator", "luxalgo indicator", "chartprime indicator",
    "algoalpha indicator", "flux charts indicator", "tradingfinder indicator", "lazybear indicator",
]
INDICATORS = [
    "rsi", "rsi divergence", "macd", "macd histogram", "stochastic", "stochastic rsi", "cci", "williams r", "adx",
    "atr", "bollinger bands", "keltner channel", "donchian channel", "ichimoku", "parabolic sar", "supertrend",
    "ema crossover", "ema 9 21", "ema 20 50", "ema 200", "sma 50 200", "hull moving average", "vwap bands",
    "squeeze momentum", "ttm squeeze", "wavetrend", "qqe", "ut bot", "halftrend", "chandelier exit",
    "ssl channel", "zero lag ema", "kalman filter", "lorentzian classification", "machine learning indicator",
    "nadaraya watson", "gaussian channel", "pivot points", "camarilla pivots", "fibonacci pivots", "obv", "mfi",
    "chaikin money flow", "accumulation distribution", "volume oscillator", "relative volume", "aroon", "vortex",
    "fisher transform", "awesome oscillator", "ultimate oscillator", "coppock curve", "trix", "dmi", "elder ray",
    "market cipher", "volume weighted macd", "smart money indicator", "liquidity indicator", "fvg indicator",
    "order block indicator", "session indicator", "killzone indicator", "trend strength indicator", "regime filter",
]
MARKETS = [
    "forex", "eurusd", "gbpusd", "usdjpy", "gbpjpy", "audusd", "usdcad", "xauusd gold", "silver xagusd", "oil wti",
    "nasdaq nq", "us30 dow", "sp500 es", "dax ger40", "nikkei", "us stocks", "tesla stock", "nvidia stock",
    "crypto", "bitcoin", "ethereum", "futures", "micro futures mnq", "indices",
]
STYLES = ["1 minute", "5 minute", "15 minute", "1 hour", "4 hour", "daily", "scalping", "intraday", "swing"]


def build():
    q = []
    q += CONCEPTS
    q += [f"{i} strategy" for i in INDICATORS]
    q += [f"{c} {m}" for c, m in itertools.product(CONCEPTS, MARKETS)]
    q += [f"{i} {s} strategy" for i, s in itertools.product(INDICATORS, STYLES)]
    q += [f"{c} {s}" for c, s in itertools.product(CONCEPTS[:40], STYLES)]
    q += [f"{i} {m}" for i, m in itertools.product(INDICATORS[:30], MARKETS[:12])]
    seen, out = set(), []
    for x in q:
        k = " ".join(x.lower().split())
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main():
    try:
        st = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        st = {"note": "", "queue": [], "done": {}, "videos": {}, "ideas": [], "runs": [], "finished_at": None}
    st["note"] = ("סייר הטיקטוק (טל, 14.11.2026): חיפושים בטיקטוק דרך הכרום במחשב הבוטים, בקצב איטי. "
                  "queue = החיפושים לפי הסדר, done = חיפוש -> {at, videos, ideas}, videos = מזהה סרטון -> מה נראה בו, "
                  "ideas = רעיונות חדשים לסייר (אסטרטגיה / אינדיקטור / לימוד) שעוד לא בספרייה.")
    have = set(st.get("queue") or [])
    st["queue"] = list(st.get("queue") or []) + [x for x in build() if x not in have]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(st["queue"]), "searches,", len(st.get("done") or {}), "done")


if __name__ == "__main__":
    main()
