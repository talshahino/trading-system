"""Central configuration. Values come from .env (see .env.example) with sane defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

# --- Keys ---
# Keys pasted from a phone often carry hidden characters (spaces, line breaks,
# right-to-left marks from a Hebrew keyboard). Invisible ones are removed; anything
# else that is not plain ASCII is reported by name (never by value).
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c"
                                    "\u202d\u202e\u2066\u2067\u2068\u2069\ufeff\u00a0"))
KEY_PROBLEMS = []


def _key(name):
    raw = os.getenv(name, "")
    v = raw.translate(_INVISIBLE).strip()
    bad = [i for i, ch in enumerate(v) if ord(ch) > 126 or ord(ch) < 33]
    if bad:
        KEY_PROBLEMS.append(
            f"{name}: contains {len(bad)} non-key characters (letters/symbols such as Hebrew "
            f"or spaces) at positions {bad[0] + 1}-{bad[-1] + 1} of {len(v)}. "
            "Paste only the key itself.")
    return v


ALPACA_KEY = _key("ALPACA_API_KEY_ID")
ALPACA_SECRET = _key("ALPACA_API_SECRET_KEY")
ANTHROPIC_KEY = _key("ANTHROPIC_API_KEY")

# --- Data ---
ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_TRADING_URL = os.getenv("ALPACA_TRADING_URL", "https://paper-api.alpaca.markets")
# "sip" = all US exchanges (historical older than 15 min is available on the free plan),
# "iex" = IEX only (thinner, more missing minutes).
BAR_FEED = os.getenv("BAR_FEED", "sip")

# --- Classifier ---
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
# The model may already "know" what happened after news from before this date, which
# inflates results. The untouched test period always starts after it.
# Claude Haiku 5.5: June 2026. Update if you switch models.
MODEL_KNOWLEDGE_CUTOFF = os.getenv("MODEL_KNOWLEDGE_CUTOFF", "2026-06-30")
# USD per million tokens, used for the cost estimate (Haiku 5.5, prompts under 100K tokens).
CLAUDE_INPUT_USD_PER_MTOK = float(os.getenv("CLAUDE_INPUT_USD_PER_MTOK", "0.10"))
CLAUDE_OUTPUT_USD_PER_MTOK = float(os.getenv("CLAUDE_OUTPUT_USD_PER_MTOK", "0.50"))

# --- Backtest assumptions ---
# Seconds between news publication and our order reaching the market (realistic: 30-90s).
ENTRY_LATENCY_SEC = int(os.getenv("ENTRY_LATENCY_SEC", "60"))
# Round-trip cost in basis points (spread + slippage + fees). 10 bps = 0.10%.
# Scaled by liquidity score: thin stocks cost more.
COST_BPS_LIQUID = float(os.getenv("COST_BPS_LIQUID", "5"))       # liquidity score 100
COST_BPS_ILLIQUID = float(os.getenv("COST_BPS_ILLIQUID", "40"))  # liquidity score 1

# --- Position sizing (used to turn % returns into dollars) ---
EQUITY = float(os.getenv("EQUITY", "10000"))
# Size each trade so a typical bad move (2 sigma over the holding time) loses RISK_PCT of equity.
RISK_PCT = float(os.getenv("RISK_PCT", "0.5")) / 100
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "25")) / 100
# Never be more than this share of an average minute's dollar volume.
MAX_PCT_OF_MINUTE_VOLUME = float(os.getenv("MAX_PCT_OF_MINUTE_VOLUME", "5")) / 100
HORIZONS_MIN = [1, 5, 15, 60]

# --- Tracks: separate strategies, each with its own rule, report and (later) account ---
TRACKS = {
    "intraday": {"label": "יומי (דקות עד שעה)", "unit": "m", "horizons": HORIZONS_MIN,
                 "default_h": 15, "unit_he": "דק׳"},
    "swing":    {"label": "בינוני (עד שבוע)", "unit": "d", "horizons": [1, 2, 3, 5],
                 "default_h": 3, "unit_he": "ימים"},
    "position": {"label": "דו־שבועי", "unit": "d", "horizons": [7, 10],
                 "default_h": 10, "unit_he": "ימים"},
}
# Multi-day tracks are measured against the market (SPY), as if each trade were hedged
# with an opposite SPY position. This is the cost of that hedge, in bps.
HEDGE_COST_BPS = float(os.getenv("HEDGE_COST_BPS", "2"))
MARKET_SYMBOL = os.getenv("MARKET_SYMBOL", "SPY")
# News tagged with many tickers is usually a roundup, not a stock-specific event.
MAX_SYMBOLS_PER_NEWS = int(os.getenv("MAX_SYMBOLS_PER_NEWS", "2"))
# Only trade during regular hours (09:30-16:00 New York).
RTH_ONLY = os.getenv("RTH_ONLY", "1") == "1"
# Share of the period (by time) used to pick rules; the rest is the untouched test.
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.7"))
MIN_TRADES_FOR_RULE = int(os.getenv("MIN_TRADES_FOR_RULE", "30"))
