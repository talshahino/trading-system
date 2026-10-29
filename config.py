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

# --- Time budget ---
# GitHub stops a job after 6 hours, possibly before the data cache is saved. The run
# therefore stops itself cleanly before that, keeping everything downloaded so far;
# the next run continues from there.
import time as _time
RUN_STARTED = _time.time()
MAX_RUN_MINUTES = float(os.getenv("MAX_RUN_MINUTES", "0"))  # 0 = no limit (local runs)


class TimeBudgetReached(Exception):
    pass


def check_time_budget():
    if MAX_RUN_MINUTES and (_time.time() - RUN_STARTED) / 60 > MAX_RUN_MINUTES:
        raise TimeBudgetReached()


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
# Batch mode for historical runs: many headlines in one job, half the price, usually done
# within an hour. Live trading always uses single calls.
USE_BATCH = os.getenv("USE_BATCH", "1") == "1"
BATCH_MIN = int(os.getenv("BATCH_MIN", "200"))           # smaller jobs go one by one
BATCH_CHUNK = int(os.getenv("BATCH_CHUNK", "10000"))     # headlines per batch
BATCH_POLL_SEC = int(os.getenv("BATCH_POLL_SEC", "60"))
BATCH_MAX_WAIT_MIN = int(os.getenv("BATCH_MAX_WAIT_MIN", "240"))

# --- Backtest assumptions ---
# Seconds between news publication and our order reaching the market (realistic: 30-90s).
ENTRY_LATENCY_SEC = int(os.getenv("ENTRY_LATENCY_SEC", "60"))
# Round-trip cost in basis points (spread + slippage + fees). 10 bps = 0.10%.
# Scaled by liquidity score: thin stocks cost more.
COST_BPS_LIQUID = float(os.getenv("COST_BPS_LIQUID", "5"))       # liquidity score 100
COST_BPS_ILLIQUID = float(os.getenv("COST_BPS_ILLIQUID", "40"))  # liquidity score 1

# --- Position sizing (used to turn % returns into dollars) ---
EQUITY = float(os.getenv("EQUITY", "10000"))    # demo account starts at $10,000

# --- Risk rules (the user's decisions, see the spec) ---
# Daily loss limit: trading stops for the rest of the day once the day's loss reaches it.
DAILY_LOSS_PCT = float(os.getenv("DAILY_LOSS_PCT", "2")) / 100
# Max loss measured from the STARTING balance: at 9,000 of 10,000 everything stops
# (demo too) for a system review.
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "10")) / 100
# Risk scales with the balance, but less than proportionally (square root), both ways:
#   scale = (equity / starting equity) ** RISK_SCALING_POWER, kept between the two bounds.
#   +21% balance -> x1.10 (daily limit 2.2%); 9,000 (the stop line) -> x0.95 (1.9%).
RISK_SCALING_POWER = float(os.getenv("RISK_SCALING_POWER", "0.5"))
RISK_SCALE_MIN = float(os.getenv("RISK_SCALE_MIN", "0.5"))
RISK_SCALE_MAX = float(os.getenv("RISK_SCALE_MAX", "1.5"))
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
# Smart exits. Besides a plain time exit, every trade is also replayed with a stop-loss and
# a profit target sized to the stock's own volatility over the holding period:
#   stop   = k_stop   x sigma,  target = k_target x sigma,  sigma = daily vol x sqrt(days held)
# The trade exits at whichever is hit first; otherwise at the end of the holding period.
# Same bar touches both -> counted as a stop (conservative). A gap through the stop fills
# at the open, i.e. worse than the stop.
#   trailing: the stop follows the best price reached (locks in gains), no fixed target.
EXIT_MODES = {
    "time":    None,                # reference only: exit at the end of the holding period
    "st1x2":   (1.0, 2.0, False),   # tight stop, target twice as far
    "st15x3":  (1.5, 3.0, False),   # wider stop, target twice as far
    "trail1":  (1.0, None, True),   # trailing stop 1 sigma behind the best price
    "trail15": (1.5, None, True),   # trailing stop 1.5 sigma behind the best price
}
EXIT_LABELS_HE = {"time": "יציאה בסוף הזמן (להשוואה בלבד)", "st1x2": "סטופ 1σ, יעד 2σ",
                  "st15x3": "סטופ 1.5σ, יעד 3σ", "trail1": "סטופ נגרר 1σ", "trail15": "סטופ נגרר 1.5σ"}
# The user's rule: exits must be smart. The plain time exit is shown for comparison but
# is never chosen as the trading rule unless this is set to 1.
ALLOW_TIME_EXIT = os.getenv("ALLOW_TIME_EXIT", "0") == "1"

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
