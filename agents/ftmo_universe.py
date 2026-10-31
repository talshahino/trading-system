"""The 94 FTMO instruments the user chose (31.10.2026), where their history comes from, and
FTMO's own cost specs (commission, swap) taken from ftmo.com/en/symbols
(data/ftmo_symbols/responses.json, fetched by .github/workflows/fetch_symbols.yml).

Internal name -> (FTMO code, Dukascopy id or None, news currencies).
Dukascopy id None = history from Alpaca (no bid/ask there, so the spread is estimated).
"""

from __future__ import annotations

import json
from pathlib import Path

FX = {
    "EURUSD": ("EUR/USD", "eurusd", ["EUR", "USD"]), "GBPUSD": ("GBP/USD", "gbpusd", ["GBP", "USD"]),
    "USDJPY": ("USD/JPY", "usdjpy", ["USD", "JPY"]), "USDCHF": ("USD/CHF", "usdchf", ["USD", "CHF"]),
    "USDCAD": ("USD/CAD", "usdcad", ["USD", "CAD"]), "AUDUSD": ("AUD/USD", "audusd", ["AUD", "USD"]),
    "NZDUSD": ("NZD/USD", "nzdusd", ["NZD", "USD"]), "EURJPY": ("EUR/JPY", "eurjpy", ["EUR", "JPY"]),
    "GBPJPY": ("GBP/JPY", "gbpjpy", ["GBP", "JPY"]), "EURGBP": ("EUR/GBP", "eurgbp", ["EUR", "GBP"]),
}
INDICES = {
    "US30": ("US30.cash", "usa30idxusd", ["USD"]), "US100": ("US100.cash", "usatechidxusd", ["USD"]),
    "US500": ("US500.cash", "usa500idxusd", ["USD"]), "US2000": ("US2000.cash", "ussc2000idxusd", ["USD"]),
    "GER40": ("GER40.cash", "deuidxeur", ["EUR"]), "UK100": ("UK100.cash", "gbridxgbp", ["GBP"]),
    "JP225": ("JP225.cash", "jpnidxjpy", ["JPY"]), "FRA40": ("FRA40.cash", "fraidxeur", ["EUR"]),
    "EU50": ("EU50.cash", "eusidxeur", ["EUR"]), "AUS200": ("AUS200.cash", "ausidxaud", ["AUD"]),
    "HK50": ("HK50.cash", "hkgidxhkd", ["HKD", "CNY"]), "SPN35": ("SPN35.cash", "espidxeur", ["EUR"]),
    "N25": ("N25.cash", "nldidxeur", ["EUR"]), "DXY": ("DXY.cash", "dollaridxusd", ["USD"]),
}
METALS = {
    "XAUUSD": ("XAU/USD", "xauusd", ["USD"]), "XAGUSD": ("XAG/USD", "xagusd", ["USD"]),
    "XPTUSD": ("XPT/USD", "xptcmdusd", ["USD"]), "COPPER": ("XCU/USD", "coppercmdusd", ["USD", "CNY"]),
}
ENERGY = {
    "WTI": ("USOIL.cash", "lightcmdusd", ["USD", "CAD"]), "BRENT": ("UKOIL.cash", "brentcmdusd", ["USD"]),
    "NATGAS": ("NATGAS.cash", "gascmdusd", ["USD"]),
}
CRYPTO = {
    "BTCUSD": ("BTCUSD", "btcusd", ["USD"]), "ETHUSD": ("ETHUSD", "ethusd", ["USD"]),
    "SOL/USD": ("SOLUSD", None, ["USD"]), "XRP/USD": ("XRPUSD", None, ["USD"]),
}
_US = ["AAPL", "AMZN", "BA", "BABA", "BAC", "CSCO", "CVX", "DIS", "FDX", "GE", "GM", "GOOG", "IBM", "INTC",
       "JNJ", "JPM", "KO", "MCD", "MSFT", "NFLX", "NVDA", "PFE", "QCOM", "T", "TSLA", "V", "WMT", "XOM", "ZM",
       "RTX", "LMT", "PLTR", "AMD", "AVGO", "SBUX", "NKE", "SNOW", "AZN"]
STOCKS = {s: (s, f"{s.lower()}ususd", ["USD"]) for s in _US}
STOCKS.update({
    "META": ("META", "fbususd", ["USD"]), "BRK.B": ("BRK.B", "brkbususd", ["USD"]),
    "ADSGn": ("ADSGn", "adsdeeur", ["EUR"]), "AIRF": ("AIRF", "airfreur", ["EUR"]),
    "ALVG": ("ALVG", "alvdeeur", ["EUR"]), "BAYGn": ("BAYGn", "bayndeeur", ["EUR"]),
    "DBKGn": ("DBKGn", "dbkdeeur", ["EUR"]), "IBE": ("IBE", "ibeeseur", ["EUR"]),
    "LVMH": ("LVMH", "mcfreur", ["EUR"]), "RACE": ("RACE", "raceiteur", ["EUR"]),
    "SAN": ("SAN", "saneseur", ["EUR"]), "SIEGn": ("SIEGn", "siedeeur", ["EUR"]),
    "VOWG_p": ("VOWG_p", "vow3deeur", ["EUR"]), "ASML": ("ASML", "asmlnleur", ["EUR"]),
    "TTE": ("TTE", "fpfreur", ["EUR"]), "BMW": ("BMW", "bmwdeeur", ["EUR"]),
    "MBG": ("MBG", "daideeur", ["EUR"]),
    # not on Dukascopy: Alpaca history
    "MSTR": ("MSTR", None, ["USD"]), "GME": ("GME", None, ["USD"]), "ARM": ("ARM", None, ["USD"]),
    "SPCX": ("SPCX", None, ["USD"]),
})

GROUPS = {"fx": FX, "index": INDICES, "metal": METALS, "energy": ENERGY, "crypto": CRYPTO, "stock": STOCKS}
ALL = {k: v for g in GROUPS.values() for k, v in g.items()}
SYMBOLS = list(ALL)
CLASS = {k: g for g, d in GROUPS.items() for k in d}
DUKA = {k: v[1] for k, v in ALL.items() if v[1]}
CURRENCIES = {k: v[2] for k, v in ALL.items()}

# value of 1 unit of the base currency in USD, for the forex $-per-lot commission (rough, stable enough)
BASE_USD = {"EUR": 1.1, "GBP": 1.3, "USD": 1.0, "AUD": 0.65, "NZD": 0.6}

_SPEC_FILE = Path(__file__).resolve().parent.parent / "data" / "ftmo_symbols" / "responses.json"
_SPECS = None


def spec(symbol: str) -> dict:
    """FTMO's spec record for an internal symbol name."""
    global _SPECS
    if _SPECS is None:
        d = {c["url"]: c["body"] for c in json.loads(_SPEC_FILE.read_text(encoding="utf-8"))}
        rows = d["https://ftmo.com/wp-json/ftmo/symbols"]["data"]["symbols"]
        _SPECS = {r["code"]: r for r in rows}
    return _SPECS[ALL[symbol][0]]
