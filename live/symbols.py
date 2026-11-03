"""Find the bot's instruments on any MetaTrader server.

The bots know every instrument by FTMO's name (US100.cash, XAU/USD, JP225.cash ...). Other servers name
them differently (USTEC, XAUUSD, JPN225 ...), or have a share with the same ticker (GOLD = Barrick Gold,
CL = Colgate). Rules:
- an index / commodity / metal / currency / crypto only matches a symbol that is NOT a share;
- a share only matches a share with the same ticker (an exchange suffix like .NAS / .US is fine);
- a symbol that exists but cannot be traded (trade mode "disabled") does not count.
"""
from __future__ import annotations

import re

ALIASES = {
    "US100": ["US100", "NAS100", "USTEC", "NDX100", "NQ100", "NASDAQ100", "USTECH100"],
    "US30": ["US30", "DJ30", "WS30", "DJI30", "DOW30", "USA30"],
    "US500": ["US500", "SPX500", "SP500", "USA500", "SPX"],
    "US2000": ["US2000", "RUSSELL2000", "RUT2000", "RTY", "USSC2000"],
    "JP225": ["JP225", "JPN225", "NIKKEI225", "NIK225", "JAPAN225"],
    "FRA40": ["FRA40", "FRANCE40", "CAC40", "F40"],
    "SPN35": ["SPN35", "SPA35", "ESP35", "SPAIN35", "IBEX35"],
    "GER40": ["GER40", "DE40", "DAX40", "GER30", "DAX"],
    "UK100": ["UK100", "FTSE100", "GB100"],
    "NATGAS": ["NATGAS", "NGAS", "XNGUSD", "NATURALGAS"],
    "WTI": ["USOIL", "WTI", "XTIUSD", "CRUDEOIL", "OILWTI", "WTIUSD"],
    "BRENT": ["UKOIL", "BRENT", "XBRUSD"],
    "XAUUSD": ["XAUUSD", "GOLD"],
    "XAGUSD": ["XAGUSD", "SILVER"],
    "XRP/USD": ["XRPUSD", "XRPUSDT", "RIPPLE"],
}
SHARE = re.compile(r"stock|share|equit|nasdaq|nyse|xetra", re.I)
CCY = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "PLN", "HUF", "CZK", "MXN", "ZAR", "TRY", "SGD", "HKD", "CNH", "ILS"}


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def is_share_symbol(s):
    return bool(SHARE.search(getattr(s, "path", "") or ""))


def kind(sym, ftmo_name):
    n = norm(ftmo_name)
    if sym in ALIASES or "/" in ftmo_name or ".cash" in ftmo_name.lower():
        return "other"
    if len(n) == 6 and n[:3] in CCY and n[3:] in CCY:
        return "other"
    return "share"


def resolve(sym, ftmo_name, all_syms, tradable_only=True):
    """The symbol object on this server for the bot's instrument, or None."""
    k = kind(sym, ftmo_name)
    exact = norm(ftmo_name)
    want = [exact, norm(sym)] + [norm(x) for x in ALIASES.get(sym, [])]
    hits = []
    for s in all_syms:
        if tradable_only and getattr(s, "trade_mode", 4) == 0:          # SYMBOL_TRADE_MODE_DISABLED
            continue
        share = is_share_symbol(s)
        n, base = norm(s.name), norm(re.split(r"[._#\-]", s.name)[0])
        if k == "other":
            if share:
                continue
            if n in want or base in want:
                hits.append(s)
        else:
            if base == norm(sym) or n == exact:
                hits.append(s)
    hits.sort(key=lambda s: (norm(s.name) != exact, norm(s.name) not in want, len(s.name)))
    return hits[0] if hits else None
