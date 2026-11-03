"""Which of bot 2's and bot 3's instruments exist on another MetaTrader server (e.g. MetaQuotes-Demo)?

Read-only: it only lists symbols, it never trades. Run it while a MetaTrader logged in to that demo is open:
    python live/check_symbols.py "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
(check_symbols.bat does this for the standard MetaTrader 5 install.) The result is printed and saved to
live/check_symbols.txt.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import MetaTrader5 as mt5

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agents import ftmo_universe as U  # noqa: E402

# other names brokers use for the same instrument
ALIASES = {
    "US100": ["US100", "NAS100", "USTEC", "NDX100", "NQ100", "NASDAQ100", "USTECH100"],
    "US30": ["US30", "DJ30", "WS30", "DJI30", "DOW30", "USA30"],
    "US500": ["US500", "SPX500", "SP500", "USA500", "SPX"],
    "US2000": ["US2000", "RUSSELL2000", "RUT2000", "RTY", "USSC2000"],
    "JP225": ["JP225", "JPN225", "NIKKEI225", "NIK225", "JAPAN225"],
    "FRA40": ["FRA40", "FRANCE40", "CAC40", "F40"],
    "SPN35": ["SPN35", "SPA35", "ESP35", "SPAIN35", "IBEX35"],
    "NATGAS": ["NATGAS", "NGAS", "XNGUSD", "NG", "NATURALGAS"],
    "WTI": ["USOIL", "WTI", "XTIUSD", "CL", "CRUDEOIL", "OILWTI"],
    "XAUUSD": ["XAUUSD", "GOLD"],
    "XAGUSD": ["XAGUSD", "SILVER"],
    "XRP/USD": ["XRPUSD", "XRPUSDT", "RIPPLE"],
}


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def find(sym, all_syms):
    """Symbols on this server that are the same instrument (exact name, alias, or ticker with a suffix like .NAS / .US)."""
    want = [norm(x) for x in ALIASES.get(sym, [])] + [norm(U.ALL[sym][0]), norm(sym)]
    hits = []
    for s in all_syms:
        n = norm(s.name)
        base = norm(re.split(r"[._#\-]", s.name)[0])
        if n in want or base in want:
            hits.append(s)
    hits.sort(key=lambda s: (norm(s.name) not in want, len(s.name)))
    return hits


def main():
    terminal = sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files\MetaTrader 5\terminal64.exe"
    if not mt5.initialize(path=terminal):
        sys.exit(f"MetaTrader did not answer ({mt5.last_error()}). Open it, log in to the demo account and run again.")
    ai = mt5.account_info()
    all_syms = mt5.symbols_get() or []
    out = [f"server: {ai.server if ai else '?'} | account: {ai.login if ai else '?'} | symbols on the server: {len(all_syms)}", ""]
    for bot in ("bot2", "bot3"):
        cfg = json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
        syms = sorted({c.split("|")[2] for c in cfg["combos"]})
        have = 0
        out.append(f"=== {bot}: {len(cfg['combos'])} combos, {len(syms)} instruments ===")
        for sym in syms:
            hits = find(sym, all_syms)
            if hits:
                have += 1
                h = hits[0]
                tradable = "tradable" if h.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL else f"trade mode {h.trade_mode}"
                out.append(f"  OK   {sym:9} -> {h.name:16} ({h.path}) {tradable}")
            else:
                out.append(f"  --   {sym:9} -> not on this server")
        out.append(f"  {have} of {len(syms)} instruments available")
        out.append("")
    mt5.shutdown()
    text = "\n".join(out)
    print(text)
    (ROOT / "live" / "check_symbols.txt").write_text(text, encoding="utf-8")
    print(f"\nsaved to {ROOT / 'live' / 'check_symbols.txt'}")


if __name__ == "__main__":
    main()
