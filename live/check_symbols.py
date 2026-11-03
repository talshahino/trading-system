"""Which of bot 2's and bot 3's instruments exist on another MetaTrader server (e.g. MetaQuotes-Demo)?

Read-only: it only lists symbols, it never trades. Run it while a MetaTrader logged in to that demo is open:
    python live/check_symbols.py "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
(check_symbols.bat does this for the standard MetaTrader 5 install.) The result is printed and saved to
live/check_symbols.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import MetaTrader5 as mt5

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agents import ftmo_universe as U  # noqa: E402

sys.path.insert(0, str(ROOT / "live"))
import symbols as SY  # noqa: E402


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
            h = SY.resolve(sym, U.ALL[sym][0], all_syms)
            if h:
                have += 1
                out.append(f"  OK   {sym:9} -> {h.name:16} ({h.path})")
            else:
                d = SY.resolve(sym, U.ALL[sym][0], all_syms, tradable_only=False)
                out.append(f"  --   {sym:9} -> " + (f"{d.name} exists but trading is disabled ({d.path})" if d else "not on this server"))
        out.append(f"  {have} of {len(syms)} instruments can be traded")
        out.append("")
    mt5.shutdown()
    text = "\n".join(out)
    print(text)
    (ROOT / "live" / "check_symbols.txt").write_text(text, encoding="utf-8")
    print(f"\nsaved to {ROOT / 'live' / 'check_symbols.txt'}")


if __name__ == "__main__":
    main()
