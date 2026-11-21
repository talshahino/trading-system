"""Bot 9 - the pool (user, 21.11.2026: "why not really run another virtual bot with these strategies and see whether
their success is better than all the rest?"). Virtual, no money.

120 scout strategies x instruments that passed the basic bar in the bots' tests (regular / strong, earning on both
30.04-31.12.2025 and 2026) but are in no bot - mostly declined because the bot was crowded or they added nothing to
it - at most 3 per instrument, chosen on 2025 (live/bots/pool.json, "scout"). Each runs exactly as bot 5ב runs its
scout strategies (live/watch_own.scout5_trades: the strategy's timeframe, FTMO's costs with the live spread, the
calendar, out before a 2+ hour break - a funded account), on bot 1's MetaTrader candles.

Its own account: 0.4% of the balance per instrument, split between that instrument's strategies; a new trade risks at
most the room under 4.5% a day / 9% in all (open risk counted), under a quarter of its risk left -> not taken; a trade
first seen more than an hour after its fill is not booked (no hindsight). State: live/pool_bot_state.json + branch live.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

LIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(LIVE))
import reporter as REP  # noqa: E402
import watch_own as W  # noqa: E402

NAME = "pool_bot"
STATE_F = LIVE / f"{NAME}_state.json"
CONF_F = LIVE / "bots" / "pool.json"
LATE_MIN = 60
REP.ACCOUNT[0] = NAME


def log(m):
    try:
        with open(LIVE / f"{NAME}.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
    except Exception:  # noqa: BLE001
        pass


def rd(p, d=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return d


def pday(ts):
    return pd.Timestamp(ts).tz_convert("Europe/Prague").date().isoformat()


def main():
    port = os.environ.get("SOURCE_PORT")
    if not port:
        log("no SOURCE_PORT (bot 1's local port): the manager starts this bot")
        return
    src, cal = W.Source(port), W.Calendar()
    conf = rd(CONF_F, {}) or {}
    st = rd(STATE_F, {}) or {}
    now = pd.Timestamp.now(tz="UTC")
    st.setdefault("started", now.isoformat(timespec="seconds"))
    st.setdefault("initial", float(conf.get("initial", 100000.0)))
    st.setdefault("book", {})
    st.setdefault("equity_curve", [])
    st.setdefault("tally", {})
    log(f"started; {len(conf.get('scout') or {})} strategies, {len(st['book'])} booked")
    last_push, missing = 0.0, set()
    nb_cal = None                       # bot 7's calendar, for the strategies that came from bot 7 with a release guard
    while True:
        t_loop = time.time()
        try:
            conf = rd(CONF_F, conf) or conf
            scouts = conf.get("scout") or {}
            by_sym = {}
            for combo, v in scouts.items():
                sid, tf, sym = combo.split("|")
                by_sym.setdefault(sym, []).append({"sid": sid, "tf": tf, "since": v.get("since") or st["started"],
                                                   "card": v.get("card"), "plan": v.get("plan")})
            if any(v.get("plan", {}).get("release_guard") for v in scouts.values() if isinstance(v.get("plan"), dict)):
                try:
                    if nb_cal is None:
                        import news_bot as NB
                        nb_cal = NB.Calendar()
                    nb_cal.refresh()
                except Exception as e:  # noqa: BLE001
                    log(f"bot 7's calendar not read ({str(e)[-200:]})")
            for s in by_sym:
                W.SYMBOLS.setdefault(s, [s, f"{s}.cash"])
            try:
                src.spreads(list(by_sym))
            except Exception:  # noqa: BLE001
                pass
            book = st["book"]
            now = pd.Timestamp.now(tz="UTC")
            today = pday(now)
            if st.get("day") != today:
                st["day"], st["day_start"] = today, balance(st)
            seen_open = {}
            for sym, scs in by_sym.items():
                try:
                    df = src.update(sym)
                except Exception as e:  # noqa: BLE001
                    if sym not in missing:
                        log(f"{sym}: no candles ({e})")
                        missing.add(sym)
                    continue
                if df is None or not len(df):
                    if sym not in missing:
                        log(f"{sym}: no candles at bot 1's MetaTrader")
                        missing.add(sym)
                    continue
                per = float(conf.get("risk_per_symbol", 0.004)) / len(scs)
                for sc in scs:
                    try:
                        hi = src.h1(sym) if sc["tf"] in ("1Hour", "4Hour") else None
                        recs = W.scout5_trades(sym, df, sc, cal, hi=hi)
                    except Exception as e:  # noqa: BLE001
                        log(f"{sym} {sc['sid']} {sc['tf']}: {str(e)[-200:]}")
                        continue
                    for rec in recs:
                        key = f"{sym}|{rec['tal_key']}"
                        b = book.get(key)
                        if b is None:
                            b = {"stream": f"{sc['sid']}|{sc['tf']}|{sym}", "symbol": sym, "side": "buy" if rec["dir"] > 0 else "sell",
                                 "open_t": pd.Timestamp(rec["time"]).isoformat(), "open_px": rec["entry"], "sl0": rec["stop"],
                                 "tp0": rec["target"], "seen": now.isoformat(timespec="seconds"), "status": "open", "pnl": 0.0}
                            if now - pd.Timestamp(rec["time"]) > pd.Timedelta(minutes=LATE_MIN):
                                b.update(status="skipped", risk_usd=0.0, why="late")
                            else:
                                bal = balance(st)
                                # fixed risk on the initial amount (Tal, 21.11.2026: in the replay compounding earned less
                                # here - the $5,000 daily cap stops a bigger account sooner); "compound": true -> the balance
                                want = (bal if conf.get("compound") else st["initial"]) * per
                                open_risk = sum(x["risk_usd"] for x in book.values() if x["status"] == "open")
                                day_pnl = sum(x["pnl"] for x in book.values() if x["status"] == "closed" and x.get("day") == today)
                                room = min(conf.get("day_limit", 0.045) * st["initial"] + day_pnl - open_risk,
                                           bal - open_risk - st["initial"] * (1 - conf.get("total_limit", 0.09)))
                                if room < 0.25 * want:
                                    b.update(status="skipped", risk_usd=0.0, why="no room")
                                else:
                                    b["risk_usd"] = round(min(want, room), 2)
                                    REP.notify(f"בוט 9 · מאגר: {sym} עסקה וירטואלית נפתחה",
                                               f"{'קנייה' if rec['dir'] > 0 else 'מכירה'} · {sc['sid']} {sc['tf']} · "
                                               f"{rec['entry']} · סטופ {rec['stop']} · יעד {rec['target']}", priority=3)
                            book[key] = b
                        if b["status"] != "open":
                            continue
                        if rec["status"] == "closed":
                            r = float(rec["r"])
                            pnl = round(r * b["risk_usd"], 2)
                            b.update(status="closed", r=round(r, 4), pnl=pnl, close_t=pd.Timestamp(rec["exit_time"]).isoformat(),
                                     close_px=rec["exit"], why=rec["reason"], day=today)
                            x = st["tally"].setdefault(b["stream"], {"n": 0, "sum_r": 0.0, "wins": 0, "usd": 0.0})
                            x["n"] += 1
                            x["sum_r"] = round(x["sum_r"] + r, 4)
                            x["wins"] += r > 0
                            x["usd"] = round(x["usd"] + pnl, 2)
                            REP.notify(f"בוט 9 · מאגר: {sym} נסגרה", f"{b['stream']} · {r:+.2f}R · {pnl:+,.0f}$", priority=3)
                        else:
                            seen_open[key] = rec
            # an open trade its strategy no longer shows (the data reloaded): closed at 0 after a day
            for key, b in book.items():
                if b["status"] == "open" and key not in seen_open and now - pd.Timestamp(b["seen"]) > pd.Timedelta(days=1):
                    b.update(status="closed", r=0.0, pnl=0.0, why="לא נראתה עוד", day=today)
            bal = balance(st)
            unreal = sum(b["risk_usd"] * float(seen_open[k].get("r_now") or 0.0) for k, b in book.items()
                         if b["status"] == "open" and k in seen_open)
            eq = round(bal + unreal, 2)
            if not st["equity_curve"] or now - pd.Timestamp(st["equity_curve"][-1][0]) > pd.Timedelta(minutes=3):
                st["equity_curve"] = (st["equity_curve"] + [[now.isoformat(timespec="seconds"), bal, eq]])[-2000:]
            cut = (now - pd.Timedelta(days=21)).isoformat()
            st["book"] = book = {k: b for k, b in book.items() if b["status"] == "open" or (b.get("close_t") or b["seen"]) >= cut}
            out = view(st, conf, seen_open, now, bal, eq)
            STATE_F.write_text(json.dumps(out | {"book": book, "day": st["day"], "day_start": st["day_start"],
                                                 "tally": st["tally"], "equity_curve": st["equity_curve"]},
                                          ensure_ascii=False, default=str), encoding="utf-8")
            if time.time() - last_push > 60:
                last_push = time.time()
                err = REP.push_state(NAME, dict(out, tally=st["tally"]))
                if err:
                    log(err)
        except Exception:  # noqa: BLE001
            log(traceback.format_exc()[-800:])
        time.sleep(max(20, 60 - (time.time() - t_loop)))


def balance(st):
    return round(st["initial"] + sum(b["pnl"] for b in st["book"].values() if b["status"] == "closed")
                 + st.get("archived_pnl", 0.0), 2)


def view(st, conf, seen_open, now, bal, eq):
    book = st["book"]
    pos, closed = [], []
    for k, b in book.items():
        if b["status"] == "open" and k in seen_open:
            rec = seen_open[k]
            pos.append({"ticket": k, "side": b["side"], "dir": "קנייה" if b["side"] == "buy" else "מכירה", "combo": b["stream"],
                        "symbol": b["symbol"], "lots": "וירטואלי", "open": b["open_px"], "now": rec.get("px_now"),
                        "sl": rec.get("stop_now") or b["sl0"], "tp": b["tp0"], "sl0": b["sl0"], "tp0": b["tp0"],
                        "sl_usd": -b["risk_usd"], "tp_usd": None, "pnl": round(b["risk_usd"] * float(rec.get("r_now") or 0), 2),
                        "since": b["open_t"], "pip": 1.0, "free": False})
        elif b["status"] == "closed":
            closed.append({"id": k, "combo": b["stream"], "symbol": b["symbol"], "side": b["side"], "lots": "וירטואלי",
                           "open_t": b["open_t"], "open_px": b["open_px"], "close_t": b.get("close_t"), "close_px": b.get("close_px"),
                           "sl0": b["sl0"], "tp0": b["tp0"], "sl": b["sl0"], "tp": b["tp0"], "gross": b["pnl"], "swap": 0,
                           "comm": 0, "net": b["pnl"], "r_net": b.get("r"), "why": b.get("why")})
    closed.sort(key=lambda x: str(x.get("close_t") or ""))
    return {"bot": NAME, "label": conf.get("label", "בוט 9 · מאגר (וירטואלי)"), "watch_only": True, "rules": conf.get("note", ""),
            "account": "וירטואלי · בוט 9", "server": "צפייה בלבד · בלי כסף", "initial": st["initial"],
            "risk": conf.get("risk_per_symbol"), "balance": bal, "equity": eq, "day_start": st.get("day_start"),
            "started": st["started"], "updated_utc": now.isoformat(timespec="seconds"), "positions": pos,
            "closed": closed[-300:], "equity_curve": st["equity_curve"][-1000:],
            "skipped": sum(1 for b in book.values() if b["status"] == "skipped"),
            "vacct": {"initial": st["initial"], "started": st["started"]}}


if __name__ == "__main__":
    main()
