"""Bot 6 - bot 5's gap method on an FTMO Swing account (user, 21.11.2026 22:28: "an excellent idea" - bot 6 no longer
trades futures: "give it a new role - test bot 5's method on a Swing account, with no close before the weekend and no
stop around the news"). Virtual, no money.

Exactly bot 5's gap tracks as they run now, read from live/bots/bot5.json on every loop (live/watch_own.load_settings:
the tracks, each instrument's own stop / target "sym_cfg", the tracks taken off, the risk per track / instrument) - so
every change made to bot 5 reaches bot 6 at once. The only differences are FTMO Swing's rules:
  * nothing is closed or cancelled before a market break of 2+ hours (the weekend, a daily close, a holiday): WKND off,
    BRKH off, no pre-break window, no forced exit at a gap in the data
  * no news restriction: no "no entry / out" window around FTMO's restricted events (the method's own news guards -
    PROT cal15 / NEWSX - stay, they are part of the method, not FTMO's rule)
  * FTMO's swap is charged for every night held (watch_own.swap_r), as on the account
Its own account: $100,000, the risk of each track / instrument as bot 5, from the balance (compounding), a new trade
risks at most the room under 4.5% a day (5% of the initial amount below the midnight balance) / 9% in all, under a
quarter of its risk left -> not taken; a trade first seen more than an hour after its fill is not booked. Leverage
1:30 / 1:15 is not modelled (bot 5's indices, metals and FX use well under it - the margin check of 21.11.2026).
State: live/watch_fut_state.json + branch live (the app's bot 6 card).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(LIVE))
import reporter as REP  # noqa: E402
import watch_own as W  # noqa: E402

TG = W.TG
NAME = "watch_fut"                    # the app's bot 6 card
STATE_F = LIVE / "swing_bot_state.json"
OUT_F = LIVE / f"{NAME}_state.json"
INITIAL, DAY_LIMIT, TOTAL_LIMIT, LATE_MIN = 100_000.0, 0.045, 0.09, 60
REP.ACCOUNT[0] = NAME


def log(m):
    try:
        with open(LIVE / "swing_bot.log", "a", encoding="utf-8") as f:
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


def swing_cfg(cfg):
    c = dict(cfg)
    c["WKND"] = 0                     # no close / cancel before a break (TG._norm drops WKND 0)
    c.pop("BRKH", None)
    c.pop("REARM", None)
    return TG._norm(c)


def balance(st):
    return round(INITIAL + sum(b["pnl"] for b in st["book"].values() if b["status"] == "closed") + st.get("archived_pnl", 0.0), 2)


def main():
    port = os.environ.get("SOURCE_PORT")
    if not port:
        log("no SOURCE_PORT (bot 1's local port): the manager starts this bot")
        return
    src, cal = W.Source(port), W.Calendar()
    st = rd(STATE_F, {}) or {}
    now = pd.Timestamp.now(tz="UTC")
    st.setdefault("started", now.isoformat(timespec="seconds"))
    st.setdefault("book", {})
    st.setdefault("equity_curve", [])
    st.setdefault("tally", {})
    log(f"started; {len(st['book'])} booked")
    last_push = 0.0
    while True:
        t_loop = time.time()
        try:
            W.load_settings()                                   # bot 5's tracks, as bot 5 runs them now
            try:
                cal.refresh()
            except Exception:  # noqa: BLE001
                pass
            syms = sorted(set(W.TAL5_SYMBOLS) | {s for v in W.TAL5_SYMS.values() for s in v})
            try:
                src.spreads(syms)
            except Exception:  # noqa: BLE001
                pass
            book = st["book"]
            now = pd.Timestamp.now(tz="UTC")
            today = pday(now)
            if st.get("day") != today:
                st["day"], st["day_start"] = today, balance(st)
            seen_open = {}
            for name in syms:
                try:
                    df = src.update(name)
                except Exception as e:  # noqa: BLE001
                    log(f"{name}: no candles ({str(e)[-150:]})")
                    continue
                if df is None or len(df) < 3000:
                    continue
                n = len(df)
                cst = W.cost_of(name, df)
                try:
                    calm = cal.cal_masks(df, name)
                except Exception:  # noqa: BLE001
                    calm = {}
                D = TG.Data(name, df[["o", "h", "l", "c"]], np.asarray(cst, float), np.zeros(n, bool), np.zeros(n, bool),
                            TG.stop_buffer(name), calm)
                for trk, (pair, cfg) in W.TAL5.items():
                    if name not in (W.TAL5_SYMS.get(trk) or W.TAL5_SYMBOLS) or trk in W.TAL5_OFF:
                        continue
                    cfg = dict(cfg, **((W.TAL5_SYMCFG.get(trk) or {}).get(name) or {}))
                    try:
                        recs = TG.live_trades(D, pair, swing_cfg(cfg))
                    except Exception as e:  # noqa: BLE001
                        log(f"{name} {trk}: {str(e)[-200:]}")
                        continue
                    for x in recs:
                        if x["status"] not in ("open", "closed") or not x.get("time"):
                            continue
                        key = f"{name}|{x['id_key']}|{trk}"
                        r_x, rn_x = x.get("r"), x.get("r_now")
                        w_ = W.swap_r(name, x["dir"], x["entry"], x.get("stop"), x["time"], x.get("exit_time"))
                        if w_:
                            r_x = None if r_x is None else float(r_x) + w_
                            rn_x = None if rn_x is None else float(rn_x) + w_
                        b = book.get(key)
                        if b is None:
                            b = {"stream": f"gap:{trk}#{name}", "track": trk, "symbol": name,
                                 "side": "buy" if x["dir"] > 0 else "sell", "open_t": pd.Timestamp(x["time"]).isoformat(),
                                 "open_px": x["entry"], "sl0": x.get("stop"), "tp0": x.get("target"),
                                 "seen": now.isoformat(timespec="seconds"), "status": "open", "pnl": 0.0}
                            if now - pd.Timestamp(x["time"]) > pd.Timedelta(minutes=LATE_MIN):
                                b.update(status="skipped", risk_usd=0.0, why="late")
                            else:
                                bal = balance(st)
                                rk = W.TAL5_RISK.get(f"{trk}|{name}", W.TAL5_RISK.get(trk, 0.002)) * W.TAL5_SCALE[0]
                                want = bal * rk
                                open_risk = sum(z["risk_usd"] for z in book.values() if z["status"] == "open")
                                day_pnl = sum(z["pnl"] for z in book.values() if z["status"] == "closed" and z.get("day") == today)
                                room = min(DAY_LIMIT * INITIAL + day_pnl - open_risk,
                                           bal - open_risk - INITIAL * (1 - TOTAL_LIMIT))
                                if room < 0.25 * want or want <= 0:
                                    b.update(status="skipped", risk_usd=0.0, why="no room")
                                else:
                                    b["risk_usd"] = round(min(want, room), 2)
                                    REP.notify(f"בוט 6 · Swing: {name} עסקה וירטואלית נפתחה",
                                               f"{'קנייה' if x['dir'] > 0 else 'מכירה'} · {trk} · {x['entry']} · סטופ {x.get('stop')}"
                                               f" · יעד {x.get('target')}", priority=3)
                            book[key] = b
                        if b["status"] != "open":
                            continue
                        if x["status"] == "closed" and r_x is not None:
                            r = float(r_x)
                            pnl = round(r * b["risk_usd"], 2)
                            b.update(status="closed", r=round(r, 4), pnl=pnl, close_t=pd.Timestamp(x["exit_time"]).isoformat(),
                                     close_px=x.get("exit"), why=x.get("reason"), day=today)
                            y = st["tally"].setdefault(b["stream"], {"n": 0, "sum_r": 0.0, "wins": 0, "usd": 0.0})
                            y["n"] += 1
                            y["sum_r"] = round(y["sum_r"] + r, 4)
                            y["wins"] += r > 0
                            y["usd"] = round(y["usd"] + pnl, 2)
                            REP.notify(f"בוט 6 · Swing: {name} נסגרה", f"{trk} · {r:+.2f}R · {pnl:+,.0f}$", priority=3)
                        else:
                            seen_open[key] = dict(x, r_now=rn_x, px_now=float(df["c"].iloc[-1]))
            # an open trade its track no longer shows (the data window moved past its setup) for a day: closed at its
            # last known result (r_now), not at 0
            for key, b in book.items():
                if b["status"] == "open" and key not in seen_open and now - pd.Timestamp(b["seen"]) > pd.Timedelta(days=1):
                    r = float(b.get("last_r") or 0.0)
                    b.update(status="closed", r=r, pnl=round(r * b["risk_usd"], 2), why="לא נראתה עוד", day=today,
                             close_t=now.isoformat(timespec="seconds"))
            for key, x in seen_open.items():
                book[key]["seen"] = now.isoformat(timespec="seconds")
                book[key]["last_r"] = x.get("r_now")
            bal = balance(st)
            unreal = sum(b["risk_usd"] * float(seen_open[k].get("r_now") or 0.0) for k, b in book.items()
                         if b["status"] == "open" and k in seen_open)
            eq = round(bal + unreal, 2)
            if not st["equity_curve"] or now - pd.Timestamp(st["equity_curve"][-1][0]) > pd.Timedelta(minutes=3):
                st["equity_curve"] = (st["equity_curve"] + [[now.isoformat(timespec="seconds"), bal, eq]])[-2000:]
            cut = (now - pd.Timedelta(days=30)).isoformat()
            keep = {k: b for k, b in book.items() if b["status"] == "open" or (b.get("close_t") or b["seen"]) >= cut}
            st["archived_pnl"] = round(st.get("archived_pnl", 0.0) + sum(b["pnl"] for k, b in book.items()
                                                                          if k not in keep and b["status"] == "closed"), 2)
            st["book"] = book = keep
            out = view(st, seen_open, now, bal, eq)
            STATE_F.write_text(json.dumps(st, ensure_ascii=False, default=str), encoding="utf-8")
            OUT_F.write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
            if time.time() - last_push > 60:
                last_push = time.time()
                err = REP.push_state(NAME, out)
                if err:
                    log(err)
        except Exception:  # noqa: BLE001
            log(traceback.format_exc()[-800:])
        time.sleep(max(20, 60 - (time.time() - t_loop)))


def view(st, seen_open, now, bal, eq):
    book = st["book"]
    pos, closed = [], []
    for k, b in book.items():
        if b["status"] == "open" and k in seen_open:
            x = seen_open[k]
            pos.append({"ticket": k, "side": b["side"], "dir": "קנייה" if b["side"] == "buy" else "מכירה", "combo": b["stream"],
                        "symbol": b["symbol"], "lots": "וירטואלי", "open": b["open_px"], "now": x.get("px_now"),
                        "sl": x.get("stop") or b["sl0"], "tp": b["tp0"], "sl0": b["sl0"], "tp0": b["tp0"],
                        "sl_usd": -b["risk_usd"], "tp_usd": None, "pnl": round(b["risk_usd"] * float(x.get("r_now") or 0), 2),
                        "since": b["open_t"], "pip": 1.0, "free": False})
        elif b["status"] == "closed":
            closed.append({"id": k, "combo": b["stream"], "symbol": b["symbol"], "side": b["side"], "lots": "וירטואלי",
                           "open_t": b["open_t"], "open_px": b["open_px"], "close_t": b.get("close_t"), "close_px": b.get("close_px"),
                           "sl0": b["sl0"], "tp0": b["tp0"], "sl": b["sl0"], "tp": b["tp0"], "gross": b["pnl"], "swap": 0,
                           "comm": 0, "net": b["pnl"], "r_net": b.get("r"), "why": b.get("why")})
    closed.sort(key=lambda x: str(x.get("close_t") or ""))
    return {"bot": NAME, "label": "בוט 6 · שיטת בוט 5 בחשבון Swing (וירטואלי)", "watch_only": True,
            "rules": "מסלולי הגאפ של בוט 5 בדיוק כפי שהם רצים עכשיו, בחשבון FTMO Swing: בלי סגירה לפני סופ\"ש / הפסקה, "
                     "בלי עצירה סביב חדשות, עם עמלת לילה. שומר 4.5% ביום / 9% בסך הכול.",
            "account": "וירטואלי · בוט 6", "server": "FTMO Swing · צפייה בלבד · בלי כסף", "initial": INITIAL,
            "balance": bal, "equity": eq, "day_start": st.get("day_start"), "started": st["started"],
            "updated_utc": now.isoformat(timespec="seconds"), "positions": pos, "closed": closed[-300:],
            "equity_curve": st["equity_curve"][-1000:], "tally": st["tally"],
            "skipped": sum(1 for b in book.values() if b["status"] == "skipped"),
            "vacct": {"initial": INITIAL, "started": st["started"]}}


if __name__ == "__main__":
    main()
