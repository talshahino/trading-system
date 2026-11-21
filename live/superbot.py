"""Bot 8 - the super bot (user, 20.11.2026: "a new bot - a super bot that takes the top of the strategies, maximises
the profit and minimises the risk"; name: "בוט על"). Virtual, no money.

It trades nothing of its own: every 20 seconds it reads the trades the other bots already take on this computer -
bot 5's gap tracks (live/watch_own_state.json), bot 5ב's scout strategies (watch_scout), bots 1-4 (each bot's most
recently updated account, live/<account>_state.json) and bot 7 (news_bot) - and books each one in its own account at
the risk the super bot gives that strategy (live/bots/superbot.json: chosen on 2025 by agents/superbot_study.py,
bucket_vol - the risk split between the bots by their 2025 volatility, the whole scaled so 2025's deepest fall stays
at 7% and its worst day at 3%).

  sizing   at the first sight of a trade (opened after the super bot started): risk = balance x the strategy's risk;
           a new trade risks at most the room under 4.5% a day (from the day's start, Prague, open risk counted) and 9%
           in all; under a quarter of its risk left -> not taken
  result   the source trade's R x the risk: R = the source's net result / its risk at the first stop
           (bot 7 gives r_net; the others from entry, first stop, exit, net / gross)
  open     shown with the source's current price (R now x the risk)
State: live/superbot_state.json (the manager serves it to the app), and branch live every minute.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

LIVE = Path(__file__).resolve().parent
sys.path.insert(0, str(LIVE))
import reporter as REP  # noqa: E402

NAME = "superbot"
STATE_F = LIVE / f"{NAME}_state.json"
CONF_F = LIVE / "bots" / "superbot.json"
POLL = 20
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


def runners():
    """{bot number: the state of that bot's most recently updated account}."""
    best = {}
    for f in LIVE.glob("*_state.json"):
        if f.name.startswith(("watch_", "news_bot", "superbot", "manager")):
            continue
        j = rd(f)
        if not isinstance(j, dict) or "closed" not in j:
            continue
        m = re.search(r"בוט\s*(\d)", str(j.get("label") or ""))
        if not m or m.group(1) not in "1234":
            continue
        n = m.group(1)
        if n not in best or str(j.get("updated_utc") or "") > str(best[n].get("updated_utc") or ""):
            best[n] = j
    return best


def stream_of(bucket, t):
    """The strategy's key in the risk table (as agents/superbot_study names it)."""
    if bucket == "gap":
        tid = str(t.get("id") or t.get("ticket") or "")
        if "|tal|" not in tid:
            return None
        return f"gap:{tid.split('|')[-1]}#{t.get('symbol')}"
    return t.get("combo")


def sources():
    """[(bucket, kind open/closed, trade)]."""
    out = []
    for bucket, f in (("gap", "watch_own"), ("5b", "watch_scout"), ("bot7", "news_bot")):
        j = rd(LIVE / f"{f}_state.json", {}) or {}
        out += [(bucket, "open", t) for t in j.get("positions") or []]
        out += [(bucket, "closed", t) for t in j.get("closed") or []]
    for n, j in runners().items():
        out += [(f"bot{n}", "open", t) for t in j.get("positions") or []]
        out += [(f"bot{n}", "closed", t) for t in j.get("closed") or []]
    return out


def side(t):
    s = str(t.get("side") or "")
    return 1 if s == "buy" else -1 if s == "sell" else 0


def r_closed(t):
    if t.get("r_net") is not None:
        return float(t["r_net"])
    d, e, s0, x = side(t), t.get("open_px"), t.get("sl0"), t.get("close_px")
    if not d or not e or not s0 or x is None or abs(float(e) - float(s0)) <= 0:
        return None
    rg = d * (float(x) - float(e)) / abs(float(e) - float(s0))
    g, n = t.get("gross"), t.get("net")
    if g and n is not None and abs(float(g)) > 1e-9:
        return rg * float(n) / float(g)
    return rg


def r_open(t):
    d, e, s0, x = side(t), t.get("open"), t.get("sl0"), t.get("now")
    if not d or not e or not s0 or x is None or abs(float(e) - float(s0)) <= 0:
        return 0.0
    return d * (float(x) - float(e)) / abs(float(e) - float(s0))


def open_time(t):
    return str(t.get("open_t") or t.get("since") or "")


CRYPTO = ("BTC", "ETH", "LTC", "XRP", "SOL", "ADA", "DOGE", "BNB", "DOT", "AVAX", "LINK", "XMR", "DASH", "NEO", "BCH", "XLM")


def weekend(now):
    """FTMO's weekend break (user, 21.11.2026 "check it keeps the rules - trades are still open and the market is
    closed"): an FTMO Account (standard) must have nothing open over a 2+ hour break (ftmo.com FAQ "Can I hold positions
    overnight or over the weekend?" - the Challenge / Verification and Swing accounts may). From Friday 22:45 Prague
    (15 minutes before FX / metals / indices close) to Sunday 23:00 Prague: the super bot closes its open trades at the
    source's last price and opens none (crypto trades on)."""
    p = now.tz_convert("Europe/Prague")
    wd, hm = p.weekday(), p.hour * 60 + p.minute
    return (wd == 4 and hm >= 22 * 60 + 45) or wd == 5 or (wd == 6 and hm < 23 * 60)


def is_crypto(sym):
    return str(sym or "").upper().startswith(CRYPTO)


def pday(ts):
    return pd.Timestamp(ts).tz_convert("Europe/Prague").date().isoformat()


def main():
    conf = rd(CONF_F, {}) or {}
    st = rd(STATE_F, {}) or {}
    now = pd.Timestamp.now(tz="UTC")
    st.setdefault("started", now.isoformat(timespec="seconds"))
    st.setdefault("initial", float(conf.get("initial", 100000.0)))
    st.setdefault("balance", st["initial"])
    st.setdefault("book", {})            # key -> {stream, bucket, risk_usd, status, r, pnl, ...}
    st.setdefault("equity_curve", [])
    # every strategy's live record for good (user, 21.11.2026: after 3-4 weeks give more risk to the strategies that
    # earn live and less to the fading ones) - the book keeps only 14 days, this tally keeps n / sum R / sum R^2 / wins
    st.setdefault("tally", {})
    log(f"started; {len(st['book'])} booked")
    last_push = 0.0
    while True:
        try:
            conf = rd(CONF_F, conf) or conf
            risk_of, bdef = conf.get("risk") or {}, conf.get("bucket_risk") or {}
            now = pd.Timestamp.now(tz="UTC")
            today = pday(now)
            if st.get("day") != today:
                st["day"], st["day_start"] = today, st["balance"]
            book = st["book"]
            seen_open = {}
            for bucket, kind, t in sources():
                ot = open_time(t)
                if not ot or ot < st["started"]:
                    continue
                s = stream_of(bucket, t)
                if not s:
                    continue
                key = f"{bucket}|{s}|{ot[:16]}"
                b = book.get(key)
                if b is None and weekend(now) and not is_crypto(t.get("symbol")):
                    continue                     # nothing new over the weekend break (a source's trade is seen after it)
                if b is None:
                    rk_w = float(risk_of.get(s, bdef.get(bucket, 0.0)) or 0.0)
                    want = st["balance"] * rk_w
                    open_risk = sum(x["risk_usd"] for x in book.values() if x["status"] == "open")
                    day_pnl = sum(x["pnl"] for x in book.values() if x["status"] == "closed" and x.get("day") == today)
                    room = min(conf.get("day_limit", 0.045) * st["initial"] + day_pnl - open_risk,
                               st["balance"] - open_risk - st["initial"] * (1 - conf.get("total_limit", 0.09)))
                    b = {"stream": s, "bucket": bucket, "symbol": t.get("symbol"), "combo": t.get("combo"), "side": t.get("side"),
                         "open_t": ot, "open_px": t.get("open_px", t.get("open")), "sl0": t.get("sl0"), "tp0": t.get("tp0"),
                         "seen": now.isoformat(timespec="seconds"), "status": "open", "pnl": 0.0}
                    if want <= 0 or room < conf.get("min_share", 0.25) * want:
                        b.update(status="skipped", risk_usd=0.0, why="no room" if want > 0 else "no risk")
                    else:
                        b["risk_usd"] = round(min(want, room), 2)
                    book[key] = b
                if b["status"] != "open":
                    continue
                if kind == "closed":
                    r = r_closed(t)
                    if r is None:
                        b.update(status="skipped", why="no first stop")
                        continue
                    pnl = round(r * b["risk_usd"], 2)
                    b.update(status="closed", r=round(r, 4), pnl=pnl, close_t=t.get("close_t"), close_px=t.get("close_px"),
                             why=t.get("why") or t.get("reason"), day=today)
                    st["balance"] = round(st["balance"] + pnl, 2)
                    tally(st, b, r)
                else:
                    seen_open[key] = (t, r_open(t))
            # the weekend break: out at the source's last price (FTMO Account standard)
            if weekend(now):
                for key, b in book.items():
                    if b["status"] == "open" and not is_crypto(b.get("symbol")):
                        t, r = seen_open.get(key, ({}, 0.0))
                        pnl = round(r * b["risk_usd"], 2)
                        b.update(status="closed", r=round(r, 4), pnl=pnl, close_t=now.isoformat(timespec="seconds"),
                                 close_px=t.get("now"), why="לפני סגירת השוק (סוף שבוע)", day=today)
                        st["balance"] = round(st["balance"] + pnl, 2)
                        tally(st, b, r)
                        seen_open.pop(key, None)
            # open trades whose source no longer shows them and never closed (a source reset): closed at 0 after 2 days
            for key, b in book.items():
                if b["status"] == "open" and key not in seen_open and now - pd.Timestamp(b["seen"]) > pd.Timedelta(days=2):
                    b.update(status="closed", r=0.0, pnl=0.0, why="המקור לא הראה סגירה", day=today)
            unreal = sum(b["risk_usd"] * seen_open[k][1] for k, b in book.items() if b["status"] == "open" and k in seen_open)
            equity = round(st["balance"] + unreal, 2)
            st["equity"] = equity
            if not st["equity_curve"] or now - pd.Timestamp(st["equity_curve"][-1][0]) > pd.Timedelta(minutes=3):
                st["equity_curve"] = (st["equity_curve"] + [[now.isoformat(timespec="seconds"), st["balance"], equity]])[-2000:]
            # keep 14 days of closed / skipped
            cut = (now - pd.Timedelta(days=14)).isoformat()
            st["book"] = book = {k: b for k, b in book.items() if b["status"] == "open" or (b.get("close_t") or b["seen"]) >= cut}
            out = view(st, conf, seen_open, now)
            STATE_F.write_text(json.dumps(out | {"book": book, "day": st["day"], "tally": st["tally"]}, ensure_ascii=False,
                                          default=str), encoding="utf-8")
            if time.time() - last_push > 60:
                last_push = time.time()
                pub = dict(out, tally=st["tally"])
                err = REP.push_state(NAME, pub)
                if err:
                    log(err)
        except Exception:  # noqa: BLE001
            log(traceback.format_exc()[-800:])
        time.sleep(POLL)


def tally(st, b, r):
    x = st["tally"].setdefault(b["stream"], {"bucket": b["bucket"], "n": 0, "sum_r": 0.0, "sum_r2": 0.0, "wins": 0,
                                             "usd": 0.0, "first": b["open_t"]})
    x["n"] += 1
    x["sum_r"] = round(x["sum_r"] + r, 4)
    x["sum_r2"] = round(x["sum_r2"] + r * r, 4)
    x["wins"] += r > 0
    x["usd"] = round(x["usd"] + b["pnl"], 2)
    x["last"] = b.get("close_t")


def view(st, conf, seen_open, now):
    """The state in the shape the app shows for a virtual account."""
    book = st["book"]
    pos, closed = [], []
    for k, b in book.items():
        if b["status"] == "open" and k in seen_open:
            t, r = seen_open[k]
            pos.append({"ticket": k, "side": b["side"], "dir": "קנייה" if b["side"] == "buy" else "מכירה", "combo": b["combo"],
                        "symbol": b["symbol"], "lots": "וירטואלי", "open": b["open_px"], "now": t.get("now"), "sl": t.get("sl"),
                        "tp": t.get("tp"), "sl0": b["sl0"], "tp0": b["tp0"], "sl_usd": -b["risk_usd"], "tp_usd": None,
                        "pnl": round(b["risk_usd"] * r, 2), "since": b["open_t"], "pip": t.get("pip", 1.0), "free": False,
                        "source": b["bucket"]})
        elif b["status"] == "closed":
            closed.append({"id": k, "combo": b["combo"], "symbol": b["symbol"], "side": b["side"], "lots": "וירטואלי",
                           "open_t": b["open_t"], "open_px": b["open_px"], "close_t": b.get("close_t"), "close_px": b.get("close_px"),
                           "sl0": b["sl0"], "tp0": b["tp0"], "sl": b["sl0"], "tp": b["tp0"], "gross": b["pnl"], "swap": 0,
                           "comm": 0, "net": b["pnl"], "r_net": b.get("r"), "why": b.get("why"), "source": b["bucket"]})
    closed.sort(key=lambda x: str(x.get("close_t") or ""))
    by = {}
    for b in book.values():
        if b["status"] == "closed":
            x = by.setdefault(b["bucket"], {"closed": 0, "sum_usd": 0.0, "wins": 0})
            x["closed"] += 1
            x["sum_usd"] = round(x["sum_usd"] + b["pnl"], 2)
            x["wins"] += b["pnl"] > 0
    skipped = sum(1 for b in book.values() if b["status"] == "skipped")
    return {"bot": NAME, "label": conf.get("label", "בוט על (וירטואלי)"), "watch_only": True,
            "rules": conf.get("note", ""), "account": "וירטואלי · בוט על", "server": "צפייה בלבד · בלי כסף",
            "initial": st["initial"], "risk": conf.get("scale"), "balance": st["balance"], "equity": st["equity"],
            "day_start": st.get("day_start"), "started": st["started"], "updated_utc": now.isoformat(timespec="seconds"),
            "positions": pos, "closed": closed[-300:], "equity_curve": st["equity_curve"][-1000:],
            "summary": by, "skipped": skipped, "bucket_risk": conf.get("bucket_risk"),
            "vacct": {"initial": st["initial"], "started": st["started"]}}


if __name__ == "__main__":
    main()
