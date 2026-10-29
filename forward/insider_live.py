"""Forward test (paper money only): the insider-purchase rule, fixed in advance, run on NEW
SEC filings every trading day in the Alpaca Paper account. Nobody has seen this data, so
after ~3 months it is the cleanest test there is.

THE RULE (decided on 2026-10-29, before any forward data existed; do not tune it):
  - Form 4, open-market purchase (code P), by the CEO / CFO / President,
    total bought in that day's filings >= $100,000.
  - Stock price >= $1, average traded value >= $1M a day, daily volatility >= 5.5%
    (volatility score 80+), tradable on Alpaca. Not already held.
  - Buy the next trading day. Size: risk 1.5% of a virtual $10,000 account against a
    2-sigma move over 10 days (max 25% of the account, max 10 positions).
  - Exit: stop 1.5 sigma below, target 3 sigma above (sigma = daily vol x sqrt(10)),
    or after 10 trading days, whichever comes first.
  - Risk rule: no new trades once the virtual account is 10% below $10,000.

State lives in forward/state.json and every action is appended to forward/insider_log.csv;
the workflow commits both back to the repository. forward/report.html is the scoreboard.
"""
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from agents.news_agent import _headers as alpaca_headers  # noqa: E402

HERE = Path(__file__).resolve().parent
STATE = HERE / "state.json"
LOG = HERE / "insider_log.csv"
REPORT = HERE / "report.html"

START_EQUITY = 10_000.0
MIN_VALUE = 100_000
MIN_PRICE = 1.0
MIN_DOLLAR_VOL = 1e6
MIN_DAILY_VOL = 0.055
HOLD_DAYS = 10
RISK_PCT = 0.015   # changed from 0.5% on 2026-10-29 (day 1, user's decision); sizing only, the rule is unchanged
MAX_POS_PCT = 0.25
MAX_POSITIONS = 10
STOP_SIG, TARGET_SIG = 1.5, 3.0
TITLE_RE = re.compile(r"chief executive|\bceo\b|chief financial|\bcfo\b|president", re.I)
TRADE_URL = config.ALPACA_TRADING_URL
DATA_URL = config.ALPACA_DATA_URL


# ---------------- SEC ----------------
def _sec_get(url):
    hdr = {"User-Agent": f"trading-system research {config.SEC_CONTACT_EMAIL}"}
    for attempt in range(5):
        try:
            r = requests.get(url, headers=hdr, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429, 500, 502, 503):
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(0.12)                    # < 10 requests a second
        return r.text
    raise RuntimeError(f"SEC unreachable: {url}")


def form4_files(day):
    """File names of all Form 4 filings in the SEC daily index for `day` (None if no index)."""
    q = (day.month - 1) // 3 + 1
    txt = _sec_get(f"https://www.sec.gov/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx")
    if txt is None:
        return None
    files = []
    for line in txt.splitlines():
        if line.startswith("4 "):
            m = re.search(r"(edgar/data/\S+\.txt)\s*$", line)
            if m:
                files.append(m.group(1))
    return sorted(set(files))


def _t(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else ""


def parse_form4(text):
    """[(symbol, title, value)] for open-market purchases in one filing."""
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", text, re.S)
    if not m:
        return []
    try:
        doc = ET.fromstring(m.group(0))
    except ET.ParseError:
        return []
    if _t(doc, "documentType") != "4":
        return []
    sym = _t(doc, "issuer/issuerTradingSymbol").upper()
    titles = " ".join(_t(o, "reportingOwnerRelationship/officerTitle") for o in doc.findall("reportingOwner"))
    value = 0.0
    for tr in doc.findall("nonDerivativeTable/nonDerivativeTransaction"):
        if _t(tr, "transactionCoding/transactionCode") != "P":
            continue
        if _t(tr, "transactionAmounts/transactionAcquiredDisposedCode/value") != "A":
            continue
        try:
            sh = float(_t(tr, "transactionAmounts/transactionShares/value") or 0)
            px = float(_t(tr, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        value += sh * px
    return [(sym, titles, value)] if value > 0 and sym else []


def signals_for(day):
    files = form4_files(day)
    if files is None:
        return None
    print(f"  {day}: {len(files)} Form 4 filings")
    agg = {}
    for i, f in enumerate(files, 1):
        txt = _sec_get(f"https://www.sec.gov/Archives/{f}")
        for sym, title, value in parse_form4(txt or ""):
            if TITLE_RE.search(title):
                agg[sym] = agg.get(sym, 0.0) + value
        if i % 250 == 0:
            print(f"    {i}/{len(files)}")
    return {s: v for s, v in agg.items() if v >= MIN_VALUE and re.fullmatch(r"[A-Z]{1,5}", s)}


# ---------------- Alpaca ----------------
def _api(method, path, base=TRADE_URL, **kw):
    r = requests.request(method, base + path, headers=alpaca_headers(), timeout=30, **kw)
    if r.status_code >= 400:
        return None, f"{r.status_code} {r.text[:200]}"
    return (r.json() if r.text else {}), None


def daily_stats(sym):
    end = datetime.utcnow() - timedelta(minutes=20)
    start = end - timedelta(days=45)
    data, err = _api("GET", "/v2/stocks/bars", base=DATA_URL, params={
        "symbols": sym, "timeframe": "1Day", "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "adjustment": "all", "feed": config.BAR_FEED, "limit": 100})
    bars = (data or {}).get("bars", {}).get(sym, []) if data else []
    if len(bars) < 15:
        return None
    c = np.array([b["c"] for b in bars])
    v = np.array([b["v"] for b in bars])
    rets = np.diff(c) / c[:-1]
    return {"close": c[-1], "dv": float(np.mean((c * v)[-20:])), "vol": float(np.std(rets[-20:], ddof=1))}


def last_price(sym):
    data, _ = _api("GET", f"/v2/stocks/{sym}/trades/latest", base=DATA_URL, params={"feed": "iex"})
    try:
        return float(data["trade"]["p"])
    except (TypeError, KeyError):
        return None


# ---------------- bookkeeping ----------------
def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"last_day": None, "open": {}, "closed": [], "halted": False}


def log(rows):
    df = pd.DataFrame(rows)
    df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def trading_days_between(d0, d1):
    return int(np.busday_count(d0, d1))


def refresh(state):
    """Fill prices of new trades and detect exits (stop/target filled by the broker)."""
    positions, _ = _api("GET", "/v2/positions")
    held = {p["symbol"]: p for p in (positions or [])}
    for sym, t in list(state["open"].items()):
        if t.get("entry") is None:
            o, _ = _api("GET", f"/v2/orders:by_client_order_id", params={"client_order_id": t["order_id"]})
            if o and o.get("filled_avg_price"):
                t["entry"] = float(o["filled_avg_price"])
                t["qty"] = float(o["filled_qty"])
            elif o and o.get("status") in ("canceled", "expired", "rejected"):
                del state["open"][sym]
                log([{"time": datetime.utcnow().isoformat(), "symbol": sym, "action": "not filled",
                      "detail": o.get("status")}])
                continue
        if t.get("entry") is not None and sym not in held:
            # Gone from the account: the stop or the target was hit. Find the exit price.
            o, _ = _api("GET", "/v2/orders", params={"status": "closed", "symbols": sym, "limit": 20, "nested": "true"})
            exit_px = None
            for od in o or []:
                for leg in [od] + (od.get("legs") or []):
                    if leg.get("side") == "sell" and leg.get("filled_avg_price"):
                        exit_px = float(leg["filled_avg_price"])
            exit_px = exit_px or t["entry"]
            pnl = (exit_px - t["entry"]) * t["qty"]
            state["closed"].append({**t, "symbol": sym, "exit": exit_px, "pnl": pnl,
                                    "exit_day": str(date.today()), "reason": "stop/target"})
            del state["open"][sym]
            log([{"time": datetime.utcnow().isoformat(), "symbol": sym, "action": "closed (stop/target)",
                  "detail": f"pnl {pnl:.2f}"}])
    unreal = sum(float(held[s]["unrealized_pl"]) for s in state["open"] if s in held)
    realized = sum(c["pnl"] for c in state["closed"])
    return START_EQUITY + realized + unreal, held


def time_exits(state, held):
    today = date.today()
    for sym, t in list(state["open"].items()):
        if t.get("entry") is None or sym not in held:
            continue
        if trading_days_between(date.fromisoformat(t["entry_day"]), today) >= HOLD_DAYS:
            _api("DELETE", "/v2/orders", params={"symbols": sym})        # cancel bracket legs
            time.sleep(1)
            _api("DELETE", f"/v2/positions/{sym}")
            px = float(held[sym]["current_price"])
            pnl = (px - t["entry"]) * t["qty"]
            state["closed"].append({**t, "symbol": sym, "exit": px, "pnl": pnl,
                                    "exit_day": str(today), "reason": "10 days"})
            del state["open"][sym]
            log([{"time": datetime.utcnow().isoformat(), "symbol": sym, "action": "closed (10 days)",
                  "detail": f"pnl {pnl:.2f}"}])


def enter(state, equity, sym, value):
    if sym in state["open"] or len(state["open"]) >= MAX_POSITIONS:
        return f"skip ({'held' if sym in state['open'] else 'max positions'})"
    asset, _ = _api("GET", f"/v2/assets/{sym}")
    if not asset or not asset.get("tradable"):
        return "skip (not tradable)"
    st = daily_stats(sym)
    if not st:
        return "skip (no price history)"
    if st["close"] < MIN_PRICE or st["dv"] < MIN_DOLLAR_VOL or st["vol"] < MIN_DAILY_VOL:
        return f"skip (price {st['close']:.2f}, $vol {st['dv'] / 1e6:.1f}M, vol {st['vol'] * 100:.1f}%)"
    px = last_price(sym) or st["close"]
    sig = st["vol"] * math.sqrt(HOLD_DAYS)
    dollars = min(equity * RISK_PCT / (2 * sig), equity * MAX_POS_PCT, st["dv"] / 390 * 0.05)
    qty = int(dollars // px)
    if qty < 1:
        return f"skip (size ${dollars:.0f} < 1 share)"
    oid = f"ins-{sym}-{date.today():%Y%m%d}"
    body = {"symbol": sym, "qty": str(qty), "side": "buy", "type": "market", "time_in_force": "gtc",
            "order_class": "bracket", "client_order_id": oid,
            "take_profit": {"limit_price": f"{px * (1 + TARGET_SIG * sig):.2f}"},
            "stop_loss": {"stop_price": f"{px * (1 - STOP_SIG * sig):.2f}"}}
    o, err = _api("POST", "/v2/orders", json=body)
    if err:
        return f"order rejected: {err}"
    state["open"][sym] = {"order_id": oid, "entry_day": str(date.today()), "ref_price": px, "qty": qty,
                          "entry": None, "sigma": sig, "insider_value": value}
    return f"bought {qty} @ ~{px:.2f} (stop {px * (1 - STOP_SIG * sig):.2f}, target {px * (1 + TARGET_SIG * sig):.2f})"


def write_report(state, equity):
    closed = pd.DataFrame(state["closed"])
    n = len(closed)
    wins = int((closed["pnl"] > 0).sum()) if n else 0
    rows = "".join(f"<tr><td>{c['symbol']}</td><td>{c['entry_day']}</td><td>{c['exit_day']}</td>"
                   f"<td>{c['entry']:.2f}</td><td>{c['exit']:.2f}</td><td>{c['pnl']:+.2f}</td><td>{c['reason']}</td></tr>"
                   for c in state["closed"][::-1])
    opens = "".join(f"<tr><td>{s}</td><td>{t['entry_day']}</td><td>{t.get('entry') or '—'}</td><td>{t['qty']}</td>"
                    f"<td>${t['insider_value']:,.0f}</td></tr>" for s, t in state["open"].items())
    REPORT.write_text(f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Insider Forward Test</title>
<style>body{{font:15px/1.6 system-ui;margin:0;background:#f7f7f5;color:#1d1d1b}}main{{max-width:760px;margin:auto;padding:20px 16px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}}td,th{{padding:6px 8px;border-bottom:1px solid #e3e3de;text-align:start}}
@media (prefers-color-scheme:dark){{body{{background:#151514;color:#ececea}}table{{background:#1e1e1c}}td,th{{border-color:#33332f}}}}</style>
</head><body><main><h1>ניסוי קדימה: קניות מנכ״לים (כסף וירטואלי)</h1>
<p>עודכן {datetime.utcnow():%Y-%m-%d %H:%M} UTC. חשבון וירטואלי: <b>${equity:,.2f}</b> (התחלה $10,000).
{'<b>המסחר נעצר: ירידה של 10% מהפתיחה.</b>' if state.get('halted') else ''}</p>
<p>עסקאות שנסגרו: {n}, מרוויחות: {wins}{f' ({wins / n:.0%})' if n else ''}, רווח/הפסד סגור: ${closed['pnl'].sum() if n else 0:+,.2f}.</p>
<h2>פוזיציות פתוחות</h2><table><tr><th>מניה</th><th>כניסה</th><th>מחיר</th><th>כמות</th><th>קנייה של המנהל</th></tr>{opens}</table>
<h2>עסקאות שנסגרו</h2><table><tr><th>מניה</th><th>כניסה</th><th>יציאה</th><th>מחיר כניסה</th><th>מחיר יציאה</th><th>רווח $</th><th>סיבה</th></tr>{rows}</table>
<p style="color:#888;font-size:13px">הכלל נקבע מראש ב־29.10.2026 ולא משתנה במהלך הניסוי. חשבון Paper בלבד.</p>
</main></body></html>""", encoding="utf-8")


def main():
    if not config.SEC_CONTACT_EMAIL:
        sys.exit("Missing SEC_CONTACT_EMAIL")
    state = load_state()
    clock, err = _api("GET", "/v2/clock")
    if err:
        sys.exit(f"Alpaca: {err}")
    equity, held = refresh(state)
    time_exits(state, held)
    equity, held = refresh(state)
    print(f"virtual equity ${equity:,.2f}, open positions {len(state['open'])}")
    if equity <= START_EQUITY * (1 - config.MAX_DRAWDOWN_PCT):
        state["halted"] = True
    if not clock.get("is_open"):
        # Holiday or outside hours: keep the filings for the next run that can trade.
        print("market closed: exits checked, no new trades, filings kept for the next run")
        write_report(state, equity)
        STATE.write_text(json.dumps(state, indent=1, default=str))
        return

    today = date.today()
    last = date.fromisoformat(state["last_day"]) if state["last_day"] else today - timedelta(days=1)
    if state["last_day"] is None:
        last = last - timedelta(days=1)            # first run: only yesterday's filings
    day = last + timedelta(days=1)
    rows = []
    while day < today:
        if day.weekday() < 5:
            sig = signals_for(day)
            if sig is not None:
                print(f"  signals {day}: {sig}")
                for sym, value in sorted(sig.items(), key=lambda kv: -kv[1]):
                    if state["halted"]:
                        res = "skip (halted: 10% below start)"
                    else:
                        res = enter(state, equity, sym, value)
                    print(f"    {sym} ${value:,.0f}: {res}")
                    rows.append({"time": datetime.utcnow().isoformat(), "symbol": sym, "action": res,
                                 "detail": f"filed {day}, CEO/CFO bought ${value:,.0f}"})
        state["last_day"] = str(day)
        day += timedelta(days=1)
    if rows:
        log(rows)
    write_report(state, equity)
    STATE.write_text(json.dumps(state, indent=1, default=str))


if __name__ == "__main__":
    main()
