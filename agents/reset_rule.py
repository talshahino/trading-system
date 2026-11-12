"""Bot 7's account reset - the manager decides it (user, 12.11.2026):

  "each time a strategy of bot 7 is new or improved, the account starts again from $100,000" - but only the manager
  decides: the bot's agent makes the changes, and the manager judges whether a change is material to the strategy's
  style; if it is, it resets the account or goes on.
  "after two weeks with no reset the bot is fairly stable: from there the manager does not reset any more. If it sees
  instability again it sends a message / asks to reset again."
  "when you reset, the trades are kept - for the manager and for the bot's agent - to learn from the strategies and the
  trades themselves where to improve and what to keep."

What counts as material (the manager's judgement, fixed here so it is the same every run):
  material      a strategy added or removed, an instrument / timeframe added or removed, a rule of a strategy
                changed (a filter, entry, stop, target - live/bots/bot7.json plans), the strategies' code changed
                (agents/event_strategies.py, the strategy list in live/news_bot.py)
  not material  a risk multiplier (live/risk.json), a text, a card id - the account goes on

The decision goes to reports/agents/decisions.json as type "reset" (approved, or "proposed" once the bot is stable),
the bots' computer applies it (live/manager.py auto_apply -> live/news_bot_reset.json -> live/news_bot.py), and undo
brings the account before back with every trade since. Before each reset the account's trades are archived in
data/agents/bot7_archive/ (the bot 7 agent reads them: data/agents/strategies/shocks.md).
"""
from __future__ import annotations

import hashlib
import json
import re

import pandas as pd

from agents import agentkit as K

ROOT = K.ROOT
BOT = "bot7"
STATE = ROOT / "data/agents/bot7_reset.json"
ARCHIVE = ROOT / "data/agents/bot7_archive"
STABLE_DAYS = 14
CODE = ("agents/event_strategies.py",)


def _sha(b):
    return hashlib.sha1(b).hexdigest()[:12]


def fingerprint():
    """What bot 7 trades and by which rules - risk multipliers, texts and card ids left out."""
    parts = {}
    try:
        d = json.loads((ROOT / "live/bots/bot7.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        d = {}
    parts["combos"] = sorted(d.get("combos") or [])
    parts["plans"] = {k: {a: b for a, b in (v or {}).items() if a not in ("card", "why", "note", "at")}
                      for k, v in sorted((d.get("plans") or {}).items())}
    for f in CODE:
        p = ROOT / f
        parts[f] = _sha(p.read_bytes()) if p.exists() else None
    try:
        src = (ROOT / "live/news_bot.py").read_text(encoding="utf-8")
        m = re.search(r"^STRATS\s*=\s*(.+)$", src, re.M)
        parts["strats"] = m.group(1).strip() if m else None
    except Exception:  # noqa: BLE001
        parts["strats"] = None
    return _sha(json.dumps(parts, sort_keys=True).encode()), parts


def _what_changed(a, b):
    out = []
    ca, cb = set(a.get("combos") or []), set(b.get("combos") or [])
    if cb - ca:
        out.append(f"נוספו {len(cb - ca)} אסטרטגיות/מכשירים")
    if ca - cb:
        out.append(f"הוסרו {len(ca - cb)} אסטרטגיות/מכשירים")
    if a.get("plans") != b.get("plans"):
        ks = {k for k in set(a.get("plans") or {}) | set(b.get("plans") or {})
              if (a.get("plans") or {}).get(k) != (b.get("plans") or {}).get(k)}
        out.append("כלל השתנה ב־" + ", ".join(sorted(ks)[:4]))
    if any(a.get(f) != b.get(f) for f in CODE) or a.get("strats") != b.get("strats"):
        out.append("קוד האסטרטגיות השתנה")
    return " · ".join(out) or "שינוי באסטרטגיה"


def archive(live_state, v):
    """The account's trades before a reset, for the manager and the bot 7 agent to learn from."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    va = live_state.get("vacct") or {}
    led = va.get("ledger") or {}
    trades = [t for t in (live_state.get("trades") or []) if t.get("id") in led] or (live_state.get("trades") or [])
    pnl = sum((x.get("pnl") or 0) for x in led.values())
    rs = [x.get("r") for x in led.values() if x.get("r") is not None]
    doc = {"bot": BOT, "account_v": v, "started": va.get("started"), "ended": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "initial": va.get("initial"), "pnl": round(pnl, 2), "trades_n": len(led),
           "wins": sum(1 for r in rs if r > 0), "mean_r": round(sum(rs) / len(rs), 3) if rs else None,
           "ledger": led, "trades": trades}
    f = ARCHIVE / f"account_v{v}.json"
    f.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    try:
        rel = f.relative_to(ROOT).as_posix()
    except ValueError:
        rel = f.as_posix()
    return rel, doc


def check(add, did, live_json, now, ucards=None, approved=()):
    """Called by agents/autodecide.decide. add(decision) records a decision; did(*parts) makes its id."""
    fp, parts = fingerprint()
    live = live_json("live/news_bot_state.json") or {}
    va = live.get("vacct") or {}
    cur_v = int(va.get("v") or 1)
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        st = None
    if not st:
        # first run: today's strategies are the baseline, the account's own start is the last reset
        st = {"fp": fp, "parts": parts, "last_reset": va.get("started") or now.isoformat(timespec="seconds"),
              "account_v": cur_v, "resets": [], "stable": False, "proposed": None}
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    # a reset the user approved by hand (he said yes to the manager's request)
    p = st.get("proposed")
    if p and (p["id"] in set(approved) or ((ucards or {}).get(p["id"]) or {}).get("decision") == "approved"):
        _reset(add, did, st, live, cur_v, now, p["fp"], p["what"], "טל אישר את בקשת המנהל לאפס")
        st["proposed"] = None
    days = (now - pd.Timestamp(st["last_reset"])).total_seconds() / 86400
    st["stable"] = days >= STABLE_DAYS
    if fp != st["fp"]:
        what = _what_changed(st.get("parts") or {}, parts)
        if not st["stable"]:
            _reset(add, did, st, live, cur_v, now, fp, what,
                   f"שינוי מהותי בסגנון האסטרטגיה ({what}), והבוט עוד לא יציב ({days:.0f} ימים מהאיפוס האחרון, פחות מ־{STABLE_DAYS})")
        else:
            # stable (two weeks with no reset): no reset by itself - it asks the user
            pid = did("reset_ask", BOT, fp)
            add({"id": pid, "bot": BOT, "type": "reset", "kind": "reset", "combo": None,
                 "title": "בוט 7: המנהל מבקש לאפס את החשבון ל־100,000$",
                 "decision": "proposed",
                 "why": f"שינוי מהותי ({what}) אחרי {days:.0f} ימים יציבים - הבוט יציב, אז המנהל לא מאפס לבד ומבקש את אישורך",
                 "effect": "לא אופס. לאישור: כתוב לClaude \"אשר איפוס בוט 7\""})
            st["proposed"] = {"id": pid, "fp": fp, "what": what, "at": now.isoformat(timespec="seconds")}
        st["fp"], st["parts"] = fp, parts
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def _reset(add, did, st, live, cur_v, now, fp, what, why):
    new_v = max(cur_v, int(st.get("account_v") or 1)) + 1
    path, doc = archive(live, cur_v)
    rid = did("reset", BOT, new_v, fp)
    add({"id": rid, "bot": BOT, "type": "reset", "kind": "reset", "combo": None,
         "title": f"בוט 7: החשבון מתחיל מחדש מ־100,000$ (חשבון {new_v})",
         "decision": "approved", "why": why,
         "effect": (f"החשבון הקודם ({doc['trades_n']} עסקאות, {doc['pnl']:+,.0f}$) נשמר ב־{path} "
                    "לסוכן של בוט 7 ולמנהל, ללמוד ממנו מה לשפר ומה לשמר. ביטול מחזיר את החשבון הקודם עם כל העסקאות"),
         "archive": path, "action": {"bot": BOT, "kind": "reset", "v": new_v}})
    st["resets"].append({"v": new_v, "at": now.isoformat(timespec="seconds"), "what": what, "archive": path, "id": rid})
    st["last_reset"], st["account_v"], st["stable"] = now.isoformat(timespec="seconds"), new_v, False
