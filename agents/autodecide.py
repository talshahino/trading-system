"""The manager decides in the user's place (user, 11.11.2026: "הוא מחליט על הכל ... בחמש בבוקר אנחנו מקבלים את רשימת
ההחלטות שהתקבלו ... עם אפשרות לחזור צעד אחורה"; read decisions leave the list, an archive keeps 30 days).

Every manager run (agent_manager.py, every 3 hours and at night) goes over everything that waits for a decision and
decides it by fixed rules, writing each decision to reports/agents/decisions.json:
  agent cards     strong / regular after the manager's own judgement -> approved (the bot agent builds it into the
                  bot; code changes still reach the bots' computer as an update)
  health alerts   approved = the bot agent investigates and proposes a fix
  risk.json recs  (bots 1-4, the bot agents' risk / removal recommendations) -> applied on the bots' computer by
                  live/manager.py (auto_apply), which keeps the value before, so "undo" puts it back:
                    remove, tp, sl, dir, hours, vol -> approved (they passed the bot agent's both-halves tests)
                    a risk multiplier change -> approved from 5 live trades of that strategy; fewer -> waits
                    lowering bot 3's risk -> never (user); the user's own earlier "no" -> never (data/agents/user_rules.json)
Never decided here: the fixed loss limits (daily 5%, total 10%) - no card or recommendation can touch them.
The user's own decision always wins: an undone decision is not made again for the same recommendation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from agents import agentkit as K

ROOT = K.ROOT
OUT = K.CARDS_DIR / "decisions.json"
KEEP_DAYS = 30
MIN_TRADES_MULT = 5
NO_RISK_DOWN = {"bot3"}
KIND_HE = {"mult": "סיכון", "remove": "הסרה", "dir": "כיוון", "tp": "יעד", "sl": "סטופ", "hours": "שעות", "vol": "תנודתיות"}


def _git_show(path):
    from agents import repo_io as RIO
    return RIO.live_json(path)


def user_rules():
    try:
        return json.loads((ROOT / "data/agents/user_rules.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _did(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def load():
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"decisions": []}


def decide(bots, now=None):
    """bots: manager.json bots (cards already judged). Changes card statuses in place for what it decides.
    Returns the decisions document."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    at = now.isoformat(timespec="seconds")
    doc = load()
    old = {d["id"]: d for d in doc.get("decisions", [])}
    user = _git_show("live/agent_decisions_state.json")
    ucards, uread, uundo = user.get("cards", {}) or {}, set(user.get("read", []) or []), set(user.get("undo", []) or [])
    rules = user_rules()
    never = {(x["bot"], x["combo"], x["kind"]) for x in rules.get("never", [])}
    # an undone recommendation is not made again for 30 days, even when a later sample brings it back
    never |= {(d["bot"], d.get("combo"), d["kind"]) for d in old.values() if d["id"] in uundo and d["type"] == "risk"}
    new = []

    def add(d):
        if d["id"] in old:
            return old[d["id"]]
        d.update(at=at, by="manager")
        old[d["id"]] = d
        new.append(d)
        return d

    # 1) the agents' cards (all bots)
    for bot, b in bots.items():
        for c in b.get("cards", []):
            if (ucards.get(c["id"]) or {}).get("decision"):           # the user decided it himself
                continue
            if c["tier"] not in ("strong", "regular"):
                continue
            did = _did("card", c["id"])
            if did in uundo:
                c["status"], c["verdict"] = "rejected", c["verdict"] + " · בוטל על ידך"
                continue
            what = ("הסוכן של הבוט יבדוק ויציע תיקון" if c["kind"] == "info" else
                    "הסוכן של הבוט יבנה את השינוי בבוט")
            d = add({"id": did, "bot": bot, "type": "card", "card": c["id"], "kind": c["kind"], "tier": c["tier"],
                     "title": c["title"], "decision": "approved", "why": c["verdict"], "effect": what})
            c["status"], c["by"], c["decided"], c["decision_id"] = "approved", "manager", d["at"], did
    # 2) the bot agents' risk / removal recommendations (bots 1-4)
    try:
        risk = json.loads((ROOT / "reports/trade_review/risk.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        risk = {}
    cur = (_git_show("live/manager_state.json").get("risk") or {})
    sample = risk.get("sample")
    for bot, x in (risk.get("bots") or {}).items():
        dec = cur.get(bot, {})
        rej = dec.get("rejected", {}) or {}
        for c in x.get("combos", []):
            combo = c["combo"]
            mult = float((dec.get("mult") or {}).get(combo, 1) or 1) if combo in (dec.get("mult") or {}) else 1.0
            for r in c.get("recs") or []:
                kind, val = r["kind"], r.get("value")
                if kind == "mult" and abs(float(val) - mult) < 1e-9:
                    continue
                if kind == "remove" and mult == 0:
                    continue
                if kind not in ("mult", "remove") and _same((dec.get(kind) or {}).get(combo), val):
                    continue
                if kind != "remove" and rej.get(f"{kind}:{combo}") == sample:
                    continue                                          # the user declined it in this sample
                if (bot, combo, kind) in never:
                    continue
                did = _did("risk", bot, combo, kind, json.dumps(val), sample)
                if did in uundo:
                    continue
                if kind == "mult" and bot in NO_RISK_DOWN and float(val) < mult:
                    continue                                          # user: never lower bot 3's risk
                if kind == "mult" and (c.get("trades") or 0) < MIN_TRADES_MULT:
                    continue                                          # too few live trades to move the risk yet
                set_kind, set_val = ("mult", 0) if kind == "remove" else (kind, val)
                prev = mult if set_kind == "mult" else (dec.get(set_kind) or {}).get(combo)
                add({"id": did, "bot": bot, "type": "risk", "kind": kind, "combo": combo,
                     "title": f"{combo}: {r.get('label') or KIND_HE.get(kind, kind)}", "decision": "approved",
                     "why": r.get("why") or "", "effect": "יוחל במחשב הבוטים תוך כ־10 דקות",
                     "action": {"bot": bot, "combo": combo, "kind": set_kind, "value": set_val, "prev": None if prev == 1.0 and set_kind == "mult" else prev}})
    # keep 30 days; read / undone marks come from the app (through the bots' computer)
    cutoff = now - pd.Timedelta(days=KEEP_DAYS)
    keep = [d for d in old.values() if pd.Timestamp(d["at"]) >= cutoff]
    for d in keep:
        d["read"] = d["id"] in uread
        d["undone"] = d["id"] in uundo
    keep.sort(key=lambda d: d["at"], reverse=True)
    doc = {"updated": at, "new": [d["id"] for d in new], "decisions": keep}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"manager decided {len(new)} new ({sum(d['type'] == 'risk' for d in new)} risk), {len(keep)} in the last {KEEP_DAYS} days")
    return doc
