"""The bots' manager agent (user, 10-11.11.2026; plan "סוכני שיפור וסוכן מנהל") - nightly, after the bot agents.

It runs no backtests itself. It reads the ready cards of every bot agent (reports/agents/<bot>/cards.json) and judges
them by the fixed rules (agents/agentkit.py judge()), the same way for every bot:
  strong   -> proposed at the bot's own risk
  regular  -> proposed at a risk the bot agent sets from the history (no automatic half - user, 10.11.2026)
  watch    -> kept, shown folded, comes back when it has more trades
  rejected -> shown folded with the reason
It also looks across a bot: several cards that change the same thing -> only the best one is proposed; and keeps the
user's decisions and notes (live/agent_decisions_state.json on branch live, written by the app through the bots'
computer) - an approved or rejected card is not proposed again; a note gets an answer here, and the bot's own agent
answers it in depth on its next run (data/agents/replies.json).
The risk / removal recommendations of bots 1-4 (risk_review.py -> reports/trade_review/risk.json) are the bot agents'
own recommendations; the manager counts them per bot so the app shows everything in one place.
Writes reports/agents/manager.json (the app's "מנהל הבוטים").
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from agents import agentkit as K

OUT = Path("reports/agents/manager.json")
BOTS = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4", "bot5": "בוט 5 · סקלפינג",
        "bot6": "בוט 6 · חוזים עתידיים", "bot7": "בוט 7 · פרסומים כלכליים"}
RISK_KEYS = {"bot1": "bot1", "bot2": "bot2", "bot3": "bot3", "bot4": "bot4"}
AGENT_ON = {"bot5"}                     # bots whose agent already runs (stage 1: bot 5; the others follow)


def decisions():
    try:
        subprocess.run(["git", "fetch", "-q", "origin", "live"], check=False)
        raw = subprocess.run(["git", "show", "origin/live:live/agent_decisions_state.json"], capture_output=True, text=True).stdout
        return json.loads(raw).get("cards", {}) if raw else {}
    except Exception:  # noqa: BLE001
        return {}


def replies():
    try:
        return json.loads(Path("data/agents/replies.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def risk_counts():
    try:
        d = json.loads(Path("reports/trade_review/risk.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for b, x in d.get("bots", {}).items():
        out[b] = sum(len(c.get("recs") or []) for c in x.get("combos", []))
    return out


def ops():
    """The manager's second job (user, 11.11.2026): run the agents at maximum efficiency and solve their problems.
    Per agent: work done / left, time per experiment, how long the rest takes at tonight's budget, failed or stuck
    runs (GitHub Actions), and what the manager did or proposes about it."""
    out = {}
    try:
        import agent_bot5 as A5
        idx = A5.load_index()
        planned = [(t, k, v, c) for t, k, v, c in A5.variants()]
        per = {}
        for trk, knob, val, cfg in planned:
            for sym in A5.TRACKS[trk]["symbols"]:
                key = f"{trk}|{A5.vid(cfg)}|{sym}"
                p_ = per.setdefault(trk, {"done": 0, "left": 0, "secs": []})
                if key in idx:
                    p_["done"] += 1
                    if idx[key].get("secs"):
                        p_["secs"].append(idx[key]["secs"])
                else:
                    p_["left"] += 1
        budget = float(json.loads(Path("data/agents/plan.json").read_text()).get("bot5", {}).get("minutes", 30)) \
            if Path("data/agents/plan.json").exists() else 30.0
        tracks, problems, actions = {}, [], []
        for trk, p_ in per.items():
            avg = sum(p_["secs"]) / len(p_["secs"]) if p_["secs"] else None
            left_min = None if avg is None else round(p_["left"] * avg / 60, 1)
            nights = None if left_min is None else round(left_min / budget, 1)
            tracks[trk] = {"done": p_["done"], "left": p_["left"], "sec_per_exp": None if avg is None else round(avg, 1),
                           "left_minutes": left_min, "nights_at_budget": nights}
            if nights is not None and nights > 7:
                problems.append(f"{A5.TRK_NAME.get(trk, trk)}: נשארו {p_['left']} ניסויים, בערך {left_min / 60:.0f} שעות - "
                                f"{nights:.0f} לילות בתקציב הנוכחי")
                actions.append(f"{A5.TRK_NAME.get(trk, trk)}: להריץ במקביל לפי מכשיר, או להעביר למחשב הבוטים")
        try:
            raw = subprocess.run(["gh", "api", "repos/talshahino/trading-system/actions/workflows/agents.yml/runs?per_page=20",
                                  "--jq", "[.workflow_runs[]|{n:.run_number,s:.status,c:.conclusion,t:.created_at}]"],
                                 capture_output=True, text=True, timeout=60).stdout
            runs = json.loads(raw) if raw.strip() else []
        except Exception:  # noqa: BLE001
            runs = []
        recent = [r for r in runs if pd.Timestamp(r["t"]) > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)]
        bad = [r for r in recent if r["s"] == "completed" and r["c"] not in ("success", "skipped")]
        if bad:
            problems.append(f"{len(bad)} ריצות של הסוכנים נכשלו ביומיים האחרונים (#{', #'.join(str(r['n']) for r in bad)})")
            actions.append("הסוכן החכם של המנהל יבדוק את הלוג ויריץ מחדש")
        last = max((pd.Timestamp(v["at"]) for v in idx.values()), default=None)
        if last is not None and sum(p_["left"] for p_ in per.values()) and pd.Timestamp.now(tz="UTC") - last > pd.Timedelta(hours=36):
            problems.append("אין ניסויים חדשים כבר יותר מ־36 שעות, למרות שיש עבודה")
        out["bot5"] = {"tracks": tracks, "budget_minutes": budget, "running": sum(1 for r in recent if r["s"] != "completed"),
                       "problems": problems, "actions": actions, "last_experiment": None if last is None else last.isoformat()}
    except Exception as e:  # noqa: BLE001
        out["bot5"] = {"problems": [f"לא הצלחתי לקרוא את מצב הסוכן ({e})"]}
    return out


def main():
    dec, rep, rc = decisions(), replies(), risk_counts()
    try:
        old = json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        old = {}
    first_seen = {c["id"]: c.get("first_seen") for b in old.get("bots", {}).values() for c in b.get("cards", [])}
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    bots = {}
    for bot, name in BOTS.items():
        # cards.json = the nightly experiments; cards_smart.json = the bot agent's own discoveries and ideas (its
        # nightly Claude session: patterns, news reactions, the scout's strategies, the user's notes)
        f = K.CARDS_DIR / bot / "cards.json"
        src = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        g = K.CARDS_DIR / bot / "cards_smart.json"
        smart = json.loads(g.read_text(encoding="utf-8")).get("cards", []) if g.exists() else []
        cards = []
        for c in src.get("cards", []) + smart:
            c = dict(c)
            c["tier"], c["verdict"] = K.judge(c)            # the manager's own judgement is the one shown
            d = dec.get(c["id"]) or {}
            c["status"] = d.get("decision") or "pending"
            c["note"] = d.get("note")
            c["reply"] = (rep.get(c["id"]) or {}).get("text")
            if c["note"] and not c["reply"]:
                c["reply"] = "קיבלתי. הסוכן של הבוט יבדוק את ההערה בריצה הבאה ויענה כאן."
            c["first_seen"] = first_seen.get(c["id"]) or now
            cards.append(c)
        # across the bot: one proposal per thing changed (same track + knob) - the best on the unseen period
        best = {}
        for c in cards:
            if c["tier"] in ("strong", "regular") and c["kind"] == "improve":
                k = (c["change"] or {}).get("track"), (c["change"] or {}).get("knob")
                if k not in best or (c["unseen"]["mean"] or 0) > (best[k]["unseen"]["mean"] or 0):
                    best[k] = c
        for c in cards:
            if c["tier"] in ("strong", "regular") and c["kind"] == "improve":
                k = (c["change"] or {}).get("track"), (c["change"] or {}).get("knob")
                if best[k] is not c:
                    c["tier"], c["verdict"] = "watch", "יש הצעה טובה יותר לאותו שינוי"
        order = {"strong": 0, "regular": 1, "watch": 2, "rejected": 3}
        cards.sort(key=lambda c: (c["status"] != "pending", order.get(c["tier"], 9), -(c["unseen"] or {}).get("mean", 0) if c.get("unseen") else 0))
        pend = [c for c in cards if c["status"] == "pending" and c["tier"] in ("strong", "regular")]
        bots[bot] = {"name": name, "agent": bot in AGENT_ON, "updated": src.get("updated"), "summary": src.get("summary"),
                     "tested": src.get("tested"), "progress": [src.get("done"), src.get("planned")],
                     "risk_recs": rc.get(RISK_KEYS.get(bot, ""), 0), "pending": len(pend), "cards": cards}
    new = [c["id"] for b in bots.values() for c in b["cards"]
           if c["status"] == "pending" and c["tier"] in ("strong", "regular") and c["first_seen"] == now]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    op = ops()
    for b, o in op.items():
        if b in bots:
            bots[b]["ops"] = o
    OUT.write_text(json.dumps({"updated": now, "new": new, "bots": bots, "ops": op}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"manager: {sum(b['pending'] for b in bots.values())} proposals waiting, {len(new)} new tonight")


if __name__ == "__main__":
    main()
