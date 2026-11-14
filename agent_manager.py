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
import re
import subprocess
from pathlib import Path

import pandas as pd

from agents import agentkit as K
from agents import autodecide as AD
from agents import repo_io as RIO

OUT = Path("reports/agents/manager.json")
BOTS = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4", "bot5": "בוט 5 - סקאלפינג",
        "bot6": "בוט 6 · חוזים עתידיים", "bot7": "בוט 7 · פרסומים כלכליים"}
RISK_KEYS = {"bot1": "bot1", "bot2": "bot2", "bot3": "bot3", "bot4": "bot4"}
AGENT_ON = set(BOTS)                    # every bot has its smart agent (11.11.2026: bot 5 first, then all)


def decisions():
    return RIO.live_json("live/agent_decisions_state.json").get("cards", {})


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


def _ops_slot(out, b):
    """out[b] with problems / actions of the same length (the app shows actions[i] under problems[i])."""
    o = out.setdefault(b, {})
    o.setdefault("problems", [])
    o["actions"] = (list(o.get("actions") or []) + [None] * len(o["problems"]))[:len(o["problems"])]
    return o


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
        # a failure that a later successful run already replaced is not a problem any more (manager, 12.11.2026)
        ok_after = max((r["n"] for r in recent if r["c"] == "success"), default=0)
        # ... nor one that the bots' computer replaced: since 12.11 the agents run there, and the GitHub run is only the
        # fallback (manager, 13.11.2026: run #9 of 12.11 08:23 - a commit clash - was shown as a problem a day later)
        try:
            pc_last = max((pd.Timestamp(v) for v in (json.loads(Path("reports/agents/host.json").read_text(encoding="utf-8"))
                                                     .get("agent_last") or {}).values()), default=None)
        except Exception:  # noqa: BLE001
            pc_last = None
        bad = [r for r in recent if r["s"] == "completed" and r["c"] not in ("success", "skipped") and r["n"] > ok_after
               and not (pc_last is not None and pc_last > pd.Timestamp(r["t"]) + pd.Timedelta(hours=6))]
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
    # the bots' computer: an agent that ran but whose gap-method section did not change, and what its log says about
    # failures (manager, 12.11.2026: bot 6 ran at 16:43 and its section stayed at 02:50 with no error anywhere)
    try:
        h = json.loads(Path("reports/agents/host.json").read_text(encoding="utf-8"))
        res = json.loads(Path("data/agents/strategies/tal_gap_results.json").read_text(encoding="utf-8"))
        last, tails = h.get("agent_last") or {}, h.get("agent_tail") or {}
        try:
            pcm = (json.loads(Path("data/agents/plan.json").read_text(encoding="utf-8")).get("pc") or {})
        except Exception:  # noqa: BLE001
            pcm = {}
        now = pd.Timestamp.now(tz="UTC")
        for b in ("bot5", "bot6"):
            ran, upd = last.get(b), (res.get(b) or {}).get("updated")
            # agent_last is stamped when the agent STARTS: one still inside its minutes (+ the host's hour of grace)
            # has not written yet (manager, 13.11.2026: bot 6 started 03:42 and was flagged at 04:02 while running)
            if ran and now - pd.Timestamp(ran) < pd.Timedelta(minutes=float(pcm.get(b, 60)) + 60):
                continue
            if ran and upd and pd.Timestamp(ran) - pd.Timestamp(upd) > pd.Timedelta(hours=2):
                o = _ops_slot(out, b)
                o["problems"].append(
                    f"שיטת הגאפ: הסוכן רץ ב־{pd.Timestamp(ran):%d.%m %H:%M} UTC, והתוצאות לא התעדכנו מאז "
                    f"{pd.Timestamp(upd):%d.%m %H:%M}")
                o["actions"].append("הסוכן החכם של המנהל יבדוק את agent_tail ב־host.json ויתקן")
        # an agent that failed, or ended far inside its minutes while it still has work (manager, 13.11.2026: bot 2 ran
        # 2 of its 30 minutes at 05:28 and published nothing; the round's error list was gone after the next install)
        for x in h.get("agent_errors_24h") or []:
            name = str(x.get("err", "")).split(":")[0]
            o = _ops_slot(out, name.replace("-new", "") or "pc")
            o["problems"].append(f"הסוכן {name} נכשל ב־{str(x.get('at', ''))[11:16]} UTC: {str(x.get('err', ''))[len(name) + 2:][:140]}")
            o["actions"].append("הסוכן החכם של המנהל יבדוק את הלוג ויתקן")
        for b, r in (h.get("agent_runs") or {}).items():
            if r.get("adopted") or not r.get("budget_sec") or r.get("err"):
                continue
            tail = " ".join(tails.get(b) or [])
            idle = re.search(r"\b0 left\b|nothing left|nothing to test", tail)
            if r.get("sec", 0) < 0.15 * r["budget_sec"] and not idle and re.search(r"[1-9]\d* (combos )?left", tail):
                o = _ops_slot(out, b.replace("-new", ""))
                o["problems"].append(f"הסוכן {b} רץ רק {r['sec'] // 60} מתוך {r['budget_sec'] // 60} דקות "
                                     f"(התחיל {str(r.get('at', ''))[11:16]} UTC) למרות שיש לו עבודה")
                o["actions"].append("הסוכן החכם של המנהל יבדוק למה הסוכן נגמר מוקדם")
        for b, lines in tails.items():
            bad = [x for x in lines if "failed" in x or "Traceback" in x or "Error" in x]
            if bad:
                o = _ops_slot(out, b.replace("-new", ""))
                o["problems"].append(f"בלוג של הסוכן ({b}): {bad[-1][:160]}")
                o["actions"].append("הסוכן החכם של המנהל יבדוק את השורה ויתקן אם זו תקלה")
    except Exception:  # noqa: BLE001
        pass
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
        # cards_health.json = the cheap layer every bot has (agents/health.py, every few hours): live vs backtest
        nf = K.CARDS_DIR / bot / "cards_new.json"         # bots 5-6: the scout's strategies (agent_portfolio --new-only)
        smart += json.loads(nf.read_text(encoding="utf-8")).get("cards", []) if nf.exists() else []
        hf = K.CARDS_DIR / bot / "cards_health.json"
        smart += json.loads(hf.read_text(encoding="utf-8")).get("cards", []) if hf.exists() else []
        # any other card file of the bot (bot 5's GitHub tests of 13.11.2026: cards_pool.json, cards_wide*.json)
        for xf in sorted((K.CARDS_DIR / bot).glob("cards_*.json")):
            if xf.name in ("cards_smart.json", "cards_new.json", "cards_health.json"):
                continue
            try:
                smart += json.loads(xf.read_text(encoding="utf-8")).get("cards", [])
            except Exception:  # noqa: BLE001
                pass
        hh = K.CARDS_DIR / bot / "health.json"
        health = json.loads(hh.read_text(encoding="utf-8")) if hh.exists() else None
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
        # across the bot: one proposal per thing changed (same track + knob) - the one with the most total R on the
        # choosing period (13.11.2026, data/agents/objective.md: chosen to 2025 only and by the total, not by the R a
        # trade on 2026 as before)
        def _tot(c):
            t = c.get("train") or {}
            return (t.get("eff_n") or t.get("n") or 0) * t["mean"] if t.get("mean") is not None else -9e9
        # only a live proposal competes (14.11.2026, bot 6's smart agent): a card already withdrawn / declined kept
        # every later proposal of the same track + knob at "watch" - 38531db8b5eb (bot 6 t5_1 + MAXC 0.2, the fix of
        # the 13.11 loss) was withdrawn and then held its re-issued copy back, so the fix could never be decided
        try:
            gone = {x.get("card") for x in json.loads(Path("reports/agents/decisions.json").read_text(encoding="utf-8")).get("decisions", [])
                    if x.get("type") == "card" and x.get("bot") == bot and x.get("decision") in ("withdrawn", "declined")}
        except Exception:  # noqa: BLE001
            gone = set()
        live_st = lambda c: c["status"] in ("pending", "approved") and c["id"] not in gone  # noqa: E731
        best = {}
        for c in cards:
            if c["tier"] in ("strong", "regular") and c["kind"] == "improve" and live_st(c):
                ch = c["change"] or {}
                k = ch.get("track") or ch.get("combo"), ch.get("knob") or ch.get("family")
                if k not in best or _tot(c) > _tot(best[k]):
                    best[k] = c
        for c in cards:
            if c["tier"] in ("strong", "regular") and c["kind"] == "improve" and live_st(c):
                ch = c["change"] or {}
                k = ch.get("track") or ch.get("combo"), ch.get("knob") or ch.get("family")
                if best[k] is not c:
                    c["tier"], c["verdict"] = "watch", "יש הצעה טובה יותר לאותו שינוי"
        order = {"strong": 0, "regular": 1, "watch": 2, "rejected": 3}
        cards.sort(key=lambda c: (c["status"] != "pending", order.get(c["tier"], 9), -((c.get("unseen") or {}).get("mean") or 0)))
        pend = [c for c in cards if c["status"] == "pending" and c["tier"] in ("strong", "regular")]
        bots[bot] = {"name": name, "agent": bot in AGENT_ON, "updated": src.get("updated"), "summary": src.get("summary"),
                     "tested": src.get("tested"), "progress": [src.get("done"), src.get("planned")],
                     "risk_recs": rc.get(RISK_KEYS.get(bot, ""), 0), "pending": len(pend), "cards": cards,
                     "health": health}
    # the manager decides in the user's place (user, 11.11.2026); the user sees the list at 05:00 and can undo
    try:
        dd = AD.decide(bots)
    except Exception as e:  # noqa: BLE001
        dd = {"decisions": [], "new": [], "error": str(e)}
        print("autodecide failed:", e)
    for b in bots.values():
        b["pending"] = sum(1 for c in b["cards"] if c["status"] == "pending" and c["tier"] in ("strong", "regular"))
    new = [c["id"] for b in bots.values() for c in b["cards"]
           if c["status"] == "pending" and c["tier"] in ("strong", "regular") and c["first_seen"] == now]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    op = ops()
    for b, o in op.items():
        if b in bots:
            bots[b]["ops"] = o
    # the health layer's problems are the manager's problems too (stale bot, limits, a bot far below its backtest)
    for b, x in bots.items():
        hp = (x.get("health") or {}).get("problems") or []
        if hp:
            o = bots[b].setdefault("ops", {})
            o["problems"] = (o.get("problems") or []) + hp
            o["actions"] = (o.get("actions") or []) + [None] * len(hp)
    OUT.write_text(json.dumps({"updated": now, "new": new, "bots": bots, "ops": op,
                                "decisions": {"new": dd.get("new", []), "unread": sum(1 for d in dd["decisions"] if not d.get("read") and not d.get("undone"))}}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"manager: {sum(b['pending'] for b in bots.values())} proposals waiting, {len(new)} new tonight")


if __name__ == "__main__":
    main()
