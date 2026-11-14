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
                    the user's own earlier "no" -> never (data/agents/user_rules.json)
Never decided here: the fixed loss limits (daily 5%, total 10%) - no card or recommendation can touch them.
The user's own decision always wins: an undone decision is not made again for the same recommendation.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from agents import agentkit as K

ROOT = K.ROOT
OUT = K.CARDS_DIR / "decisions.json"
KEEP_DAYS = 30
MIN_TRADES_MULT = 5
NO_RISK_DOWN = set()                   # (until 12.11.2026 bot 3; now the user: bot 3 works like the other bots)
KIND_HE = {"mult": "סיכון", "remove": "הסרה", "dir": "כיוון", "tp": "יעד", "sl": "סטופ", "hours": "שעות", "vol": "תנודתיות"}



def _in_bot(bot, c):
    """The card's change is already in live/bots/<bot>.json (a new strategy among its combos, an improvement's card id
    in its plan): a whole-bot check after the build compares the bot with itself."""
    try:
        cfg = json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return True
    combo = (c.get("change") or {}).get("combo")
    if c.get("kind") == "new":
        return combo in (cfg.get("combos") or [])
    p = (cfg.get("plans") or {}).get(combo) or {}
    return c.get("id") in {v for k, v in p.items() if str(k).endswith("card")}

def _git_show(path):
    from agents import repo_io as RIO
    return RIO.live_json(path)


def user_rules():
    try:
        return json.loads((ROOT / "data/agents/user_rules.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


RISK_BOTS = ("bot1", "bot2", "bot3", "bot4")   # risk per strategy is applied on the bots' computer (live/risk.json)
STAR_MULT = 0.7                                 # "maybe luck": the lowered risk a small edge goes in with
STAR_GAIN, STAR_N = 0.10, 100                   # ... when the gain is under 0.10R a trade, or under 100 trades shown


def _total(s):
    """The period's total R (time-weighted trades x R a trade), or None."""
    if not s or s.get("mean") is None:
        return None
    return float(s.get("eff_n") or s.get("n") or 0) * float(s["mean"])


def maybe_luck(c):
    u, b = c.get("unseen") or {}, c.get("baseline") or {}
    if u.get("mean") is None:
        return False
    gain = u["mean"] - (b.get("mean") or 0.0) if c.get("kind") == "improve" and b else u["mean"]
    return gain < STAR_GAIN or (u.get("eff_n") or 0) < STAR_N


def star_floor(c, cur_m):
    """The lowest "maybe luck" risk that still keeps the strategy's 2026 total at least what runs now (THE GOAL, user,
    12.11.2026: the total, not R a trade). Manager, 13.11.2026 night: TV29 US30 (+17.0R vs +14.8R in 2026) and TV33 MSTR
    (+16.5R vs +14.6R) in bot 1 went in at x0.7 - 0.7 x 17.0 = 11.9R, less than the 14.8R x1.05 they made before the
    change; the discount turned a bigger total into a smaller one. None when it does not apply."""
    if c.get("kind") != "improve":
        return None
    tn, tb = _total(c.get("unseen")), _total(c.get("baseline"))
    if not tn or not tb or tn <= 0 or tb <= 0:
        return None
    return min(cur_m, math.ceil(cur_m * tb / tn * 100) / 100)


FAMILY_MIN, FAMILY_T = 4, 1.0                  # a scout strategy tested on >= 4 instrument/timeframe pairs of one bot ...


def _family_pick(cards, c):
    """A new scout strategy on ONE instrument, picked out of a family that loses on the choosing period (to 2025).
    Manager, 14.11.2026: SC010|15Min (Toto SBL) was tested on ~15 instruments in bots 2-4; together they lost on 2024-2025
    and on 2026, and ARM alone (+0.06R a trade to 2025, t 0.32, 28 trades) was approved in all three bots - the pick
    among many is luck, not an edge chosen on the period to 2025 (objective.md). Such a card waits (watch) unless its own
    choosing-period result is clear (t >= 1). Returns the reason, or None."""
    combo = (c.get("change") or {}).get("combo") or ""
    if combo.count("|") != 2:
        return None
    # the family is the strategy on every timeframe and instrument the bot tested (manager, 14.11.2026 night: bot 5 / 6
    # test 4 instruments per timeframe, so a family of one timeframe never reached 5 - SC071|15Min|US30 was approved at
    # t 0.33 on 28 trades while SC071 on its 8 pairs lost -101R to 2025; bot 7's SC043|4Hour|EURJPY likewise, -374R)
    fam = combo.split("|", 1)[0] + "|"
    tr = [x.get("train") or {} for x in cards
          if x.get("kind") == "new" and ((x.get("change") or {}).get("combo") or "").startswith(fam)]
    if len(tr) < FAMILY_MIN:
        return None
    pooled = sum(_total(s) or 0.0 for s in tr)
    own_t = (c.get("train") or {}).get("t")
    # the best of n pairs reaches t ~ sqrt(2 ln n) by chance alone, so the bar grows with the family (manager, 14.11.2026
    # midday: SC097|1Hour|JP225 in bot 3, t 1.44 - the best of 12 pairs that together lost -414R to 2025, 11 of them
    # negative - passed the fixed t >= 1 and was approved)
    bar = max(FAMILY_T, math.sqrt(2 * math.log(len(tr))) - 0.5)
    if pooled >= 0 or (own_t is not None and own_t >= bar):
        return None
    return (f"נבחר מתוך {len(tr)} מכשירים / טווחי זמן של אותה אסטרטגיה שיחד הפסידו עד 2025 ({pooled:+.1f}R), "
            f"והתוצאה שלו עד 2025 לא בולטת מעל המזל (t {own_t if own_t is not None else '-'}, צריך {bar:.2f}) - כנראה מזל, מחכה")


def tried_and_failed(bot, combo):
    """bots 1-4 / 7: the exits, filters and trade-management tests of the strategy have run (reports/agents/<bot>/exp)
    and no improvement card for it is strong or regular now."""
    if not combo or combo.count("|") != 2:
        return False
    ex = ROOT / "reports" / "agents" / bot / "exp"
    key = combo.replace("|", "_").replace("/", "-")
    ran = [f for f in ("exits", "filters", "manage") if (ex / f"{f}_{key}.json").exists()]
    if len(ran) < 2:                          # not tried enough yet (the manage family is new - two families are enough)
        return False
    for f in (ROOT / "reports" / "agents" / bot).glob("cards*.json"):
        try:
            cs = json.loads(f.read_text(encoding="utf-8")).get("cards", [])
        except Exception:  # noqa: BLE001
            continue
        for c in cs:
            if c.get("kind") == "improve" and (c.get("change") or {}).get("combo") == combo and c.get("tier") in ("strong", "regular"):
                return False
    return True


def _did(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def load():
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"decisions": []}


FIX_FILE = ROOT / "data/agents/fix_attempts.json"


def fix_attempts():
    """{"bot|combo-or-track": {"status": "trying" | "failed" | "fixed", "tried": [...], "at": ...}} - written by the
    bot agents when they try to repair a losing strategy (user, 12.11.2026: first fix, only then cut)."""
    try:
        return json.loads(FIX_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _target(c):
    ch = c.get("change") or {}
    return str(ch.get("combo") or ch.get("track") or "")


def _risk_card(c, bot):
    """The multiplier a bots 1-4 risk card sets for one strategy (0.5-1.5, what live/runner.py reads), else None."""
    ch = c.get("change") or {}
    if c.get("kind") != "risk" or bot not in RISK_BOTS or not ch.get("combo") or ch.get("kind") != "mult":
        return None
    try:
        v = float(ch.get("to"))
    except (TypeError, ValueError):
        return None
    if abs(v - 0.001) < 1e-9:
        # the instrument's minimum lot (agents/loss_review.MIN_LOT): no edge left, live data keeps coming (user,
        # 13.11.2026) - until then such a card was approved and never applied (rounded to 0.0, outside 0.5-1.5)
        return 0.001
    v = round(v, 2)
    if (c.get("detail") or {}).get("account_fit") and 0.1 <= v < 0.5:
        # the whole bot sized back into its own limits (bots 1-4 smart agent, 14.11.2026: SC001|1Min|JP225 at x0.7 took
        # bot 2's 2026 replay to a 10.3% fall and 7 days of -3% or worse); live/manager accepts any value up to 3
        return v
    return v if 0.5 <= v <= 1.5 else None


RULE_KINDS = ("tp", "sl", "hours", "dir", "vol")


def _rule_card(c, bot):
    """A bots 1-4 agent's card on one of a strategy's live rules (risk_review's tp / sl / hours / dir / vol,
    change {"combo", "kind": <rule>, "from", "to"}; "to" None = the rule is taken back, the strategy trades as tested).
    Returns (kind, value) - applied on the bots' computer like a risk card (live/manager.py risk_change) - else None.
    13.11.2026: such a card (bot 1 TT45|15Min|CSCO: the stop x0.6 of 07.11, worse than the strategy's own stop in the
    fair test on both periods) had no way in - it was judged a cut and then waited for nothing."""
    ch = c.get("change") or {}
    if c.get("kind") != "risk" or bot not in RISK_BOTS or not ch.get("combo") or ch.get("kind") not in RULE_KINDS:
        return None
    v, k = ch.get("to"), ch["kind"]
    if v is None:
        return k, None
    try:
        if k == "dir" and int(v) in (1, -1):
            return k, int(v)
        if k in ("tp", "sl") and (0.3 if k == "tp" else 0.5) <= float(v) < 1:
            return k, round(float(v), 2)
        if k == "hours" and len(v) == 2 and 0 <= int(v[0]) < int(v[1]) <= 24:
            return k, [int(v[0]), int(v[1])]
    except (TypeError, ValueError):
        return None
    return None


def _cuts(c):
    """A card that only cuts: removes a strategy, stops a track or lowers a risk."""
    ch = c.get("change") or {}
    if c.get("kind") == "remove":
        return True
    if c.get("kind") == "risk" and ch.get("kind") in RULE_KINDS:
        return False                         # a live rule changed / taken back is a fix, not a cut
    if c.get("kind") == "risk" and ((c.get("detail") or {}).get("account_fit") or ch.get("knob") == "RISK_SCALE"):
        # the whole bot sized into its own fixed limits (tal_gap.risk_scale, chosen on 2024-2025; bot 6's smart agent,
        # 13.11.2026 night): not a cut of a losing strategy - the gap tracks earn in R, but at today's risk the account
        # falls past its limits in the test (bot 6: FTMO Futures 1.5% a day / 4.5% trailing, failed in January 2026).
        # Fixing the strategies cannot fix the account's size, so "fix first" does not hold it (red line: the limits)
        return False
    if c.get("kind") == "risk":
        try:
            return float(ch.get("to", ch.get("value", 1))) < float(ch.get("from", 1))
        except (TypeError, ValueError):
            return True
    return False


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
    cur_risk = (_git_show("live/manager_state.json").get("risk") or {})
    never = {(x["bot"], x["combo"], x["kind"]) for x in rules.get("never", [])}
    fix_not_cut = set(rules.get("fix_not_cut", []))
    # THE RULE (user, 12.11.2026): "first fix and improve the strategies; only if that fails, lower the risk or remove".
    # A cut is approved only after the bot's agent tried to fix that strategy and failed (data/agents/fix_attempts.json)
    fixes = fix_attempts()
    fix_first = rules.get("fix_first", True)

    def fix_failed(bot, target):
        x = fixes.get(f"{bot}|{target}") or fixes.get(f"{bot}|*")
        if bool(x) and x.get("status") == "failed":
            return True
        # the experiments themselves are the attempt (13.11.2026: the agents tested and failed but did not record it,
        # and the manager waited forever): every family ran on the strategy and none brought an approvable fix
        return tried_and_failed(bot, target)
    # an undone recommendation is not made again for 30 days, even when a later sample brings it back
    never |= {(d["bot"], d.get("combo"), d["kind"]) for d in old.values() if d["id"] in uundo and d["type"] == "risk"}
    # the watch (agents/watch.py) took it back: like an undo, not made again (unless the user undid the revert)
    reverted = {d["of"] for d in old.values() if d.get("decision") == "reverted" and d["id"] not in uundo}
    never |= {(d["bot"], d.get("combo"), d["kind"]) for d in old.values() if d["id"] in reverted and d["type"] == "risk"}
    uundo = uundo | reverted
    new = []

    def add(d):
        if d["id"] in old:
            return old[d["id"]]
        d.update(at=at, by="manager")
        old[d["id"]] = d
        new.append(d)
        return d

    try:
        built = {k for k, v in json.loads((ROOT / "data/agents/replies.json").read_text(encoding="utf-8")).items()
                 if isinstance(v, dict) and v.get("built")}
    except Exception:  # noqa: BLE001
        built = set()
    # the user's own "bring them back" (12.11.2026: "first fix - bring back what was cut and try to make money with it"):
    # data/agents/user_reverts.json lists cut decisions to take back; the bots' computer restores the value before
    try:
        for rid in json.loads((ROOT / "data/agents/user_reverts.json").read_text(encoding="utf-8")).get("ids", []):
            o = old.get(rid)
            if o and o.get("decision") == "approved" and o.get("type") == "risk":
                add({"id": _did("userrevert", rid), "bot": o["bot"], "type": "risk", "kind": o["kind"], "combo": o.get("combo"),
                     "title": "חוזר לבוט: " + o["title"], "decision": "reverted", "of": rid,
                     "why": "טל: קודם מתקנים ומטייבים - ההפחתה / ההסרה בוטלה, והסוכן של הבוט ינסה להרוויח איתה",
                     "effect": "הסיכון חוזר לערך שלפני ההחלטה, תוך כ־10 דקות"})
    except FileNotFoundError:
        pass
    # ONE change per strategy at a time (13.11.2026: TV58|1Hour|USDCAD bot 3 got a new exit plan and a market filter
    # approved in the same minute, each tested alone against the old version - the pair never tested together, and the
    # exit plan replaced one built 5 hours before). Of the new improvement cards of one strategy, only the one with the
    # most total R on the choosing period (to 2025) goes in now; the others wait for the next experiment, run on top of the built one.
    # A card whose experiment ran before a change on its strategy was built is stale the same way: it waits.
    try:
        reps = json.loads((ROOT / "data/agents/replies.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        reps = {}
    built_at = {}                                               # (bot, combo) -> the last time a change was built
    for bot, b in bots.items():
        for c in b.get("cards", []):
            r = reps.get(c["id"]) if isinstance(reps.get(c["id"]), dict) else {}
            combo = (c.get("change") or {}).get("combo")
            if r.get("built") and combo and c.get("kind") == "improve":
                k = (bot, combo)
                built_at[k] = max(built_at.get(k, ""), str(r.get("at") or ""))
    for o in old.values():                                      # cards built before that left the cards list
        if o.get("type") == "card" and o.get("kind") == "improve" and o.get("decision") == "approved" \
                and (reps.get(o.get("card")) or {}).get("built") and o["title"].count("|") >= 2:
            k = (o["bot"], o["title"].lstrip("⭐ ").split(":")[0].strip())
            built_at[k] = max(built_at.get(k, ""), str((reps.get(o.get("card")) or {}).get("at") or ""))
    best_now = {}                                               # (bot, combo) -> the new card with the most total gain
    for bot, b in bots.items():
        for c in b.get("cards", []):
            combo = (c.get("change") or {}).get("combo")
            if c.get("kind") != "improve" or c.get("tier") not in ("strong", "regular") or not combo \
                    or _did("card", c["id"]) in old or (ucards.get(c["id"]) or {}).get("decision"):
                continue
            gain = _total(c.get("train")) or 0.0                # chosen on the period to 2025 (objective.md)
            if (bot, combo) not in best_now or gain > best_now[(bot, combo)][1]:
                best_now[(bot, combo)] = (c["id"], gain)
    # 1) the agents' cards (all bots)
    for bot, b in bots.items():
        for c in b.get("cards", []):
            if (ucards.get(c["id"]) or {}).get("decision"):           # the user decided it himself
                continue
            did = _did("card", c["id"])
            prev = old.get(did)
            if c["tier"] not in ("strong", "regular"):
                # approved earlier, judged lower now (a stricter rule): not built yet -> withdrawn, never built;
                # already built -> stays approved (the bot agent proposes a revert card if it hurts)
                if prev and prev.get("decision") == "approved" and did not in uundo:
                    # a new strategy not built yet is withdrawn too (13.11.2026: SC021|5Min|XAUUSD for bot 5 was approved
                    # at 00:25, the +3R rule judged it 'watch' at 00:29, and it stayed "approved - to be built" because
                    # only improvement cards were ever withdrawn). Only bot 5's: bot 6's approved scout strategies are
                    # its fix attempts on its own world (agent_bot6._approved_scout), they stay approved while tested.
                    if c["id"] in built or (c["kind"] != "improve" and not (c["kind"] == "new" and bot == "bot5")):
                        c["status"], c["by"], c["decided"], c["decision_id"] = "approved", "manager", prev["at"], did
                    else:
                        prev.update(decision="withdrawn", withdrawn_at=at)
                        add({"id": _did("withdraw", c["id"]), "bot": bot, "type": "card", "card": c["id"],
                             "kind": c["kind"], "tier": c["tier"], "title": "המנהל חזר בו: " + c["title"],
                             "decision": "withdrawn", "why": c["verdict"], "effect": "לא ייבנה בבוט", "of": did})
                continue
            if did in uundo:
                c["status"], c["verdict"] = "rejected", c["verdict"] + " · בוטל על ידך"
                continue
            if prev and prev.get("decision") == "approved" and c["kind"] in ("new", "improve") and not prev.get("whole_bot") \
                    and c["id"] not in built and not _in_bot(bot, c):
                # approved without the whole-bot check (it said "unknown" - manager, 14.11.2026: all 21 new / improve
                # cards of bots 1-4 since 13.11, and bot 7's 6 scout strategies of 04:18) and not built yet: checked now
                # that it can run; one that adds no 2026 profit to the whole bot, or only risk, is withdrawn, never built
                try:
                    from agents import portfolio_check as PC
                    whole = PC.check(bot, c)
                except Exception:  # noqa: BLE001
                    whole = None
                if whole is not None and not whole["ok"]:
                    prev.update(decision="withdrawn", withdrawn_at=at)
                    add({"id": _did("withdraw", c["id"]), "bot": bot, "type": "card", "card": c["id"],
                         "kind": c["kind"], "tier": c["tier"], "title": "המנהל חזר בו: " + c["title"],
                         "decision": "withdrawn", "why": "בדיקת הבוט השלם (לא רצה כשאושר): " + whole["why"],
                         "effect": "לא ייבנה בבוט", "of": did, "whole_bot": {k: whole[k] for k in ("before", "after")}})
                    c["status"], c["by"], c["verdict"] = "rejected", "manager", c["verdict"] + " · " + whole["why"]
                    continue
                if whole is not None:
                    prev["whole_bot"] = {k: whole[k] for k in ("before", "after")}
            ccombo = (c.get("change") or {}).get("combo")
            if did not in old and c["kind"] == "improve" and ccombo:
                if best_now.get((bot, ccombo), (c["id"],))[0] != c["id"]:
                    c["status"], c["verdict"] = "watch", (c["verdict"] + " · מחכה: שינוי אחר באסטרטגיה הזו נכנס עכשיו - "
                                                         "ייבדק שוב מעליו בניסוי הבא")
                    continue
                exp_at = str((c.get("detail") or {}).get("exp_at") or c.get("made") or "")
                if exp_at and built_at.get((bot, ccombo), "") > exp_at:
                    c["status"], c["verdict"] = "watch", (c["verdict"] + " · מחכה: שינוי אחר באסטרטגיה הזו נבנה אחרי הניסוי - "
                                                         "ייבדק שוב מעליו בניסוי הבא")
                    continue
            if fix_first and _cuts(c) and bot not in fix_not_cut and not fix_failed(bot, _target(c)):
                c["status"], c["verdict"] = "watch", (c["verdict"] + " · קודם מתקנים: הסוכן של הבוט צריך לנסות לתקן את האסטרטגיה, "
                                                     "ורק אם לא יצליח - הפחתה או הסרה")
                continue
            if bot in fix_not_cut and _cuts(c):
                # the user (12.11.2026, bot 7): "not to lower or remove - to fix, to improve the strategies, to find
                # the way to make money" - a card that only cuts is not approved; the bot agent must bring a fix
                c["status"], c["verdict"] = "rejected", c["verdict"] + " · טל: בבוט הזה לא מורידים ולא מסירים - מתקנים"
                continue
            from agents import builder as BLD
            buildable = BLD.buildable({**c, "bot": bot})
            risk_card = _risk_card(c, bot)
            rule_card = _rule_card(c, bot)
            what = ("לידיעה בלבד: אין מה לבנות, הסוכן של הבוט ממשיך לעקוב" if c["kind"] == "info" else
                    "נבנה בבוט אוטומטית ויותקן תוך כ־20 דקות" if buildable else
                    "יוחל במחשב הבוטים תוך כ־10 דקות" if risk_card or rule_card else
                    "שינוי בקוד: הסוכן של הבוט יבנה אותו בריצת הלילה")
            why, star, mult, whole = c["verdict"], False, None, None
            is_new = did not in old
            tot_new, tot_base = _total(c.get("unseen")), _total(c.get("baseline"))
            if is_new and c["kind"] == "improve" and tot_new is not None and tot_base is not None and tot_new <= tot_base:
                # THE GOAL (user, 12.11.2026, data/agents/objective.md): the TOTAL profit, not R per trade - a change
                # goes in only if its 2026 total beats what runs now (12.11: a built change earned more a trade, but
                # fewer trades made less in all - it had to be taken back)
                why = f"ב־2026 בסך הכל {tot_new:+.1f}R מול {tot_base:+.1f}R היום - לא מגדיל את הרווח הכולל"
                add({"id": did, "bot": bot, "type": "card", "card": c["id"], "kind": c["kind"], "tier": c["tier"],
                     "title": c["title"], "decision": "declined", "why": why, "effect": "לא ייבנה בבוט"})
                c["status"], c["by"], c["verdict"] = "rejected", "manager", c["verdict"] + " · " + why
                continue
            fam = _family_pick(b.get("cards", []), c) if is_new and c["kind"] == "new" else None
            if fam:
                c["status"], c["verdict"] = "watch", c["verdict"] + " · " + fam
                continue
            if is_new and c["kind"] in ("improve", "new"):
                # the whole bot with and without it, on 2026 (agents/portfolio_check.py; user, 12.11.2026)
                try:
                    from agents import portfolio_check as PC
                    whole = PC.check(bot, c)
                except Exception as e:  # noqa: BLE001
                    whole, why = None, why + f" · בדיקת הבוט השלם נכשלה ({str(e)[:80]})"
                if whole is not None and not whole["ok"]:
                    add({"id": did, "bot": bot, "type": "card", "card": c["id"], "kind": c["kind"], "tier": c["tier"],
                         "title": c["title"], "decision": "declined", "why": whole["why"], "effect": "לא ייבנה בבוט",
                         "whole_bot": {k: whole[k] for k in ("before", "after")}})
                    c["status"], c["by"], c["verdict"] = "rejected", "manager", c["verdict"] + " · " + whole["why"]
                    continue
                ncombo = (c.get("change") or {}).get("combo")
                if whole is None and c["kind"] == "new" and ncombo:
                    # one new strategy in one bot until its whole-bot check (manager, 12.11.2026: SC001|1Min|JP225, a thin
                    # +0.02R a trade, was approved in bots 7, 2 and 3 within 90 minutes from the same backtest - three
                    # accounts on one unproven edge). Another bot takes it when its own whole-bot check shows more profit.
                    others = sorted({o["bot"] for o in old.values() if o.get("type") == "card" and o.get("kind") == "new"
                                     and o.get("decision") == "approved" and o["bot"] != bot and o["id"] not in uundo
                                     and ncombo in (o.get("title") or "")})
                    if others:
                        c["status"], c["verdict"] = "watch", (c["verdict"] + f" · כבר רץ ב{', '.join('בוט ' + o[3:] for o in others)} - נכנס גם לבוט הזה "
                                                              "רק כשבדיקת הבוט השלם תראה שהוא מגדיל את הרווח הכולל")
                        continue
                if whole is not None:
                    why += " · " + whole["why"]
                    mult = whole.get("mult")
                elif bot in RISK_BOTS:
                    why += " · בלי בדיקת בוט שלם עדיין (העסקאות של חלק מהאסטרטגיות עוד לא נשמרו)"
                # "maybe luck" (user, 12.11.2026): a small edge on few trades goes in at a lowered risk with a star,
                # and the watch decides later: back to the full risk when it keeps winning, lower when it doesn't
                star = maybe_luck(c)
                if star:
                    mult = min(mult or 1.0, STAR_MULT)
                    scombo = (c.get("change") or {}).get("combo")
                    fl = star_floor(c, float(((cur_risk.get(bot) or {}).get("mult") or {}).get(scombo, 1.0))) if scombo else None
                    if fl is not None and fl > mult and not (whole and whole.get("mult") and whole["mult"] < 1):
                        mult = fl                # not below what keeps the 2026 total at least what runs now
                    why += f" · ⭐ אולי מזל: נכנס בסיכון ×{mult:g} ובמעקב מוקפד"
            d = add({"id": did, "bot": bot, "type": "card", "card": c["id"], "kind": c["kind"], "tier": c["tier"],
                     "title": ("⭐ " if star else "") + c["title"], "decision": "approved", "why": why, "effect": what,
                     "star": star, **({"whole_bot": {k: whole[k] for k in ("before", "after")}} if whole else {})})
            c["status"], c["by"], c["decided"], c["decision_id"] = "approved", "manager", d["at"], did
            combo = (c.get("change") or {}).get("combo")
            if is_new and rule_card:
                rk, rv = rule_card
                cur_v = ((cur_risk.get(bot) or {}).get(rk) or {}).get(combo)
                if not _same(cur_v, rv):
                    add({"id": _did("cardrule", c["id"]), "bot": bot, "type": "risk", "kind": rk, "combo": combo,
                         "title": c["title"], "decision": "approved", "why": why, "smart_rule": True,
                         "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "card_decision": did,
                         "action": {"bot": bot, "combo": combo, "kind": rk, "value": rv, "prev": cur_v}})
                continue
            if is_new and risk_card:
                # a bot agent's risk card for one strategy of bots 1-4 (change {"combo", "kind": "mult", "from", "to"}):
                # applied by the bots' computer like the star's risk, linked to the card - an undo of the card brings
                # the value before back (live/manager.py auto_apply). Until 13.11.2026 such a card was approved and
                # then waited for a nightly session that had no way to apply it.
                cur_m = float(((cur_risk.get(bot) or {}).get("mult") or {}).get(combo, 1.0))
                if abs(risk_card - cur_m) > 1e-9:
                    add({"id": _did("cardrisk", c["id"]), "bot": bot, "type": "risk", "kind": "mult", "combo": combo,
                         "title": f"{combo}: סיכון ×{risk_card:g}", "decision": "approved", "why": why,
                         "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "card_decision": did,
                         "action": {"bot": bot, "combo": combo, "kind": "mult", "value": risk_card,
                                    "prev": None if cur_m == 1.0 else cur_m}})
                continue
            if not is_new or not (mult and mult < 1 and buildable and bot in RISK_BOTS and combo):
                continue
            # the lowered risk of the new change, applied by the bots' computer like any risk decision; undone or
            # taken back together with the card (live/manager.py auto_apply)
            cur_m = float(((cur_risk.get(bot) or {}).get("mult") or {}).get(combo, 1.0))
            # "maybe luck" is one discount per strategy, not one per card: a second starred card for a strategy
            # already at x0.7 does not take it to x0.49 (12.11.2026: those fell under the 0.5 floor and failed)
            star_only = star and not (whole and whole.get("mult") and whole["mult"] < 1)
            value = round(min(cur_m, mult) if star_only else max(0.5, mult * cur_m), 2)
            if value >= cur_m:
                continue
            add({"id": _did("cardrisk", c["id"]), "bot": bot, "type": "risk", "kind": "mult", "combo": combo,
                 "title": f"{combo}: סיכון ×{mult:g}" + (" (⭐ אולי מזל)" if star else " (בדיקת הבוט השלם)"),
                 "decision": "approved", "why": why, "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "card_decision": did,
                 "star": star, "action": {"bot": bot, "combo": combo, "kind": "mult", "value": value,
                                          "prev": None if cur_m == 1.0 else cur_m}})
    # 1b) a "maybe luck" risk already applied under the total floor (star_floor, manager 13.11.2026 night) is raised to
    # the floor once, linked to its card like the star's own risk (an undo of the card brings the value before back)
    cards_by_id = {c["id"]: c for b in bots.values() for c in b.get("cards", [])}
    for o in list(old.values()):
        a = o.get("action") or {}
        if not (o.get("type") == "risk" and o.get("star") and o.get("decision") == "approved" and a.get("kind") == "mult"
                and o.get("card_decision") and o["id"] not in uundo and o["card_decision"] not in uundo):
            continue
        cd = old.get(o["card_decision"]) or {}
        c = cards_by_id.get(cd.get("card"))
        if not c or cd.get("decision") != "approved":
            continue
        cur_v = ((cur_risk.get(o["bot"]) or {}).get("mult") or {}).get(a["combo"])
        if cur_v is None or abs(float(cur_v) - float(a["value"])) > 1e-9:
            continue                                            # changed since (by the watch, a later card or the user)
        fl = star_floor(c, float(a.get("prev") or 1.0))
        if fl is None or fl <= float(a["value"]) + 1e-9:
            continue
        add({"id": _did("starfloor", o["id"]), "bot": o["bot"], "type": "risk", "kind": "mult", "combo": a["combo"],
             "title": f"{a['combo']}: סיכון ×{fl:g} במקום ×{float(a['value']):g}",
             "decision": "approved", "card_decision": o["card_decision"], "star": True,
             "why": (f"ב־2026 השינוי הרוויח {_total(c.get('unseen')):+.1f}R מול {_total(c.get('baseline')):+.1f}R לפניו, אבל "
                     f"בסיכון ×{float(a['value']):g} הסך הכולל יוצא נמוך ממה שהיה. ×{fl:g} הוא הסיכון הנמוך ביותר ששומר "
                     "על הסך הכולל של היום - ועדיין זהיר כי אולי מזל"),
             "effect": "יוחל במחשב הבוטים תוך כ־10 דקות",
             "action": {"bot": o["bot"], "combo": a["combo"], "kind": "mult", "value": fl, "prev": float(a["value"])}})
    # 2) the bot agents' risk / removal recommendations (bots 1-4)
    try:
        risk = json.loads((ROOT / "reports/trade_review/risk.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        risk = {}
    cur = (_git_show("live/manager_state.json").get("risk") or {})
    sample = risk.get("sample")
    # a rule a bot agent's card changed or took back (above) is not brought back by the morning review for 30 days
    smart_rules = {(d["bot"], d.get("combo"), d["kind"]) for d in list(old.values()) + new
                   if d.get("type") == "risk" and d.get("smart_rule") and d.get("decision") == "approved"
                   and d.get("card_decision") not in uundo}
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
                if (bot, combo, kind) in never or (bot, combo, kind) in smart_rules:
                    continue
                did = _did("risk", bot, combo, kind, json.dumps(val), sample)
                if did in uundo:
                    continue
                if fix_first and (kind == "remove" or (kind == "mult" and float(val) < mult)) and not fix_failed(bot, combo):
                    continue                                          # first fix (user, 12.11.2026); cut only after that failed
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
    # 3) bot 7's account reset (user, 12.11.2026): the manager decides it on a material change of bot 7's strategies,
    # stops resetting by itself after two stable weeks (then it asks), and archives the trades first (agents/reset_rule.py)
    try:
        from agents import reset_rule as RR
        RR.check(add, _did, _git_show, now, ucards, rules.get("bot7_reset_ok", []))
    except Exception as e:  # noqa: BLE001
        print(f"bot 7 reset rule failed: {e}")
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
