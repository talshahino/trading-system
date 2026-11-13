"""The daily loss review (user, 12.11.2026): every day a bot ends with a loss, a professional report of what lost, why,
and how not to lose that way again - and the manager makes the bot's agent act on it, so losses shrink over time.

Runs on the bots' computer in the continuous layer (every 10 minutes); a day is reviewed once it is over (FTMO's day,
midnight Prague time), for every bot whose closed trades of that day add up to a loss. Per report
(reports/agents/losses/<bot>/<YYYY-MM-DD>.json, the list in reports/agents/losses/index.json, read by the app):

  1. the day in money     net P&L in $ and in % of the day's opening balance, how much of the bot's daily limit (4.5%)
                          and of FTMO's (5%) it used, trades / winners / losers, gross vs costs, largest loss, profit
                          factor of the day
  2. expected or not      the day's result in R against what the backtest of each traded strategy expects (sum of the
                          expected R, its spread): a normal bad day (within the noise) or an abnormal one (z < -2)
  3. every losing trade   strategy, direction, entry / exit, result in R, the exit reason, how far it went in its favour
                          first (MFE) and against (MAE), the cost - and its CAUSE, one of:
                            entry       stopped without ever moving in its favour (MFE < 0.3R): the entry or the direction
                            management  was 1R+ (or 60%+ of the way to the target) in profit and still ended in a loss
                            execution   lost more than 1.15R: slippage / a gap through the stop / costs
                            rules       closed by a rule (restricted news, session end, market break, weekend, holiday)
                            time        a time exit at a loss
                            normal      a plain stop after a fair try (part of the strategy's normal losses)
  4. the day as a book    losers on the same side and asset class together (correlation), the share of the loss that
                          came from one strategy / one instrument
  5. recurring            the same strategy or the same cause in 3+ loss days of the last 30: a pattern, not luck
  6. actions              for every cause, what is done, at once and by whom:
                            management -> the strategy's breakeven / partial-exit test runs first (agent_portfolio
                                          "manage"), and the bot's agent gets the task
                            entry      -> its market-filter and entry tests run first ("filters", "exits")
                            execution  -> the bot's agent checks slippage / spread / gap handling of that instrument
                            rules      -> the bot's agent checks whether the strategy should hold through such events
                            correlated -> the bot's agent proposes a cap on same-side exposure (whole-bot check)
                            recurring  -> "first fix" (user, 12.11.2026): the bot's agent must fix the strategy; only
                                          if it cannot, lower its risk
                          the tasks go to data/agents/replies.json "_tasks" (the bots' nightly agents do them first,
                          and answer there), the first-run tests to data/agents/priority.json (the mechanical agents)
  7. trend                loss days, average loss day and the share of each cause in the last 7 vs the 7 before:
                          are the losses getting smaller?

  python -m agents.loss_review
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from agents import health as H

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "losses"
TASKS = ROOT / "data" / "agents" / "replies.json"
PRIO = ROOT / "data" / "agents" / "priority.json"
TZ = "Europe/Prague"                     # FTMO's day
BOT_DAILY, FTMO_DAILY = 4.5, 5.0
BACK_DAYS = 14                           # days reviewed back (first run, or a computer that was off)
NAMES = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4", "bot5": "בוט 5 - סקאלפינג",
         "bot6": "בוט 6 · חוזים", "bot7": "בוט 7 · פרסומים"}
CAUSE_HE = {"entry": "כניסה / כיוון", "management": "ניהול העסקה", "execution": "ביצוע (החלקה / פער / עלויות)",
            "rules": "סגירה בגלל כלל", "time": "יציאת זמן", "normal": "סטופ אחרי תנועה קטנה לטובה"}
RULE_WORDS = ("news", "restricted", "session", "break", "weekend", "holiday", "חדשות", "סשן", "הפסקה", "סוף שבוע", "חג",
              "סגירה בשעה קבועה")


def _read(p, d):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return d


def _write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def trade_rows(state):
    rows = []
    for c in state.get("closed") or []:
        o, cl, sl0 = _f(c.get("open_px")), _f(c.get("close_px")), _f(c.get("sl0"))
        net, gross = _f(c.get("net")), _f(c.get("gross"))
        if None in (o, cl, sl0, net) or not c.get("close_t"):
            continue
        side = 1 if c.get("side") == "buy" else -1
        risk = abs(o - sl0)
        if risk <= 0:
            continue
        r_px = side * (cl - o) / risk
        r = r_px * (net / gross) if gross and gross * net > 0 else (net / abs(gross) * abs(r_px) if gross else r_px)
        tp0, mfe, mae = _f(c.get("tp0")), _f(c.get("mfe")), _f(c.get("mae"))
        rows.append({
            "combo": c.get("combo"), "symbol": c.get("symbol"), "side": side, "open_t": c.get("open_t"),
            "close_t": c.get("close_t"), "open_px": o, "close_px": cl, "sl0": sl0, "tp0": tp0, "net": net,
            "gross": gross, "costs": round((gross or net) - net, 2) if gross is not None else None, "r": round(r, 3),
            "mfe_r": round(side * (mfe - o) / risk, 2) if mfe is not None else None,
            "mae_r": round(side * (o - mae) / risk, 2) if mae is not None else None,
            "tp_way": round(side * (mfe - o) / (side * (tp0 - o)), 2) if (mfe is not None and tp0 and side * (tp0 - o) > 0) else None,
            "why": str(c.get("why") or c.get("reason") or "")})
    t = pd.DataFrame(rows)
    if len(t):
        t["close"] = pd.to_datetime(t["close_t"], utc=True, format="ISO8601")
        t["day"] = t["close"].dt.tz_convert(TZ).dt.date.astype(str)
    return t


def cause(x):
    w = x["why"].lower()
    if any(k in w for k in RULE_WORDS):
        return "rules"
    if x["r"] < -1.15:
        return "execution"
    if (x["mfe_r"] is not None and x["mfe_r"] >= 1.0) or (x["tp_way"] is not None and x["tp_way"] >= 0.6):
        return "management"
    if "זמן" in w or "time" in w:
        return "time"
    if x["mfe_r"] is not None and x["mfe_r"] < 0.3:
        return "entry"
    return "normal"


def day_open_balance(state, day, fallback):
    """The balance at the start of the FTMO day, from the bot's equity curve."""
    start = pd.Timestamp(day).tz_localize(TZ).tz_convert("UTC")
    bal = None
    for p in state.get("equity_curve") or []:
        try:
            t = pd.Timestamp(p[0])
            t = t.tz_localize("UTC") if t.tzinfo is None else t
        except Exception:  # noqa: BLE001
            continue
        if t <= start:
            bal = float(p[1])
        else:
            break
    return bal or fallback


def expected(bot, day_trades, exp):
    m = s2 = 0.0
    k = 0
    for c in day_trades["combo"]:
        e = H.expect_for(bot, c, exp)
        if not e or e.get("mean") is None:
            continue
        m += e["mean"]
        s2 += (e.get("sd") or 1.2) ** 2
        k += 1
    return (m, math.sqrt(s2), k) if k else (None, None, 0)


def review(bot, state, day, t, history, exp):
    d = t[t["day"] == day].copy()
    d["cause"] = d.apply(cause, axis=1)
    initial = float(state.get("initial") or 100_000)
    bal0 = day_open_balance(state, day, float(state.get("balance") or initial))
    net = float(d["net"].sum())
    pct = net / bal0 * 100
    los, win = d[d["net"] < 0], d[d["net"] > 0]
    gl, gp = float(-los["net"].sum()), float(win["net"].sum())
    em, esd, ek = expected(bot, d, exp)
    z = ((float(d["r"].sum()) - em) / esd) if esd else None
    normal = z is None or z > -2
    # the book: same side / same asset class together
    from agents import ftmo_universe as U
    los = los.assign(cls=[U.CLASS.get(str(c).split("|")[-1], "?") for c in los["combo"]])
    groups = los.groupby(["cls", "side"]).agg(n=("net", "size"), loss=("net", "sum")).reset_index()
    corr = [{"class": g.cls, "side": "קנייה" if g.side > 0 else "מכירה", "trades": int(g.n), "loss": round(float(g.loss), 2)}
            for g in groups.itertuples() if g.n >= 3]
    by_combo = los.groupby("combo")["net"].sum().sort_values()
    top = by_combo.index[0] if len(by_combo) else None
    top_share = round(float(by_combo.iloc[0]) / -gl * 100, 1) if len(by_combo) and gl else None
    # recurring over the last 30 days of reports
    rec_combo, rec_cause = {}, {}
    for h in history:
        for x in h.get("losers", []):
            rec_combo[x["combo"]] = rec_combo.get(x["combo"], 0) + 1
        for c_, n_ in (h.get("causes") or {}).items():
            rec_cause[c_] = rec_cause.get(c_, 0) + (1 if n_ else 0)
    for c in set(los["combo"]):
        rec_combo[c] = rec_combo.get(c, 0) + 1
    causes = los["cause"].value_counts().to_dict()
    for c_ in causes:
        rec_cause[c_] = rec_cause.get(c_, 0) + 1
    recurring = sorted([c for c in set(los["combo"]) if rec_combo.get(c, 0) >= 3])
    rec_causes = sorted([c for c in causes if rec_cause.get(c, 0) >= 3])
    losers = [{"combo": x.combo, "side": "קנייה" if x.side > 0 else "מכירה", "open_t": x.open_t, "close_t": x.close_t,
               "entry": x.open_px, "exit": x.close_px, "stop": x.sl0, "target": x.tp0, "r": x.r, "net": round(x.net, 2),
               "costs": x.costs, "mfe_r": x.mfe_r, "mae_r": x.mae_r, "exit_reason": x.why, "cause": x.cause,
               "cause_he": CAUSE_HE[x.cause], "recurring": rec_combo.get(x.combo, 0) >= 3}
              for x in los.sort_values("net").itertuples()]
    actions = plan_actions(bot, day, losers, corr, recurring, normal, pct)
    r_usd = bal0 * float(state.get("risk") or 0.004)            # 1R in dollars (the bot's risk per trade)
    CAUSE_PLAIN = {"entry": "נעצרו בלי לזוז לטובה בכלל - הבעיה בכניסה או בכיוון",
                   "management": "היו ברווח ונגמרו בהפסד - הבעיה בניהול העסקה",
                   "execution": "הפסידו יותר מהסיכון המתוכנן - הבעיה בביצוע (החלקה / פער / עלויות)",
                   "rules": "נסגרו בגלל כלל (חדשות / סוף מסחר / הפסקת שוק)",
                   "time": "נסגרו לפי זמן בהפסד", "normal": "נעצרו בסטופ אחרי תנועה קטנה לטובה"}
    lines = [
        f"מה קרה: {NAMES.get(bot, bot)} הפסיד {abs(net):,.0f}$ ב־{day} ({pct:+.2f}% מהחשבון). "
        f"{len(d)} עסקאות: {len(win)} הרוויחו {gp:,.0f}$, {len(los)} הפסידו {gl:,.0f}$.",
        (f"מול הצפי: הבדיקה צפתה ליום כזה {em:+.1f}R ({em * r_usd:+,.0f}$). בפועל {float(d['r'].sum()):+.1f}R. "
         f"הפער: {float(d['r'].sum()) - em:+.1f}R ({(float(d['r'].sum()) - em) * r_usd:+,.0f}$) - צריך לסגור אותו.") if esd else
        "מול הצפי: לאסטרטגיות של היום אין עדיין צפי מהבדיקה.",
        "למה: " + "; ".join(f"{v} עסקאות {CAUSE_PLAIN[k]}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) + ".",
    ]
    if top:
        lines.append(f"ההפסד הגדול ביותר: {top}, {abs(by_combo.iloc[0]):,.0f}$ ({top_share:g}% מכל ההפסד).")
    if corr:
        lines.append("הפסדים שבאו יחד: " + "; ".join(f"{c['trades']} עסקאות {c['side']} ב{c['class']} ({abs(c['loss']):,.0f}$)" for c in corr) + ".")
    if recurring:
        lines.append("חוזרות על עצמן: " + ", ".join(recurring) + " - הפסידו ב־3 ימים או יותר בחודש האחרון.")
    return {"bot": bot, "day": day, "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "summary": lines, "pnl": round(net, 2), "pnl_pct": round(pct, 3), "open_balance": round(bal0, 2),
            "limit_used": {"bot": round(abs(min(0, pct)) / BOT_DAILY * 100, 1), "ftmo": round(abs(min(0, pct)) / FTMO_DAILY * 100, 1)},
            "trades": int(len(d)), "winners": int(len(win)), "losers_n": int(len(los)),
            "gross_profit": round(gp, 2), "gross_loss": round(gl, 2), "profit_factor": round(gp / gl, 2) if gl else None,
            "costs": round(float(d["costs"].dropna().sum()), 2), "day_r": round(float(d["r"].sum()), 2),
            "expected_r": round(em, 2) if em is not None else None, "expected_sd": round(esd, 2) if esd else None,
            "z": round(z, 2) if z is not None else None, "normal": normal, "causes": causes, "correlated": corr,
            "top_loser": top, "top_share": top_share, "recurring": recurring, "recurring_causes": rec_causes,
            "losers": losers, "actions": actions}


def plan_actions(bot, day, losers, corr, recurring, normal, pct):
    acts = []
    port = bot in ("bot1", "bot2", "bot3", "bot4", "bot7")
    for x in losers:
        c, k = x["combo"], x["cause"]
        if k == "management":
            acts.append({"combo": c, "cause": k, "priority": ["manage"] if port else [],
                         "task": f"{c}: היה ב־{x['mfe_r']}R רווח ונסגר ב־{x['r']}R. לבדוק נקודת איזון מוקדמת יותר / יציאה חלקית / יעד קרוב יותר, ולהציע כרטיס."})
        elif k == "entry":
            acts.append({"combo": c, "cause": k, "priority": ["filters", "exits"] if port else [],
                         "task": f"{c}: נעצר בלי לזוז לטובה (MFE {x['mfe_r']}R). לבדוק אם הכניסה נגד המגמה הגבוהה / במצב שוק לא מתאים / מוקדמת מדי, ולהציע מסנן או כניסה אחרת."})
        elif k == "execution":
            acts.append({"combo": c, "cause": k, "priority": [],
                         "task": f"{c}: הפסד {x['r']}R - יותר מהסיכון המתוכנן. לבדוק החלקה, מרווח ברגע הסגירה ופער מעבר לסטופ, ולתקן בקוד או בגודל הפוזיציה."})
        elif k == "rules":
            acts.append({"combo": c, "cause": k, "priority": [],
                         "task": f"{c}: נסגר בגלל כלל ({x['exit_reason']}) ב־{x['r']}R. לבדוק אם עדיף לא להיכנס לפני אירוע כזה, או לצאת מוקדם יותר."})
        elif k == "time":
            acts.append({"combo": c, "cause": k, "priority": ["exits"] if port else [],
                         "task": f"{c}: יציאת זמן בהפסד. יציאות לפי זמן בלבד לא רצויות (טל) - להחליף ביציאה חכמה."})
    for c in recurring:
        acts.append({"combo": c, "cause": "recurring", "priority": ["exits", "filters", "manage"] if port else [],
                     "task": f"{c}: הפסד חוזר (3+ ימים בחודש). קודם לתקן (טל: 'קודם לתקן ולשפר'): לנתח את כל ההפסדים שלו, למצוא את המכנה המשותף ולהציע תיקון; רק אם אין תיקון - להוריד סיכון."})
    for g in corr:
        acts.append({"combo": None, "cause": "correlated", "priority": [],
                     "task": f"{g['trades']} הפסדים יחד ב{g['side']} של {g['class']} ({g['loss']:,.0f}$): להציע תקרה לחשיפה לאותו צד באותו סוג נכס, לבדוק בבדיקת הבוט השלם."})
    if not normal:
        acts.append({"combo": None, "cause": "abnormal", "priority": [],
                     "task": f"היום ({pct:+.2f}%) רחוק מאוד מהצפי של הבדיקה: לבדוק קודם תקלה בקוד או בנתונים (פער בין החי לבדיקה), ואז לתקן את האסטרטגיות."})
    seen, out = set(), []
    for a in acts:
        k = (a["combo"], a["cause"])
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


def card_effect(card, months, bot_risk_pct, mult=None):
    """% of the account a month the change is worth, from the card's own test on the unseen 2026 data:
    improve - (new mean - old mean); new - the new strategy's mean; remove / risk down - the losses no longer taken."""
    u, b = card.get("unseen") or {}, card.get("baseline") or {}
    if u.get("mean") is None:
        return None
    per_month = (u.get("n") or 0) / months * (card.get("risk_pct") or bot_risk_pct)
    k = card.get("kind")
    if k == "improve" or (k == "risk" and (card.get("change") or {}).get("kind") in ("tp", "sl", "hours", "dir", "vol")):
        # a live rule changed / taken back (13.11.2026) is worth its own test's gain, like an improvement
        return round(((u["mean"] or 0) - (b.get("mean") or 0)) * per_month, 2)
    if k == "new":
        return round((u["mean"] or 0) * per_month, 2)
    if k in ("remove", "risk"):
        m = mult if mult is not None else ((card.get("change") or {}).get("value") if k == "risk" else 0)
        m = 0 if m is None else float(m)
        return round(-(u["mean"] or 0) * (1 - m) * per_month, 2)
    return None


# bot 5's gap tracks as the virtual account names them (live/watch_own.py TAL5_TF / tal5_tf): TAL|<name>|<symbol>
_TAL5_FIXED = {"t15_1": "15Min", "t1h_15": "1Hour", "t4h_1h": "4Hour", "t15_5": "15Min5m", "t4h_15": "4Hour15"}
_TF_COMBO = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "4h": "4Hour", "1d": "1Day"}


def tal5_track(bot, combo, cfg):
    """(track, "high/low") of a bot 5 gap-method combo, or None (13.11.2026 night: the loss report of 12.11 kept
    TAL|5Min1m|.. "open" although 5m/1m was out at risk 0 since that night, and kept TAL|15Min|.. open although its
    track's rules were rebuilt - the combo names were never matched to the tracks)."""
    if bot != "bot5" or not str(combo).startswith("TAL|"):
        return None
    mid, sym = combo.split("|")[1], combo.split("|")[-1]
    for trk, v in (cfg.get("tal_tracks") or {}).items():
        pr = list(v.get("pair") or [])[:2]
        if v.get("symbols") and sym not in v["symbols"]:
            continue                                     # an instrument's own track (agents/tal_wide.py)
        if len(pr) == 2 and (_TAL5_FIXED.get(trk) or f"{_TF_COMBO.get(pr[0], pr[0])}{pr[1]}") == mid:
            return trk, f"{pr[0]}/{pr[1]}"
    return None


def improvement_plan(rep, decisions, cards, replies, undo, bot_risk_pct):
    """For every losing strategy of the report: the changes made since (approved, not undone / reverted), each with
    what it is worth in % of the account a month (from its own test on 2026), and what is still open."""
    day0 = pd.Timestamp(rep["day"]).tz_localize(TZ).tz_convert("UTC")
    gone = {x.get("of") for x in decisions if x.get("decision") == "reverted"} | set(undo)
    months = max(1.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp("2026-01-01", tz="UTC")).days / 30.4)
    rows, total = [], 0.0
    for x in rep.get("losers", []):
        c = x["combo"]
        if any(r["combo"] == c for r in rows):
            continue
        fixes = []
        for d in decisions:
            if d.get("decision") != "approved" or d["id"] in gone or pd.Timestamp(d["at"]) < day0 or d.get("kind") == "info":
                continue
            a, card = d.get("action") or {}, cards.get(d.get("card")) or {}
            dc = a.get("combo") or (card.get("change") or {}).get("combo") or (d["title"].split(":")[0].strip() if d["title"].count("|") >= 2 else None)
            if dc != c:
                continue
            if d.get("type") == "risk" and a.get("kind") == "mult" and a.get("value") and float(a["value"]) >= 1:
                continue                     # a risk raise is not a fix of a loss
            if d.get("loss_report"):
                continue                     # the manager's own cut for a missing fix is not the fix itself
            eff = card_effect(card, months, bot_risk_pct, a.get("value") if a.get("kind") == "mult" else None) if card else None
            what = d["title"]
            if d.get("type") == "risk" and a.get("kind") == "mult":
                what = f"סיכון ×{a.get('value')}" + (" (הוסרה)" if not a.get("value") else "")
            elif d.get("type") == "risk" and d.get("kind") in ("tp", "sl", "hours", "dir", "vol"):
                what = d["title"]                # a live rule changed / taken back (a bot agent's card)
            elif d.get("type") == "risk":
                what = "הוסרה מהבוט"
            fixes.append({"what": what, "decision": d["id"], "at": d["at"], "effect_month_pct": eff})
            total += eff or 0
        # changes made in the bot itself (a nightly agent's built card, a removal) count too
        cfg = _read(ROOT / "live" / "bots" / f"{rep['bot']}.json", {}) or {}
        if cfg.get("combos") is not None and c not in cfg["combos"] and not any(("הוסרה" in (f["what"] or "")) or ("להסיר" in (f["what"] or "")) for f in fixes):
            fixes.append({"what": "הוסרה מהבוט", "decision": None, "at": None, "effect_month_pct": None})
        tt = tal5_track(rep["bot"], c, cfg)
        if tt:
            trk, pn = tt
            rv = (cfg.get("risk") or {}).get(trk)
            if rv is not None and float(rv) == 0 and not any("הוסרה" in (f["what"] or "") for f in fixes):
                fixes.append({"what": f"הוסרה מהבוט (מסלול {pn}, סיכון 0)", "decision": None, "at": None, "effect_month_pct": None})
            # a built card on this track (its rules, its risk) or on all the gap tracks together (tal_all), after the day
            for cid, card in cards.items():
                ch, r_ = card.get("change") or {}, replies.get(cid) or {}
                mine = (ch.get("method") == "tal_gap" and ch.get("pair") == pn) or ch.get("track") in (trk, "tal_all")
                if mine and r_.get("built") and r_.get("built") != "skipped" and str(r_.get("at") or "") >= rep["day"] \
                        and cid not in undo and not any(f.get("card") == cid for f in fixes):
                    eff = card_effect(card, months, bot_risk_pct)
                    fixes.append({"what": card.get("title"), "decision": None, "card": cid, "at": r_.get("at"), "effect_month_pct": eff})
                    total += eff or 0
        for cid, card in cards.items():
            r_ = replies.get(cid) or {}
            if (card.get("change") or {}).get("combo") == c and r_.get("built") and r_.get("built") != "skipped" \
                    and str(r_.get("at") or "") >= rep["day"] and not any(f.get("card") == cid for f in fixes) \
                    and not any(f["decision"] and (next((d for d in decisions if d["id"] == f["decision"]), {}) or {}).get("card") == cid for f in fixes):
                eff = card_effect(card, months, bot_risk_pct)
                fixes.append({"what": card.get("title"), "decision": None, "card": cid, "at": r_.get("at"), "effect_month_pct": eff})
                total += eff or 0
        rows.append({"combo": c, "cause": x["cause_he"], "fixes": fixes, "open": not fixes and x["cause"] != "rules"})
    opened = [r["combo"] for r in rows if r["open"]]
    return {"rows": rows, "open": opened, "total_month_pct": round(total, 2),
            "status": "סגור - לכל אסטרטגיה שהפסידה יש תיקון" if not opened else f"פתוח: {len(opened)} אסטרטגיות בלי תיקון עדיין"}


# The failure score and the risk cut - one linear formula, no fixed steps (user, 13.11.2026):
#   F = 0.40*test + 0.25*often + 0.15*size + 0.20*late          (each part 0..1, so F is 0..1)
#     test  = clip((0.2 - m) / 0.4)   m = 2026 mean R per trade as it trades now: +0.2R -> 0, -0.2R -> 1 (no test: 0.5)
#     often = clip(d / 8)             d = live loss days of the strategy in the last 30 days: 8 days -> 1
#     size  = clip((L - 0.5) / 1.5)   L = today's loss in R: 0.5R -> 0, 2R -> 1
#     late  = clip(w / 10)            w = days without a fix after the 2 days of grace: 10 days -> 1
#   risk = 1 - 0.75 * F                 F 0 -> x1 (no cut), F 1 -> x0.25 (the floor; removing is a separate decision)
W_TEST, W_OFTEN, W_SIZE, W_LATE, CUT_MAX = 0.40, 0.25, 0.15, 0.20, 0.75


def _clip(x):
    return max(0.0, min(1.0, float(x)))


def failure_score(bot, combo, rep, idx):
    """F (0..1), the risk multiplier it gives, and every part in plain words."""
    from agents import portfolio_check as PC
    try:
        tr = PC.current_trades(bot, combo, PC.bot_cfg(bot))
    except Exception:  # noqa: BLE001
        tr = None
    if tr:
        m = sum(x[3] for x in tr) / len(tr)
        test, t_he = _clip((0.2 - m) / 0.4), f"בדיקה 2026: {m:+.2f}R לעסקה ({len(tr)} עסקאות)"
    else:
        test, t_he = 0.5, "בדיקה 2026: עוד אין (נחשב חצי)"
    day0 = pd.Timestamp(rep["day"])
    d = 0
    for r in idx.get("reports", []):
        if r["bot"] == bot and 0 <= (day0 - pd.Timestamp(r["day"])).days <= 30:
            doc = _read(OUT / bot / f"{r['day']}.json", {}) or {}
            d += any(x.get("combo") == combo for x in doc.get("losers", []))
    often = _clip(d / 8)
    loss_r = -sum(x["r"] for x in rep.get("losers", []) if x["combo"] == combo)
    size = _clip((loss_r - 0.5) / 1.5)
    w = max(0, (pd.Timestamp.now(tz=TZ).date() - day0.date()).days - 2)
    late = _clip(w / 10)
    F = W_TEST * test + W_OFTEN * often + W_SIZE * size + W_LATE * late
    mult = round(1 - CUT_MAX * F, 2)
    parts = [f"{t_he} → {test:.2f}", f"ימי הפסד בחודש: {d} → {often:.2f}", f"ההפסד היום: {loss_r:.1f}R → {size:.2f}",
             f"ימים בלי תיקון: {w} → {late:.2f}"]
    return round(F, 3), mult, parts


def enforce(rep, plan, decisions, have_ids, at, idx=None):
    """The manager owns it (user, 13.11.2026: "the manager takes the improvement into his hands"): a losing strategy
    still without a fix 2 days after its report goes down in risk by the linear failure formula above (user 13.11),
    until a fix is found ("first fix, and only if there is none, lower the risk").
    The level is computed again every run - if it grows, the risk goes down more. Bots 1-4, where risk per strategy is
    applied on the bots' computer."""
    out = []
    if rep["bot"] not in ("bot1", "bot2", "bot3", "bot4") or rep["day"] < ENFORCE_FROM:
        return out
    levels = {}
    for row in plan["rows"]:
        if row["open"]:
            row["fail"], row["fail_mult"], row["fail_parts"] = failure_score(rep["bot"], row["combo"], rep, idx or {})
            levels[row["combo"]] = (row["fail"], row["fail_mult"], row["fail_parts"])
    from agents import portfolio_check as PC
    mm = PC.mults(rep["bot"])
    gone = {x.get("of") for x in decisions if x.get("decision") == "reverted"}
    # a fix was found: the manager's cut comes back (the fix decides the strategy's risk from now on)
    for row in plan["rows"]:
        if row["open"] or not row["fixes"]:
            continue
        cuts = [d for d in decisions if d.get("loss_report") == f"{rep['bot']}/{rep['day']}" and d.get("type") == "risk"
                and (d.get("action") or {}).get("combo") == row["combo"] and d.get("decision") == "approved" and d["id"] not in gone]
        if not cuts:
            continue
        first = min(cuts, key=lambda d: d["at"])
        back = float(first["action"].get("prev") or 1.0)
        did = hashlib.sha1(f"lossback|{rep['bot']}|{row['combo']}|{rep['day']}".encode()).hexdigest()[:12]
        cur = float(mm.get(row["combo"], 1.0) or 1.0)
        if did in have_ids or back <= cur:
            continue
        out.append({"id": did, "bot": rep["bot"], "type": "risk", "kind": "mult", "combo": row["combo"],
                    "title": f"{row['combo']}: סיכון חוזר ל־×{back} - נמצא תיקון אחרי דוח ההפסד של {rep['day']}",
                    "decision": "approved", "by": "manager", "read": False, "undone": False, "at": at,
                    "why": "המנהל: ההורדה הייתה רק עד שיימצא תיקון. נמצא: " + "; ".join(f["what"] for f in row["fixes"] if f.get("what")),
                    "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "loss_back": f"{rep['bot']}/{rep['day']}",
                    "action": {"bot": rep["bot"], "combo": row["combo"], "kind": "mult", "value": back, "prev": cur}})
        have_ids.add(did)
    if (pd.Timestamp.now(tz=TZ).date() - pd.Timestamp(rep["day"]).date()).days < 2:
        return out
    for c in plan["open"]:
        F, fm, parts = levels[c]
        did = hashlib.sha1(f"lossrisk|{rep['bot']}|{c}|{fm}".encode()).hexdigest()[:12]
        if did in have_ids:
            continue
        cur = float(mm.get(c, 1.0) or 1.0)
        # from the risk before the manager's own cuts, so a higher level replaces a lower one instead of stacking
        base = cur
        for d in decisions:
            if d.get("type") == "risk" and (d.get("action") or {}).get("combo") == c and d["id"] not in gone \
                    and d.get("loss_report") and (d.get("action") or {}).get("prev") is not None:
                base = max(base, float(d["action"]["prev"]))
        val = round(base * fm, 2)
        if val >= cur - 0.02:                     # the formula moves daily; a change under 0.02 is not worth a decision
            continue
        out.append({"id": did, "bot": rep["bot"], "type": "risk", "kind": "mult", "combo": c,
                    "title": f"{c}: סיכון ×{val} - ציון כישלון {F:.2f}, אין תיקון אחרי דוח ההפסד של {rep['day']}",
                    "decision": "approved", "by": "manager", "read": False, "undone": False, "at": at,
                    "why": "המנהל: אין תיקון שעבר את הבדיקה, אז הסיכון יורד לפי ציון הכישלון. "
                           "ציון = 0.40×בדיקה + 0.25×תדירות + 0.15×גודל + 0.20×זמן: " + "; ".join(parts)
                           + f" ← ציון {F:.2f}. סיכון = 1 − 0.75×ציון = ×{fm}. כשיימצא תיקון - הסיכון חוזר; אם מחמיר - יורד עוד.",
                    "effect": "יוחל במחשב הבוטים תוך כ־10 דקות", "loss_report": f"{rep['bot']}/{rep['day']}", "fail": F,
                    "action": {"bot": rep["bot"], "combo": c, "kind": "mult", "value": val, "prev": cur}})
        have_ids.add(did)
    return out


def follow_up(idx):
    """Every run, for the reports of the last 21 days: the improvement plan, the manager's enforcement, and a task
    sent back to the bot's agent when it answered without a fix."""
    from agents import builder as BLD
    from agents import repo_io as RIO
    dpath = ROOT / "reports" / "agents" / "decisions.json"
    ddoc = _read(dpath, {"decisions": []})
    decisions = ddoc.get("decisions", [])
    user = RIO.live_json("live/agent_decisions_state.json")
    undo = set(user.get("undo") or [])
    cards = BLD.all_cards()
    replies = _read(TASKS, {})
    have = {x["id"] for x in decisions}
    at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    adds, changed_tasks = [], False
    for r in idx["reports"]:
        if (pd.Timestamp.now(tz=TZ).date() - pd.Timestamp(r["day"]).date()).days > 21:
            continue
        f = OUT / r["bot"] / f"{r['day']}.json"
        rep = _read(f, None)
        if not rep:
            continue
        cfg = _read(ROOT / "live" / "bots" / f"{r['bot']}.json", {}) or {}
        risk_pct = float(cfg.get("risk") or 0.004) * 100 if not isinstance(cfg.get("risk"), dict) else 0.4
        plan = improvement_plan(rep, decisions, cards, replies, undo, risk_pct)
        rep["plan"] = plan
        adds += enforce(rep, plan, decisions, have, at, idx)
        _write(f, rep)
        r["status"], r["open"], r["month_pct"] = plan["status"], len(plan["open"]), plan["total_month_pct"]
        # an answer without a fix is not an answer: the task goes back to the bot's agent
        for tk in replies.get("_tasks", []):
            if tk.get("from") != "loss_review" or tk.get("report") != f"reports/agents/losses/{r['bot']}/{r['day']}.json":
                continue
            s = (tk.get("status") or {}).get(r["bot"]) or {}
            if plan["open"] and s.get("state") == "done":
                s["state"] = "reopened"
                s["text"] = (s.get("text") or "") + (f"\nהמנהל ({at[:16].replace('T', ' ')}): לא התקבל. 'אין באג' ו'יום רגיל' אינם תשובה - "
                             f"לכל אסטרטגיה שהפסידה צריך תיקון שנבדק (יעד / סטופ / כניסה / מסנן / ניהול עסקה / שעות / כיוון), "
                             f"או הוכחה שכל המשפחות נבדקו ונכשלו - ואז הסיכון יורד. עדיין פתוח: {', '.join(plan['open'])}.")
                tk.setdefault("status", {})[r["bot"]] = s
                changed_tasks = True
    if adds:
        ddoc["decisions"] = adds + decisions
        ddoc["new"] = list(ddoc.get("new") or []) + [x["id"] for x in adds]
        _write(dpath, ddoc)
        for x in adds:
            print("enforced:", x["title"])
    if changed_tasks:
        _write(TASKS, replies)
    return adds


def trend(bot, reports):
    now = pd.Timestamp.now(tz=TZ).normalize()
    def window(a, b):
        rs = [r for r in reports if a <= pd.Timestamp(r["day"]).tz_localize(TZ) < b]
        return {"loss_days": len(rs), "avg_loss_pct": round(float(np.mean([r["pnl_pct"] for r in rs])), 3) if rs else 0.0,
                "worst_pct": round(min([r["pnl_pct"] for r in rs]), 3) if rs else 0.0}
    last, prev = window(now - pd.Timedelta(days=7), now), window(now - pd.Timedelta(days=14), now - pd.Timedelta(days=7))
    better = (last["loss_days"] < prev["loss_days"]) or (last["loss_days"] == prev["loss_days"] and last["avg_loss_pct"] > prev["avg_loss_pct"])
    return {"last7": last, "prev7": prev, "improving": better if prev["loss_days"] else None}


def send_tasks(rep):
    """The manager's demand: the bot's agent gets every action as a task (the nightly agents do these first)."""
    d = _read(TASKS, {})
    tasks = d.setdefault("_tasks", [])
    have = {t.get("id") for t in tasks if isinstance(t, dict)}
    tid = hashlib.sha1(f"loss|{rep['bot']}|{rep['day']}".encode()).hexdigest()[:12]
    if tid not in have and rep["actions"]:
        # one task per loss day: the report and every action in it; answered with the root cause and the fix
        tasks.append({"id": tid, "at": rep["day"], "bots": [rep["bot"]], "from": "loss_review", "urgent": True,
                      "report": f"reports/agents/losses/{rep['bot']}/{rep['day']}.json",
                      "task": (f"[דוח הפסד {rep['day']}, {rep['pnl']:+,.0f}$ / {rep['pnl_pct']:+.2f}%] עבור על הדוח ועל כל פעולה בו: "
                               "מצא את הסיבה האמיתית לכל הפסד (לא רק הקטגוריה), בדוק את התיקון בבדיקה ההוגנת, ושלח כרטיס. "
                               "ענה כאן: הסיבה, התיקון, ומה השתנה. הפעולות: " + " | ".join(a["task"] for a in rep["actions"])),
                      "status": {}})
        _write(TASKS, d)
    pr = _read(PRIO, {})
    for a in rep["actions"]:
        if a["combo"] and a["priority"]:
            p = pr.setdefault(rep["bot"], {}).setdefault(a["combo"], {"families": [], "why": []})
            p["families"] = sorted(set(p["families"]) | set(a["priority"]))
            p["why"] = (p["why"] + [f"{rep['day']}: {CAUSE_HE.get(a['cause'], a['cause'])}"])[-5:]
            p["at"] = rep["at"]
    _write(PRIO, pr)


ENFORCE_FROM = "2026-11-12"   # the manager's enforcement: reports from the day the loss review started
FORMAT = 2          # 13.11.2026: plain words, no "normal day", the improvement plan - older reports are written again


def run():
    states = H.live_states()
    idx = _read(OUT / "index.json", {"reports": []})
    if idx.get("format") != FORMAT:
        idx = {"reports": [], "format": FORMAT}
    have = {(r["bot"], r["day"]) for r in idx["reports"]}
    today = pd.Timestamp.now(tz=TZ).date()
    new = []
    for bot in NAMES:
        st = H.state_of(bot, states)
        if not st:
            continue
        t = trade_rows(st)
        if t.empty:
            continue
        exp = H.expectations(bot)
        days = sorted({d for d in t["day"] if pd.Timestamp(d).date() < today
                       and (pd.Timestamp(today) - pd.Timestamp(d)).days <= BACK_DAYS})
        for day in days:
            if (bot, day) in have or float(t.loc[t["day"] == day, "net"].sum()) >= 0:
                continue
            hist = [_read(OUT / bot / f"{r['day']}.json", {}) for r in idx["reports"]
                    if r["bot"] == bot and 0 < (pd.Timestamp(day) - pd.Timestamp(r["day"])).days <= 30]
            rep = review(bot, st, day, t, hist, exp)
            rep["trend"] = trend(bot, [x for x in idx["reports"] if x["bot"] == bot] + [{"day": day, "pnl_pct": rep["pnl_pct"]}])
            _write(OUT / bot / f"{day}.json", rep)
            send_tasks(rep)
            idx["reports"].append({"bot": bot, "day": day, "pnl": rep["pnl"], "pnl_pct": rep["pnl_pct"],
                                   "headline": rep["summary"][0], "actions": len(rep["actions"])})
            have.add((bot, day))
            new.append(rep)
            print(f"{bot} {day}: {rep['pnl']:+.0f}$ ({rep['pnl_pct']:+.2f}%), {len(rep['actions'])} actions")
    idx["reports"] = sorted(idx["reports"], key=lambda r: (r["day"], r["bot"]), reverse=True)[:400]
    try:
        follow_up(idx)
    except Exception as e:  # noqa: BLE001
        print(f"follow-up failed: {e}")
    idx["updated"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    _write(OUT / "index.json", idx)
    print(f"loss review: {len(new)} new reports")
    return new


if __name__ == "__main__":
    run()
