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
BACK_DAYS = 7                            # days reviewed back (first run, or a computer that was off)
NAMES = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4", "bot5": "בוט 5 - סקאלפינג",
         "bot6": "בוט 6 · חוזים", "bot7": "בוט 7 · פרסומים"}
CAUSE_HE = {"entry": "כניסה / כיוון", "management": "ניהול העסקה", "execution": "ביצוע (החלקה / פער / עלויות)",
            "rules": "סגירה בגלל כלל", "time": "יציאת זמן", "normal": "הפסד רגיל של האסטרטגיה"}
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
    lines = [
        f"{NAMES.get(bot, bot)} · {day}: {net:+,.0f}$ ({pct:+.2f}% מיתרת הפתיחה {bal0:,.0f}$), "
        f"{len(d)} עסקאות - {len(win)} רווח, {len(los)} הפסד. ניצול המגבלה היומית: {abs(min(0, pct)) / BOT_DAILY * 100:.0f}% "
        f"ממגבלת הבוט (4.5%), {abs(min(0, pct)) / FTMO_DAILY * 100:.0f}% ממגבלת FTMO (5%).",
        (f"מול הבדיקה: {float(d['r'].sum()):+.2f}R ביום מול {em:+.2f}R צפוי (פיזור ±{esd:.2f}R) - "
         + ("יום רע רגיל, בתוך הרעש הסטטיסטי." if normal else "חריג: הרבה מתחת למה שהבדיקה מאפשרת.")) if esd else
        "אין לבדיקה צפי לאסטרטגיות של היום - אי אפשר לומר אם זה רגיל.",
        f"גורמי ההפסד: " + ", ".join(f"{CAUSE_HE[k]} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])) + ".",
    ]
    if top:
        lines.append(f"ההפסד הגדול: {top} ({by_combo.iloc[0]:,.0f}$, {top_share:g}% מההפסדים).")
    if corr:
        lines.append("ריכוז: " + "; ".join(f"{c['trades']} הפסדים יחד ב{c['side']} של {c['class']} ({c['loss']:,.0f}$)" for c in corr) + ".")
    if recurring:
        lines.append("חוזר: " + ", ".join(recurring) + " - הפסד ב־3 ימים ומעלה בחודש האחרון. דפוס, לא מזל.")
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
                     "task": f"יום חריג ({pct:+.2f}%): הרבה מתחת למה שהבדיקה מאפשרת. לבדוק קודם תקלה בקוד או בנתונים (פער בין החי לבדיקה), לפני שינוי באסטרטגיות."})
    seen, out = set(), []
    for a in acts:
        k = (a["combo"], a["cause"])
        if k not in seen:
            seen.add(k)
            out.append(a)
    return out


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


def run():
    states = H.live_states()
    idx = _read(OUT / "index.json", {"reports": []})
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
                                   "normal": rep["normal"], "headline": rep["summary"][0], "actions": len(rep["actions"])})
            have.add((bot, day))
            new.append(rep)
            print(f"{bot} {day}: {rep['pnl']:+.0f}$ ({rep['pnl_pct']:+.2f}%), {len(rep['actions'])} actions")
    if new:
        idx["reports"] = sorted(idx["reports"], key=lambda r: (r["day"], r["bot"]), reverse=True)[:400]
        idx["updated"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        _write(OUT / "index.json", idx)
    print(f"loss review: {len(new)} new reports")
    return new


if __name__ == "__main__":
    run()
