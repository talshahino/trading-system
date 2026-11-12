"""Health agent of every bot (user, 11.11.2026: "תשתית לכל" - the cheap layer of the bot agents, for all 7 bots).

No backtests here: it reads each bot's live state (branch "live", live/<name>_state.json) and compares the live trades
with what the backtest promised, every few hours. What it finds goes to the manager as cards (kind "info") and as a
health summary that the app shows in each bot's agent section.

Checks, per bot and per strategy combo:
  worse     the live mean R is far below the backtest mean: z = (live - backtest) / (sd / sqrt(n)) <= -2, n >= 8
  streak    the current run of losses is unlikely under the backtest success rate: (1 - win)^L < 2%, L >= 4
  day/total the account is near the fixed limits (daily 5%, total 10% - user, permanent rule): an alert from -3% / -7%
  stale     the bot has not written its state for over 45 minutes on a trading day (a problem for the manager)
Backtest expectations: bots 1-4 from reports/trade_review/risk.json (the live rules' backtest); bot 5 from its own
agent (reports/agents/bot5/cards.json summary); bots 6-7 from data/agents/expect.json when there (else vs zero).
Writes reports/agents/<bot>/health.json and reports/agents/<bot>/cards_health.json.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from agents import agentkit as K

ROOT = K.ROOT
SINCE = "2026-11-01"
STATE_PREFIX = {"bot1": "bot1", "bot2": "bot2_", "bot3": "bot3_", "bot4": "bot4_", "bot5": "watch_own",
                "bot6": "watch_fut", "bot7": "news_bot"}
NAMES = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4", "bot5": "בוט 5 · סקלפינג",
         "bot6": "בוט 6 · חוזים עתידיים", "bot7": "בוט 7 · פרסומים כלכליים"}
NO_RISK_DOWN = {"bot3"}            # user: don't lower bot 3's risk - its alerts never suggest it
MIN_N, Z_BAD, STREAK_P, STREAK_MIN = 8, -2.0, 0.02, 4
DAY_WARN, TOTAL_WARN, STALE_MIN = -3.0, -7.0, 45


def live_states():
    """{file name: state dict} from origin/live (falls back to a local live/ folder)."""
    out = {}
    from agents import repo_io as RIO
    for n in RIO.live_names("live"):
        if n.endswith("_state.json"):
            s = RIO.live_json(n)
            if s:
                out[Path(n).name] = s
    if not out:
        for f in (ROOT / "live").glob("*_state.json"):
            try:
                out[f.name] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    return out


def state_of(bot, states):
    p = STATE_PREFIX[bot]
    hits = [s for n, s in states.items() if n.startswith(p) and (p.endswith("_") or n == f"{p}_state.json")]
    return hits[0] if hits else None


def watch_trades(state, bot):
    """Virtual bots 5-6: their full watch record (every closed setup with its R after costs), not only the
    virtual account's trades since its last restart."""
    rows, pre = [], {"bot5": "OWN", "bot6": "FUT"}.get(bot, "W")
    for w in state.get("watch") or []:
        if w.get("status") != "closed" or w.get("skipped") or w.get("r") is None:
            continue
        trk = w.get("trk") or (w["id"].split("|")[3] if w.get("id", "").count("|") >= 3 else "")
        tf = "1Min" if str(trk).endswith("1m") else "15Min"
        combo = f"TAL|1Hour|{w['symbol']}" if trk == "tal" else f"{pre}|{tf}|{w['symbol']}"   # the user's gap method
        rows.append({"id": w["id"], "combo": combo, "time": w.get("time"),
                     "close": w.get("exit_time") or w.get("time"), "r": float(w["r"]), "net": None})
    return rows


def trades(state, bot=None):
    """Closed trades in R after costs: price R x net / gross (the costs the bot actually paid)."""
    rows = watch_trades(state, bot) if state.get("watch") else []
    if rows:
        state = {}
    for c in state.get("closed") or []:
        try:
            side = 1 if c["side"] == "buy" else -1
            o, sl0 = float(c["open_px"]), float(c.get("sl0") or 0)
            risk = side * (o - sl0) if sl0 else 0
            if risk <= 0:
                continue
            r = side * (float(c["close_px"]) - o) / risk
            g, n = float(c.get("gross") or 0), float(c.get("net") or 0)
            if g and abs(g) > 1e-9 and np.sign(g) == np.sign(n) and abs(n / g) < 3:
                r *= n / g
            elif g and n and np.sign(g) != np.sign(n):
                r = n / (abs(g) / abs(r)) if r else r      # a small winner turned into a loss by the costs
            rows.append({"id": str(c["id"]), "combo": c["combo"], "time": c["open_t"], "close": c.get("close_t") or c["open_t"],
                         "r": r, "net": n})
        except Exception:  # noqa: BLE001
            continue
    t = pd.DataFrame(rows, columns=["id", "combo", "time", "close", "r", "net"])
    if t.empty:
        return t
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    t["close"] = pd.to_datetime(t["close"], utc=True, format="ISO8601")
    return t[t["time"] >= pd.Timestamp(SINCE, tz="UTC")].drop_duplicates("id").sort_values("close").reset_index(drop=True)


def expectations(bot):
    """{combo: {mean, n, win, sd}} from the backtest that matches the live rules; {} when there is none."""
    out = {}
    if bot in ("bot1", "bot2", "bot3", "bot4"):
        try:
            d = json.loads((ROOT / "reports/trade_review/risk.json").read_text(encoding="utf-8"))
            for c in d["bots"].get(bot, {}).get("combos", []):
                b = (c.get("detail") or {}).get("profit", {}).get("backtest") or {}
                sd = ((c.get("detail") or {}).get("stability", {}).get("backtest") or {}).get("sd_r")
                if b.get("trades"):
                    out[c["combo"]] = {"mean": b.get("mean_r"), "n": b.get("trades"), "win": (b.get("win_pct") or 0) / 100, "sd": sd}
        except Exception:  # noqa: BLE001
            pass
    elif bot == "bot5":
        try:
            d = json.loads((K.CARDS_DIR / "bot5" / "cards.json").read_text(encoding="utf-8"))
            for trk, v in (d.get("summary") or {}).items():
                b = (v.get("backtest") or {}).get("whole") or (v.get("backtest") or {}).get("unseen") or {}
                if b.get("mean") is not None:
                    out["@" + trk] = {"mean": b["mean"], "n": b.get("n"), "win": (b.get("win") or 0) / 100, "sd": None}
        except Exception:  # noqa: BLE001
            pass
    try:
        extra = json.loads((ROOT / "data/agents/expect.json").read_text(encoding="utf-8")).get(bot, {})
        out.update(extra)
    except Exception:  # noqa: BLE001
        pass
    return out


def expect_for(bot, combo, exp):
    if combo in exp:
        return exp[combo]
    if bot == "bot5":                           # bot 5: by track (the 1-minute track or the 15-minute one)
        return exp.get("@v1m" if "|1Min|" in combo else "@v5")
    return None


def streak(r):
    L = 0
    for x in reversed(list(r)):
        if x < 0:
            L += 1
        else:
            break
    return L


def check(bot, state, now):
    t = trades(state, bot)
    exp = expectations(bot)
    combos, cards, problems = [], [], []
    for combo, g in (t.groupby("combo") if not t.empty else []):
        r = g["r"].to_numpy()
        n, m = len(r), float(np.mean(r))
        e = expect_for(bot, combo, exp) or {}
        sd = e.get("sd") or max(float(np.std(r, ddof=1)) if n > 2 else 0.0, 1.0)   # no backtest sd: at least 1R
        z = (m - e["mean"]) / (sd / math.sqrt(n)) if e.get("mean") is not None and n >= 2 else None
        L = streak(r)
        win = e.get("win") or None
        p_streak = (1 - win) ** L if win else None
        row = {"combo": combo, "n": n, "mean": round(m, 3), "sum": round(float(r.sum()), 2),
               "win": round(float((r > 0).mean()) * 100, 1), "streak": L,
               "bt_mean": e.get("mean"), "bt_n": e.get("n"), "z": None if z is None else round(z, 2), "flags": []}
        week = now.strftime("%G-W%V")
        if z is not None and n >= MIN_N and z <= Z_BAD:
            row["flags"].append("worse")
            act = ("הסוכן של הבוט יבדוק מה השתנה ויציע תיקון" if bot in NO_RISK_DOWN
                   else "הסוכן של הבוט יבדוק מה השתנה ויציע תיקון, הורדת סיכון או הסרה")
            cards.append(K.card(bot, "info", f"{combo}: בפועל גרוע בהרבה מהבדיקה",
                                f"{n} עסקאות בפועל, {m:+.2f}R לעסקה, מול {e['mean']:+.2f}R בבדיקה על נתוני העבר "
                                f"(סטייה של {abs(z):.1f} סטיות תקן). אישור: {act}.",
                                None, None, change={"check": "worse", "combo": combo, "week": week},
                                detail={"live": row}))
        if p_streak is not None and L >= STREAK_MIN and p_streak < STREAK_P:
            row["flags"].append("streak")
            cards.append(K.card(bot, "info", f"{combo}: {L} הפסדים ברצף",
                                f"בבדיקה האסטרטגיה מצליחה ב־{win * 100:.0f}% מהעסקאות, אז רצף כזה קורה רק ב־{p_streak * 100:.1f}% מהמקרים. "
                                f"אישור: הסוכן של הבוט יבדוק אם משהו בשוק או בביצוע השתנה.",
                                None, None, change={"check": "streak", "combo": combo, "week": week},
                                detail={"live": row}))
        combos.append(row)
    # the whole bot: all its live trades against the backtest of the same combos (weighted by the live trades)
    ex = [(c["n"], c["bt_mean"]) for c in combos if c["bt_mean"] is not None]
    nb = sum(n for n, _ in ex)
    if nb >= 20:
        bm = sum(n * m for n, m in ex) / nb
        lm = float(np.mean([c["mean"] * c["n"] for c in combos if c["bt_mean"] is not None])) * len(ex) / nb
        zb = (lm - bm) / (1.0 / math.sqrt(nb))
        if zb <= Z_BAD:
            problems.append(f"כל הבוט: {lm:+.2f}R לעסקה בפועל מול {bm:+.2f}R בבדיקה ({nb} עסקאות)")
            cards.append(K.card(bot, "info", "הבוט כולו גרוע מהבדיקה",
                                f"{nb} עסקאות בפועל, {lm:+.2f}R לעסקה, מול {bm:+.2f}R בבדיקה על נתוני העבר של אותן אסטרטגיות "
                                f"(סטייה של {abs(zb):.1f} סטיות תקן). אישור: הסוכן של הבוט יבדוק אם הבעיה בעלויות, בביצוע או בשוק, ויציע תיקון.",
                                None, None, change={"check": "bot_worse", "week": now.strftime("%G-W%V")}))
    # the account against the fixed limits (daily 5%, total 10%)
    try:
        eq, ds, ini = float(state.get("equity")), float(state.get("day_start")), float(state.get("initial"))
        day = (eq / ds - 1) * 100 if ds else 0.0
        tot = (eq / ini - 1) * 100 if ini else 0.0
    except Exception:  # noqa: BLE001
        day = tot = None
    if day is not None and day <= DAY_WARN:
        problems.append(f"הפסד יומי {day:.1f}% - קרוב למגבלה הקבועה של 5%")
        cards.append(K.card(bot, "info", f"הפסד יומי {day:.1f}%", "החשבון מתקרב למגבלת ההפסד היומי הקבועה (5%). "
                            "הבוט עוצר לבד ב־5%. אישור: הסוכן יבדוק אילו עסקאות גרמו לזה.", None, None,
                            change={"check": "day", "day": now.strftime("%Y-%m-%d")}))
    if tot is not None and tot <= TOTAL_WARN:
        problems.append(f"הפסד כולל {tot:.1f}% - קרוב למגבלה הקבועה של 10%")
        cards.append(K.card(bot, "info", f"הפסד כולל {tot:.1f}%", "החשבון מתקרב למגבלת ההפסד הכולל הקבועה (10%). "
                            "אישור: הסוכן יציע הורדת סיכון לכל הבוט עד שהחשבון יתאושש.", None, None,
                            change={"check": "total", "week": now.strftime("%G-W%V")}))
    # stale: no state for a while on a trading day (Mon-Fri)
    up = pd.to_datetime(state.get("updated_utc"), utc=True, errors="coerce", format="ISO8601")
    age = None if pd.isna(up) else (now - up).total_seconds() / 60
    if age is not None and age > STALE_MIN and now.weekday() < 5:
        problems.append(f"הבוט לא עדכן מצב כבר {age / 60:.1f} שעות")
    allr = t["r"].to_numpy() if not t.empty else np.array([])
    bt_all = [v["mean"] for v in exp.values() if v.get("mean") is not None]
    combos.sort(key=lambda x: (-len(x["flags"]), x["z"] if x["z"] is not None else 9))
    return {"bot": bot, "name": NAMES[bot], "updated": now.isoformat(timespec="seconds"),
            "state_updated": None if pd.isna(up) else up.isoformat(timespec="seconds"),
            "minutes_since_state": None if age is None else round(age),
            "balance": state.get("balance"), "equity": state.get("equity"),
            "day_pct": None if day is None else round(day, 2), "total_pct": None if tot is None else round(tot, 2),
            "trades": int(len(allr)), "mean_r": None if not len(allr) else round(float(allr.mean()), 3),
            "sum_r": None if not len(allr) else round(float(allr.sum()), 2),
            "bt_mean_r": None if not bt_all else round(float(np.mean(bt_all)), 3),
            "flagged": sum(1 for c in combos if c["flags"]), "combos": combos, "problems": problems}, cards


def main():
    now = pd.Timestamp.now(tz="UTC")
    states = live_states()
    for bot in STATE_PREFIX:
        d = K.CARDS_DIR / bot
        d.mkdir(parents=True, exist_ok=True)
        st = state_of(bot, states)
        if st is None:
            h, cards = {"bot": bot, "name": NAMES[bot], "updated": now.isoformat(timespec="seconds"),
                        "problems": ["לא נמצא מצב חי של הבוט"], "combos": []}, []
        else:
            h, cards = check(bot, st, now)
        (d / "health.json").write_text(json.dumps(h, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        (d / "cards_health.json").write_text(json.dumps({"bot": bot, "updated": h["updated"], "cards": cards},
                                                        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"{bot}: {h.get('trades', 0)} trades, {h.get('flagged', 0)} flagged combos, {len(cards)} cards, "
              f"problems: {'; '.join(h['problems']) or 'none'}")


if __name__ == "__main__":
    main()
