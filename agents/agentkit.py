"""Shared toolkit of the improvement agents (user, 10-11.11.2026; plan "סוכני שיפור וסוכן מנהל").

  time weight   w = max(FLOOR, 1 - (age / HORIZON)^POWER)   age in years; 1.0 today, 0.94 a year ago, 0.75 two years,
                0.44 three, FLOOR from ~3.5 years on (the market learns: yesterday's smart money is today's dumb money)
  weighted stats  mean, t, effective number of trades (sum w)^2 / sum w^2 - a decision needs >= MIN_EFF_N
  erosion       is the edge fading: the last 6 months against the 18 months before, and the weighted slope of R in time
  card          the one format every bot agent sends the manager (see card())
  tiers         the manager's fixed rules (judge()): 'strong' -> the bot's risk, 'regular' -> risk set from the history,
                'watch' -> kept for later, 'rejected'. No automatic half-risk (user, 10.11.2026).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON, FLOOR, POWER = 4.0, 0.2, 2.0       # the user's formula (10.11.2026); the agents may propose others (tune_weights)
MIN_EFF_N = 30
GATE_T = 3.17                                # GATE.md section 7
ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = ROOT / "reports" / "agents"


def weights(times, now=None, horizon=HORIZON, floor=FLOOR, power=POWER):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    t = pd.to_datetime(pd.Series(times), utc=True, format="ISO8601")
    age = ((now - t).dt.total_seconds() / (365.25 * 86400)).clip(lower=0).to_numpy()
    return np.maximum(floor, 1 - (age / horizon) ** power)


def wstats(r, w=None):
    r = np.asarray(r, dtype=float)
    w = np.ones_like(r) if w is None else np.asarray(w, dtype=float)
    if len(r) == 0 or w.sum() <= 0:
        return {"n": 0, "eff_n": 0.0, "mean": None, "t": None, "win": None}
    m = float(np.sum(w * r) / w.sum())
    eff = float(w.sum() ** 2 / np.sum(w ** 2))
    var = float(np.sum(w * (r - m) ** 2) / w.sum())
    t = m / math.sqrt(var) * math.sqrt(eff) if var > 0 and eff >= 5 else None
    return {"n": int(len(r)), "eff_n": round(eff, 1), "mean": round(m, 4), "t": None if t is None else round(t, 2),
            "win": round(float(np.sum(w * (r > 0)) / w.sum()) * 100, 1)}


def erosion(times, r, now=None):
    """The last 6 months against the 18 months before; and the weighted slope of R per year."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    t = pd.to_datetime(pd.Series(times), utc=True, format="ISO8601").reset_index(drop=True)
    r = pd.Series(np.asarray(r, dtype=float))
    recent = r[t >= now - pd.Timedelta(days=183)]
    before = r[(t < now - pd.Timedelta(days=183)) & (t >= now - pd.Timedelta(days=730))]
    yrs = ((t - now).dt.total_seconds() / (365.25 * 86400)).to_numpy()
    slope = float(np.polyfit(yrs, r.to_numpy(), 1)[0]) if len(r) >= 20 else None
    out = {"recent_n": int(len(recent)), "recent_mean": None if recent.empty else round(float(recent.mean()), 4),
           "before_n": int(len(before)), "before_mean": None if before.empty else round(float(before.mean()), 4),
           "slope_per_year": None if slope is None else round(slope, 4)}
    out["fading"] = bool(len(recent) >= 15 and len(before) >= 15 and out["recent_mean"] < out["before_mean"] - 0.1
                         and (slope is None or slope < 0))
    return out


def halves(times, r):
    t = pd.to_datetime(pd.Series(times), utc=True, format="ISO8601").reset_index(drop=True)
    r = pd.Series(np.asarray(r, dtype=float))
    if len(r) < 4:
        return [None, None]
    mid = t.sort_values().iloc[len(t) // 2]
    a, b = r[t < mid], r[t >= mid]
    return [round(float(a.mean()), 4) if len(a) else None, round(float(b.mean()), 4) if len(b) else None]


def card(bot, kind, title, what, train, unseen, *, whole=None, halves_=None, neighbours=None, tested=1, cost_r=None,
         risk_pct=None, detail=None, baseline=None, change=None):
    """One proposal, in the fixed format. train / unseen / whole / baseline: wstats() dicts.
    kind: improve | new | remove | risk | weights | info.  change: what to set if approved (machine-readable)."""
    c = {"bot": bot, "kind": kind, "title": title, "what": what, "train": train, "unseen": unseen, "whole": whole,
         "halves": halves_, "neighbours": neighbours, "tested": int(tested), "cost_r": cost_r, "risk_pct": risk_pct,
         "baseline": baseline, "change": change, "detail": detail or {},
         "made": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    key = json.dumps({"bot": bot, "kind": kind, "change": change or title}, sort_keys=True, default=str)
    c["id"] = hashlib.sha1(key.encode()).hexdigest()[:12]
    return c


def judge(c):
    """The manager's fixed rules. Returns (tier, reason in Hebrew). Gentle enough that proposals keep coming
    (user, 10.11.2026: 'some will come by luck, they will be lowered later by the live results')."""
    u, w, h, nb = c.get("unseen") or {}, c.get("whole") or {}, c.get("halves") or [None, None], c.get("neighbours") or {}
    base = (c.get("baseline") or {}).get("mean")
    if c["kind"] in ("remove", "risk", "info"):
        return "regular", "המלצה של סוכן הבוט, נבדקה ברמת הבוט"
    if not u or u.get("mean") is None or u.get("eff_n", 0) < 10:
        return "watch", "מעט מדי עסקאות בתקופה שלא נראתה - נשמר להמשך"
    if u["mean"] <= 0:
        return "rejected", f"הפסיד בתקופה שלא נראתה ({u['mean']:+.2f}R לעסקה)"
    if base is not None and u["mean"] <= base:
        return "rejected", f"לא טוב מהמצב הקיים בתקופה שלא נראתה ({u['mean']:+.2f}R מול {base:+.2f}R)"
    gain = None if base is None else u["mean"] - base
    if gain is not None and gain < 0.02:
        return "watch", f"טוב מהמצב הקיים רק ב־{gain:+.3f}R לעסקה בתקופה שלא נראתה - קטן מדי, נשמר להמשך"
    # the gain itself must stand out from the noise of the two unseen samples (standard error from mean / t)
    bt = (c.get("baseline") or {}).get("t")
    se = lambda m, t: abs(m / t) if m and t else None  # noqa: E731
    s1, s2 = se(u["mean"], u.get("t")), se(base, bt) if base is not None else None
    gain_t = gain / math.sqrt(s1 ** 2 + s2 ** 2) if gain is not None and s1 and s2 else None
    strong = ((w.get("t") or 0) >= GATE_T and all(x is not None and x > 0 for x in h) and (nb.get("share_positive") or 0) >= 0.5
              and (gain is None or (gain >= 0.05 and (gain_t or 0) >= 2.0)))
    if strong:
        return "strong", (f"עובר את הסף המלא: מבחן {w['t']:.2f}, חיובי בשני החצאים, {nb.get('share_positive', 0) * 100:.0f}% מהשילובים הקרובים מחזיקים"
                          + (f", {gain:+.2f}R לעסקה מעל המצב הקיים" if gain is not None else ""))
    if u.get("eff_n", 0) < MIN_EFF_N:
        return "watch", f"חיובי בתקופה שלא נראתה, אבל רק {u.get('eff_n')} עסקאות אפקטיביות (צריך {MIN_EFF_N}) - נשמר להמשך"
    note = f" · נבדקו {c['tested']} שילובים, אז חלק מהיתרון עלול להיות מזל" if c.get("tested", 1) > 50 else ""
    return "regular", (f"חיובי בתקופה שלא נראתה ({u['mean']:+.2f}R לעסקה, {u['eff_n']} עסקאות אפקטיביות"
                       + (f", {gain:+.2f}R מעל המצב הקיים" if gain is not None else "") + f"){note}")


def write_cards(bot, cards, extra=None):
    d = CARDS_DIR / bot
    d.mkdir(parents=True, exist_ok=True)
    (d / "cards.json").write_text(json.dumps({"bot": bot, "updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                              "cards": cards, **(extra or {})}, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
