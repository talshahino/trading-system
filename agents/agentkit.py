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


NEW_MIN_TOTAL = 3.0       # R on 2026: the least a new strategy must add to the total

# A SMOOTHER CURVE (the user, 16.11.2026 20:28, task mgr1611curve: "many small wins every day rather than one big win";
# data/agents/tal_thinking.md: "a way that clearly smooths the curve on both periods and still earns comes to the
# manager as a card even if its total is somewhat lower - the manager decides"). Until 17.11.2026 such a card was
# rejected by the per-trade rule below before the manager could weigh it (bot 6's smart agent, 17.11 night: bot 6's
# first two curve cards, 7d78b68e5e0c and 22290561222a, both "rejected"). A card whose change says test "curve" carries
# c["curve"] = {"train": {"live": {...}, "way": {...}}, "unseen": {...}}, each with sum (R), green (% green days),
# dd (largest fall from the peak, R), worst (worst day, R) and sd (daily spread, R). It passes when on EACH period:
# more green days (CURVE_GREEN points), a smaller fall, a worst day not worse, at least CURVE_KEEP of today's total -
# and THE MAIN GOAL (more profit AND less risk): the profit per unit of daily risk (sum / sd) is not lower, so less
# profit is accepted only together with at least as much less risk (the same rule as "more profit only with less risk").
CURVE_GREEN = 2.0
CURVE_KEEP = 0.8


def curve_ok(c):
    """(ok, reason) for a card with a "curve" block (see above); (None, "") when the block is missing / incomplete."""
    cv = c.get("curve") or {}
    parts = []
    for p, he in (("train", "עד 2025"), ("unseen", "2026")):
        x = cv.get(p) or {}
        a, b = x.get("way") or {}, x.get("live") or {}
        if any(d.get(k) is None for d in (a, b) for k in ("sum", "green", "dd", "worst", "sd")):
            return None, ""
        ra = a["sum"] / a["sd"] if a["sd"] else None
        rb = b["sum"] / b["sd"] if b["sd"] else None
        fails = []
        if not (a["sum"] > 0 and a["sum"] >= CURVE_KEEP * b["sum"]):
            fails.append(f"סה\"כ {b['sum']:+.1f}R ← {a['sum']:+.1f}R (פחות מ־{CURVE_KEEP:.0%})")
        if a["green"] < b["green"] + CURVE_GREEN:
            fails.append(f"ימים ירוקים {b['green']}% ← {a['green']}%")
        if not a["dd"] < b["dd"]:
            fails.append(f"ירידה מהשיא {b['dd']}R ← {a['dd']}R")
        if a["worst"] < b["worst"]:
            fails.append(f"היום הגרוע {b['worst']}R ← {a['worst']}R")
        if ra is None or rb is None or ra < rb:
            fails.append(f"רווח ליחידת סיכון יומי {rb or 0:.1f} ← {ra or 0:.1f}")
        if fails:
            return False, f"{he}: " + ", ".join(fails)
        parts.append(f"{he}: ימים ירוקים {b['green']}% ← {a['green']}%, ירידה {b['dd']}R ← {a['dd']}R, "
                     f"רווח ליחידת סיכון {rb:.1f} ← {ra:.1f}")
    return True, "; ".join(parts)


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
    # THE GOAL (user, 12.11.2026, data/agents/objective.md): chosen only on the period to 2025-12-31, 2026 only shows
    # the result. A proposal that lost to 2025 and is positive only in 2026 was chosen by 2026 - out of hundreds of
    # scout strategies some are positive in 2026 by chance (manager, 13.11.2026: 18 of 32 approved new strategies,
    # e.g. SC002|1Hour|ARM in four bots, -0.09R to 2025 and +0.00R in 2026)
    t = c.get("train") or {}
    if t.get("mean") is not None and (t.get("eff_n") or t.get("n") or 0) >= 10 and t["mean"] <= 0:
        return "rejected", (f"הפסיד בתקופה עד 2025 ({t['mean']:+.2f}R לעסקה) - חיובי רק ב־2026, כלומר נבחר לפי 2026 "
                            "ועלול להיות מזל")
    # the same for a barely positive choosing period: a new strategy must earn its place on the period to 2025 too -
    # at least NEW_MIN_TOTAL R there, or a clear t >= 1 (manager, 14.11.2026: SC069|15Min|XAUUSD, +0.01R x 107 = +1.2R
    # to 2025 and t 0.15, was approved into bots 1, 4 and 5 on its 2026 result; SC020|15Min|GBPJPY +0.1R, t 0.01 -
    # both picked by 2026, against objective.md)
    tn = (t.get("eff_n") or t.get("n") or 0)
    if (c["kind"] == "new" and base is None and t.get("mean") is not None and tn >= 10
            and tn * t["mean"] < NEW_MIN_TOTAL and (t.get("t") or 0) < 1.0):
        return "watch", (f"עד 2025 בסך הכל רק {tn * t['mean']:+.1f}R (t {t.get('t') or 0:.2f}) - "
                         f"היתרון נראה רק ב־2026, כלומר נבחר לפי 2026. נשמר להמשך")
    # a new strategy must add to the TOTAL: under +3R on the whole of 2026 it only adds risk and a share of the daily
    # loss limit (manager, 13.11.2026: SC021|5Min|XAUUSD, +0.001R x 188 = +0.2R in 2026, +0.02R to 2025, approved into
    # bot 5; SC002|1Hour|ARM, +0.2R in 2026, ran in four bots)
    if c["kind"] == "new" and base is None and (u.get("eff_n") or u.get("n") or 0) * u["mean"] < NEW_MIN_TOTAL:
        return "watch", (f"ב־2026 בסך הכל רק {(u.get('eff_n') or u.get('n') or 0) * u['mean']:+.1f}R "
                         f"({u['mean']:+.3f}R לעסקה) - פחות מ־{NEW_MIN_TOTAL:g}R, לא מגדיל את הרווח הכולל. נשמר להמשך")
    if (c.get("change") or {}).get("test") == "curve" and c.get("curve"):
        ok, why = curve_ok(c)
        if ok:
            return "regular", ("עקומה חלקה יותר בשתי התקופות, והרווח ליחידת סיכון לא יורד (בקשת טל 16.11, "
                               "הרבה ניצחונות קטנים) - " + why)
        if ok is False:
            return "rejected", "העקומה לא חלקה מספיק בשתי התקופות - " + why
    if base is not None and u["mean"] <= base:
        return "rejected", f"לא טוב מהמצב הקיים בתקופה שלא נראתה ({u['mean']:+.2f}R מול {base:+.2f}R)"
    # more per trade but fewer trades can mean less money in total (a limit entry, a filter): the total on the
    # unseen period must not fall below today's (manager, 12.11.2026 - bot 1 TV21|15Min|ASML: +0.35R x 73 = +25.7R
    # was approved over +0.16R x 292 = +46.6R)
    bn = (c.get("baseline") or {}).get("n")
    if base is not None and bn and u.get("n"):
        tot, tot0 = u["n"] * u["mean"], bn * base
        if tot < tot0:
            return "watch", (f"יותר רווח לעסקה אבל פחות רווח בסך הכל בתקופה שלא נראתה ({u['n']} עסקאות {tot:+.1f}R"
                             f" מול {bn} עסקאות {tot0:+.1f}R היום) - נשמר להמשך")
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


def json_layout(cards):
    """indent=1 for a readable file; without the whitespace when the file is big (15.11.2026: bot 2's 25k cards)."""
    return {"indent": 1} if len(cards) <= 2000 else {"separators": (",", ":")}


REJECTED_SHARDS = 4      # cards_rejected_0..3.json - always all four, so no stale shard is ever left behind


def rejected_new(c):
    return c.get("kind") == "new" and c.get("tier") == "rejected"


def write_cards(bot, cards, extra=None):
    """cards.json + the rejected new strategies in REJECTED_SHARDS files by id (manager, 15.11.2026 night: with the
    scout's pack bot 1 has 45,125 cards - cards.json 36 MB even with the short rejected cards; GitHub refused bot 2's
    40 MB file (422) and bot 1's 21:40 run was lost, so the app and the GitHub agents saw 14.11 09:01's file).
    agent_manager / autodecide read every cards*.json, agent_portfolio.families_of reads them too."""
    d = CARDS_DIR / bot
    d.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    keep = [c for c in cards if not rejected_new(c)]
    shards = [[] for _ in range(REJECTED_SHARDS)]
    for c in cards:
        if rejected_new(c):
            shards[int(hashlib.sha1(str(c.get("id")).encode()).hexdigest()[:6], 16) % REJECTED_SHARDS].append(c)
    (d / "cards.json").write_text(json.dumps({"bot": bot, "updated": now, "cards": keep, **(extra or {}),
                                              "rejected_new_in": f"cards_rejected_0..{REJECTED_SHARDS - 1}.json"},
                                             ensure_ascii=False, default=str, **json_layout(keep)),
                                  encoding="utf-8")
    for i, sh in enumerate(shards):
        (d / f"cards_rejected_{i}.json").write_text(json.dumps({"bot": bot, "updated": now, "cards": sh},
                                                               ensure_ascii=False, default=str, **json_layout(sh)),
                                                    encoding="utf-8")


def read_cards(bot, with_rejected=True):
    """Every card of the bot's nightly experiments: cards.json + the rejected-new shards (see write_cards)."""
    d = CARDS_DIR / bot
    names = ["cards.json"] + ([f"cards_rejected_{i}.json" for i in range(REJECTED_SHARDS)] if with_rejected else [])
    out = []
    for nm in names:
        try:
            out += json.loads((d / nm).read_text(encoding="utf-8")).get("cards", [])
        except Exception:  # noqa: BLE001
            pass
    return out
