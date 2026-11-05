"""Risk per strategy ("ניהול בוטים" in the app): every combo of every bot gets its demo record and a
recommended risk multiplier. Nothing changes until the user approves it in the app, one by one.

The rule (agreed with the user, 05.11.2026):
  start      every combo trades at the bot's risk (multiplier 1)
  evidence   the combo's mean R per trade in the bot's accounts since 2026-11-01, against the whole bot's
             mean R; weight w = n / (n + 20): 2 trades -> 0.09, 10 -> 0.33, 20 -> 0.5, 60 -> 0.75
  target     1 + 0.5 * w * clip((combo mean - bot mean) / 0.4 R, -1, 1)
  limits     0.5 .. 1.5; the bot's combos together keep an average of 1 (the bot never risks more in total)
  advice     only when the target is at least 0.05 away from the current multiplier (steps of 0.05)
  sample     recommendations are refreshed once a week (sample = the week's Monday); a declined one comes
             back only in a later sample, and only if the record still calls for it
  removal    GATE.md rule 4 (trade_review.removal_rule): proposed as "remove" = multiplier 0, no new trades

  python risk_review.py --states <dir with *_state.json> [--manager <manager_state.json>]
Writes reports/trade_review/risk.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import trade_review as TR

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "trade_review" / "risk.json"
HOLDOUT_FROM = TR.HOLDOUT_FROM
K_TRADES, FULL_EDGE_R, SPAN = 20, 0.4, 0.5
LO, HI, STEP = 0.5, 1.5, 0.05


def demo_trades(states):
    """All closed demo trades since 2026-11-01, with R where the first stop is known."""
    rows = []
    for f in sorted(Path(states).glob("*_state.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        acc = f.stem.replace("_state", "")
        for c in s.get("closed") or []:
            try:
                side = 1 if c["side"] == "buy" else -1
                o, sl0 = float(c["open_px"]), float(c.get("sl0") or 0)
                risk = side * (o - sl0) if sl0 else 0
                r = side * (float(c["close_px"]) - o) / risk if risk > 0 else np.nan
                rows.append({"account": acc, "bot": acc.split("_")[0], "id": str(c["id"]), "combo": c["combo"],
                             "open_t": c["open_t"], "r": r, "net": float(c.get("net") or 0)})
            except Exception:  # noqa: BLE001
                continue
    t = pd.DataFrame(rows, columns=["account", "bot", "id", "combo", "open_t", "r", "net"])
    if len(t):
        t = t[pd.to_datetime(t["open_t"], utc=True) >= pd.Timestamp(HOLDOUT_FROM, tz="UTC")]
        t = t.drop_duplicates(["account", "id"])
    return t


def targets(stats, bot_mean, now):
    """Target multiplier per combo, then scaled so the active combos average 1 inside 0.5..1.5."""
    raw = {}
    for c, s in stats.items():
        if now.get(c, 1.0) == 0:
            continue                                   # removed: stays at 0 until the user brings it back
        n = s["n_r"]
        w = n / (n + K_TRADES)
        rel = float(np.clip((s["mean_r"] - bot_mean) / FULL_EDGE_R, -1, 1)) if n else 0.0
        raw[c] = 1 + SPAN * w * rel
    if not raw:
        return {}
    v = np.array(list(raw.values()))
    for _ in range(50):
        v = np.clip(v / v.mean(), LO, HI)
    return {c: float(x) for c, x in zip(raw, v)}


def week_of(ts):
    d = ts.tz_convert("Asia/Jerusalem").normalize()
    return (d - pd.Timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--manager", default="")
    args = ap.parse_args()
    now_ts = pd.Timestamp.now(tz="UTC")
    sample = week_of(now_ts)
    mgr = json.loads(Path(args.manager).read_text(encoding="utf-8")) if args.manager and Path(args.manager).exists() else {}
    decided = mgr.get("risk") or {}
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t = demo_trades(args.states)
    removal = {}
    try:
        lt = TR.live_trades(args.states)
        if len(lt):
            removal = {r["combo"]: r for r in TR.removal_rule(lt)}
    except Exception as e:  # noqa: BLE001
        print("removal rule skipped:", e)
    out = {"updated": now_ts.isoformat(timespec="seconds"), "sample": sample, "since": HOLDOUT_FROM,
           "rule": {"k_trades": K_TRADES, "full_edge_r": FULL_EDGE_R, "span": SPAN, "lo": LO, "hi": HI, "step": STEP},
           "bots": {}}
    for b in (1, 2, 3, 4):
        bot = f"bot{b}"
        cfg = json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
        mine = t[t["bot"] == bot] if len(t) else t
        cur = {k: float(v) for k, v in ((decided.get(bot) or {}).get("mult") or {}).items()}
        rejected = (decided.get(bot) or {}).get("rejected") or {}
        bot_mean = float(mine["r"].mean()) if len(mine) and mine["r"].notna().any() else 0.0
        stats = {}
        for c in cfg["combos"]:
            g = mine[mine["combo"] == c] if len(mine) else mine
            rr = g["r"].dropna() if len(g) else pd.Series(dtype=float)
            stats[c] = {"trades": int(len(g)), "wins": int((g["net"] > 0).sum()) if len(g) else 0,
                        "n_r": int(len(rr)), "mean_r": float(rr.mean()) if len(rr) else 0.0,
                        "sum_r": float(rr.sum()) if len(rr) else 0.0, "net": float(g["net"].sum()) if len(g) else 0.0}
        tg = targets(stats, bot_mean, cur)
        # recommendations are frozen for the week: a new sample replaces them, a daily run only refreshes the record
        old = ((prev.get("bots") or {}).get(bot) or {}).get("combos") or []
        frozen = {x["combo"]: x.get("rec") for x in old} if prev.get("sample") == sample else None
        rows = []
        for c in cfg["combos"]:
            s, m_now = stats[c], cur.get(c, 1.0)
            if frozen is not None and c in frozen:
                rec = frozen[c]
            else:
                rec = None
                if m_now > 0 and c in tg:
                    m = round(round(tg[c] / STEP) * STEP, 2)
                    if abs(m - m_now) >= STEP - 1e-9:
                        rec = {"mult": m, "action": "up" if m > m_now else "down"}
            rm = removal.get(c)
            if rm and rm.get("remove") and m_now > 0:
                rec = {"mult": 0.0, "action": "remove",
                       "why": f"{rm['cum_r']:+.1f}R ב־{rm['trades']} עסקאות, מתחת לקו {rm['line_r']:.1f}R (כלל ההסרה)"}
            if rec and abs(rec["mult"] - m_now) < 1e-9:
                rec = None                             # already applied
            if rec and rejected.get(c) == sample and rec["action"] != "remove":
                rec = dict(rec, rejected=True)
            rows.append({"combo": c, "trades": s["trades"], "wins": s["wins"],
                         "win_pct": round(100 * s["wins"] / s["trades"], 1) if s["trades"] else None,
                         "mean_r": round(s["mean_r"], 3), "sum_r": round(s["sum_r"], 2), "net": round(s["net"], 2),
                         "mult": m_now, "risk_pct": round(cfg["risk"] * 100 * m_now, 4),
                         "target": round(tg.get(c, m_now), 3), "rec": rec})
        out["bots"][bot] = {"risk_pct": cfg["risk"] * 100, "trades": int(len(mine)),
                            "mean_r": round(bot_mean, 3), "combos": rows}
        print(bot, len(mine), "trades,", sum(1 for r in rows if r["rec"]), "recommendations")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
