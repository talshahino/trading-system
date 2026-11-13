"""A fast, fair test of the fixes for one gap track of bot 5 after a loss report (user, 13.11.2026: "no 'normal bad
day' - every losing track gets a tested fix, with its worth in % of the account a month, and it is built; only if
there is no fix - lower the risk").

  python -m agents.tal_fix_run --track t15_1

The track's rules as they run now (live/bots/bot5.json) against one family at a time - the gap's age (EXPIRY), the
bigger picture (TREND), the trade management (EXIT), and combinations - on bot 5's five indices with FTMO's real
1-minute spread + commission (agent_bot5.tal_load). CHOSEN on the period to 2025-12-31 only (the total time-weighted
R), shown on 2026 (unseen). The winner is an 'improve' card the builder builds at once (change {"method": "tal_gap"}),
only when it beats what runs on BOTH periods; its worth = (2026 total R new - now) / months x the track's risk.
data/agents/fix_attempts.json "bot5|<track>" is updated: fixed / failed. Runs on GitHub (workflow tal_fix.yml) or on
the bots' computer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents import agentkit as K
from agents import tal_gap as TG

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "tal_fix"
FIX = ROOT / "data" / "agents" / "fix_attempts.json"
CARDS = ROOT / "reports" / "agents" / "bot5" / "cards_smart.json"
SPLIT = TG.SPLIT


def candidates(cfg):
    """(family, cfg) - one family at a time, then the strongest pairs of families."""
    out = []
    for e in (8, 12, 16, 24, 48):
        if e != cfg.get("EXPIRY"):
            out.append(("EXPIRY", dict(cfg, EXPIRY=e)))
    for m in ("day", "up20"):
        out.append(("TREND", dict(cfg, TREND=m)))
    for x in ("be50", "be70", "be382", "be618", "p50_618"):
        out.append(("EXIT", dict(cfg, EXIT=x)))
    for e in (16, 24):
        for m in ("day", "up20"):
            out.append(("EXPIRY+TREND", dict(cfg, EXPIRY=e, TREND=m)))
    return [(f, TG._norm(c)) for f, c in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="t15_1")
    a = ap.parse_args()
    import agent_bot5 as A
    s5 = json.loads((ROOT / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    v = s5["tal_tracks"][a.track]
    pair = next(p for p in TG.PAIRS if p[0] == v["pair"][0] and p[1] == v["pair"][1])
    pn = TG.pair_name(pair)
    now_cfg = TG._norm(dict(v["cfg"]))
    risk = float((s5.get("risk") or {}).get(a.track, 0.0028)) * float(s5.get("tal_risk_scale", 1.0) or 1.0)
    cands = [("now", now_cfg)] + candidates(now_cfg)
    frames = {i: [] for i in range(len(cands))}
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in A.TAL_SYMBOLS:
        D = A.tal_load(sym)
        if D is None:
            raise SystemExit(f"no data for {sym}")
        gc = {}
        for i, (_, c) in enumerate(cands):
            t = TG.simulate(D, pair, c, gc)
            t["sym"] = sym
            frames[i].append(t)
        print(f"{sym}: {len(cands)} versions")
    rows = []
    months = max(1.0, (pd.Timestamp.now(tz="UTC") - SPLIT).days / 30.4)
    for i, (fam, c) in enumerate(cands):
        t = pd.concat(frames[i], ignore_index=True)
        t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
        t.to_csv(OUT / f"{a.track}_{TG.vid(c)}.csv.gz", index=False)
        un = t[t["time"] >= SPLIT]
        day = un.groupby(un["time"].dt.strftime("%Y-%m-%d"))["r"].sum() if len(un) else pd.Series(dtype=float)
        cum = un["r"].cumsum()
        rows.append({"family": fam, "cfg": c, "train_w": round(TG.score(t), 2), "unseen_total": round(float(un["r"].sum()), 2),
                     "unseen_worst_day": round(float(day.min()), 2) if len(day) else None,
                     "unseen_max_fall": round(float((cum.cummax() - cum).max()), 2) if len(cum) else None,
                     **{k: v_ for k, v_ in TG._stats(t).items()}})
    now = rows[0]
    best = max(rows[1:], key=lambda r: r["train_w"])          # chosen on 2024-2025 only
    win = best["train_w"] > now["train_w"] and best["unseen_total"] > now["unseen_total"]
    worth = round((best["unseen_total"] - now["unseen_total"]) / months * risk * 100, 3)
    doc = {"track": a.track, "pair": pn, "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "risk": risk,
           "months_2026": round(months, 1), "now": now, "chosen": best, "fixed": bool(win), "worth_month_pct": worth if win else 0.0,
           "rows": [{k: r[k] for k in ("family", "cfg", "train_w", "unseen_total", "unseen_worst_day", "unseen_max_fall")} | {
               "train_mean": r["train"]["mean"], "unseen_mean": r["unseen"]["mean"], "unseen_n": r["unseen"]["n"]} for r in rows]}
    (OUT / f"{a.track}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("track", "fixed", "worth_month_pct")}, ensure_ascii=False))
    for r in doc["rows"]:
        print(f"  {r['family']:<13} train {r['train_w']:+8.1f}  2026 {r['unseen_total']:+7.1f}R  n {r['unseen_n']}  {r['cfg']}")
    # the fix record
    try:
        fx = json.loads(FIX.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        fx = {}
    k = f"bot5|{a.track}"
    rec = fx.setdefault(k, {"status": "trying", "tried": []})
    rec.setdefault("tried", []).append(
        f"{doc['at'][:10]}: tal_fix_run - {len(cands) - 1} גרסאות (תוקף הגאפ, TREND, ניהול, שילובים) מול מה שרץ "
        f"({now['train_w']:+.1f}R משוקלל ב־2024-2025 / {now['unseen_total']:+.1f}R ב־2026). נבחרה על 2024-2025: {best['family']} "
        f"{json.dumps({kk: vv for kk, vv in best['cfg'].items() if now_cfg.get(kk) != vv})} - {best['train_w']:+.1f}R / {best['unseen_total']:+.1f}R"
        + (f" → תיקון, {worth:+.2f}% מהחשבון בחודש." if win else " → לא עדיפה בשתי התקופות."))
    rec.update(status="fixed" if win else "failed", at=doc["at"], by="bot5 agent (tal_fix_run)")
    FIX.write_text(json.dumps(fx, ensure_ascii=False, indent=1), encoding="utf-8")
    if not win:
        return
    hi, lo = pn.split("/")
    chg = {kk: vv for kk, vv in best["cfg"].items() if now_cfg.get(kk) != vv}
    what = (f"תיקון להפסד של 12.11 במסלול {a.track} ({TG.TF_HE[hi]} / {TG.TF_HE[lo]}): " + ", ".join(f"{kk} = {vv}" for kk, vv in chg.items())
            + f". נבדקו {len(cands) - 1} גרסאות של משפחות שונות מול הכללים שרצים, בחירה על 2024-2025 בלבד (סה\"כ משוקלל "
            f"{now['train_w']:+.1f}R ← {best['train_w']:+.1f}R), הצגה על 2026: סה\"כ {now['unseen_total']:+.1f}R ← {best['unseen_total']:+.1f}R, "
            f"{best['unseen']['n']} עסקאות, {best['unseen']['mean']:+.2f}R לעסקה (היום {now['unseen']['mean']:+.2f}R על {now['unseen']['n']}); "
            f"היום הגרוע {now['unseen_worst_day']:+.1f}R ← {best['unseen_worst_day']:+.1f}R, הירידה הגדולה {now['unseen_max_fall']:.1f}R ← "
            f"{best['unseen_max_fall']:.1f}R. שווי: {worth:+.2f}% מהחשבון בחודש (בסיכון {risk * 100:.2f}% לעסקה).")
    card = K.card("bot5", "improve", f"שיטת הגאפ · {TG.TF_HE[hi]} / {TG.TF_HE[lo]}: תיקון ההפסד של 12.11 ({best['family']})", what,
                  best["train"], best["unseen"], whole=best["whole"], halves_=best["halves"], cost_r=best["cost_r"],
                  tested=len(cands) - 1, risk_pct=round(risk * 100, 3), baseline=now["unseen"],
                  neighbours={"values": [r["family"] for r in rows[1:]],
                              "share_positive": round(float(np.mean([r["unseen_total"] > now["unseen_total"] for r in rows[1:]])), 2)},
                  detail={"tal_fix": f"reports/agents/bot5/tal_fix/{a.track}.json", "worth_month_pct": worth},
                  change={"method": "tal_gap", "pair": pn, "cfg": best["cfg"], "track": f"tal_gap|{pn}", "knob": "cfg"})
    card["tier"], card["verdict"] = K.judge(card)
    try:
        cd = json.loads(CARDS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        cd = {"bot": "bot5", "cards": []}
    cd["cards"] = [c for c in cd.get("cards", []) if c.get("id") != card["id"]] + [card]
    cd["updated"] = doc["at"]
    CARDS.write_text(json.dumps(cd, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("card:", card["id"], card["tier"], card["verdict"])


if __name__ == "__main__":
    main()
