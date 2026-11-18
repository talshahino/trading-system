"""Gold strategies of bots 1-4: no new entry in the minutes before an important USD release (task tal1811gold4h).

Bot 7's rule (card ee3116706ab9) tested on the funded bots with their own trades as they trade now: windows of
1 / 2 / 4 hours before a high (or high + medium) impact USD release. One setting for all the gold strategies, chosen
on 30.04.2025-31.12.2025 over the four bots together, shown on 2026; every number is the whole bot's replay
(agents/portfolio_check.replay, with the fixed 4.5% / 9% open-risk rule). Output: reports/agents/gold_release_study.json
"""
import bisect
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents import portfolio_check as PC  # noqa: E402

BOTS = ("bot1", "bot2", "bot3", "bot4")
WINDOWS = (60, 120, 240)
IMPACTS = {"high": {"high"}, "high+medium": {"high", "medium"}}


def releases(impact):
    items = json.loads((ROOT / "reports" / "agents" / "bot7" / "calendar_full.json").read_text(encoding="utf-8"))["items"]
    return sorted({int(pd.Timestamp(i["date"]).timestamp() // 60) for i in items if i.get("date")
                   and i.get("impact") in impact and str(i.get("instrument", "")).split(" ")[0] == "USD"})


def blocked(minute, rel, win):
    k = bisect.bisect_left(rel, minute)
    return k < len(rel) and rel[k] - minute <= win


def filt(book, rel, win):
    out, saved = {}, []
    for c, (tr, m) in book.items():
        if not c.endswith("|XAUUSD"):
            out[c] = (tr, m)
            continue
        keep = [t for t in tr if not blocked(t[0], rel, win)]
        saved += [t[3] * m for t in tr if blocked(t[0], rel, win)]
        out[c] = (keep, m)
    return out, {"n": len(saved), "r": round(sum(saved), 2)}


def main():
    rows, books = {}, {}
    for b in BOTS:
        cfg = PC.bot_cfg(b)
        books[b] = {k: PC.book_now(b, cfg, k)[0] for k in ("tr25", "tr")}
        books[b]["risk"] = PC.bot_risk(b, cfg)
    base = {b: {k: PC.replay(books[b][k], books[b]["risk"]) for k in ("tr25", "tr")} for b in BOTS}
    for name, imp in IMPACTS.items():
        rel = releases(imp)
        for w in WINDOWS:
            v = f"{name}|pre{w}"
            rows[v] = {}
            for b in BOTS:
                rows[v][b] = {}
                for k in ("tr25", "tr"):
                    bk, sv = filt(books[b][k], rel, w)
                    rows[v][b][k] = {**PC.replay(bk, books[b]["risk"]), "skipped": sv}
    score = {v: round(sum(r[b]["tr25"]["total"] - base[b]["tr25"]["total"] for b in BOTS), 2) for v, r in rows.items()}
    best = max(score, key=score.get)
    verdict = {}
    for b in BOTS:
        a, z = rows[best][b], base[b]
        verdict[b] = bool(a["tr25"]["total"] > z["tr25"]["total"] and a["tr"]["total"] > z["tr"]["total"]
                          and a["tr25"]["max_dd"] <= z["tr25"]["max_dd"] and a["tr"]["max_dd"] <= z["tr"]["max_dd"])
    doc = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "task": "tal1811gold4h",
           "gold": {b: [c for c in books[b]["tr"] if c.endswith("|XAUUSD")] for b in BOTS if b in books},
           "base": base, "score_2025": score, "best": best, "passes": verdict, "rows": rows}
    (ROOT / "reports" / "agents" / "gold_release_study.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("best on 2025:", best, score)
    for b in BOTS:
        print(b, "base 25/26", base[b]["tr25"]["total"], base[b]["tr25"]["max_dd"], "|", base[b]["tr"]["total"], base[b]["tr"]["max_dd"])
        for v in rows:
            r = rows[v][b]
            print("   ", v, r["tr25"]["total"], r["tr25"]["max_dd"], r["tr25"]["skipped"], "|", r["tr"]["total"], r["tr"]["max_dd"], r["tr"]["skipped"])
    print("passes", verdict)


if __name__ == "__main__":
    main()
