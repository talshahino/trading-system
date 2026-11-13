"""Tal's gap method on every FTMO forex pair, the metals and the other indices - NOW, on GitHub in parallel (user,
13.11.2026 10:11: "I want it in today's trading, move it ahead of everything"). The same test as agents/tal_wide.py
(which runs on the bots' computer a slice at a time), split over many GitHub machines:

  python -m agents.tal_wide_gh run --symbols XAUUSD,EURUSD --minutes 140 --out wide_out
  python -m agents.tal_wide_gh merge --parts parts/ --out reports/agents/bot5/tal/wide

run:   phase 1 (base, the indices' choice, what bot 5 runs) on its instruments first, then phase 2 (every variant)
merge: all parts into one run directory -> tal_gap_results.json bot5.wide (+ the table) and the "new" cards
       (reports/agents/bot5/cards_wide.json) - chosen on the period to 2025-12-31 only, shown on 2026 (tal_wide.cards)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "merge"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--minutes", type=float, default=140)
    ap.add_argument("--out", default="wide_out")
    ap.add_argument("--parts", default="parts")
    ap.add_argument("--exit", default="", help="tp30: every variant with the whole position out at 30%% of the way")
    ap.add_argument("--key", default="wide", help="tal_gap_results.json bot5.<key>")
    a = ap.parse_args()
    import agent_bot5 as A5
    from agents import tal_wide as TW
    if a.mode == "run":
        syms = [s.strip() for s in a.symbols.split(",") if s.strip()]
        if not a.exit:
            TW.run(A5.tal_load, Path(a.out), a.minutes * 60, A5.wide_chosen(), symbols=syms)
            return
        # one EXIT on every variant of every pair (user, 13.11.2026: "take 100% at 30% of the way - try it")
        import time
        import pandas as pd
        from agents import tal_gap as TG
        out = Path(a.out)
        (out / "idx").mkdir(parents=True, exist_ok=True)
        t0, now, missing = time.time(), pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), {}
        cfgs, seen = [], set()
        for c in [dict(TG.BASE)] + [cfg for _, _, cfg in TG.variants()]:
            c = dict(c, EXIT=a.exit)
            if TG.vid(c) not in seen:
                seen.add(TG.vid(c))
                cfgs.append(c)
        for s in syms:
            if time.time() - t0 > a.minutes * 60:
                break
            D = A5.tal_load(s)
            if D is None:
                missing[s] = {"at": now, "why": "no minute data / spread"}
                continue
            gc = {}
            for p in TG.PAIRS:
                pn = TG.pair_name(p)
                for c in cfgs:
                    key = f"{pn.replace('/', '_')}_{TG.vid(c)}_{s}"
                    t = TG.simulate(D, p, c, gc)
                    t.to_csv(out / f"{key}.csv.gz", index=False)
                    (out / "idx" / f"{key}.json").write_text(json.dumps(
                        {"at": now, "pair": pn, "phase": 2, "cfg": c, "sym": s, "file": f"{key}.csv.gz", "n": len(t)}, default=str))
            print(f"{s}: {len(cfgs) * len(TG.PAIRS)} runs ({time.time() - t0:.0f}s)")
        (out / TW.STATUS).write_text(json.dumps({"at": now, "symbols": syms, "missing": missing}))
        return
    from agents import tal_gap as TG
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    (out / "idx").mkdir(parents=True, exist_ok=True)
    missing, symbols = {}, []
    for part in sorted(Path(a.parts).glob("*")):
        if not part.is_dir():
            continue
        for f in part.rglob("*.csv.gz"):
            shutil.copy2(f, out / f.name)
        for f in part.rglob("idx/*.json"):
            shutil.copy2(f, out / "idx" / f.name)
        for f in part.rglob(TW.STATUS):
            try:
                st = json.loads(f.read_text())
                missing.update(st.get("missing") or {})
                symbols += st.get("symbols") or []
            except Exception:  # noqa: BLE001
                pass
    widx = {f.stem: json.loads(f.read_text()) for f in (out / "idx").glob("*.json")}
    print(f"merged {len(widx)} runs, missing {sorted(missing)}")
    summ = TW.summarize(widx, out)
    st = {"at": max((v["at"] for v in widx.values()), default=None), "symbols": symbols, "missing": missing,
          "runs": len(widx), "where": "GitHub (agents/tal_wide_gh.py)"}
    (out / TW.STATUS).write_text(json.dumps(st, ensure_ascii=False, default=str))
    doc = json.loads(TG.RESULTS.read_text(encoding="utf-8"))
    doc.setdefault("bot5", {})[a.key] = {"status": st, "table": TW.table(summ), "pairs": summ}
    TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    s5 = json.loads((ROOT / "live/bots/bot5.json").read_text(encoding="utf-8"))
    live_sym = {(sy, f"{v['pair'][0]}/{v['pair'][1]}") for v in (s5.get("tal_tracks") or {}).values()
                for sy in (v.get("symbols") or [])}
    rk = [float(s5["risk"][k]) for k in (s5.get("tal_tracks") or {}) if float((s5.get("risk") or {}).get(k, 0) or 0) > 0]
    cards = TW.cards("bot5", summ, round(min(rk) * 100, 3) if rk else 0.28, len(widx), live_sym)
    import pandas as pd
    (ROOT / f"reports/agents/bot5/cards_{a.key}.json").write_text(json.dumps(
        {"bot": "bot5", "updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "cards": cards},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for r in TW.table(summ)[:40]:
        print(r)
    print(f"{len(cards)} new cards")


if __name__ == "__main__":
    main()
