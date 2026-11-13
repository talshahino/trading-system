"""Tal's gap method on many instruments at once - one setting for a whole GROUP, chosen on the group's pooled trades
(user, 13.11.2026 10:55: "I don't accept that the world doesn't pass - find a way to pass most of them").

Why pooled: per instrument there are 30-200 trades a period, so the best of hundreds of settings on one instrument is
mostly luck (that is what the per-instrument test showed: chosen on 2024-2025, lost on 2026). One setting chosen on all
28 forex pairs together has thousands of trades behind it - a fair test, and if it holds on 2026 it trades all of them.
And what killed forex: the cost share of the risk and the quiet hours -> the new knobs MAXC / SESS (tal_gap.filt_ok),
with the bigger-picture filter TREND.

  python -m agents.tal_pool_gh run --symbols EURUSD,GBPUSD --out pool_out
  python -m agents.tal_pool_gh merge --parts parts

Groups: fx (28 FTMO pairs), metals (gold, silver, platinum), indices (bot 5's five + the eight others). Per group and
timeframe pair: chosen = max pooled train mean x eff_n (period to 2025-12-31 only), shown on 2026 pooled and per
instrument. A group/pair whose chosen setting earns on both periods (train t >= 1.5, 2026 mean > 0 on >= 60 trades)
gives a "new" card per instrument of the group that earned on the TRAIN period with it (chosen on train only).
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {"fx": ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD", "EURCAD",
                 "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD",
                 "GBPUSD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
          "metals": ["XAUUSD", "XAGUSD", "XPTUSD"],
          "indices": ["US100", "US30", "US500", "GER40", "JP225", "UK100", "FRA40", "EU50", "AUS200", "HK50", "N25",
                      "SPN35", "US2000"]}
GROUP_HE = {"fx": "צמדי המטבעות", "metals": "המתכות", "indices": "המדדים"}
MAXC = [None, 0.1, 0.2]
SESS = [None, "lnny"]
TREND = [None, "day"]
MIN_UNSEEN_POOL = 60


def cfgs_for(pn, chosen):
    """The base, the indices' choice and what runs on that pair, each with every MAXC x SESS x TREND."""
    from agents import tal_gap as TG
    bases = [dict(TG.BASE)] + [TG._norm(dict(c)) for c in chosen.get(pn, []) if c]
    out, seen = [], set()
    for b, mc, se, tr in itertools.product(bases, MAXC, SESS, TREND):
        c = {k: v for k, v in b.items() if k not in ("MAXC", "SESS", "TREND")}
        if mc:
            c["MAXC"] = mc
        if se:
            c["SESS"] = se
        if tr:
            c["TREND"] = tr
        if TG.vid(c) not in seen:
            seen.add(TG.vid(c))
            out.append(c)
    return out


def run(syms, out):
    import agent_bot5 as A5
    from agents import tal_gap as TG
    out = Path(out)
    (out / "idx").mkdir(parents=True, exist_ok=True)
    chosen = A5.wide_chosen()
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    t0 = time.time()
    for s in syms:
        D = A5.tal_load(s)
        if D is None:
            print(f"{s}: missing")
            continue
        gc, n = {}, 0
        for p in TG.PAIRS:
            pn = TG.pair_name(p)
            for c in cfgs_for(pn, chosen):
                key = f"{pn.replace('/', '_')}_{TG.vid(c)}_{s}"
                t = TG.simulate(D, p, c, gc)
                t.to_csv(out / f"{key}.csv.gz", index=False)
                (out / "idx" / f"{key}.json").write_text(json.dumps(
                    {"at": now, "pair": pn, "cfg": c, "sym": s, "file": f"{key}.csv.gz", "n": len(t)}, default=str))
                n += 1
        print(f"{s}: {n} runs ({time.time() - t0:.0f}s)")


def merge(parts):
    from agents import agentkit as K
    from agents import tal_gap as TG
    out = ROOT / "pool_merged"
    (out / "idx").mkdir(parents=True, exist_ok=True)
    for part in Path(parts).glob("*"):
        for f in part.rglob("*.csv.gz"):
            shutil.copy2(f, out / f.name)
        for f in part.rglob("idx/*.json"):
            shutil.copy2(f, out / "idx" / f.name)
    idx = [json.loads(f.read_text()) for f in (out / "idx").glob("*.json")]
    print(f"merged {len(idx)} runs")
    cache = {}

    def trades(v):
        if v["file"] not in cache:
            t = pd.read_csv(out / v["file"])
            t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
            cache[v["file"]] = t
        return cache[v["file"]]
    by = {}
    for v in idx:
        g = next((g for g, ss in GROUPS.items() if v["sym"] in ss), None)
        if g:
            by.setdefault((g, v["pair"], TG.vid(v["cfg"])), []).append(v)
    res, cards = {}, []
    s5 = json.loads((ROOT / "live/bots/bot5.json").read_text(encoding="utf-8"))
    rk = [float(s5["risk"][k]) for k in (s5.get("tal_tracks") or {}) if float((s5.get("risk") or {}).get(k, 0) or 0) > 0]
    risk_pct = round(min(rk) * 100, 3) if rk else 0.28
    live_sym = {(sy, f"{v['pair'][0]}/{v['pair'][1]}") for v in (s5.get("tal_tracks") or {}).values()
                for sy in (v.get("symbols") or [])}
    for g in GROUPS:
        for p in TG.PAIRS:
            pn = TG.pair_name(p)
            rows = []
            for (g_, pn_, _), vs in by.items():
                if g_ != g or pn_ != pn:
                    continue
                t = pd.concat([trades(v) for v in vs], ignore_index=True).sort_values("time")
                st = TG._stats(t)
                rows.append({"cfg": vs[0]["cfg"], "vs": vs, **st})
            ok = [r for r in rows if (r["train"]["eff_n"] or 0) >= K.MIN_EFF_N and r["train"]["mean"] is not None]
            if not ok:
                continue
            ch = max(ok, key=lambda r: r["train"]["mean"] * r["train"]["eff_n"])
            per = {}
            for v in ch["vs"]:
                st = TG._stats(trades(v))
                per[v["sym"]] = {"train": st["train"], "unseen": st["unseen"], "cost_r": st["cost_r"]}
            share = round(sum(1 for r in rows if (r["unseen"]["mean"] or -1) > 0) / len(rows), 2)
            res.setdefault(g, {})[pn] = {"cfg": ch["cfg"], "train": ch["train"], "unseen": ch["unseen"],
                                         "cost_r": ch["cost_r"], "tried": len(rows), "share_positive": share,
                                         "per_symbol": per}
            tr, u = ch["train"], ch["unseen"]
            if (tr["mean"] or 0) <= 0 or (tr.get("t") or 0) < 1.5 or (u["mean"] or 0) <= 0 or (u["n"] or 0) < MIN_UNSEEN_POOL:
                continue
            hi, lo = pn.split("/")
            name = f"{TG.TF_HE.get(hi, hi)} / {TG.TF_HE.get(lo, lo)}"
            for s, x in per.items():
                if (s, pn) in live_sym or (x["train"]["mean"] or 0) <= 0:
                    continue          # chosen on train only: an instrument that lost on 2024-2025 with it stays out
                what = (f"שיטת הגאפ של טל על {s}, {name}, בהגדרה אחת לכל {GROUP_HE[g]}: "
                        + ", ".join(f"{k}={v}" for k, v in ch["cfg"].items()) + ". "
                        f"נבחרה על כל {GROUP_HE[g]} יחד, 2024-2025 בלבד ({tr['mean']:+.2f}R לעסקה, {tr['n']} עסקאות), "
                        f"ומוצגת על 2026 על כולם: {u['mean']:+.2f}R לעסקה, {u['n']} עסקאות, {u['win']}% הצלחה. "
                        f"{s} עצמו: 2024-2025 {x['train']['mean']:+.2f}R ({x['train']['n']}), 2026 "
                        f"{(x['unseen']['mean'] or 0):+.2f}R ({x['unseen']['n']}). מסלול חדש רק למכשיר הזה, בסיכון של מסלולי הגאפ.")
                cards.append(K.card("bot5", "new", f"שיטת הגאפ של טל · {s} · {name} (הגדרת קבוצה)", what, tr, u,
                                    whole=ch["whole"], halves_=ch["halves"], cost_r=x["cost_r"], tested=len(rows),
                                    risk_pct=risk_pct, neighbours={"values": [], "share_positive": share},
                                    change={"method": "tal_gap_sym", "sym": s, "pair": pn, "cfg": ch["cfg"],
                                            "track": f"tal_gap|{pn}|{s}", "knob": "cfg"}))
    doc = json.loads(TG.RESULTS.read_text(encoding="utf-8"))
    doc.setdefault("bot5", {})["pool"] = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                          "groups": res, "runs": len(idx)}
    TG.RESULTS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (ROOT / "reports/agents/bot5/cards_pool.json").write_text(json.dumps(
        {"bot": "bot5", "updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "cards": cards},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for g, ps in res.items():
        for pn, x in ps.items():
            print(g, pn, x["cfg"], "train", x["train"]["mean"], x["train"]["n"], x["train"].get("t"),
                  "2026", x["unseen"]["mean"], x["unseen"]["n"], x["unseen"].get("t"),
                  "pos", sum(1 for y in x["per_symbol"].values() if (y["unseen"]["mean"] or -1) > 0), "/", len(x["per_symbol"]))
    print(f"{len(cards)} new cards")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "merge"])
    ap.add_argument("--symbols", default="")
    ap.add_argument("--out", default="pool_out")
    ap.add_argument("--parts", default="parts")
    a = ap.parse_args()
    if a.mode == "run":
        run([s for s in a.symbols.split(",") if s], a.out)
    else:
        merge(a.parts)
