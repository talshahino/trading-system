"""A second look at the strategies that did not pass (user, 20.11.2026 22:11: "run a fresh test on all the strategies,
including scenarios to maximise profit and cut risk - moving the stop, the target or the entry - maybe we find mistakes
and put in good strategies").

Every scout strategy x instrument the bots' agents tested (reports/agents/<bot>/exp/pack, the "new" family, as the
strategy's own words) that did NOT pass but came close - at least 60 trades on 30.04-31.12.2025 and a mean of at least
-0.03R a trade there (an edge the costs or the exits may have eaten) - through the exits grid of the backtest machine
(run_rules --optimize, the same 72 variants the bots use: target own / 1 / 1.5 / 2 / 3R / none x stop distance x
breakeven x entry offset), FTMO's costs, the live rules of the bot that tested it.

  The choice (2025 only): the variant with the most total R on 2025, at least 30 trades and t >= 2 there.
  Passes when on 2026 that variant earns (mean > 0) with at least 20 trades. Shown with how many such passes chance
  alone would give (72 variants x thousands of combos find winners by luck - the bots' whole-bot check decides after).
Results only: the passers go to the bots' agents as cards, nothing is built here.

  python -m agents.rescue_gh plan                      # the job list (bot:symbol:chunk) for the workflow's matrix
  python -m agents.rescue_gh run --job bot1:XAUUSD:0 --minutes 150
  python -m agents.rescue_gh report
"""
from __future__ import annotations

import argparse
import json
import lzma
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "rescue"
BOTS = ["bot5", "bot7", "bot1", "bot2", "bot3", "bot4"]
N_MIN, MEAN_MIN, CHUNK = 60, -0.03, 48


def candidates():
    """{combo: (bot, train n, train mean, unseen n, unseen mean)} - the first bot that tested it."""
    out = {}
    for b in BOTS:
        for f in sorted((ROOT / "reports" / "agents" / b / "exp" / "pack").glob("new_*.json.xz")):
            for _, doc in json.loads(lzma.open(f).read()).items():
                c = doc.get("combo")
                st = doc.get("stats") or {}
                tr, un = st.get("train") or {}, st.get("unseen") or {}
                if not c or c in out or tr.get("mean") is None:
                    continue
                if (tr.get("n") or 0) >= N_MIN and tr["mean"] >= MEAN_MIN \
                        and not (tr["mean"] > 0.05 and (un.get("mean") or -1) > 0.05):
                    out[c] = (b, tr.get("n"), tr["mean"], un.get("n"), un.get("mean"))
    return out


def plan():
    c = candidates()
    groups = {}
    for combo, v in c.items():
        sid, tf, sym = combo.split("|")
        k = (v[0], sym, "1m" if tf == "1Min" else "")
        groups.setdefault(k, []).append(combo)
    jobs = []
    for (b, sym, one), cs in sorted(groups.items()):
        n = 10 if one else CHUNK
        for i in range(0, len(cs), n):
            jobs.append(f"{b}:{sym}:{one or 'x'}{i // n}")
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "rescue_plan.json").write_text(json.dumps({f"{b}:{s}:{o}": v for (b, s, o), v in groups.items()}),
                                                      encoding="utf-8")
    return jobs


def run(job, minutes):
    import agent_portfolio as A
    b, sym, part = job.split(":")
    one = part.startswith("1m")
    i = int(part[2:] if one else part[1:])
    cs = sorted(c for c, v in candidates().items() if v[0] == b and c.split("|")[2] == sym
                and (c.split("|")[1] == "1Min") == one)
    n = 10 if one else CHUNK
    combos = cs[i * n:(i + 1) * n]
    OUT.mkdir(parents=True, exist_ok=True)
    A.BOT_NOW["bot"] = b
    t0 = time.time()
    try:
        t, rows = A.run_rules(combos, "exits", minutes)
    except Exception as e:  # noqa: BLE001
        (OUT / f"{job.replace(':', '_')}.json").write_text(json.dumps({"error": str(e)[-400:], "combos": combos}),
                                                         encoding="utf-8")
        print(job, "failed", str(e)[-300:])
        return
    out = {}
    for c in combos:
        sid, tf, s = c.split("|")
        tc = t[(t["sid"] == sid) & (t["tf"] == tf) & (t["symbol"] == s)]
        vs = {}
        for v, g in tc.groupby("variant") if "variant" in tc else []:
            x = A.variant_stats(g)
            vs[v] = {"train": x["train"], "unseen": x["unseen"], "fading": (x.get("erosion") or {}).get("fading")}
        best = pick(vs)
        doc = {"bot": b, "variants": vs, "best": best}
        if best and "variant" in tc:
            doc["tr"] = A.trades_2026(tc[tc["variant"] == best])
            doc["tr25"] = A.trades_2026(tc[tc["variant"] == best], train=True)
        out[c] = doc
    (OUT / f"{job.replace(':', '_')}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"{job}: {len(combos)} combos in {(time.time() - t0) / 60:.0f} min")


def pick(vs):
    ok = [v for v, x in vs.items() if (x["train"].get("n") or 0) >= 30 and (x["train"].get("t") or 0) >= 2
          and x["train"].get("mean") is not None]
    return max(ok, key=lambda v: vs[v]["train"]["mean"] * (vs[v]["train"].get("eff_n") or vs[v]["train"]["n"]), default=None)


def report():
    docs = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name == "report.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if "error" not in d:
            docs.update(d)
    chosen, passed = [], []
    for c, d in docs.items():
        b = d.get("best")
        if not b:
            continue
        x = d["variants"][b]
        row = {"combo": c, "bot": d["bot"], "variant": b, "train": x["train"], "unseen": x["unseen"], "fading": x.get("fading")}
        chosen.append(row)
        if (x["unseen"].get("n") or 0) >= 20 and (x["unseen"].get("mean") or -1) > 0:
            passed.append(row)
    passed.sort(key=lambda r: -(r["unseen"]["mean"] * r["unseen"]["n"]))
    # chance: a variant chosen on 2025 with no real edge earns on 2026 about half the time
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "tested": len(docs), "chosen": len(chosen),
           "passed": len(passed), "chance_half": round(len(chosen) / 2),
           "passed_strong": [r for r in passed if (r["unseen"].get("t") or 0) >= 2 and not r["fading"]],
           "passed_all": passed[:300]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"tested {len(docs)}, chosen on 2025 {len(chosen)}, earned on 2026 {len(passed)} (chance ~{len(chosen) // 2}), "
          f"strong {len(rep['passed_strong'])}")
    for r in rep["passed_strong"][:40]:
        print(r["bot"], r["combo"], r["variant"], r["train"]["n"], r["train"]["mean"], r["unseen"]["n"], r["unseen"]["mean"], r["unseen"]["t"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--job", default="")
    ap.add_argument("--minutes", type=int, default=150)
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(plan()))
    elif a.mode == "run":
        run(a.job, a.minutes)
    else:
        report()
