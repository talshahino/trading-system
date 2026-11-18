"""Bot 5's gap method - Tal's open questions, on the rules that run NOW (live/bots/bot5.json tal_tracks, since 18.11.2026
the stop beyond the impulse's start + room). The queue on the bots' computer restarts whenever a track's rules change,
so these come here, on all the tracks at once:
  size    b5mingap1811    "too small, that does not count as a gap": MIN_GAP 0.05-0.3 x the average range, GAPLEG 1-5% of the impulse
  gapx    tal1711gapclose only a wick through the whole gap closes it
  firstg  tal1711review   only the first gap of each impulse
  follow  b5follow1611    no follow-through after N candles of the entry = out
Every family: the value chosen on the period to 2025-12-31 (time-weighted total), shown on 2026 (unseen). FTMO's costs and
break rule, as the live bot. Results only - nothing in the bot changes here.

  python -m agents.tal_asks_study plan
  python -m agents.tal_asks_study run --sym US100
  python -m agents.tal_asks_study report
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from agents import agentkit as K
from agents import tal_gap as TG

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "agents" / "bot5" / "tal_asks"
PARTS = OUT / "parts"
BUDGET_MIN = 150
SKIP = ("EURUSD", "USDJPY", "XAUUSD")     # Tal, 18.11: these three are followed live for a week or two first (tal1811fx)
FAMS = {"size": {"mg05": {"MIN_GAP": 0.05}, "mg10": {"MIN_GAP": 0.1}, "mg15": {"MIN_GAP": 0.15}, "mg30": {"MIN_GAP": 0.3},
                 "gl01": {"GAPLEG": 0.01}, "gl02": {"GAPLEG": 0.02}, "gl03": {"GAPLEG": 0.03}, "gl05": {"GAPLEG": 0.05}},
        "gapx": {"full": {"GAPX": "full"}},
        "firstg": {"fg1": {"FIRSTG": "1"}, "fglo": {"FIRSTG": "lo"}},
        "follow": {"x1": {"FOLLOW": "x1"}, "x2": {"FOLLOW": "x2"}, "x3": {"FOLLOW": "x3"}}}


def tracks():
    s5 = json.loads((ROOT / "live" / "bots" / "bot5.json").read_text(encoding="utf-8"))
    return {k: v for k, v in (s5.get("tal_tracks") or {}).items() if not any(k.endswith("@" + x) for x in SKIP)}


def syms_of(track):
    import agent_bot5 as A
    return [track.split("@")[1]] if "@" in track else list(A.TAL_SYMBOLS)


def group_of(track):
    return track.split("@")[0] + ("@" if "@" in track else "")


def ways_of(cfg):
    """{way: cfg}: "now" and every value that really changes the rules (a size floor only above the track's own)."""
    out = {"now": cfg}
    for vals in FAMS.values():
        for w, x in vals.items():
            if "MIN_GAP" in x and x["MIN_GAP"] <= float(cfg.get("MIN_GAP") or 0):
                continue
            out[w] = TG._norm(dict(cfg, **x))
    return out


def plan():
    return sorted({s for t in tracks() for s in syms_of(t)})


def run(sym):
    import agent_bot5 as A
    PARTS.mkdir(parents=True, exist_ok=True)
    D = A.tal_load(sym)
    if D is None:
        raise SystemExit(f"no data for {sym}")
    out, t_end = {}, time.time() + BUDGET_MIN * 60
    for tk, tv in tracks().items():
        if sym not in syms_of(tk):
            continue
        pair = next(p for p in TG.PAIRS if p[0] == tv["pair"][0] and p[1] == tv["pair"][1])
        gc = {}
        for w, c in ways_of(TG.ftmo_cfg(tv["cfg"])).items():
            if time.time() > t_end:
                out.setdefault(tk, {})[w] = {"error": "time budget"}
                continue
            try:
                t = TG.simulate(D, pair, c, gc)
            except Exception as e:  # noqa: BLE001
                out.setdefault(tk, {})[w] = {"error": f"{type(e).__name__}: {e}"}
                continue
            out.setdefault(tk, {})[w] = [[str(a), round(float(b), 4), round(float(x), 4)]
                                         for a, b, x in zip(t["time"], t["r"], t["cost_r"])]
        print(sym, tk, {w: len(x) if isinstance(x, list) else x for w, x in out[tk].items()}, flush=True)
        (PARTS / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")
    (PARTS / f"{sym}.json").write_text(json.dumps(out), encoding="utf-8")


def _view(d):
    """One period of one way: the trades, the time-weighted total, the deepest fall, the worst day, the green days."""
    if not len(d):
        return {"n": 0, "total": 0.0, "wtotal": 0.0, "mean": None, "win": None, "max_fall": 0.0, "worst_day": 0.0, "green": None}
    d = d.sort_values("time")
    cum = d["r"].cumsum()
    day = d.groupby(d["time"].dt.strftime("%Y-%m-%d"))["r"].sum()
    return {"n": int(len(d)), "total": round(float(d["r"].sum()), 2),
            "wtotal": round(float((d["r"] * K.weights(d["time"])).sum()), 2), "mean": round(float(d["r"].mean()), 4),
            "win": round(float((d["r"] > 0).mean()) * 100, 1), "max_fall": round(float((cum.cummax().clip(lower=0) - cum).max()), 2),
            "worst_day": round(float(day.min()), 2), "green": round(float((day > 0).mean()) * 100, 1)}


def _stats(d):
    tr, un = d[d["time"] < TG.SPLIT], d[d["time"] >= TG.SPLIT]
    return {"train": _view(tr), "unseen": _view(un), "k_train": K.wstats(tr["r"], K.weights(tr["time"])) if len(tr) else K.wstats([]),
            "k_unseen": K.wstats(un["r"], K.weights(un["time"])) if len(un) else K.wstats([]),
            "k_whole": K.wstats(d["r"], K.weights(d["time"])) if len(d) else K.wstats([]),
            "halves": K.halves(d["time"], d["r"]) if len(d) >= 4 else None,
            "cost_r": round(float(d["cost"].mean()), 3) if len(d) else None}


def report():
    rows, errs = {}, []
    for f in sorted(PARTS.glob("*.json")):
        for tk, wz in json.loads(f.read_text(encoding="utf-8")).items():
            for w, x in wz.items():
                if isinstance(x, dict):
                    errs.append([f.stem, tk, w, x.get("error")])
                    continue
                rows.setdefault(group_of(tk), {}).setdefault(w, []).extend(x)
    rep = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "split": str(TG.SPLIT.date()), "errors": errs[:40], "groups": {}}
    for g, wz in rows.items():
        st = {}
        for w, x in wz.items():
            d = pd.DataFrame(x, columns=["time", "r", "cost"])
            d["time"] = pd.to_datetime(d["time"], utc=True, format="ISO8601")
            st[w] = _stats(d)
        now = st.get("now")
        fams = {}
        for fam, vals in FAMS.items():
            have = [w for w in vals if w in st]
            best = max(have, key=lambda w: st[w]["train"]["wtotal"], default=None)
            if best is None or now is None or st[best]["train"]["wtotal"] <= now["train"]["wtotal"]:
                fams[fam] = {"tested": len(have), "chosen": None, "passes": False}
                continue
            u, nu = st[best]["unseen"], now["unseen"]
            fams[fam] = {"tested": len(have), "chosen": best, "passes": bool(u["total"] > nu["total"] and u["worst_day"] >= nu["worst_day"]
                                                                            and u["max_fall"] <= nu["max_fall"] * 1.1)}
        rep["groups"][g] = {"ways": st, "families": fams}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for g, v in rep["groups"].items():
        n = v["ways"].get("now")
        print(g, "now", n and (n["train"]["n"], n["train"]["total"], n["unseen"]["n"], n["unseen"]["total"]), v["families"])
    print("errors:", len(errs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "report"])
    ap.add_argument("--sym", default="")
    a = ap.parse_args()
    if a.mode == "plan":
        print(json.dumps(plan()))
    elif a.mode == "run":
        run(a.sym)
    else:
        report()
