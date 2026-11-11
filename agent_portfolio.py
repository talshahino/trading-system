"""The smart agent of bots 1-4 (FTMO / MetaQuotes demo portfolios) and of bot 7 (economic releases, virtual)
(user, 11.11.2026: "לבנות את כולם עכשיו" - every bot gets its own improvement agent, like bot 5's).

Every run, within its time budget, one bot:
 1. experiments  every combo of the bot (strategy|timeframe|instrument), with the bot's live rules (FTMO Standard:
                 closed before market breaks, stocks closed at the session end, holidays, breakeven at 60% of the way
                 to the target), through the bots' own backtest machine (run_rules.py):
                   exits    72 trade-management variants: target (own / 1 / 1.5 / 2 / 3R / none) x stop distance
                            (x0.75 .. x1.5) x entry (market / a limit 0.25R or 0.5R better)      run_rules --optimize
                   filters  market-regime filters: with the 200 trend, ADX trend / range, volatility normal / high / low
                                                                                                   run_rules --filters
                 A combo's results are kept (reports/agents/<bot>/exp/) and re-run only after RERUN_DAYS.
 2. new          strategies the scout agent brought (data/agents/scout/inbox.json) for this bot: each on the bot's
                 instruments and the timeframes the source states, same live rules.
 3. judge        the same fair method as bot 5: chosen on 30.04.2025-31.12.2025 only, shown on 2026 (never seen while
                 choosing), every number time-weighted (agents/agentkit.py - recent trades count more); the other
                 variants of the same combo give the robustness; the baseline is the combo as it trades now
 4. cards        to the manager (reports/agents/<bot>/cards.json), who decides. Nothing changes in the bot here.

Where it runs: the bots' computer (free) for every instrument with Dukascopy history; stocks and Alpaca crypto need
the Alpaca keys, which only GitHub has - those combos run in GitHub's nightly job (--where github).

  python agent_portfolio.py --bot bot1 [--minutes 20] [--where pc|github|all]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from agents import agentkit as K
from agents import ftmo_universe as U
from agents import repo_io as RIO

ROOT = Path(__file__).resolve().parent
TEST_FROM, SPLIT = "2025-04-30", "2026-01-01"
T_SPLIT = pd.Timestamp(SPLIT, tz="UTC")
START = "2024-01-01"
RERUN_DAYS = 14
FAMILIES = ("exits", "filters")
LIVE_RULES = ["--break-exit", "--stock-session-exit", "--news-exit", "--breakeven", "0.6"]
NAMES = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4", "bot7": "בוט 7 · פרסומים כלכליים",
         "bot5": "בוט 5 · סקלפינג", "bot6": "בוט 6 · חוזים עתידיים"}
RISK = {"bot7": 0.5, "bot5": 0.4, "bot6": 0.4}
# bots 5-6 have their own experiment agents (agent_bot5.py / agent_bot6.py); here only the scout's strategies, on
# their instruments: the scalping bot's low-spread list (user, 09.11.2026) and the three index futures
NEW_ONLY = {"bot5": ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "EURJPY", "US100", "US30", "US500", "GER40"],
            "bot6": ["US100", "US500", "US30"]}                         # % per trade where the bot has no json (bot 7: the virtual account)
FILTER_HE = {"with_trend200": "רק עם המגמה (ממוצע 200)", "adx_trend": "רק בשוק במגמה (ADX מעל 20)",
             "adx_range": "רק בשוק בדשדוש (ADX מתחת 20)", "vol_normal": "רק בתנודתיות רגילה",
             "vol_high": "רק בתנודתיות גבוהה", "vol_low": "רק בתנודתיות נמוכה", "none": "בלי סינון"}


def explain_exit(v):
    tg, _, st, en = v.split("|")
    out = []
    if tg != "own":
        out.append("בלי יעד" if tg == "none" else f"יעד {tg}")
    if st != "stop1":
        out.append(f"סטופ x{st[4:]}")
    if en != "entry0":
        out.append(f"כניסה בלימיט {en[5:]}R טוב יותר")
    return " · ".join(out) or "כמו היום"


# ---------- the bot ----------
def bot_combos(bot):
    if bot in NEW_ONLY:
        return []
    if bot == "bot7":
        s = RIO.live_json("live/news_bot_state.json")
        return list(s.get("combos") or [f"{e}|5Min|{n}" for e in ("EV01", "EV02", "EV03") for n in ("US100", "US500", "US30")])
    j = json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
    return list(j["combos"])


def bot_risk(bot):
    if bot in RISK:
        return RISK[bot]
    try:
        return round(float(json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))["risk"]) * 100, 3)
    except Exception:  # noqa: BLE001
        return None


def needs_alpaca(sym):
    return sym not in U.DUKA


def runs_here(sym, where):
    """pc: everything with Dukascopy history (and the rest too if this computer has the Alpaca keys);
    github: the Alpaca instruments, and everything when the bots' computer is not running the agents."""
    if where == "all":
        return True
    if where == "pc":
        return not needs_alpaca(sym) or bool(os.environ.get("ALPACA_API_KEY_ID"))
    return needs_alpaca(sym) or not pc_active()


def pc_active():
    try:
        h = json.loads((ROOT / "reports/agents/host.json").read_text(encoding="utf-8"))
        return (pd.Timestamp.now(tz="UTC") - pd.Timestamp(h.get("smart_at") or "1970-01-01T00:00:00Z")) < pd.Timedelta(hours=8)
    except Exception:  # noqa: BLE001
        return False


# ---------- experiments ----------
def key_of(combo):
    return hashlib.sha1(combo.encode()).hexdigest()[:10]


class Store:
    def __init__(self, bot):
        self.dir = ROOT / "reports" / "agents" / bot / "exp"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, fam, combo):
        return self.dir / f"{fam}_{combo.replace('|', '_').replace('/', '-')}.json"

    def get(self, fam, combo):
        p = self.path(fam, combo)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, fam, combo, doc):
        self.path(fam, combo).write_text(json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8")

    def age_days(self, fam, combo):
        d = self.get(fam, combo)
        return 999 if not d else (pd.Timestamp.now(tz="UTC") - pd.Timestamp(d["at"])).days


def _wst(r, t):
    return K.wstats(r, K.weights(t)) if len(r) else K.wstats([])


def variant_stats(t):
    t = t.copy()
    t["time"] = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    tr, un = t[t["time"] < T_SPLIT], t[t["time"] >= T_SPLIT]
    return {"train": _wst(tr["r"].to_numpy(float), tr["time"]), "unseen": _wst(un["r"].to_numpy(float), un["time"]),
            "whole": _wst(t["r"].to_numpy(float), t["time"]), "halves": K.halves(t["time"], t["r"]),
            "erosion": K.erosion(t["time"], t["r"]) if len(t) >= 20 else None}


def run_rules(combos, fam, budget_min, extra=None):
    """One run of the backtest machine for a few combos of one instrument; returns its trades (all variants)."""
    out_dir = ROOT / "output"
    sym = combos[0].split("|")[2]
    tag = sym.replace(",", "_").replace("/", "-")[:60]
    for f in (out_dir / f"combo_trades_{tag}.csv.gz", out_dir / f"optimize_{tag}.json", out_dir / f"filters_{tag}.json"):
        f.unlink(missing_ok=True)
    cf = out_dir / f"agent_combos_{tag}.txt"
    out_dir.mkdir(exist_ok=True)
    cf.write_text("\n".join(combos), encoding="utf-8")
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [sys.executable, "-u", "run_rules.py", "--market", "ftmo", "--start", START, "--end", end, "--test-from", TEST_FROM,
           "--combos-file", str(cf), "--symbols", sym] + LIVE_RULES + (extra or [])
    if fam == "exits":
        cmd += ["--optimize", SPLIT]
    elif fam == "filters":
        cmd += ["--filters", SPLIT]
    env = dict(os.environ, MAX_RUN_MINUTES=str(max(5, int(budget_min))))
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=int(budget_min * 60 + 900))
    tail = "\n".join((r.stdout or "").strip().splitlines()[-4:] + (r.stderr or "").strip().splitlines()[-3:])
    f = out_dir / f"combo_trades_{tag}.csv.gz"
    if not f.exists():
        raise RuntimeError(f"run_rules {sym}: no trades file ({r.returncode}) {tail[-400:]}")
    t = pd.read_csv(f)
    if t.empty and "NO DATA" in (r.stdout or ""):
        nd = [x for x in (r.stdout or "").splitlines() if "NO DATA" in x]
        raise RuntimeError(f"no market data: {nd[0].strip()[:200] if nd else sym}")
    rows = {}
    jf = out_dir / f"{'optimize' if fam == 'exits' else 'filters'}_{tag}.json"
    if fam in FAMILIES and jf.exists():
        rows = {x["combo"]: x for x in json.loads(jf.read_text(encoding="utf-8"))}
    return t, rows


def experiments(bot, budget_s, where):
    st = Store(bot)
    todo = []
    for combo in bot_combos(bot):
        sym = combo.split("|")[2]
        if not runs_here(sym, where):
            continue
        for fam in FAMILIES:
            age = st.age_days(fam, combo)
            if age >= RERUN_DAYS:
                todo.append((0 if age == 999 else 1, fam, -age, sym, combo))
    todo.sort(key=lambda x: (x[0], x[3], x[1], x[2]))          # never-run first, one instrument's combos together
    t0, done = time.time(), 0
    groups = {}
    for x in todo:
        groups.setdefault((x[3], x[1]), []).append(x[4])
    for (sym, fam), combos in groups.items():
        left = budget_s - (time.time() - t0)
        if left < 120:
            break
        ts = time.time()
        try:
            t, rows = run_rules(combos, fam, left / 60)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym} {fam}: failed - {str(e)[-300:]}")
            for c in combos:                                          # a combo that cannot run is retried tomorrow
                st.put(fam, c, {"at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RERUN_DAYS - 1)).isoformat(),
                                "combo": c, "error": str(e)[-300:]})
            continue
        for c in combos:
            sid, tf, s = c.split("|")
            tc = t[(t["sid"] == sid) & (t["tf"] == tf) & (t["symbol"] == s)]
            doc = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "combo": c, "family": fam,
                   "secs": round((time.time() - ts) / max(1, len(combos)), 1), "variants": {}}
            for v, g in tc.groupby("variant") if "variant" in tc else []:
                doc["variants"][v] = variant_stats(g)
            # the machine's own unweighted stats of every variant (the filters keep only two variants' trades)
            doc["grid"] = [{"variant": x["variant"], "train": x["train"], "valid": x["valid"]}
                           for x in (rows.get(c) or {}).get("all", [])]
            st.put(fam, c, doc)
            done += 1
        print(f"  {sym} {fam}: {len(combos)} combos ({time.time() - ts:.0f}s)")
    left = len(todo) - done
    print(f"experiments: {done} run, {left} left for the next runs")
    return done, left


# ---------- new strategies from the scout ----------
def inbox():
    try:
        return json.loads((ROOT / "data/agents/scout/inbox.json").read_text(encoding="utf-8")).get("strategies", [])
    except Exception:  # noqa: BLE001
        return []


def bot_symbols(bot):
    return NEW_ONLY.get(bot) or sorted({c.split("|")[2] for c in bot_combos(bot)})


def test_new(bot, budget_s, where, symbols=None, timeframes=None):
    """Every scout strategy addressed to this bot, on the bot's instruments, once (re-run after RERUN_DAYS)."""
    st = Store(bot)
    t0, done = time.time(), 0
    syms = symbols or bot_symbols(bot)
    for s_ in inbox():
        if bot not in (s_.get("bots") or []) or not s_.get("id"):
            continue
        sid = s_["id"]
        tfs = [x for x in (s_.get("timeframes") or []) if x in ("1Min", "5Min", "15Min", "1Hour", "4Hour")] or (timeframes or ["1Hour"])
        if bot in NEW_ONLY:
            tfs = [x for x in tfs if x in ("1Min", "5Min", "15Min")]          # scalping bots: short timeframes only
        mine = [x for x in syms if runs_here(x, where)]
        if s_.get("markets"):
            mine = [x for x in mine if U.CLASS.get(x) in s_["markets"]] or mine
        for sym in mine:
            combos = [f"{sid}|{tf}|{sym}" for tf in tfs]
            if all(st.age_days("new", c) < RERUN_DAYS for c in combos):
                continue
            left = budget_s - (time.time() - t0)
            if left < 120:
                print(f"new strategies: stopped for time, {done} tested")
                return done
            try:
                t, _ = run_rules(combos, "new", left / 60)
            except Exception as e:  # noqa: BLE001
                for c in combos:
                    st.put("new", c, {"at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RERUN_DAYS - 1)).isoformat(),
                                      "combo": c, "error": str(e)[-300:]})
                continue
            for c in combos:
                sid_, tf, s = c.split("|")
                tc = t[(t["sid"] == sid_) & (t["tf"] == tf) & (t["symbol"] == s)]
                st.put("new", c, {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "combo": c, "family": "new",
                                  "scout": {k: s_.get(k) for k in ("name", "source", "url", "added")},
                                  "stats": variant_stats(tc) if len(tc) else None})
                done += 1
    print(f"new strategies: {done} combos tested")
    return done


# ---------- cards ----------
def cards_for(bot):
    st = Store(bot)
    risk = bot_risk(bot)
    cards, tested = [], 0
    for combo in bot_combos(bot):
        for fam in FAMILIES:
            d = st.get(fam, combo)
            if not d or not d.get("variants"):
                continue
            base_key = "own|be0.6|stop1|entry0" if fam == "exits" else "none"
            vs = d["variants"]
            base = vs.get(base_key)
            if not base:
                continue
            tested += len(d.get("grid") or vs)
            if base.get("erosion") and base["erosion"].get("fading") and fam == "exits":
                e = base["erosion"]
                cards.append(K.card(bot, "info", f"{combo}: היתרון נשחק",
                                    f"בחצי השנה האחרונה {e['recent_mean']:+.2f}R לעסקה, לעומת {e['before_mean']:+.2f}R בשנה וחצי שלפני",
                                    base["train"], base["unseen"], whole=base["whole"], halves_=base["halves"],
                                    detail={"erosion": e}, change={"combo": combo, "kind": "review"}))
            cand = [(v, x) for v, x in vs.items() if v != base_key and (x["train"].get("eff_n") or 0) >= 10]
            if not cand:
                continue
            v, best = max(cand, key=lambda vx: vx[1]["train"]["mean"] if vx[1]["train"]["mean"] is not None else -9)
            if (best["train"]["mean"] or -9) <= (base["train"]["mean"] or 0):
                continue
            grid = d.get("grid") or []
            bval = next((g["valid"]["avg"] for g in grid if g["variant"] == base_key), None)
            others = [g for g in grid if g["variant"] != base_key and g["valid"]["n"] >= 5]
            share = (round(sum(1 for g in others if g["valid"]["avg"] > (bval or 0)) / len(others), 2) if others and bval is not None else None)
            what = explain_exit(v) if fam == "exits" else FILTER_HE.get(v, v)
            cards.append(K.card(
                bot, "improve", f"{combo}: {what}",
                f"{'ניהול העסקה' if fam == 'exits' else 'סינון מצב שוק'} לאסטרטגיה {combo}: {what}. "
                f"נבחר על 30.04-31.12.2025 בלבד, ומוצג על 2026 שלא נראתה, עם החוקים של הבוט.",
                best["train"], best["unseen"], whole=best["whole"], halves_=best["halves"],
                neighbours={"values": len(grid) or len(vs), "share_positive": share},
                tested=len(grid) or len(vs), risk_pct=risk, baseline=base["unseen"],
                change={"combo": combo, "family": fam, "from": base_key, "to": v}))
    # the scout's strategies
    for f in sorted(st.dir.glob("new_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        s = d.get("stats")
        if not s:
            continue
        tested += 1
        sc = d.get("scout") or {}
        cards.append(K.card(
            bot, "new", f"אסטרטגיה חדשה: {d['combo']}" + (f" · {sc.get('name')}" if sc.get("name") else ""),
            f"הסייר הביא אותה ({sc.get('source') or '?'}). נבדקה על הכלים של הבוט עם החוקים שלו: 30.04-31.12.2025, ומוצגת על 2026.",
            s["train"], s["unseen"], whole=s["whole"], halves_=s["halves"], neighbours={"share_positive": None},
            tested=1, risk_pct=risk, baseline=None,
            detail={"scout": sc}, change={"combo": d["combo"], "kind": "add"}))
    for c in cards:
        c["tier"], c["verdict"] = K.judge(c)
    return cards, tested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True, choices=sorted(NAMES))
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--where", default="pc", choices=["pc", "github", "all"])
    ap.add_argument("--no-cards", action="store_true")
    a = ap.parse_args()
    budget = a.minutes * 60
    t0 = time.time()
    done, left = experiments(a.bot, budget * 0.8, a.where)
    new = test_new(a.bot, budget - (time.time() - t0), a.where)
    if a.no_cards:
        return
    cards, tested = cards_for(a.bot)
    if a.bot in NEW_ONLY:                                        # its own cards.json belongs to agent_bot5 / agent_bot6
        d = K.CARDS_DIR / a.bot
        d.mkdir(parents=True, exist_ok=True)
        (d / "cards_new.json").write_text(json.dumps({"bot": a.bot, "updated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                                                      "cards": cards}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(json.dumps({"bot": a.bot, "new_cards": len(cards)}, ensure_ascii=False))
        return
    combos = bot_combos(a.bot)
    st = Store(a.bot)
    have = sum(1 for c in combos for fam in FAMILIES if (st.get(fam, c) or {}).get("variants"))
    K.write_cards(a.bot, cards, {"tested": tested, "done": have, "planned": len(combos) * len(FAMILIES),
                                 "run": {"experiments": done, "left": left, "new": new, "where": a.where}})
    print(json.dumps({"bot": a.bot, "cards": len(cards), "tiers": pd.Series([c["tier"] for c in cards]).value_counts().to_dict()
                      if cards else {}, "done": have, "planned": len(combos) * len(FAMILIES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
