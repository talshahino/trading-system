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
BOT_NOW = {}
FAMILIES = ("exits", "filters", "manage", "fib", "ladder", "pyr")   # manage (12.11.2026): breakeven point x partial exit per strategy;
# fib (12.11.2026, user's standing task): the same at the Fibonacci levels of the way to the target - built as "manage"
# ladder (13.11.2026, user's standing task): the entry split into 2-4 limit orders with the same total risk, stop and
# target (run_rules --manage-grid ladder); the live bots need runner code for it, built when a card is approved
# pyr (14.11.2026, user's task tal1411anti): add to a winning trade (half a position at +1R, ...) with the whole stop
# moved so a stop still loses exactly 1R (run_rules --manage-grid pyr); bots 1-4 only (PYR_BOTS), runner code on approval
# FTMO's real stock spreads: ~2.4x the history (measured on the demo 10.11.2026, run_rules --stock-spread-mult). Until
# 13.11.2026 the experiments ran without it, so every stock combo's exits / filters / manage / fib / new results were
# measured with too-cheap costs (the 10.11 stress test: SS01 WMT +15.6R -> -4.0R) - those results are run again (COSTS).
# And FTMO's restricted events minute by minute (13.11.2026, run_rules --news-minute): the live bots close a position 3
# minutes before an event on its instrument; the experiments held it through the release (bot 2 QB11 USDJPY 9.11:
# +7.7R in the backtest, +2.8R on the demo) - results of the instruments the events touch are run again too.
STOCK_SPREAD_MULT = 2.4
COSTS = f"stock-x{STOCK_SPREAD_MULT}+news-minute"
# no new stock trade in the last 15 minutes of the session, as the live bots (13.11.2026, run_rules.stock_late_mask);
# results stored before carry no stamp - the whole-bot check drops those entries when it reads them (portfolio_check)
STOCK_ENTRY = "live15"
LIVE_RULES = ["--break-exit", "--stock-session-exit", "--news-exit", "--news-minute", "--breakeven", "0.6",
              "--stock-spread-mult", str(STOCK_SPREAD_MULT)]
_NEWS_HIT = {}


def news_touched(sym):
    """Does any FTMO restricted event touch this instrument (fx pairs of the currency, US indices, oil, gold, DXY)?"""
    if sym not in _NEWS_HIT:
        try:
            from agents import ftmo_calendar as CAL
            if "ins" not in _NEWS_HIT:
                h = json.loads((ROOT / "data" / "ftmo_calendar" / "history.json").read_text(encoding="utf-8"))
                _NEWS_HIT["ins"] = {ins for _, ins in CAL.parse(h)[0]}
            _NEWS_HIT[sym] = any(CAL.restricted_affects(sym, i) for i in _NEWS_HIT["ins"])
        except Exception:  # noqa: BLE001
            _NEWS_HIT[sym] = False
    return _NEWS_HIT[sym]
NAMES = {"bot1": "בוט 1", "bot2": "בוט 2", "bot3": "בוט 3", "bot4": "בוט 4", "bot7": "בוט 7 · פרסומים כלכליים",
         "bot5": "בוט 5 - סקאלפינג", "bot6": "בוט 6 · חוזים עתידיים"}
RISK = {"bot7": 0.5, "bot5": 0.4, "bot6": 0.4}
# bots 5-6 have their own experiment agents (agent_bot5.py / agent_bot6.py); here only the scout's strategies, on
# their instruments: the scalping bot's low-spread list (user, 09.11.2026) and the three index futures
NEW_ONLY = {"bot5": ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "EURJPY", "US100", "US30", "US500", "GER40"],
            "bot6": ["US100", "US500", "US30"]}                         # % per trade where the bot has no json (bot 7: the virtual account)
FILTER_HE = {"with_trend200": "רק עם המגמה (ממוצע 200)", "adx_trend": "רק בשוק במגמה (ADX מעל 20)",
             "adx_range": "רק בשוק בדשדוש (ADX מתחת 20)", "vol_normal": "רק בתנודתיות רגילה",
             "vol_high": "רק בתנודתיות גבוהה", "vol_low": "רק בתנודתיות נמוכה", "none": "בלי סינון",
             "ema9_20": "רק בכיוון EMA 9/20", "bbw_p20": "לא ברוחב בולינגר נמוך (מתחת לאחוזון 20)"}
# bots whose filter tests also try the user's EMA 9/20 and Bollinger-width filters (task tal1411ema, 14.11.2026):
# live/news_bot.regime knows them for bot 7, live/runner._regime for bots 1-4 (since 14.11.2026)
FILTERS_EXTRA = {"bot1", "bot2", "bot3", "bot4", "bot7"}


def explain_manage(v):
    """be0.6|part0.3@0.5 -> Hebrew; ber2.5 = no-target strategies (R in profit)."""
    b, pa = v.split("|")
    tgt = b.startswith("be") and not b.startswith("ber")
    x = float(b[2:] if tgt else b[3:])
    out = ["בלי נקודת איזון" if x == 0 else (f"איזון ב־{x * 100:g}% מהדרך ליעד" if tgt else f"איזון ב־{x:g}R רווח")]
    if pa != "part0":
        fr, w = pa[4:].split("@")
        out.append(f"{float(fr) * 100:g}% יוצאים " + (f"ב־{float(w) * 100:g}% מהדרך" if tgt else f"ב־{float(w):g}R") + " והשאר לאיזון")
    return " · ".join(out)


def explain_ladder(v):
    """lad0,0.25,0.5 -> Hebrew."""
    lv = [float(x) for x in v[3:].split(",")] if v != "lad0" else [0.0]
    if lv == [0.0]:
        return "כניסה אחת בפתיחה"
    parts = ["בפתיחה" if x == 0 else f"בנסיגה של {x:g}R" for x in lv]
    return f"כניסה מדורגת ב־{len(lv)} חלקים שווים ({', '.join(parts)}), אותו סיכון כולל, סטופ ויעד; חלק שלא התמלא תוך 3 נרות מתבטל"


PYR_BOTS = {"bot1", "bot2", "bot3", "bot4"}


def explain_pyr(v):
    """pyr1@0.5 -> Hebrew."""
    if v == "pyr0":
        return "בלי הוספה לעסקה"
    at, fr = v[3:].split("@")
    x = (float(fr) * float(at) - 1) / (1 + float(fr))
    return (f"ב־{float(at):g}R רווח מוסיפים {float(fr):g} פוזיציה, והסטופ של הכול זז ל־{x:+.2f}R כך שסטופ מפסיד בדיוק 1R "
            f"(בלי הגדלה אחרי הפסד)")


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
        cs = list(s.get("combos") or [f"{e}|5Min|{n}" for e in ("EV01", "EV02", "EV03") for n in ("US100", "US500", "US30")])
        # a removed combo (card kind "remove": plans "off_from", e.g. EV01 from 14.11.2026, card d51b540869aa) trades no
        # more - no experiments on it (13.11.2026 night: EV01's 16 instruments would take the ladder / Fibonacci queue)
        try:
            pl = json.loads((ROOT / "live" / "bots" / "bot7.json").read_text(encoding="utf-8")).get("plans") or {}
        except Exception:  # noqa: BLE001
            pl = {}
        return [c for c in cs if not (pl.get(c) or {}).get("off_from")]
    j = json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8"))
    return list(j["combos"])


def base_variant(bot, combo, fam, doc=None):
    """The variant the combo trades now (manage: the bots' default when no card changed it)."""
    fam = "manage" if fam == "fib" else fam
    v = built_variant(bot, combo, fam)
    if fam in ("ladder", "pyr"):
        return v
    if v is None and fam == "manage":
        ht = (doc or {}).get("has_target")
        if ht is None and doc:
            ht = any(k.startswith("be") and not k.startswith("ber") for k in (doc.get("variants") or {}))
        v = "be0.6|part0" if ht else "ber2.5|part0"
    return v


def built_variant(bot, combo, fam):
    """The variant a combo trades now: an approved card built into live/bots/<bot>.json "plans" (12.11.2026),
    else the strategy's own (exits) / no filter. New cards are measured against this one."""
    try:
        p = (json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8")).get("plans") or {}).get(combo) or {}
    except Exception:  # noqa: BLE001
        p = {}
    if fam == "filters":
        return p.get("filter") or "none"
    if fam == "ladder":
        return "lad" + ",".join(f"{float(x):g}" for x in p["ladder"]) if p.get("ladder") else "lad0"
    if fam == "pyr":
        return "pyr0"                                   # no bot adds to a trade yet (runner code on approval)
    if fam in ("manage", "fib"):
        nt = (p.get("target") == "none")
        if p.get("be_way") is None and p.get("be_r") is None and not p.get("partial"):
            return None                                 # the default: with a target be0.6, without ber2.5 (known after the run)
        be = p.get("be_r", 2.5) if nt or p.get("be_r") is not None else p.get("be_way", 0.6)
        pa = p.get("partial")
        b = f"ber{float(be):g}" if (nt or p.get("be_r") is not None) else f"be{float(be):g}"
        return f"{b}|part{'0' if not pa else f'{float(pa[0]):g}@{float(pa[1]):g}'}"
    if not any(k in p for k in ("target", "stop", "limit")):
        return "own|be0.6|stop1|entry0"
    tg = p.get("target")
    tg = "own" if tg is None else ("none" if tg == "none" else f"{float(tg):g}R")
    return f"{tg}|be0.6|stop{float(p.get('stop', 1)):g}|entry{float(p.get('limit', 0)):g}"


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
    return (needs_alpaca(sym) and not pc_active(alpaca=True)) or not pc_active()


def pc_active(alpaca=False):
    """The bots' computer runs the agents (and, with alpaca=True, also has the Alpaca keys)."""
    try:
        h = json.loads((ROOT / "reports/agents/host.json").read_text(encoding="utf-8"))
        on = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(h.get("smart_at") or "1970-01-01T00:00:00Z")) < pd.Timedelta(hours=8)
        return on and (h.get("alpaca") is True if alpaca else True)
    except Exception:  # noqa: BLE001
        return False


# ---------- experiments ----------
def key_of(combo):
    return hashlib.sha1(combo.encode()).hexdigest()[:10]


# instruments whose history source changed: results made before are of another market and are run again
# (12.11.2026: FTMO's ASML and RACE are the US listings, agents/ftmo_universe.py)
SOURCE_SINCE = {"ASML": "2026-11-12T16:00:00+00:00", "RACE": "2026-11-12T16:00:00+00:00"}   # after the fix is installed on the bots' computer


def stale_source(doc):
    if not doc:
        return False
    sym = str(doc.get("combo") or "").split("|")[-1]
    if sym in SOURCE_SINCE and str(doc.get("at") or "") < SOURCE_SINCE[sym]:
        return True
    # measured before FTMO's real stock spreads / the restricted events were part of the experiments (13.11.2026)
    return doc.get("costs") != COSTS and (U.CLASS.get(sym) == "stock" or news_touched(sym))


class Store:
    def __init__(self, bot):
        self.dir = ROOT / "reports" / "agents" / bot / "exp"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, fam, combo):
        return self.dir / f"{fam}_{combo.replace('|', '_').replace('/', '-')}.json"

    def get(self, fam, combo):
        p = self.path(fam, combo)
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return None if stale_source(d) else d

    def put(self, fam, combo, doc):
        doc = {**doc, "costs": COSTS, "stock_entry": STOCK_ENTRY}
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
            "erosion": K.erosion(t["time"], t["r"]) if len(t) >= 20 else None,
            # how the trades ended (target / stop / breakeven / exit signal / session...) per period (12.11.2026, user's
            # question on QB11's 8R target: how often is it really reached)
            "exit_mix": {"train": _mix(tr), "unseen": _mix(un)}}


def _mix(t):
    if t is None or t.empty or "reason" not in t:
        return {}
    rs = t["reason"].fillna("?").astype(str)
    be = rs.str.contains("סטופ") & (t["r"].abs() < 0.15)              # a stop at the entry = breakeven
    rs = rs.where(~be, "איזון")
    return {k: int(v) for k, v in rs.value_counts().items()}


def tr_ok(d):
    """The stored 2026 trades are there and in real minutes (before 12.11.2026 noon they were stored 1000x too coarse:
    pandas 3 keeps times in microseconds, and those files must be run again)."""
    tr = d.get("tr")
    if not isinstance(tr, dict):
        return False
    rows = [x for v in tr.values() for x in (v or [])[:1]]
    return all(x[0] > 20_000_000 for x in rows)          # 2008 onwards in minutes since 1970


def trades_2026(t, train=False):
    """Compact 2026 trades: [entry minute, exit minute, direction, R] (minutes since 1970, UTC).
    train=True: the choosing period's trades instead (30.04.2025-31.12.2025), so that whole-bot rules (e.g. a trade
    against an open position on the same instrument, agents/portfolio_check.side_conflicts) are chosen on 2025 and
    only shown on 2026, like every other proposal (13.11.2026)."""
    if t is None or t.empty:
        return []
    tt = pd.to_datetime(t["time"], utc=True, format="ISO8601")
    t = t[(tt >= pd.Timestamp(TEST_FROM, tz="UTC")) & (tt < T_SPLIT)] if train else t[tt >= T_SPLIT]
    # minutes from the time itself, not from the storage unit: pandas 3 keeps these times in microseconds, so the old
    # "int64 // 60e9" gave 1000-minute steps and the whole-bot replay put all of 2026 into ~20 days of 1970 (12.11.2026)
    ep, mn = pd.Timestamp("1970-01-01", tz="UTC"), pd.Timedelta(minutes=1)
    a = (pd.to_datetime(t["time"], utc=True, format="ISO8601") - ep) // mn
    b = (pd.to_datetime(t["exit_time"].fillna(t["time"]), utc=True, format="ISO8601") - ep) // mn
    return [[int(x), int(y), int(d), round(float(r), 3)] for x, y, d, r in zip(a, b, t["dir"].fillna(0), t["r"])]


def run_rules(combos, fam, budget_min, extra=None):
    """One run of the backtest machine for a few combos of one instrument; returns its trades (all variants)."""
    out_dir = ROOT / "output"
    sym = combos[0].split("|")[2]
    tag = sym.replace(",", "_").replace("/", "-")[:60]
    for f in (out_dir / f"combo_trades_{tag}.csv.gz", out_dir / f"optimize_{tag}.json", out_dir / f"filters_{tag}.json",
              out_dir / f"manage_{tag}.json"):
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
        if BOT_NOW.get("bot") in FILTERS_EXTRA:
            cmd += ["--filters-extra"]               # the user's EMA 9/20 + Bollinger-width filters (14.11.2026)
    elif fam in ("manage", "fib", "ladder", "pyr"):
        cmd += ["--manage", SPLIT] + (["--manage-grid", fam] if fam in ("fib", "ladder", "pyr") else [])
    if fam in ("exits", "filters", "manage", "fib", "ladder", "pyr"):
        bf = plans_file(BOT_NOW.get("bot"), out_dir)
        if bf is not None:
            cmd += ["--plans-file", str(bf)]          # on top of what the bot trades now (exits / filters too: 13.11.2026;
            #                                           the morning review's live rules too: 14.11.2026)
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
    jf = out_dir / f"{ {'exits': 'optimize', 'filters': 'filters', 'manage': 'manage', 'fib': 'manage', 'ladder': 'manage', 'pyr': 'manage'}.get(fam, 'x') }_{tag}.json"
    if fam in FAMILIES and jf.exists():
        rows = {x["combo"]: x for x in json.loads(jf.read_text(encoding="utf-8"))}
    return t, rows


def legacy_tr(d):
    """Trades stored before the minutes fix (12.11.2026): their times are in 1000-minute steps (2026 ~ 29,500)."""
    return any(tr and tr[0][0] < 10_000_000 for tr in (d.get("tr") or {}).values())


def fixing(bot):
    """The bot's strategies its agent must try to fix first (data/agents/fix_attempts.json, status trying): run first."""
    try:
        fa = json.loads((ROOT / "data/agents/fix_attempts.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    return {k.split("|", 1)[1] for k, v in fa.items() if k.startswith(bot + "|") and (v or {}).get("status") == "trying"}


def loss_priority(bot):
    """The tests the daily loss review asks for first (data/agents/priority.json, agents/loss_review.py)."""
    try:
        return json.loads((ROOT / "data" / "agents" / "priority.json").read_text(encoding="utf-8")).get(bot) or {}
    except Exception:  # noqa: BLE001
        return {}


def plan_now(bot, combo):
    """The combo's built plan in live/bots/<bot>.json (exits / filter / management), without the card ids, and the
    morning review's live rules on it ("rules": target / stop factor, hours, direction - live/runner.combo_rules): the
    experiments run on top of both (run_rules.py --plans-file), so a result measured on another plan is stale."""
    try:
        p = (json.loads((ROOT / "live" / "bots" / f"{bot}.json").read_text(encoding="utf-8")).get("plans") or {}).get(combo) or {}
    except Exception:  # noqa: BLE001
        p = {}
    out = {k: v for k, v in sorted(p.items()) if k != "card" and not k.endswith("_card")}
    r = live_rules(bot).get(combo)
    if r:
        out["rules"] = r
    return out


_LIVE_RULES = {}
RULE_KINDS = ("tp", "sl", "hours", "dir")


def live_rules(bot):
    """{combo: {tp, sl, hours, dir}} - the morning review's rules the live bot applies on top of its plans (branch live,
    live/manager_state.json "risk", the same source as the risk multipliers). 14.11.2026: the experiments ran without
    them, so for 14 strategies of bots 1-4 they measured a version that does not trade (bot 1: TV02 INTC stop x0.6,
    TT45 WTI target x0.7, ...) and the whole-bot replay counted trades the bot never takes. Read once per run."""
    if bot in _LIVE_RULES:
        return _LIVE_RULES[bot]
    out = {}
    if bot in ("bot1", "bot2", "bot3", "bot4"):
        try:
            from agents import repo_io as RIO
            sec = ((RIO.live_json("live/manager_state.json").get("risk") or {}).get(bot) or {})
            for k in RULE_KINDS:
                for combo, v in (sec.get(k) or {}).items():
                    if v is not None and v != [] and v != {}:
                        out.setdefault(combo, {})[k] = v
        except Exception:  # noqa: BLE001
            out = {}
    _LIVE_RULES[bot] = out
    return out


def plans_file(bot, out_dir):
    """live/bots/<bot>.json's plans with the live rules merged in ("rules" per combo), for run_rules --plans-file."""
    bf = ROOT / "live" / "bots" / f"{bot}.json"
    if not bf.exists():
        return None
    try:
        plans = json.loads(bf.read_text(encoding="utf-8")).get("plans") or {}
    except Exception:  # noqa: BLE001
        return bf
    rules = live_rules(bot)
    if not rules:
        return bf
    merged = {c: dict(p) for c, p in plans.items()}
    for c, r in rules.items():
        merged.setdefault(c, {})["rules"] = r
    f = out_dir / f"plans_{bot}.json"
    f.write_text(json.dumps({"plans": merged}, ensure_ascii=False), encoding="utf-8")
    return f


def replay_family(plan):
    """The experiment family whose trades the whole-bot check takes for a strategy (portfolio_check.current_trades)."""
    if plan.get("ladder"):
        return "ladder"
    if plan.get("filter"):
        return "filters"
    if any(k in plan for k in ("be_way", "be_r", "partial")):
        return "manage"
    return "exits"


def experiments(bot, budget_s, where):
    st = Store(bot)
    todo = []
    fixing_ = fixing(bot)
    for combo in bot_combos(bot):
        sym = combo.split("|")[2]
        if not runs_here(sym, where):
            continue
        prio = (loss_priority(bot).get(combo) or {})
        pl = plan_now(bot, combo)
        for fam in FAMILIES:
            if fam == "pyr" and bot not in PYR_BOTS:
                continue
            age = st.age_days(fam, combo)
            d = st.get(fam, combo) or {}
            if d.get("variants") and age < RERUN_DAYS and (d.get("plan") or {}) != pl:
                # the bot's plan for this strategy changed since this experiment (13.11.2026: TV58 USDCAD bot 3 - a
                # filter card measured on the old exits): run again on top of what runs now, before the regular re-runs
                todo.append((0.5, fam, -age, sym, combo))
                continue
            # the loss day the review asked about ("why": "2026-11-12: ..."): a result measured after that day, with
            # today's costs (Store.get drops stale ones), already answers it (13.11.2026: the reports were written again
            # in a new format at 04:46 with a new "at", and every strategy tested that night with the real costs was
            # queued once more ahead of the ones never tested - TV11 USDJPY, TV41 NZDUSD, TV10 US100 waited)
            loss_day = max((str(w)[:10] for w in (prio.get("why") or [])), default=str(prio.get("at") or "")[:10])
            if fam in (prio.get("families") or []) and str(d.get("at") or "")[:10] <= loss_day:
                # the daily loss review asked for this test (agents/loss_review.py): first, before anything else
                todo.append((-1, fam, -age, sym, combo))
                continue
            if fam == "filters" and bot in FILTERS_EXTRA and d.get("variants") and not d.get("filters_extra"):
                # measured before the user's EMA 9/20 + Bollinger-width filters (task tal1411ema, 14.11.2026): again
                todo.append((1, fam, -age, sym, combo))
                continue
            if age >= RERUN_DAYS or (d.get("variants") and not tr_ok(d)):    # (12.11.2026: trades for the whole-bot check)
                pr = 0 if age == 999 or combo in fixing_ else 1
                if pr == 0 and combo not in fixing_ and fam != replay_family(pl):
                    # the whole-bot check reads one family per strategy (its built filter / management, else the
                    # exits): that one first for every strategy, the other families after (13.11.2026: bot 1's 2026
                    # replay saw 8 of 46 strategies - every re-run went through all families, one instrument at a time)
                    pr = 0.8
                if fam in ("fib", "ladder", "pyr") and combo not in fixing_:
                    pr = 1.5                     # (the ladder too: a new family for every strategy, 13.11.2026)                     # the Fibonacci grid after the re-runs the whole-bot check waits for
                todo.append((pr, fam, -age, sym, combo))
            elif d.get("variants") and legacy_tr(d):                         # trades stored in 1000-minute steps
                todo.append((2, fam, -age, sym, combo))
    # a 1-minute strategy runs in its own group, after the instrument's other ones (manager, 13.11.2026 night: bots 2-4
    # put SC001|1Min|JP225 / USDCAD in one exits run with the instrument's 15Min-4Hour strategies; the 1-minute one
    # needs more than the run's minutes on ~1M bars, so the whole group ran out of time every run since 31.10 and
    # TT16 JP225, TV02 JP225, IM03 / QC04 / SC002 USDCAD never got their exits tested)
    one_min = lambda c: c.split("|")[1] == "1Min"                      # noqa: E731
    # never-run first, one instrument's combos together. Inside one priority the instrument whose oldest result is the
    # oldest goes first, not the alphabet (bots 1-4 smart agent, 14.11.2026: bot 1's loss-review list fills every run's
    # minutes, sorted A-Z, so USDJPY / WTI / XAUUSD / XAGUSD / XPTUSD were never reached - SE02 and TV11 USDJPY, TV20
    # XAUUSD, TT45 WTI, 18 of bot 1's 46 strategies, had no result at all, while ARM / ASML / ETHUSD ran again daily)
    oldest = {}
    for x in todo:
        oldest[(x[0], x[3])] = min(oldest.get((x[0], x[3]), 0), x[2])
    # (a 1-minute strategy still goes after the tier's other ones - one such run can take a whole run's minutes)
    todo.sort(key=lambda x: (x[0], one_min(x[4]), oldest[(x[0], x[3])], x[3], x[1], x[2]))
    t0, done = time.time(), 0
    groups = {}
    for x in todo:
        groups.setdefault((x[3], x[1], x[4] if one_min(x[4]) else ""), []).append(x[4])
    for (sym, fam, _one), combos in groups.items():
        left = budget_s - (time.time() - t0)
        if left < 120:
            break
        ts = time.time()
        try:
            t, rows = run_rules(combos, fam, left / 60)
        except Exception as e:  # noqa: BLE001
            if "Time budget reached" in str(e) and left < 0.6 * budget_s:
                # the run's own minutes ran out on its last group (13.11.2026: JP225 / USDCAD in bots 2-4 every run):
                # not a failure - nothing is stored, so the group keeps its place and runs first next time
                print(f"  {sym} {fam}: stopped for time ({left / 60:.0f} min were left), first next run")
                break
            print(f"  {sym} {fam}: failed - {str(e)[-300:]}")
            # a combo that cannot run is retried tomorrow; a 1-minute one that ran out of a whole run's minutes alone
            # in 3 days (it would take most of every run and still not finish - manager, 13.11.2026 night)
            wait = 3 if (_one and "Time budget reached" in str(e)) else 1
            for c in combos:
                st.put(fam, c, {"at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RERUN_DAYS - wait)).isoformat(),
                                "combo": c, "error": str(e)[-300:]})
            continue
        for c in combos:
            sid, tf, s = c.split("|")
            tc = t[(t["sid"] == sid) & (t["tf"] == tf) & (t["symbol"] == s)]
            doc = {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "combo": c, "family": fam,
                   "secs": round((time.time() - ts) / max(1, len(combos)), 1), "variants": {}, "plan": plan_now(bot, c)}
            for v, g in tc.groupby("variant") if "variant" in tc else []:
                doc["variants"][v] = variant_stats(g)
            # the 2026 trades of the variants a card can come from (the one in the bot now, the default and the five
            # best on the choosing period), for the whole-bot check (agents/portfolio_check.py)
            doc["has_target"] = (rows.get(c) or {}).get("has_target")
            keep = {base_variant(bot, c, fam, doc), {"exits": "own|be0.6|stop1|entry0", "filters": "none"}.get(fam, base_variant(bot, c, fam, doc))}
            keep |= set(sorted((v for v, x in doc["variants"].items() if (x["train"].get("eff_n") or 0) >= 10),
                               key=lambda v: -(doc["variants"][v]["train"]["mean"] or -9))[:5])
            # ... and the five with the most TOTAL R on the choosing period (manager, 14.11.2026: the cards pick the
            # variant by total R - objective.md - but only the best five by R a trade kept their trades, so the card's
            # own variant often had none, the whole-bot check returned "unknown" and every improve card in bots 1-4
            # since 13.11 went in without it)
            keep |= set(sorted((v for v, x in doc["variants"].items() if (x["train"].get("eff_n") or 0) >= 10
                                and x["train"].get("mean") is not None),
                               key=lambda v: -(doc["variants"][v]["train"]["eff_n"] or 0) * doc["variants"][v]["train"]["mean"])[:5])
            doc["tr"] = {v: trades_2026(tc[tc["variant"] == v]) for v in keep if "variant" in tc and v in doc["variants"]}
            # the choosing period's trades of the variant in the bot now (whole-bot rules chosen on 2025, 13.11.2026)
            b0 = base_variant(bot, c, fam, doc)
            if "variant" in tc and b0 in doc["variants"]:
                doc["tr25"] = {b0: trades_2026(tc[tc["variant"] == b0], train=True)}
            # the machine's own unweighted stats of every variant (the filters keep only two variants' trades)
            doc["grid"] = [{"variant": x["variant"], "train": x["train"], "valid": x["valid"]}
                           for x in (rows.get(c) or {}).get("all", [])]
            if fam == "filters" and bot in FILTERS_EXTRA:
                doc["filters_extra"] = 1
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


def can_test(bot, s_):
    """Every bot's agent tests every new strategy the scout brings - whether it fits the bot is the test's answer, not
    the scout's guess (user, 12.11.2026; the scout's "bots" field is only a hint). The one limit is technical: the
    scalping bots (5-6) trade 1-15 minute charts, so a strategy whose source states only longer timeframes has nothing
    to run there."""
    tfs = [x for x in (s_.get("timeframes") or []) if x in ("1Min", "5Min", "15Min", "1Hour", "4Hour")] or ["1Hour"]
    return bot not in NEW_ONLY or any(x in ("1Min", "5Min", "15Min") for x in tfs)


def untested_new(bot, where):
    st = Store(bot)
    for s_ in inbox():
        if not s_.get("id") or not can_test(bot, s_):
            continue
        tfs = [x for x in (s_.get("timeframes") or []) if x in ("1Min", "5Min", "15Min", "1Hour", "4Hour")] or ["1Hour"]
        for sym in bot_symbols(bot):
            if runs_here(sym, where) and any(st.age_days("new", f"{s_['id']}|{tf}|{sym}") >= RERUN_DAYS for tf in tfs):
                return True
    return False


NEW_BATCH = 30                         # combos of one instrument in one backtest run (its bars are loaded once)


def _new_caps(bot):
    """Per-instrument batch sizes cut after a batch timed out (reports/agents/<bot>/new_batch.json)."""
    try:
        return json.loads((ROOT / "reports" / "agents" / bot / "new_batch.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_new_caps(bot, caps):
    p = ROOT / "reports" / "agents" / bot / "new_batch.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(caps, indent=1), encoding="utf-8")


def test_new(bot, budget_s, where, symbols=None, timeframes=None):
    """Every scout strategy, on the bot's instruments and the timeframes its source states, once (re-run after RERUN_DAYS).
    The due combos of one instrument go into one backtest run, several strategies together (manager, 12.11.2026: one
    run per strategy x instrument loaded the same bars again each time - ~20 of ~1,000 due combos a day per bot)."""
    st = Store(bot)
    t0, done = time.time(), 0
    syms = symbols or bot_symbols(bot)
    queue = []                                                   # (sym, strategy, combos), in the scout's order
    for s_ in inbox():
        if not s_.get("id") or not can_test(bot, s_):
            continue
        sid = s_["id"]
        tfs = [x for x in (s_.get("timeframes") or []) if x in ("1Min", "5Min", "15Min", "1Hour", "4Hour")] or (timeframes or ["1Hour"])
        if bot in NEW_ONLY:
            tfs = [x for x in tfs if x in ("1Min", "5Min", "15Min")]          # scalping bots: short timeframes only
        # on ALL the bot's instruments, not only the markets the source names (user: test everything, each strategy
        # where it wins)
        mk = set(s_.get("markets") or [])
        for sym in [x for x in syms if runs_here(x, where)]:
            combos = [f"{sid}|{tf}|{sym}" for tf in tfs]
            if any(st.age_days("new", c) >= RERUN_DAYS for c in combos):
                fresh = any(st.age_days("new", c) >= 999 for c in combos)
                queue.append((sym, s_, combos, (0 if fresh else 1,                          # never tested before a re-run
                                                0 if not mk or U.CLASS.get(sym) in mk else 1)))  # the source's markets first
    # 13.11.2026 (bots 1-4 smart agent): in the scout's order, strategy by strategy on all the bot's instruments, the
    # 30-combo batches of one instrument took SC001-SC028 every time - by 13.11 bots 1-4 had tested nothing after SC028
    # (82 strategies, ~1,250-3,650 combos waiting in each bot) and SC001 would come back for its 14-day re-run first.
    # Now: never-tested combos before re-runs, the markets the source names before the others, and among those the
    # strategies tested on the fewest of the bot's instruments first - so every new strategy gets its first fair test
    # within a run or two. Nothing is skipped: the rest follows in the same order in the next runs.
    cover = {}
    for s_ in inbox():
        if s_.get("id"):
            cover[s_["id"]] = sum(1 for sym in syms for tf in (s_.get("timeframes") or ["1Hour"])
                                  if st.age_days("new", f"{s_['id']}|{tf}|{sym}") < 999)
    queue.sort(key=lambda q: (q[3][0], q[3][1], cover.get(q[1]["id"], 0)))     # stable: the scout's order within
    queue = [q[:3] for q in queue]

    def store(t, s_, combos):
        n = 0
        for c in combos:
            sid_, tf, s = c.split("|")
            tc = t[(t["sid"] == sid_) & (t["tf"] == tf) & (t["symbol"] == s)]
            st.put("new", c, {"at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "combo": c, "family": "new",
                              "tr": {"new": trades_2026(tc)} if len(tc) else {},
                              "tr25": {"new": trades_2026(tc, train=True)} if len(tc) else {},
                              "scout": {k: s_.get(k) for k in ("name", "source", "url", "added")},
                              "stats": variant_stats(tc) if len(tc) else None})
            n += 1
        return n

    def failed(combos, e):
        for c in combos:
            st.put("new", c, {"at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RERUN_DAYS - 1)).isoformat(),
                              "combo": c, "error": str(e)[-300:]})

    caps = _new_caps(bot)
    while queue:
        left = budget_s - (time.time() - t0)
        if left < 120:
            print(f"new strategies: stopped for time, {done} tested, {sum(len(q[2]) for q in queue)} combos left")
            return done
        sym = queue[0][0]
        cap = int(caps.get(sym, NEW_BATCH))
        batch, n = [], 0
        for q in list(queue):
            if q[0] == sym and (n == 0 or n + len(q[2]) <= cap):
                batch.append(q)
                n += len(q[2])
                queue.remove(q)
        try:
            t, _ = run_rules([c for q in batch for c in q[2]], "new", left / 60)
            for _, s_, combos in batch:
                done += store(t, s_, combos)
        except Exception as e:  # noqa: BLE001
            # 14.11.2026 (manager): a 30-combo batch on the metals (XAGUSD / XAUUSD / XPTUSD, with 1-minute combos) did
            # not finish inside the run's time - run_rules stopped on its time budget, no trades file, every strategy
            # put back, nothing stored - so bots 2-4 tested 0 new combos run after run, the same batch first every
            # time. Now a timed-out batch halves that instrument's batch size (kept for the next runs) and goes back
            # first in the queue; a single combo that still times out is marked failed and retried in a day.
            if "Time budget reached" in str(e) or "timed out" in str(e).lower():
                if left < 480:                       # little time was left anyway: not the batch's fault
                    queue[0:0] = batch
                    print(f"new strategies: stopped for time, {done} tested, {sum(len(q[2]) for q in queue)} combos left")
                    return done
                if n > 1:
                    caps[sym] = max(1, n // 2)
                    _save_new_caps(bot, caps)
                    print(f"new strategies: {sym} batch of {n} timed out, now {caps[sym]}")
                    queue[0:0] = batch
                else:
                    failed(batch[0][2], e)
                continue
            if len(batch) == 1 or "no market data" in str(e):
                for q in batch:
                    failed(q[2], e)
                continue
            # one strategy may have broken the whole run: each one alone, so the others still get their result
            for _, s_, combos in batch:
                if budget_s - (time.time() - t0) < 120:
                    queue.insert(0, (sym, s_, combos))
                    continue
                try:
                    t, _ = run_rules(combos, "new", (budget_s - (time.time() - t0)) / 60)
                    done += store(t, s_, combos)
                except Exception as e1:  # noqa: BLE001
                    failed(combos, e1)
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
            base_key = base_variant(bot, combo, fam, d)
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
            # THE GOAL (user, 12.11.2026, data/agents/objective.md): the variant with the most TOTAL R on the choosing
            # period (trades x R a trade), not the most R a trade (13.11.2026: TV58|1Hour|USDCAD bot 3 got a limit-entry
            # variant with 28 trades over the bot's version with more trades and more R in all)
            def _tot(x):
                return (x.get("eff_n") or 0) * x["mean"] if x.get("mean") is not None else -9e9
            v, best = max(cand, key=lambda vx: _tot(vx[1]["train"]))
            base_tot = _tot(base["train"]) if base["train"].get("mean") is not None else 0.0
            if _tot(best["train"]) <= max(0.0, base_tot):
                continue
            grid = d.get("grid") or []
            bval = next((g["valid"]["avg"] for g in grid if g["variant"] == base_key), None)
            others = [g for g in grid if g["variant"] != base_key and g["valid"]["n"] >= 5]
            share = (round(sum(1 for g in others if g["valid"]["avg"] > (bval or 0)) / len(others), 2) if others and bval is not None else None)
            what = (explain_exit(v) if fam == "exits" else explain_manage(v) if fam in ("manage", "fib")
                    else explain_ladder(v) if fam == "ladder" else explain_pyr(v) if fam == "pyr" else FILTER_HE.get(v, v))
            cards.append(K.card(
                bot, "improve", f"{combo}: {what}",
                f"{ {'exits': 'יעד / סטופ / כניסה', 'manage': 'ניהול העסקה (נקודת איזון / יציאה חלקית)', 'fib': 'ניהול העסקה ברמות פיבונאצ׳י (נקודת איזון / יציאה חלקית)', 'ladder': 'כניסה מדורגת', 'pyr': 'הוספה לעסקה מנצחת (פירמידה)'}.get(fam, 'סינון מצב שוק') } לאסטרטגיה {combo}: {what}. "
                f"נבחר על 30.04-31.12.2025 בלבד, ומוצג על 2026 שלא נראתה, עם החוקים של הבוט.",
                best["train"], best["unseen"], whole=best["whole"], halves_=best["halves"],
                neighbours={"values": len(grid) or len(vs), "share_positive": share},
                tested=len(grid) or len(vs), risk_pct=risk, baseline=base["unseen"],
                detail={"exp_at": d.get("at"), **({"needs_code": "live/runner.py: כמה הוראות לימיט לאסטרטגיה אחת - נבנה "
                                                   "כשהכרטיס מאושר"} if fam == "ladder" else
                                                  {"needs_code": "live/runner.py: הוספה לעסקה פתוחה והזזת הסטופ של "
                                                   "הכול - נבנה כשהכרטיס מאושר"} if fam == "pyr" else {})},
                change={"combo": combo, "family": "manage" if fam == "fib" else fam, "from": base_key, "to": v,
                        **({"src": "fib"} if fam == "fib" else {})}))
    # the scout's strategies
    for f in sorted(st.dir.glob("new_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if stale_source(d):
            continue
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
    BOT_NOW["bot"] = a.bot
    budget = a.minutes * 60
    t0 = time.time()
    if a.bot == "bot7" and a.where in ("pc", "all"):
        # bot 7's fix study (agents/news_fix.py): the three strategies pooled over all instruments, split by release
        # type / instrument / exits - at most once every 3 days, in its own process, up to half the budget
        try:
            from agents import news_fix as NF
            if NF.stale():
                mins = max(8.0, a.minutes / 2)
                subprocess.run([sys.executable, "-m", "agents.news_fix", "--minutes", str(mins)], cwd=str(ROOT),
                               timeout=int(mins * 60 + 1800))
        except Exception as e:  # noqa: BLE001 - the study's failure never stops the experiments
            print(f"bot7 fix study failed: {e}")
        # EV04, the straddle before a release (the user's task tal1411strad, 14.11.2026, agents/news_straddle.py): a
        # third of the budget, BEFORE the entry study (the user's priority, 14.11.2026), gold first
        try:
            from agents import news_straddle as NS
            if NS.stale():
                mins = max(6.0, a.minutes / 3)
                subprocess.run([sys.executable, "-m", "agents.news_straddle", "--minutes", str(mins)], cwd=str(ROOT),
                               timeout=int(mins * 60 + 1800))
        except Exception as e:  # noqa: BLE001
            print(f"bot7 straddle study failed: {e}")
        # fix attempt 6 for EV01 (13.11.2026): the entry's minute after the release, the spike size, the side and the
        # surprise (agents/news_entry.py), up to a third of the budget, once the fix study is complete
        try:
            from agents import news_entry as NE
            if NE.stale() and not NF.stale():
                # half the budget (14.11.2026: the user's rule - fix the losers first; EV03's attempt 5 was 9/16 after
                # a whole round and attempt 6, the box before the release, waits behind it)
                mins = max(6.0, a.minutes / 2)
                subprocess.run([sys.executable, "-m", "agents.news_entry", "--minutes", str(mins)], cwd=str(ROOT),
                               timeout=int(mins * 60 + 1800))
        except Exception as e:  # noqa: BLE001
            print(f"bot7 entry study failed: {e}")
    # the scout's strategies first, with up to a third of the time, while some are still untested (12.11.2026: with
    # 0.8 of the budget for the experiments, the leftover was under the 2-minute minimum and they never ran)
    first = test_new(a.bot, budget / 3, a.where) if untested_new(a.bot, a.where) else 0
    done, left = experiments(a.bot, budget - (time.time() - t0), a.where)
    new = first + test_new(a.bot, budget - (time.time() - t0), a.where)
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
