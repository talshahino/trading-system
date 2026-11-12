"""The improvement agent of bot 6 (CME micro index futures, virtual; user, 10-11.11.2026).

Bot 6 trades the scalping bot's methods (gap tracks v5 on 15 minutes and the 1-minute track) on MNQ / MES / MYM, so its
agent is bot 5's agent (agent_bot5.py: the same experiments, knobs, fair split and cards) with bot 6's own world:
  instruments  the index CFDs the bot reads (US100 / US500 / US30), sized and costed as micro futures:
               a tick of spread + the commission and a tick of slippage per contract (live/fut_account.py)
  sessions     FTMO Futures: everything closed 16:05-18:00 New York and over the weekend, no new entry from 15:45
  live record  the bot's virtual trades (live/watch_fut_state.json on branch live)
A method one of the two agents finds profitable reaches the other by itself: both test the same knobs, each on its own
instruments and costs (user, 11.11.2026), and every bot 6 card says what bot 5's agent found for the same change.

The user's gap method (data/agents/strategies/tal_gap.md, agents/tal_gap.py) is tested here too, on bot 6's world, up to
TAL_SHARE of each run until every pair and variant is done (then again every 14 days); results next to bot 5's in
data/agents/strategies/tal_gap_results.json and as cards in cards.json.

  python agent_bot6.py [--minutes 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import agent_bot5 as A
from agents import agentkit as K
from agents import repo_io as RIO
from agents import tal_gap as TG
from live import fut_account as FA

A.BOT = "bot6"
A.OUT = Path("reports/agents/bot6")
A.EXP = A.OUT / "exp"
A.IDX = A.EXP / "idx"
A.TRACKS = {
    "v5": {"symbols": ["US100", "US500", "US30"], "risk_pct": FA.RISK_V5 * 100, "base": dict(A.TRACKS["v5"]["base"])},
    "v1m": {"symbols": ["US100", "US500", "US30"], "risk_pct": FA.RISK_V1M * 100, "base": dict(A.TRACKS["v1m"]["base"])},
}
A.TRK_NAME = {"v5": "חוזים · מסלול 15 דקות (v5)", "v1m": "חוזים · מסלול הדקה"}
TAL_DIR = A.OUT / "tal"
TAL_SHARE = 0.5                 # of each run's minutes, while the user's method still has untested pairs
WORLD = {"symbols": ["US100", "US500", "US30"], "contracts": "MNQ / MES / MYM (micro)",
         "costs": f"tick spread + ${FA.COMM:g} commission + {FA.SLIP_TICKS:g} tick slippage per contract",
         "sessions": "FTMO Futures: closed 16:05-18:00 New York and weekends (forced exit), no new entry from 15:45",
         "stop_buffer": "2 ticks", "split": "train to 2025-12-31, unseen 2026"}


def prep(s, end):
    O15 = A.O15
    df = O15.P.bars(s, "1Min", max(A.START, O15.RP.MINUTE_START), end)
    if df is None or df.empty:
        return None
    if "v" not in df:
        df["v"] = 1.0
    _, pv, tick = FA.SPEC[s]
    cost = np.full(len(df), tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv)    # price units, per contract
    shut, late = FA.session_masks(df)
    ev, _ = O15.NC.load()
    blk = O15.NC.blocked(pd.Series(df.index), s, ev, minutes=2).to_numpy()
    noent = np.asarray(late, bool) | blk
    stop_at = np.asarray(shut, bool) | np.r_[blk[1:], False]
    return df, cost, stop_at, noent


def health():
    s = RIO.live_json("live/watch_fut_state.json")
    out = {}
    for t in s.get("closed", []) or []:
        try:
            risk = abs(float(t["open_px"]) - float(t["sl0"]))
            if risk <= 0:
                continue
            side = 1 if t.get("side") == "buy" else -1
            r = (float(t["close_px"]) - float(t["open_px"])) * side / risk
        except Exception:  # noqa: BLE001
            continue
        trk = "v1m" if "|1Min|" in (t.get("combo") or "") else "v5"
        out.setdefault(trk, []).append((t.get("close_t"), r))
    return {k: K.wstats([r for _, r in v], K.weights([x for x, _ in v])) for k, v in out.items()}


def bot5_view():
    """What bot 5's agent found for each change (its cards), shown on bot 6's card of the same change."""
    try:
        d = json.loads(Path("reports/agents/bot5/cards.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {(c.get("change") or {}).get("track", "") + "|" + str((c.get("change") or {}).get("knob")) + "|" + str((c.get("change") or {}).get("to")):
            {"tier": c.get("tier"), "unseen": (c.get("unseen") or {}).get("mean")} for c in d.get("cards", []) if c.get("kind") == "improve"}


A.prep = prep
A.health = health


def tal_load(s):
    """Bot 6's world for the user's gap method: micro-futures costs, the futures session, the calendar masks."""
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = A.O15.P.bars(s, "1Min", max(A.START, A.O15.RP.MINUTE_START), end)
    if df is None or df.empty:
        return None
    _, pv, tick = FA.SPEC[s]
    cost = np.full(len(df), tick + (FA.COMM + FA.SLIP_TICKS * tick * pv) / pv)
    shut, late = FA.session_masks(df)
    ev, _ = A.O15.NC.load()
    times = pd.Series(df.index)
    cal = {f"cal{m}": A.O15.NC.blocked(times, s, ev, minutes=m).to_numpy() for m in (15, 60)}
    return TG.Data(s, df, cost, shut, late, 2 * tick, cal)


def tal(budget_s):
    idx, left = TG.run("bot6", WORLD["symbols"], tal_load, TAL_DIR, budget_s)
    pairs, answers = TG.summarize(idx, WORLD["symbols"], TAL_DIR)
    planned = len(TG.PAIRS) * len(TG.variants()) * len(WORLD["symbols"])
    doc = TG.write_results("bot6", pairs, answers, WORLD, [planned - left, planned])
    return TG.cards("bot6", pairs, doc.get("bot5"), "בוט 5", risk_pct=FA.RISK_V5 * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=None)
    a, _ = ap.parse_known_args()
    mins = a.minutes if a.minutes is not None else float(A.plan().get("minutes", 30))
    t0, tcards = time.time(), []
    try:
        tcards = tal(TAL_SHARE * mins * 60)
    except Exception as e:  # noqa: BLE001 - the user's method must not stop the nightly experiments
        print(f"tal_gap failed: {e}")
    rest = max(1.0, mins - (time.time() - t0) / 60)
    argv = [x for i, x in enumerate(sys.argv) if x != "--minutes" and (i == 0 or sys.argv[i - 1] != "--minutes")]
    sys.argv = argv + ["--minutes", f"{rest:.1f}"]
    A.main()
    f = A.OUT / "cards.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    for c in tcards:
        c["tier"], c["verdict"] = K.judge(c)
    d["cards"] = [c for c in d.get("cards", []) if (c.get("change") or {}).get("method") != "tal_gap"] + tcards
    view = bot5_view()
    for c in d.get("cards", []):
        ch = c.get("change") or {}
        b5 = view.get(f"{ch.get('track')}|{ch.get('knob')}|{ch.get('to')}")
        if b5:
            c["detail"]["bot5"] = b5
            c["what"] += f" · בבוט 5 אותו שינוי: {b5['tier']}, {b5['unseen']:+.2f}R בתקופה שלא נראתה" if b5.get("unseen") is not None else ""
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
