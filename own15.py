"""The user's own strategy, version 3 (user, 06.11.2026, drawn and marked on charts): identified on the 15-minute chart;
entry on the 1-minute gap inside the 15-minute gap (variant), target the nearest 1-minute swing high (variant).

  impulse   a leg of 8-30 fifteen-minute candles from a swing low to a swing high (mirror for sells) that is
            clearly bigger than the recent history: its size >= K x the typical move over the same number of
            candles in the previous ~5 days (median of |close[t] - close[t-n]|), and its average body >= 1.5 x
            the average body of the 50 candles before it ("a mass of buyers")
  entry     the FIRST fair-value gap of the leg in time (candle k-1's high below candle k+1's low); when the price
            comes back and TOUCHES the gap's near edge -> buy (the impulse's direction)
  stop      the swing low the leg started from, minus 4 pips (instruments without pips: 0.1 x ATR)
  target    the nearest swing high above the entry (usually a lower high made during the pullback), minus a
            little (2 pips / 0.05 x ATR) to close safely
  rules     risk:reward at least 1:1; the setup lives 8 hours after the leg's top is confirmed; it is gone if the
            price goes above the leg's top first, or below the stop; FTMO restricted news: no entry 2 minutes
            either side, positions closed 2 minutes before; positions closed before market breaks; one position
            at a time per instrument
  filter    variant: only when the 1-hour trend (EMA20 vs EMA50) agrees with the impulse, or not filtered
Fills and exits are walked on the 1-minute bars (to know which came first); costs = the real spread at the fill
minute + FTMO commission. TRAIN 2024-01-01..2025-04-30, VALID after.

  python own15.py --symbols EURUSD,GBPUSD
Writes output/own15_<tag>.csv.gz.
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
import run_rules as RR
from agents import own_core as CORE
from agents import own_gaps as OG
from agents.own_core import VARIANTS, pip_of, resample, swings, comm_px, run_symbol  # noqa: F401

# (K, 1h-trend filter, entry: '15m' gap edge / '1m' gap inside the 15m gap, target: '15m' / '1m' nearest swing high)
# last5: the last 1-minute turning point before the entry (5 minutes each side) beyond it (the user's marks, 06.11.2026);
# last5x: the same without the 1:1 minimum
# struct: the last real correction before the entry (the user, 06.11.2026): the last 1-minute turning point beyond the
# entry after which the price made a new low below the low before it (a high in a sideways pause does not count);
# structx: the same without the 1:1 minimum


# name -> (K impulse size vs typical, BODY_X, MIN_GAP, entry mode); 'base' = exactly the watch bot
CONFIGS = {"base": (2.0, 2.5, 0.33, "1m"),
           "k15": (1.5, 2.5, 0.33, "1m"),     # smaller impulses count
           "b18": (2.0, 1.8, 0.33, "1m"),     # less strong candle bodies
           "g0": (2.0, 2.5, 0.0, "1m"),       # any gap size
           "e15": (2.0, 2.5, 0.33, "15m"),    # enter at the 15-minute gap's edge, no 1-minute gap needed
           "all": (1.5, 1.8, 0.15, "15m"),    # all of them looser together
           # user (09.11.2026): entry at the body of the candle that leads the move (wide track's filters)
           "wgc": (2.0, 1.8, 0.33, "gc"), "wgo": (2.0, 1.8, 0.33, "go"),   # first gap's impulse candle: close / open
           "wfc": (2.0, 1.8, 0.33, "fc"), "wfo": (2.0, 1.8, 0.33, "fo")}   # leg's first candle: close / open


# user (09.11.2026): "half profit and stop to break-even" - at +pl x the risk close half, move the stop to the entry,
# the other half goes on to the same target (or back to the entry). Research only, computed from the same 1-minute path.
PARTIALS = (0.5, 0.75, 1.0)


# user (09.11.2026, a US30 trade on TradingView): "there was a swing high on the way - take 60% of the money there and
# move to break-even; that's the strategy". Level = the nearest 1-minute swing high (5 minutes each side) between the
# entry and the target, made after the leg started and confirmed by the fill (mirror for sells).
SWING_FRAC = 0.6
WAY = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # share of the way from the entry to the target


def swing_level(x, df, nth=1):
    d, px, tgt = x["dir"], x["entry"], x["target"]
    h1, l1 = df["h"].to_numpy(float), df["l"].to_numpy(float)
    i0 = df.index.get_loc(pd.Timestamp(x["time"]))
    a0 = df.index.searchsorted(pd.Timestamp(x["leg_start"]))
    arr = h1 if d > 0 else l1
    w = 5
    seg = pd.Series(arr[a0:i0 + 1])
    ext = (seg.rolling(2 * w + 1, center=True).max() if d > 0 else seg.rolling(2 * w + 1, center=True).min()).to_numpy()
    lv = [arr[a0 + j] for j in range(len(seg)) if arr[a0 + j] == ext[j] and a0 + j + w <= i0
          and d * (arr[a0 + j] - px) > 0 and d * (tgt - arr[a0 + j]) > 0]
    # nth=2 (user, 09.11.2026: "the next swing after that"): the second distinct swing level on the way
    lv = sorted(set(lv), reverse=d < 0)
    if len(lv) < nth:
        return None
    return lv[nth - 1]


def partial_exit(x, df, stop_at, pl, frac=0.5, lvl=None, tag=None):
    if x.get("status") != "closed":
        return None
    d, px, stop, tgt = x["dir"], x["entry"], x["stop"], x["target"]
    risk = d * (px - stop)
    if risk <= 0:
        return None
    h1, l1, o1, c1 = (df[k].to_numpy(float) for k in ("h", "l", "o", "c"))
    i0 = df.index.get_loc(pd.Timestamp(x["time"]))
    cost_r = x["cost_r"]
    if lvl is None:
        lvl = px + d * pl * risk
    else:
        pl = d * (lvl - px) / risk
    half = None                       # bar where the half was taken
    for q in range(i0, len(df)):
        hit_stop = (l1[q] <= stop) if d > 0 else (h1[q] >= stop)
        if half is None:
            if hit_stop:                                  # same bar as the level: count the stop (the worse case)
                ex = (min(stop, o1[q]) if d > 0 else max(stop, o1[q])) if q > i0 else stop
                r = d * (ex - px) / risk
                break
            if q > i0 and stop_at[q]:
                r = d * (c1[q] - px) / risk
                break
            if q > i0 and ((h1[q] >= lvl) if d > 0 else (l1[q] <= lvl)):
                if d * (tgt - px) <= pl * risk:           # the target comes first: plain trade
                    return None
                half, first = q, pl
            continue
        if (l1[q] <= px) if d > 0 else (h1[q] >= px):     # back to the entry: the second half at 0
            r = frac * first + (1 - frac) * d * ((min(px, o1[q]) if d > 0 else max(px, o1[q])) - px) / risk
            break
        if (h1[q] >= tgt) if d > 0 else (l1[q] <= tgt):
            r = frac * first + (1 - frac) * d * ((max(tgt, o1[q]) if d > 0 else min(tgt, o1[q])) - px) / risk
            break
        if stop_at[q]:
            r = frac * first + (1 - frac) * d * (c1[q] - px) / risk
            break
    else:
        return None
    y = dict(x)
    y["variant"] = f"{x['variant']}|{tag or f'p{pl:g}'}"
    if tag:
        y["part_px"] = lvl
    y["exit_time"] = df.index[q]
    y["reason"] = ("half+" if half is not None else "") + x["reason"]
    y["r"] = round(r - cost_r, 4)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    ap.add_argument("--configs", default="", help="looser variants to test, e.g. base,k15,b18 (see CONFIGS)")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tag = a.symbols.replace(",", "_").replace("/", "-")[:60]
    ev, _ = NC.load()
    rows = []
    for s in [x for x in a.symbols.split(",") if x]:
        t0 = time.time()
        try:
            df = P.bars(s, "1Min", max(a.start, RP.MINUTE_START), end)
            sp = P.spread_1m(s, a.start, end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-120:]}")
            continue
        if df is None or df.empty or sp is None or len(sp) == 0:
            print(f"{s}: no data / no real spread - skipped")
            continue
        if "v" not in df:
            df["v"] = 1.0
        c = df["c"].to_numpy(float)
        cost = np.abs(sp.sort_index().reindex(df.index, method="ffill").to_numpy(float)) + comm_px(s, c)
        cost = np.where(np.isfinite(cost), cost, np.nanmedian(cost))
        stop_at = RR.break_mask(df, "1Min")
        sm = RR.session_mask(df, s, "1Min")
        if sm is not None:
            stop_at = stop_at | np.asarray(sm, bool)
        nm, noent = RR.news_masks(df, s, "1Min")
        blk = NC.blocked(pd.Series(df.index), s, ev, minutes=2).to_numpy()
        noent = np.asarray(noent, bool) | blk
        stop_at = stop_at | np.asarray(nm, bool) | np.r_[blk[1:], False]
        if a.configs:
            # research only (user, 09.11.2026: more trades?): the bot's 'struct' rules with one filter loosened at a time.
            # own_core's module constants are set for each run and restored after, the bot's file is not changed.
            keep = (CORE.BODY_X, CORE.MIN_GAP, CORE.STOP_SW)
            for name in a.configs.split(","):
                if name.startswith("H"):
                    # v6 (user, 10.11.2026): first gap of a move, first touch, big move; "H<stop>[d]" e.g. Hgap, Horigind, Hs2d
                    keepg = (OG.STOP6, OG.DOMINANT, OG.ARM6, OG.MOVE_W)
                    OG.ARM6 = name.startswith("HA")                  # "HA...": only after the move turned
                    core = name[2:] if OG.ARM6 else name[1:]
                    OG.MOVE_W = 2
                    if core.startswith("W"):                         # "HW6gapd": a move ends only at a 6-each-side swing
                        OG.MOVE_W = int(core[1]); core = core[2:]
                    OG.STOP6, OG.DOMINANT = core.rstrip("d") or "gap", core.endswith("d")
                    try:
                        gx = OG.run_gaps6(s, df, cost, stop_at, noent, variant=name)
                    finally:
                        OG.STOP6, OG.DOMINANT, OG.ARM6, OG.MOVE_W = keepg
                    for x in gx:
                        rows.append(x)
                        if x.get("status") == "closed":
                            for w in WAY:
                                lvw = x["entry"] + w * (x["target"] - x["entry"])
                                y = partial_exit(x, df, stop_at, 0, SWING_FRAC, lvw, f"pt{w:g}")
                                rows.append(y if y is not None else dict(x, variant=f"{x['variant']}|pt{w:g}"))
                    continue
                if name.startswith("G"):
                    # v5 (user, 09.11.2026): every gap of a strong move; "G<stop>[d]" e.g. Gs2, Gs3, Ggap, Gs2d (dominant)
                    keepg = (OG.STOP, OG.DOMINANT, OG.TFM, OG.V5_MAX_COST)
                    core, _, tf = name[1:].partition("@")          # "Ggap@5": detection on 5-minute candles
                    OG.TFM = int(tf or 15)
                    OG.V5_MAX_COST = 0.10 if tf else None
                    OG.STOP, OG.DOMINANT = core.rstrip("d"), core.endswith("d")
                    try:
                        gx = OG.run_gaps(s, df, cost, stop_at, noent, variant=name)
                    finally:
                        OG.STOP, OG.DOMINANT, OG.TFM, OG.V5_MAX_COST = keepg
                    for x in gx:
                        rows.append(x)
                        if x.get("status") == "closed":
                            for w in WAY:
                                lvw = x["entry"] + w * (x["target"] - x["entry"])
                                y = partial_exit(x, df, stop_at, 0, SWING_FRAC, lvw, f"pt{w:g}")
                                rows.append(y if y is not None else dict(x, variant=f"{x['variant']}|pt{w:g}"))
                    continue
                base_name, _, sw = name.partition("@s")         # "b18@s3": the stop beyond the 3rd swing back
                v4 = base_name.endswith("+v4")                   # "+v4": dominant move + open gap (user, 09.11.2026)
                K, bx, mg, em = CONFIGS[base_name.replace("+v4", "")]
                CORE.BODY_X, CORE.MIN_GAP, CORE.STOP_SW = bx, mg, int(sw or 0)
                CORE.DOMINANT = CORE.GAP_OPEN = v4
                for x in run_symbol(s, df, cost, stop_at, noent, [(K, False, em, "struct")]):
                    x["variant"] = name
                    rows.append(x)
                    for pl in PARTIALS:
                        y = partial_exit(x, df, stop_at, pl)
                        if y is not None:
                            rows.append(y)
                    if x.get("status") == "closed":
                        lv = swing_level(x, df)
                        y = partial_exit(x, df, stop_at, 0, SWING_FRAC, lv, "sw60") if lv is not None else None
                        if y is None:                       # no swing on the way / target first: the plain trade
                            y = dict(x, variant=f"{x['variant']}|sw60")
                        rows.append(y)
                        lv2 = swing_level(x, df, 2)
                        y = partial_exit(x, df, stop_at, 0, SWING_FRAC, lv2, "sw2") if lv2 is not None else None
                        if y is None:
                            y = dict(x, variant=f"{x['variant']}|sw2")
                        rows.append(y)
                        # user (09.11.2026): optimise - part of the position out at a share of the way to the target,
                        # stop to break-even (60% out here; other shares are derived from the same path in the analysis)
                        for w in WAY:
                            lvw = x["entry"] + w * (x["target"] - x["entry"])
                            y = partial_exit(x, df, stop_at, 0, SWING_FRAC, lvw, f"pt{w:g}")
                            rows.append(y if y is not None else dict(x, variant=f"{x['variant']}|pt{w:g}"))
            CORE.BODY_X, CORE.MIN_GAP, CORE.STOP_SW = keep
            CORE.DOMINANT = CORE.GAP_OPEN = False
        else:
            rows += run_symbol(s, df, cost, stop_at, noent, VARIANTS)
        print(f"{s}: {sum(1 for x in rows if x['symbol'] == s)} trades (all variants), {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / f"own15_{tag}.csv.gz", index=False)
    print("trades:", len(rows))


if __name__ == "__main__":
    main()
