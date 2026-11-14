"""Indicator strategy builder (task tal1411indstrat, the user 14.11.2026 17:34: "I want the indicators to work as a
builder of NEW strategies, not only as a filter for existing ones").

Every library indicator used here is given an explicit role:
- DIRECTIONS: a trend / regime direction per bar (+1 / -1 / 0);
- TRIGGERS: an entry event per bar (long, short) - a pullback, an oscillator turn, a structure retest;
- REGIMES: a "market is trending / expanding" gate for breakout triggers.
Strategies (IB###, same contract as agents/rules_agent: fn(df) -> (long, short, plan)):
  (b) direction from one indicator + trigger from another, in the direction's favour only - combinations WITH LOGIC
      (a trend filter + a pullback / momentum entry in that trend; a volatility regime + a squeeze breakout), not a
      blind search of every pair: each trigger is paired with the three directions that suit its type;
  (c) direction on the 4-hour chart (completed 4H bars only) + the trigger on the 5 / 15-minute chart;
  (a) the single-indicator strategies already exist as the scout's SC strategies (one or more per indicator).
Exits, fixed risk per trade: E1 = stop 1.5 ATR, target 2R; E2 = stop 2 ATR, no target, 2.5 ATR trailing stop and
out when the direction flips. The bots' agents test every IB like any scout strategy (live rules, FTMO costs, choice
up to 2025, shown on 2026) and correct their bar for the number of combinations tried (the user's rule).
No repaint: every component is a closed-bar library indicator; the 4H direction is taken from the last COMPLETED
4H bar."""
import numpy as np
import pandas as pd

import agents.indicators as I
from agents.indicators._core import cols, ema, shift


def _b(x):
    return np.nan_to_num(np.asarray(x, dtype=float)) > 0


def _sign(up, dn):
    return np.where(_b(up), 1.0, np.where(_b(dn), -1.0, 0.0))


# ---------------- directions (+1 / -1 / 0) ----------------
def d_supertrend(df):
    return np.sign(np.nan_to_num(I.supertrend(df)["trend"]))


def d_halftrend(df):
    return np.where(np.asarray(I.halftrend(df)["trend"]) == 0, 1.0, -1.0)      # HalfTrend: 0 = up, 1 = down


def d_ichimoku(df):
    k = I.ichimoku(df)
    return _sign(k["above_cloud"], k["below_cloud"])


def d_gaussian(df):
    g = I.gaussian_channel(df)
    c = cols(df)[3]
    return _sign(_b(g["rising"]) & (c > g["filt"]), ~_b(g["rising"]) & (c < g["filt"]))


def d_zero_lag(df):
    return np.sign(np.nan_to_num(I.zero_lag_trend(df)["trend"]))


def d_ema_stack(df):
    e = I.ema_stack(df)
    return _sign(e["full_up"], e["full_dn"])


def d_smc(df):
    return np.sign(np.nan_to_num(I.smc(df, 5)["trend"]))


def d_mcginley(df):
    c = cols(df)[3]
    m = I.mcginley(df)
    pm = shift(m, 1)
    with np.errstate(invalid="ignore"):
        return _sign((c > m) & (m > pm), (c < m) & (m < pm))


DIRECTIONS = {
    "supertrend": (d_supertrend, "SuperTrend 10x3 side"),
    "halftrend": (d_halftrend, "HalfTrend side"),
    "ichimoku": (d_ichimoku, "price above / below the Ichimoku cloud"),
    "gaussian": (d_gaussian, "Gaussian Channel filter slope and side"),
    "zero_lag": (d_zero_lag, "Zero Lag Trend side"),
    "ema_stack": (d_ema_stack, "EMA 9/21/50/100/200 stack fully ordered"),
    "smc": (d_smc, "SMC structure trend (k 5)"),
    "mcginley": (d_mcginley, "close vs a sloping McGinley Dynamic"),
}


# ---------------- triggers (long event, short event) ----------------
def t_rsi_pull(df):
    r = I.rsi_regime(df)
    return _b(r["pull_long"]), _b(r["pull_short"])


def t_wavetrend(df):
    """WaveTrend cross on the pullback side of zero (wt1 crosses wt2 up while under 0 / down while over 0) - the
    with-trend use; the OB / OS crosses of the original are counter-trend (SC227)."""
    from agents.indicators._core import cross_down, cross_up
    w = I.wavetrend(df)
    w1, w2 = np.asarray(w["wt1"], float), np.asarray(w["wt2"], float)
    return cross_up(w1, w2) & (w1 < 0), cross_down(w1, w2) & (w1 > 0)


def t_qqe(df):
    q = I.qqe(df)
    return _b(q["buy"]), _b(q["sell"])


def t_laguerre(df):
    q = I.laguerre_rsi(df)
    return _b(q["buy"]), _b(q["sell"])


def t_fisher(df):
    q = I.fisher_transform(df)
    return _b(q["buy"]), _b(q["sell"])


def t_cisd(df):
    q = I.cisd(df)
    return _b(q["bull"]), _b(q["bear"])


def t_ifvg(df):
    q = I.ifvg(df)
    return _b(q["retest_long"]), _b(q["retest_short"])


def t_ob_retest(df):
    q = I.smc(df, 5)
    return _b(q["ob_retest_long"]), _b(q["ob_retest_short"])


def t_woodies(df):
    q = I.woodies_cci(df)
    return _b(q["zlr_long"]), _b(q["zlr_short"])


def t_elder(df):
    q = I.elder_ray(df)
    bl, bs = _b(q["buy"]), _b(q["sell"])
    return bl & ~np.r_[False, bl[:-1]], bs & ~np.r_[False, bs[:-1]]


def t_ut_bot(df):
    q = I.ut_bot(df)
    return _b(q["buy"]), _b(q["sell"])


def t_squeeze(df):
    q = I.squeeze_momentum(df)
    f = _b(q["fired"])
    return f & _b(q["val"] > 0), f & _b(q["val"] < 0)


TRIGGERS = {
    "rsi_pull": (t_rsi_pull, "RSI range-shift pullback turn (40-80 / 20-60)", "pullback"),
    "wavetrend": (t_wavetrend, "WaveTrend cross on the pullback side of zero", "oscillator"),
    "qqe": (t_qqe, "QQE cross of its trailing line", "oscillator"),
    "laguerre": (t_laguerre, "Laguerre RSI turn out of 0.2 / 0.8", "oscillator"),
    "fisher": (t_fisher, "Fisher Transform trigger cross", "oscillator"),
    "cisd": (t_cisd, "CISD close", "structure"),
    "ifvg": (t_ifvg, "Inversion FVG retest", "structure"),
    "ob_retest": (t_ob_retest, "SMC order-block retest", "structure"),
    "woodies": (t_woodies, "Woodies CCI zero-line reject", "pullback"),
    "elder": (t_elder, "Elder Ray power turn", "pullback"),
    "ut_bot": (t_ut_bot, "UT Bot trailing-stop cross", "momentum"),
}

# which directions suit which trigger type (the "logic" of the pairing)
PAIRING = {
    "pullback": ["ema_stack", "supertrend", "ichimoku"],
    "oscillator": ["supertrend", "gaussian", "zero_lag"],
    "structure": ["smc", "ema_stack", "halftrend"],
    "momentum": ["ichimoku", "mcginley", "halftrend"],
}


# ---------------- regimes for breakout triggers ----------------
def r_adx(df):
    a = I.adx_di(df)
    return np.nan_to_num(a["adx"]) >= 20


def r_er(df):
    return np.abs(np.nan_to_num(I.efficiency_ratio(df))) >= 0.3


def r_chop(df):
    return np.nan_to_num(I.choppiness(df), nan=100.0) < 50


REGIMES = {"adx20": (r_adx, "ADX(14) >= 20"), "er03": (r_er, "|Kaufman ER| >= 0.3"),
           "chop50": (r_chop, "Choppiness < 50")}

EXITS = {
    "E1": ("stop 1.5 ATR, 2R", lambda dirv: {"stop": ("atr", 1.5), "target": ("R", 2.0)}),
    "E2": ("stop 2 ATR, 2.5 ATR trail, out on a direction flip",
           lambda dirv: {"stop": ("atr", 2.0), "target": None, "trail_atr": 2.5,
                         "exit_long": dirv < 0, "exit_short": dirv > 0}),
}


def _htf_dir(df, fn, rule="4h"):
    """Direction computed on completed `rule` bars, mapped to every base bar (known from the bar opening at the HTF
    bar's right edge on)."""
    g = df[["o", "h", "l", "c", "v"]].resample(rule, label="right", closed="left").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna()
    if len(g) < 5:
        return np.zeros(len(df))
    vals = np.asarray(fn(g), dtype=float)
    k = g.index.searchsorted(df.index, side="right") - 1
    return np.where(k >= 0, vals[np.clip(k, 0, None)], 0.0)


def make_combo(dname, tname, ename, htf=None):
    dfn, tfn = DIRECTIONS[dname][0], TRIGGERS[tname][0]
    plan_fn = EXITS[ename][1]

    def fn(df):
        dirv = _htf_dir(df, dfn, htf) if htf else dfn(df)
        lo, sh = tfn(df)
        L = lo & (dirv > 0)
        S = sh & (dirv < 0) & ~L
        return L, S, plan_fn(dirv)
    fn.__name__ = f"ib_{dname}_{tname}_{ename}" + (f"_htf{htf}" if htf else "")
    fn.__doc__ = (f"Indicator builder: direction = {DIRECTIONS[dname][1]}" + (f" on the {htf} chart" if htf else "") +
                  f"; entry = {TRIGGERS[tname][1]} in that direction; exit {EXITS[ename][0]}.")
    return fn


def make_regime_breakout(rname, ename):
    rfn = REGIMES[rname][0]
    plan_fn = EXITS[ename][1]

    def fn(df):
        lo, sh = t_squeeze(df)
        ok = rfn(df)
        c = cols(df)[3]
        dirv = np.sign(c - ema(c, 50))
        return lo & ok, sh & ok & ~(lo & ok), plan_fn(dirv)
    fn.__name__ = f"ib_squeeze_{rname}_{ename}"
    fn.__doc__ = (f"Indicator builder: LazyBear squeeze release in the momentum's direction, only while "
                  f"{REGIMES[rname][1]} (volatility regime + breakout); exit {EXITS[ename][0]} (E2's flip = the close "
                  f"crossing the EMA50).")
    return fn


def build():
    """dict IB### -> (name, fn, timeframes, meta)."""
    out = {}
    k = 1

    def add(name, fn, tfs, meta):
        nonlocal k
        out[f"IB{k:03d}"] = (name, fn, tfs, meta)
        k += 1
    for tname, (_, tdesc, ttype) in TRIGGERS.items():
        for dname in PAIRING[ttype]:
            for ename in EXITS:
                add(f"IB: {DIRECTIONS[dname][1]} + {tdesc}, {EXITS[ename][0]}", make_combo(dname, tname, ename),
                    ["5Min", "15Min", "1Hour", "4Hour"], {"direction": dname, "trigger": tname, "exit": ename})
    for rname in REGIMES:
        for ename in EXITS:
            add(f"IB: LazyBear squeeze release while {REGIMES[rname][1]}, {EXITS[ename][0]}",
                make_regime_breakout(rname, ename), ["5Min", "15Min", "1Hour", "4Hour"],
                {"regime": rname, "trigger": "squeeze", "exit": ename})
    for dname in ("supertrend", "ema_stack", "ichimoku"):
        for tname in ("cisd", "ifvg", "rsi_pull", "wavetrend"):
            add(f"IB: 4H {DIRECTIONS[dname][1]} + {TRIGGERS[tname][1]} on the entry chart, {EXITS['E1'][0]}",
                make_combo(dname, tname, "E1", "4h"), ["5Min", "15Min"],
                {"direction": dname, "htf": "4h", "trigger": tname, "exit": "E1"})
    return out


BUILT = build()
