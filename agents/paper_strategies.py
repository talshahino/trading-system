"""Strategies from academic papers (J. Finance, JFE, arXiv, working papers) and Quantpedia's tests,
coded from the published rules. Same contract as agents/rules_agent.py.

AP01 Tokyo-fix "W": long USD into the 09:55 JST fix, short USD after it  Krohn, Mueller, Whelan (JF 2024)
AP02 Long USD into the 14:15 ECB reference fix                          Krohn et al.; Breedon & Ranaldo
AP03 Gotobi days on USDJPY                                              arXiv 2301.13204
AP04 FOMC-cycle even weeks long                                         Cieslak, Morse, Vissing-Jorgensen (JF 2019)
AP05 Drift before the jobs report / ISM                                 Hu, Pan, Wang, Zhu (JFE 2022)
AP06 Reversal after an extreme 1-hour move in large US shares           Zawadowski, Andor, Kertesz
AP07 FX contrarian the morning after an abnormal day                    Caporale & Plastun (2021)
AP08 Oil momentum the morning after an abnormal day                     Caporale & Plastun (2021)
AP09 Following FX price jumps (Lee-Mykland jump test)                   Ficura (2019)
AP10 Gold short in the COMEX day session 08:20-13:30 NY                 Blose et al. (2018)
AP11 Bitcoin long 21:00-23:00 UTC                                       Quantpedia (Padysak & Vojtko)
AP12 Return since the last close -> last half hour                      Baltussen, Da, Lammers, Martens (JFE 2021)

Clock times are read on each bar's CLOSE time in the paper's own time zone (so daylight-saving shifts
are handled): "enter at 17:00" = a signal on the bar closing at 17:00, filled at the next bar's open.
None of the papers set stops (their tests are pure time windows); ours are a share of the daily ATR
and noted. Where a paper limits the instruments (only the pairs that passed its tests), so do we.
"""
import numpy as np
import pandas as pd

from agents.book_strategies import _days, _on_bars
from agents.seasonal_strategies import FOMC
from agents import ftmo_universe as U

NY, TKY, BER, LDN, PAR = "America/New_York", "Asia/Tokyo", "Europe/Berlin", "Europe/London", "Europe/Paris"
FOMC_ALL = sorted({"2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18",
                   "2024-11-07", "2024-12-18"} | set(FOMC))


def _z(n):
    return np.zeros(n, bool)


def _sym(df):
    return df.attrs.get("symbol") or ""


def _close_clock(df, tz):
    """Per bar: minutes of day, weekday and date of the bar's close time in `tz`."""
    step = df.index.to_series().diff().median()
    t = (df.index + step).tz_convert(tz)
    return (t.hour * 60 + t.minute).to_numpy(), t.dayofweek.to_numpy(), np.array([d.isoformat() for d in t.date]), t


def _dstop(df, k):
    """Stop levels k daily ATRs from the signal bar's close."""
    date, day, _, _ = _days(df)
    pc = day["c"].shift(1)
    tr = pd.concat([day["h"] - day["l"], (day["h"] - pc).abs(), (day["l"] - pc).abs()], axis=1).max(axis=1)
    da = _on_bars(date, tr.rolling(14).mean().shift(1))
    c = df["c"].to_numpy()
    return c - k * da, c + k * da


def _usd_dir(sym):
    """+1 if buying the symbol is buying USD, -1 if selling it is, 0 for non-USD instruments."""
    if U.CLASS.get(sym) != "fx":
        return 0
    if sym.startswith("USD"):
        return 1
    if sym.endswith("USD"):
        return -1
    return 0


def _hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------- AP01
def s_tokyo_fix(df):
    """Every trading day: long USD from 17:05 New York (just after the rollover) to the 09:55 Tokyo
    fix, then short USD from the fix to 02:00 New York. Stop 1 daily ATR."""
    d = _usd_dir(_sym(df))
    n = len(df)
    if not d:
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    ny_m, ny_dow, _, _ = _close_clock(df, NY)
    tk_m, _, _, _ = _close_clock(df, TKY)
    pre = (ny_m == _hm("17:05")) & np.isin(ny_dow, [6, 0, 1, 2, 3])
    fix = tk_m == _hm("09:55")
    post_end = ny_m == _hm("02:00")
    usd_long, usd_short = pre, fix
    L = usd_long if d > 0 else usd_short
    S = usd_short if d > 0 else usd_long
    sl, ss = _dstop(df, 1.0)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": None, "force_exit": fix | post_end}


# ---------------------------------------------------------------- AP02
def s_ecb_fix(df):
    """Monday-Friday: long USD from 02:00 New York to the 14:15 Berlin ECB fix. Stop 1 daily ATR."""
    d = _usd_dir(_sym(df))
    n = len(df)
    if not d:
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    ny_m, ny_dow, _, _ = _close_clock(df, NY)
    be_m, _, _, _ = _close_clock(df, BER)
    go = (ny_m == _hm("02:00")) & (ny_dow < 5)
    sl, ss = _dstop(df, 1.0)
    L, S = (go, _z(n)) if d > 0 else (_z(n), go)
    return L, S, {"stop": ("level", sl, ss), "target": None, "force_exit": be_m == _hm("14:15")}


# ---------------------------------------------------------------- AP03
def _gotobi(dates):
    """Gotobi days (5, 10, ..., 30 and the month's last day if the 30th does not exist), moved back to
    Friday when they fall on a weekend."""
    out = set()
    for d in pd.date_range(str(min(dates)), str(max(dates)), freq="D"):
        last = (d + pd.offsets.MonthEnd(0)).day
        if d.day in (5, 10, 15, 20, 25, 30) or (d.month == 2 and d.day == last):
            x = d
            while x.dayofweek >= 5:
                x -= pd.Timedelta(days=1)
            out.add(x.date().isoformat())
    return out


def s_gotobi(df):
    """USDJPY on Gotobi days (Tuesday-Friday): long 03:00 -> 09:55 Tokyo; then short to 12:00 Tokyo
    if the long was in profit at 09:00. Stop 0.75 daily ATR."""
    n = len(df)
    if _sym(df) != "USDJPY":
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    m, dow, date, _ = _close_clock(df, TKY)
    g = _gotobi(date)
    gd = np.array([x in g for x in date]) & np.isin(dow, [1, 2, 3, 4])
    c = df["c"].to_numpy()
    at3 = pd.Series(np.where(gd & (m == _hm("03:00")), c, np.nan)).groupby(date).transform("max").to_numpy()
    at9 = pd.Series(np.where(gd & (m == _hm("09:00")), c, np.nan)).groupby(date).transform("max").to_numpy()
    L = gd & (m == _hm("03:00"))
    S = gd & (m == _hm("09:55")) & (at9 > at3)
    sl, ss = _dstop(df, 0.75)
    return L, S, {"stop": ("level", sl, ss), "target": None,
                  "force_exit": gd & ((m == _hm("09:55")) | (m == _hm("12:00")))}


# ---------------------------------------------------------------- AP04
def s_fomc_cycle(df):
    """US trading days counted from each FOMC decision (day 0): long in the "even weeks" -1..3, 9..13,
    19..23, 29..33 - bought at the 16:00 New York close before each block, sold at the 16:00 close of
    its last day. Closed on Fridays and bought back after the weekend (FTMO weekend rule). US indices
    only. Stop 2 daily ATRs."""
    n = len(df)
    if _sym(df) not in ("US500", "US100", "US30", "US2000"):
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    m, dow, date, _ = _close_clock(df, NY)
    days = sorted(set(d for d, w in zip(date, dow) if w < 5))
    pos = {d: i for i, d in enumerate(days)}
    f_idx = sorted(pos[f] for f in FOMC_ALL if f in pos)
    blocks = [(-1, 3), (9, 13), (19, 23), (29, 33)]
    inb = {}
    for i in range(len(days)):
        ok = False
        for k, f in enumerate(f_idx):
            nxt = f_idx[k + 1] if k + 1 < len(f_idx) else 10**9
            rel = i - f
            if rel >= -1 and i < nxt - 1:
                ok = any(a <= rel <= b for a, b in blocks)
                break
            if rel < -1 and i == f - 1:
                ok = True
                break
        inb[days[i]] = ok
    nxt_day = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    in_next = np.array([inb.get(nxt_day.get(d, ""), False) for d in date])
    in_today = np.array([inb.get(d, False) for d in date])
    close16 = m == _hm("16:00")
    L = close16 & in_next & (dow < 4)
    first_of_day = np.r_[True, date[1:] != date[:-1]]
    L |= first_of_day & in_today & (dow == 0)
    fe = close16 & ((~in_next) | (dow == 4))
    sl, _ = _dstop(df, 2.0)
    return L, _z(n), {"stop": ("level", sl, sl), "target": None, "force_exit": fe}


# ---------------------------------------------------------------- AP05
def _first_friday(d):
    x = d.replace(day=1)
    while x.dayofweek != 4:
        x += pd.Timedelta(days=1)
    return x


def _first_bday(d):
    x = d.replace(day=1)
    while x.dayofweek >= 5:
        x += pd.Timedelta(days=1)
    return x


def s_pre_announcement(df):
    """US indices: long from the 16:00 New York close before the jobs report (first Friday, 08:30) and
    the ISM manufacturing index (first business day, 10:00; skipped on Mondays) until shortly before
    the release (the bar closing at least 5 minutes before it). Stop 1 daily ATR."""
    n = len(df)
    if _sym(df) not in ("US500", "US100", "US30", "US2000"):
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    m, dow, date, t = _close_clock(df, NY)
    step = int(df.index.to_series().diff().median() / pd.Timedelta(minutes=1))
    months = pd.date_range(pd.Timestamp(str(min(date))) - pd.offsets.MonthBegin(1), pd.Timestamp(str(max(date))), freq="MS")
    ev = {}
    for mo in months:
        ev[_first_friday(mo).date().isoformat()] = _hm("08:30")
        b = _first_bday(mo)
        if b.dayofweek != 0:
            ev.setdefault(b.date().isoformat(), _hm("10:00"))
    fomc = set(FOMC_ALL)
    days = list(dict.fromkeys(date))
    nxt = {days[i]: days[i + 1] for i in range(len(days) - 1)}
    nd = np.array([nxt.get(d, "") for d in date])
    L = (m == _hm("16:00")) & np.array([x in ev and x not in fomc for x in nd])
    lim = np.array([ev.get(d, -1) for d in date])
    fe = (lim > 0) & (m <= lim - 5) & (m + step > lim - 5)
    sl, _ = _dstop(df, 1.0)
    return L, _z(n), {"stop": ("level", sl, sl), "target": None, "force_exit": fe}


# ---------------------------------------------------------------- AP06
def s_extreme_reversal(df):
    """Large US shares, 09:35-15:00 New York: a move of 4%+ within 10-60 minutes that is also 8x the
    usual move for that clock time (last 60 days) -> fade it at the next open; out after 60 minutes
    or at 15:55. One event per day. Stop 4 ATR."""
    n = len(df)
    if U.CLASS.get(_sym(df)) != "stock" or "USD" not in U.CURRENCIES.get(_sym(df), []):
        return _z(n), _z(n), {"stop": ("atr", 4.0), "target": None}
    m, dow, date, _ = _close_clock(df, NY)
    step = max(1, int(df.index.to_series().diff().median() / pd.Timedelta(minutes=1)))
    c = df["c"].to_numpy()
    sess = (m >= _hm("09:35")) & (m <= _hm("15:00"))
    L, S = _z(n), _z(n)
    used = set()
    max_k = max(2, 60 // step)
    for k in range(2 if step <= 5 else 1, max_k + 1):
        r = c / np.r_[np.full(k, np.nan), c[:-k]] - 1
        same_day = np.r_[np.zeros(k, bool), date[k:] == date[:-k]]
        r = np.where(same_day, r, np.nan)
        v = (pd.DataFrame({"m": m, "a": np.abs(r)}).groupby("m")["a"]
             .transform(lambda s: s.shift(1).rolling(60, min_periods=20).mean()).to_numpy())
        dn = sess & (r <= -0.04) & (np.abs(r) >= 8 * v)
        up = sess & (r >= 0.04) & (r >= 8 * v)
        for i in np.flatnonzero(dn | up):
            if date[i] in used:
                continue
            used.add(date[i])
            if dn[i]:
                L[i] = True
            else:
                S[i] = True
    return L, S & ~L, {"stop": ("atr", 4.0), "target": None, "max_bars": max(1, 60 // step) + 1,
                       "force_exit": m == _hm("15:55") if step <= 5 else m == _hm("16:00") - step}


# ---------------------------------------------------------------- AP07 / AP08
def _server_days(df):
    """FTMO server day = the 17:00 New York day (server midnight). Per bar: day key, first/last flag."""
    t = df.index.tz_convert(NY) + pd.Timedelta(hours=7)
    key = np.array([d.isoformat() for d in t.date])
    first = np.r_[True, key[1:] != key[:-1]]
    last = np.r_[key[1:] != key[:-1], True]
    step = df.index.to_series().diff().median()
    tc = (df.index + step).tz_convert(NY) + pd.Timedelta(hours=7)
    return key, first, last, (tc.hour * 60 + tc.minute).to_numpy()


def _abnormal(df, k=2.0, n=30):
    key, first, last, m = _server_days(df)
    o, c = df["o"].to_numpy(), df["c"].to_numpy()
    g = pd.DataFrame({"k": key, "o": o, "c": c}).groupby("k", sort=False)
    r = g["c"].last() / g["o"].first() - 1
    mu, sd = r.rolling(n).mean().shift(1), r.rolling(n).std().shift(1)
    pos = pd.Series(key).map(r > mu + k * sd).fillna(False).to_numpy(bool)
    neg = pd.Series(key).map(r < mu - k * sd).fillna(False).to_numpy(bool)
    return key, first, last, m, pos & last, neg & last


# exit server-time (minutes, -1 = end of day) per pair and direction; only pairs that passed the paper
AP07_RULES = {("EURUSD", "pos"): _hm("09:00"), ("AUDUSD", "pos"): _hm("09:00"), ("USDJPY", "pos"): _hm("07:00"),
              ("EURJPY", "pos"): -1, ("AUDUSD", "neg"): _hm("08:00"), ("USDJPY", "neg"): -1}


def s_fx_abnormal(df):
    """FX: after an abnormal server day (open-to-close beyond mean +/- 2 sd of the last 30 days) fade it
    from the next day's first bar, out at the pair's tested hour (server time). Only the pair /
    direction pairs that passed the paper's t-test. Stop 0.5 daily ATR."""
    sym = _sym(df)
    n = len(df)
    key, first, last, m, pos, neg = _abnormal(df)
    L, S = _z(n), _z(n)
    xl, xs = _z(n), _z(n)
    nl = np.r_[last[1:], False]           # exit "at the end of the day": the bar before the day's last
    if (sym, "pos") in AP07_RULES:
        S = pos
        x = AP07_RULES[(sym, "pos")]
        xs = nl if x < 0 else (m == x)
    if (sym, "neg") in AP07_RULES:
        L = neg
        x = AP07_RULES[(sym, "neg")]
        xl = nl if x < 0 else (m == x)
    sl, ss = _dstop(df, 0.5)
    return L, S & ~L, {"stop": ("level", sl, ss), "target": None, "exit_long": xl, "exit_short": xs}


def s_oil_abnormal(df):
    """Oil: after an abnormal server day, follow it from the next day's first bar; longs out at 09:00
    server time, shorts at 10:00. Stop 0.5 daily ATR."""
    n = len(df)
    if _sym(df) not in ("WTI", "BRENT"):
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    key, first, last, m, pos, neg = _abnormal(df)
    sl, ss = _dstop(df, 0.5)
    return pos, neg & ~pos, {"stop": ("level", sl, ss), "target": None,
                             "exit_long": m == _hm("09:00"), "exit_short": m == _hm("10:00")}


# ---------------------------------------------------------------- AP09
AP09_RULES = {("EURUSD", "15Min"): (0.90, 9), ("GBPUSD", "15Min"): (0.99, 12), ("USDCHF", "15Min"): (0.90, 6),
              ("EURUSD", "1Hour"): (0.90, 2)}


def s_fx_jumps(df):
    """Lee-Mykland jump test on log returns (bipower variation over K bars, n = bars per day); a jump up
    -> long, down -> short, held h bars (the paper's out-of-sample pairs and holding periods). Stop 3 ATR."""
    sym, tf = _sym(df), df.attrs.get("tf", "")
    n = len(df)
    if (sym, tf) not in AP09_RULES:
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    p, h = AP09_RULES[(sym, tf)]
    K, nday = (156, 96) if tf == "15Min" else (78, 24)
    c = df["c"].to_numpy()
    r = np.r_[np.nan, np.diff(np.log(c))]
    bp = pd.Series(np.abs(r) * np.abs(np.r_[np.nan, r[:-1]])).shift(1).rolling(K - 2).sum().to_numpy() / (K - 2)
    Lst = r / np.sqrt(bp)
    cc = np.sqrt(2 / np.pi)
    ln = np.log(nday)
    Cn = np.sqrt(2 * ln) / cc - (np.log(np.pi) + np.log(ln)) / (2 * cc * np.sqrt(2 * ln))
    Sn = 1 / (cc * np.sqrt(2 * ln))
    xi = (np.abs(Lst) - Cn) / Sn
    beta = -np.log(-np.log(p))
    jump = xi > beta
    return jump & (r > 0), jump & (r < 0), {"stop": ("atr", 3.0), "target": None, "max_bars": h}


# ---------------------------------------------------------------- AP10
def s_gold_day_short(df):
    """Gold: short at 08:20 New York, cover at 13:30 New York (the COMEX day session), Monday-Friday.
    Stop 0.75 daily ATR."""
    n = len(df)
    if _sym(df) != "XAUUSD":
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    m, dow, _, _ = _close_clock(df, NY)
    step = int(df.index.to_series().diff().median() / pd.Timedelta(minutes=1))
    go = (m == _hm("08:20") if step <= 5 else m == _hm("08:15")) & (dow < 5)
    _, ss = _dstop(df, 0.75)
    return _z(n), go, {"stop": ("level", ss, ss), "target": None, "force_exit": m == _hm("13:30")}


# ---------------------------------------------------------------- AP11
def s_btc_evening(df):
    """Bitcoin (and ether as a check): long 21:00 -> 23:00 UTC every day. Stop 0.5 daily ATR."""
    n = len(df)
    if _sym(df) not in ("BTCUSD", "ETHUSD"):
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    m, _, _, _ = _close_clock(df, "UTC")
    sl, _ = _dstop(df, 0.5)
    return m == _hm("21:00"), _z(n), {"stop": ("level", sl, sl), "target": None, "force_exit": m == _hm("23:00")}


# ---------------------------------------------------------------- AP12
CASH_CLOSE = {"US500": (NY, "16:00"), "US100": (NY, "16:00"), "US30": (NY, "16:00"), "US2000": (NY, "16:00"),
              "GER40": (BER, "17:30"), "EU50": (PAR, "17:30"), "FRA40": (PAR, "17:30"), "UK100": (LDN, "16:30"),
              "JP225": (TKY, "15:30")}


def s_rod_last_half_hour(df):
    """Index cash close T: at T-30 minutes, the return since the previous day's T decides - up buys,
    down sells - for the last half hour; out at T. Stop 3 ATR."""
    sym = _sym(df)
    n = len(df)
    if sym not in CASH_CLOSE:
        return _z(n), _z(n), {"stop": ("atr", 3.0), "target": None}
    tz, tclose = CASH_CLOSE[sym]
    m, dow, date, t = _close_clock(df, tz)
    T = _hm(tclose)
    if sym == "JP225":
        T = np.where(np.array(date) < "2024-11-05", _hm("15:00"), _hm("15:30"))
    else:
        T = np.full(n, T)
    c = df["c"].to_numpy()
    at_T = m == T
    ref = pd.Series(np.where(at_T, c, np.nan)).ffill().shift(1)
    ref = pd.Series(np.where(at_T, np.nan, ref.to_numpy())).to_numpy()
    sig = (m == T - 30) & (dow < 5)
    r = c / ref - 1
    return sig & (r > 0), sig & (r < 0), {"stop": ("atr", 3.0), "target": None, "force_exit": at_T}


# id: (name, fn, timeframes)
PAPERS = {
    "AP01": ("Tokyo-fix W: long USD into the 09:55 fix, short after (Krohn et al., JF 2024)", s_tokyo_fix, ["5Min"]),
    "AP02": ("Long USD into the 14:15 ECB fix (Krohn et al.; Breedon & Ranaldo)", s_ecb_fix, ["5Min"]),
    "AP03": ("Gotobi days on USDJPY (arXiv 2023)", s_gotobi, ["5Min"]),
    "AP04": ("FOMC-cycle even weeks long (Cieslak et al., JF 2019)", s_fomc_cycle, ["1Hour"]),
    "AP05": ("Drift before the jobs report / ISM (Hu et al., JFE 2022)", s_pre_announcement, ["15Min"]),
    "AP06": ("Reversal after an extreme 1-hour move, US shares (Zawadowski et al.)", s_extreme_reversal, ["5Min"]),
    "AP07": ("FX contrarian the morning after an abnormal day (Caporale & Plastun)", s_fx_abnormal, ["1Hour"]),
    "AP08": ("Oil momentum the morning after an abnormal day (Caporale & Plastun)", s_oil_abnormal, ["1Hour"]),
    "AP09": ("Following FX price jumps, Lee-Mykland test (Ficura 2019)", s_fx_jumps, ["15Min", "1Hour"]),
    "AP10": ("Gold short in the COMEX day session (Blose et al. 2018)", s_gold_day_short, ["5Min"]),
    "AP11": ("Bitcoin long 21:00-23:00 UTC (Quantpedia)", s_btc_evening, ["1Hour"]),
    "AP12": ("Return since the last close -> last half hour (Baltussen et al., JFE 2021)", s_rod_last_half_hour, ["15Min"]),
}
