"""Scalping bot, step 2 (user, 06.11.2026): entries that carry information beyond the last minutes' move,
on the instruments and hours that have room (step 1: the indices at their open move 15-36 x the cost in 30
minutes), with the market's direction and the latest economic news (user: "כיווניות שוק והודעות").

Entries (decided on a closed 1-minute bar, entered at the next open):
  ORB15 / ORB30   first close beyond the first 15 / 30 minutes' range after the session open, within 2 hours
  GAP             5 minutes after the open, the gap from the previous session's close (> 0.25 daily ATR):
                  follow it (gap_follow) or fade it (gap_fade)
  PDX             first break of yesterday's high / low within 2 hours of the open: follow (pdx_follow) or the
                  close back inside after it (pdx_fade)
  NEWS            a high-impact release for the instrument's currency (FTMO restricted events): the reaction
                  from 1 minute before to 2 minutes after; entered 3 minutes after (FTMO forbids trading 2
                  minutes either side) in the reaction's direction (news_follow) or against it (news_fade),
                  only when the reaction is at least 3 x the cost
Market direction filter, each entry three ways: all / with the daily trend / against it (daily trend =
yesterday's close above / below the 20-day average of daily closes).
Judged as scalp_map.py: the signed move after H = 15, 30, 60, 120 minutes in units of THAT minute's real cost
(spread + commission), net of one cost; cut at forced exits. TRAIN 2024-01-01..2025-04-30, VALID after.

  python scalp_stage2.py --symbols US100,GER40
Writes output/scalp_s2_<tag>.json.
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd

import config
from agents import news_calendar as NC
from agents import poc_agent as P
from agents import rules_agent as R
import run_poc as RP
import run_rules as RR
from scalp_map import TRAIN_FROM, VALID_FROM, comm_px, st

HS = [15, 30, 60, 120]
US = ("America/New_York", 570)
OPENS = {"US100": [US], "US30": [US], "US500": [US], "US2000": [US], "NVDA": [US], "TSLA": [US], "AAPL": [US],
         "GER40": [("Europe/Berlin", 540)], "EU50": [("Europe/Berlin", 540)], "FRA40": [("Europe/Paris", 540)],
         "UK100": [("Europe/London", 480)], "JP225": [("Asia/Tokyo", 540)],
         "XAUUSD": [("Europe/London", 480), US], "XAGUSD": [("Europe/London", 480), US], "WTI": [US],
         "EURUSD": [("Europe/London", 480), US], "GBPUSD": [("Europe/London", 480), US],
         "USDJPY": [("Asia/Tokyo", 540), US], "BTCUSD": [US]}
US_IDX = {"US100", "US30", "US500", "US2000"}


def news_times(sym):
    """high-impact releases that move this instrument: the news calendar until it ends (04.2025), then FTMO's
    own restricted-event list (it names the instruments: 'USD + US Indices + XAUUSD + DXY')"""
    ev, real_end = NC.load()
    cur = NC.CURRENCIES.get(sym) or (["USD"] if sym in ("NVDA", "TSLA", "AAPL") else [])
    a = ev[(ev["currency"].isin(cur)) & (ev["time"] <= real_end) & ~ev["event"].str.contains(r"\(rule\)")]["time"]
    out = list(a)
    try:
        hist = json.load(open(RR.Path(RR.__file__).parent / "data" / "ftmo_calendar" / "history.json", encoding="utf-8"))
        for i in hist:
            if not i.get("restriction"):
                continue
            t = pd.Timestamp(i["date"]).tz_convert("UTC")
            if t <= real_end:
                continue
            ins = i.get("instrument", "")
            toks = set(re.split(r"[\s+,]+", ins))
            if (toks & set(cur)) or (sym in toks) or ("US Indices" in ins and sym in US_IDX):
                out.append(t)
    except Exception as e:  # noqa: BLE001
        print("ftmo calendar:", e)
    return pd.DatetimeIndex(sorted(set(out)))


def first_true(mask, day):
    return mask & (pd.Series(mask.astype(int)).groupby(day).cumsum().to_numpy() == 1)


def signals(df, sym, cost):
    o, h, l, c = (df[x].to_numpy(float) for x in "ohlc")
    n = len(c)
    idx = df.index
    # daily trend (UTC days; yesterday's close vs its 20-day average) and daily ATR
    dc = pd.Series(c, index=idx).resample("1D").last().dropna()
    dh = pd.Series(h, index=idx).resample("1D").max().reindex(dc.index)
    dl = pd.Series(l, index=idx).resample("1D").min().reindex(dc.index)
    trend_d = np.sign(dc - dc.rolling(20).mean()).shift(1)
    datr = pd.Series(R.atr(dh.to_numpy(), dl.to_numpy(), dc.to_numpy(), 14), index=dc.index).shift(1)
    day_utc = idx.normalize()
    trend = trend_d.reindex(day_utc).to_numpy()
    atr_d = datr.reindex(day_utc).to_numpy()
    S = {}

    def add(name, L, Sh):
        S.setdefault(name, [np.zeros(n, bool), np.zeros(n, bool)])
        S[name][0] |= L
        S[name][1] |= Sh

    for tz, om in OPENS.get(sym, [US]):
        t = idx.tz_convert(tz)
        m = np.asarray(t.hour * 60 + t.minute)
        day = pd.Series(t.date).to_numpy()
        for rng_min, nm in ((15, "orb15"), (30, "orb30")):
            inr = (m >= om) & (m < om + rng_min)
            s = pd.DataFrame({"d": day, "h": np.where(inr, h, np.nan), "l": np.where(inr, l, np.nan)})
            hi, lo = s.groupby("d")["h"].transform("max").to_numpy(), s.groupby("d")["l"].transform("min").to_numpy()
            win = (m >= om + rng_min) & (m < om + 120) & np.isfinite(hi)
            add(nm, first_true(win & (c > hi), day), first_true(win & (c < lo), day))
        # gap: previous session's last close vs the open, judged 5 minutes after the open
        sess = (m >= om) & (m < om + 390)
        last_c = pd.Series(np.where(sess, c, np.nan)).groupby(day).transform("last").to_numpy()
        prev_c = pd.Series(last_c).groupby(day).first()
        prev_c = pd.Series(day).map(prev_c.shift(1)).to_numpy(float)
        open_px = pd.Series(np.where(m == om, o, np.nan)).groupby(day).transform("max").to_numpy()
        gap = open_px - prev_c
        at = m == om + 5
        big = np.abs(gap) > 0.25 * atr_d
        add("gap_follow", at & big & (gap > 0), at & big & (gap < 0))
        add("gap_fade", at & big & (gap < 0), at & big & (gap > 0))
        # yesterday's high / low (UTC day), first break within 2 hours of the open
        ph, pl = pd.Series(day_utc).map(dh.shift(1)).to_numpy(float), pd.Series(day_utc).map(dl.shift(1)).to_numpy(float)
        w2 = (m >= om) & (m < om + 120)
        add("pdx_follow", first_true(w2 & (c > ph), day), first_true(w2 & (c < pl), day))
        add("pdx_fade", first_true(w2 & (l < pl) & (c > pl), day), first_true(w2 & (h > ph) & (c < ph), day))
    # news reaction
    nt = news_times(sym)
    pos = idx.searchsorted(nt)
    L, Sh = np.zeros(n, bool), np.zeros(n, bool)
    Lf, Sf = np.zeros(n, bool), np.zeros(n, bool)
    for k, ti in zip(pos, nt):
        if k < 1 or k + 3 >= n or idx[k] != ti:
            continue
        b0, b1 = k - 1, k + 2          # close a minute before -> close 2 minutes after (the decision bar)
        if (idx[b1] - idx[b0]) > pd.Timedelta(minutes=4):
            continue
        mv = c[b1] - c[b0]
        if not np.isfinite(cost[b1 + 1]) or abs(mv) < 3 * cost[b1 + 1]:
            continue
        (L if mv > 0 else Sh)[b1] = True
        (Sf if mv > 0 else Lf)[b1] = True
    add("news_follow", L, Sh)
    add("news_fade", Lf, Sf)
    out = {}
    for nm, (a, b) in S.items():
        out[nm + "|all"] = (a, b)
        out[nm + "|with"] = (a & (trend > 0), b & (trend < 0))
        out[nm + "|against"] = (a & (trend < 0), b & (trend > 0))
    return out, len(nt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="")
    a = ap.parse_args()
    end = a.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tag = a.symbols.replace(",", "_").replace("/", "-")[:60]
    res = []
    for s in [x for x in a.symbols.split(",") if x]:
        t0 = time.time()
        try:
            df = P.bars(s, "1Min", max(a.start, RP.MINUTE_START), end)
            sp = P.spread_1m(s, a.start, end)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: NO DATA {str(e)[-120:]}")
            continue
        if df is None or df.empty or sp is None or len(sp) == 0:
            print(f"{s}: no data or no real spread - skipped")
            continue
        o, c = df["o"].to_numpy(float), df["c"].to_numpy(float)
        n = len(c)
        cost = np.abs(sp.sort_index().reindex(df.index, method="ffill").to_numpy(float)) + comm_px(s, c)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, np.nan)
        stop = RR.break_mask(df, "1Min")
        sm = RR.session_mask(df, s, "1Min")
        if sm is not None:
            stop = stop | np.asarray(sm, bool)
        nm_, _ = RR.news_masks(df, s, "1Min")
        stop = stop | np.asarray(nm_, bool)
        cs = np.r_[0, np.cumsum(stop)]
        tm = df.index
        trn, val = (tm >= TRAIN_FROM) & (tm < VALID_FROM), tm >= VALID_FROM
        sig, n_news = signals(df, s, cost)
        row = {"symbol": s, "news_events": n_news, "entries": {}}
        for name, (L, Sh) in sig.items():
            idx = np.flatnonzero(L | Sh)
            idx = idx[idx + 2 < n]
            if not len(idx):
                continue
            d = np.where(L[idx], 1.0, -1.0)
            cell = {}
            for hh in HS:
                ok = (idx + hh < n)
                j, dj = idx[ok], d[ok]
                ok2 = (cs[j + hh] - cs[j + 1]) == 0
                j, dj = j[ok2], dj[ok2]
                e = o[j + 1]
                g = dj * (c[j + hh] - e) / cost[j + 1] - 1
                for per, pm in (("train", trn), ("valid", val)):
                    k = pm[j]
                    cell.setdefault(per, {})[f"H{hh}"] = st(g[k])
            row["entries"][name] = cell
        res.append(row)
        print(f"{s}: {n:,} bars, {n_news} news, {time.time() - t0:.0f}s", flush=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / f"scalp_s2_{tag}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("symbols:", len(res))


if __name__ == "__main__":
    main()
