"""Synthetic news + minute bars, used ONLY to prove the pipeline runs end to end
without API keys. A small reaction to news is planted on purpose so the report has
something to find. These numbers say nothing about real markets."""
import numpy as np
import pandas as pd

POS = ["{s} beats Q3 estimates, raises guidance", "{s} wins $2B government contract",
       "Analyst upgrades {s} to buy", "FDA approves {s} lead drug", "{s} announces $1B buyback"]
NEG = ["{s} misses revenue estimates, cuts outlook", "{s} faces SEC investigation",
       "Analyst downgrades {s} to sell", "{s} announces stock offering", "{s} recalls flagship product"]
NEUTRAL = ["{s} to present at industry conference", "{s} schedules earnings call",
           "{s} CEO interviewed on business TV", "{s} files routine annual report"]


def make_demo(symbols=None, days=90, seed=7):
    rng = np.random.default_rng(seed)
    symbols = symbols or [f"DEMO{i:02d}" for i in range(20)]
    sessions = pd.bdate_range("2026-03-02", periods=days)
    n = days * 390
    bars, news, caps = {}, [], {}
    nid = 0
    for s in symbols:
        # Each demo stock gets a size: small caps trade less, move more, and (planted)
        # react more strongly to news.
        cap = 10 ** rng.uniform(8, 12)
        caps[s] = cap
        smallness = 1 - (np.log10(cap) - 8) / 4          # 1 = tiny, 0 = mega
        minute_vol = 0.0005 + 0.0015 * smallness
        fast = 0.002 + 0.008 * smallness                 # 20-100 bps over 15 minutes
        slow = 0.015                                     # 1.5% drift over the next 3 sessions
        dollar_per_min = cap * 0.004 / 390
        idx = pd.DatetimeIndex(np.concatenate([
            pd.date_range(pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta("9h30min"),
                          periods=390, freq="min").tz_convert("UTC").values for d in sessions]), tz="UTC")
        drift = np.zeros(n)
        for day in range(days):
            for _ in range(rng.poisson(1.2)):
                k = day * 390 + int(rng.integers(5, 330))
                kind = rng.choice(["pos", "neg", "neu"], p=[0.35, 0.35, 0.3])
                tmpl = rng.choice({"pos": POS, "neg": NEG, "neu": NEUTRAL}[kind])
                true = {"pos": 1, "neg": -1, "neu": 0}[kind]
                # Planted effects: a fast one (minutes, bigger in small caps) for every
                # directional item, and a slow one (days) only for earnings-type news.
                if true and rng.random() < 0.7:
                    drift[k + 1:k + 16] += true * fast / 15
                if true and "estimates" in tmpl and rng.random() < 0.7:
                    drift[k + 16:k + 16 + 3 * 390] += true * slow / (3 * 390)
                t = idx[k] + pd.Timedelta(seconds=int(rng.integers(0, 59)))
                news.append({"id": nid, "time": t, "headline": tmpl.format(s=s), "summary": "",
                             "source": "demo", "symbols": [s]})
                nid += 1
        rets = rng.normal(0, minute_vol, n) + drift
        # Overnight gaps between sessions.
        rets[390::390] += rng.normal(0, 0.006, days - 1)
        close = (5 + rng.random() * 150) * np.exp(np.cumsum(rets))
        opn = np.concatenate([[close[0]], close[:-1]])
        bars[s] = pd.DataFrame({"o": opn, "h": np.maximum(opn, close) * 1.0002,
                                "l": np.minimum(opn, close) * 0.9998, "c": close,
                                "v": (dollar_per_min / close * rng.uniform(0.3, 1.7, n)).astype(int) + 1},
                               index=idx)
    return symbols, pd.DataFrame(news).sort_values("time").reset_index(drop=True), bars, caps
