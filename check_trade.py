"""One hand-drawn trade walked on 1-minute data (user's examples): from a time, a limit entry, stop and target.
  python check_trade.py --symbol US30 --from "2024-03-06 18:30" --dir 1 --entry 38611.5 --stop 38555.7 --target 38842
Prints when it filled and how it ended."""
import argparse

import pandas as pd

from agents import poc_agent as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--from", dest="frm", required=True, help="UTC")
    ap.add_argument("--dir", type=int, required=True)
    ap.add_argument("--entry", type=float, required=True)
    ap.add_argument("--stop", type=float, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--days", type=int, default=10)
    a = ap.parse_args()
    t0 = pd.Timestamp(a.frm, tz="UTC")
    df = P.bars(a.symbol, "1Min", t0.strftime("%Y-%m-%d"), (t0 + pd.Timedelta(days=a.days)).strftime("%Y-%m-%d"))
    df = df[df.index >= t0]
    d, fill = a.dir, None
    for t, r in df.iterrows():
        if fill is None:
            if (d > 0 and r.l <= a.entry) or (d < 0 and r.h >= a.entry):
                fill = t
                print("filled", t, "(Israel", t.tz_convert("Asia/Jerusalem"), ")")
            else:
                continue
        hit_s = (d > 0 and r.l <= a.stop) or (d < 0 and r.h >= a.stop)
        hit_t = (d > 0 and r.h >= a.target) or (d < 0 and r.l <= a.target)
        if hit_s or hit_t:
            print("STOP" if hit_s else "TARGET", t, "(Israel", t.tz_convert("Asia/Jerusalem"), ")", "both in one minute" if hit_s and hit_t else "")
            seg = df[(df.index >= fill) & (df.index <= t)]
            print("lowest", seg.l.min(), "highest", seg.h.max())
            return
    print("open at end of data; last", df.c.iloc[-1], "low", df[df.index >= fill].l.min() if fill is not None else None,
          "high", df[df.index >= fill].h.max() if fill is not None else None)


if __name__ == "__main__":
    main()
