"""Trend-following track: a conservative, well-documented base strategy on liquid ETFs.

No searching here. The rules are fixed in advance from decades of published research,
so there is nothing to overfit and no luck test is needed:

  trend      Each ETF is held only while its price is above its 200-day average (checked at
             each month end), weighted by inverse volatility; the rest sits in short-term
             Treasuries (SHV).
  trend_vt   The same, scaled to a target volatility of 8% a year (never more than 100%
             invested), to keep drawdowns inside the 10% account stop.
  momentum   The 3 ETFs with the best 12-month return (skipping the last month) that are
             also positive; otherwise short-term Treasuries.
  Benchmarks buy-and-hold S&P 500 (SPY) and 60/40 (SPY/IEF).

Monthly rebalance at the close of the last trading day, 5 bps cost per side.

  python run_trend.py --start 2016-01-01 --end 2026-10-28
  python run_trend.py --demo
"""
import argparse
import html
import math

import numpy as np
import pandas as pd

import config

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC"]
CASH = "SHV"
COST = 0.0005          # per side
TARGET_VOL = 0.08
NAMES_HE = {"trend": "מעקב מגמה (ממוצע 200 יום)", "trend_vt": "מעקב מגמה + יעד תנודתיות 8%",
            "momentum": "מומנטום: 3 החזקים", "spy": "S&P 500 קנה והחזק (השוואה)",
            "6040": "60/40 מניות/אג״ח (השוואה)"}


def load_prices(start, end, demo=False):
    syms = UNIVERSE + [CASH]
    if demo:
        rng = np.random.default_rng(1)
        idx = pd.bdate_range(start, end)
        out = {}
        for i, s in enumerate(syms):
            drift = 0.0001 if s == CASH else rng.uniform(0.0001, 0.0005)
            vol = 0.0005 if s == CASH else rng.uniform(0.006, 0.018)
            regime = np.sin(np.arange(len(idx)) / rng.uniform(150, 400)) * 0.0008   # planted trends
            out[s] = 100 * np.exp(np.cumsum(drift + regime + rng.normal(0, vol, len(idx))))
        return pd.DataFrame(out, index=idx)
    from agents import market_data_agent
    frames = {}
    for s in syms:
        b = market_data_agent.get_daily_bars(s, start, end)
        if b is not None and not b.empty:
            frames[s] = b["c"]
            print(f"  {s}: {len(b)} days")
    return pd.DataFrame(frames).sort_index().ffill()


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return set(s.groupby([idx.year, idx.month]).max())


def weights_trend(px, rets, t):
    hist = px.loc[:t]
    if len(hist) < 200:
        return None
    sma = hist[UNIVERSE].iloc[-200:].mean()
    last = hist[UNIVERSE].iloc[-1]
    vol = rets.loc[:t, UNIVERSE].iloc[-60:].std()
    inv = 1 / vol.replace(0, np.nan)
    base = inv / inv.sum()
    w = base.where(last > sma, 0.0).fillna(0.0)
    w[CASH] = 1 - w.sum()
    return w


def weights_momentum(px, rets, t, top=3):
    hist = px.loc[:t]
    if len(hist) < 252:
        return None
    m = hist[UNIVERSE].iloc[-21] / hist[UNIVERSE].iloc[-252] - 1      # 12-1 month return
    pick = m[m > 0].sort_values(ascending=False).head(top)
    w = pd.Series(0.0, index=UNIVERSE + [CASH])
    for s in pick.index:
        w[s] = 1 / top
    w[CASH] = 1 - w.sum()
    return w


def simulate(px, kind):
    rets = px.pct_change().fillna(0.0)
    cols = UNIVERSE + [CASH]
    me = month_ends(px.index)
    w = pd.Series(0.0, index=cols)
    w[CASH] = 1.0
    started = False
    daily = []
    for i, t in enumerate(px.index):
        r = float((w * rets.loc[t, cols]).sum()) if i else 0.0
        # Weights drift with prices between rebalances.
        if i:
            grown = w * (1 + rets.loc[t, cols])
            w = grown / grown.sum() if grown.sum() > 0 else w
        cost = 0.0
        if t in me:
            if kind == "spy":
                nw = pd.Series(0.0, index=cols); nw["SPY"] = 1.0
            elif kind == "6040":
                nw = pd.Series(0.0, index=cols); nw["SPY"] = 0.6; nw["IEF"] = 0.4
            elif kind == "momentum":
                nw = weights_momentum(px, rets, t)
            else:
                nw = weights_trend(px, rets, t)
                if nw is not None and kind == "trend_vt":
                    # Scale to target vol using the portfolio's recent realized volatility.
                    port = (rets.loc[:t, cols].iloc[-60:] * nw).sum(axis=1)
                    pv = port.std() * math.sqrt(252)
                    k = min(1.0, TARGET_VOL / pv) if pv > 0 else 1.0
                    risky = nw.drop(CASH) * k
                    nw = pd.concat([risky, pd.Series({CASH: 1 - risky.sum()})])
            if nw is not None:
                nw = nw.reindex(cols).fillna(0.0)
                cost = float((nw - w).abs().sum()) * COST
                w = nw
                started = True
        if started:
            daily.append((t, r - cost))
    return pd.Series(dict(daily)).sort_index()


def stats(r):
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * math.sqrt(252)
    dd = eq / eq.cummax() - 1
    # The account's own rule: stop if 10% below the STARTING balance.
    below = eq[eq <= 1 - config.MAX_DRAWDOWN_PCT]
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    return {"cagr": cagr, "vol": vol, "sharpe": (r.mean() * 252) / vol if vol else np.nan,
            "mdd": dd.min(), "mdd_date": dd.idxmin(), "worst_month": monthly.min(),
            "pos_years": float((yearly > 0).mean()), "yearly": yearly,
            "halt_date": below.index[0] if len(below) else None, "equity": eq}


def dd_risk(r, runs=5000, seed=7):
    """Chance that a random year (built from 1-month blocks of history) falls 10% below its start."""
    m = ((1 + r).groupby([r.index.year, r.index.month]).prod() - 1).to_numpy()
    if len(m) < 24:
        return np.nan
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(runs):
        path = np.cumprod(1 + rng.choice(m, 12))
        hits += path.min() <= 1 - config.MAX_DRAWDOWN_PCT
    return hits / runs


def _svg(curves):
    w, h = 640, 220
    allv = np.concatenate([c.values for c in curves.values()])
    lo, hi = np.log(allv.min()), np.log(allv.max())
    colors = {"trend": "#1f8a4c", "trend_vt": "#2b6cb0", "momentum": "#b7791f", "spy": "#888", "6040": "#bbb"}
    lines, legend = "", ""
    for k, c in curves.items():
        x = np.linspace(0, w, len(c))
        y = h - (np.log(c.values) - lo) / (hi - lo or 1) * h
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x[::3], y[::3]))
        lines += f"<polyline points='{pts}' fill='none' stroke='{colors[k]}' stroke-width='2'/>"
        legend += f"<span style='color:{colors[k]}'>■</span> {html.escape(NAMES_HE[k])} &nbsp; "
    return (f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='none' style='width:100%;height:220px;"
            f"background:var(--card)'>{lines}</svg><p class='mut'>{legend}(סקאלה לוגריתמית)</p>")


def report(results, period, mode):
    pct = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:+.1f}%"
    rows = ""
    for k, s in results.items():
        st = s["st"]
        halt = st["halt_date"].strftime("%Y-%m") if st.get("halt_date") is not None else "לא"
        rows += (f"<tr><td>{html.escape(NAMES_HE[k])}</td><td>{pct(st['cagr'])}</td><td>{st['vol'] * 100:.1f}%</td>"
                 f"<td>{st['sharpe']:.2f}</td><td>{pct(st['mdd'])}</td><td>{pct(st['worst_month'])}</td>"
                 f"<td>{st['pos_years'] * 100:.0f}%</td><td>{halt}</td><td>{s['risk'] * 100:.0f}%</td></tr>")
    years = sorted(set().union(*[set(s["st"]["yearly"].index) for s in results.values()]))
    yh = "".join(f"<th>{html.escape(NAMES_HE[k])}</th>" for k in results)
    yrows = "".join("<tr><td>" + str(y) + "</td>" + "".join(
        f"<td>{pct(s['st']['yearly'].get(y, np.nan))}</td>" for s in results.values()) + "</tr>" for y in years)
    halves = ""
    for k, s in results.items():
        a, b = s["first"], s["second"]
        halves += (f"<tr><td>{html.escape(NAMES_HE[k])}</td><td>{pct(a.get('cagr'))}</td><td>{pct(a.get('mdd'))}</td>"
                   f"<td>{pct(b.get('cagr'))}</td><td>{pct(b.get('mdd'))}</td></tr>")
    best = max((k for k in results if k not in ("spy", "6040")), key=lambda k: results[k]["st"]["sharpe"])
    page = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Trend Backtest</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e3de}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ececea;--mut:#9a9a94;--card:#1e1e1c;--line:#33332f}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif}}
main{{max-width:820px;margin:auto;padding:20px 16px 40px}} h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px}}
.mut{{color:var(--mut);font-size:13px}} .tw{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:start;white-space:nowrap}}
</style></head><body><main>
<h1>בקטסט · מעקב מגמה ומומנטום על קרנות סל</h1>
<p class="mut">{html.escape(mode)} · {html.escape(period)} · 12 קרנות סל (מניות ארה״ב, חו״ל, שווקים מתעוררים, נדל״ן, אג״ח, זהב, סחורות) · איזון חודשי · עלות 5 bps לצד · מזומן = אג״ח ממשלתי קצר (SHV)</p>
<p>הכללים נקבעו מראש לפי מחקר של עשרות שנים, בלי חיפוש ובלי התאמה לנתונים, ולכן אין צורך בבדיקת מזל.
האסטרטגיה עם היחס הטוב ביותר בין תשואה לסיכון (Sharpe): <b>{html.escape(NAMES_HE[best])}</b>.</p>
<h2>סיכום</h2>
<div class="tw"><table><thead><tr><th>אסטרטגיה</th><th>תשואה שנתית</th><th>תנודתיות</th><th>Sharpe</th>
<th>ירידה מקסימלית</th><th>החודש הגרוע</th><th>שנים חיוביות</th><th>עצירה ב־10% מהפתיחה?</th><th>סיכוי לירידה של 10% בשנה</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="mut">"עצירה ב־10% מהפתיחה" = האם חשבון שהתחיל בתחילת התקופה היה יורד ל־90% מיתרת הפתיחה (כלל הסיכון שלך), ומתי.
"סיכוי לירידה של 10% בשנה" = שנה אקראית שנבנתה מחודשים היסטוריים, 5,000 פעמים.</p>
{_svg({k: s['st']['equity'] for k, s in results.items()})}
<h2>יציבות: חצי ראשון מול חצי שני</h2>
<div class="tw"><table><thead><tr><th>אסטרטגיה</th><th>תשואה שנתית, חצי 1</th><th>ירידה מקס׳, חצי 1</th>
<th>תשואה שנתית, חצי 2</th><th>ירידה מקס׳, חצי 2</th></tr></thead><tbody>{halves}</tbody></table></div>
<h2>לפי שנה</h2>
<div class="tw"><table><thead><tr><th>שנה</th>{yh}</tr></thead><tbody>{yrows}</tbody></table></div>
<p class="mut">ביצועי עבר אינם מבטיחים תשואה עתידית. זה כלי מחקר, לא ייעוץ השקעות.</p>
</main></body></html>"""
    path = config.OUTPUT_DIR / "report_trend.html"
    path.write_text(page, encoding="utf-8")
    return path, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-10-28")
    args = ap.parse_args()
    print("[1] prices")
    px = load_prices(args.start, args.end, args.demo)
    px = px.dropna(how="any")
    print(f"    {len(px)} days, {px.index[0].date()} → {px.index[-1].date()}")
    results = {}
    for k in ("trend", "trend_vt", "momentum", "spy", "6040"):
        print(f"[2] {k}")
        r = simulate(px, k)
        mid = r.index[len(r) // 2]
        results[k] = {"st": stats(r), "risk": dd_risk(r), "first": stats(r[r.index < mid]),
                      "second": stats(r[r.index >= mid])}
        st = results[k]["st"]
        print(f"    CAGR {st['cagr'] * 100:.1f}%  vol {st['vol'] * 100:.1f}%  Sharpe {st['sharpe']:.2f}  MDD {st['mdd'] * 100:.1f}%")
    path, best = report(results, f"{px.index[0].date()} עד {px.index[-1].date()}",
                        "מצב דמו (נתונים סינתטיים)" if args.demo else "נתונים אמיתיים (Alpaca, מחירים מותאמים לדיבידנדים)")
    print(f"\n=== RESULT ===\nbest: {best}  {path}")


if __name__ == "__main__":
    main()
