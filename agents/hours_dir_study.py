"""Hours / direction study (bots 1-4 smart agent, 17.11.2026): for every strategy of a bot, its stored trades as it
trades now (agents/portfolio_check.current_trades, FTMO costs) - a block of Israel-time hours with no new entry
(2-8 hours, the runner's "hours" rule) or buy / sell only (the "dir" rule), chosen on 30.04-31.12.2025 (time-weighted
total R, the cut trades lost clearly: t <= -2.5 on >= 15 trades) and shown on 2026 (passes at t <= -1 on >= 10 cut
trades). Strategies that already have an hours / dir rule are skipped.  python agents/hours_dir_study.py bot1"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from agents import portfolio_check as PC
from agents import agentkit as K
from agents import repo_io as RIO
MS = (RIO.live_json('live/manager_state.json') or {}).get('risk') or {}
bot = sys.argv[1]
cfg = PC.bot_cfg(bot)
book, missing = PC.book_now(bot, cfg)
book25, _ = PC.book_now(bot, cfg, key="tr25")
rules = MS.get(bot, {})
def frame(tr):
    t = pd.DataFrame(PC._minutes(tr), columns=['a', 'b', 'd', 'r'])
    ts = pd.to_datetime(t['a'], unit='m', utc=True)
    t['h'] = ts.dt.tz_convert('Asia/Jerusalem').dt.hour
    t['time'] = ts
    return t
def tstat(x):
    x = np.asarray(x, float)
    if len(x) < 3 or x.std(ddof=1) == 0: return 0.0
    return x.mean() / x.std(ddof=1) * math.sqrt(len(x))
out = []
for c, (tr, m) in book.items():
    tr25 = (book25.get(c) or (None,))[0]
    if not tr or not tr25 or m <= 0.01: continue
    A, B = frame(tr25), frame(tr)
    A = A[A['time'] >= pd.Timestamp('2025-04-30', tz='UTC')]
    if len(A) < 40 or len(B) < 30: continue
    wA = K.weights(A['time'])
    cands = []
    for d in (1, -1):
        cands.append(('dir', d, A['d'] != d, B['d'] != d))
    for a in range(24):
        for L in range(2, 9):
            b = a + L
            if b > 24: continue
            cands.append(('hours', [a, b], (A['h'] >= a) & (A['h'] < b), (B['h'] >= a) & (B['h'] < b)))
    best = None
    for kind, v, dropA, dropB in cands:
        if kind == 'hours' and c in (rules.get('hours') or {}): continue
        if kind == 'dir' and c in (rules.get('dir') or {}): continue
        nA = int(dropA.sum())
        if nA < 15 or nA > 0.6 * len(A): continue
        gainA = -float((A['r'][dropA] * wA[dropA.to_numpy()]).sum())
        tA = tstat(A['r'][dropA])
        if gainA <= 0 or tA > -2.5: continue
        if best is None or gainA > best['gainA']:
            best = dict(kind=kind, v=v, nA=nA, gainA=gainA, tA=tA, dropB=dropB, dropA=dropA)
    if not best: continue
    dB = best['dropB']
    nB = int(dB.sum())
    gB = -float(B['r'][dB].sum())
    tB = tstat(B['r'][dB])
    out.append(dict(combo=c, kind=best['kind'], v=best['v'], nA=best['nA'], nAll=len(A), gainA=round(best['gainA'], 1), tA=round(best['tA'], 2),
                    nB=nB, nBall=len(B), gainB=round(gB, 1), tB=round(tB, 2),
                    baseA=round(A['r'].sum(), 1), baseB=round(B['r'].sum(), 1), mult=m))
for o in sorted(out, key=lambda o: -o['gainB']):
    ok = 'PASS' if o['nB'] >= 10 and o['tB'] <= -1.0 and o['gainB'] > 0 else '    '
    print(ok, json.dumps(o, ensure_ascii=False))
print('combos', len(book), 'missing', len(missing), 'selected', len(out))
