"""Support & Resistance with hold-rate and retests (OUR BUILD - Flux Charts 'S&R Strategy' / 'Retest Marker' are
closed or described only; Adaptive Support and Resistance Zones [BigBeluga]
https://www.tradingview.com/script/E8byEL6g-Adaptive-Support-and-Resistance-Zones-BigBeluga/ described: pivots,
an ATR merge threshold, ATR-wide zones, a 'Break Sensitivity'. Tasks tal1411set4 (1), Lima Signal, tal1411set6 (3)).
Levels = confirmed swing pivots (k each side, known k bars late), merged when within merge x ATR; a TEST = a bar whose
range reaches the zone (level +/- zone x ATR); a HOLD = a test whose close stays on the level's side; hold_pct =
holds / tests. A BREAK = a close beyond the zone by brk x ATR; the broken level flips role (support -> resistance);
RETEST = the first bar that comes back into the flipped zone and closes away from it (rejection). No repaint."""
import numpy as np

from agents.indicators._core import atr, cols, pivots


def sr_levels(df, k=5, merge=0.5, zone=0.25, brk=0.1, max_levels=30):
    """dict: sup, res (nearest live support below / resistance above the close), sup_tests, sup_hold (0..1),
    res_tests, res_hold, break_up / break_dn (close through a resistance / support), retest_long (a broken
    resistance retested as support and rejected), retest_short, retest_level, retest_tests (tests the level had
    before it broke), pre_break_swing (for retest_short: the last swing high before the break; long: the swing low)."""
    o, h, l, c, v = cols(df)
    n = len(c)
    a = atr(h, l, c, 14)
    phc, plc, phv, plv, _, _ = pivots(h, l, k)
    lv = []  # dict(p, tests, holds, role +1 support / -1 resistance, broken_at, retested, tests_before, swing)
    keys = ["sup", "res", "sup_tests", "sup_hold", "res_tests", "res_hold", "retest_level", "retest_tests",
            "pre_break_swing"]
    out = {x: np.full(n, np.nan) for x in keys}
    for x in ["break_up", "break_dn", "retest_long", "retest_short"]:
        out[x] = np.zeros(n, bool)
    last_sh = last_sl = np.nan
    for i in range(n):
        ai = a[i] if np.isfinite(a[i]) else 0.0
        zw = zone * ai
        for L in lv:
            p = L["p"]
            if L["role"] == 1:  # support
                if c[i] < p - zw - brk * ai:
                    L.update(role=-1, broken_at=i, retested=False, tests_before=L["tests"], swing=last_sh)
                    out["break_dn"][i] = True
                    L["tests"], L["holds"] = 0, 0
                elif l[i] <= p + zw:
                    L["tests"] += 1
                    L["holds"] += int(c[i] >= p - zw)
                    if L.get("broken_at", -1) >= 0 and not L["retested"] and i > L["broken_at"] and c[i] > p + zw:
                        out["retest_long"][i] = True
                        out["retest_level"][i] = p
                        out["retest_tests"][i] = L["tests_before"]
                        out["pre_break_swing"][i] = L["swing"]
                        L["retested"] = True
            else:  # resistance
                if c[i] > p + zw + brk * ai:
                    L.update(role=1, broken_at=i, retested=False, tests_before=L["tests"], swing=last_sl)
                    out["break_up"][i] = True
                    L["tests"], L["holds"] = 0, 0
                elif h[i] >= p - zw:
                    L["tests"] += 1
                    L["holds"] += int(c[i] <= p + zw)
                    if L.get("broken_at", -1) >= 0 and not L["retested"] and i > L["broken_at"] and c[i] < p - zw:
                        out["retest_short"][i] = True
                        out["retest_level"][i] = p
                        out["retest_tests"][i] = L["tests_before"]
                        out["pre_break_swing"][i] = L["swing"]
                        L["retested"] = True
        for conf, val, role in ((phc[i], phv[i], -1), (plc[i], plv[i], 1)):
            if not conf:
                continue
            if role == -1:
                last_sh = val
            else:
                last_sl = val
            near = [L for L in lv if abs(L["p"] - val) <= merge * ai]
            if near:
                near[0]["tests"] += 1
                near[0]["holds"] += 1
            else:
                lv.append({"p": val, "tests": 1, "holds": 1, "role": role if (val < c[i]) == (role == 1) else -role,
                           "broken_at": -1, "retested": True, "tests_before": 0, "swing": np.nan})
        lv = lv[-max_levels:]
        sups = [L for L in lv if L["role"] == 1 and L["p"] <= c[i]]
        ress = [L for L in lv if L["role"] == -1 and L["p"] >= c[i]]
        if sups:
            s = max(sups, key=lambda x: x["p"])
            out["sup"][i], out["sup_tests"][i] = s["p"], s["tests"]
            out["sup_hold"][i] = s["holds"] / max(s["tests"], 1)
        if ress:
            r = min(ress, key=lambda x: x["p"])
            out["res"][i], out["res_tests"][i] = r["p"], r["tests"]
            out["res_hold"][i] = r["holds"] / max(r["tests"], 1)
    return out
