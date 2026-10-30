"""Tests for Trading Prophet (נביא המסחר). Run: python -m pytest tests -q"""

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

from equity import Bars, EquitySeries, Point, build_from_trades  # noqa: E402
from ftmo_sim import Simulator, summarize_rolling, verdict  # noqa: E402
from mt5_deals import Trade, parse_rows  # noqa: E402
from news import NewsEvent, symbol_currencies, violations  # noqa: E402
import json  # noqa: E402

UTC = dt.timezone.utc
RULES = json.loads((ROOT / "config" / "ftmo_presets.json").read_text())["ftmo_2step_100k"]
POLICY = json.loads((ROOT / "config" / "ftmo_presets.json").read_text())["prophet_policy"]


def T(day, hour=10, minute=0):
    # 2025-01-06 is a Monday
    return dt.datetime(2025, 1, 6, hour, minute, tzinfo=UTC) + dt.timedelta(days=day)


def series_from(steps, initial=100_000.0):
    """steps: list of (ts, balance, equity_low, open_count); opens = ts where count rises."""
    pts = [Point(T(0, 0, 1), initial, initial, 0)] + [Point(*s) for s in steps]
    opens = [p.ts for i, p in enumerate(pts) if i and p.open_count > pts[i - 1].open_count]
    return EquitySeries(pts, initial, opens, "full", [])


def daily_trades(days, pnl_each, initial=100_000.0):
    steps, bal = [], initial
    for d in range(days):
        steps.append((T(d, 9), bal, bal, 1))
        bal += pnl_each
        steps.append((T(d, 15), bal, bal, 0))
    return series_from(steps, initial)


def test_steady_winner_passes_both_phases():
    s = daily_trades(30, 800)  # +0.8%/day
    res = Simulator(s, RULES).challenge(0)
    assert [p.status for p in res.phases] == ["passed", "passed"]
    assert res.phases[0].trading_days >= 4


def test_min_trading_days_delays_pass():
    steps = [(T(0, 9), 100_000, 100_000, 1), (T(0, 15), 111_000, 111_000, 0)]
    for d in range(1, 5):
        steps += [(T(d, 9), 111_000, 111_000, 1), (T(d, 15), 111_000, 111_000, 0)]
    res = Simulator(series_from(steps), RULES).phase(0, "Challenge", 10)
    assert res.status == "passed" and res.trading_days == 4


def test_floating_loss_breaches_daily_even_if_trade_closes_green():
    steps = [(T(0, 9), 100_000, 100_000, 1),
             (T(0, 11), 100_000, 94_900, 1),   # -5.1% floating
             (T(0, 15), 101_000, 101_000, 0)]
    res = Simulator(series_from(steps), RULES).phase(0, "Challenge", 10)
    assert res.status == "breach_daily"


def test_daily_floor_resets_at_prague_midnight():
    # -3% before midnight Prague, -3% after: two different days -> no daily breach
    steps = [(T(0, 20), 100_000, 100_000, 1), (T(0, 22), 97_000, 97_000, 0),  # 23:00 Prague
             (T(1, 0, 30), 97_000, 97_000, 1), (T(1, 2), 94_000, 94_000, 0)]
    res = Simulator(series_from(steps), RULES).phase(0, "Challenge", 10)
    assert res.status == "incomplete"
    # same -6% inside one Prague day -> breach
    steps2 = [(T(0, 8), 100_000, 100_000, 1), (T(0, 10), 97_000, 97_000, 0),
              (T(0, 11), 97_000, 97_000, 1), (T(0, 12), 94_000, 94_000, 0)]
    assert Simulator(series_from(steps2), RULES).phase(0, "C", 10).status == "breach_daily"


def test_static_max_loss_breach():
    steps, bal = [], 100_000
    for d in range(4):
        steps += [(T(d, 9), bal, bal, 1)]
        bal -= 2_600
        steps += [(T(d, 15), bal, bal, 0)]
    res = Simulator(series_from(steps), RULES).phase(0, "C", 10)
    assert res.status == "breach_max"


def test_rolling_and_verdict_pass():
    s = daily_trades(200, 300)
    sim = Simulator(s, RULES)
    roll = summarize_rolling(sim.rolling(7))
    cont = sim.challenge(0)
    v = verdict(s, cont, sim.weekly_track(52), roll, 0, POLICY)
    assert roll["pass_rate_pct"] == 100.0
    assert v["verdict"] == "PASS"


def test_news_fail_policy():
    s = daily_trades(200, 300)
    sim = Simulator(s, RULES)
    v = verdict(s, sim.challenge(0), sim.weekly_track(), summarize_rolling(sim.rolling()), 3, POLICY)
    assert v["verdict"] == "FAIL"


def test_closed_only_is_provisional():
    s = daily_trades(200, 300)
    s.quality = "closed_only"
    sim = Simulator(s, RULES)
    v = verdict(s, sim.challenge(0), sim.weekly_track(), summarize_rolling(sim.rolling()), 0, POLICY)
    assert v["verdict"] == "PROVISIONAL"


def test_symbol_currencies():
    assert symbol_currencies("EURUSD.cash") == {"EUR", "USD"}
    assert symbol_currencies("XAUUSD") == {"XAU", "USD"}
    assert symbol_currencies("US100.cash") == {"USD"}
    assert symbol_currencies("GER40.cash") == {"EUR"}


def test_news_window():
    tr = Trade("EURUSD", 1, T(0, 12, 33), T(0, 14), 1.1, 1.101, 1, 100)
    ev = [NewsEvent(T(0, 12, 30), "USD", "high"), NewsEvent(T(0, 12, 33), "JPY", "high")]
    hits = violations([tr], ev, 5)
    assert len(hits) == 1 and hits[0]["currency"] == "USD"
    assert violations([tr], [NewsEvent(T(0, 12, 20), "USD", "high")], 5) == []


def test_deals_parse_partial_close_and_equity_rebuild():
    hdr = ["Time", "Deal", "Symbol", "Type", "Direction", "Volume", "Price", "Order",
           "Commission", "Swap", "Profit", "Balance", "Comment"]
    rows = [hdr,
            ["2025.01.06 00:00:00", "1", "", "balance", "", "", "", "", "0", "0", "100000", "100 000.00", ""],
            ["2025.01.06 12:00:00", "2", "EURUSD", "buy", "in", "2.0", "1.10000", "2", "-14", "0", "0", "99 986.00", ""],
            ["2025.01.06 13:00:00", "3", "EURUSD", "sell", "out", "1.0", "1.10100", "3", "0", "0", "100", "100 086.00", ""],
            ["2025.01.06 14:00:00", "4", "EURUSD", "sell", "out", "1.0", "1.10200", "4", "0", "0", "200", "100 286.00", ""]]
    p = parse_rows(rows, "UTC")
    assert p.mode == "full" and len(p.trades) == 2 and p.initial_deposit == 100_000
    t1 = p.trades[0]
    assert t1.direction == 1 and t1.volume == 1.0 and t1.commission == pytest.approx(-7)
    assert t1.value_per_price_unit == pytest.approx(100_000)  # 1 lot EURUSD: $1 per 0.00001*...
    bars = Bars()
    bars.add("EURUSD", [(dt.datetime(2025, 1, 6, 12, m, tzinfo=UTC), 1.1005, 1.0990) for m in range(60)])
    s = build_from_trades(p.trades, 100_000, bars)
    assert s.quality == "full"
    worst = min(pt.equity_low for pt in s.points)
    # both lots open at 1.0990 low: 2 * (-0.0010 * 100000) - 14 = -214
    assert worst == pytest.approx(100_000 - 214)


def test_mt5_tester_ftmo_gate(tmp_path):
    sys.path.insert(0, str(ROOT / "mt5_tester" / "scripts"))
    from mt5_batch_tester import evaluate_ftmo_finalist
    hdr = "<tr><td>Time</td><td>Deal</td><td>Symbol</td><td>Type</td><td>Direction</td><td>Volume</td><td>Price</td><td>Order</td><td>Commission</td><td>Swap</td><td>Profit</td><td>Balance</td><td>Comment</td></tr>"
    rows = ["<tr><td>2025.01.06 00:00:00</td><td>1</td><td></td><td>balance</td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>100000</td><td>100 000.00</td><td></td></tr>"]
    bal, deal = 100_000.0, 2
    day = dt.date(2025, 1, 6)
    for _ in range(260):
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        d = day.strftime("%Y.%m.%d")
        rows.append(f"<tr><td>{d} 10:00:00</td><td>{deal}</td><td>EURUSD</td><td>buy</td><td>in</td><td>1.0</td><td>1.10000</td><td>{deal}</td><td>0</td><td>0</td><td>0</td><td>{bal:.2f}</td><td></td></tr>")
        bal += 300
        rows.append(f"<tr><td>{d} 14:00:00</td><td>{deal+1}</td><td>EURUSD</td><td>sell</td><td>out</td><td>1.0</td><td>1.10300</td><td>{deal+1}</td><td>0</td><td>0</td><td>300</td><td>{bal:.2f}</td><td></td></tr>")
        deal += 2
        day += dt.timedelta(days=1)
    rep = tmp_path / "Bot_EURUSD.htm"
    rep.write_text("<html><table>" + hdr + "".join(rows) + "</table></html>", encoding="utf-16")
    passed, reason, res = evaluate_ftmo_finalist({"report_file": str(rep)}, {"report_tz": "UTC"})
    assert res["verdict"]["verdict"] == "PROVISIONAL"  # no bars -> floating loss unknown
    assert not passed
    passed, _, _ = evaluate_ftmo_finalist({"report_file": str(rep)}, {"report_tz": "UTC", "accept_provisional": True,
                                                                     "policy_overrides": {"news_policy": "flag"}})
    assert passed
    assert (tmp_path / "Bot_EURUSD.htm.prophet.md").read_text(encoding="utf-8").startswith("# נביא המסחר")
