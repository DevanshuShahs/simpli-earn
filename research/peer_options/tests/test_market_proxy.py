import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from peer_options.calendar import ET, TradingCalendar
from peer_options.events import DateStatus, LinkType, make_event
from peer_options.market.prices import (
    apply_splits, cap_end_before_holdout, date_chunks, fetch_prices, jump_flags, normalize_eod,
)
from peer_options.study.proxy import parkinson_var, peer_outcomes

CAL = TradingCalendar.from_yaml()


def bars(start: date, n: int, close_fn, range_pct=0.02) -> pd.DataFrame:
    """SYNTHETIC daily bars on consecutive trading days."""
    days, d = [], start
    while len(days) < n:
        if CAL.is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    close = np.array([close_fn(i) for i in range(n)], float)
    return pd.DataFrame({"date": days, "open": close, "high": close * (1 + range_pct), "low": close * (1 - range_pct),
                         "close": close, "volume": 1.0})


def test_date_chunks_cover_range_without_overlap():
    ch = date_chunks(date(2025, 8, 1), date(2026, 9, 30))
    assert ch[0][0] == date(2025, 8, 1) and ch[-1][1] == date(2026, 9, 30)
    assert all((b - a).days <= 363 for a, b in ch)
    assert all(ch[i + 1][0] == ch[i][1] + timedelta(days=1) for i in range(len(ch) - 1))


def test_normalize_uses_et_date_and_dedupes():
    raw = pd.DataFrame({
        "created": pd.to_datetime(["2026-06-11 17:15:32-04:00", "2026-06-12 17:15:06-04:00", "2026-06-12 17:16:00-04:00"]),
        "open": [1, 2, 3], "high": [2, 3, 4], "low": [1, 1, 1], "close": [1.5, 2.5, 3.5], "volume": [10, 20, 30]})
    df = normalize_eod(raw)
    assert list(df["date"]) == [date(2026, 6, 11), date(2026, 6, 12)] and df.iloc[1]["close"] == 3.5
    assert df["available_at"].dt.tz is not None


def test_split_adjustment_removes_the_jump_and_flagging_finds_it():
    df = bars(date(2026, 6, 8), 6, lambda i: 2400.0 if i < 4 else 250.0)
    assert len(jump_flags(df)) == 1
    adj = apply_splits(df, "KLAC", {"KLAC": [{"date": df.iloc[4]["date"], "ratio": 10}]})
    assert jump_flags(adj) == [] and adj["close"].round(1).tolist() == [240.0] * 4 + [250.0] * 2
    assert adj.iloc[0]["volume"] == 10.0
    assert apply_splits(df, "OTHER", {"KLAC": []})["close"].tolist() == df["close"].tolist()


def test_repo_split_table_has_klac():
    raw = bars(date(2026, 6, 9), 4, lambda i: [2139.37, 2135.64, 2411.64, 254.54][i])
    assert jump_flags(apply_splits(raw, "KLAC")) == []  # uses config/splits.yaml


def test_outcome_fetch_never_reaches_the_holdout():
    assert cap_end_before_holdout(date(2026, 12, 1)) == date(2026, 9, 30)
    assert cap_end_before_holdout(date(2026, 8, 1)) == date(2026, 8, 1)
    assert cap_end_before_holdout(date(2026, 12, 1), allow_holdout=True) == date(2026, 12, 1)


def test_fetch_prices_chunks_requests():
    calls = []

    class Fake:
        def stock_history_eod(self, symbol, start_date, end_date):
            calls.append((start_date, end_date))
            d = pd.DataFrame({"created": [pd.Timestamp(f"{start_date} 17:15", tz=ET)],
                              "open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [1]})
            return d

    out = fetch_prices(Fake(), "SYN", date(2025, 8, 1), date(2026, 9, 30))
    assert len(calls) == 2 and len(out) == 2 and all((b - a).days < 365 for a, b in calls)


def event_for(call_et: datetime, report_et: datetime):
    return make_event(event_id="E", announcer="A", peer="B", link_type=LinkType.SUPPLIER, call_start=call_et,
                      peer_report=report_et, date_status=DateStatus.CONFIRMED,
                      status_as_of=call_et - timedelta(days=10), source="synthetic")


def test_outcomes_alignment_abnormal_return_and_exit_cap():
    # call 02:00 ET Thu 2026-04-16 -> day 0 is the same day; B reports AMC Wed 2026-04-22 -> exit that close.
    e = event_for(datetime(2026, 4, 16, 2, 0, tzinfo=ET), datetime(2026, 4, 22, 16, 5, tzinfo=ET))
    peer = bars(date(2026, 3, 1), 60, lambda i: 100 * 1.01 ** i)
    bench = bars(date(2026, 3, 1), 60, lambda i: 100.0)
    out = peer_outcomes(e, peer, bench, CAL)
    assert [o.day for o in out] == [date(2026, 4, 16), date(2026, 4, 17), date(2026, 4, 20), date(2026, 4, 21), date(2026, 4, 22)]
    assert [o.k for o in out] == [0, 1, 2, 3, 4]          # capped at the exit date, not at horizon 5
    assert [round(o.car, 6) for o in out] == pytest.approx([math.log(1.01) * (k + 1) for k in range(5)], abs=1e-6)
    # same range every day for both -> relative realized-vol change is zero
    assert all(abs(o.abn_logrv) < 1e-9 for o in out)


def test_outcomes_detect_vol_expansion_vs_benchmark():
    e = event_for(datetime(2026, 4, 16, 2, 0, tzinfo=ET), datetime(2026, 4, 28, 16, 5, tzinfo=ET))
    peer = bars(date(2026, 3, 1), 60, lambda i: 100.0, range_pct=0.01)
    wide = peer["date"] >= date(2026, 4, 16)
    peer.loc[wide, "high"], peer.loc[wide, "low"] = 102.0, 98.0   # range 2x wider => variance 4x from day 0
    bench = bars(date(2026, 3, 1), 60, lambda i: 100.0, range_pct=0.01)
    out = peer_outcomes(e, peer, bench, CAL)
    assert out[0].abn_logrv == pytest.approx(math.log((math.log(1.02 / 0.98) / math.log(1.01 / 0.99)) ** 2), rel=0.02)
    assert out[0].abn_logrv == pytest.approx(math.log(4), rel=0.02)


def test_no_outcome_when_baseline_history_is_missing_or_window_untradable():
    e = event_for(datetime(2026, 4, 16, 2, 0, tzinfo=ET), datetime(2026, 4, 22, 16, 5, tzinfo=ET))
    short = bars(date(2026, 4, 10), 20, lambda i: 100.0)         # < 20 days of history before entry
    assert peer_outcomes(e, short, short, CAL) == []
    stm = event_for(datetime(2026, 4, 22, 16, 30, tzinfo=ET), datetime(2026, 4, 23, 3, 30, tzinfo=ET))  # window -1
    full = bars(date(2026, 3, 1), 60, lambda i: 100.0)
    assert peer_outcomes(stm, full, full, CAL) == []
