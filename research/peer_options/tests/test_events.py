from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from conftest import utc
from peer_options.calendar import ET
from peer_options.events import (
    DateStatus, EventValidationError, LinkType, events_as_of, make_event, read_csv,
    validate_event, validate_events, write_csv,
)
from peer_options.seed import build_seed

AS_OF = utc(2026, 10, 7)


def ev(**kw):
    base = dict(
        event_id="X", announcer="TXN", peer="NXPI", link_type=LinkType.INDUSTRY_PEER,
        call_start=datetime(2026, 10, 21, 16, 30, tzinfo=ET),
        peer_report=datetime(2026, 10, 27, 16, 30, tzinfo=ET),
        date_status=DateStatus.ESTIMATED, status_as_of=AS_OF,
    )
    base.update(kw)
    return make_event(**base)


def test_window_derivation_after_close_call(cal):
    e = ev()
    # signal 19:30 ET Oct 21 -> enter Oct 22 close; AMC report Oct 27 -> exit Oct 27 close
    assert e.window_trading_days == 3
    assert validate_event(e, cal) == []


def test_bmo_peer_exits_prior_close(cal):
    e = ev(peer_report=datetime(2026, 10, 29, 4, 30, tzinfo=ET))
    assert e.window_trading_days == 4  # Oct 22 -> Oct 28 close


def test_pre_open_call_enters_same_day(cal):
    e = ev(call_start=datetime(2026, 10, 15, 2, 0, tzinfo=ET),
           peer_report=datetime(2026, 10, 21, 17, 0, tzinfo=ET))
    assert e.window_trading_days == 4  # Oct 15 -> Oct 21


def test_confirmed_requires_source(cal):
    e = ev(date_status=DateStatus.CONFIRMED)
    assert any("source" in m for m in validate_event(e, cal))
    assert validate_event(ev(date_status=DateStatus.CONFIRMED, source="co. press release"), cal) == []


def test_rejects_naive_and_non_utc(cal):
    e = replace(ev(), peer_report_datetime=datetime(2026, 10, 27, 20, 30))
    assert any("UTC" in m for m in validate_event(e, cal))


def test_rejects_peer_before_signal_and_no_window(cal):
    e = ev(peer_report=datetime(2026, 10, 21, 20, 0, tzinfo=ET))  # same evening as signal
    errs = validate_event(e, cal)
    assert any("no tradable window" in m for m in errs)


def test_rejects_wrong_window_and_signal_before_call_end(cal):
    e = ev()
    assert any("window_trading_days" in m for m in validate_event(replace(e, window_trading_days=9), cal))
    bad = replace(e, signal_available_at=e.announcer_call_end.replace(minute=0, hour=e.announcer_call_end.hour - 1))
    assert any("signal_available_at" in m for m in validate_event(bad, cal))


def test_duplicate_ids_rejected(cal):
    with pytest.raises(EventValidationError, match="duplicate"):
        validate_events([ev(), ev()], cal)


def test_events_as_of_excludes_estimated_and_future_confirmations():
    conf = ev(event_id="C", date_status=DateStatus.CONFIRMED, source="s", status_as_of=utc(2026, 10, 10))
    est = ev(event_id="E")
    assert events_as_of([conf, est], utc(2026, 10, 9)) == []
    assert [e.event_id for e in events_as_of([conf, est], utc(2026, 10, 10))] == ["C"]


def test_seed_is_valid_and_roundtrips(tmp_path, cal):
    seed = build_seed()
    validate_events(seed, cal)
    assert len(seed) == 6
    assert {e.date_status for e in seed} == {DateStatus.CONFIRMED, DateStatus.ESTIMATED}
    assert all(e.needs_reverify for e in seed)
    write_csv(seed, tmp_path / "s.csv")
    assert read_csv(tmp_path / "s.csv") == seed


def test_committed_seed_matches_generator():
    from peer_options.seed import OUT
    assert read_csv(OUT) == build_seed()
