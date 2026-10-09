import csv
from dataclasses import replace
from datetime import datetime

import pytest

from conftest import utc
from peer_options.calendar import ET
from peer_options.events import (
    DateStatus, EventValidationError, LinkType, events_as_of, is_decision_usable, make_event, read_csv,
    first_reaction_session, is_pre_earnings_tradable, validate_event, validate_events, write_csv,
)
from peer_options.seed import SEEDS, SOURCES, build_seed

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


def test_report_before_signal_is_an_error(cal):
    e = ev(peer_report=datetime(2026, 10, 21, 18, 0, tzinfo=ET))  # before signal_available_at 19:30 ET
    assert any("after signal_available_at" in m for m in validate_event(e, cal))


def test_zero_and_negative_windows_are_tagged_not_rejected(cal):
    # BMO report the morning after an evening call: entry Oct 22 close == exit Oct 22 close
    zero = ev(peer_report=datetime(2026, 10, 23, 3, 30, tzinfo=ET))
    assert zero.window_trading_days == 0 and "no_pre_earnings_window" in zero.tags
    assert validate_event(zero, cal) == [] and not is_pre_earnings_tradable(zero)
    # report 03:30 ET Oct 22: the earliest entry (Oct 22 close) is after the report
    neg = ev(peer_report=datetime(2026, 10, 22, 3, 30, tzinfo=ET))
    assert neg.window_trading_days == -1 and "peer_reports_before_entry" in neg.tags
    assert validate_event(neg, cal) == [] and not is_pre_earnings_tradable(neg)


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
    seed, pending = build_seed()
    validate_events(seed, cal)
    assert len(seed) == 28
    assert {p.event_id for p in pending} == {"TXN_ON_3Q26", "TXN_MCHP_3Q26"}  # dates not announced yet
    assert {e.date_status for e in seed} == {DateStatus.CONFIRMED}  # only announced dates are events
    flagged = {e.event_id for e in seed if e.needs_reverify}
    assert "TSMC_KLA_2026Q3_LIVE" in flagged and "TXN_MCHP_3Q25" in flagged
    assert not flagged & {"TSMC_KLA_2025Q3", "TXN_NXPI_3Q25"}
    write_csv(seed, tmp_path / "s.csv")
    assert read_csv(tmp_path / "s.csv") == seed


def test_committed_seed_matches_generator():
    from peer_options.seed import OUT
    assert read_csv(OUT) == build_seed()[0]


@pytest.mark.parametrize("fname", [f for f, _ in SOURCES])
def test_user_columns_agree_with_independent_calendar_math(fname, cal):
    """Check the user's derived columns against our calendar, not against our own window logic."""
    by_id = {e.event_id: e for e in build_seed()[0]}
    with (SEEDS / fname).open(newline="") as fh:
        for r in csv.DictReader(fh):
            e = by_id.get(r["event_id"])
            if e is None:
                continue
            call_d = e.announcer_call_start.astimezone(ET).date()
            rep_d = e.peer_report_datetime.astimezone(ET).date()
            assert cal.trading_days_between(call_d, rep_d) + 1 == int(r["trading_days_call_to_report_inclusive"]), e.event_id
            assert first_reaction_session(e.peer_report_datetime, cal).isoformat() == r["first_reaction_session"], e.event_id
            assert is_decision_usable(e) == (r["announced_before_announcer_call"] == "true"), e.event_id
            assert (rep_d - e.status_as_of.astimezone(ET).date()).days == int(r["days_announced_before_report"]), e.event_id


def test_known_untradable_stm_events():
    by_id = {e.event_id: e for e in build_seed()[0]}
    assert [by_id[f"TXN_STM_{q}"].window_trading_days for q in ("3Q25", "4Q25", "1Q26", "2Q26", "3Q26")] == [0, 0, -1, -1, 4]


def test_imputations_are_tagged():
    by_id = {e.event_id: e for e in build_seed()[0]}
    nx = by_id["TXN_NXPI_3Q25"]  # release clock time not stated -> imputed 16:05 ET
    assert nx.peer_report_datetime.hour == 20 and nx.peer_report_datetime.minute == 5
    assert "release_clock_time_not_stated" in nx.tags
    mc = by_id["TXN_MCHP_3Q25"]  # date-only, 30-day assumed announcement -> end of ET day
    assert "announce_time_eod_imputed" in mc.tags and mc.needs_reverify
    assert by_id["TSMC_KLA_2026Q2"].tags.count("stock_split_adjust") == 1
    assert "confounder:ASML@2026-10-14" in by_id["TSMC_KLA_2026Q3_LIVE"].tags
    assert (by_id["TSMC_KLA_2026Q2"].announcer_call_end - by_id["TSMC_KLA_2026Q2"].announcer_call_start).seconds == 5400


def test_unannounced_date_is_not_decision_usable():
    late = ev(date_status=DateStatus.CONFIRMED, source="s", status_as_of=utc(2026, 10, 26))
    assert not is_decision_usable(late)  # announced after the signal was available
