import pytest

from conftest import utc
from peer_options.pit_store import EARNINGS_DATE, LookaheadError, PointInTimeStore, Record

T0, T1, T2 = utc(2026, 10, 1), utc(2026, 10, 5), utc(2026, 10, 9)


def store() -> PointInTimeStore:
    s = PointInTimeStore()
    s.put(Record("tone", "TXN", T1, {"v": 1.0}))
    return s


def test_future_record_is_excluded():
    s = store()
    s.put(Record("tone", "TXN", T2, {"v": 99.0}))  # injected future-dated record
    assert [r.payload["v"] for r in s.history("tone", "TXN", as_of=T1)] == [1.0]
    assert s.latest("tone", "TXN", as_of=T0) is None
    assert s.latest("tone", "TXN", as_of=T2).payload["v"] == 99.0


def test_as_of_is_required_and_must_be_tz_aware():
    s = store()
    with pytest.raises(TypeError):
        s.history("tone", "TXN")  # type: ignore[call-arg]
    with pytest.raises(LookaheadError):
        s.history("tone", "TXN", as_of=utc(2026, 10, 5).replace(tzinfo=None))


def test_availability_boundary_is_inclusive():
    assert store().latest("tone", "TXN", as_of=T1) is not None


def test_visibility_is_monotone_in_as_of():
    s = store()
    for t in (T1, T2):
        s.put(Record("tone", "TXN", t, {}))
    counts = [len(s.history("tone", "TXN", as_of=t)) for t in (T0, T1, T2)]
    assert counts == sorted(counts)


def test_estimated_earnings_date_never_served():
    s = PointInTimeStore()
    s.put(Record(EARNINGS_DATE, "NXPI", T1, {"date_status": "estimated", "report_datetime": T2}))
    assert s.earnings_date("NXPI", as_of=utc(2026, 12, 31)) is None


def test_confirmed_earnings_date_only_after_confirmation():
    s = PointInTimeStore()
    s.put(Record(EARNINGS_DATE, "LRCX", T1, {"date_status": "confirmed", "report_datetime": T2}))
    assert s.earnings_date("LRCX", as_of=T0) is None
    assert s.earnings_date("LRCX", as_of=T1) == T2


def test_later_revision_supersedes_and_can_unconfirm():
    s = PointInTimeStore()
    s.put(Record(EARNINGS_DATE, "ON", T0, {"date_status": "confirmed", "report_datetime": T1}))
    s.put(Record(EARNINGS_DATE, "ON", T1, {"date_status": "estimated", "report_datetime": T2}))
    assert s.earnings_date("ON", as_of=T0) == T1
    assert s.earnings_date("ON", as_of=T2) is None
