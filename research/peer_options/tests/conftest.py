from datetime import datetime, timezone

import pytest

from peer_options.calendar import TradingCalendar


@pytest.fixture(scope="session")
def cal() -> TradingCalendar:
    return TradingCalendar.from_yaml()


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)
