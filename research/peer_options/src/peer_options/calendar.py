"""US equity trading calendar (weekdays minus an explicit holiday list)."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "calendar.yaml"


class TradingCalendar:
    def __init__(self, holidays: set[date]) -> None:
        self.holidays = frozenset(holidays)

    @classmethod
    def from_yaml(cls, path: Path = _DEFAULT_PATH) -> "TradingCalendar":
        raw = yaml.safe_load(path.read_text())["holidays"]
        return cls({d if isinstance(d, date) else date.fromisoformat(d) for d in raw})

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def next_trading_day(self, d: date, *, inclusive: bool = False) -> date:
        cur = d if inclusive else d + timedelta(days=1)
        while not self.is_trading_day(cur):
            cur += timedelta(days=1)
        return cur

    def prev_trading_day(self, d: date, *, inclusive: bool = False) -> date:
        cur = d if inclusive else d - timedelta(days=1)
        while not self.is_trading_day(cur):
            cur -= timedelta(days=1)
        return cur

    def trading_days_between(self, start: date, end: date) -> int:
        """Number of trading-day steps from start to end (start == end -> 0)."""
        if end < start:
            raise ValueError("end before start")
        n, cur = 0, start
        while cur < end:
            cur = self.next_trading_day(cur)
            n += 1
        return n

    def first_open_at_or_after(self, ts: datetime) -> date:
        """First trading day whose 09:30 ET open is at or after ts."""
        local = ts.astimezone(ET)
        d = local.date()
        if self.is_trading_day(d) and local.time() <= MARKET_OPEN:
            return d
        return self.next_trading_day(d)

    def last_close_at_or_before(self, ts: datetime) -> date:
        """Latest trading day whose 16:00 ET close is at or before ts."""
        local = ts.astimezone(ET)
        d = local.date()
        if self.is_trading_day(d) and local.time() >= MARKET_CLOSE:
            return d
        return self.prev_trading_day(d)
