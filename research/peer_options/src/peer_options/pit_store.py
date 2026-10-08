"""Point-in-time store. Every record carries available_at; every query requires as_of.

A record is visible at `as_of` only if available_at <= as_of. For a given
(kind, key), the visible record with the latest available_at supersedes earlier ones
(e.g. a revised earnings date). Earnings dates are further restricted to confirmed ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

EARNINGS_DATE = "earnings_date"


class LookaheadError(ValueError):
    """Raised when a query lacks a valid as_of."""


@dataclass(frozen=True)
class Record:
    kind: str
    key: str
    available_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)


def _check_ts(ts: datetime, name: str) -> None:
    if not isinstance(ts, datetime) or ts.tzinfo is None or ts.utcoffset() is None:
        raise LookaheadError(f"{name} must be a tz-aware datetime")


class PointInTimeStore:
    def __init__(self) -> None:
        self._records: list[Record] = []

    def put(self, record: Record) -> None:
        _check_ts(record.available_at, "available_at")
        self._records.append(record)

    def history(self, kind: str, key: str, *, as_of: datetime) -> list[Record]:
        """All records visible at as_of, oldest first."""
        _check_ts(as_of, "as_of")
        vis = [
            r for r in self._records
            if r.kind == kind and r.key == key and r.available_at <= as_of
        ]
        return sorted(vis, key=lambda r: r.available_at)

    def latest(self, kind: str, key: str, *, as_of: datetime) -> Record | None:
        h = self.history(kind, key, as_of=as_of)
        return h[-1] if h else None

    def earnings_date(self, ticker: str, *, as_of: datetime) -> datetime | None:
        """Report datetime for ticker, only if the latest visible record is confirmed."""
        r = self.latest(EARNINGS_DATE, ticker, as_of=as_of)
        if r is None or r.payload.get("date_status") != "confirmed":
            return None
        return r.payload["report_datetime"]
