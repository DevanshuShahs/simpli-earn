"""Events table: one row per (announcer call, not-yet-reported peer) pair, with validators."""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from peer_options.calendar import TradingCalendar

ENTRY_RULE = "first_open_after_signal__enter_at_close"
EXIT_RULE = "close_before_peer_report"


class LinkType(str, Enum):
    SUPPLIER = "supplier"
    CUSTOMER = "customer"
    INDUSTRY_PEER = "industry_peer"


class DateStatus(str, Enum):
    CONFIRMED = "confirmed"
    ESTIMATED = "estimated"


class EventValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    event_id: str
    announcer: str
    peer: str
    link_type: LinkType
    announcer_call_start: datetime  # UTC
    announcer_call_end: datetime  # UTC
    call_end_status: DateStatus  # estimated unless the true end time is known
    peer_report_datetime: datetime  # UTC
    date_status: DateStatus  # status of peer_report_datetime
    status_as_of: datetime  # UTC, when date_status was last checked
    source: str  # required when date_status == confirmed
    signal_available_at: datetime  # UTC
    entry_rule: str
    exit_rule: str
    window_trading_days: int
    needs_reverify: bool = True
    tags: tuple[str, ...] = ()
    notes: str = ""


def derive_window(
    signal_available_at: datetime, peer_report_datetime: datetime, cal: TradingCalendar
) -> tuple[date, date, int]:
    """(entry_date, exit_date, window_trading_days). Entries and exits are at the close."""
    entry = cal.first_open_at_or_after(signal_available_at)
    exit_ = cal.last_close_at_or_before(peer_report_datetime)
    if exit_ < entry:
        return entry, exit_, -1
    return entry, exit_, cal.trading_days_between(entry, exit_)


def make_event(
    *,
    event_id: str,
    announcer: str,
    peer: str,
    link_type: LinkType,
    call_start: datetime,
    peer_report: datetime,
    date_status: DateStatus,
    status_as_of: datetime,
    source: str = "",
    call_duration: timedelta = timedelta(minutes=60),
    transcript_lag: timedelta = timedelta(minutes=120),
    call_end_status: DateStatus = DateStatus.ESTIMATED,
    cal: TradingCalendar | None = None,
    needs_reverify: bool = True,
    tags: tuple[str, ...] = (),
    notes: str = "",
) -> Event:
    """Build an Event, deriving call end, signal availability and window from the rules."""
    cal = cal or TradingCalendar.from_yaml()
    call_end = call_start + call_duration
    available = call_end + transcript_lag
    _, _, n = derive_window(available, peer_report, cal)
    return Event(
        event_id=event_id,
        announcer=announcer,
        peer=peer,
        link_type=link_type,
        announcer_call_start=call_start.astimezone(timezone.utc),
        announcer_call_end=call_end.astimezone(timezone.utc),
        call_end_status=call_end_status,
        peer_report_datetime=peer_report.astimezone(timezone.utc),
        date_status=date_status,
        status_as_of=status_as_of.astimezone(timezone.utc),
        source=source,
        signal_available_at=available.astimezone(timezone.utc),
        entry_rule=ENTRY_RULE,
        exit_rule=EXIT_RULE,
        window_trading_days=n,
        needs_reverify=needs_reverify,
        tags=tags,
        notes=notes,
    )


def validate_event(e: Event, cal: TradingCalendar | None = None) -> list[str]:
    cal = cal or TradingCalendar.from_yaml()
    errs: list[str] = []
    for name in (
        "announcer_call_start",
        "announcer_call_end",
        "peer_report_datetime",
        "status_as_of",
        "signal_available_at",
    ):
        v = getattr(e, name)
        if v.tzinfo is None or v.utcoffset() != timedelta(0):
            errs.append(f"{name} must be tz-aware UTC")
    if errs:
        return errs
    if e.announcer == e.peer:
        errs.append("announcer and peer must differ")
    if e.date_status is DateStatus.CONFIRMED and not e.source.strip():
        errs.append("date_status=confirmed requires a non-empty source")
    if e.date_status is DateStatus.CONFIRMED and e.status_as_of >= e.peer_report_datetime:
        errs.append("confirmed date must have status_as_of before the report itself")
    if e.announcer_call_end <= e.announcer_call_start:
        errs.append("announcer_call_end must be after announcer_call_start")
    if e.signal_available_at < e.announcer_call_end:
        errs.append("signal_available_at must be >= announcer_call_end")
    if e.peer_report_datetime <= e.signal_available_at:
        errs.append("peer_report_datetime must be after signal_available_at")
    if e.entry_rule != ENTRY_RULE or e.exit_rule != EXIT_RULE:
        errs.append(f"unknown entry/exit rule ({e.entry_rule!r}, {e.exit_rule!r})")
    else:
        entry, exit_, n = derive_window(e.signal_available_at, e.peer_report_datetime, cal)
        if n < 1:
            errs.append(f"no tradable window (entry {entry}, exit {exit_})")
        elif n != e.window_trading_days:
            errs.append(f"window_trading_days={e.window_trading_days} but rules give {n}")
    return errs


def validate_events(events: list[Event], cal: TradingCalendar | None = None) -> None:
    cal = cal or TradingCalendar.from_yaml()
    problems: list[str] = []
    seen: set[str] = set()
    for e in events:
        if e.event_id in seen:
            problems.append(f"{e.event_id}: duplicate event_id")
        seen.add(e.event_id)
        problems += [f"{e.event_id}: {m}" for m in validate_event(e, cal)]
    if problems:
        raise EventValidationError("\n".join(problems))


def is_decision_usable(e: Event) -> bool:
    """True if the peer's report date was already public when the trade signal became available.

    Events failing this can be studied ex post but not traded on a known-date basis.
    """
    return e.date_status is DateStatus.CONFIRMED and e.status_as_of <= e.signal_available_at


def events_as_of(events: list[Event], as_of: datetime) -> list[Event]:
    """Events whose peer report date was publicly confirmed at as_of.

    Estimated dates are never usable for trading decisions.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware")
    return [
        e
        for e in events
        if e.date_status is DateStatus.CONFIRMED and e.status_as_of <= as_of
    ]


_ISO = "%Y-%m-%dT%H:%M:%SZ"
_DT_FIELDS = {
    "announcer_call_start",
    "announcer_call_end",
    "peer_report_datetime",
    "status_as_of",
    "signal_available_at",
}


def write_csv(events: list[Event], path: Path) -> None:
    names = [f.name for f in fields(Event)]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for e in events:
            row: dict[str, object] = {}
            for n in names:
                v = getattr(e, n)
                if n in _DT_FIELDS:
                    v = v.strftime(_ISO)
                elif isinstance(v, Enum):
                    v = v.value
                elif n == "tags":
                    v = ";".join(v)
                row[n] = v
            w.writerow(row)


def read_csv(path: Path) -> list[Event]:
    out: list[Event] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            kw: dict[str, object] = dict(row)
            for n in _DT_FIELDS:
                kw[n] = datetime.strptime(row[n], _ISO).replace(tzinfo=timezone.utc)
            kw["link_type"] = LinkType(row["link_type"])
            kw["date_status"] = DateStatus(row["date_status"])
            kw["call_end_status"] = DateStatus(row["call_end_status"])
            kw["window_trading_days"] = int(row["window_trading_days"])
            kw["needs_reverify"] = row["needs_reverify"] == "True"
            kw["tags"] = tuple(t for t in row["tags"].split(";") if t)
            out.append(Event(**kw))  # type: ignore[arg-type]
    return out
