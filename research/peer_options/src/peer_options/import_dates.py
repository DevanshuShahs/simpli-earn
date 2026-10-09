"""Import user-supplied event-date CSVs (data/seeds/*_events.csv) into Events.

Mapping: announcer_call_start_utc -> call start; peer_release_time_et -> peer_report_datetime;
date_announced_et -> status_as_of (when the peer date became public); sources are concatenated.
Imputations (always tagged, never silent):
  * empty release time: after_close -> 16:05 ET; before_open -> the peer call time if pre-open else 06:00 ET
    (only which side of the close the report falls matters for the exit rule);
  * date-only announcement -> end of that ET day, the latest possible moment (conservative).
Rows without a peer report date are returned as `pending` and are not events.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from peer_options.calendar import ET, MARKET_OPEN, TradingCalendar
from peer_options.events import DateStatus, Event, LinkType, make_event

TICKERS = {
    "TSMC": "TSM", "KLA": "KLAC", "LRCX": "LRCX", "TXN": "TXN",
    "NXPI": "NXPI", "STM": "STM", "ON": "ON", "MCHP": "MCHP",
}
# Flags that make a row's date suspect enough to re-verify; the rest are tagged only.
MATERIAL_FLAGS = {"announcement_date_assumed_30d"}


@dataclass(frozen=True)
class Pending:
    event_id: str
    reason: str
    note: str


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _release_time(r: dict[str, str]) -> tuple[datetime, bool]:
    if r["peer_release_time_et"].strip():
        return _dt(r["peer_release_time_et"]), False
    d = date.fromisoformat(r["peer_report_date"])
    timing = r["peer_report_timing"]
    if timing == "after_close":
        return datetime.combine(d, time(16, 5), ET), True
    if timing == "before_open":
        call = _dt(r["peer_call_start_et"]).astimezone(ET) if r["peer_call_start_et"].strip() else None
        if call and call.date() == d and call.time() < MARKET_OPEN:
            return call, True
        return datetime.combine(d, time(6, 0), ET), True
    raise ValueError(f"{r['event_id']}: cannot place release time (timing={timing!r})")


def _announced(s: str) -> tuple[datetime, bool]:
    if len(s.strip()) == 10:
        return datetime.combine(date.fromisoformat(s.strip()), time(23, 59, 59), ET), True
    return _dt(s), False


def import_dates_csv(
    path: Path,
    *,
    link_type: LinkType,
    call_durations: dict[str, timedelta] | None = None,
    default_duration: timedelta = timedelta(minutes=60),
    tickers: dict[str, str] = TICKERS,
    cal: TradingCalendar | None = None,
) -> tuple[list[Event], list[Pending]]:
    durations = call_durations or {}
    events: list[Event] = []
    pending: list[Pending] = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if not r["peer_report_date"].strip() or not r["date_announced_et"].strip():
                pending.append(Pending(r["event_id"], r["assumption_flag"], r["notes"]))
                continue
            flags = [f for f in r["assumption_flag"].split(";") if f and f != "none"]
            tags = list(flags)
            report, rel_imputed = _release_time(r)
            announced, ann_imputed = _announced(r["date_announced_et"])
            if rel_imputed and "release_clock_time_not_stated" not in flags:
                tags.append("release_time_imputed")
            if ann_imputed:
                tags.append("announce_time_eod_imputed")
            if "stock split" in r["notes"].lower():
                tags.append("stock_split_adjust")
            official = "official" in r["announcer_time_source"].lower()
            events.append(
                make_event(
                    event_id=r["event_id"],
                    announcer=tickers[r["announcer"]],
                    peer=tickers[r["peer"]],
                    link_type=link_type,
                    call_start=_dt(r["announcer_call_start_utc"]),
                    peer_report=report,
                    date_status=DateStatus.CONFIRMED,
                    status_as_of=announced,
                    source=f"{r['peer_report_source']} | announced: {r['date_announced_source']}",
                    call_duration=durations.get(r["announcer"], default_duration),
                    call_end_status=DateStatus.CONFIRMED if official else DateStatus.ESTIMATED,
                    cal=cal,
                    needs_reverify=r["status"] != "historical" or bool(MATERIAL_FLAGS & set(flags)),
                    tags=tuple(dict.fromkeys(tags)),
                    notes=" ".join(x for x in (r["notes"], f"status={r['status']}") if x),
                )
            )
    return events, pending
