"""Import user-supplied event-date CSVs (e.g. data/seeds/tsmc_kla_dates.csv) into Events.

Mapping: announcer_call_start_utc -> call start; peer_release_time_et -> peer_report_datetime;
date_announced_et -> status_as_of (the moment the peer date became public);
peer_report_source + date_announced_source -> source.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from peer_options.calendar import TradingCalendar
from peer_options.events import DateStatus, Event, LinkType, make_event

TICKERS = {"TSMC": "TSM", "KLA": "KLAC"}


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def import_dates_csv(
    path: Path,
    *,
    link_type: LinkType,
    call_duration: timedelta,
    tickers: dict[str, str] = TICKERS,
    extra_tags: tuple[str, ...] = (),
    cal: TradingCalendar | None = None,
) -> list[Event]:
    out: list[Event] = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            flag = r["assumption_flag"].strip()
            tags = list(extra_tags)
            if flag and flag != "none":
                tags.append(flag)
            if "stock split" in r["notes"].lower():
                tags.append("stock_split_adjust")
            official = "official" in r["announcer_time_source"].lower()
            out.append(
                make_event(
                    event_id=r["event_id"],
                    announcer=tickers[r["announcer"]],
                    peer=tickers[r["peer"]],
                    link_type=link_type,
                    call_start=_dt(r["announcer_call_start_utc"]),
                    peer_report=_dt(r["peer_release_time_et"]),
                    date_status=DateStatus.CONFIRMED,
                    status_as_of=_dt(r["date_announced_et"]),
                    source=f"{r['peer_report_source']} | announced: {r['date_announced_source']}",
                    call_duration=call_duration,
                    call_end_status=DateStatus.CONFIRMED if official else DateStatus.ESTIMATED,
                    cal=cal,
                    needs_reverify=r["status"] != "historical" or bool(flag and flag != "none"),
                    tags=tuple(tags),
                    notes=" ".join(x for x in (r["notes"], f"status={r['status']}") if x),
                )
            )
    return out
