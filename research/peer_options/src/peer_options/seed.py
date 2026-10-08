"""Generate data/seeds/events_seed.csv from the dates supplied on 2026-10-07.

All times below are US/Eastern as given by the user; none was independently verified
(`needs_reverify=True`). Run: python -m peer_options.seed
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from peer_options.calendar import ET
from dataclasses import replace
from datetime import timedelta

from peer_options.events import DateStatus, Event, LinkType, make_event, validate_events, write_csv
from peer_options.import_dates import import_dates_csv

SEED_AS_OF = datetime(2026, 10, 7, tzinfo=timezone.utc)
SRC = "user_provided_2026-10-07"
SEEDS = Path(__file__).resolve().parents[2] / "data" / "seeds"
OUT = SEEDS / "events_seed.csv"
TSMC_CALL = timedelta(minutes=90)  # official IR page: 14:00-15:30 Taiwan


def et(month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=ET)


def build_seed() -> list[Event]:
    C, E = DateStatus.CONFIRMED, DateStatus.ESTIMATED
    tsm = dict(
        announcer="TSM",
        call_start=et(10, 15, 2),
        tags=("confounder:ASML@2026-10-14",),
    )
    txn = dict(announcer="TXN", call_start=et(10, 21, 16, 30), link_type=LinkType.INDUSTRY_PEER)
    rows = [
        make_event(
            event_id="TSM-LRCX-2026Q3", peer="LRCX", link_type=LinkType.SUPPLIER,
            peer_report=et(10, 21, 17), date_status=C, status_as_of=SEED_AS_OF, source=SRC,
            call_duration=TSMC_CALL,
            notes="TSMC call 02:00 ET (14:00 Taipei). Peer time is the call start; release is likely earlier.",
            **tsm,
        ),
        make_event(
            event_id="TXN-NXPI-2026Q3", peer="NXPI",
            peer_report=et(10, 27, 16, 30), date_status=E, status_as_of=SEED_AS_OF,
            notes="~Oct 27 after close, unconfirmed; another calendar says Nov 2.", **txn,
        ),
        make_event(
            event_id="TXN-STM-2026Q3", peer="STM",
            peer_report=et(10, 29, 4, 30), date_status=C, status_as_of=SEED_AS_OF, source=SRC,
            notes="Call 4:30 am ET before the European open; release likely earlier.", **txn,
        ),
        make_event(
            event_id="TXN-ON-2026Q3", peer="ON",
            peer_report=et(11, 2, 7), date_status=E, status_as_of=SEED_AS_OF,
            notes="~Nov 2 before the open (user-confirmed BMO), date unconfirmed; 07:00 ET is a placeholder.", **txn,
        ),
        make_event(
            event_id="TXN-MCHP-2026Q3", peer="MCHP",
            peer_report=et(11, 4, 16, 30), date_status=E, status_as_of=SEED_AS_OF,
            notes="Nov 4 or 5 after close, unconfirmed; earlier date used.", **txn,
        ),
    ]
    imported = import_dates_csv(
        SEEDS / "tsmc_kla_dates.csv", link_type=LinkType.SUPPLIER, call_duration=TSMC_CALL
    )
    # Confounders known for the live window only; historical ones are not catalogued yet.
    live_tags = ("confounder:ASML@2026-10-14", "confounder:LRCX_call@2026-10-21")
    rows += [replace(e, tags=e.tags + live_tags) if e.event_id.endswith("_LIVE") else e for e in imported]
    validate_events(rows)
    return rows


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(build_seed(), OUT)
    print(f"wrote {OUT}")
