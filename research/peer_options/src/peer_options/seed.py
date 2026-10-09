"""Build data/seeds/events_seed.csv from the user-supplied *_events.csv files.

Run: python -m peer_options.seed
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from peer_options.events import Event, LinkType, validate_events, write_csv
from peer_options.import_dates import Pending, import_dates_csv

SEEDS = Path(__file__).resolve().parents[2] / "data" / "seeds"
OUT = SEEDS / "events_seed.csv"
TSMC_CALL = timedelta(minutes=90)  # official IR page: 14:00-15:30 Taiwan

SOURCES = [
    ("tsmc_kla_events.csv", LinkType.SUPPLIER),
    ("tsmc_lrcx_events.csv", LinkType.SUPPLIER),
    ("txn_peers_events.csv", LinkType.INDUSTRY_PEER),
]
# Other reporters inside a live window (known only for the current quarter).
LIVE_CONFOUNDERS = {
    "TSMC_KLA_2026Q3_LIVE": ("ASML@2026-10-14", "LRCX_call@2026-10-21"),
    "TSMC_LRCX_2026Q3_LIVE": ("ASML@2026-10-14", "TXN@2026-10-21"),
}


def build_seed() -> tuple[list[Event], list[Pending]]:
    events: list[Event] = []
    pending: list[Pending] = []
    for name, link in SOURCES:
        ev, pend = import_dates_csv(SEEDS / name, link_type=link, call_durations={"TSMC": TSMC_CALL})
        events += ev
        pending += pend
    events = [
        replace(e, tags=e.tags + tuple(f"confounder:{c}" for c in LIVE_CONFOUNDERS.get(e.event_id, ())))
        for e in events
    ]
    validate_events(events)
    return events, pending


if __name__ == "__main__":
    evs, pend = build_seed()
    write_csv(evs, OUT)
    print(f"wrote {len(evs)} events to {OUT}")
    for p in pend:
        print(f"pending (no date yet): {p.event_id} - {p.note}")
