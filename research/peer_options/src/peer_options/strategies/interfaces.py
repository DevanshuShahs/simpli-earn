"""Trade-variant interfaces and a paper-trade log. No broker integration, no backtester."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol


class Variant(str, Enum):
    PRE_EARNINGS_VOL_DRIFT = "pre_earnings_vol_drift"
    HOLD_THROUGH_EARNINGS = "hold_through_earnings"
    SKEW = "skew"


@dataclass(frozen=True)
class Decision:
    event_id: str
    variant: Variant
    side: str            # 'long_vol' | 'short_vol' | 'long_skew' | 'short_skew' | 'none'
    entry_date: date
    exit_date: date
    decided_at: datetime  # must be >= the event's signal_available_at
    rationale: str = ""


class TradeVariant(Protocol):
    variant: Variant

    def decide(self, event_id: str, *, as_of: datetime) -> Decision:
        """Decide using only information available at as_of."""
        ...


class PaperTradeLog:
    """Append-only JSONL log of hypothetical decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, d: Decision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = asdict(d)
        row.update(
            variant=d.variant.value,
            entry_date=d.entry_date.isoformat(),
            exit_date=d.exit_date.isoformat(),
            decided_at=d.decided_at.isoformat(),
        )
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
