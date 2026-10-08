"""Own-history baselines. Never imputes: too little history gives NaN plus a reason."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, stdev

MIN_CALLS = 4
MAX_CALLS = 8


@dataclass(frozen=True)
class HistoryPoint:
    available_at: datetime
    value: float


@dataclass(frozen=True)
class BaselineResult:
    z: float
    n_obs: int
    reason: str = ""


def history_as_of(points: list[HistoryPoint], as_of: datetime) -> list[HistoryPoint]:
    """Points visible at as_of, oldest first. as_of is required."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware")
    return sorted((p for p in points if p.available_at <= as_of), key=lambda p: p.available_at)


def zscore_vs_history(
    value: float,
    history: list[HistoryPoint],
    *,
    as_of: datetime,
    min_obs: int = MIN_CALLS,
    max_obs: int = MAX_CALLS,
) -> BaselineResult:
    """z-score of value against the firm's previous max_obs visible calls (min_obs required)."""
    vis = history_as_of(history, as_of)[-max_obs:]
    if math.isnan(value):
        return BaselineResult(math.nan, len(vis), "value is NaN")
    if len(vis) < min_obs:
        return BaselineResult(math.nan, len(vis), f"need >= {min_obs} prior calls, have {len(vis)}")
    vals = [p.value for p in vis]
    sd = stdev(vals)
    if sd == 0:
        return BaselineResult(math.nan, len(vis), "zero variance in history")
    return BaselineResult((value - mean(vals)) / sd, len(vis))
