"""Event-study (H1/H2) specification. Interface only; the estimator is not built yet."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class EventStudySpec:
    outcome: str = "abnormal_iv_change"
    horizon_days: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    controls: tuple[str, ...] = (
        "announcer_iv_change", "industry_date_avg_iv_change", "announcer_earnings_surprise",
        "announcer_stock_return", "peer_iv_rank", "days_to_peer_report",
    )
    fixed_effects: tuple[str, ...] = ("date",)
    cluster: tuple[str, ...] = ("date", "industry")


@dataclass(frozen=True)
class DayCoefficient:
    day: int
    coef: float
    std_err: float
    n_obs: int


class EventStudy(Protocol):
    def run(self, spec: EventStudySpec, *, allow_holdout: bool = False) -> list[DayCoefficient]:
        """Day-by-day coefficients on S_BA from A's call through day 5."""
        ...


def abnormal_iv(peer_iv_change: float, sector_etf_iv_change: float, market_vol_change: float,
                beta_sector: float, beta_market: float) -> float:
    """B's IV change net of sector-ETF and market-vol components (betas estimated elsewhere)."""
    return peer_iv_change - beta_sector * sector_etf_iv_change - beta_market * market_vol_change
