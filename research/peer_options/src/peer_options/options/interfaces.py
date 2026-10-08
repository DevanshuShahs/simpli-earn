"""Options data interface and per-ticker-day feature schema. No ingestion in this session."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


@dataclass(frozen=True)
class OptionDayFeatures:
    ticker: str
    trade_date: date
    available_at: datetime      # when this EOD snapshot could be used
    atm_iv_30d: float           # constant-maturity ATM implied vol (30d)
    atm_iv_60d: float
    put_skew_25d: float         # IV(25d put) - IV(ATM)
    bid_ask_pct_mid: float
    open_interest: float
    implied_earnings_move: float | None = None


FEATURE_COLUMNS = [
    "ticker", "trade_date", "available_at", "atm_iv_30d", "atm_iv_60d",
    "put_skew_25d", "bid_ask_pct_mid", "open_interest", "implied_earnings_move",
]


class OptionsProvider(Protocol):
    def features(
        self, ticker: str, start: date, end: date, *, as_of: datetime
    ) -> list[OptionDayFeatures]:
        """Rows with trade_date in [start, end] and available_at <= as_of."""
        ...
