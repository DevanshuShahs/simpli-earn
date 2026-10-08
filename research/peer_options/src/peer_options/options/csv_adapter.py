"""CSV/Parquet adapter for OptionsProvider. Reads pre-computed feature rows only."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from peer_options.options.interfaces import FEATURE_COLUMNS, OptionDayFeatures


class FileOptionsProvider:
    def __init__(self, path: Path) -> None:
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        self._df = df

    def features(
        self, ticker: str, start: date, end: date, *, as_of: datetime
    ) -> list[OptionDayFeatures]:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be tz-aware")
        d = self._df
        d = d[
            (d["ticker"] == ticker)
            & (d["trade_date"] >= start)
            & (d["trade_date"] <= end)
            & (d["available_at"] <= pd.Timestamp(as_of))
        ].sort_values("trade_date")
        return [
            OptionDayFeatures(
                ticker=r.ticker,
                trade_date=r.trade_date,
                available_at=r.available_at.to_pydatetime(),
                atm_iv_30d=float(r.atm_iv_30d),
                atm_iv_60d=float(r.atm_iv_60d),
                put_skew_25d=float(r.put_skew_25d),
                bid_ask_pct_mid=float(r.bid_ask_pct_mid),
                open_interest=float(r.open_interest),
                implied_earnings_move=(
                    None if pd.isna(r.implied_earnings_move) else float(r.implied_earnings_move)
                ),
            )
            for r in d.itertuples(index=False)
        ]
