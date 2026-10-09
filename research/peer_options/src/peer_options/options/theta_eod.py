"""ThetaData end-of-day option quotes (Free plan: option_history_eod, list_expirations, SOFR) with a parquet cache.

One request per (symbol, expiration, date range); ThetaData returns every strike and side for each trade date.
Raw pulls live under data/raw/options (gitignored). Nothing here needs a paid plan.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from peer_options.calendar import ET

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "raw" / "options"
NORMALIZED = ["date", "expiry", "strike", "right", "bid", "ask", "close", "volume"]


def normalize_option_eod(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=NORMALIZED)
    created = pd.to_datetime(raw["created"], utc=True)
    df = pd.DataFrame({
        "date": created.dt.tz_convert(ET).dt.date,
        "expiry": pd.to_datetime(raw["expiration"]).dt.date,
        "strike": raw["strike"].astype(float),
        "right": raw["right"].astype(str).str.upper().str[0],
        "bid": raw["bid"].astype(float), "ask": raw["ask"].astype(float),
        "close": raw["close"].astype(float), "volume": raw["volume"].astype(float),
    })
    return df.drop_duplicates(["date", "expiry", "strike", "right"], keep="last").reset_index(drop=True)[NORMALIZED]


def needed_expirations(all_expirations: list[date], dates: list[date], *, min_dte: int = 7, max_dte: int = 120) -> list[date]:
    """Expirations that are within [min_dte, max_dte] of at least one requested trade date."""
    return sorted(e for e in all_expirations if any(min_dte <= (e - d).days <= max_dte for d in dates))


class OptionEodStore:
    def __init__(self, client_factory: Callable[[], object] | None = None, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self._factory = client_factory
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self._factory is None:
                from thetadata import ThetaClient

                self._factory = lambda: ThetaClient(dotenv_path=Path.home() / ".thetadata" / "env", dataframe_type="pandas")
            self._client = self._factory()
        return self._client

    def expirations(self, symbol: str) -> list[date]:
        path = self.cache_dir / f"{symbol}_expirations_{date.today()}.parquet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            self.client.option_list_expirations(symbol).to_parquet(path)
        return sorted(pd.to_datetime(pd.read_parquet(path)["expiration"]).dt.date)

    def chain_days(self, symbol: str, expiration: date, start: date, end: date) -> pd.DataFrame:
        """Normalized EOD quotes for one expiration over [start, end]; empty frame if the vendor has none."""
        path = self.cache_dir / f"{symbol}_{expiration}_{start}_{end}.parquet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return pd.read_parquet(path)
        try:
            raw = self.client.option_history_eod(symbol=symbol, expiration=expiration, start_date=start,
                                                 end_date=min(end, expiration), strike="*", right="both")
        except Exception as e:  # noqa: BLE001 - only "no data" is an expected, cacheable outcome
            if type(e).__name__ != "NoDataFoundError":
                raise
            raw = None
        df = normalize_option_eod(raw)
        df.to_parquet(path)
        return df

    def _spans(self, symbol: str, expiration: date) -> list[tuple[date, date]]:
        out = []
        for p in self.cache_dir.glob(f"{symbol}_{expiration}_*_*.parquet"):
            a, b = p.stem.split("_")[-2:]
            out.append((date.fromisoformat(a), date.fromisoformat(b)))
        return out

    def ensure_days(self, symbol: str, expiration: date, days: list[date]) -> None:
        """Fetch only the days not already covered by a cached span (one request over the missing min..max)."""
        spans = self._spans(symbol, expiration)
        missing = [d for d in days if not any(a <= d <= b for a, b in spans)]
        if missing:
            self.chain_days(symbol, expiration, min(missing), max(missing))

    def load_expiration(self, symbol: str, expiration: date) -> pd.DataFrame:
        """Everything cached for one expiration, deduplicated by (date, strike, right)."""
        frames = [pd.read_parquet(p) for p in sorted(self.cache_dir.glob(f"{symbol}_{expiration}_*_*.parquet"))]
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame(columns=NORMALIZED)
        return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "expiry", "strike", "right"], keep="last")

    def sofr(self, start: date, end: date) -> pd.Series:
        """SOFR in percent by date (forward-filled over weekends/holidays by the caller)."""
        path = self.cache_dir / f"SOFR_{start}_{end}.parquet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            self.client.interest_rate_history_eod(symbol="SOFR", start_date=start, end_date=end).to_parquet(path)
        r = pd.read_parquet(path)
        return pd.Series(r["rate"].astype(float).to_numpy(), index=pd.to_datetime(r["created"]).dt.date).sort_index()


def rate_on(sofr: pd.Series, d: date) -> float:
    """Most recent SOFR (percent) on or before d; never uses later dates."""
    prior = sofr[sofr.index <= d]
    if prior.empty:
        raise ValueError(f"no SOFR on or before {d}")
    return float(prior.iloc[-1])


def window_dates(days: list[date], pad_before: int = 0) -> tuple[date, date]:
    return min(days) - timedelta(days=pad_before), max(days)
