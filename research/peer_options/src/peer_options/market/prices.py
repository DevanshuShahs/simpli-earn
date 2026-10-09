"""Daily OHLC from ThetaData (free tier: stock/index EOD) with chunked fetch, caching and split adjustment.

ThetaData allows at most 365 days per request, so ranges are split into <=364-day chunks. EOD bars are stamped with
the time the vendor produced them (~17:15 ET), which is used as `available_at`. Prices are unadjusted at the vendor,
so known splits come from config/splits.yaml and unexplained large jumps are flagged, never silently fixed.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from peer_options.calendar import ET
from peer_options.prereg import PreregError, holdout_bounds

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "raw" / "market"
SPLITS = ROOT / "config" / "splits.yaml"
EOD_COLUMNS = ["date", "open", "high", "low", "close", "volume", "available_at"]


def date_chunks(start: date, end: date, max_days: int = 364) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=max_days - 1), end)
        out.append((cur, stop))
        cur = stop + timedelta(days=1)
    return out


def normalize_eod(raw: pd.DataFrame) -> pd.DataFrame:
    created = pd.to_datetime(raw["created"], utc=True)
    df = pd.DataFrame({
        "date": created.dt.tz_convert(ET).dt.date,
        "open": raw["open"].astype(float), "high": raw["high"].astype(float),
        "low": raw["low"].astype(float), "close": raw["close"].astype(float),
        "volume": raw["volume"].astype(float), "available_at": created,
    })
    df = df[df["close"] > 0]
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)[EOD_COLUMNS]


def cap_end_before_holdout(end: date, *, allow_holdout: bool = False) -> date:
    """Never fetch outcome data inside the holdout unless explicitly allowed (and then the caller logs it)."""
    if allow_holdout:
        return end
    try:
        start_h, _ = holdout_bounds()
    except PreregError:
        return end
    return min(end, start_h - timedelta(days=1))


def apply_splits(df: pd.DataFrame, symbol: str, splits: dict | None = None) -> pd.DataFrame:
    splits = splits if splits is not None else yaml.safe_load(SPLITS.read_text())["splits"]
    out = df.copy()
    for s in splits.get(symbol, []):
        d = s["date"] if isinstance(s["date"], date) else date.fromisoformat(s["date"])
        pre = out["date"] < d
        for c in ("open", "high", "low", "close"):
            out.loc[pre, c] = out.loc[pre, c] / s["ratio"]
        out.loc[pre, "volume"] = out.loc[pre, "volume"] * s["ratio"]
    return out


def jump_flags(df: pd.DataFrame, threshold: float = 0.35) -> list[date]:
    """Days whose close-to-close move exceeds threshold: candidate unadjusted splits or data errors."""
    r = df["close"].pct_change().abs()
    return list(df.loc[r > threshold, "date"])


def fetch_prices(client, symbol: str, start: date, end: date, *, kind: str = "stock") -> pd.DataFrame:
    fn = client.stock_history_eod if kind == "stock" else client.index_history_eod
    frames = [fn(symbol=symbol, start_date=a, end_date=b) for a, b in date_chunks(start, end)]
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        raise ValueError(f"no EOD data for {symbol} {start}..{end}")
    return normalize_eod(pd.concat(frames, ignore_index=True))


def load_prices(
    symbol: str, start: date, end: date, *, kind: str = "stock", cache_dir: Path = CACHE_DIR,
    client_factory: Callable[[], object] | None = None, adjust: bool = True,
) -> pd.DataFrame:
    """Cached, split-adjusted daily bars. `end` should already be holdout-capped by the caller."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}_{start}_{end}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        if client_factory is None:
            from thetadata import ThetaClient

            client_factory = lambda: ThetaClient(dotenv_path=Path.home() / ".thetadata" / "env", dataframe_type="pandas")
        df = fetch_prices(client_factory(), symbol, start, end, kind=kind)
        df.to_parquet(path)
    return apply_splits(df, symbol) if adjust else df
