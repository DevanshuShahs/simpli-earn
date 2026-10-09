"""Per ticker-day option features from one end-of-day chain. Vendor-independent.

Chain columns: expiry (date), strike, right ('C'/'P'), bid, ask, iv, delta, open_interest.
Rules: no extrapolation in strike; none in maturity unless max_extrap_days > 0 (flat, nearest expiry); quotes with bid <= 0 or ask < bid are dropped;
maturity interpolation is linear in total variance (iv^2 * T); skew is linear in delta then in days.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from peer_options.calendar import ET
from peer_options.options.interfaces import OptionDayFeatures

CHAIN_COLUMNS = ["expiry", "strike", "right", "bid", "ask", "iv", "delta", "open_interest"]
NAN = math.nan
MAX_STRIKE_GAP = 0.10  # bracketing strikes must lie within 10% of spot


def clean_chain(chain: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CHAIN_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(f"chain missing columns: {missing}")
    c = chain.copy()
    c["expiry"] = pd.to_datetime(c["expiry"]).dt.date
    c["right"] = c["right"].str.upper().str[0]
    ok = (c["bid"] > 0) & (c["ask"] >= c["bid"]) & (c["iv"] > 0) & np.isfinite(c["iv"]) & c["right"].isin(["C", "P"])
    return c[ok].reset_index(drop=True)


def _interp(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """Linear interpolation, NaN outside the sampled range (no extrapolation)."""
    if len(xs) < 2:
        return NAN
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x < xs[0] or x > xs[-1]:
        return NAN
    return float(np.interp(x, xs, ys))


def _atm_iv(slice_: pd.DataFrame, spot: float) -> float:
    """IV at spot: linear in strike between the bracketing strikes, averaging calls and puts at each strike."""
    by_k = slice_.groupby("strike")["iv"].mean().sort_index()
    below, above = by_k[by_k.index <= spot], by_k[by_k.index >= spot]
    if below.empty or above.empty:
        return NAN
    k0, k1 = below.index[-1], above.index[0]
    if (spot - k0) > MAX_STRIKE_GAP * spot or (k1 - spot) > MAX_STRIKE_GAP * spot:
        return NAN
    return float(below.iloc[-1]) if k0 == k1 else _interp(spot, np.array([k0, k1]), np.array([below.iloc[-1], above.iloc[0]]))


def _put_iv_at_delta(slice_: pd.DataFrame, target: float = -0.25) -> float:
    p = slice_[(slice_["right"] == "P") & slice_["delta"].notna()].drop_duplicates("strike")
    return _interp(target, p["delta"].to_numpy(float), p["iv"].to_numpy(float))


def _bracket(per_expiry: dict[date, float], asof: date, target_days: int, max_extrap_days: int = 0):
    """Expiries bracketing the target. With max_extrap_days > 0 and only one side available within that many days,
    the nearest expiry is returned on both sides (flat extrapolation: the target inherits that expiry's value)."""
    pts = sorted((((e - asof).days), v) for e, v in per_expiry.items() if not math.isnan(v) and (e - asof).days >= 1)
    lo = [p for p in pts if p[0] <= target_days]
    hi = [p for p in pts if p[0] >= target_days]
    if lo and hi:
        return lo[-1], hi[0]
    if hi and hi[0][0] - target_days <= max_extrap_days:
        return hi[0], hi[0]
    if lo and target_days - lo[-1][0] <= max_extrap_days:
        return lo[-1], lo[-1]
    return None


def constant_maturity_iv(per_expiry_iv: dict[date, float], asof: date, target_days: int, max_extrap_days: int = 0) -> float:
    b = _bracket(per_expiry_iv, asof, target_days, max_extrap_days)
    if b is None:
        return NAN
    (d0, v0), (d1, v1) = b
    if d0 == d1:
        return v0
    w0, w1 = v0**2 * d0, v1**2 * d1  # total variance (the 1/365 cancels in the ratio)
    w = w0 + (w1 - w0) * (target_days - d0) / (d1 - d0)
    return math.sqrt(w / target_days)


def constant_maturity_linear(per_expiry: dict[date, float], asof: date, target_days: int, max_extrap_days: int = 0) -> float:
    b = _bracket(per_expiry, asof, target_days, max_extrap_days)
    if b is None:
        return NAN
    (d0, v0), (d1, v1) = b
    return v0 if d0 == d1 else v0 + (v1 - v0) * (target_days - d0) / (d1 - d0)


def implied_earnings_move(chain: pd.DataFrame, spot: float, reaction_date: date) -> tuple[float, date | None]:
    """ATM straddle mid / spot for the first expiry on or after the reaction session (raw, unadjusted)."""
    exps = sorted(e for e in chain["expiry"].unique() if e >= reaction_date)
    for e in exps:
        s = chain[chain["expiry"] == e]
        k = s.loc[(s["strike"] - spot).abs().idxmin(), "strike"] if not s.empty else None
        c = s[(s["strike"] == k) & (s["right"] == "C")]
        p = s[(s["strike"] == k) & (s["right"] == "P")]
        if len(c) and len(p):
            m = ((c.iloc[0]["bid"] + c.iloc[0]["ask"]) + (p.iloc[0]["bid"] + p.iloc[0]["ask"])) / 2
            return float(m / spot), e
    return NAN, None


def compute_features(
    ticker: str,
    trade_date: date,
    chain: pd.DataFrame,
    spot: float,
    *,
    reaction_date: date | None = None,
    available_after_close: timedelta = timedelta(hours=4),
    max_extrap_days: int = 0,
) -> OptionDayFeatures:
    """Features for one ticker-day. available_at = 16:00 ET + available_after_close (default 20:00 ET)."""
    c = clean_chain(chain)
    c = c[c["expiry"] > trade_date]
    per_atm = {e: _atm_iv(g, spot) for e, g in c.groupby("expiry")}
    per_skew = {e: _put_iv_at_delta(g) - per_atm[e] for e, g in c.groupby("expiry")}
    days = c["expiry"].map(lambda e: (e - trade_date).days)
    near = c[(days >= 15) & (days <= 45) & c["delta"].abs().between(0.3, 0.7)]
    spread = ((near["ask"] - near["bid"]) / ((near["ask"] + near["bid"]) / 2)).median() if len(near) else NAN
    oi_rows = c.loc[(days >= 1) & (days <= 60) & (c["delta"].abs() >= 0.1), "open_interest"]
    oi = float(oi_rows.sum()) if oi_rows.notna().any() else NAN  # NaN when the source has no open interest
    move = implied_earnings_move(c, spot, reaction_date)[0] if reaction_date else None
    avail = datetime.combine(trade_date, time(16, 0), ET) + available_after_close
    return OptionDayFeatures(
        ticker=ticker, trade_date=trade_date, available_at=avail,
        atm_iv_30d=constant_maturity_iv(per_atm, trade_date, 30, max_extrap_days),
        atm_iv_60d=constant_maturity_iv(per_atm, trade_date, 60, max_extrap_days),
        put_skew_25d=constant_maturity_linear(per_skew, trade_date, 30, max_extrap_days),
        bid_ask_pct_mid=float(spread), open_interest=oi,
        implied_earnings_move=None if move is None or math.isnan(move) else move,
    )
