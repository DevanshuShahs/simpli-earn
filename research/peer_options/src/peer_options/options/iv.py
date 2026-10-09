"""Implied vol from end-of-day option quotes (no vendor greeks needed).

Method: per expiry, the forward comes from put-call parity at the strikes nearest spot; IV is Black-76 inverted on the
mid of each strike's out-of-the-money option; calls and puts at a strike share that IV (parity). Delta is the
forward delta N(d1) (undiscounted). Limits: European pricing on American options (early-exercise premium ignored,
small for near-the-money OTM quotes) and no dividend/borrow inputs (absorbed in the parity forward).
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from peer_options.options.features import CHAIN_COLUMNS

NAN = math.nan


def _n(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black76(F: float, K: float, T: float, sigma: float, r: float, right: str) -> float:
    df = math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = df * (F * _n(d1) - K * _n(d2))
    return call if right == "C" else call - df * (F - K)


def forward_delta(F: float, K: float, T: float, sigma: float, right: str) -> float:
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _n(d1) if right == "C" else _n(d1) - 1.0


def implied_vol(price: float, F: float, K: float, T: float, r: float, right: str,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Bisection on Black-76; NaN when the price is outside no-arbitrage bounds (no extrapolation, no clamping)."""
    df = math.exp(-r * T)
    intrinsic = df * max(F - K, 0.0) if right == "C" else df * max(K - F, 0.0)
    upper = df * F if right == "C" else df * K
    if not (math.isfinite(price) and intrinsic + 1e-10 < price < upper):
        return NAN
    if black76(F, K, T, hi, r, right) < price or black76(F, K, T, lo, r, right) > price:
        return NAN
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if black76(F, K, T, mid, r, right) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)


def implied_forward(calls: dict[float, float], puts: dict[float, float], spot: float, T: float, r: float,
                    n: int = 4) -> float:
    """Median of K + e^{rT}(C - P) over the n strikes nearest spot that have both a call and a put mid."""
    common = sorted(set(calls) & set(puts), key=lambda k: abs(k - spot))[:n]
    if not common:
        return NAN
    return float(np.median([k + math.exp(r * T) * (calls[k] - puts[k]) for k in common]))


def chain_with_iv(eod: pd.DataFrame, spot: float, asof: date, rate_pct: float, *,
                  max_rel_spread: float = 0.5, strike_band: float = 0.5) -> pd.DataFrame:
    """One trade date of normalized EOD quotes (expiry, strike, right C/P, bid, ask) -> features.py chain frame.

    Quotes with bid <= 0, ask < bid or relative spread above max_rel_spread are ignored; strikes further than
    strike_band from spot are ignored. open_interest is NaN (not available on the Free plan).
    """
    r = rate_pct / 100.0
    out = []
    q = eod[(eod["bid"] > 0) & (eod["ask"] >= eod["bid"])].copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2
    q = q[(q["ask"] - q["bid"]) / q["mid"] <= max_rel_spread]
    q = q[(q["strike"] / spot - 1).abs() <= strike_band]
    for exp, g in q.groupby("expiry"):
        T = (exp - asof).days / 365.0
        if T <= 0:
            continue
        calls = g[g["right"] == "C"].set_index("strike")["mid"].to_dict()
        puts = g[g["right"] == "P"].set_index("strike")["mid"].to_dict()
        F = implied_forward(calls, puts, spot, T, r)
        if not math.isfinite(F) or F <= 0:
            continue
        iv_by_strike: dict[float, float] = {}
        for k in sorted(set(calls) | set(puts)):
            side, px = ("C", calls.get(k)) if k >= F else ("P", puts.get(k))
            if px is None:
                continue
            v = implied_vol(px, F, k, T, r, side)
            if math.isfinite(v):
                iv_by_strike[k] = v
        for row in g.itertuples(index=False):
            v = iv_by_strike.get(row.strike)
            if v is None:
                continue
            out.append(dict(expiry=exp, strike=row.strike, right=row.right, bid=row.bid, ask=row.ask, iv=v,
                            delta=forward_delta(F, row.strike, T, v, row.right), open_interest=NAN))
    return pd.DataFrame(out, columns=CHAIN_COLUMNS)
