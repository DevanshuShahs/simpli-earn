import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from peer_options.options.features import (
    clean_chain, compute_features, constant_maturity_iv, implied_earnings_move,
)

# ALL DATA HERE IS SYNTHETIC: Black-Scholes prices from a made-up smile, not market data.
SPOT, TRADE = 100.0, date(2026, 3, 2)


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(right, s, k, t, v):
    d1 = (math.log(s / k) + 0.5 * v * v * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    if right == "C":
        return s * _cdf(d1) - k * _cdf(d2), _cdf(d1)
    return k * _cdf(-d2) - s * _cdf(-d1), _cdf(d1) - 1


def make_chain(vol_fn, expiries_days, strikes=None, half_spread=0.02):
    strikes = strikes if strikes is not None else np.arange(70, 130.01, 1.0)
    rows = []
    for d in expiries_days:
        t = d / 365
        for k in strikes:
            v = vol_fn(k, d)
            for r in "CP":
                px, delta = bs(r, SPOT, k, t, v)
                rows.append(dict(expiry=TRADE + timedelta(days=d), strike=k, right=r, bid=max(px - half_spread, 0.01),
                                 ask=px + half_spread, iv=v, delta=delta, open_interest=100))
    return pd.DataFrame(rows)


def test_flat_vol_recovered_at_constant_maturity_and_zero_skew():
    f = compute_features("SYN", TRADE, make_chain(lambda k, d: 0.30, [14, 28, 45, 90]), SPOT)
    assert f.atm_iv_30d == pytest.approx(0.30, abs=1e-6)  # interpolated between 28d and 45d
    assert f.atm_iv_60d == pytest.approx(0.30, abs=1e-6)  # between 45d and 90d
    assert f.put_skew_25d == pytest.approx(0.0, abs=1e-6)
    assert f.implied_earnings_move is None


def test_variance_interpolation_between_different_vols():
    chain = make_chain(lambda k, d: 0.20 if d == 14 else 0.40, [14, 44])
    f = compute_features("SYN", TRADE, chain, SPOT)
    w = (0.20**2 * 14 + (0.40**2 * 44 - 0.20**2 * 14) * (30 - 14) / (44 - 14))
    assert f.atm_iv_30d == pytest.approx(math.sqrt(w / 30), abs=1e-6)


def test_put_skew_positive_for_downward_sloping_smile():
    smile = lambda k, d: 0.30 - 0.25 * math.log(k / SPOT)  # lower strikes -> higher IV
    f = compute_features("SYN", TRADE, make_chain(smile, [28, 45]), SPOT)
    assert 0.0 < f.put_skew_25d < 0.1
    # analytic check: find the strike with put delta -0.25 and compare its IV to ATM
    k25 = next(k for k in np.arange(130, 70, -0.01) if bs("P", SPOT, k, 30 / 365, smile(k, 30))[1] >= -0.25)
    assert f.put_skew_25d == pytest.approx(smile(k25, 30) - 0.30, abs=0.004)


def test_no_extrapolation_when_maturity_not_bracketed():
    f = compute_features("SYN", TRADE, make_chain(lambda k, d: 0.3, [7, 14, 21]), SPOT)  # nothing >= 30d
    assert math.isnan(f.atm_iv_30d) and math.isnan(f.atm_iv_60d) and math.isnan(f.put_skew_25d)
    assert math.isnan(constant_maturity_iv({}, TRADE, 30))


def test_no_bracketing_strike_gives_nan():
    chain = make_chain(lambda k, d: 0.3, [28, 45], strikes=np.arange(80, 94, 1.0))  # all strikes far below spot
    assert math.isnan(compute_features("SYN", TRADE, chain, SPOT).atm_iv_30d)


def test_bad_quotes_are_dropped():
    chain = make_chain(lambda k, d: 0.3, [28, 45])
    bad = chain.copy()
    bad.loc[bad["strike"].between(98, 102), "bid"] = 0.0  # zero-bid ATM quotes
    assert len(clean_chain(bad)) < len(clean_chain(chain))
    crossed = chain.copy(); crossed["ask"] = crossed["bid"] - 0.01
    assert clean_chain(crossed).empty
    with pytest.raises(ValueError, match="missing"):
        clean_chain(chain.drop(columns=["iv"]))


def test_spread_open_interest_and_availability():
    f = compute_features("SYN", TRADE, make_chain(lambda k, d: 0.3, [28, 45]), SPOT)
    assert 0 < f.bid_ask_pct_mid < 0.05
    assert f.open_interest > 0
    assert f.available_at == datetime(2026, 3, 2, 20, 0, tzinfo=f.available_at.tzinfo)
    assert f.available_at.utcoffset() == timedelta(hours=-5)  # EST in early March


def test_only_expiries_after_trade_date_and_implied_move():
    chain = make_chain(lambda k, d: 0.30, [1, 28, 45])
    move, exp = implied_earnings_move(clean_chain(chain), SPOT, TRADE + timedelta(days=2))
    assert exp == TRADE + timedelta(days=28)  # first expiry on/after the reaction session
    straddle = sum(bs(r, SPOT, 100.0, 28 / 365, 0.30)[0] for r in "CP")
    assert move == pytest.approx(straddle / SPOT, abs=0.001)
    f = compute_features("SYN", TRADE, chain, SPOT, reaction_date=TRADE + timedelta(days=2))
    assert f.implied_earnings_move == pytest.approx(move)
