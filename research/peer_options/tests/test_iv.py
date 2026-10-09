import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from peer_options.options.features import compute_features
from peer_options.options.iv import black76, chain_with_iv, forward_delta, implied_forward, implied_vol
from peer_options.options.theta_eod import (
    OptionEodStore, needed_expirations, normalize_option_eod, rate_on,
)

# ALL DATA HERE IS SYNTHETIC (Black-76 prices from made-up vols), not market data.
SPOT, ASOF, R = 100.0, date(2026, 3, 2), 0.04


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("sigma", [0.08, 0.30, 1.2])
def test_implied_vol_roundtrip(right, sigma):
    F, K, T = 101.0, 105.0, 45 / 365
    px = black76(F, K, T, sigma, R, right)
    assert implied_vol(px, F, K, T, R, right) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_violations():
    F, K, T = 100.0, 100.0, 30 / 365
    assert math.isnan(implied_vol(0.0, F, K, T, R, "C"))                      # no time value
    assert math.isnan(implied_vol(1e6, F, K, T, R, "C"))                      # above the upper bound
    deep = math.exp(-R * T) * (F - 50.0)
    assert math.isnan(implied_vol(deep - 0.01, F, 50.0, T, R, "C"))           # below intrinsic
    assert math.isnan(implied_vol(float("nan"), F, K, T, R, "P"))


def test_forward_delta_put_call_relation():
    c, p = forward_delta(100, 95, 0.1, 0.3, "C"), forward_delta(100, 95, 0.1, 0.3, "P")
    assert c - p == pytest.approx(1.0) and 0 < c < 1


def test_implied_forward_from_parity():
    F, T = 102.0, 40 / 365
    strikes = [96.0, 99.0, 102.0, 105.0, 108.0]
    calls = {k: black76(F, k, T, 0.3, R, "C") for k in strikes}
    puts = {k: black76(F, k, T, 0.3, R, "P") for k in strikes}
    assert implied_forward(calls, puts, SPOT, T, R) == pytest.approx(F, abs=1e-6)
    assert math.isnan(implied_forward({}, puts, SPOT, T, R))


def synthetic_eod(vol_fn, days=(30, 60), F_by_days=lambda d: 100.4, half=0.02):
    rows = []
    for d in days:
        T, F = d / 365, F_by_days(d)
        for k in np.arange(70.0, 130.01, 2.5):
            for right in "CP":
                px = black76(F, k, T, vol_fn(k, d), R, right)
                rows.append(dict(date=ASOF, expiry=ASOF + timedelta(days=d), strike=float(k), right=right,
                                 bid=max(px - half, 0.01), ask=px + half, close=px, volume=10.0))
    return pd.DataFrame(rows)


def test_chain_with_iv_recovers_flat_vol_and_features():
    chain = chain_with_iv(synthetic_eod(lambda k, d: 0.30), SPOT, ASOF, R * 100)
    assert not chain.empty and chain["open_interest"].isna().all()
    otm = chain[(chain["strike"] >= 100.4) & (chain["right"] == "C")]
    assert otm["iv"].between(0.28, 0.32).all()           # half-cent quote noise only
    f = compute_features("SYN", ASOF, chain, SPOT)
    assert f.atm_iv_30d == pytest.approx(0.30, abs=0.01) and abs(f.put_skew_25d) < 0.01
    assert math.isnan(f.open_interest)                    # no OI source -> NaN, not 0


def test_chain_with_iv_sees_downward_smile_as_positive_put_skew():
    smile = lambda k, d: 0.30 - 0.25 * math.log(k / 100.4)
    f = compute_features("SYN", ASOF, chain_with_iv(synthetic_eod(smile), SPOT, ASOF, R * 100), SPOT)
    assert f.atm_iv_30d == pytest.approx(0.30, abs=0.015) and 0.01 < f.put_skew_25d < 0.1


def test_bad_quotes_and_wide_spreads_are_excluded():
    eod = synthetic_eod(lambda k, d: 0.30)
    eod.loc[eod["strike"].between(98, 103), "bid"] = 0.0           # zero bids around ATM
    chain = chain_with_iv(eod, SPOT, ASOF, R * 100)
    assert not chain["strike"].between(98, 103).any()
    wide = synthetic_eod(lambda k, d: 0.30, half=50.0)             # absurdly wide market everywhere
    assert chain_with_iv(wide, SPOT, ASOF, R * 100).empty


def test_normalize_option_eod_uses_et_date():
    raw = pd.DataFrame({"created": pd.to_datetime(["2026-07-06 17:19:18-04:00"] * 2),
                        "expiration": ["2026-07-17"] * 2, "strike": [128.0, 128.0], "right": ["CALL", "PUT"],
                        "bid": [1.0, 0.1], "ask": [1.2, 0.2], "close": [1.1, 0.15], "volume": [3, 4]})
    df = normalize_option_eod(raw)
    assert list(df["right"]) == ["C", "P"] and df.iloc[0]["date"] == date(2026, 7, 6)
    assert df.iloc[0]["expiry"] == date(2026, 7, 17)
    assert normalize_option_eod(None).empty


def test_needed_expirations_window():
    exps = [ASOF + timedelta(days=d) for d in (3, 10, 45, 119, 150)]
    got = needed_expirations(exps, [ASOF], min_dte=7, max_dte=120)
    assert got == [ASOF + timedelta(days=d) for d in (10, 45, 119)]


def test_rate_on_never_uses_future_dates():
    s = pd.Series([3.6, 3.7], index=[date(2026, 7, 13), date(2026, 7, 15)])
    assert rate_on(s, date(2026, 7, 14)) == 3.6 and rate_on(s, date(2026, 7, 16)) == 3.7
    with pytest.raises(ValueError):
        rate_on(s, date(2026, 7, 1))


def test_store_caches_and_tolerates_no_data(tmp_path):
    calls = []

    class NoDataFoundError(Exception):
        pass

    class Fake:
        def option_history_eod(self, **kw):
            calls.append(kw)
            if kw["expiration"] == date(2026, 8, 1):
                raise NoDataFoundError("none")
            return pd.DataFrame({"created": pd.to_datetime(["2026-07-06 17:19:18-04:00"]), "expiration": ["2026-07-17"],
                                 "strike": [100.0], "right": ["CALL"], "bid": [1.0], "ask": [1.1], "close": [1.0], "volume": [1]})

    st = OptionEodStore(client_factory=Fake, cache_dir=tmp_path)
    a, b = date(2026, 7, 6), date(2026, 7, 10)
    assert len(st.chain_days("X", date(2026, 7, 17), a, b)) == 1
    st.chain_days("X", date(2026, 7, 17), a, b)
    assert len(calls) == 1                                          # second call served from cache
    assert st.chain_days("X", date(2026, 8, 1), a, b).empty         # vendor "no data" -> empty, cached
    assert calls[0]["end_date"] == b                                # end capped at expiration if earlier


def test_ensure_days_requests_only_uncovered_days(tmp_path):
    calls = []

    class Fake:
        def option_history_eod(self, **kw):
            calls.append((kw["start_date"], kw["end_date"]))
            return pd.DataFrame({"created": pd.to_datetime([f"{kw['start_date']} 17:15:00-04:00"]), "expiration": ["2026-08-21"],
                                 "strike": [100.0], "right": ["CALL"], "bid": [1.0], "ask": [1.1], "close": [1.0], "volume": [1]})

    st = OptionEodStore(client_factory=Fake, cache_dir=tmp_path)
    exp = date(2026, 8, 21)
    st.ensure_days("X", exp, [date(2026, 7, 15), date(2026, 7, 16)])
    st.ensure_days("X", exp, [date(2026, 7, 15), date(2026, 7, 16)])             # fully covered: no request
    st.ensure_days("X", exp, [date(2026, 7, 16), date(2026, 7, 20), date(2026, 7, 21)])  # only the new days
    assert calls == [(date(2026, 7, 15), date(2026, 7, 16)), (date(2026, 7, 20), date(2026, 7, 21))]
    assert set(st.load_expiration("X", exp)["date"]) == {date(2026, 7, 15), date(2026, 7, 20)}
