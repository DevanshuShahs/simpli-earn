from datetime import date

import pandas as pd
import pytest

from conftest import utc
from peer_options.options.csv_adapter import FileOptionsProvider
from peer_options.options.interfaces import FEATURE_COLUMNS

# SYNTHETIC rows for testing the adapter only. Not market data.
ROWS = [
    ["LRCX", "2026-10-15", "2026-10-15T21:00:00Z", 0.40, 0.38, 0.05, 0.02, 1000, None],
    ["LRCX", "2026-10-16", "2026-10-16T21:00:00Z", 0.41, 0.39, 0.06, 0.02, 1100, 0.07],
]


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "synthetic.csv"
    pd.DataFrame(ROWS, columns=FEATURE_COLUMNS).to_csv(p, index=False)
    return p


def test_filters_by_as_of(path):
    prov = FileOptionsProvider(path)
    a, b = date(2026, 10, 1), date(2026, 10, 31)
    assert len(prov.features("LRCX", a, b, as_of=utc(2026, 10, 16, 12))) == 1
    rows = prov.features("LRCX", a, b, as_of=utc(2026, 10, 17))
    assert [r.implied_earnings_move for r in rows] == [None, 0.07]


def test_requires_tz_aware_as_of_and_columns(path, tmp_path):
    with pytest.raises(ValueError):
        FileOptionsProvider(path).features("LRCX", date(2026, 10, 1), date(2026, 10, 31),
                                           as_of=utc(2026, 10, 17).replace(tzinfo=None))
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"ticker": ["X"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing"):
        FileOptionsProvider(bad)
