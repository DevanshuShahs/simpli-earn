import json
from datetime import date

import pytest
import yaml

from peer_options import prereg
from peer_options.prereg import HoldoutAccessError, HoldoutGuard, PreregError

DOC = {"version": 1, "holdout": {"start": "2026-01-01", "end": "2026-03-31"}, "x": [1, 2]}


@pytest.fixture
def files(tmp_path):
    p = tmp_path / "prereg.yaml"
    p.write_text(yaml.safe_dump(DOC))
    return p, tmp_path / "prereg.lock"


def test_draft_prereg_cannot_be_locked_until_holdout_defined(tmp_path):
    with pytest.raises(PreregError, match="holdout"):
        prereg.lock(prereg.PREREG, tmp_path / "l")
    assert not prereg.LOCK.exists()  # the draft in the repo is intentionally unlocked


def test_lock_verify_and_tamper_detection(files):
    p, lk = files
    with pytest.raises(PreregError, match="not locked"):
        prereg.verify(p, lk)
    d = prereg.lock(p, lk)
    assert prereg.verify(p, lk) == d
    p.write_text(yaml.safe_dump({**DOC, "x": [1, 3]}))  # one-value edit
    with pytest.raises(PreregError, match="hash"):
        prereg.verify(p, lk)


def test_lock_never_overwritten(files):
    p, lk = files
    prereg.lock(p, lk)
    with pytest.raises(PreregError, match="already exists"):
        prereg.lock(p, lk)


def test_hash_ignores_key_order_and_formatting(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("a: 1\nb: [1, 2]\n")
    b.write_text("b:\n  - 1\n  - 2\na: 1\n")
    assert prereg.canonical_hash(a) == prereg.canonical_hash(b)


def test_holdout_guard_blocks_and_logs(files, tmp_path):
    p, _ = files
    g = HoldoutGuard.from_prereg(p, tmp_path / "log" / "a.jsonl")
    g.check([date(2025, 12, 31), date(2026, 4, 1)])  # outside: fine, no log
    assert not (tmp_path / "log" / "a.jsonl").exists()
    with pytest.raises(HoldoutAccessError):
        g.check([date(2026, 2, 1)])
    g.check([date(2026, 2, 1)], allow_holdout=True, purpose="final eval")
    rows = [json.loads(x) for x in (tmp_path / "log" / "a.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["purpose"] == "final eval" and rows[0]["n_dates"] == 1
