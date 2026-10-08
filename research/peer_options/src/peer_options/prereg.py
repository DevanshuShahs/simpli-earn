"""Pre-registration lock and holdout guard.

lock():   hash prereg.yaml and write prereg.lock (refuses if the holdout is undefined).
verify(): recompute the hash; study code must call this before producing results.
HoldoutGuard: any access to dates inside the holdout needs allow_holdout=True and is logged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg.yaml"
LOCK = ROOT / "prereg.lock"
ACCESS_LOG = ROOT / "logs" / "holdout_access.jsonl"


class PreregError(RuntimeError):
    pass


class HoldoutAccessError(PreregError):
    pass


def canonical_hash(path: Path = PREREG) -> str:
    doc = yaml.safe_load(path.read_text())
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def holdout_bounds(path: Path = PREREG) -> tuple[date, date]:
    h = yaml.safe_load(path.read_text()).get("holdout") or {}
    if not h.get("start") or not h.get("end"):
        raise PreregError("holdout start/end undefined in prereg.yaml")
    s, e = (d if isinstance(d, date) else date.fromisoformat(d) for d in (h["start"], h["end"]))
    if s > e:
        raise PreregError("holdout start after end")
    return s, e


def lock(path: Path = PREREG, lock_path: Path = LOCK) -> str:
    holdout_bounds(path)  # fails closed when undefined
    if lock_path.exists():
        raise PreregError(f"{lock_path.name} already exists; locks are never overwritten")
    digest = canonical_hash(path)
    lock_path.write_text(digest + "\n")
    return digest


def verify(path: Path = PREREG, lock_path: Path = LOCK) -> str:
    if not lock_path.exists():
        raise PreregError("prereg is not locked; no results may be produced")
    expected = lock_path.read_text().strip()
    actual = canonical_hash(path)
    if actual != expected:
        raise PreregError(f"prereg.yaml hash {actual[:12]} != locked {expected[:12]}")
    return actual


@dataclass
class HoldoutGuard:
    start: date
    end: date
    log_path: Path = ACCESS_LOG

    @classmethod
    def from_prereg(cls, path: Path = PREREG, log_path: Path = ACCESS_LOG) -> "HoldoutGuard":
        s, e = holdout_bounds(path)
        return cls(s, e, log_path)

    def check(self, dates: list[date], *, allow_holdout: bool = False, purpose: str = "") -> None:
        hit = [d for d in dates if self.start <= d <= self.end]
        if not hit:
            return
        if not allow_holdout:
            raise HoldoutAccessError(
                f"{len(hit)} date(s) fall in holdout {self.start}..{self.end}; pass allow_holdout=True"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps({
                "accessed_at": datetime.now(timezone.utc).isoformat(),
                "purpose": purpose,
                "n_dates": len(hit),
                "first": min(hit).isoformat(),
                "last": max(hit).isoformat(),
            }) + "\n")
