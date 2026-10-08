"""Link weights between announcer and peer. Only industry membership is implemented.

Hooks: implement LinkWeights for disclosed supplier/customer links or transcript mentions
and pass it wherever a LinkWeights is accepted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml

_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "links.yaml"


class LinkWeights(Protocol):
    def weight(self, announcer: str, peer: str) -> float: ...


class IndustryMembershipWeights:
    def __init__(self, config: dict) -> None:
        self._w = config["weights"]
        self._t = config["tickers"]

    @classmethod
    def from_yaml(cls, path: Path = _DEFAULT) -> "IndustryMembershipWeights":
        return cls(yaml.safe_load(path.read_text()))

    def weight(self, announcer: str, peer: str) -> float:
        a, b = self._t.get(announcer), self._t.get(peer)
        if a is None or b is None:
            raise KeyError(f"no industry mapping for {announcer if a is None else peer}")
        if a["sub_industry"] == b["sub_industry"]:
            return float(self._w["same_sub_industry"])
        if a["group"] == b["group"]:
            return float(self._w["same_industry_group"])
        return float(self._w["different"])


class DisclosedLinkWeights:
    """Hook: weights from disclosed supplier/customer relationships (not implemented)."""

    def weight(self, announcer: str, peer: str) -> float:
        raise NotImplementedError


class TranscriptMentionWeights:
    """Hook: weights from how often A's call mentions B (not implemented)."""

    def weight(self, announcer: str, peer: str) -> float:
        raise NotImplementedError
