"""Peer signal: S_BA = link weight x announcer signal. NaN propagates."""
from __future__ import annotations

from peer_options.links import LinkWeights


def peer_signal(announcer: str, peer: str, announcer_signal: float, weights: LinkWeights) -> float:
    return weights.weight(announcer, peer) * announcer_signal
