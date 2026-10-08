"""Uncertainty / hedging density.

DEFAULT_HEDGE_TERMS is a small hand-written seed list, not the Loughran-McDonald
Uncertainty list. Pass the LM list (from the master dictionary) for the real study.
"""
from __future__ import annotations

import math

from peer_options.signals.tone import tokenize

DEFAULT_HEDGE_TERMS = frozenset(
    "may might could possibly perhaps approximately roughly uncertain uncertainty "
    "depends depend assume assuming believe expect anticipate appears probably "
    "unclear volatile volatility variability fluctuate".split()
)


def hedge_density(text: str, terms: frozenset[str] | set[str] = DEFAULT_HEDGE_TERMS) -> float:
    """Fraction of tokens that are hedge terms; NaN for empty text."""
    toks = tokenize(text)
    if not toks:
        return math.nan
    return sum(t in terms for t in toks) / len(toks)
