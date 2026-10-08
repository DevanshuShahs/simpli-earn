"""Uncertainty / hedging density.

DEFAULT_HEDGE_TERMS is a small hand-written fallback, not the Loughran-McDonald Uncertainty
list. The real study passes `uncertainty_terms_from_master_csv(...)`.
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


def uncertainty_terms_from_master_csv(path: str) -> frozenset[str]:
    """Loughran-McDonald Uncertainty word list (lowercased) from the master dictionary CSV."""
    import csv

    with open(path, newline="") as fh:
        return frozenset(
            row["Word"].lower() for row in csv.DictReader(fh) if float(row["Uncertainty"] or 0) > 0
        )
