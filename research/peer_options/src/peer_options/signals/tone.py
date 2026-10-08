"""Tone scoring behind a ToneScorer protocol (FinBERT, Loughran-McDonald, or a test stub)."""
from __future__ import annotations

import math
import re
from typing import Protocol

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.replace("\n", " ")) if s.strip()]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class ToneScorer(Protocol):
    name: str

    def score_sentences(self, sentences: list[str]) -> list[float]:
        """Per-sentence tone in [-1, 1] (positive minus negative)."""
        ...


def tone_score(text: str, scorer: ToneScorer) -> float:
    """Mean sentence tone; NaN for empty text."""
    sents = split_sentences(text)
    if not sents:
        return math.nan
    vals = scorer.score_sentences(sents)
    return sum(vals) / len(vals)


class LoughranMcDonaldScorer:
    """(pos - neg) / (pos + neg) per sentence, using a supplied LM word list.

    The official Loughran-McDonald master dictionary is NOT bundled; load it with
    `from_master_csv`. Tests use a tiny synthetic list passed to the constructor.
    """

    name = "loughran_mcdonald"

    def __init__(self, positive: set[str], negative: set[str]) -> None:
        self.positive, self.negative = positive, negative

    @classmethod
    def from_master_csv(cls, path: str) -> "LoughranMcDonaldScorer":
        import csv

        pos: set[str] = set()
        neg: set[str] = set()
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                w = row["Word"].lower()
                if float(row.get("Positive", 0) or 0) > 0:
                    pos.add(w)
                if float(row.get("Negative", 0) or 0) > 0:
                    neg.add(w)
        return cls(pos, neg)

    def score_sentences(self, sentences: list[str]) -> list[float]:
        out = []
        for s in sentences:
            toks = tokenize(s)
            p = sum(t in self.positive for t in toks)
            n = sum(t in self.negative for t in toks)
            out.append((p - n) / (p + n) if p + n else 0.0)
        return out
