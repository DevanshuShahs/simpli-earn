"""FinBERT tone scorer reusing the repo's sentiment helpers (no edits to sentiment/).

Heavy: needs the optional `finbert` extra (torch, transformers) and downloads
ProsusAI/finbert (~440 MB) on first use. Not exercised by the test suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_SENTIMENT = Path(__file__).resolve().parents[5] / "sentiment"
FINBERT_MODEL = "ProsusAI/finbert"


def label_to_tone(label: str, score: float) -> float:
    """Map a FinBERT label+confidence to [-1, 1]."""
    lab = label.lower()
    if lab == "positive":
        return score
    if lab == "negative":
        return -score
    if lab == "neutral":
        return 0.0
    raise ValueError(f"unexpected FinBERT label {label!r}")


class FinBertScorer:
    name = "finbert"

    def __init__(self, model_name: str = FINBERT_MODEL, hf_token: str | None = None) -> None:
        sys.path.insert(0, str(_REPO_SENTIMENT))
        # Reuse the existing loader/inference rather than reimplementing them.
        from text_insights_relevant import load_classifier, run_inference  # type: ignore

        self._run = run_inference
        self._clf, _ = load_classifier(model_name, hf_token, None)

    def score_sentences(self, sentences: list[str]) -> list[float]:
        return [label_to_tone(r["label"], r["score"]) for r in self._run(self._clf, sentences)]
