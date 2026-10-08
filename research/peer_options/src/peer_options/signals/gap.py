"""Prepared-vs-Q&A tone gap: G = tone_prepared - tone_QA over executive speech."""
from __future__ import annotations

import math

from peer_options.signals.tone import ToneScorer, tone_score
from peer_options.transcripts.parser import ParsedTranscript, Role, Section


def tone_gap(t: ParsedTranscript, scorer: ToneScorer) -> float:
    prepared = tone_score(t.text(Section.PREPARED), scorer)
    qa = tone_score(t.text(Section.QA), scorer)
    if math.isnan(prepared) or math.isnan(qa):
        return math.nan
    return prepared - qa


def analyst_tone(t: ParsedTranscript, scorer: ToneScorer) -> float:
    """Robustness: tone of analyst questions, scored separately from executive speech."""
    return tone_score(t.text(Section.QA, roles=(Role.ANALYST,)), scorer)
