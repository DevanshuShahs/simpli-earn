import math

import pytest

from conftest import utc
from peer_options.links import IndustryMembershipWeights
from peer_options.signals.baseline import HistoryPoint, zscore_vs_history
from peer_options.signals.finbert import label_to_tone
from peer_options.signals.gap import analyst_tone, tone_gap
from peer_options.signals.hedging import hedge_density
from peer_options.signals.peer import peer_signal
from peer_options.signals.tone import LoughranMcDonaldScorer, split_sentences, tone_score
from peer_options.transcripts.parser import parse_transcript
from test_parser import BARE

# SYNTHETIC word lists, not the real Loughran-McDonald dictionary.
SCORER = LoughranMcDonaldScorer(positive={"strong", "improved", "optimistic", "robust"},
                                negative={"weakness", "worried", "decline"})


def hist(vals):
    return [HistoryPoint(utc(2026, 1, i + 1), v) for i, v in enumerate(vals)]


def test_split_sentences():
    assert split_sentences("One thing. Two things! Three?") == ["One thing.", "Two things!", "Three?"]


def test_tone_score_and_empty():
    assert tone_score("Strong quarter. Weakness in Asia.", SCORER) == 0.0
    assert tone_score("Strong results. Improved margins.", SCORER) == 1.0
    assert math.isnan(tone_score("", SCORER))


def test_hedge_density():
    assert hedge_density("we may see approximately flat") == pytest.approx(2 / 5)
    assert math.isnan(hedge_density(""))


def test_gap_uses_executive_speech_only():
    t = parse_transcript(BARE)
    # prepared exec sentences score +1, +1, 0 -> 2/3; Q&A exec sentences score +1, +1 -> 1
    assert tone_gap(t, SCORER) == pytest.approx(2 / 3 - 1)
    # analyst sentences score 0 and -1, scored separately from executives
    assert analyst_tone(t, SCORER) == pytest.approx(-0.5)


def test_baseline_needs_min_four_calls_and_never_imputes():
    r = zscore_vs_history(1.0, hist([0.0, 1.0, 2.0]), as_of=utc(2026, 6, 1))
    assert math.isnan(r.z) and r.n_obs == 3 and "need >= 4" in r.reason


def test_baseline_zscore_and_window_cap():
    h = hist([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    r = zscore_vs_history(9.0, h, as_of=utc(2026, 6, 1))
    assert r.n_obs == 8  # only the latest 8
    assert r.z == pytest.approx((9 - 6.5) / 2.449489742783178)


def test_baseline_excludes_future_history():
    h = hist([1, 2, 3, 4]) + [HistoryPoint(utc(2026, 12, 1), 1000.0)]
    r = zscore_vs_history(2.5, h, as_of=utc(2026, 6, 1))
    assert r.n_obs == 4 and r.z == pytest.approx(0.0)
    assert math.isnan(zscore_vs_history(2.5, h, as_of=utc(2026, 1, 3)).z)  # only 3 visible


def test_baseline_zero_variance():
    assert "zero variance" in zscore_vs_history(1.0, hist([1, 1, 1, 1]), as_of=utc(2026, 6, 1)).reason


def test_industry_link_weights():
    w = IndustryMembershipWeights.from_yaml()
    assert w.weight("TXN", "NXPI") == 1.0   # same sub-industry
    assert w.weight("TSM", "LRCX") == 0.5   # same group, different sub-industry
    with pytest.raises(KeyError):
        w.weight("TSM", "ZZZZ")


def test_peer_signal_scales_and_propagates_nan():
    w = IndustryMembershipWeights.from_yaml()
    assert peer_signal("TSM", "LRCX", 2.0, w) == 1.0
    assert math.isnan(peer_signal("TSM", "LRCX", math.nan, w))


def test_finbert_label_mapping():
    assert label_to_tone("positive", 0.9) == 0.9
    assert label_to_tone("Negative", 0.8) == -0.8
    assert label_to_tone("neutral", 0.99) == 0.0
    with pytest.raises(ValueError):
        label_to_tone("LABEL_0", 0.5)
