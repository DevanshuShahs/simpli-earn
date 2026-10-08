from pathlib import Path

import pytest

from peer_options.signals.hedging import uncertainty_terms_from_master_csv
from peer_options.signals.tone import LoughranMcDonaldScorer

MASTER = Path(__file__).resolve().parents[1] / "data" / "raw" / "LM_MasterDictionary_1993-2025.csv"

# SYNTHETIC mini master dictionary in the real column layout (nonzero = year added).
MINI = """Word,Seq_num,Word Count,Word Proportion,Average Proportion,Std Dev,Doc Count,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal,Constraining,Complexity,Syllables,Source
GOOD,1,1,0,0,0,1,0,2009,0,0,0,0,0,0,1,x
BAD,2,1,0,0,0,1,2009,0,0,0,0,0,0,0,1,x
MAYBE,3,1,0,0,0,1,0,0,2009,0,0,0,0,0,2,x
NEUTRAL,4,1,0,0,0,1,0,0,0,0,0,0,0,0,2,x
"""


def test_loader_columns(tmp_path):
    p = tmp_path / "mini.csv"
    p.write_text(MINI)
    s = LoughranMcDonaldScorer.from_master_csv(str(p))
    assert s.positive == {"good"} and s.negative == {"bad"}
    assert uncertainty_terms_from_master_csv(str(p)) == {"maybe"}
    assert s.score_sentences(["Good and bad.", "Good results.", "Neutral."]) == [0.0, 1.0, 0.0]


@pytest.mark.skipif(not MASTER.exists(), reason="LM master dictionary not downloaded")
def test_real_master_dictionary_sanity():
    s = LoughranMcDonaldScorer.from_master_csv(str(MASTER))
    unc = uncertainty_terms_from_master_csv(str(MASTER))
    assert {"loss", "decline"} <= s.negative
    assert {"improved", "strong"} <= s.positive
    assert {"uncertain", "may"} <= unc
    assert 1500 < len(s.negative) < 4000 and 200 < len(s.positive) < 600 and 200 < len(unc) < 500
