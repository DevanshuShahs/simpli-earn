import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev

import pytest

from peer_options import prereg
from peer_options.events import DateStatus, LinkType, make_event
from peer_options.links import IndustryMembershipWeights
from peer_options.scoring import LM_MASTER, ScoringError, SignalLog, score_call
from peer_options.signals.tone import LoughranMcDonaldScorer

# SYNTHETIC calls: sentences are 'good' (+1), 'bad' (-1) or neutral (0) under this toy word list.
SCORER = LoughranMcDonaldScorer(positive={"good"}, negative={"bad"})
UNC = frozenset({"maybe"})
UTC = timezone.utc
STARTS = [datetime(2025, 1, 16, 6, tzinfo=UTC), datetime(2025, 4, 17, 6, tzinfo=UTC),
          datetime(2025, 7, 17, 6, tzinfo=UTC), datetime(2025, 10, 16, 6, tzinfo=UTC),
          datetime(2026, 1, 15, 6, tzinfo=UTC), datetime(2026, 4, 16, 6, tzinfo=UTC)]
WEIGHTS = IndustryMembershipWeights({
    "weights": {"same_sub_industry": 1.0, "same_industry_group": 0.5, "different": 0.0},
    "tickers": {"SYN": {"group": "G", "sub_industry": "A"}, "PEER": {"group": "G", "sub_industry": "B"}},
})


def transcript(g: int, q: int) -> str:
    """g of 4 prepared sentences and q of 4 Q&A executive sentences are 'good'; the rest neutral."""
    sent = lambda n, word: ("Things are %s. " % word) * n
    return (
        "Prepared Remarks:\n\nPat Exec -- Chief Executive Officer\n"
        + sent(g, "good") + sent(4 - g, "neutral")
        + "\n\nQuestions and Answers:\n\nAlex Analyst -- Some Bank, Analyst\nWhat about demand?\n\n"
        "Pat Exec -- Chief Executive Officer\n" + sent(q, "good") + sent(4 - q, "neutral") + "\n"
    )


def make_env(tmp_path: Path, calls: list[tuple[int, int]]):
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    rows = []
    for i, (g, q) in enumerate(calls):
        period = f"P{i}"
        (tdir / f"SYN_{period}.txt").write_text(transcript(g, q))
        rows.append(dict(file=f"SYN_{period}.txt", ticker="SYN", period=period,
                         call_start_utc=STARTS[i].strftime("%Y-%m-%dT%H:%M:%SZ"),
                         published_utc="", source="synthetic"))
    man = tdir / "manifest.csv"
    with man.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return man


def event_for(i: int):
    s = STARTS[i]
    return make_event(
        event_id=f"SYN-PEER-P{i}", announcer="SYN", peer="PEER", link_type=LinkType.INDUSTRY_PEER,
        call_start=s, peer_report=s + timedelta(days=7, hours=15), date_status=DateStatus.CONFIRMED,
        status_as_of=s - timedelta(days=20), source="synthetic")


def run(man, i, **kw):
    return score_call("SYN", f"P{i}", events=[event_for(i)], weights=WEIGHTS, scorer=SCORER,
                      uncertainty_terms=UNC, manifest_path=man, lm_path=None,
                      verify_prereg=lambda: "hash123", **kw)


HIST = [(4, 0), (2, 0), (4, 2), (1, 1)]          # tone 0.5, 0.25, 0.75, 0.25 ; gap 1.0, 0.5, 0.5, 0.0
CUR = (3, 1)                                      # tone 0.5 ; gap 0.5


def test_z_scores_use_only_earlier_calls(tmp_path):
    man = make_env(tmp_path, HIST + [CUR])
    rec = run(man, 4)
    tones, gaps = [0.5, 0.25, 0.75, 0.25], [1.0, 0.5, 0.5, 0.0]
    assert rec["n_prior_calls"] == 4
    assert rec["raw"]["tone"] == pytest.approx(0.5) and rec["raw"]["gap"] == pytest.approx(0.5)
    assert rec["z"]["tone"]["z"] == pytest.approx((0.5 - mean(tones)) / stdev(tones))
    assert rec["z"]["gap"]["z"] == pytest.approx((0.5 - mean(gaps)) / stdev(gaps))
    assert rec["z"]["hedge"]["z"] is None and "zero variance" in rec["z"]["hedge"]["reason"]


def test_later_call_never_changes_earlier_score(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base = run(make_env(tmp_path / "a", HIST + [CUR]), 4)
    with_future = run(make_env(tmp_path / "b", HIST + [CUR, (0, 4)]), 4)  # an extreme call AFTER the scored one
    assert with_future["z"] == base["z"] and with_future["n_prior_calls"] == 4


def test_fewer_than_four_prior_calls_gives_none_not_imputation(tmp_path):
    rec = run(make_env(tmp_path, HIST[:3] + [CUR]), 3)
    assert rec["n_prior_calls"] == 3
    assert rec["z"]["tone"]["z"] is None and "need >= 4" in rec["z"]["tone"]["reason"]
    assert rec["peer_signals"][0]["S_BA"]["tone"] is None


def test_peer_signal_is_link_weight_times_z(tmp_path):
    rec = run(make_env(tmp_path, HIST + [CUR]), 4)
    ps = rec["peer_signals"][0]
    assert ps["event_id"] == "SYN-PEER-P4" and ps["link_weight"] == 0.5
    assert ps["S_BA"]["tone"] == pytest.approx(0.5 * rec["z"]["tone"]["z"])


def test_late_flag_and_timing(tmp_path):
    man = make_env(tmp_path, HIST + [CUR])
    on_time = run(man, 4, scored_at=STARTS[4] + timedelta(minutes=95))
    assert on_time["signal_available_at"] == "2026-01-15T07:30:00Z" and on_time["late"] is False
    assert run(man, 4, scored_at=STARTS[4] + timedelta(hours=23))["late"] is True


def test_unlocked_prereg_blocks_scoring(tmp_path):
    man = make_env(tmp_path, HIST + [CUR])

    def boom():
        raise prereg.PreregError("hash mismatch")

    with pytest.raises(prereg.PreregError):
        score_call("SYN", "P4", events=[], weights=WEIGHTS, scorer=SCORER, uncertainty_terms=UNC,
                   manifest_path=man, lm_path=None, verify_prereg=boom)


def test_missing_transcript_in_manifest(tmp_path):
    with pytest.raises(ScoringError, match="not in"):
        score_call("SYN", "P9", events=[], weights=WEIGHTS, scorer=SCORER, uncertainty_terms=UNC,
                   manifest_path=make_env(tmp_path, HIST + [CUR]), lm_path=None, verify_prereg=lambda: "h")


def test_log_is_hash_chained_append_only_and_one_shot(tmp_path):
    man = make_env(tmp_path, HIST + [CUR, (2, 2)])
    log = SignalLog(tmp_path / "logs" / "signals.jsonl")
    r1 = log.append(run(man, 4))
    r2 = log.append(run(man, 5))
    assert r1["prev_hash"] == "genesis" and r2["prev_hash"] == r1["record_hash"] and log.verify() == 2
    with pytest.raises(ScoringError, match="already logged"):
        log.append(run(man, 4))  # no re-scoring after the fact
    lines = log.path.read_text().splitlines()
    tampered = json.loads(lines[0]); tampered["raw"]["tone"] = 0.99
    log.path.write_text(json.dumps(tampered, sort_keys=True) + "\n" + lines[1] + "\n")
    with pytest.raises(ScoringError, match="broken"):
        log.verify()


@pytest.mark.skipif(not (LM_MASTER.exists() and (LM_MASTER.parent / "transcripts" / "TSM_2Q26.txt").exists()),
                    reason="real transcripts / LM dictionary not present (gitignored)")
def test_real_tsmc_2q26_matches_pilot_table():
    from peer_options.seed import build_seed
    from peer_options.signals.hedging import uncertainty_terms_from_master_csv

    rec = score_call(
        "TSM", "2Q26", events=build_seed()[0], weights=IndustryMembershipWeights.from_yaml(),
        scorer=LoughranMcDonaldScorer.from_master_csv(str(LM_MASTER)),
        uncertainty_terms=uncertainty_terms_from_master_csv(str(LM_MASTER)))
    # 12 prior transcripts exist (2Q23-1Q26); the baseline uses the last 8 (2Q24-1Q26).
    assert rec["n_prior_calls"] == 12 and rec["late"] is True and rec["in_holdout"] is False
    assert rec["z"]["tone"]["z"] == pytest.approx(-0.58, abs=0.01)
    assert rec["z"]["hedge"]["z"] == pytest.approx(0.97, abs=0.01)
    assert rec["z"]["gap"]["z"] == pytest.approx(1.13, abs=0.01)
    assert {p["event_id"] for p in rec["peer_signals"]} == {"TSMC_KLA_2026Q2", "TSMC_LRCX_2026Q2"}
    assert len(rec["transcript_sha256"]) == 64 and rec["prereg_hash"] == prereg.verify()
