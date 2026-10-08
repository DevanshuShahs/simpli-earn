# Peer-options research module

Question: does announcer A's earnings-call content (tone change, prepared-vs-Q&A gap, hedging) predict
not-yet-reported peer B's implied vol / skew beyond A's own IV change and the industry average?
If the effect is purely industry-level and rational, the signals add nothing and the write-up says so.
End-of-day data only; no broker integration; paper-trade log only.

## Hypotheses (see prereg.yaml, draft, unlocked)
H1 abnormal peer IV change; H2 builds over days 0-5 (tradable core); H3 realized vs implied earnings move;
H4 put-skew change from directional tone; H5 audio stress (deferred).

## Decisions
- Options data: none yet. `OptionsProvider` interface + CSV/Parquet adapter schema only.
- Tone: `ProsusAI/finbert` via the repo's `load_classifier`/`run_inference` (the repo has no FinBERT, only
  SubjECTiveQA relevance/specificity models); Loughran-McDonald as the comparison baseline.
- Transcripts: Seeking Alpha-style text (`RAG/transcripts/`), which has no call end time.
  `call_end = start + 60 min`, flagged `estimated`; `signal_available_at = call_end + 120 min`.
- Audio deferred.

## Layout
`src/peer_options/`: `calendar`, `events`, `seed`, `pit_store`, `prereg`, `links`,
`transcripts/parser`, `signals/{tone,hedging,gap,baseline,peer,finbert}`, `options/*`, `study/interfaces`,
`strategies/interfaces`. Config in `config/`, seed events in `data/seeds/`, tests in `tests/`.

## Rules implemented
- Entry: close of the first trading day whose 09:30 ET open is at/after `signal_available_at`.
  Exit: close before B reports (BMO -> prior close, AMC -> same-day close). Window = trading-day steps.
- Point-in-time: every record has `available_at`; `as_of` is a required keyword; estimated earnings dates are
  never served; a later revision supersedes (and can un-confirm) an earlier date.
- Baselines: previous 4-8 calls, minimum 4, else NaN with a reason (never imputed), history filtered by `as_of`.
- Pre-registration: canonical SHA-256 of prereg.yaml in `prereg.lock`; locking is refused while the holdout is
  undefined; holdout access needs `allow_holdout=True` and is appended to `logs/holdout_access.jsonl`.

## Status after session 1
Done: scaffold, events table + validators + seed, PIT store + lookahead tests, parser, signals + tests,
draft prereg, interfaces for options / event study / trade variants.
Not built: option-data ingestion, abnormal-IV estimation, event-study regression, backtester, audio.

## Needs from the user
- Verify every seed date (`needs_reverify=true`), notably TSM "2:00 ET" (read as 02:00 ET), MCHP Nov 4 vs 5, NXPI Oct 27 vs Nov 2.
- A vetted source for historical (past-quarter) call and report dates; none are seeded.
- Holdout period and `min_events` in prereg.yaml (required before it can be locked).
- Approval to download FinBERT (~440 MB, needs `pip install .[finbert]`) and the official LM master dictionary.
- Ticker -> sector ETF map when option data work starts.

## Run
`cd research/peer_options && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest`
