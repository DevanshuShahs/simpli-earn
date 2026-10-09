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
- Tone: Loughran-McDonald dictionary only (master dictionary 1993-2025 in `data/raw/`, gitignored; re-download
  from sraf.nd.edu). `signals/finbert.py` is kept as an unused, untested optional adapter (the repo has no FinBERT,
  only SubjECTiveQA relevance/specificity models). Hedging uses the LM Uncertainty list.
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

## Transcript pipeline
`config/tsmc_pdf_to_txt.py` converts TSMC IR (LSEG StreetEvents) PDFs to the parser's layout and appends
`data/raw/transcripts/manifest.csv` (including `published_utc` read from the LSEG footer). Needs `pymupdf`
(pypdf drops word spaces on these PDFs). Text marked "(added by company after the call)" is removed.
TSMC's site blocks scripted downloads (Cloudflare), so PDFs are saved by hand into `data/raw/pdfs/`.
Open issue: the 2Q26 transcript was generated 2026-07-17 05:19Z, ~23h after the 06:00Z call, not 2h after.
Events still use call end + 120 min for `signal_available_at`; decide whether to use `published_utc` instead.

## Needs from the user
- Event dates: done for TSM->KLAC, TSM->LRCX, TXN->NXPI/STM/ON/MCHP (28 events from `data/seeds/*_events.csv`).
  Two live events have no date yet and are pending: TXN->ON and TXN->MCHP (3Q26). Re-check the IR calendars.
  onsemi has reported after the close since 4Q25; the live date is unknown.
- 4 STM events cannot be traded pre-earnings (report the morning after TI's evening call): window 0 or -1.
- Transcripts for >= 4 (ideally 8) prior calls per announcer (TSMC from 3Q24, TXN likewise) in `data/raw/transcripts/`.
- Holdout period and `min_events` in prereg.yaml (required before it can be locked).
- Ticker -> sector ETF map when option data work starts.

## Run
`cd research/peer_options && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest`
