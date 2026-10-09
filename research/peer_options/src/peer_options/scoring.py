"""Score one announcer call and log the signals before any outcome exists.

    python -m peer_options.scoring --ticker TSM --period 3Q26 [--dry-run]

Reads the converted transcript via data/raw/transcripts/manifest.csv, scores it with the Loughran-McDonald
dictionary, z-scores each signal against the same firm's earlier calls (as_of = signal_available_at), derives
S_BA = link_weight x signal for every event of that call, and appends one record to logs/signals.jsonl.

Integrity: requires a verified prereg lock; each record carries the hash of the previous record (tamper-evident);
a call can be logged only once (no re-scoring after peeking); `late` flags records scored after the signal
would have been available (e.g. from the next-day vendor transcript rather than the live call).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import yaml

from peer_options import prereg
from peer_options.events import Event
from peer_options.links import IndustryMembershipWeights, LinkWeights
from peer_options.signals.baseline import BaselineResult, HistoryPoint, zscore_vs_history
from peer_options.signals.gap import analyst_tone, tone_gap
from peer_options.signals.hedging import hedge_density, uncertainty_terms_from_master_csv
from peer_options.signals.tone import LoughranMcDonaldScorer, ToneScorer, tone_score
from peer_options.transcripts.parser import Section, parse_transcript

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "transcripts" / "manifest.csv"
LM_MASTER = RAW / "LM_MasterDictionary_1993-2025.csv"
SIGNAL_LOG = ROOT / "logs" / "signals.jsonl"
SIGNALS = ("tone", "hedge", "gap")  # analyst_tone is logged raw as a robustness input, not baselined


class ScoringError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawSignals:
    tone: float          # mean sentence tone of executive speech (prepared + Q&A)
    hedge: float         # LM uncertainty density of executive speech
    gap: float           # tone_prepared - tone_qa, executive speech
    analyst_tone: float  # analyst questions, robustness only


def raw_signals(text: str, scorer: ToneScorer, uncertainty_terms: frozenset[str]) -> RawSignals:
    t = parse_transcript(text)
    executive = t.text(Section.PREPARED) + "\n" + t.text(Section.QA)
    return RawSignals(
        tone=tone_score(executive, scorer),
        hedge=hedge_density(executive, uncertainty_terms),
        gap=tone_gap(t, scorer),
        analyst_tone=analyst_tone(t, scorer),
    )


def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def signal_delay() -> timedelta:
    timing = yaml.safe_load(prereg.PREREG.read_text())["timing"]
    return timedelta(minutes=int(timing["signal_available_minutes_after_start"]))


# ---------------------------------------------------------------- append-only, hash-chained log
class SignalLog:
    def __init__(self, path: Path = SIGNAL_LOG) -> None:
        self.path = path

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _hash(rec: dict) -> str:
        body = {k: v for k, v in rec.items() if k != "record_hash"}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def append(self, rec: dict) -> dict:
        existing = self.records()
        key = (rec["ticker"], rec["period"])
        if any((r["ticker"], r["period"]) == key for r in existing):
            raise ScoringError(f"{key} is already logged; a call is scored once (no re-scoring)")
        rec = {**rec, "prev_hash": existing[-1]["record_hash"] if existing else "genesis"}
        rec["record_hash"] = self._hash(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def verify(self) -> int:
        prev = "genesis"
        recs = self.records()
        for i, r in enumerate(recs):
            if r["prev_hash"] != prev or r["record_hash"] != self._hash(r):
                raise ScoringError(f"signal log broken at record {i}")
            prev = r["record_hash"]
        return len(recs)


# ---------------------------------------------------------------- scoring
def score_call(
    ticker: str,
    period: str,
    *,
    events: list[Event],
    weights: LinkWeights,
    scorer: ToneScorer,
    uncertainty_terms: frozenset[str],
    manifest_path: Path = MANIFEST,
    lm_path: Path | None = LM_MASTER,
    scored_at: datetime | None = None,
    verify_prereg: Callable[[], str] = prereg.verify,
) -> dict:
    prereg_hash = verify_prereg()
    scored_at = scored_at or datetime.now(timezone.utc)
    rows = [r for r in csv.DictReader(manifest_path.open()) if r["ticker"] == ticker]
    cur = next((r for r in rows if r["period"] == period), None)
    if cur is None:
        raise ScoringError(f"{ticker} {period} not in {manifest_path.name}; convert the transcript first")
    start = _parse_utc(cur["call_start_utc"])
    delay = signal_delay()
    available = start + delay

    tdir = manifest_path.parent
    cur_path = tdir / cur["file"]
    raw = raw_signals(cur_path.read_text(), scorer, uncertainty_terms)

    # History: strictly earlier calls of the same firm, each visible at its own availability time.
    hist: dict[str, list[HistoryPoint]] = {k: [] for k in SIGNALS}
    n_prior = 0
    for r in rows:
        s0 = _parse_utc(r["call_start_utc"])
        if s0 >= start:
            continue  # the call itself and anything later is never baseline
        h = raw_signals((tdir / r["file"]).read_text(), scorer, uncertainty_terms)
        n_prior += 1
        for k in SIGNALS:
            hist[k].append(HistoryPoint(s0 + delay, getattr(h, k)))

    z: dict[str, BaselineResult] = {
        k: zscore_vs_history(getattr(raw, k), hist[k], as_of=available) for k in SIGNALS
    }

    mine = [e for e in events if e.announcer == ticker and e.announcer_call_start == start]
    peers = []
    for e in sorted(mine, key=lambda e: e.event_id):
        w = weights.weight(ticker, e.peer)
        peers.append({
            "event_id": e.event_id, "peer": e.peer, "link_type": e.link_type.value, "link_weight": w,
            "window_trading_days": e.window_trading_days,
            "S_BA": {k: _num(w * z[k].z) for k in SIGNALS},
        })

    try:
        b = prereg.holdout_bounds()
        in_holdout = prereg.is_holdout_event(start, b)
    except prereg.PreregError:
        in_holdout = False
    return {
        "kind": "announcer_signals",
        "ticker": ticker, "period": period,
        "call_start_utc": cur["call_start_utc"],
        "signal_available_at": available.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scored_at": scored_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "late": scored_at > available + timedelta(minutes=30),
        "in_holdout": in_holdout,
        "transcript_file": cur["file"],
        "transcript_sha256": _sha256(cur_path),
        "transcript_published_utc": cur.get("published_utc", ""),
        "prereg_hash": prereg_hash,
        "scorer": scorer.name,
        "lm_dictionary_sha256": _sha256(lm_path) if lm_path and lm_path.exists() else None,
        "n_prior_calls": n_prior,
        "raw": {k: _num(v) for k, v in asdict(raw).items()},
        "z": {k: {"z": _num(v.z), "n_obs": v.n_obs, "reason": v.reason} for k, v in z.items()},
        "peer_signals": peers,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--period", required=True)
    ap.add_argument("--log", type=Path, default=SIGNAL_LOG)
    ap.add_argument("--dry-run", action="store_true", help="print the record without appending to the log")
    a = ap.parse_args(argv)

    from peer_options.seed import build_seed

    events, _pending = build_seed()
    rec = score_call(
        a.ticker, a.period,
        events=events,
        weights=IndustryMembershipWeights.from_yaml(),
        scorer=LoughranMcDonaldScorer.from_master_csv(str(LM_MASTER)),
        uncertainty_terms=uncertainty_terms_from_master_csv(str(LM_MASTER)),
    )
    if not a.dry_run:
        rec = SignalLog(a.log).append(rec)
    print(json.dumps(rec, indent=2, sort_keys=True))
    if rec["late"]:
        print("WARNING: scored >30 min after signal_available_at (late=true); "
              "this is not a live-call signal.", file=sys.stderr)
    if not a.dry_run:
        print(f"appended to {a.log} (record_hash {rec['record_hash'][:12]})", file=sys.stderr)


if __name__ == "__main__":
    main()
