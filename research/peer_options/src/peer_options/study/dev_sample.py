"""The development sample: TSMC-announcer events outside the holdout, each with its peer signal S_BA.

Shared by the exploratory reports. Scores calls with score_call (no log write), so nothing here touches the
holdout or the append-only signal log.
"""
from __future__ import annotations

import pandas as pd

from peer_options import prereg
from peer_options.events import Event, eligible_for_study
from peer_options.links import IndustryMembershipWeights
from peer_options.scoring import LM_MASTER, score_call
from peer_options.seed import build_seed
from peer_options.signals.hedging import uncertainty_terms_from_master_csv
from peer_options.signals.tone import LoughranMcDonaldScorer


def dev_events_with_signals(announcer: str = "TSM") -> list[tuple[Event, dict[str, float | None]]]:
    prereg.verify()
    bounds = prereg.holdout_bounds()
    events, _ = build_seed()
    dev = [e for e in eligible_for_study(events)
           if e.announcer == announcer and not prereg.is_holdout_event(e.announcer_call_start, bounds)]
    manifest = pd.read_csv(LM_MASTER.parent / "transcripts" / "manifest.csv")
    scorer = LoughranMcDonaldScorer.from_master_csv(str(LM_MASTER))
    unc = uncertainty_terms_from_master_csv(str(LM_MASTER))
    weights = IndustryMembershipWeights.from_yaml()
    records: dict[str, dict] = {}
    out = []
    for e in dev:
        start = e.announcer_call_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        period = manifest.loc[manifest["call_start_utc"] == start, "period"].iloc[0]
        if period not in records:
            records[period] = score_call(announcer, period, events=events, weights=weights, scorer=scorer,
                                         uncertainty_terms=unc)
        s = next(p for p in records[period]["peer_signals"] if p["event_id"] == e.event_id)["S_BA"]
        out.append((e, s))
    return out
