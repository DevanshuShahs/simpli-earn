"""EXPLORATORY, NOT PREREGISTERED. Development-sample look at peers' realized reaction vs announcer call signals.

No option data is available (ThetaData Free plan), so this uses free daily stock bars: abnormal return and abnormal
realized-vol (Parkinson) of each peer vs SMH after the TSMC call, up to the close before the peer reports.
Descriptive only: n is tiny, signals repeat across peers within a quarter, and no p-values are reported.
Holdout events (announcer call on/after the preregistered holdout start) are excluded and their dates never fetched.
    python scripts/proxy_report.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peer_options import prereg  # noqa: E402
from peer_options.calendar import TradingCalendar  # noqa: E402
from peer_options.events import eligible_for_study  # noqa: E402
from peer_options.links import IndustryMembershipWeights  # noqa: E402
from peer_options.market.prices import cap_end_before_holdout, jump_flags, load_prices  # noqa: E402
from peer_options.scoring import LM_MASTER, score_call  # noqa: E402
from peer_options.seed import build_seed  # noqa: E402
from peer_options.signals.hedging import uncertainty_terms_from_master_csv  # noqa: E402
from peer_options.signals.tone import LoughranMcDonaldScorer  # noqa: E402
from peer_options.study.proxy import peer_outcomes  # noqa: E402

BENCH, START = "SMH", date(2025, 8, 1)
KS = (1, 3, 5)


def main() -> None:
    prereg.verify()
    bounds = prereg.holdout_bounds()
    cal = TradingCalendar.from_yaml()
    events, _ = build_seed()
    dev = [e for e in eligible_for_study(events)
           if e.announcer == "TSM" and not prereg.is_holdout_event(e.announcer_call_start, bounds)]
    end = cap_end_before_holdout(date.today())
    print(f"development events: {len(dev)} | price data {START}..{end} (holdout starts {bounds[0]})")

    px = {s: load_prices(s, START, end) for s in {e.peer for e in dev} | {BENCH}}
    for s, df in px.items():
        print(f"  {s}: {len(df)} bars {df['date'].min()}..{df['date'].max()}, jumps>35%: {jump_flags(df)}")
    prereg.HoldoutGuard.from_prereg().check([d for df in px.values() for d in df["date"]])  # raises if any touch

    scorer = LoughranMcDonaldScorer.from_master_csv(str(LM_MASTER))
    unc = uncertainty_terms_from_master_csv(str(LM_MASTER))
    weights = IndustryMembershipWeights.from_yaml()
    manifest = pd.read_csv(LM_MASTER.parent / "transcripts" / "manifest.csv")
    sig = {}
    for e in dev:
        start = e.announcer_call_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        period = manifest.loc[manifest["call_start_utc"] == start, "period"].iloc[0]
        if period not in sig:
            sig[period] = score_call("TSM", period, events=events, weights=weights, scorer=scorer, uncertainty_terms=unc)
    rows = []
    for e in dev:
        period = manifest.loc[manifest["call_start_utc"] == e.announcer_call_start.strftime("%Y-%m-%dT%H:%M:%SZ"), "period"].iloc[0]
        s = next(p for p in sig[period]["peer_signals"] if p["event_id"] == e.event_id)["S_BA"]
        out = {o.k: o for o in peer_outcomes(e, px[e.peer], px[BENCH], cal)}
        row = {"event": e.event_id, "peer": e.peer, "n_days": len(out), **{f"S_{k}": s[k] for k in ("tone", "hedge", "gap")}}
        for k in KS:
            row[f"car{k}"] = out[k].car if k in out else float("nan")
            row[f"rv{k}"] = out[k].abn_logrv if k in out else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.float_format", lambda v: f"{v:7.3f}")
    print("\n", df.to_string(index=False))
    out_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "proxy_report.csv", index=False)

    print("\nSpearman rho (Pearson on average ranks; signal vs outcome), descriptive only; n = events with both values")
    print(f"{'':12}" + "".join(f"{o + str(k):>9}" for o in ("car", "rv") for k in KS))
    for sname in ("S_tone", "S_hedge", "S_gap"):
        cells = []
        for o in ("car", "rv"):
            for k in KS:
                pair = df[[sname, f"{o}{k}"]].dropna()
                cells.append(f"{pair[sname].rank().corr(pair[f'{o}{k}'].rank()):9.2f}" if len(pair) >= 4 else f"{'n/a':>9}")
        print(f"{sname:12}" + "".join(cells))
    print(f"\nn events = {len(df)}, distinct signal values per signal = "
          f"{ {c: df[c].nunique() for c in ('S_tone','S_hedge','S_gap')} } (both peers share one TSMC call per quarter)")


if __name__ == "__main__":
    main()
