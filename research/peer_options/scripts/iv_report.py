"""Development-sample look at peers' abnormal implied-vol change after the TSMC call (descriptive, pilot).

Option data = ThetaData Free-plan EOD quotes; IV is computed here (parity forward + Black-76 on OTM mids), so there
is no vendor IV, delta or open interest. Outcome per event and day k (day 0 = entry day, k stops at the close before the
peer reports):  [ln IV30_B(k) - ln IV30_B(ref)] - [ln IV30_SMH(k) - ln IV30_SMH(ref)], ref = close before entry;
the same difference for the 25-delta put skew. Controls and date fixed effects from the prereg are NOT estimated:
n is far below min_events, so this is descriptive only. Holdout dates are never fetched.
Maturity: 30d IV interpolates in total variance between listed expiries; if only one side exists within 14 days it takes
that expiry's IV (flat). Names with monthly-only options (KLAC) rely on this part of the time.
    python scripts/iv_report.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peer_options import prereg  # noqa: E402
from peer_options.calendar import TradingCalendar  # noqa: E402
from peer_options.events import derive_window  # noqa: E402
from peer_options.market.prices import cap_end_before_holdout, load_prices  # noqa: E402
from peer_options.options.features import compute_features  # noqa: E402
from peer_options.options.iv import chain_with_iv  # noqa: E402
from peer_options.options.theta_eod import OptionEodStore, needed_expirations, rate_on  # noqa: E402
from peer_options.study.dev_sample import dev_events_with_signals  # noqa: E402

BENCH, START, KS, MIN_DTE, MAX_DTE = "SMH", date(2025, 8, 1), (1, 3, 5), 1, 100
MAX_EXTRAP = 14  # flat-vol fallback (days) when no listed expiry brackets the 30d target, e.g. monthly-only KLAC
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cal, end = TradingCalendar.from_yaml(), cap_end_before_holdout(date.today())
    sig_events = dev_events_with_signals("TSM")
    # trading days needed per event: the reference close, then entry .. min(exit, entry + 5 steps)
    plan = {}
    for e, _ in sig_events:
        entry, exit_, n = derive_window(e.signal_available_at, e.peer_report_datetime, cal)
        days, d = [], entry
        for _k in range(min(5, n) + 1):
            days.append(d)
            d = cal.next_trading_day(d)
        plan[e.event_id] = (cal.prev_trading_day(entry), days)
    dates_by_symbol: dict[str, set] = {BENCH: set()}
    for e, _ in sig_events:
        ref, days = plan[e.event_id]
        for sym in (e.peer, BENCH):
            dates_by_symbol.setdefault(sym, set()).update([ref, *days])
    prereg.HoldoutGuard.from_prereg().check([d for ds in dates_by_symbol.values() for d in ds])

    store = OptionEodStore()
    sofr = store.sofr(START, end)
    jobs = []
    for sym, ds in dates_by_symbol.items():
        for exp in needed_expirations(store.expirations(sym), sorted(ds), min_dte=MIN_DTE, max_dte=MAX_DTE):
            jobs.append((sym, exp, [d for d in ds if MIN_DTE <= (exp - d).days <= MAX_DTE]))
    print(f"{len(sig_events)} events | {len(jobs)} expiries (only uncached days are requested)", flush=True)

    def pull(j):
        sym, exp, qd = j
        store.ensure_days(sym, exp, qd)
        return j

    with ThreadPoolExecutor(max_workers=1) as pool:  # the Free plan rejects concurrent requests
        for i, _j in enumerate(pool.map(pull, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"  checked {i}/{len(jobs)}", flush=True)
    chains = {s_: [store.load_expiration(s_, exp) for s_2, exp, _ in jobs if s_2 == s_] for s_ in dates_by_symbol}

    spots = {s: load_prices(s, START, end, adjust=False).set_index("date")["close"] for s in dates_by_symbol}
    feats = {}
    for sym, ds in dates_by_symbol.items():
        allq = pd.concat([c for c in chains[sym] if len(c)], ignore_index=True)
        for d in sorted(ds):
            day_q = allq[allq["date"] == d]
            ch = chain_with_iv(day_q, float(spots[sym][d]), d, rate_on(sofr, d)) if len(day_q) else pd.DataFrame()
            feats[(sym, d)] = compute_features(sym, d, ch, float(spots[sym][d]), max_extrap_days=MAX_EXTRAP) if len(ch) else None
    flat = [dict(symbol=s, date=d, iv30=f.atm_iv_30d, iv60=f.atm_iv_60d, skew=f.put_skew_25d, spread=f.bid_ask_pct_mid)
            for (s, d), f in feats.items() if f is not None]
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(out_dir / "iv_features.csv", index=False)

    rows = []
    for e, s in sig_events:
        ref, days = plan[e.event_id]
        row = {"event": e.event_id, "peer": e.peer, **{f"S_{k}": s[k] for k in ("tone", "hedge", "gap")}}

        def chg(sym, d, attr):
            f0, f1 = feats.get((sym, ref)), feats.get((sym, d))
            if f0 is None or f1 is None:
                return np.nan
            a, b = getattr(f0, attr), getattr(f1, attr)
            if not (np.isfinite(a) and np.isfinite(b)):
                return np.nan
            return (np.log(b) - np.log(a)) if attr == "atm_iv_30d" else (b - a)

        row["iv30_ref"] = getattr(feats.get((e.peer, ref)), "atm_iv_30d", np.nan)
        for k in KS:
            d = days[k] if k < len(days) else None
            for name, attr in (("div", "atm_iv_30d"), ("dskew", "put_skew_25d")):
                row[f"{name}{k}"] = (chg(e.peer, d, attr) - chg(BENCH, d, attr)) if d else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "iv_report.csv", index=False)
    pd.set_option("display.width", 240, "display.float_format", lambda v: f"{v:7.3f}")
    print("\n", df.to_string(index=False))
    print("\nSpearman rho (Pearson on average ranks), descriptive only; blank if n<4")
    print(f"{'':10}" + "".join(f"{o + str(k):>9}" for o in ("div", "dskew") for k in KS))
    for sname in ("S_tone", "S_hedge", "S_gap"):
        cells = []
        for o in ("div", "dskew"):
            for k in KS:
                pr = df[[sname, f"{o}{k}"]].dropna()
                cells.append(f"{pr[sname].rank().corr(pr[f'{o}{k}'].rank()):9.2f}" if len(pr) >= 4 else f"{'n/a':>9}")
        print(f"{sname:10}" + "".join(cells))
    print(f"\nevents={len(df)}, distinct signal values={ {c: df[c].nunique() for c in ('S_tone','S_hedge','S_gap')} }")


if __name__ == "__main__":
    main()
