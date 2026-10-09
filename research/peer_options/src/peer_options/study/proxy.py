"""EXPLORATORY realized-price proxy for the peer-options hypotheses (not preregistered).

Used only while no option data is available. Outcomes are peer B's reaction after the announcer's call, relative to a
benchmark ETF, up to the close before B reports:
  car_k      cumulative abnormal log return from the close before entry through day k (B minus benchmark)
  abn_logrv_k log of mean Parkinson variance over days 0..k relative to the trailing base window, B minus benchmark
Day 0 is the entry day from the events table. Days stop at the exit date, so B's own earnings never enter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from peer_options.calendar import TradingCalendar
from peer_options.events import Event, derive_window


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    return np.log(high / low) ** 2 / (4 * math.log(2))


@dataclass(frozen=True)
class DayOutcome:
    event_id: str
    peer: str
    k: int
    day: date
    car: float
    abn_logrv: float


def peer_outcomes(
    event: Event, peer_px: pd.DataFrame, bench_px: pd.DataFrame, cal: TradingCalendar,
    *, horizon: int = 5, base_days: int = 20,
) -> list[DayOutcome]:
    entry, exit_, n = derive_window(event.signal_available_at, event.peer_report_datetime, cal)
    if n < 1:
        return []
    p, b = peer_px.set_index("date"), bench_px.set_index("date")
    ref = cal.prev_trading_day(entry)
    base_dates = []
    d = ref
    while len(base_dates) < base_days:
        base_dates.append(d)
        d = cal.prev_trading_day(d)
    if any(x not in p.index or x not in b.index for x in [ref, *base_dates]):
        return []  # insufficient history for the baseline: no outcome, never imputed
    base_p = parkinson_var(p.loc[base_dates, "high"], p.loc[base_dates, "low"]).mean()
    base_b = parkinson_var(b.loc[base_dates, "high"], b.loc[base_dates, "low"]).mean()
    out, day, vp, vb = [], entry, [], []
    for k in range(min(horizon, n) + 1):
        if day not in p.index or day not in b.index or day > exit_:
            break
        vp.append(float(parkinson_var(p.loc[[day], "high"], p.loc[[day], "low"]).iloc[0]))
        vb.append(float(parkinson_var(b.loc[[day], "high"], b.loc[[day], "low"]).iloc[0]))
        car = math.log(p.loc[day, "close"] / p.loc[ref, "close"]) - math.log(b.loc[day, "close"] / b.loc[ref, "close"])
        lrv = math.log(np.mean(vp) / base_p) - math.log(np.mean(vb) / base_b)
        out.append(DayOutcome(event.event_id, event.peer, k, day, car, lrv))
        day = cal.next_trading_day(day)
    return out
