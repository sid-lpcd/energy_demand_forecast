"""Baseline demand forecasts (Week 2) — the numbers every later model must beat.

Each baseline takes the observed `demand` Series (indexed by UTC timestamp,
half-hourly, uniformly spaced per `edf.data.clean.check_no_gaps`) and returns
a forecast Series aligned to the same index. Every baseline here is a pure lag
of `demand` — forecast at `t` only ever uses values strictly before `t`, so
none of them can leak the future (see PLAN.md's time-series correctness
rules). Early rows where the required lag reaches before the start of the
data come back as NaN; `edf.evaluate.evaluate` drops those before scoring.
"""

from __future__ import annotations

import pandas as pd

PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = PERIODS_PER_DAY * 7


def naive(demand: pd.Series) -> pd.Series:
    """Forecast = last observed value (t-1, 30 minutes ago)."""
    return demand.shift(1)


def previous_day_same_time(demand: pd.Series) -> pd.Series:
    """Forecast = same half-hour-of-day, one day ago (t-48)."""
    return demand.shift(PERIODS_PER_DAY)


def previous_week_same_time(demand: pd.Series) -> pd.Series:
    """Forecast = same half-hour-of-day and day-of-week, one week ago (t-336).

    This is the seasonal-naive baseline: its MAE becomes the MASE denominator
    for every other model, Week 2 onward (see `edf.evaluate.compare_models`).
    """
    return demand.shift(PERIODS_PER_WEEK)


def trailing_moving_average(demand: pd.Series, n_days: int = 4) -> pd.Series:
    """Forecast = mean of the last `n_days` same-half-hour-of-day observations.

    E.g. n_days=4 averages t-48, t-96, t-144, t-192 — smooths day-to-day noise
    while still tracking daily (not weekly) seasonality, at the cost of
    needing `n_days` of history before it can produce a forecast.

    `skipna=False`: a row with fewer than `n_days` lags available (the
    warm-up period) returns NaN rather than a partial, weaker-than-intended
    average.
    """
    lags = [demand.shift(PERIODS_PER_DAY * k) for k in range(1, n_days + 1)]
    return pd.concat(lags, axis=1).mean(axis=1, skipna=False)


BASELINES = {
    "naive": naive,
    "previous_day_same_time": previous_day_same_time,
    "previous_week_same_time": previous_week_same_time,
    "trailing_moving_average": trailing_moving_average,
}
