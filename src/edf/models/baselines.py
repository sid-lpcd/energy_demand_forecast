"""Baseline demand forecasts (Week 2) — the numbers every later model must beat.

Each baseline takes the observed `demand` Series (indexed by UTC timestamp,
half-hourly, uniformly spaced per `edf.data.clean.check_no_gaps`) and returns
a forecast Series aligned to the same index. Every baseline here is a pure lag
of `demand` — forecast at `t` only ever uses values strictly before `t`, so
none of them can leak the future (see PLAN.md's time-series correctness
rules). Early rows where the required lag reaches before the start of the
data come back as NaN; `edf.evaluate.evaluate` drops those before scoring.

`naive`/`previous_day_same_time`/`previous_week_same_time`/
`trailing_moving_average` are fixed at a 30-minute-ahead horizon. `valid_baselines`
generalizes this to the other horizons in `edf.config.HORIZONS` (1h/1d/7d) —
see its docstring for why a fixed-lag baseline can become invalid, not just
worse, at a longer horizon.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pandas as pd

from edf.config import PERIODS_PER_DAY, PERIODS_PER_WEEK


def persistence(demand: pd.Series, horizon_periods: int) -> pd.Series:
    """Forecast = the last value known at issue time, carried flat to the target.

    Generalizes `naive` (horizon_periods=1) to an arbitrary forecast horizon:
    whatever a forecaster's most recent actual observation was `horizon_periods`
    steps before the target, unchanged. Valid at every horizon by
    construction — there's no "more recent" data it could be leaking.
    """
    return demand.shift(horizon_periods)


def naive(demand: pd.Series) -> pd.Series:
    """Forecast = last observed value (t-1, 30 minutes ago)."""
    return persistence(demand, horizon_periods=1)


def previous_day_same_time(demand: pd.Series) -> pd.Series:
    """Forecast = same half-hour-of-day, one day ago (t-48)."""
    return demand.shift(PERIODS_PER_DAY)


def previous_week_same_time(demand: pd.Series) -> pd.Series:
    """Forecast = same half-hour-of-day and day-of-week, one week ago (t-336).

    This is the seasonal-naive baseline: its MAE becomes the MASE denominator
    for every other model, Week 2 onward (see `edf.evaluate.compare_models").
    Its 336-period lag is also longer than every horizon in
    `edf.config.HORIZONS`, so it stays valid (and a consistent MASE
    denominator) across every horizon's scenario analysis.
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


def trailing_weekly_moving_average(demand: pd.Series, n_weeks: int = 4) -> pd.Series:
    """Forecast = mean of the last `n_weeks` same-half-hour-and-day-of-week observations.

    The weekly analogue of `trailing_moving_average`, for horizons (e.g. 7
    days ahead) where the daily lags it uses wouldn't be available yet.
    """
    lags = [demand.shift(PERIODS_PER_WEEK * k) for k in range(1, n_weeks + 1)]
    return pd.concat(lags, axis=1).mean(axis=1, skipna=False)


BASELINES: dict[str, Callable[[pd.Series], pd.Series]] = {
    "naive": naive,
    "previous_day_same_time": previous_day_same_time,
    "previous_week_same_time": previous_week_same_time,
    "trailing_moving_average": trailing_moving_average,
}

# (baseline, minimum lag in periods it's built on) — a fixed-lag baseline is
# only usable at a horizon whose lead time doesn't exceed its own lag (see
# `valid_baselines`).
_FIXED_LAG_BASELINES: dict[str, tuple[Callable[[pd.Series], pd.Series], int]] = {
    "previous_day_same_time": (previous_day_same_time, PERIODS_PER_DAY),
    "previous_week_same_time": (previous_week_same_time, PERIODS_PER_WEEK),
    "trailing_moving_average": (trailing_moving_average, PERIODS_PER_DAY),
    "trailing_weekly_moving_average": (trailing_weekly_moving_average, PERIODS_PER_WEEK),
}


def valid_baselines(horizon_periods: int) -> dict[str, Callable[[pd.Series], pd.Series]]:
    """The baselines that can genuinely be computed at a given forecast horizon.

    A fixed-lag baseline leaks data a forecaster wouldn't have yet once its
    lag is shorter than the horizon — e.g. at a 7-day horizon (336 periods),
    "same time yesterday" (lag 48) isn't known when the forecast has to be
    issued, so `previous_day_same_time` and `trailing_moving_average` are
    excluded there. `persistence` (the generalized `naive`) is included at
    every horizon, since it's defined as "the most recent value available at
    issue time" — there's nothing for it to leak.
    """
    valid = {
        name: fn
        for name, (fn, min_lag) in _FIXED_LAG_BASELINES.items()
        if min_lag >= horizon_periods
    }
    valid["persistence"] = partial(persistence, horizon_periods=horizon_periods)
    return valid
