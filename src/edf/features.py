"""Feature engineering for the Week 3 LightGBM model(s).

Two kinds of feature live here, and they behave very differently with
respect to forecast horizon:

- **Time/calendar features** (`time_features`, plus the `is_bank_holiday`/
  `is_school_holiday`/`lockdown_level`/`event_tier`/`solar_eclipse_pct`
  columns already in the canonical table) are deterministic facts about the
  *target* timestamp — a bank holiday in a month's time is just as knowable
  today as it is tomorrow. They're valid inputs at every horizon.
- **Lag/rolling features** (`lag_features`, `rolling_features`) are built
  from `demand` itself, so — exactly as with `edf.baselines.valid_baselines`
  — a given lag is only a legitimate input at a horizon shorter than or
  equal to its own lag; otherwise it would use data a forecaster wouldn't
  actually have at issue time yet. `rolling_features` reuses
  `edf.baselines.trailing_moving_average`/`trailing_weekly_moving_average`
  directly, so the seasonal-naive baselines double as model features here.

`wind`/`solar`/`interconnector` are deliberately never used as features —
per PLAN.md, they're same-period actuals with no forecast archive, and using
them here would misrepresent the model's deployable accuracy (Week 4b is
where that gets fixed properly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edf.baselines import (
    PERIODS_PER_DAY,
    PERIODS_PER_WEEK,
    trailing_moving_average,
    trailing_weekly_moving_average,
)

PERIODS_PER_YEAR = round(365.25 * PERIODS_PER_DAY)

LAG_CANDIDATES: tuple[int, ...] = (1, 2, PERIODS_PER_DAY, PERIODS_PER_WEEK)

CALENDAR_COLUMNS: tuple[str, ...] = (
    "is_bank_holiday",
    "is_school_holiday",
    "lockdown_level",
    "event_tier",
    "solar_eclipse_pct",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "lockdown_level",
    "event_tier",
    "hour_of_day",
    "day_of_week",
    "month",
)


def fourier_terms(index: pd.DatetimeIndex, period_periods: float, order: int, label: str) -> pd.DataFrame:
    """`order` sin/cos harmonic pairs of a seasonal cycle `period_periods` long.

    `t` is the integer position of each timestamp since a fixed UTC epoch
    (not since the start of `index`), so the same absolute timestamp always
    maps to the same phase regardless of which slice of the data it's
    computed on — required for train and validation features to line up.
    """
    t = (index - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(minutes=30)
    t = t.to_numpy(dtype=float)
    columns = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period_periods
        columns[f"fourier_{label}_sin_{k}"] = np.sin(angle)
        columns[f"fourier_{label}_cos_{k}"] = np.cos(angle)
    return pd.DataFrame(columns, index=index)


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic features of the timestamp itself: valid at any horizon."""
    local = index  # already UTC; half-hour-of-day/day-of-week are UTC-based like the rest of the project
    minutes_since_midnight = local.hour * 60 + local.minute
    hour_of_day = (minutes_since_midnight // 30).astype("int16")

    base = pd.DataFrame(
        {
            "hour_of_day": hour_of_day,
            "day_of_week": local.dayofweek.astype("int16"),
            "month": local.month.astype("int16"),
        },
        index=index,
    )
    daily = fourier_terms(index, PERIODS_PER_DAY, order=2, label="daily")
    weekly = fourier_terms(index, PERIODS_PER_WEEK, order=2, label="weekly")
    annual = fourier_terms(index, PERIODS_PER_YEAR, order=2, label="annual")
    return pd.concat([base, daily, weekly, annual], axis=1)


def lag_features(
    demand: pd.Series, horizon_periods: int, lags: tuple[int, ...] = LAG_CANDIDATES
) -> pd.DataFrame:
    """Lag columns valid at `horizon_periods` — i.e. lag >= horizon_periods.

    A shorter lag would need data that doesn't exist yet at issue time (see
    module docstring / `edf.baselines.valid_baselines`).
    """
    columns = {f"lag_{lag}": demand.shift(lag) for lag in lags if lag >= horizon_periods}
    return pd.DataFrame(columns, index=demand.index)


def rolling_features(demand: pd.Series, horizon_periods: int) -> pd.DataFrame:
    """Rolling-mean features, reusing the seasonal-naive baselines as inputs.

    Included only when their own minimum lag is valid at `horizon_periods`
    (same rule as `lag_features`).
    """
    columns = {}
    if PERIODS_PER_DAY >= horizon_periods:
        columns["trailing_moving_average"] = trailing_moving_average(demand)
    if PERIODS_PER_WEEK >= horizon_periods:
        columns["trailing_weekly_moving_average"] = trailing_weekly_moving_average(demand)
    return pd.DataFrame(columns, index=demand.index)


def deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time + calendar features: valid at every horizon, indexed like `df`.

    Split out from `build_feature_table` so `edf.forecast.recursive_forecast`
    can precompute this once for the whole table and slice it by position at
    each recursion step, rather than only ever getting it pre-filtered to one
    horizon's non-NaN rows.
    """
    X = pd.concat([time_features(df.index), df[list(CALENDAR_COLUMNS)]], axis=1)
    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def build_feature_table(df: pd.DataFrame, horizon_periods: int) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble (X, y) for training/scoring a demand model at a given horizon.

    `df` is the canonical table (`edf.data.clean.build_canonical_table`).
    Rows with any NaN feature (the lag/rolling warm-up period at the very
    start of the data) are dropped; `X` and `y` come back aligned on the
    surviving index.
    """
    demand = df["demand"]
    X = pd.concat(
        [
            deterministic_features(df),
            lag_features(demand, horizon_periods),
            rolling_features(demand, horizon_periods),
        ],
        axis=1,
    )

    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    y = demand.loc[valid]
    return X, y
