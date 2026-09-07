"""Extreme-event / day-type bucket definitions for Week 6.

Each bucket is an independent boolean flag over the canonical index — these
are NOT a mutually-exclusive partition (a half-hour can be cold, windy, and
a weekend all at once). They're separate "stress test" slices: evaluate the
model's accuracy/calibration restricted to each one and compare against the
overall metric, per PLAN.md. Thresholds are computed from data and
documented here, not eyeballed.
"""

from __future__ import annotations

import pandas as pd

CHRISTMAS_START_MONTH_DAY = (12, 24)  # inclusive
CHRISTMAS_END_MONTH_DAY = (1, 1)  # inclusive, wraps into the new year


def month_relative_percentile_bucket(
    values: pd.Series, low_pct: float = 10, high_pct: float = 90
) -> tuple[pd.Series, pd.Series]:
    """(is_low, is_high): whether each row is below/above the low/high
    percentile *for its own calendar month*, using thresholds computed from
    `values`' own distribution within that month.

    "Cold" needs to mean unusually cold *for the time of year* — a fixed,
    non-month-relative threshold would only ever flag winter days as cold
    and summer days as hot, missing the actual extremes (an unusually cold
    July day, an unusually mild January one).
    """
    months = values.index.month
    low_threshold = values.groupby(months).transform(lambda s: s.quantile(low_pct / 100))
    high_threshold = values.groupby(months).transform(lambda s: s.quantile(high_pct / 100))
    return values < low_threshold, values > high_threshold


def overall_percentile_bucket(
    values: pd.Series, low_pct: float = 10, high_pct: float = 90
) -> tuple[pd.Series, pd.Series]:
    """(is_low, is_high) against fixed thresholds over the whole series —
    unlike temperature, wind doesn't need month-relative treatment here:
    the buckets exist to capture genuinely extreme embedded-generation
    periods (relevant to the demand-suppression mechanism), not "extreme
    for the season"."""
    low_threshold = values.quantile(low_pct / 100)
    high_threshold = values.quantile(high_pct / 100)
    return values < low_threshold, values > high_threshold


def is_christmas_period(index: pd.DatetimeIndex) -> pd.Series:
    """Dec 24 - Jan 1 inclusive (wraps the year boundary) — the UK's
    lowest-demand calendar stretch, driven by widespread time off work."""
    month, day = index.month, index.day
    in_december = (month == 12) & (day >= CHRISTMAS_START_MONTH_DAY[1])
    in_january = (month == 1) & (day <= CHRISTMAS_END_MONTH_DAY[1])
    return pd.Series(in_december | in_january, index=index)


def is_weekend(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.dayofweek >= 5, index=index)


def build_day_type_buckets(
    temperature_c: pd.Series, wind_capacity_factor: pd.Series
) -> pd.DataFrame:
    """All Week 6 buckets as boolean columns, aligned to `temperature_c`'s index.

    Columns: `is_cold`, `is_hot` (month-relative temperature percentile),
    `is_low_wind`, `is_high_wind` (overall capacity-factor percentile),
    `is_christmas`, `is_weekend`.
    """
    is_cold, is_hot = month_relative_percentile_bucket(temperature_c)
    is_low_wind, is_high_wind = overall_percentile_bucket(wind_capacity_factor)
    index = temperature_c.index
    return pd.DataFrame(
        {
            "is_cold": is_cold,
            "is_hot": is_hot,
            "is_low_wind": is_low_wind,
            "is_high_wind": is_high_wind,
            "is_christmas": is_christmas_period(index),
            "is_weekend": is_weekend(index),
        },
        index=index,
    )
