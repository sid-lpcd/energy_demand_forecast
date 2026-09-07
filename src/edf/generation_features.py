"""Capacity-factor feature engineering for the Week 4b wind/solar generation forecasts.

Wind/solar generation is a strongly weather-driven, *not* demand-driven
process — a real forecaster predicts it from a weather forecast, not from
its own recent lag history the way `edf.features` builds demand features.
So this module deliberately doesn't offer lag/rolling features at all: per
PLAN.md, `capacity_factor ~ f(weather forecast, calendar)` only.

Capacity factor (`generation / capacity`), not raw MW, is the actual
target, so installed-capacity growth over 2020-2025 doesn't get conflated
with "was it windy" — see PLAN.md Week 4b.
"""

from __future__ import annotations

import pandas as pd

from edf.features import time_features


def capacity_factor(generation: pd.Series, capacity: pd.Series) -> pd.Series:
    return generation / capacity


def build_wind_feature_table(
    df: pd.DataFrame, weather: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) for the wind capacity-factor model: y = wind capacity factor.

    `weather` (`edf.weather_features.build_weather_feature_table`) must have
    a `wind_speed_ms` column, aligned to `df`'s index. Only `wind_speed_ms`
    is used — cloud cover/radiation/temperature aren't physically relevant
    to wind generation, so including them would just be noise for a
    tree model to (over)fit around.
    """
    X = pd.concat([time_features(df.index), weather[["wind_speed_ms"]]], axis=1)
    y = capacity_factor(df["wind"], df["wind_capacity"])

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]
