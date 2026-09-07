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

import numpy as np
import pandas as pd

from edf.data.weather import GB_CITIES
from edf.features import time_features

# Population-weighted GB centroid — the same weighting `edf.data.weather` uses
# to aggregate weather across cities, reused here for a single representative
# point for solar geometry. Solar elevation varies little across GB's
# latitude span for this purpose, so one point (rather than population-
# weighting the angle itself, city by city) is an adequate simplification.
_TOTAL_POPULATION = sum(c.population_millions for c in GB_CITIES)
GB_CENTROID_LAT = sum(c.lat * c.population_millions for c in GB_CITIES) / _TOTAL_POPULATION
GB_CENTROID_LON = sum(c.lon * c.population_millions for c in GB_CITIES) / _TOTAL_POPULATION


def capacity_factor(generation: pd.Series, capacity: pd.Series) -> pd.Series:
    return generation / capacity


def solar_elevation_deg(
    index: pd.DatetimeIndex, lat_deg: float = GB_CENTROID_LAT, lon_deg: float = GB_CENTROID_LON
) -> pd.Series:
    """Solar elevation angle in degrees, clipped to 0 below the horizon.

    Standard declination/hour-angle formula (Cooper's equation for
    declination; ignores the ~±15-minute equation-of-time correction, which
    is immaterial at half-hourly resolution). This is a *deterministic*
    function of timestamp and location — no weather data or API call
    needed — so it doubles as the hard physical ceiling PLAN.md calls for:
    generation is exactly 0 whenever this is 0 (below horizon), regardless
    of weather.
    """
    day_of_year = index.dayofyear.to_numpy()
    declination_deg = 23.45 * np.sin(np.radians(360 / 365 * (284 + day_of_year)))

    utc_hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    solar_time = utc_hour + lon_deg / 15  # longitude-only correction
    hour_angle_deg = 15 * (solar_time - 12)

    lat_rad = np.radians(lat_deg)
    decl_rad = np.radians(declination_deg)
    hour_rad = np.radians(hour_angle_deg)
    sin_elevation = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(
        decl_rad
    ) * np.cos(hour_rad)
    elevation_deg = np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))
    return pd.Series(np.clip(elevation_deg, 0, None), index=index)


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


def build_solar_feature_table(
    df: pd.DataFrame, weather: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) for the solar capacity-factor model: y = solar capacity factor.

    `weather` must have `cloud_cover_pct`/`shortwave_radiation_wm2` columns.
    Features: time features, cloud cover + shortwave radiation (the
    weather-driven component), `solar_elevation_deg` (the deterministic
    ceiling — zero at night regardless of weather), and `solar_eclipse_pct`
    (already in the canonical table, `edf.data.calendar_events`) — a real,
    if rare, physical driver of solar generation.
    """
    X = pd.concat(
        [
            time_features(df.index),
            weather[["cloud_cover_pct", "shortwave_radiation_wm2"]],
            solar_elevation_deg(df.index).rename("solar_elevation_deg"),
            df["solar_eclipse_pct"],
        ],
        axis=1,
    )
    y = capacity_factor(df["solar"], df["solar_capacity"])

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]
