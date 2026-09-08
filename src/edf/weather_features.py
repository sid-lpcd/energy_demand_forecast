"""Weather-derived model features (Week 4).

Loads one of `edf.data.weather`'s already-downloaded hourly source parquets,
interpolates it onto the canonical table's half-hourly index (linear/time
interpolation — sensible for smoothly-varying quantities like temperature,
unlike a step/forward-fill), and adds heating/cooling degree features.

This module deliberately has no opinion on *which* source is valid for a
given experiment (ERA5 hindsight vs. the genuine day-ahead forecast, and
over what date range) — see PLAN.md Week 4 and `edf.data.weather`'s
docstring for that; it only knows how to turn one named source into a
feature table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path("data/raw/weather")

# UK energy-industry convention (e.g. BEIS/National Grid degree-day series):
# 15.5C base for heating, since UK buildings retain useful heat from internal
# gains/solar down to roughly that outdoor temperature. 22C for cooling is a
# common, if less standardised, UK counterpart -- GB demand is much less
# cooling-driven than heating-driven, so this matters less, but is included
# for symmetry and because the odd hot spell does show up in demand.
HEATING_BASE_C = 15.5
COOLING_BASE_C = 22.0


def heating_degree(temperature_c: pd.Series, base: float = HEATING_BASE_C) -> pd.Series:
    """How far below `base` it is (0 when at or above) -- a heating-demand proxy."""
    return (base - temperature_c).clip(lower=0)


def cooling_degree(temperature_c: pd.Series, base: float = COOLING_BASE_C) -> pd.Series:
    """How far above `base` it is (0 when at or below) -- a cooling-demand proxy."""
    return (temperature_c - base).clip(lower=0)


def cumulative_degree(degree: pd.Series, window_periods: int) -> pd.Series:
    """Trailing rolling sum of a degree-day series over `window_periods` half-hours.

    A persistence proxy for sustained heating/cooling stress -- e.g. day 3 of
    a heatwave plausibly drives more demand response than day 1 at the same
    instantaneous temperature, which the point-in-time `heating_degree`/
    `cooling_degree` value alone can't distinguish. Only looks backward
    (`rolling`'s default, right-aligned window), so it's exactly as safe at
    any forecast horizon as the point-in-time degree feature already is --
    both depend only on weather at or before the window's right edge.
    """
    return degree.rolling(window_periods, min_periods=window_periods).sum()


def load_weather_series(
    source: str, index: pd.DatetimeIndex, raw_dir: Path = DEFAULT_RAW_DIR
) -> pd.DataFrame:
    """Load one hourly weather source and interpolate it onto `index` (half-hourly).

    Points in `index` outside the source's own time range come back NaN
    (interpolation doesn't extrapolate) -- callers drop those via
    `edf.features.build_feature_table`'s NaN-row handling, same as any other
    feature's warm-up period.
    """
    hourly = pd.read_parquet(raw_dir / f"{source}.parquet").set_index("timestamp").sort_index()
    combined_index = hourly.index.union(index)
    # limit_area="inside": fill only between two real observations, never
    # extrapolate past the source's actual start/end (pandas' interpolate()
    # otherwise holds the boundary value flat outside that range, which would
    # silently misrepresent points the source has no data for as "known").
    interpolated = hourly.reindex(combined_index).interpolate(method="time", limit_area="inside")
    return interpolated.reindex(index)


def build_weather_feature_table(
    source: str, index: pd.DatetimeIndex, raw_dir: Path = DEFAULT_RAW_DIR
) -> pd.DataFrame:
    """Weather feature columns for `source`, aligned to `index`.

    Columns: `temperature_c`, `wind_speed_ms`, `cloud_cover_pct`,
    `shortwave_radiation_wm2`, `heating_degree`, `cooling_degree`.
    """
    weather = load_weather_series(source, index, raw_dir)
    weather["heating_degree"] = heating_degree(weather["temperature_c"])
    weather["cooling_degree"] = cooling_degree(weather["temperature_c"])
    return weather
