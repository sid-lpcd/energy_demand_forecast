import numpy as np
import pandas as pd
import pytest

from edf.generation_features import (
    build_solar_feature_table,
    build_wind_feature_table,
    capacity_factor,
    solar_elevation_deg,
)


def test_capacity_factor_is_generation_over_capacity():
    generation = pd.Series([100.0, 200.0, 300.0])
    capacity = pd.Series([1000.0, 1000.0, 1000.0])
    result = capacity_factor(generation, capacity)
    assert result.tolist() == [0.1, 0.2, 0.3]


def test_build_wind_feature_table_target_matches_capacity_factor():
    index = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {"wind": [100.0, 200.0, 300.0, 400.0, 500.0], "wind_capacity": [1000.0] * 5},
        index=index,
    )
    weather = pd.DataFrame({"wind_speed_ms": [5.0, 6.0, 7.0, 8.0, 9.0]}, index=index)

    X, y = build_wind_feature_table(df, weather)

    assert y.tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert "wind_speed_ms" in X.columns
    assert "hour_of_day" in X.columns
    assert list(X.index) == list(y.index)


def test_build_wind_feature_table_drops_rows_with_missing_weather():
    index = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    df = pd.DataFrame({"wind": [100.0, 200.0, 300.0], "wind_capacity": [1000.0] * 3}, index=index)
    weather = pd.DataFrame({"wind_speed_ms": [5.0, np.nan, 7.0]}, index=index)

    X, _ = build_wind_feature_table(df, weather)

    assert index[1] not in X.index
    assert len(X) == 2


def test_solar_elevation_deg_zero_at_midnight():
    index = pd.date_range("2024-06-21", periods=1, freq="30min", tz="UTC")
    assert solar_elevation_deg(index).iloc[0] == 0.0


def test_solar_elevation_deg_matches_known_solstice_maxima():
    # At solar noon (~12:00 UTC for a GB longitude near 0), max elevation is
    # 90 - lat + declination (summer) / 90 - lat - declination (winter) --
    # the standard textbook check for this formula.
    lat = 52.0
    summer_noon = pd.date_range("2024-06-21T12:00", periods=1, tz="UTC")
    winter_noon = pd.date_range("2024-12-21T12:00", periods=1, tz="UTC")

    summer_elevation = solar_elevation_deg(summer_noon, lat_deg=lat, lon_deg=0.0).iloc[0]
    winter_elevation = solar_elevation_deg(winter_noon, lat_deg=lat, lon_deg=0.0).iloc[0]

    assert summer_elevation == pytest.approx(90 - lat + 23.45, abs=0.5)
    assert winter_elevation == pytest.approx(90 - lat - 23.45, abs=0.5)


def test_solar_elevation_deg_never_negative():
    index = pd.date_range("2024-01-01", periods=48, freq="30min", tz="UTC")
    assert (solar_elevation_deg(index) >= 0).all()


def test_build_solar_feature_table_target_matches_capacity_factor():
    index = pd.date_range("2024-06-01", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {
            "solar": [0.0, 100.0, 200.0, 50.0],
            "solar_capacity": [1000.0] * 4,
            "solar_eclipse_pct": [0.0] * 4,
        },
        index=index,
    )
    weather = pd.DataFrame(
        {"cloud_cover_pct": [50.0, 40.0, 30.0, 60.0], "shortwave_radiation_wm2": [0.0, 200.0, 400.0, 100.0]},
        index=index,
    )

    X, y = build_solar_feature_table(df, weather)

    assert y.tolist() == [0.0, 0.1, 0.2, 0.05]
    assert {"cloud_cover_pct", "shortwave_radiation_wm2", "solar_elevation_deg", "solar_eclipse_pct"}.issubset(
        X.columns
    )
    assert list(X.index) == list(y.index)


def test_build_solar_feature_table_drops_rows_with_missing_weather():
    index = pd.date_range("2024-06-01", periods=3, freq="30min", tz="UTC")
    df = pd.DataFrame(
        {"solar": [0.0, 100.0, 200.0], "solar_capacity": [1000.0] * 3, "solar_eclipse_pct": [0.0] * 3},
        index=index,
    )
    weather = pd.DataFrame(
        {"cloud_cover_pct": [50.0, np.nan, 30.0], "shortwave_radiation_wm2": [0.0, 200.0, 400.0]},
        index=index,
    )

    X, _ = build_solar_feature_table(df, weather)

    assert index[1] not in X.index
    assert len(X) == 2
