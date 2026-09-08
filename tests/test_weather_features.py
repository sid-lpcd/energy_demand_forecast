import numpy as np
import pandas as pd

from edf.weather_features import (
    build_weather_feature_table,
    cooling_degree,
    cumulative_degree,
    heating_degree,
    load_weather_series,
)


def test_heating_degree_zero_when_at_or_above_base():
    temps = pd.Series([15.5, 20.0, 30.0])
    assert (heating_degree(temps) == 0).all()


def test_heating_degree_positive_below_base():
    temps = pd.Series([10.5, 0.0])
    result = heating_degree(temps)
    assert result.tolist() == [5.0, 15.5]


def test_cooling_degree_zero_when_at_or_below_base():
    temps = pd.Series([22.0, 15.0, -5.0])
    assert (cooling_degree(temps) == 0).all()


def test_cooling_degree_positive_above_base():
    temps = pd.Series([25.0, 30.0])
    result = cooling_degree(temps)
    assert result.tolist() == [3.0, 8.0]


def test_cumulative_degree_sums_trailing_window():
    degree = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = cumulative_degree(degree, window_periods=3)
    # first two entries are NaN (window not yet full); from index 2 onward,
    # each value is the sum of itself and the two preceding entries.
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == [6.0, 9.0, 12.0]


def test_cumulative_degree_only_looks_backward():
    # a spike at the end must not leak into earlier windows' sums
    degree = pd.Series([0.0, 0.0, 0.0, 100.0])
    result = cumulative_degree(degree, window_periods=2)
    assert result.iloc[2] == 0.0
    assert result.iloc[3] == 100.0


def _write_hourly_source(tmp_path, name: str) -> None:
    hourly_index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": hourly_index,
            "temperature_c": [10.0, 12.0, 14.0],
            "wind_speed_ms": [1.0, 2.0, 3.0],
        }
    )
    df.to_parquet(tmp_path / f"{name}.parquet", index=False)


def test_load_weather_series_interpolates_hourly_to_half_hourly(tmp_path):
    _write_hourly_source(tmp_path, "test_source")
    target_index = pd.date_range("2024-01-01", periods=5, freq="30min", tz="UTC")

    result = load_weather_series("test_source", target_index, raw_dir=tmp_path)

    np.testing.assert_allclose(
        result["temperature_c"].to_numpy(), [10.0, 11.0, 12.0, 13.0, 14.0]
    )


def test_load_weather_series_nan_outside_source_range(tmp_path):
    _write_hourly_source(tmp_path, "test_source")
    # source only covers 2024-01-01 00:00-02:00; ask for a point well after it
    target_index = pd.date_range("2024-01-02", periods=1, freq="30min", tz="UTC")

    result = load_weather_series("test_source", target_index, raw_dir=tmp_path)

    assert result["temperature_c"].isna().all()


def test_build_weather_feature_table_includes_degree_columns(tmp_path):
    _write_hourly_source(tmp_path, "test_source")
    target_index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")

    result = build_weather_feature_table("test_source", target_index, raw_dir=tmp_path)

    assert "heating_degree" in result.columns
    assert "cooling_degree" in result.columns
    np.testing.assert_allclose(result["heating_degree"].to_numpy(), [5.5, 3.5, 1.5])
    assert (result["cooling_degree"] == 0).all()
