import numpy as np
import pandas as pd

from edf.generation_features import build_wind_feature_table, capacity_factor


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
