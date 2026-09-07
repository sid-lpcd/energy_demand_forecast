import pandas as pd

from edf.quantile import enforce_monotonic_quantiles, quantile_crossing_rate


def test_quantile_crossing_rate_zero_when_already_monotonic():
    index = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    predictions = {
        0.1: pd.Series([10.0, 20.0, 30.0], index=index),
        0.5: pd.Series([15.0, 25.0, 35.0], index=index),
        0.9: pd.Series([20.0, 30.0, 40.0], index=index),
    }
    assert quantile_crossing_rate(predictions) == 0.0


def test_quantile_crossing_rate_detects_violations():
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    predictions = {
        0.1: pd.Series([15.0, 10.0], index=index),  # row 0 crosses: P10 > P50
        0.5: pd.Series([10.0, 20.0], index=index),
        0.9: pd.Series([20.0, 30.0], index=index),
    }
    assert quantile_crossing_rate(predictions) == 0.5


def test_enforce_monotonic_quantiles_leaves_already_sorted_rows_unchanged():
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    predictions = {
        0.1: pd.Series([10.0, 20.0], index=index),
        0.5: pd.Series([15.0, 25.0], index=index),
        0.9: pd.Series([20.0, 30.0], index=index),
    }
    result = enforce_monotonic_quantiles(predictions)
    assert result[0.1].tolist() == [10.0, 20.0]
    assert result[0.5].tolist() == [15.0, 25.0]
    assert result[0.9].tolist() == [20.0, 30.0]


def test_enforce_monotonic_quantiles_sorts_crossed_row():
    index = pd.date_range("2024-01-01", periods=1, freq="30min", tz="UTC")
    predictions = {
        0.1: pd.Series([15.0], index=index),  # crossed: 15 > 10
        0.5: pd.Series([10.0], index=index),
        0.9: pd.Series([20.0], index=index),
    }
    result = enforce_monotonic_quantiles(predictions)
    assert result[0.1].iloc[0] == 10.0
    assert result[0.5].iloc[0] == 15.0
    assert result[0.9].iloc[0] == 20.0


def test_enforce_monotonic_quantiles_preserves_index_and_alpha_keys():
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    predictions = {
        0.1: pd.Series([1.0, 2.0], index=index),
        0.9: pd.Series([3.0, 4.0], index=index),
    }
    result = enforce_monotonic_quantiles(predictions)
    assert set(result) == {0.1, 0.9}
    assert list(result[0.1].index) == list(index)
