import numpy as np
import pandas as pd
import pytest

from edf.evaluate import (
    compare_models,
    evaluate,
    evaluate_by_fold,
    mae,
    mape,
    mase,
    rmse,
    yearly_folds,
)


def test_mae():
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, 33.0])
    assert mae(y_true, y_pred) == pytest.approx((2 + 2 + 3) / 3)


def test_rmse():
    y_true = pd.Series([0.0, 0.0])
    y_pred = pd.Series([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((9 + 16) / 2))


def test_mape():
    y_true = pd.Series([100.0, 200.0])
    y_pred = pd.Series([110.0, 180.0])
    assert mape(y_true, y_pred) == pytest.approx((0.10 + 0.10) / 2 * 100)


def test_mase_scales_by_seasonal_naive_mae():
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([12.0, 18.0])
    assert mase(y_true, y_pred, seasonal_naive_mae=4.0) == pytest.approx(2.0 / 4.0)


def test_evaluate_drops_nan_rows_before_scoring():
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    y_pred = pd.Series([np.nan, 18.0, 33.0, np.nan], index=index)

    result = evaluate(y_true, y_pred)

    assert result["n"] == 2
    assert result["mae"] == pytest.approx((2 + 3) / 2)


def test_evaluate_includes_mase_only_when_seasonal_naive_mae_given():
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([12.0, 18.0])

    assert "mase" not in evaluate(y_true, y_pred)
    assert "mase" in evaluate(y_true, y_pred, seasonal_naive_mae=4.0)


def test_yearly_folds_covers_each_calendar_year():
    index = pd.date_range("2023-06-01", "2024-06-01", freq="30min", tz="UTC")
    folds = yearly_folds(index)

    assert [start.year for start, _ in folds] == [2023, 2024]
    assert folds[0][0] == pd.Timestamp("2023-01-01", tz="UTC")
    assert folds[0][1] == pd.Timestamp("2023-12-31 23:59:59", tz="UTC")


def test_evaluate_by_fold_scores_each_fold_independently():
    index = pd.DatetimeIndex(
        ["2023-03-01", "2023-09-01", "2024-03-01"], tz="UTC"
    )
    y_true = pd.Series([10.0, 20.0, 30.0], index=index)
    y_pred = pd.Series([11.0, 22.0, 27.0], index=index)
    folds = yearly_folds(index)

    result = evaluate_by_fold(y_true, y_pred, folds)

    assert list(result.index) == [2023, 2024]
    assert result.loc[2023, "n"] == 2
    assert result.loc[2024, "n"] == 1


def test_compare_models_uses_seasonal_naive_per_fold_mae_as_mase_denominator():
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    predictions = {
        "seasonal_naive": pd.Series([8.0, 18.0, 32.0, 42.0], index=index),
        "better_model": pd.Series([9.0, 19.0, 31.0, 41.0], index=index),
    }
    folds = yearly_folds(index)

    result = compare_models(y_true, predictions, folds, seasonal_naive_name="seasonal_naive")

    # The seasonal-naive model always scores MASE == 1.0 against itself.
    assert result.loc[("seasonal_naive", 2024), "mase"] == pytest.approx(1.0)
    # A model with half the seasonal-naive's error scores MASE == 0.5.
    assert result.loc[("better_model", 2024), "mase"] == pytest.approx(0.5)
