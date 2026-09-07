import numpy as np
import pandas as pd
import pytest

from edf.evaluate import (
    bias,
    compare_models,
    evaluate,
    evaluate_by_bucket,
    evaluate_by_fold,
    evaluate_quantiles_by_bucket,
    interval_sharpness,
    mae,
    mape,
    mase,
    p95_abs_error,
    picp,
    pinball_loss,
    quantile_coverage,
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


def test_bias_sign_convention_positive_means_under_forecasting():
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([8.0, 16.0])  # forecast consistently below actual
    assert bias(y_true, y_pred) == pytest.approx((2 + 4) / 2)


def test_bias_sign_convention_negative_means_over_forecasting():
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([12.0, 24.0])  # forecast consistently above actual
    assert bias(y_true, y_pred) == pytest.approx((-2 - 4) / 2)


def test_bias_zero_for_unbiased_but_noisy_forecast():
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([8.0, 22.0])  # errors cancel out, despite nonzero MAE
    assert bias(y_true, y_pred) == pytest.approx(0.0)
    assert mae(y_true, y_pred) == pytest.approx(2.0)


def test_p95_abs_error_matches_quantile_of_absolute_errors():
    y_true = pd.Series(range(100), dtype=float)
    y_pred = pd.Series([0.0] * 100)  # absolute error == y_true itself, 0..99
    assert p95_abs_error(y_true, y_pred) == pytest.approx(y_true.quantile(0.95))


def test_pinball_loss_zero_for_perfect_prediction():
    y_true = pd.Series([10.0, 20.0, 30.0])
    assert pinball_loss(y_true, y_true, alpha=0.1) == pytest.approx(0.0)
    assert pinball_loss(y_true, y_true, alpha=0.9) == pytest.approx(0.0)


def test_pinball_loss_penalizes_underprediction_more_at_high_alpha():
    # at alpha=0.9, under-predicting (forecast below actual) should cost
    # more than over-predicting by the same margin -- that asymmetry is
    # exactly what makes alpha=0.9 training converge to the 90th percentile.
    y_true = pd.Series([100.0])
    under = pinball_loss(y_true, pd.Series([90.0]), alpha=0.9)  # 10 below actual
    over = pinball_loss(y_true, pd.Series([110.0]), alpha=0.9)  # 10 above actual
    assert under > over


def test_pinball_loss_penalizes_overprediction_more_at_low_alpha():
    y_true = pd.Series([100.0])
    under = pinball_loss(y_true, pd.Series([90.0]), alpha=0.1)
    over = pinball_loss(y_true, pd.Series([110.0]), alpha=0.1)
    assert over > under


def test_quantile_coverage_matches_fraction_at_or_below():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_quantile = pd.Series([3.0] * 5)  # 3 of 5 actuals are <= 3
    assert quantile_coverage(y_true, y_pred_quantile) == pytest.approx(0.6)


def test_picp_counts_actuals_inside_the_interval():
    y_true = pd.Series([1.0, 5.0, 10.0, 15.0, 20.0])
    lower = pd.Series([2.0] * 5)
    upper = pd.Series([16.0] * 5)
    # 5, 10, 15 fall inside [2, 16]; 1 and 20 don't
    assert picp(y_true, lower, upper) == pytest.approx(3 / 5)


def test_interval_sharpness_is_mean_width():
    lower = pd.Series([10.0, 20.0])
    upper = pd.Series([15.0, 40.0])
    assert interval_sharpness(lower, upper) == pytest.approx((5.0 + 20.0) / 2)


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


def test_evaluate_by_bucket_scores_all_row_and_each_bucket():
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    y_pred = pd.Series([12.0, 18.0, 33.0, 44.0], index=index)
    buckets = pd.DataFrame(
        {"is_cold": [True, True, False, False], "is_weekend": [False, True, True, False]},
        index=index,
    )

    result = evaluate_by_bucket(y_true, y_pred, buckets)

    assert list(result.index) == ["all", "is_cold", "is_weekend"]
    assert result.loc["all", "n"] == 4
    assert result.loc["is_cold", "n"] == 2
    assert result.loc["is_cold", "mae"] == pytest.approx((2 + 2) / 2)
    assert result.loc["is_weekend", "n"] == 2
    assert result.loc["is_weekend", "mae"] == pytest.approx((2 + 3) / 2)


def test_evaluate_by_bucket_uses_per_bucket_seasonal_naive_mae():
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    y_pred = pd.Series([11.0, 21.0, 31.0, 41.0], index=index)  # error 1 everywhere
    # naive is nearly exact in the cold bucket, way off outside it
    naive = pd.Series([10.1, 20.1, 0.0, 0.0], index=index)
    buckets = pd.DataFrame({"is_cold": [True, True, False, False]}, index=index)

    result = evaluate_by_bucket(y_true, y_pred, buckets, seasonal_naive_pred=naive)

    # naive MAE is tiny in is_cold -> MASE there is much larger than 1
    assert result.loc["is_cold", "mase"] > 1.0
    # naive MAE outside is_cold is large, so MASE there is small
    assert result.loc["all", "mase"] < 1.0


def test_evaluate_quantiles_by_bucket_reports_pinball_and_picp_per_bucket():
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    quantile_preds = {
        0.1: pd.Series([5.0, 15.0, 25.0, 35.0], index=index),
        0.5: pd.Series([10.0, 20.0, 30.0, 40.0], index=index),
        0.9: pd.Series([15.0, 25.0, 35.0, 45.0], index=index),
    }
    buckets = pd.DataFrame({"is_cold": [True, True, False, False]}, index=index)

    result = evaluate_quantiles_by_bucket(y_true, quantile_preds, buckets)

    assert list(result.index) == ["all", "is_cold"]
    assert {"pinball_0.1", "pinball_0.5", "pinball_0.9", "picp", "n"}.issubset(result.columns)
    # actual is always inside [P10, P90] in this fixture -> PICP == 1.0 everywhere
    assert result.loc["all", "picp"] == pytest.approx(1.0)
    assert result.loc["is_cold", "picp"] == pytest.approx(1.0)
    assert result.loc["is_cold", "n"] == 2
