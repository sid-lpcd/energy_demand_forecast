import numpy as np
import pandas as pd
import pytest

from edf.significance import (
    absolute_error_loss,
    apply_significance_tests,
    benjamini_hochberg,
    diebold_mariano_test,
    dm_test_mae,
    squared_error_loss,
)


def test_absolute_error_loss():
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = pd.Series([12.0, 18.0, 33.0])
    pd.testing.assert_series_equal(absolute_error_loss(y_true, y_pred), pd.Series([2.0, 2.0, 3.0]))


def test_squared_error_loss():
    y_true = pd.Series([10.0, 20.0])
    y_pred = pd.Series([12.0, 18.0])
    pd.testing.assert_series_equal(squared_error_loss(y_true, y_pred), pd.Series([4.0, 4.0]))


def test_dm_test_identical_forecasts_gives_p_value_one():
    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.normal(size=500))
    pred = y_true + rng.normal(scale=0.1, size=500)
    result = diebold_mariano_test(
        absolute_error_loss(y_true, pred), absolute_error_loss(y_true, pred), h=1
    )
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_dm_test_detects_a_clearly_better_forecast():
    rng = np.random.default_rng(1)
    n = 2000
    y_true = pd.Series(rng.normal(scale=10, size=n))
    good_pred = y_true + rng.normal(scale=1.0, size=n)
    bad_pred = y_true + rng.normal(scale=5.0, size=n)

    result = dm_test_mae(y_true, good_pred, bad_pred, h=1)

    assert result["mean_diff"] < 0  # good_pred has lower average loss
    assert result["p_value"] < 0.01


def test_dm_test_no_difference_gives_large_p_value():
    rng = np.random.default_rng(2)
    n = 2000
    y_true = pd.Series(rng.normal(scale=10, size=n))
    pred_a = y_true + rng.normal(scale=2.0, size=n)
    pred_b = y_true + rng.normal(scale=2.0, size=n)

    result = dm_test_mae(y_true, pred_a, pred_b, h=1)

    assert result["p_value"] > 0.05


def test_dm_test_respects_horizon_lag_widens_variance_under_autocorrelated_noise():
    # Autocorrelated (not i.i.d.) loss differential -- h=1 (no HAC correction) should understate
    # variance relative to h=48, giving a smaller p-value for the same data. This is exactly the
    # failure mode a naive paired t-test has on this project's autocorrelated demand series.
    rng = np.random.default_rng(3)
    n = 3000
    ar_noise = np.zeros(n)
    for i in range(1, n):
        ar_noise[i] = 0.9 * ar_noise[i - 1] + rng.normal(scale=1.0)
    y_true = pd.Series(rng.normal(scale=10, size=n))
    pred_a = y_true + pd.Series(ar_noise)
    pred_b = y_true + pd.Series(ar_noise) + 0.05  # tiny, mostly-noise difference

    result_h1 = dm_test_mae(y_true, pred_a, pred_b, h=1)
    result_h48 = dm_test_mae(y_true, pred_a, pred_b, h=48)

    assert result_h48["p_value"] >= result_h1["p_value"]


def test_dm_test_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        diebold_mariano_test(pd.Series([1.0]), pd.Series([1.0]), h=1)


def test_dm_test_drops_unaligned_nan_rows():
    loss_a = pd.Series([1.0, 2.0, np.nan, 3.0, 2.5, 1.5, 2.0])
    loss_b = pd.Series([1.1, 2.1, 0.5, np.nan, 2.6, 1.4, 2.1])
    result = diebold_mariano_test(loss_a, loss_b, h=1)
    assert result["n"] == 5


def test_benjamini_hochberg_all_significant_when_all_p_values_tiny():
    result = benjamini_hochberg({"a": 0.001, "b": 0.002, "c": 0.0005}, alpha=0.05)
    assert result["reject"].all()
    assert list(result.index) == ["c", "a", "b"]  # sorted ascending by p_value


def test_benjamini_hochberg_controls_false_discoveries_among_mixed_p_values():
    # Classic textbook-style example: a handful of clearly significant p-values mixed with
    # several that are individually below 0.05 but shouldn't survive FDR correction.
    p_values = {
        "strong_1": 0.0001,
        "strong_2": 0.0008,
        "borderline_1": 0.03,
        "borderline_2": 0.04,
        "borderline_3": 0.045,
        "null_1": 0.6,
        "null_2": 0.8,
    }
    result = benjamini_hochberg(p_values, alpha=0.05)
    assert result.loc["strong_1", "reject"]
    assert result.loc["strong_2", "reject"]
    assert not result.loc["null_1", "reject"]
    assert not result.loc["null_2", "reject"]


def test_benjamini_hochberg_q_values_are_monotonic_with_rank():
    result = benjamini_hochberg({"a": 0.5, "b": 0.01, "c": 0.2, "d": 0.001}, alpha=0.05)
    by_rank = result.sort_values("rank")
    assert by_rank["q_value"].is_monotonic_increasing


def test_benjamini_hochberg_q_value_never_exceeds_one():
    result = benjamini_hochberg({"a": 0.9, "b": 0.99}, alpha=0.05)
    assert (result["q_value"] <= 1.0).all()


def test_apply_significance_tests_combines_and_corrects():
    rng = np.random.default_rng(4)
    n = 1000
    y_true = pd.Series(rng.normal(scale=10, size=n))
    strong_a = y_true + rng.normal(scale=1.0, size=n)
    strong_b = y_true + rng.normal(scale=6.0, size=n)
    weak_a = y_true + rng.normal(scale=2.0, size=n)
    weak_b = y_true + rng.normal(scale=2.0, size=n)

    tests = {
        "strong_effect": lambda: dm_test_mae(y_true, strong_a, strong_b, h=1),
        "weak_effect": lambda: dm_test_mae(y_true, weak_a, weak_b, h=1),
    }
    result = apply_significance_tests(tests, alpha=0.05)

    assert result.loc["strong_effect", "reject"]
    assert not result.loc["weak_effect", "reject"]
    assert "mean_diff" in result.columns
    assert "dm_stat" in result.columns
