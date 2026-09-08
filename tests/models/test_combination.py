import numpy as np
import pandas as pd
import pytest

from edf.models.combination import combine_forecasts, fit_combination_weight


def test_combine_forecasts_weighted_average():
    pred_a = pd.Series([10.0, 20.0])
    pred_b = pd.Series([0.0, 0.0])
    result = combine_forecasts(pred_a, pred_b, weight_a=0.25)
    assert result.tolist() == [2.5, 5.0]


def test_fit_combination_weight_favours_the_more_accurate_forecast():
    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.normal(size=2000))
    pred_a = y_true + rng.normal(scale=0.1, size=2000)  # accurate, low-noise
    pred_b = y_true + rng.normal(scale=5.0, size=2000)  # much noisier

    weight_a = fit_combination_weight(y_true, pred_a, pred_b)

    assert weight_a > 0.9


def test_fit_combination_weight_equal_uncorrelated_errors_gives_half():
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.normal(size=5000))
    pred_a = y_true + rng.normal(scale=1.0, size=5000)
    pred_b = y_true + rng.normal(scale=1.0, size=5000)

    weight_a = fit_combination_weight(y_true, pred_a, pred_b)

    assert weight_a == pytest.approx(0.5, abs=0.05)


def test_combination_of_a_weaker_forecast_can_still_beat_the_stronger_one_alone():
    # Reproduces the Bates-Granger result this project's own notebook relies on:
    # forecast B is much noisier than A but has partly-decorrelated errors, so a
    # small weight on B still reduces the combined error below A's alone.
    rng = np.random.default_rng(2)
    n = 5000
    shared_noise = rng.normal(scale=1.0, size=n)
    y_true = pd.Series(rng.normal(size=n))
    pred_a = y_true + shared_noise * 0.3 + rng.normal(scale=0.5, size=n)
    pred_b = y_true + shared_noise * 5.0 + rng.normal(scale=1.0, size=n)

    weight_a = fit_combination_weight(y_true, pred_a, pred_b)
    combined = combine_forecasts(pred_a, pred_b, weight_a)

    mae_a = (y_true - pred_a).abs().mean()
    mae_combined = (y_true - combined).abs().mean()
    assert mae_combined < mae_a
