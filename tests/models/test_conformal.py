import numpy as np
import pandas as pd
import pytest

from edf.models.conformal import (
    apply_cqr_correction,
    conformity_scores,
    fit_cqr_correction,
)


def test_conformity_scores_zero_when_actual_exactly_on_boundary():
    y = pd.Series([10.0, 20.0])
    lower = pd.Series([10.0, 15.0])
    upper = pd.Series([15.0, 20.0])
    scores = conformity_scores(y, lower, upper)
    assert scores.tolist() == [0.0, 0.0]


def test_conformity_scores_positive_when_actual_outside_interval():
    y = pd.Series([5.0, 25.0])
    lower = pd.Series([10.0, 10.0])
    upper = pd.Series([20.0, 20.0])
    scores = conformity_scores(y, lower, upper)
    assert scores.tolist() == [5.0, 5.0]


def test_conformity_scores_negative_when_actual_well_inside_interval():
    y = pd.Series([15.0])
    lower = pd.Series([10.0])
    upper = pd.Series([20.0])
    scores = conformity_scores(y, lower, upper)
    assert scores[0] == -5.0


def test_fit_cqr_correction_zero_when_interval_already_covers_exactly_at_target_rate():
    # 10 calibration rows, exactly 8 inside [lower, upper] (80% raw coverage) and
    # the 2 outside sit right on the boundary (score 0) -- correction should be ~0.
    y = pd.Series(np.arange(10, dtype=float))
    lower = pd.Series(np.full(10, 0.0))
    upper = pd.Series(np.full(10, 9.0))
    q_hat = fit_cqr_correction(y, lower, upper, coverage=0.8)
    assert q_hat == pytest.approx(0.0)


def test_fit_cqr_correction_positive_when_interval_too_narrow():
    rng = np.random.default_rng(0)
    y_calib = pd.Series(rng.normal(scale=10.0, size=500))
    # Deliberately too-narrow interval: raw coverage well below the 80% nominal.
    lower_calib = pd.Series(np.full(500, -2.0))
    upper_calib = pd.Series(np.full(500, 2.0))
    q_hat = fit_cqr_correction(y_calib, lower_calib, upper_calib, coverage=0.8)
    assert q_hat > 0


def test_apply_cqr_correction_widens_interval_by_q_hat():
    lower = pd.Series([10.0, 20.0])
    upper = pd.Series([15.0, 25.0])
    new_lower, new_upper = apply_cqr_correction(lower, upper, q_hat=2.0)
    assert new_lower.tolist() == [8.0, 18.0]
    assert new_upper.tolist() == [17.0, 27.0]


def test_cqr_correction_achieves_at_least_nominal_coverage_on_held_out_data():
    # The actual guarantee this whole module exists for: fit q_hat on one
    # (exchangeable) sample, apply it to a *different* held-out sample from
    # the same distribution, and check achieved coverage is close to nominal.
    rng = np.random.default_rng(1)
    n = 2000
    y_calib = pd.Series(rng.normal(size=n))
    y_test = pd.Series(rng.normal(size=n))
    # An intentionally overconfident (too-narrow) raw interval, same on both sets.
    lower_calib, upper_calib = pd.Series(np.full(n, -0.5)), pd.Series(np.full(n, 0.5))
    lower_test, upper_test = pd.Series(np.full(n, -0.5)), pd.Series(np.full(n, 0.5))

    q_hat = fit_cqr_correction(y_calib, lower_calib, upper_calib, coverage=0.8)
    corrected_lower, corrected_upper = apply_cqr_correction(lower_test, upper_test, q_hat)

    achieved_coverage = ((y_test >= corrected_lower) & (y_test <= corrected_upper)).mean()
    assert achieved_coverage >= 0.78  # allow small finite-sample slack around the 0.8 target
