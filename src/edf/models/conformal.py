"""Conformalized Quantile Regression (CQR) — Stretch goal, PLAN.md.

Romano, Patterson & Candès (2019), "Conformalized Quantile Regression":
wraps an existing quantile regressor's [lower, upper] interval with a single
scalar correction, fit on a held-out calibration set, giving a finite-sample
marginal coverage guarantee regardless of whether the underlying quantile
model is itself well-calibrated. Directly targets the tail-compression
failure Weeks 5-7 found in this project's LightGBM quantile model (PICP
65.2% for a nominal 80% interval, both tails compressed toward the median) —
CQR corrects PICP by construction, by widening or narrowing the interval, not
by retraining anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conformity_scores(
    y_calib: pd.Series, lower_calib: pd.Series, upper_calib: pd.Series
) -> np.ndarray:
    """Signed distance each calibration-set actual falls outside [lower, upper].

    Positive when the actual falls outside the interval on either side,
    negative (but capped at the shortfall, not zero) when it's already
    inside — the standard CQR nonconformity score.
    """
    return np.maximum(lower_calib - y_calib, y_calib - upper_calib).to_numpy()


def fit_cqr_correction(
    y_calib: pd.Series,
    lower_calib: pd.Series,
    upper_calib: pd.Series,
    coverage: float = 0.8,
) -> float:
    """The additive correction `q_hat` that gives marginal coverage `coverage`.

    Uses Romano et al.'s finite-sample-corrected empirical quantile of the
    calibration conformity scores — `ceil((n+1) * coverage) / n`, not a plain
    `coverage`-quantile — which is what makes the resulting coverage
    guarantee exact (given an exchangeable calibration set) rather than only
    asymptotic.
    """
    scores = conformity_scores(y_calib, lower_calib, upper_calib)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_cqr_correction(
    lower: pd.Series, upper: pd.Series, q_hat: float
) -> tuple[pd.Series, pd.Series]:
    """Widen (or, if `q_hat` < 0, narrow) [lower, upper] by the fitted correction."""
    return lower - q_hat, upper + q_hat
