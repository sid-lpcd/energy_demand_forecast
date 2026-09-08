"""Forecast combination (Week 8 follow-up): blending two independent forecasts.

Bates & Granger (1969)'s classical result: a linear combination of two
forecasts can beat the more accurate one alone, even when the other is much
weaker, *if* their errors aren't too highly correlated — the weaker forecast
still contributes decorrelated information the stronger one lacks. Verified
empirically for this project in `notebooks/20`: our from-scratch model loses
to Elexon's NDF forecast in every regime checked, but a ~16%-weighted blend
of the two beats NDF alone by 3.75%, fit and evaluated on non-overlapping
halves of `VALIDATION` to keep the comparison honest.
"""

from __future__ import annotations

import pandas as pd


def fit_combination_weight(y_true: pd.Series, pred_a: pd.Series, pred_b: pd.Series) -> float:
    """Minimum-variance combination weight on `pred_a` (weight on `pred_b` is `1 - w`).

    The standard Bates-Granger formula: `w = (var_b - cov) / (var_a + var_b - 2*cov)`,
    derived from minimizing `Var(w*e_a + (1-w)*e_b)` over the two forecasts'
    error series. Not clipped to `[0, 1]` -- a weight outside that range is a
    legitimate (if less common) result when one forecast's errors are
    strongly correlated with, but larger than, the other's; callers should
    treat an extreme weight as a signal to inspect the inputs, not silently
    clip it.

    Fit only on data the resulting weight will not later be evaluated
    against -- e.g. one split of `VALIDATION` -- or the reported combination
    accuracy will be optimistic, the same walk-forward discipline as every
    other fitted parameter in this project.
    """
    error_a = y_true - pred_a
    error_b = y_true - pred_b
    var_a, var_b = error_a.var(), error_b.var()
    covariance = error_a.cov(error_b)
    return (var_b - covariance) / (var_a + var_b - 2 * covariance)


def combine_forecasts(pred_a: pd.Series, pred_b: pd.Series, weight_a: float) -> pd.Series:
    """Weighted-average combination: `weight_a * pred_a + (1 - weight_a) * pred_b`."""
    return weight_a * pred_a + (1 - weight_a) * pred_b
