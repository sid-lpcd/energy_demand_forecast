"""Statistical significance testing for forecast comparisons, and multiple-comparison correction.

Motivated by a gap in this project's own methodology, not by PLAN.md: every comparison run across
Weeks 1-8 and their follow-ups (weather ablation, direct-vs-recursive, the NDF/blend headline
result, CQR-vs-baseline calibration, ...) reports a point-estimate MAE/PICP delta ("27.65%
improvement", "3.55% improvement") with no test of whether that delta is distinguishable from
noise, and no correction for the fact that dozens of such comparisons were run over the project's
life -- exactly the setting where some "findings" are expected to be spurious by chance alone.

**Why not a plain paired t-test on the loss differential:** GB demand (and therefore forecast
error) is strongly autocorrelated at both the daily and weekly scale -- this project's own baseline
work (`edf.models.baselines`) is built entirely around that fact. A paired t-test assumes i.i.d.
differences and would understate the true variance of the mean loss differential, giving
artificially small p-values. The fix used here (`diebold_mariano_test`) is the standard tool for
exactly this problem (Diebold & Mariano, 1995, "Comparing Predictive Accuracy"): a Newey-West/
Bartlett-kernel HAC variance estimator on the loss-differential series, with truncation lag `h - 1`
where `h` is the forecast horizon in periods -- the theoretically motivated choice for h-step-ahead
forecast errors, which are order-(h-1) moving-average correlated even under a correctly specified
model. The Harvey-Leybourne-Newbold (1997) small-sample correction is applied by default, comparing
against a Student's t (T-1 df) rather than the raw asymptotic normal -- a strict improvement in
finite samples, and converges to the same answer at this project's sample sizes (thousands of rows).

**Multiple comparisons:** `benjamini_hochberg` implements the standard FDR-controlling procedure
(Benjamini & Hochberg, 1995) -- chosen over a Bonferroni-style family-wise error correction because
this project's comparisons are exploratory and largely independent research questions ("does
weather help", "does the blend beat NDF"), not a single confirmatory claim where any false positive
is unacceptable; FDR control is the standard, less conservative choice for that setting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats


def absolute_error_loss(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    return (y_true - y_pred).abs()


def squared_error_loss(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    return (y_true - y_pred) ** 2


def _newey_west_long_run_variance(d: np.ndarray, max_lag: int) -> float:
    """Bartlett-kernel HAC variance of `d`'s sample mean (Newey & West, 1987).

    Bartlett weights (`1 - k / (max_lag + 1)`) guarantee a non-negative estimate, unlike an
    unweighted sum of autocovariances. `max_lag=0` reduces to the plain sample variance.
    """
    t = len(d)
    d_centered = d - d.mean()
    gamma_0 = float(np.dot(d_centered, d_centered) / t)
    variance = gamma_0
    for lag in range(1, max_lag + 1):
        gamma_k = float(np.dot(d_centered[lag:], d_centered[:-lag]) / t)
        weight = 1 - lag / (max_lag + 1)
        variance += 2 * weight * gamma_k
    return variance


def diebold_mariano_test(
    loss_a: pd.Series,
    loss_b: pd.Series,
    h: int,
    small_sample_correction: bool = True,
) -> dict[str, float]:
    """Diebold-Mariano test of equal predictive accuracy between two aligned loss series.

    `loss_a`/`loss_b` are per-timestamp pointwise losses (e.g. `absolute_error_loss(y_true, pred)`)
    for two competing forecasts, already aligned on a shared index -- not the raw predictions
    themselves, so any loss function (MAE-style, squared-error, or a 0/1 interval-coverage
    indicator for a calibration comparison) can be tested with the same machinery. `h` is the
    forecast horizon in periods, used as the HAC truncation lag (`h - 1`); use `h=1` for
    one-step-ahead or non-overlapping forecasts.

    Returns `{"mean_diff": ..., "dm_stat": ..., "p_value": ..., "n": ...}`. `mean_diff` is
    `mean(loss_a - loss_b)`: negative means `a` has lower average loss (more accurate).
    Two-sided p-value -- this tests "are they different", not "which one is better" (read the sign
    of `mean_diff` / `dm_stat` for direction).
    """
    aligned = pd.DataFrame({"a": loss_a, "b": loss_b}).dropna()
    if len(aligned) < 2:
        raise ValueError("need at least 2 aligned, non-NaN observations for a DM test")
    d = (aligned["a"] - aligned["b"]).to_numpy()
    t = len(d)
    max_lag = max(h - 1, 0)

    d_bar = float(d.mean())
    long_run_variance = _newey_west_long_run_variance(d, max_lag)

    if long_run_variance <= 0:
        # Degenerate case: the two loss series are (near-)identical every period.
        return {"mean_diff": d_bar, "dm_stat": 0.0, "p_value": 1.0, "n": float(t)}

    dm_stat = d_bar / np.sqrt(long_run_variance / t)

    if small_sample_correction:
        # Harvey, Leybourne & Newbold (1997) correction; compare against Student's t(T-1).
        correction = np.sqrt((t + 1 - 2 * h + h * (h - 1) / t) / t)
        dm_stat *= correction
        p_value = float(2 * stats.t.sf(np.abs(dm_stat), df=t - 1))
    else:
        p_value = float(2 * stats.norm.sf(np.abs(dm_stat)))

    return {"mean_diff": d_bar, "dm_stat": float(dm_stat), "p_value": p_value, "n": float(t)}


def dm_test_mae(
    y_true: pd.Series, pred_a: pd.Series, pred_b: pd.Series, h: int
) -> dict[str, float]:
    """`diebold_mariano_test` on absolute-error loss -- the DM-test analogue of an MAE comparison."""
    return diebold_mariano_test(
        absolute_error_loss(y_true, pred_a), absolute_error_loss(y_true, pred_b), h=h
    )


def benjamini_hochberg(
    p_values: dict[str, float], alpha: float = 0.05
) -> pd.DataFrame:
    """Benjamini-Hochberg (1995) false-discovery-rate correction across a named set of p-values.

    Returns a `DataFrame` indexed by hypothesis name (original input order), with columns
    `p_value`, `rank` (1 = smallest p-value), `q_value` (the BH-adjusted p-value: the smallest FDR
    at which this hypothesis would be rejected), and `reject` (bool, at the given `alpha`).

    `q_value` is computed via the standard step-up formula, enforced monotonically non-decreasing
    from the largest p-value downward (`q_(i) = min(q_(i+1), m/i * p_(i))`) so adjusted p-values
    never decrease as raw p-values increase -- a `q_value` can otherwise exceed 1 or violate
    ordering without this step.
    """
    names = list(p_values.keys())
    raw = np.array([p_values[name] for name in names], dtype=float)
    m = len(raw)

    order = np.argsort(raw)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    sorted_p = raw[order]
    q_sorted = sorted_p * m / ranks[order]
    # enforce monotonicity from the largest p-value down (the standard BH step-up adjustment)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q_values = np.empty(m)
    q_values[order] = q_sorted

    result = pd.DataFrame(
        {"p_value": raw, "rank": ranks, "q_value": q_values, "reject": q_values <= alpha},
        index=names,
    )
    return result.sort_values("p_value")


def apply_significance_tests(
    tests: dict[str, Callable[[], dict[str, float]]], alpha: float = 0.05
) -> pd.DataFrame:
    """Run a named collection of significance-test callables (each returning a dict with a
    `p_value` key, e.g. `diebold_mariano_test`'s output), then apply `benjamini_hochberg` across
    all of them at once.

    Central place to run "every headline comparison this project makes, tested together" -- the
    honest way to report p-values when many hypotheses were explored, per this module's docstring.
    """
    raw_results = {name: test_fn() for name, test_fn in tests.items()}
    fdr = benjamini_hochberg({name: r["p_value"] for name, r in raw_results.items()}, alpha=alpha)
    extra = pd.DataFrame(raw_results).T.drop(columns=["p_value"])
    return fdr.join(extra).sort_values("p_value")
