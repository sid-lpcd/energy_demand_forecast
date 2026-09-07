"""Metrics and the walk-forward evaluation harness reused by every model, Week 2 onward.

Built once here so Weeks 3-7 don't each rewrite scoring logic (see PLAN.md).
This module only knows how to fold a date range into per-calendar-year
windows and score prediction series against actuals over those windows — it
has no opinion on which split (train/validation/test, fixed in `edf.config`)
a caller should look at. In particular it does not defend against evaluating
on `edf.config.TEST`; callers must respect that themselves, per PLAN.md's
"don't peek at test data" rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error, in percent. Assumes y_true is never zero."""
    return float(((y_true - y_pred).abs() / y_true.abs()).mean() * 100)


def mase(y_true: pd.Series, y_pred: pd.Series, seasonal_naive_mae: float) -> float:
    """MAE scaled by a seasonal-naive baseline's MAE over the same evaluation window.

    <1 means the model beats seasonal-naive (previous-week-same-time); >=1
    means it doesn't. `seasonal_naive_mae` must come from scoring the same
    y_true window, or the two numbers aren't comparable.
    """
    return mae(y_true, y_pred) / seasonal_naive_mae


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean signed error (actual - forecast). +ve = under-forecasting on average,
    -ve = over-forecasting. MAE/RMSE only see error *magnitude*, so a model can
    have a small MAE while still being systematically off in one direction —
    this is what catches that."""
    return float((y_true - y_pred).mean())


def p95_abs_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """95th percentile of absolute error — a cheap tail-risk read: how bad are
    this model's worst-case misses, not just its typical one (MAE/RMSE)."""
    return float((y_true - y_pred).abs().quantile(0.95))


def evaluate(
    y_true: pd.Series, y_pred: pd.Series, seasonal_naive_mae: float | None = None
) -> dict[str, float]:
    """MAE/RMSE/MAPE/bias/P95 error (and MASE, if a seasonal-naive MAE is given).

    Rows where either series is NaN — e.g. a baseline's warm-up period, before
    its longest lag reaches back to the start of the data — are dropped
    before scoring.
    """
    aligned = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    y_t, y_p = aligned["y_true"], aligned["y_pred"]
    metrics = {
        "mae": mae(y_t, y_p),
        "rmse": rmse(y_t, y_p),
        "mape": mape(y_t, y_p),
        "bias": bias(y_t, y_p),
        "p95_abs_error": p95_abs_error(y_t, y_p),
        "n": float(len(aligned)),
    }
    if seasonal_naive_mae is not None:
        metrics["mase"] = metrics["mae"] / seasonal_naive_mae
    return metrics


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, alpha: float) -> float:
    """Quantile (pinball) loss at quantile level `alpha` (0 < alpha < 1).

    Asymmetric: under- and over-prediction are penalized in proportion to
    `alpha` vs. `1 - alpha`, so the loss is minimized exactly at the true
    alpha-quantile of the conditional distribution — this asymmetry is the
    whole reason training with it produces calibrated quantiles at all,
    rather than just another point estimate.
    """
    error = y_true - y_pred
    return float(np.maximum(alpha * error, (alpha - 1) * error).mean())


def quantile_coverage(y_true: pd.Series, y_pred_quantile: pd.Series) -> float:
    """Fraction of actuals at or below a predicted quantile.

    Should equal that quantile's `alpha` if the model is well calibrated —
    the building block for a reliability diagram (nominal alpha on one
    axis, this observed value on the other).
    """
    return float((y_true <= y_pred_quantile).mean())


def picp(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Prediction interval coverage probability: the fraction of actuals
    falling inside [lower, upper]. For a nominal interval (e.g. P10/P90,
    nominally 80%), a well-calibrated model's PICP should sit close to that
    nominal level — well below it means the interval is too narrow
    (overconfident), well above means too wide (underconfident).
    """
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def interval_sharpness(lower: pd.Series, upper: pd.Series) -> float:
    """Mean prediction-interval width.

    A well-calibrated but very wide interval is less useful than a narrow
    one, so this is always reported alongside `picp`, never instead of it —
    calibration alone can't tell a genuinely sharp forecast from one that's
    just wide enough to "cover" anything.
    """
    return float((upper - lower).mean())


def evaluate_by_bucket(
    y_true: pd.Series,
    y_pred: pd.Series,
    buckets: pd.DataFrame,
    seasonal_naive_pred: pd.Series | None = None,
) -> pd.DataFrame:
    """Score (y_true, y_pred) restricted to each boolean column in `buckets`
    (Week 6 day-type flags — not mutually exclusive, a row can be in
    several), plus an `"all"` row for the unrestricted reference.

    If `seasonal_naive_pred` is given, each bucket's MASE is scaled by
    *that bucket's own* seasonal-naive MAE, not one overall value — so a
    bucket's MASE reflects whether the model struggles more there than the
    naive baseline also would, not just a shift in absolute error size.
    """
    rows = {}
    for name in ["all", *buckets.columns]:
        mask = pd.Series(True, index=y_true.index) if name == "all" else buckets[name]
        naive_mae = (
            evaluate(y_true[mask], seasonal_naive_pred[mask])["mae"]
            if seasonal_naive_pred is not None
            else None
        )
        rows[name] = evaluate(y_true[mask], y_pred[mask], seasonal_naive_mae=naive_mae)
    return pd.DataFrame(rows).T


def evaluate_quantiles_by_bucket(
    y_true: pd.Series,
    quantile_preds: dict[float, pd.Series],
    buckets: pd.DataFrame,
    low_alpha: float = 0.1,
    high_alpha: float = 0.9,
) -> pd.DataFrame:
    """Pinball loss per quantile + PICP[low_alpha, high_alpha], restricted
    to each bucket in `buckets` (plus an `"all"` row).
    """
    rows = {}
    for name in ["all", *buckets.columns]:
        mask = pd.Series(True, index=y_true.index) if name == "all" else buckets[name]
        row = {
            f"pinball_{alpha}": pinball_loss(y_true[mask], pred[mask], alpha)
            for alpha, pred in quantile_preds.items()
        }
        row["picp"] = picp(
            y_true[mask], quantile_preds[low_alpha][mask], quantile_preds[high_alpha][mask]
        )
        row["n"] = int(mask.sum())
        rows[name] = row
    return pd.DataFrame(rows).T


def yearly_folds(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) boundaries for one fold per calendar year present in `index`.

    Boundaries only — this generator doesn't fit or score anything. A later
    model fits on data strictly before a fold's `start` and scores on
    [start, end]; enforcing "don't use the future" is the caller's job.
    """
    years = sorted(index.year.unique())
    return [
        (
            pd.Timestamp(f"{year}-01-01", tz=index.tz),
            pd.Timestamp(f"{year}-12-31 23:59:59", tz=index.tz),
        )
        for year in years
    ]


def evaluate_by_fold(
    y_true: pd.Series,
    y_pred: pd.Series,
    folds: list[tuple[pd.Timestamp, pd.Timestamp]],
    seasonal_naive_mae_by_fold: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Score one (y_true, y_pred) pair per fold. Returns one row per fold year."""
    rows = []
    for start, end in folds:
        mask = (y_true.index >= start) & (y_true.index <= end)
        naive_mae = (
            seasonal_naive_mae_by_fold.get(start.year)
            if seasonal_naive_mae_by_fold is not None
            else None
        )
        row = evaluate(y_true[mask], y_pred[mask], seasonal_naive_mae=naive_mae)
        row["year"] = start.year
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def compare_models(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    folds: list[tuple[pd.Timestamp, pd.Timestamp]],
    seasonal_naive_name: str,
) -> pd.DataFrame:
    """Score multiple named prediction series against the same actuals, per fold.

    `seasonal_naive_name` must be a key in `predictions`; its per-fold MAE
    becomes that fold's MASE denominator for every model (including itself,
    which always scores MASE == 1.0), keeping results on one comparable scale
    across Weeks 2-7 per PLAN.md's guiding principle #2.
    """
    naive_mae_by_fold = {}
    for start, end in folds:
        mask = (y_true.index >= start) & (y_true.index <= end)
        naive_mae_by_fold[start.year] = evaluate(y_true[mask], predictions[seasonal_naive_name][mask])[
            "mae"
        ]

    tables = []
    for name, y_pred in predictions.items():
        table = evaluate_by_fold(y_true, y_pred, folds, seasonal_naive_mae_by_fold=naive_mae_by_fold)
        table["model"] = name
        tables.append(table.reset_index())
    return pd.concat(tables, ignore_index=True).set_index(["model", "year"]).sort_index()
