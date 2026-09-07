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
