"""Walk-forward hyperparameter tuning for the Week 3 LightGBM models.

Tuning happens entirely inside `edf.config.TRAIN` (2020-2023) via expanding-
window walk-forward CV (PLAN.md's "tune via walk-forward CV, not k-fold"
rule) — `VALIDATION` is never touched during search, including for deciding
how many trees to grow: early stopping happens against each CV fold's own
held-out year, and the final model's `n_estimators` is fixed from that,
*before* the final model ever sees `VALIDATION`. Early-stopping directly
against `VALIDATION` would let the exact set used for final reporting
quietly influence model selection — a subtle leak, not an outright one, but
one that would inflate the reported comparison.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from edf.evaluate import mae, yearly_folds

PARAM_DISTRIBUTIONS: dict[str, list[object]] = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_samples": [20, 50, 100],
}

DEFAULT_MAX_ESTIMATORS = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 30


def walk_forward_cv_folds(
    index: pd.DatetimeIndex,
) -> list[tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]]:
    """Expanding-window `(train_range, val_range)` pairs, one per year after the first.

    E.g. for years 2020-2023: train on 2020 -> validate 2021, train on
    2020-2021 -> validate 2022, train on 2020-2022 -> validate 2023. Reuses
    `edf.evaluate.yearly_folds` for the year boundaries themselves.
    """
    year_bounds = yearly_folds(index)
    return [
        ((year_bounds[0][0], year_bounds[i - 1][1]), year_bounds[i])
        for i in range(1, len(year_bounds))
    ]


def _fit_one_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_range: tuple[pd.Timestamp, pd.Timestamp],
    val_range: tuple[pd.Timestamp, pd.Timestamp],
    params: dict[str, object],
    max_estimators: int,
    early_stopping_rounds: int,
) -> tuple[float, int]:
    X_train = X.loc[train_range[0] : train_range[1]]
    y_train = y.loc[train_range[0] : train_range[1]]
    X_val = X.loc[val_range[0] : val_range[1]]
    y_val = y.loc[val_range[0] : val_range[1]]

    model = lgb.LGBMRegressor(n_estimators=max_estimators, random_state=42, **params)
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    preds = pd.Series(model.predict(X_val), index=X_val.index)
    return mae(y_val, preds), model.best_iteration_


def tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]],
    param_distributions: dict[str, list[object]] = PARAM_DISTRIBUTIONS,
    n_trials: int = 10,
    random_state: int = 42,
    max_estimators: int = DEFAULT_MAX_ESTIMATORS,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> tuple[dict[str, object], int, pd.DataFrame]:
    """Randomized walk-forward CV search over `param_distributions`.

    Returns `(best_params, best_n_estimators, trials)`: `best_n_estimators`
    is the mean early-stopped iteration count across folds for the winning
    params, rounded — the tree count the final model (trained on the full
    `TRAIN` set, no early stopping) should use. `trials` has one row per
    candidate, sorted best-first, for inspection/logging.
    """
    candidates = list(
        ParameterSampler(param_distributions, n_iter=n_trials, random_state=random_state)
    )

    trial_rows = []
    for i, params in enumerate(candidates):
        t0 = time.time()
        fold_maes, fold_iters = [], []
        for train_range, val_range in folds:
            fold_mae, best_iter = _fit_one_fold(
                X, y, train_range, val_range, params, max_estimators, early_stopping_rounds
            )
            fold_maes.append(fold_mae)
            fold_iters.append(best_iter)
        trial_rows.append(
            {
                "trial_index": i,
                **params,
                "cv_mae": float(np.mean(fold_maes)),
                "cv_n_estimators": round(np.mean(fold_iters)),
                "seconds": time.time() - t0,
            }
        )

    trials = pd.DataFrame(trial_rows).sort_values("cv_mae").reset_index(drop=True)
    best_row = trials.iloc[0]
    best_params = candidates[int(best_row["trial_index"])]
    return best_params, int(best_row["cv_n_estimators"]), trials
