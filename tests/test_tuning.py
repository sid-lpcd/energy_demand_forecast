import numpy as np
import pandas as pd

from edf.tuning import tune_lightgbm, walk_forward_cv_folds


def test_walk_forward_cv_folds_are_expanding_windows():
    index = pd.date_range("2020-01-01", "2023-06-01", freq="D", tz="UTC")

    folds = walk_forward_cv_folds(index)

    assert len(folds) == 3  # validate on 2021, 2022, 2023 in turn
    (train0, val0), (train1, val1), (train2, val2) = folds
    assert train0[0].year == 2020 and train0[1].year == 2020
    assert val0[0].year == 2021
    assert train1[0].year == 2020 and train1[1].year == 2021  # expands to include 2021
    assert val1[0].year == 2022
    assert train2[1].year == 2022
    assert val2[0].year == 2023


def _synthetic_train_data(n_years: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=365 * n_years, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    day_of_year = index.dayofyear.to_numpy()
    # a signal with clear seasonal structure, so tuning has something real to find
    y = 100 + 10 * np.sin(2 * np.pi * day_of_year / 365) + rng.normal(0, 1, len(index))
    X = pd.DataFrame(
        {
            "day_of_year": day_of_year,
            "day_of_week": index.dayofweek.to_numpy(),
            "noise": rng.normal(0, 1, len(index)),
        },
        index=index,
    )
    return X, pd.Series(y, index=index)


def test_tune_lightgbm_returns_params_from_the_given_distribution():
    X, y = _synthetic_train_data()
    folds = walk_forward_cv_folds(X.index)
    distributions = {"num_leaves": [7, 15], "learning_rate": [0.1], "min_child_samples": [5]}

    best_params, best_n_estimators, trials = tune_lightgbm(
        X,
        y,
        folds,
        param_distributions=distributions,
        n_trials=2,
        max_estimators=50,
        early_stopping_rounds=5,
    )

    assert best_params["num_leaves"] in distributions["num_leaves"]
    assert best_params["learning_rate"] in distributions["learning_rate"]
    assert best_n_estimators > 0
    assert len(trials) == 2
    assert trials["cv_mae"].is_monotonic_increasing  # sorted best (lowest MAE) first


def test_tune_lightgbm_picks_the_lower_cv_mae_trial():
    X, y = _synthetic_train_data()
    folds = walk_forward_cv_folds(X.index)
    # one sane config, one deliberately poor one (single leaf can't fit anything)
    distributions = {"num_leaves": [2, 31], "learning_rate": [0.1], "min_child_samples": [5]}

    best_params, _, trials = tune_lightgbm(
        X,
        y,
        folds,
        param_distributions=distributions,
        n_trials=2,
        max_estimators=50,
        early_stopping_rounds=5,
    )

    assert best_params["num_leaves"] == 31
    assert trials.iloc[0]["cv_mae"] < trials.iloc[1]["cv_mae"]
