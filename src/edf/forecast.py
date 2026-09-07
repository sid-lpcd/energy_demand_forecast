"""LightGBM demand models: a "direct" model per horizon, and a recursive rollout.

Two strategies for covering multiple forecast horizons (`edf.config.HORIZONS`)
from LightGBM models, per PLAN.md Week 3:

- **Direct**: train a separate model per horizon, each using only the
  lag/rolling features valid at that horizon (`edf.features.build_feature_table`).
  No error compounding, but one model per horizon.
- **Recursive**: train a single model at the shortest horizon (30 minutes,
  where every lag/rolling feature in `edf.features` is valid), then roll it
  forward step by step to reach longer horizons, feeding each step's own
  prediction back in as the next step's short lags. One model, but errors
  can compound over many steps, and features that were real at issue time
  become synthetic once the chain runs past their lag (see
  `recursive_forecast`'s docstring for exactly which features that hits and
  when).

Comparing the two (`notebooks/06_lightgbm_horizons.ipynb`) is what tells us
whether the compounding-error cost of "one model, applied recursively" is
worth paying versus training four separate models.
"""

from __future__ import annotations

from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from edf.baselines import PERIODS_PER_DAY, PERIODS_PER_WEEK
from edf.features import LAG_CANDIDATES, deterministic_features

DEFAULT_LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "random_state": 42,
}

_TRAILING_DAILY_LAGS = tuple(PERIODS_PER_DAY * k for k in range(1, 5))
_TRAILING_WEEKLY_LAGS = tuple(PERIODS_PER_WEEK * k for k in range(1, 5))


class Predictor(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None = None,
    **params: object,
) -> lgb.LGBMRegressor:
    """Fit a LightGBM regressor with this project's default hyperparameters.

    Hyperparameters are fixed, sane defaults, not tuned — walk-forward
    hyperparameter tuning is separate, later Week 3 work (PLAN.md), out of
    scope for the direct-vs-recursive comparison this trains models for.

    `sample_weight`, if given, up- or down-weights individual rows in the
    loss — e.g. Week 6's bias-correction experiment upweights extreme-bucket
    rows so a systematic under/over-forecast there costs the optimizer more,
    directly countering "rare regime diluted by average loss" rather than
    requiring a differently-shaped loss function.
    """
    model = lgb.LGBMRegressor(**{**DEFAULT_LGBM_PARAMS, **params})
    model.fit(X, y, sample_weight=sample_weight)
    return model


def recursive_forecast(
    model: Predictor,
    df: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    horizon_periods: int,
) -> pd.Series:
    """Roll a horizon=1 `model` forward `horizon_periods` steps per issue time.

    At step k (1..horizon_periods), every feature is rebuilt for the target
    time `issue_time + k` exactly as `edf.features.build_feature_table` would
    at horizon=1 — except each lag/rolling input that reaches past the issue
    time is taken from this simulation's own prior predictions instead of
    real `demand`, since a genuine forecaster wouldn't have that value yet.
    Concretely, for a lag/rolling component at absolute lag `L`: real data is
    used while `k <= L`, and this run's own step-`(k - L)` prediction once
    `k > L`. `lag_336` and `trailing_weekly_moving_average` (min lag 336)
    never flip to synthetic within this project's horizons (max 336), so
    they stay real for the whole rollout; `lag_1`/`lag_2` flip almost
    immediately, and `trailing_moving_average`'s first term flips once
    `k > 48`.

    Returns one horizon-periods-ahead forecast per issue time, indexed by
    the *target* timestamp (`issue_time + horizon_periods`).
    """
    demand = df["demand"]
    demand_arr = demand.to_numpy(dtype=float)
    det_features = deterministic_features(df)

    pos_lookup = pd.Series(np.arange(len(demand)), index=demand.index)
    t0_pos = pos_lookup.loc[issue_times].to_numpy()
    n_chains = len(t0_pos)

    max_lag = max((*LAG_CANDIDATES, *_TRAILING_DAILY_LAGS, *_TRAILING_WEEKLY_LAGS))
    if t0_pos.min() < max_lag - 1:
        # Without this, `demand_arr[t0_pos + rel]` with a negative index would
        # silently wrap around to the *end* of the array instead of raising.
        raise ValueError(
            f"issue_times must be at least {max_lag - 1} periods into `df` "
            "(need real history back to the longest rolling lag)"
        )

    # column k holds each chain's prediction for its k-th step ahead (k=0 unused)
    pred_buffer = np.full((n_chains, horizon_periods + 1), np.nan)

    def lookup(k: int, lag: int) -> np.ndarray:
        rel = k - lag
        if rel <= 0:
            return demand_arr[t0_pos + rel]
        return pred_buffer[:, rel]

    for k in range(1, horizon_periods + 1):
        target_pos = t0_pos + k
        X_step = det_features.iloc[target_pos].reset_index(drop=True)

        for lag in LAG_CANDIDATES:
            X_step[f"lag_{lag}"] = lookup(k, lag)

        X_step["trailing_moving_average"] = np.mean(
            [lookup(k, lag) for lag in _TRAILING_DAILY_LAGS], axis=0
        )
        X_step["trailing_weekly_moving_average"] = np.mean(
            [lookup(k, lag) for lag in _TRAILING_WEEKLY_LAGS], axis=0
        )

        pred_buffer[:, k] = model.predict(X_step)

    result_index = demand.index[t0_pos + horizon_periods]
    return pd.Series(pred_buffer[:, horizon_periods], index=result_index)
