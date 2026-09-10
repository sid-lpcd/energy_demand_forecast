"""NGBoost: a distributional alternative to per-quantile LightGBM — Stretch goal, PLAN.md.

Week 5 found LightGBM's independent per-quantile models (`edf.models.quantile`)
systematically compress both tails (PICP 65.2% for a nominal 80% interval) — a
structural property of training each alpha level as its own unrelated model,
not a tuning artifact. NGBoost (Duan et al. 2020) fits one full probability
distribution per row via natural-gradient boosting, so every quantile comes
from a single coherent CDF instead of N independently-optimized pinball-loss
models: quantile crossing is impossible by construction, and the tails are
whatever the fitted distribution's own tails are, not a separately-trained
alpha level. This module doesn't assume NGBoost is *better* — only that it
fails differently (a misspecified distribution family, if GB demand isn't
really Normal, vs. LightGBM's crossing/tail-compression) — worth comparing
empirically, which is what the stretch-goal notebook does.
"""

from __future__ import annotations

import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal

DEFAULT_NGB_PARAMS: dict[str, object] = {
    "n_estimators": 300,
    "learning_rate": 0.02,
    "random_state": 42,
}


def encode_categoricals(
    X: pd.DataFrame, reference_columns: pd.Index | None = None
) -> pd.DataFrame:
    """One-hot encode pandas `category` columns for NGBoost's sklearn-tree base learner.

    LightGBM elsewhere in this project consumes `category` dtype natively
    (`edf.features.demand.deterministic_features`); NGBoost's default base
    learner is a plain `DecisionTreeRegressor`, which can't, so the expansion
    happens here rather than in the shared feature-table builder. Pass
    `reference_columns` (the fitted TRAIN encoding's columns) when encoding a
    later split, so a category level absent from that split doesn't shift
    column order or count relative to what the model was trained on.
    """
    categorical_columns = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    encoded = pd.get_dummies(X, columns=categorical_columns)
    if reference_columns is not None:
        encoded = encoded.reindex(columns=reference_columns, fill_value=0)
    return encoded


def train_ngboost(X: pd.DataFrame, y: pd.Series, **params: object) -> NGBRegressor:
    """Fit an NGBoost regressor with a Normal output distribution.

    `X` must already be fully numeric (see `encode_categoricals`) — unlike
    LightGBM elsewhere in this project, NGBoost's default base learner can't
    take pandas `category` columns natively.
    """
    model = NGBRegressor(Dist=Normal, verbose=False, **{**DEFAULT_NGB_PARAMS, **params})
    model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
    return model


def ngboost_quantiles(
    model: NGBRegressor, X: pd.DataFrame, alphas: list[float]
) -> dict[float, pd.Series]:
    """Per-alpha quantiles read off NGBoost's fitted per-row Normal distribution.

    Derived from one `ppf` call per alpha against a single fitted
    distribution per row, not from N independently-trained models —
    non-crossing is guaranteed by construction (a monotonic CDF), unlike
    `edf.models.quantile`'s rearrangement fix for LightGBM's version of the
    same problem.
    """
    dist = model.pred_dist(X.to_numpy(dtype=float))
    return {alpha: pd.Series(dist.ppf(alpha), index=X.index) for alpha in alphas}
