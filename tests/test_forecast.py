import numpy as np
import pandas as pd
import pytest

from edf.baselines import PERIODS_PER_WEEK
from edf.features import build_feature_table
from edf.forecast import recursive_forecast, train_lightgbm


def _canonical_df(n: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "demand": np.arange(n, dtype=float),
            "is_bank_holiday": [False] * n,
            "is_school_holiday": [False] * n,
            "lockdown_level": pd.Categorical(["none"] * n, categories=["none", "partial", "full"]),
            "event_tier": pd.Categorical(
                ["none"] * n, categories=["none", "small", "medium", "large"]
            ),
            "solar_eclipse_pct": [0.0] * n,
        },
        index=index,
    )


class _PersistencePlusOnePredictor:
    """predict(X) = X['lag_1'] + 1 — makes the recursive chain hand-verifiable:
    k steps ahead from t0 should equal demand[t0] + k."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["lag_1"] + 1).to_numpy()


class _Lag336Predictor:
    """predict(X) = X['lag_336'] — probes whether lag_336 is ever taken from
    the (still-empty-at-that-lag) prediction buffer instead of real data."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_336"].to_numpy()


def test_train_lightgbm_smoke():
    n = PERIODS_PER_WEEK * 4 + 50  # long enough for trailing_weekly_moving_average's warm-up
    df = _canonical_df(n)
    X, y = build_feature_table(df, horizon_periods=1)

    model = train_lightgbm(X, y, n_estimators=5)
    preds = model.predict(X)

    assert preds.shape == (len(X),)


def test_recursive_forecast_persistence_chain_matches_hand_computed_values():
    n = 2000
    df = _canonical_df(n)
    issue_times = df.index[1500:1503]

    result = recursive_forecast(_PersistencePlusOnePredictor(), df, issue_times, horizon_periods=5)

    expected_index = issue_times + pd.Timedelta(minutes=30 * 5)
    expected_values = df.loc[issue_times, "demand"].to_numpy() + 5
    assert list(result.index) == list(expected_index)
    np.testing.assert_allclose(result.to_numpy(), expected_values)


def test_recursive_forecast_lag_336_stays_real_never_synthetic():
    n = 2000
    df = _canonical_df(n)
    issue_times = df.index[1500:1503]
    horizon_periods = 10

    result = recursive_forecast(_Lag336Predictor(), df, issue_times, horizon_periods=horizon_periods)

    # At the final step, lag_336 for target (t0 + horizon) is demand at
    # (t0 + horizon - 336) — always real here since horizon (10) << 336.
    pos = df.index.get_indexer(issue_times)
    expected = df["demand"].to_numpy()[pos + horizon_periods - PERIODS_PER_WEEK]
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_recursive_forecast_at_horizon_one_matches_direct_model_features():
    n = 2000
    df = _canonical_df(n)
    X, y = build_feature_table(df, horizon_periods=1)
    model = train_lightgbm(X, y, n_estimators=5)

    issue_times = df.index[1500:1510]
    target_times = issue_times + pd.Timedelta(minutes=30)

    recursive_preds = recursive_forecast(model, df, issue_times, horizon_periods=1)
    direct_preds = model.predict(X.loc[target_times])

    np.testing.assert_allclose(recursive_preds.to_numpy(), direct_preds, rtol=1e-6)


def test_recursive_forecast_raises_if_issue_times_too_close_to_start():
    n = 100
    df = _canonical_df(n)
    issue_times = df.index[:5]

    with pytest.raises(ValueError):
        recursive_forecast(_PersistencePlusOnePredictor(), df, issue_times, horizon_periods=2)
