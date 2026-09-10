import lightgbm as lgb
import numpy as np
import pandas as pd

from edf.features.demand import build_feature_table
from edf.features.weather import build_weather_feature_table
from edf.models.baselines import PERIODS_PER_WEEK
from edf.models.forecast import train_lightgbm
from edf.models.registry import (
    DEFAULT_LGBM_PARAMS,
    QUANTILE_ALPHAS,
    _bias_correction_extras,
    _fit_quantile_models_with_cqr,
    _tune,
    load_registry,
    save_registry,
)


def _canonical_df(n: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "demand": 20000 + 5000 * np.sin(np.arange(n) / 48) + rng.normal(0, 10, n),
            "wind": rng.uniform(0, 3000, n),
            "wind_capacity": np.full(n, 6000.0),
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


def _weather_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    temperature = 10 + 5 * np.sin(np.arange(len(index)) / (48 * 365) * 2 * np.pi)
    return pd.DataFrame(
        {
            "temperature_c": temperature,
            "heating_degree": (15.5 - temperature).clip(min=0),
            "cooling_degree": (temperature - 22.0).clip(min=0),
        },
        index=index,
    )


def test_bias_correction_extras_columns_and_weight_range():
    n = PERIODS_PER_WEEK * 4 + 50
    df = _canonical_df(n)
    weather = _weather_df(df.index)

    extra, sample_weight = _bias_correction_extras(df, weather)

    assert set(extra.columns) == {"is_christmas", "heating_degree_1d_cum", "cooling_degree_1d_cum"}
    assert extra["is_christmas"].isin([0, 1]).all()
    assert sample_weight.index.equals(df.index)
    assert set(sample_weight.unique()) <= {1.0, 3.0}


def test_tune_falls_back_to_defaults_when_window_too_short_for_a_cv_fold():
    # Everything inside one calendar year -> walk_forward_cv_folds has nothing to fold on.
    n = 500
    df = _canonical_df(n)
    X, y = build_feature_table(df, horizon_periods=1)

    params, n_estimators = _tune(X, y)

    assert params == DEFAULT_LGBM_PARAMS
    assert n_estimators == DEFAULT_LGBM_PARAMS["n_estimators"]


def test_fit_quantile_models_with_cqr_returns_one_model_per_alpha_and_a_finite_q_hat():
    # Long enough to exceed the calibration slice with fit data left over, short enough
    # (single calendar year) that `_tune` falls back to defaults instead of running a real
    # walk-forward CV search -- keeps this test fast.
    n = PERIODS_PER_WEEK * 30
    df = _canonical_df(n)
    X, y = build_feature_table(df, horizon_periods=1)

    quantiles, q_hat, params = _fit_quantile_models_with_cqr(X, y, calibration_periods=500)

    assert set(quantiles) == set(QUANTILE_ALPHAS)
    assert np.isfinite(q_hat)
    assert params["n_estimators"] == DEFAULT_LGBM_PARAMS["n_estimators"]


def test_fit_quantile_models_with_cqr_never_trains_on_the_calibration_tail():
    n = PERIODS_PER_WEEK * 10
    df = _canonical_df(n)
    X, y = build_feature_table(df, horizon_periods=1)
    calibration_periods = 200

    quantiles, _, params = _fit_quantile_models_with_cqr(
        X, y, calibration_periods=calibration_periods
    )

    # A model trained on exactly X's fit-only slice (same deterministic params -- a single
    # calendar year, so `_tune` always falls back to DEFAULT_LGBM_PARAMS) should predict
    # identically to what `_fit_quantile_models_with_cqr` returned, proving the calibration
    # tail was excluded from training, not just unused afterwards.
    X_fit, y_fit = X.iloc[:-calibration_periods], y.iloc[:-calibration_periods]
    reference = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **params)
    reference.fit(X_fit, y_fit)

    np.testing.assert_allclose(quantiles[0.5].predict(X), reference.predict(X))


def test_save_and_load_registry_round_trips_predictions(tmp_path):
    n = PERIODS_PER_WEEK * 4 + 50
    df = _canonical_df(n)
    weather = _weather_df(df.index)
    X, y = build_feature_table(df, horizon_periods=1, weather=weather)
    model = train_lightgbm(X, y, n_estimators=5)
    quantile_model = train_lightgbm(X, y, n_estimators=5)

    registry = {
        "30min": {
            "point": model,
            "quantiles": {0.1: quantile_model, 0.5: quantile_model, 0.9: quantile_model},
            "metadata": {
                "horizon_periods": 1,
                "bias_corrected": False,
                "point_feature_columns": list(X.columns),
                "quantile_feature_columns": list(X.columns),
                "training_start": str(X.index.min()),
                "training_end": str(X.index.max()),
                "n_training_rows": len(X),
                "weather_source": "synthetic",
                "point_params": {},
                "quantile_params": {},
            },
        }
    }

    out_dir = tmp_path / "models_registry"
    save_registry(registry, out_dir=out_dir)
    loaded = load_registry(registry_dir=out_dir)

    assert set(loaded) == {"30min"}
    assert loaded["30min"]["metadata"] == registry["30min"]["metadata"]
    np.testing.assert_allclose(loaded["30min"]["point"].predict(X), model.predict(X))
    for alpha in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(
            loaded["30min"]["quantiles"][alpha].predict(X), quantile_model.predict(X)
        )


def test_build_weather_feature_table_smoke_for_registry_source(tmp_path, monkeypatch):
    # Confirms edf.features.weather.build_weather_feature_table (used directly by
    # edf.models.registry.train_final_models) still round-trips a parquet written
    # in the same shape edf.data.weather.build_source_parquet produces.
    index = pd.date_range("2024-03-07", periods=200, freq="h", tz="UTC")
    weather = pd.DataFrame({"timestamp": index, "temperature_c": np.linspace(0, 20, len(index))})
    raw_dir = tmp_path / "weather"
    raw_dir.mkdir()
    weather.to_parquet(raw_dir / "open_meteo_day_ahead.parquet", index=False)

    half_hourly_index = pd.date_range("2024-03-07", periods=200, freq="30min", tz="UTC")
    result = build_weather_feature_table("open_meteo_day_ahead", half_hourly_index, raw_dir=raw_dir)

    assert "heating_degree" in result.columns
    assert result.index.equals(half_hourly_index)
