"""Train and persist the final per-horizon demand models, for the live-inference app.

Nothing is persisted anywhere else in this project — `edf.tracking.log_model_run` only logs
metrics/params from backtests. This module trains once (`train_final_models`), saves the fitted
models plus a per-horizon `metadata.json` (`save_registry`), and reloads them for serving
(`load_registry`).

**Weather source: `open_meteo_day_ahead`, not ERA5 hindsight.** The live app feeds these models a
real live weather *forecast* (`edf.data.weather.fetch_open_meteo_live_forecast`), so training must
use a genuinely forecast-like source too, or there's a train/serve skew on top of the usual
observed-vs-forecast gap PLAN.md already warns about. `open_meteo_day_ahead` is `edf.data.weather`'s
verified ~24h-ahead forecast, valid from `DAY_AHEAD_ARCHIVE_START` (2024-03-07) — `build_feature_table`
drops any row without weather, so every model here is trained only on 2024-03-07 through the end of
`VALIDATION` (2025-12-31), a much shorter window than `TRAIN` alone. This is the honest cost of a
genuinely deployable model, not an oversight.

**Trained on `TRAIN` + `VALIDATION` (2020-2025, weather-restricted as above), not `TRAIN` alone.**
These are final, deployed models, not a backtest being scored — `VALIDATION` isn't being evaluated
against here, so using it to fit the last model doesn't violate the "never peek" discipline.
`TEST` (2026) stays untouched (provisional/unverified per `edf.config`).

**Bias-corrected recipe (`notebooks/14`, `notebooks/20`) applied to `1d` only.** Adding an
`is_christmas` feature, a 1-day cumulative heating/cooling-degree feature, and 3x sample weight on
`is_hot`/`is_high_wind`/`is_christmas` rows was empirically verified to reduce point-model bias in
those buckets *at the `1d` horizon* — it was never tested at 30min/1h/7d, so applying it there would
be an unverified extrapolation this project's own conventions don't allow. Note also that this
recipe was originally verified against several years of ERA5-driven `1d` rows; re-fit here against
the much shorter day-ahead-weather window, its effect size may differ.

**Known limitation, `7d` horizon specifically**: `open_meteo_day_ahead` is a genuine forecast only
at its own ~24h lead time. Training the `7d` model against this source (indexed to each row's own
target timestamp) implicitly assumes ~24h-quality weather information 7 days out, which is more
accurate than any real 7-day-ahead forecast (including the live one the app will actually feed it
at serve time). This is a known, stated train/serve mismatch for `7d` specifically, not a bug —
fixing it would need a genuine multi-day-ahead forecast archive, which no free source publishes
historically (the same reasoning `edf.data.weather`'s docstring already applies to `wind`/`solar`).

**Quantile models (all horizons) intentionally skip the bias-correction recipe entirely** —
`notebooks/14` found it measurably *worsens* PICP (interval calibration) in exactly the buckets it
targets, even though it improves the point model and even pinball loss. Quantile models always use
the plain feature set.

**CQR-corrected intervals (all horizons), per `notebooks/23_calibration_robustness_comparison.ipynb`.**
That notebook found the plain LightGBM quantile model badly overconfident (PICP 62.6% for a nominal
80% [P10, P90] interval) and that Conformalized Quantile Regression (Romano et al. 2019,
`edf.models.conformal`) fixes most of the gap (PICP 75.1%) at a modest sharpness cost, for less
engineering effort than the notebook's other candidate (NGBoost, which lost on every axis there
because matching LightGBM's tuning budget wasn't computationally feasible). `_fit_quantile_models_with_cqr`
below applies that same recipe here: the trailing `CQR_CALIBRATION_PERIODS` rows of each horizon's
quantile-training data are held out of quantile-model fitting entirely (and out of hyperparameter
tuning) and used only to fit a single scalar correction (`cqr_q_hat`, stored in metadata), applied to
P10/P90 at serve time (`app.predict`). The point model is unaffected — CQR only touches the interval.
Only the `1d` horizon was directly validated by the notebook; the same mechanism is applied to every
horizon here on the reasonable premise that CQR's coverage guarantee doesn't depend on horizon,
though the achieved-PICP number for 30min/1h/7d hasn't itself been separately checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from edf.config import HORIZONS, TRAIN, VALIDATION
from edf.data.weather import DAY_AHEAD_ARCHIVE_START
from edf.features.buckets import (
    is_christmas_period,
    month_relative_percentile_bucket,
    overall_percentile_bucket,
)
from edf.features.demand import build_feature_table
from edf.features.generation import capacity_factor
from edf.features.weather import build_weather_feature_table, cumulative_degree
from edf.models.conformal import fit_cqr_correction
from edf.models.forecast import DEFAULT_LGBM_PARAMS, train_lightgbm
from edf.models.quantile import enforce_monotonic_quantiles
from edf.models.tuning import tune_lightgbm, walk_forward_cv_folds

DEFAULT_REGISTRY_DIR = Path("models_registry")
DEFAULT_CANONICAL_PATH = Path("data/processed/gb_energy_2020_2025.parquet")
WEATHER_SOURCE = "open_meteo_day_ahead"

QUANTILE_ALPHAS: tuple[float, ...] = (0.1, 0.5, 0.9)
BIAS_CORRECTED_HORIZONS: tuple[str, ...] = ("1d",)  # only horizon the recipe is verified for
CUMULATIVE_DEGREE_WINDOW_PERIODS = 48  # 1 day, notebooks/20's adopted window
EXTREME_SAMPLE_WEIGHT = 3.0  # notebooks/14's adopted weight for is_hot/is_high_wind/is_christmas
CQR_CALIBRATION_PERIODS = 48 * 90  # trailing 90 days, notebooks/23's conformal calibration set
CQR_COVERAGE = 0.8  # nominal [P10, P90] interval, matches notebooks/23

TUNING_N_TRIALS = 8  # kept small: this runs once, locally, per horizon -- not a hot path


def _load_canonical(path: Path = DEFAULT_CANONICAL_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    start, end = TRAIN[0], VALIDATION[1]
    return df.loc[start:end]


def _bias_correction_extras(df: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """`is_christmas` + 1-day cumulative degree-days, and the extreme-bucket sample weight.

    Per `notebooks/14`/`notebooks/20` (see module docstring) -- `1d` only.
    """
    extra = pd.DataFrame(index=df.index)
    extra["is_christmas"] = is_christmas_period(df.index).astype(int)
    extra["heating_degree_1d_cum"] = cumulative_degree(
        weather["heating_degree"], CUMULATIVE_DEGREE_WINDOW_PERIODS
    )
    extra["cooling_degree_1d_cum"] = cumulative_degree(
        weather["cooling_degree"], CUMULATIVE_DEGREE_WINDOW_PERIODS
    )

    wind_capacity_factor = capacity_factor(df["wind"], df["wind_capacity"])
    _, is_hot = month_relative_percentile_bucket(weather["temperature_c"])
    _, is_high_wind = overall_percentile_bucket(wind_capacity_factor)
    is_christmas = extra["is_christmas"].astype(bool)

    is_extreme = is_hot.fillna(False) | is_high_wind.fillna(False) | is_christmas
    sample_weight = pd.Series(1.0, index=df.index)
    sample_weight[is_extreme] = EXTREME_SAMPLE_WEIGHT
    return extra, sample_weight


def _fit_quantile_models_with_cqr(
    X: pd.DataFrame,
    y: pd.Series,
    calibration_periods: int = CQR_CALIBRATION_PERIODS,
    coverage: float = CQR_COVERAGE,
) -> tuple[dict[float, lgb.LGBMRegressor], float, dict[str, object]]:
    """Fit P10/P50/P90 quantile models on all but a trailing calibration slice, then fit a CQR
    correction on that slice.

    The trailing `calibration_periods` rows of `X`/`y` are excluded from both tuning and quantile
    model training entirely, exactly matching `notebooks/23_calibration_robustness_comparison.ipynb`'s
    split discipline -- letting the calibration set leak into training would invalidate CQR's
    coverage guarantee (Romano et al. 2019 requires it to be held out, not just unused for that one
    model). Returns `(quantile_models, cqr_q_hat, quantile_params_metadata)`.
    """
    n_calib = min(calibration_periods, len(X) - 1)
    X_fit, y_fit = X.iloc[:-n_calib], y.iloc[:-n_calib]
    X_calib, y_calib = X.iloc[-n_calib:], y.iloc[-n_calib:]

    quantile_params, quantile_n_estimators = _tune(X_fit, y_fit)

    quantiles, calib_preds = {}, {}
    for alpha in QUANTILE_ALPHAS:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            **{**quantile_params, "n_estimators": quantile_n_estimators},
        )
        model.fit(X_fit, y_fit)
        quantiles[alpha] = model
        calib_preds[alpha] = pd.Series(model.predict(X_calib), index=X_calib.index)

    calib_fixed = enforce_monotonic_quantiles(calib_preds)
    q_hat = fit_cqr_correction(y_calib, calib_fixed[0.1], calib_fixed[0.9], coverage=coverage)
    params_metadata = {**quantile_params, "n_estimators": quantile_n_estimators}
    return quantiles, q_hat, params_metadata


def _tune(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, object], int]:
    """Walk-forward-CV-tuned hyperparameters, falling back to the project defaults
    if the window is too short for even one CV fold (e.g. the day-ahead-weather
    window can span as little as two calendar years)."""
    cv_folds = walk_forward_cv_folds(X.index)
    if not cv_folds:
        return DEFAULT_LGBM_PARAMS, DEFAULT_LGBM_PARAMS["n_estimators"]
    best_params, best_n_estimators, _ = tune_lightgbm(X, y, cv_folds, n_trials=TUNING_N_TRIALS)
    return best_params, best_n_estimators


def train_final_models(
    canonical_path: Path = DEFAULT_CANONICAL_PATH,
) -> dict[str, dict[str, object]]:
    """Train the point + P10/P50/P90 quantile models for every `edf.config.HORIZONS` entry.

    Returns `{horizon_name: {"point": model, "quantiles": {alpha: model}, "metadata": {...}}}`.
    """
    df = _load_canonical(canonical_path)
    weather = build_weather_feature_table(WEATHER_SOURCE, df.index)
    if weather["temperature_c"].first_valid_index() is not None:
        assert weather["temperature_c"].first_valid_index() >= DAY_AHEAD_ARCHIVE_START, (
            "open_meteo_day_ahead data reaches earlier than its verified archive start -- "
            "re-check edf.data.weather.DAY_AHEAD_ARCHIVE_START before trusting this training run"
        )

    registry: dict[str, dict[str, object]] = {}
    for horizon_name, horizon_periods in HORIZONS.items():
        X_base, y = build_feature_table(df, horizon_periods, weather=weather)

        bias_corrected = horizon_name in BIAS_CORRECTED_HORIZONS
        if bias_corrected:
            extra, sample_weight = _bias_correction_extras(df, weather)
            X_point = X_base.join(extra.loc[X_base.index])
            sample_weight = sample_weight.loc[X_base.index]
        else:
            X_point, sample_weight = X_base, None

        point_params, point_n_estimators = _tune(X_point, y)
        point_model = train_lightgbm(
            X_point, y, sample_weight=sample_weight, **{**point_params, "n_estimators": point_n_estimators}
        )

        quantiles, cqr_q_hat, quantile_params = _fit_quantile_models_with_cqr(X_base, y)

        registry[horizon_name] = {
            "point": point_model,
            "quantiles": quantiles,
            "metadata": {
                "horizon_periods": horizon_periods,
                "bias_corrected": bias_corrected,
                "point_feature_columns": list(X_point.columns),
                "quantile_feature_columns": list(X_base.columns),
                "training_start": str(X_base.index.min()),
                "training_end": str(X_base.index.max()),
                "n_training_rows": len(X_base),
                "weather_source": WEATHER_SOURCE,
                "point_params": {**point_params, "n_estimators": point_n_estimators},
                "quantile_params": quantile_params,
                "cqr_q_hat": cqr_q_hat,
                "cqr_coverage": CQR_COVERAGE,
                "cqr_calibration_periods": min(CQR_CALIBRATION_PERIODS, len(X_base) - 1),
            },
        }
    return registry


def save_registry(registry: dict[str, dict[str, object]], out_dir: Path = DEFAULT_REGISTRY_DIR) -> None:
    for horizon_name, entry in registry.items():
        horizon_dir = out_dir / horizon_name
        horizon_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(entry["point"], horizon_dir / "point.joblib")
        for alpha, model in entry["quantiles"].items():
            joblib.dump(model, horizon_dir / f"quantile_{alpha}.joblib")
        (horizon_dir / "metadata.json").write_text(json.dumps(entry["metadata"], indent=2))


def load_registry(registry_dir: Path = DEFAULT_REGISTRY_DIR) -> dict[str, dict[str, object]]:
    """Load whatever horizons/quantiles are actually on disk under `registry_dir`.

    Deliberately discovers horizon subdirectories and `quantile_<alpha>.joblib`
    files from disk rather than trusting `edf.config.HORIZONS`/`QUANTILE_ALPHAS`
    to match what was last persisted there.
    """
    registry: dict[str, dict[str, object]] = {}
    for horizon_dir in sorted(p for p in registry_dir.iterdir() if p.is_dir()):
        metadata = json.loads((horizon_dir / "metadata.json").read_text())
        quantiles = {
            float(path.stem.removeprefix("quantile_")): joblib.load(path)
            for path in sorted(horizon_dir.glob("quantile_*.joblib"))
        }
        registry[horizon_dir.name] = {
            "point": joblib.load(horizon_dir / "point.joblib"),
            "quantiles": quantiles,
            "metadata": metadata,
        }
    return registry


def main() -> None:
    registry = train_final_models()
    save_registry(registry)
    for horizon_name, entry in registry.items():
        meta = entry["metadata"]
        print(
            f"{horizon_name}: {meta['n_training_rows']} rows "
            f"({meta['training_start']} .. {meta['training_end']}), "
            f"bias_corrected={meta['bias_corrected']}"
        )


if __name__ == "__main__":
    main()
