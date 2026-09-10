"""Core live-prediction logic: run every persisted horizon model against live inputs.

Kept separate from `app.routes` so it can be unit-tested against a fake `LiveInputs`/registry
without an HTTP layer in the way.
"""

from __future__ import annotations

import pandas as pd

from app.live_pipeline import LiveInputs, build_live_features
from edf.models.combination import HEADLINE_COMBINATION_WEIGHT, combine_forecasts
from edf.models.conformal import apply_cqr_correction
from edf.models.quantile import enforce_monotonic_quantiles

QUANTILE_ALPHAS = (0.1, 0.5, 0.9)


def predict_all_horizons(registry: dict, live_inputs: LiveInputs) -> dict[str, dict]:
    """One prediction dict per horizon: `issue_time`, `target_time`, `point`, `p10`/`p50`/`p90`.

    `enforce_monotonic_quantiles` (Week 5, `edf.models.quantile`) fixes up the rare case where
    the independently-trained quantile models cross (P10 > P50 at a given row) -- reused as-is,
    not reimplemented here. `apply_cqr_correction` (`notebooks/23_calibration_robustness_comparison.ipynb`,
    `edf.models.conformal`) then widens P10/P90 by each horizon's own `cqr_q_hat`
    (`edf.models.registry`) -- the plain quantile model was found badly overconfident (PICP 62.6%
    for a nominal 80% interval), and this conformal correction recovers most of that gap. P50 is
    left untouched; CQR only corrects interval coverage, not the point/median forecast.
    """
    results: dict[str, dict] = {}
    for horizon_name, entry in registry.items():
        metadata = entry["metadata"]
        features = build_live_features(
            horizon_name, metadata["horizon_periods"], metadata, live_inputs
        )
        point = float(entry["point"].predict(features.point_row)[0])
        raw_quantiles = {
            alpha: pd.Series([float(model.predict(features.quantile_row)[0])])
            for alpha, model in entry["quantiles"].items()
        }
        sorted_quantiles = enforce_monotonic_quantiles(raw_quantiles)
        lower, upper = apply_cqr_correction(
            sorted_quantiles[0.1], sorted_quantiles[0.9], metadata["cqr_q_hat"]
        )

        results[horizon_name] = {
            "issue_time": features.issue_time,
            "target_time": features.target_time,
            "point": point,
            "p10": float(lower.iloc[0]),
            "p50": float(sorted_quantiles[0.5].iloc[0]),
            "p90": float(upper.iloc[0]),
        }
    return results


def ndf_comparison(live_inputs: LiveInputs, horizon_results: dict[str, dict]) -> dict | None:
    """The project's actual headline result (model blended with NDF beats NDF alone,
    `edf.models.combination.HEADLINE_COMBINATION_WEIGHT`) -- only meaningful at NDF's own
    cardinal-point lead time, so this is a secondary panel keyed to the `1d` tile, not a 5th
    horizon tile.

    Approximation, stated in the returned `note`: rather than re-running the model at each
    cardinal point's exact clock time, this pairs the `1d` tile's own (fixed, issue_time+24h)
    forecast with whichever published NDF cardinal point lands nearest to it in time.
    """
    if "1d" not in horizon_results or live_inputs.ndf.empty:
        return None

    target_time = horizon_results["1d"]["target_time"]
    ndf = live_inputs.ndf.copy()
    ndf["distance"] = (ndf["timestamp"] - target_time).abs()
    nearest = ndf.sort_values("distance").iloc[0]

    our_point = horizon_results["1d"]["point"]
    ndf_forecast = float(nearest["forecast_demand_mw"])
    combined = combine_forecasts(
        pd.Series([our_point]), pd.Series([ndf_forecast]), HEADLINE_COMBINATION_WEIGHT
    ).iloc[0]

    return {
        "cardinal_point_time": nearest["timestamp"],
        "cardinal_point_type": nearest["CP_TYPE"],
        "ndf_forecast_mw": ndf_forecast,
        "our_forecast_mw": our_point,
        "combination_weight": HEADLINE_COMBINATION_WEIGHT,
        "combined_forecast_mw": float(combined),
        "note": (
            "Compared against the nearest published NDF cardinal point to our 24h-ahead target "
            "time, not the exact same instant -- NDF publishes ~12 fixed times/day, not a "
            "continuous half-hourly series."
        ),
    }
