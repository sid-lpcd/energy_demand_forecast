"""`/api/predict`: one call computes all `edf.config.HORIZONS` from live data."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request

from app.predict import ndf_comparison, predict_all_horizons
from app.schemas import HorizonPrediction, NdfComparison, PredictionResponse

router = APIRouter()


def _iso(ts: pd.Timestamp) -> str:
    return ts.isoformat()


@router.get("/api/predict", response_model=PredictionResponse)
def get_predictions(request: Request) -> PredictionResponse:
    registry = request.app.state.registry
    live_inputs = request.app.state.live_cache.get(pd.Timestamp.now(tz="UTC"))

    horizon_results = predict_all_horizons(registry, live_inputs)
    comparison = ndf_comparison(live_inputs, horizon_results)
    issue_time = next(iter(horizon_results.values()))["issue_time"]

    return PredictionResponse(
        issue_time=_iso(issue_time),
        weather_as_of=_iso(live_inputs.fetched_at),
        horizons={
            name: HorizonPrediction(
                target_time=_iso(r["target_time"]),
                point_mw=round(r["point"], 1),
                p10_mw=round(r["p10"], 1),
                p50_mw=round(r["p50"], 1),
                p90_mw=round(r["p90"], 1),
            )
            for name, r in horizon_results.items()
        },
        ndf_comparison=NdfComparison(
            cardinal_point_time=_iso(comparison["cardinal_point_time"]),
            cardinal_point_type=comparison["cardinal_point_type"],
            ndf_forecast_mw=round(comparison["ndf_forecast_mw"], 1),
            our_forecast_mw=round(comparison["our_forecast_mw"], 1),
            combination_weight=comparison["combination_weight"],
            combined_forecast_mw=round(comparison["combined_forecast_mw"], 1),
            note=comparison["note"],
        )
        if comparison is not None
        else None,
    )
