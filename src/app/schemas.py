"""Pydantic response models for `/api/predict`."""

from __future__ import annotations

from pydantic import BaseModel


class HorizonPrediction(BaseModel):
    target_time: str
    point_mw: float
    p10_mw: float
    p50_mw: float
    p90_mw: float


class NdfComparison(BaseModel):
    cardinal_point_time: str
    cardinal_point_type: str
    ndf_forecast_mw: float
    our_forecast_mw: float
    combination_weight: float
    combined_forecast_mw: float
    note: str


class PredictionResponse(BaseModel):
    issue_time: str
    weather_as_of: str
    horizons: dict[str, HorizonPrediction]
    ndf_comparison: NdfComparison | None = None
