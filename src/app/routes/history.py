"""`/api/history`: the precomputed daily actual/NDF/our-forecast/blended comparison series.

Precomputed offline (`edf.models.history_comparison`, see its docstring) -- this route only reads
whatever's already loaded into `app.state.history_comparison` at startup, no live computation.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request

from app.schemas import HistoryComparisonResponse, HistoryDay
from edf.models.history_comparison import SCOTTISH_TRANSFER_ADVISORY, WEATHER_UPPER_BOUND_NOTE

router = APIRouter()


def _or_none(value: float) -> float | None:
    return None if pd.isna(value) else round(value, 1)


@router.get("/api/history", response_model=HistoryComparisonResponse)
def get_history(request: Request) -> HistoryComparisonResponse:
    daily: pd.DataFrame | None = request.app.state.history_comparison
    if daily is None:
        return HistoryComparisonResponse(
            days=[],
            weather_upper_bound_note=WEATHER_UPPER_BOUND_NOTE,
            scottish_transfer_advisory=SCOTTISH_TRANSFER_ADVISORY,
        )

    return HistoryComparisonResponse(
        days=[
            HistoryDay(
                date=str(row.date),
                actual_demand_mw=_or_none(row.actual_demand_mw),
                ndf_forecast_mw=round(row.ndf_forecast_mw, 1),
                our_forecast_mw=round(row.our_forecast_mw, 1),
                blended_forecast_mw=round(row.blended_forecast_mw, 1),
            )
            for row in daily.itertuples()
        ],
        weather_upper_bound_note=WEATHER_UPPER_BOUND_NOTE,
        scottish_transfer_advisory=SCOTTISH_TRANSFER_ADVISORY,
    )
