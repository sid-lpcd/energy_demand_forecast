"""Live NESO day-ahead national demand forecast (NDF), current resource.

Same cardinal-point schema as `edf.data.demand_forecast_benchmark`'s frozen historic archive --
this is NESO's current, continuously-updated "Day Ahead National Demand Forecast" resource instead
(`DAYSAHEAD`/`TARGETDATE`/`FORECASTDEMAND`/`CARDINALPOINT`/`CP_TYPE`/`CP_ST_TIME`/`CP_END_TIME`,
verified live 2026-09-08). The resource ID is stable; the filename it redirects to is date-stamped
and changes daily (confirmed: the CKAN download endpoint ignores whatever filename is appended and
always redirects to the current file), so requests must follow redirects.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from edf.data.demand_forecast_benchmark import cardinal_point_to_utc

LIVE_NDF_URL = (
    "https://api.neso.energy/dataset/8fbc8a09-06af-4c90-886f-d3025d38a349"
    "/resource/aec5601a-7f3e-4c4c-bf56-d8e4184d3c5b/download/current.csv"
)


def parse_live_ndf_csv(csv_text: str) -> pd.DataFrame:
    """`csv_text` -> `timestamp`/`forecast_demand_mw` frame, day-ahead (`DAYSAHEAD == 1`) only.

    Verified live (2026-09-08): unlike the historic archive's dashed `TARGETDATE` strings
    (`"2020-01-01"`), the live resource's `TARGETDATE` is a bare `YYYYMMDD` integer
    (`20260909`). `cardinal_point_to_utc`'s untyped `pd.to_datetime` call silently treats a raw
    int64 as nanoseconds-since-epoch instead of a date, producing 1969/1970 garbage timestamps --
    caught by a live smoke test, not by any existing unit test (those all pass already-dashed
    strings). Explicitly parsed with `format="%Y%m%d"` here, before it ever reaches
    `cardinal_point_to_utc`.
    """
    df = pd.read_csv(StringIO(csv_text), dtype={"CP_ST_TIME": str})
    day_ahead = df[df["DAYSAHEAD"] == 1].copy()
    target_date = pd.to_datetime(day_ahead["TARGETDATE"].astype(str), format="%Y%m%d")
    day_ahead["timestamp"] = cardinal_point_to_utc(target_date, day_ahead["CP_ST_TIME"])
    day_ahead = day_ahead.dropna(subset=["timestamp"])
    return (
        day_ahead[["timestamp", "TARGETDATE", "CARDINALPOINT", "CP_TYPE", "FORECASTDEMAND"]]
        .rename(columns={"FORECASTDEMAND": "forecast_demand_mw"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_live_ndf(url: str = LIVE_NDF_URL, timeout: int = 30) -> pd.DataFrame:
    """Currently-published day-ahead NDF cardinal points (today's and tomorrow's target dates)."""
    response = requests.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return parse_live_ndf_csv(response.text)
