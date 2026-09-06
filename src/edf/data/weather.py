"""GB weather data from three independent, free, keyless sources.

1. **Open-Meteo Historical Weather API** (ERA5/ERA5-Land reanalysis) —
   primary source, full 2020-2025 coverage.
2. **NASA POWER API** — independent reanalysis (CERES/GMAO), used to
   cross-check Open-Meteo's numbers and as an alternative solar-radiation
   source.
3. **Open-Meteo Historical Forecast API** — *archived forecast-model output*,
   not reanalysis. Empirically verified (2026-09-06, by diffing this endpoint
   against source #1) that it silently falls back to identical ERA5 values
   before 2022-03-01 — it is only ever requested from that date onward, so it
   never masquerades as forecast data when it secretly isn't. This is what
   lets Week 4 test realistic forecast-based accuracy, not just the
   observed-weather upper bound, for the latter part of the window.

Weather is aggregated across a small set of GB population centres into one
population-weighted GB series per source — a coarse proxy using ballpark
conurbation populations, not a full gridded population weighting.
Documented as a stated simplification (see PLAN.md).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import requests

FORECAST_ARCHIVE_START = pd.Timestamp("2022-03-01", tz="UTC")


class City(NamedTuple):
    name: str
    lat: float
    lon: float
    population_millions: float  # ballpark conurbation population, weighting proxy only


GB_CITIES: list[City] = [
    City("London", 51.5072, -0.1276, 9.0),
    City("Birmingham", 52.4862, -1.8904, 2.6),
    City("Manchester", 53.4808, -2.2426, 2.8),
    City("Glasgow", 55.8642, -4.2518, 1.8),
    City("Leeds", 53.8008, -1.5491, 1.9),
]


def _population_weights() -> pd.Series:
    weights = pd.Series({c.name: c.population_millions for c in GB_CITIES})
    return weights / weights.sum()


def fetch_open_meteo_historical(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """ERA5/ERA5-Land reanalysis — the "observed weather" primary source."""
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,wind_speed_10m,cloud_cover,shortwave_radiation",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(h["time"], utc=True),
            "temperature_c": h["temperature_2m"],
            "wind_speed_ms": h["wind_speed_10m"],
            "cloud_cover_pct": h["cloud_cover"],
            "shortwave_radiation_wm2": h["shortwave_radiation"],
        }
    )


def fetch_nasa_power(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Independent reanalysis (CERES/GMAO) — cross-check for source #1."""
    resp = requests.get(
        "https://power.larc.nasa.gov/api/temporal/hourly/point",
        params={
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "latitude": lat,
            "longitude": lon,
            "community": "RE",
            "parameters": "T2M,WS10M,ALLSKY_SFC_SW_DWN",
            "format": "JSON",
        },
        timeout=60,
    )
    resp.raise_for_status()
    params = resp.json()["properties"]["parameter"]
    index = pd.to_datetime(list(params["T2M"].keys()), format="%Y%m%d%H", utc=True)
    return pd.DataFrame(
        {
            "timestamp": index,
            "temperature_c": list(params["T2M"].values()),
            "wind_speed_ms": list(params["WS10M"].values()),
            # Wh/m^2 for a one-hour bucket is numerically ~= average W/m^2 that hour.
            "shortwave_radiation_wm2": list(params["ALLSKY_SFC_SW_DWN"].values()),
        }
    ).sort_values("timestamp").reset_index(drop=True)


def fetch_open_meteo_forecast_archive(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Archived forecast-model output. Only valid from FORECAST_ARCHIVE_START."""
    if pd.Timestamp(start, tz="UTC") < FORECAST_ARCHIVE_START:
        raise ValueError(
            f"Historical Forecast API has no genuine forecast data before "
            f"{FORECAST_ARCHIVE_START.date()} (verified empirically — it silently "
            f"falls back to ERA5 reanalysis before that date). Requested start={start}."
        )
    resp = requests.get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,wind_speed_10m,cloud_cover,shortwave_radiation",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(h["time"], utc=True),
            "temperature_c": h["temperature_2m"],
            "wind_speed_ms": h["wind_speed_10m"],
            "cloud_cover_pct": h["cloud_cover"],
            "shortwave_radiation_wm2": h["shortwave_radiation"],
        }
    )


_FETCHERS = {
    "open_meteo_historical": fetch_open_meteo_historical,
    "nasa_power": fetch_nasa_power,
    "open_meteo_forecast_archive": fetch_open_meteo_forecast_archive,
}


def population_weighted_gb_series(
    fetch: callable, start: str, end: str, cities: Sequence[City] = GB_CITIES
) -> pd.DataFrame:
    """Fetch one source for every reference city and population-weight them into one GB series."""
    weights = _population_weights()
    per_city = {}
    for city in cities:
        df = fetch(city.lat, city.lon, start, end).set_index("timestamp")
        per_city[city.name] = df

    value_cols = per_city[cities[0].name].columns
    combined = pd.DataFrame(index=per_city[cities[0].name].index)
    for col in value_cols:
        weighted = sum(per_city[c.name][col] * weights[c.name] for c in cities)
        combined[col] = weighted
    return combined.reset_index()


def build_source_parquet(source: str, start: str, end: str, out_dir: Path) -> Path:
    if source not in _FETCHERS:
        raise ValueError(f"Unknown source {source!r}, expected one of {list(_FETCHERS)}")
    df = population_weighted_gb_series(_FETCHERS[source], start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/weather"))
    parser.add_argument(
        "--sources", nargs="+", default=["open_meteo_historical", "nasa_power"],
        help="open_meteo_historical, nasa_power, open_meteo_forecast_archive "
        "(the latter is clipped to >= 2022-03-01 automatically if requested with an earlier start)",
    )
    args = parser.parse_args()

    for source in args.sources:
        start = args.start
        if source == "open_meteo_forecast_archive" and pd.Timestamp(start, tz="UTC") < FORECAST_ARCHIVE_START:
            start = str(FORECAST_ARCHIVE_START.date())
        out_path = build_source_parquet(source, start, args.end, args.out_dir)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
