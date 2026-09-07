"""GB weather data from four independent, free, keyless sources.

1. **Open-Meteo Historical Weather API** (ERA5/ERA5-Land reanalysis) —
   primary source, full 2020-2025 coverage. Never available ahead of time in
   reality (reanalysis is finalized well after the fact by design).
2. **NASA POWER API** — independent reanalysis (CERES/GMAO), used to
   cross-check Open-Meteo's numbers and as an alternative solar-radiation
   source. Same "never ahead of time" caveat as #1.
3. **Open-Meteo Historical Forecast API** — a **short-lead nowcast archive**,
   NOT a day-ahead forecast. (Corrected 2026-09-06: earlier docs here wrongly
   called this "archived forecasts". Its own documentation says it "stitches
   together the first few hours of each model run" — a continuous nowcast,
   typically ~0-3h lead, not a fixed lead time. It's still more honest than
   ERA5 hindsight since a real model run couldn't see the future, but it does
   not represent a deployable day-ahead forecast.) Empirically verified
   (2026-09-06) that it silently falls back to identical ERA5 values before
   2022-03-01 under the default "Best Match" model selection.
4. **Open-Meteo Previous Runs API** — genuine **fixed lead-time** forecasts
   (`_previous_day1` = the forecast issued ~24h before the target hour, i.e.
   an actual day-ahead forecast). Empirically verified (2026-09-06) that each
   variable has its own archive-start date under "Best Match", all-null
   before it — temperature from 2024-02-04, `shortwave_radiation_previous_day1`
   only from 2024-03-07.

**Model pinning (added 2026-09-07):** sources #3 and #4 originally used
Open-Meteo's default "Best Match" model selection, which — its own docs say —
resolves to "the most suitable high-resolution model" *per endpoint*, not
guaranteed to be the same underlying model on both. `notebooks/02_explore_weather_data.ipynb`
found this mattered in practice: the day-ahead source's naive error against
ERA5 came out *lower* than the nowcast source's, backwards from what more
lead time should give — a symptom of comparing two different models, not
just two different lead times. Both fetchers now pin `models=ecmwf_ifs025`
explicitly so the only thing that differs between them is lead time. This
changed the empirical archive-start dates (re-verified 2026-09-07, same
null-scanning method as before, now against the pinned model): the nowcast
archive's binding variable (`shortwave_radiation`) only becomes available
from **2024-03-06** under this specific model — much later than the
"Best Match" default's 2022-03-01, since Open-Meteo's archived history for
one fixed model doesn't extend as far back as its always-pick-the-best-available
selection does. The day-ahead source's own start barely moved
(`shortwave_radiation_previous_day1` from 2024-03-07, same as before pinning)
since it was already effectively anchored to a similarly-recent model. Net
effect: the two sources now overlap validly only from 2024-03-07 onward —
unchanged for the day-ahead-based Week 4 ablation (which only ever needed
this slice), but the nowcast archive's usable window shrank by about two
years.

All four are free and require no API key.

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

PINNED_MODEL = "ecmwf_ifs025"  # explicit model for sources #3/#4 — see module docstring
NOWCAST_ARCHIVE_START = pd.Timestamp("2024-03-06", tz="UTC")  # binding var: shortwave_radiation
DAY_AHEAD_ARCHIVE_START = pd.Timestamp("2024-03-07", tz="UTC")  # latest of the 4 variables' cutovers


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


def fetch_open_meteo_nowcast_archive(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Short-lead (~0-3h) nowcast archive. NOT a day-ahead forecast — see module docstring.

    Only valid from NOWCAST_ARCHIVE_START (all-null before that, under the
    pinned model — see module docstring for why it's pinned at all).
    """
    if pd.Timestamp(start, tz="UTC") < NOWCAST_ARCHIVE_START:
        raise ValueError(
            f"Historical Forecast API has no data for model={PINNED_MODEL!r} before "
            f"{NOWCAST_ARCHIVE_START.date()} (verified empirically). Requested start={start}."
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
            "models": PINNED_MODEL,
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


def fetch_open_meteo_day_ahead(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Genuine ~24h-ahead forecast (Previous Runs API, `_previous_day1`).

    Only valid from DAY_AHEAD_ARCHIVE_START (all-null before that, under the
    pinned model — verified empirically).
    """
    if pd.Timestamp(start, tz="UTC") < DAY_AHEAD_ARCHIVE_START:
        raise ValueError(
            f"Previous Runs API day-1 data for model={PINNED_MODEL!r} is null before "
            f"{DAY_AHEAD_ARCHIVE_START.date()} (verified empirically). Requested start={start}."
        )
    variables = [
        "temperature_2m_previous_day1",
        "wind_speed_10m_previous_day1",
        "cloud_cover_previous_day1",
        "shortwave_radiation_previous_day1",
    ]
    resp = requests.get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(variables),
            "wind_speed_unit": "ms",
            "models": PINNED_MODEL,
            "timezone": "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(h["time"], utc=True),
            "temperature_c": h["temperature_2m_previous_day1"],
            "wind_speed_ms": h["wind_speed_10m_previous_day1"],
            "cloud_cover_pct": h["cloud_cover_previous_day1"],
            "shortwave_radiation_wm2": h["shortwave_radiation_previous_day1"],
        }
    )


_FETCHERS = {
    "open_meteo_historical": fetch_open_meteo_historical,
    "nasa_power": fetch_nasa_power,
    "open_meteo_nowcast_archive": fetch_open_meteo_nowcast_archive,
    "open_meteo_day_ahead": fetch_open_meteo_day_ahead,
}

_MIN_START = {
    "open_meteo_nowcast_archive": NOWCAST_ARCHIVE_START,
    "open_meteo_day_ahead": DAY_AHEAD_ARCHIVE_START,
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
        help=f"one or more of {list(_FETCHERS)} "
        "(sources with a verified minimum start date are clipped to it automatically)",
    )
    args = parser.parse_args()

    for source in args.sources:
        start = args.start
        min_start = _MIN_START.get(source)
        if min_start is not None and pd.Timestamp(start, tz="UTC") < min_start:
            start = str(min_start.date())
        out_path = build_source_parquet(source, start, args.end, args.out_dir)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
