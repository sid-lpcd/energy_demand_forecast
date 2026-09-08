import pandas as pd
import pytest

from edf.data.weather import (
    _FETCHERS,
    DAY_AHEAD_ARCHIVE_START,
    GB_CITIES,
    NOWCAST_ARCHIVE_START,
    _population_weights,
    fetch_open_meteo_day_ahead,
    fetch_open_meteo_live_forecast,
    fetch_open_meteo_nowcast_archive,
    population_weighted_gb_series,
)


def test_population_weights_sum_to_one_and_favour_london():
    weights = _population_weights()
    assert weights.sum() == pytest.approx(1.0)
    assert weights["London"] == weights.max()


def test_nowcast_archive_rejects_dates_before_verified_cutover():
    with pytest.raises(ValueError, match="2024-03-06"):
        fetch_open_meteo_nowcast_archive(51.5, -0.1, "2024-03-05", "2024-03-06")


def test_day_ahead_rejects_dates_before_verified_cutover():
    with pytest.raises(ValueError, match="2024-03-07"):
        fetch_open_meteo_day_ahead(51.5, -0.1, "2024-03-06", "2024-03-07")


def test_archive_start_constants_not_moved_accidentally():
    # Regression guard: both constants were verified empirically against the
    # live APIs, pinned to PINNED_MODEL (see weather.py docstring) -- changing
    # them silently would reintroduce either accepting all-null nowcast data,
    # or silently accepting all-null "day ahead" data, as if either were real.
    assert NOWCAST_ARCHIVE_START == pd.Timestamp("2024-03-06", tz="UTC")
    assert DAY_AHEAD_ARCHIVE_START == pd.Timestamp("2024-03-07", tz="UTC")


def test_live_forecast_is_registered_and_has_the_shared_column_contract():
    assert _FETCHERS["open_meteo_live_forecast"] is fetch_open_meteo_live_forecast


def test_population_weighted_series_matches_manual_weighted_average():
    index = pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC")

    def fake_fetch(lat, lon, start, end):
        city = next(c for c in GB_CITIES if c.lat == lat)
        # Give each city a distinct constant temperature equal to its population,
        # so the weighted average has an easily-checked expected value.
        value = city.population_millions
        return pd.DataFrame({"timestamp": index, "temperature_c": [value] * len(index)})

    combined = population_weighted_gb_series(fake_fetch, "2022-01-01", "2022-01-01")

    weights = _population_weights()
    expected = sum(c.population_millions * weights[c.name] for c in GB_CITIES)
    assert combined["temperature_c"].round(6).eq(round(expected, 6)).all()
    assert list(combined["timestamp"]) == list(index)
