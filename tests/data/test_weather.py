import pandas as pd
import pytest

from edf.data.weather import (
    FORECAST_ARCHIVE_START,
    GB_CITIES,
    _population_weights,
    fetch_open_meteo_forecast_archive,
    population_weighted_gb_series,
)


def test_population_weights_sum_to_one_and_favour_london():
    weights = _population_weights()
    assert weights.sum() == pytest.approx(1.0)
    assert weights["London"] == weights.max()


def test_forecast_archive_rejects_dates_before_verified_cutover():
    with pytest.raises(ValueError, match="2022-03-01"):
        fetch_open_meteo_forecast_archive(51.5, -0.1, "2021-12-31", "2022-01-01")


def test_forecast_archive_start_is_a_date_not_moved_accidentally():
    # Regression guard: this constant was verified empirically against the
    # live API (see weather.py docstring) -- changing it silently would
    # reintroduce the ERA5-fallback-mislabelled-as-forecast bug.
    assert FORECAST_ARCHIVE_START == pd.Timestamp("2022-03-01", tz="UTC")


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
