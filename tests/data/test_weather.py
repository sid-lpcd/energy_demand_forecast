import pandas as pd
import pytest
import requests

import edf.data.weather as weather_module
from edf.data.weather import (
    _FETCHERS,
    DAY_AHEAD_ARCHIVE_START,
    GB_CITIES,
    NOWCAST_ARCHIVE_START,
    PINNED_MODEL,
    _population_weights,
    fetch_open_meteo_day_ahead,
    fetch_open_meteo_live_forecast,
    fetch_open_meteo_live_forecast_multi,
    fetch_open_meteo_nowcast_archive,
    population_weighted_gb_series,
    population_weighted_live_forecast,
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


class _FakeResponse:
    def __init__(self, status_code: int, json_body=None):
        self.status_code = status_code
        self._json_body = json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_body


def _live_forecast_payload(temperature_by_city: dict[str, float]) -> list[dict]:
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    return [
        {
            "hourly": {
                "time": [t.isoformat() for t in index],
                "temperature_2m": [temperature_by_city[city.name]] * len(index),
                "apparent_temperature": [temperature_by_city[city.name]] * len(index),
                "wind_speed_10m": [1.0] * len(index),
                "cloud_cover": [0.0] * len(index),
                "shortwave_radiation": [0.0] * len(index),
            }
        }
        for city in GB_CITIES
    ]


def test_live_forecast_multi_issues_one_request_for_all_cities(monkeypatch):
    calls = []
    temps = {city.name: city.population_millions for city in GB_CITIES}

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return _FakeResponse(200, _live_forecast_payload(temps))

    monkeypatch.setattr(weather_module.requests, "get", fake_get)

    per_city = fetch_open_meteo_live_forecast_multi(GB_CITIES, "2026-01-01", "2026-01-02")

    assert len(calls) == 1
    params = calls[0][1]
    assert params["latitude"] == ",".join(str(c.lat) for c in GB_CITIES)
    assert params["longitude"] == ",".join(str(c.lon) for c in GB_CITIES)
    assert params["models"] == PINNED_MODEL
    assert set(per_city) == {c.name for c in GB_CITIES}
    for city in GB_CITIES:
        assert per_city[city.name]["temperature_c"].eq(temps[city.name]).all()


def test_live_forecast_multi_retries_on_429_then_succeeds(monkeypatch):
    responses = [_FakeResponse(429), _FakeResponse(429), _FakeResponse(200, _live_forecast_payload(
        {city.name: 1.0 for city in GB_CITIES}
    ))]

    def fake_get(url, params, timeout):
        return responses.pop(0)

    monkeypatch.setattr(weather_module.requests, "get", fake_get)
    monkeypatch.setattr(weather_module.time, "sleep", lambda seconds: None)

    per_city = fetch_open_meteo_live_forecast_multi(GB_CITIES, "2026-01-01", "2026-01-02")

    assert not responses
    assert set(per_city) == {c.name for c in GB_CITIES}


def test_live_forecast_multi_raises_after_exhausting_retries(monkeypatch):
    def fake_get(url, params, timeout):
        return _FakeResponse(429)

    monkeypatch.setattr(weather_module.requests, "get", fake_get)
    monkeypatch.setattr(weather_module.time, "sleep", lambda seconds: None)

    with pytest.raises(requests.HTTPError):
        fetch_open_meteo_live_forecast_multi(GB_CITIES, "2026-01-01", "2026-01-02")


def test_live_forecast_multi_uses_proxy_url_when_env_var_set(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return _FakeResponse(200, _live_forecast_payload({city.name: 1.0 for city in GB_CITIES}))

    monkeypatch.setattr(weather_module.requests, "get", fake_get)
    monkeypatch.setenv("OPEN_METEO_LIVE_FORECAST_URL", "https://weather-proxy.example.vercel.app/api/forecast")

    fetch_open_meteo_live_forecast_multi(GB_CITIES, "2026-01-01", "2026-01-02")

    assert calls == ["https://weather-proxy.example.vercel.app/api/forecast"]


def test_live_forecast_multi_defaults_to_open_meteo_directly(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return _FakeResponse(200, _live_forecast_payload({city.name: 1.0 for city in GB_CITIES}))

    monkeypatch.setattr(weather_module.requests, "get", fake_get)
    monkeypatch.delenv("OPEN_METEO_LIVE_FORECAST_URL", raising=False)

    fetch_open_meteo_live_forecast_multi(GB_CITIES, "2026-01-01", "2026-01-02")

    assert calls == ["https://api.open-meteo.com/v1/forecast"]


def test_population_weighted_live_forecast_matches_manual_weighted_average(monkeypatch):
    temps = {city.name: city.population_millions for city in GB_CITIES}

    def fake_get(url, params, timeout):
        return _FakeResponse(200, _live_forecast_payload(temps))

    monkeypatch.setattr(weather_module.requests, "get", fake_get)

    combined = population_weighted_live_forecast("2026-01-01", "2026-01-02")

    weights = _population_weights()
    expected = sum(c.population_millions * weights[c.name] for c in GB_CITIES)
    assert combined["temperature_c"].round(6).eq(round(expected, 6)).all()


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
