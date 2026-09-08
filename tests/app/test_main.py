import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.live_pipeline as live_pipeline_module
from app.main import app


def _fake_live_inputs(now_utc: pd.Timestamp) -> live_pipeline_module.LiveInputs:
    """Synthetic stand-in for `fetch_live_inputs` -- real calendar/feature-building logic runs
    unmodified against these values, only the network-fetched numbers are fake, so this exercises
    the real live/train feature-parity path (`app.live_pipeline.build_live_features`) against the
    actual committed model registry, without hitting NESO/Open-Meteo in the test suite.
    """
    history_index = pd.date_range(now_utc - pd.Timedelta(days=35), now_utc, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    demand_history = pd.DataFrame(
        {
            "timestamp": history_index,
            "demand": 20000
            + 5000 * np.sin(np.arange(len(history_index)) / 48)
            + rng.normal(0, 5, len(history_index)),
        }
    )

    weather_index = pd.date_range(
        now_utc - pd.Timedelta(days=1), now_utc + pd.Timedelta(days=9), freq="h", tz="UTC"
    )
    weather_forecast = pd.DataFrame(
        {
            "timestamp": weather_index,
            "temperature_c": 10 + 5 * np.sin(np.arange(len(weather_index)) / 24),
            "apparent_temperature_c": 9 + 5 * np.sin(np.arange(len(weather_index)) / 24),
            "wind_speed_ms": np.full(len(weather_index), 5.0),
            "cloud_cover_pct": np.full(len(weather_index), 50.0),
            "shortwave_radiation_wm2": np.full(len(weather_index), 100.0),
        }
    )

    ndf_index = pd.date_range(now_utc, now_utc + pd.Timedelta(days=2), freq="6h", tz="UTC")
    ndf = pd.DataFrame(
        {
            "timestamp": ndf_index,
            "TARGETDATE": ndf_index.strftime("%Y%m%d"),
            "CARDINALPOINT": [f"{i}A" for i in range(len(ndf_index))],
            "CP_TYPE": ["P"] * len(ndf_index),
            "forecast_demand_mw": np.full(len(ndf_index), 25000.0),
        }
    )

    return live_pipeline_module.LiveInputs(
        demand_history=demand_history, weather_forecast=weather_forecast, ndf=ndf, fetched_at=now_utc
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(live_pipeline_module, "fetch_live_inputs", _fake_live_inputs)
    with TestClient(app) as test_client:
        yield test_client


def test_index_page_ok(client):
    assert client.get("/").status_code == 200


def test_predictions_page_ok(client):
    assert client.get("/predictions").status_code == 200


def test_report_page_ok(client):
    assert client.get("/report").status_code == 200


def test_predict_returns_all_horizons_with_monotonic_quantiles(client):
    response = client.get("/api/predict")

    assert response.status_code == 200
    data = response.json()
    assert set(data["horizons"]) == {"30min", "1h", "1d", "7d"}
    for horizon in data["horizons"].values():
        assert horizon["p10_mw"] <= horizon["p50_mw"] <= horizon["p90_mw"]
        assert horizon["point_mw"] > 0


def test_predict_includes_ndf_comparison_panel(client):
    response = client.get("/api/predict")

    comparison = response.json()["ndf_comparison"]
    assert comparison is not None
    assert comparison["combination_weight"] == pytest.approx(0.1744)
    assert comparison["ndf_forecast_mw"] == 25000.0


def test_predict_omits_ndf_comparison_when_no_ndf_rows(monkeypatch):
    def _no_ndf(now_utc: pd.Timestamp) -> live_pipeline_module.LiveInputs:
        live_inputs = _fake_live_inputs(now_utc)
        live_inputs.ndf = live_inputs.ndf.iloc[0:0]
        return live_inputs

    monkeypatch.setattr(live_pipeline_module, "fetch_live_inputs", _no_ndf)
    with TestClient(app) as test_client:
        response = test_client.get("/api/predict")

    assert response.status_code == 200
    assert response.json()["ndf_comparison"] is None
