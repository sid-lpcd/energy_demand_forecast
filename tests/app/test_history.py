import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def test_history_endpoint_returns_empty_days_when_csv_not_present():
    # No monkeypatching: models_registry/ is real and committed, but reports/ndf_comparison_last_year.csv
    # is a locally-generated artifact this test suite doesn't assume exists.
    with TestClient(app) as client:
        if client.app.state.history_comparison is not None:
            return  # CSV happens to exist in this environment -- covered by the populated-data test instead
        response = client.get("/api/history")

    assert response.status_code == 200
    data = response.json()
    assert data["days"] == []
    assert data["weather_upper_bound_note"]
    assert data["scottish_transfer_advisory"]


def test_history_endpoint_serves_populated_comparison_data():
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    daily = pd.DataFrame(
        {
            "date": index.date.astype(str),
            "actual_demand_mw": [20000.0, np.nan, 21000.0],
            "ndf_forecast_mw": [20500.0, 20800.0, 21200.0],
            "our_forecast_mw": [19800.0, 20600.0, 21400.0],
            "blended_forecast_mw": [20378.0, 20765.0, 21235.0],
        }
    )

    with TestClient(app) as client:
        client.app.state.history_comparison = daily
        response = client.get("/api/history")

    assert response.status_code == 200
    data = response.json()
    assert len(data["days"]) == 3
    assert data["days"][0]["date"] == "2026-01-01"
    assert data["days"][0]["actual_demand_mw"] == 20000.0
    assert data["days"][1]["actual_demand_mw"] is None
    assert data["days"][2]["ndf_forecast_mw"] == 21200.0
