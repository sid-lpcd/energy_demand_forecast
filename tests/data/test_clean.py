import pandas as pd
import pytest

from edf.data.clean import (
    INTERCONNECTOR_FLOW_COLUMNS,
    build_canonical_table,
    check_no_gaps,
    check_no_negative_demand,
)


def _raw_frame(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(timestamps)
    data = {
        "timestamp": timestamps,
        "ND": [30000 + i for i in range(n)],
        "EMBEDDED_WIND_GENERATION": [1000 + i for i in range(n)],
        "EMBEDDED_SOLAR_GENERATION": [500 + i for i in range(n)],
        "EMBEDDED_WIND_CAPACITY": [15000] * n,
        "EMBEDDED_SOLAR_CAPACITY": [14000] * n,
    }
    for col in INTERCONNECTOR_FLOW_COLUMNS:
        data[col] = [10] * n
    return pd.DataFrame(data)


def test_check_no_gaps_passes_on_regular_series():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="30min", tz="UTC")
    check_no_gaps(pd.Series(timestamps))  # should not raise


def test_check_no_gaps_raises_on_gap():
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00Z", "2024-01-01T00:30Z", "2024-01-01T01:30Z"]
    )
    with pytest.raises(ValueError):
        check_no_gaps(pd.Series(timestamps))


def test_check_no_gaps_raises_on_duplicate():
    timestamps = pd.to_datetime(
        ["2024-01-01T00:00Z", "2024-01-01T00:30Z", "2024-01-01T00:30Z"]
    )
    with pytest.raises(ValueError):
        check_no_gaps(pd.Series(timestamps))


def test_check_no_negative_demand_raises():
    with pytest.raises(ValueError):
        check_no_negative_demand(pd.Series([100, -1, 200]))


def test_check_no_negative_demand_passes():
    check_no_negative_demand(pd.Series([100, 0, 200]))  # should not raise


def test_build_canonical_table_columns_and_values():
    timestamps = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    raw = _raw_frame(timestamps)
    result = build_canonical_table(raw)

    assert result.index.name == "timestamp"
    assert list(result.index) == list(timestamps)
    assert result["demand"].tolist() == raw["ND"].tolist()
    assert result["wind"].tolist() == raw["EMBEDDED_WIND_GENERATION"].tolist()
    assert result["solar"].tolist() == raw["EMBEDDED_SOLAR_GENERATION"].tolist()
    assert (result["interconnector"] == len(INTERCONNECTOR_FLOW_COLUMNS) * 10).all()
    assert result["wind_capacity"].tolist() == raw["EMBEDDED_WIND_CAPACITY"].tolist()
    assert result["solar_capacity"].tolist() == raw["EMBEDDED_SOLAR_CAPACITY"].tolist()
    # calendar features merged in
    assert "is_bank_holiday" in result.columns
    assert "lockdown_level" in result.columns


def test_build_canonical_table_flags_new_years_day_bank_holiday():
    timestamps = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    raw = _raw_frame(timestamps)
    result = build_canonical_table(raw)

    assert result["is_bank_holiday"].all()


def test_build_canonical_table_interconnector_can_net_negative():
    timestamps = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    raw = _raw_frame(timestamps)
    raw["IFA_FLOW"] = [-2000, -2000]
    result = build_canonical_table(raw)

    assert (result["interconnector"] < 0).all()
