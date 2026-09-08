import numpy as np
import pandas as pd
import pytest

from app.live_pipeline import (
    LOOKBACK_DAYS,
    LiveInputs,
    LiveInputsCache,
    build_live_features,
    settlement_floor,
)
from edf.data.calendar_events import build_calendar_features
from edf.features.demand import build_feature_table
from edf.features.weather import build_weather_feature_table_from_frame


def _live_inputs(history_days: int = LOOKBACK_DAYS + 5, issue_time: pd.Timestamp | None = None):
    issue_time = issue_time or pd.Timestamp("2026-06-15 12:00:00", tz="UTC")
    history_index = pd.date_range(
        issue_time - pd.Timedelta(days=history_days), issue_time, freq="30min", tz="UTC"
    )
    rng = np.random.default_rng(0)
    demand_history = pd.DataFrame(
        {
            "timestamp": history_index,
            "demand": 20000 + 5000 * np.sin(np.arange(len(history_index)) / 48) + rng.normal(0, 5, len(history_index)),
        }
    )

    weather_index = pd.date_range(
        issue_time - pd.Timedelta(days=1), issue_time + pd.Timedelta(days=9), freq="h", tz="UTC"
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

    ndf = pd.DataFrame(columns=["timestamp", "TARGETDATE", "CARDINALPOINT", "CP_TYPE", "forecast_demand_mw"])
    return LiveInputs(
        demand_history=demand_history, weather_forecast=weather_forecast, ndf=ndf, fetched_at=issue_time
    )


def _expected_columns(live_inputs: LiveInputs, horizon_periods: int) -> list[str]:
    issue_time = live_inputs.demand_history["timestamp"].max()
    target_time = issue_time + pd.Timedelta(minutes=30 * horizon_periods)
    start = issue_time - pd.Timedelta(days=LOOKBACK_DAYS)
    index = pd.date_range(start, target_time, freq="30min", tz="UTC")
    demand = live_inputs.demand_history.set_index("timestamp")["demand"].reindex(index).rename("demand")
    calendar = build_calendar_features(index)
    df = pd.concat([demand, calendar], axis=1)
    weather = build_weather_feature_table_from_frame(live_inputs.weather_forecast, df.index)
    X_base, _ = build_feature_table(df, horizon_periods, weather=weather)
    return list(X_base.columns)


def test_settlement_floor_rounds_down_to_half_hour():
    assert settlement_floor(pd.Timestamp("2026-09-08T10:47:00Z")) == pd.Timestamp("2026-09-08T10:30:00Z")
    assert settlement_floor(pd.Timestamp("2026-09-08T10:00:00Z")) == pd.Timestamp("2026-09-08T10:00:00Z")
    assert settlement_floor(pd.Timestamp("2026-09-08T10:29:59Z")) == pd.Timestamp("2026-09-08T10:00:00Z")


def test_build_live_features_target_time_and_columns_for_non_bias_corrected_horizon():
    live_inputs = _live_inputs()
    expected_columns = _expected_columns(live_inputs, horizon_periods=1)
    metadata = {
        "horizon_periods": 1,
        "bias_corrected": False,
        "point_feature_columns": expected_columns,
        "quantile_feature_columns": expected_columns,
    }

    result = build_live_features("30min", 1, metadata, live_inputs)

    assert result.target_time == result.issue_time + pd.Timedelta(minutes=30)
    assert list(result.point_row.columns) == expected_columns
    assert list(result.quantile_row.columns) == expected_columns
    assert len(result.point_row) == 1
    assert result.point_row.index[0] == result.target_time


def test_build_live_features_adds_bias_corrected_columns_for_1d():
    live_inputs = _live_inputs()
    base_columns = _expected_columns(live_inputs, horizon_periods=48)
    point_columns = [*base_columns, "is_christmas", "heating_degree_1d_cum", "cooling_degree_1d_cum"]
    metadata = {
        "horizon_periods": 48,
        "bias_corrected": True,
        "point_feature_columns": point_columns,
        "quantile_feature_columns": base_columns,
    }

    result = build_live_features("1d", 48, metadata, live_inputs)

    assert list(result.point_row.columns) == point_columns
    assert list(result.quantile_row.columns) == base_columns
    # quantile row must NOT carry the bias-correction extras -- notebooks/14 found those
    # measurably worsen quantile calibration, so they must never reach the quantile models.
    assert "is_christmas" not in result.quantile_row.columns


def test_build_live_features_raises_on_missing_expected_column():
    live_inputs = _live_inputs()
    metadata = {
        "horizon_periods": 1,
        "bias_corrected": False,
        "point_feature_columns": ["a_column_that_will_never_exist"],
        "quantile_feature_columns": ["a_column_that_will_never_exist"],
    }

    with pytest.raises(ValueError, match="feature-parity"):
        build_live_features("30min", 1, metadata, live_inputs)


def test_build_live_features_raises_when_history_too_short_for_horizon():
    # Only a few hours of history -- can't satisfy trailing_weekly_moving_average's warm-up.
    live_inputs = _live_inputs(history_days=1)
    expected_columns = _expected_columns(_live_inputs(history_days=LOOKBACK_DAYS + 5), horizon_periods=1)
    metadata = {
        "horizon_periods": 1,
        "bias_corrected": False,
        "point_feature_columns": expected_columns,
        "quantile_feature_columns": expected_columns,
    }

    with pytest.raises(ValueError, match="no complete feature row"):
        build_live_features("30min", 1, metadata, live_inputs)


def test_live_inputs_cache_only_refetches_when_settlement_period_changes():
    calls = []

    def fake_fetch(now_utc):
        calls.append(now_utc)
        return _live_inputs(issue_time=now_utc)

    cache = LiveInputsCache()
    import app.live_pipeline as live_pipeline_module

    original = live_pipeline_module.fetch_live_inputs
    live_pipeline_module.fetch_live_inputs = fake_fetch
    try:
        t0 = pd.Timestamp("2026-06-15T10:05:00Z")
        t1 = pd.Timestamp("2026-06-15T10:25:00Z")  # same settlement period as t0
        t2 = pd.Timestamp("2026-06-15T10:35:00Z")  # next settlement period

        cache.get(t0)
        cache.get(t1)
        cache.get(t2)
    finally:
        live_pipeline_module.fetch_live_inputs = original

    assert len(calls) == 2


def test_live_inputs_cache_serves_stale_inputs_when_refetch_fails():
    good_inputs = _live_inputs(issue_time=pd.Timestamp("2026-06-15T10:00:00Z"))
    calls = {"n": 0}

    def flaky_fetch(now_utc):
        calls["n"] += 1
        if calls["n"] == 1:
            return good_inputs
        raise RuntimeError("upstream rate limited")

    cache = LiveInputsCache()
    import app.live_pipeline as live_pipeline_module

    original = live_pipeline_module.fetch_live_inputs
    live_pipeline_module.fetch_live_inputs = flaky_fetch
    try:
        t0 = pd.Timestamp("2026-06-15T10:05:00Z")
        t1 = pd.Timestamp("2026-06-15T10:35:00Z")  # next settlement period -> refetch attempted

        first = cache.get(t0)
        second = cache.get(t1)
    finally:
        live_pipeline_module.fetch_live_inputs = original

    assert calls["n"] == 2
    assert second is first is good_inputs


def test_live_inputs_cache_raises_when_first_fetch_fails_with_no_stale_fallback():
    def always_fails(now_utc):
        raise RuntimeError("upstream rate limited")

    cache = LiveInputsCache()
    import app.live_pipeline as live_pipeline_module

    original = live_pipeline_module.fetch_live_inputs
    live_pipeline_module.fetch_live_inputs = always_fails
    try:
        with pytest.raises(RuntimeError, match="upstream rate limited"):
            cache.get(pd.Timestamp("2026-06-15T10:05:00Z"))
    finally:
        live_pipeline_module.fetch_live_inputs = original
