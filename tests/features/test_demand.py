import numpy as np
import pandas as pd

from edf.features.demand import (
    CALENDAR_COLUMNS,
    build_feature_table,
    deterministic_features,
    fourier_terms,
    lag_features,
    rolling_features,
    time_features,
)
from edf.models.baselines import PERIODS_PER_DAY, PERIODS_PER_WEEK


def _canonical_df(n: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "demand": np.arange(n, dtype=float),
            "is_bank_holiday": [False] * n,
            "is_school_holiday": [False] * n,
            "lockdown_level": pd.Categorical(["none"] * n, categories=["none", "partial", "full"]),
            "event_tier": pd.Categorical(
                ["none"] * n, categories=["none", "small", "medium", "large"]
            ),
            "solar_eclipse_pct": [0.0] * n,
        },
        index=index,
    )


def test_fourier_terms_bounded_and_periodic():
    index = pd.date_range("2024-01-01", periods=200, freq="30min", tz="UTC")
    terms = fourier_terms(index, period_periods=PERIODS_PER_DAY, order=2, label="daily")

    assert list(terms.columns) == [
        "fourier_daily_sin_1",
        "fourier_daily_cos_1",
        "fourier_daily_sin_2",
        "fourier_daily_cos_2",
    ]
    assert (terms.abs() <= 1).all().all()
    # one full daily period later, values must repeat
    later = fourier_terms(
        index + pd.Timedelta(minutes=30 * PERIODS_PER_DAY), period_periods=PERIODS_PER_DAY, order=2, label="daily"
    )
    np.testing.assert_allclose(terms.to_numpy(), later.to_numpy(), atol=1e-9)


def test_time_features_ranges():
    index = pd.date_range("2024-01-01", periods=PERIODS_PER_DAY * 3, freq="30min", tz="UTC")
    result = time_features(index)

    assert result["hour_of_day"].between(0, 47).all()
    assert result["day_of_week"].between(0, 6).all()
    assert result["month"].between(1, 12).all()
    assert result.shape[1] == 3 + 4 + 4 + 4  # base cols + 3 fourier groups x (order=2 x sin/cos)


def test_lag_features_filters_by_horizon():
    demand = pd.Series(np.arange(PERIODS_PER_WEEK + 10, dtype=float))

    at_shortest = lag_features(demand, horizon_periods=1)
    assert set(at_shortest.columns) == {"lag_1", "lag_2", "lag_48", "lag_336"}

    at_weekly = lag_features(demand, horizon_periods=PERIODS_PER_WEEK)
    assert set(at_weekly.columns) == {"lag_336"}


def test_rolling_features_filters_by_horizon():
    demand = pd.Series(np.arange(PERIODS_PER_WEEK * 4 + 10, dtype=float))

    at_shortest = rolling_features(demand, horizon_periods=1)
    assert set(at_shortest.columns) == {"trailing_moving_average", "trailing_weekly_moving_average"}

    at_daily_edge = rolling_features(demand, horizon_periods=PERIODS_PER_DAY + 1)
    assert set(at_daily_edge.columns) == {"trailing_weekly_moving_average"}


def test_deterministic_features_includes_calendar_columns():
    df = _canonical_df(10)
    result = deterministic_features(df)

    for col in CALENDAR_COLUMNS:
        assert col in result.columns
    assert str(result["lockdown_level"].dtype) == "category"
    assert str(result["hour_of_day"].dtype) == "category"


def test_build_feature_table_drops_warmup_rows_and_aligns_with_target():
    n = PERIODS_PER_WEEK * 4 + 20
    df = _canonical_df(n)

    X, y = build_feature_table(df, horizon_periods=1)

    assert not X.isna().any().any()
    assert list(X.index) == list(y.index)
    # every row should have all 4 lag columns + both rolling columns present at horizon=1
    assert {"lag_1", "lag_2", "lag_48", "lag_336"}.issubset(X.columns)
    assert y.equals(df.loc[X.index, "demand"])


def test_build_feature_table_horizon_336_only_has_valid_lags():
    n = PERIODS_PER_WEEK * 4 + 20
    df = _canonical_df(n)

    X, _ = build_feature_table(df, horizon_periods=PERIODS_PER_WEEK)

    assert "lag_1" not in X.columns
    assert "lag_336" in X.columns
    assert "trailing_moving_average" not in X.columns
    assert "trailing_weekly_moving_average" in X.columns


def test_build_feature_table_merges_optional_weather_columns():
    n = PERIODS_PER_WEEK * 4 + 20
    df = _canonical_df(n)
    weather = pd.DataFrame(
        {"temperature_c": np.arange(n, dtype=float), "heating_degree": np.zeros(n)},
        index=df.index,
    )

    X, y = build_feature_table(df, horizon_periods=1, weather=weather)

    assert {"temperature_c", "heating_degree"}.issubset(X.columns)
    assert not X.isna().any().any()
    assert y.equals(df.loc[X.index, "demand"])


def test_build_feature_table_drops_rows_where_weather_is_missing():
    n = PERIODS_PER_WEEK * 4 + 20
    df = _canonical_df(n)
    weather = pd.DataFrame(
        {"temperature_c": np.arange(n, dtype=float)},
        index=df.index,
    )
    weather.iloc[-1] = np.nan  # simulate weather source not covering the last row

    X, _ = build_feature_table(df, horizon_periods=1, weather=weather)

    assert df.index[-1] not in X.index
