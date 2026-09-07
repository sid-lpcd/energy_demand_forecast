import numpy as np
import pandas as pd

from edf.baselines import (
    PERIODS_PER_DAY,
    PERIODS_PER_WEEK,
    naive,
    persistence,
    previous_day_same_time,
    previous_week_same_time,
    trailing_moving_average,
    trailing_weekly_moving_average,
    valid_baselines,
)


def _demand_series(n: int) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    return pd.Series(range(n), index=index, dtype=float)


def test_naive_shifts_by_one_period():
    demand = _demand_series(5)
    result = naive(demand)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].tolist() == demand.iloc[:-1].tolist()


def test_previous_day_same_time_shifts_by_48_periods():
    demand = _demand_series(PERIODS_PER_DAY + 3)
    result = previous_day_same_time(demand)

    assert result.iloc[:PERIODS_PER_DAY].isna().all()
    assert result.iloc[PERIODS_PER_DAY] == demand.iloc[0]
    assert result.iloc[PERIODS_PER_DAY + 2] == demand.iloc[2]


def test_previous_week_same_time_shifts_by_336_periods():
    demand = _demand_series(PERIODS_PER_WEEK + 2)
    result = previous_week_same_time(demand)

    assert result.iloc[:PERIODS_PER_WEEK].isna().all()
    assert result.iloc[PERIODS_PER_WEEK] == demand.iloc[0]
    assert result.iloc[PERIODS_PER_WEEK + 1] == demand.iloc[1]


def test_trailing_moving_average_averages_last_n_days_same_time():
    demand = _demand_series(PERIODS_PER_DAY * 4 + 1)
    result = trailing_moving_average(demand, n_days=4)

    # At the first fully-available index, the average is over values
    # 0, 48, 96, 144 (four prior days at the same half-hour-of-day).
    idx = PERIODS_PER_DAY * 4
    expected = np.mean([demand.iloc[idx - PERIODS_PER_DAY * k] for k in range(1, 5)])
    assert result.iloc[idx] == expected


def test_trailing_moving_average_nan_during_warmup():
    demand = _demand_series(PERIODS_PER_DAY * 4 + 1)
    result = trailing_moving_average(demand, n_days=4)

    assert result.iloc[: PERIODS_PER_DAY * 4].isna().all()


def test_persistence_shifts_by_given_horizon():
    demand = _demand_series(10)

    result = persistence(demand, horizon_periods=3)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].tolist() == demand.iloc[:-3].tolist()


def test_naive_is_persistence_at_horizon_one():
    demand = _demand_series(5)

    assert naive(demand).equals(persistence(demand, horizon_periods=1))


def test_trailing_weekly_moving_average_averages_last_n_weeks_same_time():
    demand = _demand_series(PERIODS_PER_WEEK * 4 + 1)
    result = trailing_weekly_moving_average(demand, n_weeks=4)

    idx = PERIODS_PER_WEEK * 4
    expected = np.mean([demand.iloc[idx - PERIODS_PER_WEEK * k] for k in range(1, 5)])
    assert result.iloc[idx] == expected
    assert result.iloc[:idx].isna().all()


def test_valid_baselines_excludes_fixed_lags_shorter_than_horizon():
    # 7-day horizon (336 periods): the daily-lag baselines (lag 48) would need
    # data that isn't known yet at issue time, so they must be excluded.
    valid = valid_baselines(horizon_periods=PERIODS_PER_WEEK)

    assert "previous_day_same_time" not in valid
    assert "trailing_moving_average" not in valid
    assert "previous_week_same_time" in valid
    assert "trailing_weekly_moving_average" in valid
    assert "persistence" in valid


def test_valid_baselines_includes_everything_at_the_shortest_horizon():
    valid = valid_baselines(horizon_periods=1)

    assert set(valid) == {
        "previous_day_same_time",
        "previous_week_same_time",
        "trailing_moving_average",
        "trailing_weekly_moving_average",
        "persistence",
    }


def test_valid_baselines_persistence_uses_requested_horizon():
    demand = _demand_series(5)
    valid = valid_baselines(horizon_periods=2)

    assert valid["persistence"](demand).equals(persistence(demand, horizon_periods=2))
