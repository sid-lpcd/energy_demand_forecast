import numpy as np
import pandas as pd

from edf.baselines import (
    PERIODS_PER_DAY,
    PERIODS_PER_WEEK,
    naive,
    previous_day_same_time,
    previous_week_same_time,
    trailing_moving_average,
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
