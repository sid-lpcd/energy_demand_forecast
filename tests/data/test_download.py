import pandas as pd

from edf.data.download import settlement_periods_to_utc


def test_regular_day_first_and_last_period():
    dates = pd.Series(["15-JUN-2022", "15-JUN-2022"])
    periods = pd.Series([1, 48])
    result = settlement_periods_to_utc(dates, periods)

    assert result.iloc[0] == pd.Timestamp("2022-06-14T23:00:00Z")  # 00:00 BST
    assert result.iloc[1] == pd.Timestamp("2022-06-15T22:30:00Z")  # 23:30 BST


def test_spring_forward_day_has_46_periods_and_no_duplicate_gap():
    # 27-MAR-2022: UK clocks went forward, this is a 23-hour day.
    dates = pd.Series(["27-MAR-2022"] * 46)
    periods = pd.Series(range(1, 47))
    result = settlement_periods_to_utc(dates, periods)

    assert result.is_monotonic_increasing
    assert result.is_unique
    assert result.iloc[0] == pd.Timestamp("2022-03-27T00:00:00Z")  # 00:00 GMT
    # Last period lands at 23:30 local (BST, UTC+1) on the same UTC day.
    assert result.iloc[-1] == pd.Timestamp("2022-03-27T22:30:00Z")


def test_handles_inconsistent_date_formats_across_years():
    # NESO's own yearly CSVs disagree on SETTLEMENT_DATE format.
    dates = pd.Series(["01-JAN-2020", "01-Jan-23", "2025-01-01"])
    periods = pd.Series([1, 1, 1])
    result = settlement_periods_to_utc(dates, periods)

    assert list(result.dt.date.astype(str)) == ["2020-01-01", "2023-01-01", "2025-01-01"]


def test_fall_back_day_has_50_periods_and_no_duplicate_gap():
    # 30-OCT-2022: UK clocks went back, this is a 25-hour day.
    dates = pd.Series(["30-OCT-2022"] * 50)
    periods = pd.Series(range(1, 51))
    result = settlement_periods_to_utc(dates, periods)

    assert result.is_monotonic_increasing
    assert result.is_unique
    assert result.iloc[0] == pd.Timestamp("2022-10-29T23:00:00Z")  # 00:00 BST
    assert result.iloc[-1] == pd.Timestamp("2022-10-30T23:30:00Z")  # 23:30 GMT
