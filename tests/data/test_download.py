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


def test_iso_dates_do_not_get_day_month_swapped():
    # Regression guard for a real bug found 2026-09-08: pandas' `dayfirst=True`
    # silently swapped month/day for "YYYY-MM-DD" strings whenever both
    # components were <=12 (e.g. "2025-01-06" parsed as 2025-06-01), even
    # though the ISO format is actually unambiguous. `test_handles_inconsistent_
    # date_formats_across_years` above didn't catch it because "2025-01-01" is
    # self-symmetric (day == month) -- deliberately use an asymmetric pair here.
    # Period 20 (~mid-morning local) keeps the UTC calendar date equal to the
    # local one regardless of BST/GMT, so this isolates the swap bug from the
    # (correct, separately tested) local-midnight-crosses-UTC-day behavior.
    dates = pd.Series(["2025-01-06", "2025-06-01", "2025-03-04"])
    periods = pd.Series([20, 20, 20])
    result = settlement_periods_to_utc(dates, periods)

    assert list(result.dt.date.astype(str)) == ["2025-01-06", "2025-06-01", "2025-03-04"]


def test_fall_back_day_has_50_periods_and_no_duplicate_gap():
    # 30-OCT-2022: UK clocks went back, this is a 25-hour day.
    dates = pd.Series(["30-OCT-2022"] * 50)
    periods = pd.Series(range(1, 51))
    result = settlement_periods_to_utc(dates, periods)

    assert result.is_monotonic_increasing
    assert result.is_unique
    assert result.iloc[0] == pd.Timestamp("2022-10-29T23:00:00Z")  # 00:00 BST
    assert result.iloc[-1] == pd.Timestamp("2022-10-30T23:30:00Z")  # 23:30 GMT
