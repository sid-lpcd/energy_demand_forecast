import pandas as pd

from edf.data.demand_forecast_benchmark import cardinal_point_to_utc


def test_regular_times_convert_correctly():
    dates = pd.Series(["2022-06-15", "2022-06-15", "2022-06-15"])
    times = pd.Series(["30", "930", "1700"])
    result = cardinal_point_to_utc(dates, times)

    assert result.iloc[0] == pd.Timestamp("2022-06-14T23:30:00Z")  # 00:30 BST
    assert result.iloc[1] == pd.Timestamp("2022-06-15T08:30:00Z")  # 09:30 BST
    assert result.iloc[2] == pd.Timestamp("2022-06-15T16:00:00Z")  # 17:00 BST


def test_2400_rolls_to_midnight_of_the_next_day():
    dates = pd.Series(["2022-06-15"])
    times = pd.Series(["2400"])
    result = cardinal_point_to_utc(dates, times)

    assert result.iloc[0] == pd.Timestamp("2022-06-15T23:00:00Z")  # 2022-06-16 00:00 BST


def test_winter_time_has_no_utc_offset():
    dates = pd.Series(["2022-01-15"])
    times = pd.Series(["800"])
    result = cardinal_point_to_utc(dates, times)

    assert result.iloc[0] == pd.Timestamp("2022-01-15T08:00:00Z")  # GMT, no offset


def test_dst_nonexistent_local_time_does_not_raise():
    # 2022-03-27: UK clocks went forward at 01:00, so 01:30 local never existed.
    dates = pd.Series(["2022-03-27"])
    times = pd.Series(["130"])
    result = cardinal_point_to_utc(dates, times)
    assert result.notna().all()  # shifted forward, not dropped or raised
