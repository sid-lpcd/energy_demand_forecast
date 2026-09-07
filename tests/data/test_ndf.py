import pandas as pd

from edf.data.ndf import fetch_day_ahead_ndf


def test_fetch_day_ahead_ndf_keeps_only_day_before_publishes():
    # two revisions of the same settlement period: one published the day
    # before (day-ahead, keep) and one published same-day (intraday, drop)
    target = pd.Timestamp("2024-01-02T12:00:00Z")

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [target, target],
                "settlement_date": [pd.Timestamp("2024-01-02", tz="UTC")] * 2,
                "publish_time": [
                    pd.Timestamp("2024-01-01T23:45:00Z"),  # day-ahead
                    pd.Timestamp("2024-01-02T08:00:00Z"),  # intraday, same day
                ],
                "demand": [20000.0, 20500.0],
            }
        )

    result = fetch_day_ahead_ndf("2024-01-02", "2024-01-02", fetch=fake_fetch)

    assert len(result) == 1
    assert result.loc[target] == 20000.0


def test_fetch_day_ahead_ndf_keeps_latest_of_multiple_day_before_publishes():
    target = pd.Timestamp("2024-01-02T12:00:00Z")

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [target, target],
                "settlement_date": [pd.Timestamp("2024-01-02", tz="UTC")] * 2,
                "publish_time": [
                    pd.Timestamp("2024-01-01T10:00:00Z"),  # earlier revision
                    pd.Timestamp("2024-01-01T23:45:00Z"),  # latest, still day-before
                ],
                "demand": [19000.0, 20000.0],
            }
        )

    result = fetch_day_ahead_ndf("2024-01-02", "2024-01-02", fetch=fake_fetch)

    assert len(result) == 1
    assert result.loc[target] == 20000.0


def test_fetch_day_ahead_ndf_chunks_into_1_day_windows():
    calls = []

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls.append((start, end))
        return pd.DataFrame(columns=["timestamp", "settlement_date", "publish_time", "demand"])

    fetch_day_ahead_ndf("2024-01-01", "2024-01-05", fetch=fake_fetch)

    # start-1 day (for day-before coverage) through end, in <= 1-day windows:
    # 2023-12-31 .. 2024-01-05 inclusive = 6 calls
    assert len(calls) == 6
    for start, end in calls:
        assert (end - start) <= pd.Timedelta(days=1)


def test_fetch_day_ahead_ndf_returns_empty_series_when_no_data():
    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "settlement_date", "publish_time", "demand"])

    result = fetch_day_ahead_ndf("2024-01-01", "2024-01-01", fetch=fake_fetch)

    assert result.empty
    assert result.name == "ndf_day_ahead"
