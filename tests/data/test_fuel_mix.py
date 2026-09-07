from itertools import pairwise

import pandas as pd

from edf.data.fuel_mix import fetch_fuel_mix


def test_fetch_fuel_mix_pivots_to_wide_format():
    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        index = pd.date_range(start, periods=2, freq="30min", tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": list(index) * 2,
                "fuel_type": ["PS"] * 2 + ["NUCLEAR"] * 2,
                "generation_mw": [100.0, 150.0, 5000.0, 5010.0],
            }
        )

    result = fetch_fuel_mix("2024-01-01", "2024-01-02", fetch=fake_fetch)

    assert set(result.columns) == {"PS", "NUCLEAR"}
    assert result.index.name == "timestamp"
    assert result["PS"].tolist() == [100.0, 150.0]
    assert result["NUCLEAR"].tolist() == [5000.0, 5010.0]


def test_fetch_fuel_mix_chunks_into_max_7_day_windows():
    calls = []

    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        calls.append((start, end))
        return pd.DataFrame(
            {"timestamp": [start], "fuel_type": ["PS"], "generation_mw": [1.0]}
        )

    fetch_fuel_mix("2024-01-01", "2024-01-20", fetch=fake_fetch)

    # 20-day span with a 7-day max window -> 3 calls (7 + 7 + 6 days)
    assert len(calls) == 3
    for start, end in calls:
        assert (end - start) <= pd.Timedelta(days=7)
    # windows are contiguous, no gap or overlap
    for (_, end), (next_start, _) in pairwise(calls):
        assert next_start == end + pd.Timedelta(seconds=1)


def test_fetch_fuel_mix_drops_duplicate_timestamp_fuel_type_pairs():
    def fake_fetch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        # simulate the same row appearing twice (e.g. overlapping windows)
        return pd.DataFrame(
            {
                "timestamp": [start, start],
                "fuel_type": ["PS", "PS"],
                "generation_mw": [100.0, 100.0],
            }
        )

    result = fetch_fuel_mix("2024-01-01", "2024-01-01T00:00:00Z", fetch=fake_fetch)

    assert len(result) == 1
