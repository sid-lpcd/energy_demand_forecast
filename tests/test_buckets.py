import numpy as np
import pandas as pd

from edf.buckets import (
    build_day_type_buckets,
    is_christmas_period,
    is_weekend,
    month_relative_percentile_bucket,
    overall_percentile_bucket,
    percentile_bin_buckets,
)


def test_month_relative_percentile_bucket_flags_relative_to_own_month():
    # January values 0..99 (cold month, low absolute values), July values
    # 900..999 (hot month, high absolute values) -- both months' extremes
    # should be caught relative to *their own* distribution.
    jan_index = pd.date_range("2024-01-01", periods=100, freq="D")
    jul_index = pd.date_range("2024-07-01", periods=100, freq="D")
    index = jan_index.append(jul_index)
    values = pd.Series(list(range(100)) + list(range(900, 1000)), index=index)

    is_low, is_high = month_relative_percentile_bucket(values, low_pct=10, high_pct=90)

    # lowest ~10 January values flagged low, within January
    assert is_low.loc[jan_index].sum() == 10
    assert is_high.loc[jan_index].sum() == 10
    # July's low 10% are its own low values (900s), not compared against January
    assert is_low.loc[jul_index].sum() == 10
    assert is_high.loc[jul_index].sum() == 10


def test_overall_percentile_bucket_uses_fixed_thresholds():
    values = pd.Series(range(100), dtype=float)
    is_low, is_high = overall_percentile_bucket(values, low_pct=10, high_pct=90)

    assert is_low.sum() == 10
    assert is_high.sum() == 10
    assert is_low.iloc[0]  # value 0 is in the bottom 10
    assert is_high.iloc[-1]  # value 99 is in the top 10


def test_is_christmas_period_covers_year_boundary():
    index = pd.DatetimeIndex(
        ["2023-12-23", "2023-12-24", "2023-12-31", "2024-01-01", "2024-01-02"]
    )
    result = is_christmas_period(index)
    assert result.tolist() == [False, True, True, True, False]


def test_is_weekend_flags_saturday_sunday():
    index = pd.date_range("2024-01-01", periods=7, freq="D")  # Mon..Sun
    result = is_weekend(index)
    assert result.tolist() == [False, False, False, False, False, True, True]


def test_build_day_type_buckets_returns_expected_columns():
    index = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    temperature_c = pd.Series(rng.normal(10, 5, len(index)), index=index)
    wind_cf = pd.Series(rng.uniform(0, 1, len(index)), index=index)

    result = build_day_type_buckets(temperature_c, wind_cf)

    expected_columns = {"is_cold", "is_hot", "is_low_wind", "is_high_wind", "is_christmas", "is_weekend"}
    assert set(result.columns) == expected_columns
    assert list(result.index) == list(index)
    assert result["is_cold"].dtype == bool


def test_percentile_bin_buckets_partitions_evenly_and_exhaustively():
    values = pd.Series(range(100), dtype=float)

    result = percentile_bin_buckets(values, n_bins=5, prefix="renewable")

    assert set(result.columns) == {f"renewable_q{i}" for i in range(1, 6)}
    # every row belongs to exactly one bin
    assert (result.sum(axis=1) == 1).all()
    # each bin has an equal share (100 values, 5 even bins -> 20 each)
    assert (result.sum(axis=0) == 20).all()


def test_percentile_bin_buckets_orders_bins_low_to_high():
    values = pd.Series(range(10), dtype=float)

    result = percentile_bin_buckets(values, n_bins=2, prefix="x")

    # lowest values (0-4) should be in bin 1, highest (5-9) in bin 2
    assert result.loc[result["x_q1"]].index.tolist() == list(range(5))
    assert result.loc[result["x_q2"]].index.tolist() == list(range(5, 10))


def test_percentile_bin_buckets_drops_degenerate_bins_on_heavy_ties():
    # almost all zeros -> can't form 10 distinct quantile edges
    values = pd.Series([0.0] * 95 + list(range(1, 6)))

    result = percentile_bin_buckets(values, n_bins=10, prefix="x")

    assert result.shape[1] < 10
    assert (result.sum(axis=1) == 1).all()
