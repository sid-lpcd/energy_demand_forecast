import numpy as np
import pandas as pd

from edf.models.combination import HEADLINE_COMBINATION_WEIGHT
from edf.models.history_comparison import aggregate_daily_comparison, merge_demand_sources


def test_merge_demand_sources_prefers_live_on_overlap_and_dedupes():
    index = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
    archive = pd.DataFrame({"timestamp": index, "demand": [1.0, 2.0, 3.0, 4.0]})
    live = pd.DataFrame({"timestamp": index[2:], "demand": [30.0, 40.0]})  # overlaps last 2 rows

    merged = merge_demand_sources(archive, live)

    assert list(merged["timestamp"]) == list(index)
    assert list(merged["demand"]) == [1.0, 2.0, 30.0, 40.0]


def test_merge_demand_sources_sorts_even_if_inputs_are_out_of_order():
    index = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    archive = pd.DataFrame({"timestamp": index[::-1], "demand": [3.0, 2.0, 1.0]})
    live = pd.DataFrame({"timestamp": [], "demand": []})

    merged = merge_demand_sources(archive, live)

    assert list(merged["timestamp"]) == list(index)
    assert list(merged["demand"]) == [1.0, 2.0, 3.0]


def test_aggregate_daily_comparison_blends_and_daily_means():
    # Two half-hourly periods per day, two days -- easy to hand-check the daily mean.
    index = pd.date_range("2026-01-01", periods=4, freq="12h", tz="UTC")
    actual = pd.Series([100.0, 200.0, 300.0, 400.0], index=index)
    ndf = pd.Series([110.0, 190.0, 310.0, 390.0], index=index)
    our_forecast = pd.Series([90.0, 210.0, 290.0, 410.0], index=index)

    daily = aggregate_daily_comparison(
        actual, ndf, our_forecast, weight_a=0.5, window_days=30
    )

    assert list(daily["date"].astype(str)) == ["2026-01-01", "2026-01-02"]
    assert daily["actual_demand_mw"].tolist() == [150.0, 350.0]
    assert daily["ndf_forecast_mw"].tolist() == [150.0, 350.0]
    # blended = 0.5*our_forecast + 0.5*ndf, averaged per day
    expected_blend_day1 = (0.5 * 90.0 + 0.5 * 110.0 + 0.5 * 210.0 + 0.5 * 190.0) / 2
    assert daily["blended_forecast_mw"].iloc[0] == expected_blend_day1


def test_aggregate_daily_comparison_keeps_forecast_rows_with_no_actual_yet():
    index = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    actual = pd.Series([100.0, np.nan, np.nan], index=index)  # actual lags behind
    ndf = pd.Series([110.0, 210.0, 310.0], index=index)
    our_forecast = pd.Series([90.0, 190.0, 290.0], index=index)

    daily = aggregate_daily_comparison(actual, ndf, our_forecast, window_days=30)

    assert len(daily) == 3
    assert daily["actual_demand_mw"].iloc[0] == 100.0
    assert daily["actual_demand_mw"].iloc[1:].isna().all()
    assert daily["ndf_forecast_mw"].notna().all()


def test_aggregate_daily_comparison_trims_to_trailing_window():
    index = pd.date_range("2025-01-01", periods=400, freq="D", tz="UTC")
    series = pd.Series(np.arange(len(index), dtype=float), index=index)

    daily = aggregate_daily_comparison(series, series, series, window_days=365)

    span_days = (pd.Timestamp(daily["date"].max()) - pd.Timestamp(daily["date"].min())).days
    assert span_days <= 365
    assert daily["date"].max() == index.max().date()


def test_default_headline_weight_used_when_unspecified():
    index = pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC")
    ndf = pd.Series([100.0], index=index)
    our_forecast = pd.Series([200.0], index=index)
    actual = pd.Series([150.0], index=index)

    daily = aggregate_daily_comparison(actual, ndf, our_forecast, window_days=1)

    expected = HEADLINE_COMBINATION_WEIGHT * 200.0 + (1 - HEADLINE_COMBINATION_WEIGHT) * 100.0
    assert daily["blended_forecast_mw"].iloc[0] == expected
