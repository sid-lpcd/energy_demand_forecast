from edf.data.live_demand import parse_live_demand_csv

_CSV = """SETTLEMENT_DATE,SETTLEMENT_PERIOD,ND,FORECAST_ACTUAL_INDICATOR,TSD
2026-08-01,1,21548,A,24451
2026-08-01,2,21176,A,24012
2026-08-01,3,0,A,0
2026-09-15,46,0,F,0
2026-09-15,47,0,F,0
"""


def test_parses_only_actual_rows():
    result = parse_live_demand_csv(_CSV)

    assert list(result["demand"]) == [21548, 21176]
    assert result["timestamp"].is_monotonic_increasing


def test_output_columns_are_timestamp_and_demand():
    result = parse_live_demand_csv(_CSV)

    assert list(result.columns) == ["timestamp", "demand"]


def test_forecast_placeholder_rows_are_dropped():
    result = parse_live_demand_csv(_CSV)

    # The two "F" rows and the "A"-tagged-but-still-zero row all have ND=0 --
    # a real regression here would let placeholder zeros leak into what
    # should be real demand history.
    assert (result["demand"] == 0).sum() == 0


def test_actual_tagged_rows_still_pending_settlement_are_dropped():
    # Regression guard for a real bug found via a live smoke test (2026-09-08):
    # the most recent "A"-tagged row(s) can still carry a placeholder ND=0
    # ahead of the real reading landing -- FORECAST_ACTUAL_INDICATOR alone
    # isn't sufficient to identify genuine actuals.
    result = parse_live_demand_csv(_CSV)

    assert len(result) == 2
