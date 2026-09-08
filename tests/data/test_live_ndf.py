from edf.data.live_ndf import parse_live_ndf_csv

_CSV = """DAYSAHEAD,TARGETDATE,FORECASTDEMAND,CARDINALPOINT,CP_TYPE,CP_ST_TIME,CP_END_TIME,F_Point
1,20260909,20373,1F,F,30,30,
1,20260909,19970,1A,P,200,230,
2,20260910,18535,1B,T,430,730,Om
"""


def test_keeps_only_day_ahead_rows():
    result = parse_live_ndf_csv(_CSV)

    assert list(result["forecast_demand_mw"]) == [20373, 19970]


def test_output_columns():
    result = parse_live_ndf_csv(_CSV)

    assert list(result.columns) == [
        "timestamp",
        "TARGETDATE",
        "CARDINALPOINT",
        "CP_TYPE",
        "forecast_demand_mw",
    ]


def test_result_is_sorted_by_timestamp():
    result = parse_live_ndf_csv(_CSV)

    assert result["timestamp"].is_monotonic_increasing


def test_bare_yyyymmdd_targetdate_is_not_misparsed_as_epoch_nanoseconds():
    # Regression guard for a real bug found via a live smoke test (2026-09-08): the live
    # resource's TARGETDATE is a bare YYYYMMDD int ("20260909"), unlike the historic archive's
    # dashed date strings -- an untyped pd.to_datetime silently reads a raw int64 as
    # nanoseconds-since-epoch instead of a date, producing 1969/1970 garbage.
    result = parse_live_ndf_csv(_CSV)

    # 00:30 and 02:00 local (BST, UTC+1) on the 2026-09-09 target date -- the
    # first period crosses into the previous UTC calendar day, the second
    # doesn't; both landing in 2026, not 1969/1970, is what this guards.
    assert list(result["timestamp"].dt.date.astype(str)) == ["2026-09-08", "2026-09-09"]
