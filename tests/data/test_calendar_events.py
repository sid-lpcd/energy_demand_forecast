from datetime import date

import pandas as pd

from edf.data.calendar_events import (
    ALL_EVENTS,
    bank_holidays,
    build_calendar_features,
    clap_for_carers_events,
    lockdown_level,
)


def test_bank_holidays_include_one_off_dates():
    bh = bank_holidays(range(2020, 2024))
    # VE Day 75th anniversary move, Platinum Jubilee extra day,
    # Queen's state funeral, Coronation of Charles III.
    assert date(2020, 5, 8) in bh
    assert date(2020, 5, 4) not in bh  # the date it was moved *from*
    assert date(2022, 6, 3) in bh
    assert date(2022, 9, 19) in bh
    assert date(2023, 5, 8) in bh


def test_lockdown_level_boundaries():
    assert lockdown_level(date(2020, 3, 22)) == "none"
    assert lockdown_level(date(2020, 3, 23)) == "full"
    assert lockdown_level(date(2020, 5, 12)) == "full"
    assert lockdown_level(date(2020, 5, 13)) == "partial"
    assert lockdown_level(date(2021, 7, 18)) == "partial"
    assert lockdown_level(date(2021, 7, 19)) == "none"


def test_clap_for_carers_is_ten_thursdays():
    events = clap_for_carers_events()
    assert len(events) == 10
    assert all(pd.Timestamp(e.start).day_name() == "Thursday" for e in events)
    assert events[0].start.startswith("2020-03-26")
    assert events[-1].start.startswith("2020-05-28")


def test_build_calendar_features_requires_utc_index():
    naive_index = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=4, freq="30min"))
    try:
        build_calendar_features(naive_index)
        assert False, "expected ValueError for non-UTC index"
    except ValueError:
        pass


def test_euro_2020_final_is_large_tier():
    index = pd.date_range("2021-07-11T18:00Z", "2021-07-11T22:00Z", freq="30min", tz="UTC")
    feats = build_calendar_features(index)
    kickoff_utc = pd.Timestamp("2021-07-11T20:00:00Z")  # 8pm BST kickoff = 19:00Z, mid-match by 20:00Z
    assert feats.loc[kickoff_utc, "event_tier"] == "large"
    assert "Euro 2020 Final" in feats.loc[kickoff_utc, "event_name"]


def test_ordinary_day_has_no_events():
    index = pd.date_range("2024-03-05T12:00Z", periods=2, freq="30min", tz="UTC")
    feats = build_calendar_features(index)
    assert (feats["event_tier"] == "none").all()
    assert not feats["is_bank_holiday"].any()
    assert (feats["lockdown_level"] == "none").all()


def test_overlapping_events_keep_the_higher_tier():
    # Rugby World Cup 2023 (small) overlaps Cricket World Cup 2023 (small);
    # neither should ever be downgraded/upgraded incorrectly by the other.
    index = pd.date_range("2023-10-20T12:00Z", periods=2, freq="30min", tz="UTC")
    feats = build_calendar_features(index)
    assert (feats["event_tier"] == "small").all()


def test_solar_eclipse_profile_is_bounded_and_zero_outside_window():
    index = pd.date_range("2022-10-25T08:00Z", "2022-10-25T14:00Z", freq="30min", tz="UTC")
    feats = build_calendar_features(index)
    assert feats["solar_eclipse_pct"].min() == 0.0
    assert 0 < feats["solar_eclipse_pct"].max() <= 28.0
    # well before the eclipse window (starts 08:58Z)
    assert feats.loc[pd.Timestamp("2022-10-25T08:00Z"), "solar_eclipse_pct"] == 0.0


def test_all_events_have_start_before_end():
    for event in ALL_EVENTS:
        assert pd.Timestamp(event.start) <= pd.Timestamp(event.end)
