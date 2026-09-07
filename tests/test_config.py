from itertools import pairwise

import pandas as pd

from edf.config import HORIZONS, PERIODS_PER_DAY, PERIODS_PER_WEEK, SPLITS, TEST, TRAIN, VALIDATION


def test_splits_are_contiguous_and_non_overlapping():
    ordered = [TRAIN, VALIDATION, TEST]
    for (_, end), (start, _) in pairwise(ordered):
        assert pd.Timestamp(start) == pd.Timestamp(end) + pd.Timedelta(days=1)


def test_splits_dict_matches_named_constants():
    assert SPLITS == {"train": TRAIN, "validation": VALIDATION, "test": TEST}


def test_horizons_are_positive_and_increasing():
    periods = list(HORIZONS.values())
    assert periods == sorted(periods)
    assert all(p > 0 for p in periods)


def test_horizons_bound_by_periods_per_week():
    assert HORIZONS["1d"] == PERIODS_PER_DAY
    assert HORIZONS["7d"] == PERIODS_PER_WEEK
