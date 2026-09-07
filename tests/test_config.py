from itertools import pairwise

import pandas as pd

from edf.config import SPLITS, TEST, TRAIN, VALIDATION


def test_splits_are_contiguous_and_non_overlapping():
    ordered = [TRAIN, VALIDATION, TEST]
    for (_, end), (start, _) in pairwise(ordered):
        assert pd.Timestamp(start) == pd.Timestamp(end) + pd.Timedelta(days=1)


def test_splits_dict_matches_named_constants():
    assert SPLITS == {"train": TRAIN, "validation": VALIDATION, "test": TEST}
