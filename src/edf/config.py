"""Project-wide constants that must stay fixed once decided.

The train/validation/test date ranges are decided once in Week 1 and must not
change later — see PLAN.md's time-series correctness rules. Every walk-forward
split (Week 2 onward) is built from these boundaries so no later week can
accidentally evaluate against `TEST` data "just to check".

COVID's acute disruption (2020-2021) falls entirely inside `TRAIN`, so
reported validation/test accuracy is not distorted by it (see PLAN.md Week 6
for how the training-time COVID period is still put to use).
"""

from __future__ import annotations

TRAIN = ("2020-01-01", "2023-12-31")
VALIDATION = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2025-12-31")

SPLITS: dict[str, tuple[str, str]] = {
    "train": TRAIN,
    "validation": VALIDATION,
    "test": TEST,
}
