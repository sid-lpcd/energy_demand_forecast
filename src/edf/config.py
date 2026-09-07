"""Project-wide constants that must stay fixed once decided.

The train/validation/test date ranges are decided once in Week 1 and must not
change later — see PLAN.md's time-series correctness rules. Every walk-forward
split (Week 2 onward) is built from these boundaries so no later week can
accidentally evaluate against `TEST` data "just to check".

COVID's acute disruption (2020-2021) falls entirely inside `TRAIN`, so
reported validation/test accuracy is not distorted by it (see PLAN.md Week 6
for how the training-time COVID period is still put to use).

`HORIZONS` fixes the forecast lead times evaluated everywhere in the project
(Week 2's baseline scenario analysis onward), in half-hourly settlement
periods. This matters because it's not just a knob: a model or baseline
scored at a short horizon (e.g. 30 minutes ahead) is solving an easier
problem than one scored a day or a week ahead, since much more recent data is
available at issue time — see PLAN.md Week 2's finding that a naive 30-minute
persistence forecast trivially beats the seasonal baselines. `1d` is the
horizon that's actually comparable to NESO's own day-ahead forecast benchmark
(`edf.data.demand_forecast_benchmark`).
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

PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = PERIODS_PER_DAY * 7

HORIZONS: dict[str, int] = {
    "30min": 1,
    "1h": 2,
    "1d": PERIODS_PER_DAY,
    "7d": PERIODS_PER_WEEK,
}
