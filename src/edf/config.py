"""Project-wide constants that must stay fixed once decided.

The train/validation/test date ranges are decided once in Week 1 and must not
change later — see PLAN.md's time-series correctness rules. Every walk-forward
split (Week 2 onward) is built from these boundaries so no later week can
accidentally evaluate against `TEST` data "just to check".

**Redefined 2026-09-08 (shifted forward one year), see PLAN.md's split-redefinition note.**
Original (Week 1-Follow-up 5): `TRAIN`=2020-2023, `VALIDATION`=2024, `TEST`=2025 (never
evaluated against). By 2026-09-08, 2025 had fully elapsed and was sitting in `data/`
complete and untouched -- promoting it into `VALIDATION` gives every future week more
data and a genuinely fresh evaluation year, without violating the "never peek at TEST"
rule: nothing was ever evaluated against 2025 under the old definition, so relabeling
it `VALIDATION` isn't peeking, it's simply choosing to start using a year that was
sitting there unused. **Every notebook/PLAN.md section written before this date reports
results against the *old* `VALIDATION` (2024) -- that data doesn't disappear, it's now
inside `TRAIN`, but those historical numbers were not recomputed under this new split.**
New `TEST` (2026) is **provisional and NOT YET VERIFIED** -- NESO's own metadata for the
2026 resource flags a known issue ("missing Scottish transfer data") and the CSV
filename/URL pattern changed from prior years (`demanddataupdate_2026.csv`, not
`demanddata_2026.csv`); `edf.data.download` has not been updated for it yet, and the
canonical table does not yet contain any real 2026 rows. This range exists now only to
satisfy `tests/test_config.py`'s contiguous-splits invariant -- **do not evaluate
anything against `TEST` until the 2026 data source has been verified**, the same
live-verification standard every other data source in this project was held to
(see `edf.data.weather`'s `NOWCAST_ARCHIVE_START`/`DAY_AHEAD_ARCHIVE_START` for the
precedent).

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

TRAIN = ("2020-01-01", "2024-12-31")
VALIDATION = ("2025-01-01", "2025-12-31")
TEST = ("2026-01-01", "2026-12-31")  # provisional, unverified -- see module docstring

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
