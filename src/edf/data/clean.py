"""Build the canonical, cleaned half-hourly dataset used by every later week.

Combines the raw NESO demand data (`edf.data.download`) with the calendar/
event features (`edf.data.calendar_events`) into one table on the canonical
columns decided in Week 1 (see PLAN.md "Canonical column decisions"):

- `demand` = ND (National Demand), not TSD — TSD adds back discretionary,
  price-driven pump-storage pumping, which the model has no features to
  explain.
- `wind`/`solar` = embedded wind/solar generation, used as model *features*
  (never subtracted from `demand` — see PLAN.md's embedded-generation-netting
  note; ND already has this effect invisibly baked in).
- `interconnector` = sum of all `*_FLOW` columns (net GB import, +import/
  -export).

The raw exploration notebook (`notebooks/01_explore_raw_data.ipynb`) already
checked the full 2020-2025 raw series for missing settlement periods and
negative demand and found none, so `check_no_gaps`/`check_no_negative_demand`
here exist as regression guards against that finding changing (e.g. a future
re-download, or extending the date range), not as data-repair logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from edf.data.calendar_events import build_calendar_features

INTERCONNECTOR_FLOW_COLUMNS = [
    "IFA_FLOW",
    "IFA2_FLOW",
    "BRITNED_FLOW",
    "MOYLE_FLOW",
    "EAST_WEST_FLOW",
    "NEMO_FLOW",
    "NSL_FLOW",
    "ELECLINK_FLOW",
    "VIKING_FLOW",
    "GREENLINK_FLOW",
]


def check_no_gaps(timestamps: pd.Series) -> None:
    """Raise if the (already-UTC) timestamps aren't uniformly 30 minutes apart.

    Settlement periods are always exactly 30 minutes apart in absolute
    (UTC) time — clock-change days only change how many periods make up
    that local day (46/50 instead of 48), not the UTC spacing between
    consecutive periods. So a plain "every diff is 30 minutes" check
    correctly covers clock-change days without special-casing them.
    """
    diffs = pd.Series(timestamps).diff().dropna()
    bad = diffs[diffs != pd.Timedelta(minutes=30)]
    if not bad.empty:
        raise ValueError(
            f"Found {len(bad)} gap(s)/duplicate(s)/overlap(s) in timestamps, "
            f"e.g. at index {bad.index[0]}"
        )


def check_no_negative_demand(demand: pd.Series) -> None:
    negative = demand < 0
    if negative.any():
        raise ValueError(f"Found {negative.sum()} negative demand value(s)")


def build_canonical_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical modelling table from the raw NESO demand frame.

    `raw` is the frame produced by `edf.data.download.build_raw_parquet`
    (must have a UTC tz-aware `timestamp` column plus the NESO source
    columns). Returns a frame indexed by `timestamp` with `demand`, `wind`,
    `solar`, `interconnector`, and the calendar/event/lockdown columns from
    `build_calendar_features`.
    """
    check_no_gaps(raw["timestamp"])
    check_no_negative_demand(raw["ND"])

    index = pd.DatetimeIndex(raw["timestamp"], name="timestamp")
    core = pd.DataFrame(
        {
            "demand": raw["ND"].to_numpy(),
            "wind": raw["EMBEDDED_WIND_GENERATION"].to_numpy(),
            "solar": raw["EMBEDDED_SOLAR_GENERATION"].to_numpy(),
            "interconnector": raw[INTERCONNECTOR_FLOW_COLUMNS].sum(axis=1).to_numpy(),
        },
        index=index,
    )
    calendar = build_calendar_features(index)
    return core.join(calendar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/historic_demand_2020_2025.parquet")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/gb_energy_2020_2025.parquet")
    )
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw)
    canonical = build_canonical_table(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(args.out)
    print(f"Wrote {args.out} ({len(canonical)} rows, {canonical.index.min()} to {canonical.index.max()})")


if __name__ == "__main__":
    main()
