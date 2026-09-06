"""Download NESO Historic Demand Data and store it as a single raw parquet file.

Source: NESO Data Portal, "Historic Demand Data" dataset
https://www.neso.energy/data-portal/historic-demand-data

Each yearly CSV is half-hourly (SETTLEMENT_PERIOD 1-48, or 46/50 on UK clock-change
days) and includes national demand (ND, TSD), embedded wind/solar generation, and
interconnector flows. Columns are saved as-is (raw, unmodified values) with one
added `timestamp` column (UTC) — see `settlement_periods_to_utc` for why that
conversion isn't a straight "add N half hours".

This module only downloads and concatenates; cleaning (missing periods, outliers,
picking the canonical demand/wind/solar/interconnector columns) is a separate step.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import requests

DATASET_ID = "8f2fe0af-871c-488d-8bad-960426f24601"
DOWNLOAD_URL_TEMPLATE = (
    "https://api.neso.energy/dataset/{dataset_id}/resource/{resource_id}"
    "/download/demanddata_{year}.csv"
)

# Per-year resource IDs, from https://www.neso.energy/data-portal/historic-demand-data
RESOURCE_IDS: dict[int, str] = {
    2020: "33ba6857-2a55-479f-9308-e5c4c53d4381",
    2021: "18c69c42-f20d-46f0-84e9-e279045befc6",
    2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
}

DEFAULT_YEARS: tuple[int, ...] = tuple(sorted(RESOURCE_IDS))


def csv_url(year: int) -> str:
    return DOWNLOAD_URL_TEMPLATE.format(
        dataset_id=DATASET_ID, resource_id=RESOURCE_IDS[year], year=year
    )


def download_year_csv(year: int, raw_dir: Path, force: bool = False) -> Path:
    """Fetch one year's CSV into raw_dir, unmodified. Skips if already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"demanddata_{year}.csv"
    if dest.exists() and not force:
        return dest
    response = requests.get(csv_url(year), timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def settlement_periods_to_utc(dates: pd.Series, periods: pd.Series) -> pd.Series:
    """Convert SETTLEMENT_DATE + SETTLEMENT_PERIOD to a tz-aware UTC timestamp.

    Settlement periods are consecutive half hours from *local* (Europe/London)
    midnight — 46 periods on the March clock-change day, 50 on the October one.
    UK clock changes happen at 01:00/02:00, so local midnight itself is never
    ambiguous or nonexistent; localizing midnight and then adding a fixed-size
    UTC offset per period handles both short and long days correctly without
    special-casing them.

    SETTLEMENT_DATE's own text format is inconsistent across NESO's yearly
    files (seen: "01-JAN-2020", "01-Jan-23", "2025-01-01"), so dates are parsed
    with `format="mixed"` rather than one fixed strptime format.
    """
    local_midnight = pd.to_datetime(dates, format="mixed", dayfirst=True).dt.tz_localize(
        "Europe/London"
    )
    utc_midnight = local_midnight.dt.tz_convert("UTC")
    offsets = pd.to_timedelta((periods.astype("int64") - 1) * 30, unit="min")
    return utc_midnight + offsets


def load_year(year: int, raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"demanddata_{year}.csv")
    df.insert(0, "timestamp", settlement_periods_to_utc(df["SETTLEMENT_DATE"], df["SETTLEMENT_PERIOD"]))
    return df


def build_raw_parquet(
    years: Sequence[int],
    raw_dir: Path,
    out_path: Path,
    force_download: bool = False,
) -> Path:
    frames = []
    for year in years:
        download_year_csv(year, raw_dir, force=force_download)
        frames.append(load_year(year, raw_dir))
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/historic_demand_2020_2025.parquet")
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    out_path = build_raw_parquet(args.years, args.raw_dir, args.out, force_download=args.force_download)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
