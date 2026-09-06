"""NESO's own historical day-ahead demand forecast — a genuine operational
benchmark: "does our model beat what NESO actually forecast and used, in
production, one day ahead?"

Source: NESO Data Portal, "Day Ahead Demand Forecast" dataset, "Historic Day
Ahead Demand Forecasts" resource (`archive_1dayahead.csv`). Confirmed via a
real download (2026-09-06) that this contains ~12 "cardinal points" per
target day (overnight minimum, morning/evening peaks, etc.) rather than a
full half-hourly series, each published twice daily (~09:00 and ~12:00) for
the next day — `DAYSAHEAD` is always 1 in the archive, no other lead times
are retained historically. Coarser than half-hourly, but it's a real
operational forecast NESO used, not an ML upper-bound estimate.

Of the two same-day publications, this module keeps the *earliest* one per
(target date, cardinal point) — the more conservative, furthest-ahead-of-time
forecast — as "the" day-ahead value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

CSV_URL = (
    "https://api.neso.energy/dataset/8fbc8a09-06af-4c90-886f-d3025d38a349"
    "/resource/9847e7bb-986e-49be-8138-717b25933fbb/download/archive_1dayahead.csv"
)


def download_csv(raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "archive_1dayahead.csv"
    if dest.exists() and not force:
        return dest
    response = requests.get(CSV_URL, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def cardinal_point_to_utc(target_date: pd.Series, cp_st_time: pd.Series) -> pd.Series:
    """TARGETDATE + CP_ST_TIME (local Europe/London "HMM"/"HHMM", "2400" = next-day
    midnight) -> UTC timestamp. DST-ambiguous/nonexistent local times become NaT.
    """
    time_str = cp_st_time.astype(str).str.zfill(4)
    hour = time_str.str[:2].astype(int)
    minute = time_str.str[2:].astype(int)
    rolls_to_next_day = hour == 24
    hour = hour.where(~rolls_to_next_day, 0)

    dates = pd.to_datetime(target_date) + pd.to_timedelta(
        rolls_to_next_day.astype(int), unit="D"
    )
    local_naive = dates + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    local = local_naive.dt.tz_localize(
        "Europe/London", nonexistent="shift_forward", ambiguous="NaT"
    )
    return local.dt.tz_convert("UTC")


def load_and_clean(csv_path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"CP_ST_TIME": str})
    target_date = pd.to_datetime(df["TARGETDATE"])
    df = df[(target_date >= start) & (target_date <= end)].copy()

    # Keep the earliest (furthest-ahead) of the ~2 same-day publications per
    # (target date, cardinal point).
    df = df.sort_values("FORECAST_TIMESTAMP")
    df = df.drop_duplicates(subset=["TARGETDATE", "CARDINALPOINT"], keep="first")

    df["timestamp"] = cardinal_point_to_utc(df["TARGETDATE"], df["CP_ST_TIME"])
    n_before = len(df)
    df = df.dropna(subset=["timestamp"])
    dropped = n_before - len(df)
    if dropped:
        print(f"Dropped {dropped} DST-ambiguous cardinal-point timestamps")

    return (
        df[["timestamp", "TARGETDATE", "CARDINALPOINT", "CP_TYPE", "FORECASTDEMAND", "FORECAST_TIMESTAMP"]]
        .rename(columns={"FORECASTDEMAND": "forecast_demand_mw"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def build_parquet(
    raw_dir: Path,
    out_path: Path,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    force_download: bool = False,
) -> Path:
    csv_path = download_csv(raw_dir, force=force_download)
    df = load_and_clean(csv_path, start, end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/neso_day_ahead_demand_forecast.parquet")
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    out_path = build_parquet(args.raw_dir, args.out, args.start, args.end, args.force_download)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
