"""Elexon BMRS generation-by-fuel-type data (`FUELHH`) — free, keyless, half-hourly.

Fetched to test one specific hypothesis from `notebooks/16`'s NESO-benchmark
comparison: NESO's day-ahead forecast beats ours almost entirely at the
overnight demand trough (+31.4%), plausibly because NESO has administrative
visibility into large, schedulable loads that no public weather/calendar
dataset carries. `PS` (pumped storage — plants like Dinorwig, dispatched
specifically to manage demand troughs/peaks) is the fuel type most directly
tied to *deliberate* trough/peak management, so its dispatch pattern is a
plausible (imperfect, indirect) proxy for exactly that kind of foreknowledge.

**This is same-period actual generation, not a forecast** — same caveat as
`wind`/`solar`/`interconnector` throughout this project (see PLAN.md's Week 1
leakage warning and Week 4b/notebooks/11). Used here only as a "leaky upper
bound" test of whether there's genuine signal worth pursuing at all, exactly
the same honest-ablation-first pattern Week 4b used for wind/solar — not
presented as a deployable feature.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH"
MAX_WINDOW_DAYS = 7  # the API's own hard limit on publishDateTimeFrom/To span


def fetch_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One API call (<= 7 days). Long format: timestamp, fuel_type, generation_mw.

    `startTime` in the response is already the settlement period's UTC start
    — verified empirically against a known settlement period (2026-09-07) —
    so it aligns directly with this project's canonical timestamp convention,
    no settlement-period arithmetic needed here.
    """
    resp = requests.get(
        API_URL,
        params={
            "publishDateTimeFrom": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "publishDateTimeTo": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        timeout=60,
    )
    resp.raise_for_status()
    rows = resp.json()["data"]
    result = pd.DataFrame(rows)
    result["timestamp"] = pd.to_datetime(result["startTime"], utc=True)
    return result.rename(columns={"fuelType": "fuel_type", "generation": "generation_mw"})[
        ["timestamp", "fuel_type", "generation_mw"]
    ]


def fetch_fuel_mix(
    start: str,
    end: str,
    fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame] = fetch_window,
) -> pd.DataFrame:
    """Half-hourly generation by fuel type (MW), wide format: one column per
    fuel type, indexed by UTC timestamp.

    Chunks the request into <= 7-day windows (the API's own limit,
    `MAX_WINDOW_DAYS`) and concatenates — `fetch` is injectable so this
    chunking/pivoting logic can be tested without a live network call.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    frames = []
    window_start = start_ts
    while window_start <= end_ts:
        window_end = min(
            window_start + pd.Timedelta(days=MAX_WINDOW_DAYS) - pd.Timedelta(seconds=1), end_ts
        )
        frames.append(fetch(window_start, window_end))
        window_start = window_end + pd.Timedelta(seconds=1)

    long_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp", "fuel_type"])
    wide = long_df.pivot(index="timestamp", columns="fuel_type", values="generation_mw")
    wide.columns.name = None
    return wide.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out", type=Path, default=Path("data/raw/elexon_fuel_mix.parquet"))
    args = parser.parse_args()

    df = fetch_fuel_mix(args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"Wrote {args.out} ({len(df)} rows, {df.index.min()} to {df.index.max()})")


if __name__ == "__main__":
    main()
