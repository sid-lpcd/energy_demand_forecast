"""Elexon BMRS National Demand Forecast (`NDF`) — free, keyless, genuinely day-ahead.

Unlike `src/edf/data/demand_forecast_benchmark.py` (NESO's own portal archive,
~12 coarse "cardinal points"/day), this is Elexon's underlying feed: a
half-hourly national demand forecast, republished roughly every 30 minutes
as a rolling horizon gets closer to delivery — verified live (2026-09-06)
against the API, archive starts 2021-06-14 and is still live today.

**Day-ahead selection rule** (`fetch_day_ahead_ndf`): NDF is a *rolling*
forecast — the same settlement period gets re-published repeatedly as
delivery approaches, each revision more accurate than the last. To get a
value that would genuinely have been available "the day before" (not an
intraday revision benefiting from same-day information), this keeps only
revisions published on the calendar day immediately before the target
settlement date, and of those, the *latest* one — i.e. the most-refined
version issued near the end of that prior day (empirically ~23:45 UTC),
which is what a real forecasting operation would actually have in hand at
that point. This is unlike `demand_forecast_benchmark.py`'s "earliest of
~2 publications" rule (that dataset only retains 1-2 snapshots/day, so
"earliest" was the only way to guarantee advance notice there); here we
have much denser data, so "latest publish still on the day before" is the
more realistic and more precise equivalent.

Not leaky: this is a genuine advance forecast, not a same-period actual —
unlike `src/edf/data/fuel_mix.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF"
MAX_WINDOW_DAYS = 1  # the API's own hard limit on publishDateTimeFrom/To span
ARCHIVE_START = pd.Timestamp("2021-06-14", tz="UTC")  # verified empirically; nothing before this


def fetch_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One API call (<= 1 day of *publish* time). Long format, one row per
    (settlement period, publish revision): timestamp, settlement_date,
    publish_time, demand. `boundary` is filtered to "N" (national) — the
    only other value seen is a much smaller-magnitude regional series, not
    the national demand forecast this module is after.
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
    if result.empty:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "settlement_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "publish_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "demand": pd.Series(dtype="float64"),
            }
        )
    result = result[result["boundary"] == "N"]
    result["timestamp"] = pd.to_datetime(result["startTime"], utc=True)
    result["publish_time"] = pd.to_datetime(result["publishTime"], utc=True)
    result["settlement_date"] = pd.to_datetime(result["settlementDate"], utc=True)
    return result[["timestamp", "settlement_date", "publish_time", "demand"]]


def fetch_day_ahead_ndf(
    start: str,
    end: str,
    fetch: Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame] = fetch_window,
) -> pd.Series:
    """Day-ahead NDF demand (MW), indexed by UTC settlement-period timestamp.

    Fetches publishes from one calendar day before `start` through `end`
    (chunked into `MAX_WINDOW_DAYS`-day windows — the API's own limit) so
    every target settlement period has its "day before" publishes covered,
    then applies the day-ahead selection rule described in the module
    docstring. `fetch` is injectable for testing without a live network call.
    """
    start_ts = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=1)
    end_ts = pd.Timestamp(end, tz="UTC")

    frames = []
    window_start = start_ts
    while window_start <= end_ts:
        window_end = min(
            window_start + pd.Timedelta(days=MAX_WINDOW_DAYS) - pd.Timedelta(seconds=1), end_ts
        )
        frames.append(fetch(window_start, window_end))
        window_start = window_end + pd.Timedelta(seconds=1)

    long_df = pd.concat(frames, ignore_index=True)
    if long_df.empty:
        return pd.Series(name="ndf_day_ahead", dtype=float)
    long_df["publish_time"] = pd.to_datetime(long_df["publish_time"], utc=True)
    long_df["settlement_date"] = pd.to_datetime(long_df["settlement_date"], utc=True)

    is_day_ahead = long_df["publish_time"].dt.floor("D") == long_df["settlement_date"] - pd.Timedelta(
        days=1
    )
    day_ahead = long_df[is_day_ahead].sort_values("publish_time")
    day_ahead = day_ahead.drop_duplicates(subset=["timestamp"], keep="last")
    return day_ahead.set_index("timestamp")["demand"].rename("ndf_day_ahead").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-06-14")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--out", type=Path, default=Path("data/raw/elexon_ndf_day_ahead.parquet"))
    args = parser.parse_args()

    series = fetch_day_ahead_ndf(args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(args.out)
    print(f"Wrote {args.out} ({len(series)} rows, {series.index.min()} to {series.index.max()})")


if __name__ == "__main__":
    main()
