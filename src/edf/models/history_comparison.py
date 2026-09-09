"""Build the daily actual/NDF/our-forecast/blended comparison series shown on `/predictions`.

This is a batch/offline builder (`uv run python -m edf.models.history_comparison`), not something
the deployed app runs live: the canonical historical data it reads is gitignored and never shipped
to Render. Its output -- a small daily CSV -- is what actually ships (see `DEFAULT_OUT_PATH`),
committed the same way `models_registry/` is. Re-run this script to refresh the chart; the live app
just reads whatever CSV is on disk.

**Model: reuses `notebooks/21`'s exact methodology**, the project's published headline result --
a `1d`-horizon point model trained *only* on `TRAIN` (2020-2024), so its predictions are genuinely
out-of-sample for every date this module displays (all of `VALIDATION`=2025 and `TEST`=2026, both
untouched by training). This is deliberately a different, separate model object from
`models_registry/`'s live-serving model, which is trained through 2025-12-31 (see its own
docstring) and would be in-sample/leaked for any 2025 date shown here.

**Weather: ERA5 hindsight (`open_meteo_historical`), not a forecast** -- same as `notebooks/21`.
Per this project's stated rule (`CLAUDE.md`, `PLAN.md`): this makes the "our forecast" line an
accuracy *upper bound*, not deployable production accuracy. The app must show this caveat next to
the chart, not just here.

**2026 data**: `edf.config.TEST` (2026) has no real rows in the canonical processed table yet (see
its docstring) -- NESO's 2026 demand resource needed a filename-pattern fix and carries a
NESO-acknowledged "missing Scottish transfer data" advisory. This module live-fetches a 2026 tail
from three independently re-fetchable *historical* archives (not live-only snapshots): NESO's
`Historic Demand Data 2026` resource plus the rolling `Demand Data Update` feed for demand actuals
(verified byte-identical on their overlap, 2026-09-09), Elexon's NDF API
(`edf.data.ndf.fetch_day_ahead_ndf`) for NDF, and Open-Meteo's archive API for ERA5. The Scottish-
transfer advisory is still open upstream; spot-checked (2026-09-09) `ND - ENGLAND_WALES_DEMAND`
across all available 2026 rows and found no dropout/discontinuity, but this is not a substitute for
NESO actually resolving it -- surface the caveat, don't silently trust it.

**Rightmost edge**: ERA5 is hindsight, so there's no weather (and therefore no prediction) for a
date that hasn't happened yet. The "our forecast"/NDF lines' latest point is *today's* demand, as
predicted one day ahead (issued yesterday) -- not tomorrow's. `actual_demand` simply has no value
past whatever NESO has actually published, which typically trails a little further behind that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from edf import config
from edf.data.calendar_events import build_calendar_features
from edf.data.live_demand import fetch_live_demand_actuals, parse_live_demand_csv
from edf.data.ndf import fetch_day_ahead_ndf
from edf.data.weather import fetch_open_meteo_historical, population_weighted_gb_series
from edf.features.buckets import build_day_type_buckets, is_christmas_period
from edf.features.demand import CALENDAR_COLUMNS, build_feature_table
from edf.features.generation import capacity_factor
from edf.features.weather import (
    build_weather_feature_table,
    build_weather_feature_table_from_frame,
    cumulative_degree,
)
from edf.models.combination import HEADLINE_COMBINATION_WEIGHT, combine_forecasts
from edf.models.forecast import train_lightgbm

DEFAULT_CANONICAL_PATH = Path("data/processed/gb_energy_2020_2025.parquet")
DEFAULT_WEATHER_RAW_DIR = Path("data/raw/weather")
DEFAULT_NDF_PATH = Path("data/raw/elexon_ndf_day_ahead.parquet")
DEFAULT_OUT_PATH = Path("reports/ndf_comparison_last_year.csv")

# "Historic Demand Data 2026" resource -- see edf.data.live_demand's docstring for the sibling
# "Demand Data Update" (rolling, ~45 days) resource this is merged with.
HISTORIC_DEMAND_2026_URL = (
    "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601"
    "/resource/8a4a771c-3929-4e56-93ad-cdf13219dea5/download/demanddataupdate_2026.csv"
)

HORIZON_PERIODS = config.HORIZONS["1d"]
# Adopted recipe (notebooks/14, notebooks/20, notebooks/21) -- see registry.py's module docstring
# for the same constants used by the live-serving model's 1d recipe.
POINT_PARAMS = {"num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.05}
POINT_N_ESTIMATORS = 689
CUMULATIVE_DEGREE_WINDOW_PERIODS = 48  # 1 day
EXTREME_SAMPLE_WEIGHT = 3.0
DISPLAY_WINDOW_DAYS = 365

SCOTTISH_TRANSFER_ADVISORY = (
    "NESO's 2026 demand data carries an open advisory (\"missing Scottish transfer data\"); "
    "spot-checked 2026-09-09 with no dropout found, but treat 2026 figures with that caveat."
)
WEATHER_UPPER_BOUND_NOTE = (
    "\"Our forecast\" uses actual historical weather (ERA5), not a real forecast -- an accuracy "
    "upper bound, not deployable production accuracy."
)


def fetch_2026_demand_tail() -> pd.DataFrame:
    """Real (non-forecast) 2026 demand actuals: the yearly archive resource (deeper history) merged
    with the rolling live feed (most recent ~45 days) -- verified byte-identical on their overlap.
    """
    resp = requests.get(HISTORIC_DEMAND_2026_URL, timeout=60)
    resp.raise_for_status()
    archive = parse_live_demand_csv(resp.text)
    live = fetch_live_demand_actuals()
    return merge_demand_sources(archive, live)


def merge_demand_sources(archive: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Union two `timestamp`/`demand` frames, preferring `live`'s value on any overlapping
    timestamp (it's the fresher of the two on any date both cover)."""
    combined = pd.concat([archive, live])
    return (
        combined.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_2026_weather_tail(index: pd.DatetimeIndex) -> pd.DataFrame:
    """ERA5 hindsight weather feature table for `index`'s 2026 dates, live-fetched (no on-disk
    archive covers 2026 yet)."""
    start = index.min().strftime("%Y-%m-%d")
    end = index.max().strftime("%Y-%m-%d")
    hourly = population_weighted_gb_series(fetch_open_meteo_historical, start, end)
    return build_weather_feature_table_from_frame(hourly, index)


def sample_weight_for_training(canonical_df: pd.DataFrame, canonical_weather: pd.DataFrame) -> pd.Series:
    """Extreme-bucket sample weight (notebooks/14/20/21's adopted recipe). Needs `wind`/
    `wind_capacity`, only ever available for canonical (pre-2026) rows -- training never touches
    the 2026 tail, so this is fine restricted to `canonical_df`'s own index."""
    wind_cf = capacity_factor(canonical_df["wind"], canonical_df["wind_capacity"])
    day_buckets = build_day_type_buckets(canonical_weather["temperature_c"], wind_cf)
    problem_mask = day_buckets["is_hot"] | day_buckets["is_high_wind"] | day_buckets["is_christmas"]
    weight = pd.Series(1.0, index=canonical_df.index)
    weight[problem_mask] = EXTREME_SAMPLE_WEIGHT
    return weight


def train_model_final(X_base_full: pd.DataFrame, y_full: pd.Series, sample_weight: pd.Series) -> object:
    """Train the notebooks/21 point model on `TRAIN` only, sliced from `X_base_full` (built over
    the *combined* canonical+2026-tail range so train- and predict-time categorical feature
    encodings come from a single `deterministic_features` call and can't silently diverge)."""
    train_start, train_end = config.TRAIN
    X_train = X_base_full.loc[train_start:train_end]
    y_train = y_full.loc[X_train.index]
    return train_lightgbm(
        X_train,
        y_train,
        sample_weight=sample_weight.loc[X_train.index],
        n_estimators=POINT_N_ESTIMATORS,
        **POINT_PARAMS,
    )


def aggregate_daily_comparison(
    actual: pd.Series,
    ndf: pd.Series,
    our_forecast: pd.Series,
    weight_a: float = HEADLINE_COMBINATION_WEIGHT,
    window_end: pd.Timestamp | None = None,
    window_days: int = DISPLAY_WINDOW_DAYS,
) -> pd.DataFrame:
    """Blend, daily-aggregate (mean), and trim to a trailing `window_days`-day window ending
    `window_end` (defaults to `our_forecast`'s own latest timestamp).

    `actual` is allowed to run out before `ndf`/`our_forecast` do (it lags real time) -- rows past
    its coverage simply carry a NaN `actual_demand_mw` rather than being dropped, so the two
    forecast lines aren't truncated to match a laggier actuals series.
    """
    blended = combine_forecasts(our_forecast, ndf, weight_a=weight_a)

    if window_end is None:
        window_end = our_forecast.index.max()
    window_start = window_end - pd.Timedelta(days=window_days)

    frame = pd.DataFrame(
        {
            "actual_demand_mw": actual,
            "ndf_forecast_mw": ndf,
            "our_forecast_mw": our_forecast,
            "blended_forecast_mw": blended,
        }
    )
    frame = frame.loc[(frame.index > window_start) & (frame.index <= window_end)]

    daily = frame.groupby(frame.index.tz_convert("UTC").date).mean()
    daily.index.name = "date"
    return daily.reset_index()


def build_comparison_series(
    canonical_path: Path = DEFAULT_CANONICAL_PATH,
    weather_raw_dir: Path = DEFAULT_WEATHER_RAW_DIR,
    ndf_path: Path = DEFAULT_NDF_PATH,
    window_days: int = DISPLAY_WINDOW_DAYS,
) -> pd.DataFrame:
    """End-to-end: assemble a single continuous 2020-through-today frame (canonical + a live-
    fetched 2026 tail), train on `TRAIN` only, predict across the whole range, blend with NDF, and
    return the trimmed daily comparison table (not written to disk -- see `main`)."""
    canonical_df = pd.read_parquet(canonical_path)
    canonical_weather = build_weather_feature_table(
        "open_meteo_historical", canonical_df.index, raw_dir=weather_raw_dir
    )

    tail_demand = fetch_2026_demand_tail()
    tail_index = pd.date_range(
        pd.Timestamp("2026-01-01", tz="UTC"), tail_demand["timestamp"].max(), freq="30min"
    )
    tail_weather = fetch_2026_weather_tail(tail_index)
    tail_calendar = build_calendar_features(tail_index)

    demand_full = pd.concat(
        [canonical_df["demand"], tail_demand.set_index("timestamp")["demand"]]
    ).sort_index()
    demand_full = demand_full[~demand_full.index.duplicated(keep="last")]

    weather_full = pd.concat([canonical_weather, tail_weather]).sort_index()
    weather_full = weather_full[~weather_full.index.duplicated(keep="last")]

    calendar_cols = list(CALENDAR_COLUMNS)
    df_full = pd.concat([canonical_df[calendar_cols], tail_calendar[calendar_cols]]).sort_index()
    df_full = df_full[~df_full.index.duplicated(keep="last")]
    df_full["demand"] = demand_full.reindex(df_full.index)

    era5_full = weather_full.copy()
    era5_full["is_christmas"] = is_christmas_period(df_full.index).astype(int)
    era5_full["heating_degree_1d_cum"] = cumulative_degree(
        era5_full["heating_degree"], CUMULATIVE_DEGREE_WINDOW_PERIODS
    )
    era5_full["cooling_degree_1d_cum"] = cumulative_degree(
        era5_full["cooling_degree"], CUMULATIVE_DEGREE_WINDOW_PERIODS
    )
    X_base_full, y_full = build_feature_table(df_full, HORIZON_PERIODS, weather=era5_full)

    sample_weight = sample_weight_for_training(canonical_df, canonical_weather)
    model = train_model_final(X_base_full, y_full, sample_weight)

    our_forecast = pd.Series(model.predict(X_base_full), index=X_base_full.index)

    ndf_canonical = pd.read_parquet(ndf_path)["ndf_day_ahead"]
    ndf_tail = fetch_day_ahead_ndf("2026-01-01", tail_demand["timestamp"].max().strftime("%Y-%m-%d"))
    ndf_full = pd.concat([ndf_canonical, ndf_tail])
    ndf_full = ndf_full[~ndf_full.index.duplicated(keep="last")].reindex(our_forecast.index)

    return aggregate_daily_comparison(
        actual=y_full.reindex(our_forecast.index),
        ndf=ndf_full,
        our_forecast=our_forecast,
        window_days=window_days,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--window-days", type=int, default=DISPLAY_WINDOW_DAYS)
    args = parser.parse_args()

    daily = build_comparison_series(window_days=args.window_days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(daily)} days, {daily['date'].min()} .. {daily['date'].max()})")


if __name__ == "__main__":
    main()
