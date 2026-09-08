"""Live feature assembly + TTL cache for the live-inference app.

Builds one feature row per horizon from freshly-fetched live data, reusing exactly the same
feature-building functions `edf.models.registry.train_final_models` used at training time
(`edf.features.demand.build_feature_table`, `edf.features.weather.build_weather_feature_table_from_frame`,
`edf.data.calendar_events.build_calendar_features`) -- the live pipeline never reimplements
lag/rolling/calendar arithmetic itself, so a live/train mismatch would have to come from genuinely
different *inputs*, not a second, drifted copy of the feature logic.

Upstream data (`fetch_live_inputs`) is refetched only when the half-hourly settlement boundary
changes (`settlement_floor`, `LiveInputsCache`) -- a page refresh within the same settlement period
reuses the cached inputs; crossing a boundary triggers a refetch. Inference itself always reruns
per request (cheap; the cost here is the upstream network calls, not `.predict()`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from edf.data.calendar_events import build_calendar_features
from edf.data.live_demand import fetch_live_demand_actuals
from edf.data.live_ndf import fetch_live_ndf
from edf.data.weather import population_weighted_live_forecast
from edf.features.buckets import is_christmas_period
from edf.features.demand import build_feature_table
from edf.features.weather import build_weather_feature_table_from_frame, cumulative_degree
from edf.models.registry import CUMULATIVE_DEGREE_WINDOW_PERIODS

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 30  # >= trailing_weekly_moving_average's 28-day need, plus margin
WEATHER_FORECAST_DAYS_AHEAD = 9  # comfortably covers the 7d horizon's target


def settlement_floor(now_utc: pd.Timestamp) -> pd.Timestamp:
    """Floor `now_utc` to its current half-hour settlement boundary -- the cache key
    (10:00-10:30, 10:30-11:00, ...)."""
    return now_utc.floor("30min")


@dataclass
class LiveInputs:
    demand_history: pd.DataFrame  # timestamp, demand -- real actuals only
    weather_forecast: pd.DataFrame  # timestamp + weather columns, live forecast
    ndf: pd.DataFrame
    fetched_at: pd.Timestamp


def fetch_live_inputs(now_utc: pd.Timestamp) -> LiveInputs:
    demand = fetch_live_demand_actuals()
    start = now_utc.strftime("%Y-%m-%d")
    end = (now_utc + pd.Timedelta(days=WEATHER_FORECAST_DAYS_AHEAD)).strftime("%Y-%m-%d")
    weather = population_weighted_live_forecast(start, end)
    ndf = fetch_live_ndf()
    return LiveInputs(demand_history=demand, weather_forecast=weather, ndf=ndf, fetched_at=now_utc)


@dataclass
class LiveInputsCache:
    """Refetches `fetch_live_inputs` only when the settlement-period key changes.

    If a refetch fails (e.g. a live-forecast rate limit) and a previous fetch already
    succeeded, serves that stale cache rather than failing the request outright -- slightly
    stale live inputs are a better response than a 500. Only propagates the exception when
    there's no prior cache to fall back on.
    """

    _key: pd.Timestamp | None = field(default=None, init=False)
    _inputs: LiveInputs | None = field(default=None, init=False)

    def get(self, now_utc: pd.Timestamp) -> LiveInputs:
        key = settlement_floor(now_utc)
        if key != self._key or self._inputs is None:
            try:
                self._inputs = fetch_live_inputs(now_utc)
                self._key = key
            except Exception:
                if self._inputs is None:
                    raise
                logger.warning(
                    "fetch_live_inputs failed for %s; serving stale inputs fetched at %s",
                    now_utc,
                    self._inputs.fetched_at,
                    exc_info=True,
                )
        return self._inputs


@dataclass
class HorizonFeatures:
    horizon_name: str
    issue_time: pd.Timestamp
    target_time: pd.Timestamp
    point_row: pd.DataFrame  # single row, columns == metadata["point_feature_columns"]
    quantile_row: pd.DataFrame  # single row, columns == metadata["quantile_feature_columns"]


def _extended_demand_calendar_frame(
    demand_history: pd.DataFrame, issue_time: pd.Timestamp, target_time: pd.Timestamp
) -> pd.DataFrame:
    start = issue_time - pd.Timedelta(days=LOOKBACK_DAYS)
    index = pd.date_range(start, target_time, freq="30min", tz="UTC")
    demand = demand_history.set_index("timestamp")["demand"].reindex(index).rename("demand")
    calendar = build_calendar_features(index)
    return pd.concat([demand, calendar], axis=1)


def _select_columns(df: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: live-assembled features are missing columns the model was trained on: "
            f"{missing} -- a live/train feature-parity bug, not something to silently work around"
        )
    return df[columns]


def build_live_features(
    horizon_name: str, horizon_periods: int, metadata: dict, live_inputs: LiveInputs
) -> HorizonFeatures:
    """One feature row for `horizon_name`, built the same way `edf.models.registry` built it
    at training time, from `live_inputs` instead of the canonical parquet.

    Raises if a required column is missing from the live-assembled features relative to
    `metadata`'s recorded training-time feature list.
    """
    issue_time = live_inputs.demand_history["timestamp"].max()
    target_time = issue_time + pd.Timedelta(minutes=30 * horizon_periods)

    df = _extended_demand_calendar_frame(live_inputs.demand_history, issue_time, target_time)
    weather = build_weather_feature_table_from_frame(live_inputs.weather_forecast, df.index)

    X_base, _ = build_feature_table(df, horizon_periods, weather=weather)
    if target_time not in X_base.index:
        raise ValueError(
            f"{horizon_name}: target time {target_time} has no complete feature row -- likely "
            "insufficient live demand history or weather coverage for this horizon"
        )

    if metadata["bias_corrected"]:
        extra = pd.DataFrame(index=df.index)
        extra["is_christmas"] = is_christmas_period(df.index).astype(int)
        extra["heating_degree_1d_cum"] = cumulative_degree(
            weather["heating_degree"], CUMULATIVE_DEGREE_WINDOW_PERIODS
        )
        extra["cooling_degree_1d_cum"] = cumulative_degree(
            weather["cooling_degree"], CUMULATIVE_DEGREE_WINDOW_PERIODS
        )
        X_point = X_base.join(extra.loc[X_base.index])
    else:
        X_point = X_base

    point_row = _select_columns(X_point, metadata["point_feature_columns"], horizon_name).loc[
        [target_time]
    ]
    quantile_row = _select_columns(
        X_base, metadata["quantile_feature_columns"], horizon_name
    ).loc[[target_time]]

    return HorizonFeatures(
        horizon_name=horizon_name,
        issue_time=issue_time,
        target_time=target_time,
        point_row=point_row,
        quantile_row=quantile_row,
    )
