"""Live GB demand actuals: NESO's rolling "Demand Data Update" feed.

Same schema/semantics as the historic yearly files `edf.data.download` parses (`demand` = ND, per
PLAN.md's canonical column decision) -- just a different, continuously-overwritten resource
covering roughly the last ~45 days plus a short forward-looking tail. Verified live (2026-09-08):
identical columns, including `ND` and `FORECAST_ACTUAL_INDICATOR`, to the yearly historic CSVs.
Rows tagged `"F"` in that column are placeholder forecast rows (`ND=0`), not real actuals, and are
dropped here -- only `"A"` (actual) rows are ever returned.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from edf.data.download import settlement_periods_to_utc

LIVE_DEMAND_URL = (
    "https://api.neso.energy/dataset/7a12172a-939c-404c-b581-a6128b74f588"
    "/resource/177f6fa4-ae49-4182-81ea-0c6b35f26ca6/download/demanddataupdate.csv"
)


def parse_live_demand_csv(csv_text: str) -> pd.DataFrame:
    """`csv_text` -> `timestamp`/`demand` frame, actuals only, sorted.

    Verified live (2026-09-08): the most recent one or two `"A"`-tagged rows can still carry a
    placeholder `ND=0` -- the period has closed but the real reading hasn't landed in the feed
    yet, ahead of `FORECAST_ACTUAL_INDICATOR` flipping. GB national demand is never actually 0
    (`edf.data.clean.check_no_negative_demand` already guards the historic data against `<0`,
    but `0` itself needs its own filter here, live-only), so these are dropped too, not just
    `"F"` rows.
    """
    df = pd.read_csv(StringIO(csv_text))
    actual = df[(df["FORECAST_ACTUAL_INDICATOR"] == "A") & (df["ND"] > 0)].copy()
    actual["timestamp"] = settlement_periods_to_utc(
        actual["SETTLEMENT_DATE"], actual["SETTLEMENT_PERIOD"]
    )
    return (
        actual[["timestamp", "ND"]]
        .rename(columns={"ND": "demand"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_live_demand_actuals(url: str = LIVE_DEMAND_URL, timeout: int = 30) -> pd.DataFrame:
    """Real (non-forecast) settlement-period demand actuals from NESO's rolling feed."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_live_demand_csv(response.text)
