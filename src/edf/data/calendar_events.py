"""Calendar and one-off event features for GB electricity demand.

Three independent things live here, each with a different mechanism:

1. **Bank/school holidays and COVID lockdown level** — behavioural drivers of
   ordinary demand (offices/schools closed).
2. **Tiered "TV pickup" events** — nationally-televised moments (football
   internationals, cup finals) documented by NESO to cause synchronised
   demand spikes when millions of households use appliances at the same
   moment (e.g. half-time kettles). See PLAN.md Week 1 for the research and
   sourcing behind every date and tier below.
3. **Solar eclipses** — a *physical* driver of embedded solar generation
   (not a demand/TV-behaviour mechanism), kept as a separate column.

All dates/times below were verified against primary or authoritative
secondary sources (NESO, Institute for Government, Wikipedia match records,
official tournament sites) on 2026-09-06 — see PLAN.md for citations. This
is a hand-curated table, not a scraped/API-fetched one: the event count is
small and stable (historical facts don't change), so a small verified table
is more reliable than a scraper.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter

LONDON = "Europe/London"


def bank_holidays(years: range) -> set[date]:
    """England & Wales bank holidays, used as a GB-wide proxy.

    Verified (2026-09-06) that the `holidays` package already includes the
    one-off dates that matter for 2020-2025: the VE Day 75th anniversary move
    (8 May 2020), the Platinum Jubilee extra holiday (3 Jun 2022), the Queen's
    state funeral (19 Sep 2022), and the Coronation of Charles III
    (8 May 2023) — no manual patch needed.
    """
    return set(holidays.UnitedKingdom(subdiv="England", years=list(years)).keys())


def school_holiday_ranges(years: range) -> list[tuple[date, date]]:
    """Approximate England school holiday windows (inclusive start/end dates).

    This is a coarse, rule-based approximation, not a per-local-authority
    calendar — England has ~150 LAs with slightly different dates and there
    is no single authoritative national school calendar. Half-terms use fixed
    calendar weeks; the Easter holiday floats with Easter Sunday. Documented
    as a stated simplification, see PLAN.md.
    """
    ranges: list[tuple[date, date]] = []
    for year in years:
        e = easter(year)
        ranges.append((date(year, 12, 19), date(year + 1, 1, 2)))  # Christmas
        ranges.append((date(year, 2, 15), date(year, 2, 23)))  # February half-term
        ranges.append((e - timedelta(days=9), e + timedelta(days=8)))  # Easter
        ranges.append((date(year, 5, 25), date(year, 6, 2)))  # May half-term
        ranges.append((date(year, 7, 22), date(year, 9, 1)))  # Summer
        ranges.append((date(year, 10, 22), date(year, 10, 31)))  # October half-term
    return ranges


# --- COVID lockdown level -----------------------------------------------
# Source: Institute for Government, "Timeline of UK government coronavirus
# lockdowns and measures, March 2020 to December 2021" (England national
# timeline, used as a GB-wide proxy), corroborated by the House of Commons
# Library (CBP-9068). See PLAN.md Data sources.

_FULL_LOCKDOWN_RANGES = [
    (date(2020, 3, 23), date(2020, 5, 12)),
    (date(2020, 11, 5), date(2020, 12, 1)),
    (date(2021, 1, 6), date(2021, 3, 28)),
]
_PARTIAL_RESTRICTION_RANGES = [
    (date(2020, 5, 13), date(2020, 9, 13)),
    (date(2020, 9, 14), date(2020, 11, 4)),
    (date(2020, 12, 2), date(2021, 1, 5)),
    (date(2021, 3, 29), date(2021, 7, 18)),
]


def lockdown_level(d: date) -> str:
    """Return 'full', 'partial', or 'none' for a given calendar date."""
    for start, end in _FULL_LOCKDOWN_RANGES:
        if start <= d <= end:
            return "full"
    for start, end in _PARTIAL_RESTRICTION_RANGES:
        if start <= d <= end:
            return "partial"
    return "none"


# --- Tiered TV-pickup / national events ----------------------------------
# tier follows the user's scope call: internationals + cup finals = "large";
# other genuinely national moments (Olympics, major domestic cricket/rugby/
# election coverage) are "medium" or "small" based on documented audience
# size and whether the event has a synchronised "everyone watching the same
# break at once" mechanism (evening free-to-air TV) vs. not (all-day cricket,
# subscription/PPV boxing, small audience relative to football).


@dataclass(frozen=True)
class Event:
    start: str  # ISO date or datetime, local Europe/London time unless noted
    end: str
    name: str
    category: str
    tier: str  # "large" | "medium" | "small"


EVENTS: list[Event] = [
    # --- Euro 2020 (played summer 2021) — England, all "large" ---
    Event("2021-06-13T14:00", "2021-06-13T16:00", "Euro 2020: England v Croatia (group)", "football_intl", "large"),
    Event("2021-06-18T20:00", "2021-06-18T22:00", "Euro 2020: England v Scotland (group)", "football_intl", "large"),
    Event("2021-06-22T20:00", "2021-06-22T22:00", "Euro 2020: England v Czech Republic (group)", "football_intl", "large"),
    Event("2021-06-29T17:00", "2021-06-29T19:00", "Euro 2020: England v Germany (R16)", "football_intl", "large"),
    # NESO-documented: ~1GW pickup at half-time, ~1.6GW at full-time.
    Event("2021-07-03T20:00", "2021-07-03T22:00", "Euro 2020: England v Ukraine (QF)", "football_intl", "large"),
    Event("2021-07-07T20:00", "2021-07-07T22:00", "Euro 2020: England v Denmark (SF)", "football_intl", "large"),
    Event("2021-07-11T20:00", "2021-07-11T22:30", "Euro 2020 Final: England v Italy", "football_intl", "large"),
    # --- World Cup 2022 (Qatar) — England ---
    Event("2022-11-21T13:00", "2022-11-21T15:00", "World Cup 2022: England v Iran (group)", "football_intl", "large"),
    Event("2022-11-25T19:00", "2022-11-25T21:00", "World Cup 2022: England v USA (group)", "football_intl", "large"),
    Event("2022-11-29T19:00", "2022-11-29T21:00", "World Cup 2022: England v Wales (group)", "football_intl", "large"),
    Event("2022-12-04T15:00", "2022-12-04T17:00", "World Cup 2022: England v Senegal (R16)", "football_intl", "large"),
    Event("2022-12-10T19:00", "2022-12-10T21:00", "World Cup 2022: England v France (QF)", "football_intl", "large"),
    # --- Women's Euro 2022 — England ---
    Event("2022-07-06T20:00", "2022-07-06T22:00", "Women's Euro 2022: England v Austria (group)", "football_intl", "large"),
    Event("2022-07-11T20:00", "2022-07-11T22:00", "Women's Euro 2022: England v Norway (group)", "football_intl", "large"),
    Event("2022-07-15T20:00", "2022-07-15T22:00", "Women's Euro 2022: England v Northern Ireland (group)", "football_intl", "large"),
    Event("2022-07-20T20:00", "2022-07-20T22:00", "Women's Euro 2022: England v Spain (QF)", "football_intl", "large"),
    Event("2022-07-26T20:00", "2022-07-26T22:00", "Women's Euro 2022: England v Sweden (SF)", "football_intl", "large"),
    Event("2022-07-31T17:00", "2022-07-31T19:30", "Women's Euro 2022 Final: England v Germany", "football_intl", "large"),
    # Record UK women's-football TV audience (~17.4M).
    # --- Women's World Cup 2023 — England (morning UK kickoffs) ---
    Event("2023-07-22T10:30", "2023-07-22T12:30", "Women's World Cup 2023: England v Haiti (group)", "football_intl", "large"),
    Event("2023-07-28T09:30", "2023-07-28T11:30", "Women's World Cup 2023: England v Denmark (group)", "football_intl", "large"),
    Event("2023-08-01T11:00", "2023-08-01T13:00", "Women's World Cup 2023: England v China (group)", "football_intl", "large"),
    Event("2023-08-07T08:30", "2023-08-07T10:30", "Women's World Cup 2023: England v Nigeria (R16)", "football_intl", "large"),
    Event("2023-08-12T10:30", "2023-08-12T12:30", "Women's World Cup 2023: England v Colombia (QF)", "football_intl", "large"),
    Event("2023-08-16T11:00", "2023-08-16T13:00", "Women's World Cup 2023: England v Australia (SF)", "football_intl", "large"),
    Event("2023-08-20T11:00", "2023-08-20T13:00", "Women's World Cup 2023 Final: England v Spain", "football_intl", "large"),
    # --- Domestic cup finals (all years, Wembley) ---
    Event("2020-08-01T17:30", "2020-08-01T19:30", "FA Cup Final 2020", "cup_final", "large"),
    Event("2021-05-15T17:15", "2021-05-15T19:15", "FA Cup Final 2021", "cup_final", "large"),
    Event("2022-05-14T16:45", "2022-05-14T18:45", "FA Cup Final 2022", "cup_final", "large"),
    Event("2023-06-03T15:00", "2023-06-03T17:00", "FA Cup Final 2023", "cup_final", "large"),
    Event("2024-05-25T16:30", "2024-05-25T18:30", "FA Cup Final 2024", "cup_final", "large"),
    Event("2025-05-17T16:30", "2025-05-17T18:30", "FA Cup Final 2025", "cup_final", "large"),
    Event("2020-03-01T16:30", "2020-03-01T18:30", "EFL Cup Final 2020", "cup_final", "large"),
    Event("2021-04-25T16:30", "2021-04-25T18:30", "EFL Cup Final 2021", "cup_final", "large"),
    Event("2022-02-27T15:00", "2022-02-27T17:00", "EFL Cup Final 2022", "cup_final", "large"),
    Event("2023-02-26T15:00", "2023-02-26T17:00", "EFL Cup Final 2023", "cup_final", "large"),
    Event("2024-02-25T16:30", "2024-02-25T18:30", "EFL Cup Final 2024", "cup_final", "large"),
    Event("2025-03-16T16:30", "2025-03-16T18:30", "EFL Cup Final 2025", "cup_final", "large"),
    # --- Champions League Final (all years; "cup final" category = large) ---
    Event("2020-08-23T21:00", "2020-08-23T23:00", "Champions League Final 2020: Bayern v PSG", "cup_final", "large"),
    Event("2021-05-29T20:00", "2021-05-29T22:00", "Champions League Final 2021: Man City v Chelsea", "cup_final", "large"),
    Event("2022-05-28T20:00", "2022-05-28T22:00", "Champions League Final 2022: Liverpool v Real Madrid", "cup_final", "large"),
    Event("2023-06-10T20:00", "2023-06-10T22:00", "Champions League Final 2023: Man City v Inter", "cup_final", "large"),
    Event("2024-06-01T20:00", "2024-06-01T22:00", "Champions League Final 2024: Real Madrid v Dortmund", "cup_final", "large"),
    Event("2025-05-31T20:00", "2025-05-31T22:00", "Champions League Final 2025: PSG v Inter", "cup_final", "large"),
    # --- Medium tier: Olympics (date-range flag, not per-event) ---
    Event("2021-07-23", "2021-08-08", "Tokyo 2020 Olympics", "olympics", "medium"),
    Event("2024-07-26", "2024-08-11", "Paris 2024 Olympics", "olympics", "medium"),
    # --- Medium tier: Clap for Carers (NESO-documented ~800MW surge) ---
    # Weekly, Thursdays 20:00 local, 26 Mar - 28 May 2020. Generated below.
    # --- Small tier: lower-reach or non-synchronised events ---
    Event("2023-06-16", "2023-07-31", "The Ashes 2023 (Test cricket, no single sync. break)", "cricket", "small"),
    Event("2023-09-08", "2023-10-28", "Rugby World Cup 2023 (England: 3rd place, no final)", "rugby", "small"),
    Event("2023-10-05", "2023-11-04", "Cricket World Cup 2023 (England group stage, unusual TZ)", "cricket", "small"),
    Event("2024-09-21T20:00", "2024-09-21T23:00", "Joshua v Dubois (Wembley, Sky Sports PPV)", "boxing", "small"),
    Event("2024-07-04T22:00", "2024-07-05T04:00", "UK General Election night 2024 (~4.5M peak, non-sync.)", "election", "small"),
]


def clap_for_carers_events() -> list[Event]:
    """NESO documented an ~800MW demand surge from this weekly ritual.

    Thursdays 20:00 local time, 26 March - 28 May 2020 (the ten Thursdays of
    the first national lockdown during which the clap was widely observed).
    """
    events = []
    d = date(2020, 3, 26)
    end = date(2020, 5, 28)
    while d <= end:
        events.append(
            Event(f"{d.isoformat()}T20:00", f"{d.isoformat()}T20:30", "Clap for Carers", "clap_for_carers", "medium")
        )
        d += timedelta(days=7)
    return events


ALL_EVENTS: list[Event] = EVENTS + clap_for_carers_events()


# --- Solar eclipses (physical driver of embedded solar generation) -------
# Times are UTC (GMT); note 10 Jun 2021 is during BST (UTC+1) so local clock
# times were one hour later than the UTC values below.

SOLAR_ECLIPSES = [
    # (start_utc, peak_utc, end_utc, max_obscuration_pct_uk, note)
    ("2021-06-10T09:08", "2021-06-10T10:13", "2021-06-10T11:22", 50, "Annular; ~30% south UK, ~50% north"),
    ("2022-10-25T08:58", "2022-10-25T11:00", "2022-10-25T13:02", 28, "Partial; ~15% London to ~28% Shetland"),
    ("2025-03-29T10:07", "2025-03-29T11:03", "2025-03-29T12:00", 60, "Partial; ~40% SE to ~60% NW"),
]


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build calendar/event feature columns aligned to a UTC timestamp index.

    Columns: is_bank_holiday, is_school_holiday, lockdown_level, event_tier
    (none/small/medium/large — max tier if events overlap), event_name,
    solar_eclipse_pct (0 if none).
    """
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("index must be UTC tz-aware")

    local = index.tz_convert(LONDON)
    dates = local.date
    years = range(local.year.min(), local.year.max() + 1)

    bh = bank_holidays(years)
    is_bank_holiday = pd.Series([d in bh for d in dates], index=index)

    school_ranges = school_holiday_ranges(years)
    is_school_holiday = pd.Series(
        [any(start <= d <= end for start, end in school_ranges) for d in dates], index=index
    )

    lockdown = pd.Series([lockdown_level(d) for d in dates], index=index, dtype="category")

    tier_rank = {"none": 0, "small": 1, "medium": 2, "large": 3}
    event_tier = pd.Series("none", index=index, dtype=object)
    event_name = pd.Series(pd.NA, index=index, dtype=object)
    for event in ALL_EVENTS:
        start = pd.Timestamp(event.start, tz=LONDON)
        end = pd.Timestamp(event.end, tz=LONDON)
        if "T" not in event.end:
            end += pd.Timedelta(days=1)  # date-only end is inclusive of that whole day
        mask = (local >= start) & (local < end)
        higher = mask & (event_tier.map(tier_rank) < tier_rank[event.tier])
        event_tier.loc[higher] = event.tier
        event_name.loc[higher] = event.name

    solar_eclipse_pct = pd.Series(0.0, index=index)
    for start_s, peak_s, end_s, max_pct, _note in SOLAR_ECLIPSES:
        start = pd.Timestamp(start_s + ":00Z")
        peak = pd.Timestamp(peak_s + ":00Z")
        end = pd.Timestamp(end_s + ":00Z")
        mask = (index >= start) & (index <= end)
        if not mask.any():
            continue
        # Triangular obscuration profile: 0 at start/end, max_pct at peak.
        t = index[mask]
        before_peak = t <= peak
        frac = np.where(
            before_peak,
            (t - start) / (peak - start),
            (end - t) / (end - peak),
        )
        solar_eclipse_pct.loc[mask] = np.clip(frac, 0, 1) * max_pct

    return pd.DataFrame(
        {
            "is_bank_holiday": is_bank_holiday,
            "is_school_holiday": is_school_holiday,
            "lockdown_level": lockdown,
            "event_tier": pd.Categorical(event_tier, categories=["none", "small", "medium", "large"]),
            "event_name": event_name,
            "solar_eclipse_pct": solar_eclipse_pct,
        },
        index=index,
    )
