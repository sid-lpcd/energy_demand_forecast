# Findings

A living log of concrete, checked findings from data exploration — updated whenever an analysis
produces a real result, not every intermediate exploration step. See `PLAN.md` for the plan and
`notebooks/` for the underlying analysis behind each entry.

## 1. Demand shape & the "is the decline real?" question

- GB demand is **winter-peaking** with no material summer air-conditioning peak — structurally
  different from summer-peaking systems (e.g. most of the US). Cold, not heat, is the extreme worth
  caring about most (see §2).
- ND declined ~3.7% from 2020 to 2025, but a "true demand" proxy (`ND + embedded wind + embedded
  solar`, reconstructing what ND is hiding — see `PLAN.md`'s embedded-generation note) declined only
  ~1.2% over the same period. **Roughly two-thirds of the apparent ND decline is a growing-embedded
  -generation visibility artifact, not real demand reduction** — though a genuine ~1.2% underlying
  decline remains, not yet explained (not weather-normalized). Verified the arithmetic holds exactly
  at the row level (`true_demand_proxy − (ND + wind_gen + solar_gen)` = 0 for every half-hour), not
  just approximately at the annual aggregate.
- **This is almost entirely a solar story, not a wind one** — worth being precise about, since
  "wind+solar capacity grew ~39%" (an earlier, blurrier framing) obscures that wind and solar moved
  in opposite directions. 2020→2025: wind capacity +1.4% (6,515→6,606 MW) while wind *generation*
  **fell** ~6% (2,077→1,951 MW, capacity factor 31.9%→29.5%) — most plausibly 2025 was simply a
  less windy year on average, a weather-variability effect, not a capacity one. Solar capacity grew
  **+57.8%** (13,188→20,813 MW) and solar generation grew almost exactly in proportion (+55.1%,
  1,374→2,130 MW, capacity factor flat at ~10%) — a clean, capacity-driven trend. Capacity and
  generation are different scales (capacity factor ~10-32%, nowhere near 100%) and shouldn't be
  compared directly — a real correction made after a user question caught the first draft
  conflating them. Source: `notebooks/01_explore_raw_data.ipynb`.
- 2021 was the highest-demand year in both ND and the true-demand proxy — plausibly a colder winter
  rather than a COVID-rebound effect (lockdown 3 was still running through Q1 2021). Not yet checked
  against weather data.

## 2. Weather

- Open-Meteo (ERA5) and NASA POWER agree closely but not identically — 0.98 temperature correlation,
  ~1°C mean absolute difference. Genuinely independent sources, not redundant.
- Open-Meteo's "Historical Forecast API" is a short-lead (~0-3h) **nowcast**, not a day-ahead
  forecast — corrected after initially mischaracterizing it as "archived forecasts" (see `PLAN.md`).
- **Unresolved**: a genuine day-ahead forecast source (`_previous_day1`) shows *lower* MAE against
  ERA5 hindsight than the short-lead nowcast does — even after pinning both endpoints to the same
  explicit model (`gfs_seamless`), ruling out "different default model" as the explanation. The
  naive "more lead time = more forecast error" expectation doesn't hold here and the cause isn't
  understood yet. Flagged as a Week 4 pre-req, not resolved. Source:
  `notebooks/02_explore_weather_data.ipynb`.
- **Cold vs. heat asymmetry, confirmed with real numbers**: the 13 Dec 2022 cold snap
  (-17.3°C overnight, Braemar) added **+5,319 MW** average demand vs. a control day (~17% higher,
  peak +4,869 MW); the 19 Jul 2022 record heatwave (40.3°C, Coningsby, UK's hottest day on record)
  added only **+712 MW** (~2.8% higher, peak +888 MW). **Cold has ~7x the demand impact heat does**
  — consistent with GB's gas-heated, low-AC-penetration housing stock. Source:
  `notebooks/03_interesting_days.ipynb`.

## 3. COVID as a natural experiment

Not yet investigated in depth — a later thread. `lockdown_level` (none/partial/full) is already
built in `src/edf/data/calendar_events.py`, sourced from the Institute for Government's timeline.

## 4. TV/event "pickup" — verified in our own data, not just cited from NESO

- **Euro 2020 R16, England v Germany (29 Jun 2021)**: NESO's own article reports ~1GW pickup at
  half-time, ~1.6GW at full-time. At half-hourly resolution, the match evening's demand declines
  *more slowly* than a control evening's from kickoff through full-time — the same direction as
  NESO's reported effect, though the sharp minute-scale spike itself isn't resolvable at this
  granularity (a stated resolution limitation, not a failure to find the effect).
- **Euro 2020 Final, England v Italy (11 Jul 2021)**: a clearer pattern — demand dips *below* a
  control evening during play, then crosses over and *exceeds* it from full-time through the
  penalty shootout (the match went to extra time + penalties, ending ~22:30 BST).
- **Day-type "how unusual" ranking**, confirmed with real December 2023 numbers: ordinary weekday
  (~32.2 GW avg) > ordinary weekend (~28.2 GW) > bank holiday/Boxing Day (~23.8 GW) > Christmas Day
  (~22.8 GW). The weekday-to-weekend gap is comparable in size to the weekend-to-holiday gap — worth
  remembering when weighing calendar features later.

## 5. Renewable growth & interconnectors

- Embedded wind+solar capacity grew ~39% 2020→2025 (see §1).
- **Interconnector go-live dates are directly visible in the raw flow columns** — a nice concrete
  verification that the data reflects real-world construction: NSL (Norway) near-zero before
  1 Oct 2021, then ~997 MW average magnitude after; Viking (Denmark) near-zero before 29 Dec 2023,
  then ~763 MW after; Greenlink (Ireland) near-zero before 30 Jan 2025, then ~403 MW after.
- GB is a net interconnector **importer** on average in every year 2020-2025 **except 2022**
  (net exporter) — plausibly tied to the 2022 European gas price crisis following Russia's invasion
  of Ukraine (Feb 2022), which may have made continental gas-fired generation relatively more
  expensive than GB's own mix that year. Not yet checked against actual price data — a hypothesis,
  not a confirmed cause.
- Weak but directionally sensible correlations: interconnector flow vs. wind generation ≈ **-0.16**
  (some tendency to export more / import less when windy); vs. demand ≈ **+0.21** (some tendency to
  import more at high demand). Consistent with interconnectors acting as a release valve in both
  directions, just not a dominant one.

### Ideas for future data (renewable-growth context — not yet sourced, for later)

Factors plausibly driving the embedded wind/solar capacity growth seen in the data, worth sourcing
if a fuller causal/contextual story is wanted:

- **Smart Export Guarantee** (SEG, started Jan 2020) — pays small-scale generators for exports,
  replacing the closed Feed-in Tariff scheme; overlaps our window almost exactly.
- **The 2021-2023 European gas price crisis** — sharply higher household/business energy bills,
  creating a strong private payback incentive for rooftop solar specifically in this period.
- **Falling solar panel / battery storage costs** (global LCOE trend).
- **CfD (Contracts for Difference) auction rounds** (AR4 2022, AR5 2023, AR6 2024) — mostly
  larger-scale; worth checking how much of this is embedded vs. transmission-connected capacity.
- **UK Net Zero by 2050 legal commitment** (June 2019) and associated policy support.
- DNO grid-connection queue reforms / permitted development rights changes for small-scale
  installations.
- Possible sources to pull from later: DESNZ (Dept for Energy Security & Net Zero) solar/wind
  deployment statistics; Ofgem's SEG/FiT installation registers; historical wholesale gas/electricity
  prices (to test the 2022 price-crisis interconnector hypothesis directly).

## 6. Data-quality "gotchas" found (methodology notes — all corrected, not left as bugs)

- Embedded wind/solar generation is already invisibly netted into ND (confirmed via NESO's own FAQ)
  — not a separate additive component. Corrected the original Week 7 plan, which had double-
  subtracted it.
- Grouping by UTC calendar date trivially always shows 48 settlement periods/day — the real
  clock-change anomalies (46/50-period days) only show up when grouping by the raw local
  `SETTLEMENT_DATE`.
- `SETTLEMENT_DATE`'s text format is inconsistent across NESO's own yearly files (`01-JAN-2020`,
  `01-Jan-23`, `2025-01-01`).
- `SCOTTISH_TRANSFER` is genuinely absent pre-2023 (NESO added the column), not a data defect.
- Weather source #3 was originally mischaracterized as "archived forecasts" — corrected after
  re-reading Open-Meteo's own docs properly instead of trusting the first read.
- `wind`/`solar`/`interconnector` have no free historical forecast archive available anywhere — a
  genuine, permanent limitation (Week 4b proposes forecasting them ourselves instead of solving the
  data-availability problem directly, since it can't be).

---

*Last updated: 2026-09-06.*
