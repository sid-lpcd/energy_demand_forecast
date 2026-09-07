# Plan — Machine Learning for Probabilistic Short-Term Electricity Demand Forecasting in Great Britain

Last updated: 2026-09-06

## Why this project

GB electricity demand forecasting is a real operational problem: the system operator (NESO) and
suppliers forecast demand every settlement period to balance the grid, and forecast error has a
direct cost — more balancing actions, more reliance on fast-reacting (often gas) reserve, and in
some conditions more renewable curtailment. Better *probabilistic* forecasts (not just a point
number, but a calibrated range) let a system lean less on conservative fossil backup and more on
variable renewables with confidence.

This project is a genuine research exercise, not just a leaderboard exercise: it ends with an
honest, quantified attempt to answer "does this actually matter, and by how much?" — which is what
makes it relevant to climate-tech / energy-systems / EA-adjacent audiences, not just an ML resume
line.

## Guiding principles (apply every week)

1. **Time-aware validation only.** Never shuffle. Use expanding-window / walk-forward backtests
   (e.g. train on year N, validate on N+1, roll forward) from Week 1 onward. A random train/test
   split on time series data silently leaks the future into training and invalidates every later
   result.
2. **One comparable scale across all weeks.** Report MAE and RMSE, but also **MASE** (MAE scaled
   against a seasonal-naive baseline) from Week 2 onward, so Week 3's LightGBM, Week 4's weather
   model, and Week 5's probabilistic model are all comparable to the same baseline on the same
   scale.
3. **Everything reusable lives in `src/edf`, not just notebooks.** Notebooks are for exploration;
   once code is used twice, it becomes a tested function in `src/edf/`.
4. **Write the limitation down when you hit it, not in Week 8.** E.g. "this uses observed weather,
   not forecast weather" belongs in a running `LIMITATIONS.md`-style note the moment you notice it.
5. **Small, dated commits.** Each week ends with a working, reproducible state and a short summary
   of what was learned — including negative results (e.g. "weather barely helped for wind, a lot
   for temperature-driven demand").

## Data sources

| Feature | Source | Notes |
|---|---|---|
| Demand, embedded wind/solar, interconnector flows | [NESO Data Portal](https://www.neso.energy/data-portal) — Historic Demand Data (half-hourly, 2009–present) | Free, no key. Confirm exact field names/units when downloading (ND vs TSD, embedded generation estimates vs metered). |
| Weather — 4 independent sources (done, see Week 1) | [Open-Meteo Historical Weather API](https://open-meteo.com/) (ERA5 reanalysis, primary, never available ahead of time by design), [NASA POWER API](https://power.larc.nasa.gov/) (independent reanalysis, cross-check), [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) (a **short-lead nowcast archive, NOT a day-ahead forecast** — corrected 2026-09-06, see Week 1 — valid from 2022-03-01, verified empirically), [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api) `_previous_day1` (**genuine ~24h-ahead forecast**, valid from 2024-03-07, verified empirically per-variable) | All free/keyless. Population-weighted across London/Birmingham/Manchester/Glasgow/Leeds (ballpark conurbation-population proxy, not a full gridded weighting). Open-Meteo vs. NASA POWER temperature correlation 0.98 on the full 2020-2025 series — confirms the sources are genuinely independent yet agree, not redundant. |
| NESO's own day-ahead demand forecast (done, see Week 1) | [NESO Data Portal, "Day Ahead Demand Forecast"](https://www.neso.energy/data-portal/1-day-ahead-demand-forecast) — "Historic Day Ahead Demand Forecasts" resource, 2018–present | A genuine deployed operational forecast, not an ML estimate — used as a benchmark ("does our model beat what NESO actually forecast?"), not a model input. ~12 "cardinal points"/day (overnight minimum, peaks, etc.), not full half-hourly resolution. See `src/edf/data/demand_forecast_benchmark.py`. |
| UK bank holidays / school holidays / event calendar (done, see Week 1) | `holidays` Python package (verified to already include one-off dates — VE Day move, Platinum Jubilee, Queen's funeral, Coronation); hand-curated event table for school holidays, football/TV events, Olympics, Clap for Carers, solar eclipses in `src/edf/data/calendar_events.py` | Demand drops sharply on bank holidays; some GB demand spikes ("TV pickup") are driven by nationally-watched TV moments — see Week 1 for the full researched/sourced list and tiering, and for why a measured TV-audience dataset isn't used here. |
| Balancing costs / imbalance prices (Week 8 impact estimate) | [Elexon BMRS](https://www.bmreports.com/) / NESO balancing costs data | Used only for a back-of-envelope impact estimate, not as a model input. |
| Carbon intensity | [Carbon Intensity API (National Grid ESO)](https://carbonintensity.org.uk/) | Used for the Week 8 impact estimate — connects forecast error to plausible emissions impact. |
| UK COVID-19 restriction periods (for a 3-level `lockdown_level` stress-test feature) | [Institute for Government, "Timeline of UK government coronavirus lockdowns and measures, March 2020 to December 2021"](https://www.instituteforgovernment.org.uk/data-visualisation/timeline-coronavirus-lockdowns), corroborated by [House of Commons Library, "Coronavirus: A history of English lockdown laws" (CBP-9068)](https://commonslibrary.parliament.uk/research-briefings/cbp-9068/) | England's national timeline used as a GB-wide proxy — devolved nations sometimes diverged (e.g. Wales' Oct 2020 "firebreak"); documented simplification, not modelled separately. See Week 1 for the exact date ranges. |

**Known limitations to state explicitly (not to solve — except the first, see Week 4b):**

- **`wind`, `solar`, and `interconnector` (as currently used in Weeks 1-3) are same-period actuals,
  not anything available ahead of time.** Checked directly (2026-09-06): NESO's own "Demand Data
  Update" dataset does publish a 7-day-ahead embedded wind/solar forecast, but it's a **rolling
  window only** (last month + next 7 days, continuously overwritten) — NESO does not retain a
  historical archive of what that forecast said on any past date, and nobody scraped it in real
  time for 2020-2025. So unlike weather, there is no free historical forecast archive for these at
  all, not even a mischaracterized one. **Week 4b below fixes this** by forecasting wind/solar
  ourselves from weather and only using *those* forecasts as demand-model features; interconnector
  flow is harder (price/economics-driven, not just weather-driven) and is more likely to end up
  documented as an unfixed limitation than actually solved — see Week 4b.
- This project uses *observed* historical weather as a feature (for `wind`/`solar`'s inputs and
  directly in Week 4), but a real operational forecaster only has *forecast* weather at lead time.
  The 4th weather source (Open-Meteo Previous Runs API, genuine day-ahead) fixes this for the
  2024-03-07-onward slice; before that, results are an upper bound — call this out in the final
  report rather than presenting it as if it were deployable as-is for the full 2020-2025 window.
- **No freely-available, systematically-collected TV-audience dataset exists for this project's
  window.** Checked directly (2026-09-06): BARB (the official UK measurement body) publishes daily
  "overnight" ratings, but only to industry subscribers; its free public site only offers weekly
  top-30/50 *aggregate* charts (browse-by-week, no bulk historical export found). This isn't a
  forecast-availability problem the way wind/solar is — football/TV event *scheduling* is known
  months ahead, so there's no leakage risk — it just means the event-tier labels in
  `calendar_events.py` are a researched qualitative judgement (cross-checked against one hard NESO
  MW figure for Euro 2020), not a continuously measured variable. Good enough to use as a categorical
  feature; not something to overstate as precise.
- **Embedded wind/solar generation is already invisibly netted into the demand figures — it is not
  a separate additive component.** Per NESO's own FAQ for this dataset: demand figures "are based
  upon the Total Generation Output from Transmission contracted units only," and NESO "do[es] not
  receive any metering from DNOs [Distribution Network Operators]." Rooftop solar and small/
  distribution-connected wind reduce local consumption *before* it reaches transmission-metered
  points, so `ND` is suppressed by embedded generation by physics, not by any subtraction NESO
  performs — which is exactly why `EMBEDDED_WIND_GENERATION`/`EMBEDDED_SOLAR_GENERATION` exist as a
  separate *modelled* (not metered) estimate. Practical consequence for this project: `wind` and
  `solar` are used as **predictive features** (they explain real weather-driven dips in demand),
  never subtracted from `demand` a second time — see Week 7 for what this changes.

## Week 1 — Data & Project Setup

- Set up the repo (done): `src/edf` package, `tests/`, `data/{raw,processed}` (gitignored),
  `notebooks/`, `reports/`.
- Downloader (done): `src/edf/data/download.py` pulls NESO historic demand data for 2020–2025 into
  `data/raw/` and builds a combined raw parquet, idempotent (skip if already downloaded). Raw-data
  exploration (done): `notebooks/01_explore_raw_data.ipynb`.
- **Canonical column decisions (finalized):**
  - `demand` = **ND** (National Demand), not TSD. TSD adds pump-storage pumping and station load
    back in — pumping in particular is a discretionary, price-driven decision, not a
    weather/calendar-driven demand signal, so including it would add noise the model has no
    features to explain. TSD stays available in the raw table if needed later.
  - `wind` = `EMBEDDED_WIND_GENERATION`, `solar` = `EMBEDDED_SOLAR_GENERATION` — used as **features**
    of the demand model, not subtracted from `demand`. See the "known limitations" note above: this
    embedded generation is already invisibly baked into `ND`, so subtracting it again would
    double-count it and produce a number with no clean physical meaning.
  - `interconnector` = sum of all `*_FLOW` columns (net GB import, +import/−export). Individual
    links are highly collinear (all driven by the same GB-vs-Europe price differential); the raw
    per-link columns remain available if link-specific analysis is ever needed.
- **Build the cleaned, canonical half-hourly table (done):** `src/edf/data/clean.py` joins the raw
  demand data with the calendar/event features into `data/processed/gb_energy_2020_2025.parquet` —
  indexed by `timestamp` (UTC, tz-aware), with `demand`, `wind`, `solar`, `interconnector`, plus
  `is_bank_holiday`/`is_school_holiday`/`lockdown_level`/`event_tier`/`event_name`/
  `solar_eclipse_pct`. `check_no_gaps`/`check_no_negative_demand` are regression guards for the
  zero-gaps/zero-negative-demand finding from `notebooks/01_explore_raw_data.ipynb`, not repair
  logic — nothing needed fixing. Tested in `tests/data/test_clean.py`.
- Add a `lockdown_level` calendar feature (`none` / `partial` / `full`) for the COVID-19 stress-test
  work in Weeks 6/8, using England's national restriction timeline (Institute for Government,
  corroborated by the House of Commons Library — see Data sources) as a GB-wide proxy:
  - **`full`** (legal "stay at home" order in force): 2020-03-23 to 2020-05-12; 2020-11-05 to
    2020-12-01; 2021-01-06 to 2021-03-28.
  - **`partial`** (tiers / rule-of-six / exit-roadmap steps, no stay-at-home order):
    2020-05-13 to 2020-09-13; 2020-09-14 to 2020-11-04; 2020-12-02 to 2021-01-05; 2021-03-29 to
    2021-07-18.
  - **`none`**: everything else (pre-2020-03-23, and from 2021-07-19 — "Freedom Day" — onward).
- **Calendar/event and weather feature datasets (done):** `src/edf/data/calendar_events.py` and
  `src/edf/data/weather.py`, both tested (`tests/data/test_calendar_events.py`,
  `tests/data/test_weather.py`) and run end-to-end against live sources.
  - **Bank holidays**: England & Wales via the `holidays` package — verified it already includes
    every one-off date needed (VE Day move, Jubilee, Queen's funeral, Coronation), no manual patch.
  - **School holidays**: approximate England term-time windows (Christmas/half-terms fixed weeks,
    Easter floating off Easter Sunday) — a stated approximation, not a per-LA calendar.
  - **Tiered "TV pickup" events**: a hand-curated table of nationally-televised moments documented
    by NESO to cause synchronised demand spikes (e.g. Euro 2020 England v Germany: NESO-recorded
    ~1GW pickup at half-time, ~1.6GW at full-time — one of the largest events in UK grid history).
    Tiers — **large**: all England international-tournament matches (Euro 2020, World Cup 2022,
    Women's Euro 2022, Women's World Cup 2023) and all major domestic cup finals (FA Cup, EFL Cup,
    Champions League Final); **medium**: the Olympics (Tokyo 2020, Paris 2024, as a date-range flag)
    and "Clap for Carers" (NESO-documented ~800MW surge, Thursdays 20:00, 26 Mar–28 May 2020);
    **small**: lower-reach or non-synchronised events kept for completeness (Ashes 2023, Rugby World
    Cup 2023 — England didn't reach the final, Cricket World Cup 2023, boxing — subscription/PPV
    not free-to-air, the 2024 General Election night — ~4.5M peak audience, an order of magnitude
    below Euro 2020's ~30M). Excluded entirely: routine weekly Premier League/EFL fixtures (too
    frequent, diffuse per-game effect, expensive to source reliably), Wimbledon (no British finalist
    2020–2025), local/regional events, political/budget announcements.
  - **Solar eclipses** (a *physical* driver of embedded solar generation, not a TV/behaviour
    mechanism, so kept as its own `solar_eclipse_pct` column): 10 Jun 2021, 25 Oct 2022,
    29 Mar 2025 — exact UK start/peak/end times and obscuration percentages sourced and verified.
  - **Weather, 4 sources**: Open-Meteo Historical (ERA5, primary), NASA POWER (independent
    cross-check — 0.98 temperature correlation with Open-Meteo on the full run, confirming the
    sources agree while being genuinely different), the Open-Meteo Historical Forecast API (a
    short-lead **nowcast** archive, valid from 2022-03-01), and the Open-Meteo Previous Runs API
    `_previous_day1` (a genuine **day-ahead** forecast, valid from 2024-03-07). Both cutovers needed
    a live check, not just documentation — and the correction mattered: the Historical Forecast API
    was originally (wrongly) described here as "archived real forecasts"; its own docs actually say
    it "stitches together the first few hours of each model run" (a nowcast, ~0-3h lead, not
    day-ahead). Both start dates were empirically verified by diffing/checking the live APIs (not
    assumed from documentation) and are guarded in code (`NOWCAST_ARCHIVE_START`,
    `DAY_AHEAD_ARCHIVE_START` in `src/edf/data/weather.py`) — the day-ahead cutover in particular
    needed per-variable checking, since `shortwave_radiation_previous_day1` starts a month later
    (2024-03-07) than `temperature_2m_previous_day1` (2024-02-04); the later of the two is used so
    the resulting dataset has zero nulls.
  - **NESO's own day-ahead demand forecast** (`src/edf/data/demand_forecast_benchmark.py`): a real
    2018-present archive of NESO's deployed day-ahead forecast, ~12 "cardinal points"/day (not full
    half-hourly). Kept as a benchmark ("does our model beat the real one?"), not a model input.
- Handle: missing settlement periods, clock-change days (23/25-hour days), obvious sensor/reporting
  outliers, unit consistency (MW throughout). (Raw-data check already found zero gaps/duplicates/
  negative values across 2020–2025 — see the exploration notebook — so this step may mostly be
  confirming there's nothing left to do rather than fixing anything.)
- **Train/validation/test split (done):** fixed in `src/edf/config.py` — train 2020–2023,
  validation 2024, test 2025 — for the rest of the project so no week accidentally peeks at test
  data. (COVID's acute disruption falls entirely inside the training window under this split, so
  reported val/test accuracy isn't distorted by it — see Week 6 for how the training-time COVID
  period is still put to use.)
- Plot: full time series, one winter week and one summer week zoomed in, and a demand-by-hour /
  demand-by-day-of-week seasonality plot.
- **Deliverable (done):** `data/processed/gb_energy_2020_2025.parquet` (with `lockdown_level`) +
  `notebooks/01_explore_raw_data.ipynb` (data-quality checks + seasonality/winter-summer plots) +
  unit tests for the cleaning functions (`tests/data/test_clean.py`, `tests/test_config.py`) —
  missing-period handling, timezone correctness, `lockdown_level` date-range assignment. **Week 1 is
  complete; Week 2 (baselines) is next.**

## Week 2 — Baselines (done)

- **Baselines (done), `src/edf/baselines.py`:** naive (last observed value, t-1), previous-day-
  same-time (t-48), previous-week-same-time (t-336, seasonal naive — the MASE denominator),
  trailing moving average (mean of the last 4 same-half-hour-of-day values, `skipna=False` so a
  partial-history warm-up row returns NaN rather than a silently weaker average). All pure lags of
  `demand`, so none can leak the future. Tested in `tests/test_baselines.py`.
- **Walk-forward evaluation harness (done), `src/edf/evaluate.py`:** `mae`/`rmse`/`mape`/`mase`,
  `yearly_folds` (per-calendar-year fold boundaries), `evaluate_by_fold`, and `compare_models`
  (scores multiple named prediction series per fold, using one model's per-fold MAE as that fold's
  MASE denominator for every model). Built generically so Weeks 3–7 reuse it rather than
  rewriting scoring logic. Tested in `tests/test_evaluate.py`.
- **Deliverable (done):** `notebooks/04_baselines.ipynb` — per-year and aggregate MAE/RMSE/MAPE/
  MASE table plus a MASE-by-year plot, evaluated only over `TRAIN`+`VALIDATION` (2020–2024);
  `TEST` (2025) stays untouched per the correctness rules.
- **Finding worth carrying into Week 3:** at half-hourly resolution, `naive` (t-1) scores
  MASE ≈ 0.32–0.35 — far beating the seasonal baselines — because GB demand barely moves in 30
  minutes. This is expected, not a result to present as-is: NESO's own day-ahead benchmark
  (`edf.data.demand_forecast_benchmark`) forecasts a full day ahead, not 30 minutes ahead, so a
  future model scored at t-1 lead time would be solving an easier problem and isn't comparable to
  it.
- **Horizon scenario analysis (done), `notebooks/05_baseline_horizons.ipynb`:** the finding above
  was resolved by making forecast horizon an explicit, fixed parameter — `edf.config.HORIZONS`
  (`30min`/`1h`/`1d`/`7d`, in half-hourly periods: 1/2/48/336). `edf.baselines.persistence`
  generalizes `naive` to any horizon (`demand.shift(horizon_periods)`), and
  `edf.baselines.valid_baselines(horizon_periods)` excludes any fixed-lag baseline shorter than
  the horizon (e.g. `previous_day_same_time`, lag 48, is invalid at the `7d` horizon — that data
  wouldn't exist yet at a real week-ahead issue time), adding `trailing_weekly_moving_average` as
  the `7d`-horizon analogue of `trailing_moving_average`. Result confirms the finding was purely a
  horizon artifact: `persistence`'s MASE rises from ≈0.34 (`30min`) to exactly match
  `previous_day_same_time` at `1d` and `previous_week_same_time` at `7d` (MASE = 1.0), since at
  those horizons `persistence` *is* that baseline by construction. **Week 3's LightGBM model
  should be evaluated primarily at the `1d` horizon** — it's the one genuinely comparable to the
  NESO day-ahead benchmark.
- **Experiment tracking (done), `src/edf/tracking.py`:** every (horizon, baseline) run's per-fold
  and mean MAE/RMSE/MAPE/MASE are logged to MLflow, one experiment per horizon (`baselines-30min`,
  `baselines-1h`, `baselines-1d`, `baselines-7d`) so results aren't compared across incompatible
  horizon scales by accident. Local sqlite store at `mlruns/mlflow.db` (gitignored, regenerated by
  rerunning the notebooks — MLflow 3's plain filesystem backend is deprecated, hence sqlite over
  `file:`). View with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`. Tested in
  `tests/test_tracking.py` against a temp sqlite db, not the real store.

## Week 3 — ML (LightGBM)

- **Feature engineering (done), `src/edf/features.py`:** `time_features` (half-hour-of-day,
  day-of-week, month, plus daily/weekly/annual Fourier terms, order 2) — deterministic facts about
  the *target* timestamp, valid at every horizon. `lag_features`/`rolling_features` — lag_1/lag_2/
  lag_48/lag_336 and the Week 2 `trailing_moving_average`/`trailing_weekly_moving_average`
  baselines reused directly as model inputs — filtered by horizon exactly like
  `edf.baselines.valid_baselines` (a lag shorter than the horizon would use data not yet known at
  issue time). `wind`/`solar`/`interconnector` are deliberately excluded per PLAN.md's leakage
  warning (Week 4b fixes this properly). Tested in `tests/test_features.py`.
- **Two strategies for covering all four horizons (done), `src/edf/forecast.py`:**
  - **Direct**: one LightGBM model per horizon, each trained only on that horizon's valid features
    (`build_feature_table`).
  - **Recursive**: one model trained at `30min` (every feature valid there), rolled forward via
    `recursive_forecast` — a vectorized simulator that feeds each step's own prediction back in as
    the next step's short lags, batched across all evaluation points per step (336 steps × ~17.5k
    rows for the `7d` horizon runs in ~13s). `lag_336`/`trailing_weekly_moving_average` never need a
    synthetic value within these horizons (their min lag, 336, exceeds every horizon tested);
    `lag_1`/`lag_2` and `trailing_moving_average`'s shorter terms do. Verified in
    `tests/test_forecast.py` that `recursive_forecast` at `horizon_periods=1` is provably identical
    to the direct model (same features, same order) — the key architectural invariant.
  - Default LightGBM hyperparameters (`DEFAULT_LGBM_PARAMS`), not tuned — walk-forward
    hyperparameter tuning is separate, later work; this comparison answers direct-vs-recursive
    specifically.
- **Deliverable (done):** `notebooks/06_lightgbm_horizons.ipynb` — full comparison table (direct,
  recursive, every valid baseline) per horizon, **scored on `VALIDATION` (2024) only** (unlike
  Week 2's baselines, a fitted model scored on its own `TRAIN` data would be in-sample/optimistic),
  logged to MLflow (`baselines`/`direct`/`recursive` runs now share one `forecast-<horizon>`
  experiment per horizon, so the whole comparison is queryable together — `edf.tracking.log_model_run`
  gained an `extra_tags` param for the `strategy` dimension).
- **Result — recursive is competitive at one extra step, then falls behind:**

  | horizon | direct MASE | recursive MASE |
  |---|---|---|
  | `30min` | 0.105 | *(= direct)* |
  | `1h` (2 steps) | 0.180 | **0.172** (recursive slightly better) |
  | `1d` (48 steps) | 0.623 | 0.634 |
  | `7d` (336 steps) | 0.782 | 0.824 |

  At `1h`, recursive edges out direct: with only one compounding step, the cost is negligible, and
  the recursive base model gets to use `lag_1`/`lag_2` (via its own prediction as a proxy) that the
  direct `1h` model has to do without entirely (too short a lag to be valid at that horizon). By
  `7d`, 336 steps of compounding catch up with it. **Practical takeaway: use direct per-horizon
  models for `1d`/`7d`; recursive is a reasonable, cheaper choice at `1h` if only one model is
  wanted.** More importantly, **both strategies beat every baseline at every horizon**, including
  `7d` (best baseline MASE = 1.0 by construction; direct LightGBM MASE ≈ 0.78) — the model is
  learning real structure, not just leaning on the same lags the baselines already use.
- **Feature importance (`1d` direct model):** dominated by **annual-seasonality Fourier terms**, not
  `lag_336`/`hour_of_day` as originally guessed — consistent with this project's own weather
  exploration finding that temperature is the single biggest external driver of GB demand, so a
  sensible result even though it wasn't the anticipated one. `lag_48` (same lag as the horizon
  itself), weekly structure (`fourier_weekly_*`/`day_of_week`), and both trailing-average baseline
  features rank next; `lag_336`/`hour_of_day` matter but rank lower than expected.
- **Hyperparameter tuning (done), `src/edf/tuning.py`, `notebooks/07_lightgbm_tuned.ipynb`:**
  randomized walk-forward CV search (`num_leaves`/`learning_rate`/`min_child_samples`, 10 trials)
  entirely inside `TRAIN` (2020-2023) — expanding-window folds (train 2020→validate 2021, train
  2020-21→validate 2022, train 2020-22→validate 2023), each candidate's tree count chosen by early
  stopping *within* its CV fold, with the final model's `n_estimators` fixed from the average
  stopped iteration across folds before it's ever retrained on the full `TRAIN` set. `VALIDATION`
  is never touched during search — early-stopping directly against it would let the exact set used
  for final reporting quietly influence model selection.
  - **Two more metrics added, `edf.evaluate`:** `bias` (mean signed error — MAE/RMSE only see error
    magnitude, this catches systematic over/under-forecasting) and `p95_abs_error` (a cheap
    tail-risk read, ahead of Week 6). `edf.tracking.log_model_run` gained `extra_params`/
    `extra_metrics` for logging tuned hyperparameters and wall-clock tune/train time.
  - **Tuning helped direct models at every horizon** (MASE: `30min` 0.105→0.102, `1d` 0.623→0.606,
    `7d` 0.782→0.754) but **made the recursive strategy worse at `1d`/`7d`** (0.634→0.652,
    0.824→0.863) — only `1h` (effectively still one step) improved. **Finding worth keeping, not a
    bug:** the tuning objective was the `30min` model's own one-step CV error, which has no reason
    to favor hyperparameters that are also robust to being recursively chained — the winning
    config fits the one-step problem more tightly, plausibly leaning harder on `lag_1`/`lag_2`,
    exactly the features that turn synthetic soonest under recursion. Reinforces (with a second,
    independent reason beyond compounding error) that direct per-horizon models are the better
    choice past a couple of steps.
  - **Bias/tail metrics reveal structure MASE alone hides:** `7d` direct is the most *biased*
    strategy (≈ -116 MW, consistently over-forecasting) despite having the better MASE of the two
    `7d` strategies; `7d` recursive is far less biased (≈ -22 MW) but has a fatter tail
    (P95 ≈ 5133 MW vs. ≈ 4102 MW) — scattered rather than systematically skewed errors. At `1h`,
    recursive wins on both MASE *and* tail error — a clean win there, not just a marginal one.

## Week 4 — Weather

- Weather data (done, Week 1): population-weighted temperature, wind speed, cloud cover, and
  shortwave radiation from the 4 sources in `src/edf/data/weather.py`.
- Add heating/cooling degree-day features (temperature is normally the single biggest external
  driver of demand).
- Run an ablation: identical model/split with vs. without weather features, using the Open-Meteo
  Historical (ERA5) series as the primary "observed weather" feature set. Report the % MAE/RMSE
  reduction, not just "it got better."
- **A second, more honest ablation this project can actually run (most can't):** repeat the
  comparison on the 2024-03-07-onward slice using the Open-Meteo **Previous Runs API** (`_previous_day1`
  — a genuine ~24h-ahead forecast, not reanalysis) instead of the ERA5 series. (Corrected 2026-09-06:
  the Historical Forecast API is *not* this — its own docs say it "stitches together the first few
  hours of each model run," making it a short-lead nowcast, not a day-ahead forecast; it's kept as a
  4th, separately-labelled source, but the day-ahead comparison uses the Previous Runs API only.) The
  gap between the ERA5-based improvement and the real-day-ahead-based improvement is a direct,
  measured answer to "how much of Week 4's gain survives contact with what a forecaster would
  actually have known at lead time" — turning the Week 1 "weather is observed, not forecast"
  limitation into an honest reported number for the 2024-03-07-onward slice, rather than leaving it
  as an unaddressed caveat.
- **Pre-req found in `notebooks/02_explore_weather_data.ipynb` (2026-09-06), fixed (2026-09-07):**
  naively, the day-ahead source's error against ERA5 came out *lower* than the nowcast source's —
  the opposite of what more lead time should give. Cause confirmed: both endpoints defaulted to
  "Best Match", which Open-Meteo's docs say resolves to "the most suitable high-resolution model"
  *per endpoint* — not guaranteed to be the same model, so the comparison conflated lead time with
  model choice. **Fixed** by pinning both fetchers to an explicit `models="ecmwf_ifs025"`
  (`src/edf/data/weather.py`, `PINNED_MODEL`) — re-checking against live data now gives the
  expected ordering (nowcast MAE 0.374°C vs. ERA5, day-ahead MAE 0.546°C — day-ahead genuinely
  worse, as more lead time should give). Side effect: pinning shrank the nowcast archive's own
  usable window from 2022-03-01 to **2024-03-06** (Open-Meteo's archived history for one fixed
  model doesn't reach as far back as its always-pick-the-best selection does) — re-verified
  empirically, same null-scanning method as the original Week 1 check.
  `DAY_AHEAD_ARCHIVE_START` (2024-03-07) was unaffected, so the honest-ablation slice below doesn't
  shrink.
- **Hypothesis stated before running (done):** temperature isn't represented anywhere in the Week 3
  feature set, so adding it (directly, plus heating/cooling degree-day features,
  `src/edf/weather_features.py`, UK convention 15.5°C/22°C bases) should measurably reduce `1d`-ahead
  error. Confirmed, and by more than expected — see below.
- **Deliverable (done):** `notebooks/08_weather_ablation.ipynb`, `1d` horizon, tuned direct models
  (same walk-forward-CV procedure as `notebooks/07`), scored on `VALIDATION` only.
  - **Ablation A (full `VALIDATION`):** ERA5 weather cuts MAE by **27.65%** (MASE 0.606 → 0.439) —
    the single largest accuracy gain found anywhere in this project so far, bigger than the
    direct-vs-recursive choice or hyperparameter tuning.
  - **Ablation B (honest, `2024-03-07`-onward slice):** reuses the *same* ERA5-trained model, fed
    day-ahead-forecast weather instead of ERA5 at inference time (no way to train a day-ahead-weather
    model — the archive only starts 2024-03-07). ERA5 gives 25.84% MAE improvement on this slice,
    day-ahead gives **26.79%** — nearly all of the gain survives contact with realistic
    forecast-quality weather. The day-ahead-fed evaluation actually scored marginally *better* than
    ERA5-fed on this slice; the `bias` metric explains why (weather-on-ERA5 bias ≈ -191 MW,
    weather-on-day-ahead ≈ -7 MW) — plausibly a coincidental bias cancellation specific to this
    ~10-month window (day-ahead forecasts are numerically smoothed relative to ERA5 reanalysis),
    not evidence that forecast-quality weather beats hindsight weather in general. **The robust
    conclusion is the narrower one: the 25.84%-vs-26.79% gap is small, so the weather gain is real,
    not a hindsight-only artifact** — a stronger, more defensible claim than "day-ahead wins."
  - **Feature importance:** `shortwave_radiation_wm2` (rank 5/30) and `wind_speed_ms` (rank 7/30)
    both rank *above* `temperature_c` (13/30) and `heating_degree` (17/30) — not what the
    temperature-only hypothesis predicted. Ties back to Week 1's embedded-generation-netting note:
    `demand` already has embedded wind/solar invisibly netted in, and while `wind`/`solar` actuals
    are correctly never used as features (same-period, not forecastable), weather variables that
    *correlate* with embedded generation (wind speed with embedded wind, shortwave radiation with
    embedded solar) are legitimate forecast-based proxies for that suppression effect — and
    evidently a strong one. `cooling_degree` ranks lowest (24/30), as expected for a
    heating-driven, not cooling-driven, climate.
  - **Pre-req fix (done):** see the model-pinning note above — resolved before trusting this ablation.

## Week 4b — Forecasting the Inputs (fixes the wind/solar/interconnector leakage)

Motivated by a real gap found 2026-09-06: `wind`/`solar`/`interconnector` are used as demand-model
features, but historically they're only available as same-period *actuals* — no free historical
forecast archive exists for them (unlike weather). The fix is to **forecast them ourselves** from
data that genuinely is available ahead of time, and feed *those* forecasts into the demand model
instead of the outturn values — this is also simply how a real operational forecasting pipeline
would be built (chained/cascaded forecasts), not a project-specific workaround.

- **Wind**: generation is a strongly nonlinear (roughly cubic below rated wind speed, flat at rated
  power, then a hard cutoff at very high wind speed) function of wind speed. Model
  **capacity factor** (`EMBEDDED_WIND_GENERATION / EMBEDDED_WIND_CAPACITY`), not raw MW — capacity
  grows steadily over 2020-2025 as new wind farms connect, and that trend has nothing to do with
  whether it was windy, so predicting raw MW directly would conflate the two. Train
  `capacity_factor ~ f(day-ahead wind speed forecast, calendar)`, then
  `wind_forecast = predicted_capacity_factor × capacity` (capacity itself is slow-moving and safe to
  take as known — it's public information about connected generation, not something requiring a
  forecast).
- **Solar**: same idea, but solar has a strong deterministic ceiling from solar geometry (zero at
  night, a hard maximum set by time of day/year and solar elevation angle, regardless of weather) on
  top of the weather-driven component (cloud cover, irradiance forecast). This ceiling makes solar
  generally *more* predictable than wind given a forecast. Model
  `capacity_factor ~ f(day-ahead shortwave radiation forecast, cloud cover forecast, solar elevation
  angle)`, then multiply by `EMBEDDED_SOLAR_CAPACITY`.
- **Interconnector flow is the hard one, and likely stays a stated limitation rather than a solved
  problem.** Flow is driven mainly by cross-border *price* differentials (GB vs. France/Netherlands/
  Norway/Ireland generation costs, carbon prices, scheduled link maintenance) — a multi-country
  energy-economics problem, not primarily a weather one, so the wind/solar approach doesn't transfer.
  The better fix isn't modelling it from scratch: **ENTSO-E's Transparency Platform publishes
  real day-ahead *scheduled* cross-border commercial exchanges**, which would be a far more accurate
  day-ahead interconnector feature than anything built here — flagged as a possible future data
  source, not committed to, since ENTSO-E access needs a (free) registration and is a new API surface
  to integrate. If not pursued, use a seasonal-naive interconnector forecast and say so plainly in
  the limitations — interconnector flow is a smaller contributor to demand variance than temperature
  or wind/solar, so a weaker feature here is a lower-cost simplification than it would be elsewhere.
- **Evaluation**: this is a forecasting sub-problem in its own right — evaluate the wind/solar
  capacity-factor models with their own MAE/RMSE against a seasonal-naive capacity-factor baseline,
  *before* plugging them into the demand model. Then rerun the Week 3/4 demand model with
  self-forecasted wind/solar (and interconnector, however handled) instead of outturn values, on the
  2024-03-07-onward slice where a genuine day-ahead weather forecast exists to drive it, and compare
  against the outturn-based (leaky, upper-bound) version.
- **Deliverable:** wind/solar capacity-factor forecast models + their own accuracy metrics, plus a
  demand-model comparison table: outturn-based features (upper bound) vs. self-forecasted features
  (realistic), with the interconnector-handling choice stated explicitly either way.
- **Canonical table (done):** `wind_capacity`/`solar_capacity` added (`src/edf/data/clean.py`).
  Observed: embedded wind capacity is nearly flat (6515-6606 MW, ~1.4% growth 2020-2025) while
  embedded solar capacity grew ~70% (13040-22126 MW) — consistent with most new GB wind capacity
  this period being offshore/transmission-connected (not embedded), while new solar is
  predominantly small-scale/distribution-connected.
- **Wind capacity-factor model (done), `notebooks/09_wind_capacity_factor.ipynb`:**
  `src/edf/generation_features.py` builds `capacity_factor ~ f(wind speed, calendar)` — no lag/
  rolling features, since generation is weather-driven not demand-driven (per PLAN.md's formula).
  Same TRAIN-on-ERA5 / honest-check-on-day-ahead-slice structure as Week 4 (day-ahead wind speed
  only exists from 2024-03-07, zero overlap with `TRAIN`).
  - **Result:** MAE 0.0827 vs. the seasonal-naive capacity-factor baseline's 0.1979 — a **58%
    reduction** (MASE 0.418), a bigger relative gain than weather gave the demand model (27.65%).
    Makes sense: weather has almost no weekly pattern to exploit, so "same half-hour last week" is
    a much weaker prior for wind than for demand.
  - **Honest check:** the day-ahead-vs-ERA5 pattern from Week 4 repeats (day-ahead-fed MAE 0.0803
    vs. ERA5-fed 0.0818, marginally better) but this time *without* a large bias gap explaining it
    (0.024 vs. 0.021, both small) — more likely genuine slice-specific noise than a systematic
    effect. Conclusion stands either way: the gap is small, so the gain is real, not hindsight-only.
  - **Visual finding:** the model tracks *when* wind picks up/drops correctly but visibly
    underestimates the *height* of high-wind peaks (e.g. one actual peak ≈0.61 vs. predicted
    ≈0.53) — masked by a small-looking average bias (0.014) because it's averaged over the whole
    distribution. Plausible cause: population-weighted weather (a documented Week 1 simplification)
    likely understates wind speed at actual (rural/coastal) wind-farm sites relative to city
    centres, especially during high-wind events. Noted as a real limitation, not fixed now.
- **Solar capacity-factor model (done), `notebooks/10_solar_capacity_factor.ipynb`:**
  `edf.generation_features.solar_elevation_deg` computes solar elevation directly from
  timestamp + a population-weighted GB centroid (standard declination/hour-angle formula, no new
  dependency, no API call) — a deterministic ceiling feature, verified against the textbook
  solstice-noon check (90 − lat ± 23.45°) and against midnight being exactly 0. Combined with
  `cloud_cover_pct`/`shortwave_radiation_wm2` (weather-driven component) and the existing
  `solar_eclipse_pct` canonical column.
  - **Result:** MAE 0.0162 vs. the seasonal-naive baseline's 0.0383 — a **57.7% reduction** (MASE
    0.422), essentially the same relative gain as wind (58%), but a visibly different *fit
    character*: night-time zeros are captured exactly and daytime peaks track actual generation
    tightly (including day-to-day cloud variation), unlike wind's peak-underestimation. Confirms
    the stated hypothesis — the deterministic ceiling genuinely makes solar easier to fit well
    than wind, given a forecast.
  - **Honest check — the one case where day-ahead genuinely underperforms ERA5:** MASE 0.422
    (ERA5) → 0.470 (day-ahead), an 11.3% relative degradation, unlike the demand model and wind
    (both showed day-ahead marginally *beating* ERA5, traced to noise/bias-cancellation). This is
    the textbook-expected direction, and it's a useful cross-check that those two "surprising"
    results weren't an evaluation-methodology bug — the same method gives the expected answer
    here. Plausible physical reason: cloud formation is far more short-timescale/chaotic than
    large-scale wind patterns, so a 24h-ahead cloud-cover/irradiance forecast has more genuine
    skill to lose than a 24h-ahead wind-speed forecast. Still small next to the seasonal-naive
    baseline (1.0) — most of the gain survives, just not "almost all" the way it did for wind/demand.
  - **Metric pitfall confirmed:** `mape` is `inf` for this target (solar capacity factor is
    exactly zero for roughly half of every day, and `mape` divides by the actual value) — worse
    than wind's near-zero-but-nonzero case. MAE/RMSE/MASE are the only trustworthy numbers here;
    worth remembering for Week 5's probabilistic work if other zero-heavy targets come up.
- **Does self-forecasted wind/solar actually help the demand model? (done),
  `notebooks/11_wind_solar_demand_impact.ipynb`:** three `1d` demand models, evaluated on the
  identical `2024-03-07`-onward slice with day-ahead weather: no wind/solar (Week 4's baseline),
  outturn wind/solar as features (leaky upper bound), self-forecasted wind/solar (from
  `notebooks/09`/`10`, converted capacity factor → MW) as features (the deployable version).

  | model | MAE | MASE | improvement vs. no-wind/solar |
  |---|---|---|---|
  | no wind/solar | 877.5 | 0.472 | — |
  | outturn wind/solar | 680.6 | **0.366** | **22.4%** |
  | self-forecast wind/solar | 876.7 | 0.472 | **0.09%** |

  **The theoretical ceiling is real (22.4%) — genuine signal beyond what the Week 4 weather
  proxies already capture — but the self-forecast version recovers essentially none of it**
  (0.09% is noise, MASE identical to three decimals). Two compounding reasons, both visible in
  the notebook: (1) the self-forecast itself is too noisy to help — wind's self-forecast runs
  systematically low across an entire sample week, not just at peaks (echoing `notebooks/09`'s
  visual finding); (2) some of the outturn gain may reflect information no weather-driven model
  could ever predict (curtailment, mechanical outages, dispatch decisions), not just imperfect
  wind-speed-to-generation modelling. **This closes the loop from Week 4: weather variables
  already proxy for most of embedded generation's weather-explainable effect on demand, so
  building a dedicated (imperfect) generation forecast on top doesn't pay for itself.**
  **Recommendation: don't add self-forecasted wind/solar to the deployed demand model** — keep
  the Week 4 weather features, which already capture this signal more cheaply and more reliably.
  Also flagged a real pitfall: `wind`/`solar` rank highly (4/32, 7/32) in the outturn-trained
  model's feature importance, which reads as "wind/solar matter a lot" — true only for the
  outturn-fed model, and says nothing about the deployable (self-forecast) version's actual
  accuracy. The comparison table, not the importance chart, is the number that matters for a
  deployment decision.
- **Interconnector:** left as a stated limitation, not modelled, per the plan above — flow is
  price-driven, not weather-driven, so the wind/solar approach doesn't transfer, and the smaller
  contributor-to-demand-variance argument (vs. temperature/wind/solar) makes this an acceptable
  simplification to state rather than solve.

## Week 5 — Probabilistic Forecasting

- **Done, `notebooks/12_probabilistic_forecasting.ipynb`:** LightGBM `quantile` objective at
  `1d` horizon, same feature set as Week 4's weather-on(ERA5) model. Hyperparameters tuned once
  via walk-forward CV at α = 0.5 (pinball loss at 0.5 is exactly `0.5 × MAE`, so the existing
  MAE-based `edf.tuning` harness is *already* correct pinball-loss tuning at that alpha, no code
  changes needed) and reused across every other alpha — a stated simplification, not tuning each
  quantile separately.
- **Quantile crossing (done), `src/edf/quantile.py`:** 4.7% of `VALIDATION` rows have P10 > P50
  or P50 > P90 before correction — a direct consequence of training each quantile as a fully
  independent model. Fixed via the standard rearrangement trick (sort each row's predicted
  values, reassign to the correspondingly-ordered alphas). Every metric below reflects the
  corrected predictions.
- **Calibration metrics (done), `src/edf/evaluate.py`:** `pinball_loss`, `quantile_coverage`,
  `picp`, `interval_sharpness`.
  - **PICP for the nominal-80% [P10, P90] interval: 65.2%** — meaningfully overconfident. Not a
    tuning artifact: an untuned check earlier showed essentially the same gap (~67%), so proper
    walk-forward-CV tuning didn't fix it — this is a structural property of independent
    per-quantile LightGBM on this data, not a hyperparameter problem.
  - **Reliability diagram shows a textbook tail-compression pattern**, not vague miscalibration:
    observed coverage sits *above* the diagonal for α < 0.5 (e.g. α=0.05 → observed 0.140) and
    *below* it for α > 0.5 (α=0.95 → observed 0.898), crossing almost exactly at the median
    (α=0.4-0.6 within ~3 points of nominal). **The model knows the center of the distribution
    well but systematically compresses both tails toward it** — a well-documented failure mode of
    independent per-quantile tree boosting (no mechanism ties extreme quantiles to the full
    distribution shape, and boosted trees under-extrapolate into the sparser regions extreme
    values occupy). Confirmed visually in the one-week fan chart: actual demand repeatedly hugs
    or dips below the P10-P90 band's lower edge during down-slopes.
  - **This is a direct preview of Week 6's own stated expectation** ("calibration tends to break
    down in the tails") — Week 5 already shows exactly that pattern, before any explicit
    extreme-day bucketing.
- **Deliverable (done):** calibration plots (reliability diagram + fan chart) + the plain
  statement PLAN.md asks for: **overconfident, specifically and substantially in both tails,
  well-calibrated near the median.** Not fixed here — a genuine fix (conformal prediction, an
  explicit distributional model) is exactly the stretch-goal comparison already listed below
  ("LightGBM quantile regression vs. a distributional alternative... for calibration
  robustness") — this result is what motivates that stretch goal, not a gap to paper over.

## Week 6 — Extreme Events (done)

- **Day-type buckets (done), `src/edf/buckets.py`:** six independent boolean flags (not a
  mutually-exclusive partition), thresholds in code — `is_cold`/`is_hot` (month-relative
  temperature percentile, 10th/90th, so "cold" means unusually cold *for the time of year*, not
  just "a winter day"), `is_low_wind`/`is_high_wind` (overall wind capacity-factor percentile —
  no month-relativity needed, these exist to capture genuinely extreme embedded-generation
  periods), `is_christmas` (Dec 24-Jan 1, wraps the year boundary), `is_weekend`.
- **Per-bucket evaluation harness (done), `src/edf/evaluate.py`:** `evaluate_by_bucket`
  (MAE/RMSE/MASE/bias, using each bucket's *own* seasonal-naive MAE as its MASE denominator, not
  one overall value) and `evaluate_quantiles_by_bucket` (pinball loss per quantile + PICP).
  Evaluated on `notebooks/13_extreme_events.ipynb`'s `1d` point model and Week 5's quantile
  models — reusing their already-tuned hyperparameters, no new tuning.
- **Point accuracy holds up on weather extremes** (MASE close to or better than the 0.44 overall
  figure for `is_cold`/`is_hot`/`is_low_wind`/`is_high_wind`) but **the model systematically
  over-forecasts on every demand-suppressing extreme** — `is_hot` (bias -560 MW), `is_high_wind`
  (-796 MW), `is_christmas` (-1242 MW) — while demand-boosting/neutral buckets show small
  positive bias. Only `is_christmas`/`is_weekend` (calendar-, not weather-driven) are clearly
  harder on MASE.
- **PICP collapses exactly where bias is worst** — `is_hot` PICP 0.558, `is_christmas` PICP
  0.426, both far below the already-low 0.652 overall (`notebooks/12`). **Confirms PLAN's
  anticipated pattern exactly, and reveals the mechanism**: a systematic over-forecast pulls the
  *entire* quantile band upward, so actual demand falls below even P10 more often — bias and
  tail-miscalibration are the same finding, not two separate ones. `is_low_wind` (near-zero
  bias) is the one bucket *better* calibrated than average (PICP 0.709), consistent with this.
- **COVID case study (done):** the naive-only within-2020 before/after comparison is misleading
  (mixes the COVID effect with the ordinary Feb→May seasonal trend that always weakens
  `previous_week_same_time`) — the correct comparison is the same calendar window across years.
  That isolated comparison shows **naive genuinely blindsided** (2020 MAE 3101 vs. 2396/2511 in
  2022/2023, ~25-30% worse — a real, isolated COVID effect) but **the point model essentially
  unaffected** (674 vs. 634/667 — within normal year-to-year variation), contradicting PLAN's
  prior expectation ("every method... was blindsided") for the ML model specifically. **Honest
  reason, not a win**: the model is trained on 2020-2021 data, so it had the chance to fit this
  exact historical pattern directly — unlike a live forecaster with zero warning before 23 March
  2020. This is the case-study limitation PLAN.md itself already flags ("not a held-out accuracy
  number"), now made concrete. Visually confirmed: naive stays pinned near pre-lockdown levels
  for ~1 week after the announcement (its own "last week" reference hasn't seen the lockdown
  yet) while the point model tracks the collapse almost immediately — a purely mechanical,
  train/test-independent limitation of persistence-based forecasting.
- **Deliverable (done):** `notebooks/13_extreme_events.ipynb` — per-bucket metrics tables +
  commentary + the COVID case study with plots, ready for Week 8.
- **Follow-up: can the extreme-bucket bias be fixed? (done), `notebooks/14_extreme_bucket_bias_fix.ipynb`:**
  two targeted interventions — an `is_christmas` feature (Dec 24-Jan 1, broader than the
  existing bank-holiday flags) and 3x sample-weighting (`edf.forecast.train_lightgbm` gained a
  `sample_weight` param) on `is_hot`/`is_high_wind`/`is_christmas` training rows — compared via
  4 point-model variants (baseline/+feature/+weighting/+both).
  - **Each lever fixes its own bucket, and they don't stack additively.** The `is_christmas`
    feature is the effective lever for Christmas bias (-1242→-1025 MW, 17.5% reduction;
    weighting alone barely moves it, -1242→-1208); weighting is the effective lever for
    `is_hot`/`is_high_wind` (-560→-501, -796→-686, ~11-14% reduction; the feature alone barely
    moves these). Combining both is never the single best for any one bucket — a compromise
    between the two specialised fixes. Overall accuracy barely moves throughout (MAE 884-891,
    MASE 0.44) — the fixes are targeted, not a free general improvement.
  - **Re-running the combined recipe for the quantile models found something more important than "did it work": pinball loss improved in every bucket, but PICP got *worse* in exactly the three targeted buckets** (`is_hot` 0.558→0.529, `is_high_wind` 0.609→0.583,
    `is_christmas` 0.426→0.384) even as it improved overall (0.652→0.666). Reducing bias shifts
    the whole quantile band toward the true centre (genuinely lower pinball loss) but if the
    band's width doesn't widen correspondingly, a tighter-but-imperfectly-centred interval can
    cover the truth *less* often — a concrete, instructive demonstration of why pinball loss and
    PICP are evaluated separately (Week 5) rather than either alone standing in for calibration.
    **Conclusion: this recipe is worth keeping for the point model, but does not fix the
    probabilistic model's calibration problem** — that remains open, and points more clearly at
    the stretch goal below (a genuinely distributional model / conformal prediction) as the right
    next step for calibration specifically, separate from bias-correction.
  - **Isolated which lever actually caused the PICP regression**, via the same 4-way split
    applied to the quantile models: `is_hot` PICP — A 0.558, B(+feature) 0.562, C(+weighting)
    **0.515**, D(+both) 0.529; `is_high_wind` — A 0.609, B 0.590, C **0.570**, D 0.583;
    `is_christmas` — A 0.426, B 0.414, C 0.414, D **0.384**. **Sample weighting is the dominant
    cause** for `is_hot`/`is_high_wind` (a larger regression alone than the feature causes); the
    `is_christmas` feature independently hurts `is_christmas` PICP too, despite being the lever
    that *helped* its point bias. Neither lever is additive when combined — D is the worst
    outcome for `is_christmas` specifically. Both levers *improve* overall PICP (0.652→0.664-
    0.666) — calibration is being redistributed away from the extreme buckets toward the
    majority of rows, the opposite of what was wanted. Plausible mechanism: fitting these rows'
    point value more precisely also makes the model more "confident" (narrower) about them
    specifically, since it now effectively sees more of them during training.
  - **Decision: keep both interventions for the point model** (10-18% bias reduction in the
    targeted buckets, no cost to overall accuracy). **Do not apply either to the quantile/
    probabilistic model** — both measurably worsen calibration in exactly the buckets (extreme
    events) where good calibration matters most, even though they improve the training objective
    (pinball loss) and the aggregate PICP number. The metric that's easy to optimize and the
    metric that actually matters point in different directions here, and the latter wins.

## Week 7 — Renewable / Net Demand (done)

Originally planned as `net_demand = demand - wind - solar`. That's now known to be wrong for this
dataset: `wind`/`solar` here are *embedded* generation, already invisibly netted into `demand`
(`ND`) by physics before NESO ever sees it (see the Week 1 / Data sources note). Subtracting them
again double-counts the same effect and produces a number without a clean physical meaning — it is
**not** the GB analogue of the CAISO "duck curve" net-load concept.

- **Done, `notebooks/15_renewable_net_demand.ipynb`:** bucketed `VALIDATION` by combined embedded
  `wind + solar` output into deciles (`edf.buckets.percentile_bin_buckets`, new — Week 6's flags
  only captured the top/bottom 10%; this Week 7 question needed the *whole* range to check for a
  trend, not just the extremes), evaluated with the same harness as Week 6
  (`evaluate_by_bucket`/`evaluate_quantiles_by_bucket`). Models used are the Week 6 follow-up's
  kept decisions: point model = bias-corrected recipe (`is_christmas` feature + weighting);
  quantile model = untouched Week 5 baseline (the corrected recipe was found to worsen PICP, so
  never adopted there).
  - **Point accuracy degrades smoothly and continuously across the entire range**, not just at
    the extremes: MAE climbs from 558 MW (decile 1, lowest wind+solar) to 1417 MW (decile 10) —
    2.5x worse. Bias crosses zero smoothly around decile 4-5, from +294 MW (under-forecast, low
    renewable) to -1078 MW (over-forecast, high renewable) — a clean, physically coherent
    gradient, not a step change. A sharper, more complete version of Week 6's `is_high_wind`
    finding (bias -796 MW using wind alone): combining wind+solar and looking at the whole
    spectrum shows the degradation spans the *entire* distribution, not just the top decile.
  - **PICP behaves differently — flat until the very top decile, then a cliff.** Deciles 1-9 sit
    in a noisy but roughly stable 0.60-0.72 band; decile 10 alone drops to **0.52**. Calibration
    failure is a concentrated tail phenomenon (the most extreme ~10% of renewable-output
    half-hours), not a trend across above-average renewable output the way point accuracy is —
    an important distinction: the accuracy cost of high renewable output is gradual and roughly
    predictable, but the calibration cost is a genuine cliff at the extreme tail specifically.
  - **Synthesis, no new fix attempted**: both findings trace back to the same root cause already
    identified in Weeks 4/4b/6-follow-up (embedded generation suppresses demand by physics before
    measurement; weather features only partially proxy for it; a dedicated wind/solar forecast
    doesn't close the gap either). This notebook's job was characterizing *where and how* the
    problem shows up, not re-solving it — the calibration piece still needs a structurally
    different fix (the conformal-prediction / distributional-model stretch goal), consistent with
    the Week 6 follow-up's conclusion.
- **Stretch, not pursued:** a literal GB duck-curve/net-demand analysis needs *transmission-scale*
  wind/solar generation (Elexon BMRS fuel-mix data) — out of scope for the core 8-week plan.
- **Deliverable (done):** per-embedded-renewable-decile error table + plots (MAE/bias/PICP vs.
  decile) + the accuracy-vs-calibration shape distinction above.

## NESO Benchmark Comparison (done, ahead of Week 8)

Not a numbered week — a direct answer to "does our model beat what NESO actually forecast?",
using the benchmark collected in Week 1 (`src/edf/data/demand_forecast_benchmark.py`) but never
compared against until now. Done ahead of Week 8 because the report needs this number.

- **Done, `notebooks/16_neso_benchmark_comparison.ipynb`:** NESO's archive isn't half-hourly —
  ~12 "cardinal points"/day (overnight minimum, peaks, etc.), each kept as the earliest of ~2
  same-day publications (the more conservative, furthest-ahead value — a genuine ~1-day-ahead
  comparison). Our current point model (bias-corrected recipe) scored on *exactly* those 4,148
  shared `VALIDATION` timestamps, not our full half-hourly set — comparing different target rows
  would say nothing.
  - **NESO's real forecast beats ours, honestly: MAE 782.97 (MASE 0.41) vs. our 835.75 (MASE
    0.44) — NESO is ~6.3% more accurate.** Stated plainly, not spun: a single-person project
    doesn't beat a national grid operator's professional forecasting operation, and that's an
    expected, credible result. Both share a similar over-forecasting tendency (bias -241 MW NESO
    vs. -94 MW ours) — NESO's edge isn't from being less biased, its typical errors are just
    tighter overall (RMSE 1085 vs. 1115).
  - **The gap is not uniform — it's concentrated almost entirely at the overnight minimum.**
    Broken down by `CP_TYPE`: at troughs (n=857), NESO wins by **31.4%**; at peaks (n=1757), the
    gap nearly disappears (+3.4%); at the broader "other" cardinal points (n=1534), **we're
    actually ~7.9% more accurate than NESO**. Physically sensible: overnight minimum demand is
    disproportionately driven by a smaller number of large, schedulable loads (industrial
    baseload, planned operations) that a system operator with direct administrative visibility
    into consumer schedules can predict far better than a weather/calendar/lag feature model —
    exactly the kind of knowledge no amount of better weather data would recover. Peaks are
    driven by the same broad weather/calendar-driven population behaviour our feature set
    targets, so the gap there is small.
  - **Headline for Week 8's report**: not "we matched NESO" (false) or "our model isn't good
    enough" (wrong takeaway) — the honest, specific claim is *where* a from-scratch model can and
    can't compete with an operator holding direct administrative visibility into demand. A
    concrete "what you'd do with more time" candidate: trough-specific features capturing known
    large-consumer schedules, if such data were ever accessible.

### Follow-up: is any of NESO's trough edge recoverable from free public data?

Direct test of the "what you'd do with more time" candidate above, using **Elexon BMRS
`FUELHH`** (`src/edf/data/fuel_mix.py`, `notebooks/17_fuel_mix_trough_test.ipynb`) — free,
keyless, half-hourly generation by fuel type. Added `PS` (pumped storage, e.g. Dinorwig — the
fuel type most directly tied to *deliberate* trough/peak management), plus `NUCLEAR`/`CCGT`, as
features and retrained the bias-corrected point model.

- **This is a leaky, same-period-actual test, explicitly** — `FUELHH` is outturn generation, not
  a forecast, exactly the same caveat this project already applies to `wind`/`solar`/
  `interconnector`. The point isn't deployability, it's whether genuine signal exists at all
  (same honest-ablation-first pattern as Week 4b).
- **Real but modest: trough MAE 901.71 → 865.84 (~4%), with no cost overall (885.70 → 882.00).**
  Even with this leaky, strictly-more-informative-than-any-deployable-feature version, we're
  still ~40% behind NESO at the trough (vs. ~46% at baseline) — the gap barely narrows. Confirms
  the Week 8 headline above empirically: NESO's real edge is very likely private consumer-schedule
  knowledge, not something recoverable from any currently-available free public dataset.
- **The improvement isn't coming from the hypothesized mechanism.** `PS` — the fuel type the test
  was built around — ranks only 16/34 in feature importance; `NUCLEAR` (3/34) and `CCGT` (4/34)
  dominate instead. Likely because `CCGT` (the system's main dispatched-to-match-demand
  generation) near-tautologically mirrors the *whole* demand curve, so it's useful everywhere,
  while `PS`'s genuine signal — visually confirmed to track each night's trough/peak precisely
  (sharp negative dip exactly at the trough, positive surge at the morning peak) — is only sharply
  informative in one narrow regime and so gets less credit from a loss-minimizing tree.
- **Not pursued further**: a possible next step (sample-weighting trough rows specifically when
  training with fuel-mix features, forcing the model to lean on `PS` over `CCGT`) is noted but not
  attempted — the leaky ceiling already found is far enough below NESO that the expected payoff is
  small.

### Follow-up 2: a genuinely deployable source closes almost all of it — `notebooks/18`

The fuel-mix conclusion above ("not recoverable from any currently-available free public
dataset") turns out to be right about generation-side proxies specifically, but wrong as a
general claim. **Elexon BMRS `NDF`** (`src/edf/data/ndf.py`,
`notebooks/18_ndf_residual_correction.ipynb`) is National Grid ESO's own half-hourly demand
forecast, republished on a rolling basis — a genuine forecast (not a same-period actual like
`FUELHH`), so this one isn't leaky. Kept only the revision still published on the calendar day
before delivery, matching what would genuinely have been available at prediction time.

- **NDF alone is dramatically more accurate than our from-scratch model, across the *entire*
  half-hourly VALIDATION year** (MAE 534.46 vs. our matched-TRAIN baseline's 886.36 — full
  half-hourly comparison, n=17,568), not just at the trough. This is a richer comparison than
  the original NESO benchmark (`notebooks/16`) could do, since that one was limited to ~12
  coarse cardinal points/day (the only resolution the NESO portal archive retains); NDF is
  published half-hourly, so it can be compared against the full VALIDATION set.
- **A residual-correction model (predict `actual − NDF`, add it back) barely helps, and only at
  the trough.** Trough MAE improves only marginally (502.07 → 500.58) with a real bias
  reduction (63.37 → 31.31), but overall it's actually slightly *worse* than NDF alone (534.46
  → 542.58) and reintroduces bias NDF alone didn't have. NDF is already close to as good as it
  gets; our own weather/calendar/lag features don't have much left to add on top of it.
- **Real cost, controlled for fairly**: NDF's archive only starts 2021-06-14, so any model using
  it loses ~1.5 years of TRAIN. The baseline above is retrained on the *same* matched TRAIN
  window for a fair comparison (890.30 trough MAE vs. the original full-TRAIN baseline's
  901.71 — barely different, confirming the shorter TRAIN isn't what's driving NDF's advantage).
- **Caveat on "beating" the `notebooks/16` NESO archive number (618.64 at the trough)**: not a
  clean apples-to-apples claim. NDF's day-ahead value here is the *latest* revision still on the
  day before (a shorter lead than "day ahead" implies); the NESO portal archive instead kept the
  *earliest* of ~2 same-day publications (a more conservative, further-ahead value). Different
  lead-time conventions on nominally the same underlying NESO forecast produce different
  accuracy — the fresher revision naturally looks sharper.
- **Practical framing going forward**: not "beat NESO from scratch," and not "a correction layer
  adds real value" (the original hypothesis here, not borne out) — the honest, useful
  conclusion is that NDF itself is a strong forecast, and what this project's own model can
  still meaningfully add is coverage for the ~1.5 years before NDF's archive starts, or a
  fallback for whenever NDF is unavailable, plus the modest trough-specific bias reduction the
  correction does provide.

### Follow-up 3: other candidate features — one ruled out, one tested and null

Broader "what else could help" pass beyond large events and NDF. Three ideas scoped:

- **Heating/cooling degree-days**: turned out to already be implemented (`weather_features.py`,
  `heating_degree`/`cooling_degree`) and already included in every model built this session via
  `build_weather_feature_table` — not a new idea, already in use.
- **NESO's Demand Flexibility Service (DFS) events**: checked NESO's own live-event data
  directly rather than assuming it'd be useful. The entire 2022/23 archive is 2 live events
  (2023-01-23/24, 5 half-hourly periods total,
  [NESO data portal](https://www.neso.energy/data-portal/demand-flexibility-service-live-events)),
  and reporting confirms winter 2023/24 had no live activations (mild weather) — the year-round
  service only started 2024-11-27, right at the tail of VALIDATION. **Ruled out, not built**:
  VALIDATION (2024) has essentially zero events to evaluate against, so this isn't testable in
  this project's fixed split, regardless of whether the underlying effect is real.
- **Solar elevation angle as a demand feature** (`notebooks/19_solar_elevation_demand_feature.ipynb`):
  reused the already-tested `generation_features.solar_elevation_deg` (previously only used for
  the wind/solar generation model) as a demand feature — free, deterministic, zero new data.
  **Near-null result**: overall MAE is a wash (885.70 → 887.08, marginally worse); at the
  twilight regime specifically (`0 < solar_elevation_deg <= 10`, n=1,913) MAE is unchanged
  (810.70 → 810.14) but bias improves ~19% (-119.31 → -96.86). Feature importance rank 21/32 —
  barely used. Likely explanation: the existing `hour_of_day`/`month` categoricals plus
  daily/annual Fourier terms already encode most of the same seasonal-daylight information, so
  this is largely redundant rather than new signal. **Not adopted** as a standard feature.

**Takeaway across all three follow-ups (fuel-mix, NDF, this one)**: this model's from-scratch
feature set looks close to saturated for what's freely and legitimately available — NDF itself
remains the highest-leverage lever found, not further feature engineering on our own model.

## Week 8 — Publish + Impact Estimate

- GitHub repo: proper README, methodology, results, and an explicit **Limitations** section
  (including the observed-vs-forecast-weather caveat, the embedded-generation-netting caveat, and
  any others found along the way).
- Include the Week 6 COVID lockdown case study as a short discussion section: a concrete, honest
  illustration of a regime shock no forecasting approach could have anticipated, which pairs
  naturally with the Week 5 calibration discussion (no realistically-sized P10/P90 interval "covers"
  a once-in-a-generation demand shock — what matters operationally is fast model re-fitting once a
  regime change is recognized, not pre-emptive coverage of it).
- **Impact back-of-envelope (the EA-relevant part):** using Elexon/NESO balancing-cost data and the
  Carbon Intensity API, sketch a rough, clearly-caveated estimate of what a forecast-error
  reduction of the magnitude found in Weeks 3–5 could plausibly be worth — in £ of avoided
  balancing cost and/or tCO2 of avoided fossil-backup generation. This is explicitly a **directional
  estimate, not a causal claim** — say so, and show the arithmetic so a reader can disagree with an
  assumption rather than just the conclusion.
- Write the short technical report: *Machine Learning for Probabilistic Short-Term Electricity
  Demand Forecasting in Great Britain* — structure: motivation → data → methodology → baseline vs.
  ML vs. probabilistic results → calibration → extreme-event/net-demand findings → impact estimate
  → limitations → what you'd do with more time.

## Stretch (only after Week 8, don't let these creep earlier)

- Compare LightGBM quantile regression against a distributional alternative (e.g. NGBoost or a
  simple Gaussian-process/conformal-prediction baseline) for calibration robustness.
- A literal GB "duck curve" analysis using transmission-scale wind/solar generation (Elexon BMRS
  fuel mix), per the Week 7 note — embedded generation alone can't reconstruct it.
- A tiny Streamlit/FastAPI demo that serves a live P10/P50/P90 forecast.
