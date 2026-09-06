# Plan — Machine Learning for Probabilistic Short-Term Electricity Demand Forecasting in Great Britain

Last updated: 2026-09-05

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
| Weather (temperature, wind speed, cloud/irradiance) | [Open-Meteo Historical Weather API](https://open-meteo.com/) (free, no key) or ERA5 reanalysis via Copernicus CDS | Use a small set of GB population-weighted stations/grid points (e.g. London, Birmingham, Manchester, Glasgow, Leeds) rather than a single point — GB demand responds to population-weighted temperature, not one city's. |
| UK bank holidays / school holidays | `holidays` Python package + manual list for school holidays | Demand drops sharply on bank holidays; worth encoding explicitly rather than hoping calendar features catch it. |
| Balancing costs / imbalance prices (Week 8 impact estimate) | [Elexon BMRS](https://www.bmreports.com/) / NESO balancing costs data | Used only for a back-of-envelope impact estimate, not as a model input. |
| Carbon intensity | [Carbon Intensity API (National Grid ESO)](https://carbonintensity.org.uk/) | Used for the Week 8 impact estimate — connects forecast error to plausible emissions impact. |

**Known limitation to state explicitly (not to solve):** this project uses *observed* historical
weather as a feature, but an operational forecaster only has *forecast* weather at lead time.
Results here are therefore an upper bound on what weather features could contribute in production
— call this out in the final report rather than presenting it as if it were deployable as-is.

## Week 1 — Data & Project Setup

- Set up the repo (done): `src/edf` package, `tests/`, `data/{raw,processed}` (gitignored),
  `notebooks/`, `reports/`.
- Write a small downloader (`src/edf/data/download.py`) that pulls NESO historic demand data for
  2020–2025 into `data/raw/`, idempotent (skip if already downloaded).
- Build a cleaned, canonical half-hourly (or hourly, pick one and be consistent) table with:
  `timestamp` (UTC, tz-aware), `demand`, `wind`, `solar`, `interconnector`.
- Handle: missing settlement periods, clock-change days (23/25-hour days), obvious sensor/reporting
  outliers, unit consistency (MW throughout).
- Decide and document the train/validation/test split up front, e.g. train 2020–2023, validation
  2024, test 2025 — fixed for the rest of the project so no week accidentally peeks at test data.
- Plot: full time series, one winter week and one summer week zoomed in, and a demand-by-hour /
  demand-by-day-of-week seasonality plot.
- **Deliverable:** `data/processed/gb_energy_2020_2025.parquet` + a data-quality notebook +
  unit tests for the cleaning functions (missing-period handling, outlier rules, timezone
  correctness).

## Week 2 — Baselines

- Implement, as functions in `src/edf/baselines.py`:
  - naive (last observed value)
  - previous-day-same-time
  - previous-week-same-time (seasonal naive — this becomes the MASE denominator)
  - trailing moving average (e.g. mean of last 4 same-time-of-day observations)
- Build the walk-forward evaluation harness once, in `src/edf/evaluate.py` — every later model
  (Weeks 3–7) reuses this, it is not rewritten per week.
- Report MAE, RMSE, MAPE, and MASE for each baseline, per split fold, not just a single aggregate
  number.
- **Deliverable:** a baseline comparison table/plot; this is the number every later week must beat.

## Week 3 — ML (LightGBM)

- Feature engineering in `src/edf/features.py`: calendar features (half-hour-of-day, day-of-week,
  month, UK bank holiday flag), Fourier terms for daily/weekly/annual seasonality, lag features
  (t-1, t-2, t-48 i.e. same time yesterday, t-336 i.e. same time last week), rolling
  mean/std windows.
- Train LightGBM with the Week 1 time-based split; tune via walk-forward CV, not k-fold.
- Compare against Week 2 baselines on MAE/RMSE/MASE, and check feature importance to sanity-check
  the model learned something sensible (e.g. lag-336 and hour-of-day should dominate).
- **Deliverable:** trained model + comparison table + short write-up of which features mattered.

## Week 4 — Weather

- Pull GB population-weighted temperature, wind speed, and cloud cover / irradiance for
  2020–2025 from Open-Meteo/ERA5.
- Add heating/cooling degree-day features (temperature is normally the single biggest external
  driver of demand).
- Run an ablation: identical model/split with vs. without weather features. Report the % MAE/RMSE
  reduction, not just "it got better."
- **This is the first real experiment — state the hypothesis before running it**: e.g. "weather
  should reduce demand-forecast error more than wind/solar-forecast error, since wind/solar are
  already directly observed in the target-adjacent NESO estimates." Then check if that's true.
- **Deliverable:** ablation table + short experiment write-up (hypothesis, result, honest
  interpretation — including if the effect is smaller than expected).

## Week 5 — Probabilistic Forecasting

- Train LightGBM with the `quantile` objective at α = 0.1, 0.5, 0.9 to produce P10/P50/P90.
- Evaluate calibration properly, not just point accuracy:
  - **Pinball loss** per quantile.
  - **PICP** (prediction interval coverage probability) — the fraction of actuals falling inside
    [P10, P90] should be ~80%; report the actual number.
  - A reliability diagram (nominal vs. observed coverage).
  - Interval sharpness (P90–P10 width) — a wide-but-well-calibrated interval is less useful than a
    narrow-and-well-calibrated one, so report both.
- **Deliverable:** calibration plots + a plain statement of whether the model is over- or
  under-confident, and where.

## Week 6 — Extreme Events

- Define day-type buckets explicitly and reproducibly (e.g. cold = mean temp below the 10th
  percentile for that month, hot = above 90th, high/low wind = generation percentile, Christmas =
  fixed date range, weekend = calendar) — put the thresholds in code, not eyeballed.
- Evaluate the Week 3 point model *and* the Week 5 probabilistic model separately per bucket:
  MAE/RMSE/MASE and pinball loss/PICP per bucket.
- The likely interesting finding: point accuracy may hold up reasonably on extreme days, but
  **calibration (PICP) is the one that tends to break down in the tails** — that mismatch, if you
  find it, is worth a dedicated paragraph in the final report.
- **Deliverable:** per-bucket metrics table + commentary on where the model is least trustworthy.

## Week 7 — Renewable / Net Demand

- Compute `net_demand = demand - wind - solar` and forecast it directly (net demand is what
  dispatchable generation actually has to serve).
- Compare net-demand forecast error against the demand-only forecast error, specifically during
  high-renewable-share periods (the GB "duck curve" analogue — fast net-demand ramps around
  sunset/sunrise or high-wind events).
- Connect to a real operational question: **are the largest net-demand forecast errors concentrated
  during periods that matter most for curtailment/balancing decisions?**
- **Deliverable:** net-demand model + analysis of where/when errors concentrate relative to
  renewable share.

## Week 8 — Publish + Impact Estimate

- GitHub repo: proper README, methodology, results, and an explicit **Limitations** section
  (including the observed-vs-forecast-weather caveat from Week 1, and any others found along the
  way).
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
- Swap in real historical weather *forecasts* (if a source can be found) instead of observed
  weather, to get a realistic-deployment error estimate rather than an upper bound.
- A tiny Streamlit/FastAPI demo that serves a live P10/P50/P90 forecast.
