# CLAUDE.md — energy_demand_forecast

This is a personal research/portfolio project: probabilistic short-term GB electricity demand
forecasting. See `PLAN.md` for the week-by-week plan, current phase, data sources, and the
end-of-project impact write-up goal. Read it before starting work each session and check off /
update it as phases complete.

## Stack

- Python (managed with `uv`), package layout under `src/edf/`.
- Core libs: pandas, numpy, lightgbm, scikit-learn, matplotlib, requests, holidays, pyarrow,
  python-dateutil.
- Tests: pytest, under `tests/`, mirroring `src/edf/` structure.
- Lint: ruff.

Run things with `uv run ...` (e.g. `uv run pytest`, `uv run python -m edf.data.download`) rather
than activating a venv manually.

## Repo structure

```
src/edf/          reusable code: data loading/cleaning, features, baselines, models, evaluation
tests/            unit tests, mirrors src/edf/
notebooks/        exploration only — nothing here should be load-bearing for results
data/raw/         downloaded, unmodified source data (gitignored, never commit)
data/processed/   cleaned/feature tables (gitignored, never commit — regenerable from raw + code)
reports/          plots, metrics tables, the final written report
```

Notebooks are for exploration and plots, not for logic used more than once. Once a function is
used a second time (in another notebook, in evaluation, in a later week), it moves into
`src/edf/` with a unit test.

## Time-series correctness rules (the easiest way to quietly invalidate every result)

- **Never shuffle for train/test or CV.** Use expanding-window / walk-forward splits only. If you
  see `train_test_split` or `KFold` with `shuffle=True` anywhere near this data, it's a bug.
- **Respect the fixed split from Week 1** (train/validation/test date ranges, defined once in
  `PLAN.md` / `src/edf/config.py`). Don't let a later week quietly evaluate against test data
  "just to check."
- **Watch for feature leakage from the future**, not just from the split: a lag or rolling feature
  computed using `.shift(-1)` or a centered window will leak the target. Any new feature should be
  checked against "would this value have actually been known at prediction time?"
- **Weather is observed, not forecast**, in this project's data (see `PLAN.md` limitations). Don't
  present weather-augmented results as if they represent deployable production accuracy — they're
  an upper bound. Say so wherever the weather results are reported.
- **Never subtract `wind`/`solar` from `demand` to build a "net demand" series.** They're embedded
  (distribution-connected) generation, already invisibly netted into `demand` (`ND`) before NESO
  ever measures it — confirmed via NESO's own FAQ for this dataset. Subtracting them again
  double-counts the same effect. Use them as model features instead; see `PLAN.md` Week 7 for the
  corrected framing.
- **The Open-Meteo "Historical Forecast API" is a short-lead nowcast (~0-3h), not a day-ahead
  forecast** — its own docs say it stitches together the first few hours of each model run. Don't
  call it a "forecast" without that qualifier; for anything claiming day-ahead accuracy, use
  `fetch_open_meteo_day_ahead` (Previous Runs API, `_previous_day1`) instead.
- **Two verified-empirically date guards in `src/edf/data/weather.py`, both regression-tested —
  don't relax either without re-checking the live API first:**
  `NOWCAST_ARCHIVE_START` (2022-03-01: before this, the nowcast endpoint silently returns identical
  ERA5 values, not real short-lead output) and `DAY_AHEAD_ARCHIVE_START` (2024-03-07: before this,
  `_previous_day1` values are null for at least one variable — `shortwave_radiation_previous_day1`
  starts a month later than `temperature_2m_previous_day1`, so the later date is used everywhere).
- **`wind`/`solar`/`interconnector` are same-period actuals, not available ahead of time in
  deployment, and there is no free historical forecast archive for them** (checked: NESO's own
  7-day-ahead embedded wind/solar forecast is a rolling window with no retained history). Don't use
  them as demand-model features and call the result "deployable" — see `PLAN.md` Week 4b for the
  fix (forecast wind/solar from weather via capacity factor; interconnector is harder and likely
  stays a stated limitation).

## Testing

Per-week deliverables should include tests, not just notebooks — this project treats data cleaning,
feature engineering, and the evaluation harness (metrics, walk-forward splitting) as code that can
silently break and needs regression coverage, same as any other software. At minimum: cleaning
functions (missing periods, outlier handling, timezone correctness), metric functions (MAE/RMSE/
MASE/pinball loss — easy to get subtly wrong), and the walk-forward split generator.

## Data handling

- Nothing under `data/` is committed (see `.gitignore`). If a dataset is small enough to be worth
  version-controlling as a fixture for tests, put a tiny sample under `tests/fixtures/`, not in
  `data/`.
- No API keys should be needed for the primary data sources (NESO, Open-Meteo, NASA POWER, Carbon
  Intensity API are all free/keyless). If a source later requires a key, it goes in a local `.env`
  (gitignored), never hardcoded.
- `src/edf/data/calendar_events.py`'s event table (bank/school holidays, football/TV events,
  Olympics, Clap for Carers, solar eclipses) is deliberately hand-curated, not scraped/API-fetched —
  the event count is small and stable (historical facts don't change), so a small verified table is
  more reliable than a scraper. Extend it the same way: research + cite, don't automate the lookup.

## Interaction style

This is a personal project, so per the user's global instructions: act as a direct collaborator —
implement, don't gatekeep with a Socratic back-and-forth. Still flag it directly when a proposed
approach would introduce time-series leakage or invalidate a comparison — that's a correctness
issue, not a style preference.
