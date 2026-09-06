# CLAUDE.md — energy_demand_forecast

This is a personal research/portfolio project: probabilistic short-term GB electricity demand
forecasting. See `PLAN.md` for the week-by-week plan, current phase, data sources, and the
end-of-project impact write-up goal. Read it before starting work each session and check off /
update it as phases complete.

## Stack

- Python (managed with `uv`), package layout under `src/edf/`.
- Core libs: pandas, numpy, lightgbm, scikit-learn, matplotlib, requests, holidays, pyarrow.
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
- No API keys should be needed for the primary data sources (NESO, Open-Meteo, Carbon Intensity API
  are all free/keyless). If a source later requires a key, it goes in a local `.env` (gitignored),
  never hardcoded.

## Interaction style

This is a personal project, so per the user's global instructions: act as a direct collaborator —
implement, don't gatekeep with a Socratic back-and-forth. Still flag it directly when a proposed
approach would introduce time-series leakage or invalidate a comparison — that's a correctness
issue, not a style preference.
