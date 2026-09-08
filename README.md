# Machine Learning for Probabilistic Short-Term Electricity Demand Forecasting in Great Britain

A personal research project forecasting GB electricity demand (point and probabilistic) from free,
public data — walk-forward validated throughout, benchmarked against NESO's own operational
forecasts, and closed out with a directional estimate of real-world impact.

**Read the full write-up: [`reports/technical_report.md`](reports/technical_report.md).**

## Headline result

A from-scratch LightGBM model, trained on weather/calendar/lag features alone, does not beat NESO's
own day-ahead demand forecast (NDF) — it loses in every regime checked. But **combining the two
forecasts (a ~17% weight on the from-scratch model, ~83% on NDF) beats NDF alone by 3.55%**,
verified out-of-sample on a full, independent year (2025) after an earlier check on 2024 found a
closely matching 3.75%. A rough, clearly-caveated back-of-envelope estimate puts that at roughly
£74k–£1.65M/year in avoided balancing cost and ~72,600–120,000 tCO2/year in avoided fossil-backup
emissions, depending on how much of GB's balancing-cost bill is assumed sensitive to demand-forecast
accuracy specifically (§8 of the report shows the full arithmetic).

## Quick start

```bash
uv sync                  # installs from pyproject.toml (Python >=3.11)
uv run pytest             # run the test suite
uv run jupyter lab         # open notebooks/ to reproduce any result
```

## Live demo

`src/app` is a small FastAPI app that serves this project's persisted models
(`models_registry/`, built by `uv run python -m edf.models.registry`) against **live** data —
NESO's rolling demand feed, a live Open-Meteo weather forecast, and NESO's current day-ahead
forecast — recomputing a real 30-minute/1-hour/1-day/7-day prediction on every request, not a
canned demo. See `src/app/live_pipeline.py`'s docstring for the caching/refresh design and
`src/edf/models/registry.py`'s docstring for exactly which recipe each horizon uses and why.

```bash
uv run uvicorn app.main:app --reload   # http://127.0.0.1:8000/predictions
```

### Deploying the demo

Build and run the container locally first:

```bash
docker build -t edf-demo .
docker run --rm -p 7860:7860 edf-demo   # http://127.0.0.1:7860/predictions
```

To deploy for free on [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK):

1. Create a new Space, SDK = Docker.
2. Add it as a git remote and push this repo to it (the Space builds the root `Dockerfile`
   automatically — no extra config needed; it already exposes port 7860, which Spaces expects):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. The Space rebuilds on every push. `models_registry/` is committed to the repo (small, joblib
   artifacts, same "worth version-controlling" reasoning as `calendar_events.py`'s hand-curated
   table), so no separate model upload/training step is needed at deploy time.

Data (`data/raw/`, `data/processed/`) is gitignored and regenerated from free, keyless APIs (NESO,
Elexon BMRS, Open-Meteo, NASA POWER, Carbon Intensity API) via the modules in `src/edf/data/` — see
[`PLAN.md`](PLAN.md)'s "Data sources" table for exactly which endpoint backs which feature.

## Repo structure

```
src/edf/          reusable, tested code
  data/             loading, cleaning, source-specific fetchers (NESO, Open-Meteo, ...) plus
                    live_demand.py/live_ndf.py (live counterparts, for src/app)
  features/         feature engineering (demand lags/calendar, weather, generation, buckets)
  models/           baselines, LightGBM models, tuning, quantile, forecast combination, and
                    registry.py (trains + persists the final per-horizon models to disk)
  config.py         fixed train/validation/test split and other project-wide constants
  evaluate.py        walk-forward evaluation harness and metrics
  tracking.py         MLflow experiment-tracking helpers
src/app/          FastAPI live-inference demo (intro, live predictions, report pages)
models_registry/  persisted models (small joblib files, committed) -- src/app's model source
tests/            unit tests, mirrors src/edf/ and src/app/
notebooks/        numbered, sequential analysis — each one a single question, run end to end
reports/          the technical report and a running findings log
data/             gitignored; raw + processed tables, regenerated from source
```

## More detail

- [`reports/technical_report.md`](reports/technical_report.md) — the full write-up: motivation,
  data, methodology, results, calibration, extreme events, the NESO/NDF comparison, the impact
  estimate, limitations, and what's next.
- [`PLAN.md`](PLAN.md) — the complete week-by-week working log this report is distilled from,
  including every negative result, correction, and follow-up along the way.
- [`reports/FINDINGS.md`](reports/FINDINGS.md) — a running log of smaller, concrete exploratory
  findings not central enough for the main report.
- [`CLAUDE.md`](CLAUDE.md) — project conventions and AI-collaboration notes.

## Status

Week 8 (publish + impact estimate) in progress — the technical report and impact estimate are
drafted; repo polish and a final limitations pass remain. See PLAN.md's Week 8 section for the exact
remaining checklist.
