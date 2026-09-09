# Machine Learning for Probabilistic Short-Term Electricity Demand Forecasting in Great Britain

A personal research project forecasting GB electricity demand (point and probabilistic) from free,
public data — walk-forward validated throughout, benchmarked against NESO's own operational
forecasts, and closed out with a directional estimate of real-world impact.

**Read the full write-up: [`reports/technical_report.md`](reports/technical_report.md).**
**Try the live demo: [energy-forecast.sid-silva.com](https://energy-forecast.sid-silva.com/predictions)**
(free-tier host — sleeps after 15min idle, first load can take ~30-60s to wake up).

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

`/predictions` also shows a trailing-12-months chart (actual demand vs. NDF vs. our blended
forecast), precomputed by `uv run python -m edf.models.history_comparison` into
`reports/ndf_comparison_last_year.csv` (committed, like `models_registry/` — the app just reads it,
never recomputes it live). Re-run that command to refresh the chart. See its docstring for why it
retrains a separate, genuinely out-of-sample model rather than reusing the live-serving one, and for
the two caveats it surfaces on the chart (ERA5-hindsight weather as an accuracy upper bound, and
NESO's open advisory on 2026 demand data).

```bash
uv run uvicorn app.main:app --reload   # http://127.0.0.1:8000/predictions
```

### Deploying the demo

Build and run the container locally first:

```bash
docker build -t edf-demo .
docker run --rm -p 7860:7860 edf-demo   # http://127.0.0.1:7860/predictions
```

To deploy for free on [Render](https://render.com) (Docker-native, no card required as of
2026 — Hugging Face Spaces changed its Docker SDK to a paid-only, PRO-subscription feature
mid-2026, so it's no longer a free option for this):

1. Push this repo to GitHub (Render deploys from a connected repo, not a direct `git push`).
2. In the Render dashboard: New → Web Service → connect the repo → Environment = Docker (it
   detects the root `Dockerfile` automatically; no build/start command needed since the
   Dockerfile's `CMD` already binds to Render's injected `$PORT`).
3. Choose the Free instance type and deploy. `models_registry/` is committed to the repo (small,
   joblib artifacts, same "worth version-controlling" reasoning as `calendar_events.py`'s
   hand-curated table), so no separate model upload/training step is needed at deploy time.

Render's free tier sleeps after 15 minutes of inactivity (cold start ~30-60s on the next
request) — an acceptable tradeoff for a low-traffic personal demo, not a real production setup.
[Koyeb](https://www.koyeb.com) is a reasonable fallback if that 750-hour/month cap ever binds,
though its free tier's 0.1 vCPU allocation is tighter for this image's lightgbm/pandas footprint.

**Optional — if live weather 429s recur:** Render's free tier shares its outbound IP pool with
other Render customers, and Open-Meteo's free API rate-limits by client IP, so this app's live
forecast call can occasionally get 429'd by *other* apps' traffic. `weather-proxy/` is a one-file
Vercel function that proxies the same call through a different host/IP; see its README to deploy
it, then set `OPEN_METEO_LIVE_FORECAST_URL` to the deployed URL in Render's Environment tab. The
app calls Open-Meteo directly when this env var is unset, so it's opt-in.

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

Complete: all 8 planned weeks plus follow-up work, technical report and impact estimate written,
live demo deployed at [energy-forecast.sid-silva.com](https://energy-forecast.sid-silva.com/predictions).
See `PLAN.md`'s Stretch section for possible future extensions (not required for this project's
scope).
