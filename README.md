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
uv run pytest             # run the test suite (134 tests)
uv run jupyter lab         # open notebooks/ to reproduce any result
```

Data (`data/raw/`, `data/processed/`) is gitignored and regenerated from free, keyless APIs (NESO,
Elexon BMRS, Open-Meteo, NASA POWER, Carbon Intensity API) via the modules in `src/edf/data/` — see
[`PLAN.md`](PLAN.md)'s "Data sources" table for exactly which endpoint backs which feature.

## Repo structure

```
src/edf/          reusable, tested code: data loading/cleaning, features, models, evaluation
tests/            unit tests, mirrors src/edf/ (134 tests)
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
