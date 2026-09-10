# Machine Learning for Probabilistic Short-Term Electricity Demand Forecasting in Great Britain

*A personal research project. Code and full week-by-week working notes: [`PLAN.md`](../PLAN.md).
Every result below is reproducible from a numbered notebook in [`notebooks/`](../notebooks/).*

## 1. Motivation

Great Britain's electricity system operator (NESO) forecasts national demand for every
half-hourly settlement period, every day, to balance supply and demand in real time. Forecast
error is not a purely academic inconvenience — it has a direct operational cost: more reliance on
fast-reacting (often gas) balancing reserve, more expensive balancing actions, and in some
conditions more curtailment of renewable generation that would otherwise have been usable. Better
*probabilistic* forecasts — a calibrated range, not just a point estimate — let a system lean less
on conservative fossil backup and more on variable renewables with confidence.

This project asks two linked questions: (1) how much of NESO's forecasting problem can be
reproduced from scratch, using only free, public data and a fairly standard ML toolchain, and (2)
does the answer actually matter, in £ and tCO2 terms, at the accuracy differences found. It is
built as a genuine research exercise — including reporting negative results and correcting earlier
mistakes in the open — not a leaderboard exercise.

## 2. Data

| Source | What it provides | Notes |
|---|---|---|
| [NESO Data Portal — Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data) | Half-hourly national demand, embedded wind/solar generation, interconnector flows, 2020–present | Free, keyless. `demand` = **ND** (National Demand), not TSD — TSD reintroduces pump-storage pumping, a price-driven decision, not a weather/calendar-driven signal. |
| Open-Meteo (4 endpoints) + NASA POWER | Population-weighted (5 GB conurbations) temperature, wind speed, cloud cover, shortwave radiation | ERA5 reanalysis (primary, hindsight), NASA POWER (independent cross-check, 0.98 temperature correlation with ERA5), a genuine ~24h-ahead forecast archive (`_previous_day1`, valid from 2024-03-07), and a short-lead nowcast archive kept separately after an early mischaracterization as "archived forecasts" was caught and corrected. |
| NESO's own day-ahead demand forecast | ~12 coarse "cardinal points"/day, 2018–present | Used only as a benchmark, never as a model input. |
| Elexon BMRS `NDF` | National Grid ESO's own half-hourly demand forecast, republished on a rolling basis, archive from 2021-06-14 | A genuine forecast (kept only the revision published the day before delivery); used both as a benchmark and, later, as one input to a forecast-combination step. |
| `holidays` package + a hand-curated event table | Bank/school holidays, tiered TV/sporting "pickup" events (e.g. Euro 2020, NESO-documented ~1–1.6 GW swings), solar eclipses, a 3-level COVID lockdown flag | Hand-curated because the event count is small and stable — research + citation, not scraping. |
| Elexon BMRS `FUELHH` | Half-hourly generation by fuel type | Used only in a leaky, same-period-actual diagnostic test (never a deployable feature). |
| Carbon Intensity API | GB grid carbon intensity | Impact-estimate input (see §8). |

**Data decisions worth stating explicitly**: `wind`/`solar` (embedded generation) are used as
*features*, never subtracted from `demand` a second time — NESO's own FAQ confirms embedded
generation is already invisibly netted into ND by physics before it's ever measured, so
subtracting it again would double-count the effect. `interconnector` is the sum of all individual
link flows (individual links are highly collinear, all driven by the same GB-vs-Europe price
differential).

*(A date-parsing bug corrupting part of the 2025 data was found and fixed during this project —
see PLAN.md's Follow-up 7 for details; all results below use the corrected data.)*

## 3. Methodology

**Time-aware validation only.** No random train/test splits — expanding-window / walk-forward
splits throughout. The project's splits were fixed once (Week 1) as `TRAIN`=2020–2023,
`VALIDATION`=2024, `TEST`=2025, and later extended forward by one year
(`TRAIN`=2020–2024, `VALIDATION`=2025, `TEST`=2026 provisional) once 2025 had fully elapsed and sat
unused — a genuinely idle year being promoted into service, not a reopening of anything already
evaluated against. `TEST` under either definition has never been evaluated against for model
selection; every number reported below is `VALIDATION`-only unless stated otherwise.

**One comparable scale across every result.** MAE and RMSE are reported alongside **MASE**
(MAE scaled against the seasonal-naive baseline), so results from different weeks and different
feature sets stay comparable.

**Forecast horizon is an explicit, fixed parameter.** An early result (a 30-minute-ahead
persistence forecast trivially beating every seasonal baseline) turned out to be a horizon
artifact, not a real finding — GB demand barely moves in 30 minutes. `1d` (48 settlement periods)
is used throughout the results below: it's the horizon genuinely comparable to NESO's own
day-ahead forecast.

**Model**: LightGBM, direct per-horizon models (found to beat a single recursively-rolled model at
this horizon), with a randomized walk-forward-CV hyperparameter search confined entirely to
`TRAIN`. **Features**: deterministic time/calendar features (half-hour-of-day, day-of-week, month,
Fourier terms at daily/weekly/annual periods, bank/school holidays, event tiers), lag and
trailing-average demand features (filtered to only those valid at the forecast horizon — a lag
shorter than the horizon would use data not yet known at issue time), and weather features
(temperature, heating/cooling degree-days, wind speed, cloud cover, shortwave radiation, and — the
most recent addition — a 1-day trailing cumulative degree-day feature). `wind`/`solar`/
`interconnector` outturn values are deliberately **never** used as model features, since they are
same-period actuals with no free historical forecast archive — using them would misrepresent
deployable accuracy.

## 4. Results: baseline → ML → weather → probabilistic

| model | `1d`-horizon MASE | note |
|---|---|---|
| seasonal-naive (previous week, same time) | 1.00 | MASE denominator by construction |
| LightGBM, calendar+lag features only, untuned | 0.623 | beats every baseline |
| LightGBM, tuned | 0.606 | walk-forward CV inside `TRAIN` only |
| **+ ERA5 weather (heating/cooling degree-days, wind, cloud, radiation)** | **0.439** | **27.65% MAE reduction — the single largest gain found anywhere in this project** |
| + ERA5 weather, honest day-ahead-forecast-weather check (2024-03-07-on slice) | ~equal to the ERA5-fed result | 25.84% (ERA5) vs. 26.79% (day-ahead) — nearly all of the weather gain survives contact with realistic forecast-quality weather |
| + self-forecasted wind/solar (from weather, deployable) | ~no change | 0.09% improvement — the leaky, outturn-fed upper bound was 22.4%, but a from-scratch forecast of wind/solar can't recover it; **not adopted** |

**Significance-tested (§9): is any of this real, or noise?** Re-run independently under a
Diebold-Mariano test (`notebooks/24_significance_testing.ipynb`) — weather-on vs. weather-off,
**p = 6.9e-51**; outturn wind/solar vs. none, **p = 7.7e-16**; self-forecast wind/solar vs. none,
**p = 0.0042** (a genuine update to the "0.09%, noise" framing above — see §9). All three survive
Benjamini-Hochberg FDR correction alongside three other headline comparisons tested together.
| + 1-day cumulative degree-days (this session) | modest overall (~0.2%), real bias fix in extreme buckets | see §6 |

**Weather is the standout lever, and it's mostly a proxy effect, not a direct one.**
`shortwave_radiation_wm2` and `wind_speed_ms` rank *above* `temperature_c` in feature importance —
not what a temperature-only hypothesis predicted. The likely mechanism: embedded wind/solar
generation is already netted into `demand` before it's ever measured, so weather variables that
correlate with that suppression effect (wind speed with embedded wind, shortwave radiation with
embedded solar) are legitimate forecast-based proxies for it, and evidently strong ones.

**Probabilistic forecasting** (LightGBM `quantile` objective, same feature set): the nominal-80%
[P10, P90] interval achieved only **65.2% empirical coverage** (PICP) — meaningfully overconfident,
and not a tuning artifact (an untuned check showed essentially the same gap). The reliability
diagram shows a textbook **tail-compression** pattern: the model knows the centre of the
distribution well but systematically narrows both tails, a well-documented failure mode of
independent per-quantile tree boosting. See §5.

## 5. Calibration

The overconfidence in §4 is not uniform — it concentrates exactly where it matters most.

- **Bias and calibration failure are the same finding, not two separate ones.** The point model
  systematically over-forecasts on every demand-suppressing extreme (`is_hot`: bias −560 MW,
  `is_high_wind`: −796 MW, `is_christmas`: −1,242 MW), and PICP collapses in exactly those buckets
  (`is_hot` PICP 0.558, `is_christmas` 0.426, both far below the 0.652 overall figure) — a
  systematic over-forecast pulls the whole quantile band upward, so actual demand falls below even
  P10 more often.
- A targeted fix (an `is_christmas` calendar feature + 3× sample-weighting on the three worst
  buckets) reduced point-model bias 10–18% in the targeted buckets with no cost to overall
  accuracy — but **made quantile-model calibration *worse* in those same buckets**, even as it
  improved the pinball-loss objective and the aggregate PICP number. Kept for the point model,
  explicitly **not** applied to the quantile model — a concrete demonstration of why pinball loss
  and PICP need to be evaluated separately, not assumed to move together.
- **Renewable-output deciles show two different shapes.** Point accuracy degrades smoothly and
  continuously across the whole wind+solar output range (MAE 558→1,417 MW from lowest to highest
  decile) — a gradual, physically coherent gradient. PICP, by contrast, is flat and stable across
  deciles 1–9 and then falls off a cliff at decile 10 (0.52) — calibration failure is a
  concentrated tail phenomenon, accuracy degradation is not.

**Stretch: does a structurally different model fix the overconfidence, rather than just describing
it?** Two candidates compared head-to-head on `1d`, on a fresh evaluation year (`VALIDATION`=2025):
Conformalized Quantile Regression (CQR — Romano et al. 2019, a post-hoc correction fit on a held-out
calibration year) and NGBoost (Duan et al. 2019 — one Normal distribution boosted per row, so
quantiles come from a single coherent CDF and can't cross by construction). Full comparison in
[`notebooks/23_calibration_robustness_comparison.ipynb`](../notebooks/23_calibration_robustness_comparison.ipynb).

| model | pinball P10 | pinball P50 | pinball P90 | PICP (nom. 80%) | sharpness |
|---|---|---|---|---|---|
| LightGBM quantile (baseline) | 285.35 | **485.91** | 215.33 | **62.6%** | 2,146 MW |
| LightGBM + CQR | **250.42** | 485.91 (untouched) | 216.27 | **75.1%** | 2,775 MW |
| NGBoost | 263.99 | 528.95 | 233.36 | 69.1% | 2,643 MW |

**CQR wins outright.** A single scalar correction, fit on one held-out year and costing nothing to
retrain, lifts PICP from 62.6% to 75.1% — most, not all, of the way to nominal 80% (the shortfall is
plausibly because the calibration year and the evaluation year aren't perfectly exchangeable, the one
assumption CQR's guarantee actually needs) — while *improving* lower-tail pinball loss, not just
widening the interval defensively. **NGBoost, a genuinely different model, lost on every metric here
— for a specific, honest reason, not because the idea is wrong**: its median forecast is 8.9% worse
than LightGBM's because matching LightGBM's walk-forward-CV tuning budget was computationally
infeasible in this project's time budget (one 600-tree NGBoost fit took 862s vs. ~313s for LightGBM's
*entire* tuned quantile ensemble). Its one clean structural win — zero quantile crossing by
construction, vs. LightGBM's 3.6% needing a rearrangement patch — wasn't enough to make up the gap.
**Decision: CQR is now applied to every horizon in the live demo** (`edf.models.registry`,
`app.predict`); NGBoost remains a legitimate idea for later, conditional on a real tuning budget.
**Significance-tested (§9)**: an independent DM test on the coverage-indicator series (baseline
non-coverage vs. CQR non-coverage) gives **p ≈ 4.7e-255** — the most significant of every
comparison tested, expected given CQR's near-uniform per-row correction (see §9).

## 6. Extreme events and the newest feature finding

**COVID-19 lockdown (2020)**, examined as a case study, not a held-out accuracy number (the model
was trained on 2020–2021 data, so it had the chance to fit this exact historical pattern — a stated
limitation of the case study itself). Isolating the correct comparison (same calendar window across
years, not a naive within-2020 before/after) shows naive forecasting genuinely blindsided (~25–30%
worse than the equivalent window in other years) while the point model was essentially unaffected —
visually, the naive baseline stays pinned near pre-lockdown levels for about a week after the
announcement, while the point model tracks the collapse almost immediately.

**Two new feature ideas, sourced from a literature search rather than intuition, tested this
session**: substituting apparent ("feels-like") temperature for dry-bulb temperature in the
degree-day formula was rejected (0.50% worse); a **1-day trailing cumulative heating/cooling
degree-day feature** was adopted — a small overall MAE gain (0.23%) but a real, broad bias
reduction concentrated in exactly the buckets already known to be weak (`is_hot`: 14.5% smaller
bias, `is_high_wind`: 5.6%, `is_christmas`: 12.6%), the first improvement this project has found in
several rounds of feature research. Longer cumulative windows (2–7 days) made things worse,
monotonically — the literature's "longer persistence helps" framing didn't transfer to a model that
already carries explicit multi-week demand-lag features.

## 7. Does this beat NESO — and the forecast-combination result

**No, not from scratch, and the honest comparison is worse than an early, coarser check
suggested.** Against NESO's own portal archive (~12 cardinal points/day), the from-scratch model
lost overall (~6.3%) but *won* at "other" (non-peak, non-trough) cardinal points by 7.9%. Re-checked
against Elexon's `NDF` — a denser, fresher benchmark — that edge disappears entirely: NDF beats the
from-scratch model in **every** regime checked (trough 43.8%, peak 40.8%, other 46.9% — its
*largest* margin). A residual-correction model (predict `actual − NDF`, add back) barely helped and
made things slightly worse overall.

**A plain linear forecast combination does better than NDF alone — the first improvement over NDF
found by any mechanism.** Error correlation between the from-scratch model and NDF is only 0.36,
and the from-scratch model is individually more accurate on 33% of half-hours despite losing on
every aggregate regime — textbook Bates-Granger combination territory. A combination weight (≈0.17
on the from-scratch model, fit via expanding-window walk-forward, zero overlap with the evaluation
window) beat NDF alone on two independent, non-overlapping years:

| | weight fit on | evaluated on | NDF alone MAE | combined MAE | improvement |
|---|---|---|---|---|---|
| first check | 2024 H1 | 2024 H2 | 533.03 | 513.05 | 3.75% |
| final, corrected | 2021–2024 (full walk-forward) | all of 2025 | 593.14 | 572.11 | **3.55%** |

**Significance-tested (§9): yes, this is real.** An independently re-run Diebold-Mariano test on
this exact comparison — blend vs. NDF alone, half-hourly, `VALIDATION`=2025 — gives
**p = 1.55e-9** (mean loss differential −24.81 MW/row in the blend's favor), and survives
Benjamini-Hochberg FDR correction alongside five other headline comparisons tested together (§9).
This is the project's most important and most heavily-scrutinized claim, so it's also the one
most worth a rigor check beyond a single point-estimate percentage.

A more flexible regime-aware stacking meta-model (LightGBM combining both forecasts plus
calendar/bucket context) initially did *worse* than the simple blend (overfitting on a
single-season, six-month fit window — its `is_hot` bias flipped to +352 MW). Refitting on a
proper multi-year, multi-season out-of-sample set fixed most of that gap (515.28 vs. the blend's
513.07) but still didn't beat the simple weighted average — added complexity without measured
benefit, not evidence regime-awareness can't work with more data.

**This combination result is the project's final headline forecasting number.** It survived being
independently re-derived on a second year with a real, previously-undetected data bug found and
fixed along the way (§2) — about as strong a robustness check as this project has produced.

## 8. Impact estimate

A rough, deliberately transparent back-of-envelope estimate of what §7's forecast-combination
result — 21.03 MW average absolute-error reduction versus NDF alone, on the corrected 2025
`VALIDATION` — could plausibly be worth. Full arithmetic in
[`notebooks/22_impact_estimate.ipynb`](../notebooks/22_impact_estimate.ipynb).
**This is a directional estimate, not a causal claim** — shown in full so a reader can disagree
with an assumption rather than just the conclusion.

**Method**: treat the 21.03 MW average error reduction as if sustained across every settlement
period of a year (184,223 MWh of "avoided forecast-uncertainty volume," ≈0.08% of GB's annual
demand), then price it two ways. **£**: NESO's own half-hourly Balancing Services Use of System
(BSUoS) cost data, FY2024-25, matched to this project's own demand data over the identical dates,
gives two rates — a **broad** one (all six BSUoS cost categories ÷ total demand, £8.95/MWh, which
generously treats the *entire* system-operation cost as forecast-sensitive) and a **narrow** one
(only the `Positive Reserve` + `Energy Imbalance` categories — the two most mechanistically tied to
"the system needed more than expected" — £0.40/MWh, the more defensible figure). **tCO2**: the
Carbon Intensity API's standard generation-type factors for CCGT (394 gCO2/kWh, GB's typical
marginal plant) and OCGT (651 gCO2/kWh, faster-reacting peaking plant, a plausible upper bound).

| | low end (narrow £ rate / CCGT) | high end (broad £ rate / OCGT) |
|---|---|---|
| Avoided balancing cost, £/year | ~£74,000 | ~£1,650,000 |
| Avoided emissions, tCO2/year | ~72,600 | ~120,000 |

**The wide range is the honest answer, not a flaw to resolve** — public aggregate data can't
determine how much of GB's balancing bill is actually sensitive to demand-forecast accuracy
specifically. If a single number is wanted, the narrow-rate/CCGT low end (~£74k, ~72,600 tCO2/year)
is the more mechanistically defensible one — it isolates the cost categories closest to demand
imbalance and uses typical, not extreme, marginal generation.

**What this number represents, precisely**: not "this project's model saves the system this much" —
NDF is NESO's own already-deployed forecast, not something this project replaces. It's "combining a
free, from-scratch model with NESO's own published forecast — a technique not currently known to be
in operational use — would be worth roughly this much per year if adopted." A smaller, more honest
claim than "beating NESO," and a more useful one for judging whether any of this actually matters.

## 9. Statistical significance and false-discovery-rate correction

Every result above (§4–§8) reports a point-estimate delta — "27.65% improvement", "3.55%
improvement" — with no test of whether it's distinguishable from noise, and no correction for the
dozens of comparisons this project has run over its life. That gap is addressed directly here for
the project's six most load-bearing, currently-live comparisons
(`notebooks/24_significance_testing.ipynb`, `src/edf/significance.py`), all trained on `TRAIN`
(2020–2024) only and evaluated on the full, untouched `VALIDATION` year (2025).

**Method**: the **Diebold-Mariano test** (1995) on the paired per-row loss differential between
two forecasts, using a Newey-West/Bartlett HAC variance estimator with truncation lag `h−1`
(`h`=48, the `1d` horizon in half-hourly periods) — the standard tool for this setting, chosen
over a plain paired t-test because GB demand's strong day/week autocorrelation violates a t-test's
i.i.d. assumption, understating the true variance and giving artificially small p-values. Then
**Benjamini-Hochberg FDR correction** (1995) across all six p-values tested together, appropriate
for a set of independent exploratory questions rather than one confirmatory claim.

| comparison | mean loss diff. | p-value | BH q-value | survives α=0.05? |
|---|---|---|---|---|
| blend vs. NDF alone (§7, the project's headline result) | −24.81 MW | 1.55e-9 | 1.55e-9 | yes |
| our model alone vs. NDF alone (§7) | +336.23 MW | 5.22e-53 | 1.57e-52 | yes |
| weather-on vs. weather-off (§4) | −359.34 MW | 6.92e-51 | 1.38e-50 | yes |
| outturn wind/solar vs. none (§4) | −161.29 MW | 7.74e-16 | 1.16e-15 | yes |
| self-forecast wind/solar vs. none (§4) | −39.05 MW | 0.0042 | 0.0042 | yes |
| CQR vs. baseline non-coverage (§5) | −0.158 | 4.66e-255 | 2.80e-254 | yes |

**All six survive FDR correction.** With p-values this small — even the weakest, self-forecast
wind/solar at p=0.0042, clears the 0.05/6≈0.0083 a naive Bonferroni correction would have
required — there was nothing genuinely borderline in this particular set for BH to catch. That's a
real result (these six claims are solid), not evidence the correction was pointless: a more
marginal set of comparisons (e.g. the individual extreme-bucket findings in §6) would be a more
realistic place to expect FDR correction to actually change a conclusion, and hasn't been tested
this way yet (see §11).

**One finding changed on re-test, reported honestly rather than smoothed over.** Self-forecast
wind/solar (§4) was originally found to give a 0.09% MAE change ("noise, not signal") under the
project's older split and CV-tuned hyperparameters. Under the current split and untuned but
matched-capacity models, the same comparison shows a small but statistically significant ~4.1% MAE
reduction (p=0.0042) — a genuine update attributable to the different configuration, not a
contradiction. It remains the weakest and smallest effect of the six by a wide margin, and doesn't
overturn the practical recommendation not to deploy it (§4) — but "small, real, and not clearly
worth the complexity" is a more accurate summary than "zero effect."

**Caveat applying to all six**: statistical significance answers "is this effect distinguishable
from zero," not "is this effect large enough to matter" — with n=17,519 half-hourly rows, even
modest effects are detectable, as the self-forecast wind/solar result illustrates directly. The
practical-significance judgment calls made throughout §4–§8 (e.g. "not adopted", "kept for the
point model") stand on their own merits and aren't overridden by a small p-value alone.

## 10. Limitations

- **Weather is observed (ERA5 hindsight) for most of the project's date range, not forecast.** The
  one exception — a genuine day-ahead weather forecast archive — only starts 2024-03-07; results
  using it are an honest lower bound on how much of the weather gain survives real-world lead time,
  not a claim the whole 2020–2025 window's weather results are deployable as reported.
- **No free historical forecast archive exists for `wind`/`solar`/`interconnector`.** A from-scratch
  forecast of wind/solar was built and tested; it didn't recover the leaky upper-bound gain.
  Interconnector flow (price-driven, not weather-driven) remains an unaddressed limitation.
- **Embedded generation is invisibly netted into `demand` by physics, not by any subtraction NESO
  performs** — this project's `wind`/`solar` features exploit that correlation but cannot fully
  reconstruct a true net-demand ("duck curve") series without transmission-scale generation data.
- **No systematically-collected TV-audience dataset exists for this project's window**; the
  event-tier table is a researched qualitative judgement, cross-checked against one hard NESO MW
  figure, not a continuously measured variable.
- **Population-weighted weather across 5 conurbations is a coarse proxy**, not a full gridded
  weighting — a stated, unfixed simplification.
- **The COVID case study is not a held-out accuracy number** — the point model was trained on data
  including the lockdown period itself, so its apparent robustness there is a different claim than
  genuine zero-warning forecasting skill.
- **`TEST` (now 2026, provisional) has never been evaluated against.** All results above are
  `VALIDATION`-only. A real, NESO-acknowledged data-quality issue in the 2026 source (and a changed
  CSV filename pattern) means 2026 is not yet usable even as `VALIDATION` — see PLAN.md's Follow-up
  6.
- **The forecast-combination weight is fit once and applied flat** across the whole evaluation
  window; a deployed version would need periodic recalibration, standard practice for any
  combination weight in the forecast-combination literature.
- **Only two independent train/evaluate splits back the combination result** (2024 H1/H2, and the
  2021–2024/2025 split) — not full walk-forward CV across many rolling windows. Both agreeing this
  closely is reassuring, not conclusive.

## 11. What you'd do with more time

- **Give NGBoost (§5) a fair tuning budget.** Its loss in the CQR/NGBoost comparison traces to an
  un-tuned, shallow default base learner, not to the Normal-distribution assumption being wrong — a
  proper walk-forward-CV search (deeper trees, more estimators, possibly a skewed `Dist` given this
  project's own bias findings) is expensive (~15min per fit at this project's data volume) but not
  yet ruled out as competitive with CQR.
- A literal GB "duck curve" / net-load analysis using transmission-scale wind/solar generation
  (Elexon BMRS fuel mix) — embedded generation alone can't reconstruct it.
- Verify the 2026 data source properly (the Scottish-transfer-data gap NESO itself has flagged,
  and the new CSV filename pattern) and extend the combination result to a third independent year.
- Regional decomposition (GB's 14 Grid Supply Point regions) rather than one national-aggregate
  model — published GB net-load work suggests region-specific weather effects (e.g. wind speed
  mattering more for Scotland, solar irradiance more for southern England) that a single national
  model averages away.
- A properly walk-forward-validated (not single-split) version of the forecast-combination weight,
  and a test of whether combination helps the *probabilistic* forecasts the same way it helped the
  point forecast.
- Economic/social signal features (GDP, unemployment, or news-derived signals) — a UK/Ireland
  academic study found ~6% improvement from this class of feature; not yet tried here.
- **Extend §9's significance testing to the extreme-bucket findings** (`is_hot`/`is_high_wind`/
  `is_christmas` bias fixes, the renewable-decile PICP cliff) — smaller, more marginal effects than
  the six headline comparisons tested so far, and a more realistic place for BH-FDR correction to
  actually flip a conclusion rather than confirm an already-robust one.
