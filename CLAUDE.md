# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 AI Prediction Engine — a living-season system for Formula 1 race and championship prediction. Ingests historical and live data from the Jolpica (Ergast mirror) and OpenF1 APIs, computes multi-dimensional driver/constructor skill ratings, engineers 100+ features, trains five XGBoost models, and runs Monte Carlo simulations for race and season outcomes. Surfaces everything through a Streamlit dashboard.

Training window: 10 rolling seasons. Current split: train ≤N-2, validation N-1, test N (auto-advances via `_rolling_split_seasons()`).

## Commands

### Full pipeline (data ingestion through model training and validation)
```bash
python scripts/run_pipeline.py --mode full
python scripts/run_pipeline.py --mode full --refresh          # re-fetch all API data from scratch
python scripts/run_pipeline.py --mode full --skip-fastf1      # skip FastF1 telemetry (much faster, use during dev)
```

### Single race prediction
```bash
python scripts/run_pipeline.py --mode predict --race "Bahrain" --season 2024 --round 1 --simulations 5000
```

### Season simulation / Championship snapshot
```bash
python scripts/run_pipeline.py --mode simulate-season --season 2024 --simulations 10000
python scripts/run_pipeline.py --mode championship --season 2024
```

### Incremental update (detects newly completed rounds and retrains incrementally)
```bash
python scripts/run_pipeline.py --mode update
python scripts/run_pipeline.py --mode update --skip-fastf1
```

### Race weekend workflow
```bash
python scripts/race_weekend.py --season 2025 --round 1 --session FP2          # Friday evening analysis after FP2
python scripts/race_weekend.py --season 2025 --round 1 --session FP3          # Saturday morning after FP3
python scripts/race_weekend.py --season 2025 --round 1 --session QUALI        # Pre-race prediction after qualifying
python scripts/race_weekend.py --season 2025 --round 1 --session LIVE         # Poll live race feed (--polls N --interval-seconds S)
python scripts/race_weekend.py --season 2025 --round 1 --session POST         # Post-race scoring and pipeline refresh
python scripts/race_weekend.py --season 2025 --round 1 --session POST --skip-fastf1
```

### Model audit (diagnostic charts, calibration analysis, feature importance deltas vs baseline)
```bash
python scripts/model_audit.py
```
Outputs to `data/processed/audit/`. Requires all pipeline artifacts + test predictions + calibration CSVs.

### Rebuild from cache (re-derive all analytical artifacts from cached data, zero API calls)
```bash
python scripts/rebuild_local_artifacts.py
```

### Dashboard
```bash
streamlit run dashboard/app.py
```
Requires all pipeline artifacts in `data/processed/`. Dark theme: background `#0a0a0f`, accent `#e10600`, font Space Grotesk.

### Tests
```bash
python -m pytest tests/                                                # all tests
python -m pytest tests/test_elo.py                                     # single file
python -m pytest tests/test_elo.py::test_expected_score_is_symmetric   # single test
```
Test coverage: `test_elo.py`, `test_features.py`, `test_tsi.py`, `test_regulation.py`, `test_simulator.py`. No tests yet for: `data_pipeline`, `model_training`, `fastf1_pipeline`, `weather_pipeline`, `car_performance`, `circuit_analysis`, `explainability`, `whatif_simulator`, `dashboard`.

### Install
```bash
pip install -r requirements.txt
```

## Architecture

### Pipeline execution order

Orchestrated by `scripts/run_pipeline.py` → `ensure_full_pipeline()`. Each stage feeds the next — ordering is strict:

```
data_pipeline → weather_pipeline → fastf1_pipeline → elo_rating → true_skill → car_performance → circuit_tsi → circuit_analysis → feature_engineering → model_training → system_integrity
```

1. **Data ingestion** (`src/data_pipeline.py`) — fetches results, qualifying, standings, laps, pitstops from Jolpica and OpenF1. Caches raw JSON + CSV per season under `data/raw/`. Produces `race_results.csv`, `calendar.csv`, standings CSVs, `pitstops.csv`. Rate-limited: Jolpica 0.75s, OpenF1 0.10s between requests. Max retries: 3, backoff: 1.5s exponential.
2. **Weather enrichment** (`src/weather_pipeline.py`) — fetches historical weather from Open-Meteo Archive API using per-circuit GPS coordinates (`CIRCUIT_COORDS` dict). Classification logic: >2mm precipitation or wet weathercode → `wet`; 0.5–2mm → `mixed`; otherwise → `dry`. Manual overrides in `WET_RACE_REGISTRY` for known misclassifications. Outputs `race_weather.csv`, merges `race_condition`, `is_wet_race`, `is_mixed_race` into race results.
3. **FastF1 telemetry** (`src/fastf1_pipeline.py`) — extracts practice, qualifying, sprint, and race telemetry for 2022+ seasons via OpenF1 API. Computes track evolution, sector times, long-run pace, tyre degradation rates, speed traps, compound usage. Slow stage. Failures logged to `fastf1_failures.csv`. Session coverage tracked in `openf1_weekend_coverage.csv`.
4. **ELO ratings** (`src/elo_rating.py`) — pairwise ELO updates for every driver pair in each race. Three separate rating tracks:
   - **Driver dry ELO**: starts 1500 (rookies 1480), K=48 first 20 races for rookies, K=32 base, K=24 after 50 races
   - **Driver wet ELO**: starts 1500, constant K=48, only updated on wet/mixed races
   - **Constructor ELO**: starts 1500, K=32/24, uses best-finishing driver per constructor
   - Pairwise weight: `1 + min(gap, 10) / 10` scaled by DNF multiplier (mechanical=0.6x, accident=1.0x, other=0.8x)
   - All ratings clamped to [1200, 2800]
5. **TrueSkill Index** (`src/true_skill.py`) — teammate-adjusted skill decomposition. Produces `tsi_qualifying`, `tsi_race`, `tsi_composite`, `tsi_rolling_20`, `tsi_wet`, `tsi_dry`, `tsi_delta`. Normalized to [-1, 1]. Saved as `tsi_driver_race.csv`, `tsi_head_to_head.csv`, `tsi_career_summary.csv`.
6. **Car Performance Index** (`src/car_performance.py`) — TSI-adjusted constructor performance. Strips out driver skill to isolate car contribution. Produces `cpi_history.csv` with `cpi_performance`, `cpi_qualifying`, `cpi_race`, `cpi_reliability`, `cpi_development_rate`, `cpi_regulation_adaptability`. Regulation years (2017, 2022, 2026) affect CPI reset logic.
7. **Circuit TSI** (`src/circuit_tsi.py`) — car-adjusted circuit-specific driver performance. For each driver × circuit pair, computes `circuit_delta = finish_position - CPI_expected_position` and tracks rolling scores normalized via tanh to [-1, 1]. Negative = circuit specialist (outperforms car prediction). Outputs `circuit_tsi_matrix.csv`. Validates known specialists (Hamilton-Silverstone, Perez-Baku, etc.) on every build.
8. **Circuit analysis** (`src/circuit_analysis.py`) — KMeans clustering (k selected from [3,4,5,6] by silhouette score) over circuit DNA features. Outputs `circuit_clusters.csv` and `driver_circuit_archetypes.csv` (per-driver fit score by cluster). Also exposes `get_circuit_specialists()`, `get_driver_circuit_profile()`, `detect_circuit_momentum()` wrappers over circuit TSI.
9. **Feature engineering** (`src/feature_engineering.py`) — builds the unified feature matrix from all upstream signals. Defines the authoritative `FEATURE_COLUMNS` (used by models A–D) and `DNF_FEATURE_COLUMNS` (used by model E). Also provides `build_race_inference_frame()` for real-time prediction input construction. See Feature Columns section below.
10. **Model training** (`src/model_training.py`) — see Model Stack section below.
11. **System validation** (`src/system_integrity.py`) — five automated checks run after every full pipeline:
    - `no_future_leakage`: verifies rolling features are properly lagged (first race row must have zero career stats)
    - `probabilities_sum_to_one`: validates saved prediction payloads in `data/predictions/`
    - `elo_ranges`: driver/constructor ELO within [1200, 2800]
    - `tsi_ranges`: all TSI metrics within [-1, 1]
    - `critical_feature_nans`: ensures `qualifying_position`, `driver_elo`, `constructor_elo`, `tsi_composite`, `cpi_performance` have zero nulls

### Data sources and coverage boundaries

| Source | Coverage | Used for | Rate limit |
|--------|----------|----------|------------|
| Jolpica (Ergast mirror) | All historical seasons | Results, qualifying, standings, laps, pitstops | 0.75s between requests |
| OpenF1 | 2023+ | Sessions, stints, pit stops, live race data | 0.10s between requests |
| FastF1 / OpenF1 telemetry | 2022+ | Practice/qualifying/race telemetry features | Session-level caching |
| Open-Meteo Archive | All historical | Race-day weather (temperature, precipitation, wind, weathercode) | 0.50s, 3 retries |

### The five models (A through E)

| Model | Target | Type | Feature set | Eval metrics | Calibration |
|-------|--------|------|-------------|--------------|-------------|
| **A** | `finish_position` | XGBRegressor | `FEATURE_COLUMNS` | MAE | None |
| **B** | `top1_finish` (win) | XGBClassifier | `FEATURE_COLUMNS` | log_loss, AUC, Brier | Isotonic regression (`CalibratedClassifierCV`) |
| **C** | `top3_finish` (podium) | XGBClassifier | `FEATURE_COLUMNS` | log_loss, AUC, Brier | Isotonic regression |
| **D** | `top10_finish` (points) | XGBClassifier | `FEATURE_COLUMNS` | log_loss, AUC, Brier | Isotonic regression |
| **E** | `dnf_flag` | XGBClassifier | `DNF_FEATURE_COLUMNS` | AUC, Brier, log_loss | Isotonic regression |

**Model E uses a completely different feature set** (`DNF_FEATURE_COLUMNS`) than A–D. This is intentional — DNF prediction relies on reliability signals, track risk, mechanical history, and grid position rather than performance features. Adding a feature to `FEATURE_COLUMNS` does not affect model E; you must separately update `DNF_FEATURE_COLUMNS`.

All classifiers use `scale_pos_weight = negative_count / positive_count` for label imbalance. All models use recency-weighted samples via exponential decay: `weight = 0.97^(days_since_race / 30)`. Hyperparameter search: `GridSearchCV` with `GroupKFold` (groups=season, k=5), grid over `n_estimators=[200,400]`, `max_depth=[3,5]`, `learning_rate=[0.03,0.08]`, `subsample=[0.8,1.0]`.

Production models are retrained on all available data after eval. Base (uncalibrated) models saved separately as `model_X_base.pkl` for incremental updates. Incremental retraining (`retrain_incremental`) continues training via XGBoost's `xgb_model=` parameter; triggers full retrain after 5 incremental updates or when the latest season in the feature matrix changes.

Artifacts per model: `model_X.pkl`, `model_X_base.pkl` (B–E only), `feature_importance_model_X.csv`, `feature_importance_model_X.json` (Plotly), `test_predictions_model_X.csv`, `calibration_model_X.csv`, `calibration_buckets_model_X.csv`. Global: `model_metrics.json`, `model_manifest.json`, `incremental_training_state.json`.

### Feature columns

`FEATURE_COLUMNS` (models A–D) is organized into these signal groups:

| Group | Example features |
|-------|-----------------|
| Skill ratings | `driver_elo`, `driver_wet_elo`, `constructor_elo` |
| TrueSkill | `tsi_qualifying`, `tsi_race`, `tsi_composite`, `tsi_rolling_20`, `tsi_delta`, `tsi_wet`, `tsi_dry`, `tsi_sample_size` |
| Wet-race skill | `wet_elo_vs_field`, `wet_race_win_rate`, `wet_race_podium_rate` |
| Car performance | `cpi_qualifying`, `cpi_race`, `cpi_reliability`, `cpi_performance`, `cpi_development_rate`, `cpi_regulation_adaptability` |
| Rolling driver stats | `avg_finish_last_5`, `avg_qualifying_last_5`, `win_rate_career`, `podium_rate_career`, `dnf_rate_last_10`, `laps_led_last_3`, `consistency_score` |
| Circuit history | `circuit_avg_finish`, `circuit_best_finish`, `races_at_circuit` |
| Circuit TSI | `circuit_tsi_score`, `circuit_tsi_confidence`, `circuit_tsi_sample_size`, `circuit_tsi_std`, `circuit_type_tsi`, `circuit_home_race` |
| Team stats | `team_avg_finish_last_5`, `team_reliability_last_10`, `team_pit_stop_avg` |
| Race context | `qualifying_position`, `teammate_gap_qualifying`, `grid_penalty`, `season_round`, `championship_pressure`, `pace_delta_last_3` |
| Weather | `race_precipitation_mm`, `race_max_wind_kmh`, `race_temperature_c`, `is_wet_race`, `is_mixed_race` |
| Circuit DNA | `circuit_overtake_difficulty`, `is_street_circuit`, `circuit_high_speed_pct`, `circuit_tyre_deg_severity`, `circuit_altitude_m`, `circuit_lap_length_km`, `circuit_sc_probability`, `circuit_drs_zones` |
| Qualifying telemetry | `best_sector_1_ms`, `best_sector_2_ms`, `best_sector_3_ms`, `sector_consistency`, `max_speed_trap`, `avg_throttle_pct`, `brake_points` |
| Weekend telemetry | Practice features (`fp_*`), qualifying deep features, sprint features, track evolution features, session context, OpenF1 fallback features |

`DNF_FEATURE_COLUMNS` (model E only): `team_reliability_last_10`, `cpi_reliability`, `dnf_rate_last_10`, `track_risk`, `laps_led_last_3`, `dnf_mechanical_last_3`, `dnf_accident_last_3`, `teammate_dnf_same_race`, `constructor_dnf_rate_season`, `new_part_race`, `season_round`, `grid_position`, `race_precipitation_mm`, `race_max_wind_kmh`, `is_wet_race`, `is_mixed_race`.

### Prediction flow (inference time)

```
predict_race() → build_race_inference_frame() → predict_from_frame() → normalize_probabilities()
     ↓                                                                         ↓
 Constructs features for                                              Model A: expected finish
 a single race using live                                             Model B: win probability (normalized to sum=1)
 roster, ELO, TSI, CPI,                                              Model C: podium probability
 circuit DNA, weather                                                 Model D: points probability
                                                                      Model E: DNF probability
                                                                              ↓
                                                              simulate_race() [Monte Carlo]
                                                                              ↓
                                                              Win/podium probs with p10/p90 confidence intervals
```

`build_race_inference_frame()` in `feature_engineering.py` is the single entry point for constructing model inputs at prediction time. It mirrors the training feature pipeline but for a single race. Every prediction path (race simulator, season simulator, dashboard, what-if, race weekend) flows through this function.

### Simulation layer

- **Race simulation** (`src/race_simulator.py`) — Monte Carlo (default 10,000 iterations, seed 42). Takes `RaceConfig` dataclass. Uses model predictions as base probabilities, adds Gaussian noise (`base_order_noise=0.18`, `uncertainty_noise_scale=0.22`), samples DNFs per driver, applies regulation uncertainty multipliers. Minimum win probability floor: 0.5%. Max DNF probability cap: 65%, floor: 1%. Outputs `RaceSimulationResult` with win/podium probabilities, expected points, and p10/p90 confidence intervals.
- **Season simulation** (`src/season_simulator.py`) — simulates remaining races in a season. Builds `RaceConfig` for each unfinished round using calendar + roster + ELO-based qualifying estimates. Produces WDC/WCC probabilities, projected final standings, probability evolution over rounds.
- **What-if simulator** (`src/whatif_simulator.py`) — counterfactual driver/team swap scenarios. "What if driver X drove for team Y?" Swaps CPI while preserving driver TSI and ELO to estimate alternate championship outcomes.

### Regulation handling

`src/regulation_handler.py` + `REGULATION_CHANGES` in `src/registry.py`. Major regulation years: **2022, 2026**. During regulation resets:
- Constructor prior ELO regressed 40% toward mean (1500): `elo * 0.60 + 1500 * 0.40`
- Season uncertainty multiplier: **0.60** for rounds 1–5, **0.75** for rounds 6–10, **0.90** for rounds 11+
- Dashboard and predictions display regulation-reset warnings for rounds ≤10

### Race weekend operations

`scripts/race_weekend.py` — five session modes for live race weekends:
- **FP2**: Friday evening analysis using all available practice data (long-run pace, sector strengths, compound usage)
- **FP3**: Saturday morning analysis with the full practice set
- **QUALI**: Post-qualifying pre-race prediction — ingests qualifying results, generates full prediction with confidence report, SHAP explanations, tire strategy, and persists to `data/predictions/`
- **LIVE**: Polls the live race feed at configurable intervals (`--polls N --interval-seconds S`) and refreshes position probabilities
- **POST**: Post-race pipeline refresh, scores saved pre-race prediction against actual results, generates accuracy report, optionally triggers incremental retraining

Uses `CalibrationEngine` for confidence scoring, `PredictionExplainer` for per-driver SHAP narrative, `TireModel` for degradation curves and strategy recommendations.

### Dashboard sections

`dashboard/app.py` — Streamlit app. Five sections:
1. **Race Predictor** — select race, run predictions and Monte Carlo simulation, view win/podium/points probabilities, upset probability, tire strategy, SHAP explanations
2. **Championship Simulator** — season-long Monte Carlo, WDC/WCC probabilities, projected standings, probability evolution charts
3. **Driver Analytics** — ELO history plots, TSI head-to-head comparisons, career rankings, circuit archetype fit
4. **Team Analytics** — CPI trajectories, development rate tracking, end-of-season CPI projections
5. **Model Performance** — metrics dashboard, calibration curves, feature importance charts, prediction accuracy history

### Supporting modules

- `src/tire_model.py` — `TireModel` class. Fits degradation curves per circuit-compound pair from stint data. Estimates `deg_rate`, `cliff_lap`, `r_squared`. Used in race weekend strategy analysis.
- `src/explainability.py` — `PredictionExplainer`. SHAP-based per-prediction explanations. Maps feature names to human-readable labels (`FEATURE_LABELS` dict). SHAP is optional — `try: import shap` with fallback.
- `src/calibration_engine.py` — `CalibrationEngine` and `ConfidenceReport`. Race prediction confidence scoring accounting for feature coverage, regulation resets, and historical calibration.
- `src/prediction_tracker.py` — persists pre-race predictions as JSON to `data/predictions/{season}_R{round:02d}_pre_race.json`. `score_prediction()` compares against actual results. `aggregate_season_accuracy()` summarizes a full season.
- `src/championship_tracker.py` — `ChampionshipTracker` class. Live standings from data, momentum index, mathematical elimination detection, championship scenarios (`ChampionshipScenario` with `required_results` text).
- `src/scheduler.py` — `auto_update()` checks Jolpica for newly completed rounds, re-runs the pipeline for the current season only, and triggers incremental or full retraining as appropriate.

## Hardcoded data dictionaries (maintenance required)

Five static data dictionaries must be kept in sync when the F1 calendar changes (new circuits, removed circuits, new seasons):

| Dictionary | Location | Purpose | Keys |
|-----------|----------|---------|------|
| `DRIVER_TEAM_MAP` | `src/registry.py` | Season roster: driver_id → constructor_id | season (int) → driver dict |
| `REGULATION_CHANGES` | `src/registry.py` | Major regulation years | season → "major" |
| `CIRCUIT_DNA` | `src/feature_engineering.py` | Circuit characteristics for features (high_speed_pct, tyre_deg_severity, overtaking_difficulty, sc_probability, altitude, lap_length, drs_zones) | circuit_id string |
| `CIRCUIT_COORDS` | `src/weather_pipeline.py` | GPS coordinates for weather API lookups | circuit_id → (lat, lon) |
| `CIRCUIT_METADATA` | `src/utils.py` | Legacy overtake difficulty and street circuit flag | circuit_id string |
| `TRACK_RISK_SCORES` | `src/feature_engineering.py` | Per-circuit DNF risk score for model E (Monaco=0.84, Monza=0.54, etc.) | circuit_id → float |
| `WET_RACE_REGISTRY` | `src/weather_pipeline.py` | Manual override for known wet/mixed races misclassified by weather API | (season, round) → "wet"/"mixed" |

**When adding a new circuit:** update `CIRCUIT_DNA`, `CIRCUIT_COORDS`, `CIRCUIT_METADATA`, and `TRACK_RISK_SCORES`. If any are missing, the pipeline uses conservative defaults (e.g., overtaking_difficulty=5, sc_probability=0.40, track_risk=0.45).

**When starting a new season:** update `DRIVER_TEAM_MAP` with the full grid. If it's a regulation-change year, add it to `REGULATION_CHANGES`.

## Config pattern

Every module defines its own `CONFIG` dict that extends `src.utils.CONFIG` (aliased as `BASE_CONFIG`). All file paths use `pathlib.Path`. The project root is auto-detected from `utils.py` location. Key base paths: `data_dir`, `raw_dir`, `processed_dir`, `predictions_dir`, `logs_dir`.

## Data layout

```
data/
  raw/                          # cached API responses (JSON + CSV per season per endpoint)
    fastf1_cache/               # FastF1 session cache (parquet)
    openf1/                     # OpenF1 round-level cache
  processed/                    # all pipeline outputs
    race_results.csv            # unified race result dataset (the core table)
    calendar.csv                # season calendar with circuit metadata
    features.csv                # feature matrix (one row per driver per race)
    model_A.pkl ... model_E.pkl # trained models
    model_B_base.pkl ... model_E_base.pkl  # uncalibrated base models
    driver_elo_history.csv      # per-race ELO snapshots (before/after)
    constructor_elo_history.csv
    tsi_driver_race.csv         # per-race TSI snapshots
    tsi_head_to_head.csv        # teammate comparison records
    cpi_history.csv             # constructor performance index over time
    circuit_clusters.csv        # KMeans cluster assignments
    race_weather.csv            # weather classification per race
    fastf1_features.csv         # telemetry-derived features
    model_metrics.json          # evaluation metrics for all models
    model_manifest.json         # model paths, feature lists, split info
    audit/                      # model_audit.py outputs
  predictions/                  # pre-race prediction payloads (JSON)
logs/
  pipeline.log                  # main pipeline log
  predictions.log               # prediction-specific events
  errors.log                    # error-level events
```

## Critical integration points

1. **Feature columns are the contract between feature engineering and model training.** `FEATURE_COLUMNS` and `DNF_FEATURE_COLUMNS` in `feature_engineering.py` are imported by `model_training.py`, `race_simulator.py`, `explainability.py`. When adding a new feature: (a) add to the appropriate column list, (b) compute it in `_build_features()`, (c) ensure `build_race_inference_frame()` populates it at prediction time, (d) retrain all models.

2. **`build_race_inference_frame()` must mirror training features exactly.** Any mismatch between training-time feature computation and inference-time computation causes silent prediction errors. This is the most common source of bugs when adding features.

3. **`race_condition` flows through the entire system.** Classified in `weather_pipeline` → merged into `race_results.csv` → drives wet ELO updates in `elo_rating` → populates `is_wet_race`/`is_mixed_race` in feature engineering → affects model predictions and simulation noise. Changing weather classification thresholds has cascading effects.

4. **Incremental retraining stores base models separately.** `CalibratedClassifierCV` wraps the XGBoost model; the inner (uncalibrated) model is needed for `xgb_model=` continuation. If you change model architecture, both the calibrated and base model must be re-saved.

5. **FastF1 telemetry features are sparse for pre-2022 seasons.** Missing values filled with 0.0. The `model_audit.py` script tracks feature availability % vs gain contribution. Features with <30% training availability and low gain may be candidates for removal. When adding new telemetry features, update the appropriate column list in `fastf1_pipeline.py` (`PRACTICE_FEATURE_COLUMNS`, `QUALIFYING_DEEP_FEATURE_COLUMNS`, etc.) and add to `FEATURE_COLUMNS` in `feature_engineering.py`.

6. **Points system.** `F1_POINTS` in `src/utils.py` maps position → points (P1=25, P2=18, ..., P10=1). `points_for_position()` helper. Used by race simulator, season simulator, and championship tracker. If F1 changes the points system, update this single dict.
