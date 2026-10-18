# Original implementation reference

This document preserves the original application documentation. Commands below run from the repository root. Its production-readiness claims are historical and do not describe the current Pit Wall release. For the current app, use the [main guide](../../README.md).

# F1 AI Prediction Engine

Production-ready Python project for end-to-end Formula 1 race and championship prediction:

- Data ingestion from the Jolpica Ergast mirror and OpenF1
- Unified race-result dataset construction
- Driver and constructor ELO ratings
- Feature engineering with rolling, career, team, and circuit features
- XGBoost model training with validation and test evaluation
- Monte Carlo race simulation
- Monte Carlo season simulation
- Streamlit dashboard for prediction, analytics, and model diagnostics

## Project Structure

```text
f1-ai-engine/
  data/
    raw/
    processed/
  src/
    data_pipeline.py
    feature_engineering.py
    elo_rating.py
    model_training.py
    race_simulator.py
    season_simulator.py
    utils.py
  dashboard/
    app.py
  scripts/
    run_pipeline.py
  requirements.txt
  README.md
```

## Setup

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --mode full
```

The full pipeline will:

1. Fetch and cache raw Jolpica/OpenF1 API data to `data/raw/`
2. Build processed tables in `data/processed/`
3. Generate driver and constructor ELO histories
4. Build the feature matrix
5. Train and evaluate four XGBoost models
6. Save model artifacts, metrics, predictions, and feature-importance outputs

Logs are written to `logs/pipeline.log` and echoed to the console.

## CLI Usage

Run the complete pipeline:

```bash
python scripts/run_pipeline.py --mode full
```

Refresh cached API data and rebuild:

```bash
python scripts/run_pipeline.py --mode full --refresh
```

Predict a specific race:

```bash
python scripts/run_pipeline.py --mode predict --race "Bahrain" --season 2024 --round 1
```

Simulate a full season:

```bash
python scripts/run_pipeline.py --mode simulate-season --season 2024
```

You can also control Monte Carlo iteration counts:

```bash
python scripts/run_pipeline.py --mode predict --race "Monza" --season 2024 --round 16 --simulations 5000
python scripts/run_pipeline.py --mode simulate-season --season 2024 --simulations 10000
```

## Dashboard

After running the full pipeline:

```bash
streamlit run dashboard/app.py
```

Dashboard sections:

- Race Predictor
- Championship Simulator
- Driver Analytics
- Team Analytics
- Model Performance

Dashboard typography uses `Exo 2` for headings and `JetBrains Mono` for dense telemetry/data views.

## Data Notes

- Historical training data uses seasons 2015-2024
- Jolpica is used as the Ergast-compatible source for results, qualifying, and standings
- OpenF1 is used to enrich 2023-2024 race data with session, stint, and pit-stop information when available
- Lap times and pit stops are fetched for the most recent three seasons in the configured range
- Raw API payloads and flattened CSVs are cached per season to avoid refetching on reruns
- The unified `race_results.csv` file includes DNF classification and merged qualifying positions

## Modeling Notes

- Driver ELO starts at `1500`
- K-factor is `32`, reduced to `24` after 50 races
- Pairwise ELO updates use finishing-order comparisons across all drivers in a race
- Classifiers use `scale_pos_weight` to address label imbalance
- Splits are:
  - Train: `2015-2022`
  - Validation: `2023`
  - Test: `2024`

## Key Artifacts

Saved under `data/processed/`:

- `race_results.csv`
- `driver_elo_history.csv`
- `constructor_elo_history.csv`
- `features.csv`
- `model_A.pkl`
- `model_B.pkl`
- `model_C.pkl`
- `model_D.pkl`
- `model_metrics.json`
- `model_manifest.json`
- `feature_importance_model_*.csv`
- `test_predictions_model_*.csv`
- `calibration_model_B.csv`

## Engineering Standards Implemented

- Path handling uses `pathlib.Path`
- Logging uses the standard `logging` module
- Console and file logging are both enabled
- Functions include docstrings and type hints
- API requests include rate limiting and retry logic
- Cache fallback is used when API requests fail after retries
- Data validation checks required columns after each major stage
