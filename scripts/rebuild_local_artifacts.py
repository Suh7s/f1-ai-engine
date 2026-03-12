"""Force a local analytical rebuild from cached datasets only."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import load_processed_inputs
from src.fastf1_pipeline import run_fastf1_pipeline
from src.elo_rating import run_elo_pipeline
from src.true_skill import run_tsi_pipeline
from src.car_performance import run_cpi_pipeline
from src.circuit_tsi import run_circuit_tsi_pipeline
from src.circuit_analysis import run_circuit_analysis_pipeline
from src.feature_engineering import run_feature_pipeline
from src.model_training import run_training_pipeline
from src.system_integrity import run_system_validation


def main() -> None:
    pipeline_data = load_processed_inputs()
    race_results = pipeline_data["race_results"]
    fastf1_features, fastf1_failures = run_fastf1_pipeline(race_results=race_results, refresh=False, skip=False)
    driver_elo_history, constructor_elo_history = run_elo_pipeline(race_results)
    tsi_artifacts = run_tsi_pipeline(race_results=race_results)
    cpi_artifacts = run_cpi_pipeline(race_results=race_results, tsi_driver_race=tsi_artifacts.driver_race)
    circuit_tsi_artifacts = run_circuit_tsi_pipeline(race_results=race_results, cpi_history=cpi_artifacts.cpi_history)
    circuit_artifacts = run_circuit_analysis_pipeline(race_results=race_results)
    features = run_feature_pipeline(
        race_results=race_results,
        driver_elo_history=driver_elo_history,
        constructor_elo_history=constructor_elo_history,
        pitstops_df=pipeline_data["pitstops"],
        fastf1_features_df=fastf1_features,
        tsi_driver_race=tsi_artifacts.driver_race,
        cpi_history=cpi_artifacts.cpi_history,
        circuit_tsi_matrix=circuit_tsi_artifacts.circuit_tsi_matrix,
        cluster_frame=circuit_artifacts.cluster_frame,
    )
    training_result = run_training_pipeline(features)
    metrics = training_result.get("metrics", training_result)
    print("MODEL_A_TEST_MAE", metrics["A"]["test_metrics"]["mae"])
    print("MODEL_E_TEST_AUC", metrics["E"]["test_metrics"]["auc"])
    for result in run_system_validation(
        features_df=features,
        driver_elo_history=driver_elo_history,
        constructor_elo_history=constructor_elo_history,
        tsi_driver_race=tsi_artifacts.driver_race,
    ):
        print(f"{result.name}\tpassed={result.passed}\tdetails={result.details}")
    print("FASTF1_FAILURES", len(fastf1_failures))


if __name__ == "__main__":
    main()
