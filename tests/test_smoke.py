"""Smoke tests for previously untested modules."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd


def test_weather_pipeline_loads() -> None:
    from src.weather_pipeline import WeatherPipeline

    pipeline = WeatherPipeline()
    assert pipeline is not None


def test_weather_pipeline_classifies_conditions() -> None:
    from src.weather_pipeline import classify_race_conditions

    assert classify_race_conditions(0.0, 800) == "DRY"
    assert classify_race_conditions(3.0, 61) == "WET"
    assert classify_race_conditions(1.0, 800) == "MIXED"


def test_season_simulator_loads() -> None:
    from src.season_simulator import SeasonSimulator

    simulator = SeasonSimulator()
    assert simulator is not None


def test_season_simulator_returns_valid_probabilities() -> None:
    from src.season_simulator import SeasonSimulator

    simulator = SeasonSimulator()
    mock_standings = pd.DataFrame(
        {
            "driver_id": ["verstappen", "norris", "leclerc"],
            "points": [100, 90, 80],
            "constructor_id": ["red_bull", "mclaren", "ferrari"],
        }
    )
    mock_remaining = [
        {"season": 2025, "round": 20, "circuit_id": "bahrain"},
        {"season": 2025, "round": 21, "circuit_id": "monza"},
    ]
    with patch.object(
        simulator,
        "simulate_race",
        return_value={"verstappen": 0.5, "norris": 0.3, "leclerc": 0.2},
    ):
        result = simulator.simulate_season(mock_standings, mock_remaining, n_simulations=100)
    assert "wdc_probabilities" in result
    total_prob = sum(result["wdc_probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01


def test_championship_tracker_loads() -> None:
    from src.championship_tracker import ChampionshipTracker

    tracker = ChampionshipTracker()
    assert tracker is not None


def test_championship_tracker_standings() -> None:
    from src.championship_tracker import ChampionshipTracker

    tracker = ChampionshipTracker()
    standings = tracker.get_current_standings(2025)
    assert isinstance(standings, pd.DataFrame)
    assert "driver_id" in standings.columns
    assert "points" in standings.columns


def test_explainability_loads() -> None:
    from src.explainability import PredictionExplainer

    explainer = PredictionExplainer()
    assert explainer is not None


def test_explainability_returns_explanation() -> None:
    from src.explainability import PredictionExplainer

    explainer = PredictionExplainer()
    mock_features = pd.Series(
        {
            "qualifying_position": 1,
            "driver_elo": 2400,
            "constructor_elo": 2100,
            "circuit_tsi_score": -0.3,
            "car_circuit_fit": 0.5,
        }
    )
    with patch.object(
        explainer,
        "_compute_shap_values",
        return_value=np.array([0.18, 0.12, 0.09, 0.07, -0.03]),
    ):
        result = explainer.explain_race_prediction("verstappen", mock_features, "model_b")
    assert result is not None
    assert len(result.top_factors) >= 3


def test_prediction_tracker_loads() -> None:
    from src.prediction_tracker import PredictionTracker

    tracker = PredictionTracker()
    assert tracker is not None


def test_prediction_tracker_saves_and_loads() -> None:
    from src.prediction_tracker import PredictionTracker

    tracker = PredictionTracker()
    mock_prediction = {
        "season": 2025,
        "round": 1,
        "gp_name": "Bahrain Grand Prix",
        "predictions": {"verstappen": 0.45, "norris": 0.25},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tracker.save_prediction(mock_prediction, output_dir=tmpdir)
        assert os.path.exists(path)
        loaded = tracker.load_prediction(path)
        assert loaded["gp_name"] == "Bahrain Grand Prix"


def test_calibration_engine_loads() -> None:
    from src.calibration_engine import CalibrationEngine

    engine = CalibrationEngine()
    assert engine is not None


def test_calibration_engine_confidence_levels() -> None:
    from src.calibration_engine import CalibrationEngine

    engine = CalibrationEngine()
    full_features = pd.DataFrame(
        {
            "fp_long_run_pace": [1.2],
            "quali_gap_to_pole_ms": [50.0],
            "is_wet_race": [0],
            "season_round": [10],
            "circuit_tsi_score": [-0.2],
            "circuit_id": ["bahrain"],
        }
    )
    result = engine.compute_prediction_confidence(full_features, 2025, 10)
    assert result.confidence_label in ["HIGH", "MEDIUM", "LOW", "VERY LOW"]


def test_whatif_simulator_loads() -> None:
    from src.whatif_simulator import WhatIfSimulator

    simulator = WhatIfSimulator()
    assert simulator is not None


def test_scheduler_loads() -> None:
    from src.scheduler import Scheduler

    scheduler = Scheduler()
    assert scheduler is not None


def test_scheduler_detects_new_data() -> None:
    from src.scheduler import Scheduler

    scheduler = Scheduler()
    with patch.object(
        scheduler,
        "fetch_calendar",
        return_value=pd.DataFrame({"round": [1, 2, 3], "is_completed": [True, True, False]}),
    ):
        with patch.object(scheduler, "get_cached_rounds", return_value=[1]):
            new_rounds = scheduler.check_for_new_data()
            assert 2 in new_rounds


def test_system_integrity_loads() -> None:
    from src.system_integrity import SystemIntegrity

    integrity = SystemIntegrity()
    assert integrity is not None


def test_pipeline_stage_ordering() -> None:
    from scripts.run_pipeline import PIPELINE_STAGES

    assert len(PIPELINE_STAGES) >= 8
    for stage in PIPELINE_STAGES:
        assert "name" in stage
        assert "fn" in stage
        assert callable(stage["fn"])


def test_cached_pipeline_rebuild_executes() -> None:
    from src.car_architecture import run_car_architecture_pipeline
    from src.car_performance import run_cpi_pipeline
    from src.circuit_analysis import run_circuit_analysis_pipeline
    from src.circuit_tsi import run_circuit_tsi_pipeline
    from src.data_pipeline import run_data_pipeline
    from src.elo_rating import run_elo_pipeline
    from src.fastf1_pipeline import run_fastf1_pipeline
    from src.feature_engineering import run_feature_pipeline
    from src.model_training import run_training_pipeline
    from src.true_skill import run_tsi_pipeline
    from src.weather_pipeline import run_weather_pipeline

    data_outputs = run_data_pipeline(refresh=False)
    race_results = data_outputs["race_results"]
    assert not race_results.empty

    fastf1_features, fastf1_failures = run_fastf1_pipeline(
        race_results=race_results,
        refresh=False,
        skip=False,
    )
    assert isinstance(fastf1_features, pd.DataFrame)
    assert isinstance(fastf1_failures, pd.DataFrame)

    race_weather, weather_enriched = run_weather_pipeline(race_results=race_results, refresh=False)
    assert not race_weather.empty
    assert not weather_enriched.empty

    driver_elo_history, constructor_elo_history = run_elo_pipeline(race_results=weather_enriched)
    tsi_artifacts = run_tsi_pipeline(race_results=weather_enriched)
    cpi_artifacts = run_cpi_pipeline(
        race_results=weather_enriched,
        tsi_driver_race=tsi_artifacts.driver_race,
    )
    circuit_tsi_artifacts = run_circuit_tsi_pipeline(
        race_results=weather_enriched,
        cpi_history=cpi_artifacts.cpi_history,
    )
    circuit_artifacts = run_circuit_analysis_pipeline(race_results=weather_enriched)
    car_architecture = run_car_architecture_pipeline(
        race_results=weather_enriched,
        fastf1_features=fastf1_features,
    )

    assert not driver_elo_history.empty
    assert not constructor_elo_history.empty
    assert not tsi_artifacts.driver_race.empty
    assert not cpi_artifacts.cpi_history.empty
    assert not circuit_tsi_artifacts.circuit_tsi_matrix.empty
    assert not circuit_artifacts.cluster_frame.empty
    assert not car_architecture.summary.empty

    features = run_feature_pipeline(
        race_results=weather_enriched,
        driver_elo_history=driver_elo_history,
        constructor_elo_history=constructor_elo_history,
        pitstops_df=data_outputs["pitstops"],
        fastf1_features_df=fastf1_features,
        tsi_driver_race=tsi_artifacts.driver_race,
        cpi_history=cpi_artifacts.cpi_history,
        circuit_tsi_matrix=circuit_tsi_artifacts.circuit_tsi_matrix,
        cluster_frame=circuit_artifacts.cluster_frame,
    )
    training_outputs = run_training_pipeline(features_df=features, race_results_df=weather_enriched)

    assert len(features) >= 4000
    assert training_outputs["manifest"]["split_counts"]["test"] >= 100
    assert training_outputs["metrics"]["E"]["test_metrics"]["auc"] >= 0.72
