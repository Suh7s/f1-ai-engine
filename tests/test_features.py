from __future__ import annotations

import pandas as pd

from src.feature_engineering import build_feature_matrix


def _synthetic_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "race_name": "Round 1",
                "race_date": "2025-03-01",
                "race_time": "12:00:00Z",
                "circuit_id": "bahrain",
                "circuit_name": "Bahrain",
                "driver_id": "driver_a",
                "driver_code": "A",
                "driver_number": 1,
                "driver_given_name": "Driver",
                "driver_family_name": "A",
                "constructor_id": "team_x",
                "constructor_name": "Team X",
                "grid_position": 1,
                "finish_position": 1,
                "points": 25.0,
                "status": "Finished",
                "laps_completed": 57,
                "fastest_lap_rank": 1,
                "qualifying_position": 1,
                "grid_penalty": 0,
                "dnf_flag": 0,
                "dnf_reason": "finished",
                "q1": "1:31.000",
                "q2": "1:30.500",
                "q3": "1:30.000",
                "race_precipitation_mm": 0.0,
                "race_max_wind_kmh": 10.0,
                "race_temperature_c": 25.0,
                "weathercode": 0,
                "race_condition": "dry",
                "is_wet_race": 0,
                "is_mixed_race": 0,
            },
            {
                "season": 2025,
                "round": 1,
                "race_name": "Round 1",
                "race_date": "2025-03-01",
                "race_time": "12:00:00Z",
                "circuit_id": "bahrain",
                "circuit_name": "Bahrain",
                "driver_id": "driver_b",
                "driver_code": "B",
                "driver_number": 2,
                "driver_given_name": "Driver",
                "driver_family_name": "B",
                "constructor_id": "team_x",
                "constructor_name": "Team X",
                "grid_position": 2,
                "finish_position": 2,
                "points": 18.0,
                "status": "Finished",
                "laps_completed": 57,
                "fastest_lap_rank": 2,
                "qualifying_position": 2,
                "grid_penalty": 0,
                "dnf_flag": 0,
                "dnf_reason": "finished",
                "q1": "1:31.500",
                "q2": "1:31.000",
                "q3": "1:30.500",
                "race_precipitation_mm": 0.0,
                "race_max_wind_kmh": 10.0,
                "race_temperature_c": 25.0,
                "weathercode": 0,
                "race_condition": "dry",
                "is_wet_race": 0,
                "is_mixed_race": 0,
            },
            {
                "season": 2025,
                "round": 2,
                "race_name": "Round 2",
                "race_date": "2025-03-08",
                "race_time": "12:00:00Z",
                "circuit_id": "jeddah",
                "circuit_name": "Jeddah",
                "driver_id": "driver_a",
                "driver_code": "A",
                "driver_number": 1,
                "driver_given_name": "Driver",
                "driver_family_name": "A",
                "constructor_id": "team_x",
                "constructor_name": "Team X",
                "grid_position": 2,
                "finish_position": 2,
                "points": 18.0,
                "status": "Finished",
                "laps_completed": 50,
                "fastest_lap_rank": 2,
                "qualifying_position": 2,
                "grid_penalty": 0,
                "dnf_flag": 0,
                "dnf_reason": "finished",
                "q1": "1:28.000",
                "q2": "1:27.500",
                "q3": "1:27.000",
                "race_precipitation_mm": 0.0,
                "race_max_wind_kmh": 8.0,
                "race_temperature_c": 27.0,
                "weathercode": 0,
                "race_condition": "dry",
                "is_wet_race": 0,
                "is_mixed_race": 0,
            },
            {
                "season": 2025,
                "round": 2,
                "race_name": "Round 2",
                "race_date": "2025-03-08",
                "race_time": "12:00:00Z",
                "circuit_id": "jeddah",
                "circuit_name": "Jeddah",
                "driver_id": "driver_b",
                "driver_code": "B",
                "driver_number": 2,
                "driver_given_name": "Driver",
                "driver_family_name": "B",
                "constructor_id": "team_x",
                "constructor_name": "Team X",
                "grid_position": 1,
                "finish_position": 1,
                "points": 25.0,
                "status": "Finished",
                "laps_completed": 50,
                "fastest_lap_rank": 1,
                "qualifying_position": 1,
                "grid_penalty": 0,
                "dnf_flag": 0,
                "dnf_reason": "finished",
                "q1": "1:27.800",
                "q2": "1:27.200",
                "q3": "1:26.900",
                "race_precipitation_mm": 0.0,
                "race_max_wind_kmh": 8.0,
                "race_temperature_c": 27.0,
                "weathercode": 0,
                "race_condition": "dry",
                "is_wet_race": 0,
                "is_mixed_race": 0,
            },
        ]
    )


def test_rolling_features_are_lagged() -> None:
    race_results = _synthetic_results()
    driver_elo_history = pd.DataFrame(
        [
            {"driver_id": driver_id, "season": season, "round": round_number, "dry_elo_before": 1500.0, "wet_elo_before": 1500.0}
            for driver_id in ["driver_a", "driver_b"]
            for season, round_number in [(2025, 1), (2025, 2)]
        ]
    )
    constructor_elo_history = pd.DataFrame(
        [
            {"constructor_id": "team_x", "season": 2025, "round": 1, "elo_before": 1500.0},
            {"constructor_id": "team_x", "season": 2025, "round": 2, "elo_before": 1510.0},
        ]
    )
    features = build_feature_matrix(
        race_results=race_results,
        driver_elo_history=driver_elo_history,
        constructor_elo_history=constructor_elo_history,
        pitstops_df=pd.DataFrame(),
        fastf1_features_df=pd.DataFrame(),
    )
    driver_a_round_1 = features.loc[(features["driver_id"] == "driver_a") & (features["round"] == 1)].iloc[0]
    driver_a_round_2 = features.loc[(features["driver_id"] == "driver_a") & (features["round"] == 2)].iloc[0]
    assert driver_a_round_1["win_rate_career"] == 0.0
    assert driver_a_round_1["podium_rate_career"] == 0.0
    assert driver_a_round_2["win_rate_career"] == 1.0
    assert driver_a_round_2["avg_finish_last_5"] == 1.0

