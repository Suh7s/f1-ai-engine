from __future__ import annotations

import pandas as pd
import pytest

from src.race_simulator import RaceConfig, simulate_race


def _mock_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_id": "driver_a",
                "driver_name": "Driver A",
                "constructor_id": "team_1",
                "constructor_name": "Team 1",
                "qualifying_position": 1,
                "driver_wet_elo": 1600.0,
                "driver_elo": 1600.0,
                "race_condition": "dry",
                "is_wet_race": 0,
                "is_mixed_race": 0,
                "relative_win_probability": 0.6,
                "expected_finish": 1.5,
                "team_reliability_last_10": 0.95,
                "dnf_prob": 0.05,
                "season_uncertainty": 1.0,
                "regulation_warning": 0,
            },
            {
                "driver_id": "driver_b",
                "driver_name": "Driver B",
                "constructor_id": "team_2",
                "constructor_name": "Team 2",
                "qualifying_position": 2,
                "driver_wet_elo": 1550.0,
                "driver_elo": 1550.0,
                "race_condition": "dry",
                "is_wet_race": 0,
                "is_mixed_race": 0,
                "relative_win_probability": 0.3,
                "expected_finish": 2.1,
                "team_reliability_last_10": 0.93,
                "dnf_prob": 0.06,
                "season_uncertainty": 1.0,
                "regulation_warning": 0,
            },
            {
                "driver_id": "driver_c",
                "driver_name": "Driver C",
                "constructor_id": "team_3",
                "constructor_name": "Team 3",
                "qualifying_position": 3,
                "driver_wet_elo": 1500.0,
                "driver_elo": 1500.0,
                "race_condition": "dry",
                "is_wet_race": 0,
                "is_mixed_race": 0,
                "relative_win_probability": 0.1,
                "expected_finish": 3.2,
                "team_reliability_last_10": 0.90,
                "dnf_prob": 0.07,
                "season_uncertainty": 1.0,
                "regulation_warning": 0,
            },
        ]
    )


def test_monte_carlo_probabilities_sum_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    base_frame = _mock_base_frame()

    monkeypatch.setattr("src.race_simulator._build_base_frame", lambda config, apply_circuit_advantage_score=True: base_frame.copy())
    result = simulate_race(
        RaceConfig(
            drivers=["driver_a", "driver_b", "driver_c"],
            qualifying_positions={"driver_a": 1, "driver_b": 2, "driver_c": 3},
            circuit_id="bahrain",
            season=2025,
            round=1,
            n_simulations=1000,
        )
    )
    assert abs(result.win_probs["relative_win_probability"].sum() - 1.0) < 1e-9
    assert all(abs(row[["P1", "P2", "P3"]].sum() - 1.0) < 1e-9 for _, row in result.position_distributions.iterrows())


def test_monte_carlo_performance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.race_simulator._build_base_frame", lambda config, apply_circuit_advantage_score=True: _mock_base_frame().copy())
    result = simulate_race(
        RaceConfig(
            drivers=["driver_a", "driver_b", "driver_c"],
            qualifying_positions={"driver_a": 1, "driver_b": 2, "driver_c": 3},
            circuit_id="bahrain",
            season=2025,
            round=1,
            n_simulations=10000,
        )
    )
    assert result.simulation_runs == 10000
    assert not result.win_probs.empty
