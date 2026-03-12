from __future__ import annotations

import pandas as pd

from src.elo_rating import _expected_score, calculate_elo_histories


def test_expected_score_is_symmetric() -> None:
    assert _expected_score(1600, 1400) > 0.5
    assert _expected_score(1400, 1600) < 0.5
    assert abs((_expected_score(1600, 1400) + _expected_score(1400, 1600)) - 1.0) < 1e-9


def test_elo_updates_reward_the_winner() -> None:
    race_results = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "race_name": "Test GP",
                "circuit_id": "bahrain",
                "driver_id": "driver_a",
                "constructor_id": "team_a",
                "finish_position": 1,
                "laps_completed": 57,
                "grid_position": 1,
                "dnf_reason": "finished",
                "race_condition": "dry",
            },
            {
                "season": 2025,
                "round": 1,
                "race_name": "Test GP",
                "circuit_id": "bahrain",
                "driver_id": "driver_b",
                "constructor_id": "team_b",
                "finish_position": 2,
                "laps_completed": 57,
                "grid_position": 2,
                "dnf_reason": "finished",
                "race_condition": "dry",
            },
        ]
    )
    driver_history, _ = calculate_elo_histories(race_results)
    driver_a = driver_history.loc[driver_history["driver_id"] == "driver_a"].iloc[0]
    driver_b = driver_history.loc[driver_history["driver_id"] == "driver_b"].iloc[0]
    assert driver_a["dry_elo_after"] > driver_a["dry_elo_before"]
    assert driver_b["dry_elo_after"] < driver_b["dry_elo_before"]

