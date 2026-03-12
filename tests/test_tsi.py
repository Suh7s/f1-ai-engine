from __future__ import annotations

import pandas as pd

from src.true_skill import build_tsi_artifacts


def test_tsi_head_to_head_is_symmetric() -> None:
    race_results = pd.DataFrame(
        [
            {
                "season": 2025,
                "round": 1,
                "race_name": "Test GP",
                "race_date": "2025-03-01",
                "circuit_id": "bahrain",
                "driver_id": "driver_a",
                "driver_name": "Driver A",
                "driver_number": 1,
                "constructor_id": "team_x",
                "qualifying_position": 1,
                "finish_position": 1,
                "points": 25.0,
                "dnf_flag": 0,
                "dnf_reason": "finished",
                "q1": "1:31.000",
                "q2": "1:30.500",
                "q3": "1:30.000",
                "is_wet_race": 0,
                "is_mixed_race": 0,
            },
            {
                "season": 2025,
                "round": 1,
                "race_name": "Test GP",
                "race_date": "2025-03-01",
                "circuit_id": "bahrain",
                "driver_id": "driver_b",
                "driver_name": "Driver B",
                "driver_number": 2,
                "constructor_id": "team_x",
                "qualifying_position": 2,
                "finish_position": 2,
                "points": 18.0,
                "dnf_flag": 0,
                "dnf_reason": "finished",
                "q1": "1:31.400",
                "q2": "1:30.900",
                "q3": "1:30.600",
                "is_wet_race": 0,
                "is_mixed_race": 0,
            },
        ]
    )
    artifacts = build_tsi_artifacts(race_results=race_results)
    pair = artifacts.head_to_head.sort_values("driver_id").reset_index(drop=True)
    assert len(pair) == 2
    assert abs(pair.loc[0, "qualifying_delta_pct"] + pair.loc[1, "qualifying_delta_pct"]) < 1e-12
    assert abs(pair.loc[0, "race_delta_norm"] + pair.loc[1, "race_delta_norm"]) < 1e-12

