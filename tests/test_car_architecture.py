from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.car_architecture as car_architecture_module
from src.car_architecture import (
    _prior_architecture,
    compute_car_circuit_fit,
    get_car_architecture,
    update_car_architecture_from_race,
)
from src.race_simulator import RaceConfig, simulate_race
from src.utils import save_dataframe


def _empty_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.car_architecture.load_car_architecture_history", lambda: pd.DataFrame())


def test_mercedes_monza_advantage_2026(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_history(monkeypatch)
    fit = compute_car_circuit_fit("mercedes", "monza", 2026, 2)
    assert fit.fit_score > 0.3, f"Mercedes should have Monza advantage, got {fit.fit_score}"


def test_ferrari_monaco_advantage_2026(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_history(monkeypatch)
    fit = compute_car_circuit_fit("ferrari", "monaco", 2026, 2)
    assert fit.fit_score > 0.3, f"Ferrari should have Monaco advantage, got {fit.fit_score}"


def test_ferrari_monza_disadvantage_2026(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_history(monkeypatch)
    fit = compute_car_circuit_fit("ferrari", "monza", 2026, 2)
    assert fit.fit_score < 0.1, f"Ferrari should not have Monza advantage, got {fit.fit_score}"


def test_mercedes_monaco_disadvantage_2026(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_history(monkeypatch)
    fit = compute_car_circuit_fit("mercedes", "monaco", 2026, 2)
    assert fit.fit_score < 0.1, f"Mercedes should not have Monaco advantage, got {fit.fit_score}"


def test_circuit_advantage_changes_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    def _frame(_: RaceConfig, apply_circuit_advantage_score: bool = True) -> pd.DataFrame:
        win_probs = [0.34, 0.33, 0.33] if not apply_circuit_advantage_score else [0.42, 0.29, 0.29]
        return pd.DataFrame(
            [
                {
                    "driver_id": "russell",
                    "driver_name": "George Russell",
                    "constructor_id": "mercedes",
                    "constructor_name": "Mercedes",
                    "qualifying_position": 2,
                    "driver_wet_elo": 1600.0,
                    "driver_elo": 1600.0,
                    "race_condition": "dry",
                    "is_wet_race": 0,
                    "is_mixed_race": 0,
                    "relative_win_probability": win_probs[0],
                    "expected_finish": 2.0,
                    "team_reliability_last_10": 0.96,
                    "dnf_prob": 0.04,
                    "season_uncertainty": 1.0,
                    "regulation_warning": 0,
                    "circuit_advantage_score": 0.45 if apply_circuit_advantage_score else 0.0,
                },
                {
                    "driver_id": "leclerc",
                    "driver_name": "Charles Leclerc",
                    "constructor_id": "ferrari",
                    "constructor_name": "Ferrari",
                    "qualifying_position": 1,
                    "driver_wet_elo": 1590.0,
                    "driver_elo": 1590.0,
                    "race_condition": "dry",
                    "is_wet_race": 0,
                    "is_mixed_race": 0,
                    "relative_win_probability": win_probs[1],
                    "expected_finish": 2.1,
                    "team_reliability_last_10": 0.94,
                    "dnf_prob": 0.05,
                    "season_uncertainty": 1.0,
                    "regulation_warning": 0,
                    "circuit_advantage_score": -0.15 if apply_circuit_advantage_score else 0.0,
                },
                {
                    "driver_id": "verstappen",
                    "driver_name": "Max Verstappen",
                    "constructor_id": "red_bull",
                    "constructor_name": "Red Bull",
                    "qualifying_position": 3,
                    "driver_wet_elo": 1610.0,
                    "driver_elo": 1610.0,
                    "race_condition": "dry",
                    "is_wet_race": 0,
                    "is_mixed_race": 0,
                    "relative_win_probability": win_probs[2],
                    "expected_finish": 2.2,
                    "team_reliability_last_10": 0.95,
                    "dnf_prob": 0.05,
                    "season_uncertainty": 1.0,
                    "regulation_warning": 0,
                    "circuit_advantage_score": -0.10 if apply_circuit_advantage_score else 0.0,
                },
            ]
        )

    monkeypatch.setattr("src.race_simulator._build_base_frame", _frame)
    monza_config = RaceConfig(
        drivers=["russell", "leclerc", "verstappen"],
        qualifying_positions={"russell": 2, "leclerc": 1, "verstappen": 3},
        circuit_id="monza",
        season=2026,
        round=2,
        n_simulations=3000,
    )
    probs_without = simulate_race(monza_config, apply_circuit_advantage=False)
    probs_with = simulate_race(monza_config, apply_circuit_advantage=True)
    merc_driver = probs_with.win_probs.loc[probs_with.win_probs["driver_id"] == "russell", "relative_win_probability"].iloc[0]
    baseline = probs_without.win_probs.loc[probs_without.win_probs["driver_id"] == "russell", "relative_win_probability"].iloc[0]
    assert merc_driver > baseline * 1.1, "Mercedes driver should have higher Monza win prob with architecture"


def test_architecture_updates_with_race_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history_path = tmp_path / "car_architecture_history.csv"
    summary_path = tmp_path / "car_architecture_summary.csv"
    metadata_path = tmp_path / "car_architecture_metadata.json"
    monkeypatch.setitem(car_architecture_module.CONFIG, "history_path", history_path)
    monkeypatch.setitem(car_architecture_module.CONFIG, "summary_path", summary_path)
    monkeypatch.setitem(car_architecture_module.CONFIG, "metadata_path", metadata_path)

    prior = _prior_architecture("mercedes", 2026, 1)
    assert prior is not None
    row = pd.DataFrame([prior.as_row()])
    save_dataframe(row, history_path)
    save_dataframe(row, summary_path)

    arch_before = get_car_architecture("mercedes", 2026, 1)
    update_car_architecture_from_race(
        "mercedes",
        2026,
        2,
        {
            "constructors": {
                "mercedes": {
                    "straight_line_speed": 0.96,
                    "slow_corner_grip": 0.58,
                    "high_speed_cornering": 0.80,
                    "mechanical_grip": 0.73,
                    "tyre_management": 0.76,
                    "downforce_level": 0.42,
                }
            }
        },
    )
    arch_after = get_car_architecture("mercedes", 2026, 2)
    assert arch_after.confidence > arch_before.confidence, "Confidence should increase after update"
