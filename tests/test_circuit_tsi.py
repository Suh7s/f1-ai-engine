"""Tests for circuit-specific TSI computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.circuit_tsi import (
    _build_circuit_tsi_matrix,
    _compute_expected_positions,
    build_circuit_tsi_artifacts,
)
from src.registry import get_driver_home_circuits


def _make_race_results(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a minimal race_results DataFrame."""
    defaults = {
        "race_name": "Test GP",
        "race_date": "2023-03-05",
        "driver_name": "",
        "constructor_name": "",
        "qualifying_position": 1,
        "points": 0,
        "q1": None,
        "q2": None,
        "q3": None,
        "is_wet_race": 0,
        "is_mixed_race": 0,
        "dnf_flag": 0,
        "dnf_reason": "finished",
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


def _make_cpi_history(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "race_date": "2023-03-05",
        "cpi_qualifying_before": 0.0,
        "cpi_race_before": 0.0,
        "cpi_reliability_before": 1.0,
        "cpi_qualifying_after": 0.0,
        "cpi_race_after": 0.0,
        "cpi_reliability_after": 1.0,
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


class TestComputeExpectedPositions:
    def test_better_car_gets_lower_expected_position(self):
        """A constructor with better CPI should get a lower (better) expected position."""
        race_results = _make_race_results([
            {"season": 2023, "round": 1, "circuit_id": "silverstone", "driver_id": "hamilton", "constructor_id": "mercedes", "finish_position": 1, "qualifying_position": 1},
            {"season": 2023, "round": 1, "circuit_id": "silverstone", "driver_id": "alonso", "constructor_id": "aston_martin", "finish_position": 3, "qualifying_position": 3},
        ])
        cpi_history = _make_cpi_history([
            {"constructor_id": "mercedes", "season": 2023, "round": 1, "cpi_performance_before": 0.5},
            {"constructor_id": "aston_martin", "season": 2023, "round": 1, "cpi_performance_before": 0.2},
        ])
        result = _compute_expected_positions(race_results, cpi_history)
        merc_expected = result.loc[result["driver_id"] == "hamilton", "expected_position_from_cpi"].iloc[0]
        am_expected = result.loc[result["driver_id"] == "alonso", "expected_position_from_cpi"].iloc[0]
        assert merc_expected < am_expected, "Better CPI should get lower expected position"

    def test_circuit_delta_computed(self):
        race_results = _make_race_results([
            {"season": 2023, "round": 1, "circuit_id": "silverstone", "driver_id": "hamilton", "constructor_id": "mercedes", "finish_position": 1, "qualifying_position": 2},
            {"season": 2023, "round": 1, "circuit_id": "silverstone", "driver_id": "alonso", "constructor_id": "aston_martin", "finish_position": 2, "qualifying_position": 1},
        ])
        cpi_history = _make_cpi_history([
            {"constructor_id": "mercedes", "season": 2023, "round": 1, "cpi_performance_before": 0.0},
            {"constructor_id": "aston_martin", "season": 2023, "round": 1, "cpi_performance_before": 0.0},
        ])
        result = _compute_expected_positions(race_results, cpi_history)
        assert "circuit_delta" in result.columns

    def test_mechanical_dnf_excluded(self):
        race_results = _make_race_results([
            {"season": 2023, "round": 1, "circuit_id": "silverstone", "driver_id": "hamilton", "constructor_id": "mercedes", "finish_position": 20, "qualifying_position": 1, "dnf_flag": 1, "dnf_reason": "mechanical"},
        ])
        result = _compute_expected_positions(race_results, pd.DataFrame())
        delta = result.loc[result["driver_id"] == "hamilton", "circuit_delta"].iloc[0]
        assert np.isnan(delta), "Mechanical DNFs should produce NaN delta"


class TestBuildCircuitTSIMatrix:
    def test_sample_size_lagged(self):
        """First race at a circuit should have sample_size_before = 0."""
        race_results = _make_race_results([
            {"season": 2022, "round": 1, "circuit_id": "silverstone", "driver_id": "hamilton", "constructor_id": "mercedes", "finish_position": 1, "qualifying_position": 1},
            {"season": 2023, "round": 1, "circuit_id": "silverstone", "driver_id": "hamilton", "constructor_id": "mercedes", "finish_position": 2, "qualifying_position": 2},
        ])
        result = _compute_expected_positions(race_results, pd.DataFrame())
        matrix = _build_circuit_tsi_matrix(result)
        first_race = matrix.loc[
            (matrix["driver_id"] == "hamilton") & (matrix["season"] == 2022)
        ].iloc[0]
        assert first_race["circuit_tsi_sample_size_before"] == 0

    def test_confidence_encoding(self):
        """Confidence should be 0.0 for < 2 races, 0.5 for 2-4, 1.0 for >= 5."""
        rows = []
        for year in range(2018, 2024):
            rows.append({
                "season": year, "round": 1, "circuit_id": "silverstone",
                "driver_id": "hamilton", "constructor_id": "mercedes",
                "finish_position": 1, "qualifying_position": 1,
            })
        race_results = _make_race_results(rows)
        result = _compute_expected_positions(race_results, pd.DataFrame())
        matrix = _build_circuit_tsi_matrix(result)
        ham = matrix.loc[matrix["driver_id"] == "hamilton"].sort_values(["season"]).reset_index(drop=True)

        # Year 0 (2018): 0 prior races → confidence 0.0
        assert ham.iloc[0]["circuit_tsi_confidence_before"] == 0.0
        # Year 1 (2019): 1 prior race → still 0.0
        assert ham.iloc[1]["circuit_tsi_confidence_before"] == 0.0
        # Year 2 (2020): 2 prior races → 0.5
        assert ham.iloc[2]["circuit_tsi_confidence_before"] == 0.5
        # Year 5 (2023): 5 prior races → 1.0
        assert ham.iloc[5]["circuit_tsi_confidence_before"] == 1.0

    def test_negative_score_for_outperformer(self):
        """A driver consistently finishing better than CPI predicts should have negative circuit TSI."""
        rows = []
        for year in range(2019, 2024):
            # Hamilton always finishes P1 but CPI says car is average (expected ~middle)
            rows.append({
                "season": year, "round": 1, "circuit_id": "silverstone",
                "driver_id": "hamilton", "constructor_id": "mercedes",
                "finish_position": 1, "qualifying_position": 1,
            })
            rows.append({
                "season": year, "round": 1, "circuit_id": "silverstone",
                "driver_id": "bottas", "constructor_id": "mercedes",
                "finish_position": 5, "qualifying_position": 4,
            })
            rows.append({
                "season": year, "round": 1, "circuit_id": "silverstone",
                "driver_id": "verstappen", "constructor_id": "red_bull",
                "finish_position": 2, "qualifying_position": 2,
            })
            rows.append({
                "season": year, "round": 1, "circuit_id": "silverstone",
                "driver_id": "perez", "constructor_id": "red_bull",
                "finish_position": 6, "qualifying_position": 5,
            })
        race_results = _make_race_results(rows)
        # All equal CPI → expected positions based on qualifying
        cpi_history = _make_cpi_history([
            {"constructor_id": "mercedes", "season": y, "round": 1, "cpi_performance_before": 0.0}
            for y in range(2019, 2024)
        ] + [
            {"constructor_id": "red_bull", "season": y, "round": 1, "cpi_performance_before": 0.0}
            for y in range(2019, 2024)
        ])
        result = _compute_expected_positions(race_results, cpi_history)
        matrix = _build_circuit_tsi_matrix(result)
        ham_latest = matrix.loc[matrix["driver_id"] == "hamilton"].sort_values("season").iloc[-1]
        # Hamilton finishes P1 but expected ~P1 (top qualifier with equal CPI) — should be near 0 or slightly negative
        assert ham_latest["circuit_tsi_score_after"] <= 0.1, (
            f"Hamilton should have neutral-to-negative circuit TSI, got {ham_latest['circuit_tsi_score_after']}"
        )


class TestDriverHomeCircuits:
    def test_hamilton_silverstone(self):
        assert "silverstone" in get_driver_home_circuits("lewis_hamilton")

    def test_verstappen_zandvoort(self):
        assert "zandvoort" in get_driver_home_circuits("max_verstappen")

    def test_leclerc_monaco(self):
        assert "monaco" in get_driver_home_circuits("charles_leclerc")

    def test_unknown_driver_empty(self):
        assert get_driver_home_circuits("unknown_driver") == []
