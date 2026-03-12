"""System-wide validation checks for pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils import CONFIG as BASE_CONFIG, load_dataframe, load_json

CONFIG: dict[str, Any] = {
    "processed_dir": BASE_CONFIG["processed_dir"],
    "predictions_dir": BASE_CONFIG["predictions_dir"],
    "features_path": BASE_CONFIG["processed_dir"] / "features.csv",
    "driver_elo_path": BASE_CONFIG["processed_dir"] / "driver_elo_history.csv",
    "constructor_elo_path": BASE_CONFIG["processed_dir"] / "constructor_elo_history.csv",
    "tsi_driver_race_path": BASE_CONFIG["processed_dir"] / "tsi_driver_race.csv",
}


@dataclass(slots=True)
class ValidationResult:
    """Structured validation result."""

    name: str
    passed: bool
    details: str


def _validate_no_future_leakage(features: pd.DataFrame) -> ValidationResult:
    """Check that key rolling features behave like lagged values."""

    violations = 0
    for _, driver_frame in features.sort_values(["driver_id", "race_date", "season", "round"]).groupby("driver_id", sort=False):
        if len(driver_frame) < 2:
            continue
        first = driver_frame.iloc[0]
        if float(first["win_rate_career"]) != 0.0 or float(first["podium_rate_career"]) != 0.0:
            violations += 1
        if float(first["dnf_rate_last_10"]) != 0.0:
            violations += 1
    return ValidationResult(
        name="no_future_leakage",
        passed=violations == 0,
        details="Checked lagged driver-career features on each driver's first race row",
    )


def _validate_probabilities() -> ValidationResult:
    """Ensure saved prediction payloads sum to one for win probabilities."""

    violations = 0
    checked = 0
    for path in sorted(CONFIG["predictions_dir"].glob("*_pre_race.json")):
        payload = load_json(path, default=None)
        if payload is None:
            continue
        probs = [float(row.get("relative_win_probability", 0.0)) for row in payload.get("predictions", [])]
        if not probs:
            continue
        checked += 1
        if not np.isclose(sum(probs), 1.0, atol=1e-6):
            violations += 1
    return ValidationResult(
        name="probabilities_sum_to_one",
        passed=violations == 0,
        details=f"Checked {checked} saved pre-race prediction payload(s)",
    )


def _validate_elo_ranges(driver_elo: pd.DataFrame, constructor_elo: pd.DataFrame) -> ValidationResult:
    """Ensure Elo values remain within a plausible operating band."""

    driver_ok = driver_elo[["dry_elo_before", "dry_elo_after", "wet_elo_before", "wet_elo_after"]].apply(
        lambda series: series.between(1200, 2800).all()
    ).all()
    constructor_ok = constructor_elo[["elo_before", "elo_after"]].apply(lambda series: series.between(1200, 2800).all()).all()
    return ValidationResult(
        name="elo_ranges",
        passed=bool(driver_ok and constructor_ok),
        details="Validated driver and constructor Elo ratings remain within 1200-2800",
    )


def _validate_tsi_ranges(tsi_driver_race: pd.DataFrame) -> ValidationResult:
    """Ensure TSI values remain in the expected normalized band."""

    if tsi_driver_race.empty:
        return ValidationResult(name="tsi_ranges", passed=False, details="Missing tsi_driver_race artifact")
    columns = [
        "tsi_qualifying_before",
        "tsi_qualifying_after",
        "tsi_race_before",
        "tsi_race_after",
        "tsi_composite_before",
        "tsi_composite_after",
        "tsi_rolling_20_before",
        "tsi_rolling_20_after",
        "tsi_wet_before",
        "tsi_wet_after",
        "tsi_dry_before",
        "tsi_dry_after",
    ]
    bounded = True
    for column in columns:
        series = pd.to_numeric(tsi_driver_race[column], errors="coerce").dropna()
        if not series.empty and not series.between(-1.0, 1.0).all():
            bounded = False
            break
    return ValidationResult(
        name="tsi_ranges",
        passed=bounded,
        details="Validated normalized TSI metrics remain within [-1, 1]",
    )


def _validate_critical_nans(features: pd.DataFrame) -> ValidationResult:
    """Ensure core feature columns are populated."""

    critical = ["qualifying_position", "driver_elo", "constructor_elo", "tsi_composite", "cpi_performance"]
    missing = [column for column in critical if column not in features.columns or features[column].isna().any()]
    return ValidationResult(
        name="critical_feature_nans",
        passed=not missing,
        details="Missing or null critical feature columns: " + (", ".join(missing) if missing else "none"),
    )


def run_system_validation(
    features_df: pd.DataFrame | None = None,
    driver_elo_history: pd.DataFrame | None = None,
    constructor_elo_history: pd.DataFrame | None = None,
    tsi_driver_race: pd.DataFrame | None = None,
) -> list[ValidationResult]:
    """Run all requested integrity checks."""

    features = features_df if features_df is not None else load_dataframe(CONFIG["features_path"], parse_dates=["race_date"])
    driver_elo = driver_elo_history if driver_elo_history is not None else load_dataframe(CONFIG["driver_elo_path"])
    constructor_elo = constructor_elo_history if constructor_elo_history is not None else load_dataframe(CONFIG["constructor_elo_path"])
    tsi = tsi_driver_race if tsi_driver_race is not None else load_dataframe(CONFIG["tsi_driver_race_path"], parse_dates=["race_date"])

    return [
        _validate_no_future_leakage(features),
        _validate_probabilities(),
        _validate_elo_ranges(driver_elo, constructor_elo),
        _validate_tsi_ranges(tsi),
        _validate_critical_nans(features),
    ]


class SystemIntegrity:
    """Compatibility wrapper around the integrity checks."""

    def run(self) -> list[ValidationResult]:
        return run_system_validation()
