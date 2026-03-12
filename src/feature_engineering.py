"""Feature engineering for race outcome modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.car_performance import build_cpi_artifacts
from src.data_pipeline import load_calendar
from src.elo_rating import get_constructor_elo, get_driver_dry_elo, get_driver_wet_elo
from src.fastf1_pipeline import (
    OPENF1_FALLBACK_FEATURE_COLUMNS,
    PRACTICE_FEATURE_COLUMNS,
    QUALIFYING_DEEP_FEATURE_COLUMNS,
    QUALIFYING_FEATURE_COLUMNS,
    RACE_FEATURE_COLUMNS,
    SESSION_CONTEXT_COLUMNS,
    SPRINT_FEATURE_COLUMNS,
    TRACK_EVOLUTION_FEATURE_COLUMNS,
    WEEKEND_FEATURE_COLUMNS,
)
from src.circuit_tsi import build_circuit_tsi_artifacts, get_archetype_tsi, get_circuit_tsi
from src.registry import get_constructor_name, get_driver_home_circuits, get_driver_name, get_likely_team, get_season_roster
from src.true_skill import build_tsi_artifacts
from src.utils import (
    CONFIG as BASE_CONFIG,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_dataframe,
    save_json,
    safe_ratio,
    timing_to_milliseconds,
    validate_columns,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "raw_dir": BASE_CONFIG["raw_dir"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "driver_elo_path": BASE_CONFIG["processed_dir"] / "driver_elo_history.csv",
    "constructor_elo_path": BASE_CONFIG["processed_dir"] / "constructor_elo_history.csv",
    "pitstops_path": BASE_CONFIG["processed_dir"] / "pitstops.csv",
    "fastf1_features_path": BASE_CONFIG["processed_dir"] / "fastf1_features.csv",
    "features_path": BASE_CONFIG["processed_dir"] / "features.csv",
    "feature_columns_path": BASE_CONFIG["processed_dir"] / "feature_columns.json",
    "tsi_driver_race_path": BASE_CONFIG["processed_dir"] / "tsi_driver_race.csv",
    "cpi_history_path": BASE_CONFIG["processed_dir"] / "cpi_history.csv",
}

FEATURE_COLUMNS: list[str] = [
    "driver_elo",
    "driver_wet_elo",
    "tsi_qualifying",
    "tsi_race",
    "tsi_composite",
    "tsi_rolling_20",
    "tsi_delta",
    "tsi_wet",
    "tsi_dry",
    "tsi_sample_size",
    "wet_elo_vs_field",
    "wet_race_win_rate",
    "wet_race_podium_rate",
    "avg_finish_last_5",
    "avg_qualifying_last_5",
    "win_rate_career",
    "podium_rate_career",
    "dnf_rate_last_10",
    "laps_led_last_3",
    "consistency_score",
    "circuit_avg_finish",
    "circuit_best_finish",
    "races_at_circuit",
    "circuit_tsi_score",
    "circuit_tsi_confidence",
    "circuit_tsi_sample_size",
    "circuit_tsi_std",
    "circuit_type_tsi",
    "circuit_home_race",
    "constructor_elo",
    "cpi_qualifying",
    "cpi_race",
    "cpi_reliability",
    "cpi_performance",
    "cpi_development_rate",
    "cpi_regulation_adaptability",
    "team_avg_finish_last_5",
    "team_reliability_last_10",
    "team_pit_stop_avg",
    "pace_delta_last_3",
    "qualifying_position",
    "teammate_gap_qualifying",
    "grid_penalty",
    "season_round",
    "championship_pressure",
    "race_precipitation_mm",
    "race_max_wind_kmh",
    "race_temperature_c",
    "is_wet_race",
    "is_mixed_race",
    "circuit_overtake_difficulty",
    "is_street_circuit",
    "circuit_high_speed_pct",
    "circuit_tyre_deg_severity",
    "circuit_altitude_m",
    "circuit_lap_length_km",
    "circuit_sc_probability",
    "circuit_drs_zones",
    "best_sector_1_ms",
    "best_sector_2_ms",
    "best_sector_3_ms",
    "sector_consistency",
    "max_speed_trap",
    "avg_throttle_pct",
    "brake_points",
    *OPENF1_FALLBACK_FEATURE_COLUMNS,
    *PRACTICE_FEATURE_COLUMNS,
    *QUALIFYING_DEEP_FEATURE_COLUMNS,
    *SPRINT_FEATURE_COLUMNS,
    *TRACK_EVOLUTION_FEATURE_COLUMNS,
    *SESSION_CONTEXT_COLUMNS,
]

DNF_FEATURE_COLUMNS: list[str] = [
    "team_reliability_last_10",
    "cpi_reliability",
    "dnf_rate_last_10",
    "track_risk",
    "laps_led_last_3",
    "dnf_mechanical_last_3",
    "dnf_accident_last_3",
    "teammate_dnf_same_race",
    "constructor_dnf_rate_season",
    "new_part_race",
    "season_round",
    "grid_position",
    "race_precipitation_mm",
    "race_max_wind_kmh",
    "is_wet_race",
    "is_mixed_race",
]

ANALYTICS_ONLY_COLUMNS: list[str] = [*RACE_FEATURE_COLUMNS, "race_condition", "weathercode"]

TRACK_RISK_SCORES: dict[str, float] = {
    "bahrain": 0.35,
    "jeddah": 0.62,
    "monaco": 0.71,
    "baku": 0.81,
    "spa": 0.69,
    "silverstone": 0.48,
    "hungaroring": 0.38,
    "zandvoort": 0.44,
    "monza": 0.52,
    "yas_marina": 0.32,
    "marina_bay": 0.74,
    "suzuka": 0.51,
    "interlagos": 0.63,
    "albert_park": 0.58,
    "shanghai": 0.45,
    "miami": 0.52,
    "imola": 0.61,
    "catalunya": 0.38,
    "red_bull_ring": 0.55,
    "americas": 0.49,
    "rodriguez": 0.42,
    "las_vegas": 0.58,
    "vegas": 0.58,
    "losail": 0.44,
    "lusail": 0.44,
    "portimao": 0.46,
    "istanbul": 0.51,
    "mugello": 0.67,
    "sochi": 0.41,
    "nurburgring": 0.63,
    "ricard": 0.40,
    "hockenheimring": 0.54,
    "sepang": 0.56,
    "villeneuve": 0.59,
}

CIRCUIT_DNA: dict[str, dict[str, Any]] = {
    "albert_park": {"high_speed_pct": 0.43, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 6, "altitude_m": 14, "lap_length_km": 5.278, "sc_probability": 0.56, "drs_zones": 4},
    "americas": {"high_speed_pct": 0.47, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 7, "altitude_m": 161, "lap_length_km": 5.513, "sc_probability": 0.47, "drs_zones": 2},
    "bahrain": {"high_speed_pct": 0.54, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 8, "altitude_m": 7, "lap_length_km": 5.412, "sc_probability": 0.38, "drs_zones": 3},
    "baku": {"high_speed_pct": 0.56, "street_circuit": True, "tyre_deg_severity": 2, "overtaking_difficulty": 8, "altitude_m": -28, "lap_length_km": 6.003, "sc_probability": 0.71, "drs_zones": 2},
    "catalunya": {"high_speed_pct": 0.46, "street_circuit": False, "tyre_deg_severity": 5, "overtaking_difficulty": 4, "altitude_m": 109, "lap_length_km": 4.657, "sc_probability": 0.28, "drs_zones": 2},
    "hockenheimring": {"high_speed_pct": 0.48, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 6, "altitude_m": 103, "lap_length_km": 4.574, "sc_probability": 0.43, "drs_zones": 2},
    "hungaroring": {"high_speed_pct": 0.31, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 2, "altitude_m": 264, "lap_length_km": 4.381, "sc_probability": 0.58, "drs_zones": 2},
    "imola": {"high_speed_pct": 0.42, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 4, "altitude_m": 35, "lap_length_km": 4.909, "sc_probability": 0.36, "drs_zones": 1},
    "interlagos": {"high_speed_pct": 0.49, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 8, "altitude_m": 785, "lap_length_km": 4.309, "sc_probability": 0.64, "drs_zones": 2},
    "istanbul": {"high_speed_pct": 0.47, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 6, "altitude_m": 99, "lap_length_km": 5.338, "sc_probability": 0.37, "drs_zones": 2},
    "jeddah": {"high_speed_pct": 0.63, "street_circuit": True, "tyre_deg_severity": 2, "overtaking_difficulty": 7, "altitude_m": 15, "lap_length_km": 6.174, "sc_probability": 0.62, "drs_zones": 3},
    "losail": {"high_speed_pct": 0.52, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 7, "altitude_m": 9, "lap_length_km": 5.419, "sc_probability": 0.34, "drs_zones": 1},
    "marina_bay": {"high_speed_pct": 0.26, "street_circuit": True, "tyre_deg_severity": 3, "overtaking_difficulty": 3, "altitude_m": 15, "lap_length_km": 4.94, "sc_probability": 0.85, "drs_zones": 4},
    "miami": {"high_speed_pct": 0.49, "street_circuit": True, "tyre_deg_severity": 3, "overtaking_difficulty": 6, "altitude_m": 3, "lap_length_km": 5.412, "sc_probability": 0.52, "drs_zones": 3},
    "monaco": {"high_speed_pct": 0.15, "street_circuit": True, "tyre_deg_severity": 1, "overtaking_difficulty": 1, "altitude_m": 10, "lap_length_km": 3.337, "sc_probability": 0.92, "drs_zones": 1},
    "monza": {"high_speed_pct": 0.72, "street_circuit": False, "tyre_deg_severity": 2, "overtaking_difficulty": 9, "altitude_m": 162, "lap_length_km": 5.793, "sc_probability": 0.34, "drs_zones": 2},
    "mugello": {"high_speed_pct": 0.58, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 4, "altitude_m": 292, "lap_length_km": 5.245, "sc_probability": 0.28, "drs_zones": 1},
    "nurburgring": {"high_speed_pct": 0.43, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 5, "altitude_m": 620, "lap_length_km": 5.148, "sc_probability": 0.39, "drs_zones": 2},
    "portimao": {"high_speed_pct": 0.44, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 6, "altitude_m": 109, "lap_length_km": 4.653, "sc_probability": 0.31, "drs_zones": 1},
    "red_bull_ring": {"high_speed_pct": 0.59, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 8, "altitude_m": 678, "lap_length_km": 4.318, "sc_probability": 0.47, "drs_zones": 3},
    "ricard": {"high_speed_pct": 0.51, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 7, "altitude_m": 408, "lap_length_km": 5.842, "sc_probability": 0.25, "drs_zones": 2},
    "rodriguez": {"high_speed_pct": 0.57, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 8, "altitude_m": 2240, "lap_length_km": 4.304, "sc_probability": 0.49, "drs_zones": 3},
    "sepang": {"high_speed_pct": 0.51, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 7, "altitude_m": 18, "lap_length_km": 5.543, "sc_probability": 0.33, "drs_zones": 2},
    "shanghai": {"high_speed_pct": 0.53, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 8, "altitude_m": 4, "lap_length_km": 5.451, "sc_probability": 0.41, "drs_zones": 2},
    "silverstone": {"high_speed_pct": 0.67, "street_circuit": False, "tyre_deg_severity": 5, "overtaking_difficulty": 7, "altitude_m": 153, "lap_length_km": 5.891, "sc_probability": 0.36, "drs_zones": 2},
    "sochi": {"high_speed_pct": 0.44, "street_circuit": False, "tyre_deg_severity": 2, "overtaking_difficulty": 4, "altitude_m": 0, "lap_length_km": 5.848, "sc_probability": 0.29, "drs_zones": 2},
    "spa": {"high_speed_pct": 0.69, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 9, "altitude_m": 420, "lap_length_km": 7.004, "sc_probability": 0.52, "drs_zones": 2},
    "suzuka": {"high_speed_pct": 0.49, "street_circuit": False, "tyre_deg_severity": 4, "overtaking_difficulty": 3, "altitude_m": 50, "lap_length_km": 5.807, "sc_probability": 0.34, "drs_zones": 1},
    "vegas": {"high_speed_pct": 0.64, "street_circuit": True, "tyre_deg_severity": 2, "overtaking_difficulty": 7, "altitude_m": 610, "lap_length_km": 6.201, "sc_probability": 0.58, "drs_zones": 2},
    "villeneuve": {"high_speed_pct": 0.58, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 8, "altitude_m": 13, "lap_length_km": 4.361, "sc_probability": 0.63, "drs_zones": 3},
    "yas_marina": {"high_speed_pct": 0.45, "street_circuit": False, "tyre_deg_severity": 2, "overtaking_difficulty": 6, "altitude_m": 3, "lap_length_km": 5.281, "sc_probability": 0.33, "drs_zones": 2},
    "zandvoort": {"high_speed_pct": 0.39, "street_circuit": False, "tyre_deg_severity": 3, "overtaking_difficulty": 3, "altitude_m": 6, "lap_length_km": 4.259, "sc_probability": 0.45, "drs_zones": 2},
}

LOGGER = get_logger(__name__)


def track_risk_value(circuit_id: str) -> float:
    """Return a manual DNF-risk prior for a circuit."""

    return float(TRACK_RISK_SCORES.get(str(circuit_id), 0.52))


def _normalize_driver_number(value: Any) -> str | None:
    """Normalize driver numbers across cached sources."""

    if pd.isna(value) or value in (None, ""):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _unique_columns(columns: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""

    return list(dict.fromkeys(columns))


def _load_lap_lead_summary() -> pd.DataFrame:
    """Load recent lap data and aggregate laps led per driver-race."""

    lap_frames: list[pd.DataFrame] = []
    for path in sorted(CONFIG["raw_dir"].glob("laps_*.csv")):
        lap_frames.append(pd.read_csv(path, usecols=["season", "round", "driver_id", "position"]))

    if not lap_frames:
        return pd.DataFrame(columns=["season", "round", "driver_id", "laps_led_race"])

    laps = pd.concat(lap_frames, ignore_index=True)
    laps["position"] = pd.to_numeric(laps["position"], errors="coerce")
    return (
        laps.loc[laps["position"] == 1]
        .groupby(["season", "round", "driver_id"], as_index=False)
        .size()
        .rename(columns={"size": "laps_led_race"})
    )


def _load_fastf1_features() -> pd.DataFrame:
    """Load cached FastF1 features if present."""

    path = CONFIG["fastf1_features_path"]
    if not path.exists():
        return pd.DataFrame(columns=["season", "round", "driver_number", *WEEKEND_FEATURE_COLUMNS])
    frame = load_dataframe(path)
    if frame.empty:
        return pd.DataFrame(columns=["season", "round", "driver_number", *WEEKEND_FEATURE_COLUMNS])
    frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number)
    for column in WEEKEND_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def _load_tsi_driver_race(race_results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load or build persisted TSI race-level features."""

    path = CONFIG["tsi_driver_race_path"]
    if path.exists():
        return load_dataframe(path, parse_dates=["race_date"])
    return build_tsi_artifacts(race_results=race_results).driver_race


def _load_cpi_history(
    race_results: pd.DataFrame | None = None,
    tsi_driver_race: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load or build persisted CPI history."""

    path = CONFIG["cpi_history_path"]
    if path.exists():
        return load_dataframe(path, parse_dates=["race_date"])
    return build_cpi_artifacts(race_results=race_results, tsi_driver_race=tsi_driver_race).cpi_history


def _prepare_results(race_results: pd.DataFrame) -> pd.DataFrame:
    """Standardize race results for feature computation."""

    validate_columns(
        race_results,
        [
            "season",
            "round",
            "race_name",
            "race_date",
            "circuit_id",
            "driver_id",
            "constructor_id",
            "driver_number",
            "grid_position",
            "finish_position",
            "qualifying_position",
            "grid_penalty",
            "points",
            "dnf_flag",
            "dnf_reason",
            "q1",
            "q2",
            "q3",
        ],
        "race results for features",
    )

    frame = race_results.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number)
    frame = frame.sort_values(["driver_id", "season", "round"]).reset_index(drop=True)
    field_size = frame.groupby(["season", "round"])["driver_id"].transform("count")
    finish_position = pd.to_numeric(frame["finish_position"], errors="coerce")
    qualifying_position = pd.to_numeric(frame["qualifying_position"], errors="coerce").fillna(
        pd.to_numeric(frame["grid_position"], errors="coerce")
    )
    grid_position = pd.to_numeric(frame["grid_position"], errors="coerce").fillna(qualifying_position)
    fallback_positions = (
        finish_position.fillna(field_size)
        .clip(lower=1, upper=field_size)
    )
    qualifying_position = qualifying_position.fillna(fallback_positions)
    grid_position = grid_position.fillna(qualifying_position).fillna(fallback_positions)
    base_columns = pd.DataFrame(
        {
            "field_size": field_size,
            "finish_position": finish_position,
            "finish_position_filled": finish_position.fillna(field_size),
            "qualifying_position": qualifying_position,
            "grid_position": grid_position,
            "points": pd.to_numeric(frame["points"], errors="coerce").fillna(0.0),
            "grid_penalty": frame["grid_penalty"].fillna(False).astype(int),
            "dnf_reason": frame["dnf_reason"].fillna("finished"),
            "race_precipitation_mm": pd.to_numeric(
                frame["race_precipitation_mm"] if "race_precipitation_mm" in frame.columns else pd.Series(0.0, index=frame.index),
                errors="coerce",
            ).fillna(0.0),
            "race_max_wind_kmh": pd.to_numeric(
                frame["race_max_wind_kmh"] if "race_max_wind_kmh" in frame.columns else pd.Series(0.0, index=frame.index),
                errors="coerce",
            ).fillna(0.0),
            "race_temperature_c": pd.to_numeric(
                frame["race_temperature_c"] if "race_temperature_c" in frame.columns else pd.Series(0.0, index=frame.index),
                errors="coerce",
            ).fillna(0.0),
            "is_wet_race": pd.to_numeric(
                frame["is_wet_race"] if "is_wet_race" in frame.columns else pd.Series(0, index=frame.index),
                errors="coerce",
            ).fillna(0).astype(int),
            "is_mixed_race": pd.to_numeric(
                frame["is_mixed_race"] if "is_mixed_race" in frame.columns else pd.Series(0, index=frame.index),
                errors="coerce",
            ).fillna(0).astype(int),
            "weathercode": pd.to_numeric(
                frame["weathercode"] if "weathercode" in frame.columns else pd.Series(np.nan, index=frame.index),
                errors="coerce",
            ),
            "race_condition": frame["race_condition"] if "race_condition" in frame.columns else "dry",
        },
        index=frame.index,
    )
    frame = frame.drop(columns=base_columns.columns.intersection(frame.columns), errors="ignore")
    frame = pd.concat([frame, base_columns], axis=1)
    frame["driver_name"] = (
        frame["driver_given_name"].fillna("").str.strip()
        + " "
        + frame["driver_family_name"].fillna("").str.strip()
    ).str.strip()
    frame.loc[frame["driver_name"] == "", "driver_name"] = frame.loc[frame["driver_name"] == "", "driver_id"]
    return frame


def _attach_base_enrichment(
    frame: pd.DataFrame,
    lap_lead_summary: pd.DataFrame,
    fastf1_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach same-weekend and telemetry helper columns."""

    enriched = frame.merge(
        lap_lead_summary,
        how="left",
        on=["season", "round", "driver_id"],
    )
    enriched["laps_led_race"] = pd.to_numeric(enriched["laps_led_race"], errors="coerce").fillna(0.0)

    qualifying_time_frame = enriched[["q3", "q2", "q1"]].copy()
    for column in qualifying_time_frame.columns:
        qualifying_time_frame[column] = qualifying_time_frame[column].map(timing_to_milliseconds)
    enriched["best_qualifying_time_ms"] = qualifying_time_frame.bfill(axis=1).iloc[:, 0]

    team_keys = ["season", "round", "constructor_id"]
    valid_qual_counts = enriched.groupby(team_keys)["best_qualifying_time_ms"].transform(lambda series: series.notna().sum())
    teammate_time_sum = enriched.groupby(team_keys)["best_qualifying_time_ms"].transform("sum")
    teammate_time = pd.Series(np.nan, index=enriched.index, dtype=float)
    teammate_mask = valid_qual_counts > 1
    teammate_time.loc[teammate_mask] = (
        teammate_time_sum.loc[teammate_mask] - enriched.loc[teammate_mask, "best_qualifying_time_ms"]
    ) / (valid_qual_counts.loc[teammate_mask] - 1)

    race_reference = enriched.groupby(["season", "round"])["best_qualifying_time_ms"].transform("median")
    fallback_gap = (
        enriched.groupby(team_keys)["qualifying_position"].transform("mean") - enriched["qualifying_position"]
    ) / np.clip(enriched["field_size"], 1, None)
    enriched["teammate_gap_qualifying"] = np.where(
        enriched["best_qualifying_time_ms"].notna() & teammate_time.notna() & race_reference.gt(0),
        (teammate_time - enriched["best_qualifying_time_ms"]) / race_reference,
        fallback_gap.fillna(0.0),
    )

    teammate_dnf_sum = enriched.groupby(team_keys)["dnf_flag"].transform("sum")
    team_driver_count = enriched.groupby(team_keys)["driver_id"].transform("count")
    enriched["teammate_dnf_current"] = np.where(
        team_driver_count > 1,
        np.clip((teammate_dnf_sum - enriched["dnf_flag"]) / (team_driver_count - 1), 0.0, 1.0),
        0.0,
    )

    if not fastf1_features_df.empty:
        fastf1_merge = fastf1_features_df.copy()
        fastf1_merge["driver_number"] = fastf1_merge["driver_number"].apply(_normalize_driver_number)
        enriched = enriched.merge(
            fastf1_merge,
            how="left",
            on=["season", "round", "driver_number"],
        )
    else:
        default_weekend = pd.DataFrame(
            {column: np.nan for column in WEEKEND_FEATURE_COLUMNS},
            index=enriched.index,
        )
        enriched = pd.concat([enriched, default_weekend], axis=1)
    return enriched


def _build_pitstop_team_frame(pitstops_df: pd.DataFrame, race_results: pd.DataFrame) -> pd.DataFrame:
    """Attach constructor IDs to pit stop records."""

    if pitstops_df.empty:
        return pd.DataFrame(columns=["season", "round", "constructor_id", "team_race_pit_stop_avg"])

    validate_columns(
        pitstops_df,
        ["season", "round", "driver_id", "duration_ms"],
        "pitstops for features",
    )
    constructor_map = race_results[["season", "round", "driver_id", "constructor_id"]].drop_duplicates()
    merged = pitstops_df.merge(constructor_map, how="left", on=["season", "round", "driver_id"])
    return (
        merged.groupby(["season", "round", "constructor_id"], as_index=False)["duration_ms"]
        .mean()
        .rename(columns={"duration_ms": "team_race_pit_stop_avg"})
    )


def _apply_driver_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add driver-level rolling and career features."""

    grouped = frame.groupby("driver_id", sort=False)
    avg_finish_last_5 = grouped["finish_position_filled"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    avg_qualifying_last_5 = grouped["qualifying_position"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    races_before = grouped.cumcount()
    is_win = frame["finish_position_filled"].eq(1).astype(int)
    is_podium = frame["finish_position_filled"].le(3).astype(int)
    career_rollup = pd.DataFrame(
        {"driver_id": frame["driver_id"], "is_win": is_win, "is_podium": is_podium},
        index=frame.index,
    )
    career_grouped = career_rollup.groupby("driver_id", sort=False)
    wins_before = career_grouped["is_win"].cumsum() - is_win
    podiums_before = career_grouped["is_podium"].cumsum() - is_podium
    driver_columns = pd.DataFrame(
        {
            "avg_finish_last_5": avg_finish_last_5,
            "avg_qualifying_last_5": avg_qualifying_last_5,
            "races_before": races_before,
            "is_win": is_win,
            "is_podium": is_podium,
            "wins_before": wins_before,
            "podiums_before": podiums_before,
            "win_rate_career": np.where(races_before > 0, wins_before / races_before, 0.0),
            "podium_rate_career": np.where(
                races_before > 0,
                podiums_before / races_before,
                0.0,
            ),
            "dnf_rate_last_10": grouped["dnf_flag"].transform(
                lambda series: series.shift(1).rolling(10, min_periods=1).mean()
            ),
            "laps_led_last_3": grouped["laps_led_race"].transform(
                lambda series: series.shift(1).rolling(3, min_periods=1).sum()
            ),
            "consistency_score": grouped["finish_position_filled"].transform(
                lambda series: series.shift(1).rolling(10, min_periods=2).std()
            ),
            "teammate_dnf_same_race": grouped["teammate_dnf_current"].transform(
                lambda series: series.shift(1).rolling(10, min_periods=1).mean()
            ),
        },
        index=frame.index,
    )
    frame = frame.drop(columns=driver_columns.columns.intersection(frame.columns), errors="ignore")
    frame = pd.concat([frame, driver_columns], axis=1)

    wet_start = (frame["is_wet_race"].eq(1) | frame["is_mixed_race"].eq(1)).astype(int)
    wet_wins = (frame["is_win"] * wet_start).astype(int)
    wet_podiums = (frame["is_podium"] * wet_start).astype(int)
    wet_rollup = pd.DataFrame(
        {
            "driver_id": frame["driver_id"],
            "wet_start": wet_start,
            "wet_wins_event": wet_wins,
            "wet_podiums_event": wet_podiums,
        },
        index=frame.index,
    )
    wet_grouped = wet_rollup.groupby("driver_id", sort=False)
    wet_columns = pd.DataFrame(
        {
            "wet_start": wet_start,
            "wet_wins_event": wet_wins,
            "wet_podiums_event": wet_podiums,
            "wet_starts_before": wet_grouped["wet_start"].cumsum() - wet_start,
            "wet_wins_before": wet_grouped["wet_wins_event"].cumsum() - wet_wins,
            "wet_podiums_before": wet_grouped["wet_podiums_event"].cumsum() - wet_podiums,
        },
        index=frame.index,
    )
    frame = pd.concat([frame, wet_columns], axis=1)
    rate_columns = pd.DataFrame(
        {
            "wet_race_win_rate": np.where(
                frame["wet_starts_before"] > 0,
                frame["wet_wins_before"] / frame["wet_starts_before"],
                0.0,
            ),
            "wet_race_podium_rate": np.where(
                frame["wet_starts_before"] > 0,
                frame["wet_podiums_before"] / frame["wet_starts_before"],
                0.0,
            ),
        },
        index=frame.index,
    )
    frame = pd.concat([frame, rate_columns], axis=1)
    frame = frame.drop(columns=["wet_start", "wet_wins_event", "wet_podiums_event"])

    circuit_grouped = frame.groupby(["driver_id", "circuit_id"], sort=False)
    circuit_columns = pd.DataFrame(
        {
            "circuit_avg_finish": circuit_grouped["finish_position_filled"].transform(
                lambda series: series.shift(1).expanding(min_periods=1).mean()
            ),
            "circuit_best_finish": circuit_grouped["finish_position_filled"].transform(
                lambda series: series.shift(1).expanding(min_periods=1).min()
            ),
            "races_at_circuit": circuit_grouped.cumcount(),
        },
        index=frame.index,
    )
    frame = frame.drop(columns=circuit_columns.columns.intersection(frame.columns), errors="ignore")
    frame = pd.concat([frame, circuit_columns], axis=1)

    mechanical_counts = pd.Series(index=frame.index, dtype=float)
    accident_counts = pd.Series(index=frame.index, dtype=float)
    for _, driver_history in frame.groupby("driver_id", sort=False):
        recent_dnfs: list[str] = []
        for row in driver_history.itertuples():
            mechanical_counts.loc[row.Index] = float(recent_dnfs.count("mechanical"))
            accident_counts.loc[row.Index] = float(recent_dnfs.count("accident"))
            if int(row.dnf_flag) == 1:
                normalized_reason = str(row.dnf_reason)
                if normalized_reason not in {"mechanical", "accident"}:
                    normalized_reason = "other"
                recent_dnfs.append(normalized_reason)
                recent_dnfs = recent_dnfs[-3:]

    dnf_history = pd.DataFrame(
        {
            "dnf_mechanical_last_3": mechanical_counts.astype(float),
            "dnf_accident_last_3": accident_counts.astype(float),
        },
        index=frame.index,
    )
    frame = frame.drop(columns=dnf_history.columns.intersection(frame.columns), errors="ignore")
    frame = pd.concat([frame, dnf_history], axis=1)
    return frame


def _apply_team_features(frame: pd.DataFrame, pitstops_df: pd.DataFrame) -> pd.DataFrame:
    """Add constructor-level rolling features."""

    team_race = (
        frame.groupby(["constructor_id", "season", "round"], as_index=False)
        .agg(
            team_race_finish_mean=("finish_position_filled", "mean"),
            team_race_dnf_rate=("dnf_flag", "mean"),
        )
        .sort_values(["constructor_id", "season", "round"])
    )

    team_grouped = team_race.groupby("constructor_id", sort=False)
    team_season_grouped = team_race.groupby(["constructor_id", "season"], sort=False)
    team_race["team_avg_finish_last_5"] = team_grouped["team_race_finish_mean"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean()
    )
    team_race["team_reliability_last_10"] = 1.0 - team_grouped["team_race_dnf_rate"].transform(
        lambda series: series.shift(1).rolling(10, min_periods=1).mean()
    )
    team_race["constructor_dnf_rate_season"] = team_season_grouped["team_race_dnf_rate"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    team_race["constructor_finish_season_avg"] = team_season_grouped["team_race_finish_mean"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    team_race["constructor_finish_last_3"] = team_season_grouped["team_race_finish_mean"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    team_race["pace_delta_last_3"] = team_race["constructor_finish_season_avg"] - team_race["constructor_finish_last_3"]

    pitstop_team = _build_pitstop_team_frame(pitstops_df, frame)
    if not pitstop_team.empty:
        pitstop_team = pitstop_team.sort_values(["constructor_id", "season", "round"])
        pitstop_grouped = pitstop_team.groupby("constructor_id", sort=False)
        pitstop_team["team_pit_stop_avg"] = pitstop_grouped["team_race_pit_stop_avg"].transform(
            lambda series: series.shift(1).rolling(5, min_periods=1).mean()
        )
        team_race = team_race.merge(
            pitstop_team[["constructor_id", "season", "round", "team_pit_stop_avg"]],
            how="left",
            on=["constructor_id", "season", "round"],
        )
    else:
        team_race["team_pit_stop_avg"] = np.nan

    frame = frame.merge(
        team_race[
            [
                "constructor_id",
                "season",
                "round",
                "team_avg_finish_last_5",
                "team_reliability_last_10",
                "team_pit_stop_avg",
                "constructor_dnf_rate_season",
                "pace_delta_last_3",
            ]
        ],
        how="left",
        on=["constructor_id", "season", "round"],
    )
    return frame


def _apply_season_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-race season-context features using strictly past races."""

    season_driver_grouped = frame.groupby(["season", "driver_id"], sort=False)
    frame["driver_points_before"] = season_driver_grouped["points"].cumsum() - frame["points"]
    frame["leader_points_before"] = frame.groupby(["season", "round"])["driver_points_before"].transform("max")
    season_total_rounds = frame.groupby("season")["round"].transform("max").astype(int)
    races_remaining = np.clip(season_total_rounds - frame["round"].astype(int) + 1, 1, None)
    frame["championship_pressure"] = (
        frame["driver_points_before"] - frame["leader_points_before"]
    ) / races_remaining
    frame["new_part_race"] = frame["round"].astype(int).le(3).astype(int)
    return frame


def _apply_elo_features(
    frame: pd.DataFrame,
    driver_elo_history: pd.DataFrame,
    constructor_elo_history: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pre-race dry and wet ELO values into the feature matrix."""

    validate_columns(
        driver_elo_history,
        ["driver_id", "season", "round", "dry_elo_before", "wet_elo_before"],
        "driver ELO history for features",
    )
    validate_columns(
        constructor_elo_history,
        ["constructor_id", "season", "round", "elo_before"],
        "constructor ELO history for features",
    )

    frame = frame.merge(
        driver_elo_history[
            ["driver_id", "season", "round", "dry_elo_before", "wet_elo_before"]
        ].rename(columns={"dry_elo_before": "driver_elo", "wet_elo_before": "driver_raw_wet_elo"}),
        how="left",
        on=["driver_id", "season", "round"],
    )
    frame = frame.merge(
        constructor_elo_history[["constructor_id", "season", "round", "elo_before"]].rename(
            columns={"elo_before": "constructor_elo"}
        ),
        how="left",
        on=["constructor_id", "season", "round"],
    )
    frame["driver_wet_elo"] = np.where(
        frame["race_condition"].isin(["wet", "mixed"]),
        frame["driver_raw_wet_elo"],
        frame["driver_elo"],
    )
    frame["wet_elo_vs_field"] = frame["driver_wet_elo"] - frame.groupby(["season", "round"])["driver_wet_elo"].transform("mean")
    return frame


def _apply_tsi_features(frame: pd.DataFrame, tsi_driver_race: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-race TSI features into the training frame."""

    if tsi_driver_race.empty:
        for column in [
            "tsi_qualifying",
            "tsi_race",
            "tsi_composite",
            "tsi_rolling_20",
            "tsi_delta",
            "tsi_wet",
            "tsi_dry",
            "tsi_sample_size",
        ]:
            frame[column] = np.nan if column != "tsi_sample_size" else 0.0
        return frame

    merge_columns = [
        "season",
        "round",
        "driver_id",
        "tsi_qualifying_before",
        "tsi_race_before",
        "tsi_composite_before",
        "tsi_rolling_20_before",
        "tsi_delta_before",
        "tsi_wet_before",
        "tsi_dry_before",
        "tsi_sample_size_before",
    ]
    merged = frame.merge(
        tsi_driver_race[merge_columns].rename(
            columns={
                "tsi_qualifying_before": "tsi_qualifying",
                "tsi_race_before": "tsi_race",
                "tsi_composite_before": "tsi_composite",
                "tsi_rolling_20_before": "tsi_rolling_20",
                "tsi_delta_before": "tsi_delta",
                "tsi_wet_before": "tsi_wet",
                "tsi_dry_before": "tsi_dry",
                "tsi_sample_size_before": "tsi_sample_size",
            }
        ),
        how="left",
        on=["season", "round", "driver_id"],
    )
    return merged


def _apply_cpi_features(frame: pd.DataFrame, cpi_history: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-race CPI features into the training frame."""

    if cpi_history.empty:
        for column in [
            "cpi_qualifying",
            "cpi_race",
            "cpi_reliability",
            "cpi_performance",
            "cpi_development_rate",
            "cpi_regulation_adaptability",
        ]:
            frame[column] = np.nan
        return frame

    merge_columns = [
        "season",
        "round",
        "constructor_id",
        "cpi_qualifying_before",
        "cpi_race_before",
        "cpi_reliability_before",
        "cpi_performance_before",
        "cpi_development_rate_before",
        "cpi_regulation_adaptability",
    ]
    merged = frame.merge(
        cpi_history[merge_columns].rename(
            columns={
                "cpi_qualifying_before": "cpi_qualifying",
                "cpi_race_before": "cpi_race",
                "cpi_reliability_before": "cpi_reliability",
                "cpi_performance_before": "cpi_performance",
                "cpi_development_rate_before": "cpi_development_rate",
            }
        ),
        how="left",
        on=["season", "round", "constructor_id"],
    )
    return merged


def _compute_sc_probability_lookup(frame: pd.DataFrame) -> dict[str, float]:
    """Estimate circuit safety-car probability from historical disruption proxies."""

    race_level = (
        frame.groupby(["season", "round", "circuit_id"], as_index=False)
        .agg(
            accident_dnfs=("dnf_reason", lambda series: int(series.eq("accident").sum())),
            total_dnfs=("dnf_flag", "sum"),
            wet_indicator=("is_wet_race", "max"),
            mixed_indicator=("is_mixed_race", "max"),
        )
    )
    race_level["sc_proxy"] = (
        race_level["accident_dnfs"].ge(1)
        | race_level["total_dnfs"].ge(3)
        | race_level["wet_indicator"].ge(1)
        | race_level["mixed_indicator"].ge(1)
    ).astype(float)
    lookup = race_level.groupby("circuit_id")["sc_proxy"].mean().to_dict()
    for circuit_id, metadata in CIRCUIT_DNA.items():
        lookup.setdefault(circuit_id, float(metadata["sc_probability"]))
    return {str(key): float(value) for key, value in lookup.items()}


def circuit_dna_values(circuit_id: str, sc_probability_lookup: dict[str, float] | None = None) -> dict[str, Any]:
    """Return manual circuit DNA with a data-derived SC probability override."""

    metadata = CIRCUIT_DNA.get(
        str(circuit_id),
        {
            "high_speed_pct": 0.45,
            "street_circuit": False,
            "tyre_deg_severity": 3,
            "overtaking_difficulty": 5,
            "altitude_m": 100,
            "lap_length_km": 5.0,
            "sc_probability": 0.40,
            "drs_zones": 2,
        },
    )
    sc_probability = float(
        (sc_probability_lookup or {}).get(str(circuit_id), float(metadata["sc_probability"]))
    )
    return {
        "circuit_overtake_difficulty": float(metadata["overtaking_difficulty"]),
        "is_street_circuit": int(bool(metadata["street_circuit"])),
        "circuit_high_speed_pct": float(metadata["high_speed_pct"]),
        "circuit_tyre_deg_severity": float(metadata["tyre_deg_severity"]),
        "circuit_altitude_m": float(metadata["altitude_m"]),
        "circuit_lap_length_km": float(metadata["lap_length_km"]),
        "circuit_sc_probability": sc_probability,
        "circuit_drs_zones": float(metadata["drs_zones"]),
    }


def _apply_circuit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach static circuit metadata plus a data-derived SC probability."""

    sc_lookup = _compute_sc_probability_lookup(frame)
    circuit_features = frame["circuit_id"].apply(lambda circuit_id: circuit_dna_values(circuit_id, sc_lookup)).apply(pd.Series)
    frame = pd.concat([frame.reset_index(drop=True), circuit_features.reset_index(drop=True)], axis=1)
    frame["track_risk"] = frame["circuit_id"].map(track_risk_value).astype(float)
    return frame


def _apply_circuit_tsi_features(
    frame: pd.DataFrame,
    circuit_tsi_matrix: pd.DataFrame,
    cluster_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach car-adjusted circuit performance features to the feature matrix."""
    if circuit_tsi_matrix.empty:
        default_tsi = pd.DataFrame(
            {
                "circuit_tsi_score": 0.0,
                "circuit_tsi_confidence": 0.0,
                "circuit_tsi_sample_size": 0,
                "circuit_tsi_std": 0.0,
                "circuit_type_tsi": 0.0,
                "circuit_home_race": 0,
            },
            index=frame.index,
        )
        return pd.concat([frame, default_tsi], axis=1)

    # Merge circuit TSI _before columns (strict lag — no future leakage)
    tsi_cols = [
        "driver_id", "circuit_id", "season", "round",
        "circuit_tsi_score_before", "circuit_tsi_confidence_before",
        "circuit_tsi_sample_size_before", "circuit_tsi_std_before",
    ]
    available = [c for c in tsi_cols if c in circuit_tsi_matrix.columns]
    tsi_merge = circuit_tsi_matrix[available].drop_duplicates(
        subset=["driver_id", "circuit_id", "season", "round"], keep="last",
    )
    frame = frame.merge(tsi_merge, on=["driver_id", "circuit_id", "season", "round"], how="left")
    lagged_tsi = pd.DataFrame(
        {
            "circuit_tsi_score": frame.pop("circuit_tsi_score_before").astype(float) if "circuit_tsi_score_before" in frame.columns else 0.0,
            "circuit_tsi_confidence": frame.pop("circuit_tsi_confidence_before").astype(float) if "circuit_tsi_confidence_before" in frame.columns else 0.0,
            "circuit_tsi_sample_size": frame.pop("circuit_tsi_sample_size_before").fillna(0).astype(int) if "circuit_tsi_sample_size_before" in frame.columns else 0,
            "circuit_tsi_std": frame.pop("circuit_tsi_std_before").astype(float) if "circuit_tsi_std_before" in frame.columns else 0.0,
        },
        index=frame.index,
    )
    frame = frame.drop(columns=lagged_tsi.columns.intersection(frame.columns), errors="ignore")
    frame = pd.concat([frame, lagged_tsi], axis=1)

    # Circuit type TSI: driver's average performance across circuits in the same archetype cluster
    if cluster_frame is not None and not cluster_frame.empty and "cluster_id" in cluster_frame.columns:
        circuit_to_cluster = dict(zip(cluster_frame["circuit_id"].astype(str), cluster_frame["cluster_id"].astype(int)))
        frame["_target_cluster"] = frame["circuit_id"].map(circuit_to_cluster)

        # For each driver-race, compute archetype TSI from prior circuit_tsi data in same cluster
        circuit_tsi_with_cluster = circuit_tsi_matrix.copy()
        circuit_tsi_with_cluster["_cluster"] = circuit_tsi_with_cluster["circuit_id"].map(circuit_to_cluster)

        def _lookup_archetype_tsi(row: pd.Series) -> float:
            target_cluster = row.get("_target_cluster")
            if pd.isna(target_cluster):
                return float("nan")
            prior = circuit_tsi_with_cluster.loc[
                (circuit_tsi_with_cluster["driver_id"] == row["driver_id"])
                & (circuit_tsi_with_cluster["_cluster"] == int(target_cluster))
                & (
                    (circuit_tsi_with_cluster["season"] < int(row["season"]))
                    | (
                        (circuit_tsi_with_cluster["season"] == int(row["season"]))
                        & (circuit_tsi_with_cluster["round"] < int(row["round"]))
                    )
                )
            ]
            if prior.empty:
                return float("nan")
            latest_per = prior.sort_values(["season", "round"]).groupby("circuit_id").tail(1)
            scores = latest_per["circuit_tsi_score_after"].dropna()
            return float(scores.mean()) if not scores.empty else float("nan")

        frame["circuit_type_tsi"] = frame.apply(_lookup_archetype_tsi, axis=1)
        frame = frame.drop(columns=["_target_cluster"], errors="ignore")
    else:
        frame["circuit_type_tsi"] = np.nan

    # Home race indicator
    frame["circuit_home_race"] = frame.apply(
        lambda row: int(str(row["circuit_id"]) in get_driver_home_circuits(str(row["driver_id"]))),
        axis=1,
    )
    return frame


def _apply_fastf1_fallbacks(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill qualifying-session FastF1 features with conservative historical fallbacks."""

    grouped = frame.groupby("driver_id", sort=False)
    for column in QUALIFYING_FEATURE_COLUMNS:
        historical_mean = grouped[column].transform(lambda series: series.shift(1).rolling(5, min_periods=1).mean())
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(historical_mean)
    return frame


def _finalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill conservative defaults for sparse early-career rows."""

    frame["driver_elo"] = frame["driver_elo"].fillna(1500.0)
    frame["driver_raw_wet_elo"] = frame["driver_raw_wet_elo"].fillna(1500.0)
    frame["driver_wet_elo"] = frame["driver_wet_elo"].fillna(frame["driver_elo"])
    frame["tsi_qualifying"] = frame["tsi_qualifying"].fillna(0.0)
    frame["tsi_race"] = frame["tsi_race"].fillna(0.0)
    frame["tsi_composite"] = frame["tsi_composite"].fillna(0.0)
    frame["tsi_rolling_20"] = frame["tsi_rolling_20"].fillna(frame["tsi_composite"])
    frame["tsi_delta"] = frame["tsi_delta"].fillna(0.0)
    frame["tsi_wet"] = frame["tsi_wet"].fillna(frame["tsi_composite"])
    frame["tsi_dry"] = frame["tsi_dry"].fillna(frame["tsi_composite"])
    frame["tsi_sample_size"] = frame["tsi_sample_size"].fillna(0.0)
    frame["wet_elo_vs_field"] = frame["wet_elo_vs_field"].fillna(0.0)
    frame["wet_race_win_rate"] = frame["wet_race_win_rate"].fillna(0.0)
    frame["wet_race_podium_rate"] = frame["wet_race_podium_rate"].fillna(0.0)
    frame["constructor_elo"] = frame["constructor_elo"].fillna(1500.0)
    frame["cpi_qualifying"] = frame["cpi_qualifying"].fillna(0.0)
    frame["cpi_race"] = frame["cpi_race"].fillna(0.0)
    frame["cpi_reliability"] = frame["cpi_reliability"].fillna(frame["team_reliability_last_10"].fillna(1.0))
    frame["cpi_performance"] = frame["cpi_performance"].fillna(0.0)
    frame["cpi_development_rate"] = frame["cpi_development_rate"].fillna(0.0)
    frame["cpi_regulation_adaptability"] = frame["cpi_regulation_adaptability"].fillna(frame["cpi_performance"])
    frame["avg_finish_last_5"] = frame["avg_finish_last_5"].fillna(frame["qualifying_position"])
    frame["avg_qualifying_last_5"] = frame["avg_qualifying_last_5"].fillna(frame["qualifying_position"])
    frame["win_rate_career"] = frame["win_rate_career"].fillna(0.0)
    frame["podium_rate_career"] = frame["podium_rate_career"].fillna(0.0)
    frame["dnf_rate_last_10"] = frame["dnf_rate_last_10"].fillna(0.0)
    frame["laps_led_last_3"] = frame["laps_led_last_3"].fillna(0.0)
    frame["consistency_score"] = frame["consistency_score"].fillna(frame["finish_position_filled"].std(ddof=0))
    frame["circuit_avg_finish"] = frame["circuit_avg_finish"].fillna(frame["avg_finish_last_5"])
    frame["circuit_best_finish"] = frame["circuit_best_finish"].fillna(frame["avg_finish_last_5"])
    frame["races_at_circuit"] = frame["races_at_circuit"].fillna(0).astype(int)
    frame["circuit_type_tsi"] = frame["circuit_type_tsi"].fillna(0.0) if "circuit_type_tsi" in frame.columns else 0.0
    frame["circuit_tsi_score"] = frame["circuit_tsi_score"].fillna(
        frame["circuit_type_tsi"] if "circuit_type_tsi" in frame.columns else 0.0
    ) if "circuit_tsi_score" in frame.columns else 0.0
    frame["circuit_tsi_confidence"] = frame["circuit_tsi_confidence"].fillna(0.0) if "circuit_tsi_confidence" in frame.columns else 0.0
    frame["circuit_tsi_sample_size"] = frame["circuit_tsi_sample_size"].fillna(0).astype(int) if "circuit_tsi_sample_size" in frame.columns else 0
    frame["circuit_tsi_std"] = frame["circuit_tsi_std"].fillna(0.0) if "circuit_tsi_std" in frame.columns else 0.0
    frame["circuit_home_race"] = frame["circuit_home_race"].fillna(0).astype(int) if "circuit_home_race" in frame.columns else 0
    frame["team_avg_finish_last_5"] = frame["team_avg_finish_last_5"].fillna(frame["avg_finish_last_5"])
    frame["team_reliability_last_10"] = frame["team_reliability_last_10"].fillna(1.0)
    frame["constructor_dnf_rate_season"] = frame["constructor_dnf_rate_season"].fillna(
        1.0 - frame["team_reliability_last_10"]
    )
    frame["pace_delta_last_3"] = frame["pace_delta_last_3"].fillna(0.0)
    pit_stop_default = frame["team_pit_stop_avg"].median()
    if pd.isna(pit_stop_default):
        pit_stop_default = 2500.0
    frame["team_pit_stop_avg"] = frame["team_pit_stop_avg"].fillna(pit_stop_default)
    frame["season_round"] = frame["round"].astype(int)
    frame["teammate_gap_qualifying"] = frame["teammate_gap_qualifying"].fillna(0.0)
    frame["championship_pressure"] = frame["championship_pressure"].fillna(0.0)
    frame["dnf_mechanical_last_3"] = frame["dnf_mechanical_last_3"].fillna(0.0)
    frame["dnf_accident_last_3"] = frame["dnf_accident_last_3"].fillna(0.0)
    frame["teammate_dnf_same_race"] = frame["teammate_dnf_same_race"].fillna(0.0)
    frame["new_part_race"] = frame["new_part_race"].fillna(frame["season_round"].le(3).astype(int)).astype(int)
    frame["track_risk"] = frame["track_risk"].fillna(0.52)
    frame["race_precipitation_mm"] = frame["race_precipitation_mm"].fillna(0.0)
    frame["race_max_wind_kmh"] = frame["race_max_wind_kmh"].fillna(frame["race_max_wind_kmh"].median())
    frame["race_temperature_c"] = frame["race_temperature_c"].fillna(frame["race_temperature_c"].median())
    frame["is_wet_race"] = frame["is_wet_race"].fillna(0).astype(int)
    frame["is_mixed_race"] = frame["is_mixed_race"].fillna(0).astype(int)
    frame["circuit_overtake_difficulty"] = frame["circuit_overtake_difficulty"].fillna(5.0)
    frame["is_street_circuit"] = frame["is_street_circuit"].fillna(0).astype(int)
    frame["circuit_high_speed_pct"] = frame["circuit_high_speed_pct"].fillna(0.45)
    frame["circuit_tyre_deg_severity"] = frame["circuit_tyre_deg_severity"].fillna(3.0)
    frame["circuit_altitude_m"] = frame["circuit_altitude_m"].fillna(100.0)
    frame["circuit_lap_length_km"] = frame["circuit_lap_length_km"].fillna(5.0)
    frame["circuit_sc_probability"] = frame["circuit_sc_probability"].fillna(0.40)
    frame["circuit_drs_zones"] = frame["circuit_drs_zones"].fillna(2.0)
    for column in QUALIFYING_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in OPENF1_FALLBACK_FEATURE_COLUMNS + PRACTICE_FEATURE_COLUMNS + QUALIFYING_DEEP_FEATURE_COLUMNS + TRACK_EVOLUTION_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in SESSION_CONTEXT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["weekend_session_count", "has_fp1", "has_fp2", "has_fp3", "has_quali", "has_race", "has_sprint", "has_sprint_shootout"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    for column in ["fp_ran_soft", "fp_ran_medium", "fp_ran_hard", "is_sprint_weekend"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    for column in ["sprint_points_scored", "sprint_compounds_used_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in RACE_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_feature_matrix(
    race_results: pd.DataFrame | None = None,
    driver_elo_history: pd.DataFrame | None = None,
    constructor_elo_history: pd.DataFrame | None = None,
    pitstops_df: pd.DataFrame | None = None,
    lap_lead_summary: pd.DataFrame | None = None,
    fastf1_features_df: pd.DataFrame | None = None,
    tsi_driver_race: pd.DataFrame | None = None,
    cpi_history: pd.DataFrame | None = None,
    circuit_tsi_matrix: pd.DataFrame | None = None,
    cluster_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full model-ready feature matrix."""

    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    if driver_elo_history is None:
        driver_elo_history = load_dataframe(CONFIG["driver_elo_path"])
    if constructor_elo_history is None:
        constructor_elo_history = load_dataframe(CONFIG["constructor_elo_path"])
    if pitstops_df is None:
        pitstops_df = load_dataframe(CONFIG["pitstops_path"])
    if lap_lead_summary is None:
        lap_lead_summary = _load_lap_lead_summary()
    if fastf1_features_df is None:
        fastf1_features_df = _load_fastf1_features()
    if tsi_driver_race is None:
        tsi_driver_race = _load_tsi_driver_race(race_results=race_results)
    if cpi_history is None:
        cpi_history = _load_cpi_history(race_results=race_results, tsi_driver_race=tsi_driver_race)
    if circuit_tsi_matrix is None:
        ctsi_path = CONFIG["processed_dir"] / "circuit_tsi_matrix.csv"
        if ctsi_path.exists():
            circuit_tsi_matrix = load_dataframe(ctsi_path)
        else:
            circuit_tsi_matrix = build_circuit_tsi_artifacts(
                race_results=race_results, cpi_history=cpi_history,
            ).circuit_tsi_matrix
    if cluster_frame is None:
        cluster_path = CONFIG["processed_dir"] / "circuit_clusters.csv"
        if cluster_path.exists():
            cluster_frame = load_dataframe(cluster_path)
        else:
            cluster_frame = pd.DataFrame()

    frame = _prepare_results(race_results)
    frame = _attach_base_enrichment(frame, lap_lead_summary, fastf1_features_df)
    frame = _apply_elo_features(frame, driver_elo_history, constructor_elo_history)
    frame = _apply_tsi_features(frame, tsi_driver_race)
    frame = _apply_driver_features(frame)
    frame = _apply_team_features(frame, pitstops_df)
    frame = _apply_cpi_features(frame, cpi_history)
    frame = _apply_season_context_features(frame)
    frame = _apply_circuit_features(frame)
    frame = _apply_circuit_tsi_features(frame, circuit_tsi_matrix, cluster_frame)
    frame = _apply_fastf1_fallbacks(frame)
    frame = _finalize_features(frame)

    finish_position = frame["finish_position_filled"]
    target_columns = pd.DataFrame(
        {
            "finish_position": finish_position,
            "top1_finish": finish_position.eq(1).astype(int),
            "top3_finish": finish_position.le(3).astype(int),
            "top10_finish": finish_position.le(10).astype(int),
            "points_scored": frame["points"].astype(float),
        },
        index=frame.index,
    )
    frame = frame.drop(columns=target_columns.columns.intersection(frame.columns), errors="ignore")
    frame = pd.concat([frame, target_columns], axis=1)

    selected_columns = _unique_columns(
        [
            "season",
            "round",
            "race_name",
            "race_date",
            "circuit_id",
            "circuit_name",
            "driver_id",
            "driver_name",
            "driver_number",
            "constructor_id",
            "constructor_name",
            *FEATURE_COLUMNS,
            *DNF_FEATURE_COLUMNS,
            *ANALYTICS_ONLY_COLUMNS,
            "finish_position",
            "top1_finish",
            "top3_finish",
            "top10_finish",
            "points_scored",
            "dnf_flag",
        ]
    )
    feature_matrix = frame[selected_columns].sort_values(["season", "round", "qualifying_position", "driver_id"])

    validate_columns(feature_matrix, FEATURE_COLUMNS, "feature matrix")
    validate_columns(feature_matrix, DNF_FEATURE_COLUMNS, "DNF feature matrix")
    validate_columns(
        feature_matrix,
        ["finish_position", "top1_finish", "top3_finish", "top10_finish", "points_scored"],
        "feature matrix targets",
    )
    return feature_matrix.reset_index(drop=True)


def run_feature_pipeline(
    race_results: pd.DataFrame | None = None,
    driver_elo_history: pd.DataFrame | None = None,
    constructor_elo_history: pd.DataFrame | None = None,
    pitstops_df: pd.DataFrame | None = None,
    fastf1_features_df: pd.DataFrame | None = None,
    tsi_driver_race: pd.DataFrame | None = None,
    cpi_history: pd.DataFrame | None = None,
    circuit_tsi_matrix: pd.DataFrame | None = None,
    cluster_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build and persist engineered features."""

    ensure_directories()
    features = build_feature_matrix(
        race_results=race_results,
        driver_elo_history=driver_elo_history,
        constructor_elo_history=constructor_elo_history,
        pitstops_df=pitstops_df,
        fastf1_features_df=fastf1_features_df,
        tsi_driver_race=tsi_driver_race,
        cpi_history=cpi_history,
        circuit_tsi_matrix=circuit_tsi_matrix,
        cluster_frame=cluster_frame,
    )
    save_dataframe(features, CONFIG["features_path"])
    save_json(
        {
            "feature_columns": FEATURE_COLUMNS,
            "dnf_feature_columns": DNF_FEATURE_COLUMNS,
            "analytics_only_columns": ANALYTICS_ONLY_COLUMNS,
        },
        CONFIG["feature_columns_path"],
    )
    LOGGER.info("Saved feature matrix with %s rows and %s feature columns", len(features), len(FEATURE_COLUMNS))
    return features


def _historical_cutoff(frame: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    """Return all rows before a target race."""

    return frame.loc[
        (frame["season"] < season) | ((frame["season"] == season) & (frame["round"] < round_number))
    ].copy()


def _team_pit_stop_average(
    pitstops_df: pd.DataFrame,
    race_results: pd.DataFrame,
    constructor_id: str,
    season: int,
    round_number: int,
) -> float:
    """Compute a constructor's pre-race average pit stop time over the last five races."""

    if pitstops_df.empty:
        return float("nan")
    pitstop_team = _build_pitstop_team_frame(pitstops_df, race_results)
    pitstop_team = pitstop_team.loc[
        (pitstop_team["constructor_id"] == constructor_id)
        & ((pitstop_team["season"] < season) | ((pitstop_team["season"] == season) & (pitstop_team["round"] < round_number)))
    ].sort_values(["season", "round"])
    if pitstop_team.empty:
        return float("nan")
    return float(pitstop_team.tail(5)["team_race_pit_stop_avg"].mean())


def _rolling_driver_mean(history: pd.DataFrame, column: str, window: int = 5) -> float:
    """Return a recent historical mean for an inference-time fallback column."""

    if history.empty or column not in history.columns:
        return float("nan")
    series = pd.to_numeric(history[column], errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float(series.tail(window).mean())


def _resolve_inference_weather(
    frame: pd.DataFrame,
    current_race_frame: pd.DataFrame,
    circuit_id: str,
    weather_mode: str | None,
) -> dict[str, Any]:
    """Return the weather context to use for a race prediction."""

    normalized_mode = str(weather_mode).strip().lower() if weather_mode else None
    if normalized_mode not in {None, "dry", "mixed", "wet"}:
        normalized_mode = None

    source = current_race_frame.copy()
    if source.empty:
        source = frame.loc[frame["circuit_id"] == circuit_id].copy()
    if source.empty:
        source = frame.copy()

    precipitation = float(pd.to_numeric(source.get("race_precipitation_mm"), errors="coerce").median()) if "race_precipitation_mm" in source else 0.0
    wind = float(pd.to_numeric(source.get("race_max_wind_kmh"), errors="coerce").median()) if "race_max_wind_kmh" in source else 0.0
    temperature = float(pd.to_numeric(source.get("race_temperature_c"), errors="coerce").median()) if "race_temperature_c" in source else 25.0
    condition = normalized_mode or (
        str(current_race_frame["race_condition"].iloc[0]) if not current_race_frame.empty and "race_condition" in current_race_frame.columns else "dry"
    )

    if condition == "wet":
        precipitation = max(precipitation, 3.0)
    elif condition == "mixed":
        precipitation = min(max(precipitation, 0.8), 2.0)
    else:
        precipitation = 0.0 if normalized_mode == "dry" else precipitation

    return {
        "race_condition": condition,
        "race_precipitation_mm": precipitation,
        "race_max_wind_kmh": wind,
        "race_temperature_c": temperature,
        "is_wet_race": int(condition == "wet"),
        "is_mixed_race": int(condition == "mixed"),
    }


def _prior_tsi_snapshot(
    tsi_driver_race: pd.DataFrame,
    driver_id: str,
    season: int,
    round_number: int,
) -> dict[str, Any]:
    """Resolve the latest available TSI snapshot before a target race."""

    if tsi_driver_race.empty:
        return {}
    exact = tsi_driver_race.loc[
        (tsi_driver_race["driver_id"] == str(driver_id))
        & (tsi_driver_race["season"] == int(season))
        & (tsi_driver_race["round"] == int(round_number))
    ].tail(1)
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "tsi_qualifying": row["tsi_qualifying_before"],
            "tsi_race": row["tsi_race_before"],
            "tsi_composite": row["tsi_composite_before"],
            "tsi_rolling_20": row["tsi_rolling_20_before"],
            "tsi_delta": row["tsi_delta_before"],
            "tsi_wet": row["tsi_wet_before"],
            "tsi_dry": row["tsi_dry_before"],
            "tsi_sample_size": row["tsi_sample_size_before"],
        }
    prior = tsi_driver_race.loc[
        (tsi_driver_race["driver_id"] == str(driver_id))
        & (
            (tsi_driver_race["season"] < int(season))
            | ((tsi_driver_race["season"] == int(season)) & (tsi_driver_race["round"] < int(round_number)))
        )
    ].sort_values(["season", "round"]).tail(1)
    if prior.empty:
        return {}
    row = prior.iloc[0]
    return {
        "tsi_qualifying": row["tsi_qualifying_after"],
        "tsi_race": row["tsi_race_after"],
        "tsi_composite": row["tsi_composite_after"],
        "tsi_rolling_20": row["tsi_rolling_20_after"],
        "tsi_delta": row["tsi_delta_after"],
        "tsi_wet": row["tsi_wet_after"],
        "tsi_dry": row["tsi_dry_after"],
        "tsi_sample_size": row["tsi_sample_size_after"],
    }


def _prior_cpi_snapshot(
    cpi_history: pd.DataFrame,
    constructor_id: str,
    season: int,
    round_number: int,
) -> dict[str, Any]:
    """Resolve the latest available CPI snapshot before a target race."""

    if cpi_history.empty:
        return {}
    exact = cpi_history.loc[
        (cpi_history["constructor_id"] == str(constructor_id))
        & (cpi_history["season"] == int(season))
        & (cpi_history["round"] == int(round_number))
    ].tail(1)
    if not exact.empty:
        row = exact.iloc[0]
        return {
            "cpi_qualifying": row["cpi_qualifying_before"],
            "cpi_race": row["cpi_race_before"],
            "cpi_reliability": row["cpi_reliability_before"],
            "cpi_performance": row["cpi_performance_before"],
            "cpi_development_rate": row["cpi_development_rate_before"],
            "cpi_regulation_adaptability": row["cpi_regulation_adaptability"],
        }
    prior = cpi_history.loc[
        (cpi_history["constructor_id"] == str(constructor_id))
        & (
            (cpi_history["season"] < int(season))
            | ((cpi_history["season"] == int(season)) & (cpi_history["round"] < int(round_number)))
        )
    ].sort_values(["season", "round"]).tail(1)
    if prior.empty:
        return {}
    row = prior.iloc[0]
    return {
        "cpi_qualifying": row["cpi_qualifying_after"],
        "cpi_race": row["cpi_race_after"],
        "cpi_reliability": row["cpi_reliability_after"],
        "cpi_performance": row["cpi_performance_after"],
        "cpi_development_rate": row["cpi_development_rate_after"],
        "cpi_regulation_adaptability": row["cpi_regulation_adaptability"],
    }


def _inference_circuit_tsi(
    driver_id: str,
    circuit_id: str,
    season: int,
    round_number: int,
) -> dict[str, Any]:
    """Compute circuit TSI features for a single driver at inference time."""
    snapshot = get_circuit_tsi(driver_id, circuit_id, season, round_number)
    archetype = get_archetype_tsi(driver_id, circuit_id, season, round_number)
    score = float(snapshot.get("circuit_tsi_score", 0.0) or 0.0)
    confidence = float(snapshot.get("circuit_tsi_confidence", 0.0))
    # Fallback cascade: if no circuit-specific data, use archetype
    if confidence == 0.0 and archetype != 0.0:
        score = archetype
    return {
        "circuit_tsi_score": score,
        "circuit_tsi_confidence": confidence,
        "circuit_tsi_sample_size": int(snapshot.get("circuit_tsi_sample_size", 0)),
        "circuit_tsi_std": float(snapshot.get("circuit_tsi_std", 0.0) or 0.0),
        "circuit_type_tsi": archetype,
        "circuit_home_race": int(str(circuit_id) in get_driver_home_circuits(str(driver_id))),
    }


def build_race_inference_frame(
    drivers_list: list[str],
    qualifying_positions: list[int] | dict[str, int],
    circuit_id: str,
    season: int,
    round_number: int,
    race_results: pd.DataFrame | None = None,
    pitstops_df: pd.DataFrame | None = None,
    lap_lead_summary: pd.DataFrame | None = None,
    fastf1_features_df: pd.DataFrame | None = None,
    tsi_driver_race: pd.DataFrame | None = None,
    cpi_history: pd.DataFrame | None = None,
    weather_mode: str | None = None,
) -> pd.DataFrame:
    """Build a model-ready feature frame for an upcoming race."""

    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    if pitstops_df is None:
        pitstops_df = load_dataframe(CONFIG["pitstops_path"])
    if lap_lead_summary is None:
        lap_lead_summary = _load_lap_lead_summary()
    if fastf1_features_df is None:
        fastf1_features_df = _load_fastf1_features()
    if tsi_driver_race is None:
        tsi_driver_race = _load_tsi_driver_race(race_results=race_results)
    if cpi_history is None:
        cpi_history = _load_cpi_history(race_results=race_results, tsi_driver_race=tsi_driver_race)

    frame = _prepare_results(race_results)
    frame = _attach_base_enrichment(frame, lap_lead_summary, fastf1_features_df)
    history = _historical_cutoff(frame, season, round_number)
    current_race_frame = frame.loc[(frame["season"] == season) & (frame["round"] == round_number)].copy()
    current_weekend_features = pd.DataFrame()
    if not fastf1_features_df.empty:
        current_weekend_features = fastf1_features_df.copy()
        current_weekend_features["driver_number"] = current_weekend_features["driver_number"].apply(_normalize_driver_number)
        current_weekend_features = current_weekend_features.loc[
            (current_weekend_features["season"] == int(season)) & (current_weekend_features["round"] == int(round_number))
        ].drop_duplicates(subset=["season", "round", "driver_number"], keep="last")

    if isinstance(qualifying_positions, list):
        qualifying_map = {driver_id: qualifying_positions[idx] for idx, driver_id in enumerate(drivers_list)}
    else:
        qualifying_map = qualifying_positions

    latest_driver_rows = (
        history.sort_values(["season", "round"])
        .groupby("driver_id", as_index=False)
        .tail(1)
        .set_index("driver_id")
    )
    latest_constructor_map = {
        driver_id: str(latest_driver_rows.loc[driver_id, "constructor_id"])
        for driver_id in drivers_list
        if driver_id in latest_driver_rows.index
    }
    season_history = history.loc[history["season"] == season].copy()
    season_calendar = load_calendar(season, allow_fetch=False)
    season_total_rounds = (
        int(season_calendar["round"].max())
        if not season_calendar.empty
        else int(frame.loc[frame["season"] == season, "round"].max()) if not frame.loc[frame["season"] == season].empty else int(round_number)
    )
    season_points = season_history.groupby("driver_id")["points"].sum() if not season_history.empty else pd.Series(dtype=float)
    leader_points_before = float(season_points.max()) if not season_points.empty else 0.0
    weather_context = _resolve_inference_weather(frame, current_race_frame, circuit_id, weather_mode)
    sc_lookup = _compute_sc_probability_lookup(frame)
    roster_lookup = {row["driver_id"]: row for row in get_season_roster(season)}
    calendar_race = season_calendar.loc[season_calendar["round"] == int(round_number)].head(1) if not season_calendar.empty else pd.DataFrame()
    calendar_race_name = (
        str(calendar_race["race_name"].iloc[0])
        if not calendar_race.empty and "race_name" in calendar_race.columns
        else f"{circuit_id.replace('_', ' ').title()} Grand Prix"
    )
    calendar_circuit_name = (
        str(calendar_race["circuit_name"].iloc[0])
        if not calendar_race.empty and "circuit_name" in calendar_race.columns
        else circuit_id.replace("_", " ").title()
    )

    rows: list[dict[str, Any]] = []
    for driver_id in drivers_list:
        driver_history = history.loc[history["driver_id"] == driver_id].sort_values(["season", "round"])
        latest_row = latest_driver_rows.loc[driver_id] if driver_id in latest_driver_rows.index else None
        roster_row = roster_lookup.get(driver_id, {})
        constructor_id = str(latest_row["constructor_id"]) if latest_row is not None else str(
            roster_row.get("constructor_id") or get_likely_team(driver_id, season) or "unknown"
        )
        constructor_name = str(latest_row["constructor_name"]) if latest_row is not None else str(
            roster_row.get("constructor_name") or get_constructor_name(constructor_id)
        )
        driver_name = str(latest_row["driver_name"]) if latest_row is not None else str(
            roster_row.get("driver_name") or get_driver_name(driver_id)
        )
        driver_number = _normalize_driver_number(latest_row["driver_number"]) if latest_row is not None else None

        circuit_history = driver_history.loc[driver_history["circuit_id"] == circuit_id]
        team_history = history.loc[history["constructor_id"] == constructor_id].sort_values(["season", "round"])
        team_race_summary = (
            team_history.groupby(["season", "round"], as_index=False)
            .agg(team_finish=("finish_position_filled", "mean"), team_dnf=("dnf_flag", "mean"))
            .sort_values(["season", "round"])
        )
        team_season_summary = team_race_summary.loc[team_race_summary["season"] == season].copy()
        current_race_row = current_race_frame.loc[current_race_frame["driver_id"] == driver_id].tail(1)
        current_weekend_row = (
            current_weekend_features.loc[current_weekend_features["driver_number"] == driver_number].head(1)
            if driver_number is not None and not current_weekend_features.empty
            else pd.DataFrame()
        )

        qualifying_position = int(qualifying_map.get(driver_id, len(rows) + 1))
        if not current_race_row.empty:
            teammate_gap_qualifying = float(current_race_row["teammate_gap_qualifying"].iloc[0])
        else:
            teammate_ids = [
                candidate
                for candidate in drivers_list
                if candidate != driver_id and latest_constructor_map.get(candidate) == constructor_id
            ]
            teammate_gap_qualifying = (
                float(np.mean([qualifying_map.get(candidate, qualifying_position) for candidate in teammate_ids]) - qualifying_position)
                / max(len(drivers_list), 1)
                if teammate_ids
                else 0.0
            )

        season_avg_finish = float(team_season_summary["team_finish"].mean()) if not team_season_summary.empty else float(qualifying_position)
        last3_avg_finish = float(team_season_summary.tail(3)["team_finish"].mean()) if not team_season_summary.empty else float(qualifying_position)
        driver_points_before = float(season_points.get(driver_id, 0.0))
        races_remaining = max(season_total_rounds - int(round_number) + 1, 1)
        dry_elo = get_driver_dry_elo(driver_id, season, round_number)
        wet_elo = get_driver_wet_elo(driver_id, season, round_number)
        active_weather_elo = wet_elo if weather_context["race_condition"] in {"wet", "mixed"} else dry_elo
        wet_history_mask = driver_history["race_condition"].isin(["wet", "mixed"]) if "race_condition" in driver_history.columns else pd.Series(False, index=driver_history.index)
        tsi_snapshot = _prior_tsi_snapshot(tsi_driver_race, driver_id=driver_id, season=season, round_number=round_number)
        cpi_snapshot = _prior_cpi_snapshot(cpi_history, constructor_id=constructor_id, season=season, round_number=round_number)

        base_row: dict[str, Any] = {
            "season": int(season),
            "round": int(round_number),
            "race_name": calendar_race_name,
            "circuit_id": circuit_id,
            "circuit_name": str(latest_row["circuit_name"]) if latest_row is not None and latest_row["circuit_id"] == circuit_id else calendar_circuit_name,
            "driver_id": driver_id,
            "driver_name": driver_name,
            "driver_number": driver_number,
            "constructor_id": constructor_id,
            "constructor_name": constructor_name,
            "grid_position": float(qualifying_position),
            "driver_elo": dry_elo,
            "driver_wet_elo": active_weather_elo,
            "tsi_qualifying": float(tsi_snapshot.get("tsi_qualifying", 0.0) or 0.0),
            "tsi_race": float(tsi_snapshot.get("tsi_race", 0.0) or 0.0),
            "tsi_composite": float(tsi_snapshot.get("tsi_composite", 0.0) or 0.0),
            "tsi_rolling_20": float(tsi_snapshot.get("tsi_rolling_20", tsi_snapshot.get("tsi_composite", 0.0)) or 0.0),
            "tsi_delta": float(tsi_snapshot.get("tsi_delta", 0.0) or 0.0),
            "tsi_wet": float(tsi_snapshot.get("tsi_wet", tsi_snapshot.get("tsi_composite", 0.0)) or 0.0),
            "tsi_dry": float(tsi_snapshot.get("tsi_dry", tsi_snapshot.get("tsi_composite", 0.0)) or 0.0),
            "tsi_sample_size": float(tsi_snapshot.get("tsi_sample_size", 0.0) or 0.0),
            "wet_race_win_rate": safe_ratio(float(driver_history.loc[wet_history_mask, "finish_position_filled"].eq(1).sum()), float(wet_history_mask.sum())),
            "wet_race_podium_rate": safe_ratio(float(driver_history.loc[wet_history_mask, "finish_position_filled"].le(3).sum()), float(wet_history_mask.sum())),
            "avg_finish_last_5": float(driver_history.tail(5)["finish_position_filled"].mean()) if not driver_history.empty else float(qualifying_position),
            "avg_qualifying_last_5": float(driver_history.tail(5)["qualifying_position"].mean()) if not driver_history.empty else float(qualifying_position),
            "win_rate_career": safe_ratio(float(driver_history["finish_position_filled"].eq(1).sum()), float(len(driver_history))),
            "podium_rate_career": safe_ratio(float(driver_history["finish_position_filled"].le(3).sum()), float(len(driver_history))),
            "dnf_rate_last_10": float(driver_history.tail(10)["dnf_flag"].mean()) if not driver_history.empty else 0.0,
            "laps_led_last_3": float(driver_history.tail(3)["laps_led_race"].sum()) if not driver_history.empty else 0.0,
            "dnf_mechanical_last_3": float(driver_history.loc[driver_history["dnf_flag"] == 1, "dnf_reason"].tail(3).eq("mechanical").sum()) if not driver_history.empty else 0.0,
            "dnf_accident_last_3": float(driver_history.loc[driver_history["dnf_flag"] == 1, "dnf_reason"].tail(3).eq("accident").sum()) if not driver_history.empty else 0.0,
            "teammate_dnf_same_race": float(driver_history.tail(10)["teammate_dnf_current"].mean()) if not driver_history.empty else 0.0,
            "consistency_score": float(driver_history.tail(10)["finish_position_filled"].std(ddof=0)) if len(driver_history) >= 2 else 0.0,
            "circuit_avg_finish": float(circuit_history["finish_position_filled"].mean()) if not circuit_history.empty else float(driver_history.tail(5)["finish_position_filled"].mean()) if not driver_history.empty else float(qualifying_position),
            "circuit_best_finish": float(circuit_history["finish_position_filled"].min()) if not circuit_history.empty else float(driver_history.tail(5)["finish_position_filled"].mean()) if not driver_history.empty else float(qualifying_position),
            "races_at_circuit": int(len(circuit_history)),
            **_inference_circuit_tsi(driver_id, circuit_id, season, round_number),
            "constructor_elo": get_constructor_elo(constructor_id, season, round_number),
            "cpi_qualifying": float(cpi_snapshot.get("cpi_qualifying", 0.0) or 0.0),
            "cpi_race": float(cpi_snapshot.get("cpi_race", 0.0) or 0.0),
            "cpi_reliability": float(cpi_snapshot.get("cpi_reliability", 1.0) or 1.0),
            "cpi_performance": float(cpi_snapshot.get("cpi_performance", 0.0) or 0.0),
            "cpi_development_rate": float(cpi_snapshot.get("cpi_development_rate", 0.0) or 0.0),
            "cpi_regulation_adaptability": float(cpi_snapshot.get("cpi_regulation_adaptability", 0.0) or 0.0),
            "team_avg_finish_last_5": float(team_race_summary.tail(5)["team_finish"].mean()) if not team_race_summary.empty else float(qualifying_position),
            "team_reliability_last_10": 1.0 - float(team_race_summary.tail(10)["team_dnf"].mean()) if not team_race_summary.empty else 1.0,
            "team_pit_stop_avg": _team_pit_stop_average(pitstops_df, frame, constructor_id, season, round_number),
            "constructor_dnf_rate_season": float(team_season_summary["team_dnf"].mean()) if not team_season_summary.empty else 0.0,
            "pace_delta_last_3": season_avg_finish - last3_avg_finish,
            "qualifying_position": float(qualifying_position),
            "teammate_gap_qualifying": teammate_gap_qualifying,
            "grid_penalty": 0,
            "season_round": int(round_number),
            "championship_pressure": (driver_points_before - leader_points_before) / races_remaining,
            "new_part_race": int(round_number <= 3),
            "track_risk": track_risk_value(circuit_id),
            **weather_context,
            **circuit_dna_values(circuit_id, sc_lookup),
        }
        for column in QUALIFYING_FEATURE_COLUMNS:
            if not current_weekend_row.empty and column in current_weekend_row.columns and pd.notna(current_weekend_row[column].iloc[0]):
                base_row[column] = float(pd.to_numeric(current_weekend_row[column].iloc[0], errors="coerce"))
            elif not current_race_row.empty and column in current_race_row.columns and pd.notna(current_race_row[column].iloc[0]):
                base_row[column] = float(pd.to_numeric(current_race_row[column].iloc[0], errors="coerce"))
            else:
                base_row[column] = _rolling_driver_mean(driver_history, column)

        for column in OPENF1_FALLBACK_FEATURE_COLUMNS + PRACTICE_FEATURE_COLUMNS + QUALIFYING_DEEP_FEATURE_COLUMNS + TRACK_EVOLUTION_FEATURE_COLUMNS:
            if not current_weekend_row.empty and column in current_weekend_row.columns and pd.notna(current_weekend_row[column].iloc[0]):
                base_row[column] = float(pd.to_numeric(current_weekend_row[column].iloc[0], errors="coerce"))
            elif not current_race_row.empty and column in current_race_row.columns and pd.notna(current_race_row[column].iloc[0]):
                base_row[column] = float(pd.to_numeric(current_race_row[column].iloc[0], errors="coerce"))
            else:
                base_row[column] = _rolling_driver_mean(driver_history, column)

        for column in SESSION_CONTEXT_COLUMNS:
            if not current_weekend_row.empty and column in current_weekend_row.columns and pd.notna(current_weekend_row[column].iloc[0]):
                base_row[column] = float(pd.to_numeric(current_weekend_row[column].iloc[0], errors="coerce"))
            elif not current_race_row.empty and column in current_race_row.columns and pd.notna(current_race_row[column].iloc[0]):
                base_row[column] = float(pd.to_numeric(current_race_row[column].iloc[0], errors="coerce"))
            else:
                base_row[column] = float(_rolling_driver_mean(driver_history, column))

        for column in ["has_fp1", "has_fp2", "has_fp3", "has_quali", "has_race", "has_sprint", "has_sprint_shootout", "weekend_session_count"]:
            if pd.isna(base_row.get(column)):
                base_row[column] = 0.0

        is_sprint_source = 0.0
        if not current_weekend_row.empty and "is_sprint_weekend" in current_weekend_row.columns and pd.notna(current_weekend_row["is_sprint_weekend"].iloc[0]):
            is_sprint_source = float(pd.to_numeric(current_weekend_row["is_sprint_weekend"].iloc[0], errors="coerce"))
        elif not current_race_row.empty and "is_sprint_weekend" in current_race_row.columns and pd.notna(current_race_row["is_sprint_weekend"].iloc[0]):
            is_sprint_source = float(pd.to_numeric(current_race_row["is_sprint_weekend"].iloc[0], errors="coerce"))
        base_row["is_sprint_weekend"] = is_sprint_source

        for column in ["sprint_points_scored", "sprint_compounds_used_count"]:
            if pd.isna(base_row.get(column)):
                base_row[column] = 0.0 if not base_row["is_sprint_weekend"] else float(_rolling_driver_mean(driver_history, column))

        for column in ["sprint_finish_position", "sprint_vs_expected", "sprint_primary_compound_code", "sprint_tyre_life_used"]:
            if column not in base_row or pd.isna(base_row[column]):
                base_row[column] = float(_rolling_driver_mean(driver_history, column)) if base_row["is_sprint_weekend"] else np.nan

        for column in ["fp_ran_soft", "fp_ran_medium", "fp_ran_hard"]:
            if column not in base_row or pd.isna(base_row[column]):
                base_row[column] = 0.0
        rows.append(base_row)

    inference_frame = pd.DataFrame(rows)
    inference_frame["wet_elo_vs_field"] = inference_frame["driver_wet_elo"] - inference_frame["driver_wet_elo"].mean()
    pit_stop_default = inference_frame["team_pit_stop_avg"].median()
    if pd.isna(pit_stop_default):
        pit_stop_default = 2500.0
    inference_frame["team_pit_stop_avg"] = inference_frame["team_pit_stop_avg"].fillna(pit_stop_default)
    inference_frame["constructor_dnf_rate_season"] = inference_frame["constructor_dnf_rate_season"].fillna(
        1.0 - inference_frame["team_reliability_last_10"]
    )
    for column in FEATURE_COLUMNS + DNF_FEATURE_COLUMNS:
        inference_frame[column] = pd.to_numeric(inference_frame[column], errors="coerce")
    inference_frame["is_street_circuit"] = inference_frame["is_street_circuit"].fillna(0).astype(int)
    inference_frame["is_wet_race"] = inference_frame["is_wet_race"].fillna(0).astype(int)
    inference_frame["is_mixed_race"] = inference_frame["is_mixed_race"].fillna(0).astype(int)
    validate_columns(inference_frame, FEATURE_COLUMNS, "race inference frame")
    validate_columns(inference_frame, DNF_FEATURE_COLUMNS, "race inference DNF frame")
    return inference_frame.sort_values("qualifying_position").reset_index(drop=True)
