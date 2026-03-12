"""Weekend session ingestion and telemetry feature extraction for recent seasons."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline import OpenF1Client, load_calendar
from src.utils import (
    CONFIG as BASE_CONFIG,
    detect_current_season,
    ensure_directories,
    get_logger,
    load_dataframe,
    load_json,
    load_parquet,
    save_dataframe,
    save_json,
    save_parquet,
    validate_columns,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "raw_dir": BASE_CONFIG["raw_dir"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "fastf1_cache_dir": BASE_CONFIG["raw_dir"] / "fastf1_cache",
    "fastf1_features_path": BASE_CONFIG["processed_dir"] / "fastf1_features.csv",
    "fastf1_failures_path": BASE_CONFIG["processed_dir"] / "fastf1_failures.csv",
    "fastf1_failures_json_path": BASE_CONFIG["processed_dir"] / "fastf1_failures.json",
    "openf1_coverage_path": BASE_CONFIG["processed_dir"] / "openf1_weekend_coverage.csv",
    "openf1_round_cache_dir": BASE_CONFIG["raw_dir"] / "openf1",
    "ergast_mirror_url": "https://api.jolpi.ca/ergast/f1",
    "season_start": 2022,
    "season_end": detect_current_season(),
}

QUALIFYING_FEATURE_COLUMNS: list[str] = [
    "best_sector_1_ms",
    "best_sector_2_ms",
    "best_sector_3_ms",
    "sector_consistency",
    "max_speed_trap",
    "avg_throttle_pct",
    "brake_points",
]

RACE_FEATURE_COLUMNS: list[str] = [
    "avg_lap_time_ms",
    "lap_time_std",
    "avg_tyre_life",
    "tyre_degradation_rate",
    "pit_stop_count",
    "position_changes_lap1",
    "avg_gap_to_leader_ms",
]

PRACTICE_FEATURE_COLUMNS: list[str] = [
    "fp_best_lap_rank",
    "fp_long_run_pace",
    "fp_long_run_rank",
    "fp_pace_delta_to_teammate",
    "fp_sector_1_rank",
    "fp_sector_2_rank",
    "fp_sector_3_rank",
    "fp_sector_strength_score",
    "fp_compounds_run_count",
    "fp_ran_soft",
    "fp_ran_medium",
    "fp_ran_hard",
    "fp_soft_deg_rate",
    "fp_medium_deg_rate",
    "fp_hard_deg_rate",
    "fp_preferred_compound_code",
    "fp_lap_time_std",
    "fp_track_limit_violations",
    "fp_improvement_fp1_to_fp3",
    "fp_team_pace_delta_vs_last_race",
]

QUALIFYING_DEEP_FEATURE_COLUMNS: list[str] = [
    "q1_lap_time_ms",
    "q2_lap_time_ms",
    "q3_lap_time_ms",
    "quali_gap_to_pole_ms",
    "quali_gap_to_teammate_ms",
    "quali_improvement_rate",
    "sector_dominance_score",
    "tyre_used_in_q3_code",
    "track_position_in_quali",
]

SPRINT_FEATURE_COLUMNS: list[str] = [
    "is_sprint_weekend",
    "sprint_finish_position",
    "sprint_points_scored",
    "sprint_vs_expected",
    "sprint_compounds_used_count",
    "sprint_primary_compound_code",
    "sprint_tyre_life_used",
]

TRACK_EVOLUTION_FEATURE_COLUMNS: list[str] = [
    "track_evolution_delta",
    "fp1_to_race_temp_delta",
    "rubber_benefit_index",
]

SESSION_CONTEXT_COLUMNS: list[str] = [
    "fp1_air_temp_c",
    "fp1_track_temp_c",
    "race_air_temp_c",
    "race_track_temp_c",
    "weekend_session_count",
    "has_fp1",
    "has_fp2",
    "has_fp3",
    "has_quali",
    "has_race",
    "has_sprint",
    "has_sprint_shootout",
]

OPENF1_FALLBACK_FEATURE_COLUMNS: list[str] = [
    "openf1_best_sector_1_ms",
    "openf1_best_sector_2_ms",
    "openf1_best_sector_3_ms",
    "openf1_sector_consistency",
    "openf1_max_speed_trap",
]

WEEKEND_FEATURE_COLUMNS: list[str] = [
    *QUALIFYING_FEATURE_COLUMNS,
    *RACE_FEATURE_COLUMNS,
    *OPENF1_FALLBACK_FEATURE_COLUMNS,
    *PRACTICE_FEATURE_COLUMNS,
    *QUALIFYING_DEEP_FEATURE_COLUMNS,
    *SPRINT_FEATURE_COLUMNS,
    *TRACK_EVOLUTION_FEATURE_COLUMNS,
    *SESSION_CONTEXT_COLUMNS,
]

COMPOUND_CODE: dict[str, int] = {
    "unknown": 0,
    "hard": 1,
    "medium": 2,
    "soft": 3,
    "intermediate": 4,
    "wet": 5,
    "supersoft": 6,
    "ultrasoft": 7,
    "hypersoft": 8,
}

LOGGER = get_logger(__name__)
WEEKEND_SESSION_CACHE: dict[tuple[int, int, bool], tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}


def _unique_columns(columns: list[str]) -> list[str]:
    """Preserve order while removing duplicate column names."""

    return list(dict.fromkeys(columns))


def _weekend_feature_columns() -> list[str]:
    """Return the full persisted weekend feature schema."""

    return _unique_columns(["season", "round", "driver_number", *WEEKEND_FEATURE_COLUMNS])


def _empty_weekend_feature_frame() -> pd.DataFrame:
    """Return an empty weekend feature frame with the persisted schema."""

    return pd.DataFrame(columns=_weekend_feature_columns())


def _ensure_weekend_feature_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure a weekend feature frame contains every persisted weekend column."""

    ensured = frame.copy()
    for column in _weekend_feature_columns():
        if column not in ensured.columns:
            ensured[column] = np.nan
    return ensured[_weekend_feature_columns()]


def _session_slug(session_name: str) -> str:
    """Create a filesystem-safe session slug."""

    return re.sub(r"[^A-Za-z0-9]+", "_", str(session_name).strip()).strip("_").lower() or "session"


def _openf1_round_dir(season: int, round_number: int) -> Path:
    """Return the cache directory for one OpenF1 race weekend."""

    return CONFIG["openf1_round_cache_dir"] / str(int(season)) / f"R{int(round_number):02d}"


def _openf1_session_path(season: int, round_number: int, session_name: str) -> Path:
    """Return the parquet cache path for one normalized session."""

    return _openf1_round_dir(season, round_number) / f"{_session_slug(session_name)}.parquet"


def _openf1_session_index_path(season: int, round_number: int) -> Path:
    """Return the index cache for a weekend's session manifest."""

    return _openf1_round_dir(season, round_number) / "sessions.csv"


def _normalize_openf1_session_group(session_name: str, session_type: str | None = None) -> str:
    """Normalize OpenF1 session labels into a stable session-group vocabulary."""

    label = " ".join(str(value) for value in [session_name, session_type] if value).lower()
    if "practice 1" in label or "free practice 1" in label or "fp1" in label:
        return "FP1"
    if "practice 2" in label or "free practice 2" in label or "fp2" in label:
        return "FP2"
    if "practice 3" in label or "free practice 3" in label or "fp3" in label:
        return "FP3"
    if "sprint qualifying" in label or "sprint shootout" in label or "shootout" in label:
        return "SPRINT_SHOOTOUT"
    if "sprint" in label:
        return "SPRINT"
    if "qualifying" in label:
        return "QUALI"
    if "race" in label:
        return "RACE"
    return str(session_name or session_type or "UNKNOWN").upper().replace(" ", "_")


def _coerce_openf1_duration_ms(series: pd.Series) -> pd.Series:
    """Normalize OpenF1 duration-like columns to milliseconds."""

    def convert(value: Any) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("nan")
        text = str(value).strip()
        if text == "":
            return float("nan")
        try:
            if ":" in text:
                minutes_text, seconds_text = text.split(":", maxsplit=1)
                return (int(minutes_text) * 60.0 + float(seconds_text)) * 1000.0
            numeric_value = float(text)
            return numeric_value * 1000.0 if abs(numeric_value) < 1000.0 else numeric_value
        except (TypeError, ValueError):
            return float("nan")

    return series.map(convert).astype(float)


def _normalize_openf1_compound(value: Any) -> str:
    """Normalize tyre-compound labels."""

    if pd.isna(value) or value in (None, ""):
        return "unknown"
    text = str(value).strip().lower()
    if text in COMPOUND_CODE:
        return text
    if "soft" in text and "super" not in text and "ultra" not in text and "hyper" not in text:
        return "soft"
    if "medium" in text:
        return "medium"
    if "hard" in text:
        return "hard"
    if "inter" in text:
        return "intermediate"
    if "wet" in text:
        return "wet"
    return text


def _compound_to_code(value: Any) -> int:
    """Encode a tyre compound into a stable numeric code."""

    return int(COMPOUND_CODE.get(_normalize_openf1_compound(value), 0))


def _parse_segment_cell(value: Any) -> list[float]:
    """Parse an OpenF1 mini-sector payload into a numeric list."""

    if isinstance(value, list):
        return [float(item) for item in value if pd.notna(item)]
    if isinstance(value, tuple):
        return [float(item) for item in value if pd.notna(item)]
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist() if pd.notna(item)]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str) and value.strip() == "":
        return []
    text = str(value).strip()
    if text == "":
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [float(item) for item in parsed if pd.notna(item)]
    except Exception as exc:
        LOGGER.debug("Non-critical error parsing segment values: %s", exc)
        return []
    return []


def _track_limit_violation_proxy(frame: pd.DataFrame) -> float:
    """Return a conservative proxy for track-limit excursions from sparse OpenF1 data."""

    if frame.empty:
        return float("nan")
    invalid_sectors = (
        frame[["duration_sector_1_ms", "duration_sector_2_ms", "duration_sector_3_ms"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    segment_anomalies = 0
    for column in ["segments_sector_1", "segments_sector_2", "segments_sector_3"]:
        if column not in frame.columns:
            continue
        segment_anomalies += int(
            frame[column]
            .map(_parse_segment_cell)
            .map(lambda values: int(any(value <= 0 for value in values)) if values else 0)
            .sum()
        )
    return float(invalid_sectors + segment_anomalies)


def _safe_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Return dense ranks while preserving NaNs."""

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="dense", ascending=ascending)


def _stint_relative_laps(frame: pd.DataFrame) -> pd.Series:
    """Return the lap index within a stint."""

    if frame.empty or "stint_number" not in frame.columns or "lap_number" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame.groupby(["session_group", "driver_number", "stint_number"], dropna=False)["lap_number"].rank(method="first")


def _fit_slope(x: pd.Series, y: pd.Series) -> float:
    """Fit a simple linear slope if enough finite observations exist."""

    x_numeric = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_numeric = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_numeric) & np.isfinite(y_numeric)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.polyfit(x_numeric[mask], y_numeric[mask], 1)[0])


def _assign_stints_to_laps(laps_df: pd.DataFrame, stints_df: pd.DataFrame) -> pd.DataFrame:
    """Attach stint and compound information to lap records."""

    if laps_df.empty:
        return laps_df.copy()
    enriched = laps_df.copy()
    for column in ["stint_number", "compound", "tyre_age_at_start", "lap_start", "lap_end"]:
        if column not in enriched.columns:
            enriched[column] = np.nan if column != "compound" else "unknown"
    if stints_df.empty:
        enriched["compound"] = enriched["compound"].fillna("unknown")
        return enriched

    stints = stints_df.copy()
    stints["compound"] = stints["compound"].map(_normalize_openf1_compound)
    for row in stints.itertuples(index=False):
        mask = (
            enriched["driver_number"].eq(str(row.driver_number))
            & pd.to_numeric(enriched["lap_number"], errors="coerce").between(
                float(row.lap_start) if pd.notna(row.lap_start) else -np.inf,
                float(row.lap_end) if pd.notna(row.lap_end) else np.inf,
                inclusive="both",
            )
        )
        if not mask.any():
            continue
        enriched.loc[mask, "stint_number"] = row.stint_number
        enriched.loc[mask, "compound"] = row.compound
        enriched.loc[mask, "tyre_age_at_start"] = row.tyre_age_at_start
        enriched.loc[mask, "lap_start"] = row.lap_start
        enriched.loc[mask, "lap_end"] = row.lap_end

    enriched["compound"] = enriched["compound"].fillna("unknown").map(_normalize_openf1_compound)
    return enriched


def _merge_openf1_session_payloads(
    season: int,
    round_number: int,
    session_row: pd.Series,
    laps_df: pd.DataFrame,
    stints_df: pd.DataFrame,
    position_df: pd.DataFrame,
    car_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten OpenF1 session payloads into a driver/lap-level parquet dataset."""

    laps_input = laps_df.copy()
    stints_input = stints_df.copy()
    position_input = position_df.copy()
    car_input = car_df.copy()
    for frame in [laps_input, stints_input, position_input, car_input]:
        if "driver_number" in frame.columns:
            frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number)

    laps = _assign_stints_to_laps(laps_input, stints_input)
    if laps.empty:
        driver_numbers = sorted(
            set(position_input.get("driver_number", pd.Series(dtype=str)).astype(str))
            | set(stints_input.get("driver_number", pd.Series(dtype=str)).astype(str))
            | set(car_input.get("driver_number", pd.Series(dtype=str)).astype(str))
        )
        laps = pd.DataFrame({"driver_number": [value for value in driver_numbers if value not in {"", "nan"}]})
        laps["lap_number"] = np.nan
        laps["lap_duration_ms"] = np.nan
        laps["duration_sector_1_ms"] = np.nan
        laps["duration_sector_2_ms"] = np.nan
        laps["duration_sector_3_ms"] = np.nan
        laps["i1_speed"] = np.nan
        laps["i2_speed"] = np.nan
        laps["st_speed"] = np.nan
        laps["is_pit_out_lap"] = 0
        for column in ["segments_sector_1", "segments_sector_2", "segments_sector_3"]:
            laps[column] = ""

    position_summary = pd.DataFrame(columns=["driver_number", "session_opening_position", "session_final_position"])
    if not position_input.empty:
        ordered = position_input.sort_values("date")
        position_summary = ordered.groupby("driver_number", as_index=False).agg(
            session_opening_position=("position", "first"),
            session_final_position=("position", "last"),
        )
        position_summary["driver_number"] = position_summary["driver_number"].apply(_normalize_driver_number)

    car_summary = pd.DataFrame(columns=["driver_number", "telemetry_speedtrap_max", "telemetry_speedtrap_hits"])
    if not car_input.empty:
        car_summary = car_input.groupby("driver_number", as_index=False).agg(
            telemetry_speedtrap_max=("speed", "max"),
            telemetry_speedtrap_hits=("speed", "size"),
        )
        car_summary["driver_number"] = car_summary["driver_number"].apply(_normalize_driver_number)

    weather_summary = {
        "session_air_temperature_c": float(pd.to_numeric(weather_df.get("air_temperature"), errors="coerce").median())
        if not weather_df.empty and "air_temperature" in weather_df.columns
        else np.nan,
        "session_track_temperature_c": float(pd.to_numeric(weather_df.get("track_temperature"), errors="coerce").median())
        if not weather_df.empty and "track_temperature" in weather_df.columns
        else np.nan,
        "session_humidity_pct": float(pd.to_numeric(weather_df.get("humidity"), errors="coerce").median())
        if not weather_df.empty and "humidity" in weather_df.columns
        else np.nan,
        "session_rainfall_mm": float(pd.to_numeric(weather_df.get("rainfall"), errors="coerce").median())
        if not weather_df.empty and "rainfall" in weather_df.columns
        else np.nan,
        "session_wind_speed_kmh": float(pd.to_numeric(weather_df.get("wind_speed"), errors="coerce").median())
        if not weather_df.empty and "wind_speed" in weather_df.columns
        else np.nan,
        "session_wind_direction_deg": float(pd.to_numeric(weather_df.get("wind_direction"), errors="coerce").median())
        if not weather_df.empty and "wind_direction" in weather_df.columns
        else np.nan,
    }

    for frame in [laps, position_summary, car_summary]:
        if "driver_number" in frame.columns:
            frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number).astype("string")

    merged = laps.merge(position_summary, how="left", on="driver_number").merge(car_summary, how="left", on="driver_number")
    merged["season"] = int(season)
    merged["round"] = int(round_number)
    merged["session_key"] = int(session_row["session_key"])
    merged["session_name"] = str(session_row["session_name"])
    merged["session_type"] = str(session_row.get("session_type", session_row.get("session_name", "")))
    merged["session_group"] = str(session_row["session_group"])
    merged["date_start"] = pd.to_datetime(session_row.get("date_start"), errors="coerce", utc=True)
    merged["date_end"] = pd.to_datetime(session_row.get("date_end"), errors="coerce", utc=True)
    for column, value in weather_summary.items():
        merged[column] = value
    return merged


def _load_or_fetch_session_parquet(
    client: OpenF1Client,
    season: int,
    round_number: int,
    session_row: pd.Series,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load or fetch one OpenF1 session cache parquet and its coverage metadata."""

    def safe_fetch_frame(endpoint: str, params: dict[str, Any]) -> pd.DataFrame:
        try:
            return pd.DataFrame(client.fetch(endpoint, params=params))
        except Exception as exc:
            LOGGER.warning(
                "OpenF1 partial fetch failed for %s R%s %s %s: %s",
                season,
                round_number,
                session_row.get("session_name"),
                endpoint,
                exc,
            )
            return pd.DataFrame()

    parquet_path = _openf1_session_path(season, round_number, str(session_row["session_name"]))
    if parquet_path.exists():
        cached = load_parquet(parquet_path)
        if not refresh or not cached.empty:
            return cached, {
                "season": int(season),
                "round": int(round_number),
                "session_key": int(session_row["session_key"]),
                "session_name": str(session_row["session_name"]),
                "session_group": str(session_row["session_group"]),
                "parquet_path": str(parquet_path),
                "driver_rows": int(len(cached)),
                "has_laps": int("lap_duration_ms" in cached.columns and cached["lap_duration_ms"].notna().any()),
                "has_stints": int("stint_number" in cached.columns and cached["stint_number"].notna().any()),
                "has_position": int("session_final_position" in cached.columns and cached["session_final_position"].notna().any()),
                "has_car_data": int("telemetry_speedtrap_max" in cached.columns and cached["telemetry_speedtrap_max"].notna().any()),
                "has_weather": int("session_air_temperature_c" in cached.columns and cached["session_air_temperature_c"].notna().any()),
            }
    if not parquet_path.exists() and not refresh:
        return pd.DataFrame(), {
            "season": int(season),
            "round": int(round_number),
            "session_key": int(session_row["session_key"]),
            "session_name": str(session_row["session_name"]),
            "session_group": str(session_row["session_group"]),
            "parquet_path": str(parquet_path),
            "driver_rows": 0,
            "has_laps": 0,
            "has_stints": 0,
            "has_position": 0,
            "has_car_data": 0,
            "has_weather": 0,
        }

    session_key = int(session_row["session_key"])
    laps_payload = safe_fetch_frame("/laps", {"session_key": session_key})
    stints_payload = safe_fetch_frame("/stints", {"session_key": session_key})
    position_payload = safe_fetch_frame("/position", {"session_key": session_key})
    car_payload = safe_fetch_frame("/car_data", {"session_key": session_key, "speed": ">=300"})
    weather_payload = safe_fetch_frame("/weather", {"session_key": session_key})

    if not laps_payload.empty:
        if "driver_number" not in laps_payload.columns and "driver_num" in laps_payload.columns:
            laps_payload["driver_number"] = laps_payload["driver_num"]
        laps_payload["driver_number"] = laps_payload["driver_number"].apply(_normalize_driver_number)
        laps_payload["lap_number"] = pd.to_numeric(laps_payload.get("lap_number", laps_payload.get("lap")), errors="coerce")
        laps_payload["lap_duration_ms"] = _coerce_openf1_duration_ms(
            laps_payload.get("lap_duration", laps_payload.get("lap_time", pd.Series(dtype=float)))
        )
        laps_payload["duration_sector_1_ms"] = _coerce_openf1_duration_ms(laps_payload.get("duration_sector_1", pd.Series(dtype=float)))
        laps_payload["duration_sector_2_ms"] = _coerce_openf1_duration_ms(laps_payload.get("duration_sector_2", pd.Series(dtype=float)))
        laps_payload["duration_sector_3_ms"] = _coerce_openf1_duration_ms(laps_payload.get("duration_sector_3", pd.Series(dtype=float)))
        for column in ["i1_speed", "i2_speed", "st_speed"]:
            laps_payload[column] = pd.to_numeric(laps_payload.get(column), errors="coerce")
        laps_payload["is_pit_out_lap"] = pd.to_numeric(
            laps_payload.get("is_pit_out_lap", pd.Series(0, index=laps_payload.index)),
            errors="coerce",
        ).fillna(0).astype(int)
        for column in ["segments_sector_1", "segments_sector_2", "segments_sector_3"]:
            if column not in laps_payload.columns:
                laps_payload[column] = ""
    else:
        laps_payload = pd.DataFrame(
            columns=[
                "driver_number",
                "lap_number",
                "lap_duration_ms",
                "duration_sector_1_ms",
                "duration_sector_2_ms",
                "duration_sector_3_ms",
                "i1_speed",
                "i2_speed",
                "st_speed",
                "is_pit_out_lap",
                "segments_sector_1",
                "segments_sector_2",
                "segments_sector_3",
            ]
        )

    if not stints_payload.empty:
        if "driver_number" not in stints_payload.columns and "driver_num" in stints_payload.columns:
            stints_payload["driver_number"] = stints_payload["driver_num"]
        stints_payload["driver_number"] = stints_payload["driver_number"].apply(_normalize_driver_number)
        stints_payload["stint_number"] = pd.to_numeric(
            stints_payload.get("stint_number", stints_payload.get("stint")),
            errors="coerce",
        )
        stints_payload["lap_start"] = pd.to_numeric(stints_payload.get("lap_start"), errors="coerce")
        stints_payload["lap_end"] = pd.to_numeric(stints_payload.get("lap_end"), errors="coerce")
        stints_payload["tyre_age_at_start"] = pd.to_numeric(stints_payload.get("tyre_age_at_start"), errors="coerce")
        stints_payload["compound"] = stints_payload.get("compound", pd.Series("unknown", index=stints_payload.index)).map(
            _normalize_openf1_compound
        )
    else:
        stints_payload = pd.DataFrame(
            columns=["driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]
        )

    if not position_payload.empty:
        if "driver_number" not in position_payload.columns and "driver_num" in position_payload.columns:
            position_payload["driver_number"] = position_payload["driver_num"]
        position_payload["driver_number"] = position_payload["driver_number"].apply(_normalize_driver_number)
        position_payload["position"] = pd.to_numeric(position_payload.get("position"), errors="coerce")
        position_payload["date"] = pd.to_datetime(position_payload.get("date"), errors="coerce", utc=True)
    else:
        position_payload = pd.DataFrame(columns=["driver_number", "position", "date"])

    if not car_payload.empty:
        if "driver_number" not in car_payload.columns and "driver_num" in car_payload.columns:
            car_payload["driver_number"] = car_payload["driver_num"]
        car_payload["driver_number"] = car_payload["driver_number"].apply(_normalize_driver_number)
        car_payload["speed"] = pd.to_numeric(car_payload.get("speed"), errors="coerce")
    else:
        car_payload = pd.DataFrame(columns=["driver_number", "speed"])

    if not weather_payload.empty:
        for column in ["air_temperature", "track_temperature", "humidity", "rainfall", "wind_speed", "wind_direction"]:
            weather_payload[column] = pd.to_numeric(weather_payload.get(column), errors="coerce")
    else:
        weather_payload = pd.DataFrame(
            columns=["air_temperature", "track_temperature", "humidity", "rainfall", "wind_speed", "wind_direction"]
        )

    session_frame = _merge_openf1_session_payloads(
        season=season,
        round_number=round_number,
        session_row=session_row,
        laps_df=laps_payload,
        stints_df=stints_payload,
        position_df=position_payload,
        car_df=car_payload,
        weather_df=weather_payload,
    )
    save_parquet(session_frame, parquet_path)
    return session_frame, {
        "season": int(season),
        "round": int(round_number),
        "session_key": session_key,
        "session_name": str(session_row["session_name"]),
        "session_group": str(session_row["session_group"]),
        "parquet_path": str(parquet_path),
        "driver_rows": int(len(session_frame)),
        "has_laps": int(not laps_payload.empty),
        "has_stints": int(not stints_payload.empty),
        "has_position": int(not position_payload.empty),
        "has_car_data": int(not car_payload.empty),
        "has_weather": int(not weather_payload.empty),
    }


def _import_fastf1() -> Any:
    """Import FastF1 lazily so non-FastF1 runs can still work."""

    try:
        import fastf1  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FastF1 is not installed. Run `pip install -r requirements.txt` to enable telemetry ingestion."
        ) from exc
    return fastf1


def _configure_fastf1_sources(fastf1: Any) -> None:
    """Redirect FastF1's legacy Ergast dependency to the Jolpica mirror."""

    try:
        fastf1.ergast.interface.BASE_URL = CONFIG["ergast_mirror_url"]
    except AttributeError:
        LOGGER.debug("FastF1 ergast interface base URL not available for override")

    try:
        fastf1.ergast.legacy.base_url = CONFIG["ergast_mirror_url"]
    except AttributeError:
        LOGGER.debug("FastF1 ergast legacy base URL not available for override")


def _weekend_cache_paths(season: int, round_number: int) -> tuple[Path, Path]:
    """Return parquet and JSON cache paths for a race weekend."""

    return (
        CONFIG["raw_dir"] / f"fastf1_{season}_{round_number}.parquet",
        CONFIG["raw_dir"] / f"fastf1_{season}_{round_number}.json",
    )


def _fallback_round_sessions_from_cache(season: int, round_number: int) -> pd.DataFrame:
    """Resolve a weekend's sessions from cached season-level OpenF1 session listings."""

    candidate_paths = [
        CONFIG["processed_dir"] / "openf1_sessions.csv",
        CONFIG["raw_dir"] / "openf1_sessions_all.csv",
        CONFIG["raw_dir"] / f"openf1_sessions_{season}.csv",
    ]
    sessions = pd.DataFrame()
    for path in candidate_paths:
        if path.exists():
            sessions = load_dataframe(path, parse_dates=["date_start"])
            if "season" in sessions.columns:
                sessions = sessions.loc[sessions["season"] == int(season)].copy()
            if not sessions.empty:
                break
    if sessions.empty:
        return sessions

    calendar = load_calendar(season, allow_fetch=False)
    calendar_row = calendar.loc[calendar["round"] == int(round_number)].head(1)
    if calendar_row.empty:
        return pd.DataFrame(columns=sessions.columns)

    race_date = pd.to_datetime(calendar_row["date"].iloc[0], errors="coerce")
    sessions["date_start"] = pd.to_datetime(sessions["date_start"], errors="coerce", utc=True)
    sessions["meeting_id"] = (
        sessions.get("meeting_name", pd.Series("", index=sessions.index)).fillna("")
        + "|"
        + sessions.get("location", pd.Series("", index=sessions.index)).fillna("")
        + "|"
        + sessions.get("country_name", pd.Series("", index=sessions.index)).fillna("")
    ).str.strip("|")
    summary = (
        sessions.groupby("meeting_id", as_index=False)
        .agg(meeting_start=("date_start", "min"))
        .sort_values("meeting_start")
    )
    if pd.notna(race_date):
        target = pd.Timestamp(race_date).tz_localize("UTC") if pd.Timestamp(race_date).tzinfo is None else pd.Timestamp(race_date)
        summary["date_gap"] = (summary["meeting_start"] - target).abs()
        meeting_id = str(summary.sort_values("date_gap").iloc[0]["meeting_id"])
    else:
        if int(round_number) > len(summary):
            return pd.DataFrame(columns=sessions.columns)
        meeting_id = str(summary.iloc[int(round_number) - 1]["meeting_id"])
    return sessions.loc[sessions["meeting_id"] == meeting_id].copy()


def fetch_all_sessions(
    season: int,
    round_number: int,
    refresh: bool = False,
    client: OpenF1Client | None = None,
) -> pd.DataFrame:
    """Fetch and cache every available OpenF1 session for a race weekend."""

    ensure_directories([_openf1_round_dir(season, round_number)])
    index_path = _openf1_session_index_path(season, round_number)
    if int(season) < 2023:
        empty = pd.DataFrame(
            columns=[
                "season",
                "round",
                "session_key",
                "session_name",
                "session_type",
                "session_group",
                "date_start",
                "date_end",
                "parquet_path",
                "driver_rows",
                "has_laps",
                "has_stints",
                "has_position",
                "has_car_data",
                "has_weather",
            ]
        )
        save_dataframe(empty, index_path)
        return empty
    if index_path.exists():
        cached_index = load_dataframe(index_path)
        if not cached_index.empty or not refresh:
            return cached_index

    openf1_client = client or OpenF1Client()
    raw_sessions = pd.DataFrame()
    if not refresh:
        try:
            raw_sessions = _fallback_round_sessions_from_cache(season, round_number)
        except Exception as exc:
            LOGGER.warning("Cached OpenF1 session fallback failed for %s R%s: %s", season, round_number, exc)
            raw_sessions = pd.DataFrame()

    if raw_sessions.empty:
        try:
            raw_sessions = pd.DataFrame(
                openf1_client.fetch("/sessions", params={"year": int(season), "round_number": int(round_number)})
            )
        except Exception as exc:
            LOGGER.warning("Direct OpenF1 round session lookup failed for %s R%s: %s", season, round_number, exc)

    if raw_sessions.empty and refresh:
        try:
            raw_sessions = _fallback_round_sessions_from_cache(season, round_number)
        except Exception as exc:
            LOGGER.warning("Cached OpenF1 session fallback failed for %s R%s: %s", season, round_number, exc)
            raw_sessions = pd.DataFrame()

    if raw_sessions.empty:
        empty = pd.DataFrame(
            columns=[
                "season",
                "round",
                "session_key",
                "session_name",
                "session_type",
                "session_group",
                "date_start",
                "date_end",
                "parquet_path",
                "driver_rows",
                "has_laps",
                "has_stints",
                "has_position",
                "has_car_data",
                "has_weather",
            ]
        )
        save_dataframe(empty, index_path)
        return empty

    sessions = raw_sessions.copy()
    if "session_key" not in sessions.columns and "key" in sessions.columns:
        sessions["session_key"] = sessions["key"]
    if "session_name" not in sessions.columns and "session_type" in sessions.columns:
        sessions["session_name"] = sessions["session_type"]
    if "session_type" not in sessions.columns:
        sessions["session_type"] = sessions["session_name"]
    sessions["session_group"] = sessions.apply(
        lambda row: _normalize_openf1_session_group(row.get("session_name"), row.get("session_type")),
        axis=1,
    )
    sessions["date_start"] = pd.to_datetime(sessions.get("date_start"), errors="coerce", utc=True)
    sessions["date_end"] = pd.to_datetime(sessions.get("date_end"), errors="coerce", utc=True)
    sessions = sessions.dropna(subset=["session_key"]).drop_duplicates(subset=["session_key"]).sort_values("date_start")

    coverage_rows: list[dict[str, Any]] = []
    for row in sessions.itertuples(index=False):
        session_series = pd.Series(row._asdict())
        try:
            _, coverage = _load_or_fetch_session_parquet(
                client=openf1_client,
                season=int(season),
                round_number=int(round_number),
                session_row=session_series,
                refresh=refresh,
            )
            coverage_rows.append(coverage)
        except Exception as exc:
            LOGGER.warning(
                "OpenF1 full-session fetch failed for %s R%s %s: %s",
                season,
                round_number,
                session_series.get("session_name"),
                exc,
            )
            coverage_rows.append(
                {
                    "season": int(season),
                    "round": int(round_number),
                    "session_key": int(session_series["session_key"]),
                    "session_name": str(session_series["session_name"]),
                    "session_group": str(session_series["session_group"]),
                    "parquet_path": str(_openf1_session_path(season, round_number, str(session_series["session_name"]))),
                    "driver_rows": 0,
                    "has_laps": 0,
                    "has_stints": 0,
                    "has_position": 0,
                    "has_car_data": 0,
                    "has_weather": 0,
                }
            )

    coverage_frame = pd.DataFrame(coverage_rows)
    save_dataframe(coverage_frame, index_path)
    return coverage_frame


def _load_weekend_session_frames(
    season: int,
    round_number: int,
    refresh: bool = False,
    client: OpenF1Client | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load a weekend's OpenF1 session manifest and parquet payloads."""

    cache_key = (int(season), int(round_number), bool(refresh))
    cached = WEEKEND_SESSION_CACHE.get(cache_key)
    if cached is not None:
        coverage_frame, session_frames = cached
        return coverage_frame.copy(), {key: frame.copy() for key, frame in session_frames.items()}

    coverage = fetch_all_sessions(season=season, round_number=round_number, refresh=refresh, client=client)
    session_frames: dict[str, pd.DataFrame] = {}
    for row in coverage.itertuples(index=False):
        parquet_path = Path(row.parquet_path)
        if not parquet_path.exists():
            continue
        session_frames[str(row.session_group)] = load_parquet(parquet_path)
    WEEKEND_SESSION_CACHE[cache_key] = (coverage.copy(), {key: frame.copy() for key, frame in session_frames.items()})
    return coverage, session_frames


def _normalize_driver_number(value: Any) -> str | None:
    """Normalize driver numbers across FastF1 and processed race data."""

    if pd.isna(value) or value in (None, ""):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _timedelta_to_ms(value: Any) -> float:
    """Convert timedelta-like objects to milliseconds."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    try:
        return float(pd.to_timedelta(value).total_seconds() * 1000.0)
    except (TypeError, ValueError):
        return float("nan")


def _series_to_ms(series: pd.Series) -> pd.Series:
    """Convert a timedelta/string series to milliseconds."""

    return pd.to_timedelta(series, errors="coerce").dt.total_seconds() * 1000.0


def _session_driver_number_map(session: Any) -> dict[str, str]:
    """Build a driver-code to driver-number lookup from session metadata."""

    results = getattr(session, "results", None)
    if results is None or getattr(results, "empty", True):
        return {}

    lookup: dict[str, str] = {}
    frame = pd.DataFrame(results).copy()
    if "DriverNumber" not in frame.columns:
        return {}

    frame["DriverNumber"] = frame["DriverNumber"].apply(_normalize_driver_number)
    for _, row in frame.iterrows():
        driver_number = row.get("DriverNumber")
        if driver_number is None or pd.isna(driver_number):
            continue
        for key in ("Abbreviation", "BroadcastName", "FullName", "LastName", "TeamName"):
            value = row.get(key)
            if pd.notna(value):
                lookup[str(value).strip()] = str(driver_number)
    return lookup


def normalize_driver_numbers(laps_df: pd.DataFrame, drivers_df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize FastF1 driver-number naming and typing across seasons."""

    normalized = laps_df.copy()
    if "DriverNumber" not in normalized.columns:
        if "Driver" in normalized.columns:
            normalized = normalized.rename(columns={"Driver": "DriverNumber"})
        else:
            raise ValueError(
                "No driver identifier column found in FastF1 laps. "
                f"Available columns: {list(normalized.columns)}"
            )

    normalized["DriverNumber"] = normalized["DriverNumber"].apply(_normalize_driver_number)
    if drivers_df is not None and not drivers_df.empty and "driver_number" in drivers_df.columns:
        valid_numbers = set(drivers_df["driver_number"].dropna().astype(str).str.strip())
        invalid = sorted(
            {
                value
                for value in normalized["DriverNumber"].dropna().astype(str).str.strip().unique().tolist()
                if value not in valid_numbers
            }
        )
        if invalid:
            LOGGER.warning("Unknown driver numbers in FastF1 laps: %s", invalid)
    return normalized


def load_session_safe(year: int, round_number: int, session_type: str) -> Any:
    """Load a FastF1 session with year-specific workarounds."""

    session = fastf1.get_session(year, round_number, session_type)
    if year == 2022:
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    else:
        session.load(
            laps=True,
            telemetry=bool(session_type == "Q"),
            weather=True,
            messages=False,
        )
    return session


def _normalized_session_laps(session: Any) -> pd.DataFrame:
    """Return session laps with a normalized driver-number column."""

    laps = normalize_driver_numbers(getattr(session, "laps", pd.DataFrame()), pd.DataFrame(getattr(session, "results", None)))
    if laps.empty:
        return pd.DataFrame()

    driver_lookup = _session_driver_number_map(session)
    if "Driver" in laps.columns:
        mapped_driver_numbers = laps["Driver"].map(lambda value: driver_lookup.get(str(value).strip()))
        laps["DriverNumber"] = laps["DriverNumber"].where(laps["DriverNumber"].notna(), mapped_driver_numbers)

    laps["DriverNumber"] = laps["DriverNumber"].apply(_normalize_driver_number)
    return laps.dropna(subset=["DriverNumber"]).copy()


def _rising_edge_count(values: pd.Series) -> int:
    """Count rising-edge events in a boolean-ish telemetry channel."""

    if values.empty:
        return 0
    signal = pd.to_numeric(values, errors="coerce").fillna(0.0) > 0.0
    previous = signal.shift(1, fill_value=False)
    return int((signal & ~previous).sum())


def _green_flag_mask(status_series: pd.Series) -> pd.Series:
    """Return a mask for laps run under normal green-flag conditions."""

    normalized = status_series.fillna("1").astype(str).str.strip()
    return normalized.eq("1")


def _clean_race_laps(driver_laps: pd.DataFrame) -> pd.DataFrame:
    """Filter out pit laps, safety-car laps, and invalid race laps."""

    if driver_laps.empty:
        return driver_laps.copy()

    frame = driver_laps.copy()
    if "LapTime" in frame.columns:
        frame["lap_time_ms"] = _series_to_ms(frame["LapTime"])
    else:
        frame["lap_time_ms"] = np.nan

    mask = frame["lap_time_ms"].notna()
    if "PitInTime" in frame.columns:
        mask &= frame["PitInTime"].isna()
    if "PitOutTime" in frame.columns:
        mask &= frame["PitOutTime"].isna()
    if "TrackStatus" in frame.columns:
        mask &= _green_flag_mask(frame["TrackStatus"])
    if "IsAccurate" in frame.columns:
        mask &= frame["IsAccurate"].fillna(True).astype(bool)

    cleaned = frame.loc[mask].copy()
    if cleaned.empty:
        cleaned = frame.loc[frame["lap_time_ms"].notna()].copy()
    return cleaned


def _compound_degradation_slopes(driver_laps: pd.DataFrame) -> list[float]:
    """Fit within-stint lap time slopes as a tyre-degradation proxy."""

    if driver_laps.empty or "lap_time_ms" not in driver_laps.columns:
        return []

    slopes: list[float] = []
    group_columns = ["Stint"] if "Stint" in driver_laps.columns else ["Compound"]
    for _, stint_df in driver_laps.groupby(group_columns, dropna=False):
        stint_df = stint_df.copy()
        if stint_df.empty or len(stint_df) < 3:
            continue
        if "TyreLife" in stint_df.columns and stint_df["TyreLife"].notna().sum() >= 3:
            x = pd.to_numeric(stint_df["TyreLife"], errors="coerce").to_numpy(dtype=float)
        else:
            x = np.arange(len(stint_df), dtype=float)
        y = pd.to_numeric(stint_df["lap_time_ms"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if int(valid_mask.sum()) < 3:
            continue
        slope = np.polyfit(x[valid_mask], y[valid_mask], 1)[0]
        slopes.append(float(slope))
    return slopes


def _qualifying_driver_frame(session: Any) -> pd.DataFrame:
    """Return a normalized qualifying-laps frame with driver numbers."""

    return _normalized_session_laps(session)


def _race_driver_frame(session: Any) -> pd.DataFrame:
    """Return a normalized race-laps frame with helper gap columns."""

    laps = _normalized_session_laps(session)
    if laps.empty:
        return pd.DataFrame()
    if "Time" in laps.columns:
        laps["time_ms"] = _series_to_ms(laps["Time"])
        leader_time = laps.groupby("LapNumber")["time_ms"].transform("min")
        laps["gap_to_leader_ms"] = laps["time_ms"] - leader_time
    else:
        laps["gap_to_leader_ms"] = np.nan
    return laps


def _extract_qualifying_features(session: Any, weekend_results: pd.DataFrame) -> pd.DataFrame:
    """Extract driver-level qualifying telemetry features for one session."""

    laps = _qualifying_driver_frame(session)
    if laps.empty:
        return pd.DataFrame(columns=["season", "round", "driver_number", *QUALIFYING_FEATURE_COLUMNS])

    rows: list[dict[str, Any]] = []
    driver_numbers = sorted(set(weekend_results["driver_number"].dropna().apply(_normalize_driver_number)))
    for driver_number in driver_numbers:
        driver_laps = laps.loc[laps["DriverNumber"] == driver_number].copy()
        if driver_laps.empty:
            continue

        sector_1_ms = _series_to_ms(driver_laps["Sector1Time"]) if "Sector1Time" in driver_laps.columns else pd.Series(dtype=float)
        sector_2_ms = _series_to_ms(driver_laps["Sector2Time"]) if "Sector2Time" in driver_laps.columns else pd.Series(dtype=float)
        sector_3_ms = _series_to_ms(driver_laps["Sector3Time"]) if "Sector3Time" in driver_laps.columns else pd.Series(dtype=float)
        sector_stds = [
            value
            for value in [
                float(sector_1_ms.std(ddof=0)) if not sector_1_ms.empty else np.nan,
                float(sector_2_ms.std(ddof=0)) if not sector_2_ms.empty else np.nan,
                float(sector_3_ms.std(ddof=0)) if not sector_3_ms.empty else np.nan,
            ]
            if pd.notna(value)
        ]
        fastest_lap = driver_laps.pick_fastest()
        avg_throttle_pct = np.nan
        brake_points = np.nan
        if fastest_lap is not None and getattr(fastest_lap, "empty", False) is False:
            try:
                car_data = fastest_lap.get_car_data()
                avg_throttle_pct = float(pd.to_numeric(car_data.get("Throttle"), errors="coerce").mean())
                brake_points = float(_rising_edge_count(car_data.get("Brake", pd.Series(dtype=float))))
            except Exception as exc:
                LOGGER.warning(
                    "Failed telemetry extraction for qualifying season=%s round=%s driver=%s: %s",
                    int(weekend_results["season"].iloc[0]),
                    int(weekend_results["round"].iloc[0]),
                    driver_number,
                    exc,
                )

        rows.append(
            {
                "season": int(weekend_results["season"].iloc[0]),
                "round": int(weekend_results["round"].iloc[0]),
                "driver_number": driver_number,
                "best_sector_1_ms": float(sector_1_ms.min()) if not sector_1_ms.empty else np.nan,
                "best_sector_2_ms": float(sector_2_ms.min()) if not sector_2_ms.empty else np.nan,
                "best_sector_3_ms": float(sector_3_ms.min()) if not sector_3_ms.empty else np.nan,
                "sector_consistency": float(np.mean(sector_stds)) if sector_stds else np.nan,
                "max_speed_trap": float(pd.to_numeric(driver_laps.get("SpeedST"), errors="coerce").max()),
                "avg_throttle_pct": avg_throttle_pct,
                "brake_points": brake_points,
            }
        )

    return pd.DataFrame(rows)


def _extract_race_features(session: Any, weekend_results: pd.DataFrame) -> pd.DataFrame:
    """Extract driver-level race pace and stint features for one session."""

    laps = _race_driver_frame(session)
    if laps.empty:
        return pd.DataFrame(columns=["season", "round", "driver_number", *RACE_FEATURE_COLUMNS])

    grid_lookup = {
        _normalize_driver_number(row.driver_number): float(row.grid_position)
        for row in weekend_results[["driver_number", "grid_position"]].itertuples(index=False)
        if _normalize_driver_number(row.driver_number) is not None
    }
    rows: list[dict[str, Any]] = []
    driver_numbers = sorted(set(weekend_results["driver_number"].dropna().apply(_normalize_driver_number)))
    for driver_number in driver_numbers:
        driver_laps = laps.loc[laps["DriverNumber"] == driver_number].copy()
        if driver_laps.empty:
            continue
        clean_laps = _clean_race_laps(driver_laps)
        clean_laps["lap_time_ms"] = pd.to_numeric(clean_laps["lap_time_ms"], errors="coerce")

        avg_tyre_life = np.nan
        if "Stint" in driver_laps.columns:
            stint_lengths = driver_laps.groupby("Stint", dropna=False)["LapNumber"].count()
            if not stint_lengths.empty:
                avg_tyre_life = float(stint_lengths.mean())

        pit_stop_count = float(driver_laps["PitInTime"].notna().sum()) if "PitInTime" in driver_laps.columns else np.nan
        if pd.isna(pit_stop_count) and "Stint" in driver_laps.columns:
            pit_stop_count = float(max(pd.to_numeric(driver_laps["Stint"], errors="coerce").max() - 1.0, 0.0))

        lap1_position = np.nan
        if "LapNumber" in driver_laps.columns and "Position" in driver_laps.columns:
            lap1_row = driver_laps.loc[pd.to_numeric(driver_laps["LapNumber"], errors="coerce") == 1].head(1)
            if not lap1_row.empty:
                lap1_position = float(pd.to_numeric(lap1_row["Position"], errors="coerce").iloc[0])

        slopes = _compound_degradation_slopes(clean_laps)
        rows.append(
            {
                "season": int(weekend_results["season"].iloc[0]),
                "round": int(weekend_results["round"].iloc[0]),
                "driver_number": driver_number,
                "avg_lap_time_ms": float(clean_laps["lap_time_ms"].mean()) if not clean_laps.empty else np.nan,
                "lap_time_std": float(clean_laps["lap_time_ms"].std(ddof=0)) if len(clean_laps) >= 2 else np.nan,
                "avg_tyre_life": avg_tyre_life,
                "tyre_degradation_rate": float(np.mean(slopes)) if slopes else np.nan,
                "pit_stop_count": pit_stop_count,
                "position_changes_lap1": float(grid_lookup.get(driver_number, np.nan) - lap1_position)
                if pd.notna(lap1_position)
                else np.nan,
                "avg_gap_to_leader_ms": float(pd.to_numeric(clean_laps["gap_to_leader_ms"], errors="coerce").mean())
                if not clean_laps.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _session_clean_laps(session_df: pd.DataFrame) -> pd.DataFrame:
    """Return non-pit, finite-duration laps from a normalized OpenF1 session frame."""

    if session_df.empty:
        return session_df.copy()
    frame = session_df.copy()
    frame["lap_duration_ms"] = pd.to_numeric(frame.get("lap_duration_ms"), errors="coerce")
    frame["lap_number"] = pd.to_numeric(frame.get("lap_number"), errors="coerce")
    if "stint_number" in frame.columns:
        frame["stint_number"] = pd.to_numeric(frame["stint_number"], errors="coerce")
    clean = frame.loc[
        frame["lap_duration_ms"].between(40_000.0, 200_000.0, inclusive="both")
        & pd.to_numeric(frame.get("is_pit_out_lap", 0), errors="coerce").fillna(0).eq(0)
    ].copy()
    return clean


def _field_best_lap_table(session_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Summarize best-lap pace and sectors for one session."""

    clean = _session_clean_laps(session_df)
    if clean.empty:
        return pd.DataFrame(
            columns=[
                "driver_number",
                f"{prefix}_best_lap_ms",
                f"{prefix}_best_lap_rank",
                f"{prefix}_sector_1_rank",
                f"{prefix}_sector_2_rank",
                f"{prefix}_sector_3_rank",
                f"{prefix}_lap_time_std",
                f"{prefix}_track_limit_violations",
                f"{prefix}_max_speed_trap",
            ]
        )

    best = clean.groupby("driver_number", as_index=False).agg(
        **{
            f"{prefix}_best_lap_ms": ("lap_duration_ms", "min"),
            f"{prefix}_best_sector_1_ms": ("duration_sector_1_ms", "min"),
            f"{prefix}_best_sector_2_ms": ("duration_sector_2_ms", "min"),
            f"{prefix}_best_sector_3_ms": ("duration_sector_3_ms", "min"),
            f"{prefix}_lap_time_std": ("lap_duration_ms", lambda series: float(pd.to_numeric(series, errors="coerce").std(ddof=0))),
            f"{prefix}_max_speed_trap": ("st_speed", "max"),
        }
    )
    best[f"{prefix}_best_lap_rank"] = _safe_rank(best[f"{prefix}_best_lap_ms"], ascending=True)
    best[f"{prefix}_sector_1_rank"] = _safe_rank(best[f"{prefix}_best_sector_1_ms"], ascending=True)
    best[f"{prefix}_sector_2_rank"] = _safe_rank(best[f"{prefix}_best_sector_2_ms"], ascending=True)
    best[f"{prefix}_sector_3_rank"] = _safe_rank(best[f"{prefix}_best_sector_3_ms"], ascending=True)

    violations = (
        clean.groupby("driver_number", as_index=False)
        .apply(lambda frame: pd.Series({f"{prefix}_track_limit_violations": _track_limit_violation_proxy(frame)}), include_groups=False)
        .reset_index()
    )
    best = best.merge(violations[["driver_number", f"{prefix}_track_limit_violations"]], how="left", on="driver_number")
    return best


def _practice_long_run_table(session_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute long-run practice pace and degradation summaries for one session."""

    clean = _session_clean_laps(session_df)
    if clean.empty:
        return pd.DataFrame(
            columns=[
                "driver_number",
                f"{prefix}_long_run_pace",
                f"{prefix}_long_run_rank",
                f"{prefix}_soft_deg_rate",
                f"{prefix}_medium_deg_rate",
                f"{prefix}_hard_deg_rate",
                f"{prefix}_compounds_run_count",
                f"{prefix}_ran_soft",
                f"{prefix}_ran_medium",
                f"{prefix}_ran_hard",
                f"{prefix}_preferred_compound_code",
            ]
        )

    clean["compound"] = clean.get("compound", "unknown").map(_normalize_openf1_compound)
    clean["stint_lap_index"] = _stint_relative_laps(clean)
    long_run = clean.loc[clean["stint_lap_index"] >= 5].copy()
    if long_run.empty:
        long_run = clean.copy()

    base = long_run.groupby("driver_number", as_index=False).agg(
        **{
            f"{prefix}_long_run_pace": ("lap_duration_ms", "median"),
            f"{prefix}_compounds_run_count": ("compound", lambda series: int(pd.Series(series).replace("unknown", np.nan).dropna().nunique())),
        }
    )
    base[f"{prefix}_long_run_rank"] = _safe_rank(base[f"{prefix}_long_run_pace"], ascending=True)

    for compound in ["soft", "medium", "hard"]:
        compound_mask = long_run["compound"].eq(compound)
        compound_frame = long_run.loc[compound_mask].copy()
        if compound_frame.empty:
            base[f"{prefix}_{compound}_deg_rate"] = np.nan
            base[f"{prefix}_ran_{compound}"] = 0
            continue
        deg = compound_frame.groupby("driver_number", as_index=False).apply(
            lambda frame: pd.Series({f"{prefix}_{compound}_deg_rate": _fit_slope(frame["stint_lap_index"], frame["lap_duration_ms"])}),
            include_groups=False,
        ).reset_index()
        base = base.merge(deg[["driver_number", f"{prefix}_{compound}_deg_rate"]], how="left", on="driver_number")
        base[f"{prefix}_ran_{compound}"] = base["driver_number"].isin(compound_frame["driver_number"]).astype(int)

    preferred = (
        long_run.groupby(["driver_number", "compound"], as_index=False)["lap_duration_ms"]
        .median()
        .sort_values(["driver_number", "lap_duration_ms"])
        .drop_duplicates(subset=["driver_number"], keep="first")
        .rename(columns={"compound": f"{prefix}_preferred_compound"})
    )
    preferred[f"{prefix}_preferred_compound_code"] = preferred[f"{prefix}_preferred_compound"].map(_compound_to_code)
    base = base.merge(preferred[["driver_number", f"{prefix}_preferred_compound_code"]], how="left", on="driver_number")
    return base


def _primary_practice_group(session_frames: dict[str, pd.DataFrame]) -> str | None:
    """Return the best available practice session in priority order FP3 -> FP2 -> FP1."""

    for group in ["FP3", "FP2", "FP1"]:
        if group in session_frames and not session_frames[group].empty:
            return group
    return None


def _weekend_driver_numbers(
    weekend_results: pd.DataFrame,
    session_frames: dict[str, pd.DataFrame],
) -> list[str]:
    """Return the union of driver numbers present in results or session data."""

    numbers = set()
    if not weekend_results.empty and "driver_number" in weekend_results.columns:
        numbers |= set(weekend_results["driver_number"].dropna().apply(_normalize_driver_number))
    for frame in session_frames.values():
        if not frame.empty and "driver_number" in frame.columns:
            numbers |= set(frame["driver_number"].dropna().astype(str))
    return sorted(value for value in numbers if value not in {"", "nan", None})


def extract_practice_features(
    season: int,
    round_number: int,
    weekend_results: pd.DataFrame | None = None,
    refresh: bool = False,
    client: OpenF1Client | None = None,
) -> pd.DataFrame:
    """Extract aggregated practice-session features for a race weekend."""

    coverage, session_frames = _load_weekend_session_frames(season, round_number, refresh=refresh, client=client)
    practice_frames = {group: frame for group, frame in session_frames.items() if group in {"FP1", "FP2", "FP3"}}
    driver_numbers = _weekend_driver_numbers(weekend_results if weekend_results is not None else pd.DataFrame(), practice_frames)
    if not driver_numbers:
        return pd.DataFrame(columns=["season", "round", "driver_number", *PRACTICE_FEATURE_COLUMNS, *SESSION_CONTEXT_COLUMNS])

    primary_group = _primary_practice_group(practice_frames)
    primary_prefix = primary_group.lower() if primary_group else "fp3"

    tables: dict[str, pd.DataFrame] = {}
    for group, frame in practice_frames.items():
        prefix = group.lower()
        best = _field_best_lap_table(frame, prefix)
        long_run = _practice_long_run_table(frame, prefix)
        tables[group] = best.merge(long_run, how="outer", on="driver_number")

    rows: list[dict[str, Any]] = []
    for driver_number in driver_numbers:
        row: dict[str, Any] = {
            "season": int(season),
            "round": int(round_number),
            "driver_number": driver_number,
            "weekend_session_count": int(len(coverage)),
            "has_fp1": int("FP1" in practice_frames and not practice_frames["FP1"].empty),
            "has_fp2": int("FP2" in practice_frames and not practice_frames["FP2"].empty),
            "has_fp3": int("FP3" in practice_frames and not practice_frames["FP3"].empty),
            "has_quali": int("QUALI" in session_frames and not session_frames["QUALI"].empty),
            "has_race": int("RACE" in session_frames and not session_frames["RACE"].empty),
            "has_sprint": int("SPRINT" in session_frames and not session_frames["SPRINT"].empty),
            "has_sprint_shootout": int("SPRINT_SHOOTOUT" in session_frames and not session_frames["SPRINT_SHOOTOUT"].empty),
        }
        for context_name, session_group, source_column in [
            ("fp1_air_temp_c", "FP1", "session_air_temperature_c"),
            ("fp1_track_temp_c", "FP1", "session_track_temperature_c"),
            ("race_air_temp_c", "RACE", "session_air_temperature_c"),
            ("race_track_temp_c", "RACE", "session_track_temperature_c"),
        ]:
            source = session_frames.get(session_group, pd.DataFrame())
            value = float(pd.to_numeric(source.get(source_column), errors="coerce").median()) if not source.empty and source_column in source.columns else np.nan
            row[context_name] = value

        primary_table = tables.get(primary_group, pd.DataFrame())
        if primary_group and not primary_table.empty and driver_number in primary_table["driver_number"].tolist():
            primary_row = primary_table.loc[primary_table["driver_number"] == driver_number].iloc[0]
            row["fp_best_lap_rank"] = float(primary_row.get(f"{primary_prefix}_best_lap_rank", np.nan))
            row["fp_long_run_pace"] = float(primary_row.get(f"{primary_prefix}_long_run_pace", np.nan))
            row["fp_long_run_rank"] = float(primary_row.get(f"{primary_prefix}_long_run_rank", np.nan))
            row["fp_sector_1_rank"] = float(primary_row.get(f"{primary_prefix}_sector_1_rank", np.nan))
            row["fp_sector_2_rank"] = float(primary_row.get(f"{primary_prefix}_sector_2_rank", np.nan))
            row["fp_sector_3_rank"] = float(primary_row.get(f"{primary_prefix}_sector_3_rank", np.nan))
            row["fp_sector_strength_score"] = float(
                np.nanmean(
                    [
                        primary_row.get(f"{primary_prefix}_sector_1_rank", np.nan),
                        primary_row.get(f"{primary_prefix}_sector_2_rank", np.nan),
                        primary_row.get(f"{primary_prefix}_sector_3_rank", np.nan),
                    ]
                )
            )
            row["fp_lap_time_std"] = float(primary_row.get(f"{primary_prefix}_lap_time_std", np.nan))
            row["fp_track_limit_violations"] = float(primary_row.get(f"{primary_prefix}_track_limit_violations", np.nan))
            row["fp_compounds_run_count"] = float(primary_row.get(f"{primary_prefix}_compounds_run_count", np.nan))
            ran_soft = pd.to_numeric(primary_row.get(f"{primary_prefix}_ran_soft", 0), errors="coerce")
            ran_medium = pd.to_numeric(primary_row.get(f"{primary_prefix}_ran_medium", 0), errors="coerce")
            ran_hard = pd.to_numeric(primary_row.get(f"{primary_prefix}_ran_hard", 0), errors="coerce")
            row["fp_ran_soft"] = int(0 if pd.isna(ran_soft) else ran_soft)
            row["fp_ran_medium"] = int(0 if pd.isna(ran_medium) else ran_medium)
            row["fp_ran_hard"] = int(0 if pd.isna(ran_hard) else ran_hard)
            row["fp_soft_deg_rate"] = float(primary_row.get(f"{primary_prefix}_soft_deg_rate", np.nan))
            row["fp_medium_deg_rate"] = float(primary_row.get(f"{primary_prefix}_medium_deg_rate", np.nan))
            row["fp_hard_deg_rate"] = float(primary_row.get(f"{primary_prefix}_hard_deg_rate", np.nan))
            row["fp_preferred_compound_code"] = float(primary_row.get(f"{primary_prefix}_preferred_compound_code", np.nan))
        else:
            for column in PRACTICE_FEATURE_COLUMNS:
                row[column] = np.nan

        if "FP1" in tables and "FP3" in tables:
            fp1_best = tables["FP1"].loc[tables["FP1"]["driver_number"] == driver_number, "fp1_best_lap_ms"]
            fp3_best = tables["FP3"].loc[tables["FP3"]["driver_number"] == driver_number, "fp3_best_lap_ms"]
            if not fp1_best.empty and not fp3_best.empty and pd.notna(fp1_best.iloc[0]) and pd.notna(fp3_best.iloc[0]):
                row["fp_improvement_fp1_to_fp3"] = float((fp1_best.iloc[0] - fp3_best.iloc[0]) / max(fp1_best.iloc[0], 1.0))
            else:
                row["fp_improvement_fp1_to_fp3"] = np.nan
        else:
            row["fp_improvement_fp1_to_fp3"] = np.nan

        if weekend_results is not None and not weekend_results.empty and "constructor_id" in weekend_results.columns:
            team_row = weekend_results.loc[
                weekend_results["driver_number"].apply(_normalize_driver_number) == driver_number
            ].head(1)
            constructor_id = str(team_row["constructor_id"].iloc[0]) if not team_row.empty else None
            if constructor_id and primary_group:
                team_primary = primary_table.merge(
                    weekend_results[["driver_number", "constructor_id"]].assign(
                        driver_number=lambda frame: frame["driver_number"].apply(_normalize_driver_number)
                    ),
                    how="left",
                    on="driver_number",
                )
                teammate = team_primary.loc[
                    (team_primary["constructor_id"] == constructor_id) & (team_primary["driver_number"] != driver_number)
                ]
                self_row = team_primary.loc[team_primary["driver_number"] == driver_number]
                if not self_row.empty and not teammate.empty:
                    self_best = float(self_row[f"{primary_prefix}_best_lap_ms"].iloc[0])
                    teammate_best = float(teammate[f"{primary_prefix}_best_lap_ms"].iloc[0])
                    session_median = float(team_primary[f"{primary_prefix}_best_lap_ms"].median())
                    row["fp_pace_delta_to_teammate"] = (teammate_best - self_best) / max(session_median, 1.0)
                else:
                    row["fp_pace_delta_to_teammate"] = np.nan
            else:
                row["fp_pace_delta_to_teammate"] = np.nan
        else:
            row["fp_pace_delta_to_teammate"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def extract_qualifying_features(
    season: int,
    round_number: int,
    weekend_results: pd.DataFrame | None = None,
    refresh: bool = False,
    client: OpenF1Client | None = None,
) -> pd.DataFrame:
    """Extract detailed qualifying features for one race weekend."""

    _, session_frames = _load_weekend_session_frames(season, round_number, refresh=refresh, client=client)
    quali_df = session_frames.get("QUALI", pd.DataFrame())
    if weekend_results is None:
        race_results_path = CONFIG["race_results_path"]
        weekend_results = load_dataframe(race_results_path).loc[
            lambda frame: (frame["season"] == int(season)) & (frame["round"] == int(round_number))
        ].copy() if Path(race_results_path).exists() else pd.DataFrame()

    if weekend_results.empty and quali_df.empty:
        return pd.DataFrame(columns=["season", "round", "driver_number", *QUALIFYING_DEEP_FEATURE_COLUMNS, *OPENF1_FALLBACK_FEATURE_COLUMNS])

    results = weekend_results.copy()
    if not results.empty:
        results["driver_number"] = results["driver_number"].apply(_normalize_driver_number)
        for source, target in [("q1", "q1_lap_time_ms"), ("q2", "q2_lap_time_ms"), ("q3", "q3_lap_time_ms")]:
            results[target] = _coerce_openf1_duration_ms(results[source]) if source in results.columns else np.nan
        results["best_qualifying_time_ms"] = results[["q3_lap_time_ms", "q2_lap_time_ms", "q1_lap_time_ms"]].bfill(axis=1).iloc[:, 0]
    else:
        results = pd.DataFrame(columns=["driver_number", "constructor_id", "q1_lap_time_ms", "q2_lap_time_ms", "q3_lap_time_ms", "best_qualifying_time_ms"])

    field_best = _field_best_lap_table(quali_df, "openf1") if not quali_df.empty else pd.DataFrame(columns=["driver_number"])
    driver_numbers = _weekend_driver_numbers(results, {"QUALI": quali_df})

    quali_progress = pd.DataFrame(columns=["driver_number", "track_position_in_quali", "tyre_used_in_q3_code"])
    if not quali_df.empty:
        clean = _session_clean_laps(quali_df)
        if not clean.empty:
            progress_rows: list[dict[str, Any]] = []
            for driver_number, driver_frame in clean.groupby("driver_number", sort=False):
                ordered = driver_frame.sort_values("lap_number").reset_index(drop=True)
                best_idx = int(pd.to_numeric(ordered["lap_duration_ms"], errors="coerce").idxmin()) if not ordered.empty else 0
                normalized_index = best_idx / max(len(ordered) - 1, 1)
                last_stint = ordered.dropna(subset=["compound"]).tail(1)
                progress_rows.append(
                    {
                        "driver_number": driver_number,
                        "track_position_in_quali": float(normalized_index),
                        "tyre_used_in_q3_code": float(_compound_to_code(last_stint["compound"].iloc[0])) if not last_stint.empty else np.nan,
                    }
                )
            quali_progress = pd.DataFrame(progress_rows)

    rows: list[dict[str, Any]] = []
    pole_time = float(results["best_qualifying_time_ms"].min()) if not results.empty and results["best_qualifying_time_ms"].notna().any() else (
        float(field_best["openf1_best_lap_ms"].min()) if not field_best.empty and field_best["openf1_best_lap_ms"].notna().any() else np.nan
    )
    for driver_number in driver_numbers:
        result_row = results.loc[results["driver_number"] == driver_number].head(1)
        openf1_row = field_best.loc[field_best["driver_number"] == driver_number].head(1)
        progress_row = quali_progress.loc[quali_progress["driver_number"] == driver_number].head(1)
        q1 = float(result_row["q1_lap_time_ms"].iloc[0]) if not result_row.empty else np.nan
        q2 = float(result_row["q2_lap_time_ms"].iloc[0]) if not result_row.empty else np.nan
        q3 = float(result_row["q3_lap_time_ms"].iloc[0]) if not result_row.empty else np.nan
        best = float(result_row["best_qualifying_time_ms"].iloc[0]) if not result_row.empty else (
            float(openf1_row["openf1_best_lap_ms"].iloc[0]) if not openf1_row.empty else np.nan
        )
        constructor_id = str(result_row["constructor_id"].iloc[0]) if not result_row.empty and "constructor_id" in result_row.columns else None

        teammate_gap = np.nan
        if constructor_id is not None and not results.empty:
            teammate = results.loc[
                (results["constructor_id"] == constructor_id) & (results["driver_number"] != driver_number)
            ].head(1)
            if not teammate.empty and pd.notna(best):
                teammate_gap = float(best - float(teammate["best_qualifying_time_ms"].iloc[0]))

        sector_dominance = np.nan
        if not openf1_row.empty:
            sector_dominance = float(
                np.nanmean(
                    [
                        openf1_row["openf1_sector_1_rank"].iloc[0] if "openf1_sector_1_rank" in openf1_row.columns else np.nan,
                        openf1_row["openf1_sector_2_rank"].iloc[0] if "openf1_sector_2_rank" in openf1_row.columns else np.nan,
                        openf1_row["openf1_sector_3_rank"].iloc[0] if "openf1_sector_3_rank" in openf1_row.columns else np.nan,
                    ]
                )
            )

        rows.append(
            {
                "season": int(season),
                "round": int(round_number),
                "driver_number": driver_number,
                "q1_lap_time_ms": q1,
                "q2_lap_time_ms": q2,
                "q3_lap_time_ms": q3,
                "quali_gap_to_pole_ms": float(best - pole_time) if pd.notna(best) and pd.notna(pole_time) else np.nan,
                "quali_gap_to_teammate_ms": teammate_gap,
                "quali_improvement_rate": float((q1 - best) / max(q1, 1.0)) if pd.notna(q1) and pd.notna(best) else np.nan,
                "sector_dominance_score": sector_dominance,
                "tyre_used_in_q3_code": float(progress_row["tyre_used_in_q3_code"].iloc[0]) if not progress_row.empty else np.nan,
                "track_position_in_quali": float(progress_row["track_position_in_quali"].iloc[0]) if not progress_row.empty else np.nan,
                "openf1_best_sector_1_ms": float(openf1_row["openf1_best_sector_1_ms"].iloc[0]) if not openf1_row.empty and "openf1_best_sector_1_ms" in openf1_row.columns else np.nan,
                "openf1_best_sector_2_ms": float(openf1_row["openf1_best_sector_2_ms"].iloc[0]) if not openf1_row.empty and "openf1_best_sector_2_ms" in openf1_row.columns else np.nan,
                "openf1_best_sector_3_ms": float(openf1_row["openf1_best_sector_3_ms"].iloc[0]) if not openf1_row.empty and "openf1_best_sector_3_ms" in openf1_row.columns else np.nan,
                "openf1_sector_consistency": float(openf1_row["openf1_lap_time_std"].iloc[0]) if not openf1_row.empty and "openf1_lap_time_std" in openf1_row.columns else np.nan,
                "openf1_max_speed_trap": float(openf1_row["openf1_max_speed_trap"].iloc[0]) if not openf1_row.empty and "openf1_max_speed_trap" in openf1_row.columns else np.nan,
            }
        )

    return pd.DataFrame(rows)


def extract_sprint_features(
    season: int,
    round_number: int,
    weekend_results: pd.DataFrame | None = None,
    refresh: bool = False,
    client: OpenF1Client | None = None,
) -> pd.DataFrame:
    """Extract sprint-weekend features where a sprint session exists."""

    _, session_frames = _load_weekend_session_frames(season, round_number, refresh=refresh, client=client)
    sprint_df = session_frames.get("SPRINT", pd.DataFrame())
    shootout_df = session_frames.get("SPRINT_SHOOTOUT", pd.DataFrame())
    driver_numbers = _weekend_driver_numbers(
        weekend_results if weekend_results is not None else pd.DataFrame(),
        {"SPRINT": sprint_df, "SPRINT_SHOOTOUT": shootout_df},
    )
    if not driver_numbers:
        return pd.DataFrame(columns=["season", "round", "driver_number", *SPRINT_FEATURE_COLUMNS])

    sprint_rank = pd.DataFrame(columns=["driver_number", "sprint_finish_position", "sprint_grid_position"])
    if not sprint_df.empty:
        final_positions = (
            sprint_df.dropna(subset=["session_final_position"])
            .groupby("driver_number", as_index=False)["session_final_position"]
            .last()
            .rename(columns={"session_final_position": "sprint_finish_position"})
        )
        opening_positions = (
            sprint_df.dropna(subset=["session_opening_position"])
            .groupby("driver_number", as_index=False)["session_opening_position"]
            .first()
            .rename(columns={"session_opening_position": "sprint_grid_position"})
        )
        sprint_rank = final_positions.merge(opening_positions, how="outer", on="driver_number")

    if not shootout_df.empty:
        shootout_rank = _field_best_lap_table(shootout_df, "shootout")
        sprint_rank = sprint_rank.merge(
            shootout_rank[["driver_number", "shootout_best_lap_rank"]].rename(columns={"shootout_best_lap_rank": "shootout_rank"}),
            how="left",
            on="driver_number",
        )
        sprint_rank["sprint_grid_position"] = sprint_rank["sprint_grid_position"].fillna(sprint_rank.get("shootout_rank"))

    sprint_compounds = pd.DataFrame(columns=["driver_number", "sprint_compounds_used_count", "sprint_primary_compound_code", "sprint_tyre_life_used"])
    if not sprint_df.empty:
        compound_rows: list[dict[str, Any]] = []
        for driver_number, driver_frame in sprint_df.groupby("driver_number", sort=False):
            compounds = pd.Series(driver_frame.get("compound", "unknown")).map(_normalize_openf1_compound)
            primary_compound = compounds.replace("unknown", np.nan).dropna().mode()
            tyre_life = (
                pd.to_numeric(driver_frame.get("lap_end"), errors="coerce")
                - pd.to_numeric(driver_frame.get("lap_start"), errors="coerce")
                + 1.0
            )
            compound_rows.append(
                {
                    "driver_number": driver_number,
                    "sprint_compounds_used_count": int(compounds.replace("unknown", np.nan).dropna().nunique()),
                    "sprint_primary_compound_code": float(_compound_to_code(primary_compound.iloc[0])) if not primary_compound.empty else np.nan,
                    "sprint_tyre_life_used": float(pd.to_numeric(tyre_life, errors="coerce").max()) if not pd.to_numeric(tyre_life, errors="coerce").dropna().empty else np.nan,
                }
            )
        sprint_compounds = pd.DataFrame(compound_rows)

    rows: list[dict[str, Any]] = []
    for driver_number in driver_numbers:
        sprint_row = sprint_rank.loc[sprint_rank["driver_number"] == driver_number].head(1)
        compound_row = sprint_compounds.loc[sprint_compounds["driver_number"] == driver_number].head(1)
        finish_position = float(sprint_row["sprint_finish_position"].iloc[0]) if not sprint_row.empty and "sprint_finish_position" in sprint_row.columns else np.nan
        grid_position = float(sprint_row["sprint_grid_position"].iloc[0]) if not sprint_row.empty and "sprint_grid_position" in sprint_row.columns else np.nan
        sprint_points = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}.get(int(finish_position), 0) if pd.notna(finish_position) else 0
        rows.append(
            {
                "season": int(season),
                "round": int(round_number),
                "driver_number": driver_number,
                "is_sprint_weekend": int(not sprint_df.empty),
                "sprint_finish_position": finish_position,
                "sprint_points_scored": float(sprint_points),
                "sprint_vs_expected": float(finish_position - grid_position) if pd.notna(finish_position) and pd.notna(grid_position) else np.nan,
                "sprint_compounds_used_count": float(compound_row["sprint_compounds_used_count"].iloc[0]) if not compound_row.empty else np.nan,
                "sprint_primary_compound_code": float(compound_row["sprint_primary_compound_code"].iloc[0]) if not compound_row.empty else np.nan,
                "sprint_tyre_life_used": float(compound_row["sprint_tyre_life_used"].iloc[0]) if not compound_row.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_track_evolution(
    season: int,
    round_number: int,
    weekend_results: pd.DataFrame | None = None,
    refresh: bool = False,
    client: OpenF1Client | None = None,
) -> pd.DataFrame:
    """Compute weekend track-evolution features for one race weekend."""

    _, session_frames = _load_weekend_session_frames(season, round_number, refresh=refresh, client=client)
    fp1 = session_frames.get("FP1", pd.DataFrame())
    quali = session_frames.get("QUALI", pd.DataFrame())
    race = session_frames.get("RACE", pd.DataFrame())
    driver_numbers = _weekend_driver_numbers(
        weekend_results if weekend_results is not None else pd.DataFrame(),
        {"FP1": fp1, "QUALI": quali},
    )
    if not driver_numbers:
        return pd.DataFrame(columns=["season", "round", "driver_number", *TRACK_EVOLUTION_FEATURE_COLUMNS])

    fp1_best = _field_best_lap_table(fp1, "fp1")
    quali_best = _field_best_lap_table(quali, "quali")
    if weekend_results is not None and not weekend_results.empty:
        qualifying_table = extract_qualifying_features(season, round_number, weekend_results=weekend_results, refresh=refresh, client=client)
        qualifying_best_lookup = qualifying_table.set_index("driver_number")["q3_lap_time_ms"].combine_first(
            qualifying_table.set_index("driver_number")["q2_lap_time_ms"].combine_first(
                qualifying_table.set_index("driver_number")["q1_lap_time_ms"]
            )
        )
    else:
        qualifying_best_lookup = pd.Series(dtype=float)

    fp1_pole = float(fp1_best["fp1_best_lap_ms"].min()) if not fp1_best.empty and fp1_best["fp1_best_lap_ms"].notna().any() else np.nan
    pole_time = float(qualifying_best_lookup.min()) if not qualifying_best_lookup.empty and qualifying_best_lookup.notna().any() else (
        float(quali_best["quali_best_lap_ms"].min()) if not quali_best.empty and "quali_best_lap_ms" in quali_best.columns else np.nan
    )
    track_evolution = (fp1_pole - pole_time) / max(pole_time, 1.0) if pd.notna(fp1_pole) and pd.notna(pole_time) else np.nan
    fp1_track_temp = float(pd.to_numeric(fp1.get("session_track_temperature_c"), errors="coerce").median()) if not fp1.empty else np.nan
    race_track_temp = float(pd.to_numeric(race.get("session_track_temperature_c"), errors="coerce").median()) if not race.empty else np.nan
    temp_delta = float(race_track_temp - fp1_track_temp) if pd.notna(race_track_temp) and pd.notna(fp1_track_temp) else np.nan

    improvement_lookup: dict[str, float] = {}
    field_improvements: list[float] = []
    fp1_lookup = fp1_best.set_index("driver_number")["fp1_best_lap_ms"] if not fp1_best.empty else pd.Series(dtype=float)
    quali_lookup = (
        qualifying_best_lookup
        if not qualifying_best_lookup.empty
        else (quali_best.set_index("driver_number")["quali_best_lap_ms"] if not quali_best.empty else pd.Series(dtype=float))
    )
    for driver_number in driver_numbers:
        if driver_number in fp1_lookup.index and driver_number in quali_lookup.index and pd.notna(fp1_lookup[driver_number]) and pd.notna(quali_lookup[driver_number]):
            improvement = float((fp1_lookup[driver_number] - quali_lookup[driver_number]) / max(fp1_lookup[driver_number], 1.0))
            improvement_lookup[driver_number] = improvement
            field_improvements.append(improvement)
    field_median = float(np.nanmedian(field_improvements)) if field_improvements else np.nan

    rows = [
        {
            "season": int(season),
            "round": int(round_number),
            "driver_number": driver_number,
            "track_evolution_delta": track_evolution,
            "fp1_to_race_temp_delta": temp_delta,
            "rubber_benefit_index": float(improvement_lookup.get(driver_number, np.nan) - field_median)
            if pd.notna(field_median)
            else np.nan,
        }
        for driver_number in driver_numbers
    ]
    return pd.DataFrame(rows)


def _save_failure_log(failures: list[dict[str, Any]]) -> pd.DataFrame:
    """Persist FastF1 failures for later audit/reporting."""

    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=["season", "round", "session_type", "error"])
    save_dataframe(failure_frame, CONFIG["fastf1_failures_path"])
    save_json(failures, CONFIG["fastf1_failures_json_path"])
    return failure_frame


def _merge_weekend_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge per-driver weekend feature frames on the persisted key."""

    merged: pd.DataFrame | None = None
    for frame in frames:
        if frame is None or frame.empty:
            continue
        current = frame.copy()
        current["driver_number"] = current["driver_number"].apply(_normalize_driver_number)
        current = current.drop_duplicates(subset=["season", "round", "driver_number"], keep="last")
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current, how="outer", on=["season", "round", "driver_number"])
    if merged is None:
        return _empty_weekend_feature_frame()
    return _ensure_weekend_feature_schema(merged)


def _apply_team_practice_delta(weekend_features: pd.DataFrame, race_results: pd.DataFrame) -> pd.DataFrame:
    """Add a weekend-vs-previous-weekend team pace delta based on practice rank."""

    if weekend_features.empty:
        return _ensure_weekend_feature_schema(weekend_features)

    constructor_map = race_results[["season", "round", "driver_number", "constructor_id"]].drop_duplicates().copy()
    constructor_map["driver_number"] = constructor_map["driver_number"].apply(_normalize_driver_number)
    enriched = weekend_features.merge(
        constructor_map,
        how="left",
        on=["season", "round", "driver_number"],
    )
    team_weekend = (
        enriched.groupby(["constructor_id", "season", "round"], as_index=False)["fp_best_lap_rank"]
        .median()
        .sort_values(["constructor_id", "season", "round"])
    )
    team_weekend["fp_team_pace_delta_vs_last_race"] = (
        team_weekend.groupby("constructor_id", sort=False)["fp_best_lap_rank"].shift(1) - team_weekend["fp_best_lap_rank"]
    )
    enriched = enriched.drop(columns=["fp_team_pace_delta_vs_last_race"], errors="ignore").merge(
        team_weekend[["constructor_id", "season", "round", "fp_team_pace_delta_vs_last_race"]],
        how="left",
        on=["constructor_id", "season", "round"],
    )
    return _ensure_weekend_feature_schema(enriched.drop(columns=["constructor_id"], errors="ignore"))


def _fetch_weekend_fastf1(
    fastf1: Any,
    season: int,
    round_number: int,
    weekend_results: pd.DataFrame,
    refresh: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fetch or load FastF1 features for one race weekend."""

    parquet_path, json_path = _weekend_cache_paths(season, round_number)
    if parquet_path.exists():
        cached_weekend = load_parquet(parquet_path)
        cached_failures = load_json(json_path, default=[])
        if not cached_weekend.empty or not refresh:
            return cached_weekend, cached_failures
        if season < 2023 and cached_failures:
            LOGGER.info(
                "Using cached FastF1 failure artifact for season=%s round=%s because earlier telemetry fetches failed",
                season,
                round_number,
            )
            return cached_weekend, cached_failures

    failures: list[dict[str, Any]] = []
    q_features = pd.DataFrame(columns=["season", "round", "driver_number", *QUALIFYING_FEATURE_COLUMNS])
    r_features = pd.DataFrame(columns=["season", "round", "driver_number", *RACE_FEATURE_COLUMNS])

    try:
        qualifying_session = load_session_safe(season, round_number, "Q")
        if hasattr(qualifying_session, "_laps"):
            qualifying_session._laps = normalize_driver_numbers(qualifying_session.laps, weekend_results)
        q_features = _extract_qualifying_features(qualifying_session, weekend_results)
    except Exception as exc:
        failures.append({"season": season, "round": round_number, "session_type": "Q", "error": str(exc)})
        LOGGER.warning("FastF1 qualifying fetch failed for %s R%s: %s", season, round_number, exc)

    try:
        race_session = load_session_safe(season, round_number, "R")
        if hasattr(race_session, "_laps"):
            race_session._laps = normalize_driver_numbers(race_session.laps, weekend_results)
        r_features = _extract_race_features(race_session, weekend_results)
    except Exception as exc:
        failures.append({"season": season, "round": round_number, "session_type": "R", "error": str(exc)})
        LOGGER.warning("FastF1 race fetch failed for %s R%s: %s", season, round_number, exc)

    weekend_frame = q_features.merge(
        r_features,
        how="outer",
        on=["season", "round", "driver_number"],
    )
    if weekend_frame.empty:
        weekend_frame = pd.DataFrame(
            columns=["season", "round", "driver_number", *QUALIFYING_FEATURE_COLUMNS, *RACE_FEATURE_COLUMNS]
        )
    save_parquet(weekend_frame, parquet_path)
    save_json(failures, json_path)
    return weekend_frame, failures


def build_fastf1_feature_frame(
    race_results: pd.DataFrame,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch FastF1 telemetry plus OpenF1 full-session weekend features."""

    validate_columns(
        race_results,
        ["season", "round", "driver_id", "driver_number", "grid_position"],
        "race results for FastF1 enrichment",
    )

    fastf1: Any | None = None
    failures: list[dict[str, Any]] = []
    try:
        fastf1 = _import_fastf1()
        ensure_directories([CONFIG["fastf1_cache_dir"]])
        _configure_fastf1_sources(fastf1)
        fastf1.Cache.enable_cache(str(CONFIG["fastf1_cache_dir"]))
    except ImportError as exc:
        LOGGER.warning("FastF1 import unavailable, continuing with OpenF1-only weekend features: %s", exc)
        failures.append({"season": None, "round": None, "session_type": "FASTF1_IMPORT", "error": str(exc)})

    eligible = race_results.loc[
        race_results["season"].between(CONFIG["season_start"], CONFIG["season_end"])
    ].copy()
    eligible["driver_number"] = eligible["driver_number"].apply(_normalize_driver_number)
    weekend_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    openf1_client = OpenF1Client()

    for (season, round_number), weekend_results in eligible.groupby(["season", "round"], sort=True):
        season_int = int(season)
        round_int = int(round_number)
        coverage = fetch_all_sessions(season=season_int, round_number=round_int, refresh=False, client=openf1_client)
        if not coverage.empty:
            coverage_frames.append(coverage)

        fastf1_weekend = pd.DataFrame(columns=["season", "round", "driver_number", *QUALIFYING_FEATURE_COLUMNS, *RACE_FEATURE_COLUMNS])
        weekend_failures: list[dict[str, Any]] = []
        if fastf1 is not None:
            fastf1_weekend, weekend_failures = _fetch_weekend_fastf1(
                fastf1=fastf1,
                season=season_int,
                round_number=round_int,
                weekend_results=weekend_results,
                refresh=refresh,
            )

        practice_features = extract_practice_features(
            season=season_int,
            round_number=round_int,
            weekend_results=weekend_results,
            refresh=refresh,
            client=openf1_client,
        )
        qualifying_features = extract_qualifying_features(
            season=season_int,
            round_number=round_int,
            weekend_results=weekend_results,
            refresh=refresh,
            client=openf1_client,
        )
        sprint_features = extract_sprint_features(
            season=season_int,
            round_number=round_int,
            weekend_results=weekend_results,
            refresh=refresh,
            client=openf1_client,
        )
        track_evolution_features = compute_track_evolution(
            season=season_int,
            round_number=round_int,
            weekend_results=weekend_results,
            refresh=refresh,
            client=openf1_client,
        )

        weekend_frame = _merge_weekend_feature_frames(
            [
                fastf1_weekend,
                practice_features,
                qualifying_features,
                sprint_features,
                track_evolution_features,
            ]
        )
        weekend_frames.append(weekend_frame)
        failures.extend(weekend_failures)

    if weekend_frames:
        fastf1_features = _ensure_weekend_feature_schema(pd.concat(weekend_frames, ignore_index=True))
    else:
        fastf1_features = _empty_weekend_feature_frame()

    fastf1_features["driver_number"] = fastf1_features["driver_number"].apply(_normalize_driver_number)
    fastf1_features = _apply_team_practice_delta(
        fastf1_features.drop_duplicates(subset=["season", "round", "driver_number"], keep="last"),
        race_results,
    )
    save_dataframe(fastf1_features, CONFIG["fastf1_features_path"])

    if coverage_frames:
        coverage_frame = pd.concat(coverage_frames, ignore_index=True).sort_values(["season", "round", "session_group", "session_name"])
    else:
        coverage_frame = pd.DataFrame(
            columns=[
                "season",
                "round",
                "session_key",
                "session_name",
                "session_group",
                "parquet_path",
                "driver_rows",
                "has_laps",
                "has_stints",
                "has_position",
                "has_car_data",
                "has_weather",
            ]
        )
    save_dataframe(coverage_frame, CONFIG["openf1_coverage_path"])
    failure_frame = _save_failure_log(failures)
    LOGGER.info(
        "Weekend telemetry pipeline complete: %s feature rows, %s session coverage rows, %s recorded fetch failures",
        len(fastf1_features),
        len(coverage_frame),
        len(failure_frame),
    )
    return fastf1_features, failure_frame


def run_fastf1_pipeline(
    race_results: pd.DataFrame | None = None,
    refresh: bool = False,
    skip: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the FastF1 stage, optionally skipping it for quick iterations."""

    ensure_directories()
    if skip:
        empty_features = _empty_weekend_feature_frame()
        empty_failures = pd.DataFrame(columns=["season", "round", "session_type", "error"])
        save_dataframe(empty_features, CONFIG["fastf1_features_path"])
        save_dataframe(empty_failures, CONFIG["fastf1_failures_path"])
        save_json([], CONFIG["fastf1_failures_json_path"])
        save_dataframe(
            pd.DataFrame(
                columns=[
                    "season",
                    "round",
                    "session_key",
                    "session_name",
                    "session_group",
                    "parquet_path",
                    "driver_rows",
                    "has_laps",
                    "has_stints",
                    "has_position",
                    "has_car_data",
                    "has_weather",
                ]
            ),
            CONFIG["openf1_coverage_path"],
        )
        LOGGER.info("Skipping FastF1 pipeline by request")
        return empty_features, empty_failures

    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    return build_fastf1_feature_frame(race_results, refresh=refresh)
