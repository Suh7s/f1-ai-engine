"""Operational race-weekend workflow for the living F1 AI system."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_pipeline import ensure_full_pipeline  # noqa: E402
from src.calibration_engine import CalibrationEngine  # noqa: E402
from src.car_architecture import (  # noqa: E402
    build_constructor_event_evidence_frame,
    get_car_architecture,
    update_car_architecture_from_race,
)
from src.data_pipeline import (  # noqa: E402
    ErgastClient,
    OpenF1Client,
    fetch_pit_stops_openf1,
    fetch_sessions_openf1,
    fetch_stints,
    load_calendar,
    normalize_qualifying_payload,
)
from src.elo_rating import get_driver_dry_elo  # noqa: E402
from src.feature_engineering import build_race_inference_frame  # noqa: E402
from src.fastf1_pipeline import (  # noqa: E402
    compute_track_evolution,
    extract_practice_features,
    extract_qualifying_features,
    extract_sprint_features,
    fetch_all_sessions,
    WEEKEND_FEATURE_COLUMNS,
)
from src.prediction_tracker import (  # noqa: E402
    build_race_accuracy_report,
    load_pre_race_prediction,
    save_pre_race_prediction,
    score_prediction,
)
from src.race_simulator import RaceConfig, compute_upset_probability, simulate_race  # noqa: E402
from src.registry import get_driver_name, get_season_roster  # noqa: E402
from src.scheduler import auto_update  # noqa: E402
from src.tire_model import TireModel  # noqa: E402
from src.utils import CONFIG as BASE_CONFIG, format_race_label, get_gp_name, get_logger, load_dataframe, save_json, slugify  # noqa: E402
from src.weather_pipeline import CIRCUIT_COORDS  # noqa: E402

CONFIG: dict[str, Any] = {
    "processed_dir": BASE_CONFIG["processed_dir"],
    "predictions_dir": BASE_CONFIG["predictions_dir"],
    "forecast_url": "https://api.open-meteo.com/v1/forecast",
    "forecast_timeout": 30,
    "live_poll_seconds": 60,
}

LOGGER = get_logger(__name__)
CALIBRATION_ENGINE = CalibrationEngine()
TIRE_MODEL = TireModel()


def _race_metadata(season: int, round_number: int) -> dict[str, Any]:
    """Return calendar metadata for a race round."""

    calendar = load_calendar(season, allow_fetch=True)
    row = calendar.loc[calendar["round"] == int(round_number)].head(1)
    if row.empty:
        return {
            "season": int(season),
            "round": int(round_number),
            "race_name": get_gp_name(season, round_number),
            "circuit_id": "unknown",
            "date": None,
            "country": None,
        }
    return {
        "season": int(season),
        "round": int(round_number),
        "race_name": str(row["race_name"].iloc[0]),
        "circuit_id": str(row["circuit_id"].iloc[0]),
        "circuit_name": str(row["circuit_name"].iloc[0]) if "circuit_name" in row.columns else str(row["circuit_id"].iloc[0]),
        "country": str(row["country"].iloc[0]) if "country" in row.columns else None,
        "date": pd.to_datetime(row["date"].iloc[0], errors="coerce"),
    }


def _meeting_sessions_for_round(client: OpenF1Client, season: int, round_number: int) -> pd.DataFrame:
    """Resolve the OpenF1 meeting sessions corresponding to a season round."""

    try:
        sessions = fetch_sessions_openf1(client, season)
    except Exception as exc:
        LOGGER.debug("Non-critical error fetching OpenF1 sessions: %s", exc)
        cached_paths = [
            CONFIG["processed_dir"] / "openf1_sessions.csv",
            BASE_CONFIG["raw_dir"] / "openf1_sessions_all.csv",
            BASE_CONFIG["raw_dir"] / f"openf1_sessions_{season}.csv",
        ]
        sessions = pd.DataFrame()
        for path in cached_paths:
            if path.exists():
                sessions = load_dataframe(path, parse_dates=["date_start"])
                sessions = sessions.loc[sessions["season"] == int(season)].copy() if "season" in sessions.columns else sessions
                if not sessions.empty:
                    break
    if sessions.empty:
        return sessions

    frame = sessions.copy()
    frame["date_start"] = pd.to_datetime(frame["date_start"], errors="coerce", utc=True)
    frame["meeting_id"] = (
        frame["meeting_name"].fillna("")
        + "|"
        + frame["location"].fillna("")
        + "|"
        + frame["country_name"].fillna("")
    ).str.strip("|")

    meeting_summary = (
        frame.groupby("meeting_id", as_index=False)
        .agg(meeting_start=("date_start", "min"))
        .sort_values("meeting_start")
        .reset_index(drop=True)
    )

    metadata = _race_metadata(season, round_number)
    race_date = metadata.get("date")
    if pd.notna(race_date):
        target = pd.Timestamp(race_date).tz_localize("UTC") if pd.Timestamp(race_date).tzinfo is None else pd.Timestamp(race_date)
        meeting_summary["date_gap"] = (meeting_summary["meeting_start"] - target).abs()
        meeting_id = str(meeting_summary.sort_values("date_gap").iloc[0]["meeting_id"])
    else:
        if round_number > len(meeting_summary):
            return pd.DataFrame(columns=frame.columns)
        meeting_id = str(meeting_summary.iloc[int(round_number) - 1]["meeting_id"])

    return frame.loc[frame["meeting_id"] == meeting_id].sort_values("date_start").reset_index(drop=True)


def _resolve_session_key(meeting_sessions: pd.DataFrame, session_label: str) -> int | None:
    """Return the most relevant OpenF1 session key for a weekend stage."""

    if meeting_sessions.empty:
        return None
    label = session_label.strip().lower()
    alias_map = {
        "fp3": ["practice 3", "free practice 3", "fp3"],
        "quali": ["qualifying"],
        "live": ["race"],
        "post": ["race"],
    }
    aliases = alias_map.get(label, [label])
    for alias in aliases:
        matched = meeting_sessions.loc[
            meeting_sessions["session_name"].astype(str).str.lower().str.contains(alias, na=False)
        ].copy()
        if not matched.empty:
            return int(matched.iloc[0]["session_key"])
    return int(meeting_sessions.iloc[0]["session_key"])


def _fetch_openf1_laps(client: OpenF1Client, session_key: int | str) -> pd.DataFrame:
    """Fetch OpenF1 lap data for a session."""

    payload = client.fetch("/laps", params={"session_key": session_key})
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(columns=["driver_number", "lap_number", "lap_duration", "position"])

    for source, target in [
        ("duration_sector_1", "sector_1"),
        ("duration_sector_2", "sector_2"),
        ("duration_sector_3", "sector_3"),
    ]:
        if source in frame.columns and target not in frame.columns:
            frame[target] = pd.to_numeric(frame[source], errors="coerce")

    if "lap_duration" not in frame.columns:
        if "lap_time" in frame.columns:
            frame["lap_duration"] = pd.to_numeric(frame["lap_time"], errors="coerce")
        elif "lap_duration_seconds" in frame.columns:
            frame["lap_duration"] = pd.to_numeric(frame["lap_duration_seconds"], errors="coerce")
        else:
            frame["lap_duration"] = np.nan

    if "lap_number" not in frame.columns:
        if "lap" in frame.columns:
            frame["lap_number"] = pd.to_numeric(frame["lap"], errors="coerce")
        else:
            frame["lap_number"] = np.nan

    if "position" not in frame.columns:
        frame["position"] = np.nan

    if "driver_number" not in frame.columns and "driver_num" in frame.columns:
        frame["driver_number"] = frame["driver_num"]

    frame["driver_number"] = frame["driver_number"].astype(str)
    for column in ["lap_duration", "lap_number", "position", "st_speed"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _driver_lookup(season: int) -> pd.DataFrame:
    """Build a driver-number to driver-id mapping from processed data and the manual registry."""

    race_results_path = CONFIG["processed_dir"] / "race_results.csv"
    if race_results_path.exists():
        race_results = load_dataframe(race_results_path)
        mapping = (
            race_results[["driver_number", "driver_id", "driver_name", "constructor_id", "constructor_name"]]
            .dropna(subset=["driver_id"])
            .copy()
        )
        mapping["driver_number"] = mapping["driver_number"].astype(str)
        mapping = mapping.drop_duplicates(subset=["driver_number", "driver_id"], keep="last")
    else:
        mapping = pd.DataFrame(columns=["driver_number", "driver_id", "driver_name", "constructor_id", "constructor_name"])

    roster = pd.DataFrame(get_season_roster(season))
    if roster.empty:
        roster = pd.DataFrame(columns=["driver_number", "driver_id", "driver_name", "constructor_id", "constructor_name"])
    if "driver_number" not in roster.columns:
        roster["driver_number"] = pd.NA
    roster["driver_number"] = roster["driver_number"].astype(str)
    combined = pd.concat([mapping, roster], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=["driver_number", "driver_id"], keep="last")


def _attach_driver_identity(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    """Attach canonical driver ids and names to OpenF1 driver-number data."""

    if frame.empty:
        return frame
    lookup = _driver_lookup(season)
    enriched = frame.copy()
    enriched["driver_number"] = enriched["driver_number"].astype(str)
    enriched = enriched.merge(
        lookup[["driver_number", "driver_id", "driver_name", "constructor_id", "constructor_name"]],
        how="left",
        on="driver_number",
    )
    enriched["driver_id"] = enriched["driver_id"].fillna(enriched["driver_number"].map(lambda number: f"driver_{number}"))
    enriched["driver_name"] = enriched["driver_name"].fillna(enriched["driver_id"].map(get_driver_name))
    return enriched


def _weather_forecast(circuit_id: str, race_date: Any) -> dict[str, Any]:
    """Fetch a simple race-day forecast and classify expected conditions."""

    coords = CIRCUIT_COORDS.get(str(circuit_id))
    if coords is None:
        return {
            "race_precipitation_mm": 0.0,
            "race_max_wind_kmh": np.nan,
            "race_temperature_c": np.nan,
            "weather_mode": "dry",
        }

    target_date = pd.to_datetime(race_date, errors="coerce")
    if pd.isna(target_date):
        target_date = pd.Timestamp(datetime.utcnow().date())

    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "start_date": target_date.strftime("%Y-%m-%d"),
        "end_date": target_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max,weather_code",
        "timezone": "GMT",
    }

    try:
        response = requests.get(CONFIG["forecast_url"], params=params, timeout=CONFIG["forecast_timeout"])
        response.raise_for_status()
        payload = response.json().get("daily", {})
        precipitation = float((payload.get("precipitation_sum") or [0.0])[0] or 0.0)
        wind = float((payload.get("wind_speed_10m_max") or [np.nan])[0] or np.nan)
        temperature = float((payload.get("temperature_2m_max") or [np.nan])[0] or np.nan)
        weather_code = int((payload.get("weather_code") or [0])[0] or 0)
    except Exception as exc:
        LOGGER.warning("Weather forecast fetch failed for %s: %s", circuit_id, exc)
        precipitation = 0.0
        wind = np.nan
        temperature = np.nan
        weather_code = 0

    if precipitation > 2.0 or weather_code in {61, 63, 65, 80, 81, 82}:
        weather_mode = "wet"
    elif 0.5 <= precipitation <= 2.0:
        weather_mode = "mixed"
    else:
        weather_mode = "dry"

    return {
        "race_precipitation_mm": precipitation,
        "race_max_wind_kmh": wind,
        "race_temperature_c": temperature,
        "weather_mode": weather_mode,
    }


def _normalize_driver_number(value: Any) -> str | None:
    """Normalize driver numbers across cached and live weekend sources."""

    if pd.isna(value) or value in (None, ""):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _merge_driver_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge weekend feature frames on the per-driver key."""

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
        return pd.DataFrame(columns=["season", "round", "driver_number"])
    return merged


def _provisional_qualifying_from_roster(season: int, round_number: int) -> pd.DataFrame:
    """Build a fallback grid using current dry ELO when live qualifying data is unavailable."""

    roster = pd.DataFrame(get_season_roster(season))
    if roster.empty:
        return pd.DataFrame(columns=["driver_id", "driver_name", "qualifying_position"])

    if "driver_name" not in roster.columns:
        roster["driver_name"] = roster["driver_id"].map(get_driver_name)
    else:
        roster["driver_name"] = roster["driver_name"].fillna(roster["driver_id"].map(get_driver_name))
    roster["elo_proxy"] = roster["driver_id"].map(lambda driver_id: get_driver_dry_elo(driver_id, season, round_number))
    roster = roster.sort_values("elo_proxy", ascending=False).reset_index(drop=True)
    roster["qualifying_position"] = np.arange(1, len(roster) + 1)
    return roster[["driver_id", "driver_name", "qualifying_position"]]


def _weekend_context_frame(
    season: int,
    round_number: int,
    qualifying: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a per-driver weekend context frame from the season roster."""

    roster = pd.DataFrame(get_season_roster(season))
    if roster.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "round",
                "driver_id",
                "driver_name",
                "driver_number",
                "constructor_id",
                "constructor_name",
                "qualifying_position",
            ]
        )

    weekend = roster.copy()
    weekend["season"] = int(season)
    weekend["round"] = int(round_number)
    if "driver_name" not in weekend.columns:
        weekend["driver_name"] = weekend["driver_id"].map(get_driver_name)
    else:
        weekend["driver_name"] = weekend["driver_name"].fillna(weekend["driver_id"].map(get_driver_name))
    if "driver_number" not in weekend.columns:
        weekend["driver_number"] = pd.NA
    if "constructor_id" not in weekend.columns:
        weekend["constructor_id"] = pd.NA
    if "constructor_name" not in weekend.columns:
        weekend["constructor_name"] = weekend["constructor_id"]
    weekend["driver_number"] = weekend["driver_number"].apply(_normalize_driver_number)
    if qualifying is not None and not qualifying.empty:
        qualifying_frame = qualifying.copy()
        weekend = weekend.merge(
            qualifying_frame[["driver_id", "qualifying_position"]].drop_duplicates("driver_id"),
            how="left",
            on="driver_id",
        )
    else:
        weekend["qualifying_position"] = np.nan
    if "qualifying_position" not in weekend.columns:
        weekend["qualifying_position"] = np.nan
    return weekend


def _weekend_feature_snapshot(
    season: int,
    round_number: int,
    qualifying: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and assemble the current round's weekend-session features."""

    weekend_context = _weekend_context_frame(season, round_number, qualifying=qualifying)
    client = OpenF1Client()
    coverage = fetch_all_sessions(season=season, round_number=round_number, refresh=False, client=client)
    practice = extract_practice_features(
        season=season,
        round_number=round_number,
        weekend_results=weekend_context,
        refresh=False,
        client=client,
    )
    qualifying_features = extract_qualifying_features(
        season=season,
        round_number=round_number,
        weekend_results=weekend_context,
        refresh=False,
        client=client,
    )
    sprint_features = extract_sprint_features(
        season=season,
        round_number=round_number,
        weekend_results=weekend_context,
        refresh=False,
        client=client,
    )
    track_evolution = compute_track_evolution(
        season=season,
        round_number=round_number,
        weekend_results=weekend_context,
        refresh=False,
        client=client,
    )
    merged = _merge_driver_frames([practice, qualifying_features, sprint_features, track_evolution])
    if merged.empty:
        return coverage, pd.DataFrame(
            columns=[
                "season",
                "round",
                "driver_id",
                "driver_name",
                "driver_number",
                "constructor_id",
                "constructor_name",
                "qualifying_position",
                *WEEKEND_FEATURE_COLUMNS,
            ]
        )
    for column in WEEKEND_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return (
        coverage,
        merged.merge(
            weekend_context[
                [
                    "season",
                    "round",
                    "driver_id",
                    "driver_name",
                    "driver_number",
                    "constructor_id",
                    "constructor_name",
                    "qualifying_position",
                ]
            ],
            how="left",
            on=["season", "round", "driver_number"],
        ),
    )


def _weekend_confidence_level(feature_frame: pd.DataFrame) -> str:
    """Translate current weekend coverage into a user-facing confidence label."""

    if feature_frame.empty:
        return "LOW"
    has_quali = bool(pd.to_numeric(feature_frame.get("has_quali"), errors="coerce").fillna(0).max() >= 1)
    has_fp1 = bool(pd.to_numeric(feature_frame.get("has_fp1"), errors="coerce").fillna(0).max() >= 1)
    has_fp2 = bool(pd.to_numeric(feature_frame.get("has_fp2"), errors="coerce").fillna(0).max() >= 1)
    has_fp3 = bool(pd.to_numeric(feature_frame.get("has_fp3"), errors="coerce").fillna(0).max() >= 1)
    if has_quali and has_fp1 and has_fp2 and has_fp3:
        return "HIGH"
    if has_quali and (has_fp1 or has_fp2 or has_fp3):
        return "MEDIUM"
    return "LOW"


def _team_degradation_report(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate compound degradation estimates at the constructor level."""

    if feature_frame.empty:
        return pd.DataFrame(columns=["constructor_name", "soft_deg", "medium_deg", "hard_deg"])
    team_deg = (
        feature_frame.groupby("constructor_name", as_index=False)
        .agg(
            soft_deg=("fp_soft_deg_rate", "mean"),
            medium_deg=("fp_medium_deg_rate", "mean"),
            hard_deg=("fp_hard_deg_rate", "mean"),
        )
        .sort_values(["soft_deg", "medium_deg", "hard_deg"], ascending=True)
        .reset_index(drop=True)
    )
    return team_deg


def _upgrade_detection_report(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize team pace deltas versus the last race weekend."""

    if feature_frame.empty:
        return pd.DataFrame(columns=["constructor_name", "pace_delta_vs_last_race"])
    report = (
        feature_frame.groupby("constructor_name", as_index=False)["fp_team_pace_delta_vs_last_race"]
        .median()
        .rename(columns={"fp_team_pace_delta_vs_last_race": "pace_delta_vs_last_race"})
        .sort_values("pace_delta_vs_last_race", ascending=False)
        .reset_index(drop=True)
    )
    return report


def _setup_direction_report(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Infer coarse setup direction from sector-rank balance and speed-trap strength."""

    if feature_frame.empty:
        return pd.DataFrame(columns=["constructor_name", "setup_direction"])
    team = (
        feature_frame.groupby("constructor_name", as_index=False)
        .agg(
            sector_1_rank=("fp_sector_1_rank", "mean"),
            sector_3_rank=("fp_sector_3_rank", "mean"),
            speed_trap=("max_speed_trap", "mean"),
        )
    )
    speed_threshold = float(pd.to_numeric(team["speed_trap"], errors="coerce").median()) if not team.empty else np.nan
    team["setup_direction"] = np.where(
        team["speed_trap"].ge(speed_threshold) & team["sector_3_rank"].gt(team["sector_1_rank"]),
        "Low drag",
        "High downforce",
    )
    return team[["constructor_name", "setup_direction", "sector_1_rank", "sector_3_rank", "speed_trap"]].sort_values("constructor_name")


def _practice_strategy_report(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate early stint strategy tendencies from practice degradation."""

    if feature_frame.empty:
        return pd.DataFrame(columns=["driver_name", "one_stop_viable", "likely_compound"])
    score = (
        -0.40 * np.clip(pd.to_numeric(feature_frame["fp_soft_deg_rate"], errors="coerce").fillna(0.0), -2000.0, 2000.0) / 1000.0
        - 0.25 * np.clip(pd.to_numeric(feature_frame["fp_medium_deg_rate"], errors="coerce").fillna(0.0), -2000.0, 2000.0) / 1000.0
        + 0.10 * np.where(pd.to_numeric(feature_frame["fp_preferred_compound_code"], errors="coerce").fillna(0).eq(2), 1.0, 0.0)
        + 0.12 * np.where(pd.to_numeric(feature_frame["fp_preferred_compound_code"], errors="coerce").fillna(0).eq(1), 1.0, 0.0)
    )
    one_stop_viable = 1.0 / (1.0 + np.exp(-score))
    compound_lookup = {1: "hard", 2: "medium", 3: "soft", 4: "intermediate", 5: "wet"}
    preferred_code = pd.to_numeric(feature_frame["fp_preferred_compound_code"], errors="coerce")
    return pd.DataFrame(
        {
            "driver_name": feature_frame["driver_name"],
            "constructor_name": feature_frame["constructor_name"],
            "one_stop_viable": one_stop_viable,
            "likely_compound": preferred_code.map(compound_lookup).fillna("unknown"),
        }
    ).sort_values("one_stop_viable", ascending=False)


def _fp3_report(laps: pd.DataFrame, stints: pd.DataFrame, season: int) -> dict[str, Any]:
    """Build practice pace, long-run, and degradation analysis from OpenF1 laps."""

    enriched_laps = _attach_driver_identity(laps, season)
    if enriched_laps.empty:
        return {"pace_rankings": [], "long_run_analysis": [], "tyre_deg_estimates": []}

    clean = enriched_laps.loc[
        enriched_laps["lap_duration"].between(60.0, 140.0, inclusive="both")
    ].copy()

    pace = (
        clean.groupby(["driver_id", "driver_name"], as_index=False)
        .agg(
            best_lap_seconds=("lap_duration", "min"),
            median_lap_seconds=("lap_duration", "median"),
            lap_count=("lap_duration", "size"),
        )
        .sort_values(["best_lap_seconds", "median_lap_seconds"])
        .reset_index(drop=True)
    )

    long_run_rows: list[dict[str, Any]] = []
    tyre_deg_rows: list[dict[str, Any]] = []
    if not stints.empty:
        enriched_stints = _attach_driver_identity(stints, season)
        for row in enriched_stints.itertuples(index=False):
            driver_laps = clean.loc[
                (clean["driver_number"] == str(row.driver_number))
                & (clean["lap_number"] >= float(row.lap_start or 0))
                & (clean["lap_number"] <= float(row.lap_end or 0))
            ].sort_values("lap_number")
            if len(driver_laps) < 3:
                continue
            long_run_rows.append(
                {
                    "driver_id": row.driver_id,
                    "driver_name": row.driver_name,
                    "compound": row.compound,
                    "stint_number": int(row.stint_number) if pd.notna(row.stint_number) else np.nan,
                    "avg_lap_seconds": float(driver_laps["lap_duration"].mean()),
                    "laps_in_stint": int(len(driver_laps)),
                }
            )
            x = np.arange(len(driver_laps), dtype=float)
            y = driver_laps["lap_duration"].to_numpy(dtype=float)
            slope = float(np.polyfit(x, y, 1)[0]) if len(driver_laps) >= 3 else np.nan
            tyre_deg_rows.append(
                {
                    "driver_id": row.driver_id,
                    "driver_name": row.driver_name,
                    "compound": row.compound,
                    "tyre_deg_seconds_per_lap": slope,
                }
            )

    long_run = pd.DataFrame(long_run_rows)
    tyre_deg = pd.DataFrame(tyre_deg_rows)
    if not long_run.empty:
        long_run = long_run.sort_values("avg_lap_seconds").reset_index(drop=True)
    if not tyre_deg.empty:
        tyre_deg = tyre_deg.sort_values("tyre_deg_seconds_per_lap").reset_index(drop=True)

    return {
        "pace_rankings": pace.to_dict(orient="records"),
        "long_run_analysis": long_run.to_dict(orient="records"),
        "tyre_deg_estimates": tyre_deg.to_dict(orient="records"),
    }


def _fetch_quali_order(season: int, round_number: int, client: ErgastClient, openf1_client: OpenF1Client) -> pd.DataFrame:
    """Resolve the qualifying order from Jolpica or OpenF1 if needed."""

    race_results_path = CONFIG["processed_dir"] / "race_results.csv"
    if race_results_path.exists():
        race_results = load_dataframe(race_results_path)
        cached = race_results.loc[
            (race_results["season"] == int(season)) & (race_results["round"] == int(round_number))
        ].copy()
        if not cached.empty and "qualifying_position" in cached.columns:
            if "driver_name" not in cached.columns:
                if {"driver_given_name", "driver_family_name"}.issubset(cached.columns):
                    cached["driver_name"] = (
                        cached["driver_given_name"].fillna("").astype(str).str.strip()
                        + " "
                        + cached["driver_family_name"].fillna("").astype(str).str.strip()
                    ).str.strip()
                else:
                    cached["driver_name"] = cached["driver_id"].map(get_driver_name)
            cached["qualifying_position"] = pd.to_numeric(cached["qualifying_position"], errors="coerce")
            cached = cached.sort_values("qualifying_position")
            return cached[["driver_id", "driver_name", "qualifying_position"]].drop_duplicates("driver_id").reset_index(drop=True)

    try:
        payload = client.fetch_paginated(f"{season}/{round_number}/qualifying.json", "QualifyingResults", limit=100)
        qualifying = normalize_qualifying_payload(payload, season)
        qualifying = qualifying.loc[qualifying["round"] == int(round_number)].copy()
        if not qualifying.empty:
            qualifying["qualifying_position"] = pd.to_numeric(qualifying["qualifying_position"], errors="coerce")
            return qualifying.sort_values("qualifying_position").reset_index(drop=True)
    except Exception as exc:
        LOGGER.warning("Jolpica qualifying fetch failed for %s round %s: %s", season, round_number, exc)

    meeting_sessions = _meeting_sessions_for_round(openf1_client, season, round_number)
    session_key = _resolve_session_key(meeting_sessions, "quali")
    if session_key is None:
        return pd.DataFrame(columns=["driver_id", "driver_name", "qualifying_position"])

    try:
        laps = _attach_driver_identity(_fetch_openf1_laps(openf1_client, session_key), season)
    except Exception as exc:
        LOGGER.warning("OpenF1 qualifying fallback failed for %s round %s: %s", season, round_number, exc)
        return pd.DataFrame(columns=["driver_id", "driver_name", "qualifying_position"])
    if laps.empty:
        return pd.DataFrame(columns=["driver_id", "driver_name", "qualifying_position"])

    ranking = (
        laps.loc[laps["lap_duration"].between(60.0, 140.0, inclusive="both")]
        .groupby(["driver_id", "driver_name"], as_index=False)["lap_duration"]
        .min()
        .sort_values("lap_duration")
        .reset_index(drop=True)
    )
    ranking["qualifying_position"] = np.arange(1, len(ranking) + 1)
    return ranking


def _strategy_predictions(inference_frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate one-stop versus two-stop tendencies per driver."""

    frame = inference_frame.copy()
    two_stop_score = (
        0.38 * (frame["circuit_tyre_deg_severity"] / 5.0)
        + 0.16 * np.clip(frame["race_temperature_c"].fillna(frame["race_temperature_c"].median()) / 35.0, 0.0, 1.2)
        + 0.12 * frame["circuit_sc_probability"]
        + 0.10 * frame["is_street_circuit"]
        + 0.08 * (1.0 - frame["team_reliability_last_10"])
        + 0.08 * np.clip(frame["qualifying_position"] / max(len(frame), 1), 0.0, 1.0)
        - 0.06 * np.clip(frame["team_pit_stop_avg"].fillna(frame["team_pit_stop_avg"].median()) / 3500.0, 0.0, 1.0)
    )
    if "fp_soft_deg_rate" in frame.columns:
        two_stop_score += 0.12 * np.clip(pd.to_numeric(frame["fp_soft_deg_rate"], errors="coerce").fillna(0.0), -2000.0, 2000.0) / 1000.0
    if "tyre_used_in_q3_code" in frame.columns:
        two_stop_score += np.where(pd.to_numeric(frame["tyre_used_in_q3_code"], errors="coerce").fillna(0).eq(3), 0.10, -0.02)
    if "weather_mode" in frame.columns:
        two_stop_score += np.where(frame["weather_mode"].eq("wet"), -0.08, 0.0)

    two_stop_probability = 1.0 / (1.0 + np.exp(-(two_stop_score - 0.45) * 4.2))
    one_stop_probability = 1.0 - two_stop_probability
    undercut_score = (
        0.30 * frame["circuit_sc_probability"]
        + 0.25 * (frame["circuit_tyre_deg_severity"] / 5.0)
        + 0.15 * np.clip(pd.to_numeric(frame.get("quali_gap_to_pole_ms"), errors="coerce").fillna(0.0) / 1000.0, 0.0, 2.0)
        + 0.10 * np.where(pd.to_numeric(frame.get("tyre_used_in_q3_code"), errors="coerce").fillna(0).eq(3), 1.0, 0.0)
        + 0.10 * np.clip(pd.to_numeric(frame.get("fp_soft_deg_rate"), errors="coerce").fillna(0.0), -2000.0, 2000.0) / 1000.0
    )
    undercut_probability = 1.0 / (1.0 + np.exp(-(undercut_score - 0.35) * 4.0))
    compound_lookup = {
        1: "hard",
        2: "medium",
        3: "soft",
        4: "intermediate",
        5: "wet",
    }
    preferred_compound_code = pd.to_numeric(frame.get("fp_preferred_compound_code"), errors="coerce").fillna(
        pd.to_numeric(frame.get("tyre_used_in_q3_code"), errors="coerce")
    )
    return pd.DataFrame(
        {
            "driver_id": frame["driver_id"],
            "driver_name": frame["driver_name"],
            "one_stop_probability": one_stop_probability,
            "two_stop_probability": two_stop_probability,
            "undercut_probability": undercut_probability,
            "preferred_strategy_compound": preferred_compound_code.map(compound_lookup).fillna("unknown"),
        }
    ).sort_values("two_stop_probability", ascending=False)


def _simulation_payload(
    metadata: dict[str, Any],
    simulation: Any,
    strategy: pd.DataFrame,
    weather_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON-serializable prediction payload."""

    position_columns = [column for column in simulation.position_distributions.columns if column.startswith("P")]
    expected_finish_frame = simulation.position_distributions.copy()
    if "driver_id" not in expected_finish_frame.columns and "index" in expected_finish_frame.columns:
        expected_finish_frame = expected_finish_frame.rename(columns={"index": "driver_id"})
    expected_finish_frame = expected_finish_frame[["driver_id", "driver_name", *position_columns]].copy()
    expected_finish_frame["expected_finish"] = 0.0
    for column in position_columns:
        position_number = int(column.removeprefix("P"))
        expected_finish_frame["expected_finish"] += (
            pd.to_numeric(expected_finish_frame[column], errors="coerce").fillna(0.0) * position_number
        )

    summary = (
        simulation.win_probs[
            [
                "driver_id",
                "driver_name",
                "constructor_id",
                "constructor_name",
                "relative_win_probability",
                "relative_win_probability_p10",
                "relative_win_probability_p90",
            ]
        ]
        .merge(
            expected_finish_frame[["driver_id", "expected_finish"]],
            how="left",
            on="driver_id",
        )
        .merge(
            simulation.podium_probs[
                ["driver_id", "podium_probability", "podium_probability_p10", "podium_probability_p90"]
            ],
            how="left",
            on="driver_id",
        )
        .merge(
            simulation.expected_points[
                ["driver_id", "expected_points", "expected_points_p10", "expected_points_p90"]
            ],
            how="left",
            on="driver_id",
        )
        .merge(strategy, how="left", on=["driver_id", "driver_name"])
        .sort_values("relative_win_probability", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "season": metadata["season"],
        "round": metadata["round"],
        "race_name": metadata["race_name"],
        "circuit_id": metadata["circuit_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weather_context": weather_context,
        "regulation_warning": simulation.regulation_warning,
        "season_uncertainty": simulation.season_uncertainty,
        "prediction_health_status": simulation.prediction_health.status if simulation.prediction_health else None,
        "prediction_health_warning": simulation.prediction_health.warning if simulation.prediction_health else None,
        "prediction_health_entropy": simulation.prediction_health.entropy if simulation.prediction_health else None,
        "prediction_health_top_driver_probability": (
            simulation.prediction_health.top_driver_probability if simulation.prediction_health else None
        ),
        "prediction_floor_applied": simulation.prediction_health.floor_applied if simulation.prediction_health else False,
        "predictions": summary.to_dict(orient="records"),
        "position_distributions": simulation.position_distributions.to_dict(orient="records"),
        "strategy": strategy.to_dict(orient="records"),
    }


def _practice_stage_payload(
    metadata: dict[str, Any],
    weekend_features: pd.DataFrame,
    weather_context: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Build a JSON-serializable practice-stage report."""

    long_run = (
        weekend_features[
            ["driver_name", "constructor_name", "fp_long_run_pace", "fp_long_run_rank", "fp_preferred_compound_code"]
        ]
        .sort_values(["fp_long_run_rank", "fp_long_run_pace"])
        .reset_index(drop=True)
    ) if not weekend_features.empty else pd.DataFrame()
    tyre_deg = _team_degradation_report(weekend_features)
    upgrades = _upgrade_detection_report(weekend_features)
    setup = _setup_direction_report(weekend_features)
    strategy = _practice_strategy_report(weekend_features)

    return {
        "season": metadata["season"],
        "round": metadata["round"],
        "race_name": metadata["race_name"],
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weather_context": weather_context,
        "long_run_pace": long_run.to_dict(orient="records"),
        "team_tyre_deg": tyre_deg.to_dict(orient="records"),
        "upgrade_detection": upgrades.to_dict(orient="records"),
        "setup_direction": setup.to_dict(orient="records"),
        "early_strategy": strategy.to_dict(orient="records"),
    }


def _augmented_fastf1_feature_store(weekend_features: pd.DataFrame) -> pd.DataFrame:
    """Combine persisted weekend features with the current-round snapshot."""

    historical_path = CONFIG["processed_dir"] / "fastf1_features.csv"
    if historical_path.exists():
        historical = load_dataframe(historical_path)
    else:
        historical = pd.DataFrame(columns=["season", "round", "driver_number", *WEEKEND_FEATURE_COLUMNS])
    keep_columns = ["season", "round", "driver_number", *WEEKEND_FEATURE_COLUMNS]
    current = weekend_features.copy()
    for column in keep_columns:
        if column not in historical.columns:
            historical[column] = np.nan
        if column not in current.columns:
            current[column] = np.nan
    combined = pd.concat([historical[keep_columns], current[keep_columns]], ignore_index=True)
    combined["driver_number"] = combined["driver_number"].apply(_normalize_driver_number)
    return combined.drop_duplicates(subset=["season", "round", "driver_number"], keep="last").reset_index(drop=True)


def run_fp2_mode(season: int, round_number: int, dry_run: bool = False) -> None:
    """Run Friday evening analysis after FP2 using all available practice data."""

    if dry_run:
        LOGGER.info("DRY RUN: using existing pipeline artifacts for FP2 mode")
    else:
        ensure_full_pipeline(skip_fastf1=True)
    metadata = _race_metadata(season, round_number)
    weather_context = _weather_forecast(metadata["circuit_id"], metadata["date"])
    _, weekend_features = _weekend_feature_snapshot(season, round_number)
    payload = _practice_stage_payload(metadata, weekend_features, weather_context, stage="FP2")

    output_path = CONFIG["predictions_dir"] / f"{season}_{slugify(metadata['race_name'])}_fp2.json"
    if dry_run:
        LOGGER.info("DRY RUN: skipping FP2 analysis write to %s", output_path)
    else:
        save_json(payload, output_path)
        LOGGER.info("FP2 analysis saved to %s", output_path)
    if payload["long_run_pace"]:
        LOGGER.info("Long run pace rankings:\n%s", pd.DataFrame(payload["long_run_pace"]).head(10).to_string(index=False))
    if payload["team_tyre_deg"]:
        LOGGER.info("Tyre degradation by team:\n%s", pd.DataFrame(payload["team_tyre_deg"]).to_string(index=False))
    if payload["upgrade_detection"]:
        LOGGER.info("Upgrade detection:\n%s", pd.DataFrame(payload["upgrade_detection"]).head(10).to_string(index=False))
    if payload["early_strategy"]:
        LOGGER.info("Early strategy outlook:\n%s", pd.DataFrame(payload["early_strategy"]).head(10).to_string(index=False))
    LOGGER.info(
        "Sunday forecast | temp=%.1fC wind=%.1f km/h weather=%s",
        weather_context["race_temperature_c"],
        weather_context["race_max_wind_kmh"],
        weather_context["weather_mode"],
    )


def run_fp3_mode(season: int, round_number: int, dry_run: bool = False) -> None:
    """Run Saturday morning analysis after FP3 using the full practice set."""

    if dry_run:
        LOGGER.info("DRY RUN: using existing pipeline artifacts for FP3 mode")
    else:
        ensure_full_pipeline(skip_fastf1=True)
    metadata = _race_metadata(season, round_number)
    weather_context = _weather_forecast(metadata["circuit_id"], metadata["date"])
    _, weekend_features = _weekend_feature_snapshot(season, round_number)
    payload = _practice_stage_payload(metadata, weekend_features, weather_context, stage="FP3")

    output_path = CONFIG["predictions_dir"] / f"{season}_{slugify(metadata['race_name'])}_fp3.json"
    if dry_run:
        LOGGER.info("DRY RUN: skipping FP3 analysis write to %s", output_path)
    else:
        save_json(payload, output_path)
        LOGGER.info("FP3 analysis saved to %s", output_path)
    if payload["long_run_pace"]:
        LOGGER.info("Updated pace rankings:\n%s", pd.DataFrame(payload["long_run_pace"]).head(10).to_string(index=False))
    if payload["team_tyre_deg"]:
        LOGGER.info("Refined tyre degradation:\n%s", pd.DataFrame(payload["team_tyre_deg"]).to_string(index=False))
    if payload["setup_direction"]:
        LOGGER.info("Setup direction:\n%s", pd.DataFrame(payload["setup_direction"]).to_string(index=False))
    if payload["early_strategy"]:
        LOGGER.info("Strategy outlook:\n%s", pd.DataFrame(payload["early_strategy"]).head(10).to_string(index=False))


def run_quali_mode(season: int, round_number: int, simulations: int, dry_run: bool = False) -> None:
    """Run the pre-race prediction workflow after qualifying."""

    if dry_run:
        LOGGER.info("DRY RUN: using existing pipeline artifacts for QUALI mode")
    else:
        ensure_full_pipeline(skip_fastf1=True)
    metadata = _race_metadata(season, round_number)
    jolpica_client = ErgastClient()
    openf1_client = OpenF1Client()
    qualifying = _fetch_quali_order(season, round_number, jolpica_client, openf1_client)
    if qualifying.empty:
        LOGGER.warning("No live qualifying data found for %s; using a provisional ELO-based grid", format_race_label(season, round_number))
        qualifying = _provisional_qualifying_from_roster(season, round_number)
    if qualifying.empty:
        raise RuntimeError(f"No qualifying order or provisional roster available for {format_race_label(season, round_number)}")

    driver_ids = qualifying["driver_id"].tolist()
    qualifying_map = {
        row.driver_id: int(row.qualifying_position)
        for row in qualifying[["driver_id", "qualifying_position"]].itertuples(index=False)
    }
    weather_context = _weather_forecast(metadata["circuit_id"], metadata["date"])
    _, weekend_features = _weekend_feature_snapshot(season, round_number, qualifying=qualifying)
    inference_fastf1_features = _augmented_fastf1_feature_store(weekend_features)
    simulation = simulate_race(
        RaceConfig(
            drivers=driver_ids,
            qualifying_positions=qualifying_map,
            circuit_id=metadata["circuit_id"],
            season=season,
            round=round_number,
            weather_mode=str(weather_context["weather_mode"]),
            n_simulations=simulations,
        )
    )
    inference_frame = build_race_inference_frame(
        drivers_list=driver_ids,
        qualifying_positions=qualifying_map,
        circuit_id=metadata["circuit_id"],
        season=season,
        round_number=round_number,
        fastf1_features_df=inference_fastf1_features,
        weather_mode=str(weather_context["weather_mode"]),
    )
    inference_frame["weather_mode"] = str(weather_context["weather_mode"])
    strategy = _strategy_predictions(inference_frame)
    payload = _simulation_payload(metadata, simulation, strategy, weather_context)
    confidence_report = CALIBRATION_ENGINE.compute_prediction_confidence(
        inference_frame,
        season=season,
        round_number=round_number,
    )
    upset_probability = compute_upset_probability(
        circuit_id=metadata["circuit_id"],
        season=season,
        round_number=round_number,
        weather_forecast=weather_context,
    )
    lead_driver = str(inference_frame.sort_values("qualifying_position").iloc[0]["driver_id"])
    lead_strategy = TIRE_MODEL.predict_optimal_strategy(
        driver_id=lead_driver,
        circuit_id=metadata["circuit_id"],
        grid_position=int(inference_frame.sort_values("qualifying_position").iloc[0]["qualifying_position"]),
        competitors=inference_frame["driver_id"].tolist(),
        weather_forecast=weather_context,
    )
    payload["confidence"] = confidence_report.confidence_label
    payload["confidence_score"] = confidence_report.confidence_score
    payload["confidence_reasons"] = confidence_report.key_uncertainties
    payload["confidence_recommendation"] = confidence_report.recommendation
    payload["upset_probability"] = upset_probability
    payload["lead_strategy_recommendation"] = {
        "driver_id": lead_strategy.driver_id,
        "recommended_strategy": lead_strategy.recommended_strategy,
        "alternative_strategy": lead_strategy.alternative_strategy,
        "undercut_lap": lead_strategy.undercut_lap,
        "overcut_viable": lead_strategy.overcut_viable,
        "strategy_confidence": lead_strategy.strategy_confidence,
    }
    payload["quali_gaps"] = (
        inference_frame[["driver_name", "constructor_name", "qualifying_position", "quali_gap_to_pole_ms", "tyre_used_in_q3_code"]]
        .sort_values("qualifying_position")
        .to_dict(orient="records")
    )
    path = None
    if dry_run:
        LOGGER.info("DRY RUN: skipping pre-race prediction write for season %s round %s", season, round_number)
    else:
        path = save_pre_race_prediction(season, round_number, payload)

    top_predictions = pd.DataFrame(payload["predictions"]).head(10).copy()
    top_predictions["relative_win_range"] = top_predictions.apply(
        lambda row: (
            f"{row['relative_win_probability']:.1%} "
            f"({row['relative_win_probability_p10']:.1%}-{row['relative_win_probability_p90']:.1%})"
        ),
        axis=1,
    )
    top_predictions["podium_range"] = top_predictions.apply(
        lambda row: f"{row['podium_probability']:.1%} ({row['podium_probability_p10']:.1%}-{row['podium_probability_p90']:.1%})",
        axis=1,
    )

    if path is not None:
        LOGGER.info("Saved pre-race prediction to %s", path)
    LOGGER.info(
        "Pre-race prediction | %s | weather=%s precip=%.1fmm",
        metadata["race_name"],
        weather_context["weather_mode"],
        weather_context["race_precipitation_mm"],
    )
    LOGGER.info(
        "Top predictions:\n%s",
        top_predictions[
            [
                "driver_name",
                "constructor_name",
                "relative_win_range",
                "podium_range",
                "expected_finish",
                "expected_points",
            ]
        ].to_string(index=False),
    )
    expected_order = pd.DataFrame(payload["predictions"]).sort_values("expected_finish")[["driver_name", "expected_finish"]]
    LOGGER.info("Expected order:\n%s", expected_order.head(10).to_string(index=False))
    LOGGER.info("Strategy outlook:\n%s", strategy.head(10).to_string(index=False))
    LOGGER.info(
        "Prediction confidence: %s (%.2f) | %s",
        payload["confidence"],
        payload["confidence_score"],
        payload["confidence_recommendation"],
    )
    LOGGER.info("Confidence reasons: %s", "; ".join(payload["confidence_reasons"]) if payload["confidence_reasons"] else "none")
    LOGGER.info("Upset probability: %.2f", payload["upset_probability"])
    if payload.get("prediction_health_status"):
        LOGGER.info(
            "Prediction health: %s | entropy=%.3f | top_driver_prob=%.1f%% | floor_applied=%s",
            payload["prediction_health_status"],
            float(payload.get("prediction_health_entropy") or 0.0),
            100.0 * float(payload.get("prediction_health_top_driver_probability") or 0.0),
            bool(payload.get("prediction_floor_applied")),
        )
    if payload.get("prediction_health_warning"):
        LOGGER.warning("Prediction health warning: %s", payload["prediction_health_warning"])
    LOGGER.info("Lead strategy recommendation: %s", payload["lead_strategy_recommendation"])
    LOGGER.info(
        "Qualifying gap to pole:\n%s",
        inference_frame[["driver_name", "qualifying_position", "quali_gap_to_pole_ms"]]
        .sort_values("qualifying_position")
        .head(10)
        .to_string(index=False),
    )
    LOGGER.info(
        "Undercut outlook:\n%s",
        strategy[["driver_name", "undercut_probability", "preferred_strategy_compound"]].head(10).to_string(index=False),
    )
    if weather_context["weather_mode"] in {"wet", "mixed"}:
        LOGGER.warning("Wet race warning: forecast indicates %s conditions", weather_context["weather_mode"])
    if simulation.prediction_health and simulation.prediction_health.status == "COLLAPSED":
        print("⚠️  COLLAPSED DISTRIBUTION WARNING")
        print("This prediction is based primarily on historical ELO")
        print("Confidence will improve significantly after qualifying")
        print("Do not treat win probabilities as meaningful until")
        print("qualifying data is available")
    if simulation.regulation_warning:
        LOGGER.warning(
            "Regulation uncertainty warning: %s confidence multiplier=%.2f",
            format_race_label(season, round_number),
            simulation.season_uncertainty,
        )


def run_live_mode(season: int, round_number: int, simulations: int, polls: int, interval_seconds: int) -> None:
    """Poll the live race feed and refresh position probabilities."""

    ensure_full_pipeline(skip_fastf1=True)
    metadata = _race_metadata(season, round_number)
    client = OpenF1Client()
    meeting_sessions = _meeting_sessions_for_round(client, season, round_number)
    session_key = _resolve_session_key(meeting_sessions, "live")
    if session_key is None:
        raise RuntimeError(f"No race session found for {format_race_label(season, round_number)}")

    for poll_index in range(max(1, polls)):
        laps = _attach_driver_identity(_fetch_openf1_laps(client, session_key), season)
        pits = _attach_driver_identity(fetch_pit_stops_openf1(client, session_key), season)
        if laps.empty:
            LOGGER.info("No live lap data yet for %s", metadata["race_name"])
        else:
            latest = (
                laps.sort_values(["lap_number", "position"])
                .groupby("driver_id", as_index=False)
                .tail(1)
                .sort_values(["lap_number", "position", "lap_duration"])
                .reset_index(drop=True)
            )
            current_order = latest["driver_id"].tolist()
            qualifying_map = {
                row.driver_id: int(index + 1)
                for index, row in enumerate(latest.itertuples(index=False))
            }
            simulation = simulate_race(
                RaceConfig(
                    drivers=current_order,
                    qualifying_positions=qualifying_map,
                    circuit_id=metadata["circuit_id"],
                    season=season,
                    round=round_number,
                    n_simulations=simulations,
                )
            )
            live_summary = simulation.win_probs.head(10)[
                ["driver_name", "relative_win_probability", "relative_win_probability_p10", "relative_win_probability_p90"]
            ]
            LOGGER.info(
                "Live poll %s/%s | lap=%s | pit_stops=%s\n%s",
                poll_index + 1,
                polls,
                int(pd.to_numeric(latest["lap_number"], errors="coerce").max()),
                len(pits),
                live_summary.to_string(index=False),
            )
        if poll_index + 1 < polls:
            time.sleep(interval_seconds)


def run_post_mode(season: int, round_number: int, skip_fastf1: bool = False, dry_run: bool = False) -> None:
    """Refresh the pipeline after the race and score the saved pre-race prediction."""

    if dry_run:
        LOGGER.info("DRY RUN: skipping auto-update and pipeline writes for season %s round %s", season, round_number)
        update_result = {"updated": False, "dry_run": True, "season": season, "round": round_number}
    else:
        update_result = auto_update(skip_fastf1=skip_fastf1)
        ensure_full_pipeline(skip_fastf1=skip_fastf1)
    race_results = load_dataframe(CONFIG["processed_dir"] / "race_results.csv")
    actual = race_results.loc[(race_results["season"] == season) & (race_results["round"] == round_number)].copy()
    if actual.empty:
        raise RuntimeError(f"No final results available for season {season} round {round_number}")

    prediction_payload = load_pre_race_prediction(season, round_number)
    if prediction_payload is None:
        LOGGER.warning("No saved pre-race prediction found for season %s round %s", season, round_number)
        LOGGER.info("Post-race update result:\n%s", update_result)
        return

    scoring = score_prediction(prediction_payload, actual, persist=not dry_run)
    LOGGER.info("Prediction score:\n%s", scoring)

    fastf1_features = load_dataframe(CONFIG["processed_dir"] / "fastf1_features.csv")
    round_fastf1 = fastf1_features.loc[
        (fastf1_features["season"] == int(season)) & (fastf1_features["round"] == int(round_number))
    ].copy()
    evidence_frame = build_constructor_event_evidence_frame(actual, round_fastf1)
    if evidence_frame.empty:
        LOGGER.info("No architecture evidence - skipping update for season %s round %s", season, round_number)
        LOGGER.info("Post-race update result:\n%s", update_result)
        LOGGER.info("%s", build_race_accuracy_report(season, round_number))
        return

    if dry_run:
        LOGGER.info("DRY RUN: architecture updates skipped for season %s round %s", season, round_number)
        LOGGER.info("Post-race update result:\n%s", update_result)
        LOGGER.info("%s", build_race_accuracy_report(season, round_number))
        return

    update_rows: list[dict[str, Any]] = []
    target_round = int(round_number) + 1
    for constructor_id in sorted(evidence_frame["constructor_id"].astype(str).unique().tolist()):
        before = get_car_architecture(constructor_id, season, target_round - 1)
        after = update_car_architecture_from_race(
            constructor_id=constructor_id,
            season=int(season),
            round_number=target_round,
            race_data=evidence_frame.loc[evidence_frame["constructor_id"] == constructor_id].copy(),
        )
        update_rows.append(
            {
                "constructor_id": constructor_id,
                "straight_line_speed_before": before.straight_line_speed,
                "straight_line_speed_after": after.straight_line_speed,
                "slow_corner_grip_before": before.slow_corner_grip,
                "slow_corner_grip_after": after.slow_corner_grip,
                "high_speed_cornering_before": before.high_speed_cornering,
                "high_speed_cornering_after": after.high_speed_cornering,
                "mechanical_grip_before": before.mechanical_grip,
                "mechanical_grip_after": after.mechanical_grip,
                "tyre_management_before": before.tyre_management,
                "tyre_management_after": after.tyre_management,
                "downforce_level_before": before.downforce_level,
                "downforce_level_after": after.downforce_level,
                "confidence_after": after.confidence,
            }
        )

    if update_rows:
        update_summary = pd.DataFrame(update_rows).sort_values("constructor_id").reset_index(drop=True)
        LOGGER.info("Architecture update summary for target round %s:\n%s", target_round, update_summary.to_string(index=False))

    LOGGER.info("Post-race update result:\n%s", update_result)
    LOGGER.info("%s", build_race_accuracy_report(season, round_number))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for race-weekend operations."""

    parser = argparse.ArgumentParser(description="Operational race-weekend commands for the F1 AI engine")
    parser.add_argument("--session", required=True, choices=["FP2", "FP3", "QUALI", "LIVE", "POST"])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--polls", type=int, default=3, help="Number of live polling cycles")
    parser.add_argument("--interval-seconds", type=int, default=CONFIG["live_poll_seconds"])
    parser.add_argument("--skip-fastf1", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Dispatch a race-weekend session workflow."""

    args = parse_args()
    session = args.session.upper()
    if session == "FP2":
        run_fp2_mode(args.season, args.round, dry_run=args.dry_run)
    elif session == "FP3":
        run_fp3_mode(args.season, args.round, dry_run=args.dry_run)
    elif session == "QUALI":
        run_quali_mode(args.season, args.round, args.simulations, dry_run=args.dry_run)
    elif session == "LIVE":
        run_live_mode(args.season, args.round, args.simulations, args.polls, args.interval_seconds)
    else:
        run_post_mode(args.season, args.round, skip_fastf1=args.skip_fastf1, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
