"""Live race data fetcher for the dashboard — wraps OpenF1 API."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import OpenF1Client, fetch_sessions_openf1, fetch_stints, fetch_pit_stops_openf1  # noqa: E402
from src.utils import CONFIG as BASE_CONFIG, load_dataframe, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

PROCESSED = BASE_CONFIG["processed_dir"]


@dataclass
class LiveState:
    """Snapshot of current race state from OpenF1."""

    session_key: int | None
    current_lap: int
    total_laps: int
    positions: pd.DataFrame      # driver_id, driver_name, constructor_id, position, gap_to_leader, last_lap_time, sector_1, sector_2, sector_3
    stints: pd.DataFrame          # driver_id, stint_number, compound, lap_start, lap_end, tyre_age
    pit_stops: pd.DataFrame       # driver_id, lap_number, pit_duration
    weather: dict[str, Any]       # air_temp, track_temp, humidity, rainfall, wind_speed, wind_direction
    is_live: bool                 # whether a session is actually in progress
    error: str | None = None


def _driver_lookup() -> pd.DataFrame:
    """Build driver_number -> driver_id/name mapping from race results."""
    rr_path = PROCESSED / "race_results.csv"
    if not rr_path.exists():
        return pd.DataFrame(columns=["driver_number", "driver_id", "driver_name", "constructor_id", "constructor_name"])
    rr = load_dataframe(rr_path)
    if "driver_name" not in rr.columns:
        rr["driver_name"] = (
            rr["driver_given_name"].fillna("").astype(str).str.strip()
            + " "
            + rr["driver_family_name"].fillna("").astype(str).str.strip()
        ).str.strip()
    mapping = (
        rr[["driver_number", "driver_id", "driver_name", "constructor_id", "constructor_name"]]
        .dropna(subset=["driver_id"])
        .copy()
    )
    mapping["driver_number"] = mapping["driver_number"].astype(str)
    return mapping.drop_duplicates(subset=["driver_number", "driver_id"], keep="last")


def _attach_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach driver identity columns to OpenF1 data keyed by driver_number."""
    if frame.empty or "driver_number" not in frame.columns:
        return frame
    lookup = _driver_lookup()
    enriched = frame.copy()
    enriched["driver_number"] = enriched["driver_number"].astype(str)
    enriched = enriched.merge(
        lookup[["driver_number", "driver_id", "driver_name", "constructor_id", "constructor_name"]],
        how="left", on="driver_number",
    )
    return enriched


def find_session_key(client: OpenF1Client, season: int, round_number: int, session_type: str = "Race") -> int | None:
    """Find the OpenF1 session_key for a given race weekend session."""
    try:
        sessions = fetch_sessions_openf1(client, season)
    except Exception as exc:
        LOGGER.debug("Non-critical error fetching sessions for session key lookup: %s", exc)
        # Try cached
        for p in [PROCESSED / "openf1_sessions.csv", BASE_CONFIG["raw_dir"] / f"openf1_sessions_{season}.csv"]:
            if p.exists():
                sessions = load_dataframe(p)
                break
        else:
            return None

    if sessions.empty:
        return None

    sessions["date_start"] = pd.to_datetime(sessions["date_start"], errors="coerce", utc=True)

    # Build meeting order to map to round number
    meetings = (
        sessions.groupby(sessions["meeting_name"].fillna("") + "|" + sessions["location"].fillna(""), as_index=False)
        .agg(meeting_start=("date_start", "min"))
        .sort_values("meeting_start")
        .reset_index(drop=True)
    )
    meetings["round_est"] = range(1, len(meetings) + 1)

    if round_number > len(meetings):
        return None

    meeting_key = meetings.iloc[round_number - 1].name
    meeting_id = meetings.index[meeting_key] if isinstance(meeting_key, int) else meeting_key

    # Match meeting
    sessions["meeting_key"] = sessions["meeting_name"].fillna("") + "|" + sessions["location"].fillna("")
    target_meeting = meetings.iloc[round_number - 1]
    target_key = str(target_meeting.iloc[0]) if hasattr(target_meeting, "iloc") else str(target_meeting)

    # Find the right meeting's sessions
    meeting_sessions = sessions.loc[
        sessions["meeting_key"] == sessions.loc[sessions["date_start"] == meetings.iloc[round_number - 1]["meeting_start"], "meeting_key"].iloc[0]
    ] if not sessions.empty else pd.DataFrame()

    if meeting_sessions.empty:
        # Fallback: just try to find by session type across all sessions
        meeting_sessions = sessions

    # Find the race/quali/etc session
    type_lower = session_type.lower()
    aliases = {"race": ["race"], "qualifying": ["qualifying"], "sprint": ["sprint"]}
    search_terms = aliases.get(type_lower, [type_lower])

    for term in search_terms:
        matched = meeting_sessions.loc[
            meeting_sessions["session_name"].astype(str).str.lower().str.contains(term, na=False)
        ]
        if not matched.empty:
            return int(matched.sort_values("date_start").iloc[-1]["session_key"])

    return None


def fetch_live_positions(client: OpenF1Client, session_key: int) -> pd.DataFrame:
    """Fetch current position data from OpenF1."""
    payload = client.fetch("/position", params={"session_key": session_key})
    if not payload:
        return pd.DataFrame()

    df = pd.DataFrame(payload)
    if df.empty:
        return df

    df["driver_number"] = df["driver_number"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep only latest position per driver
    latest = df.sort_values("date").groupby("driver_number", as_index=False).tail(1)
    latest = latest.sort_values("position").reset_index(drop=True)
    return _attach_identity(latest)


def fetch_live_laps(client: OpenF1Client, session_key: int) -> pd.DataFrame:
    """Fetch lap timing data from OpenF1."""
    payload = client.fetch("/laps", params={"session_key": session_key})
    if not payload:
        return pd.DataFrame()

    df = pd.DataFrame(payload)
    if df.empty:
        return df

    df["driver_number"] = df["driver_number"].astype(str)

    # Normalize column names
    col_map = {
        "duration_sector_1": "sector_1", "duration_sector_2": "sector_2", "duration_sector_3": "sector_3",
        "lap_time": "lap_duration", "lap_duration_seconds": "lap_duration", "lap": "lap_number",
    }
    for src, tgt in col_map.items():
        if src in df.columns and tgt not in df.columns:
            df[tgt] = pd.to_numeric(df[src], errors="coerce")

    for c in ["lap_duration", "lap_number", "position", "sector_1", "sector_2", "sector_3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return _attach_identity(df)


def fetch_live_weather(client: OpenF1Client, session_key: int) -> dict[str, Any]:
    """Fetch latest weather data from OpenF1."""
    payload = client.fetch("/weather", params={"session_key": session_key})
    if not payload:
        return {"air_temp": None, "track_temp": None, "humidity": None, "rainfall": False, "wind_speed": None, "wind_direction": None}

    df = pd.DataFrame(payload)
    if df.empty:
        return {"air_temp": None, "track_temp": None, "humidity": None, "rainfall": False, "wind_speed": None, "wind_direction": None}

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        latest = df.sort_values("date").tail(1).iloc[0]
    else:
        latest = df.tail(1).iloc[0]

    return {
        "air_temp": float(latest.get("air_temperature", latest.get("air_temp", 0))) if pd.notna(latest.get("air_temperature", latest.get("air_temp"))) else None,
        "track_temp": float(latest.get("track_temperature", latest.get("track_temp", 0))) if pd.notna(latest.get("track_temperature", latest.get("track_temp"))) else None,
        "humidity": float(latest.get("humidity", 0)) if pd.notna(latest.get("humidity")) else None,
        "rainfall": bool(latest.get("rainfall", False)),
        "wind_speed": float(latest.get("wind_speed", 0)) if pd.notna(latest.get("wind_speed")) else None,
        "wind_direction": float(latest.get("wind_direction", 0)) if pd.notna(latest.get("wind_direction")) else None,
    }


def fetch_live_state(season: int, round_number: int) -> LiveState:
    """Fetch complete live race state from OpenF1 API.

    Returns a LiveState snapshot with positions, stints, pit stops, and weather.
    Works during live sessions and also for recently completed sessions.
    """
    client = OpenF1Client()

    session_key = find_session_key(client, season, round_number, "Race")
    if session_key is None:
        return LiveState(
            session_key=None, current_lap=0, total_laps=0,
            positions=pd.DataFrame(), stints=pd.DataFrame(), pit_stops=pd.DataFrame(),
            weather={}, is_live=False, error="No race session found for this round. The session may not have started yet."
        )

    try:
        # Positions
        positions = fetch_live_positions(client, session_key)

        # Laps (for timing data)
        laps = fetch_live_laps(client, session_key)
        current_lap = int(laps["lap_number"].max()) if not laps.empty and "lap_number" in laps.columns else 0

        # Merge latest lap time into positions
        if not laps.empty and not positions.empty and "driver_number" in laps.columns:
            latest_laps = laps.sort_values("lap_number").groupby("driver_number", as_index=False).tail(1)
            timing_cols = [c for c in ["driver_number", "lap_duration", "sector_1", "sector_2", "sector_3", "st_speed"] if c in latest_laps.columns]
            if len(timing_cols) > 1:
                positions = positions.merge(latest_laps[timing_cols], how="left", on="driver_number")

        # Stints
        try:
            stints_raw = fetch_stints(client, session_key)
            stints = _attach_identity(stints_raw) if not stints_raw.empty else pd.DataFrame()
        except Exception as exc:
            LOGGER.debug("Non-critical error fetching stints: %s", exc)
            stints = pd.DataFrame()

        # Pit stops
        try:
            pits_raw = fetch_pit_stops_openf1(client, session_key)
            pit_stops = _attach_identity(pits_raw) if not pits_raw.empty else pd.DataFrame()
        except Exception as exc:
            LOGGER.debug("Non-critical error fetching pit stops: %s", exc)
            pit_stops = pd.DataFrame()

        # Weather
        weather = fetch_live_weather(client, session_key)

        is_live = current_lap > 0  # If we have lap data, something is happening

        return LiveState(
            session_key=session_key,
            current_lap=current_lap,
            total_laps=0,  # OpenF1 doesn't directly provide this
            positions=positions,
            stints=stints,
            pit_stops=pit_stops,
            weather=weather,
            is_live=is_live,
        )

    except Exception as e:
        LOGGER.error("Failed to fetch live state: %s", e)
        return LiveState(
            session_key=session_key, current_lap=0, total_laps=0,
            positions=pd.DataFrame(), stints=pd.DataFrame(), pit_stops=pd.DataFrame(),
            weather={}, is_live=False, error=str(e),
        )
