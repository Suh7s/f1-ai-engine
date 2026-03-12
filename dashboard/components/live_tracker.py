"""Live and replay session merger for the cinematic track-map experience."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.circuit_maps import CIRCUIT_REGISTRY, circuit_coordinates, resolve_circuit_id
from src.data_pipeline import OpenF1Client, fetch_sessions_openf1
from src.data_universe import _match_sessions_to_rounds  # noqa: PLC2701
from src.driver_database import DRIVER_DATABASE, get_driver_profile
from src.team_database import get_team_profile
from src.utils import CONFIG as BASE_CONFIG, get_logger, load_calendar_frame, load_dataframe

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Current merged position/timing snapshot for one session."""

    timestamp: datetime
    positions: pd.DataFrame
    track_status: str
    safety_car_deployed: bool
    session_key: int | None
    session_name: str
    circuit_id: str
    messages: pd.DataFrame


def _session_group(session_name: str) -> str:
    label = str(session_name or "").strip().lower()
    if "qual" in label:
        return "QUALI"
    if "sprint" in label and "qual" not in label:
        return "SPRINT"
    if "race" in label:
        return "RACE"
    return "PRACTICE"


def _safe_fetch(client: OpenF1Client, endpoint: str, params: dict[str, Any]) -> pd.DataFrame:
    try:
        payload = client.fetch(f"/{endpoint}", params=params)
    except Exception as exc:
        LOGGER.debug("Non-critical error fetching OpenF1 %s: %s", endpoint, exc)
        return pd.DataFrame()
    frame = pd.DataFrame(payload)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    return frame


@lru_cache(maxsize=1)
def _driver_lookup() -> pd.DataFrame:
    frame = load_dataframe(BASE_CONFIG["processed_dir"] / "race_results.csv") if (BASE_CONFIG["processed_dir"] / "race_results.csv").exists() else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["driver_number", "driver_id", "constructor_id"])
    sort_columns = [column for column in ["season", "round"] if column in frame.columns]
    working = frame.sort_values(sort_columns) if sort_columns else frame.copy()
    working = working[["driver_number", "driver_id", "constructor_id"]].dropna(subset=["driver_id"]).copy()
    working["driver_number"] = working["driver_number"].astype(str)
    return working.drop_duplicates(subset=["driver_number"], keep="last")


@lru_cache(maxsize=256)
def _weekend_driver_lookup(season: int, round_number: int) -> pd.DataFrame:
    frame = load_dataframe(BASE_CONFIG["processed_dir"] / "race_results.csv") if (BASE_CONFIG["processed_dir"] / "race_results.csv").exists() else pd.DataFrame()
    if frame.empty or "driver_number" not in frame.columns:
        return _driver_lookup()
    working = frame.copy()
    working["driver_number"] = working["driver_number"].astype(str)
    weekend = working.loc[(working["season"] == int(season)) & (working["round"] == int(round_number))].copy() if {"season", "round"}.issubset(working.columns) else pd.DataFrame()
    if weekend.empty:
        weekend = working
    columns = [column for column in ["driver_number", "driver_id", "constructor_id", "driver_given_name", "driver_family_name", "finish_position"] if column in weekend.columns]
    weekend = weekend[columns].dropna(subset=["driver_number"]).copy()
    return weekend.drop_duplicates(subset=["driver_number"], keep="last")


@lru_cache(maxsize=256)
def _weekend_finish_lookup_by_driver_id(season: int, round_number: int) -> dict[str, float]:
    frame = load_dataframe(BASE_CONFIG["processed_dir"] / "race_results.csv") if (BASE_CONFIG["processed_dir"] / "race_results.csv").exists() else pd.DataFrame()
    if frame.empty or not {"season", "round", "driver_id", "finish_position"}.issubset(frame.columns):
        return {}
    weekend = frame.loc[(frame["season"] == int(season)) & (frame["round"] == int(round_number))].copy()
    if weekend.empty:
        return {}
    finish_series = pd.to_numeric(weekend["finish_position"], errors="coerce")
    weekend = weekend.assign(finish_position=finish_series).dropna(subset=["driver_id", "finish_position"])
    return weekend.drop_duplicates(subset=["driver_id"], keep="last").set_index("driver_id")["finish_position"].to_dict()


def _load_session_driver_manifest(base: Any) -> pd.DataFrame:
    path = base / "drivers.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = load_dataframe(path)
    if frame.empty or "driver_number" not in frame.columns:
        return pd.DataFrame()
    working = frame.copy()
    working["driver_number"] = working["driver_number"].astype(str)
    manifest = working[["driver_number"]].copy()
    if "broadcast_name" in working.columns and working["broadcast_name"].notna().any():
        manifest["driver_name"] = working["broadcast_name"]
    elif "full_name" in working.columns:
        manifest["driver_name"] = working["full_name"]
    if "name_acronym" in working.columns:
        manifest["driver_abbr"] = working["name_acronym"]
    if "team_colour" in working.columns:
        manifest["team_color"] = working["team_colour"]
    if "team_name" in working.columns:
        manifest["team_name"] = working["team_name"]
    return manifest.drop_duplicates(subset=["driver_number"], keep="last")


def _normalize_identity_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


@lru_cache(maxsize=1)
def _driver_identity_index() -> tuple[dict[str, str], dict[str, str]]:
    abbr_map: dict[str, str | None] = {}
    name_map: dict[str, str | None] = {}
    for driver_id, profile in DRIVER_DATABASE.items():
        abbr = str(profile.abbreviation or "").upper()
        if abbr:
            abbr_map[abbr] = driver_id if abbr not in abbr_map else abbr_map[abbr]
        tokens = {
            _normalize_identity_token(profile.full_name),
            _normalize_identity_token(profile.driver_id),
            _normalize_identity_token(profile.full_name.split()[-1] if profile.full_name else ""),
        }
        for token in {token for token in tokens if token}:
            if token in name_map and name_map[token] != driver_id:
                name_map[token] = ""
            elif token not in name_map:
                name_map[token] = driver_id
    unique_abbr = {key: value for key, value in abbr_map.items() if value}
    unique_name = {key: value for key, value in name_map.items() if value}
    return unique_abbr, unique_name


def _resolve_driver_id_from_identity(driver_name: Any, driver_abbr: Any, fallback_driver_id: Any) -> str:
    fallback = str(fallback_driver_id or "").strip()
    target_abbr = str(driver_abbr or "").strip().upper()
    target_name = _normalize_identity_token(driver_name)
    if fallback:
        profile = get_driver_profile(fallback)
        if target_abbr and profile.abbreviation.upper() == target_abbr:
            return fallback
        if target_name and (
            _normalize_identity_token(profile.full_name) == target_name
            or _normalize_identity_token(profile.full_name.split()[-1] if profile.full_name else "") == target_name
        ):
            return fallback
    abbr_map, name_map = _driver_identity_index()
    if target_abbr and target_abbr in abbr_map:
        return abbr_map[target_abbr]
    if target_name and target_name in name_map:
        return name_map[target_name]
    return fallback


def _driver_numbers(drivers: pd.DataFrame) -> list[str]:
    if drivers.empty or "driver_number" not in drivers.columns:
        return []
    numbers: list[str] = []
    for value in drivers["driver_number"].dropna().tolist():
        try:
            normalized = str(int(float(value)))
        except (TypeError, ValueError):
            normalized = str(value).strip()
        if normalized and normalized not in numbers:
            numbers.append(normalized)
    return numbers


def _safe_fetch_partitioned(
    client: OpenF1Client,
    endpoint: str,
    session_key: int,
    driver_numbers: list[str],
    extra_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    params = {"session_key": session_key, **(extra_params or {})}
    if not driver_numbers:
        return _safe_fetch(client, endpoint, params)
    frames = [_safe_fetch(client, endpoint, {**params, "driver_number": driver_number}) for driver_number in driver_numbers]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _normalize_team_color(value: Any, team_id: Any) -> str:
    if pd.notna(value) and str(value).strip():
        color = str(value).strip()
        return color if color.startswith("#") else f"#{color}"
    return get_team_profile(str(team_id or "")).primary_color


def _enrich_position_frame(frame: pd.DataFrame, season: int, round_number: int, base: Any, circuit_id: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    working = frame.copy()
    if "position" in working.columns and "position_in_race" not in working.columns:
        working = working.rename(columns={"position": "position_in_race"})
    if "driver_number" in working.columns:
        working["driver_number"] = working["driver_number"].astype(str)

    session_drivers = _load_session_driver_manifest(base)
    if not session_drivers.empty:
        working = working.merge(session_drivers, how="left", on="driver_number", suffixes=("", "_session"))

    lookup = _weekend_driver_lookup(int(season), int(round_number))
    if not lookup.empty:
        lookup = lookup.rename(
            columns={
                "driver_id": "lookup_driver_id",
                "constructor_id": "lookup_constructor_id",
                "finish_position": "lookup_finish_position",
            }
        )
        working = working.merge(lookup, how="left", on="driver_number")

    if "constructor_id" not in working.columns:
        working["constructor_id"] = pd.NA
    if "driver_id" not in working.columns:
        working["driver_id"] = pd.NA

    invalid_driver_ids = working["driver_id"].isna() | working["driver_id"].astype(str).str.fullmatch(r"\d+")
    if "lookup_driver_id" in working.columns:
        working.loc[invalid_driver_ids, "driver_id"] = working.loc[invalid_driver_ids, "lookup_driver_id"].fillna(working.loc[invalid_driver_ids, "driver_id"])
    working["driver_id"] = working["driver_id"].fillna(working["driver_number"])
    if "lookup_constructor_id" in working.columns:
        working["constructor_id"] = working["constructor_id"].fillna(working["lookup_constructor_id"])

    def profile_from_row(row: pd.Series) -> Any:
        return get_driver_profile(str(row.get("driver_id") or row.get("driver_number") or ""))

    working["driver_name"] = working.apply(
        lambda row: row["driver_name"]
        if pd.notna(row.get("driver_name")) and str(row.get("driver_name")).strip()
        else (
            f"{str(row.get('driver_given_name', '')).strip()} {str(row.get('driver_family_name', '')).strip()}".strip()
            if pd.notna(row.get("driver_given_name")) or pd.notna(row.get("driver_family_name"))
            else profile_from_row(row).full_name
        ),
        axis=1,
    )
    working["driver_abbr"] = working.apply(
        lambda row: row["driver_abbr"] if pd.notna(row.get("driver_abbr")) and str(row.get("driver_abbr")).strip() else profile_from_row(row).abbreviation,
        axis=1,
    )
    working["driver_id"] = working.apply(
        lambda row: _resolve_driver_id_from_identity(row.get("driver_name"), row.get("driver_abbr"), row.get("driver_id")),
        axis=1,
    )
    working["team_id"] = working["constructor_id"].fillna(working["driver_id"].map(lambda driver_id: get_driver_profile(str(driver_id)).current_team))
    working["team_color"] = working.apply(lambda row: _normalize_team_color(row.get("team_color"), row.get("team_id")), axis=1)
    working["circuit_id"] = circuit_id
    if "gap_to_leader" not in working.columns:
        working["gap_to_leader"] = pd.NA
    if "interval" not in working.columns:
        working["interval"] = pd.NA
    working["gap_to_leader"] = working["gap_to_leader"].fillna("Leader")
    working["interval"] = working["interval"].fillna("-")
    working["position_in_race"] = pd.to_numeric(working.get("position_in_race"), errors="coerce")
    if "lookup_finish_position" in working.columns:
        working["position_in_race"] = working["position_in_race"].fillna(pd.to_numeric(working["lookup_finish_position"], errors="coerce"))
    return working.sort_values(["position_in_race", "driver_abbr"], na_position="last").reset_index(drop=True)


def resolve_session_key(season: int, round_number: int, session_group: str = "RACE") -> tuple[int | None, str, str]:
    """Resolve a session key for a calendar round."""

    base = BASE_CONFIG["raw_dir"] / "openf1" / f"{int(season)}" / f"R{int(round_number):02d}"
    sessions_path = base / "sessions.parquet"
    if sessions_path.exists():
        cached = load_dataframe(sessions_path)
        if not cached.empty and "session_name" in cached.columns:
            weekend = cached.copy()
            if "circuit_id" not in weekend.columns or not weekend["circuit_id"].notna().any():
                calendar = load_calendar_frame(int(season), allow_fetch=False)
                match = calendar.loc[calendar["round"] == int(round_number)] if not calendar.empty else pd.DataFrame()
                if not match.empty:
                    weekend["circuit_id"] = resolve_circuit_id(str(match.iloc[0]["circuit_id"]))
            weekend["session_group"] = weekend["session_name"].map(_session_group)
            matched = weekend.loc[weekend["session_group"] == str(session_group).upper()].sort_values("date_start") if "date_start" in weekend.columns else weekend.loc[weekend["session_group"] == str(session_group).upper()]
            if matched.empty:
                matched = weekend.sort_values("date_start") if "date_start" in weekend.columns else weekend
            if not matched.empty:
                row = matched.iloc[-1]
                return (
                    int(row["session_key"]) if pd.notna(row.get("session_key")) else None,
                    str(row.get("session_name") or ""),
                    str(row.get("circuit_id") or ""),
                )

    client = OpenF1Client()
    try:
        sessions = fetch_sessions_openf1(client, int(season))
    except Exception as exc:
        LOGGER.debug("Non-critical error fetching sessions for round lookup: %s", exc)
        return None, "", ""
    calendar = load_calendar_frame(int(season), allow_fetch=False)
    mapped = _match_sessions_to_rounds(calendar, sessions)
    if mapped.empty:
        return None, "", ""
    weekend = mapped.loc[mapped["round"] == int(round_number)].copy()
    if weekend.empty:
        return None, "", ""
    weekend["session_group"] = weekend["session_name"].map(_session_group)
    matched = weekend.loc[weekend["session_group"] == str(session_group).upper()].sort_values("date_start")
    if matched.empty:
        matched = weekend.sort_values("date_start")
    row = matched.iloc[-1]
    return int(row["session_key"]), str(row["session_name"]), str(row["circuit_id"])


def _latest_by_driver(frame: pd.DataFrame, keep_columns: list[str]) -> pd.DataFrame:
    if frame.empty or "driver_number" not in frame.columns:
        return pd.DataFrame(columns=keep_columns)
    working = frame.copy()
    working["driver_number"] = working["driver_number"].astype(str)
    if "date" in working.columns:
        working = working.sort_values("date")
    latest = working.groupby("driver_number", as_index=False).tail(1).copy()
    for column in keep_columns:
        if column not in latest.columns:
            latest[column] = pd.NA
    return latest[keep_columns]


def _position_status(messages: pd.DataFrame) -> tuple[str, bool]:
    if messages.empty:
        return "GREEN", False

    ordered = messages.copy()
    if "date" in ordered.columns:
        ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce", utc=True)
        ordered = ordered.sort_values("date", ascending=False, na_position="last")
    else:
        ordered = ordered.iloc[::-1]

    def row_text(row: pd.Series) -> str:
        fields = [row.get("category"), row.get("flag"), row.get("message"), row.get("scope"), row.get("status")]
        return " ".join(str(value).strip() for value in fields if pd.notna(value) and str(value).strip()).upper()

    for _, row in ordered.iterrows():
        text = row_text(row)
        if not text:
            continue
        if "RED FLAG" in text or "SESSION SUSPENDED" in text:
            return "RED FLAG", False
        if any(token in text for token in ["VIRTUAL SAFETY CAR ENDING", "VSC ENDING", "GREEN FLAG", "TRACK CLEAR", "ALL CLEAR", "SAFETY CAR IN THIS LAP", "SAFETY CAR ENDING"]):
            return "GREEN", False
        if "VIRTUAL SAFETY CAR" in text or re.search(r"\bVSC\b", text):
            return "VIRTUAL SAFETY CAR", True
        if "SAFETY CAR" in text:
            return "SAFETY CAR", True
        if "YELLOW" in text:
            return "YELLOW", False
    return "GREEN", False


def _merge_live_frames(
    circuit_id: str,
    drivers: pd.DataFrame,
    location: pd.DataFrame,
    position: pd.DataFrame,
    intervals: pd.DataFrame,
    car_data: pd.DataFrame,
    stints: pd.DataFrame,
    laps: pd.DataFrame,
) -> pd.DataFrame:
    location_latest = _latest_by_driver(location, ["driver_number", "x", "y", "z"])
    if location_latest.empty and not position.empty:
        location_latest = _latest_by_driver(position, ["driver_number"])
    position_latest = _latest_by_driver(position.rename(columns={"position": "position_in_race"}), ["driver_number", "position_in_race"])
    interval_latest = _latest_by_driver(intervals, ["driver_number", "gap_to_leader", "interval"])
    car_latest = _latest_by_driver(car_data, ["driver_number", "speed", "n_gear", "throttle", "brake", "drs"])
    stint_latest = _latest_by_driver(
        stints.rename(columns={"tyre_age_at_start": "tyre_age", "stint": "stint_number"}),
        ["driver_number", "compound", "tyre_age", "stint_number"],
    )
    lap_latest = _latest_by_driver(
        laps.rename(
            columns={
                "lap_number": "current_lap",
                "lap_duration": "last_lap_time",
                "duration_sector_1": "sector_1",
                "duration_sector_2": "sector_2",
                "duration_sector_3": "sector_3",
            }
        ),
        ["driver_number", "current_lap", "last_lap_time", "sector_1", "sector_2", "sector_3"],
    )

    merged = location_latest
    for frame in [position_latest, interval_latest, car_latest, stint_latest, lap_latest]:
        if frame.empty:
            continue
        merged = merged.merge(frame, how="left", on="driver_number")

    if not drivers.empty:
        identity = drivers.copy()
        identity["driver_number"] = identity["driver_number"].astype(str)
        identity_view = identity[["driver_number"]].copy()
        if "broadcast_name" in identity.columns and identity["broadcast_name"].notna().any():
            identity_view["driver_name"] = identity["broadcast_name"]
        elif "full_name" in identity.columns:
            identity_view["driver_name"] = identity["full_name"]
        if "name_acronym" in identity.columns:
            identity_view["driver_abbr"] = identity["name_acronym"]
        if "team_colour" in identity.columns:
            identity_view["team_color"] = identity["team_colour"]
        if "team_name" in identity.columns:
            identity_view["team_name"] = identity["team_name"]
        merged["driver_number"] = merged["driver_number"].astype(str)
        merged = merged.merge(
            identity_view,
            how="left",
            on="driver_number",
        )

    merged["driver_number"] = merged["driver_number"].astype(str)
    lookup = _driver_lookup()
    if not lookup.empty:
        merged = merged.merge(lookup, how="left", on="driver_number", suffixes=("", "_lookup"))
    if "constructor_id" not in merged.columns:
        merged["constructor_id"] = pd.NA
    merged["driver_id"] = merged["driver_id"].fillna(merged["driver_number"])

    def profile_from_id(driver_id: str) -> Any:
        return get_driver_profile(str(driver_id))

    merged["driver_name"] = merged.apply(
        lambda row: row["driver_name"] if pd.notna(row.get("driver_name")) and str(row.get("driver_name")).strip() else profile_from_id(str(row["driver_id"])).full_name,
        axis=1,
    )
    merged["driver_abbr"] = merged.apply(
        lambda row: row["driver_abbr"] if pd.notna(row.get("driver_abbr")) and str(row.get("driver_abbr")).strip() else profile_from_id(str(row["driver_id"])).abbreviation,
        axis=1,
    )
    merged["driver_id"] = merged.apply(
        lambda row: _resolve_driver_id_from_identity(row.get("driver_name"), row.get("driver_abbr"), row.get("driver_id")),
        axis=1,
    )
    merged["team_id"] = merged["constructor_id"].fillna(merged["driver_id"].map(lambda driver_id: get_driver_profile(str(driver_id)).current_team))
    merged["team_color"] = merged.apply(
        lambda row: f"#{str(row['team_color']).strip()}" if pd.notna(row.get("team_color")) and not str(row.get("team_color")).startswith("#")
        else (row["team_color"] if pd.notna(row.get("team_color")) else get_team_profile(str(row.get("team_id") or "")).primary_color),
        axis=1,
    )
    merged["circuit_id"] = circuit_id
    merged["gap_to_leader"] = merged["gap_to_leader"].fillna("Leader")
    merged["interval"] = merged["interval"].fillna("-")
    merged["position_in_race"] = pd.to_numeric(merged.get("position_in_race"), errors="coerce")
    merged = merged.sort_values(["position_in_race", "driver_abbr"], na_position="last").reset_index(drop=True)
    return merged


def _merge_position_snapshots(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    if previous.empty:
        return current
    if current.empty:
        return previous
    combined = pd.concat([previous, current], ignore_index=True, sort=False)
    if "driver_number" in combined.columns:
        combined["driver_number"] = combined["driver_number"].astype(str)
        return combined.drop_duplicates(subset=["driver_number"], keep="last").sort_values(["position_in_race", "driver_abbr"], na_position="last").reset_index(drop=True)
    return combined


def _filter_cached_session(frame: pd.DataFrame, session_group: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "session_group" not in frame.columns:
        return frame
    return frame.loc[frame["session_group"] == str(session_group).upper()].copy()


def _supplement_missing_replay_positions(
    merged: pd.DataFrame,
    circuit_id: str,
    drivers: pd.DataFrame,
    position_rank: pd.DataFrame,
    intervals: pd.DataFrame,
    car_data: pd.DataFrame,
    stints: pd.DataFrame,
    laps: pd.DataFrame,
) -> pd.DataFrame:
    if drivers.empty and position_rank.empty:
        return merged
    expected_numbers = (
        drivers["driver_number"].dropna().astype(str).tolist()
        if not drivers.empty and "driver_number" in drivers.columns
        else position_rank["driver_number"].dropna().astype(str).tolist()
    )
    expected_numbers = list(dict.fromkeys(expected_numbers))
    if not expected_numbers:
        return merged
    present_numbers = set(merged["driver_number"].dropna().astype(str).tolist()) if not merged.empty and "driver_number" in merged.columns else set()
    missing_numbers = [driver_number for driver_number in expected_numbers if driver_number not in present_numbers]
    if not missing_numbers:
        return merged

    ranking = _latest_by_driver(position_rank, ["driver_number", "position"])
    ranking = ranking.rename(columns={"position": "position_in_race"})
    ranking["position_in_race"] = pd.to_numeric(ranking["position_in_race"], errors="coerce")
    ranking = ranking.loc[ranking["driver_number"].isin(missing_numbers)].copy()
    if ranking.empty:
        ranking = pd.DataFrame({"driver_number": missing_numbers, "position_in_race": list(range(len(present_numbers) + 1, len(expected_numbers) + 1))})
    ranking = ranking.sort_values("position_in_race", na_position="last").reset_index(drop=True)

    coords = circuit_coordinates(circuit_id)
    fallback_rows: list[dict[str, Any]] = []
    total_expected = max(len(expected_numbers), 1)
    for index, row in ranking.iterrows():
        position_value = float(row["position_in_race"]) if pd.notna(row["position_in_race"]) else float(len(present_numbers) + index + 1)
        if coords.empty:
            x_value = 0.0
            y_value = 0.0
        else:
            path_index = min(max(int((position_value / total_expected) * (len(coords) - 1)), 0), len(coords) - 1)
            x_value = float(coords.iloc[path_index]["x"])
            y_value = float(coords.iloc[path_index]["y"])
        fallback_rows.append(
            {
                "driver_number": str(row["driver_number"]),
                "date": pd.Timestamp.utcnow(),
                "x": x_value,
                "y": y_value,
                "z": 0.0,
            }
        )
    fallback_location = pd.DataFrame(fallback_rows)
    fallback_merged = _merge_live_frames(circuit_id, drivers, fallback_location, position_rank, intervals, car_data, stints, laps)
    if fallback_merged.empty:
        return merged
    combined = pd.concat([merged, fallback_merged], ignore_index=True, sort=False)
    combined["driver_number"] = combined["driver_number"].astype(str)
    return combined.drop_duplicates(subset=["driver_number"], keep="first").sort_values(["position_in_race", "driver_abbr"], na_position="last").reset_index(drop=True)


def _format_since(last_poll_time: datetime | None) -> str | None:
    if last_poll_time is None:
        return None
    stamp = pd.Timestamp(last_poll_time)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def poll_live_positions(
    session_key: int,
    circuit_id: str,
    last_poll_time: datetime | None = None,
    previous_snapshot: PositionSnapshot | None = None,
) -> PositionSnapshot:
    """Fetch the latest merged position snapshot for a live or replayable session."""

    client = OpenF1Client()
    incremental_params = {"session_key": session_key}
    since = _format_since(last_poll_time)
    if since is not None:
        incremental_params["date>"] = since
    drivers = _safe_fetch(client, "drivers", {"session_key": session_key})
    driver_numbers = _driver_numbers(drivers)
    location = _safe_fetch_partitioned(client, "location", session_key, driver_numbers, extra_params={"date>": since} if since is not None else None)
    position = _safe_fetch(client, "position", incremental_params)
    intervals = _safe_fetch(client, "intervals", incremental_params)
    car_data = _safe_fetch_partitioned(client, "car_data", session_key, driver_numbers, extra_params={"date>": since} if since is not None else None)
    stints = _safe_fetch(client, "stints", {"session_key": session_key})
    laps = _safe_fetch(client, "laps", incremental_params)
    race_control = _safe_fetch(client, "race_control", incremental_params)

    positions = _merge_live_frames(circuit_id, drivers, location, position, intervals, car_data, stints, laps)
    if not positions.empty:
        positions["snapshot_time"] = datetime.now(timezone.utc)
    if previous_snapshot is not None:
        positions = _merge_position_snapshots(previous_snapshot.positions, positions)
        if previous_snapshot.messages is not None and not previous_snapshot.messages.empty and not race_control.empty:
            race_control = pd.concat([previous_snapshot.messages, race_control], ignore_index=True, sort=False).drop_duplicates()
        elif previous_snapshot.messages is not None and not previous_snapshot.messages.empty and race_control.empty:
            race_control = previous_snapshot.messages.copy()
    track_status, safety_car_deployed = _position_status(race_control)
    return PositionSnapshot(
        timestamp=datetime.now(timezone.utc),
        positions=positions,
        track_status=track_status,
        safety_car_deployed=safety_car_deployed,
        session_key=session_key,
        session_name="Live Session",
        circuit_id=circuit_id,
        messages=race_control.sort_values("date", ascending=False).head(25) if not race_control.empty else pd.DataFrame(),
    )


def _offline_replay_snapshot(season: int, round_number: int, session_name: str) -> PositionSnapshot:
    results_path = BASE_CONFIG["processed_dir"] / "race_results.csv"
    results = load_dataframe(results_path) if results_path.exists() else pd.DataFrame()
    race_frame = (
        results.loc[(results["season"] == int(season)) & (results["round"] == int(round_number))].copy()
        if not results.empty and {"season", "round"}.issubset(results.columns)
        else pd.DataFrame()
    )
    if race_frame.empty:
        calendar = load_calendar_frame(int(season), allow_fetch=False)
        circuit_id = ""
        if not calendar.empty and {"round", "circuit_id"}.issubset(calendar.columns):
            match = calendar.loc[calendar["round"] == int(round_number)]
            if not match.empty:
                circuit_id = resolve_circuit_id(str(match.iloc[0]["circuit_id"]))
        return PositionSnapshot(
            timestamp=datetime.now(timezone.utc),
            positions=pd.DataFrame(),
            track_status="NO DATA",
            safety_car_deployed=False,
            session_key=None,
            session_name=session_name,
            circuit_id=circuit_id,
            messages=pd.DataFrame(),
        )

    circuit_id = resolve_circuit_id(str(race_frame["circuit_id"].iloc[0]))
    coords = circuit_coordinates(circuit_id)
    x_path = np.asarray(coords["x"])
    y_path = np.asarray(coords["y"])
    ordered = race_frame.sort_values("finish_position").reset_index(drop=True)
    driver_rows: list[dict[str, Any]] = []
    total_points = max(len(ordered), 1)
    final_lap = int(CIRCUIT_REGISTRY.get(circuit_id, CIRCUIT_REGISTRY["silverstone"]).laps)
    for index, row in enumerate(ordered.itertuples(index=False), start=1):
        path_index = int((index / total_points) * max(len(x_path) - 1, 0)) if len(x_path) else 0
        profile = get_driver_profile(str(row.driver_id))
        team = get_team_profile(str(row.constructor_id))
        driver_rows.append(
            {
                "driver_id": row.driver_id,
                "driver_number": getattr(row, "driver_number", None),
                "driver_name": getattr(row, "driver_name", profile.full_name),
                "driver_abbr": profile.abbreviation,
                "team_id": row.constructor_id,
                "team_color": team.primary_color,
                "position_in_race": int(row.finish_position),
                "gap_to_leader": "Leader" if int(row.finish_position) == 1 else f"+{int(row.finish_position) - 1} pos",
                "interval": f"+{1} pos" if int(row.finish_position) > 1 else "Leader",
                "compound": getattr(row, "primary_compound", "--"),
                "speed": np.nan,
                "current_lap": final_lap,
                "svg_x": float(x_path[path_index]) if len(x_path) else 0.0,
                "svg_y": float(y_path[path_index]) if len(y_path) else 0.0,
            }
        )
    return PositionSnapshot(
        timestamp=datetime.now(timezone.utc),
        positions=pd.DataFrame(driver_rows),
        track_status="CHEQUERED FLAG",
        safety_car_deployed=False,
        session_key=None,
        session_name=session_name,
        circuit_id=circuit_id,
        messages=pd.DataFrame(),
    )


def load_replay_snapshot(season: int, round_number: int, session_group: str = "RACE") -> PositionSnapshot:
    """Load the latest cached replay snapshot for a weekend session."""

    base = BASE_CONFIG["raw_dir"] / "openf1" / f"{int(season)}" / f"R{int(round_number):02d}"
    circuit_id = ""
    session_name = session_group
    position_path = base / "position.parquet"
    race_control_path = base / "race_control.parquet"
    if not position_path.exists():
        return _offline_replay_snapshot(season, round_number, session_name=str(session_group).title())
    if position_path.exists():
        positions = _filter_cached_session(load_dataframe(position_path), session_group)
        if not positions.empty:
            if "date" in positions.columns:
                positions["date"] = pd.to_datetime(positions["date"], errors="coerce", utc=True)
                positions = positions.sort_values("date")
            if "driver_number" in positions.columns:
                positions["driver_number"] = positions["driver_number"].astype(str)
                if "position_in_race" in positions.columns:
                    positions["position_in_race"] = (
                        positions.groupby("driver_number")["position_in_race"].transform(lambda series: series.ffill().bfill())
                    )
            circuit_id = str(positions["circuit_id"].dropna().iloc[0]) if "circuit_id" in positions.columns and positions["circuit_id"].notna().any() else ""
            drivers = _filter_cached_session(load_dataframe(base / "drivers.parquet"), session_group) if (base / "drivers.parquet").exists() else pd.DataFrame()
            intervals = _filter_cached_session(load_dataframe(base / "intervals.parquet"), session_group) if (base / "intervals.parquet").exists() else pd.DataFrame()
            car_data = _filter_cached_session(load_dataframe(base / "car_data.parquet"), session_group) if (base / "car_data.parquet").exists() else pd.DataFrame()
            stints = _filter_cached_session(load_dataframe(base / "stints.parquet"), session_group) if (base / "stints.parquet").exists() else pd.DataFrame()
            laps = _filter_cached_session(load_dataframe(base / "laps.parquet"), session_group) if (base / "laps.parquet").exists() else pd.DataFrame()
            location = positions[[column for column in ["driver_number", "date", "x", "y", "z"] if column in positions.columns]].copy()
            position_rank = positions[[column for column in ["driver_number", "date", "position_in_race"] if column in positions.columns]].rename(columns={"position_in_race": "position"})
            latest = _merge_live_frames(circuit_id, drivers, location, position_rank, intervals, car_data, stints, laps)
            latest = _supplement_missing_replay_positions(latest, circuit_id, drivers, position_rank, intervals, car_data, stints, laps)
            latest = _enrich_position_frame(latest, season, round_number, base, circuit_id)
            messages = load_dataframe(race_control_path) if race_control_path.exists() else pd.DataFrame()
            track_status, safety_car_deployed = _position_status(messages)
            if str(session_group).upper() == "RACE":
                finish_lookup = _weekend_finish_lookup_by_driver_id(season, round_number)
                if finish_lookup:
                    latest["classified_finish_position"] = latest["driver_id"].map(finish_lookup)
                if "lookup_finish_position" in latest.columns:
                    latest["position_in_race"] = pd.to_numeric(
                        latest.get("classified_finish_position", pd.Series(dtype=float)),
                        errors="coerce",
                    ).fillna(pd.to_numeric(latest["lookup_finish_position"], errors="coerce")).fillna(
                        pd.to_numeric(latest["position_in_race"], errors="coerce")
                    )
                    latest = latest.sort_values(["position_in_race", "driver_abbr"], na_position="last").reset_index(drop=True)
                track_status = "CHEQUERED FLAG"
                safety_car_deployed = False
            return PositionSnapshot(
                timestamp=datetime.now(timezone.utc),
                positions=latest.sort_values(["position_in_race", "driver_abbr"], na_position="last").reset_index(drop=True),
                track_status=track_status,
                safety_car_deployed=safety_car_deployed,
                session_key=int(latest["session_key"].dropna().iloc[0]) if "session_key" in latest.columns and latest["session_key"].notna().any() else None,
                session_name=session_name,
                circuit_id=circuit_id,
                messages=messages.sort_values("date", ascending=False).head(25) if not messages.empty and "date" in messages.columns else messages,
            )

    return _offline_replay_snapshot(season, round_number, session_name=session_name or str(session_group).title())


def load_session_snapshot(
    season: int,
    round_number: int,
    session_group: str = "RACE",
    prefer_live: bool = False,
    last_poll_time: datetime | None = None,
    previous_snapshot: PositionSnapshot | None = None,
) -> PositionSnapshot:
    """Load a live snapshot when possible, otherwise fall back to replay data."""

    if prefer_live:
        session_key, session_name, circuit_id = resolve_session_key(season, round_number, session_group=session_group)
        if session_key is not None:
            snapshot = poll_live_positions(
                session_key,
                circuit_id,
                last_poll_time=last_poll_time,
                previous_snapshot=previous_snapshot,
            )
            if not snapshot.positions.empty or not snapshot.messages.empty:
                return PositionSnapshot(
                    timestamp=snapshot.timestamp,
                    positions=snapshot.positions,
                    track_status=snapshot.track_status,
                    safety_car_deployed=snapshot.safety_car_deployed,
                    session_key=snapshot.session_key,
                    session_name=session_name or snapshot.session_name,
                    circuit_id=circuit_id or snapshot.circuit_id,
                    messages=snapshot.messages,
                )
    return load_replay_snapshot(season, round_number, session_group=session_group)


def timing_tower_frame(snapshot: PositionSnapshot) -> pd.DataFrame:
    """Return a classic timing-tower dataframe from a snapshot."""

    if snapshot.positions.empty:
        return pd.DataFrame(columns=["Pos", "Driver", "Team", "Gap", "Int", "Last Lap", "Tyre", "Speed", "DRS"])

    def format_tyre_age(value: Any) -> str:
        if value is None or value == "" or pd.isna(value):
            return ""
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return ""

    rows: list[dict[str, Any]] = []
    for driver in snapshot.positions.itertuples(index=False):
        tyre = str(getattr(driver, "compound", "--") or "--")
        age = getattr(driver, "tyre_age", getattr(driver, "tyre_age_at_start", ""))
        rows.append(
            {
                "Pos": int(getattr(driver, "position_in_race")) if pd.notna(getattr(driver, "position_in_race")) else "-",
                "Driver": getattr(driver, "driver_abbr", getattr(driver, "driver_number", "--")),
                "Team": getattr(driver, "team_id", "--"),
                "Gap": getattr(driver, "gap_to_leader", "-"),
                "Int": getattr(driver, "interval", "-"),
                "Last Lap": getattr(driver, "last_lap_time", "-"),
                "Tyre": f"{tyre[:1].upper()}{format_tyre_age(age)}",
                "Speed": getattr(driver, "speed", "-"),
                "DRS": "ON" if pd.notna(getattr(driver, "drs", pd.NA)) and int(getattr(driver, "drs", 0)) > 0 else "OFF",
                "_team_color": getattr(driver, "team_color", "#888888"),
            }
        )
    return pd.DataFrame(rows)


def render_live_timing_tower(snapshot: PositionSnapshot) -> None:
    """Render a timing tower with subtle team-color row accents."""

    frame = timing_tower_frame(snapshot)
    if frame.empty:
        st.info("No live timing rows are available for this session.")
        return

    def style_rows(row: pd.Series) -> list[str]:
        accent = str(row["_team_color"])
        return [f"background: linear-gradient(90deg, {accent}22, transparent);" if column != "_team_color" else "" for column in row.index]

    display = frame.drop(columns=["_team_color"])
    styled = frame.style.apply(style_rows, axis=1).hide(axis="columns", subset=["_team_color"])
    st.dataframe(styled, use_container_width=True, height=620)
