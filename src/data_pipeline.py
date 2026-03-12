"""Data ingestion and normalization for the F1 AI prediction engine."""

from __future__ import annotations

import datetime
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from src.utils import (
    CONFIG as BASE_CONFIG,
    coerce_numeric,
    ensure_directories,
    get_logger,
    load_dataframe,
    load_json,
    save_dataframe,
    save_json,
    season_window,
    validate_columns,
)


def get_current_season() -> int:
    """Return the active F1 season using runtime evaluation."""

    now = datetime.datetime.now()
    if now.month < 2:
        return now.year - 1
    return now.year


def _current_season_window() -> list[int]:
    """Return the rolling ingestion window for the live season."""

    return season_window(get_current_season(), lookback_years=10)


_DEFAULT_SEASON_WINDOW = _current_season_window()

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "raw_dir": BASE_CONFIG["raw_dir"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "base_url": "https://api.jolpi.ca/ergast/f1",
    "openf1_base_url": "https://api.openf1.org/v1",
    "season_start": _DEFAULT_SEASON_WINDOW[0],
    "season_end": _DEFAULT_SEASON_WINDOW[-1],
    "refresh_start_season": 2023,
    "recent_lap_seasons": [
        season for season in _DEFAULT_SEASON_WINDOW if season >= max(_DEFAULT_SEASON_WINDOW[-1] - 2, 2022)
    ],
    "recent_openf1_seasons": [season for season in _DEFAULT_SEASON_WINDOW if season >= 2023],
    "request_limit": 100,
    "heavy_request_limit": 2000,
    "openf1_rate_limit_seconds": 0.10,
    "request_timeout": 30,
    "max_retries": 3,
    "retry_backoff_seconds": 1.5,
    "rate_limit_seconds": 0.75,
}

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CachedDataset:
    """Bundle cache paths and metadata for a season-level endpoint."""

    name: str
    season: int
    csv_path: Path
    json_path: Path


def _cache_paths(name: str, season: int) -> CachedDataset:
    """Return standardized cache file locations for a dataset."""

    return CachedDataset(
        name=name,
        season=season,
        csv_path=CONFIG["raw_dir"] / f"{name}_{season}.csv",
        json_path=CONFIG["raw_dir"] / f"{name}_{season}.json",
    )


def _duration_to_milliseconds(value: str | None) -> float | None:
    """Convert a timing string into milliseconds."""

    if value is None or value == "":
        return None

    if ":" in value:
        minutes_str, seconds_str = value.split(":", maxsplit=1)
        return (int(minutes_str) * 60.0 + float(seconds_str)) * 1000.0
    return float(value) * 1000.0


def _status_is_finished(status: str | None) -> bool:
    """Identify statuses that represent a classified finish."""

    if not status:
        return False
    return bool(re.match(r"^(Finished|\+\d+ Laps?|\+\d+ Lap)$", status))


def classify_dnf_status(status: str | None, finish_position: Any) -> tuple[int, str]:
    """Classify whether a race status represents a DNF and its broad category."""

    mechanical_keywords = {
        "engine",
        "gearbox",
        "transmission",
        "hydraulics",
        "power unit",
        "battery",
        "electrical",
        "electronics",
        "driveshaft",
        "oil leak",
        "water leak",
        "fuel pressure",
        "fuel system",
        "turbo",
        "radiator",
        "clutch",
        "brakes",
        "suspension",
        "puncture",
        "tyre",
        "wheel",
        "mechanical",
        "exhaust",
        "cooling",
    }
    accident_keywords = {
        "accident",
        "collision",
        "crash",
        "damage",
        "spun off",
        "spun",
        "barrier",
        "contact",
        "hit",
    }

    status_text = (status or "").strip().lower()
    finish_numeric = pd.to_numeric(pd.Series([finish_position]), errors="coerce").iloc[0]

    if pd.notna(finish_numeric) and _status_is_finished(status):
        return 0, "finished"
    if status_text == "":
        return 1, "other"
    if any(keyword in status_text for keyword in mechanical_keywords):
        return 1, "mechanical"
    if any(keyword in status_text for keyword in accident_keywords):
        return 1, "accident"
    return 1, "other"


class ErgastClient:
    """Thin Jolpica client with Ergast-compatible pagination and rate limiting."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.last_request_time = 0.0

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_time
        sleep_seconds = CONFIG["rate_limit_seconds"] - elapsed
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _request_json(self, relative_path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a single Ergast request with retry logic."""

        url = f"{CONFIG['base_url'].rstrip('/')}/{relative_path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                self._respect_rate_limit()
                response = self.session.get(url, params=params, timeout=CONFIG["request_timeout"])
                self.last_request_time = time.monotonic()
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                status_code = exc.response.status_code if getattr(exc, "response", None) is not None else None
                LOGGER.warning(
                    "Request failed for %s on attempt %s/%s: %s",
                    url,
                    attempt,
                    CONFIG["max_retries"],
                    exc,
                )
                if attempt < CONFIG["max_retries"]:
                    sleep_seconds = float(2 ** attempt)
                    if status_code == 429:
                        retry_after = int(getattr(exc.response, "headers", {}).get("Retry-After", 60))
                        sleep_seconds = max(sleep_seconds, float(retry_after))
                    time.sleep(sleep_seconds)

        raise RuntimeError(f"Failed to fetch {url}") from last_error

    def fetch_paginated(self, relative_path: str, child_key: str, limit: int | None = None) -> dict[str, Any]:
        """Fetch all pages for a season endpoint and combine nested race payloads."""

        combined_response: dict[str, Any] | None = None
        race_map: dict[tuple[str, str], dict[str, Any]] = {}
        offset = 0
        page_limit = int(limit or CONFIG["request_limit"])

        while True:
            payload = self._request_json(
                relative_path,
                params={"limit": page_limit, "offset": offset},
            )
            mr_data = payload["MRData"]
            race_table = mr_data["RaceTable"]
            races = race_table.get("Races", [])

            if combined_response is None:
                combined_response = payload

            for race in races:
                race_key = (race["season"], race["round"])
                if race_key not in race_map:
                    race_copy = dict(race)
                    race_copy[child_key] = list(race.get(child_key, []))
                    race_map[race_key] = race_copy
                else:
                    race_map[race_key][child_key].extend(race.get(child_key, []))

            total = int(mr_data.get("total", len(races)))
            offset += page_limit
            if offset >= total:
                break

        if combined_response is None:
            raise ValueError(f"No paginated payload returned for {relative_path}")
        combined_response["MRData"]["RaceTable"]["Races"] = [
            race_map[key] for key in sorted(race_map.keys(), key=lambda item: (int(item[0]), int(item[1])))
        ]
        combined_response["MRData"]["offset"] = "0"
        combined_response["MRData"]["limit"] = str(page_limit)
        combined_response["MRData"]["total"] = str(total if "total" in locals() else len(race_map))
        return combined_response


class OpenF1Client:
    """HTTP client for the OpenF1 API."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.last_request_time = 0.0

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_time
        sleep_seconds = CONFIG["openf1_rate_limit_seconds"] - elapsed
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def fetch(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch a list payload from OpenF1 with retry handling."""

        url = f"{CONFIG['openf1_base_url'].rstrip('/')}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                self._respect_rate_limit()
                response = self.session.get(url, params=params, timeout=CONFIG["request_timeout"])
                self.last_request_time = time.monotonic()
                if response.status_code == 404:
                    return []
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list):
                    return payload
                raise ValueError(f"Unexpected OpenF1 payload type for {url}: {type(payload)!r}")
            except requests.RequestException as exc:
                last_error = exc
                status_code = exc.response.status_code if getattr(exc, "response", None) is not None else None
                LOGGER.warning(
                    "Network error fetching data for %s on attempt %s/%s: %s",
                    url,
                    attempt,
                    CONFIG["max_retries"],
                    exc,
                )
                if attempt < CONFIG["max_retries"]:
                    sleep_seconds = float(2 ** attempt)
                    if status_code == 429:
                        retry_after = int(getattr(exc.response, "headers", {}).get("Retry-After", 60))
                        sleep_seconds = max(sleep_seconds, float(retry_after))
                    time.sleep(sleep_seconds)
            except ValueError as exc:
                last_error = exc
                LOGGER.error("Invalid response data: %s", exc)
                if attempt < CONFIG["max_retries"]:
                    time.sleep(float(2 ** attempt))

        raise RuntimeError(f"Failed to fetch {url}") from last_error


def _openf1_cache_paths(name: str, season: int) -> CachedDataset:
    """Return standardized cache file locations for OpenF1 season aggregates."""

    return CachedDataset(
        name=name,
        season=season,
        csv_path=CONFIG["raw_dir"] / f"openf1_{name}_{season}.csv",
        json_path=CONFIG["raw_dir"] / f"openf1_{name}_{season}.json",
    )


def _normalize_driver_number(value: Any) -> str | None:
    """Normalize driver numbers across Jolpica and OpenF1 payloads."""

    if pd.isna(value) or value in (None, ""):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def fetch_sessions_openf1(client: OpenF1Client, year: int) -> pd.DataFrame:
    """Fetch OpenF1 sessions for a given year."""

    payload = client.fetch("/sessions", params={"year": year})
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "meeting_key",
                "session_key",
                "session_type",
                "session_name",
                "meeting_name",
                "location",
                "country_name",
                "circuit_short_name",
                "date_start",
            ]
        )

    frame["season"] = year
    frame["date_start"] = pd.to_datetime(frame.get("date_start"), errors="coerce", utc=True)
    keep_columns = [
        "season",
        "meeting_key",
        "session_key",
        "session_type",
        "session_name",
        "meeting_name",
        "location",
        "country_name",
        "circuit_short_name",
        "date_start",
    ]
    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[keep_columns].drop_duplicates(subset=["session_key"]).sort_values("date_start").reset_index(drop=True)


def fetch_stints(client: OpenF1Client, session_key: int | str) -> pd.DataFrame:
    """Fetch tire stint data for a session from OpenF1."""

    payload = client.fetch("/stints", params={"session_key": session_key})
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "session_key",
                "driver_number",
                "stint_number",
                "compound",
                "lap_start",
                "lap_end",
                "tyre_age_at_start",
            ]
        )

    rename_map = {
        "stint": "stint_number",
        "compound": "compound",
        "lap_start": "lap_start",
        "lap_end": "lap_end",
        "tyre_age_at_start": "tyre_age_at_start",
    }
    frame = frame.rename(columns=rename_map)
    frame["session_key"] = session_key
    frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number)
    for column in ["stint_number", "lap_start", "lap_end", "tyre_age_at_start"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    keep_columns = [
        "session_key",
        "driver_number",
        "stint_number",
        "compound",
        "lap_start",
        "lap_end",
        "tyre_age_at_start",
    ]
    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[keep_columns].dropna(subset=["driver_number"]).reset_index(drop=True)


def fetch_pit_stops_openf1(client: OpenF1Client, session_key: int | str) -> pd.DataFrame:
    """Fetch pit stop data for a session from OpenF1."""

    payload = client.fetch("/pit", params={"session_key": session_key})
    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(columns=["session_key", "driver_number", "pit_duration", "lap_number"])

    frame["session_key"] = session_key
    frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number)
    if "pit_duration" not in frame.columns:
        if "duration" in frame.columns:
            frame["pit_duration"] = pd.to_numeric(frame["duration"], errors="coerce")
        else:
            frame["pit_duration"] = pd.NA
    if "lap_number" not in frame.columns:
        if "lap" in frame.columns:
            frame["lap_number"] = pd.to_numeric(frame["lap"], errors="coerce")
        else:
            frame["lap_number"] = pd.NA
    frame["pit_duration"] = pd.to_numeric(frame["pit_duration"], errors="coerce")
    frame["lap_number"] = pd.to_numeric(frame["lap_number"], errors="coerce")
    keep_columns = ["session_key", "driver_number", "pit_duration", "lap_number"]
    return frame[keep_columns].dropna(subset=["driver_number"]).reset_index(drop=True)


def _match_openf1_sessions_to_rounds(results_df: pd.DataFrame, sessions_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Match OpenF1 race sessions to Jolpica race rounds by chronological order."""

    season_races = (
        results_df.loc[results_df["season"] == season, ["season", "round", "race_name", "race_date", "circuit_id", "circuit_name"]]
        .drop_duplicates()
        .sort_values("round")
        .reset_index(drop=True)
    )
    if season_races.empty or sessions_df.empty:
        return pd.DataFrame(
            columns=["season", "round", "session_key", "session_name", "meeting_name", "date_start"]
        )

    race_sessions = sessions_df.loc[
        sessions_df["session_name"].astype(str).str.lower().eq("race")
    ].sort_values("date_start").reset_index(drop=True)
    if race_sessions.empty:
        return pd.DataFrame(
            columns=["season", "round", "session_key", "session_name", "meeting_name", "date_start"]
        )

    race_count = min(len(season_races), len(race_sessions))
    matched = season_races.iloc[:race_count].copy()
    session_slice = race_sessions.iloc[:race_count].reset_index(drop=True)
    matched = pd.concat(
        [
            matched.reset_index(drop=True),
            session_slice[["session_key", "session_name", "meeting_name", "date_start"]].reset_index(drop=True),
        ],
        axis=1,
    )
    return matched


def _fetch_or_load_openf1_year_dataset(
    name: str,
    season: int,
    builder: Callable[[OpenF1Client, int], tuple[Any, pd.DataFrame]],
    refresh: bool = False,
    client: OpenF1Client | None = None,
) -> pd.DataFrame:
    """Fetch or load a season-level OpenF1 aggregate."""

    cache = _openf1_cache_paths(name, season)
    if cache.csv_path.exists() and not refresh:
        return load_dataframe(cache.csv_path)

    if client is None:
        client = OpenF1Client()

    try:
        raw_payload, frame = builder(client, season)
        save_json(raw_payload, cache.json_path)
        save_dataframe(frame, cache.csv_path)
        LOGGER.info("Fetched OpenF1 %s season %s with %s rows", name, season, len(frame))
        return frame
    except Exception as exc:
        LOGGER.warning("Falling back to cached OpenF1 %s season %s after error: %s", name, season, exc)
        if cache.csv_path.exists():
            return load_dataframe(cache.csv_path)
        raise


def _build_openf1_sessions_dataset(client: OpenF1Client, season: int) -> tuple[Any, pd.DataFrame]:
    """Fetch and return the OpenF1 session listing for a season."""

    payload = client.fetch("/sessions", params={"year": season})
    frame = pd.DataFrame(payload)
    if frame.empty:
        empty = pd.DataFrame(
            columns=[
                "season",
                "session_key",
                "session_name",
                "meeting_name",
                "location",
                "country_name",
                "circuit_short_name",
                "date_start",
            ]
        )
        return payload, empty

    frame["season"] = season
    frame["date_start"] = pd.to_datetime(frame.get("date_start"), errors="coerce", utc=True)
    keep_columns = [
        "season",
        "session_key",
        "session_name",
        "meeting_name",
        "location",
        "country_name",
        "circuit_short_name",
        "date_start",
    ]
    for column in keep_columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return payload, frame[keep_columns].drop_duplicates(subset=["session_key"]).sort_values("date_start").reset_index(drop=True)


def _build_openf1_stints_dataset(
    client: OpenF1Client,
    season: int,
    results_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
) -> tuple[Any, pd.DataFrame]:
    """Fetch and aggregate OpenF1 stints for a season."""

    session_map = _match_openf1_sessions_to_rounds(results_df, sessions_df, season)
    raw_payload: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for row in session_map.itertuples(index=False):
        try:
            frame = fetch_stints(client, row.session_key)
        except Exception as exc:
            LOGGER.warning(
                "Skipping OpenF1 stints for season %s round %s session %s: %s",
                season,
                int(row.round),
                row.session_key,
                exc,
            )
            continue
        if frame.empty:
            continue
        frame["season"] = season
        frame["round"] = int(row.round)
        raw_payload.append({"session_key": row.session_key, "round": int(row.round), "data": frame.to_dict(orient="records")})
        frames.append(frame)

    if not frames:
        empty = pd.DataFrame(
            columns=[
                "season",
                "round",
                "session_key",
                "driver_number",
                "stint_number",
                "compound",
                "lap_start",
                "lap_end",
                "tyre_age_at_start",
            ]
        )
        return raw_payload, empty

    frame = pd.concat(frames, ignore_index=True)
    frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number)
    return raw_payload, frame


def _build_openf1_pit_dataset(
    client: OpenF1Client,
    season: int,
    results_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
) -> tuple[Any, pd.DataFrame]:
    """Fetch and aggregate OpenF1 pit stop data for a season."""

    session_map = _match_openf1_sessions_to_rounds(results_df, sessions_df, season)
    raw_payload: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for row in session_map.itertuples(index=False):
        try:
            frame = fetch_pit_stops_openf1(client, row.session_key)
        except Exception as exc:
            LOGGER.warning(
                "Skipping OpenF1 pit data for season %s round %s session %s: %s",
                season,
                int(row.round),
                row.session_key,
                exc,
            )
            continue
        if frame.empty:
            continue
        frame["season"] = season
        frame["round"] = int(row.round)
        raw_payload.append({"session_key": row.session_key, "round": int(row.round), "data": frame.to_dict(orient="records")})
        frames.append(frame)

    if not frames:
        empty = pd.DataFrame(columns=["season", "round", "session_key", "driver_number", "pit_duration", "lap_number"])
        return raw_payload, empty

    frame = pd.concat(frames, ignore_index=True)
    frame["driver_number"] = frame["driver_number"].apply(_normalize_driver_number)
    return raw_payload, frame


def _enrich_recent_races_with_openf1(
    race_results: pd.DataFrame,
    session_map_df: pd.DataFrame,
    stints_df: pd.DataFrame,
    openf1_pit_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge aggregated OpenF1 stint and pit stop metrics into race results."""

    enriched = race_results.copy()
    if "driver_number" in enriched.columns:
        enriched["driver_number_key"] = enriched["driver_number"].apply(_normalize_driver_number)
    else:
        enriched["driver_number_key"] = pd.NA

    session_map = session_map_df[["season", "round", "session_key"]].drop_duplicates() if not session_map_df.empty else pd.DataFrame(columns=["season", "round", "session_key"])
    enriched = enriched.merge(session_map, how="left", on=["season", "round"])

    if not stints_df.empty:
        stint_work = stints_df.copy()
        stint_work["driver_number"] = stint_work["driver_number"].apply(_normalize_driver_number)
        stint_work["stint_length"] = pd.to_numeric(stint_work["lap_end"], errors="coerce") - pd.to_numeric(
            stint_work["lap_start"], errors="coerce"
        ) + 1
        stint_summary = (
            stint_work.sort_values(["season", "round", "driver_number", "stint_number"])
            .groupby(["season", "round", "driver_number"], as_index=False)
            .agg(
                total_stints=("stint_number", "nunique"),
                max_stint_laps=("stint_length", "max"),
                tyre_age_at_start=("tyre_age_at_start", "mean"),
                primary_compound=("compound", lambda series: series.mode().iloc[0] if not series.mode().empty else series.iloc[0]),
            )
            .rename(columns={"driver_number": "driver_number_key"})
        )
        enriched = enriched.merge(
            stint_summary,
            how="left",
            on=["season", "round", "driver_number_key"],
        )
    else:
        enriched["total_stints"] = pd.NA
        enriched["max_stint_laps"] = pd.NA
        enriched["tyre_age_at_start"] = pd.NA
        enriched["primary_compound"] = pd.NA

    if not openf1_pit_df.empty:
        pit_work = openf1_pit_df.copy()
        pit_work["driver_number"] = pit_work["driver_number"].apply(_normalize_driver_number)
        pit_summary = (
            pit_work.groupby(["season", "round", "driver_number"], as_index=False)
            .agg(
                openf1_pit_stop_count=("pit_duration", "count"),
                pit_duration=("pit_duration", "mean"),
            )
            .rename(columns={"driver_number": "driver_number_key"})
        )
        enriched = enriched.merge(
            pit_summary,
            how="left",
            on=["season", "round", "driver_number_key"],
        )
    else:
        enriched["openf1_pit_stop_count"] = pd.NA
        enriched["pit_duration"] = pd.NA

    return enriched.drop(columns=["driver_number_key"])


def normalize_results_payload(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Flatten Ergast race result payloads into tabular form."""

    records: list[dict[str, Any]] = []
    races = payload["MRData"]["RaceTable"].get("Races", [])

    for race in races:
        race_round = int(race["round"])
        circuit_id = race["Circuit"]["circuitId"]
        circuit_name = race["Circuit"]["circuitName"]
        race_name = race["raceName"]
        race_date = race.get("date")
        race_time = race.get("time")

        for result in race.get("Results", []):
            driver = result["Driver"]
            constructor = result["Constructor"]
            finish_position = pd.to_numeric(result.get("position"), errors="coerce")
            dnf_flag, dnf_reason = classify_dnf_status(result.get("status"), finish_position)
            fastest_lap = result.get("FastestLap", {})
            records.append(
                {
                    "season": season,
                    "round": race_round,
                    "race_name": race_name,
                    "race_date": race_date,
                    "race_time": race_time,
                    "circuit_id": circuit_id,
                    "circuit_name": circuit_name,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code"),
                    "driver_number": driver.get("permanentNumber"),
                    "driver_given_name": driver.get("givenName"),
                    "driver_family_name": driver.get("familyName"),
                    "constructor_id": constructor["constructorId"],
                    "constructor_name": constructor["name"],
                    "grid_position": pd.to_numeric(result.get("grid"), errors="coerce"),
                    "finish_position": finish_position,
                    "finish_position_text": result.get("positionText"),
                    "points": pd.to_numeric(result.get("points"), errors="coerce"),
                    "status": result.get("status"),
                    "laps_completed": pd.to_numeric(result.get("laps"), errors="coerce"),
                    "fastest_lap_rank": pd.to_numeric(fastest_lap.get("rank"), errors="coerce"),
                    "fastest_lap_time": fastest_lap.get("Time", {}).get("time"),
                    "fastest_lap_avg_speed": fastest_lap.get("AverageSpeed", {}).get("speed"),
                    "dnf_flag": dnf_flag,
                    "dnf_reason": dnf_reason,
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "round",
                "race_name",
                "race_date",
                "race_time",
                "circuit_id",
                "circuit_name",
                "driver_id",
                "driver_code",
                "driver_number",
                "driver_given_name",
                "driver_family_name",
                "constructor_id",
                "constructor_name",
                "grid_position",
                "finish_position",
                "finish_position_text",
                "points",
                "status",
                "laps_completed",
                "fastest_lap_rank",
                "fastest_lap_time",
                "fastest_lap_avg_speed",
                "dnf_flag",
                "dnf_reason",
            ]
        )
    frame = frame.drop_duplicates(subset=["season", "round", "driver_id"], keep="last")
    validate_columns(
        frame,
        [
            "season",
            "round",
            "circuit_id",
            "driver_id",
            "constructor_id",
            "grid_position",
            "finish_position",
            "points",
            "status",
            "laps_completed",
            "fastest_lap_rank",
            "dnf_flag",
            "dnf_reason",
        ],
        "normalized race results",
    )
    return frame


def normalize_qualifying_payload(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Flatten Ergast qualifying payloads into tabular form."""

    records: list[dict[str, Any]] = []
    races = payload["MRData"]["RaceTable"].get("Races", [])

    for race in races:
        race_round = int(race["round"])
        circuit_id = race["Circuit"]["circuitId"]
        for result in race.get("QualifyingResults", []):
            driver = result["Driver"]
            constructor = result["Constructor"]
            records.append(
                {
                    "season": season,
                    "round": race_round,
                    "circuit_id": circuit_id,
                    "driver_id": driver["driverId"],
                    "constructor_id": constructor["constructorId"],
                    "qualifying_position": pd.to_numeric(result.get("position"), errors="coerce"),
                    "q1": result.get("Q1"),
                    "q2": result.get("Q2"),
                    "q3": result.get("Q3"),
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "round", "circuit_id", "driver_id", "constructor_id", "qualifying_position", "q1", "q2", "q3"]
        )
    frame = frame.drop_duplicates(subset=["season", "round", "driver_id"], keep="last")
    validate_columns(
        frame,
        ["season", "round", "circuit_id", "driver_id", "constructor_id", "qualifying_position"],
        "normalized qualifying results",
    )
    return frame


def normalize_laps_payload(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Flatten Ergast lap time payloads into tabular form."""

    records: list[dict[str, Any]] = []
    races = payload["MRData"]["RaceTable"].get("Races", [])

    for race in races:
        race_round = int(race["round"])
        circuit_id = race["Circuit"]["circuitId"]
        for lap in race.get("Laps", []):
            lap_number = int(lap["number"])
            for timing in lap.get("Timings", []):
                records.append(
                    {
                        "season": season,
                        "round": race_round,
                        "circuit_id": circuit_id,
                        "driver_id": timing["driverId"],
                        "lap": lap_number,
                        "position": pd.to_numeric(timing.get("position"), errors="coerce"),
                        "lap_time": timing.get("time"),
                        "lap_time_ms": _duration_to_milliseconds(timing.get("time")),
                    }
                )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "round", "circuit_id", "driver_id", "lap", "position", "lap_time", "lap_time_ms"]
        )
    frame = frame.drop_duplicates(subset=["season", "round", "driver_id", "lap"], keep="last")
    validate_columns(
        frame,
        ["season", "round", "circuit_id", "driver_id", "lap", "position", "lap_time_ms"],
        "normalized lap times",
    )
    return frame


def normalize_laps_payloads(payloads: list[dict[str, Any]], season: int) -> pd.DataFrame:
    """Flatten multiple per-round lap payloads into one table."""

    frames = [normalize_laps_payload(payload, season) for payload in payloads if payload.get("MRData")]
    if not frames:
        return pd.DataFrame(
            columns=["season", "round", "circuit_id", "driver_id", "lap", "position", "lap_time", "lap_time_ms"]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["season", "round", "driver_id", "lap"],
        keep="last",
    )


def normalize_pitstops_payload(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """Flatten Ergast pit stop payloads into tabular form."""

    records: list[dict[str, Any]] = []
    races = payload["MRData"]["RaceTable"].get("Races", [])

    for race in races:
        race_round = int(race["round"])
        circuit_id = race["Circuit"]["circuitId"]
        for stop in race.get("PitStops", []):
            records.append(
                {
                    "season": season,
                    "round": race_round,
                    "circuit_id": circuit_id,
                    "driver_id": stop["driverId"],
                    "stop": pd.to_numeric(stop.get("stop"), errors="coerce"),
                    "lap": pd.to_numeric(stop.get("lap"), errors="coerce"),
                    "pit_time": stop.get("time"),
                    "duration": stop.get("duration"),
                    "duration_ms": _duration_to_milliseconds(stop.get("duration")),
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "round", "circuit_id", "driver_id", "stop", "lap", "pit_time", "duration", "duration_ms"]
        )
    frame = frame.drop_duplicates(subset=["season", "round", "driver_id", "stop"], keep="last")
    validate_columns(
        frame,
        ["season", "round", "circuit_id", "driver_id", "stop", "lap", "duration_ms"],
        "normalized pit stop data",
    )
    return frame


def normalize_pitstops_payloads(payloads: list[dict[str, Any]], season: int) -> pd.DataFrame:
    """Flatten multiple per-round pit-stop payloads into one table."""

    frames = [normalize_pitstops_payload(payload, season) for payload in payloads if payload.get("MRData")]
    if not frames:
        return pd.DataFrame(
            columns=["season", "round", "circuit_id", "driver_id", "stop", "lap", "pit_time", "duration", "duration_ms"]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["season", "round", "driver_id", "stop"],
        keep="last",
    )


def normalize_driver_standings_payload(payloads: list[dict[str, Any]], season: int) -> pd.DataFrame:
    """Flatten a season of per-round driver standings."""

    records: list[dict[str, Any]] = []
    for payload in payloads:
        standings_lists = payload["MRData"]["StandingsTable"].get("StandingsLists", [])
        for standings in standings_lists:
            round_number = int(standings["round"])
            for standing in standings.get("DriverStandings", []):
                constructors = standing.get("Constructors", [])
                constructor_id = constructors[0]["constructorId"] if constructors else None
                records.append(
                    {
                        "season": season,
                        "round": round_number,
                        "driver_id": standing["Driver"]["driverId"],
                        "constructor_id": constructor_id,
                        "position": pd.to_numeric(standing.get("position"), errors="coerce"),
                        "points": pd.to_numeric(standing.get("points"), errors="coerce"),
                        "wins": pd.to_numeric(standing.get("wins"), errors="coerce"),
                    }
                )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id", "position", "points", "wins"])
    validate_columns(
        frame,
        ["season", "round", "driver_id", "constructor_id", "position", "points", "wins"],
        "normalized driver standings",
    )
    return frame


def normalize_constructor_standings_payload(payloads: list[dict[str, Any]], season: int) -> pd.DataFrame:
    """Flatten a season of per-round constructor standings."""

    records: list[dict[str, Any]] = []
    for payload in payloads:
        standings_lists = payload["MRData"]["StandingsTable"].get("StandingsLists", [])
        for standings in standings_lists:
            round_number = int(standings["round"])
            for standing in standings.get("ConstructorStandings", []):
                records.append(
                    {
                        "season": season,
                        "round": round_number,
                        "constructor_id": standing["Constructor"]["constructorId"],
                        "position": pd.to_numeric(standing.get("position"), errors="coerce"),
                        "points": pd.to_numeric(standing.get("points"), errors="coerce"),
                        "wins": pd.to_numeric(standing.get("wins"), errors="coerce"),
                    }
                )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["season", "round", "constructor_id", "position", "points", "wins"])
    validate_columns(
        frame,
        ["season", "round", "constructor_id", "position", "points", "wins"],
        "normalized constructor standings",
    )
    return frame


def normalize_calendar_payload(
    payload: dict[str, Any],
    season: int,
    completed_rounds: set[int] | None = None,
) -> pd.DataFrame:
    """Flatten a season calendar payload and mark completed rounds."""

    completed = {int(round_number) for round_number in (completed_rounds or set())}
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    records: list[dict[str, Any]] = []
    for race in races:
        location = race.get("Circuit", {}).get("Location", {})
        round_number = int(race.get("round", 0))
        records.append(
            {
                "season": int(season),
                "round": round_number,
                "race_name": race.get("raceName"),
                "circuit_id": race.get("Circuit", {}).get("circuitId"),
                "circuit_name": race.get("Circuit", {}).get("circuitName"),
                "country": location.get("country"),
                "date": race.get("date"),
                "is_completed": int(round_number in completed),
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["season", "round", "race_name", "circuit_id", "circuit_name", "country", "date", "is_completed"]
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values("round").reset_index(drop=True)


def _calendar_cache(season: int) -> CachedDataset:
    """Return cache paths for a season calendar."""

    return _cache_paths("calendar", season)


def fetch_calendar(
    client: ErgastClient,
    season: int,
    completed_rounds: set[int] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fetch a season calendar from Jolpica."""

    payload = client._request_json(f"{season}/races.json")
    return payload, normalize_calendar_payload(payload, season, completed_rounds=completed_rounds)


def fetch_or_load_calendar(
    season: int,
    completed_rounds: set[int] | None = None,
    refresh: bool = False,
    client: ErgastClient | None = None,
) -> pd.DataFrame:
    """Fetch or load a season calendar with completion flags."""

    ensure_directories()
    cache = _calendar_cache(season)
    if cache.csv_path.exists() and not refresh:
        frame = load_dataframe(cache.csv_path, parse_dates=["date"])
        if completed_rounds is not None and not frame.empty:
            frame["is_completed"] = frame["round"].astype(int).isin(completed_rounds).astype(int)
            save_dataframe(frame, cache.csv_path)
        return frame

    client = client or ErgastClient()
    try:
        payload, frame = fetch_calendar(client, season, completed_rounds=completed_rounds)
        save_json(payload, cache.json_path)
        save_dataframe(frame, cache.csv_path)
        LOGGER.info("Fetched calendar season %s with %s races", season, len(frame))
        return frame
    except Exception as exc:
        LOGGER.warning("Falling back to cached calendar season %s after error: %s", season, exc)
        if cache.csv_path.exists():
            return load_dataframe(cache.csv_path, parse_dates=["date"])
        cached_payload = load_json(cache.json_path)
        if cached_payload is not None:
            frame = normalize_calendar_payload(cached_payload, season, completed_rounds=completed_rounds)
            save_dataframe(frame, cache.csv_path)
            return frame
        raise


def get_cached_rounds(season: int) -> list[int]:
    """Return locally cached completed rounds for a season."""

    results_cache = _cache_paths("results", season)
    if results_cache.csv_path.exists():
        frame = load_dataframe(results_cache.csv_path)
        if not frame.empty and "round" in frame.columns:
            return sorted(frame["round"].dropna().astype(int).unique().tolist())

    processed_results = CONFIG["processed_dir"] / "race_results.csv"
    if processed_results.exists():
        frame = load_dataframe(processed_results)
        frame = frame.loc[frame["season"] == season]
        if not frame.empty:
            return sorted(frame["round"].dropna().astype(int).unique().tolist())
    return []


def _load_cached_csv_or_raise(cache: CachedDataset) -> pd.DataFrame:
    """Load an existing CSV cache or raise if unavailable."""

    LOGGER.info("Using cached dataset: %s", cache.csv_path)
    return load_dataframe(cache.csv_path)


def fetch_or_load(
    cache_name: str,
    season: int,
    fetcher: Callable[[ErgastClient, int], tuple[Any, pd.DataFrame]],
    refresh: bool = False,
    client: ErgastClient | None = None,
) -> pd.DataFrame:
    """Fetch a season-level dataset or reuse an existing cache."""

    ensure_directories()
    cache = _cache_paths(cache_name, season)
    if cache.csv_path.exists() and not refresh:
        return _load_cached_csv_or_raise(cache)

    if client is None:
        client = ErgastClient()

    try:
        raw_payload, frame = fetcher(client, season)
        save_json(raw_payload, cache.json_path)
        save_dataframe(frame, cache.csv_path)
        LOGGER.info("Fetched %s season %s with %s rows", cache_name, season, len(frame))
        return frame
    except Exception as exc:
        LOGGER.warning("Falling back to cache for %s season %s after fetch failure: %s", cache_name, season, exc)
        if cache.csv_path.exists():
            return _load_cached_csv_or_raise(cache)
        fallback_payload = load_json(cache.json_path)
        if fallback_payload is not None:
            if cache_name == "results":
                frame = normalize_results_payload(fallback_payload, season)
            elif cache_name == "qualifying":
                frame = normalize_qualifying_payload(fallback_payload, season)
            elif cache_name == "laps":
                frame = normalize_laps_payload(fallback_payload, season)
            elif cache_name == "pitstops":
                frame = normalize_pitstops_payload(fallback_payload, season)
            elif cache_name == "driver_standings":
                frame = normalize_driver_standings_payload(fallback_payload, season)
            elif cache_name == "constructor_standings":
                frame = normalize_constructor_standings_payload(fallback_payload, season)
            else:
                raise ValueError(f"Unsupported cache name: {cache_name}")
            save_dataframe(frame, cache.csv_path)
            return frame
        raise


def fetch_results(client: ErgastClient, season: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fetch all race results for a season."""

    payload = client.fetch_paginated(f"{season}/results.json", child_key="Results")
    return payload, normalize_results_payload(payload, season)


def fetch_qualifying(client: ErgastClient, season: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fetch all qualifying results for a season."""

    payload = client.fetch_paginated(f"{season}/qualifying.json", child_key="QualifyingResults")
    return payload, normalize_qualifying_payload(payload, season)


def fetch_laps(client: ErgastClient, season: int, rounds: list[int]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Fetch lap times round-by-round for a season."""

    payloads: list[dict[str, Any]] = []
    for round_number in rounds:
        payloads.append(
            client.fetch_paginated(
                f"{season}/{round_number}/laps.json",
                child_key="Laps",
                limit=CONFIG["heavy_request_limit"],
            )
        )
    return payloads, normalize_laps_payloads(payloads, season)


def fetch_pitstops(client: ErgastClient, season: int, rounds: list[int]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Fetch pit stop data round-by-round for a season."""

    payloads: list[dict[str, Any]] = []
    for round_number in rounds:
        payloads.append(
            client.fetch_paginated(
                f"{season}/{round_number}/pitstops.json",
                child_key="PitStops",
                limit=CONFIG["heavy_request_limit"],
            )
        )
    return payloads, normalize_pitstops_payloads(payloads, season)


def fetch_driver_standings(
    client: ErgastClient,
    season: int,
    rounds: list[int],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Fetch per-round driver standings for a season."""

    payloads: list[dict[str, Any]] = []
    for round_number in rounds:
        payloads.append(client._request_json(f"{season}/{round_number}/driverStandings.json"))
    return payloads, normalize_driver_standings_payload(payloads, season)


def fetch_constructor_standings(
    client: ErgastClient,
    season: int,
    rounds: list[int],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Fetch per-round constructor standings for a season."""

    payloads: list[dict[str, Any]] = []
    for round_number in rounds:
        payloads.append(client._request_json(f"{season}/{round_number}/constructorStandings.json"))
    return payloads, normalize_constructor_standings_payload(payloads, season)


def fetch_or_load_standings(
    cache_name: str,
    season: int,
    rounds: list[int],
    refresh: bool = False,
    client: ErgastClient | None = None,
) -> pd.DataFrame:
    """Fetch standings data with cache-fallback behavior."""

    ensure_directories()
    cache = _cache_paths(cache_name, season)
    if cache.csv_path.exists() and not refresh:
        return load_dataframe(cache.csv_path)

    if client is None:
        client = ErgastClient()

    try:
        if cache_name == "driver_standings":
            payloads, frame = fetch_driver_standings(client, season, rounds)
        elif cache_name == "constructor_standings":
            payloads, frame = fetch_constructor_standings(client, season, rounds)
        else:
            raise ValueError(f"Unsupported standings cache: {cache_name}")
        save_json(payloads, cache.json_path)
        save_dataframe(frame, cache.csv_path)
        LOGGER.info("Fetched %s season %s with %s rows", cache_name, season, len(frame))
        return frame
    except Exception as exc:
        LOGGER.warning("Falling back to cached %s season %s after error: %s", cache_name, season, exc)
        if cache.csv_path.exists():
            return load_dataframe(cache.csv_path)
        cached_payloads = load_json(cache.json_path)
        if cached_payloads is not None:
            if cache_name == "driver_standings":
                frame = normalize_driver_standings_payload(cached_payloads, season)
            else:
                frame = normalize_constructor_standings_payload(cached_payloads, season)
            save_dataframe(frame, cache.csv_path)
            return frame
        raise


def fetch_or_load_roundwise_dataset(
    cache_name: str,
    season: int,
    rounds: list[int],
    fetcher: Callable[[ErgastClient, int, list[int]], tuple[list[dict[str, Any]], pd.DataFrame]],
    refresh: bool = False,
    client: ErgastClient | None = None,
) -> pd.DataFrame:
    """Fetch a per-round season aggregate with cache fallback."""

    ensure_directories()
    cache = _cache_paths(cache_name, season)
    if cache.csv_path.exists() and not refresh:
        return load_dataframe(cache.csv_path)

    if client is None:
        client = ErgastClient()

    try:
        payloads, frame = fetcher(client, season, rounds)
        save_json(payloads, cache.json_path)
        save_dataframe(frame, cache.csv_path)
        LOGGER.info("Fetched %s season %s with %s rows", cache_name, season, len(frame))
        return frame
    except Exception as exc:
        LOGGER.warning("Falling back to cached %s season %s after error: %s", cache_name, season, exc)
        if cache.csv_path.exists():
            return load_dataframe(cache.csv_path)
        cached_payloads = load_json(cache.json_path)
        if cached_payloads is not None:
            if cache_name == "laps":
                frame = normalize_laps_payloads(cached_payloads, season)
            elif cache_name == "pitstops":
                frame = normalize_pitstops_payloads(cached_payloads, season)
            else:
                raise ValueError(f"Unsupported roundwise cache: {cache_name}")
            save_dataframe(frame, cache.csv_path)
            return frame
        raise


def build_unified_race_results(results_df: pd.DataFrame, qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """Merge raw results and qualifying into a unified race-level training table."""

    validate_columns(
        results_df,
        [
            "season",
            "round",
            "circuit_id",
            "driver_id",
            "constructor_id",
            "grid_position",
            "finish_position",
            "points",
            "status",
            "laps_completed",
            "fastest_lap_rank",
            "dnf_flag",
            "dnf_reason",
        ],
        "results dataframe",
    )
    validate_columns(
        qualifying_df,
        ["season", "round", "driver_id", "constructor_id", "qualifying_position"],
        "qualifying dataframe",
    )

    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "round",
                "race_name",
                "race_date",
                "race_time",
                "circuit_id",
                "circuit_name",
                "driver_id",
                "driver_code",
                "driver_number",
                "driver_given_name",
                "driver_family_name",
                "constructor_id",
                "constructor_name",
                "grid_position",
                "finish_position",
                "points",
                "status",
                "laps_completed",
                "fastest_lap_rank",
                "qualifying_position",
                "grid_penalty",
                "dnf_flag",
                "dnf_reason",
                "q1",
                "q2",
                "q3",
            ]
        )

    merged = results_df.merge(
        qualifying_df[
            [
                "season",
                "round",
                "driver_id",
                "constructor_id",
                "qualifying_position",
                "q1",
                "q2",
                "q3",
            ]
        ],
        how="left",
        on=["season", "round", "driver_id", "constructor_id"],
    )

    merged["grid_position"] = coerce_numeric(merged["grid_position"])
    merged["qualifying_position"] = coerce_numeric(merged["qualifying_position"])
    merged["finish_position"] = coerce_numeric(merged["finish_position"])
    merged["points"] = coerce_numeric(merged["points"], fill_value=0.0)
    merged["laps_completed"] = coerce_numeric(merged["laps_completed"])
    merged["fastest_lap_rank"] = coerce_numeric(merged["fastest_lap_rank"])
    merged["qualifying_position"] = merged["qualifying_position"].fillna(merged["grid_position"])
    merged["grid_penalty"] = (
        merged["grid_position"].notna()
        & merged["qualifying_position"].notna()
        & (merged["grid_position"] > merged["qualifying_position"])
    )

    unified_columns = [
        "season",
        "round",
        "race_name",
        "race_date",
        "race_time",
        "circuit_id",
        "circuit_name",
        "driver_id",
        "driver_code",
        "driver_number",
        "driver_given_name",
        "driver_family_name",
        "constructor_id",
        "constructor_name",
        "grid_position",
        "finish_position",
        "points",
        "status",
        "laps_completed",
        "fastest_lap_rank",
        "qualifying_position",
        "grid_penalty",
        "dnf_flag",
        "dnf_reason",
        "q1",
        "q2",
        "q3",
    ]
    unified = merged[unified_columns].sort_values(["season", "round", "finish_position", "grid_position"])
    validate_columns(
        unified,
        [
            "season",
            "round",
            "circuit_id",
            "driver_id",
            "constructor_id",
            "grid_position",
            "finish_position",
            "points",
            "status",
            "laps_completed",
            "fastest_lap_rank",
            "qualifying_position",
            "dnf_flag",
            "dnf_reason",
        ],
        "unified race results",
    )
    return unified.reset_index(drop=True)


def run_data_pipeline(
    refresh: bool = False,
    seasons: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the living ingestion stage and persist both raw and processed outputs."""

    ensure_directories()
    client = ErgastClient()
    openf1_client = OpenF1Client()
    dynamic_window = _current_season_window()
    active_window = sorted({int(season) for season in (seasons or dynamic_window)})
    target_refresh_seasons = {int(season) for season in (seasons or active_window)}
    current_season = get_current_season()
    recent_lap_seasons = [season for season in active_window if season >= max(active_window[-1] - 2, 2022)]
    recent_openf1_seasons = [season for season in active_window if season >= 2023]
    refresh_start_season = int(CONFIG["refresh_start_season"])

    LOGGER.info("Starting data pipeline for seasons %s-%s", active_window[0], active_window[-1])

    def should_refresh(season: int) -> bool:
        return bool(
            refresh
            and season in target_refresh_seasons
            and (season >= refresh_start_season or not _cache_paths("results", season).csv_path.exists())
        )

    results_frames = [
        fetch_or_load(
            "results",
            season,
            fetch_results,
            refresh=should_refresh(season),
            client=client,
        )
        for season in active_window
    ]
    results_df = pd.concat(results_frames, ignore_index=True) if results_frames else pd.DataFrame()

    completed_rounds_by_season = {
        season: set(results_df.loc[results_df["season"] == season, "round"].dropna().astype(int).unique().tolist())
        for season in active_window
    }

    calendar_frames = [
        fetch_or_load_calendar(
            season=season,
            completed_rounds=completed_rounds_by_season.get(season, set()),
            refresh=should_refresh(season),
            client=client,
        )
        for season in active_window
    ]
    calendar_df = pd.concat(calendar_frames, ignore_index=True) if calendar_frames else pd.DataFrame()

    qualifying_frames = [
        fetch_or_load(
            "qualifying",
            season,
            fetch_qualifying,
            refresh=should_refresh(season),
            client=client,
        )
        for season in active_window
    ]
    qualifying_df = pd.concat(qualifying_frames, ignore_index=True) if qualifying_frames else pd.DataFrame()
    if not qualifying_df.empty:
        qualifying_df = qualifying_df.loc[
            qualifying_df.apply(
                lambda row: int(row["round"]) in completed_rounds_by_season.get(int(row["season"]), set()),
                axis=1,
            )
        ].reset_index(drop=True)

    rounds_by_season = {
        season: sorted(completed_rounds_by_season.get(season, set()))
        for season in active_window
    }

    driver_standings_frames = [
        fetch_or_load_standings(
            "driver_standings",
            season,
            rounds_by_season.get(season, []),
            refresh=should_refresh(season),
            client=client,
        )
        for season in active_window
    ]
    constructor_standings_frames = [
        fetch_or_load_standings(
            "constructor_standings",
            season,
            rounds_by_season.get(season, []),
            refresh=should_refresh(season),
            client=client,
        )
        for season in active_window
    ]
    lap_frames = [
        fetch_or_load_roundwise_dataset(
            "laps",
            season,
            rounds_by_season.get(season, []),
            fetch_laps,
            refresh=should_refresh(season),
            client=client,
        )
        for season in recent_lap_seasons
    ]
    pitstop_frames = [
        fetch_or_load_roundwise_dataset(
            "pitstops",
            season,
            rounds_by_season.get(season, []),
            fetch_pitstops,
            refresh=should_refresh(season),
            client=client,
        )
        for season in recent_lap_seasons
    ]

    driver_standings_df = pd.concat(driver_standings_frames, ignore_index=True) if driver_standings_frames else pd.DataFrame()
    constructor_standings_df = pd.concat(constructor_standings_frames, ignore_index=True) if constructor_standings_frames else pd.DataFrame()
    lap_times_df = pd.concat(lap_frames, ignore_index=True) if lap_frames else pd.DataFrame()
    pitstops_df = pd.concat(pitstop_frames, ignore_index=True) if pitstop_frames else pd.DataFrame()
    unified_results = build_unified_race_results(results_df, qualifying_df)

    openf1_sessions_frames: list[pd.DataFrame] = []
    openf1_session_map_frames: list[pd.DataFrame] = []
    openf1_stints_frames: list[pd.DataFrame] = []
    openf1_pit_frames: list[pd.DataFrame] = []

    for season in recent_openf1_seasons:
        completed_rounds = rounds_by_season.get(season, [])
        if not completed_rounds:
            continue

        sessions_df = _fetch_or_load_openf1_year_dataset(
            "sessions",
            season,
            _build_openf1_sessions_dataset,
            refresh=should_refresh(season),
            client=openf1_client,
        )
        openf1_sessions_frames.append(sessions_df)

        season_results = results_df.loc[results_df["season"] == season].copy()
        session_map_df = _match_openf1_sessions_to_rounds(season_results, sessions_df, season)
        if season == current_season:
            session_map_df = session_map_df.loc[session_map_df["round"].isin(completed_rounds)].copy()
        openf1_session_map_frames.append(session_map_df)

        stints_df = _fetch_or_load_openf1_year_dataset(
            "stints",
            season,
            lambda client_arg, year: _build_openf1_stints_dataset(client_arg, year, season_results, sessions_df),
            refresh=should_refresh(season),
            client=openf1_client,
        )
        pit_openf1_df = _fetch_or_load_openf1_year_dataset(
            "pit",
            season,
            lambda client_arg, year: _build_openf1_pit_dataset(client_arg, year, season_results, sessions_df),
            refresh=should_refresh(season),
            client=openf1_client,
        )

        if season == current_season:
            stints_df = stints_df.loc[stints_df["round"].isin(completed_rounds)].copy()
            pit_openf1_df = pit_openf1_df.loc[pit_openf1_df["round"].isin(completed_rounds)].copy()

        if not stints_df.empty:
            openf1_stints_frames.append(stints_df)
        if not pit_openf1_df.empty:
            openf1_pit_frames.append(pit_openf1_df)

    openf1_sessions_df = (
        pd.concat(openf1_sessions_frames, ignore_index=True)
        if openf1_sessions_frames
        else pd.DataFrame(columns=["season", "session_key", "session_name", "meeting_name", "location", "country_name", "circuit_short_name", "date_start"])
    )
    openf1_session_map_df = (
        pd.concat(openf1_session_map_frames, ignore_index=True)
        if openf1_session_map_frames
        else pd.DataFrame(columns=["season", "round", "session_key", "session_name", "meeting_name", "date_start"])
    )
    openf1_stints_df = (
        pd.concat(openf1_stints_frames, ignore_index=True)
        if openf1_stints_frames
        else pd.DataFrame(columns=["season", "round", "session_key", "driver_number", "stint_number", "compound", "lap_start", "lap_end", "tyre_age_at_start"])
    )
    openf1_pit_df = (
        pd.concat(openf1_pit_frames, ignore_index=True)
        if openf1_pit_frames
        else pd.DataFrame(columns=["season", "round", "session_key", "driver_number", "pit_duration", "lap_number"])
    )

    unified_results = _enrich_recent_races_with_openf1(
        unified_results,
        openf1_session_map_df,
        openf1_stints_df,
        openf1_pit_df,
    )

    save_dataframe(results_df, CONFIG["raw_dir"] / "results_all.csv")
    save_dataframe(calendar_df, CONFIG["raw_dir"] / "calendar_all.csv")
    save_dataframe(qualifying_df, CONFIG["raw_dir"] / "qualifying_all.csv")
    save_dataframe(driver_standings_df, CONFIG["raw_dir"] / "driver_standings_all.csv")
    save_dataframe(constructor_standings_df, CONFIG["raw_dir"] / "constructor_standings_all.csv")
    save_dataframe(lap_times_df, CONFIG["raw_dir"] / "lap_times_all.csv")
    save_dataframe(pitstops_df, CONFIG["raw_dir"] / "pitstops_all.csv")
    save_dataframe(openf1_sessions_df, CONFIG["raw_dir"] / "openf1_sessions_all.csv")
    save_dataframe(openf1_session_map_df, CONFIG["raw_dir"] / "openf1_session_map_all.csv")
    save_dataframe(openf1_stints_df, CONFIG["raw_dir"] / "openf1_stints_all.csv")
    save_dataframe(openf1_pit_df, CONFIG["raw_dir"] / "openf1_pit_all.csv")

    save_dataframe(unified_results, CONFIG["processed_dir"] / "race_results.csv")
    save_dataframe(calendar_df, CONFIG["processed_dir"] / "calendar.csv")
    save_dataframe(driver_standings_df, CONFIG["processed_dir"] / "driver_standings.csv")
    save_dataframe(constructor_standings_df, CONFIG["processed_dir"] / "constructor_standings.csv")
    save_dataframe(lap_times_df, CONFIG["processed_dir"] / "lap_times.csv")
    save_dataframe(pitstops_df, CONFIG["processed_dir"] / "pitstops.csv")
    save_dataframe(openf1_sessions_df, CONFIG["processed_dir"] / "openf1_sessions.csv")
    save_dataframe(openf1_session_map_df, CONFIG["processed_dir"] / "openf1_session_map.csv")
    save_dataframe(openf1_stints_df, CONFIG["processed_dir"] / "openf1_stints.csv")
    save_dataframe(openf1_pit_df, CONFIG["processed_dir"] / "openf1_pit.csv")

    LOGGER.info(
        "Data pipeline complete: results=%s qualifying=%s calendars=%s laps=%s pitstops=%s openf1_stints=%s",
        len(unified_results),
        len(qualifying_df),
        len(calendar_df),
        len(lap_times_df),
        len(pitstops_df),
        len(openf1_stints_df),
    )

    return {
        "results": results_df,
        "calendar": calendar_df,
        "qualifying": qualifying_df,
        "driver_standings": driver_standings_df,
        "constructor_standings": constructor_standings_df,
        "lap_times": lap_times_df,
        "pitstops": pitstops_df,
        "openf1_sessions": openf1_sessions_df,
        "openf1_session_map": openf1_session_map_df,
        "openf1_stints": openf1_stints_df,
        "openf1_pit": openf1_pit_df,
        "race_results": unified_results,
    }


def load_calendar(season: int, allow_fetch: bool = True) -> pd.DataFrame:
    """Load a processed season calendar, optionally fetching it if missing."""

    processed_path = CONFIG["processed_dir"] / "calendar.csv"
    if processed_path.exists():
        frame = load_dataframe(processed_path, parse_dates=["date"])
        season_frame = frame.loc[frame["season"] == int(season)].copy()
        if not season_frame.empty:
            return season_frame.sort_values("round").reset_index(drop=True)

    cache = _calendar_cache(int(season))
    if cache.csv_path.exists():
        return load_dataframe(cache.csv_path, parse_dates=["date"]).sort_values("round").reset_index(drop=True)

    if not allow_fetch:
        return pd.DataFrame(columns=["season", "round", "race_name", "circuit_id", "circuit_name", "country", "date", "is_completed"])

    client = ErgastClient()
    completed_rounds = set(get_cached_rounds(int(season)))
    return fetch_or_load_calendar(int(season), completed_rounds=completed_rounds, refresh=False, client=client)


def load_processed_inputs() -> dict[str, pd.DataFrame]:
    """Load processed outputs from disk for downstream stages."""

    datasets = {
        "race_results": load_dataframe(CONFIG["processed_dir"] / "race_results.csv"),
        "calendar": load_dataframe(CONFIG["processed_dir"] / "calendar.csv", parse_dates=["date"]),
        "driver_standings": load_dataframe(CONFIG["processed_dir"] / "driver_standings.csv"),
        "constructor_standings": load_dataframe(CONFIG["processed_dir"] / "constructor_standings.csv"),
        "lap_times": load_dataframe(CONFIG["processed_dir"] / "lap_times.csv"),
        "pitstops": load_dataframe(CONFIG["processed_dir"] / "pitstops.csv"),
    }
    for name, frame in datasets.items():
        if name != "calendar" and frame.empty:
            raise ValueError(f"Processed dataset {name} is empty")
    return datasets
