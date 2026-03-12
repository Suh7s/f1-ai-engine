"""Master public-data orchestrator for the F1 AI platform."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.circuit_maps import build_circuit_svg, resolve_circuit_id
from src.data_pipeline import OpenF1Client, fetch_sessions_openf1
from src.utils import (
    CONFIG as BASE_CONFIG,
    ensure_directories,
    get_logger,
    load_calendar_frame,
    load_dataframe,
    save_parquet,
    slugify,
)


OPENF1_DIR = BASE_CONFIG["raw_dir"] / "openf1"
RACE_CONTROL_DIR = BASE_CONFIG["raw_dir"] / "race_control"
PROCESSED_COVERAGE_PATH = BASE_CONFIG["processed_dir"] / "data_universe_coverage.csv"
RACE_RESULTS_PATH = BASE_CONFIG["processed_dir"] / "race_results.csv"
LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """Definition of one OpenF1 endpoint handled by the orchestrator."""

    endpoint: str
    file_name: str
    high_volume: bool = False


@dataclass(frozen=True, slots=True)
class IngestionCoverage:
    """Coverage summary for one weekend session."""

    season: int
    round: int
    circuit_id: str
    session_key: int
    session_name: str
    session_group: str
    has_drivers: int
    has_laps: int
    has_stints: int
    has_pit: int
    has_position: int
    has_car_data: int
    has_intervals: int
    has_race_control: int
    has_weather: int
    has_team_radio: int


OPENF1_ENDPOINTS: dict[str, EndpointSpec] = {
    "drivers": EndpointSpec("drivers", "drivers.parquet"),
    "laps": EndpointSpec("laps", "laps.parquet"),
    "stints": EndpointSpec("stints", "stints.parquet"),
    "pit": EndpointSpec("pit", "pit.parquet"),
    "location": EndpointSpec("location", "location_live.parquet", high_volume=True),
    "position": EndpointSpec("position", "position_rank.parquet"),
    "car_data": EndpointSpec("car_data", "car_data_live.parquet", high_volume=True),
    "intervals": EndpointSpec("intervals", "intervals.parquet"),
    "race_control": EndpointSpec("race_control", "race_control.parquet"),
    "weather": EndpointSpec("weather", "weather.parquet"),
    "team_radio": EndpointSpec("team_radio", "team_radio.parquet"),
}


def _session_group(session_name: str) -> str:
    label = str(session_name or "").strip().lower()
    if "qual" in label:
        return "QUALI"
    if "sprint" in label and "shootout" not in label and "qual" not in label:
        return "SPRINT"
    if "race" in label:
        return "RACE"
    return "PRACTICE"


def _safe_fetch(client: OpenF1Client, endpoint: str, params: dict[str, Any]) -> pd.DataFrame:
    try:
        payload = client.fetch(f"/{endpoint}", params=params)
    except Exception as exc:
        LOGGER.debug("Non-critical error fetching OpenF1 %s with %s: %s", endpoint, params, exc)
        return pd.DataFrame()
    frame = pd.DataFrame(payload)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    return frame


def _driver_numbers(drivers: pd.DataFrame) -> list[str]:
    if drivers.empty or "driver_number" not in drivers.columns:
        return []
    values = drivers["driver_number"].dropna().tolist()
    numbers: list[str] = []
    for value in values:
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


def _downsample_time_series(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce", utc=True)
    working = working.dropna(subset=["date"]).sort_values("date")
    if working.empty:
        return working
    resampled_frames: list[pd.DataFrame] = []
    for group_keys, group in working.groupby(keys, dropna=False, sort=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        sampled = (
            group.set_index("date")
            .resample("1s")
            .last()
            .dropna(how="all")
            .reset_index()
        )
        for key, value in zip(keys, group_keys, strict=False):
            sampled[key] = value
        resampled_frames.append(sampled)
    if not resampled_frames:
        return pd.DataFrame(columns=frame.columns)
    combined = pd.concat(resampled_frames, ignore_index=True, sort=False)
    for column in frame.columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined[frame.columns]


def _merge_location_and_position(location: pd.DataFrame, position: pd.DataFrame) -> pd.DataFrame:
    if location.empty and position.empty:
        return pd.DataFrame()
    if location.empty:
        return position.rename(columns={"position": "position_in_race"}).copy()
    if position.empty:
        merged = location.copy()
        if "position" in merged.columns and "position_in_race" not in merged.columns:
            merged = merged.rename(columns={"position": "position_in_race"})
        return merged

    left = location.copy()
    right = position.copy()
    key_columns = [column for column in ["session_key", "driver_number"] if column in left.columns and column in right.columns]
    if not key_columns:
        key_columns = ["driver_number"] if "driver_number" in left.columns and "driver_number" in right.columns else []
    if "date" not in left.columns or "date" not in right.columns or not key_columns:
        merged = left.merge(right, how="left", on=key_columns, suffixes=("", "_rank"))
    else:
        left["date"] = pd.to_datetime(left["date"], errors="coerce", utc=True)
        right["date"] = pd.to_datetime(right["date"], errors="coerce", utc=True)
        left = left.dropna(subset=["date"]).sort_values("date")
        right = right.dropna(subset=["date"]).sort_values("date")
        if left.empty:
            return pd.DataFrame(columns=location.columns)
        if right.empty:
            merged = left.copy()
            if "position" in merged.columns and "position_in_race" not in merged.columns:
                merged = merged.rename(columns={"position": "position_in_race"})
            return merged
        merged = pd.merge_asof(
            left,
            right.rename(columns={"position": "position_in_race"}),
            on="date",
            by=key_columns,
            direction="nearest",
            tolerance=pd.Timedelta(seconds=2),
            suffixes=("", "_rank"),
        )
    if "position" in merged.columns and "position_in_race" not in merged.columns:
        merged = merged.rename(columns={"position": "position_in_race"})
    return merged


def _weekend_identity_lookup(season: int, round_number: int) -> pd.DataFrame:
    if not RACE_RESULTS_PATH.exists():
        return pd.DataFrame()
    race_results = load_dataframe(RACE_RESULTS_PATH)
    if race_results.empty or "driver_number" not in race_results.columns:
        return pd.DataFrame()
    weekend = race_results.loc[
        (race_results["season"] == int(season)) & (race_results["round"] == int(round_number))
    ].copy()
    if weekend.empty:
        return pd.DataFrame()
    weekend["driver_number"] = weekend["driver_number"].astype(str)
    keep = [
        column
        for column in [
            "driver_number",
            "driver_id",
            "constructor_id",
            "driver_given_name",
            "driver_family_name",
            "driver_code",
        ]
        if column in weekend.columns
    ]
    return weekend[keep].drop_duplicates(subset=["driver_number"], keep="last")


def _enrich_with_session_identity(
    frame: pd.DataFrame,
    drivers: pd.DataFrame,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    if frame.empty or "driver_number" not in frame.columns:
        return frame

    working = frame.copy()
    working["driver_number"] = working["driver_number"].astype(str)

    if not drivers.empty and "driver_number" in drivers.columns:
        manifest = drivers.copy()
        manifest["driver_number"] = manifest["driver_number"].astype(str)
        manifest_view = manifest[["driver_number"]].copy()
        if "broadcast_name" in manifest.columns and manifest["broadcast_name"].notna().any():
            manifest_view["driver_name"] = manifest["broadcast_name"]
        elif "full_name" in manifest.columns:
            manifest_view["driver_name"] = manifest["full_name"]
        if "name_acronym" in manifest.columns:
            manifest_view["driver_abbr"] = manifest["name_acronym"]
        if "team_name" in manifest.columns:
            manifest_view["team_name"] = manifest["team_name"]
        if "team_colour" in manifest.columns:
            manifest_view["team_color"] = manifest["team_colour"]
        working = working.merge(
            manifest_view.drop_duplicates(subset=["driver_number"], keep="last"),
            how="left",
            on="driver_number",
        )

    identity_lookup = _weekend_identity_lookup(season, round_number)
    if not identity_lookup.empty:
        working = working.merge(identity_lookup, how="left", on="driver_number")

    if "driver_name" not in working.columns:
        working["driver_name"] = pd.NA
    has_driver_name = working["driver_name"].notna() & working["driver_name"].astype(str).str.strip().ne("")
    if {"driver_given_name", "driver_family_name"}.issubset(working.columns):
        working.loc[~has_driver_name, "driver_name"] = (
            working.loc[~has_driver_name, "driver_given_name"].fillna("").astype(str).str.strip()
            + " "
            + working.loc[~has_driver_name, "driver_family_name"].fillna("").astype(str).str.strip()
        ).str.strip()

    if "driver_abbr" not in working.columns:
        working["driver_abbr"] = pd.NA
    if "driver_code" in working.columns:
        missing_abbr = working["driver_abbr"].isna() | working["driver_abbr"].astype(str).str.strip().eq("")
        working.loc[missing_abbr, "driver_abbr"] = working.loc[missing_abbr, "driver_code"]

    return working


def _weekend_directory(season: int, round_number: int) -> Path:
    return OPENF1_DIR / f"{int(season)}" / f"R{int(round_number):02d}"


def _match_sessions_to_rounds(calendar: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    if calendar.empty or sessions.empty:
        return pd.DataFrame()

    working = sessions.copy()
    working["date_start"] = pd.to_datetime(working["date_start"], errors="coerce", utc=True)
    if "meeting_key" in working.columns and working["meeting_key"].notna().any():
        working["meeting_group"] = working["meeting_key"].astype("Int64").astype(str)
    else:
        working["meeting_group"] = (
            working["meeting_name"].fillna("")
            + "|"
            + working["location"].fillna("")
            + "|"
            + working["country_name"].fillna("")
        )

    meeting_rows: list[dict[str, Any]] = []
    for meeting_group, meeting in working.groupby("meeting_group", dropna=False, sort=False):
        ordered = meeting.sort_values("date_start")
        race_sessions = ordered.loc[
            ordered["session_name"].astype(str).str.contains("race", case=False, na=False)
            & ~ordered["session_name"].astype(str).str.contains("sprint", case=False, na=False)
        ]
        meeting_rows.append(
            {
                "meeting_group": meeting_group,
                "meeting_start": ordered["date_start"].min(),
                "meeting_reference_date": race_sessions["date_start"].max() if not race_sessions.empty else ordered["date_start"].max(),
            }
        )
    grouped = pd.DataFrame(meeting_rows).sort_values("meeting_reference_date").reset_index(drop=True)

    rows: list[pd.DataFrame] = []
    for calendar_row in calendar.sort_values("round").itertuples(index=False):
        target_date = pd.Timestamp(calendar_row.date).tz_localize("UTC") if pd.Timestamp(calendar_row.date).tzinfo is None else pd.Timestamp(calendar_row.date)
        summary = grouped.copy()
        summary["date_gap"] = (summary["meeting_reference_date"] - target_date).abs()
        summary["is_reasonable_window"] = summary["date_gap"] <= pd.Timedelta(days=7)
        if summary["is_reasonable_window"].any():
            summary = summary.loc[summary["is_reasonable_window"]].copy()
        match = summary.sort_values(["date_gap", "meeting_start"]).head(1)
        if match.empty:
            continue
        meeting_group = str(match["meeting_group"].iloc[0])
        meeting_sessions = working.loc[working["meeting_group"] == meeting_group].copy()
        meeting_sessions["season"] = int(calendar_row.season)
        meeting_sessions["round"] = int(calendar_row.round)
        meeting_sessions["race_name"] = str(calendar_row.race_name)
        meeting_sessions["circuit_id"] = resolve_circuit_id(str(calendar_row.circuit_id))
        rows.append(meeting_sessions)
        grouped = grouped.loc[grouped["meeting_group"] != meeting_group].reset_index(drop=True)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["season", "round", "date_start"]).reset_index(drop=True)


class DataUniverse:
    """Coordinate public data ingestion into a stable raw-data layout."""

    def __init__(self, client: OpenF1Client | None = None) -> None:
        self.client = client or OpenF1Client()

    def ingest_season(self, season: int, refresh: bool = False) -> pd.DataFrame:
        """Ingest all mapped OpenF1 weekend sessions for a season."""

        calendar = load_calendar_frame(int(season), allow_fetch=False)
        sessions = fetch_sessions_openf1(self.client, int(season))
        mapped = _match_sessions_to_rounds(calendar, sessions)
        if mapped.empty:
            return pd.DataFrame()

        coverage_rows: list[dict[str, Any]] = []
        for round_number, round_sessions in mapped.groupby("round", sort=True):
            coverage_rows.extend(self.ingest_round(int(season), int(round_number), sessions_df=round_sessions, refresh=refresh))

        coverage = pd.DataFrame(coverage_rows)
        if not coverage.empty:
            ensure_directories([PROCESSED_COVERAGE_PATH.parent])
            coverage.to_csv(PROCESSED_COVERAGE_PATH, index=False)
        return coverage

    def ingest_round(
        self,
        season: int,
        round_number: int,
        sessions_df: pd.DataFrame | None = None,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Ingest one race weekend into unified raw storage."""

        calendar = load_calendar_frame(int(season), allow_fetch=False)
        calendar_row = calendar.loc[calendar["round"] == int(round_number)].head(1)
        if calendar_row.empty:
            return []

        if sessions_df is None:
            sessions = fetch_sessions_openf1(self.client, int(season))
            sessions_df = _match_sessions_to_rounds(calendar, sessions).loc[lambda frame: frame["round"] == int(round_number)]
        if sessions_df.empty:
            return []

        circuit_id = resolve_circuit_id(str(calendar_row["circuit_id"].iloc[0]))
        build_circuit_svg(circuit_id, refresh=refresh)
        output_dir = _weekend_directory(season, round_number)
        ensure_directories([output_dir])
        save_parquet(sessions_df, output_dir / "sessions.parquet")

        driver_frames: list[pd.DataFrame] = []
        lap_frames: list[pd.DataFrame] = []
        stint_frames: list[pd.DataFrame] = []
        pit_frames: list[pd.DataFrame] = []
        position_frames: list[pd.DataFrame] = []
        car_frames: list[pd.DataFrame] = []
        interval_frames: list[pd.DataFrame] = []
        race_control_frames: list[pd.DataFrame] = []
        weather_frames: list[pd.DataFrame] = []
        radio_frames: list[pd.DataFrame] = []
        coverage_rows: list[dict[str, Any]] = []

        for session in sessions_df.itertuples(index=False):
            session_name = str(session.session_name)
            session_group = _session_group(session_name)
            base_columns = {
                "season": int(season),
                "round": int(round_number),
                "circuit_id": circuit_id,
                "session_key": int(session.session_key),
                "session_name": session_name,
                "session_group": session_group,
            }

            drivers = _safe_fetch(self.client, "drivers", {"session_key": session.session_key})
            driver_numbers = _driver_numbers(drivers)
            laps = _safe_fetch(self.client, "laps", {"session_key": session.session_key})
            stints = _safe_fetch(self.client, "stints", {"session_key": session.session_key})
            pit = _safe_fetch(self.client, "pit", {"session_key": session.session_key})
            location = _safe_fetch_partitioned(self.client, "location", int(session.session_key), driver_numbers)
            position_rank = _safe_fetch(self.client, "position", {"session_key": session.session_key})
            car_data = _safe_fetch_partitioned(self.client, "car_data", int(session.session_key), driver_numbers)
            intervals = _safe_fetch(self.client, "intervals", {"session_key": session.session_key})
            race_control = _safe_fetch(self.client, "race_control", {"session_key": session.session_key})
            weather = _safe_fetch(self.client, "weather", {"session_key": session.session_key})
            team_radio = _safe_fetch(self.client, "team_radio", {"session_key": session.session_key})

            merged_position = _merge_location_and_position(location, position_rank)
            reduced_position = _downsample_time_series(merged_position, ["session_key", "driver_number"]) if not merged_position.empty else merged_position
            reduced_car_data = _downsample_time_series(car_data, ["session_key", "driver_number"]) if not car_data.empty else car_data
            reduced_position = _enrich_with_session_identity(reduced_position, drivers, season, round_number)
            reduced_car_data = _enrich_with_session_identity(reduced_car_data, drivers, season, round_number)
            intervals = _enrich_with_session_identity(intervals, drivers, season, round_number)
            stints = _enrich_with_session_identity(stints, drivers, season, round_number)
            laps = _enrich_with_session_identity(laps, drivers, season, round_number)

            for frame, collection in [
                (drivers, driver_frames),
                (laps, lap_frames),
                (stints, stint_frames),
                (pit, pit_frames),
                (reduced_position, position_frames),
                (reduced_car_data, car_frames),
                (intervals, interval_frames),
                (race_control, race_control_frames),
                (weather, weather_frames),
                (team_radio, radio_frames),
            ]:
                if frame.empty:
                    continue
                enriched = frame.copy()
                for key, value in base_columns.items():
                    if key not in enriched.columns:
                        enriched[key] = value
                collection.append(enriched)

            if not location.empty:
                live_path = output_dir / f"{slugify(session_name)}_location_live.parquet"
                save_parquet(location.assign(**base_columns), live_path)
            if not car_data.empty:
                live_path = output_dir / f"{slugify(session_name)}_car_data_live.parquet"
                save_parquet(car_data.assign(**base_columns), live_path)

            coverage_rows.append(
                asdict(
                    IngestionCoverage(
                        season=int(season),
                        round=int(round_number),
                        circuit_id=circuit_id,
                        session_key=int(session.session_key),
                        session_name=session_name,
                        session_group=session_group,
                        has_drivers=int(not drivers.empty),
                        has_laps=int(not laps.empty),
                        has_stints=int(not stints.empty),
                        has_pit=int(not pit.empty),
                        has_position=int(not merged_position.empty),
                        has_car_data=int(not car_data.empty),
                        has_intervals=int(not intervals.empty),
                        has_race_control=int(not race_control.empty),
                        has_weather=int(not weather.empty),
                        has_team_radio=int(not team_radio.empty),
                    )
                )
            )

        outputs = {
            "drivers.parquet": driver_frames,
            "laps.parquet": lap_frames,
            "stints.parquet": stint_frames,
            "pit.parquet": pit_frames,
            "position.parquet": position_frames,
            "car_data.parquet": car_frames,
            "intervals.parquet": interval_frames,
            "race_control.parquet": race_control_frames,
            "weather.parquet": weather_frames,
            "team_radio.parquet": radio_frames,
        }
        for file_name, frames in outputs.items():
            combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            save_parquet(combined, output_dir / file_name)

        race_control_frame = pd.concat(race_control_frames, ignore_index=True, sort=False) if race_control_frames else pd.DataFrame()
        if not race_control_frame.empty:
            race_control_output = RACE_CONTROL_DIR / f"{int(season)}" / f"R{int(round_number):02d}.parquet"
            ensure_directories([race_control_output.parent])
            race_only = race_control_frame.loc[race_control_frame["session_group"] == "RACE"].copy()
            save_parquet(race_only if not race_only.empty else race_control_frame, race_control_output)

        return coverage_rows


def _scan_openf1_position_coverage() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position_path in sorted(OPENF1_DIR.glob("*/R*/position.parquet")):
        try:
            frame = load_dataframe(position_path)
        except Exception as exc:
            LOGGER.debug("Non-critical error reading position coverage %s: %s", position_path, exc)
            continue
        if frame.empty:
            continue
        season = int(position_path.parents[1].name)
        round_number = int(position_path.parent.name[1:])
        if "circuit_id" in frame.columns and frame["circuit_id"].notna().any():
            circuit_id = resolve_circuit_id(str(frame["circuit_id"].dropna().iloc[0]))
        else:
            calendar = load_calendar_frame(season, allow_fetch=False)
            match = calendar.loc[calendar["round"] == round_number] if not calendar.empty else pd.DataFrame()
            circuit_id = resolve_circuit_id(str(match["circuit_id"].iloc[0])) if not match.empty else ""
        session_groups = frame["session_group"].dropna().astype(str).unique().tolist() if "session_group" in frame.columns else ["RACE"]
        for session_group in sorted(session_groups):
            session_frame = frame.loc[frame["session_group"] == session_group].copy() if "session_group" in frame.columns else frame
            if session_frame.empty:
                continue
            session_name = (
                str(session_frame["session_name"].dropna().iloc[0])
                if "session_name" in session_frame.columns and session_frame["session_name"].notna().any()
                else session_group.title()
            )
            rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "session_name": session_name,
                    "session_group": session_group,
                    "circuit_id": circuit_id,
                    "position_rows": int(len(session_frame)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["season", "round", "session_name", "session_group", "circuit_id", "position_rows"])
    return pd.DataFrame(rows).sort_values(["season", "round", "session_name"]).reset_index(drop=True)


def load_data_universe_coverage() -> pd.DataFrame:
    """Load the latest persisted coverage summary if present."""

    if PROCESSED_COVERAGE_PATH.exists():
        return load_dataframe(PROCESSED_COVERAGE_PATH)
    return _scan_openf1_position_coverage()


def openf1_position_coverage() -> pd.DataFrame:
    """Return the sessions with cached position/location overlays available."""

    coverage = load_data_universe_coverage()
    if not coverage.empty and "has_position" in coverage.columns:
        return (
            coverage.loc[coverage["has_position"] == 1, ["season", "round", "session_name", "session_group", "circuit_id"]]
            .sort_values(["season", "round", "session_name"])
            .reset_index(drop=True)
        )
    return coverage
