"""Compatibility registry backed by the canonical metadata databases."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.driver_database import DRIVER_DATABASE, get_driver_profile, resolve_driver_id
from src.regulations_database import REGULATION_HISTORY
from src.team_database import TEAM_DATABASE, get_team_profile, resolve_team_id
from src.utils import CONFIG as BASE_CONFIG, detect_current_season, load_dataframe


ROOKIE_REGISTRY: dict[int, list[str]] = {
    2025: ["antonelli", "bortoleto", "doohan", "hadjar"],
    2026: ["arvid_lindblad"],
}

DRIVER_HOME_CIRCUITS: dict[str, list[str]] = {
    "albon": ["silverstone"],
    "alonso": ["catalunya"],
    "antonelli": ["imola", "monza"],
    "bearman": ["silverstone"],
    "bortoleto": ["interlagos"],
    "button": ["silverstone"],
    "colapinto": ["interlagos"],
    "gasly": ["ricard"],
    "hadjar": ["ricard"],
    "hamilton": ["silverstone"],
    "hulkenberg": ["hockenheimring", "nurburgring"],
    "lawson": ["red_bull_ring"],
    "leclerc": ["monaco"],
    "max_verstappen": ["zandvoort"],
    "norris": ["silverstone"],
    "ocon": ["ricard"],
    "perez": ["rodriguez"],
    "piastri": ["albert_park"],
    "russell": ["silverstone"],
    "sainz": ["catalunya"],
    "stroll": ["villeneuve"],
    "tsunoda": ["suzuka"],
    "vettel": ["hockenheimring", "nurburgring"],
    "zhou": ["shanghai"],
}

CURRENT_SEASON = detect_current_season()
REGULATION_CHANGES: dict[int, str] = {year: change.type for year, change in REGULATION_HISTORY.items()}
DRIVER_DISPLAY_NAMES: dict[str, str] = {driver_id: profile.full_name for driver_id, profile in DRIVER_DATABASE.items()}
CONSTRUCTOR_DISPLAY_NAMES: dict[str, str] = {team_id: profile.short_name for team_id, profile in TEAM_DATABASE.items()}


@lru_cache(maxsize=1)
def _race_results() -> pd.DataFrame:
    path = BASE_CONFIG["processed_dir"] / "race_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return load_dataframe(path)


def _latest_season_available() -> int:
    frame = _race_results()
    if frame.empty:
        return CURRENT_SEASON
    return int(frame["season"].max())


def _latest_round_for_season(season: int) -> int | None:
    frame = _race_results()
    season_frame = frame.loc[frame["season"] == int(season)] if not frame.empty else pd.DataFrame()
    if season_frame.empty:
        return None
    return int(season_frame["round"].max())


def _season_roster_from_results(season: int) -> pd.DataFrame:
    frame = _race_results()
    if frame.empty:
        return pd.DataFrame()
    target_season = int(season)
    if target_season not in frame["season"].unique():
        target_season = _latest_season_available()
    season_frame = frame.loc[frame["season"] == target_season].copy()
    if season_frame.empty:
        return pd.DataFrame()
    latest_round = int(season_frame["round"].max())
    roster = season_frame.loc[season_frame["round"] == latest_round].copy()
    roster["driver_id"] = roster["driver_id"].map(resolve_driver_id)
    roster["constructor_id"] = roster["constructor_id"].map(resolve_team_id)
    return roster.drop_duplicates(subset=["driver_id"], keep="last")


def get_driver_team_map(season: int) -> dict[str, str]:
    """Return the best-known roster mapping for a season."""

    roster = _season_roster_from_results(int(season))
    if roster.empty:
        return {}
    return roster.set_index("driver_id")["constructor_id"].to_dict()


DRIVER_TEAM_MAP: dict[int, dict[str, str]] = {
    int(season): get_driver_team_map(int(season))
    for season in sorted(_race_results()["season"].dropna().astype(int).unique().tolist())
} if not _race_results().empty else {}


def get_rookies(season: int) -> set[str]:
    """Return the known rookies for a season."""

    return set(ROOKIE_REGISTRY.get(int(season), []))


def is_rookie(driver_id: str, season: int) -> bool:
    """Return whether a driver is in the rookie registry for a season."""

    return resolve_driver_id(driver_id) in get_rookies(int(season))


def get_constructor_name(constructor_id: str) -> str:
    """Return a display-friendly constructor name."""

    profile = get_team_profile(constructor_id)
    return profile.short_name


def get_driver_name(driver_id: str) -> str:
    """Return a display-friendly driver name."""

    profile = get_driver_profile(driver_id)
    return profile.full_name


def get_season_roster(season: int) -> list[dict[str, Any]]:
    """Return the current best-known driver roster for a season."""

    roster = _season_roster_from_results(int(season))
    if roster.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in roster.itertuples(index=False):
        profile = get_driver_profile(str(row.driver_id))
        team_profile = get_team_profile(str(row.constructor_id))
        rows.append(
            {
                "season": int(season),
                "driver_id": profile.driver_id,
                "driver_name": profile.full_name,
                "driver_number": profile.number,
                "constructor_id": team_profile.team_id,
                "constructor_name": team_profile.short_name,
                "is_rookie": int(is_rookie(profile.driver_id, season)),
            }
        )
    return sorted(rows, key=lambda payload: (payload["constructor_id"], payload["driver_id"]))


def get_likely_team(driver_id: str, season: int) -> str | None:
    """Return the best-known constructor mapping for a driver-season pair."""

    resolved_driver = resolve_driver_id(driver_id)
    mapping = get_driver_team_map(int(season))
    if resolved_driver in mapping:
        return mapping[resolved_driver]
    profile = get_driver_profile(resolved_driver)
    return profile.current_team


def get_regulation_change_type(season: int) -> str | None:
    """Return the regulation change type for a season if tracked."""

    change = REGULATION_HISTORY.get(int(season))
    return change.type if change else None


def get_driver_home_circuits(driver_id: str) -> list[str]:
    """Return the list of home circuits for a driver."""

    return DRIVER_HOME_CIRCUITS.get(resolve_driver_id(driver_id), [])
