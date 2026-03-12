from __future__ import annotations

from src.driver_database import DRIVER_DATABASE, get_driver_profile, resolve_driver_id
from src.registry import get_season_roster
from src.team_database import TEAM_DATABASE, get_team_profile, resolve_team_id


def test_driver_database_has_expected_depth() -> None:
    assert len(DRIVER_DATABASE) >= 53
    hamilton = get_driver_profile("lewis_hamilton")
    assert hamilton.driver_id == "hamilton"
    assert hamilton.full_name == "Lewis Hamilton"
    assert hamilton.current_team == "ferrari"


def test_team_database_has_expected_depth() -> None:
    assert len(TEAM_DATABASE) >= 19
    red_bull = get_team_profile("red_bull")
    assert red_bull.short_name == "Red Bull"
    assert red_bull.primary_color == "#3671C6"


def test_registry_roster_uses_canonical_ids() -> None:
    roster = get_season_roster(2026)
    driver_ids = {row["driver_id"] for row in roster}
    assert "hamilton" in driver_ids
    assert "lewis_hamilton" not in driver_ids
    assert resolve_driver_id("charles_leclerc") == "leclerc"
    assert resolve_team_id("kick_sauber") == "sauber"
