"""Canonical constructor metadata for the F1 AI platform."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from src.utils import CONFIG as BASE_CONFIG, load_dataframe


@dataclass(frozen=True, slots=True)
class TeamProfile:
    """Structured constructor profile used across dashboard and pipeline flows."""

    team_id: str
    full_name: str
    short_name: str
    nationality: str
    base: str
    team_principal: str
    founded: int
    championships: list[int]
    primary_color: str
    secondary_color: str
    current_drivers: list[str]
    power_unit_supplier: str
    notes: str = ""


TEAM_ALIASES: dict[str, str] = {
    "kick_sauber": "sauber",
    "visa_cash_app_rb": "rb",
    "racing_bulls": "rb",
    "alpha_tauri": "alphatauri",
}

TEAM_SEED_DATA: dict[str, dict[str, Any]] = {
    "red_bull": {
        "full_name": "Oracle Red Bull Racing",
        "short_name": "Red Bull",
        "nationality": "Austrian",
        "base": "Milton Keynes, United Kingdom",
        "team_principal": "Christian Horner",
        "founded": 2005,
        "championships": [2010, 2011, 2012, 2013, 2022, 2023],
        "primary_color": "#3671C6",
        "secondary_color": "#CC1E4A",
        "power_unit_supplier": "Red Bull Ford Powertrains",
        "notes": "High-rake era benchmark, then dominant under the 2022 ground-effect reset.",
    },
    "ferrari": {
        "full_name": "Scuderia Ferrari HP",
        "short_name": "Ferrari",
        "nationality": "Italian",
        "base": "Maranello, Italy",
        "team_principal": "Fred Vasseur",
        "founded": 1950,
        "championships": [1961, 1964, 1975, 1976, 1977, 1979, 1982, 1983, 1999, 2000, 2001, 2002, 2003, 2004, 2007, 2008],
        "primary_color": "#E8002D",
        "secondary_color": "#FFFFFF",
        "power_unit_supplier": "Ferrari",
        "notes": "The sport's oldest and most decorated constructor.",
    },
    "mclaren": {
        "full_name": "McLaren Formula 1 Team",
        "short_name": "McLaren",
        "nationality": "British",
        "base": "Woking, United Kingdom",
        "team_principal": "Andrea Stella",
        "founded": 1966,
        "championships": [1974, 1984, 1985, 1988, 1989, 1990, 1991, 1998, 2024],
        "primary_color": "#FF8000",
        "secondary_color": "#47C7FC",
        "power_unit_supplier": "Mercedes",
        "notes": "Returned to the front in the current cost-cap era with strong development velocity.",
    },
    "mercedes": {
        "full_name": "Mercedes-AMG PETRONAS Formula One Team",
        "short_name": "Mercedes",
        "nationality": "German",
        "base": "Brackley, United Kingdom",
        "team_principal": "Toto Wolff",
        "founded": 2010,
        "championships": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        "primary_color": "#27F4D2",
        "secondary_color": "#6CD3BF",
        "power_unit_supplier": "Mercedes",
        "notes": "Turbo-hybrid dynasty with the strongest sustained championship run of the modern era.",
    },
    "aston_martin": {
        "full_name": "Aston Martin Aramco Formula One Team",
        "short_name": "Aston Martin",
        "nationality": "British",
        "base": "Silverstone, United Kingdom",
        "team_principal": "Andy Cowell",
        "founded": 2021,
        "championships": [],
        "primary_color": "#229971",
        "secondary_color": "#00A19C",
        "power_unit_supplier": "Mercedes",
        "notes": "The modern Silverstone-based outfit traces its roots through Jordan, Midland, Spyker, Force India and Racing Point.",
    },
    "alpine": {
        "full_name": "BWT Alpine Formula One Team",
        "short_name": "Alpine",
        "nationality": "French",
        "base": "Enstone, United Kingdom",
        "team_principal": "Oliver Oakes",
        "founded": 2021,
        "championships": [],
        "primary_color": "#2293D1",
        "secondary_color": "#FF87BC",
        "power_unit_supplier": "Renault",
        "notes": "Factory Renault successor centered around the Enstone chassis operation.",
    },
    "sauber": {
        "full_name": "Stake F1 Team Kick Sauber",
        "short_name": "Sauber",
        "nationality": "Swiss",
        "base": "Hinwil, Switzerland",
        "team_principal": "Mattia Binotto",
        "founded": 1993,
        "championships": [],
        "primary_color": "#52E252",
        "secondary_color": "#0B0D12",
        "power_unit_supplier": "Ferrari",
        "notes": "Legacy Sauber identity used for the transition period before the Audi works entry.",
    },
    "audi": {
        "full_name": "Audi Formula 1 Team",
        "short_name": "Audi",
        "nationality": "German",
        "base": "Hinwil, Switzerland",
        "team_principal": "Mattia Binotto",
        "founded": 2026,
        "championships": [],
        "primary_color": "#B30F24",
        "secondary_color": "#E6E6E6",
        "power_unit_supplier": "Audi",
        "notes": "Works entry built on the Sauber chassis operation with Audi power units.",
    },
    "rb": {
        "full_name": "Visa Cash App Racing Bulls Formula One Team",
        "short_name": "Racing Bulls",
        "nationality": "Italian",
        "base": "Faenza, Italy",
        "team_principal": "Laurent Mekies",
        "founded": 2024,
        "championships": [],
        "primary_color": "#6692FF",
        "secondary_color": "#FFFFFF",
        "power_unit_supplier": "Red Bull Powertrains",
        "notes": "Faenza-based Red Bull sister team operating under the Racing Bulls identity.",
    },
    "alphatauri": {
        "full_name": "Scuderia AlphaTauri",
        "short_name": "AlphaTauri",
        "nationality": "Italian",
        "base": "Faenza, Italy",
        "team_principal": "Franz Tost",
        "founded": 2020,
        "championships": [],
        "primary_color": "#4E7C9B",
        "secondary_color": "#FFFFFF",
        "power_unit_supplier": "Honda",
        "notes": "Short-lived branding era between Toro Rosso and Racing Bulls.",
    },
    "toro_rosso": {
        "full_name": "Scuderia Toro Rosso",
        "short_name": "Toro Rosso",
        "nationality": "Italian",
        "base": "Faenza, Italy",
        "team_principal": "Franz Tost",
        "founded": 2006,
        "championships": [],
        "primary_color": "#20394C",
        "secondary_color": "#AAB6C3",
        "power_unit_supplier": "Ferrari",
        "notes": "Red Bull junior team precursor to AlphaTauri and Racing Bulls.",
    },
    "haas": {
        "full_name": "MoneyGram Haas F1 Team",
        "short_name": "Haas",
        "nationality": "American",
        "base": "Kannapolis, United States / Banbury, United Kingdom",
        "team_principal": "Ayao Komatsu",
        "founded": 2016,
        "championships": [],
        "primary_color": "#B6BABD",
        "secondary_color": "#E10600",
        "power_unit_supplier": "Ferrari",
        "notes": "Ferrari-powered independent constructor with a lean, partnership-heavy operating model.",
    },
    "williams": {
        "full_name": "Atlassian Williams Racing",
        "short_name": "Williams",
        "nationality": "British",
        "base": "Grove, United Kingdom",
        "team_principal": "James Vowles",
        "founded": 1977,
        "championships": [1980, 1981, 1986, 1987, 1992, 1993, 1994, 1996, 1997],
        "primary_color": "#37BEDD",
        "secondary_color": "#005AFF",
        "power_unit_supplier": "Mercedes",
        "notes": "Historic champion rebuilding under a modern technical structure.",
    },
    "force_india": {
        "full_name": "Sahara Force India F1 Team",
        "short_name": "Force India",
        "nationality": "Indian",
        "base": "Silverstone, United Kingdom",
        "team_principal": "Otmar Szafnauer",
        "founded": 2008,
        "championships": [],
        "primary_color": "#F596C8",
        "secondary_color": "#FFFFFF",
        "power_unit_supplier": "Mercedes",
        "notes": "Efficient midfield benchmark that later evolved into Racing Point and Aston Martin.",
    },
    "racing_point": {
        "full_name": "BWT Racing Point F1 Team",
        "short_name": "Racing Point",
        "nationality": "British",
        "base": "Silverstone, United Kingdom",
        "team_principal": "Otmar Szafnauer",
        "founded": 2018,
        "championships": [],
        "primary_color": "#F596C8",
        "secondary_color": "#1D1F2B",
        "power_unit_supplier": "Mercedes",
        "notes": "Transition identity between Force India and Aston Martin.",
    },
    "alfa": {
        "full_name": "Alfa Romeo F1 Team ORLEN",
        "short_name": "Alfa Romeo",
        "nationality": "Swiss",
        "base": "Hinwil, Switzerland",
        "team_principal": "Alessandro Alunni Bravi",
        "founded": 2019,
        "championships": [],
        "primary_color": "#900000",
        "secondary_color": "#FFFFFF",
        "power_unit_supplier": "Ferrari",
        "notes": "Sauber-operated entry under the Alfa Romeo branding cycle.",
    },
    "renault": {
        "full_name": "Renault Sport Formula One Team",
        "short_name": "Renault",
        "nationality": "French",
        "base": "Enstone, United Kingdom",
        "team_principal": "Cyril Abiteboul",
        "founded": 2016,
        "championships": [],
        "primary_color": "#FFF500",
        "secondary_color": "#000000",
        "power_unit_supplier": "Renault",
        "notes": "Works re-entry after Lotus, preceding the Alpine rebrand.",
    },
    "manor": {
        "full_name": "Manor Racing MRT",
        "short_name": "Manor",
        "nationality": "British",
        "base": "Banbury, United Kingdom",
        "team_principal": "Dave Ryan",
        "founded": 2010,
        "championships": [],
        "primary_color": "#6B7280",
        "secondary_color": "#FF6B35",
        "power_unit_supplier": "Mercedes",
        "notes": "Last iteration of the Marussia/Manor backmarker program.",
    },
    "cadillac": {
        "full_name": "Cadillac Formula 1 Team",
        "short_name": "Cadillac",
        "nationality": "American",
        "base": "Silverstone, United Kingdom",
        "team_principal": "Graeme Lowdon",
        "founded": 2026,
        "championships": [],
        "primary_color": "#0A7CFF",
        "secondary_color": "#D9D9D9",
        "power_unit_supplier": "Ferrari",
        "notes": "GM-backed entry beginning as a Ferrari customer before its long-term works plan matures.",
    },
}


def resolve_team_id(team_id: str) -> str:
    """Normalize aliases onto the canonical constructor id used by the platform."""

    normalized = str(team_id or "").strip().lower()
    return TEAM_ALIASES.get(normalized, normalized)


@lru_cache(maxsize=1)
def _latest_current_drivers() -> dict[str, list[str]]:
    """Infer the latest known roster for each constructor from processed race results."""

    path = BASE_CONFIG["processed_dir"] / "race_results.csv"
    if not path.exists():
        return {}

    frame = load_dataframe(path)
    if frame.empty:
        return {}

    season = int(frame["season"].max())
    season_frame = frame.loc[frame["season"] == season].copy()
    if season_frame.empty:
        return {}
    round_number = int(season_frame["round"].max())
    latest = season_frame.loc[season_frame["round"] == round_number, ["constructor_id", "driver_id"]].drop_duplicates()
    latest["constructor_id"] = latest["constructor_id"].map(resolve_team_id)
    return {
        constructor_id: sorted(group["driver_id"].dropna().astype(str).unique().tolist())
        for constructor_id, group in latest.groupby("constructor_id", sort=True)
    }


def build_team_database() -> dict[str, TeamProfile]:
    """Build the canonical constructor registry with latest inferred lineups."""

    latest_rosters = _latest_current_drivers()
    database: dict[str, TeamProfile] = {}
    for team_id, payload in TEAM_SEED_DATA.items():
        database[team_id] = TeamProfile(
            team_id=team_id,
            full_name=str(payload["full_name"]),
            short_name=str(payload["short_name"]),
            nationality=str(payload["nationality"]),
            base=str(payload["base"]),
            team_principal=str(payload["team_principal"]),
            founded=int(payload["founded"]),
            championships=[int(year) for year in payload["championships"]],
            primary_color=str(payload["primary_color"]),
            secondary_color=str(payload["secondary_color"]),
            current_drivers=latest_rosters.get(team_id, []),
            power_unit_supplier=str(payload["power_unit_supplier"]),
            notes=str(payload.get("notes", "")),
        )
    return database


TEAM_DATABASE = build_team_database()


def get_team_profile(team_id: str) -> TeamProfile:
    """Return the canonical team profile for a constructor id or alias."""

    resolved = resolve_team_id(team_id)
    if resolved in TEAM_DATABASE:
        return TEAM_DATABASE[resolved]
    return TeamProfile(
        team_id=resolved,
        full_name=resolved.replace("_", " ").title(),
        short_name=resolved.replace("_", " ").title(),
        nationality="Unknown",
        base="Unknown",
        team_principal="Unknown",
        founded=0,
        championships=[],
        primary_color="#888888",
        secondary_color="#BBBBBB",
        current_drivers=[],
        power_unit_supplier="Unknown",
    )


def get_team_palette(team_id: str) -> tuple[str, str]:
    """Return the primary and secondary livery colors for a constructor."""

    profile = get_team_profile(team_id)
    return profile.primary_color, profile.secondary_color
