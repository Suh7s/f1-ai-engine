"""Canonical driver metadata and computed career summaries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from src.utils import CONFIG as BASE_CONFIG, load_dataframe, load_json


@dataclass(frozen=True, slots=True)
class CareerStats:
    """Computed career summary for one driver."""

    races: int
    wins: int
    podiums: int
    poles: int
    fastest_laps: int
    championships: int
    points: float


@dataclass(frozen=True, slots=True)
class DriverProfile:
    """Structured driver profile used across the platform."""

    driver_id: str
    full_name: str
    nationality: str
    date_of_birth: str
    number: int | None
    abbreviation: str
    current_team: str | None
    helmet_color: str
    car_color: str
    debut_year: int
    debut_race: str
    career_stats: CareerStats
    championships: list[int]
    notable_records: list[str]


DRIVER_ALIASES: dict[str, str] = {
    "alex_albon": "albon",
    "alexander_albon": "albon",
    "charles_leclerc": "leclerc",
    "esteban_ocon": "ocon",
    "franco_colapinto": "colapinto",
    "gabriel_bortoleto": "bortoleto",
    "george_russell": "russell",
    "isack_hadjar": "hadjar",
    "jack_doohan": "doohan",
    "kimi_antonelli": "antonelli",
    "kevin_magnussen": "kevin_magnussen",
    "lando_norris": "norris",
    "lewis_hamilton": "hamilton",
    "liam_lawson": "lawson",
    "nico_hulkenberg": "hulkenberg",
    "oliver_bearman": "bearman",
    "oscar_piastri": "piastri",
    "pierre_gasly": "gasly",
    "sergio_perez": "perez",
    "valtteri_bottas": "bottas",
    "yuki_tsunoda": "tsunoda",
    "zhou_guanyu": "zhou",
}

DRIVER_HELMET_COLORS: dict[str, str] = {
    "max_verstappen": "#1E41FF",
    "norris": "#F47B20",
    "piastri": "#FFB347",
    "hamilton": "#8C1D40",
    "leclerc": "#FF4040",
    "russell": "#00D2BE",
    "antonelli": "#00A5A5",
    "alonso": "#006F62",
    "stroll": "#1A8A74",
    "gasly": "#FF87BC",
    "colapinto": "#4DA3FF",
    "hulkenberg": "#2D2D2D",
    "bortoleto": "#74FF7A",
    "hadjar": "#4F7CFF",
    "lawson": "#EAEAEA",
    "tsunoda": "#F2F2F2",
    "bearman": "#C0C0C0",
    "ocon": "#D4D4D4",
    "sainz": "#163BBF",
    "albon": "#4DC7FF",
    "perez": "#49D084",
    "vettel": "#008A4D",
    "raikkonen": "#C00000",
    "button": "#F7D046",
    "rosberg": "#8CD9D1",
}

KNOWN_CHAMPIONSHIPS: dict[str, list[int]] = {
    "alonso": [2005, 2006],
    "button": [2009],
    "hamilton": [2008, 2014, 2015, 2017, 2018, 2019, 2020],
    "max_verstappen": [2021, 2022, 2023, 2024],
    "raikkonen": [2007],
    "rosberg": [2016],
    "vettel": [2010, 2011, 2012, 2013],
}

NOTABLE_RECORDS: dict[str, list[str]] = {
    "alonso": ["Youngest world champion of his era", "Multiple Le Mans and endurance-racing successes"],
    "button": ["2009 world champion", "Renowned for smooth wet-weather tyre management"],
    "hamilton": ["Most career wins", "Most career poles", "Joint record seven world championships"],
    "max_verstappen": ["Most wins in a single season (19 in 2023)", "Youngest Formula 1 race winner", "Most consecutive wins (10 in 2023)"],
    "norris": ["McLaren title contender in the ground-effect era", "One of the strongest tyre-preservation profiles on the current grid"],
    "piastri": ["Sprint winner before first Grand Prix win", "Elite high-speed confidence early in career"],
    "leclerc": ["Monaco Grand Prix winner", "One-lap specialist with strong pole conversion profile"],
    "russell": ["Mercedes race winner", "Strong qualifying benchmark across midfield and front-running machinery"],
    "vettel": ["Youngest double, triple and quadruple world champion at the time", "Nine consecutive wins in 2013"],
}

DEBUT_OVERRIDES: dict[str, tuple[int, str]] = {
    "aitken": (2020, "Sakhir Grand Prix 2020"),
    "albon": (2019, "Australian Grand Prix 2019"),
    "alonso": (2001, "Australian Grand Prix 2001"),
    "antonelli": (2025, "Australian Grand Prix 2025"),
    "bearman": (2024, "Saudi Arabian Grand Prix 2024"),
    "bortoleto": (2025, "Australian Grand Prix 2025"),
    "bottas": (2013, "Australian Grand Prix 2013"),
    "brendon_hartley": (2017, "United States Grand Prix 2017"),
    "button": (2000, "Australian Grand Prix 2000"),
    "colapinto": (2024, "Italian Grand Prix 2024"),
    "de_vries": (2022, "Italian Grand Prix 2022"),
    "doohan": (2024, "Abu Dhabi Grand Prix 2024"),
    "ericsson": (2014, "Australian Grand Prix 2014"),
    "gasly": (2017, "Malaysian Grand Prix 2017"),
    "giovinazzi": (2017, "Australian Grand Prix 2017"),
    "grosjean": (2009, "European Grand Prix 2009"),
    "gutierrez": (2013, "Australian Grand Prix 2013"),
    "hadjar": (2025, "Australian Grand Prix 2025"),
    "hamilton": (2007, "Australian Grand Prix 2007"),
    "haryanto": (2016, "Australian Grand Prix 2016"),
    "hulkenberg": (2010, "Bahrain Grand Prix 2010"),
    "jolyon_palmer": (2016, "Australian Grand Prix 2016"),
    "kevin_magnussen": (2014, "Australian Grand Prix 2014"),
    "kubica": (2006, "Hungarian Grand Prix 2006"),
    "kvyat": (2014, "Australian Grand Prix 2014"),
    "latifi": (2020, "Austrian Grand Prix 2020"),
    "lawson": (2023, "Dutch Grand Prix 2023"),
    "leclerc": (2018, "Australian Grand Prix 2018"),
    "massa": (2002, "Australian Grand Prix 2002"),
    "max_verstappen": (2015, "Australian Grand Prix 2015"),
    "mazepin": (2021, "Bahrain Grand Prix 2021"),
    "mick_schumacher": (2021, "Bahrain Grand Prix 2021"),
    "nasr": (2015, "Australian Grand Prix 2015"),
    "norris": (2019, "Australian Grand Prix 2019"),
    "ocon": (2016, "Belgian Grand Prix 2016"),
    "perez": (2011, "Australian Grand Prix 2011"),
    "piastri": (2023, "Bahrain Grand Prix 2023"),
    "pietro_fittipaldi": (2020, "Sakhir Grand Prix 2020"),
    "raikkonen": (2001, "Australian Grand Prix 2001"),
    "resta": (2011, "Australian Grand Prix 2011"),
    "ricciardo": (2011, "British Grand Prix 2011"),
    "rosberg": (2006, "Bahrain Grand Prix 2006"),
    "russell": (2019, "Australian Grand Prix 2019"),
    "sainz": (2015, "Australian Grand Prix 2015"),
    "sargeant": (2023, "Bahrain Grand Prix 2023"),
    "sirotkin": (2018, "Australian Grand Prix 2018"),
    "stroll": (2017, "Australian Grand Prix 2017"),
    "tsunoda": (2021, "Bahrain Grand Prix 2021"),
    "vandoorne": (2016, "Bahrain Grand Prix 2016"),
    "vettel": (2007, "United States Grand Prix 2007"),
    "wehrlein": (2016, "Australian Grand Prix 2016"),
    "zhou": (2022, "Bahrain Grand Prix 2022"),
}


def resolve_driver_id(driver_id: str) -> str:
    """Normalize aliases onto the canonical driver id used in processed data."""

    normalized = str(driver_id or "").strip().lower()
    return DRIVER_ALIASES.get(normalized, normalized)


@lru_cache(maxsize=1)
def _cached_driver_seed_frame() -> pd.DataFrame:
    """Build a driver master table from cached Jolpica race-result payloads."""

    rows: list[dict[str, Any]] = []
    raw_dir = BASE_CONFIG["raw_dir"]
    for path in sorted(raw_dir.glob("results_*.json")):
        payload = load_json(path, default={}) or {}
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                rows.append(
                    {
                        "driver_id": resolve_driver_id(driver.get("driverId")),
                        "full_name": " ".join(
                            [str(driver.get("givenName", "")).strip(), str(driver.get("familyName", "")).strip()]
                        ).strip(),
                        "date_of_birth": str(driver.get("dateOfBirth", "") or ""),
                        "nationality": str(driver.get("nationality", "") or ""),
                        "number": pd.to_numeric(driver.get("permanentNumber"), errors="coerce"),
                        "abbreviation": str(driver.get("code", "") or ""),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["driver_id", "full_name", "date_of_birth", "nationality", "number", "abbreviation"])
    frame = frame.dropna(subset=["driver_id"]).copy()
    frame["number"] = pd.to_numeric(frame["number"], errors="coerce")
    return frame.groupby("driver_id", as_index=False).agg(
        full_name=("full_name", lambda values: next((value for value in values if value), "")),
        date_of_birth=("date_of_birth", lambda values: next((value for value in values if value), "")),
        nationality=("nationality", lambda values: next((value for value in values if value), "")),
        number=("number", lambda values: next((int(value) for value in values if pd.notna(value)), pd.NA)),
        abbreviation=("abbreviation", lambda values: next((value for value in values if value), "")),
    )


def _fallback_name(driver_id: str) -> str:
    """Build a readable full name from a canonical driver id."""

    return " ".join(part.capitalize() for part in str(driver_id).split("_"))


def _career_stats(rows: pd.DataFrame, championships: list[int]) -> CareerStats:
    """Compute career aggregates from the processed results store."""

    if rows.empty:
        return CareerStats(0, 0, 0, 0, 0, len(championships), 0.0)

    points_series = rows["points"] if "points" in rows.columns else pd.Series(dtype=float)
    return CareerStats(
        races=int(rows[["season", "round"]].drop_duplicates().shape[0]),
        wins=int(rows["finish_position"].eq(1).sum()),
        podiums=int(rows["finish_position"].le(3).sum()),
        poles=int(rows.get("qualifying_position", pd.Series(dtype=float)).eq(1).sum()),
        fastest_laps=int(rows.get("fastest_lap_rank", pd.Series(dtype=float)).eq(1).sum()),
        championships=len(championships),
        points=float(pd.to_numeric(points_series, errors="coerce").fillna(0.0).sum()),
    )


def build_driver_database() -> dict[str, DriverProfile]:
    """Build the canonical driver registry from cached results plus curated overlays."""

    race_results_path = BASE_CONFIG["processed_dir"] / "race_results.csv"
    race_results = load_dataframe(race_results_path) if race_results_path.exists() else pd.DataFrame()
    driver_seed = _cached_driver_seed_frame().set_index("driver_id") if not _cached_driver_seed_frame().empty else pd.DataFrame()
    latest_team_map: dict[str, str | None] = {}
    if not race_results.empty:
        ordered = race_results.sort_values(["season", "round", "finish_position"])
        latest_rows = ordered.groupby("driver_id", as_index=False).tail(1)
        latest_team_map = latest_rows.set_index("driver_id")["constructor_id"].to_dict()

    database: dict[str, DriverProfile] = {}
    all_driver_ids = set(latest_team_map) | set(driver_seed.index.tolist())
    if not race_results.empty:
        all_driver_ids |= set(race_results["driver_id"].dropna().astype(str).tolist())

    from src.team_database import get_team_profile

    for raw_driver_id in sorted(all_driver_ids):
        driver_id = resolve_driver_id(raw_driver_id)
        rows = race_results.loc[race_results["driver_id"] == driver_id].copy() if not race_results.empty else pd.DataFrame()
        seed = driver_seed.loc[driver_id] if not driver_seed.empty and driver_id in driver_seed.index else pd.Series(dtype=object)
        full_name = str(seed.get("full_name") or "").strip()
        if not full_name and not rows.empty:
            latest_name = rows.iloc[-1]
            full_name = f"{str(latest_name.get('driver_given_name', '')).strip()} {str(latest_name.get('driver_family_name', '')).strip()}".strip()
        full_name = full_name or _fallback_name(driver_id)

        championships = list(KNOWN_CHAMPIONSHIPS.get(driver_id, []))
        latest_team = latest_team_map.get(driver_id)
        team_profile = get_team_profile(str(latest_team or ""))
        debut_row = rows.sort_values(["season", "round"]).head(1) if not rows.empty else pd.DataFrame()
        debut_year = int(debut_row["season"].iloc[0]) if not debut_row.empty else 0
        debut_race = (
            f"{str(debut_row['race_name'].iloc[0])} {debut_year}"
            if not debut_row.empty
            else "Unknown"
        )
        if driver_id in DEBUT_OVERRIDES:
            debut_year, debut_race = DEBUT_OVERRIDES[driver_id]
        abbreviation = str(seed.get("abbreviation") or "").strip()
        if not abbreviation and not rows.empty and "driver_code" in rows.columns:
            abbreviation = str(rows["driver_code"].dropna().astype(str).iloc[-1]) if rows["driver_code"].notna().any() else ""
        abbreviation = abbreviation or full_name.split()[-1][:3].upper()
        helmet_color = DRIVER_HELMET_COLORS.get(driver_id, team_profile.primary_color)
        number_value = seed.get("number")
        number = int(number_value) if pd.notna(number_value) else None

        database[driver_id] = DriverProfile(
            driver_id=driver_id,
            full_name=full_name,
            nationality=str(seed.get("nationality") or "Unknown"),
            date_of_birth=str(seed.get("date_of_birth") or ""),
            number=number,
            abbreviation=abbreviation,
            current_team=latest_team,
            helmet_color=helmet_color,
            car_color=team_profile.primary_color,
            debut_year=debut_year,
            debut_race=debut_race,
            career_stats=_career_stats(rows, championships),
            championships=championships,
            notable_records=list(NOTABLE_RECORDS.get(driver_id, [])),
        )
    return database


DRIVER_DATABASE = build_driver_database()


def get_driver_profile(driver_id: str) -> DriverProfile:
    """Return a canonical driver profile for a driver id or alias."""

    resolved = resolve_driver_id(driver_id)
    if resolved in DRIVER_DATABASE:
        return DRIVER_DATABASE[resolved]
    return DriverProfile(
        driver_id=resolved,
        full_name=_fallback_name(resolved),
        nationality="Unknown",
        date_of_birth="",
        number=None,
        abbreviation=_fallback_name(resolved).split()[-1][:3].upper(),
        current_team=None,
        helmet_color="#888888",
        car_color="#888888",
        debut_year=0,
        debut_race="Unknown",
        career_stats=CareerStats(0, 0, 0, 0, 0, 0, 0.0),
        championships=[],
        notable_records=[],
    )
