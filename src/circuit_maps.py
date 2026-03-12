"""Circuit map registry, vector asset caching, and coordinate transforms."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils import CONFIG as BASE_CONFIG, ensure_directories, get_logger, load_json, save_json, slugify


LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LapRecord:
    """Official or platform-tracked circuit lap record."""

    time: str
    driver: str
    year: int


@dataclass(frozen=True, slots=True)
class CircuitInfo:
    """Canonical metadata and rendering hints for a circuit."""

    circuit_id: str
    gp_name: str
    full_name: str
    country: str
    city: str
    first_gp: int
    lap_length_km: float
    laps: int
    lap_record: LapRecord
    drs_zones: int
    svg_path: str
    openf1_circuit_key: str
    timezone: str
    coordinates: tuple[float, float]
    sector_breaks: tuple[float, float] = (0.33, 0.66)
    drs_zone_spans: tuple[tuple[float, float], ...] = ()
    pit_lane_span: tuple[float, float] | None = (0.93, 1.0)
    cache_terms: tuple[str, ...] = ()
    track_rotation_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """Affine-ish transform plus source bounds for mapping telemetry to the SVG viewBox."""

    source_min_x: float
    source_max_x: float
    source_min_y: float
    source_max_y: float
    target_min_x: float = 60.0
    target_max_x: float = 1140.0
    target_min_y: float = 60.0
    target_max_y: float = 640.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    flip_x: bool = False
    flip_y: bool = True


RAW_CIRCUITS_DIR = BASE_CONFIG["raw_dir"] / "circuits"
FASTF1_CACHE_DIR = BASE_CONFIG["raw_dir"] / "fastf1_cache"

CIRCUIT_ALIASES: dict[str, str] = {
    "autodromo_enzo_e_dino_ferrari": "imola",
    "jose_carlos_pace": "interlagos",
    "las_vegas": "vegas",
    "lusail": "losail",
    "paul_ricard": "ricard",
    "spielberg": "red_bull_ring",
}

BACINGER_MAPPING = {
    "albert_park": "au-1953",
    "americas": "us-2012",
    "baku": "az-2016",
    "bahrain": "bh-2002",
    "catalunya": "es-1991",
    "hockenheimring": "de-1932",
    "hungaroring": "hu-1986",
    "imola": "it-1953",
    "interlagos": "br-1940",
    "istanbul": "tr-2005",
    "jeddah": "sa-2021",
    "vegas": "us-2023",
    "losail": "qa-2004",
    "marina_bay": "sg-2008",
    "miami": "us-2022",
    "monaco": "mc-1929",
    "monza": "it-1922",
    "mugello": "it-1914",
    "nurburgring": "de-1927",
    "portimao": "pt-2008",
    "red_bull_ring": "at-1969",
    "ricard": "fr-1969",
    "rodriguez": "mx-1962",
    "sepang": "my-1999",
    "shanghai": "cn-2004",
    "silverstone": "gb-1948",
    "sochi": "ru-2014",
    "spa": "be-1925",
    "suzuka": "jp-1962",
    "villeneuve": "ca-1978",
    "yas_marina": "ae-2009",
    "zandvoort": "nl-1948",
}


def _circuit_svg_path(circuit_id: str) -> str:
    return str(RAW_CIRCUITS_DIR / f"{circuit_id}.svg")


def _circuit_geojson_path(circuit_id: str) -> Path:
    return RAW_CIRCUITS_DIR / f"{circuit_id}.geojson"


def _circuit(
    circuit_id: str,
    gp_name: str,
    full_name: str,
    country: str,
    city: str,
    first_gp: int,
    lap_length_km: float,
    laps: int,
    lap_record: tuple[str, str, int],
    drs_zones: int,
    timezone: str,
    coordinates: tuple[float, float],
    sector_breaks: tuple[float, float] = (0.33, 0.66),
    drs_zone_spans: tuple[tuple[float, float], ...] = (),
    pit_lane_span: tuple[float, float] | None = (0.93, 1.0),
) -> CircuitInfo:
    return CircuitInfo(
        circuit_id=circuit_id,
        gp_name=gp_name,
        full_name=full_name,
        country=country,
        city=city,
        first_gp=first_gp,
        lap_length_km=lap_length_km,
        laps=laps,
        lap_record=LapRecord(*lap_record),
        drs_zones=drs_zones,
        svg_path=_circuit_svg_path(circuit_id),
        openf1_circuit_key=circuit_id,
        timezone=timezone,
        coordinates=coordinates,
        sector_breaks=sector_breaks,
        drs_zone_spans=drs_zone_spans,
        pit_lane_span=pit_lane_span,
        cache_terms=(gp_name, full_name, circuit_id),
    )


_REGISTRY: dict[str, CircuitInfo] = {
    "albert_park": _circuit("albert_park", "Australian Grand Prix", "Albert Park Circuit", "Australia", "Melbourne", 1996, 5.278, 58, ("1:19.813", "Charles Leclerc", 2024), 4, "Australia/Melbourne", (-37.8497, 144.9680), drs_zone_spans=((0.08, 0.14), (0.28, 0.33), (0.57, 0.62), (0.82, 0.88))),
    "americas": _circuit("americas", "United States Grand Prix", "Circuit of the Americas", "United States", "Austin", 2012, 5.513, 56, ("1:36.169", "Charles Leclerc", 2019), 2, "America/Chicago", (30.1328, -97.6411), drs_zone_spans=((0.08, 0.15), (0.58, 0.67))),
    "bahrain": _circuit("bahrain", "Bahrain Grand Prix", "Bahrain International Circuit", "Bahrain", "Sakhir", 2004, 5.412, 57, ("1:31.447", "Pedro de la Rosa", 2005), 3, "Asia/Bahrain", (26.0325, 50.5106), drs_zone_spans=((0.03, 0.10), (0.36, 0.43), (0.71, 0.77))),
    "baku": _circuit("baku", "Azerbaijan Grand Prix", "Baku City Circuit", "Azerbaijan", "Baku", 2016, 6.003, 51, ("1:43.009", "Charles Leclerc", 2019), 2, "Asia/Baku", (40.3725, 49.8533), drs_zone_spans=((0.03, 0.10), (0.53, 0.63))),
    "catalunya": _circuit("catalunya", "Spanish Grand Prix", "Circuit de Barcelona-Catalunya", "Spain", "Montmelo", 1991, 4.657, 66, ("1:16.330", "Max Verstappen", 2023), 2, "Europe/Madrid", (41.5700, 2.2611), drs_zone_spans=((0.08, 0.15), (0.62, 0.71))),
    "hockenheimring": _circuit("hockenheimring", "German Grand Prix", "Hockenheimring Baden-Wurttemberg", "Germany", "Hockenheim", 1970, 4.574, 67, ("1:13.780", "Kimi Raikkonen", 2004), 2, "Europe/Berlin", (49.3278, 8.5656), drs_zone_spans=((0.12, 0.20), (0.54, 0.62))),
    "hungaroring": _circuit("hungaroring", "Hungarian Grand Prix", "Hungaroring", "Hungary", "Mogyorod", 1986, 4.381, 70, ("1:16.627", "Lewis Hamilton", 2020), 2, "Europe/Budapest", (47.5789, 19.2486), drs_zone_spans=((0.07, 0.15), (0.63, 0.70))),
    "imola": _circuit("imola", "Emilia Romagna Grand Prix", "Autodromo Enzo e Dino Ferrari", "Italy", "Imola", 1980, 4.909, 63, ("1:15.484", "Lewis Hamilton", 2020), 1, "Europe/Rome", (44.3439, 11.7167), drs_zone_spans=((0.06, 0.13),)),
    "interlagos": _circuit("interlagos", "Sao Paulo Grand Prix", "Autodromo Jose Carlos Pace", "Brazil", "Sao Paulo", 1973, 4.309, 71, ("1:10.540", "Valtteri Bottas", 2018), 2, "America/Sao_Paulo", (-23.7036, -46.6997), drs_zone_spans=((0.03, 0.10), (0.74, 0.82))),
    "istanbul": _circuit("istanbul", "Turkish Grand Prix", "Istanbul Park", "Turkey", "Istanbul", 2005, 5.338, 58, ("1:24.770", "Juan Pablo Montoya", 2005), 2, "Europe/Istanbul", (40.9517, 29.0500), drs_zone_spans=((0.08, 0.15), (0.57, 0.65))),
    "jeddah": _circuit("jeddah", "Saudi Arabian Grand Prix", "Jeddah Corniche Circuit", "Saudi Arabia", "Jeddah", 2021, 6.174, 50, ("1:30.734", "Lewis Hamilton", 2021), 3, "Asia/Riyadh", (21.6319, 39.1044), drs_zone_spans=((0.06, 0.12), (0.37, 0.43), (0.71, 0.78))),
    "losail": _circuit("losail", "Qatar Grand Prix", "Lusail International Circuit", "Qatar", "Lusail", 2021, 5.419, 57, ("1:22.384", "Lando Norris", 2024), 1, "Asia/Qatar", (25.4900, 51.4542), drs_zone_spans=((0.08, 0.16),)),
    "marina_bay": _circuit("marina_bay", "Singapore Grand Prix", "Marina Bay Street Circuit", "Singapore", "Singapore", 2008, 4.928, 62, ("1:34.486", "Daniel Ricciardo", 2024), 4, "Asia/Singapore", (1.2914, 103.8640), drs_zone_spans=((0.07, 0.12), (0.26, 0.31), (0.55, 0.60), (0.83, 0.88))),
    "miami": _circuit("miami", "Miami Grand Prix", "Miami International Autodrome", "United States", "Miami Gardens", 2022, 5.412, 57, ("1:29.708", "Max Verstappen", 2023), 3, "America/New_York", (25.9581, -80.2389), drs_zone_spans=((0.03, 0.10), (0.39, 0.48), (0.72, 0.80))),
    "monaco": _circuit("monaco", "Monaco Grand Prix", "Circuit de Monaco", "Monaco", "Monte Carlo", 1929, 3.337, 78, ("1:12.909", "Lewis Hamilton", 2021), 1, "Europe/Monaco", (43.7347, 7.4206), drs_zone_spans=((0.05, 0.10),)),
    "monza": _circuit("monza", "Italian Grand Prix", "Autodromo Nazionale Monza", "Italy", "Monza", 1950, 5.793, 53, ("1:21.046", "Rubens Barrichello", 2004), 2, "Europe/Rome", (45.6156, 9.2811), drs_zone_spans=((0.04, 0.12), (0.53, 0.61))),
    "mugello": _circuit("mugello", "Tuscan Grand Prix", "Autodromo Internazionale del Mugello", "Italy", "Scarperia e San Piero", 2020, 5.245, 59, ("1:18.833", "Lewis Hamilton", 2020), 1, "Europe/Rome", (43.9975, 11.3719), drs_zone_spans=((0.08, 0.15),)),
    "nurburgring": _circuit("nurburgring", "Eifel Grand Prix", "Nurburgring", "Germany", "Nurburg", 1951, 5.148, 60, ("1:28.139", "Max Verstappen", 2020), 1, "Europe/Berlin", (50.3356, 6.9475), drs_zone_spans=((0.08, 0.15),)),
    "portimao": _circuit("portimao", "Portuguese Grand Prix", "Autodromo Internacional do Algarve", "Portugal", "Portimao", 2020, 4.653, 66, ("1:18.750", "Lewis Hamilton", 2020), 1, "Europe/Lisbon", (37.2276, -8.6270), drs_zone_spans=((0.07, 0.15),)),
    "red_bull_ring": _circuit("red_bull_ring", "Austrian Grand Prix", "Red Bull Ring", "Austria", "Spielberg", 1970, 4.318, 71, ("1:05.619", "Carlos Sainz", 2020), 3, "Europe/Vienna", (47.2197, 14.7647), drs_zone_spans=((0.02, 0.09), (0.34, 0.41), (0.69, 0.76))),
    "ricard": _circuit("ricard", "French Grand Prix", "Circuit Paul Ricard", "France", "Le Castellet", 1971, 5.842, 53, ("1:32.740", "Sebastian Vettel", 2019), 2, "Europe/Paris", (43.2506, 5.7913), drs_zone_spans=((0.06, 0.15), (0.49, 0.58))),
    "rodriguez": _circuit("rodriguez", "Mexico City Grand Prix", "Autodromo Hermanos Rodriguez", "Mexico", "Mexico City", 1963, 4.304, 71, ("1:17.774", "Valtteri Bottas", 2021), 3, "America/Mexico_City", (19.4042, -99.0907), drs_zone_spans=((0.03, 0.12), (0.29, 0.35), (0.68, 0.74))),
    "sepang": _circuit("sepang", "Malaysian Grand Prix", "Sepang International Circuit", "Malaysia", "Sepang", 1999, 5.543, 56, ("1:34.223", "Juan Pablo Montoya", 2004), 2, "Asia/Kuala_Lumpur", (2.7608, 101.7380), drs_zone_spans=((0.07, 0.16), (0.55, 0.64))),
    "shanghai": _circuit("shanghai", "Chinese Grand Prix", "Shanghai International Circuit", "China", "Shanghai", 2004, 5.451, 56, ("1:32.238", "Michael Schumacher", 2004), 2, "Asia/Shanghai", (31.3389, 121.2200), drs_zone_spans=((0.04, 0.12), (0.55, 0.64))),
    "silverstone": _circuit("silverstone", "British Grand Prix", "Silverstone Circuit", "United Kingdom", "Silverstone", 1950, 5.891, 52, ("1:27.097", "Max Verstappen", 2020), 2, "Europe/London", (52.0786, -1.0169), drs_zone_spans=((0.06, 0.13), (0.50, 0.59))),
    "sochi": _circuit("sochi", "Russian Grand Prix", "Sochi Autodrom", "Russia", "Sochi", 2014, 5.848, 53, ("1:35.761", "Lewis Hamilton", 2019), 2, "Europe/Moscow", (43.4057, 39.9578), drs_zone_spans=((0.03, 0.10), (0.49, 0.58))),
    "spa": _circuit("spa", "Belgian Grand Prix", "Circuit de Spa-Francorchamps", "Belgium", "Stavelot", 1950, 7.004, 44, ("1:46.286", "Valtteri Bottas", 2018), 2, "Europe/Brussels", (50.4372, 5.9714), drs_zone_spans=((0.04, 0.12), (0.54, 0.62))),
    "suzuka": _circuit("suzuka", "Japanese Grand Prix", "Suzuka International Racing Course", "Japan", "Suzuka", 1987, 5.807, 53, ("1:30.983", "Lewis Hamilton", 2019), 1, "Asia/Tokyo", (34.8431, 136.5414), drs_zone_spans=((0.07, 0.16),)),
    "vegas": _circuit("vegas", "Las Vegas Grand Prix", "Las Vegas Strip Circuit", "United States", "Las Vegas", 2023, 6.201, 50, ("1:34.876", "Lando Norris", 2024), 2, "America/Los_Angeles", (36.1147, -115.1728), drs_zone_spans=((0.05, 0.13), (0.49, 0.59))),
    "villeneuve": _circuit("villeneuve", "Canadian Grand Prix", "Circuit Gilles Villeneuve", "Canada", "Montreal", 1978, 4.361, 70, ("1:13.078", "Valtteri Bottas", 2019), 3, "America/Toronto", (45.5006, -73.5228), drs_zone_spans=((0.04, 0.10), (0.34, 0.40), (0.71, 0.78))),
    "yas_marina": _circuit("yas_marina", "Abu Dhabi Grand Prix", "Yas Marina Circuit", "United Arab Emirates", "Abu Dhabi", 2009, 5.281, 58, ("1:26.103", "Max Verstappen", 2021), 2, "Asia/Dubai", (24.4672, 54.6031), drs_zone_spans=((0.06, 0.14), (0.49, 0.57))),
    "zandvoort": _circuit("zandvoort", "Dutch Grand Prix", "Circuit Zandvoort", "Netherlands", "Zandvoort", 1952, 4.259, 72, ("1:11.097", "Lewis Hamilton", 2021), 2, "Europe/Amsterdam", (52.3888, 4.5409), drs_zone_spans=((0.04, 0.12), (0.54, 0.63))),
}


_REGISTRY["lusail"] = _REGISTRY["losail"]

CIRCUIT_REGISTRY = _REGISTRY

CIRCUIT_TRANSFORMS: dict[str, dict[str, Any]] = {
    circuit_id: asdict(
        CoordinateTransform(
            source_min_x=0.0,
            source_max_x=1200.0,
            source_min_y=0.0,
            source_max_y=700.0,
            rotation=info.track_rotation_deg,
        )
    )
    for circuit_id, info in CIRCUIT_REGISTRY.items()
}


def resolve_circuit_id(circuit_id: str) -> str:
    """Normalize aliases onto the canonical circuit id used by the platform."""

    normalized = str(circuit_id or "").strip().lower()
    return CIRCUIT_ALIASES.get(normalized, normalized)


def circuit_transform_path(circuit_id: str) -> Path:
    """Return the persisted transform JSON path for a circuit."""

    resolved = resolve_circuit_id(circuit_id)
    return RAW_CIRCUITS_DIR / f"{resolved}_transforms.json"


def _normalize_path_label(path: Path) -> str:
    try:
        relative = path.relative_to(FASTF1_CACHE_DIR)
    except ValueError:
        relative = Path(*path.parts[-4:])
    return slugify("/".join(relative.parts))


def _path_search_tokens(info: CircuitInfo) -> set[str]:
    tokens = {slugify(info.circuit_id), slugify(info.gp_name), slugify(info.full_name)}
    if "grand_prix" in slugify(info.gp_name):
        tokens.add(slugify(info.gp_name.replace("Grand Prix", "")))
    return {token for token in tokens if token}


def _load_fastf1_position_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = pd.read_pickle(path)
    except Exception as exc:
        LOGGER.debug("Non-critical error loading FastF1 position payload %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _trace_score(frame: pd.DataFrame) -> tuple[int, int]:
    if frame.empty or not {"X", "Y"}.issubset(frame.columns):
        return (-1, -1)
    working = frame[["X", "Y"]].dropna().copy()
    nonzero = working.loc[(working["X"] != 0) | (working["Y"] != 0)]
    return (int(nonzero.drop_duplicates().shape[0]), int(len(nonzero)))


def _representative_trace(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data", {})
    best = pd.DataFrame()
    best_score = (-1, -1)
    for frame in data.values():
        if not isinstance(frame, pd.DataFrame):
            continue
        if {"X", "Y"}.issubset(frame.columns):
            candidate = frame[["Date", "X", "Y"]].copy() if "Date" in frame.columns else frame[["X", "Y"]].copy()
            score = _trace_score(candidate)
            if score > best_score:
                best = candidate
                best_score = score
    return best


def _extract_loop(trace: pd.DataFrame) -> pd.DataFrame:
    coords = trace.copy()
    coords = coords.rename(columns={"X": "x", "Y": "y"})
    if "x" in coords.columns and "y" in coords.columns:
        coords["x"] = pd.to_numeric(coords["x"], errors="coerce")
        coords["y"] = pd.to_numeric(coords["y"], errors="coerce")
    coords = coords.dropna(subset=["x", "y"]).reset_index(drop=True)
    if len(coords) < 20:
        return coords[["x", "y"]]
    if "Date" in coords.columns:
        coords = coords.sort_values("Date").reset_index(drop=True)
    start = coords.loc[0, ["x", "y"]].to_numpy(dtype=float)
    initial_distances = np.sqrt((coords["x"] - start[0]) ** 2 + (coords["y"] - start[1]) ** 2)
    initial_motion = np.where(initial_distances > 25.0)[0]
    if len(initial_motion) > 0:
        coords = coords.iloc[max(int(initial_motion[0]) - 5, 0) :].reset_index(drop=True)
    if len(coords) < 20:
        return coords[["x", "y"]]
    coords[["x", "y"]] = coords[["x", "y"]].rolling(window=9, min_periods=1, center=True).mean()
    start = coords.loc[0, ["x", "y"]].to_numpy(dtype=float)
    distances = np.sqrt((coords["x"] - start[0]) ** 2 + (coords["y"] - start[1]) ** 2)
    span = max(float(coords["x"].max() - coords["x"].min()), float(coords["y"].max() - coords["y"].min()), 1.0)
    threshold = max(span * 0.03, 25.0)
    moved_mask = distances > threshold
    candidates = np.where((distances < threshold * 0.75) & moved_mask.cummax())[0]
    loop_end = next((int(index) for index in candidates if index > max(200, len(coords) // 6)), None)
    if loop_end is not None:
        coords = coords.iloc[: loop_end + 1].reset_index(drop=True)
    step = max(1, len(coords) // 900)
    coords = coords.iloc[::step].reset_index(drop=True)
    if not coords.empty and not np.allclose(coords.iloc[0][["x", "y"]].to_numpy(dtype=float), coords.iloc[-1][["x", "y"]].to_numpy(dtype=float)):
        coords = pd.concat([coords, coords.head(1)], ignore_index=True)
    return coords[["x", "y"]]


def _placeholder_track(circuit_id: str) -> pd.DataFrame:
    seed = abs(hash(resolve_circuit_id(circuit_id))) % 360
    theta = np.linspace(0.0, 2.0 * math.pi, 720)
    radius_x = 1.0 + 0.18 * np.sin(3.0 * theta + math.radians(seed))
    radius_y = 0.72 + 0.14 * np.cos(4.0 * theta + math.radians(seed // 2))
    x = radius_x * np.cos(theta)
    y = radius_y * np.sin(theta)
    return pd.DataFrame({"x": x, "y": y})


def _iter_fastf1_position_candidates(info: CircuitInfo) -> list[Path]:
    if not FASTF1_CACHE_DIR.exists():
        return []
    search_tokens = _path_search_tokens(info)
    matches: list[Path] = []
    for candidate in sorted(FASTF1_CACHE_DIR.rglob("position_data.ff1pkl"), reverse=True):
        label = _normalize_path_label(candidate)
        if any(token in label for token in search_tokens):
            matches.append(candidate)
    return matches


def _normalize_to_viewbox(points: pd.DataFrame) -> tuple[pd.DataFrame, CoordinateTransform]:
    source_min_x = float(points["x"].min())
    source_max_x = float(points["x"].max())
    source_min_y = float(points["y"].min())
    source_max_y = float(points["y"].max())
    source_width = max(source_max_x - source_min_x, 1e-6)
    source_height = max(source_max_y - source_min_y, 1e-6)
    target_min_x, target_max_x = 60.0, 1140.0
    target_min_y, target_max_y = 60.0, 640.0
    normalized = points.copy()
    normalized["x"] = target_min_x + ((points["x"] - source_min_x) / source_width) * (target_max_x - target_min_x)
    normalized["y"] = target_max_y - ((points["y"] - source_min_y) / source_height) * (target_max_y - target_min_y)
    transform = CoordinateTransform(
        source_min_x=source_min_x,
        source_max_x=source_max_x,
        source_min_y=source_min_y,
        source_max_y=source_max_y,
        target_min_x=target_min_x,
        target_max_x=target_max_x,
        target_min_y=target_min_y,
        target_max_y=target_max_y,
    )
    return normalized, transform


def _write_svg(points: pd.DataFrame, output_path: Path) -> CoordinateTransform:
    normalized, transform = _normalize_to_viewbox(points)
    point_string = " ".join(f"{row.x:.2f},{row.y:.2f}" for row in normalized.itertuples(index=False))
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700">'
        '<rect width="1200" height="700" fill="none" />'
        f'<polyline points="{point_string}" fill="none" stroke="#ffffff" stroke-width="18" '
        'stroke-linecap="round" stroke-linejoin="round" />'
        "</svg>"
    )
    output_path.write_text(svg, encoding="utf-8")
    return transform


def _existing_svg_requires_rebuild(svg_path: Path, transform_path: Path, info: CircuitInfo) -> bool:
    if not svg_path.exists() or not transform_path.exists():
        return True
    coords = parse_svg_to_coordinates(svg_path)
    if coords.empty:
        return True
    transform = load_json(transform_path, default={}) or {}
    geometry_source = str(transform.get("geometry_source", "")).strip().lower()
    if geometry_source == "remote":
        return False
    if resolve_circuit_id(info.circuit_id) in BACINGER_MAPPING:
        return True
    if geometry_source in {"fastf1_cache", "openf1_location"}:
        return False
    return len(coords) == 720 and bool(_iter_fastf1_position_candidates(info))


def _fetch_openf1_location_trace(info: CircuitInfo) -> pd.DataFrame:
    coverage_path = BASE_CONFIG["processed_dir"] / "data_universe_coverage.csv"
    openf1_dir = BASE_CONFIG["raw_dir"] / "openf1"

    if coverage_path.exists() and openf1_dir.exists():
        try:
            coverage = pd.read_csv(coverage_path)
            matches = coverage[coverage["circuit_id"] == info.circuit_id]
            for _, match in matches.sort_values(by=["season", "round"], ascending=False).iterrows():
                season = int(match["season"])
                round_num = int(match["round"])
                session_name = str(match["session_name"])
                live_path = openf1_dir / str(season) / f"R{round_num:02d}" / f"{slugify(session_name)}_location_live.parquet"
                if live_path.exists():
                    df = pd.read_parquet(live_path, columns=["x", "y", "driver_number", "date"])
                    for _, group in df.groupby("driver_number"):
                            if len(group) > 500:
                                return group.sort_values("date")[["x", "y"]].copy()
        except Exception as exc:
            LOGGER.debug("Non-critical error reading OpenF1 trace coverage for %s: %s", info.circuit_id, exc)
    return pd.DataFrame()

def _frame_from_geojson_payload(payload: dict[str, Any]) -> pd.DataFrame:
    geo = payload.get("geometry", {})
    if not geo and payload.get("type") == "FeatureCollection" and payload.get("features"):
        geo = payload["features"][0].get("geometry", {})
    coordinates = geo.get("coordinates", [])
    if not coordinates:
        return pd.DataFrame()
    if geo.get("type") == "LineString":
        flat = coordinates
    else:
        flat = coordinates[0]
    if not flat or not isinstance(flat[0], list):
        return pd.DataFrame()
    return pd.DataFrame(flat, columns=["x", "y"])


def _load_cached_remote_circuit_geometry(circuit_id: str) -> pd.DataFrame:
    geojson_path = _circuit_geojson_path(circuit_id)
    if not geojson_path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.debug("Non-critical error reading cached remote geometry for %s: %s", circuit_id, exc)
        return pd.DataFrame()
    return _frame_from_geojson_payload(payload)


def _fetch_remote_circuit_geometry(circuit_id: str) -> pd.DataFrame | None:
    cached = _load_cached_remote_circuit_geometry(circuit_id)
    if not cached.empty:
        return cached

    bacinger_id = BACINGER_MAPPING.get(circuit_id)
    if not bacinger_id:
        return None

    url = f"https://raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/{bacinger_id}.geojson"
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "f1-ai-platform/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        LOGGER.debug("Non-critical error fetching remote geometry for %s: %s", circuit_id, exc)
        return None

    frame = _frame_from_geojson_payload(payload)
    if frame.empty:
        return None

    geojson_path = _circuit_geojson_path(circuit_id)
    geojson_path.write_text(json.dumps(payload), encoding="utf-8")
    return frame


def build_circuit_svg(circuit_id: str, refresh: bool = False) -> Path:
    """Ensure a cached circuit SVG exists for a circuit and return its path."""

    resolved = resolve_circuit_id(circuit_id)
    info = CIRCUIT_REGISTRY[resolved]
    svg_path = Path(info.svg_path)
    transform_path = circuit_transform_path(resolved)
    ensure_directories([svg_path.parent])
    if not refresh and not _existing_svg_requires_rebuild(svg_path, transform_path, info):
        return svg_path

    points = pd.DataFrame()
    geometry_source = "placeholder"
    geometry_reference = resolved
    remote = _fetch_remote_circuit_geometry(resolved)
    if remote is not None and not remote.empty:
        points = remote[["x", "y"]].copy()
        geometry_source = "remote"
        geometry_reference = resolved

    for candidate in _iter_fastf1_position_candidates(info):
        if not points.empty:
            break
        payload = _load_fastf1_position_payload(candidate)
        if not payload:
            continue
        trace = _representative_trace(payload)
        points = _extract_loop(trace)
        if not points.empty:
            geometry_source = "fastf1_cache"
            geometry_reference = str(candidate)
            break

    if points.empty:
        trace = _fetch_openf1_location_trace(info)
        if not trace.empty:
            points = _extract_loop(trace)
            geometry_source = "openf1_location"
            geometry_reference = info.circuit_id

    if points.empty:
        points = _placeholder_track(resolved)
        geometry_source = "placeholder"
        geometry_reference = resolved

    transform = _write_svg(points, svg_path)
    merged_transform = dict(CIRCUIT_TRANSFORMS.get(resolved, {}))
    merged_transform.update(asdict(transform))
    merged_transform["geometry_source"] = geometry_source
    merged_transform["geometry_reference"] = geometry_reference
    save_json(merged_transform, transform_path)
    CIRCUIT_TRANSFORMS[resolved] = merged_transform
    try:
        circuit_coordinates.cache_clear()
    except Exception as exc:
        LOGGER.debug("Non-critical error clearing circuit coordinate cache: %s", exc)
    return svg_path


def load_circuit_svg(circuit_id: str, refresh: bool = False) -> str:
    """Return the SVG markup for a circuit, building it if needed."""

    path = build_circuit_svg(circuit_id, refresh=refresh)
    return path.read_text(encoding="utf-8")


def parse_svg_to_coordinates(svg_source: str | Path) -> pd.DataFrame:
    """Parse a polyline or path-based SVG into plot-ready coordinates."""

    content = Path(svg_source).read_text(encoding="utf-8") if isinstance(svg_source, Path) else str(svg_source)
    polyline_match = re.search(r'points="([^"]+)"', content)
    if polyline_match:
        pairs = []
        for token in polyline_match.group(1).strip().split():
            if "," not in token:
                continue
            x_str, y_str = token.split(",", maxsplit=1)
            pairs.append((float(x_str), float(y_str)))
        return pd.DataFrame(pairs, columns=["x", "y"])

    path_match = re.search(r'd="([^"]+)"', content)
    if path_match:
        path_data = path_match.group(1)
        tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", path_data)
        if not tokens:
            return pd.DataFrame(columns=["x", "y"])

        def to_point(x_value: float, y_value: float, relative: bool, current: tuple[float, float]) -> tuple[float, float]:
            return (current[0] + x_value, current[1] + y_value) if relative else (x_value, y_value)

        def cubic_point(
            p0: tuple[float, float],
            p1: tuple[float, float],
            p2: tuple[float, float],
            p3: tuple[float, float],
            t: float,
        ) -> tuple[float, float]:
            inv = 1.0 - t
            x_value = (inv**3) * p0[0] + 3 * (inv**2) * t * p1[0] + 3 * inv * (t**2) * p2[0] + (t**3) * p3[0]
            y_value = (inv**3) * p0[1] + 3 * (inv**2) * t * p1[1] + 3 * inv * (t**2) * p2[1] + (t**3) * p3[1]
            return x_value, y_value

        def quadratic_point(
            p0: tuple[float, float],
            p1: tuple[float, float],
            p2: tuple[float, float],
            t: float,
        ) -> tuple[float, float]:
            inv = 1.0 - t
            x_value = (inv**2) * p0[0] + 2 * inv * t * p1[0] + (t**2) * p2[0]
            y_value = (inv**2) * p0[1] + 2 * inv * t * p1[1] + (t**2) * p2[1]
            return x_value, y_value

        points: list[tuple[float, float]] = []
        index = 0
        command = "M"
        current = (0.0, 0.0)
        start = (0.0, 0.0)
        last_control_cubic: tuple[float, float] | None = None
        last_control_quad: tuple[float, float] | None = None

        while index < len(tokens):
            token = tokens[index]
            if re.fullmatch(r"[MmLlHhVvCcSsQqTtAaZz]", token):
                command = token
                index += 1
            relative = command.islower()
            upper = command.upper()

            if upper == "Z":
                points.append(start)
                current = start
                last_control_cubic = None
                last_control_quad = None
                continue

            if upper in {"M", "L", "T"}:
                x_value = float(tokens[index])
                y_value = float(tokens[index + 1])
                index += 2
                target = to_point(x_value, y_value, relative, current)
                if upper == "T" and last_control_quad is not None:
                    control = (2 * current[0] - last_control_quad[0], 2 * current[1] - last_control_quad[1])
                    for step in range(1, 13):
                        points.append(quadratic_point(current, control, target, step / 12.0))
                    last_control_quad = control
                else:
                    points.append(target)
                    last_control_quad = None
                current = target
                if upper == "M":
                    start = target
                    command = "l" if relative else "L"
                last_control_cubic = None
                continue

            if upper == "H":
                x_value = float(tokens[index])
                index += 1
                current = (current[0] + x_value, current[1]) if relative else (x_value, current[1])
                points.append(current)
                last_control_cubic = None
                last_control_quad = None
                continue

            if upper == "V":
                y_value = float(tokens[index])
                index += 1
                current = (current[0], current[1] + y_value) if relative else (current[0], y_value)
                points.append(current)
                last_control_cubic = None
                last_control_quad = None
                continue

            if upper == "C":
                x1, y1, x2, y2, x3, y3 = (float(value) for value in tokens[index : index + 6])
                index += 6
                control1 = to_point(x1, y1, relative, current)
                control2 = to_point(x2, y2, relative, current)
                target = to_point(x3, y3, relative, current)
                for step in range(1, 17):
                    points.append(cubic_point(current, control1, control2, target, step / 16.0))
                current = target
                last_control_cubic = control2
                last_control_quad = None
                continue

            if upper == "S":
                x2, y2, x3, y3 = (float(value) for value in tokens[index : index + 4])
                index += 4
                control1 = (
                    (2 * current[0] - last_control_cubic[0], 2 * current[1] - last_control_cubic[1])
                    if last_control_cubic is not None
                    else current
                )
                control2 = to_point(x2, y2, relative, current)
                target = to_point(x3, y3, relative, current)
                for step in range(1, 17):
                    points.append(cubic_point(current, control1, control2, target, step / 16.0))
                current = target
                last_control_cubic = control2
                last_control_quad = None
                continue

            if upper == "Q":
                x1, y1, x2, y2 = (float(value) for value in tokens[index : index + 4])
                index += 4
                control = to_point(x1, y1, relative, current)
                target = to_point(x2, y2, relative, current)
                for step in range(1, 13):
                    points.append(quadratic_point(current, control, target, step / 12.0))
                current = target
                last_control_quad = control
                last_control_cubic = None
                continue

            if upper == "A":
                x_value = float(tokens[index + 5])
                y_value = float(tokens[index + 6])
                index += 7
                current = to_point(x_value, y_value, relative, current)
                points.append(current)
                last_control_cubic = None
                last_control_quad = None
                continue

            index += 1

        return pd.DataFrame(points, columns=["x", "y"]).dropna().reset_index(drop=True)

    return pd.DataFrame(columns=["x", "y"])


@lru_cache(maxsize=128)
def circuit_coordinates(circuit_id: str) -> pd.DataFrame:
    """Return cached SVG coordinates for a circuit."""

    resolved = resolve_circuit_id(circuit_id)
    build_circuit_svg(resolved)
    return parse_svg_to_coordinates(Path(CIRCUIT_REGISTRY[resolved].svg_path))


def load_circuit_transform(circuit_id: str) -> dict[str, Any]:
    """Load a persisted transform, falling back to the registry defaults."""

    resolved = resolve_circuit_id(circuit_id)
    path = circuit_transform_path(resolved)
    if path.exists():
        transform = load_json(path, default={}) or {}
        if transform:
            return transform
    return dict(CIRCUIT_TRANSFORMS[resolved])


def map_openf1_to_svg(x: float, y: float, circuit_id: str) -> tuple[float, float]:
    """Map telemetry coordinates into the circuit SVG coordinate space."""

    resolved = resolve_circuit_id(circuit_id)
    transform = load_circuit_transform(resolved)
    source_min_x = float(transform.get("source_min_x", 0.0))
    source_max_x = float(transform.get("source_max_x", 1.0))
    source_min_y = float(transform.get("source_min_y", 0.0))
    source_max_y = float(transform.get("source_max_y", 1.0))
    target_min_x = float(transform.get("target_min_x", 60.0))
    target_max_x = float(transform.get("target_max_x", 1140.0))
    target_min_y = float(transform.get("target_min_y", 60.0))
    target_max_y = float(transform.get("target_max_y", 640.0))

    norm_x = 0.5 if math.isclose(source_max_x, source_min_x) else (float(x) - source_min_x) / (source_max_x - source_min_x)
    norm_y = 0.5 if math.isclose(source_max_y, source_min_y) else (float(y) - source_min_y) / (source_max_y - source_min_y)
    if transform.get("flip_x", False):
        norm_x = 1.0 - norm_x
    if transform.get("flip_y", True):
        norm_y = 1.0 - norm_y

    svg_x = target_min_x + norm_x * (target_max_x - target_min_x)
    svg_y = target_min_y + norm_y * (target_max_y - target_min_y)

    scale_x = float(transform.get("scale_x", 1.0))
    scale_y = float(transform.get("scale_y", 1.0))
    offset_x = float(transform.get("offset_x", 0.0))
    offset_y = float(transform.get("offset_y", 0.0))
    rotation = math.radians(float(transform.get("rotation", 0.0)))
    center_x = (target_min_x + target_max_x) / 2.0
    center_y = (target_min_y + target_max_y) / 2.0
    shifted_x = (svg_x - center_x) * scale_x
    shifted_y = (svg_y - center_y) * scale_y
    rotated_x = shifted_x * math.cos(rotation) - shifted_y * math.sin(rotation)
    rotated_y = shifted_x * math.sin(rotation) + shifted_y * math.cos(rotation)
    return rotated_x + center_x + offset_x, rotated_y + center_y + offset_y


def _sample_track_fraction(coords: pd.DataFrame, fraction: float) -> dict[str, float]:
    if coords.empty:
        return {"x": 0.0, "y": 0.0}
    index = min(max(int(round(float(fraction) * (len(coords) - 1))), 0), len(coords) - 1)
    row = coords.iloc[index]
    return {"x": float(row["x"]), "y": float(row["y"])}


def circuit_layer_geometry(circuit_id: str) -> dict[str, Any]:
    """Return sector, DRS and pit-lane geometry derived from the cached SVG path."""

    resolved = resolve_circuit_id(circuit_id)
    coords = circuit_coordinates(resolved)
    info = CIRCUIT_REGISTRY[resolved]
    sector_points = {
        1: _sample_track_fraction(coords, info.sector_breaks[0]),
        2: _sample_track_fraction(coords, info.sector_breaks[1]),
        3: _sample_track_fraction(coords, 0.99),
    }
    drs_coords: list[dict[str, list[float]]] = []
    for start_fraction, end_fraction in info.drs_zone_spans:
        start_index = min(max(int(start_fraction * max(len(coords) - 1, 1)), 0), max(len(coords) - 1, 0))
        end_index = min(max(int(end_fraction * max(len(coords) - 1, 1)), 0), max(len(coords) - 1, 0))
        segment = coords.iloc[start_index : end_index + 1]
        drs_coords.append({"x": segment["x"].tolist(), "y": segment["y"].tolist()})

    pit_lane = None
    if info.pit_lane_span and not coords.empty:
        start_fraction, end_fraction = info.pit_lane_span
        start_index = min(max(int(start_fraction * (len(coords) - 1)), 0), len(coords) - 1)
        end_index = min(max(int(end_fraction * (len(coords) - 1)), 0), len(coords) - 1)
        segment = coords.iloc[start_index : end_index + 1].copy()
        segment["y"] = segment["y"] - 18.0
        pit_lane = {"x": segment["x"].tolist(), "y": segment["y"].tolist()}

    return {
        "sector_boundaries": sector_points,
        "drs_zones_coords": drs_coords,
        "pit_lane_coords": pit_lane,
    }
