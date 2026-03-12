"""Historical weather ingestion and wet-race classification."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils import (
    CONFIG as BASE_CONFIG,
    ensure_directories,
    get_logger,
    load_dataframe,
    load_json,
    save_dataframe,
    save_json,
    validate_columns,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "raw_dir": BASE_CONFIG["raw_dir"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "weather_path": BASE_CONFIG["processed_dir"] / "race_weather.csv",
    "forecast_dir": BASE_CONFIG["raw_dir"] / "weather" / "forecasts",
    "base_url": "https://archive-api.open-meteo.com/v1/archive",
    "forecast_url": "https://api.open-meteo.com/v1/forecast",
    "request_timeout": 30,
    "max_retries": 3,
    "retry_backoff_seconds": 2.0,
    "rate_limit_seconds": 0.50,
}

CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "albert_park": (-37.8497, 144.9680),
    "americas": (30.1328, -97.6411),
    "bahrain": (26.0325, 50.5106),
    "baku": (40.3725, 49.8533),
    "catalunya": (41.5700, 2.2611),
    "hockenheimring": (49.3278, 8.5656),
    "hungaroring": (47.5789, 19.2486),
    "imola": (44.3439, 11.7167),
    "interlagos": (-23.7036, -46.6997),
    "istanbul": (40.9517, 29.0500),
    "jeddah": (21.6319, 39.1044),
    "losail": (25.4900, 51.4542),
    "lusail": (25.4900, 51.4542),
    "marina_bay": (1.2914, 103.8640),
    "miami": (25.9581, -80.2389),
    "monaco": (43.7347, 7.4206),
    "monza": (45.6156, 9.2811),
    "mugello": (43.9975, 11.3719),
    "nurburgring": (50.3356, 6.9475),
    "portimao": (37.2276, -8.6270),
    "red_bull_ring": (47.2197, 14.7647),
    "spielberg": (47.2197, 14.7647),
    "ricard": (43.2506, 5.7913),
    "paul_ricard": (43.2506, 5.7913),
    "rodriguez": (19.4042, -99.0907),
    "sepang": (2.7608, 101.7380),
    "shanghai": (31.3389, 121.2200),
    "silverstone": (52.0786, -1.0169),
    "sochi": (43.4057, 39.9578),
    "spa": (50.4372, 5.9714),
    "suzuka": (34.8431, 136.5414),
    "vegas": (36.1147, -115.1728),
    "villeneuve": (45.5006, -73.5228),
    "yas_marina": (24.4672, 54.6031),
    "zandvoort": (52.3888, 4.5409),
}

WET_WEATHER_CODES: set[int] = {61, 63, 65, 80, 81, 82}
WET_RACE_REGISTRY: dict[tuple[int, int], str] = {
    (2016, 2): "wet",
    (2017, 7): "wet",
    (2018, 14): "wet",
    (2019, 14): "wet",
    (2021, 13): "mixed",
    (2021, 14): "wet",
    (2022, 9): "wet",
    (2023, 6): "wet",
}

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WeatherForecast:
    """Condensed race-weekend forecast for the dashboard and CLI."""

    race_day_rain_probability: float
    race_day_temperature: float
    session_forecasts: dict[str, dict[str, Any]]
    confidence: str


class OpenMeteoClient:
    """Thin historical-weather client with retry and cache-friendly pacing."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.last_request_time = 0.0

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_time
        sleep_seconds = CONFIG["rate_limit_seconds"] - elapsed
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def fetch_archive(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Fetch Open-Meteo archive data for a circuit/date range."""

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                    "weather_code",
                ]
            ),
            "timezone": "GMT",
        }
        last_error: Exception | None = None
        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                self._respect_rate_limit()
                response = self.session.get(CONFIG["base_url"], params=params, timeout=CONFIG["request_timeout"])
                self.last_request_time = time.monotonic()
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                LOGGER.warning(
                    "Open-Meteo request failed for lat=%s lon=%s %s to %s on attempt %s/%s: %s",
                    latitude,
                    longitude,
                    start_date,
                    end_date,
                    attempt,
                    CONFIG["max_retries"],
                    exc,
                )
                if attempt < CONFIG["max_retries"]:
                    time.sleep(CONFIG["retry_backoff_seconds"] * attempt)
        raise RuntimeError("Failed to fetch Open-Meteo archive payload") from last_error

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone_name: str = "GMT",
    ) -> dict[str, Any]:
        """Fetch hourly forecast data for an upcoming race weekend."""

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "precipitation_probability",
                    "precipitation",
                    "weathercode",
                    "windspeed_10m",
                    "visibility",
                    "cloudcover",
                ]
            ),
            "timezone": timezone_name,
        }
        last_error: Exception | None = None
        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                self._respect_rate_limit()
                response = self.session.get(CONFIG["forecast_url"], params=params, timeout=CONFIG["request_timeout"])
                self.last_request_time = time.monotonic()
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < CONFIG["max_retries"]:
                    time.sleep(CONFIG["retry_backoff_seconds"] * attempt)
        raise RuntimeError("Failed to fetch Open-Meteo forecast payload") from last_error


class WeatherPipeline:
    """Thin wrapper around the functional weather pipeline."""

    def __init__(self, client: OpenMeteoClient | None = None) -> None:
        self.client = client if client is not None else OpenMeteoClient()

    def run(self, race_results: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        return run_weather_pipeline(race_results, refresh=refresh)


def _cache_paths(circuit_id: str, year: int) -> tuple[Path, Path]:
    """Return raw JSON and CSV cache paths for a circuit-season weather query."""

    return (
        CONFIG["raw_dir"] / f"weather_{circuit_id}_{year}.json",
        CONFIG["raw_dir"] / f"weather_{circuit_id}_{year}.csv",
    )


def _normalize_weather_payload(payload: dict[str, Any], circuit_id: str, year: int | None = None) -> pd.DataFrame:
    """Convert a daily Open-Meteo response into tabular form."""

    daily = payload.get("daily", {})
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
            "temperature_2m_max": daily.get("temperature_2m_max", []),
            "temperature_2m_min": daily.get("temperature_2m_min", []),
            "precipitation_sum": daily.get("precipitation_sum", []),
            "windspeed_10m_max": daily.get("wind_speed_10m_max", []),
            "weathercode": daily.get("weather_code", []),
        }
    )
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "circuit_id",
                "season",
                "date",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "windspeed_10m_max",
                "weathercode",
                "weather_confidence",
            ]
        )

    frame["circuit_id"] = circuit_id
    frame["season"] = year if year is not None else frame["date"].dt.year.astype(int)
    frame["weathercode"] = pd.to_numeric(frame["weathercode"], errors="coerce")
    frame["weather_confidence"] = 1.0
    for column in ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "windspeed_10m_max"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        [
            "circuit_id",
            "season",
            "date",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "weathercode",
            "weather_confidence",
        ]
    ].dropna(subset=["date"]).reset_index(drop=True)


def _save_year_cache(circuit_id: str, year: int, frame: pd.DataFrame) -> None:
    """Persist per-year normalized weather caches for one circuit."""

    json_path, csv_path = _cache_paths(circuit_id, year)
    save_dataframe(frame, csv_path)
    save_json(frame.to_dict(orient="records"), json_path)


def _placeholder_weather_frame(circuit_id: str, circuit_races: pd.DataFrame) -> pd.DataFrame:
    """Build a dry-default placeholder weather frame when the API is unavailable."""

    placeholder = (
        circuit_races[["season", "race_date"]]
        .drop_duplicates()
        .rename(columns={"race_date": "date"})
        .copy()
    )
    placeholder["circuit_id"] = circuit_id
    placeholder["temperature_2m_max"] = np.nan
    placeholder["temperature_2m_min"] = np.nan
    placeholder["precipitation_sum"] = 0.0
    placeholder["windspeed_10m_max"] = np.nan
    placeholder["weathercode"] = np.nan
    placeholder["weather_confidence"] = 0.0
    return placeholder[
        [
            "circuit_id",
            "season",
            "date",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "weathercode",
            "weather_confidence",
        ]
    ].reset_index(drop=True)


def _fetch_or_load_circuit_weather_history(
    client: OpenMeteoClient,
    circuit_id: str,
    circuit_races: pd.DataFrame,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch one circuit-wide weather history range and split it into yearly caches."""

    years = sorted(circuit_races["season"].astype(int).unique().tolist())
    cached_frames: list[pd.DataFrame] = []
    missing_years: list[int] = []
    for year in years:
        _, csv_path = _cache_paths(circuit_id, year)
        if csv_path.exists():
            cached_frames.append(load_dataframe(csv_path, parse_dates=["date"]))
        else:
            missing_years.append(year)

    if not missing_years and cached_frames:
        return pd.concat(cached_frames, ignore_index=True)

    if circuit_id not in CIRCUIT_COORDS:
        raise KeyError(f"Missing CIRCUIT_COORDS entry for circuit_id={circuit_id}")

    latitude, longitude = CIRCUIT_COORDS[circuit_id]
    start_date = circuit_races["race_date"].min().date().isoformat()
    end_date = circuit_races["race_date"].max().date().isoformat()
    try:
        payload = client.fetch_archive(latitude, longitude, start_date, end_date)
        full_frame = _normalize_weather_payload(payload, circuit_id)
        for year in years:
            year_frame = full_frame.loc[full_frame["date"].dt.year == year].copy()
            _save_year_cache(circuit_id, year, year_frame)
        LOGGER.info(
            "Fetched weather history for %s covering %s to %s and split into %s yearly caches",
            circuit_id,
            start_date,
            end_date,
            len(years),
        )
        return pd.concat([full_frame.loc[full_frame["date"].dt.year == year].copy() for year in years], ignore_index=True)
    except Exception as exc:
        LOGGER.warning("Falling back to cached weather for %s after error: %s", circuit_id, exc)
        fallback_frames: list[pd.DataFrame] = []
        for year in years:
            json_path, csv_path = _cache_paths(circuit_id, year)
            if csv_path.exists():
                fallback_frames.append(load_dataframe(csv_path, parse_dates=["date"]))
                continue
            cached_payload = load_json(json_path)
            if cached_payload is not None:
                if isinstance(cached_payload, list):
                    frame = pd.DataFrame(cached_payload)
                    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                    if "weather_confidence" not in frame.columns:
                        frame["weather_confidence"] = 1.0
                    fallback_frames.append(frame)
                else:
                    fallback_frames.append(_normalize_weather_payload(cached_payload, circuit_id, year))
        if fallback_frames:
            return pd.concat(fallback_frames, ignore_index=True)
        if circuit_id in {"mugello", "nurburgring"} and 2020 in years:
            LOGGER.warning(
                "Using weather_confidence=0.0 fallback for limited COVID-era weather coverage at %s 2020",
                circuit_id,
            )
        placeholder = _placeholder_weather_frame(circuit_id, circuit_races)
        for year in years:
            year_frame = placeholder.loc[placeholder["season"] == year].copy()
            _save_year_cache(circuit_id, year, year_frame)
        LOGGER.warning(
            "Using placeholder dry weather defaults for %s because Open-Meteo failed and no cache was available",
            circuit_id,
        )
        return placeholder


def classify_race_condition(
    season: int,
    round_number: int,
    precipitation_sum: float | None,
    weathercode: float | None,
) -> str:
    """Classify a race as dry, mixed, or wet using weather plus registry overrides."""

    override = WET_RACE_REGISTRY.get((int(season), int(round_number)))
    if override in {"wet", "mixed"}:
        return override

    precipitation = float(precipitation_sum or 0.0)
    code = int(weathercode) if pd.notna(weathercode) else None
    if precipitation > 2.0 or code in WET_WEATHER_CODES:
        return "wet"
    if 0.5 <= precipitation <= 2.0:
        return "mixed"
    return "dry"


def classify_race_conditions(precipitation_sum: float | None, weathercode: float | None) -> str:
    """Classify a race without season-level overrides for smoke-test compatibility."""

    precipitation = float(precipitation_sum or 0.0)
    code = int(weathercode) if weathercode is not None and pd.notna(weathercode) else None
    if precipitation > 2.0 or code in WET_WEATHER_CODES:
        return "WET"
    if precipitation >= 0.5:
        return "MIXED"
    return "DRY"


def build_race_weather_frame(race_results: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Fetch and classify weather for every race in the dataset."""

    validate_columns(
        race_results,
        ["season", "round", "circuit_id", "race_date"],
        "race results for weather enrichment",
    )

    races = (
        race_results[["season", "round", "circuit_id", "race_name", "race_date"]]
        .drop_duplicates()
        .copy()
    )
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    races = races.dropna(subset=["race_date"]).sort_values(["season", "round"]).reset_index(drop=True)

    client = OpenMeteoClient()
    weather_frames: list[pd.DataFrame] = []
    for circuit_id, group in races.groupby("circuit_id", sort=True):
        weather_frames.append(
            _fetch_or_load_circuit_weather_history(
                client=client,
                circuit_id=str(circuit_id),
                circuit_races=group,
                refresh=refresh,
            )
        )

    if weather_frames:
        weather_daily = pd.concat(weather_frames, ignore_index=True)
    else:
        weather_daily = pd.DataFrame(
            columns=[
                "circuit_id",
                "season",
                "date",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "windspeed_10m_max",
                "weathercode",
                "weather_confidence",
            ]
        )

    race_weather = races.merge(
        weather_daily.rename(columns={"date": "race_date_lookup"}),
        how="left",
        left_on=["season", "circuit_id", "race_date"],
        right_on=["season", "circuit_id", "race_date_lookup"],
    )
    if "weather_confidence" not in race_weather.columns:
        race_weather["weather_confidence"] = 0.0
    race_weather["race_precipitation_mm"] = pd.to_numeric(race_weather["precipitation_sum"], errors="coerce").fillna(0.0)
    race_weather["race_max_wind_kmh"] = pd.to_numeric(race_weather["windspeed_10m_max"], errors="coerce").fillna(0.0)
    race_weather["race_temperature_c"] = (
        pd.to_numeric(race_weather["temperature_2m_max"], errors="coerce").fillna(0.0)
        + pd.to_numeric(race_weather["temperature_2m_min"], errors="coerce").fillna(0.0)
    ) / 2.0
    race_weather["weathercode"] = pd.to_numeric(race_weather["weathercode"], errors="coerce")
    race_weather["weather_confidence"] = pd.to_numeric(race_weather["weather_confidence"], errors="coerce").fillna(0.0)
    race_weather["race_condition"] = race_weather.apply(
        lambda row: classify_race_condition(
            int(row["season"]),
            int(row["round"]),
            float(row["race_precipitation_mm"]),
            row["weathercode"],
        ),
        axis=1,
    )
    race_weather["is_wet_race"] = race_weather["race_condition"].eq("wet").astype(int)
    race_weather["is_mixed_race"] = race_weather["race_condition"].eq("mixed").astype(int)

    selected_columns = [
        "season",
        "round",
        "circuit_id",
        "race_name",
        "race_date",
        "race_precipitation_mm",
        "race_max_wind_kmh",
        "race_temperature_c",
        "weathercode",
        "weather_confidence",
        "race_condition",
        "is_wet_race",
        "is_mixed_race",
    ]
    race_weather = race_weather[selected_columns].sort_values(["season", "round"]).reset_index(drop=True)
    validate_columns(
        race_weather,
        [
            "season",
            "round",
            "circuit_id",
            "race_precipitation_mm",
            "race_max_wind_kmh",
            "race_temperature_c",
            "is_wet_race",
            "is_mixed_race",
            "race_condition",
        ],
        "race weather frame",
    )
    return race_weather


def run_weather_pipeline(
    race_results: pd.DataFrame | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cached weather features and enrich processed race results."""

    ensure_directories()
    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    race_results = race_results.copy()
    race_results["race_date"] = pd.to_datetime(race_results["race_date"], errors="coerce")
    overlapping_weather_columns = [
        "race_precipitation_mm",
        "race_max_wind_kmh",
        "race_temperature_c",
        "weathercode",
        "weather_confidence",
        "race_condition",
        "is_wet_race",
        "is_mixed_race",
    ]
    race_results = race_results.drop(
        columns=[column for column in overlapping_weather_columns if column in race_results.columns],
        errors="ignore",
    )

    race_weather = build_race_weather_frame(race_results, refresh=refresh)
    enriched = race_results.merge(
        race_weather,
        how="left",
        on=["season", "round", "circuit_id", "race_name", "race_date"],
    )
    enriched["race_precipitation_mm"] = pd.to_numeric(enriched["race_precipitation_mm"], errors="coerce").fillna(0.0)
    enriched["race_max_wind_kmh"] = pd.to_numeric(enriched["race_max_wind_kmh"], errors="coerce").fillna(0.0)
    enriched["race_temperature_c"] = pd.to_numeric(enriched["race_temperature_c"], errors="coerce").fillna(0.0)
    enriched["is_wet_race"] = pd.to_numeric(enriched["is_wet_race"], errors="coerce").fillna(0).astype(int)
    enriched["is_mixed_race"] = pd.to_numeric(enriched["is_mixed_race"], errors="coerce").fillna(0).astype(int)
    enriched["race_condition"] = enriched["race_condition"].fillna("dry")
    enriched["weathercode"] = pd.to_numeric(enriched["weathercode"], errors="coerce")
    enriched["weather_confidence"] = pd.to_numeric(enriched["weather_confidence"], errors="coerce").fillna(0.0)

    save_dataframe(race_weather, CONFIG["weather_path"])
    save_dataframe(enriched, CONFIG["race_results_path"])
    LOGGER.info(
        "Weather pipeline complete: %s races | dry=%s mixed=%s wet=%s",
        len(race_weather),
        int(race_weather["race_condition"].eq("dry").sum()),
        int(race_weather["race_condition"].eq("mixed").sum()),
        int(race_weather["race_condition"].eq("wet").sum()),
    )
    return race_weather, enriched


def _forecast_cache_path(season: int, round_number: int, circuit_id: str) -> Path:
    return CONFIG["forecast_dir"] / f"{int(season)}_{int(round_number):02d}_{circuit_id}.json"


def _forecast_confidence(target_date: pd.Timestamp) -> str:
    target = target_date.tz_convert("UTC") if target_date.tzinfo is not None else target_date.tz_localize("UTC")
    days_out = max(int((target - pd.Timestamp(datetime.now(timezone.utc))).total_seconds() // 86400), 0)
    if days_out <= 3:
        return "high"
    if days_out <= 7:
        return "medium"
    return "low"


def _session_window(label: str) -> tuple[int, int]:
    windows = {
        "FP1": (13, 14),
        "FP2": (17, 18),
        "FP3": (13, 14),
        "Qualifying": (17, 18),
        "Race": (15, 17),
    }
    return windows[label]


def _collapse_session_forecasts(hourly: pd.DataFrame, weekend_start: pd.Timestamp) -> dict[str, dict[str, Any]]:
    sessions = {
        "FP1": weekend_start.normalize() + pd.Timedelta(days=0),
        "FP2": weekend_start.normalize() + pd.Timedelta(days=0),
        "FP3": weekend_start.normalize() + pd.Timedelta(days=1),
        "Qualifying": weekend_start.normalize() + pd.Timedelta(days=1),
        "Race": weekend_start.normalize() + pd.Timedelta(days=2),
    }
    forecasts: dict[str, dict[str, Any]] = {}
    for session_name, session_date in sessions.items():
        start_hour, end_hour = _session_window(session_name)
        window = hourly.loc[
            (hourly["time"].dt.normalize() == session_date.normalize())
            & (hourly["time"].dt.hour >= start_hour)
            & (hourly["time"].dt.hour <= end_hour)
        ].copy()
        if window.empty:
            forecasts[session_name] = {"temperature_c": None, "rain_probability": None, "wind_kmh": None, "weathercode": None}
            continue
        forecasts[session_name] = {
            "temperature_c": float(pd.to_numeric(window["temperature_2m"], errors="coerce").mean()),
            "rain_probability": float(pd.to_numeric(window["precipitation_probability"], errors="coerce").mean()),
            "precipitation_mm": float(pd.to_numeric(window["precipitation"], errors="coerce").mean()),
            "wind_kmh": float(pd.to_numeric(window["windspeed_10m"], errors="coerce").mean()),
            "visibility_m": float(pd.to_numeric(window["visibility"], errors="coerce").mean()),
            "cloudcover_pct": float(pd.to_numeric(window["cloudcover"], errors="coerce").mean()),
            "weathercode": float(pd.to_numeric(window["weathercode"], errors="coerce").median()),
        }
    return forecasts


def fetch_race_weekend_forecast(season: int, round_number: int, refresh: bool = False) -> WeatherForecast | None:
    """Fetch and cache an upcoming race-weekend forecast when the event is within ten days."""

    calendar = load_dataframe(BASE_CONFIG["processed_dir"] / "calendar.csv", parse_dates=["date"]) if (BASE_CONFIG["processed_dir"] / "calendar.csv").exists() else pd.DataFrame()
    row = calendar.loc[(calendar["season"] == int(season)) & (calendar["round"] == int(round_number))].head(1)
    if row.empty:
        return None
    circuit_id = str(row["circuit_id"].iloc[0])
    race_date = pd.Timestamp(row["date"].iloc[0])
    if race_date.tzinfo is None:
        race_date = race_date.tz_localize("UTC")
    days_until_race = (race_date - pd.Timestamp(datetime.now(timezone.utc))).days
    if days_until_race > 10:
        return None

    cache_path = _forecast_cache_path(season, round_number, circuit_id)
    if cache_path.exists() and not refresh:
        payload = load_json(cache_path, default=None)
        if payload:
            return WeatherForecast(
                race_day_rain_probability=float(payload["race_day_rain_probability"]),
                race_day_temperature=float(payload["race_day_temperature"]),
                session_forecasts=dict(payload["session_forecasts"]),
                confidence=str(payload["confidence"]),
            )

    coords = CIRCUIT_COORDS.get(circuit_id)
    if coords is None:
        return None
    forecast_start = (race_date - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    forecast_end = race_date.strftime("%Y-%m-%d")
    client = OpenMeteoClient()
    try:
        payload = client.fetch_forecast(coords[0], coords[1], forecast_start, forecast_end)
    except Exception as exc:
        LOGGER.debug("Non-critical error fetching forecast for %s %s R%s: %s", circuit_id, season, round_number, exc)
        return None
    hourly = pd.DataFrame(payload.get("hourly", {}))
    if hourly.empty:
        return None
    hourly["time"] = pd.to_datetime(hourly["time"], errors="coerce")
    session_forecasts = _collapse_session_forecasts(hourly, race_date - pd.Timedelta(days=2))
    race_window = hourly.loc[hourly["time"].dt.normalize() == race_date.normalize()].copy()
    forecast = WeatherForecast(
        race_day_rain_probability=float(pd.to_numeric(race_window["precipitation_probability"], errors="coerce").mean()),
        race_day_temperature=float(pd.to_numeric(race_window["temperature_2m"], errors="coerce").mean()),
        session_forecasts=session_forecasts,
        confidence=_forecast_confidence(race_date),
    )
    ensure_directories([cache_path.parent])
    save_json(
        {
            "race_day_rain_probability": forecast.race_day_rain_probability,
            "race_day_temperature": forecast.race_day_temperature,
            "session_forecasts": forecast.session_forecasts,
            "confidence": forecast.confidence,
        },
        cache_path,
    )
    return forecast
