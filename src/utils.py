"""Shared utilities for the F1 AI prediction engine."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import unicodedata
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

CONFIG: dict[str, Any] = {
    "project_root": Path(__file__).resolve().parents[1],
}
CONFIG["data_dir"] = CONFIG["project_root"] / "data"
CONFIG["raw_dir"] = CONFIG["data_dir"] / "raw"
CONFIG["processed_dir"] = CONFIG["data_dir"] / "processed"
CONFIG["legacy_predictions_dir"] = CONFIG["data_dir"] / "predictions"
CONFIG["predictions_dir"] = CONFIG["processed_dir"] / "predictions"
CONFIG["logs_dir"] = CONFIG["project_root"] / "logs"
CONFIG["log_file"] = CONFIG["logs_dir"] / "pipeline.log"
CONFIG["pipeline_log_file"] = CONFIG["logs_dir"] / "pipeline.log"
CONFIG["predictions_log_file"] = CONFIG["logs_dir"] / "predictions.log"
CONFIG["errors_log_file"] = CONFIG["logs_dir"] / "errors.log"

F1_POINTS: dict[int, int] = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

CIRCUIT_METADATA: dict[str, dict[str, Any]] = {
    "albert_park": {"overtake_difficulty": 0.58, "is_street_circuit": False},
    "americas": {"overtake_difficulty": 0.72, "is_street_circuit": False},
    "autodromo_enzo_e_dino_ferrari": {
        "overtake_difficulty": 0.42,
        "is_street_circuit": False,
    },
    "avus": {"overtake_difficulty": 0.48, "is_street_circuit": False},
    "bahrain": {"overtake_difficulty": 0.84, "is_street_circuit": False},
    "baku": {"overtake_difficulty": 0.75, "is_street_circuit": True},
    "catalunya": {"overtake_difficulty": 0.44, "is_street_circuit": False},
    "galvez": {"overtake_difficulty": 0.52, "is_street_circuit": False},
    "hockenheimring": {"overtake_difficulty": 0.63, "is_street_circuit": False},
    "hungaroring": {"overtake_difficulty": 0.22, "is_street_circuit": False},
    "indianapolis": {"overtake_difficulty": 0.61, "is_street_circuit": False},
    "interlagos": {"overtake_difficulty": 0.77, "is_street_circuit": False},
    "istanbul": {"overtake_difficulty": 0.64, "is_street_circuit": False},
    "jeddah": {"overtake_difficulty": 0.67, "is_street_circuit": True},
    "jose_carlos_pace": {"overtake_difficulty": 0.76, "is_street_circuit": False},
    "las_vegas": {"overtake_difficulty": 0.71, "is_street_circuit": True},
    "vegas": {"overtake_difficulty": 0.71, "is_street_circuit": True},
    "losail": {"overtake_difficulty": 0.69, "is_street_circuit": False},
    "marina_bay": {"overtake_difficulty": 0.36, "is_street_circuit": True},
    "miami": {"overtake_difficulty": 0.64, "is_street_circuit": True},
    "monaco": {"overtake_difficulty": 0.10, "is_street_circuit": True},
    "monza": {"overtake_difficulty": 0.90, "is_street_circuit": False},
    "mugello": {"overtake_difficulty": 0.45, "is_street_circuit": False},
    "nurburgring": {"overtake_difficulty": 0.55, "is_street_circuit": False},
    "portimao": {"overtake_difficulty": 0.59, "is_street_circuit": False},
    "red_bull_ring": {"overtake_difficulty": 0.79, "is_street_circuit": False},
    "ricard": {"overtake_difficulty": 0.66, "is_street_circuit": False},
    "rodriguez": {"overtake_difficulty": 0.81, "is_street_circuit": False},
    "sepang": {"overtake_difficulty": 0.74, "is_street_circuit": False},
    "shanghai": {"overtake_difficulty": 0.83, "is_street_circuit": False},
    "silverstone": {"overtake_difficulty": 0.68, "is_street_circuit": False},
    "sochi": {"overtake_difficulty": 0.39, "is_street_circuit": False},
    "spa": {"overtake_difficulty": 0.88, "is_street_circuit": False},
    "suzuka": {"overtake_difficulty": 0.31, "is_street_circuit": False},
    "villeneuve": {"overtake_difficulty": 0.82, "is_street_circuit": False},
    "yas_marina": {"overtake_difficulty": 0.57, "is_street_circuit": False},
    "yeongam": {"overtake_difficulty": 0.62, "is_street_circuit": False},
    "zandvoort": {"overtake_difficulty": 0.29, "is_street_circuit": False},
}

_LOGGING_READY = False


class _JsonFormatter(logging.Formatter):
    """Structured JSON formatter for file-based application logs."""

    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._RESERVED and not key.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default)


class _PredictionFilter(logging.Filter):
    """Route explicit prediction events to the prediction log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "prediction_event", False))


class _ErrorFilter(logging.Filter):
    """Route error and critical records to the error log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(record.levelno >= logging.ERROR)


def setup_logging(log_dir: str | Path = "logs") -> dict[str, RotatingFileHandler]:
    """Create rotating file handlers for the main application logs."""

    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return {
        "pipeline": RotatingFileHandler(
            target_dir / "pipeline.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
        "predictions": RotatingFileHandler(
            target_dir / "predictions.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
        "errors": RotatingFileHandler(
            target_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        ),
    }


def ensure_directories(paths: Iterable[Path] | None = None) -> None:
    """Create standard project directories if they do not already exist."""

    target_paths = list(
        paths
        or [
            CONFIG["raw_dir"],
            CONFIG["processed_dir"],
            CONFIG["predictions_dir"],
            CONFIG["legacy_predictions_dir"],
            CONFIG["logs_dir"],
        ]
    )
    for path in target_paths:
        path.mkdir(parents=True, exist_ok=True)


def detect_current_season(reference_time: datetime | None = None) -> int:
    """Return the active F1 season year based on the current date."""

    now = reference_time or datetime.now()
    season = int(now.year)
    if now.month < 2:
        season -= 1
    return season


def timing_to_milliseconds(timing_str: Any) -> float:
    """Convert common F1 timing strings to milliseconds.

    Note: may return `np.nan`, which is still a valid float.
    """

    if pd.isna(timing_str) or timing_str in ("", None):
        return float("nan")

    text = str(timing_str).strip()
    try:
        parts = text.split(":")
        if len(parts) == 1:
            return float(parts[0]) * 1000.0
        if len(parts) == 2:
            return (float(parts[0]) * 60.0 + float(parts[1])) * 1000.0
        if len(parts) == 3:
            return (float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])) * 1000.0
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return float("nan")


def season_window(current_season: int | None = None, lookback_years: int = 10) -> list[int]:
    """Return the rolling training window of seasons ending at the active season."""

    active_season = int(current_season or detect_current_season())
    start_season = max(1950, active_season - int(lookback_years))
    return list(range(start_season, active_season + 1))


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a logger configured for both console and file output."""

    global _LOGGING_READY

    ensure_directories()
    if not _LOGGING_READY:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        json_formatter = _JsonFormatter()

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        handlers = setup_logging(CONFIG["logs_dir"])
        pipeline_handler = handlers["pipeline"]
        pipeline_handler.setFormatter(json_formatter)

        prediction_handler = handlers["predictions"]
        prediction_handler.setFormatter(json_formatter)
        prediction_handler.addFilter(_PredictionFilter())

        error_handler = handlers["errors"]
        error_handler.setFormatter(json_formatter)
        error_handler.addFilter(_ErrorFilter())

        root_logger.addHandler(console_handler)
        root_logger.addHandler(pipeline_handler)
        root_logger.addHandler(prediction_handler)
        root_logger.addHandler(error_handler)
        _LOGGING_READY = True

    return logging.getLogger(name)


def validate_columns(df: pd.DataFrame, expected_columns: Sequence[str], context: str) -> None:
    """Raise a clear error if a DataFrame is missing required columns."""

    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing expected columns: {missing}")


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Persist a DataFrame to CSV."""

    ensure_directories([path.parent])
    df.to_csv(path, index=index)


def load_dataframe(path: Path, parse_dates: Sequence[str] | None = None) -> pd.DataFrame:
    """Load a DataFrame from CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=list(parse_dates or []))


def save_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Persist a DataFrame to Parquet."""

    ensure_directories([path.parent])
    df.to_parquet(path, index=index)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a DataFrame from Parquet."""

    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path)


def _json_default(value: Any) -> Any:
    """Convert non-JSON-native objects into serializable values."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_json(data: Any, path: Path) -> None:
    """Persist a JSON-serializable structure to disk."""

    ensure_directories([path.parent])
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path, default: Any | None = None) -> Any:
    """Load JSON data from disk if present, otherwise return the provided default."""

    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_probabilities(values: Sequence[float], minimum: float = 1e-9) -> np.ndarray:
    """Return a probability vector that sums to one."""

    weights = np.asarray(values, dtype=float)
    if weights.size == 0:
        return weights
    weights = np.nan_to_num(weights, nan=minimum, posinf=minimum, neginf=minimum)
    weights = np.clip(weights, minimum, None)
    total = float(weights.sum())
    if total <= 0:
        return np.full(weights.shape, 1.0 / len(weights))
    return weights / total


def points_for_position(position: int, fastest_lap: bool = False) -> int:
    """Return championship points for a finishing position."""

    base_points = F1_POINTS.get(int(position), 0)
    if fastest_lap and 1 <= int(position) <= 10:
        base_points += 1
    return base_points


def sort_race_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a race-index-sorted copy of a DataFrame."""

    sort_columns = [column for column in ["season", "round", "finish_position", "qualifying_position"] if column in df.columns]
    return df.sort_values(sort_columns).reset_index(drop=True)


def circuit_feature_values(circuit_id: str) -> dict[str, Any]:
    """Return lookup-based circuit metadata with conservative defaults."""

    metadata = CIRCUIT_METADATA.get(
        str(circuit_id),
        {"overtake_difficulty": 0.5, "is_street_circuit": False},
    )
    return {
        "circuit_overtake_difficulty": float(metadata["overtake_difficulty"]),
        "is_street_circuit": bool(metadata["is_street_circuit"]),
    }


def latest_prior_rows(
    df: pd.DataFrame,
    group_column: str,
    season: int,
    round_number: int,
) -> pd.DataFrame:
    """Filter rows before a race and retain the latest row per group."""

    historical = df.loc[
        (df["season"] < season) | ((df["season"] == season) & (df["round"] < round_number))
    ].copy()
    if historical.empty:
        return historical
    historical = historical.sort_values(["season", "round"])
    return historical.groupby(group_column, as_index=False).tail(1)


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Compute a ratio while guarding against divide-by-zero."""

    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def coerce_numeric(series: pd.Series, fill_value: float | None = None) -> pd.Series:
    """Convert a Series to numeric form with optional NA fill."""

    numeric = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        numeric = numeric.fillna(fill_value)
    return numeric


def describe_season_split(df: pd.DataFrame) -> Mapping[str, int]:
    """Return counts by season for quick logging and validation."""

    validate_columns(df, ["season"], "season split frame")
    return df["season"].value_counts().sort_index().to_dict()


def slugify(value: Any) -> str:
    """Convert display text into a stable lowercase filename slug."""

    normalized = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    compact = "".join(ch.lower() if ch.isalnum() else "_" for ch in normalized)
    return "_".join(part for part in compact.split("_") if part)


def load_calendar_frame(season: int, allow_fetch: bool = False) -> pd.DataFrame:
    """Load a season calendar without introducing a module-level import cycle."""

    from src.data_pipeline import load_calendar

    return load_calendar(int(season), allow_fetch=allow_fetch)


def get_gp_name(season: int, round_number: int) -> str:
    """Return the official Grand Prix name for a season round."""

    calendar = load_calendar_frame(season, allow_fetch=False)
    race = calendar.loc[calendar["round"] == int(round_number)].head(1)
    if race.empty:
        return f"Round {int(round_number)}"
    return str(race["race_name"].iloc[0])


def get_circuit_name(season: int, round_number: int) -> str:
    """Return the official circuit name for a season round."""

    calendar = load_calendar_frame(season, allow_fetch=False)
    race = calendar.loc[calendar["round"] == int(round_number)].head(1)
    if race.empty:
        return "Unknown Circuit"
    if "circuit_name" in race.columns and pd.notna(race["circuit_name"].iloc[0]):
        return str(race["circuit_name"].iloc[0])
    return str(race["circuit_id"].iloc[0]).replace("_", " ").title()


def format_race_label(season: int, round_number: int) -> str:
    """Return a fully formatted GP label for dashboards and logs."""

    calendar = load_calendar_frame(season, allow_fetch=False)
    total_rounds = int(calendar["round"].max()) if not calendar.empty else int(round_number)
    gp_name = get_gp_name(season, round_number)
    return f"{gp_name} - Round {int(round_number)} of {total_rounds} ({int(season)})"
