"""Strict, versioned contracts shared by CLI, API, training and forecasting."""

from __future__ import annotations
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
import hashlib
import json
import math
import os
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "2.0"
FEATURE_VERSION = "pit-v2.3"


class Stage(StrEnum):
    PRE_WEEKEND = "pre_weekend"
    POST_PRACTICE = "post_practice"
    POST_QUALIFYING = "post_qualifying"
    POST_SPRINT = "post_sprint"
    LIVE = "live"


class Evidence(StrEnum):
    RECONSTRUCTION = "historical_reconstruction"
    HOLDOUT = "held_out_evaluation"
    PROSPECTIVE = "prospective_forecast"
    ILLUSTRATIVE = "illustrative_scenario"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Scenario(StrictModel):
    grid: dict[str, int] = Field(default_factory=dict)
    teams: dict[str, str] = Field(default_factory=dict)
    rain_probability: float | None = Field(default=None, ge=0, le=1)
    strength_delta: dict[str, float] = Field(default_factory=dict)

    @field_validator("grid")
    @classmethod
    def valid_grid(cls, value):
        if any(v < 1 for v in value.values()) or len(set(value.values())) != len(value):
            raise ValueError("Grid positions must be positive and unique")
        return value

    @field_validator("strength_delta")
    @classmethod
    def finite_delta(cls, value):
        if any(not math.isfinite(v) or abs(v) > 3 for v in value.values()):
            raise ValueError("Strength adjustments must be finite and within [-3, 3]")
        return value


class ForecastRequest(StrictModel):
    event_id: str = Field(pattern=r"^\d{4}-\d{2}$")
    stage: Stage = Stage.PRE_WEEKEND
    as_of_utc: datetime
    data_version: str | None = None
    model_version: str = "baseline:rating"
    scenario: Scenario | None = None
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    simulations: int = Field(default=10000, ge=100, le=100000)
    revision_of: str | None = None

    @field_validator("as_of_utc")
    @classmethod
    def aware_time(cls, value):
        if value.tzinfo is None:
            raise ValueError("as_of_utc must include a timezone")
        return value.astimezone(timezone.utc)


class SeasonRequest(StrictModel):
    season: int = Field(ge=2014, le=2100)
    as_of_utc: datetime
    data_version: str | None = None
    simulations: int = Field(default=10000, ge=100, le=100000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    revision_of: str | None = None

    @field_validator("as_of_utc")
    @classmethod
    def aware_time(cls, value):
        return ForecastRequest.aware_time(value)


class JobRequest(StrictModel):
    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Unavailable(ValueError):
    """A requested result cannot be supported by available evidence."""


class Settings:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("F1ENGINE_HOME", ".f1engine")).resolve()
        self.threads = 4
        self.job_budget_seconds = 45 * 60
        self.max_candidates = 12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp(value: str | datetime) -> datetime:
    result = (
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if not isinstance(value, datetime)
        else value
    )
    if result.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value}")
    return result.astimezone(timezone.utc)


def clean(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return clean(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, Path, StrEnum)):
        return str(value)
    return value


def canonical(value: Any) -> str:
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _source_code_version() -> str:
    source = Path(__file__).parent
    return digest({p.name: p.read_text() for p in sorted(source.glob("*.py"))})


_LOADED_CODE_VERSION = _source_code_version()


def code_version() -> str:
    # An already-imported process must not claim newly edited files as its running code.
    if _source_code_version() != _LOADED_CODE_VERSION:
        raise Unavailable("Application source changed while running; restart before generating new evidence")
    return _LOADED_CODE_VERSION
