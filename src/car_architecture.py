"""Constructor car-architecture profiling and circuit-fit analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.circuit_analysis import CIRCUIT_DEMANDS
from src.utils import (
    CONFIG as BASE_CONFIG,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_dataframe,
    save_json,
    timing_to_milliseconds,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "fastf1_features_path": BASE_CONFIG["processed_dir"] / "fastf1_features.csv",
    "history_path": BASE_CONFIG["processed_dir"] / "car_architecture_history.csv",
    "summary_path": BASE_CONFIG["processed_dir"] / "car_architecture_summary.csv",
    "metadata_path": BASE_CONFIG["processed_dir"] / "car_architecture_metadata.json",
}

LOGGER = get_logger(__name__)

ARCHITECTURE_DIMENSIONS: list[str] = [
    "straight_line_speed",
    "slow_corner_grip",
    "high_speed_cornering",
    "mechanical_grip",
    "tyre_management",
    "downforce_level",
]

_HIGH_SPEED_SECTOR = {
    "spa": "sector_1_pct",
    "silverstone": "sector_3_pct",
    "suzuka": "sector_1_pct",
    "zandvoort": "sector_1_pct",
    "mugello": "sector_2_pct",
}
_SLOW_CORNER_SECTOR = {
    "monaco": "sector_1_pct",
    "hungaroring": "sector_2_pct",
    "marina_bay": "sector_3_pct",
    "baku": "sector_1_pct",
}
_MECHANICAL_GRIP_SECTOR = {
    "baku": "sector_1_pct",
    "monaco": "sector_1_pct",
    "rodriguez": "sector_3_pct",
    "villeneuve": "sector_2_pct",
    "albert_park": "sector_3_pct",
}

_PRIOR_VALUES_2026: dict[str, dict[str, Any]] = {
    "mercedes": {
        "straight_line_speed": 0.92,
        "slow_corner_grip": 0.61,
        "high_speed_cornering": 0.78,
        "mechanical_grip": 0.71,
        "tyre_management": 0.74,
        "downforce_level": 0.45,
        "confidence": 0.40,
        "notes": "2026 active aero dominant on straights",
    },
    "ferrari": {
        "straight_line_speed": 0.71,
        "slow_corner_grip": 0.91,
        "high_speed_cornering": 0.89,
        "mechanical_grip": 0.83,
        "tyre_management": 0.79,
        "downforce_level": 0.82,
        "confidence": 0.40,
        "notes": "2026 high downforce, cornering specialist",
    },
    "red_bull": {
        "straight_line_speed": 0.78,
        "slow_corner_grip": 0.81,
        "high_speed_cornering": 0.85,
        "mechanical_grip": 0.86,
        "tyre_management": 0.88,
        "downforce_level": 0.72,
        "confidence": 0.40,
        "notes": "2026 regulation adaptation uncertain but balanced",
    },
    "mclaren": {
        "straight_line_speed": 0.83,
        "slow_corner_grip": 0.79,
        "high_speed_cornering": 0.84,
        "mechanical_grip": 0.81,
        "tyre_management": 0.85,
        "downforce_level": 0.68,
        "confidence": 0.40,
        "notes": "2025 champion baseline carried into 2026",
    },
    "aston_martin": {
        "straight_line_speed": 0.73,
        "slow_corner_grip": 0.72,
        "high_speed_cornering": 0.74,
        "mechanical_grip": 0.71,
        "tyre_management": 0.76,
        "downforce_level": 0.65,
        "confidence": 0.35,
        "notes": "Stable but not extreme in any single dimension",
    },
    "alpine": {
        "straight_line_speed": 0.69,
        "slow_corner_grip": 0.66,
        "high_speed_cornering": 0.70,
        "mechanical_grip": 0.64,
        "tyre_management": 0.67,
        "downforce_level": 0.59,
        "confidence": 0.35,
        "notes": "Middle-of-grid package with modest aero load",
    },
    "audi": {
        "straight_line_speed": 0.77,
        "slow_corner_grip": 0.68,
        "high_speed_cornering": 0.72,
        "mechanical_grip": 0.69,
        "tyre_management": 0.71,
        "downforce_level": 0.56,
        "confidence": 0.34,
        "notes": "Power-unit efficiency expected to be a strength",
    },
    "rb": {
        "straight_line_speed": 0.74,
        "slow_corner_grip": 0.72,
        "high_speed_cornering": 0.75,
        "mechanical_grip": 0.70,
        "tyre_management": 0.72,
        "downforce_level": 0.63,
        "confidence": 0.35,
        "notes": "Red Bull family DNA with less absolute load",
    },
    "haas": {
        "straight_line_speed": 0.76,
        "slow_corner_grip": 0.62,
        "high_speed_cornering": 0.66,
        "mechanical_grip": 0.63,
        "tyre_management": 0.60,
        "downforce_level": 0.50,
        "confidence": 0.34,
        "notes": "Straight-line biased with higher tyre stress",
    },
    "williams": {
        "straight_line_speed": 0.84,
        "slow_corner_grip": 0.58,
        "high_speed_cornering": 0.67,
        "mechanical_grip": 0.61,
        "tyre_management": 0.63,
        "downforce_level": 0.43,
        "confidence": 0.34,
        "notes": "Historic low-drag tendency retained",
    },
    "cadillac": {
        "straight_line_speed": 0.70,
        "slow_corner_grip": 0.64,
        "high_speed_cornering": 0.66,
        "mechanical_grip": 0.65,
        "tyre_management": 0.68,
        "downforce_level": 0.58,
        "confidence": 0.30,
        "notes": "Expansion-team prior with conservative balance",
    },
}


@dataclass(slots=True)
class CarArchitecture:
    """Constructor architecture snapshot available for a target round."""

    constructor_id: str
    season: int
    round: int
    straight_line_speed: float
    slow_corner_grip: float
    high_speed_cornering: float
    mechanical_grip: float
    tyre_management: float
    downforce_level: float
    confidence: float
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        for column in [*ARCHITECTURE_DIMENSIONS, "confidence"]:
            row[column] = float(np.clip(row[column], 0.0, 1.0))
        row["constructor_id"] = str(self.constructor_id)
        row["season"] = int(self.season)
        row["round"] = int(self.round)
        row["architecture_signature"] = (
            f"SLS {row['straight_line_speed']:.2f} | "
            f"SCG {row['slow_corner_grip']:.2f} | "
            f"HSC {row['high_speed_cornering']:.2f} | "
            f"MECH {row['mechanical_grip']:.2f} | "
            f"TYRE {row['tyre_management']:.2f} | "
            f"DF {row['downforce_level']:.2f}"
        )
        return row


@dataclass(slots=True)
class CarCircuitFit:
    """How well a car architecture matches a circuit demand profile."""

    fit_score: float
    fit_components: dict[str, float]
    primary_advantage: str
    primary_disadvantage: str
    confidence: float
    notes: str


@dataclass(slots=True)
class CarArchitectureArtifacts:
    """Persisted architecture artifacts."""

    history: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, Any]


def _clip01(value: Any, default: float = 0.5) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def _normalize_driver_number(value: Any) -> str | None:
    if pd.isna(value) or value in (None, ""):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _rank_percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(0.5, index=series.index, dtype=float)
    if valid.nunique() <= 1:
        result = pd.Series(0.5, index=series.index, dtype=float)
        result.loc[valid.index] = 0.5
        return result
    ranked = valid.rank(method="average", ascending=not higher_is_better)
    normalized = 1.0 - (ranked - 1.0) / max(len(valid) - 1.0, 1.0)
    result = pd.Series(0.5, index=series.index, dtype=float)
    result.loc[normalized.index] = normalized.astype(float)
    return result


def _neutral_architecture(
    constructor_id: str,
    season: int,
    round_number: int,
    confidence: float = 0.20,
    notes: str = "neutral baseline",
) -> CarArchitecture:
    return CarArchitecture(
        constructor_id=str(constructor_id),
        season=int(season),
        round=int(round_number),
        straight_line_speed=0.50,
        slow_corner_grip=0.50,
        high_speed_cornering=0.50,
        mechanical_grip=0.50,
        tyre_management=0.50,
        downforce_level=0.50,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        notes=str(notes),
    )


def _prior_architecture(constructor_id: str, season: int, round_number: int = 1) -> CarArchitecture | None:
    if int(season) != 2026:
        return None
    prior = _PRIOR_VALUES_2026.get(str(constructor_id))
    if prior is None:
        return None
    return CarArchitecture(
        constructor_id=str(constructor_id),
        season=int(season),
        round=int(round_number),
        straight_line_speed=_clip01(prior["straight_line_speed"]),
        slow_corner_grip=_clip01(prior["slow_corner_grip"]),
        high_speed_cornering=_clip01(prior["high_speed_cornering"]),
        mechanical_grip=_clip01(prior["mechanical_grip"]),
        tyre_management=_clip01(prior["tyre_management"]),
        downforce_level=_clip01(prior["downforce_level"]),
        confidence=_clip01(prior.get("confidence", 0.40)),
        notes=str(prior.get("notes", "")),
    )


def _blend_architecture(
    season_prior: CarArchitecture,
    evidence_means: dict[str, float],
    target_round: int,
    n_races: int,
) -> CarArchitecture:
    update_weight = float(np.clip(1.0 - (0.9 ** max(int(n_races), 0)), 0.0, 1.0))
    values = {
        dimension: _clip01(
            (1.0 - update_weight) * getattr(season_prior, dimension)
            + update_weight * evidence_means.get(dimension, getattr(season_prior, dimension))
        )
        for dimension in ARCHITECTURE_DIMENSIONS
    }
    confidence = float(np.clip(season_prior.confidence + 0.08 * max(int(n_races), 0), 0.0, 1.0))
    notes = (
        f"{season_prior.notes or 'data-driven blend'}; "
        f"{max(int(n_races), 0)} race evidence inputs, weight={update_weight:.2f}"
    )
    return CarArchitecture(
        constructor_id=season_prior.constructor_id,
        season=int(season_prior.season),
        round=int(target_round),
        confidence=confidence,
        notes=notes,
        **values,
    )


def _constructor_event_frame(
    race_results: pd.DataFrame,
    fastf1_features: pd.DataFrame,
) -> pd.DataFrame:
    if race_results.empty:
        return pd.DataFrame()

    results = race_results.copy()
    results["driver_number"] = results["driver_number"].apply(_normalize_driver_number)
    results["best_qualifying_time_ms"] = (
        pd.concat(
            [
                results.get("q3", pd.Series(index=results.index)).map(timing_to_milliseconds),
                results.get("q2", pd.Series(index=results.index)).map(timing_to_milliseconds),
                results.get("q1", pd.Series(index=results.index)).map(timing_to_milliseconds),
            ],
            axis=1,
        )
        .bfill(axis=1)
        .iloc[:, 0]
    )
    results["pole_time_ms"] = results.groupby(["season", "round"])["best_qualifying_time_ms"].transform("min")
    results["qualifying_gap_pct"] = (
        results["best_qualifying_time_ms"] - results["pole_time_ms"]
    ) / np.clip(results["pole_time_ms"], 1.0, None)

    features = fastf1_features.copy() if not fastf1_features.empty else pd.DataFrame()
    if not features.empty:
        features["driver_number"] = features["driver_number"].apply(_normalize_driver_number)
        features["speed_trap_proxy"] = pd.concat(
            [
                pd.to_numeric(features.get("openf1_max_speed_trap"), errors="coerce"),
                pd.to_numeric(features.get("max_speed_trap"), errors="coerce"),
            ],
            axis=1,
        ).bfill(axis=1).iloc[:, 0]
        deg_candidates = [
            pd.to_numeric(features.get("tyre_degradation_rate"), errors="coerce"),
            pd.to_numeric(features.get("fp_soft_deg_rate"), errors="coerce"),
            pd.to_numeric(features.get("fp_medium_deg_rate"), errors="coerce"),
            pd.to_numeric(features.get("fp_hard_deg_rate"), errors="coerce"),
        ]
        features["deg_rate_proxy"] = pd.concat(deg_candidates, axis=1).mean(axis=1, skipna=True)
        feature_columns = [
            "season",
            "round",
            "driver_number",
            "speed_trap_proxy",
            "deg_rate_proxy",
            "best_sector_1_ms",
            "best_sector_2_ms",
            "best_sector_3_ms",
        ]
        features = features[feature_columns].drop_duplicates(
            subset=["season", "round", "driver_number"],
            keep="last",
        )
        results = results.merge(features, how="left", on=["season", "round", "driver_number"])
    else:
        results["speed_trap_proxy"] = np.nan
        results["deg_rate_proxy"] = np.nan
        results["best_sector_1_ms"] = np.nan
        results["best_sector_2_ms"] = np.nan
        results["best_sector_3_ms"] = np.nan

    constructor_frame = (
        results.groupby(
            ["season", "round", "circuit_id", "constructor_id", "constructor_name"],
            as_index=False,
        )
        .agg(
            speed_trap_proxy=("speed_trap_proxy", "mean"),
            deg_rate_proxy=("deg_rate_proxy", "mean"),
            best_sector_1_ms=("best_sector_1_ms", "mean"),
            best_sector_2_ms=("best_sector_2_ms", "mean"),
            best_sector_3_ms=("best_sector_3_ms", "mean"),
            qualifying_gap_pct=("qualifying_gap_pct", "mean"),
            drivers=("driver_id", "nunique"),
        )
        .sort_values(["season", "round", "constructor_id"])
        .reset_index(drop=True)
    )

    group_keys = ["season", "round"]
    constructor_frame["speed_pct"] = constructor_frame.groupby(group_keys)["speed_trap_proxy"].transform(
        lambda series: _rank_percentile(series, higher_is_better=True)
    )
    constructor_frame["sector_1_pct"] = constructor_frame.groupby(group_keys)["best_sector_1_ms"].transform(
        lambda series: _rank_percentile(series, higher_is_better=False)
    )
    constructor_frame["sector_2_pct"] = constructor_frame.groupby(group_keys)["best_sector_2_ms"].transform(
        lambda series: _rank_percentile(series, higher_is_better=False)
    )
    constructor_frame["sector_3_pct"] = constructor_frame.groupby(group_keys)["best_sector_3_ms"].transform(
        lambda series: _rank_percentile(series, higher_is_better=False)
    )
    constructor_frame["pace_pct"] = constructor_frame.groupby(group_keys)["qualifying_gap_pct"].transform(
        lambda series: _rank_percentile(series, higher_is_better=False)
    )
    constructor_frame["deg_pct"] = constructor_frame.groupby(group_keys)["deg_rate_proxy"].transform(
        lambda series: _rank_percentile(series, higher_is_better=False)
    )
    for column in ["speed_pct", "sector_1_pct", "sector_2_pct", "sector_3_pct", "pace_pct", "deg_pct"]:
        constructor_frame[column] = pd.to_numeric(constructor_frame[column], errors="coerce").fillna(0.5)
    return constructor_frame


def _sector_signal(row: pd.Series, mapping: dict[str, str], fallback: float) -> float:
    column = mapping.get(str(row["circuit_id"]))
    if column is None:
        return float(fallback)
    value = row.get(column)
    if pd.isna(value):
        return float(fallback)
    return float(np.clip(float(value), 0.0, 1.0))


def _downforce_evidence(pace_pct: float, demand_downforce: float) -> float:
    center = float(np.clip(demand_downforce, 0.0, 1.0))
    offset = (float(np.clip(pace_pct, 0.0, 1.0)) - 0.5) * 0.8
    if center >= 0.60:
        return float(np.clip(center + offset, 0.0, 1.0))
    if center <= 0.40:
        return float(np.clip(center - offset, 0.0, 1.0))
    return float(np.clip(center + 0.25 * offset, 0.0, 1.0))


def _event_architecture_evidence(row: pd.Series) -> dict[str, float]:
    demand = CIRCUIT_DEMANDS.get(str(row["circuit_id"]))
    if demand is None:
        demand = CIRCUIT_DEMANDS["bahrain"]
    pace_pct = float(np.clip(row.get("pace_pct", 0.5), 0.0, 1.0))
    speed_pct = float(np.clip(row.get("speed_pct", 0.5), 0.0, 1.0))
    deg_pct = float(np.clip(row.get("deg_pct", 0.5), 0.0, 1.0))
    slow_sector = _sector_signal(row, _SLOW_CORNER_SECTOR, fallback=0.55 * row.get("sector_2_pct", 0.5) + 0.45 * pace_pct)
    high_sector = _sector_signal(row, _HIGH_SPEED_SECTOR, fallback=0.45 * row.get("sector_1_pct", 0.5) + 0.55 * pace_pct)
    mech_sector = _sector_signal(row, _MECHANICAL_GRIP_SECTOR, fallback=0.5 * row.get("sector_1_pct", 0.5) + 0.5 * pace_pct)

    return {
        "straight_line_speed": _clip01(
            0.65 * speed_pct + 0.35 * (pace_pct if demand.straight_line_speed >= 0.65 else 0.5)
        ),
        "slow_corner_grip": _clip01(
            0.60 * slow_sector + 0.40 * (pace_pct if demand.slow_corner_grip >= 0.60 else 0.5)
        ),
        "high_speed_cornering": _clip01(
            0.60 * high_sector + 0.40 * (pace_pct if demand.high_speed_cornering >= 0.60 else 0.5)
        ),
        "mechanical_grip": _clip01(
            0.55 * mech_sector + 0.20 * deg_pct + 0.25 * (pace_pct if demand.mechanical_grip >= 0.60 else 0.5)
        ),
        "tyre_management": _clip01(
            0.75 * deg_pct + 0.25 * (pace_pct if demand.tyre_management >= 0.60 else 0.5)
        ),
        "downforce_level": _downforce_evidence(pace_pct=pace_pct, demand_downforce=demand.downforce_level),
    }


def _season_start_architecture(
    constructor_id: str,
    season: int,
    previous_snapshot: CarArchitecture | None,
) -> CarArchitecture:
    prior = _prior_architecture(constructor_id, season, round_number=1)
    if prior is not None:
        return prior
    if previous_snapshot is not None:
        carried = previous_snapshot.as_row()
        return CarArchitecture(
            constructor_id=str(constructor_id),
            season=int(season),
            round=1,
            straight_line_speed=_clip01(carried["straight_line_speed"]),
            slow_corner_grip=_clip01(carried["slow_corner_grip"]),
            high_speed_cornering=_clip01(carried["high_speed_cornering"]),
            mechanical_grip=_clip01(carried["mechanical_grip"]),
            tyre_management=_clip01(carried["tyre_management"]),
            downforce_level=_clip01(carried["downforce_level"]),
            confidence=float(np.clip(float(carried["confidence"]) * 0.80, 0.25, 0.90)),
            notes=f"carry-over from {previous_snapshot.season}",
        )
    return _neutral_architecture(
        constructor_id=str(constructor_id),
        season=int(season),
        round_number=1,
        confidence=0.24,
        notes="neutral baseline before sufficient data",
    )


def build_car_architecture_artifacts(
    race_results: pd.DataFrame | None = None,
    fastf1_features: pd.DataFrame | None = None,
) -> CarArchitectureArtifacts:
    """Build architecture history snapshots keyed by target round."""

    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    if fastf1_features is None:
        fastf1_features = load_dataframe(CONFIG["fastf1_features_path"])

    constructor_events = _constructor_event_frame(race_results, fastf1_features)
    season_order = sorted(pd.to_numeric(race_results.get("season"), errors="coerce").dropna().astype(int).unique().tolist())
    if 2026 not in season_order and any(key in _PRIOR_VALUES_2026 for key in _PRIOR_VALUES_2026):
        season_order.append(2026)
        season_order = sorted(set(season_order))

    history_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"seasons": season_order, "validation_checks": {}}
    carry_over: dict[str, CarArchitecture] = {}

    for season in season_order:
        season_events = constructor_events.loc[constructor_events["season"] == int(season)].copy()
        constructors = set(season_events["constructor_id"].astype(str).unique().tolist())
        if int(season) == 2026:
            constructors.update(_PRIOR_VALUES_2026.keys())
        if not constructors:
            continue

        season_priors: dict[str, CarArchitecture] = {}
        evidence_history: dict[str, list[dict[str, float]]] = {}
        current_snapshots: dict[str, CarArchitecture] = {}
        for constructor_id in sorted(constructors):
            start_arch = _season_start_architecture(constructor_id, season, carry_over.get(constructor_id))
            season_priors[constructor_id] = start_arch
            evidence_history[constructor_id] = []
            current_snapshots[constructor_id] = start_arch
            history_rows.append(start_arch.as_row())

        for round_number in sorted(season_events["round"].astype(int).unique().tolist()):
            round_events = season_events.loc[season_events["round"] == int(round_number)].copy()
            active_constructors = sorted(round_events["constructor_id"].astype(str).unique().tolist())
            for constructor_id in active_constructors:
                event_row = round_events.loc[round_events["constructor_id"] == str(constructor_id)].head(1)
                if event_row.empty:
                    continue
                evidence_history[constructor_id].append(_event_architecture_evidence(event_row.iloc[0]))
                evidence_frame = pd.DataFrame(evidence_history[constructor_id])
                evidence_means = {
                    dimension: float(evidence_frame[dimension].mean()) if dimension in evidence_frame.columns else getattr(season_priors[constructor_id], dimension)
                    for dimension in ARCHITECTURE_DIMENSIONS
                }
                current_snapshots[constructor_id] = _blend_architecture(
                    season_prior=season_priors[constructor_id],
                    evidence_means=evidence_means,
                    target_round=int(round_number) + 1,
                    n_races=len(evidence_history[constructor_id]),
                )
            for constructor_id in active_constructors:
                history_rows.append(current_snapshots[constructor_id].as_row())

        for constructor_id, snapshot in current_snapshots.items():
            carry_over[constructor_id] = snapshot

    history = pd.DataFrame(history_rows).drop_duplicates(
        subset=["constructor_id", "season", "round"],
        keep="last",
    )
    if history.empty:
        history = pd.DataFrame(columns=["constructor_id", "season", "round", *ARCHITECTURE_DIMENSIONS, "confidence", "notes", "architecture_signature"])
    if not race_results.empty and "constructor_name" in race_results.columns:
        constructor_names = (
            race_results[["constructor_id", "constructor_name"]]
            .dropna()
            .drop_duplicates(subset=["constructor_id"], keep="last")
        )
        history = history.merge(constructor_names, how="left", on="constructor_id")
    history = history.sort_values(["season", "round", "constructor_id"]).reset_index(drop=True)
    summary = (
        history.sort_values(["constructor_id", "season", "round"])
        .groupby(["season", "constructor_id"], as_index=False)
        .tail(1)
        .sort_values(["season", "constructor_id"])
        .reset_index(drop=True)
    )
    metadata["rows"] = int(len(history))
    metadata["seasons_present"] = sorted(history["season"].dropna().astype(int).unique().tolist()) if not history.empty else []
    return CarArchitectureArtifacts(history=history, summary=summary, metadata=metadata)


def build_constructor_event_evidence_frame(
    race_results: pd.DataFrame,
    fastf1_features: pd.DataFrame,
) -> pd.DataFrame:
    """Expose the constructor-round evidence frame for live post-race updates."""

    return _constructor_event_frame(race_results=race_results, fastf1_features=fastf1_features)


def run_car_architecture_pipeline(
    race_results: pd.DataFrame | None = None,
    fastf1_features: pd.DataFrame | None = None,
) -> CarArchitectureArtifacts:
    """Build and persist architecture artifacts."""

    ensure_directories()
    artifacts = build_car_architecture_artifacts(race_results=race_results, fastf1_features=fastf1_features)
    save_dataframe(artifacts.history, CONFIG["history_path"])
    save_dataframe(artifacts.summary, CONFIG["summary_path"])
    save_json(artifacts.metadata, CONFIG["metadata_path"])
    LOGGER.info("Saved car architecture artifacts | history=%s summary=%s", len(artifacts.history), len(artifacts.summary))
    return artifacts


def load_car_architecture_history() -> pd.DataFrame:
    path = Path(CONFIG["history_path"])
    if not path.exists():
        return run_car_architecture_pipeline().history
    return load_dataframe(path)


def load_car_architecture_summary() -> pd.DataFrame:
    path = Path(CONFIG["summary_path"])
    if not path.exists():
        return run_car_architecture_pipeline().summary
    return load_dataframe(path)


def get_car_architecture(
    constructor_id: str,
    season: int,
    round_number: int,
) -> CarArchitecture:
    """Return the latest architecture snapshot available for the requested target round."""

    history = load_car_architecture_history()
    if not history.empty:
        subset = history.loc[
            (history["constructor_id"] == str(constructor_id))
            & (pd.to_numeric(history["season"], errors="coerce") == int(season))
            & (pd.to_numeric(history["round"], errors="coerce") <= int(round_number))
        ].sort_values(["season", "round"])
        if not subset.empty:
            row = subset.iloc[-1]
            return CarArchitecture(
                constructor_id=str(row["constructor_id"]),
                season=int(row["season"]),
                round=int(row["round"]),
                straight_line_speed=_clip01(row["straight_line_speed"]),
                slow_corner_grip=_clip01(row["slow_corner_grip"]),
                high_speed_cornering=_clip01(row["high_speed_cornering"]),
                mechanical_grip=_clip01(row["mechanical_grip"]),
                tyre_management=_clip01(row["tyre_management"]),
                downforce_level=_clip01(row["downforce_level"]),
                confidence=_clip01(row["confidence"]),
                notes=str(row.get("notes", "")),
            )
        prior_seasons = history.loc[
            (history["constructor_id"] == str(constructor_id))
            & (pd.to_numeric(history["season"], errors="coerce") < int(season))
        ].sort_values(["season", "round"])
        if not prior_seasons.empty:
            row = prior_seasons.iloc[-1]
            return CarArchitecture(
                constructor_id=str(constructor_id),
                season=int(season),
                round=int(round_number),
                straight_line_speed=_clip01(row["straight_line_speed"]),
                slow_corner_grip=_clip01(row["slow_corner_grip"]),
                high_speed_cornering=_clip01(row["high_speed_cornering"]),
                mechanical_grip=_clip01(row["mechanical_grip"]),
                tyre_management=_clip01(row["tyre_management"]),
                downforce_level=_clip01(row["downforce_level"]),
                confidence=float(np.clip(float(row["confidence"]) * 0.80, 0.25, 0.90)),
                notes=f"carry-over from {int(row['season'])}",
            )

    prior = _prior_architecture(constructor_id, season, round_number)
    if prior is not None:
        return prior
    return _neutral_architecture(str(constructor_id), int(season), int(round_number))


def compute_car_architecture(
    constructor_id: str,
    season: int,
    round_number: int | None = None,
) -> CarArchitecture:
    """Public convenience wrapper for retrieving an architecture snapshot."""

    target_round = 99 if round_number is None else int(round_number)
    return get_car_architecture(constructor_id=str(constructor_id), season=int(season), round_number=target_round)


def compute_car_circuit_fit(
    constructor_id: str,
    circuit_id: str,
    season: int,
    round_number: int,
) -> CarCircuitFit:
    """Score how well a constructor architecture matches a circuit's demands."""

    architecture = get_car_architecture(constructor_id, season, round_number)
    demands = CIRCUIT_DEMANDS.get(str(circuit_id), CIRCUIT_DEMANDS["bahrain"])
    fit_components: dict[str, float] = {}
    primary_dimension = str(demands.primary_differentiator)

    for dimension in [
        "straight_line_speed",
        "slow_corner_grip",
        "high_speed_cornering",
        "mechanical_grip",
        "tyre_management",
    ]:
        demand_weight = float(np.clip(getattr(demands, dimension), 0.0, 1.0))
        arch_score = float(np.clip(getattr(architecture, dimension), 0.0, 1.0))
        surplus_bonus = max(arch_score - demand_weight, 0.0) * demand_weight * 0.35
        shortfall_penalty = max(demand_weight - arch_score, 0.0) * demand_weight * 0.90
        component = surplus_bonus - shortfall_penalty
        if dimension == primary_dimension:
            component += (arch_score - 0.5) * 0.45
        fit_components[dimension] = float(component)

    downforce_importance = max(0.25, abs(float(demands.downforce_level) - 0.5) * 2.0)
    downforce_mismatch = abs(float(architecture.downforce_level) - float(demands.downforce_level))
    fit_components["downforce_fit"] = float((0.5 - downforce_mismatch) * downforce_importance)

    raw_fit = float(sum(fit_components.values()))
    fit_score = float(np.tanh(raw_fit * 2.0))
    primary_advantage = max(fit_components, key=fit_components.get)
    primary_disadvantage = min(fit_components, key=fit_components.get)
    tone = "advantage" if fit_score > 0.10 else "disadvantage" if fit_score < -0.10 else "neutral"
    notes = (
        f"{constructor_id} at {circuit_id}: {tone}. "
        f"Primary edge {primary_advantage.replace('_', ' ')}, "
        f"biggest weakness {primary_disadvantage.replace('_', ' ')}."
    )
    return CarCircuitFit(
        fit_score=fit_score,
        fit_components=fit_components,
        primary_advantage=primary_advantage,
        primary_disadvantage=primary_disadvantage,
        confidence=float(architecture.confidence),
        notes=notes,
    )


def _extract_from_mapping(payload: dict[str, Any], constructor_id: str, aliases: list[str]) -> float | None:
    lowered = str(constructor_id)
    if lowered in payload and isinstance(payload[lowered], dict):
        nested = payload[lowered]
        for alias in aliases:
            if alias in nested and nested[alias] is not None:
                return _clip01(nested[alias])
    for alias in aliases:
        value = payload.get(alias)
        if isinstance(value, dict) and lowered in value and value[lowered] is not None:
            return _clip01(value[lowered])
        if alias in payload and not isinstance(payload[alias], dict):
            return _clip01(payload[alias])
    constructors = payload.get("constructors")
    if isinstance(constructors, dict) and lowered in constructors and isinstance(constructors[lowered], dict):
        nested = constructors[lowered]
        for alias in aliases:
            if alias in nested and nested[alias] is not None:
                return _clip01(nested[alias])
    return None


def _extract_evidence_dict(
    constructor_id: str,
    race_data: Any,
) -> dict[str, float]:
    aliases = {
        "straight_line_speed": ["straight_line_speed", "speed_trap_evidence", "new_speed_trap"],
        "slow_corner_grip": ["slow_corner_grip", "slow_corner_evidence"],
        "high_speed_cornering": ["high_speed_cornering", "high_speed_evidence"],
        "mechanical_grip": ["mechanical_grip", "mechanical_grip_evidence"],
        "tyre_management": ["tyre_management", "deg_rate_evidence", "new_deg_rate"],
        "downforce_level": ["downforce_level", "downforce_evidence"],
    }
    evidence: dict[str, float] = {}

    if isinstance(race_data, pd.DataFrame) and not race_data.empty:
        frame = race_data.copy()
        if "constructor_id" in frame.columns:
            frame = frame.loc[frame["constructor_id"] == str(constructor_id)]
        if not frame.empty:
            row = frame.iloc[0]
            if set(ARCHITECTURE_DIMENSIONS).issubset(frame.columns):
                return {dimension: _clip01(row[dimension]) for dimension in ARCHITECTURE_DIMENSIONS}
            if {"season", "round", "circuit_id", "constructor_id"}.issubset(frame.columns):
                return _event_architecture_evidence(row)

    if isinstance(race_data, pd.Series):
        if set(ARCHITECTURE_DIMENSIONS).issubset(race_data.index):
            return {dimension: _clip01(race_data[dimension]) for dimension in ARCHITECTURE_DIMENSIONS}
        return _event_architecture_evidence(race_data)

    if isinstance(race_data, dict):
        for dimension, keys in aliases.items():
            value = _extract_from_mapping(race_data, str(constructor_id), keys)
            if value is not None:
                evidence[dimension] = value
        if evidence:
            return evidence

    return evidence


def save_car_architecture(architecture: CarArchitecture) -> None:
    """Persist a single architecture snapshot into the history and summary artifacts."""

    history = load_car_architecture_history()
    row = pd.DataFrame([architecture.as_row()])
    if "constructor_name" in history.columns:
        match = history.loc[history["constructor_id"] == str(architecture.constructor_id), "constructor_name"]
        row["constructor_name"] = str(match.dropna().iloc[-1]) if not match.dropna().empty else str(architecture.constructor_id).replace("_", " ").title()
    if history.empty:
        history = row
    else:
        history = pd.concat([history, row], ignore_index=True)
        history = history.drop_duplicates(subset=["constructor_id", "season", "round"], keep="last")
    history = history.sort_values(["season", "round", "constructor_id"]).reset_index(drop=True)
    save_dataframe(history, CONFIG["history_path"])

    summary = (
        history.sort_values(["constructor_id", "season", "round"])
        .groupby(["season", "constructor_id"], as_index=False)
        .tail(1)
        .sort_values(["season", "constructor_id"])
        .reset_index(drop=True)
    )
    save_dataframe(summary, CONFIG["summary_path"])


def update_car_architecture_from_race(
    constructor_id: str,
    season: int,
    round_number: int,
    race_data: Any,
) -> CarArchitecture:
    """Update a constructor architecture snapshot for the requested target round."""

    target_round = int(round_number)
    season = int(season)
    evidence = _extract_evidence_dict(str(constructor_id), race_data)
    previous_snapshot = get_car_architecture(str(constructor_id), season, max(target_round - 1, 1))
    season_prior = get_car_architecture(str(constructor_id), season, 1)

    if not evidence:
        retained = CarArchitecture(
            constructor_id=str(constructor_id),
            season=season,
            round=target_round,
            straight_line_speed=previous_snapshot.straight_line_speed,
            slow_corner_grip=previous_snapshot.slow_corner_grip,
            high_speed_cornering=previous_snapshot.high_speed_cornering,
            mechanical_grip=previous_snapshot.mechanical_grip,
            tyre_management=previous_snapshot.tyre_management,
            downforce_level=previous_snapshot.downforce_level,
            confidence=float(np.clip(previous_snapshot.confidence + 0.08, 0.0, 1.0)),
            notes=f"{previous_snapshot.notes}; retained due to missing race evidence",
        )
        save_car_architecture(retained)
        return retained

    prior_races = max(target_round - 2, 0)
    prev_weight = float(np.clip(1.0 - (0.9 ** prior_races), 0.0, 1.0))
    new_races = max(target_round - 1, 1)

    evidence_means: dict[str, float] = {}
    for dimension in ARCHITECTURE_DIMENSIONS:
        new_value = _clip01(evidence.get(dimension, getattr(previous_snapshot, dimension)))
        if prev_weight <= 1e-9 or prior_races <= 0:
            mean_previous = getattr(previous_snapshot, dimension)
            combined_mean = new_value
        else:
            mean_previous = (
                getattr(previous_snapshot, dimension) - (1.0 - prev_weight) * getattr(season_prior, dimension)
            ) / max(prev_weight, 1e-9)
            mean_previous = _clip01(mean_previous, default=getattr(previous_snapshot, dimension))
            combined_mean = _clip01((mean_previous * prior_races + new_value) / max(new_races, 1))
        evidence_means[dimension] = combined_mean

    updated = _blend_architecture(
        season_prior=season_prior,
        evidence_means=evidence_means,
        target_round=target_round,
        n_races=new_races,
    )
    save_car_architecture(updated)
    LOGGER.info(
        "Updated %s %s architecture | round=%s confidence=%.2f",
        constructor_id,
        season,
        target_round,
        updated.confidence,
    )
    return updated
