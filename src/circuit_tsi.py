"""Circuit-specific True Skill Index — car-adjusted driver-circuit performance."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.utils import (
    CONFIG as BASE_CONFIG,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_dataframe,
    save_json,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "cpi_history_path": BASE_CONFIG["processed_dir"] / "cpi_history.csv",
    "circuit_tsi_path": BASE_CONFIG["processed_dir"] / "circuit_tsi_matrix.csv",
    "circuit_tsi_metadata_path": BASE_CONFIG["processed_dir"] / "circuit_tsi_metadata.json",
    "cluster_path": BASE_CONFIG["processed_dir"] / "circuit_clusters.csv",
    "min_races_high": 5,
    "min_races_medium": 2,
}

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CircuitTSIArtifacts:
    """Persisted circuit TSI artifact bundle."""

    circuit_tsi_matrix: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CircuitTSIResult:
    """Compatibility wrapper for circuit TSI lookups."""

    score: float | None
    raw: float | None
    std: float | None
    sample_size: int
    confidence: str


def _compute_expected_positions(
    race_results: pd.DataFrame,
    cpi_history: pd.DataFrame,
) -> pd.DataFrame:
    """For each race, compute the CPI-based expected finishing position per driver.

    Approach: rank constructors by their CPI_performance_before for the race.
    Within each constructor, use qualifying position to break ties between teammates.
    The expected position = the CPI-rank position (1-indexed).
    """
    results = race_results.copy()
    results["season"] = results["season"].astype(int)
    results["round"] = results["round"].astype(int)
    results["qualifying_position"] = pd.to_numeric(results["qualifying_position"], errors="coerce")
    results["finish_position"] = pd.to_numeric(results["finish_position"], errors="coerce")

    if cpi_history.empty:
        results["expected_position_from_cpi"] = results["qualifying_position"]
        dnf_reason = results["dnf_reason"].fillna("finished").astype(str).str.lower()
        mechanical_mask = dnf_reason.eq("mechanical")
        fill_position = results["finish_position"].fillna(
            results.groupby(["season", "round"])["driver_id"].transform("count")
        )
        results["circuit_delta"] = np.where(
            mechanical_mask, np.nan,
            fill_position - results["expected_position_from_cpi"],
        )
        return results

    # Build a CPI lookup: (constructor_id, season, round) -> cpi_performance_before
    cpi = cpi_history.copy()
    cpi["season"] = cpi["season"].astype(int)
    cpi["round"] = cpi["round"].astype(int)
    cpi_col = "cpi_performance_before" if "cpi_performance_before" in cpi.columns else "cpi_performance_after"
    cpi_lookup = cpi.set_index(["constructor_id", "season", "round"])[cpi_col].to_dict()

    expected_positions = []
    for (season, rnd), race_group in results.groupby(["season", "round"], sort=False):
        race = race_group.copy()
        # Look up CPI for each driver's constructor
        race["cpi_val"] = race["constructor_id"].map(
            lambda cid, s=season, r=rnd: cpi_lookup.get((cid, int(s), int(r)), 0.0)
        )
        # Higher cpi_performance = better car.
        # Sort descending so rank 1 = best car = expected P1.
        race = race.sort_values(
            ["cpi_val", "qualifying_position"],
            ascending=[False, True],
        ).reset_index(drop=True)
        race["expected_position_from_cpi"] = np.arange(1, len(race) + 1, dtype=float)
        expected_positions.append(race[["season", "round", "driver_id", "expected_position_from_cpi", "cpi_val"]])

    if not expected_positions:
        results["expected_position_from_cpi"] = results["qualifying_position"]
        results["circuit_delta"] = 0.0
        return results

    expected_df = pd.concat(expected_positions, ignore_index=True)
    results = results.merge(
        expected_df[["season", "round", "driver_id", "expected_position_from_cpi"]],
        on=["season", "round", "driver_id"],
        how="left",
    )
    results["expected_position_from_cpi"] = results["expected_position_from_cpi"].fillna(
        results["qualifying_position"]
    )

    # Circuit delta: actual - expected. Negative = outperformed car prediction (good).
    # Exclude mechanical DNFs (inflated finish positions don't reflect driver skill).
    dnf_reason = results["dnf_reason"].fillna("finished").astype(str).str.lower()
    mechanical_mask = dnf_reason.eq("mechanical")
    fill_position = results["finish_position"].fillna(results.groupby(["season", "round"])["driver_id"].transform("count"))
    results["circuit_delta"] = np.where(
        mechanical_mask,
        np.nan,
        fill_position - results["expected_position_from_cpi"],
    )
    return results


def _build_circuit_tsi_matrix(
    results_with_deltas: pd.DataFrame,
) -> pd.DataFrame:
    """Build rolling circuit TSI scores per driver × circuit pair.

    Key insight: we compare each driver's circuit-specific delta against their
    OWN career-average delta across ALL circuits. This isolates true circuit
    affinity — "does this driver specifically outperform HERE vs their usual
    level?" — rather than just measuring absolute outperformance (which all
    top drivers share equally, causing saturation).

    For each driver-race, compute the circuit_tsi using only data from
    prior appearances (strict lag — shift(1) equivalent).
    """
    df = results_with_deltas.copy()
    df = df.sort_values(["driver_id", "season", "round"]).reset_index(drop=True)

    min_high = int(CONFIG["min_races_high"])
    min_medium = int(CONFIG["min_races_medium"])

    # Step 1: Compute each driver's expanding career-average delta across ALL circuits.
    # This gives us the "baseline" that we subtract per-circuit to isolate specialization.
    driver_career_avg: dict[str, list[tuple[int, int, float]]] = {}
    for driver_id, driver_group in df.groupby("driver_id", sort=False):
        driver_sorted = driver_group.sort_values(["season", "round"]).reset_index(drop=True)
        valid_deltas = driver_sorted["circuit_delta"].copy()
        # Expanding mean of ALL prior deltas (lagged by 1 race)
        career_expanding = valid_deltas.shift(1).expanding(min_periods=1).mean()
        entries = []
        for idx in range(len(driver_sorted)):
            row = driver_sorted.iloc[idx]
            entries.append((int(row["season"]), int(row["round"]), float(career_expanding.iloc[idx]) if pd.notna(career_expanding.iloc[idx]) else np.nan))
        driver_career_avg[str(driver_id)] = entries

    # Build lookup: (driver_id, season, round) -> career_avg_delta_before
    career_avg_lookup: dict[tuple[str, int, int], float] = {}
    for driver_id, entries in driver_career_avg.items():
        for season, rnd, avg in entries:
            career_avg_lookup[(driver_id, season, rnd)] = avg

    # Also compute career avg including current race (for _after columns)
    driver_career_avg_after: dict[tuple[str, int, int], float] = {}
    for driver_id, driver_group in df.groupby("driver_id", sort=False):
        driver_sorted = driver_group.sort_values(["season", "round"]).reset_index(drop=True)
        valid_deltas = driver_sorted["circuit_delta"].copy()
        career_expanding_after = valid_deltas.expanding(min_periods=1).mean()
        for idx in range(len(driver_sorted)):
            row = driver_sorted.iloc[idx]
            val = float(career_expanding_after.iloc[idx]) if pd.notna(career_expanding_after.iloc[idx]) else np.nan
            driver_career_avg_after[(str(driver_id), int(row["season"]), int(row["round"]))] = val

    # Step 2: Build per-circuit metrics, now subtracting the career average
    df = df.sort_values(["driver_id", "circuit_id", "season", "round"]).reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for (driver_id, circuit_id), group in df.groupby(["driver_id", "circuit_id"], sort=False):
        group = group.sort_values(["season", "round"]).reset_index(drop=True)
        deltas = group["circuit_delta"]

        # Raw circuit-specific expanding stats
        expanding_mean_before = deltas.shift(1).expanding(min_periods=1).mean()
        expanding_std_before = deltas.shift(1).expanding(min_periods=2).std(ddof=0)
        sample_before = deltas.shift(1).expanding().count()

        expanding_mean_after = deltas.expanding(min_periods=1).mean()
        expanding_std_after = deltas.expanding(min_periods=2).std(ddof=0)
        sample_after = deltas.expanding().count()

        # Look up career average for each row and compute relative delta
        career_before = group.apply(
            lambda r: career_avg_lookup.get((str(driver_id), int(r["season"]), int(r["round"])), np.nan),
            axis=1,
        )
        career_after = group.apply(
            lambda r: driver_career_avg_after.get((str(driver_id), int(r["season"]), int(r["round"])), np.nan),
            axis=1,
        )

        # Relative circuit delta = circuit_mean - career_mean
        # Negative = driver outperforms HERE more than their average → circuit specialist
        relative_before = expanding_mean_before - career_before
        relative_after = expanding_mean_after - career_after

        # Normalize with tanh — now the input range is much tighter (typically -3 to +3)
        # so scale=3.0 gives good spread without saturation
        group["circuit_tsi_score_before"] = np.tanh(relative_before / 3.0)
        group["circuit_tsi_score_after"] = np.tanh(relative_after / 3.0)
        group["circuit_tsi_raw_before"] = relative_before
        group["circuit_tsi_raw_after"] = relative_after
        group["circuit_tsi_std_before"] = expanding_std_before.fillna(0.0)
        group["circuit_tsi_std_after"] = expanding_std_after.fillna(0.0)
        group["circuit_tsi_sample_size_before"] = sample_before.fillna(0).astype(int)
        group["circuit_tsi_sample_size_after"] = sample_after.fillna(0).astype(int)

        group["circuit_tsi_confidence_before"] = np.where(
            group["circuit_tsi_sample_size_before"] >= min_high, 1.0,
            np.where(group["circuit_tsi_sample_size_before"] >= min_medium, 0.5, 0.0),
        )
        group["circuit_tsi_confidence_after"] = np.where(
            group["circuit_tsi_sample_size_after"] >= min_high, 1.0,
            np.where(group["circuit_tsi_sample_size_after"] >= min_medium, 0.5, 0.0),
        )
        frames.append(group)

    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    return output


def _validate_known_specialists(matrix: pd.DataFrame) -> dict[str, Any]:
    """Check known circuit specialists against computed circuit TSI."""
    validations: list[dict[str, Any]] = []

    known_specialists = [
        ("hamilton", "silverstone", "strongly negative"),
        ("alonso", "monaco", "strongly negative"),
        ("alonso", "hungaroring", "strongly negative"),
        ("perez", "baku", "strongly negative"),
        ("perez", "monaco", "strongly negative"),
        ("sainz", "monaco", "strongly negative"),
        ("max_verstappen", "spa", "negative"),
        ("max_verstappen", "zandvoort", "strongly negative"),
        ("hamilton", "monaco", "negative"),
    ]

    for driver_pattern, circuit_id, expected_direction in known_specialists:
        rows = matrix.loc[
            matrix["driver_id"].str.contains(driver_pattern, case=False, na=False)
            & (matrix["circuit_id"] == circuit_id)
        ]
        if rows.empty:
            validations.append({
                "driver": driver_pattern,
                "circuit": circuit_id,
                "expected": expected_direction,
                "actual_score": None,
                "status": "NO_DATA",
            })
            continue

        latest = rows.sort_values(["season", "round"]).iloc[-1]
        score = float(latest["circuit_tsi_score_after"]) if pd.notna(latest["circuit_tsi_score_after"]) else None
        raw = float(latest["circuit_tsi_raw_after"]) if pd.notna(latest["circuit_tsi_raw_after"]) else None

        if score is None:
            status = "NO_SCORE"
        elif "strongly" in expected_direction:
            status = "PASS" if score < -0.05 else "WARN"
        else:
            status = "PASS" if score < 0 else "WARN"

        validations.append({
            "driver": driver_pattern,
            "circuit": circuit_id,
            "expected": expected_direction,
            "actual_score": score,
            "actual_raw": raw,
            "sample_size": int(latest["circuit_tsi_sample_size_after"]),
            "status": status,
        })

    return {"specialist_validations": validations}


def build_circuit_tsi_artifacts(
    race_results: pd.DataFrame | None = None,
    cpi_history: pd.DataFrame | None = None,
) -> CircuitTSIArtifacts:
    """Build circuit TSI artifacts from race results and CPI history."""
    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    if cpi_history is None:
        path = CONFIG["cpi_history_path"]
        cpi_history = load_dataframe(path) if path.exists() else pd.DataFrame()

    results_with_deltas = _compute_expected_positions(race_results, cpi_history)
    circuit_tsi_matrix = _build_circuit_tsi_matrix(results_with_deltas)

    metadata: dict[str, Any] = {
        "total_driver_circuit_pairs": int(
            circuit_tsi_matrix.groupby(["driver_id", "circuit_id"]).ngroups
        ) if not circuit_tsi_matrix.empty else 0,
        "total_rows": len(circuit_tsi_matrix),
        "circuits_covered": sorted(circuit_tsi_matrix["circuit_id"].unique().tolist()) if not circuit_tsi_matrix.empty else [],
    }

    if not circuit_tsi_matrix.empty:
        validation = _validate_known_specialists(circuit_tsi_matrix)
        metadata.update(validation)
        for v in validation.get("specialist_validations", []):
            level = LOGGER.info if v["status"] == "PASS" else LOGGER.warning
            level(
                "Circuit TSI validation: %s at %s | expected=%s score=%.3f status=%s",
                v["driver"], v["circuit"], v["expected"],
                v.get("actual_score", 0.0) or 0.0, v["status"],
            )

    return CircuitTSIArtifacts(circuit_tsi_matrix=circuit_tsi_matrix, metadata=metadata)


def run_circuit_tsi_pipeline(
    race_results: pd.DataFrame | None = None,
    cpi_history: pd.DataFrame | None = None,
) -> CircuitTSIArtifacts:
    """Build and persist circuit TSI artifacts."""
    ensure_directories()
    artifacts = build_circuit_tsi_artifacts(race_results=race_results, cpi_history=cpi_history)
    if not artifacts.circuit_tsi_matrix.empty:
        save_dataframe(artifacts.circuit_tsi_matrix, CONFIG["circuit_tsi_path"])
    save_json(artifacts.metadata, CONFIG["circuit_tsi_metadata_path"])
    LOGGER.info(
        "Saved circuit TSI artifacts | pairs=%s rows=%s",
        artifacts.metadata.get("total_driver_circuit_pairs", 0),
        len(artifacts.circuit_tsi_matrix),
    )
    _load_circuit_tsi_matrix.cache_clear()
    return artifacts


@lru_cache(maxsize=1)
def _load_circuit_tsi_matrix(
    race_results: pd.DataFrame | None = None,
    cpi_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load or build the circuit TSI matrix."""
    path = CONFIG["circuit_tsi_path"]
    if path.exists():
        return load_dataframe(path)
    return build_circuit_tsi_artifacts(
        race_results=race_results, cpi_history=cpi_history,
    ).circuit_tsi_matrix


def _resolve_driver_id_for_matrix(driver_id: str, matrix: pd.DataFrame) -> str:
    """Resolve common driver-id aliases against the persisted circuit TSI matrix."""

    candidate = str(driver_id).strip().lower()
    if matrix.empty:
        return candidate

    known = {str(value).lower() for value in matrix["driver_id"].dropna().unique().tolist()}
    if candidate in known:
        return candidate

    tail = candidate.split("_")[-1]
    if tail in known:
        return tail

    partial_matches = sorted(identifier for identifier in known if candidate in identifier or identifier in candidate)
    return partial_matches[0] if partial_matches else candidate


class TrueSkill:
    """Compatibility facade exposing circuit TSI lookups under the legacy name."""

    def __init__(self) -> None:
        self._matrix = pd.DataFrame()

    def load(self) -> None:
        self._matrix = _load_circuit_tsi_matrix()

    def compute_circuit_tsi(self, driver_id: str, circuit_id: str) -> CircuitTSIResult:
        matrix = self._matrix if not self._matrix.empty else _load_circuit_tsi_matrix()
        if matrix.empty:
            return CircuitTSIResult(score=None, raw=None, std=None, sample_size=0, confidence="none")

        resolved_driver = _resolve_driver_id_for_matrix(driver_id, matrix)
        subset = matrix.loc[
            (matrix["driver_id"].astype(str).str.lower() == resolved_driver)
            & (matrix["circuit_id"].astype(str) == str(circuit_id))
        ].sort_values(["season", "round"])
        if subset.empty:
            return CircuitTSIResult(score=None, raw=None, std=None, sample_size=0, confidence="none")

        latest = subset.iloc[-1]
        sample_size = int(latest.get("circuit_tsi_sample_size_after", 0) or 0)
        if sample_size >= CONFIG["min_races_high"]:
            confidence = "high"
        elif sample_size >= CONFIG["min_races_medium"]:
            confidence = "medium"
        else:
            confidence = "low"
        return CircuitTSIResult(
            score=float(latest["circuit_tsi_score_after"]) if pd.notna(latest["circuit_tsi_score_after"]) else None,
            raw=float(latest["circuit_tsi_raw_after"]) if pd.notna(latest["circuit_tsi_raw_after"]) else None,
            std=float(latest["circuit_tsi_std_after"]) if pd.notna(latest["circuit_tsi_std_after"]) else None,
            sample_size=sample_size,
            confidence=confidence,
        )


def get_circuit_tsi(
    driver_id: str,
    circuit_id: str,
    season: int,
    round_number: int,
) -> dict[str, Any]:
    """Return the circuit TSI snapshot for a driver at a circuit before a race."""
    matrix = _load_circuit_tsi_matrix()
    if matrix.empty:
        return _empty_circuit_tsi()

    subset = matrix.loc[
        (matrix["driver_id"] == str(driver_id))
        & (matrix["circuit_id"] == str(circuit_id))
        & (
            (matrix["season"] < int(season))
            | ((matrix["season"] == int(season)) & (matrix["round"] < int(round_number)))
        )
    ].sort_values(["season", "round"])

    if subset.empty:
        return _empty_circuit_tsi()

    row = subset.iloc[-1]
    return {
        "circuit_tsi_score": float(row["circuit_tsi_score_after"]) if pd.notna(row["circuit_tsi_score_after"]) else 0.0,
        "circuit_tsi_raw": float(row["circuit_tsi_raw_after"]) if pd.notna(row["circuit_tsi_raw_after"]) else 0.0,
        "circuit_tsi_std": float(row["circuit_tsi_std_after"]) if pd.notna(row["circuit_tsi_std_after"]) else 0.0,
        "circuit_tsi_sample_size": int(row["circuit_tsi_sample_size_after"]),
        "circuit_tsi_confidence": float(row["circuit_tsi_confidence_after"]),
    }


def _empty_circuit_tsi() -> dict[str, Any]:
    return {
        "circuit_tsi_score": 0.0,
        "circuit_tsi_raw": 0.0,
        "circuit_tsi_std": 0.0,
        "circuit_tsi_sample_size": 0,
        "circuit_tsi_confidence": 0.0,
    }


def get_archetype_tsi(
    driver_id: str,
    circuit_id: str,
    season: int,
    round_number: int,
) -> float:
    """Return the driver's average circuit TSI across all circuits in the same archetype cluster.

    Falls back to 0.0 if cluster data or circuit TSI is unavailable.
    """
    cluster_path = CONFIG["cluster_path"]
    if not cluster_path.exists():
        return 0.0

    clusters = load_dataframe(cluster_path)
    if clusters.empty or "cluster_id" not in clusters.columns:
        return 0.0

    target_cluster = clusters.loc[clusters["circuit_id"] == str(circuit_id), "cluster_id"]
    if target_cluster.empty:
        return 0.0
    cluster_id = int(target_cluster.iloc[0])

    # Get all circuits in the same cluster
    cluster_circuits = set(clusters.loc[clusters["cluster_id"] == cluster_id, "circuit_id"].tolist())

    matrix = _load_circuit_tsi_matrix()
    if matrix.empty:
        return 0.0

    # Get driver's circuit TSI across all cluster circuits, using only prior data
    subset = matrix.loc[
        (matrix["driver_id"] == str(driver_id))
        & (matrix["circuit_id"].isin(cluster_circuits))
        & (
            (matrix["season"] < int(season))
            | ((matrix["season"] == int(season)) & (matrix["round"] < int(round_number)))
        )
    ]
    if subset.empty:
        return 0.0

    # Use latest score per circuit
    latest_per_circuit = (
        subset.sort_values(["season", "round"])
        .groupby("circuit_id", as_index=False)
        .tail(1)
    )
    scores = latest_per_circuit["circuit_tsi_score_after"].dropna()
    return float(scores.mean()) if not scores.empty else 0.0


def get_circuit_specialists(
    circuit_id: str,
    season: int | None = None,
    round_number: int | None = None,
    top_n: int = 8,
) -> pd.DataFrame:
    """Return top specialist drivers at a circuit, sorted by circuit TSI (best first)."""
    matrix = _load_circuit_tsi_matrix()
    if matrix.empty:
        return pd.DataFrame()

    if season is not None and round_number is not None:
        subset = matrix.loc[
            (matrix["circuit_id"] == str(circuit_id))
            & (
                (matrix["season"] < int(season))
                | ((matrix["season"] == int(season)) & (matrix["round"] <= int(round_number)))
            )
        ]
    else:
        subset = matrix.loc[matrix["circuit_id"] == str(circuit_id)]

    if subset.empty:
        return pd.DataFrame()

    latest = (
        subset.sort_values(["season", "round"])
        .groupby("driver_id", as_index=False)
        .tail(1)
    )
    latest = latest.loc[latest["circuit_tsi_sample_size_after"] >= 2].copy()
    latest = latest.sort_values("circuit_tsi_score_after", ascending=True).head(int(top_n))

    result_cols = [
        "driver_id", "circuit_id",
        "circuit_tsi_score_after", "circuit_tsi_raw_after",
        "circuit_tsi_std_after", "circuit_tsi_sample_size_after",
        "circuit_tsi_confidence_after",
    ]
    available = [c for c in result_cols if c in latest.columns]
    return latest[available].reset_index(drop=True)


def get_driver_circuit_profile(
    driver_id: str,
    season: int | None = None,
    round_number: int | None = None,
) -> dict[str, Any]:
    """Return a driver's best/worst circuits and archetype affinities."""
    matrix = _load_circuit_tsi_matrix()
    if matrix.empty:
        return {"best_circuits": [], "worst_circuits": [], "specialist_rating": 0.0}

    if season is not None and round_number is not None:
        subset = matrix.loc[
            (matrix["driver_id"] == str(driver_id))
            & (
                (matrix["season"] < int(season))
                | ((matrix["season"] == int(season)) & (matrix["round"] <= int(round_number)))
            )
        ]
    else:
        subset = matrix.loc[matrix["driver_id"] == str(driver_id)]

    if subset.empty:
        return {"best_circuits": [], "worst_circuits": [], "specialist_rating": 0.0}

    latest = (
        subset.sort_values(["season", "round"])
        .groupby("circuit_id", as_index=False)
        .tail(1)
    )
    eligible = latest.loc[latest["circuit_tsi_sample_size_after"] >= 2].copy()
    if eligible.empty:
        return {"best_circuits": [], "worst_circuits": [], "specialist_rating": 0.0}

    sorted_asc = eligible.sort_values("circuit_tsi_score_after", ascending=True)
    best = sorted_asc.head(5)[["circuit_id", "circuit_tsi_score_after", "circuit_tsi_sample_size_after"]].to_dict("records")
    worst = sorted_asc.tail(5).sort_values("circuit_tsi_score_after", ascending=False)[
        ["circuit_id", "circuit_tsi_score_after", "circuit_tsi_sample_size_after"]
    ].to_dict("records")

    # Specialist rating: how variable is the driver across circuits?
    # High std = specialist (very circuit-dependent), low std = consistent everywhere
    scores = eligible["circuit_tsi_score_after"].dropna()
    specialist_rating = float(scores.std()) if len(scores) >= 3 else 0.0

    return {
        "best_circuits": best,
        "worst_circuits": worst,
        "specialist_rating": specialist_rating,
    }


def detect_circuit_momentum(
    driver_id: str,
    circuit_id: str,
    n_recent: int = 3,
) -> dict[str, Any]:
    """Detect whether a driver's recent form at a circuit is improving or declining."""
    matrix = _load_circuit_tsi_matrix()
    if matrix.empty:
        return {"trend": "neutral", "recent_avg": 0.0, "career_avg": 0.0, "delta": 0.0}

    subset = matrix.loc[
        (matrix["driver_id"] == str(driver_id))
        & (matrix["circuit_id"] == str(circuit_id))
        & matrix["circuit_delta"].notna()
    ].sort_values(["season", "round"])

    if len(subset) < 2:
        return {"trend": "insufficient_data", "recent_avg": 0.0, "career_avg": 0.0, "delta": 0.0}

    career_avg = float(subset["circuit_delta"].mean())
    recent = subset.tail(int(n_recent))
    recent_avg = float(recent["circuit_delta"].mean())
    delta = recent_avg - career_avg  # negative = improving (finishing better than career average)

    if delta < -1.0:
        trend = "improving"
    elif delta > 1.0:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "recent_avg": recent_avg,
        "career_avg": career_avg,
        "delta": delta,
        "recent_races": len(recent),
        "total_races": len(subset),
    }
