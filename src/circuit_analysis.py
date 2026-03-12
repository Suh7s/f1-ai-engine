"""Circuit archetype clustering and driver fit analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.circuit_tsi import (
    detect_circuit_momentum as _detect_circuit_momentum,
    get_circuit_specialists as _get_circuit_specialists,
    get_driver_circuit_profile as _get_driver_circuit_profile,
)
from src.feature_engineering import CIRCUIT_DNA
from src.true_skill import build_tsi_artifacts
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
    "cluster_path": BASE_CONFIG["processed_dir"] / "circuit_clusters.csv",
    "driver_fit_path": BASE_CONFIG["processed_dir"] / "driver_circuit_archetypes.csv",
    "metadata_path": BASE_CONFIG["processed_dir"] / "circuit_cluster_metadata.json",
    "k_range": [3, 4, 5, 6],
    "random_state": 42,
}

LOGGER = get_logger(__name__)

CLUSTER_FEATURES: list[str] = [
    "high_speed_pct",
    "tyre_deg_severity",
    "overtaking_difficulty",
    "sc_probability",
    "altitude_m",
    "lap_length_km",
    "historical_wet_frequency",
    "drs_zones",
    "historical_average_position_changes",
]


@dataclass(slots=True)
class CircuitArtifacts:
    """Persisted circuit analysis artifact bundle."""

    cluster_frame: pd.DataFrame
    driver_fit: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(slots=True)
class CircuitDemands:
    """Circuit demand profile used for architecture fit scoring."""

    straight_line_speed: float
    slow_corner_grip: float
    high_speed_cornering: float
    mechanical_grip: float
    tyre_management: float
    downforce_level: float
    primary_differentiator: str
    notes: str = ""


_EXPLICIT_CIRCUIT_DEMANDS: dict[str, CircuitDemands] = {
    "monza": CircuitDemands(0.95, 0.20, 0.40, 0.30, 0.55, 0.05, "straight_line_speed", "Straight line speed overwhelmingly dominant"),
    "monaco": CircuitDemands(0.15, 0.95, 0.10, 0.85, 0.40, 0.98, "slow_corner_grip", "Maximum downforce, mechanical grip dominant"),
    "silverstone": CircuitDemands(0.55, 0.40, 0.92, 0.70, 0.85, 0.70, "high_speed_cornering", "High speed corners and high tyre deg shape the lap"),
    "spa": CircuitDemands(0.80, 0.45, 0.88, 0.60, 0.65, 0.55, "high_speed_cornering", "Eau Rouge and Pouhon reward high-speed confidence"),
    "hungaroring": CircuitDemands(0.25, 0.90, 0.50, 0.85, 0.80, 0.88, "slow_corner_grip", "Monaco without walls: high downforce and traction"),
    "baku": CircuitDemands(0.88, 0.82, 0.30, 0.88, 0.60, 0.65, "straight_line_speed", "Unique blend of straight line speed and mechanical grip"),
    "zandvoort": CircuitDemands(0.35, 0.70, 0.85, 0.80, 0.88, 0.82, "high_speed_cornering", "Banked corners reward aero efficiency"),
    "marina_bay": CircuitDemands(0.30, 0.92, 0.25, 0.88, 0.55, 0.95, "slow_corner_grip", "Bumpy hot street circuit, maximum downforce"),
    "rodriguez": CircuitDemands(0.85, 0.65, 0.60, 0.70, 0.70, 0.35, "straight_line_speed", "Altitude and long straights dominate"),
}


def _heuristic_circuit_demands(circuit_id: str) -> CircuitDemands:
    metadata = CIRCUIT_DNA.get(
        str(circuit_id),
        {
            "high_speed_pct": 0.45,
            "tyre_deg_severity": 3.0,
            "overtaking_difficulty": 5.0,
            "sc_probability": 0.40,
            "altitude_m": 100.0,
            "street_circuit": False,
        },
    )
    high_speed_pct = float(metadata["high_speed_pct"])
    tyre_deg = float(metadata["tyre_deg_severity"])
    street = float(bool(metadata.get("street_circuit", False)))
    altitude = float(metadata.get("altitude_m", 0.0))

    straight_line_speed = np.clip(0.15 + 1.05 * high_speed_pct, 0.15, 0.92)
    slow_corner_grip = np.clip(0.25 + 0.55 * (1.0 - high_speed_pct) + 0.18 * street, 0.20, 0.92)
    high_speed_cornering = np.clip(0.25 + 0.95 * high_speed_pct, 0.20, 0.92)
    mechanical_grip = np.clip(0.30 + 0.45 * (1.0 - high_speed_pct) + 0.20 * street, 0.25, 0.90)
    tyre_management = np.clip(0.35 + 0.12 * tyre_deg, 0.35, 0.90)
    downforce_level = np.clip(0.18 + 0.72 * (1.0 - high_speed_pct) + 0.10 * street - 0.10 * (altitude > 1000.0), 0.08, 0.92)

    dimension_map = {
        "straight_line_speed": straight_line_speed,
        "slow_corner_grip": slow_corner_grip,
        "high_speed_cornering": high_speed_cornering,
        "mechanical_grip": mechanical_grip,
        "tyre_management": tyre_management,
    }
    primary_differentiator = max(dimension_map, key=dimension_map.get)
    if downforce_level >= 0.85 and primary_differentiator not in {"slow_corner_grip", "mechanical_grip"}:
        primary_differentiator = "slow_corner_grip"
    notes = "Heuristic demand profile derived from circuit DNA"
    return CircuitDemands(
        straight_line_speed=float(straight_line_speed),
        slow_corner_grip=float(slow_corner_grip),
        high_speed_cornering=float(high_speed_cornering),
        mechanical_grip=float(mechanical_grip),
        tyre_management=float(tyre_management),
        downforce_level=float(downforce_level),
        primary_differentiator=str(primary_differentiator),
        notes=notes,
    )


CIRCUIT_ARCHITECTURE_REQUIREMENTS: dict[str, CircuitDemands] = {
    circuit_id: _EXPLICIT_CIRCUIT_DEMANDS.get(circuit_id, _heuristic_circuit_demands(circuit_id))
    for circuit_id in sorted(CIRCUIT_DNA.keys())
}

CIRCUIT_DEMANDS = CIRCUIT_ARCHITECTURE_REQUIREMENTS


def _prepare_circuit_frame(race_results: pd.DataFrame) -> pd.DataFrame:
    """Build the circuit-level feature frame used for clustering."""

    results = race_results.copy()
    results["is_wet_race"] = pd.to_numeric(results.get("is_wet_race"), errors="coerce").fillna(0).astype(int)
    results["is_mixed_race"] = pd.to_numeric(results.get("is_mixed_race"), errors="coerce").fillna(0).astype(int)
    results["grid_position"] = pd.to_numeric(results.get("grid_position"), errors="coerce")
    results["finish_position"] = pd.to_numeric(results.get("finish_position"), errors="coerce")
    results["position_delta_abs"] = (results["grid_position"] - results["finish_position"]).abs()

    historical = (
        results.groupby("circuit_id", as_index=False)
        .agg(
            historical_wet_frequency=(
                "is_wet_race",
                lambda series: float(np.mean(series.to_numpy(dtype=float))),
            ),
            historical_mixed_frequency=(
                "is_mixed_race",
                lambda series: float(np.mean(series.to_numpy(dtype=float))),
            ),
            historical_average_position_changes=(
                "position_delta_abs",
                lambda series: float(pd.to_numeric(series, errors="coerce").mean()),
            ),
            races=("season", "count"),
        )
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for circuit_id, metadata in CIRCUIT_DNA.items():
        historical_row = historical.loc[historical["circuit_id"] == circuit_id].head(1)
        rows.append(
            {
                "circuit_id": circuit_id,
                "high_speed_pct": float(metadata["high_speed_pct"]),
                "tyre_deg_severity": float(metadata["tyre_deg_severity"]),
                "overtaking_difficulty": float(metadata["overtaking_difficulty"]),
                "sc_probability": float(metadata["sc_probability"]),
                "altitude_m": float(metadata["altitude_m"]),
                "lap_length_km": float(metadata["lap_length_km"]),
                "drs_zones": float(metadata["drs_zones"]),
                "historical_wet_frequency": float(historical_row["historical_wet_frequency"].iloc[0]) if not historical_row.empty else 0.0,
                "historical_average_position_changes": float(historical_row["historical_average_position_changes"].iloc[0]) if not historical_row.empty else 0.0,
                "historical_mixed_frequency": float(historical_row["historical_mixed_frequency"].iloc[0]) if not historical_row.empty else 0.0,
                "races": int(historical_row["races"].iloc[0]) if not historical_row.empty else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("circuit_id").reset_index(drop=True)


def _cluster_label(center: pd.Series) -> str:
    """Assign a human-readable label to a circuit cluster center."""

    if center["overtaking_difficulty"] > 7:
        return "Technical Street Circuit"
    if center["high_speed_pct"] > 0.4 and center["tyre_deg_severity"] < 3:
        return "Power Circuit"
    if center["tyre_deg_severity"] > 3.5:
        return "High Degradation Circuit"
    if center["sc_probability"] > 0.6:
        return "High Safety Car Circuit"
    if center["altitude_m"] > 1000:
        return "High Altitude Circuit"
    return "Balanced Circuit"


def _fit_clusters(circuit_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster circuits and choose the best k using silhouette score."""

    features = circuit_frame[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    best_score = float("-inf")
    best_labels: np.ndarray | None = None
    best_k = CONFIG["k_range"][0]
    best_model: KMeans | None = None
    silhouette_by_k: dict[str, float] = {}

    for k in CONFIG["k_range"]:
        if len(circuit_frame) <= int(k):
            continue
        model = KMeans(n_clusters=int(k), random_state=int(CONFIG["random_state"]), n_init="auto")
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled, labels))
        silhouette_by_k[str(k)] = score
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = int(k)
            best_model = model

    if best_labels is None or best_model is None:
        raise ValueError("Unable to fit circuit clusters with the available circuit frame")

    cluster_frame = circuit_frame.copy()
    cluster_frame["cluster_id"] = best_labels.astype(int)
    center_rows: list[dict[str, Any]] = []
    label_map: dict[int, str] = {}
    cluster_members: list[dict[str, Any]] = []
    for cluster_id in sorted(cluster_frame["cluster_id"].unique().tolist()):
        member_frame = cluster_frame.loc[cluster_frame["cluster_id"] == cluster_id].copy()
        center = member_frame[CLUSTER_FEATURES].mean()
        label = _cluster_label(center)
        if label in label_map.values():
            label = f"{label} {cluster_id + 1}"
        label_map[int(cluster_id)] = label
        center_rows.append({"cluster_id": int(cluster_id), "cluster_label": label, **center.to_dict()})
        circuits = sorted(member_frame["circuit_id"].astype(str).tolist())
        cluster_members.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_label": label,
                "centroid": {feature: float(center[feature]) for feature in CLUSTER_FEATURES},
                "circuits": circuits,
            }
        )
        LOGGER.info("Circuit cluster %s | label=%s | centroid=%s | circuits=%s", cluster_id, label, center.to_dict(), circuits)
    cluster_frame["cluster_label"] = cluster_frame["cluster_id"].map(label_map)
    cluster_lookup = dict(zip(cluster_frame["circuit_id"].astype(str), cluster_frame["cluster_id"].astype(int)))
    monza_cluster = cluster_lookup.get("monza")
    monaco_cluster = cluster_lookup.get("monaco")
    if monza_cluster is None or monaco_cluster is None:
        raise ValueError("Monza and Monaco must be present in circuit clustering output")
    if monza_cluster == monaco_cluster:
        raise ValueError("Monza and Monaco must be in different clusters — check features")
    metadata = {
        "selected_k": int(best_k),
        "silhouette_score": float(best_score),
        "silhouette_by_k": silhouette_by_k,
        "cluster_centers": center_rows,
        "clusters": cluster_members,
        "monza_cluster": int(monza_cluster),
        "monaco_cluster": int(monaco_cluster),
    }
    return cluster_frame, metadata


def _driver_archetype_fit(
    race_results: pd.DataFrame,
    cluster_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compute driver performance deltas by circuit archetype using TSI race deltas."""

    tsi_driver_race = build_tsi_artifacts(race_results=race_results).driver_race
    if tsi_driver_race.empty:
        return pd.DataFrame()

    circuit_map = cluster_frame[["circuit_id", "cluster_id", "cluster_label"]].copy()
    merged = tsi_driver_race.merge(circuit_map, how="left", on="circuit_id")
    driver_overall = merged.groupby(["driver_id", "driver_name"], as_index=False).agg(
        overall_race_tsi=("tsi_race_event", "mean"),
        overall_composite_tsi=("tsi_composite_event", "mean"),
        sample_size=("race_valid", "sum"),
    )
    archetype = (
        merged.groupby(["driver_id", "driver_name", "cluster_id", "cluster_label"], as_index=False)
        .agg(
            archetype_race_tsi=("tsi_race_event", "mean"),
            archetype_composite_tsi=("tsi_composite_event", "mean"),
            archetype_races=("round", "count"),
        )
        .merge(driver_overall, how="left", on=["driver_id", "driver_name"])
    )
    archetype["race_tsi_delta_vs_career"] = archetype["archetype_race_tsi"] - archetype["overall_race_tsi"]
    archetype["composite_tsi_delta_vs_career"] = (
        archetype["archetype_composite_tsi"] - archetype["overall_composite_tsi"]
    )
    return archetype.sort_values(
        ["cluster_label", "composite_tsi_delta_vs_career", "archetype_races"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _fit_summary(driver_fit: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Return the strongest circuit-archetype fits and misfits."""

    if driver_fit.empty:
        return {"fits": [], "misfits": []}

    eligible = driver_fit.loc[
        (driver_fit["archetype_races"] >= 8) & (driver_fit["sample_size"] >= 20)
    ].copy()
    if eligible.empty:
        eligible = driver_fit.loc[driver_fit["archetype_races"] >= 5].copy()
    if eligible.empty:
        eligible = driver_fit.copy()

    fits = (
        eligible.sort_values(["composite_tsi_delta_vs_career", "archetype_races"], ascending=[True, False])
        .head(3)
        .loc[:, ["driver_id", "driver_name", "cluster_label", "composite_tsi_delta_vs_career", "archetype_races"]]
        .to_dict(orient="records")
    )
    misfits = (
        eligible.sort_values(["composite_tsi_delta_vs_career", "archetype_races"], ascending=[False, False])
        .head(3)
        .loc[:, ["driver_id", "driver_name", "cluster_label", "composite_tsi_delta_vs_career", "archetype_races"]]
        .to_dict(orient="records")
    )
    return {"fits": fits, "misfits": misfits}


def build_circuit_artifacts(race_results: pd.DataFrame | None = None) -> CircuitArtifacts:
    """Build circuit archetype artifacts."""

    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    circuit_frame = _prepare_circuit_frame(race_results)
    cluster_frame, metadata = _fit_clusters(circuit_frame)
    driver_fit = _driver_archetype_fit(race_results, cluster_frame)
    metadata["driver_fit_summary"] = _fit_summary(driver_fit)
    return CircuitArtifacts(cluster_frame=cluster_frame, driver_fit=driver_fit, metadata=metadata)


def run_circuit_analysis_pipeline(race_results: pd.DataFrame | None = None) -> CircuitArtifacts:
    """Build and persist circuit clustering artifacts."""

    ensure_directories()
    artifacts = build_circuit_artifacts(race_results=race_results)
    save_dataframe(artifacts.cluster_frame, CONFIG["cluster_path"])
    save_dataframe(artifacts.driver_fit, CONFIG["driver_fit_path"])
    save_json(artifacts.metadata, CONFIG["metadata_path"])
    LOGGER.info(
        "Saved circuit analysis artifacts | circuits=%s driver_fit=%s",
        len(artifacts.cluster_frame),
        len(artifacts.driver_fit),
    )
    return artifacts


def load_circuit_clusters() -> pd.DataFrame:
    """Load persisted circuit cluster assignments."""

    path = Path(CONFIG["cluster_path"])
    if not path.exists():
        return run_circuit_analysis_pipeline().cluster_frame
    return load_dataframe(path)


def load_driver_archetype_fit() -> pd.DataFrame:
    """Load persisted driver archetype performance data."""

    path = Path(CONFIG["driver_fit_path"])
    if not path.exists():
        return run_circuit_analysis_pipeline().driver_fit
    return load_dataframe(path)


def get_circuit_specialists(
    circuit_id: str,
    season: int | None = None,
    round_number: int | None = None,
    top_n: int = 8,
) -> pd.DataFrame:
    """Return top specialist drivers at a circuit, car-adjusted."""
    return _get_circuit_specialists(circuit_id, season=season, round_number=round_number, top_n=top_n)


def get_driver_circuit_profile(
    driver_id: str,
    season: int | None = None,
    round_number: int | None = None,
) -> dict[str, Any]:
    """Full breakdown of where a driver over/underperforms."""
    return _get_driver_circuit_profile(driver_id, season=season, round_number=round_number)


def detect_circuit_momentum(
    driver_id: str,
    circuit_id: str,
    n_recent: int = 3,
) -> dict[str, Any]:
    """Is this driver improving or declining at this circuit recently?"""
    return _detect_circuit_momentum(driver_id, circuit_id, n_recent=n_recent)
