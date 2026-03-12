"""Monte Carlo simulation of a single Formula 1 race."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.car_architecture import compute_car_circuit_fit
from src.circuit_tsi import get_archetype_tsi, get_circuit_tsi
from src.elo_rating import get_driver_dry_elo
from src.feature_engineering import build_race_inference_frame
from src.model_training import predict_race
from src.regulation_handler import RegulationHandler
from src.registry import get_season_roster
from src.tire_model import compute_strategy_fit
from src.utils import (
    CONFIG as BASE_CONFIG,
    get_logger,
    load_dataframe,
    normalize_probabilities,
    points_for_position,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "default_iterations": 10_000,
    "random_seed": 42,
    "max_dnf_probability": 0.65,
    "min_dnf_probability": 0.01,
    "default_lap_ceiling": 58,
    "minimum_win_probability_floor": 0.005,
    "base_order_noise": 0.18,
    "uncertainty_noise_scale": 0.22,
}

LOGGER = get_logger(__name__)
REGULATION_HANDLER = RegulationHandler()


@dataclass(slots=True)
class RaceConfig:
    """Configuration for a single-race simulation."""

    drivers: list[str]
    qualifying_positions: list[int] | dict[str, int]
    circuit_id: str
    season: int
    round: int
    weather_mode: str | None = None
    n_simulations: int = CONFIG["default_iterations"]
    random_seed: int = CONFIG["random_seed"]
    apply_circuit_advantage: bool = True


@dataclass(slots=True)
class RaceSimulationResult:
    """Aggregated outputs from a race simulation."""

    win_probs: pd.DataFrame
    podium_probs: pd.DataFrame
    points_probs: pd.DataFrame
    expected_points: pd.DataFrame
    position_distributions: pd.DataFrame
    full_order_distribution: pd.DataFrame
    simulation_runs: int
    regulation_warning: bool
    season_uncertainty: float
    prediction_health: PredictionHealth | None = None
    circuit_advantages: pd.DataFrame | None = None


@dataclass(slots=True)
class PredictionHealth:
    """Health report for an input win-probability distribution."""

    status: str
    warning: str | None
    entropy: float
    top_driver_probability: float
    floor_applied: bool = False


@dataclass(slots=True)
class CircuitAdvantage:
    """Unified circuit-advantage signal for a driver-car package."""

    combined_score: float
    driver_affinity: float
    car_fit: float
    strategy_fit: float
    primary_factor: str
    human_readable: str


def _block_mean_interval(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate p10/p90 uncertainty bands from simulation blocks."""

    if samples.ndim != 2:
        raise ValueError("Expected a 2D simulation sample matrix")

    n_simulations = samples.shape[0]
    if n_simulations == 0:
        zero = np.zeros(samples.shape[1], dtype=float)
        return zero, zero

    target_blocks = min(200, max(25, n_simulations // 50))
    block_size = max(1, n_simulations // target_blocks)
    usable_rows = max(block_size, (n_simulations // block_size) * block_size)
    trimmed = samples[:usable_rows]
    block_means = trimmed.reshape(-1, block_size, samples.shape[1]).mean(axis=1)
    return (
        np.quantile(block_means, 0.10, axis=0),
        np.quantile(block_means, 0.90, axis=0),
    )


def _normalize_rows(values: np.ndarray, minimum: float = 1e-12) -> np.ndarray:
    """Normalize a 2D array row-wise."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D array for row normalization")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, minimum, None)
    row_sums = matrix.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) <= 0
    if np.any(zero_rows):
        matrix[zero_rows] = 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums


def check_prediction_health(win_probs: np.ndarray, driver_ids: list[str]) -> PredictionHealth:
    """Detect collapsed or diffuse probability distributions."""

    normalized = normalize_probabilities(np.asarray(win_probs, dtype=float))
    top_driver_prob = float(normalized.max()) if normalized.size else 0.0
    entropy = float(-np.sum(normalized * np.log(np.clip(normalized, 1e-10, None)))) if normalized.size else 0.0
    max_entropy = float(np.log(len(normalized))) if normalized.size else 1.0
    normalized_entropy = entropy / max(max_entropy, 1e-10)

    if top_driver_prob > 0.85:
        return PredictionHealth(
            status="COLLAPSED",
            warning=(
                f"Distribution collapsed: top driver has {top_driver_prob:.0%}. "
                "Missing qualifying/practice data likely cause. Treat as very low confidence."
            ),
            entropy=normalized_entropy,
            top_driver_probability=top_driver_prob,
        )
    if top_driver_prob > 0.60:
        top_driver = driver_ids[int(normalized.argmax())] if driver_ids else "unknown"
        return PredictionHealth(
            status="CONCENTRATED",
            warning=f"High concentration on {top_driver}. Confident prediction but verify data completeness.",
            entropy=normalized_entropy,
            top_driver_probability=top_driver_prob,
        )
    if normalized_entropy > 0.85:
        return PredictionHealth(
            status="DIFFUSE",
            warning="Very spread distribution - highly uncertain race.",
            entropy=normalized_entropy,
            top_driver_probability=top_driver_prob,
        )
    return PredictionHealth(
        status="HEALTHY",
        warning=None,
        entropy=normalized_entropy,
        top_driver_probability=top_driver_prob,
    )


def _apply_probability_floor(win_probs: np.ndarray, minimum_probability: float) -> np.ndarray:
    """Apply a minimum floor to each driver's win probability and renormalize."""

    floored = np.maximum(np.asarray(win_probs, dtype=float), float(minimum_probability))
    return normalize_probabilities(floored)


def _normalize_tsi_to_advantage(score: float, confidence: float, archetype_score: float = 0.0) -> float:
    """Convert negative-is-good circuit TSI into a bounded advantage score."""

    direct = -float(score) * max(float(confidence), 0.20 if abs(float(score)) > 1e-9 else 0.0)
    archetype = -float(archetype_score) * 0.45
    return float(np.clip(np.tanh((direct + archetype) * 2.0), -1.0, 1.0))


def generate_advantage_narrative(
    driver_id: str,
    constructor_id: str,
    circuit_id: str,
    driver_affinity_score: float,
    car_fit_score: float,
    strategy_fit_score: float,
) -> str:
    """Create a short plain-English summary of the circuit advantage balance."""

    if car_fit_score >= max(abs(driver_affinity_score), abs(strategy_fit_score)):
        anchor = "car architecture fit"
    elif abs(driver_affinity_score) >= abs(strategy_fit_score):
        anchor = "driver circuit affinity"
    else:
        anchor = "strategy fit"
    direction = "advantage" if (0.50 * car_fit_score + 0.35 * driver_affinity_score + 0.15 * strategy_fit_score) > 0 else "headwind"
    return (
        f"{driver_id} in the {constructor_id} carries a {direction} at {circuit_id} "
        f"driven primarily by {anchor}."
    )


def compute_circuit_advantage(
    driver_id: str,
    constructor_id: str,
    circuit_id: str,
    season: int,
    round_number: int,
) -> CircuitAdvantage:
    """Combine driver affinity, car fit, and strategy fit into one circuit score."""

    circuit_snapshot = get_circuit_tsi(driver_id=str(driver_id), circuit_id=str(circuit_id), season=int(season), round_number=int(round_number))
    archetype_snapshot = get_archetype_tsi(driver_id=str(driver_id), circuit_id=str(circuit_id), season=int(season), round_number=int(round_number))
    driver_affinity_score = _normalize_tsi_to_advantage(
        score=float(circuit_snapshot.get("circuit_tsi_score", 0.0) or 0.0),
        confidence=float(circuit_snapshot.get("circuit_tsi_confidence", 0.0) or 0.0),
        archetype_score=float(archetype_snapshot or 0.0),
    )

    car_fit = compute_car_circuit_fit(
        constructor_id=str(constructor_id),
        circuit_id=str(circuit_id),
        season=int(season),
        round_number=int(round_number),
    )
    car_fit_score = float(np.clip(car_fit.fit_score, -1.0, 1.0))

    strategy_fit = compute_strategy_fit(
        constructor_id=str(constructor_id),
        circuit_id=str(circuit_id),
        season=int(season),
        round_number=int(round_number),
    )
    strategy_fit_score = float(np.clip((strategy_fit.score - 0.5) * 2.0, -1.0, 1.0))

    combined_score = float(
        np.clip(
            0.50 * car_fit_score + 0.35 * driver_affinity_score + 0.15 * strategy_fit_score,
            -1.0,
            1.0,
        )
    )
    factor_values = {
        "car_fit": car_fit_score,
        "driver_affinity": driver_affinity_score,
        "strategy_fit": strategy_fit_score,
    }
    primary_factor = max(factor_values, key=lambda key: abs(factor_values[key]))
    return CircuitAdvantage(
        combined_score=combined_score,
        driver_affinity=driver_affinity_score,
        car_fit=car_fit_score,
        strategy_fit=strategy_fit_score,
        primary_factor=primary_factor,
        human_readable=generate_advantage_narrative(
            driver_id=str(driver_id),
            constructor_id=str(constructor_id),
            circuit_id=str(circuit_id),
            driver_affinity_score=driver_affinity_score,
            car_fit_score=car_fit_score,
            strategy_fit_score=strategy_fit_score,
        ),
    )


def apply_circuit_advantage(
    win_probs: np.ndarray,
    drivers: list[tuple[str, str]],
    circuit_id: str,
    season: int,
    round_number: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply circuit-aware driver-car multipliers to a win distribution."""

    rows: list[dict[str, Any]] = []
    multipliers: list[float] = []
    for driver_id, constructor_id in drivers:
        advantage = compute_circuit_advantage(
            driver_id=str(driver_id),
            constructor_id=str(constructor_id),
            circuit_id=str(circuit_id),
            season=int(season),
            round_number=int(round_number),
        )
        multiplier = float(np.exp(advantage.combined_score * 0.7))
        multipliers.append(multiplier)
        rows.append(
            {
                "driver_id": str(driver_id),
                "constructor_id": str(constructor_id),
                "circuit_advantage_score": advantage.combined_score,
                "circuit_driver_affinity": advantage.driver_affinity,
                "circuit_car_fit": advantage.car_fit,
                "circuit_strategy_fit": advantage.strategy_fit,
                "circuit_primary_factor": advantage.primary_factor,
                "circuit_advantage_narrative": advantage.human_readable,
            }
        )

    adjusted = normalize_probabilities(np.asarray(win_probs, dtype=float) * np.asarray(multipliers, dtype=float))
    return adjusted, pd.DataFrame(rows)


def _recover_collapsed_base_distribution(base_frame: pd.DataFrame) -> pd.DataFrame:
    """Blend in circuit and grid priors when the winner model collapses."""

    if base_frame.empty or "relative_win_probability" not in base_frame.columns:
        return base_frame

    current = normalize_probabilities(base_frame["relative_win_probability"].to_numpy(dtype=float))
    health = check_prediction_health(current, base_frame["driver_id"].astype(str).tolist())
    if health.status != "COLLAPSED":
        return base_frame

    flattened = normalize_probabilities(np.power(np.clip(current, 1e-6, None), 0.55))
    if "circuit_advantage_score" in base_frame.columns:
        circuit_prior = normalize_probabilities(np.exp(base_frame["circuit_advantage_score"].to_numpy(dtype=float) * 1.25))
    else:
        circuit_prior = np.ones(len(base_frame), dtype=float) / max(len(base_frame), 1)
    qualifying_prior = normalize_probabilities(
        1.0 / np.clip(base_frame["qualifying_position"].to_numpy(dtype=float), 1.0, None)
    )
    base_frame["relative_win_probability"] = normalize_probabilities(
        0.30 * flattened + 0.50 * circuit_prior + 0.20 * qualifying_prior
    )
    return base_frame


def _ordering_strength(base_frame: pd.DataFrame) -> np.ndarray:
    """Construct a stable ordering-strength vector for vectorized race sampling."""

    qualifying_component = normalize_probabilities(
        1.0 / np.clip(base_frame["qualifying_position"].to_numpy(dtype=float), 1.0, None)
    )
    finish_component = normalize_probabilities(
        1.0 / np.clip(base_frame["expected_finish"].to_numpy(dtype=float), 1.0, None)
    )
    elo_source = "driver_wet_elo" if "driver_wet_elo" in base_frame.columns else "driver_elo"
    elo = base_frame[elo_source].to_numpy(dtype=float)
    elo_component = normalize_probabilities((elo - elo.min()) / (elo.max() - elo.min() + 1e-6) + 0.2)
    reliability_component = normalize_probabilities(
        np.clip(base_frame["team_reliability_last_10"].to_numpy(dtype=float), 0.05, 1.0)
    )
    win_component = normalize_probabilities(base_frame["relative_win_probability"].to_numpy(dtype=float))

    if "circuit_advantage_score" in base_frame.columns:
        circuit_advantage = base_frame["circuit_advantage_score"].fillna(0.0).to_numpy(dtype=float)
        affinity_shifted = circuit_advantage - circuit_advantage.min() + 0.1
        circuit_affinity_component = normalize_probabilities(affinity_shifted)
    # Circuit affinity fallback: negative circuit_tsi_score = specialist (outperforms car at this circuit)
    elif "circuit_tsi_score" in base_frame.columns:
        circuit_tsi = base_frame["circuit_tsi_score"].fillna(0.0).to_numpy(dtype=float)
        circuit_confidence = base_frame["circuit_tsi_confidence"].fillna(0.0).to_numpy(dtype=float) if "circuit_tsi_confidence" in base_frame.columns else np.ones_like(circuit_tsi)
        affinity_raw = np.clip(-circuit_tsi * circuit_confidence, -1.0, 1.0)
        affinity_shifted = affinity_raw - affinity_raw.min() + 0.1
        circuit_affinity_component = normalize_probabilities(affinity_shifted)
    else:
        circuit_affinity_component = np.ones(len(base_frame)) / len(base_frame)

    return normalize_probabilities(
        0.50 * win_component
        + 0.18 * qualifying_component
        + 0.13 * finish_component
        + 0.06 * elo_component
        + 0.03 * reliability_component
        + 0.10 * circuit_affinity_component
    )


def _build_base_frame(config: RaceConfig, apply_circuit_advantage_score: bool = True) -> pd.DataFrame:
    """Build the base prediction and feature frame used across simulations."""

    prediction_frame = predict_race(
        drivers_list=config.drivers,
        qualifying_positions=config.qualifying_positions,
        circuit_id=config.circuit_id,
        season=config.season,
        round_number=config.round,
        weather_mode=config.weather_mode,
    )
    prediction_frame["relative_win_probability"] = normalize_probabilities(
        prediction_frame["raw_win_score"].to_numpy(dtype=float)
    )
    feature_frame = build_race_inference_frame(
        drivers_list=config.drivers,
        qualifying_positions=config.qualifying_positions,
        circuit_id=config.circuit_id,
        season=config.season,
        round_number=config.round,
        weather_mode=config.weather_mode,
    )
    merge_columns = [
        "driver_id",
        "driver_elo",
        "driver_wet_elo",
        "dnf_rate_last_10",
        "dnf_mechanical_last_3",
        "dnf_accident_last_3",
        "team_reliability_last_10",
        "constructor_elo",
        "team_avg_finish_last_5",
        "team_pit_stop_avg",
        "track_risk",
        "circuit_tsi_score",
        "circuit_tsi_confidence",
    ]
    feature_context = feature_frame[merge_columns].copy()
    overlapping = [column for column in merge_columns if column != "driver_id" and column in prediction_frame.columns]
    if overlapping:
        feature_context = feature_context.drop(columns=overlapping, errors="ignore")
    base_frame = prediction_frame.merge(
        feature_context,
        how="left",
        on="driver_id",
    )
    base_frame = base_frame.sort_values("qualifying_position").reset_index(drop=True)
    season_uncertainty = REGULATION_HANDLER.get_season_uncertainty(config.season, config.round)
    regulation_warning = REGULATION_HANDLER.has_reset_warning(config.season, config.round)
    if regulation_warning:
        constructor_prior = base_frame["constructor_id"].map(
            lambda constructor_id: REGULATION_HANDLER.get_constructor_prior(str(constructor_id), config.season)
        )
        original_constructor_elo = np.clip(base_frame["constructor_elo"].to_numpy(dtype=float), 1200.0, 2200.0)
        adjustment_factor = np.clip(
            constructor_prior.to_numpy(dtype=float) / np.clip(original_constructor_elo, 1.0, None),
            0.85,
            1.15,
        )
        base_frame["constructor_elo"] = constructor_prior.to_numpy(dtype=float)
        base_frame["relative_win_probability"] = normalize_probabilities(
            base_frame["relative_win_probability"].to_numpy(dtype=float) * adjustment_factor
        )
        base_frame["expected_finish"] = np.clip(
            base_frame["expected_finish"].to_numpy(dtype=float) / adjustment_factor,
            1.0,
            float(len(base_frame)),
        )
    base_frame["season_uncertainty"] = float(season_uncertainty)
    base_frame["regulation_warning"] = int(regulation_warning)

    if apply_circuit_advantage_score and not base_frame.empty:
        adjusted_probs, advantage_frame = apply_circuit_advantage(
            win_probs=base_frame["relative_win_probability"].to_numpy(dtype=float),
            drivers=list(
                zip(
                    base_frame["driver_id"].astype(str).tolist(),
                    base_frame["constructor_id"].astype(str).tolist(),
                )
            ),
            circuit_id=str(config.circuit_id),
            season=int(config.season),
            round_number=int(config.round),
        )
        base_frame["relative_win_probability"] = adjusted_probs
        base_frame = base_frame.merge(advantage_frame, how="left", on=["driver_id", "constructor_id"])
    else:
        base_frame["circuit_advantage_score"] = 0.0
        base_frame["circuit_driver_affinity"] = 0.0
        base_frame["circuit_car_fit"] = 0.0
        base_frame["circuit_strategy_fit"] = 0.0
        base_frame["circuit_primary_factor"] = "none"
        base_frame["circuit_advantage_narrative"] = ""
    base_frame = _recover_collapsed_base_distribution(base_frame)
    return base_frame


def compute_upset_probability(
    circuit_id: str,
    season: int,
    round_number: int,
    weather_forecast: dict[str, Any],
) -> float:
    """Estimate the chance that a race produces a surprising result."""

    processed_results_path = BASE_CONFIG["processed_dir"] / "race_results.csv"
    race_results = load_dataframe(processed_results_path) if processed_results_path.exists() else pd.DataFrame()
    actual_race = race_results.loc[
        (race_results["season"] == int(season)) & (race_results["round"] == int(round_number))
    ].copy()

    if not actual_race.empty:
        actual_race = actual_race.sort_values("qualifying_position")
        drivers = actual_race["driver_id"].tolist()
        qualifying_positions = {
            row.driver_id: int(row.qualifying_position)
            for row in actual_race[["driver_id", "qualifying_position"]].itertuples(index=False)
        }
        weather_mode = str(actual_race["race_condition"].iloc[0]) if "race_condition" in actual_race.columns else weather_forecast.get("weather_mode")
    else:
        roster = get_season_roster(int(season))
        drivers = sorted(
            [row["driver_id"] for row in roster],
            key=lambda driver_id: get_driver_dry_elo(driver_id, season=int(season), round_number=int(round_number)),
            reverse=True,
        )
        qualifying_positions = {driver_id: idx + 1 for idx, driver_id in enumerate(drivers)}
        weather_mode = weather_forecast.get("weather_mode")

    if not drivers:
        return 0.50

    inference_frame = build_race_inference_frame(
        drivers_list=drivers,
        qualifying_positions=qualifying_positions,
        circuit_id=str(circuit_id),
        season=int(season),
        round_number=int(round_number),
        weather_mode=str(weather_mode) if weather_mode is not None else None,
    )
    predictions = predict_race(
        drivers_list=drivers,
        qualifying_positions=qualifying_positions,
        circuit_id=str(circuit_id),
        season=int(season),
        round_number=int(round_number),
        weather_mode=str(weather_mode) if weather_mode is not None else None,
    )

    win_probs = predictions["relative_win_probability"].to_numpy(dtype=float) if "relative_win_probability" in predictions.columns else normalize_probabilities(predictions["raw_win_score"].to_numpy(dtype=float))
    entropy = float(-np.sum(win_probs * np.log(np.clip(win_probs, 1e-9, None))))
    max_entropy = float(np.log(len(win_probs)))
    field_competitiveness = entropy / max(max_entropy, 1e-9)

    top_prob = float(np.max(win_probs))
    concentration = 1.0 - top_prob

    precipitation = float(weather_forecast.get("race_precipitation_mm", 0.0) or 0.0)
    weather_mode_normalized = str(weather_forecast.get("weather_mode", weather_mode or "dry")).lower()
    if weather_mode_normalized == "mixed":
        weather_volatility = 0.95
    elif weather_mode_normalized == "wet":
        weather_volatility = 0.75
    else:
        weather_volatility = float(np.clip(precipitation / 3.0, 0.0, 0.6))

    track_row = inference_frame.head(1)
    circuit_sc_probability = float(track_row["circuit_sc_probability"].iloc[0]) if not track_row.empty else 0.5
    overtake_difficulty = float(track_row["circuit_overtake_difficulty"].iloc[0]) / 10.0 if not track_row.empty else 0.5
    street_bonus = 0.10 * float(track_row["is_street_circuit"].iloc[0]) if not track_row.empty else 0.0
    regulation_uncertainty = 1.0 - float(REGULATION_HANDLER.get_season_uncertainty(int(season), int(round_number)))

    recent_reliability_issues = 0.0
    if not actual_race.empty:
        recent_window = race_results.loc[
            (race_results["season"] == int(season))
            & (race_results["round"].between(max(1, int(round_number) - 3), int(round_number)))
        ].copy()
        if not recent_window.empty:
            recent_reliability_issues = float(recent_window["dnf_reason"].astype(str).str.lower().eq("mechanical").mean())

    upset_score = (
        0.22 * circuit_sc_probability
        + 0.20 * weather_volatility
        + 0.24 * field_competitiveness
        + 0.12 * concentration
        + 0.12 * overtake_difficulty
        + 0.06 * regulation_uncertainty
        + 0.04 * recent_reliability_issues
        + street_bonus
    )
    return float(np.clip(upset_score, 0.0, 1.0))


def _widen_interval(
    mean_values: np.ndarray,
    lower_values: np.ndarray,
    upper_values: np.ndarray,
    uncertainty: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen uncertainty intervals during low-confidence regulation reset periods."""

    if uncertainty >= 1.0:
        return lower_values, upper_values
    widening_factor = 1.0 / max(float(uncertainty), 1e-6)
    widened_lower = np.clip(mean_values - (mean_values - lower_values) * widening_factor, lower_bound, upper_bound)
    widened_upper = np.clip(mean_values + (upper_values - mean_values) * widening_factor, lower_bound, upper_bound)
    return widened_lower, widened_upper


def _simulate_race_matrices(
    base_frame: pd.DataFrame,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize race-order, points, DNF and fastest-lap sampling for all simulations."""

    if int(n_simulations) <= 0:
        raise ValueError("n_simulations must be positive")

    frame = base_frame.reset_index(drop=True).copy()
    n_drivers = len(frame)
    if n_drivers == 0:
        raise ValueError("Cannot simulate a race with no drivers")

    lap_ceiling = int(CONFIG["default_lap_ceiling"])
    season_uncertainty = float(frame["season_uncertainty"].iloc[0]) if "season_uncertainty" in frame.columns else 1.0
    noise_sigma = float(
        CONFIG["base_order_noise"] + max(0.0, 1.0 - season_uncertainty) * CONFIG["uncertainty_noise_scale"]
    )

    dnf_probs = np.clip(
        pd.to_numeric(frame.get("dnf_prob", 0.05), errors="coerce").fillna(0.05).to_numpy(dtype=float),
        float(CONFIG["min_dnf_probability"]),
        float(CONFIG["max_dnf_probability"]),
    )
    dnf_matrix = rng.random((int(n_simulations), n_drivers)) < dnf_probs.reshape(1, -1)
    all_dnf_rows = dnf_matrix.all(axis=1)
    if np.any(all_dnf_rows):
        survivor_index = int(np.argmin(dnf_probs))
        dnf_matrix[all_dnf_rows, survivor_index] = False

    retirement_laps = np.full((int(n_simulations), n_drivers), lap_ceiling, dtype=int)
    sampled_retirement_laps = np.clip(
        np.ceil(rng.beta(4.0, 2.0, size=(int(n_simulations), n_drivers)) * lap_ceiling).astype(int),
        1,
        lap_ceiling,
    )
    retirement_laps[dnf_matrix] = sampled_retirement_laps[dnf_matrix]

    order_strength = _ordering_strength(frame).reshape(1, -1)
    order_noise = np.clip(
        rng.normal(loc=1.0, scale=max(noise_sigma, 1e-6), size=(int(n_simulations), n_drivers)),
        0.05,
        None,
    )
    noisy_strength = _normalize_rows(order_strength * order_noise)

    uniform_samples = np.clip(rng.random((int(n_simulations), n_drivers)), 1e-12, 1.0)
    plackett_luce_keys = -np.log(uniform_samples) / noisy_strength
    provisional_order = np.argsort(plackett_luce_keys, axis=1)

    row_index = np.arange(int(n_simulations))[:, None]
    provisional_rank = np.empty_like(provisional_order)
    provisional_rank[row_index, provisional_order] = np.arange(n_drivers)

    dnf_back_score = (
        n_drivers
        + (lap_ceiling - retirement_laps).astype(float)
        + provisional_rank.astype(float) / max(float(n_drivers), 1.0)
    )
    final_sort_score = np.where(~dnf_matrix, provisional_rank.astype(float), dnf_back_score)
    final_order = np.argsort(final_sort_score, axis=1)

    points_schedule = np.zeros(n_drivers, dtype=float)
    standard_points = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
    points_schedule[: min(len(standard_points), n_drivers)] = standard_points[: min(len(standard_points), n_drivers)]

    position_index = np.arange(n_drivers)
    points_matrix = np.zeros((int(n_simulations), n_drivers), dtype=float)
    points_matrix[row_index, final_order] = np.broadcast_to(points_schedule, (int(n_simulations), n_drivers))

    top_n = min(10, n_drivers)
    elo_source = "driver_wet_elo" if "driver_wet_elo" in frame.columns else "driver_elo"
    fastest_base = normalize_probabilities(
        (1.0 / np.clip(frame["expected_finish"].to_numpy(dtype=float), 1.0, None))
        + (
            frame[elo_source].to_numpy(dtype=float)
            / max(float(frame[elo_source].max()), 1.0)
        )
    )
    top_finishers = final_order[:, :top_n]
    fastest_weights = _normalize_rows(fastest_base[top_finishers])
    cumulative = np.cumsum(fastest_weights, axis=1)
    draws = rng.random((int(n_simulations), 1))
    fastest_positions = np.minimum((draws > cumulative).sum(axis=1), top_n - 1)
    fastest_lap_driver_indices = top_finishers[np.arange(int(n_simulations)), fastest_positions]
    points_matrix[np.arange(int(n_simulations)), fastest_lap_driver_indices] += 1.0

    finishing_positions = np.empty_like(final_order)
    finishing_positions[row_index, final_order] = position_index
    return final_order, finishing_positions, points_matrix, fastest_lap_driver_indices


def _simulate_single_race(
    base_frame: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, int]]:
    """Run a single race simulation and return the finishing order and points."""

    final_order, _, points_matrix, _ = _simulate_race_matrices(base_frame, n_simulations=1, rng=rng)
    order_indices = final_order[0]
    driver_ids = base_frame["driver_id"].astype(str).tolist()
    finishing_order = [driver_ids[int(index)] for index in order_indices]
    points = {
        driver_ids[int(driver_index)]: int(points_matrix[0, int(driver_index)])
        for driver_index in range(len(driver_ids))
    }
    return finishing_order, points


def simulate_race(
    race_config: RaceConfig | dict[str, Any],
    apply_circuit_advantage: bool | None = None,
) -> RaceSimulationResult:
    """Run Monte Carlo simulation for a single race weekend."""

    config = race_config if isinstance(race_config, RaceConfig) else RaceConfig(**race_config)
    if apply_circuit_advantage is None:
        apply_circuit_advantage = bool(getattr(config, "apply_circuit_advantage", True))
    base_frame = _build_base_frame(config, apply_circuit_advantage_score=bool(apply_circuit_advantage))
    base_frame = base_frame.copy()
    for column, default in {
        "circuit_advantage_score": 0.0,
        "circuit_driver_affinity": 0.0,
        "circuit_car_fit": 0.0,
        "circuit_strategy_fit": 0.0,
        "circuit_primary_factor": "none",
        "circuit_advantage_narrative": "",
    }.items():
        if column not in base_frame.columns:
            base_frame[column] = default
    season_uncertainty = (
        float(base_frame["season_uncertainty"].iloc[0]) if "season_uncertainty" in base_frame.columns else 1.0
    )
    regulation_warning = bool(
        int(base_frame["regulation_warning"].iloc[0]) if "regulation_warning" in base_frame.columns else 0
    )
    rng = np.random.default_rng(config.random_seed)
    driver_ids = base_frame["driver_id"].tolist()
    n_drivers = len(driver_ids)
    prediction_health = check_prediction_health(
        base_frame["relative_win_probability"].to_numpy(dtype=float),
        [str(driver_id) for driver_id in driver_ids],
    )
    if prediction_health.status == "COLLAPSED":
        base_frame["relative_win_probability"] = _apply_probability_floor(
            base_frame["relative_win_probability"].to_numpy(dtype=float),
            minimum_probability=float(CONFIG["minimum_win_probability_floor"]),
        )
        prediction_health.floor_applied = True

    final_order_matrix, finishing_positions, points_matrix, _ = _simulate_race_matrices(
        base_frame,
        n_simulations=int(config.n_simulations),
        rng=rng,
    )

    position_count_matrix = np.zeros((n_drivers, n_drivers), dtype=int)
    np.add.at(
        position_count_matrix,
        (final_order_matrix.ravel(), np.tile(np.arange(n_drivers), int(config.n_simulations))),
        1,
    )

    win_indicator_matrix = (finishing_positions == 0).astype(float)
    podium_indicator_matrix = (finishing_positions < 3).astype(float)
    points_indicator_matrix = (points_matrix > 0).astype(float)

    base_info = base_frame.copy()
    defaults: dict[str, Any] = {
        "driver_name": base_info["driver_id"].astype(str) if "driver_id" in base_info.columns else "",
        "constructor_id": "",
        "constructor_name": "",
        "driver_wet_elo": base_info["driver_elo"] if "driver_elo" in base_info.columns else np.nan,
        "race_condition": "dry",
        "is_wet_race": 0,
        "is_mixed_race": 0,
    }
    for column, default in defaults.items():
        if column not in base_info.columns:
            base_info[column] = default
    base_info = base_info[
        [
            "driver_id",
            "driver_name",
            "constructor_id",
            "constructor_name",
            "driver_wet_elo",
            "race_condition",
            "is_wet_race",
            "is_mixed_race",
        ]
    ].drop_duplicates()

    win_p10, win_p90 = _block_mean_interval(win_indicator_matrix)
    podium_p10, podium_p90 = _block_mean_interval(podium_indicator_matrix)
    points_p10, points_p90 = _block_mean_interval(points_indicator_matrix)
    expected_points_p10 = np.quantile(points_matrix, 0.10, axis=0)
    expected_points_p90 = np.quantile(points_matrix, 0.90, axis=0)

    win_probs = base_info.merge(
        pd.DataFrame(
            {
                "driver_id": driver_ids,
                "relative_win_probability": win_indicator_matrix.mean(axis=0),
                "relative_win_probability_p10": win_p10,
                "relative_win_probability_p90": win_p90,
            }
        ),
        how="left",
        on="driver_id",
    ).sort_values("relative_win_probability", ascending=False)
    win_lower, win_upper = _widen_interval(
        win_probs["relative_win_probability"].to_numpy(dtype=float),
        win_probs["relative_win_probability_p10"].to_numpy(dtype=float),
        win_probs["relative_win_probability_p90"].to_numpy(dtype=float),
        season_uncertainty,
        0.0,
        1.0,
    )
    win_probs["relative_win_probability_p10"] = win_lower
    win_probs["relative_win_probability_p90"] = win_upper
    win_probs["win_prob"] = win_probs["relative_win_probability"]
    win_probs["win_prob_p10"] = win_probs["relative_win_probability_p10"]
    win_probs["win_prob_p90"] = win_probs["relative_win_probability_p90"]

    podium_probs = base_info.merge(
        pd.DataFrame(
            {
                "driver_id": driver_ids,
                "podium_probability": podium_indicator_matrix.mean(axis=0),
                "podium_probability_p10": podium_p10,
                "podium_probability_p90": podium_p90,
            }
        ),
        how="left",
        on="driver_id",
    ).sort_values("podium_probability", ascending=False)
    podium_lower, podium_upper = _widen_interval(
        podium_probs["podium_probability"].to_numpy(dtype=float),
        podium_probs["podium_probability_p10"].to_numpy(dtype=float),
        podium_probs["podium_probability_p90"].to_numpy(dtype=float),
        season_uncertainty,
        0.0,
        1.0,
    )
    podium_probs["podium_probability_p10"] = podium_lower
    podium_probs["podium_probability_p90"] = podium_upper

    points_probs = base_info.merge(
        pd.DataFrame(
            {
                "driver_id": driver_ids,
                "points_probability": points_indicator_matrix.mean(axis=0),
                "points_probability_p10": points_p10,
                "points_probability_p90": points_p90,
            }
        ),
        how="left",
        on="driver_id",
    ).sort_values("points_probability", ascending=False)
    points_lower, points_upper = _widen_interval(
        points_probs["points_probability"].to_numpy(dtype=float),
        points_probs["points_probability_p10"].to_numpy(dtype=float),
        points_probs["points_probability_p90"].to_numpy(dtype=float),
        season_uncertainty,
        0.0,
        1.0,
    )
    points_probs["points_probability_p10"] = points_lower
    points_probs["points_probability_p90"] = points_upper

    expected_points = base_info.merge(
        pd.DataFrame(
            {
                "driver_id": driver_ids,
                "expected_points": points_matrix.mean(axis=0),
                "expected_points_p10": expected_points_p10,
                "expected_points_p90": expected_points_p90,
            }
        ),
        how="left",
        on="driver_id",
    ).sort_values("expected_points", ascending=False)
    expected_lower, expected_upper = _widen_interval(
        expected_points["expected_points"].to_numpy(dtype=float),
        expected_points["expected_points_p10"].to_numpy(dtype=float),
        expected_points["expected_points_p90"].to_numpy(dtype=float),
        season_uncertainty,
        0.0,
        26.0,
    )
    expected_points["expected_points_p10"] = expected_lower
    expected_points["expected_points_p90"] = expected_upper

    position_columns = [f"P{position}" for position in range(1, len(driver_ids) + 1)]
    position_distributions = pd.concat(
        [
            base_info.set_index("driver_id"),
            pd.DataFrame(position_count_matrix / int(config.n_simulations), index=driver_ids, columns=position_columns),
        ],
        axis=1,
    ).reset_index()

    unique_orders, order_counts = np.unique(final_order_matrix, axis=0, return_counts=True)
    order_distribution = pd.DataFrame(
        {
            "finishing_order": [
                " > ".join(str(driver_ids[int(driver_index)]) for driver_index in order)
                for order in unique_orders
            ],
            "count": order_counts,
        }
    )
    order_distribution["probability"] = order_distribution["count"] / int(config.n_simulations)
    order_distribution = order_distribution.sort_values("count", ascending=False).reset_index(drop=True)

    LOGGER.info(
        "Race simulation complete for %s %s %s with %s iterations",
        config.season,
        config.round,
        config.circuit_id,
        config.n_simulations,
    )

    return RaceSimulationResult(
        win_probs=win_probs.reset_index(drop=True),
        podium_probs=podium_probs.reset_index(drop=True),
        points_probs=points_probs.reset_index(drop=True),
        expected_points=expected_points.reset_index(drop=True),
        position_distributions=position_distributions,
        full_order_distribution=order_distribution,
        simulation_runs=config.n_simulations,
        regulation_warning=regulation_warning,
        season_uncertainty=season_uncertainty,
        prediction_health=prediction_health,
        circuit_advantages=(
            base_frame[
                [
                    "driver_id",
                    "driver_name",
                    "constructor_id",
                    "constructor_name",
                    "circuit_advantage_score",
                    "circuit_driver_affinity",
                    "circuit_car_fit",
                    "circuit_strategy_fit",
                    "circuit_primary_factor",
                    "circuit_advantage_narrative",
                ]
            ]
            .sort_values("circuit_advantage_score", ascending=False)
            .reset_index(drop=True)
            if "circuit_advantage_score" in base_frame.columns
            else None
        ),
    )
