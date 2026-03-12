"""Single entry point for the living F1 AI season system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.championship_tracker import ChampionshipTracker  # noqa: E402
from src.circuit_analysis import run_circuit_analysis_pipeline  # noqa: E402
from src.circuit_tsi import run_circuit_tsi_pipeline  # noqa: E402
from src.car_architecture import run_car_architecture_pipeline  # noqa: E402
from src.car_performance import run_cpi_pipeline  # noqa: E402
from src.data_pipeline import load_calendar, load_processed_inputs, run_data_pipeline  # noqa: E402
from src.elo_rating import get_constructor_elo, get_driver_dry_elo, get_driver_elo  # noqa: E402
from src.fastf1_pipeline import run_fastf1_pipeline  # noqa: E402
from src.feature_engineering import run_feature_pipeline  # noqa: E402
from src.model_training import load_metrics, predict_race, run_training_pipeline  # noqa: E402
from src.prediction_tracker import aggregate_season_accuracy  # noqa: E402
from src.race_simulator import RaceConfig, simulate_race  # noqa: E402
from src.registry import get_driver_name, get_season_roster  # noqa: E402
from src.scheduler import auto_update  # noqa: E402
from src.season_simulator import build_remaining_races_for_season, simulate_season  # noqa: E402
from src.system_integrity import run_system_validation  # noqa: E402
from src.true_skill import run_tsi_pipeline  # noqa: E402
from src.utils import CONFIG as BASE_CONFIG, detect_current_season, format_race_label, get_gp_name, get_logger, load_dataframe  # noqa: E402
from src.weather_pipeline import run_weather_pipeline  # noqa: E402

CONFIG: dict[str, Any] = {
    "project_root": PROJECT_ROOT,
    "processed_dir": BASE_CONFIG["processed_dir"],
}

LOGGER = get_logger(__name__)

PIPELINE_STAGES: list[dict[str, Any]] = [
    {"name": "data", "fn": run_data_pipeline},
    {"name": "weather", "fn": run_weather_pipeline},
    {"name": "fastf1", "fn": run_fastf1_pipeline},
    {"name": "tsi", "fn": run_tsi_pipeline},
    {"name": "cpi", "fn": run_cpi_pipeline},
    {"name": "circuit_tsi", "fn": run_circuit_tsi_pipeline},
    {"name": "circuit_analysis", "fn": run_circuit_analysis_pipeline},
    {"name": "car_architecture", "fn": run_car_architecture_pipeline},
    {"name": "features", "fn": run_feature_pipeline},
    {"name": "training", "fn": run_training_pipeline},
    {"name": "validation", "fn": run_system_validation},
]


def _artifacts_exist(skip_fastf1: bool = False) -> bool:
    """Return whether the core processed artifacts already exist."""

    required_paths = [
        CONFIG["processed_dir"] / "race_results.csv",
        CONFIG["processed_dir"] / "calendar.csv",
        CONFIG["processed_dir"] / "race_weather.csv",
        CONFIG["processed_dir"] / "driver_elo_history.csv",
        CONFIG["processed_dir"] / "constructor_elo_history.csv",
        CONFIG["processed_dir"] / "openf1_weekend_coverage.csv",
        CONFIG["processed_dir"] / "tsi_driver_race.csv",
        CONFIG["processed_dir"] / "cpi_history.csv",
        CONFIG["processed_dir"] / "circuit_clusters.csv",
        CONFIG["processed_dir"] / "car_architecture_history.csv",
        CONFIG["processed_dir"] / "car_architecture_summary.csv",
        CONFIG["processed_dir"] / "features.csv",
        CONFIG["processed_dir"] / "model_A.pkl",
        CONFIG["processed_dir"] / "model_B.pkl",
        CONFIG["processed_dir"] / "model_C.pkl",
        CONFIG["processed_dir"] / "model_D.pkl",
        CONFIG["processed_dir"] / "model_E.pkl",
    ]
    if not all(path.exists() for path in required_paths):
        return False

    if not skip_fastf1:
        fastf1_path = CONFIG["processed_dir"] / "fastf1_features.csv"
        if not fastf1_path.exists():
            return False
        try:
            if load_dataframe(fastf1_path).empty:
                return False
        except Exception as exc:
            LOGGER.debug("Non-critical error checking fastf1 artifacts: %s", exc)
            return False
    return True


def ensure_full_pipeline(refresh: bool = False, skip_fastf1: bool = False) -> None:
    """Run the end-to-end pipeline if artifacts are missing or a refresh was requested."""

    if refresh or not _artifacts_exist(skip_fastf1=skip_fastf1):
        LOGGER.info("Running full living-season pipeline")
        pipeline_data = run_data_pipeline(refresh=refresh)
        LOGGER.info("Skipping DataUniverse enrichment during full rebuild; it is auxiliary to model training.")
        _, race_results = run_weather_pipeline(pipeline_data["race_results"], refresh=refresh)
        fastf1_features, fastf1_failures = run_fastf1_pipeline(
            race_results=race_results,
            refresh=refresh,
            skip=skip_fastf1,
        )
        from src.elo_rating import run_elo_pipeline  # noqa: E402

        driver_elo_history, constructor_elo_history = run_elo_pipeline(race_results)
        tsi_artifacts = run_tsi_pipeline(race_results=race_results)
        cpi_artifacts = run_cpi_pipeline(race_results=race_results, tsi_driver_race=tsi_artifacts.driver_race)
        circuit_tsi_artifacts = run_circuit_tsi_pipeline(race_results=race_results, cpi_history=cpi_artifacts.cpi_history)
        circuit_artifacts = run_circuit_analysis_pipeline(race_results=race_results)
        architecture_artifacts = run_car_architecture_pipeline(
            race_results=race_results,
            fastf1_features=fastf1_features,
        )
        features = run_feature_pipeline(
            race_results=race_results,
            driver_elo_history=driver_elo_history,
            constructor_elo_history=constructor_elo_history,
            pitstops_df=pipeline_data["pitstops"],
            fastf1_features_df=fastf1_features,
            tsi_driver_race=tsi_artifacts.driver_race,
            cpi_history=cpi_artifacts.cpi_history,
            circuit_tsi_matrix=circuit_tsi_artifacts.circuit_tsi_matrix,
            cluster_frame=circuit_artifacts.cluster_frame,
        )
        run_training_pipeline(features)
        validations = run_system_validation(
            features_df=features,
            driver_elo_history=driver_elo_history,
            constructor_elo_history=constructor_elo_history,
            tsi_driver_race=tsi_artifacts.driver_race,
        )
        for validation in validations:
            log_fn = LOGGER.info if validation.passed else LOGGER.error
            log_fn("validation %s | passed=%s | %s", validation.name, validation.passed, validation.details)
        if not fastf1_failures.empty:
            LOGGER.warning("FastF1 completed with %s logged session failures", len(fastf1_failures))
        LOGGER.info(
            "Car architecture artifacts built | history=%s summary=%s",
            len(architecture_artifacts.history),
            len(architecture_artifacts.summary),
        )


def _resolve_future_race(
    race_name: str | None,
    season: int,
    round_number: int,
) -> tuple[str, list[str], dict[str, int]]:
    """Resolve an upcoming or uncompleted race from the live season calendar and roster."""

    remaining = build_remaining_races_for_season(season, from_round=max(0, round_number - 1))
    calendar = load_calendar(season, allow_fetch=False)
    calendar_row = calendar.loc[calendar["round"] == int(round_number)].head(1) if not calendar.empty else pd.DataFrame()

    def matches(config: RaceConfig) -> bool:
        if int(config.round) != int(round_number):
            return False
        if not race_name:
            return True
        lowered = race_name.strip().lower()
        if config.circuit_id.lower() == lowered:
            return True
        if not calendar_row.empty:
            return any(
                lowered in str(calendar_row[column].iloc[0]).lower()
                for column in ["race_name", "circuit_id", "circuit_name", "country"]
                if column in calendar_row.columns
            )
        return lowered in config.circuit_id.lower()

    for config in remaining:
        if matches(config):
            if isinstance(config.qualifying_positions, dict):
                qualifying_map = {driver_id: int(position) for driver_id, position in config.qualifying_positions.items()}
            else:
                qualifying_map = {driver_id: int(config.qualifying_positions[idx]) for idx, driver_id in enumerate(config.drivers)}
            return config.circuit_id, list(config.drivers), qualifying_map

    roster = get_season_roster(season)
    if roster:
        ordering = sorted(
            [row["driver_id"] for row in roster],
            key=lambda driver_id: get_driver_elo(driver_id, season, round_number),
            reverse=True,
        )
        qualifying_map = {driver_id: index + 1 for index, driver_id in enumerate(ordering)}
        circuit_id = (
            str(calendar_row["circuit_id"].iloc[0])
            if not calendar_row.empty and "circuit_id" in calendar_row.columns
            else (race_name or "unknown")
        )
        return circuit_id, ordering, qualifying_map

    raise ValueError(f"Could not resolve race '{race_name}' for season {season} round {round_number}")


def _resolve_race_entries(race_name: str | None, season: int, round_number: int) -> tuple[str, list[str], dict[str, int]]:
    """Resolve a race configuration from completed data or future calendar context."""

    datasets = load_processed_inputs()
    race_results = datasets["race_results"].copy()
    race_frame = race_results.loc[(race_results["season"] == season) & (race_results["round"] == round_number)].copy()

    if race_name and not race_frame.empty:
        lowered = race_name.strip().lower()
        race_frame = race_frame.loc[
            race_frame["race_name"].str.lower().eq(lowered)
            | race_frame["circuit_name"].str.lower().eq(lowered)
            | race_frame["circuit_id"].str.lower().eq(lowered)
            | race_frame["race_name"].str.lower().str.contains(lowered, na=False)
            | race_frame["circuit_name"].str.lower().str.contains(lowered, na=False)
            | race_frame["circuit_id"].str.lower().str.contains(lowered, na=False)
        ].copy()

    if race_frame.empty:
        return _resolve_future_race(race_name, season, round_number)

    race_frame = race_frame.sort_values("qualifying_position")
    circuit_id = str(race_frame["circuit_id"].iloc[0])
    driver_ids = race_frame["driver_id"].tolist()
    qualifying_map = {
        row.driver_id: int(row.qualifying_position)
        for row in race_frame[["driver_id", "qualifying_position"]].itertuples(index=False)
    }
    return circuit_id, driver_ids, qualifying_map


def _prediction_summary(simulation: Any) -> pd.DataFrame:
    """Build a compact driver-level prediction summary."""

    summary = (
        simulation.win_probs[
            [
                "driver_id",
                "driver_name",
                "relative_win_probability",
                "relative_win_probability_p10",
                "relative_win_probability_p90",
            ]
        ]
        .merge(
            simulation.podium_probs[
                ["driver_id", "podium_probability", "podium_probability_p10", "podium_probability_p90"]
            ],
            how="left",
            on="driver_id",
        )
        .merge(
            simulation.expected_points[
                ["driver_id", "expected_points", "expected_points_p10", "expected_points_p90"]
            ],
            how="left",
            on="driver_id",
        )
        .sort_values("relative_win_probability", ascending=False)
        .reset_index(drop=True)
    )
    summary["relative_win_range"] = summary.apply(
        lambda row: (
            f"{row['relative_win_probability']:.1%} "
            f"({row['relative_win_probability_p10']:.1%}-{row['relative_win_probability_p90']:.1%})"
        ),
        axis=1,
    )
    summary["podium_range"] = summary.apply(
        lambda row: f"{row['podium_probability']:.1%} ({row['podium_probability_p10']:.1%}-{row['podium_probability_p90']:.1%})",
        axis=1,
    )
    summary["expected_points_range"] = summary.apply(
        lambda row: f"{row['expected_points']:.2f} ({row['expected_points_p10']:.2f}-{row['expected_points_p90']:.2f})",
        axis=1,
    )
    return summary


def run_full_mode(refresh: bool = False, skip_fastf1: bool = False) -> None:
    """Execute the full pipeline and log model metrics."""

    ensure_full_pipeline(refresh=refresh, skip_fastf1=skip_fastf1)
    metrics = load_metrics()
    LOGGER.info("Full pipeline completed successfully")
    LOGGER.info("Model metrics:\n%s", str(metrics))


def run_predict_mode(race_name: str | None, season: int, round_number: int, simulations: int) -> None:
    """Predict a specific race and log user-facing outputs."""

    race_label = format_race_label(season, round_number)
    ensure_full_pipeline()
    circuit_id, driver_ids, qualifying_map = _resolve_race_entries(race_name, season, round_number)
    predictions = predict_race(
        drivers_list=driver_ids,
        qualifying_positions=qualifying_map,
        circuit_id=circuit_id,
        season=season,
        round_number=round_number,
    )
    simulation = simulate_race(
        RaceConfig(
            drivers=driver_ids,
            qualifying_positions=qualifying_map,
            circuit_id=circuit_id,
            season=season,
            round=round_number,
            n_simulations=simulations,
        )
    )
    summary = _prediction_summary(simulation)
    merged = predictions[["driver_id", "constructor_name", "qualifying_position", "expected_finish", "dnf_prob"]].merge(
        summary,
        how="left",
        on="driver_id",
    )
    LOGGER.info(
        "Race prediction | %s | request=%s | circuit=%s\n%s",
        race_label,
        race_name or get_gp_name(season, round_number),
        circuit_id,
        merged[
            [
                "driver_name",
                "constructor_name",
                "qualifying_position",
                "relative_win_range",
                "podium_range",
                "expected_points_range",
                "expected_finish",
                "dnf_prob",
            ]
        ].to_string(index=False),
    )
    if simulation.regulation_warning:
        LOGGER.warning(
            "Regulation Reset Warning: %s uncertainty multiplier=%.2f",
            race_label,
            simulation.season_uncertainty,
        )


def run_simulate_season_mode(season: int, simulations: int) -> None:
    """Simulate the remainder of a season and log championship outlooks."""

    ensure_full_pipeline()
    remaining_races = build_remaining_races_for_season(season)
    tracker = ChampionshipTracker(season)
    current_standings = tracker._current_driver_points()
    result = simulate_season(
        remaining_races=remaining_races,
        current_standings=current_standings,
        n_simulations=simulations,
    )
    LOGGER.info("WDC probabilities for season %s:\n%s", season, result.wdc_probs.to_string(index=False))
    LOGGER.info("WCC probabilities for season %s:\n%s", season, result.wcc_probs.to_string(index=False))
    LOGGER.info("Projected final standings:\n%s", result.projected_standings.to_string(index=False))


def run_update_mode(skip_fastf1: bool = False) -> None:
    """Refresh the active season if a new round is available."""

    ensure_full_pipeline(skip_fastf1=skip_fastf1)
    result = auto_update(skip_fastf1=skip_fastf1)
    LOGGER.info("Update result:\n%s", result)


def run_championship_mode(season: int, simulations: int) -> None:
    """Print the current championship state, momentum, and title scenarios."""

    ensure_full_pipeline()
    tracker = ChampionshipTracker(season)
    snapshot = tracker.snapshot(n_simulations=simulations)
    standings = snapshot["driver_standings"].copy()
    probs = snapshot["wdc_probabilities"].copy()
    standings = standings.merge(probs, how="left", on="driver_id").fillna({"wdc_probability": 0.0})
    standings["momentum"] = 0.0
    momentum_ids = standings.sort_values("points", ascending=False).head(5)["driver_id"].tolist()
    for driver_id in momentum_ids:
        standings.loc[standings["driver_id"] == driver_id, "momentum"] = tracker.get_momentum_index(str(driver_id))
    standings["momentum_arrow"] = standings["momentum"].map(
        lambda value: "↑" if value > 0.01 else ("↓" if value < -0.01 else "→")
    )
    standings["driver_name"] = standings["driver_id"].map(get_driver_name)

    LOGGER.info(
        "Championship snapshot | season=%s after=%s completed_round=%s races_remaining=%s\n%s",
        season,
        get_gp_name(season, snapshot["current_round"]) if snapshot["current_round"] else "Season start",
        snapshot["current_round"],
        snapshot["races_remaining"],
        standings[
            ["position", "driver_name", "constructor_id", "points", "wdc_probability", "momentum_arrow", "momentum"]
        ].sort_values("position").to_string(index=False),
    )

    accuracy = aggregate_season_accuracy(season)
    LOGGER.info("Prediction accuracy summary for season %s:\n%s", season, accuracy)

    top_three = standings.sort_values("points", ascending=False).head(3)
    for row in top_three.itertuples(index=False):
        scenario = tracker.get_championship_scenarios(str(row.driver_id))
        LOGGER.info(
            "Scenario | %s | eliminated=%s | max_points=%s | %s",
            get_driver_name(str(row.driver_id)),
            scenario.mathematically_eliminated,
            scenario.max_possible_points,
            scenario.required_results,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the F1 AI living-season pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "predict", "simulate-season", "update", "championship"],
        required=True,
        help="Execution mode",
    )
    parser.add_argument("--race", type=str, default=None, help="Race name, circuit name, or circuit id")
    parser.add_argument("--season", type=int, default=detect_current_season(), help="Season to use")
    parser.add_argument("--round", type=int, default=1, help="Round number to use")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh cached raw API data")
    parser.add_argument("--skip-fastf1", action="store_true", help="Skip the FastF1 telemetry stage")
    parser.add_argument("--simulations", type=int, default=10_000, help="Number of simulation iterations")
    return parser.parse_args()


def main() -> None:
    """Dispatch the requested pipeline mode."""

    args = parse_args()
    if args.mode == "full":
        run_full_mode(refresh=args.refresh, skip_fastf1=args.skip_fastf1)
    elif args.mode == "predict":
        run_predict_mode(args.race, args.season, args.round, args.simulations)
    elif args.mode == "simulate-season":
        run_simulate_season_mode(args.season, args.simulations)
    elif args.mode == "update":
        run_update_mode(skip_fastf1=args.skip_fastf1)
    else:
        run_championship_mode(args.season, args.simulations)


if __name__ == "__main__":
    main()
