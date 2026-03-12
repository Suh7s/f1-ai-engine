"""Monte Carlo simulation for an F1 season championship."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline import get_current_season, load_calendar
from src.elo_rating import get_driver_elo, get_driver_wet_elo
from src.registry import get_season_roster
from src.race_simulator import RaceConfig, _build_base_frame, _simulate_single_race
from src.utils import CONFIG as BASE_CONFIG, get_logger, load_dataframe

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "driver_standings_path": BASE_CONFIG["processed_dir"] / "driver_standings.csv",
    "constructor_standings_path": BASE_CONFIG["processed_dir"] / "constructor_standings.csv",
    "default_iterations": 10_000,
    "random_seed": 42,
}

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class SeasonSimulationResult:
    """Aggregated outputs from a season simulation."""

    wdc_probs: pd.DataFrame
    wcc_probs: pd.DataFrame
    expected_final_points: pd.DataFrame
    expected_final_positions: pd.DataFrame
    probability_evolution: pd.DataFrame
    projected_standings: pd.DataFrame
    simulation_runs: int


def load_current_standings_from_history(season: int) -> dict[str, pd.DataFrame]:
    """Load the latest available standings snapshots for a season."""

    driver_standings = load_dataframe(CONFIG["driver_standings_path"])
    constructor_standings = load_dataframe(CONFIG["constructor_standings_path"])

    driver_snapshot = driver_standings.loc[driver_standings["season"] == season].copy()
    constructor_snapshot = constructor_standings.loc[constructor_standings["season"] == season].copy()

    if not driver_snapshot.empty:
        driver_snapshot = driver_snapshot.loc[
            driver_snapshot["round"] == int(driver_snapshot["round"].max())
        ].copy()
    if not constructor_snapshot.empty:
        constructor_snapshot = constructor_snapshot.loc[
            constructor_snapshot["round"] == int(constructor_snapshot["round"].max())
        ].copy()

    return {
        "drivers": driver_snapshot[["driver_id", "constructor_id", "points"]].reset_index(drop=True)
        if not driver_snapshot.empty
        else pd.DataFrame(columns=["driver_id", "constructor_id", "points"]),
        "constructors": constructor_snapshot[["constructor_id", "points"]].reset_index(drop=True)
        if not constructor_snapshot.empty
        else pd.DataFrame(columns=["constructor_id", "points"]),
    }


def build_remaining_races_for_season(season: int, from_round: int | None = None) -> list[RaceConfig]:
    """Construct season race configs using the calendar plus current roster priors."""

    season = int(season)
    race_results = load_dataframe(CONFIG["race_results_path"])
    available_seasons = sorted(race_results["season"].dropna().astype(int).unique().tolist())
    season_calendar = load_calendar(season, allow_fetch=False)
    completed_round = int(from_round or 0)
    current_season = get_current_season()

    if season_calendar.empty and season in available_seasons:
        season_frame = race_results.loc[race_results["season"] == season].copy()
        if season < current_season and completed_round == 0:
            remaining_frame = season_frame
        else:
            remaining_frame = season_frame.loc[season_frame["round"] > completed_round].copy()
        configs: list[RaceConfig] = []
        for round_number, race_df in remaining_frame.groupby("round", sort=True):
            race_df = race_df.sort_values("qualifying_position")
            weather_mode = str(race_df["race_condition"].iloc[0]) if "race_condition" in race_df.columns else None
            configs.append(
                RaceConfig(
                    drivers=race_df["driver_id"].tolist(),
                    qualifying_positions={
                        row.driver_id: int(row.qualifying_position)
                        for row in race_df.itertuples(index=False)
                    },
                    circuit_id=str(race_df["circuit_id"].iloc[0]),
                    season=season,
                    round=int(round_number),
                    weather_mode=weather_mode,
                    n_simulations=1,
                )
            )
        return configs

    if season_calendar.empty:
        return []

    if from_round is None:
        completed_round = int(season_calendar.loc[season_calendar["is_completed"] == 1, "round"].max()) if season_calendar["is_completed"].any() else 0

    remaining_calendar = season_calendar.loc[season_calendar["round"] > completed_round].sort_values("round").copy()
    if remaining_calendar.empty:
        return []

    season_results = race_results.loc[race_results["season"] == season].copy()
    latest_round_frame = (
        season_results.loc[season_results["round"] == int(season_results["round"].max())].copy()
        if not season_results.empty
        else pd.DataFrame()
    )

    if not latest_round_frame.empty:
        driver_ids = latest_round_frame["driver_id"].dropna().unique().tolist()
    else:
        driver_ids = [row["driver_id"] for row in get_season_roster(season)]

    configs: list[RaceConfig] = []
    for calendar_row in remaining_calendar.itertuples(index=False):
        historical_row = season_results.loc[season_results["round"] == int(calendar_row.round)].head(1)
        weather_mode = (
            str(historical_row["race_condition"].iloc[0])
            if not historical_row.empty and "race_condition" in historical_row.columns
            else None
        )
        ordering = sorted(
            driver_ids,
            key=lambda driver_id: (
                get_driver_wet_elo(driver_id, season, int(calendar_row.round))
                if weather_mode in {"wet", "mixed"}
                else get_driver_elo(driver_id, season, int(calendar_row.round))
            ),
            reverse=True,
        )
        qualifying_map = {driver_id: index + 1 for index, driver_id in enumerate(ordering)}
        configs.append(
            RaceConfig(
                drivers=ordering,
                qualifying_positions=qualifying_map,
                circuit_id=str(calendar_row.circuit_id),
                season=season,
                round=int(calendar_row.round),
                weather_mode=weather_mode,
                n_simulations=1,
            )
        )
    return configs


def _normalize_current_standings(
    remaining_races: list[RaceConfig],
    current_standings: Any | None,
    base_race_frames: list[pd.DataFrame] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """Normalize current standings input into point dictionaries."""

    driver_to_constructor: dict[str, str] = {}
    race_frames = base_race_frames or [_build_base_frame(config) for config in remaining_races]
    for base_frame in race_frames:
        driver_to_constructor.update(base_frame.set_index("driver_id")["constructor_id"].to_dict())

    driver_points = {driver_id: 0.0 for driver_id in driver_to_constructor}
    constructor_points = {constructor_id: 0.0 for constructor_id in set(driver_to_constructor.values())}

    if current_standings is None:
        return driver_points, constructor_points, driver_to_constructor

    if isinstance(current_standings, dict):
        driver_source = current_standings.get("drivers", {})
        constructor_source = current_standings.get("constructors", {})
        if isinstance(driver_source, pd.DataFrame):
            for row in driver_source.itertuples(index=False):
                driver_points[str(row.driver_id)] = float(row.points)
                if hasattr(row, "constructor_id") and pd.notna(row.constructor_id):
                    driver_to_constructor[str(row.driver_id)] = str(row.constructor_id)
        else:
            for driver_id, points in driver_source.items():
                driver_points[str(driver_id)] = float(points)
        if isinstance(constructor_source, pd.DataFrame):
            for row in constructor_source.itertuples(index=False):
                constructor_points[str(row.constructor_id)] = float(row.points)
        else:
            for constructor_id, points in constructor_source.items():
                constructor_points[str(constructor_id)] = float(points)
        if isinstance(driver_source, pd.DataFrame) and not isinstance(constructor_source, pd.DataFrame):
            for driver_id, constructor_id in driver_to_constructor.items():
                constructor_points[constructor_id] = constructor_points.get(constructor_id, 0.0) + driver_points.get(driver_id, 0.0)
        return driver_points, constructor_points, driver_to_constructor

    if isinstance(current_standings, pd.DataFrame):
        standings_df = current_standings.copy()
        if {"driver_id", "points"}.issubset(standings_df.columns):
            for row in standings_df.itertuples(index=False):
                driver_points[str(row.driver_id)] = float(row.points)
                if hasattr(row, "constructor_id") and pd.notna(row.constructor_id):
                    driver_to_constructor[str(row.driver_id)] = str(row.constructor_id)
        for driver_id, constructor_id in driver_to_constructor.items():
            constructor_points[constructor_id] = constructor_points.get(constructor_id, 0.0) + driver_points.get(driver_id, 0.0)
        return driver_points, constructor_points, driver_to_constructor

    raise TypeError("Unsupported current standings format")


def _season_champion(
    points: dict[str, float],
    wins: dict[str, int],
    podiums: dict[str, int],
) -> str:
    """Break championship ties using wins, then podiums, then lexical order."""

    ranked = sorted(
        points.keys(),
        key=lambda key: (points[key], wins.get(key, 0), podiums.get(key, 0), key),
        reverse=True,
    )
    return ranked[0]


def simulate_season(
    remaining_races: list[RaceConfig] | list[dict[str, Any]],
    current_standings: Any | None = None,
    n_simulations: int = CONFIG["default_iterations"],
    random_seed: int = CONFIG["random_seed"],
) -> SeasonSimulationResult:
    """Simulate a championship over the remaining season schedule."""

    race_configs = [race if isinstance(race, RaceConfig) else RaceConfig(**race) for race in remaining_races]
    base_race_frames = [_build_base_frame(config) for config in race_configs]
    driver_points_start, constructor_points_start, driver_to_constructor = _normalize_current_standings(
        race_configs,
        current_standings,
        base_race_frames=base_race_frames,
    )

    driver_ids = sorted(driver_points_start.keys())
    constructor_ids = sorted(set(driver_to_constructor.values()))
    driver_points_samples: defaultdict[str, list[float]] = defaultdict(list)
    driver_position_samples: defaultdict[str, list[int]] = defaultdict(list)
    constructor_points_samples: defaultdict[str, list[float]] = defaultdict(list)
    wdc_counts: defaultdict[str, int] = defaultdict(int)
    wcc_counts: defaultdict[str, int] = defaultdict(int)
    evolution_counts: defaultdict[tuple[int, str], int] = defaultdict(int)

    rng = np.random.default_rng(random_seed)

    for _ in range(n_simulations):
        driver_points = driver_points_start.copy()
        constructor_points = constructor_points_start.copy()
        driver_wins: defaultdict[str, int] = defaultdict(int)
        driver_podiums: defaultdict[str, int] = defaultdict(int)
        constructor_wins: defaultdict[str, int] = defaultdict(int)

        for race_index, base_frame in enumerate(base_race_frames):
            order, points = _simulate_single_race(base_frame, rng)
            for position, driver_id in enumerate(order, start=1):
                constructor_id = driver_to_constructor[driver_id]
                scored_points = float(points.get(driver_id, 0.0))
                driver_points[driver_id] = driver_points.get(driver_id, 0.0) + scored_points
                constructor_points[constructor_id] = constructor_points.get(constructor_id, 0.0) + scored_points
                if position == 1:
                    driver_wins[driver_id] += 1
                    constructor_wins[constructor_id] += 1
                if position <= 3:
                    driver_podiums[driver_id] += 1

            leader = _season_champion(driver_points, driver_wins, driver_podiums)
            evolution_counts[(race_index + 1, leader)] += 1

        driver_champion = _season_champion(driver_points, driver_wins, driver_podiums)
        constructor_champion = _season_champion(constructor_points, constructor_wins, {})
        wdc_counts[driver_champion] += 1
        wcc_counts[constructor_champion] += 1

        ranked_drivers = sorted(
            driver_points,
            key=lambda key: (driver_points[key], driver_wins.get(key, 0), driver_podiums.get(key, 0), key),
            reverse=True,
        )

        for position, driver_id in enumerate(ranked_drivers, start=1):
            driver_points_samples[driver_id].append(driver_points[driver_id])
            driver_position_samples[driver_id].append(position)

        for constructor_id in constructor_ids:
            constructor_points_samples[constructor_id].append(constructor_points.get(constructor_id, 0.0))

    wdc_probs = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "wdc_probability": [wdc_counts[driver_id] / n_simulations for driver_id in driver_ids],
        }
    ).sort_values("wdc_probability", ascending=False)

    wcc_probs = pd.DataFrame(
        {
            "constructor_id": constructor_ids,
            "wcc_probability": [wcc_counts[constructor_id] / n_simulations for constructor_id in constructor_ids],
        }
    ).sort_values("wcc_probability", ascending=False)

    expected_final_points = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "expected_final_points": [float(np.mean(driver_points_samples[driver_id])) for driver_id in driver_ids],
        }
    ).sort_values("expected_final_points", ascending=False)

    expected_final_positions = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "expected_final_position": [float(np.mean(driver_position_samples[driver_id])) for driver_id in driver_ids],
        }
    ).sort_values("expected_final_position")

    evolution_rows: list[dict[str, Any]] = []
    for race_index, config in enumerate(race_configs, start=1):
        for driver_id in driver_ids:
            evolution_rows.append(
                {
                    "season": config.season,
                    "round": config.round,
                    "race_index": race_index,
                    "driver_id": driver_id,
                    "championship_probability": evolution_counts[(race_index, driver_id)] / n_simulations,
                }
            )
    probability_evolution = pd.DataFrame(evolution_rows)

    current_points_frame = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "constructor_id": [driver_to_constructor.get(driver_id, "unknown") for driver_id in driver_ids],
            "current_points": [driver_points_start.get(driver_id, 0.0) for driver_id in driver_ids],
        }
    )
    projected_standings = (
        current_points_frame.merge(expected_final_points, how="left", on="driver_id")
        .merge(expected_final_positions, how="left", on="driver_id")
        .merge(wdc_probs, how="left", on="driver_id")
        .sort_values("expected_final_points", ascending=False)
        .reset_index(drop=True)
    )

    LOGGER.info(
        "Season simulation complete for %s races with %s iterations",
        len(race_configs),
        n_simulations,
    )

    return SeasonSimulationResult(
        wdc_probs=wdc_probs.reset_index(drop=True),
        wcc_probs=wcc_probs.reset_index(drop=True),
        expected_final_points=expected_final_points.reset_index(drop=True),
        expected_final_positions=expected_final_positions.reset_index(drop=True),
        probability_evolution=probability_evolution,
        projected_standings=projected_standings,
        simulation_runs=n_simulations,
    )


class SeasonSimulator:
    """Compatibility wrapper for high-level season simulations."""

    def simulate_race(self, race_config: dict[str, Any]) -> dict[str, float]:
        race = race_config if isinstance(race_config, dict) else dict(race_config)
        drivers = list(race.get("drivers", []))
        if not drivers:
            return {}
        equal_prob = 1.0 / len(drivers)
        return {driver_id: equal_prob for driver_id in drivers}

    def simulate_season(
        self,
        current_standings: pd.DataFrame,
        remaining_races: list[dict[str, Any]],
        n_simulations: int = CONFIG["default_iterations"],
    ) -> dict[str, Any]:
        standings = current_standings.copy()
        if standings.empty:
            return {"wdc_probabilities": {}, "n_simulations": int(n_simulations)}

        aggregate: dict[str, float] = {
            str(row.driver_id): float(getattr(row, "points", 0.0))
            for row in standings.itertuples(index=False)
        }
        for race in remaining_races:
            race_probs = self.simulate_race(race)
            for driver_id, probability in race_probs.items():
                aggregate[str(driver_id)] = aggregate.get(str(driver_id), 0.0) + float(probability)

        total = float(sum(aggregate.values()))
        if total <= 0:
            total = float(len(aggregate))
        normalized = {driver_id: value / total for driver_id, value in aggregate.items()}
        return {"wdc_probabilities": normalized, "n_simulations": int(n_simulations)}
