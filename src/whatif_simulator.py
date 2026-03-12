"""Counterfactual driver/team and regulation scenario simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from src.car_performance import get_cpi, predict_end_of_season_cpi
from src.data_pipeline import load_calendar
from src.feature_engineering import build_race_inference_frame
from src.model_training import predict_from_frame
from src.race_simulator import _simulate_race_matrices
from src.registry import get_constructor_name, get_driver_name
from src.true_skill import get_tsi
from src.utils import CONFIG as BASE_CONFIG, load_dataframe

CONFIG: dict[str, Any] = {
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "random_seed": 42,
}


@dataclass(slots=True)
class WhatIfResult:
    """Result of a driver/team swap simulation."""

    driver_id: str
    target_constructor_id: str
    season: int
    replaced_driver_id: str
    expected_points: float
    expected_wins: float
    championship_probability: float
    actual_points: float
    actual_wins: int
    actual_championship: bool
    comparison_table: pd.DataFrame


@dataclass(slots=True)
class RegScenarioResult:
    """Result of a regulation-adaptation scenario."""

    constructor_id: str
    regulation_year: int
    adaptation_scenario: str
    projected_cpi: float
    confidence_interval: tuple[float, float]
    narrative: str


@dataclass(slots=True)
class ChampionshipScenarioResult:
    """Result of a championship replay scenario."""

    season: int
    champion: str
    standings: pd.DataFrame
    narrative: str


class WhatIfSimulator:
    """Run counterfactual season scenarios using the trained race model."""

    def __init__(self, race_results: pd.DataFrame | None = None, random_seed: int = CONFIG["random_seed"]) -> None:
        self.race_results = race_results if race_results is not None else load_dataframe(CONFIG["race_results_path"])
        self.random_seed = int(random_seed)

    def _season_results(self, season: int) -> pd.DataFrame:
        """Return season results sorted by round and grid."""

        frame = self.race_results.loc[self.race_results["season"] == int(season)].copy()
        frame["qualifying_position"] = pd.to_numeric(frame["qualifying_position"], errors="coerce")
        return frame.sort_values(["round", "qualifying_position", "driver_id"]).reset_index(drop=True)

    @staticmethod
    def _normalize_driver_token(value: str) -> str:
        """Normalize user-facing driver identifiers to a stable comparison token."""

        normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
        return normalized.strip().lower().replace("-", "_").replace(" ", "_")

    def _resolve_driver_id(self, driver_id: str, season: int) -> str:
        """Resolve aliases such as lewis_hamilton to the canonical season driver_id."""

        season_results = self._season_results(season)
        requested = self._normalize_driver_token(driver_id)
        if requested in set(season_results["driver_id"].astype(str)):
            return requested

        candidate_map: dict[str, str] = {}
        for row in season_results[
            ["driver_id", "driver_given_name", "driver_family_name", "driver_code"]
        ].drop_duplicates().itertuples(index=False):
            canonical = str(row.driver_id)
            variants = {
                canonical,
                self._normalize_driver_token(canonical),
                self._normalize_driver_token(row.driver_family_name),
                self._normalize_driver_token(f"{row.driver_given_name}_{row.driver_family_name}"),
                self._normalize_driver_token(f"{row.driver_given_name}{row.driver_family_name}"),
                self._normalize_driver_token(row.driver_code),
            }
            for variant in variants:
                if variant:
                    candidate_map[variant] = canonical
        return candidate_map.get(requested, str(driver_id))

    def _target_seat(self, constructor_id: str, season: int) -> str:
        """Choose the seat to replace for a constructor-season."""

        season_results = self._season_results(season)
        team_rows = season_results.loc[season_results["constructor_id"] == str(constructor_id)].copy()
        if team_rows.empty:
            raise ValueError(f"No results found for constructor_id={constructor_id} season={season}")
        points = team_rows.groupby("driver_id", as_index=False)["points"].sum().sort_values(["points", "driver_id"])
        return str(points.iloc[0]["driver_id"])

    def _swap_frame(
        self,
        base_frame: pd.DataFrame,
        simulated_driver_id: str,
        replaced_driver_id: str,
        target_constructor_id: str,
        season: int,
        round_number: int,
    ) -> pd.DataFrame:
        """Swap or replace driver-specific skill signals within a race frame."""

        frame = base_frame.copy()
        target_index = frame.index[frame["driver_id"] == str(replaced_driver_id)]
        if len(target_index) == 0:
            return frame
        target_index = int(target_index[0])

        simulated_index_values = frame.index[frame["driver_id"] == str(simulated_driver_id)]
        simulated_index = int(simulated_index_values[0]) if len(simulated_index_values) else None

        target_constructor_name = get_constructor_name(target_constructor_id)
        simulated_tsi = get_tsi(simulated_driver_id, season=int(season), round_number=int(round_number))
        replaced_tsi = get_tsi(replaced_driver_id, season=int(season), round_number=int(round_number))

        target_snapshot = frame.loc[target_index].copy()
        if simulated_index is not None:
            simulated_snapshot = frame.loc[simulated_index].copy()
            frame.loc[target_index, ["driver_id", "driver_name"]] = [str(simulated_driver_id), get_driver_name(simulated_driver_id)]
            frame.loc[simulated_index, ["driver_id", "driver_name"]] = [str(replaced_driver_id), get_driver_name(replaced_driver_id)]
            for column in [
                "driver_elo",
                "driver_wet_elo",
                "wet_elo_vs_field",
                "avg_finish_last_5",
                "avg_qualifying_last_5",
                "win_rate_career",
                "podium_rate_career",
                "dnf_rate_last_10",
                "laps_led_last_3",
                "consistency_score",
                "tsi_qualifying",
                "tsi_race",
                "tsi_composite",
                "tsi_rolling_20",
                "tsi_delta",
                "tsi_wet",
                "tsi_dry",
            ]:
                if column in frame.columns:
                    frame.loc[target_index, column] = simulated_snapshot[column]
                    frame.loc[simulated_index, column] = target_snapshot[column]
        else:
            frame.loc[target_index, "driver_id"] = str(simulated_driver_id)
            frame.loc[target_index, "driver_name"] = get_driver_name(simulated_driver_id)

        for prefix, snapshot in [("target", simulated_tsi), ("replaced", replaced_tsi)]:
            if prefix == "target":
                idx = target_index
            elif simulated_index is None:
                continue
            else:
                idx = simulated_index
            if "tsi_qualifying" in frame.columns:
                frame.loc[idx, "tsi_qualifying"] = snapshot.get("tsi_qualifying", frame.loc[idx, "tsi_qualifying"])
            if "tsi_race" in frame.columns:
                frame.loc[idx, "tsi_race"] = snapshot.get("tsi_race", frame.loc[idx, "tsi_race"])
            if "tsi_composite" in frame.columns:
                frame.loc[idx, "tsi_composite"] = snapshot.get("tsi_composite", frame.loc[idx, "tsi_composite"])
            if "tsi_rolling_20" in frame.columns:
                frame.loc[idx, "tsi_rolling_20"] = snapshot.get("tsi_rolling_20", frame.loc[idx, "tsi_rolling_20"])
            if "tsi_delta" in frame.columns:
                frame.loc[idx, "tsi_delta"] = snapshot.get("tsi_delta", frame.loc[idx, "tsi_delta"])
            if "tsi_wet" in frame.columns:
                frame.loc[idx, "tsi_wet"] = snapshot.get("tsi_wet", frame.loc[idx, "tsi_wet"])
            if "tsi_dry" in frame.columns:
                frame.loc[idx, "tsi_dry"] = snapshot.get("tsi_dry", frame.loc[idx, "tsi_dry"])

        frame.loc[target_index, ["constructor_id", "constructor_name"]] = [str(target_constructor_id), target_constructor_name]
        qual_delta = float(replaced_tsi.get("tsi_qualifying", 0.0) - simulated_tsi.get("tsi_qualifying", 0.0))
        if not np.isfinite(qual_delta):
            qual_delta = 0.0
        fallback_grid = pd.Series(np.arange(1, len(frame) + 1, dtype=float), index=frame.index)
        frame["whatif_quali_sort"] = (
            pd.to_numeric(frame.get("qualifying_position"), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fallback_grid)
        )
        frame.loc[target_index, "whatif_quali_sort"] = frame.loc[target_index, "whatif_quali_sort"] - 3.0 * qual_delta
        ranked_grid = frame["whatif_quali_sort"].replace([np.inf, -np.inf], np.nan).fillna(fallback_grid).rank(method="first")
        frame["qualifying_position"] = ranked_grid.astype(int)
        frame = frame.drop(columns=["whatif_quali_sort"], errors="ignore")
        return frame.sort_values("qualifying_position").reset_index(drop=True)

    def _base_frame_for_round(self, race_frame: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
        """Build the model feature frame for an actual race weekend."""

        driver_ids = race_frame.sort_values("qualifying_position")["driver_id"].tolist()
        qualifying_map = {
            row.driver_id: int(row.qualifying_position)
            for row in race_frame[["driver_id", "qualifying_position"]].itertuples(index=False)
        }
        return build_race_inference_frame(
            drivers_list=driver_ids,
            qualifying_positions=qualifying_map,
            circuit_id=str(race_frame["circuit_id"].iloc[0]),
            season=int(season),
            round_number=int(round_number),
            weather_mode=str(race_frame["race_condition"].iloc[0]) if "race_condition" in race_frame.columns else None,
        )

    def _simulation_base_frame(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Convert a prepared feature frame into a race-simulation base frame."""

        predictions = predict_from_frame(feature_frame)
        merge_columns = [
            "driver_id",
            "driver_name",
            "constructor_id",
            "constructor_name",
            "qualifying_position",
            "driver_elo",
            "driver_wet_elo",
            "team_reliability_last_10",
            "expected_finish",
            "dnf_prob",
            "relative_win_probability",
        ]
        base = predictions.merge(
            feature_frame.drop(columns=[column for column in predictions.columns if column in feature_frame.columns and column != "driver_id"]),
            how="left",
            on="driver_id",
        )
        base = base.loc[:, ~base.columns.duplicated()].copy()
        for column in merge_columns:
            if column not in base.columns and column in feature_frame.columns:
                base[column] = feature_frame[column]
        base["relative_win_probability"] = pd.to_numeric(base["relative_win_probability"], errors="coerce").fillna(0.0)
        base["dnf_prob"] = pd.to_numeric(base["dnf_prob"], errors="coerce").fillna(0.05)
        base["team_reliability_last_10"] = pd.to_numeric(base["team_reliability_last_10"], errors="coerce").fillna(0.85)
        return base.sort_values("qualifying_position").reset_index(drop=True)

    def simulate_driver_swap(
        self,
        driver_id: str,
        target_constructor_id: str,
        season: int,
        n_simulations: int = 10000,
    ) -> WhatIfResult:
        """Simulate a full season if a driver raced for a target constructor."""

        resolved_driver_id = self._resolve_driver_id(driver_id, season=int(season))
        season_results = self._season_results(season)
        replaced_driver_id = self._target_seat(constructor_id=target_constructor_id, season=season)
        actual_driver_points = (
            season_results.loc[season_results["driver_id"] == str(resolved_driver_id), "points"].sum()
        )
        actual_driver_wins = int(
            season_results.loc[
                (season_results["driver_id"] == str(resolved_driver_id))
                & pd.to_numeric(season_results["finish_position"], errors="coerce").eq(1),
                "round",
            ].nunique()
        )
        actual_standings = (
            season_results.groupby("driver_id", as_index=False)["points"]
            .sum()
            .sort_values(["points", "driver_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        actual_champion = bool(
            not actual_standings.empty and str(actual_standings.iloc[0]["driver_id"]) == str(resolved_driver_id)
        )

        race_frames: list[pd.DataFrame] = []
        for round_number, race_frame in season_results.groupby("round", sort=True):
            base_frame = self._base_frame_for_round(race_frame, season=season, round_number=int(round_number))
            swapped = self._swap_frame(
                base_frame=base_frame,
                simulated_driver_id=str(resolved_driver_id),
                replaced_driver_id=str(replaced_driver_id),
                target_constructor_id=str(target_constructor_id),
                season=int(season),
                round_number=int(round_number),
            )
            race_frames.append(self._simulation_base_frame(swapped))

        rng = np.random.default_rng(self.random_seed)
        season_driver_ids = sorted(
            {
                str(simulated_driver_id)
                for frame in race_frames
                for simulated_driver_id in frame["driver_id"].astype(str).tolist()
            }
        )
        if str(resolved_driver_id) not in season_driver_ids:
            season_driver_ids.append(str(resolved_driver_id))
        season_driver_lookup = {driver_key: idx for idx, driver_key in enumerate(season_driver_ids)}
        simulated_driver_index = season_driver_lookup[str(resolved_driver_id)]

        season_points = np.zeros((int(n_simulations), len(season_driver_ids)), dtype=float)
        season_wins = np.zeros((int(n_simulations), len(season_driver_ids)), dtype=np.int16)

        for base_frame in race_frames:
            finishing_order, _, points_matrix, _ = _simulate_race_matrices(
                base_frame=base_frame,
                n_simulations=int(n_simulations),
                rng=rng,
            )
            race_driver_indices = np.array(
                [season_driver_lookup[str(driver_key)] for driver_key in base_frame["driver_id"].astype(str).tolist()],
                dtype=int,
            )
            season_points[:, race_driver_indices] += points_matrix
            race_winners = race_driver_indices[finishing_order[:, 0]]
            season_wins[np.arange(int(n_simulations)), race_winners] += 1

        tie_break = np.arange(len(season_driver_ids), dtype=float) / max(float(len(season_driver_ids)), 1.0)
        ranking_score = season_points * 10000.0 + season_wins.astype(float) * 100.0 + tie_break.reshape(1, -1)
        champion_indices = np.argmax(ranking_score, axis=1)
        championship_wins = int(np.sum(champion_indices == simulated_driver_index))

        season_ranking = np.argsort(-ranking_score, axis=1)
        season_rank_positions = np.empty_like(season_ranking)
        season_rank_positions[np.arange(int(n_simulations))[:, None], season_ranking] = np.arange(len(season_driver_ids))

        sample_rows = min(int(n_simulations), 500)
        comparison_table = pd.DataFrame(
            {
                "driver_id": str(resolved_driver_id),
                "points": season_points[:sample_rows, simulated_driver_index],
                "wins": season_wins[:sample_rows, simulated_driver_index],
                "rank": season_rank_positions[:sample_rows, simulated_driver_index] + 1,
            }
        )
        return WhatIfResult(
            driver_id=str(resolved_driver_id),
            target_constructor_id=str(target_constructor_id),
            season=int(season),
            replaced_driver_id=str(replaced_driver_id),
            expected_points=float(np.mean(season_points[:, simulated_driver_index])),
            expected_wins=float(np.mean(season_wins[:, simulated_driver_index])),
            championship_probability=float(championship_wins / max(int(n_simulations), 1)),
            actual_points=float(actual_driver_points),
            actual_wins=int(actual_driver_wins),
            actual_championship=actual_champion,
            comparison_table=comparison_table,
        )

    def simulate_regulation_scenario(
        self,
        constructor_id: str,
        regulation_year: int,
        adaptation_scenario: str,
    ) -> RegScenarioResult:
        """Project a constructor's regulation adaptation outcome."""

        projection = predict_end_of_season_cpi(constructor_id=constructor_id, season=int(regulation_year))
        base_cpi = float(projection.get("projected_cpi", 0.0) if np.isfinite(projection.get("projected_cpi", np.nan)) else 0.0)
        baseline = get_cpi(constructor_id=constructor_id, season=int(regulation_year), round_number=5)
        adaptability = float(
            baseline.get("cpi_regulation_adaptability", np.nan)
            if np.isfinite(baseline.get("cpi_regulation_adaptability", np.nan))
            else base_cpi
        )
        sigma = max(abs(adaptability) * 0.25, 0.05)
        scenario_map = {
            "best_case": adaptability + sigma,
            "base": adaptability,
            "worst_case": adaptability - sigma,
        }
        projected_cpi = float(scenario_map.get(str(adaptation_scenario), adaptability))
        interval = (projected_cpi - 1.96 * sigma, projected_cpi + 1.96 * sigma)
        narrative = (
            f"{get_constructor_name(constructor_id)} in {regulation_year} under a {adaptation_scenario.replace('_', ' ')} scenario "
            f"projects to a CPI of {projected_cpi:.3f} based on a historical regulation-adaptation baseline of {adaptability:.3f}."
        )
        return RegScenarioResult(
            constructor_id=str(constructor_id),
            regulation_year=int(regulation_year),
            adaptation_scenario=str(adaptation_scenario),
            projected_cpi=projected_cpi,
            confidence_interval=(float(interval[0]), float(interval[1])),
            narrative=narrative,
        )

    def simulate_championship_scenario(
        self,
        season: int,
        modifications: dict[str, Any],
    ) -> ChampionshipScenarioResult:
        """Replay a championship with simple result overrides and driver swaps."""

        season_results = self._season_results(season)
        modified = season_results.copy()

        for override in modifications.get("dnf_overrides", []):
            mask = (
                modified["round"].eq(int(override["round"]))
                & modified["driver_id"].eq(str(override["driver_id"]))
            )
            if not mask.any():
                continue
            if "new_finish_position" in override:
                modified.loc[mask, "finish_position"] = float(override["new_finish_position"])
            if "points" in override:
                modified.loc[mask, "points"] = float(override["points"])
            modified.loc[mask, "dnf_flag"] = 0
            modified.loc[mask, "dnf_reason"] = "finished"

        standings = (
            modified.groupby(["driver_id", "constructor_id"], as_index=False)["points"]
            .sum()
            .sort_values(["points", "driver_id"], ascending=[False, True])
            .reset_index(drop=True)
        )
        standings["position"] = np.arange(1, len(standings) + 1)
        champion = str(standings.iloc[0]["driver_id"]) if not standings.empty else "unknown"
        narrative = f"{get_driver_name(champion)} leads the replayed {season} standings after applying {len(modifications.get('dnf_overrides', []))} explicit override(s)."

        if modifications.get("driver_swaps"):
            first_swap = modifications["driver_swaps"][0]
            swap_result = self.simulate_driver_swap(
                driver_id=str(first_swap["driver_id"]),
                target_constructor_id=str(first_swap["target_constructor_id"]),
                season=int(season),
                n_simulations=int(first_swap.get("n_simulations", 3000)),
            )
            narrative += (
                f" A companion swap simulation projects {get_driver_name(first_swap['driver_id'])} to score "
                f"{swap_result.expected_points:.1f} points with {get_constructor_name(first_swap['target_constructor_id'])}."
            )

        return ChampionshipScenarioResult(
            season=int(season),
            champion=champion,
            standings=standings,
            narrative=narrative,
        )
