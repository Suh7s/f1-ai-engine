"""Live championship state tracking and scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline import get_current_season, load_calendar
from src.season_simulator import build_remaining_races_for_season, simulate_season
from src.utils import CONFIG as BASE_CONFIG, get_gp_name, load_dataframe

CONFIG: dict[str, Any] = {
    "processed_dir": BASE_CONFIG["processed_dir"],
    "driver_standings_path": BASE_CONFIG["processed_dir"] / "driver_standings.csv",
    "constructor_standings_path": BASE_CONFIG["processed_dir"] / "constructor_standings.csv",
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "championship_history_path": BASE_CONFIG["processed_dir"] / "championship_history.csv",
}


@dataclass(slots=True)
class ChampionshipScenario:
    """Human-readable title-contender scenario summary."""

    mathematically_eliminated: bool
    max_possible_points: int
    required_results: str


@dataclass(slots=True)
class ChampionshipHistoryEntry:
    """Season-end driver and constructor title summary."""

    season: int
    driver_champion: str
    constructor_champion: str
    margin_of_victory: float
    decisive_race: str


def _decisive_race_name(driver_standings: pd.DataFrame, season: int) -> str:
    """Estimate the first round where the eventual champion became uncatchable."""

    season_frame = driver_standings.loc[driver_standings["season"] == int(season)].copy()
    if season_frame.empty:
        return "Unknown"
    max_round = int(season_frame["round"].max())
    for round_number in sorted(season_frame["round"].dropna().astype(int).unique().tolist()):
        current = season_frame.loc[season_frame["round"] == round_number].sort_values("points", ascending=False).reset_index(drop=True)
        if len(current) < 2:
            continue
        leader = float(current.iloc[0]["points"])
        challenger = float(current.iloc[1]["points"])
        remaining = max(max_round - int(round_number), 0)
        if leader > challenger + remaining * 26:
            return get_gp_name(season, round_number)
    return get_gp_name(season, max_round)


def build_championship_history(
    driver_standings: pd.DataFrame | None = None,
    constructor_standings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a season-level WDC/WCC history table from cached standings."""

    drivers = driver_standings if driver_standings is not None else load_dataframe(CONFIG["driver_standings_path"])
    constructors = constructor_standings if constructor_standings is not None else load_dataframe(CONFIG["constructor_standings_path"])
    if drivers.empty or constructors.empty:
        return pd.DataFrame(columns=["season", "driver_champion", "constructor_champion", "margin_of_victory", "decisive_race"])

    rows: list[dict[str, Any]] = []
    for season in sorted(set(drivers["season"].astype(int).tolist()) & set(constructors["season"].astype(int).tolist())):
        driver_final = drivers.loc[drivers["season"] == season].copy()
        constructor_final = constructors.loc[constructors["season"] == season].copy()
        if driver_final.empty or constructor_final.empty:
            continue
        driver_round = int(driver_final["round"].max())
        constructor_round = int(constructor_final["round"].max())
        driver_final = driver_final.loc[driver_final["round"] == driver_round].sort_values("points", ascending=False).reset_index(drop=True)
        constructor_final = constructor_final.loc[constructor_final["round"] == constructor_round].sort_values("points", ascending=False).reset_index(drop=True)
        if driver_final.empty or constructor_final.empty:
            continue
        champion_margin = float(driver_final.iloc[0]["points"]) - float(driver_final.iloc[1]["points"]) if len(driver_final) > 1 else float(driver_final.iloc[0]["points"])
        rows.append(
            {
                "season": int(season),
                "driver_champion": str(driver_final.iloc[0]["driver_id"]),
                "constructor_champion": str(constructor_final.iloc[0]["constructor_id"]),
                "margin_of_victory": champion_margin,
                "decisive_race": _decisive_race_name(drivers, season),
            }
        )
    history = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    if not history.empty:
        history.to_csv(CONFIG["championship_history_path"], index=False)
    return history


def load_championship_history() -> pd.DataFrame:
    """Load or build the cached championship-history table."""

    if CONFIG["championship_history_path"].exists():
        return load_dataframe(CONFIG["championship_history_path"])
    return build_championship_history()


CHAMPIONSHIP_HISTORY = load_championship_history()


class ChampionshipTracker:
    """Maintain live championship state for a season."""

    def __init__(self, season: int | None = None) -> None:
        self.season = int(season or get_current_season())
        self.driver_standings = load_dataframe(CONFIG["driver_standings_path"])
        self.constructor_standings = load_dataframe(CONFIG["constructor_standings_path"])
        self.race_results = load_dataframe(CONFIG["race_results_path"])
        self.calendar = load_calendar(self.season, allow_fetch=False)

    def _season_driver_standings(self) -> pd.DataFrame:
        """Return the latest driver standings snapshot for the season."""

        frame = self.driver_standings.loc[self.driver_standings["season"] == self.season].copy()
        if frame.empty:
            current = (
                self.race_results.loc[self.race_results["season"] == self.season]
                .groupby(["driver_id", "constructor_id"], as_index=False)["points"]
                .sum()
            )
            current["position"] = current["points"].rank(method="dense", ascending=False).astype(int)
            return current.sort_values(["position", "driver_id"]).reset_index(drop=True)

        latest_round = int(frame["round"].max())
        frame = frame.loc[frame["round"] == latest_round].copy()
        return frame.sort_values(["position", "driver_id"]).reset_index(drop=True)

    def _season_constructor_standings(self) -> pd.DataFrame:
        """Return the latest constructor standings snapshot for the season."""

        frame = self.constructor_standings.loc[self.constructor_standings["season"] == self.season].copy()
        if frame.empty:
            results = self.race_results.loc[self.race_results["season"] == self.season].copy()
            current = results.groupby("constructor_id", as_index=False)["points"].sum()
            current["position"] = current["points"].rank(method="dense", ascending=False).astype(int)
            return current.sort_values(["position", "constructor_id"]).reset_index(drop=True)

        latest_round = int(frame["round"].max())
        frame = frame.loc[frame["round"] == latest_round].copy()
        return frame.sort_values(["position", "constructor_id"]).reset_index(drop=True)

    def get_current_standings(self, season: int | None = None) -> pd.DataFrame:
        """Return the latest driver standings snapshot for the requested season."""

        if season is None or int(season) == self.season:
            return self._season_driver_standings()
        return ChampionshipTracker(int(season))._season_driver_standings()

    def _current_round(self) -> int:
        """Return the latest completed round for the season."""

        standings = self._season_driver_standings()
        if "round" in standings.columns and not standings.empty:
            return int(standings["round"].max())
        if not self.calendar.empty and "is_completed" in self.calendar.columns:
            completed = self.calendar.loc[self.calendar["is_completed"] == 1]
            if not completed.empty:
                return int(completed["round"].max())
        return 0

    def _current_driver_points(self) -> pd.DataFrame:
        """Return current driver points with constructor mapping."""

        frame = self._season_driver_standings().copy()
        if "points" not in frame.columns:
            frame["points"] = 0.0
        if "constructor_id" not in frame.columns:
            constructor_lookup = (
                self.race_results.loc[self.race_results["season"] == self.season, ["driver_id", "constructor_id"]]
                .drop_duplicates("driver_id")
                .set_index("driver_id")["constructor_id"]
                .to_dict()
            )
            frame["constructor_id"] = frame["driver_id"].map(constructor_lookup)
        return frame[["driver_id", "constructor_id", "points"]].copy()

    def update(self, race_results: pd.DataFrame) -> None:
        """Update local state from a new race-results frame."""

        updated = self.race_results.loc[self.race_results["season"] != self.season].copy()
        combined = pd.concat([updated, race_results], ignore_index=True).drop_duplicates(
            subset=["season", "round", "driver_id"], keep="last"
        )
        self.race_results = combined.sort_values(["season", "round", "finish_position"]).reset_index(drop=True)

    def get_wdc_probability(self, n_simulations: int = 10_000) -> pd.DataFrame:
        """Run the season simulator for the remaining calendar."""

        current_round = self._current_round()
        remaining_races = build_remaining_races_for_season(self.season, from_round=current_round)
        current_standings = self._current_driver_points()
        if not remaining_races:
            standings = current_standings.sort_values("points", ascending=False).reset_index(drop=True)
            if standings.empty:
                return pd.DataFrame(columns=["driver_id", "wdc_probability"])
            standings["wdc_probability"] = 0.0
            standings.loc[standings.index[0], "wdc_probability"] = 1.0
            return standings[["driver_id", "wdc_probability"]]

        result = simulate_season(
            remaining_races=remaining_races,
            current_standings=current_standings,
            n_simulations=n_simulations,
        )
        return result.wdc_probs

    def get_championship_scenarios(self, driver_id: str) -> ChampionshipScenario:
        """Describe what a driver needs for the title from the current state."""

        standings = self._current_driver_points().sort_values("points", ascending=False).reset_index(drop=True)
        if standings.empty or driver_id not in standings["driver_id"].tolist():
            return ChampionshipScenario(True, 0, "Driver not present in current standings.")

        driver_points = float(standings.loc[standings["driver_id"] == driver_id, "points"].iloc[0])
        leader_points = float(standings["points"].max())
        current_round = self._current_round()
        remaining = self.calendar.loc[self.calendar["round"] > current_round].copy() if not self.calendar.empty else pd.DataFrame()
        races_remaining = int(len(remaining))
        max_possible_points = int(driver_points + races_remaining * (points_for_win := 26))
        mathematically_eliminated = max_possible_points < leader_points

        if mathematically_eliminated:
            required = "Mathematically eliminated."
        elif races_remaining == 0:
            required = "Season complete."
        else:
            gap = max(0.0, leader_points - driver_points)
            wins_needed = int(np.ceil(gap / points_for_win))
            required = (
                f"Needs roughly {wins_needed} win(s) plus the leader to drop points "
                f"across the remaining {races_remaining} race(s)."
            )

        return ChampionshipScenario(
            mathematically_eliminated=mathematically_eliminated,
            max_possible_points=max_possible_points,
            required_results=required,
        )

    def _wdc_probability_at_round(self, completed_round: int, n_simulations: int = 3000) -> pd.DataFrame:
        """Compute title probabilities from the state after a specific round."""

        standings = self.driver_standings.loc[
            (self.driver_standings["season"] == self.season) & (self.driver_standings["round"] == completed_round)
        ].copy()
        if standings.empty:
            return pd.DataFrame(columns=["driver_id", "wdc_probability"])
        remaining_races = build_remaining_races_for_season(self.season, from_round=completed_round)
        if not remaining_races:
            standings = standings.sort_values("points", ascending=False).reset_index(drop=True)
            standings["wdc_probability"] = 0.0
            standings.loc[standings.index[0], "wdc_probability"] = 1.0
            return standings[["driver_id", "wdc_probability"]]
        result = simulate_season(
            remaining_races=remaining_races,
            current_standings=standings[["driver_id", "constructor_id", "points"]],
            n_simulations=n_simulations,
        )
        return result.wdc_probs

    def get_momentum_index(self, driver_id: str, window: int = 3) -> float:
        """Return the change in championship probability versus `window` races ago."""

        current_round = self._current_round()
        if current_round <= 0:
            return 0.0

        current_probs = self.get_wdc_probability(n_simulations=3000)
        current_prob = float(
            current_probs.loc[current_probs["driver_id"] == driver_id, "wdc_probability"].iloc[0]
        ) if driver_id in current_probs["driver_id"].tolist() else 0.0

        prior_round = max(0, current_round - int(window))
        if prior_round == 0:
            return current_prob

        prior_probs = self._wdc_probability_at_round(prior_round, n_simulations=3000)
        prior_prob = float(
            prior_probs.loc[prior_probs["driver_id"] == driver_id, "wdc_probability"].iloc[0]
        ) if driver_id in prior_probs["driver_id"].tolist() else 0.0
        return current_prob - prior_prob

    def snapshot(self, n_simulations: int = 10_000) -> dict[str, Any]:
        """Return a full season-status snapshot for CLI and dashboard use."""

        standings = self._season_driver_standings()
        constructor_standings = self._season_constructor_standings()
        wdc_probs = self.get_wdc_probability(n_simulations=n_simulations)
        current_round = self._current_round()
        remaining_races = self.calendar.loc[self.calendar["round"] > current_round].copy() if not self.calendar.empty else pd.DataFrame()
        return {
            "season": self.season,
            "current_round": current_round,
            "races_remaining": int(len(remaining_races)),
            "driver_standings": standings,
            "constructor_standings": constructor_standings,
            "wdc_probabilities": wdc_probs,
        }
