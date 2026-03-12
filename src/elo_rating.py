"""Driver and constructor ELO ratings for Formula 1."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.registry import is_rookie
from src.utils import (
    CONFIG as BASE_CONFIG,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_dataframe,
    validate_columns,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "driver_elo_path": BASE_CONFIG["processed_dir"] / "driver_elo_history.csv",
    "constructor_elo_path": BASE_CONFIG["processed_dir"] / "constructor_elo_history.csv",
    "starting_elo": 1500.0,
    "starting_wet_elo": 1500.0,
    "rookie_starting_elo": 1480.0,
    "rookie_k_factor": 48.0,
    "rookie_high_k_races": 20,
    "base_k_factor": 32.0,
    "experienced_k_factor": 24.0,
    "experienced_after_races": 50,
    "wet_k_factor": 48.0,
    "max_gap_weight": 10,
    "min_elo": 1200.0,
    "max_elo": 2800.0,
}

LOGGER = get_logger(__name__)


def _starting_driver_elo(driver_id: str, season: int, first_appearance_season: int, dataset_start_season: int) -> float:
    """Return the starting ELO for a driver entering the tracked dataset."""

    if is_rookie(driver_id, season):
        return float(CONFIG["rookie_starting_elo"])
    if first_appearance_season == season and season > dataset_start_season:
        return float(CONFIG["rookie_starting_elo"])
    return float(CONFIG["starting_elo"])


def _dry_k_factor(race_count: int, rookie_profile: bool = False) -> float:
    """Return the appropriate dry-ELO K-factor for a driver."""

    if rookie_profile and race_count < CONFIG["rookie_high_k_races"]:
        return float(CONFIG["rookie_k_factor"])
    if race_count >= CONFIG["experienced_after_races"]:
        return CONFIG["experienced_k_factor"]
    return CONFIG["base_k_factor"]


def _wet_k_factor() -> float:
    """Return the constant wet-ELO K-factor."""

    return float(CONFIG["wet_k_factor"])


def _pair_weight(position_gap: int, loser_dnf_reason: str) -> float:
    """Compute a pairwise comparison weight using gap and DNF context."""

    gap_weight = 1.0 + min(max(position_gap, 1), CONFIG["max_gap_weight"]) / float(CONFIG["max_gap_weight"])
    dnf_multiplier = {
        "mechanical": 0.60,
        "accident": 1.00,
        "other": 0.80,
        "finished": 1.00,
    }.get(loser_dnf_reason, 0.80)
    return gap_weight * dnf_multiplier


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Return expected pairwise score for player A."""

    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))


def _clip_elo(rating: float) -> float:
    """Constrain Elo to the configured production operating band."""

    return float(min(max(rating, CONFIG["min_elo"]), CONFIG["max_elo"]))


def _assign_race_ranks(race_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a deterministic finishing rank, including ordering DNFs by laps completed."""

    frame = race_df.copy()
    validate_columns(frame, ["finish_position", "laps_completed", "grid_position"], "race ranking frame")
    classified_mask = frame["finish_position"].notna()
    max_classified = int(frame.loc[classified_mask, "finish_position"].max()) if classified_mask.any() else 0

    frame["elo_rank"] = frame["finish_position"]
    dnf_order = (
        frame.loc[~classified_mask]
        .sort_values(["laps_completed", "grid_position"], ascending=[False, True])
        .index.tolist()
    )
    for offset, row_index in enumerate(dnf_order, start=1):
        frame.loc[row_index, "elo_rank"] = max_classified + offset

    frame["elo_rank"] = frame["elo_rank"].rank(method="first").astype(int)
    return frame.sort_values("elo_rank").reset_index(drop=True)


def _race_condition_multipliers(race_condition: str) -> tuple[float, float]:
    """Return dry/wet update weights for a race condition."""

    normalized = str(race_condition or "dry").strip().lower()
    if normalized == "wet":
        return 0.0, 1.0
    if normalized == "mixed":
        return 0.5, 0.5
    return 1.0, 0.0


def _active_driver_rating(dry_elo: float, wet_elo: float, race_condition: str) -> float:
    """Return the condition-appropriate driver rating for constructor aggregation."""

    normalized = str(race_condition or "dry").strip().lower()
    if normalized == "wet":
        return wet_elo
    if normalized == "mixed":
        return 0.5 * dry_elo + 0.5 * wet_elo
    return dry_elo


def calculate_elo_histories(
    race_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute driver dry/wet ELO histories and condition-aware constructor ELO."""

    validate_columns(
        race_results,
        [
            "season",
            "round",
            "circuit_id",
            "driver_id",
            "constructor_id",
            "finish_position",
            "laps_completed",
            "grid_position",
            "dnf_reason",
        ],
        "race results for ELO",
    )

    ratings_dry: defaultdict[str, float] = defaultdict(lambda: CONFIG["starting_elo"])
    ratings_wet: defaultdict[str, float] = defaultdict(lambda: CONFIG["starting_wet_elo"])
    race_counts: defaultdict[str, int] = defaultdict(int)
    rookie_profiles: dict[str, bool] = {}
    driver_records: list[dict[str, Any]] = []
    constructor_records: list[dict[str, Any]] = []

    ordered_results = race_results.sort_values(["season", "round", "finish_position", "grid_position"]).copy()
    if "race_condition" not in ordered_results.columns:
        ordered_results["race_condition"] = "dry"
    dataset_start_season = int(ordered_results["season"].min()) if not ordered_results.empty else 0
    first_appearance_by_driver = (
        ordered_results.groupby("driver_id")["season"].min().astype(int).to_dict()
        if not ordered_results.empty
        else {}
    )

    for (season, round_number), race_df in ordered_results.groupby(["season", "round"], sort=True):
        ranked_race = _assign_race_ranks(race_df)
        race_condition = str(ranked_race["race_condition"].iloc[0]) if "race_condition" in ranked_race.columns else "dry"
        dry_multiplier, wet_multiplier = _race_condition_multipliers(race_condition)
        driver_ids = ranked_race["driver_id"].tolist()

        for driver_id in driver_ids:
            if driver_id not in rookie_profiles:
                starting_elo = _starting_driver_elo(
                    driver_id=driver_id,
                    season=int(season),
                    first_appearance_season=int(first_appearance_by_driver.get(driver_id, season)),
                    dataset_start_season=dataset_start_season,
                )
                ratings_dry[driver_id] = starting_elo
                ratings_wet[driver_id] = starting_elo
                rookie_profiles[driver_id] = bool(starting_elo < CONFIG["starting_elo"])

        dry_before = {driver_id: float(ratings_dry[driver_id]) for driver_id in driver_ids}
        wet_before = {driver_id: float(ratings_wet[driver_id]) for driver_id in driver_ids}
        dry_delta_by_driver = {driver_id: 0.0 for driver_id in driver_ids}
        wet_delta_by_driver = {driver_id: 0.0 for driver_id in driver_ids}

        constructor_before = (
            ranked_race[["constructor_id", "driver_id"]]
            .drop_duplicates()
            .assign(
                elo_before=lambda frame: frame["driver_id"].map(
                    lambda driver_id: _active_driver_rating(
                        dry_before[driver_id],
                        wet_before[driver_id],
                        race_condition,
                    )
                )
            )
            .groupby("constructor_id", as_index=False)["elo_before"]
            .mean()
        )

        ranked_records = ranked_race.to_dict(orient="records")
        for i, winner in enumerate(ranked_records[:-1]):
            for loser in ranked_records[i + 1 :]:
                position_gap = int(loser["elo_rank"] - winner["elo_rank"])
                pair_weight = _pair_weight(position_gap, str(loser.get("dnf_reason", "other")))
                winner_id = winner["driver_id"]
                loser_id = loser["driver_id"]

                if dry_multiplier > 0.0:
                    expected_winner_dry = _expected_score(dry_before[winner_id], dry_before[loser_id])
                    expected_loser_dry = 1.0 - expected_winner_dry
                    dry_delta_by_driver[winner_id] += (
                        _dry_k_factor(race_counts[winner_id], rookie_profiles.get(winner_id, False))
                        * pair_weight
                        * dry_multiplier
                        * (1.0 - expected_winner_dry)
                    )
                    dry_delta_by_driver[loser_id] += (
                        _dry_k_factor(race_counts[loser_id], rookie_profiles.get(loser_id, False))
                        * pair_weight
                        * dry_multiplier
                        * (0.0 - expected_loser_dry)
                    )

                if wet_multiplier > 0.0:
                    expected_winner_wet = _expected_score(wet_before[winner_id], wet_before[loser_id])
                    expected_loser_wet = 1.0 - expected_winner_wet
                    wet_delta_by_driver[winner_id] += (
                        _wet_k_factor() * pair_weight * wet_multiplier * (1.0 - expected_winner_wet)
                    )
                    wet_delta_by_driver[loser_id] += (
                        _wet_k_factor() * pair_weight * wet_multiplier * (0.0 - expected_loser_wet)
                    )

        dry_after = {driver_id: _clip_elo(dry_before[driver_id] + dry_delta_by_driver[driver_id]) for driver_id in driver_ids}
        wet_after = {driver_id: _clip_elo(wet_before[driver_id] + wet_delta_by_driver[driver_id]) for driver_id in driver_ids}

        constructor_after = (
            ranked_race[["constructor_id", "driver_id"]]
            .drop_duplicates()
            .assign(
                elo_after=lambda frame: frame["driver_id"].map(
                    lambda driver_id: _active_driver_rating(
                        dry_after[driver_id],
                        wet_after[driver_id],
                        race_condition,
                    )
                )
            )
            .groupby("constructor_id", as_index=False)["elo_after"]
            .mean()
        )

        race_name = ranked_race["race_name"].iloc[0] if "race_name" in ranked_race.columns else None
        circuit_id = ranked_race["circuit_id"].iloc[0]

        for row in ranked_records:
            driver_id = row["driver_id"]
            constructor_id = row["constructor_id"]
            active_before = _active_driver_rating(dry_before[driver_id], wet_before[driver_id], race_condition)
            active_after = _active_driver_rating(dry_after[driver_id], wet_after[driver_id], race_condition)
            driver_records.append(
                {
                    "driver_id": driver_id,
                    "constructor_id": constructor_id,
                    "season": int(season),
                    "round": int(round_number),
                    "race_name": race_name,
                    "circuit_id": circuit_id,
                    "race_condition": race_condition,
                    "dry_elo_before": float(dry_before[driver_id]),
                    "dry_elo_after": float(dry_after[driver_id]),
                    "dry_elo_delta": float(dry_delta_by_driver[driver_id]),
                    "wet_elo_before": float(wet_before[driver_id]),
                    "wet_elo_after": float(wet_after[driver_id]),
                    "wet_elo_delta": float(wet_delta_by_driver[driver_id]),
                    "elo_before": float(active_before),
                    "elo_after": float(active_after),
                    "elo_delta": float(active_after - active_before),
                    "race_count_before": int(race_counts[driver_id]),
                    "elo_rank": int(row["elo_rank"]),
                }
            )
            ratings_dry[driver_id] = float(dry_after[driver_id])
            ratings_wet[driver_id] = float(wet_after[driver_id])
            race_counts[driver_id] += 1

        constructor_before_map = constructor_before.set_index("constructor_id")["elo_before"].to_dict()
        constructor_after_map = constructor_after.set_index("constructor_id")["elo_after"].to_dict()
        for constructor_id in sorted(ranked_race["constructor_id"].dropna().unique()):
            constructor_records.append(
                {
                    "constructor_id": constructor_id,
                    "season": int(season),
                    "round": int(round_number),
                    "race_name": race_name,
                    "circuit_id": circuit_id,
                    "race_condition": race_condition,
                    "elo_before": float(constructor_before_map.get(constructor_id, CONFIG["starting_elo"])),
                    "elo_after": float(constructor_after_map.get(constructor_id, CONFIG["starting_elo"])),
                    "driver_count": int((ranked_race["constructor_id"] == constructor_id).sum()),
                }
            )

    driver_history = pd.DataFrame(driver_records).sort_values(["season", "round", "elo_rank"])
    constructor_history = pd.DataFrame(constructor_records).sort_values(["season", "round", "constructor_id"])
    validate_columns(
        driver_history,
        ["driver_id", "season", "round", "dry_elo_before", "dry_elo_after", "wet_elo_before", "wet_elo_after"],
        "driver ELO history",
    )
    validate_columns(
        constructor_history,
        ["constructor_id", "season", "round", "elo_before", "elo_after"],
        "constructor ELO history",
    )
    return driver_history.reset_index(drop=True), constructor_history.reset_index(drop=True)


def run_elo_pipeline(race_results: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist driver and constructor ELO histories."""

    ensure_directories()
    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])

    driver_history, constructor_history = calculate_elo_histories(race_results)
    save_dataframe(driver_history, CONFIG["driver_elo_path"])
    save_dataframe(constructor_history, CONFIG["constructor_elo_path"])
    LOGGER.info(
        "Saved driver ELO history (%s rows) and constructor ELO history (%s rows)",
        len(driver_history),
        len(constructor_history),
    )
    return driver_history, constructor_history


def _load_driver_history() -> pd.DataFrame:
    """Load persisted driver ELO history."""

    return load_dataframe(CONFIG["driver_elo_path"])


def _load_constructor_history() -> pd.DataFrame:
    """Load persisted constructor ELO history."""

    return load_dataframe(CONFIG["constructor_elo_path"])


def _driver_elo_at(driver_id: str, season: int, round_number: int, before_column: str, after_column: str) -> float:
    """Return a driver's ELO immediately before a specified race."""

    history = _load_driver_history()
    driver_history = history.loc[history["driver_id"] == driver_id].copy()
    if driver_history.empty:
        if is_rookie(driver_id, season):
            return float(CONFIG["rookie_starting_elo"])
        return float(CONFIG["starting_elo"])

    exact = driver_history.loc[
        (driver_history["season"] == season) & (driver_history["round"] == round_number)
    ]
    if not exact.empty:
        return float(exact.sort_values(["season", "round"]).iloc[-1][before_column])

    prior = driver_history.loc[
        (driver_history["season"] < season)
        | ((driver_history["season"] == season) & (driver_history["round"] < round_number))
    ].sort_values(["season", "round"])
    if prior.empty:
        if is_rookie(driver_id, season):
            return float(CONFIG["rookie_starting_elo"])
        return float(CONFIG["starting_elo"])
    return float(prior.iloc[-1][after_column])


def get_driver_dry_elo(driver_id: str, season: int, round_number: int) -> float:
    """Return a driver's dry ELO immediately before the specified race."""

    return _driver_elo_at(driver_id, season, round_number, "dry_elo_before", "dry_elo_after")


def get_driver_wet_elo(driver_id: str, season: int, round_number: int) -> float:
    """Return a driver's wet ELO immediately before the specified race."""

    return _driver_elo_at(driver_id, season, round_number, "wet_elo_before", "wet_elo_after")


def get_driver_elo(driver_id: str, season: int, round_number: int) -> float:
    """Backward-compatible alias for dry ELO."""

    return get_driver_dry_elo(driver_id, season, round_number)


def get_all_elos_at(season: int, round_number: int) -> pd.DataFrame:
    """Return pre-race dry and wet ELO values for all drivers."""

    history = _load_driver_history().sort_values(["driver_id", "season", "round"])
    rows: list[dict[str, Any]] = []
    for driver_id in sorted(history["driver_id"].unique().tolist()):
        rows.append(
            {
                "driver_id": driver_id,
                "dry_elo": get_driver_dry_elo(driver_id, season, round_number),
                "wet_elo": get_driver_wet_elo(driver_id, season, round_number),
                "elo": get_driver_dry_elo(driver_id, season, round_number),
                "season": season,
                "round": round_number,
            }
        )
    return pd.DataFrame(rows).sort_values("dry_elo", ascending=False).reset_index(drop=True)


def plot_elo_history(driver_ids: list[str], rating_type: str = "dry") -> go.Figure:
    """Build a Plotly figure showing dry, wet, or both ELO histories."""

    history = _load_driver_history()
    filtered = history.loc[history["driver_id"].isin(driver_ids)].copy()
    if filtered.empty:
        figure = go.Figure()
        figure.update_layout(template="plotly_dark", title="No ELO history available for selected drivers")
        return figure

    filtered["race_label"] = filtered["season"].astype(str) + " R" + filtered["round"].astype(int).astype(str)
    rating_type = str(rating_type).strip().lower()
    figure = go.Figure()

    if rating_type == "both":
        columns = [("dry_elo_after", "Dry"), ("wet_elo_after", "Wet")]
    elif rating_type == "wet":
        columns = [("wet_elo_after", "Wet")]
    else:
        columns = [("dry_elo_after", "Dry")]

    for driver_id, driver_history in filtered.groupby("driver_id", sort=True):
        for column, label in columns:
            figure.add_trace(
                go.Scatter(
                    x=driver_history["race_label"],
                    y=driver_history[column],
                    mode="lines",
                    name=f"{driver_id} {label}" if len(columns) > 1 else driver_id,
                    hovertemplate="%{x}<br>ELO=%{y:.1f}<extra>%{fullData.name}</extra>",
                )
            )

    figure.update_layout(
        template="plotly_dark",
        title="Driver ELO History",
        xaxis_title="Race",
        yaxis_title="ELO",
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        font={"family": "Space Grotesk, sans-serif", "color": "#f5f5f5"},
    )
    return figure


def get_constructor_elo(constructor_id: str, season: int, round_number: int) -> float:
    """Return a constructor's condition-aware ELO immediately before a specified race."""

    history = _load_constructor_history()
    constructor_history = history.loc[history["constructor_id"] == constructor_id].copy()
    if constructor_history.empty:
        return float(CONFIG["starting_elo"])

    exact = constructor_history.loc[
        (constructor_history["season"] == season) & (constructor_history["round"] == round_number)
    ]
    if not exact.empty:
        return float(exact.iloc[-1]["elo_before"])

    prior = constructor_history.loc[
        (constructor_history["season"] < season)
        | ((constructor_history["season"] == season) & (constructor_history["round"] < round_number))
    ].sort_values(["season", "round"])
    if prior.empty:
        return float(CONFIG["starting_elo"])
    return float(prior.iloc[-1]["elo_after"])
