"""Lightweight auto-update scheduler for the living F1 season system."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_pipeline import (
    ErgastClient,
    fetch_results,
    get_current_season,
    get_cached_rounds,
    run_data_pipeline,
)
from src.elo_rating import run_elo_pipeline
from src.fastf1_pipeline import run_fastf1_pipeline
from src.feature_engineering import run_feature_pipeline
from src.model_training import MODEL_PATHS, retrain_incremental, retrain_models
from src.utils import CONFIG as BASE_CONFIG, get_logger, load_dataframe
from src.weather_pipeline import run_weather_pipeline

CONFIG: dict[str, Any] = {
    "processed_dir": BASE_CONFIG["processed_dir"],
    "features_path": BASE_CONFIG["processed_dir"] / "features.csv",
}

LOGGER = get_logger(__name__)


def _load_existing_features() -> pd.DataFrame:
    """Load the current feature matrix if it exists."""

    path = CONFIG["features_path"]
    if not path.exists():
        return pd.DataFrame()
    return load_dataframe(path, parse_dates=["race_date"])


def check_for_new_data() -> list[int]:
    """Return newly completed rounds for the active season that are not cached locally."""

    current_season = get_current_season()
    client = ErgastClient()
    try:
        _, latest_results = fetch_results(client, current_season)
    except Exception as exc:
        LOGGER.warning("Unable to check for new current-season data; falling back to no-op update: %s", exc)
        return []
    completed_rounds = sorted(latest_results["round"].dropna().astype(int).unique().tolist())
    cached_rounds = sorted(get_cached_rounds(current_season))
    return sorted(set(completed_rounds) - set(cached_rounds))


def update_elo_ratings(race_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild driver and constructor ELO histories from the refreshed race results."""

    return run_elo_pipeline(race_results)


def auto_update(skip_fastf1: bool = False) -> dict[str, Any]:
    """Refresh the current season, rebuild features, and retrain incrementally if new rounds exist."""

    current_season = get_current_season()
    new_rounds = check_for_new_data()
    if not new_rounds:
        LOGGER.info("No new completed rounds detected for season %s", current_season)
        return {"updated": False, "season": current_season, "new_rounds": []}

    LOGGER.info("New rounds detected for season %s: %s", current_season, new_rounds)
    previous_features = _load_existing_features()

    pipeline_data = run_data_pipeline(refresh=True, seasons=[current_season])
    _, race_results = run_weather_pipeline(pipeline_data["race_results"], refresh=True)
    fastf1_features, fastf1_failures = run_fastf1_pipeline(
        race_results=race_results,
        refresh=True,
        skip=skip_fastf1,
    )
    driver_elo_history, constructor_elo_history = update_elo_ratings(race_results)
    features = run_feature_pipeline(
        race_results=race_results,
        driver_elo_history=driver_elo_history,
        constructor_elo_history=constructor_elo_history,
        pitstops_df=pipeline_data["pitstops"],
        fastf1_features_df=fastf1_features,
    )

    new_rows = features.loc[
        (features["season"] == current_season) & (features["round"].astype(int).isin(new_rounds))
    ].copy()
    if not previous_features.empty:
        historical_keys = set(
            previous_features[["season", "round", "driver_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        new_rows = new_rows.loc[
            ~new_rows[["season", "round", "driver_id"]]
            .apply(tuple, axis=1)
            .isin(historical_keys)
        ].copy()

    models_ready = all(path.exists() for path in MODEL_PATHS.values())
    if new_rows.empty or not models_ready:
        retrain_result = retrain_models(refresh=True)
    else:
        retrain_result = retrain_incremental(new_rows)

    LOGGER.info(
        "Living-season update complete for season %s | new_rounds=%s | fastf1_failures=%s",
        current_season,
        new_rounds,
        len(fastf1_failures),
    )
    return {
        "updated": True,
        "season": current_season,
        "new_rounds": new_rounds,
        "retrain_result": retrain_result,
        "fastf1_failures": fastf1_failures.to_dict(orient="records"),
    }


class Scheduler:
    """Compatibility wrapper for scheduler smoke tests and simple polling flows."""

    def __init__(self, season: int | None = None) -> None:
        self.season = int(season or get_current_season())

    def fetch_calendar(self) -> pd.DataFrame:
        path = CONFIG["processed_dir"] / "calendar.csv"
        return load_dataframe(path) if path.exists() else pd.DataFrame()

    def get_cached_rounds(self) -> list[int]:
        return sorted(get_cached_rounds(self.season))

    def check_for_new_data(self) -> list[int]:
        calendar = self.fetch_calendar()
        if not calendar.empty and {"round", "is_completed"}.issubset(calendar.columns):
            completed = calendar.loc[pd.to_numeric(calendar["is_completed"], errors="coerce").fillna(0).astype(int) == 1]
            completed_rounds = sorted(completed["round"].dropna().astype(int).unique().tolist())
            return sorted(set(completed_rounds) - set(self.get_cached_rounds()))
        return check_for_new_data()
