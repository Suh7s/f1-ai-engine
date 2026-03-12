"""Constructor Car Performance Index (CPI) derived from TSI-adjusted team results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline import load_calendar
from src.true_skill import build_tsi_artifacts
from src.utils import (
    CONFIG as BASE_CONFIG,
    ensure_directories,
    get_logger,
    load_dataframe,
    save_dataframe,
    save_json,
    timing_to_milliseconds,
    validate_columns,
)

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "cpi_history_path": BASE_CONFIG["processed_dir"] / "cpi_history.csv",
    "cpi_summary_path": BASE_CONFIG["processed_dir"] / "cpi_summary.csv",
    "cpi_metadata_path": BASE_CONFIG["processed_dir"] / "cpi_metadata.json",
    "regulation_years": [2017, 2022, 2026],
}

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class CPIArtifacts:
    """Persisted CPI artifact bundle."""

    cpi_history: pd.DataFrame
    cpi_summary: pd.DataFrame
    metadata: dict[str, Any]


def _prepare_results(race_results: pd.DataFrame) -> pd.DataFrame:
    """Standardize race results for CPI computation."""

    validate_columns(
        race_results,
        [
            "season",
            "round",
            "race_date",
            "race_name",
            "driver_id",
            "constructor_id",
            "constructor_name",
            "qualifying_position",
            "finish_position",
            "points",
            "dnf_reason",
            "q1",
            "q2",
            "q3",
        ],
        "CPI race results",
    )
    frame = race_results.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame["qualifying_position"] = pd.to_numeric(frame["qualifying_position"], errors="coerce")
    frame["finish_position"] = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame["points"] = pd.to_numeric(frame["points"], errors="coerce").fillna(0.0)
    frame["field_size"] = frame.groupby(["season", "round"])["driver_id"].transform("count").astype(float)
    frame["finish_position_adj"] = frame["finish_position"].fillna(frame["field_size"])
    frame["best_qualifying_time_ms"] = (
        pd.concat(
            [
                frame["q3"].map(timing_to_milliseconds),
                frame["q2"].map(timing_to_milliseconds),
                frame["q1"].map(timing_to_milliseconds),
            ],
            axis=1,
        )
        .bfill(axis=1)
        .iloc[:, 0]
    )
    frame["pole_time_ms"] = frame.groupby(["season", "round"])["best_qualifying_time_ms"].transform("min")
    frame["dnf_reason"] = frame["dnf_reason"].fillna("finished").astype(str).str.lower()
    return frame.sort_values(["season", "round", "constructor_id", "driver_id"]).reset_index(drop=True)


def _normalize_skill(signal: pd.Series) -> pd.Series:
    """Convert negative-is-better TSI values into positive skill expectation scores."""

    clipped = signal.clip(-1.0, 1.0).fillna(0.0)
    return (1.0 - clipped) / 2.0


def _qualifying_score(frame: pd.DataFrame) -> pd.Series:
    """Map qualifying performance to a 0-1 field-relative pace score."""

    time_gap = (
        frame["best_qualifying_time_ms"].to_numpy(dtype=float)
        - frame["pole_time_ms"].to_numpy(dtype=float)
    ) / np.clip(frame["pole_time_ms"].to_numpy(dtype=float), 1.0, None)
    time_score = 1.0 - np.clip(time_gap / 0.03, 0.0, 1.0)
    position_score = 1.0 - (
        (frame["qualifying_position"].to_numpy(dtype=float) - 1.0)
        / np.clip(frame["field_size"].to_numpy(dtype=float) - 1.0, 1.0, None)
    )
    return pd.Series(
        np.where(np.isfinite(time_score), time_score, position_score),
        index=frame.index,
        dtype=float,
    )


def _race_score(frame: pd.DataFrame) -> pd.Series:
    """Map race execution and points to a 0-1 score."""

    finish_score = 1.0 - (
        (frame["finish_position_adj"].to_numpy(dtype=float) - 1.0)
        / np.clip(frame["field_size"].to_numpy(dtype=float) - 1.0, 1.0, None)
    )
    points_score = frame["points"].to_numpy(dtype=float) / 26.0
    return pd.Series(0.7 * finish_score + 0.3 * points_score, index=frame.index, dtype=float)


def _rolling_slope(values: pd.Series, window: int = 5) -> pd.Series:
    """Compute a rolling slope for a team time series."""

    series = values.astype(float).reset_index(drop=True)
    slopes: list[float] = []
    for idx in range(len(series)):
        start = max(0, idx - window + 1)
        window_values = series.iloc[start : idx + 1].dropna()
        if len(window_values) < 2:
            slopes.append(np.nan)
            continue
        x = np.arange(len(window_values), dtype=float)
        y = window_values.to_numpy(dtype=float)
        slopes.append(float(np.polyfit(x, y, deg=1)[0]))
    return pd.Series(slopes, index=values.index, dtype=float)


def _historical_regulation_score(frame: pd.DataFrame) -> pd.Series:
    """Return a prior regulation-adaptation score for each constructor row."""

    values = pd.Series(index=frame.index, dtype=float)
    regulation_years = [int(year) for year in CONFIG["regulation_years"]]
    for constructor_id, constructor_frame in frame.groupby("constructor_id", sort=False):
        constructor_frame = constructor_frame.sort_values(["season", "round"]).copy()
        season_scores: list[tuple[int, float]] = []
        constructor_values: list[float] = []
        for row in constructor_frame.itertuples(index=False):
            prior_scores = [score for year, score in season_scores if year < int(row.season)]
            if prior_scores:
                constructor_values.append(float(np.mean(prior_scores)))
            elif int(row.season) in regulation_years and int(row.round) <= 5:
                ongoing = constructor_frame.loc[
                    (constructor_frame["season"] == int(row.season))
                    & (constructor_frame["round"] <= int(row.round))
                ]
                constructor_values.append(float(ongoing["cpi_performance_raw"].mean()) if not ongoing.empty else np.nan)
            else:
                constructor_values.append(np.nan)

            if int(row.season) in regulation_years and int(row.round) == 5:
                first_five = constructor_frame.loc[
                    (constructor_frame["season"] == int(row.season)) & (constructor_frame["round"] <= 5)
                ]
                if not first_five.empty:
                    season_scores.append((int(row.season), float(first_five["cpi_performance_raw"].mean())))
        values.loc[constructor_frame.index] = constructor_values
    return values.sort_index()


def build_cpi_artifacts(
    race_results: pd.DataFrame | None = None,
    tsi_driver_race: pd.DataFrame | None = None,
) -> CPIArtifacts:
    """Build CPI artifacts using pre-race TSI baselines."""

    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    if tsi_driver_race is None:
        tsi_driver_race = build_tsi_artifacts(race_results=race_results).driver_race

    frame = _prepare_results(race_results)
    if tsi_driver_race.empty:
        frame["tsi_qualifying_before"] = 0.0
        frame["tsi_race_before"] = 0.0
        frame["tsi_composite_before"] = 0.0
    else:
        tsi_merge = tsi_driver_race[
            [
                "season",
                "round",
                "driver_id",
                "tsi_qualifying_before",
                "tsi_race_before",
                "tsi_composite_before",
            ]
        ].copy()
        frame = frame.merge(tsi_merge, how="left", on=["season", "round", "driver_id"])
        for column in ["tsi_qualifying_before", "tsi_race_before", "tsi_composite_before"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    frame["actual_qualifying_score"] = _qualifying_score(frame)
    frame["actual_race_score"] = _race_score(frame)
    frame["expected_qualifying_score"] = _normalize_skill(frame["tsi_qualifying_before"])
    frame["expected_race_score"] = _normalize_skill(frame["tsi_race_before"])
    frame["mechanical_reliability_race"] = (~frame["dnf_reason"].eq("mechanical")).astype(float)

    constructor_race = (
        frame.groupby(["season", "round", "race_name", "race_date", "constructor_id", "constructor_name"], as_index=False)
        .agg(
            driver_count=("driver_id", "count"),
            field_size=("field_size", "max"),
            actual_qualifying_score=("actual_qualifying_score", "mean"),
            actual_race_score=("actual_race_score", "mean"),
            expected_qualifying_score=("expected_qualifying_score", "mean"),
            expected_race_score=("expected_race_score", "mean"),
            mechanical_reliability_race=("mechanical_reliability_race", "mean"),
            mean_points=("points", "mean"),
            mean_finish_position=("finish_position_adj", "mean"),
        )
        .sort_values(["constructor_id", "season", "round"])
        .reset_index(drop=True)
    )

    constructor_race["cpi_qualifying_raw"] = (
        constructor_race["actual_qualifying_score"] - constructor_race["expected_qualifying_score"]
    )
    constructor_race["cpi_race_raw"] = constructor_race["actual_race_score"] - constructor_race["expected_race_score"]
    constructor_race["cpi_performance_raw"] = 0.5 * (
        constructor_race["cpi_qualifying_raw"] + constructor_race["cpi_race_raw"]
    )

    frames: list[pd.DataFrame] = []
    for constructor_id, constructor_frame in constructor_race.groupby("constructor_id", sort=False):
        constructor_frame = constructor_frame.sort_values(["season", "round"]).copy()
        constructor_frame["cpi_qualifying_before"] = (
            constructor_frame["cpi_qualifying_raw"].shift(1).rolling(5, min_periods=1).mean()
        )
        constructor_frame["cpi_qualifying_after"] = constructor_frame["cpi_qualifying_raw"].rolling(5, min_periods=1).mean()
        constructor_frame["cpi_race_before"] = (
            constructor_frame["cpi_race_raw"].shift(1).rolling(5, min_periods=1).mean()
        )
        constructor_frame["cpi_race_after"] = constructor_frame["cpi_race_raw"].rolling(5, min_periods=1).mean()
        constructor_frame["cpi_reliability_before"] = (
            constructor_frame["mechanical_reliability_race"].shift(1).rolling(10, min_periods=1).mean()
        )
        constructor_frame["cpi_reliability_after"] = (
            constructor_frame["mechanical_reliability_race"].rolling(10, min_periods=1).mean()
        )
        constructor_frame["cpi_performance_before"] = (
            constructor_frame["cpi_performance_raw"].shift(1).rolling(5, min_periods=1).mean()
        )
        constructor_frame["cpi_performance_after"] = (
            constructor_frame["cpi_performance_raw"].rolling(5, min_periods=1).mean()
        )
        constructor_frame["cpi_development_rate_before"] = _rolling_slope(
            constructor_frame["cpi_performance_raw"].shift(1),
            window=5,
        )
        constructor_frame["cpi_development_rate_after"] = _rolling_slope(
            constructor_frame["cpi_performance_raw"],
            window=5,
        )
        frames.append(constructor_frame)

    cpi_history = pd.concat(frames, ignore_index=True).sort_values(["season", "round", "constructor_id"]).reset_index(drop=True)
    cpi_history["cpi_regulation_adaptability"] = _historical_regulation_score(cpi_history)

    latest = (
        cpi_history.sort_values(["constructor_id", "race_date", "season", "round"])
        .groupby("constructor_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest["cpi_rank"] = latest["cpi_performance_after"].rank(method="dense", ascending=False).astype(int)
    cpi_summary = latest[
        [
            "constructor_id",
            "constructor_name",
            "season",
            "round",
            "cpi_qualifying_after",
            "cpi_race_after",
            "cpi_reliability_after",
            "cpi_performance_after",
            "cpi_development_rate_after",
            "cpi_regulation_adaptability",
            "cpi_rank",
        ]
    ].rename(
        columns={
            "season": "latest_season",
            "round": "latest_round",
            "cpi_qualifying_after": "cpi_qualifying",
            "cpi_race_after": "cpi_race",
            "cpi_reliability_after": "cpi_reliability",
            "cpi_performance_after": "cpi_performance",
            "cpi_development_rate_after": "cpi_development_rate",
        }
    ).sort_values(["cpi_performance", "constructor_name"], ascending=[False, True]).reset_index(drop=True)

    metadata = {
        "season_range": [int(cpi_history["season"].min()), int(cpi_history["season"].max())] if not cpi_history.empty else [],
        "regulation_years": [int(year) for year in CONFIG["regulation_years"]],
    }
    return CPIArtifacts(cpi_history=cpi_history, cpi_summary=cpi_summary, metadata=metadata)


def run_cpi_pipeline(
    race_results: pd.DataFrame | None = None,
    tsi_driver_race: pd.DataFrame | None = None,
) -> CPIArtifacts:
    """Build and persist CPI artifacts."""

    ensure_directories()
    artifacts = build_cpi_artifacts(race_results=race_results, tsi_driver_race=tsi_driver_race)
    save_dataframe(artifacts.cpi_history, CONFIG["cpi_history_path"])
    save_dataframe(artifacts.cpi_summary, CONFIG["cpi_summary_path"])
    save_json(artifacts.metadata, CONFIG["cpi_metadata_path"])
    LOGGER.info(
        "Saved CPI artifacts | cpi_history=%s cpi_summary=%s",
        len(artifacts.cpi_history),
        len(artifacts.cpi_summary),
    )
    return artifacts


def _load_cpi_history() -> pd.DataFrame:
    """Load the persisted CPI history, building it if necessary."""

    path = Path(CONFIG["cpi_history_path"])
    if not path.exists():
        return run_cpi_pipeline().cpi_history
    return load_dataframe(path, parse_dates=["race_date"])


def get_cpi(constructor_id: str, season: int, round_number: int) -> dict[str, Any]:
    """Return the latest available CPI snapshot for a constructor."""

    history = _load_cpi_history()
    subset = history.loc[
        (history["constructor_id"] == str(constructor_id))
        & (
            (history["season"] < int(season))
            | ((history["season"] == int(season)) & (history["round"] <= int(round_number)))
        )
    ].sort_values(["season", "round"])
    if subset.empty:
        return {
            "constructor_id": str(constructor_id),
            "season": int(season),
            "round": int(round_number),
            "cpi_qualifying": np.nan,
            "cpi_race": np.nan,
            "cpi_reliability": np.nan,
            "cpi_performance": np.nan,
            "cpi_development_rate": np.nan,
            "cpi_regulation_adaptability": np.nan,
        }
    row = subset.iloc[-1]
    return {
        "constructor_id": str(row["constructor_id"]),
        "constructor_name": row.get("constructor_name"),
        "season": int(season),
        "round": int(round_number),
        "snapshot_season": int(row["season"]),
        "snapshot_round": int(row["round"]),
        "cpi_qualifying": float(row["cpi_qualifying_after"]) if pd.notna(row["cpi_qualifying_after"]) else np.nan,
        "cpi_race": float(row["cpi_race_after"]) if pd.notna(row["cpi_race_after"]) else np.nan,
        "cpi_reliability": float(row["cpi_reliability_after"]) if pd.notna(row["cpi_reliability_after"]) else np.nan,
        "cpi_performance": float(row["cpi_performance_after"]) if pd.notna(row["cpi_performance_after"]) else np.nan,
        "cpi_development_rate": float(row["cpi_development_rate_after"]) if pd.notna(row["cpi_development_rate_after"]) else np.nan,
        "cpi_regulation_adaptability": float(row["cpi_regulation_adaptability"]) if pd.notna(row["cpi_regulation_adaptability"]) else np.nan,
    }


def get_development_trajectory(constructor_id: str, season: int) -> pd.DataFrame:
    """Return race-by-race CPI evolution for a constructor-season."""

    history = _load_cpi_history()
    trajectory = history.loc[
        (history["constructor_id"] == str(constructor_id)) & (history["season"] == int(season))
    ].copy()
    columns = [
        "season",
        "round",
        "race_name",
        "race_date",
        "constructor_id",
        "constructor_name",
        "cpi_qualifying_after",
        "cpi_race_after",
        "cpi_reliability_after",
        "cpi_performance_after",
        "cpi_development_rate_after",
    ]
    return trajectory[columns].rename(
        columns={
            "cpi_qualifying_after": "cpi_qualifying",
            "cpi_race_after": "cpi_race",
            "cpi_reliability_after": "cpi_reliability",
            "cpi_performance_after": "cpi_performance",
            "cpi_development_rate_after": "cpi_development_rate",
        }
    ).reset_index(drop=True)


def predict_end_of_season_cpi(constructor_id: str, season: int) -> dict[str, Any]:
    """Project a constructor's end-of-season CPI using recent development rate."""

    trajectory = get_development_trajectory(constructor_id=constructor_id, season=season)
    if trajectory.empty:
        return {
            "constructor_id": str(constructor_id),
            "season": int(season),
            "projected_cpi": np.nan,
            "confidence_interval": (np.nan, np.nan),
        }
    latest = trajectory.iloc[-1]
    current_cpi = float(latest["cpi_performance"])
    development_rate = float(latest["cpi_development_rate"]) if pd.notna(latest["cpi_development_rate"]) else 0.0
    calendar = load_calendar(int(season), allow_fetch=False)
    total_rounds = int(calendar["round"].max()) if not calendar.empty else int(trajectory["round"].max())
    rounds_remaining = max(total_rounds - int(latest["round"]), 0)
    recent = trajectory.tail(5).copy()
    if len(recent) >= 2:
        x = np.arange(len(recent), dtype=float)
        fit = np.polyfit(x, recent["cpi_performance"].to_numpy(dtype=float), deg=1)
        residuals = recent["cpi_performance"].to_numpy(dtype=float) - np.polyval(fit, x)
        sigma = float(np.std(residuals, ddof=0))
    else:
        sigma = 0.05
    projected = current_cpi + development_rate * rounds_remaining
    interval = (
        float(projected - 1.96 * sigma * np.sqrt(max(rounds_remaining, 1))),
        float(projected + 1.96 * sigma * np.sqrt(max(rounds_remaining, 1))),
    )
    return {
        "constructor_id": str(constructor_id),
        "constructor_name": latest["constructor_name"],
        "season": int(season),
        "rounds_remaining": int(rounds_remaining),
        "projected_cpi": float(projected),
        "confidence_interval": interval,
        "current_cpi": current_cpi,
        "development_rate": development_rate,
    }
