"""Driver True Skill Index (TSI) based on teammate-adjusted race comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    "head_to_head_path": BASE_CONFIG["processed_dir"] / "tsi_head_to_head.csv",
    "driver_race_path": BASE_CONFIG["processed_dir"] / "tsi_driver_race.csv",
    "career_summary_path": BASE_CONFIG["processed_dir"] / "tsi_career_summary.csv",
    "metadata_path": BASE_CONFIG["processed_dir"] / "tsi_metadata.json",
    "rolling_window": 20,
    "rolling_min_periods": 3,
}

LOGGER = get_logger(__name__)

ERA_BOUNDARIES: list[tuple[int, int, str]] = [
    (2016, 2016, "2016_transition"),
    (2017, 2021, "2017_regs"),
    (2022, 2025, "2022_ground_effect"),
    (2026, 2099, "2026_next_gen"),
]


@dataclass(slots=True)
class TSIArtifacts:
    """Persisted TSI artifact bundle."""

    head_to_head: pd.DataFrame
    driver_race: pd.DataFrame
    career_summary: pd.DataFrame
    metadata: dict[str, Any]


def _era_for_season(season: int) -> str:
    """Return the era bucket for a season."""

    season_int = int(season)
    for start, end, label in ERA_BOUNDARIES:
        if start <= season_int <= end:
            return label
    return "unknown"


def _qualifying_time_ms(frame: pd.DataFrame) -> pd.Series:
    """Return each driver's best qualifying time in milliseconds."""

    columns = ["q3", "q2", "q1"]
    converted = frame[columns].copy()
    for column in columns:
        converted[column] = converted[column].map(timing_to_milliseconds)
    return converted.bfill(axis=1).iloc[:, 0]


def _condition_bucket(frame: pd.DataFrame) -> pd.Series:
    """Collapse race conditions into dry or wet buckets."""

    wet_mask = frame["is_wet_race"].fillna(0).astype(int).eq(1) | frame["is_mixed_race"].fillna(0).astype(int).eq(1)
    return np.where(wet_mask, "wet", "dry")


def _prepare_results(race_results: pd.DataFrame) -> pd.DataFrame:
    """Prepare race results for teammate head-to-head comparisons."""

    validate_columns(
        race_results,
        [
            "season",
            "round",
            "race_name",
            "race_date",
            "driver_id",
            "constructor_id",
            "qualifying_position",
            "finish_position",
            "points",
            "dnf_flag",
            "dnf_reason",
            "q1",
            "q2",
            "q3",
            "is_wet_race",
            "is_mixed_race",
        ],
        "TSI race results",
    )
    frame = race_results.copy()
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce")
    frame["qualifying_position"] = pd.to_numeric(frame["qualifying_position"], errors="coerce")
    frame["finish_position"] = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame["points"] = pd.to_numeric(frame["points"], errors="coerce").fillna(0.0)
    frame["field_size"] = frame.groupby(["season", "round"])["driver_id"].transform("count").astype(float)
    frame["finish_position_adj"] = frame["finish_position"].fillna(frame["field_size"])
    frame["best_qualifying_time_ms"] = _qualifying_time_ms(frame)
    frame["pole_time_ms"] = frame.groupby(["season", "round"])["best_qualifying_time_ms"].transform("min")
    frame["condition_bucket"] = _condition_bucket(frame)
    frame["era"] = frame["season"].map(_era_for_season)
    frame["driver_name"] = frame.get("driver_name")
    if frame["driver_name"].isna().all():
        frame["driver_name"] = (
            frame.get("driver_given_name", pd.Series("", index=frame.index)).fillna("").str.strip()
            + " "
            + frame.get("driver_family_name", pd.Series("", index=frame.index)).fillna("").str.strip()
        ).str.strip()
    frame.loc[frame["driver_name"].eq(""), "driver_name"] = frame.loc[frame["driver_name"].eq(""), "driver_id"]
    frame["dnf_reason"] = frame["dnf_reason"].fillna("finished").astype(str).str.lower()
    return frame.sort_values(["season", "round", "constructor_id", "driver_id"]).reset_index(drop=True)


def _build_head_to_head_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Build symmetric driver-vs-teammate rows for each constructor race."""

    base_columns = [
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "constructor_id",
        "driver_id",
        "driver_name",
        "driver_number",
        "qualifying_position",
        "finish_position_adj",
        "points",
        "field_size",
        "best_qualifying_time_ms",
        "pole_time_ms",
        "condition_bucket",
        "era",
        "dnf_flag",
        "dnf_reason",
    ]
    base = frame[base_columns].copy()
    merged = base.merge(
        base,
        how="inner",
        on=["season", "round", "race_name", "race_date", "circuit_id", "constructor_id", "field_size", "condition_bucket", "era"],
        suffixes=("_driver", "_teammate"),
    )
    merged = merged.loc[merged["driver_id_driver"] != merged["driver_id_teammate"]].copy()
    if merged.empty:
        return pd.DataFrame()

    pole_time = np.clip(merged["pole_time_ms_driver"].to_numpy(dtype=float), 1.0, None)
    qualifying_valid = merged["best_qualifying_time_ms_driver"].notna() & merged["best_qualifying_time_ms_teammate"].notna()
    merged["qualifying_valid"] = qualifying_valid.astype(int)
    merged["qualifying_delta_pct"] = np.where(
        qualifying_valid,
        (
            merged["best_qualifying_time_ms_driver"].to_numpy(dtype=float)
            - merged["best_qualifying_time_ms_teammate"].to_numpy(dtype=float)
        )
        / pole_time,
        np.nan,
    )

    race_valid = (
        ~merged["dnf_reason_driver"].eq("mechanical")
        & ~merged["dnf_reason_teammate"].eq("mechanical")
        & merged["finish_position_adj_driver"].notna()
        & merged["finish_position_adj_teammate"].notna()
    )
    merged["race_valid"] = race_valid.astype(int)
    merged["race_delta"] = np.where(
        race_valid,
        merged["finish_position_adj_driver"].to_numpy(dtype=float)
        - merged["finish_position_adj_teammate"].to_numpy(dtype=float),
        np.nan,
    )
    merged["race_delta_norm"] = np.where(
        race_valid,
        merged["race_delta"].to_numpy(dtype=float) / np.clip(merged["field_size"].to_numpy(dtype=float) - 1.0, 1.0, None),
        np.nan,
    )
    merged["points_delta"] = np.where(
        race_valid,
        merged["points_driver"].to_numpy(dtype=float) - merged["points_teammate"].to_numpy(dtype=float),
        np.nan,
    )
    merged["points_delta_norm"] = np.where(
        race_valid,
        merged["points_delta"].to_numpy(dtype=float) / 26.0,
        np.nan,
    )
    merged["direct_h2h_result"] = np.where(
        ~race_valid,
        np.nan,
        np.sign(-merged["race_delta"].to_numpy(dtype=float)),
    )
    output_columns = {
        "driver_id_driver": "driver_id",
        "driver_name_driver": "driver_name",
        "driver_number_driver": "driver_number",
        "driver_id_teammate": "teammate_id",
        "driver_name_teammate": "teammate_name",
        "driver_number_teammate": "teammate_number",
        "qualifying_position_driver": "qualifying_position",
        "qualifying_position_teammate": "teammate_qualifying_position",
        "finish_position_adj_driver": "finish_position_adj",
        "finish_position_adj_teammate": "teammate_finish_position_adj",
        "points_driver": "points",
        "points_teammate": "teammate_points",
        "pole_time_ms_driver": "pole_time_ms",
        "condition_bucket": "condition_bucket",
        "era": "era",
    }
    return merged.rename(columns=output_columns)[list(output_columns.values()) + [
        "season",
        "round",
        "race_name",
        "race_date",
        "circuit_id",
        "constructor_id",
        "field_size",
        "qualifying_valid",
        "qualifying_delta_pct",
        "race_valid",
        "race_delta",
        "race_delta_norm",
        "points_delta",
        "points_delta_norm",
        "direct_h2h_result",
    ]].sort_values(["season", "round", "constructor_id", "driver_id"]).reset_index(drop=True)


def _add_era_adjustment(driver_race: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach era-adjusted normalized signals to driver-race rows."""

    if driver_race.empty:
        empty_metadata = {
            "era_spreads": {},
            "season_range": [],
        }
        return driver_race, empty_metadata

    metadata: dict[str, Any] = {"era_spreads": {}}
    updated = driver_race.copy()
    spread_columns = {
        "qualifying_delta_pct": "qualifying_spread",
        "race_delta_norm": "race_spread",
        "points_delta_norm": "points_spread",
    }
    for value_column, label in spread_columns.items():
        global_spread = float(updated[value_column].abs().mean()) if updated[value_column].notna().any() else 1.0
        global_spread = global_spread if global_spread > 0 else 1.0
        era_spread = (
            updated.groupby("era")[value_column]
            .apply(lambda series: float(series.dropna().abs().mean()) if series.dropna().size else global_spread)
            .replace(0.0, global_spread)
            .to_dict()
        )
        metadata["era_spreads"][label] = {str(key): float(value) for key, value in era_spread.items()}
        updated[f"{value_column}_era_spread"] = updated["era"].map(era_spread).fillna(global_spread).astype(float)
        updated[f"{value_column}_era_adj"] = updated[value_column] / np.clip(updated[f"{value_column}_era_spread"], 1e-9, None)
        updated[f"{value_column}_signal"] = np.tanh(updated[f"{value_column}_era_adj"])

    updated["tsi_qualifying_event"] = updated["qualifying_delta_pct_signal"]
    updated["tsi_race_event"] = updated["race_delta_norm_signal"]
    updated["tsi_points_event"] = updated["points_delta_norm_signal"]
    updated["tsi_composite_event"] = 0.4 * updated["tsi_qualifying_event"] + 0.6 * updated["tsi_race_event"]
    metadata["season_range"] = [
        int(updated["season"].min()),
        int(updated["season"].max()),
    ]
    return updated, metadata


def _rolling_series(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Return a rolling mean that ignores sparse leading values."""

    return series.rolling(window=window, min_periods=min_periods).mean()


def _aggregate_driver_race(head_to_head: pd.DataFrame) -> pd.DataFrame:
    """Collapse pairwise rows into one row per driver-race and compute rolling/career TSI."""

    if head_to_head.empty:
        return pd.DataFrame()

    aggregated = (
        head_to_head.groupby(
            [
                "season",
                "round",
                "race_name",
                "race_date",
                "circuit_id",
                "constructor_id",
                "driver_id",
                "driver_name",
                "driver_number",
                "condition_bucket",
                "era",
            ],
            as_index=False,
        )
        .agg(
            teammate_count=("teammate_id", "nunique"),
            teammate_ids=("teammate_id", lambda values: "|".join(sorted({str(value) for value in values}))),
            teammate_names=("teammate_name", lambda values: "|".join(sorted({str(value) for value in values}))),
            field_size=("field_size", "max"),
            qualifying_delta_pct=("qualifying_delta_pct", "mean"),
            qualifying_valid=("qualifying_valid", "max"),
            race_delta=("race_delta", "mean"),
            race_delta_norm=("race_delta_norm", "mean"),
            race_valid=("race_valid", "max"),
            points_delta=("points_delta", "mean"),
            points_delta_norm=("points_delta_norm", "mean"),
            direct_h2h_result=("direct_h2h_result", "mean"),
        )
        .sort_values(["driver_id", "race_date", "season", "round"])
        .reset_index(drop=True)
    )
    aggregated, metadata = _add_era_adjustment(aggregated)
    rolling_window = int(CONFIG["rolling_window"])
    rolling_min_periods = int(CONFIG["rolling_min_periods"])

    frames: list[pd.DataFrame] = []
    for driver_id, driver_frame in aggregated.groupby("driver_id", sort=False):
        driver_frame = driver_frame.sort_values(["race_date", "season", "round"]).reset_index(drop=True)

        qual_signal = driver_frame["tsi_qualifying_event"]
        race_signal = driver_frame["tsi_race_event"]
        comp_signal = driver_frame["tsi_composite_event"]
        race_raw = driver_frame["race_delta_norm"]
        race_mask = driver_frame["race_valid"].eq(1)
        wet_mask = driver_frame["condition_bucket"].eq("wet")
        dry_mask = driver_frame["condition_bucket"].eq("dry")

        driver_frame["tsi_sample_size_before"] = race_mask.shift(1, fill_value=False).astype(int).cumsum()
        driver_frame["tsi_sample_size_after"] = race_mask.astype(int).cumsum()

        driver_frame["tsi_qualifying_before"] = qual_signal.shift(1).expanding(min_periods=1).mean()
        driver_frame["tsi_qualifying_after"] = qual_signal.expanding(min_periods=1).mean()
        driver_frame["tsi_race_before"] = race_signal.shift(1).expanding(min_periods=1).mean()
        driver_frame["tsi_race_after"] = race_signal.expanding(min_periods=1).mean()
        driver_frame["tsi_composite_before"] = 0.4 * driver_frame["tsi_qualifying_before"] + 0.6 * driver_frame["tsi_race_before"]
        driver_frame["tsi_composite_after"] = 0.4 * driver_frame["tsi_qualifying_after"] + 0.6 * driver_frame["tsi_race_after"]

        driver_frame["tsi_qualifying_raw_before"] = driver_frame["qualifying_delta_pct"].shift(1).expanding(min_periods=1).mean()
        driver_frame["tsi_qualifying_raw_after"] = driver_frame["qualifying_delta_pct"].expanding(min_periods=1).mean()
        driver_frame["tsi_race_raw_before"] = race_raw.shift(1).expanding(min_periods=1).mean()
        driver_frame["tsi_race_raw_after"] = race_raw.expanding(min_periods=1).mean()

        driver_frame["tsi_consistency_before"] = race_raw.shift(1).expanding(min_periods=2).std(ddof=0)
        driver_frame["tsi_consistency_after"] = race_raw.expanding(min_periods=2).std(ddof=0)

        driver_frame["tsi_rolling_20_before"] = _rolling_series(comp_signal.shift(1), rolling_window, rolling_min_periods)
        driver_frame["tsi_rolling_20_after"] = _rolling_series(comp_signal, rolling_window, rolling_min_periods)
        driver_frame["tsi_delta_before"] = driver_frame["tsi_rolling_20_before"] - driver_frame["tsi_composite_before"]
        driver_frame["tsi_delta_after"] = driver_frame["tsi_rolling_20_after"] - driver_frame["tsi_composite_after"]

        wet_signal = comp_signal.where(wet_mask)
        dry_signal = comp_signal.where(dry_mask)
        driver_frame["tsi_wet_before"] = wet_signal.shift(1).expanding(min_periods=1).mean()
        driver_frame["tsi_wet_after"] = wet_signal.expanding(min_periods=1).mean()
        driver_frame["tsi_dry_before"] = dry_signal.shift(1).expanding(min_periods=1).mean()
        driver_frame["tsi_dry_after"] = dry_signal.expanding(min_periods=1).mean()
        driver_frame["tsi_wet_sample_before"] = wet_mask.shift(1, fill_value=False).astype(int).cumsum()
        driver_frame["tsi_wet_sample_after"] = wet_mask.astype(int).cumsum()
        driver_frame["tsi_dry_sample_before"] = dry_mask.shift(1, fill_value=False).astype(int).cumsum()
        driver_frame["tsi_dry_sample_after"] = dry_mask.astype(int).cumsum()

        driver_frame["tsi_career_before"] = driver_frame["tsi_composite_before"]
        driver_frame["tsi_career_after"] = driver_frame["tsi_composite_after"]
        driver_frame["tsi_era_adjusted_before"] = driver_frame["tsi_composite_before"]
        driver_frame["tsi_era_adjusted_after"] = driver_frame["tsi_composite_after"]
        driver_frame["tsi_points_before"] = driver_frame["tsi_points_event"].shift(1).expanding(min_periods=1).mean()
        driver_frame["tsi_points_after"] = driver_frame["tsi_points_event"].expanding(min_periods=1).mean()

        frames.append(driver_frame)

    output = pd.concat(frames, ignore_index=True).sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    output.attrs["metadata"] = metadata
    return output


def _build_career_summary(driver_race: pd.DataFrame) -> pd.DataFrame:
    """Return the latest available TSI summary for each driver."""

    if driver_race.empty:
        return pd.DataFrame()
    latest = (
        driver_race.sort_values(["driver_id", "race_date", "season", "round"])
        .groupby("driver_id", as_index=False)
        .tail(1)
        .copy()
    )
    for source, target in [
        ("tsi_composite_after", "tsi_composite_rank"),
        ("tsi_wet_after", "tsi_wet_rank"),
        ("tsi_dry_after", "tsi_dry_rank"),
    ]:
        ranks = latest[source].rank(method="dense", ascending=True, na_option="bottom")
        latest[target] = ranks.fillna(len(latest) + 1).astype(int)
    summary_columns = [
        "driver_id",
        "driver_name",
        "season",
        "round",
        "constructor_id",
        "tsi_qualifying_raw_after",
        "tsi_qualifying_after",
        "tsi_race_raw_after",
        "tsi_race_after",
        "tsi_points_after",
        "tsi_composite_after",
        "tsi_rolling_20_after",
        "tsi_delta_after",
        "tsi_consistency_after",
        "tsi_sample_size_after",
        "tsi_wet_after",
        "tsi_dry_after",
        "tsi_wet_sample_after",
        "tsi_dry_sample_after",
        "tsi_composite_rank",
        "tsi_wet_rank",
        "tsi_dry_rank",
    ]
    return latest[summary_columns].rename(
        columns={
            "season": "latest_season",
            "round": "latest_round",
            "tsi_qualifying_raw_after": "tsi_qualifying_raw",
            "tsi_qualifying_after": "tsi_qualifying",
            "tsi_race_raw_after": "tsi_race_raw",
            "tsi_race_after": "tsi_race",
            "tsi_points_after": "tsi_points",
            "tsi_composite_after": "tsi_composite",
            "tsi_rolling_20_after": "tsi_rolling_20",
            "tsi_delta_after": "tsi_delta",
            "tsi_consistency_after": "tsi_consistency",
            "tsi_sample_size_after": "tsi_sample_size",
            "tsi_wet_after": "tsi_wet",
            "tsi_dry_after": "tsi_dry",
            "tsi_wet_sample_after": "tsi_wet_sample_size",
            "tsi_dry_sample_after": "tsi_dry_sample_size",
        }
    ).sort_values(["tsi_composite", "tsi_sample_size", "driver_name"], ascending=[True, False, True]).reset_index(drop=True)


def build_tsi_artifacts(race_results: pd.DataFrame | None = None) -> TSIArtifacts:
    """Build TSI artifacts from processed race results."""

    if race_results is None:
        race_results = load_dataframe(CONFIG["race_results_path"])
    prepared = _prepare_results(race_results)
    head_to_head = _build_head_to_head_pairs(prepared)
    driver_race = _aggregate_driver_race(head_to_head)
    metadata = dict(driver_race.attrs.get("metadata", {}))
    metadata["rolling_window"] = int(CONFIG["rolling_window"])
    metadata["rolling_min_periods"] = int(CONFIG["rolling_min_periods"])
    career_summary = _build_career_summary(driver_race)
    return TSIArtifacts(
        head_to_head=head_to_head,
        driver_race=driver_race,
        career_summary=career_summary,
        metadata=metadata,
    )


def run_tsi_pipeline(race_results: pd.DataFrame | None = None) -> TSIArtifacts:
    """Build and persist TSI artifacts to processed storage."""

    ensure_directories()
    artifacts = build_tsi_artifacts(race_results=race_results)
    save_dataframe(artifacts.head_to_head, CONFIG["head_to_head_path"])
    save_dataframe(artifacts.driver_race, CONFIG["driver_race_path"])
    save_dataframe(artifacts.career_summary, CONFIG["career_summary_path"])
    save_json(artifacts.metadata, CONFIG["metadata_path"])
    LOGGER.info(
        "Saved TSI artifacts | head_to_head=%s driver_race=%s career_summary=%s",
        len(artifacts.head_to_head),
        len(artifacts.driver_race),
        len(artifacts.career_summary),
    )
    return artifacts


def _load_driver_race() -> pd.DataFrame:
    """Load the persisted driver-race TSI table, building it if necessary."""

    path = Path(CONFIG["driver_race_path"])
    if not path.exists():
        return run_tsi_pipeline().driver_race
    return load_dataframe(path, parse_dates=["race_date"])


def _load_head_to_head() -> pd.DataFrame:
    """Load the persisted head-to-head table, building it if necessary."""

    path = Path(CONFIG["head_to_head_path"])
    if not path.exists():
        return run_tsi_pipeline().head_to_head
    return load_dataframe(path, parse_dates=["race_date"])


def _load_career_summary() -> pd.DataFrame:
    """Load the persisted career summary, building it if necessary."""

    path = Path(CONFIG["career_summary_path"])
    if not path.exists():
        return run_tsi_pipeline().career_summary
    return load_dataframe(path)


def get_tsi(driver_id: str, season: int, round_number: int) -> dict[str, Any]:
    """Return the latest TSI snapshot for a driver at or before a race round."""

    driver_race = _load_driver_race()
    subset = driver_race.loc[
        (driver_race["driver_id"] == str(driver_id))
        & (
            (driver_race["season"] < int(season))
            | ((driver_race["season"] == int(season)) & (driver_race["round"] <= int(round_number)))
        )
    ].sort_values(["season", "round"])
    if subset.empty:
        return {
            "driver_id": str(driver_id),
            "season": int(season),
            "round": int(round_number),
            "tsi_qualifying": np.nan,
            "tsi_race": np.nan,
            "tsi_composite": np.nan,
            "tsi_rolling_20": np.nan,
            "tsi_career": np.nan,
            "tsi_delta": np.nan,
            "tsi_consistency": np.nan,
            "tsi_sample_size": 0,
            "tsi_wet": np.nan,
            "tsi_dry": np.nan,
        }
    row = subset.iloc[-1]
    return {
        "driver_id": str(row["driver_id"]),
        "driver_name": row.get("driver_name"),
        "season": int(season),
        "round": int(round_number),
        "snapshot_season": int(row["season"]),
        "snapshot_round": int(row["round"]),
        "tsi_qualifying": float(row["tsi_qualifying_after"]),
        "tsi_race": float(row["tsi_race_after"]),
        "tsi_qualifying_raw": float(row["tsi_qualifying_raw_after"]),
        "tsi_race_raw": float(row["tsi_race_raw_after"]),
        "tsi_points": float(row["tsi_points_after"]),
        "tsi_composite": float(row["tsi_composite_after"]),
        "tsi_career": float(row["tsi_career_after"]),
        "tsi_rolling_20": float(row["tsi_rolling_20_after"]) if pd.notna(row["tsi_rolling_20_after"]) else np.nan,
        "tsi_delta": float(row["tsi_delta_after"]) if pd.notna(row["tsi_delta_after"]) else np.nan,
        "tsi_consistency": float(row["tsi_consistency_after"]) if pd.notna(row["tsi_consistency_after"]) else np.nan,
        "tsi_sample_size": int(row["tsi_sample_size_after"]),
        "tsi_wet": float(row["tsi_wet_after"]) if pd.notna(row["tsi_wet_after"]) else np.nan,
        "tsi_dry": float(row["tsi_dry_after"]) if pd.notna(row["tsi_dry_after"]) else np.nan,
        "tsi_wet_sample_size": int(row["tsi_wet_sample_after"]),
        "tsi_dry_sample_size": int(row["tsi_dry_sample_after"]),
    }


def get_tsi_ranking(season: int, round_number: int) -> pd.DataFrame:
    """Return a driver ranking snapshot by TSI composite at or before a round."""

    driver_race = _load_driver_race()
    valid = driver_race.loc[
        (driver_race["season"] < int(season))
        | ((driver_race["season"] == int(season)) & (driver_race["round"] <= int(round_number)))
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    latest = (
        valid.sort_values(["driver_id", "race_date", "season", "round"])
        .groupby("driver_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest = latest.rename(
        columns={
            "tsi_composite_after": "tsi_composite",
            "tsi_qualifying_after": "tsi_qualifying",
            "tsi_race_after": "tsi_race",
            "tsi_rolling_20_after": "tsi_rolling_20",
            "tsi_delta_after": "tsi_delta",
            "tsi_consistency_after": "tsi_consistency",
            "tsi_sample_size_after": "tsi_sample_size",
            "tsi_wet_after": "tsi_wet",
            "tsi_dry_after": "tsi_dry",
        }
    )
    latest = latest[
        [
            "driver_id",
            "driver_name",
            "constructor_id",
            "season",
            "round",
            "tsi_qualifying",
            "tsi_race",
            "tsi_composite",
            "tsi_rolling_20",
            "tsi_delta",
            "tsi_consistency",
            "tsi_sample_size",
            "tsi_wet",
            "tsi_dry",
        ]
    ].sort_values(["tsi_composite", "tsi_sample_size", "driver_name"], ascending=[True, False, True]).reset_index(drop=True)
    latest.insert(0, "rank", np.arange(1, len(latest) + 1))
    return latest


def compare_drivers(driver_id_1: str, driver_id_2: str) -> dict[str, Any]:
    """Return a TSI comparison between two drivers plus any direct teammate record."""

    summary = _load_career_summary().set_index("driver_id")
    driver_1 = summary.loc[str(driver_id_1)].to_dict() if str(driver_id_1) in summary.index else {"driver_id": str(driver_id_1)}
    driver_2 = summary.loc[str(driver_id_2)].to_dict() if str(driver_id_2) in summary.index else {"driver_id": str(driver_id_2)}

    head_to_head = _load_head_to_head()
    direct = head_to_head.loc[
        ((head_to_head["driver_id"] == str(driver_id_1)) & (head_to_head["teammate_id"] == str(driver_id_2)))
        | ((head_to_head["driver_id"] == str(driver_id_2)) & (head_to_head["teammate_id"] == str(driver_id_1)))
    ].copy()

    if direct.empty:
        direct_summary = {
            "races": 0,
            "driver_1_ahead_pct": np.nan,
            "average_race_delta_norm": np.nan,
            "average_qualifying_delta_pct": np.nan,
        }
    else:
        driver_1_rows = direct.loc[direct["driver_id"] == str(driver_id_1)].copy()
        direct_summary = {
            "races": int(driver_1_rows["race_valid"].sum()),
            "driver_1_ahead_pct": float((driver_1_rows["direct_h2h_result"] > 0).mean()) if not driver_1_rows.empty else np.nan,
            "average_race_delta_norm": float(driver_1_rows["race_delta_norm"].mean()) if not driver_1_rows.empty else np.nan,
            "average_qualifying_delta_pct": float(driver_1_rows["qualifying_delta_pct"].mean()) if not driver_1_rows.empty else np.nan,
        }

    return {
        "driver_1": driver_1,
        "driver_2": driver_2,
        "direct_head_to_head": direct_summary,
    }


def plot_tsi_history(driver_ids: list[str]) -> go.Figure:
    """Plot rolling and career TSI history for one or more drivers."""

    driver_race = _load_driver_race()
    filtered = driver_race.loc[driver_race["driver_id"].isin([str(driver_id) for driver_id in driver_ids])].copy()
    figure = go.Figure()
    if filtered.empty:
        return figure

    for driver_id, driver_frame in filtered.groupby("driver_id", sort=False):
        driver_frame = driver_frame.sort_values(["race_date", "season", "round"])
        label = str(driver_frame["driver_name"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=driver_frame["race_date"],
                y=driver_frame["tsi_rolling_20_after"],
                mode="lines",
                name=f"{label} rolling 20",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=driver_frame["race_date"],
                y=driver_frame["tsi_career_after"],
                mode="lines",
                name=f"{label} career",
                line={"dash": "dot"},
            )
        )
    figure.update_layout(
        template="plotly_dark",
        title="TSI History",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font={"family": "Space Grotesk, sans-serif", "color": "#f5f5f5"},
        xaxis_title="Race date",
        yaxis_title="TSI composite (lower is better)",
    )
    return figure


def latest_career_ranking(limit: int = 10) -> pd.DataFrame:
    """Return the latest career ranking table, sorted by composite TSI."""

    summary = _load_career_summary()
    if summary.empty:
        return summary
    return summary.sort_values(["tsi_composite", "tsi_sample_size", "driver_name"], ascending=[True, False, True]).head(int(limit)).reset_index(drop=True)
