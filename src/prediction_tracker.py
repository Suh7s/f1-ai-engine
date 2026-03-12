"""Persistence and scoring of pre-race predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import CONFIG as BASE_CONFIG, ensure_directories, get_gp_name, load_json, points_for_position, save_json, slugify

CONFIG: dict[str, Any] = {
    "predictions_dir": BASE_CONFIG["predictions_dir"],
    "legacy_predictions_dir": BASE_CONFIG["legacy_predictions_dir"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
}


class PredictionTracker:
    """Small OO wrapper around the prediction persistence helpers."""

    def save_prediction(self, payload: dict[str, Any], output_dir: str | Path | None = None) -> Path:
        if output_dir is None:
            season = int(payload.get("season", 0))
            round_number = int(payload.get("round", 0))
            return save_pre_race_prediction(season, round_number, payload)

        output_path = Path(output_dir)
        ensure_directories([output_path])
        file_path = output_path / (
            f"{int(payload.get('season', 0))}_{int(payload.get('round', 0)):02d}_"
            f"{slugify(payload.get('gp_name', payload.get('race_name', 'prediction')))}.json"
        )
        save_json(payload, file_path)
        return file_path

    def load_prediction(self, path: str | Path) -> dict[str, Any] | None:
        return load_json(Path(path), default=None)

    def score_prediction(
        self,
        prediction_payload: dict[str, Any],
        actual_results: pd.DataFrame,
        persist: bool = True,
    ) -> dict[str, Any]:
        return score_prediction(prediction_payload, actual_results, persist=persist)


def prediction_path(season: int, round_number: int) -> Path:
    """Return the canonical pre-race prediction path."""

    gp_slug = slugify(get_gp_name(season, round_number))
    return CONFIG["predictions_dir"] / f"{int(season)}_{gp_slug}_pre_race.json"


def _legacy_prediction_path(season: int, round_number: int) -> Path:
    """Return the legacy round-based prediction path."""

    return CONFIG["legacy_predictions_dir"] / f"{int(season)}_R{int(round_number):02d}_pre_race.json"


def _sorted_prediction_rows(prediction_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return prediction rows sorted by expected finish."""

    rows = list(prediction_payload.get("predictions", []))
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("expected_finish", 999.0)),
            -float(row.get("relative_win_probability", row.get("win_prob", 0.0))),
        ),
    )


def save_pre_race_prediction(
    season: int,
    round_number: int,
    payload: dict[str, Any],
) -> Path:
    """Persist a pre-race prediction payload without applying retention limits."""

    ensure_directories([CONFIG["predictions_dir"]])
    path = prediction_path(season, round_number)
    save_json(payload, path)
    return path


def load_pre_race_prediction(season: int, round_number: int) -> dict[str, Any] | None:
    """Load a saved pre-race prediction if it exists."""

    primary = prediction_path(season, round_number)
    if primary.exists():
        return load_json(primary, default=None)
    legacy = _legacy_prediction_path(season, round_number)
    return load_json(legacy, default=None)


def score_prediction(
    prediction_payload: dict[str, Any],
    actual_results: pd.DataFrame,
    persist: bool = True,
) -> dict[str, Any]:
    """Score a prediction payload against actual race results."""

    actual = actual_results.copy().sort_values(["finish_position", "grid_position"])
    actual = actual.loc[actual["driver_id"].notna()].copy()
    actual_top5 = actual.head(5)["driver_id"].tolist()
    actual_podium = actual.head(3)["driver_id"].tolist()
    actual_winner = actual_top5[0] if actual_top5 else None
    actual_positions = actual.set_index("driver_id")["finish_position"].to_dict()

    predicted_rows = _sorted_prediction_rows(prediction_payload)
    predicted_top5 = [row["driver_id"] for row in predicted_rows[:5]]
    predicted_podium = predicted_top5[:3]
    predicted_winner = predicted_top5[0] if predicted_top5 else None

    mean_position_error = float(
        np.mean(
            [
                abs(float(row.get("expected_finish", np.nan)) - float(actual_positions.get(row["driver_id"], np.nan)))
                for row in predicted_rows
                if row["driver_id"] in actual_positions and pd.notna(row.get("expected_finish"))
            ]
        )
    )

    win_prob_lookup = {row["driver_id"]: float(row.get("relative_win_probability", row.get("win_prob", 0.0))) for row in predicted_rows}
    brier_score = float(
        np.mean(
            [
                (win_prob_lookup.get(driver_id, 0.0) - float(driver_id == actual_winner)) ** 2
                for driver_id in actual["driver_id"].tolist()
            ]
        )
    ) if not actual.empty else 0.0

    scoring = {
        "season": int(prediction_payload.get("season", 0)),
        "round": int(prediction_payload.get("round", 0)),
        "race_name": prediction_payload.get("race_name"),
        "win_correct": bool(predicted_winner == actual_winner),
        "podium_correct": int(sum(driver_id in actual_podium for driver_id in predicted_podium)),
        "top5_positions_correct": int(
            sum(
                predicted_driver == actual_driver
                for predicted_driver, actual_driver in zip(predicted_top5, actual_top5, strict=False)
            )
        ),
        "mean_position_error": mean_position_error,
        "brier_score": brier_score,
        "predicted_winner": predicted_winner,
        "actual_winner": actual_winner,
        "predicted_top5": predicted_top5,
        "actual_top5": actual_top5,
    }
    enriched_payload = dict(prediction_payload)
    enriched_payload["accuracy"] = scoring
    if persist:
        save_pre_race_prediction(scoring["season"], scoring["round"], enriched_payload)
    return scoring


def list_saved_predictions(season: int | None = None) -> list[Path]:
    """Return saved prediction files, optionally filtered by season."""

    ensure_directories([CONFIG["predictions_dir"], CONFIG["legacy_predictions_dir"]])
    paths = sorted(set(CONFIG["predictions_dir"].glob("*_pre_race.json")) | set(CONFIG["legacy_predictions_dir"].glob("*_pre_race.json")))
    if season is None:
        return paths
    prefix = f"{int(season)}_"
    return [path for path in paths if path.name.startswith(prefix)]


def aggregate_season_accuracy(season: int) -> dict[str, Any]:
    """Aggregate saved prediction accuracy across a season."""

    rows: list[dict[str, Any]] = []
    for path in list_saved_predictions(season):
        payload = load_json(path, default=None)
        if payload is None or "accuracy" not in payload:
            continue
        rows.append(dict(payload["accuracy"]))

    if not rows:
        return {
            "season": int(season),
            "season_win_accuracy": 0.0,
            "season_podium_accuracy": 0.0,
            "season_calibration": 0.0,
            "best_prediction": None,
            "worst_prediction": None,
            "rolling_accuracy": [],
            "per_race": [],
        }

    frame = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
    frame["winner_hit"] = frame["win_correct"].astype(int)
    frame["rolling_win_accuracy_last_5"] = frame["winner_hit"].rolling(5, min_periods=1).mean()

    best_prediction_row = frame.sort_values(["mean_position_error", "brier_score"]).iloc[0].to_dict()
    worst_prediction_row = frame.sort_values(["mean_position_error", "brier_score"], ascending=[False, False]).iloc[0].to_dict()

    calibration_rows: list[dict[str, Any]] = []
    for path in list_saved_predictions(season):
        payload = load_json(path, default=None)
        if payload is None:
            continue
        accuracy = payload.get("accuracy", {})
        predictions = payload.get("predictions", [])
        actual_winner = accuracy.get("actual_winner")
        for row in predictions:
            calibration_rows.append(
                {
                    "round": int(payload.get("round", 0)),
                    "driver_id": row["driver_id"],
                    "predicted": float(row.get("relative_win_probability", row.get("win_prob", 0.0))),
                    "actual": float(row["driver_id"] == actual_winner),
                }
            )

    calibration_score = 0.0
    if calibration_rows:
        calibration_frame = pd.DataFrame(calibration_rows)
        calibration_score = float(np.mean((calibration_frame["predicted"] - calibration_frame["actual"]) ** 2))

    return {
        "season": int(season),
        "season_win_accuracy": float(frame["winner_hit"].mean()),
        "season_podium_accuracy": float(frame["podium_correct"].mean()),
        "season_calibration": calibration_score,
        "best_prediction": best_prediction_row,
        "worst_prediction": worst_prediction_row,
        "rolling_accuracy": frame[["round", "rolling_win_accuracy_last_5"]].to_dict(orient="records"),
        "per_race": frame.to_dict(orient="records"),
        "races_scored": int(len(frame)),
    }


def build_race_accuracy_report(season: int, round_number: int) -> str:
    """Return a short text summary for a post-race prediction score."""

    payload = load_pre_race_prediction(season, round_number)
    if payload is None or "accuracy" not in payload:
        return "No saved pre-race prediction found for this race."

    accuracy = payload["accuracy"]
    predicted_top3 = ", ".join(accuracy.get("predicted_top5", [])[:3])
    actual_top3 = ", ".join(accuracy.get("actual_top5", [])[:3])
    return (
        f"We predicted: {predicted_top3}. "
        f"Actual: {actual_top3}. "
        f"Win prediction: {'CORRECT' if accuracy.get('win_correct') else 'WRONG'}. "
        f"Podium: {int(accuracy.get('podium_correct', 0))}/3 correct."
    )
