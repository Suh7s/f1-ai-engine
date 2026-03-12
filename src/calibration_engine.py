"""Prediction-confidence calibration and uncertainty reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.prediction_tracker import list_saved_predictions
from src.regulation_handler import RegulationHandler
from src.utils import CONFIG as BASE_CONFIG, load_dataframe, load_json

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "features_path": BASE_CONFIG["processed_dir"] / "features.csv",
}

REGULATION_HANDLER = RegulationHandler()


@dataclass(slots=True)
class ConfidenceReport:
    """Model-confidence report for a race prediction input frame."""

    confidence_score: float
    confidence_label: str
    key_uncertainties: list[str]
    recommendation: str


@dataclass(slots=True)
class CalibrationReport:
    """Empirical calibration summary across saved predictions."""

    summary: pd.DataFrame
    confidence_curve: pd.DataFrame


class CalibrationEngine:
    """Assess prediction confidence before inference and audit it over time."""

    def __init__(self, features_df: pd.DataFrame | None = None) -> None:
        self.features_df = features_df if features_df is not None else load_dataframe(CONFIG["features_path"])

    @staticmethod
    def _confidence_label(score: float) -> str:
        """Map a numeric confidence score to the dashboard label."""

        if score >= 0.78:
            return "HIGH"
        if score >= 0.58:
            return "MEDIUM"
        if score >= 0.38:
            return "LOW"
        return "VERY LOW"

    def compute_prediction_confidence(
        self,
        race_features: pd.DataFrame,
        season: int,
        round_number: int,
    ) -> ConfidenceReport:
        """Assess confidence and key uncertainties before making a prediction."""

        frame = race_features.copy()
        if frame.empty:
            return ConfidenceReport(
                confidence_score=0.20,
                confidence_label="VERY LOW",
                key_uncertainties=["No race feature frame available"],
                recommendation="High variance race, wide margins apply",
            )

        score = 0.50
        uncertainties: list[tuple[float, str]] = []

        session_columns = ["has_fp1", "has_fp2", "has_fp3", "has_quali", "has_race"]
        session_coverage = float(frame[session_columns].mean().mean()) if set(session_columns).issubset(frame.columns) else 0.0
        score += 0.20 * (session_coverage - 0.5)
        if session_coverage < 0.6:
            uncertainties.append((0.20, "Weekend session coverage is incomplete"))

        wet_fraction = float(frame[["is_wet_race", "is_mixed_race"]].max(axis=1).mean()) if {"is_wet_race", "is_mixed_race"}.issubset(frame.columns) else 0.0
        score -= 0.20 * wet_fraction
        if wet_fraction >= 0.5:
            uncertainties.append((0.19, "Rain or mixed conditions reduce historical transferability"))

        season_uncertainty = float(REGULATION_HANDLER.get_season_uncertainty(int(season), int(round_number)))
        score *= season_uncertainty
        if season_uncertainty < 1.0:
            uncertainties.append((1.0 - season_uncertainty, "Early regulation-cycle uncertainty remains elevated"))

        circuit_id = str(frame["circuit_id"].iloc[0]) if "circuit_id" in frame.columns and not frame.empty else None
        historical_rows = self.features_df.loc[self.features_df["circuit_id"] == circuit_id].copy() if circuit_id else pd.DataFrame()
        circuit_representation = float(min(len(historical_rows) / 120.0, 1.0))
        score += 0.10 * (circuit_representation - 0.5)
        if circuit_representation < 0.35:
            uncertainties.append((0.12, "Circuit is lightly represented in the training window"))

        if "fp_long_run_pace" in frame.columns:
            pace = pd.to_numeric(frame["fp_long_run_pace"], errors="coerce").dropna()
            if len(pace) >= 3:
                spread = float(pace.quantile(0.75) - pace.quantile(0.25))
                if spread <= 100.0:
                    score -= 0.12
                    uncertainties.append((0.12, "Practice long-run pace is tightly compressed"))
                else:
                    score += 0.06
        elif "qualifying_position" in frame.columns:
            ordered = pd.to_numeric(frame["qualifying_position"], errors="coerce").dropna().sort_values()
            if len(ordered) >= 3 and float(ordered.iloc[2] - ordered.iloc[0]) <= 2.0:
                score -= 0.08
                uncertainties.append((0.08, "Expected front-running group is tightly packed"))

        if {"fp_preferred_compound_code", "driver_id"}.issubset(frame.columns):
            top_slice = frame.sort_values("qualifying_position").head(6)
            strategy_diversity = int(top_slice["fp_preferred_compound_code"].nunique(dropna=True))
            if strategy_diversity >= 3:
                score -= 0.06
                uncertainties.append((0.06, "Top contenders appear split across tyre approaches"))

        missing_rate = float(frame.isna().mean().mean())
        score -= 0.12 * missing_rate
        if missing_rate >= 0.20:
            uncertainties.append((0.10, "Feature completeness is weaker than ideal"))

        score = float(np.clip(score, 0.05, 0.95))
        label = self._confidence_label(score)
        top_uncertainties = [text for _, text in sorted(uncertainties, key=lambda item: item[0], reverse=True)[:3]]
        if label == "HIGH":
            recommendation = "Trust this prediction"
        elif label == "MEDIUM":
            recommendation = "Treat as directional only"
        else:
            recommendation = "High variance race, wide margins apply"

        return ConfidenceReport(
            confidence_score=score,
            confidence_label=label,
            key_uncertainties=top_uncertainties,
            recommendation=recommendation,
        )

    def track_calibration_over_time(self) -> CalibrationReport:
        """Assess whether saved prediction confidence labels align with realized accuracy."""

        rows: list[dict[str, Any]] = []
        for path in list_saved_predictions():
            payload = load_json(path, default=None)
            if payload is None:
                continue
            confidence_value = payload.get("confidence")
            if confidence_value is None:
                continue
            label = str(confidence_value).upper()
            accuracy = payload.get("accuracy", {})
            if not accuracy:
                continue
            podium_hits = float(accuracy.get("podium_hits", 0.0))
            rows.append(
                {
                    "season": int(payload.get("season", 0)),
                    "round": int(payload.get("round", 0)),
                    "confidence_label": label,
                    "top3_accuracy": podium_hits / 3.0,
                    "winner_correct": float(bool(accuracy.get("winner_correct", False))),
                }
            )
        if not rows:
            empty = pd.DataFrame(columns=["confidence_label", "count", "avg_top3_accuracy", "winner_accuracy"])
            return CalibrationReport(summary=empty, confidence_curve=empty.copy())

        frame = pd.DataFrame(rows)
        order = {"VERY LOW": 0.20, "LOW": 0.40, "MEDIUM": 0.65, "HIGH": 0.85}
        summary = (
            frame.groupby("confidence_label", as_index=False)
            .agg(
                count=("season", "size"),
                avg_top3_accuracy=("top3_accuracy", "mean"),
                winner_accuracy=("winner_correct", "mean"),
            )
        )
        summary["confidence_score"] = summary["confidence_label"].map(order).fillna(0.5)
        summary = summary.sort_values("confidence_score").reset_index(drop=True)
        confidence_curve = summary[["confidence_score", "avg_top3_accuracy", "winner_accuracy"]].copy()
        return CalibrationReport(summary=summary, confidence_curve=confidence_curve)

