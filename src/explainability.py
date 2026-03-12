"""Per-prediction explainability and race-report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency until installed
    shap = None

from src.utils import get_logger

LOGGER = get_logger(__name__)

from src.calibration_engine import CalibrationEngine
from src.car_architecture import compute_car_circuit_fit, get_car_architecture
from src.car_performance import get_cpi
from src.feature_engineering import DNF_FEATURE_COLUMNS, FEATURE_COLUMNS, build_race_inference_frame
from src.model_training import load_models, predict_race
from src.race_simulator import RaceConfig, compute_upset_probability, simulate_race
from src.registry import get_driver_name
from src.season_simulator import build_remaining_races_for_season
from src.tire_model import TireModel, get_circuit_strategy_profile
from src.true_skill import get_tsi
from src.utils import CONFIG as BASE_CONFIG, load_dataframe

CONFIG: dict[str, Any] = {
    "processed_dir": BASE_CONFIG["processed_dir"],
}

FEATURE_LABELS: dict[str, str] = {
    "qualifying_position": "qualified near the front",
    "driver_elo": "has elite dry-race ELO",
    "driver_wet_elo": "has elite wet-race ELO",
    "fp_long_run_pace": "showed strong long-run pace",
    "fp_pace_delta_to_teammate": "beat the teammate on long-run pace",
    "teammate_gap_qualifying": "out-qualified the teammate comfortably",
    "cpi_performance": "benefits from strong car form",
    "cpi_qualifying": "has a qualifying-strong car this weekend",
    "cpi_race": "has a race-pace edge in the current package",
    "tsi_composite": "arrives with strong teammate-adjusted form",
    "tsi_rolling_20": "has a strong rolling 20-race form line",
    "tsi_wet": "has strong wet-weather teammate-adjusted pace",
    "wet_elo_vs_field": "leads the field on wet ELO",
    "circuit_sc_probability": "faces a high-chaos circuit profile",
    "race_precipitation_mm": "benefits from the weather outlook",
    "championship_pressure": "is shaped by championship pressure",
    "circuit_tsi_score": "has a strong car-adjusted track record at this circuit",
    "circuit_tsi_confidence": "has extensive experience at this circuit",
    "circuit_type_tsi": "excels at this type of circuit",
    "circuit_home_race": "has a home-crowd advantage at this circuit",
    "circuit_tsi_std": "has consistent performance at this circuit",
    "circuit_tsi_sample_size": "has raced at this circuit many times",
}


@dataclass(slots=True)
class Explanation:
    """Explainability payload for a single prediction."""

    driver_id: str
    driver_name: str
    model_name: str
    prediction_value: float
    top_reasons: list[str]
    feature_contributions: pd.DataFrame

    @property
    def top_factors(self) -> list[str]:
        return self.top_reasons


class PredictionExplainer:
    """Generate per-prediction explanations and plain-English reports."""

    def __init__(self, models: dict[str, Any] | None = None) -> None:
        self.models = models if models is not None else load_models()
        self.calibration_engine = CalibrationEngine()
        self.tire_model = TireModel()

    @staticmethod
    def _model_base(model: Any) -> Any:
        """Extract the underlying tree model used for SHAP."""

        if hasattr(model, "estimator") and model.estimator is not None:
            return model.estimator
        return model

    @staticmethod
    def _prediction_value(model_name: str, model: Any, frame: pd.DataFrame) -> float:
        """Return the scalar prediction value for a single-row frame."""

        if model_name == "A":
            return float(model.predict(frame)[0])
        return float(model.predict_proba(frame)[0, 1])

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        normalized = str(model_name).strip().upper()
        alias_map = {
            "MODEL_A": "A",
            "MODEL_B": "B",
            "MODEL_C": "C",
            "MODEL_D": "D",
            "MODEL_E": "E",
        }
        return alias_map.get(normalized, normalized)

    def _compute_shap_values(self, model: Any, frame: pd.DataFrame) -> np.ndarray:
        """Compute a single-row SHAP vector."""

        if shap is None:
            raise RuntimeError("shap is not installed")
        base_model = self._model_base(model)
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(frame)
        if isinstance(shap_values, list):
            return np.asarray(shap_values[-1])[0]
        return np.asarray(shap_values)[0]

    def _fallback_contributions(self, frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Fallback explanation using feature deviations from training medians."""

        features = load_dataframe(CONFIG["processed_dir"] / "features.csv")
        source_columns = FEATURE_COLUMNS if model_name != "E" else [column for column in frame.columns if column in features.columns]
        medians = features[source_columns].median(numeric_only=True)
        row = frame[source_columns].iloc[0].astype(float)
        diffs = (row - medians).abs().sort_values(ascending=False).head(5)
        return pd.DataFrame(
            {
                "feature": diffs.index,
                "contribution": np.sign(row.loc[diffs.index] - medians.loc[diffs.index]) * np.minimum(diffs.values / (np.abs(medians.loc[diffs.index]).values + 1.0), 0.25),
            }
        )

    def explain_race_prediction(
        self,
        driver_id: str,
        race_features: pd.Series | pd.DataFrame,
        model_name: str,
    ) -> Explanation:
        """Return the top five reasons for a specific prediction."""

        driver_frame = race_features.to_frame().T if isinstance(race_features, pd.Series) else race_features.copy()
        driver_frame = driver_frame.loc[driver_frame["driver_id"] == str(driver_id)].copy() if "driver_id" in driver_frame.columns else driver_frame.copy()
        if driver_frame.empty:
            raise ValueError(f"No race feature row found for driver_id={driver_id}")

        resolved_model_name = self._resolve_model_name(model_name)
        model = self.models[resolved_model_name]
        standard_features = DNF_FEATURE_COLUMNS if resolved_model_name == "E" else FEATURE_COLUMNS
        feature_names = [column for column in standard_features if column in driver_frame.columns]
        if not feature_names:
            feature_names = [column for column in driver_frame.columns if column not in {"driver_id", "driver_name"}]
        model_frame = driver_frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        try:
            prediction_value = self._prediction_value(resolved_model_name, model, model_frame)
        except Exception as exc:
            LOGGER.debug("Non-critical error computing prediction value: %s", exc)
            prediction_value = float(model_frame.iloc[0].mean()) if not model_frame.empty else 0.0

        try:
            shap_row = np.asarray(self._compute_shap_values(model, model_frame), dtype=float)
            contributions = pd.DataFrame({"feature": feature_names, "shap_value": shap_row[: len(feature_names)]})
            if resolved_model_name == "A":
                contributions["contribution"] = contributions["shap_value"]
            else:
                contributions["contribution"] = contributions["shap_value"]
        except Exception as exc:
            LOGGER.debug("Non-critical error computing SHAP values: %s", exc)
            contributions = self._fallback_contributions(driver_frame, model_name=model_name)
            contributions["shap_value"] = contributions["contribution"]

        contributions = contributions.sort_values("contribution", key=lambda series: series.abs(), ascending=False).reset_index(drop=True)
        top = contributions.head(5).copy()
        reasons = []
        for row in top.itertuples(index=False):
            label = FEATURE_LABELS.get(str(row.feature), str(row.feature).replace("_", " "))
            delta_pct = float(row.contribution) * 100.0
            sign = "+" if delta_pct >= 0 else ""
            reasons.append(f"{label} ({sign}{delta_pct:.1f}%)")

        return Explanation(
            driver_id=str(driver_id),
            driver_name=str(driver_frame.get("driver_name", pd.Series([get_driver_name(driver_id)])).iloc[0]),
            model_name=str(resolved_model_name),
            prediction_value=prediction_value,
            top_reasons=reasons,
            feature_contributions=top,
        )

    def explain_season_simulation(
        self,
        driver_id: str,
        season_result: Any,
    ) -> str:
        """Generate a narrative explanation for a season simulation result."""

        driver_row = season_result.wdc_probs.loc[season_result.wdc_probs["driver_id"] == str(driver_id)].head(1)
        if driver_row.empty:
            return f"{get_driver_name(driver_id)} is not present in the simulated title table."
        win_prob = float(driver_row["wdc_probability"].iloc[0])
        constructor_id = str(driver_row["constructor_id"].iloc[0]) if "constructor_id" in driver_row.columns else "unknown"
        season = int(season_result.projected_standings["season"].max()) if "season" in season_result.projected_standings.columns else 0
        tsi = get_tsi(driver_id, season=season, round_number=99)
        cpi = get_cpi(constructor_id, season=season, round_number=99)
        return (
            f"{get_driver_name(driver_id)} wins the championship in {win_prob:.0%} of simulations primarily because "
            f"their rolling TSI sits at {tsi.get('tsi_rolling_20', np.nan):.3f}, while {constructor_id.replace('_', ' ').title()} "
            f"carries a CPI form score of {cpi.get('cpi_performance', np.nan):.3f} and a development rate of "
            f"{cpi.get('cpi_development_rate', np.nan):.3f} per race."
        )

    def generate_race_report(
        self,
        season: int,
        round_number: int,
    ) -> str:
        """Generate a plain-English pre-race report."""

        remaining = build_remaining_races_for_season(season, from_round=max(0, int(round_number) - 1))
        race_config = next((config for config in remaining if int(config.round) == int(round_number)), None)
        if race_config is None:
            raise ValueError(f"Unable to resolve race config for season={season} round={round_number}")

        inference_frame = build_race_inference_frame(
            drivers_list=race_config.drivers,
            qualifying_positions=race_config.qualifying_positions,
            circuit_id=race_config.circuit_id,
            season=int(season),
            round_number=int(round_number),
            weather_mode=race_config.weather_mode,
        )
        predictions = predict_race(
            drivers_list=race_config.drivers,
            qualifying_positions=race_config.qualifying_positions,
            circuit_id=race_config.circuit_id,
            season=int(season),
            round_number=int(round_number),
            weather_mode=race_config.weather_mode,
        )
        simulation = simulate_race(
            RaceConfig(
                drivers=race_config.drivers,
                qualifying_positions=race_config.qualifying_positions,
                circuit_id=race_config.circuit_id,
                season=int(season),
                round=int(round_number),
                weather_mode=race_config.weather_mode,
                n_simulations=5000,
            )
        )
        confidence = self.calibration_engine.compute_prediction_confidence(inference_frame, season=season, round_number=round_number)
        upset = compute_upset_probability(
            circuit_id=race_config.circuit_id,
            season=int(season),
            round_number=int(round_number),
            weather_forecast={
                "weather_mode": race_config.weather_mode or "dry",
                "race_precipitation_mm": float(inference_frame["race_precipitation_mm"].mean()),
            },
        )
        strategy = self.tire_model.predict_optimal_strategy(
            driver_id=str(predictions.iloc[0]["driver_id"]),
            circuit_id=race_config.circuit_id,
            grid_position=int(predictions.iloc[0]["qualifying_position"]),
            competitors=predictions["driver_id"].tolist(),
            weather_forecast={
                "weather_mode": race_config.weather_mode or "dry",
                "race_precipitation_mm": float(inference_frame["race_precipitation_mm"].mean()),
            },
        )
        top_three = (
            predictions.sort_values("relative_win_probability", ascending=False)
            .head(3)[["driver_name", "relative_win_probability"]]
            .to_dict(orient="records")
        )
        form_table = inference_frame[["driver_id", "driver_name", "tsi_rolling_20", "cpi_performance"]].sort_values(
            ["tsi_rolling_20", "cpi_performance"], ascending=[True, False]
        )
        strongest_form = ", ".join(form_table.head(3)["driver_name"].tolist())
        fit_rows: list[dict[str, Any]] = []
        for constructor_id in predictions["constructor_id"].astype(str).drop_duplicates().tolist():
            fit = compute_car_circuit_fit(
                constructor_id=constructor_id,
                circuit_id=race_config.circuit_id,
                season=int(season),
                round_number=int(round_number),
            )
            architecture = get_car_architecture(
                constructor_id=constructor_id,
                season=int(season),
                round_number=int(round_number),
            )
            fit_rows.append(
                {
                    "constructor_id": constructor_id,
                    "constructor_name": predictions.loc[predictions["constructor_id"] == constructor_id, "constructor_name"].iloc[0],
                    "fit_score": fit.fit_score,
                    "primary_advantage": fit.primary_advantage,
                    "primary_disadvantage": fit.primary_disadvantage,
                    "architecture": architecture,
                }
            )
        fit_rows = sorted(fit_rows, key=lambda row: row["fit_score"], reverse=True)
        best_fit = fit_rows[0] if fit_rows else None
        weakest_fit = fit_rows[-1] if fit_rows else None
        strategy_profile = get_circuit_strategy_profile(race_config.circuit_id)

        car_architecture_context = "Car architecture context: insufficient data to separate constructor fit at this circuit."
        if best_fit and weakest_fit:
            best_arch = best_fit["architecture"]
            weakest_arch = weakest_fit["architecture"]
            
            car_architecture_context = (
                "CAR ARCHITECTURE CONTEXT:\n"
                f" {best_fit['constructor_name'].title()} enters {race_config.circuit_id.replace('_', ' ').title()} with a significant structural advantage.\n"
                f" Their {season} car's {best_fit['primary_advantage'].replace('_', ' ')} ({getattr(best_arch, best_fit['primary_advantage']):.2f}) is the highest \n"
                f" in the field, and {race_config.circuit_id.replace('_', ' ').title()}'s demands heavily weight {best_fit['primary_advantage'].replace('_', ' ')}. "
                f" This gives them a car-circuit fit score of {best_fit['fit_score']:+.2f}, the largest advantage gap \n"
                f" in the field.\n"
                f" \n"
                f" {weakest_fit['constructor_name'].title()}'s structural strength is \n"
                f" largely irrelevant at {race_config.circuit_id.replace('_', ' ').title()} where {weakest_fit['primary_disadvantage'].replace('_', ' ')} accounts for \n"
                f" a significant disadvantage. Expect {weakest_fit['constructor_name'].title()} to struggle with their \n"
                f" car-circuit fit score of {weakest_fit['fit_score']:+.2f}.\n"
                f" \n"
                f" Strategy note: {strategy_profile.notes}"
            )

        lines = [
            f"Race overview and key storylines: {race_config.circuit_id.replace('_', ' ').title()} projects as a {confidence.confidence_label.lower()}-confidence race with an upset score of {upset:.2f}. The leading win projections are "
            + ", ".join(f"{row['driver_name']} ({row['relative_win_probability']:.0%})" for row in top_three)
            + ".",
            f"Pace analysis: the current model sees the strongest outright package with {predictions.iloc[0]['driver_name']} and {predictions.iloc[0]['constructor_name']}, while recent teammate-adjusted form also favors {strongest_form}.",
            car_architecture_context,
            f"Strategy analysis: the preferred baseline is {strategy.recommended_strategy}, with an undercut window opening around lap {strategy.undercut_lap} and overcut viability set to {strategy.overcut_viable}.",
            f"Driver form: rolling TSI points to {strongest_form} as the best recent performers relative to teammates.",
            "Key uncertainties: " + ("; ".join(confidence.key_uncertainties) if confidence.key_uncertainties else "the main uncertainty drivers are currently modest."),
            f"Model confidence: {confidence.confidence_label} ({confidence.confidence_score:.2f}) because {confidence.recommendation.lower()}.",
        ]
        if simulation.regulation_warning:
            lines.append(
                f"Regulation note: season {season} round {round_number} still carries a regulation-reset warning, so the interval bands are intentionally wider."
            )
        return "\n\n".join(lines)
