"""Tire degradation modelling and strategy recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import CIRCUIT_DNA
from src.car_performance import get_cpi
from src.utils import CONFIG as BASE_CONFIG, get_logger, load_dataframe

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "raw_dir": BASE_CONFIG["raw_dir"],
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "openf1_stints_path": BASE_CONFIG["processed_dir"] / "openf1_stints.csv",
    "lap_times_path": BASE_CONFIG["raw_dir"] / "lap_times_all.csv",
}

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class DegradationCurve:
    """Fitted tyre-degradation curve for a circuit-compound pair."""

    circuit_id: str
    compound: str
    model_type: str
    base_pace: float
    deg_rate: float
    cliff_lap: int | None
    r_squared: float


@dataclass(slots=True)
class CompoundProfile:
    """Historical compound profile for a circuit."""

    circuit_id: str
    compound_metrics: pd.DataFrame
    best_strategic_compound: str | None
    typical_winning_strategy: str


@dataclass(slots=True)
class StrategyPrediction:
    """Recommended race strategy output."""

    driver_id: str
    circuit_id: str
    recommended_strategy: list[tuple[int, str]]
    alternative_strategy: list[tuple[int, str]]
    undercut_lap: int
    overcut_viable: bool
    strategy_confidence: float


@dataclass(slots=True)
class StrategyProfile:
    """Circuit-level strategic baseline."""

    typical_stops: int
    undercut_power: float
    overcut_power: float
    safety_car_probability: float
    sc_impact_on_strategy: str
    optimal_start_compound: str
    pit_window_open: int
    pit_window_close: int
    drs_dependency: float
    notes: str = ""


@dataclass(slots=True)
class StrategyFit:
    """How well a constructor's strengths fit a circuit strategy profile."""

    score: float
    recommended_approach: str
    sc_strategy_readiness: float
    pit_window_recommendation: tuple[int, int]
    overcut_fit: float
    undercut_fit: float


_EXPLICIT_CIRCUIT_STRATEGY_PROFILE: dict[str, StrategyProfile] = {
    "monza": StrategyProfile(1, 0.30, 0.70, 0.45, "HIGH", "MEDIUM", 18, 35, 0.95, "Track position critical. SC likely. 1-stop baseline."),
    "monaco": StrategyProfile(1, 0.95, 0.10, 0.75, "EXTREME", "MEDIUM", 20, 50, 0.10, "Qualifying position dominates. Undercut or nothing."),
    "silverstone": StrategyProfile(2, 0.65, 0.55, 0.50, "MEDIUM", "MEDIUM", 14, 28, 0.60, "High tyre deg makes the 2-stop the normal play."),
    "spa": StrategyProfile(1, 0.75, 0.60, 0.65, "HIGH", "MEDIUM", 16, 32, 0.75, "Weather wildcard and long pit straight make timing critical."),
    "baku": StrategyProfile(1, 0.80, 0.40, 0.80, "EXTREME", "MEDIUM", 20, 40, 0.85, "SC almost guaranteed. Strategy built around interruption timing."),
}


def _heuristic_strategy_profile(circuit_id: str) -> StrategyProfile:
    metadata = CIRCUIT_DNA.get(
        str(circuit_id),
        {
            "high_speed_pct": 0.45,
            "tyre_deg_severity": 3.0,
            "sc_probability": 0.40,
            "drs_zones": 2.0,
            "street_circuit": False,
        },
    )
    tyre_deg = float(metadata["tyre_deg_severity"])
    sc_probability = float(metadata["sc_probability"])
    drs_dependency = float(np.clip(0.25 + 0.18 * float(metadata["drs_zones"]) + 0.10 * float(metadata["high_speed_pct"]), 0.10, 0.95))
    typical_stops = 2 if tyre_deg >= 4.0 else 1
    undercut_power = float(np.clip(0.35 + 0.25 * float(metadata["street_circuit"]) + 0.10 * tyre_deg, 0.20, 0.95))
    overcut_power = float(np.clip(0.30 + 0.14 * tyre_deg - 0.08 * float(metadata["street_circuit"]), 0.10, 0.85))
    sc_impact = "EXTREME" if sc_probability >= 0.70 else "HIGH" if sc_probability >= 0.55 else "MEDIUM"
    pit_window_open = 14 if typical_stops == 2 else 18
    pit_window_close = 28 if typical_stops == 2 else 36
    optimal_start = "MEDIUM" if tyre_deg >= 3.0 else "HARD"
    return StrategyProfile(
        typical_stops=int(typical_stops),
        undercut_power=undercut_power,
        overcut_power=overcut_power,
        safety_car_probability=sc_probability,
        sc_impact_on_strategy=sc_impact,
        optimal_start_compound=optimal_start,
        pit_window_open=int(pit_window_open),
        pit_window_close=int(pit_window_close),
        drs_dependency=drs_dependency,
        notes="Heuristic strategy profile derived from circuit DNA",
    )


CIRCUIT_STRATEGY_PROFILE: dict[str, StrategyProfile] = {
    circuit_id: _EXPLICIT_CIRCUIT_STRATEGY_PROFILE.get(circuit_id, _heuristic_strategy_profile(circuit_id))
    for circuit_id in sorted(CIRCUIT_DNA.keys())
}


def get_circuit_strategy_profile(circuit_id: str) -> StrategyProfile:
    """Return the strategic baseline for a circuit."""

    return CIRCUIT_STRATEGY_PROFILE.get(str(circuit_id), _heuristic_strategy_profile(str(circuit_id)))


def compute_strategy_fit(
    constructor_id: str,
    circuit_id: str,
    season: int,
    round_number: int,
) -> StrategyFit:
    """How well a constructor's strengths match the strategic demands of a circuit."""

    from src.car_architecture import get_car_architecture

    architecture = get_car_architecture(constructor_id=str(constructor_id), season=int(season), round_number=int(round_number))
    strategy = get_circuit_strategy_profile(circuit_id)
    cpi_snapshot = get_cpi(constructor_id=str(constructor_id), season=int(season), round_number=int(round_number))
    reliability = float(np.clip(cpi_snapshot.get("cpi_reliability", np.nan), 0.0, 1.0))
    if np.isnan(reliability):
        reliability = 0.85

    overcut_fit = float(np.clip(architecture.tyre_management * strategy.overcut_power, 0.0, 1.0))
    undercut_fit = float(np.clip(architecture.straight_line_speed * strategy.undercut_power, 0.0, 1.0))
    sc_benefit = float(np.clip(reliability * strategy.safety_car_probability, 0.0, 1.0))
    score = float(np.clip((overcut_fit + undercut_fit + sc_benefit) / 3.0, 0.0, 1.0))
    recommended = "undercut" if undercut_fit > overcut_fit else "overcut"

    return StrategyFit(
        score=score,
        recommended_approach=recommended,
        sc_strategy_readiness=sc_benefit,
        pit_window_recommendation=(int(strategy.pit_window_open), int(strategy.pit_window_close)),
        overcut_fit=overcut_fit,
        undercut_fit=undercut_fit,
    )


class TireModel:
    """Fit tyre-degradation curves and derive strategy recommendations."""

    def __init__(
        self,
        race_results: pd.DataFrame | None = None,
        stints_df: pd.DataFrame | None = None,
        lap_times_df: pd.DataFrame | None = None,
    ) -> None:
        self.race_results = race_results if race_results is not None else load_dataframe(CONFIG["race_results_path"])
        self.stints_df = stints_df if stints_df is not None else load_dataframe(CONFIG["openf1_stints_path"])
        self.lap_times_df = lap_times_df if lap_times_df is not None else load_dataframe(CONFIG["lap_times_path"])
        self._stint_lap_frame = self._build_stint_lap_frame()

    def _build_stint_lap_frame(self) -> pd.DataFrame:
        """Attach driver ids and circuit ids to stint intervals and expand to stint laps."""

        if self.stints_df.empty or self.lap_times_df.empty:
            return pd.DataFrame(
                columns=[
                    "season",
                    "round",
                    "circuit_id",
                    "driver_id",
                    "compound",
                    "stint_number",
                    "stint_lap",
                    "lap_time_ms",
                ]
            )

        driver_lookup = self.race_results[
            ["season", "round", "driver_number", "driver_id", "circuit_id"]
        ].dropna(subset=["driver_id"]).copy()
        driver_lookup["driver_number"] = pd.to_numeric(driver_lookup["driver_number"], errors="coerce")
        stints = self.stints_df.copy()
        stints["driver_number"] = pd.to_numeric(stints["driver_number"], errors="coerce")
        stints = stints.merge(
            driver_lookup,
            how="left",
            on=["season", "round", "driver_number"],
        )
        stints = stints.dropna(subset=["driver_id", "circuit_id"]).copy()

        laps = self.lap_times_df.copy()
        laps["lap"] = pd.to_numeric(laps["lap"], errors="coerce")
        laps["lap_time_ms"] = pd.to_numeric(laps["lap_time_ms"], errors="coerce")
        merged = stints.merge(
            laps[["season", "round", "circuit_id", "driver_id", "lap", "lap_time_ms"]],
            how="left",
            on=["season", "round", "circuit_id", "driver_id"],
        )
        merged = merged.loc[
            merged["lap"].between(
                pd.to_numeric(merged["lap_start"], errors="coerce"),
                pd.to_numeric(merged["lap_end"], errors="coerce"),
                inclusive="both",
            )
        ].copy()
        merged["stint_lap"] = (
            pd.to_numeric(merged["lap"], errors="coerce")
            - pd.to_numeric(merged["lap_start"], errors="coerce")
            + 1.0
        )
        merged["compound"] = merged["compound"].fillna("UNKNOWN").astype(str).str.upper()
        return merged.dropna(subset=["lap_time_ms", "stint_lap"]).reset_index(drop=True)

    @staticmethod
    def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Return coefficient of determination."""

        if y_true.size == 0:
            return float("nan")
        ss_res = float(np.sum(np.square(y_true - y_pred)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        if ss_tot <= 1e-9:
            return 1.0
        return 1.0 - ss_res / ss_tot

    def fit_degradation_curve(
        self,
        stint_laps: pd.DataFrame,
        compound: str,
        circuit_id: str,
    ) -> DegradationCurve:
        """Fit a degradation curve for a circuit-compound sample."""

        frame = stint_laps.copy()
        if frame.empty:
            return DegradationCurve(
                circuit_id=str(circuit_id),
                compound=str(compound).upper(),
                model_type="linear",
                base_pace=np.nan,
                deg_rate=np.nan,
                cliff_lap=None,
                r_squared=np.nan,
            )

        series = (
            frame.groupby("stint_lap", as_index=False)["lap_time_ms"]
            .median()
            .sort_values("stint_lap")
            .reset_index(drop=True)
        )
        x = series["stint_lap"].to_numpy(dtype=float)
        y = series["lap_time_ms"].to_numpy(dtype=float)
        base_pace = float(np.nanmin(y))
        compound_upper = str(compound).upper()

        if compound_upper == "SOFT" and len(series) >= 4:
            excess = np.clip(y - base_pace + 1.0, 1.0, None)
            slope, intercept = np.polyfit(x, np.log(excess), deg=1)
            predicted = base_pace - 1.0 + np.exp(intercept + slope * x)
            incremental = np.diff(predicted)
            cliff_candidates = np.where(incremental > np.median(incremental) * 1.5)[0]
            cliff_lap = int(x[cliff_candidates[0] + 1]) if cliff_candidates.size else None
            deg_rate = float(slope)
            model_type = "exponential"
        else:
            slope, intercept = np.polyfit(x, y, deg=1)
            predicted = intercept + slope * x
            deg_rate = float(slope)
            step_change = np.diff(y)
            cliff_candidates = np.where(step_change > np.median(step_change) + np.std(step_change))[0]
            cliff_lap = int(x[cliff_candidates[0] + 1]) if cliff_candidates.size else None
            model_type = "linear"

        return DegradationCurve(
            circuit_id=str(circuit_id),
            compound=compound_upper,
            model_type=model_type,
            base_pace=base_pace,
            deg_rate=deg_rate,
            cliff_lap=cliff_lap,
            r_squared=self._r_squared(y, predicted),
        )

    def get_circuit_compound_profile(self, circuit_id: str) -> CompoundProfile:
        """Return the historical compound profile for a circuit."""

        frame = self._stint_lap_frame.loc[self._stint_lap_frame["circuit_id"] == str(circuit_id)].copy()
        if frame.empty:
            return CompoundProfile(
                circuit_id=str(circuit_id),
                compound_metrics=pd.DataFrame(),
                best_strategic_compound=None,
                typical_winning_strategy="No historical stint data available",
            )

        curves: list[dict[str, Any]] = []
        for compound, compound_frame in frame.groupby("compound", sort=False):
            curve = self.fit_degradation_curve(compound_frame, compound=compound, circuit_id=circuit_id)
            curves.append(
                {
                    "compound": curve.compound,
                    "model_type": curve.model_type,
                    "base_pace": curve.base_pace,
                    "deg_rate": curve.deg_rate,
                    "cliff_lap": curve.cliff_lap,
                    "r_squared": curve.r_squared,
                }
            )
        compound_metrics = pd.DataFrame(curves).sort_values(["deg_rate", "base_pace"]).reset_index(drop=True)

        best_compound = None
        if not compound_metrics.empty:
            score = compound_metrics["deg_rate"].rank(ascending=True, method="dense") + compound_metrics["base_pace"].rank(ascending=True, method="dense")
            best_compound = str(compound_metrics.loc[score.idxmin(), "compound"])

        winners = (
            self.race_results.loc[
                (self.race_results["circuit_id"] == str(circuit_id))
                & pd.to_numeric(self.race_results["finish_position"], errors="coerce").eq(1)
            ][["season", "round", "driver_id"]]
            .drop_duplicates()
        )
        winning_stints = self.stints_df.merge(
            winners,
            how="inner",
            on=["season", "round"],
        ).copy()
        if not winning_stints.empty:
            strategy = (
                winning_stints.sort_values(["season", "round", "stint_number"])["compound"]
                .astype(str)
                .str.upper()
                .groupby([winning_stints["season"], winning_stints["round"]])
                .agg(lambda values: " -> ".join(values))
                .mode()
            )
            typical_strategy = str(strategy.iloc[0]) if not strategy.empty else "No clear historical mode"
        else:
            typical_strategy = "No clear historical mode"

        return CompoundProfile(
            circuit_id=str(circuit_id),
            compound_metrics=compound_metrics,
            best_strategic_compound=best_compound,
            typical_winning_strategy=typical_strategy,
        )

    def predict_optimal_strategy(
        self,
        driver_id: str,
        circuit_id: str,
        grid_position: int,
        competitors: list[str],
        weather_forecast: dict[str, Any],
    ) -> StrategyPrediction:
        """Predict a race strategy using historical degradation profiles and grid context."""

        profile = self.get_circuit_compound_profile(circuit_id)
        strategy_profile = get_circuit_strategy_profile(circuit_id)
        precip = float(weather_forecast.get("race_precipitation_mm", weather_forecast.get("precipitation_mm", 0.0)) or 0.0)
        wet_mode = str(weather_forecast.get("weather_mode", "")).lower() in {"wet", "mixed"} or precip >= 1.0

        if wet_mode:
            recommended = [(0, "INTERMEDIATE"), (max(strategy_profile.pit_window_open, 22 - int(grid_position)), "INTERMEDIATE")]
            alternative = [(0, "WET"), (max(strategy_profile.pit_window_open - 2, 20 - int(grid_position)), "INTERMEDIATE")]
            undercut_lap = max(strategy_profile.pit_window_open - 4, 12)
            overcut_viable = False
            confidence = 0.45
        else:
            compound_metrics = profile.compound_metrics.copy()
            if compound_metrics.empty:
                recommended = [(0, strategy_profile.optimal_start_compound), (strategy_profile.pit_window_open + 2, "HARD")]
                alternative = [(0, "HARD"), (strategy_profile.pit_window_close, "MEDIUM")]
                if strategy_profile.typical_stops >= 2:
                    recommended = [
                        (0, strategy_profile.optimal_start_compound),
                        (strategy_profile.pit_window_open, "HARD"),
                        (min(strategy_profile.pit_window_close + 14, strategy_profile.pit_window_close + 18), "MEDIUM"),
                    ]
                    alternative = [(0, "HARD"), (strategy_profile.pit_window_open + 10, "MEDIUM")]
                undercut_lap = max(strategy_profile.pit_window_open - 2, 12)
                overcut_viable = strategy_profile.overcut_power >= 0.50
                confidence = 0.35
            else:
                preferred = profile.best_strategic_compound or str(compound_metrics.iloc[0]["compound"])
                stable = (
                    compound_metrics.sort_values(["deg_rate", "cliff_lap", "base_pace"], ascending=[True, True, True])
                    .iloc[0]["compound"]
                )
                deg_row = compound_metrics.loc[compound_metrics["compound"] == preferred].head(1)
                cliff_lap = int(deg_row["cliff_lap"].iloc[0]) if not deg_row.empty and pd.notna(deg_row["cliff_lap"].iloc[0]) else 18
                two_stop = int(strategy_profile.typical_stops >= 2 or (len(competitors) >= 18 and cliff_lap <= 16))
                first_stop = int(np.clip(max(strategy_profile.pit_window_open, cliff_lap - 2), strategy_profile.pit_window_open, strategy_profile.pit_window_close))
                if two_stop:
                    second_stop = min(first_stop + 16, strategy_profile.pit_window_close + 14)
                    recommended = [(0, preferred), (first_stop, "MEDIUM"), (second_stop, str(stable))]
                    alternative = [(0, strategy_profile.optimal_start_compound), (first_stop + 5, str(stable))]
                else:
                    recommended = [(0, preferred), (first_stop, str(stable))]
                    alternative = [(0, "HARD"), (max(first_stop + 8, strategy_profile.pit_window_close), "MEDIUM")]
                undercut_lap = max(strategy_profile.pit_window_open - 1, first_stop - 3, 10)
                overcut_viable = bool(strategy_profile.overcut_power >= 0.50 and cliff_lap >= 20 and int(grid_position) <= 6)
                confidence = float(
                    np.clip(
                        0.55
                        + 0.05 * bool(profile.best_strategic_compound)
                        + 0.05 * (strategy_profile.typical_stops == 1)
                        + 0.02 * overcut_viable,
                        0.0,
                        0.85,
                    )
                )

        return StrategyPrediction(
            driver_id=str(driver_id),
            circuit_id=str(circuit_id),
            recommended_strategy=recommended,
            alternative_strategy=alternative,
            undercut_lap=int(undercut_lap),
            overcut_viable=bool(overcut_viable),
            strategy_confidence=float(confidence),
        )
