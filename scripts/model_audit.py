"""Model audit diagnostics for the F1 AI prediction engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fastf1_pipeline import (  # noqa: E402
    OPENF1_FALLBACK_FEATURE_COLUMNS,
    PRACTICE_FEATURE_COLUMNS,
    QUALIFYING_DEEP_FEATURE_COLUMNS,
    SESSION_CONTEXT_COLUMNS,
    SPRINT_FEATURE_COLUMNS,
    TRACK_EVOLUTION_FEATURE_COLUMNS,
)
from src.utils import CONFIG as BASE_CONFIG, ensure_directories, load_dataframe, load_json, save_dataframe, save_json  # noqa: E402

CONFIG: dict[str, Any] = {
    "project_root": PROJECT_ROOT,
    "processed_dir": BASE_CONFIG["processed_dir"],
    "audit_dir": BASE_CONFIG["processed_dir"] / "audit",
    "baseline_gain_dir": BASE_CONFIG["processed_dir"] / "audit" / "baseline_gain",
    "background": "#0a0a0f",
    "foreground": "#f5f5f5",
    "accent": "#e10600",
}


def _chart_layout(figure: go.Figure, title: str) -> go.Figure:
    """Apply consistent styling to audit charts."""

    figure.update_layout(
        template="plotly_dark",
        title=title,
        plot_bgcolor=CONFIG["background"],
        paper_bgcolor=CONFIG["background"],
        font={"family": "Space Grotesk, sans-serif", "color": CONFIG["foreground"]},
        colorway=[CONFIG["accent"], "#ff6b6b", "#ffd166", "#06d6a0", "#118ab2"],
    )
    return figure


def _bucket_label(position: int) -> str:
    """Map a finishing position to an error-analysis bucket."""

    if position <= 5:
        return "P1-P5"
    if position <= 10:
        return "P6-P10"
    if position <= 15:
        return "P11-P15"
    return "P16-P20"


def _load_required_artifacts() -> dict[str, Any]:
    """Load processed artifacts required by the audit."""

    processed = CONFIG["processed_dir"]
    required = {
        "features": processed / "features.csv",
        "metrics": processed / "model_metrics.json",
        "manifest": processed / "model_manifest.json",
        "test_predictions_A": processed / "test_predictions_model_A.csv",
        "test_predictions_B": processed / "test_predictions_model_B.csv",
        "test_predictions_C": processed / "test_predictions_model_C.csv",
        "test_predictions_D": processed / "test_predictions_model_D.csv",
        "test_predictions_E": processed / "test_predictions_model_E.csv",
        "calibration_B": processed / "calibration_model_B.csv",
        "calibration_C": processed / "calibration_model_C.csv",
        "calibration_D": processed / "calibration_model_D.csv",
        "calibration_E": processed / "calibration_model_E.csv",
        "fastf1_failures": processed / "fastf1_failures.csv",
        "openf1_coverage": processed / "openf1_weekend_coverage.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing audit prerequisites: {missing}")

    return {
        "features": load_dataframe(required["features"]),
        "metrics": load_json(required["metrics"]),
        "manifest": load_json(required["manifest"]),
        "test_predictions_A": load_dataframe(required["test_predictions_A"]),
        "test_predictions_B": load_dataframe(required["test_predictions_B"]),
        "test_predictions_C": load_dataframe(required["test_predictions_C"]),
        "test_predictions_D": load_dataframe(required["test_predictions_D"]),
        "test_predictions_E": load_dataframe(required["test_predictions_E"]),
        "calibration_B": load_dataframe(required["calibration_B"]),
        "calibration_C": load_dataframe(required["calibration_C"]),
        "calibration_D": load_dataframe(required["calibration_D"]),
        "calibration_E": load_dataframe(required["calibration_E"]),
        "fastf1_failures": load_dataframe(required["fastf1_failures"]) if required["fastf1_failures"].exists() else pd.DataFrame(),
        "openf1_coverage": load_dataframe(required["openf1_coverage"]) if required["openf1_coverage"].exists() else pd.DataFrame(),
    }


def _save_html_chart(figure: go.Figure, path: Path) -> None:
    """Save a Plotly chart as HTML."""

    path.write_text(figure.to_html(include_plotlyjs="cdn"), encoding="utf-8")


def _coalesce_columns(frame: pd.DataFrame, preferred: str, alternates: list[str]) -> pd.Series:
    """Return the first available column from a preferred/alternate list."""

    for column in [preferred, *alternates]:
        if column in frame.columns:
            return frame[column]
    raise KeyError(f"None of the expected columns are present: {[preferred, *alternates]}")


def _feature_gain_deltas() -> pd.DataFrame:
    """Compare current feature gains against a saved baseline run if present."""

    rows: list[pd.DataFrame] = []
    for label in ["A", "B", "C", "D", "E"]:
        current_path = CONFIG["processed_dir"] / f"feature_importance_model_{label}.csv"
        baseline_path = CONFIG["baseline_gain_dir"] / f"feature_importance_model_{label}.csv"
        if not current_path.exists() or not baseline_path.exists():
            continue

        current = load_dataframe(current_path)
        baseline = load_dataframe(baseline_path)
        current_value = "gain" if "gain" in current.columns else "importance"
        baseline_value = "gain" if "gain" in baseline.columns else "importance"
        merged = (
            current[["feature", current_value]]
            .rename(columns={current_value: "current_gain"})
            .merge(
                baseline[["feature", baseline_value]].rename(columns={baseline_value: "previous_gain"}),
                how="outer",
                on="feature",
            )
            .fillna(0.0)
        )
        merged["model"] = label
        merged["gain_delta"] = merged["current_gain"] - merged["previous_gain"]
        rows.append(merged[["model", "feature", "previous_gain", "current_gain", "gain_delta"]])

    if not rows:
        return pd.DataFrame(columns=["model", "feature", "previous_gain", "current_gain", "gain_delta"])

    return pd.concat(rows, ignore_index=True).sort_values("gain_delta", ascending=False).reset_index(drop=True)


def _aggregate_feature_gains() -> pd.DataFrame:
    """Aggregate feature gains across all saved model-importance files."""

    rows: list[pd.DataFrame] = []
    for label in ["A", "B", "C", "D", "E"]:
        path = CONFIG["processed_dir"] / f"feature_importance_model_{label}.csv"
        if not path.exists():
            continue
        frame = load_dataframe(path)
        value_column = "gain" if "gain" in frame.columns else "importance"
        current = frame[["feature", value_column]].rename(columns={value_column: "gain"})
        current["model"] = label
        rows.append(current)
    if not rows:
        return pd.DataFrame(columns=["feature", "aggregate_gain", "models_present"])
    all_rows = pd.concat(rows, ignore_index=True)
    return (
        all_rows.groupby("feature", as_index=False)
        .agg(aggregate_gain=("gain", "sum"), models_present=("model", "nunique"))
        .sort_values("aggregate_gain", ascending=False)
        .reset_index(drop=True)
    )


def _new_weekend_features() -> list[str]:
    """Return the newly introduced weekend-ingestion feature set."""

    return [
        *PRACTICE_FEATURE_COLUMNS,
        *QUALIFYING_DEEP_FEATURE_COLUMNS,
        *SPRINT_FEATURE_COLUMNS,
        *TRACK_EVOLUTION_FEATURE_COLUMNS,
        *SESSION_CONTEXT_COLUMNS,
        *OPENF1_FALLBACK_FEATURE_COLUMNS,
    ]


def _feature_availability_report(features: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    """Measure weekend-feature sparsity on the training split and attach aggregate gain."""

    split_seasons = manifest.get("split_seasons", {})
    train_max = int(split_seasons.get("train_max", int(features["season"].max()) - 2))
    train_rows = features.loc[features["season"] <= train_max].copy()
    aggregate_gain = _aggregate_feature_gains().set_index("feature") if not features.empty else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature in _new_weekend_features():
        if feature not in features.columns:
            continue
        availability = float(train_rows[feature].notna().mean() * 100.0) if not train_rows.empty else 0.0
        gain_value = float(aggregate_gain.loc[feature, "aggregate_gain"]) if not aggregate_gain.empty and feature in aggregate_gain.index else 0.0
        rows.append(
            {
                "feature": feature,
                "training_availability_pct": availability,
                "aggregate_gain": gain_value,
            }
        )
    return pd.DataFrame(rows).sort_values(["aggregate_gain", "training_availability_pct"], ascending=[False, False]).reset_index(drop=True)


def _session_coverage_report(coverage: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Summarize weekend session coverage and sprint-weekend capture."""

    if coverage.empty:
        empty = pd.DataFrame(columns=["season", "round", "has_fp1", "has_fp2", "has_fp3", "has_quali", "has_race", "has_sprint", "full_core_coverage"])
        return empty, {"full_core_rounds": 0, "sprint_weekends": 0}

    frame = coverage.copy()
    for column in ["has_laps", "has_stints", "has_position", "has_weather"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["session_ready"] = (
        frame["has_laps"].ge(1)
        | frame["has_stints"].ge(1)
        | frame["has_position"].ge(1)
        | frame["has_weather"].ge(1)
    ).astype(int)
    session_flags = (
        frame.assign(flag_value=lambda current: current["session_ready"])
        .pivot_table(index=["season", "round"], columns="session_group", values="flag_value", aggfunc="max", fill_value=0)
        .reset_index()
    )
    rename_map = {
        "FP1": "has_fp1",
        "FP2": "has_fp2",
        "FP3": "has_fp3",
        "QUALI": "has_quali",
        "RACE": "has_race",
        "SPRINT": "has_sprint",
    }
    session_flags = session_flags.rename(columns=rename_map)
    for column in rename_map.values():
        if column not in session_flags.columns:
            session_flags[column] = 0
    session_flags["full_core_coverage"] = (
        session_flags[["has_fp1", "has_fp2", "has_fp3", "has_quali", "has_race"]].sum(axis=1).eq(5).astype(int)
    )
    summary = {
        "full_core_rounds": int(session_flags["full_core_coverage"].sum()),
        "sprint_weekends": int(session_flags["has_sprint"].sum()),
    }
    return session_flags.sort_values(["season", "round"]).reset_index(drop=True), summary


def main() -> None:
    """Run the model audit and save artifacts."""

    ensure_directories([CONFIG["audit_dir"]])
    artifacts = _load_required_artifacts()

    features = artifacts["features"]
    manifest = artifacts["manifest"]
    pred_a = artifacts["test_predictions_A"]
    pred_b = artifacts["test_predictions_B"]
    metrics = artifacts["metrics"]
    coverage = artifacts["openf1_coverage"]

    merged = pred_a.merge(
        features[["season", "round", "driver_id", "driver_name", "race_name", "qualifying_position"]],
        how="left",
        on=["season", "round", "driver_id"],
    )
    merged["driver_name_resolved"] = _coalesce_columns(merged, "driver_name", ["driver_name_x", "driver_name_y"])
    merged["race_name_resolved"] = _coalesce_columns(merged, "race_name", ["race_name_x", "race_name_y"])
    merged["qualifying_position_resolved"] = _coalesce_columns(
        merged,
        "qualifying_position",
        ["qualifying_position_x", "qualifying_position_y"],
    )
    merged["abs_error"] = (merged["predicted_finish_position"] - merged["actual_finish_position"]).abs()
    merged["actual_bucket"] = merged["actual_finish_position"].astype(int).map(_bucket_label)
    merged["race_condition"] = _coalesce_columns(merged, "race_condition", ["race_condition_x", "race_condition_y"]).fillna("dry")
    merged["race_label"] = merged.apply(
        lambda row: f"{str(row['race_name_resolved'])} {int(row['season'])} - {str(row['driver_name_resolved'])}",
        axis=1,
    )

    mae_by_bucket = (
        merged.groupby("actual_bucket", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    mae_by_bucket["actual_bucket"] = pd.Categorical(
        mae_by_bucket["actual_bucket"],
        ["P1-P5", "P6-P10", "P11-P15", "P16-P20"],
        ordered=True,
    )
    mae_by_bucket = mae_by_bucket.sort_values("actual_bucket")
    save_dataframe(mae_by_bucket, CONFIG["audit_dir"] / "mae_by_position_bucket.csv")

    mae_chart = px.bar(
        mae_by_bucket,
        x="actual_bucket",
        y="mae",
        color="mae",
        color_continuous_scale=["#2d0504", CONFIG["accent"]],
    )
    _save_html_chart(
        _chart_layout(mae_chart, "MAE by True Finish Position Bucket"),
        CONFIG["audit_dir"] / "mae_by_position_bucket.html",
    )

    wet_dry_mae = (
        merged.groupby("race_condition", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
        .sort_values("race_condition")
    )
    save_dataframe(wet_dry_mae, CONFIG["audit_dir"] / "mae_by_race_condition.csv")
    wet_dry_chart = px.bar(
        wet_dry_mae,
        x="race_condition",
        y="mae",
        color="mae",
        color_continuous_scale=["#2d0504", CONFIG["accent"]],
    )
    _save_html_chart(
        _chart_layout(wet_dry_chart, "MAE by Race Condition"),
        CONFIG["audit_dir"] / "mae_by_race_condition.html",
        )

    session_coverage, session_coverage_summary = _session_coverage_report(coverage)
    save_dataframe(session_coverage, CONFIG["audit_dir"] / "openf1_session_coverage.csv")

    feature_availability = _feature_availability_report(features, manifest)
    save_dataframe(feature_availability, CONFIG["audit_dir"] / "feature_availability.csv")
    if not feature_availability.empty:
        feature_availability_chart = px.bar(
            feature_availability.head(20).sort_values("training_availability_pct"),
            x="training_availability_pct",
            y="feature",
            orientation="h",
            color="aggregate_gain",
            color_continuous_scale=["#2d0504", CONFIG["accent"]],
        )
        _save_html_chart(
            _chart_layout(feature_availability_chart, "Weekend Feature Availability vs Aggregate Gain"),
            CONFIG["audit_dir"] / "feature_availability.html",
        )

    worst_misses = merged.sort_values("abs_error", ascending=False).head(10)[
        [
            "race_label",
            "race_name_resolved",
            "driver_name_resolved",
            "qualifying_position_resolved",
            "predicted_finish_position",
            "actual_finish_position",
            "abs_error",
        ]
    ].rename(
        columns={
            "race_label": "race_label",
            "race_name_resolved": "race_name",
            "driver_name_resolved": "driver_name",
            "qualifying_position_resolved": "qualifying_position",
        }
    )
    save_dataframe(worst_misses, CONFIG["audit_dir"] / "worst_10_misses.csv")

    for label in ["B", "C", "D"]:
        calibration_frame = artifacts[f"calibration_{label}"]
        calibration_chart = px.line(
            calibration_frame,
            x="prob_pred",
            y="prob_true",
            markers=True,
        )
        calibration_chart.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "#f5f5f5"},
                name="Perfect calibration",
            )
        )
        _save_html_chart(
            _chart_layout(calibration_chart, f"Calibration Curve Model {label}"),
            CONFIG["audit_dir"] / f"calibration_model_{label}.html",
        )

    for label in ["A", "B", "C", "D", "E"]:
        importance = load_dataframe(CONFIG["processed_dir"] / f"feature_importance_model_{label}.csv")
        value_column = "gain" if "gain" in importance.columns else "importance"
        importance_chart = px.bar(
            importance.head(15).sort_values(value_column),
            x=value_column,
            y="feature",
            orientation="h",
            color=value_column,
            color_continuous_scale=["#2d0504", CONFIG["accent"]],
        )
        _save_html_chart(
            _chart_layout(importance_chart, f"Feature Importance Model {label}"),
            CONFIG["audit_dir"] / f"feature_importance_model_{label}.html",
        )

    dnf_summary = {
        "validation_metrics": metrics["E"]["validation_metrics"],
        "test_metrics": metrics["E"]["test_metrics"],
    }
    save_json(dnf_summary, CONFIG["audit_dir"] / "dnf_model_performance.json")
    gain_delta_frame = _feature_gain_deltas()
    if not gain_delta_frame.empty:
        save_dataframe(gain_delta_frame, CONFIG["audit_dir"] / "feature_gain_improvements.csv")

    aggregate_gains = _aggregate_feature_gains()
    top_new_weekend_features = aggregate_gains.loc[
        aggregate_gains["feature"].isin(_new_weekend_features())
    ].head(5)
    if not top_new_weekend_features.empty:
        save_dataframe(top_new_weekend_features, CONFIG["audit_dir"] / "top_new_weekend_features.csv")

    calibration_dnf_chart = px.line(
        artifacts["calibration_E"],
        x="prob_pred",
        y="prob_true",
        markers=True,
    )
    calibration_dnf_chart.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "#f5f5f5"},
            name="Perfect calibration",
        )
    )
    _save_html_chart(
        _chart_layout(calibration_dnf_chart, "Calibration Curve Model E"),
        CONFIG["audit_dir"] / "calibration_model_E.html",
    )

    win_over_30 = pred_b.loc[pred_b["predicted_probability"] > 0.30].copy()
    win_over_30_summary = {
        "count": int(len(win_over_30)),
        "actual_win_rate": float(win_over_30["target"].mean()) if not win_over_30.empty else None,
        "avg_predicted_probability": float(win_over_30["predicted_probability"].mean()) if not win_over_30.empty else None,
    }

    summary = {
        "overall_test_mae": float(pred_a.assign(abs_error=(pred_a["predicted_finish_position"] - pred_a["actual_finish_position"]).abs())["abs_error"].mean()),
        "mae_by_position_bucket": mae_by_bucket.to_dict(orient="records"),
        "mae_by_race_condition": wet_dry_mae.to_dict(orient="records"),
        "win_model_over_30_summary": win_over_30_summary,
        "dnf_model_test_metrics": metrics["E"]["test_metrics"],
        "top_gain_improvements": gain_delta_frame.head(3).to_dict(orient="records"),
        "top_new_weekend_features": top_new_weekend_features.to_dict(orient="records"),
        "feature_availability": feature_availability.to_dict(orient="records"),
        "session_coverage_summary": session_coverage_summary,
        "fastf1_failures": artifacts["fastf1_failures"].to_dict(orient="records"),
    }
    save_json(summary, CONFIG["audit_dir"] / "audit_summary.json")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
