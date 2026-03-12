"""Model training and inference for F1 race predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from xgboost import XGBClassifier, XGBRegressor

from src.feature_engineering import DNF_FEATURE_COLUMNS, FEATURE_COLUMNS, build_race_inference_frame
from src.utils import (
    CONFIG as BASE_CONFIG,
    describe_season_split,
    ensure_directories,
    get_logger,
    load_dataframe,
    load_json,
    normalize_probabilities,
    save_dataframe,
    save_json,
    validate_columns,
)


def _safe_grid_search_jobs() -> int:
    """Use parallel grid search only when the runtime supports loky process discovery."""

    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, PermissionError, OSError, ValueError):
        return 1
    return -1

CONFIG: dict[str, Any] = {
    "project_root": BASE_CONFIG["project_root"],
    "processed_dir": BASE_CONFIG["processed_dir"],
    "features_path": BASE_CONFIG["processed_dir"] / "features.csv",
    "pitstops_path": BASE_CONFIG["processed_dir"] / "pitstops.csv",
    "race_results_path": BASE_CONFIG["processed_dir"] / "race_results.csv",
    "metrics_path": BASE_CONFIG["processed_dir"] / "model_metrics.json",
    "manifest_path": BASE_CONFIG["processed_dir"] / "model_manifest.json",
    "incremental_state_path": BASE_CONFIG["processed_dir"] / "incremental_training_state.json",
    "random_state": 42,
    "cv_splits": 5,
    "grid_search_jobs": _safe_grid_search_jobs(),
    "param_grid": {
        "n_estimators": [200, 400],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.08],
        "subsample": [0.8, 1.0],
    },
}

MODEL_PATHS: dict[str, Path] = {
    "A": CONFIG["processed_dir"] / "model_A.pkl",
    "B": CONFIG["processed_dir"] / "model_B.pkl",
    "C": CONFIG["processed_dir"] / "model_C.pkl",
    "D": CONFIG["processed_dir"] / "model_D.pkl",
    "E": CONFIG["processed_dir"] / "model_E.pkl",
}

BASE_MODEL_PATHS: dict[str, Path] = {
    "B": CONFIG["processed_dir"] / "model_B_base.pkl",
    "C": CONFIG["processed_dir"] / "model_C_base.pkl",
    "D": CONFIG["processed_dir"] / "model_D_base.pkl",
    "E": CONFIG["processed_dir"] / "model_E_base.pkl",
}

LOGGER = get_logger(__name__)


def get_current_season() -> int:
    """Return the current F1 season year using runtime evaluation."""

    now = datetime.now()
    if now.month < 2:
        return now.year - 1
    return now.year


def _current_season_window() -> list[int]:
    """Return the rolling training window for the live season."""

    current_season = get_current_season()
    return list(range(current_season - 10, current_season + 1))

MODEL_TARGETS: dict[str, str] = {
    "A": "finish_position",
    "B": "top1_finish",
    "C": "top3_finish",
    "D": "top10_finish",
    "E": "dnf_flag",
}

TARGET_COLUMNS: list[str] = [
    "finish_position",
    "top1_finish",
    "top3_finish",
    "top10_finish",
    "dnf_flag",
]


@dataclass(slots=True)
class ModelResult:
    """Container for a trained model and its evaluation metadata."""

    label: str
    model: Any
    best_params: dict[str, Any]
    cv_best_score: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance_path: Path
    feature_names: list[str]
    base_model: Any | None = None
    calibration_bucket_path: Path | None = None


def _prepare_feature_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce the saved feature matrix."""

    validate_columns(
        features_df,
        [
            "season",
            "round",
            "race_date",
            *FEATURE_COLUMNS,
            *DNF_FEATURE_COLUMNS,
            "finish_position",
            "top1_finish",
            "top3_finish",
            "top10_finish",
            "dnf_flag",
        ],
        "training feature frame",
    )
    prepared = features_df.copy()
    prepared["race_date"] = pd.to_datetime(prepared["race_date"], errors="coerce")
    if prepared["race_date"].isna().any():
        raise ValueError("Feature frame contains null race_date values required for recency weighting")
    for column in set(FEATURE_COLUMNS + DNF_FEATURE_COLUMNS):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["dnf_flag"] = pd.to_numeric(prepared["dnf_flag"], errors="coerce").fillna(0).astype(int)
    return prepared


def _windowed_feature_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    """Filter the feature matrix to the rolling living-season training window."""

    season_window = _current_season_window()
    window_start = season_window[0]
    window_end = season_window[-1]
    frame = features_df.loc[features_df["season"].between(window_start, window_end)].copy()
    if frame.empty:
        raise ValueError(f"No feature rows found in rolling window {window_start}-{window_end}")
    return frame


def get_evaluation_season(race_results: pd.DataFrame) -> int:
    """Pick a holdout season with enough completed races for reliable evaluation."""

    if race_results.empty:
        raise ValueError("Race results are required to determine the evaluation season")

    current_season = int(pd.to_numeric(race_results["season"], errors="coerce").dropna().max())
    current_season_rounds = int(
        race_results.loc[pd.to_numeric(race_results["season"], errors="coerce") == current_season, "round"]
        .dropna()
        .astype(int)
        .nunique()
    )
    if current_season_rounds < 5:
        evaluation_season = current_season - 1
        LOGGER.info(
            "Current season %s has only %s completed rounds. Using %s as the holdout test season.",
            current_season,
            current_season_rounds,
            evaluation_season,
        )
        return evaluation_season
    return current_season


def build_train_test_split(
    features: pd.DataFrame,
    race_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train/validation/test matrices using a holdout season with enough rows."""

    test_season = int(get_evaluation_season(race_results))
    val_season = test_season - 1

    test_mask = features["season"] == test_season
    val_mask = features["season"] == val_season
    train_mask = features["season"] < val_season

    X_train = features.loc[train_mask, FEATURE_COLUMNS].copy()
    X_val = features.loc[val_mask, FEATURE_COLUMNS].copy()
    X_test = features.loc[test_mask, FEATURE_COLUMNS].copy()

    y_train = features.loc[train_mask, TARGET_COLUMNS].copy()
    y_val = features.loc[val_mask, TARGET_COLUMNS].copy()
    y_test = features.loc[test_mask, TARGET_COLUMNS].copy()

    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError(
            "Training, validation, and test splits must all be non-empty | "
            f"train={len(X_train)} val={len(X_val)} test={len(X_test)}"
        )

    train_seasons = features.loc[train_mask, "season"].dropna().astype(int)
    LOGGER.info(
        "Split: train=%s rows (%s-%s), val=%s rows (%s), test=%s rows (%s)",
        len(X_train),
        int(train_seasons.min()),
        int(train_seasons.max()),
        len(X_val),
        val_season,
        len(X_test),
        test_season,
    )

    if len(X_test) < 100:
        raise ValueError(
            f"Test set has only {len(X_test)} rows (season {test_season}). "
            "Minimum 100 required for reliable metrics."
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _split_datasets(features_df: pd.DataFrame, race_results_df: pd.DataFrame) -> dict[str, Any]:
    """Split features into train, validation, and test partitions by season."""

    windowed = _windowed_feature_frame(features_df)
    test_season = int(get_evaluation_season(race_results_df))
    available = sorted(windowed["season"].dropna().astype(int).unique().tolist())
    eligible_test_seasons = [season for season in available if season <= test_season]
    if not eligible_test_seasons:
        raise ValueError(f"No feature rows available for evaluation season {test_season}")
    test_season = int(eligible_test_seasons[-1])
    validation_season = test_season - 1
    if validation_season not in available:
        raise ValueError(
            f"Validation season {validation_season} is missing from the feature matrix; "
            f"available seasons: {available}"
        )

    build_train_test_split(windowed, race_results_df)

    train_df = windowed.loc[windowed["season"] < validation_season].copy()
    validation_df = windowed.loc[windowed["season"] == validation_season].copy()
    test_df = windowed.loc[windowed["season"] == test_season].copy()
    production_df = windowed.copy()

    LOGGER.info(
        "Dataset split counts | train=%s validation=%s test=%s | seasons=%s | eval_split=train<%s val=%s test=%s",
        len(train_df),
        len(validation_df),
        len(test_df),
        describe_season_split(windowed),
        validation_season,
        validation_season,
        test_season,
    )
    return {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
        "production": production_df,
        "split_seasons": {
            "train_max": validation_season - 1,
            "validation": validation_season,
            "test": test_season,
        },
    }


def _build_cv(groups: pd.Series) -> GroupKFold:
    """Create the cross-validation splitter."""

    unique_groups = int(groups.nunique())
    if unique_groups < CONFIG["cv_splits"]:
        raise ValueError("Not enough unique seasons for GroupKFold cross-validation")
    return GroupKFold(n_splits=CONFIG["cv_splits"])


def _sample_weights(
    frame: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    decay_rate: float = 0.97,
) -> np.ndarray:
    """Compute recency weights using exponential decay by race age."""

    race_dates = pd.to_datetime(frame["race_date"], errors="coerce")
    if race_dates.isna().any():
        raise ValueError("Cannot compute sample weights with null race_date values")
    reference = pd.Timestamp(reference_date) if reference_date is not None else pd.Timestamp(race_dates.max())
    days_since_race = (reference - race_dates).dt.days.clip(lower=0)
    weights = np.power(float(decay_rate), days_since_race / 30.0)
    return weights.to_numpy(dtype=float)


def _classifier_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Compute classifier evaluation metrics."""

    clipped_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    try:
        auc = roc_auc_score(y_true, clipped_prob)
    except ValueError:
        auc = float("nan")
    return {
        "log_loss": float(log_loss(y_true, clipped_prob, labels=[0, 1])),
        "auc": float(auc),
        "brier_score": float(brier_score_loss(y_true, clipped_prob)),
    }


def _probability_bucket_report(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Compute predicted-vs-actual probability buckets in 10% increments."""

    frame = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).astype(float),
            "y_prob": np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0),
        }
    )
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"{int(lower*100)}-{int(upper*100)}%" for lower, upper in zip(bins[:-1], bins[1:])]
    frame["bucket"] = pd.cut(
        frame["y_prob"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    bucket_report = (
        frame.groupby("bucket", observed=False)
        .agg(
            count=("y_true", "size"),
            avg_predicted=("y_prob", "mean"),
            actual_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    bucket_report["avg_predicted"] = bucket_report["avg_predicted"].fillna(0.0)
    bucket_report["actual_rate"] = bucket_report["actual_rate"].fillna(0.0)
    bucket_report["abs_gap"] = (bucket_report["avg_predicted"] - bucket_report["actual_rate"]).abs()
    return bucket_report


def _save_feature_importance(model: Any, label: str, feature_names: list[str]) -> Path:
    """Persist feature importance values and a Plotly figure using XGBoost gain."""

    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": [
                float(
                    gain_scores.get(feature_name, gain_scores.get(f"f{idx}", 0.0))
                )
                for idx, feature_name in enumerate(feature_names)
            ],
        }
    ).sort_values("gain", ascending=False)

    csv_path = CONFIG["processed_dir"] / f"feature_importance_model_{label}.csv"
    json_path = CONFIG["processed_dir"] / f"feature_importance_model_{label}.json"
    save_dataframe(importance, csv_path)

    figure = px.bar(
        importance.head(15).sort_values("gain"),
        x="gain",
        y="feature",
        orientation="h",
        template="plotly_dark",
        title=f"Model {label} Feature Importance",
        color="gain",
        color_continuous_scale=["#5f0f00", "#e10600"],
    )
    figure.update_layout(
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        font={"family": "Space Grotesk, sans-serif", "color": "#f5f5f5"},
    )
    json_path.write_text(figure.to_json(), encoding="utf-8")
    return csv_path


def _save_calibration_outputs(label: str, y_true: pd.Series, y_prob: np.ndarray) -> Path:
    """Persist calibration outputs for a classifier."""

    bucket_report = _probability_bucket_report(y_true, y_prob)
    bucket_path = CONFIG["processed_dir"] / f"calibration_buckets_model_{label}.csv"
    save_dataframe(bucket_report, bucket_path)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    curve_path = CONFIG["processed_dir"] / f"calibration_model_{label}.csv"
    save_dataframe(pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true}), curve_path)
    return bucket_path


def _calibrate_classifier(
    base_model: XGBClassifier,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    sample_weight: np.ndarray,
    model_name: str,
    cv: int | str = "prefit",
) -> CalibratedClassifierCV | XGBClassifier:
    """Calibrate a classifier with a method appropriate for the positive-label count."""

    n_positive = int(pd.to_numeric(y_calibration, errors="coerce").fillna(0).sum())
    LOGGER.info("%s: %s positive calibration samples", model_name, n_positive)
    method = "isotonic"
    if n_positive < 50:
        LOGGER.warning(
            "%s: only %s positive samples available. Using Platt scaling (sigmoid) instead of isotonic.",
            model_name,
            n_positive,
        )
        method = "sigmoid"

    calibration_weights = np.asarray(sample_weight, dtype=np.float64)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    try:
        calibrated_model.fit(X_calibration, y_calibration, sample_weight=calibration_weights)
    except ValueError as exc:
        if method == "sigmoid" and "Buffer dtype mismatch" in str(exc):
            LOGGER.warning(
                "%s: sigmoid calibration rejected weighted fitting (%s). Falling back to the uncalibrated model.",
                model_name,
                exc,
            )
            return base_model
        else:
            raise

    proba = calibrated_model.predict_proba(X_calibration)[:, 1]
    unique_vals = int(len(np.unique(np.round(proba, 2))))
    if unique_vals < 5:
        LOGGER.error(
            "%s: calibration is degenerate with only %s unique probability values. Falling back to the uncalibrated model.",
            model_name,
            unique_vals,
        )
        return base_model

    LOGGER.info(
        "%s: calibration OK with %s unique probability values in range [%.3f, %.3f]",
        model_name,
        unique_vals,
        float(np.min(proba)),
        float(np.max(proba)),
    )
    return calibrated_model


def _train_regression_model(split_data: dict[str, Any]) -> ModelResult:
    """Train the finish-position regression model."""

    train_df = split_data["train"]
    validation_df = split_data["validation"]
    test_df = split_data["test"]
    production_df = split_data["production"]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["finish_position"]
    groups = train_df["season"]
    train_weights = _sample_weights(train_df)

    estimator = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
    )
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=CONFIG["param_grid"],
        scoring="neg_mean_absolute_error",
        cv=_build_cv(groups),
        n_jobs=CONFIG["grid_search_jobs"],
        verbose=0,
    )
    grid.fit(X_train, y_train, groups=groups, sample_weight=train_weights)
    evaluation_model: XGBRegressor = grid.best_estimator_

    validation_predictions = evaluation_model.predict(validation_df[FEATURE_COLUMNS])
    test_predictions = evaluation_model.predict(test_df[FEATURE_COLUMNS])

    prediction_frame = test_df[
        ["season", "round", "driver_id", "driver_name", "race_name", "qualifying_position", "race_condition"]
    ].copy()
    prediction_frame["actual_finish_position"] = test_df["finish_position"]
    prediction_frame["predicted_finish_position"] = test_predictions
    save_dataframe(prediction_frame, CONFIG["processed_dir"] / "test_predictions_model_A.csv")

    final_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
        **grid.best_params_,
    )
    final_model.fit(
        production_df[FEATURE_COLUMNS],
        production_df["finish_position"],
        sample_weight=_sample_weights(production_df),
    )

    importance_path = _save_feature_importance(final_model, "A", FEATURE_COLUMNS)
    return ModelResult(
        label="A",
        model=final_model,
        base_model=final_model,
        best_params=grid.best_params_,
        cv_best_score=float(-grid.best_score_),
        validation_metrics={
            "mae": float(mean_absolute_error(validation_df["finish_position"], validation_predictions))
        },
        test_metrics={
            "mae": float(mean_absolute_error(test_df["finish_position"], test_predictions))
        },
        feature_importance_path=importance_path,
        feature_names=FEATURE_COLUMNS,
    )


def _train_calibrated_classifier(
    split_data: dict[str, Any],
    label: str,
    target: str,
    feature_names: list[str],
) -> ModelResult:
    """Train and calibrate a binary classifier for a given target."""

    train_df = split_data["train"]
    validation_df = split_data["validation"]
    test_df = split_data["test"]
    production_df = split_data["production"]
    is_dnf_model = bool(label == "E")

    X_train = train_df[feature_names]
    y_train = train_df[target]
    groups = train_df["season"]
    train_weights = _sample_weights(train_df, decay_rate=0.94 if is_dnf_model else 0.97)

    X_validation = validation_df[feature_names]
    y_validation = validation_df[target]
    validation_weights = _sample_weights(validation_df)

    if is_dnf_model:
        dnf_rate = float(y_train.mean())
        scale_pos_weight = ((1.0 - dnf_rate) / dnf_rate) if dnf_rate > 0 else 1.0
        LOGGER.info("DNF rate: %.3f, scale_pos_weight: %.1f", dnf_rate, scale_pos_weight)
    else:
        positive_count = max(int(y_train.sum()), 1)
        negative_count = max(int((1 - y_train).sum()), 1)
        scale_pos_weight = negative_count / positive_count

    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr" if is_dnf_model else "logloss",
        random_state=CONFIG["random_state"],
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=CONFIG["param_grid"],
        scoring="average_precision" if is_dnf_model else "neg_log_loss",
        cv=_build_cv(groups),
        n_jobs=CONFIG["grid_search_jobs"],
        verbose=0,
    )
    grid.fit(X_train, y_train, groups=groups, sample_weight=train_weights)
    evaluation_base_model: XGBClassifier = grid.best_estimator_

    if is_dnf_model:
        calibrated_model = _calibrate_classifier(
            evaluation_base_model,
            X_train,
            y_train,
            train_weights,
            model_name=label,
            cv=5,
        )
    else:
        calibrated_model = _calibrate_classifier(
            evaluation_base_model,
            X_validation,
            y_validation,
            validation_weights,
            model_name=label,
        )

    validation_prob = calibrated_model.predict_proba(X_validation)[:, 1]
    test_prob = calibrated_model.predict_proba(test_df[feature_names])[:, 1]
    bucket_path = _save_calibration_outputs(label, test_df[target], test_prob)

    prediction_frame = test_df[
        ["season", "round", "driver_id", "driver_name", "race_name", "qualifying_position", "race_condition"]
    ].copy()
    prediction_frame["target"] = test_df[target]
    prediction_frame["predicted_probability"] = test_prob
    save_dataframe(prediction_frame, CONFIG["processed_dir"] / f"test_predictions_model_{label}.csv")

    final_base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr" if is_dnf_model else "logloss",
        random_state=CONFIG["random_state"],
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        **grid.best_params_,
    )
    final_base_model.fit(
        production_df[feature_names],
        production_df[target],
        sample_weight=_sample_weights(production_df, decay_rate=0.94 if is_dnf_model else 0.97),
    )

    if is_dnf_model:
        final_calibrated_model = _calibrate_classifier(
            final_base_model,
            production_df[feature_names],
            production_df[target],
            _sample_weights(production_df, decay_rate=0.94),
            model_name=label,
            cv=5,
        )
    else:
        calibration_source = _latest_calibration_frame(production_df, target)
        final_calibrated_model = _calibrate_classifier(
            final_base_model,
            calibration_source[feature_names],
            calibration_source[target],
            _sample_weights(calibration_source),
            model_name=label,
        )

    importance_path = _save_feature_importance(final_base_model, label, feature_names)
    validation_metrics = _classifier_metrics(y_validation, validation_prob)
    test_metrics = _classifier_metrics(test_df[target], test_prob)

    if label == "E":
        validation_metrics = {
            "auc": validation_metrics["auc"],
            "brier_score": validation_metrics["brier_score"],
            "log_loss": validation_metrics["log_loss"],
        }
        test_metrics = {
            "auc": test_metrics["auc"],
            "brier_score": test_metrics["brier_score"],
            "log_loss": test_metrics["log_loss"],
        }

    return ModelResult(
        label=label,
        model=final_calibrated_model,
        base_model=final_base_model,
        best_params={**grid.best_params_, "scale_pos_weight": float(scale_pos_weight)},
        cv_best_score=float(-grid.best_score_),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        feature_importance_path=importance_path,
        feature_names=feature_names,
        calibration_bucket_path=bucket_path,
    )


def run_training_pipeline(
    features_df: pd.DataFrame | None = None,
    race_results_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train all requested models and persist artifacts."""

    ensure_directories()
    previous_metrics = load_json(CONFIG["metrics_path"], default={})
    if features_df is None:
        features_df = load_dataframe(CONFIG["features_path"])
    if race_results_df is None:
        race_results_df = load_dataframe(CONFIG["race_results_path"])

    features_df = _prepare_feature_frame(features_df)
    split_data = _split_datasets(features_df, race_results_df)
    LOGGER.info(
        "TEST SET: %s rows, season %s - metrics are reliable",
        len(split_data["test"]),
        split_data["split_seasons"]["test"],
    )

    regression_result = _train_regression_model(split_data)
    win_result = _train_calibrated_classifier(split_data, label="B", target="top1_finish", feature_names=FEATURE_COLUMNS)
    podium_result = _train_calibrated_classifier(split_data, label="C", target="top3_finish", feature_names=FEATURE_COLUMNS)
    points_result = _train_calibrated_classifier(split_data, label="D", target="top10_finish", feature_names=FEATURE_COLUMNS)
    dnf_result = _train_calibrated_classifier(split_data, label="E", target="dnf_flag", feature_names=DNF_FEATURE_COLUMNS)

    results = {
        "A": regression_result,
        "B": win_result,
        "C": podium_result,
        "D": points_result,
        "E": dnf_result,
    }

    manifest: dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "dnf_feature_columns": DNF_FEATURE_COLUMNS,
        "split_counts": {name: len(df) for name, df in split_data.items() if isinstance(df, pd.DataFrame)},
        "split_seasons": split_data["split_seasons"],
        "models": {},
    }

    metrics_summary: dict[str, Any] = {}
    for label, result in results.items():
        joblib.dump(result.model, MODEL_PATHS[label])
        if label in BASE_MODEL_PATHS and result.base_model is not None:
            joblib.dump(result.base_model, BASE_MODEL_PATHS[label])

        metrics_summary[label] = {
            "best_params": result.best_params,
            "cv_best_score": result.cv_best_score,
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
        }

        if result.calibration_bucket_path is not None:
            calibration_buckets = load_dataframe(result.calibration_bucket_path)
            calibration_key = f"calibration_{split_data['split_seasons']['test']}"
            metrics_summary[label][calibration_key] = calibration_buckets.to_dict(orient="records")

        manifest["models"][label] = {
            "path": str(MODEL_PATHS[label]),
            "feature_importance_path": str(result.feature_importance_path),
            "feature_names": result.feature_names,
        }
        if label in BASE_MODEL_PATHS:
            manifest["models"][label]["base_model_path"] = str(BASE_MODEL_PATHS[label])
        if result.calibration_bucket_path is not None:
            manifest["models"][label]["calibration_bucket_path"] = str(result.calibration_bucket_path)

    save_json(metrics_summary, CONFIG["metrics_path"])
    save_json(manifest, CONFIG["manifest_path"])
    save_json(
        {
            "current_season": get_current_season(),
            "rolling_window": _current_season_window(),
            "incremental_updates_since_full": 0,
            "last_full_retrain_season": int(split_data["production"]["season"].max()),
        },
        CONFIG["incremental_state_path"],
    )
    LOGGER.info(
        "model_update",
        extra={
            "event": "model_update",
            "before_metrics": previous_metrics,
            "after_metrics": metrics_summary,
            "split_seasons": split_data["split_seasons"],
        },
    )
    LOGGER.info("Model training complete. Saved models A-E to %s", CONFIG["processed_dir"])
    return {"metrics": metrics_summary, "manifest": manifest}


def load_models() -> dict[str, Any]:
    """Load all trained model artifacts from disk."""

    missing = [label for label, path in MODEL_PATHS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trained model artifacts for labels: {missing}")
    return {label: joblib.load(path) for label, path in MODEL_PATHS.items()}


def predict_from_frame(inference_frame: pd.DataFrame, models: dict[str, Any] | None = None) -> pd.DataFrame:
    """Predict race outcomes from a prepared inference frame."""

    active_models = models if models is not None else load_models()
    X_inference = inference_frame[FEATURE_COLUMNS]
    X_dnf = inference_frame[DNF_FEATURE_COLUMNS]
    expected_finish = active_models["A"].predict(X_inference)
    raw_win_score = active_models["B"].predict_proba(X_inference)[:, 1]
    podium_prob = active_models["C"].predict_proba(X_inference)[:, 1]
    points_prob = active_models["D"].predict_proba(X_inference)[:, 1]
    dnf_prob = active_models["E"].predict_proba(X_dnf)[:, 1]

    predictions = inference_frame[
        [
            "driver_id",
            "driver_name",
            "constructor_id",
            "constructor_name",
            "qualifying_position",
            "driver_wet_elo",
            "race_condition",
            "is_wet_race",
            "is_mixed_race",
        ]
    ].copy()
    predictions["raw_win_score"] = np.clip(raw_win_score, 0.0, 1.0)
    predictions["relative_win_probability"] = normalize_probabilities(predictions["raw_win_score"].to_numpy(dtype=float))
    predictions["podium_prob"] = np.clip(podium_prob, 0.0, 1.0)
    predictions["points_prob"] = np.clip(points_prob, 0.0, 1.0)
    predictions["dnf_prob"] = np.clip(dnf_prob, 0.0, 1.0)
    predictions["expected_finish"] = np.clip(expected_finish, 1.0, float(len(predictions)))
    predictions = predictions.sort_values(["expected_finish", "qualifying_position"]).reset_index(drop=True)
    LOGGER.info(
        "prediction_batch",
        extra={
            "prediction_event": True,
            "season": int(inference_frame["season"].iloc[0]) if "season" in inference_frame.columns else None,
            "round": int(inference_frame["round"].iloc[0]) if "round" in inference_frame.columns else None,
            "circuit_id": str(inference_frame["circuit_id"].iloc[0]) if "circuit_id" in inference_frame.columns else None,
            "feature_rows": inference_frame[["driver_id", *FEATURE_COLUMNS, *DNF_FEATURE_COLUMNS]].copy(),
            "predictions": predictions.copy(),
        },
    )
    return predictions


def predict_race(
    drivers_list: list[str],
    qualifying_positions: list[int] | dict[str, int],
    circuit_id: str,
    season: int,
    round_number: int,
    weather_mode: str | None = None,
) -> pd.DataFrame:
    """Predict a single race for a given driver list and grid order."""

    inference_frame = build_race_inference_frame(
        drivers_list=drivers_list,
        qualifying_positions=qualifying_positions,
        circuit_id=circuit_id,
        season=season,
        round_number=round_number,
        race_results=load_dataframe(CONFIG["race_results_path"]),
        pitstops_df=load_dataframe(CONFIG["pitstops_path"]),
        weather_mode=weather_mode,
    )
    predictions = predict_from_frame(inference_frame)
    LOGGER.info(
        "Predicted race %s round %s at %s for %s drivers",
        season,
        round_number,
        circuit_id,
        len(predictions),
    )
    return predictions


def _latest_calibration_frame(features_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a recent calibration slice with enough class diversity if possible."""

    latest_season = int(features_df["season"].max())
    calibration_frame = features_df.loc[features_df["season"] == latest_season].copy()
    if calibration_frame.empty or calibration_frame[target].nunique() < 2:
        calibration_frame = features_df.sort_values("race_date").tail(min(len(features_df), 500)).copy()
    return calibration_frame


def _fit_incremental_model(
    base_model: XGBRegressor | XGBClassifier,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    sample_weight: np.ndarray,
) -> XGBRegressor | XGBClassifier:
    """Continue training an existing XGBoost sklearn model on new rows."""

    updated_model = base_model.__class__(**base_model.get_params())
    updated_model.fit(
        X_new,
        y_new,
        sample_weight=sample_weight,
        xgb_model=base_model.get_booster(),
    )
    return updated_model


def retrain_incremental(
    new_races_df: pd.DataFrame,
    full_retrain: bool = False,
) -> dict[str, Any]:
    """Incrementally update trained models after a newly completed race."""

    ensure_directories()
    full_features = _windowed_feature_frame(_prepare_feature_frame(load_dataframe(CONFIG["features_path"])))
    new_features = _prepare_feature_frame(new_races_df)
    if new_features.empty:
        return {"updated": False, "reason": "No new rows provided"}

    state = load_json(
        CONFIG["incremental_state_path"],
        default={
            "incremental_updates_since_full": 0,
            "last_full_retrain_season": int(full_features["season"].max()),
        },
    )
    if (
        full_retrain
        or not all(path.exists() for path in MODEL_PATHS.values())
        or int(state.get("incremental_updates_since_full", 0)) >= 5
        or int(state.get("last_full_retrain_season", 0)) != int(full_features["season"].max())
    ):
        return run_training_pipeline(full_features)

    models = load_models()
    base_models: dict[str, Any] = {
        "A": models["A"],
        "B": joblib.load(BASE_MODEL_PATHS["B"]),
        "C": joblib.load(BASE_MODEL_PATHS["C"]),
        "D": joblib.load(BASE_MODEL_PATHS["D"]),
        "E": joblib.load(BASE_MODEL_PATHS["E"]),
    }

    updated_labels: list[str] = []
    manifest = load_json(CONFIG["manifest_path"], default={})
    metrics = load_json(CONFIG["metrics_path"], default={})

    for label, target in MODEL_TARGETS.items():
        feature_names = DNF_FEATURE_COLUMNS if label == "E" else FEATURE_COLUMNS
        X_new = new_features[feature_names]
        y_new = new_features[target]
        if label != "A" and y_new.nunique() < 2:
            LOGGER.info("Skipping incremental update for model %s because new batch has a single target class", label)
            continue

        updated_base = _fit_incremental_model(
            base_models[label],
            X_new,
            y_new,
            _sample_weights(new_features),
        )

        if label == "A":
            updated_model = updated_base
        else:
            calibration_frame = _latest_calibration_frame(full_features, target)
            if calibration_frame[target].nunique() < 2:
                updated_model = updated_base
            else:
                updated_model = _calibrate_classifier(
                    updated_base,
                    calibration_frame[feature_names],
                    calibration_frame[target],
                    _sample_weights(calibration_frame),
                )
            joblib.dump(updated_base, BASE_MODEL_PATHS[label])

        if label == "A":
            joblib.dump(updated_base, MODEL_PATHS[label])
        else:
            joblib.dump(updated_model, MODEL_PATHS[label])
        if label == "A":
            updated_base_model_for_importance = updated_base
        else:
            updated_base_model_for_importance = updated_base

        importance_path = _save_feature_importance(updated_base_model_for_importance, label, feature_names)
        if isinstance(manifest, dict):
            manifest.setdefault("models", {})
            manifest["models"].setdefault(label, {})
            manifest["models"][label]["path"] = str(MODEL_PATHS[label])
            manifest["models"][label]["feature_importance_path"] = str(importance_path)
            manifest["models"][label]["feature_names"] = feature_names
            if label in BASE_MODEL_PATHS:
                manifest["models"][label]["base_model_path"] = str(BASE_MODEL_PATHS[label])
        updated_labels.append(label)

    state["incremental_updates_since_full"] = int(state.get("incremental_updates_since_full", 0)) + 1
    state["last_incremental_update_season"] = int(new_features["season"].max())
    state["last_incremental_update_round"] = int(new_features["round"].max())

    if isinstance(metrics, dict):
        metrics["incremental_update"] = {
            "updated_labels": updated_labels,
            "last_update_season": int(new_features["season"].max()),
            "last_update_round": int(new_features["round"].max()),
            "updates_since_full": int(state["incremental_updates_since_full"]),
        }

    save_json(state, CONFIG["incremental_state_path"])
    save_json(manifest, CONFIG["manifest_path"])
    save_json(metrics, CONFIG["metrics_path"])
    LOGGER.info(
        "Incremental retraining complete for models %s using %s new rows",
        ",".join(updated_labels),
        len(new_features),
    )
    return {"updated": True, "updated_models": updated_labels, "rows": len(new_features)}


def retrain_models(refresh: bool = False) -> dict[str, Any]:
    """Public helper to retrain the full model stack from the saved feature matrix."""

    features_df = _prepare_feature_frame(load_dataframe(CONFIG["features_path"]))
    return run_training_pipeline(features_df if refresh else _windowed_feature_frame(features_df))


def load_metrics() -> dict[str, Any]:
    """Load saved model metrics if present."""

    metrics = load_json(CONFIG["metrics_path"])
    if metrics is None:
        raise FileNotFoundError(f"Missing model metrics at {CONFIG['metrics_path']}")
    return metrics
