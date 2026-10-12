"""Inspectable baselines and safe JSON model artifacts; no legacy pickle loading."""

from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid
import numpy as np
from scipy.special import expit
from sklearn.linear_model import Ridge, LogisticRegression
from .contracts import FEATURE_VERSION, Unavailable, canonical, code_version, digest, now_utc
from .features import FEATURES, matrix
from .store import Store

BASELINES = ("baseline:grid", "baseline:form", "baseline:rating")


def baseline(frame: dict, name="baseline:rating") -> dict:
    if name not in BASELINES:
        raise ValueError("Unknown baseline")
    parts = []
    scores = []
    hazards = []
    for r in frame["rows"]:
        grid = (
            r["grid_fraction"]
            if r["grid_fraction"] is not None
            else (r["recent_quali"] if r["recent_quali"] is not None else 0.5)
        )
        form = r["recent_finish"] if r["recent_finish"] is not None else 0.5
        if name == "baseline:grid":
            values = {"grid_or_prior": -3 * grid}
        elif name == "baseline:form":
            values = {"recent_finish_or_prior": -3 * form}
        else:
            values = {
                "driver_strength": 1.4 * r["driver_strength"],
                "team_strength": 1.2 * r["team_strength"],
                "grid_or_prior": -1.5 * grid,
                "recent_finish_or_prior": -0.5 * form,
            }
        scores.append(sum(values.values()))
        parts.append(values)
        hazards.append(0.5 * r["retirement_rate"] + 0.5 * r["team_retirement_rate"])
    return {
        "scores": np.array(scores),
        "retirement": np.array(hazards),
        "disqualification": np.array([r["disqualification_rate"] for r in frame["rows"]]),
        "explanations": parts,
        "temperature": 1.0,
        "retirement_beta": [2.0, 2.0],
        "model_version": name,
        "calibrated": False,
        "model_uncertainty_sigma": 0.15,
        "assumptions": [
            "Baseline coefficients and Beta retirement timing are explicit unvalidated priors.",
            "Grid is estimated from prior qualifying form where observed grid is unavailable.",
            "Season team uncertainty uses a 0.15 log-strength prior until calibrated.",
            "Disqualification risk uses a population Beta(1,499) prior and 100-entry driver shrinkage; not independently calibrated. Retirement includes DNS, but excludes lapped finishers and disqualifications.",
        ],
    }


def prepare_array(frames: list[dict], columns: list[str], medians=None, mean=None, scale=None):
    x = np.vstack([matrix(f, columns) for f in frames])
    if medians is None:
        medians = np.array(
            [np.nanmedian(x[:, j]) if np.any(np.isfinite(x[:, j])) else 0.0 for j in range(x.shape[1])]
        )
    missing = np.isnan(x).astype(float)
    filled = np.where(np.isnan(x), medians, x)
    expanded = np.hstack([filled, missing])
    if mean is None:
        mean = expanded.mean(axis=0)
        scale = expanded.std(axis=0)
        scale[scale < 1e-8] = 1.0
    return (expanded - mean) / scale, np.asarray(medians), np.asarray(mean), np.asarray(scale)


def targets(frame: dict, actual: list[dict]):
    lookup = {r["driver_id"]: r for r in actual}
    if set(lookup) != {r["driver_id"] for r in frame["rows"]}:
        raise ValueError("Training targets and entries must match exactly")
    finish = np.array([lookup[r["driver_id"]]["position"] for r in frame["rows"]], float)
    retired = np.array([lookup[r["driver_id"]].get("retired", False) for r in frame["rows"]], int)
    return finish, retired


def fit_candidate(
    examples: list[tuple[dict, list[dict]]], family="linear", parameters=None, columns=None
) -> tuple[dict, object | None]:
    if len(examples) < 8:
        raise Unavailable("At least eight completed training races are required")
    columns = list(columns or FEATURES)
    if family == "linear":
        # Regularized driver and team effects; unseen entrants shrink to the population prediction.
        columns += ["driver:" + d for d in sorted({r["driver_id"] for f, _ in examples for r in f["rows"]})]
        columns += ["team:" + t for t in sorted({r["team_id"] for f, _ in examples for r in f["rows"]})]
    parameters = parameters or {"alpha": 10.0}
    frames = [f for f, _ in examples]
    x, medians, mean, scale = prepare_array(frames, columns)
    finishes = []
    retirements = []
    groups = []
    weights = []
    beta_samples = []
    latest = max(f["event"]["start_at"] for f in frames)
    from .contracts import timestamp

    for q, (frame, actual) in enumerate(examples):
        finish, retired = targets(frame, actual)
        finishes.extend((len(finish) - finish) / max(len(finish) - 1, 1))
        retirements.extend(retired)
        groups.extend([q] * len(finish))
        age = max((timestamp(latest) - timestamp(frame["event"]["start_at"])).days, 0)
        weights.extend([0.97 ** (age / 30)] * len(finish))
        laps = max([r.get("laps", 0) for r in actual], default=0)
        if laps:
            beta_samples.extend(
                [r.get("laps", 0) / laps for r in actual if r.get("retired") and 0 < r.get("laps", 0) < laps]
            )
    y = np.array(finishes)
    dnf = np.array(retirements)
    qid = np.array(groups)
    weights = np.array(weights)
    excluded = np.concatenate(
        [
            np.array(
                [
                    {r["driver_id"]: r for r in actual}[entry["driver_id"]].get("disqualified", False)
                    for entry in frame["rows"]
                ]
            )
            for frame, actual in examples
        ]
    )
    survivor = (dnf == 0) & ~excluded
    ranker = None
    if family == "linear":
        fitted = Ridge(alpha=parameters.get("alpha", 10.0)).fit(
            x[survivor], y[survivor], sample_weight=weights[survivor]
        )
        model = {"coef": fitted.coef_.tolist(), "intercept": float(fitted.intercept_)}
    elif family == "xgboost":
        from xgboost import XGBRanker

        ranker = XGBRanker(
            objective="rank:pairwise",
            n_jobs=4,
            random_state=42,
            tree_method="hist",
            n_estimators=200,
            max_depth=parameters.get("max_depth", 3),
            learning_rate=parameters.get("learning_rate", 0.08),
            subsample=0.8,
            colsample_bytree=0.8,
        )
        ranker.fit(x[survivor], y[survivor], qid=qid[survivor])
        model = {}
    else:
        raise ValueError("Unknown model family")
    if len(np.unique(dnf)) == 2:
        hazard = LogisticRegression(C=0.5, max_iter=500, random_state=42).fit(x, dnf, sample_weight=weights)
        reliability = {"coef": hazard.coef_[0].tolist(), "intercept": float(hazard.intercept_[0])}
    else:
        reliability = {"prior": float((dnf.sum() + 1) / (len(dnf) + 10))}
    beta = [2.0, 2.0]
    if len(beta_samples) >= 10:
        mu = float(np.mean(beta_samples))
        var = float(np.var(beta_samples))
        concentration = max(mu * (1 - mu) / max(var, 1e-4) - 1, 0.5)
        beta = [max(mu * concentration, 0.2), max((1 - mu) * concentration, 0.2)]
    metadata = {
        "family": family,
        "parameters": parameters,
        "columns": columns,
        "medians": medians.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "model": model,
        "reliability": reliability,
        "retirement_beta": beta,
        "temperature": 1.0,
        "hazard_calibration": None,
        "calibrated": False,
        "train_events": [f["event"]["event_id"] for f in frames],
        "calibration_events": [],
        "feature_version": FEATURE_VERSION,
        "code_version": code_version(),
        "created_at": now_utc(),
        "model_uncertainty_sigma": 0.15,
        "status": "candidate",
        "assumptions": [
            "Fuel loads and private team information are unobserved.",
            "Residual season-level uncertainty is provisional until validated.",
            "Disqualification risk is an empirical shrunken classification prior, not independently calibrated; retirement includes DNS and excludes lapped finishers.",
        ],
    }
    return metadata, ranker


def predict_candidate(frame: dict, metadata: dict, ranker=None, model_version="candidate") -> dict:
    x, *_ = prepare_array(
        [frame], metadata["columns"], metadata["medians"], metadata["mean"], metadata["scale"]
    )
    if metadata["family"] == "linear":
        coef = np.array(metadata["model"]["coef"])
        scores = x @ coef + metadata["model"]["intercept"]
        names = metadata["columns"] + [f"{c}_missing" for c in metadata["columns"]]
        explanations = [
            {**dict(zip(names, (row * coef).tolist())), "intercept": metadata["model"]["intercept"]}
            for row in x
        ]
    else:
        if ranker is None:
            raise Unavailable("Ranking model is missing")
        scores = ranker.predict(x)
        # Tree SHAP contributions are computed by XGBoost itself in raw-score space.
        import xgboost as xgb

        contributions = ranker.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)
        names = metadata["columns"] + [f"{c}_missing" for c in metadata["columns"]] + ["intercept"]
        explanations = [dict(zip(names, row.tolist())) for row in contributions]
    reliability = metadata["reliability"]
    raw = (
        expit(x @ np.array(reliability["coef"]) + reliability["intercept"])
        if "coef" in reliability
        else np.full(len(x), reliability["prior"])
    )
    cal = metadata.get("hazard_calibration")
    if cal:
        logits = np.log(np.clip(raw, 1e-8, 1 - 1e-8) / (1 - np.clip(raw, 1e-8, 1 - 1e-8)))
        raw = expit(cal["slope"] * logits + cal["intercept"])
    return {
        "scores": scores,
        "retirement": raw,
        "disqualification": np.array([r["disqualification_rate"] for r in frame["rows"]]),
        "explanations": explanations,
        "temperature": metadata["temperature"],
        "retirement_beta": metadata["retirement_beta"],
        "model_version": model_version,
        "calibrated": metadata["calibrated"],
        "model_uncertainty_sigma": metadata.get("model_uncertainty_sigma", 0.15),
        "assumptions": metadata["assumptions"],
    }


def publish_model(store: Store, metadata: dict, ranker=None) -> str:
    if set(metadata["train_events"]) & set(metadata["calibration_events"]):
        raise ValueError("Calibration and training events overlap")
    if not metadata.get("calibrated") or not metadata.get("calibration_events"):
        raise ValueError("Only independently calibrated artifacts can be published")
    if not np.isfinite(metadata["temperature"]) or metadata["temperature"] <= 0:
        raise ValueError("Invalid calibration temperature")
    arrays = ["medians", "mean", "scale", "retirement_beta"]
    for key in arrays:
        if not np.isfinite(metadata[key]).all():
            raise ValueError(f"Nonfinite model {key}")
    if np.any(np.array(metadata["scale"]) <= 0) or np.any(np.array(metadata["retirement_beta"]) <= 0):
        raise ValueError("Invalid scale or retirement parameters")
    if metadata["family"] == "xgboost" and ranker is None:
        raise ValueError("Missing ranking artifact")
    for section in ("model", "reliability"):
        for value in metadata[section].values():
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite model coefficient")
    # Publication registers a candidate; activation additionally requires a promotion evidence record.
    body = {**metadata, "status": "candidate"}
    stage = store.root / "models" / f".staging-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        if ranker is not None:
            ranker.save_model(stage / "ranker.json")
            body["ranker_hash"] = (
                __import__("hashlib").sha256((stage / "ranker.json").read_bytes()).hexdigest()
            )
        identifier = digest(body)
        (stage / "model.json").write_text(canonical(body))
        target = store.root / "models" / identifier
        if not target.exists():
            os.replace(stage, target)
        with store.db() as db:
            db.execute("INSERT OR IGNORE INTO models VALUES (?,?)", (identifier, canonical(body)))
        return identifier
    finally:
        if stage.exists():
            __import__("shutil").rmtree(stage)


def load_model(store: Store, identifier: str):
    with store.db() as db:
        if identifier == "active" or identifier.startswith("active:"):
            name = "model:" + identifier.split(":", 1)[1] if ":" in identifier else "model:pre_weekend"
            row = db.execute("SELECT value FROM pointers WHERE name=?", (name,)).fetchone()
            if not row:
                return None, None, "baseline:rating"
            identifier = row["value"]
        if identifier in BASELINES:
            return None, None, identifier
        row = db.execute("SELECT body FROM models WHERE id=?", (identifier,)).fetchone()
    if not row:
        raise Unavailable("Unknown model artifact")
    metadata = json.loads(row["body"])
    if digest(metadata) != identifier or metadata["feature_version"] != FEATURE_VERSION:
        raise Unavailable("Model integrity/schema check failed")
    artifact = store.root / "models" / identifier / "model.json"
    if not artifact.exists() or json.loads(artifact.read_text()) != metadata:
        raise Unavailable("Model metadata was changed or is missing")
    ranker = None
    if metadata["family"] == "xgboost":
        path = store.root / "models" / identifier / "ranker.json"
        if (
            not path.exists()
            or __import__("hashlib").sha256(path.read_bytes()).hexdigest() != metadata["ranker_hash"]
        ):
            raise Unavailable("Ranking artifact integrity check failed")
        from xgboost import XGBRanker

        ranker = XGBRanker(n_jobs=4)
        ranker.load_model(path)
    return metadata, ranker, identifier


def predict(frame: dict, store: Store, identifier="baseline:rating") -> dict:
    if identifier.startswith("baseline:"):
        return baseline(frame, identifier)
    requested_identifier = identifier
    metadata, ranker, identifier = load_model(
        store, "active:" + frame.get("stage", "pre_weekend") if identifier == "active" else identifier
    )
    if metadata is None:
        result = baseline(frame, identifier)
        result["assumptions"].append(f"No promoted challenger at this stage: explicitly using {identifier}.")
        return result
    if metadata.get("prediction_stage") and metadata["prediction_stage"] != frame.get("stage"):
        if requested_identifier == "active":
            result = baseline(frame)
            result["assumptions"].append(
                "Promoted model is validated for another stage; using the rating baseline."
            )
            return result
        raise Unavailable("This model was validated for a different prediction stage")
    return predict_candidate(frame, metadata, ranker, identifier)
