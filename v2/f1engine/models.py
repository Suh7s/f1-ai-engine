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
from .contracts import FEATURE_VERSION, Unavailable, canonical, code_version, digest, now_utc, timestamp
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
    if len(lookup) != len(actual) or set(lookup) != {r["driver_id"] for r in frame["rows"]}:
        raise ValueError("Training targets and entries must match exactly")
    finish = np.array([lookup[r["driver_id"]]["position"] for r in frame["rows"]], float)
    if sorted(finish) != list(range(1, len(finish) + 1)):
        raise ValueError("Training targets must contain a complete, unique classification")
    retired = np.array([lookup[r["driver_id"]].get("retired", False) for r in frame["rows"]], int)
    return finish, retired


def finisher_targets(frame, actual):
    finish, retired = targets(frame, actual)
    lookup = {r["driver_id"]: r for r in actual}
    valid = (retired == 0) & ~np.array(
        [lookup[r["driver_id"]].get("disqualified", False) for r in frame["rows"]]
    )
    y = np.zeros(len(finish))
    indices = np.flatnonzero(valid)[np.argsort(finish[valid])]
    y[indices] = np.linspace(1, 0, len(indices))
    return y


def fit_candidate(
    examples: list[tuple[dict, list[dict]]], family="linear", parameters=None, columns=None
) -> tuple[dict, object | None]:
    if len(examples) < 8:
        raise Unavailable("At least eight completed training races are required")
    examples = sorted(examples, key=lambda ex: ex[0]["event"]["start_at"])
    if family == "ensemble":
        # Components and mixture weights use separate chronological calibration windows.
        from .evaluation import calibrate

        split = max(8, int(len(examples) * 0.75))
        if len(examples) - split < 4:
            raise Unavailable("Ensembles require at least 12 training races before weight calibration")
        components, rankers = [], []
        for fam, params in (
            ("calibrated_baseline", {"name": "baseline:rating"}),
            ("linear", {"alpha": 10.0}),
            ("xgboost", {"max_depth": 2, "learning_rate": 0.08}),
        ):
            meta, ranker = fit_candidate(examples[:split], fam, params, columns)
            components.append(calibrate(meta, ranker, examples[split:]))
            rankers.append(ranker)
        return {
            "family": family,
            "parameters": {},
            "components": components,
            "weights": [1 / 3] * 3,
            "train_events": [f["event"]["event_id"] for f, _ in examples],
            "train_end_at": max(r["available_at"] for _, a in examples for r in a),
            "calibration_events": [],
            "calibrated": False,
            "temperature": 1.0,
            "feature_version": FEATURE_VERSION,
            "code_version": code_version(),
            "created_at": now_utc(),
            "status": "candidate",
            "assumptions": [
                "Mixture weights fit chronological out-of-sample full-field distributions; component disagreement is not a complete estimate of model uncertainty."
            ],
        }, rankers
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
        finishes.extend(finisher_targets(frame, actual))
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
        group_weights = np.array([weights[qid == q][0] for q in np.unique(qid[survivor])])
        ranker.fit(x[survivor], y[survivor], qid=qid[survivor], sample_weight=group_weights)
        model = {}
    elif family == "calibrated_baseline":
        if parameters.get("name") not in BASELINES:
            raise ValueError("Unknown calibrated baseline")
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
        "train_end_at": max(r["available_at"] for _, actual in examples for r in actual),
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
    if metadata["family"] == "ensemble":
        parts = [predict_candidate(frame, m, r) for m, r in zip(metadata["components"], ranker)]
        weights = np.asarray(metadata["weights"])
        scores = sum(w * p["scores"] / p["temperature"] for w, p in zip(weights, parts))
        explanations = [
            {
                f"component_{j}_weighted_normalized_score": float(
                    weights[j] * p["scores"][i] / p["temperature"]
                )
                for j, p in enumerate(parts)
            }
            for i in range(len(frame["rows"]))
        ]
        return {
            "scores": scores,
            "retirement": sum(w * p["retirement"] for w, p in zip(weights, parts)),
            "disqualification": parts[0]["disqualification"],
            "temperature": 1.0,
            "retirement_beta": [2.0, 2.0],
            "components": parts,
            "weights": weights,
            "explanations": explanations,
            "model_version": model_version,
            "calibrated": metadata["calibrated"],
            "model_uncertainty_sigma": 0.15,
            "assumptions": metadata["assumptions"]
            + [
                "Each simulation draws one complete component distribution. Explanation values describe the weighted mean normalized strength, not causal effects or the mixture probability."
            ],
        }
    x, *_ = prepare_array(
        [frame], metadata["columns"], metadata["medians"], metadata["mean"], metadata["scale"]
    )
    if metadata["family"] == "calibrated_baseline":
        masked = {
            **frame,
            "rows": [
                {**row, **({"grid_fraction": None} if "grid_fraction" not in metadata["columns"] else {})}
                for row in frame["rows"]
            ],
        }
        result = baseline(masked, metadata["parameters"]["name"])
        scores, explanations = result["scores"], result["explanations"]
    elif metadata["family"] == "linear":
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


def validate_model(metadata: dict, ranker=None):
    if metadata["family"] not in {"linear", "xgboost", "calibrated_baseline", "ensemble"}:
        raise ValueError("Unknown model family")
    if set(metadata["train_events"]) & set(metadata["calibration_events"]):
        raise ValueError("Calibration and training events overlap")
    if not metadata.get("calibrated") or not metadata.get("calibration_events"):
        raise ValueError("Only independently calibrated artifacts can be published")
    if not np.isfinite(metadata["temperature"]) or metadata["temperature"] <= 0:
        raise ValueError("Invalid calibration temperature")
    if not metadata.get("train_end_at") or not metadata.get("calibration_start_at"):
        raise ValueError("Chronological training/calibration boundaries are required")
    if timestamp(metadata["train_end_at"]) >= timestamp(metadata["calibration_start_at"]):
        raise ValueError("Calibration must follow all training outcomes")
    if metadata["family"] == "ensemble":
        weights = np.asarray(metadata["weights"])
        components = metadata["components"]
        if (
            weights.shape != (len(components),)
            or len(components) < 2
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-10)
        ):
            raise ValueError("Invalid mixture weights")
        if ranker is None or len(ranker) != len(components):
            raise ValueError("Missing ensemble component artifacts")
        for m, r in zip(components, ranker):
            if m["family"] == "ensemble":
                raise ValueError("Nested ensembles are unsupported")
            validate_model(m, r)
            if set(m["train_events"] + m["calibration_events"]) & set(metadata["calibration_events"]):
                raise ValueError("Ensemble weight calibration overlaps component fitting")
        return
    width = len(metadata["columns"])
    if width < 1 or len(set(metadata["columns"])) != width:
        raise ValueError("Invalid model feature columns")
    for key, shape in (
        ("medians", (width,)),
        ("mean", (2 * width,)),
        ("scale", (2 * width,)),
        ("retirement_beta", (2,)),
    ):
        if np.shape(metadata[key]) != shape:
            raise ValueError(f"Invalid model {key} shape")
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
        part = metadata[section]
        if "coef" in part and np.shape(part["coef"]) != (2 * width,):
            raise ValueError("Invalid model coefficient shape")
        if "intercept" in part and np.shape(part["intercept"]) != ():
            raise ValueError("Invalid model intercept")
    if metadata["family"] == "linear" and not {"coef", "intercept"}.issubset(metadata["model"]):
        raise ValueError("Missing linear coefficients")
    reliability = metadata["reliability"]
    if "prior" in reliability and not 0 <= reliability["prior"] <= 1:
        raise ValueError("Invalid retirement prior")
    cal = metadata.get("hazard_calibration")
    if cal is not None and (
        set(cal) != {"slope", "intercept"}
        or not all(np.shape(v) == () and np.isfinite(v) for v in cal.values())
    ):
        raise ValueError("Invalid hazard calibration")
    if metadata["family"] == "xgboost" and ranker.n_features_in_ != 2 * width:
        raise ValueError("Ranking feature schema mismatch")


def _write_rankers(metadata, ranker, directory, prefix="ranker"):
    body = json.loads(canonical(metadata))
    if metadata["family"] == "ensemble":
        body["components"] = [
            _write_rankers(m, r, directory, f"component-{i}")
            for i, (m, r) in enumerate(zip(metadata["components"], ranker))
        ]
    elif ranker is not None:
        path = directory / (prefix + ".json")
        ranker.save_model(path)
        body["ranker_file"] = path.name
        body["ranker_hash"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    return body


def publish_model(store: Store, metadata: dict, ranker=None) -> str:
    if metadata["feature_version"] != FEATURE_VERSION or metadata["code_version"] != code_version():
        raise ValueError("Model was fitted under different code or feature definitions")
    validate_model(metadata, ranker)
    # Publication registers a candidate; activation additionally requires a promotion evidence record.
    body = {**metadata, "status": "candidate"}
    stage = store.root / "models" / f".staging-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        body = _write_rankers(body, ranker, stage)
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


def _read_rankers(metadata, directory):
    if metadata["family"] == "ensemble":
        return [_read_rankers(m, directory) for m in metadata["components"]]
    if metadata["family"] != "xgboost":
        return None
    filename = metadata.get("ranker_file", "ranker.json")
    if Path(filename).name != filename:
        raise Unavailable("Invalid ranking artifact path")
    path = directory / filename
    if (
        not path.exists()
        or __import__("hashlib").sha256(path.read_bytes()).hexdigest() != metadata["ranker_hash"]
    ):
        raise Unavailable("Ranking artifact integrity check failed")
    from xgboost import XGBRanker

    ranker = XGBRanker(n_jobs=4)
    ranker.load_model(path)
    return ranker


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
    ranker = _read_rankers(metadata, artifact.parent)
    try:
        validate_model(metadata, ranker)
    except (ValueError, KeyError, TypeError) as exc:
        raise Unavailable(f"Model validation failed: {exc}") from exc
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
