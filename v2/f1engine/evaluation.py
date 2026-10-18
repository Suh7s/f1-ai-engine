"""Chronological evaluation, independent calibration, and evidence-gated promotion."""

from __future__ import annotations
from datetime import timedelta
import json
import math
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from .contracts import ForecastRequest, Stage, Unavailable, code_version, digest, now_utc, timestamp
from .features import FEATURES, FEATURE_GROUPS, build_features
from .models import BASELINES, baseline, fit_candidate, predict_candidate, publish_model, load_model, targets
from .simulation import sample_race, sample_prediction
from .store import Store

# The published V2 checkpoint already inspected 2025. This project-level fact must
# survive a fresh local database; deleting local evidence cannot reseal a holdout.
HOLDOUT_2025_RETIRED = True


def examples(
    store: Store, version: str, stage: Stage = Stage.POST_QUALIFYING, max_season=2024, check=lambda: None
):
    results = store.table("results", version)
    sessions = store.table("sessions", version)
    out = []
    for event in sorted(store.table("events", version), key=lambda r: r["start_at"]):
        check()
        if event["season"] > max_season:
            continue
        actual = [r for r in results if r["event_id"] == event["event_id"] and r["session"] == "race"]
        if len(actual) < 2:
            continue
        if stage == Stage.PRE_WEEKEND:
            cutoff = timestamp(event["start_at"]) - timedelta(days=3)
        else:
            qualifying = [
                r for r in results if r["event_id"] == event["event_id"] and r["session"] == "qualifying"
            ]
            if not qualifying:
                continue
            cutoff = max(timestamp(r["available_at"]) for r in qualifying)
            if cutoff >= timestamp(event["start_at"]):
                continue
        try:
            frame = build_features(
                store,
                ForecastRequest(
                    event_id=event["event_id"], stage=stage, as_of_utc=cutoff, data_version=version
                ),
            )
            targets(frame, actual)
            out.append((frame, actual))
        except (Unavailable, ValueError):
            continue
    return out


def calibrate(metadata, ranker, calibration_examples):
    if len(calibration_examples) < 4:
        raise Unavailable("At least four independent calibration races are required")
    event_ids = [f["event"]["event_id"] for f, _ in calibration_examples]
    if set(event_ids) & set(metadata["train_events"]):
        raise ValueError("Calibration overlaps training")
    start = min(f["as_of_utc"] for f, _ in calibration_examples)
    if timestamp(metadata["train_end_at"]) >= timestamp(start):
        raise ValueError("Calibration must follow all training outcomes")
    boundaries = {
        "calibration_start_at": start,
        "calibration_end_at": max(r["available_at"] for _, a in calibration_examples for r in a),
        "calibration_events": event_ids,
    }
    if metadata["family"] == "ensemble":
        count = len(metadata["components"])
        h, b = np.zeros((count, count)), np.zeros(count)
        for f, actual in calibration_examples:
            seed = int(digest(f["event"]["event_id"])[:8], 16)
            n = len(f["rows"])
            cdfs = []
            for m, r in zip(metadata["components"], ranker):
                p = predict_candidate(f, m, r)
                samples = sample_prediction(
                    p,
                    f["event"]["season"],
                    4000,
                    np.random.default_rng(seed),
                    laps=f["event"].get("scheduled_laps") or 60,
                    final_round=f["event"].get("double_points", False),
                )
                cdfs.append(
                    (samples["position"][:, :, None] <= np.arange(1, n)[None, None, :]).mean(axis=0).ravel()
                )
            c = np.asarray(cdfs)
            finish, _ = targets(f, actual)
            observed = (finish[:, None] <= np.arange(1, n)).ravel()
            h += c @ c.T / c.shape[1]
            b += c @ observed / c.shape[1]
        optimum = minimize(
            lambda w: float(w @ h @ w - 2 * w @ b),
            np.full(count, 1 / count),
            jac=lambda w: 2 * h @ w - 2 * b,
            method="SLSQP",
            bounds=[(0, 1)] * count,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(count)},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not optimum.success:
            raise Unavailable("Mixture calibration did not converge")
        weights = np.maximum(optimum.x, 0)
        weights /= weights.sum()
        return {
            **metadata,
            **boundaries,
            "weights": weights.tolist(),
            "calibrated": True,
            "weight_objective": "Race-averaged normalized RPS on independent chronological predictions",
            "weight_simulations": 4000,
        }
    predictions = [predict_candidate(f, metadata, ranker) for f, _ in calibration_examples]

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        loss = 0.0
        count = 0
        for (f, a), p in zip(calibration_examples, predictions):
            finish, dnf = targets(f, a)
            lookup = {r["driver_id"]: r for r in a}
            valid = (dnf == 0) & ~np.array(
                [lookup[r["driver_id"]].get("disqualified", False) for r in f["rows"]]
            )
            order = np.argsort(finish[valid])
            logits = p["scores"][valid][order] / temperature
            for i in range(max(len(logits) - 1, 0)):
                loss += logsumexp(logits[i:]) - logits[i]
                count += 1
        return loss / max(count, 1)

    optimum = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not optimum.success or not np.isfinite(optimum.fun):
        raise Unavailable("Ranking calibration did not converge")
    metadata = {
        **metadata,
        **boundaries,
        "temperature": float(np.exp(optimum.x)),
        "calibration_events": event_ids,
        "calibrated": True,
    }
    raw = np.concatenate([p["retirement"] for p in predictions])
    labels = np.concatenate([targets(f, a)[1] for f, a in calibration_examples])
    if len(np.unique(labels)) == 2:
        z = np.log(np.clip(raw, 1e-8, 1 - 1e-8) / (1 - np.clip(raw, 1e-8, 1 - 1e-8))).reshape(-1, 1)
        clf = LogisticRegression(C=1, max_iter=500).fit(z, labels)
        metadata["hazard_calibration"] = {
            "slope": float(clf.coef_[0, 0]),
            "intercept": float(clf.intercept_[0]),
        }
    else:
        metadata["assumptions"] = metadata["assumptions"] + [
            "Calibration cohort has one retirement class; retirement calibration was unavailable."
        ]
    metadata["calibration_loss"] = float(optimum.fun)
    return metadata


def score_samples(frame, actual, samples):
    from .scoring import distribution_metrics

    positions = samples["position"]
    n = len(frame["rows"])
    drivers = [
        {
            "driver_id": row["driver_id"],
            "position_probabilities": (
                np.bincount(positions[:, j], minlength=n + 1)[1:] / len(positions)
            ).tolist(),
            "points_probability": float((samples["points"][:, j] > 0).mean()),
            "retirement_probability": float(samples["retired"][:, j].mean()),
        }
        for j, row in enumerate(frame["rows"])
    ]
    rain = frame["rows"][0].get("rain_probability")
    return {
        "event_id": frame["event"]["event_id"],
        "season": frame["event"]["season"],
        "circuit": frame["event"]["circuit_id"],
        "stage": frame.get("stage", "unknown"),
        "weather": "unknown" if rain is None else "wet_forecast" if rain >= 0.5 else "dry_forecast",
        **distribution_metrics(drivers, actual, len(positions)),
    }


def evaluate_model(examples_, model_fn, simulations=2000, check=lambda: None):
    rows = []
    for frame, actual in examples_:
        check()
        model = model_fn(frame)
        seed = int(digest(frame["event"]["event_id"])[:8], 16)
        samples = sample_prediction(
            model,
            frame["event"]["season"],
            simulations,
            np.random.default_rng(seed),
            laps=frame["event"].get("scheduled_laps") or 60,
            final_round=frame["event"].get("double_points", False),
        )
        rows.append(score_samples(frame, actual, samples))
    return rows


def report(rows: list[dict]) -> dict:
    metrics = [
        "rps",
        "winner_log_loss",
        "podium_brier",
        "points_brier",
        "retirement_brier",
        "position_mae",
        "interval_coverage",
    ]
    if not rows:
        return {"races": 0, "metrics": {}, "breakdowns": {}, "calibration": {}}
    breakdowns = {}
    for key in ["season", "circuit", "weather", "stage"]:
        breakdowns[key] = {
            str(v): {
                "races": sum(r.get(key, "unknown") == v for r in rows),
                **{m: float(np.mean([r[m] for r in rows if r.get(key, "unknown") == v])) for m in metrics},
            }
            for v in sorted({r.get(key, "unknown") for r in rows})
        }
    curves = {}
    for target in ("win", "podium", "points", "retirement"):
        pairs = [p for row in rows for p in row["calibration_pairs"][target]]
        bins = []
        for b in range(10):
            selected = [(p, y) for p, y in pairs if min(int(p * 10), 9) == b]
            if selected:
                bins.append(
                    {
                        "predicted": float(np.mean([p for p, _ in selected])),
                        "observed": float(np.mean([y for _, y in selected])),
                        "count": len(selected),
                    }
                )
        curves[target] = bins
    return {
        "races": len(rows),
        "metrics": {m: float(np.mean([r[m] for r in rows])) for m in metrics},
        "breakdowns": breakdowns,
        "calibration": curves,
    }


def promotion_gate(candidate: list[dict], reference: list[dict], seed=42):
    c = {r["event_id"]: r for r in candidate}
    b = {r["event_id"]: r for r in reference}
    if set(c) != set(b) or len(c) < 15:
        return {
            "promote": False,
            "reason": "At least 15 identical paired race events are required",
            "paired_races": len(set(c) & set(b)),
        }
    ids = sorted(c)
    diff = np.array([c[k]["rps"] - b[k]["rps"] for k in ids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(5000, len(diff)))].mean(axis=1)
    interval = np.quantile(means, [0.025, 0.975]).tolist()
    cl = float(np.mean([c[k]["winner_log_loss"] for k in ids]))
    bl = float(np.mean([b[k]["winner_log_loss"] for k in ids]))
    passes = interval[1] < 0 and cl <= bl * 1.02
    return {
        "promote": bool(passes),
        "paired_races": len(ids),
        "rps_difference": float(diff.mean()),
        "rps_difference_95ci": interval,
        "winner_log_loss_candidate": cl,
        "winner_log_loss_baseline": bl,
        "reason": "Supported RPS improvement without >2% winner log-loss degradation"
        if passes
        else "Evidence is insufficient; retain the baseline",
        "bootstrap_unit": "race",
        "bootstrap_replicates": 5000,
    }


def experiment(
    store: Store, parameters: dict, check=lambda: None, progress=lambda x, m: None, checkpoint=None
):
    checkpoint = {} if checkpoint is None else checkpoint
    version = parameters.get("data_version") or checkpoint.get("data_version") or store.current_version()
    stage = Stage(parameters.get("stage", "post_qualifying"))
    if stage == Stage.LIVE:
        raise ValueError("Live models require replay evaluation, not pre-race experiment mode")
    all_examples = examples(store, version, stage, max_season=2024, check=check)
    years = {f["event"]["season"] for f, _ in all_examples}
    if not set([2021, 2022, 2023, 2024]).issubset(years):
        raise Unavailable(
            "Development requires data through 2024 with training and independent 2021–2024 calibration/evaluation coverage"
        )
    candidates = (
        [("calibrated_baseline", {"name": name}) for name in BASELINES]
        + [("linear", {"alpha": a}) for a in (1.0, 10.0)]
        + [("xgboost", {"max_depth": d, "learning_rate": lr}) for d in (2, 3, 4) for lr in (0.03, 0.08)]
        + [("ensemble", {})]
    )
    limit = min(int(parameters.get("candidates", 12)), 12)
    if limit < 1:
        raise ValueError("Candidate count must be positive")
    results = checkpoint.get("candidates", [])
    if (
        checkpoint.get("data_version", version) != version
        or checkpoint.get("code_version", code_version()) != code_version()
    ):
        raise ValueError("Cannot resume an experiment against different code or data")
    for idx, (family, params) in enumerate(candidates[:limit]):
        if idx < len(results):
            continue
        check()
        candidate_rows = []
        baseline_rows = {name: [] for name in BASELINES}
        for test_year in (2022, 2023, 2024):
            training = [ex for ex in all_examples if ex[0]["event"]["season"] < test_year - 1]
            calibration = [ex for ex in all_examples if ex[0]["event"]["season"] == test_year - 1]
            testing = [ex for ex in all_examples if ex[0]["event"]["season"] == test_year]
            meta, model = fit_candidate(training, family, params)
            meta = calibrate(meta, model, calibration)
            candidate_rows += evaluate_model(
                testing, lambda f: predict_candidate(f, meta, model), check=check
            )
            for name in BASELINES:
                baseline_rows[name] += evaluate_model(testing, lambda f, n=name: baseline(f, n), check=check)
        strongest = min(BASELINES, key=lambda name: report(baseline_rows[name])["metrics"]["rps"])
        result = {
            "index": idx,
            "family": family,
            "parameters": params,
            "rows": candidate_rows,
            "report": report(candidate_rows),
            "strongest_baseline": strongest,
            "baseline_report": report(baseline_rows[strongest]),
            "baseline_rows": baseline_rows[strongest],
            "gate": promotion_gate(candidate_rows, baseline_rows[strongest]),
        }
        results.append(result)
        checkpoint.update(candidates=results, data_version=version, code_version=code_version())
        progress((idx + 1) / (limit + 1), f"Completed candidate {idx + 1}/{limit}")
    best = min(results, key=lambda r: r["report"]["metrics"]["rps"])
    training = [ex for ex in all_examples if ex[0]["event"]["season"] <= 2023]
    calibration = [ex for ex in all_examples if ex[0]["event"]["season"] == 2024]
    meta, model = fit_candidate(training, best["family"], best["parameters"])
    meta = calibrate(meta, model, calibration)
    meta.update(data_version=version, prediction_stage=stage.value, development_report=best["report"])
    identifier = publish_model(store, meta, model)
    # Ablations are performed on the development window, never on the sealed holdout.
    ablations = {}
    train_ab = [ex for ex in all_examples if ex[0]["event"]["season"] <= 2022]
    cal_ab = [ex for ex in all_examples if ex[0]["event"]["season"] == 2023]
    test_ab = [ex for ex in all_examples if ex[0]["event"]["season"] == 2024]
    for name, removed in FEATURE_GROUPS.items():
        check()
        coverage = {
            window: {c: sum(f["coverage"].get(c, 0) > 0 for f, _ in cohort) for c in removed}
            for window, cohort in (("training", train_ab), ("calibration", cal_ab), ("evaluation", test_ab))
        }
        supported = [c for c in removed if all(coverage[w][c] >= 4 for w in coverage)]
        if not supported:
            ablations[name] = {
                "status": "unverified_missing_inputs",
                "coverage_races": coverage,
                "reason": "No feature in this group has at least four races of data in every training/calibration/evaluation window; value is unverified",
            }
            continue
        reduced = [c for c in FEATURES if c not in removed]
        am, ar = fit_candidate(train_ab, best["family"], best["parameters"], reduced)
        am = calibrate(am, ar, cal_ab)
        ablations[name] = {
            "coverage_races": coverage,
            "supported_features": supported,
            "unsupported_features": [c for c in removed if c not in supported],
            "report": report(evaluate_model(test_ab, lambda f: predict_candidate(f, am, ar), check=check)),
        }
    opened = HOLDOUT_2025_RETIRED or any(
        e["kind"] == "holdout_opened" and e.get("season") == 2025 for e in store.evidence()
    )
    body = {
        "kind": "development_experiment",
        "evidence_kind": "illustrative_scenario"
        if store.manifest(version)["illustrative"]
        else "historical_reconstruction",
        "data_version": version,
        "code_version": code_version(),
        "stage": stage.value,
        "candidates": results,
        "selected_candidate": identifier,
        "selected_index": best["index"],
        "ablations": ablations,
        "holdout_policy": (
            "2025 was previously opened and is development evidence for any revised candidate. Promotion requires a newly registered prospective cohort."
            if opened
            else "2025 remains unopened. Model is a candidate, not promoted."
        ),
        "benchmark_limitations": [
            "Historical input publication times are reconstructed.",
            "Calibration bins have finite sample sizes.",
            "Comparisons are limited to the named models and available races.",
        ],
    }
    evidence_id = store.add_evidence(body)
    if not store.manifest(version)["illustrative"]:
        with store.db() as db:
            name = "model:" + stage.value
            row = db.execute("SELECT value FROM pointers WHERE name=?", (name,)).fetchone()
            if not row or row["value"].startswith("baseline:"):
                db.execute("INSERT OR REPLACE INTO pointers VALUES (?,?)", (name, best["strongest_baseline"]))
    freeze_id = store.add_evidence(
        {
            "kind": "candidate_freeze",
            "model_version": identifier,
            "data_version": version,
            "code_version": code_version(),
            "stage": stage.value,
            "development_evidence": evidence_id,
            "baseline": best["strongest_baseline"],
            "holdout_season": None if opened else 2025,
            "evaluation_policy": "prospective_only" if opened else "sealed_2025",
            "prospective_policy": "First pair per event; first 15 registered events by race start; one evaluation after all 15 outcomes; same cutoff/data/seed and 10000 simulations for both models",
            "promotion_policy": {
                "min_paired_races": 15,
                "rps_upper95_less_than": 0,
                "winner_log_loss_max_ratio": 1.02,
            },
            "illustrative": store.manifest(version)["illustrative"],
        }
    )
    return {
        "model_version": identifier,
        "freeze_id": freeze_id,
        "evidence_id": evidence_id,
        "development_gate": best["gate"],
        "status": "candidate_frozen",
    }


def holdout(store: Store, parameters: dict, check=lambda: None, progress=lambda x, m: None):
    if HOLDOUT_2025_RETIRED:
        raise ValueError(
            "The published project already inspected 2025. Restore the frozen original checkpoint to reproduce that report; revised candidates require prospective evidence."
        )
    freeze_id = parameters.get("freeze_id")
    evidence = store.evidence()
    freeze = next((e for e in evidence if e["id"] == freeze_id and e["kind"] == "candidate_freeze"), None)
    if not freeze:
        raise ValueError("Freeze a candidate before opening the holdout")
    previous = next((e for e in evidence if e["kind"] == "holdout_opened" and e["season"] == 2025), None)
    completed = next((e for e in evidence if e["kind"] == "holdout_report" and e.get("season") == 2025), None)
    if completed or (previous and previous["freeze_id"] != freeze_id):
        raise ValueError(
            "2025 has already been opened; it cannot be repeatedly claimed as an untouched holdout"
        )
    if freeze["code_version"] != code_version():
        raise ValueError("Code changed since candidate freeze")
    version = parameters.get("data_version") or store.current_version()
    if previous and previous["data_version"] != version:
        raise ValueError("Holdout resume must use the same dataset")
    meta, model, identifier = load_model(store, freeze["model_version"])
    selected = examples(store, version, Stage(freeze["stage"]), max_season=2025, check=check)
    testing = [ex for ex in selected if ex[0]["event"]["season"] == 2025]
    if len(testing) < 15:
        raise Unavailable("At least 15 complete holdout races required before opening")
    if not previous:
        store.add_evidence(
            {
                "kind": "holdout_opened",
                "season": 2025,
                "freeze_id": freeze_id,
                "model_version": identifier,
                "data_version": version,
                "policy": "Any future revisions informed by these results must treat 2025 as development data.",
            }
        )
    candidate = evaluate_model(testing, lambda f: predict_candidate(f, meta, model), check=check)
    reference = evaluate_model(testing, lambda f: baseline(f, freeze["baseline"]), check=check)
    gate = promotion_gate(candidate, reference)
    report_id = store.add_evidence(
        {
            "kind": "holdout_report",
            "evidence_kind": "held_out_evaluation",
            "season": 2025,
            "freeze_id": freeze_id,
            "model_version": identifier,
            "data_version": version,
            "candidate": report(candidate),
            "baseline": report(reference),
            "candidate_rows": candidate,
            "baseline_rows": reference,
            "gate": gate,
            "illustrative": store.manifest(version)["illustrative"],
        }
    )
    if gate["promote"] and not store.manifest(version)["illustrative"]:
        with store.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO pointers VALUES (?,?)", ("model:" + freeze["stage"], identifier)
            )
        store.add_evidence(
            {"kind": "model_promotion", "model_version": identifier, "evidence_id": report_id, "gate": gate}
        )
    transfer = [
        ex
        for ex in examples(store, version, Stage(freeze["stage"]), max_season=2026, check=check)
        if ex[0]["event"]["season"] == 2026
    ]
    transfer_id = None
    if transfer:
        transfer_id = store.add_evidence(
            {
                "kind": "regulation_transfer",
                "evidence_kind": "historical_reconstruction",
                "season": 2026,
                "model_version": identifier,
                "report": report(
                    evaluate_model(transfer, lambda f: predict_candidate(f, meta, model), check=check)
                ),
            }
        )
    return {
        "report_id": report_id,
        "gate": gate,
        "promoted": bool(gate["promote"] and not store.manifest(version)["illustrative"]),
        "transfer_report": transfer_id,
    }


def score_ledger(store: Store):
    from .scoring import distribution_metrics, SCORING_VERSION

    version = store.current_version()
    results = store.table("results", version, as_of=now_utc())
    # Evidence is newest first. Keep its identity so corrected outcomes append linked scores.
    existing = {}
    for e in store.evidence():
        if e["kind"] == "forecast_score":
            existing.setdefault(e["forecast_id"], e)
    scored, skipped = [], []
    for forecast in store.forecasts(100000):
        if forecast["kind"] != "race" or forecast["evidence_kind"] == "illustrative_scenario":
            continue
        actual = sorted(
            [r for r in results if r["event_id"] == forecast["event"]["event_id"] and r["session"] == "race"],
            key=lambda r: r["driver_id"],
        )
        if not actual:
            continue
        outcome_hash = digest(actual)
        old = existing.get(forecast["id"])
        if (
            old
            and old.get("actual_content_hash") == outcome_hash
            and old.get("scoring_version") == SCORING_VERSION
        ):
            continue
        try:
            metrics = distribution_metrics(forecast["drivers"], actual, forecast["simulations"])
        except ValueError as exc:
            skipped.append({"forecast_id": forecast["id"], "reason": str(exc)})
            continue
        scored.append(
            store.add_evidence(
                {
                    "kind": "forecast_score",
                    "forecast_id": forecast["id"],
                    "revision_of": old["id"] if old else None,
                    "evidence_kind": forecast["evidence_kind"],
                    "event_id": forecast["event"]["event_id"],
                    "stage": forecast["request"]["stage"],
                    "model_version": forecast["model_version"],
                    "forecast_code_version": forecast["code_version"],
                    "actual_data_version": version,
                    "actual_content_hash": outcome_hash,
                    "actual_source_ids": sorted({r["source_id"] for r in actual}),
                    "forecast_ledger_hash": forecast["ledger_hash"],
                    **metrics,
                }
            )
        )
    return {"scored": len(scored), "evidence_ids": scored, "unscorable": skipped}


def scorecards(store: Store):
    from collections import defaultdict

    forecasts = {f["id"]: f for f in store.forecasts(100000)}
    latest = {}
    for row in store.evidence():
        if row["kind"] == "forecast_score":
            latest.setdefault(row["forecast_id"], row)
    groups = defaultdict(dict)
    for identifier, score in latest.items():
        f = forecasts.get(identifier)
        if not f or "calibration_pairs" not in score:
            continue
        key = (
            f["evidence_kind"],
            f["request"]["stage"],
            f["model_version"],
            f["code_version"],
            score["scoring_version"],
        )
        event = f["event"]["event_id"]
        old = groups[key].get(event)
        if not old or f["created_at"] < old[0]["created_at"]:
            groups[key][event] = (f, score)
    output = []
    for key, selected in sorted(groups.items()):
        rows = [
            {**s, "season": f["event"]["season"], "circuit": f["event"]["circuit_id"], "weather": "unknown"}
            for f, s in selected.values()
        ]
        output.append(
            {
                "evidence_kind": key[0],
                "stage": key[1],
                "model_version": key[2],
                "code_version": key[3],
                "scoring_version": key[4],
                "selection_policy": "First recorded forecast per event, stage, model and code version; latest corrected outcome score",
                "forecast_ids": [f["id"] for f, _ in selected.values()],
                "report": report(rows),
            }
        )
    return output
