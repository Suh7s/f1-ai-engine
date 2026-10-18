"""Preregistered paired forecasts and one fixed-size prospective promotion cohort."""

from .contracts import ForecastRequest, Unavailable, code_version, now_utc, timestamp
from .models import load_model
from .simulation import forecast
from .evaluation import promotion_gate, report, score_ledger


def frozen_candidate(store, identifier):
    freeze = next(
        (e for e in store.evidence() if e["id"] == identifier and e["kind"] == "candidate_freeze"), None
    )
    if not freeze:
        raise Unavailable("Unknown frozen candidate")
    if freeze["code_version"] != code_version():
        raise Unavailable("Code changed since candidate freeze; register a new candidate")
    if freeze.get("illustrative"):
        raise Unavailable("Illustrative models cannot enter prospective promotion")
    load_model(store, freeze["model_version"])
    return freeze


def collect_pair(store, parameters, checkpoint=None):
    checkpoint = {} if checkpoint is None else checkpoint
    freeze = frozen_candidate(store, parameters["freeze_id"])
    event_id = parameters["event_id"]
    existing = next(
        (
            e
            for e in store.evidence()
            if e["kind"] == "prospective_pair"
            and e["freeze_id"] == freeze["id"]
            and e["event_id"] == event_id
        ),
        None,
    )
    if existing:
        return existing
    cutoff = checkpoint.setdefault("as_of_utc", now_utc())
    version = checkpoint.setdefault("data_version", store.current_version())
    if timestamp(cutoff) <= timestamp(freeze["created_at"]):
        raise ValueError("Prospective collection must follow candidate registration")
    request = ForecastRequest(
        event_id=event_id,
        stage=freeze["stage"],
        as_of_utc=cutoff,
        data_version=version,
        simulations=10000,
        seed=42,
    )
    ids = []
    for role, model in (("candidate", freeze["model_version"]), ("baseline", freeze["baseline"])):
        if checkpoint.get(role):
            saved = store.forecast(checkpoint[role])
        else:
            saved = forecast(store, request.model_copy(update={"model_version": model}), persist=False)
            if saved["evidence_kind"] != "prospective_forecast":
                raise Unavailable(
                    "Both forecasts must be recorded before the event using current captured inputs"
                )
            saved = store.save_forecast(
                {**saved, "prospective_registration": freeze["id"], "comparison_role": role}
            )
            checkpoint[role] = saved["id"]
        ids.append(saved["id"])
    body = {
        "kind": "prospective_pair",
        "evidence_kind": "prospective_forecast",
        "freeze_id": freeze["id"],
        "event_id": event_id,
        "stage": freeze["stage"],
        "candidate_forecast": ids[0],
        "baseline_forecast": ids[1],
        "as_of_utc": cutoff,
        "data_version": version,
        "simulations": 10000,
        "selection_policy": "First recorded pair per event; first 15 registered events ordered by race start; evaluate once when every outcome is available",
    }
    identifier = store.add_evidence(body)
    return {**body, "id": identifier}


def evaluate_prospective(store, parameters):
    freeze = frozen_candidate(store, parameters["freeze_id"])
    existing = next(
        (
            e
            for e in store.evidence()
            if e["kind"] == "prospective_comparison" and e["freeze_id"] == freeze["id"]
        ),
        None,
    )
    if existing:
        return existing
    pairs = [
        e for e in store.evidence() if e["kind"] == "prospective_pair" and e["freeze_id"] == freeze["id"]
    ]
    # Freeze the selection rule before any outcomes; no repeated looks at a growing cohort.
    pairs.sort(key=lambda e: (store.forecast(e["candidate_forecast"])["event"]["start_at"], e["created_at"]))
    unique = {}
    for pair in pairs:
        unique.setdefault(pair["event_id"], pair)
    cohort = list(unique.values())[:15]
    if len(cohort) < 15:
        return {
            "status": "collecting",
            "registered_events": len(cohort),
            "required_events": 15,
            "promoted": False,
        }
    score_ledger(store)
    scores = {}
    for row in store.evidence():
        if row["kind"] == "forecast_score":
            scores.setdefault(row["forecast_id"], row)
    missing = [
        p["event_id"]
        for p in cohort
        if p["candidate_forecast"] not in scores or p["baseline_forecast"] not in scores
    ]
    if missing:
        return {"status": "awaiting_outcomes", "missing_events": missing, "promoted": False}
    candidate = [scores[p["candidate_forecast"]] for p in cohort]
    baseline = [scores[p["baseline_forecast"]] for p in cohort]
    for c, b in zip(candidate, baseline):
        if (
            c["actual_content_hash"] != b["actual_content_hash"]
            or c["scoring_version"] != b["scoring_version"]
        ):
            raise Unavailable("Paired forecasts have inconsistent outcome or scoring versions")
    gate = promotion_gate(candidate, baseline)
    body = {
        "kind": "prospective_comparison",
        "evidence_kind": "prospective_forecast",
        "freeze_id": freeze["id"],
        "cohort": [p["id"] for p in cohort],
        "candidate": report(candidate),
        "baseline": report(baseline),
        "candidate_rows": candidate,
        "baseline_rows": baseline,
        "gate": gate,
        "promoted": bool(gate["promote"]),
    }
    identifier = store.add_evidence(body)
    if gate["promote"]:
        with store.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO pointers VALUES (?,?)",
                ("model:" + freeze["stage"], freeze["model_version"]),
            )
        store.add_evidence(
            {
                "kind": "model_promotion",
                "model_version": freeze["model_version"],
                "evidence_id": identifier,
                "gate": gate,
            }
        )
    return {**body, "id": identifier}
