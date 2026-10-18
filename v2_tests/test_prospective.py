import pytest
from f1engine.evaluation import holdout
from f1engine import prospective
from f1engine.contracts import Unavailable


def test_new_database_does_not_reseal_published_holdout(store):
    assert not any(e["kind"] == "holdout_opened" for e in store.evidence())
    with pytest.raises(ValueError, match="already inspected 2025"):
        holdout(store, {})


def test_prospective_report_waits_for_fixed_cohort_without_looking_at_scores(store, monkeypatch):
    freeze = {"id": "registration", "stage": "pre_weekend"}
    monkeypatch.setattr(prospective, "frozen_candidate", lambda *a: freeze)
    monkeypatch.setattr(prospective, "score_ledger", lambda *a: pytest.fail("No interim score look allowed"))
    result = prospective.evaluate_prospective(store, {"freeze_id": "registration"})
    assert result == {
        "status": "collecting",
        "registered_events": 0,
        "required_events": 15,
        "promoted": False,
    }


def test_collection_rejects_reconstruction_and_never_saves_it_as_prospective(store, monkeypatch):
    freeze = {
        "id": "registration",
        "stage": "pre_weekend",
        "created_at": "2020-01-01T00:00:00Z",
        "model_version": "baseline:rating",
        "baseline": "baseline:grid",
    }
    monkeypatch.setattr(prospective, "frozen_candidate", lambda *a: freeze)
    monkeypatch.setattr(
        prospective, "forecast", lambda *a, **k: {"evidence_kind": "historical_reconstruction"}
    )
    with pytest.raises(Unavailable, match="before the event"):
        prospective.collect_pair(store, {"freeze_id": "registration", "event_id": "2020-01"})
    assert store.forecasts() == []


def test_already_recorded_pair_is_idempotent(store, monkeypatch):
    freeze = {"id": "registration"}
    monkeypatch.setattr(prospective, "frozen_candidate", lambda *a: freeze)
    monkeypatch.setattr(prospective, "forecast", lambda *a, **k: pytest.fail("Pair must not be regenerated"))
    identifier = store.add_evidence(
        {
            "kind": "prospective_pair",
            "freeze_id": "registration",
            "event_id": "2026-03",
            "candidate_forecast": "a",
            "baseline_forecast": "b",
        }
    )
    assert (
        prospective.collect_pair(store, {"freeze_id": "registration", "event_id": "2026-03"})["id"]
        == identifier
    )
