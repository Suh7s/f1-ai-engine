import json
import sqlite3
from datetime import timedelta
import numpy as np
import pytest
from f1engine.contracts import ForecastRequest, Scenario, Unavailable, timestamp
from f1engine.features import build_features, matrix
from f1engine.simulation import forecast, reproduce
from f1engine.models import baseline, publish_model, load_model, fit_candidate
from f1engine.evaluation import examples, calibrate, promotion_gate


def test_startup_without_files(tmp_path):
    from f1engine.api import create_app
    from fastapi.testclient import TestClient

    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/v1/health").json()["status"] == "ready"
        assert client.get("/api/v1/events").json() == []


def test_history_and_telemetry_preserved(store):
    old = store.current_version()
    counts = {k: len(store.table(k)) for k in ("results", "laps", "entries")}
    assert store.publish({"results": [], "laps": []}, illustrative=True) == old
    new = dict(store.table("events")[-1])
    new["name"] = "Updated calendar"
    newversion = store.publish({"events": [new]}, illustrative=True)
    assert old != newversion
    for k, n in counts.items():
        assert len(store.table(k)) == n
    assert store.table("events", old)[-1]["name"] != "Updated calendar"


def test_failed_publication_preserves_pointer(store):
    old = store.current_version()
    with pytest.raises(ValueError):
        store.publish({"entries": [{"event_id": "2026-03", "driver_id": "new"}]}, illustrative=True)
    assert store.current_version() == old


def test_integrity_validation(store):
    manifest = store.manifest()
    part = manifest["tables"]["results"][0]
    path = store.root / "datasets" / manifest["id"] / part["path"]
    path.write_bytes(b"broken")
    with pytest.raises(Unavailable, match="integrity"):
        store.table("results")


def test_future_records_cannot_change_earlier_features(store, request_):
    a = build_features(store, request_)
    rows = [
        {**r, "position": 99, "points": 1000}
        for r in store.table("results")
        if timestamp(r["available_at"]) > request_.as_of_utc
    ]
    store.publish({"results": rows}, illustrative=True)
    b = build_features(store, request_)
    assert a["feature_hash"] == b["feature_hash"]
    np.testing.assert_allclose(matrix(a), matrix(b), equal_nan=True)


def test_training_inference_same_snapshot(store, request_):
    from f1engine.models import prepare_array

    frame = build_features(store, request_)
    x, med, mean, scale = prepare_array([frame], list(frame["coverage"]))
    y, *_ = prepare_array([build_features(store, request_)], list(frame["coverage"]), med, mean, scale)
    np.testing.assert_array_equal(x, y)


def test_transfer_effective_at_event(store, request_):
    rows = [r for r in store.table("entries") if r["event_id"] == request_.event_id]
    rows[0] = {**rows[0], "team_id": "new_team", "team_name": "New Team"}
    store.publish({"entries": [rows[0]]}, illustrative=True)
    f = build_features(store, request_)
    assert f["rows"][0]["team_id"] == "new_team"
    assert f["rows"][0]["team_strength"] == 0


def test_sprint_stage_inputs_and_cutoff(store, request_):
    frame = build_features(store, request_)
    assert all(r["sprint_points"] is not None for r in frame["rows"])
    pre = build_features(store, request_.model_copy(update={"stage": "pre_weekend"}))
    assert all(r["sprint_fraction"] is None for r in pre["rows"])
    early = request_.model_copy(update={"as_of_utc": request_.as_of_utc - timedelta(days=2)})
    with pytest.raises(Unavailable):
        build_features(store, early)


def test_race_weather_is_never_forecast(store, request_):
    raw = store.raw("test", "fixture://weather", {})
    row = {
        "event_id": request_.event_id,
        "issued_at": "2026-04-01T00:00:00Z",
        "rain_probability": 1,
        "available_at": "2026-04-01T00:00:00Z",
        "source_id": raw,
        "availability_basis": "observed_race_weather",
    }
    store.publish({"weather": [row]}, illustrative=True)
    assert build_features(store, request_)["rows"][0]["rain_probability"] is None


def test_forecast_reproduces_and_ledger_append_only(store, request_):
    f = forecast(store, request_)
    assert reproduce(store, f["id"])["match"]
    corrected = forecast(store, request_.model_copy(update={"seed": 43, "revision_of": f["id"]}))
    assert corrected["request"]["revision_of"] == f["id"]
    assert store.verify_ledger()["valid"]
    with pytest.raises(sqlite3.IntegrityError):
        with store.db() as db:
            db.execute("DELETE FROM forecasts")
    assert len(store.forecasts()) == 2


def test_scenarios_separate_and_source_versions_present(store, request_):
    f = forecast(store, request_.model_copy(update={"scenario": Scenario(strength_delta={"driver_00": 1})}))
    assert f["evidence_kind"] == "illustrative_scenario"
    for key in (
        "code_version",
        "feature_hash",
        "data_version",
        "rules_version",
        "source_ids",
        "seed",
        "simulations",
        "coverage",
        "missing_inputs",
    ):
        assert key in f


def test_naive_dates_and_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        ForecastRequest(event_id="2026-03", as_of_utc="2026-04-05T12:00:00")
    with pytest.raises(ValueError):
        Scenario(rain_probability=1.1)
    with pytest.raises(ValueError):
        Scenario(strength_delta={"d": float("nan")})


def test_candidate_calibration_independent_and_atomic(store, monkeypatch):
    ex = examples(store, store.current_version())
    training = [x for x in ex if x[0]["event"]["season"] <= 2022]
    cal = [x for x in ex if x[0]["event"]["season"] == 2023]
    meta, ranker = fit_candidate(training)
    with pytest.raises(ValueError):
        publish_model(store, meta)
    with pytest.raises(ValueError):
        calibrate(meta, ranker, training)
    meta = calibrate(meta, ranker, cal)
    import f1engine.models as models

    original = models.os.replace
    monkeypatch.setattr(models.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        publish_model(store, meta)
    with store.db() as db:
        assert db.execute("SELECT count(*) FROM models").fetchone()[0] == 0
    monkeypatch.setattr(models.os, "replace", original)
    identifier = publish_model(store, meta)
    assert load_model(store, identifier)[0]["calibrated"]
    (store.root / "models" / identifier / "model.json").write_text("{}")
    with pytest.raises(Unavailable):
        load_model(store, identifier)


def test_nonfinite_model_cannot_publish(store):
    ex = examples(store, store.current_version())
    meta, m = fit_candidate(ex[:8])
    meta = calibrate(meta, m, ex[8:12])
    meta["model"]["coef"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        publish_model(store, meta)


def test_promotion_requires_both_primary_and_logloss():
    baseline_rows = [{"event_id": str(i), "rps": 0.2, "winner_log_loss": 1.0} for i in range(20)]
    better = [{**r, "rps": 0.18, "winner_log_loss": 1.01} for r in baseline_rows]
    assert promotion_gate(better, baseline_rows)["promote"]
    assert not promotion_gate([{**r, "winner_log_loss": 1.03} for r in better], baseline_rows)["promote"]
    assert not promotion_gate(baseline_rows, baseline_rows)["promote"]
    assert not promotion_gate(better[:5], baseline_rows[:5])["promote"]


def test_model_cache_never_survives_new_artifact(store, request_):
    f = forecast(store, request_, persist=False)
    assert (
        forecast(store, request_.model_copy(update={"model_version": "baseline:grid"}), persist=False)[
            "prediction_hash"
        ]
        != f["prediction_hash"]
    )
    with pytest.raises(Unavailable):
        forecast(store, request_.model_copy(update={"model_version": "missing"}))
