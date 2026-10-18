import json
import time
from datetime import timedelta
import pytest
from fastapi.testclient import TestClient
from f1engine.api import create_app
from f1engine.contracts import Unavailable, timestamp, ForecastRequest
from f1engine.demo import replay_fixture, DEMO_EVENT
from f1engine.replay import replay
from f1engine.simulation import forecast
from f1engine import jobs


def test_replay_no_future_deterministic_and_duplicates():
    messages = replay_fixture(drivers=12)
    at = messages[23]["received_at"]
    first = replay(messages, DEMO_EVENT, at)
    truncated = replay(messages[:24], DEMO_EVENT, at)
    assert first == truncated
    assert replay(messages[::-1] + [messages[0]], DEMO_EVENT, at) == first
    assert first["message_count"] == 24


def test_corrected_message_visible_only_when_received():
    messages = replay_fixture(drivers=12)
    at = messages[23]["received_at"]
    correction = {
        **messages[12],
        "revision": 1,
        "elapsed": 999,
        "received_at": (timestamp(at) + timedelta(seconds=2)).isoformat(),
    }
    before = replay(messages + [correction], DEMO_EVENT, at)
    after = replay(messages + [correction], DEMO_EVENT, correction["received_at"])
    assert before["visible_hash"] != after["visible_hash"]
    assert next(d for d in after["drivers"] if d["driver_id"] == correction["driver_id"])["elapsed"] == 999


@pytest.mark.parametrize("future_field", ["received_at", "occurred_at"])
def test_malformed_future_payload_cannot_break_earlier_replay(future_field):
    messages = replay_fixture(drivers=12)[:24]
    at = messages[-1]["received_at"]
    future_time = (timestamp(at) + timedelta(seconds=2)).isoformat()
    malformed = {**messages[-1], "id": "future-invalid", "lap": -1, future_field: future_time}
    assert replay(messages + [malformed], DEMO_EVENT, at) == replay(messages, DEMO_EVENT, at)
    # Once the observation is visible, invalid input must fail transparently.
    with pytest.raises(ValueError):
        replay(messages + [malformed], DEMO_EVENT, future_time)


def test_conflicting_duplicate_rejected():
    m = replay_fixture(drivers=12)[:12]
    with pytest.raises(ValueError, match="Conflicting"):
        replay(m + [{**m[0], "elapsed": 3}], DEMO_EVENT, m[-1]["received_at"])


def test_control_updates_without_a_new_lap_and_pit_laps_excluded():
    m = replay_fixture(drivers=12)[:24]
    m[-1] = {**m[-1], "pit": True, "lap_seconds": 300}
    at = (timestamp(m[-1]["received_at"]) + timedelta(seconds=1)).isoformat()
    control = {
        "id": "red",
        "kind": "race_control",
        "event_id": DEMO_EVENT,
        "occurred_at": at,
        "received_at": at,
        "status": "red",
    }
    state = replay(m + [control], DEMO_EVENT, at)
    assert state["status"] == "red"
    driver = next(r for r in state["drivers"] if r["driver_id"] == m[-1]["driver_id"])
    assert driver["pace_laps"] == 1 and driver["pace_seconds"] < 100


def test_missing_distance_and_individual_stale_driver_block_live(store):
    messages = replay_fixture(drivers=12)
    cutoff = messages[-1]["received_at"]
    # One driver's stream disappears while everybody else continues to update.
    m = [r for r in messages if r["driver_id"] != "driver_00" or r["lap"] == 1]
    state = replay(m, DEMO_EVENT, cutoff)
    req = ForecastRequest(event_id=DEMO_EVENT, stage="live", as_of_utc=cutoff, simulations=100)
    assert not state["stale"]
    with pytest.raises(Unavailable, match="stale timing"):
        forecast(store, req, live_state=state)
    event = next(e for e in store.table("events") if e["event_id"] == DEMO_EVENT)
    store.publish({"events": [{**event, "scheduled_laps": None}]}, illustrative=True)
    with pytest.raises(Unavailable, match="scheduled race distance"):
        forecast(store, req, live_state=replay(messages, DEMO_EVENT, cutoff))


def test_replay_stale_and_reconnect(store):
    m = replay_fixture(drivers=12)
    state = replay(m, DEMO_EVENT, m[11]["received_at"])
    assert not state["stale"]
    stale = replay(m[:12], DEMO_EVENT, "2026-04-05T14:20:00Z")
    assert stale["stale"]
    req = ForecastRequest(event_id=DEMO_EVENT, stage="live", as_of_utc=stale["as_of_utc"], simulations=100)
    with pytest.raises(Unavailable, match="stale"):
        forecast(store, req, live_state=stale)
    assert not replay(m, DEMO_EVENT, m[23]["received_at"])["stale"]


def test_replay_retired_driver_remains_retired(store):
    m = replay_fixture(drivers=12)[:24]
    m[-1] = {**m[-1], "retired": True}
    state = replay(m, DEMO_EVENT, m[-1]["received_at"])
    result = forecast(
        store,
        ForecastRequest(event_id=DEMO_EVENT, stage="live", as_of_utc=state["as_of_utc"], simulations=200),
        live_state=state,
    )
    assert (
        next(d for d in result["drivers"] if d["driver_id"] == m[-1]["driver_id"])["retirement_probability"]
        == 1
    )


def test_durable_jobs_recover_cancel_resume(store, monkeypatch):
    j = jobs.submit(store, "score", launch=False)
    jobs.update(store, j["id"], status="running", pid=None, launched_epoch=0, checkpoint={"completed": 3})
    jobs.recover(store)
    assert jobs.get_job(store, j["id"])["status"] == "interrupted"
    jobs.resume(store, j["id"], launch=False)
    assert jobs.get_job(store, j["id"])["checkpoint"] == {"completed": 3}
    jobs.cancel(store, j["id"])
    jobs.run(store, j["id"])
    assert jobs.get_job(store, j["id"])["status"] == "cancelled"
    jobs.resume(store, j["id"], launch=False)
    monkeypatch.setattr(jobs, "dispatch", lambda *args: {"ok": True})
    jobs.run(store, j["id"])
    assert jobs.get_job(store, j["id"])["status"] == "completed"


def test_job_time_budget_pauses_and_preserves_checkpoint(store, monkeypatch):
    j = jobs.submit(store, "score", launch=False, budget_seconds=1)

    def work(store, kind, p, check, progress, checkpoint):
        checkpoint["completed"] = 2
        time.sleep(2)

    monkeypatch.setattr(jobs, "dispatch", work)
    jobs.run(store, j["id"])
    result = jobs.get_job(store, j["id"])
    assert result["status"] == "paused"
    assert result["checkpoint"] == {"completed": 2}


def test_api_all_workspaces_missing_stale_and_failure(store, request_):
    with TestClient(create_app(store.root)) as c:
        assert c.get("/api/v1/events").status_code == 200
        assert c.get("/api/v1/health").json()["ledger"]["valid"]
        f = c.post("/api/v1/forecasts", json=request_.model_dump(mode="json"))
        assert f.status_code == 200
        assert c.post("/api/v1/features", json=request_.model_dump(mode="json")).status_code == 200
        assert (
            c.post(
                "/api/v1/championship",
                json={"season": 2026, "as_of_utc": str(request_.as_of_utc), "simulations": 100},
            ).status_code
            == 200
        )
        assert c.get("/api/v1/evidence").status_code == 200
        assert c.get("/api/v1/models").status_code == 200
        assert c.get("/api/v1/jobs").status_code == 200
        assert c.get("/api/v1/sources/" + f.json()["source_ids"][0]).status_code == 200
        assert (
            c.post(
                "/api/v1/forecasts", json={**request_.model_dump(mode="json"), "model_version": "missing"}
            ).status_code
            == 409
        )
        assert len(c.get("/api/v1/forecasts").json()) == 2
        assert (
            c.post(
                "/api/v1/forecasts",
                json=request_.model_dump(mode="json"),
                headers={"Origin": "https://evil.example"},
            ).status_code
            == 403
        )
        assert c.get("/api/v1/health", headers={"Host": "evil.example"}).status_code == 403
