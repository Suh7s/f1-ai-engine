import numpy as np
import pytest
from f1engine.scoring import distribution_metrics
from f1engine.evaluation import score_ledger, scorecards, score_samples
from f1engine.simulation import forecast, sample_race
from f1engine.features import build_features


def test_saved_forecast_scores_match_sample_scoring(store, request_):
    frame = build_features(store, request_)
    n = len(frame["rows"])
    samples = sample_race(np.zeros(n), np.zeros(n), 2026, 1000, np.random.default_rng(9))
    actual = [
        {"driver_id": r["driver_id"], "position": j + 1, "points": int(j < 10), "retired": False}
        for j, r in enumerate(frame["rows"])
    ]
    drivers = [
        {
            "driver_id": r["driver_id"],
            "position_probabilities": (
                np.bincount(samples["position"][:, j], minlength=n + 1)[1:] / 1000
            ).tolist(),
            "points_probability": float((samples["points"][:, j] > 0).mean()),
            "retirement_probability": 0.0,
        }
        for j, r in enumerate(frame["rows"])
    ]
    a = score_samples(frame, actual, samples)
    b = distribution_metrics(drivers, actual, 1000)
    assert all(a[k] == v for k, v in b.items())
    with pytest.raises(ValueError, match="every entrant"):
        distribution_metrics(drivers, actual + [actual[0]], 1000)


def test_outcome_corrections_append_scores_and_scorecards_do_not_count_repeats(store, request_):
    body = forecast(store, request_, persist=False)
    body["evidence_kind"] = "historical_reconstruction"
    first = store.save_forecast(body)
    second = store.save_forecast(body)
    source = store.raw("test", "fixture://outcome", {"version": 1})
    actual = [
        {
            "event_id": request_.event_id,
            "session": "race",
            "driver_id": r["driver_id"],
            "team_id": r["team_id"],
            "position": j + 1,
            "points": int(j < 10),
            "retired": False,
            "available_at": "2026-04-05T17:00:00Z",
            "availability_basis": "illustrative_fixture",
            "source_id": source,
        }
        for j, r in enumerate(body["drivers"])
    ]
    store.publish({"results": actual}, illustrative=True)
    assert score_ledger(store)["scored"] == 2
    assert score_ledger(store)["scored"] == 0
    cards = scorecards(store)
    assert len(cards) == 1 and cards[0]["report"]["races"] == 1
    assert cards[0]["forecast_ids"] == [first["id"]]
    source2 = store.raw("test", "fixture://outcome", {"version": 2})
    changed = [
        {
            **r,
            "position": 2 if r["position"] == 1 else 1 if r["position"] == 2 else r["position"],
            "source_id": source2,
            "available_at": "2026-04-06T00:00:00Z",
        }
        for r in actual
    ]
    store.publish({"results": changed}, illustrative=True)
    assert score_ledger(store)["scored"] == 2
    scores = [e for e in store.evidence() if e["kind"] == "forecast_score"]
    assert len(scores) == 4 and all(e["revision_of"] for e in scores[:2])
    assert scorecards(store)[0]["report"]["races"] == 1
