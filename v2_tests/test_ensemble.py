import copy
import numpy as np
import pytest
from f1engine.evaluation import examples, calibrate
from f1engine.models import fit_candidate, predict_candidate, publish_model, load_model, finisher_targets
from f1engine.simulation import sample_prediction, forecast, reproduce
from f1engine.contracts import ForecastRequest, Unavailable


def test_calibration_rejects_earlier_disjoint_races(store):
    ex = examples(store, store.current_version())
    meta, model = fit_candidate(ex[8:16])
    with pytest.raises(ValueError, match="follow all training outcomes"):
        calibrate(meta, model, ex[:4])


def test_finisher_performance_is_independent_of_retirement_classification():
    frame = {"rows": [{"driver_id": d} for d in "abcd"]}
    actual = [{"driver_id": d, "position": i + 1, "retired": d in "cd"} for i, d in enumerate("abcd")]
    np.testing.assert_equal(finisher_targets(frame, actual)[:2], [1, 0])
    # A classified retirement ahead of a finisher must not change conditional finisher ranks.
    actual[1]["position"], actual[2]["position"] = 3, 2
    np.testing.assert_equal(finisher_targets(frame, actual)[:2], [1, 0])


@pytest.mark.parametrize("fault", ["coef", "calibration", "scale", "prior"])
def test_invalid_artifact_never_publishes(store, fault):
    ex = examples(store, store.current_version())
    meta, model = fit_candidate(ex[:8])
    meta = calibrate(meta, model, ex[8:12])
    if fault == "coef":
        meta["model"]["coef"] = [0]
    elif fault == "calibration":
        meta["hazard_calibration"] = {"slope": float("nan"), "intercept": 0}
    elif fault == "scale":
        meta["scale"][0] = 0
    else:
        meta["reliability"] = {"prior": 1.1}
    with pytest.raises(ValueError):
        publish_model(store, meta, model)
    assert not list((store.root / "models").iterdir())


def test_ensemble_calibration_publication_joint_outcomes_and_reproduction(store):
    ex = examples(store, store.current_version())
    meta, models = fit_candidate(ex[:16], "ensemble")
    meta = calibrate(meta, models, ex[16:20])
    assert sum(meta["weights"]) == pytest.approx(1)
    assert min(meta["weights"]) >= 0
    identifier = publish_model(store, meta, models)
    loaded, rankers, _ = load_model(store, identifier)
    frame = ex[20][0]
    a = predict_candidate(frame, meta, models)
    b = predict_candidate(frame, loaded, rankers)
    choices = np.arange(600) % 3
    sa = sample_prediction(a, 2022, 600, np.random.default_rng(42), choices=choices)
    sb = sample_prediction(b, 2022, 600, np.random.default_rng(42), choices=choices)
    for key in sa:
        np.testing.assert_equal(sa[key], sb[key])
    np.testing.assert_equal(
        np.sort(sa["order"], axis=1), np.broadcast_to(np.arange(len(frame["rows"])), sa["order"].shape)
    )
    request = ForecastRequest(
        event_id=frame["event"]["event_id"],
        stage="post_qualifying",
        as_of_utc=frame["as_of_utc"],
        model_version=identifier,
        simulations=300,
    )
    f = forecast(store, request)
    assert sum(d["win_probability"] for d in f["drivers"]) == pytest.approx(1)
    assert reproduce(store, f["id"])["match"]
    # A component file changing after publication invalidates the whole mixture.
    path = store.root / "models" / identifier / "component-2.json"
    path.write_text("{}")
    with pytest.raises(Unavailable, match="integrity"):
        load_model(store, identifier)
