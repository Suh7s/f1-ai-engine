import numpy as np
import pytest
from f1engine.simulation import sample_race, forecast, simulate_season, reproduce
from f1engine.contracts import SeasonRequest, Unavailable
from f1engine.rules import Rules, countback_order, classification
from f1engine.demo import DEMO_CUTOFF


@pytest.mark.parametrize(
    "season,sprint,first,total",
    [
        (2014, False, 25, 101),
        (2021, True, 3, 6),
        (2022, True, 8, 36),
        (2025, False, 25, 101),
        (2026, False, 25, 101),
    ],
)
def test_season_specific_points(season, sprint, first, total):
    schedule = Rules(season).schedule(20, sprint=sprint)
    assert schedule[0] == first
    assert schedule.sum() == total


def test_double_points_and_fastest_lap():
    assert Rules(2014).schedule(20, final_round=True).sum() == 202
    assert not Rules(2018).fastest_lap_bonus()
    assert Rules(2024).fastest_lap_bonus()
    assert not Rules(2025).fastest_lap_bonus()
    assert not Rules(2026).fastest_lap_bonus()
    assert not Rules(2024).fastest_lap_bonus(sprint=True)


def test_shortened_races():
    assert Rules(2021).schedule(20, distance=0.5)[0] == 12.5
    assert Rules(2023).schedule(20, distance=0.2)[0] == 6
    assert Rules(2023).schedule(20, distance=0.4)[0] == 13
    assert Rules(2023).schedule(20, distance=0.6)[0] == 19
    assert Rules(2023).schedule(20, green_laps=1).sum() == 0
    assert Rules(2023).schedule(20, sprint=True, distance=0.4).sum() == 0


def test_countback_wins_then_seconds():
    points = np.array([[100.0, 100.0, 100.0]])
    counts = np.array([[[2, 3, 0], [2, 4, 0], [1, 10, 0]]])
    assert countback_order(points, counts, np.random.default_rng(42)).tolist() == [[1, 0, 2]]


def test_full_permutations_and_probability_conservation(store, request_):
    rng = np.random.default_rng(42)
    sample = sample_race(np.arange(20) / 20, np.full(20, 0.15), 2026, 10000, rng)
    assert np.all(np.sort(sample["order"], axis=1) == np.arange(20))
    assert np.all(np.sort(sample["position"], axis=1) == np.arange(1, 21))
    f = forecast(store, request_, persist=False)
    d = f["drivers"]
    p = np.array([r["position_probabilities"] for r in d])
    np.testing.assert_allclose(p.sum(axis=0), 1)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    for r in d:
        assert r["win_probability"] == pytest.approx(r["position_probabilities"][0])
        assert r["podium_probability"] == pytest.approx(sum(r["position_probabilities"][:3]))
        for other in d:
            if r == other:
                continue
            assert f["head_to_head"][r["driver_id"]][other["driver_id"]] + f["head_to_head"][
                other["driver_id"]
            ][r["driver_id"]] == pytest.approx(1)


def test_retirement_classification_completed_laps():
    strengths = np.array([[100, 0, -100.0]])
    completed = np.array([[40, 59, 60]])
    assert classification(strengths, completed, np.random.default_rng(3)).tolist() == [[2, 1, 0]]


def test_season_consistency_and_already_completed_sprint(store):
    request = SeasonRequest(season=2026, as_of_utc=DEMO_CUTOFF, simulations=1000)
    a = simulate_season(store, request, persist=False)
    b = simulate_season(store, request, persist=False)
    assert a == b
    assert sum(r["title_probability"] for r in a["drivers"]) == pytest.approx(1)
    standings = {r["driver_id"]: r["points"] for r in store.table("standings") if r["event_id"] == "2026-02"}
    sprint = {
        r["driver_id"]: r["points"]
        for r in store.table("results")
        if r["event_id"] == "2026-03" and r["session"] == "sprint"
    }
    for d in a["drivers"]:
        assert d["current_points"] == standings[d["driver_id"]] + sprint[d["driver_id"]]


def test_future_actuals_cannot_change_historical_season(store):
    r = SeasonRequest(season=2026, as_of_utc=DEMO_CUTOFF, simulations=200)
    a = simulate_season(store, r, persist=False)
    future = [
        {**x, "event_id": "2026-04", "available_at": "2026-04-19T17:00:00Z"}
        for x in store.table("results")
        if x["event_id"] == "2026-02" and x["session"] == "race"
    ]
    assert len(future) == len(a["drivers"])
    store.publish({"results": future}, illustrative=True)
    b = simulate_season(store, r, persist=False)
    assert a["drivers"] == b["drivers"]
    assert a["constructors"] == b["constructors"]
    store.publish({"results": [{**x, "points": 999, "position": 1} for x in future]}, illustrative=True)
    c = simulate_season(store, r, persist=False)
    assert a["drivers"] == c["drivers"]
    assert a["constructors"] == c["constructors"]


def test_championship_revisions_are_linked_and_cannot_cross_seasons(store):
    request = SeasonRequest(season=2026, as_of_utc=DEMO_CUTOFF, simulations=200)
    original = simulate_season(store, request)
    revised = simulate_season(store, request.model_copy(update={"seed": 43, "revision_of": original["id"]}))
    assert revised["request"]["revision_of"] == original["id"]
    assert reproduce(store, revised["id"])["match"]
    with pytest.raises(ValueError, match="same forecast"):
        store.save_forecast({**revised, "request": {**revised["request"], "season": 2025}})
    assert len(store.forecasts()) == 2
    assert store.verify_ledger()["valid"]


def test_disqualification_distinct_from_retirement():
    result = sample_race(
        np.array([10.0, 0.0, -1.0]),
        np.zeros(3),
        2026,
        100,
        np.random.default_rng(4),
        disqualification=np.array([1.0, 0.0, 0.0]),
    )
    assert np.all(result["position"][:, 0] == 3)
    assert np.all(result["points"][:, 0] == 0)
    assert not result["retired"][:, 0].any()
    assert result["disqualified"][:, 0].all()


def test_no_points_if_all_cars_stop_before_two_laps():
    result = sample_race(
        np.array([0.0, 0.0]), np.ones(2), 2026, 100, np.random.default_rng(4), beta=(0.01, 100.0)
    )
    assert not result["points"].any()
