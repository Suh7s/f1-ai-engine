from datetime import timedelta
import json
import numpy as np
import pytest
from f1engine.sources import session_time, sync_fastf1, PublicClient, sync_season
from f1engine.contracts import Unavailable, SeasonRequest, timestamp
from f1engine.features import build_features
from f1engine.recorder import normalize, TOPICS
from f1engine.models import fit_candidate, predict_candidate, publish_model, load_model, predict
from f1engine.evaluation import examples, calibrate
from f1engine.simulation import simulate_season, reproduce, forecast


def test_calendar_date_parsing():
    assert timestamp(session_time({"date": "2026-10-25", "time": "20:00:00Z"}, "race")).hour == 20
    assert (
        timestamp(
            session_time(
                {
                    "date": "2026-10-25",
                    "time": "20:00:00Z",
                    "Qualifying": {"date": "2026-10-24", "time": "19:00:00Z"},
                },
                "qualifying",
            )
        ).day
        == 24
    )


def test_failed_network_update_preserves_dataset(store, monkeypatch):
    old = store.current_version()
    monkeypatch.setattr(PublicClient, "get", lambda *a, **k: (_ for _ in ()).throw(Unavailable("offline")))
    with pytest.raises(Unavailable):
        sync_season(store, 2024)
    assert store.current_version() == old


def test_fastf1_import_and_empty_download_does_not_erase_laps(store, monkeypatch):
    import fastf1, pandas as pd

    class Session:
        laps = pd.DataFrame()
        date = pd.Timestamp("2026-04-03")

        def load(self, **kwargs):
            assert kwargs["telemetry"] is False

    monkeypatch.setattr(fastf1, "get_session", lambda *a: Session())
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda *a: None)
    old = store.current_version()
    count = len(store.table("laps"))
    with pytest.raises(Unavailable, match="previous telemetry retained"):
        sync_fastf1(store, 2026, 3, "FP1")
    assert store.current_version() == old
    assert len(store.table("laps")) == count


@pytest.mark.parametrize("absolute", [False, True])
def test_fastf1_lap_clock_uses_session_origin(store, monkeypatch, absolute):
    import fastf1, pandas as pd

    class Session:
        date = pd.Timestamp("2026-04-03T14:00:00")
        session_start_time = pd.Timedelta(hours=1)
        laps = pd.DataFrame(
            [
                {
                    "DriverNumber": "1",
                    "LapNumber": 2,
                    "LapTime": pd.Timedelta(seconds=90),
                    "Time": pd.Timedelta(hours=1, minutes=3),
                    "Compound": "MEDIUM",
                    "TyreLife": 2,
                    "TrackStatus": "1",
                    "IsAccurate": True,
                    "LapStartDate": pd.Timestamp("2026-04-03T14:11:30") if absolute else pd.NaT,
                }
            ]
        )

        def load(self, **kwargs):
            assert kwargs["telemetry"] is False

    monkeypatch.setattr(fastf1, "get_session", lambda *a: Session())
    monkeypatch.setattr(fastf1.Cache, "enable_cache", lambda *a: None)
    publish = store.publish
    monkeypatch.setattr(store, "publish", lambda changes: publish(changes, illustrative=True))
    sync_fastf1(store, 2026, 3, "FP1")
    row = next(
        r
        for r in store.table("laps")
        if r["event_id"] == "2026-03" and r["driver_id"] == "driver_00" and r["lap"] == 2
    )
    assert row["available_at"] == ("2026-04-03T14:13:00+00:00" if absolute else "2026-04-03T14:03:00+00:00")
    assert row["availability_basis"] == (
        "reconstructed_absolute_lap_end" if absolute else "reconstructed_scheduled_session_clock"
    )


@pytest.mark.parametrize("failure", [None, "empty", "wrong_round"])
def test_historical_standings_publish_both_tables_atomically(store, monkeypatch, failure):
    import f1engine.sources as sources

    event_id = "2024-02"
    old = store.current_version()
    before = {name: store.table(name) for name in ("standings", "constructor_standings")}
    source = store.raw("jolpica", "fixture://standings", {"test": True})

    def pages(client, suffix):
        assert suffix.startswith("2024/2/")
        driver = suffix.endswith("driverStandings")
        table = "standings" if driver else "constructor_standings"
        group = "DriverStandings" if driver else "ConstructorStandings"
        rows = []
        for row in before[table]:
            if row["event_id"] != event_id:
                continue
            item = {"points": str(row["points"] + 1), "wins": "0"}
            if driver:
                item.update(
                    Driver={"driverId": row["driver_id"]}, Constructors=[{"constructorId": "wrong_seat"}]
                )
            else:
                item["Constructor"] = {"constructorId": row["team_id"]}
            rows.append(item)
        if not driver and failure == "empty":
            rows = []
        yield (
            {
                "StandingsLists": [
                    {"season": "2024", "round": "3" if failure == "wrong_round" else "2", group: rows}
                ]
            },
            source,
        )

    monkeypatch.setattr(sources, "_race_payloads", pages)
    publish = store.publish
    monkeypatch.setattr(store, "publish", lambda changes: publish(changes, illustrative=True))
    if failure:
        with pytest.raises(Unavailable):
            sources.sync_standings(store, event_id)
        assert store.current_version() == old
    else:
        sources.sync_standings(store, event_id)
        for table, rows in before.items():
            current = store.table(table)
            assert len(current) == len(rows)
            assert [r for r in current if r["event_id"] != event_id] == [
                r for r in rows if r["event_id"] != event_id
            ]
        updated = next(r for r in store.table("standings") if r["event_id"] == event_id)
        assert updated["team_id"] != "wrong_seat"
        assert store.table("standings", old) == before["standings"]


def test_climate_prior_only_from_past(store, request_):
    raw = store.raw("climate", "fixture://climate", {})
    row = {
        "event_id": request_.event_id,
        "history_end": "2025-12-31",
        "rain_probability": 0.3,
        "available_at": "2026-01-01T00:00:00Z",
        "source_id": raw,
        "availability_basis": "reconstructed_climate_prior",
    }
    store.publish({"climate": [row]}, illustrative=True)
    assert build_features(store, request_)["rows"][0]["weather_basis"] == "climate_prior"
    store.publish(
        {
            "climate": [
                {
                    **row,
                    "history_end": "2026-12-31",
                    "available_at": "2027-01-01T00:00:00Z",
                    "rain_probability": 1,
                }
            ]
        },
        illustrative=True,
    )
    assert build_features(store, request_)["rows"][0]["rain_probability"] == 0.3


def test_explicit_source_resolution(store):
    row = store.table("entries")[0]
    source = store.raw("openf1", "fixture://supplement", {})
    store.publish({"entries": [{**row, "team_id": "conflict", "source_id": source}]}, illustrative=True)
    e = store.evidence()[0]
    assert e["kind"] == "source_resolution"
    assert "rule" in e["resolutions"][0]


def test_free_recorder_normalization_tracks_arrival_and_laps():
    envelope = {
        "received_at": "2026-04-05T14:01:32Z",
        "payload": [
            "TimingData",
            {"Lines": {"1": {"NumberOfLaps": 1, "LastLapTime": {"Value": "1:30.000"}}}},
            "2026-04-05T14:01:30Z",
        ],
    }
    rows = normalize([envelope], "2026-03", "2026-04-05T14:00:00Z", {"1": "driver_00"})
    assert rows[0]["elapsed"] == 90
    assert rows[0]["lap_seconds"] == 90
    assert timestamp(rows[0]["received_at"]) > timestamp(rows[0]["occurred_at"])
    assert "CarData.z" not in TOPICS and "Position.z" not in TOPICS


def test_xgboost_json_publication_and_faithful_contributions(store):
    ex = examples(store, store.current_version())
    meta, model = fit_candidate(ex[:8], family="xgboost", parameters={"max_depth": 2})
    meta = calibrate(meta, model, ex[8:12])
    identifier = publish_model(store, meta, model)
    loaded, ranker, _ = load_model(store, identifier)
    a = predict_candidate(ex[12][0], meta, model)
    b = predict_candidate(ex[12][0], loaded, ranker)
    np.testing.assert_allclose(a["scores"], b["scores"])
    np.testing.assert_allclose([sum(c.values()) for c in b["explanations"]], b["scores"], atol=1e-5)


def test_season_reproduction(store, request_):
    f = simulate_season(store, SeasonRequest(season=2026, as_of_utc=request_.as_of_utc, simulations=100))
    assert reproduce(store, f["id"])["match"]


def test_active_model_resolution_saved_and_reproducible(store, request_):
    f = forecast(store, request_.model_copy(update={"model_version": "active"}))
    assert f["request"]["model_version"] == "baseline:rating"
    assert reproduce(store, f["id"])["match"]


def test_linear_identity_shrinkage_and_stage_guard(store, request_):
    ex = examples(store, store.current_version())
    meta, m = fit_candidate(ex[:8])
    meta = calibrate(meta, m, ex[8:12])
    meta["prediction_stage"] = "post_qualifying"
    assert any(c.startswith("driver:") for c in meta["columns"])
    identifier = publish_model(store, meta)
    with pytest.raises(Unavailable, match="different prediction stage"):
        predict(build_features(store, request_), store, identifier)


@pytest.mark.parametrize(
    "status,retired,disqualified",
    [
        ("Finished", False, False),
        ("Lapped", False, False),
        ("+1 Lap", False, False),
        ("+2 Laps", False, False),
        ("Retired", True, False),
        ("Did not start", True, False),
        ("Disqualified", False, True),
        ("Excluded", False, True),
        ("Engine", True, False),
    ],
)
def test_jolpica_outcome_vocabulary(status, retired, disqualified):
    from f1engine.sources import outcome_flags

    flags = outcome_flags(status)
    assert flags["retired"] == retired
    assert flags["disqualified"] == disqualified
