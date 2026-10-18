"""Replay archived observations with explicit reconstructed arrival assumptions."""

from datetime import timedelta
import numpy as np
from .contracts import ForecastRequest, Unavailable, timestamp, code_version, digest
from .features import build_features
from .models import baseline
from .session_data import normalize_openf1_laps, track_states, entry_number_map
from .replay import replay, sample_remaining
from .evaluation import score_samples


def archived_messages(bundle, event_id, number_map, source_id, latency_seconds=1):
    if not 0 <= latency_seconds <= 120:
        raise ValueError("Reconstructed arrival latency must be 0–120 seconds")
    starts = [timestamp(r["date"]) for r in bundle["race_control"] if r.get("message") == "SESSION STARTED"]
    if not starts:
        raise Unavailable("Archive lacks an explicit race-start observation")
    origin = min(starts)
    laps = normalize_openf1_laps(bundle, event_id, "race", number_map, source_id)
    messages, revisions = [], {}
    for row in sorted(laps, key=lambda r: (r["available_at"], r["driver_id"], r["lap"])):
        identifier = f"{row['driver_id']}:{row['lap']}"
        revision = revisions.get(identifier, -1) + 1
        revisions[identifier] = revision
        end = timestamp(row["end_at"])
        messages.append(
            {
                "id": identifier,
                "revision": revision,
                "event_id": event_id,
                "kind": "timing",
                "driver_id": row["driver_id"],
                "lap": row["lap"],
                "occurred_at": end.isoformat(),
                "received_at": (
                    timestamp(row["available_at"]) + timedelta(seconds=latency_seconds)
                ).isoformat(),
                "elapsed": (end - origin).total_seconds(),
                "lap_seconds": row["seconds"],
                "compound": row["compound"],
                "tyre_age": row["tyre_age"],
                "pit": row["pit"],
                "pace_valid": row["accurate"] and row["green"] and row["lap"] > 1,
                "status": "green" if row["green"] else "unknown",
            }
        )
    for i, (at, status) in enumerate(track_states(bundle["race_control"])):
        messages.append(
            {
                "id": f"control:{i}",
                "event_id": event_id,
                "kind": "race_control",
                "occurred_at": at.isoformat(),
                "received_at": (at + timedelta(seconds=latency_seconds)).isoformat(),
                "status": status,
            }
        )
    return sorted(messages, key=lambda m: (m["received_at"], m["id"], m.get("revision", 0)))


def evaluate_archive(store, parameters, check=lambda: None, progress=lambda x, m: None):
    event_id, source_id = parameters["event_id"], parameters["source_id"]
    version = parameters.get("data_version") or store.current_version()
    laps = int(parameters["scheduled_laps"])
    if not 1 <= laps <= 200 or not parameters.get("distance_source"):
        raise ValueError("A sourced scheduled race distance is required")
    bundle = store.source(source_id)["payload"]
    number_map = entry_number_map(store.table("entries", version), event_id)
    messages = archived_messages(
        bundle, event_id, number_map, source_id, parameters.get("latency_seconds", 1)
    )
    actual = [
        r for r in store.table("results", version) if r["event_id"] == event_id and r["session"] == "race"
    ]
    results, unavailable = [], []
    for lap in range(3, laps, 3):
        check()
        arrivals = [
            timestamp(m["received_at"])
            for m in messages
            if m["kind"] == "timing" and m["lap"] == lap and m.get("revision", 0) == 0
        ]
        if not arrivals:
            continue
        cutoff = max(arrivals).isoformat()
        state = replay(messages, event_id, cutoff)
        prefix = [m for m in messages if timestamp(m["received_at"]) <= timestamp(cutoff)]
        if state != replay(prefix, event_id, cutoff) or state != replay(
            list(reversed(messages)), event_id, cutoff
        ):
            raise AssertionError("Replay depends on future messages or import order")
        try:
            frame = build_features(
                store,
                ForecastRequest(event_id=event_id, stage="live", as_of_utc=cutoff, data_version=version),
            )
            prediction = baseline(frame)
            samples = sample_remaining(frame, prediction, state, 2000, np.random.default_rng(42), laps)
            repeat = sample_remaining(frame, prediction, state, 2000, np.random.default_rng(42), laps)
            if any(not np.array_equal(samples[k], repeat[k]) for k in samples):
                raise AssertionError("Replay simulation is not deterministic")
            if not np.all(np.sort(samples["order"], axis=1) == np.arange(len(frame["rows"]))):
                raise AssertionError("Replay lost or duplicated an entrant")
            results.append(
                {
                    "cutoff_lap": lap,
                    "as_of_utc": cutoff,
                    "state_hash": state["visible_hash"],
                    "feature_hash": frame["feature_hash"],
                    "source_ids": frame["source_ids"],
                    **score_samples(frame, actual, samples),
                }
            )
        except Unavailable as exc:
            unavailable.append({"cutoff_lap": lap, "as_of_utc": cutoff, "reason": str(exc)})
        progress(lap / laps, f"Replayed cutoff near lap {lap}")
    body = {
        "kind": "replay_reconstruction",
        "evidence_kind": "historical_reconstruction",
        "event_id": event_id,
        "code_version": code_version(),
        "data_version": version,
        "source_id": source_id,
        "source_records": store.verify_sources([source_id]),
        "scheduled_laps": laps,
        "distance_source": parameters["distance_source"],
        "messages": len(messages),
        "messages_hash": digest(messages),
        "evaluated_cutoffs": len(results),
        "unavailable_cutoffs": unavailable,
        "rows": results,
        "deterministic": True,
        "prefix_invariant": True,
        "seed": 42,
        "simulations": 2000,
        "latency_seconds_assumption": parameters.get("latency_seconds", 1),
        "limitations": [
            "Historical arrival times are reconstructed, not captured live.",
            "Cutoffs from one race are correlated and are not counted as independent race samples.",
            "No retirement is inferred from final results or the absence of later laps; missing retirement messages can make forecasts unavailable.",
            "Future pit strategy, penalties, neutralizations and weather are not modeled. These metrics do not establish live forecasting readiness.",
        ],
    }
    identifier = store.add_evidence(body)
    return {**body, "evidence_id": identifier}
