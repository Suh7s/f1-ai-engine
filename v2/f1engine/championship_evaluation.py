"""Historical season projections; correlated cutoffs never masquerade as independent seasons."""

from datetime import timedelta
import numpy as np
from .contracts import SeasonRequest, Unavailable, timestamp, code_version
from .simulation import simulate_season
from .models import BASELINES


def championship_metrics(predicted, actual, identity, simulations):
    p = {r[identity]: r for r in predicted}
    a = {r[identity]: r for r in actual}
    if not a or sorted(r["position"] for r in a.values()) != list(range(1, len(a) + 1)):
        raise Unavailable("Complete official championship classification is required for scoring")
    winner = next(k for k, v in a.items() if v["position"] == 1)
    probability = p.get(winner, {}).get("title_probability", 0)
    ids = sorted(set(p) & set(a))
    rps = None
    if set(p) == set(a):
        n = len(ids)
        cdf = np.cumsum([p[k]["position_probabilities"] for k in ids], axis=1)[:, :-1]
        observed = np.array([a[k]["position"] for k in ids])[:, None] <= np.arange(1, n)
        rps = float(np.mean((cdf - observed) ** 2))
    return {
        "full_field_rps": rps,
        "title_log_loss": float(-np.log((probability * simulations + 0.5) / (simulations + 0.5 * len(a)))),
        "title_brier": float(
            np.mean(
                [(p.get(k, {}).get("title_probability", 0) - int(k == winner)) ** 2 for k in set(a) | set(p)]
            )
        ),
        "points_mae_known_entries": float(
            np.mean([abs(p[k]["expected_points"] - a[k]["points"]) for k in ids])
        ),
        "points_interval_coverage_known_entries": float(
            np.mean([p[k]["points_p10"] <= a[k]["points"] <= p[k]["points_p90"] for k in ids])
        ),
        "known_entries_scored": len(ids),
        "unforeseen_entries": sorted(set(a) - set(p)),
        "forecast_only_entries": sorted(set(p) - set(a)),
        "winner_log_loss_estimator": "Jeffreys finite-Monte-Carlo count correction; saved probabilities are unchanged",
    }


def evaluate_championship(store, parameters, check=lambda: None, progress=lambda x, m: None):
    version = parameters.get("data_version") or store.current_version()
    season = int(parameters["season"])
    events = sorted(
        [e for e in store.table("events", version) if e["season"] == season], key=lambda e: e["start_at"]
    )
    if not events:
        raise Unavailable("No season calendar")
    standings = store.table("standings", version)
    constructors = store.table("constructor_standings", version)
    final_event = events[-1]["event_id"]
    actual = [r for r in standings if r["event_id"] == final_event]
    actual_teams = [r for r in constructors if r["event_id"] == final_event]
    snapshots = sorted(
        {
            r["event_id"]
            for r in standings
            if r["event_id"].startswith(f"{season}-") and r["event_id"] != final_event
        }
    )
    requested = parameters.get("events", snapshots)
    if not requested or set(requested) - set(snapshots):
        raise Unavailable("Official intermediate standings snapshots must be acquired before evaluation")
    rows = []
    for index, event_id in enumerate(requested):
        check()
        records = [r for r in standings + constructors if r["event_id"] == event_id]
        cutoff = max(timestamp(r["available_at"]) for r in records) + timedelta(seconds=1)
        for model in BASELINES:
            check()
            projection = simulate_season(
                store,
                SeasonRequest(season=season, as_of_utc=cutoff, data_version=version, simulations=10000),
                persist=False,
                model_version=model,
            )
            rows.append(
                {
                    "after_event": event_id,
                    "as_of_utc": cutoff.isoformat(),
                    "model_version": model,
                    "prediction_hash": projection["prediction_hash"],
                    "source_ids": projection["source_ids"],
                    "feature_hashes": projection["feature_hashes"],
                    "remaining_events": projection["remaining_events"],
                    "drivers": championship_metrics(projection["drivers"], actual, "driver_id", 10000),
                    "constructors": championship_metrics(
                        projection["constructors"], actual_teams, "team_id", 10000
                    ),
                }
            )
        progress((index + 1) / len(requested), f"Scored projections after {event_id}")
    body = {
        "kind": "championship_reconstruction",
        "evidence_kind": "historical_reconstruction",
        "season": season,
        "independent_seasons": 1,
        "data_version": version,
        "code_version": code_version(),
        "rows": rows,
        "seed": 42,
        "simulations": 10000,
        "actual_source_ids": sorted({r["source_id"] for r in actual + actual_teams}),
        "limitations": [
            "Publication times are reconstructed; final outcomes are used only after forecasting for scoring.",
            "Multiple cutoffs in one season are correlated; no pooled accuracy or promotion claim.",
            "Driver full-field RPS is unavailable if future substitutions introduced entrants unknown at the cutoff.",
            "Persistent team strength, qualifying noise and simulated-form evolution remain provisional assumptions; this evaluation does not itself calibrate them.",
        ],
    }
    return {**body, "evidence_id": store.add_evidence(body)}
