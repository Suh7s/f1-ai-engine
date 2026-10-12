"""Vectorized coherent race and season outcomes with explicit scoring assumptions."""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
from .contracts import (
    Evidence,
    ForecastRequest,
    SeasonRequest,
    Stage,
    Unavailable,
    clean,
    code_version,
    digest,
    now_utc,
    timestamp,
)
from .features import build_features
from .models import predict, load_model
from .rules import Rules, RULE_VERSION, classification, countback_order
from .store import Store


def sample_race(
    scores,
    retirement,
    season: int,
    simulations: int,
    rng,
    *,
    temperature=1.0,
    laps=60,
    beta=(2.0, 2.0),
    sprint=False,
    final_round=False,
    strengths=None,
    disqualification=None,
):
    n = len(scores)
    if n < 2 or simulations < 1 or not np.isfinite(scores).all() or not np.isfinite(retirement).all():
        raise ValueError("Finite inputs and at least two entrants required")
    if not np.all((np.asarray(retirement) >= 0) & (np.asarray(retirement) <= 1)) or temperature <= 0:
        raise ValueError("Invalid probability or temperature")
    rules = Rules(season)
    score = (
        np.broadcast_to(np.asarray(scores) / temperature, (simulations, n)).copy()
        if strengths is None
        else strengths / temperature
    )
    hazard = np.asarray(retirement)
    if sprint:
        hazard = 1 - np.power(1 - hazard, 1 / 3)
    retired = rng.random((simulations, n)) < hazard
    completed = np.full((simulations, n), laps)
    sampled = np.floor(rng.beta(*beta, size=(simulations, n)) * laps).astype(int)
    completed[retired] = sampled[retired]
    order = classification(score, completed, rng)
    dq = (
        np.zeros((simulations, n), dtype=bool)
        if disqualification is None
        else rng.random((simulations, n)) < np.asarray(disqualification)
    )
    order = np.take_along_axis(
        order, np.argsort(np.take_along_axis(dq, order, axis=1), axis=1, kind="stable"), axis=1
    )
    position = np.empty_like(order)
    row = np.arange(simulations)[:, None]
    position[row, order] = np.arange(1, n + 1)
    leader_laps = completed.max(axis=1)
    classified = (completed >= np.floor(0.9 * leader_laps[:, None])) & ~dq
    points = np.zeros((simulations, n))
    schedule = rules.schedule(n, sprint=sprint, final_round=final_round)
    points[row, order] = np.broadcast_to(schedule, (simulations, n))
    for distance in np.unique(leader_laps[leader_laps < laps]):
        affected = np.flatnonzero(leader_laps == distance)
        points[affected[:, None], order[affected]] = rules.schedule(
            n,
            sprint=sprint,
            distance=distance / laps,
            green_laps=min(int(distance), 2),
            final_round=final_round,
        )
    points[~classified] = 0
    if rules.fastest_lap_bonus(sprint=sprint):
        # Fastest lap can belong to a driver outside the top ten, yielding no bonus.
        fastest = np.argmax(score + rng.gumbel(size=(simulations, n)), axis=1)
        valid = (position[np.arange(simulations), fastest] <= 10) & classified[
            np.arange(simulations), fastest
        ]
        valid &= leader_laps >= 2
        if season >= 2022:
            valid &= leader_laps / laps >= 0.5
        points[np.arange(simulations)[valid], fastest[valid]] += 1
    return {
        "order": order,
        "position": position,
        "points": points,
        "retired": retired,
        "disqualified": dq,
        "laps": completed,
        "classified": classified,
    }


def summarize(frame: dict, model: dict, samples: dict, simulations: int) -> dict:
    n = len(frame["rows"])
    positions = samples["position"]
    out = []
    win = (positions == 1).mean(axis=0)
    for i, r in enumerate(frame["rows"]):
        pmf = np.bincount(positions[:, i], minlength=n + 1)[1:] / simulations
        p = float(win[i])
        mcse = float(np.sqrt(p * (1 - p) / simulations))
        contributions = model["explanations"][i]
        factors = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        out.append(
            {
                "driver_id": r["driver_id"],
                "name": r["name"],
                "code": r["code"],
                "team_id": r["team_id"],
                "team_name": r["team_name"],
                "grid": r["grid"],
                "win_probability": p,
                "podium_probability": float((positions[:, i] <= min(3, n)).mean()),
                "points_probability": float((samples["points"][:, i] > 0).mean()),
                "retirement_probability": float(samples["retired"][:, i].mean()),
                "disqualification_probability": float(
                    samples.get("disqualified", np.zeros_like(samples["retired"]))[:, i].mean()
                ),
                "expected_finish": float(positions[:, i].mean()),
                "position_p10": float(np.quantile(positions[:, i], 0.1, method="inverted_cdf")),
                "position_p90": float(np.quantile(positions[:, i], 0.9, method="inverted_cdf")),
                "expected_points": float(samples["points"][:, i].mean()),
                "position_probabilities": pmf.tolist(),
                "win_sampling_se": mcse,
                "top_factors": [
                    {"feature": k, "contribution": v, "kind": "model_estimate"} for k, v in factors
                ],
            }
        )
    head_to_head = {
        r["driver_id"]: {
            other["driver_id"]: float((positions[:, i] < positions[:, j]).mean())
            for j, other in enumerate(frame["rows"])
            if j != i
        }
        for i, r in enumerate(frame["rows"])
    }
    return {
        "drivers": sorted(out, key=lambda r: r["expected_finish"]),
        "head_to_head": head_to_head,
        "uncertainty": {
            "outcome": "Position p10/p90 are predictive outcome quantiles.",
            "sampling": "win_sampling_se is Monte Carlo error, not uncertainty about model correctness.",
            "model": "Not estimated for individual forecasts; use held-out evidence and race-bootstrap comparisons.",
            "calibrated": model["calibrated"],
        },
        "assumptions": model["assumptions"],
        "number_kinds": {
            "probabilities": "model_estimate",
            "expected_finish": "model_estimate",
            "expected_points": "model_estimate",
            "grid": "observation_or_user_assumption",
            "simulations": "derived_calculation",
        },
    }


def forecast(store: Store, request: ForecastRequest, *, persist=True, live_state=None) -> dict:
    version = request.data_version or store.current_version()
    request = request.model_copy(update={"data_version": version})
    if request.stage == Stage.LIVE and live_state is None:
        raise ValueError("Live forecasts require a validated replay/live state")
    frame = build_features(store, request)
    model = predict(frame, store, request.model_version)
    request = request.model_copy(update={"model_version": model["model_version"]})
    if request.scenario:
        model["scores"] = model["scores"] + np.array(
            [request.scenario.strength_delta.get(r["driver_id"], 0) for r in frame["rows"]]
        )
    laps = frame["event"].get("scheduled_laps") or 60
    assumptions = list(model["assumptions"])
    if not frame["event"].get("scheduled_laps"):
        assumptions.append(
            "Scheduled distance missing: retirement timing uses a 60-lap normalized distance assumption."
        )
    rng = np.random.default_rng(request.seed)
    if live_state is None:
        samples = sample_race(
            model["scores"],
            model["retirement"],
            frame["event"]["season"],
            request.simulations,
            rng,
            temperature=model["temperature"],
            laps=laps,
            beta=model["retirement_beta"],
            disqualification=model["disqualification"],
            final_round=frame["event"].get("double_points", False),
        )
    else:
        from .replay import sample_remaining

        samples = sample_remaining(frame, model, live_state, request.simulations, rng, laps)
    result = summarize(frame, model, samples, request.simulations)
    result["assumptions"] = assumptions
    generated = timestamp(now_utc())
    if frame["illustrative"] or request.scenario:
        evidence = Evidence.ILLUSTRATIVE
    elif (
        request.as_of_utc >= timestamp(store.manifest(version)["created_at"])
        and generated <= timestamp(frame["event"]["start_at"])
        and request.stage != Stage.LIVE
    ):
        evidence = Evidence.PROSPECTIVE
    else:
        evidence = Evidence.RECONSTRUCTION
    body = {
        "schema_version": "2.0",
        "kind": "race",
        "event": frame["event"],
        "request": request.model_dump(mode="json"),
        "evidence_kind": evidence.value,
        "model_version": model["model_version"],
        "data_version": version,
        "feature_version": frame["feature_version"],
        "feature_hash": frame["feature_hash"],
        "code_version": code_version(),
        "rules_version": RULE_VERSION,
        "seed": request.seed,
        "simulations": request.simulations,
        "source_ids": frame["source_ids"],
        "availability_bases": frame["availability_bases"],
        "coverage": frame["coverage"],
        "missing_inputs": frame["missing"],
        "features": frame["rows"],
        "historical_comparisons": frame["historical_comparisons"],
        "practice_filter": frame["practice_filter"],
        "generated_at": now_utc(),
        "live_state": live_state,
        **result,
    }
    if live_state:
        if live_state.get("source_id"):
            body["source_ids"] = sorted(set(body["source_ids"]) | {live_state["source_id"]})
        body["assumptions"] += [
            "Live pace prior: 90 seconds minus 0.25 × model strength; weight grows with observed green laps up to 80%. Unvalidated replay model.",
            "Unobserved pit stops, neutralizations and weather changes are not modeled in the live baseline.",
        ]
        result["assumptions"] = body["assumptions"]
    body["prediction_hash"] = digest({k: result[k] for k in ("drivers", "head_to_head", "uncertainty")})
    return store.save_forecast(body) if persist else clean(body)


def reproduce(store: Store, identifier: str) -> dict:
    saved = store.forecast(identifier)
    if saved["code_version"] != code_version():
        raise Unavailable(
            "Source code differs from the saved forecast; restore that code version to reproduce it"
        )
    if saved["kind"] == "championship":
        current = simulate_season(
            store,
            SeasonRequest(**saved["request"]),
            persist=False,
            model_version=saved["model_versions"][0] if saved["model_versions"] else "baseline:rating",
        )
        return {
            "forecast_id": identifier,
            "match": current["prediction_hash"] == saved["prediction_hash"],
            "expected": saved["prediction_hash"],
            "actual": current["prediction_hash"],
        }
    current = forecast(
        store, ForecastRequest(**saved["request"]), persist=False, live_state=saved.get("live_state")
    )
    return {
        "forecast_id": identifier,
        "match": current["prediction_hash"] == saved["prediction_hash"],
        "expected": saved["prediction_hash"],
        "actual": current["prediction_hash"],
    }


def simulate_season(store: Store, request: SeasonRequest, persist=True, model_version=None) -> dict:
    version = request.data_version or store.current_version()
    manifest = store.manifest(version)
    if not manifest["illustrative"] and request.as_of_utc > timestamp(now_utc()):
        raise ValueError("A championship projection cannot claim a future information cutoff")
    request = request.model_copy(update={"data_version": version})
    events = sorted(
        [
            e
            for e in store.table("events", version)
            if e["season"] == request.season and timestamp(e["available_at"]) <= request.as_of_utc
        ],
        key=lambda e: e["start_at"],
    )
    if not events:
        raise Unavailable("No season calendar available at the cutoff")
    results = [
        r
        for r in store.table("results", version)
        if r["event_id"] in {e["event_id"] for e in events}
        and timestamp(r["available_at"]) <= request.as_of_utc
    ]
    completed = {r["event_id"] for r in results if r["session"] == "race"}
    remaining = [
        e for e in events if e["event_id"] not in completed and timestamp(e["start_at"]) >= request.as_of_utc
    ]
    stale = [
        e for e in events if e["event_id"] not in completed and timestamp(e["start_at"]) < request.as_of_utc
    ]
    if stale:
        raise Unavailable(
            "Past races have missing final results; refresh before forecasting the championship"
        )
    standings = [
        r
        for r in store.table("standings", version)
        if r["event_id"] in completed and timestamp(r["available_at"]) <= request.as_of_utc
    ]
    constructors = [
        r
        for r in store.table("constructor_standings", version)
        if r["event_id"] in completed and timestamp(r["available_at"]) <= request.as_of_utc
    ]
    last = max(completed, default=None)
    standings = [r for r in standings if r["event_id"] == last]
    constructors = [r for r in constructors if r["event_id"] == last]
    if completed and (not standings or not constructors):
        raise Unavailable(
            "Official standings at the cutoff are required; driver points are not reassigned to current constructors"
        )
    entries = [
        r
        for r in store.table("entries", version)
        if r["event_id"] in {e["event_id"] for e in events}
        and timestamp(r["available_at"]) <= request.as_of_utc
    ]
    driver_ids = sorted({r["driver_id"] for r in entries} | {r["driver_id"] for r in standings})
    teams = sorted({r["team_id"] for r in entries} | {r["team_id"] for r in constructors})
    if len(driver_ids) < 2 or not teams:
        raise Unavailable("Insufficient championship entries")
    di = {v: i for i, v in enumerate(driver_ids)}
    ti = {v: i for i, v in enumerate(teams)}
    n = request.simulations
    rng = np.random.default_rng(request.seed)
    points = np.zeros((n, len(di)))
    tp = np.zeros((n, len(ti)))
    counts = np.zeros((n, len(di), len(di)), dtype=int)
    tc = np.zeros((n, len(ti), len(di)), dtype=int)
    for row in standings:
        points[:, di[row["driver_id"]]] = row["points"]
    for row in constructors:
        tp[:, ti[row["team_id"]]] = row["points"]
    for row in results:
        if (
            row["session"] == "race"
            and row["driver_id"] in di
            and row["team_id"] in ti
            and 1 <= row["position"] <= len(di)
        ):
            counts[:, di[row["driver_id"]], row["position"] - 1] += 1
            tc[:, ti[row["team_id"]], row["position"] - 1] += 1
    # Standings following the last GP do not yet include a later completed sprint.
    for row in results:
        if row["session"] == "sprint" and row["event_id"] not in completed:
            points[:, di[row["driver_id"]]] += row["points"]
            tp[:, ti[row["team_id"]]] += row["points"]
    start_points = points[0].copy()
    team_shift = rng.normal(0, 0.15, (n, len(ti)))  # shared once per season trajectory, explicit prior
    model_versions = set()
    feature_hashes = {}
    evolution = []
    source_ids = set(r["source_id"] for r in standings + constructors + results)
    future_form = {d: np.zeros(n) for d in driver_ids}
    fixed_model = model_version or (
        load_model(store, "active:pre_weekend")[2] if remaining else "baseline:rating"
    )
    for event in remaining:
        frame = build_features(
            store,
            ForecastRequest(
                event_id=event["event_id"],
                as_of_utc=request.as_of_utc,
                data_version=version,
                stage=Stage.PRE_WEEKEND,
                simulations=n,
                seed=request.seed,
            ),
        )
        feature_hashes[event["event_id"]] = frame["feature_hash"]
        source_ids.update(frame["source_ids"])
        model = predict(frame, store, fixed_model)
        model_versions.add(model["model_version"])
        mapping = np.array([di[r["driver_id"]] for r in frame["rows"]])
        team_map = np.array([ti[r["team_id"]] for r in frame["rows"]])
        strengths = (
            model["scores"][None, :]
            + team_shift[:, team_map]
            + np.column_stack([future_form[r["driver_id"]] for r in frame["rows"]])
        )
        # Future qualifying uncertainty is sampled independently of final-race ordering.
        grid_noise = rng.gumbel(0, 0.2, strengths.shape)
        strengths = strengths + grid_noise
        already_sprint = any(r["event_id"] == event["event_id"] and r["session"] == "sprint" for r in results)
        sessions = ([True] if event.get("sprint") and not already_sprint else []) + [False]
        for sprint in sessions:
            samples = sample_race(
                model["scores"],
                model["retirement"],
                request.season,
                n,
                rng,
                temperature=model["temperature"],
                laps=20 if sprint else event.get("scheduled_laps") or 60,
                beta=model["retirement_beta"],
                disqualification=model["disqualification"],
                sprint=sprint,
                final_round=event.get("double_points", False),
                strengths=strengths,
            )
            points[:, mapping] += samples["points"]
            for j, t in enumerate(team_map):
                tp[:, t] += samples["points"][:, j]
            if not sprint:
                for j, d in enumerate(mapping):
                    np.add.at(counts, (np.arange(n), d, samples["position"][:, j] - 1), 1)
                    np.add.at(tc, (np.arange(n), team_map[j], samples["position"][:, j] - 1), 1)
                    future_form[driver_ids[d]] = 0.9 * future_form[driver_ids[d]] + 0.1 * (
                        0.5 - (samples["position"][:, j] - 1) / max(len(mapping) - 1, 1)
                    )
        leader = countback_order(points, counts, rng)[:, 0]
        evolution.append(
            {
                "event_id": event["event_id"],
                "label": "Probability of leading after this round, not title probability",
                "leader_probabilities": {d: float((leader == i).mean()) for d, i in di.items()},
            }
        )
    order = countback_order(points, counts, rng)
    team_order = countback_order(tp, tc, rng)
    position = np.empty_like(order)
    position[np.arange(n)[:, None], order] = np.arange(1, len(di) + 1)
    rules = Rules(request.season)
    max_remaining = sum(
        rules.schedule(1, final_round=e.get("double_points", False))[0]
        + (1 if rules.fastest_lap_bonus() else 0)
        + (
            rules.schedule(1, sprint=True)[0]
            if e.get("sprint")
            and not any(r["event_id"] == e["event_id"] and r["session"] == "sprint" for r in results)
            else 0
        )
        for e in remaining
    )
    drivers = [
        {
            "driver_id": d,
            "current_points": start_points[i],
            "title_probability": float((order[:, 0] == i).mean()),
            "expected_points": float(points[:, i].mean()),
            "points_p10": float(np.quantile(points[:, i], 0.1)),
            "points_p90": float(np.quantile(points[:, i], 0.9)),
            "expected_position": float(position[:, i].mean()),
            "position_probabilities": (np.bincount(position[:, i], minlength=len(di) + 1)[1:] / n).tolist(),
            "eliminated": bool(start_points[i] + max_remaining < max(start_points)),
        }
        for d, i in di.items()
    ]
    out = {
        "kind": "championship",
        "schema_version": "2.0",
        "season": request.season,
        "request": request.model_dump(mode="json"),
        "data_version": version,
        "code_version": code_version(),
        "rules_version": RULE_VERSION,
        "evidence_kind": Evidence.ILLUSTRATIVE.value
        if manifest["illustrative"]
        else (
            Evidence.PROSPECTIVE.value
            if remaining and request.as_of_utc >= timestamp(manifest["created_at"])
            else Evidence.RECONSTRUCTION.value
        ),
        "seed": request.seed,
        "simulations": n,
        "source_ids": sorted(source_ids),
        "model_versions": sorted(model_versions),
        "feature_hashes": feature_hashes,
        "feature_version": __import__("f1engine.contracts", fromlist=["FEATURE_VERSION"]).FEATURE_VERSION,
        "remaining_events": len(remaining),
        "drivers": sorted(drivers, key=lambda d: -d["title_probability"]),
        "constructors": [
            {
                "team_id": t,
                "title_probability": float((team_order[:, 0] == i).mean()),
                "expected_points": float(tp[:, i].mean()),
                "points_p10": float(np.quantile(tp[:, i], 0.1)),
                "points_p90": float(np.quantile(tp[:, i], 0.9)),
            }
            for t, i in ti.items()
        ],
        "evolution": evolution,
        "coverage": {
            "official_standings": bool(standings),
            "remaining_events": len(remaining),
            "known_driver_entries": len(di),
        },
        "missing_inputs": [
            "Future grids, weather and roster changes are uncertain; current roster and explicit priors are used."
        ],
        "assumptions": [
            "Future entries use the known/provisional roster at the information cutoff.",
            "Team-strength sigma=0.15 is an unvalidated season-persistent prior.",
            "Exact unresolved countback ties use random adjudication.",
            "Elimination uses a conservative maximum-points bound; equality is not declared eliminated.",
            "Full-distance future races are assumed; cancellations and shortened races are not modeled.",
        ],
        "number_kinds": {
            "current_points": "observation",
            "title_probability": "model_estimate",
            "expected_points": "model_estimate",
            "remaining_events": "derived_calculation",
        },
    }
    out["prediction_hash"] = digest(
        {k: out[k] for k in ("drivers", "constructors", "evolution", "assumptions")}
    )
    return store.save_forecast(out) if persist else clean(out)
