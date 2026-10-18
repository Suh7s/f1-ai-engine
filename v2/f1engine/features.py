"""One point-in-time feature builder for inference and chronological evaluation."""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import math
import numpy as np
from .contracts import FEATURE_VERSION, ForecastRequest, Stage, Unavailable, digest, timestamp
from .store import Store

FEATURES = [
    "grid_fraction",
    "driver_strength",
    "team_strength",
    "recent_finish",
    "recent_quali",
    "circuit_finish",
    "retirement_rate",
    "team_retirement_rate",
    "practice_relative_pace",
    "qualifying_gap",
    "sprint_fraction",
    "rain_probability",
    "experience",
]
FEATURE_GROUPS = {
    "telemetry": ["practice_relative_pace"],
    "circuit": ["circuit_finish"],
    "weekend": ["practice_relative_pace", "qualifying_gap", "sprint_fraction", "grid_fraction"],
}


def lap_seconds(value):
    if value is None:
        return None
    try:
        parts = str(value).split(":")
        return float(parts[-1]) + (60 * float(parts[-2]) if len(parts) > 1 else 0)
    except (ValueError, TypeError):
        return None


def _mean(values):
    return float(np.mean(values)) if values else None


def build_features(store: Store, request: ForecastRequest) -> dict:
    version = request.data_version or store.current_version()
    manifest = store.manifest(version)
    cutoff = request.as_of_utc
    events = {
        r["event_id"]: r
        for r in store.table("events", version, as_of=cutoff)
        if timestamp(r["available_at"]) <= cutoff
    }
    event = events.get(request.event_id)
    if not event:
        raise Unavailable("Event calendar is unavailable at this cutoff")
    target_start = timestamp(event["start_at"])
    if cutoff > target_start and request.stage != Stage.LIVE:
        raise ValueError("Pre-race forecast cutoff cannot be after the scheduled race start")
    if cutoff > datetime.now(timezone.utc) and not manifest["illustrative"]:
        raise ValueError("A forecast cannot claim information from a future cutoff")
    entries = sorted(
        [
            r
            for r in store.table("entries", version, as_of=cutoff)
            if r["event_id"] == request.event_id and timestamp(r["available_at"]) <= cutoff
        ],
        key=lambda r: r["driver_id"],
    )
    if not entries:
        known_entries = [
            r
            for r in store.table("entries", version, as_of=cutoff)
            if r["event_id"] in events
            and timestamp(r["available_at"]) <= cutoff
            and timestamp(events[r["event_id"]]["start_at"]) <= cutoff
        ]
        if known_entries:
            last_event = max({r["event_id"] for r in known_entries}, key=lambda eid: events[eid]["start_at"])
            entries = sorted(
                [
                    {**r, "event_id": request.event_id, "availability_basis": "provisional_previous_roster"}
                    for r in known_entries
                    if r["event_id"] == last_event
                ],
                key=lambda r: r["driver_id"],
            )
    if len(entries) < 2:
        raise Unavailable("At least two identified entries are required at this cutoff")
    if not manifest["illustrative"] and len(entries) < 15:
        raise Unavailable("Entry coverage is incomplete for a full-field F1 forecast")
    if len({e["driver_id"] for e in entries}) != len(entries):
        raise ValueError("Duplicate driver entry")
    rows = store.table("results", version, as_of=cutoff)
    known = [r for r in rows if timestamp(r["available_at"]) <= cutoff]
    history = [
        r
        for r in known
        if r["session"] == "race"
        and r["event_id"] in events
        and timestamp(events[r["event_id"]]["start_at"]) < target_start
        and r["event_id"] != request.event_id
    ]
    history.sort(key=lambda r: (events[r["event_id"]]["start_at"], r["position"], r["driver_id"]))
    races = defaultdict(list)
    for row in history:
        races[row["event_id"]].append(row)
    driver_strength = defaultdict(float)
    team_strength = defaultdict(float)
    by_driver = defaultdict(list)
    by_team = defaultdict(list)
    for event_id, race in races.items():
        size = len(race)
        team_scores = defaultdict(list)
        for result in race:
            driver, team = result["driver_id"], result["team_id"]
            normalized = (size - result["position"]) / max(size - 1, 1)
            if not result.get("retired", False) and not result.get("disqualified", False):
                performance = 2 * normalized - 1
                driver_strength[driver] = 0.88 * driver_strength[driver] + 0.12 * (
                    performance - 0.35 * team_strength[team]
                )
                team_scores[team].append(performance - 0.35 * driver_strength[driver])
            record = {
                **result,
                "finish_fraction": (result["position"] - 1) / max(size - 1, 1),
                "grid_fraction": (result.get("grid", 0) - 1) / max(size - 1, 1)
                if result.get("grid", 0) > 0
                else None,
                "circuit_id": events[event_id]["circuit_id"],
            }
            by_driver[driver].append(record)
            by_team[team].append(record)
        for team, scores in team_scores.items():
            team_strength[team] = 0.85 * team_strength[team] + 0.15 * float(np.mean(scores))
    allowed = set()
    if request.stage != Stage.PRE_WEEKEND:
        allowed |= {"practice1", "practice2", "practice3"}
    if request.stage in {Stage.POST_QUALIFYING, Stage.POST_SPRINT, Stage.LIVE}:
        allowed |= {"qualifying", "sprint"}
    current = [r for r in known if r["event_id"] == request.event_id and r["session"] in allowed]
    qualifying = {r["driver_id"]: r for r in current if r["session"] == "qualifying"}
    sprint = {r["driver_id"]: r for r in current if r["session"] == "sprint"}
    if request.stage == Stage.POST_QUALIFYING and not qualifying:
        raise Unavailable("Qualifying has not been published by this cutoff")
    if request.stage == Stage.POST_SPRINT and not sprint:
        raise Unavailable("Sprint results have not been published by this cutoff")
    qualifying_times = {
        d: next(
            (lap_seconds(r.get(k)) for k in ("q3", "q2", "q1") if lap_seconds(r.get(k)) is not None), None
        )
        for d, r in qualifying.items()
    }
    pole = min([t for t in qualifying_times.values() if t is not None], default=None)
    laps = [
        r
        for r in store.table("laps", version, as_of=cutoff)
        if r["event_id"] == request.event_id
        and r["session"] in allowed
        and r["session"].startswith("practice")
        and timestamp(r["available_at"]) <= cutoff
    ]
    usable = [
        r
        for r in laps
        if r.get("green")
        and not r.get("pit")
        and r.get("accurate")
        and r.get("seconds") is not None
        and math.isfinite(r["seconds"])
        and r["seconds"] > 0
        and r.get("tyre_age") is not None
        and math.isfinite(r["tyre_age"])
        and 2 <= r["tyre_age"] <= 25
        and r.get("compound") not in {None, "", "UNKNOWN"}
    ]
    # Compare like session/compound/age bins. Relative pace avoids mixing different track evolution.
    bins = defaultdict(list)
    for lap in usable:
        bins[(lap["session"], lap.get("compound"), int(lap["tyre_age"]) // 5)].append(lap)
    practice = defaultdict(list)
    for group in bins.values():
        if len({r["driver_id"] for r in group}) < 2:
            continue
        median = float(np.median([r["seconds"] for r in group]))
        for lap in group:
            if 0.95 * median <= lap["seconds"] <= 1.07 * median:
                practice[lap["driver_id"]].append(lap["seconds"] / median - 1)
    weather = [
        r
        for r in store.table("weather", version, as_of=cutoff)
        if r["event_id"] == request.event_id
        and timestamp(r["issued_at"]) <= cutoff
        and timestamp(r["available_at"]) <= cutoff
        and r["availability_basis"] in {"observed_issue_time", "observed_capture_time"}
        and timestamp(store.source(r["source_id"])["retrieved_at"]) <= cutoff
    ]
    weather.sort(key=lambda r: r["issued_at"])
    selected_weather = weather[-1] if weather else None
    # No measured race weather is used. Absence remains explicit, not a fictitious dry forecast.
    climate = [
        r
        for r in store.table("climate", version, as_of=cutoff)
        if r["event_id"] == request.event_id
        and timestamp(r["available_at"]) <= cutoff
        and r["history_end"] < str(cutoff.date())
    ]
    climate.sort(key=lambda r: r["history_end"])
    selected_climate = climate[-1] if climate else None
    rain = (
        selected_weather.get("rain_probability")
        if selected_weather
        else selected_climate["rain_probability"]
        if selected_climate
        else None
    )
    missing = []
    if not weather:
        missing.append(
            "No archived issued weather forecast: using a prior-month climate estimate, not a race-hour forecast."
            if selected_climate
            else "No issued weather forecast or climate prior: weather remains missing. Capture climate history to supply a prior; training-only imputation is explicit."
        )
    if not history:
        missing.append("No prior race history: population priors only")
    if any(e["availability_basis"].startswith("provisional") for e in entries):
        missing.append(
            "Entry list is provisional from the previous race; transfers and substitutions are not confirmed"
        )
    if qualifying:
        missing.append(
            "Starting-grid penalties are not verified; qualifying rank is a grid proxy, not an official starting grid."
        )
    scenario = request.scenario
    driver_ids = {e["driver_id"] for e in entries}
    if scenario:
        overrides = set(scenario.grid) | set(scenario.teams) | set(scenario.strength_delta)
        if overrides - driver_ids:
            raise ValueError("Scenario contains an unknown entrant")
        if scenario.rain_probability is not None:
            rain = scenario.rain_probability
    population_disqualification = (sum(r.get("disqualified", False) for r in history) + 1) / (
        len(history) + 500
    )
    feature_rows = []
    for entry in entries:
        driver = entry["driver_id"]
        team = scenario.teams.get(driver, entry["team_id"]) if scenario else entry["team_id"]
        past = by_driver[driver]
        team_past = by_team[team]
        grid = qualifying.get(driver, {}).get("position")
        if scenario and driver in scenario.grid:
            grid = scenario.grid[driver]
        field = len(entries)
        if grid and grid > field:
            raise ValueError("Grid position exceeds the field size")
        recent_grids = [r["grid_fraction"] for r in past[-5:] if r["grid_fraction"] is not None]
        circuit_past = [r["finish_fraction"] for r in past if r["circuit_id"] == event["circuit_id"]]
        row = {
            "driver_id": driver,
            "name": entry.get("name", driver),
            "code": entry.get("code", driver[:3].upper()),
            "team_id": team,
            "team_name": entry.get("team_name", team) if team == entry["team_id"] else team,
            "grid": grid,
            "grid_basis": "user_assumption"
            if scenario and driver in scenario.grid
            else "qualifying_rank_proxy"
            if grid
            else "unavailable",
            "grid_fraction": (grid - 1) / max(field - 1, 1) if grid else None,
            "driver_strength": driver_strength[driver],
            "team_strength": team_strength[team],
            "recent_finish": _mean([r["finish_fraction"] for r in past[-5:]]),
            "recent_quali": _mean(recent_grids),
            "circuit_finish": _mean(circuit_past[-5:]),
            "disqualification_rate": (
                sum(r.get("disqualified", False) for r in past) + 100 * population_disqualification
            )
            / (len(past) + 100),
            "retirement_rate": (sum(r.get("retired", False) for r in past[-20:]) + 1)
            / (min(len(past), 20) + 10),
            "team_retirement_rate": (sum(r.get("retired", False) for r in team_past[-40:]) + 2)
            / (min(len(team_past), 40) + 20),
            "practice_relative_pace": float(np.median(practice[driver]))
            if len(practice[driver]) >= 3
            else None,
            "qualifying_gap": (qualifying_times[driver] / pole - 1)
            if qualifying_times.get(driver) and pole
            else None,
            "sprint_fraction": (sprint[driver]["position"] - 1) / max(field - 1, 1)
            if driver in sprint
            else None,
            "rain_probability": rain,
            "weather_basis": "user_assumption"
            if scenario and scenario.rain_probability is not None
            else "captured_forecast"
            if selected_weather
            else "climate_prior"
            if selected_climate
            else "missing",
            "experience": len(past),
            "history_races": len(past),
            "sprint_points": sprint.get(driver, {}).get("points"),
        }
        feature_rows.append(row)
    grids = [r["grid"] for r in feature_rows if r["grid"] is not None]
    if len(grids) != len(set(grids)):
        raise ValueError("Combined scenario and observed grid must have unique positions")
    used = (
        history
        + entries
        + current
        + usable
        + ([selected_weather] if selected_weather else [selected_climate] if selected_climate else [])
        + [event]
    )
    sources = sorted({r["source_id"] for r in used})
    source_records = store.verify_sources(sources)
    return {
        "event": event,
        "as_of_utc": cutoff.isoformat(),
        "stage": request.stage.value if hasattr(request.stage, "value") else request.stage,
        "rows": feature_rows,
        "feature_version": FEATURE_VERSION,
        "feature_hash": digest(feature_rows),
        "data_version": version,
        "source_ids": sources,
        "source_records": source_records,
        "missing": missing,
        "illustrative": manifest["illustrative"],
        "coverage": {
            name: sum(r[name] is not None for r in feature_rows) / len(feature_rows) for name in FEATURES
        },
        "provenance": {
            name: (
                "user_assumption"
                if name == "rain_probability" and scenario and scenario.rain_probability is not None
                else "derived_calculation"
            )
            for name in FEATURES
        },
        "availability_bases": sorted({r["availability_basis"] for r in used}),
        "practice_filter": {
            "received_laps": len(laps),
            "eligible_laps": len(usable),
            "fuel_load": "unobserved",
        },
        "historical_comparisons": {
            d: [
                {"event_id": r["event_id"], "position": r["position"], "retired": r.get("retired", False)}
                for r in by_driver[d][-5:]
            ]
            for d in driver_ids
        },
    }


def matrix(frame: dict, columns: list[str] | None = None) -> np.ndarray:
    def value(row, column):
        if column.startswith("driver:"):
            return float(row["driver_id"] == column.split(":", 1)[1])
        if column.startswith("team:"):
            return float(row["team_id"] == column.split(":", 1)[1])
        return row.get(column) if row.get(column) is not None else np.nan

    return np.array([[value(r, c) for c in (columns or FEATURES)] for r in frame["rows"]], dtype=float)
