"""Arrival-time replay: only messages actually received by the cutoff are visible."""

from datetime import datetime
from typing import Literal
import numpy as np
from pydantic import Field, field_validator
from .contracts import StrictModel, ForecastRequest, Unavailable, digest, timestamp
from .rules import Rules


class Message(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    revision: int = Field(default=0, ge=0)
    event_id: str = Field(pattern=r"^\d{4}-\d{2}$")
    occurred_at: datetime
    received_at: datetime
    kind: Literal["timing"] = "timing"
    driver_id: str
    lap: int = Field(ge=0, le=200)
    elapsed: float = Field(ge=0, allow_inf_nan=False)
    lap_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    compound: str | None = None
    tyre_age: int | None = Field(default=None, ge=0)
    x: float | None = Field(default=None, allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)
    retired: bool = False
    disqualified: bool = False
    status: Literal["green", "yellow", "safety_car", "red", "finished"] = "green"

    @field_validator("occurred_at", "received_at")
    @classmethod
    def aware(cls, value):
        return ForecastRequest.aware_time(value)


def replay(messages: list[dict], event_id: str, as_of_utc: str, stale_after_seconds=30) -> dict:
    cutoff = timestamp(as_of_utc)
    visible = {}
    duplicates = 0
    for raw in messages:
        # Parsing is not forecasting; later records never enter visible state or its hash.
        m = Message(**raw)
        if m.event_id != event_id or m.received_at > cutoff or m.occurred_at > cutoff:
            continue
        if m.received_at < m.occurred_at:
            raise ValueError("Message received before its observation timestamp")
        old = visible.get(m.id)
        if old:
            if old.revision == m.revision:
                if old.model_dump() != m.model_dump():
                    raise ValueError("Conflicting message with identical id and revision")
                duplicates += 1
            if (old.revision, old.received_at) >= (m.revision, m.received_at):
                continue
        visible[m.id] = m
    if not visible:
        raise Unavailable("No timing observations have arrived by this cutoff")
    ordered = sorted(visible.values(), key=lambda m: (m.occurred_at, m.received_at, m.id))
    drivers = {}
    history = {}
    for m in ordered:
        drivers[m.driver_id] = m.model_dump(mode="json")
        if m.lap_seconds and m.status == "green" and not m.retired:
            history.setdefault(m.driver_id, []).append(m.lap_seconds)
    last = max(m.received_at for m in visible.values())
    age = (cutoff - last).total_seconds()
    tower = sorted(drivers.values(), key=lambda r: (-r["lap"], r["elapsed"], r["driver_id"]))
    for i, r in enumerate(tower):
        r["position"] = i + 1
        r["pace_seconds"] = (
            float(np.median(history.get(r["driver_id"], [])[-5:])) if history.get(r["driver_id"]) else None
        )
        r["pace_laps"] = len(history.get(r["driver_id"], []))
        r["gap_seconds"] = r["elapsed"] - tower[0]["elapsed"] if r["lap"] == tower[0]["lap"] else None
    payload = [
        m.model_dump(mode="json")
        for m in sorted(visible.values(), key=lambda m: (m.received_at, m.id, m.revision))
    ]
    return {
        "event_id": event_id,
        "as_of_utc": cutoff.isoformat(),
        "last_received_at": last.isoformat(),
        "age_seconds": age,
        "stale": age > stale_after_seconds,
        "status": ordered[-1].status,
        "drivers": tower,
        "message_count": len(payload),
        "visible_hash": digest(payload),
        "number_kinds": {
            "lap": "observation",
            "elapsed": "observation",
            "gap_seconds": "derived_calculation",
            "pace_seconds": "derived_calculation",
            "age_seconds": "derived_calculation",
        },
        "mode": "replay",
        "position_coordinates": "observed feed coordinates; no track boundary geometry supplied",
        "limitations": [
            "Arrival timestamps are required. No feed is described as live unless a connected recorder supplies fresh observations.",
            "Track coordinates are not inferred from lap timing.",
        ],
    }


def sample_remaining(frame, model, state, simulations, rng, total_laps):
    if state["event_id"] != frame["event"]["event_id"]:
        raise ValueError("Replay event mismatch")
    if state["stale"]:
        raise Unavailable("Timing feed is stale; retain the last valid forecast")
    known = {r["driver_id"]: r for r in state["drivers"]}
    rows = frame["rows"]
    n = len(rows)
    if set(known) != {r["driver_id"] for r in rows}:
        raise Unavailable("Full-field timing is required for a live forecast")
    if state["status"] == "red":
        raise Unavailable("Race suspended: remaining-distance assumptions require confirmation")
    elapsed = np.array([known[r["driver_id"]]["elapsed"] for r in rows])
    laps = np.array([known[r["driver_id"]]["lap"] for r in rows])
    if np.any(laps > total_laps):
        raise ValueError("Observed laps exceed scheduled distance")
    done = np.array([known[r["driver_id"]]["retired"] for r in rows])
    left = total_laps - laps
    prior_pace = 90.0 - np.asarray(model["scores"]) * 0.25
    observed = np.array([known[r["driver_id"]]["pace_seconds"] or prior_pace[i] for i, r in enumerate(rows)])
    weight = np.array([min(known[r["driver_id"]]["pace_laps"] / 15, 0.8) for r in rows])
    pace = weight * observed + (1 - weight) * prior_pace
    hazard = 1 - np.power(1 - np.asarray(model["retirement"]), left / max(total_laps, 1))
    retired = done[None, :] | (rng.random((simulations, n)) < hazard)
    completed = np.broadcast_to(np.full(n, total_laps), (simulations, n)).copy()
    loss = np.floor(rng.beta(*model["retirement_beta"], size=(simulations, n)) * left).astype(int)
    completed[retired] = np.broadcast_to(laps, (simulations, n))[retired] + loss[retired]
    completed[:, done] = laps[done]
    finish = elapsed + left * pace + rng.normal(0, np.sqrt(left) * 0.6, size=(simulations, n))
    order = np.lexsort((finish, -completed), axis=1)
    dq = rng.random((simulations, n)) < model.get("disqualification", np.zeros(n))
    dq |= np.array([known[r["driver_id"]].get("disqualified", False) for r in rows])[None, :]
    order = np.take_along_axis(
        order, np.argsort(np.take_along_axis(dq, order, axis=1), axis=1, kind="stable"), axis=1
    )
    position = np.empty_like(order)
    ix = np.arange(simulations)[:, None]
    position[ix, order] = np.arange(1, n + 1)
    classified = (completed >= np.floor(0.9 * completed.max(axis=1, keepdims=True))) & ~dq
    points = np.zeros((simulations, n))
    rules = Rules(frame["event"]["season"])
    points[ix, order] = rules.schedule(n, final_round=frame["event"].get("double_points", False))
    points[~classified] = 0
    if rules.fastest_lap_bonus():
        fastest = np.argmin(pace + rng.normal(0, 0.4, size=(simulations, n)), axis=1)
        ii = np.arange(simulations)
        valid = (position[ii, fastest] <= 10) & classified[ii, fastest]
        points[ii[valid], fastest[valid]] += 1
    return {
        "order": order,
        "position": position,
        "points": points,
        "retired": retired,
        "disqualified": dq,
        "laps": completed,
        "classified": classified,
    }
