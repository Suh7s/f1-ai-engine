"""Deterministic fictional fixtures. Never scored or advertised as real F1 evidence."""

from datetime import datetime, timedelta, timezone
import numpy as np
from .contracts import now_utc
from .rules import Rules
from .store import Store

DEMO_CUTOFF = "2026-04-05T12:00:00+00:00"
DEMO_EVENT = "2026-03"


def load_demo(store: Store, drivers=20, rounds=8):
    if store.current_version():
        if not store.manifest()["illustrative"]:
            raise ValueError("Use a separate storage directory for the demo")
        return store.current_version()
    rng = np.random.default_rng(1967)
    tables = {
        k: []
        for k in (
            "events",
            "entries",
            "sessions",
            "results",
            "laps",
            "standings",
            "constructor_standings",
            "weather",
        )
    }
    raw = store.raw(
        "illustrative_fixture",
        "fixture://fictional-championship-v2",
        {"seed": 1967, "drivers": drivers, "rounds": rounds},
        "2026-01-01T00:00:00+00:00",
    )

    def row(at, **kw):
        return {
            **kw,
            "source_id": raw,
            "available_at": at.isoformat(),
            "availability_basis": "illustrative_fixture",
        }

    for year in range(2020, 2027):
        dp = np.zeros(drivers)
        tp = np.zeros((drivers + 1) // 2)
        calendar = datetime(year - 1, 12, 1, tzinfo=timezone.utc)
        for rd in range(1, rounds + 1):
            eid = f"{year}-{rd:02d}"
            start = datetime(year, 3, 8, 14, tzinfo=timezone.utc) + timedelta(days=14 * (rd - 1))
            sprint = year >= 2021 and rd % 3 == 0
            tables["events"].append(
                row(
                    calendar,
                    event_id=eid,
                    season=year,
                    round=rd,
                    name=f"Illustrative Grand Prix {rd}",
                    circuit_id=f"fictional_{rd % 4}",
                    circuit_name=f"Circuit {rd % 4 + 1}",
                    start_at=start.isoformat(),
                    sprint=sprint,
                    scheduled_laps=60,
                    latitude=25.5,
                    longitude=51.4,
                )
            )
            for i in range(drivers):
                team = i // 2
                tables["entries"].append(
                    row(
                        calendar,
                        event_id=eid,
                        driver_id=f"driver_{i:02d}",
                        name=f"Pilot {i + 1:02d}",
                        code=f"P{i + 1:02d}",
                        team_id=f"team_{team:02d}",
                        team_name=f"Constructor {team + 1:02d}",
                        number=str(i + 1),
                    )
                )
            for session, delta in (
                [("practice1", -2), ("qualifying", -1)] + ([("sprint", -1)] if sprint else []) + [("race", 0)]
            ):
                at = start + timedelta(days=delta, hours=-4 if session == "sprint" else 0)
                tables["sessions"].append(
                    row(
                        calendar,
                        event_id=eid,
                        session=session,
                        start_at=at.isoformat(),
                        end_at=(at + timedelta(hours=2)).isoformat(),
                    )
                )
                if session == "practice1":
                    for i in range(drivers):
                        for lap in range(1, 11):
                            tables["laps"].append(
                                row(
                                    at + timedelta(minutes=lap * 2),
                                    event_id=eid,
                                    session=session,
                                    driver_id=f"driver_{i:02d}",
                                    lap=lap,
                                    seconds=90 + i * 0.06 + rng.normal(0, 0.15),
                                    green=True,
                                    pit=lap == 1,
                                    accurate=True,
                                    compound="MEDIUM",
                                    tyre_age=lap,
                                )
                            )
                    continue
                # Future race outcomes deliberately absent from the current demo season.
                if year == 2026 and rd >= 3 and session == "race":
                    continue
                if year == 2026 and rd > 3 and session != "practice1":
                    continue
                order = np.argsort(np.arange(drivers) * 0.1 + rng.normal(0, 1, drivers))
                pos = np.empty(drivers, int)
                pos[order] = np.arange(1, drivers + 1)
                hazard = rng.random(drivers) < 0.1 if session != "qualifying" else np.zeros(drivers, bool)
                if session != "qualifying":
                    order = np.lexsort((pos, hazard))
                    pos[order] = np.arange(1, drivers + 1)
                scores = (
                    Rules(year).schedule(drivers, sprint=session == "sprint")
                    if session != "qualifying"
                    else np.zeros(drivers)
                )
                for i in range(drivers):
                    points = float(scores[pos[i] - 1])
                    completed = int(rng.integers(5, 50)) if hazard[i] else 60
                    tables["results"].append(
                        row(
                            at + timedelta(hours=3),
                            event_id=eid,
                            session=session,
                            driver_id=f"driver_{i:02d}",
                            team_id=f"team_{i // 2:02d}",
                            position=int(pos[i]),
                            grid=int(pos[i]),
                            points=points,
                            laps=completed,
                            retired=bool(hazard[i]),
                            status="Engine" if hazard[i] else "Finished",
                            q1=f"1:{30 + i * 0.06:06.3f}",
                            q2=None,
                            q3=None,
                        )
                    )
                    dp[i] += points
                    tp[i // 2] += points
                if session == "race":
                    for i in range(drivers):
                        tables["standings"].append(
                            row(
                                at + timedelta(hours=4),
                                event_id=eid,
                                driver_id=f"driver_{i:02d}",
                                team_id=f"team_{i // 2:02d}",
                                points=float(dp[i]),
                            )
                        )
                    for i in range(len(tp)):
                        tables["constructor_standings"].append(
                            row(
                                at + timedelta(hours=4),
                                event_id=eid,
                                team_id=f"team_{i:02d}",
                                points=float(tp[i]),
                            )
                        )
    return store.publish(tables, illustrative=True)


def replay_fixture(event_id=DEMO_EVENT, drivers=20):
    start = datetime(2026, 4, 5, 14, tzinfo=timezone.utc)
    messages = []
    for lap in range(1, 11):
        for i in range(drivers):
            at = start + timedelta(seconds=lap * 90 + i * 0.6)
            messages.append(
                {
                    "id": f"{lap}-{i}",
                    "revision": 0,
                    "event_id": event_id,
                    "occurred_at": at.isoformat(),
                    "received_at": (at + timedelta(seconds=1)).isoformat(),
                    "kind": "timing",
                    "driver_id": f"driver_{i:02d}",
                    "lap": lap,
                    "elapsed": lap * 90 + i * 0.6,
                    "lap_seconds": 90 + i * 0.02,
                    "compound": "MEDIUM",
                    "tyre_age": lap,
                    "retired": False,
                    "status": "green",
                }
            )
    return messages
