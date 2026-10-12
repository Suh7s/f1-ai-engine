"""Small, bounded public-source adapters. No imports or constructors fetch data."""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import time
from typing import Callable
import httpx
from .contracts import Unavailable, now_utc, timestamp
from .store import Store

MECHANICAL = (
    "engine",
    "gearbox",
    "electrical",
    "hydraulic",
    "transmission",
    "suspension",
    "brake",
    "power",
    "oil",
    "water",
    "fuel",
    "clutch",
    "mechanical",
)


class PublicClient:
    def __init__(self, store: Store, check: Callable = lambda: None):
        self.store, self.check = store, check
        self.last = {}
        self.client = httpx.Client(
            timeout=30, follow_redirects=True, headers={"User-Agent": "F1Engine/2.0 personal-research"}
        )

    def get(self, url: str, params: dict | None = None, source: str = "jolpica") -> tuple[dict | list, str]:
        # Conservative rates accommodate both per-second and rolling quotas.
        interval = {"jolpica": 7.3, "openf1": 2.1, "open-meteo": 1.1}.get(source, 2.1)
        for attempt in range(4):
            self.check()
            wait = self.last.get(source, 0) + interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self.last[source] = time.monotonic()
            try:
                response = self.client.get(url, params=params)
                if response.status_code in (429, 500, 502, 503, 504):
                    try:
                        retry = float(response.headers.get("Retry-After", 2**attempt))
                    except ValueError:
                        retry = 60
                    self._sleep(min(max(retry, 2**attempt), 120))
                    continue
                response.raise_for_status()
                payload = response.json()
                identifier = self.store.raw(source, str(response.url), payload)
                return payload, identifier
            except (httpx.TransportError, ValueError):
                if attempt == 3:
                    raise
                self._sleep(2**attempt)
        raise Unavailable(f"Source temporarily unavailable: {source}; previous data retained")

    def _sleep(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.check()
            time.sleep(min(0.25, max(deadline - time.monotonic(), 0)))

    def close(self):
        self.client.close()


def outcome_flags(status: str) -> dict:
    normalized = status.strip().casefold()
    disqualified = normalized in {"disqualified", "excluded"}
    return {
        "retired": not (normalized in {"finished", "lapped"} or normalized.startswith("+") or disqualified),
        "disqualified": disqualified,
        "did_not_start": normalized in {"did not start", "did not qualify", "did not prequalify"},
    }


def provenance(source_id: str, available_at: str, basis="reconstructed_session_end") -> dict:
    return {"source_id": source_id, "available_at": available_at, "availability_basis": basis}


def session_time(race: dict, kind: str) -> str:
    if kind == "race":
        day = race.get("date")
        clock = race.get("time", "15:00:00Z")
    else:
        node = race.get({"qualifying": "Qualifying", "sprint": "Sprint"}.get(kind, kind), {})
        if not node.get("date"):
            return (timestamp(session_time(race, "race")) - timedelta(days=1)).isoformat()
        day, clock = node["date"], node.get("time", "15:00:00Z")
    return f"{day}T{clock}"


def _race_payloads(client: PublicClient, suffix: str):
    offset = 0
    while True:
        payload, source_id = client.get(
            f"https://api.jolpi.ca/ergast/f1/{suffix}.json", {"limit": 100, "offset": offset}
        )
        mr = payload["MRData"]
        table = mr.get("RaceTable", mr.get("StandingsTable", {}))
        yield table, source_id
        limit = int(mr.get("limit", 100))
        offset += limit
        if offset >= int(mr.get("total", 0)):
            break


def sync_season(store: Store, season: int, check=lambda: None, progress=lambda value, message: None) -> str:
    if not 2014 <= season <= datetime.now(timezone.utc).year + 1:
        raise ValueError("Season is outside supported ingestion range")
    changes = {
        name: []
        for name in ("events", "entries", "results", "sessions", "standings", "constructor_standings")
    }
    client = PublicClient(store, check)
    calendar = {}
    try:
        for table, source in _race_payloads(client, str(season)):
            for race in table.get("Races", []):
                event = f"{season}-{int(race['round']):02d}"
                calendar[event] = race
                circuit = race.get("Circuit", {})
                loc = circuit.get("Location", {})
                changes["events"].append(
                    {
                        "event_id": event,
                        "season": season,
                        "round": int(race["round"]),
                        "name": race["raceName"],
                        "circuit_id": circuit["circuitId"],
                        "country": loc.get("country"),
                        "latitude": float(loc.get("lat", 0)),
                        "longitude": float(loc.get("long", 0)),
                        "start_at": session_time(race, "race"),
                        "sprint": "Sprint" in race,
                        "scheduled_laps": None,
                        **provenance(source, f"{season - 1}-12-01T00:00:00+00:00", "reconstructed_calendar"),
                    }
                )
                for kind in ("race", "qualifying", "sprint"):
                    if kind == "sprint" and "Sprint" not in race:
                        continue
                    start = session_time(race, kind)
                    changes["sessions"].append(
                        {
                            "event_id": event,
                            "session": kind,
                            "start_at": start,
                            "end_at": (timestamp(start) + timedelta(hours=3)).isoformat(),
                            **provenance(
                                source, f"{season - 1}-12-01T00:00:00+00:00", "reconstructed_calendar"
                            ),
                        }
                    )
        if season == 2014 and changes["events"]:
            max(changes["events"], key=lambda e: e["round"])["double_points"] = True
        for i, (suffix, field, kind) in enumerate(
            (
                ("results", "Results", "race"),
                ("qualifying", "QualifyingResults", "qualifying"),
                ("sprint", "SprintResults", "sprint"),
            )
        ):
            progress(0.1 + i * 0.18, f"Fetching {season} {kind}")
            for table, source in _race_payloads(client, f"{season}/{suffix}"):
                for race in table.get("Races", []):
                    event = f"{season}-{int(race['round']):02d}"
                    event_info = calendar.get(event, race)
                    available = (timestamp(session_time(event_info, kind)) + timedelta(hours=3)).isoformat()
                    for result in race.get(field, []):
                        d = result["Driver"]
                        t = result["Constructor"]
                        status = result.get("status", "Finished")
                        flags = (
                            outcome_flags(status)
                            if kind != "qualifying"
                            else {"retired": False, "disqualified": False, "did_not_start": False}
                        )
                        retired = flags["retired"]
                        changes["results"].append(
                            {
                                "event_id": event,
                                "session": kind,
                                "driver_id": d["driverId"],
                                "team_id": t["constructorId"],
                                "position": int(result.get("position", result.get("positionText", 99)) or 99),
                                "grid": int(result.get("grid", result.get("position", 0)) or 0),
                                "points": float(result.get("points", 0)),
                                "laps": int(result.get("laps", 0)),
                                "status": status,
                                **flags,
                                "mechanical": retired and any(s in status.lower() for s in MECHANICAL),
                                "fastest_lap_rank": int(result.get("FastestLap", {}).get("rank", 0)),
                                "q1": result.get("Q1"),
                                "q2": result.get("Q2"),
                                "q3": result.get("Q3"),
                                **provenance(source, available),
                            }
                        )
                        # Result-derived entry lists are explicitly historical reconstructions.
                        changes["entries"].append(
                            {
                                "event_id": event,
                                "driver_id": d["driverId"],
                                "name": f"{d['givenName']} {d['familyName']}",
                                "code": d.get("code", d["familyName"][:3].upper()),
                                "number": d.get("permanentNumber"),
                                "team_id": t["constructorId"],
                                "team_name": t["name"],
                                **provenance(
                                    source,
                                    (
                                        timestamp(session_time(event_info, "race")) - timedelta(days=7)
                                    ).isoformat(),
                                    "reconstructed_entry",
                                ),
                            }
                        )
        # Latest official standings include sprint points and penalties; never infer team points from current seats.
        for suffix, group, target in (
            ("driverStandings", "DriverStandings", "standings"),
            ("constructorStandings", "ConstructorStandings", "constructor_standings"),
        ):
            for table, source in _race_payloads(client, f"{season}/{suffix}"):
                for snapshot in table.get("StandingsLists", []):
                    event = f"{season}-{int(snapshot['round']):02d}"
                    if event not in calendar:
                        continue
                    available = (
                        timestamp(session_time(calendar[event], "race")) + timedelta(hours=4)
                    ).isoformat()
                    for row in snapshot.get(group, []):
                        identity = (
                            {
                                "driver_id": row["Driver"]["driverId"],
                                "team_id": row.get("Constructors", [{}])[-1].get("constructorId"),
                            }
                            if target == "standings"
                            else {"team_id": row["Constructor"]["constructorId"]}
                        )
                        changes[target].append(
                            {
                                "event_id": event,
                                "points": float(row["points"]),
                                "wins": int(row["wins"]),
                                "position": int(row["position"]) if row.get("position") else None,
                                **identity,
                                **provenance(source, available),
                            }
                        )
        # Upcoming entrants are provisional assumptions from the latest completed event, not confirmations.
        known = {r["event_id"] for r in changes["entries"]}
        last = max(known, default=None)
        if last:
            roster = [r for r in changes["entries"] if r["event_id"] == last]
            last_end = (timestamp(session_time(calendar[last], "race")) + timedelta(hours=4)).isoformat()
            for event in calendar:
                if event > last and event not in known:
                    changes["entries"].extend(
                        [
                            {
                                **r,
                                "event_id": event,
                                "available_at": last_end,
                                "availability_basis": "provisional_previous_roster",
                            }
                            for r in roster
                        ]
                    )
        progress(0.9, "Validating and publishing complete season update")
        return store.publish(changes)
    finally:
        client.close()


def sync_weather(store: Store, event_id: str, check=lambda: None) -> str:
    event = next((e for e in store.table("events") if e["event_id"] == event_id), None)
    if not event:
        raise Unavailable("Unknown event")
    day = timestamp(event["start_at"]).date()
    today = datetime.now(timezone.utc).date()
    if not 0 <= (day - today).days <= 15:
        raise Unavailable(
            "Forecasts are only fetched for future events within 15 days; historical actual weather is not substituted"
        )
    client = PublicClient(store, check)
    try:
        payload, source = client.get(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": event["latitude"],
                "longitude": event["longitude"],
                "hourly": "temperature_2m,precipitation_probability",
                "timezone": "UTC",
                "forecast_days": min((day - today).days + 1, 16),
            },
            "open-meteo",
        )
        issued = now_utc()
        hourly = payload["hourly"]
        target = (
            timestamp(event["start_at"]).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")
        )
        if target not in hourly["time"]:
            raise Unavailable("Provider has no forecast for the requested race time")
        idx = hourly["time"].index(target)
        row = {
            "event_id": event_id,
            "issued_at": issued,
            "valid_at": target + ":00+00:00",
            "temperature_c": hourly["temperature_2m"][idx],
            "rain_probability": hourly["precipitation_probability"][idx] / 100,
            **provenance(source, issued, "observed_issue_time"),
        }
        return store.publish({"weather": [row]})
    finally:
        client.close()


def sync_fastf1(store: Store, season: int, round_number: int, session: str, check=lambda: None) -> str:
    import fastf1

    if season < 2018:
        raise ValueError("Detailed session ingestion starts in 2018")
    labels = {
        "FP1": "practice1",
        "FP2": "practice2",
        "FP3": "practice3",
        "Q": "qualifying",
        "S": "sprint",
        "R": "race",
    }
    if session not in labels:
        raise ValueError("Unsupported session")
    cache = store.root / "fastf1-cache"
    cache.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    check()
    loaded = fastf1.get_session(season, round_number, session)
    loaded.load(laps=True, telemetry=False, weather=True, messages=True)
    check()
    event_id = f"{season}-{round_number:02d}"
    entries = store.table("entries")
    number_map = {str(r.get("number")): r["driver_id"] for r in entries if r["event_id"] == event_id}
    existing_event = next((e for e in store.table("events") if e["event_id"] == event_id), None)
    if not existing_event:
        raise Unavailable("Sync this season before acquiring detailed sessions")
    laps = []
    raw_rows = loaded.laps.reset_index(drop=True).to_json(orient="records", date_format="iso")
    source = store.raw(
        "fastf1", f"fastf1://{season}/{round_number}/{session}", __import__("json").loads(raw_rows)
    )
    start = loaded.date.to_pydatetime().replace(tzinfo=timezone.utc)
    for _, lap in loaded.laps.iterrows():
        number = str(lap.get("DriverNumber", ""))
        driver = number_map.get(number)
        if not driver:
            continue
        lap_time = lap.get("LapTime")
        end = lap.get("Time")
        if __import__("pandas").isna(lap_time) or __import__("pandas").isna(end):
            continue
        available = (start + end.to_pytimedelta()).isoformat()
        laps.append(
            {
                "event_id": event_id,
                "session": labels[session],
                "driver_id": driver,
                "lap": int(lap["LapNumber"]),
                "seconds": lap_time.total_seconds(),
                "compound": str(lap.get("Compound", "UNKNOWN")),
                "tyre_age": float(lap.get("TyreLife", 0)),
                "pit": bool(
                    __import__("pandas").notna(lap.get("PitInTime"))
                    or __import__("pandas").notna(lap.get("PitOutTime"))
                ),
                "green": str(lap.get("TrackStatus", "")) == "1",
                "accurate": bool(lap.get("IsAccurate", False)),
                **provenance(source, available, "reconstructed_lap_end"),
            }
        )
    if not laps:
        raise Unavailable("No valid identified laps; previous telemetry retained")
    return store.publish({"laps": laps})


def sync_openf1(store: Store, event_id: str, session_key: int, check=lambda: None) -> str:
    """Supplement session coverage without fetching car_data/location by default."""
    client = PublicClient(store, check)
    try:
        sessions, source = client.get(
            "https://api.openf1.org/v1/sessions", {"session_key": session_key}, "openf1"
        )
        if len(sessions) != 1:
            raise Unavailable("Session key must identify one session")
        row = sessions[0]
        current = datetime.now(timezone.utc)
        if (
            timestamp(row["date_start"]) - timedelta(minutes=30)
            <= current
            <= timestamp(row["date_end"]) + timedelta(minutes=30)
        ):
            raise Unavailable("Active OpenF1 sessions require paid access; use replay or the free recorder")
        event = next((r for r in store.table("events") if r["event_id"] == event_id), None)
        if not event or abs((timestamp(row["date_start"]) - timestamp(event["start_at"])).days) > 4:
            raise ValueError("OpenF1 session does not match the selected event window")
        payload, lap_source = client.get(
            "https://api.openf1.org/v1/laps", {"session_key": session_key}, "openf1"
        )
        store.add_evidence(
            {
                "kind": "source_coverage",
                "event_id": event_id,
                "session_key": session_key,
                "session_source": source,
                "lap_source": lap_source,
                "lap_rows": len(payload),
                "status": "raw_supplement",
                "note": "Retained raw supplement; FastF1 remains authoritative for practice features.",
            }
        )
        return source
    finally:
        client.close()


def sync_climate(store: Store, event_id: str, check=lambda: None) -> str:
    """Five prior years, same calendar month; no target-race observations enter the prior."""
    event = next((e for e in store.table("events") if e["event_id"] == event_id), None)
    if not event:
        raise Unavailable("Unknown event")
    when = timestamp(event["start_at"])
    end_year = when.year - 1
    if end_year >= datetime.now(timezone.utc).year:
        raise ValueError("Climate history is not yet complete")
    client = PublicClient(store, check)
    try:
        payload, source = client.get(
            "https://archive-api.open-meteo.com/v1/archive",
            {
                "latitude": event["latitude"],
                "longitude": event["longitude"],
                "start_date": f"{end_year - 4}-01-01",
                "end_date": f"{end_year}-12-31",
                "daily": "precipitation_sum",
                "timezone": "UTC",
            },
            "open-meteo",
        )
        daily = payload.get("daily", {})
        selected = [
            p
            for date, p in zip(daily.get("time", []), daily.get("precipitation_sum", []))
            if int(date[5:7]) == when.month and p is not None
        ]
        if len(selected) < 100:
            raise Unavailable("Insufficient climate-prior coverage; previous data retained")
        row = {
            "event_id": event_id,
            "history_end": f"{end_year}-12-31",
            "rain_probability": sum(v >= 1 for v in selected) / len(selected),
            "sample_days": len(selected),
            "definition": "Fraction of days with at least 1mm precipitation in the same month over five prior years; daily climate prior, not a race-hour forecast.",
            **provenance(source, f"{when.year}-01-01T00:00:00Z", "reconstructed_climate_prior"),
        }
        return store.publish({"climate": [row]})
    finally:
        client.close()
