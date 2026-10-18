"""Optional unauthenticated FastF1 recorder with local arrival timestamps.

Partial/unavailable free feeds remain unavailable. Capture never requests credentials.
"""

from datetime import datetime
import json
from pathlib import Path
import signal
import time
from .contracts import canonical, digest, now_utc, timestamp, Unavailable

TOPICS = [
    "Heartbeat",
    "DriverList",
    "SessionInfo",
    "SessionStatus",
    "TimingData",
    "TimingAppData",
    "TrackStatus",
    "LapCount",
]


def capture(path, minutes=45, positions=False):
    from fastf1.livetiming.client import SignalRClient

    if not 1 <= minutes <= 45:
        raise ValueError("Recorder budget is 1–45 minutes")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    class ArrivalClient(SignalRClient):
        def _on_message(self, msg):
            received = now_utc()
            self._t_last_message = time.time()
            if isinstance(msg, list):
                payload = msg
            else:
                payload = {"snapshot": getattr(msg, "result", {})}
            envelope = {"received_at": received, "payload": payload}
            with output.open("a") as file:
                file.write(canonical(envelope) + "\n")

    client = ArrivalClient(str(output) + ".fastf1", filemode="a", timeout=60, no_auth=True)
    client.topics = TOPICS + (["Position.z"] if positions else [])

    def stop(*_):
        raise TimeoutError("Recording budget reached")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, minutes * 60)
    try:
        client.start()
    except (TimeoutError, KeyboardInterrupt):
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if client._connection:
            client._connection.stop()
        if client._output_file:
            client._output_file.close()
    return {
        "path": str(output.resolve()),
        "free_feed": True,
        "available": output.exists() and output.stat().st_size > 0,
        "note": "Verify coverage before forecasting; unauthenticated feeds may be empty or partial.",
    }


def merge(old, new):
    for key, value in new.items():
        if isinstance(value, dict):
            old[key] = merge(old.get(key, {}) if isinstance(old.get(key), dict) else {}, value)
        else:
            old[key] = value
    return old


def normalize(envelopes, event_id, session_start, number_map):
    """Convert timestamped TimingData deltas; never invent arrival times for legacy files."""
    from .features import lap_seconds

    start = timestamp(session_start)
    state = {}
    tyres = {}
    status = "unknown"
    output = []
    counts = {}
    xy = {}
    crossings = {}
    dirty = set()
    for env in sorted(envelopes, key=lambda e: timestamp(e["received_at"])):
        received = timestamp(env["received_at"])
        payload = env["payload"]
        if isinstance(payload, dict):
            continue  # initial snapshots lack reliable observation timestamps
        if not isinstance(payload, list) or len(payload) != 3:
            continue
        topic, data, at = payload
        if not at:
            continue
        occurred = timestamp(at)
        if received < occurred:
            raise ValueError("Invalid arrival timestamp")
        if isinstance(data, str) and not topic.endswith(".z"):
            data = json.loads(data)
        if topic == "TrackStatus":
            status = {
                "1": "green",
                "2": "yellow",
                "4": "safety_car",
                "5": "red",
                "6": "safety_car",
                "7": "green",
            }.get(str(data.get("Status")), status)
            if status != "green":
                dirty.update(state)
            output.append(
                {
                    "id": "track:" + occurred.isoformat(),
                    "event_id": event_id,
                    "kind": "race_control",
                    "occurred_at": occurred.isoformat(),
                    "received_at": received.isoformat(),
                    "status": status,
                }
            )
            continue
        if topic == "TimingAppData":
            for number, line in data.get("Lines", {}).items():
                tyres[number] = merge(tyres.get(number, {}), line)
            continue
        if topic == "Position.z":
            import base64, zlib

            decoded = json.loads(zlib.decompress(base64.b64decode(data), -zlib.MAX_WBITS))
            for group in decoded.get("Position", []):
                for number, value in group.get("Entries", {}).items():
                    if "X" in value and "Y" in value:
                        xy[number] = [value["X"], value["Y"]]
            continue
        if topic != "TimingData":
            continue
        for number, delta in data.get("Lines", {}).items():
            if number not in number_map:
                continue
            state[number] = merge(state.get(number, {}), delta)
            current = state[number]
            if current.get("InPit") or current.get("PitOut"):
                dirty.add(number)
            # Timestamp lap completions only. Other deltas cannot create fictitious lap crossings.
            if "NumberOfLaps" not in delta and not delta.get("Retired"):
                continue
            lap = int(current.get("NumberOfLaps", 0))
            if lap < 1:
                continue
            elapsed = (occurred - start).total_seconds()
            if elapsed < 0:
                continue
            stints = tyres.get(number, {}).get("Stints", {})
            stint = list(stints.values())[-1] if isinstance(stints, dict) and stints else {}
            identifier = f"{number}:{lap}"
            revision = counts.get(identifier, -1) + 1
            counts[identifier] = revision
            previous = crossings.get(number)
            same_crossing = previous is not None and previous["lap"] == lap
            crossing_at = timestamp(previous["occurred_at"]) if same_crossing else occurred
            # Never reuse a previous lap's LastLapTime after a partial timing delta.
            seconds = lap_seconds(delta.get("LastLapTime", {}).get("Value"))
            if seconds is None and previous is not None and previous["lap"] == lap - 1:
                seconds = (crossing_at - timestamp(previous["occurred_at"])).total_seconds()
            if seconds is None and same_crossing:
                seconds = previous["lap_seconds"]
            pit = number in dirty or bool(current.get("InPit") or current.get("PitOut"))
            valid = seconds is not None and status == "green" and not pit and "InPit" in current
            out = {
                "id": identifier,
                "revision": revision,
                "event_id": event_id,
                "occurred_at": crossing_at.isoformat(),
                "received_at": received.isoformat(),
                "kind": "timing",
                "driver_id": number_map[number],
                "lap": lap,
                "elapsed": (crossing_at - start).total_seconds(),
                "lap_seconds": seconds,
                "pit": pit,
                "pace_valid": valid,
                "compound": stint.get("Compound"),
                "tyre_age": int(stint.get("TotalLaps", 0)) if stint else None,
                "retired": bool(current.get("Retired", False)),
                "status": status,
            }
            if number in xy:
                out.update(x=xy[number][0], y=xy[number][1])
            output.append(out)
            crossings[number] = out
            if not same_crossing:
                dirty.discard(number)
    if not any(m["kind"] == "timing" for m in output):
        raise Unavailable("Recording has no identifiable, timestamped timing observations")
    return output
