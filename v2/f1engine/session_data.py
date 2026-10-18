"""Conservative, timestamped normalization of public archived session records."""

from datetime import timedelta
import math
import re
from .contracts import timestamp, Unavailable


def entry_number_map(entries, event_id):
    result = {}
    for row in entries:
        if row["event_id"] != event_id or row.get("number") is None:
            continue
        number = str(row["number"])
        if number in result and result[number] != row["driver_id"]:
            raise Unavailable("Conflicting event-specific car numbers; session identities are unavailable")
        result[number] = row["driver_id"]
    return result


def track_states(messages):
    """Maintain separate track and sector flags; CLEAR does not restart a red-flag session."""
    track, sectors = "unknown", set()
    timeline = []
    for message in sorted(messages, key=lambda r: timestamp(r["date"])):
        flag, scope = message.get("flag"), message.get("scope")
        text = message.get("message", "").upper()
        if flag == "GREEN" and scope == "Track":
            track, sectors = "green", set()
        elif flag == "RED" or text == "SESSION ABORTED":
            track = "red"
        elif flag == "CHEQUERED" or text == "SESSION FINISHED":
            track = "finished"
        elif "SAFETY CAR DEPLOYED" in text or "VIRTUAL SAFETY CAR DEPLOYED" in text:
            track = "safety_car"
        elif flag in {"YELLOW", "DOUBLE YELLOW"}:
            sectors.add(message.get("sector") if scope == "Sector" else "track")
        elif flag == "CLEAR":
            if scope == "Sector":
                sectors.discard(message.get("sector"))
            elif scope == "Track":
                sectors.clear()
        status = "yellow" if track == "green" and sectors else track
        timeline.append((timestamp(message["date"]), status))
    return timeline


def normalize_openf1_laps(bundle, event_id, session, number_map, source_id):
    """Reconstruct lap-end availability; original message-arrival times are unavailable."""
    controls = bundle["race_control"]
    timeline = track_states(controls)
    pits = {(str(r["driver_number"]), r["lap_number"]) for r in bundle["pit"]}
    rows = []
    for lap in bundle["laps"]:
        number, duration = str(lap["driver_number"]), lap.get("lap_duration")
        if number not in number_map or not lap.get("date_start") or duration is None:
            continue
        if not math.isfinite(duration) or duration <= 0:
            continue
        start = timestamp(lap["date_start"])
        end = start + timedelta(seconds=duration)
        lap_number = int(lap["lap_number"])
        stints = [
            r
            for r in bundle["stints"]
            if str(r["driver_number"]) == number
            and r["lap_start"] <= lap_number <= (r.get("lap_end") or lap_number)
        ]
        stint = max(stints, key=lambda r: r["stint_number"]) if stints else {}
        statuses = [status for at, status in timeline if at <= start][-1:]
        statuses += [status for at, status in timeline if start < at <= end]
        sectors = [lap.get(f"duration_sector_{i}") for i in (1, 2, 3)]
        complete = all(v is not None and math.isfinite(v) and v > 0 for v in sectors)
        accurate = complete and abs(sum(sectors) - duration) <= 0.02
        row = {
            "event_id": event_id,
            "session": session,
            "driver_id": number_map[number],
            "lap": lap_number,
            "seconds": float(duration),
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "compound": stint.get("compound"),
            "tyre_age": (stint["tyre_age_at_start"] + lap_number - stint["lap_start"] + 1)
            if stint.get("tyre_age_at_start") is not None
            else None,
            "pit": bool(
                lap.get("is_pit_out_lap")
                or (number, lap_number) in pits
                or (number, lap_number + 1) in pits
                or lap_number == stint.get("lap_end")
            ),
            "green": bool(statuses) and all(s == "green" for s in statuses),
            "accurate": bool(accurate),
            "source_id": source_id,
            "available_at": end.isoformat(),
            "availability_basis": "reconstructed_archived_lap_end",
            "timing_quality": "sector-sum checked; archive publication and correction arrival times unavailable",
        }
        rows.append(row)
        # An explicitly numbered deletion becomes visible when the control message occurs.
        # Unnumbered deletions conservatively invalidate that driver's earlier laps at that time.
        decisions = []
        for control in controls:
            text = control.get("message", "").upper()
            car = re.search(r"\bCAR (\d+)\b", text)
            target = re.search(r"\bLAP (\d+)\b", text)
            driver_number = (
                str(control.get("driver_number"))
                if control.get("driver_number") is not None
                else car.group(1)
                if car
                else None
            )
            if driver_number != number or timestamp(control["date"]) < end:
                continue
            if target and int(target.group(1)) != lap_number:
                continue
            if "DELETED" in text or (target and "REINSTATED" in text):
                decisions.append(control)
        for deletion in sorted(decisions, key=lambda r: timestamp(r["date"])):
            rows.append(
                {
                    **row,
                    "accurate": accurate if "REINSTATED" in deletion.get("message", "").upper() else False,
                    "available_at": deletion["date"],
                    "availability_basis": "reconstructed_lap_steward_revision",
                }
            )
    if not rows:
        raise Unavailable("Archived session has no identified complete lap timestamps")
    return rows
