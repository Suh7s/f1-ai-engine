"""Explicit, append-only reconstruction repairs from retained public source records."""

from .contracts import Unavailable, code_version
from .session_data import entry_number_map, normalize_openf1_laps


def repair_identities(store, check=lambda: None):
    version = store.current_version()
    if store.manifest(version)["illustrative"]:
        raise Unavailable("Identity repair applies to observed Jolpica/OpenF1 archives only")
    cache, repaired = {}, []
    affected_events = set()
    number_changes = 0
    entries = store.table("entries", version)
    for entry in entries:
        check()
        identifier = entry["source_id"]
        if identifier not in cache:
            source = store.source(identifier)
            numbers = {}
            for race in source["payload"].get("MRData", {}).get("RaceTable", {}).get("Races", []):
                event = f"{race['season']}-{int(race['round']):02d}"
                for key in ("Results", "SprintResults", "QualifyingResults"):
                    for result in race.get(key, []):
                        numbers[(event, result["Driver"]["driverId"])] = result.get("number")
            cache[identifier] = numbers
        key = (entry["event_id"], entry["driver_id"])
        if key not in cache[identifier]:
            continue  # Provisional rosters retain their explicit assumption.
        number = cache[identifier][key]
        updated = {
            **entry,
            "number": number,
            "number_basis": "event_specific_result" if number is not None else "unavailable",
        }
        if updated != entry:
            number_changes += entry.get("number") != number
            if entry.get("number") != number:
                affected_events.add(entry["event_id"])
            repaired.append(updated)
    merged = {(r["event_id"], r["driver_id"]): r for r in entries}
    merged.update({(r["event_id"], r["driver_id"]): r for r in repaired})
    old_laps = store.table("laps", version)
    new_laps = []
    retired_keys = 0
    bundles = []
    for source_id in sorted({r["source_id"] for r in old_laps}):
        check()
        source = store.source(source_id)
        bundle = source["payload"]
        if (
            source["source"] != "openf1"
            or not isinstance(bundle, dict)
            or not {"laps", "stints", "pit", "race_control", "session"}.issubset(bundle)
        ):
            continue
        related = [r for r in old_laps if r["source_id"] == source_id]
        event, session = related[0]["event_id"], related[0]["session"]
        if event not in affected_events:
            continue
        mapping = entry_number_map(list(merged.values()), event)
        normalized = normalize_openf1_laps(bundle, event, session, mapping, source_id)
        keys = {(r["driver_id"], r["lap"]) for r in normalized}
        for row in related:
            if (row["driver_id"], row["lap"]) not in keys:
                new_laps.append(
                    {
                        **row,
                        "accurate": False,
                        "availability_basis": "reconstruction_correction_invalid_identity",
                    }
                )
                retired_keys += 1
        new_laps.extend(normalized)
        bundles.append(source_id)
    updated = store.publish({"entries": repaired, "laps": new_laps})
    body = {
        "kind": "identity_repair",
        "parent_data_version": version,
        "data_version": updated,
        "code_version": code_version(),
        "entry_metadata_revisions": len(repaired),
        "car_number_corrections": number_changes,
        "rebuilt_lap_rows": len(new_laps),
        "invalidated_misidentified_keys": retired_keys,
        "bundle_sources": bundles,
        "reason": "Historical car numbers come from event results, never the mutable current driver profile. Rebuilt archived lap identities preserve earlier dataset versions; obsolete keys are explicitly invalidated rather than deleted.",
    }
    return {**body, "evidence_id": store.add_evidence(body)}
