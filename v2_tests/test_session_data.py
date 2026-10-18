from f1engine.session_data import normalize_openf1_laps, track_states
from f1engine.contracts import timestamp
import pytest
from f1engine.session_data import entry_number_map
from f1engine.contracts import Unavailable


def bundle():
    return {
        "laps": [
            {
                "driver_number": 1,
                "lap_number": 3,
                "date_start": "2024-08-30T15:02:00Z",
                "lap_duration": 90.0,
                "duration_sector_1": 30.0,
                "duration_sector_2": 30.0,
                "duration_sector_3": 30.0,
                "is_pit_out_lap": False,
            }
        ],
        "stints": [
            {
                "driver_number": 1,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 10,
                "compound": "MEDIUM",
                "tyre_age_at_start": 0,
            }
        ],
        "pit": [],
        "race_control": [{"date": "2024-08-30T15:00:00Z", "flag": "GREEN", "scope": "Track"}],
    }


def test_duplicate_car_numbers_fail_before_mapping_timing():
    entries = [{"event_id": "2024-16", "driver_id": d, "number": "1"} for d in ("norris", "max_verstappen")]
    with pytest.raises(Unavailable, match="Conflicting event-specific"):
        entry_number_map(entries, "2024-16")


def test_lap_filter_marks_crossing_neutralization_and_missing_sectors():
    data = bundle()
    good = normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source")[0]
    assert good["green"] and good["accurate"] and not good["pit"] and good["tyre_age"] == 3
    data["race_control"].append(
        {"date": "2024-08-30T15:03:00Z", "flag": "DOUBLE YELLOW", "scope": "Sector", "sector": 9}
    )
    assert not normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source")[0]["green"]
    data["laps"][0]["duration_sector_2"] = None
    assert not normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source")[0]["accurate"]


def test_clear_flag_does_not_end_red_flag_and_future_flags_do_not_change_lap():
    controls = [
        {"date": "2024-08-30T15:00:00Z", "flag": "RED", "scope": "Track"},
        {"date": "2024-08-30T15:01:00Z", "flag": "CLEAR", "scope": "Track"},
    ]
    assert track_states(controls)[-1][1] == "red"
    data = bundle()
    before = normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source")
    data["race_control"].append({"date": "2024-08-30T15:04:00Z", "flag": "RED", "scope": "Track"})
    assert normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source") == before


def test_deletion_is_a_later_revision_and_pit_neighbours_excluded():
    data = bundle()
    data["race_control"].append(
        {"date": "2024-08-30T15:05:00Z", "driver_number": 1, "lap_number": 3, "message": "LAP TIME DELETED"}
    )
    rows = normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source")
    assert rows[0]["accurate"] and not rows[1]["accurate"]
    assert timestamp(rows[1]["available_at"]) > timestamp(rows[0]["available_at"])
    data["pit"] = [{"driver_number": 1, "lap_number": 4}]
    assert normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source")[0]["pit"]


def test_text_identifies_deleted_lap_not_race_control_leader_lap():
    data = bundle()
    data["race_control"] += [
        {
            "date": "2024-08-30T15:05:00Z",
            "driver_number": None,
            "lap_number": 5,
            "message": "CAR 1 (VER) TIME 1:30.000 DELETED - TRACK LIMITS LAP 3",
        },
        {
            "date": "2024-08-30T15:06:00Z",
            "driver_number": None,
            "lap_number": 6,
            "message": "CAR 1 (VER) LAP 3 TIME REINSTATED",
        },
    ]
    rows = normalize_openf1_laps(data, "2024-16", "practice2", {"1": "driver"}, "source")
    assert [r["accurate"] for r in rows] == [True, False, True]
