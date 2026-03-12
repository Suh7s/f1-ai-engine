from __future__ import annotations

import pandas as pd

from dashboard.components.live_tracker import _position_status, load_replay_snapshot, timing_tower_frame
from dashboard.components.track_map import render_circuit_map
from src.circuit_maps import build_circuit_svg, load_circuit_transform, parse_svg_to_coordinates


def test_track_map_renders_circuit_and_driver_overlay() -> None:
    positions = pd.DataFrame(
        [
            {
                "driver_number": "1",
                "driver_abbr": "VER",
                "driver_name": "Max Verstappen",
                "team_color": "#3671C6",
                "position_in_race": 1,
                "gap_to_leader": "Leader",
                "current_lap": 23,
                "compound": "Medium",
                "speed": 301,
                "x": 0.0,
                "y": 0.0,
            }
        ]
    )
    figure = render_circuit_map("bahrain", positions_df=positions, session_type="race")
    assert len(figure.data) >= 3
    assert figure.layout.title.text.startswith("Bahrain Grand Prix")


def test_replay_snapshot_falls_back_to_local_results_without_openf1() -> None:
    snapshot = load_replay_snapshot(2026, 1)
    assert snapshot.circuit_id
    assert not snapshot.positions.empty
    figure = render_circuit_map(snapshot.circuit_id, positions_df=snapshot.positions.head(5))
    tower = timing_tower_frame(snapshot)
    assert len(figure.data) >= 3
    assert not tower.empty


def test_circuit_builder_uses_distinct_cached_geometry() -> None:
    bahrain = parse_svg_to_coordinates(build_circuit_svg("bahrain", refresh=True))
    monaco = parse_svg_to_coordinates(build_circuit_svg("monaco", refresh=True))
    assert load_circuit_transform("bahrain")["geometry_source"] == "remote"
    assert load_circuit_transform("monaco")["geometry_source"] == "remote"
    assert len(bahrain) > 50
    assert len(monaco) > 50
    assert len(bahrain) != 720
    assert len(monaco) != 720
    assert len(bahrain) != len(monaco)


def test_position_status_uses_recency_and_vsc_priority() -> None:
    vsc_messages = pd.DataFrame([{"date": "2026-03-09T12:00:00Z", "message": "VIRTUAL SAFETY CAR DEPLOYED"}])
    assert _position_status(vsc_messages) == ("VIRTUAL SAFETY CAR", True)

    resumed_messages = pd.DataFrame(
        [
            {"date": "2026-03-09T12:00:00Z", "message": "RED FLAG"},
            {"date": "2026-03-09T12:05:00Z", "message": "GREEN FLAG"},
        ]
    )
    assert _position_status(resumed_messages) == ("GREEN", False)
