from __future__ import annotations

from src.utils import format_race_label, get_circuit_name, get_gp_name


def test_gp_name_helpers_resolve_calendar_rows() -> None:
    assert get_gp_name(2026, 1) == "Australian Grand Prix"
    assert get_gp_name(2026, 2) == "Chinese Grand Prix"
    assert get_gp_name(2026, 4) == "Bahrain Grand Prix"


def test_circuit_name_helper_resolves_circuit() -> None:
    assert get_circuit_name(2026, 1) == "Albert Park Grand Prix Circuit"
    assert get_circuit_name(2026, 2) == "Shanghai International Circuit"


def test_race_label_uses_gp_name_not_round_only() -> None:
    label = format_race_label(2026, 2)
    assert "Chinese Grand Prix" in label
    assert "Round 2" in label
    assert "2026" in label
