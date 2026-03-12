from __future__ import annotations

from src.regulation_handler import RegulationHandler


def test_regulation_handler_major_reset_behavior() -> None:
    handler = RegulationHandler()
    assert handler.has_reset_warning(2026, 2) is True
    assert handler.get_season_uncertainty(2026, 2) == 0.60
    assert handler.get_season_uncertainty(2026, 8) == 0.75
    assert handler.get_season_uncertainty(2026, 12) == 0.90
    assert handler.get_season_uncertainty(2025, 5) == 1.0
