"""Helpers for handling regulation-era uncertainty and constructor priors."""

from __future__ import annotations

from dataclasses import dataclass

from src.elo_rating import get_constructor_elo
from src.registry import REGULATION_CHANGES


@dataclass(slots=True)
class RegulationHandler:
    """Apply honest uncertainty adjustments around major rule resets."""

    mean_elo: float = 1500.0

    def get_constructor_prior(self, constructor_id: str, season: int) -> float:
        """Return a regulation-adjusted constructor ELO prior."""

        season = int(season)
        if season not in REGULATION_CHANGES:
            return float(get_constructor_elo(constructor_id, season, round_number=0))

        if REGULATION_CHANGES[season] == "major":
            historical_elo = float(get_constructor_elo(constructor_id, season - 1, round_number=999))
            return historical_elo * 0.60 + self.mean_elo * 0.40
        return float(get_constructor_elo(constructor_id, season, round_number=0))

    def get_season_uncertainty(self, season: int, round_number: int) -> float:
        """Return a confidence multiplier for a season/round pair."""

        season = int(season)
        round_number = int(round_number)
        if season in REGULATION_CHANGES and REGULATION_CHANGES[season] == "major":
            if round_number <= 5:
                return 0.60
            if round_number <= 10:
                return 0.75
            return 0.90
        return 1.0

    def has_reset_warning(self, season: int, round_number: int) -> bool:
        """Return whether a regulation-reset warning should be shown."""

        return bool(
            int(season) in REGULATION_CHANGES
            and REGULATION_CHANGES[int(season)] == "major"
            and int(round_number) <= 10
        )
