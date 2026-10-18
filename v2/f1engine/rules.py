"""Explicit versioned sporting rules, classification, and championship countback."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .contracts import Unavailable

RULE_VERSION = "sporting-2014-2026-v2.1"
RULE_SOURCE = "https://www.fia.com/regulation/category/110"
GP_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)


@dataclass(frozen=True)
class Rules:
    season: int
    version: str = RULE_VERSION

    def __post_init__(self):
        if not 2014 <= self.season <= 2026:
            raise Unavailable(
                "Implemented scoring profiles cover only 2014–2026; add and validate a new profile first"
            )

    def schedule(
        self,
        field: int,
        *,
        sprint=False,
        distance=1.0,
        green_laps=2,
        final_round=False,
        completed_laps=None,
        suspended_not_resumed=True,
    ) -> np.ndarray:
        if not 0 <= distance <= 1:
            raise ValueError("Distance fraction must be within [0,1]")
        points = np.zeros(field)
        if distance == 0 or (self.season >= 2022 and green_laps < 2):
            return points
        # Before 2022, safety-car laps counted towards this threshold (Spa 2021).
        # The distance for points may differ from the countback classification lap.
        if self.season <= 2021 and completed_laps is not None and completed_laps <= 2:
            return points
        if self.season == 2022 and not sprint and not suspended_not_resumed:
            # The 2022 reduced-points rule only applied if a suspension ended the race.
            # The FIA closed this wording gap for 2023 (Japan 2022 received full points).
            distance = 1.0
        if sprint:
            if self.season < 2021:
                raise ValueError("Sprint scoring does not exist before 2021")
            values = np.array([3, 2, 1] if self.season == 2021 else [8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
            if distance < 0.5:
                return points
        elif self.season >= 2022:
            values = np.array(
                [6, 4, 3, 2, 1]
                if distance < 0.25
                else [13, 10, 8, 6, 5, 4, 3, 2, 1]
                if distance < 0.5
                else [19, 14, 12, 9, 8, 6, 5, 3, 2, 1]
                if distance < 0.75
                else GP_POINTS,
                dtype=float,
            )
        else:
            values = GP_POINTS * (0.5 if distance < 0.75 else 1.0)
        if self.season == 2014 and final_round and not sprint:
            values = values * 2
        points[: min(field, len(values))] = values[:field]
        return points

    def fastest_lap_bonus(self, *, sprint=False, distance=1.0) -> bool:
        return not sprint and 2019 <= self.season <= 2024 and (self.season < 2022 or distance >= 0.5)


def classification(strength: np.ndarray, completed_laps: np.ndarray, rng: np.random.Generator):
    """Most completed laps first; sampled performance breaks equal-distance ties."""
    logits = strength - np.max(strength, axis=1, keepdims=True)
    keys = -np.log(np.clip(rng.random(strength.shape), 1e-15, 1)) / np.exp(np.clip(logits, -50, 0))
    return np.lexsort((keys, -completed_laps), axis=1)


def countback_order(points: np.ndarray, counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Wins, seconds, thirds, ...; unresolved exact ties use an explicit random assumption."""
    tie = rng.random(points.shape)
    keys = [tie] + [counts[:, :, p] for p in range(counts.shape[2] - 1, -1, -1)] + [points]
    # Descending all fields. The last lexsort key is primary.
    return np.flip(np.lexsort(tuple(keys), axis=1), axis=1)
