"""One scoring contract for sampled backtests and immutable saved forecasts."""

import numpy as np

SCORING_VERSION = "full-field-rps-v2.1"


def distribution_metrics(drivers, actual, simulations):
    ids = [r["driver_id"] for r in drivers]
    lookup = {r["driver_id"]: r for r in actual}
    n = len(ids)
    if n < 2 or len(set(ids)) != n or len(lookup) != len(actual) or set(ids) != set(lookup):
        raise ValueError("Scoring requires exactly one target and prediction for every entrant")
    pmf = np.array([r["position_probabilities"] for r in drivers], dtype=float)
    if pmf.shape != (n, n) or not np.isfinite(pmf).all() or np.any(pmf < 0):
        raise ValueError("Invalid position distribution")
    if not np.allclose(pmf.sum(axis=1), 1, atol=1e-9) or not np.allclose(pmf.sum(axis=0), 1, atol=1e-9):
        raise ValueError("Position distributions do not conserve the whole field")
    finish = np.array([lookup[d]["position"] for d in ids])
    if sorted(finish.tolist()) != list(range(1, n + 1)):
        raise ValueError("Targets must be a complete unique classification")
    cdf = np.cumsum(pmf, axis=1)
    rps = np.mean((cdf[:, :-1] - (finish[:, None] <= np.arange(1, n))) ** 2)
    podium = pmf[:, : min(3, n)].sum(axis=1)
    points = np.array([r["points_probability"] for r in drivers])
    retirement = np.array([r["retirement_probability"] for r in drivers])
    expected = pmf @ np.arange(1, n + 1)
    low = np.argmax(cdf >= 0.1 - 1e-12, axis=1) + 1
    high = np.argmax(cdf >= 0.9 - 1e-12, axis=1) + 1
    winner = np.flatnonzero(finish == 1)[0]
    win_estimator = (pmf[:, 0] * simulations + 0.5) / (simulations + 0.5 * n)
    truth_points = np.array([lookup[d].get("points", 0) > 0 for d in ids])
    truth_retirement = np.array([lookup[d].get("retired", False) for d in ids])
    for values in (points, retirement):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid event probability")
    return {
        "rps": float(rps),
        "winner_log_loss": float(-np.log(win_estimator[winner])),
        "podium_brier": float(np.mean((podium - (finish <= min(3, n))) ** 2)),
        "points_brier": float(np.mean((points - truth_points) ** 2)),
        "retirement_brier": float(np.mean((retirement - truth_retirement) ** 2)),
        "position_mae": float(np.mean(np.abs(expected - finish))),
        "interval_coverage": float(np.mean((finish >= low) & (finish <= high))),
        "driver_count": n,
        "scoring_version": SCORING_VERSION,
        "winner_log_loss_estimator": "Jeffreys finite-Monte-Carlo correction; displayed probabilities remain unchanged",
        "calibration_pairs": {
            "win": list(zip(pmf[:, 0].tolist(), (finish == 1).astype(int).tolist())),
            "podium": list(zip(podium.tolist(), (finish <= min(3, n)).astype(int).tolist())),
            "points": list(zip(points.tolist(), truth_points.astype(int).tolist())),
            "retirement": list(zip(retirement.tolist(), truth_retirement.astype(int).tolist())),
        },
    }
