"""Measured local performance, always labelled with dataset and hardware context."""

import platform
import tempfile
import time
from .contracts import ForecastRequest, SeasonRequest, code_version
from .demo import load_demo, DEMO_EVENT, DEMO_CUTOFF
from .simulation import forecast, simulate_season
from .store import Store


def benchmark(evidence_store: Store):
    with tempfile.TemporaryDirectory(prefix="f1engine-benchmark-") as root:
        store = Store(root)
        load_demo(store, rounds=24)
        race = ForecastRequest(
            event_id=DEMO_EVENT, as_of_utc=DEMO_CUTOFF, stage="post_sprint", simulations=10000
        )
        season = SeasonRequest(season=2026, as_of_utc=DEMO_CUTOFF, simulations=10000)
        forecast(store, race, persist=False)
        simulate_season(store, season, persist=False)
        timings = {}
        for name, fn, target in [
            ("race", lambda: forecast(store, race, persist=False), 10),
            ("season", lambda: simulate_season(store, season, persist=False), 30),
        ]:
            values = []
            for _ in range(3):
                start = time.perf_counter()
                fn()
                values.append(time.perf_counter() - start)
            timings[name] = {
                "seconds": values,
                "max_seconds": max(values),
                "target_seconds": target,
                "passed": max(values) < target,
            }
        result = {
            "kind": "performance_benchmark",
            "evidence_kind": "illustrative_scenario",
            "code_version": code_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "simulations": 10000,
            "field": 20,
            "remaining_rounds": 22,
            "timings": timings,
            "scope": "Cached illustrative stress fixture with 20 drivers and 22 remaining rounds; does not measure prediction accuracy.",
        }
        result["evidence_id"] = evidence_store.add_evidence(result)
        return result
