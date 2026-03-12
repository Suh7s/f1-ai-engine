#!/usr/bin/env python3
"""Automatic F1 pipeline scheduler — runs after each race weekend.

Designed to be triggered by cron or launchd. Checks the F1 calendar,
determines if a session recently completed, and runs the appropriate
pipeline update.

Usage:
    python scripts/auto_scheduler.py              # check and update if needed
    python scripts/auto_scheduler.py --force      # force update regardless of timing
    python scripts/auto_scheduler.py --dry-run    # show what would happen without running
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import ErgastClient, fetch_calendar, get_cached_rounds, get_current_season
from src.scheduler import auto_update, check_for_new_data
from src.utils import CONFIG as BASE_CONFIG, get_logger

LOGGER = get_logger("auto_scheduler")
STATE_PATH = BASE_CONFIG["processed_dir"] / "auto_scheduler_state.json"


def _load_state() -> dict[str, Any]:
    """Load scheduler state (last update timestamp, rounds processed)."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_update_utc": None, "last_round": 0, "season": get_current_season(), "history": []}


def _save_state(state: dict[str, Any]) -> None:
    """Persist scheduler state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _get_race_schedule() -> list[dict[str, Any]]:
    """Fetch the current season calendar with race datetimes."""
    try:
        current_season = get_current_season()
        client = ErgastClient()
        _, calendar_df = fetch_calendar(client, current_season)
        if calendar_df.empty:
            return []
        races = []
        for _, row in calendar_df.iterrows():
            race_dt = None
            if "race_datetime" in row and row["race_datetime"]:
                try:
                    race_dt = datetime.fromisoformat(str(row["race_datetime"]).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            if race_dt is None and "date" in row:
                try:
                    race_dt = datetime.fromisoformat(str(row["date"]) + "T15:00:00+00:00")
                except (ValueError, TypeError):
                    continue
            if race_dt is None:
                continue
            if race_dt.tzinfo is None:
                race_dt = race_dt.replace(tzinfo=timezone.utc)
            races.append({
                "round": int(row.get("round", 0)),
                "race_name": str(row.get("race_name", row.get("raceName", ""))),
                "circuit_id": str(row.get("circuit_id", "")),
                "race_datetime_utc": race_dt,
            })
        return sorted(races, key=lambda r: r["round"])
    except Exception as exc:
        LOGGER.warning("Failed to fetch calendar: %s", exc)
        return []


def _should_update(state: dict[str, Any], schedule: list[dict[str, Any]], window_hours: int = 8) -> dict[str, Any]:
    """Determine if an update should run.

    Returns a dict with 'should_update' bool and reason string.
    An update is triggered if:
    1. A race finished within the last `window_hours` hours, AND
    2. We haven't already processed that round.
    """
    now = datetime.now(timezone.utc)
    new_rounds = check_for_new_data()

    if new_rounds:
        last_processed = state.get("last_round", 0)
        truly_new = [r for r in new_rounds if r > last_processed]
        if truly_new:
            return {
                "should_update": True,
                "reason": f"New completed rounds detected: {truly_new}",
                "new_rounds": truly_new,
            }

    # Also check calendar timing — a race may have just finished but API not yet updated
    for race in schedule:
        race_end_est = race["race_datetime_utc"] + timedelta(hours=2)  # race ~2hrs
        if race_end_est <= now <= race_end_est + timedelta(hours=window_hours):
            if race["round"] > state.get("last_round", 0):
                return {
                    "should_update": True,
                    "reason": f"Race R{race['round']} ({race['race_name']}) recently finished — waiting for API data",
                    "new_rounds": [race["round"]],
                    "pending": True,
                }

    return {"should_update": False, "reason": "No new race data available"}


def run_update(skip_fastf1: bool = True, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Main entry point: check if update needed and run it."""
    state = _load_state()
    current_season = get_current_season()

    # Reset state if season changed
    if state.get("season") != current_season:
        state = {"last_update_utc": None, "last_round": 0, "season": current_season, "history": []}

    schedule = _get_race_schedule()
    check = _should_update(state, schedule)

    if not check["should_update"] and not force:
        LOGGER.info("No update needed: %s", check["reason"])
        return {"action": "skip", "reason": check["reason"]}

    reason = "Forced update" if force else check["reason"]
    LOGGER.info("Starting update: %s", reason)

    if dry_run:
        LOGGER.info("[DRY RUN] Would run auto_update(skip_fastf1=%s)", skip_fastf1)
        return {"action": "dry_run", "reason": reason}

    # If API data not ready yet, try but don't fail hard
    if check.get("pending"):
        new_rounds = check_for_new_data()
        if not new_rounds:
            LOGGER.info("Race recently finished but API data not available yet. Will retry next check.")
            return {"action": "pending", "reason": "API data not ready yet, will retry"}

    result = auto_update(skip_fastf1=skip_fastf1)

    if result.get("updated"):
        new_rounds = result.get("new_rounds", [])
        state["last_update_utc"] = datetime.now(timezone.utc).isoformat()
        state["last_round"] = max(new_rounds) if new_rounds else state.get("last_round", 0)
        state["history"].append({
            "timestamp": state["last_update_utc"],
            "rounds": new_rounds,
            "skip_fastf1": skip_fastf1,
        })
        # Keep only last 30 history entries
        state["history"] = state["history"][-30:]
        _save_state(state)
        LOGGER.info("Update complete: rounds %s processed", new_rounds)
    else:
        LOGGER.info("auto_update returned no changes")

    return {"action": "updated" if result.get("updated") else "no_change", "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 auto-scheduler: check for new race data and update pipeline")
    parser.add_argument("--force", action="store_true", help="Force update regardless of timing")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without running")
    parser.add_argument("--skip-fastf1", action="store_true", default=True, help="Skip FastF1 telemetry (default: True)")
    parser.add_argument("--with-fastf1", action="store_true", help="Include FastF1 telemetry (slow)")
    args = parser.parse_args()

    skip_fastf1 = not args.with_fastf1

    result = run_update(skip_fastf1=skip_fastf1, force=args.force, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
