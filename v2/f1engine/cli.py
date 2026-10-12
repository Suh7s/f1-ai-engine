"""One local launch command. No network access or artifact loading at startup."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .contracts import canonical, ForecastRequest, SeasonRequest, Settings
from .store import Store


def main():
    parser = argparse.ArgumentParser(prog="f1engine", description="Evidence-first local F1 forecasting")
    parser.add_argument("--home", help="Data directory (default: .f1engine)")
    sub = parser.add_subparsers(dest="command", required=True)
    app = sub.add_parser("app")
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--demo", action="store_true")
    sub.add_parser("doctor")
    sub.add_parser("demo")
    sub.add_parser("verify")
    sub.add_parser("jobs")
    sync = sub.add_parser("sync")
    sync.add_argument("season", type=int)
    standings = sub.add_parser("standings")
    standings.add_argument("event_id", help="Official standings after this round, e.g. 2024-17")
    for name in ("experiment", "holdout"):
        p = sub.add_parser(name)
        p.add_argument("--parameters", default="{}", help="JSON parameters")
    for name in ("cancel", "resume"):
        p = sub.add_parser(name)
        p.add_argument("job_id")
    for name in ("forecast", "championship"):
        p = sub.add_parser(name)
        p.add_argument("--request", required=True, help="JSON request")
    p = sub.add_parser("reproduce")
    p.add_argument("forecast_id")
    p = sub.add_parser("record")
    p.add_argument("output")
    p.add_argument("--minutes", type=int, default=45)
    p.add_argument("--positions", action="store_true")
    p = sub.add_parser("normalize-recording")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--event", required=True)
    p.add_argument("--start", required=True)
    sub.add_parser("score")
    sub.add_parser("benchmark")
    args = parser.parse_args()
    root = Settings(args.home).root
    if args.command == "app":
        if args.demo:
            root = root / "demo"
        from .api import create_app
        import uvicorn

        uvicorn.run(create_app(root, args.demo), host="127.0.0.1", port=args.port)
        return
    store = Store(root)
    from . import jobs

    if args.command == "demo":
        from .demo import load_demo

        result = {"data_version": load_demo(store), "illustrative": True}
    elif args.command == "doctor":
        import importlib.metadata

        result = {
            "python": sys.version,
            "root": str(root),
            "dataset": store.current_version(),
            "ledger": store.verify_ledger(),
            "ui_built": (Path(__file__).parent / "static" / "index.html").exists(),
            "packages": {
                p: importlib.metadata.version(p)
                for p in ["fastapi", "numpy", "pandas", "pyarrow", "xgboost", "fastf1"]
            },
        }
        try:
            __import__("xgboost")
            result["xgboost"] = "ready"
        except Exception as exc:
            result["xgboost"] = f"unavailable: {exc}"
    elif args.command == "verify":
        result = store.verify_ledger()
    elif args.command == "jobs":
        jobs.recover(store)
        result = jobs.jobs(store)
    elif args.command == "sync":
        result = jobs.submit(store, "sync_season", {"season": args.season})
    elif args.command == "standings":
        result = jobs.submit(store, "sync_standings", {"event_id": args.event_id})
    elif args.command in {"experiment", "holdout"}:
        result = jobs.submit(store, args.command, json.loads(args.parameters))
    elif args.command in {"cancel", "resume"}:
        result = getattr(jobs, args.command)(store, args.job_id)
    elif args.command == "forecast":
        from .simulation import forecast

        result = forecast(store, ForecastRequest(**json.loads(args.request)))
    elif args.command == "championship":
        from .simulation import simulate_season

        result = simulate_season(store, SeasonRequest(**json.loads(args.request)))
    elif args.command == "reproduce":
        from .simulation import reproduce

        result = reproduce(store, args.forecast_id)
    elif args.command == "record":
        from .recorder import capture

        result = capture(args.output, args.minutes, args.positions)
    elif args.command == "normalize-recording":
        from .recorder import normalize

        envelopes = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
        number_map = {
            str(r["number"]): r["driver_id"] for r in store.table("entries") if r["event_id"] == args.event
        }
        messages = normalize(envelopes, args.event, args.start, number_map)
        Path(args.output).write_text(canonical(messages))
        result = {"messages": len(messages), "path": str(Path(args.output).resolve())}
    else:
        result = jobs.submit(store, args.command)
    print(json.dumps(json.loads(canonical(result)), indent=2))


if __name__ == "__main__":
    main()
