"""Local-only versioned API, immutable evidence access, and server-sent job updates."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from .contracts import (
    ForecastRequest,
    SeasonRequest,
    JobRequest,
    StrictModel,
    Unavailable,
    clean,
    now_utc,
    timestamp,
)
from .store import Store
from . import jobs


class ReplayRequest(StrictModel):
    event_id: str
    as_of_utc: str
    messages: list[dict] = Field(max_length=100000)
    predict: bool = False
    simulations: int = Field(default=10000, ge=100, le=100000)
    seed: int = 42


def create_app(root=None, demo=False):
    store = Store(root)
    if demo:
        from .demo import load_demo

        load_demo(store)

    @asynccontextmanager
    async def lifespan(app):
        jobs.recover(store)
        yield

    app = FastAPI(title="F1 Engine", version="2.0", lifespan=lifespan)
    app.state.store = store

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        hostname = request.url.hostname
        if hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
            return JSONResponse({"detail": "Local access only"}, status_code=403)
        origin = request.headers.get("origin")
        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and origin
            and (urlparse(origin).netloc != request.headers.get("host"))
        ):
            return JSONResponse({"detail": "Cross-origin changes are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(Unavailable)
    async def unavailable(request, exc):
        return JSONResponse({"detail": str(exc), "state": "unavailable"}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(Exception)
    async def failure(request, exc):
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}", "state": "failed"}, status_code=500)

    @app.get("/api/v1/health")
    def health():
        version = store.current_version()
        manifest = store.manifest(version) if version else None
        return {
            "status": "ready",
            "version": "2.0",
            "data_version": version,
            "dataset": manifest,
            "ledger": store.verify_ledger(),
            "illustrative": bool(manifest and manifest["illustrative"]),
            "now": now_utc(),
            "limits": {"training_threads": 4, "job_minutes": 45, "max_candidates": 12},
            "trust": "Unvalidated models remain candidates. Evidence categories are never pooled.",
        }

    @app.get("/api/v1/events")
    def events():
        return store.table("events") if store.current_version() else []

    @app.get("/api/v1/forecasts")
    def forecasts():
        return store.forecasts(300)

    @app.get("/api/v1/forecasts/{identifier}")
    def one_forecast(identifier: str):
        return store.forecast(identifier)

    @app.post("/api/v1/forecasts")
    def new_forecast(body: ForecastRequest):
        from .simulation import forecast

        return clean(forecast(store, body))

    @app.post("/api/v1/championship")
    def championship(body: SeasonRequest):
        from .simulation import simulate_season

        return clean(simulate_season(store, body))

    @app.post("/api/v1/forecasts/{identifier}/reproduce")
    def reproduce(identifier: str):
        from .simulation import reproduce

        return reproduce(store, identifier)

    @app.post("/api/v1/features")
    def features(body: ForecastRequest):
        from .features import build_features

        return clean(build_features(store, body))

    @app.get("/api/v1/evidence")
    def evidence():
        return store.evidence()

    @app.get("/api/v1/models")
    def models():
        with store.db() as db:
            rows = db.execute("SELECT id,body FROM models").fetchall()
            active = db.execute("SELECT value FROM pointers WHERE name='model:pre_weekend'").fetchone()
            stages = db.execute("SELECT name,value FROM pointers WHERE name LIKE 'model:%'").fetchall()
        return {
            "active": active["value"] if active else "baseline:rating",
            "active_by_stage": {r["name"].split(":", 1)[1]: r["value"] for r in stages},
            "candidates": [{"id": r["id"], **json.loads(r["body"])} for r in rows],
            "baselines": ["baseline:grid", "baseline:form", "baseline:rating"],
        }

    @app.get("/api/v1/sources/{identifier}")
    def source(identifier: str):
        with store.db() as db:
            row = db.execute("SELECT body FROM raw_records WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise Unavailable("Unknown source")
        from .contracts import digest

        body = json.loads(row["body"])
        path = store.root / "raw" / f"{identifier}.json"
        if digest(body) != identifier or not path.exists() or json.loads(path.read_text()) != body:
            raise Unavailable("Raw source integrity check failed")
        return {"id": identifier, **body}

    @app.get("/api/v1/jobs")
    def list_jobs():
        jobs.recover(store)
        return jobs.jobs(store)

    @app.post("/api/v1/jobs", status_code=202)
    def new_job(body: JobRequest):
        return jobs.submit(store, body.kind, body.parameters)

    @app.post("/api/v1/jobs/{identifier}/cancel")
    def cancel(identifier: str):
        return jobs.cancel(store, identifier)

    @app.post("/api/v1/jobs/{identifier}/resume")
    def resume(identifier: str):
        return jobs.resume(store, identifier)

    @app.get("/api/v1/jobs/stream")
    async def stream(request: Request):
        async def changes():
            previous = None
            while not await request.is_disconnected():
                data = json.dumps(jobs.jobs(store))
                if data != previous:
                    yield f"data: {data}\n\n"
                    previous = data
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(
            changes(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/api/v1/replay/demo")
    def replay_demo():
        if not store.manifest()["illustrative"]:
            raise Unavailable("Illustrative messages are restricted to the demo store")
        from .demo import replay_fixture

        return replay_fixture()

    @app.post("/api/v1/replay")
    def replay_request(body: ReplayRequest):
        from .replay import replay
        from .simulation import forecast

        state = replay(body.messages, body.event_id, body.as_of_utc)
        # Save the received envelope independently of any forecast; raw content remains immutable.
        source_id = store.raw("replay_import", "local://arrival-ordered-messages", body.messages)
        state["source_id"] = source_id
        if body.predict:
            result = forecast(
                store,
                ForecastRequest(
                    event_id=body.event_id,
                    stage="live",
                    as_of_utc=body.as_of_utc,
                    seed=body.seed,
                    simulations=body.simulations,
                ),
                live_state=state,
            )
            return {"state": state, "forecast": clean(result)}
        return {"state": state}

    static = Path(__file__).parent / "static"
    if (static / "assets").exists():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/")
    def home():
        if (static / "index.html").exists():
            return FileResponse(static / "index.html")
        return JSONResponse(
            {"detail": "UI build missing. Run npm ci and npm run build in web/. API is available at /docs."},
            status_code=503,
        )

    return app
