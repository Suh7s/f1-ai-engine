"""Durable bounded subprocess jobs. Resumes retain checkpoints and immutable inputs."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from .contracts import canonical, now_utc, Unavailable
from .store import Store

KINDS = {
    "sync_season",
    "sync_standings",
    "weather",
    "climate",
    "fastf1",
    "openf1",
    "forecast",
    "championship",
    "experiment",
    "holdout",
    "collect",
    "prospective",
    "replay_evaluation",
    "championship_evaluation",
    "repair_identities",
    "score",
    "benchmark",
}


class PauseJob(Exception):
    pass


class CancelJob(Exception):
    pass


def get_job(store, identifier):
    with store.db() as db:
        row = db.execute("SELECT body FROM jobs WHERE id=?", (identifier,)).fetchone()
    if not row:
        raise Unavailable("Unknown job")
    return json.loads(row["body"])


def jobs(store):
    with store.db() as db:
        rows = db.execute("SELECT body FROM jobs ORDER BY rowid DESC").fetchall()
    return [json.loads(r["body"]) for r in rows]


def update(store, identifier, **fields):
    with store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT body FROM jobs WHERE id=?", (identifier,)).fetchone()
        body = json.loads(row["body"])
        body.update(fields)
        body["updated_at"] = now_utc()
        db.execute("UPDATE jobs SET body=? WHERE id=?", (canonical(body), identifier))
    return body


def recover(store):
    import psutil

    for job in jobs(store):
        if job["status"] not in {"running", "queued"}:
            continue
        pid = job.get("pid")
        alive = False
        if pid:
            try:
                proc = psutil.Process(pid)
                alive = "f1engine.worker" in " ".join(proc.cmdline()) and job["id"] in proc.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not alive and time.time() - job.get("launched_epoch", 0) > 5:
            update(
                store,
                job["id"],
                status="interrupted",
                message="Worker interrupted. Resume from its last completed checkpoint.",
            )


def submit(store, kind, parameters=None, launch=True, budget_seconds=2700):
    if kind not in KINDS:
        raise ValueError("Unknown job kind")
    if not 1 <= budget_seconds <= 2700:
        raise ValueError("Job budget must be within 1–2700 seconds")
    identifier = uuid.uuid4().hex
    body = {
        "id": identifier,
        "kind": kind,
        "parameters": parameters or {},
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "budget_seconds": budget_seconds,
        "checkpoint": {},
        "cancel_requested": False,
        "attempt": 1,
        "launched_epoch": time.time(),
    }
    with store.db() as db:
        db.execute("INSERT INTO jobs VALUES (?,?)", (identifier, canonical(body)))
    if launch:
        spawn(store, identifier)
    return get_job(store, identifier)


def spawn(store, identifier):
    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[k] = "4"
    log = store.root / "jobs"
    log.mkdir(exist_ok=True)
    with (log / f"{identifier}.log").open("a") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "f1engine.worker", str(store.root), identifier],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    update(store, identifier, pid=process.pid, launched_epoch=time.time())


def cancel(store, identifier):
    job = get_job(store, identifier)
    if job["status"] in {"running", "queued"}:
        return update(
            store,
            identifier,
            cancel_requested=True,
            message="Cancellation requested; preserving completed checkpoints",
        )
    return job


def resume(store, identifier, launch=True):
    recover(store)
    job = get_job(store, identifier)
    if job["status"] not in {"paused", "interrupted", "cancelled", "failed"}:
        raise ValueError("Only paused, interrupted, cancelled or failed jobs can resume")
    update(
        store,
        identifier,
        status="queued",
        cancel_requested=False,
        error=None,
        attempt=job["attempt"] + 1,
        launched_epoch=time.time(),
    )
    if launch:
        spawn(store, identifier)
    return get_job(store, identifier)


def run(store, identifier):
    import fcntl

    # Only one expensive worker per store; queued wait consumes no experiment CPU.
    lock = (store.root / ".jobs.lock").open("a")
    while True:
        if get_job(store, identifier)["cancel_requested"]:
            update(store, identifier, status="cancelled", message="Cancelled before execution")
            lock.close()
            return
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(0.25)
    job = get_job(store, identifier)
    start = time.monotonic()
    checkpoint = job["checkpoint"]

    def alarm_handler(*_):
        raise PauseJob("45-minute execution budget reached; resume this job")

    previous = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, job["budget_seconds"])

    def check():
        if get_job(store, identifier)["cancel_requested"]:
            raise CancelJob("Cancelled by user")
        if time.monotonic() - start >= job["budget_seconds"]:
            raise PauseJob("Execution budget reached")

    def progress(fraction, message):
        check()
        update(
            store,
            identifier,
            progress=max(0, min(1, float(fraction))),
            message=message,
            checkpoint=checkpoint,
        )

    try:
        update(store, identifier, status="running", started_at=now_utc(), message="Running", pid=os.getpid())
        result = dispatch(store, job["kind"], job["parameters"], check, progress, checkpoint)
        update(
            store,
            identifier,
            status="completed",
            progress=1,
            result=result,
            checkpoint=checkpoint,
            message="Completed",
            elapsed_seconds=time.monotonic() - start,
        )
    except (PauseJob, CancelJob) as exc:
        update(
            store,
            identifier,
            status="paused" if isinstance(exc, PauseJob) else "cancelled",
            checkpoint=checkpoint,
            message=str(exc),
            elapsed_seconds=time.monotonic() - start,
        )
    except Exception as exc:
        update(
            store,
            identifier,
            status="failed",
            checkpoint=checkpoint,
            error=f"{type(exc).__name__}: {exc}",
            message="Failed; existing datasets and forecasts retained",
            elapsed_seconds=time.monotonic() - start,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        lock.close()


def dispatch(store, kind, p, check, progress, checkpoint):
    from .contracts import ForecastRequest, SeasonRequest

    if kind == "repair_identities":
        from .repairs import repair_identities

        return repair_identities(store, check)

    if kind == "replay_evaluation":
        from .replay_evaluation import evaluate_archive

        return evaluate_archive(store, p, check, progress)
    if kind == "championship_evaluation":
        from .championship_evaluation import evaluate_championship

        return evaluate_championship(store, p, check, progress)
    if kind in {"collect", "prospective"}:
        from .prospective import collect_pair, evaluate_prospective

        return collect_pair(store, p, checkpoint) if kind == "collect" else evaluate_prospective(store, p)

    if kind in {"sync_season", "sync_standings", "weather", "climate", "fastf1", "openf1"}:
        from .sources import sync_season, sync_standings, sync_weather, sync_fastf1, sync_openf1, sync_climate

        if kind == "sync_season":
            return {"data_version": sync_season(store, int(p["season"]), check, progress)}
        if kind == "sync_standings":
            return {"data_version": sync_standings(store, p["event_id"], check)}
        if kind == "climate":
            return {"data_version": sync_climate(store, p["event_id"], check)}
        if kind == "weather":
            return {"data_version": sync_weather(store, p["event_id"], check)}
        if kind == "fastf1":
            return {
                "data_version": sync_fastf1(
                    store,
                    int(p["event_id"].split("-")[0]),
                    int(p["event_id"].split("-")[1]),
                    p["session"],
                    check,
                )
            }
        return sync_openf1(store, p["event_id"], int(p["session_key"]), check)
    if kind == "forecast":
        from .simulation import forecast

        return forecast(store, ForecastRequest(**p))
    if kind == "championship":
        from .simulation import simulate_season

        return simulate_season(store, SeasonRequest(**p))
    if kind in {"experiment", "holdout", "score"}:
        from .evaluation import experiment, holdout, score_ledger

        if kind == "experiment":
            return experiment(store, p, check, progress, checkpoint)
        if kind == "holdout":
            return holdout(store, p, check, progress)
        return score_ledger(store)
    if kind == "benchmark":
        from .benchmark import benchmark

        return benchmark(store)
    raise ValueError("Unknown job")
