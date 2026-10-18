"""Versioned Parquet snapshots and transactional append-only evidence ledger."""

from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import uuid
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq
from .contracts import Settings, Unavailable, canonical, digest, now_utc, timestamp

TABLE_KEYS = {
    "events": ("event_id",),
    "entries": ("event_id", "driver_id"),
    "results": ("event_id", "session", "driver_id"),
    "sessions": ("event_id", "session"),
    "laps": ("event_id", "session", "driver_id", "lap"),
    "standings": ("event_id", "driver_id"),
    "constructor_standings": ("event_id", "team_id"),
    "weather": ("event_id", "issued_at"),
    "climate": ("event_id", "history_end"),
}


class Store:
    def __init__(self, root: str | Path | None = None):
        self.settings = Settings(root)
        self.root = self.settings.root
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("raw", "datasets", "models"):
            (self.root / directory).mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS raw_records(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, body TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pointers(name TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS forecasts(id TEXT PRIMARY KEY, body TEXT NOT NULL, created_at TEXT NOT NULL, previous_hash TEXT, hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, body TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts BEGIN SELECT RAISE(ABORT, 'Forecast ledger is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts BEGIN SELECT RAISE(ABORT, 'Forecast ledger is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT, 'Evidence is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS evidence_no_delete BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT, 'Evidence is append-only'); END;
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "metadata.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @contextmanager
    def writer(self):
        with (self.root / ".writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def raw(self, source: str, url: str, payload: Any, retrieved_at: str | None = None) -> str:
        record = {"source": source, "url": url, "retrieved_at": retrieved_at or now_utc(), "payload": payload}
        identifier = digest(record)
        path = self.root / "raw" / f"{identifier}.json"
        if not path.exists():
            temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_text(canonical(record))
            os.replace(temporary, path)
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO raw_records VALUES (?,?)", (identifier, canonical(record)))
        return identifier

    def current_version(self) -> str | None:
        with self.db() as db:
            row = db.execute("SELECT value FROM pointers WHERE name='dataset'").fetchone()
        return row["value"] if row else None

    def manifest(self, version: str | None = None) -> dict:
        version = version or self.current_version()
        with self.db() as db:
            row = db.execute("SELECT body FROM datasets WHERE id=?", (version,)).fetchone()
        if not row:
            raise Unavailable("No dataset is available. Sync data or load the illustrative demo first.")
        body = json.loads(row["body"])
        path = self.root / "datasets" / version / "manifest.json"
        if body.get("id") != version or not path.exists() or path.read_text() != row["body"]:
            raise Unavailable("Dataset manifest integrity failure")
        return body

    def source(self, identifier: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT body FROM raw_records WHERE id=?", (identifier,)).fetchone()
        path = self.root / "raw" / f"{identifier}.json"
        if (
            not row
            or hashlib.sha256(row["body"].encode()).hexdigest() != identifier
            or not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != identifier
        ):
            raise Unavailable("Raw source integrity check failed")
        return json.loads(row["body"])

    def verify_sources(self, identifiers):
        return [
            {"id": identifier, **{k: v for k, v in self.source(identifier).items() if k != "payload"}}
            for identifier in sorted(set(identifiers))
        ]

    def verify_dataset(self, version=None):
        manifest = self.manifest(version)
        tables = {name: self.table(name, manifest["id"], include_revisions=True) for name in TABLE_KEYS}
        if digest({"tables": tables, "illustrative": manifest["illustrative"]}) != manifest["id"]:
            raise Unavailable("Dataset content identity does not match its version")
        sources = self.verify_sources(manifest["sources"])
        return {
            "valid": True,
            "data_version": manifest["id"],
            "records": sum(len(rows) for rows in tables.values()),
            "sources": len(sources),
        }

    def table(
        self, name: str, version: str | None = None, *, as_of=None, include_revisions=False
    ) -> list[dict]:
        if name not in TABLE_KEYS:
            raise ValueError("Unknown table")
        manifest = self.manifest(version)
        rows = []
        for part in manifest["tables"].get(name, []):
            path = self.root / "datasets" / manifest["id"] / part["path"]
            if (
                not path.resolve().is_relative_to((self.root / "datasets" / manifest["id"]).resolve())
                or not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != part["sha256"]
            ):
                raise Unavailable(f"Dataset integrity failure: {name}")
            # ParquetFile avoids accidental Hive type coercion from partition directory names.
            rows.extend(json.loads(r["record"]) for r in pq.ParquetFile(path).read().to_pylist())
        if include_revisions:
            return rows
        cutoff = timestamp(as_of) if as_of is not None else None
        selected = []
        for row in rows:
            current = {k: v for k, v in row.items() if k != "_revisions"}
            revisions = [*row.get("_revisions", []), current]
            visible = next(
                (r for r in reversed(revisions) if cutoff is None or timestamp(r["available_at"]) <= cutoff),
                None,
            )
            if visible is not None:
                selected.append(visible)
        return selected

    def publish(self, changes: dict[str, list[dict]], *, illustrative: bool = False) -> str:
        """Merge stable keys. Empty updates are no-ops, not destructive replacements."""
        unknown = set(changes) - TABLE_KEYS.keys()
        if unknown:
            raise ValueError(f"Unknown tables: {unknown}")
        with self.writer():
            previous = self.current_version()
            if previous and self.manifest(previous)["illustrative"] != illustrative:
                raise ValueError("Illustrative and observed datasets must use separate storage roots")
            merged = {}
            resolutions = []
            for name, keys in TABLE_KEYS.items():
                existing = self.table(name, previous, include_revisions=True) if previous else []
                mapping = {tuple(r[k] for k in keys): r for r in existing}
                for row in changes.get(name, []):
                    if "_revisions" in row:
                        raise ValueError("Record revision history is managed by the store")
                    if any(row.get(k) is None for k in keys):
                        raise ValueError(f"Missing stable key for {name}")
                    for required in ("available_at", "source_id", "availability_basis"):
                        if required not in row:
                            raise ValueError(f"{name} missing provenance field {required}")
                    timestamp(row["available_at"])
                    with self.db() as db:
                        if not db.execute(
                            "SELECT 1 FROM raw_records WHERE id=?", (row["source_id"],)
                        ).fetchone():
                            raise ValueError("Unknown source record")
                    key = tuple(row[k] for k in keys)
                    # Compare content, not changing retrieval/provenance wrappers.
                    old = mapping.get(key)
                    metadata = {"source_id", "available_at", "availability_basis", "_revisions"}
                    if old and {k: v for k, v in old.items() if k not in metadata} != {
                        k: v for k, v in row.items() if k not in metadata
                    }:
                        with self.db() as db:
                            old_source = json.loads(
                                db.execute(
                                    "SELECT body FROM raw_records WHERE id=?", (old["source_id"],)
                                ).fetchone()["body"]
                            )["source"]
                            new_source = json.loads(
                                db.execute(
                                    "SELECT body FROM raw_records WHERE id=?", (row["source_id"],)
                                ).fetchone()["body"]
                            )["source"]
                        priority = {"jolpica": 3, "fastf1": 2, "openf1": 1}
                        accept = old_source == new_source or priority.get(new_source, 0) > priority.get(
                            old_source, 0
                        )
                        resolutions.append(
                            {
                                "table": name,
                                "key": list(key),
                                "previous_source": old["source_id"],
                                "incoming_source": row["source_id"],
                                "accepted": accept,
                                "rule": "same-source revision; otherwise Jolpica > FastF1 > OpenF1; unknown conflicts retain existing source",
                                "old_record_hash": digest(old),
                                "new_record_hash": digest(row),
                            }
                        )
                        if not accept:
                            continue
                    if old:
                        prior = {k: v for k, v in old.items() if k != "_revisions"}
                        if prior == row:
                            continue
                        row = {**row, "_revisions": [*old.get("_revisions", []), prior]}
                    mapping[key] = row
                merged[name] = sorted(mapping.values(), key=lambda r: tuple(str(r[k]) for k in keys))
            if resolutions:
                self.add_evidence(
                    {"kind": "source_resolution", "parent_data_version": previous, "resolutions": resolutions}
                )
            identifier = digest({"tables": merged, "illustrative": illustrative})
            with self.db() as db:
                if db.execute("SELECT 1 FROM datasets WHERE id=?", (identifier,)).fetchone():
                    db.execute("INSERT OR REPLACE INTO pointers VALUES ('dataset',?)", (identifier,))
                    return identifier
            if not any(merged.values()):
                raise ValueError("Cannot publish an empty dataset")
            stage = self.root / "datasets" / f".staging-{uuid.uuid4().hex}"
            stage.mkdir()
            manifest = {
                "id": identifier,
                "parent": previous,
                "created_at": now_utc(),
                "illustrative": illustrative,
                "record_revision_schema": 1,
                "tables": {},
                "sources": sorted(
                    {
                        revision["source_id"]
                        for rows in merged.values()
                        for r in rows
                        for revision in [r, *r.get("_revisions", [])]
                    }
                ),
            }
            try:
                for name, rows in merged.items():
                    manifest["tables"][name] = []
                    seasons = sorted({str(r["event_id"]).split("-")[0] for r in rows})
                    for season in seasons:
                        subset = [r for r in rows if str(r["event_id"]).startswith(season + "-")]
                        relative = f"{name}/season={season}/part.parquet"
                        path = stage / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        table = pa.Table.from_pylist([{"record": canonical(r)} for r in subset])
                        pq.write_table(table, path, compression="zstd")
                        manifest["tables"][name].append(
                            {
                                "path": relative,
                                "rows": len(subset),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        )
                (stage / "manifest.json").write_text(canonical(manifest))
                target = self.root / "datasets" / identifier
                if not target.exists():
                    os.replace(stage, target)
                with self.db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute(
                        "INSERT INTO datasets VALUES (?,?,?)",
                        (identifier, canonical(manifest), manifest["created_at"]),
                    )
                    db.execute("INSERT OR REPLACE INTO pointers VALUES ('dataset',?)", (identifier,))
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            return identifier

    def save_forecast(self, body: dict) -> dict:
        record = dict(body)
        record["id"] = uuid.uuid4().hex
        record["created_at"] = now_utc()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            revision = record.get("request", {}).get("revision_of")
            if revision and not db.execute("SELECT 1 FROM forecasts WHERE id=?", (revision,)).fetchone():
                raise ValueError("Revision refers to an unknown forecast")
            if revision:
                old = json.loads(
                    db.execute("SELECT body FROM forecasts WHERE id=?", (revision,)).fetchone()["body"]
                )
                identity = "season" if record.get("kind") == "championship" else "event_id"
                if old.get("kind") != record.get("kind") or old.get("request", {}).get(
                    identity
                ) != record.get("request", {}).get(identity):
                    raise ValueError("A revision must refer to the same forecast event/season and kind")
            previous = db.execute("SELECT hash FROM forecasts ORDER BY rowid DESC LIMIT 1").fetchone()
            previous_hash = previous["hash"] if previous else None
            record_hash = digest({"previous_hash": previous_hash, "body": record})
            db.execute(
                "INSERT INTO forecasts VALUES (?,?,?,?,?)",
                (record["id"], canonical(record), record["created_at"], previous_hash, record_hash),
            )
        return {**record, "ledger_hash": record_hash}

    def forecasts(self, limit: int = 100) -> list[dict]:
        with self.db() as db:
            rows = db.execute(
                "SELECT body, hash FROM forecasts ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{**json.loads(r["body"]), "ledger_hash": r["hash"]} for r in rows]

    def forecast(self, identifier: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT body,hash FROM forecasts WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise Unavailable("Unknown forecast")
        return {**json.loads(row["body"]), "ledger_hash": row["hash"]}

    def verify_ledger(self) -> dict:
        with self.db() as db:
            rows = db.execute("SELECT * FROM forecasts ORDER BY rowid").fetchall()
        previous = None
        for row in rows:
            if row["previous_hash"] != previous or row["hash"] != digest(
                {"previous_hash": previous, "body": json.loads(row["body"])}
            ):
                return {"valid": False, "records": len(rows), "failed_id": row["id"]}
            previous = row["hash"]
        return {
            "valid": True,
            "records": len(rows),
            "head": previous,
            "limitation": "Local integrity check, not an external timestamp attestation.",
        }

    def add_evidence(self, body: dict) -> str:
        record = {**body, "created_at": now_utc()}
        identifier = digest(record)
        with self.db() as db:
            db.execute(
                "INSERT OR IGNORE INTO evidence VALUES (?,?,?)",
                (identifier, canonical(record), record["created_at"]),
            )
        return identifier

    def evidence(self) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT id,body FROM evidence ORDER BY rowid DESC").fetchall()
        return [{"id": r["id"], **json.loads(r["body"])} for r in rows]
