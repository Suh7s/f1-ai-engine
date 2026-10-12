# Dependency exception and local operating boundary

The recorded audit is in `dependency-audit-python.json`; the frontend audit is in `dependency-audit-web.json`.

The initial Python audit identified vulnerabilities in the environment's pip and msgpack. Pip was upgraded to 26.2.1. The remaining advisory, **PYSEC-2026-3625**, affects msgpack 1.1.2: reusing a streaming `Unpacker` after a parsing error may cause an out-of-bounds read/process crash. The public audit returns two entries for that same advisory; these are not two distinct issues.

FastF1 3.8.3 depends on signalrcore 1.0.2, which pins `msgpack==1.1.2`. Updating msgpack alone breaks the declared dependency graph. The compatible pinned environment is retained and `pip check` passes. This is an explicit documented exception, not a clean vulnerability claim.

Exposure assessment: F1 Engine does not instantiate a MessagePack `Unpacker`; its replay import is JSON validated through Pydantic. The inspected signalrcore MessagePack implementation uses one-shot `msgpack.unpackb` and `msgpack.packb`, not streaming `Unpacker` reuse. The recorder is optional and only starts on an explicit command. This reduces the known advisory's applicability to the implemented path, but does not establish that the dependency is vulnerability-free. Upgrade the upstream recorder dependency and rerun compatibility/audit checks when its pin is fixed. Do not introduce a reusable MessagePack streaming unpacker while this exception exists.

The service binds to 127.0.0.1, checks local Host names and rejects cross-origin writes. It has no remote-user authentication and is not intended for public hosting. Raw data and local SQLite databases are stored under the selected `--home` directory. No secrets or paid feed credentials are required. JSON model artifacts are validated before use; legacy pickle artifacts are never loaded.
