"""Recover complete calendar/GP/qualifying pages already retained by an interrupted import.

No synthetic provider rows are created. Optional endpoints remain explicitly unavailable.
Uses the same normalizer as online ingestion; only its page transport is substituted.
"""

import argparse
import json
from collections import defaultdict
from urllib.parse import urlsplit, parse_qs
from f1engine import sources
from f1engine.store import Store
from f1engine.contracts import canonical


def recover(root, season):
    store = Store(root)
    pages = defaultdict(dict)
    with store.db() as db:
        for record in db.execute("SELECT id,body FROM raw_records ORDER BY rowid"):
            body = json.loads(record["body"])
            url = urlsplit(body["url"])
            if body["source"] != "jolpica" or f"/f1/{season}" not in url.path:
                continue
            suffix = url.path.split("/f1/", 1)[1].removesuffix(".json").rstrip("/")
            offset = int(parse_qs(url.query).get("offset", ["0"])[0])
            pages[suffix][offset] = (body["payload"], record["id"])
    for suffix in (str(season), f"{season}/results", f"{season}/qualifying"):
        batch = pages.get(suffix, {})
        if not batch:
            raise ValueError(f"No retained {suffix} pages")
        payload, _ = batch[0]
        total = int(payload["MRData"]["total"])
        limit = int(payload["MRData"]["limit"])
        if not all(i in batch for i in range(0, total, limit)):
            raise ValueError(f"Incomplete retained {suffix} pages")
    missing = []

    def retained(client, suffix):
        batch = pages.get(suffix, {})
        if not batch:
            missing.append(suffix)
            return
        first = next(iter(batch.values()))[0]["MRData"]
        total = int(first["total"])
        limit = int(first["limit"])
        if not all(i in batch for i in range(0, total, limit)):
            missing.append(suffix)
            return
        for _, (payload, source) in sorted(batch.items()):
            mr = payload["MRData"]
            yield mr.get("RaceTable", mr.get("StandingsTable", {})), source

    original = sources._race_payloads
    try:
        sources._race_payloads = retained
        version = sources.sync_season(store, season)
    finally:
        sources._race_payloads = original
    evidence = {
        "kind": "source_coverage",
        "season": season,
        "data_version": version,
        "status": "complete_primary_records_only",
        "retained_optional_gaps": missing,
        "primary_complete": ["calendar", "race_results", "qualifying_results"],
        "note": "Recovered retained public response pages after optional endpoint failure. No sprint/standings coverage is inferred. Pre-weekend race evaluation does not use those missing endpoint records.",
    }
    store.add_evidence(evidence)
    return evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("season", type=int)
    p.add_argument("--home", default=".f1engine")
    args = p.parse_args()
    print(canonical(recover(args.home, args.season)))
