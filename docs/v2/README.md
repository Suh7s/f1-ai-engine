# F1 Engine V2

A local race-control application built around inspectable forecasts, immutable inputs and separately reported evidence. V2 is a research release: running software and passing regression checks do not establish superior predictive accuracy.

The replacement lives in `v2/f1engine/` and `web/`. Unused V1 source, dashboards, tests, scripts and dependencies have been removed from the active tree. The [original implementation](https://github.com/Suh7s/f1-aii-engine/tree/1ecac8c) remains recoverable in Git history. V2 never loads legacy trained models.

## Install and launch

Python **3.13** is required. From the repository root:

```sh
./scripts/setup-v2.sh
. .venv-v2/bin/activate
f1engine app
```

Open <http://127.0.0.1:8765>. The packaged Pit Wall UI works without Node or external font requests at runtime. Its design uses warm paper, racing red and serif headings; the season calendar selects the same event as the forecast controls. The initial empty installation makes no network requests. Load one season at a time from Evidence, or run `f1engine sync 2024`.

The completely offline, fictional demonstration uses a separate data directory:

```sh
f1engine app --demo
```

On macOS, experimental XGBoost training needs the OpenMP runtime (`brew install libomp`). Missing OpenMP does not prevent baseline forecasting or startup. `f1engine doctor` reports availability. The tested platform is macOS ARM64; the worker uses POSIX locks and signals. Windows is not supported by this release.

`requirements-v2.lock` pins the tested Python environment, including developer tools. `web/package-lock.json` pins the frontend. Build-system dependencies are pinned in `pyproject.toml`. Use the setup script for the locked installation. `pip install .` alone respects version ranges and is not the reproducible installation route.

To change the frontend, use Node 22.12+ and run `npm ci && npm run build` inside `web/`. Built static assets are bundled with the Python wheel.

## Workspaces

- **Race:** full-field position distributions; win, podium, points and retirement probabilities; head-to-heads; driver contributions; explicit scenarios; saved forecast comparisons and source records.
- **Weekend:** comparable green-lap practice pace, qualifying rank, sprint inputs, forecast/climate labels and feature coverage. Unknown inputs remain missing. Qualifying rank is explicitly a starting-grid proxy until penalties are verified.
- **Championship:** official starting points, simulated GP/sprint points, driver and constructor title distributions, season-persistent team uncertainty and countback. Unknown future entries use the roster known at the cutoff.
- **Live/replay:** arrival-ordered observations, corrections, timing tower, tyres, gaps and remaining-race forecasts. Coordinates are shown only when supplied; no invented track location. This is an unvalidated live-model baseline.
- **Evidence:** separate evidence classes, model reports, reliability curves, promotion decisions, immutable forecast records, data status and recoverable background jobs.

Every forecast stores information cutoff, stage, exact input dataset, feature hash, model and rule versions, source IDs, seed, simulation count, code hash, coverage, assumptions and scenario overrides. Source records retain retrieval times and URLs. Raw JSON and Parquet partitions have content hashes. The SQLite ledger rejects updates/deletes and links each record to the previous hash. This is local integrity evidence, not external timestamp certification.

## Useful commands

```sh
f1engine sync 2024
f1engine standings 2024-17
f1engine jobs
f1engine cancel JOB_ID
f1engine resume JOB_ID
f1engine verify
f1engine reproduce FORECAST_ID
f1engine experiment --parameters '{"stage":"pre_weekend","candidates":12}'
f1engine collect --parameters '{"freeze_id":"FREEZE_ID","event_id":"2026-18"}'
f1engine prospective --parameters '{"freeze_id":"FREEZE_ID"}'
f1engine repair_identities
f1engine score
f1engine benchmark
```

Experiments are subprocess jobs with four training threads, at most twelve parameter candidates, one active expensive job per data directory and a 45-minute execution budget. Cancellation and time limits preserve completed candidate checkpoints. Downloads are bounded by season, with atomic publication only after the season request set completes. Failed downloads preserve previously published data; raw downloaded responses remain available. A cancelled season import currently reissues its requests on resume. Queued jobs are serialized, but FIFO order is not guaranteed.

A candidate freeze identifies the selected model and its complete selection procedure. The first 2025 holdout has already been inspected and is now retired for revised candidates, including fresh stores. New candidates register a prospective procedure. Collect paired candidate/baseline forecasts before events and evaluate the frozen first-15-registered-event cohort only once all outcomes are available. Later outcome corrections append scoring revisions. No automatic recurring collector is configured. The original completed-2026 report remains a separately labelled regulation-transfer evaluation of its original model.

A challenger is promoted only when the paired race-bootstrap 95% interval for normalized RPS improvement excludes zero in the favorable direction and winner log loss worsens by no more than 2%. At least fifteen paired races are required. Stage-specific model pointers prevent a post-qualifying model from silently serving pre-weekend forecasts. Development selects the strongest named baseline for that stage when no challenger is promoted. The candidate set includes a chronological mixture of calibrated rating, ridge and XGBoost components; ensemble weights use disjoint calibration predictions. Evaluation, not complexity, determines selection.

## API and ingestion

Interactive local API documentation: <http://127.0.0.1:8765/docs>. All endpoints use `/api/v1`. The server binds to loopback, restricts Host values and rejects cross-origin writes. This release is not configured for public hosting.

The shared feature request includes `event_id`, `stage`, `as_of_utc`, optional `data_version`, `model_version`, `scenario`, `simulations` and `seed`. Stages are `pre_weekend`, `post_practice`, `post_qualifying`, `post_sprint` and `live`. Use explicit UTC timestamps. Live forecasts require a validated replay state.

Submit additional source jobs through `POST /api/v1/jobs`:

```json
{"kind":"fastf1","parameters":{"event_id":"2024-24","session":"FP2"}}
```

Other job kinds: `sync_season`, `sync_standings`, `weather`, `climate`, `openf1`, `forecast`, `championship`, `experiment`, `holdout` (retired for revised candidates), `collect`, `prospective`, `replay_evaluation`, `championship_evaluation`, `repair_identities`, `score`, `benchmark`. The SSE endpoint `/api/v1/jobs/stream` delivers observed progress. OpenF1 supplementary jobs require an event and session key and refuse active-session paid access. FastF1 detailed sessions begin in 2018 and never request high-frequency car telemetry by default.

Jolpica supplies calendars, entries reconstructed from results, race/qualifying/sprint results and official standings. Historical publication times are generally unavailable: the recorded session-end and entry-availability times are disclosed reconstructions, not archival proof of publication. Season ingestion captures the latest standings. Use `f1engine standings YEAR-ROUND` or a `sync_standings` job with `event_id` to acquire one historical round. Driver and constructor snapshots publish together only after both validate; other rounds remain intact. A historical championship cutoff without its matching official standings snapshot is unavailable. Source conflicts follow an explicit recorded precedence (Jolpica, FastF1, OpenF1) and same-source revisions retain earlier snapshots.

FastF1 lap times use absolute lap dates when supplied. Otherwise the adapter subtracts the session clock origin before anchoring to the scheduled start; it labels this reconstruction and retains the timing references. Actual start delays and publication times remain unverified in that fallback. A stream-relative lap time must never simply be added to the scheduled start.

Open-Meteo forecasts are captured when retrieved and never replaced with observed race weather. A separate climate job computes the fraction of days with at least 1 mm precipitation in the event month across five prior years. That is an explicitly labelled daily climate prior, not an event-hour forecast. Missing weather/climate remains missing; learned imputation uses training data only.

The weather adapter labels local `captured_at`, leaves unverified `provider_issued_at` missing, and retains a legacy `issued_at` alias for compatibility. The alias is capture time, not the provider's internal model issuance time. Source retrieval timestamps establish when this application obtained the forecast. No historical weather accuracy claim has been made from these records.

Both race and championship requests accept `revision_of`. Corrections must identify an existing forecast for the same event or season and kind. Restart the application after editing its Python source: new processes reject evidence generation if their source files change while they are running. To reproduce evidence from an older code version, restore the recorded source checkpoint and restart that process.

## Free recorder and replay

```sh
f1engine record recording.ndjson --minutes 45
f1engine normalize-recording recording.ndjson replay.json --event 2026-19 --start 2026-10-25T19:00:00Z
```

The example event/time must be replaced with the selected session's verified values. `record` explicitly disables authentication and requests timing topics only. `--positions` opts into position messages. Free coverage may be partial or absent; no F1 subscription or OpenF1 paid access is required. The adapter retains local receipt timestamps. Legacy recordings without receipt timestamps cannot be promoted to arrival-time evidence. Initial untimestamped snapshots are ignored during normalization.

Import normalized JSON in Live/replay. Replays include only observations both received and occurring by the selected cutoff. Corrections apply when received; conflicting equal-version messages fail. Reconnection can append new observations. Stale or incomplete full-field timing prevents a new forecast. The replay baseline shrinks observed green-lap pace toward a disclosed pre-race prior; future pit strategy, neutralization and weather transitions are not yet calibrated. Do not describe its outputs as replay-validated competitive predictions.

## Verification

```sh
pytest -q                         # offline fixtures; browser checks skipped
ruff check v2 v2_tests
F1ENGINE_E2E_URL=http://127.0.0.1:8766 pytest -m browser
```

Browser checks require a running `f1engine app --demo --port 8766` and `python -m playwright install chromium`. Default tests use no internet or private datasets. Tests cover every defect identified in the legacy audit and the additional probability, scoring, provenance, replay and publication constraints. Evidence, measurements, vulnerability results and release limitations are in [release-evidence.md](release-evidence.md).
