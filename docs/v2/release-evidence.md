# F1 Engine V2 — current release evidence

Recorded **18 October 2026** on `codex/f1-engine-v2`. This checkpoint ships the requested **Suhas’s Pit Wall** interface and a repaired, revalidated forecasting engine. It is a usable local research application, not a proven best-in-world predictor. All five milestones have runnable capabilities; some predictive acceptance gates remain open.

## Runnable artifact

Follow the [installation guide](README.md), then run `f1engine app`. The paper, black and racing-red design follows the supplied `F1.html`: large Playfair headings, a race panel, a selectable calendar, compact workspace navigation and matching charts. Race, Weekend, Championship, Live/replay and Evidence retain their functionality. The supplied HTML’s sample results, news and “live” claims were not adopted as facts.

Fonts are bundled locally with their OFL licenses. There are no runtime Google Fonts requests. Calendar numbers identify imported observations; countdowns identify derived calculations. Offline fixtures are visibly illustrative. Mobile layouts, keyboard navigation, modal focus return, reduced motion and failed-request retention were checked. Screenshots: [desktop](pit-wall-desktop.png), [forecast](pit-wall-forecast.png), [mobile](pit-wall-mobile.png), [championship](pit-wall-championship.png), [weekend](pit-wall-weekend.png), [replay](pit-wall-live.png), [evidence](pit-wall-evidence.png). Real-data screenshots show saved estimates, not verified outcomes.

Python source hash: `f66851c119c59115c27b1f1ec8818a032780fd180e6d2624f1f9105420113f86`. Feature version: `pit-v2.4`. Scoring: `full-field-rps-v2.1`. Corrected dataset: `a9a95521c502f9829536b250635e8f7a5a632a3088db92d91ce2834c261ed7c7`.

The [12 October release report](release-evidence-2026-10-12.md) preserves the previous checkpoint and its first frozen holdout. It is historical evidence, not the current feature/model validation. The later [12 October experimental runs](validation-2026-10-12/README.md) are superseded by the corrected reruns below.

## Verification

| Check | Result | Evidence |
|---|---|---|
| Offline regression suite | **104 passed**, seven opt-in browser tests skipped | [JUnit](validation-2026-10-18/offline-tests.xml), [output](validation-2026-10-18/offline-tests.txt) |
| Browser workflows | **7 passed** | [JUnit](validation-2026-10-18/browser-tests.xml); all five workspaces, reproduction, failed-request retention, mobile width, local fonts, calendar selection and keyboard modal behavior |
| Frontend compilation | Passed | [Build output](validation-2026-10-18/frontend-build.txt); Plotly remains a separately loaded large chunk |
| Python source check / dependency compatibility | Passed | Ruff and `pip check`; installed wheel also passed dependency checks |
| Fresh installation | Passed | [Record](validation-2026-10-18/clean-install.json): separate environment outside the checkout, socket startup, packaged assets/fonts/licenses, offline forecast and exact reproduction |
| Immutable ledger and dataset integrity | Passed | [Content-hash verification](validation-2026-10-18/integrity.json) |
| Prospective race reproduction | Exact prediction-hash matches | [Candidate and baseline reproduction](validation-2026-10-18/reproduction.json) |
| Frontend security audit | Zero reported findings | [Current scan](validation-2026-10-18/npm-audit.json) |
| Python security audit | One distinct documented advisory, listed twice | [Current scan](validation-2026-10-18/python-audit.json), [exception](SECURITY.md) |

Default tests use neither internet nor private datasets. Deprecation warnings from the Starlette test client and NumPy/Pandas duration construction remain; they did not fail tests. Exact reproduction is established in the recorded local numerical environment, not across every operating system.

## Data repair and trust improvements

Historical event car numbers now come from event-specific result records. Jolpica’s mutable current `Driver.permanentNumber` had incorrectly assigned some historical laps. The [append-only repair](validation-2026-10-18/identity-repair-job.json) corrected **419 car numbers**, revised entry metadata, rebuilt **2,764 lap rows** across three retained OpenF1 bundles and explicitly invalidated one obsolete identity key. Earlier datasets and forecasts remain intact. Duplicate car-number maps now fail validation.

The repaired Monza FP2 snapshot has comparable-lap features for **20 of 20 entrants**, with pit activity, race-control states and lap deletions filtered; [coverage record](validation-2026-10-18/practice-coverage.json). This single session does **not** establish predictive benefit. Unknown fuel and publication timing remain limitations.

Point-in-time table revisions, manifest/Parquet/raw-response hash checks, model publication validation, chronology guards, finisher-only performance targets, separate retirement calibration, scoring revisions and forecast source records have regression coverage. Weather captures distinguish local retrieval from unknown provider issuance time. Historical observed race weather does not replace an issued forecast.

## Development comparisons — not held-out or prospective accuracy

Each experiment compared **12 candidates**, used four training threads, persisted resumable checkpoints and stayed within its separate 45-minute budget. Families included three calibrated probability baselines, regularized strength models, XGBoost rankings and a chronological ensemble. All calibration and ensemble weighting use earlier disjoint windows. Development folds end in 2022, 2023 and 2024.

| Stage | Evaluated races | Selected candidate | Baseline RPS | Candidate RPS | Paired RPS difference, 95% interval | Winner log loss: baseline → candidate |
|---|---:|---|---:|---:|---|---|
| Pre-weekend | 68 | Calibrated rating baseline | 0.12482348 | 0.12470460 | −0.00011887 [−0.00085176, +0.00063969] | 1.75604 → 1.74379 |
| Post-qualifying | 64 | XGBoost depth 2, learning rate 0.03 | 0.11689628 | 0.11443319 | −0.00246309 [−0.00413241, −0.00069535] | 1.64669 → 1.47838 |

The pre-weekend improvement interval crosses zero. The post-qualifying candidate clears the numerical gate **on development data used for selection**. Neither result justifies learned-model promotion. The stage baselines remain active until an eligible prospective comparison passes. Reports contain full metrics, calibration curves, named competitors and breakdowns: [pre-weekend](validation-2026-10-18/pre-weekend-development.json), [post-qualifying](validation-2026-10-18/post-qualifying-development.json). Frozen procedures: [pre-weekend registration](validation-2026-10-18/pre-weekend-development-freeze.json), [post-qualifying registration](validation-2026-10-18/post-qualifying-development-freeze.json).

Circuit and available qualifying-feature ablations ran. Practice/telemetry, sprint and weather benefits remain unverified where training/calibration/evaluation coverage is insufficient. An empty feature group cannot be advertised as an unsuccessful predictive feature. The ensemble was evaluated, not automatically selected or promoted.

The first 2025 holdout at source checkpoint `6ba9a08` remains separately documented in the archived report. Its promotion gate failed. **2025 is now retired as an untouched holdout for revised candidates**, including in a fresh metadata store. The old 2026 regulation-transfer report is a separate, older-model evaluation; current candidates have not earned a new transfer-performance claim.

## Prospective collection

A [registered candidate/baseline pair](validation-2026-10-18/prospective-pair.json) for the 2026 United States GP was saved on 18 October before the imported 25 October start, with the same cutoff, dataset, seed and 10,000 simulations. The [candidate](validation-2026-10-18/prospective-candidate.json) and [baseline](validation-2026-10-18/prospective-baseline.json) both reproduce exactly. No outcome is scored yet. Timestamps are local ledger evidence, not independent timestamp certification.

The frozen policy evaluates the first 15 registered events once every outcome is available. Interim peeking cannot promote a model. This is a registered-event cohort, not a guarantee that every upcoming scheduled event will be collected. Selective registration is a remaining limitation; no autonomous recurring collection is configured. Later outcome corrections append scoring revisions. Scorecards separate evidence class, stage, model/code and scoring version.

## Replay, championship and performance

The [Spain 2024 replay](validation-2026-10-18/replay-spain.json) evaluated **21 cutoffs**, with deterministic output and prefix invariance. The [Monza 2024 replay](validation-2026-10-18/replay-monza.json) evaluated **three cutoffs** and withheld later forecasts when driver timing became stale. Missing retirement messages were not replaced with knowledge of final results. These are historical reconstructions with an explicit one-second arrival-latency assumption, not captured live-feed evidence. Cutoffs within a race are correlated. Future pit strategy, penalties, neutralization and weather transitions remain unmodeled; live forecasting readiness is not established.

[Championship reconstruction](validation-2026-10-18/championship-2024.json) evaluates two official 2024 standings cutoffs against the final outcome, with driver and constructor metrics. That is **one independent season**, not two independent accuracy samples. Later substitutions can prevent comparable full-field driver scoring. Persistent team-strength, qualifying-noise and simulated-form assumptions are still provisional; this evaluation does not calibrate them.

[Performance measurement](validation-2026-10-18/performance.json) used a cached illustrative 20-driver fixture with 22 remaining rounds, 10,000 samples and three repetitions after warm-up:

| Workload | Slowest measured run | Required target |
|---|---:|---:|
| Race | **0.179 s** | <10 s |
| Season | **4.611 s** | <30 s |

Both targets passed on this macOS ARM64 machine. These are runtime measurements, not accuracy evidence.

## Remaining milestone gates

| Milestone | Current capability | Still open |
|---|---|---|
| Foundation | Clean installation, offline fixtures, immutable source/dataset verification, safe revisions, bounded recoverable jobs | Documented dependency advisory; Windows unsupported |
| Baseline forecasting | Reproducible whole-field race and season distributions, chronological comparisons, official standings and scoring/countback fixtures | Broad championship calibration and exceptional sporting-rule coverage |
| Competitive models | Both families plus ensemble, disjoint calibration, ablations, frozen prospective procedure and enforced promotion gate | Completed prospective cohort; sparse practice/sprint/weather evidence; current-model regulation transfer |
| Control room | Requested design across five workspaces; desktop/mobile/browser evidence; transparent missing/stale/error states | No claim of exhaustive accessibility or all-browser certification |
| Live/replay | Deterministic replay and correction/reconnect/staleness tests; two real historical archives; free recorder adapter | Actual free-feed capture/reconnect validation; strategy/weather/neutralization modeling; independent live accuracy |

No worldwide superiority, perfection or guaranteed race outcome is claimed. The original implementation remains preserved as reference, and all revised forecasts retain their predecessors.
