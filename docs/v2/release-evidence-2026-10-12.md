# F1 Engine V2 — release evidence

Recorded on **12 October 2026**. This is a usable local baseline and research release on `codex/f1-engine-v2`. The complete five-milestone plan has **not** passed every release gate. The active pre-weekend model is `baseline:rating`; the challenger did not earn promotion. Nothing here establishes worldwide superiority or perfect predictions.

## Runnable artifact and source checkpoints

Install and launch using the [V2 guide](README.md). The original implementation remains preserved. V2 has its own Python package, dependency lock, React/TypeScript application and bundled Plotly interface. The five workspaces are Race, Weekend, Championship, Live/replay and Evidence.

Commit `6ba9a08` preserves the exact source used for the frozen development, 2025 holdout and 2026 transfer reports. Its Python source hash is `02aa7e0b6f3cadd0deb21e9b9cdc31129dd1b1ef907fe3801bd32891db81b160`. Later fixes to replay envelopes, championship revisions, source-change detection and source adapters do not retroactively change those reports. Restore that checkpoint to reproduce their execution environment; future modeling revisions cannot reuse 2025 as an untouched holdout.

The current artifact has Python source hash `92da2de4ced93d745bd4cdeca061bc9afface9ae2f9f3a4f8eee529c320c26dc`. The [clean installation record](clean-install.json) includes the rebuilt wheel's hash. A separate environment installed that wheel with locked dependencies and verified startup, the empty dataset, the UI and all three bundled assets through an in-process ASGI client. No private dataset or internet request was needed.

The current execution sandbox blocks local socket binding. Consequently the latest server restart and fresh browser run were not completed here. The five recorded browser tests passed against the preceding runnable checkpoint; frontend files have not changed since that run. Launch `f1engine app` in a normal local terminal after restarting any older preview process.

## Verification results

| Check | Result | Evidence and scope |
|---|---|---|
| Offline regression suite | **66 passed**, five browser tests deselected | [JUnit report](offline-tests.xml); no network or private datasets |
| Browser workspaces | **5 passed** at checkpoint `6ba9a08` | [JUnit report](browser-tests.xml); all five workspaces, reproduction, failed-request retention; [screenshot](control-room.png) is a fictional fixture |
| Source checks | Passed | `ruff check v2 v2_tests` |
| Dependency compatibility | Passed | `pip check`; locked Python environment |
| Isolated installed wheel | Passed | [Installation record](clean-install.json); latest code, in-process startup and asset requests |
| Race and championship reproduction | **Exact prediction-hash matches** | [Reproduction record](reproduction.json); latest code and locked local numerical environment |
| Ledger integrity | Passed | Six saved official forecasts/revisions at the reproduction check; local integrity proof, not external timestamp certification |
| Python dependency audit | Documented exception remains | [Security assessment](SECURITY.md), [audit](dependency-audit-python.json) |
| Frontend dependency audit | Zero reported findings in recorded scan | [Audit](dependency-audit-web.json); dependencies unchanged afterward |

Tests cover startup without artifacts, retained history and lap data, atomic dataset/model publication, team assignments, sprint availability, timezone validation, source conflicts, model-stage selection, cache behavior, coherent field permutations/probabilities, season scoring and countback, replay corrections/duplicates/reconnection/staleness, and job interruption/cancellation/time-budget recovery. The strengthened future-data tests explicitly insert **and modify** future result rows. Malformed future message payloads cannot affect earlier replay states. A source edit while a process is running prevents new evidence from being saved under the wrong source hash.

The test suite emits a Starlette test-client deprecation warning and NumPy/Pandas duration-construction warnings. They did not fail tests. They are not evidence of numerical equivalence across other platforms or dependency versions.

## Performance on this machine

The [measured benchmark](performance.json) uses a cached, fictional 20-driver fixture with 22 remaining rounds, 10,000 samples and three timed repetitions after warm-up. It measures computational performance only.

| Workload | Slowest measured run | Acceptance target | Result |
|---|---:|---:|---|
| 10,000 full-race simulations | **0.1262 seconds** | Under 10 seconds | Passed |
| 10,000 season trajectories | **3.4789 seconds** | Under 30 seconds | Passed |

Measurements were made on the local macOS ARM64 machine. They include feature building and forecast summaries. Network acquisition, cold-cache installation and larger scenario workloads are outside this timing claim.

## Historical reconstruction and development

Public result history was acquired from 2014 onward. Model selection used chronological development folds ending in **2022, 2023 and 2024**, totaling **68 evaluated races**. Twelve candidates were compared: three regularized linear strength models and nine XGBoost ranking configurations. The selected challenger was XGBoost with depth 2 and learning rate 0.08. Calibration used earlier, disjoint chronological data; the final frozen candidate trained through 2023 and calibrated on 2024.

| Development metric | Rating baseline | Selected challenger |
|---|---:|---:|
| Race-averaged normalized RPS; lower is better | 0.12482348 | 0.12476643 |
| Winner log loss; lower is better | 1.75603510 | 1.70977747 |

The paired RPS difference was **−0.00005705**, with race-bootstrap 95% interval **[−0.00140429, +0.00132839]**. Because the interval crosses zero, the development promotion gate failed. See the [full candidate comparison and ablations](development-comparison.json) and [candidate freeze](candidate-freeze.json). A circuit-group ablation was run. Telemetry and extra weekend-feature benefits remain unverified for this pre-weekend evaluation; missing stage inputs cannot prove that those features are useless. No ensemble has been trained or promoted.

Historical publication times are reconstructed, with that limitation retained in provenance. Weather breakdowns are marked unknown where forecast/climate inputs are absent. They are not represented as dry races or retrospectively substituted weather forecasts.

## Frozen 2025 held-out evaluation

The candidate and selection procedure were frozen before opening 2025. **23 races** met full-field inclusion checks. The [holdout report](holdout-2025.json) contains race-level rows, secondary metrics, calibration curves and season/circuit/weather breakdowns.

| 2025 holdout metric | Rating baseline | Frozen challenger |
|---|---:|---:|
| Normalized RPS | 0.13094698 | 0.13051873 |
| Winner log loss | 1.87812314 | 1.95256113 |
| Podium Brier score | 0.07911281 | 0.08012734 |
| Points Brier score | 0.18069618 | 0.17998512 |
| Retirement Brier score | 0.10186786 | 0.09977657 |
| Mean absolute finishing-position error | 3.6174 | 3.6467 |
| Predictive p10–p90 interval coverage | 0.8739 | 0.8761 |

The paired RPS difference was **−0.00042824**, with 95% interval **[−0.00240252, +0.00139777]**, using 5,000 paired race bootstrap samples. Winner log loss worsened by about **3.96%**, exceeding the permitted 2%. **Both promotion requirements failed. The rating baseline remains active.** These are comparisons among the named candidates on this evaluation only.

Spain (`2025-09`) was excluded because Stroll appeared in the entrant records without a matching official race-result row. No target outcome was invented. The [exclusion record](evaluation-exclusions.json) identifies the missing identity. Failed optional source requests left the 2025 sprint and standings records unavailable; complete retained calendar, GP-result and qualifying pages were recovered through the same source normalizer. See [coverage](holdout-source-coverage.json). This pre-weekend race evaluation does not use those missing optional tables.

2025 is now inspected evidence. Any further candidate revision informed by these results must treat it as development data. The saved report remains a record of that first frozen evaluation; it is not permission to repeatedly advertise new 2025 results as untouched holdouts.

## Separate 2026 regulation-transfer evaluation

The frozen challenger was also evaluated on **17 completed 2026 races**. Its normalized RPS was **0.12995895** and winner log loss **2.21829141**. The [transfer report](transfer-2026.json) provides all secondary metrics and breakdowns. This report is separate from the development and 2025 holdout results, and is not a prospective accuracy record or a baseline-promotion claim.

## Prospective collection and correction history

The local append-only ledger contains a 22-driver pre-weekend forecast for the 2026 United States GP and a championship projection with six remaining events. Current linked revisions are exported as [race forecast](current-prospective-forecast.json) and [championship forecast](current-championship-forecast.json). Both use 10,000 samples and reproduce exactly with the current source. The events are unresolved at collection, so **no prospective accuracy result is claimed**.

During development, the source status `Lapped` was discovered to have been incorrectly interpreted as retirement. It was corrected before the holdout was opened; disqualifications were separated from retirement, and nullable standings ranks were preserved. The [data-correction record](data-correction.json) links the replaced dataset and superseded experiment. Earlier development artifacts remain under `superseded-*` and must not be presented as current results. Earlier forecasts remain in the ledger, linked to their revisions rather than deleted.

## Milestone acceptance and remaining work

| Milestone | Runnable capability and proof | Still unverified or incomplete |
|---|---|---|
| Foundation | Installed startup, offline fixtures, atomic updates, append-only ledger, bounded recoverable jobs | Latest socket-based relaunch blocked by the current execution environment; Windows unsupported |
| Baseline forecasting | Coherent full-field race/season forecasts; real chronological comparison; official standings; exact reproduction and season-rule fixtures | Championship probability calibration and broad exceptional sporting-rule validation; historical standings must be acquired for each requested cutoff |
| Competitive models | Both families, chronology-safe calibration, 12-candidate comparison, freeze/holdout gate and circuit ablation | Challenger gate failed; no ensemble; practice/telemetry/weather value and weekend-stage calibration remain unverified |
| Control room | Five workspaces, source inspection, scenarios, uncertainty labels, durable jobs and SSE; five browser checks passed | Latest backend revision needs a fresh socket/browser run once execution permissions are available |
| Live/replay | Deterministic arrival-cutoff replay, duplicates, corrections, stale-feed rejection, reconnect fixtures and optional free recorder | Recorded real-race predictive evaluation and free-feed integration remain unverified; pit strategy, neutralization and weather transitions are not calibrated |

Other limits remain visible in the app: qualifying rank is a grid proxy until penalties are verified; unknown fuel load is uncertainty; provisional entrants are not confirmed transfers; season-persistent team uncertainty and disqualification priors are not empirically calibrated. Exact unresolved countback ties use a disclosed random adjudication. Individual model uncertainty is not estimated; outcome quantiles, Monte Carlo sampling error and evaluation uncertainty are distinguished. Source-capture timestamps are local evidence, not proof of original publication time.

The dependency scan retains one distinct msgpack advisory reported twice, because the compatible upstream recorder dependency pins that version. The affected streaming-unpacker reuse pattern is absent from inspected application paths. The [documented exception](SECURITY.md) remains a release limitation, not a zero-vulnerability claim.

An attempted bounded download of one historical practice session was **not executed**: automatic approval review failed because the workspace was out of credits. The reviewer explicitly said this was a review failure, not a determination that the action was unsafe. A workspace-owner refill is required before retrying the real-data integration check. Unaffected local implementation, regression checks, wheel verification and reproduction were completed. OpenF1's [published access policy](https://openf1.org/) also keeps active-session access outside the required free dependency path.
