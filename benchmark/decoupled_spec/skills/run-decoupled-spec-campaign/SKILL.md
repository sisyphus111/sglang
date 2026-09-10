---
name: run-decoupled-spec-campaign
description: Materialize, resume, and summarize multiple decoupled-spec benchmark cases from local experiment input. Use when a request spans multiple models, modes, batch sizes, output lengths, or repeated cases; use run-decoupled-spec-benchmark for one concrete case.
---

# Run a Decoupled-Spec Campaign

Coordinate multiple complete benchmark runs without hiding the atomic server,
observer, client, or plotting stages.

## Inputs

Read [references/campaign-schema.md](references/campaign-schema.md) before
materializing a new campaign. Resolve:

- one unified server-fleet YAML per schedule mode;
- one client YAML and one observer YAML;
- the explicit experiment axes and ordering;
- a unique campaign output directory;
- stop gates, prerequisites, and the maximum deployment concurrency authorized
  by the user or execution environment.

Experiment YAML is local run input and remains untracked. Materialization saves
its exact content and SHA-256 in `campaign_manifest.json` before any server is
started.

## Workflow

1. Run `scripts/campaign.py validate --config <experiment.yaml>`.
2. Run `scripts/campaign.py materialize` into a fresh campaign directory. Never
   overwrite or repurpose an existing contract.
3. Inspect `status` and `show-case`; satisfy every recorded prerequisite before
   registering an attempt.
4. For each eligible case, create a unique `RUN_DIR` and disposable
   `RUNTIME_DIR`, then use
   `run-decoupled-spec-benchmark` for the complete server -> observer baseline ->
   client batch -> trailing samples -> plots/report lifecycle.
5. Advance the ledger one state at a time. On failure, record the first failed
   stage as `incomplete`, retain the partial directory, and use a new `RUN_DIR`
   for any retry.
6. Run `scripts/summarize.py` after the requested cases are marked `completed`.
   It reads the fixed client result files and writes campaign tables, plots,
   reports, and source hashes. Plotters may also retain SVG, but the user-facing campaign handoff
   must display or link clear high-resolution PNG figures by default unless the
   user explicitly requests vector output.

For campaigns backed by submitted tasks, each task is a server-only deployment.
Keep campaign orchestration, `runner.py`, Client,
Observer, validation, and result aggregation local, connecting to the remote
HTTP endpoints advertised by `DSPEC_SERVER_MANIFEST`. Do not upload or inline a
campaign loop into the task entrypoint. A concurrency limit counts live remote
server tasks; the local coordinator must never leave more deployments live than
that limit.

## Invariants

- A case uses one unified server config containing both verifier and drafter.
- A campaign never launches legacy independent verifier/drafter server scripts.
- Do not exceed the explicit deployment concurrency. Shared requests inside one
  SGLang batch are not independent experiment repetitions.
- Only attempts marked `completed` enter the aggregate result. The campaign
  does not seal or centrally audit their directories.
- Every case uses the fixed three-file Client contract directly under
  `RUN_DIR/client/`; campaign code
  derives cross-run metrics from those files and must not consume a legacy
  client artifact. Read the
  [client artifact contract](../send-decoupled-spec-workload/references/client-artifact-contract.md)
  before materializing or summarizing cases.
- User-facing campaign figures use ordinary linear y-axes. Non-negative metrics
  start at zero; signed effects include a visible zero baseline. Do not use
  logarithmic, broken, or truncated y-axes. Exclude clearly abnormal
  presentation outliers with a deterministic recorded rule (default:
  `Q3 + 3 * IQR`) while preserving raw inputs and recording each exclusion in
  the plot manifest.
- Do not delete failed attempts, silently replace configuration, fill missing
  measurements with zero, or report partial campaigns as complete.
- Preserve per-engine verifier/drafter telemetry and the exact local input
  specification in the campaign artifacts.
- For any cross-run statistic derived from formal `decode_metrics_windows`,
  first apply the boundary rule in
  `../observe-decoupled-spec-run/references/timing-boundaries.md`: exclude the
  earliest in-boundary window per `(target_id, dp_rank)`, record it, and only
  then weight or filter retained windows. Client-owned request metrics do not
  use this exclusion.
