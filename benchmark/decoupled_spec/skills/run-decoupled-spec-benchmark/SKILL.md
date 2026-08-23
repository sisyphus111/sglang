---
name: run-decoupled-spec-benchmark
description: Coordinate one complete, traceable SGLang decoupled-spec benchmark run from a unified Ray server-fleet config plus client and observability configs. Use whenever the user asks to run, rerun, test, or experiment with a concrete decoupled-spec case; use narrower skills only when the user explicitly limits the request to one stage.
---

# Run a Decoupled-Spec Benchmark

Produce one self-contained `RUN_DIR`; one Ray launcher owns the complete server
fleet, while client and collector remain independent manifest consumers.

## Inputs

Resolve these before launching GPU work:

- unified server-fleet, client, and observability YAML paths
- every named CLI override requested by the user
- output root and a run name containing the meaningful experiment tuple
- expected replica topology, TP sizes, and total Ray GPU allocation

Do not silently change `speculative_num_steps`, `speculative_eagle_topk`,
`speculative_num_draft_tokens`, the Python/C++ data-plane backend, batch size,
prompt/output lengths, tokenizer, chat-template mode, or CUDA Graph settings.

## Workflow

1. Read [references/lifecycle.md](references/lifecycle.md).
2. Use `operate-decoupled-spec-servers` for Ray/GPU preflight, unified config
   validation, fleet launch, manifest readiness, and shutdown.
3. Initialize one unique `RUN_DIR`. Every component in this run must receive
   that exact directory.
4. After every manifest engine is HTTP-ready, start the collector with
   `--server-manifest` and obtain successful baseline samples from every
   verifier and drafter.
5. Use `send-decoupled-spec-workload` with `--server-manifest` and an explicit
   verifier rank to inspect and submit exactly one formal streaming batch.
6. Stop the collector after the formal client window and retain trailing
   samples. Then stop the unified launcher and wait for it to collect every
   remote engine's config/status/log before continuing.
7. Use `analyze-decoupled-spec-results` to generate the single-run report and
   plots from the saved client artifacts.
8. Use `audit-decoupled-spec-artifacts` in `pre-seal` mode. Seal only after it
   passes, then run its read-only `sealed` verification.

An unqualified request such as "run the experiment", "test this tuple", or
"rerun the benchmark" means this complete lifecycle. Do not stop after server
startup or client completion. A complete run includes collector baseline and
trailing samples, all single-run plots, the Markdown report, pre-seal audit,
seal, and read-only checksum verification.

If any stage fails, preserve the unsealed `RUN_DIR`, stop only owned processes,
report the failing stage and first useful error, and use a new run directory for
the retry.

## Invariants

- Every verifier and drafter remains a separate HTTP engine, jointly owned by
  one Ray launcher and identified by `engine_id` in the server manifest.
- `batch.size=N` means one `/generate` request containing `N` tokenized inputs.
- The collector calls service-state endpoints, never `/generate`.
- The launcher must use the saved sparse quota graph and ranked peer configs;
  do not replace it with a dense full mesh. Verifier SWRR selection is sticky
  for the full request lifecycle.
- Verifier and drafter must both keep `num_waiting_reqs == 0`. Require a
  zero-waiting collector baseline before sending the client batch; any positive
  queue telemetry in either formal-window samples or either role log makes the
  run invalid and it must not be sealed or reported as a benchmark result.
- All plots are derived from saved artifacts; plotting does not rerun traffic.
- Require a “Running batch size over time” plot from sampled
  `num_running_reqs`, with separate verifier and drafter lines and a linear
  y-axis starting at zero.
- When fixed decode windows are present, require iteration latency for both
  roles plus verifier-only valid draft tail length and accept length over time.
  Plot iteration latency on a linear y-axis starting at zero. Any presentation-
  only removal of abnormal large values must be auditable and must not change
  the raw collector windows.
- No file is written inside a sealed `RUN_DIR`.

## Output Contract

Return the absolute `RUN_DIR`, the effective experiment tuple, component exit
states, the main client metrics, the pre-seal audit result, and checksum status.
Label incomplete or failed runs explicitly; do not report them as benchmark
results.
