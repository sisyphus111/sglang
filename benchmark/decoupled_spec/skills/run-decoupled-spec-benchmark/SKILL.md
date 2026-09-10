---
name: run-decoupled-spec-benchmark
description: Coordinate one complete, traceable SGLang decoupled-spec benchmark run from a unified Ray server-fleet config plus client and observability configs. Use whenever the user asks to run, rerun, test, or experiment with a concrete decoupled-spec case; use narrower skills only when the user explicitly limits the request to one stage.
---

# Run a Decoupled-Spec Benchmark

Produce one compact `RUN_DIR` plus one disposable `RUNTIME_DIR`. The Ray
launcher owns the complete server fleet; Client and Observer consume its
temporary manifest but persist only benchmark data.

## Inputs

Resolve these before launching GPU work:

- unified server-fleet, client, and observability YAML paths
- every named CLI override requested by the user
- output root, runtime root, and a run name containing the meaningful experiment tuple
- expected replica topology, TP sizes, and total Ray GPU allocation

Do not silently change `speculative_num_steps`, `speculative_eagle_topk`,
`speculative_num_draft_tokens`, the Python/C++ data-plane backend, batch size,
prompt/output lengths, tokenizer, chat-template mode, or CUDA Graph settings.

## Workflow

1. Read [references/lifecycle.md](references/lifecycle.md).
   For adaptive verifier runs, first use `profile-decoupled-spec-verifier` and
   freeze the complete profile path/SHA plus adaptive config in the experiment
   tuple.
2. Use `operate-decoupled-spec-servers` for Ray/GPU preflight, unified config
   validation, fleet launch, manifest readiness, and platform-owned teardown.
3. Manually create one unique `RUN_DIR` and one disposable `RUNTIME_DIR`.
   Server control files and logs stay under `RUNTIME_DIR`; final data stays
   under `RUN_DIR`.
4. After every manifest engine is HTTP-ready, run `runner.py` with the Client
   and Observer configs, `--server-manifest`, and an explicit verifier rank.
   Runner must obtain a successful zero-waiting baseline round from every
   target before invoking Client, then retain a complete trailing round after
   Client receives the full streaming response.
5. Confirm Runner stopped its Observer subprocess and completed
   `observer/bench_timeline.json`. Then stop the deployment through its owning environment and confirm
   that no owned task process remains; server logs are optional.
6. Use `analyze-decoupled-spec-results` to generate the single-run report and
   plots from the saved client artifacts.

For a remote submitted deployment, split ownership across the task boundary:
the submitted task launches and retains only the server fleet;
`runner.py`, Client, Observer, artifact validation, and analysis run from the
local checkout against the remote HTTP endpoints. The task entrypoint must stay
focused on `server/server.py`; do not inline the benchmark driver or move Client
and Observer into the remote task. Stop the submitted task only after the local
trailing Observer round and all requested local artifacts are complete.

An unqualified request such as "run the experiment", "test this tuple", or
"rerun the benchmark" means this complete lifecycle. Do not stop after server
startup or client completion. A complete run includes observer baseline and
trailing samples, all single-run plots, and the Markdown report.

If any stage fails, preserve the partial `RUN_DIR`, stop only owned processes,
report the failing stage and first useful error, and use a new run directory for
the retry.

## Invariants

- Every verifier and drafter remains a separate HTTP engine, jointly owned by
  one Ray launcher and identified by `engine_id` in the server manifest.
- `batch.size=N` means one `/generate` request containing `N` tokenized inputs.
- The observer calls service-state endpoints, never `/generate`.
- The launcher must use the saved sparse quota graph and ranked peer configs;
  do not replace it with a dense full mesh. Verifier SWRR selection is sticky
  for the full request lifecycle.
- Require a zero-waiting observer baseline before sending the client batch.
  For fixed-BS throughput, require `num_waiting_reqs == 0` on both roles only
  inside the full-BS verifier decode measurement window, before the first
  request exits. Prefill/batch-fill queues and post-window drain queues are
  allowed. Apply the same time boundaries to optional server-log evidence.
  Missing full-BS decode windows or queue samples cannot prove performance validity.
- All plots are derived from saved artifacts; plotting does not rerun traffic.
- A successful client writes exactly
  `client/{requests.csv,batch.json,content.json}`. The schema is fixed by
  the [client artifact contract](../send-decoupled-spec-workload/references/client-artifact-contract.md);
  do not add, rename, reorder, or restore a legacy result file without an
  explicit user request.
- Require a “Running batch size over time” plot from sampled
  `num_running_reqs`, with separate verifier and drafter lines.
- When fixed decode windows are present, require iteration latency for both
  roles plus verifier-only valid draft tail length and accept length over time.
  Every non-negative metric plot must use a linear y-axis starting at zero;
  signed effects must include zero. Logarithmic, broken, and truncated y-axes
  are forbidden. Clearly abnormal presentation outliers must be excluded with
  a deterministic recorded rule (default: `Q3 + 3 * IQR`) without changing raw
  observer windows; record thresholds and every excluded window/time/value.
- Adaptive verifier runs additionally require `adaptive_verify` windows and the
  `adaptive_verify` plot: active K, supply/conditional-accept EMAs, candidate
  modeled TPS, K residency, profile SHA, and switch counters must be present.
- `RUN_DIR` is an ordinary writable directory. There is no centralized seal,
  checksum, provenance manifest, or artifact audit lifecycle.
- `RUN_DIR/config.json` contains exactly the effective `server` and `client`
  sections. Observer owns `observer/samples.jsonl`; Runner owns
  `observer/bench_timeline.json`.
- Server manifests, statuses, resolved-config copies, and logs never enter
  `RUN_DIR`.

## Output Contract

Return the absolute `RUN_DIR`, the effective experiment tuple, component exit
states, and the main client metrics. Label incomplete or failed runs explicitly;
do not report them as benchmark results.
