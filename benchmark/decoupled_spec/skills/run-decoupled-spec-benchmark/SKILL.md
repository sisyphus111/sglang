---
name: run-decoupled-spec-benchmark
description: Coordinate one complete, traceable SGLang decoupled-spec benchmark run from independent verifier, drafter, client, and observability configs. Use whenever the user asks to run, rerun, test, or experiment with a concrete decoupled-spec case; use narrower skills only when the user explicitly limits the request to one stage.
---

# Run a Decoupled-Spec Benchmark

Produce one self-contained `RUN_DIR` without hiding the verifier, drafter,
client, or collector behind a combined launcher.

## Inputs

Resolve these before launching GPU work:

- verifier, drafter, client, and observability YAML paths
- every named CLI override requested by the user
- output root and a run name containing the meaningful experiment tuple
- expected GPU allocation, HTTP ports, and decoupled transport endpoints

Do not silently change `speculative_num_steps`, `speculative_eagle_topk`,
`speculative_num_draft_tokens`, the Python/C++ data-plane backend, batch size,
prompt/output lengths, tokenizer, chat-template mode, or CUDA Graph settings.

## Workflow

1. Read [references/lifecycle.md](references/lifecycle.md).
2. Use `operate-decoupled-spec-servers` for GPU/process/port preflight, pair
   validation, independent server launch, readiness, and shutdown.
3. Initialize one unique `RUN_DIR`. Every component in this run must receive
   that exact directory.
4. After both roles are HTTP-ready, use `observe-decoupled-spec-run` to start
   the collector and obtain successful baseline samples from both roles.
5. Use `send-decoupled-spec-workload` to inspect and submit exactly one formal
   streaming batch to the verifier.
6. Stop the collector after the formal client window and retain trailing
   samples. Stop only the verifier and drafter processes owned by this run.
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

- Verifier and drafter remain separate HTTP server processes and sessions.
- `batch.size=N` means one `/generate` request containing `N` tokenized inputs.
- The collector calls service-state endpoints, never `/generate`.
- All plots are derived from saved artifacts; plotting does not rerun traffic.
- When fixed decode windows are present, require the observability figure to
  contain scheduler cycle, mean batch size, mean context length, valid draft
  length, and accept length over time.
- No file is written inside a sealed `RUN_DIR`.

## Output Contract

Return the absolute `RUN_DIR`, the effective experiment tuple, component exit
states, the main client metrics, the pre-seal audit result, and checksum status.
Label incomplete or failed runs explicitly; do not report them as benchmark
results.
